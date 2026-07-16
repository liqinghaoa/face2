"""Unit tests for Stage 2A label-free optical condition calibration."""

from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd
import pytest

import scripts.preprocess.run_optical_condition_calibration_stage2a as stage2a
from scripts.preprocess.run_optical_condition_calibration_stage2a import (
    CalibrationFailure,
    INPUT_COLUMNS,
    audit_splits,
    build_feature_schema,
    coefficient_stability_interpretation,
    prepare_outputs,
    transform_feature_frame,
    write_failure_log_safely,
)
from utils.optical_condition_calibration import (
    ALL_TARGETS,
    CHEEK_TARGETS,
    DESIGN_FEATURE_NAMES,
    EXPECTED_CAMERAS,
    FOREHEAD_CHEEK_TARGETS,
    RidgeModel,
    build_design_matrix,
    calibrate_values,
    fit_condition_scaler,
    fit_ridge,
    spearman_rho,
    transform_conditions,
    validate_camera_values,
)


def condition_frame() -> pd.DataFrame:
    return pd.DataFrame({
        "ID": ["a", "b", "c", "d"],
        "camera_id": [EXPECTED_CAMERAS[0], EXPECTED_CAMERAS[0], EXPECTED_CAMERAS[1], EXPECTED_CAMERAS[1]],
        "relative_optical_exposure": [1.0, 3.0, 10.0, 14.0],
        "log2_iso_condition": [2.0, 4.0, 20.0, 26.0],
    })


def test_camera_specific_training_scaler_uses_population_std() -> None:
    scaler = fit_condition_scaler(condition_frame())
    honor = scaler["camera_parameters"][EXPECTED_CAMERAS[0]]["conditions"]
    xiaomi = scaler["camera_parameters"][EXPECTED_CAMERAS[1]]["conditions"]
    assert honor["relative_optical_exposure"]["mean"] == 2.0
    assert honor["relative_optical_exposure"]["population_std"] == 1.0
    assert xiaomi["relative_optical_exposure"]["mean"] == 12.0
    assert xiaomi["relative_optical_exposure"]["population_std"] == 2.0
    assert honor["log2_iso_condition"]["population_std"] == 1.0
    assert xiaomi["log2_iso_condition"]["population_std"] == 3.0


def test_validation_transform_reuses_training_parameters() -> None:
    scaler = fit_condition_scaler(condition_frame())
    val = pd.DataFrame({
        "camera_id": [EXPECTED_CAMERAS[0], EXPECTED_CAMERAS[1]],
        "relative_optical_exposure": [4.0, 16.0],
        "log2_iso_condition": [5.0, 29.0],
    })
    transformed = transform_conditions(val, scaler)
    assert transformed["z_relative_optical_exposure"].tolist() == pytest.approx([2.0, 2.0])
    assert transformed["z_log2_iso_condition"].tolist() == pytest.approx([2.0, 2.0])


def test_degenerate_training_std_outputs_zero_and_is_recorded() -> None:
    frame = condition_frame()
    frame.loc[frame["camera_id"] == EXPECTED_CAMERAS[0], "relative_optical_exposure"] = 7.0
    scaler = fit_condition_scaler(frame, std_epsilon=1e-8)
    record = scaler["camera_parameters"][EXPECTED_CAMERAS[0]]["conditions"]["relative_optical_exposure"]
    assert record["degenerate_std"] is True
    transformed = transform_conditions(frame, scaler)
    assert transformed.loc[
        transformed["camera_id"] == EXPECTED_CAMERAS[0], "z_relative_optical_exposure"
    ].eq(0).all()


def test_camera_encoding_and_unknown_or_missing_rejection() -> None:
    frame = transform_conditions(condition_frame(), fit_condition_scaler(condition_frame()))
    matrix = build_design_matrix(frame)
    assert matrix[:, 0].tolist() == [0.0, 0.0, 1.0, 1.0]
    with pytest.raises(ValueError, match="Unknown camera"):
        validate_camera_values([EXPECTED_CAMERAS[0], "Other/Camera"])
    with pytest.raises(ValueError, match="missing or empty"):
        validate_camera_values([EXPECTED_CAMERAS[0], None])


def test_design_matrix_order_and_interactions() -> None:
    frame = pd.DataFrame({
        "camera_id": [EXPECTED_CAMERAS[0], EXPECTED_CAMERAS[1]],
        "z_relative_optical_exposure": [2.0, 3.0],
        "z_log2_iso_condition": [-1.0, 4.0],
    })
    matrix = build_design_matrix(frame)
    assert DESIGN_FEATURE_NAMES == (
        "camera_xiaomi", "z_relative_optical_exposure", "z_log2_iso_condition",
        "camera_xiaomi_x_z_exposure", "camera_xiaomi_x_z_iso",
    )
    assert matrix.tolist() == [[0, 2, -1, 0, 0], [1, 3, 4, 3, 4]]


def test_ridge_unpenalized_intercept_alpha_one_and_multioutput_shape() -> None:
    x = np.zeros((4, 5), dtype=float)
    x[:, 0] = [0, 0, 1, 1]
    y = np.column_stack(([1, 1, 3, 3], [2, 2, 6, 6]))
    model = fit_ridge(x, y, ("target_a", "target_b"), alpha=1.0)
    assert model.intercept == pytest.approx([1.5, 3.0])
    assert model.coefficients[0] == pytest.approx([1.0, 2.0])
    assert model.coefficients[1:] == pytest.approx(np.zeros((4, 2)))
    assert model.predict(x).shape == (4, 2)
    with pytest.raises(ValueError, match="fixes alpha"):
        fit_ridge(x, y, ("target_a", "target_b"), alpha=0.5)


def test_json_roundtrip_recovers_identical_predictions() -> None:
    rng = np.random.default_rng(9)
    x = rng.normal(size=(20, 5))
    y = rng.normal(size=(20, 3))
    model = fit_ridge(x, y, ("a", "b", "c"))
    restored = RidgeModel.from_dict(model.to_dict())
    assert np.array_equal(model.predict(x), restored.predict(x))


def test_synthetic_acquisition_trend_is_reduced_in_residual() -> None:
    rng = np.random.default_rng(11)
    n = 300
    camera = np.where(np.arange(n) % 2 == 0, EXPECTED_CAMERAS[0], EXPECTED_CAMERAS[1])
    z_exp = rng.normal(size=n)
    z_iso = rng.normal(size=n)
    d = (camera == EXPECTED_CAMERAS[1]).astype(float)
    frame = pd.DataFrame({
        "camera_id": camera,
        "z_relative_optical_exposure": z_exp,
        "z_log2_iso_condition": z_iso,
    })
    x = build_design_matrix(frame)
    raw = 2.0 + 1.8 * d + 2.5 * z_exp - 1.2 * z_iso + 0.1 * rng.normal(size=n)
    model = fit_ridge(x, raw, ("synthetic",))
    residual = raw - model.predict(x)[:, 0]
    _, raw_rho = spearman_rho(raw, z_exp)
    _, residual_rho = spearman_rho(residual, z_exp)
    assert abs(residual_rho) < abs(raw_rho) * 0.2


def test_calibration_formula_uses_training_reference_mean() -> None:
    raw = np.array([5.0, 8.0])
    predicted = np.array([2.0, 3.0])
    residual, calibrated = calibrate_values(raw, predicted, np.array(10.0))
    assert residual.tolist() == [3.0, 5.0]
    assert calibrated.tolist() == [13.0, 15.0]


def complete_feature_frame() -> pd.DataFrame:
    rows = []
    for index in range(8):
        row = {
            "ID": str(index),
            "camera_id": EXPECTED_CAMERAS[index % 2],
            "forehead_available": 0 if index == 7 else 1,
            "relative_optical_exposure": float(index),
            "log2_iso_condition": float(index % 3),
        }
        for offset, target in enumerate(CHEEK_TARGETS):
            row[target] = 1.0 + offset + index * 0.1
        for offset, target in enumerate(FOREHEAD_CHEEK_TARGETS):
            row[target] = np.nan if index == 7 else 0.5 + offset + index * 0.05
        rows.append(row)
    return pd.DataFrame(rows)


def test_unavailable_forehead_is_not_fit_or_imputed_and_case_is_retained() -> None:
    train = complete_feature_frame()
    scaler = fit_condition_scaler(train)
    scaled = transform_conditions(train, scaler)
    x = build_design_matrix(scaled)
    cheek_y = scaled[list(CHEEK_TARGETS)].to_numpy(float)
    available = scaled["forehead_available"].eq(1).to_numpy()
    forehead_y = scaled.loc[available, list(FOREHEAD_CHEEK_TARGETS)].to_numpy(float)
    cheek_model = fit_ridge(x, cheek_y, CHEEK_TARGETS)
    forehead_model = fit_ridge(x[available], forehead_y, FOREHEAD_CHEEK_TARGETS)
    output = transform_feature_frame(
        train, 0, "train", scaler, cheek_model, forehead_model,
        cheek_y.mean(axis=0), forehead_y.mean(axis=0),
    )
    unavailable = output.loc[output["ID"] == "7"].iloc[0]
    assert len(output) == len(train)
    for target in FOREHEAD_CHEEK_TARGETS:
        for representation in ("raw", "predicted_acquisition", "residual", "calibrated"):
            assert pd.isna(unavailable[f"{representation}_{target}"])
    assert all(np.isfinite(unavailable[f"calibrated_{target}"]) for target in CHEEK_TARGETS)


def test_field_whitelist_and_schema_exclude_clinical_and_condition_features() -> None:
    lowered = " ".join(INPUT_COLUMNS).casefold()
    for forbidden in ("nyha", "label_3class", "sex", "bnp", "brightnessvalue"):
        assert forbidden not in lowered
    schema = build_feature_schema()
    calibrated = schema["calibrated_optical_feature_columns"]
    assert calibrated == [f"calibrated_{target}" for target in ALL_TARGETS]
    assert "camera_id" not in calibrated
    assert "relative_optical_exposure" not in calibrated
    assert "camera_id" in schema["forbidden_direct_nyha_classifier_columns"]


def test_identical_inputs_produce_identical_coefficients() -> None:
    rng = np.random.default_rng(123)
    x = rng.normal(size=(50, 5))
    y = rng.normal(size=(50, 3))
    first = fit_ridge(x, y, ("a", "b", "c"))
    second = fit_ridge(x.copy(), y.copy(), ("a", "b", "c"))
    assert np.array_equal(first.intercept, second.intercept)
    assert np.array_equal(first.coefficients, second.coefficients)


def test_prepare_outputs_handles_one_sided_owned_directory(tmp_path) -> None:
    experiment = tmp_path / "experiment"
    report = tmp_path / "report"
    experiment.mkdir()
    (experiment / ".stage2a_owner.json").write_text(
        json.dumps({"task": stage2a.TASK}), encoding="utf-8"
    )
    (experiment / "old.csv").write_text("x\n1\n", encoding="utf-8")
    args = argparse.Namespace(
        experiment_output_dir=experiment,
        report_output_dir=report,
        overwrite=True,
        summarize_only=False,
    )
    prepare_outputs(args)
    assert experiment.is_dir() and report.is_dir()
    assert not (experiment / "old.csv").exists()
    assert json.loads((experiment / ".stage2a_owner.json").read_text(encoding="utf-8"))["task"] == stage2a.TASK
    assert json.loads((report / ".stage2a_owner.json").read_text(encoding="utf-8"))["task"] == stage2a.TASK


def test_failure_log_does_not_write_into_unowned_report_directory(tmp_path) -> None:
    report = tmp_path / "report"
    report.mkdir()
    unrelated = report / "unrelated.txt"
    unrelated.write_text("keep", encoding="utf-8")
    args = argparse.Namespace(report_output_dir=report)
    result = write_failure_log_safely(args, CalibrationFailure("output_safety", ["stop"]))
    assert result is None
    assert unrelated.read_text(encoding="utf-8") == "keep"
    assert not (report / "run_failure.log").exists()
    assert not (report / ".stage2a_owner.json").exists()


def test_split_audit_requests_only_id_and_fold_columns(tmp_path, monkeypatch) -> None:
    ids = [f"case_{index:03d}" for index in range(500)]
    first_stage = pd.DataFrame({
        "ID": ids,
        "camera_id": [EXPECTED_CAMERAS[index % 2] for index in range(500)],
        "forehead_available": np.ones(500, dtype=int),
    })
    split_dir = tmp_path / "splits"
    split_dir.mkdir()
    split_paths = set()
    for fold in range(5):
        assignments = np.arange(500) % 5
        for role, mask in (("train", assignments != fold), ("val", assignments == fold)):
            path = split_dir / f"fold_{fold}_{role}.csv"
            pd.DataFrame({
                "ID": np.asarray(ids)[mask], "fold": assignments[mask],
                "NYHA": 4, "label_3class": 2, "SEX": "hidden",
            }).to_csv(path, index=False)
            split_paths.add(path.resolve())
    original_read_csv = stage2a.pd.read_csv

    def checked_read_csv(path, *args, **kwargs):
        if stage2a.Path(path).resolve() in split_paths:
            assert kwargs.get("usecols") == ["ID", "fold"]
            assert "nrows" not in kwargs
        return original_read_csv(path, *args, **kwargs)

    monkeypatch.setattr(stage2a.pd, "read_csv", checked_read_csv)
    args = argparse.Namespace(
        split_dir=split_dir,
        train_csv_pattern="fold_{fold}_train.csv",
        val_csv_pattern="fold_{fold}_val.csv",
    )
    result = audit_splits(args, first_stage)
    assert result["status"] == "PASS"
    assert len(result["audit"]) == 10


def test_stability_interpretation_reports_each_unstable_dimension() -> None:
    stability = pd.DataFrame([
        {"target": "a", "coefficient_name": "honor_exposure_slope", "fold_valid_n": 5,
         "sign_consistent_fold_n": 3, "mean": 0.1},
        {"target": "a", "coefficient_name": "xiaomi_exposure_slope", "fold_valid_n": 5,
         "sign_consistent_fold_n": 5, "mean": -0.2},
        {"target": "a", "coefficient_name": "honor_iso_slope", "fold_valid_n": 5,
         "sign_consistent_fold_n": 5, "mean": 0.3},
        {"target": "a", "coefficient_name": "xiaomi_iso_slope", "fold_valid_n": 5,
         "sign_consistent_fold_n": 4, "mean": -0.1},
        {"target": "a", "coefficient_name": "device_intercept_difference", "fold_valid_n": 5,
         "sign_consistent_fold_n": 5, "mean": 0.4},
    ])
    result = coefficient_stability_interpretation(stability)
    assert result["exposure_slopes"]["full_direction_consistency_n"] == 1
    assert result["iso_slopes"]["full_direction_consistency_n"] == 1
    assert result["camera_intercept_differences"]["full_direction_consistency_n"] == 1
    assert result["exposure_slopes"]["unstable_dimensions"] == [{
        "target": "a", "coefficient_name": "honor_exposure_slope",
        "sign_consistent_fold_n": 3, "fold_valid_n": 5,
    }]
