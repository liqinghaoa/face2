"""Unit tests for Stage 2B scaling, splitting, training and calibration."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import torch

from utils.optical_condition_calibration import (
    CHEEK_TARGETS,
    DESIGN_FEATURE_NAMES,
    EXPECTED_CAMERAS,
    FOREHEAD_CHEEK_TARGETS,
    build_design_matrix,
    fit_condition_scaler,
    transform_conditions,
    validate_camera_values,
)
from utils.optical_condition_calibration_nn import (
    deterministic_inner_split,
    fit_target_scaler,
    is_improvement,
    predict_original_scale,
    restore_target_scale,
    should_early_stop,
    standardize_targets,
    train_epoch_selection,
    train_final_model,
    transform_feature_frame,
)


def synthetic_frame(n: int = 80) -> pd.DataFrame:
    rows = []
    for index in range(n):
        camera = EXPECTED_CAMERAS[index % 2]
        exposure = -8.0 + 0.04 * index + (0.3 if camera == EXPECTED_CAMERAS[1] else 0)
        iso = 0.5 + 0.03 * (index % 17) + (0.2 if camera == EXPECTED_CAMERAS[1] else 0)
        row = {
            "ID": f"case_{index:03d}", "camera_id": camera,
            "relative_optical_exposure": exposure, "log2_iso_condition": iso,
            "forehead_available": 0 if index in (7, 35) else 1,
        }
        d = float(camera == EXPECTED_CAMERAS[1])
        for offset, target in enumerate(CHEEK_TARGETS):
            row[target] = 1.0 + offset + 0.2 * exposure - 0.1 * iso + 0.3 * d + 0.01 * np.sin(index)
        for offset, target in enumerate(FOREHEAD_CHEEK_TARGETS):
            row[target] = (
                np.nan if row["forehead_available"] == 0
                else 0.2 + offset * 0.3 - 0.05 * exposure + 0.08 * iso - 0.1 * d + 0.01 * np.cos(index)
            )
        rows.append(row)
    return pd.DataFrame(rows)


def short_config() -> dict:
    return {
        "torch_threads": 1, "learning_rate": 1e-2, "weight_decay": 1e-4,
        "max_epochs": 20, "minimum_epochs": 5, "early_stopping_patience": 5,
        "minimum_improvement": 1e-6, "gradient_clip_max_norm": 5.0,
    }


def test_condition_input_order_camera_encoding_and_unknown_rejection() -> None:
    frame = synthetic_frame(20)
    scaler = fit_condition_scaler(frame)
    transformed = transform_conditions(frame, scaler)
    matrix = build_design_matrix(transformed)
    assert DESIGN_FEATURE_NAMES == (
        "camera_xiaomi", "z_relative_optical_exposure", "z_log2_iso_condition",
        "camera_xiaomi_x_z_exposure", "camera_xiaomi_x_z_iso",
    )
    assert matrix.shape == (20, 5)
    assert matrix[:, 0].tolist() == [0.0, 1.0] * 10
    assert np.allclose(matrix[:, 3], matrix[:, 0] * matrix[:, 1])
    assert np.allclose(matrix[:, 4], matrix[:, 0] * matrix[:, 2])
    with pytest.raises(ValueError, match="Unknown camera"):
        validate_camera_values([EXPECTED_CAMERAS[0], "Other/Device"])


def test_stage2a_condition_scaler_is_reused_exactly() -> None:
    frame = synthetic_frame(40)
    scaler = fit_condition_scaler(frame)
    first = transform_conditions(frame, scaler)
    second = transform_conditions(frame.copy(), scaler)
    assert np.array_equal(
        first[["z_relative_optical_exposure", "z_log2_iso_condition"]].to_numpy(),
        second[["z_relative_optical_exposure", "z_log2_iso_condition"]].to_numpy(),
    )


def test_target_population_scaler_training_only_roundtrip_and_equal_weight() -> None:
    values = np.array([[1.0, 10.0, 100.0], [3.0, 14.0, 108.0], [5.0, 18.0, 116.0]])
    scaler = fit_target_scaler(values, ("a", "b", "c"))
    assert scaler["mean"] == pytest.approx([3.0, 14.0, 108.0])
    assert scaler["population_std"] == pytest.approx(np.std(values, axis=0, ddof=0))
    z = standardize_targets(values, scaler)
    assert restore_target_scale(z, scaler) == pytest.approx(values)
    tensor = torch.tensor(z, dtype=torch.float32)
    mean_all = torch.mean(tensor**2)
    mean_dimensions = torch.mean(torch.mean(tensor**2, dim=0))
    assert torch.equal(mean_all, mean_dimensions)
    shifted_val = values + 1000
    _ = standardize_targets(shifted_val, scaler)
    assert scaler["mean"] == pytest.approx([3.0, 14.0, 108.0])
    with pytest.raises(ValueError, match="population std below"):
        fit_target_scaler(np.ones((5, 3)), ("a", "b", "c"))


def test_inner_split_is_camera_stratified_deterministic_and_disjoint() -> None:
    frame = synthetic_frame(80)
    train_a, val_a, manifest_a = deterministic_inner_split(frame, 2026)
    train_b, val_b, manifest_b = deterministic_inner_split(frame.sample(frac=1, random_state=4), 2026)
    assert train_a["ID"].tolist() == train_b["ID"].tolist()
    assert val_a["ID"].tolist() == val_b["ID"].tolist()
    assert manifest_a == manifest_b
    assert not set(train_a["ID"]) & set(val_a["ID"])
    assert set(train_a["ID"]) | set(val_a["ID"]) == set(frame["ID"])
    assert set(train_a["camera_id"]) == set(val_a["camera_id"]) == set(EXPECTED_CAMERAS)
    assert manifest_a["stratification_fields"] == ["camera_id"]


def test_forehead_inner_split_excludes_unavailable_cases() -> None:
    frame = synthetic_frame(80)
    eligible = frame.loc[frame["forehead_available"].eq(1)].copy()
    train, val, _ = deterministic_inner_split(eligible, 2026)
    assert train["forehead_available"].eq(1).all()
    assert val["forehead_available"].eq(1).all()
    assert "case_007" not in set(train["ID"]) | set(val["ID"])


def test_early_stopping_minimum_patience_improvement_and_tie_rules() -> None:
    assert is_improvement(float("inf"), 1.0, 1e-6)
    assert is_improvement(1.0, 0.999999, 1e-6)
    assert not is_improvement(1.0, 0.9999995, 1e-6)
    assert not should_early_stop(49, 100, 50, 50)
    assert not should_early_stop(50, 49, 50, 50)
    assert should_early_stop(50, 50, 50, 50)
    assert should_early_stop(500, 0, 500, 0)


def test_selection_and_final_refit_are_deterministic_and_fresh(tmp_path) -> None:
    frame = synthetic_frame(80)
    inner_train, inner_val, _ = deterministic_inner_split(frame, 2026)
    context = {"inner_split_sha256": "abc", "config_sha256": "def"}
    first = train_epoch_selection(
        inner_train, inner_val, CHEEK_TARGETS, 0, "cheek", 2027,
        tmp_path / "first", short_config(), context,
    )
    second = train_epoch_selection(
        inner_train, inner_val, CHEEK_TARGETS, 0, "cheek", 2027,
        tmp_path / "second", short_config(), context,
    )
    assert first["selected"]["selected_epoch"] == second["selected"]["selected_epoch"]
    state_a = torch.load(first["checkpoint_path"], map_location="cpu", weights_only=False)["model_state_dict"]
    state_b = torch.load(second["checkpoint_path"], map_location="cpu", weights_only=False)["model_state_dict"]
    assert all(torch.equal(state_a[key], state_b[key]) for key in state_a)
    scaler = fit_condition_scaler(frame)
    final = train_final_model(
        frame, CHEEK_TARGETS, scaler, 0, "cheek", first["selected"]["selected_epoch"], 2027,
        tmp_path / "final.pth", tmp_path / "target.json", tmp_path / "manifest.json",
        short_config(), {
            "stage2a_condition_scaler_path": "stage2a/fold_0/condition_scaler.json",
            "stage2a_condition_scaler_sha256": "1", "split_sha256": "2",
            "config_sha256": "3", "first_stage_sha256": "4",
            "stage2a_manifest_sha256": "5", "git_commit": "unavailable",
            "checkpoint_relative_path": "fold_0/final.pth",
            "target_scaler_relative_path": "fold_0/target.json",
        },
    )
    assert final["payload"]["fresh_initialization_after_epoch_selection"] is True
    assert final["payload"]["validation_loader_used"] is False
    assert final["payload"]["trained_epoch_count"] == first["selected"]["selected_epoch"]
    final_state = final["model"].state_dict()
    assert any(not torch.equal(state_a[key], final_state[key]) for key in state_a)


def test_calibration_formula_and_unavailable_forehead_nan_behavior(tmp_path) -> None:
    frame = synthetic_frame(40)
    scaler = fit_condition_scaler(frame)
    cheek_scaler = fit_target_scaler(frame[list(CHEEK_TARGETS)], CHEEK_TARGETS)
    available = frame["forehead_available"].eq(1)
    forehead_scaler = fit_target_scaler(
        frame.loc[available, list(FOREHEAD_CHEEK_TARGETS)], FOREHEAD_CHEEK_TARGETS
    )
    from models.exif_conditioned_response_mlp import EXIFConditionedResponseMLP
    cheek_model = EXIFConditionedResponseMLP()
    forehead_model = EXIFConditionedResponseMLP()
    for model in (cheek_model, forehead_model):
        for parameter in model.parameters():
            torch.nn.init.zeros_(parameter)
    output = transform_feature_frame(
        frame, 0, "val", scaler, cheek_model, forehead_model, cheek_scaler, forehead_scaler
    )
    target = CHEEK_TARGETS[0]
    assert np.allclose(
        output[f"residual_nn_{target}"],
        output[f"raw_{target}"] - output[f"predicted_condition_nn_{target}"],
    )
    assert np.allclose(
        output[f"calibrated_nn_{target}"],
        output[f"residual_nn_{target}"] + cheek_scaler["mean"][0],
    )
    missing = output.loc[output["forehead_available"].eq(0)]
    for target in FOREHEAD_CHEEK_TARGETS:
        for representation in ("raw", "predicted_condition_nn", "residual_nn", "calibrated_nn"):
            assert missing[f"{representation}_{target}"].isna().all()
    assert np.isfinite(missing[[f"calibrated_nn_{target}" for target in CHEEK_TARGETS]].to_numpy()).all()
    assert len(output) == len(frame)


def test_prediction_restoration_uses_training_target_scaler() -> None:
    frame = synthetic_frame(20)
    condition_scaler = fit_condition_scaler(frame)
    target_scaler = fit_target_scaler(frame[list(CHEEK_TARGETS)], CHEEK_TARGETS)
    from models.exif_conditioned_response_mlp import EXIFConditionedResponseMLP
    model = EXIFConditionedResponseMLP()
    for parameter in model.parameters():
        torch.nn.init.zeros_(parameter)
    predicted = predict_original_scale(model, frame, condition_scaler, target_scaler)
    assert predicted == pytest.approx(np.tile(target_scaler["mean"], (len(frame), 1)))
