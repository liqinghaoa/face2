"""S1-6 Validation-only representation selection and immutable freeze."""

from __future__ import annotations

import json
import platform
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping

import numpy as np
import pandas as pd
import scipy
import yaml

from .metrics import spectral_angle_rad
from .s1_proxy_inverse import fit_group_isolated_pca, fit_proxy_wls
from .s1_revised_forward import RevisedSkinForwardModel, file_sha256, load_revised_registry


PROJECT_ROOT = Path(__file__).resolve().parents[2]
WAVELENGTHS = np.arange(400.0, 701.0, 10.0)


def _resolve(value: str | Path) -> Path:
    path = Path(value)
    return (path if path.is_absolute() else PROJECT_ROOT / path).resolve()


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def _new_output(path: Path) -> None:
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite existing S1-6 output: {path}")
    path.mkdir(parents=True, exist_ok=False)


def _spectrum(row: Mapping[str, Any], statistic: str) -> np.ndarray:
    return np.asarray([float(row[f"reflectance_{statistic}_{int(w)}nm"]) for w in WAVELENGTHS], dtype=np.float64)


def _center(values: np.ndarray) -> np.ndarray:
    return values - np.mean(values, axis=-1, keepdims=True)


def _side(region: str) -> str:
    if region == "left_cheek":
        return "image_left"
    if region == "right_cheek":
        return "image_right"
    return "other"


def _side_gains(primary: pd.DataFrame, statistic: str, epsilon: float) -> dict[str, float]:
    differences: list[float] = []
    for _, group in primary.groupby("subject_id", sort=True):
        by_region = {
            str(row.region): np.log(np.maximum(_spectrum(row._asdict(), statistic), epsilon))
            for row in group.itertuples(index=False)
        }
        if set(by_region) != {"left_cheek", "right_cheek"}:
            raise RuntimeError("Frozen Train primary cheeks are not paired")
        differences.append(float(np.mean(by_region["left_cheek"] - by_region["right_cheek"])))
    difference = float(np.median(differences))
    return {"image_left": difference / 2.0, "image_right": -difference / 2.0, "other": 0.0}


def _calibrate(
    primary: pd.DataFrame,
    statistic: str,
    epsilon: float,
    pca_components: int,
) -> dict[str, Any]:
    gains = _side_gains(primary, statistic, epsilon)
    subject_logs: list[np.ndarray] = []
    subject_ids: list[str] = []
    for subject, group in primary.groupby("subject_id", sort=True):
        logs = [
            np.log(np.maximum(_spectrum(row._asdict(), statistic), epsilon)) - gains[_side(str(row.region))]
            for row in group.itertuples(index=False)
        ]
        subject_logs.append(np.mean(np.stack(logs), axis=0))
        subject_ids.append(str(subject))
    logs = np.stack(subject_logs)
    ids = np.asarray(subject_ids)
    ref_log = logs.mean(axis=0)
    shape = _center(logs - ref_log)
    raw = logs - ref_log
    shape_pca = fit_group_isolated_pca(shape, ids, n_components=pca_components)
    raw_pca = fit_group_isolated_pca(raw, ids, n_components=pca_components)
    return {
        "side_log_gain": gains,
        "ref_log": ref_log,
        "shape_pca_mean": shape_pca.mean,
        "shape_pca_components": shape_pca.components,
        "shape_pca_singular_values": shape_pca.singular_values,
        "raw_pca_mean": raw_pca.mean,
        "raw_pca_components": raw_pca.components,
        "raw_pca_singular_values": raw_pca.singular_values,
        "training_subject_ids": tuple(subject_ids),
    }


def _project(values: np.ndarray, mean: np.ndarray, components: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    score = (values - mean) @ components
    return mean + components @ score, score


def _metrics(observed: np.ndarray, predicted: np.ndarray, target: np.ndarray, fitted: np.ndarray, epsilon: float) -> dict[str, float]:
    raw_residual = np.log(np.maximum(predicted, epsilon)) - np.log(np.maximum(observed, epsilon))
    return {
        "shape_log_rmse": float(np.sqrt(np.mean((fitted - target) ** 2))),
        "shape_mae": float(np.mean(np.abs(fitted - target))),
        "raw_log_rmse": float(np.sqrt(np.mean(raw_residual ** 2))),
        "raw_sam": float(spectral_angle_rad(predicted, observed, epsilon)),
    }


def _fit_row(
    row: Mapping[str, Any],
    calibration: Mapping[str, Any],
    registry: Any,
    statistic: str,
    epsilon: float,
    model_ids: tuple[str, ...],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    sample_id = str(row["sample_id"])
    subject_id = str(row["subject_id"])
    region = str(row["region"])
    side_log_gain = float(calibration["side_log_gain"][_side(region)])
    side_gain = float(np.exp(side_log_gain))
    observed = _spectrum(row, statistic)
    corrected = observed / side_gain
    ref_log = np.asarray(calibration["ref_log"], dtype=np.float64)
    reference = np.exp(ref_log)
    difference = np.log(np.maximum(corrected, epsilon)) - ref_log
    amplitude = float(np.mean(difference))
    target = _center(difference)
    base = {
        "sample_id": sample_id,
        "subject_id": subject_id,
        "split": str(row["split"]),
        "expression": str(row["expression"]),
        "direction": str(row["direction"]),
        "region": region,
        "analysis_role": str(row["s1_2_analysis_role"]),
    }
    records: list[dict[str, Any]] = []
    residuals: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    for model_id in model_ids:
        try:
            parameters: dict[str, float] = {}
            if model_id == "B0-S":
                fitted = np.zeros(WAVELENGTHS.size)
                predicted = reference * np.exp(amplitude + fitted) * side_gain
                design_rank = 0
                design_dimension = 0
            elif model_id == "B0-R":
                fitted = np.zeros(WAVELENGTHS.size)
                predicted = reference * side_gain
                design_rank = 0
                design_dimension = 0
            elif model_id == "B2-PCA":
                fitted, score = _project(
                    target,
                    np.asarray(calibration["shape_pca_mean"]),
                    np.asarray(calibration["shape_pca_components"]),
                )
                parameters = {"pca_1": float(score[0]), "pca_2": float(score[1])}
                predicted = reference * np.exp(amplitude + fitted) * side_gain
                design_rank = int(np.linalg.matrix_rank(np.asarray(calibration["shape_pca_components"])))
                design_dimension = 2
            elif model_id == "B2-RPCA":
                fitted_raw, score = _project(
                    difference,
                    np.asarray(calibration["raw_pca_mean"]),
                    np.asarray(calibration["raw_pca_components"]),
                )
                fitted = _center(fitted_raw)
                parameters = {"pca_raw_1": float(score[0]), "pca_raw_2": float(score[1])}
                predicted = reference * np.exp(fitted_raw) * side_gain
                design_rank = int(np.linalg.matrix_rank(np.asarray(calibration["raw_pca_components"])))
                design_dimension = 2
            else:
                fit = fit_proxy_wls(model_id, corrected, reference, registry=registry, epsilon=epsilon)
                fitted = fit.fitted_centered_log_ratio
                parameters = dict(zip(fit.parameter_names, (float(v) for v in fit.theta), strict=True))
                predicted = reference * np.exp(fit.amplitude_log + fitted) * side_gain
                design_rank = int(fit.rank)
                design_dimension = len(fit.parameter_names)
            if not np.all(np.isfinite(predicted)) or np.any(predicted <= 0):
                raise FloatingPointError("Non-positive or non-finite reconstructed spectrum")
            records.append({
                **base,
                "model_id": model_id,
                **_metrics(observed, predicted, target, fitted, epsilon),
                "a_obs": amplitude,
                "side_log_gain": side_log_gain,
                "fit_success": True,
                "parameters_finite": bool(all(np.isfinite(value) for value in parameters.values())),
                "design_rank": design_rank,
                "design_dimension": design_dimension,
                **parameters,
            })
            residuals.extend(
                {
                    **base,
                    "model_id": model_id,
                    "wavelength_nm": float(wavelength),
                    "shape_residual": float(fitted[index] - target[index]),
                    "raw_log_residual": float(np.log(predicted[index]) - np.log(observed[index])),
                }
                for index, wavelength in enumerate(WAVELENGTHS)
            )
        except Exception as exc:
            failures.append({**base, "model_id": model_id, "error": repr(exc)})
    return records, residuals, failures


def _candidate_summary(
    fits: pd.DataFrame,
    residuals: pd.DataFrame,
    selectable: list[str],
    dimensions: Mapping[str, int],
    expected_spectra: int,
    expected_subjects: int,
    gates: Mapping[str, Any],
) -> tuple[dict[str, dict[str, Any]], pd.DataFrame]:
    subject_metrics = (
        fits.groupby(["subject_id", "model_id"], as_index=False)
        .agg(shape_log_rmse=("shape_log_rmse", "mean"), raw_log_rmse=("raw_log_rmse", "mean"), raw_sam=("raw_sam", "mean"))
    )
    pivot = subject_metrics.pivot(index="subject_id", columns="model_id", values="shape_log_rmse")
    if "B0-S" not in pivot:
        raise RuntimeError("B0-S Validation baseline is missing")
    result: dict[str, dict[str, Any]] = {}
    for model_id in selectable:
        group = fits[fits.model_id == model_id]
        subject_group = subject_metrics[subject_metrics.model_id == model_id]
        band = residuals[residuals.model_id == model_id].groupby("wavelength_nm").shape_residual.median()
        complete = len(group) == expected_spectra and group.subject_id.nunique() == expected_subjects
        finite = bool(
            complete
            and np.isfinite(group[["shape_log_rmse", "raw_log_rmse", "raw_sam"]].to_numpy()).all()
            and group.parameters_finite.astype(bool).all()
        )
        full_rank = bool(complete and (group.design_rank == group.design_dimension).all())
        improved_fraction = float((pivot[model_id] < pivot["B0-S"]).mean()) if model_id in pivot else 0.0
        median_error = float(subject_group.shape_log_rmse.median()) if len(subject_group) else float("inf")
        baseline_error = float(subject_metrics[subject_metrics.model_id == "B0-S"].shape_log_rmse.median())
        ratio = median_error / baseline_error if baseline_error > 0 else float("inf")
        max_band = float(band.abs().max()) if len(band) else float("inf")
        checks = {
            "complete": complete,
            "finite": finite,
            "full_design_rank": full_rank,
            "improved_subject_fraction": improved_fraction >= float(gates["improved_subject_fraction_vs_B0S_min"]),
            "median_ratio_vs_B0S": ratio <= float(gates["median_shape_log_rmse_ratio_vs_B0S_max"]),
            "band_residual": max_band <= float(gates["max_absolute_median_band_shape_residual"]),
        }
        result[model_id] = {
            "dimension": int(dimensions[model_id]),
            "spectra": int(len(group)),
            "subjects": int(subject_group.subject_id.nunique()),
            "median_subject_shape_log_rmse": median_error,
            "median_subject_raw_log_rmse": float(subject_group.raw_log_rmse.median()) if len(subject_group) else float("inf"),
            "median_subject_raw_sam": float(subject_group.raw_sam.median()) if len(subject_group) else float("inf"),
            "improved_subject_fraction_vs_B0S": improved_fraction,
            "median_shape_log_rmse_ratio_vs_B0S": ratio,
            "max_absolute_median_band_shape_residual": max_band,
            "checks": checks,
            "eligible": bool(all(checks.values())),
        }
    return result, subject_metrics


def select_representation(
    candidates: Mapping[str, Mapping[str, Any]],
    tolerance: float,
    preference: list[str],
) -> dict[str, Any]:
    eligible = {name: value for name, value in candidates.items() if bool(value.get("eligible"))}
    if not eligible:
        return {"selected_model": None, "eligible_models": [], "best_error": None, "tolerance_limit": None}
    best_error = min(float(value["median_subject_shape_log_rmse"]) for value in eligible.values())
    limit = best_error * (1.0 + float(tolerance))
    near_best = {name: value for name, value in eligible.items() if float(value["median_subject_shape_log_rmse"]) <= limit}
    minimum_dimension = min(int(value["dimension"]) for value in near_best.values())
    dimension_matches = {name for name, value in near_best.items() if int(value["dimension"]) == minimum_dimension}
    selected = next((name for name in preference if name in dimension_matches), sorted(dimension_matches)[0])
    return {
        "selected_model": selected,
        "eligible_models": sorted(eligible),
        "near_best_models": sorted(near_best),
        "best_error": best_error,
        "tolerance_limit": limit,
        "selected_dimension": minimum_dimension,
    }


def _assert_no_null(value: Any, path: str = "root") -> None:
    if value is None:
        raise ValueError(f"Frozen specification contains null at {path}")
    if isinstance(value, Mapping):
        for key, item in value.items():
            _assert_no_null(item, f"{path}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            _assert_no_null(item, f"{path}[{index}]")


def run_s1_6(
    config_path: str | Path,
    output_dir: str | Path,
    progress: Callable[[str], None] = print,
) -> dict[str, Any]:
    config_file = _resolve(config_path)
    output = _resolve(output_dir)
    config = yaml.safe_load(config_file.read_text(encoding="utf-8"))
    if config.get("stage") != "S1-6":
        raise ValueError("Unexpected S1-6 configuration")
    policy = config.get("access_policy", {})
    if not policy.get("validation_access_allowed") or policy.get("test_access_allowed") or policy.get("raw_hsi_access_allowed"):
        raise ValueError("Invalid S1-6 access policy")
    _new_output(output)
    inputs = {key: _resolve(value) for key, value in config["inputs"].items()}
    representation_decision = _read_json(inputs["s1_5r_representation_decision"])
    if representation_decision.get("status") != "REPRESENTATION_READY_FOR_S1_6" or not representation_decision.get("next_stage_allowed"):
        raise RuntimeError("S1-5R representation decision does not authorize S1-6")
    if representation_decision.get("authorized_next_stage") != "S1-6":
        raise RuntimeError("S1-5R authorization scope mismatch")
    s1_3_decision = _read_json(inputs["s1_3_decision"])
    target_decision = _read_json(inputs["s1_2_target_domain_decision"])
    if s1_3_decision.get("status") != "PASS_FOR_S1_4" or target_decision.get("status") != "PASS_FOR_DEVELOPMENT":
        raise RuntimeError("Upstream data or target-domain decision is not valid")
    registry = load_revised_registry(inputs["model_registry"])

    progress("Loading frozen Train and authorized Validation region spectra")
    data = pd.read_parquet(inputs["region_spectra"])
    observed_splits = set(data.split.astype(str).unique())
    if not observed_splits.issubset({config["data_roles"]["train_split"], config["data_roles"]["validation_split"]}):
        raise RuntimeError(f"Unexpected split reached S1-6: {sorted(observed_splits)}")
    roles = config["data_roles"]
    train = data[data.split == roles["train_split"]].copy()
    validation = data[data.split == roles["validation_split"]].copy()
    primary_train = train[
        (train.s1_2_analysis_role == roles["train_analysis_role"])
        & (train.expression == roles["primary_expression"])
        & (train.direction == roles["primary_direction"])
        & train.region.isin(roles["primary_regions"])
        & (train.extraction_status == roles["usable_status"])
        & train.s1_2_usable_flag.astype(bool)
    ].copy()
    primary_validation = validation[
        (validation.s1_2_analysis_role == roles["validation_analysis_role"])
        & (validation.expression == roles["primary_expression"])
        & (validation.direction == roles["primary_direction"])
        & validation.region.isin(roles["primary_regions"])
        & (validation.extraction_status == roles["usable_status"])
        & validation.s1_2_usable_flag.astype(bool)
    ].copy()
    expected = (
        (primary_train.subject_id.nunique(), len(primary_train)),
        (primary_validation.subject_id.nunique(), len(primary_validation)),
    )
    required = (
        (int(roles["expected_train_subjects"]), int(roles["expected_train_primary_spectra"])),
        (int(roles["expected_validation_subjects"]), int(roles["expected_validation_primary_spectra"])),
    )
    if expected != required:
        raise RuntimeError(f"S1-6 cohort mismatch: observed={expected}, required={required}")

    statistic = str(config["calibration"]["statistic"])
    epsilon = float(config["calibration"]["reflectance_epsilon"])
    progress("Freezing full-Train calibration")
    calibration = _calibrate(primary_train, statistic, epsilon, int(config["calibration"]["pca_components"]))
    calibration_path = output / "frozen_calibration.npz"
    np.savez_compressed(
        calibration_path,
        wavelength_nm=WAVELENGTHS,
        band_weights=np.ones(WAVELENGTHS.size),
        reference_log_reflectance=calibration["ref_log"],
        reference_reflectance=np.exp(calibration["ref_log"]),
        shape_pca_mean=calibration["shape_pca_mean"],
        shape_pca_components=calibration["shape_pca_components"],
        shape_pca_singular_values=calibration["shape_pca_singular_values"],
        raw_pca_mean=calibration["raw_pca_mean"],
        raw_pca_components=calibration["raw_pca_components"],
        raw_pca_singular_values=calibration["raw_pca_singular_values"],
        training_subject_ids=np.asarray(calibration["training_subject_ids"]),
        side_log_gain_image_left=np.asarray(calibration["side_log_gain"]["image_left"]),
        side_log_gain_image_right=np.asarray(calibration["side_log_gain"]["image_right"]),
    )

    selectable = list(config["models"]["selectable"])
    all_primary_models = tuple(config["models"]["zero_dimensional_baselines"] + selectable + config["models"]["diagnostic_comparators"])
    authorized = set(representation_decision["validation_candidate_models"])
    if set(all_primary_models) != authorized:
        raise RuntimeError("S1-6 model list differs from the authorized S1-5R candidate list")
    progress("Fitting preregistered candidates on six Validation primary cheek spectra")
    fit_records: list[dict[str, Any]] = []
    residual_records: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    for row in primary_validation.sort_values(["subject_id", "region"]).to_dict("records"):
        current_fits, current_residuals, current_failures = _fit_row(
            row, calibration, registry, statistic, epsilon, all_primary_models
        )
        fit_records.extend(current_fits)
        residual_records.extend(current_residuals)
        failures.extend(current_failures)
    fits = pd.DataFrame(fit_records)
    residuals = pd.DataFrame(residual_records)
    candidate_summary, subject_metrics = _candidate_summary(
        fits,
        residuals,
        selectable,
        config["models"]["dimensions"],
        int(roles["expected_validation_primary_spectra"]),
        int(roles["expected_validation_subjects"]),
        config["selection_gates"],
    )
    selection = select_representation(
        candidate_summary,
        float(config["selection_gates"]["parsimony_relative_tolerance_from_best"]),
        list(config["models"]["same_dimension_preference"]),
    )
    selected_model = selection["selected_model"]

    stress_records: list[dict[str, Any]] = []
    stress_residuals: list[dict[str, Any]] = []
    stress_failures: list[dict[str, Any]] = []
    if selected_model is not None:
        progress(f"Running nonblocking Validation stress report for {selected_model}")
        stress = validation[
            (validation.s1_2_analysis_role == roles["stress_analysis_role"])
            & (validation.extraction_status == roles["usable_status"])
            & validation.s1_2_usable_flag.astype(bool)
        ]
        for row in stress.sort_values(["subject_id", "sample_id", "region"]).to_dict("records"):
            current_fits, current_residuals, current_failures = _fit_row(
                row, calibration, registry, statistic, epsilon, (selected_model,)
            )
            stress_records.extend(current_fits)
            stress_residuals.extend(current_residuals)
            stress_failures.extend(current_failures)
    stress_fits = pd.DataFrame(stress_records)
    stress_residual_frame = pd.DataFrame(stress_residuals)
    failures.extend(stress_failures)

    fits.to_parquet(output / "validation_per_spectrum_fits.parquet", index=False)
    fits.to_csv(output / "validation_per_spectrum_fits.csv", index=False)
    subject_metrics.to_csv(output / "validation_subject_metrics.csv", index=False)
    residuals.to_parquet(output / "validation_wavelength_residuals.parquet", index=False)
    residuals.to_csv(output / "validation_wavelength_residuals.csv", index=False)
    stress_fits.to_parquet(output / "validation_stress_fits.parquet", index=False)
    stress_fits.to_csv(output / "validation_stress_fits.csv", index=False)
    stress_residual_frame.to_parquet(output / "validation_stress_residuals.parquet", index=False)
    _write_json(output / "candidate_summary.json", {"candidates": candidate_summary, "selection": selection})
    _write_json(output / "fit_failures.json", {
        "count": len(failures),
        "primary_count": len([v for v in failures if v.get("analysis_role") == roles["validation_analysis_role"]]),
        "stress_count": len(stress_failures),
        "failures": failures,
    })

    audit = {
        "config": str(config_file),
        "config_sha256": file_sha256(config_file),
        "inputs": {key: {"path": str(path), "sha256": file_sha256(path)} for key, path in inputs.items()},
        "source_sha256": {
            Path(__file__).name: file_sha256(Path(__file__)),
            "s1_proxy_inverse.py": file_sha256(PROJECT_ROOT / "src/skin_optics_hsi/s1_proxy_inverse.py"),
            "s1_revised_forward.py": file_sha256(PROJECT_ROOT / "src/skin_optics_hsi/s1_revised_forward.py"),
            "run_s1_6_validation_freeze.py": file_sha256(PROJECT_ROOT / "scripts/skin_optics_hsi/run_s1_6_validation_freeze.py"),
        },
        "train_rows_loaded": int(len(train)),
        "train_primary_rows_used_for_calibration": int(len(primary_train)),
        "validation_rows_loaded": int(len(validation)),
        "validation_primary_rows_used_for_selection": int(len(primary_validation)),
        "validation_stress_rows_used_for_nonblocking_report": int(len(stress_fits)),
        "test_rows_read": 0,
        "raw_hsi_files_read": 0,
        "preselection_threshold_source": "config_frozen_before_validation_metric_computation",
        "python": platform.python_version(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "scipy": scipy.__version__,
    }
    _write_json(output / "input_audit.json", audit)

    status = "S1_6_FROZEN_FOR_S1_7" if selected_model is not None else "S1_6_REVISE_BEFORE_TEST"
    if selected_model is not None:
        if selected_model == "B2-PCA":
            parameter_names = ["pca_1", "pca_2"]
            evidence_level = "data_driven"
        else:
            model = RevisedSkinForwardModel(selected_model, registry)
            parameter_names = list(model.parameter_names)
            evidence_level = "physics_informed_proxy"
        frozen_spec = {
            "schema_version": 1,
            "stage": "S1-6",
            "status": "FROZEN_FOR_SINGLE_S1_7_TEST",
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "selected_representation_model": selected_model,
            "selected_representation_dimension": int(config["models"]["dimensions"][selected_model]),
            "theta_repr": {"parameter_names": parameter_names, "evidence_level": evidence_level},
            "theta_aux": {"parameter_names": ["a_obs"], "role": "analytic_log_amplitude_for_reconstruction"},
            "observation_space": "centered_log_ratio_with_separate_log_amplitude",
            "wavelength_contract": {
                "centers_nm": [float(v) for v in WAVELENGTHS],
                "ordering": "ascending",
                "effective_srf_status": str(config["freeze_policy"]["effective_srf_status"]),
                "band_weights": "uniform",
            },
            "target_domain": {
                "expression": str(roles["primary_expression"]),
                "direction": str(roles["primary_direction"]),
                "regions": list(roles["primary_regions"]),
                "eyewear_requirement_for_deployment": "no_eyewear",
                "eyewear_validation_status": "not_available_in_hyperskin",
            },
            "calibration": {
                "path": str(calibration_path),
                "sha256": file_sha256(calibration_path),
                "reference": str(config["calibration"]["reference"]),
                "side_log_gain": {key: float(value) for key, value in calibration["side_log_gain"].items()},
                "refit_on_validation_or_test": False,
            },
            "model_registry": {"path": str(inputs["model_registry"]), "sha256": file_sha256(inputs["model_registry"])},
            "selection": {
                "metric": str(config["selection_policy"]["selection_metric"]),
                "aggregation_unit": "subject",
                "gates": config["selection_gates"],
                "candidate_summary_sha256": file_sha256(output / "candidate_summary.json"),
            },
            "interpretation_boundary": {
                "layer": "deferred_to_stage2",
                "physiological_M_H_claim": "not_assessed_in_stage1",
                "coordinate_names_are_model_coordinates": True,
            },
            "formal_test_policy": {
                "authorized_stage": "S1-7",
                "maximum_formal_runs": int(config["freeze_policy"]["formal_test_runs_allowed_after_pass"]),
                "global_parameter_refit_allowed": False,
                "model_or_threshold_changes_allowed": False,
                "test_access_before_freeze": False,
            },
            "upstream_decisions": {
                "s1_5r_representation": {"path": str(inputs["s1_5r_representation_decision"]), "sha256": file_sha256(inputs["s1_5r_representation_decision"])},
                "s1_3": {"path": str(inputs["s1_3_decision"]), "sha256": file_sha256(inputs["s1_3_decision"])},
                "s1_2_target_domain": {"path": str(inputs["s1_2_target_domain_decision"]), "sha256": file_sha256(inputs["s1_2_target_domain_decision"])},
            },
        }
        _assert_no_null(frozen_spec)
        (output / "frozen_stage1_spec.yaml").write_text(
            yaml.safe_dump(frozen_spec, sort_keys=False, allow_unicode=True), encoding="utf-8"
        )

    report = [
        "# S1-6 Validation selection and freeze report",
        "",
        f"Status: `{status}`  ",
        f"Selected representation: `{selected_model if selected_model is not None else 'NONE'}`  ",
        "Test/raw-HSI reads: `0/0`",
        "",
        f"- Train calibration subjects/spectra: {primary_train.subject_id.nunique()}/{len(primary_train)}",
        f"- Validation selection subjects/spectra: {primary_validation.subject_id.nunique()}/{len(primary_validation)}",
        f"- Nonblocking Validation stress spectra: {len(stress_fits)}",
        f"- Eligible models: {', '.join(selection.get('eligible_models', [])) or 'none'}",
        f"- Near-best models under the frozen parsimony tolerance: {', '.join(selection.get('near_best_models', [])) or 'none'}",
        "",
        "The decision concerns representation feasibility only. Coordinate physiology is deferred to Stage 2.",
    ]
    (output / "S1_6_VALIDATION_REPORT.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    output_hashes = {path.name: file_sha256(path) for path in output.iterdir() if path.is_file()}
    decision = {
        "schema_version": 1,
        "stage": "S1-6",
        "status": status,
        "next_stage_allowed": selected_model is not None,
        "authorized_next_stage": "S1-7" if selected_model is not None else "NOT_AUTHORIZED",
        "selected_representation_model": selected_model if selected_model is not None else "NONE",
        "selected_representation_dimension": int(config["models"]["dimensions"][selected_model]) if selected_model is not None else 0,
        "eligible_models": selection.get("eligible_models", []),
        "near_best_models": selection.get("near_best_models", []),
        "interpretation_layer_status": "DEFERRED_TO_S2",
        "physiological_M_H_claim": "NOT_ASSESSED_IN_S1",
        "validation_rows_loaded": int(len(validation)),
        "validation_primary_rows_used_for_selection": int(len(primary_validation)),
        "test_rows_read": 0,
        "raw_hsi_files_read": 0,
        "formal_test_run_limit": 1 if selected_model is not None else 0,
        "outputs": output_hashes,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    _write_json(output / "s1_6_decision.json", decision)
    _write_json(output / "output_manifest.json", {
        "files": {path.name: file_sha256(path) for path in output.iterdir() if path.is_file()},
        "test_rows_read": 0,
        "raw_hsi_files_read": 0,
    })
    return decision


__all__ = ["run_s1_6", "select_representation"]
