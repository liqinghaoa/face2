"""S1-5 Train-only inversion, diagnostics, and decision artifacts."""

from __future__ import annotations

import json
import platform
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path
from typing import Any, Callable, Mapping

import numpy as np
import pandas as pd
import yaml
import scipy
import torch

from .metrics import spectral_angle_rad, spectral_metrics
from .model_registry import load_model_registry
from .s1_inverse import fit_registered_spectrum, profile_registered_spectrum
from .s1_spectral_sensitivity import GaussianSRFModelAdapter, band_mask, shifted_centers
from .skin_forward import RegisteredSkinForwardModel


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_S1_5_CONFIG = PROJECT_ROOT / "configs" / "skin_optics_hsi" / "s1_5_train_development_v1.yaml"
WAVELENGTHS = np.arange(400.0, 701.0, 10.0)
PARAMETER_NAMES = ("M_absorbance", "Hb_absorbance_proxy", "S_amp", "sO2")


def _sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as stream:
        value = json.load(stream)
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def _resolve(project_root: Path, value: str | Path) -> Path:
    path = Path(value)
    return (path if path.is_absolute() else project_root / path).resolve()


def _new_output(path: Path) -> None:
    if path.exists() and any(path.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty S1-5 output: {path}")
    path.mkdir(parents=True, exist_ok=True)


def _spectrum(row: Mapping[str, Any], statistic: str = "median") -> np.ndarray:
    prefix = f"reflectance_{statistic}_"
    return np.asarray([float(row[f"{prefix}{int(wavelength)}nm"]) for wavelength in WAVELENGTHS], dtype=np.float64)


def _region_side(region: str) -> str:
    if region == "left_cheek":
        return "image_left"
    if region == "right_cheek":
        return "image_right"
    return "other"


def _objective_from_prediction(
    predicted: np.ndarray,
    observed: np.ndarray,
    fit_cfg: Mapping[str, Any],
) -> float:
    epsilon = float(fit_cfg["reflectance_epsilon"])
    delta = float(fit_cfg["log_pseudo_huber_delta"])
    difference = np.log(np.maximum(predicted, epsilon)) - np.log(np.maximum(observed, epsilon))
    robust = delta * delta * (np.sqrt(1.0 + (difference / delta) ** 2) - 1.0)
    sam = spectral_angle_rad(predicted, observed, epsilon)
    return float(np.mean(robust) + float(fit_cfg["sam_weight"]) * sam * sam)


def _estimate_side_calibration(primary: pd.DataFrame) -> dict[str, Any]:
    records: list[dict[str, Any]] = []
    for subject_id, group in primary.groupby("subject_id", sort=True):
        by_region = {str(row.region): _spectrum(row._asdict()) for row in group.itertuples(index=False)}
        if set(by_region) != {"left_cheek", "right_cheek"}:
            raise RuntimeError(f"Primary paired cheeks missing for {subject_id}")
        log_difference = np.log(np.maximum(by_region["left_cheek"], 1e-6)) - np.log(
            np.maximum(by_region["right_cheek"], 1e-6)
        )
        records.append(
            {
                "subject_id": subject_id,
                "mean_log_left_minus_right": float(np.mean(log_difference)),
                "median_log_left_minus_right": float(np.median(log_difference)),
            }
        )
    differences = np.asarray([record["mean_log_left_minus_right"] for record in records])
    d = float(np.median(differences))
    return {
        "method": "median_subject_mean_log_left_minus_right",
        "subject_count": len(records),
        "median_log_left_minus_right": d,
        "side_log_gain_by_region": {"image_left": d / 2.0, "image_right": -d / 2.0, "other": 0.0},
        "subject_estimates": records,
        "status": "estimated_from_train_primary_only",
    }


def _correct_observation_for_side(spectrum: np.ndarray, region: str, gains: Mapping[str, float]) -> np.ndarray:
    return spectrum / np.exp(float(gains[_region_side(region)]))


def _train_mean(primary: pd.DataFrame, gains: Mapping[str, float], excluded_subject: str | None = None) -> np.ndarray:
    selected = primary if excluded_subject is None else primary[primary["subject_id"] != excluded_subject]
    corrected = [
        _correct_observation_for_side(_spectrum(row._asdict()), str(row.region), gains)
        for row in selected.itertuples(index=False)
    ]
    return np.mean(np.stack(corrected), axis=0)


def _fit_record(
    row: Mapping[str, Any],
    model_id: str,
    model: RegisteredSkinForwardModel,
    fit: Any,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    base = {
        "sample_id": str(row["sample_id"]),
        "subject_id": str(row["subject_id"]),
        "expression": str(row["expression"]),
        "direction": str(row["direction"]),
        "region": str(row["region"]),
        "analysis_role": str(row["s1_2_analysis_role"]),
        "model_id": model_id,
    }
    record: dict[str, Any] = {
        **base,
        "fit_status": "SUCCESS",
        "objective": float(fit.objective),
        **{name: float("nan") for name in PARAMETER_NAMES},
        **fit.metrics,
        "jacobian_rank": int(fit.identifiability["jacobian_rank"]),
        "jacobian_condition_number": float(fit.identifiability["jacobian_condition_number"]),
        "solution_cluster_count": int(fit.identifiability["solution_cluster_count"]),
        "near_optimal_start_count": int(fit.identifiability["near_optimal_start_count"]),
        "any_boundary": bool(any(fit.identifiability["boundary_flags"])),
        "multistart_theta_relative_range": json.dumps(fit.identifiability["multistart_theta_relative_range"]),
        "jacobian_singular_values": json.dumps(fit.identifiability["jacobian_singular_values"]),
        "local_covariance_diagonal_unscaled": json.dumps(
            fit.identifiability["local_covariance_diagonal_unscaled"]
        ),
    }
    for name, value, boundary in zip(
        model.parameter_names,
        fit.theta,
        fit.identifiability["boundary_flags"],
        strict=True,
    ):
        record[name] = float(value)
        record[f"boundary_{name}"] = bool(boundary)
    start_rows = [
        {
            **base,
            "parameter_names": json.dumps(list(model.parameter_names)),
            "initial_theta": json.dumps(start["initial_theta"]),
            "final_theta": json.dumps(start["final_theta"]),
            **{key: value for key, value in start.items() if key not in {"initial_theta", "final_theta"}},
        }
        for start in fit.starts
    ]
    observed = _spectrum(row)
    residual_rows = [
        {
            **base,
            "wavelength_nm": float(wavelength),
            "observed_reflectance": float(observed[index]),
            "predicted_reflectance": float(fit.predicted[index]),
            "residual": float(fit.predicted[index] - observed[index]),
            "log_residual": float(np.log(max(fit.predicted[index], 1e-6)) - np.log(max(observed[index], 1e-6))),
            "selected_in_fit": bool(fit.valid_band_mask[index]),
        }
        for index, wavelength in enumerate(WAVELENGTHS)
    ]
    return record, start_rows, residual_rows


def _fit_primary(
    primary: pd.DataFrame,
    registry: Any,
    config: Mapping[str, Any],
    side_calibration: Mapping[str, Any],
    output: Path,
    progress: Callable[[str], None],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    fit_cfg = config["fit"]
    gains = side_calibration["side_log_gain_by_region"]
    global_params = {"side_log_gain_by_region": gains}
    train_mean_all = _train_mean(primary, gains)
    fits: list[dict[str, Any]] = []
    starts: list[dict[str, Any]] = []
    residuals: list[dict[str, Any]] = []
    total = len(primary) * len(config["models"]["primary"])
    completed = 0
    models = {model_id: RegisteredSkinForwardModel(model_id, registry) for model_id in config["models"]["primary"]}
    for row_tuple in primary.sort_values(["subject_id", "region"]).itertuples(index=False):
        row = row_tuple._asdict()
        observed = _spectrum(row)
        side = _region_side(str(row["region"]))
        for model_id, model in models.items():
            base = {
                "sample_id": str(row["sample_id"]),
                "subject_id": str(row["subject_id"]),
                "expression": str(row["expression"]),
                "direction": str(row["direction"]),
                "region": str(row["region"]),
                "analysis_role": str(row["s1_2_analysis_role"]),
                "model_id": model_id,
            }
            try:
                if model_id == "B0":
                    reference = _train_mean(primary, gains, excluded_subject=str(row["subject_id"]))
                    predicted = np.asarray(
                        model.forward_numpy(
                            None,
                            WAVELENGTHS,
                            global_params={**global_params, "train_mean_reflectance": reference},
                            observation_context={"region_side": side},
                        )
                    )
                    metric = spectral_metrics(predicted, observed, float(fit_cfg["reflectance_epsilon"]))
                    fits.append(
                        {
                            **base,
                            "fit_status": "SUCCESS",
                            "objective": _objective_from_prediction(predicted, observed, fit_cfg),
                            **{name: float("nan") for name in PARAMETER_NAMES},
                            **metric,
                            "jacobian_rank": 0,
                            "jacobian_condition_number": float("nan"),
                            "solution_cluster_count": 1,
                            "near_optimal_start_count": 1,
                            "any_boundary": False,
                            "b0_reference": "leave_one_subject_out_train_mean",
                        }
                    )
                    residuals.extend(
                        {
                            **base,
                            "wavelength_nm": float(wavelength),
                            "observed_reflectance": float(observed[index]),
                            "predicted_reflectance": float(predicted[index]),
                            "residual": float(predicted[index] - observed[index]),
                            "log_residual": float(np.log(max(predicted[index], 1e-6)) - np.log(max(observed[index], 1e-6))),
                            "selected_in_fit": True,
                        }
                        for index, wavelength in enumerate(WAVELENGTHS)
                    )
                else:
                    result = fit_registered_spectrum(
                        observed,
                        WAVELENGTHS,
                        model,
                        fit_cfg,
                        n_starts=int(fit_cfg["n_starts_primary"]),
                        global_params=global_params,
                        observation_context={"region_side": side},
                    )
                    record, start_rows, residual_rows = _fit_record(row, model_id, model, result)
                    fits.append(record)
                    starts.extend(start_rows)
                    residuals.extend(residual_rows)
            except Exception as exc:
                fits.append(
                    {
                        **base,
                        "fit_status": "FAIL",
                        "failure_type": type(exc).__name__,
                        "failure_message": str(exc),
                        **{name: float("nan") for name in PARAMETER_NAMES},
                    }
                )
            completed += 1
            if completed % 16 == 0 or completed == total:
                progress(f"S1-5 primary fits: {completed}/{total}")
                _write_json(output / "progress.json", {"phase": "primary", "completed": completed, "total": total})
    _write_json(
        output / "train_global_calibration.json",
        {
            "side_calibration": side_calibration,
            "train_mean_reflectance": train_mean_all.tolist(),
            "wavelength_nm": WAVELENGTHS.tolist(),
            "b0_development_comparison": "leave_one_subject_out_by_subject",
            "future_validation_reference": "full_Train_primary_mean_fixed_here",
        },
    )
    return pd.DataFrame(fits), pd.DataFrame(starts), pd.DataFrame(residuals)


def _hb_trigger(
    primary_residuals: pd.DataFrame,
    registry: Any,
    config: Mapping[str, Any],
) -> dict[str, Any]:
    p2 = primary_residuals[primary_residuals["model_id"] == "P2"].copy()
    pivot = p2.groupby(["subject_id", "wavelength_nm"], sort=True)["log_residual"].mean().unstack()
    model = RegisteredSkinForwardModel("P3-O", registry)
    reference = np.asarray([0.13, 0.18])
    low = np.asarray(model.forward_numpy(reference, WAVELENGTHS, np.asarray([0.20])))
    high = np.asarray(model.forward_numpy(reference, WAVELENGTHS, np.asarray([0.90])))
    contrast = np.log(np.maximum(high, 1e-6)) - np.log(np.maximum(low, 1e-6))
    design = np.stack([np.ones_like(WAVELENGTHS), (WAVELENGTHS - WAVELENGTHS.mean()) / np.ptp(WAVELENGTHS)], axis=1)
    projector = design @ np.linalg.pinv(design)
    contrast_detrended = contrast - projector @ contrast
    correlations: list[float] = []
    for row in pivot.to_numpy(dtype=np.float64):
        detrended = row - projector @ row
        correlations.append(float(np.corrcoef(detrended, contrast_detrended)[0, 1]))
    values = np.asarray(correlations)
    median = float(np.nanmedian(values))
    threshold = config["models"]["p3_o_activation"]
    consistent = np.sign(values) == np.sign(median)
    consistent &= np.abs(values) >= float(threshold["subject_correlation_abs_min"])
    fraction = float(np.mean(consistent))
    activated = bool(
        abs(median) >= float(threshold["median_detrended_residual_contrast_correlation_abs_min"])
        and fraction >= float(threshold["consistent_subject_fraction_min"])
    )
    return {
        "method": "P2_subject_symmetric_log_residual_vs_detrended_sO2_contrast",
        "subject_count": int(len(values)),
        "median_correlation": median,
        "median_absolute_correlation": abs(median),
        "consistent_subject_fraction": fraction,
        "subject_correlations": [float(value) for value in values],
        "thresholds": threshold,
        "P3-O_activated": activated,
        "P4_activated": False,
    }


def _primary_subject_comparison(fits: pd.DataFrame) -> pd.DataFrame:
    successful = fits[fits["fit_status"] == "SUCCESS"]
    metrics = ["objective", "rmse", "mae", "log_rmse", "log_mae", "sam_rad"]
    grouped = successful.groupby(["subject_id", "model_id"], sort=True)[metrics].median().reset_index()
    for metric in metrics:
        pivot = grouped.pivot(index="subject_id", columns="model_id", values=metric)
        for comparator in ("B0", "B1", "P2"):
            if comparator in pivot:
                grouped[f"delta_{metric}_vs_{comparator}"] = [
                    float(row[metric] - pivot.loc[row["subject_id"], comparator])
                    for _, row in grouped.iterrows()
                ]
    return grouped


def _parameter_shift(fits: pd.DataFrame, registry: Any) -> dict[str, Any]:
    p2 = fits[(fits["model_id"] == "P2") & (fits["fit_status"] == "SUCCESS")]
    p3 = fits[(fits["model_id"] == "P3-S") & (fits["fit_status"] == "SUCCESS")]
    merged = p2.merge(p3, on=["sample_id", "subject_id", "region"], suffixes=("_P2", "_P3S"))
    result: dict[str, Any] = {"row_count": int(len(merged))}
    for name in ("M_absorbance", "Hb_absorbance_proxy"):
        parameter = registry.parameter(name)
        shift = np.abs(merged[f"{name}_P3S"] - merged[f"{name}_P2"]) / (parameter.maximum - parameter.minimum)
        result[name] = {
            "median_normalized_absolute_shift": float(np.median(shift)),
            "p95_normalized_absolute_shift": float(np.quantile(shift, 0.95)),
            "median_signed_shift": float(np.median(merged[f"{name}_P3S"] - merged[f"{name}_P2"])),
        }
    return result


def _diagnostic_summary(
    fits: pd.DataFrame,
    residuals: pd.DataFrame,
    subject_comparison: pd.DataFrame,
    registry: Any,
    config: Mapping[str, Any],
) -> dict[str, Any]:
    successful = fits[fits["fit_status"] == "SUCCESS"]
    model_summary: dict[str, Any] = {}
    for model_id, group in successful.groupby("model_id", sort=False):
        residual_group = residuals[residuals["model_id"] == model_id]
        median_by_band = residual_group.groupby("wavelength_nm")["log_residual"].median()
        parameter_boundary: dict[str, float] = {}
        model = registry.model(model_id)
        for name in model.flat_parameter_names:
            column = f"boundary_{name}"
            parameter_boundary[name] = float(group[column].eq(True).mean()) if column in group else 0.0
        model_summary[model_id] = {
            "fit_count": int(len(group)),
            "failure_count": int(((fits["model_id"] == model_id) & (fits["fit_status"] != "SUCCESS")).sum()),
            "median_log_rmse": float(group["log_rmse"].median()),
            "median_sam_rad": float(group["sam_rad"].median()),
            "any_boundary_fraction": float(group["any_boundary"].fillna(False).mean()),
            "parameter_boundary_fraction": parameter_boundary,
            "multi_cluster_fraction": float((group["solution_cluster_count"] > 1).mean()) if model_id != "B0" else 0.0,
            "jacobian_condition_number_median": (
                float(group["jacobian_condition_number"].median()) if model_id != "B0" else None
            ),
            "max_abs_median_log_residual_across_bands": float(np.max(np.abs(median_by_band))),
        }
    subject_pivot = subject_comparison.pivot(index="subject_id", columns="model_id", values="log_rmse")
    p3_relative = 1.0 - subject_pivot["P3-S"] / subject_pivot["P2"]
    model_summary["P3-S"]["median_relative_log_rmse_improvement_vs_P2"] = float(np.median(p3_relative))
    best_physical = min(model_summary[model_id]["median_log_rmse"] for model_id in ("B1", "P2", "P3-S"))
    best_physical_id = min(("B1", "P2", "P3-S"), key=lambda model_id: model_summary[model_id]["median_log_rmse"])
    best_physical_to_b0 = best_physical / model_summary["B0"]["median_log_rmse"]
    shift = _parameter_shift(fits, registry)
    thresholds = config["development_decision"]["diagnostic_review_thresholds_not_final_test_gates"]
    review_flags = {
        "physical_models_fail_to_match_B0": best_physical_to_b0
        > float(thresholds["best_physical_to_B0_median_log_rmse_ratio_max"]),
        "P2_boundary_collapse": model_summary["P2"]["any_boundary_fraction"]
        > float(thresholds["any_parameter_boundary_fraction_max"]),
        "P3S_boundary_collapse": model_summary["P3-S"]["any_boundary_fraction"]
        > float(thresholds["any_parameter_boundary_fraction_max"]),
        "P3S_multicluster": model_summary["P3-S"]["multi_cluster_fraction"]
        > float(thresholds["multicluster_fit_fraction_max"]),
        "M_or_H_material_shift": max(
            shift["M_absorbance"]["median_normalized_absolute_shift"],
            shift["Hb_absorbance_proxy"]["median_normalized_absolute_shift"],
        )
        > float(thresholds["median_normalized_M_or_H_shift_P2_to_P3S_max"]),
        "P3S_structured_residual": model_summary["P3-S"]["max_abs_median_log_residual_across_bands"]
        > float(thresholds["median_absolute_log_residual_per_band_max"]),
    }
    p4_rule = config["models"]["p4_activation"]
    scattering_evidence = bool(
        model_summary["P3-S"]["median_relative_log_rmse_improvement_vs_P2"]
        >= float(p4_rule["p3_s_median_log_rmse_relative_improvement_min"])
        and model_summary["P3-S"]["parameter_boundary_fraction"].get("S_amp", 1.0)
        <= float(p4_rule["s_amp_boundary_fraction_max"])
    )
    return {
        "models": model_summary,
        "best_physical_model_by_median_log_rmse": best_physical_id,
        "best_physical_to_B0_median_log_rmse_ratio": float(best_physical_to_b0),
        "P2_to_P3S_parameter_shift": shift,
        "scattering_residual_evidence": scattering_evidence,
        "review_flags": review_flags,
        "review_thresholds_are_development_diagnostics_not_final_gates": thresholds,
    }


def _symmetric_primary(primary: pd.DataFrame, gains: Mapping[str, float], statistic: str) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for subject_id, group in primary.groupby("subject_id", sort=True):
        spectra: list[np.ndarray] = []
        for row in group.itertuples(index=False):
            spectra.append(_correct_observation_for_side(_spectrum(row._asdict(), statistic), str(row.region), gains))
        spectrum = np.mean(np.stack(spectra), axis=0)
        record: dict[str, Any] = {
            "sample_id": f"{subject_id}_neutral_front_symmetric_{statistic}",
            "subject_id": subject_id,
            "expression": "neutral",
            "direction": "front",
            "region": "symmetric_cheeks",
            "s1_2_analysis_role": "primary_development_symmetric",
        }
        for index, wavelength in enumerate(WAVELENGTHS.astype(int)):
            record[f"reflectance_median_{wavelength}nm"] = float(spectrum[index])
        rows.append(record)
    return pd.DataFrame(rows)


def _run_perturbations(
    primary: pd.DataFrame,
    primary_fits: pd.DataFrame,
    registry: Any,
    config: Mapping[str, Any],
    gains: Mapping[str, float],
    progress: Callable[[str], None],
) -> pd.DataFrame:
    fit_cfg = config["fit"]
    perturb_cfg = config["perturbation"]
    repeats = int(perturb_cfg["repeats"])
    rng = np.random.default_rng(int(perturb_cfg["random_seed"]))
    rows: list[dict[str, Any]] = []
    targets = primary_fits[(primary_fits["model_id"].isin(["P2", "P3-S"])) & (primary_fits["fit_status"] == "SUCCESS")]
    source = primary.set_index(["sample_id", "region"])
    total = len(targets) * repeats
    completed = 0
    for fit_row in targets.itertuples(index=False):
        raw = source.loc[(fit_row.sample_id, fit_row.region)].to_dict()
        observed = _spectrum(raw)
        mad = np.asarray([float(raw[f"reflectance_mad_{int(w)}nm"]) for w in WAVELENGTHS])
        trimmed = np.asarray([float(raw[f"reflectance_trimmed_mean_{int(w)}nm"]) for w in WAVELENGTHS])
        pixels = min(int(raw["effective_pixels"]), 100)
        sigma = np.maximum.reduce(
            [1.4826 * mad / np.sqrt(max(pixels, 1)), np.abs(trimmed - observed), 0.005 * observed]
        )
        model = RegisteredSkinForwardModel(fit_row.model_id, registry)
        baseline = np.asarray([getattr(fit_row, name) for name in model.parameter_names], dtype=np.float64)
        spans = np.asarray([upper - lower for lower, upper in model.bounds])
        for repeat in range(repeats):
            noisy = np.maximum(observed + rng.normal(0.0, sigma), float(perturb_cfg["clip_min"]))
            try:
                result = fit_registered_spectrum(
                    noisy,
                    WAVELENGTHS,
                    model,
                    fit_cfg,
                    n_starts=int(fit_cfg["n_starts_sensitivity"]),
                    global_params={"side_log_gain_by_region": gains},
                    observation_context={"region_side": _region_side(fit_row.region)},
                )
                record = {
                    "sample_id": fit_row.sample_id,
                    "subject_id": fit_row.subject_id,
                    "region": fit_row.region,
                    "model_id": fit_row.model_id,
                    "repeat": repeat,
                    "status": "SUCCESS",
                    "sigma_median": float(np.median(sigma)),
                }
                for index, name in enumerate(model.parameter_names):
                    record[name] = float(result.theta[index])
                    record[f"normalized_shift_{name}"] = float(abs(result.theta[index] - baseline[index]) / spans[index])
                rows.append(record)
            except Exception as exc:
                rows.append(
                    {
                        "sample_id": fit_row.sample_id,
                        "subject_id": fit_row.subject_id,
                        "region": fit_row.region,
                        "model_id": fit_row.model_id,
                        "repeat": repeat,
                        "status": "FAIL",
                        "failure": f"{type(exc).__name__}: {exc}",
                    }
                )
            completed += 1
            if completed % 80 == 0 or completed == total:
                progress(f"S1-5 perturbations: {completed}/{total}")
    return pd.DataFrame(rows)


def _run_profiles(
    symmetric: pd.DataFrame,
    registry: Any,
    config: Mapping[str, Any],
    progress: Callable[[str], None],
) -> pd.DataFrame:
    fit_cfg = config["fit"]
    profile_cfg = config["profile_likelihood"]
    rows: list[dict[str, Any]] = []
    total = len(symmetric) * len(profile_cfg["models"])
    completed = 0
    for source_row in symmetric.itertuples(index=False):
        row = source_row._asdict()
        observed = _spectrum(row)
        for model_id in profile_cfg["models"]:
            model = RegisteredSkinForwardModel(model_id, registry)
            fit = fit_registered_spectrum(
                observed, WAVELENGTHS, model, fit_cfg, n_starts=int(fit_cfg["n_starts_stress"])
            )
            profiles = profile_registered_spectrum(
                observed,
                WAVELENGTHS,
                model,
                fit.theta,
                fit_cfg,
                profile_cfg["grid_fractions"],
            )
            rows.extend({"subject_id": row["subject_id"], **item} for item in profiles)
            completed += 1
            if completed % 12 == 0 or completed == total:
                progress(f"S1-5 profiles: {completed}/{total}")
    return pd.DataFrame(rows)


def _run_diagnostic_conditions(
    diagnostic: pd.DataFrame,
    registry: Any,
    config: Mapping[str, Any],
    gains: Mapping[str, float],
    progress: Callable[[str], None],
) -> pd.DataFrame:
    fit_cfg = config["fit"]
    rows: list[dict[str, Any]] = []
    total = len(diagnostic) * len(config["stress_fit"]["models"])
    completed = 0
    for source_row in diagnostic.sort_values(["subject_id", "sample_id", "region"]).itertuples(index=False):
        row = source_row._asdict()
        observed = _spectrum(row)
        for model_id in config["stress_fit"]["models"]:
            model = RegisteredSkinForwardModel(model_id, registry)
            try:
                fit = fit_registered_spectrum(
                    observed,
                    WAVELENGTHS,
                    model,
                    fit_cfg,
                    n_starts=int(fit_cfg["n_starts_stress"]),
                    global_params={"side_log_gain_by_region": gains},
                    observation_context={"region_side": _region_side(str(row["region"]))},
                )
                record, _, _ = _fit_record(row, model_id, model, fit)
                rows.append(record)
            except Exception as exc:
                rows.append(
                    {
                        "sample_id": row["sample_id"],
                        "subject_id": row["subject_id"],
                        "expression": row["expression"],
                        "direction": row["direction"],
                        "region": row["region"],
                        "analysis_role": row["s1_2_analysis_role"],
                        "model_id": model_id,
                        "fit_status": "FAIL",
                        "failure": f"{type(exc).__name__}: {exc}",
                    }
                )
            completed += 1
            if completed % 80 == 0 or completed == total:
                progress(f"S1-5 diagnostic condition fits: {completed}/{total}")
    return pd.DataFrame(rows)


def _condition_stability(primary_fits: pd.DataFrame, diagnostic_fits: pd.DataFrame, registry: Any) -> pd.DataFrame:
    reference = primary_fits[
        (primary_fits["model_id"].isin(["P2", "P3-S"])) & (primary_fits["fit_status"] == "SUCCESS")
    ]
    ref = reference.set_index(["subject_id", "model_id", "region"])
    rows: list[dict[str, Any]] = []
    for row in diagnostic_fits[diagnostic_fits["fit_status"] == "SUCCESS"].itertuples(index=False):
        key = (row.subject_id, row.model_id, row.region)
        if key not in ref.index:
            continue
        baseline = ref.loc[key]
        record = {
            "subject_id": row.subject_id,
            "model_id": row.model_id,
            "region": row.region,
            "expression": row.expression,
            "direction": row.direction,
            "sample_id": row.sample_id,
        }
        model = registry.model(row.model_id)
        for name in model.flat_parameter_names:
            span = registry.parameter(name).maximum - registry.parameter(name).minimum
            delta = float(getattr(row, name) - baseline[name])
            record[f"delta_{name}"] = delta
            record[f"normalized_abs_delta_{name}"] = abs(delta) / span
        rows.append(record)
    return pd.DataFrame(rows)


def _region_stability(primary_fits: pd.DataFrame, diagnostic_fits: pd.DataFrame, registry: Any) -> pd.DataFrame:
    reference = primary_fits[
        (primary_fits["model_id"].isin(["P2", "P3-S"])) & (primary_fits["fit_status"] == "SUCCESS")
    ]
    reference = reference.groupby(["subject_id", "model_id"], sort=True)[list(PARAMETER_NAMES)].mean(numeric_only=True)
    target = diagnostic_fits[
        (diagnostic_fits["fit_status"] == "SUCCESS")
        & (diagnostic_fits["expression"] == "neutral")
        & (diagnostic_fits["direction"] == "front")
        & diagnostic_fits["region"].isin(["forehead", "whole_skin"])
    ]
    rows: list[dict[str, Any]] = []
    for row in target.itertuples(index=False):
        baseline = reference.loc[(row.subject_id, row.model_id)]
        record = {"subject_id": row.subject_id, "model_id": row.model_id, "region": row.region}
        for name in registry.model(row.model_id).flat_parameter_names:
            span = registry.parameter(name).maximum - registry.parameter(name).minimum
            delta = float(getattr(row, name) - baseline[name])
            record[f"delta_{name}"] = delta
            record[f"normalized_abs_delta_{name}"] = abs(delta) / span
        rows.append(record)
    return pd.DataFrame(rows)


def _sensitivity_summary(frame: pd.DataFrame, registry: Any) -> dict[str, Any]:
    successful = frame[frame["status"] == "SUCCESS"].copy()
    baseline = successful[
        (successful["sensitivity_kind"] == "band_set")
        & (successful["sensitivity_value"] == "full_31")
    ].set_index(["subject_id", "model_id"])
    summary: dict[str, Any] = {
        "status": "descriptive_Train_only",
        "failure_count": int((frame["status"] != "SUCCESS").sum()),
        "cases": {},
    }
    for (model_id, kind, value), group in successful.groupby(
        ["model_id", "sensitivity_kind", "sensitivity_value"], sort=False
    ):
        shifts: dict[str, Any] = {}
        for name in registry.model(model_id).flat_parameter_names:
            span = registry.parameter(name).maximum - registry.parameter(name).minimum
            values = []
            for row in group.itertuples(index=False):
                values.append(abs(float(getattr(row, name)) - float(baseline.loc[(row.subject_id, model_id), name])) / span)
            shifts[name] = {
                "median_normalized_abs_shift": float(np.median(values)),
                "p95_normalized_abs_shift": float(np.quantile(values, 0.95)),
            }
        summary["cases"][f"{model_id}|{kind}|{value}"] = {
            "n": int(len(group)),
            "median_log_rmse": float(group["log_rmse"].median()),
            "median_sam_rad": float(group["sam_rad"].median()),
            "parameter_shift_vs_full_31": shifts,
        }
    return summary


def _identifiability_report(
    fits: pd.DataFrame,
    profiles: pd.DataFrame,
    perturbations: pd.DataFrame,
) -> dict[str, Any]:
    report: dict[str, Any] = {"models": {}, "profile_scope": "44_Train_subject_symmetric_primary_spectra"}
    physical = fits[(fits["fit_status"] == "SUCCESS") & (fits["model_id"] != "B0")]
    for model_id, group in physical.groupby("model_id", sort=False):
        profile_group = profiles[profiles["model_id"] == model_id]
        perturb_group = perturbations[
            (perturbations["model_id"] == model_id) & (perturbations["status"] == "SUCCESS")
        ]
        model_result: dict[str, Any] = {
            "fit_count": int(len(group)),
            "full_jacobian_rank_fraction": float(
                (group["jacobian_rank"] == len([x for x in PARAMETER_NAMES if group[x].notna().any()])).mean()
            ),
            "jacobian_condition_number": {
                "median": float(group["jacobian_condition_number"].median()),
                "p95": float(group["jacobian_condition_number"].quantile(0.95)),
            },
            "multiple_near_optimal_cluster_fraction": float((group["solution_cluster_count"] > 1).mean()),
            "profile": {},
            "perturbation": {},
        }
        for parameter, parameter_group in profile_group.groupby("profiled_parameter", sort=False):
            by_subject = parameter_group.groupby("subject_id").agg(
                delta_range=("delta_objective", lambda values: float(values.max() - values.min())),
                minimizing_fraction=(
                    "grid_fraction",
                    lambda values: float(
                        values.iloc[
                            np.argmin(parameter_group.loc[values.index, "delta_objective"].to_numpy())
                        ]
                    ),
                ),
            )
            model_result["profile"][parameter] = {
                "median_delta_objective_range": float(by_subject["delta_range"].median()),
                "minimum_at_outer_grid_fraction": float(
                    ((by_subject["minimizing_fraction"] <= 0.10) | (by_subject["minimizing_fraction"] >= 0.90)).mean()
                ),
            }
        for name in PARAMETER_NAMES:
            column = f"normalized_shift_{name}"
            if column in perturb_group and perturb_group[column].notna().any():
                model_result["perturbation"][name] = {
                    "median_normalized_abs_shift": float(perturb_group[column].median()),
                    "p95_normalized_abs_shift": float(perturb_group[column].quantile(0.95)),
                }
        report["models"][model_id] = model_result
    return report


def _failure_report(
    primary_fits: pd.DataFrame,
    diagnostic_fits: pd.DataFrame,
    perturbations: pd.DataFrame,
    sensitivities: pd.DataFrame,
    profiles: pd.DataFrame,
) -> dict[str, Any]:
    return {
        "primary_fit_failures": primary_fits[primary_fits["fit_status"] != "SUCCESS"].to_dict("records"),
        "diagnostic_fit_failures": diagnostic_fits[diagnostic_fits["fit_status"] != "SUCCESS"].to_dict("records"),
        "perturbation_failures": perturbations[perturbations["status"] != "SUCCESS"].to_dict("records"),
        "sensitivity_failures": sensitivities[sensitivities["status"] != "SUCCESS"].to_dict("records"),
        "profile_nonconvergence_count": int((~profiles["success"].astype(bool)).sum()),
    }


def _fit_sensitivity_case(
    observed: np.ndarray,
    model: Any,
    wavelength: np.ndarray,
    fit_cfg: Mapping[str, Any],
    n_starts: int,
    selected_mask: np.ndarray | None = None,
) -> Any:
    return fit_registered_spectrum(
        observed,
        wavelength,
        model,
        fit_cfg,
        n_starts=n_starts,
        band_mask=selected_mask,
    )


def _run_sensitivities(
    symmetric_median: pd.DataFrame,
    symmetric_trimmed: pd.DataFrame,
    symmetric_zero_side_gain: pd.DataFrame,
    registry: Any,
    config: Mapping[str, Any],
    progress: Callable[[str], None],
) -> pd.DataFrame:
    fit_cfg = config["fit"]
    sensitivity = config["sensitivity"]
    n_starts = int(fit_cfg["n_starts_sensitivity"])
    rows: list[dict[str, Any]] = []
    cases: list[tuple[str, str, Any]] = []
    for name in sensitivity["band_sets"]:
        cases.append(("band_set", name, name))
    for value in sensitivity["center_shift_nm"]:
        cases.append(("center_shift_nm", str(float(value)), float(value)))
    for value in sensitivity["assumed_gaussian_fwhm_nm"]:
        cases.append(("gaussian_fwhm_nm", str(float(value)), float(value)))
    cases.append(("statistic", "trimmed_mean", "trimmed_mean"))
    cases.append(("side_gain", "zero", "zero"))
    total = len(symmetric_median) * len(sensitivity["models"]) * len(cases)
    completed = 0
    trimmed_index = symmetric_trimmed.set_index("subject_id")
    zero_side_index = symmetric_zero_side_gain.set_index("subject_id")
    for source in symmetric_median.itertuples(index=False):
        observed_median = _spectrum(source._asdict())
        observed_trimmed = _spectrum(trimmed_index.loc[source.subject_id].to_dict())
        observed_zero_side = _spectrum(zero_side_index.loc[source.subject_id].to_dict())
        for model_id in sensitivity["models"]:
            base_model = RegisteredSkinForwardModel(model_id, registry)
            for kind, label, value in cases:
                try:
                    observed = observed_median
                    wavelength = WAVELENGTHS
                    model: Any = base_model
                    selected: np.ndarray | None = None
                    if kind == "band_set":
                        selected = band_mask(WAVELENGTHS, str(value), registry)
                    elif kind == "center_shift_nm":
                        interior = band_mask(WAVELENGTHS, sensitivity["center_shift_band_set"], registry)
                        observed = observed_median[interior]
                        wavelength = shifted_centers(WAVELENGTHS[interior], float(value), registry)
                    elif kind == "gaussian_fwhm_nm" and float(value) > 0:
                        model = GaussianSRFModelAdapter(base_model, WAVELENGTHS, float(value))
                    elif kind == "statistic":
                        observed = observed_trimmed
                    elif kind == "side_gain":
                        observed = observed_zero_side
                    fit = _fit_sensitivity_case(observed, model, wavelength, fit_cfg, n_starts, selected)
                    record = {
                        "subject_id": source.subject_id,
                        "model_id": model_id,
                        "sensitivity_kind": kind,
                        "sensitivity_value": label,
                        "status": "SUCCESS",
                        "log_rmse": float(fit.metrics["log_rmse"]),
                        "sam_rad": float(fit.metrics["sam_rad"]),
                    }
                    for index, name in enumerate(base_model.parameter_names):
                        record[name] = float(fit.theta[index])
                    rows.append(record)
                except Exception as exc:
                    rows.append(
                        {
                            "subject_id": source.subject_id,
                            "model_id": model_id,
                            "sensitivity_kind": kind,
                            "sensitivity_value": label,
                            "status": "FAIL",
                            "failure": f"{type(exc).__name__}: {exc}",
                        }
                    )
                completed += 1
                if completed % 100 == 0 or completed == total:
                    progress(f"S1-5 spectral sensitivities: {completed}/{total}")
    return pd.DataFrame(rows)


def _threshold_draft(fits: pd.DataFrame, perturbations: pd.DataFrame, profiles: pd.DataFrame) -> dict[str, Any]:
    result: dict[str, Any] = {
        "status": "train_reference_distributions_only_not_frozen_gates",
        "warning": "S1-6 Validation must choose and freeze thresholds; these Train quantiles are not Test gates.",
        "models": {},
    }
    for model_id, group in fits[fits["fit_status"] == "SUCCESS"].groupby("model_id", sort=False):
        result["models"][model_id] = {
            metric: {
                "median": float(group[metric].median()),
                "p90": float(group[metric].quantile(0.90)),
                "p95": float(group[metric].quantile(0.95)),
            }
            for metric in ("log_rmse", "sam_rad")
        }
    successful_perturb = perturbations[perturbations["status"] == "SUCCESS"]
    result["perturbation"] = {}
    for model_id, group in successful_perturb.groupby("model_id", sort=False):
        columns = [column for column in group if column.startswith("normalized_shift_")]
        result["perturbation"][model_id] = {
            column: {"median": float(group[column].median()), "p95": float(group[column].quantile(0.95))}
            for column in columns
            if group[column].notna().any()
        }
    result["profiles"] = {
        "row_count": int(len(profiles)),
        "delta_objective_is_descriptive_only": True,
    }
    return result


def _report(
    decision: Mapping[str, Any],
    diagnostics: Mapping[str, Any],
    hb_trigger: Mapping[str, Any],
) -> str:
    lines = [
        "# S1-5 Train-only 反演与诊断报告",
        "",
        f"- 状态：`{decision['status']}`",
        f"- 建议：`{decision['recommendation']}`",
        f"- 下一阶段授权：`{decision['authorized_next_stage']}`",
        f"- Train 主域光谱：{decision['primary_spectrum_count']} 条 / {decision['primary_subject_count']} 名受试者",
        "- Validation/Test/原始 HSI 内容访问计数：0/0/0",
        "",
        "## 主模型摘要",
        "",
        "| 模型 | median log-RMSE | median SAM | 任一边界比例 | 多解聚类比例 | 最大逐波段中位绝对 log 残差 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for model_id, item in diagnostics["models"].items():
        lines.append(
            f"| `{model_id}` | {item['median_log_rmse']:.6f} | {item['median_sam_rad']:.6f} | "
            f"{item['any_boundary_fraction']:.3f} | {item['multi_cluster_fraction']:.3f} | "
            f"{item['max_abs_median_log_residual_across_bands']:.6f} |"
        )
    lines.extend(
        [
            "",
        "## 机制诊断",
        "",
            f"- 最佳物理候选相对 B0 的 median log-RMSE 比值：{diagnostics['best_physical_to_B0_median_log_rmse_ratio']:.3f}（低于 1 才优于 B0）。",
            f"- P3-S 相对 P2 的受试者中位 log-RMSE 改善：{diagnostics['models']['P3-S']['median_relative_log_rmse_improvement_vs_P2']:.3%}。",
            f"- P3-O Train 残差触发：`{hb_trigger['P3-O_activated']}`；去趋势残差与 sO2 对比的中位相关为 {hb_trigger['median_correlation']:.4f}，一致受试者比例为 {hb_trigger['consistent_subject_fraction']:.3f}。",
            f"- 开发期 review flags：`{json.dumps(diagnostics['review_flags'], ensure_ascii=False)}`。",
            "",
            "S1-5 的数值阈值只是 Train 参考分布与预定义 review flag，不是正式 Test gate。若状态为 REVISE，必须先修改模型结构或参数合同并重新从新版本运行；不得查看 Validation/Test 来修补 Train 失败。",
            "",
        ]
    )
    return "\n".join(lines)


def run_s1_5_train_development(
    config_path: str | Path = DEFAULT_S1_5_CONFIG,
    output_dir: str | Path | None = None,
    *,
    progress: Callable[[str], None] | None = None,
    supersedes_decision_path: str | Path | None = None,
) -> dict[str, Any]:
    """Run the complete S1-5 development protocol using Train rows only."""

    notify = progress or (lambda _: None)
    path = Path(config_path).resolve()
    with path.open("r", encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    if not isinstance(config, dict):
        raise ValueError("S1-5 config must be a mapping")
    project_root = PROJECT_ROOT
    inputs = {name: _resolve(project_root, value) for name, value in config["inputs"].items()}
    destination = (
        Path(output_dir).resolve()
        if output_dir is not None
        else (project_root / "data" / "processed" / "HyperSkin_Stage1_v1" / "train_development" / "s1_5_v1")
    )
    supersedes_path = Path(supersedes_decision_path).resolve() if supersedes_decision_path is not None else None
    if supersedes_path is not None and not supersedes_path.is_file():
        raise FileNotFoundError(f"Missing superseded S1-5 decision: {supersedes_path}")
    _new_output(destination)

    s1_3 = _read_json(inputs["s1_3_decision"])
    s1_4 = _read_json(inputs["s1_4_decision"])
    if s1_3.get("status") != "PASS_FOR_S1_4" or int(s1_3.get("test_access_count", -1)) != 0:
        raise RuntimeError("S1-3 gate failed")
    if s1_4.get("status") != "PASS_FOR_S1_5" or s1_4.get("authorized_next_stage") != "S1-5":
        raise RuntimeError("S1-4 did not authorize S1-5")
    if int(s1_4.get("test_access_count", -1)) != 0:
        raise RuntimeError("S1-4 Test isolation failed")
    if _sha256(inputs["model_registry"]) != s1_4["source_registry_sha256"]:
        raise RuntimeError("S1-4 model registry hash mismatch")
    s1_3_source = _read_json(Path(s1_3["source_decision_path"]))
    if _sha256(Path(s1_3["source_decision_path"])) != s1_3["source_decision_sha256"]:
        raise RuntimeError("S1-3 source decision hash mismatch")
    expected_region_hash = s1_3_source["outputs"]["region_spectra_parquet"]["sha256"]
    if _sha256(inputs["region_spectra"]) != expected_region_hash:
        raise RuntimeError("S1-3 region_spectra hash mismatch")
    if _sha256(inputs["band_reliability"]) != s1_3["band_reliability_summary_sha256"]:
        raise RuntimeError("S1-3 band reliability hash mismatch")

    registry = load_model_registry(inputs["model_registry"])
    frame = pd.read_parquet(inputs["region_spectra"], filters=[("split", "==", "train")])
    if frame.empty or set(frame["split"]) != {"train"}:
        raise RuntimeError("Filtered S1-5 input must contain Train rows only")
    roles = config["data_roles"]
    usable = frame[frame["extraction_status"] == roles["usable_status"]].copy()
    primary_mask = (
        (usable["s1_2_analysis_role"] == roles["primary_analysis_role"])
        & (usable["expression"] == roles["primary_expression"])
        & (usable["direction"] == roles["primary_direction"])
        & usable["region"].isin(roles["primary_regions"])
    )
    primary = usable[primary_mask].copy()
    if len(primary) != 88 or primary["subject_id"].nunique() != 44:
        raise RuntimeError("Expected 88 paired primary cheek spectra from 44 Train subjects")
    diagnostic = usable[~primary_mask].copy()
    notify(f"S1-5 inputs: primary={len(primary)}, diagnostic={len(diagnostic)}, Train-only")
    implementation_paths = {
        "s1_train_development.py": Path(__file__).resolve(),
        "s1_inverse.py": (Path(__file__).parent / "s1_inverse.py").resolve(),
        "skin_forward.py": (Path(__file__).parent / "skin_forward.py").resolve(),
        "model_registry.py": (Path(__file__).parent / "model_registry.py").resolve(),
        "s1_spectral_sensitivity.py": (Path(__file__).parent / "s1_spectral_sensitivity.py").resolve(),
    }
    _write_json(
        destination / "input_audit.json",
        {
            "config_path": str(path),
            "config_sha256": _sha256(path),
            "inputs": {name: {"path": str(value), "sha256": _sha256(value)} for name, value in inputs.items()},
            "implementation": {
                name: {"path": str(value), "sha256": _sha256(value)} for name, value in implementation_paths.items()
            },
            "environment": {
                "python": platform.python_version(),
                "numpy": np.__version__,
                "pandas": pd.__version__,
                "scipy": scipy.__version__,
                "torch": torch.__version__,
            },
            "filtered_split": "train",
            "train_row_count": int(len(frame)),
            "usable_train_row_count": int(len(usable)),
            "primary_row_count": int(len(primary)),
            "primary_subject_count": int(primary["subject_id"].nunique()),
            "diagnostic_row_count": int(len(diagnostic)),
            "validation_row_access_count": 0,
            "test_row_access_count": 0,
            "raw_hsi_content_access_count": 0,
        },
    )

    side = _estimate_side_calibration(primary)
    primary_fits, starts, residuals = _fit_primary(primary, registry, config, side, destination, notify)
    primary_fits.to_parquet(destination / "primary_model_fits.parquet", index=False)
    primary_fits.to_csv(destination / "primary_model_fits.csv", index=False)
    starts.to_parquet(destination / "multistart_solutions.parquet", index=False)
    residuals.to_parquet(destination / "wavelength_residuals.parquet", index=False)
    residuals.to_csv(destination / "wavelength_residuals.csv", index=False)

    comparison = _primary_subject_comparison(primary_fits)
    comparison.to_parquet(destination / "subject_model_comparison.parquet", index=False)
    comparison.to_csv(destination / "subject_model_comparison.csv", index=False)
    hb = _hb_trigger(residuals, registry, config)
    diagnostics = _diagnostic_summary(primary_fits, residuals, comparison, registry, config)
    p4_cfg = config["models"]["p4_activation"]
    hb["P4_activated"] = bool(hb["P3-O_activated"] and diagnostics["scattering_residual_evidence"])
    hb["P4_rule"] = p4_cfg
    _write_json(destination / "hb_shape_trigger.json", hb)
    _write_json(destination / "primary_diagnostic_summary.json", diagnostics)

    symmetric_median = _symmetric_primary(primary, side["side_log_gain_by_region"], "median")
    symmetric_trimmed = _symmetric_primary(primary, side["side_log_gain_by_region"], "trimmed_mean")
    symmetric_zero_side = _symmetric_primary(
        primary, {"image_left": 0.0, "image_right": 0.0, "other": 0.0}, "median"
    )
    perturbations = _run_perturbations(primary, primary_fits, registry, config, side["side_log_gain_by_region"], notify)
    perturbations.to_parquet(destination / "perturbation_stability.parquet", index=False)
    profiles = _run_profiles(symmetric_median, registry, config, notify)
    profiles.to_parquet(destination / "profile_likelihood.parquet", index=False)
    diagnostic_fits = _run_diagnostic_conditions(
        diagnostic, registry, config, side["side_log_gain_by_region"], notify
    )
    diagnostic_fits.to_parquet(destination / "diagnostic_condition_fits.parquet", index=False)
    stability = _condition_stability(primary_fits, diagnostic_fits, registry)
    stability.to_parquet(destination / "condition_stability.parquet", index=False)
    region_stability = _region_stability(primary_fits, diagnostic_fits, registry)
    region_stability.to_parquet(destination / "region_stability.parquet", index=False)
    sensitivities = _run_sensitivities(
        symmetric_median, symmetric_trimmed, symmetric_zero_side, registry, config, notify
    )
    sensitivities.to_parquet(destination / "spectral_sensitivity.parquet", index=False)
    sensitivity_summary = _sensitivity_summary(sensitivities, registry)
    _write_json(destination / "spectral_sensitivity_summary.json", sensitivity_summary)

    identifiability = _identifiability_report(primary_fits, profiles, perturbations)
    _write_json(destination / "identifiability_report.json", identifiability)
    failures = _failure_report(primary_fits, diagnostic_fits, perturbations, sensitivities, profiles)
    _write_json(destination / "fit_failures.json", failures)

    threshold_draft = _threshold_draft(primary_fits, perturbations, profiles)
    _write_json(destination / "train_threshold_draft.json", threshold_draft)
    review_flags = diagnostics["review_flags"]
    primary_failures = int((primary_fits["fit_status"] != "SUCCESS").sum())
    if primary_failures == len(primary_fits):
        recommendation = "STOP"
        status = "STOP"
        next_allowed = False
    elif any(bool(value) for value in review_flags.values()):
        recommendation = "REVISE_BEFORE_S1_6"
        status = "S1_5_COMPLETE_REVISE_BEFORE_S1_6"
        next_allowed = False
    else:
        recommendation = "READY_FOR_S1_6"
        status = "PASS_FOR_S1_6"
        next_allowed = True

    output_names = [
        "input_audit.json",
        "train_global_calibration.json",
        "primary_model_fits.parquet",
        "primary_model_fits.csv",
        "multistart_solutions.parquet",
        "wavelength_residuals.parquet",
        "wavelength_residuals.csv",
        "subject_model_comparison.parquet",
        "subject_model_comparison.csv",
        "hb_shape_trigger.json",
        "primary_diagnostic_summary.json",
        "perturbation_stability.parquet",
        "profile_likelihood.parquet",
        "diagnostic_condition_fits.parquet",
        "condition_stability.parquet",
        "region_stability.parquet",
        "spectral_sensitivity.parquet",
        "spectral_sensitivity_summary.json",
        "identifiability_report.json",
        "fit_failures.json",
        "train_threshold_draft.json",
    ]
    decision: dict[str, Any] = {
        "schema_version": 1,
        "stage": "S1-5",
        "status": status,
        "recommendation": recommendation,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "supersedes": (
            {
                "decision_path": str(supersedes_path),
                "decision_sha256": _sha256(supersedes_path),
                "reason": "complete_zero-side_gain_sensitivity_and_persist_identifiability_failure_and_code_provenance",
            }
            if supersedes_path is not None
            else None
        ),
        "primary_spectrum_count": int(len(primary)),
        "primary_subject_count": int(primary["subject_id"].nunique()),
        "diagnostic_spectrum_count": int(len(diagnostic)),
        "primary_fit_failure_count": primary_failures,
        "primary_models": list(config["models"]["primary"]),
        "P3-O_activated": bool(hb["P3-O_activated"]),
        "P4_activated": bool(hb["P4_activated"]),
        "review_flags": review_flags,
        "effective_srf_status": "missing",
        "validation_row_access_count": 0,
        "test_access_count": 0,
        "raw_hsi_content_access_count": 0,
        "outputs": {
            name: {"path": str(destination / name), "sha256": _sha256(destination / name)} for name in output_names
        },
        "next_stage_allowed": next_allowed,
        "authorized_next_stage": "S1-6" if next_allowed else None,
    }
    report_path = destination / "S1_5_TRAIN_REPORT.md"
    report_path.write_text(_report(decision, diagnostics, hb), encoding="utf-8", newline="\n")
    decision["outputs"][report_path.name] = {"path": str(report_path), "sha256": _sha256(report_path)}
    _write_json(destination / "s1_5_decision.json", decision)
    _write_json(destination / "progress.json", {"phase": "complete", "status": status})
    notify(f"S1-5 complete: {status}")
    return decision
