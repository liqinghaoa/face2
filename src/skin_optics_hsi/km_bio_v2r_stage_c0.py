"""Train-only candidate ladder for KM-BIO-v2R R-C0."""

from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml
from scipy.optimize import least_squares

from .km_bio_inverse import FitSettings, acceptable_intervals, fit_bounded_spectrum, profile_parameter, spectral_metrics
from .km_bio_observation import sha256_file
from .km_bio_v2r import (
    AS_BOUNDS,
    DELTA_BS_BOUNDS,
    G0_BOUNDS,
    estimate_global_gain,
    forward_preloaded_numpy,
    load_optical_numpy,
    profile_identifiability,
)


FIT_LO = np.array([0.0, 0.0], dtype=np.float64)
FIT_HI = np.array([0.43, 0.10], dtype=np.float64)


def _resolve(root: Path, value: str) -> Path:
    path = Path(value)
    return (root / path).resolve() if not path.is_absolute() else path.resolve()


def _jsonable(value: Any) -> Any:
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    return value


def _write_csv(frame: pd.DataFrame, path: Path) -> None:
    export = frame.copy()
    for column in export.columns:
        if export[column].dtype == object:
            export[column] = export[column].map(
                lambda value: json.dumps(_jsonable(value), ensure_ascii=False, separators=(",", ":"))
                if isinstance(value, (list, tuple, dict, np.ndarray)) else value
            )
    export.to_csv(path, index=False, encoding="utf-8-sig")


def _settings(config: dict[str, Any]) -> FitSettings:
    raw = config["solver"]["individual"]
    return FitSettings(
        epsilon=float(config["analysis"]["epsilon"]),
        sobol_starts=int(raw["sobol_starts"]),
        seed=int(raw["random_seed"]),
        ftol=float(raw["ftol"]),
        xtol=float(raw["xtol"]),
        gtol=float(raw["gtol"]),
        max_nfev=int(raw["max_nfev_per_start"]),
    )


def validate_fold_assignments(subjects: list[str], assignments: dict[Any, list[str]]) -> pd.DataFrame:
    rows = [
        {"subject_id": str(subject), "outer_fold": int(fold)}
        for fold, values in assignments.items() for subject in values
    ]
    frame = pd.DataFrame(rows).sort_values(["outer_fold", "subject_id"]).reset_index(drop=True)
    if frame["subject_id"].duplicated().any():
        raise ValueError("A subject occurs in more than one frozen outer fold")
    if set(frame["subject_id"]) != set(map(str, subjects)) or len(frame) != len(subjects):
        raise ValueError("Frozen outer folds do not exactly cover the R-B subjects")
    return frame


def _spectral_gate(summary: dict[str, float], gates: dict[str, Any]) -> dict[str, bool]:
    return {
        "median_logrmse": summary["median_logrmse"] <= float(gates["subject_median_logrmse_max"]),
        "p90_logrmse": summary["p90_logrmse"] <= float(gates["subject_p90_logrmse_max"]),
        "median_rmse": summary["median_rmse"] <= float(gates["subject_median_rmse_max"]),
        "median_sam_deg": summary["median_sam_deg"] <= float(gates["subject_median_sam_deg_max"]),
        "maximum_abs_median_signed_band_residual": summary["maximum_abs_median_signed_band_residual"] <= float(gates["maximum_abs_median_signed_band_residual"]),
        "reference_better_fraction": summary["model_better_than_reference_fraction"] >= float(gates["outer_fold_reference_better_fraction_min"]),
        "model_to_reference_median_error_ratio": summary["median_model_to_reference_error_ratio"] <= float(gates["model_to_reference_median_error_ratio_max"]),
    }


def _summarize_candidate(metrics: pd.DataFrame, residuals: pd.DataFrame) -> dict[str, Any]:
    median_band = residuals.groupby("wavelength_nm", sort=True)["signed_residual"].median()
    return {
        "subject_count": int(len(metrics)),
        "all_solver_converged": bool(metrics["solver_converged"].all()),
        "median_logrmse": float(metrics["logrmse"].median()),
        "p90_logrmse": float(metrics["logrmse"].quantile(0.90)),
        "median_rmse": float(metrics["rmse"].median()),
        "median_sam_deg": float(metrics["sam_deg"].median()),
        "median_centered_logrmse": float(metrics["centered_logrmse"].median()),
        "absolute_global_median_subject_mean_log_residual": float(abs(metrics["mean_log_residual"].median())),
        "maximum_abs_median_signed_band_residual": float(median_band.abs().max()),
        "model_better_than_reference_fraction": float(metrics["model_better_than_reference"].mean()),
        "median_model_to_reference_error_ratio": float(metrics["model_to_reference_error_ratio"].median()),
        "f_mel_lower_boundary_fraction": float(metrics["f_mel_at_lower"].mean()),
        "f_mel_upper_boundary_fraction": float(metrics["f_mel_at_upper"].mean()),
        "f_mel_any_boundary_fraction": float((metrics["f_mel_at_lower"] | metrics["f_mel_at_upper"]).mean()),
        "f_blood_upper_boundary_fraction": float(metrics["f_blood_at_upper"].mean()),
        "median_signed_residual_by_wavelength": {str(int(k)): float(v) for k, v in median_band.items()},
    }


def _residual_direction_score(candidate: str, summary: dict[str, Any], config: dict[str, Any]) -> float:
    rule = config["upgrade_gates"]["residual_direction"][candidate]
    if candidate == "V2R-P":
        band = summary["median_signed_residual_by_wavelength"]
        soret = np.mean([band[str(int(w))] for w in rule["soret_nm"]])
        q_band = np.mean([band[str(int(w))] for w in rule["q_nm"]])
        return float(abs(soret - q_band))
    if candidate == "V2R-PS":
        return float(summary["median_centered_logrmse"])
    if candidate == "V2R-PSG":
        return float(summary["absolute_global_median_subject_mean_log_residual"])
    raise KeyError(candidate)


def compare_candidates(
    previous_name: str,
    current_name: str,
    previous: pd.DataFrame,
    current: pd.DataFrame,
    previous_summary: dict[str, Any],
    current_summary: dict[str, Any],
    config: dict[str, Any],
) -> dict[str, Any]:
    joined = previous[["subject_id", "logrmse"]].merge(
        current[["subject_id", "logrmse"]], on="subject_id", suffixes=("_previous", "_current"), validate="one_to_one"
    )
    gates = config["upgrade_gates"]
    reduction = (previous_summary["median_logrmse"] - current_summary["median_logrmse"]) / previous_summary["median_logrmse"]
    improved = float((joined["logrmse_current"] < joined["logrmse_previous"]).mean())
    previous_score = _residual_direction_score(current_name, previous_summary, config)
    current_score = _residual_direction_score(current_name, current_summary, config)
    residual_reduction = (previous_score - current_score) / previous_score if previous_score > 0 else (1.0 if current_score < previous_score else 0.0)
    residual_rule = gates["residual_direction"][current_name]
    residual_pass = residual_reduction >= float(residual_rule["relative_reduction_min"])
    if residual_rule.get("strict_improvement", False):
        residual_pass = residual_pass and current_score < previous_score
    tolerance = int(gates["boundary_nonworsening_subject_tolerance"]) / len(current)
    checks = {
        "all_final_spectral_gates": bool(all(current_summary["spectral_gate_checks"].values())),
        "median_logrmse_relative_reduction": bool(reduction >= float(gates["median_logrmse_relative_reduction_min"])),
        "improved_subject_fraction": bool(improved >= float(gates["improved_subject_fraction_min"])),
        "f_blood_upper_boundary_absolute": bool(current_summary["f_blood_upper_boundary_fraction"] <= float(gates["f_blood_upper_boundary_fraction_max"])),
        "f_mel_any_boundary_absolute": bool(current_summary["f_mel_any_boundary_fraction"] <= float(gates["f_mel_any_boundary_fraction_max"])),
        "f_blood_boundary_nonworsening": bool(current_summary["f_blood_upper_boundary_fraction"] <= previous_summary["f_blood_upper_boundary_fraction"] + tolerance),
        "f_mel_boundary_nonworsening": bool(current_summary["f_mel_any_boundary_fraction"] <= previous_summary["f_mel_any_boundary_fraction"] + tolerance),
        "candidate_specific_residual_direction": bool(residual_pass),
    }
    return {
        "previous_candidate": previous_name,
        "candidate": current_name,
        "median_logrmse_relative_reduction": float(reduction),
        "improved_subject_fraction": improved,
        "previous_residual_direction_score": previous_score,
        "candidate_residual_direction_score": current_score,
        "residual_direction_relative_reduction": float(residual_reduction),
        "checks": checks,
        "cheap_upgrade_gate_pass": bool(all(checks.values())),
    }


def _candidate_kwargs(candidate: str, fold_globals: dict[int, dict[str, float]] | None, fold: int) -> dict[str, float]:
    if candidate == "V2R-0":
        return {"diameter_um": 0.0, "scattering_amplitude": 1.0, "delta_bs": 0.0, "g0": 1.0}
    if candidate == "V2R-P":
        return {"diameter_um": 15.0, "scattering_amplitude": 1.0, "delta_bs": 0.0, "g0": 1.0}
    values = fold_globals[fold] if fold_globals is not None else {}
    return {
        "diameter_um": 15.0,
        "scattering_amplitude": float(values["A_s"]),
        "delta_bs": float(values["delta_bs"]),
        "g0": float(values.get("g0", 1.0)),
    }


def _fit_candidate(
    candidate: str,
    manifest: pd.DataFrame,
    fold_manifest: pd.DataFrame,
    wavelength: np.ndarray,
    fit_mask: np.ndarray,
    optical_full: dict[str, np.ndarray],
    config: dict[str, Any],
    fold_globals: dict[int, dict[str, float]] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    observed_columns = [f"observed_reflectance_{int(w)}nm" for w in wavelength]
    base_settings = _settings(config)
    metrics_rows, prediction_rows, residual_rows, reference_rows = [], [], [], []
    assigned = manifest.merge(fold_manifest, on="subject_id", validate="one_to_one")
    for fold in sorted(assigned["outer_fold"].unique()):
        training = assigned.loc[assigned["outer_fold"].ne(fold)]
        held = assigned.loc[assigned["outer_fold"].eq(fold)]
        train_observed = training[observed_columns].to_numpy(dtype=np.float64)[:, fit_mask]
        epsilon = base_settings.epsilon
        reference = np.exp(np.mean(np.log(train_observed + epsilon), axis=0)) - epsilon
        kwargs = _candidate_kwargs(candidate, fold_globals, int(fold))
        for local_index, (_, item) in enumerate(held.iterrows()):
            observed_full = item[observed_columns].to_numpy(dtype=np.float64)
            observed = observed_full[fit_mask]
            optical_fit = {name: values[fit_mask] for name, values in optical_full.items()}
            fit_wavelength = wavelength[fit_mask]

            def forward_fit(theta: np.ndarray) -> np.ndarray:
                return forward_preloaded_numpy(theta, optical_fit, wavelength_nm=fit_wavelength, s0=float(config["parameters"]["fixed_s0"]), **kwargs)

            seed = base_settings.seed + int(fold) * 1000 + local_index + 10000 * list(config["candidates"]["order"]).index(candidate)
            fit = fit_bounded_spectrum(observed, forward_fit, FIT_LO, FIT_HI, replace(base_settings, seed=seed))
            if not fit["success"]:
                raise RuntimeError(f"{candidate} failed to converge for {item['subject_id']}")
            prediction_full = forward_preloaded_numpy(
                fit["theta"], optical_full, wavelength_nm=wavelength,
                s0=float(config["parameters"]["fixed_s0"]), **kwargs,
            )
            prediction = prediction_full[fit_mask]
            metrics = spectral_metrics(prediction, observed, epsilon)
            log_residual = np.log(prediction + epsilon) - np.log(observed + epsilon)
            centered = log_residual - log_residual.mean()
            ref_metrics = spectral_metrics(reference, observed, epsilon)
            u = (fit["theta"] - FIT_LO) / (FIT_HI - FIT_LO)
            row = {
                "candidate": candidate, "subject_id": str(item["subject_id"]), "capture_id": str(item["capture_id"]), "outer_fold": int(fold),
                "solver_converged": True, "selected_start_index": int(fit["selected_start_index"]),
                "f_mel": float(fit["theta"][0]), "f_blood": float(fit["theta"][1]), "c_tHb_eq_g_l": float(150.0 * fit["theta"][1]),
                **metrics,
                "centered_logrmse": float(np.sqrt(np.mean(centered**2))),
                "mean_log_residual": float(log_residual.mean()),
                "reference_logrmse": float(ref_metrics["logrmse"]),
                "model_to_reference_error_ratio": float(metrics["logrmse"] / ref_metrics["logrmse"]),
                "model_better_than_reference": bool(metrics["logrmse"] < ref_metrics["logrmse"]),
                "f_mel_at_lower": bool(u[0] <= float(config["parameters"]["boundary_tolerance_normalized"])),
                "f_mel_at_upper": bool(u[0] >= 1.0 - float(config["parameters"]["boundary_tolerance_normalized"])),
                "f_blood_at_lower": bool(u[1] <= float(config["parameters"]["boundary_tolerance_normalized"])),
                "f_blood_at_upper": bool(u[1] >= 1.0 - float(config["parameters"]["boundary_tolerance_normalized"])),
                **kwargs,
            }
            metrics_rows.append(row)
            reference_rows.append({
                "candidate": candidate, "subject_id": str(item["subject_id"]), "outer_fold": int(fold),
                "reference_logrmse": float(ref_metrics["logrmse"]), "model_logrmse": float(metrics["logrmse"]),
                "error_ratio": row["model_to_reference_error_ratio"], "model_better": row["model_better_than_reference"],
            })
            for band_index, wave in enumerate(wavelength):
                prediction_rows.append({"candidate": candidate, "subject_id": str(item["subject_id"]), "outer_fold": int(fold), "wavelength_nm": int(wave), "observed_reflectance": float(observed_full[band_index]), "predicted_reflectance": float(prediction_full[band_index]), "band_role": "fit" if fit_mask[band_index] else "edge_diagnostic"})
                residual_rows.append({"candidate": candidate, "subject_id": str(item["subject_id"]), "outer_fold": int(fold), "wavelength_nm": int(wave), "signed_residual": float(prediction_full[band_index] - observed_full[band_index]), "log_residual": float(np.log(prediction_full[band_index] + epsilon) - np.log(observed_full[band_index] + epsilon)), "band_role": "fit" if fit_mask[band_index] else "edge_diagnostic"})
    return pd.DataFrame(metrics_rows), pd.DataFrame(prediction_rows), pd.DataFrame(residual_rows), pd.DataFrame(reference_rows)


def _joint_shape_fit(
    observed: np.ndarray,
    optical: dict[str, np.ndarray],
    wavelength: np.ndarray,
    initial_theta: np.ndarray,
    config: dict[str, Any],
) -> dict[str, Any]:
    raw = config["solver"]["global_shape"]
    epsilon = float(config["analysis"]["epsilon"])
    n = len(observed)

    def unpack(x: np.ndarray) -> tuple[float, float, np.ndarray]:
        amplitude = AS_BOUNDS[0] + x[0] * (AS_BOUNDS[1] - AS_BOUNDS[0])
        delta = DELTA_BS_BOUNDS[0] + x[1] * (DELTA_BS_BOUNDS[1] - DELTA_BS_BOUNDS[0])
        theta = FIT_LO + x[2:].reshape(n, 2) * (FIT_HI - FIT_LO)
        return float(amplitude), float(delta), theta

    def residual(x: np.ndarray) -> np.ndarray:
        amplitude, delta, theta = unpack(x)
        predicted = np.asarray([
            forward_preloaded_numpy(t, optical, wavelength_nm=wavelength, s0=float(config["parameters"]["fixed_s0"]), diameter_um=15.0, scattering_amplitude=amplitude, delta_bs=delta, g0=1.0)
            for t in theta
        ])
        values = np.log(predicted + epsilon) - np.log(observed + epsilon)
        return (values - values.mean(axis=1, keepdims=True)).ravel()

    theta_u = (initial_theta - FIT_LO) / (FIT_HI - FIT_LO)
    candidates = []
    for global_u in raw["start_global_u"]:
        start = np.concatenate([np.asarray(global_u, dtype=np.float64), theta_u.ravel()])
        result = least_squares(
            residual, start, bounds=(np.zeros_like(start), np.ones_like(start)), method="trf", loss="linear",
            ftol=float(raw["ftol"]), xtol=float(raw["xtol"]), gtol=float(raw["gtol"]), max_nfev=int(raw["max_nfev_per_start"]),
        )
        loss = float(np.sqrt(np.mean(residual(result.x) ** 2)))
        if result.success and np.isfinite(loss):
            candidates.append((loss, result))
    if not candidates:
        raise RuntimeError("No valid Train-global shape solution")
    loss, best = min(candidates, key=lambda value: value[0])
    amplitude, delta, theta = unpack(best.x)

    def conditional_profile(parameter_index: int, parameter: str, bounds: tuple[float, float]) -> list[dict[str, Any]]:
        grid = np.linspace(0.0, 1.0, int(raw["profile_grid_points"]), dtype=np.float64)
        if not np.any(np.isclose(grid, best.x[parameter_index], atol=1e-14, rtol=0.0)):
            grid = np.sort(np.append(grid, best.x[parameter_index]))
        rows = []
        remaining = np.asarray([i for i in range(len(best.x)) if i != parameter_index])
        for fixed_u in grid:
            start_reduced = best.x[remaining]

            def reduced_residual(z: np.ndarray) -> np.ndarray:
                full = best.x.copy()
                full[parameter_index] = fixed_u
                full[remaining] = z
                return residual(full)

            result = least_squares(
                reduced_residual, start_reduced, bounds=(np.zeros_like(start_reduced), np.ones_like(start_reduced)),
                method="trf", loss="linear", ftol=float(raw["ftol"]), xtol=float(raw["xtol"]), gtol=float(raw["gtol"]),
                max_nfev=int(raw["profile_conditional_max_nfev"]),
            )
            value = bounds[0] + float(fixed_u) * (bounds[1] - bounds[0])
            rows.append({parameter: value, "centered_logrmse": float(np.sqrt(np.mean(reduced_residual(result.x) ** 2))), "success": bool(result.success)})
        return rows

    a_profile = conditional_profile(0, "A_s", AS_BOUNDS)
    d_profile = conditional_profile(1, "delta_bs", DELTA_BS_BOUNDS)
    return {"A_s": amplitude, "delta_bs": delta, "theta": theta, "centered_logrmse": loss, "A_s_profile": a_profile, "delta_bs_profile": d_profile}


def _estimate_ps_globals(
    manifest: pd.DataFrame,
    fold_manifest: pd.DataFrame,
    wavelength: np.ndarray,
    fit_mask: np.ndarray,
    optical: dict[str, np.ndarray],
    p_metrics: pd.DataFrame,
    config: dict[str, Any],
) -> tuple[dict[int, dict[str, float]], pd.DataFrame, pd.DataFrame, dict[int, np.ndarray]]:
    observed_columns = [f"observed_reflectance_{int(w)}nm" for w in wavelength]
    assigned = manifest.merge(fold_manifest, on="subject_id", validate="one_to_one")
    optical_fit = {name: values[fit_mask] for name, values in optical.items()}
    fold_globals, global_rows, profile_rows, train_theta = {}, [], [], {}
    for fold in sorted(assigned["outer_fold"].unique()):
        training = assigned.loc[assigned["outer_fold"].ne(fold)].copy()
        initial = training[["subject_id"]].merge(p_metrics[["subject_id", "f_mel", "f_blood"]], on="subject_id", validate="one_to_one")[["f_mel", "f_blood"]].to_numpy(dtype=np.float64)
        fit = _joint_shape_fit(training[observed_columns].to_numpy(dtype=np.float64)[:, fit_mask], optical_fit, wavelength[fit_mask], initial, config)
        a_ident = profile_identifiability(fit["A_s_profile"], "A_s", "centered_logrmse", AS_BOUNDS)
        d_ident = profile_identifiability(fit["delta_bs_profile"], "delta_bs", "centered_logrmse", DELTA_BS_BOUNDS)
        fold_globals[int(fold)] = {"A_s": fit["A_s"], "delta_bs": fit["delta_bs"], "g0": 1.0}
        train_theta[int(fold)] = fit["theta"]
        tol = float(config["parameters"]["boundary_tolerance_normalized"])
        global_rows.append({
            "candidate": "V2R-PS", "outer_fold": int(fold), "A_s": fit["A_s"], "delta_bs": fit["delta_bs"], "g0": 1.0,
            "training_centered_logrmse": fit["centered_logrmse"],
            "A_s_boundary": bool((fit["A_s"] - AS_BOUNDS[0]) / (AS_BOUNDS[1] - AS_BOUNDS[0]) <= tol or (fit["A_s"] - AS_BOUNDS[0]) / (AS_BOUNDS[1] - AS_BOUNDS[0]) >= 1 - tol),
            "delta_bs_boundary": bool((fit["delta_bs"] - DELTA_BS_BOUNDS[0]) / (DELTA_BS_BOUNDS[1] - DELTA_BS_BOUNDS[0]) <= tol or (fit["delta_bs"] - DELTA_BS_BOUNDS[0]) / (DELTA_BS_BOUNDS[1] - DELTA_BS_BOUNDS[0]) >= 1 - tol),
            "g0_boundary": False, "A_s_profile_identifiable": a_ident["identifiable"], "delta_bs_profile_identifiable": d_ident["identifiable"], "g0_profile_identifiable": True,
        })
        for parameter, rows, ident in (("A_s", fit["A_s_profile"], a_ident), ("delta_bs", fit["delta_bs_profile"], d_ident)):
            for row in rows:
                profile_rows.append({"candidate": "V2R-PS", "outer_fold": int(fold), "parameter": parameter, "value": row[parameter], "loss": row["centered_logrmse"], "loss_kind": "centered_logrmse", "profile_identifiable": ident["identifiable"], "acceptable_normalized_span": ident["acceptable_normalized_span"]})
    return fold_globals, pd.DataFrame(global_rows), pd.DataFrame(profile_rows), train_theta


def _estimate_psg_globals(
    manifest: pd.DataFrame,
    fold_manifest: pd.DataFrame,
    wavelength: np.ndarray,
    fit_mask: np.ndarray,
    optical: dict[str, np.ndarray],
    ps_globals: dict[int, dict[str, float]],
    ps_train_theta: dict[int, np.ndarray],
    config: dict[str, Any],
) -> tuple[dict[int, dict[str, float]], pd.DataFrame, pd.DataFrame]:
    observed_columns = [f"observed_reflectance_{int(w)}nm" for w in wavelength]
    assigned = manifest.merge(fold_manifest, on="subject_id", validate="one_to_one")
    optical_fit = {name: values[fit_mask] for name, values in optical.items()}
    globals_out, global_rows, profile_rows = {}, [], []
    for fold in sorted(assigned["outer_fold"].unique()):
        training = assigned.loc[assigned["outer_fold"].ne(fold)]
        values = ps_globals[int(fold)]
        predicted = np.asarray([
            forward_preloaded_numpy(theta, optical_fit, wavelength_nm=wavelength[fit_mask], s0=float(config["parameters"]["fixed_s0"]), diameter_um=15.0, scattering_amplitude=values["A_s"], delta_bs=values["delta_bs"], g0=1.0)
            for theta in ps_train_theta[int(fold)]
        ])
        observed = training[observed_columns].to_numpy(dtype=np.float64)[:, fit_mask]
        gain = estimate_global_gain(observed, predicted, bounds=G0_BOUNDS, profile_count=int(config["solver"]["global_gain"]["profile_uniform_grid_points"]))
        ident = profile_identifiability(gain["profile"], "g0", "raw_logrmse", G0_BOUNDS)
        globals_out[int(fold)] = {"A_s": values["A_s"], "delta_bs": values["delta_bs"], "g0": gain["g0"]}
        scaled = gain["g0"] * predicted
        above_one_fraction = float((scaled > 1.0 + 1e-12).mean())
        global_rows.append({
            "candidate": "V2R-PSG", "outer_fold": int(fold), **globals_out[int(fold)], "training_raw_logrmse": gain["raw_logrmse"],
            "A_s_boundary": False, "delta_bs_boundary": False, "g0_boundary": bool(gain["at_lower"] or gain["at_upper"]),
            "A_s_profile_identifiable": True, "delta_bs_profile_identifiable": True, "g0_profile_identifiable": ident["identifiable"],
            "training_prediction_above_one_fraction": above_one_fraction,
        })
        for row in gain["profile"]:
            profile_rows.append({"candidate": "V2R-PSG", "outer_fold": int(fold), "parameter": "g0", "value": row["g0"], "loss": row["raw_logrmse"], "loss_kind": "raw_logrmse", "profile_identifiable": ident["identifiable"], "acceptable_normalized_span": ident["acceptable_normalized_span"]})
    return globals_out, pd.DataFrame(global_rows), pd.DataFrame(profile_rows)


def _global_gate(candidate: str, globals_frame: pd.DataFrame, config: dict[str, Any]) -> dict[str, bool]:
    gates = config["upgrade_gates"]
    count_limit = int(gates["global_boundary_reject_fold_count_at_least"])
    parameters = ["A_s", "delta_bs"] if candidate == "V2R-PS" else ["g0"]
    checks: dict[str, bool] = {}
    references = {"A_s": 1.0, "delta_bs": 0.0, "g0": 1.0}
    for parameter in parameters:
        values = globals_frame[parameter].to_numpy(dtype=np.float64)
        direction_fraction = max(np.mean(values >= references[parameter]), np.mean(values <= references[parameter]))
        checks[f"{parameter}_direction_consistent"] = bool(direction_fraction >= float(gates["global_direction_consistent_fold_fraction_min"]))
        checks[f"{parameter}_not_frequently_boundary"] = bool(int(globals_frame[f"{parameter}_boundary"].sum()) < count_limit)
        checks[f"{parameter}_profiles_identifiable"] = bool(globals_frame[f"{parameter}_profile_identifiable"].all())
    if candidate == "V2R-PSG":
        checks["observation_scale_physical"] = bool((globals_frame["training_prediction_above_one_fraction"] == 0.0).all())
    return checks


def _individual_profile_and_sensitivity_gate(
    candidate: str,
    metrics: pd.DataFrame,
    manifest: pd.DataFrame,
    wavelength: np.ndarray,
    fit_mask: np.ndarray,
    optical: dict[str, np.ndarray],
    config: dict[str, Any],
) -> dict[str, Any]:
    # This expensive gate is reached only after every cheaper upgrade gate passes.
    observed_columns = [f"observed_reflectance_{int(w)}nm" for w in wavelength]
    indexed = manifest.set_index("subject_id")
    optical_fit = {name: values[fit_mask] for name, values in optical.items()}
    settings = replace(_settings(config), sobol_starts=8)
    profile_ok = {"f_mel": [], "f_blood": []}
    sensitivity_ok = {"f_mel": [], "f_blood": []}
    span = FIT_HI - FIT_LO
    for row_index, row in metrics.reset_index(drop=True).iterrows():
        observed = indexed.loc[row["subject_id"], observed_columns].to_numpy(dtype=np.float64)[fit_mask]
        kwargs = {name: float(row[name]) for name in ("diameter_um", "scattering_amplitude", "delta_bs", "g0")}

        def forward(theta: np.ndarray, *, s0: float | None = None, thickness: float = 0.060, hb_scale: float = 1.0, diameter: float | None = None) -> np.ndarray:
            varied = dict(optical_fit)
            if hb_scale != 1.0:
                varied = {name: values.copy() for name, values in optical_fit.items()}
                varied["mua_hbo2"] *= hb_scale
                varied["mua_hb"] *= hb_scale
            local = dict(kwargs)
            if diameter is not None:
                local["diameter_um"] = diameter
            return forward_preloaded_numpy(theta, varied, wavelength_nm=wavelength[fit_mask], s0=float(config["parameters"]["fixed_s0"] if s0 is None else s0), epidermis_thickness_mm=thickness, **local)

        best_u = (np.array([row["f_mel"], row["f_blood"]]) - FIT_LO) / span
        for index, name in enumerate(("f_mel", "f_blood")):
            profile, _ = profile_parameter(observed, lambda u: forward(FIT_LO + np.asarray(u) * span), best_u, index, settings, settings.seed + row_index * 1000 + index * 100)
            intervals = acceptable_intervals(profile, float(row["logrmse"]), float(config["parameters"]["profile_delta_logrmse"]), float(span[index]))
            profile_ok[name].append(bool(intervals["envelope_span"] <= float(config["parameters"]["profile_span_max"][index])))
        variants = []
        sens = config["fixed_quantity_sensitivity"]
        variants.extend([{"s0": float(v)} for v in sens["fixed_s0"]])
        variants.extend([{"thickness": float(v)} for v in sens["epidermis_thickness_mm"]])
        variants.extend([{"hb_scale": float(v) / 150.0} for v in sens["whole_blood_hb_g_l"]])
        if candidate != "V2R-0":
            variants.extend([{"diameter": float(v)} for v in sens["packaging_diameter_um_for_packaged_candidates"]])
        shifts = []
        for variant in variants:
            fit = fit_bounded_spectrum(observed, lambda theta, v=variant: forward(theta, **v), FIT_LO, FIT_HI, settings)
            if not fit["success"]:
                shifts.append(np.full(2, np.inf))
            else:
                shifts.append(np.abs(fit["theta"] - np.array([row["f_mel"], row["f_blood"]])))
        maximum = np.max(np.asarray(shifts), axis=0)
        for index, name in enumerate(("f_mel", "f_blood")):
            sensitivity_ok[name].append(bool(maximum[index] <= float(config["parameters"]["robustness_shift_max"][index])))
    coverage = {
        name: {"profile": float(np.mean(profile_ok[name])), "fixed_quantity_sensitivity": float(np.mean(sensitivity_ok[name]))}
        for name in ("f_mel", "f_blood")
    }
    minimum = float(config["parameters"]["profile_reliable_coverage_min"])
    checks = {
        f"{name}_profile_coverage": values["profile"] >= minimum
        for name, values in coverage.items()
    }
    checks.update({
        f"{name}_fixed_quantity_sensitivity_coverage": values["fixed_quantity_sensitivity"] >= float(config["fixed_quantity_sensitivity"]["required_stable_coverage_min"])
        for name, values in coverage.items()
    })
    return {"coverage": coverage, "checks": checks, "pass": bool(all(checks.values()))}


def run_stage_c0(config_path: str | Path, project_root: str | Path) -> dict[str, Any]:
    root = Path(project_root).resolve()
    config_file = Path(config_path).resolve()
    config = yaml.safe_load(config_file.read_text(encoding="utf-8"))
    paths = {name: _resolve(root, value) for name, value in config["inputs"].items()}
    implementation_paths = {name: _resolve(root, value) for name, value in config["implementation"].items()}
    output = _resolve(root, config["output_directory"])
    if output.exists() and bool(config["access_policy"]["refuse_existing_output"]):
        raise FileExistsError(f"Refusing to overwrite R-C0 output: {output}")

    formula_audit = json.loads(paths["formula_audit"].read_text(encoding="utf-8"))
    observation_audit = json.loads(paths["observation_audit"].read_text(encoding="utf-8"))
    checks = {
        "formula_audit_pass": formula_audit.get("status") == "PASS" and formula_audit.get("hsi_read") is False,
        "observation_audit_pass": observation_audit.get("status") == "PASS_FOR_V2R_TRAIN_INVERSION",
        "r_c0_authorized": observation_audit.get("authorized_next_stage") == "R-C0_CANDIDATE_LADDER" and observation_audit.get("r_c0_train_inversion_allowed") is True,
        "observation_manifest_hash": sha256_file(paths["observation_manifest"]) == observation_audit["outputs"]["symmetric_manifest_parquet"]["sha256"],
        "formula_contract_hash_chain": sha256_file(paths["formula_contract"]) == observation_audit["source_hashes"]["v2r_formula_contract"],
        "formula_audit_hash_chain": sha256_file(paths["formula_audit"]) == observation_audit["source_hashes"]["v2r_formula_audit"],
        "train_only_access_policy": config["access_policy"]["allowed_split"] == "train" and not any(config["access_policy"][key] for key in ("validation_content_allowed", "test_content_allowed", "clinical_500_content_allowed", "raw_hsi_content_allowed", "rgb_content_allowed")),
    }
    if not all(checks.values()):
        raise ValueError(f"R-C0 preflight failed: {checks}")

    manifest = pd.read_parquet(paths["observation_manifest"])
    if set(manifest["split"].astype(str)) != {"train"} or not manifest["input_quality_status"].eq("PASS").all() or len(manifest) != 44:
        raise ValueError("R-C0 manifest violates the frozen Train-only observation contract")
    fold_manifest = validate_fold_assignments(manifest["subject_id"].astype(str).tolist(), config["folds"]["assignments"])
    checks["frozen_fold_exact_coverage"] = True

    wavelength = np.arange(400.0, 701.0, 10.0)
    configured_fit = np.asarray(config["analysis"]["fit_centers_nm"], dtype=np.float64)
    fit_mask = np.isin(wavelength, configured_fit)
    if fit_mask.sum() != 27 or not np.array_equal(wavelength[fit_mask], configured_fit):
        raise ValueError("Frozen 420-680 nm fit bands are invalid")
    optical = load_optical_numpy(wavelength, paths["optical_asset_10nm"])

    output.mkdir(parents=True)
    _write_csv(fold_manifest, output / "fold_manifest.csv")
    fold_hash = sha256_file(output / "fold_manifest.csv")
    (output / "fold_manifest.sha256").write_text(f"{fold_hash}  fold_manifest.csv\n", encoding="ascii")

    all_metrics, all_predictions, all_residuals, all_references = [], [], [], []
    all_globals, all_profiles, comparisons = [], [], []
    candidate_summaries: dict[str, Any] = {}
    metrics_by_candidate: dict[str, pd.DataFrame] = {}
    stop_reason = None
    executed = []

    for candidate in ("V2R-0", "V2R-P"):
        metrics, predictions, residuals, references = _fit_candidate(candidate, manifest, fold_manifest, wavelength, fit_mask, optical, config)
        summary = _summarize_candidate(metrics, residuals.loc[residuals["band_role"].eq("fit")])
        summary["spectral_gate_checks"] = _spectral_gate(summary, config["spectral_gates"])
        summary["spectral_gate_pass"] = bool(all(summary["spectral_gate_checks"].values()))
        candidate_summaries[candidate] = summary
        metrics_by_candidate[candidate] = metrics
        all_metrics.append(metrics); all_predictions.append(predictions); all_residuals.append(residuals); all_references.append(references)
        executed.append(candidate)

    comparison = compare_candidates("V2R-0", "V2R-P", metrics_by_candidate["V2R-0"], metrics_by_candidate["V2R-P"], candidate_summaries["V2R-0"], candidate_summaries["V2R-P"], config)
    if comparison["cheap_upgrade_gate_pass"]:
        expensive = _individual_profile_and_sensitivity_gate("V2R-P", metrics_by_candidate["V2R-P"], manifest, wavelength, fit_mask, optical, config)
    else:
        expensive = {"status": "NOT_EVALUATED_DUE_TO_PRIOR_FAILED_GATE", "checks": {}, "pass": False}
    comparison["individual_profile_and_sensitivity"] = expensive
    comparison["upgrade_pass"] = bool(comparison["cheap_upgrade_gate_pass"] and expensive["pass"])
    comparisons.append(comparison)

    ps_globals = None
    ps_train_theta = None
    if not comparison["upgrade_pass"]:
        stop_reason = "V2R-P_FAILED_UPGRADE_GATE"
    else:
        ps_globals, global_frame, profile_frame, ps_train_theta = _estimate_ps_globals(manifest, fold_manifest, wavelength, fit_mask, optical, metrics_by_candidate["V2R-P"], config)
        metrics, predictions, residuals, references = _fit_candidate("V2R-PS", manifest, fold_manifest, wavelength, fit_mask, optical, config, ps_globals)
        summary = _summarize_candidate(metrics, residuals.loc[residuals["band_role"].eq("fit")])
        summary["spectral_gate_checks"] = _spectral_gate(summary, config["spectral_gates"])
        summary["spectral_gate_pass"] = bool(all(summary["spectral_gate_checks"].values()))
        candidate_summaries["V2R-PS"] = summary; metrics_by_candidate["V2R-PS"] = metrics
        all_metrics.append(metrics); all_predictions.append(predictions); all_residuals.append(residuals); all_references.append(references)
        all_globals.append(global_frame); all_profiles.append(profile_frame); executed.append("V2R-PS")
        comparison = compare_candidates("V2R-P", "V2R-PS", metrics_by_candidate["V2R-P"], metrics, candidate_summaries["V2R-P"], summary, config)
        global_checks = _global_gate("V2R-PS", global_frame, config)
        comparison["global_checks"] = global_checks
        comparison["cheap_upgrade_gate_pass"] = bool(comparison["cheap_upgrade_gate_pass"] and all(global_checks.values()))
        expensive = _individual_profile_and_sensitivity_gate("V2R-PS", metrics, manifest, wavelength, fit_mask, optical, config) if comparison["cheap_upgrade_gate_pass"] else {"status": "NOT_EVALUATED_DUE_TO_PRIOR_FAILED_GATE", "checks": {}, "pass": False}
        comparison["individual_profile_and_sensitivity"] = expensive
        comparison["upgrade_pass"] = bool(comparison["cheap_upgrade_gate_pass"] and expensive["pass"])
        comparisons.append(comparison)
        if not comparison["upgrade_pass"]:
            stop_reason = "V2R-PS_FAILED_UPGRADE_GATE"
        else:
            psg_globals, gain_frame, gain_profiles = _estimate_psg_globals(manifest, fold_manifest, wavelength, fit_mask, optical, ps_globals, ps_train_theta, config)
            metrics, predictions, residuals, references = _fit_candidate("V2R-PSG", manifest, fold_manifest, wavelength, fit_mask, optical, config, psg_globals)
            summary = _summarize_candidate(metrics, residuals.loc[residuals["band_role"].eq("fit")])
            summary["spectral_gate_checks"] = _spectral_gate(summary, config["spectral_gates"])
            summary["spectral_gate_pass"] = bool(all(summary["spectral_gate_checks"].values()))
            candidate_summaries["V2R-PSG"] = summary; metrics_by_candidate["V2R-PSG"] = metrics
            all_metrics.append(metrics); all_predictions.append(predictions); all_residuals.append(residuals); all_references.append(references)
            all_globals.append(gain_frame); all_profiles.append(gain_profiles); executed.append("V2R-PSG")
            comparison = compare_candidates("V2R-PS", "V2R-PSG", metrics_by_candidate["V2R-PS"], metrics, candidate_summaries["V2R-PS"], summary, config)
            global_checks = _global_gate("V2R-PSG", gain_frame, config)
            comparison["global_checks"] = global_checks
            comparison["cheap_upgrade_gate_pass"] = bool(comparison["cheap_upgrade_gate_pass"] and all(global_checks.values()))
            expensive = _individual_profile_and_sensitivity_gate("V2R-PSG", metrics, manifest, wavelength, fit_mask, optical, config) if comparison["cheap_upgrade_gate_pass"] else {"status": "NOT_EVALUATED_DUE_TO_PRIOR_FAILED_GATE", "checks": {}, "pass": False}
            comparison["individual_profile_and_sensitivity"] = expensive
            comparison["upgrade_pass"] = bool(comparison["cheap_upgrade_gate_pass"] and expensive["pass"])
            comparisons.append(comparison)
            stop_reason = None if comparison["upgrade_pass"] else "V2R-PSG_FAILED_ACCEPTANCE_GATE"

    metrics_frame = pd.concat(all_metrics, ignore_index=True)
    prediction_frame = pd.concat(all_predictions, ignore_index=True)
    residual_frame = pd.concat(all_residuals, ignore_index=True)
    reference_frame = pd.concat(all_references, ignore_index=True)
    globals_frame = pd.concat(all_globals, ignore_index=True) if all_globals else pd.DataFrame(columns=["candidate", "outer_fold", "A_s", "delta_bs", "g0"])
    profiles_frame = pd.concat(all_profiles, ignore_index=True) if all_profiles else pd.DataFrame(columns=["candidate", "outer_fold", "parameter", "value", "loss", "loss_kind"])
    comparison_frame = pd.DataFrame([{k: v for k, v in row.items() if k not in ("checks", "global_checks", "individual_profile_and_sensitivity")} | {"checks_json": row["checks"], "global_checks_json": row.get("global_checks", {}), "individual_gate_json": row["individual_profile_and_sensitivity"]} for row in comparisons])
    outputs = {
        "candidate_subject_metrics.csv": metrics_frame,
        "oof_predictions.csv": prediction_frame,
        "oof_residuals.csv": residual_frame,
        "outer_fold_reference_metrics.csv": reference_frame,
        "candidate_comparisons.csv": comparison_frame,
        "fold_global_parameters.csv": globals_frame,
        "global_parameter_profiles.csv": profiles_frame,
    }
    for name, frame in outputs.items():
        _write_csv(frame, output / name)

    decision = {
        "schema_version": 1, "stage": "KM-BIO-v2R-R-C0", "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "R_C0_STOPPED_AT_FAILED_UPGRADE" if stop_reason else "R_C0_CANDIDATE_LADDER_PASS",
        "executed_candidates": executed, "unexecuted_candidates": [name for name in config["candidates"]["order"] if name not in executed],
        "stop_reason": stop_reason, "candidate_summaries": candidate_summaries, "comparisons": comparisons,
        "validation_authorized": False, "validation_test_500_reads": 0,
        "interpretation": "Train-only candidate selection; parameters remain model-conditional effective quantities.",
    }
    (output / "candidate_decision.json").write_text(json.dumps(_jsonable(decision), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    audit = {
        "schema_version": 1, "stage": "KM-BIO-v2R-R-C0", "status": decision["status"], "created_utc": decision["created_utc"],
        "preflight_checks": checks, "fold_manifest_sha256": fold_hash, "executed_candidates": executed,
        "stop_reason": stop_reason, "candidate_summaries": candidate_summaries,
        "data_access": {"manifest_rows_read": int(len(manifest)), "raw_hsi_reads": 0, "rgb_reads": 0, "validation_reads": 0, "test_reads": 0, "clinical_500_reads": 0},
        "input_hashes": {name: sha256_file(path) for name, path in paths.items()},
        "implementation_hashes": {name: sha256_file(path) for name, path in implementation_paths.items()},
    }
    (output / "audit_summary.json").write_text(json.dumps(_jsonable(audit), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    lines = [
        "# KM-BIO-v2R R-C0 candidate ladder", "", f"- Status: `{decision['status']}`", f"- Executed: `{', '.join(executed)}`",
        f"- Stop reason: `{stop_reason or 'none'}`", f"- Frozen fold manifest SHA-256: `{fold_hash}`", "",
    ]
    for name in executed:
        summary = candidate_summaries[name]
        lines.extend([f"## {name}", "", f"- Median / P90 logRMSE: `{summary['median_logrmse']:.6f}` / `{summary['p90_logrmse']:.6f}`", f"- Median RMSE / SAM: `{summary['median_rmse']:.6f}` / `{summary['median_sam_deg']:.3f} deg`", f"- Spectral gate: `{summary['spectral_gate_pass']}`", ""])
    (output / "audit_summary.md").write_text("\n".join(lines), encoding="utf-8")

    artifact_rows = []
    for path in sorted(output.iterdir(), key=lambda p: p.name):
        if path.name != "artifact_hash_manifest.csv" and path.is_file():
            artifact_rows.append({"role": "output", "name": path.name, "path": str(path), "sha256": sha256_file(path)})
    for name, path in paths.items():
        artifact_rows.append({"role": "input", "name": name, "path": str(path), "sha256": sha256_file(path)})
    for name, path in implementation_paths.items():
        artifact_rows.append({"role": "implementation", "name": name, "path": str(path), "sha256": sha256_file(path)})
    artifact_rows.append({"role": "config", "name": "candidate_ladder_config", "path": str(config_file), "sha256": sha256_file(config_file)})
    _write_csv(pd.DataFrame(artifact_rows), output / "artifact_hash_manifest.csv")
    return decision
