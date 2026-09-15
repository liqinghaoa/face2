"""KM-BIO-v2R.1 R-C0R resumed PS/PSG candidate ladder.

This module deliberately does not call the historical R-C0 runner.  It reads
the R-C0 V2R-0/P artifacts as immutable inputs, then independently computes
only the registered V2R-PS and V2R-PSG candidates.
"""

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

from .km_bio_inverse import FitSettings, fit_bounded_spectrum, spectral_metrics
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


class ContractError(RuntimeError):
    """A hard R-C0R integrity or implementation violation."""


def _resolve(root: Path, value: str) -> Path:
    path = Path(value)
    return (root / path).resolve() if not path.is_absolute() else path.resolve()


def _jsonable(value: Any) -> Any:
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def _write_csv(frame: pd.DataFrame, path: Path) -> None:
    export = frame.copy()
    for column in export.columns:
        if export[column].dtype == object:
            export[column] = export[column].map(
                lambda value: json.dumps(_jsonable(value), ensure_ascii=False, separators=(",", ":"))
                if isinstance(value, (dict, list, tuple, np.ndarray)) else value
            )
    export.to_csv(path, index=False, encoding="utf-8-sig")


def _fit_settings(config: dict[str, Any]) -> FitSettings:
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
    if len(frame) != len(subjects) or frame["subject_id"].duplicated().any():
        raise ContractError("Frozen fold assignment contains duplicate or missing subjects")
    if set(frame["subject_id"]) != set(map(str, subjects)):
        raise ContractError("Frozen fold assignment differs from the R-B subject set")
    return frame


def _validate_reflectance(values: np.ndarray, context: str, *, require_unit_interval: bool) -> None:
    array = np.asarray(values, dtype=np.float64)
    if not np.isfinite(array).all():
        raise ContractError(f"Non-finite reflectance in {context}")
    if np.any(array < 0.0):
        raise ContractError(f"Negative reflectance in {context}")
    if require_unit_interval and np.any(array > 1.0 + 1e-12):
        raise ContractError(f"Unclipped reflectance above one in {context}")


def _spectral_summary(metrics: pd.DataFrame, residuals: pd.DataFrame) -> dict[str, Any]:
    median_band = residuals.groupby("wavelength_nm", sort=True)["signed_residual"].median()
    return {
        "subject_count": int(len(metrics)),
        "all_solver_converged": bool(metrics["solver_converged"].all()),
        "median_logrmse": float(metrics["logrmse"].median()),
        "p90_logrmse": float(metrics["logrmse"].quantile(0.90)),
        "median_rmse": float(metrics["rmse"].median()),
        "median_sam_deg": float(metrics["sam_deg"].median()),
        "median_centered_logrmse": float(metrics["centered_logrmse"].median()),
        "median_mean_log_residual": float(metrics["mean_log_residual"].median()),
        "maximum_abs_median_signed_band_residual": float(median_band.abs().max()),
        "model_better_than_reference_fraction": float(metrics["model_better_than_reference"].mean()),
        "median_model_to_reference_error_ratio": float(metrics["model_to_reference_error_ratio"].median()),
        "f_mel_lower_boundary_fraction": float(metrics["f_mel_at_lower"].mean()),
        "f_mel_upper_boundary_fraction": float(metrics["f_mel_at_upper"].mean()),
        "f_mel_any_boundary_fraction": float((metrics["f_mel_at_lower"] | metrics["f_mel_at_upper"]).mean()),
        "f_blood_lower_boundary_fraction": float(metrics["f_blood_at_lower"].mean()),
        "f_blood_upper_boundary_fraction": float(metrics["f_blood_at_upper"].mean()),
        "median_signed_residual_by_wavelength": {str(int(wave)): float(value) for wave, value in median_band.items()},
    }


def _strong_target_checks(summary: dict[str, Any], targets: dict[str, Any]) -> dict[str, bool]:
    return {
        "median_logrmse": summary["median_logrmse"] <= float(targets["subject_median_logrmse_max"]),
        "p90_logrmse": summary["p90_logrmse"] <= float(targets["subject_p90_logrmse_max"]),
        "median_rmse": summary["median_rmse"] <= float(targets["subject_median_rmse_max"]),
        "median_sam_deg": summary["median_sam_deg"] <= float(targets["subject_median_sam_deg_max"]),
        "maximum_abs_median_signed_band_residual": summary["maximum_abs_median_signed_band_residual"] <= float(targets["maximum_abs_median_signed_band_residual"]),
        "reference_better_fraction": summary["model_better_than_reference_fraction"] >= float(targets["outer_fold_reference_better_fraction_min"]),
        "model_to_reference_median_error_ratio": summary["median_model_to_reference_error_ratio"] <= float(targets["model_to_reference_median_error_ratio_max"]),
    }


def _history_paths(root: Path, config: dict[str, Any]) -> dict[str, Path]:
    return {key: _resolve(root, value) for key, value in config["historical_r_c0"].items() if isinstance(value, str) and ("/" in value or "\\" in value)}


def verify_historical_r_c0(root: Path, config: dict[str, Any]) -> tuple[dict[str, pd.DataFrame], pd.DataFrame]:
    paths = _history_paths(root, config)
    history = config["historical_r_c0"]
    required = ("decision_json", "audit_json", "artifact_hash_manifest", "fold_manifest", "subject_metrics", "oof_predictions", "oof_residuals", "outer_fold_reference_metrics")
    if any(not paths[name].is_file() for name in required):
        raise ContractError("A required historical R-C0 artifact is missing")
    artifact = pd.read_csv(paths["artifact_hash_manifest"], encoding="utf-8-sig")
    rows = []
    for _, item in artifact.iterrows():
        path = Path(str(item["path"]))
        actual = sha256_file(path) if path.is_file() else None
        rows.append({
            "artifact_name": str(item["name"]), "role": str(item["role"]), "path": str(path),
            "expected_sha256": str(item["sha256"]), "actual_sha256": actual,
            "matches": bool(actual == str(item["sha256"])),
        })
    verification = pd.DataFrame(rows)
    if not verification["matches"].all():
        raise ContractError("Historical R-C0 artifact hash mismatch")
    decision = json.loads(paths["decision_json"].read_text(encoding="utf-8"))
    audit = json.loads(paths["audit_json"].read_text(encoding="utf-8"))
    if decision.get("status") != history["expected_status"] or audit.get("status") != history["expected_status"]:
        raise ContractError("Historical R-C0 status differs from the frozen resume contract")
    if decision.get("executed_candidates") != history["reuse_candidates"]:
        raise ContractError("Historical R-C0 did not contain exactly the reusable V2R-0/P candidates")
    if decision.get("unexecuted_candidates") != ["V2R-PS", "V2R-PSG"]:
        raise ContractError("Historical R-C0 candidate provenance is inconsistent")
    if sha256_file(paths["fold_manifest"]) != history["expected_fold_manifest_sha256"]:
        raise ContractError("Historical frozen-fold manifest hash mismatch")
    frames = {
        "metrics": pd.read_csv(paths["subject_metrics"], encoding="utf-8-sig"),
        "predictions": pd.read_csv(paths["oof_predictions"], encoding="utf-8-sig"),
        "residuals": pd.read_csv(paths["oof_residuals"], encoding="utf-8-sig"),
        "reference": pd.read_csv(paths["outer_fold_reference_metrics"], encoding="utf-8-sig"),
        "folds": pd.read_csv(paths["fold_manifest"], encoding="utf-8-sig"),
    }
    for name, frame in frames.items():
        if frame.empty:
            raise ContractError(f"Historical R-C0 {name} artifact is empty")
    for candidate in history["reuse_candidates"]:
        if int(frames["metrics"].loc[frames["metrics"]["candidate"].eq(candidate), "subject_id"].nunique()) != 44:
            raise ContractError(f"Historical {candidate} metrics do not contain 44 subjects")
        expected_rows = 44 * 31
        if len(frames["predictions"].loc[frames["predictions"]["candidate"].eq(candidate)]) != expected_rows:
            raise ContractError(f"Historical {candidate} prediction rows are incomplete")
        if len(frames["residuals"].loc[frames["residuals"]["candidate"].eq(candidate)]) != expected_rows:
            raise ContractError(f"Historical {candidate} residual rows are incomplete")
    return frames, verification


def _shape_joint_fit(
    observed: np.ndarray,
    optical: dict[str, np.ndarray],
    wavelength: np.ndarray,
    initial_theta: np.ndarray,
    config: dict[str, Any],
) -> dict[str, Any]:
    """Jointly estimate fold-global scattering and train-subject theta by shape."""

    raw = config["solver"]["global_shape"]
    epsilon = float(config["analysis"]["epsilon"])
    n_subjects = observed.shape[0]
    if observed.shape[1] != len(wavelength) or initial_theta.shape != (n_subjects, 2):
        raise ContractError("Shape optimizer received misaligned training inputs")

    def unpack(x: np.ndarray) -> tuple[float, float, np.ndarray]:
        amplitude = AS_BOUNDS[0] + x[0] * (AS_BOUNDS[1] - AS_BOUNDS[0])
        delta = DELTA_BS_BOUNDS[0] + x[1] * (DELTA_BS_BOUNDS[1] - DELTA_BS_BOUNDS[0])
        theta = FIT_LO + x[2:].reshape(n_subjects, 2) * (FIT_HI - FIT_LO)
        return float(amplitude), float(delta), theta

    def residual(x: np.ndarray) -> np.ndarray:
        amplitude, delta, theta = unpack(x)
        prediction = np.asarray([
            forward_preloaded_numpy(
                value, optical, wavelength_nm=wavelength, s0=float(config["parameters"]["fixed_s0"]),
                diameter_um=15.0, scattering_amplitude=amplitude, delta_bs=delta, g0=1.0,
            ) for value in theta
        ], dtype=np.float64)
        _validate_reflectance(prediction, "PS Train-global shape optimization", require_unit_interval=True)
        log_residual = np.log(prediction + epsilon) - np.log(observed + epsilon)
        return (log_residual - log_residual.mean(axis=1, keepdims=True)).ravel()

    theta_u = (initial_theta - FIT_LO) / (FIT_HI - FIT_LO)
    candidates: list[dict[str, Any]] = []
    for start_index, global_u in enumerate(raw["start_global_u"]):
        start = np.concatenate([np.asarray(global_u, dtype=np.float64), theta_u.ravel()])
        result = least_squares(
            residual, start, bounds=(np.zeros_like(start), np.ones_like(start)), method="trf", loss="linear",
            ftol=float(raw["ftol"]), xtol=float(raw["xtol"]), gtol=float(raw["gtol"]), max_nfev=int(raw["max_nfev_per_start"]),
        )
        loss = float(np.sqrt(np.mean(residual(result.x) ** 2)))
        if np.isfinite(loss):
            amplitude, delta, theta = unpack(result.x)
            candidates.append({
                "start_index": start_index, "result": result, "loss": loss, "A_s": amplitude,
                "delta_bs": delta, "theta": theta,
            })
    if not candidates:
        raise ContractError("PS Train-global shape optimizer produced no finite candidate")
    best_loss = min(item["loss"] for item in candidates)
    tie = float(raw["tie_loss_tolerance"])
    tied = [item for item in candidates if item["loss"] <= best_loss + tie]
    best = min(tied, key=lambda item: (abs(item["A_s"] - 1.0) + abs(item["delta_bs"]), item["start_index"]))

    def profile_parameter(index: int, name: str, bounds: tuple[float, float]) -> list[dict[str, Any]]:
        grid = np.linspace(0.0, 1.0, int(raw["profile_grid_points"]), dtype=np.float64)
        if not np.any(np.isclose(grid, best["result"].x[index], atol=1e-14, rtol=0.0)):
            grid = np.sort(np.append(grid, best["result"].x[index]))
        remaining = np.asarray([value for value in range(len(best["result"].x)) if value != index], dtype=np.int64)
        rows: list[dict[str, Any]] = []
        for point_index, fixed_u in enumerate(grid):
            start_reduced = best["result"].x[remaining]

            def conditional(z: np.ndarray) -> np.ndarray:
                full = best["result"].x.copy()
                full[index] = fixed_u
                full[remaining] = z
                return residual(full)

            result = least_squares(
                conditional, start_reduced, bounds=(np.zeros_like(start_reduced), np.ones_like(start_reduced)),
                method="trf", loss="linear", ftol=float(raw["ftol"]), xtol=float(raw["xtol"]),
                gtol=float(raw["gtol"]), max_nfev=int(raw["profile_conditional_max_nfev"]),
            )
            loss = float(np.sqrt(np.mean(conditional(result.x) ** 2)))
            if not np.isfinite(loss):
                raise ContractError(f"Non-finite {name} profile loss at point {point_index}")
            rows.append({
                name: float(bounds[0] + fixed_u * (bounds[1] - bounds[0])),
                "centered_logrmse": loss, "solver_success": bool(result.success), "nfev": int(result.nfev),
                "grid_kind": "uniform_or_exact_optimum",
            })
        return rows

    a_profile = profile_parameter(0, "A_s", AS_BOUNDS)
    delta_profile = profile_parameter(1, "delta_bs", DELTA_BS_BOUNDS)
    return {
        "A_s": best["A_s"], "delta_bs": best["delta_bs"], "theta": best["theta"],
        "centered_logrmse": best["loss"], "start_index": best["start_index"],
        "optimizer_success": bool(best["result"].success), "A_s_profile": a_profile,
        "delta_bs_profile": delta_profile,
    }


def _estimate_ps_globals(
    manifest: pd.DataFrame,
    fold_manifest: pd.DataFrame,
    wavelength: np.ndarray,
    fit_mask: np.ndarray,
    optical: dict[str, np.ndarray],
    historical_p_metrics: pd.DataFrame,
    config: dict[str, Any],
) -> tuple[dict[int, dict[str, float]], pd.DataFrame, pd.DataFrame, dict[int, dict[str, Any]], pd.DataFrame]:
    columns = [f"observed_reflectance_{int(value)}nm" for value in wavelength]
    assigned = manifest.merge(fold_manifest, on="subject_id", validate="one_to_one")
    optical_fit = {name: values[fit_mask] for name, values in optical.items()}
    globals_by_fold: dict[int, dict[str, float]] = {}
    global_rows, profile_rows, membership_rows = [], [], []
    train_cache: dict[int, dict[str, Any]] = {}
    reporting = config["global_identifiability_reporting"]
    for fold in sorted(assigned["outer_fold"].unique()):
        training = assigned.loc[assigned["outer_fold"].ne(fold)].copy().sort_values("subject_id")
        held = assigned.loc[assigned["outer_fold"].eq(fold)]
        initial = training[["subject_id"]].merge(
            historical_p_metrics[["subject_id", "f_mel", "f_blood"]], on="subject_id", how="left", validate="one_to_one"
        )[["f_mel", "f_blood"]].to_numpy(dtype=np.float64)
        if not np.isfinite(initial).all():
            raise ContractError(f"Historical V2R-P initial theta missing for PS fold {fold}")
        observed = training[columns].to_numpy(dtype=np.float64)[:, fit_mask]
        fit = _shape_joint_fit(observed, optical_fit, wavelength[fit_mask], initial, config)
        a_ident = profile_identifiability(
            fit["A_s_profile"], "A_s", "centered_logrmse", AS_BOUNDS,
            delta_logrmse=float(reporting["acceptable_loss_delta_logrmse"]),
            max_normalized_span=float(reporting["maximum_acceptable_normalized_envelope_span"]),
        )
        delta_ident = profile_identifiability(
            fit["delta_bs_profile"], "delta_bs", "centered_logrmse", DELTA_BS_BOUNDS,
            delta_logrmse=float(reporting["acceptable_loss_delta_logrmse"]),
            max_normalized_span=float(reporting["maximum_acceptable_normalized_envelope_span"]),
        )
        tolerance = float(reporting["boundary_tolerance_normalized"])
        a_u = (fit["A_s"] - AS_BOUNDS[0]) / (AS_BOUNDS[1] - AS_BOUNDS[0])
        delta_u = (fit["delta_bs"] - DELTA_BS_BOUNDS[0]) / (DELTA_BS_BOUNDS[1] - DELTA_BS_BOUNDS[0])
        globals_by_fold[int(fold)] = {"A_s": fit["A_s"], "delta_bs": fit["delta_bs"], "g0": 1.0}
        train_cache[int(fold)] = {"subject_id": training["subject_id"].astype(str).tolist(), "theta": fit["theta"], "observed": observed}
        global_rows.append({
            "candidate": "V2R-PS", "outer_fold": int(fold), "A_s": fit["A_s"], "delta_bs": fit["delta_bs"], "g0": 1.0,
            "training_subject_count": int(len(training)), "held_subject_count": int(len(held)),
            "training_centered_logrmse": fit["centered_logrmse"], "global_start_index": fit["start_index"],
            "global_optimizer_success": fit["optimizer_success"], "A_s_boundary": bool(a_u <= tolerance or a_u >= 1.0 - tolerance),
            "delta_bs_boundary": bool(delta_u <= tolerance or delta_u >= 1.0 - tolerance), "g0_boundary": False,
            "A_s_profile_identifiable": bool(a_ident["identifiable"]), "delta_bs_profile_identifiable": bool(delta_ident["identifiable"]),
            "g0_profile_identifiable": True,
        })
        membership_rows.extend([
            {"candidate": "V2R-PS", "outer_fold": int(fold), "subject_id": subject, "role": "train_global_estimation"}
            for subject in training["subject_id"].astype(str)
        ])
        membership_rows.extend([
            {"candidate": "V2R-PS", "outer_fold": int(fold), "subject_id": subject, "role": "held_out_oof"}
            for subject in held["subject_id"].astype(str)
        ])
        for name, profile, ident in (("A_s", fit["A_s_profile"], a_ident), ("delta_bs", fit["delta_bs_profile"], delta_ident)):
            for row in profile:
                profile_rows.append({
                    "candidate": "V2R-PS", "outer_fold": int(fold), "parameter": name, "value": row[name],
                    "loss": row["centered_logrmse"], "loss_kind": "centered_logrmse", "solver_success": row["solver_success"],
                    "nfev": row["nfev"], "grid_kind": row["grid_kind"], "profile_source": "estimated_in_fold",
                    "profile_identifiable": ident["identifiable"], "acceptable_normalized_span": ident["acceptable_normalized_span"],
                })
    return globals_by_fold, pd.DataFrame(global_rows), pd.DataFrame(profile_rows), train_cache, pd.DataFrame(membership_rows)


def _estimate_psg_globals(
    manifest: pd.DataFrame,
    fold_manifest: pd.DataFrame,
    wavelength: np.ndarray,
    fit_mask: np.ndarray,
    optical: dict[str, np.ndarray],
    ps_globals: dict[int, dict[str, float]],
    train_cache: dict[int, dict[str, Any]],
    ps_profile_rows: pd.DataFrame,
    config: dict[str, Any],
) -> tuple[dict[int, dict[str, float]], pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    assigned = manifest.merge(fold_manifest, on="subject_id", validate="one_to_one")
    optical_fit = {name: values[fit_mask] for name, values in optical.items()}
    raw = config["solver"]["global_gain"]
    reporting = config["global_identifiability_reporting"]
    tolerance = float(reporting["boundary_tolerance_normalized"])
    globals_by_fold: dict[int, dict[str, float]] = {}
    global_rows, profile_rows, membership_rows = [], [], []
    for fold in sorted(assigned["outer_fold"].unique()):
        cache = train_cache[int(fold)]
        values = ps_globals[int(fold)]
        prediction = np.asarray([
            forward_preloaded_numpy(
                theta, optical_fit, wavelength_nm=wavelength[fit_mask], s0=float(config["parameters"]["fixed_s0"]),
                diameter_um=15.0, scattering_amplitude=values["A_s"], delta_bs=values["delta_bs"], g0=1.0,
            ) for theta in cache["theta"]
        ], dtype=np.float64)
        _validate_reflectance(prediction, f"PSG fold {fold} unscaled Train prediction", require_unit_interval=True)
        gain = estimate_global_gain(cache["observed"], prediction, bounds=tuple(raw["bounds"]), profile_count=int(raw["profile_uniform_grid_points"]))
        scaled = float(gain["g0"]) * prediction
        _validate_reflectance(scaled, f"PSG fold {fold} scaled Train prediction", require_unit_interval=True)
        ident = profile_identifiability(
            gain["profile"], "g0", "raw_logrmse", G0_BOUNDS,
            delta_logrmse=float(reporting["acceptable_loss_delta_logrmse"]),
            max_normalized_span=float(reporting["maximum_acceptable_normalized_envelope_span"]),
        )
        g_u = (gain["g0"] - G0_BOUNDS[0]) / (G0_BOUNDS[1] - G0_BOUNDS[0])
        globals_by_fold[int(fold)] = {"A_s": values["A_s"], "delta_bs": values["delta_bs"], "g0": float(gain["g0"])}
        ps_row = ps_profile_rows.loc[(ps_profile_rows["outer_fold"].eq(fold)) & (ps_profile_rows["parameter"].eq("A_s"))].iloc[0]
        delta_row = ps_profile_rows.loc[(ps_profile_rows["outer_fold"].eq(fold)) & (ps_profile_rows["parameter"].eq("delta_bs"))].iloc[0]
        global_rows.append({
            "candidate": "V2R-PSG", "outer_fold": int(fold), **globals_by_fold[int(fold)],
            "training_subject_count": int(len(cache["subject_id"])), "held_subject_count": int((assigned["outer_fold"] == fold).sum()),
            "training_raw_logrmse": float(gain["raw_logrmse"]), "unclipped_g0": float(gain["unclipped_g0"]),
            "A_s_boundary": bool(values["A_s"] <= AS_BOUNDS[0] + tolerance * (AS_BOUNDS[1] - AS_BOUNDS[0]) or values["A_s"] >= AS_BOUNDS[1] - tolerance * (AS_BOUNDS[1] - AS_BOUNDS[0])),
            "delta_bs_boundary": bool(values["delta_bs"] <= DELTA_BS_BOUNDS[0] + tolerance * (DELTA_BS_BOUNDS[1] - DELTA_BS_BOUNDS[0]) or values["delta_bs"] >= DELTA_BS_BOUNDS[1] - tolerance * (DELTA_BS_BOUNDS[1] - DELTA_BS_BOUNDS[0])),
            "g0_boundary": bool(g_u <= tolerance or g_u >= 1.0 - tolerance),
            "A_s_profile_identifiable": bool(ps_row["profile_identifiable"]), "delta_bs_profile_identifiable": bool(delta_row["profile_identifiable"]),
            "g0_profile_identifiable": bool(ident["identifiable"]), "training_scaled_prediction_above_one_count": int((scaled > 1.0 + 1e-12).sum()),
        })
        membership_rows.extend([
            {"candidate": "V2R-PSG", "outer_fold": int(fold), "subject_id": subject, "role": "train_global_estimation"}
            for subject in cache["subject_id"]
        ])
        for parameter in ("A_s", "delta_bs"):
            frozen = ps_profile_rows.loc[(ps_profile_rows["outer_fold"].eq(fold)) & (ps_profile_rows["parameter"].eq(parameter))].copy()
            frozen["candidate"] = "V2R-PSG"
            frozen["profile_source"] = "frozen_from_V2R-PS"
            profile_rows.extend(frozen.to_dict("records"))
        for row in gain["profile"]:
            profile_rows.append({
                "candidate": "V2R-PSG", "outer_fold": int(fold), "parameter": "g0", "value": row["g0"],
                "loss": row["raw_logrmse"], "loss_kind": "raw_logrmse", "solver_success": True, "nfev": 0,
                "grid_kind": "uniform_201_or_exact_optimum", "profile_source": "estimated_in_fold",
                "profile_identifiable": ident["identifiable"], "acceptable_normalized_span": ident["acceptable_normalized_span"],
            })
    return globals_by_fold, pd.DataFrame(global_rows), pd.DataFrame(profile_rows), pd.DataFrame(membership_rows)


def _candidate_kwargs(candidate: str, fold: int, globals_by_fold: dict[int, dict[str, float]]) -> dict[str, float]:
    values = globals_by_fold[int(fold)]
    return {
        "diameter_um": 15.0, "scattering_amplitude": float(values["A_s"]),
        "delta_bs": float(values["delta_bs"]), "g0": float(values["g0"]),
    }


def _fit_oof_candidate(
    candidate: str,
    manifest: pd.DataFrame,
    fold_manifest: pd.DataFrame,
    wavelength: np.ndarray,
    fit_mask: np.ndarray,
    optical: dict[str, np.ndarray],
    globals_by_fold: dict[int, dict[str, float]],
    config: dict[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    columns = [f"observed_reflectance_{int(value)}nm" for value in wavelength]
    assigned = manifest.merge(fold_manifest, on="subject_id", validate="one_to_one")
    settings = _fit_settings(config)
    optical_fit = {name: values[fit_mask] for name, values in optical.items()}
    metrics_rows, prediction_rows, residual_rows, reference_rows, start_rows = [], [], [], [], []
    for fold in sorted(assigned["outer_fold"].unique()):
        training = assigned.loc[assigned["outer_fold"].ne(fold)]
        held = assigned.loc[assigned["outer_fold"].eq(fold)]
        epsilon = settings.epsilon
        train_observed = training[columns].to_numpy(dtype=np.float64)[:, fit_mask]
        reference = np.exp(np.mean(np.log(train_observed + epsilon), axis=0)) - epsilon
        _validate_reflectance(reference, f"outer-fold reference {fold}", require_unit_interval=True)
        kwargs = _candidate_kwargs(candidate, int(fold), globals_by_fold)
        for local_index, (_, item) in enumerate(held.iterrows()):
            observed_full = item[columns].to_numpy(dtype=np.float64)
            observed = observed_full[fit_mask]

            def forward_fit(theta: np.ndarray) -> np.ndarray:
                prediction = forward_preloaded_numpy(
                    theta, optical_fit, wavelength_nm=wavelength[fit_mask], s0=float(config["parameters"]["fixed_s0"]), **kwargs,
                )
                _validate_reflectance(prediction, f"{candidate} fit forward {item['subject_id']}", require_unit_interval=True)
                return prediction

            seed = settings.seed + 100000 * (1 if candidate == "V2R-PS" else 2) + 1000 * int(fold) + local_index
            fit = fit_bounded_spectrum(observed, forward_fit, FIT_LO, FIT_HI, replace(settings, seed=seed))
            if not fit["success"]:
                raise ContractError(f"{candidate} has no valid converged OOF solution for {item['subject_id']}")
            prediction_full = forward_preloaded_numpy(
                fit["theta"], optical, wavelength_nm=wavelength, s0=float(config["parameters"]["fixed_s0"]), **kwargs,
            )
            _validate_reflectance(prediction_full, f"{candidate} OOF prediction {item['subject_id']}", require_unit_interval=True)
            prediction = prediction_full[fit_mask]
            metrics = spectral_metrics(prediction, observed, epsilon)
            log_residual = np.log(prediction + epsilon) - np.log(observed + epsilon)
            centered = log_residual - log_residual.mean()
            reference_metrics = spectral_metrics(reference, observed, epsilon)
            u = (fit["theta"] - FIT_LO) / (FIT_HI - FIT_LO)
            selected = fit["starts"][fit["selected_start_index"]]
            metrics_rows.append({
                "candidate": candidate, "subject_id": str(item["subject_id"]), "capture_id": str(item["capture_id"]), "outer_fold": int(fold),
                "solver_converged": True, "selected_start_index": int(fit["selected_start_index"]), "selected_nfev": int(selected["nfev"]),
                "selected_status": int(selected["status"]), "selected_optimality": float(selected["optimality"]),
                "f_mel": float(fit["theta"][0]), "f_blood": float(fit["theta"][1]), "c_tHb_eq_g_l": float(150.0 * fit["theta"][1]),
                **metrics, "centered_logrmse": float(np.sqrt(np.mean(centered ** 2))), "mean_log_residual": float(log_residual.mean()),
                "reference_logrmse": float(reference_metrics["logrmse"]),
                "model_to_reference_error_ratio": float(metrics["logrmse"] / reference_metrics["logrmse"]),
                "model_better_than_reference": bool(metrics["logrmse"] < reference_metrics["logrmse"]),
                "f_mel_at_lower": bool(u[0] <= float(config["parameters"]["boundary_tolerance_normalized"])),
                "f_mel_at_upper": bool(u[0] >= 1.0 - float(config["parameters"]["boundary_tolerance_normalized"])),
                "f_blood_at_lower": bool(u[1] <= float(config["parameters"]["boundary_tolerance_normalized"])),
                "f_blood_at_upper": bool(u[1] >= 1.0 - float(config["parameters"]["boundary_tolerance_normalized"])),
                **kwargs,
            })
            reference_rows.append({
                "candidate": candidate, "subject_id": str(item["subject_id"]), "outer_fold": int(fold),
                "reference_logrmse": float(reference_metrics["logrmse"]), "model_logrmse": float(metrics["logrmse"]),
                "error_ratio": float(metrics["logrmse"] / reference_metrics["logrmse"]), "model_better": bool(metrics["logrmse"] < reference_metrics["logrmse"]),
                "reference_source": "outer_fold_train_subject_log_geometric_mean",
            })
            for band_index, wave in enumerate(wavelength):
                role = "fit" if fit_mask[band_index] else "edge_diagnostic"
                prediction_rows.append({
                    "candidate": candidate, "subject_id": str(item["subject_id"]), "outer_fold": int(fold), "wavelength_nm": int(wave),
                    "observed_reflectance": float(observed_full[band_index]), "predicted_reflectance": float(prediction_full[band_index]), "band_role": role,
                })
                residual_rows.append({
                    "candidate": candidate, "subject_id": str(item["subject_id"]), "outer_fold": int(fold), "wavelength_nm": int(wave),
                    "signed_residual": float(prediction_full[band_index] - observed_full[band_index]),
                    "log_residual": float(np.log(prediction_full[band_index] + epsilon) - np.log(observed_full[band_index] + epsilon)), "band_role": role,
                })
            for start in fit["starts"]:
                start_rows.append({
                    "candidate": candidate, "subject_id": str(item["subject_id"]), "outer_fold": int(fold), "start_index": int(start["start_index"]),
                    "selected": bool(start["start_index"] == fit["selected_start_index"]), "success": bool(start["success"]), "valid": bool(start["valid"]),
                    "logrmse": float(start["logrmse"]), "cost": float(start["cost"]), "nfev": int(start["nfev"]),
                    "status": int(start["status"]), "optimality": float(start["optimality"]), "initial_theta": start["initial_theta"], "final_theta": start["final_theta"],
                })
    return (pd.DataFrame(metrics_rows), pd.DataFrame(prediction_rows), pd.DataFrame(residual_rows), pd.DataFrame(reference_rows), pd.DataFrame(start_rows))


def _profile_summary(globals_frame: pd.DataFrame, profile_frame: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    limit = int(config["global_identifiability_reporting"]["frequent_boundary_fold_count"])
    rows = []
    for candidate, group in globals_frame.groupby("candidate", sort=False):
        parameters = ["A_s", "delta_bs"] if candidate == "V2R-PS" else ["A_s", "delta_bs", "g0"]
        for parameter in parameters:
            profile = profile_frame.loc[(profile_frame["candidate"].eq(candidate)) & (profile_frame["parameter"].eq(parameter))]
            rows.append({
                "candidate": candidate, "parameter": parameter, "fold_count": int(group["outer_fold"].nunique()),
                "boundary_fold_count": int(group[f"{parameter}_boundary"].sum()),
                "frequently_boundary": bool(int(group[f"{parameter}_boundary"].sum()) >= limit),
                "all_fold_profiles_identifiable": bool(group[f"{parameter}_profile_identifiable"].all()),
                "profile_row_count": int(len(profile)), "profile_source": ";".join(sorted(profile["profile_source"].unique())),
            })
    return pd.DataFrame(rows)


def _select_candidate(summary_frame: pd.DataFrame, config: dict[str, Any]) -> dict[str, Any]:
    ordered = summary_frame.sort_values("median_logrmse", kind="stable").reset_index(drop=True)
    best_value = float(ordered.loc[0, "median_logrmse"])
    tolerance = float(config["selection"]["simpler_candidate_if_absolute_difference_at_most"])
    eligible = ordered.loc[ordered["median_logrmse"].le(best_value + tolerance)].copy()
    complexity = {name: index for index, name in enumerate(config["selection"]["complexity_order"])}
    eligible["complexity_rank"] = eligible["candidate"].map(complexity)
    selected = eligible.sort_values(["complexity_rank", "median_logrmse"], kind="stable").iloc[0]
    return {
        "primary_metric": config["selection"]["primary_metric"], "best_observed_median_logrmse": best_value,
        "tie_tolerance": tolerance, "eligible_candidates": eligible["candidate"].tolist(), "selected_candidate": str(selected["candidate"]),
        "selection_reason": "lowest_median_logrmse" if len(eligible) == 1 else "within_0p005_of_best_select_lower_complexity",
        "selected_median_logrmse": float(selected["median_logrmse"]), "ranking": ordered[["candidate", "median_logrmse"]].to_dict("records"),
    }


def run_stage_c0r(config_path: str | Path, project_root: str | Path) -> dict[str, Any]:
    root = Path(project_root).resolve()
    config_file = Path(config_path).resolve()
    config = yaml.safe_load(config_file.read_text(encoding="utf-8"))
    if config["task_id"] != "R-C0R_RESUME_REGISTERED_CANDIDATE_LADDER" or config["execution"]["stop_after_first_failed_upgrade"]:
        raise ContractError("R-C0R configuration does not declare the resumed non-stopping ladder")
    if config["candidates"]["run"] != ["V2R-PS", "V2R-PSG"] or config["historical_r_c0"]["rerun_candidates"]:
        raise ContractError("R-C0R candidate scope deviates from the registered resume contract")
    if any(config["execution"][key] for key in ("run_r_c1", "run_best_o", "run_validation", "run_test", "run_clinical_500", "run_rgb_encoder_training")):
        raise ContractError("R-C0R configuration attempts to run an out-of-scope stage")
    if any(config["access_policy"][key] for key in ("raw_hsi_content_allowed", "rgb_content_allowed", "validation_content_allowed", "test_content_allowed", "clinical_500_content_allowed")):
        raise ContractError("R-C0R access policy permits prohibited data")
    output = _resolve(root, config["output_directory"])
    if output.exists() and config["access_policy"]["refuse_existing_output"]:
        raise FileExistsError(f"Refusing to overwrite R-C0R output: {output}")

    paths = {name: _resolve(root, value) for name, value in config["inputs"].items()}
    implementation = {name: _resolve(root, value) for name, value in config["implementation"].items()}
    formula_audit = json.loads(paths["formula_audit"].read_text(encoding="utf-8"))
    observation_audit = json.loads(paths["observation_audit"].read_text(encoding="utf-8"))
    preflight = {
        "formula_audit_pass": formula_audit.get("status") == "PASS" and formula_audit.get("hsi_read") is False,
        "observation_audit_pass": observation_audit.get("status") == "PASS_FOR_V2R_TRAIN_INVERSION",
        "observation_manifest_hash": sha256_file(paths["observation_manifest"]) == observation_audit["outputs"]["symmetric_manifest_parquet"]["sha256"],
        "formula_contract_hash_chain": sha256_file(paths["formula_contract"]) == observation_audit["source_hashes"]["v2r_formula_contract"],
        "formula_audit_hash_chain": sha256_file(paths["formula_audit"]) == observation_audit["source_hashes"]["v2r_formula_audit"],
    }
    if not all(preflight.values()):
        raise ContractError(f"R-C0R formula/observation preflight failed: {preflight}")
    historical, historical_hashes = verify_historical_r_c0(root, config)
    manifest = pd.read_parquet(paths["observation_manifest"])
    if len(manifest) != 44 or set(manifest["split"].astype(str)) != {"train"} or not manifest["input_quality_status"].eq("PASS").all():
        raise ContractError("R-B manifest does not satisfy the 44-subject Train-only contract")
    fold_manifest = validate_fold_assignments(manifest["subject_id"].astype(str).tolist(), config["folds"]["assignments"])
    old_fold = historical["folds"].sort_values(["outer_fold", "subject_id"]).reset_index(drop=True)
    fold_equal = old_fold.equals(fold_manifest)
    expected_hash = str(config["folds"]["expected_sha256"])
    fixed_fold_verification = pd.DataFrame([{
        "check": "historical_manifest_sha256", "expected": expected_hash,
        "actual": sha256_file(_history_paths(root, config)["fold_manifest"]), "pass": sha256_file(_history_paths(root, config)["fold_manifest"]) == expected_hash,
    }, {
        "check": "historical_and_v2r1_assignment_content_equal", "expected": "exact_match", "actual": bool(fold_equal), "pass": bool(fold_equal),
    }, {
        "check": "subject_coverage", "expected": 44, "actual": int(fold_manifest["subject_id"].nunique()), "pass": int(fold_manifest["subject_id"].nunique()) == 44,
    }, {
        "check": "fold_sizes", "expected": "9,9,9,9,8", "actual": ",".join(map(str, fold_manifest.groupby("outer_fold").size().tolist())), "pass": sorted(fold_manifest.groupby("outer_fold").size().tolist()) == [8, 9, 9, 9, 9],
    }])
    if not fixed_fold_verification["pass"].all():
        raise ContractError("R-C0R fixed-fold verification failed")
    preflight["historical_r_c0_hashes_pass"] = True
    preflight["fixed_fold_verification_pass"] = True

    wavelength = np.arange(400.0, 701.0, 10.0)
    fit_wavelength = np.asarray(config["analysis"]["fit_centers_nm"], dtype=np.float64)
    fit_mask = np.isin(wavelength, fit_wavelength)
    if fit_mask.sum() != 27 or not np.array_equal(wavelength[fit_mask], fit_wavelength):
        raise ContractError("R-C0R wavelength role contract differs from 420-680 nm")
    optical = load_optical_numpy(wavelength, paths["optical_asset_10nm"])
    _validate_reflectance(manifest[[f"observed_reflectance_{int(value)}nm" for value in wavelength]].to_numpy(dtype=np.float64), "R-B symmetric manifest", require_unit_interval=True)

    output.mkdir(parents=True)
    _write_csv(historical_hashes, output / "historical_r_c0_hash_verification.csv")
    _write_csv(fixed_fold_verification, output / "fixed_fold_verification.csv")
    _write_csv(fold_manifest, output / "fold_manifest.csv")
    fold_hash = sha256_file(output / "fold_manifest.csv")
    (output / "fold_manifest.sha256").write_text(f"{fold_hash}  fold_manifest.csv\n", encoding="ascii")
    if fold_hash != expected_hash:
        raise ContractError("R-C0R emitted fold manifest hash differs from the historical frozen manifest")

    p_metrics = historical["metrics"].loc[historical["metrics"]["candidate"].eq("V2R-P")].copy()
    ps_globals, ps_global_frame, ps_profiles, train_cache, ps_membership = _estimate_ps_globals(
        manifest, fold_manifest, wavelength, fit_mask, optical, p_metrics, config
    )
    ps_metrics, ps_predictions, ps_residuals, ps_reference, ps_starts = _fit_oof_candidate(
        "V2R-PS", manifest, fold_manifest, wavelength, fit_mask, optical, ps_globals, config
    )
    psg_globals, psg_global_frame, psg_profiles, psg_membership = _estimate_psg_globals(
        manifest, fold_manifest, wavelength, fit_mask, optical, ps_globals, train_cache, ps_profiles, config
    )
    psg_metrics, psg_predictions, psg_residuals, psg_reference, psg_starts = _fit_oof_candidate(
        "V2R-PSG", manifest, fold_manifest, wavelength, fit_mask, optical, psg_globals, config
    )
    new_metrics = pd.concat([ps_metrics, psg_metrics], ignore_index=True)
    new_predictions = pd.concat([ps_predictions, psg_predictions], ignore_index=True)
    new_residuals = pd.concat([ps_residuals, psg_residuals], ignore_index=True)
    new_reference = pd.concat([ps_reference, psg_reference], ignore_index=True)
    new_starts = pd.concat([ps_starts, psg_starts], ignore_index=True)
    global_frame = pd.concat([ps_global_frame, psg_global_frame], ignore_index=True)
    profile_frame = pd.concat([ps_profiles, psg_profiles], ignore_index=True)
    membership_frame = pd.concat([ps_membership, psg_membership], ignore_index=True)
    global_summary = _profile_summary(global_frame, profile_frame, config)
    if set(new_metrics["candidate"]) != {"V2R-PS", "V2R-PSG"} or len(new_metrics) != 88:
        raise ContractError("R-C0R did not produce complete PS/PSG OOF metric rows")
    if not new_metrics["solver_converged"].all() or not np.isfinite(new_metrics[["logrmse", "rmse", "sam_deg"]].to_numpy()).all():
        raise ContractError("R-C0R produced invalid individual OOF metrics")

    # References are deterministic from the fixed outer training folds.  The
    # historical values must agree even though V2R-0/P are not re-fitted.
    historical_reference = historical["reference"].copy()
    reference_check_rows = []
    for candidate in ("V2R-PS", "V2R-PSG"):
        joined = new_reference.loc[new_reference["candidate"].eq(candidate)].merge(
            historical_reference.loc[historical_reference["candidate"].eq("V2R-P")][["subject_id", "outer_fold", "reference_logrmse"]],
            on=["subject_id", "outer_fold"], suffixes=("_new", "_historical"), validate="one_to_one"
        )
        maximum = float(np.max(np.abs(joined["reference_logrmse_new"] - joined["reference_logrmse_historical"])))
        reference_check_rows.append({"candidate": candidate, "maximum_abs_outer_fold_reference_logrmse_difference": maximum, "pass": maximum <= 1e-12})
    reference_check = pd.DataFrame(reference_check_rows)
    if not reference_check["pass"].all():
        raise ContractError("R-C0R outer-fold reference spectra differ from historical fixed references")

    four_metrics = pd.concat([historical["metrics"], new_metrics], ignore_index=True, sort=False)
    four_predictions = pd.concat([historical["predictions"], new_predictions], ignore_index=True, sort=False)
    four_residuals = pd.concat([historical["residuals"], new_residuals], ignore_index=True, sort=False)
    four_reference = pd.concat([historical["reference"], new_reference], ignore_index=True, sort=False)
    summary_rows = []
    for candidate in config["selection"]["complexity_order"]:
        metric = four_metrics.loc[four_metrics["candidate"].eq(candidate)].copy()
        residual = four_residuals.loc[(four_residuals["candidate"].eq(candidate)) & (four_residuals["band_role"].eq("fit"))].copy()
        if len(metric) != 44 or len(residual) != 44 * 27:
            raise ContractError(f"Four-candidate aggregation is incomplete for {candidate}")
        summary = _spectral_summary(metric, residual)
        summary["candidate"] = candidate
        summary["strong_spectral_checks"] = _strong_target_checks(summary, config["strong_spectral_targets"])
        summary["strong_spectral_target_pass"] = bool(all(summary["strong_spectral_checks"].values()))
        summary_rows.append(summary)
    summary_frame = pd.DataFrame(summary_rows)
    pair_rows = []
    order = config["selection"]["complexity_order"]
    for left_index, left in enumerate(order):
        for right in order[left_index + 1:]:
            joined = four_metrics.loc[four_metrics["candidate"].eq(left), ["subject_id", "logrmse"]].merge(
                four_metrics.loc[four_metrics["candidate"].eq(right), ["subject_id", "logrmse"]], on="subject_id", suffixes=("_left", "_right"), validate="one_to_one"
            )
            pair_rows.append({
                "left_candidate": left, "right_candidate": right,
                "median_logrmse_left": float(joined["logrmse_left"].median()), "median_logrmse_right": float(joined["logrmse_right"].median()),
                "right_minus_left_median_logrmse": float(joined["logrmse_right"].median() - joined["logrmse_left"].median()),
                "right_better_subject_fraction": float((joined["logrmse_right"] < joined["logrmse_left"]).mean()),
            })
    pair_frame = pd.DataFrame(pair_rows)
    selection = _select_candidate(summary_frame, config)

    outputs: dict[str, pd.DataFrame] = {
        "ps_psg_subject_metrics.csv": new_metrics,
        "ps_psg_multistart_results.csv": new_starts,
        "ps_psg_oof_predictions.csv": new_predictions,
        "ps_psg_oof_residuals.csv": new_residuals,
        "ps_psg_outer_fold_reference_metrics.csv": new_reference,
        "train_global_membership.csv": membership_frame,
        "fold_global_parameters.csv": global_frame,
        "global_parameter_profiles.csv": profile_frame,
        "global_profile_summary.csv": global_summary,
        "outer_fold_reference_reproduction_check.csv": reference_check,
        "four_candidate_subject_metrics.csv": four_metrics,
        "four_candidate_oof_predictions.csv": four_predictions,
        "four_candidate_oof_residuals.csv": four_residuals,
        "four_candidate_outer_fold_reference_metrics.csv": four_reference,
        "four_candidate_summary.csv": summary_frame,
        "four_candidate_pairwise_comparisons.csv": pair_frame,
    }
    for name, frame in outputs.items():
        _write_csv(frame, output / name)
    selected = selection["selected_candidate"]
    decision = {
        "schema_version": 1, "task_id": config["task_id"], "protocol_version": config["protocol_version"],
        "created_utc": datetime.now(timezone.utc).isoformat(), "status": "R_C0R_COMPLETE",
        "historical_candidates_reused_read_only": config["historical_r_c0"]["reuse_candidates"], "new_candidates_executed": config["candidates"]["run"],
        "selection": selection, "selected_candidate": selected,
        "candidate_summaries": summary_rows, "strong_spectral_targets_are_diagnostic": True,
        "global_identifiability_is_diagnostic": True, "r_c1_executed": False, "best_o_executed": False,
        "validation_executed": False, "test_executed": False, "clinical_500_executed": False, "rgb_encoder_training_executed": False,
        "next_registered_stage": "R-C1", "interpretation": "R-C0R completion is a four-candidate Train-only comparison, not a Validation result or physiological-parameter claim.",
    }
    (output / "candidate_selection.json").write_text(json.dumps(_jsonable(selection), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (output / "r_c0r_decision.json").write_text(json.dumps(_jsonable(decision), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    audit = {
        "schema_version": 1, "task_id": config["task_id"], "protocol_version": config["protocol_version"], "status": "R_C0R_COMPLETE",
        "created_utc": decision["created_utc"], "preflight_checks": preflight, "fold_manifest_sha256": fold_hash,
        "historical_artifact_hash_entries": int(len(historical_hashes)), "historical_artifact_hash_failures": int((~historical_hashes["matches"]).sum()),
        "reference_reproduction_checks": reference_check.to_dict("records"),
        "data_access": {"r_b_manifest_content_reads": 1, "historical_r_c0_artifact_content_reads": 8, "raw_hsi_reads": 0, "rgb_reads": 0, "validation_reads": 0, "test_reads": 0, "clinical_500_reads": 0},
        "reflectance_clipping_applied": False,
        "input_hashes": {name: sha256_file(path) for name, path in paths.items()},
        "implementation_hashes": {name: sha256_file(path) for name, path in implementation.items()},
    }
    (output / "audit_summary.json").write_text(json.dumps(_jsonable(audit), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    report = [
        "# KM-BIO-v2R.1 R-C0R resumed candidate ladder", "", "- Status: `R_C0R_COMPLETE`",
        "- Historical candidates reused read-only: `V2R-0`, `V2R-P`", "- Newly executed candidates: `V2R-PS`, `V2R-PSG`",
        f"- Selected candidate: `{selected}`", f"- Frozen fold SHA-256: `{fold_hash}`", "",
        "| Candidate | Median logRMSE | P90 logRMSE | Median RMSE | Median SAM deg | Better than reference | Median model/reference ratio | Strong targets |", "|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    for _, row in summary_frame.iterrows():
        report.append(
            f"| {row['candidate']} | {row['median_logrmse']:.6f} | {row['p90_logrmse']:.6f} | {row['median_rmse']:.6f} | {row['median_sam_deg']:.3f} | {row['model_better_than_reference_fraction']:.3f} | {row['median_model_to_reference_error_ratio']:.3f} | {row['strong_spectral_target_pass']} |"
        )
    report.extend(["", "Global-profile width and boundary flags are reported in `global_profile_summary.csv`; they are diagnostic under v2R.1 and did not stop PSG.", ""])
    (output / "R_C0R_REPORT.md").write_text("\n".join(report), encoding="utf-8")

    artifact_rows = []
    for name, path in paths.items():
        artifact_rows.append({"role": "input", "name": name, "path": str(path), "sha256": sha256_file(path)})
    for name, path in _history_paths(root, config).items():
        if path.is_file():
            artifact_rows.append({"role": "historical_read_only_input", "name": name, "path": str(path), "sha256": sha256_file(path)})
    artifact_rows.append({"role": "config", "name": "r_c0r_config", "path": str(config_file), "sha256": sha256_file(config_file)})
    for name, path in implementation.items():
        artifact_rows.append({"role": "implementation", "name": name, "path": str(path), "sha256": sha256_file(path)})
    for path in sorted(output.iterdir(), key=lambda item: item.name):
        if path.is_file() and path.name != "artifact_hash_manifest.csv":
            artifact_rows.append({"role": "output", "name": path.name, "path": str(path), "sha256": sha256_file(path)})
    _write_csv(pd.DataFrame(artifact_rows), output / "artifact_hash_manifest.csv")
    return decision
