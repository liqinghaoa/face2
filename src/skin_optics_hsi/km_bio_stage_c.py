"""Train-only Stage C execution for the KM-BIO-v1 observation contract."""

from __future__ import annotations

import json
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd
import yaml
from scipy.stats import spearmanr

from .km_bio_inverse import (
    FitSettings,
    acceptable_intervals,
    fit_bounded_spectrum,
    log_jacobian_u,
    profile_parameter,
    sobol_plus_center,
    spectral_metrics,
)
from .km_bio_observation import sha256_file
from .km_bio_v1 import forward_preloaded_numpy, load_optical_numpy


def _resolve(root: Path, value: str) -> Path:
    path = Path(value)
    return (root / path).resolve() if not path.is_absolute() else path.resolve()


def _json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def _write_frame(frame: pd.DataFrame, parquet: Path, csv: Path) -> None:
    frame.to_parquet(parquet, index=False)
    export = frame.copy()
    for column in export.columns:
        if export[column].dtype == object:
            export[column] = export[column].map(lambda value: _json(value) if isinstance(value, (list, dict)) else value)
    export.to_csv(csv, index=False, encoding="utf-8-sig")


def _fit_settings(raw: dict[str, Any]) -> FitSettings:
    return FitSettings(
        epsilon=float(raw["reflectance_epsilon"]),
        sobol_starts=int(raw["sobol_starts"]),
        seed=int(raw["random_seed"]),
        ftol=float(raw["ftol"]),
        xtol=float(raw["xtol"]),
        gtol=float(raw["gtol"]),
        max_nfev=int(raw["max_nfev_per_start"]),
    )


class GaussianForward:
    def __init__(self, optical: dict[str, np.ndarray], fine_wavelength: np.ndarray, centers: np.ndarray, fwhm_nm: float, support_sigma: float) -> None:
        self.optical = optical
        self.centers = centers
        sigma = fwhm_nm / (2.0 * np.sqrt(2.0 * np.log(2.0)))
        weights = []
        for center in centers:
            row = np.exp(-0.5 * ((fine_wavelength - center) / sigma) ** 2)
            row[np.abs(fine_wavelength - center) > support_sigma * sigma] = 0.0
            row /= row.sum()
            weights.append(row)
        self.weights = np.asarray(weights, dtype=np.float64)

    def __call__(self, theta: np.ndarray, *, thickness: float = 0.060, scattering: float = 1.0, hb: float = 150.0) -> np.ndarray:
        fine = forward_preloaded_numpy(theta, self.optical, thickness, scattering, hb)
        return self.weights @ fine


def _main_row(item: pd.Series, fit: dict[str, Any], parameter_names: list[str], wavelength: np.ndarray, profile_updated: bool) -> dict[str, Any]:
    row: dict[str, Any] = {
        "subject_id": item["subject_id"], "capture_id": item["capture_id"], "roi": item["roi"],
        "solver_status": "CONVERGED" if fit["success"] else "FAILED",
        "selected_start_index": fit.get("selected_start_index", -1),
        "profile_updated_main_solution": profile_updated,
    }
    if not fit["success"]:
        row.update({name: np.nan for name in parameter_names})
        row.update({"logrmse": np.nan, "rmse": np.nan, "sam_deg": np.nan})
        return row
    row.update({name: float(value) for name, value in zip(parameter_names, fit["theta"])})
    row.update(fit["metrics"])
    for index, value in enumerate(fit["prediction"]):
        row[f"reflectance_hat_{int(wavelength[index])}nm"] = float(value)
        row[f"residual_{int(wavelength[index])}nm"] = float(fit["residual"][index])
    return row


def _start_rows(item: pd.Series, fit: dict[str, Any], parameter_names: list[str]) -> list[dict[str, Any]]:
    rows = []
    for start in fit["starts"]:
        row = {
            "subject_id": item["subject_id"], "capture_id": item["capture_id"], "roi": item["roi"],
            "start_index": start["start_index"], "logrmse": start["logrmse"], "cost": start["cost"],
            "success": start["success"], "valid": start["valid"], "status": start["status"],
            "nfev": start["nfev"], "optimality": start["optimality"], "message": start["message"],
        }
        for index, name in enumerate(parameter_names):
            row[f"initial_{name}"] = start["initial_theta"][index]
            row[f"final_{name}"] = start["final_theta"][index]
        rows.append(row)
    return rows


def _profile_outputs(
    item: pd.Series,
    observed: np.ndarray,
    forward_theta: Callable[[np.ndarray], np.ndarray],
    fit: dict[str, Any],
    lower: np.ndarray,
    upper: np.ndarray,
    parameter_names: list[str],
    settings: FitSettings,
    profile_cfg: dict[str, Any],
    seed_base: int,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], bool]:
    span = upper - lower

    def forward_u(u: np.ndarray) -> np.ndarray:
        return forward_theta(lower + np.asarray(u) * span)

    all_profiles: list[list[dict[str, Any]]] = []
    all_start_rows: list[list[dict[str, Any]]] = []
    for index in range(3):
        profile, starts = profile_parameter(
            observed, forward_u, fit["u"], index, settings, seed_base + index * 10000,
            main_delta=float(profile_cfg["main_delta_logrmse"]),
            refinement_width_u=float(profile_cfg["refinement_width_scaled"]),
        )
        all_profiles.append(profile)
        all_start_rows.append(starts)
    profile_best = min(row["logrmse"] for profile in all_profiles for row in profile if row["success"])
    updated = False
    if profile_best < fit["metrics"]["logrmse"] - 1e-9:
        best_profile = min((row for profile in all_profiles for row in profile if row["success"]), key=lambda row: row["logrmse"])
        starts = np.vstack([sobol_plus_center(3, settings.sobol_starts, settings.seed), best_profile["conditional_u"]])
        fit = fit_bounded_spectrum(observed, forward_theta, lower, upper, settings, initial_u=starts)
        updated = True
        all_profiles, all_start_rows = [], []
        for index in range(3):
            profile, profile_starts = profile_parameter(
                observed, forward_u, fit["u"], index, settings, seed_base + 50000 + index * 10000,
                main_delta=float(profile_cfg["main_delta_logrmse"]),
                refinement_width_u=float(profile_cfg["refinement_width_scaled"]),
            )
            all_profiles.append(profile)
            all_start_rows.append(profile_starts)

    jacobian = log_jacobian_u(forward_theta, fit["u"], lower, upper, settings.epsilon)
    profile_rows: list[dict[str, Any]] = []
    start_rows: list[dict[str, Any]] = []
    reliability_rows: list[dict[str, Any]] = []
    deltas = [float(profile_cfg["main_delta_logrmse"]), *[float(v) for v in profile_cfg["sensitivity_delta_logrmse"]]]
    global_optimum = min(
        fit["metrics"]["logrmse"],
        min(row["logrmse"] for profile in all_profiles for row in profile if row["success"]),
    )
    for index, name in enumerate(parameter_names):
        profile = all_profiles[index]
        intervals_by_delta = {}
        for delta in deltas:
            result = acceptable_intervals(profile, global_optimum, delta, span[index])
            accepted_u = [
                float(row["conditional_u"][index])
                for candidate_profile in all_profiles
                for row in candidate_profile
                if row["success"] and row["logrmse"] <= global_optimum + delta
            ]
            accepted_u.extend(
                float(row["final_u_full"][index])
                for candidate_starts in all_start_rows
                for row in candidate_starts
                if row["valid"] and row["logrmse"] <= global_optimum + delta
            )
            if accepted_u:
                result["accepted_solution_min"] = lower[index] + min(accepted_u) * span[index]
                result["accepted_solution_max"] = lower[index] + max(accepted_u) * span[index]
                result["connected_interval_total_span"] = result["total_span"]
                result["total_span"] = result["accepted_solution_max"] - result["accepted_solution_min"]
                result["envelope_span"] = result["total_span"]
                result["accepted_solution_count"] = len(accepted_u)
            intervals_by_delta[str(delta)] = result
        for row in profile:
            conditional_u = row.get("conditional_u")
            profile_rows.append({
                "subject_id": item["subject_id"], "capture_id": item["capture_id"], "roi": item["roi"],
                "profiled_parameter": name, "fixed_u": row["fixed_u"],
                "fixed_value": lower[index] + row["fixed_u"] * span[index], "logrmse": row["logrmse"],
                "success": row["success"], "grid_kind": row["grid_kind"],
                "conditional_theta": None if conditional_u is None else (lower + np.asarray(conditional_u) * span).tolist(),
            })
        for row in all_start_rows[index]:
            full = np.asarray(row["final_u_full"])
            start_rows.append({
                "subject_id": item["subject_id"], "capture_id": item["capture_id"], "roi": item["roi"],
                "profiled_parameter": name, "fixed_u": row["fixed_u"], "grid_kind": row["grid_kind"],
                "start_index": row["start_index"], "logrmse": row["logrmse"], "success": row["success"],
                "valid": row["valid"], "status": row["status"], "nfev": row["nfev"],
                "final_theta": (lower + full * span).tolist(),
            })
        main_intervals = intervals_by_delta[str(float(profile_cfg["main_delta_logrmse"]))]
        reliability_rows.append({
            "subject_id": item["subject_id"], "capture_id": item["capture_id"], "roi": item["roi"],
            "parameter": name, "theta_hat": float(fit["theta"][index]),
            "profile_optimum_logrmse": float(global_optimum),
            "intervals_delta_0p005": main_intervals["intervals"],
            "total_span_delta_0p005": main_intervals["total_span"],
            "envelope_span_delta_0p005": main_intervals["envelope_span"],
            "intervals_delta_0p0025": intervals_by_delta["0.0025"]["intervals"],
            "intervals_delta_0p010": intervals_by_delta["0.01"]["intervals"],
            "boundary_limited": bool(fit["u"][index] < 0.01 or fit["u"][index] > 0.99),
            "jacobian_singular_values": jacobian["singular_values"],
            "jacobian_column_correlation": jacobian["column_correlation"],
        })
    return fit, profile_rows, start_rows, reliability_rows, updated


def run_stage_c(config_path: str | Path, project_root: str | Path) -> dict[str, Any]:
    root = Path(project_root).resolve()
    config_file = Path(config_path).resolve()
    config = yaml.safe_load(config_file.read_text(encoding="utf-8"))
    paths = {name: _resolve(root, value) for name, value in config["inputs"].items()}
    output = _resolve(root, config["output_directory"])
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite Stage C output: {output}")
    observation_audit = json.loads(paths["observation_audit"].read_text(encoding="utf-8"))
    if observation_audit.get("status") != "PASS_FOR_TRAIN_INVERSION" or not observation_audit.get("next_stage_allowed"):
        raise ValueError("Stage B does not authorize Train inversion")
    if sha256_file(paths["observation_manifest"]) != observation_audit["outputs"]["input_manifest_parquet"]["sha256"]:
        raise ValueError("Observation manifest hash differs from Stage B audit")
    manifest = pd.read_parquet(paths["observation_manifest"])
    if set(manifest["split"]) != {config["access_policy"]["allowed_split"]} or not manifest["input_quality_status"].eq("PASS").all():
        raise ValueError("Stage C received an unauthorized split or failed observation")

    parameter_names = list(config["parameters"]["names"])
    lower = np.asarray(config["parameters"]["lower"], dtype=np.float64)
    upper = np.asarray(config["parameters"]["upper"], dtype=np.float64)
    span = upper - lower
    wavelength = np.arange(400.0, 701.0, 10.0)
    observed_columns = [f"observed_reflectance_{int(value)}nm" for value in wavelength]
    optical = load_optical_numpy(wavelength, paths["optical_asset_10nm"])
    settings = _fit_settings(config["solver"])

    def nominal(theta: np.ndarray) -> np.ndarray:
        return forward_preloaded_numpy(theta, optical)

    main_fits: dict[tuple[str, str], dict[str, Any]] = {}
    profile_rows: list[dict[str, Any]] = []
    profile_start_rows: list[dict[str, Any]] = []
    reliability_rows: list[dict[str, Any]] = []
    profile_updated: dict[tuple[str, str], bool] = {}
    for row_index, item in manifest.iterrows():
        observed = item[observed_columns].to_numpy(dtype=np.float64)
        fit = fit_bounded_spectrum(observed, nominal, lower, upper, settings)
        if fit["success"]:
            fit, profiles, starts, reliability, updated = _profile_outputs(
                item, observed, nominal, fit, lower, upper, parameter_names, settings,
                config["profile"], settings.seed + row_index * 100000,
            )
            profile_rows.extend(profiles)
            profile_start_rows.extend(starts)
            reliability_rows.extend(reliability)
            profile_updated[(str(item["capture_id"]), str(item["roi"]))] = updated
        main_fits[(str(item["capture_id"]), str(item["roi"]))] = fit
        print(f"Stage C main/profile {row_index + 1}/{len(manifest)}", flush=True)

    main_rows: list[dict[str, Any]] = []
    main_start_rows: list[dict[str, Any]] = []
    for _, item in manifest.iterrows():
        fit = main_fits[(str(item["capture_id"]), str(item["roi"]))]
        main_rows.append(_main_row(item, fit, parameter_names, wavelength, profile_updated.get((str(item["capture_id"]), str(item["roi"])), False)))
        main_start_rows.extend(_start_rows(item, fit, parameter_names))
    main = pd.DataFrame(main_rows)
    reliability = pd.DataFrame(reliability_rows)

    single_gate = config["spectral_gates"]
    main["single_spectrum_gate"] = (
        main["solver_status"].eq("CONVERGED")
        & main["logrmse"].le(float(single_gate["single_spectrum_logrmse_max"]))
        & main["rmse"].le(float(single_gate["single_spectrum_rmse_max"]))
        & main["sam_deg"].le(float(single_gate["single_spectrum_sam_deg_max"]))
    )
    reliability = reliability.merge(main[["capture_id", "roi", "single_spectrum_gate", "f_blood"]], on=["capture_id", "roi"], how="left", validate="many_to_one")
    span_limits = dict(zip(parameter_names, config["parameters"]["profile_span_max"]))
    reliability["low_blood_signal"] = reliability["parameter"].eq("s") & reliability["f_blood"].le(float(config["parameters"]["low_blood_threshold"]))
    reliability["profile_gate"] = reliability.apply(lambda row: row["total_span_delta_0p005"] <= float(span_limits[row["parameter"]]), axis=1)
    reliability["base_parameter_reliable"] = reliability["single_spectrum_gate"] & ~reliability["boundary_limited"] & ~reliability["low_blood_signal"] & reliability["profile_gate"]

    # Subject-level aggregation and leave-one-subject-out fixed reference control.
    subject_rows: list[dict[str, Any]] = []
    loo_rows: list[dict[str, Any]] = []
    subject_observed = {}
    for subject, group in manifest.groupby("subject_id"):
        subject_observed[subject] = np.mean(group[observed_columns].to_numpy(dtype=np.float64), axis=0)
    for subject, group in main.groupby("subject_id"):
        item = {"subject_id": subject, "n_cheeks": len(group)}
        for metric in ("logrmse", "rmse", "sam_deg"):
            item[metric] = float(group[metric].mean())
        for wave in wavelength:
            item[f"signed_residual_{int(wave)}nm"] = float(group[f"residual_{int(wave)}nm"].mean())
        others = [value for key, value in subject_observed.items() if key != subject]
        reference = np.exp(np.mean(np.log(np.asarray(others) + settings.epsilon), axis=0)) - settings.epsilon
        cheek_errors = []
        for _, observation in manifest.loc[manifest["subject_id"].eq(subject)].iterrows():
            values = observation[observed_columns].to_numpy(dtype=np.float64)
            cheek_errors.append(spectral_metrics(reference, values, settings.epsilon)["logrmse"])
        loo_error = float(np.mean(cheek_errors))
        ratio = item["logrmse"] / loo_error
        item.update({"loo_reference_logrmse": loo_error, "model_to_loo_error_ratio": ratio, "model_better_than_loo": item["logrmse"] < loo_error})
        subject_rows.append(item)
        loo_rows.append({"subject_id": subject, "loo_reference_logrmse": loo_error, "model_logrmse": item["logrmse"], "error_ratio": ratio, "model_better": item["model_better_than_loo"]})
    subject_summary = pd.DataFrame(subject_rows)
    loo = pd.DataFrame(loo_rows)

    # Fixed-assumption and observation perturbations.
    sensitivity_specs: list[dict[str, Any]] = []
    for value in config["sensitivity"]["epidermis_thickness_mm"]:
        sensitivity_specs.append({"setting": f"epidermis_{value:.3f}mm", "kind": "thickness", "value": float(value)})
    for value in config["sensitivity"]["scattering_scale"]:
        sensitivity_specs.append({"setting": f"scattering_x{value:.1f}", "kind": "scattering", "value": float(value)})
    for value in config["sensitivity"]["whole_blood_hb_g_l"]:
        sensitivity_specs.append({"setting": f"whole_blood_hb_{int(value)}gL", "kind": "hb", "value": float(value)})
    for value in config["sensitivity"]["observation_scale"]:
        sensitivity_specs.append({"setting": f"observation_scale_{value:.2f}", "kind": "observation_scale", "value": float(value)})
    for value in config["sensitivity"]["observation_tilt"]:
        sensitivity_specs.append({"setting": f"observation_tilt_{value:+.2f}", "kind": "observation_tilt", "value": float(value)})
    subset = (wavelength >= float(config["sensitivity"]["bandwidth_subset_nm"][0])) & (wavelength <= float(config["sensitivity"]["bandwidth_subset_nm"][1]))
    sensitivity_specs.append({"setting": "subset_420_680_point", "kind": "subset_point", "value": 0.0})
    fine_wavelength = np.arange(400.0, 701.0, float(config["sensitivity"]["gaussian_grid_step_nm"]))
    fine_optical = load_optical_numpy(fine_wavelength, paths["optical_asset_1nm"])
    gaussian = {
        float(value): GaussianForward(fine_optical, fine_wavelength, wavelength[subset], float(value), float(config["sensitivity"]["gaussian_support_sigma"]))
        for value in config["sensitivity"]["gaussian_fwhm_nm"]
    }
    for value in config["sensitivity"]["gaussian_fwhm_nm"]:
        sensitivity_specs.append({"setting": f"subset_420_680_gaussian_{int(value)}nm", "kind": "gaussian", "value": float(value)})

    sensitivity_rows: list[dict[str, Any]] = []
    for row_index, item in manifest.iterrows():
        observed_full = item[observed_columns].to_numpy(dtype=np.float64)
        base = main_fits[(str(item["capture_id"]), str(item["roi"]))]
        for spec in sensitivity_specs:
            kind, value = spec["kind"], spec["value"]
            observed_fit = observed_full
            forward_fit: Callable[[np.ndarray], np.ndarray] = nominal
            if kind == "thickness":
                forward_fit = lambda theta, v=value: forward_preloaded_numpy(theta, optical, epidermis_thickness_mm=v)
            elif kind == "scattering":
                forward_fit = lambda theta, v=value: forward_preloaded_numpy(theta, optical, scattering_scale=v)
            elif kind == "hb":
                forward_fit = lambda theta, v=value: forward_preloaded_numpy(theta, optical, whole_blood_hb_g_l=v)
            elif kind == "observation_scale":
                observed_fit = observed_full * value
            elif kind == "observation_tilt":
                observed_fit = observed_full * (1.0 + value * (wavelength - 550.0) / 150.0)
            elif kind == "subset_point":
                observed_fit = observed_full[subset]
                subset_optical = {name: array[subset] for name, array in optical.items()}
                forward_fit = lambda theta, values=subset_optical: forward_preloaded_numpy(theta, values)
            elif kind == "gaussian":
                observed_fit = observed_full[subset]
                forward_fit = gaussian[value]
            fit = fit_bounded_spectrum(observed_fit, forward_fit, lower, upper, settings)
            result = {
                "subject_id": item["subject_id"], "capture_id": item["capture_id"], "roi": item["roi"],
                "setting": spec["setting"], "kind": kind, "value": value,
                "success": fit["success"], "valid_start_count": sum(row["valid"] for row in fit["starts"]),
            }
            if fit["success"]:
                for index, name in enumerate(parameter_names):
                    result[name] = float(fit["theta"][index])
                    result[f"shift_{name}"] = float(fit["theta"][index] - base["theta"][index])
                    result[f"boundary_{name}"] = bool(fit["u"][index] < 0.01 or fit["u"][index] > 0.99)
                result.update(fit["metrics"])
                result["hb_absorption_content_scale"] = float(value * fit["theta"][1]) if kind == "hb" else np.nan
            sensitivity_rows.append(result)
        print(f"Stage C sensitivity {row_index + 1}/{len(manifest)}", flush=True)
    sensitivity = pd.DataFrame(sensitivity_rows)

    # Free gain is diagnostic only and never replaces the main fit.
    gain_rows = []
    gain_lower = np.append(lower, float(config["free_gain_diagnostic"]["lower"]))
    gain_upper = np.append(upper, float(config["free_gain_diagnostic"]["upper"]))
    for row_index, item in manifest.iterrows():
        observed = item[observed_columns].to_numpy(dtype=np.float64)
        fit = fit_bounded_spectrum(observed, lambda theta: theta[3] * nominal(theta[:3]), gain_lower, gain_upper, settings)
        base = main_fits[(str(item["capture_id"]), str(item["roi"]))]
        row = {"subject_id": item["subject_id"], "capture_id": item["capture_id"], "roi": item["roi"], "success": fit["success"]}
        if fit["success"]:
            row.update({"f_mel": fit["theta"][0], "f_blood": fit["theta"][1], "s": fit["theta"][2], "gain": fit["theta"][3], **fit["metrics"], "main_logrmse": base["metrics"]["logrmse"], "logrmse_ratio_to_main": fit["metrics"]["logrmse"] / base["metrics"]["logrmse"]})
        gain_rows.append(row)
    gain = pd.DataFrame(gain_rows)

    # Per-parameter robustness and side-confounding summaries.
    robustness_rows = []
    sensitivity_limits = dict(zip(parameter_names, config["parameters"]["robustness_shift_max"]))
    for parameter in parameter_names:
        base_parameter = reliability.loc[reliability["parameter"].eq(parameter), ["capture_id", "roi", "base_parameter_reliable"]]
        merged = sensitivity.merge(base_parameter, on=["capture_id", "roi"], how="left", validate="many_to_one")
        originally = merged.loc[merged["base_parameter_reliable"]].copy()
        originally["setting_stable"] = originally["success"] & ~originally[f"boundary_{parameter}"].fillna(True) & originally[f"shift_{parameter}"].abs().le(float(sensitivity_limits[parameter]))
        per_roi = originally.groupby(["capture_id", "roi"])["setting_stable"].all() if len(originally) else pd.Series(dtype=bool)
        stable_fraction = float(per_roi.mean()) if len(per_roi) else 0.0
        for setting, group in merged.groupby("setting"):
            valid = group["success"] & group[parameter].notna()
            rho = float(spearmanr(group.loc[valid, parameter], main.loc[valid.to_numpy(), parameter]).statistic) if int(valid.sum()) >= 3 else np.nan
            robustness_rows.append({
                "parameter": parameter, "setting": setting, "n_success": int(valid.sum()),
                "median_abs_shift": float(group.loc[valid, f"shift_{parameter}"].abs().median()) if valid.any() else np.nan,
                "p90_abs_shift": float(group.loc[valid, f"shift_{parameter}"].abs().quantile(0.9)) if valid.any() else np.nan,
                "spearman_rank": rho,
                "stable_coverage_all_expected": float((valid & ~group[f"boundary_{parameter}"].fillna(True) & group[f"shift_{parameter}"].abs().le(float(sensitivity_limits[parameter]))).mean()),
                "originally_reliable_all_settings_stable_fraction": stable_fraction,
            })
    robustness = pd.DataFrame(robustness_rows)

    side_rows = []
    side = manifest[["subject_id", "roi", *observed_columns]].merge(main[["subject_id", "roi", *parameter_names]], on=["subject_id", "roi"], validate="one_to_one")
    left = side.loc[side["roi"].eq("left_cheek")].set_index("subject_id")
    right = side.loc[side["roi"].eq("right_cheek")].set_index("subject_id")
    brightness_difference = left[observed_columns].mean(axis=1) - right[observed_columns].mean(axis=1)
    confounding_flags = {}
    for index, parameter in enumerate(parameter_names):
        difference = left[parameter] - right[parameter]
        rho = float(spearmanr(brightness_difference, difference).statistic)
        consistency = float(max((difference > 0).mean(), (difference < 0).mean()))
        material = abs(float(difference.median())) >= float(config["observation_confounding"]["material_side_difference"][index])
        flag = bool(abs(rho) >= float(config["observation_confounding"]["brightness_parameter_spearman_abs"]) or (consistency >= float(config["observation_confounding"]["parameter_side_consistency_fraction"]) and material))
        confounding_flags[parameter] = flag
        side_rows.append({"parameter": parameter, "spearman_brightness_vs_parameter_difference": rho, "same_direction_fraction": consistency, "median_image_left_minus_right": float(difference.median()), "median_abs_difference": float(difference.abs().median()), "observation_confounding": flag})
    side_summary = pd.DataFrame(side_rows)

    # Pre-registered gates and final Stage C decision.
    residual_columns = [f"signed_residual_{int(value)}nm" for value in wavelength]
    gates = {
        "all_input_spectra_converged": bool(main["solver_status"].eq("CONVERGED").all()),
        "input_coverage": float(len(main) / int(observation_audit["counts"]["region_spectra"])) >= float(single_gate["input_coverage_min"]),
        "subject_median_logrmse": float(subject_summary["logrmse"].median()) <= float(single_gate["subject_median_logrmse_max"]),
        "subject_p90_logrmse": float(subject_summary["logrmse"].quantile(0.9)) <= float(single_gate["subject_p90_logrmse_max"]),
        "subject_median_rmse": float(subject_summary["rmse"].median()) <= float(single_gate["subject_median_rmse_max"]),
        "subject_median_sam": float(subject_summary["sam_deg"].median()) <= float(single_gate["subject_median_sam_deg_max"]),
        "maximum_abs_median_signed_band_residual": float(subject_summary[residual_columns].median().abs().max()) <= float(single_gate["maximum_abs_median_signed_band_residual"]),
        "loo_better_subject_fraction": float(subject_summary["model_better_than_loo"].mean()) >= float(single_gate["loo_better_subject_fraction_min"]),
        "loo_median_error_ratio": float(subject_summary["model_to_loo_error_ratio"].median()) <= float(single_gate["loo_median_error_ratio_max"]),
    }
    spectral_pass = bool(all(gates.values()))
    parameter_summary_rows = []
    parameter_ready = {}
    for parameter in parameter_names:
        group = reliability.loc[reliability["parameter"].eq(parameter)]
        base_coverage = float(group["base_parameter_reliable"].mean())
        robust_fraction = float(robustness.loc[robustness["parameter"].eq(parameter), "originally_reliable_all_settings_stable_fraction"].iloc[0])
        robust_pass = robust_fraction >= float(config["sensitivity"]["required_stable_fraction"])
        ready = bool(base_coverage >= 0.80 and robust_pass)
        parameter_ready[parameter] = ready
        parameter_summary_rows.append({"parameter": parameter, "base_reliable_count": int(group["base_parameter_reliable"].sum()), "expected_count": len(group), "base_reliable_coverage": base_coverage, "all_sensitivity_stable_fraction_among_reliable": robust_fraction, "robustness_gate": robust_pass, "parameter_ready": ready})
    parameter_summary = pd.DataFrame(parameter_summary_rows)
    primary_confounding = bool(confounding_flags["f_mel"] or confounding_flags["f_blood"])
    if not spectral_pass:
        status = "REVISE_OBSERVATION_OR_MODEL"
    elif primary_confounding:
        status = "REVISE_OBSERVATION_OR_MODEL"
    elif parameter_ready["f_mel"] and parameter_ready["f_blood"]:
        status = "CONDITIONAL_BIO_READY_WITH_S" if parameter_ready["s"] else "CONDITIONAL_BIO_READY"
    else:
        status = "SPECTRAL_ONLY"

    output.mkdir(parents=True, exist_ok=False)
    frames = {
        "main_results": main,
        "main_multistart_results": pd.DataFrame(main_start_rows),
        "subject_summary": subject_summary,
        "loo_reference_control": loo,
        "profile_grid": pd.DataFrame(profile_rows),
        "profile_multistart_results": pd.DataFrame(profile_start_rows),
        "parameter_reliability": reliability,
        "parameter_summary": parameter_summary,
        "sensitivity_results": sensitivity,
        "sensitivity_summary": robustness,
        "free_gain_diagnostic": gain,
        "side_confounding": side_summary,
    }
    output_files: dict[str, dict[str, str]] = {}
    for name, frame in frames.items():
        parquet, csv = output / f"{name}.parquet", output / f"{name}.csv"
        _write_frame(frame, parquet, csv)
        output_files[f"{name}_parquet"] = {"path": str(parquet), "sha256": sha256_file(parquet)}
        output_files[f"{name}_csv"] = {"path": str(csv), "sha256": sha256_file(csv)}

    summary = {
        "schema_version": 1, "stage": config["stage"], "model_id": "KM2L-HF-v1", "status": status,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "spectral_gate_pass": spectral_pass, "spectral_gates": gates,
        "spectral_statistics": {
            "subject_median_logrmse": float(subject_summary["logrmse"].median()),
            "subject_p90_logrmse": float(subject_summary["logrmse"].quantile(0.9)),
            "subject_median_rmse": float(subject_summary["rmse"].median()),
            "subject_median_sam_deg": float(subject_summary["sam_deg"].median()),
            "maximum_abs_median_signed_band_residual": float(subject_summary[residual_columns].median().abs().max()),
            "loo_better_subject_fraction": float(subject_summary["model_better_than_loo"].mean()),
            "loo_median_error_ratio": float(subject_summary["model_to_loo_error_ratio"].median()),
        },
        "parameter_summary": parameter_summary.to_dict(orient="records"),
        "observation_confounding": {"primary_flag": primary_confounding, "by_parameter": confounding_flags, "details": side_summary.to_dict(orient="records")},
        "free_gain_diagnostic": {"median_logrmse": float(gain["logrmse"].median()), "median_ratio_to_main": float(gain["logrmse_ratio_to_main"].median()), "median_gain": float(gain["gain"].median())},
        "counts": {"subjects": int(main["subject_id"].nunique()), "region_spectra": int(len(main)), "main_converged": int(main["solver_status"].eq("CONVERGED").sum()), "validation_content_reads": 0, "test_content_reads": 0},
        "source_hashes": {"config": sha256_file(config_file), **{name: sha256_file(path) for name, path in paths.items()}},
        "outputs": output_files,
        "next_stage_allowed": status in {"CONDITIONAL_BIO_READY", "CONDITIONAL_BIO_READY_WITH_S"},
        "authorized_next_stage": "D_VALIDATION_REVIEW" if status in {"CONDITIONAL_BIO_READY", "CONDITIONAL_BIO_READY_WITH_S"} else None,
    }
    decision_path = output / "stage_c_decision.json"
    decision_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    report = (
        "# KM-BIO-v1 Stage C decision\n\n"
        f"- Status: `{status}`\n"
        f"- Spectral gate: `{spectral_pass}`\n"
        f"- Subject median / p90 logRMSE: {summary['spectral_statistics']['subject_median_logrmse']:.6f} / {summary['spectral_statistics']['subject_p90_logrmse']:.6f}\n"
        f"- Subject median RMSE / SAM: {summary['spectral_statistics']['subject_median_rmse']:.6f} / {summary['spectral_statistics']['subject_median_sam_deg']:.3f} deg\n"
        f"- LOO better fraction / median error ratio: {summary['spectral_statistics']['loo_better_subject_fraction']:.3f} / {summary['spectral_statistics']['loo_median_error_ratio']:.3f}\n"
        f"- Primary observation-confounding flag: `{primary_confounding}`\n"
        f"- Validation/Test content reads: 0/0\n\n"
        "This is a Train-only internal feasibility decision. Parameters remain conditional on the registered KM model and observation assumptions.\n"
    )
    report_path = output / "STAGE_C_KM_BIO_DECISION.md"
    report_path.write_text(report, encoding="utf-8")
    manifest_rows = [{"role": "source", "name": name, "path": str(path), "sha256": sha256_file(path)} for name, path in paths.items()]
    manifest_rows += [{"role": "output", "name": name, **value} for name, value in output_files.items()]
    manifest_rows += [
        {"role": "output", "name": "stage_c_decision", "path": str(decision_path), "sha256": sha256_file(decision_path)},
        {"role": "output", "name": "decision_report", "path": str(report_path), "sha256": sha256_file(report_path)},
    ]
    pd.DataFrame(manifest_rows).to_csv(output / "artifact_hash_manifest.csv", index=False, encoding="utf-8-sig")
    return summary
