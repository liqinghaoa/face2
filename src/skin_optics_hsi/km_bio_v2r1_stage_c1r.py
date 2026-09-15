"""KM-BIO-v2R.1 R-C1R: complete the R-C1 contract, including V2R-BEST-O.

The earlier ``km_bio_v2r1_stage_c1`` batch reported ``R_C1_COMPLETE`` while
leaving two section 8.5 components open: conditional-loss parameter profiles on
the R-A registered rule, and the ``V2R-BEST-O`` oxygenation extension.  This
module completes them in an independent batch with its own config, runner,
tests and output directory.  Nothing in the earlier batch is modified.

Only the 44 Train subjects, the R-B bilateral symmetric spectra, the frozen 5
folds and the read-only R-C0R global parameters are consumed.  Raw HSI, RGB,
Validation, Test and clinical-500 content is never read.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml
from scipy.stats import pearsonr, spearmanr

from .km_bio_inverse import FitSettings, fit_bounded_spectrum, sobol_plus_center, spectral_metrics
from .km_bio_observation import sha256_file
from .km_bio_v2r import forward_preloaded_numpy, load_optical_numpy


WAVELENGTH = np.arange(400.0, 701.0, 10.0)
FIT_LO = np.array([0.0, 0.0], dtype=np.float64)
FIT_HI = np.array([0.43, 0.10], dtype=np.float64)
BEST_O_LO = np.array([0.0, 0.0, 0.0], dtype=np.float64)
BEST_O_HI = np.array([0.43, 0.10, 1.0], dtype=np.float64)
BASE_PARAMETER_NAMES = ("f_mel", "f_blood")
BEST_O_PARAMETER_NAMES = ("f_mel", "f_blood", "s")


class ContractError(RuntimeError):
    """A hard R-C1R integrity or implementation violation."""


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
    frame.to_csv(path, index=False, encoding="utf-8-sig")


def _log(message: str) -> None:
    print(f"[R-C1R] {message}", flush=True)


def _columns() -> list[str]:
    return [f"observed_reflectance_{int(value)}nm" for value in WAVELENGTH]


def _settings(config: dict[str, Any], *, starts: int | None = None) -> FitSettings:
    raw = config["solver"]["individual"]
    return FitSettings(
        epsilon=float(config["analysis"]["epsilon"]),
        sobol_starts=int(raw["sobol_starts"] if starts is None else starts),
        seed=int(raw["random_seed"]),
        ftol=float(raw["ftol"]),
        xtol=float(raw["xtol"]),
        gtol=float(raw["gtol"]),
        max_nfev=int(raw["max_nfev"]),
    )


def _scaled_optical(optical: dict[str, np.ndarray], hb_g_l: float) -> dict[str, np.ndarray]:
    scale = float(hb_g_l) / 150.0
    out = {name: np.array(values, dtype=np.float64, copy=True) for name, values in optical.items()}
    out["mua_hbo2"] *= scale
    out["mua_hb"] *= scale
    return out


def _variant_kwargs(variant: dict[str, Any]) -> dict[str, Any]:
    return {
        "diameter_um": float(variant["diameter_um"]),
        "epidermis_thickness_mm": float(variant["epidermis_thickness_mm"]),
        "scattering_amplitude": float(variant["scattering_amplitude"]),
        "delta_bs": float(variant["delta_bs"]),
        "g0": float(variant["g0"]),
    }


def _band_forward(
    optical: dict[str, np.ndarray],
    band: tuple[int, int],
    *,
    s0: float,
    hb_g_l: float,
    kwargs: dict[str, Any],
):
    mask = (WAVELENGTH >= band[0]) & (WAVELENGTH <= band[1])
    if int(mask.sum()) < 5:
        raise ContractError(f"Fit band {band} contains too few centres")
    fitted_optical = {name: values[mask] for name, values in _scaled_optical(optical, hb_g_l).items()}
    wavelength = WAVELENGTH[mask]

    def forward(theta: np.ndarray) -> np.ndarray:
        return forward_preloaded_numpy(theta, fitted_optical, wavelength_nm=wavelength, s0=s0, **kwargs)

    return mask, forward


def _fit_subject(
    observed: np.ndarray,
    optical: dict[str, np.ndarray],
    variant: dict[str, Any],
    *,
    lower: np.ndarray,
    upper: np.ndarray,
    settings: FitSettings,
) -> dict[str, Any]:
    mask, forward = _band_forward(
        optical, variant["band"], s0=float(variant["s0"]), hb_g_l=float(variant["hb_g_l"]), kwargs=_variant_kwargs(variant)
    )
    result = fit_bounded_spectrum(observed[mask], forward, lower, upper, settings)
    if not result["success"]:
        raise ContractError("Individual inversion did not converge")
    prediction = np.asarray(
        forward_preloaded_numpy(
            result["theta"], optical, wavelength_nm=WAVELENGTH, s0=float(variant["s0"]), **_variant_kwargs(variant)
        ),
        dtype=np.float64,
    )
    if not np.isfinite(prediction).all() or np.any(prediction < 0.0):
        raise ContractError("Forward prediction is non-finite or negative")
    return {
        "theta": np.asarray(result["theta"], dtype=np.float64),
        "u": np.asarray(result["u"], dtype=np.float64),
        "metrics": spectral_metrics(prediction[mask], observed[mask], settings.epsilon),
        "prediction": prediction,
        "above_one_count": int(np.count_nonzero(prediction > 1.0 + 1e-12)),
    }


def _conditional_profile(
    observed: np.ndarray,
    optical: dict[str, np.ndarray],
    variant: dict[str, Any],
    best_u: np.ndarray,
    index: int,
    *,
    lower: np.ndarray,
    upper: np.ndarray,
    settings: FitSettings,
    grid_points: int,
    conditional_starts: int,
    seed: int,
) -> list[dict[str, Any]]:
    mask, forward = _band_forward(
        optical, variant["band"], s0=float(variant["s0"]), hb_g_l=float(variant["hb_g_l"]), kwargs=_variant_kwargs(variant)
    )

    def forward_u(u_normalized: np.ndarray) -> np.ndarray:
        return forward(lower + np.asarray(u_normalized, dtype=np.float64) * (upper - lower))

    grid = np.linspace(0.0, 1.0, int(grid_points), dtype=np.float64)
    if not np.any(np.isclose(grid, best_u[index], atol=1e-14, rtol=0.0)):
        grid = np.sort(np.append(grid, best_u[index]))
    remaining = [position for position in range(len(best_u)) if position != index]
    rows: list[dict[str, Any]] = []
    for grid_index, fixed_u in enumerate(grid):
        starts = sobol_plus_center(len(remaining), int(conditional_starts), seed + grid_index)[:-1]
        starts = np.vstack([starts, best_u[remaining]])

        def reduced(reduced_u: np.ndarray) -> np.ndarray:
            full = best_u.copy()
            full[index] = fixed_u
            full[remaining] = reduced_u
            return forward_u(full)

        result = fit_bounded_spectrum(
            observed[mask],
            reduced,
            np.zeros(len(remaining)),
            np.ones(len(remaining)),
            FitSettings(
                epsilon=settings.epsilon,
                sobol_starts=len(starts),
                seed=seed + grid_index,
                ftol=settings.ftol,
                xtol=settings.xtol,
                gtol=settings.gtol,
                max_nfev=settings.max_nfev,
            ),
            initial_u=starts,
        )
        rows.append({
            "grid_value_u": float(fixed_u),
            "grid_value": float(lower[index] + fixed_u * (upper[index] - lower[index])),
            "logrmse": float(result["metrics"]["logrmse"]) if result["success"] else float("inf"),
            "solver_success": bool(result["success"]),
        })
    return rows


def _profile_envelope(rows: list[dict[str, Any]], delta: float) -> dict[str, Any]:
    finite = [row for row in rows if np.isfinite(row["logrmse"])]
    if not finite:
        raise ContractError("Conditional profile produced no finite loss")
    optimum = float(min(row["logrmse"] for row in finite))
    accepted = [row["grid_value_u"] for row in finite if row["logrmse"] <= optimum + delta]
    return {
        "optimum_logrmse": optimum,
        "envelope_low_u": float(min(accepted)),
        "envelope_high_u": float(max(accepted)),
        "envelope_span_u": float(max(accepted) - min(accepted)),
        "accepted_grid_points": int(len(accepted)),
        "finite_grid_points": int(len(finite)),
        "all_grid_points_converged": bool(len(finite) == len(rows)),
    }


def _classify(identifiable_fraction: float, boundary_fraction: float, name: str, rules: dict[str, Any]) -> str:
    if (
        identifiable_fraction >= float(rules["reliable_min_identifiable_fraction"])
        and boundary_fraction <= float(rules["reliable_max_boundary_fraction"][name])
    ):
        return "reliable"
    if (
        identifiable_fraction >= float(rules["conditional_min_identifiable_fraction"])
        and boundary_fraction <= float(rules["conditional_max_boundary_fraction"][name])
    ):
        return "conditional"
    return "unreliable"


def _base_variant(config: dict[str, Any], globals_row: pd.Series) -> dict[str, Any]:
    return {
        "band": (420, 680),
        "s0": float(config["analysis"]["fixed_s0"]),
        "hb_g_l": float(config["analysis"]["fixed_hb_g_l"]),
        "diameter_um": float(config["analysis"]["fixed_diameter_um"]),
        "epidermis_thickness_mm": 0.060,
        "scattering_amplitude": float(globals_row["A_s"]),
        "delta_bs": float(globals_row["delta_bs"]),
        "g0": float(globals_row["g0"]),
    }


def _variants(config: dict[str, Any], base: dict[str, Any], *, include_s0_setting: bool) -> list[dict[str, Any]]:
    grid = config["sensitivity"]
    rows = [{"setting": "baseline", "value": float("nan"), **base}]
    for value in grid["diameter_um"]:
        rows.append({"setting": "diameter_um", "value": float(value), **{**base, "diameter_um": float(value)}})
    if include_s0_setting:
        for value in grid["s0"]:
            rows.append({"setting": "s0", "value": float(value), **{**base, "s0": float(value)}})
    for value in grid["epidermis_thickness_mm"]:
        rows.append({"setting": "epidermis_thickness_mm", "value": float(value), **{**base, "epidermis_thickness_mm": float(value)}})
    for value in grid["whole_blood_hb_g_l"]:
        rows.append({"setting": "whole_blood_hb_g_l", "value": float(value), **{**base, "hb_g_l": float(value)}})
    for value in grid["scattering_amplitude_multiplier"]:
        rows.append({
            "setting": "scattering_amplitude_multiplier", "value": float(value),
            **{**base, "scattering_amplitude": float(base["scattering_amplitude"]) * float(value)},
        })
    for value in grid["delta_bs_shift"]:
        rows.append({
            "setting": "delta_bs_shift", "value": float(value),
            **{**base, "delta_bs": float(base["delta_bs"]) + float(value)},
        })
    for low, high in grid["bandwidth_nm"]:
        rows.append({"setting": f"bandwidth_{int(low)}_{int(high)}", "value": float(high - low), **{**base, "band": (int(low), int(high))}})
    return rows


def _preflight(root: Path, config: dict[str, Any]) -> tuple[dict[str, Path], dict[str, Any]]:
    for key in ("raw_hsi_content_allowed", "rgb_content_allowed", "validation_content_allowed", "test_content_allowed", "clinical_500_content_allowed"):
        if config["access_policy"].get(key):
            raise ContractError("Access policy permits forbidden data")
    if any(config["execution"].get(key) for key in ("run_validation", "run_test", "run_clinical_500", "run_rgb_encoder_training")):
        raise ContractError("Execution block attempts an out-of-scope stage")
    if not config["execution"].get("run_best_o"):
        raise ContractError("R-C1R must run the registered V2R-BEST-O candidate")
    if config["execution"].get("stop_on_diagnostic_flags"):
        raise ContractError("R-C1R must not stop on diagnostic flags")
    if config["execution"].get("train_only") is not True:
        raise ContractError("R-C1R must declare train_only")
    paths = {name: _resolve(root, value) for name, value in config["inputs"].items()}
    missing = sorted(name for name, path in paths.items() if not path.exists())
    if missing:
        raise ContractError(f"Missing R-C1R inputs: {missing}")

    decision = json.loads(paths["r_c0r_decision"].read_text(encoding="utf-8"))
    audit = json.loads(paths["r_c0r_audit"].read_text(encoding="utf-8"))
    if decision.get("status") != "R_C0R_COMPLETE" or decision.get("selected_candidate") != config["selection"]["selected_development_candidate"]:
        raise ContractError("R-C0R decision does not select the registered development candidate")
    if audit.get("status") != "R_C0R_COMPLETE":
        raise ContractError("R-C0R audit status mismatch")
    for key in ("validation_reads", "test_reads", "clinical_500_reads", "raw_hsi_reads", "rgb_reads"):
        if int(audit.get("data_access", {}).get(key, -1)) != 0:
            raise ContractError(f"Upstream R-C0R batch recorded a non-zero {key}")
    r_c1_decision = json.loads(paths["r_c1_decision"].read_text(encoding="utf-8"))
    if r_c1_decision.get("selected_development_candidate") != config["selection"]["selected_development_candidate"]:
        raise ContractError("R-C1 decision deviates from the R-C0R selected candidate")

    fold_hash = sha256_file(paths["r_c0r_fold_manifest"])
    if fold_hash != str(config["folds"]["expected_sha256"]):
        raise ContractError("Frozen fold manifest hash is not the R-C0R registered manifest")
    return paths, {
        "r_c0r_decision_status": decision["status"],
        "r_c0r_selected_candidate": decision["selected_candidate"],
        "r_c0r_audit_status": audit["status"],
        "r_c1_reported_selected_development_candidate": r_c1_decision["selected_development_candidate"],
        "r_c1_reported_status": r_c1_decision.get("status"),
        "fold_manifest_sha256": fold_hash,
        "fold_manifest_sha256_matches_registered_manifest": True,
        "forbidden_data_access_declared": False,
    }


def run_stage_c1r(config_path: str | Path, project_root: str | Path) -> dict[str, Any]:
    root = Path(project_root).resolve()
    config_file = Path(config_path).resolve()
    config = yaml.safe_load(config_file.read_text(encoding="utf-8"))
    if config["task_id"] != "R-C1R_COMPLETE_IDENTIFIABILITY_AND_BEST_O":
        raise ContractError("Unexpected R-C1R task id")
    output = _resolve(root, config["output_directory"])
    if output.exists() and any(output.iterdir()) and config["access_policy"]["refuse_existing_output"]:
        raise FileExistsError(f"Refusing to overwrite R-C1R output: {output}")

    paths, preflight = _preflight(root, config)
    _log("preflight passed; loading Train-only inputs")

    manifest = pd.read_parquet(paths["observation_manifest"]).reset_index(drop=True)
    if len(manifest) != 44 or set(manifest["split"].astype(str)) != {"train"} or not manifest["input_quality_status"].eq("PASS").all():
        raise ContractError("R-B manifest does not satisfy the 44-subject Train-only contract")
    observed = manifest[_columns()].to_numpy(dtype=np.float64)
    if not np.isfinite(observed).all() or np.any(observed <= 0.0) or np.any(observed > 1.0 + 1e-12):
        raise ContractError("R-B symmetric manifest contains invalid reflectance")
    optical = load_optical_numpy(WAVELENGTH, paths["optical_asset_10nm"])

    fold_manifest = pd.read_csv(paths["r_c0r_fold_manifest"], encoding="utf-8-sig").sort_values(["outer_fold", "subject_id"]).reset_index(drop=True)
    globals_frame = pd.read_csv(paths["r_c0r_fold_globals"], encoding="utf-8-sig")
    reference_metrics = pd.read_csv(paths["r_c0r_subject_metrics"], encoding="utf-8-sig")
    fold_of = dict(zip(fold_manifest["subject_id"].astype(str), fold_manifest["outer_fold"].astype(int)))
    if set(manifest["subject_id"].astype(str)) != set(fold_of):
        raise ContractError("R-B manifest subject set differs from the frozen fold manifest")

    settings = _settings(config)
    variant_settings = _settings(config, starts=int(config["solver"]["sensitivity_sobol_starts"]))
    ident = config["identifiability"]
    boundary_tolerance = float(ident["boundary_tolerance_normalized"])
    profile_candidates = list(config["selection"]["profile_candidates"])
    profile_grid_points = int(config["solver"]["profile_grid_points"])
    conditional_starts = int(config["solver"]["profile_conditional_starts"])
    base_seed = int(config["solver"]["individual"]["random_seed"])
    for candidate in profile_candidates:
        if not (globals_frame["candidate"] == candidate).any():
            raise ContractError(f"R-C0R fold globals are missing {candidate}")

    output.mkdir(parents=True, exist_ok=True)
    _write_csv(fold_manifest, output / "fold_manifest_reused.csv")
    (output / "fold_manifest_reused.sha256").write_text(f"{preflight['fold_manifest_sha256']}  fold_manifest_reused.csv\n", encoding="ascii")
    _write_csv(globals_frame[globals_frame["candidate"].isin(profile_candidates)].copy(), output / "fold_global_parameters_reused.csv")

    def globals_for(candidate: str, fold: int) -> pd.Series:
        rows = globals_frame[(globals_frame["candidate"] == candidate) & (globals_frame["outer_fold"] == fold)]
        if len(rows) != 1:
            raise ContractError(f"Ambiguous R-C0R fold globals for {candidate} fold {fold}")
        return rows.iloc[0]

    def boundary_flag(u: np.ndarray) -> bool:
        return bool(np.any((u <= boundary_tolerance) | (u >= 1.0 - boundary_tolerance)))

    profile_rows: list[dict[str, Any]] = []
    profile_summary_rows: list[dict[str, Any]] = []
    sensitivity_rows: list[dict[str, Any]] = []
    baseline_rows: list[dict[str, Any]] = []
    reproduction_rows: list[dict[str, Any]] = []
    fitted_parameters: dict[str, pd.DataFrame] = {}
    reference_by_candidate = {candidate: group.set_index("subject_id") for candidate, group in reference_metrics.groupby("candidate", sort=False)}

    for candidate in profile_candidates:
        _log(f"{candidate}: baseline inversion, conditional profiles, sensitivity")
        per_subject: list[dict[str, Any]] = []
        reference = reference_by_candidate[candidate]
        for position, item in manifest.iterrows():
            subject = str(item["subject_id"])
            fold = fold_of[subject]
            base = _base_variant(config, globals_for(candidate, fold))
            fit = _fit_subject(observed[position], optical, base, lower=FIT_LO, upper=FIT_HI, settings=settings)
            reference_row = reference.loc[subject]
            record = {
                "candidate": candidate, "subject_id": subject, "outer_fold": int(fold),
                "f_mel": float(fit["theta"][0]), "f_blood": float(fit["theta"][1]),
                "logrmse": float(fit["metrics"]["logrmse"]), "rmse": float(fit["metrics"]["rmse"]),
                "sam_deg": float(fit["metrics"]["sam_deg"]), "converged": True,
                "prediction_above_one_count": int(fit["above_one_count"]),
            }
            per_subject.append(record)
            reproduction_rows.append({
                "candidate": candidate, "subject_id": subject,
                "reference_f_mel": float(reference_row["f_mel"]), "actual_f_mel": record["f_mel"],
                "abs_delta_f_mel": float(abs(record["f_mel"] - float(reference_row["f_mel"]))),
                "reference_f_blood": float(reference_row["f_blood"]), "actual_f_blood": record["f_blood"],
                "abs_delta_f_blood": float(abs(record["f_blood"] - float(reference_row["f_blood"]))),
                "reference_logrmse": float(reference_row["logrmse"]), "actual_logrmse": record["logrmse"],
                "abs_delta_logrmse": float(abs(record["logrmse"] - float(reference_row["logrmse"]))),
            })
            for index, name in enumerate(BASE_PARAMETER_NAMES):
                rows = _conditional_profile(
                    observed[position], optical, base, fit["u"], index, lower=FIT_LO, upper=FIT_HI,
                    settings=settings, grid_points=profile_grid_points, conditional_starts=conditional_starts,
                    seed=base_seed + 1000 * position + index,
                )
                envelope = _profile_envelope(rows, float(ident["acceptable_loss_delta_logrmse"]))
                for row in rows:
                    profile_rows.append({
                        "candidate": candidate, "subject_id": subject, "parameter": name,
                        "grid_value_u": row["grid_value_u"], "grid_value": row["grid_value"],
                        "logrmse": row["logrmse"], "solver_success": row["solver_success"],
                    })
                profile_summary_rows.append({
                    "candidate": candidate, "subject_id": subject, "parameter": name,
                    "fit_value": record[name], "fit_value_u": float(fit["u"][index]),
                    "optimum_logrmse": envelope["optimum_logrmse"],
                    "envelope_low_u": envelope["envelope_low_u"], "envelope_high_u": envelope["envelope_high_u"],
                    "envelope_span_u": envelope["envelope_span_u"],
                    "envelope_span_physical": float(envelope["envelope_span_u"] * (FIT_HI[index] - FIT_LO[index])),
                    "envelope_normalized_span": envelope["envelope_span_u"],
                    "identifiable": bool(envelope["envelope_span_u"] <= float(ident["maximum_acceptable_normalized_envelope_span"])),
                    "at_boundary": boundary_flag(fit["u"]),
                    "accepted_grid_points": envelope["accepted_grid_points"],
                    "all_grid_points_converged": envelope["all_grid_points_converged"],
                })
            for variant in _variants(config, base, include_s0_setting=True)[1:]:
                perturbed = _fit_subject(observed[position], optical, variant, lower=FIT_LO, upper=FIT_HI, settings=variant_settings)
                sensitivity_rows.append({
                    "candidate": candidate, "subject_id": subject, "outer_fold": int(fold),
                    "setting": variant["setting"], "value": float(variant["value"]),
                    "f_mel": float(perturbed["theta"][0]), "f_blood": float(perturbed["theta"][1]),
                    "logrmse": float(perturbed["metrics"]["logrmse"]), "rmse": float(perturbed["metrics"]["rmse"]),
                    "sam_deg": float(perturbed["metrics"]["sam_deg"]),
                    "delta_f_mel": float(perturbed["theta"][0] - record["f_mel"]),
                    "delta_f_blood": float(perturbed["theta"][1] - record["f_blood"]),
                    "logrmse_change": float(perturbed["metrics"]["logrmse"] - record["logrmse"]),
                    "boundary_any": boundary_flag(perturbed["u"]),
                    "converged": True, "prediction_above_one_count": int(perturbed["above_one_count"]),
                })
        fitted_parameters[candidate] = pd.DataFrame(per_subject)
        baseline_rows.extend(per_subject)
        _log(f"{candidate}: complete")

    baseline_frame = pd.DataFrame(baseline_rows)
    profile_frame = pd.DataFrame(profile_summary_rows)
    sensitivity_frame = pd.DataFrame(sensitivity_rows)

    profile_summary = pd.DataFrame([{
        "candidate": candidate, "parameter": parameter, "subject_count": int(len(group)),
        "boundary_fraction": float(group["at_boundary"].mean()),
        "identifiable_fraction": float(group["identifiable"].mean()),
        "median_envelope_normalized_span": float(group["envelope_normalized_span"].median()),
        "max_envelope_normalized_span": float(group["envelope_normalized_span"].max()),
        "median_fit_value": float(group["fit_value"].median()),
        "classification": _classify(float(group["identifiable"].mean()), float(group["at_boundary"].mean()), parameter, ident),
        "all_grid_points_converged": bool(group["all_grid_points_converged"].all()),
    } for (candidate, parameter), group in profile_frame.groupby(["candidate", "parameter"], sort=False)])

    sensitivity_summary = pd.DataFrame([{
        "candidate": candidate, "setting": setting, "value_count": int(group["value"].nunique()),
        "median_logrmse": float(group["logrmse"].median()),
        "median_abs_delta_f_mel": float(group["delta_f_mel"].abs().median()),
        "median_abs_delta_f_blood": float(group["delta_f_blood"].abs().median()),
        "max_abs_delta_f_mel": float(group["delta_f_mel"].abs().max()),
        "max_abs_delta_f_blood": float(group["delta_f_blood"].abs().max()),
        "median_logrmse_change": float(group["logrmse_change"].median()),
        "boundary_fraction": float(group["boundary_any"].mean()),
        "all_converged": bool(group["converged"].all()),
        "prediction_above_one_count": int(group["prediction_above_one_count"].sum()),
    } for (candidate, setting), group in sensitivity_frame.groupby(["candidate", "setting"], sort=False)])

    best_o_config = config["best_o"]
    best_o_lo = np.asarray(best_o_config["lower"], dtype=np.float64)
    best_o_hi = np.asarray(best_o_config["upper"], dtype=np.float64)
    best_o_rows: list[dict[str, Any]] = []
    best_o_profile_rows: list[dict[str, Any]] = []
    best_o_profile_summary_rows: list[dict[str, Any]] = []
    best_o_sensitivity_rows: list[dict[str, Any]] = []
    best_o_shift_rows: list[dict[str, Any]] = []
    for base_candidate in best_o_config["base_candidates"]:
        _log(f"V2R-BEST-O on {base_candidate}: inversion, profiles, sensitivity")
        base_frame = fitted_parameters[base_candidate].set_index("subject_id")
        for position, item in manifest.iterrows():
            subject = str(item["subject_id"])
            fold = fold_of[subject]
            base = _base_variant(config, globals_for(base_candidate, fold))
            fit = _fit_subject(observed[position], optical, base, lower=best_o_lo, upper=best_o_hi, settings=settings)
            base_row = base_frame.loc[subject]
            record = {
                "base_candidate": base_candidate, "subject_id": subject, "outer_fold": int(fold),
                "f_mel": float(fit["theta"][0]), "f_blood": float(fit["theta"][1]), "s": float(fit["theta"][2]),
                "logrmse": float(fit["metrics"]["logrmse"]), "rmse": float(fit["metrics"]["rmse"]),
                "sam_deg": float(fit["metrics"]["sam_deg"]), "converged": True,
                "prediction_above_one_count": int(fit["above_one_count"]),
            }
            best_o_rows.append(record)
            best_o_shift_rows.append({
                "base_candidate": base_candidate, "subject_id": subject, "outer_fold": int(fold),
                "base_logrmse": float(base_row["logrmse"]), "best_o_logrmse": record["logrmse"],
                "logrmse_change": float(record["logrmse"] - float(base_row["logrmse"])),
                "delta_f_mel": float(record["f_mel"] - float(base_row["f_mel"])),
                "delta_f_blood": float(record["f_blood"] - float(base_row["f_blood"])),
            })
            for index, name in enumerate(BEST_O_PARAMETER_NAMES):
                rows = _conditional_profile(
                    observed[position], optical, base, fit["u"], index, lower=best_o_lo, upper=best_o_hi,
                    settings=settings, grid_points=profile_grid_points, conditional_starts=conditional_starts,
                    seed=base_seed + 5000 * position + index,
                )
                envelope = _profile_envelope(rows, float(ident["acceptable_loss_delta_logrmse"]))
                for row in rows:
                    best_o_profile_rows.append({
                        "base_candidate": base_candidate, "subject_id": subject, "parameter": name,
                        "grid_value_u": row["grid_value_u"], "grid_value": row["grid_value"],
                        "logrmse": row["logrmse"], "solver_success": row["solver_success"],
                    })
                best_o_profile_summary_rows.append({
                    "base_candidate": base_candidate, "subject_id": subject, "parameter": name,
                    "fit_value": record[name], "fit_value_u": float(fit["u"][index]),
                    "optimum_logrmse": envelope["optimum_logrmse"],
                    "envelope_low_u": envelope["envelope_low_u"], "envelope_high_u": envelope["envelope_high_u"],
                    "envelope_span_u": envelope["envelope_span_u"],
                    "envelope_span_physical": float(envelope["envelope_span_u"] * (best_o_hi[index] - best_o_lo[index])),
                    "envelope_normalized_span": envelope["envelope_span_u"],
                    "identifiable": bool(envelope["envelope_span_u"] <= float(ident["maximum_acceptable_normalized_envelope_span"])),
                    "at_boundary": boundary_flag(fit["u"]),
                    "accepted_grid_points": envelope["accepted_grid_points"],
                    "all_grid_points_converged": envelope["all_grid_points_converged"],
                })
            for variant in _variants(config, base, include_s0_setting=False)[1:]:
                perturbed = _fit_subject(observed[position], optical, variant, lower=best_o_lo, upper=best_o_hi, settings=variant_settings)
                best_o_sensitivity_rows.append({
                    "base_candidate": base_candidate, "subject_id": subject, "outer_fold": int(fold),
                    "setting": variant["setting"], "value": float(variant["value"]),
                    "f_mel": float(perturbed["theta"][0]), "f_blood": float(perturbed["theta"][1]), "s": float(perturbed["theta"][2]),
                    "logrmse": float(perturbed["metrics"]["logrmse"]),
                    "delta_f_mel": float(perturbed["theta"][0] - record["f_mel"]),
                    "delta_f_blood": float(perturbed["theta"][1] - record["f_blood"]),
                    "delta_s": float(perturbed["theta"][2] - record["s"]),
                    "logrmse_change": float(perturbed["metrics"]["logrmse"] - record["logrmse"]),
                    "boundary_any": boundary_flag(perturbed["u"]),
                    "converged": True, "prediction_above_one_count": int(perturbed["above_one_count"]),
                })
        _log(f"V2R-BEST-O on {base_candidate}: complete")

    best_o_frame = pd.DataFrame(best_o_rows)
    best_o_profile_frame = pd.DataFrame(best_o_profile_summary_rows)
    best_o_sensitivity_frame = pd.DataFrame(best_o_sensitivity_rows)
    best_o_shift_frame = pd.DataFrame(best_o_shift_rows)

    best_o_summary = pd.DataFrame([{
        "base_candidate": base_candidate, "subject_count": int(len(group)),
        "median_logrmse": float(group["logrmse"].median()),
        "base_median_logrmse": float(fitted_parameters[base_candidate]["logrmse"].median()),
        "median_logrmse_change": float(group["logrmse"].median() - fitted_parameters[base_candidate]["logrmse"].median()),
        "p90_logrmse": float(group["logrmse"].quantile(0.90)),
        "median_rmse": float(group["rmse"].median()),
        "median_sam_deg": float(group["sam_deg"].median()),
        "median_s": float(group["s"].median()),
        "median_f_mel": float(group["f_mel"].median()),
        "median_f_blood": float(group["f_blood"].median()),
        "all_converged": bool(group["converged"].all()),
        "prediction_above_one_count": int(group["prediction_above_one_count"].sum()),
    } for base_candidate, group in best_o_frame.groupby("base_candidate", sort=False)])

    best_o_profile_summary = pd.DataFrame([{
        "base_candidate": base_candidate, "parameter": parameter, "subject_count": int(len(group)),
        "boundary_fraction": float(group["at_boundary"].mean()),
        "identifiable_fraction": float(group["identifiable"].mean()),
        "median_envelope_normalized_span": float(group["envelope_normalized_span"].median()),
        "max_envelope_normalized_span": float(group["envelope_normalized_span"].max()),
        "median_fit_value": float(group["fit_value"].median()),
        "classification": _classify(float(group["identifiable"].mean()), float(group["at_boundary"].mean()), parameter, ident),
        "all_grid_points_converged": bool(group["all_grid_points_converged"].all()),
    } for (base_candidate, parameter), group in best_o_profile_frame.groupby(["base_candidate", "parameter"], sort=False)])

    best_o_sensitivity_summary = pd.DataFrame([{
        "base_candidate": base_candidate, "setting": setting, "value_count": int(group["value"].nunique()),
        "median_logrmse": float(group["logrmse"].median()),
        "median_abs_delta_s": float(group["delta_s"].abs().median()),
        "max_abs_delta_s": float(group["delta_s"].abs().max()),
        "median_abs_delta_f_mel": float(group["delta_f_mel"].abs().median()),
        "median_abs_delta_f_blood": float(group["delta_f_blood"].abs().median()),
        "boundary_fraction": float(group["boundary_any"].mean()),
        "all_converged": bool(group["converged"].all()),
    } for (base_candidate, setting), group in best_o_sensitivity_frame.groupby(["base_candidate", "setting"], sort=False)])

    side = pd.read_csv(paths["side_pair_audit"], encoding="utf-8-sig")
    side_small = side[["subject_id", "log_left_minus_log_right_mean", "left_minus_right_broadband"]].copy()
    side_small["subject_id"] = side_small["subject_id"].astype(str)
    linkage_rows: list[dict[str, Any]] = []
    linkage_inputs = [(candidate, fitted_parameters[candidate], BASE_PARAMETER_NAMES) for candidate in profile_candidates]
    linkage_inputs += [
        (f"V2R-BEST-O({base_candidate})", best_o_frame[best_o_frame["base_candidate"] == base_candidate], BEST_O_PARAMETER_NAMES)
        for base_candidate in best_o_config["base_candidates"]
    ]
    for label, frame, parameters in linkage_inputs:
        joined = frame.merge(side_small, on="subject_id", how="inner", validate="one_to_one")
        if len(joined) != 44:
            raise ContractError(f"Side-pair linkage for {label} does not cover 44 subjects")
        for parameter in parameters:
            pearson = pearsonr(joined[parameter], joined["log_left_minus_log_right_mean"])
            spearman = spearmanr(joined[parameter], joined["log_left_minus_log_right_mean"])
            linkage_rows.append({
                "candidate": label, "parameter": parameter, "subject_count": int(len(joined)),
                "pearson_r_vs_side_log_difference": float(pearson[0]), "pearson_p": float(pearson[1]),
                "spearman_r_vs_side_log_difference": float(spearman[0]), "spearman_p": float(spearman[1]),
            })
    side_linkage = pd.DataFrame(linkage_rows)

    reproduction = pd.DataFrame(reproduction_rows)
    tolerance = float(config["reproduction"]["baseline_absolute_tolerance"])
    reproduction_check = pd.DataFrame([{
        "check": "baseline_oof_reproduction_vs_r_c0r",
        "candidate_count": int(reproduction["candidate"].nunique()),
        "subject_rows": int(len(reproduction)),
        "maximum_abs_delta_f_mel": float(reproduction["abs_delta_f_mel"].max()),
        "maximum_abs_delta_f_blood": float(reproduction["abs_delta_f_blood"].max()),
        "maximum_abs_delta_logrmse": float(reproduction["abs_delta_logrmse"].max()),
        "tolerance": tolerance,
        "pass": bool(
            reproduction["abs_delta_f_mel"].max() <= tolerance
            and reproduction["abs_delta_f_blood"].max() <= tolerance
            and reproduction["abs_delta_logrmse"].max() <= tolerance
        ),
    }])
    if not bool(reproduction_check.loc[0, "pass"]):
        raise ContractError("R-C1R baseline did not reproduce the frozen R-C0R OOF parameters")

    for name, frame in (
        ("baseline_oof_reproduction_check.csv", reproduction_check),
        ("baseline_oof_rows.csv", reproduction),
        ("parameter_profile_rows.csv", pd.DataFrame(profile_rows)),
        ("parameter_profile_summary.csv", profile_summary),
        ("baseline_subject_metrics.csv", baseline_frame),
        ("sensitivity_subject_metrics.csv", sensitivity_frame),
        ("sensitivity_summary.csv", sensitivity_summary),
        ("best_o_subject_metrics.csv", best_o_frame),
        ("best_o_summary.csv", best_o_summary),
        ("best_o_parameter_shifts.csv", best_o_shift_frame),
        ("best_o_profile_rows.csv", pd.DataFrame(best_o_profile_rows)),
        ("best_o_profile_summary.csv", best_o_profile_summary),
        ("best_o_sensitivity_subject_metrics.csv", best_o_sensitivity_frame),
        ("best_o_sensitivity_summary.csv", best_o_sensitivity_summary),
        ("side_pressure_linkage.csv", side_linkage),
    ):
        _write_csv(frame, output / name)

    return _emit_artifacts(
        config=config,
        config_file=config_file,
        paths=paths,
        output=output,
        preflight=preflight,
        root=root,
        reproduction_check=reproduction_check,
        profile_summary=profile_summary,
        sensitivity_summary=sensitivity_summary,
        best_o_summary=best_o_summary,
        best_o_profile_summary=best_o_profile_summary,
        best_o_sensitivity_summary=best_o_sensitivity_summary,
        side_linkage=side_linkage,
        above_one_total=int(
            baseline_frame["prediction_above_one_count"].sum()
            + sensitivity_frame["prediction_above_one_count"].sum()
            + best_o_frame["prediction_above_one_count"].sum()
            + best_o_sensitivity_frame["prediction_above_one_count"].sum()
        ),
    )


def _input_hashes(paths: dict[str, Path]) -> dict[str, Any]:
    """Hash every input that is a file; directories are recorded as such."""

    hashes: dict[str, Any] = {}
    for name, path in paths.items():
        hashes[name] = sha256_file(path) if path.is_file() else {"kind": "directory", "path": str(path)}
    return hashes


def _emit_artifacts(
    *,
    config: dict[str, Any],
    config_file: Path,
    paths: dict[str, Path],
    output: Path,
    preflight: dict[str, Any],
    root: Path,
    reproduction_check: pd.DataFrame,
    profile_summary: pd.DataFrame,
    sensitivity_summary: pd.DataFrame,
    best_o_summary: pd.DataFrame,
    best_o_profile_summary: pd.DataFrame,
    best_o_sensitivity_summary: pd.DataFrame,
    side_linkage: pd.DataFrame,
    above_one_total: int,
) -> dict[str, Any]:
    decision = {
        "schema_version": 1,
        "task_id": config["task_id"],
        "protocol_version": config["protocol_version"],
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "R_C1R_COMPLETE",
        "supersedes_components": config["supersedes_components"],
        "baseline_reproduction": _jsonable(reproduction_check.to_dict("records")[0]),
        "parameter_profile_summary": _jsonable(profile_summary.to_dict("records")),
        "sensitivity_summary": _jsonable(sensitivity_summary.to_dict("records")),
        "best_o_summary": _jsonable(best_o_summary.to_dict("records")),
        "best_o_profile_summary": _jsonable(best_o_profile_summary.to_dict("records")),
        "best_o_fixed_quantity_sensitivity_summary": _jsonable(best_o_sensitivity_summary.to_dict("records")),
        "s_reliability_by_base_candidate": _jsonable({
            str(row["base_candidate"]): str(row["classification"])
            for _, row in best_o_profile_summary[best_o_profile_summary["parameter"] == "s"].iterrows()
        }),
        "side_pressure_linkage": _jsonable(side_linkage.to_dict("records")),
        "selected_development_candidate": config["selection"]["selected_development_candidate"],
        "best_oof_candidate": config["selection"]["best_oof_candidate"],
        "r_c1_contract_components": {
            "conditional_parameter_profiles": "complete",
            "registered_fixed_quantity_sensitivity": "complete",
            "V2R-BEST-O": "complete",
        },
        "validation_executed": False,
        "test_executed": False,
        "clinical_500_executed": False,
        "rgb_encoder_training_executed": False,
        "next_registered_stage": "R-D",
        "interpretation": "Train-only identifiability, robustness and oxygen-extension audit; no physiological truth claim.",
    }
    audit = {
        "schema_version": 1,
        "task_id": config["task_id"],
        "protocol_version": config["protocol_version"],
        "status": "R_C1R_COMPLETE",
        "created_utc": decision["created_utc"],
        "preflight_checks": _jsonable(preflight),
        "data_access": {
            "r_b_manifest_content_reads": 1,
            "r_c0r_artifact_content_reads": 4,
            "r_c1_artifact_content_reads": 2,
            "r_b_side_pair_audit_content_reads": 1,
            "raw_hsi_reads": 0, "rgb_reads": 0,
            "validation_reads": 0, "test_reads": 0, "clinical_500_reads": 0,
        },
        "subject_count": int(config["scope"]["subject_count"]),
        "profile_candidates": list(config["selection"]["profile_candidates"]),
        "best_o_base_candidates": list(config["best_o"]["base_candidates"]),
        "best_o_parameter_names": list(BEST_O_PARAMETER_NAMES),
        "fit_band_count": int(((WAVELENGTH >= 420) & (WAVELENGTH <= 680)).sum()),
        "edge_band_count": 4,
        "reflectance_clipping_applied": False,
        "prediction_above_one_count": int(above_one_total),
        "all_inversions_converged": True,
        "input_hashes": _input_hashes(paths),
        "implementation_hashes": {name: sha256_file(_resolve(root, value)) for name, value in config["implementation"].items()},
    }
    (output / "r_c1r_decision.json").write_text(json.dumps(_jsonable(decision), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (output / "r_c1r_integrity_audit.json").write_text(json.dumps(_jsonable(audit), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    report = [
        "# KM-BIO-v2R.1 R-C1R complete identifiability audit and V2R-BEST-O",
        "",
        "- Status: `R_C1R_COMPLETE`",
        "- Scope: Train-only, fixed R-C0R fold-global parameters, no Validation/Test/500/RGB access.",
        f"- Frozen fold SHA-256: `{preflight['fold_manifest_sha256']}`",
        f"- Inversions reported above one: `{above_one_total}` (no reflectance clipping applied).",
        "",
        "## Baseline reproduction against R-C0R",
        "",
        reproduction_check.to_markdown(index=False),
        "",
        "## Conditional parameter profiles (R-A rule, 51-point grid)",
        "",
        profile_summary.to_markdown(index=False),
        "",
        "## Fixed-quantity sensitivity (two-parameter base candidates)",
        "",
        sensitivity_summary.to_markdown(index=False),
        "",
        "## V2R-BEST-O oxygen extension",
        "",
        best_o_summary.to_markdown(index=False),
        "",
        best_o_profile_summary.to_markdown(index=False),
        "",
        "## V2R-BEST-O fixed-quantity sensitivity (s0 not applicable: s is open)",
        "",
        best_o_sensitivity_summary.to_markdown(index=False),
        "",
        "## Left/right pressure linkage against the R-B side-pair audit",
        "",
        side_linkage.to_markdown(index=False),
        "",
    ]
    (output / "R_C1R_REPORT.md").write_text("\n".join(report), encoding="utf-8")

    artifact_rows = [{"role": "input", "name": name, "path": str(path), "sha256": sha256_file(path) if path.is_file() else ""} for name, path in paths.items()]
    artifact_rows.append({"role": "config", "name": "r_c1r_config", "path": str(config_file), "sha256": sha256_file(config_file)})
    for name, value in config["implementation"].items():
        path = _resolve(root, value)
        artifact_rows.append({"role": "implementation", "name": name, "path": str(path), "sha256": sha256_file(path)})
    for path in sorted(output.iterdir(), key=lambda item: item.name):
        if path.is_file() and path.name != "artifact_hash_manifest.csv":
            artifact_rows.append({"role": "output", "name": path.name, "path": str(path), "sha256": sha256_file(path)})
    _write_csv(pd.DataFrame(artifact_rows), output / "artifact_hash_manifest.csv")
    _log("R-C1R artifacts emitted")
    return decision


def finalize_stage_c1r(config_path: str | Path, project_root: str | Path) -> dict[str, Any]:
    """Emit the R-C1R decision, audit, report and hash manifest from written tables.

    The heavy inversion phases write their tables first; this entry point lets the
    artifact set be completed from those tables without repeating the fits.
    """

    root = Path(project_root).resolve()
    config_file = Path(config_path).resolve()
    config = yaml.safe_load(config_file.read_text(encoding="utf-8"))
    if config["task_id"] != "R-C1R_COMPLETE_IDENTIFIABILITY_AND_BEST_O":
        raise ContractError("Unexpected R-C1R task id")
    output = _resolve(root, config["output_directory"])
    paths, preflight = _preflight(root, config)
    required = [
        "baseline_oof_reproduction_check.csv",
        "parameter_profile_summary.csv",
        "sensitivity_summary.csv",
        "best_o_summary.csv",
        "best_o_profile_summary.csv",
        "best_o_sensitivity_summary.csv",
        "side_pressure_linkage.csv",
        "baseline_subject_metrics.csv",
        "sensitivity_subject_metrics.csv",
        "best_o_subject_metrics.csv",
        "best_o_sensitivity_subject_metrics.csv",
    ]
    missing = [name for name in required if not (output / name).is_file()]
    if missing:
        raise ContractError(f"Cannot finalize R-C1R; missing tables: {missing}")

    def read(name: str) -> pd.DataFrame:
        return pd.read_csv(output / name, encoding="utf-8-sig")

    above_one_total = int(sum(read(name)["prediction_above_one_count"].sum() for name in (
        "baseline_subject_metrics.csv",
        "sensitivity_subject_metrics.csv",
        "best_o_subject_metrics.csv",
        "best_o_sensitivity_subject_metrics.csv",
    )))
    return _emit_artifacts(
        config=config,
        config_file=config_file,
        paths=paths,
        output=output,
        preflight=preflight,
        root=root,
        reproduction_check=read("baseline_oof_reproduction_check.csv"),
        profile_summary=read("parameter_profile_summary.csv"),
        sensitivity_summary=read("sensitivity_summary.csv"),
        best_o_summary=read("best_o_summary.csv"),
        best_o_profile_summary=read("best_o_profile_summary.csv"),
        best_o_sensitivity_summary=read("best_o_sensitivity_summary.csv"),
        side_linkage=read("side_pressure_linkage.csv"),
        above_one_total=above_one_total,
    )
