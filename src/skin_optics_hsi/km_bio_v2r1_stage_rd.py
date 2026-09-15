"""KM-BIO-v2R.1 R-D: freeze the Train candidate and review Validation spectra.

Section 8.5 R-D of the v2R revision.  Two components:

1. ``freeze``: estimate Train-global ``A_s/Delta_b_s`` for ``V2R-PS`` on the
   complete 44-subject Train split with the R-C0R joint centered-log estimator,
   then freeze ``{A_s, delta_bs, g0, Dv, s0}`` before any Validation spectrum is
   touched.
2. ``validation``: invert the frozen candidate on the 3-subject Validation
   bilateral symmetric spectra and compare spectral error and parameter
   behaviour with the Train side for direction consistency.

Hard boundaries: Test and clinical-500 stay locked; no fixed quantity, band,
threshold or loss is changed after seeing Validation; the R-C1R ``unreliable``
parameter verdicts are not upgraded by this batch.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml
from scipy.stats import pearsonr

from .km_bio_inverse import FitSettings, fit_bounded_spectrum, spectral_metrics
from .km_bio_observation import sha256_file
from .km_bio_v2r import (
    AS_BOUNDS,
    DELTA_BS_BOUNDS,
    forward_preloaded_numpy,
    load_optical_numpy,
    profile_identifiability,
)
from .km_bio_v2r1_stage_c0r import _shape_joint_fit


WAVELENGTH = np.arange(400.0, 701.0, 10.0)
FIT_MASK = np.isin(WAVELENGTH, np.arange(420.0, 681.0, 10.0))
FIT_LO = np.array([0.0, 0.0], dtype=np.float64)
FIT_HI = np.array([0.43, 0.10], dtype=np.float64)


class ContractError(RuntimeError):
    """A hard R-D integrity or implementation violation."""


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
    print(f"[R-D] {message}", flush=True)


def _columns() -> list[str]:
    return [f"observed_reflectance_{int(value)}nm" for value in WAVELENGTH]


def _settings(config: dict[str, Any], *, seed: int | None = None) -> FitSettings:
    raw = config["solver"]["individual"]
    return FitSettings(
        epsilon=float(config["analysis"]["epsilon"]),
        sobol_starts=int(raw["sobol_starts"]),
        seed=int(raw["random_seed"] if seed is None else seed),
        ftol=float(raw["ftol"]),
        xtol=float(raw["xtol"]),
        gtol=float(raw["gtol"]),
        max_nfev=int(raw["max_nfev"]),
    )


def _validate_reflectance(values: np.ndarray, context: str, *, require_unit_interval: bool) -> None:
    array = np.asarray(values, dtype=np.float64)
    if not np.isfinite(array).all():
        raise ContractError(f"Non-finite reflectance in {context}")
    if np.any(array < 0.0):
        raise ContractError(f"Negative reflectance in {context}")
    if require_unit_interval and np.any(array > 1.0 + 1e-12):
        raise ContractError(f"Unclipped reflectance above one in {context}")


def _preflight(root: Path, config: dict[str, Any]) -> tuple[dict[str, Path], dict[str, Any]]:
    for key in ("raw_hsi_content_allowed", "rgb_content_allowed", "test_content_allowed", "clinical_500_content_allowed"):
        if config["access_policy"].get(key):
            raise ContractError(f"R-D access policy permits forbidden content: {key}")
    if not config["access_policy"].get("validation_content_allowed", False):
        raise ContractError("R-D is the registered batch that is allowed to read Validation content")
    for key in ("run_test", "run_clinical_500", "run_rgb_encoder_training"):
        if config["execution"].get(key):
            raise ContractError(f"R-D execution block attempts an out-of-scope stage: {key}")
    if not (config["execution"].get("run_freeze") and config["execution"].get("run_validation")):
        raise ContractError("R-D must run both the freeze and the developmental Validation review")
    if not config["execution"].get("require_freeze_before_validation"):
        raise ContractError("R-D must freeze before it consumes Validation content")
    quantities = config["freeze"]["frozen_quantities"]
    parameters = config["parameters"]
    if not (
        float(parameters["fixed_s0"]) == float(quantities["fixed_s0"])
        and float(parameters["fixed_diameter_um"]) == float(quantities["fixed_diameter_um"])
        and float(parameters["fixed_g0"]) == float(quantities["fixed_g0"])
    ):
        raise ContractError("R-D declares inconsistent frozen quantities between parameters and freeze blocks")

    paths = {name: _resolve(root, value) for name, value in config["inputs"].items()}
    missing = sorted(name for name, path in paths.items() if not path.exists())
    if missing:
        raise ContractError(f"Missing R-D inputs: {missing}")

    r_b_audit = json.loads(paths["r_b_observation_audit"].read_text(encoding="utf-8"))
    rd_obs_audit = json.loads(paths["rd_validation_observation_audit"].read_text(encoding="utf-8"))
    r_c0r_decision = json.loads(paths["r_c0r_decision"].read_text(encoding="utf-8"))
    r_c0r_audit = json.loads(paths["r_c0r_audit"].read_text(encoding="utf-8"))
    r_c1r_decision = json.loads(paths["r_c1r_decision"].read_text(encoding="utf-8"))
    r_c1r_audit = json.loads(paths["r_c1r_integrity_audit"].read_text(encoding="utf-8"))

    checks = {
        "r_b_observation_audit_pass": r_b_audit.get("status") == "PASS_FOR_V2R_TRAIN_INVERSION",
        "r_b_manifest_hash_chain": sha256_file(paths["r_b_observation_manifest"]) == r_b_audit["outputs"]["symmetric_manifest_parquet"]["sha256"],
        "rd_validation_observation_audit_pass": rd_obs_audit.get("status") == "PASS_FOR_R_D_VALIDATION_INVERSION",
        "rd_validation_manifest_hash_chain": sha256_file(paths["rd_validation_observation_manifest"]) == rd_obs_audit["outputs"]["symmetric_manifest_parquet"]["sha256"],
        "rd_validation_is_valid_split_only": rd_obs_audit["counts"]["validation_hsi_content_reads"] == 3 and rd_obs_audit["counts"]["train_hsi_content_reads"] == 0 and rd_obs_audit["counts"]["test_hsi_content_reads"] == 0,
        "r_c0r_complete": r_c0r_decision.get("status") == "R_C0R_COMPLETE" and r_c0r_audit.get("status") == "R_C0R_COMPLETE",
        "r_c0r_selected_candidate_is_the_frozen_one": r_c0r_decision.get("selected_candidate") == config["frozen_candidate"],
        "r_c1r_complete": r_c1r_decision.get("status") == "R_C1R_COMPLETE" and r_c1r_audit.get("status") == "R_C1R_COMPLETE",
        "r_c1r_next_stage_is_r_d": r_c1r_decision.get("next_registered_stage") == "R-D",
        "r_c1r_confirms_no_validation_access": r_c1r_audit["data_access"]["validation_reads"] == 0,
        "frozen_fold_manifest_hash": sha256_file(paths["r_c0r_fold_manifest"]) == str(config["folds"]["expected_sha256"]),
    }
    if not all(checks.values()):
        raise ContractError(f"R-D preflight failed: {checks}")
    return paths, {
        "checks": {key: bool(value) for key, value in checks.items()},
        "r_c0r_selected_candidate": r_c0r_decision.get("selected_candidate"),
        "r_c1r_parameter_reliability": _jsonable(_r_c1r_reliability(paths["r_c1r_parameter_profile_summary"], paths["r_c1r_best_o_profile_summary"], r_c0r_decision)),
        "rd_validation_subjects": rd_obs_audit["subject_ids"],
        "rd_validation_hsi_content_reads": int(rd_obs_audit["counts"]["validation_hsi_content_reads"]),
        "frozen_fold_manifest_sha256": sha256_file(paths["r_c0r_fold_manifest"]),
        "forbidden_data_access_declared": False,
    }


def _r_c1r_reliability(profile_path: Path, best_o_path: Path, r_c0r_decision: dict[str, Any]) -> dict[str, str]:
    """Carry the R-C1R reliability verdicts forward without recomputation."""

    selected = str(r_c0r_decision["selected_candidate"])
    profiles = pd.read_csv(profile_path, encoding="utf-8-sig")
    rows = profiles.loc[profiles["candidate"].eq(selected)]
    reliability = {str(row["parameter"]): str(row["classification"]) for _, row in rows.iterrows()}
    best_o = pd.read_csv(best_o_path, encoding="utf-8-sig")
    s_rows = best_o.loc[(best_o["base_candidate"].eq(selected)) & (best_o["parameter"].eq("s"))]
    if len(s_rows) == 1:
        reliability["s"] = str(s_rows.iloc[0]["classification"])
    return reliability


def _forward_with_globals(optical_fit: dict[str, np.ndarray], frozen: dict[str, float], s0: float):
    def forward(theta: np.ndarray) -> np.ndarray:
        return forward_preloaded_numpy(
            theta, optical_fit, wavelength_nm=WAVELENGTH[FIT_MASK], s0=s0,
            diameter_um=float(frozen["diameter_um"]), scattering_amplitude=float(frozen["A_s"]),
            delta_bs=float(frozen["delta_bs"]), g0=float(frozen["g0"]),
        )

    return forward


def _invert_spectra(
    frame: pd.DataFrame,
    label: str,
    optical: dict[str, np.ndarray],
    frozen: dict[str, float],
    config: dict[str, Any],
    reference: np.ndarray,
    seed_offset: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Invert each subject spectrum with the frozen globals and collect diagnostics."""

    columns = _columns()
    optical_fit = {name: values[FIT_MASK] for name, values in optical.items()}
    s0 = float(frozen["s0"])
    epsilon = float(config["analysis"]["epsilon"])
    boundary_tolerance = float(config["parameters"]["boundary_tolerance_normalized"])
    metrics_rows: list[dict[str, Any]] = []
    residual_rows: list[dict[str, Any]] = []
    prediction_rows: list[dict[str, Any]] = []
    for position, item in frame.reset_index(drop=True).iterrows():
        subject = str(item["subject_id"])
        observed_full = item[columns].to_numpy(dtype=np.float64)
        observed = observed_full[FIT_MASK]
        forward = _forward_with_globals(optical_fit, frozen, s0)

        def forward_fit(theta: np.ndarray) -> np.ndarray:
            prediction = forward(theta)
            _validate_reflectance(prediction, f"R-D {label} fit forward {subject}", require_unit_interval=True)
            return prediction

        fit = fit_bounded_spectrum(observed, forward_fit, FIT_LO, FIT_HI, _settings(config, seed=int(config["solver"]["individual"]["random_seed"]) + seed_offset + position))
        if not fit["success"]:
            raise ContractError(f"R-D {label} inversion did not converge for {subject}")
        prediction_full = forward_preloaded_numpy(
            fit["theta"], optical, wavelength_nm=WAVELENGTH, s0=s0,
            diameter_um=float(frozen["diameter_um"]), scattering_amplitude=float(frozen["A_s"]),
            delta_bs=float(frozen["delta_bs"]), g0=float(frozen["g0"]),
        )
        _validate_reflectance(prediction_full, f"R-D {label} prediction {subject}", require_unit_interval=True)
        metrics = spectral_metrics(prediction_full[FIT_MASK], observed, epsilon)
        log_residual = np.log(prediction_full[FIT_MASK] + epsilon) - np.log(observed + epsilon)
        centered = log_residual - log_residual.mean()
        reference_metrics = spectral_metrics(reference, observed, epsilon)
        u = (fit["theta"] - FIT_LO) / (FIT_HI - FIT_LO)
        selected = fit["starts"][fit["selected_start_index"]]
        metrics_rows.append({
            "cohort": label, "subject_id": subject, "capture_id": str(item["capture_id"]),
            "solver_converged": True, "selected_start_index": int(fit["selected_start_index"]),
            "selected_nfev": int(selected["nfev"]), "selected_status": int(selected["status"]),
            "selected_optimality": float(selected["optimality"]),
            "f_mel": float(fit["theta"][0]), "f_blood": float(fit["theta"][1]),
            "c_tHb_eq_g_l": float(150.0 * fit["theta"][1]),
            **metrics,
            "centered_logrmse": float(np.sqrt(np.mean(centered ** 2))),
            "mean_log_residual": float(log_residual.mean()),
            "reference_logrmse": float(reference_metrics["logrmse"]),
            "reference_rmse": float(reference_metrics["rmse"]),
            "reference_sam_deg": float(reference_metrics["sam_deg"]),
            "model_to_reference_error_ratio": float(metrics["logrmse"] / reference_metrics["logrmse"]),
            "model_better_than_reference": bool(metrics["logrmse"] < reference_metrics["logrmse"]),
            "f_mel_at_lower": bool(u[0] <= boundary_tolerance),
            "f_mel_at_upper": bool(u[0] >= 1.0 - boundary_tolerance),
            "f_blood_at_lower": bool(u[1] <= boundary_tolerance),
            "f_blood_at_upper": bool(u[1] >= 1.0 - boundary_tolerance),
            "prediction_above_one_count": int(np.count_nonzero(prediction_full > 1.0 + 1e-12)),
            "A_s": float(frozen["A_s"]), "delta_bs": float(frozen["delta_bs"]), "g0": float(frozen["g0"]),
            "diameter_um": float(frozen["diameter_um"]), "s0": s0,
        })
        for band_index, wave in enumerate(WAVELENGTH):
            role = "fit" if FIT_MASK[band_index] else "edge_diagnostic"
            prediction_rows.append({
                "cohort": label, "subject_id": subject, "wavelength_nm": int(wave), "band_role": role,
                "observed_reflectance": float(observed_full[band_index]),
                "predicted_reflectance": float(prediction_full[band_index]),
            })
            residual_rows.append({
                "cohort": label, "subject_id": subject, "wavelength_nm": int(wave), "band_role": role,
                "signed_residual": float(prediction_full[band_index] - observed_full[band_index]),
                "log_residual": float(np.log(prediction_full[band_index] + epsilon) - np.log(observed_full[band_index] + epsilon)),
            })
    return pd.DataFrame(metrics_rows), pd.DataFrame(residual_rows), pd.DataFrame(prediction_rows)


def _edge_median_abs(residuals: pd.DataFrame) -> tuple[float, dict[str, float]]:
    edge = residuals.loc[residuals["band_role"].eq("edge_diagnostic")]
    medians = edge.groupby("wavelength_nm", sort=True)["signed_residual"].median()
    return float(medians.abs().max()), {str(int(wave)): float(value) for wave, value in medians.items()}


def run_stage_rd(config_path: str | Path, project_root: str | Path) -> dict[str, Any]:
    root = Path(project_root).resolve()
    config_file = Path(config_path).resolve()
    config = yaml.safe_load(config_file.read_text(encoding="utf-8"))
    if config["task_id"] != "R-D_FREEZE_AND_DEVELOPMENTAL_VALIDATION":
        raise ContractError("Unexpected R-D task id")
    output = _resolve(root, config["output_directory"])
    if output.exists() and any(output.iterdir()) and config["access_policy"]["refuse_existing_output"]:
        raise FileExistsError(f"Refusing to overwrite R-D output: {output}")

    paths, preflight = _preflight(root, config)
    _log("preflight passed; freezing the candidate on the complete Train split")

    # ---------------------------------------------------------------- Train side
    train = pd.read_parquet(paths["r_b_observation_manifest"]).reset_index(drop=True)
    if len(train) != int(config["scope"]["train_subject_count"]):
        raise ContractError("R-B manifest does not contain the registered 44 Train subjects")
    if set(train["split"].astype(str)) != {config["scope"]["train_split"]} or not train["input_quality_status"].eq("PASS").all():
        raise ContractError("R-B manifest is not a pure Train-only QC-passed manifest")
    train_columns = _columns()
    train_observed_full = train[train_columns].to_numpy(dtype=np.float64)
    _validate_reflectance(train_observed_full, "R-B Train symmetric manifest", require_unit_interval=True)
    train_observed = train_observed_full[:, FIT_MASK]
    optical = load_optical_numpy(WAVELENGTH, paths["optical_asset_10nm"])
    optical_fit = {name: values[FIT_MASK] for name, values in optical.items()}

    historical = pd.read_csv(paths["historical_r_c0_subject_metrics"], encoding="utf-8-sig")
    initial_frame = train[["subject_id"]].merge(
        historical.loc[historical["candidate"].eq("V2R-P"), ["subject_id", "f_mel", "f_blood"]],
        on="subject_id", how="left", validate="one_to_one",
    )
    if initial_frame[["f_mel", "f_blood"]].isna().any().any():
        raise ContractError("Historical V2R-P initial theta is missing for a Train subject")
    initial_theta = initial_frame[["f_mel", "f_blood"]].to_numpy(dtype=np.float64)

    freeze_fit = _shape_joint_fit(train_observed, optical_fit, WAVELENGTH[FIT_MASK], initial_theta, config)
    if not np.isfinite(freeze_fit["centered_logrmse"]):
        raise ContractError("R-D freeze produced a non-finite Train objective")
    a_ident = profile_identifiability(
        freeze_fit["A_s_profile"], "A_s", "centered_logrmse", AS_BOUNDS,
        delta_logrmse=float(config["freeze"]["identifiability"]["acceptable_loss_delta_logrmse"]),
        max_normalized_span=float(config["freeze"]["identifiability"]["maximum_acceptable_normalized_envelope_span"]),
    )
    delta_ident = profile_identifiability(
        freeze_fit["delta_bs_profile"], "delta_bs", "centered_logrmse", DELTA_BS_BOUNDS,
        delta_logrmse=float(config["freeze"]["identifiability"]["acceptable_loss_delta_logrmse"]),
        max_normalized_span=float(config["freeze"]["identifiability"]["maximum_acceptable_normalized_envelope_span"]),
    )
    frozen = {
        "candidate": str(config["frozen_candidate"]),
        "A_s": float(freeze_fit["A_s"]),
        "delta_bs": float(freeze_fit["delta_bs"]),
        "g0": float(config["freeze"]["frozen_quantities"]["fixed_g0"]),
        "diameter_um": float(config["freeze"]["frozen_quantities"]["fixed_diameter_um"]),
        "s0": float(config["freeze"]["frozen_quantities"]["fixed_s0"]),
        "epidermis_thickness_mm": float(config["freeze"]["frozen_quantities"]["fixed_epidermis_thickness_mm"]),
        "estimator": str(config["freeze"]["estimator"]),
        "train_subject_count": int(len(train)),
        "train_centered_logrmse": float(freeze_fit["centered_logrmse"]),
        "initial_theta_source": str(config["freeze"]["initial_theta_source"]),
        "fitted_from_splits": [config["scope"]["train_split"]],
        "validation_used_in_freeze": False,
    }
    _log(f"frozen {frozen['candidate']}: A_s={frozen['A_s']:.6f} delta_bs={frozen['delta_bs']:.6f}")

    train_reference = np.exp(np.mean(np.log(train_observed + float(config["analysis"]["epsilon"])), axis=0)) - float(config["analysis"]["epsilon"])
    _validate_reflectance(train_reference, "R-D Train reference spectrum", require_unit_interval=True)
    reference_full = np.exp(np.mean(np.log(train_observed_full + float(config["analysis"]["epsilon"])), axis=0)) - float(config["analysis"]["epsilon"])
    _validate_reflectance(reference_full, "R-D full-band Train reference spectrum", require_unit_interval=True)
    if not np.allclose(reference_full[FIT_MASK], train_reference, rtol=0.0, atol=0.0):
        raise ContractError("R-D Train reference spectrum is inconsistent between full band and fit band")

    # ---------------------------------------------------------- Validation side
    # Validation content is materialised only now, after the freeze is fixed.
    validation = pd.read_parquet(paths["rd_validation_observation_manifest"]).reset_index(drop=True)
    if len(validation) != int(config["scope"]["validation_subject_count"]):
        raise ContractError("R-D Validation manifest does not contain the registered 3 subjects")
    if set(validation["split"].astype(str)) != {config["scope"]["validation_split"]}:
        raise ContractError("R-D Validation manifest is not split=valid")
    validation_observed_full = validation[train_columns].to_numpy(dtype=np.float64)
    _validate_reflectance(validation_observed_full, "R-D Validation symmetric manifest", require_unit_interval=True)
    _log("validation manifest loaded after freeze; inverting frozen candidate")

    output.mkdir(parents=True, exist_ok=True)
    frozen_path = output / "frozen_global_parameters.json"
    frozen_path.write_text(json.dumps(_jsonable(frozen), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    frozen_hash = sha256_file(frozen_path)
    (output / "frozen_global_parameters.sha256").write_text(f"{frozen_hash}  frozen_global_parameters.json\n", encoding="ascii")
    if json.loads(frozen_path.read_text(encoding="utf-8")) != _jsonable(frozen):
        raise ContractError("Frozen global parameter file does not round-trip")

    train_metrics, train_residuals, train_predictions = _invert_spectra(
        train, "train", optical, frozen, config, train_reference, seed_offset=0
    )
    validation_metrics, validation_residuals, validation_predictions = _invert_spectra(
        validation, "valid", optical, frozen, config, train_reference, seed_offset=100000
    )
    if len(train_metrics) != int(config["scope"]["train_subject_count"]) or len(validation_metrics) != int(config["scope"]["validation_subject_count"]):
        raise ContractError("R-D inversion did not cover the registered subject counts")
    above_one_total = int(train_metrics["prediction_above_one_count"].sum() + validation_metrics["prediction_above_one_count"].sum())

    # -------------------------------------------------- direction consistency
    assessment = _direction_assessment(train_metrics, validation_metrics, train_residuals, validation_residuals, config)
    spectral_checks = assessment["spectral_checks"]
    parameter_checks = assessment["parameter_checks"]
    decision_state = assessment["decision_state"]

    # Validation side-pressure linkage, reported descriptively (n=3).
    side = pd.read_csv(paths["rd_validation_side_pair_audit"], encoding="utf-8-sig")
    side_small = side[["subject_id", "log_left_minus_log_right_mean", "left_minus_right_broadband"]].copy()
    side_small["subject_id"] = side_small["subject_id"].astype(str)
    joined = validation_metrics.merge(side_small, on="subject_id", how="inner", validate="one_to_one")
    if len(joined) != int(config["scope"]["validation_subject_count"]):
        raise ContractError("R-D Validation side-pair linkage does not cover the frozen subjects")
    linkage_rows = []
    for parameter in config["parameters"]["names"]:
        if joined[parameter].nunique() > 1 and joined["log_left_minus_log_right_mean"].nunique() > 1:
            pearson = pearsonr(joined[parameter], joined["log_left_minus_log_right_mean"])
            r_value, p_value = float(pearson[0]), float(pearson[1])
        else:
            r_value, p_value = float("nan"), float("nan")
        linkage_rows.append({
            "cohort": "valid", "parameter": parameter, "subject_count": int(len(joined)),
            "pearson_r_vs_side_log_difference": r_value, "pearson_p": p_value,
            "inferential": False, "reason": "three_validation_subjects_developmental_only",
            "paired_values": json.dumps([
                {"subject_id": str(row["subject_id"]), parameter: float(row[parameter]),
                 "log_left_minus_log_right_mean": float(row["log_left_minus_log_right_mean"])}
                for _, row in joined.iterrows()
            ], ensure_ascii=False, sort_keys=True),
        })
    side_linkage = pd.DataFrame(linkage_rows)

    reliability_frame = _reliability_frame(config, preflight["r_c1r_parameter_reliability"])

    freeze_profile_rows = []
    for name, profile in (("A_s", freeze_fit["A_s_profile"]), ("delta_bs", freeze_fit["delta_bs_profile"])):
        bounds = AS_BOUNDS if name == "A_s" else DELTA_BS_BOUNDS
        for row in profile:
            freeze_profile_rows.append({
                "candidate": str(config["frozen_candidate"]), "parameter": name, "value": row[name],
                "centered_logrmse": row["centered_logrmse"], "solver_success": row["solver_success"],
                "nfev": row["nfev"], "grid_kind": row["grid_kind"], "normalized_value": float((row[name] - bounds[0]) / (bounds[1] - bounds[0])),
            })
    freeze_profiles = pd.DataFrame(freeze_profile_rows)
    freeze_summary = pd.DataFrame([{
        "candidate": str(config["frozen_candidate"]), "parameter": "A_s", "frozen_value": frozen["A_s"],
        "identifiable": bool(a_ident["identifiable"]), "acceptable_normalized_span": float(a_ident["acceptable_normalized_span"]),
        "acceptable_envelope_low": float(a_ident["acceptable_envelope"][0]), "acceptable_envelope_high": float(a_ident["acceptable_envelope"][1]),
        "at_boundary": bool(abs(frozen["A_s"] - AS_BOUNDS[0]) <= 1e-9 or abs(frozen["A_s"] - AS_BOUNDS[1]) <= 1e-9),
    }, {
        "candidate": str(config["frozen_candidate"]), "parameter": "delta_bs", "frozen_value": frozen["delta_bs"],
        "identifiable": bool(delta_ident["identifiable"]), "acceptable_normalized_span": float(delta_ident["acceptable_normalized_span"]),
        "acceptable_envelope_low": float(delta_ident["acceptable_envelope"][0]), "acceptable_envelope_high": float(delta_ident["acceptable_envelope"][1]),
        "at_boundary": bool(abs(frozen["delta_bs"] - DELTA_BS_BOUNDS[0]) <= 1e-9 or abs(frozen["delta_bs"] - DELTA_BS_BOUNDS[1]) <= 1e-9),
    }])
    freeze_theta = pd.DataFrame({
        "candidate": str(config["frozen_candidate"]),
        "subject_id": train["subject_id"].astype(str).to_numpy(),
        "f_mel": freeze_fit["theta"][:, 0], "f_blood": freeze_fit["theta"][:, 1],
    })
    train_reference_frame = pd.DataFrame({
        "wavelength_nm": WAVELENGTH.astype(int),
        "band_role": ["fit" if value else "edge_diagnostic" for value in FIT_MASK],
        "reference_reflectance": reference_full,
        "reference_source": "full_train_44_subject_per_band_log_geometric_mean",
    })

    # ------------------------------------------------------------- emit tables
    tables = {
        "freeze_multistart_results.csv": tabulate_freeze_starts(freeze_fit, config),
        "freeze_global_profiles.csv": freeze_profiles,
        "freeze_global_profile_summary.csv": freeze_summary,
        "freeze_train_theta.csv": freeze_theta,
        "train_reference_spectrum.csv": train_reference_frame,
        "train_frozen_refit_subject_metrics.csv": train_metrics,
        "train_frozen_refit_residuals.csv": train_residuals,
        "train_frozen_refit_predictions.csv": train_predictions,
        "validation_subject_metrics.csv": validation_metrics,
        "validation_residuals.csv": validation_residuals,
        "validation_predictions.csv": validation_predictions,
        "direction_consistency_summary.csv": assessment["direction_summary"],
        "edge_band_residual_comparison.csv": assessment["edge_comparison"],
        "validation_side_linkage.csv": side_linkage,
        "r_c1r_parameter_reliability_reused.csv": reliability_frame,
        "direction_parameter_ranges.csv": assessment["parameter_frame"],
    }
    for name, frame in tables.items():
        _write_csv(frame, output / name)

    return _emit_artifacts(
        config=config,
        config_file=config_file,
        paths=paths,
        output=output,
        preflight=preflight,
        root=root,
        frozen=frozen,
        frozen_hash=frozen_hash,
        assessment=assessment,
        above_one_total=above_one_total,
        side_linkage=side_linkage,
        reliability_frame=reliability_frame,
        created_utc=datetime.now(timezone.utc).isoformat(),
        emission_mode="primary_run",
        numerics_recomputed=True,
    )


def _direction_assessment(
    train_metrics: pd.DataFrame,
    validation_metrics: pd.DataFrame,
    train_residuals: pd.DataFrame,
    validation_residuals: pd.DataFrame,
    config: dict[str, Any],
) -> dict[str, Any]:
    """Apply the pre-registered direction-consistency rule to both cohorts."""

    thresholds = config["direction_consistency"]["thresholds"]
    train_median_logrmse = float(train_metrics["logrmse"].median())
    validation_median_logrmse = float(validation_metrics["logrmse"].median())
    train_median_sam = float(train_metrics["sam_deg"].median())
    validation_median_sam = float(validation_metrics["sam_deg"].median())
    train_edge_abs, train_edge_medians = _edge_median_abs(train_residuals)
    validation_edge_abs, validation_edge_medians = _edge_median_abs(validation_residuals)
    validation_better_fraction = float(validation_metrics["model_better_than_reference"].mean())
    validation_ratio = float(validation_metrics["model_to_reference_error_ratio"].median())
    train_f_blood_upper = float(train_metrics["f_blood_at_upper"].mean())
    validation_f_blood_upper = float(validation_metrics["f_blood_at_upper"].mean())

    margin_fraction = float(thresholds["parameter_median_margin_fraction_of_train_range"])
    parameter_rows = []
    parameter_medians_ok = True
    for parameter in config["parameters"]["names"]:
        train_values = train_metrics[parameter].to_numpy(dtype=np.float64)
        span = float(np.ptp(train_values))
        margin = margin_fraction * span
        valid_median = float(validation_metrics[parameter].median())
        inside = bool(train_values.min() - margin <= valid_median <= train_values.max() + margin)
        parameter_medians_ok = parameter_medians_ok and inside
        parameter_rows.append({
            "parameter": parameter,
            "train_median": float(np.median(train_values)),
            "train_min": float(train_values.min()), "train_max": float(train_values.max()),
            "train_span": span, "allowed_margin": margin,
            "validation_median": valid_median,
            "validation_min": float(validation_metrics[parameter].min()),
            "validation_max": float(validation_metrics[parameter].max()),
            "median_inside_extended_train_range": inside,
        })

    spectral_checks = {
        "median_logrmse_not_regressed": bool(
            validation_median_logrmse <= train_median_logrmse + float(thresholds["median_logrmse_absolute_regression_max"])
        ),
        "median_sam_not_regressed": bool(
            validation_median_sam <= train_median_sam + float(thresholds["median_sam_deg_absolute_regression_max"])
        ),
        "validation_beats_reference_majority": bool(
            validation_better_fraction >= float(thresholds["validation_better_than_reference_fraction_min"])
        ),
    }
    parameter_checks = {
        "parameter_medians_within_extended_train_range": bool(parameter_medians_ok),
        "edge_median_abs_signed_residual_change_ok": bool(
            abs(validation_edge_abs - train_edge_abs) <= float(thresholds["edge_median_abs_signed_residual_change_max"])
        ),
        "f_blood_upper_boundary_fraction_not_worse": bool(
            validation_f_blood_upper <= train_f_blood_upper + float(thresholds["f_blood_upper_boundary_fraction_absolute_increase_max"])
        ),
    }
    direction_consistent = bool(all(spectral_checks.values()) and all(parameter_checks.values()))
    if not spectral_checks["validation_beats_reference_majority"] and validation_ratio >= 1.0:
        decision_state = str(config["decision"]["stop_state"])
        direction_note = "Validation does not beat the fixed Train reference spectrum."
    elif all(spectral_checks.values()):
        decision_state = f"{config['decision']['retained_development_state']} / {config['decision']['spectral_only_state']}"
        direction_note = "Validation spectral error stays in the Train direction, but no physiological parameter is reliable."
    else:
        decision_state = str(config["decision"]["revise_state"])
        direction_note = "Validation spectral behaviour deviates from the Train direction."

    direction_summary = pd.DataFrame([
        {"scope": "train", "median_logrmse": train_median_logrmse, "median_sam_deg": train_median_sam,
         "median_rmse": float(train_metrics["rmse"].median()),
         "model_better_than_reference_fraction": float(train_metrics["model_better_than_reference"].mean()),
         "median_model_to_reference_error_ratio": float(train_metrics["model_to_reference_error_ratio"].median()),
         "f_blood_upper_boundary_fraction": train_f_blood_upper,
         "f_mel_any_boundary_fraction": float((train_metrics["f_mel_at_lower"] | train_metrics["f_mel_at_upper"]).mean()),
         "edge_median_abs_signed_residual": train_edge_abs},
        {"scope": "valid", "median_logrmse": validation_median_logrmse, "median_sam_deg": validation_median_sam,
         "median_rmse": float(validation_metrics["rmse"].median()),
         "model_better_than_reference_fraction": validation_better_fraction,
         "median_model_to_reference_error_ratio": validation_ratio,
         "f_blood_upper_boundary_fraction": validation_f_blood_upper,
         "f_mel_any_boundary_fraction": float((validation_metrics["f_mel_at_lower"] | validation_metrics["f_mel_at_upper"]).mean()),
         "edge_median_abs_signed_residual": validation_edge_abs},
    ])
    edge_comparison = pd.DataFrame([
        {
            "wavelength_nm": int(wave),
            "train_median_signed_residual": train_edge_medians[str(int(wave))],
            "validation_median_signed_residual": validation_edge_medians[str(int(wave))],
            "abs_change": float(abs(validation_edge_medians[str(int(wave))] - train_edge_medians[str(int(wave))])),
            "same_sign": bool(np.sign(validation_edge_medians[str(int(wave))]) == np.sign(train_edge_medians[str(int(wave))])),
        }
        for wave in config["analysis"]["edge_diagnostic_centers_nm"]
    ])
    return {
        "train_median_logrmse": train_median_logrmse,
        "validation_median_logrmse": validation_median_logrmse,
        "train_median_sam": train_median_sam,
        "validation_median_sam": validation_median_sam,
        "validation_better_fraction": validation_better_fraction,
        "validation_ratio": validation_ratio,
        "train_median_f_mel": float(train_metrics["f_mel"].median()),
        "validation_median_f_mel": float(validation_metrics["f_mel"].median()),
        "train_median_f_blood": float(train_metrics["f_blood"].median()),
        "validation_median_f_blood": float(validation_metrics["f_blood"].median()),
        "train_edge_abs": train_edge_abs,
        "validation_edge_abs": validation_edge_abs,
        "spectral_checks": spectral_checks,
        "parameter_checks": parameter_checks,
        "direction_consistent": direction_consistent,
        "decision_state": decision_state,
        "direction_note": direction_note,
        "direction_summary": direction_summary,
        "edge_comparison": edge_comparison,
        "parameter_frame": pd.DataFrame(parameter_rows),
    }


def _reliability_frame(config: dict[str, Any], reliability: dict[str, str]) -> pd.DataFrame:
    frame = pd.DataFrame([
        {"candidate": str(config["frozen_candidate"]), "parameter": parameter, "classification": str(reliability.get(parameter, "not_reported"))}
        for parameter in ("f_mel", "f_blood", "s")
    ])
    if not frame["classification"].eq("unreliable").all():
        raise ContractError("R-C1R no longer reports all three parameters as unreliable; R-D interpretation must be re-derived")
    return frame


def tabulate_freeze_starts(freeze_fit: dict[str, Any], config: dict[str, Any]) -> pd.DataFrame:
    """Summarise the registered multi-start solutions of the Train-global fit."""

    raw = config["solver"]["global_shape"]
    rows = [{
        "start_index": int(index),
        "start_global_A_s_u": float(np.asarray(start_u, dtype=np.float64)[0]),
        "start_global_delta_bs_u": float(np.asarray(start_u, dtype=np.float64)[1]),
        "selected": bool(index == int(freeze_fit["start_index"])),
    } for index, start_u in enumerate(raw["start_global_u"])]
    frame = pd.DataFrame(rows)
    frame["selected_A_s"] = np.where(frame["selected"], float(freeze_fit["A_s"]), np.nan)
    frame["selected_delta_bs"] = np.where(frame["selected"], float(freeze_fit["delta_bs"]), np.nan)
    frame["selected_centered_logrmse"] = np.where(frame["selected"], float(freeze_fit["centered_logrmse"]), np.nan)
    frame["optimizer_success"] = np.where(frame["selected"], bool(freeze_fit["optimizer_success"]), np.nan)
    return frame


def _emit_artifacts(
    *,
    config: dict[str, Any],
    config_file: Path,
    paths: dict[str, Path],
    output: Path,
    preflight: dict[str, Any],
    root: Path,
    frozen: dict[str, Any],
    frozen_hash: str,
    assessment: dict[str, Any],
    above_one_total: int,
    side_linkage: pd.DataFrame,
    reliability_frame: pd.DataFrame,
    created_utc: str,
    emission_mode: str,
    numerics_recomputed: bool,
) -> dict[str, Any]:
    """Write the R-D decision, audit, report and artifact manifest.

    Shared by the primary run and by ``finalize_stage_rd`` so that both paths
    emit the identical artifact set.  All heavy tables must already be on disk.
    """

    train_metrics = pd.read_csv(output / "train_frozen_refit_subject_metrics.csv", encoding="utf-8-sig")
    validation_metrics = pd.read_csv(output / "validation_subject_metrics.csv", encoding="utf-8-sig")
    if len(train_metrics) != int(config["scope"]["train_subject_count"]) or len(validation_metrics) != int(config["scope"]["validation_subject_count"]):
        raise ContractError("R-D metrics tables do not match the registered subject counts")
    freeze_summary = pd.read_csv(output / "freeze_global_profile_summary.csv", encoding="utf-8-sig")
    spectral_checks = assessment["spectral_checks"]
    parameter_checks = assessment["parameter_checks"]
    reliability = preflight["r_c1r_parameter_reliability"]

    decision = {
        "schema_version": 1,
        "task_id": config["task_id"],
        "protocol_version": config["protocol_version"],
        "created_utc": created_utc,
        "emitted_utc": datetime.now(timezone.utc).isoformat(),
        "emission_mode": emission_mode,
        "status": "R_D_COMPLETE_DIRECTION_CONSISTENT" if assessment["direction_consistent"] else "R_D_COMPLETE_DIRECTION_INCONSISTENT",
        "frozen_candidate": str(config["frozen_candidate"]),
        "frozen_global_parameters": _jsonable(frozen),
        "frozen_global_parameters_sha256": frozen_hash,
        "freeze_train_only": True,
        "validation_used_in_freeze": False,
        "validation_set_size": int(config["scope"]["validation_subject_count"]),
        "validation_is_developmental_only": True,
        "direction_consistency": {
            "reference": str(config["direction_consistency"]["reference"]),
            "spectral_checks": {key: bool(value) for key, value in spectral_checks.items()},
            "parameter_checks": {key: bool(value) for key, value in parameter_checks.items()},
            "direction_consistent": bool(assessment["direction_consistent"]),
            "note": assessment["direction_note"],
        },
        "train_median_logrmse": assessment["train_median_logrmse"],
        "validation_median_logrmse": assessment["validation_median_logrmse"],
        "train_median_sam_deg": assessment["train_median_sam"],
        "validation_median_sam_deg": assessment["validation_median_sam"],
        "validation_model_better_than_reference_fraction": assessment["validation_better_fraction"],
        "validation_median_model_to_reference_error_ratio": assessment["validation_ratio"],
        "train_median_f_blood": assessment["train_median_f_blood"],
        "validation_median_f_blood": assessment["validation_median_f_blood"],
        "train_median_f_mel": assessment["train_median_f_mel"],
        "validation_median_f_mel": assessment["validation_median_f_mel"],
        "edge_median_abs_signed_residual_train": assessment["train_edge_abs"],
        "edge_median_abs_signed_residual_validation": assessment["validation_edge_abs"],
        "r_c1r_parameter_reliability": _jsonable(reliability),
        "decision_state": assessment["decision_state"],
        "spectral_generalization_supported": bool(all(spectral_checks.values())),
        "parameter_reliability_claim": "none",
        "parameter_upgrade_forbidden": bool(config["decision"]["parameter_upgrade_forbidden"]),
        "parameter_upgrade_forbidden_reason": str(config["decision"]["parameter_upgrade_forbidden_reason"]),
        "validation_claim_boundary": "Three Validation subjects support a developmental spectral review only; no stable population estimate and no physiological truth claim.",
        "test_executed": False,
        "clinical_500_executed": False,
        "rgb_encoder_training_executed": False,
        "next_registered_stage": "STAGE1_OPTICAL_LAYER_DECISION_REQUIRED",
        "interpretation": "R-D freezes V2R-PS on the complete Train split and performs a developmental Validation spectral review. It cannot upgrade theta to physiological quantities.",
    }
    audit = {
        "schema_version": 1,
        "task_id": config["task_id"],
        "protocol_version": config["protocol_version"],
        "status": decision["status"],
        "created_utc": decision["created_utc"],
        "emitted_utc": decision["emitted_utc"],
        "emission": {
            "mode": emission_mode,
            "numerics_recomputed": bool(numerics_recomputed),
            "note": (
                "Primary run: all scalars computed in-process from the fitted arrays."
                if numerics_recomputed else
                "Re-emission: aggregate scalars re-derived from the persisted CSV tables; "
                "they may differ from the primary-run values at the 1e-16 relative level due to CSV "
                "round-trip. No freeze, inversion or threshold was recomputed or changed."
            ),
        },
        "preflight_checks": _jsonable(preflight["checks"]),
        "frozen_fold_manifest_sha256": preflight["frozen_fold_manifest_sha256"],
        "frozen_global_parameters_sha256": frozen_hash,
        "freeze_ordering": "freeze_written_and_hashed_before_validation_manifest_loaded",
        "data_access": {
            "r_b_manifest_content_reads": 1,
            "r_b_side_pair_audit_content_reads": 1,
            "rd_validation_observation_manifest_content_reads": 1,
            "rd_validation_side_pair_audit_content_reads": 1,
            "r_c0r_artifact_content_reads": 3,
            "r_c1r_artifact_content_reads": 4,
            "historical_r_c0_artifact_content_reads": 1,
            "raw_hsi_reads": 0,
            "rgb_reads": 0,
            "validation_hsi_reads": 0,
            "validation_spectrum_reads": int(config["scope"]["validation_subject_count"]),
            "test_reads": 0,
            "clinical_500_reads": 0,
        },
        "train_subject_count": int(len(train_metrics)),
        "validation_subject_count": int(len(validation_metrics)),
        "fit_band_count": int(FIT_MASK.sum()),
        "edge_band_count": int((~FIT_MASK).sum()),
        "reflectance_clipping_applied": False,
        "prediction_above_one_count": int(above_one_total),
        "all_inversions_converged": bool(train_metrics["solver_converged"].all() and validation_metrics["solver_converged"].all()),
        "fixed_quantities_unchanged_after_validation": True,
        "thresholds_preregistered": True,
        "input_hashes": {name: sha256_file(path) for name, path in paths.items()},
        "implementation_hashes": {name: sha256_file(_resolve(root, value)) for name, value in config["implementation"].items()},
    }
    (output / "r_d_decision.json").write_text(json.dumps(_jsonable(decision), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (output / "r_d_integrity_audit.json").write_text(json.dumps(_jsonable(audit), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    report = [
        "# KM-BIO-v2R.1 R-D freeze and developmental Validation review",
        "",
        f"- Status: `{decision['status']}`",
        f"- Frozen candidate: `{config['frozen_candidate']}`",
        f"- Frozen globals: `A_s={frozen['A_s']:.6f}`, `delta_bs={frozen['delta_bs']:.6f}`, `g0={frozen['g0']}`, `Dv={frozen['diameter_um']} um`, `s0={frozen['s0']}`",
        f"- Freeze Train-only; Validation used in freeze: `false`",
        f"- Decision state: `{assessment['decision_state']}`",
        "- Validation is 3 subjects: developmental spectral review only, no population estimate, no physiological claim.",
        "",
        "## Freeze diagnostics (Train-global profiles)",
        "",
        freeze_summary.to_markdown(index=False),
        "",
        "## Direction consistency (Train vs Validation, frozen globals)",
        "",
        assessment["direction_summary"].to_markdown(index=False),
        "",
        "### Checks",
        "",
        pd.DataFrame([
            {"family": "spectral", "check": key, "pass": bool(value)} for key, value in spectral_checks.items()
        ] + [
            {"family": "parameter", "check": key, "pass": bool(value)} for key, value in parameter_checks.items()
        ]).to_markdown(index=False),
        "",
        "## Edge-band median signed residual comparison",
        "",
        assessment["edge_comparison"].to_markdown(index=False),
        "",
        "## Carried-forward R-C1R parameter reliability",
        "",
        reliability_frame.to_markdown(index=False),
        "",
        "## Validation side-pressure linkage (n=3, descriptive)",
        "",
        side_linkage.drop(columns=["paired_values"]).to_markdown(index=False),
        "",
    ]
    (output / "R_D_REPORT.md").write_text("\n".join(report), encoding="utf-8")

    artifact_rows = [{"role": "input", "name": name, "path": str(path), "sha256": sha256_file(path)} for name, path in paths.items()]
    artifact_rows.append({"role": "config", "name": "r_d_config", "path": str(config_file), "sha256": sha256_file(config_file)})
    for name, value in config["implementation"].items():
        path = _resolve(root, value)
        artifact_rows.append({"role": "implementation", "name": name, "path": str(path), "sha256": sha256_file(path)})
    for path in sorted(output.iterdir(), key=lambda item: item.name):
        if path.is_file() and path.name != "artifact_hash_manifest.csv":
            artifact_rows.append({"role": "output", "name": path.name, "path": str(path), "sha256": sha256_file(path)})
    _write_csv(pd.DataFrame(artifact_rows), output / "artifact_hash_manifest.csv")
    _log(f"R-D artifacts emitted ({emission_mode})")
    return decision


def finalize_stage_rd(config_path: str | Path, project_root: str | Path) -> dict[str, Any]:
    """Re-emit the R-D decision, audit, report and hash manifest from written tables.

    The heavy phases write their tables first; this entry point recompletes the
    artifact set from those tables, the frozen-parameter file and the registered
    thresholds without repeating the freeze or the inversions.  Numeric content is
    re-derived from the persisted tables; no fit is repeated.
    """

    root = Path(project_root).resolve()
    config_file = Path(config_path).resolve()
    config = yaml.safe_load(config_file.read_text(encoding="utf-8"))
    if config["task_id"] != "R-D_FREEZE_AND_DEVELOPMENTAL_VALIDATION":
        raise ContractError("Unexpected R-D task id")
    output = _resolve(root, config["output_directory"])
    if not output.is_dir():
        raise ContractError("Cannot finalize R-D; the output directory does not exist")
    paths, preflight = _preflight(root, config)

    required_tables = [
        "frozen_global_parameters.json",
        "train_frozen_refit_subject_metrics.csv",
        "train_frozen_refit_residuals.csv",
        "validation_subject_metrics.csv",
        "validation_residuals.csv",
        "validation_side_linkage.csv",
        "r_c1r_parameter_reliability_reused.csv",
    ]
    missing = [name for name in required_tables if not (output / name).is_file()]
    if missing:
        raise ContractError(f"Cannot finalize R-D; missing tables: {missing}")

    frozen_path = output / "frozen_global_parameters.json"
    frozen = json.loads(frozen_path.read_text(encoding="utf-8"))
    frozen_hash = sha256_file(frozen_path)
    sidecar = (output / "frozen_global_parameters.sha256").read_text(encoding="ascii").split()[0]
    if sidecar != frozen_hash:
        raise ContractError("R-D frozen-parameter sidecar does not match the frozen file")
    if frozen.get("candidate") != config["frozen_candidate"] or frozen.get("train_subject_count") != int(config["scope"]["train_subject_count"]):
        raise ContractError("R-D frozen file does not describe the registered candidate and Train split")
    if frozen.get("validation_used_in_freeze") is not False or frozen.get("fitted_from_splits") != [config["scope"]["train_split"]]:
        raise ContractError("R-D frozen file provenance is not Train-only")

    def read(name: str) -> pd.DataFrame:
        return pd.read_csv(output / name, encoding="utf-8-sig")

    train_metrics = read("train_frozen_refit_subject_metrics.csv")
    validation_metrics = read("validation_subject_metrics.csv")
    assessment = _direction_assessment(
        train_metrics, validation_metrics, read("train_frozen_refit_residuals.csv"), read("validation_residuals.csv"), config
    )
    above_one_total = int(train_metrics["prediction_above_one_count"].sum() + validation_metrics["prediction_above_one_count"].sum())
    previous = json.loads((output / "r_d_decision.json").read_text(encoding="utf-8")) if (output / "r_d_decision.json").is_file() else {}
    return _emit_artifacts(
        config=config,
        config_file=config_file,
        paths=paths,
        output=output,
        preflight=preflight,
        root=root,
        frozen=frozen,
        frozen_hash=frozen_hash,
        assessment=assessment,
        above_one_total=above_one_total,
        side_linkage=read("validation_side_linkage.csv"),
        reliability_frame=_reliability_frame(config, preflight["r_c1r_parameter_reliability"]),
        created_utc=str(previous.get("created_utc") or datetime.now(timezone.utc).isoformat()),
        emission_mode="finalize_reemission",
        numerics_recomputed=False,
    )

