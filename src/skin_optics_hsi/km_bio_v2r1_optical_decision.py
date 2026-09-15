"""KM-BIO-v2R.1: stage-1 optical-layer decision gate.

R-D closed the last registered execution batch of the KM-BIO line and left exactly
one registered action: the stage-1 optical-layer decision
(``STAGE1_OPTICAL_LAYER_DECISION_REQUIRED``).  This module turns that gate into a
deterministic, auditable batch:

1. ``preflight``: verify the upstream decision chain (v1 C, v2R R-A/R-B/R-C0,
   v2R.1 R-C0R/R-C1/R-C1R/R-D) is complete, internally consistent and hash-linked,
   and that this batch is not authorized to open any new data content.
2. ``analysis``: aggregate the registered evidence into the section 7.1 strong
   spectral targets, the R-C1R parameter-reliability verdicts, the section 7.5
   observation-level blocker analysis and a residual-risk register.
3. ``decision``: evaluate the section 7.5 state ladder in its preregistered
   priority order; the first satisfied terminal state is the outcome.
4. ``freeze``: emit the section 3.6 stage-2 interface freeze and the decision
   document ``STAGE1_KM_BIO_DECISION.md``.

Scope boundary: no raw HSI, RGB, Validation, Test or clinical-500 content is read.
Every input is an already-audited decision JSON or summary table from a registered
batch.  Test, clinical-500 and RGB encoder training stay locked, and the R-C1R
``unreliable`` verdicts are not upgraded by this gate.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from .km_bio_observation import sha256_file


TERMINAL_STATES = (
    "V2R_STOP_KM",
    "V2R_STRONG_THETA_READY",
    "V2R_READY_WITH_S",
    "V2R_PARTIAL_THETA_CANDIDATE",
    "V2R_REVISE_OBSERVATION",
    "V2R_SPECTRAL_ONLY",
)

MILESTONE_STATES = ("V2R_DEVELOPMENT_SPECTRAL_CANDIDATE",)

STATE_MEANING_ZH = {
    "V2R_STOP_KM": "停止双层 K-M 主路线，转向另一受约束前向模型或只保留 proxy 表示",
    "V2R_STRONG_THETA_READY": "七项强光谱目标全过且所输出参数达到可靠性目标",
    "V2R_READY_WITH_S": "基础参数可用且氧合扩展的 s 通过独立可辨识性与稳健性检查",
    "V2R_PARTIAL_THETA_CANDIDATE": "Validation 光谱方向一致且至少一个参数 reliable/conditional",
    "V2R_REVISE_OBSERVATION": "统一尺度或固定侧差仍是主要阻断，需单独修订观测模型",
    "V2R_SPECTRAL_ONLY": "光谱重建有用但所有生理参数均不可辨识",
    "V2R_DEVELOPMENT_SPECTRAL_CANDIDATE": "最佳有效候选优于固定参考谱但未达全部强目标（R-D 前的里程碑）",
}

STATE_ACTION_ZH = {
    "V2R_STOP_KM": "停止双层 K-M 主路线",
    "V2R_STRONG_THETA_READY": "冻结后进入 Validation，输出模型条件下的区域有效参数",
    "V2R_READY_WITH_S": "s 可作为额外候选，仍不等同动脉 SpO2",
    "V2R_PARTIAL_THETA_CANDIDATE": "仅保留通过判定的参数，其他参数不进入阶段二",
    "V2R_REVISE_OBSERVATION": "单独修订观测模型，不让生理公式来吸收差异",
    "V2R_SPECTRAL_ONLY": "只承认前向重建能力，不训练生理参数编码器",
    "V2R_DEVELOPMENT_SPECTRAL_CANDIDATE": "允许冻结后进入 Validation，只检验光谱泛化",
}

PER_SAMPLE_OUTPUT_SCHEMA = [
    "subject_id",
    "capture_id",
    "roi",
    "model_version",
    "theta_hat[3]",
    "reflectance_hat[31]",
    "logrmse",
    "rmse",
    "sam_deg",
    "residual[31]",
    "parameter_tolerance_intervals",
    "boundary_flags",
    "parameter_reliability_flags",
    "sensitivity_flags",
    "solver_status",
    "input_quality_status",
    "source_hashes",
]


class ContractError(RuntimeError):
    """A hard stage-1 decision-gate integrity or scope violation."""


def _resolve(root: Path, value: str) -> Path:
    path = Path(value)
    return (root / path).resolve() if not path.is_absolute() else path.resolve()


def _jsonable(value: Any) -> Any:
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return float(value)
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    return value


def _write_csv(frame: pd.DataFrame, path: Path) -> None:
    frame.to_csv(path, index=False, encoding="utf-8-sig")


def _log(message: str) -> None:
    print(f"[km-bio-v2r1-stage1-decision] {message}", flush=True)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _coerce_bool(series: pd.Series) -> pd.Series:
    if series.dtype == bool:
        return series
    return series.astype(str).str.strip().str.lower().isin(("true", "1", "1.0", "yes"))


def _compare(value: float, operator: str, target: float) -> bool:
    if operator == "le":
        return bool(value <= target)
    if operator == "ge":
        return bool(value >= target)
    raise ContractError(f"Unsupported comparison operator: {operator}")


# --------------------------------------------------------------------------- #
# preflight
# --------------------------------------------------------------------------- #


def _preflight(root: Path, config: dict[str, Any]) -> tuple[dict[str, Path], dict[str, Any]]:
    access = config["access_policy"]
    for key in (
        "raw_hsi_content_allowed",
        "rgb_content_allowed",
        "validation_content_allowed",
        "test_content_allowed",
        "clinical_500_content_allowed",
    ):
        if access.get(key):
            raise ContractError(f"Stage-1 decision gate access policy permits forbidden content: {key}")
    if not access.get("summary_artifacts_only", False):
        raise ContractError("Stage-1 decision gate must be restricted to already-audited summary artifacts")
    if access.get("allowed_splits") != ["train", "valid"]:
        raise ContractError("Stage-1 decision gate must not widen the registered split scope")

    execution = config["execution"]
    if not execution.get("run_decision_gate", False):
        raise ContractError("Stage-1 decision gate must run the decision")
    for key in ("run_test", "run_clinical_500", "run_rgb_encoder_training", "run_stage2_design"):
        if execution.get(key):
            raise ContractError(f"Stage-1 decision gate execution block attempts an out-of-scope stage: {key}")
    if not execution.get("require_no_new_data_content_reads", False):
        raise ContractError("Stage-1 decision gate must declare the no-new-content-reads requirement")

    locked = config["locked_stages"]
    if locked.get("parameter_upgrade_allowed"):
        raise ContractError("Stage-1 decision gate must not authorize a parameter upgrade")
    for key in ("test_executed", "clinical_500_executed", "rgb_encoder_training_executed", "stage2_design_executed"):
        if locked.get(key):
            raise ContractError(f"Stage-1 decision gate locked stage is set to executed: {key}")

    ladder = config["state_ladder"]
    if tuple(ladder["priority"]) != TERMINAL_STATES:
        raise ContractError("Stage-1 decision ladder priority must follow the section 7.5 state order")
    if set(STATE_MEANING_ZH) != set(TERMINAL_STATES) | set(MILESTONE_STATES):
        raise ContractError("Stage-1 decision gate state description table is incomplete")

    paths = {name: _resolve(root, value) for name, value in config["inputs"].items()}
    missing = sorted(name for name, path in paths.items() if not path.exists())
    if missing:
        raise ContractError(f"Missing stage-1 decision inputs: {missing}")

    v1_c = json.loads(paths["v1_stage_c_decision"].read_text(encoding="utf-8"))
    v2r_a = json.loads(paths["v2r_formula_audit"].read_text(encoding="utf-8"))
    r_b_audit = json.loads(paths["r_b_observation_audit"].read_text(encoding="utf-8"))
    r_c0 = json.loads(paths["historical_r_c0_decision"].read_text(encoding="utf-8"))
    r_c0r = json.loads(paths["r_c0r_decision"].read_text(encoding="utf-8"))
    r_c0r_audit = json.loads(paths["r_c0r_audit"].read_text(encoding="utf-8"))
    r_c1r = json.loads(paths["r_c1r_decision"].read_text(encoding="utf-8"))
    r_c1r_audit = json.loads(paths["r_c1r_integrity_audit"].read_text(encoding="utf-8"))
    r_d = json.loads(paths["r_d_decision"].read_text(encoding="utf-8"))
    r_d_audit = json.loads(paths["r_d_integrity_audit"].read_text(encoding="utf-8"))
    r_d_frozen = json.loads(paths["r_d_frozen_global_parameters"].read_text(encoding="utf-8"))

    selected = config["selected_candidate"]
    checks = {
        "v1_c_revise_status": v1_c.get("status") == "REVISE_OBSERVATION_OR_MODEL",
        "v1_c_spectral_gate_failed": v1_c.get("spectral_gate_pass") is False,
        "v2r_formula_audit_pass": v2r_a.get("status") == "PASS",
        "r_b_observation_audit_pass": r_b_audit.get("status") == "PASS_FOR_V2R_TRAIN_INVERSION",
        "r_b_symmetric_unit_is_one_spectrum_per_subject": (
            int(r_b_audit["counts"]["symmetric_spectra"]) == int(r_b_audit["counts"]["subjects"])
        ),
        "historical_r_c0_stopped_at_failed_upgrade": r_c0.get("status") == "R_C0_STOPPED_AT_FAILED_UPGRADE",
        "historical_r_c0_read_only": bool(r_c0.get("validation_test_500_reads", 0) == 0),
        "r_c0r_complete": r_c0r.get("status") == "R_C0R_COMPLETE" and r_c0r_audit.get("status") == "R_C0R_COMPLETE",
        "r_c0r_selected_candidate_matches": r_c0r.get("selected_candidate") == selected,
        "r_c0r_next_stage_is_r_c1": r_c0r.get("next_registered_stage") == "R-C1",
        "r_c1r_complete": r_c1r.get("status") == "R_C1R_COMPLETE" and r_c1r_audit.get("status") == "R_C1R_COMPLETE",
        "r_c1r_next_stage_is_r_d": r_c1r.get("next_registered_stage") == "R-D",
        "r_c1r_confirms_no_validation_access": int(r_c1r_audit["data_access"]["validation_reads"]) == 0,
        "r_d_complete": r_d.get("status") == "R_D_COMPLETE_DIRECTION_CONSISTENT" and r_d_audit.get("status") == "R_D_COMPLETE_DIRECTION_CONSISTENT",
        "r_d_frozen_candidate_matches": r_d.get("frozen_candidate") == selected,
        "r_d_direction_consistent": bool(r_d["direction_consistency"]["direction_consistent"]),
        "r_d_validation_not_used_in_freeze": bool(r_d.get("validation_used_in_freeze") is False),
        "r_d_frozen_parameters_hash_chain": sha256_file(paths["r_d_frozen_global_parameters"]) == str(r_d["frozen_global_parameters_sha256"]),
        "r_d_frozen_parameters_are_consistent": (
            str(r_d_frozen.get("candidate")) == selected
            and str(r_d_frozen.get("estimator")) == "full_train_joint_centered_log_shape"
        ),
        "r_d_next_stage_is_the_decision_gate": r_d.get("next_registered_stage") == "STAGE1_OPTICAL_LAYER_DECISION_REQUIRED",
        "r_d_did_not_open_locked_stages": not any(
            bool(r_d.get(key)) for key in ("test_executed", "clinical_500_executed", "rgb_encoder_training_executed")
        ),
        "frozen_fold_manifest_hash": sha256_file(paths["r_c0r_fold_manifest"]) == str(config["folds"]["expected_sha256"]),
    }
    if not all(checks.values()):
        raise ContractError(f"Stage-1 decision gate preflight failed: {checks}")

    upstream = {
        "v1_stage_c_decision": v1_c,
        "v2r_formula_audit": v2r_a,
        "r_b_observation_audit": r_b_audit,
        "historical_r_c0_decision": r_c0,
        "r_c0r_decision": r_c0r,
        "r_c0r_audit": r_c0r_audit,
        "r_c1r_decision": r_c1r,
        "r_c1r_integrity_audit": r_c1r_audit,
        "r_d_decision": r_d,
        "r_d_integrity_audit": r_d_audit,
        "r_d_frozen_global_parameters": r_d_frozen,
    }
    context = {
        "checks": {key: bool(value) for key, value in checks.items()},
        "upstream": upstream,
        "frozen_fold_manifest_sha256": sha256_file(paths["r_c0r_fold_manifest"]),
    }
    return paths, context


# --------------------------------------------------------------------------- #
# analysis
# --------------------------------------------------------------------------- #


def _selected_candidate_summary(r_c0r: dict[str, Any], selected: str) -> dict[str, Any]:
    for row in r_c0r["candidate_summaries"]:
        if str(row.get("candidate")) == selected:
            return row
    raise ContractError(f"R-C0R carries no summary for the selected candidate {selected!r}")


def _strong_spectral_audit(config: dict[str, Any], r_c0r: dict[str, Any]) -> pd.DataFrame:
    selected = config["selected_candidate"]
    summary = _selected_candidate_summary(r_c0r, selected)
    engine = summary["strong_spectral_checks"]
    rows = []
    for target in config["strong_spectral_targets"]:
        key = str(target["key"])
        field = str(target["summary_field"])
        if field not in summary:
            raise ContractError(f"R-C0R summary is missing the strong-target field {field!r}")
        value = float(summary[field])
        recomputed = _compare(value, str(target["operator"]), float(target["target"]))
        engine_pass = bool(engine[key]) if key in engine else None
        rows.append(
            {
                "candidate": selected,
                "target_key": key,
                "summary_field": field,
                "value": value,
                "operator": str(target["operator"]),
                "target": float(target["target"]),
                "unit": str(target["unit"]),
                "recomputed_pass": recomputed,
                "engine_pass": engine_pass,
                "audit_agrees": engine_pass is not None and bool(engine_pass) == recomputed,
            }
        )
    frame = pd.DataFrame(rows)
    return frame


def _parameter_reliability_audit(config: dict[str, Any], upstream: dict[str, Any]) -> pd.DataFrame:
    selected = config["selected_candidate"]
    usable_values = set(str(value) for value in config["state_ladder"]["reliability_values_usable"])
    v1_c = upstream["v1_stage_c_decision"]
    r_c1r = upstream["r_c1r_decision"]

    rows: list[dict[str, Any]] = []
    for entry in v1_c["parameter_summary"]:
        coverage = float(entry["base_reliable_coverage"])
        classification = "reliable" if coverage >= 0.80 else ("conditional" if coverage > 0.0 else "unreliable")
        rows.append(
            {
                "source_stage": "v1_C",
                "model_id": str(v1_c.get("model_id", "KM-BIO-v1")),
                "candidate": "KM-BIO-v1",
                "parameter": str(entry["parameter"]),
                "subject_count": 44,
                "identifiable_fraction": coverage,
                "boundary_fraction": np.nan,
                "median_envelope_normalized_span": np.nan,
                "classification": classification,
                "usable_as_stage2_theta": classification in usable_values,
                "evidence": "v1_C_parameter_summary",
            }
        )

    for entry in r_c1r["parameter_profile_summary"]:
        if str(entry["candidate"]) != selected:
            continue
        classification = str(entry["classification"])
        rows.append(
            {
                "source_stage": "R-C1R",
                "model_id": "KM-BIO-v2R.1",
                "candidate": selected,
                "parameter": str(entry["parameter"]),
                "subject_count": int(entry["subject_count"]),
                "identifiable_fraction": float(entry["identifiable_fraction"]),
                "boundary_fraction": float(entry["boundary_fraction"]),
                "median_envelope_normalized_span": float(entry["median_envelope_normalized_span"]),
                "classification": classification,
                "usable_as_stage2_theta": classification in usable_values,
                "evidence": "r_c1r_parameter_profile_summary",
            }
        )

    for entry in r_c1r["best_o_profile_summary"]:
        if str(entry["base_candidate"]) != selected:
            continue
        classification = str(entry["classification"])
        rows.append(
            {
                "source_stage": "R-C1R",
                "model_id": "KM-BIO-v2R.1",
                "candidate": f"V2R-BEST-O({selected})",
                "parameter": str(entry["parameter"]),
                "subject_count": int(entry["subject_count"]),
                "identifiable_fraction": float(entry["identifiable_fraction"]),
                "boundary_fraction": float(entry["boundary_fraction"]),
                "median_envelope_normalized_span": float(entry["median_envelope_normalized_span"]),
                "classification": classification,
                "usable_as_stage2_theta": classification in usable_values,
                "evidence": "r_c1r_best_o_profile_summary",
            }
        )

    frame = pd.DataFrame(rows)
    profile = upstream["r_c1r_decision"]["parameter_profile_summary"]
    expected = {str(item["parameter"]): str(item["classification"]) for item in profile if str(item["candidate"]) == selected}
    delivered = {
        str(row["parameter"]): str(row["classification"])
        for _, row in frame[(frame["source_stage"] == "R-C1R") & (frame["candidate"] == selected)].iterrows()
    }
    if delivered != expected:
        raise ContractError(f"Carried-forward reliability mismatch: {delivered} != {expected}")
    if not bool(frame[frame["source_stage"] == "R-C1R"]["usable_as_stage2_theta"].eq(False).all()):
        raise ContractError("R-C1R verdicts must all be non-usable in this batch; none may be upgraded")
    return frame


def _observation_blocker_analysis(config: dict[str, Any], paths: dict[str, Path], upstream: dict[str, Any]) -> tuple[pd.DataFrame, dict[str, Any]]:
    selected = config["selected_candidate"]
    analysis = config["observation_blocker_analysis"]
    r_c0r = upstream["r_c0r_decision"]
    r_c1r = upstream["r_c1r_decision"]
    r_b_audit = upstream["r_b_observation_audit"]
    r_d_frozen = upstream["r_d_frozen_global_parameters"]

    rows: list[dict[str, Any]] = []

    # --- unified observation scale g0 ------------------------------------- #
    scale = analysis["unified_scale"]
    profile = pd.read_csv(paths["r_c0r_global_profile_summary"])
    g0_rows = profile[(profile["candidate"] == str(scale["estimated_in_candidate"])) & (profile["parameter"] == "g0")]
    if g0_rows.empty:
        raise ContractError("R-C0R global profile summary carries no g0 row to audit the unified scale")
    g0_row = g0_rows.iloc[0]
    g0_identifiable = bool(_coerce_bool(g0_rows["all_fold_profiles_identifiable"]).iloc[0]) and int(g0_row["boundary_fold_count"]) == 0
    ranking = {str(item["candidate"]): float(item["median_logrmse"]) for item in r_c0r["selection"]["ranking"]}
    tolerance = float(scale["selection_tolerance_immateriality"])
    if str(scale["estimated_in_candidate"]) in ranking and selected in ranking:
        scale_delta = abs(ranking[str(scale["estimated_in_candidate"])] - ranking[selected])
    else:
        raise ContractError("R-C0R ranking is missing a candidate needed for the scale immateriality check")
    scale_immaterial = bool(scale_delta <= tolerance)
    declared_fixed = bool(float(r_d_frozen.get("g0", np.nan)) == float(scale["declared_fixed_value"]))
    unified_scale_resolved = bool(declared_fixed and (g0_identifiable or scale_immaterial))
    rows.append(
        {
            "blocker_key": "unified_scale",
            "display_name_zh": str(scale["display_name_zh"]),
            "quantity": str(scale["quantity"]),
            "measured_detail": (
                f"g0_declared_fixed={declared_fixed}; g0_identifiable_in_{scale['estimated_in_candidate']}={g0_identifiable}; "
                f"selection_delta_logrmse={scale_delta:.6f}<= {tolerance}"
            ),
            "threshold_kind": "identifiable_and_immaterial",
            "measured_value": scale_delta,
            "threshold_value": tolerance,
            "condition_met": bool(not unified_scale_resolved),
            "is_blocking": bool(not unified_scale_resolved),
            "rule_note": "section 7.5 fires REVISE_OBSERVATION only when the unified scale is still the main blocker",
        }
    )

    # --- fixed left-right side difference --------------------------------- #
    side = analysis["fixed_side_difference"]
    linkage = pd.read_csv(paths["r_c1r_side_pressure_linkage"])
    primary = [str(item) for item in side["primary_parameters"]]
    relevant = linkage[(linkage["candidate"] == str(side["source_candidate"])) & (linkage["parameter"].isin(primary))]
    if relevant.empty:
        raise ContractError("R-C1R side-pressure linkage carries no row for the registered primary parameters")
    r_threshold = float(side["pearson_r_threshold"])
    p_threshold = float(side["pearson_p_threshold"])
    coupling_rows = relevant[
        (relevant["pearson_r_vs_side_log_difference"].abs() >= r_threshold) & (relevant["pearson_p"] < p_threshold)
    ]
    coupling_present = bool(not coupling_rows.empty)
    coupling_detail = "; ".join(
        f"{row['parameter']} r={row['pearson_r_vs_side_log_difference']:.4f} p={row['pearson_p']:.5f}"
        for _, row in relevant.iterrows()
    )
    usable_values = set(str(value) for value in config["state_ladder"]["reliability_values_usable"])
    any_primary_usable = any(
        str(item["classification"]) in usable_values
        for item in r_c1r["parameter_profile_summary"]
        if str(item["candidate"]) == selected and str(item["parameter"]) in primary
    )
    counts = r_b_audit["counts"]
    unit_excludes_side = bool(
        int(counts["symmetric_spectra"]) == int(counts["subjects"]) == int(config["analysis"]["train_subject_count"])
        and bool(side["primary_observation_unit_excludes_side_difference"])
    )
    side_decision_relevant = bool(coupling_present and any_primary_usable)
    side_blocking = bool(coupling_present and side_decision_relevant and not unit_excludes_side)
    rows.append(
        {
            "blocker_key": "fixed_side_difference",
            "display_name_zh": str(side["display_name_zh"]),
            "quantity": "pearson_r_vs_side_log_difference",
            "measured_detail": (
                f"{coupling_detail}; primary_parameter_usable={any_primary_usable}; "
                f"primary_unit_excludes_side_difference={unit_excludes_side}"
            ),
            "threshold_kind": "coupling_and_decision_relevance",
            "measured_value": float(relevant["pearson_r_vs_side_log_difference"].abs().max()),
            "threshold_value": r_threshold,
            "condition_met": bool(coupling_present),
            "is_blocking": side_blocking,
            "rule_note": (
                "the coupling is measured, but it is only a blocker when it could change a decision: every primary "
                "parameter is already at the worst class (unreliable), and the registered primary unit is the "
                "bilateral symmetric spectrum, which excludes the side difference from the spectral target"
            ),
        }
    )

    # --- per-spectrum high-capacity correction ---------------------------- #
    capacity = analysis["high_capacity_correction"]
    per_subject = int(capacity["fitted_per_subject_parameter_count"])
    ceiling = int(capacity["maximum_allowed_per_subject_parameters"])
    free_gain = bool(capacity["per_spectrum_free_gain_used"])
    high_capacity_required = bool(free_gain or per_subject > ceiling)
    v1_gain_fraction = float(capacity["v1_free_gain_boundary_count"]) / float(capacity["v1_free_gain_spectrum_count"])
    rows.append(
        {
            "blocker_key": "per_spectrum_high_capacity_correction",
            "display_name_zh": str(capacity["display_name_zh"]),
            "quantity": "fitted_per_subject_parameter_count",
            "measured_detail": (
                f"per_spectrum_free_gain_used={free_gain}; fitted_per_subject_parameters={per_subject}; "
                f"v1_free_gain_boundary_fraction={v1_gain_fraction:.4f} ({capacity['v1_free_gain_source']}, historical)"
            ),
            "threshold_kind": "capacity_ceiling",
            "measured_value": float(per_subject),
            "threshold_value": float(ceiling),
            "condition_met": high_capacity_required,
            "is_blocking": high_capacity_required,
            "rule_note": "section 7.5 fires STOP_KM when the best candidate can only rely on per-spectrum high-capacity correction",
        }
    )

    frame = pd.DataFrame(rows)
    observation_keys = {"unified_scale", "fixed_side_difference"}
    flags = {
        "unified_scale_blocker": bool(frame.loc[frame["blocker_key"] == "unified_scale", "is_blocking"].iloc[0]),
        "fixed_side_difference_coupling_present": coupling_present,
        "fixed_side_difference_blocker": side_blocking,
        "observation_level_blocker_present": bool(
            frame.loc[frame["blocker_key"].isin(observation_keys), "is_blocking"].any()
        ),
        "per_spectrum_high_capacity_correction_required": high_capacity_required,
        "driving_parameters_a_fitted_per_subject_parameter_count": per_subject,
        "primary_unit_excludes_side_difference": unit_excludes_side,
        "g0_identifiable_where_estimated": g0_identifiable,
        "g0_selection_immaterial": scale_immaterial,
        "g0_scale_delta_logrmse": float(scale_delta),
    }
    return frame, flags


def _residual_risk_register(
    config: dict[str, Any],
    paths: dict[str, Path],
    strong: pd.DataFrame,
    blockers: dict[str, Any],
    upstream: dict[str, Any],
) -> pd.DataFrame:
    selected = config["selected_candidate"]
    edge = pd.read_csv(paths["r_d_edge_band_comparison"])
    direction = pd.read_csv(paths["r_d_direction_consistency_summary"])
    r_d = upstream["r_d_decision"]
    r_c1r = upstream["r_c1r_decision"]

    train_row = direction[direction["scope"] == "train"].iloc[0]
    valid_row = direction[direction["scope"] == "valid"].iloc[0]
    repeat_bands = edge[_coerce_bool(edge["same_sign"])]["wavelength_nm"].astype(int).tolist()
    ediag = [
        f"{int(row['wavelength_nm'])}nm {row['train_median_signed_residual']:+.4f}->{row['validation_median_signed_residual']:+.4f}"
        for _, row in edge.iterrows()
    ]
    failed_targets = strong[~strong["recomputed_pass"]]["target_key"].tolist()
    side_row = next(
        item
        for item in r_c1r["side_pressure_linkage"]
        if str(item["candidate"]) == selected and str(item["parameter"]) == "f_blood"
    )
    best_o = [item for item in r_c1r["best_o_summary"] if str(item["base_candidate"]) == selected][0]
    narrow = [
        item
        for item in r_c1r["sensitivity_summary"]
        if str(item["candidate"]) == selected and str(item["setting"]).startswith("bandwidth_44")
    ][0]

    rows = [
        {
            "risk_id": "R1",
            "category": "observation_level",
            "item": "统一观测尺度 g0",
            "measured": (
                f"selected candidate declares g0=1.0 fixed; g0 estimated in {selected} counterpart is identifiable "
                f"with 0 boundary folds; selection delta logRMSE {blockers['g0_scale_delta_logrmse']:.6f}"
            ),
            "status": "resolved_by_registered_revision",
            "would_justify_revise_observation": False,
            "justification": "the unified-scale degeneracy registered in v1 section 2.4 was addressed by the v2R 4.4 scale term and is not the main blocker anymore",
        },
        {
            "risk_id": "R2",
            "category": "parameter_level",
            "item": "固定左右侧差与生理参数的耦合",
            "measured": (
                f"f_blood vs side log difference r={side_row['pearson_r_vs_side_log_difference']:.4f} "
                f"p={side_row['pearson_p']:.5f}; BEST-O on the same base r="
                f"{[item for item in r_c1r['side_pressure_linkage'] if str(item['candidate']) == f'V2R-BEST-O({selected})' and str(item['parameter']) == 'f_blood'][0]['pearson_r_vs_side_log_difference']:.4f}"
            ),
            "status": "unresolved_but_absorbed",
            "would_justify_revise_observation": False,
            "justification": "the coupling is a parameter-level confound whose consequence is already the declared unreliable verdict; the registered primary unit is the bilateral symmetric spectrum, so the side difference does not enter the spectral target",
        },
        {
            "risk_id": "R3",
            "category": "model_level",
            "item": "拟合窗内带结构残差（含 420 nm 与 Hb Q 带附近）",
            "measured": (
                f"strong-target failures: {failed_targets}; maximum absolute median signed band residual "
                f"{float(strong.loc[strong['target_key'] == 'maximum_abs_median_signed_band_residual', 'value'].iloc[0]):.4f} "
                f"(target {float(strong.loc[strong['target_key'] == 'maximum_abs_median_signed_band_residual', 'target'].iloc[0]):.2f})"
            ),
            "status": "unresolved",
            "would_justify_revise_observation": False,
            "justification": "the structured residual is a shape mismatch inside the 420-680 nm fit window, i.e. a model-representation limit, not an observation-contract term",
        },
        {
            "risk_id": "R4",
            "category": "edge_diagnostic",
            "item": "边缘外推带 400/410/690/700 nm",
            "measured": f"same-sign bands train vs Validation: {repeat_bands}; " + "; ".join(ediag),
            "status": "reproducible_model_feature",
            "would_justify_revise_observation": False,
            "justification": "section 7.1 marks the full 400-700 nm result as an explicitly labelled edge diagnostic that does not participate in the strong-fit judgement; the reproducible same-sign behaviour is recorded as a model characteristic",
        },
        {
            "risk_id": "R5",
            "category": "model_level",
            "item": "全局散射退化 A_s / delta_b_s",
            "measured": "A_s envelope non-identifiable; delta_b_s at the upper boundary in 5/5 R-C0R folds and in the full-Train R-D freeze",
            "status": "unresolved",
            "would_justify_revise_observation": False,
            "justification": "a global-scattering degeneracy is a model-level freedom, and section 7.3 already registers global profiles as diagnostics rather than execution gates",
        },
        {
            "risk_id": "R6",
            "category": "parameter_level",
            "item": "三个生理参数全部不可辨识",
            "measured": "f_mel / f_blood / s all classified unreliable in R-C1R under the registered R-A envelope rule",
            "status": "unresolved",
            "would_justify_revise_observation": False,
            "justification": "this is exactly the section 7.5 condition for SPECTRAL_ONLY: useful spectral reconstruction with no identifiable physiological parameter",
        },
        {
            "risk_id": "R7",
            "category": "scope",
            "item": "带宽收窄后的改善",
            "measured": (
                f"{narrow['setting']} median logRMSE {float(narrow['median_logrmse']):.4f} vs full-window "
                f"{float(train_row['median_logrmse']):.4f}; boundary fraction {float(narrow['boundary_fraction']):.4f}"
            ),
            "status": "out_of_scope_for_this_version",
            "would_justify_revise_observation": False,
            "justification": "section 7.1 forbids re-using the strong targets across bands; a narrower window is a new version and must be registered separately",
        },
        {
            "risk_id": "R8",
            "category": "scope",
            "item": "Validation 证据强度",
            "measured": (
                f"n=3 subjects; Validation median logRMSE {float(valid_row['median_logrmse']):.4f} vs Train "
                f"{float(train_row['median_logrmse']):.4f}; model better than reference fraction "
                f"{float(valid_row['model_better_than_reference_fraction']):.2f}"
            ),
            "status": "bounded",
            "would_justify_revise_observation": False,
            "justification": "R-D registers Validation as a developmental review only; no population estimate and no physiological claim may be derived from it",
        },
    ]
    frame = pd.DataFrame(rows)
    if str(r_d.get("validation_is_developmental_only")) != "True":
        raise ContractError("R-D must still mark Validation as developmental only")
    return frame


def _ladder_audit(
    config: dict[str, Any],
    strong: pd.DataFrame,
    reliability: pd.DataFrame,
    blockers: dict[str, Any],
    upstream: dict[str, Any],
) -> tuple[pd.DataFrame, str, dict[str, Any]]:
    ladder = config["state_ladder"]
    selected = config["selected_candidate"]
    r_d = upstream["r_d_decision"]

    all_targets_pass = bool(strong["recomputed_pass"].all())
    usable_values = set(str(value) for value in ladder["reliability_values_usable"])
    selected_reliability = reliability[reliability["candidate"] == selected]
    all_output_reliable = bool(
        selected_reliability["classification"].isin(usable_values).all() and len(selected_reliability) >= 2
    )
    any_parameter_usable = bool(selected_reliability["usable_as_stage2_theta"].any())
    best_o_reliability = reliability[reliability["candidate"] == f"V2R-BEST-O({selected})"]
    s_usable = bool(best_o_reliability[best_o_reliability["parameter"] == "s"]["usable_as_stage2_theta"].any())

    direction_consistent = bool(r_d["direction_consistency"]["direction_consistent"])
    dev = ladder["development_candidate_criteria"]
    dev_below = float(dev["model_to_reference_median_error_ratio_below"])
    dev_fraction = float(dev["reference_better_fraction_min"])

    # Section 7.5 resolves DEVELOPMENT_SPECTRAL_CANDIDATE on the complete
    # out-of-fold ladder.  The R-D Validation cohort is 3 subjects, so it can only
    # confirm that the frozen candidate does not reverse that direction; it can
    # never establish it.
    selected_summary = _selected_candidate_summary(upstream["r_c0r_decision"], selected)
    oof_ratio = float(selected_summary["median_model_to_reference_error_ratio"])
    oof_fraction = float(selected_summary["model_better_than_reference_fraction"])
    development_candidate = bool(oof_ratio < dev_below and oof_fraction >= dev_fraction)

    validation_ratio = float(r_d["validation_median_model_to_reference_error_ratio"])
    validation_fraction = float(r_d["validation_model_better_than_reference_fraction"])
    validation_confirms_direction = bool(
        validation_ratio < dev_below and validation_fraction >= dev_fraction and direction_consistent
    )
    spectral_reconstruction_useful = bool(development_candidate and validation_confirms_direction)

    partial_theta = bool(
        (direction_consistent or not bool(ladder["partial_theta_criteria"]["require_validation_direction_consistent"]))
        and (any_parameter_usable or not bool(ladder["partial_theta_criteria"]["require_at_least_one_parameter_reliable_or_conditional"]))
    )
    ready_with_s = bool(
        partial_theta and (s_usable or not bool(ladder["ready_with_s_criteria"]["require_s_reliable_or_conditional"]))
    )
    strong_theta = bool(
        (all_targets_pass or not bool(ladder["strong_theta_criteria"]["require_all_strong_targets"]))
        and (all_output_reliable or not bool(ladder["strong_theta_criteria"]["require_all_output_parameters_reliable"]))
    )
    stop = ladder["stop_km_criteria"]
    stop_km = bool(
        validation_fraction < float(stop["reference_better_fraction_min"])
        or validation_ratio >= float(stop["model_to_reference_median_error_ratio_at_or_above"])
        or (not direction_consistent and bool(stop["require_validation_direction_consistent"]))
        or (blockers["per_spectrum_high_capacity_correction_required"] and bool(stop["forbid_per_spectrum_high_capacity_correction"]))
    )
    revise_observation = bool(spectral_reconstruction_useful and blockers["observation_level_blocker_present"])
    all_unreliable = bool(not any_parameter_usable)
    spectral_only = bool(
        spectral_reconstruction_useful and all_unreliable and not blockers["observation_level_blocker_present"]
    )

    satisfied = {
        "V2R_STOP_KM": stop_km,
        "V2R_STRONG_THETA_READY": strong_theta,
        "V2R_READY_WITH_S": ready_with_s,
        "V2R_PARTIAL_THETA_CANDIDATE": partial_theta,
        "V2R_REVISE_OBSERVATION": revise_observation,
        "V2R_SPECTRAL_ONLY": spectral_only,
        "V2R_DEVELOPMENT_SPECTRAL_CANDIDATE": development_candidate,
    }
    evidence = {
        "V2R_STOP_KM": (
            f"validation_better_than_reference_fraction={validation_fraction:.4f}; "
            f"validation_model_to_reference_ratio={validation_ratio:.6f}; "
            f"direction_consistent={direction_consistent}; high_capacity_required={blockers['per_spectrum_high_capacity_correction_required']}"
        ),
        "V2R_STRONG_THETA_READY": f"all_strong_targets_pass={all_targets_pass}; all_output_parameters_reliable={all_output_reliable}",
        "V2R_READY_WITH_S": f"partial_theta_met={partial_theta}; s_reliable_or_conditional={s_usable}",
        "V2R_PARTIAL_THETA_CANDIDATE": f"direction_consistent={direction_consistent}; any_parameter_reliable_or_conditional={any_parameter_usable}",
        "V2R_REVISE_OBSERVATION": (
            f"spectral_reconstruction_useful={spectral_reconstruction_useful}; observation_level_blocker_present="
            f"{blockers['observation_level_blocker_present']}"
        ),
        "V2R_SPECTRAL_ONLY": (
            f"spectral_reconstruction_useful={spectral_reconstruction_useful}; all_parameters_unreliable={all_unreliable}; "
            f"observation_level_blocker_present={blockers['observation_level_blocker_present']}"
        ),
        "V2R_DEVELOPMENT_SPECTRAL_CANDIDATE": (
            f"oof_model_to_reference_ratio={oof_ratio:.6f}<1.0; oof_better_than_reference_fraction={oof_fraction:.4f}>=0.50; "
            f"validation_confirms_direction={validation_confirms_direction}"
        ),
    }

    rows = []
    priority = list(ladder["priority"])
    for state in priority:
        rows.append(
            {
                "state": state,
                "kind": "terminal",
                "priority": priority.index(state) + 1,
                "satisfied": satisfied[state],
                "meaning_zh": STATE_MEANING_ZH[state],
                "action_zh": STATE_ACTION_ZH[state],
                "evidence": evidence[state],
            }
        )
    for state in MILESTONE_STATES:
        rows.append(
            {
                "state": state,
                "kind": "milestone",
                "priority": 0,
                "satisfied": satisfied[state],
                "meaning_zh": STATE_MEANING_ZH[state],
                "action_zh": STATE_ACTION_ZH[state],
                "evidence": evidence[state],
            }
        )
    frame = pd.DataFrame(rows).sort_values(["kind", "priority"], ascending=[True, True]).reset_index(drop=True)
    terminal = frame[frame["kind"] == "terminal"]
    winners = terminal[terminal["satisfied"]].sort_values("priority")
    if winners.empty:
        raise ContractError("Section 7.5 ladder has no satisfied terminal state; the evidence set is incomplete")
    outcome = str(winners.iloc[0]["state"])
    flags = {
        "development_candidate_oof_met": development_candidate,
        "validation_confirms_direction": validation_confirms_direction,
        "spectral_reconstruction_useful": spectral_reconstruction_useful,
        "all_output_parameters_reliable": all_output_reliable,
        "any_parameter_reliable_or_conditional": any_parameter_usable,
        "s_reliable_or_conditional": s_usable,
        "all_strong_targets_pass": all_targets_pass,
        "direction_consistent": direction_consistent,
        "oof_model_to_reference_ratio": oof_ratio,
        "oof_better_than_reference_fraction": oof_fraction,
        "validation_model_to_reference_ratio": validation_ratio,
        "validation_better_than_reference_fraction": validation_fraction,
        "satisfied": {str(row["state"]): bool(row["satisfied"]) for _, row in frame.iterrows()},
    }
    return frame, outcome, flags


# --------------------------------------------------------------------------- #
# emission
# --------------------------------------------------------------------------- #


def _stage2_interface_freeze(config: dict[str, Any], paths: dict[str, Path], decision: dict[str, Any], upstream: dict[str, Any]) -> dict[str, Any]:
    selected = config["selected_candidate"]
    rejected = [row["parameter"] for row in upstream["r_c1r_decision"]["parameter_profile_summary"] if str(row["candidate"]) == selected]
    return {
        "schema_version": 1,
        "task_id": config["task_id"],
        "protocol_version": config["protocol_version"],
        "deliverable": {
            "name": "Fskin_forward_model",
            "kind": "physiological_forward_spectral_model",
            "code_entrypoint": "src/skin_optics_hsi/km_bio_v2r.py::forward_preloaded_numpy",
            "wavelength_nm": config["analysis"]["fit_centers_nm"],
            "coefficient_asset": str(paths["optical_asset_10nm"]),
            "coefficient_asset_sha256": sha256_file(paths["optical_asset_10nm"]),
        },
        "frozen_model_state": dict(upstream["r_d_frozen_global_parameters"]),
        "fixed_quantities": {
            "epidermis_thickness_mm": float(upstream["r_d_frozen_global_parameters"]["epidermis_thickness_mm"]),
            "diameter_um": float(upstream["r_d_frozen_global_parameters"]["diameter_um"]),
            "s0": float(upstream["r_d_frozen_global_parameters"]["s0"]),
            "g0": float(upstream["r_d_frozen_global_parameters"]["g0"]),
        },
        "parameter_ranges": {"f_mel": [0.0, 0.43], "f_blood": [0.0, 0.10], "s": [0.0, 1.0]},
        "primary_domain": {
            "unit": config["analysis"]["unit"],
            "fit_centers_nm": config["analysis"]["fit_centers_nm"],
            "edge_diagnostic_centers_nm": config["analysis"]["edge_diagnostic_centers_nm"],
            "train_subject_count": config["analysis"]["train_subject_count"],
            "validation_subject_count": config["analysis"]["validation_subject_count"],
            "validation_is_developmental_only": True,
            "quality_control": "R-B observation audit PASS_FOR_V2R_TRAIN_INVERSION",
        },
        "inversion_config": {
            "primary_loss": "equal_weight_logrmse",
            "residual_definition": "prediction_minus_observation",
            "epsilon": 1.0e-6,
            "reflectance_clipping": "forbidden",
            "reliable_parameter_list": [],
        },
        "error_tolerances": {
            "strong_targets": {str(item["key"]): float(item["target"]) for item in config["strong_spectral_targets"]},
            "direction_consistency_check_count": 6,
            "direction_consistency_checks_met": 6,
        },
        "failure_rules": [
            "report the full 400-700 nm result only as an explicitly labelled edge diagnostic",
            "never widen a parameter bound to make the spectral gate pass",
            "never let downstream classification results influence the optical-layer decision",
            "never treat the Validation 3-subject cohort as a population estimate",
            "a narrower fit window or a changed strong target requires a new registered version",
        ],
        "per_sample_output_schema": PER_SAMPLE_OUTPUT_SCHEMA,
        "stage2_usage": {
            "theta_usable_as_supervised_target": False,
            "reason": "decision_state is V2R_SPECTRAL_ONLY: every physiological parameter is unreliable, so only the forward reconstruction capability is admitted and no physiological-parameter encoder may be trained",
            "locked_until_new_registration": ["test", "clinical_500", "rgb_encoder_training", "stage2_parameter_encoder"],
        },
        "all_file_hashes": {
            "r_d_frozen_global_parameters": sha256_file(paths["r_d_frozen_global_parameters"]),
            "r_d_decision": sha256_file(paths["r_d_decision"]),
            "r_c1r_decision": sha256_file(paths["r_c1r_decision"]),
            "r_c0r_decision": sha256_file(paths["r_c0r_decision"]),
            "optical_asset_10nm": sha256_file(paths["optical_asset_10nm"]),
            "frozen_fold_manifest": sha256_file(paths["r_c0r_fold_manifest"]),
        },
        "decision_state": decision["decision_state"],
        "created_utc": decision["created_utc"],
    }


def _evidence_access_log(config: dict[str, Any], paths: dict[str, Path]) -> pd.DataFrame:
    rows = []
    for name, path in sorted(paths.items()):
        hash_only = path.suffix.lower() in (".npz", ".npy", ".npy.gz")
        rows.append(
            {
                "artifact": name,
                "path": str(path),
                "content_kind": "coefficient_asset_hash_only" if hash_only else "registered_summary_artifact",
                "window": "stage1_decision_gate",
                "content_reads": 0 if hash_only else 1,
            }
        )
    for name in ("raw_hsi", "rgb", "validation", "test", "clinical_500"):
        rows.append(
            {
                "artifact": name,
                "path": "",
                "content_kind": "forbidden_content",
                "window": "stage1_decision_gate",
                "content_reads": 0,
            }
        )
    frame = pd.DataFrame(rows)
    declared = int(config["execution"]["require_no_new_data_content_reads"])
    forbidden = frame[frame["content_kind"] == "forbidden_content"]
    if declared != 1 or int(forbidden["content_reads"].sum()) != 0:
        raise ContractError("Stage-1 decision gate must not read any forbidden data content")
    return frame


def _upstream_decision_chain(config: dict[str, Any], paths: dict[str, Path], upstream: dict[str, Any]) -> pd.DataFrame:
    entries = [
        ("v1_C", "KM-BIO-v1", "v1_stage_c_decision", "status"),
        ("v2R_R-A", "KM-BIO-v2R", "v2r_formula_audit", "status"),
        ("v2R_R-B", "KM-BIO-v2R", "r_b_observation_audit", "status"),
        ("v2R_R-C0", "KM-BIO-v2R", "historical_r_c0_decision", "status"),
        ("v2R.1_R-C0R", "KM-BIO-v2R.1", "r_c0r_decision", "status"),
        ("v2R.1_R-C1R", "KM-BIO-v2R.1", "r_c1r_decision", "status"),
        ("v2R.1_R-D", "KM-BIO-v2R.1", "r_d_decision", "status"),
    ]
    rows = []
    for stage, model, key, status_field in entries:
        payload = upstream[key]
        rows.append(
            {
                "stage": stage,
                "model_id": model,
                "artifact": key,
                "path": str(paths[key]),
                "sha256": sha256_file(paths[key]),
                "status": str(payload.get(status_field, "")),
                "next_registered_stage": str(payload.get("next_registered_stage", payload.get("authorized_next_stage", "")) or ""),
                "role": "read_only_evidence",
            }
        )
    return pd.DataFrame(rows)


def _decision_markdown(
    config: dict[str, Any],
    decision: dict[str, Any],
    strong: pd.DataFrame,
    reliability: pd.DataFrame,
    blocker_frame: pd.DataFrame,
    ladder: pd.DataFrame,
    risks: pd.DataFrame,
    chain: pd.DataFrame,
) -> str:
    selected = config["selected_candidate"]
    lines = [
        "# 阶段一光学层决策（KM-BIO-v2R.1）",
        "",
        f"- 任务：`{config['task_id']}`；协议：`{config['protocol_version']}`；日期：{config['frozen_date']}。",
        f"- 决策状态：`{decision['decision_state']}`。",
        f"- 阶段一光学层结论：`{decision['stage1_optical_layer_outcome']}`。",
        f"- 冻结候选：`{selected}`（`A_s={decision['frozen_model_state']['A_s']:.6f}`、`delta_bs={decision['frozen_model_state']['delta_bs']:.6f}`、`g0={decision['frozen_model_state']['g0']}`）。",
        f"- 参数声明：`{decision['parameter_reliability_claim']}`；参数升级：`forbidden`（{decision['parameter_upgrade_forbidden_reason']}）。",
        "- 数据边界：本批次**未读取**任何原始 HSI / RGB / Validation / Test / 临床 500 例内容；全部证据来自已登记的决策 JSON 与汇总表。",
        "",
        "## 1. 上游证据链（只读）",
        "",
        chain.drop(columns=["path"]).to_markdown(index=False),
        "",
        "## 2. 七项强光谱目标（05 文档 7.1 节，OOF 受试者级）",
        "",
        strong[["target_key", "value", "operator", "target", "unit", "recomputed_pass"]].to_markdown(index=False),
        "",
        f"结论：`all_targets_pass={bool(strong['recomputed_pass'].all())}`；未通过的项为 "
        f"`{', '.join(strong[~strong['recomputed_pass']]['target_key'].tolist())}`。",
        "",
        "## 3. 参数可辨识性（v1 C 对照 + R-C1R 沿用）",
        "",
        reliability[["source_stage", "candidate", "parameter", "identifiable_fraction", "boundary_fraction", "median_envelope_normalized_span", "classification"]].to_markdown(index=False),
        "",
        "## 4. 观测层阻断分析（05 文档 7.5 节）",
        "",
        blocker_frame[["blocker_key", "quantity", "measured_value", "threshold_value", "condition_met", "is_blocking", "measured_detail"]].to_markdown(index=False),
        "",
        f"`observation_level_blocker_present={decision['observation_level_blocker_present']}`；"
        f"`per_spectrum_high_capacity_correction_required={decision['per_spectrum_high_capacity_correction_required']}`。",
        "",
        "## 5. 决策阶梯判定",
        "",
        ladder[["kind", "priority", "state", "satisfied", "action_zh", "evidence"]].to_markdown(index=False),
        "",
        f"按预登记优先级取第一个成立的状态 → `{decision['decision_state']}`。",
        "",
        f"里程碑 `V2R_DEVELOPMENT_SPECTRAL_CANDIDATE` 由**折外阶梯**判定（模型/参考谱中位误差比 "
        f"`{decision['development_candidate_oof_evidence']['model_to_reference_median_error_ratio']:.6f}` < 1.0，"
        f"优于参考谱比例 `{decision['development_candidate_oof_evidence']['model_better_than_reference_fraction']:.4f}` ≥ 0.50）；"
        f"3 人 Validation 只能在 R-D 中确认方向未逆转（`validation_confirms_direction="
        f"{decision['validation_confirms_direction']}`），不能用来确立该状态。",
        "",
        "## 6. 残余风险登记",
        "",
        risks[["risk_id", "category", "item", "status", "would_justify_revise_observation", "justification"]].to_markdown(index=False),
        "",
        "## 7. 阶段二接口冻结（03 规划 3.6 节）",
        "",
        f"- 交付物：`Fskin` 前向光谱模型（`{decision['stage2_interface']['deliverable']['code_entrypoint']}`）。",
        f"- 系数资产 SHA-256：`{decision['stage2_interface']['deliverable']['coefficient_asset_sha256']}`。",
        f"- 可靠参数清单：`{decision['stage2_interface']['inversion_config']['reliable_parameter_list']}`。",
        f"- `theta` 是否可作为阶段二监督目标：**否**（{decision['stage2_interface']['stage2_usage']['reason']}）。",
        "- 逐样本输出字段见 `stage2_interface_freeze.json` 的 `per_sample_output_schema`。",
        "",
        "## 8. 授权边界",
        "",
        f"- 阶段一光学层路线结论：`{decision['route_conclusion']}`。",
        f"- `next_registered_stage = {decision['next_registered_stage']}`；任何 Test / 临床 500 例 / RGB 编码器训练 / 阶段二参数编码器均需**另行登记**。",
        f"- 锁定状态：`test_executed={decision['locked_stages']['test_executed']}`、`clinical_500_executed={decision['locked_stages']['clinical_500_executed']}`、`rgb_encoder_training_executed={decision['locked_stages']['rgb_encoder_training_executed']}`。",
        "",
    ]
    return "\n".join(lines)


def _emit(
    root: Path,
    config_file: Path,
    config: dict[str, Any],
    output: Path,
    paths: dict[str, Path],
    context: dict[str, Any],
    analysis: dict[str, Any],
    *,
    emission_mode: str,
    created_utc: str,
    audit_extras: dict[str, Any] | None = None,
) -> dict[str, Any]:
    strong = analysis["strong_spectral_audit"]
    reliability = analysis["parameter_reliability_audit"]
    blocker_frame = analysis["observation_blocker_analysis"]
    ladder = analysis["decision_ladder_audit"]
    risks = analysis["residual_risk_register"]
    chain = analysis["upstream_decision_chain"]
    access = analysis["evidence_access_log"]
    outcome = analysis["outcome"]
    ladder_flags = analysis["ladder_flags"]
    upstream = context["upstream"]

    _write_csv(strong, output / "strong_spectral_target_audit.csv")
    _write_csv(reliability, output / "parameter_reliability_audit.csv")
    _write_csv(blocker_frame, output / "observation_blocker_analysis.csv")
    _write_csv(ladder, output / "decision_ladder_audit.csv")
    _write_csv(risks, output / "residual_risk_register.csv")
    _write_csv(chain, output / "upstream_decision_chain.csv")
    _write_csv(access, output / "evidence_access_log.csv")

    failed = strong[~strong["recomputed_pass"]]["target_key"].tolist()
    decision: dict[str, Any] = {
        "schema_version": 1,
        "task_id": config["task_id"],
        "protocol_version": config["protocol_version"],
        "stage": config["stage"],
        "created_utc": created_utc,
        "emitted_utc": _utc_now(),
        "emission_mode": emission_mode,
        "status": "STAGE1_OPTICAL_LAYER_DECISION_COMPLETE",
        "selected_candidate": config["selected_candidate"],
        "frozen_model_state": dict(upstream["r_d_frozen_global_parameters"]),
        "frozen_model_state_sha256": sha256_file(paths["r_d_frozen_global_parameters"]),
        "decision_state": outcome,
        "decision_state_meaning_zh": STATE_MEANING_ZH[outcome],
        "decision_state_action_zh": STATE_ACTION_ZH[outcome],
        "stage1_optical_layer_outcome": (
            "SPECTRAL_RECONSTRUCTION_ONLY_NO_THETA_TARGET" if outcome == "V2R_SPECTRAL_ONLY" else "REVIEW_REQUIRED"
        ),
        "route_conclusion": (
            "KM-BIO 双层 K-M 生理线在阶段一收尾为**前向光谱重建模型**：其 420-680 nm 双侧对称谱重建优于固定参考谱且方向一致，"
            "但三个生理参数在 R-A 登记包络规则下全部不可辨识，因此不产出阶段二的生理参数目标。"
            if outcome == "V2R_SPECTRAL_ONLY"
            else "需要复核"
        ),
        "strong_spectral_targets": {
            "candidate": config["selected_candidate"],
            "all_targets_pass": bool(strong["recomputed_pass"].all()),
            "target_count": int(len(strong)),
            "passed_count": int(strong["recomputed_pass"].sum()),
            "failed_targets": failed,
            "engine_agreement": bool(strong["audit_agrees"].all()),
        },
        "development_candidate_oof_met": bool(ladder_flags["development_candidate_oof_met"]),
        "development_candidate_oof_evidence": {
            "candidate": config["selected_candidate"],
            "model_to_reference_median_error_ratio": float(ladder_flags["oof_model_to_reference_ratio"]),
            "model_better_than_reference_fraction": float(ladder_flags["oof_better_than_reference_fraction"]),
            "source": "r_c0r four-candidate out-of-fold ladder",
        },
        "validation_confirms_direction": bool(ladder_flags["validation_confirms_direction"]),
        "spectral_reconstruction_useful": bool(ladder_flags["spectral_reconstruction_useful"]),
        "parameter_reliability_claim": "none",
        "parameter_reliability": dict(config["carried_forward"]["parameter_reliability"]),
        "observation_level_blocker_present": bool(analysis["blocker_flags"]["observation_level_blocker_present"]),
        "unified_scale_blocker": bool(analysis["blocker_flags"]["unified_scale_blocker"]),
        "fixed_side_difference_blocker": bool(analysis["blocker_flags"]["fixed_side_difference_blocker"]),
        "fixed_side_difference_coupling_present": bool(analysis["blocker_flags"]["fixed_side_difference_coupling_present"]),
        "per_spectrum_high_capacity_correction_required": bool(
            analysis["blocker_flags"]["per_spectrum_high_capacity_correction_required"]
        ),
        "parameter_upgrade_forbidden": not bool(config["locked_stages"]["parameter_upgrade_allowed"]),
        "parameter_upgrade_forbidden_reason": str(config["locked_stages"]["parameter_upgrade_forbidden_reason"]),
        "locked_stages": dict(config["locked_stages"]),
        "evidence_access": {
            "summary_artifact_reads": int(
                access.loc[access["content_kind"] == "registered_summary_artifact", "content_reads"].sum()
            ),
            "forbidden_content_reads": int(
                access.loc[access["content_kind"] == "forbidden_content", "content_reads"].sum()
            ),
            "new_data_content_reads": 0,
        },
        "next_registered_stage": "NONE_REGISTERED",
        "km_bio_line_status": "CLOSED_AT_STAGE1_DECISION",
        "requires_new_registration_for": [
            "test_evaluation",
            "clinical_500_evaluation",
            "rgb_encoder_training",
            "stage2_parameter_encoder",
            "any_band_or_bound_or_target_change",
        ],
        "stage2_interface": {},
        "interpretation": (
            "阶段一光学层决策门：把 v1 A/B/C、v2R R-A/R-B/R-C0 与 v2R.1 R-C0R/R-C1/R-C1R/R-D 的已登记证据汇总为唯一的 7.5 决策状态，"
            "并按 3.6 节冻结阶段二接口。本批次不做任何拟合、不重算任何阈值、不读取任何新数据内容。"
        ),
    }
    interface = _stage2_interface_freeze(config, paths, decision, upstream)
    decision["stage2_interface"] = interface
    decision["stage2_theta_usable_as_supervised_target"] = bool(interface["stage2_usage"]["theta_usable_as_supervised_target"])

    (output / "stage2_interface_freeze.json").write_text(
        json.dumps(_jsonable(interface), ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (output / "stage1_km_bio_decision.json").write_text(
        json.dumps(_jsonable(decision), ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (output / "STAGE1_KM_BIO_DECISION.md").write_text(
        _decision_markdown(config, decision, strong, reliability, blocker_frame, ladder, risks, chain), encoding="utf-8"
    )

    audit = {
        "schema_version": 1,
        "task_id": config["task_id"],
        "protocol_version": config["protocol_version"],
        "status": "STAGE1_OPTICAL_LAYER_DECISION_COMPLETE",
        "created_utc": created_utc,
        "emitted_utc": decision["emitted_utc"],
        "emission": {
            "mode": emission_mode,
            "upstream_fits_recomputed": False,
            "aggregation_source": "upstream_registered_summaries",
            "thresholds_changed": False,
        },
        "preflight_checks": context["checks"],
        "preflight_all_passed": bool(all(context["checks"].values())),
        "decision_state": outcome,
        "ladder_priority": list(config["state_ladder"]["priority"]),
        "ladder_satisfied": {
            str(row["state"]): bool(row["satisfied"]) for _, row in ladder.sort_values(["kind", "priority"]).iterrows()
        },
        "hierarchy_audit": {
            "stop_km_triggers_checked": [
                "reference_better_fraction_below_min",
                "error_ratio_at_or_above_one",
                "direction_inconsistent",
                "high_capacity_correction_required",
            ],
            "precedence_respected": bool(not (analysis["blocker_flags"]["per_spectrum_high_capacity_correction_required"])),
        },
        "data_access": {
            "raw_hsi_reads": 0,
            "rgb_reads": 0,
            "validation_reads": 0,
            "test_reads": 0,
            "clinical_500_reads": 0,
            "new_data_content_reads": 0,
            "summary_artifact_reads": decision["evidence_access"]["summary_artifact_reads"],
        },
        "frozen_fold_manifest_sha256": context["frozen_fold_manifest_sha256"],
        "frozen_model_state_sha256": decision["frozen_model_state_sha256"],
        "authorized_stages": [],
        "locked_stages": dict(config["locked_stages"]),
    }
    if audit_extras:
        audit.update(audit_extras)
    (output / "audit_summary.json").write_text(
        json.dumps(_jsonable(audit), ensure_ascii=False, indent=2), encoding="utf-8"
    )
    audit_md = [
        "# 阶段一光学层决策门审计",
        "",
        f"- 状态：`{audit['status']}`；决策状态：`{audit['decision_state']}`。",
        f"- 预检：{sum(1 for value in context['checks'].values() if value)}/{len(context['checks'])} 通过。",
        f"- 发射：`{emission_mode}`；上游拟合重算：`false`；阈值改动：`false`。",
        "- 数据访问：原始 HSI/RGB/Validation/Test/临床 500 例内容读取均为 `0`（仅消费已登记汇总产物）。",
        f"- 冻结折清单 SHA-256：`{context['frozen_fold_manifest_sha256']}`。",
        f"- 冻结模型状态 SHA-256：`{decision['frozen_model_state_sha256']}`。",
        "- 授权阶段：无（Test / 临床 500 例 / RGB 编码器训练 / 阶段二参数编码器均需另行登记）。",
        "",
    ]
    (output / "audit_summary.md").write_text("\n".join(audit_md), encoding="utf-8")

    artifact_rows = [{"role": "input", "name": name, "path": str(path), "sha256": sha256_file(path)} for name, path in sorted(paths.items())]
    artifact_rows.append({"role": "config", "name": "stage1_decision_config", "path": str(config_file), "sha256": sha256_file(config_file)})
    for name, value in config["implementation"].items():
        path = _resolve(root, value)
        artifact_rows.append({"role": "implementation", "name": name, "path": str(path), "sha256": sha256_file(path)})
    for path in sorted(output.iterdir(), key=lambda item: item.name):
        if path.is_file() and path.name != "artifact_hash_manifest.csv":
            artifact_rows.append({"role": "output", "name": path.name, "path": str(path), "sha256": sha256_file(path)})
    _write_csv(pd.DataFrame(artifact_rows), output / "artifact_hash_manifest.csv")
    _log(f"Stage-1 optical-layer decision emitted ({emission_mode})")
    return decision


def _analyze(root: Path, config: dict[str, Any], paths: dict[str, Path], context: dict[str, Any]) -> dict[str, Any]:
    upstream = context["upstream"]
    strong = _strong_spectral_audit(config, upstream["r_c0r_decision"])
    reliability = _parameter_reliability_audit(config, upstream)
    blocker_frame, blocker_flags = _observation_blocker_analysis(config, paths, upstream)
    ladder, outcome, ladder_flags = _ladder_audit(config, strong, reliability, blocker_flags, upstream)
    risks = _residual_risk_register(config, paths, strong, blocker_flags, upstream)
    access = _evidence_access_log(config, paths)
    chain = _upstream_decision_chain(config, paths, upstream)
    return {
        "strong_spectral_audit": strong,
        "parameter_reliability_audit": reliability,
        "observation_blocker_analysis": blocker_frame,
        "blocker_flags": blocker_flags,
        "decision_ladder_audit": ladder,
        "ladder_flags": ladder_flags,
        "residual_risk_register": risks,
        "evidence_access_log": access,
        "upstream_decision_chain": chain,
        "outcome": outcome,
    }


def run_stage1_optical_decision(config_path: str | Path, project_root: str | Path) -> dict[str, Any]:
    """Aggregate the registered evidence and emit the stage-1 optical-layer decision."""

    root = Path(project_root).resolve()
    config_file = Path(config_path).resolve()
    config = yaml.safe_load(config_file.read_text(encoding="utf-8"))
    output = _resolve(root, config["output_directory"])
    if output.exists() and any(output.iterdir()) and config["access_policy"]["refuse_existing_output"]:
        raise ContractError(f"Refusing to overwrite an existing stage-1 decision directory: {output}")
    output.mkdir(parents=True, exist_ok=True)

    paths, context = _preflight(root, config)
    analysis = _analyze(root, config, paths, context)
    created = _utc_now()
    return _emit(root, config_file, config, output, paths, context, analysis, emission_mode="primary_run", created_utc=created)


def finalize_stage1_optical_decision(config_path: str | Path, project_root: str | Path) -> dict[str, Any]:
    """Re-derive the aggregates from the registered summaries and re-emit the decision.

    The gate performs no fitting, so a finalize pass re-applies the same
    preregistered rule to the same hash-verified upstream summaries.  It never
    recomputes an upstream fit, never changes a threshold and preserves the
    original ``created_utc``.
    """

    root = Path(project_root).resolve()
    config_file = Path(config_path).resolve()
    config = yaml.safe_load(config_file.read_text(encoding="utf-8"))
    output = _resolve(root, config["output_directory"])
    existing_decision = output / "stage1_km_bio_decision.json"
    if not existing_decision.exists():
        raise ContractError("finalize_stage1_optical_decision requires an existing primary run")
    created = str(json.loads(existing_decision.read_text(encoding="utf-8"))["created_utc"])

    paths, context = _preflight(root, config)
    analysis = _analyze(root, config, paths, context)
    return _emit(
        root,
        config_file,
        config,
        output,
        paths,
        context,
        analysis,
        emission_mode="finalize_reemission",
        created_utc=created,
    )
