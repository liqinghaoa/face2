"""Reclassify existing S1-5R Train evidence for representation-layer feasibility.

This script reads only the already-audited Train-only S1-5R outputs.  It does
not reopen raw HSI, Validation, or Test data and it does not overwrite the
historical S1-5R decision.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _resolve(path: str | Path) -> Path:
    p = Path(path)
    return p if p.is_absolute() else PROJECT_ROOT / p


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
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


def _new_output(path: Path) -> None:
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite existing output: {path}")
    path.mkdir(parents=True, exist_ok=False)


def _finite(value: Any) -> bool:
    try:
        return bool(pd.notna(value)) and float(value) == float(value) and abs(float(value)) < float("inf")
    except (TypeError, ValueError):
        return False


def _candidate_summary(evidence: dict[str, Any], model_id: str, expected_n: int, expected_subjects: int) -> dict[str, Any]:
    summary = evidence.get("model_summary", {}).get(model_id)
    if not isinstance(summary, dict):
        return {"model_id": model_id, "available": False, "pass": False, "reason": "missing_model_summary"}
    required = ("n", "subjects", "median_shape_log_rmse", "median_raw_log_rmse", "median_raw_sam", "boundary_fraction")
    finite = all(_finite(summary.get(key)) for key in required)
    shape = float(summary["median_shape_log_rmse"]) if _finite(summary.get("median_shape_log_rmse")) else None
    boundary = float(summary["boundary_fraction"]) if _finite(summary.get("boundary_fraction")) else None
    passed = bool(
        int(summary.get("n", -1)) == expected_n
        and int(summary.get("subjects", -1)) == expected_subjects
        and finite
    )
    return {
        "model_id": model_id,
        "available": True,
        "n": int(summary.get("n", -1)),
        "subjects": int(summary.get("subjects", -1)),
        "median_shape_log_rmse": shape,
        "median_raw_log_rmse": float(summary["median_raw_log_rmse"]) if _finite(summary.get("median_raw_log_rmse")) else None,
        "median_raw_sam": float(summary["median_raw_sam"]) if _finite(summary.get("median_raw_sam")) else None,
        "boundary_fraction": boundary,
        "pass": passed,
        "reason": "ok" if passed else "count_or_finite_metric_failure",
    }


def run(config_path: str | Path, output_dir: str | Path) -> dict[str, Any]:
    config_file = _resolve(config_path)
    output = _resolve(output_dir)
    _new_output(output)
    config = yaml.safe_load(config_file.read_text(encoding="utf-8"))
    if config.get("stage") != "S1-5R-REPRESENTATION":
        raise ValueError("Unexpected representation reclassification stage")

    inputs = {key: _resolve(value) for key, value in config["inputs"].items()}
    prior = _read_json(inputs["prior_decision"])
    evidence = _read_json(inputs["train_evidence"])
    failures = _read_json(inputs["fit_failures"])
    fits = pd.read_parquet(inputs["per_spectrum_fits"])

    if prior.get("validation_rows_read") != 0 or prior.get("test_rows_read") != 0 or prior.get("raw_hsi_files_read") != 0:
        raise RuntimeError("Historical S1-5R output violates the representation reclassification access policy")
    expected_n = int(config["data_roles"]["expected_primary_spectra"])
    expected_subjects = int(config["data_roles"]["expected_subjects"])
    primary_ids = list(config["representation_candidates"]["primary"])
    comparator_ids = list(config["representation_candidates"]["comparator"])
    all_ids = primary_ids + comparator_ids + list(config["representation_candidates"]["diagnostic_rejected"])
    if set(fits.model_id.unique()) != set(all_ids):
        raise ValueError(f"Unexpected model set in historical Train fits: {sorted(fits.model_id.unique())}")
    counts = fits.groupby("model_id").size().to_dict()
    if any(int(counts.get(model_id, 0)) != expected_n for model_id in all_ids):
        raise ValueError(f"Historical Train fit counts are incomplete: {counts}")
    if int(failures.get("count", -1)) != 0:
        raise ValueError("Representation reclassification requires zero historical fit failures")

    summaries = {model_id: _candidate_summary(evidence, model_id, expected_n, expected_subjects) for model_id in all_ids}
    gates = config["representation_gates"]
    d2 = summaries["D2-MH"]
    b0s = summaries["B0-S"]
    d3 = summaries["D3-MHG"]
    b2 = summaries["B2-PCA"]
    d2_vs_b0s_fraction = float(evidence["D2_better_than_B0S_fraction"])
    d2_to_b0s_ratio = float(evidence["D2_to_B0S_median_shape_log_rmse_ratio"])
    gain_retention = float(evidence["median_physical_gain_retention"])
    perturbation_shift = float(evidence["perturbation_median_normalized_shift_max"])
    representation_checks = {
        "train_fit_complete": all(summaries[m]["pass"] for m in all_ids),
        "d2_better_than_b0s": d2_vs_b0s_fraction >= float(gates["d2_better_than_b0s_fraction_min"]),
        "d2_b0s_ratio": d2_to_b0s_ratio <= float(gates["d2_to_b0s_median_shape_log_rmse_ratio_max"]),
        "d2_gain_retention": gain_retention >= float(gates["d2_physical_gain_retention_min"]),
        "d2_perturbation": perturbation_shift <= float(gates["d2_perturbation_median_normalized_shift_max"]),
        "primary_boundary": max(float(summaries[m]["boundary_fraction"]) for m in primary_ids) <= float(gates["max_primary_boundary_fraction"]),
        "d3_has_incremental_train_gain": float(d3["median_shape_log_rmse"]) < float(d2["median_shape_log_rmse"]),
        "b2_comparator_available": bool(b2["pass"]),
        "semantic_mh_gate_not_used": gates["semantic_mh_gate"] is False,
    }
    ready = bool(all(representation_checks.values()))
    candidate_assessment = {
        **summaries,
        "D2-MH": {**d2, "role": "primary_validation_candidate"},
        "D3-MHG": {**d3, "role": "higher_dimensional_validation_candidate"},
        "D1-M": {**summaries["D1-M"], "role": "lower_dimensional_comparator"},
        "B0-S": {**b0s, "role": "nuisance_only_baseline"},
        "B2-PCA": {**b2, "role": "data_driven_comparator"},
        "K2-MH-KM": {**summaries["K2-MH-KM"], "role": "diagnostic_rejected_semi_mechanistic"},
    }
    representation_evidence = {
        "stage": "S1-5R-REPRESENTATION",
        "source_decision": str(inputs["prior_decision"]),
        "source_decision_sha256": _sha256(inputs["prior_decision"]),
        "access_audit": {
            "train_rows_read": expected_n,
            "validation_rows_read": 0,
            "test_rows_read": 0,
            "raw_hsi_files_read": 0,
        },
        "candidate_assessment": candidate_assessment,
        "metrics_reused_from_train_evidence": {
            "D2_better_than_B0S_fraction": d2_vs_b0s_fraction,
            "D2_to_B0S_median_shape_log_rmse_ratio": d2_to_b0s_ratio,
            "median_physical_gain_retention": gain_retention,
            "perturbation_median_normalized_shift_max": perturbation_shift,
        },
        "representation_checks": representation_checks,
        "interpretation_layer": "deferred_to_stage2",
        "note": "M/H semantic stability, nuisance-invariance of individual coordinates, and physiological identity are not S1 representation gates.",
    }
    evidence_path = output / "representation_evidence.json"
    evidence_path.write_text(json.dumps(representation_evidence, ensure_ascii=False, indent=2), encoding="utf-8")
    decision = {
        "schema_version": 1,
        "stage": "S1-5R-REPRESENTATION",
        "status": "REPRESENTATION_READY_FOR_S1_6" if ready else "REPRESENTATION_REVISE",
        "next_stage_allowed": ready,
        "authorized_next_stage": "S1-6" if ready else None,
        "selected_representation_model": None,
        "validation_candidate_models": primary_ids + comparator_ids,
        "interpretation_layer_status": "DEFERRED_TO_S2",
        "physiological_M_H_claim": "NOT_ASSESSED_IN_S1",
        "validation_rows_read": 0,
        "test_rows_read": 0,
        "raw_hsi_files_read": 0,
        "representation_checks": representation_checks,
        "source_decision": str(inputs["prior_decision"]),
        "source_decision_sha256": _sha256(inputs["prior_decision"]),
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    decision_path = output / "s1_5r_representation_decision.json"
    decision_path.write_text(json.dumps(decision, ensure_ascii=False, indent=2), encoding="utf-8")
    report = f"""# S1-5R Representation-layer reclassification

Status: `{decision['status']}`  
Next stage allowed: `{str(ready).lower()}`  
Interpretation layer: `DEFERRED_TO_S2`

This version reuses only the historical Train-only S1-5R v7 evidence. It does not read Validation, Test, or raw HSI content and does not overwrite the historical decision.

- Train primary spectra: {expected_n}
- Train subjects: {expected_subjects}
- Candidate representations for S1-6: {', '.join(primary_ids + comparator_ids)}
- D2/B0-S median shape RMSE ratio: {d2_to_b0s_ratio:.6f}
- D3 median shape log-RMSE: {float(d3['median_shape_log_rmse']):.6f}
- B2-PCA median shape log-RMSE: {float(b2['median_shape_log_rmse']):.6f}
- M/H semantic gate used: `false`

The result establishes Train-level representation feasibility only. It does not select the final dimension/model and does not establish M/H physiological meaning. Final selection remains an S1-6 Validation task.
"""
    report_path = output / "S1_5R_REPRESENTATION_REPORT.md"
    report_path.write_text(report, encoding="utf-8")
    inputs_audit = {
        "inputs": {key: {"path": str(path), "sha256": _sha256(path)} for key, path in inputs.items()},
        "outputs": {"representation_evidence.json": _sha256(evidence_path),
                     "s1_5r_representation_decision.json": _sha256(decision_path),
                     "S1_5R_REPRESENTATION_REPORT.md": _sha256(report_path)},
        "validation_rows_read": 0,
        "test_rows_read": 0,
        "raw_hsi_files_read": 0,
    }
    (output / "input_audit.json").write_text(json.dumps(inputs_audit, ensure_ascii=False, indent=2), encoding="utf-8")
    return decision


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    print(json.dumps(run(args.config, args.output_dir), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
