from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def write_failure_reports(root: Path, preflight: dict[str, Any], reproduction: dict[str, Any] | None, reason: str) -> dict[str, Any]:
    reports = root / "reports"
    logs = root / "logs"
    reports.mkdir(parents=True, exist_ok=True)
    logs.mkdir(parents=True, exist_ok=True)
    summary = {
        "experiment_name": "Stage3_A0_FrozenRGB_Paired_ExposureGamma_Stress_v1",
        "status": "failed_before_counterfactual_stress",
        "failure_reason": reason,
        "preflight": preflight,
        "reproduction": reproduction,
        "counterfactual_inference_executed": False,
        "training_executed": False,
        "checkpoint_modified": False,
        "stage3_a1_executed": False,
    }
    (reports / "stage3_a0_machine_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    (reports / "stage3_a0_evidence_classification.json").write_text(
        json.dumps({
            "counterfactual_luminance_sensitivity_level": "indeterminate",
            "within_camera_evidence_level": "indeterminate",
            "brightness_shortcut_support": "indeterminate",
            "reason": "Original OOF reproduction gate failed; counterfactual stress test was not run.",
        }, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (reports / "stage3_a0_to_a1_decision.md").write_text(
        "# Stage3-A0 to A1 Decision\n\nDo not proceed to Stage3-A1 from this run because the original OOF reproduction gate failed.\n",
        encoding="utf-8",
    )
    (reports / "stage3_a0_report.md").write_text(
        "# Stage3-A0 Frozen RGB Paired Exposure/Gamma Stress\n\n"
        "The paired counterfactual luminance stress test stopped before image brightness manipulation because the frozen original OOF reproduction gate failed.\n\n"
        f"- failure reason: {reason}\n"
        f"- preflight pass: {preflight.get('pass')}\n"
        f"- reproduction pass: {None if reproduction is None else reproduction.get('pass')}\n"
        f"- max probability absolute difference: {None if reproduction is None else reproduction.get('max_probability_abs_diff')}\n"
        f"- prediction match count: {None if reproduction is None else reproduction.get('prediction_match_count')}\n\n"
        "No model training, checkpoint modification, threshold optimization, counterfactual inference, or Stage3-A1 execution was performed.\n",
        encoding="utf-8",
    )
    inventory = {str(p.relative_to(root)): p.stat().st_size for p in root.rglob("*") if p.is_file()}
    (reports / "stage3_a0_output_inventory.json").write_text(json.dumps(inventory, ensure_ascii=False, indent=2), encoding="utf-8")
    (logs / "run.log").write_text(f"Stage3-A0 stopped before counterfactual stress: {reason}\n", encoding="utf-8")
    (logs / "warnings.csv").write_text("warning\n", encoding="utf-8")
    (logs / "failures.csv").write_text("failure\n" + reason.replace("\n", " ") + "\n", encoding="utf-8")
    return summary


def write_success_placeholder(root: Path, preflight: dict[str, Any], reproduction: dict[str, Any]) -> dict[str, Any]:
    raise NotImplementedError("Counterfactual stress execution is intentionally unavailable until the OOF reproduction gate passes.")
