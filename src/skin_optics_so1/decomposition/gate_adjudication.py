"""SO-1C engineering learnability gate adjudication from frozen R1/R2 evidence."""

from __future__ import annotations

from typing import Any


def adjudicate_learnability_gate_v2(
    *,
    r2_best_record: dict[str, Any],
    r2_best_summary: dict[str, Any],
    overfit1_result: dict[str, Any],
    paired_overfit2_result: dict[str, Any],
    tests_passed: bool,
) -> dict[str, Any]:
    """Apply Gate v2 without altering the historical strict R2 decision."""
    channels = r2_best_summary["channels"]
    values = {
        "loss_reduction": float(r2_best_record["loss_reduction"]),
        "M_MAE": float(channels["M"]["MAE"]),
        "H_MAE": float(channels["H"]["MAE"]),
        "S_MAE": float(channels["S"]["MAE"]),
        "P_pred_std": float(channels["P"]["pred"]["std"]),
        "P_active_recall": float(channels["P"]["active_region_recall"]),
    }
    checks = {
        "loss_reduction_ge_0_95": values["loss_reduction"] >= 0.95,
        "M_MAE_le_0_03": values["M_MAE"] <= 0.03,
        "H_MAE_le_0_03": values["H_MAE"] <= 0.03,
        "S_MAE_le_0_03": values["S_MAE"] <= 0.03,
        "P_pred_std_gt_0": values["P_pred_std"] > 0.0,
        "P_active_recall_gt_0": values["P_active_recall"] > 0.0,
        "all_channel_mae_finite": all(
            __import__("math").isfinite(float(channels[name]["MAE"])) for name in ("M", "H", "S", "P")
        ),
        "overfit1_pass": bool(overfit1_result["best_summary"]["channels"]["H"]["MAE"] <= 0.01),
        "paired_overfit2_pass": bool(paired_overfit2_result["best_summary"]["channels"]["H"]["MAE"] <= 0.015),
        "tests_pass": bool(tests_passed),
    }
    passed = all(checks.values())
    return {
        "historical_R2_strict_gate": "FAIL",
        "historical_R2_H_MAE": values["H_MAE"],
        "learnability_gate_v2": "PASS" if passed else "FAIL",
        "gate_adjudication_decision": "ALLOW_SMOKE" if passed else "BLOCK_SMOKE",
        "values": values,
        "checks": checks,
    }
