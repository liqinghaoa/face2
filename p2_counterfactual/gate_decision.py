from __future__ import annotations

from typing import Any


DEFAULT_GATE_CONFIG = {
    "comparison": {"candidate": "p2_a2_relighting", "reference": "p2_a1_colorjitter"},
    "performance_tolerance": {
        "macro_auc_max_drop": 0.01,
        "macro_f1_max_drop": 0.02,
        "balanced_accuracy_max_drop": 0.02,
        "sensitivity_max_drop": 0.03,
        "specificity_max_drop": 0.03,
    },
    "positive_signals": {"any_classification_metric_improved": True, "any_stability_metric_improved": True},
    "stability_metrics": {
        "prediction_std_lower_is_better": True,
        "case_flip_rate_lower_is_better": True,
        "worst_light_auc_higher_is_better": True,
    },
    "require_no_performance_collapse": True,
}


def decide_p2_a2_gate(
    *,
    model_metrics: dict[str, dict[str, float]],
    stability_metrics: dict[str, dict[str, float]],
    bootstrap_rows: list[dict[str, Any]] | None = None,
    gate_config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    cfg = dict(DEFAULT_GATE_CONFIG)
    if gate_config:
        cfg.update(gate_config)
    candidate = cfg["comparison"]["candidate"]
    reference = cfg["comparison"]["reference"]
    cand_m = model_metrics[candidate]
    ref_m = model_metrics[reference]
    cand_s = stability_metrics[candidate]
    ref_s = stability_metrics[reference]
    deltas = {
        "macro_auc": float(cand_m["macro_auc"] - ref_m["macro_auc"]),
        "macro_f1": float(cand_m["macro_f1"] - ref_m["macro_f1"]),
        "balanced_accuracy": float(cand_m["balanced_accuracy"] - ref_m["balanced_accuracy"]),
        "patient_sensitivity": float(cand_m["patient_sensitivity"] - ref_m["patient_sensitivity"]),
        "control_specificity": float(cand_m["control_specificity"] - ref_m["control_specificity"]),
        "mean_prediction_std": float(cand_s["mean_prediction_std"] - ref_s["mean_prediction_std"]),
        "case_level_flip_rate": float(cand_s["case_level_flip_rate"] - ref_s["case_level_flip_rate"]),
        "worst_light_auc": float(cand_s["worst_light_auc"] - ref_s["worst_light_auc"]),
        "mean_feature_cosine": float(cand_s["mean_feature_cosine"] - ref_s["mean_feature_cosine"]),
    }
    tol = cfg["performance_tolerance"]
    checks = {
        "macro_auc": deltas["macro_auc"] >= -float(tol["macro_auc_max_drop"]),
        "macro_f1": deltas["macro_f1"] >= -float(tol["macro_f1_max_drop"]),
        "balanced_accuracy": deltas["balanced_accuracy"] >= -float(tol["balanced_accuracy_max_drop"]),
        "patient_sensitivity": deltas["patient_sensitivity"] >= -float(tol["sensitivity_max_drop"]),
        "control_specificity": deltas["control_specificity"] >= -float(tol["specificity_max_drop"]),
    }
    no_collapse = all(checks.values())
    positive_signals = {
        "macro_auc_improved": deltas["macro_auc"] > 0,
        "macro_f1_improved": deltas["macro_f1"] > 0,
        "balanced_accuracy_improved": deltas["balanced_accuracy"] > 0,
        "mean_prediction_std_lower": deltas["mean_prediction_std"] < 0,
        "case_level_flip_rate_lower": deltas["case_level_flip_rate"] < 0,
        "worst_light_auc_improved": deltas["worst_light_auc"] > 0,
    }
    any_positive = any(positive_signals.values())
    gate_pass = bool(no_collapse and any_positive)
    ci = {}
    for row in bootstrap_rows or []:
        if row.get("model_a") == reference and row.get("model_b") == candidate:
            ci[row["metric"]] = [row.get("ci_lower"), row.get("ci_upper")]
    return {
        "candidate": candidate,
        "reference": reference,
        "metric_values": {"candidate": {**cand_m, **cand_s}, "reference": {**ref_m, **ref_s}},
        "metric_deltas": deltas,
        "confidence_intervals": ci,
        "tolerances": tol,
        "no_performance_collapse_checks": checks,
        "no_performance_collapse": no_collapse,
        "positive_signals": positive_signals,
        "gate_pass": gate_pass,
        "a3_should_run": gate_pass,
        "decision_reason": "A2 preserved required performance tolerance and has at least one positive signal." if gate_pass else "A2 failed performance tolerance or has no positive classification/stability signal.",
    }


def decide_a3_increment(
    *,
    model_metrics: dict[str, dict[str, float]],
    stability_metrics: dict[str, dict[str, float]],
) -> dict[str, Any]:
    if "p2_a3_full_consistency" not in model_metrics:
        return {"status": "not_available", "recommended_for_p2_b": False, "reason": "A3 was not run."}
    a2 = "p2_a2_relighting"
    a3 = "p2_a3_full_consistency"
    delta_auc = model_metrics[a3]["macro_auc"] - model_metrics[a2]["macro_auc"]
    delta_f1 = model_metrics[a3]["macro_f1"] - model_metrics[a2]["macro_f1"]
    delta_ba = model_metrics[a3]["balanced_accuracy"] - model_metrics[a2]["balanced_accuracy"]
    delta_std = stability_metrics[a3]["mean_prediction_std"] - stability_metrics[a2]["mean_prediction_std"]
    delta_flip = stability_metrics[a3]["case_level_flip_rate"] - stability_metrics[a2]["case_level_flip_rate"]
    delta_worst = stability_metrics[a3]["worst_light_auc"] - stability_metrics[a2]["worst_light_auc"]
    classification_improved = bool(delta_auc > 0 or delta_f1 > 0 or delta_ba > 0)
    stability_improved = bool(delta_std < 0 or delta_flip < 0 or delta_worst > 0)
    performance_collapse = bool(delta_auc < -0.01 or delta_f1 < -0.02 or delta_ba < -0.02)
    recommended = bool((classification_improved or stability_improved) and not performance_collapse)
    return {
        "status": "available",
        "classification_improved": classification_improved,
        "stability_improved": stability_improved,
        "performance_collapse": performance_collapse,
        "recommended_for_p2_b": recommended,
        "deltas": {
            "macro_auc": float(delta_auc),
            "macro_f1": float(delta_f1),
            "balanced_accuracy": float(delta_ba),
            "mean_prediction_std": float(delta_std),
            "case_level_flip_rate": float(delta_flip),
            "worst_light_auc": float(delta_worst),
        },
        "reason": "A3 shows classification or stability increment without configured collapse." if recommended else "A3 does not show a safe incremental signal over A2.",
    }
