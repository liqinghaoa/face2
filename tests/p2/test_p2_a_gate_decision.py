from __future__ import annotations

from p2_counterfactual.gate_decision import decide_p2_a2_gate


def _metrics(auc: float, f1: float, ba: float, sens: float, spec: float):
    return {"macro_auc": auc, "macro_f1": f1, "balanced_accuracy": ba, "patient_sensitivity": sens, "control_specificity": spec}


def _stability(std: float, flip: float, worst: float, cosine: float = 0.9):
    return {"mean_prediction_std": std, "case_level_flip_rate": flip, "worst_light_auc": worst, "mean_feature_cosine": cosine}


def test_gate_passes_when_stability_improves_and_performance_kept() -> None:
    result = decide_p2_a2_gate(
        model_metrics={
            "p2_a1_colorjitter": _metrics(0.80, 0.70, 0.72, 0.85, 0.60),
            "p2_a2_relighting": _metrics(0.795, 0.69, 0.71, 0.84, 0.59),
        },
        stability_metrics={
            "p2_a1_colorjitter": _stability(0.10, 0.20, 0.70),
            "p2_a2_relighting": _stability(0.08, 0.15, 0.72),
        },
    )
    assert result["gate_pass"] is True


def test_gate_fails_on_performance_collapse_or_no_signal() -> None:
    collapse = decide_p2_a2_gate(
        model_metrics={
            "p2_a1_colorjitter": _metrics(0.80, 0.70, 0.72, 0.85, 0.60),
            "p2_a2_relighting": _metrics(0.75, 0.69, 0.71, 0.84, 0.59),
        },
        stability_metrics={
            "p2_a1_colorjitter": _stability(0.10, 0.20, 0.70),
            "p2_a2_relighting": _stability(0.08, 0.15, 0.72),
        },
    )
    assert collapse["gate_pass"] is False
    no_signal = decide_p2_a2_gate(
        model_metrics={
            "p2_a1_colorjitter": _metrics(0.80, 0.70, 0.72, 0.85, 0.60),
            "p2_a2_relighting": _metrics(0.80, 0.70, 0.72, 0.85, 0.60),
        },
        stability_metrics={
            "p2_a1_colorjitter": _stability(0.10, 0.20, 0.70),
            "p2_a2_relighting": _stability(0.10, 0.20, 0.70),
        },
    )
    assert no_signal["gate_pass"] is False
