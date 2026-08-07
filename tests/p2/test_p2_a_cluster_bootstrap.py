from __future__ import annotations

import pandas as pd

from p2_counterfactual.cluster_bootstrap import P2AExperimentFrames, paired_cluster_bootstrap_comparison


def _frames(exp: str, offset: float) -> P2AExperimentFrames:
    original = pd.DataFrame(
        [
            {"case_id": "a1", "patient_group_id": "g1", "label": 0, "prob_control": 0.8 + offset, "prob_patient": 0.2 - offset},
            {"case_id": "a2", "patient_group_id": "g1", "label": 0, "prob_control": 0.7 + offset, "prob_patient": 0.3 - offset},
            {"case_id": "b1", "patient_group_id": "g2", "label": 1, "prob_control": 0.3 - offset, "prob_patient": 0.7 + offset},
            {"case_id": "b2", "patient_group_id": "g2", "label": 1, "prob_control": 0.2 - offset, "prob_patient": 0.8 + offset},
        ]
    )
    stability = original[["case_id", "patient_group_id", "label"]].copy()
    stability["prediction_std"] = 0.1 - offset
    stability["case_flip"] = 0
    stability["mean_feature_cosine"] = 0.8 + offset
    for preset in ["neutral_front", "left", "right", "top", "dim_front", "bright_front"]:
        stability[f"prob_{preset}"] = original["prob_patient"]
    return P2AExperimentFrames(exp, original, stability)


def test_paired_cluster_bootstrap_is_reproducible_and_finite() -> None:
    rows_a = paired_cluster_bootstrap_comparison(_frames("a", 0.0), _frames("b", 0.05), iterations=20, seed=2026)
    rows_b = paired_cluster_bootstrap_comparison(_frames("a", 0.0), _frames("b", 0.05), iterations=20, seed=2026)
    assert rows_a == rows_b
    assert len(rows_a) > 0
    assert all(row["valid_iterations"] > 0 for row in rows_a)
