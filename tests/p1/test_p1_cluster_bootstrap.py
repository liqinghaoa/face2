from __future__ import annotations

import pandas as pd

from utils.p1_cluster_bootstrap import paired_patient_cluster_visit_bootstrap, patient_cluster_bootstrap


def test_patient_cluster_bootstrap_uses_patient_group_resampling():
    frame = pd.DataFrame(
        [
            {"case_id": "c1", "patient_group_id": "g1", "label_binary": 0, "prob_control": 0.9, "prob_patient": 0.1},
            {"case_id": "c2", "patient_group_id": "g1", "label_binary": 1, "prob_control": 0.2, "prob_patient": 0.8},
            {"case_id": "c3", "patient_group_id": "g2", "label_binary": 0, "prob_control": 0.8, "prob_patient": 0.2},
            {"case_id": "c4", "patient_group_id": "g2", "label_binary": 1, "prob_control": 0.1, "prob_patient": 0.9},
        ]
    )
    result = patient_cluster_bootstrap(frame, iterations=20, seed=2026)
    assert result["cluster_unit"] == "patient_group_id"
    assert result["visit_count"] == 4
    assert result["valid_iterations"] > 0


def test_paired_bootstrap_accepts_pred_binary_baseline():
    component = pd.DataFrame(
        [
            {"case_id": "c1", "patient_group_id": "g1", "label_binary": 0, "prob_control": 0.9, "prob_patient": 0.1},
            {"case_id": "c2", "patient_group_id": "g1", "label_binary": 1, "prob_control": 0.2, "prob_patient": 0.8},
            {"case_id": "c3", "patient_group_id": "g2", "label_binary": 0, "prob_control": 0.8, "prob_patient": 0.2},
            {"case_id": "c4", "patient_group_id": "g2", "label_binary": 1, "prob_control": 0.1, "prob_patient": 0.9},
        ]
    )
    baseline = pd.DataFrame(
        [
            {"case_id": "c1", "patient_group_id": "g1", "label_binary": 0, "prob_patient": 0.2, "pred_binary": 0},
            {"case_id": "c2", "patient_group_id": "g1", "label_binary": 1, "prob_patient": 0.7, "pred_binary": 1},
            {"case_id": "c3", "patient_group_id": "g2", "label_binary": 0, "prob_patient": 0.3, "pred_binary": 0},
            {"case_id": "c4", "patient_group_id": "g2", "label_binary": 1, "prob_patient": 0.6, "pred_binary": 1},
        ]
    )
    result = paired_patient_cluster_visit_bootstrap(component, baseline, iterations=20, seed=2026)
    assert result["cluster_unit"] == "patient_group_id"
    assert "delta" in result["point_estimates"]

