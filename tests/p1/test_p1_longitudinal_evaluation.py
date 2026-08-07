from pathlib import Path

import pandas as pd

from utils.p1_cluster_bootstrap import (
    cluster_bootstrap_visit_metrics,
    paired_patient_cluster_visit_bootstrap,
)
from utils.p1_longitudinal_label_audit import audit_longitudinal_labels_from_frame


def _synthetic_longitudinal_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "case_id": ["c1", "c2", "c3", "c4"],
            "patient_group_id": ["g1", "g1", "g2", "g3"],
            "fold": [0, 0, 1, 1],
            "label_original": [0, 3, 2, 2],
            "label_3class": [0, 2, 1, 1],
            "label_binary": [0, 1, 1, 1],
            "prob_control": [0.9, 0.2, 0.3, 0.4],
            "prob_patient": [0.1, 0.8, 0.7, 0.6],
            "pred_binary": [0, 1, 1, 1],
        }
    )


def test_same_patient_group_can_change_labels_without_hard_failure(tmp_path: Path):
    frame = _synthetic_longitudinal_frame()
    audit, summary = audit_longitudinal_labels_from_frame(frame, tmp_path)
    assert len(audit) == 3
    assert summary["unique_patient_group_count"] == 3
    assert summary["multi_case_group_count"] == 1
    assert summary["groups_with_nyha_change"] == 1
    assert summary["groups_with_three_class_change"] == 1
    assert summary["groups_with_binary_label_change"] == 1
    assert summary["cases_in_binary_conflict_groups"] == 2
    assert (tmp_path / "patient_group_longitudinal_label_audit.csv").is_file()
    assert (tmp_path / "patient_group_longitudinal_label_summary.json").is_file()


def test_cluster_bootstrap_uses_patient_group_resampling_and_skips_single_class_iterations():
    frame = pd.DataFrame(
        {
            "case_id": ["a1", "a2", "b1", "b2"],
            "patient_group_id": ["a", "a", "b", "b"],
            "fold": [0, 0, 1, 1],
            "label_binary": [0, 0, 1, 1],
            "prob_control": [0.9, 0.8, 0.2, 0.1],
            "prob_patient": [0.1, 0.2, 0.8, 0.9],
            "pred_binary": [0, 0, 1, 1],
        }
    )
    result = cluster_bootstrap_visit_metrics(frame, iterations=20, seed=7)
    assert result["cluster_unit"] == "patient_group_id"
    assert result["metric_unit"] == "visit_case"
    assert result["unique_clusters"] == 2
    assert result["visit_count"] == 4
    assert result["failed_iterations"] > 0
    assert result["valid_iterations"] < 20
    assert set(result["ci95"]).issubset(
        {
            "macro_auc",
            "accuracy",
            "macro_f1",
            "balanced_accuracy",
            "patient_sensitivity",
            "control_specificity",
        }
    )


def test_paired_cluster_bootstrap_keeps_visit_level_metrics_and_cluster_sampling():
    frame = pd.DataFrame(
        {
            "case_id": ["a1", "a2", "b1", "b2"],
            "patient_group_id": ["a", "a", "b", "b"],
            "fold": [0, 0, 1, 1],
            "label_binary": [0, 0, 1, 1],
            "prob_control": [0.9, 0.8, 0.2, 0.1],
            "prob_patient": [0.1, 0.2, 0.8, 0.9],
            "pred_binary": [0, 0, 1, 1],
        }
    )
    historical = pd.DataFrame(
        {
            "case_id": ["a1", "a2", "b1", "b2"],
            "patient_group_id": ["a", "a", "b", "b"],
            "binary_label": [0, 0, 1, 1],
            "prob_patient": [0.2, 0.3, 0.7, 0.8],
            "pred_class": [0, 0, 1, 1],
        }
    )
    result = paired_patient_cluster_visit_bootstrap(frame, historical, iterations=20, seed=11)
    assert result["cluster_unit"] == "patient_group_id"
    assert result["metric_unit"] == "visit_case"
    assert result["point_estimates"]["delta"]
    assert result["failed_iterations"] >= 0
    assert "delta_roc_auc" in result["ci95"] or result["valid_iterations"] == 0
