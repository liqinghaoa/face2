import json
from pathlib import Path

import numpy as np
import pandas as pd

from utils.p1_cluster_bootstrap import compute_visit_metrics

ROOT = Path(__file__).resolve().parents[2]
EXP = ROOT / "experiments/500Data/P1_RGB_P0Aligned_ResNet18_5fold_v1"

EXPECTED_VISIT_METRICS = {
    "macro_auc": 0.8278260869565217,
    "accuracy": 0.79,
    "macro_precision": 0.7012498802796667,
    "macro_recall": 0.6898362507058159,
    "macro_f1": 0.6950493439204457,
    "balanced_accuracy": 0.6898362507058159,
    "pr_auc": 0.9441728720715097,
    "patient_sensitivity": 0.8753246753246753,
    "control_specificity": 0.5043478260869565,
    "ppv": 0.8553299492385786,
    "npv": 0.5471698113207547,
}
EXPECTED_CM = [[58, 57], [48, 337]]


def test_visit_oof_complete_and_metrics_match_expected_case_results():
    case = pd.read_csv(EXP / "oof/oof_predictions_case.csv", dtype={"case_id": str, "patient_group_id": str})
    assert len(case) == case.case_id.nunique() == 500
    assert not case.case_id.isna().any()
    assert not case.case_id.duplicated().any()
    probs = case[["prob_control", "prob_patient"]].to_numpy(dtype=float)
    assert np.isfinite(probs).all()
    assert np.allclose(probs.sum(axis=1), 1.0, atol=1e-5)
    assert np.array_equal(probs.argmax(axis=1), case.pred_binary.to_numpy(dtype=int))

    metrics = compute_visit_metrics(case)
    visit_json = json.loads((EXP / "oof/oof_metrics_visit.json").read_text())
    for key, expected in EXPECTED_VISIT_METRICS.items():
        assert np.isclose(metrics[key], expected)
        assert np.isclose(visit_json[key], expected)
    assert np.asarray(metrics["confusion_matrix"]).tolist() == EXPECTED_CM
    assert visit_json["confusion_matrix"] == EXPECTED_CM


def test_longitudinal_audit_and_evaluation_protocol_correction_outputs():
    audit = pd.read_csv(EXP / "metadata/patient_group_longitudinal_label_audit.csv")
    summary = json.loads((EXP / "metadata/patient_group_longitudinal_label_summary.json").read_text())
    assert len(audit) == summary["unique_patient_group_count"] == 483
    assert summary["multi_case_group_count"] == 17
    assert summary["groups_with_nyha_change"] == 15
    assert summary["groups_with_three_class_change"] == 10
    assert summary["groups_with_binary_label_change"] == 0
    assert summary["cases_in_binary_conflict_groups"] == 0

    correction = json.loads((EXP / "metadata/evaluation_protocol_correction.json").read_text())
    assert correction["correction_version"] == "P1_EVALUATION_PROTOCOL_V1_1"
    assert correction["training_changed"] is False
    assert correction["checkpoints_changed"] is False
    assert correction["oof_probabilities_changed"] is False
    assert correction["folds_changed"] is False
    assert correction["labels_changed"] is False
    assert correction["case_metrics_recomputed"] is True
    assert correction["group_metrics_deprecated"] is True
    assert correction["cluster_bootstrap_recomputed"] is True
    assert correction["primary_evaluation_level"] == "visit_case"


def test_fold_visit_metrics_and_cluster_bootstrap_outputs_are_primary():
    folds = pd.read_csv(EXP / "summary/fold_metrics_visit.csv")
    assert len(folds) == 5
    assert set(
        [
            "fold",
            "n_visits",
            "n_patient_groups",
            "control_count",
            "patient_count",
            "best_epoch",
            "macro_auc",
            "accuracy",
            "macro_precision",
            "macro_recall",
            "macro_f1",
            "balanced_accuracy",
            "patient_sensitivity",
            "control_specificity",
        ]
    ).issubset(folds.columns)
    assert (folds.n_visits == 100).all()
    assert (folds.control_count == 23).all()
    assert (folds.patient_count == 77).all()

    bootstrap = json.loads((EXP / "summary/p1_rgb_visit_metrics_cluster_bootstrap.json").read_text())
    assert bootstrap["cluster_unit"] == "patient_group_id"
    assert bootstrap["metric_unit"] == "visit_case"
    assert bootstrap["unique_clusters"] == 483
    assert bootstrap["visit_count"] == 500
    assert bootstrap["valid_iterations"] >= 1900
    assert np.isclose(bootstrap["point_estimates"]["macro_auc"], EXPECTED_VISIT_METRICS["macro_auc"])
    assert {
        "macro_auc",
        "accuracy",
        "macro_f1",
        "balanced_accuracy",
        "patient_sensitivity",
        "control_specificity",
    }.issubset(bootstrap["ci95"])

    paired = json.loads((EXP / "summary/paired_patient_cluster_visit_bootstrap.json").read_text())
    assert paired["cluster_unit"] == "patient_group_id"
    assert paired["metric_unit"] == "visit_case"
    assert paired["unique_clusters"] == 483
    assert paired["visit_count"] == 500
    assert "delta_roc_auc" in paired["ci95"]
    assert "delta_accuracy" in paired["point_estimates"]["delta"]


def test_group_artifacts_are_deprecated_not_formal_primary_outputs():
    deprecated = EXP / "deprecated_group_evaluation"
    payload = json.loads((deprecated / "DEPRECATED.json").read_text())
    assert payload["invalid_assumption"] == "All visits within one patient_group_id share one binary label."
    assert payload["replacement_protocol"] == (
        "Visit/case-level evaluation with patient_group_id used only for fold grouping and clustered resampling."
    )
    for name in [
        "oof_predictions_group.csv",
        "oof_metrics_group.json",
        "oof_confusion_matrix_group.csv",
        "fold_metrics_group.csv",
        "paired_cluster_bootstrap.json",
    ]:
        assert (deprecated / name).is_file()

    run_manifest = json.loads((EXP / "run_manifest.json").read_text())
    assert run_manifest["status"] == "P1_RGB_BASELINE_READY_WITH_CORRECTED_EVALUATION_PROTOCOL"
    assert run_manifest["group_level_metrics_deprecated"] is True
    assert run_manifest["visit_level_metrics_primary"] is True
