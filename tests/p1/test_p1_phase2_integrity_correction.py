from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts/p1"))

from audit_and_correct_p1_phase2_summary import EXPERIMENTS, _compute_fold_metrics


EXP_ROOT = ROOT / "experiments/500Data/P1_Component_Sweep_v1"
SUMMARY = EXP_ROOT / "summary"
REPORT = ROOT / "reports/p1_component_sweep_phase2_results.md"
INTEGRITY_REPORT = ROOT / "reports/p1_phase2_result_integrity_correction_report.md"
CORRECTION = EXP_ROOT / "metadata/p1_phase2_summary_integrity_correction.json"


def _fold_prediction(experiment: str, fold: int) -> pd.DataFrame:
    return pd.read_csv(EXP_ROOT / experiment / f"fold_{fold}" / "val_predictions_visit.csv")


def _oof(experiment: str) -> pd.DataFrame:
    return pd.concat([_fold_prediction(experiment, fold) for fold in range(5)], ignore_index=True)


def test_all_completed_experiments_have_500_oof_and_five_100_case_folds() -> None:
    main = pd.read_csv(SUMMARY / "p1_component_main_results.csv")
    completed = main[main["status"] == "COMPLETED"]
    assert set(completed["experiment"]) == set(EXPERIMENTS)
    for experiment in EXPERIMENTS:
        fold_sizes = [len(_fold_prediction(experiment, fold)) for fold in range(5)]
        oof = _oof(experiment)
        assert fold_sizes == [100, 100, 100, 100, 100]
        assert len(oof) == 500
        assert oof["case_id"].nunique() == 500
        assert int(completed.loc[completed["experiment"] == experiment, "n_oof"].iloc[0]) == 500


def test_oof_rows_are_real_prediction_rows_not_predicted_control_count() -> None:
    main = pd.read_csv(SUMMARY / "p1_component_main_results.csv")
    fold = pd.read_csv(SUMMARY / "p1_component_fold_results.csv")
    report = REPORT.read_text(encoding="utf-8")
    for experiment in EXPERIMENTS:
        subset = fold[fold["experiment"] == experiment]
        predicted_control = int(subset["predicted_control_count"].sum())
        predicted_patient = int(subset["predicted_patient_count"].sum())
        n_oof = int(main.loc[main["experiment"] == experiment, "n_oof"].iloc[0])
        assert n_oof == 500
        assert n_oof == predicted_control + predicted_patient
        assert n_oof != predicted_control
        assert f"| {experiment} | 5 | 500 | {predicted_control} | {predicted_patient} |" in report


def test_prediction_probabilities_and_argmax_are_valid_without_group_oof() -> None:
    forbidden = {"oof_predictions_group.csv", "metrics_group.json", "confusion_matrix_group.csv"}
    active_forbidden = [
        path
        for path in EXP_ROOT.rglob("*")
        if path.is_file() and path.name in forbidden and "deprecated" not in [part.lower() for part in path.parts]
    ]
    assert active_forbidden == []
    for experiment in EXPERIMENTS:
        oof = _oof(experiment)
        probs = oof[["prob_control", "prob_patient"]].astype(float).to_numpy()
        assert np.isfinite(probs).all()
        assert ((0 <= probs) & (probs <= 1)).all()
        assert np.allclose(probs.sum(axis=1), 1.0, atol=1e-6)
        assert np.array_equal(oof["pred_binary"].astype(int).to_numpy(), probs.argmax(axis=1))


def test_p1_l_fold_sensitivity_specificity_and_confusion_match_oof() -> None:
    fold = pd.read_csv(SUMMARY / "p1_component_fold_results.csv")
    confusion = pd.read_csv(SUMMARY / "p1_component_fold_confusion_matrices.csv")
    main = pd.read_csv(SUMMARY / "p1_component_main_results.csv")
    p1_l = fold[fold["experiment"] == "p1_l"].sort_values("fold")
    assert len(p1_l) == 5
    assert p1_l["patient_sensitivity"].notna().all()
    assert p1_l["control_specificity"].notna().all()
    p1_l_confusion = confusion[confusion["experiment"] == "p1_l"].sort_values("fold")
    assert p1_l_confusion[["tn", "fp", "fn", "tp"]].sum().to_dict() == {"tn": 89, "fp": 26, "fn": 122, "tp": 263}
    metrics = json.loads((EXP_ROOT / "p1_l/oof/oof_metrics_visit.json").read_text(encoding="utf-8"))
    assert metrics["confusion_matrix"] == [[89, 26], [122, 263]]
    main_row = main[main["experiment"] == "p1_l"].iloc[0]
    assert np.isclose(float(main_row["patient_sensitivity"]), 263 / 385)
    assert np.isclose(float(main_row["control_specificity"]), 89 / 115)


def test_all_30_fold_metrics_recompute_to_stored_values() -> None:
    fold = pd.read_csv(SUMMARY / "p1_component_fold_results.csv")
    audit = pd.read_csv(SUMMARY / "p1_phase2_fold_metric_recalculation_audit.csv")
    assert len(audit) == 300
    assert set(audit["status"]) == {"MATCH"}
    for experiment in EXPERIMENTS:
        for fold_id in range(5):
            predictions = _fold_prediction(experiment, fold_id)
            metrics = _compute_fold_metrics(predictions)
            stored = fold[(fold["experiment"] == experiment) & (fold["fold"] == fold_id)].iloc[0]
            for metric in [
                "macro_auc",
                "accuracy",
                "macro_precision",
                "macro_recall",
                "macro_f1",
                "balanced_accuracy",
                "patient_sensitivity",
                "control_specificity",
                "predicted_control_count",
                "predicted_patient_count",
            ]:
                assert np.isclose(float(stored[metric]), float(metrics[metric]), atol=1e-10)


def test_fold_distribution_mean_sd_and_pooled_oof_auc_are_distinct_definitions() -> None:
    fold = pd.read_csv(SUMMARY / "p1_component_fold_results.csv")
    main = pd.read_csv(SUMMARY / "p1_component_main_results.csv")
    dist = pd.read_csv(SUMMARY / "p1_component_fold_distribution_summary.csv")
    assert set(dist["experiment"]) == set(EXPERIMENTS)
    for experiment in EXPERIMENTS:
        subset = fold[fold["experiment"] == experiment]
        row = dist[dist["experiment"] == experiment].iloc[0]
        main_row = main[main["experiment"] == experiment].iloc[0]
        assert np.isclose(row["fold_macro_auc_mean"], subset["macro_auc"].mean())
        assert np.isclose(row["fold_macro_auc_std"], subset["macro_auc"].std(ddof=1))
        assert np.isclose(row["pooled_oof_macro_auc"], main_row["macro_auc"])
        assert np.isclose(row["pooled_minus_fold_mean_auc"], main_row["macro_auc"] - subset["macro_auc"].mean())


def test_pairing_status_reports_and_guardrail_metadata() -> None:
    paired = pd.read_csv(SUMMARY / "p1_component_paired_comparisons.csv")
    status = pd.read_csv(SUMMARY / "p1_component_status.csv")
    correction = json.loads(CORRECTION.read_text(encoding="utf-8"))
    report = REPORT.read_text(encoding="utf-8")
    integrity_report = INTEGRITY_REPORT.read_text(encoding="utf-8")
    assert len(paired) == 6
    assert (paired["pair_count"] == 500).all()
    assert (paired["matched_rows"] == 500).all()
    assert (paired[["label_mismatch", "fold_mismatch", "patient_group_mismatch"]] == 0).all().all()
    assert set(status["result_integrity_status"]) == {"VERIFIED"}
    assert set(status["summary_correction_version"]) == {"P1_PHASE2_SUMMARY_CORRECTION_V1"}
    assert correction["training_rerun"] is False
    assert correction["checkpoints_modified"] is False
    assert correction["oof_probabilities_modified"] is False
    assert correction["fold_predictions_modified"] is False
    assert correction["stage_a_started"] is False
    assert correction["stage_b_started"] is False
    assert correction["stage_c_started"] is False
    assert "Fold Macro-AUC mean ± SD" in report
    assert "Pooled OOF Macro-AUC" in report
    assert "P1_PHASE2_RESULTS_INTEGRITY_VERIFIED" in integrity_report


def test_integrity_script_does_not_call_training_optimizer_or_stage_abc() -> None:
    source = (ROOT / "scripts/p1/audit_and_correct_p1_phase2_summary.py").read_text(encoding="utf-8")
    forbidden_tokens = [
        "P1ComponentTrainer",
        ".fit_fold(",
        "optimizer.step",
        ".backward(",
        "run_stage_a",
        "run_stage_b",
        "run_stage_c",
    ]
    for token in forbidden_tokens:
        assert token not in source
