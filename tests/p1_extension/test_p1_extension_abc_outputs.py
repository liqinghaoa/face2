from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
ABC_ROOT = ROOT / "experiments" / "500Data" / "P1_Extension_ABC_v1"
REPORTS = ROOT / "reports"


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def test_stage_success_markers_and_stage_d_not_started() -> None:
    assert _read_json(ABC_ROOT / "stage_a" / "_STAGE_SUCCESS.json")["status"] == "STAGE_A_COMPLEMENTARITY_ANALYSIS_COMPLETE"
    assert _read_json(ABC_ROOT / "stage_b" / "_STAGE_SUCCESS.json")["status"] == "STAGE_B_FUSION_EXPERIMENTS_COMPLETE"
    assert _read_json(ABC_ROOT / "stage_c" / "_STAGE_SUCCESS.json")["status"] == "STAGE_C_ACQUISITION_AUDIT_COMPLETE"
    manifest = _read_json(ABC_ROOT / "metadata" / "abc_run_manifest.json")
    assert manifest["final_status"] == "P1_EXTENSION_ABC_COMPLETE"
    assert manifest["stage_d_started"] is False


def test_stage_b_has_three_smokes_fifteen_folds_and_500_case_oof() -> None:
    stage_b = ABC_ROOT / "stage_b"
    experiments = ("rgb_rgb_capacity_control", "rgb_s_independent_dual", "rgb_r_independent_dual")
    for experiment in experiments:
        assert (stage_b / "smoke" / experiment / "_SMOKE_SUCCESS.json").is_file()
        for fold in range(5):
            assert (stage_b / experiment / f"fold_{fold}" / "_FOLD_SUCCESS.json").is_file()
        oof = pd.read_csv(stage_b / experiment / "oof" / "oof_predictions_visit.csv", dtype={"case_id": str})
        assert len(oof) == 500
        assert oof["case_id"].nunique() == 500
        assert not (stage_b / experiment / "oof" / "oof_predictions_group.csv").exists()
    fold_success = list(stage_b.glob("*/fold_*/_FOLD_SUCCESS.json"))
    formal_fold_success = [path for path in fold_success if "\\smoke\\" not in str(path)]
    assert len(formal_fold_success) == 15


def test_stage_b_summary_contains_pooled_and_fold_mean_sd_metrics() -> None:
    main = pd.read_csv(ABC_ROOT / "stage_b" / "summary" / "stage_b_main_results.csv")
    required = {
        "macro_auc",
        "macro_f1",
        "balanced_accuracy",
        "fold_macro_auc_mean",
        "fold_macro_auc_std",
        "fold_macro_f1_mean",
        "fold_macro_f1_std",
        "fold_balanced_accuracy_mean",
        "fold_balanced_accuracy_std",
        "pooled_minus_fold_mean_auc",
    }
    assert required.issubset(main.columns)
    assert set(main["experiment"]) == {"rgb_rgb_capacity_control", "rgb_s_independent_dual", "rgb_r_independent_dual"}


def test_stage_c_probe_oof_correlation_ci_and_flags() -> None:
    stage_c = ABC_ROOT / "stage_c"
    probe_oof = pd.read_csv(stage_c / "acquisition_probe_oof_predictions.csv", dtype={"case_id": str})
    assert len(probe_oof) == 1500
    assert probe_oof.groupby("probe")["case_id"].nunique().to_dict() == {
        "camera": 500,
        "combined_acquisition": 500,
        "exif_numeric": 500,
    }
    corr = pd.read_csv(stage_c / "audit" / "model_exif_correlations.csv")
    within = pd.read_csv(stage_c / "audit" / "model_exif_within_label_correlations.csv")
    for frame in (corr, within):
        assert {"rho", "rho_ci_low", "rho_ci_high", "bootstrap_iterations", "bootstrap_valid_iterations"}.issubset(frame.columns)
        assert set(frame["bootstrap_iterations"]) == {2000}
    flags = pd.read_csv(stage_c / "audit" / "acquisition_confounding_flags.csv")
    assert "causal_claim" in flags.columns
    assert not flags["acquisition_dependence_flag"].astype(str).str.contains("不存在任何混杂|no acquisition dependence", case=False, regex=True).any()


def test_required_summary_and_report_files_exist() -> None:
    summary_files = [
        "stage_a_main_results.csv",
        "stage_a_error_complementarity.csv",
        "stage_a_fixed_fusion_results.csv",
        "stage_b_main_results.csv",
        "stage_b_fold_results.csv",
        "stage_b_paired_comparisons.csv",
        "stage_b_capacity_control_analysis.csv",
        "stage_b_training_stability.csv",
        "stage_c_acquisition_probe_results.csv",
        "stage_c_exif_correlations.csv",
        "stage_c_stratified_auc_results.csv",
        "stage_c_acquisition_confounding_flags.csv",
        "stage_d_candidate_matrix.csv",
        "abc_execution_status.csv",
        "abc_result_matrix.md",
    ]
    for name in summary_files:
        assert (ABC_ROOT / "summary" / name).is_file(), name
    for name in [
        "p1_extension_stage_a_oof_complementarity_report.md",
        "p1_extension_stage_b_fusion_results_report.md",
        "p1_extension_stage_c_acquisition_audit_report.md",
        "p1_extension_abc_decision_report.md",
    ]:
        assert (REPORTS / name).is_file(), name
