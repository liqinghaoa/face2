from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "experiments" / "500Data" / "P1_RGBSR_TripleBranch_v1"


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def test_stage_d_final_status_and_no_stage_d_v2() -> None:
    manifest = _read_json(OUT / "metadata" / "run_manifest.json")
    assert manifest["final_status"] == "P1_STAGE_D_RGBSR_V1_COMPLETE"
    assert manifest["stage_d_v2_started"] is False
    approval = _read_json(OUT / "metadata" / "STAGE_D_V1_EXECUTION_APPROVED.json")
    assert approval["stage_d_v2_approved"] is False
    assert approval["camera_features_enabled"] is False
    assert approval["exif_features_enabled"] is False


def test_stage_d_smoke_and_five_formal_folds_complete() -> None:
    smoke = _read_json(OUT / "smoke" / "SMOKE_SUCCESS.json")
    assert smoke["prediction_rows"] == 100
    assert smoke["finite_loss"] is True
    assert smoke["finite_logits"] is True
    assert smoke["peak_gpu_memory_recorded"] is True
    for fold in range(5):
        success = _read_json(OUT / f"fold_{fold}" / "_FOLD_SUCCESS.json")
        assert success["prediction_rows"] == 100
        assert success["unique_case_ids"] == 100
        assert (OUT / f"fold_{fold}" / "preprocessing" / "shading_fitted_parameters.json").is_file()
        assert (OUT / f"fold_{fold}" / "preprocessing" / "residual_fitted_parameters.json").is_file()


def test_stage_d_oof_complete_visit_case_only() -> None:
    oof = pd.read_csv(OUT / "oof" / "oof_predictions_visit.csv", dtype={"case_id": str})
    assert len(oof) == 500
    assert oof["case_id"].nunique() == 500
    assert {"case_id", "patient_group_id", "fold", "label_original", "label_3class", "label_binary", "prob_control", "prob_patient", "pred_binary", "best_epoch", "checkpoint_path"}.issubset(oof.columns)
    assert not (OUT / "oof" / "oof_predictions_group.csv").exists()
    audit = _read_json(OUT / "metadata" / "oof_alignment_audit.json")
    assert audit == {
        "rows": 500,
        "unique_case_id": 500,
        "duplicate": 0,
        "missing": 0,
        "unexpected": 0,
        "label_mismatch": 0,
        "fold_mismatch": 0,
        "patient_group_mismatch": 0,
    }


def test_stage_d_summary_and_report_outputs_exist() -> None:
    for name in [
        "stage_d_v1_main_results.csv",
        "stage_d_v1_comparison_table.csv",
        "stage_d_v1_paired_bootstrap.csv",
        "stage_d_v1_fold_results.csv",
        "stage_d_v1_training_stability.csv",
        "stage_d_v1_result_matrix.md",
    ]:
        assert (OUT / "summary" / name).is_file(), name
    main = pd.read_csv(OUT / "summary" / "stage_d_v1_main_results.csv")
    required = {"macro_auc", "macro_f1", "balanced_accuracy", "patient_sensitivity", "control_specificity", "fold_macro_auc_mean", "fold_macro_auc_std"}
    assert required.issubset(main.columns)
    comparison = pd.read_csv(OUT / "summary" / "stage_d_v1_comparison_table.csv")
    assert {"P1-RGB", "RGB+RGB capacity control", "RGB+S independent dual", "RGB+R independent dual", "P1-S", "P1-R"} == set(comparison["comparator"])
    report = ROOT / "reports" / "p1_stage_d_rgbsr_triplebranch_v1_results.md"
    assert report.is_file()
    text = report.read_text(encoding="utf-8")
    assert "stage_d_v2_started = false" in text
    assert "threshold_search_performed = false" in text
