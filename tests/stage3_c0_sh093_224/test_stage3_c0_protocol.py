from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

ROOT = Path(__file__).resolve().parents[2] / "experiments" / "lighting_confounding" / "Stage3_C0_SH093_224_SameCameraSignal_and_DeviceDomain_Audit_v1"
B0 = Path(__file__).resolve().parents[2] / "experiments" / "lighting_confounding" / "Stage3_B0_XiaomiOnly_ResNet18_ControlVsPatient_Group5Fold_v1"


def read(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, dtype={"sample_id": str, "patient_group_id": str}, encoding="utf-8-sig")


def test_preflight_and_shared_assets_pass():
    p = json.loads((ROOT / "preflight/stage3_c0_preflight_summary.json").read_text(encoding="utf-8"))
    a = json.loads((ROOT / "shared_asset_audit/sh093_224_asset_audit.json").read_text(encoding="utf-8"))
    assert p["status"] == "passed"
    assert a["status"] == "passed"
    assert a["n_inventory"] == 500
    assert a["n_failures"] == 0


def test_sh093_images_are_strict_224_rgb():
    inv = read(ROOT / "shared_asset_audit/sh093_224_asset_inventory.csv")
    assert len(inv) == 500
    assert inv["size"].eq("224x224").all()
    assert inv["mode"].eq("RGB").all()
    assert inv["dtype"].eq("uint8").all()
    assert inv["decode_ok"].all()


def test_c0a_cohort_counts_camera_and_images():
    c = read(ROOT / "c0a_disease/cohort/c0a_xiaomi_sh093_cohort.csv")
    assert len(c) == 233 and c.sample_id.nunique() == 233
    assert (c.binary_label == 0).sum() == 115
    assert (c.binary_label == 1).sum() == 118
    assert c.camera_model_normalized.eq("M2006J10C").all()
    assert c.image_path.map(lambda x: Path(x).is_file()).all()


def test_c0a_reuses_b0_outer_and_inner_rows():
    audit = json.loads((ROOT / "c0a_disease/splits/c0a_split_identity_audit.json").read_text(encoding="utf-8"))
    assert audit["status"] == "passed"
    assert audit["outer_row_identity"] and audit["inner_row_identity"]
    a = read(ROOT / "c0a_disease/splits/c0a_reused_outer_split.csv")
    b = read(B0 / "splits/xiaomi_control_patient_group5fold_v1.csv")
    keys = ["sample_id", "patient_group_id", "binary_label", "outer_fold"]
    assert a[keys].sort_values("sample_id").reset_index(drop=True).equals(b[keys].sort_values("sample_id").reset_index(drop=True))


def test_c0a_group_safe_inner_and_outer():
    outer = read(ROOT / "c0a_disease/splits/c0a_reused_outer_split.csv")
    for f in range(5):
        g = set(outer.loc[outer.outer_fold == f, "patient_group_id"])
        assert not g.intersection(set(outer.loc[outer.outer_fold != f, "patient_group_id"]))
    inner = read(ROOT / "c0a_disease/splits/c0a_reused_inner_split_inventory.csv")
    for f in range(5):
        x = inner[inner.outer_fold == f]
        assert not set(x.loc[x.inner_split == "inner_train", "patient_group_id"]).intersection(set(x.loc[x.inner_split == "inner_val", "patient_group_id"]))


def test_c0a_oof_contract_and_threshold():
    oof = read(ROOT / "c0a_disease/oof/c0a_xiaomi_sh093_oof_predictions.csv")
    assert len(oof) == 233 and oof.sample_id.nunique() == 233
    assert np.isfinite(oof.probability).all() and np.isfinite(oof.logit).all()
    assert set(oof.prediction_05.unique()).issubset({0, 1})
    assert ((oof.probability >= 0.5).astype(int) == oof.prediction_05).all()
    assert set(oof.outer_fold) == set(range(5))


def test_c0a_nested_checkpoint_contract_and_no_forbidden_inputs():
    contract = json.loads((ROOT / "c0a_disease/training_contract/c0a_training_contract.json").read_text(encoding="utf-8"))
    assert contract["name"] == "stage3_b0_fallback_contract"
    assert contract["nested"] is True and contract["threshold"] == 0.5
    assert contract["loss"] == "BCEWithLogitsLoss(pos_weight=1.0)"
    c = read(ROOT / "c0a_disease/cohort/c0a_xiaomi_sh093_cohort.csv")
    assert "camera_label" not in c.columns and "BrightnessValue" not in c.columns
    assert not c.camera_model_normalized.str.contains("HONOR|BVL-AN00", case=False, regex=True).any()
    for f in range(5):
        assert (ROOT / f"c0a_disease/fold_{f}/checkpoints/best_auc.pth").is_file()
        assert (ROOT / f"c0a_disease/fold_{f}/checkpoints/last.pth").is_file()


def test_c0b_patient_only_cohort_and_camera_labels():
    audit = json.loads((ROOT / "c0b_camera/cohort/c0b_camera_cohort_audit.json").read_text(encoding="utf-8"))
    c = read(ROOT / "c0b_camera/cohort/c0b_patient_only_camera_cohort.csv")
    assert audit["status"] == "passed"
    assert len(c) == 385 and (c.binary_label == 1).all()
    assert (c.task_label == 0).sum() == 118
    assert (c.task_label == 1).sum() == 267
    assert set(c.camera_model_normalized) == {"M2006J10C", "BVL-AN00"}


def test_c0b_outer_inner_group_safe_and_both_classes():
    s = read(ROOT / "c0b_camera/splits/c0b_patient_camera_group5fold.csv")
    for f in range(5):
        x = s[s.outer_fold == f]
        assert set(x.task_label) == {0, 1}
        assert not set(x.patient_group_id).intersection(set(s.loc[s.outer_fold != f, "patient_group_id"]))
    inner = read(ROOT / "c0b_camera/splits/c0b_inner_split_inventory.csv")
    for f in range(5):
        x = inner[inner.outer_fold == f]
        assert set(x.loc[x.inner_split == "inner_train", "task_label"]) == {0, 1}
        assert set(x.loc[x.inner_split == "inner_val", "task_label"]) == {0, 1}
        assert not set(x.loc[x.inner_split == "inner_train", "patient_group_id"]).intersection(set(x.loc[x.inner_split == "inner_val", "patient_group_id"]))


def test_c0b_oof_baselines_and_class_weight_contract():
    oof = read(ROOT / "c0b_camera/oof/c0b_patient_camera_oof_predictions.csv")
    assert len(oof) == 385 and oof.sample_id.nunique() == 385
    assert np.isfinite(oof.probability).all()
    contract = json.loads((ROOT / "c0b_camera/training_contract/c0b_training_contract.json").read_text(encoding="utf-8"))
    assert contract["positive"] == "HONOR"
    assert "inner train" in contract["pos_weight"]
    assert contract["no_weighted_sampler"] is True
    assert (ROOT / "c0b_camera/baselines/covariate_baseline_metrics.csv").is_file()
    assert (ROOT / "c0b_camera/baselines/lowlevel_metrics.csv").is_file()


def test_matching_is_group_level_and_not_model_selected():
    c = json.loads((ROOT / "c0b_camera/matched/matching_contract.json").read_text(encoding="utf-8"))
    pairs = read(ROOT / "c0b_camera/matched/matched_group_pairs.csv")
    assert c["matching_unit"] == "patient_group_id"
    assert c["n_pairs"] >= 60
    assert pairs.xiaomi_group.nunique() == len(pairs)
    assert pairs.honor_group.nunique() == len(pairs)
    assert (ROOT / "c0b_camera/matched/matched_metrics.csv").is_file()


def test_required_reports_and_decision_fields():
    paths = ["c0a_disease/reports/c0a_report.md", "c0a_disease/reports/c0a_machine_summary.json", "c0b_camera/reports/c0b_report.md", "c0b_camera/reports/c0b_machine_summary.json", "joint/reports/stage3_c0_joint_report.md", "joint/reports/stage3_c0_final_decision.json", "joint/reports/stage3_c0_output_inventory.json", "matched_control/preflight/matched_control_decision.json"]
    assert all((ROOT / p).is_file() for p in paths)
    d = json.loads((ROOT / "joint/reports/stage3_c0_final_decision.json").read_text(encoding="utf-8"))
    assert set(d) >= {"c0a_same_camera_signal_level", "c0b_camera_domain_predictability_level", "full500_sh093_interpretation", "sh093_main_binary_candidate", "next_stage_recommendation"}
    assert d["matched_control_status"] == "unavailable"


def test_no_forbidden_followup_experiment_entered():
    report = (ROOT / "joint/reports/stage3_c0_joint_report.md").read_text(encoding="utf-8")
    assert "No multi-SH, fixedcam, Exposure/Gamma, ROI, or Stage3-B1 experiment was run." in report
