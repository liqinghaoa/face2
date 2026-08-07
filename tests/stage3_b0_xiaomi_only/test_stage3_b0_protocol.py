from __future__ import annotations

from pathlib import Path

import pandas as pd
import torch

from stage3_b0_xiaomi_only.pipeline import (
    EXPECTED_CONTROL,
    EXPECTED_PATIENT,
    EXPECTED_TOTAL,
    OUTPUT_DIR,
    XIAOMI_MODEL,
    build_xiaomi_cohort,
    normalize_camera,
    validate_inner_splits,
    validate_oof,
    validate_outer_split,
)


def test_camera_normalization_exact() -> None:
    assert normalize_camera(" M2006J10C ") == XIAOMI_MODEL
    assert normalize_camera("bvl-an00") != XIAOMI_MODEL


def test_xiaomi_cohort_counts_if_built(tmp_path: Path) -> None:
    cohort = build_xiaomi_cohort(tmp_path)
    assert len(cohort) == EXPECTED_TOTAL
    assert int((cohort["binary_label"] == 0).sum()) == EXPECTED_CONTROL
    assert int((cohort["binary_label"] == 1).sum()) == EXPECTED_PATIENT
    assert cohort["camera_model_normalized"].nunique() == 1
    assert cohort["camera_model_normalized"].iloc[0] == XIAOMI_MODEL
    assert not cohort["sample_id"].duplicated().any()
    assert cohort["image_path"].map(lambda p: Path(p).is_file()).all()


def test_generated_outer_and_inner_splits_are_group_safe() -> None:
    split_path = OUTPUT_DIR / "splits" / "xiaomi_control_patient_group5fold_v1.csv"
    inner_root = OUTPUT_DIR / "splits" / "inner"
    if not split_path.is_file():
        return
    outer = pd.read_csv(split_path, dtype={"sample_id": "string", "patient_group_id": "string"})
    assert not validate_outer_split(outer)
    inner_frames = []
    for fold in range(5):
        path = inner_root / f"fold_{fold}_inner_split.csv"
        assert path.is_file()
        inner_frames.append(pd.read_csv(path, dtype={"sample_id": "string", "patient_group_id": "string"}))
    assert not validate_inner_splits(outer, pd.concat(inner_frames, ignore_index=True))


def test_oof_contract_if_built() -> None:
    path = OUTPUT_DIR / "oof" / "xiaomi_b0_oof_predictions.csv"
    if not path.is_file():
        return
    oof = pd.read_csv(path, dtype={"sample_id": "string", "patient_group_id": "string"})
    validate_oof(OUTPUT_DIR, oof)
    assert len(oof) == EXPECTED_TOTAL
    assert oof["sample_id"].nunique() == EXPECTED_TOTAL
    assert set(oof["prediction_threshold_05"].unique()).issubset({0, 1})
    assert ((oof["probability_patient"] >= 0.5).astype(int) == oof["prediction_threshold_05"].astype(int)).all()


def test_loss_contract_bce_pos_weight_one() -> None:
    loss = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor([1.0]))
    assert float(loss.pos_weight.item()) == 1.0
