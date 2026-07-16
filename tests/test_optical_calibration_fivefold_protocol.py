"""Lock the Stage 2A run to the existing ordinary 500-case five-fold split."""

from __future__ import annotations

from scripts.preprocess.run_optical_condition_calibration_stage2a import (
    audit_splits,
    load_args,
    load_first_stage,
)
from utils.optical_condition_calibration import EXPECTED_CAMERAS

EXPECTED_SPLIT_SHA256 = "fe5102c02890c546f323b0a94ebc5b125ebcfeb50e62d2d43f0564b4b383f24b"


def protocol():
    args = load_args([])
    first_stage = load_first_stage(args)
    return args, first_stage, audit_splits(args, first_stage)


def test_fixed_split_sha_and_exact_500_id_oof_coverage() -> None:
    _, first_stage, info = protocol()
    assert info["split_sha256"] == EXPECTED_SPLIT_SHA256
    validation_ids = []
    for fold in range(5):
        validation_ids.extend(info["splits"][fold]["val"]["ID"].tolist())
    assert len(first_stage) == 500
    assert len(validation_ids) == 500
    assert len(set(validation_ids)) == 500
    assert set(validation_ids) == set(first_stage["ID"])


def test_every_fold_is_400_100_disjoint_and_complete() -> None:
    _, first_stage, info = protocol()
    all_ids = set(first_stage["ID"])
    for fold in range(5):
        train = info["splits"][fold]["train"]
        val = info["splits"][fold]["val"]
        train_ids, val_ids = set(train["ID"]), set(val["ID"])
        assert len(train) == train["ID"].nunique() == 400
        assert len(val) == val["ID"].nunique() == 100
        assert not train_ids & val_ids
        assert train_ids | val_ids == all_ids
        assert (val["fold"] == fold).all()
        assert not (train["fold"] == fold).any()


def test_both_devices_and_available_foreheads_exist_in_each_training_fold() -> None:
    _, first_stage, info = protocol()
    indexed = first_stage.set_index("ID")
    for fold in range(5):
        train_ids = info["splits"][fold]["train"]["ID"].tolist()
        train = indexed.loc[train_ids]
        assert set(train["camera_id"]) == set(EXPECTED_CAMERAS)
        assert int(train["forehead_available"].sum()) > 6


def test_split_reader_exposes_only_id_and_fold() -> None:
    _, _, info = protocol()
    for fold in range(5):
        for role in ("train", "val"):
            assert info["splits"][fold][role].columns.tolist() == ["ID", "fold"]
