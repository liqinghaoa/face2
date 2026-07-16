"""Protocol locks for Stage 2B before formal nonlinear calibration."""

from __future__ import annotations

import json
import inspect

import numpy as np

from scripts.train.run_optical_condition_calibration_stage2b import (
    EXPECTED_SPLIT_SHA256,
    FIRST_STAGE_COLUMNS,
    STAGE2A_COLUMNS,
    audit_splits,
    load_args,
    load_first_stage,
    load_stage2a_fold,
    load_stage2a_inputs,
    run_fold,
    select_rows,
    validate_stage2a_fold_alignment,
)
from utils.optical_condition_calibration import EXPECTED_CAMERAS
from utils.optical_condition_calibration_nn import deterministic_inner_split, sha256_file


def protocol():
    args = load_args([])
    first_stage = load_first_stage(args)
    stage2a = load_stage2a_inputs(args)
    splits = audit_splits(args, first_stage)
    return args, first_stage, stage2a, splits


def test_first_stage_stage2a_oof_and_fixed_split_are_complete() -> None:
    _, first_stage, stage2a, splits = protocol()
    assert len(first_stage) == first_stage["ID"].nunique() == 500
    assert len(stage2a["oof"]) == stage2a["oof"]["ID"].nunique() == 500
    assert splits["split_sha256"] == EXPECTED_SPLIT_SHA256
    val_ids = [identifier for fold in range(5) for identifier in splits["splits"][fold]["val"]["ID"]]
    assert len(val_ids) == len(set(val_ids)) == 500
    assert set(val_ids) == set(first_stage["ID"])


def test_every_outer_fold_is_400_100_disjoint_and_has_both_cameras() -> None:
    _, first_stage, _, splits = protocol()
    indexed = first_stage.set_index("ID")
    for fold in range(5):
        train = splits["splits"][fold]["train"]
        val = splits["splits"][fold]["val"]
        assert len(train) == train["ID"].nunique() == 400
        assert len(val) == val["ID"].nunique() == 100
        assert not set(train["ID"]) & set(val["ID"])
        assert set(indexed.loc[train["ID"], "camera_id"]) == set(EXPECTED_CAMERAS)


def test_stage2b_raw_and_outer_z_match_stage2a_for_all_folds() -> None:
    args, first_stage, _, splits = protocol()
    for fold in range(5):
        train = select_rows(first_stage, splits["splits"][fold]["train"]["ID"].tolist())
        val = select_rows(first_stage, splits["splits"][fold]["val"]["ID"].tolist())
        result = validate_stage2a_fold_alignment(fold, train, val, load_stage2a_fold(args, fold))
        assert result["maximum_z_absolute_error"] < 1e-12


def test_inner_splits_have_both_cameras_no_outer_val_and_fc_is_available() -> None:
    _, first_stage, _, splits = protocol()
    for fold in range(5):
        outer_train = select_rows(first_stage, splits["splits"][fold]["train"]["ID"].tolist())
        outer_val_ids = set(splits["splits"][fold]["val"]["ID"])
        for frame in (
            outer_train,
            outer_train.loc[outer_train["forehead_available"].eq(1)].copy(),
        ):
            inner_train, inner_val, manifest = deterministic_inner_split(frame, 2026 + fold * 100)
            assert set(inner_train["camera_id"]) == set(inner_val["camera_id"]) == set(EXPECTED_CAMERAS)
            assert not (set(inner_train["ID"]) | set(inner_val["ID"])) & outer_val_ids
            assert manifest["train_id_sha256"] and manifest["val_id_sha256"]
        fc_train, fc_val, _ = deterministic_inner_split(
            outer_train.loc[outer_train["forehead_available"].eq(1)].copy(), 2026 + fold * 100
        )
        assert fc_train["forehead_available"].eq(1).all()
        assert fc_val["forehead_available"].eq(1).all()


def test_stage2a_manifest_hashes_and_label_free_contract_are_complete() -> None:
    args, _, stage2a, splits = protocol()
    manifest = stage2a["manifest"]
    assert manifest["status"] == "COMPLETE"
    assert manifest["clinical_labels_loaded"] is False
    assert manifest["nyha_used"] is False
    assert manifest["split"]["sha256"] == splits["split_sha256"]
    assert manifest["oof_output_sha256"] == sha256_file(args.stage2a_oof)
    assert set(manifest["per_fold_split_hashes"]) == {str(fold) for fold in range(5)}


def test_input_whitelists_exclude_clinical_image_qc_and_raw_exif_fields() -> None:
    allowed = " ".join((*FIRST_STAGE_COLUMNS, *STAGE2A_COLUMNS)).casefold()
    for forbidden in (
        "nyha", "label_3class", "sex", "bnp", "brightnessvalue",
        "exposuretime", "fnumber", "isospeedratings", "valid_skin_fraction",
        "image_path", "global_image",
    ):
        assert forbidden not in allowed
    args = load_args([])
    source = (args.project_root / "scripts/train/run_optical_condition_calibration_stage2b.py").read_text(encoding="utf-8")
    assert 'usecols=["ID", "fold"]' in source
    assert "split_regenerated\": False" in source


def test_outer_validation_rows_are_deferred_until_after_selection_and_refit() -> None:
    source = inspect.getsource(run_fold)
    outer_val_load = source.index("outer_val = select_rows")
    assert source.index("train_epoch_selection(") < outer_val_load
    assert source.index("train_final_model(") < outer_val_load
    assert source.index('load_stage2a_fold(args, fold, roles=("val",))') > outer_val_load
