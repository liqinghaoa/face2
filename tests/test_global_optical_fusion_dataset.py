from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
from PIL import Image

from datasets.global_optical_fusion_dataset import GlobalOpticalFusionDataset
from datasets.nyha_3class_face_dataset import build_transforms
from utils.optical_feature_preprocessor import (
    AVAILABILITY_COLUMN,
    RAW_FEATURE_COLUMNS,
    FeatureScaler,
    assert_classifier_feature_path,
    fit_feature_scaler,
    load_feature_frame,
)


def _fixture(tmp_path: Path):
    ids = ["001-1", "002", "003", "004", "005", "006"]
    image_root = tmp_path / "images"
    image_root.mkdir()
    for index, identifier in enumerate(ids):
        Image.new("RGB", (12, 12), (index * 10, 20, 30)).save(image_root / f"{identifier}.png")
    split = pd.DataFrame({
        "ID": ids, "patient_group_id": [f"p{i}" for i in range(6)],
        "NYHA": [0, 1, 3, 0, 2, 4], "label_3class": [0, 1, 2, 0, 1, 2],
        "label_3class_name": ["normal", "mild", "severe", "normal", "mild", "severe"],
        "SEX": [0, 1, 0, 1, 0, 1], "sex_name": ["female", "male"] * 3,
        "fold": [0] * 6,
    })
    split_path = tmp_path / "split.csv"
    split.to_csv(split_path, index=False)
    available = np.array([1, 1, 0, 1, 0, 1])
    values = np.arange(36, dtype=float).reshape(6, 6) / 10 + np.arange(6)
    values[available == 0, 3:] = np.nan
    features = pd.DataFrame(values, columns=RAW_FEATURE_COLUMNS)
    features.insert(0, AVAILABILITY_COLUMN, available)
    features.insert(0, "ID", ids)
    source = tmp_path / "features.csv"
    features.to_csv(source, index=False)
    schema = tmp_path / "schema.json"
    schema.write_text(json.dumps({"derived_observation_columns": list(RAW_FEATURE_COLUMNS)}), encoding="utf-8")
    manifest = tmp_path / "manifest.json"
    manifest.write_text('{"status":"COMPLETE"}', encoding="utf-8")
    config = tmp_path / "config.yaml"
    config.write_text("test: true\n", encoding="utf-8")
    scaler = fit_feature_scaler(
        features, "global_raw", 0, source_path=source, schema_path=schema,
        upstream_manifest_path=manifest, split_path=split_path,
        code_paths=[Path(__file__)], config_path=config, project_root=tmp_path,
    )
    return ids, image_root, split_path, features, scaler, source, schema


def test_scaler_population_rules_and_roundtrip(tmp_path):
    ids, _, split_path, features, scaler, source, _ = _fixture(tmp_path)
    values = features[list(RAW_FEATURE_COLUMNS)].to_numpy(float)
    available = features[AVAILABILITY_COLUMN].to_numpy() == 1
    expected_mean = [values[:, i].mean() if i < 3 else values[available, i].mean() for i in range(6)]
    expected_std = [values[:, i].std(ddof=0) if i < 3 else values[available, i].std(ddof=0) for i in range(6)]
    assert scaler.ddof == 0
    assert scaler.mean == pytest.approx(expected_mean)
    assert scaler.std == pytest.approx(expected_std)
    assert scaler.valid_n == [6, 6, 6, 4, 4, 4]
    assert scaler.train_id_sha256
    assert scaler.source_sha256
    aux = scaler.transform(features)
    assert aux.dtype == np.float32
    assert aux.shape == (6, 7)
    assert np.all(aux[~available, 3:6] == 0)
    assert np.any(aux[~available, :3] != 0)
    assert np.array_equal(aux[:, -1], available.astype(np.float32))
    path = tmp_path / "scaler.json"
    scaler.save_json(path)
    restored = FeatureScaler.load_json(path)
    assert restored == scaler
    assert np.array_equal(restored.transform(features), aux)


def test_scaler_uses_train_only_and_rejects_low_std(tmp_path):
    _, _, _, features, scaler, *_ = _fixture(tmp_path)
    changed_val = features.copy()
    changed_val.loc[:, list(RAW_FEATURE_COLUMNS)[:3]] += 10_000
    assert scaler.mean != pytest.approx(changed_val[list(RAW_FEATURE_COLUMNS)].mean(skipna=True).tolist())
    bad = features.copy()
    bad[RAW_FEATURE_COLUMNS[0]] = 1.0
    with pytest.raises(ValueError, match="below"):
        fit_feature_scaler(
            bad, "global_raw", 0, source_path=tmp_path / "features.csv",
            schema_path=tmp_path / "schema.json", upstream_manifest_path=tmp_path / "manifest.json",
            split_path=tmp_path / "split.csv", code_paths=[Path(__file__)],
            config_path=tmp_path / "config.yaml", project_root=tmp_path,
        )


def test_strict_missing_pattern(tmp_path):
    *_, features, scaler, _, _ = _fixture(tmp_path)
    bad = features.copy()
    bad.loc[0, RAW_FEATURE_COLUMNS[3]] = np.nan
    with pytest.raises(ValueError, match="Available"):
        scaler.transform(bad)
    bad = features.copy()
    bad.loc[bad[AVAILABILITY_COLUMN] == 0, RAW_FEATURE_COLUMNS[3]] = 0.0
    with pytest.raises(ValueError, match="Unavailable"):
        scaler.transform(bad)


@pytest.mark.parametrize(
    ("variant", "width"),
    [("global_only", 0), ("global_mask", 1), ("global_raw", 7)],
)
def test_dataset_preserves_string_ids_order_and_aux_shape(tmp_path, variant, width):
    ids, image_root, split_path, features, scaler, *_ = _fixture(tmp_path)
    feature_frame = None if variant == "global_only" else (
        features[["ID", AVAILABILITY_COLUMN]] if variant == "global_mask" else features
    )
    dataset = GlobalOpticalFusionDataset(
        split_path, variant=variant, fold=0, split_role="train",
        image_root=image_root, transform=build_transforms("val", image_size=16),
        feature_frame=feature_frame, scaler=scaler if variant == "global_raw" else None,
    )
    assert dataset.ids == ids
    item = dataset[0]
    assert item["ID"] == "001-1"
    assert item["aux_features"].dtype == torch.float32
    assert item["aux_features"].shape == (width,)
    assert "image_path" not in item
    assert not {"NYHA", "SEX", "fold"}.intersection(
        set(features.columns) - {"ID", AVAILABILITY_COLUMN, *RAW_FEATURE_COLUMNS}
    )


@pytest.mark.parametrize("mutation", ["duplicate", "missing", "extra"])
def test_dataset_rejects_non_bijective_feature_join(tmp_path, mutation):
    _, image_root, split_path, features, _, *_ = _fixture(tmp_path)
    if mutation == "duplicate":
        altered = pd.concat([features, features.iloc[[0]]], ignore_index=True)
    elif mutation == "missing":
        altered = features.iloc[:-1]
    else:
        altered = pd.concat([features, features.iloc[[0]].assign(ID="extra")], ignore_index=True)
    with pytest.raises(ValueError, match="duplicate|exactly"):
        GlobalOpticalFusionDataset(
            split_path, variant="global_mask", fold=0, split_role="train",
            image_root=image_root, feature_frame=altered[["ID", AVAILABILITY_COLUMN]],
        )


def test_stage2_fold_and_role_are_strict(tmp_path):
    ids, image_root, split_path, features, _, *_ = _fixture(tmp_path)
    stage = features.rename(columns=dict(zip(RAW_FEATURE_COLUMNS, [
        "calibrated_cheek_mean_log2_y", "calibrated_cheek_mean_log2_rg",
        "calibrated_cheek_mean_log2_bg", "calibrated_forehead_minus_cheek_log2_y",
        "calibrated_forehead_minus_cheek_log2_rg", "calibrated_forehead_minus_cheek_log2_bg",
    ])))
    stage["fold"] = 1
    stage["split_role"] = "train"
    with pytest.raises(ValueError, match="fold"):
        GlobalOpticalFusionDataset(
            split_path, variant="global_stage2a", fold=0, split_role="train",
            image_root=image_root, feature_frame=stage, scaler=None,
        )
    stage["fold"] = 0
    stage["split_role"] = "val"
    with pytest.raises(ValueError, match="split_role"):
        GlobalOpticalFusionDataset(
            split_path, variant="global_stage2a", fold=0, split_role="train",
            image_root=image_root, feature_frame=stage, scaler=None,
        )


def test_allowlist_loader_and_oof_guard(tmp_path):
    ids, _, _, features, _, source, schema = _fixture(tmp_path)
    extra = features.assign(camera_id="forbidden", NYHA=4, residual_x=99)
    extra.to_csv(source, index=False)
    loaded = load_feature_frame(source, "global_raw", ids, schema_path=schema)
    assert loaded.columns.tolist() == ["ID", AVAILABILITY_COLUMN, *RAW_FEATURE_COLUMNS]
    oof = tmp_path / "oof_calibrated_features.csv"
    features.to_csv(oof, index=False)
    with pytest.raises(ValueError, match="OOF"):
        assert_classifier_feature_path(oof)


def test_global_only_rejects_any_optical_frame(tmp_path):
    _, image_root, split_path, features, *_ = _fixture(tmp_path)
    with pytest.raises(ValueError, match="must not"):
        GlobalOpticalFusionDataset(
            split_path, variant="global_only", fold=0, split_role="train",
            image_root=image_root, feature_frame=features,
        )
