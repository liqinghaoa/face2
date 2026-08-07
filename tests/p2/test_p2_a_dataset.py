from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

from p2_counterfactual.assets import PRESET_NAMES
from p2_counterfactual.dataset import P2ASingleRGBDataset, load_p2_manifest

MANIFEST = Path("data/processed/P2_Counterfactual_Relighting500_v1/manifests/p2_training_manifest.csv")


def test_dataset_reads_manifest_and_preserves_flags() -> None:
    frame = load_p2_manifest(MANIFEST)
    assert len(frame) == 500
    assert frame.groupby("patient_group_id")["fold"].nunique().max() == 1
    assert frame["p1_qc_flag"].sum() > 0
    assert "boundary_uncertain" in frame.columns


def test_a0_returns_original_single_rgb_without_residual_or_shading() -> None:
    ds = P2ASingleRGBDataset(MANIFEST, fold=0, split="train", input_mode="original", training=True, max_cases=4, seed=2026)
    sample = ds[0]
    assert sample["image"].shape == (3, 224, 224)
    assert sample["source_type"] == "original"
    assert sample["preset_name"] is None
    assert "signed_residual" not in sample
    assert "shading" not in sample


def test_original_rgb_override_dir_replaces_manifest_original_rgb(tmp_path: Path) -> None:
    frame = load_p2_manifest(MANIFEST).head(1).copy()
    case_id = str(frame.iloc[0]["case_id"])
    override = tmp_path / "meanbg"
    override.mkdir()
    expected = np.full((224, 224, 3), 127, dtype=np.uint8)
    Image.fromarray(expected, mode="RGB").save(override / f"{case_id}.png")
    ds = P2ASingleRGBDataset(
        frame,
        fold=int(frame.iloc[0]["fold"]),
        split="all",
        input_mode="original",
        training=False,
        original_rgb_override_dir=override,
        horizontal_flip_probability=0.0,
        normalization_mean=(0.0, 0.0, 0.0),
        normalization_std=(1.0, 1.0, 1.0),
    )
    sample = ds[0]
    assert sample["source_type"] == "original"
    assert np.isclose(float(sample["image"].mean()), 127.0 / 255.0, atol=1e-6)


def test_a2_can_sample_original_and_relighted_with_named_presets() -> None:
    random.seed(2026)
    ds = P2ASingleRGBDataset(MANIFEST, fold=0, split="train", input_mode="relight_mix", training=True, max_cases=8, seed=2026)
    seen_sources = set()
    seen_presets = set()
    for idx in range(80):
        item = ds[idx % len(ds)]
        seen_sources.add(item["source_type"])
        if item["preset_name"] is not None:
            seen_presets.add(item["preset_name"])
    assert {"original", "relighted"}.issubset(seen_sources)
    assert seen_presets.issubset(set(PRESET_NAMES))
    assert len(seen_presets) >= 4


def test_a3_returns_paired_original_and_counterfactual() -> None:
    random.seed(2026)
    ds = P2ASingleRGBDataset(MANIFEST, fold=0, split="train", input_mode="paired", training=True, max_cases=4, seed=2026)
    sample = ds[0]
    assert sample["original_image"].shape == (3, 224, 224)
    assert sample["counterfactual_image"].shape == (3, 224, 224)
    assert sample["preset_name"] in PRESET_NAMES


def test_a2_sampling_is_label_independent_with_reasonable_tolerance() -> None:
    random.seed(123)
    frame = pd.concat(
        [
            load_p2_manifest(MANIFEST).query("binary_label == 0").head(20),
            load_p2_manifest(MANIFEST).query("binary_label == 1").head(20),
        ],
        ignore_index=True,
    )
    ds = P2ASingleRGBDataset(frame, fold=99, split="all", input_mode="relight_mix", training=True, max_cases=None, seed=2026)
    counts = {0: {"original": 0, "relighted": 0}, 1: {"original": 0, "relighted": 0}}
    for idx in range(400):
        item = ds[idx % len(ds)]
        counts[item["label"]][item["source_type"]] += 1
    for label_counts in counts.values():
        ratio = label_counts["relighted"] / max(1, label_counts["original"] + label_counts["relighted"])
        assert 0.35 <= ratio <= 0.65
