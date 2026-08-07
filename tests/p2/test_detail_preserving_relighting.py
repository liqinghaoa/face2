from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from p2_counterfactual.detail_preserving_relighting import (
    EPSILON,
    MEANBG_ROOT,
    OUTPUT_ROOT,
    generate_detail_preserving_images,
)
from p2_counterfactual.dataset import P2ASingleRGBDataset, load_p2_manifest
from p2_counterfactual.path_utils import resolve_project_path


def test_ratio_formula_preserves_background_and_shape() -> None:
    meanbg = np.full((4, 4, 3), 0.4, np.float32)
    original = np.ones((4, 4), np.float32)
    relighted = np.stack([np.ones((4, 4), np.float32) * (index + 1) for index in range(6)], axis=0)
    alpha = np.zeros((4, 4), np.float32)
    alpha[:2, :2] = 1.0
    out = generate_detail_preserving_images(meanbg, original, relighted, alpha, epsilon=EPSILON)
    assert out.shape == (6, 4, 4, 3)
    assert out.dtype == np.float32
    assert np.isfinite(out).all()
    assert out.min() >= 0.0
    assert out.max() <= 1.0
    assert np.allclose(out[:, 2:, :, :], meanbg[None, 2:, :, :])
    assert np.all(out[1, :2, :2, :] > out[0, :2, :2, :])


def test_single_channel_shading_broadcasts_to_rgb() -> None:
    meanbg = np.ones((3, 3, 3), np.float32) * 0.2
    original = np.ones((1, 3, 3), np.float32)
    relighted = np.ones((6, 3, 3, 1), np.float32) * 2.0
    alpha = np.ones((1, 3, 3), np.float32)
    out = generate_detail_preserving_images(meanbg, original, relighted, alpha)
    assert out.shape == (6, 3, 3, 3)
    assert np.allclose(out, np.clip(0.2 * (2.0 / (1.0 + EPSILON)), 0, 1), atol=1e-6)


def test_nan_input_fails() -> None:
    meanbg = np.ones((4, 4, 3), np.float32)
    original = np.ones((4, 4, 3), np.float32)
    relighted = np.ones((6, 4, 4, 3), np.float32)
    relighted[0, 0, 0, 0] = np.nan
    alpha = np.ones((4, 4), np.float32)
    with pytest.raises(ValueError):
        generate_detail_preserving_images(meanbg, original, relighted, alpha)


def test_v2_manifest_and_dataset_when_generated() -> None:
    manifest = OUTPUT_ROOT / "manifests/p2_training_manifest.csv"
    if not manifest.is_file():
        pytest.skip("v2 detail-preserving manifest not generated")
    frame = load_p2_manifest(manifest)
    assert len(frame) == 500
    assert frame["case_id"].nunique() == 500
    if int(frame["p2_training_ready"].sum()) != 500:
        pytest.skip("v2 detail-preserving manifest exists but full-500 generation is not complete")
    assert int(frame["p2_training_ready"].sum()) == 500
    assert frame["relighting_npz_path"].str.contains("P2_DetailPreserving_Relighting500_v2").all()
    assert frame["old_deca_relighting_npz_path"].str.contains("P1_DECA_Frozen500_v1").all()
    first = frame.iloc[0]
    with np.load(resolve_project_path(first["relighting_npz_path"], Path.cwd(), require_exists=True), allow_pickle=False) as rel:
        assert rel["relighted_images"].shape == (6, 224, 224, 3)
        assert rel["relighted_images"].dtype == np.float32
        assert rel["relighted_images"].min() >= 0.0
        assert rel["relighted_images"].max() <= 1.0


def test_p2_dataset_reads_new_relighting_when_generated() -> None:
    manifest = OUTPUT_ROOT / "manifests/p2_training_manifest.csv"
    if not manifest.is_file():
        pytest.skip("v2 detail-preserving manifest not generated")
    frame = pd.read_csv(manifest, dtype={"case_id": str})
    frame = frame[frame["p2_training_ready"] == True].head(8)  # noqa: E712
    if frame.empty:
        pytest.skip("no generated v2 cases are ready")
    ds = P2ASingleRGBDataset(
        frame,
        fold=int(frame.iloc[0]["fold"]),
        split="all",
        input_mode="relight_mix",
        training=True,
        original_probability=0.0,
        horizontal_flip_probability=0.0,
        normalization_mean=(0.0, 0.0, 0.0),
        normalization_std=(1.0, 1.0, 1.0),
        original_rgb_override_dir=MEANBG_ROOT,
    )
    item = ds[0]
    assert item["source_type"] == "relighted"
    assert item["image"].shape == (3, 224, 224)
    assert Path(ds.frame.iloc[0]["relighting_npz_path"]).parts.count("P2_DetailPreserving_Relighting500_v2") == 1
