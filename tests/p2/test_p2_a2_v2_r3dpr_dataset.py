from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from losses.classification_losses import compute_class_weights
from p2_counterfactual.a2_v2_assets import MANIFEST_PATH, build_a2_v2_assets
from p2_counterfactual.assets import PRESET_NAMES
from p2_counterfactual.dataset import P2ASingleRGBDataset, load_p2_manifest, p2_a_collate
from p2_counterfactual.model import P2ASingleRGBResNet18
from p2_counterfactual.path_utils import resolve_project_path


def _manifest() -> Path:
    if not MANIFEST_PATH.is_file():
        build_a2_v2_assets()
    return MANIFEST_PATH


def test_a2_v2_manifest_real_500_cases_and_fixed_seven_views() -> None:
    frame = load_p2_manifest(_manifest())
    assert len(frame) == 500
    assert frame["case_id"].nunique() == 500
    assert frame.groupby("patient_group_id")["fold"].nunique().max() == 1
    assert set(frame["fold"].astype(int).unique()) == {0, 1, 2, 3, 4}
    for preset in PRESET_NAMES:
        column = f"relight_{preset}_path"
        assert column in frame.columns
        assert frame[column].map(lambda value: resolve_project_path(value, Path("E:/projects/face2"), require_exists=True).is_file()).all()
    assert not frame.astype(str).apply(lambda col: col.str.contains("P2_Counterfactual_Relighting500_v1|P2_DetailPreserving_Relighting500_v2", regex=True)).any().any()


def test_a2_v2_dataset_length_sampling_and_eval_sources() -> None:
    frame = load_p2_manifest(_manifest())
    train_cases = int((frame["fold"].astype(int) != 0).sum())
    train = P2ASingleRGBDataset(_manifest(), fold=0, split="train", input_mode="relight_mix", training=True, max_cases=None, seed=2026)
    assert len(train) == train_cases

    random.seed(2026)
    seen_sources: set[str] = set()
    seen_presets: set[str] = set()
    for index in range(160):
        item = train[index % len(train)]
        seen_sources.add(item["source_type"])
        if item["source_type"] == "relighted":
            seen_presets.add(str(item["preset_name"]))
            assert item["preset_index"] in range(6)
            assert Path(str(item["image_path"])).name in {f"relight_{preset}.png" for preset in PRESET_NAMES}
    assert seen_sources == {"original", "relighted"}
    assert len(seen_presets) == 6

    val = P2ASingleRGBDataset(_manifest(), fold=0, split="val", input_mode="original", training=False)
    val_item = val[0]
    assert val_item["source_type"] == "original"
    assert val_item["preset_name"] is None
    assert val_item["image"].shape == (3, 224, 224)
    assert val_item["image"].dtype == torch.float32

    rel = P2ASingleRGBDataset(
        _manifest(),
        fold=0,
        split="val",
        input_mode="original",
        training=False,
        evaluation_source="relighted",
        evaluation_preset_name="top",
    )
    rel_item = rel[0]
    assert rel_item["source_type"] == "relighted"
    assert rel_item["preset_name"] == "top"
    assert rel_item["preset_index"] == PRESET_NAMES.index("top")


def test_a2_v2_sampling_reproducible_and_multiworker_loads() -> None:
    def draw_sequence() -> list[tuple[str, str | None]]:
        random.seed(12345)
        dataset = P2ASingleRGBDataset(_manifest(), fold=0, split="train", input_mode="relight_mix", training=True, max_cases=16, seed=2026)
        sequence = []
        for index in range(len(dataset)):
            item = dataset[index]
            sequence.append((item["source_type"], item["preset_name"]))
        return sequence

    assert draw_sequence() == draw_sequence()

    dataset = P2ASingleRGBDataset(_manifest(), fold=0, split="train", input_mode="relight_mix", training=True, max_cases=8, seed=2026)
    loader = DataLoader(dataset, batch_size=4, num_workers=2, collate_fn=p2_a_collate)
    batch = next(iter(loader))
    assert batch["image"].shape == (4, 3, 224, 224)
    assert batch["label"].dtype == torch.long


def test_a2_v2_weighted_ce_and_model_contract() -> None:
    frame = load_p2_manifest(_manifest())
    train_labels = frame[frame["fold"].astype(int) != 0]["binary_label"].astype(int).tolist()
    weights = compute_class_weights(train_labels, 2)
    assert weights.shape == (2,)
    assert torch.isfinite(weights).all()

    model = P2ASingleRGBResNet18(pretrained=False, num_classes=2)
    logits, features = model(torch.randn(2, 3, 224, 224))
    assert logits.shape == (2, 2)
    assert features.shape == (2, 512)
    assert torch.isfinite(logits).all()
    assert torch.isfinite(features).all()
