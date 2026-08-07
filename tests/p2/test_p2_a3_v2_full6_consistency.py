from __future__ import annotations

import random
from pathlib import Path

import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader

from p2_counterfactual.assets import PRESET_NAMES
from p2_counterfactual.a3_v2_assets import MANIFEST_PATH, build_a3_v2_assets
from p2_counterfactual.config import resolve_p2_a_config
from p2_counterfactual.dataset import P2ASingleRGBDataset, load_p2_manifest, p2_a_collate
from p2_counterfactual.model import P2ASingleRGBResNet18
from p2_counterfactual.path_utils import resolve_project_path
from p2_counterfactual.trainer import P2AFoldTrainer

MANIFEST = MANIFEST_PATH
CONFIG = Path("config/p2/p2_a/p2_a3_v2_r3dpr_full6_consistency_meanbg.yaml")


class TinyP2Model(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.classifier = nn.Linear(3, 2)

    def forward(self, images: torch.Tensor):
        pooled = images.mean(dim=(2, 3))
        logits = self.classifier(pooled)
        features = torch.zeros(images.size(0), 512, device=images.device)
        features[:, :3] = pooled
        return logits, features


def test_a3_v2_reuses_r3dpr_manifest_with_500_cases_and_seven_views() -> None:
    if not MANIFEST.is_file():
        build_a3_v2_assets()
    frame = load_p2_manifest(MANIFEST)
    assert len(frame) == 500
    assert frame["case_id"].nunique() == 500
    assert frame.groupby("patient_group_id")["fold"].nunique().max() == 1
    assert not frame.astype(str).apply(lambda col: col.str.contains("P2_DetailPreserving_Relighting500_v2|relighted_images", regex=True)).any().any()
    for preset in PRESET_NAMES:
        column = f"relight_{preset}_path"
        assert column in frame.columns
        assert frame[column].map(lambda value: resolve_project_path(value, Path("E:/projects/face2"), require_exists=True).is_file()).all()
    for index, preset in enumerate(PRESET_NAMES, start=1):
        assert set(frame[f"preset_{index}_id"].astype(str)) == {preset}
        assert frame[f"relight_{index}_path"].astype(str).str.contains(f"relight_{preset}.png", regex=False).all()


def test_a3_v2_full6_dataset_shape_order_and_no_colorjitter() -> None:
    if not MANIFEST.is_file():
        build_a3_v2_assets()
    config = resolve_p2_a_config(CONFIG)
    dataset = P2ASingleRGBDataset(
        config["manifest_path"],
        fold=0,
        split="train",
        input_mode="full6_consistency",
        training=True,
        color_jitter_enabled=False,
        horizontal_flip_probability=0.0,
        project_root=config["project_root"],
        path_resolution=config["path_resolution"],
    )
    item = dataset[0]
    assert len(dataset) == 400
    assert item["original_image"].shape == (3, 224, 224)
    assert item["relighted_images"].shape == (6, 3, 224, 224)
    assert item["preset_names"] == list(PRESET_NAMES)
    assert item["preset_ids"] == list(range(6))
    assert len(item["relighted_paths"]) == 6
    assert all(Path(path).name == f"relight_{preset}.png" for path, preset in zip(item["relighted_paths"], PRESET_NAMES))


def test_a3_v2_synchronized_flip_is_reproducible_and_collates_multiworker() -> None:
    if not MANIFEST.is_file():
        build_a3_v2_assets()
    random.seed(2026)
    dataset = P2ASingleRGBDataset(
        MANIFEST,
        fold=0,
        split="train",
        input_mode="full6_consistency",
        training=True,
        horizontal_flip_probability=1.0,
        max_cases=8,
        seed=2026,
    )
    item = dataset[0]
    assert bool(item["was_flipped"]) is True
    assert item["preset_names"] == ["neutral_front", "right", "left", "top", "dim_front", "bright_front"]

    loader = DataLoader(dataset, batch_size=2, num_workers=2, collate_fn=p2_a_collate)
    batch = next(iter(loader))
    assert batch["original_image"].shape == (2, 3, 224, 224)
    assert batch["relighted_images"].shape == (2, 6, 3, 224, 224)
    assert batch["label"].dtype == torch.long


def test_a3_v2_shared_resnet18_seven_view_forward_contract() -> None:
    model = P2ASingleRGBResNet18(pretrained=False, num_classes=2)
    original = torch.randn(2, 3, 224, 224)
    relighted = torch.randn(2, 6, 3, 224, 224)
    views = torch.cat([original.unsqueeze(1), relighted], dim=1)
    assert views.shape == (2, 7, 3, 224, 224)
    logits, features = model(views.reshape(14, 3, 224, 224))
    logits = logits.reshape(2, 7, 2)
    features = features.reshape(2, 7, 512)
    assert logits.shape == (2, 7, 2)
    assert features.shape == (2, 7, 512)


def test_a3_v2_trainer_smoke_resume_with_tiny_model(tmp_path: Path, monkeypatch) -> None:
    config = resolve_p2_a_config(CONFIG)
    config["output_root"] = str(tmp_path / "formal")
    config["smoke_output_root"] = str(tmp_path / "smoke")
    config["training"]["batch_size"] = 2
    config["training"]["gradient_accumulation_steps"] = 2
    config["training"]["max_epochs"] = 1
    config["smoke"]["train_cases"] = 8
    config["smoke"]["val_cases"] = 4
    config["smoke"]["max_epochs"] = 1
    trainer = P2AFoldTrainer(config, fold=0, device=torch.device("cpu"), smoke=True, max_batches=1, num_workers=0)
    monkeypatch.setattr(trainer, "_make_model", lambda: TinyP2Model().to(trainer.device))
    result = trainer.train(resume=False)
    assert result["status"] == "trained"
    history = pd.read_csv(trainer.history_path)
    assert history.loc[0, "original_sample_count"] == 2
    assert history.loc[0, "relighted_sample_count"] == 12
    assert history.loc[0, "warmup_factor"] == 0.2
    assert history.loc[0, "effective_case_batch_size"] == 4
    resumed = P2AFoldTrainer(config, fold=0, device=torch.device("cpu"), smoke=True, max_batches=1, num_workers=0)
    monkeypatch.setattr(resumed, "_make_model", lambda: TinyP2Model().to(resumed.device))
    assert resumed.train(resume=True)["epochs_completed"] == 1
