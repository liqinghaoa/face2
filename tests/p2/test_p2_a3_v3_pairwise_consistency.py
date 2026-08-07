from __future__ import annotations

import random
from pathlib import Path

import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader

from p2_counterfactual.a3_v3_assets import PAIR_MANIFEST_PATH, build_a3_v3_assets, pair_cycle_order
from p2_counterfactual.assets import PRESET_NAMES
from p2_counterfactual.config import resolve_p2_a_config
from p2_counterfactual.dataset import P2ASingleRGBDataset, load_p2_manifest, p2_a_collate
from p2_counterfactual.model import P2ASingleRGBResNet18
from p2_counterfactual.trainer import P2AFoldTrainer

CONFIG = Path("config/p2/p2_a/p2_a3_v3_r3dpr_pairwise_consistency_meanbg.yaml")


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


def _ensure_assets() -> Path:
    if not PAIR_MANIFEST_PATH.is_file():
        build_a3_v3_assets()
    return PAIR_MANIFEST_PATH


def test_a3_v3_pair_manifest_has_3000_balanced_pairs() -> None:
    path = _ensure_assets()
    pairs = pd.read_csv(path, dtype={"case_id": str, "patient_group_id": str})
    assert len(pairs) == 3000
    assert pairs["case_id"].nunique() == 500
    assert pairs.groupby("case_id").size().eq(6).all()
    assert pairs.groupby("case_id")["preset_id"].nunique().eq(6).all()
    assert pairs.groupby("case_id")["fold_id"].nunique().eq(1).all()
    assert pairs.groupby("patient_group_id")["fold_id"].nunique().eq(1).all()
    assert set(pairs["preset_id"]) == set(PRESET_NAMES)
    assert not pairs.astype(str).apply(lambda col: col.str.contains("P2_DetailPreserving_Relighting500_v2|relighted_images", regex=True)).any().any()

    case_manifest = load_p2_manifest(path)
    assert len(case_manifest) == 500
    assert all(f"relight_{preset}_path" in case_manifest.columns for preset in PRESET_NAMES)


def test_a3_v3_pair_cycle_covers_each_case_once_per_epoch_and_all_presets_in_six_epochs() -> None:
    config = resolve_p2_a_config(CONFIG)
    dataset = P2ASingleRGBDataset(
        config["manifest_path"],
        fold=0,
        split="train",
        input_mode="pairwise_consistency",
        training=True,
        horizontal_flip_probability=0.0,
        max_cases=12,
        seed=2026,
        project_root=config["project_root"],
        path_resolution=config["path_resolution"],
    )
    assert len(dataset) == 12
    by_case: dict[str, list[str]] = {}
    for epoch in range(1, 7):
        dataset.set_epoch(epoch)
        seen_cases = []
        for index in range(len(dataset)):
            item = dataset[index]
            seen_cases.append(item["case_id"])
            by_case.setdefault(item["case_id"], []).append(item["preset_name"])
        assert len(seen_cases) == len(set(seen_cases)) == len(dataset)
    assert all(set(values) == set(PRESET_NAMES) for values in by_case.values())

    case_id = next(iter(by_case))
    assert pair_cycle_order(case_id, seed=2026, cycle_index=0) == pair_cycle_order(case_id, seed=2026, cycle_index=0)


def test_a3_v3_dataset_item_and_batch_shapes_are_pairwise() -> None:
    config = resolve_p2_a_config(CONFIG)
    random.seed(2026)
    dataset = P2ASingleRGBDataset(
        config["manifest_path"],
        fold=0,
        split="train",
        input_mode="pairwise_consistency",
        training=True,
        horizontal_flip_probability=1.0,
        max_cases=8,
        seed=2026,
        project_root=config["project_root"],
        path_resolution=config["path_resolution"],
    )
    item = dataset[0]
    assert item["original_image"].shape == (3, 224, 224)
    assert item["relighted_image"].shape == (3, 224, 224)
    assert "relighted_images" not in item
    assert "counterfactual_image" not in item
    assert item["preset_index"] in range(6)
    assert bool(item["was_flipped"]) is True
    assert item["case_id"] in item["pair_id"]

    loader = DataLoader(dataset, batch_size=4, num_workers=2, collate_fn=p2_a_collate)
    batch = next(iter(loader))
    assert batch["original_image"].shape == (4, 3, 224, 224)
    assert batch["relighted_image"].shape == (4, 3, 224, 224)
    all_images = torch.cat([batch["original_image"], batch["relighted_image"]], dim=0)
    assert all_images.shape == (8, 3, 224, 224)
    assert batch["label"].dtype == torch.long
    assert all(str(case_id) in str(pair_id) for case_id, pair_id in zip(batch["case_id"], batch["pair_id"]))


def test_a3_v3_shared_resnet18_pair_forward_contract() -> None:
    model = P2ASingleRGBResNet18(pretrained=False, num_classes=2)
    original = torch.randn(3, 3, 224, 224)
    relighted = torch.randn(3, 3, 224, 224)
    all_images = torch.cat([original, relighted], dim=0)
    logits, features = model(all_images)
    assert logits.shape == (6, 2)
    assert features.shape == (6, 512)
    assert logits[:3].shape == logits[3:].shape == (3, 2)
    assert features[:3].shape == features[3:].shape == (3, 512)


def test_a3_v3_trainer_tiny_smoke_resume_and_coverage_log(tmp_path: Path, monkeypatch) -> None:
    config = resolve_p2_a_config(CONFIG)
    config["output_root"] = str(tmp_path / "formal")
    config["smoke_output_root"] = str(tmp_path / "smoke")
    config["training"]["batch_size"] = 4
    config["training"]["gradient_accumulation_steps"] = 1
    config["smoke"]["max_epochs"] = 2
    config["smoke"]["train_cases"] = 12
    config["smoke"]["val_cases"] = 4
    trainer = P2AFoldTrainer(config, fold=0, device=torch.device("cpu"), smoke=True, max_batches=1, num_workers=0)
    monkeypatch.setattr(trainer, "_make_model", lambda: TinyP2Model().to(trainer.device))
    result = trainer.train(resume=False)
    assert result["status"] == "trained"
    history = pd.read_csv(trainer.history_path)
    assert history.loc[0, "original_sample_count"] == 4
    assert history.loc[0, "relighted_sample_count"] == 4
    assert history.loc[0, "images_per_step"] == 8
    assert history.loc[0, "warmup_factor"] == 0.2
    coverage = pd.read_csv(trainer.output_dir / "pair_coverage_log.csv")
    assert list(coverage["epoch"]) == [1, 2]
    assert coverage["duplicate_case_within_epoch"].eq(False).all()
    resumed = P2AFoldTrainer(config, fold=0, device=torch.device("cpu"), smoke=True, max_batches=1, num_workers=0)
    monkeypatch.setattr(resumed, "_make_model", lambda: TinyP2Model().to(resumed.device))
    assert resumed.train(resume=True)["epochs_completed"] == 2
