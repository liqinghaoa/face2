from __future__ import annotations

from pathlib import Path

import torch
from torch import nn

from p2_counterfactual.config import resolve_p2_a_config
from p2_counterfactual.trainer import P2AFoldTrainer


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


def test_trainer_single_batch_checkpoint_and_resume(tmp_path: Path, monkeypatch) -> None:
    config = resolve_p2_a_config("config/p2/p2_a/p2_a2_relighting.yaml")
    config["output_root"] = str(tmp_path / "formal")
    config["smoke_output_root"] = str(tmp_path / "smoke")
    config["training"]["batch_size"] = 4
    config["training"]["max_epochs"] = 1
    config["smoke"]["train_cases"] = 8
    config["smoke"]["val_cases"] = 6
    trainer = P2AFoldTrainer(config, fold=0, device=torch.device("cpu"), smoke=True, max_batches=1, num_workers=0)
    monkeypatch.setattr(trainer, "_make_model", lambda: TinyP2Model().to(trainer.device))
    result = trainer.train(resume=False)
    assert result["status"] == "trained"
    assert (trainer.checkpoint_dir / "last.pth").is_file()
    assert (trainer.checkpoint_dir / "best_macro_auc.pth").is_file()
    assert (trainer.output_dir / "training_history.csv").is_file()
    resumed = P2AFoldTrainer(config, fold=0, device=torch.device("cpu"), smoke=True, max_batches=1, num_workers=0)
    monkeypatch.setattr(resumed, "_make_model", lambda: TinyP2Model().to(resumed.device))
    resumed_result = resumed.train(resume=True)
    assert resumed_result["epochs_completed"] == 1


def test_a3_trainer_validation_dataset_uses_original_rgb(tmp_path: Path) -> None:
    config = resolve_p2_a_config("config/p2/p2_a/p2_a3_full_consistency.yaml")
    config["output_root"] = str(tmp_path / "formal")
    config["smoke_output_root"] = str(tmp_path / "smoke")
    config["smoke"]["val_cases"] = 4
    trainer = P2AFoldTrainer(config, fold=0, device=torch.device("cpu"), smoke=True, max_batches=1, num_workers=0)
    sample = trainer._dataset("val")[0]
    assert "image" in sample
    assert "original_image" not in sample
    assert sample["source_type"] == "original"
