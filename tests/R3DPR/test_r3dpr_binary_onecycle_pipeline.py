from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest
import torch
from torchvision import transforms

from scripts.evaluate.R3DPR.plot_r3dpr_training_diagnostics_full_cj05_onecycle import plot_experiment
from scripts.train.R3DPR.train_r3dpr_resnet18_binary_5fold_full_cj05_onecycle import (
    build_lr_scheduler,
    build_r3dpr_transforms,
)


def test_onecycle_train_transform_adds_brightness_and_contrast_jitter() -> None:
    transform = build_r3dpr_transforms(
        "train",
        320,
        256,
        [0.485, 0.456, 0.406],
        [0.229, 0.224, 0.225],
        True,
        0.05,
        0.05,
    )
    assert isinstance(transform, transforms.Compose)
    assert any(isinstance(step, transforms.ColorJitter) for step in transform.transforms)
    jitter = next(step for step in transform.transforms if isinstance(step, transforms.ColorJitter))
    assert list(jitter.brightness) == pytest.approx([0.95, 1.05])
    assert list(jitter.contrast) == pytest.approx([0.95, 1.05])


def test_onecycle_scheduler_is_batchwise_and_optional() -> None:
    parameter = torch.nn.Parameter(torch.tensor([1.0]))
    optimizer = torch.optim.AdamW([parameter], lr=1e-4)
    scheduler = build_lr_scheduler(
        optimizer,
        {
            "lr_scheduler": "onecycle",
            "onecycle_max_lr": 1e-4,
            "onecycle_div_factor": 10.0,
            "onecycle_final_div_factor": 100.0,
            "onecycle_pct_start": 0.3,
            "onecycle_anneal_strategy": "cos",
            "onecycle_cycle_momentum": False,
            "max_epochs": 2,
        },
        steps_per_epoch=2,
    )
    assert isinstance(scheduler, torch.optim.lr_scheduler.OneCycleLR)
    initial = float(optimizer.param_groups[0]["lr"])
    optimizer.step()
    scheduler.step()
    assert float(optimizer.param_groups[0]["lr"]) != pytest.approx(initial)


def test_training_diagnostics_still_write_png_only(tmp_path: Path) -> None:
    experiment_dir = tmp_path / "experiment"
    experiment_dir.mkdir()
    fold_metrics = []
    for fold in range(5):
        fold_dir = experiment_dir / f"fold_{fold}"
        fold_dir.mkdir()
        history = pd.DataFrame(
            {
                "epoch": [1, 2, 3],
                "train_loss": [0.70, 0.55, 0.40],
                "val_loss": [0.68, 0.60, 0.66],
                "train_macro_auc": [0.70, 0.86, 0.95],
                "val_macro_auc": [0.72, 0.84, 0.79],
            }
        )
        history.to_csv(fold_dir / "training_history.csv", index=False)
        fold_metrics.append({"fold": fold, "best_epoch": 2, "macro_auc": 0.84})
    pd.DataFrame(fold_metrics).to_csv(experiment_dir / "fold_metrics.csv", index=False)

    output_dir = plot_experiment(experiment_dir)
    assert len(list(output_dir.glob("fold_*_training_diagnostics.png"))) == 5
    assert not list(output_dir.glob("fold_*_training_diagnostics.svg"))
    assert not list(output_dir.glob("fold_*_training_diagnostics.pdf"))
    summary = pd.read_csv(output_dir / "training_diagnostic_summary.csv")
    assert summary["auc_drop_after_best"].tolist() == pytest.approx([0.05] * 5)
