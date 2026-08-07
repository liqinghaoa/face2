"""Train the R3DPR full baseline with ColorJitter and OneCycleLR."""

from __future__ import annotations

import argparse
import json
import logging
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import sklearn  # noqa: F401  # Initialize OpenMP before torch on Windows.
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from torchvision import transforms

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from losses.classification_losses import build_criterion
from metrics.R3DPR.binary_classification_metrics import flatten_metrics
from models.R3DPR.resnet18_binary import build_r3dpr_resnet18_binary, count_parameters
from scripts.train.R3DPR.train_r3dpr_resnet18_binary_5fold import (
    apply_training_mode,
    build_dataset,
    build_optimizer,
    classifier_linear,
    compute_class_weights,
    configure_trainability,
    evaluate,
    make_loader,
    project_path,
    save_environment,
)
from utils.R3DPR.binary_data_audit import audit_r3dpr_binary_data
from utils.experiment_utils import configure_logging, load_yaml, save_yaml, set_random_seed


LOGGER = logging.getLogger("r3dpr_binary_full_cj05_onecycle")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--fold", type=int, action="append", dest="folds")
    parser.add_argument("--epochs", type=int, default=None, help="Only for isolated smoke runs")
    return parser.parse_args()


def build_r3dpr_transforms(
    split: str,
    image_height: int,
    image_width: int,
    mean: list[float],
    std: list[float],
    horizontal_flip: bool,
    brightness: float,
    contrast: float,
) -> transforms.Compose:
    if split not in {"train", "val"}:
        raise ValueError(f"split must be 'train' or 'val', got {split!r}")
    if image_height <= 0 or image_width <= 0:
        raise ValueError("image_height and image_width must be positive")
    steps: list[object] = [transforms.Resize((int(image_height), int(image_width)))]
    if split == "train":
        if horizontal_flip:
            steps.append(transforms.RandomHorizontalFlip())
        if brightness > 0.0 or contrast > 0.0:
            steps.append(transforms.ColorJitter(brightness=float(brightness), contrast=float(contrast)))
    steps.extend([transforms.ToTensor(), transforms.Normalize(mean=mean, std=std)])
    return transforms.Compose(steps)


def build_lr_scheduler(
    optimizer: torch.optim.AdamW,
    train_cfg: dict,
    steps_per_epoch: int,
) -> torch.optim.lr_scheduler.OneCycleLR | None:
    name = str(train_cfg.get("lr_scheduler", "none")).strip().lower()
    if name == "none":
        return None
    if name != "onecycle":
        raise ValueError(f"Unsupported lr_scheduler: {name!r}")
    if steps_per_epoch < 1:
        raise ValueError("steps_per_epoch must be at least 1")
    max_lr = float(train_cfg["onecycle_max_lr"])
    div_factor = float(train_cfg.get("onecycle_div_factor", 10.0))
    final_div_factor = float(train_cfg.get("onecycle_final_div_factor", 100.0))
    pct_start = float(train_cfg.get("onecycle_pct_start", 0.3))
    anneal_strategy = str(train_cfg.get("onecycle_anneal_strategy", "cos")).strip().lower()
    if max_lr <= 0.0:
        raise ValueError("onecycle_max_lr must be positive")
    if div_factor <= 0.0 or final_div_factor <= 0.0:
        raise ValueError("onecycle div factors must be positive")
    if not 0.0 <= pct_start <= 1.0:
        raise ValueError("onecycle_pct_start must be in [0, 1]")
    if anneal_strategy not in {"cos", "linear"}:
        raise ValueError("onecycle_anneal_strategy must be 'cos' or 'linear'")
    return torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=max_lr,
        epochs=int(train_cfg.get("max_epochs", train_cfg.get("epochs", 50))),
        steps_per_epoch=int(steps_per_epoch),
        pct_start=pct_start,
        anneal_strategy=anneal_strategy,
        cycle_momentum=bool(train_cfg.get("onecycle_cycle_momentum", False)),
        div_factor=div_factor,
        final_div_factor=final_div_factor,
    )


def save_training_parameters(fold_dir: Path, fold_info: dict) -> None:
    (fold_dir / "training_parameters.json").write_text(json.dumps(fold_info, ensure_ascii=False, indent=2), encoding="utf-8")


def run_fold(config: dict, table: pd.DataFrame, output_dir: Path, fold: int, device: torch.device) -> None:
    data, train_cfg = config["data"], config["train"]
    fold_column = data["fold_column"]
    train_table = table.loc[table[fold_column] != fold].copy()
    val_table = table.loc[table[fold_column] == fold].copy()
    fold_dir = output_dir / f"fold_{fold}"
    checkpoint_dir = fold_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    aug_cfg = config.get("augmentation", {})
    train_transform = build_r3dpr_transforms(
        "train",
        int(data["image_height"]),
        int(data["image_width"]),
        config["normalize"]["mean"],
        config["normalize"]["std"],
        bool(aug_cfg.get("horizontal_flip", True)),
        float(aug_cfg.get("brightness", 0.05)),
        float(aug_cfg.get("contrast", 0.05)),
    )
    eval_transform = build_r3dpr_transforms(
        "val",
        int(data["image_height"]),
        int(data["image_width"]),
        config["normalize"]["mean"],
        config["normalize"]["std"],
        False,
        0.0,
        0.0,
    )
    train_set = build_dataset(train_table, config, "train", train_transform)
    train_eval_set = build_dataset(train_table, config, "train", eval_transform)
    val_set = build_dataset(val_table, config, "val", eval_transform)

    batch_size = int(train_cfg["batch_size"])
    workers = int(train_cfg["num_workers"])
    seed = int(train_cfg["random_seed"])
    pin_memory = bool(train_cfg["pin_memory"])
    train_loader = make_loader(train_set, batch_size, True, workers, seed + fold, pin_memory)
    train_eval_loader = make_loader(train_eval_set, batch_size, False, workers, seed + 100 + fold, pin_memory)
    val_loader = make_loader(val_set, batch_size, False, workers, seed + 1000 + fold, pin_memory)

    strategy = str(config["model"].get("trainability_strategy", "full_finetune"))
    if strategy != "full_finetune":
        raise ValueError("This experiment is fixed to full_finetune")
    batchnorm_mode = str(train_cfg.get("batchnorm_mode", "train")).strip().lower()
    if batchnorm_mode != "train":
        raise ValueError("This experiment expects train.batchnorm_mode=train")
    model = build_r3dpr_resnet18_binary(config["model"]["pretrained"], dropout=config["model"].get("dropout")).to(device)
    configure_trainability(model, strategy)
    final_linear = classifier_linear(model)
    if final_linear.in_features != 512 or final_linear.out_features != 2:
        raise RuntimeError("R3DPR baseline must use Linear(512, 2) classifier head")

    weights = compute_class_weights(train_set.labels).to(device)
    loss_name = str(train_cfg.get("loss", "weighted_cross_entropy")).strip().lower()
    criterion = build_criterion(
        loss_name,
        class_weights=weights,
        device=device,
        smoothing=float(train_cfg.get("label_smoothing_alpha", 0.0)),
        num_classes=int(config["task"]["num_classes"]),
    )
    optimizer = build_optimizer(model, strategy, train_cfg)
    scheduler = build_lr_scheduler(optimizer, train_cfg, len(train_loader))
    max_epochs = int(train_cfg.get("max_epochs", train_cfg.get("epochs", 50)))

    counts = np.bincount(np.asarray(train_set.labels), minlength=2)
    fold_info = {
        "fold": fold,
        "trainability_strategy": strategy,
        "batchnorm_mode": batchnorm_mode,
        "backbone_batchnorm_eval": False,
        "dropout": config["model"].get("dropout"),
        "loss_name": loss_name,
        "lr_scheduler": str(train_cfg.get("lr_scheduler", "none")).strip().lower(),
        "max_epochs": max_epochs,
        "onecycle_max_lr": float(train_cfg.get("onecycle_max_lr", train_cfg.get("lr", 1e-4))),
        "onecycle_div_factor": float(train_cfg.get("onecycle_div_factor", 10.0)),
        "onecycle_final_div_factor": float(train_cfg.get("onecycle_final_div_factor", 100.0)),
        "onecycle_pct_start": float(train_cfg.get("onecycle_pct_start", 0.3)),
        "onecycle_anneal_strategy": str(train_cfg.get("onecycle_anneal_strategy", "cos")),
        "onecycle_cycle_momentum": bool(train_cfg.get("onecycle_cycle_momentum", False)),
        "train_control_count": int(counts[0]),
        "train_patient_count": int(counts[1]),
        "weight_control": float(weights[0]),
        "weight_patient": float(weights[1]),
        "augmentation_horizontal_flip": bool(aug_cfg.get("horizontal_flip", True)),
        "augmentation_brightness": float(aug_cfg.get("brightness", 0.05)),
        "augmentation_contrast": float(aug_cfg.get("contrast", 0.05)),
        **count_parameters(model),
    }
    save_training_parameters(fold_dir, fold_info)
    LOGGER.info("fold=%d counts=%s class_weights=%s", fold, counts.tolist(), weights.tolist())

    best_auc, best_epoch = float("-inf"), 0
    history: list[dict] = []
    started = time.perf_counter()
    for epoch in range(1, max_epochs + 1):
        apply_training_mode(model, strategy, batchnorm_mode)
        total_loss, seen = 0.0, 0
        lr_start = float(optimizer.param_groups[0]["lr"])
        lr_peak = lr_start
        for batch in train_loader:
            image = batch["image"].to(device, non_blocking=True)
            label = batch["label"].to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(image), label)
            loss.backward()
            optimizer.step()
            if scheduler is not None:
                scheduler.step()
            current_lr = float(optimizer.param_groups[0]["lr"])
            lr_peak = max(lr_peak, current_lr)
            total_loss += float(loss.detach()) * len(label)
            seen += len(label)
        lr_end = float(optimizer.param_groups[0]["lr"])
        train_metrics, _, _ = evaluate(model, train_eval_loader, device, criterion, False)
        val_metrics, val_loss, _ = evaluate(model, val_loader, device, criterion, False)
        row = {
            "epoch": epoch,
            "learning_rate": lr_start,
            "lr_peak": lr_peak,
            "lr_end": lr_end,
            "train_loss": total_loss / seen,
            **{f"train_{key}": value for key, value in flatten_metrics(train_metrics).items()},
            "val_loss": val_loss,
            **{f"val_{key}": value for key, value in flatten_metrics(val_metrics).items()},
        }
        history.append(row)
        auc = float(val_metrics["macro_auc"])
        if auc > best_auc:
            best_auc, best_epoch = auc, epoch
            torch.save(
                {"model_state_dict": model.state_dict(), "epoch": epoch, "best_macro_auc": auc, "config": config, "fold_info": fold_info},
                checkpoint_dir / "best_macro_auc.pth",
            )
        torch.save(
            {"model_state_dict": model.state_dict(), "epoch": epoch, "best_macro_auc": best_auc, "config": config, "fold_info": fold_info},
            checkpoint_dir / "last.pth",
        )
        LOGGER.info(
            "fold=%d epoch=%d lr=%.2e train_loss=%.5f val_loss=%.5f val_macro_auc=%.4f best=%.4f",
            fold,
            epoch,
            lr_start,
            row["train_loss"],
            val_loss,
            auc,
            best_auc,
        )
    pd.DataFrame(history).to_csv(fold_dir / "training_history.csv", index=False, encoding="utf-8-sig")
    checkpoint_path = checkpoint_dir / "best_macro_auc.pth"
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    val_metrics, _, predictions = evaluate(model, val_loader, device, criterion, True, int(checkpoint["epoch"]), checkpoint_path)
    assert predictions is not None
    predictions.to_csv(fold_dir / "val_predictions.csv", index=False, encoding="utf-8-sig")
    metric_row = {
        "fold": fold,
        "best_epoch": int(checkpoint["epoch"]),
        "completed_epoch": max_epochs,
        "training_seconds": time.perf_counter() - started,
        **fold_info,
        **flatten_metrics(val_metrics),
    }
    pd.DataFrame([metric_row]).to_csv(fold_dir / "metrics.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(val_metrics["confusion_matrix"], index=["control", "patient"], columns=["control", "patient"]).to_csv(
        fold_dir / "confusion_matrix.csv", index_label="true\\pred", encoding="utf-8-sig"
    )


def main() -> Path:
    args = parse_args()
    config = load_yaml(args.config)
    if args.epochs is not None:
        config["train"]["max_epochs"] = int(args.epochs)
    if config["task"]["type"] != "binary_control_vs_patient" or int(config["task"]["num_classes"]) != 2:
        raise ValueError("R3DPR requires the fixed Control-versus-Patient binary task")
    if config["model"]["backbone"] != "resnet18" or int(config["model"]["num_classes"]) != 2:
        raise ValueError("R3DPR baseline requires ResNet18 with two output logits")
    if str(config["model"].get("trainability_strategy", "full_finetune")) != "full_finetune":
        raise ValueError("This experiment is fixed to full_finetune")
    if str(config["train"].get("batchnorm_mode", "train")).strip().lower() != "train":
        raise ValueError("This experiment expects train.batchnorm_mode=train")
    scheduler_name = str(config["train"].get("lr_scheduler", "none")).strip().lower()
    if scheduler_name != "onecycle":
        raise ValueError("This experiment requires train.lr_scheduler=onecycle")
    loss_name = str(config["train"].get("loss", "weighted_cross_entropy")).strip().lower()
    if loss_name != "weighted_cross_entropy":
        raise ValueError("This experiment requires weighted_cross_entropy")
    if config["train"]["optimizer"].lower() != "adamw" or config["train"]["monitor_metric"] != "macro_auc":
        raise ValueError("R3DPR baseline requires AdamW and validation macro_auc selection")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the R3DPR baseline; CPU fallback is disabled")
    output_dir = project_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    configure_logging(output_dir / "experiment.log")
    save_yaml(config, output_dir / "config_snapshot.yaml")
    save_environment(output_dir)
    table = audit_r3dpr_binary_data(config, output_dir, PROJECT_ROOT)
    set_random_seed(int(config["train"]["random_seed"]))
    device = torch.device("cuda")
    folds = args.folds if args.folds is not None else list(range(int(config["data"]["n_folds"])))
    if set(folds).difference(range(int(config["data"]["n_folds"]))):
        raise ValueError(f"Invalid requested folds: {folds}")
    for fold in folds:
        run_fold(config, table, output_dir, fold, device)
    (output_dir / "run_finished_at.txt").write_text(datetime.now().isoformat(timespec="seconds"), encoding="utf-8")
    print(f"EXPERIMENT_DIR={output_dir}")
    return output_dir


if __name__ == "__main__":
    main()
