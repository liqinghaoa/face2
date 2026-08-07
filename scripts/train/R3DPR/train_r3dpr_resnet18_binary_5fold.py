"""Train the independent R3DPR direct-label ResNet18 binary baseline."""

from __future__ import annotations

import argparse
import json
import logging
import platform
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

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from datasets.R3DPR.binary_face_dataset import R3DPRBinaryFaceDataset, build_r3dpr_transforms
from losses.classification_losses import build_criterion
from metrics.R3DPR.binary_classification_metrics import compute_binary_metrics, flatten_metrics
from models.R3DPR.resnet18_binary import build_r3dpr_resnet18_binary, count_parameters
from utils.R3DPR.binary_data_audit import audit_r3dpr_binary_data
from utils.experiment_utils import configure_logging, load_yaml, save_yaml, seed_worker, set_random_seed


LOGGER = logging.getLogger("r3dpr_binary")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--fold", type=int, action="append", dest="folds")
    parser.add_argument("--epochs", type=int, default=None, help="Only for isolated smoke runs")
    return parser.parse_args()


def project_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def make_loader(dataset: R3DPRBinaryFaceDataset, batch_size: int, shuffle: bool, workers: int, seed: int, pin_memory: bool) -> DataLoader:
    generator = torch.Generator()
    generator.manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=workers,
        pin_memory=pin_memory,
        persistent_workers=workers > 0,
        worker_init_fn=seed_worker,
        generator=generator,
    )


def compute_class_weights(labels: list[int]) -> torch.Tensor:
    counts = torch.bincount(torch.tensor(labels, dtype=torch.long), minlength=2).to(torch.float32)
    if (counts == 0).any():
        raise ValueError(f"Cannot calculate weighted CE with missing class: counts={counts.tolist()}")
    return counts.sum() / (2.0 * counts)


def configure_trainability(model: torch.nn.Module, strategy: str, stage: str = "head_only") -> None:
    normalized = str(strategy).strip().lower()
    if normalized == "full_finetune":
        for parameter in model.parameters():
            parameter.requires_grad = True
        return
    if normalized == "head_only":
        for parameter in model.parameters():
            parameter.requires_grad = False
        for parameter in model.fc.parameters():
            parameter.requires_grad = True
        return
    if normalized == "staged_layer4":
        for parameter in model.parameters():
            parameter.requires_grad = False
        for parameter in model.fc.parameters():
            parameter.requires_grad = True
        if stage == "layer4_plus_fc":
            for parameter in model.layer4.parameters():
                parameter.requires_grad = True
        elif stage != "head_only":
            raise ValueError(f"Unsupported staged_layer4 stage: {stage!r}")
        return
    raise ValueError(f"Unsupported R3DPR trainability_strategy: {strategy!r}")


def apply_training_mode(model: torch.nn.Module, strategy: str, batchnorm_mode: str = "train") -> None:
    model.train()
    freeze_batchnorm = (
        str(strategy).strip().lower() in {"head_only", "staged_layer4"}
        or str(batchnorm_mode).strip().lower() == "eval"
    )
    if freeze_batchnorm:
        for module in model.modules():
            if isinstance(module, (torch.nn.BatchNorm1d, torch.nn.BatchNorm2d, torch.nn.BatchNorm3d, torch.nn.SyncBatchNorm)):
                module.eval()


def classifier_linear(model: torch.nn.Module) -> torch.nn.Linear:
    if isinstance(model.fc, torch.nn.Linear):
        return model.fc
    for module in reversed(list(model.fc.modules())):
        if isinstance(module, torch.nn.Linear):
            return module
    raise RuntimeError("R3DPR classifier head has no Linear layer")


def build_optimizer(model: torch.nn.Module, strategy: str, train_cfg: dict) -> torch.optim.AdamW:
    normalized = str(strategy).strip().lower()
    weight_decay = float(train_cfg["weight_decay"])
    if normalized == "staged_layer4":
        return torch.optim.AdamW(
            [
                {"params": list(model.fc.parameters()), "lr": float(train_cfg["head_warmup_lr"]), "group_name": "fc"},
                {"params": list(model.layer4.parameters()), "lr": float(train_cfg["layer4_lr"]), "group_name": "layer4"},
            ],
            weight_decay=weight_decay,
        )
    trainable_parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if not trainable_parameters:
        raise RuntimeError("R3DPR training strategy left no trainable parameters")
    return torch.optim.AdamW(trainable_parameters, lr=float(train_cfg["lr"]), weight_decay=weight_decay)


def build_lr_scheduler(optimizer: torch.optim.AdamW, train_cfg: dict) -> torch.optim.lr_scheduler.LRScheduler | None:
    name = str(train_cfg.get("lr_scheduler", "none")).strip().lower()
    if name == "none":
        return None
    if name == "cosine":
        t_max = int(train_cfg.get("lr_scheduler_t_max", train_cfg["epochs"]))
        eta_min = float(train_cfg.get("lr_scheduler_eta_min", 0.0))
        if t_max < 1:
            raise ValueError("lr_scheduler_t_max must be at least 1")
        if eta_min < 0.0:
            raise ValueError("lr_scheduler_eta_min must be non-negative")
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=t_max, eta_min=eta_min
        )
    raise ValueError(f"Unsupported lr_scheduler: {name!r}")


def transition_to_layer4(model: torch.nn.Module, optimizer: torch.optim.AdamW, train_cfg: dict) -> None:
    configure_trainability(model, "staged_layer4", stage="layer4_plus_fc")
    for group in optimizer.param_groups:
        if group.get("group_name") == "fc":
            group["lr"] = float(train_cfg["fc_lr"])


def save_environment(output_dir: Path) -> None:
    git = subprocess.run(["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT, capture_output=True, text=True, check=False)
    payload = {
        "started_at": datetime.now().isoformat(timespec="seconds"),
        "python": sys.version,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "torchvision": __import__("torchvision").__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "git_head": git.stdout.strip() or None,
    }
    (output_dir / "environment.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


@torch.no_grad()
def evaluate(
    model: torch.nn.Module,
    data_loader: DataLoader,
    device: torch.device,
    criterion: torch.nn.Module,
    with_predictions: bool,
    selected_epoch: int | None = None,
    checkpoint_path: Path | None = None,
) -> tuple[dict, float, pd.DataFrame | None]:
    model.eval()
    labels_all: list[int] = []
    probabilities: list[np.ndarray] = []
    rows: list[dict] = []
    total_loss = 0.0
    sample_count = 0
    for batch in data_loader:
        image = batch["image"].to(device, non_blocking=True)
        label = batch["label"].to(device, non_blocking=True)
        logits = model(image)
        loss = criterion(logits, label)
        probs = torch.softmax(logits, dim=1).cpu().numpy()
        labels = label.cpu().numpy().astype(int)
        logits_np = logits.cpu().numpy()
        labels_all.extend(labels.tolist())
        probabilities.append(probs)
        total_loss += float(loss) * len(labels)
        sample_count += len(labels)
        if with_predictions:
            prediction = probs.argmax(axis=1)
            for index, target in enumerate(labels):
                rows.append({
                    "sample_id": str(batch["sample_id"][index]),
                    "patient_group_id": str(batch["patient_group_id"][index]),
                    "sex": str(batch["sex"][index]),
                    "fold": int(batch["fold"][index]),
                    "binary_label": int(target),
                    "logit_control": float(logits_np[index, 0]),
                    "logit_patient": float(logits_np[index, 1]),
                    "prob_control": float(probs[index, 0]),
                    "prob_patient": float(probs[index, 1]),
                    "pred_class": int(prediction[index]),
                    "is_correct": int(prediction[index] == target),
                    "image_path": str(batch["image_path"][index]),
                    "selected_epoch": selected_epoch,
                    "checkpoint_path": str(checkpoint_path) if checkpoint_path is not None else None,
                })
    return compute_binary_metrics(np.asarray(labels_all), np.concatenate(probabilities)), total_loss / sample_count, pd.DataFrame(rows) if with_predictions else None


def build_dataset(table: pd.DataFrame, config: dict, split: str, transform) -> R3DPRBinaryFaceDataset:
    data = config["data"]
    return R3DPRBinaryFaceDataset(
        table=table,
        image_root=project_path(data["image_root"]),
        image_filename_template=str(data["image_filename_template"]),
        transform=transform,
        id_column=data["id_column"],
        group_id_column=data["group_id_column"],
        fold_column=data["fold_column"],
        label_column=data["label_column"],
        sex_column=data["sex_column"],
    )


def run_fold(config: dict, table: pd.DataFrame, output_dir: Path, fold: int, device: torch.device) -> None:
    data, train_cfg = config["data"], config["train"]
    fold_column = data["fold_column"]
    label_column = data["label_column"]
    train_table = table.loc[table[fold_column] != fold].copy()
    val_table = table.loc[table[fold_column] == fold].copy()
    fold_dir = output_dir / f"fold_{fold}"
    checkpoint_dir = fold_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    train_transform = build_r3dpr_transforms("train", int(data["image_height"]), int(data["image_width"]), config["normalize"]["mean"], config["normalize"]["std"], bool(config["augmentation"]["horizontal_flip"]))
    eval_transform = build_r3dpr_transforms("val", int(data["image_height"]), int(data["image_width"]), config["normalize"]["mean"], config["normalize"]["std"], False)
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
    batchnorm_mode = str(train_cfg.get("batchnorm_mode", "train")).strip().lower()
    model = build_r3dpr_resnet18_binary(config["model"]["pretrained"], dropout=config["model"].get("dropout")).to(device)
    configure_trainability(model, strategy)
    final_linear = classifier_linear(model)
    if final_linear.in_features != 512 or final_linear.out_features != 2:
        raise RuntimeError("R3DPR baseline must use Linear(512, 2) classifier head")
    weights = compute_class_weights(train_set.labels).to(device)
    loss_name = str(train_cfg.get("loss", "weighted_cross_entropy")).strip().lower()
    label_smoothing_alpha = float(train_cfg.get("label_smoothing_alpha", 0.0))
    criterion = build_criterion(
        loss_name,
        class_weights=weights,
        device=device,
        smoothing=label_smoothing_alpha,
        num_classes=int(config["task"]["num_classes"]),
    )
    optimizer = build_optimizer(model, strategy, train_cfg)
    scheduler = build_lr_scheduler(optimizer, train_cfg)
    lr_scheduler_name = str(train_cfg.get("lr_scheduler", "none")).strip().lower()
    counts = np.bincount(np.asarray(train_set.labels), minlength=2)
    stage_trainable_params = None
    if strategy == "staged_layer4":
        stage_trainable_params = int(
            sum(parameter.numel() for parameter in model.layer4.parameters())
            + sum(parameter.numel() for parameter in model.fc.parameters())
        )
    fold_info = {
        "fold": fold,
        "trainability_strategy": strategy,
        "batchnorm_mode": batchnorm_mode,
        "backbone_batchnorm_eval": batchnorm_mode == "eval" or strategy in {"head_only", "staged_layer4"},
        "dropout": config["model"].get("dropout"),
        "loss_name": loss_name,
        "label_smoothing_alpha": label_smoothing_alpha,
        "lr_scheduler": lr_scheduler_name,
        "lr_scheduler_t_max": int(train_cfg.get("lr_scheduler_t_max", train_cfg["epochs"])) if lr_scheduler_name == "cosine" else None,
        "lr_scheduler_eta_min": float(train_cfg.get("lr_scheduler_eta_min", 0.0)) if lr_scheduler_name == "cosine" else None,
        "head_only_epochs": int(train_cfg.get("head_warmup_epochs", 0)) if strategy == "staged_layer4" else None,
        "layer4_plus_fc_trainable_params": stage_trainable_params,
        "train_control_count": int(counts[0]),
        "train_patient_count": int(counts[1]),
        "weight_control": float(weights[0]),
        "weight_patient": float(weights[1]),
        **count_parameters(model),
    }
    (fold_dir / "training_parameters.json").write_text(json.dumps(fold_info, ensure_ascii=False, indent=2), encoding="utf-8")
    LOGGER.info("fold=%d counts=%s class_weights=%s", fold, counts.tolist(), weights.tolist())

    best_auc, best_epoch, patience = float("-inf"), 0, 0
    history: list[dict] = []
    started = time.perf_counter()
    for epoch in range(1, int(train_cfg["epochs"]) + 1):
        phase = "head_only"
        if strategy == "staged_layer4" and epoch == int(train_cfg["head_warmup_epochs"]) + 1:
            transition_to_layer4(model, optimizer, train_cfg)
            patience = 0
            LOGGER.info("fold=%d stage transition: unfreezing layer4 + fc", fold)
        if strategy == "staged_layer4" and epoch > int(train_cfg["head_warmup_epochs"]):
            phase = "layer4_plus_fc"
        apply_training_mode(model, strategy, batchnorm_mode)
        total_loss, seen = 0.0, 0
        for batch in train_loader:
            image = batch["image"].to(device, non_blocking=True)
            label = batch["label"].to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(image), label)
            loss.backward()
            optimizer.step()
            total_loss += float(loss.detach()) * len(label)
            seen += len(label)
        train_metrics, _, _ = evaluate(model, train_eval_loader, device, criterion, False)
        val_metrics, val_loss, _ = evaluate(model, val_loader, device, criterion, False)
        row = {
            "epoch": epoch,
            "phase": phase,
            "learning_rate": float(optimizer.param_groups[0]["lr"]),
            "train_loss": total_loss / seen,
            **{f"train_{key}": value for key, value in flatten_metrics(train_metrics).items()},
            "val_loss": val_loss,
            **{f"val_{key}": value for key, value in flatten_metrics(val_metrics).items()},
        }
        history.append(row)
        auc = float(val_metrics["macro_auc"])
        if auc > best_auc:
            best_auc, best_epoch, patience = auc, epoch, 0
            torch.save({"model_state_dict": model.state_dict(), "epoch": epoch, "best_macro_auc": auc, "config": config, "fold_info": fold_info}, checkpoint_dir / "best_macro_auc.pth")
        else:
            patience += 1
        torch.save({"model_state_dict": model.state_dict(), "epoch": epoch, "best_macro_auc": best_auc, "config": config, "fold_info": fold_info}, checkpoint_dir / "last.pth")
        LOGGER.info("fold=%d epoch=%d lr=%.2e train_loss=%.5f val_loss=%.5f val_macro_auc=%.4f best=%.4f patience=%d/%d", fold, epoch, row["learning_rate"], row["train_loss"], val_loss, auc, best_auc, patience, int(train_cfg["early_stopping_patience"]))
        if scheduler is not None:
            scheduler.step()
        if patience >= int(train_cfg["early_stopping_patience"]):
            break
    pd.DataFrame(history).to_csv(fold_dir / "training_history.csv", index=False, encoding="utf-8-sig")
    checkpoint_path = checkpoint_dir / "best_macro_auc.pth"
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    val_metrics, _, predictions = evaluate(model, val_loader, device, criterion, True, int(checkpoint["epoch"]), checkpoint_path)
    assert predictions is not None
    predictions.to_csv(fold_dir / "val_predictions.csv", index=False, encoding="utf-8-sig")
    metric_row = {"fold": fold, "best_epoch": int(checkpoint["epoch"]), "training_seconds": time.perf_counter() - started, **fold_info, **flatten_metrics(val_metrics)}
    pd.DataFrame([metric_row]).to_csv(fold_dir / "metrics.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(val_metrics["confusion_matrix"], index=["control", "patient"], columns=["control", "patient"]).to_csv(fold_dir / "confusion_matrix.csv", index_label="true\\pred", encoding="utf-8-sig")


def main() -> Path:
    args = parse_args()
    config = load_yaml(args.config)
    if args.epochs is not None:
        config["train"]["epochs"] = int(args.epochs)
    if config["task"]["type"] != "binary_control_vs_patient" or int(config["task"]["num_classes"]) != 2:
        raise ValueError("R3DPR requires the fixed Control-versus-Patient binary task")
    if config["model"]["backbone"] != "resnet18" or int(config["model"]["num_classes"]) != 2:
        raise ValueError("R3DPR baseline requires ResNet18 with two output logits")
    strategy = config["model"].get("trainability_strategy")
    if strategy not in {"full_finetune", "head_only", "staged_layer4"}:
        raise ValueError("R3DPR trainability_strategy must be full_finetune, head_only, or staged_layer4")
    if strategy == "staged_layer4":
        if int(config["train"].get("head_warmup_epochs", 0)) < 1:
            raise ValueError("staged_layer4 requires train.head_warmup_epochs >= 1")
        for key in ("head_warmup_lr", "fc_lr", "layer4_lr"):
            if float(config["train"].get(key, 0.0)) <= 0.0:
                raise ValueError(f"staged_layer4 requires a positive train.{key}")
    if str(config["train"].get("batchnorm_mode", "train")).strip().lower() not in {"train", "eval"}:
        raise ValueError("train.batchnorm_mode must be train or eval")
    scheduler_name = str(config["train"].get("lr_scheduler", "none")).strip().lower()
    if scheduler_name not in {"none", "cosine"}:
        raise ValueError("train.lr_scheduler must be none or cosine")
    if scheduler_name == "cosine":
        if int(config["train"].get("lr_scheduler_t_max", config["train"]["epochs"])) < 1:
            raise ValueError("train.lr_scheduler_t_max must be at least 1")
        if float(config["train"].get("lr_scheduler_eta_min", 0.0)) < 0.0:
            raise ValueError("train.lr_scheduler_eta_min must be non-negative")
    loss_name = str(config["train"].get("loss", "weighted_cross_entropy")).strip().lower()
    if loss_name not in {"weighted_cross_entropy", "weighted_ce_label_smoothing"}:
        raise ValueError("train.loss must be weighted_cross_entropy or weighted_ce_label_smoothing")
    label_smoothing_alpha = float(config["train"].get("label_smoothing_alpha", 0.0))
    if loss_name == "weighted_cross_entropy" and label_smoothing_alpha != 0.0:
        raise ValueError("train.label_smoothing_alpha must be 0 when loss is weighted_cross_entropy")
    if loss_name == "weighted_ce_label_smoothing" and not (0.0 < label_smoothing_alpha < 1.0):
        raise ValueError("weighted_ce_label_smoothing requires 0 < train.label_smoothing_alpha < 1")
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
