"""Independent E0B Control-versus-Patient five-fold experiment entry point."""

from __future__ import annotations

import argparse
import json
import logging
import os
import platform
import sys
import time
from datetime import datetime
from pathlib import Path

import sklearn  # noqa: F401  # initialize OpenMP before torch on Windows
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from datasets.control_patient_binary_dataset import ControlPatientFaceDataset
from datasets.nyha_3class_face_dataset import build_transforms
from losses.classification_losses import compute_class_weights
from metrics.binary_classification_metrics import compute_binary_metrics, flatten_metrics
from models.nyha_backbone_factory import build_nyha_classification_model, count_parameters
from utils.e0b_binary_audit import audit_fixed_splits
from utils.experiment_utils import configure_logging, load_yaml, save_yaml, seed_worker, set_random_seed
from utils.resnet18_anti_overfit import (
    apply_train_mode,
    build_optimizer,
    build_trainability_audit,
    classifier_module,
    configure_trainability,
    normalize_strategy,
)

LOGGER = logging.getLogger("e0b_control_patient")


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


def loader(dataset, batch_size: int, shuffle: bool, workers: int, seed: int, pin_memory: bool) -> DataLoader:
    generator = torch.Generator()
    generator.manual_seed(seed)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=workers,
                      pin_memory=pin_memory, persistent_workers=workers > 0,
                      worker_init_fn=seed_worker, generator=generator)


def save_environment(output_dir: Path) -> None:
    payload = {
        "started_at": datetime.now().isoformat(timespec="seconds"), "python": sys.version,
        "platform": platform.platform(), "torch": torch.__version__,
        "torchvision": __import__("torchvision").__version__, "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda, "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "git_head": os.popen("git rev-parse HEAD 2>NUL").read().strip() or None,
    }
    (output_dir / "environment.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _batch_item(batch: dict, key: str, index: int):
    value = batch[key]
    return value[index].item() if torch.is_tensor(value) else value[index]


@torch.no_grad()
def evaluate(model: torch.nn.Module, data_loader: DataLoader, device: torch.device, fold: int,
             selected_epoch: int, checkpoint_path: Path) -> tuple[pd.DataFrame, dict]:
    model.eval(); labels_all: list[int] = []; probs_all: list[np.ndarray] = []; rows: list[dict] = []
    for batch in data_loader:
        logits = model(batch["image"].to(device, non_blocking=True))
        probs = torch.softmax(logits, dim=1).cpu().numpy(); logits_np = logits.cpu().numpy(); labels = batch["label"].numpy()
        pred = probs.argmax(axis=1); labels_all.extend(labels.astype(int)); probs_all.append(probs)
        for i, label in enumerate(labels):
            rows.append({
                "sample_id": _batch_item(batch, "sample_id", i), "patient_group_id": _batch_item(batch, "patient_group_id", i),
                "fold": int(_batch_item(batch, "fold", i)), "original_label": int(_batch_item(batch, "original_label", i)),
                "original_three_class_label": int(_batch_item(batch, "original_three_class_label", i)),
                "original_three_class_name": _batch_item(batch, "original_three_class_name", i), "binary_label": int(label),
                "logit_normal": float(logits_np[i, 0]), "logit_patient": float(logits_np[i, 1]),
                "prob_normal": float(probs[i, 0]), "prob_patient": float(probs[i, 1]), "pred_class": int(pred[i]),
                "is_correct": int(pred[i] == label), "image_path": _batch_item(batch, "image_path", i),
                "selected_epoch": selected_epoch, "checkpoint_path": str(checkpoint_path),
            })
    frame = pd.DataFrame(rows)
    metrics = compute_binary_metrics(np.asarray(labels_all), np.concatenate(probs_all))
    assert int(frame["fold"].iloc[0]) == fold
    return frame, metrics


def _classifier_out_features(model: torch.nn.Module) -> int | None:
    classifier = classifier_module(model)
    if isinstance(classifier, torch.nn.Linear):
        return int(classifier.out_features)
    for module in reversed(list(classifier.modules())):
        if isinstance(module, torch.nn.Linear):
            return int(module.out_features)
    return None


def run_fold(config: dict, output_dir: Path, fold: int, device: torch.device) -> None:
    data, train_cfg = config["data"], config["train"]
    split_dir, image_root = project_path(data["split_dir"]), project_path(data["image_root"])
    fold_dir = output_dir / f"fold_{fold}"; checkpoint_dir = fold_dir / "checkpoints"; checkpoint_dir.mkdir(parents=True, exist_ok=True)
    train_tf = build_transforms("train", data["image_size"], config["normalize"]["mean"], config["normalize"]["std"], bool(config["augmentation"]["horizontal_flip"]))
    val_tf = build_transforms("val", data["image_size"], config["normalize"]["mean"], config["normalize"]["std"], False)
    template = str(data.get("image_filename_template", "{ID}.png"))
    train_set = ControlPatientFaceDataset(split_dir / data["train_csv_pattern"].format(fold=fold), train_tf, image_root, template)
    val_set = ControlPatientFaceDataset(split_dir / data["val_csv_pattern"].format(fold=fold), val_tf, image_root, template)
    train_loader = loader(train_set, int(train_cfg["batch_size"]), True, int(train_cfg["num_workers"]), int(train_cfg["random_seed"]) + fold, bool(train_cfg["pin_memory"]))
    val_loader = loader(val_set, int(train_cfg["batch_size"]), False, int(train_cfg["num_workers"]), int(train_cfg["random_seed"]) + 1000 + fold, bool(train_cfg["pin_memory"]))
    dropout = config["model"].get("dropout", None)
    strategy = normalize_strategy(config["model"].get("trainability_strategy", "full_finetune"))
    model = build_nyha_classification_model(
        "resnet18",
        num_classes=2,
        pretrained=config["model"]["pretrained"],
        freeze_backbone=False,
        dropout=dropout,
    ).to(device)
    configure_trainability(model, strategy)
    if _classifier_out_features(model) != 2:
        raise RuntimeError("E0B classifier head must be Linear(512, 2)")
    weights = compute_class_weights(train_set.labels, num_classes=2).to(device)
    criterion = torch.nn.CrossEntropyLoss(weight=weights)
    optimizer = build_optimizer(
        model,
        strategy,
        full_lr=float(train_cfg["lr"]),
        classifier_lr=float(train_cfg.get("classifier_lr", train_cfg["lr"])),
        layer4_lr=float(train_cfg.get("layer4_lr", float(train_cfg.get("classifier_lr", train_cfg["lr"])) * 0.1)),
        weight_decay=float(train_cfg["weight_decay"]),
    )
    gradient_clip_max_norm = train_cfg.get("gradient_clip_max_norm", None)
    counts = np.bincount(np.asarray(train_set.labels), minlength=2)
    fold_info = {"fold": fold, "train_normal_count": int(counts[0]), "train_patient_count": int(counts[1]), "weight_normal": float(weights[0]), "weight_patient": float(weights[1]), **count_parameters(model)}
    (fold_dir / "training_parameters.json").write_text(json.dumps(fold_info, indent=2), encoding="utf-8")
    apply_train_mode(model, strategy)
    audit = build_trainability_audit(
        model,
        optimizer,
        strategy,
        fold=fold,
        dropout=dropout,
        gradient_clip_max_norm=None if gradient_clip_max_norm is None else float(gradient_clip_max_norm),
    )
    (fold_dir / "trainability_audit.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")
    LOGGER.info("fold=%d counts=%s class_weights=%s", fold, counts.tolist(), weights.tolist())
    best_auc, best_epoch, patience = float("-inf"), 0, 0; history: list[dict] = []; started = time.perf_counter()
    for epoch in range(1, int(train_cfg["epochs"]) + 1):
        apply_train_mode(model, strategy); total_loss = 0.0; seen = 0
        for batch in train_loader:
            image, label = batch["image"].to(device), batch["label"].to(device)
            optimizer.zero_grad(set_to_none=True); logits = model(image); loss = criterion(logits, label); loss.backward()
            if gradient_clip_max_norm is not None:
                torch.nn.utils.clip_grad_norm_(
                    [parameter for parameter in model.parameters() if parameter.requires_grad],
                    max_norm=float(gradient_clip_max_norm),
                )
            optimizer.step()
            total_loss += float(loss.detach()) * len(label); seen += len(label)
        val_frame, val_metrics = evaluate(model, val_loader, device, fold, epoch, checkpoint_dir / "best_macro_auc.pth")
        row = {"epoch": epoch, "train_loss": total_loss / seen, **flatten_metrics(val_metrics)}; history.append(row)
        auc = float(val_metrics["macro_auc"]); improved = auc > best_auc
        if improved:
            best_auc, best_epoch, patience = auc, epoch, 0
            torch.save({"model_state_dict": model.state_dict(), "epoch": epoch, "best_macro_auc": auc, "config": config, "trainability_audit": audit}, checkpoint_dir / "best_macro_auc.pth")
        else: patience += 1
        torch.save({"model_state_dict": model.state_dict(), "epoch": epoch, "best_macro_auc": best_auc, "config": config, "trainability_audit": audit}, checkpoint_dir / "last.pth")
        LOGGER.info("fold=%d epoch=%d train_loss=%.5f val_macro_auc=%.4f best=%.4f patience=%d/%d", fold, epoch, row["train_loss"], auc, best_auc, patience, int(train_cfg["early_stopping_patience"]))
        if patience >= int(train_cfg["early_stopping_patience"]): break
    pd.DataFrame(history).to_csv(fold_dir / "training_history.csv", index=False, encoding="utf-8-sig")
    checkpoint = torch.load(checkpoint_dir / "best_macro_auc.pth", map_location=device, weights_only=False); model.load_state_dict(checkpoint["model_state_dict"])
    frame, metrics = evaluate(model, val_loader, device, fold, int(checkpoint["epoch"]), checkpoint_dir / "best_macro_auc.pth")
    frame.to_csv(fold_dir / "val_predictions.csv", index=False, encoding="utf-8-sig")
    metric_row = {"fold": fold, "best_epoch": int(checkpoint["epoch"]), "training_seconds": time.perf_counter() - started, **fold_info, **flatten_metrics(metrics)}
    pd.DataFrame([metric_row]).to_csv(fold_dir / "metrics.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(metrics["confusion_matrix"], index=["control", "patient"], columns=["control", "patient"]).to_csv(fold_dir / "confusion_matrix.csv", index_label="true\\pred", encoding="utf-8-sig")


def main() -> Path:
    args = parse_args(); config = load_yaml(args.config)
    if args.epochs is not None: config["train"]["epochs"] = int(args.epochs)
    if config["model"]["num_classes"] != 2 or config["task"]["type"] != "binary_control_vs_patient": raise ValueError("E0B requires the fixed two-class task")
    if config["train"]["optimizer"].lower() != "adamw" or config["train"]["monitor_metric"] != "macro_auc": raise ValueError("E0B protocol requires AdamW and macro_auc")
    output_dir = project_path(args.output_dir); output_dir.mkdir(parents=True, exist_ok=True); configure_logging(output_dir / "experiment.log")
    save_yaml(config, output_dir / "config_snapshot.yaml"); save_environment(output_dir); audit_fixed_splits(config, output_dir)
    set_random_seed(int(config["train"]["random_seed"])); device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    folds = args.folds if args.folds is not None else list(range(int(config["data"]["n_folds"])))
    for fold in folds: run_fold(config, output_dir, fold, device)
    (output_dir / "run_finished_at.txt").write_text(datetime.now().isoformat(timespec="seconds"), encoding="utf-8")
    print(f"EXPERIMENT_DIR={output_dir}")
    return output_dir


if __name__ == "__main__": main()
