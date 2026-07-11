"""Train Global + selected ROI feature-level fusion models over fixed folds."""

from __future__ import annotations

import argparse
import logging
import math
import os
import shutil
import sys
from pathlib import Path
from typing import Any

# Import sklearn before torch in this Windows environment to avoid duplicate
# OpenMP runtime initialization issues observed in previous experiments.
import sklearn  # noqa: F401
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import f1_score
from torch import nn
from torch.utils.data import DataLoader


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from datasets.global_roi_fusion_dataset import GlobalROIFusionDataset  # noqa: E402
from losses.classification_losses import build_criterion, compute_class_weights  # noqa: E402
from metrics.classification_metrics import (  # noqa: E402
    CLASS_NAMES,
    compute_classification_metrics,
    flatten_metrics,
)
from models.global_roi_fusion_model import (  # noqa: E402
    GlobalROIFusionModel,
    count_parameters,
)
from utils.experiment_utils import (  # noqa: E402
    choose_device,
    configure_logging,
    create_experiment_dir,
    load_yaml,
    resolve_project_path,
    save_yaml,
    seed_worker,
    set_random_seed,
)


LOGGER = logging.getLogger("train_global_roi_fusion_5fold")
SUMMARY_FILES = [
    "fold_metrics_all.csv",
    "mean_metrics.csv",
    "oof_metrics.csv",
    "oof_predictions.csv",
    "summary_report.md",
]
MAIN_METRICS = [
    "macro_auc",
    "accuracy",
    "macro_precision",
    "macro_recall",
    "macro_f1",
    "weighted_f1",
    "balanced_accuracy",
]
AUXILIARY_METRICS = [
    "auc_normal",
    "auc_mild",
    "auc_severe",
    "precision_normal",
    "precision_mild",
    "precision_severe",
    "recall_normal",
    "recall_mild",
    "recall_severe",
    "f1_normal",
    "f1_mild",
    "f1_severe",
    "severe_vs_rest_auc",
    "normal_vs_abnormal_auc",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--fold", type=int, action="append", dest="folds")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume incomplete folds from last.pth and skip folds with metrics.",
    )
    return parser.parse_args()


def _resolve_loss_settings(config: dict[str, Any]) -> dict[str, Any]:
    loss_section = config.get("loss")
    if isinstance(loss_section, dict):
        return {
            "name": str(loss_section.get("name", "weighted_cross_entropy")),
            "class_weight": bool(loss_section.get("class_weight", True)),
        }
    return {
        "name": str(config["train"].get("loss", "weighted_cross_entropy")),
        "class_weight": True,
    }


def _as_path(value: str | Path | None, name: str) -> Path:
    path = resolve_project_path(value)
    if path is None:
        raise ValueError(f"{name} must not be empty")
    return path


def _read_split_csv(path: Path) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(f"Split CSV does not exist: {path}")
    frame = pd.read_csv(
        path,
        dtype={"ID": "string", "patient_group_id": "string"},
        encoding="utf-8-sig",
    )
    required = {"ID", "patient_group_id", "label_3class"}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"Split CSV is missing columns {missing}: {path}")
    labels = pd.to_numeric(frame["label_3class"], errors="coerce")
    if labels.isna().any() or not labels.isin([0, 1, 2]).all():
        bad = frame.index[labels.isna() | ~labels.isin([0, 1, 2])].tolist()
        raise ValueError(f"Invalid label_3class at rows {bad[:10]} in {path}")
    frame["label_3class"] = labels.astype(int)
    if "NYHA" in frame.columns:
        nyha = pd.to_numeric(frame["NYHA"], errors="coerce")
        expected = nyha.map(lambda x: 0 if x == 0 else (1 if x in {1, 2} else (2 if x in {3, 4} else -1)))
        bad_mask = expected.ne(frame["label_3class"])
        if bad_mask.any():
            bad_rows = frame.index[bad_mask].tolist()
            raise ValueError(f"NYHA -> label_3class mapping mismatch in {path}: {bad_rows[:10]}")
    return frame


def _expected_image_paths(
    frame: pd.DataFrame,
    global_root: Path,
    roi_roots: dict[str, Path],
    enabled_inputs: list[str],
    template: str,
) -> list[Path]:
    paths: list[Path] = []
    for identifier in frame["ID"].astype(str).tolist():
        filename = template.format(ID=identifier)
        paths.append(global_root / filename)
        for roi_name in enabled_inputs:
            if roi_name == "global":
                continue
            paths.append(roi_roots[roi_name] / filename)
    return paths


def preflight(config: dict[str, Any]) -> None:
    data = config["data"]
    split_dir = _as_path(data.get("split_dir"), "data.split_dir")
    global_root = _as_path(data.get("global_image_root"), "data.global_image_root")
    roi_roots = {
        str(name).lower(): _as_path(path, f"data.roi_roots.{name}")
        for name, path in (data.get("roi_roots") or {}).items()
    }
    enabled_inputs = [str(name).lower() for name in data["enabled_inputs"]]
    template = str(data.get("image_filename_template", "{ID}.png"))
    n_folds = int(data["n_folds"])

    if not split_dir.is_dir():
        raise FileNotFoundError(f"Split directory does not exist: {split_dir}")
    if not global_root.is_dir():
        raise FileNotFoundError(f"Global image root does not exist: {global_root}")
    if "global" not in enabled_inputs:
        raise ValueError("data.enabled_inputs must include global")
    for roi_name in [name for name in enabled_inputs if name != "global"]:
        if roi_name not in roi_roots:
            raise ValueError(f"Missing ROI root for enabled ROI: {roi_name}")
        if not roi_roots[roi_name].is_dir():
            raise FileNotFoundError(f"ROI root does not exist for {roi_name}: {roi_roots[roi_name]}")

    all_val_ids: list[str] = []
    missing_images: list[str] = []
    for fold in range(n_folds):
        train_csv = split_dir / str(data["train_csv_pattern"]).format(fold=fold)
        val_csv = split_dir / str(data["val_csv_pattern"]).format(fold=fold)
        train = _read_split_csv(train_csv)
        val = _read_split_csv(val_csv)
        train_groups = set(train["patient_group_id"].astype(str))
        val_groups = set(val["patient_group_id"].astype(str))
        overlap = sorted(train_groups.intersection(val_groups))
        if overlap:
            raise ValueError(
                f"Fold {fold} train/val patient_group_id overlap: {overlap[:10]}"
            )
        if "fold" in val.columns:
            fold_values = set(pd.to_numeric(val["fold"], errors="coerce").dropna().astype(int))
            if fold_values != {fold}:
                raise ValueError(
                    f"Fold {fold} validation CSV has unexpected fold values: {sorted(fold_values)}"
                )
        all_val_ids.extend(val["ID"].astype(str).tolist())
        for frame in (train, val):
            for path in _expected_image_paths(
                frame, global_root, roi_roots, enabled_inputs, template
            ):
                if not path.is_file():
                    missing_images.append(str(path))

    duplicated_ids = pd.Series(all_val_ids, dtype="string").duplicated(keep=False)
    if duplicated_ids.any():
        duplicates = pd.Series(all_val_ids, dtype="string")[duplicated_ids].tolist()
        raise ValueError(f"OOF validation IDs contain duplicates: {duplicates[:10]}")
    if len(all_val_ids) != 500:
        raise ValueError(f"OOF validation sample count must be 500, got {len(all_val_ids)}")
    if missing_images:
        preview = "\n".join(missing_images[:30])
        raise FileNotFoundError(
            f"Missing {len(missing_images)} required images. First 30:\n{preview}"
        )


def _build_loader(
    dataset: GlobalROIFusionDataset,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
    seed: int,
    pin_memory: bool,
) -> DataLoader:
    generator = torch.Generator()
    generator.manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=num_workers > 0,
        worker_init_fn=seed_worker,
        generator=generator,
    )


def _build_dataset(
    config: dict[str, Any],
    csv_path: Path,
    train: bool,
) -> GlobalROIFusionDataset:
    data = config["data"]
    return GlobalROIFusionDataset(
        csv_path=csv_path,
        global_image_root=_as_path(data["global_image_root"], "data.global_image_root"),
        roi_roots={
            name: _as_path(path, f"data.roi_roots.{name}")
            for name, path in (data.get("roi_roots") or {}).items()
        },
        enabled_inputs=data["enabled_inputs"],
        image_filename_template=str(data.get("image_filename_template", "{ID}.png")),
        image_size=int(data["image_size"]),
        label_col=str(data.get("label_col", "label_3class")),
        train=train,
        horizontal_flip=bool(config.get("augmentation", {}).get("horizontal_flip", True)),
        mean=config["normalize"]["mean"],
        std=config["normalize"]["std"],
    )


def _build_model(config: dict[str, Any]) -> GlobalROIFusionModel:
    model_config = config["model"]
    data_config = config["data"]
    return GlobalROIFusionModel(
        backbone=str(model_config.get("backbone", "resnet18")),
        num_classes=int(model_config.get("num_classes", 3)),
        pretrained=model_config.get("pretrained", "imagenet"),
        enabled_inputs=data_config["enabled_inputs"],
        projection_dim=int(model_config.get("projection_dim", 256)),
        dropout=float(model_config.get("dropout", 0.3)),
        freeze_backbone=bool(model_config.get("freeze_backbone", False)),
    )


def _model_inputs(batch: dict[str, Any], device: torch.device) -> dict[str, torch.Tensor | None]:
    return {
        "global_image": batch["global_image"].to(device, non_blocking=True),
        "eye_image": batch["eye_image"].to(device, non_blocking=True) if "eye_image" in batch else None,
        "cheek_image": batch["cheek_image"].to(device, non_blocking=True) if "cheek_image" in batch else None,
    }


def _compute_metrics(y_true: np.ndarray, y_prob: np.ndarray) -> dict[str, Any]:
    metrics = compute_classification_metrics(y_true, y_prob, num_classes=3)
    y_pred = np.asarray(y_prob).argmax(axis=1)
    metrics["weighted_f1"] = float(
        f1_score(y_true, y_pred, labels=[0, 1, 2], average="weighted", zero_division=0)
    )
    return metrics


def _autocast(device: torch.device, use_amp: bool):
    try:
        return torch.amp.autocast(device_type=device.type, enabled=use_amp)
    except AttributeError:
        return torch.cuda.amp.autocast(enabled=use_amp)


def _train_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    scaler,
    use_amp: bool,
) -> float:
    model.train()
    total_loss = 0.0
    total_samples = 0
    for batch in loader:
        labels = batch["label"].to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with _autocast(device, use_amp):
            logits = model(**_model_inputs(batch, device))
            loss = criterion(logits, labels)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        batch_size = labels.size(0)
        total_loss += float(loss.detach().item()) * batch_size
        total_samples += batch_size
    return total_loss / max(total_samples, 1)


@torch.no_grad()
def _validate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    use_amp: bool,
) -> tuple[float, dict[str, Any]]:
    model.eval()
    total_loss = 0.0
    total_samples = 0
    labels_all: list[np.ndarray] = []
    probs_all: list[np.ndarray] = []
    for batch in loader:
        labels = batch["label"].to(device, non_blocking=True)
        with _autocast(device, use_amp):
            logits = model(**_model_inputs(batch, device))
            loss = criterion(logits, labels)
        probs = torch.softmax(logits, dim=1)
        batch_size = labels.size(0)
        total_loss += float(loss.item()) * batch_size
        total_samples += batch_size
        labels_all.append(labels.cpu().numpy())
        probs_all.append(probs.cpu().numpy())
    return total_loss / max(total_samples, 1), _compute_metrics(
        np.concatenate(labels_all), np.concatenate(probs_all, axis=0)
    )


def _checkpoint_payload(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    fold: int,
    best_macro_auc: float,
    config: dict[str, Any],
) -> dict[str, Any]:
    return {
        "epoch": epoch,
        "fold": fold,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "best_macro_auc": best_macro_auc,
        "config": config,
    }


def _load_checkpoint(path: Path, device: torch.device) -> dict[str, Any]:
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


def _save_curves(history: pd.DataFrame, curve_dir: Path) -> None:
    curve_dir.mkdir(parents=True, exist_ok=True)
    if history.empty:
        return
    plt.figure(figsize=(7, 5))
    plt.plot(history["epoch"], history["train_loss"], label="train_loss")
    plt.plot(history["epoch"], history["val_loss"], label="val_loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(curve_dir / "loss_curve.png", dpi=160)
    plt.close()

    plt.figure(figsize=(8, 5))
    for column in (
        "val_macro_auc",
        "val_accuracy",
        "val_macro_f1",
        "val_balanced_accuracy",
    ):
        if column in history:
            plt.plot(history["epoch"], history[column], label=column)
    plt.xlabel("Epoch")
    plt.ylabel("Metric")
    plt.ylim(0.0, 1.0)
    plt.legend()
    plt.tight_layout()
    plt.savefig(curve_dir / "metrics_curve.png", dpi=160)
    plt.close()


def _load_resume_state(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    checkpoint_path: Path | None,
    log_dir: Path,
    device: torch.device,
    fold: int,
    patience: int,
) -> tuple[int, float, int, list[dict[str, Any]]]:
    if checkpoint_path is None:
        return 1, float("-inf"), 0, []
    checkpoint = _load_checkpoint(checkpoint_path, device)
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    completed_epoch = int(checkpoint.get("epoch", 0))
    best_macro_auc = float(checkpoint.get("best_macro_auc", float("-inf")))
    records: list[dict[str, Any]] = []
    epochs_without_improvement = 0
    history_path = log_dir / "train_log.csv"
    if history_path.is_file():
        history = pd.read_csv(history_path)
        history = history[pd.to_numeric(history["epoch"], errors="coerce") <= completed_epoch]
        records = history.to_dict("records")
        if "val_macro_auc" in history and not history.empty:
            best_macro_auc = float(pd.to_numeric(history["val_macro_auc"]).max())
            best_epoch = int(
                history.loc[pd.to_numeric(history["val_macro_auc"]).idxmax(), "epoch"]
            )
            epochs_without_improvement = max(completed_epoch - best_epoch, 0)
    LOGGER.info(
        "Resuming fold=%d from %s at epoch=%d with best_macro_auc=%s and patience=%d/%d",
        fold,
        checkpoint_path,
        completed_epoch + 1,
        f"{best_macro_auc:.4f}" if math.isfinite(best_macro_auc) else "unavailable",
        epochs_without_improvement,
        patience,
    )
    return completed_epoch + 1, best_macro_auc, epochs_without_improvement, records


def train_one_fold(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    fold_dir: Path,
    epochs: int,
    patience: int,
    use_amp: bool,
    fold: int,
    config: dict[str, Any],
    resume_from: Path | None = None,
) -> pd.DataFrame:
    checkpoint_dir = fold_dir / "checkpoints"
    log_dir = fold_dir / "logs"
    curve_dir = fold_dir / "curves"
    for directory in (checkpoint_dir, log_dir, curve_dir):
        directory.mkdir(parents=True, exist_ok=True)

    try:
        scaler = torch.amp.GradScaler(device.type, enabled=use_amp)
    except (AttributeError, TypeError):
        scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    start_epoch, best_macro_auc, epochs_without_improvement, records = _load_resume_state(
        model,
        optimizer,
        resume_from,
        log_dir,
        device,
        fold,
        patience,
    )
    LOGGER.info("Starting fold=%d on device=%s, AMP=%s", fold, device, use_amp)
    if start_epoch > epochs:
        history = pd.DataFrame(records)
        _save_curves(history, curve_dir)
        return history

    for epoch in range(start_epoch, epochs + 1):
        train_loss = _train_epoch(
            model, train_loader, criterion, optimizer, device, scaler, use_amp
        )
        val_loss, val_metrics = _validate(model, val_loader, criterion, device, use_amp)
        macro_auc = float(val_metrics["macro_auc"])
        record = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "val_macro_auc": macro_auc,
            "val_accuracy": float(val_metrics["accuracy"]),
            "val_macro_precision": float(val_metrics["macro_precision"]),
            "val_macro_recall": float(val_metrics["macro_recall"]),
            "val_macro_f1": float(val_metrics["macro_f1"]),
            "val_weighted_f1": float(val_metrics["weighted_f1"]),
            "val_balanced_accuracy": float(val_metrics["balanced_accuracy"]),
            "lr": float(optimizer.param_groups[0]["lr"]),
        }
        records.append(record)
        history = pd.DataFrame(records)
        history.to_csv(log_dir / "train_log.csv", index=False, encoding="utf-8-sig")

        improved = not math.isnan(macro_auc) and macro_auc > best_macro_auc
        if improved:
            best_macro_auc = macro_auc
            epochs_without_improvement = 0
            torch.save(
                _checkpoint_payload(model, optimizer, epoch, fold, best_macro_auc, config),
                checkpoint_dir / "best_macro_auc.pth",
            )
        else:
            epochs_without_improvement += 1
        torch.save(
            _checkpoint_payload(model, optimizer, epoch, fold, best_macro_auc, config),
            checkpoint_dir / "last.pth",
        )
        LOGGER.info(
            "fold=%d epoch=%d/%d train_loss=%.5f val_loss=%.5f "
            "val_macro_auc=%s val_macro_f1=%.4f best=%s patience=%d/%d",
            fold,
            epoch,
            epochs,
            train_loss,
            val_loss,
            f"{macro_auc:.4f}" if not math.isnan(macro_auc) else "nan",
            val_metrics["macro_f1"],
            f"{best_macro_auc:.4f}" if math.isfinite(best_macro_auc) else "unavailable",
            epochs_without_improvement,
            patience,
        )
        if epochs_without_improvement >= patience:
            LOGGER.info("Early stopping fold=%d at epoch=%d", fold, epoch)
            break

    history = pd.DataFrame(records)
    _save_curves(history, curve_dir)
    best_path = checkpoint_dir / "best_macro_auc.pth"
    if not best_path.is_file():
        shutil.copy2(checkpoint_dir / "last.pth", best_path)
    return history


def _batch_value(batch: dict[str, Any], key: str, index: int, default: Any = "") -> Any:
    if key not in batch:
        return default
    value = batch[key]
    if torch.is_tensor(value):
        return value[index].item()
    return value[index]


@torch.no_grad()
def evaluate_fold(
    model: nn.Module,
    loader: DataLoader,
    checkpoint_path: Path,
    device: torch.device,
    fold_dir: Path,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    checkpoint = _load_checkpoint(checkpoint_path, device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    rows: list[dict[str, Any]] = []
    labels_all: list[int] = []
    probs_all: list[np.ndarray] = []
    for batch in loader:
        logits = model(**_model_inputs(batch, device))
        probabilities = torch.softmax(logits, dim=1).cpu().numpy()
        predictions = probabilities.argmax(axis=1)
        labels = batch["label"].cpu().numpy()
        labels_all.extend(labels.astype(int).tolist())
        probs_all.append(probabilities)
        for index in range(len(labels)):
            label = int(labels[index])
            predicted = int(predictions[index])
            row = {
                "ID": _batch_value(batch, "ID", index),
                "patient_group_id": _batch_value(batch, "patient_group_id", index),
                "image_path": _batch_value(batch, "image_path", index),
                "global_image_path": _batch_value(batch, "global_image_path", index),
                "eye_image_path": _batch_value(batch, "eye_image_path", index),
                "cheek_image_path": _batch_value(batch, "cheek_image_path", index),
                "NYHA": int(_batch_value(batch, "NYHA", index, -1)),
                "SEX": int(_batch_value(batch, "SEX", index, -1)),
                "sex_name": _batch_value(batch, "sex_name", index),
                "label_3class": label,
                "label_3class_name": _batch_value(batch, "label_3class_name", index),
                "y_true": label,
                "y_pred": predicted,
                "pred_class": predicted,
                "pred_class_name": CLASS_NAMES[predicted],
                "prob_normal": float(probabilities[index, 0]),
                "prob_mild": float(probabilities[index, 1]),
                "prob_severe": float(probabilities[index, 2]),
                "correct": int(predicted == label),
                "fold": int(_batch_value(batch, "fold", index, -1)),
            }
            rows.append(row)

    prediction_dir = fold_dir / "predictions"
    metric_dir = fold_dir / "metrics"
    prediction_dir.mkdir(parents=True, exist_ok=True)
    metric_dir.mkdir(parents=True, exist_ok=True)
    prediction_frame = pd.DataFrame(rows)
    prediction_frame.to_csv(
        prediction_dir / "val_predictions.csv",
        index=False,
        encoding="utf-8-sig",
    )
    metrics = _compute_metrics(
        np.asarray(labels_all), np.concatenate(probs_all, axis=0)
    )
    metric_row = flatten_metrics(metrics)
    metric_row["fold"] = int(prediction_frame["fold"].iloc[0])
    pd.DataFrame([metric_row]).to_csv(
        metric_dir / "fold_metrics.csv",
        index=False,
        encoding="utf-8-sig",
    )
    _save_confusion_matrix(metrics["confusion_matrix"], metric_dir)
    LOGGER.info("Loaded checkpoint: %s", checkpoint_path)
    return prediction_frame, metrics


def _save_confusion_matrix(matrix: np.ndarray, metric_dir: Path) -> None:
    labels = [CLASS_NAMES[index] for index in range(3)]
    pd.DataFrame(matrix, index=labels, columns=labels).to_csv(
        metric_dir / "confusion_matrix.csv",
        index_label="true\\pred",
        encoding="utf-8-sig",
    )
    figure, axis = plt.subplots(figsize=(6, 5))
    image = axis.imshow(matrix, interpolation="nearest", cmap="Blues")
    figure.colorbar(image, ax=axis)
    axis.set(
        xticks=np.arange(3),
        yticks=np.arange(3),
        xticklabels=labels,
        yticklabels=labels,
        xlabel="Predicted class",
        ylabel="True class",
        title="Validation confusion matrix",
    )
    threshold = matrix.max() / 2.0 if matrix.size else 0
    for row in range(3):
        for column in range(3):
            axis.text(
                column,
                row,
                str(int(matrix[row, column])),
                ha="center",
                va="center",
                color="white" if matrix[row, column] > threshold else "black",
            )
    figure.tight_layout()
    figure.savefig(metric_dir / "confusion_matrix.png", dpi=180)
    plt.close(figure)


def _mean_std_rows(fold_metrics: pd.DataFrame, metric_names: list[str]) -> pd.DataFrame:
    rows = []
    for metric in metric_names:
        if metric not in fold_metrics.columns:
            raise ValueError(f"Fold metrics are missing required column: {metric}")
        values = pd.to_numeric(fold_metrics[metric], errors="coerce")
        rows.append({"metric": metric, "mean": values.mean(), "std": values.std(ddof=1)})
    return pd.DataFrame(rows)


def _format_metric(value: float) -> str:
    return "nan" if pd.isna(value) else f"{float(value):.4f}"


def _write_summary_report(
    path: Path,
    config: dict[str, Any],
    mean_frame: pd.DataFrame,
    aux_frame: pd.DataFrame,
    oof_metrics: dict[str, Any],
) -> None:
    lookup = mean_frame.set_index("metric")
    aux_lookup = aux_frame.set_index("metric")
    matrix = np.asarray(oof_metrics["confusion_matrix"])
    lines = [
        f"# {config['experiment']['name']}",
        "",
        "## Experiment setup",
        "",
        "- Model type: global_roi_fusion",
        f"- Backbone: {config['model']['backbone']}",
        f"- Pretrained weights: {config['model']['pretrained']}",
        f"- Enabled inputs: {', '.join(config['data']['enabled_inputs'])}",
        f"- Global image root: `{config['data']['global_image_root']}`",
        f"- ROI roots: `{config['data']['roi_roots']}`",
        f"- Fixed fold files: `{config['data']['split_dir']}`",
        "- Fusion: independent branch backbone -> 256-d projection -> concat classifier",
        "- Augmentation: synchronized horizontal flip only; no ColorJitter, crop, or rotation",
        "",
        "## Fold-level mean ± std",
        "",
        "| Metric | Mean ± std |",
        "|---|---:|",
    ]
    for metric in MAIN_METRICS:
        lines.append(
            f"| {metric} | {_format_metric(lookup.loc[metric, 'mean'])} "
            f"± {_format_metric(lookup.loc[metric, 'std'])} |"
        )
    lines.extend(["", "## Fold-level auxiliary metrics", "", "| Metric | Mean ± std |", "|---|---:|"])
    for metric in AUXILIARY_METRICS:
        lines.append(
            f"| {metric} | {_format_metric(aux_lookup.loc[metric, 'mean'])} "
            f"± {_format_metric(aux_lookup.loc[metric, 'std'])} |"
        )
    lines.extend(["", "## OOF metrics", "", "| Metric | Value |", "|---|---:|"])
    for metric, value in flatten_metrics(oof_metrics).items():
        lines.append(f"| {metric} | {_format_metric(value)} |")
    lines.extend(
        [
            "",
            "## OOF confusion matrix",
            "",
            "| True \\ Pred | normal | mild | severe |",
            "|---|---:|---:|---:|",
            f"| normal | {matrix[0, 0]} | {matrix[0, 1]} | {matrix[0, 2]} |",
            f"| mild | {matrix[1, 0]} | {matrix[1, 1]} | {matrix[1, 2]} |",
            f"| severe | {matrix[2, 0]} | {matrix[2, 1]} | {matrix[2, 2]} |",
            "",
            "All fold results are held-out validation results, not an independent test set.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def summarize_experiment(experiment_dir: Path, config: dict[str, Any]) -> Path:
    summary_dir = experiment_dir / "summary"
    summary_dir.mkdir(parents=True, exist_ok=True)
    n_folds = int(config["data"]["n_folds"])
    metric_frames: list[pd.DataFrame] = []
    prediction_frames: list[pd.DataFrame] = []
    for fold in range(n_folds):
        metric_path = experiment_dir / f"fold_{fold}" / "metrics" / "fold_metrics.csv"
        prediction_path = experiment_dir / f"fold_{fold}" / "predictions" / "val_predictions.csv"
        if not metric_path.is_file() or not prediction_path.is_file():
            raise FileNotFoundError(
                f"Fold {fold} artifacts are incomplete: {metric_path}, {prediction_path}"
            )
        metric_frames.append(pd.read_csv(metric_path))
        prediction_frames.append(
            pd.read_csv(
                prediction_path,
                dtype={"ID": "string", "patient_group_id": "string"},
                encoding="utf-8-sig",
            )
        )
    fold_metrics = pd.concat(metric_frames, ignore_index=True).sort_values("fold")
    fold_metrics.to_csv(summary_dir / "fold_metrics_all.csv", index=False, encoding="utf-8-sig")
    mean_frame = _mean_std_rows(fold_metrics, MAIN_METRICS)
    mean_frame.to_csv(summary_dir / "mean_metrics.csv", index=False, encoding="utf-8-sig")
    aux_frame = _mean_std_rows(fold_metrics, AUXILIARY_METRICS)

    oof = pd.concat(prediction_frames, ignore_index=True).sort_values(["fold", "ID"], kind="stable")
    if oof["ID"].duplicated().any():
        duplicates = oof.loc[oof["ID"].duplicated(keep=False), "ID"].tolist()
        raise ValueError(f"OOF predictions contain duplicate sample IDs: {duplicates[:10]}")
    if len(oof) != 500:
        raise ValueError(f"OOF predictions must contain 500 rows, got {len(oof)}")
    oof.to_csv(summary_dir / "oof_predictions.csv", index=False, encoding="utf-8-sig")
    oof_metrics = _compute_metrics(
        oof["label_3class"].to_numpy(),
        oof[["prob_normal", "prob_mild", "prob_severe"]].to_numpy(),
    )
    pd.DataFrame([flatten_metrics(oof_metrics)]).to_csv(
        summary_dir / "oof_metrics.csv", index=False, encoding="utf-8-sig"
    )
    _write_summary_report(
        summary_dir / "summary_report.md",
        config,
        mean_frame,
        aux_frame,
        oof_metrics,
    )
    return summary_dir


def _write_model_summary(experiment_dir: Path, model: nn.Module, config: dict[str, Any]) -> None:
    counts = count_parameters(model)
    lines = [
        f"model_class_name: {model.__class__.__name__}",
        f"backbone: {config['model'].get('backbone')}",
        f"pretrained: {config['model'].get('pretrained')}",
        f"enabled_inputs: {','.join(config['data'].get('enabled_inputs', []))}",
        f"projection_dim: {config['model'].get('projection_dim')}",
        f"dropout: {config['model'].get('dropout')}",
        f"freeze_backbone: {config['model'].get('freeze_backbone')}",
        f"total_params: {counts['total_params']}",
        f"trainable_params: {counts['trainable_params']}",
    ]
    (experiment_dir / "model_summary.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> Path:
    args = parse_args()
    config = load_yaml(args.config)
    if args.epochs is not None:
        if args.epochs < 1:
            raise ValueError("--epochs must be >= 1")
        config["train"]["epochs"] = args.epochs
    if args.num_workers is not None:
        if args.num_workers < 0:
            raise ValueError("--num-workers cannot be negative")
        config["train"]["num_workers"] = args.num_workers
    if args.batch_size is not None:
        if args.batch_size < 1:
            raise ValueError("--batch-size must be >= 1")
        config["train"]["batch_size"] = args.batch_size

    if str(config["model"].get("backbone", "")).lower() != "resnet18":
        raise ValueError("First global_roi_fusion version only supports backbone=resnet18")
    if bool(config["model"].get("freeze_backbone", False)):
        raise ValueError("This experiment requires freeze_backbone=false")
    if str(config["train"].get("monitor_metric", "macro_auc")) != "macro_auc":
        raise ValueError("This experiment requires monitor_metric=macro_auc")
    if str(config["train"].get("loss", "weighted_cross_entropy")) != "weighted_cross_entropy":
        raise ValueError("This experiment requires weighted_cross_entropy without focal/loss smoothing")

    preflight(config)
    seed = int(config["train"]["random_seed"])
    set_random_seed(seed)
    device = choose_device()
    use_amp = bool(config["train"].get("use_amp", False)) and device.type == "cuda"

    if args.output_dir is None:
        experiment_dir = create_experiment_dir(
            config["experiment"]["output_dir"], config["experiment"]["name"]
        )
    else:
        experiment_dir = _as_path(args.output_dir, "--output-dir")
        experiment_dir.mkdir(parents=True, exist_ok=True)
    configure_logging(experiment_dir / "experiment.log")
    save_yaml(config, experiment_dir / "config.yaml")

    LOGGER.info("Experiment directory: %s", experiment_dir)
    LOGGER.info("Training device: %s", device)
    LOGGER.info("TORCH_HOME: %s", os.environ.get("TORCH_HOME"))
    LOGGER.info("Torch hub directory: %s", torch.hub.get_dir())
    LOGGER.info("Enabled inputs: %s", config["data"]["enabled_inputs"])
    LOGGER.info("Global image root: %s", _as_path(config["data"]["global_image_root"], "data.global_image_root"))
    LOGGER.info("ROI roots: %s", config["data"].get("roi_roots", {}))

    split_dir = _as_path(config["data"]["split_dir"], "data.split_dir")
    n_folds = int(config["data"]["n_folds"])
    folds = args.folds if args.folds is not None else list(range(n_folds))
    invalid = sorted(set(folds).difference(range(n_folds)))
    if invalid:
        raise ValueError(f"Requested folds outside [0, {n_folds - 1}]: {invalid}")
    folds = list(dict.fromkeys(folds))

    train_cfg = config["train"]
    loss_settings = _resolve_loss_settings(config)
    for fold in folds:
        fold_dir = experiment_dir / f"fold_{fold}"
        fold_metrics_path = fold_dir / "metrics" / "fold_metrics.csv"
        best_checkpoint_path = fold_dir / "checkpoints" / "best_macro_auc.pth"
        if args.resume and fold_metrics_path.is_file() and best_checkpoint_path.is_file():
            LOGGER.info("Skipping completed fold %d in resume mode", fold)
            continue

        LOGGER.info("Preparing held-out validation fold %d/%d", fold, n_folds - 1)
        train_csv = split_dir / str(config["data"]["train_csv_pattern"]).format(fold=fold)
        val_csv = split_dir / str(config["data"]["val_csv_pattern"]).format(fold=fold)
        train_dataset = _build_dataset(config, train_csv, train=True)
        val_dataset = _build_dataset(config, val_csv, train=False)
        train_loader = _build_loader(
            train_dataset,
            int(train_cfg["batch_size"]),
            True,
            int(train_cfg["num_workers"]),
            seed + fold,
            bool(train_cfg.get("pin_memory", False)),
        )
        val_loader = _build_loader(
            val_dataset,
            int(train_cfg["batch_size"]),
            False,
            int(train_cfg["num_workers"]),
            seed + 1000 + fold,
            bool(train_cfg.get("pin_memory", False)),
        )

        model = _build_model(config).to(device)
        _write_model_summary(experiment_dir, model, config)
        counts = count_parameters(model)
        LOGGER.info(
            "Model parameters: total=%d, trainable=%d",
            counts["total_params"],
            counts["trainable_params"],
        )
        class_weights = compute_class_weights(
            train_dataset.labels, int(config["data"]["num_classes"])
        )
        LOGGER.info("fold=%d class weights=%s", fold, class_weights.tolist())
        criterion = build_criterion(
            loss_settings["name"],
            class_weights if loss_settings["class_weight"] else None,
            device=device,
            num_classes=int(config["data"]["num_classes"]),
        )
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=float(train_cfg["lr"]),
            weight_decay=float(train_cfg["weight_decay"]),
        )
        resume_from = None
        if args.resume:
            last_path = fold_dir / "checkpoints" / "last.pth"
            if last_path.is_file():
                resume_from = last_path
        train_one_fold(
            model,
            train_loader,
            val_loader,
            criterion,
            optimizer,
            device,
            fold_dir,
            int(train_cfg["epochs"]),
            int(train_cfg["early_stopping_patience"]),
            use_amp,
            fold,
            config,
            resume_from=resume_from,
        )
        evaluate_fold(
            model,
            val_loader,
            fold_dir / "checkpoints" / "best_macro_auc.pth",
            device,
            fold_dir,
        )
        LOGGER.info("Completed held-out validation fold %d", fold)

    if args.folds is None:
        summary_dir = summarize_experiment(experiment_dir, config)
        print(f"SUMMARY_DIR={summary_dir}")
    else:
        LOGGER.info("Fold subset requested; full 5-fold summary was not generated.")
    print(f"EXPERIMENT_DIR={experiment_dir}")
    return experiment_dir


if __name__ == "__main__":
    main()
