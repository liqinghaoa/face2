"""Train one method under the locked patient-group 5x5 nested CV protocol."""

from __future__ import annotations

import argparse
import json
import logging
import math
import random
import subprocess
import sys
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn  # noqa: F401 - initialize the Windows OpenMP runtime first
import torch
import yaml
from sklearn.metrics import confusion_matrix
from torch import nn
from torch.utils.data import DataLoader


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from datasets.nyha_3class_face_dataset import (  # noqa: E402
    NYHA3ClassFaceDataset,
    build_transforms,
)
from losses.classification_losses import compute_class_weights  # noqa: E402
from metrics.classification_metrics import (  # noqa: E402
    CLASS_NAMES,
    compute_classification_metrics,
    flatten_metrics,
)
from models.nyha_backbone_factory import build_nyha_classification_model  # noqa: E402
from models.ordinal_nyha import MonotonicCumulativeResNet18  # noqa: E402
from utils.experiment_utils import seed_worker, set_random_seed  # noqa: E402
from utils.nested_cv_protocol import (  # noqa: E402
    audit_shared_protocol,
    generate_shared_protocol,
    read_split,
    sha256_file,
)
from utils.ordinal_utils import (  # noqa: E402
    compute_cumulative_pos_weight,
    compute_ordinal_metrics,
    cumulative_logits_to_probabilities,
    encode_ordinal_targets,
    monotonic_violation_count,
)


LOGGER = logging.getLogger("train_nyha_nested5x5")
OUTPUT_ROOT = PROJECT_ROOT / "experiments/ordinal_stage1_nested5x5_500Data"
PROTOCOL_DIR = OUTPUT_ROOT / "protocol"
CONFIGS = {
    "ce": PROJECT_ROOT
    / "config/train/ordinal/nyha_3class_resnet18_meanbg_nested5x5_weighted_ce.yaml",
    "ordinal": PROJECT_ROOT
    / "config/train/ordinal/nyha_3class_resnet18_meanbg_nested5x5_monotonic_cumulative.yaml",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", choices=["ce", "ordinal"], required=True)
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--outer-fold", type=int, action="append", dest="outer_folds")
    parser.add_argument("--inner-fold", type=int, default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--skip-completed", action="store_true")
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--protocol-only", action="store_true")
    return parser.parse_args()


def resolve_project_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def load_config(path: Path) -> dict[str, Any]:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def write_json(value: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def git_commit() -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=PROJECT_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip() if result.returncode == 0 else "unavailable"


def configure_logging(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        handlers=[logging.StreamHandler(), logging.FileHandler(path, encoding="utf-8")],
        force=True,
    )


def method_experiment_dir(config: dict[str, Any], smoke_test: bool) -> Path:
    parent = OUTPUT_ROOT / "smoke_tests" if smoke_test else OUTPUT_ROOT
    name = str(config["experiment"]["name"])
    return parent / (f"{name}_SMOKE" if smoke_test else name)


def build_dataset(
    csv_path: Path, config: dict[str, Any], *, train: bool
) -> NYHA3ClassFaceDataset:
    image_size = int(config["data"]["image_size"])
    transform = build_transforms(
        "train" if train else "val",
        image_size=image_size,
        mean=config["normalize"]["mean"],
        std=config["normalize"]["std"],
        horizontal_flip=bool(config["augmentation"]["horizontal_flip"]) if train else False,
    )
    return NYHA3ClassFaceDataset(
        csv_path,
        transform=transform,
        image_root=resolve_project_path(config["data"]["image_root"]),
        image_filename_template=str(config["data"]["image_filename_template"]),
    )


def build_loader(
    dataset: NYHA3ClassFaceDataset,
    config: dict[str, Any],
    *,
    shuffle: bool,
    seed: int,
) -> DataLoader:
    generator = torch.Generator()
    generator.manual_seed(int(seed))
    workers = int(config["train"]["num_workers"])
    return DataLoader(
        dataset,
        batch_size=int(config["train"]["batch_size"]),
        shuffle=shuffle,
        num_workers=workers,
        pin_memory=bool(config["train"].get("pin_memory", False)),
        persistent_workers=workers > 0,
        worker_init_fn=seed_worker,
        generator=generator,
    )


def build_model(method: str, config: dict[str, Any]) -> nn.Module:
    if method == "ce":
        return build_nyha_classification_model(
            backbone="resnet18",
            num_classes=3,
            pretrained=True,
            freeze_backbone=False,
        )
    return MonotonicCumulativeResNet18(
        pretrained=True, min_gap=float(config["model"]["min_gap"])
    )


def loss_weights(method: str, labels: list[int], device: torch.device) -> torch.Tensor:
    label_tensor = torch.as_tensor(labels, dtype=torch.long)
    if method == "ce":
        return compute_class_weights(label_tensor.tolist(), num_classes=3).to(device)
    return compute_cumulative_pos_weight(label_tensor, num_classes=3).to(device)


def make_criterion(method: str, weights: torch.Tensor) -> nn.Module:
    if method == "ce":
        return nn.CrossEntropyLoss(weight=weights)
    return nn.BCEWithLogitsLoss(pos_weight=weights)


def probabilities_and_loss(
    method: str,
    logits: torch.Tensor,
    labels: torch.Tensor,
    criterion: nn.Module,
) -> tuple[torch.Tensor, torch.Tensor]:
    if method == "ce":
        loss = criterion(logits, labels)
        return torch.softmax(logits, dim=1), loss
    targets = encode_ordinal_targets(labels, num_classes=3)
    loss = criterion(logits, targets)
    return cumulative_logits_to_probabilities(logits), loss


def combined_metrics(y_true: np.ndarray, probabilities: np.ndarray) -> dict[str, Any]:
    metrics = compute_classification_metrics(y_true, probabilities)
    predicted = probabilities.argmax(axis=1)
    return {**flatten_metrics(metrics), **compute_ordinal_metrics(y_true, predicted)}


def train_one_epoch(
    model: nn.Module,
    dataset: NYHA3ClassFaceDataset,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    method: str,
    config: dict[str, Any],
    device: torch.device,
    seed: int,
) -> float:
    set_random_seed(seed)
    loader = build_loader(dataset, config, shuffle=True, seed=seed)
    model.train()
    total = 0.0
    count = 0
    for batch in loader:
        images = batch["image"].to(device, non_blocking=True)
        labels = batch["label"].to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        logits = model(images)
        _, loss = probabilities_and_loss(method, logits, labels, criterion)
        loss.backward()
        optimizer.step()
        total += float(loss.detach().item()) * len(labels)
        count += len(labels)
    return total / count


@torch.no_grad()
def evaluate_dataset(
    model: nn.Module,
    dataset: NYHA3ClassFaceDataset,
    criterion: nn.Module,
    method: str,
    config: dict[str, Any],
    device: torch.device,
    *,
    seed: int,
    return_rows: bool = False,
) -> tuple[float, dict[str, Any], list[dict[str, Any]], int]:
    loader = build_loader(dataset, config, shuffle=False, seed=seed)
    model.eval()
    total = 0.0
    count = 0
    labels_all: list[np.ndarray] = []
    probabilities_all: list[np.ndarray] = []
    rows: list[dict[str, Any]] = []
    violations = 0
    for batch in loader:
        images = batch["image"].to(device, non_blocking=True)
        labels = batch["label"].to(device, non_blocking=True)
        logits = model(images)
        probabilities, loss = probabilities_and_loss(method, logits, labels, criterion)
        if method == "ordinal":
            violations += monotonic_violation_count(logits)
        probs_np = probabilities.cpu().numpy()
        labels_np = labels.cpu().numpy()
        total += float(loss.item()) * len(labels)
        count += len(labels)
        labels_all.append(labels_np)
        probabilities_all.append(probs_np)
        if return_rows:
            predicted = probs_np.argmax(axis=1)
            for index in range(len(labels_np)):
                rows.append(
                    {
                        "ID": str(batch["ID"][index]),
                        "patient_group_id": str(batch["patient_group_id"][index]),
                        "NYHA": int(batch["NYHA"][index]),
                        "SEX": int(batch["SEX"][index]),
                        "label_3class": int(labels_np[index]),
                        "label_3class_name": str(batch["label_3class_name"][index]),
                        "pred_class": int(predicted[index]),
                        "pred_class_name": CLASS_NAMES[int(predicted[index])],
                        "prob_normal": float(probs_np[index, 0]),
                        "prob_mild": float(probs_np[index, 1]),
                        "prob_severe": float(probs_np[index, 2]),
                        "correct": int(predicted[index] == labels_np[index]),
                        "fold": int(batch["fold"][index]),
                    }
                )
    y_true = np.concatenate(labels_all)
    probabilities = np.concatenate(probabilities_all)
    metrics = combined_metrics(y_true, probabilities)
    return total / count, metrics, rows, violations


def cutpoint_values(model: nn.Module) -> dict[str, float]:
    if not isinstance(model, MonotonicCumulativeResNet18):
        return {}
    with torch.no_grad():
        values = model.cutpoints().detach().cpu().numpy()
    return {
        "cutpoint_0": float(values[0]),
        "cutpoint_1": float(values[1]),
        "cutpoint_gap": float(values[1] - values[0]),
    }


def checkpoint_payload(
    *,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    method: str,
    config: dict[str, Any],
    outer_fold: int,
    inner_fold: int | None,
    seed: int,
    split_hash: str,
    weights: torch.Tensor,
    stage: str,
) -> dict[str, Any]:
    return {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "epoch": int(epoch),
        "method": method,
        "task.type": config["task"]["type"],
        "outer_fold": int(outer_fold),
        "inner_fold": inner_fold,
        "stage": stage,
        "seed": int(seed),
        "config": config,
        "split_sha256": split_hash,
        "loss_weights": weights.detach().cpu().tolist(),
        "ordinal_cutpoints": cutpoint_values(model),
        "git_commit": git_commit(),
    }


def load_checkpoint(path: Path, model: nn.Module, optimizer: torch.optim.Optimizer, device: torch.device) -> int:
    checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    return int(checkpoint["epoch"])


def run_inner_fold(
    *,
    method: str,
    config: dict[str, Any],
    config_path: Path,
    experiment_dir: Path,
    outer_fold: int,
    inner_fold: int,
    epochs: int,
    resume: bool,
    skip_completed: bool,
    device: torch.device,
) -> Path:
    output_dir = experiment_dir / f"outer_fold_{outer_fold}" / f"inner_fold_{inner_fold}"
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = output_dir / "training_log.csv"
    checkpoint_path = output_dir / "last_checkpoint.pth"
    summary_path = output_dir / "inner_summary.json"
    if skip_completed and log_path.is_file() and checkpoint_path.is_file() and summary_path.is_file():
        log = pd.read_csv(log_path)
        if list(log["epoch"].astype(int)) == list(range(1, epochs + 1)):
            LOGGER.info("Skipping completed inner fold %s/%s/%s", method, outer_fold, inner_fold)
            return log_path

    train_csv = PROTOCOL_DIR / f"outer_fold_{outer_fold}" / f"inner_fold_{inner_fold}_train.csv"
    val_csv = PROTOCOL_DIR / f"outer_fold_{outer_fold}" / f"inner_fold_{inner_fold}_val.csv"
    train_dataset = build_dataset(train_csv, config, train=True)
    val_dataset = build_dataset(val_csv, config, train=False)
    seed = int(config["train"]["base_seed"]) + outer_fold * 100 + inner_fold
    set_random_seed(seed)
    model = build_model(method, config).to(device)
    weights = loss_weights(method, train_dataset.labels, device)
    criterion = make_criterion(method, weights)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config["train"]["lr"]),
        weight_decay=float(config["train"]["weight_decay"]),
    )
    records: list[dict[str, Any]] = []
    start_epoch = 1
    if resume and checkpoint_path.is_file():
        completed = load_checkpoint(checkpoint_path, model, optimizer, device)
        start_epoch = completed + 1
        if log_path.is_file():
            records = pd.read_csv(log_path).to_dict("records")
            records = [row for row in records if int(row["epoch"]) <= completed]
    split_hash = sha256_file(train_csv) + ":" + sha256_file(val_csv)
    for epoch in range(start_epoch, epochs + 1):
        train_loss = train_one_epoch(
            model,
            train_dataset,
            optimizer,
            criterion,
            method,
            config,
            device,
            seed + epoch,
        )
        val_loss, metrics, _, violations = evaluate_dataset(
            model,
            val_dataset,
            criterion,
            method,
            config,
            device,
            seed=seed + 10000 + epoch,
        )
        row: dict[str, Any] = {
            "epoch": epoch,
            "train_loss": train_loss,
            "inner_val_loss": val_loss,
            "inner_val_macro_auc": metrics["macro_auc"],
            "inner_val_balanced_accuracy": metrics["balanced_accuracy"],
            "inner_val_macro_f1": metrics["macro_f1"],
            "inner_val_ordinal_mae": metrics["ordinal_mae"],
            "inner_val_extreme_error_rate": metrics["extreme_error_rate"],
            "learning_rate": optimizer.param_groups[0]["lr"],
        }
        if method == "ordinal":
            row.update(
                {
                    "ordinal_loss": train_loss,
                    **cutpoint_values(model),
                    "monotonic_violation_count": violations,
                }
            )
        records.append(row)
        pd.DataFrame(records).to_csv(log_path, index=False, encoding="utf-8-sig")
        torch.save(
            checkpoint_payload(
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                method=method,
                config=config,
                outer_fold=outer_fold,
                inner_fold=inner_fold,
                seed=seed,
                split_hash=split_hash,
                weights=weights,
                stage="inner",
            ),
            checkpoint_path,
        )
        LOGGER.info(
            "%s outer=%d inner=%d epoch=%d/%d loss=%.4f val_auc=%.4f",
            method,
            outer_fold,
            inner_fold,
            epoch,
            epochs,
            train_loss,
            metrics["macro_auc"],
        )
    log = pd.DataFrame(records)
    best_auc = pd.to_numeric(log["inner_val_macro_auc"], errors="coerce").max()
    best_epoch = int(
        log.loc[
            np.isclose(log["inner_val_macro_auc"], best_auc, atol=1e-8, rtol=0),
            "epoch",
        ].min()
    )
    write_json(
        {
            "method": method,
            "outer_fold": outer_fold,
            "inner_fold": inner_fold,
            "epochs_completed": int(len(log)),
            "best_single_fold_epoch": best_epoch,
            "best_single_fold_macro_auc": float(best_auc),
            "seed": seed,
            "train_split_sha256": sha256_file(train_csv),
            "val_split_sha256": sha256_file(val_csv),
            "config_sha256": sha256_file(config_path),
            "monotonic_violations_total": int(log.get("monotonic_violation_count", pd.Series([0])).sum()),
        },
        summary_path,
    )
    del model, optimizer, criterion
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return log_path


def select_epoch(
    *,
    method: str,
    config: dict[str, Any],
    config_path: Path,
    experiment_dir: Path,
    outer_fold: int,
    epochs: int,
) -> int:
    frames = []
    split_hashes = []
    for inner_fold in range(5):
        path = experiment_dir / f"outer_fold_{outer_fold}" / f"inner_fold_{inner_fold}" / "training_log.csv"
        frame = pd.read_csv(path)
        if list(frame["epoch"].astype(int)) != list(range(1, epochs + 1)):
            raise RuntimeError(f"inner log is incomplete: {path}")
        frame.insert(0, "inner_fold", inner_fold)
        frames.append(frame)
        base = PROTOCOL_DIR / f"outer_fold_{outer_fold}"
        split_hashes.append(
            {
                "inner_fold": inner_fold,
                "train": sha256_file(base / f"inner_fold_{inner_fold}_train.csv"),
                "val": sha256_file(base / f"inner_fold_{inner_fold}_val.csv"),
            }
        )
    all_metrics = pd.concat(frames, ignore_index=True)
    aggregated = (
        all_metrics.groupby("epoch", as_index=False)["inner_val_macro_auc"]
        .agg(["mean", "std", "count"])
        .reset_index()
        .rename(
            columns={
                "mean": "mean_inner_val_macro_auc",
                "std": "std_inner_val_macro_auc",
                "count": "valid_fold_count",
            }
        )
    )
    if not aggregated["valid_fold_count"].eq(5).all():
        raise RuntimeError("epoch aggregation does not have five valid inner folds")
    maximum = float(aggregated["mean_inner_val_macro_auc"].max())
    tolerance = float(config["selection"]["tie_tolerance"])
    tied = aggregated[
        (aggregated["mean_inner_val_macro_auc"] - maximum).abs() <= tolerance
    ]
    selected_epoch = int(tied["epoch"].min())
    selected = aggregated[aggregated["epoch"] == selected_epoch].iloc[0]
    selection_dir = experiment_dir / f"outer_fold_{outer_fold}" / "selection"
    selection_dir.mkdir(parents=True, exist_ok=True)
    all_metrics.to_csv(selection_dir / "inner_epoch_metrics_all.csv", index=False, encoding="utf-8-sig")
    aggregated.to_csv(selection_dir / "inner_epoch_metrics_aggregated.csv", index=False, encoding="utf-8-sig")
    write_json(
        {
            "method": method,
            "outer_fold": outer_fold,
            "selected_epoch": selected_epoch,
            "selected_mean_macro_auc": float(selected["mean_inner_val_macro_auc"]),
            "selected_std_macro_auc": float(selected["std_inner_val_macro_auc"]),
            "tie_tolerance": tolerance,
            "tie_count": int(len(tied)),
            "tie_breaking": "earliest_epoch",
            "inner_split_sha256": split_hashes,
            "random_seed_base": int(config["train"]["base_seed"]),
            "config_sha256": sha256_file(config_path),
        },
        selection_dir / "selected_epoch.json",
    )
    plt.figure(figsize=(8, 5))
    plt.plot(aggregated["epoch"], aggregated["mean_inner_val_macro_auc"], label="mean inner-val Macro-AUC")
    plt.fill_between(
        aggregated["epoch"].to_numpy(),
        (aggregated["mean_inner_val_macro_auc"] - aggregated["std_inner_val_macro_auc"]).to_numpy(),
        (aggregated["mean_inner_val_macro_auc"] + aggregated["std_inner_val_macro_auc"]).to_numpy(),
        alpha=0.2,
        label="±1 SD",
    )
    plt.axvline(selected_epoch, color="red", linestyle="--", label=f"selected epoch={selected_epoch}")
    plt.xlabel("Epoch")
    plt.ylabel("Macro-AUC")
    plt.ylim(0, 1)
    plt.legend()
    plt.tight_layout()
    plt.savefig(selection_dir / "selection_curve.png", dpi=180)
    plt.close()
    return selected_epoch


def run_refit_and_outer_test(
    *,
    method: str,
    config: dict[str, Any],
    experiment_dir: Path,
    outer_fold: int,
    selected_epoch: int,
    resume: bool,
    skip_completed: bool,
    device: torch.device,
) -> None:
    outer_dir = experiment_dir / f"outer_fold_{outer_fold}"
    refit_dir = outer_dir / "refit"
    prediction_dir = outer_dir / "predictions"
    metric_dir = outer_dir / "metrics"
    for path in (refit_dir, prediction_dir, metric_dir):
        path.mkdir(parents=True, exist_ok=True)
    final_path = refit_dir / "final_refit.pth"
    last_path = refit_dir / "last_checkpoint.pth"
    log_path = refit_dir / "training_log.csv"
    predictions_path = prediction_dir / "outer_test_predictions.csv"
    metrics_path = metric_dir / "outer_test_metrics.csv"
    if skip_completed and final_path.is_file() and predictions_path.is_file() and metrics_path.is_file():
        frame = pd.read_csv(predictions_path, dtype={"ID": "string"})
        if len(frame) == 100 and frame["ID"].nunique() == 100:
            LOGGER.info("Skipping completed refit/test %s outer=%d", method, outer_fold)
            return

    outer_split_dir = resolve_project_path(config["data"]["outer_split_dir"])
    train_csv = outer_split_dir / f"fold_{outer_fold}_train.csv"
    train_dataset = build_dataset(train_csv, config, train=True)
    seed = int(config["train"]["base_seed"]) + outer_fold
    set_random_seed(seed)
    model = build_model(method, config).to(device)
    weights = loss_weights(method, train_dataset.labels, device)
    criterion = make_criterion(method, weights)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config["train"]["lr"]),
        weight_decay=float(config["train"]["weight_decay"]),
    )
    records: list[dict[str, Any]] = []
    start_epoch = 1
    if resume and last_path.is_file() and not final_path.is_file():
        completed = load_checkpoint(last_path, model, optimizer, device)
        start_epoch = completed + 1
        if log_path.is_file():
            records = pd.read_csv(log_path).to_dict("records")
            records = [row for row in records if int(row["epoch"]) <= completed]
    split_hash = sha256_file(train_csv)
    for epoch in range(start_epoch, selected_epoch + 1):
        train_loss = train_one_epoch(
            model,
            train_dataset,
            optimizer,
            criterion,
            method,
            config,
            device,
            seed + epoch,
        )
        row: dict[str, Any] = {
            "epoch": epoch,
            "train_loss": train_loss,
            "learning_rate": optimizer.param_groups[0]["lr"],
        }
        if method == "ordinal":
            row.update({"ordinal_loss": train_loss, **cutpoint_values(model)})
        records.append(row)
        pd.DataFrame(records).to_csv(log_path, index=False, encoding="utf-8-sig")
        torch.save(
            checkpoint_payload(
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                method=method,
                config=config,
                outer_fold=outer_fold,
                inner_fold=None,
                seed=seed,
                split_hash=split_hash,
                weights=weights,
                stage="refit",
            ),
            last_path,
        )
        LOGGER.info(
            "%s outer=%d refit epoch=%d/%d loss=%.4f",
            method,
            outer_fold,
            epoch,
            selected_epoch,
            train_loss,
        )
    if not final_path.is_file():
        if selected_epoch < 1 or len(records) != selected_epoch:
            raise RuntimeError(f"refit did not complete selected epoch {selected_epoch}")
        torch.save(
            checkpoint_payload(
                model=model,
                optimizer=optimizer,
                epoch=selected_epoch,
                method=method,
                config=config,
                outer_fold=outer_fold,
                inner_fold=None,
                seed=seed,
                split_hash=split_hash,
                weights=weights,
                stage="refit_final",
            ),
            final_path,
        )
    else:
        checkpoint = torch.load(final_path, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])

    # The locked outer test is first loaded only after final_refit.pth exists.
    outer_test_csv = outer_split_dir / f"fold_{outer_fold}_val.csv"
    outer_test = build_dataset(outer_test_csv, config, train=False)
    test_loss, metrics, rows, violations = evaluate_dataset(
        model,
        outer_test,
        criterion,
        method,
        config,
        device,
        seed=seed + 50000,
        return_rows=True,
    )
    predictions = pd.DataFrame(rows)
    if len(predictions) != 100 or predictions["ID"].nunique() != 100:
        raise RuntimeError(f"outer test prediction coverage invalid: outer={outer_fold}")
    predictions.to_csv(predictions_path, index=False, encoding="utf-8-sig")
    metric_row = {
        "method": method,
        "outer_fold": outer_fold,
        "selected_epoch": selected_epoch,
        "outer_test_loss": test_loss,
        "monotonic_violation_count": violations,
        **metrics,
    }
    pd.DataFrame([metric_row]).to_csv(metrics_path, index=False, encoding="utf-8-sig")
    matrix = confusion_matrix(
        predictions["label_3class"], predictions["pred_class"], labels=[0, 1, 2]
    )
    pd.DataFrame(
        matrix,
        index=[CLASS_NAMES[i] for i in range(3)],
        columns=[CLASS_NAMES[i] for i in range(3)],
    ).to_csv(metric_dir / "confusion_matrix.csv", index_label="true\\pred", encoding="utf-8-sig")
    plt.figure(figsize=(5, 4))
    plt.imshow(matrix, cmap="Blues")
    plt.colorbar()
    plt.xticks(range(3), [CLASS_NAMES[i] for i in range(3)])
    plt.yticks(range(3), [CLASS_NAMES[i] for i in range(3)])
    for i in range(3):
        for j in range(3):
            plt.text(j, i, str(int(matrix[i, j])), ha="center", va="center")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    plt.savefig(metric_dir / "confusion_matrix.png", dpi=180)
    plt.close()
    write_json(
        {
            "method": method,
            "outer_fold": outer_fold,
            "selected_epoch": selected_epoch,
            "epochs_completed": len(records),
            "full_outer_train_n": len(train_dataset),
            "outer_test_n": len(outer_test),
            "outer_train_split_sha256": split_hash,
            "outer_test_split_sha256": sha256_file(outer_test_csv),
            "loss_weights": weights.detach().cpu().tolist(),
            "final_cutpoints": cutpoint_values(model),
            "monotonic_violation_count": violations,
            "outer_test_evaluated_after_final_refit": True,
        },
        refit_dir / "refit_summary.json",
    )
    del model, optimizer, criterion
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def run_outer(
    *,
    method: str,
    config: dict[str, Any],
    config_path: Path,
    experiment_dir: Path,
    outer_fold: int,
    inner_fold: int | None,
    epochs: int,
    resume: bool,
    skip_completed: bool,
    device: torch.device,
) -> None:
    folds = [inner_fold] if inner_fold is not None else list(range(5))
    for current_inner in folds:
        run_inner_fold(
            method=method,
            config=config,
            config_path=config_path,
            experiment_dir=experiment_dir,
            outer_fold=outer_fold,
            inner_fold=current_inner,
            epochs=epochs,
            resume=resume,
            skip_completed=skip_completed,
            device=device,
        )
    if inner_fold is not None:
        return
    selected = select_epoch(
        method=method,
        config=config,
        config_path=config_path,
        experiment_dir=experiment_dir,
        outer_fold=outer_fold,
        epochs=epochs,
    )
    run_refit_and_outer_test(
        method=method,
        config=config,
        experiment_dir=experiment_dir,
        outer_fold=outer_fold,
        selected_epoch=selected,
        resume=resume,
        skip_completed=skip_completed,
        device=device,
    )


def main() -> int:
    args = parse_args()
    config_path = (args.config or CONFIGS[args.method]).resolve()
    config = load_config(config_path)
    if str(config["task"]["method"]) != args.method:
        raise ValueError("--method does not match config task.method")
    outer_split_dir = resolve_project_path(config["data"]["outer_split_dir"])
    generate_shared_protocol(outer_split_dir, PROTOCOL_DIR, base_seed=2026)
    audit_shared_protocol(outer_split_dir, PROTOCOL_DIR)
    if args.protocol_only:
        print(f"PROTOCOL_DIR={PROTOCOL_DIR}")
        print("PROTOCOL_AUDIT=PASS")
        return 0
    experiment_dir = method_experiment_dir(config, args.smoke_test)
    experiment_dir.mkdir(parents=True, exist_ok=True)
    configure_logging(experiment_dir / "run.log")
    (experiment_dir / "config.yaml").write_text(
        yaml.safe_dump(config, sort_keys=False, allow_unicode=True), encoding="utf-8"
    )
    epochs = 1 if args.smoke_test else int(config["train"]["inner_max_epochs"])
    outer_folds = args.outer_folds or list(range(5))
    if any(fold not in range(5) for fold in outer_folds):
        raise ValueError("outer folds must be in 0..4")
    if args.inner_fold is not None and args.inner_fold not in range(5):
        raise ValueError("inner fold must be in 0..4")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    LOGGER.info("method=%s device=%s epochs=%d smoke=%s", args.method, device, epochs, args.smoke_test)
    for outer_fold in outer_folds:
        run_outer(
            method=args.method,
            config=config,
            config_path=config_path,
            experiment_dir=experiment_dir,
            outer_fold=outer_fold,
            inner_fold=args.inner_fold,
            epochs=epochs,
            resume=args.resume,
            skip_completed=args.skip_completed,
            device=device,
        )
    print(f"EXPERIMENT_DIR={experiment_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
