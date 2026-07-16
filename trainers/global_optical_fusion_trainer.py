"""Independent trainer and strict checkpoint protocol for optical fusion."""

from __future__ import annotations

import math
import platform
import random
import time
from pathlib import Path
from typing import Any, Mapping

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torchvision
from torch import nn
from torch.utils.data import DataLoader

from metrics.classification_metrics import compute_classification_metrics
from utils.experiment_utils import set_random_seed


def load_torch_checkpoint(path: str | Path, device: torch.device | str = "cpu") -> dict[str, Any]:
    checkpoint_path = Path(path)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint does not exist: {checkpoint_path}")
    try:
        return torch.load(checkpoint_path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(checkpoint_path, map_location=device)


def validate_checkpoint_metadata(
    checkpoint: Mapping[str, Any], expected: Mapping[str, Any]
) -> None:
    """Reject restoration if any experiment-identity field differs."""
    required = (
        "variant",
        "fold",
        "feature_names",
        "feature_scaler_sha256",
        "split_sha256",
        "feature_source_sha256",
        "feature_schema_sha256",
        "upstream_manifest_sha256",
        "train_id_sha256",
        "val_id_sha256",
        "config_sha256",
        "implementation_signature",
        "auxiliary_input_dim",
        "fused_input_dim",
    )
    for key in required:
        if key not in checkpoint:
            raise ValueError(f"Checkpoint is missing identity field {key!r}")
        if key not in expected:
            raise ValueError(f"Expected metadata is missing identity field {key!r}")
        if checkpoint[key] != expected[key]:
            raise ValueError(
                f"Checkpoint {key} mismatch: checkpoint={checkpoint[key]!r}, "
                f"expected={expected[key]!r}"
            )


def capture_rng_state(train_loader: DataLoader) -> dict[str, Any]:
    generator = getattr(train_loader, "generator", None)
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        "train_loader_generator": generator.get_state() if generator is not None else None,
    }


def restore_rng_state(state: Mapping[str, Any], train_loader: DataLoader) -> None:
    if state.get("python") is not None:
        random.setstate(state["python"])
    if state.get("numpy") is not None:
        np.random.set_state(state["numpy"])
    if state.get("torch_cpu") is not None:
        torch.set_rng_state(state["torch_cpu"].cpu())
    if state.get("torch_cuda") is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all([item.cpu() for item in state["torch_cuda"]])
    generator = getattr(train_loader, "generator", None)
    if generator is not None and state.get("train_loader_generator") is not None:
        generator.set_state(state["train_loader_generator"].cpu())


def seed_payload(base_seed: int, fold: int) -> dict[str, int]:
    fold_seed = int(base_seed) + int(fold)
    return {
        "base_seed": int(base_seed),
        "fold_seed": fold_seed,
        "model_seed": fold_seed,
        "shuffle_seed": fold_seed + 10_000,
        "augmentation_seed": fold_seed + 20_000,
        "validation_seed": fold_seed + 30_000,
        "python_seed": fold_seed + 20_000,
        "numpy_seed": fold_seed + 20_000,
        "torch_cpu_seed": fold_seed + 20_000,
        "torch_cuda_seed": fold_seed + 20_000,
    }


def make_data_generator(seed: int) -> torch.Generator:
    generator = torch.Generator()
    generator.manual_seed(int(seed))
    return generator


def epoch_augmentation_seed(seed_info: Mapping[str, int], epoch: int) -> int:
    return int(seed_info["augmentation_seed"]) + int(epoch)


class GlobalOpticalFusionTrainer:
    """Train one variant/fold and select the earliest strict-best val macro-AUC."""

    def __init__(
        self,
        model: nn.Module,
        criterion: nn.Module,
        optimizer: torch.optim.Optimizer,
        device: torch.device,
        output_dir: str | Path,
        *,
        variant: str,
        fold: int,
        metadata: Mapping[str, Any],
        seed_info: Mapping[str, int],
        epochs: int = 50,
        patience: int = 10,
        minimum_improvement: float = 0.0,
        resume_from: str | Path | None = None,
    ) -> None:
        self.model = model.to(device)
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.variant = str(variant)
        self.fold = int(fold)
        self.metadata = dict(metadata)
        self.seed_info = dict(seed_info)
        self.epochs = int(epochs)
        self.patience = int(patience)
        self.minimum_improvement = float(minimum_improvement)
        self.resume_from = Path(resume_from) if resume_from else None

    def _loss_normalizer(self, labels: torch.Tensor) -> float:
        if (
            isinstance(self.criterion, nn.CrossEntropyLoss)
            and self.criterion.reduction == "mean"
            and self.criterion.weight is not None
        ):
            weights = self.criterion.weight.to(labels.device)
            return float(weights.index_select(0, labels.long()).sum().item())
        return float(labels.shape[0])

    def _run_epoch(
        self, loader: DataLoader, *, training: bool
    ) -> tuple[float, dict[str, Any]]:
        self.model.train(training)
        loss_total = 0.0
        normalizer_total = 0.0
        labels_all: list[np.ndarray] = []
        probabilities_all: list[np.ndarray] = []
        context = torch.enable_grad() if training else torch.no_grad()
        with context:
            for batch in loader:
                images = batch["image"].to(self.device, non_blocking=True)
                aux = batch["aux_features"].to(self.device, non_blocking=True)
                labels = batch["label"].to(self.device, non_blocking=True).long()
                if training:
                    self.optimizer.zero_grad(set_to_none=True)
                logits = self.model(images, aux)
                loss = self.criterion(logits, labels)
                if training:
                    loss.backward()
                    self.optimizer.step()
                normalizer = self._loss_normalizer(labels)
                loss_total += float(loss.item()) * normalizer
                normalizer_total += normalizer
                labels_all.append(labels.detach().cpu().numpy())
                probabilities_all.append(torch.softmax(logits.detach(), dim=1).cpu().numpy())
        metrics = compute_classification_metrics(
            np.concatenate(labels_all), np.concatenate(probabilities_all)
        )
        return loss_total / max(normalizer_total, 1.0), metrics

    def _checkpoint_payload(
        self,
        *,
        epoch: int,
        best_epoch: int,
        best_macro_auc: float,
        patience_counter: int,
        train_loader: DataLoader,
    ) -> dict[str, Any]:
        payload = dict(self.metadata)
        payload.update(
            {
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "epoch": int(epoch),
                "best_epoch": int(best_epoch),
                "best_macro_auc": float(best_macro_auc),
                "patience_counter": int(patience_counter),
                "variant": self.variant,
                "fold": self.fold,
                "rng_state": capture_rng_state(train_loader),
                "seed_info": self.seed_info,
                "python_version": platform.python_version(),
                "pytorch_version": torch.__version__,
                "torchvision_version": torchvision.__version__,
                "cuda_version": torch.version.cuda,
                "device": str(self.device),
                "clinical_fields_used": False,
                "camera_used": False,
                "exif_used": False,
                "outer_validation_tuning": True,
                "historical_inputs_modified": False,
            }
        )
        return payload

    def _load_resume(
        self, train_loader: DataLoader
    ) -> tuple[int, int, float, int, list[dict[str, Any]]]:
        if self.resume_from is None:
            return 1, 0, float("-inf"), 0, []
        checkpoint = load_torch_checkpoint(self.resume_from, self.device)
        validate_checkpoint_metadata(checkpoint, self.metadata)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        restore_rng_state(checkpoint["rng_state"], train_loader)
        completed = int(checkpoint["epoch"])
        history_path = self.output_dir / "training_log.csv"
        records: list[dict[str, Any]] = []
        if history_path.is_file():
            history = pd.read_csv(history_path)
            records = history.loc[history["epoch"] <= completed].to_dict("records")
        return (
            completed + 1,
            int(checkpoint["best_epoch"]),
            float(checkpoint["best_macro_auc"]),
            int(checkpoint["patience_counter"]),
            records,
        )

    def _save_curves(self, history: pd.DataFrame) -> None:
        if history.empty:
            return
        figure, axes = plt.subplots(1, 2, figsize=(12, 4.5))
        axes[0].plot(history["epoch"], history["train_loss"], label="train")
        axes[0].plot(history["epoch"], history["val_loss"], label="val")
        axes[0].set(xlabel="Epoch", ylabel="Loss", title="Weighted CE")
        axes[0].legend()
        for name in ("val_macro_auc", "val_accuracy", "val_macro_f1", "val_balanced_accuracy"):
            axes[1].plot(history["epoch"], history[name], label=name)
        axes[1].set(xlabel="Epoch", ylabel="Metric", ylim=(0, 1), title="Validation metrics")
        axes[1].legend(fontsize=8)
        figure.tight_layout()
        figure.savefig(self.output_dir / "training_curves.png", dpi=180)
        plt.close(figure)

    def fit(self, train_loader: DataLoader, val_loader: DataLoader) -> pd.DataFrame:
        start_epoch, best_epoch, best_auc, counter, records = self._load_resume(train_loader)
        for epoch in range(start_epoch, self.epochs + 1):
            started = time.perf_counter()
            set_random_seed(epoch_augmentation_seed(self.seed_info, epoch))
            train_loss, train_metrics = self._run_epoch(train_loader, training=True)
            val_loss, val_metrics = self._run_epoch(val_loader, training=False)
            macro_auc = float(val_metrics["macro_auc"])
            improved = math.isfinite(macro_auc) and macro_auc > best_auc + self.minimum_improvement
            if improved:
                best_auc, best_epoch, counter = macro_auc, epoch, 0
            else:
                counter += 1
            record = {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "train_accuracy": float(train_metrics["accuracy"]),
                "val_accuracy": float(val_metrics["accuracy"]),
                "train_macro_auc": float(train_metrics["macro_auc"]),
                "val_macro_auc": macro_auc,
                "val_balanced_accuracy": float(val_metrics["balanced_accuracy"]),
                "val_macro_f1": float(val_metrics["macro_f1"]),
                "learning_rate": float(self.optimizer.param_groups[0]["lr"]),
                "elapsed_seconds": time.perf_counter() - started,
                "is_best": int(improved),
                "patience_counter": counter,
            }
            records.append(record)
            pd.DataFrame(records).to_csv(
                self.output_dir / "training_log.csv", index=False, encoding="utf-8-sig"
            )
            checkpoint = self._checkpoint_payload(
                epoch=epoch, best_epoch=best_epoch, best_macro_auc=best_auc,
                patience_counter=counter, train_loader=train_loader,
            )
            torch.save(checkpoint, self.output_dir / "last_checkpoint.pth")
            if improved:
                torch.save(checkpoint, self.output_dir / "best_macro_auc.pth")
            if counter >= self.patience:
                break
        history = pd.DataFrame(records)
        self._save_curves(history)
        best_path = self.output_dir / "best_macro_auc.pth"
        if not best_path.is_file():
            raise RuntimeError("No finite validation macro-AUC was available for checkpoint selection")
        return history
