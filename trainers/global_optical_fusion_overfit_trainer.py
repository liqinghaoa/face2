"""Strict trainer for frozen-backbone and partial-layer4 overfitting controls."""

from __future__ import annotations

import math
import platform
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torchvision
from torch import nn
from torch.utils.data import DataLoader

from metrics.classification_metrics import CLASS_NAMES, compute_classification_metrics
from trainers.global_optical_fusion_trainer import (
    capture_rng_state,
    epoch_augmentation_seed,
    load_torch_checkpoint,
    restore_rng_state,
)
from utils.experiment_utils import set_random_seed
from utils.resnet_trainability import (
    assert_optimizer_contract,
    assert_trainability_contract,
    assert_training_mode_contract,
    batchnorm_training_records,
    enforce_trainability_modes,
    optimizer_group_records,
    validate_trainability_strategy,
)


CHECKPOINT_IDENTITY_FIELDS = (
    "task",
    "strategy",
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
    "trainability_trainable_parameter_names",
    "trainability_frozen_parameter_names",
    "trainability_optimizer_groups",
    "trainability_batchnorm_training_modes",
)


def validate_overfit_checkpoint_metadata(
    checkpoint: Mapping[str, Any], expected: Mapping[str, Any]
) -> None:
    """Reject resume/evaluation when any immutable experiment field differs."""
    for key in CHECKPOINT_IDENTITY_FIELDS:
        if key not in checkpoint:
            raise ValueError(f"Checkpoint is missing identity field {key!r}")
        if key not in expected:
            raise ValueError(f"Expected metadata is missing identity field {key!r}")
        if checkpoint[key] != expected[key]:
            raise ValueError(
                f"Checkpoint {key} mismatch: checkpoint={checkpoint[key]!r}, "
                f"expected={expected[key]!r}"
            )


def _clone_batchnorm_state(model: nn.Module, strategy: str) -> dict[str, dict[str, torch.Tensor]]:
    state: dict[str, dict[str, torch.Tensor]] = {}
    for name, module in model.named_modules():
        if not isinstance(module, nn.modules.batchnorm._BatchNorm):
            continue
        should_freeze = strategy == "frozen_backbone" or not name.startswith("backbone.layer4.")
        if should_freeze:
            state[name] = {
                key: value.detach().cpu().clone()
                for key, value in module.state_dict().items()
                if torch.is_tensor(value)
            }
    return state


def _assert_batchnorm_state_unchanged(
    model: nn.Module, expected: Mapping[str, Mapping[str, torch.Tensor]]
) -> None:
    modules = dict(model.named_modules())
    for name, before in expected.items():
        after = modules[name].state_dict()
        for key, value in before.items():
            if not torch.equal(value, after[key].detach().cpu()):
                raise AssertionError(f"Frozen BatchNorm state changed: {name}.{key}")


class GlobalOpticalFusionOverfitTrainer:
    """Train one locked strategy and retain the earliest strict-best val Macro-AUC."""

    def __init__(
        self,
        model: nn.Module,
        criterion: nn.Module,
        optimizer: torch.optim.Optimizer,
        device: torch.device,
        output_dir: str | Path,
        *,
        strategy: str,
        variant: str,
        fold: int,
        metadata: Mapping[str, Any],
        seed_info: Mapping[str, int],
        epochs: int = 50,
        patience: int = 10,
        minimum_improvement: float = 0.0,
        resume_from: str | Path | None = None,
    ) -> None:
        self.strategy = validate_trainability_strategy(strategy)
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
        assert_trainability_contract(self.model, self.strategy)
        assert_optimizer_contract(self.model, self.optimizer, self.strategy)
        self.frozen_batchnorm_state = _clone_batchnorm_state(self.model, self.strategy)

    def _loss_normalizer(self, labels: torch.Tensor) -> float:
        if (
            isinstance(self.criterion, nn.CrossEntropyLoss)
            and self.criterion.reduction == "mean"
            and self.criterion.weight is not None
        ):
            return float(
                self.criterion.weight.to(labels.device)
                .index_select(0, labels.long())
                .sum()
                .item()
            )
        return float(labels.shape[0])

    def _assert_gradients(self) -> None:
        for name, parameter in self.model.named_parameters():
            if parameter.requires_grad and parameter.grad is None:
                raise AssertionError(f"Trainable parameter has no gradient: {name}")
            if not parameter.requires_grad and parameter.grad is not None:
                raise AssertionError(f"Frozen parameter unexpectedly has a gradient: {name}")

    def _run_epoch(
        self, loader: DataLoader, *, training: bool
    ) -> tuple[float, dict[str, Any]]:
        self.model.train(training)
        enforce_trainability_modes(self.model, self.strategy, training=training)
        if training:
            assert_trainability_contract(self.model, self.strategy)
            assert_optimizer_contract(self.model, self.optimizer, self.strategy)
            assert_training_mode_contract(self.model, self.strategy)
        loss_total = 0.0
        normalizer_total = 0.0
        labels_all: list[np.ndarray] = []
        probabilities_all: list[np.ndarray] = []
        context = torch.enable_grad() if training else torch.inference_mode()
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
                    self._assert_gradients()
                    self.optimizer.step()
                normalizer = self._loss_normalizer(labels)
                loss_total += float(loss.item()) * normalizer
                normalizer_total += normalizer
                labels_all.append(labels.detach().cpu().numpy())
                probabilities_all.append(torch.softmax(logits.detach(), dim=1).cpu().numpy())
        if training:
            _assert_batchnorm_state_unchanged(self.model, self.frozen_batchnorm_state)
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
                "optimizer_groups_at_save": optimizer_group_records(self.model, self.optimizer),
                "epoch": int(epoch),
                "best_epoch": int(best_epoch),
                "best_macro_auc": float(best_macro_auc),
                "patience_counter": int(patience_counter),
                "strategy": self.strategy,
                "variant": self.variant,
                "fold": self.fold,
                "rng_state": capture_rng_state(train_loader),
                "seed_info": self.seed_info,
                "python_version": platform.python_version(),
                "pytorch_version": torch.__version__,
                "torchvision_version": torchvision.__version__,
                "cuda_version": torch.version.cuda,
                "cudnn_version": torch.backends.cudnn.version(),
                "device": str(self.device),
                "batchnorm_modes_at_save": batchnorm_training_records(self.model),
                "clinical_fields_used": False,
                "camera_used": False,
                "exif_used": False,
                "outer_validation_tuning": True,
                "historical_inputs_modified": False,
                "initialization_source": "independent_imagenet_resnet18",
                "initialized_from_full_checkpoint": False,
            }
        )
        return payload

    def _load_resume(
        self, train_loader: DataLoader
    ) -> tuple[int, int, float, int, list[dict[str, Any]]]:
        if self.resume_from is None:
            return 1, 0, float("-inf"), 0, []
        checkpoint = load_torch_checkpoint(self.resume_from, self.device)
        validate_overfit_checkpoint_metadata(checkpoint, self.metadata)
        if checkpoint.get("optimizer_groups_at_save") != optimizer_group_records(
            self.model, self.optimizer
        ):
            raise ValueError("Resume optimizer group structure does not match current protocol")
        self.model.load_state_dict(checkpoint["model_state_dict"], strict=True)
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        assert_optimizer_contract(self.model, self.optimizer, self.strategy)
        restore_rng_state(checkpoint["rng_state"], train_loader)
        self.frozen_batchnorm_state = _clone_batchnorm_state(self.model, self.strategy)
        completed = int(checkpoint["epoch"])
        records: list[dict[str, Any]] = []
        history_path = self.output_dir / "training_log.csv"
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

    @staticmethod
    def _metric_columns(prefix: str, metrics: Mapping[str, Any]) -> dict[str, float]:
        names = ("accuracy", "macro_auc", "macro_f1", "balanced_accuracy")
        result = {f"{prefix}_{name}": float(metrics[name]) for name in names}
        for class_name in CLASS_NAMES.values():
            result[f"{prefix}_auc_{class_name}"] = float(metrics[f"auc_{class_name}"])
            result[f"{prefix}_recall_{class_name}"] = float(metrics[f"recall_{class_name}"])
        return result

    def _save_curves(self, history: pd.DataFrame) -> None:
        if history.empty:
            return
        figure, axes = plt.subplots(1, 3, figsize=(16, 4.5))
        axes[0].plot(history["epoch"], history["train_loss"], label="train")
        axes[0].plot(history["epoch"], history["val_loss"], label="val")
        axes[0].set(xlabel="Epoch", ylabel="Loss", title="Weighted CE")
        axes[0].legend()
        axes[1].plot(history["epoch"], history["train_macro_auc"], label="train")
        axes[1].plot(history["epoch"], history["val_macro_auc"], label="val")
        axes[1].set(xlabel="Epoch", ylabel="Macro-AUC", ylim=(0, 1), title="Generalization")
        axes[1].legend()
        for class_name in CLASS_NAMES.values():
            axes[2].plot(
                history["epoch"], history[f"val_recall_{class_name}"], label=class_name
            )
        axes[2].set(xlabel="Epoch", ylabel="Recall", ylim=(0, 1), title="Validation recall")
        axes[2].legend()
        figure.tight_layout()
        figure.savefig(self.output_dir / "training_curves.png", dpi=180)
        plt.close(figure)

    def fit(self, train_loader: DataLoader, val_loader: DataLoader) -> pd.DataFrame:
        start_epoch, best_epoch, best_auc, counter, records = self._load_resume(train_loader)
        for epoch in range(start_epoch, self.epochs + 1):
            started = time.perf_counter()
            set_random_seed(epoch_augmentation_seed(self.seed_info, epoch))
            if self.device.type == "cuda":
                torch.cuda.reset_peak_memory_stats(self.device)
            train_loss, train_metrics = self._run_epoch(train_loader, training=True)
            val_loss, val_metrics = self._run_epoch(val_loader, training=False)
            macro_auc = float(val_metrics["macro_auc"])
            improved = math.isfinite(macro_auc) and macro_auc > best_auc + self.minimum_improvement
            if improved:
                best_auc, best_epoch, counter = macro_auc, epoch, 0
            else:
                counter += 1
            group_lrs = {
                str(group.get("name", f"group_{index}")): float(group["lr"])
                for index, group in enumerate(self.optimizer.param_groups)
            }
            record: dict[str, Any] = {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_loss,
                **self._metric_columns("train", train_metrics),
                **self._metric_columns("val", val_metrics),
                "classifier_learning_rate": group_lrs["classifier"],
                "layer4_learning_rate": group_lrs.get("layer4"),
                "trainable_parameter_count": int(
                    sum(parameter.numel() for parameter in self.model.parameters() if parameter.requires_grad)
                ),
                "peak_gpu_memory_bytes": (
                    int(torch.cuda.max_memory_allocated(self.device))
                    if self.device.type == "cuda"
                    else 0
                ),
                "elapsed_seconds": time.perf_counter() - started,
                "is_best": int(improved),
                "patience_counter": counter,
            }
            records.append(record)
            pd.DataFrame(records).to_csv(
                self.output_dir / "training_log.csv", index=False, encoding="utf-8-sig"
            )
            checkpoint = self._checkpoint_payload(
                epoch=epoch,
                best_epoch=best_epoch,
                best_macro_auc=best_auc,
                patience_counter=counter,
                train_loader=train_loader,
            )
            torch.save(checkpoint, self.output_dir / "last_checkpoint.pth")
            if improved:
                torch.save(checkpoint, self.output_dir / "best_macro_auc.pth")
            if counter >= self.patience:
                break
        history = pd.DataFrame(records)
        self._save_curves(history)
        if not (self.output_dir / "best_macro_auc.pth").is_file():
            raise RuntimeError("No finite validation Macro-AUC was available for selection")
        return history
