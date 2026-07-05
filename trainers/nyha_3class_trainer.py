"""Single-fold trainer for NYHA three-class classification."""

from __future__ import annotations

import logging
import math
import shutil
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader

from metrics.classification_metrics import compute_classification_metrics


LOGGER = logging.getLogger(__name__)


class NYHA3ClassTrainer:
    """Train one held-out validation fold and checkpoint by validation macro-AUC."""

    def __init__(
        self,
        model: nn.Module,
        criterion: nn.Module,
        optimizer: torch.optim.Optimizer,
        device: torch.device,
        output_dir: str | Path,
        epochs: int = 50,
        early_stopping_patience: int = 10,
        use_amp: bool = False,
        fold: int | None = None,
        config: dict[str, Any] | None = None,
        resume_from: str | Path | None = None,
    ) -> None:
        self.model = model.to(device)
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device
        self.output_dir = Path(output_dir)
        self.epochs = int(epochs)
        self.patience = int(early_stopping_patience)
        self.use_amp = bool(use_amp and device.type == "cuda")
        self.fold = fold
        self.config = config or {}
        self.resume_from = Path(resume_from) if resume_from is not None else None

        self.checkpoint_dir = self.output_dir / "checkpoints"
        self.log_dir = self.output_dir / "logs"
        self.curve_dir = self.output_dir / "curves"
        for directory in (self.checkpoint_dir, self.log_dir, self.curve_dir):
            directory.mkdir(parents=True, exist_ok=True)

        try:
            self.scaler = torch.amp.GradScaler(
                self.device.type, enabled=self.use_amp
            )
        except (AttributeError, TypeError):
            self.scaler = torch.cuda.amp.GradScaler(enabled=self.use_amp)

    def _autocast(self):
        try:
            return torch.amp.autocast(
                device_type=self.device.type, enabled=self.use_amp
            )
        except AttributeError:
            return torch.cuda.amp.autocast(enabled=self.use_amp)

    def _train_epoch(self, loader: DataLoader) -> float:
        self.model.train()
        running_loss = 0.0
        sample_count = 0
        for batch in loader:
            images = batch["image"].to(self.device, non_blocking=True)
            labels = batch["label"].to(self.device, non_blocking=True)
            self.optimizer.zero_grad(set_to_none=True)
            with self._autocast():
                logits = self.model(images)
                loss = self.criterion(logits, labels)
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()

            batch_size = labels.size(0)
            running_loss += float(loss.detach().item()) * batch_size
            sample_count += batch_size
        return running_loss / max(sample_count, 1)

    @torch.no_grad()
    def _validate(self, loader: DataLoader) -> tuple[float, dict[str, Any]]:
        self.model.eval()
        running_loss = 0.0
        sample_count = 0
        labels_all: list[np.ndarray] = []
        probabilities_all: list[np.ndarray] = []
        for batch in loader:
            images = batch["image"].to(self.device, non_blocking=True)
            labels = batch["label"].to(self.device, non_blocking=True)
            with self._autocast():
                logits = self.model(images)
                loss = self.criterion(logits, labels)
            probabilities = torch.softmax(logits, dim=1)

            batch_size = labels.size(0)
            running_loss += float(loss.item()) * batch_size
            sample_count += batch_size
            labels_all.append(labels.cpu().numpy())
            probabilities_all.append(probabilities.cpu().numpy())

        metrics = compute_classification_metrics(
            np.concatenate(labels_all), np.concatenate(probabilities_all)
        )
        return running_loss / max(sample_count, 1), metrics

    def _checkpoint_payload(
        self, epoch: int, best_macro_auc: float
    ) -> dict[str, Any]:
        return {
            "epoch": epoch,
            "fold": self.fold,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "best_macro_auc": best_macro_auc,
            "config": self.config,
        }

    def _save_curves(self, history: pd.DataFrame) -> None:
        if history.empty:
            return
        plt.figure(figsize=(7, 5))
        plt.plot(history["epoch"], history["train_loss"], label="train_loss")
        plt.plot(history["epoch"], history["val_loss"], label="val_loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()
        plt.tight_layout()
        plt.savefig(self.curve_dir / "loss_curve.png", dpi=160)
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
        plt.savefig(self.curve_dir / "metrics_curve.png", dpi=160)
        plt.close()

    def _load_resume_state(self) -> tuple[int, float, int, list[dict[str, float | int]]]:
        if self.resume_from is None:
            return 1, float("-inf"), 0, []
        if not self.resume_from.is_file():
            raise FileNotFoundError(f"Resume checkpoint does not exist: {self.resume_from}")

        checkpoint_path = self.resume_from
        try:
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
        except Exception as error:
            fallback_path = checkpoint_path.with_name("best_macro_auc.pth")
            if checkpoint_path.name != "last.pth" or not fallback_path.is_file():
                raise
            LOGGER.warning(
                "Failed to load resume checkpoint %s (%s). Falling back to %s.",
                checkpoint_path,
                error,
                fallback_path,
            )
            checkpoint_path = fallback_path
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        completed_epoch = int(checkpoint.get("epoch", 0))
        best_macro_auc = float(checkpoint.get("best_macro_auc", float("-inf")))
        history_path = self.log_dir / "train_log.csv"
        records: list[dict[str, float | int]] = []
        if history_path.is_file():
            history = pd.read_csv(history_path)
            history = history[pd.to_numeric(history["epoch"], errors="coerce") <= completed_epoch]
            records = history.to_dict("records")
            if "val_macro_auc" in history and not history.empty:
                best_macro_auc = float(pd.to_numeric(history["val_macro_auc"]).max())
                best_epoch = int(
                    history.loc[
                        pd.to_numeric(history["val_macro_auc"]).idxmax(), "epoch"
                    ]
                )
                epochs_without_improvement = max(completed_epoch - best_epoch, 0)
            else:
                epochs_without_improvement = 0
        else:
            epochs_without_improvement = 0

        start_epoch = completed_epoch + 1
        LOGGER.info(
            "Resuming fold=%s from %s at epoch=%d with best_macro_auc=%s "
            "and patience=%d/%d",
            self.fold,
            checkpoint_path,
            start_epoch,
            f"{best_macro_auc:.4f}" if math.isfinite(best_macro_auc) else "unavailable",
            epochs_without_improvement,
            self.patience,
        )
        return start_epoch, best_macro_auc, epochs_without_improvement, records

    def fit(
        self, train_loader: DataLoader, val_loader: DataLoader
    ) -> pd.DataFrame:
        (
            start_epoch,
            best_macro_auc,
            epochs_without_improvement,
            records,
        ) = self._load_resume_state()

        LOGGER.info(
            "Starting fold=%s on device=%s, AMP=%s",
            self.fold,
            self.device,
            self.use_amp,
        )
        if start_epoch > self.epochs:
            LOGGER.info(
                "Fold=%s already reached configured epochs=%d; skipping training loop.",
                self.fold,
                self.epochs,
            )
            history = pd.DataFrame(records)
            self._save_curves(history)
            return history

        for epoch in range(start_epoch, self.epochs + 1):
            train_loss = self._train_epoch(train_loader)
            val_loss, val_metrics = self._validate(val_loader)
            macro_auc = float(val_metrics["macro_auc"])
            record: dict[str, float | int] = {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "val_macro_auc": macro_auc,
                "val_accuracy": float(val_metrics["accuracy"]),
                "val_macro_precision": float(val_metrics["macro_precision"]),
                "val_macro_recall": float(val_metrics["macro_recall"]),
                "val_macro_f1": float(val_metrics["macro_f1"]),
                "val_balanced_accuracy": float(
                    val_metrics["balanced_accuracy"]
                ),
                "lr": float(self.optimizer.param_groups[0]["lr"]),
            }
            records.append(record)
            history = pd.DataFrame(records)
            history.to_csv(
                self.log_dir / "train_log.csv",
                index=False,
                encoding="utf-8-sig",
            )

            improved = not math.isnan(macro_auc) and macro_auc > best_macro_auc
            if improved:
                best_macro_auc = macro_auc
                epochs_without_improvement = 0
                torch.save(
                    self._checkpoint_payload(epoch, best_macro_auc),
                    self.checkpoint_dir / "best_macro_auc.pth",
                )
            else:
                epochs_without_improvement += 1

            torch.save(
                self._checkpoint_payload(epoch, best_macro_auc),
                self.checkpoint_dir / "last.pth",
            )
            LOGGER.info(
                "fold=%s epoch=%d/%d train_loss=%.5f val_loss=%.5f "
                "val_macro_auc=%s val_macro_f1=%.4f best=%s patience=%d/%d",
                self.fold,
                epoch,
                self.epochs,
                train_loss,
                val_loss,
                f"{macro_auc:.4f}" if not math.isnan(macro_auc) else "nan",
                val_metrics["macro_f1"],
                (
                    f"{best_macro_auc:.4f}"
                    if math.isfinite(best_macro_auc)
                    else "unavailable"
                ),
                epochs_without_improvement,
                self.patience,
            )
            if epochs_without_improvement >= self.patience:
                LOGGER.info("Early stopping fold=%s at epoch=%d", self.fold, epoch)
                break

        history = pd.DataFrame(records)
        self._save_curves(history)
        best_path = self.checkpoint_dir / "best_macro_auc.pth"
        if not best_path.is_file():
            last_path = self.checkpoint_dir / "last.pth"
            LOGGER.warning(
                "Validation macro-AUC was NaN for every epoch in fold=%s. "
                "No checkpoint was selected by macro-AUC; copying last.pth to "
                "best_macro_auc.pth only so evaluation artifacts can still be produced.",
                self.fold,
            )
            shutil.copy2(last_path, best_path)
        return history
