"""Trainer for the E1 Global ResNet18 conditional hierarchical experiment."""

from __future__ import annotations

import logging
import math
import random
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader

from metrics.e1_hierarchical_metrics import compute_e1_metrics, restore_three_class_probabilities


LOGGER = logging.getLogger(__name__)


class E1HierarchicalConditionalTrainer:
    """Train one fixed validation fold and select by restored 3-class macro-AUC."""

    def __init__(
        self, model: nn.Module, criterion: nn.Module, optimizer: torch.optim.Optimizer,
        device: torch.device, output_dir: str | Path, epochs: int, patience: int,
        use_amp: bool, fold: int, config: dict[str, Any], class_weights: dict[str, Any],
        ece_bins: int = 15, resume_from: str | Path | None = None,
    ) -> None:
        self.model, self.criterion, self.optimizer = model.to(device), criterion, optimizer
        self.device, self.epochs, self.patience = device, int(epochs), int(patience)
        self.use_amp, self.fold, self.config = bool(use_amp and device.type == "cuda"), fold, config
        self.class_weights = class_weights
        self.ece_bins = int(ece_bins)
        if self.ece_bins < 1:
            raise ValueError("E1 ECE requires at least one bin")
        self.resume_from = Path(resume_from) if resume_from is not None else None
        self.checkpoint_dir = Path(output_dir) / "checkpoints"
        self.log_dir = Path(output_dir) / "logs"
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        try:
            self.scaler = torch.amp.GradScaler(device.type, enabled=self.use_amp)
        except (AttributeError, TypeError):  # compatibility with older torch
            self.scaler = torch.cuda.amp.GradScaler(enabled=self.use_amp)

    def _autocast(self):
        try:
            return torch.amp.autocast(device_type=self.device.type, enabled=self.use_amp)
        except AttributeError:
            return torch.cuda.amp.autocast(enabled=self.use_amp)

    def _run_epoch(self, loader: DataLoader, training: bool) -> tuple[dict[str, float], dict[str, Any] | None]:
        self.model.train(training)
        totals = {"loss_total": 0.0, "loss_abnormal": 0.0, "loss_severe": 0.0}
        n_samples = 0
        labels_all: list[np.ndarray] = []
        probs_all: list[np.ndarray] = []
        abnormal_all: list[np.ndarray] = []
        severe_all: list[np.ndarray] = []
        context = torch.enable_grad() if training else torch.no_grad()
        with context:
            for batch in loader:
                images = batch["image"].to(self.device, non_blocking=True)
                labels = batch["label"].to(self.device, non_blocking=True)
                if training:
                    self.optimizer.zero_grad(set_to_none=True)
                with self._autocast():
                    outputs = self.model(images)
                    losses = self.criterion(outputs, labels)
                if training:
                    self.scaler.scale(losses["loss_total"]).backward()
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                batch_size = int(labels.size(0))
                n_samples += batch_size
                for name in totals:
                    totals[name] += float(losses[name].detach().item()) * batch_size
                if not training:
                    restored = restore_three_class_probabilities(
                        outputs["logit_abnormal"], outputs["logit_severe_cond"]
                    )
                    labels_all.append(labels.cpu().numpy())
                    probs_all.append(restored["probabilities"].cpu().numpy())
                    abnormal_all.append(restored["p_abnormal"].cpu().numpy())
                    severe_all.append(restored["p_severe_cond"].cpu().numpy())
        if n_samples == 0:
            raise ValueError("E1 loader contains no samples")
        mean_losses = {name: value / n_samples for name, value in totals.items()}
        if training:
            return mean_losses, None
        return mean_losses, compute_e1_metrics(
            np.concatenate(labels_all), np.concatenate(probs_all),
            np.concatenate(abnormal_all), np.concatenate(severe_all), self.ece_bins,
        )

    @staticmethod
    def _capture_rng_state(train_loader: DataLoader) -> dict[str, Any]:
        generator = getattr(train_loader, "generator", None)
        return {
            "python": random.getstate(),
            "numpy": np.random.get_state(),
            "torch_cpu": torch.get_rng_state(),
            "torch_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
            "train_loader_generator": generator.get_state() if generator is not None else None,
        }

    @staticmethod
    def _restore_rng_state(state: dict[str, Any] | None, train_loader: DataLoader) -> None:
        if not state:
            LOGGER.warning("Resume checkpoint has no RNG state; continuation is not bitwise-equivalent.")
            return
        if state.get("python") is not None:
            random.setstate(state["python"])
        if state.get("numpy") is not None:
            np.random.set_state(state["numpy"])
        if state.get("torch_cpu") is not None:
            torch.set_rng_state(state["torch_cpu"].cpu())
        if state.get("torch_cuda") is not None and torch.cuda.is_available():
            for index, value in enumerate(state["torch_cuda"][: torch.cuda.device_count()]):
                torch.cuda.set_rng_state(value.cpu(), device=index)
        generator = getattr(train_loader, "generator", None)
        if generator is not None and state.get("train_loader_generator") is not None:
            generator.set_state(state["train_loader_generator"].cpu())

    def _checkpoint(self, epoch: int, best_metric: float, train_loader: DataLoader) -> dict[str, Any]:
        return {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scaler_state_dict": self.scaler.state_dict(),
            "best_metric": best_metric,
            "monitor_metric": "macro_auc",
            "fold": self.fold,
            "class_counts": self.class_weights["class_counts"],
            "class_weights": self.class_weights,
            "config": self.config,
            "rng_state": self._capture_rng_state(train_loader),
        }

    @staticmethod
    def _load_checkpoint(path: Path, device: torch.device) -> dict[str, Any]:
        try:
            return torch.load(path, map_location=device, weights_only=False)
        except TypeError:
            return torch.load(path, map_location=device)

    def _load_resume_state(self, train_loader: DataLoader) -> tuple[int, float, int, list[dict[str, float | int]]]:
        if self.resume_from is None:
            return 1, float("-inf"), 0, []
        if not self.resume_from.is_file():
            raise FileNotFoundError(f"Resume checkpoint does not exist: {self.resume_from}")
        checkpoint_path = self.resume_from
        try:
            checkpoint = self._load_checkpoint(checkpoint_path, self.device)
        except Exception as error:
            fallback = checkpoint_path.with_name("best_macro_auc.pth")
            if checkpoint_path.name != "last.pth" or not fallback.is_file():
                raise
            LOGGER.warning("Failed to load %s (%s); falling back to %s", checkpoint_path, error, fallback)
            checkpoint_path = fallback
            checkpoint = self._load_checkpoint(checkpoint_path, self.device)
        if checkpoint.get("fold") != self.fold:
            raise ValueError(f"Resume checkpoint fold={checkpoint.get('fold')} does not match requested fold={self.fold}")
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        if checkpoint.get("scaler_state_dict") is not None:
            self.scaler.load_state_dict(checkpoint["scaler_state_dict"])
        self._restore_rng_state(checkpoint.get("rng_state"), train_loader)
        completed_epoch = int(checkpoint.get("epoch", 0))
        best_metric = float(checkpoint.get("best_metric", float("-inf")))
        history_path = self.log_dir / "train_log.csv"
        records: list[dict[str, float | int]] = []
        if history_path.is_file():
            history = pd.read_csv(history_path)
            history = history[pd.to_numeric(history["epoch"], errors="coerce") <= completed_epoch]
            records = history.to_dict("records")
            auc = pd.to_numeric(history["val_macro_auc"], errors="coerce") if "val_macro_auc" in history else pd.Series(dtype=float)
            if not history.empty and not auc.empty and auc.notna().any():
                best_metric = float(auc.max())
                best_epoch = int(history.loc[auc.idxmax(), "epoch"])
                wait = max(completed_epoch - best_epoch, 0)
            else:
                wait = 0
        else:
            wait = 0
        LOGGER.info("Resuming fold=%d from %s at epoch=%d; best macro-AUC=%s; patience=%d/%d",
                    self.fold, checkpoint_path, completed_epoch + 1,
                    f"{best_metric:.4f}" if math.isfinite(best_metric) else "unavailable",
                    wait, self.patience)
        return completed_epoch + 1, best_metric, wait, records

    def fit(self, train_loader: DataLoader, val_loader: DataLoader) -> pd.DataFrame:
        start_epoch, best_metric, wait, records = self._load_resume_state(train_loader)
        for epoch in range(start_epoch, self.epochs + 1):
            train_loss, _ = self._run_epoch(train_loader, training=True)
            val_loss, metrics = self._run_epoch(val_loader, training=False)
            assert metrics is not None
            current = float(metrics["macro_auc"])
            record: dict[str, float | int] = {
                "epoch": epoch,
                **{f"train_{key}": value for key, value in train_loss.items()},
                **{f"val_{key}": value for key, value in val_loss.items()},
                "val_macro_auc": current,
                "val_macro_f1": float(metrics["macro_f1"]),
                "val_balanced_accuracy": float(metrics["balanced_accuracy"]),
                "val_severe_recall": float(metrics["severe_recall"]),
                "val_ordinal_mae": float(metrics["ordinal_mae"]),
                "val_two_step_error_rate": float(metrics["two_step_error_rate"]),
                "val_qwk": float(metrics["qwk"]),
                "val_auc_normal_vs_abnormal": float(metrics["auc_normal_vs_abnormal"]),
                "val_auc_mild_vs_severe_cond": float(metrics["auc_mild_vs_severe_cond"]),
                "learning_rate": float(self.optimizer.param_groups[0]["lr"]),
            }
            records.append(record)
            pd.DataFrame(records).to_csv(self.log_dir / "train_log.csv", index=False, encoding="utf-8-sig")
            improved = not math.isnan(current) and current > best_metric
            if improved:
                best_metric, wait = current, 0
                torch.save(self._checkpoint(epoch, best_metric, train_loader), self.checkpoint_dir / "best_macro_auc.pth")
            else:
                wait += 1
            torch.save(self._checkpoint(epoch, best_metric, train_loader), self.checkpoint_dir / "last.pth")
            LOGGER.info("fold=%d epoch=%d/%d train=%.5f val=%.5f macro_auc=%s best=%s patience=%d/%d",
                        self.fold, epoch, self.epochs, train_loss["loss_total"], val_loss["loss_total"],
                        f"{current:.4f}" if math.isfinite(current) else "nan",
                        f"{best_metric:.4f}" if math.isfinite(best_metric) else "unavailable", wait, self.patience)
            if wait >= self.patience:
                break
        if not (self.checkpoint_dir / "best_macro_auc.pth").is_file():
            last_path = self.checkpoint_dir / "last.pth"
            if not last_path.is_file():
                raise RuntimeError("E1 ran no epochs and has no checkpoint to evaluate")
            LOGGER.warning("Macro-AUC was NaN for every epoch in fold=%d; copying last checkpoint for evaluation.", self.fold)
            shutil.copy2(last_path, self.checkpoint_dir / "best_macro_auc.pth")
        return pd.DataFrame(records)
