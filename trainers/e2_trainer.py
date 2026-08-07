"""Small E2-specific trainer; all metric probabilities use the shared softmax helper."""

from __future__ import annotations

import json
import logging
import math
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader

from metrics.e2_metrics import compute_e2_metrics, softmax_probabilities

LOGGER = logging.getLogger(__name__)


class E2Trainer:
    def __init__(self, model: nn.Module, criterion: nn.Module, optimizer: torch.optim.Optimizer,
                 device: torch.device, fold_dir: Path, config: dict[str, Any], fold: int,
                 class_weights: list[float], class_counts: list[int]) -> None:
        self.model, self.criterion, self.optimizer, self.device = model.to(device), criterion, optimizer, device
        self.fold_dir, self.config, self.fold = fold_dir, config, fold
        self.class_weights, self.class_counts = class_weights, class_counts
        self.ece_bins = int(config.get("metrics", {}).get("ece_bins", 15))
        if self.ece_bins < 1:
            raise ValueError("E2 ECE requires at least one confidence bin")
        self.ckpt_dir, self.log_dir = fold_dir / "checkpoints", fold_dir / "logs"
        self.ckpt_dir.mkdir(parents=True, exist_ok=True); self.log_dir.mkdir(parents=True, exist_ok=True)
        self.use_amp = bool(config["train"].get("use_amp", False) and device.type == "cuda")
        try: self.scaler = torch.amp.GradScaler(device.type, enabled=self.use_amp)
        except (AttributeError, TypeError): self.scaler = torch.cuda.amp.GradScaler(enabled=self.use_amp)

    def _autocast(self):
        try: return torch.amp.autocast(device_type=self.device.type, enabled=self.use_amp)
        except AttributeError: return torch.cuda.amp.autocast(enabled=self.use_amp)

    def _epoch(self, loader: DataLoader, train: bool) -> tuple[float, dict[str, Any] | None]:
        self.model.train(train); total = 0.0; n = 0; labels_all: list[np.ndarray] = []; probs_all: list[np.ndarray] = []
        context = torch.enable_grad() if train else torch.no_grad()
        with context:
            for batch in loader:
                global_image, roi_images, labels = batch["global_image"].to(self.device), batch["roi_images"].to(self.device), batch["label"].to(self.device)
                if train: self.optimizer.zero_grad(set_to_none=True)
                with self._autocast():
                    logits = self.model(global_image, roi_images); loss = self.criterion(logits, labels)
                if train:
                    self.scaler.scale(loss).backward(); self.scaler.step(self.optimizer); self.scaler.update()
                total += float(loss.detach()) * labels.size(0); n += labels.size(0)
                if not train:
                    labels_all.append(labels.cpu().numpy()); probs_all.append(softmax_probabilities(logits).cpu().numpy())
        if n == 0: raise ValueError("Empty E2 DataLoader")
        return total / n, None if train else compute_e2_metrics(
            np.concatenate(labels_all), np.concatenate(probs_all), self.ece_bins
        )

    def _checkpoint(self, epoch: int, best_metric: float) -> dict[str, Any]:
        return {"epoch": epoch, "model_state_dict": self.model.state_dict(), "optimizer_state_dict": self.optimizer.state_dict(),
                "scaler_state_dict": self.scaler.state_dict(), "best_metric": best_metric, "monitor_metric": "macro_auc", "fold": self.fold,
                "config": self.config, "class_counts": self.class_counts, "class_weights": self.class_weights,
                "roi_order": list(self.model.roi_order), "class_names": ["normal", "mild", "severe"], "model_type": "global_multiroi_equal_fusion"}

    def fit(self, train_loader: DataLoader, val_loader: DataLoader) -> pd.DataFrame:
        records: list[dict[str, float | int]] = []; best, wait = float("-inf"), 0
        for epoch in range(1, int(self.config["train"]["epochs"]) + 1):
            train_loss, _ = self._epoch(train_loader, True); val_loss, metrics = self._epoch(val_loader, False); assert metrics is not None
            current = float(metrics["macro_auc"])
            record = {"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss, "val_macro_auc": current, "val_macro_f1": float(metrics["macro_f1"]), "val_balanced_accuracy": float(metrics["balanced_accuracy"]), "val_severe_recall": float(metrics["severe_recall"]), "val_ordinal_mae": float(metrics["ordinal_mae"]), "val_two_step_error_rate": float(metrics["two_step_error_rate"]), "val_qwk": float(metrics["qwk"]), "learning_rate": float(self.optimizer.param_groups[0]["lr"])}
            records.append(record); pd.DataFrame(records).to_csv(self.log_dir / "train_log.csv", index=False, encoding="utf-8-sig")
            if math.isfinite(current) and current > best:
                best, wait = current, 0; torch.save(self._checkpoint(epoch, best), self.ckpt_dir / "best_macro_auc.pth")
            else: wait += 1
            torch.save(self._checkpoint(epoch, best), self.ckpt_dir / "last.pth")
            LOGGER.info("fold=%d epoch=%d train_loss=%.5f val_loss=%.5f val_macro_auc=%s best=%s patience=%d/%d", self.fold, epoch, train_loss, val_loss, f"{current:.4f}" if math.isfinite(current) else "nan", f"{best:.4f}" if math.isfinite(best) else "unavailable", wait, int(self.config["train"]["early_stopping_patience"]))
            if wait >= int(self.config["train"]["early_stopping_patience"]): break
        if not (self.ckpt_dir / "best_macro_auc.pth").is_file():
            last_path = self.ckpt_dir / "last.pth"
            if not last_path.is_file():
                raise RuntimeError(f"fold={self.fold} ran no epoch and has no last checkpoint")
            LOGGER.warning(
                "fold=%d Macro-AUC was NaN for every epoch; copying last.pth "
                "for held-out evaluation, matching E0 behavior",
                self.fold,
            )
            shutil.copy2(last_path, self.ckpt_dir / "best_macro_auc.pth")
        return pd.DataFrame(records)
