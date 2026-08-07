"""Training loop for SO-1 U-Net decomposition preflight runs."""

from __future__ import annotations

import csv
import json
import time
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.utils.data import DataLoader

from .checkpoint import load_checkpoint, save_checkpoint
from .losses import CHANNEL_NAMES, masked_smooth_l1_loss
from .metrics import MaskedMAEAggregator


class SO1DecompositionTrainer:
    def __init__(
        self,
        *,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: Any,
        device: torch.device,
        output_dir: str | Path,
        resolved_config: dict[str, Any],
        dataset_contract: dict[str, Any],
        dataset_fingerprint: dict[str, Any],
        identity: dict[str, Any],
        seed: int,
        amp_enabled: bool,
        early_stopping_patience: int | None,
        resume_from: str | Path | None = None,
    ) -> None:
        self.model = model.to(device)
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_dir = self.output_dir / "checkpoints"
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.resolved_config = resolved_config
        self.dataset_contract = dataset_contract
        self.dataset_fingerprint = dataset_fingerprint
        self.identity = identity
        self.seed = int(seed)
        self.amp_enabled = bool(amp_enabled and device.type == "cuda")
        self.grad_scaler = torch.amp.GradScaler("cuda", enabled=self.amp_enabled)
        self.early_stopping_patience = early_stopping_patience
        self.global_step = 0
        self.best_selection_metric = float("inf")
        self.early_stopping_counter = 0
        self.start_epoch = 1
        if resume_from is not None:
            checkpoint = load_checkpoint(
                resume_from,
                model=self.model,
                optimizer=self.optimizer,
                scheduler=self.scheduler,
                grad_scaler=self.grad_scaler,
                expected_identity=self.identity,
                map_location=self.device,
                restore_rng=True,
            )
            self.start_epoch = int(checkpoint["epoch"]) + 1
            self.global_step = int(checkpoint["global_step"])
            self.best_selection_metric = float(checkpoint["best_selection_metric"])
            self.early_stopping_counter = int(checkpoint["early_stopping_counter"])

    @staticmethod
    def _batch_to_device(batch: dict[str, Any], device: torch.device) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return (
            batch["linear_rgb"].to(device, non_blocking=True),
            batch["target_mhsp"].to(device, non_blocking=True),
            batch["valid_mask"].to(device, non_blocking=True),
        )

    def _run_train_epoch(self, loader: DataLoader) -> dict[str, float]:
        self.model.train()
        total_samples = 0
        loss_sums = {name: 0.0 for name in ("total", *CHANNEL_NAMES)}
        data_time = 0.0
        compute_time = 0.0
        end = time.perf_counter()
        for batch in loader:
            data_time += time.perf_counter() - end
            started = time.perf_counter()
            linear_rgb, target, mask = self._batch_to_device(batch, self.device)
            self.optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(self.device.type, enabled=self.amp_enabled):
                prediction = self.model(linear_rgb)
                losses = masked_smooth_l1_loss(prediction, target, mask)
            self.grad_scaler.scale(losses["total"]).backward()
            self.grad_scaler.step(self.optimizer)
            self.grad_scaler.update()
            batch_size = int(linear_rgb.shape[0])
            total_samples += batch_size
            for key, value in losses.items():
                loss_sums[key] += float(value.detach().cpu()) * batch_size
            self.global_step += 1
            compute_time += time.perf_counter() - started
            end = time.perf_counter()
        result = {f"train_{key}_loss": loss_sums[key] / max(total_samples, 1) for key in loss_sums}
        result["data_time"] = data_time
        result["compute_time"] = compute_time
        result["samples_per_second"] = total_samples / max(data_time + compute_time, 1.0e-9)
        return result

    @torch.inference_mode()
    def evaluate(self, loader: DataLoader, *, prefix: str = "val") -> dict[str, float]:
        self.model.eval()
        metrics = MaskedMAEAggregator()
        total_loss = {name: 0.0 for name in ("total", *CHANNEL_NAMES)}
        total_samples = 0
        for batch in loader:
            linear_rgb, target, mask = self._batch_to_device(batch, self.device)
            with torch.amp.autocast(self.device.type, enabled=self.amp_enabled):
                prediction = self.model(linear_rgb)
                losses = masked_smooth_l1_loss(prediction, target, mask)
            metrics.update(prediction.float(), target.float(), mask.float())
            batch_size = int(linear_rgb.shape[0])
            total_samples += batch_size
            for key, value in losses.items():
                total_loss[key] += float(value.detach().cpu()) * batch_size
        result = {f"{prefix}_{key}_loss": total_loss[key] / max(total_samples, 1) for key in total_loss}
        result.update({f"{prefix}_{key}": value for key, value in metrics.compute().items()})
        return result

    def _save_history(self, records: list[dict[str, Any]]) -> None:
        if not records:
            return
        path = self.output_dir / "training_history.csv"
        fieldnames = list(records[0].keys())
        with path.open("w", encoding="utf-8-sig", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(records)

    def _load_history(self) -> list[dict[str, Any]]:
        path = self.output_dir / "training_history.csv"
        if not path.is_file():
            return []
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            return list(csv.DictReader(handle))

    def fit(self, train_loader: DataLoader, val_loader: DataLoader, *, max_epochs: int) -> list[dict[str, Any]]:
        records = [
            row for row in self._load_history() if int(float(row["epoch"])) < self.start_epoch
        ]
        for epoch in range(self.start_epoch, int(max_epochs) + 1):
            started = time.perf_counter()
            if self.device.type == "cuda":
                torch.cuda.reset_peak_memory_stats(self.device)
            train = self._run_train_epoch(train_loader)
            val = self.evaluate(val_loader, prefix="val")
            if self.scheduler is not None:
                self.scheduler.step()
            selection_metric = float(val["val_selection_metric"])
            improved = selection_metric < self.best_selection_metric
            if improved:
                self.best_selection_metric = selection_metric
                self.early_stopping_counter = 0
            else:
                self.early_stopping_counter += 1
            learning_rate = float(self.optimizer.param_groups[0]["lr"])
            peak_gpu_memory = (
                int(torch.cuda.max_memory_allocated(self.device))
                if self.device.type == "cuda"
                else 0
            )
            record = {
                "epoch": epoch,
                "global_step": self.global_step,
                **train,
                **val,
                "selection_metric": selection_metric,
                "learning_rate": learning_rate,
                "epoch_time": time.perf_counter() - started,
                "gpu_peak_memory_bytes": peak_gpu_memory,
                "is_best": int(improved),
                "resumed": int(epoch == self.start_epoch and self.start_epoch > 1),
            }
            records.append(record)
            self._save_history(records)
            checkpoint_kwargs = {
                "model": self.model,
                "optimizer": self.optimizer,
                "scheduler": self.scheduler,
                "grad_scaler": self.grad_scaler,
                "epoch": epoch,
                "global_step": self.global_step,
                "best_selection_metric": self.best_selection_metric,
                "early_stopping_counter": self.early_stopping_counter,
                "identity": self.identity,
                "resolved_config": self.resolved_config,
                "dataset_contract": self.dataset_contract,
                "dataset_fingerprint": self.dataset_fingerprint,
                "seed": self.seed,
            }
            save_checkpoint(self.checkpoint_dir / "last.pt", **checkpoint_kwargs)
            save_checkpoint(self.output_dir / "last.pt", **checkpoint_kwargs)
            if improved:
                save_checkpoint(self.checkpoint_dir / "best.pt", **checkpoint_kwargs)
                save_checkpoint(self.output_dir / "best.pt", **checkpoint_kwargs)
            if (
                self.early_stopping_patience is not None
                and self.early_stopping_counter >= self.early_stopping_patience
            ):
                break
        runtime = {
            "best_selection_metric": self.best_selection_metric,
            "completed_epoch": int(records[-1]["epoch"]) if records else 0,
            "global_step": self.global_step,
            "amp_enabled": self.amp_enabled,
        }
        (self.output_dir / "runtime.json").write_text(
            json.dumps(runtime, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        return records
