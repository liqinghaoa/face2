"""Masked SO-1 decomposition metrics."""

from __future__ import annotations

import torch

from .losses import CHANNEL_NAMES


class MaskedMAEAggregator:
    def __init__(self) -> None:
        self.abs_error = torch.zeros(4, dtype=torch.float64)
        self.valid_count = 0.0

    def update(self, prediction: torch.Tensor, target: torch.Tensor, valid_mask: torch.Tensor) -> None:
        if prediction.shape != target.shape:
            raise ValueError("prediction/target shape mismatch")
        mask = valid_mask.detach().to(device=prediction.device, dtype=prediction.dtype)
        denominator = float(mask.sum().detach().cpu())
        if denominator <= 0.0:
            raise ValueError("valid_mask.sum() must be > 0 for masked MAE")
        error = torch.abs(prediction.detach() - target.detach()) * mask
        self.abs_error += error.sum(dim=(0, 2, 3)).double().cpu()
        self.valid_count += denominator

    def compute(self) -> dict[str, float]:
        if self.valid_count <= 0.0:
            raise ValueError("No valid pixels accumulated")
        values = self.abs_error / self.valid_count
        result = {f"{name}_MAE": float(values[index]) for index, name in enumerate(CHANNEL_NAMES)}
        result["selection_metric"] = result["M_MAE"] + result["H_MAE"]
        return result


class PredictionMonitoringAggregator:
    """Accumulate lightweight validation output diagnostics over valid pixels."""

    def __init__(self, *, p_active_threshold: float = 0.01) -> None:
        self.value_sum = torch.zeros(4, dtype=torch.float64)
        self.value_square_sum = torch.zeros(4, dtype=torch.float64)
        self.le_threshold_count = torch.zeros(4, dtype=torch.float64)
        self.ge_threshold_count = torch.zeros(4, dtype=torch.float64)
        self.valid_count = 0.0
        self.p_active_true_count = 0.0
        self.p_active_recovered_count = 0.0
        self.p_active_threshold = float(p_active_threshold)

    def update(self, prediction: torch.Tensor, target: torch.Tensor, valid_mask: torch.Tensor) -> None:
        if prediction.shape != target.shape or prediction.ndim != 4 or prediction.shape[1] != 4:
            raise ValueError("prediction and target must have shape [N,4,H,W]")
        if valid_mask.shape != (prediction.shape[0], 1, prediction.shape[2], prediction.shape[3]):
            raise ValueError("valid_mask must have shape [N,1,H,W]")
        pred = prediction.detach().float()
        true = target.detach().float()
        mask = valid_mask.detach().to(device=pred.device, dtype=torch.bool)
        count = float(mask.sum().cpu())
        if count <= 0.0:
            raise ValueError("valid_mask.sum() must be > 0 for prediction monitoring")
        expanded = mask.expand_as(pred)
        values = torch.where(expanded, pred, torch.zeros_like(pred))
        self.value_sum += values.sum(dim=(0, 2, 3)).double().cpu()
        self.value_square_sum += values.square().sum(dim=(0, 2, 3)).double().cpu()
        self.le_threshold_count += ((pred <= 0.01) & expanded).sum(dim=(0, 2, 3)).double().cpu()
        self.ge_threshold_count += ((pred >= 0.99) & expanded).sum(dim=(0, 2, 3)).double().cpu()
        self.valid_count += count
        p_active = (true[:, 3:4] > self.p_active_threshold) & mask
        self.p_active_true_count += float(p_active.sum().cpu())
        self.p_active_recovered_count += float(
            ((pred[:, 3:4] > self.p_active_threshold) & p_active).sum().cpu()
        )

    def compute(self) -> dict[str, float]:
        if self.valid_count <= 0.0:
            raise ValueError("No valid pixels accumulated")
        mean = self.value_sum / self.valid_count
        variance = (self.value_square_sum / self.valid_count - mean.square()).clamp_min(0.0)
        result: dict[str, float] = {}
        for index, name in enumerate(CHANNEL_NAMES):
            result[f"{name}_pred_mean"] = float(mean[index])
            result[f"{name}_pred_std"] = float(variance[index].sqrt())
            result[f"{name}_pred_le_0.01_fraction"] = float(
                self.le_threshold_count[index] / self.valid_count
            )
            result[f"{name}_pred_ge_0.99_fraction"] = float(
                self.ge_threshold_count[index] / self.valid_count
            )
        result["P_active_recall"] = (
            self.p_active_recovered_count / self.p_active_true_count
            if self.p_active_true_count > 0.0
            else float("nan")
        )
        return result


def compute_masked_mae(
    prediction: torch.Tensor, target: torch.Tensor, valid_mask: torch.Tensor
) -> dict[str, float]:
    aggregator = MaskedMAEAggregator()
    aggregator.update(prediction, target, valid_mask)
    return aggregator.compute()
