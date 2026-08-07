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


def compute_masked_mae(
    prediction: torch.Tensor, target: torch.Tensor, valid_mask: torch.Tensor
) -> dict[str, float]:
    aggregator = MaskedMAEAggregator()
    aggregator.update(prediction, target, valid_mask)
    return aggregator.compute()
