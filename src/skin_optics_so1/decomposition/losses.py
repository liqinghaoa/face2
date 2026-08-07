"""Masked SO-1 decomposition losses."""

from __future__ import annotations

import torch
import torch.nn.functional as F


CHANNEL_NAMES = ("M", "H", "S", "P")
SMOOTH_L1_BETA = 0.1


def masked_smooth_l1_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    valid_mask: torch.Tensor,
    *,
    beta: float = SMOOTH_L1_BETA,
) -> dict[str, torch.Tensor]:
    if prediction.shape != target.shape:
        raise ValueError(f"prediction/target shape mismatch: {prediction.shape} != {target.shape}")
    if prediction.ndim != 4 or prediction.shape[1] != 4:
        raise ValueError("SO-1 prediction must have shape [N,4,H,W]")
    if valid_mask.ndim != 4 or valid_mask.shape[1] != 1:
        raise ValueError("valid_mask must have shape [N,1,H,W]")
    if valid_mask.shape[0] != prediction.shape[0] or valid_mask.shape[-2:] != prediction.shape[-2:]:
        raise ValueError("valid_mask spatial shape must match prediction")
    mask = valid_mask.to(dtype=prediction.dtype)
    denominator = mask.sum()
    if float(denominator.detach().cpu()) <= 0.0:
        raise ValueError("valid_mask.sum() must be > 0 for masked SmoothL1")
    losses: dict[str, torch.Tensor] = {}
    for index, name in enumerate(CHANNEL_NAMES):
        pixel_loss = F.smooth_l1_loss(
            prediction[:, index : index + 1],
            target[:, index : index + 1],
            reduction="none",
            beta=beta,
        )
        losses[name] = (pixel_loss * mask).sum() / denominator
    losses["total"] = sum(losses[name] for name in CHANNEL_NAMES) / len(CHANNEL_NAMES)
    return losses
