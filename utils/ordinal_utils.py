"""Shared utilities for monotonic cumulative-link ordinal classification."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
from sklearn.metrics import cohen_kappa_score


def encode_ordinal_targets(labels: torch.Tensor, num_classes: int = 3) -> torch.Tensor:
    """Encode integer labels as cumulative targets I(y > k)."""
    if num_classes < 2:
        raise ValueError("num_classes must be at least 2")
    labels = labels.long()
    if labels.ndim != 1:
        raise ValueError(f"labels must have shape [B], got {tuple(labels.shape)}")
    if labels.numel() and ((labels < 0).any() or (labels >= num_classes).any()):
        raise ValueError(f"labels must be in [0, {num_classes - 1}]")
    thresholds = torch.arange(num_classes - 1, device=labels.device)
    return (labels.unsqueeze(1) > thresholds.unsqueeze(0)).to(torch.float32)


def cumulative_logits_to_probabilities(
    logits: torch.Tensor, *, atol: float = 1e-7
) -> torch.Tensor:
    """Convert two monotonic cumulative logits to three-class probabilities.

    The model contract is z0 >= z1, so q0=P(y>0) >= q1=P(y>1).
    No broad clamping or renormalization is used.
    """
    if logits.ndim != 2 or logits.shape[1] != 2:
        raise ValueError(f"ordinal logits must have shape [B, 2], got {tuple(logits.shape)}")
    if not torch.isfinite(logits).all():
        raise ValueError("ordinal logits contain non-finite values")
    if torch.any(logits[:, 0] + atol < logits[:, 1]):
        count = int((logits[:, 0] + atol < logits[:, 1]).sum().item())
        raise ValueError(f"monotonic cumulative logits violated in {count} rows")
    q = torch.sigmoid(logits)
    if torch.any(q[:, 0] + atol < q[:, 1]):
        count = int((q[:, 0] + atol < q[:, 1]).sum().item())
        raise ValueError(f"monotonic cumulative probabilities violated in {count} rows")
    probabilities = torch.stack((1.0 - q[:, 0], q[:, 0] - q[:, 1], q[:, 1]), dim=1)
    if torch.any(probabilities < -atol):
        raise ValueError("derived class probabilities contain negative values")
    # Only remove floating-point noise at the scale permitted by atol.
    probabilities = torch.where(
        (probabilities < 0) & (probabilities >= -atol),
        torch.zeros_like(probabilities),
        probabilities,
    )
    if not torch.isfinite(probabilities).all():
        raise ValueError("derived class probabilities contain non-finite values")
    row_error = torch.abs(probabilities.sum(dim=1) - 1.0)
    if torch.any(row_error > 1e-6):
        raise ValueError(f"derived probability row sum error exceeds tolerance: {row_error.max().item()}")
    return probabilities


def compute_cumulative_pos_weight(labels: torch.Tensor, num_classes: int = 3) -> torch.Tensor:
    """Compute negative/positive weight for each cumulative task."""
    targets = encode_ordinal_targets(labels, num_classes=num_classes)
    positive = targets.sum(dim=0)
    negative = targets.shape[0] - positive
    if torch.any(positive <= 0) or torch.any(negative <= 0):
        raise ValueError(
            f"cumulative tasks require both outcomes; positive={positive.tolist()}, negative={negative.tolist()}"
        )
    return negative / positive


def monotonic_violation_count(logits: torch.Tensor, atol: float = 1e-7) -> int:
    if logits.ndim != 2 or logits.shape[1] != 2:
        raise ValueError(f"ordinal logits must have shape [B, 2], got {tuple(logits.shape)}")
    return int((logits[:, 0] + atol < logits[:, 1]).sum().item())


def compute_ordinal_metrics(y_true: Any, y_pred: Any) -> dict[str, float]:
    true = np.asarray(y_true, dtype=np.int64)
    predicted = np.asarray(y_pred, dtype=np.int64)
    if true.shape != predicted.shape or true.ndim != 1:
        raise ValueError(f"ordinal metric shapes must match 1-D arrays: {true.shape}, {predicted.shape}")
    if len(true) == 0:
        raise ValueError("ordinal metrics require at least one sample")
    errors = np.abs(predicted - true)
    return {
        "ordinal_mae": float(errors.mean()),
        "within_one_accuracy": float((errors <= 1).mean()),
        "extreme_error_rate": float((errors == 2).mean()),
        "quadratic_weighted_kappa": float(
            cohen_kappa_score(true, predicted, labels=[0, 1, 2], weights="quadratic")
        ),
    }
