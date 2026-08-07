"""Unified softmax and E2 classification, ordinal, and calibration metrics."""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import torch
from sklearn.metrics import cohen_kappa_score, log_loss

from metrics.classification_metrics import compute_classification_metrics
from metrics.e1_hierarchical_metrics import expected_calibration_error

LOGGER = logging.getLogger(__name__)


def softmax_probabilities(logits: torch.Tensor) -> torch.Tensor:
    if logits.ndim != 2 or logits.shape[1] != 3:
        raise ValueError(f"E2 logits must be [B,3], got {tuple(logits.shape)}")
    return torch.softmax(logits, dim=1)


def compute_e2_metrics(
    y_true: Any,
    probabilities: Any,
    ece_bins: int = 15,
) -> dict[str, Any]:
    true, probs = np.asarray(y_true, dtype=np.int64), np.asarray(probabilities, dtype=np.float64)
    if int(ece_bins) < 1:
        raise ValueError("ece_bins must be at least 1")
    metrics = compute_classification_metrics(true, probs)
    pred = probs.argmax(axis=1)
    error = np.abs(pred - true)
    if np.unique(true).size < 2 or np.unique(pred).size < 2:
        LOGGER.warning("qwk is undefined because true labels or predictions contain only one class")
        qwk = float("nan")
    else:
        qwk = float(cohen_kappa_score(true, pred, labels=[0, 1, 2], weights="quadratic"))
    metrics.update({
        "ordinal_mae": float(error.mean()), "normal_to_severe_count": int(((true == 0) & (pred == 2)).sum()),
        "severe_to_normal_count": int(((true == 2) & (pred == 0)).sum()), "two_step_error_count": int((error == 2).sum()),
        "two_step_error_rate": float((error == 2).mean()), "qwk": qwk,
        "severe_recall": float(metrics["recall_severe"]), "multiclass_brier_score": float(np.mean(np.sum((probs - np.eye(3)[true]) ** 2, axis=1))),
        "ece_15_equal_width": expected_calibration_error(
            true, probs, n_bins=int(ece_bins)
        ),
        "nll": float(log_loss(true, probs, labels=[0, 1, 2])),
    })
    return metrics
