"""Metrics for the E0B fixed binary classification task."""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def compute_binary_metrics(y_true: Any, y_prob: Any) -> dict[str, Any]:
    """Use patient probability for binary ROC-AUC and argmax for class decisions."""
    true = np.asarray(y_true, dtype=np.int64)
    prob = np.asarray(y_prob, dtype=np.float64)
    if true.ndim != 1 or len(true) == 0:
        raise ValueError("y_true must be a non-empty vector")
    if prob.shape != (len(true), 2):
        raise ValueError(f"y_prob must have shape [N, 2], got {prob.shape}")
    if not np.isin(true, [0, 1]).all():
        raise ValueError("binary labels must be 0 or 1")
    if not np.isfinite(prob).all() or (prob < 0).any() or (prob > 1).any():
        raise ValueError("probabilities must be finite and in [0, 1]")
    if not np.allclose(prob.sum(axis=1), 1.0, atol=1e-5):
        raise ValueError("probability rows must sum to one")
    pred = prob.argmax(axis=1)
    return {
        "macro_auc": float(roc_auc_score(true, prob[:, 1])),
        "accuracy": float(accuracy_score(true, pred)),
        "macro_precision": float(precision_score(true, pred, average="macro", zero_division=0)),
        "macro_recall": float(recall_score(true, pred, average="macro", zero_division=0)),
        "macro_f1": float(f1_score(true, pred, average="macro", zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(true, pred)),
        "confusion_matrix": confusion_matrix(true, pred, labels=[0, 1]),
    }


def flatten_metrics(metrics: dict[str, Any]) -> dict[str, float]:
    return {key: float(value) for key, value in metrics.items() if np.isscalar(value)}
