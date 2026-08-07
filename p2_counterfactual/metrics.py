from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.metrics import average_precision_score, precision_score, recall_score

from metrics.binary_classification_metrics import compute_binary_metrics


def compute_p2_a_metrics(y_true: Any, y_prob: Any) -> dict[str, Any]:
    labels = np.asarray(y_true, dtype=np.int64)
    prob = np.asarray(y_prob, dtype=np.float64)
    metrics = compute_binary_metrics(labels, prob)
    pred = prob.argmax(axis=1)
    metrics.update(
        {
            "pr_auc": float(average_precision_score(labels, prob[:, 1])),
            "patient_sensitivity": float(recall_score(labels, pred, pos_label=1, zero_division=0)),
            "control_specificity": float(recall_score(labels, pred, pos_label=0, zero_division=0)),
            "ppv": float(precision_score(labels, pred, pos_label=1, zero_division=0)),
            "npv": float(precision_score(labels, pred, pos_label=0, zero_division=0)),
        }
    )
    return metrics


def json_safe(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    return value


def scalar_metrics(metrics: dict[str, Any]) -> dict[str, float]:
    return {key: float(value) for key, value in metrics.items() if np.isscalar(value)}
