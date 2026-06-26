"""Robust metrics for NYHA three-class classification."""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    precision_recall_fscore_support,
    roc_auc_score,
)


LOGGER = logging.getLogger(__name__)
CLASS_NAMES = {0: "normal", 1: "mild", 2: "severe"}


def _safe_binary_auc(
    binary_true: np.ndarray, scores: np.ndarray, metric_name: str
) -> float:
    if np.unique(binary_true).size < 2:
        LOGGER.warning(
            "%s is undefined because the validation data contains one binary class",
            metric_name,
        )
        return float("nan")
    try:
        return float(roc_auc_score(binary_true, scores))
    except ValueError as exc:
        LOGGER.warning("%s could not be computed: %s", metric_name, exc)
        return float("nan")


def compute_classification_metrics(
    y_true: Any,
    y_prob: Any,
    num_classes: int = 3,
) -> dict[str, Any]:
    """Compute main, per-class, binary auxiliary metrics and confusion matrix."""
    true = np.asarray(y_true, dtype=np.int64)
    prob = np.asarray(y_prob, dtype=np.float64)
    if true.ndim != 1:
        raise ValueError(f"y_true must have shape [N], got {true.shape}")
    if prob.shape != (len(true), num_classes):
        raise ValueError(
            f"y_prob must have shape [N, {num_classes}], got {prob.shape}"
        )
    if len(true) == 0:
        raise ValueError("Metrics require at least one prediction")
    if not np.isfinite(prob).all():
        raise ValueError("y_prob contains NaN or infinite values")
    if (prob < 0).any() or (prob > 1).any():
        raise ValueError("y_prob values must be in [0, 1]")
    if not np.allclose(prob.sum(axis=1), 1.0, atol=1e-5):
        raise ValueError("Each y_prob row must sum to 1")
    if not np.isin(true, np.arange(num_classes)).all():
        raise ValueError(f"y_true values must be in [0, {num_classes - 1}]")

    predicted = prob.argmax(axis=1)
    precision, recall, f1, _ = precision_recall_fscore_support(
        true,
        predicted,
        labels=list(range(num_classes)),
        average=None,
        zero_division=0,
    )
    result: dict[str, Any] = {
        "accuracy": float(accuracy_score(true, predicted)),
        "macro_precision": float(np.mean(precision)),
        "macro_recall": float(np.mean(recall)),
        "macro_f1": float(np.mean(f1)),
        "balanced_accuracy": float(balanced_accuracy_score(true, predicted)),
    }

    per_class_auc: list[float] = []
    for class_index in range(num_classes):
        name = CLASS_NAMES[class_index]
        binary_true = (true == class_index).astype(np.int64)
        auc = _safe_binary_auc(binary_true, prob[:, class_index], f"auc_{name}")
        per_class_auc.append(auc)
        result[f"auc_{name}"] = auc
        result[f"precision_{name}"] = float(precision[class_index])
        result[f"recall_{name}"] = float(recall[class_index])
        result[f"f1_{name}"] = float(f1[class_index])

    finite_aucs = [value for value in per_class_auc if np.isfinite(value)]
    if len(finite_aucs) != num_classes:
        LOGGER.warning(
            "macro_auc is undefined because at least one one-vs-rest class is missing"
        )
        result["macro_auc"] = float("nan")
    else:
        try:
            result["macro_auc"] = float(
                roc_auc_score(
                    true,
                    prob,
                    labels=list(range(num_classes)),
                    multi_class="ovr",
                    average="macro",
                )
            )
        except ValueError as exc:
            LOGGER.warning("macro_auc could not be computed: %s", exc)
            result["macro_auc"] = float("nan")

    result["severe_vs_rest_auc"] = _safe_binary_auc(
        (true == 2).astype(np.int64), prob[:, 2], "severe_vs_rest_auc"
    )
    result["normal_vs_abnormal_auc"] = _safe_binary_auc(
        (true == 0).astype(np.int64), prob[:, 0], "normal_vs_abnormal_auc"
    )
    result["confusion_matrix"] = confusion_matrix(
        true, predicted, labels=list(range(num_classes))
    )
    return result


def flatten_metrics(metrics: dict[str, Any]) -> dict[str, float]:
    """Remove array-valued entries so metrics can be saved as one CSV row."""
    return {
        key: float(value)
        for key, value in metrics.items()
        if np.isscalar(value) and not isinstance(value, str)
    }
