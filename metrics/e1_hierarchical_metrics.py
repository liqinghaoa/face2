"""Probability restoration and E1 boundary, ordinal, and calibration metrics."""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import torch
from sklearn.metrics import cohen_kappa_score, roc_auc_score

from metrics.classification_metrics import compute_classification_metrics


LOGGER = logging.getLogger(__name__)


def restore_three_class_probabilities(
    logit_abnormal: torch.Tensor, logit_severe_cond: torch.Tensor
) -> dict[str, torch.Tensor]:
    """Restore [Normal, Mild, Severe] joint probabilities from E1 logits."""
    if logit_abnormal.ndim != 2 or logit_abnormal.shape[1] != 1:
        raise ValueError("logit_abnormal must have shape [B,1]")
    if logit_severe_cond.shape != logit_abnormal.shape:
        raise ValueError("logit_severe_cond must have the same shape as logit_abnormal")
    p_abnormal = torch.sigmoid(logit_abnormal)
    p_severe_cond = torch.sigmoid(logit_severe_cond)
    prob_normal = 1.0 - p_abnormal
    prob_mild = p_abnormal * (1.0 - p_severe_cond)
    prob_severe = p_abnormal * p_severe_cond
    return {
        "p_abnormal": p_abnormal,
        "p_severe_cond": p_severe_cond,
        "probabilities": torch.cat([prob_normal, prob_mild, prob_severe], dim=1),
    }


def _safe_auc(target: np.ndarray, score: np.ndarray, name: str) -> float:
    if np.unique(target).size < 2:
        LOGGER.warning("%s is undefined: only one binary target class is present", name)
        return float("nan")
    return float(roc_auc_score(target, score))


def expected_calibration_error(y_true: np.ndarray, probabilities: np.ndarray, n_bins: int = 15) -> float:
    """Top-label ECE with 15 equal-width confidence bins by default."""
    pred = probabilities.argmax(axis=1)
    confidence = probabilities.max(axis=1)
    correct = (pred == y_true).astype(float)
    ece = 0.0
    for lower in np.linspace(0.0, 1.0, n_bins, endpoint=False):
        upper = lower + 1.0 / n_bins
        mask = (confidence >= lower) & ((confidence < upper) if upper < 1.0 else (confidence <= upper))
        if mask.any():
            ece += mask.mean() * abs(correct[mask].mean() - confidence[mask].mean())
    return float(ece)


def compute_e1_metrics(
    y_true: Any,
    probabilities: Any,
    p_abnormal: Any,
    p_severe_cond: Any,
    ece_bins: int = 15,
) -> dict[str, Any]:
    """Compute E0 three-class metrics plus all E1-required metrics."""
    true = np.asarray(y_true, dtype=np.int64)
    probs = np.asarray(probabilities, dtype=np.float64)
    abnormal = np.asarray(p_abnormal, dtype=np.float64).reshape(-1)
    severe_cond = np.asarray(p_severe_cond, dtype=np.float64).reshape(-1)
    if len(true) != len(probs) or len(true) != len(abnormal) or len(true) != len(severe_cond):
        raise ValueError("All E1 metric inputs must have the same number of samples")
    metrics = compute_classification_metrics(true, probs)
    pred = probs.argmax(axis=1)
    abnormal_mask = true >= 1
    metrics.update(
        {
            "auc_normal_vs_abnormal": _safe_auc(
                abnormal_mask.astype(int), abnormal, "auc_normal_vs_abnormal"
            ),
            "auc_mild_vs_severe_cond": _safe_auc(
                (true[abnormal_mask] == 2).astype(int),
                severe_cond[abnormal_mask],
                "auc_mild_vs_severe_cond",
            ),
            "ordinal_mae": float(np.abs(pred - true).mean()),
            "normal_to_severe_count": int(((true == 0) & (pred == 2)).sum()),
            "severe_to_normal_count": int(((true == 2) & (pred == 0)).sum()),
            "two_step_error_count": int((np.abs(pred - true) == 2).sum()),
            "two_step_error_rate": float((np.abs(pred - true) == 2).mean()),
            "qwk": float(cohen_kappa_score(true, pred, labels=[0, 1, 2], weights="quadratic")),
            "severe_recall": float(metrics["recall_severe"]),
            "multiclass_brier_score": float(np.mean(np.sum((probs - np.eye(3)[true]) ** 2, axis=1))),
            "ece_15_equal_width": expected_calibration_error(
                true, probs, n_bins=int(ece_bins)
            ),
            "abnormal_binary_brier_score": float(np.mean((abnormal - abnormal_mask.astype(float)) ** 2)),
            "severe_cond_binary_brier_score": float(
                np.mean((severe_cond[abnormal_mask] - (true[abnormal_mask] == 2).astype(float)) ** 2)
            ) if abnormal_mask.any() else float("nan"),
        }
    )
    return metrics
