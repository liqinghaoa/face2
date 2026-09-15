"""Spectral fit metrics used by stage one."""

from __future__ import annotations

import numpy as np


def _aligned(predicted: np.ndarray, observed: np.ndarray, epsilon: float) -> tuple[np.ndarray, np.ndarray]:
    pred = np.asarray(predicted, dtype=np.float64)
    obs = np.asarray(observed, dtype=np.float64)
    if pred.shape != obs.shape:
        raise ValueError("Predicted and observed spectra must have identical shapes")
    valid = np.isfinite(pred) & np.isfinite(obs) & (pred > epsilon) & (obs > epsilon)
    if int(valid.sum()) < 3:
        raise ValueError("At least three valid positive bands are required")
    return pred[valid], obs[valid]


def spectral_angle_rad(predicted: np.ndarray, observed: np.ndarray, epsilon: float = 1e-8) -> float:
    pred, obs = _aligned(predicted, observed, epsilon)
    denominator = np.linalg.norm(pred) * np.linalg.norm(obs)
    if denominator <= epsilon:
        return float("nan")
    cosine = float(np.dot(pred, obs) / denominator)
    return float(np.arccos(np.clip(cosine, -1.0, 1.0)))


def spectral_metrics(predicted: np.ndarray, observed: np.ndarray, epsilon: float = 1e-8) -> dict[str, float]:
    pred, obs = _aligned(predicted, observed, epsilon)
    difference = pred - obs
    log_difference = np.log(np.maximum(pred, epsilon)) - np.log(np.maximum(obs, epsilon))
    return {
        "rmse": float(np.sqrt(np.mean(difference**2))),
        "mae": float(np.mean(np.abs(difference))),
        "log_rmse": float(np.sqrt(np.mean(log_difference**2))),
        "log_mae": float(np.mean(np.abs(log_difference))),
        "sam_rad": spectral_angle_rad(pred, obs, epsilon),
    }

