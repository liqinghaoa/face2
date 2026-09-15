"""Bounded two-parameter transforms used by the inverse solver."""

from __future__ import annotations

import numpy as np


def _bounds_arrays(bounds: tuple[tuple[float, float], ...]) -> tuple[np.ndarray, np.ndarray]:
    lo = np.asarray([item[0] for item in bounds], dtype=np.float64)
    hi = np.asarray([item[1] for item in bounds], dtype=np.float64)
    if np.any(~np.isfinite(lo)) or np.any(~np.isfinite(hi)) or np.any(lo >= hi):
        raise ValueError("Bounds must be finite and strictly increasing")
    return lo, hi


def logits_to_theta(logits: np.ndarray, bounds: tuple[tuple[float, float], ...]) -> np.ndarray:
    """Map unconstrained logits to the open physical parameter ranges."""

    values = np.asarray(logits, dtype=np.float64)
    lo, hi = _bounds_arrays(bounds)
    sigmoid = np.empty_like(values, dtype=np.float64)
    positive = values >= 0
    sigmoid[positive] = 1.0 / (1.0 + np.exp(-values[positive]))
    exp_values = np.exp(values[~positive])
    sigmoid[~positive] = exp_values / (1.0 + exp_values)
    return lo + (hi - lo) * sigmoid


def theta_to_logits(theta: np.ndarray, bounds: tuple[tuple[float, float], ...]) -> np.ndarray:
    """Map physical parameters to stable unconstrained logits."""

    values = np.asarray(theta, dtype=np.float64)
    lo, hi = _bounds_arrays(bounds)
    fraction = (values - lo) / (hi - lo)
    if np.any((fraction < 0) | (fraction > 1)) or np.any(~np.isfinite(fraction)):
        raise ValueError("Theta outside configured bounds")
    fraction = np.clip(fraction, 1e-9, 1.0 - 1e-9)
    return np.log(fraction) - np.log1p(-fraction)


def boundary_flags(
    theta: np.ndarray,
    bounds: tuple[tuple[float, float], ...],
    fraction: float = 0.01,
) -> np.ndarray:
    """Return one flag per parameter when theta is near either bound."""

    values = np.asarray(theta, dtype=np.float64)
    lo, hi = _bounds_arrays(bounds)
    margin = fraction * (hi - lo)
    return (values <= lo + margin) | (values >= hi - margin)

