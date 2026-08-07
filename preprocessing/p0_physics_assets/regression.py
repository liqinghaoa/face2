"""Pixel-level black-background regression acceptance."""

from __future__ import annotations

import numpy as np

from .types import RegressionConfig, RegressionResult


def rgb_ssim(reference: np.ndarray, candidate: np.ndarray) -> float:
    """Compute mean channel-wise global SSIM for same-sized uint8 RGB rasters."""
    a = reference.astype(np.float64); b = candidate.astype(np.float64); c1, c2 = 6.5025, 58.5225
    values = []
    for channel in range(3):
        x, y = a[..., channel], b[..., channel]; mx, my = x.mean(), y.mean()
        vx, vy, cov = x.var(), y.var(), ((x-mx)*(y-my)).mean()
        values.append(((2*mx*my+c1)*(2*cov+c2))/((mx*mx+my*my+c1)*(vx+vy+c2)))
    return float(np.mean(values))


def evaluate_regression(reference: np.ndarray, candidate: np.ndarray, config: RegressionConfig) -> RegressionResult:
    """Evaluate exact P0 acceptance thresholds; no RGB-nonzero IoU surrogate."""
    if reference.shape != candidate.shape or reference.ndim != 3 or reference.shape[2] != 3: raise ValueError("regression requires same-size RGB images")
    diff = np.abs(reference.astype(np.int16)-candidate.astype(np.int16)); mae=float(diff.mean()); p99=float(np.quantile(diff, .99)); large=float((diff > config.large_diff_threshold).mean()); score=rgb_ssim(reference,candidate)
    passed = mae <= config.mae_max and p99 <= config.p99_abs_diff_max and score >= config.ssim_min and large <= config.large_diff_fraction_max
    heat = np.clip(diff.max(axis=2)*4,0,255).astype(np.uint8)
    return RegressionResult("passed" if passed else "failed", mae, float(np.sqrt((diff.astype(float)**2).mean())), float(diff.max()), p99, score, large, heat)
