from __future__ import annotations

from typing import Any

import numpy as np
from scipy import ndimage as ndi

from .color import luminance_y, rgb_uint8_to_linear

EPS = 1e-8


def _masked_values(y: np.ndarray, mask: np.ndarray) -> np.ndarray:
    return y[mask > 0].astype(float)


def robust_y_stats(y: np.ndarray, mask: np.ndarray, prefix: str) -> dict[str, float | None]:
    vals = _masked_values(y, mask)
    if vals.size == 0:
        return {f"{prefix}_{name}": None for name in ["y_mean", "y_median", "y_std", "y_iqr", "y_mad", "y_p01", "y_p05", "y_p10", "y_p25", "y_p75", "y_p90", "y_p95", "y_p99", "y_p90_minus_p10", "y_mad_over_median"]}
    median = float(np.median(vals))
    mad = float(np.median(np.abs(vals - median)))
    return {
        f"{prefix}_y_mean": float(np.mean(vals)),
        f"{prefix}_y_median": median,
        f"{prefix}_y_std": float(np.std(vals, ddof=1)) if vals.size > 1 else 0.0,
        f"{prefix}_y_iqr": float(np.quantile(vals, 0.75) - np.quantile(vals, 0.25)),
        f"{prefix}_y_mad": mad,
        f"{prefix}_y_p01": float(np.quantile(vals, 0.01)),
        f"{prefix}_y_p05": float(np.quantile(vals, 0.05)),
        f"{prefix}_y_p10": float(np.quantile(vals, 0.10)),
        f"{prefix}_y_p25": float(np.quantile(vals, 0.25)),
        f"{prefix}_y_p75": float(np.quantile(vals, 0.75)),
        f"{prefix}_y_p90": float(np.quantile(vals, 0.90)),
        f"{prefix}_y_p95": float(np.quantile(vals, 0.95)),
        f"{prefix}_y_p99": float(np.quantile(vals, 0.99)),
        f"{prefix}_y_p90_minus_p10": float(np.quantile(vals, 0.90) - np.quantile(vals, 0.10)),
        f"{prefix}_y_mad_over_median": float(mad / (median + EPS)),
    }


def fraction_ge(y: np.ndarray, mask: np.ndarray, threshold: float) -> float | None:
    vals = _masked_values(y, mask)
    if vals.size == 0:
        return None
    return float((vals >= threshold).mean())


def fraction_le(y: np.ndarray, mask: np.ndarray, threshold: float) -> float | None:
    vals = _masked_values(y, mask)
    if vals.size == 0:
        return None
    return float((vals <= threshold).mean())


def component_stats(mask_bool: np.ndarray, denom: int) -> tuple[int, int, float]:
    lab, n = ndi.label(mask_bool)
    if n == 0:
        return 0, 0, 0.0
    sizes = ndi.sum(np.ones_like(lab), lab, index=np.arange(1, n + 1))
    largest = int(np.max(sizes))
    return int(n), largest, float(largest / max(1, denom))


def specular_metrics(linear_rgb: np.ndarray, y: np.ndarray, mask: np.ndarray, y_thresholds: tuple[float, ...], chroma_thresholds: tuple[float, ...], prefix: str) -> dict[str, float | int | None]:
    out: dict[str, float | int | None] = {}
    valid = mask > 0
    denom = int(valid.sum())
    chroma = linear_rgb.max(axis=2) - linear_rgb.min(axis=2)
    for yt in y_thresholds:
        for ct in chroma_thresholds:
            candidate = valid & (y >= yt) & (chroma <= ct)
            count, largest, largest_frac = component_stats(candidate, denom)
            key = f"{prefix}_specular_y_ge_{yt:.2f}_chroma_le_{ct:.2f}"
            out[f"{key}_fraction"] = float(candidate.sum() / denom) if denom else None
            out[f"{key}_component_count"] = count
            out[f"{key}_largest_component_fraction"] = largest_frac
    return out


def shadow_metrics(y: np.ndarray, core_mask: np.ndarray, cheek_left: np.ndarray, cheek_right: np.ndarray, ratios: tuple[float, ...], sigma: float, prefix: str) -> dict[str, Any]:
    out: dict[str, Any] = {}
    valid = core_mask > 0
    denom = int(valid.sum())
    vals = y[valid]
    median = float(np.median(vals)) if vals.size else np.nan
    smooth = ndi.gaussian_filter(y.astype(float), sigma=float(sigma))
    for ratio in ratios:
        cand = valid & (smooth < ratio * median)
        count, largest, largest_frac = component_stats(cand, denom)
        key = f"{prefix}_shadow_ratio_{ratio:.2f}"
        out[f"{key}_fraction"] = float(cand.sum() / denom) if denom else None
        out[f"{key}_largest_component_fraction"] = largest_frac
        out[f"{key}_component_count"] = count
        for cheek_name, cheek in [("canvas_left_cheek", cheek_left), ("canvas_right_cheek", cheek_right)]:
            cden = int((cheek > 0).sum())
            out[f"{key}_{cheek_name}_fraction"] = float((cand & (cheek > 0)).sum() / cden) if cden else None
    return out


def input_space_metrics(rgb: np.ndarray, masks: dict[str, np.ndarray], cfg) -> dict[str, Any]:
    linear = rgb_uint8_to_linear(rgb)
    y = luminance_y(linear)
    out: dict[str, Any] = {}
    for erosion in cfg.erosion_candidates:
        out.update(robust_y_stats(y, masks[f"core_skin_e{erosion}"], f"skin_e{erosion}"))
    core = masks["core_skin_e2"]
    out.update({k.replace("skin_e2_", "skin_"): v for k, v in robust_y_stats(y, core, "skin_e2").items()})
    roi_medians = {}
    for name in ["forehead", "canvas_left_cheek", "canvas_right_cheek", "nose"]:
        vals = _masked_values(y, masks[name])
        roi_medians[name] = float(np.median(vals)) if vals.size else None
        out[f"{name}_y_median"] = roi_medians[name]
    left = roi_medians["canvas_left_cheek"]
    right = roi_medians["canvas_right_cheek"]
    forehead = roi_medians["forehead"]
    if left is not None and right is not None:
        signed = left - right
        mean_cheek = (left + right) / 2
        out["cheek_signed_difference"] = float(signed)
        out["cheek_absolute_difference"] = float(abs(signed))
        out["cheek_relative_difference"] = float(abs(signed) / (mean_cheek + EPS))
        out["mean_cheek_y"] = float(mean_cheek)
        if forehead is not None:
            f_signed = forehead - mean_cheek
            out["forehead_cheek_signed_difference"] = float(f_signed)
            out["forehead_cheek_absolute_difference"] = float(abs(f_signed))
    for threshold in cfg.brightness_threshold_candidates:
        out[f"bright_fraction_y_ge_{threshold:.2f}"] = fraction_ge(y, core, threshold)
    for threshold in cfg.darkness_threshold_candidates:
        out[f"dark_fraction_y_le_{threshold:.2f}"] = fraction_le(y, core, threshold)
    out.update(shadow_metrics(y, core, masks["canvas_left_cheek"], masks["canvas_right_cheek"], cfg.shadow_ratio_candidates, cfg.shadow_gaussian_sigma, "input"))
    spec_mask = ((core > 0) | (masks["nose"] > 0)).astype(np.uint8) * 255
    out.update(specular_metrics(linear, y, spec_mask, cfg.specular_y_threshold_candidates, cfg.specular_chroma_threshold_candidates, "input"))
    return out


def assert_monotonic_candidates(row: dict[str, Any], bright_thresholds: tuple[float, ...], dark_thresholds: tuple[float, ...], prefix: str = "") -> None:
    bright = [row.get(f"{prefix}bright_fraction_y_ge_{t:.2f}") for t in bright_thresholds]
    dark = [row.get(f"{prefix}dark_fraction_y_le_{t:.2f}") for t in dark_thresholds]
    bv = [x for x in bright if x is not None]
    dv = [x for x in dark if x is not None]
    if any(bv[i] < bv[i + 1] - 1e-12 for i in range(len(bv) - 1)):
        raise ValueError("bright fraction is not monotone non-increasing")
    if any(dv[i] > dv[i + 1] + 1e-12 for i in range(len(dv) - 1)):
        raise ValueError("dark fraction is not monotone non-decreasing")
