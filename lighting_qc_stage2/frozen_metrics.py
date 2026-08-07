from __future__ import annotations

from typing import Any

import numpy as np
from scipy import ndimage as ndi

from .color import luminance_y, rgb_uint8_to_linear
from .metrics_input_space import EPS, component_stats, fraction_ge, fraction_le, robust_y_stats
from .metrics_raw_space import channel_clipping_metrics


PRIMARY_SKIN_EROSION_PX = 2
AUXILIARY_SHADOW_RATIO = 0.55
DEEP_RELATIVE_DARK_RATIO = 0.45
AUXILIARY_SPECULAR_Y = 0.75
AUXILIARY_SPECULAR_CHROMA = 0.08
SEVERE_BRIGHT_Y = 0.80
SEVERE_DARK_Y = 0.05
DEEP_DARK_Y = 0.03


def _masked_values(y: np.ndarray, mask: np.ndarray) -> np.ndarray:
    return y[mask > 0].astype(float)


def _quantiles(y: np.ndarray, mask: np.ndarray, prefix: str, probs: tuple[float, ...]) -> dict[str, float | None]:
    vals = _masked_values(y, mask)
    out: dict[str, float | None] = {}
    for p in probs:
        key = f"{prefix}_y_p{int(round(p * 100)):02d}"
        out[key] = float(np.quantile(vals, p)) if vals.size else None
    return out


def primary_metrics(rgb: np.ndarray, masks: dict[str, np.ndarray]) -> dict[str, Any]:
    linear = rgb_uint8_to_linear(rgb)
    y = luminance_y(linear)
    core = masks["core_skin_e2"]
    out: dict[str, Any] = {}
    out.update({k.replace("skin_e2_", "skin_"): v for k, v in robust_y_stats(y, core, "skin_e2").items()})
    median = out.get("skin_y_median")
    p10 = out.get("skin_y_p10")
    p05 = out.get("skin_y_p05")
    out["dark_contrast_score"] = float((median - p10) / (median + EPS)) if median is not None and p10 is not None else None
    out["deep_dark_contrast_score"] = float((median - p05) / (median + EPS)) if median is not None and p05 is not None else None

    roi_medians: dict[str, float | None] = {}
    for name in ["forehead", "canvas_left_cheek", "canvas_right_cheek", "nose"]:
        vals = _masked_values(y, masks[name])
        roi_medians[name] = float(np.median(vals)) if vals.size else None
        out[f"{name}_y_median"] = roi_medians[name]
    out.update(_quantiles(y, masks["nose"], "nose", (0.90, 0.95, 0.99)))
    out.update(_quantiles(y, masks["forehead"], "forehead", (0.90, 0.95, 0.99)))

    left = roi_medians["canvas_left_cheek"]
    right = roi_medians["canvas_right_cheek"]
    forehead = roi_medians["forehead"]
    if left is not None and right is not None:
        signed = left - right
        mean_cheek = (left + right) / 2.0
        out["cheek_signed_difference"] = float(signed)
        out["cheek_absolute_difference"] = float(abs(signed))
        out["cheek_relative_difference"] = float(abs(signed) / (mean_cheek + EPS))
        out["mean_cheek_y"] = float(mean_cheek)
        if forehead is not None:
            f_signed = forehead - mean_cheek
            out["forehead_cheek_signed_difference"] = float(f_signed)
            out["forehead_cheek_absolute_difference"] = float(abs(f_signed))
        else:
            out["forehead_cheek_signed_difference"] = None
            out["forehead_cheek_absolute_difference"] = None
    return out


def erosion_sensitivity_metrics(rgb: np.ndarray, masks: dict[str, np.ndarray]) -> dict[str, Any]:
    y = luminance_y(rgb_uint8_to_linear(rgb))
    out: dict[str, Any] = {}
    for erosion in (0, 2, 4):
        prefix = f"skin_e{erosion}"
        out.update(robust_y_stats(y, masks[f"core_skin_e{erosion}"], prefix))
        out[f"{prefix}_pixels"] = int((masks[f"core_skin_e{erosion}"] > 0).sum())
    return out


def auxiliary_metrics(rgb: np.ndarray, masks: dict[str, np.ndarray], sigma: float = 1.2) -> dict[str, Any]:
    linear = rgb_uint8_to_linear(rgb)
    y = luminance_y(linear)
    core = masks["core_skin_e2"] > 0
    denom = int(core.sum())
    vals = y[core]
    median = float(np.median(vals)) if vals.size else np.nan
    out: dict[str, Any] = {
        "severe_bright_fraction": fraction_ge(y, masks["core_skin_e2"], SEVERE_BRIGHT_Y),
        "severe_dark_fraction": fraction_le(y, masks["core_skin_e2"], SEVERE_DARK_Y),
        "deep_dark_fraction": fraction_le(y, masks["core_skin_e2"], DEEP_DARK_Y),
    }
    smooth = ndi.gaussian_filter(y.astype(float), sigma=float(sigma))
    aux_shadow = core & (smooth < AUXILIARY_SHADOW_RATIO * median)
    count, _, largest_frac = component_stats(aux_shadow, denom)
    out["auxiliary_shadow_fraction"] = float(aux_shadow.sum() / denom) if denom else None
    out["auxiliary_shadow_largest_component_fraction"] = largest_frac
    out["auxiliary_shadow_component_count"] = count
    for out_name, mask_name in [("left_cheek", "canvas_left_cheek"), ("right_cheek", "canvas_right_cheek")]:
        cden = int((masks[mask_name] > 0).sum())
        out[f"{out_name}_auxiliary_shadow_fraction"] = float((aux_shadow & (masks[mask_name] > 0)).sum() / cden) if cden else None
    deep_relative = core & (smooth < DEEP_RELATIVE_DARK_RATIO * median)
    out["deep_relative_dark_fraction"] = float(deep_relative.sum() / denom) if denom else None

    spec_mask = (core | (masks["nose"] > 0))
    spec_denom = int(spec_mask.sum())
    chroma = linear.max(axis=2) - linear.min(axis=2)
    spec = spec_mask & (y >= AUXILIARY_SPECULAR_Y) & (chroma <= AUXILIARY_SPECULAR_CHROMA)
    spec_count, _, spec_largest_frac = component_stats(spec, spec_denom)
    out["auxiliary_specular_fraction"] = float(spec.sum() / spec_denom) if spec_denom else None
    out["auxiliary_specular_component_count"] = spec_count
    out["auxiliary_specular_largest_component_fraction"] = spec_largest_frac
    nose_count = int((spec & (masks["nose"] > 0)).sum())
    forehead_count = int((spec & (masks["forehead"] > 0)).sum())
    out["auxiliary_specular_nose_fraction"] = float(nose_count / max(1, int(spec.sum()))) if spec.any() else 0.0
    out["auxiliary_specular_forehead_fraction"] = float(forehead_count / max(1, int(spec.sum()))) if spec.any() else 0.0
    return out


def raw_frozen_metrics(rgb: np.ndarray, raw_masks: dict[str, np.ndarray]) -> dict[str, Any]:
    raw = channel_clipping_metrics(rgb, raw_masks["raw_core_skin"], (250,), (5,))
    keep = [
        "raw_r_eq_255_fraction",
        "raw_g_eq_255_fraction",
        "raw_b_eq_255_fraction",
        "raw_any_channel_eq_255_fraction",
        "raw_all_channels_eq_255_fraction",
        "raw_r_eq_0_fraction",
        "raw_g_eq_0_fraction",
        "raw_b_eq_0_fraction",
        "raw_any_channel_eq_0_fraction",
        "raw_all_channels_eq_0_fraction",
        "raw_any_channel_ge_250_fraction",
        "raw_all_channels_ge_250_fraction",
        "raw_any_channel_le_5_fraction",
        "raw_all_channels_le_5_fraction",
    ]
    return {key: raw.get(key) for key in keep}
