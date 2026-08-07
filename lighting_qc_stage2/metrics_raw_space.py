from __future__ import annotations

from typing import Any

import numpy as np

from .color import luminance_y, rgb_uint8_to_linear
from .metrics_input_space import fraction_ge, fraction_le, specular_metrics


def channel_clipping_metrics(rgb: np.ndarray, mask: np.ndarray, high: tuple[int, ...], low: tuple[int, ...]) -> dict[str, float | None]:
    valid = mask > 0
    pix = rgb[valid]
    out: dict[str, float | None] = {}
    if pix.size == 0:
        return out
    for i, ch in enumerate("rgb"):
        out[f"raw_{ch}_eq_255_fraction"] = float((pix[:, i] == 255).mean())
        out[f"raw_{ch}_eq_0_fraction"] = float((pix[:, i] == 0).mean())
    out["raw_any_channel_eq_255_fraction"] = float((pix == 255).any(axis=1).mean())
    out["raw_all_channels_eq_255_fraction"] = float((pix == 255).all(axis=1).mean())
    out["raw_any_channel_eq_0_fraction"] = float((pix == 0).any(axis=1).mean())
    out["raw_all_channels_eq_0_fraction"] = float((pix == 0).all(axis=1).mean())
    for t in high:
        out[f"raw_any_channel_ge_{t}_fraction"] = float((pix >= t).any(axis=1).mean())
        out[f"raw_all_channels_ge_{t}_fraction"] = float((pix >= t).all(axis=1).mean())
    for t in low:
        out[f"raw_any_channel_le_{t}_fraction"] = float((pix <= t).any(axis=1).mean())
        out[f"raw_all_channels_le_{t}_fraction"] = float((pix <= t).all(axis=1).mean())
    return out


def raw_space_metrics(rgb: np.ndarray, masks: dict[str, np.ndarray], cfg) -> dict[str, Any]:
    core = masks["raw_core_skin"]
    linear = rgb_uint8_to_linear(rgb)
    y = luminance_y(linear)
    out: dict[str, Any] = {}
    out.update(channel_clipping_metrics(rgb, core, cfg.clip_high_threshold_candidates, cfg.clip_low_threshold_candidates))
    for threshold in cfg.brightness_threshold_candidates:
        out[f"raw_bright_fraction_y_ge_{threshold:.2f}"] = fraction_ge(y, core, threshold)
    for threshold in cfg.darkness_threshold_candidates:
        out[f"raw_dark_fraction_y_le_{threshold:.2f}"] = fraction_le(y, core, threshold)
    spec_mask = ((core > 0) | (masks["raw_nose"] > 0)).astype(np.uint8) * 255
    out.update(specular_metrics(linear, y, spec_mask, cfg.specular_y_threshold_candidates, cfg.specular_chroma_threshold_candidates, "raw"))
    return out
