"""Colour input conversion utilities for SO-1."""

from __future__ import annotations

import numpy as np


def inverse_srgb(srgb: np.ndarray) -> np.ndarray:
    """Apply the standard inverse sRGB transfer function in float32.

    Input and output keep RGB channel order. Values are clipped to [0, 1]
    before decoding because SO-1 consumes SO-0's display-clipped sRGB image.
    """

    x = np.asarray(srgb, dtype=np.float32)
    x = np.clip(x, 0.0, 1.0)
    low = x <= np.float32(0.04045)
    out = np.empty_like(x, dtype=np.float32)
    out[low] = x[low] / np.float32(12.92)
    out[~low] = np.power((x[~low] + np.float32(0.055)) / np.float32(1.055), np.float32(2.4))
    return np.clip(out, 0.0, 1.0).astype(np.float32, copy=False)


def srgb_hwc_to_linear_chw(srgb_hwc: np.ndarray, valid_mask_hw: np.ndarray) -> np.ndarray:
    """Mask display sRGB, inverse-decode, and return [3,H,W] linear RGB."""

    mask = np.asarray(valid_mask_hw, dtype=np.float32)
    if mask.ndim != 2:
        raise ValueError(f"valid_mask_hw must be [H,W], got {mask.shape}")
    srgb = np.asarray(srgb_hwc, dtype=np.float32)
    if srgb.ndim != 3 or srgb.shape[-1] != 3:
        raise ValueError(f"srgb_hwc must be [H,W,3], got {srgb.shape}")
    masked = srgb * mask[..., None]
    return inverse_srgb(masked).transpose(2, 0, 1).astype(np.float32, copy=False)


def make_target_mhsp_chw(m: np.ndarray, h: np.ndarray, s: np.ndarray, p: np.ndarray, valid_mask_hw: np.ndarray) -> np.ndarray:
    """Return normalized target [4,H,W] with invalid pixels set to zero."""

    mask = np.asarray(valid_mask_hw, dtype=np.float32)
    target = np.stack(
        [
            np.asarray(m, dtype=np.float32),
            np.asarray(h, dtype=np.float32),
            (np.asarray(s, dtype=np.float32) - np.float32(0.25)) / np.float32(1.75),
            np.asarray(p, dtype=np.float32) / np.float32(0.10),
        ],
        axis=0,
    )
    target = np.clip(target, 0.0, 1.0)
    return (target * mask[None, ...]).astype(np.float32, copy=False)

