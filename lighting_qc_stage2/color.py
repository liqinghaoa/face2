from __future__ import annotations

import numpy as np


def inverse_srgb(srgb: np.ndarray) -> np.ndarray:
    c = np.clip(np.asarray(srgb, dtype=np.float64), 0.0, 1.0)
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)


def rgb_uint8_to_linear(rgb: np.ndarray) -> np.ndarray:
    if rgb.dtype != np.uint8 or rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ValueError("RGB image must be uint8 HxWx3")
    return inverse_srgb(rgb.astype(np.float64) / 255.0)


def luminance_y(linear_rgb: np.ndarray) -> np.ndarray:
    return 0.2126 * linear_rgb[..., 0] + 0.7152 * linear_rgb[..., 1] + 0.0722 * linear_rgb[..., 2]
