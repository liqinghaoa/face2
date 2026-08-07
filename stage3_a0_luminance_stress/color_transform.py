from __future__ import annotations

import numpy as np


def srgb_to_linear(srgb: np.ndarray) -> np.ndarray:
    x = np.asarray(srgb, dtype=np.float32)
    return np.where(x <= 0.04045, x / 12.92, ((x + 0.055) / 1.055) ** 2.4)


def linear_to_srgb(linear: np.ndarray) -> np.ndarray:
    x = np.clip(np.asarray(linear, dtype=np.float32), 0.0, 1.0)
    return np.where(x <= 0.0031308, x * 12.92, 1.055 * (x ** (1.0 / 2.4)) - 0.055)


def apply_exposure_ev_uint8(rgb: np.ndarray, mask: np.ndarray, delta_ev: float) -> np.ndarray:
    if float(delta_ev) == 0.0:
        return np.asarray(rgb, dtype=np.uint8).copy()
    img = np.asarray(rgb, dtype=np.float32) / 255.0
    out = img.copy()
    face = np.asarray(mask) > 0
    linear = srgb_to_linear(img[face])
    out[face] = linear_to_srgb(np.clip(linear * (2.0 ** float(delta_ev)), 0.0, 1.0))
    out[~face] = 0.0
    return np.clip(np.rint(out * 255.0), 0, 255).astype(np.uint8)


def apply_gamma_uint8(rgb: np.ndarray, mask: np.ndarray, gamma: float) -> np.ndarray:
    img = np.asarray(rgb, dtype=np.float32) / 255.0
    out = img.copy()
    face = np.asarray(mask) > 0
    out[face] = np.clip(img[face] ** float(gamma), 0.0, 1.0)
    out[~face] = 0.0
    return np.clip(np.rint(out * 255.0), 0, 255).astype(np.uint8)
