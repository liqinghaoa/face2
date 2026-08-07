"""Synthetic non-skin valid-mask generation."""

from __future__ import annotations

import math

import numpy as np
from scipy import ndimage

from skin_optics_so1.random_fields import rng_from_seed


MASK_TYPES = ("full", "boundary", "holes", "mixed")


def sample_mask_type(rng: np.random.Generator, probabilities: dict[str, float]) -> str:
    names = list(MASK_TYPES)
    probs = np.asarray([float(probabilities[name]) for name in names], dtype=np.float64)
    probs = probs / probs.sum()
    return str(rng.choice(names, p=probs))


def _ellipse_mask(size: int, rng: np.random.Generator) -> np.ndarray:
    yy, xx = np.mgrid[0:size, 0:size].astype(np.float32)
    cx = rng.uniform(0.15 * size, 0.85 * size)
    cy = rng.uniform(0.15 * size, 0.85 * size)
    rx = rng.uniform(0.04 * size, 0.18 * size)
    ry = rng.uniform(0.025 * size, 0.10 * size)
    theta = rng.uniform(0, math.pi)
    c, s = math.cos(theta), math.sin(theta)
    x = xx - cx
    y = yy - cy
    xr = c * x + s * y
    yr = -s * x + c * y
    return ((xr / rx) ** 2 + (yr / ry) ** 2) <= 1.0


def _boundary_invalid(size: int, rng: np.random.Generator) -> np.ndarray:
    yy, xx = np.mgrid[-1:1:complex(size), -1:1:complex(size)].astype(np.float32)
    rx = rng.uniform(0.82, 1.08)
    ry = rng.uniform(0.86, 1.10)
    face = (xx / rx) ** 2 + (yy / ry) ** 2 <= 1.0
    noise = ndimage.gaussian_filter(rng.normal(size=(size, size)).astype(np.float32), sigma=size / 18.0, mode="reflect")
    face &= noise > np.percentile(noise, rng.uniform(2, 10))
    return face


def _add_holes(mask: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    out = mask.copy()
    count = int(rng.integers(1, 5))
    for _ in range(count):
        out[_ellipse_mask(mask.shape[0], rng)] = False
    if rng.random() < 0.35:
        size = mask.shape[0]
        y0 = int(rng.uniform(0.20 * size, 0.75 * size))
        height = int(rng.uniform(0.03 * size, 0.09 * size))
        out[max(0, y0 - height) : min(size, y0 + height), :] &= rng.random(size) > 0.35
    return out


def generate_valid_mask(seed: int, size: int, mask_type: str, min_valid_fraction: float = 0.20) -> tuple[np.ndarray, float, str]:
    """Generate one [H,W] uint8 mask with valid fraction >= min_valid_fraction."""

    if mask_type not in MASK_TYPES:
        raise ValueError(f"Unknown mask_type {mask_type!r}")
    rng = rng_from_seed(seed)
    for _ in range(64):
        if mask_type == "full":
            mask = np.ones((size, size), dtype=bool)
        elif mask_type == "boundary":
            mask = _boundary_invalid(size, rng)
        elif mask_type == "holes":
            mask = _add_holes(np.ones((size, size), dtype=bool), rng)
        else:
            mask = _add_holes(_boundary_invalid(size, rng), rng)
        frac = float(mask.mean())
        if min_valid_fraction <= frac <= 1.0:
            return mask.astype(np.uint8), frac, mask_type
    raise ValueError(f"Could not generate valid {mask_type} mask with fraction >= {min_valid_fraction}")

