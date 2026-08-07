"""Deterministic synthetic M/H/S/P field generation."""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass

import numpy as np
from scipy import ndimage


COMPONENTS = ("m_field", "h_field", "s_field", "p_field", "mask", "acquisition", "camera_light")


def stable_seed(global_seed: int, split: str, base_latent_id: int | str, acquisition_variant_id: int, component_name: str) -> int:
    """Return a cross-process-stable uint64 seed based on SHA256."""

    if component_name not in COMPONENTS:
        raise ValueError(f"Unknown seed component {component_name!r}")
    payload = f"{int(global_seed)}|{split}|{base_latent_id}|{int(acquisition_variant_id)}|{component_name}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big", signed=False)


def rng_from_seed(seed: int) -> np.random.Generator:
    return np.random.Generator(np.random.PCG64(int(seed) & ((1 << 64) - 1)))


@dataclass(frozen=True)
class FieldStats:
    mean: float
    std: float
    min: float
    max: float
    boundary_fraction: float = 0.0
    nonzero_fraction: float = 0.0
    blob_count: int = 0


def _resize_smooth(field: np.ndarray, size: int) -> np.ndarray:
    zoom = (size / field.shape[0], size / field.shape[1])
    out = ndimage.zoom(field, zoom, order=3, mode="reflect", prefilter=True)
    return out[:size, :size]


def _standardize(field: np.ndarray) -> np.ndarray:
    field = np.asarray(field, dtype=np.float32)
    field = ndimage.gaussian_filter(field, sigma=1.0, mode="reflect")
    std = float(field.std())
    if std < 1e-8:
        return np.zeros_like(field, dtype=np.float32)
    return ((field - float(field.mean())) / std).astype(np.float32)


def _low_frequency_field(rng: np.random.Generator, size: int) -> np.ndarray:
    f4 = _resize_smooth(rng.normal(size=(4, 4)).astype(np.float32), size)
    f8 = _resize_smooth(rng.normal(size=(8, 8)).astype(np.float32), size)
    f16 = _resize_smooth(rng.normal(size=(16, 16)).astype(np.float32), size)
    return _standardize(0.50 * f4 + 0.30 * f8 + 0.20 * f16)


def _anisotropic_gaussian(size: int, rng: np.random.Generator, broad: bool = True) -> np.ndarray:
    yy, xx = np.mgrid[0:size, 0:size].astype(np.float32)
    cx = rng.uniform(0.15 * size, 0.85 * size)
    cy = rng.uniform(0.15 * size, 0.85 * size)
    theta = rng.uniform(0, math.pi)
    c, s = math.cos(theta), math.sin(theta)
    x = xx - cx
    y = yy - cy
    xr = c * x + s * y
    yr = -s * x + c * y
    if broad:
        sx = rng.uniform(0.08 * size, 0.24 * size)
        sy = rng.uniform(0.05 * size, 0.18 * size)
    else:
        sx = rng.uniform(0.025 * size, 0.09 * size)
        sy = rng.uniform(0.020 * size, 0.07 * size)
    return np.exp(-0.5 * ((xr / sx) ** 2 + (yr / sy) ** 2)).astype(np.float32)


def _local_structures(rng: np.random.Generator, size: int, max_count: int, amp_range: tuple[float, float], signed: bool) -> tuple[np.ndarray, int]:
    count = int(rng.integers(0, max_count + 1))
    out = np.zeros((size, size), dtype=np.float32)
    for _ in range(count):
        amp = rng.uniform(*amp_range)
        if signed and rng.random() < 0.5:
            amp = -amp
        out += np.float32(amp) * _anisotropic_gaussian(size, rng, broad=True)
    return out.astype(np.float32), count


def generate_mh_field(base: float, seed: int, size: int) -> tuple[np.ndarray, FieldStats]:
    rng = rng_from_seed(seed)
    low = _low_frequency_field(rng, size)
    amp = rng.uniform(0.04, 0.18)
    local, count = _local_structures(rng, size, 3, (0.025, 0.14), signed=True)
    field = np.clip(np.float32(base) + np.float32(amp) * low + local, 0.0, 1.0).astype(np.float32)
    boundary = float(np.mean((field <= 1e-6) | (field >= 1.0 - 1e-6)))
    return field, FieldStats(float(field.mean()), float(field.std()), float(field.min()), float(field.max()), boundary, blob_count=count)


def generate_s_field(seed: int, size: int, shading_min: float = 0.25, shading_max: float = 2.0) -> tuple[np.ndarray, FieldStats]:
    rng = rng_from_seed(seed)
    yy, xx = np.mgrid[-1:1:complex(size), -1:1:complex(size)].astype(np.float32)
    base = rng.uniform(0.75, 1.25)
    gx = rng.uniform(-0.35, 0.35)
    gy = rng.uniform(-0.35, 0.35)
    low = rng.uniform(0.02, 0.22) * _low_frequency_field(rng, size)
    shadows = np.zeros((size, size), dtype=np.float32)
    count = int(rng.integers(0, 3))
    for _ in range(count):
        shadows -= np.float32(rng.uniform(0.08, 0.35)) * _anisotropic_gaussian(size, rng, broad=True)
    s = np.clip(base + gx * xx + gy * yy + low + shadows, shading_min, shading_max).astype(np.float32)
    return s, FieldStats(float(s.mean()), float(s.std()), float(s.min()), float(s.max()), blob_count=count)


def sample_p_blob_count(rng: np.random.Generator) -> int:
    u = float(rng.random())
    if u < 0.30:
        return 0
    if u < 0.75:
        return 1
    if u < 0.95:
        return 2
    return int(rng.integers(3, 5))


def generate_p_field(seed: int, size: int, specular_max: float = 0.10) -> tuple[np.ndarray, FieldStats]:
    rng = rng_from_seed(seed)
    count = sample_p_blob_count(rng)
    p = np.zeros((size, size), dtype=np.float32)
    for _ in range(count):
        amp = rng.uniform(0.015, specular_max)
        p += np.float32(amp) * _anisotropic_gaussian(size, rng, broad=False)
    p = np.clip(p, 0.0, specular_max).astype(np.float32)
    nz = float(np.mean(p > 1e-5))
    return p, FieldStats(float(p.mean()), float(p.std()), float(p.min()), float(p.max()), nonzero_fraction=nz, blob_count=count)


def sobol_base_values(n: int, seed: int) -> np.ndarray:
    """Return deterministic 2D Sobol values in [0.05, 0.95]."""

    if n <= 0:
        return np.empty((0, 2), dtype=np.float32)
    try:
        import torch

        engine = torch.quasirandom.SobolEngine(dimension=2, scramble=True, seed=int(seed % (2**31 - 1)))
        values = engine.draw(n).cpu().numpy().astype(np.float32)
    except Exception:
        rng = rng_from_seed(seed)
        values = rng.random((n, 2), dtype=np.float32)
    return (0.05 + 0.90 * values).astype(np.float32)

