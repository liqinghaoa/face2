"""Synthetic skin reflectance for the NumPy backend."""

from __future__ import annotations

import numpy as np

from skin_optics.assets import SpectralAssets
from skin_optics.numpy_backend.absorption import dermis_absorption, epidermis_absorption
from skin_optics.numpy_backend.kubelka_munk import finite_dermis_reflectance
from skin_optics.numpy_backend.scattering import reduced_scattering


def _check_range(name: str, value: np.ndarray | float, min_value: float, max_value: float) -> None:
    arr = np.asarray(value, dtype=np.float64)
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} contains non-finite values")
    if np.any((arr < min_value) | (arr > max_value)):
        raise ValueError(f"{name} outside [{min_value}, {max_value}]")


def compute_skin_reflectance(
    assets: SpectralAssets,
    m: np.ndarray | float,
    h: np.ndarray | float,
    epidermis_thickness_cm: float = 0.006,
    dermis_thickness_cm: float = 0.20,
    allow_sensitivity_ranges: bool = False,
) -> np.ndarray:
    """Return synthetic skin reflectance with spectral dimension last."""

    del allow_sensitivity_ranges
    _check_range("melanin control", m, 0.0, 1.0)
    _check_range("hemoglobin control", h, 0.0, 1.0)
    if not 0.004 <= epidermis_thickness_cm <= 0.010:
        raise ValueError("epidermis_thickness_cm outside sensitivity bounds")
    if not 0.10 <= dermis_thickness_cm <= 0.30:
        raise ValueError("dermis_thickness_cm outside sensitivity bounds")
    epi_abs = epidermis_absorption(assets, m)
    derm_abs = dermis_absorption(assets, h)
    transmission = np.exp(-epi_abs * epidermis_thickness_cm)
    dermis_r = finite_dermis_reflectance(derm_abs, reduced_scattering(assets), dermis_thickness_cm)
    result = transmission * transmission * dermis_r
    if not np.all(np.isfinite(result)):
        raise ValueError("Non-finite skin reflectance")
    return result.astype(np.float64, copy=False)


def compute_skin_reflectance_bchw(
    assets: SpectralAssets,
    m: np.ndarray,
    h: np.ndarray,
    **kwargs: object,
) -> np.ndarray:
    """BCHW wrapper returning [B, N_lambda, H, W]."""

    r = compute_skin_reflectance(assets, m, h, **kwargs)
    if r.ndim != 5 or r.shape[1] != 1:
        raise ValueError("BCHW wrapper expects parameter shape [B,1,H,W]")
    return np.moveaxis(r[:, 0, :, :, :], -1, 1)
