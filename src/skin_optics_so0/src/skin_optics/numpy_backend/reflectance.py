"""Synthetic skin reflectance for the NumPy backend."""

from __future__ import annotations

import numpy as np

from skin_optics.assets import SpectralAssets
from skin_optics.config import SO0Config, load_so0_config
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
    epidermis_thickness_cm: float | None = None,
    dermis_thickness_cm: float | None = None,
    allow_sensitivity_ranges: bool = False,
    config: SO0Config | None = None,
    oxygenation: float | None = None,
    melanin_formula: str = "primary",
) -> np.ndarray:
    """Return synthetic skin reflectance with spectral dimension last."""

    cfg = config or load_so0_config()
    m_range = cfg.parameter("melanin_control")
    h_range = cfg.parameter("hemoglobin_control")
    _check_range("melanin control", m, float(m_range["min"]), float(m_range["max"]))
    _check_range("hemoglobin control", h, float(h_range["min"]), float(h_range["max"]))
    epi_d = cfg.primary_value("epidermis_thickness_cm") if epidermis_thickness_cm is None else float(epidermis_thickness_cm)
    der_d = cfg.primary_value("dermis_thickness_cm") if dermis_thickness_cm is None else float(dermis_thickness_cm)
    epi_sens = cfg.parameter("epidermis_thickness_cm")["sensitivity"]
    der_sens = cfg.parameter("dermis_thickness_cm")["sensitivity"]
    if allow_sensitivity_ranges and not float(epi_sens[0]) <= epi_d <= float(epi_sens[1]):
        raise ValueError("epidermis_thickness_cm outside sensitivity bounds")
    if allow_sensitivity_ranges and not float(der_sens[0]) <= der_d <= float(der_sens[1]):
        raise ValueError("dermis_thickness_cm outside sensitivity bounds")
    epi_abs = epidermis_absorption(assets, m, cfg, melanin_formula)
    derm_abs = dermis_absorption(assets, h, cfg, oxygenation)
    transmission = np.exp(-epi_abs * epi_d)
    dermis_r = finite_dermis_reflectance(derm_abs, reduced_scattering(assets), der_d)
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
