"""Absorption formulae for the NumPy backend."""

from __future__ import annotations

import numpy as np

from skin_optics.assets import SpectralAssets
from skin_optics.config import SO0Config, load_so0_config


def melanin_fraction(m: np.ndarray | float, config: SO0Config | None = None) -> np.ndarray:
    """Map melanin-sensitive control to a synthetic epidermal fraction."""

    cfg = config or load_so0_config()
    bounds = cfg.parameter("melanin_fraction")
    lo = float(bounds["min"])
    span = float(bounds["max"]) - lo
    return lo + span * np.asarray(m, dtype=np.float64)


def blood_fraction(h: np.ndarray | float, config: SO0Config | None = None) -> np.ndarray:
    """Map hemoglobin-sensitive control to a synthetic dermal blood fraction."""

    cfg = config or load_so0_config()
    bounds = cfg.parameter("blood_fraction")
    lo = float(bounds["min"])
    span = float(bounds["max"]) - lo
    return lo + span * np.asarray(h, dtype=np.float64)


def epidermis_absorption(
    assets: SpectralAssets,
    m: np.ndarray | float,
    config: SO0Config | None = None,
    melanin_formula: str = "primary",
) -> np.ndarray:
    """Compute epidermis absorption with spectral dimension last."""

    f_mel = melanin_fraction(m, config)[..., None]
    key = "mua_mel_primary_cm1" if melanin_formula == "primary" else "mua_mel_alternative_cm1"
    return f_mel * assets.derived[key] + (1.0 - f_mel) * assets.derived["mua_base_cm1"]


def dermis_absorption(
    assets: SpectralAssets,
    h: np.ndarray | float,
    config: SO0Config | None = None,
    oxygenation: float | None = None,
) -> np.ndarray:
    """Compute dermis absorption with fixed oxygenation y=0.75."""

    cfg = config or load_so0_config()
    oxy = cfg.primary_value("oxygenation") if oxygenation is None else float(oxygenation)
    if abs(oxy - 0.60) < 1e-12:
        blood_key = "mua_blood_y060_cm1"
    elif abs(oxy - 0.75) < 1e-12:
        blood_key = "mua_blood_y075_cm1"
    elif abs(oxy - 0.90) < 1e-12:
        blood_key = "mua_blood_y090_cm1"
    else:
        raise ValueError("oxygenation must be one of the configured audited values: 0.60, 0.75, 0.90")
    f_blood = blood_fraction(h, cfg)[..., None]
    return f_blood * assets.derived[blood_key] + (1.0 - f_blood) * assets.derived["mua_base_cm1"]
