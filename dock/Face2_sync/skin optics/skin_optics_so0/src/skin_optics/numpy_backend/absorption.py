"""Absorption formulae for the NumPy backend."""

from __future__ import annotations

import numpy as np

from skin_optics.assets import SpectralAssets


def melanin_fraction(m: np.ndarray | float) -> np.ndarray:
    """Map melanin-sensitive control to a synthetic epidermal fraction."""

    return 0.013 + 0.417 * np.asarray(m, dtype=np.float64)


def blood_fraction(h: np.ndarray | float) -> np.ndarray:
    """Map hemoglobin-sensitive control to a synthetic dermal blood fraction."""

    return 0.02 + 0.05 * np.asarray(h, dtype=np.float64)


def epidermis_absorption(assets: SpectralAssets, m: np.ndarray | float) -> np.ndarray:
    """Compute epidermis absorption with spectral dimension last."""

    f_mel = melanin_fraction(m)[..., None]
    return f_mel * assets.derived["mua_mel_primary_cm1"] + (1.0 - f_mel) * assets.derived["mua_base_cm1"]


def dermis_absorption(assets: SpectralAssets, h: np.ndarray | float) -> np.ndarray:
    """Compute dermis absorption with fixed oxygenation y=0.75."""

    f_blood = blood_fraction(h)[..., None]
    return f_blood * assets.derived["mua_blood_y075_cm1"] + (1.0 - f_blood) * assets.derived["mua_base_cm1"]
