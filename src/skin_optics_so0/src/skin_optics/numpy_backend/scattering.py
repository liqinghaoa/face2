"""Scattering formulae for the NumPy backend."""

from __future__ import annotations

import numpy as np

from skin_optics.assets import SpectralAssets


def reduced_scattering(assets: SpectralAssets) -> np.ndarray:
    """Return the frozen total reduced scattering spectrum."""

    return assets.derived["musp_total_cm1"].astype(np.float64, copy=False)
