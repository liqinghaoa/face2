"""Monotonicity audits."""

from __future__ import annotations

import numpy as np
from scipy.stats import spearmanr

from skin_optics.assets import SpectralAssets
from skin_optics.numpy_backend.color_spaces import xyz_to_lab_d65
from skin_optics.numpy_backend.image_formation import d65_white_xyz, render_cie_reference
from skin_optics.numpy_backend.reflectance import compute_skin_reflectance


def band_index(assets: SpectralAssets, reflectance: np.ndarray) -> np.ndarray:
    """Compute the hemoglobin-sensitive band index B_H with integration weights."""

    wl = assets.wavelength_nm
    weights = assets.weights_nm
    green = (wl >= 545.0) & (wl <= 585.0)
    red = (wl >= 650.0) & (wl <= 700.0)
    mean_green = (reflectance[..., green] * weights[green]).sum(axis=-1) / weights[green].sum()
    mean_red = (reflectance[..., red] * weights[red]).sum(axis=-1) / weights[red].sum()
    return -np.log(mean_green / mean_red)


def audit_monotonicity(assets: SpectralAssets) -> dict[str, float]:
    """Return hard monotonicity metrics."""

    m_grid = np.linspace(0.0, 1.0, 21)
    h_grid = np.linspace(0.0, 1.0, 21)
    r_m = compute_skin_reflectance(assets, m_grid, np.full_like(m_grid, 0.5))
    max_increase = float(np.max(np.diff(r_m, axis=0)))
    r_h = compute_skin_reflectance(assets, np.full_like(h_grid, 0.5), h_grid)
    rho_h = float(spearmanr(band_index(assets, r_h), h_grid).statistic)
    cie = render_cie_reference(assets, r_m, "D65")
    lstar = xyz_to_lab_d65(cie.xyz_d65, d65_white_xyz(assets))[..., 0]
    rho_l = float(spearmanr(lstar, m_grid).statistic)
    return {
        "melanin_max_reflectance_increase": max_increase,
        "hemoglobin_band_spearman": rho_h,
        "melanin_lstar_spearman": rho_l,
    }
