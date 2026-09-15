"""KM-BIO-v1: auditable two-layer Kubelka--Munk forward model.

This module is intentionally separate from the historical S1-4R/K2 model.
It implements the formula contract in
``face2_research/03_Method_and_Experiment/阶段一_双层KM生理模型实施规划.md``.
The parameters are model-conditional effective quantities, not measured
individual physiological concentrations.
"""

from __future__ import annotations

from pathlib import Path
from typing import Mapping

import numpy as np
import torch


WAVELENGTH_NM = np.arange(400.0, 701.0, 10.0, dtype=np.float64)
THETA_LO = np.array([0.0, 0.0, 0.0], dtype=np.float64)
THETA_HI = np.array([0.43, 0.10, 1.0], dtype=np.float64)
EPIDERMIS_THICKNESS_MM = 0.060
WHOLE_BLOOD_HB_G_L = 150.0
HB_MOLAR_MASS_G_MOL = 64500.0


def _check_theta(theta: np.ndarray | torch.Tensor) -> None:
    if theta.shape[-1] != 3:
        raise ValueError("theta must end with [f_mel, f_blood, s]")


def _check_wavelength(wavelength_nm: np.ndarray) -> np.ndarray:
    w = np.asarray(wavelength_nm, dtype=np.float64)
    if w.ndim != 1 or w.size == 0 or not np.all(np.isfinite(w)) or np.any(np.diff(w) <= 0):
        raise ValueError("wavelength_nm must be finite, non-empty and strictly increasing")
    return w


def _km_layer_numpy(k: np.ndarray, s: np.ndarray, thickness_mm: float) -> tuple[np.ndarray, np.ndarray]:
    """Finite-layer R/T from the published K-M equations, stably evaluated.

    The explicit K=0 and S=0 branches are analytical limits of the formula;
    they make corner audits meaningful and avoid 0/0 at exact boundaries.
    """

    if thickness_mm < 0 or not np.isfinite(thickness_mm):
        raise ValueError("thickness_mm must be finite and non-negative")
    k = np.asarray(k, dtype=np.float64)
    s = np.asarray(s, dtype=np.float64)
    if k.shape != s.shape or np.any(~np.isfinite(k)) or np.any(~np.isfinite(s)) or np.any(k < 0) or np.any(s < 0):
        raise ValueError("K and S must be aligned, finite and non-negative")
    r = np.zeros_like(k)
    t = np.ones_like(k)
    active = (k > 0) & (s > 0) & (thickness_mm > 0)
    if np.any(active):
        ka, sa = k[active], s[active]
        q = np.sqrt(ka * (ka + 2.0 * sa))
        beta = np.sqrt(ka / (ka + 2.0 * sa))
        x = q * thickness_mm
        ex = np.exp(-x)
        ex2 = ex * ex
        den = (1.0 + beta) ** 2 - (1.0 - beta) ** 2 * ex2
        r[active] = (1.0 - beta * beta) * (-np.expm1(-2.0 * x)) / den
        t[active] = 4.0 * beta * ex / den
    # Pure absorption: no backward flux.
    pure_abs = (k > 0) & (s == 0) & (thickness_mm > 0)
    t[pure_abs] = np.exp(-k[pure_abs] * thickness_mm)
    # Pure scattering: limit K -> 0 of the published solution.
    pure_scat = (k == 0) & (s > 0) & (thickness_mm > 0)
    sd = s[pure_scat] * thickness_mm
    r[pure_scat] = sd / (1.0 + sd)
    t[pure_scat] = 1.0 / (1.0 + sd)
    return r, t


def _km_semi_infinite_numpy(k: np.ndarray, s: np.ndarray) -> np.ndarray:
    k = np.asarray(k, dtype=np.float64)
    s = np.asarray(s, dtype=np.float64)
    if k.shape != s.shape or np.any(~np.isfinite(k)) or np.any(~np.isfinite(s)) or np.any(k < 0) or np.any(s < 0):
        raise ValueError("K and S must be aligned, finite and non-negative")
    q = np.sqrt(k * (k + 2.0 * s))
    den = k + s + q
    out = np.zeros_like(k)
    positive = den > 0
    out[positive] = s[positive] / den[positive]
    # No absorption and no scattering is the transparent limiting medium.
    out[~positive] = 0.0
    return out


def _optical_arrays_numpy(wavelength_nm: np.ndarray, asset_path: str | Path) -> tuple[np.ndarray, ...]:
    w = _check_wavelength(wavelength_nm)
    with np.load(Path(asset_path), allow_pickle=False) as z:
        source_w = np.asarray(z["wavelength_nm"], dtype=np.float64)
        mel = np.asarray(z["mua_mel_alternative_cm1"], dtype=np.float64) / 10.0
        bg = np.asarray(z["mua_base_cm1"], dtype=np.float64) / 10.0
        oxy = np.asarray(z["mua_hbo2_whole_blood_150gL_cm1"], dtype=np.float64) / 10.0
        deoxy = np.asarray(z["mua_hb_whole_blood_150gL_cm1"], dtype=np.float64) / 10.0
        musp = np.asarray(z["musp_total_cm1"], dtype=np.float64) / 10.0
    if any(a.shape != source_w.shape for a in (mel, bg, oxy, deoxy, musp)):
        raise ValueError("Optical asset arrays are not wavelength aligned")
    if w.min() < source_w.min() or w.max() > source_w.max():
        raise ValueError("Requested wavelength exceeds optical asset coverage")
    return tuple(np.interp(w, source_w, a) for a in (mel, bg, oxy, deoxy, musp))


def load_optical_numpy(wavelength_nm: np.ndarray, asset_path: str | Path) -> dict[str, np.ndarray]:
    """Load and align the five KM-BIO optical arrays once for repeated fits."""

    mel, bg, oxy, deoxy, musp = _optical_arrays_numpy(wavelength_nm, asset_path)
    return {
        "mua_mel": mel,
        "mua_bg": bg,
        "mua_hbo2": oxy,
        "mua_hb": deoxy,
        "musp": musp,
    }


def forward_preloaded_numpy(
    theta: np.ndarray,
    optical: Mapping[str, np.ndarray],
    epidermis_thickness_mm: float = EPIDERMIS_THICKNESS_MM,
    scattering_scale: float = 1.0,
    whole_blood_hb_g_l: float = WHOLE_BLOOD_HB_G_L,
) -> np.ndarray:
    """Evaluate KM-BIO-v1 from wavelength-aligned arrays already in mm^-1."""

    theta = np.asarray(theta, dtype=np.float64)
    _check_theta(theta)
    if theta.ndim != 1 or np.any(~np.isfinite(theta)) or np.any(theta < THETA_LO) or np.any(theta > THETA_HI):
        raise ValueError("theta must be one finite vector within KM-BIO-v1 bounds")
    if not np.isfinite(epidermis_thickness_mm) or epidermis_thickness_mm < 0:
        raise ValueError("epidermis_thickness_mm must be finite and non-negative")
    if not np.isfinite(scattering_scale) or scattering_scale <= 0:
        raise ValueError("scattering_scale must be finite and positive")
    if not np.isfinite(whole_blood_hb_g_l) or whole_blood_hb_g_l <= 0:
        raise ValueError("whole_blood_hb_g_l must be finite and positive")
    arrays = {name: np.asarray(optical[name], dtype=np.float64) for name in ("mua_mel", "mua_bg", "mua_hbo2", "mua_hb", "musp")}
    shape = arrays["mua_bg"].shape
    if any(value.shape != shape or value.ndim != 1 or np.any(~np.isfinite(value)) or np.any(value < 0) for value in arrays.values()):
        raise ValueError("optical arrays must be aligned finite non-negative vectors")
    fmel, fblood, oxygenation = theta
    blood_scale = whole_blood_hb_g_l / WHOLE_BLOOD_HB_G_L
    mua_e = (1.0 - fmel) * arrays["mua_bg"] + fmel * arrays["mua_mel"]
    blood = blood_scale * (oxygenation * arrays["mua_hbo2"] + (1.0 - oxygenation) * arrays["mua_hb"])
    mua_d = (1.0 - fblood) * arrays["mua_bg"] + fblood * blood
    musp = scattering_scale * arrays["musp"]
    re, te = _km_layer_numpy(2.0 * mua_e, musp, epidermis_thickness_mm)
    rd = _km_semi_infinite_numpy(2.0 * mua_d, musp)
    total = re + te * te * rd / (1.0 - re * rd)
    if np.any(~np.isfinite(total)) or np.any(total < -1e-12) or np.any(total > 1.0 + 1e-10):
        raise FloatingPointError("KM-BIO forward output violates finite/physical bounds")
    return total


def forward_numpy(
    theta: np.ndarray,
    wavelength_nm: np.ndarray,
    asset_path: str | Path,
    epidermis_thickness_mm: float = EPIDERMIS_THICKNESS_MM,
    scattering_scale: float = 1.0,
    whole_blood_hb_g_l: float = WHOLE_BLOOD_HB_G_L,
) -> np.ndarray:
    """Return 31-band (or requested-band) effective reflectance."""

    optical = load_optical_numpy(wavelength_nm, asset_path)
    return forward_preloaded_numpy(
        theta,
        optical,
        epidermis_thickness_mm=epidermis_thickness_mm,
        scattering_scale=scattering_scale,
        whole_blood_hb_g_l=whole_blood_hb_g_l,
    )


def km_layer_torch(k: torch.Tensor, s: torch.Tensor, thickness_mm: float) -> tuple[torch.Tensor, torch.Tensor]:
    """Differentiable interior implementation; exact corners are audited in NumPy."""

    if thickness_mm < 0:
        raise ValueError("thickness_mm must be non-negative")
    tiny = torch.finfo(k.dtype).tiny
    k_safe = torch.clamp(k, min=tiny)
    s_safe = torch.clamp(s, min=tiny)
    q = torch.sqrt(k_safe * (k_safe + 2.0 * s_safe))
    beta = torch.sqrt(k_safe / (k_safe + 2.0 * s_safe))
    x = q * thickness_mm
    ex = torch.exp(-x)
    den = (1.0 + beta) ** 2 - (1.0 - beta) ** 2 * ex * ex
    r = (1.0 - beta * beta) * (-torch.expm1(-2.0 * x)) / den
    t = 4.0 * beta * ex / den
    return r, t


def forward_torch(theta: torch.Tensor, optical: Mapping[str, torch.Tensor], epidermis_thickness_mm: float = EPIDERMIS_THICKNESS_MM) -> torch.Tensor:
    """Differentiable forward pass using preloaded, wavelength-aligned tensors.

    ``optical`` keys are ``mua_mel``, ``mua_bg``, ``mua_hbo2``, ``mua_hb`` and
    ``musp`` in mm^-1. Asset loading and hash checks remain in the NumPy path.
    """

    if theta.ndim != 1 or theta.shape[0] != 3:
        raise ValueError("theta must have shape (3,)")
    if not torch.is_floating_point(theta):
        raise TypeError("theta must be floating point")
    fmel, fblood, oxygenation = theta
    mua_e = (1.0 - fmel) * optical["mua_bg"] + fmel * optical["mua_mel"]
    mua_d = (1.0 - fblood) * optical["mua_bg"] + fblood * (oxygenation * optical["mua_hbo2"] + (1.0 - oxygenation) * optical["mua_hb"])
    re, te = km_layer_torch(2.0 * mua_e, optical["musp"], epidermis_thickness_mm)
    k_d, s_d = 2.0 * mua_d, optical["musp"]
    q_d = torch.sqrt(torch.clamp(k_d, min=torch.finfo(k_d.dtype).tiny) * (torch.clamp(k_d, min=torch.finfo(k_d.dtype).tiny) + 2.0 * torch.clamp(s_d, min=torch.finfo(s_d.dtype).tiny)))
    rd = s_d / (k_d + s_d + q_d)
    return re + te * te * rd / (1.0 - re * rd)
