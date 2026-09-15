"""KM-BIO-v2R independent formula contract implementation.

This module is intentionally independent from ``km_bio_v1``.  It implements
the v2R formula contract only; it does not read HSI observations or perform
inverse fitting.  Returned parameters remain model-conditional effective
quantities rather than direct physiological measurements.
"""

from __future__ import annotations

from pathlib import Path
from typing import Mapping

import numpy as np
import torch


MODEL_ID = "KM2L-HF-v2R"
WAVELENGTH_NM = np.arange(400.0, 701.0, 10.0, dtype=np.float64)
THETA_BIO_LO = np.array([0.0, 0.0], dtype=np.float64)
THETA_BIO_HI = np.array([0.43, 0.10], dtype=np.float64)
THETA_EXTENDED_LO = np.array([0.0, 0.0, 0.0], dtype=np.float64)
THETA_EXTENDED_HI = np.array([0.43, 0.10, 1.0], dtype=np.float64)
S0_DEFAULT = 0.70
EPIDERMIS_THICKNESS_MM = 0.060
WHOLE_BLOOD_HB_G_L = 150.0
HB_MOLAR_MASS_G_MOL = 64500.0
DV_UM_DEFAULT = 15.0
DV_UM_SENSITIVITY = (0.0, 7.5, 15.0, 30.0)
AS_DEFAULT = 1.0
DELTA_BS_DEFAULT = 0.0
AS_BOUNDS = (0.6, 1.6)
DELTA_BS_BOUNDS = (-0.5, 0.5)
G0_BOUNDS = (0.5, 1.5)
GLOBAL_PROFILE_DELTA_LOGRMSE = 0.005
GLOBAL_PROFILE_MAX_NORMALIZED_SPAN = 0.20


def _validate_wavelength(wavelength_nm: np.ndarray) -> np.ndarray:
    w = np.asarray(wavelength_nm, dtype=np.float64)
    if w.ndim != 1 or not w.size or not np.isfinite(w).all() or np.any(np.diff(w) <= 0):
        raise ValueError("wavelength_nm must be finite, non-empty and strictly increasing")
    return w


def _validate_scalar(name: str, value: float, lower: float | None = None, upper: float | None = None) -> float:
    value = float(value)
    if not np.isfinite(value):
        raise ValueError(f"{name} must be finite")
    if lower is not None and value < lower:
        raise ValueError(f"{name} must be >= {lower}")
    if upper is not None and value > upper:
        raise ValueError(f"{name} must be <= {upper}")
    return value


def _check_theta(theta: np.ndarray, s0: float) -> tuple[np.ndarray, float]:
    theta = np.asarray(theta, dtype=np.float64)
    if theta.ndim != 1 or theta.size not in (2, 3) or not np.isfinite(theta).all():
        raise ValueError("theta must be [f_mel, f_blood] or [f_mel, f_blood, s]")
    bounds_lo = THETA_BIO_LO if theta.size == 2 else THETA_EXTENDED_LO
    bounds_hi = THETA_BIO_HI if theta.size == 2 else THETA_EXTENDED_HI
    if np.any(theta < bounds_lo) or np.any(theta > bounds_hi):
        raise ValueError("theta is outside KM-BIO-v2R bounds")
    s = _validate_scalar("s0", s0, 0.0, 1.0) if theta.size == 2 else float(theta[2])
    return theta, s


def _check_optical_numpy(optical: Mapping[str, np.ndarray]) -> dict[str, np.ndarray]:
    names = ("mua_mel", "mua_bg", "mua_hbo2", "mua_hb", "musp")
    arrays = {name: np.asarray(optical[name], dtype=np.float64) for name in names}
    shape = arrays["mua_bg"].shape
    if any(value.ndim != 1 or value.shape != shape or not np.isfinite(value).all() or np.any(value < 0) for value in arrays.values()):
        raise ValueError("optical arrays must be aligned finite non-negative vectors")
    return arrays


def _km_layer_numpy(k: np.ndarray, s: np.ndarray, thickness_mm: float) -> tuple[np.ndarray, np.ndarray]:
    """Stable finite-layer two-flux K-M reflection and transmission."""

    thickness_mm = _validate_scalar("thickness_mm", thickness_mm, 0.0)
    k = np.asarray(k, dtype=np.float64)
    s = np.asarray(s, dtype=np.float64)
    if k.shape != s.shape or not np.isfinite(k).all() or not np.isfinite(s).all() or np.any(k < 0) or np.any(s < 0):
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
    pure_abs = (k > 0) & (s == 0) & (thickness_mm > 0)
    t[pure_abs] = np.exp(-k[pure_abs] * thickness_mm)
    pure_scat = (k == 0) & (s > 0) & (thickness_mm > 0)
    sd = s[pure_scat] * thickness_mm
    r[pure_scat] = sd / (1.0 + sd)
    t[pure_scat] = 1.0 / (1.0 + sd)
    return r, t


def _km_semi_infinite_numpy(k: np.ndarray, s: np.ndarray) -> np.ndarray:
    k = np.asarray(k, dtype=np.float64)
    s = np.asarray(s, dtype=np.float64)
    if k.shape != s.shape or not np.isfinite(k).all() or not np.isfinite(s).all() or np.any(k < 0) or np.any(s < 0):
        raise ValueError("K and S must be aligned, finite and non-negative")
    q = np.sqrt(k * (k + 2.0 * s))
    den = k + s + q
    out = np.zeros_like(k)
    positive = den > 0
    out[positive] = s[positive] / den[positive]
    return out


def blood_packaging_factor_numpy(mu_a_blood: np.ndarray, diameter_um: float) -> np.ndarray:
    """Return ``(1-exp(-mu_a*D))/ (mu_a*D)`` with D converted um -> mm."""

    mu = np.asarray(mu_a_blood, dtype=np.float64)
    if not np.isfinite(mu).all() or np.any(mu < 0):
        raise ValueError("mu_a_blood must be finite and non-negative")
    diameter_um = _validate_scalar("diameter_um", diameter_um, 0.0)
    z = mu * (diameter_um / 1000.0)
    out = np.ones_like(z)
    positive = z > 0
    out[positive] = -np.expm1(-z[positive]) / z[positive]
    return out


def blood_packaging_factor_torch(mu_a_blood: torch.Tensor, diameter_um: float) -> torch.Tensor:
    diameter_um = _validate_scalar("diameter_um", diameter_um, 0.0)
    if not torch.is_floating_point(mu_a_blood) or torch.any(~torch.isfinite(mu_a_blood)) or torch.any(mu_a_blood < 0):
        raise ValueError("mu_a_blood must be floating, finite and non-negative")
    z = mu_a_blood * (diameter_um / 1000.0)
    positive = z > 0
    safe_z = torch.where(positive, z, torch.ones_like(z))
    return torch.where(positive, -torch.expm1(-z) / safe_z, torch.ones_like(z))


def _scattering_numpy(musp_ref: np.ndarray, wavelength_nm: np.ndarray, amplitude: float, delta_bs: float) -> np.ndarray:
    amplitude = _validate_scalar("A_s", amplitude, *AS_BOUNDS)
    delta_bs = _validate_scalar("delta_bs", delta_bs, *DELTA_BS_BOUNDS)
    w = _validate_wavelength(wavelength_nm)
    return amplitude * musp_ref * np.power(w / 600.0, -delta_bs)


def forward_preloaded_numpy(
    theta: np.ndarray,
    optical: Mapping[str, np.ndarray],
    *,
    wavelength_nm: np.ndarray | None = None,
    s0: float = S0_DEFAULT,
    epidermis_thickness_mm: float = EPIDERMIS_THICKNESS_MM,
    diameter_um: float = DV_UM_DEFAULT,
    scattering_amplitude: float = AS_DEFAULT,
    delta_bs: float = DELTA_BS_DEFAULT,
    g0: float = 1.0,
) -> np.ndarray:
    """Evaluate v2R from aligned optical arrays in mm^-1.

    A two-element theta uses fixed ``s0``; a three-element theta opens the
    oxygenation extension.  ``g0`` is applied after the K-M reflectance and
    is deliberately not clipped.
    """

    theta, oxygenation = _check_theta(theta, s0)
    arrays = _check_optical_numpy(optical)
    if wavelength_nm is None:
        if arrays["mua_bg"].size != WAVELENGTH_NM.size:
            raise ValueError("wavelength_nm is required for non-standard optical arrays")
        w = WAVELENGTH_NM
    else:
        w = _validate_wavelength(wavelength_nm)
    if w.size != arrays["mua_bg"].size:
        raise ValueError("wavelength_nm and optical arrays must have equal length")
    thickness = _validate_scalar("epidermis_thickness_mm", epidermis_thickness_mm, 0.0)
    diameter = _validate_scalar("diameter_um", diameter_um, 0.0)
    g0 = _validate_scalar("g0", g0, *G0_BOUNDS)
    fmel, fblood = theta[:2]
    mua_e = (1.0 - fmel) * arrays["mua_bg"] + fmel * arrays["mua_mel"]
    mua_blood = oxygenation * arrays["mua_hbo2"] + (1.0 - oxygenation) * arrays["mua_hb"]
    mua_d = (1.0 - fblood) * arrays["mua_bg"] + fblood * blood_packaging_factor_numpy(mua_blood, diameter) * mua_blood
    musp = _scattering_numpy(arrays["musp"], w, scattering_amplitude, delta_bs)
    re, te = _km_layer_numpy(2.0 * mua_e, musp, thickness)
    rd = _km_semi_infinite_numpy(2.0 * mua_d, musp)
    km = re + te * te * rd / (1.0 - re * rd)
    out = g0 * km
    if not np.isfinite(out).all() or np.any(out < -1e-12):
        raise FloatingPointError("KM-BIO-v2R output is non-finite or negative")
    return out


def load_optical_numpy(wavelength_nm: np.ndarray, asset_path: str | Path) -> dict[str, np.ndarray]:
    """Load the v1 shared optical asset read-only and convert cm^-1 to mm^-1."""

    w = _validate_wavelength(wavelength_nm)
    with np.load(Path(asset_path), allow_pickle=False) as z:
        source_w = np.asarray(z["wavelength_nm"], dtype=np.float64)
        fields = {
            "mua_mel": "mua_mel_alternative_cm1",
            "mua_bg": "mua_base_cm1",
            "mua_hbo2": "mua_hbo2_whole_blood_150gL_cm1",
            "mua_hb": "mua_hb_whole_blood_150gL_cm1",
            "musp": "musp_total_cm1",
        }
        arrays = {name: np.asarray(z[field], dtype=np.float64) / 10.0 for name, field in fields.items()}
    if source_w.ndim != 1 or not np.isfinite(source_w).all() or any(a.shape != source_w.shape for a in arrays.values()):
        raise ValueError("Optical asset arrays are not wavelength aligned")
    if w.min() < source_w.min() or w.max() > source_w.max():
        raise ValueError("Requested wavelength exceeds optical asset coverage")
    return {name: np.interp(w, source_w, values) for name, values in arrays.items()}


def km_layer_torch(k: torch.Tensor, s: torch.Tensor, thickness_mm: float) -> tuple[torch.Tensor, torch.Tensor]:
    thickness_mm = _validate_scalar("thickness_mm", thickness_mm, 0.0)
    if k.shape != s.shape or not torch.is_floating_point(k) or not torch.is_floating_point(s):
        raise ValueError("K and S must be aligned floating tensors")
    tiny_k = torch.finfo(k.dtype).tiny
    tiny_s = torch.finfo(s.dtype).tiny
    k_safe = torch.clamp(k, min=tiny_k)
    s_safe = torch.clamp(s, min=tiny_s)
    q = torch.sqrt(k_safe * (k_safe + 2.0 * s_safe))
    beta = torch.sqrt(k_safe / (k_safe + 2.0 * s_safe))
    x = q * thickness_mm
    ex = torch.exp(-x)
    den = (1.0 + beta) ** 2 - (1.0 - beta) ** 2 * ex * ex
    r = (1.0 - beta * beta) * (-torch.expm1(-2.0 * x)) / den
    t = 4.0 * beta * ex / den
    return r, t


def forward_torch(
    theta: torch.Tensor,
    optical: Mapping[str, torch.Tensor],
    *,
    wavelength_nm: torch.Tensor | np.ndarray | None = None,
    s0: float = S0_DEFAULT,
    epidermis_thickness_mm: float = EPIDERMIS_THICKNESS_MM,
    diameter_um: float = DV_UM_DEFAULT,
    scattering_amplitude: float = AS_DEFAULT,
    delta_bs: float = DELTA_BS_DEFAULT,
    g0: float = 1.0,
) -> torch.Tensor:
    """Differentiable v2R forward pass for aligned optical tensors."""

    if theta.ndim != 1 or theta.numel() not in (2, 3) or not torch.is_floating_point(theta):
        raise ValueError("theta must be a floating vector with length 2 or 3")
    theta_values = theta.detach().cpu().numpy()
    theta_values, oxygenation_value = _check_theta(theta_values, s0)
    names = ("mua_mel", "mua_bg", "mua_hbo2", "mua_hb", "musp")
    if any(name not in optical for name in names):
        raise KeyError("optical is missing a required v2R field")
    shape = optical["mua_bg"].shape
    if any(optical[name].shape != shape for name in names):
        raise ValueError("optical tensors must be aligned")
    if wavelength_nm is None:
        if shape[0] != WAVELENGTH_NM.size:
            raise ValueError("wavelength_nm is required for non-standard optical tensors")
        w = torch.as_tensor(WAVELENGTH_NM, dtype=theta.dtype, device=theta.device)
    else:
        w = torch.as_tensor(wavelength_nm, dtype=theta.dtype, device=theta.device)
    if w.ndim != 1 or w.numel() != shape[0]:
        raise ValueError("wavelength_nm and optical tensors must have equal length")
    amplitude = _validate_scalar("A_s", scattering_amplitude, *AS_BOUNDS)
    delta = _validate_scalar("delta_bs", delta_bs, *DELTA_BS_BOUNDS)
    diameter = _validate_scalar("diameter_um", diameter_um, 0.0)
    thickness = _validate_scalar("epidermis_thickness_mm", epidermis_thickness_mm, 0.0)
    gain = _validate_scalar("g0", g0, *G0_BOUNDS)
    fmel, fblood = theta[0], theta[1]
    mua_e = (1.0 - fmel) * optical["mua_bg"] + fmel * optical["mua_mel"]
    oxygenation = theta[2] if theta.numel() == 3 else oxygenation_value
    mua_blood = oxygenation * optical["mua_hbo2"] + (1.0 - oxygenation) * optical["mua_hb"]
    package = blood_packaging_factor_torch(mua_blood, diameter)
    mua_d = (1.0 - fblood) * optical["mua_bg"] + fblood * package * mua_blood
    musp = amplitude * optical["musp"] * torch.pow(w / 600.0, -delta)
    re, te = km_layer_torch(2.0 * mua_e, musp, thickness)
    k_d, s_d = 2.0 * mua_d, musp
    tiny = torch.finfo(theta.dtype).tiny
    k_safe, s_safe = torch.clamp(k_d, min=tiny), torch.clamp(s_d, min=tiny)
    q_d = torch.sqrt(k_safe * (k_safe + 2.0 * s_safe))
    rd = s_d / (k_d + s_d + q_d)
    return gain * (re + te * te * rd / (1.0 - re * rd))


def estimate_shape_scales(
    observed: np.ndarray,
    prediction_grid: np.ndarray,
    amplitude_grid: np.ndarray,
    delta_grid: np.ndarray,
    epsilon: float = 1e-6,
) -> dict[str, object]:
    """Select Train-global ``A_s, delta_bs`` from full-forward predictions.

    ``prediction_grid`` must have shape ``[n_A, n_delta, n_spectra,
    n_bands]``.  The caller must generate every candidate through the full
    K-M forward model; scattering is not a multiplicative reflectance scale.
    """

    observed = np.asarray(observed, dtype=np.float64)
    predictions = np.asarray(prediction_grid, dtype=np.float64)
    amplitudes = np.asarray(amplitude_grid, dtype=np.float64)
    deltas = np.asarray(delta_grid, dtype=np.float64)
    if amplitudes.ndim != 1 or deltas.ndim != 1 or not amplitudes.size or not deltas.size:
        raise ValueError("global calibration grids must be non-empty vectors")
    expected_shape = (amplitudes.size, deltas.size, *observed.shape)
    if observed.ndim != 2 or predictions.shape != expected_shape or not np.isfinite(observed).all() or not np.isfinite(predictions).all() or np.any(observed <= 0) or np.any(predictions <= 0):
        raise ValueError("observed and prediction_grid must be aligned positive finite spectra")
    for value in amplitudes:
        _validate_scalar("A_s grid value", value, *AS_BOUNDS)
    for value in deltas:
        _validate_scalar("delta_bs grid value", value, *DELTA_BS_BOUNDS)
    log_obs = np.log(observed + epsilon)
    profiles: list[dict[str, float]] = []
    for amplitude_index, amplitude in enumerate(amplitudes):
        for delta_index, delta in enumerate(deltas):
            residual = np.log(predictions[amplitude_index, delta_index] + epsilon) - log_obs
            centered = residual - residual.mean(axis=1, keepdims=True)
            shape_loss = float(np.sqrt(np.mean(centered**2)))
            amplitude_loss = float(np.sqrt(np.mean(residual.mean(axis=1) ** 2)))
            profiles.append({"A_s": float(amplitude), "delta_bs": float(delta), "loss": shape_loss, "centered_logrmse": shape_loss, "raw_log_amplitude_rmse_diagnostic": amplitude_loss})
    best = min(profiles, key=lambda row: row["loss"])
    return {"A_s": best["A_s"], "delta_bs": best["delta_bs"], "profile": profiles, "estimation": "centered_log_shape_grid_full_km_predictions"}


def estimate_global_gain(
    observed: np.ndarray,
    predicted: np.ndarray,
    bounds: tuple[float, float] = G0_BOUNDS,
    profile_count: int = 201,
) -> dict[str, object]:
    """Estimate one bounded raw-log scale and return its amplitude profile."""

    observed = np.asarray(observed, dtype=np.float64)
    predicted = np.asarray(predicted, dtype=np.float64)
    if observed.shape != predicted.shape or observed.ndim != 2 or not np.isfinite(observed).all() or not np.isfinite(predicted).all() or np.any(observed <= 0) or np.any(predicted <= 0):
        raise ValueError("observed and predicted must be aligned positive finite spectra")
    lo, hi = (_validate_scalar("g0 lower", bounds[0], 0.0), _validate_scalar("g0 upper", bounds[1], 0.0))
    if hi <= lo:
        raise ValueError("g0 bounds must be ordered")
    if profile_count < 3:
        raise ValueError("profile_count must be at least 3")
    log_difference = np.log(observed) - np.log(predicted)
    raw = float(np.exp(np.mean(log_difference)))
    estimate = float(np.clip(raw, lo, hi))
    grid = np.unique(np.append(np.linspace(lo, hi, profile_count, dtype=np.float64), estimate))
    profile = [
        {"g0": float(value), "raw_logrmse": float(np.sqrt(np.mean((np.log(value) - log_difference) ** 2)))}
        for value in grid
    ]
    optimum_loss = float(np.sqrt(np.mean((np.log(estimate) - log_difference) ** 2)))
    return {
        "g0": estimate,
        "unclipped_g0": raw,
        "raw_logrmse": optimum_loss,
        "at_lower": estimate == lo,
        "at_upper": estimate == hi,
        "profile": profile,
        "estimation": "analytic_bounded_raw_logrmse",
    }


def profile_identifiability(
    profile: list[dict[str, float]],
    parameter: str,
    loss_key: str,
    bounds: tuple[float, float],
    *,
    delta_logrmse: float = GLOBAL_PROFILE_DELTA_LOGRMSE,
    max_normalized_span: float = GLOBAL_PROFILE_MAX_NORMALIZED_SPAN,
) -> dict[str, object]:
    """Apply the pre-registered acceptable-profile span rule."""

    lo, hi = float(bounds[0]), float(bounds[1])
    if hi <= lo or not profile:
        raise ValueError("profile bounds and rows must be non-empty and ordered")
    delta = _validate_scalar("delta_logrmse", delta_logrmse, 0.0)
    span_limit = _validate_scalar("max_normalized_span", max_normalized_span, 0.0, 1.0)
    points = [(float(row[parameter]), float(row[loss_key])) for row in profile]
    if any(not np.isfinite(value) or not np.isfinite(loss) or value < lo or value > hi for value, loss in points):
        raise ValueError("profile contains invalid parameter values or losses")
    optimum = min(loss for _, loss in points)
    accepted = [value for value, loss in points if loss <= optimum + delta]
    envelope_span = max(accepted) - min(accepted)
    normalized_span = envelope_span / (hi - lo)
    return {
        "optimum_loss": optimum,
        "delta_logrmse": delta,
        "acceptable_envelope": [min(accepted), max(accepted)],
        "acceptable_envelope_span": envelope_span,
        "acceptable_normalized_span": normalized_span,
        "max_normalized_span": span_limit,
        "identifiable": bool(normalized_span <= span_limit),
    }


def observation_scale_audit(reflectance: np.ndarray, tolerance: float = 1e-10) -> dict[str, object]:
    """Audit a scaled observation prediction without modifying or clipping it."""

    values = np.asarray(reflectance, dtype=np.float64)
    tolerance = _validate_scalar("tolerance", tolerance, 0.0)
    finite = np.isfinite(values)
    negative_count = int(np.count_nonzero(finite & (values < -tolerance)))
    above_one_count = int(np.count_nonzero(finite & (values > 1.0 + tolerance)))
    passed = bool(finite.all() and negative_count == 0 and above_one_count == 0)
    return {
        "pass": passed,
        "finite": bool(finite.all()),
        "negative_count": negative_count,
        "above_one_count": above_one_count,
        "clipped": False,
    }
