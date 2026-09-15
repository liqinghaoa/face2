"""Small, auditable inverse routines for the S1-4R proxy candidates.

The routines intentionally fit one spectrum at a time and never touch the
Hyper-Skin files.  They operate on the centered log-reflectance contract used
by D1/D2/D3 models and expose the linear-algebra diagnostics needed for the
synthetic audit.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np

from .s1_revised_forward import (
    RevisedOpticalAssets, RevisedSkinForwardModel, RevisedRegistry, centered_log_ratio, revised_basis,
)


@dataclass(frozen=True)
class ProxyFitResult:
    model_id: str
    parameter_names: tuple[str, ...]
    theta: np.ndarray
    fitted_centered_log_ratio: np.ndarray
    residual: np.ndarray
    weighted_rmse: float
    rank: int
    singular_values: np.ndarray
    covariance: np.ndarray
    amplitude_log: float


@dataclass(frozen=True)
class PcaBasis:
    mean: np.ndarray
    components: np.ndarray
    singular_values: np.ndarray
    training_subject_ids: tuple[str, ...]
    held_out_subject_id: str | None


def _weighted_lstsq(design: np.ndarray, target: np.ndarray, weights: np.ndarray) -> tuple[np.ndarray, np.ndarray, int, np.ndarray, np.ndarray]:
    x = np.asarray(design, dtype=np.float64)
    y = np.asarray(target, dtype=np.float64)
    w = np.asarray(weights, dtype=np.float64)
    if x.ndim != 2 or y.ndim != 1 or x.shape[0] != y.size or w.shape != (y.size,):
        raise ValueError("design, target and weights are misaligned")
    if np.any(~np.isfinite(x)) or np.any(~np.isfinite(y)) or np.any(w <= 0) or np.any(~np.isfinite(w)):
        raise ValueError("weighted least squares inputs must be finite")
    root_w = np.sqrt(w)
    wx = x * root_w[:, None]
    wy = y * root_w
    theta, _, rank, singular = np.linalg.lstsq(wx, wy, rcond=None)
    fitted = x @ theta
    residual = y - fitted
    dof = max(1, y.size - int(rank))
    sigma2 = float(np.sum(w * residual * residual) / dof)
    pinv = np.linalg.pinv(wx)
    covariance = sigma2 * (pinv @ pinv.T)
    return theta, residual, int(rank), np.asarray(singular), covariance


def fit_proxy_wls(
    model_id: str,
    observed_reflectance: np.ndarray,
    reference_reflectance: np.ndarray,
    *,
    registry: RevisedRegistry | None = None,
    weights: np.ndarray | None = None,
    wavelength_nm: np.ndarray | None = None,
    basis_wavelength_nm: np.ndarray | None = None,
    assets: RevisedOpticalAssets | None = None,
    basis_params: Mapping[str, float] | None = None,
    epsilon: float = 1e-8,
) -> ProxyFitResult:
    """Fit D1/D2/D3 proxy coefficients by weighted linear least squares."""
    model = RevisedSkinForwardModel(model_id, registry=registry, assets=assets)
    if model.spec.family not in {"proxy_linear", "proxy_linear_oxygenation", "centered_log_reference_baseline"}:
        raise ValueError(f"{model_id} is not a centered-log proxy model")
    observed = np.asarray(observed_reflectance, dtype=np.float64)
    reference = np.asarray(reference_reflectance, dtype=np.float64)
    wavelength = model.registry.wavelength_nm if wavelength_nm is None else np.asarray(wavelength_nm, dtype=np.float64)
    if wavelength.ndim != 1 or observed.shape != wavelength.shape or reference.shape != wavelength.shape:
        raise ValueError("S1-4R proxy fitting expects one wavelength-aligned spectrum")
    if np.any(np.diff(wavelength) <= 0) or not set(wavelength).issubset(set(model.registry.wavelength_nm)):
        raise ValueError("wavelength_nm must be an ordered subset of the official centers")
    basis_wavelength = wavelength if basis_wavelength_nm is None else np.asarray(basis_wavelength_nm, dtype=np.float64)
    if basis_wavelength.shape != wavelength.shape or np.any(np.diff(basis_wavelength) <= 0):
        raise ValueError("basis_wavelength_nm must be increasing and observation aligned")
    w = np.ones(wavelength.size, dtype=np.float64) if weights is None else np.asarray(weights, dtype=np.float64)
    if w.shape != wavelength.shape or np.any(w <= 0) or np.any(~np.isfinite(w)):
        raise ValueError("weights must be positive, finite and wavelength aligned")
    _, amplitude, target = centered_log_ratio(observed, reference, w, epsilon)
    options = dict(basis_params or {})
    basis = revised_basis(
        basis_wavelength, model.assets, weights=w,
        melanin_power=float(options.get("melanin_power", model.registry.global_value("melanin_power"))),
        sO2_ref=float(options.get("sO2_ref", model.registry.global_value("sO2_ref"))),
        vessel_diameter_um=float(options.get("vessel_diameter_um", model.registry.global_value("vessel_diameter_um"))),
    )
    columns = []
    for name in model.parameter_names:
        if name == "delta_M_OD":
            columns.append(-basis["phi_M"])
        elif name == "delta_Hb_OD":
            columns.append(-basis["phi_H"])
        elif name == "q_tilt":
            columns.append(basis["phi_G"])
        elif name == "delta_sO2":
            columns.append(basis["phi_O"])
        else:
            raise ValueError(f"Unsupported proxy parameter {name}")
    design = np.column_stack(columns) if columns else np.zeros((wavelength.size, 0), dtype=np.float64)
    if columns:
        theta, residual, rank, singular, covariance = _weighted_lstsq(design, target, w)
    else:
        theta = np.empty(0, dtype=np.float64)
        residual = target.copy()
        rank, singular = 0, np.empty(0, dtype=np.float64)
        covariance = np.empty((0, 0), dtype=np.float64)
    fitted = target - residual
    rmse = float(np.sqrt(np.sum(w * residual * residual) / np.sum(w)))
    return ProxyFitResult(model_id, model.parameter_names, theta, fitted, residual, rmse,
                          rank, singular, covariance, float(amplitude))


def fit_semi_mechanistic_least_squares(
    model_id: str,
    observed_reflectance: np.ndarray,
    *,
    registry: RevisedRegistry | None = None,
    global_params: Mapping[str, Any] | None = None,
    wavelength_nm: np.ndarray | None = None,
    max_nfev: int = 1000,
) -> dict[str, Any]:
    """Bounded deterministic fit for K2/K3 synthetic diagnostics."""
    if model_id not in {"K2-MH-KM", "K3-MHS-KM"}:
        raise ValueError("Only K2-MH-KM and K3-MHS-KM are supported")
    from scipy.optimize import least_squares

    model = RevisedSkinForwardModel(model_id, registry=registry)
    observed = np.asarray(observed_reflectance, dtype=np.float64)
    wavelength = model.registry.wavelength_nm if wavelength_nm is None else np.asarray(wavelength_nm, dtype=np.float64)
    if wavelength.ndim != 1 or observed.shape != wavelength.shape or np.any(observed <= 0) or np.any(~np.isfinite(observed)):
        raise ValueError("observed_reflectance must be a positive wavelength-aligned vector")
    if np.any(np.diff(wavelength) <= 0) or not set(wavelength).issubset(set(model.registry.wavelength_nm)):
        raise ValueError("wavelength_nm must be an ordered subset of the official centers")
    start = model.registry.reference_values_for(model_id)
    lower = np.asarray([b[0] for b in model.bounds], dtype=np.float64)
    upper = np.asarray([b[1] for b in model.bounds], dtype=np.float64)
    gp = dict(global_params or {})
    def fun(theta: np.ndarray) -> np.ndarray:
        pred = model.forward_numpy(theta, wavelength, gp)
        return np.log(pred) - np.log(observed)
    fit = least_squares(fun, np.clip(start, lower, upper), bounds=(lower, upper), max_nfev=max_nfev, method="trf")
    return {"model_id": model_id, "theta": fit.x, "cost": float(2.0 * fit.cost),
            "success": bool(fit.success), "nfev": int(fit.nfev), "message": str(fit.message),
            "residual": fun(fit.x)}


def fit_group_isolated_pca(
    spectra: np.ndarray,
    subject_ids: np.ndarray,
    *,
    held_out_subject_id: str | None = None,
    n_components: int = 2,
) -> PcaBasis:
    """Fit a PCA basis while explicitly excluding one held-out subject."""
    x = np.asarray(spectra, dtype=np.float64)
    ids = np.asarray(subject_ids).astype(str)
    if x.ndim != 2 or ids.shape != (x.shape[0],) or n_components <= 0:
        raise ValueError("spectra/subject_ids/n_components are invalid")
    keep = np.ones(ids.size, dtype=bool) if held_out_subject_id is None else ids != str(held_out_subject_id)
    if not np.any(keep) or np.unique(ids[keep]).size < 2:
        raise ValueError("PCA requires at least two training subjects")
    train = x[keep]
    mean = train.mean(axis=0)
    _, singular, vt = np.linalg.svd(train - mean, full_matrices=False)
    if n_components > vt.shape[0]:
        raise ValueError("n_components exceeds available rank")
    return PcaBasis(mean, vt[:n_components].T, singular[:n_components],
                    tuple(sorted(set(ids[keep]))), held_out_subject_id)


__all__ = ["PcaBasis", "ProxyFitResult", "fit_group_isolated_pca", "fit_proxy_wls", "fit_semi_mechanistic_least_squares"]
