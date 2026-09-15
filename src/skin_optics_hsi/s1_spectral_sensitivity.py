"""Pre-specified wavelength and effective-SRF sensitivities for S1-4/S1-5."""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np

from .model_registry import Stage1ModelRegistry
from .skin_forward import RegisteredSkinForwardModel


class GaussianSRFModelAdapter:
    """Fast NumPy adapter that integrates a registered model over Gaussian bands."""

    def __init__(
        self,
        base_model: RegisteredSkinForwardModel,
        centers_nm: np.ndarray,
        fwhm_nm: float,
        integration_step_nm: float = 0.5,
    ) -> None:
        self.base_model = base_model
        self.registry = base_model.registry
        self.model_id = base_model.model_id
        self.model_spec = base_model.model_spec
        self.parameter_names = base_model.parameter_names
        self.bounds = base_model.bounds
        self.bio_parameter_names = base_model.bio_parameter_names
        self.nuisance_parameter_names = base_model.nuisance_parameter_names
        self.centers_nm = np.asarray(centers_nm, dtype=np.float64)
        allowed = np.asarray(self.registry.raw["sensitivity"]["assumed_gaussian_fwhm_nm"], dtype=np.float64)
        if float(fwhm_nm) <= 0 or not bool(np.any(np.isclose(float(fwhm_nm), allowed, rtol=0.0, atol=1e-12))):
            raise ValueError("GaussianSRFModelAdapter requires a registered positive FWHM")
        if integration_step_nm <= 0:
            raise ValueError("integration_step_nm must be positive")
        self.fwhm_nm = float(fwhm_nm)
        lower = float(base_model.hemoglobin.wavelength_nm.min())
        upper = float(base_model.hemoglobin.wavelength_nm.max())
        self.integration_grid_nm = np.arange(lower, upper + integration_step_nm / 2.0, integration_step_nm)
        sigma = self.fwhm_nm / (2.0 * np.sqrt(2.0 * np.log(2.0)))
        weights = np.exp(
            -0.5
            * ((self.integration_grid_nm[None, :] - self.centers_nm[:, None]) / sigma) ** 2
        )
        totals = weights.sum(axis=1, keepdims=True)
        if np.any(totals <= 0):
            raise ValueError("One or more Gaussian bands have no support in the audited asset range")
        self.integration_weights = weights / totals

    def forward_flat_numpy(
        self,
        theta: np.ndarray,
        wavelength_nm: np.ndarray,
        **kwargs: Any,
    ) -> np.ndarray:
        requested = np.asarray(wavelength_nm, dtype=np.float64)
        if requested.shape != self.centers_nm.shape or not np.allclose(requested, self.centers_nm, rtol=0, atol=1e-12):
            raise ValueError("Adapter wavelength centers must match its construction centers")
        fine = np.asarray(self.base_model.forward_flat_numpy(theta, self.integration_grid_nm, **kwargs))
        return fine @ self.integration_weights.T


def band_mask(
    wavelength_nm: np.ndarray,
    sensitivity_name: str,
    registry: Stage1ModelRegistry,
) -> np.ndarray:
    """Return the pre-registered band mask without learning from Validation/Test."""

    wavelength = np.asarray(wavelength_nm, dtype=np.float64)
    if wavelength.ndim != 1 or np.any(~np.isfinite(wavelength)):
        raise ValueError("wavelength_nm must be a finite one-dimensional array")
    try:
        specification = registry.raw["sensitivity"]["band_sets"][sensitivity_name]
    except KeyError as exc:
        raise KeyError(f"Unknown pre-specified band sensitivity: {sensitivity_name}") from exc
    if "include_nm" in specification:
        selected = np.isin(wavelength, np.asarray(specification["include_nm"], dtype=np.float64))
    else:
        selected = ~np.isin(wavelength, np.asarray(specification["exclude_nm"], dtype=np.float64))
    if not bool(selected.any()):
        raise ValueError(f"Band sensitivity {sensitivity_name} selects no wavelengths")
    return selected


def shifted_centers(
    center_wavelength_nm: np.ndarray,
    shift_nm: float,
    registry: Stage1ModelRegistry,
) -> np.ndarray:
    """Apply only a center shift declared in the S1-4 registry."""

    allowed = np.asarray(registry.raw["sensitivity"]["center_shift_nm"], dtype=np.float64)
    if not bool(np.any(np.isclose(float(shift_nm), allowed, rtol=0.0, atol=1e-12))):
        raise ValueError(f"Unregistered wavelength-center shift: {shift_nm}")
    shifted = np.asarray(center_wavelength_nm, dtype=np.float64) + float(shift_nm)
    if shifted.min() < 400.0 or shifted.max() > 720.0:
        raise ValueError("Shifted centers exceed the audited spectral asset coverage")
    return shifted


def gaussian_srf_reflectance(
    model: RegisteredSkinForwardModel,
    theta_bio: np.ndarray | None,
    center_wavelength_nm: np.ndarray,
    fwhm_nm: float,
    theta_nuisance: np.ndarray | None = None,
    global_params: Mapping[str, Any] | None = None,
    *,
    observation_context: Mapping[str, Any] | None = None,
    integration_step_nm: float = 0.25,
) -> np.ndarray:
    """Evaluate an assumed Gaussian effective SRF, truncated to asset coverage.

    A zero FWHM is the nominal point-sampled model. Nonzero choices are only
    sensitivity assumptions because the release effective SRF is missing.
    """

    centers = np.asarray(center_wavelength_nm, dtype=np.float64)
    if centers.ndim != 1 or np.any(~np.isfinite(centers)) or np.any(np.diff(centers) <= 0):
        raise ValueError("center_wavelength_nm must be finite and strictly increasing")
    allowed = np.asarray(model.registry.raw["sensitivity"]["assumed_gaussian_fwhm_nm"], dtype=np.float64)
    if not bool(np.any(np.isclose(float(fwhm_nm), allowed, rtol=0.0, atol=1e-12))):
        raise ValueError(f"Unregistered Gaussian FWHM sensitivity: {fwhm_nm}")
    if float(fwhm_nm) == 0.0:
        return np.asarray(
            model.forward_numpy(
                theta_bio,
                centers,
                theta_nuisance,
                global_params,
                observation_context=observation_context,
            )
        )
    if integration_step_nm <= 0:
        raise ValueError("integration_step_nm must be positive")

    sigma = float(fwhm_nm) / (2.0 * np.sqrt(2.0 * np.log(2.0)))
    lower_asset = float(model.hemoglobin.wavelength_nm.min())
    upper_asset = float(model.hemoglobin.wavelength_nm.max())
    values: list[np.ndarray] = []
    for center in centers:
        lower = max(lower_asset, float(center) - 4.0 * sigma)
        upper = min(upper_asset, float(center) + 4.0 * sigma)
        count = max(3, int(np.ceil((upper - lower) / integration_step_nm)) + 1)
        grid = np.linspace(lower, upper, count, dtype=np.float64)
        weights = np.exp(-0.5 * ((grid - float(center)) / sigma) ** 2)
        spectrum = np.asarray(
            model.forward_numpy(
                theta_bio,
                grid,
                theta_nuisance,
                global_params,
                observation_context=observation_context,
            )
        )
        values.append(np.sum(spectrum * weights, axis=-1) / np.sum(weights))
    return np.stack(values, axis=-1)
