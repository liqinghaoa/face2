"""Deterministic multi-start inversion and identifiability diagnostics."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy.optimize import minimize
from scipy.stats import qmc

from .config import Stage1Config
from .metrics import spectral_angle_rad, spectral_metrics
from .parameterization import boundary_flags, logits_to_theta, theta_to_logits
from .skin_forward import SkinForwardModel


@dataclass(frozen=True)
class SpectrumFitResult:
    theta: np.ndarray
    predicted: np.ndarray
    metrics: dict[str, float]
    objective: float
    starts: list[dict[str, Any]]
    identifiability: dict[str, Any]


def _pseudo_huber(values: np.ndarray, delta: float) -> np.ndarray:
    scaled = values / delta
    return delta * delta * (np.sqrt(1.0 + scaled * scaled) - 1.0)


def _deterministic_start_fractions(n_starts: int) -> np.ndarray:
    sampler = qmc.Halton(d=2, scramble=False)
    points = sampler.random(n=n_starts + 1)[1:]
    return 0.05 + 0.90 * points


def _finite_difference_jacobian(
    model: SkinForwardModel,
    theta: np.ndarray,
    wavelengths: np.ndarray,
    epsilon: float,
) -> np.ndarray:
    columns: list[np.ndarray] = []
    for index, (lower, upper) in enumerate(model.config.bounds):
        step = max((upper - lower) * 1e-4, 1e-8)
        lo_theta = theta.copy()
        hi_theta = theta.copy()
        lo_theta[index] = max(lower + 1e-9, theta[index] - step)
        hi_theta[index] = min(upper - 1e-9, theta[index] + step)
        denominator = hi_theta[index] - lo_theta[index]
        lo_spectrum = np.asarray(model.forward_numpy(lo_theta, wavelengths))
        hi_spectrum = np.asarray(model.forward_numpy(hi_theta, wavelengths))
        derivative = (
            np.log(np.maximum(hi_spectrum, epsilon)) - np.log(np.maximum(lo_spectrum, epsilon))
        ) / denominator
        columns.append(derivative)
    return np.stack(columns, axis=1)


def fit_spectrum(
    observed_reflectance: np.ndarray,
    wavelength_nm: np.ndarray,
    model: SkinForwardModel,
    config: Stage1Config,
    seed_offset: int = 0,
    run_noise_audit: bool = True,
) -> SpectrumFitResult:
    """Fit one region-average reflectance spectrum."""

    observed = np.asarray(observed_reflectance, dtype=np.float64)
    wavelengths = np.asarray(wavelength_nm, dtype=np.float64)
    if observed.shape != wavelengths.shape:
        raise ValueError("Observed spectrum and wavelength grid must align")
    fit_cfg = config.raw["fit"]
    epsilon = float(fit_cfg["reflectance_epsilon"])
    valid = np.isfinite(observed) & (observed > epsilon)
    if int(valid.sum()) < int(fit_cfg["minimum_valid_bands"]):
        raise ValueError("Spectrum has too few valid positive reflectance bands")
    observed_valid = observed[valid]
    wavelength_valid = wavelengths[valid]
    bounds = config.bounds
    prior_center = np.asarray(config.prior_centers, dtype=np.float64)
    prior_scale = np.asarray(config.prior_scales, dtype=np.float64)
    prior_weight = float(fit_cfg["prior_weight"])
    huber_delta = float(fit_cfg["log_huber_delta"])
    sam_weight = float(fit_cfg["sam_weight"])

    def objective(logits: np.ndarray, target: np.ndarray) -> float:
        theta = logits_to_theta(logits, bounds)
        predicted = np.asarray(model.forward_numpy(theta, wavelength_valid))
        residual = np.log(np.maximum(predicted, epsilon)) - np.log(np.maximum(target, epsilon))
        robust_log = float(np.mean(_pseudo_huber(residual, huber_delta)))
        sam = spectral_angle_rad(predicted, target, epsilon)
        prior = float(np.mean(((theta - prior_center) / prior_scale) ** 2))
        return robust_log + sam_weight * sam * sam + prior_weight * prior

    starts: list[dict[str, Any]] = []
    best_result: Any | None = None
    best_theta: np.ndarray | None = None
    start_fractions = _deterministic_start_fractions(int(fit_cfg["n_starts"]))
    lower = np.asarray([bound[0] for bound in bounds])
    upper = np.asarray([bound[1] for bound in bounds])
    for start_index, fraction in enumerate(start_fractions):
        initial_theta = lower + fraction * (upper - lower)
        initial_logits = theta_to_logits(initial_theta, bounds)
        result = minimize(
            objective,
            initial_logits,
            args=(observed_valid,),
            method="BFGS",
            options={"maxiter": int(fit_cfg["max_iterations"]), "gtol": float(fit_cfg["gradient_tolerance"])},
        )
        theta = logits_to_theta(np.asarray(result.x), bounds)
        record = {
            "start_index": start_index,
            "initial_M_absorbance": float(initial_theta[0]),
            "initial_Hb_absorbance_proxy": float(initial_theta[1]),
            "final_M_absorbance": float(theta[0]),
            "final_Hb_absorbance_proxy": float(theta[1]),
            "objective": float(result.fun),
            "success": bool(result.success),
            "status": int(result.status),
            "iterations": int(getattr(result, "nit", -1)),
            "message": str(result.message),
        }
        starts.append(record)
        if best_result is None or float(result.fun) < float(best_result.fun):
            best_result = result
            best_theta = theta
    if best_theta is None or best_result is None:
        raise RuntimeError("No inverse optimization result was produced")

    predicted = np.asarray(model.forward_numpy(best_theta, wavelengths))
    metrics = spectral_metrics(predicted[valid], observed[valid], epsilon)
    final_thetas = np.asarray(
        [[row["final_M_absorbance"], row["final_Hb_absorbance_proxy"]] for row in starts],
        dtype=np.float64,
    )
    theta_range = np.ptp(final_thetas, axis=0)
    parameter_span = upper - lower
    jacobian = _finite_difference_jacobian(model, best_theta, wavelength_valid, epsilon)
    singular_values = np.linalg.svd(jacobian, full_matrices=False, compute_uv=False)
    condition_number = float(singular_values[0] / max(singular_values[-1], 1e-15))

    noise_thetas: list[np.ndarray] = []
    if run_noise_audit and int(fit_cfg["noise_repeats"]) > 0:
        rng = np.random.default_rng(int(fit_cfg["random_seed"]) + int(seed_offset))
        relative_sd = float(fit_cfg["noise_relative_sd"])
        for _ in range(int(fit_cfg["noise_repeats"])):
            noisy = np.maximum(observed * (1.0 + rng.normal(0.0, relative_sd, size=observed.shape)), epsilon)
            nested = fit_spectrum(noisy, wavelengths, model, config, run_noise_audit=False)
            noise_thetas.append(nested.theta)
    noise_array = np.asarray(noise_thetas, dtype=np.float64) if noise_thetas else np.empty((0, 2))
    noise_relative_sd = (
        np.std(noise_array, axis=0, ddof=1) / parameter_span if len(noise_array) > 1 else np.full(2, np.nan)
    )
    identifiability = {
        "multistart_theta_range": theta_range.tolist(),
        "multistart_theta_relative_range": (theta_range / parameter_span).tolist(),
        "boundary_flags": boundary_flags(
            best_theta,
            bounds,
            fraction=float(fit_cfg["boundary_fraction"]),
        ).tolist(),
        "jacobian_singular_values": singular_values.tolist(),
        "jacobian_condition_number": condition_number,
        "noise_theta_mean": np.mean(noise_array, axis=0).tolist() if len(noise_array) else [None, None],
        "noise_theta_sd": np.std(noise_array, axis=0, ddof=1).tolist() if len(noise_array) > 1 else [None, None],
        "noise_theta_relative_sd": noise_relative_sd.tolist(),
        "noise_repeats": int(len(noise_array)),
    }
    return SpectrumFitResult(
        theta=best_theta,
        predicted=predicted,
        metrics=metrics,
        objective=float(best_result.fun),
        starts=starts,
        identifiability=identifiability,
    )

