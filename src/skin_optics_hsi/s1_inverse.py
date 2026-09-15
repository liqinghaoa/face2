"""Variable-dimension deterministic inversion used by S1-5."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np
from scipy.optimize import least_squares
from scipy.stats import qmc

from .metrics import spectral_angle_rad, spectral_metrics
from .parameterization import boundary_flags, logits_to_theta
from .skin_forward import RegisteredSkinForwardModel


@dataclass(frozen=True)
class RegisteredSpectrumFit:
    model_id: str
    parameter_names: tuple[str, ...]
    theta: np.ndarray
    predicted: np.ndarray
    objective: float
    metrics: dict[str, float]
    starts: list[dict[str, Any]]
    identifiability: dict[str, Any]
    valid_band_mask: np.ndarray


def _pseudo_huber(values: np.ndarray, delta: float) -> np.ndarray:
    scaled = values / delta
    return delta * delta * (np.sqrt(1.0 + scaled * scaled) - 1.0)


def deterministic_start_fractions(n_starts: int, dimension: int) -> np.ndarray:
    if n_starts < 1 or dimension < 1:
        raise ValueError("n_starts and dimension must be positive")
    sampler = qmc.Halton(d=dimension, scramble=False)
    points = sampler.random(n=n_starts + 1)[1:]
    return 0.02 + 0.96 * points


def _fraction_to_logits(fraction: np.ndarray) -> np.ndarray:
    clipped = np.clip(np.asarray(fraction, dtype=np.float64), 1e-9, 1.0 - 1e-9)
    return np.log(clipped) - np.log1p(-clipped)


def objective_residual_vector(
    theta: np.ndarray,
    observed_reflectance: np.ndarray,
    wavelength_nm: np.ndarray,
    model: RegisteredSkinForwardModel,
    *,
    band_weights: np.ndarray,
    epsilon: float,
    pseudo_huber_delta: float,
    sam_weight: float,
    global_params: Mapping[str, Any] | None,
    observation_context: Mapping[str, Any] | None,
) -> np.ndarray:
    predicted = np.asarray(
        model.forward_flat_numpy(
            theta,
            wavelength_nm,
            global_params=global_params,
            observation_context=observation_context,
        ),
        dtype=np.float64,
    )
    observed = np.asarray(observed_reflectance, dtype=np.float64)
    weights = np.asarray(band_weights, dtype=np.float64)
    log_difference = np.log(np.maximum(predicted, epsilon)) - np.log(np.maximum(observed, epsilon))
    robust = _pseudo_huber(log_difference, pseudo_huber_delta)
    signed_root = np.sign(log_difference) * np.sqrt(2.0 * robust * weights / np.sum(weights))
    sam = spectral_angle_rad(predicted, observed, epsilon)
    return np.concatenate([signed_root, np.asarray([np.sqrt(2.0 * sam_weight) * sam])])


def objective_value(
    theta: np.ndarray,
    observed_reflectance: np.ndarray,
    wavelength_nm: np.ndarray,
    model: RegisteredSkinForwardModel,
    **kwargs: Any,
) -> float:
    residual = objective_residual_vector(theta, observed_reflectance, wavelength_nm, model, **kwargs)
    return float(0.5 * np.dot(residual, residual))


def _log_jacobian(
    model: RegisteredSkinForwardModel,
    theta: np.ndarray,
    wavelength_nm: np.ndarray,
    epsilon: float,
    relative_step: float,
    global_params: Mapping[str, Any] | None,
    observation_context: Mapping[str, Any] | None,
) -> np.ndarray:
    columns: list[np.ndarray] = []
    for index, (lower, upper) in enumerate(model.bounds):
        step = max((upper - lower) * relative_step, 1e-8)
        lo = theta.copy()
        hi = theta.copy()
        lo[index] = max(lower + 1e-10, theta[index] - step)
        hi[index] = min(upper - 1e-10, theta[index] + step)
        denominator = hi[index] - lo[index]
        pred_lo = np.asarray(
            model.forward_flat_numpy(lo, wavelength_nm, global_params=global_params, observation_context=observation_context)
        )
        pred_hi = np.asarray(
            model.forward_flat_numpy(hi, wavelength_nm, global_params=global_params, observation_context=observation_context)
        )
        columns.append(
            (np.log(np.maximum(pred_hi, epsilon)) - np.log(np.maximum(pred_lo, epsilon))) / denominator
        )
    return np.stack(columns, axis=1)


def _solution_cluster_count(thetas: np.ndarray, bounds: tuple[tuple[float, float], ...], threshold: float) -> int:
    lower = np.asarray([item[0] for item in bounds], dtype=np.float64)
    upper = np.asarray([item[1] for item in bounds], dtype=np.float64)
    normalized = (thetas - lower) / (upper - lower)
    centers: list[np.ndarray] = []
    for row in normalized:
        matching = [index for index, center in enumerate(centers) if np.linalg.norm(row - center) <= threshold]
        if matching:
            index = matching[0]
            centers[index] = 0.5 * (centers[index] + row)
        else:
            centers.append(row.copy())
    return len(centers)


def fit_registered_spectrum(
    observed_reflectance: np.ndarray,
    wavelength_nm: np.ndarray,
    model: RegisteredSkinForwardModel,
    fit_config: Mapping[str, Any],
    *,
    n_starts: int,
    band_mask: np.ndarray | None = None,
    band_weights: np.ndarray | None = None,
    global_params: Mapping[str, Any] | None = None,
    observation_context: Mapping[str, Any] | None = None,
) -> RegisteredSpectrumFit:
    """Fit one spectrum using deterministic Halton starts in bounded-logit space."""

    observed_full = np.asarray(observed_reflectance, dtype=np.float64)
    wavelength_full = np.asarray(wavelength_nm, dtype=np.float64)
    if observed_full.shape != wavelength_full.shape:
        raise ValueError("Observed spectrum and wavelength grid must align")
    epsilon = float(fit_config["reflectance_epsilon"])
    valid = np.isfinite(observed_full) & (observed_full > epsilon)
    if band_mask is not None:
        selected = np.asarray(band_mask, dtype=bool)
        if selected.shape != valid.shape:
            raise ValueError("band_mask must align with wavelength_nm")
        valid &= selected
    if int(valid.sum()) < int(fit_config["minimum_valid_bands"]):
        raise ValueError("Spectrum has too few valid positive selected bands")
    observed = observed_full[valid]
    wavelength = wavelength_full[valid]
    if band_weights is None:
        weights = np.ones_like(observed)
    else:
        weights_full = np.asarray(band_weights, dtype=np.float64)
        if weights_full.shape != valid.shape or np.any(~np.isfinite(weights_full)) or np.any(weights_full <= 0):
            raise ValueError("band_weights must be finite, positive, and wavelength-aligned")
        weights = weights_full[valid]

    bounds = model.bounds
    dimension = len(bounds)
    if dimension < 1:
        raise ValueError("B0 has no free parameters and must not use fit_registered_spectrum")
    starts: list[dict[str, Any]] = []
    best_theta: np.ndarray | None = None
    best_objective = float("inf")
    best_prediction: np.ndarray | None = None
    lower = np.asarray([item[0] for item in bounds], dtype=np.float64)
    upper = np.asarray([item[1] for item in bounds], dtype=np.float64)

    objective_kwargs = {
        "band_weights": weights,
        "epsilon": epsilon,
        "pseudo_huber_delta": float(fit_config["log_pseudo_huber_delta"]),
        "sam_weight": float(fit_config["sam_weight"]),
        "global_params": global_params,
        "observation_context": observation_context,
    }

    for start_index, fraction in enumerate(deterministic_start_fractions(n_starts, dimension)):
        initial_theta = lower + fraction * (upper - lower)
        result = least_squares(
            lambda logits: objective_residual_vector(
                logits_to_theta(logits, bounds), observed, wavelength, model, **objective_kwargs
            ),
            _fraction_to_logits(fraction),
            max_nfev=int(fit_config["max_function_evaluations"]),
            xtol=float(fit_config["convergence_xtol"]),
            ftol=float(fit_config["convergence_ftol"]),
            gtol=float(fit_config["convergence_gtol"]),
            method="trf",
        )
        theta = logits_to_theta(np.asarray(result.x, dtype=np.float64), bounds)
        objective = objective_value(theta, observed, wavelength, model, **objective_kwargs)
        starts.append(
            {
                "start_index": start_index,
                "initial_theta": initial_theta.tolist(),
                "final_theta": theta.tolist(),
                "objective": objective,
                "success": bool(result.success),
                "status": int(result.status),
                "function_evaluations": int(result.nfev),
                "optimality": float(result.optimality),
                "message": str(result.message),
            }
        )
        if np.isfinite(objective) and objective < best_objective:
            best_objective = objective
            best_theta = theta
            best_prediction = np.asarray(
                model.forward_flat_numpy(
                    theta,
                    wavelength_full,
                    global_params=global_params,
                    observation_context=observation_context,
                )
            )
    if best_theta is None or best_prediction is None:
        raise RuntimeError(f"No finite optimization result for {model.model_id}")

    final_thetas = np.asarray([item["final_theta"] for item in starts], dtype=np.float64)
    theta_range = np.ptp(final_thetas, axis=0)
    span = upper - lower
    jacobian = _log_jacobian(
        model,
        best_theta,
        wavelength,
        epsilon,
        float(fit_config["jacobian_relative_step"]),
        global_params,
        observation_context,
    )
    singular = np.linalg.svd(jacobian, full_matrices=False, compute_uv=False)
    tolerance = max(jacobian.shape) * np.finfo(np.float64).eps * singular[0]
    rank = int(np.sum(singular > tolerance))
    condition = float(singular[0] / max(singular[-1], 1e-15))
    jtj = jacobian.T @ jacobian
    covariance = np.linalg.pinv(jtj, rcond=1e-12)
    objectives = np.asarray([float(item["objective"]) for item in starts], dtype=np.float64)
    near_tolerance = max(
        float(fit_config.get("near_optimal_objective_absolute_tolerance", 1e-6)),
        float(fit_config.get("near_optimal_objective_relative_tolerance", 0.01)) * best_objective,
    )
    near_optimal = final_thetas[objectives <= best_objective + near_tolerance]
    identifiability = {
        "multistart_theta_range": theta_range.tolist(),
        "multistart_theta_relative_range": (theta_range / span).tolist(),
        "near_optimal_start_count": int(len(near_optimal)),
        "near_optimal_objective_tolerance": near_tolerance,
        "solution_cluster_count": _solution_cluster_count(
            near_optimal,
            bounds,
            float(fit_config["solution_cluster_normalized_distance"]),
        ),
        "boundary_flags": boundary_flags(
            best_theta,
            bounds,
            fraction=float(fit_config["boundary_fraction"]),
        ).tolist(),
        "jacobian_rank": rank,
        "jacobian_singular_values": singular.tolist(),
        "jacobian_condition_number": condition,
        "local_covariance_diagonal_unscaled": np.diag(covariance).tolist(),
    }
    metrics = spectral_metrics(best_prediction[valid], observed_full[valid], epsilon)
    return RegisteredSpectrumFit(
        model_id=model.model_id,
        parameter_names=model.parameter_names,
        theta=best_theta,
        predicted=best_prediction,
        objective=best_objective,
        metrics=metrics,
        starts=starts,
        identifiability=identifiability,
        valid_band_mask=valid,
    )


def profile_registered_spectrum(
    observed_reflectance: np.ndarray,
    wavelength_nm: np.ndarray,
    model: RegisteredSkinForwardModel,
    best_theta: np.ndarray,
    fit_config: Mapping[str, Any],
    grid_fractions: list[float] | tuple[float, ...],
    *,
    band_mask: np.ndarray | None = None,
    global_params: Mapping[str, Any] | None = None,
    observation_context: Mapping[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Profile every free parameter while optimizing all remaining parameters."""

    observed_full = np.asarray(observed_reflectance, dtype=np.float64)
    wavelength_full = np.asarray(wavelength_nm, dtype=np.float64)
    epsilon = float(fit_config["reflectance_epsilon"])
    valid = np.isfinite(observed_full) & (observed_full > epsilon)
    if band_mask is not None:
        valid &= np.asarray(band_mask, dtype=bool)
    if int(valid.sum()) < int(fit_config["minimum_valid_bands"]):
        raise ValueError("Spectrum has too few valid positive selected bands")
    observed = observed_full[valid]
    wavelength = wavelength_full[valid]
    weights = np.ones_like(observed)
    bounds = model.bounds
    lower = np.asarray([item[0] for item in bounds], dtype=np.float64)
    upper = np.asarray([item[1] for item in bounds], dtype=np.float64)
    optimum = np.asarray(best_theta, dtype=np.float64)
    if optimum.shape != lower.shape:
        raise ValueError("best_theta does not match the model dimension")
    kwargs = {
        "band_weights": weights,
        "epsilon": epsilon,
        "pseudo_huber_delta": float(fit_config["log_pseudo_huber_delta"]),
        "sam_weight": float(fit_config["sam_weight"]),
        "global_params": global_params,
        "observation_context": observation_context,
    }
    optimum_objective = objective_value(optimum, observed, wavelength, model, **kwargs)
    rows: list[dict[str, Any]] = []
    for profiled_index, name in enumerate(model.parameter_names):
        remaining = [index for index in range(len(bounds)) if index != profiled_index]
        for fraction in grid_fractions:
            if not 0.0 < float(fraction) < 1.0:
                raise ValueError("Profile grid fractions must lie strictly inside (0, 1)")
            fixed_value = lower[profiled_index] + float(fraction) * (upper[profiled_index] - lower[profiled_index])
            theta = optimum.copy()
            theta[profiled_index] = fixed_value
            success = True
            status = 0
            nfev = 0
            if remaining:
                remaining_bounds = tuple(bounds[index] for index in remaining)
                remaining_lower = lower[remaining]
                remaining_upper = upper[remaining]
                initial_fraction = (optimum[remaining] - remaining_lower) / (remaining_upper - remaining_lower)

                def residual(logits: np.ndarray) -> np.ndarray:
                    theta[remaining] = logits_to_theta(logits, remaining_bounds)
                    return objective_residual_vector(theta, observed, wavelength, model, **kwargs)

                result = least_squares(
                    residual,
                    _fraction_to_logits(initial_fraction),
                    max_nfev=int(fit_config["max_function_evaluations"]),
                    xtol=float(fit_config["convergence_xtol"]),
                    ftol=float(fit_config["convergence_ftol"]),
                    gtol=float(fit_config["convergence_gtol"]),
                    method="trf",
                )
                theta[remaining] = logits_to_theta(np.asarray(result.x), remaining_bounds)
                success = bool(result.success)
                status = int(result.status)
                nfev = int(result.nfev)
            value = objective_value(theta, observed, wavelength, model, **kwargs)
            rows.append(
                {
                    "model_id": model.model_id,
                    "profiled_parameter": name,
                    "grid_fraction": float(fraction),
                    "fixed_value": float(fixed_value),
                    "profile_objective": value,
                    "optimum_objective": optimum_objective,
                    "delta_objective": value - optimum_objective,
                    "conditional_theta": theta.tolist(),
                    "success": success,
                    "status": status,
                    "function_evaluations": nfev,
                }
            )
    return rows
