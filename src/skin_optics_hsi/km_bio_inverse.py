"""Deterministic bounded inversion and profile diagnostics for KM-BIO-v1."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import numpy as np
from scipy.optimize import least_squares
from scipy.stats import qmc


ForwardFunction = Callable[[np.ndarray], np.ndarray]


@dataclass(frozen=True)
class FitSettings:
    epsilon: float = 1e-6
    sobol_starts: int = 32
    seed: int = 20260909
    ftol: float = 1e-10
    xtol: float = 1e-10
    gtol: float = 1e-8
    max_nfev: int = 2000


def spectral_metrics(predicted: np.ndarray, observed: np.ndarray, epsilon: float = 1e-6) -> dict[str, float]:
    pred = np.asarray(predicted, dtype=np.float64)
    obs = np.asarray(observed, dtype=np.float64)
    if pred.shape != obs.shape or pred.ndim != 1 or np.any(~np.isfinite(pred)) or np.any(~np.isfinite(obs)):
        raise ValueError("Predicted and observed spectra must be aligned finite vectors")
    residual = pred - obs
    log_residual = np.log(pred + epsilon) - np.log(obs + epsilon)
    denominator = float(np.linalg.norm(pred) * np.linalg.norm(obs))
    cosine = float(np.dot(pred, obs) / denominator) if denominator > 0 else np.nan
    return {
        "logrmse": float(np.sqrt(np.mean(log_residual**2))),
        "rmse": float(np.sqrt(np.mean(residual**2))),
        "sam_deg": float(np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0)))),
    }


def sobol_plus_center(dimension: int, count: int, seed: int) -> np.ndarray:
    if dimension < 1 or count < 1 or count & (count - 1):
        raise ValueError("Sobol count must be a positive power of two")
    points = qmc.Sobol(dimension, scramble=True, seed=seed).random_base2(int(np.log2(count)))
    return np.vstack([points, np.full((1, dimension), 0.5, dtype=np.float64)])


def fit_bounded_spectrum(
    observed: np.ndarray,
    forward: ForwardFunction,
    lower: np.ndarray,
    upper: np.ndarray,
    settings: FitSettings,
    *,
    initial_u: np.ndarray | None = None,
) -> dict[str, Any]:
    observed = np.asarray(observed, dtype=np.float64)
    lower = np.asarray(lower, dtype=np.float64)
    upper = np.asarray(upper, dtype=np.float64)
    if observed.ndim != 1 or np.any(~np.isfinite(observed)) or np.any(observed <= 0):
        raise ValueError("Observed reflectance must be a finite positive vector")
    if lower.shape != upper.shape or np.any(upper <= lower):
        raise ValueError("Invalid parameter bounds")
    starts = sobol_plus_center(len(lower), settings.sobol_starts, settings.seed) if initial_u is None else np.asarray(initial_u, dtype=np.float64)
    if starts.ndim != 2 or starts.shape[1] != len(lower) or np.any(starts < 0) or np.any(starts > 1):
        raise ValueError("Initial points must be bounded scale coordinates")

    def residual(u: np.ndarray) -> np.ndarray:
        theta = lower + u * (upper - lower)
        prediction = np.asarray(forward(theta), dtype=np.float64)
        return np.log(prediction + settings.epsilon) - np.log(observed + settings.epsilon)

    records: list[dict[str, Any]] = []
    candidates: list[tuple[float, Any, np.ndarray, np.ndarray]] = []
    for index, start in enumerate(starts):
        result = least_squares(
            residual,
            start,
            bounds=(np.zeros_like(lower), np.ones_like(upper)),
            method="trf",
            loss="linear",
            ftol=settings.ftol,
            xtol=settings.xtol,
            gtol=settings.gtol,
            max_nfev=settings.max_nfev,
        )
        theta = lower + np.asarray(result.x) * (upper - lower)
        prediction = np.asarray(forward(theta), dtype=np.float64)
        metrics = spectral_metrics(prediction, observed, settings.epsilon)
        valid = bool(result.success and np.isfinite(prediction).all() and np.isfinite(metrics["logrmse"]))
        records.append({
            "start_index": index,
            "initial_u": start.tolist(),
            "initial_theta": (lower + start * (upper - lower)).tolist(),
            "final_u": np.asarray(result.x).tolist(),
            "final_theta": theta.tolist(),
            "logrmse": metrics["logrmse"],
            "cost": float(result.cost),
            "optimality": float(result.optimality),
            "success": bool(result.success),
            "valid": valid,
            "status": int(result.status),
            "nfev": int(result.nfev),
            "message": str(result.message),
        })
        if valid:
            candidates.append((metrics["logrmse"], result, theta, prediction))
    if not candidates:
        return {"success": False, "starts": records, "failure": "no_valid_converged_start"}
    _, best, theta, prediction = min(candidates, key=lambda item: item[0])
    return {
        "success": True,
        "theta": theta,
        "u": np.asarray(best.x, dtype=np.float64),
        "prediction": prediction,
        "metrics": spectral_metrics(prediction, observed, settings.epsilon),
        "residual": prediction - observed,
        "log_residual": np.log(prediction + settings.epsilon) - np.log(observed + settings.epsilon),
        "starts": records,
        "selected_start_index": int(np.argmin([row["logrmse"] if row["valid"] else np.inf for row in records])),
    }


def log_jacobian_u(forward: ForwardFunction, u: np.ndarray, lower: np.ndarray, upper: np.ndarray, epsilon: float, h: float = 1e-5) -> dict[str, Any]:
    u = np.asarray(u, dtype=np.float64)
    columns: list[np.ndarray] = []
    for index in range(len(u)):
        lo, hi = u.copy(), u.copy()
        lo[index] = max(0.0, u[index] - h)
        hi[index] = min(1.0, u[index] + h)
        theta_lo = lower + lo * (upper - lower)
        theta_hi = lower + hi * (upper - lower)
        column = (np.log(forward(theta_hi) + epsilon) - np.log(forward(theta_lo) + epsilon)) / (hi[index] - lo[index])
        columns.append(column)
    jacobian = np.stack(columns, axis=1)
    singular = np.linalg.svd(jacobian, full_matrices=False, compute_uv=False)
    correlation = np.corrcoef(jacobian, rowvar=False)
    return {"singular_values": singular.tolist(), "column_correlation": correlation.tolist()}


def _conditional_fit(
    observed: np.ndarray,
    forward: ForwardFunction,
    best_u: np.ndarray,
    fixed_index: int,
    fixed_u: float,
    settings: FitSettings,
    seed: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    remaining = [index for index in range(len(best_u)) if index != fixed_index]
    starts = sobol_plus_center(len(remaining), 8, seed)[:-1]
    starts = np.vstack([starts, best_u[remaining]])

    def reduced_forward(reduced_u: np.ndarray) -> np.ndarray:
        full_u = best_u.copy()
        full_u[fixed_index] = fixed_u
        full_u[remaining] = reduced_u
        return forward(full_u)

    reduced_settings = FitSettings(
        epsilon=settings.epsilon,
        sobol_starts=8,
        seed=seed,
        ftol=settings.ftol,
        xtol=settings.xtol,
        gtol=settings.gtol,
        max_nfev=settings.max_nfev,
    )
    reduced = fit_bounded_spectrum(
        observed,
        reduced_forward,
        np.zeros(len(remaining)),
        np.ones(len(remaining)),
        reduced_settings,
        initial_u=starts,
    )
    start_rows: list[dict[str, Any]] = []
    for row in reduced["starts"]:
        full_initial = best_u.copy()
        full_final = best_u.copy()
        full_initial[fixed_index] = fixed_u
        full_final[fixed_index] = fixed_u
        full_initial[remaining] = row["initial_theta"]
        full_final[remaining] = row["final_theta"]
        start_rows.append({**row, "initial_u_full": full_initial.tolist(), "final_u_full": full_final.tolist()})
    if not reduced["success"]:
        return {"success": False, "fixed_u": fixed_u, "logrmse": np.inf}, start_rows
    full_u = best_u.copy()
    full_u[fixed_index] = fixed_u
    full_u[remaining] = reduced["theta"]
    return {
        "success": True,
        "fixed_u": float(fixed_u),
        "conditional_u": full_u,
        "logrmse": float(reduced["metrics"]["logrmse"]),
    }, start_rows


def profile_parameter(
    observed: np.ndarray,
    forward_u: ForwardFunction,
    best_u: np.ndarray,
    parameter_index: int,
    settings: FitSettings,
    seed: int,
    *,
    main_delta: float = 0.005,
    refinement_width_u: float = 0.002,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    grid = np.linspace(0.0, 1.0, 51)
    if not np.any(np.isclose(grid, best_u[parameter_index], atol=1e-14, rtol=0)):
        grid = np.sort(np.append(grid, best_u[parameter_index]))
    aggregate: list[dict[str, Any]] = []
    starts: list[dict[str, Any]] = []
    for grid_index, fixed_u in enumerate(grid):
        result, rows = _conditional_fit(observed, forward_u, best_u, parameter_index, float(fixed_u), settings, seed + grid_index)
        aggregate.append({**result, "grid_kind": "base_or_optimum"})
        starts.extend([{**row, "fixed_u": float(fixed_u), "grid_kind": "base_or_optimum"} for row in rows])
    optimum = min(float(row["logrmse"]) for row in aggregate)
    ordered = sorted(aggregate, key=lambda row: row["fixed_u"])
    transitions = []
    for left, right in zip(ordered[:-1], ordered[1:]):
        if (left["logrmse"] <= optimum + main_delta) != (right["logrmse"] <= optimum + main_delta):
            transitions.append((float(left["fixed_u"]), float(right["fixed_u"])))
    refinement_index = 0
    for low, high in transitions:
        while high - low > refinement_width_u:
            mid = 0.5 * (low + high)
            result, rows = _conditional_fit(observed, forward_u, best_u, parameter_index, mid, settings, seed + 1000 + refinement_index)
            result["grid_kind"] = "refinement_delta_0p005"
            aggregate.append(result)
            starts.extend([{**row, "fixed_u": mid, "grid_kind": "refinement_delta_0p005"} for row in rows])
            refinement_index += 1
            low_accept = next(row for row in aggregate if row["fixed_u"] == low)["logrmse"] <= optimum + main_delta
            if (result["logrmse"] <= optimum + main_delta) == low_accept:
                low = mid
            else:
                high = mid
    return sorted(aggregate, key=lambda row: row["fixed_u"]), starts


def acceptable_intervals(profile: list[dict[str, Any]], optimum: float, delta: float, scale: float) -> dict[str, Any]:
    points = sorted((float(row["fixed_u"]), float(row["logrmse"])) for row in profile if row["success"])
    accepted = [(u, value) for u, value in points if value <= optimum + delta]
    if not accepted:
        return {"intervals": [], "total_span": np.nan, "envelope_span": np.nan}
    intervals_u: list[list[float]] = []
    start = previous = accepted[0][0]
    point_positions = {u: index for index, (u, _) in enumerate(points)}
    for u, _ in accepted[1:]:
        previous_index = point_positions[previous]
        current_index = point_positions[u]
        rejected_between = any(points[index][1] > optimum + delta for index in range(previous_index + 1, current_index))
        if rejected_between:
            intervals_u.append([start, previous])
            start = u
        previous = u
    intervals_u.append([start, previous])
    intervals = [[low * scale, high * scale] for low, high in intervals_u]
    return {
        "intervals": intervals,
        "total_span": float(sum(high - low for low, high in intervals)),
        "envelope_span": float(intervals[-1][1] - intervals[0][0]),
    }

