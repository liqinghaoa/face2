"""Low-capacity, label-free acquisition-condition calibration utilities.

This module contains only deterministic NumPy/pandas operations.  It does not
load split files, clinical labels, images, or NYHA outcomes.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

REFERENCE_CAMERA = "HONOR/BVL-AN00"
XIAOMI_CAMERA = "Xiaomi/M2006J10C"
EXPECTED_CAMERAS = (REFERENCE_CAMERA, XIAOMI_CAMERA)
CONDITION_NAMES = ("relative_optical_exposure", "log2_iso_condition")
DESIGN_FEATURE_NAMES = (
    "camera_xiaomi",
    "z_relative_optical_exposure",
    "z_log2_iso_condition",
    "camera_xiaomi_x_z_exposure",
    "camera_xiaomi_x_z_iso",
)
CHEEK_TARGETS = (
    "cheek_mean_log2_y",
    "cheek_mean_log2_rg",
    "cheek_mean_log2_bg",
)
FOREHEAD_CHEEK_TARGETS = (
    "forehead_minus_cheek_log2_y",
    "forehead_minus_cheek_log2_rg",
    "forehead_minus_cheek_log2_bg",
)
ALL_TARGETS = CHEEK_TARGETS + FOREHEAD_CHEEK_TARGETS


def validate_camera_values(values: pd.Series | Sequence[Any], require_both: bool = False) -> list[str]:
    series = pd.Series(values, dtype="object")
    if series.isna().any() or series.astype(str).str.strip().eq("").any():
        raise ValueError("camera_id contains a missing or empty value")
    unique = sorted(series.astype(str).unique().tolist())
    unknown = sorted(set(unique).difference(EXPECTED_CAMERAS))
    if unknown:
        raise ValueError(f"Unknown camera_id values: {unknown}")
    if require_both and set(unique) != set(EXPECTED_CAMERAS):
        raise ValueError(f"Both expected cameras are required, found: {unique}")
    return unique


def fit_condition_scaler(train: pd.DataFrame, std_epsilon: float = 1.0e-8) -> dict[str, Any]:
    if not math.isfinite(float(std_epsilon)) or float(std_epsilon) <= 0:
        raise ValueError("std_epsilon must be finite and positive")
    validate_camera_values(train["camera_id"], require_both=True)
    parameters: dict[str, Any] = {}
    for camera in EXPECTED_CAMERAS:
        subset = train.loc[train["camera_id"].astype(str) == camera]
        if subset.empty:
            raise ValueError(f"No training rows for camera: {camera}")
        camera_record: dict[str, Any] = {"train_n": int(len(subset)), "conditions": {}}
        for condition in CONDITION_NAMES:
            values = pd.to_numeric(subset[condition], errors="coerce").to_numpy(dtype=np.float64)
            if not np.isfinite(values).all():
                raise ValueError(f"Nonfinite training condition: {camera}/{condition}")
            mean = float(np.mean(values))
            std = float(np.std(values, ddof=0))
            camera_record["conditions"][condition] = {
                "mean": mean,
                "population_std": std,
                "degenerate_std": bool(std < float(std_epsilon)),
            }
        parameters[camera] = camera_record
    return {
        "method": "training-fold camera-specific population standardization",
        "input_field_order": list(CONDITION_NAMES),
        "std_epsilon": float(std_epsilon),
        "camera_parameters": parameters,
    }


def transform_conditions(frame: pd.DataFrame, scaler: Mapping[str, Any]) -> pd.DataFrame:
    validate_camera_values(frame["camera_id"], require_both=False)
    result = frame.copy()
    for condition in CONDITION_NAMES:
        values = pd.to_numeric(result[condition], errors="coerce").to_numpy(dtype=np.float64)
        if not np.isfinite(values).all():
            raise ValueError(f"Nonfinite condition values: {condition}")
        z = np.empty(len(result), dtype=np.float64)
        for camera in EXPECTED_CAMERAS:
            mask = result["camera_id"].astype(str).to_numpy() == camera
            if not mask.any():
                continue
            record = scaler["camera_parameters"][camera]["conditions"][condition]
            if bool(record["degenerate_std"]):
                z[mask] = 0.0
            else:
                z[mask] = (values[mask] - float(record["mean"])) / float(record["population_std"])
        result[f"z_{condition}"] = z
    return result


def build_design_matrix(frame: pd.DataFrame) -> np.ndarray:
    validate_camera_values(frame["camera_id"], require_both=False)
    d = (frame["camera_id"].astype(str).to_numpy() == XIAOMI_CAMERA).astype(np.float64)
    z_exposure = pd.to_numeric(frame["z_relative_optical_exposure"], errors="coerce").to_numpy(float)
    z_iso = pd.to_numeric(frame["z_log2_iso_condition"], errors="coerce").to_numpy(float)
    matrix = np.column_stack((d, z_exposure, z_iso, d * z_exposure, d * z_iso))
    if matrix.shape != (len(frame), len(DESIGN_FEATURE_NAMES)) or not np.isfinite(matrix).all():
        raise ValueError("Design matrix is nonfinite or has an unexpected shape")
    return matrix


@dataclass(frozen=True)
class RidgeModel:
    alpha: float
    intercept: np.ndarray
    coefficients: np.ndarray
    target_names: tuple[str, ...]
    solver: str

    def predict(self, x: np.ndarray) -> np.ndarray:
        matrix = np.asarray(x, dtype=np.float64)
        if matrix.ndim != 2 or matrix.shape[1] != self.coefficients.shape[0]:
            raise ValueError("Prediction design matrix has an unexpected shape")
        return self.intercept + matrix @ self.coefficients

    def to_dict(self) -> dict[str, Any]:
        return {
            "alpha": float(self.alpha),
            "fit_intercept": True,
            "penalize_intercept": False,
            "target_names": list(self.target_names),
            "condition_feature_names": list(DESIGN_FEATURE_NAMES),
            "intercept": self.intercept.astype(float).tolist(),
            "coefficient_matrix": self.coefficients.astype(float).tolist(),
            "coefficient_matrix_orientation": "rows=condition_feature_names, columns=target_names",
            "numerical_solver": self.solver,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "RidgeModel":
        if tuple(payload["condition_feature_names"]) != DESIGN_FEATURE_NAMES:
            raise ValueError("Unexpected condition feature order in model JSON")
        return cls(
            alpha=float(payload["alpha"]),
            intercept=np.asarray(payload["intercept"], dtype=np.float64),
            coefficients=np.asarray(payload["coefficient_matrix"], dtype=np.float64),
            target_names=tuple(str(value) for value in payload["target_names"]),
            solver=str(payload["numerical_solver"]),
        )


def fit_ridge(
    x: np.ndarray,
    y: np.ndarray,
    target_names: Sequence[str],
    alpha: float = 1.0,
) -> RidgeModel:
    matrix = np.asarray(x, dtype=np.float64)
    targets = np.asarray(y, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[1] != len(DESIGN_FEATURE_NAMES):
        raise ValueError("Ridge X must have five design columns")
    if targets.ndim == 1:
        targets = targets[:, None]
    if targets.ndim != 2 or targets.shape[0] != matrix.shape[0]:
        raise ValueError("Ridge X/Y row counts do not match")
    if targets.shape[1] != len(tuple(target_names)):
        raise ValueError("Ridge target name count does not match Y")
    if not np.isfinite(matrix).all() or not np.isfinite(targets).all():
        raise ValueError("Ridge inputs must be finite")
    if float(alpha) != 1.0:
        raise ValueError("Stage 2A fixes alpha at 1.0")
    x_aug = np.column_stack((np.ones(len(matrix), dtype=np.float64), matrix))
    penalty = np.diag([0.0, 1.0, 1.0, 1.0, 1.0, 1.0])
    lhs = x_aug.T @ x_aug + float(alpha) * penalty
    rhs = x_aug.T @ targets
    solver = "numpy.linalg.solve"
    try:
        beta = np.linalg.solve(lhs, rhs)
    except np.linalg.LinAlgError:
        beta = np.linalg.pinv(lhs) @ rhs
        solver = "numpy.linalg.pinv_fallback"
    return RidgeModel(
        alpha=float(alpha),
        intercept=beta[0].copy(),
        coefficients=beta[1:].copy(),
        target_names=tuple(str(value) for value in target_names),
        solver=solver,
    )


def calibrate_values(raw: np.ndarray, predicted: np.ndarray, reference_mean: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    raw_values = np.asarray(raw, dtype=np.float64)
    predicted_values = np.asarray(predicted, dtype=np.float64)
    mean_values = np.asarray(reference_mean, dtype=np.float64)
    residual = raw_values - predicted_values
    calibrated = residual + mean_values
    return residual, calibrated


def spearman_rho(x: pd.Series | np.ndarray, y: pd.Series | np.ndarray) -> tuple[int, float]:
    pair = pd.DataFrame({"x": x, "y": y}).apply(pd.to_numeric, errors="coerce").dropna()
    if len(pair) < 2 or pair["x"].nunique() < 2 or pair["y"].nunique() < 2:
        return int(len(pair)), math.nan
    rho = pair["x"].rank(method="average").corr(pair["y"].rank(method="average"))
    return int(len(pair)), float(rho)


def fit_error_metrics(raw: np.ndarray, predicted: np.ndarray) -> dict[str, float | int]:
    pair = np.column_stack((np.asarray(raw, float), np.asarray(predicted, float)))
    pair = pair[np.isfinite(pair).all(axis=1)]
    if len(pair) == 0:
        return {"valid_n": 0, "mae": math.nan, "rmse": math.nan, "r2": math.nan}
    errors = pair[:, 0] - pair[:, 1]
    sse = float(np.sum(errors**2))
    sst = float(np.sum((pair[:, 0] - np.mean(pair[:, 0])) ** 2))
    return {
        "valid_n": int(len(pair)),
        "mae": float(np.mean(np.abs(errors))),
        "rmse": float(np.sqrt(np.mean(errors**2))),
        "r2": float(1.0 - sse / sst) if sst > 0 else math.nan,
    }


def camera_difference_metrics(frame: pd.DataFrame, column: str) -> dict[str, Any]:
    values: dict[str, np.ndarray] = {}
    records: dict[str, Any] = {}
    for camera in EXPECTED_CAMERAS:
        array = pd.to_numeric(
            frame.loc[frame["camera_id"].astype(str) == camera, column], errors="coerce"
        ).dropna().to_numpy(float)
        values[camera] = array
        records[f"{camera}_valid_n"] = int(len(array))
        records[f"{camera}_mean"] = float(np.mean(array)) if len(array) else math.nan
        records[f"{camera}_median"] = float(np.median(array)) if len(array) else math.nan
        records[f"{camera}_std"] = float(np.std(array, ddof=1)) if len(array) > 1 else math.nan
    honor, xiaomi = values[REFERENCE_CAMERA], values[XIAOMI_CAMERA]
    mean_difference = float(np.mean(honor) - np.mean(xiaomi)) if len(honor) and len(xiaomi) else math.nan
    median_difference = float(np.median(honor) - np.median(xiaomi)) if len(honor) and len(xiaomi) else math.nan
    pooled_denom = len(honor) + len(xiaomi) - 2
    pooled_std = math.nan
    if len(honor) > 1 and len(xiaomi) > 1 and pooled_denom > 0:
        pooled_variance = ((len(honor) - 1) * np.var(honor, ddof=1) + (len(xiaomi) - 1) * np.var(xiaomi, ddof=1)) / pooled_denom
        pooled_std = float(np.sqrt(pooled_variance))
    smd = mean_difference / pooled_std if math.isfinite(pooled_std) and pooled_std > 0 else math.nan
    return {
        **records,
        "mean_difference_honor_minus_xiaomi": mean_difference,
        "median_difference_honor_minus_xiaomi": median_difference,
        "pooled_std": pooled_std,
        "standardized_mean_difference": float(smd) if math.isfinite(smd) else math.nan,
    }


def coefficient_records(fold: int, model: RidgeModel) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for target_index, target in enumerate(model.target_names):
        beta = model.coefficients[:, target_index]
        entries = {
            "intercept": model.intercept[target_index],
            **{name: beta[index] for index, name in enumerate(DESIGN_FEATURE_NAMES)},
            "honor_exposure_slope": beta[1],
            "honor_iso_slope": beta[2],
            "xiaomi_exposure_slope": beta[1] + beta[3],
            "xiaomi_iso_slope": beta[2] + beta[4],
            "device_intercept_difference": beta[0],
        }
        for coefficient_name, value in entries.items():
            rows.append({
                "fold": int(fold), "target": target,
                "coefficient_name": coefficient_name, "value": float(value),
            })
    return rows


def coefficient_stability(records: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for (target, name), group in records.groupby(["target", "coefficient_name"], sort=True):
        values = pd.to_numeric(group["value"], errors="coerce").dropna().to_numpy(float)
        positive = int((values > 0).sum())
        negative = int((values < 0).sum())
        zero = int((values == 0).sum())
        rows.append({
            "target": target,
            "coefficient_name": name,
            "fold_valid_n": int(len(values)),
            "mean": float(np.mean(values)) if len(values) else math.nan,
            "std": float(np.std(values, ddof=0)) if len(values) else math.nan,
            "min": float(np.min(values)) if len(values) else math.nan,
            "max": float(np.max(values)) if len(values) else math.nan,
            "median": float(np.median(values)) if len(values) else math.nan,
            "positive_fold_n": positive,
            "negative_fold_n": negative,
            "zero_fold_n": zero,
            "sign_consistent_fold_n": max(positive, negative, zero),
        })
    return pd.DataFrame(rows)
