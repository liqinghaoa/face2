"""Deterministic optical-phenotype-only multinomial logistic classifier."""

from __future__ import annotations

import json
import warnings
from dataclasses import asdict, dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping, Sequence

import joblib
import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression

from utils.optical_feature_preprocessor import (
    AVAILABILITY_COLUMN,
    RAW_FEATURE_COLUMNS,
    STAGE2A_FEATURE_COLUMNS,
    STAGE2B_FEATURE_COLUMNS,
    sha256_json,
)


VARIANTS = ("o_mask", "o_raw", "o_stage2a", "o_stage2b")
VARIANT_FEATURE_COLUMNS: Mapping[str, tuple[str, ...]] = MappingProxyType(
    {
        "o_mask": (),
        "o_raw": tuple(RAW_FEATURE_COLUMNS),
        "o_stage2a": tuple(STAGE2A_FEATURE_COLUMNS),
        "o_stage2b": tuple(STAGE2B_FEATURE_COLUMNS),
    }
)
VARIANT_INPUT_COLUMNS: Mapping[str, tuple[str, ...]] = MappingProxyType(
    {
        variant: (*columns, AVAILABILITY_COLUMN)
        for variant, columns in VARIANT_FEATURE_COLUMNS.items()
    }
)
VARIANT_INPUT_DIM: Mapping[str, int] = MappingProxyType(
    {variant: len(columns) for variant, columns in VARIANT_INPUT_COLUMNS.items()}
)
CLASS_NAMES = ("normal", "mild", "severe")
CLASS_ORDER = np.asarray([0, 1, 2], dtype=np.int64)
STD_EPSILON = 1.0e-8
ABSOLUTE_TOLERANCE = 1.0e-12
RELATIVE_TOLERANCE = 1.0e-12


def validate_variant(variant: str) -> str:
    normalized = str(variant).strip().lower()
    if normalized not in VARIANTS:
        raise ValueError(f"Unknown optical-only variant {variant!r}; expected {VARIANTS}")
    return normalized


def _availability(frame: pd.DataFrame) -> np.ndarray:
    if AVAILABILITY_COLUMN not in frame.columns:
        raise ValueError(f"Feature frame is missing {AVAILABILITY_COLUMN}")
    values = pd.to_numeric(frame[AVAILABILITY_COLUMN], errors="coerce").to_numpy(float)
    if not np.isfinite(values).all() or not np.isin(values, [0.0, 1.0]).all():
        raise ValueError("forehead_available must contain only finite 0/1 values")
    return values.astype(np.float64, copy=False)


def validate_feature_frame(frame: pd.DataFrame, variant: str) -> None:
    variant = validate_variant(variant)
    required = set(VARIANT_INPUT_COLUMNS[variant])
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"Feature frame is missing allowlisted columns: {missing}")
    availability = _availability(frame)
    names = VARIANT_FEATURE_COLUMNS[variant]
    if not names:
        return
    values = frame.loc[:, list(names)].apply(pd.to_numeric, errors="coerce").to_numpy(float)
    if not np.isfinite(values[:, :3]).all():
        raise ValueError("All cheek features must be finite")
    available = availability == 1
    if not np.isfinite(values[available, 3:]).all():
        raise ValueError("Available forehead features must all be finite")
    unavailable_values = values[~available, 3:]
    if unavailable_values.size and not np.isnan(unavailable_values).all():
        raise ValueError("Unavailable forehead features must all be NaN")


@dataclass(frozen=True)
class OpticalPhenotypeScaler:
    schema_version: str
    variant: str
    fold: int
    feature_names: list[str]
    mean: list[float]
    std: list[float]
    valid_n: list[int]
    forehead_available_train_n: int
    forehead_unavailable_train_n: int
    ddof: int
    std_epsilon: float
    missing_fill_after_standardization: float
    availability_position: int
    train_id_sha256: str
    feature_source_sha256: str

    def __post_init__(self) -> None:
        variant = validate_variant(self.variant)
        if variant == "o_mask":
            raise ValueError("O-Mask must not have a six-dimensional scaler")
        if self.schema_version != "optical_phenotype_classifier_scaler_v1":
            raise ValueError("Unsupported downstream scaler schema")
        if tuple(self.feature_names) != VARIANT_FEATURE_COLUMNS[variant]:
            raise ValueError("Scaler feature order does not match the locked variant")
        if not (len(self.mean) == len(self.std) == len(self.valid_n) == 6):
            raise ValueError("Scaler mean/std/valid_n must have six values")
        if self.ddof != 0 or self.availability_position != 6:
            raise ValueError("Scaler must use ddof=0 and append availability at index 6")
        if self.missing_fill_after_standardization != 0.0:
            raise ValueError("Unavailable forehead values must be filled with zero")
        mean, std = np.asarray(self.mean), np.asarray(self.std)
        if not np.isfinite(mean).all() or not np.isfinite(std).all():
            raise ValueError("Scaler statistics must be finite")
        if (std < float(self.std_epsilon)).any():
            raise ValueError("Scaler contains a standard deviation below the threshold")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @property
    def payload_sha256(self) -> str:
        return sha256_json(self.to_dict())

    def save_json(self, path: str | Path) -> None:
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("w", encoding="utf-8") as handle:
            json.dump(self.to_dict(), handle, ensure_ascii=False, indent=2, sort_keys=True)

    @classmethod
    def load_json(cls, path: str | Path) -> "OpticalPhenotypeScaler":
        with Path(path).open("r", encoding="utf-8") as handle:
            return cls(**json.load(handle))

    def transform(self, frame: pd.DataFrame) -> np.ndarray:
        validate_feature_frame(frame, self.variant)
        values = frame.loc[:, self.feature_names].to_numpy(dtype=np.float64, copy=True)
        availability = _availability(frame)
        standardized = (values - np.asarray(self.mean)) / np.asarray(self.std)
        standardized[availability == 0, 3:] = 0.0
        result = np.concatenate([standardized, availability[:, None]], axis=1)
        if result.dtype != np.float64 or not np.isfinite(result).all():
            raise ValueError("Standardized model input must be finite float64")
        return result


def fit_scaler(
    frame: pd.DataFrame,
    variant: str,
    fold: int,
    *,
    train_id_sha256: str,
    feature_source_sha256: str,
    std_epsilon: float = STD_EPSILON,
) -> OpticalPhenotypeScaler:
    variant = validate_variant(variant)
    if variant == "o_mask":
        raise ValueError("O-Mask does not fit a scaler")
    validate_feature_frame(frame, variant)
    names = list(VARIANT_FEATURE_COLUMNS[variant])
    values = frame.loc[:, names].to_numpy(dtype=np.float64, copy=True)
    availability = _availability(frame)
    mean, std, valid_n = [], [], []
    for index in range(6):
        eligible = np.ones(len(frame), dtype=bool) if index < 3 else availability == 1
        selected = values[eligible, index]
        if not np.isfinite(selected).all():
            raise ValueError(f"Non-finite values while fitting scaler feature {names[index]}")
        mean.append(float(np.mean(selected, dtype=np.float64)))
        std.append(float(np.std(selected, ddof=0, dtype=np.float64)))
        valid_n.append(int(len(selected)))
    below = [names[index] for index, value in enumerate(std) if value < std_epsilon]
    if below:
        raise ValueError(f"Feature std is below {std_epsilon}: {below}")
    return OpticalPhenotypeScaler(
        schema_version="optical_phenotype_classifier_scaler_v1",
        variant=variant,
        fold=int(fold),
        feature_names=names,
        mean=mean,
        std=std,
        valid_n=valid_n,
        forehead_available_train_n=int((availability == 1).sum()),
        forehead_unavailable_train_n=int((availability == 0).sum()),
        ddof=0,
        std_epsilon=float(std_epsilon),
        missing_fill_after_standardization=0.0,
        availability_position=6,
        train_id_sha256=str(train_id_sha256),
        feature_source_sha256=str(feature_source_sha256),
    )


def build_model_input(
    frame: pd.DataFrame,
    variant: str,
    scaler: OpticalPhenotypeScaler | None,
) -> np.ndarray:
    variant = validate_variant(variant)
    validate_feature_frame(frame, variant)
    if variant == "o_mask":
        if scaler is not None:
            raise ValueError("O-Mask must not receive a scaler")
        result = _availability(frame)[:, None]
    else:
        if scaler is None or scaler.variant != variant:
            raise ValueError(f"{variant} requires its matching fitted scaler")
        result = scaler.transform(frame)
    if result.shape != (len(frame), VARIANT_INPUT_DIM[variant]):
        raise ValueError("Model input shape violates the locked variant definition")
    if result.dtype != np.float64 or not np.isfinite(result).all():
        raise ValueError("Model input must be finite float64")
    return result


def balanced_class_weights(labels: Sequence[int]) -> dict[int, float]:
    y = np.asarray(labels, dtype=np.int64)
    counts = np.bincount(y, minlength=3)
    if y.ndim != 1 or len(y) == 0 or (counts == 0).any():
        raise ValueError("Balanced class weights require all three train classes")
    return {index: float(len(y) / (3.0 * counts[index])) for index in range(3)}


def build_classifier(random_state: int) -> LogisticRegression:
    # multi_class is intentionally omitted for sklearn 1.7: lbfgs with three classes
    # follows the multinomial loss while avoiding the deprecated explicit parameter.
    return LogisticRegression(
        penalty="l2",
        C=1.0,
        solver="lbfgs",
        fit_intercept=True,
        class_weight="balanced",
        max_iter=5000,
        tol=1.0e-8,
        random_state=int(random_state),
    )


def fit_classifier(
    model: LogisticRegression, x_train: np.ndarray, y_train: Sequence[int]
) -> list[dict[str, str]]:
    if x_train.dtype != np.float64 or not np.isfinite(x_train).all():
        raise ValueError("Logistic regression requires finite float64 input")
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        model.fit(x_train, np.asarray(y_train, dtype=np.int64))
    records = [
        {"category": warning.category.__name__, "message": str(warning.message)}
        for warning in caught
    ]
    convergence = [
        warning for warning in caught if issubclass(warning.category, ConvergenceWarning)
    ]
    if convergence or np.asarray(model.n_iter_).max() >= model.max_iter:
        raise RuntimeError(f"Logistic regression did not converge: {records}")
    if not np.array_equal(model.classes_, CLASS_ORDER):
        raise RuntimeError(f"Unexpected classifier class order: {model.classes_.tolist()}")
    if model.coef_.shape != (3, x_train.shape[1]) or model.intercept_.shape != (3,):
        raise RuntimeError("Unexpected multinomial coefficient/intercept shape")
    return records


def predict_probabilities(model: LogisticRegression, features: np.ndarray) -> np.ndarray:
    probabilities = np.asarray(model.predict_proba(features), dtype=np.float64)
    if probabilities.shape != (len(features), 3):
        raise ValueError("predict_proba must return [N,3]")
    if not np.isfinite(probabilities).all() or (probabilities < 0).any() or (probabilities > 1).any():
        raise ValueError("Classifier probabilities are invalid")
    if not np.allclose(probabilities.sum(axis=1), 1.0, atol=1.0e-12, rtol=1.0e-12):
        raise ValueError("Classifier probabilities do not sum to one")
    return probabilities


def save_model(model: LogisticRegression, path: str | Path) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, output, compress=0, protocol=4)


def load_model(path: str | Path) -> LogisticRegression:
    model = joblib.load(Path(path))
    if not isinstance(model, LogisticRegression):
        raise TypeError("Serialized object is not LogisticRegression")
    return model
