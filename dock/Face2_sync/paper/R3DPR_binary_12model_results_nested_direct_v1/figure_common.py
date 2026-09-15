"""Shared validation and styling helpers for nested-direct manuscript figures."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import matplotlib as mpl
import numpy as np
import pandas as pd


mpl.use("Agg")


DATA_DIR = Path(__file__).resolve().parent
VIEW_CONFIG = {
    "whole_face_sh93": ("A", "Whole-face"),
    "eye_roi": ("B", "Eye"),
    "cheek_roi": ("C", "Cheek"),
    "lip_roi": ("D", "Lip"),
}
BACKBONE_CONFIG = {
    "resnet18": ("ResNet-18", "#3C5488"),
    "resnet34": ("ResNet-34", "#00A087"),
    "resnet50": ("ResNet-50", "#E64B35"),
}
INTEGRITY_FLAGS = (
    "run_finished",
    "nested_direct_protocol",
    "oof_ids_unique",
    "fold_metrics_match_oof",
    "pooled_confusion_matches_oof",
    "cross_model_oof_alignment",
)


def configure_matplotlib() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
            "font.size": 8,
            "axes.linewidth": 0.8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "legend.frameon": False,
            "pdf.fonttype": 42,
            "svg.fonttype": "none",
        }
    )


def as_bool(value: object) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, str):
        return value.strip().lower() == "true"
    return bool(value)


def require_columns(frame: pd.DataFrame, columns: Iterable[str], name: str) -> None:
    missing = sorted(set(columns).difference(frame.columns))
    if missing:
        raise ValueError(f"{name} lacks required columns: {missing}")


def expected_model_keys() -> set[tuple[str, str]]:
    return {(view, backbone) for view in VIEW_CONFIG for backbone in BACKBONE_CONFIG}


def validate_integrity(inventory: pd.DataFrame, integrity: pd.DataFrame) -> None:
    require_columns(inventory, {"model_id", "input_view", "backbone", "n_oof_samples", "n_folds"}, "model_inventory.csv")
    require_columns(integrity, {"model_id", "oof_rows", "fold_metrics_rows", *INTEGRITY_FLAGS}, "integrity_checks.csv")
    if set(inventory["model_id"]) != set(integrity["model_id"]) or len(inventory) != 12:
        raise ValueError("The result package must contain the same 12 models in inventory and integrity tables")
    if set(zip(inventory["input_view"], inventory["backbone"])) != expected_model_keys():
        raise ValueError("The result package does not contain the prespecified four-view, three-backbone model set")
    failures = []
    for row in inventory.itertuples(index=False):
        if int(row.n_oof_samples) != 500 or int(row.n_folds) != 5:
            failures.append(f"{row.model_id}: n_oof_samples={row.n_oof_samples}, n_folds={row.n_folds}")
    for row in integrity.itertuples(index=False):
        false_flags = [flag for flag in INTEGRITY_FLAGS if not as_bool(getattr(row, flag))]
        if int(row.oof_rows) != 500 or int(row.fold_metrics_rows) != 5 or false_flags:
            failures.append(f"{row.model_id}: oof_rows={row.oof_rows}, fold_metrics_rows={row.fold_metrics_rows}, failed={false_flags}")
    if failures:
        raise ValueError("Result-package integrity checks failed:\n" + "\n".join(failures))


def validate_prediction_groups(predictions: pd.DataFrame) -> None:
    require_columns(
        predictions,
        {"model_id", "input_view", "backbone", "sample_id", "fold", "binary_label", "prob_control", "prob_patient", "pred_class"},
        "pooled_oof_predictions_long.csv",
    )
    if set(zip(predictions["input_view"], predictions["backbone"])) != expected_model_keys():
        raise ValueError("OOF prediction model set differs from the prespecified 12 models")
    reference: pd.DataFrame | None = None
    for view, backbone in sorted(expected_model_keys()):
        group = predictions.loc[(predictions["input_view"] == view) & (predictions["backbone"] == backbone)].copy()
        probabilities = group[["prob_control", "prob_patient"]].to_numpy(float)
        if len(group) != 500 or group["sample_id"].nunique() != 500 or set(group["fold"].astype(int)) != set(range(5)):
            raise ValueError(f"{view}/{backbone} does not contain one 500-case five-fold OOF set")
        if set(group["binary_label"].astype(int)) != {0, 1} or not np.isfinite(probabilities).all():
            raise ValueError(f"{view}/{backbone} has invalid labels or probabilities")
        if (probabilities < 0.0).any() or (probabilities > 1.0).any() or not np.allclose(probabilities.sum(axis=1), 1.0, atol=1e-5):
            raise ValueError(f"{view}/{backbone} has invalid softmax probabilities")
        if not np.array_equal(group["pred_class"].to_numpy(int), probabilities.argmax(axis=1)):
            raise ValueError(f"{view}/{backbone} hard predictions are not softmax argmax")
        keys = group.loc[:, ["sample_id", "fold", "binary_label"]].sort_values("sample_id").reset_index(drop=True)
        if reference is None:
            reference = keys
        elif not np.array_equal(keys.to_numpy(), reference.to_numpy()):
            raise ValueError(f"{view}/{backbone} OOF IDs, folds, or labels differ from the other models")
