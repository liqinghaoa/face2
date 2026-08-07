"""Patient-cluster bootstrap utilities for P1 visit/case evaluation."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, precision_score, recall_score

from metrics.binary_classification_metrics import compute_binary_metrics


VISIT_METRIC_KEYS = (
    "macro_auc",
    "accuracy",
    "macro_precision",
    "macro_recall",
    "macro_f1",
    "balanced_accuracy",
    "pr_auc",
    "patient_sensitivity",
    "control_specificity",
    "ppv",
    "npv",
)

BOOTSTRAP_CI_KEYS = (
    "macro_auc",
    "accuracy",
    "macro_f1",
    "balanced_accuracy",
    "patient_sensitivity",
    "control_specificity",
)

PAIRED_DELTA_KEYS = (
    "delta_roc_auc",
    "delta_accuracy",
    "delta_macro_f1",
    "delta_balanced_accuracy",
    "delta_sensitivity",
    "delta_specificity",
)


def _scalar_metrics(metrics: dict[str, Any]) -> dict[str, float]:
    return {key: float(value) for key, value in metrics.items() if np.isscalar(value)}


def _prediction_matrix(frame: pd.DataFrame, prob_control_col: str = "prob_control", prob_patient_col: str = "prob_patient") -> np.ndarray:
    if prob_control_col in frame.columns and prob_patient_col in frame.columns:
        probs = frame[[prob_control_col, prob_patient_col]].to_numpy(dtype=float)
    elif prob_patient_col in frame.columns:
        patient = frame[prob_patient_col].to_numpy(dtype=float)
        probs = np.column_stack([1.0 - patient, patient])
    else:
        raise ValueError(f"frame is missing {prob_patient_col!r}")
    return probs


def compute_visit_metrics(
    frame: pd.DataFrame,
    *,
    prob_control_col: str = "prob_control",
    prob_patient_col: str = "prob_patient",
) -> dict[str, Any]:
    """Compute the exact visit/case metrics used by the corrected protocol."""

    true = frame["label_binary"].astype(int).to_numpy()
    probs = _prediction_matrix(frame, prob_control_col=prob_control_col, prob_patient_col=prob_patient_col)
    metrics = compute_binary_metrics(true, probs)
    pred = probs.argmax(axis=1)
    metrics.update(
        {
            "pr_auc": float(average_precision_score(true, probs[:, 1])),
            "patient_sensitivity": float(recall_score(true, pred, pos_label=1, zero_division=0)),
            "control_specificity": float(recall_score(true, pred, pos_label=0, zero_division=0)),
            "ppv": float(precision_score(true, pred, pos_label=1, zero_division=0)),
            "npv": float(precision_score(true, pred, pos_label=0, zero_division=0)),
        }
    )
    return metrics


def _bootstrap_result_template(
    frame: pd.DataFrame,
    *,
    iterations: int,
    seed: int,
    cluster_unit: str,
    metric_unit: str,
) -> dict[str, Any]:
    return {
        "status": "available",
        "iterations": int(iterations),
        "seed": int(seed),
        "cluster_unit": cluster_unit,
        "metric_unit": metric_unit,
        "unique_clusters": int(frame[cluster_unit].astype(str).nunique()),
        "visit_count": int(len(frame)),
    }


def cluster_bootstrap_visit_metrics(
    frame: pd.DataFrame,
    *,
    iterations: int = 2000,
    seed: int = 2026,
    cluster_unit: str = "patient_group_id",
    metric_unit: str = "visit_case",
) -> dict[str, Any]:
    """Bootstrap visit/case metrics by resampling patient groups."""

    clusters = frame[cluster_unit].astype(str).unique().tolist()
    rng = np.random.default_rng(seed)
    point_estimates = _scalar_metrics(compute_visit_metrics(frame))
    bootstrap_samples: list[dict[str, float]] = []
    failed_iterations = 0

    grouped = {
        str(cluster): group.copy()
        for cluster, group in frame.groupby(cluster_unit, sort=False)
    }

    for _ in range(int(iterations)):
        picked = rng.choice(clusters, size=len(clusters), replace=True)
        sample = pd.concat([grouped[str(cluster)] for cluster in picked], ignore_index=True)
        if sample["label_binary"].nunique() < 2:
            failed_iterations += 1
            continue
        try:
            metrics = _scalar_metrics(compute_visit_metrics(sample))
        except ValueError:
            failed_iterations += 1
            continue
        bootstrap_samples.append({key: metrics[key] for key in BOOTSTRAP_CI_KEYS})

    if bootstrap_samples:
        sample_frame = pd.DataFrame(bootstrap_samples)
        bootstrap_mean = {key: float(sample_frame[key].mean()) for key in BOOTSTRAP_CI_KEYS}
        ci95 = {
            key: [
                float(sample_frame[key].quantile(0.025)),
                float(sample_frame[key].quantile(0.975)),
            ]
            for key in BOOTSTRAP_CI_KEYS
        }
    else:
        bootstrap_mean = {}
        ci95 = {}

    valid_iterations = len(bootstrap_samples)
    status = "available"
    if valid_iterations < max(1, int(iterations * 0.95)):
        status = "warning_low_valid_iterations"
    if valid_iterations == 0:
        status = "failed_no_valid_iterations"

    result = _bootstrap_result_template(
        frame,
        iterations=iterations,
        seed=seed,
        cluster_unit=cluster_unit,
        metric_unit=metric_unit,
    )
    result.update(
        {
            "status": status,
            "point_estimates": point_estimates,
            "bootstrap_mean": bootstrap_mean,
            "ci95": ci95,
            "failed_iterations": int(failed_iterations),
            "valid_iterations": int(valid_iterations),
        }
    )
    return result


def patient_cluster_bootstrap(
    frame: pd.DataFrame,
    *,
    iterations: int = 2000,
    seed: int = 2026,
    cluster_unit: str = "patient_group_id",
    metric_unit: str = "visit_case",
) -> dict[str, Any]:
    """Alias for the corrected visit/case cluster bootstrap contract."""

    return cluster_bootstrap_visit_metrics(
        frame,
        iterations=iterations,
        seed=seed,
        cluster_unit=cluster_unit,
        metric_unit=metric_unit,
    )


def load_historical_e0b_predictions(path: Path) -> pd.DataFrame:
    """Load the historical E0B prediction file used for pairing."""

    frame = pd.read_csv(path)
    required = {"sample_id", "patient_group_id", "prob_patient"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"historical E0B predictions missing columns: {sorted(missing)}")
    frame = frame.rename(columns={"sample_id": "case_id"}).copy()
    frame["case_id"] = frame["case_id"].astype(str)
    frame["patient_group_id"] = frame["patient_group_id"].astype(str)
    if "binary_label" not in frame.columns and "label_binary" in frame.columns:
        frame["binary_label"] = frame["label_binary"]
    if "binary_label" not in frame.columns:
        raise ValueError("historical E0B predictions missing binary_label/label_binary")
    frame["binary_label"] = frame["binary_label"].astype(int)
    if "pred_class" not in frame.columns and "pred_binary" in frame.columns:
        frame["pred_class"] = frame["pred_binary"]
    if "pred_class" not in frame.columns:
        raise ValueError("historical E0B predictions missing pred_class/pred_binary")
    frame["pred_class"] = frame["pred_class"].astype(int)
    return frame


def paired_patient_cluster_visit_bootstrap(
    case_frame: pd.DataFrame,
    historical_frame: pd.DataFrame,
    *,
    iterations: int = 2000,
    seed: int = 2026,
    cluster_unit: str = "patient_group_id",
    metric_unit: str = "visit_case",
) -> dict[str, Any]:
    """Cluster bootstrap deltas between P1-RGB and historical E0B."""

    case = case_frame.copy()
    case["case_id"] = case["case_id"].astype(str)
    case["patient_group_id"] = case["patient_group_id"].astype(str)
    historical = historical_frame.copy()
    historical["case_id"] = historical["case_id"].astype(str)
    historical["patient_group_id"] = historical["patient_group_id"].astype(str)
    if "prob_patient" not in historical.columns:
        raise ValueError("historical predictions must include prob_patient")
    if "binary_label" not in historical.columns and "label_binary" in historical.columns:
        historical["binary_label"] = historical["label_binary"]
    if "binary_label" not in historical.columns:
        raise ValueError("historical predictions must include binary_label or label_binary")
    if "pred_class" not in historical.columns and "pred_binary" in historical.columns:
        historical["pred_class"] = historical["pred_binary"]
    if "pred_class" not in historical.columns:
        raise ValueError("historical predictions must include pred_class or pred_binary")
    historical["binary_label"] = historical["binary_label"].astype(int)
    historical["pred_class"] = historical["pred_class"].astype(int)

    merged = case.merge(
        historical[["case_id", "prob_patient", "pred_class"]],
        on="case_id",
        how="inner",
        suffixes=("", "_historical"),
    )
    if len(merged) != len(case) or merged["case_id"].nunique() != len(case):
        raise ValueError("historical E0B predictions do not fully match the 500-case P1 frame")

    grouped = {
        str(cluster): group.copy()
        for cluster, group in merged.groupby(cluster_unit, sort=False)
    }
    clusters = list(grouped)
    rng = np.random.default_rng(seed)

    point_p1 = _scalar_metrics(compute_visit_metrics(merged))
    old_probs = np.column_stack([1.0 - merged["prob_patient_historical"].to_numpy(dtype=float), merged["prob_patient_historical"].to_numpy(dtype=float)])
    y_old = merged["label_binary"].astype(int).to_numpy() if "label_binary" in merged.columns else merged["binary_label"].astype(int).to_numpy()
    old_visit_metrics = compute_binary_metrics(y_old, old_probs)
    old_visit_metrics.update(
        {
            "pr_auc": float(average_precision_score(y_old, old_probs[:, 1])),
            "patient_sensitivity": float(
                recall_score(
                    y_old,
                    old_probs.argmax(axis=1),
                    pos_label=1,
                    zero_division=0,
                )
            ),
            "control_specificity": float(
                recall_score(
                    y_old,
                    old_probs.argmax(axis=1),
                    pos_label=0,
                    zero_division=0,
                )
            ),
            "ppv": float(
                precision_score(
                    y_old,
                    old_probs.argmax(axis=1),
                    pos_label=1,
                    zero_division=0,
                )
            ),
            "npv": float(
                precision_score(
                    y_old,
                    old_probs.argmax(axis=1),
                    pos_label=0,
                    zero_division=0,
                )
            ),
        }
    )
    point_old = _scalar_metrics(old_visit_metrics)

    delta_samples: list[dict[str, float]] = []
    failed_iterations = 0
    for _ in range(int(iterations)):
        picked = rng.choice(clusters, size=len(clusters), replace=True)
        sample = pd.concat([grouped[str(cluster)] for cluster in picked], ignore_index=True)
        y = sample["label_binary"].astype(int).to_numpy() if "label_binary" in sample.columns else sample["binary_label"].astype(int).to_numpy()
        if np.unique(y).size < 2:
            failed_iterations += 1
            continue
        old_probs = np.column_stack(
            [1.0 - sample["prob_patient_historical"].to_numpy(dtype=float), sample["prob_patient_historical"].to_numpy(dtype=float)]
        )
        try:
            p1_metrics = compute_visit_metrics(sample)
            old_metrics = compute_binary_metrics(y, old_probs)
            old_metrics.update(
                {
                    "pr_auc": float(average_precision_score(y, old_probs[:, 1])),
                    "patient_sensitivity": float(recall_score(y, old_probs.argmax(axis=1), pos_label=1, zero_division=0)),
                    "control_specificity": float(recall_score(y, old_probs.argmax(axis=1), pos_label=0, zero_division=0)),
                    "ppv": float(precision_score(y, old_probs.argmax(axis=1), pos_label=1, zero_division=0)),
                    "npv": float(precision_score(y, old_probs.argmax(axis=1), pos_label=0, zero_division=0)),
                }
            )
        except ValueError:
            failed_iterations += 1
            continue
        delta_samples.append(
            {
                "delta_roc_auc": float(p1_metrics["macro_auc"] - old_metrics["macro_auc"]),
                "delta_accuracy": float(p1_metrics["accuracy"] - old_metrics["accuracy"]),
                "delta_macro_f1": float(p1_metrics["macro_f1"] - old_metrics["macro_f1"]),
                "delta_balanced_accuracy": float(
                    p1_metrics["balanced_accuracy"] - old_metrics["balanced_accuracy"]
                ),
                "delta_sensitivity": float(
                    p1_metrics["patient_sensitivity"] - old_metrics["patient_sensitivity"]
                ),
                "delta_specificity": float(
                    p1_metrics["control_specificity"] - old_metrics["control_specificity"]
                ),
            }
        )

    if delta_samples:
        sample_frame = pd.DataFrame(delta_samples)
        bootstrap_mean = {
            key: float(sample_frame[key].mean())
            for key in PAIRED_DELTA_KEYS
        }
        ci95 = {
            key: [
                float(sample_frame[key].quantile(0.025)),
                float(sample_frame[key].quantile(0.975)),
            ]
            for key in PAIRED_DELTA_KEYS
        }
    else:
        bootstrap_mean = {}
        ci95 = {}

    valid_iterations = len(delta_samples)
    status = "available"
    if valid_iterations < max(1, int(iterations * 0.95)):
        status = "warning_low_valid_iterations"
    if valid_iterations == 0:
        status = "failed_no_valid_iterations"

    return {
        "status": status,
        "iterations": int(iterations),
        "seed": int(seed),
        "cluster_unit": cluster_unit,
        "metric_unit": metric_unit,
        "unique_clusters": int(case[cluster_unit].astype(str).nunique()),
        "visit_count": int(len(case)),
        "point_estimates": {
            "p1_rgb": point_p1,
            "historical_e0b": point_old,
            "delta": {
                "delta_roc_auc": float(point_p1["macro_auc"] - point_old["macro_auc"]),
                "delta_accuracy": float(point_p1["accuracy"] - point_old["accuracy"]),
                "delta_macro_f1": float(point_p1["macro_f1"] - point_old["macro_f1"]),
                "delta_balanced_accuracy": float(
                    point_p1["balanced_accuracy"] - point_old["balanced_accuracy"]
                ),
                "delta_sensitivity": float(
                    point_p1["patient_sensitivity"] - point_old["patient_sensitivity"]
                ),
                "delta_specificity": float(
                    point_p1["control_specificity"] - point_old["control_specificity"]
                ),
            },
        },
        "bootstrap_mean": {"delta": bootstrap_mean},
        "ci95": ci95,
        "failed_iterations": int(failed_iterations),
        "valid_iterations": int(valid_iterations),
    }
