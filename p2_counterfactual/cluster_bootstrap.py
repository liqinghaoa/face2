from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from p2_counterfactual.metrics import compute_p2_a_metrics
from p2_counterfactual.stability_metrics import _safe_auc


CLASSIFICATION_METRICS = (
    "macro_auc",
    "macro_f1",
    "balanced_accuracy",
    "patient_sensitivity",
    "control_specificity",
)

STABILITY_METRICS = (
    "mean_prediction_std",
    "case_level_flip_rate",
    "mean_abs_probability_delta",
    "worst_light_auc",
    "mean_feature_cosine",
)

ALL_COMPARISON_METRICS = (*CLASSIFICATION_METRICS, *STABILITY_METRICS)


@dataclass(frozen=True)
class P2AExperimentFrames:
    experiment_id: str
    original: pd.DataFrame
    stability: pd.DataFrame


def experiment_metric_values(original: pd.DataFrame, stability: pd.DataFrame) -> dict[str, float]:
    probs = original[["prob_control", "prob_patient"]].to_numpy(dtype=float)
    labels = original["label"].astype(int).to_numpy()
    cls = compute_p2_a_metrics(labels, probs)
    preset_cols = [column for column in stability.columns if column.startswith("prob_")]
    preset_cols = [column for column in preset_cols if column != "prob_control" and column != "prob_patient"]
    preset_auc = []
    for column in preset_cols:
        if column.startswith("prob_"):
            preset_auc.append(_safe_auc(stability["label"].astype(int).to_numpy(), stability[column].to_numpy(dtype=float)))
    finite = [value for value in preset_auc if np.isfinite(value)]
    return {
        "macro_auc": float(cls["macro_auc"]),
        "macro_f1": float(cls["macro_f1"]),
        "balanced_accuracy": float(cls["balanced_accuracy"]),
        "patient_sensitivity": float(cls["patient_sensitivity"]),
        "control_specificity": float(cls["control_specificity"]),
        "mean_prediction_std": float(stability["prediction_std"].mean()),
        "case_level_flip_rate": float(stability["case_flip"].mean()),
        "mean_abs_probability_delta": float(np.nanmean(_mean_abs_probability_delta(stability))),
        "worst_light_auc": float(np.min(finite)) if finite else float("nan"),
        "mean_feature_cosine": float(stability["mean_feature_cosine"].mean()),
    }


def _sample_by_cluster(frame: pd.DataFrame, picked: np.ndarray, cluster_unit: str) -> pd.DataFrame:
    grouped = {str(cluster): group.copy() for cluster, group in frame.groupby(cluster_unit, sort=False)}
    return pd.concat([grouped[str(cluster)] for cluster in picked], ignore_index=True)


def _align_pairwise_frames(
    a_original: pd.DataFrame,
    b_original: pd.DataFrame,
    a_stability: pd.DataFrame,
    b_stability: pd.DataFrame,
    *,
    cluster_unit: str,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    case_order = a_original["case_id"].astype(str).tolist()
    common_cases = set(case_order) & set(b_original["case_id"].astype(str))
    if len(common_cases) != len(a_original) or len(common_cases) != len(b_original):
        raise ValueError("paired comparison requires identical case sets")
    if a_stability["case_id"].astype(str).nunique() != len(case_order) or b_stability["case_id"].astype(str).nunique() != len(case_order):
        raise ValueError("stability tables must contain exactly one row per paired case")
    a_original = a_original.set_index("case_id", drop=False).loc[case_order].reset_index(drop=True)
    b_original = b_original.set_index("case_id", drop=False).loc[case_order].reset_index(drop=True)
    a_stability = a_stability.set_index("case_id", drop=False).loc[case_order].reset_index(drop=True)
    b_stability = b_stability.set_index("case_id", drop=False).loc[case_order].reset_index(drop=True)
    for frame in (b_original, a_stability, b_stability):
        if not (frame["case_id"].astype(str).to_numpy() == a_original["case_id"].astype(str).to_numpy()).all():
            raise ValueError("paired comparison frame alignment failed")
    if not (b_original[cluster_unit].astype(str).to_numpy() == a_original[cluster_unit].astype(str).to_numpy()).all():
        raise ValueError("paired comparison patient_group_id alignment failed")
    return a_original, b_original, a_stability, b_stability


def _stability_probability_columns(stability: pd.DataFrame) -> list[str]:
    return [
        column
        for column in stability.columns
        if column.startswith("prob_") and column not in {"prob_control", "prob_patient"}
    ]


def _mean_abs_probability_delta(stability: pd.DataFrame) -> np.ndarray:
    abs_cols = [column for column in stability.columns if column.startswith("abs_probability_delta_")]
    if abs_cols:
        return stability[abs_cols].to_numpy(dtype=float).mean(axis=1)
    preset_cols = _stability_probability_columns(stability)
    if not preset_cols or "original_probability" not in stability.columns:
        return np.zeros(len(stability), dtype=float)
    original = stability["original_probability"].to_numpy(dtype=float)[:, None]
    relighted = stability[preset_cols].to_numpy(dtype=float)
    return np.abs(relighted - original).mean(axis=1)


def _arrays_from_frames(original: pd.DataFrame, stability: pd.DataFrame) -> dict[str, Any]:
    preset_cols = _stability_probability_columns(stability)
    return {
        "labels": original["label"].astype(int).to_numpy(),
        "probs": original[["prob_control", "prob_patient"]].to_numpy(dtype=float),
        "prediction_std": stability["prediction_std"].to_numpy(dtype=float),
        "case_flip": stability["case_flip"].to_numpy(dtype=float),
        "mean_abs_probability_delta": _mean_abs_probability_delta(stability),
        "mean_feature_cosine": stability["mean_feature_cosine"].to_numpy(dtype=float),
        "preset_probs": stability[preset_cols].to_numpy(dtype=float) if preset_cols else np.empty((len(stability), 0), dtype=float),
    }


def _fast_binary_auc(labels: np.ndarray, scores: np.ndarray) -> float:
    labels = np.asarray(labels, dtype=np.int64)
    scores = np.asarray(scores, dtype=float)
    pos = int(np.sum(labels == 1))
    neg = int(np.sum(labels == 0))
    if pos == 0 or neg == 0:
        return float("nan")
    order = np.argsort(scores, kind="mergesort")
    sorted_scores = scores[order]
    ranks = np.empty(len(scores), dtype=float)
    start = 0
    while start < len(scores):
        end = start + 1
        while end < len(scores) and sorted_scores[end] == sorted_scores[start]:
            end += 1
        ranks[order[start:end]] = (start + 1 + end) / 2.0
        start = end
    rank_sum_pos = float(np.sum(ranks[labels == 1]))
    return float((rank_sum_pos - pos * (pos + 1) / 2.0) / (pos * neg))


def _fast_classification_values(labels: np.ndarray, probs: np.ndarray) -> dict[str, float]:
    pred = (probs[:, 1] > probs[:, 0]).astype(np.int64)
    labels = labels.astype(np.int64)
    tp = float(np.sum((labels == 1) & (pred == 1)))
    tn = float(np.sum((labels == 0) & (pred == 0)))
    fp = float(np.sum((labels == 0) & (pred == 1)))
    fn = float(np.sum((labels == 1) & (pred == 0)))
    precision_patient = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    precision_control = tn / (tn + fn) if (tn + fn) > 0 else 0.0
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    f1_patient = 2.0 * precision_patient * sensitivity / (precision_patient + sensitivity) if (precision_patient + sensitivity) > 0 else 0.0
    f1_control = 2.0 * precision_control * specificity / (precision_control + specificity) if (precision_control + specificity) > 0 else 0.0
    return {
        "macro_auc": _fast_binary_auc(labels, probs[:, 1]),
        "macro_f1": float((f1_patient + f1_control) / 2.0),
        "balanced_accuracy": float((sensitivity + specificity) / 2.0),
        "patient_sensitivity": float(sensitivity),
        "control_specificity": float(specificity),
    }


def _metric_values_from_arrays(arrays: dict[str, Any], indices: np.ndarray) -> dict[str, float]:
    labels = arrays["labels"][indices]
    probs = arrays["probs"][indices]
    cls = _fast_classification_values(labels, probs)
    preset_auc = [_fast_binary_auc(labels, arrays["preset_probs"][indices, column]) for column in range(arrays["preset_probs"].shape[1])]
    finite = [value for value in preset_auc if np.isfinite(value)]
    return {
        "macro_auc": float(cls["macro_auc"]),
        "macro_f1": float(cls["macro_f1"]),
        "balanced_accuracy": float(cls["balanced_accuracy"]),
        "patient_sensitivity": float(cls["patient_sensitivity"]),
        "control_specificity": float(cls["control_specificity"]),
        "mean_prediction_std": float(np.mean(arrays["prediction_std"][indices])),
        "case_level_flip_rate": float(np.mean(arrays["case_flip"][indices])),
        "mean_abs_probability_delta": float(np.mean(arrays["mean_abs_probability_delta"][indices])),
        "worst_light_auc": float(np.min(finite)) if finite else float("nan"),
        "mean_feature_cosine": float(np.mean(arrays["mean_feature_cosine"][indices])),
    }


def paired_cluster_bootstrap_comparison(
    model_a: P2AExperimentFrames,
    model_b: P2AExperimentFrames,
    *,
    iterations: int = 2000,
    seed: int = 2026,
    cluster_unit: str = "patient_group_id",
) -> list[dict[str, Any]]:
    a_original = model_a.original.copy()
    b_original = model_b.original.copy()
    a_stability = model_a.stability.copy()
    b_stability = model_b.stability.copy()
    for frame in (a_original, b_original, a_stability, b_stability):
        frame["case_id"] = frame["case_id"].astype(str)
        frame[cluster_unit] = frame[cluster_unit].astype(str)
    try:
        a_original, b_original, a_stability, b_stability = _align_pairwise_frames(
            a_original,
            b_original,
            a_stability,
            b_stability,
            cluster_unit=cluster_unit,
        )
    except ValueError as exc:
        raise ValueError(f"paired comparison requires identical case sets: {model_a.experiment_id} vs {model_b.experiment_id}")
    a_point = experiment_metric_values(a_original, a_stability)
    b_point = experiment_metric_values(b_original, b_stability)
    merged_clusters = a_original[["case_id", cluster_unit]].drop_duplicates()
    clusters = merged_clusters[cluster_unit].astype(str).unique().tolist()
    group_indices = {
        str(cluster): group.index.to_numpy(dtype=np.int64)
        for cluster, group in a_original.groupby(cluster_unit, sort=False)
    }
    a_arrays = _arrays_from_frames(a_original, a_stability)
    b_arrays = _arrays_from_frames(b_original, b_stability)
    rng = np.random.default_rng(int(seed))
    deltas = {metric: [] for metric in ALL_COMPARISON_METRICS}
    failed = 0
    for _ in range(int(iterations)):
        picked = rng.choice(clusters, size=len(clusters), replace=True)
        indices = np.concatenate([group_indices[str(cluster)] for cluster in picked])
        if np.unique(a_arrays["labels"][indices]).size < 2:
            failed += 1
            continue
        try:
            a_values = _metric_values_from_arrays(a_arrays, indices)
            b_values = _metric_values_from_arrays(b_arrays, indices)
        except Exception:
            failed += 1
            continue
        for metric in ALL_COMPARISON_METRICS:
            delta = float(b_values[metric] - a_values[metric])
            if np.isfinite(delta):
                deltas[metric].append(delta)
    rows = []
    for metric in ALL_COMPARISON_METRICS:
        samples = np.asarray(deltas[metric], dtype=float)
        rows.append(
            {
                "comparison": f"{model_b.experiment_id} vs {model_a.experiment_id}",
                "metric": metric,
                "model_a": model_a.experiment_id,
                "model_b": model_b.experiment_id,
                "value_a": float(a_point[metric]),
                "value_b": float(b_point[metric]),
                "difference_b_minus_a": float(b_point[metric] - a_point[metric]),
                "ci_lower": float(np.quantile(samples, 0.025)) if samples.size else float("nan"),
                "ci_upper": float(np.quantile(samples, 0.975)) if samples.size else float("nan"),
                "bootstrap_iterations": int(iterations),
                "valid_iterations": int(samples.size),
                "failed_iterations": int(failed),
            }
        )
    return rows


def load_experiment_frames(experiment_dir: str, experiment_id: str) -> P2AExperimentFrames:
    from pathlib import Path

    root = Path(experiment_dir)
    original = pd.read_csv(root / "oof/oof_predictions_original.csv", dtype={"case_id": str, "patient_group_id": str})
    stability = pd.read_csv(root / "summary/case_level_stability.csv", dtype={"case_id": str, "patient_group_id": str})
    return P2AExperimentFrames(experiment_id=experiment_id, original=original, stability=stability)
