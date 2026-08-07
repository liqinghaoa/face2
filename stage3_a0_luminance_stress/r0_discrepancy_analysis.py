from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import accuracy_score, balanced_accuracy_score, brier_score_loss, confusion_matrix, roc_auc_score


PROBABILITY_BINS = [0.0, 0.1, 0.25, 0.4, 0.6, 0.75, 0.9, 1.000000001]
PROBABILITY_BIN_LABELS = ["[0,0.1)", "[0.1,0.25)", "[0.25,0.4)", "[0.4,0.6)", "[0.6,0.75)", "[0.75,0.9)", "[0.9,1.0]"]


def build_discrepancy_frame(stored: pd.DataFrame, reproduced: pd.DataFrame, camera: pd.DataFrame, threshold: float) -> pd.DataFrame:
    left = stored.loc[:, ["sample_id", "patient_group_id", "fold", "binary_label", "prob_patient", "pred_class"]].copy()
    if {"prob_patient", "pred_class"}.issubset(reproduced.columns):
        right = reproduced.loc[:, ["sample_id", "prob_patient", "pred_class"]].copy()
    else:
        right = reproduced.loc[:, ["sample_id", "reproduced_probability", "reproduced_prediction"]].copy()
        right = right.rename(columns={"reproduced_probability": "prob_patient", "reproduced_prediction": "pred_class"})
    left["sample_id"] = left["sample_id"].astype(str)
    right["sample_id"] = right["sample_id"].astype(str)
    cam = camera.loc[:, ["sample_id", "camera_model"]].copy()
    cam["sample_id"] = cam["sample_id"].astype(str)
    frame = left.merge(right, on="sample_id", suffixes=("_stored", "_reproduced"), validate="one_to_one")
    frame = frame.merge(cam, on="sample_id", how="left", validate="one_to_one")
    frame = frame.rename(
        columns={
            "prob_patient_stored": "stored_probability",
            "prob_patient_reproduced": "reproduced_probability",
            "pred_class_stored": "stored_prediction",
            "pred_class_reproduced": "reproduced_prediction",
        }
    )
    frame["signed_difference"] = frame["reproduced_probability"].astype(float) - frame["stored_probability"].astype(float)
    frame["absolute_difference"] = frame["signed_difference"].abs()
    frame["distance_to_threshold_stored"] = (frame["stored_probability"].astype(float) - threshold).abs()
    frame["distance_to_threshold_reproduced"] = (frame["reproduced_probability"].astype(float) - threshold).abs()
    frame["threshold_crossing"] = frame["stored_prediction"].astype(int) != frame["reproduced_prediction"].astype(int)
    frame["probability_bin"] = pd.cut(frame["stored_probability"].astype(float), bins=PROBABILITY_BINS, labels=PROBABILITY_BIN_LABELS, right=False, include_lowest=True)
    return frame.sort_values(["fold", "sample_id"], kind="stable")


def binary_metrics(labels: np.ndarray, probs: np.ndarray, threshold: float = 0.5) -> dict[str, float]:
    pred = (probs >= threshold).astype(int)
    cm = confusion_matrix(labels, pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    return {
        "roc_auc": float(roc_auc_score(labels, probs)),
        "accuracy": float(accuracy_score(labels, pred)),
        "balanced_accuracy": float(balanced_accuracy_score(labels, pred)),
        "sensitivity": float(tp / (tp + fn)) if (tp + fn) else float("nan"),
        "specificity": float(tn / (tn + fp)) if (tn + fp) else float("nan"),
        "brier_score": float(brier_score_loss(labels, probs)),
    }


def _ranking_inversions(values_a: np.ndarray, values_b: np.ndarray) -> int:
    order_a = np.argsort(values_a, kind="stable")
    ranks_b = np.empty_like(order_a)
    ranks_b[np.argsort(values_b, kind="stable")] = np.arange(len(values_b))
    sequence = ranks_b[order_a]
    count = 0
    seen: list[int] = []
    for value in sequence:
        count += sum(prev > value for prev in seen)
        seen.append(int(value))
    return int(count)


def overall_stats(frame: pd.DataFrame, threshold: float = 0.5) -> dict[str, Any]:
    diff = frame["signed_difference"].to_numpy(float)
    abs_diff = np.abs(diff)
    labels = frame["binary_label"].to_numpy(int)
    stored = frame["stored_probability"].to_numpy(float)
    reproduced = frame["reproduced_probability"].to_numpy(float)
    stored_metrics = binary_metrics(labels, stored, threshold)
    reproduced_metrics = binary_metrics(labels, reproduced, threshold)
    metric_diffs = {key: float(reproduced_metrics[key] - stored_metrics[key]) for key in stored_metrics}
    return {
        "n": int(len(frame)),
        "mean_signed_difference": float(diff.mean()),
        "median_signed_difference": float(np.median(diff)),
        "std_signed_difference": float(diff.std(ddof=1)),
        "mean_absolute_difference": float(abs_diff.mean()),
        "median_absolute_difference": float(np.median(abs_diff)),
        "p90_absolute_difference": float(np.quantile(abs_diff, 0.90)),
        "p95_absolute_difference": float(np.quantile(abs_diff, 0.95)),
        "p99_absolute_difference": float(np.quantile(abs_diff, 0.99)),
        "max_absolute_difference": float(abs_diff.max()),
        "rmse": float(np.sqrt(np.mean(diff * diff))),
        "pearson": float(pearsonr(stored, reproduced).statistic),
        "spearman": float(spearmanr(stored, reproduced).statistic),
        "prediction_match_count": int((frame["stored_prediction"].astype(int) == frame["reproduced_prediction"].astype(int)).sum()),
        "threshold_crossing_count": int(frame["threshold_crossing"].sum()),
        "ranking_inversion_count": _ranking_inversions(stored, reproduced),
        "stored_metrics": stored_metrics,
        "reproduced_metrics": reproduced_metrics,
        "metric_differences": metric_diffs,
    }


def grouped_stats(frame: pd.DataFrame, by: list[str]) -> pd.DataFrame:
    rows = []
    for keys, group in frame.groupby(by, dropna=False, observed=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        row = {name: key for name, key in zip(by, keys)}
        diff = group["signed_difference"].to_numpy(float)
        abs_diff = np.abs(diff)
        row.update(
            {
                "n": int(len(group)),
                "mean_signed_difference": float(diff.mean()),
                "mean_absolute_difference": float(abs_diff.mean()),
                "median_absolute_difference": float(np.median(abs_diff)),
                "p95_absolute_difference": float(np.quantile(abs_diff, 0.95)),
                "p99_absolute_difference": float(np.quantile(abs_diff, 0.99)),
                "max_absolute_difference": float(abs_diff.max()),
                "prediction_match_count": int((group["stored_prediction"].astype(int) == group["reproduced_prediction"].astype(int)).sum()),
                "threshold_crossing_count": int(group["threshold_crossing"].sum()),
            }
        )
        rows.append(row)
    return pd.DataFrame(rows)
