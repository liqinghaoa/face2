from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


METRIC_KEYS = (
    "roc_auc",
    "accuracy",
    "macro_precision",
    "macro_recall",
    "macro_f1",
    "balanced_accuracy",
    "sensitivity",
    "specificity",
    "brier_score",
)


def compute_metrics(y_true: Any, prob_patient: Any, threshold: float = 0.5) -> dict[str, float | None]:
    y = np.asarray(y_true, dtype=int)
    p = np.asarray(prob_patient, dtype=float)
    pred = (p >= threshold).astype(int)
    out: dict[str, float | None] = {
        "roc_auc": None,
        "accuracy": float(accuracy_score(y, pred)),
        "macro_precision": float(precision_score(y, pred, average="macro", zero_division=0)),
        "macro_recall": float(recall_score(y, pred, average="macro", zero_division=0)),
        "macro_f1": float(f1_score(y, pred, average="macro", zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
        "sensitivity": float(recall_score(y, pred, pos_label=1, zero_division=0)),
        "specificity": float(recall_score(y, pred, pos_label=0, zero_division=0)),
        "brier_score": float(brier_score_loss(y, p)),
    }
    if np.unique(y).size == 2:
        out["roc_auc"] = float(roc_auc_score(y, p))
    cm = confusion_matrix(y, pred, labels=[0, 1])
    out["tn"] = float(cm[0, 0])
    out["fp"] = float(cm[0, 1])
    out["fn"] = float(cm[1, 0])
    out["tp"] = float(cm[1, 1])
    return out


def cluster_bootstrap_metrics(
    frame: pd.DataFrame,
    *,
    label_col: str = "binary_label",
    prob_col: str = "probability_patient",
    cluster_col: str = "patient_group_id",
    repeats: int = 2000,
    seed: int = 2026,
) -> dict[str, Any]:
    point = compute_metrics(frame[label_col].astype(int), frame[prob_col].astype(float))
    y_all = frame[label_col].astype(int).to_numpy()
    p_all = frame[prob_col].astype(float).to_numpy()
    cluster_values = frame[cluster_col].astype(str).to_numpy()
    clusters = pd.unique(cluster_values).tolist()
    grouped_indices = {cluster: np.flatnonzero(cluster_values == cluster) for cluster in clusters}
    rng = np.random.default_rng(seed)
    samples: list[dict[str, float]] = []
    failed = 0
    for _ in range(int(repeats)):
        picked = rng.choice(clusters, size=len(clusters), replace=True)
        idx = np.concatenate([grouped_indices[str(k)] for k in picked])
        y = y_all[idx]
        if np.unique(y).size < 2:
            failed += 1
            continue
        metrics = compute_metrics(y, p_all[idx])
        samples.append({k: float(metrics[k]) for k in METRIC_KEYS if metrics.get(k) is not None})
    ci: dict[str, list[float]] = {}
    if samples:
        sf = pd.DataFrame(samples)
        for col in sf.columns:
            ci[col] = [float(sf[col].quantile(0.025)), float(sf[col].quantile(0.975))]
    return {
        "point_estimates": point,
        "ci95": ci,
        "iterations": int(repeats),
        "valid_iterations": len(samples),
        "failed_iterations": failed,
        "seed": int(seed),
        "cluster_unit": cluster_col,
    }


def smd(control: pd.Series, patient: pd.Series) -> float | None:
    c = pd.to_numeric(control, errors="coerce").dropna().astype(float)
    p = pd.to_numeric(patient, errors="coerce").dropna().astype(float)
    if len(c) < 2 or len(p) < 2:
        return None
    pooled = np.sqrt(((len(c) - 1) * c.var(ddof=1) + (len(p) - 1) * p.var(ddof=1)) / (len(c) + len(p) - 2))
    if pooled == 0 or not np.isfinite(pooled):
        return None
    return float((p.mean() - c.mean()) / pooled)


def cluster_bootstrap_smd(
    frame: pd.DataFrame,
    *,
    value_col: str,
    label_col: str = "binary_label",
    cluster_col: str = "patient_group_id",
    repeats: int = 2000,
    seed: int = 2026,
) -> tuple[float | None, float | None, int]:
    y_all = frame[label_col].astype(int).to_numpy()
    x_all = pd.to_numeric(frame[value_col], errors="coerce").to_numpy(dtype=float)
    cluster_values = frame[cluster_col].astype(str).to_numpy()
    clusters = pd.unique(cluster_values).tolist()
    grouped_indices = {cluster: np.flatnonzero(cluster_values == cluster) for cluster in clusters}
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(int(repeats)):
        picked = rng.choice(clusters, size=len(clusters), replace=True)
        idx = np.concatenate([grouped_indices[str(k)] for k in picked])
        y = y_all[idx]
        if np.unique(y).size < 2:
            continue
        val = smd(pd.Series(x_all[idx][y == 0]), pd.Series(x_all[idx][y == 1]))
        if val is not None and np.isfinite(val):
            vals.append(val)
    if not vals:
        return None, None, 0
    return float(np.quantile(vals, 0.025)), float(np.quantile(vals, 0.975)), len(vals)
