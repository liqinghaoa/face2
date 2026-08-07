from __future__ import annotations

from typing import Any, Callable

import numpy as np
import pandas as pd


def sample_patient_groups(
    frame: pd.DataFrame,
    *,
    cluster_col: str = "patient_group_id",
    iterations: int = 2000,
    seed: int = 2026,
    statistic: Callable[[pd.DataFrame], dict[str, float | None]],
) -> dict[str, Any]:
    clusters = pd.unique(frame[cluster_col].astype(str)).tolist()
    grouped = {cluster: frame.index[frame[cluster_col].astype(str) == cluster].to_numpy() for cluster in clusters}
    rng = np.random.default_rng(seed)
    rows: list[dict[str, float]] = []
    invalid = 0
    for _ in range(iterations):
        picked = rng.choice(clusters, size=len(clusters), replace=True)
        idx = np.concatenate([grouped[str(cluster)] for cluster in picked])
        stat = statistic(frame.loc[idx].copy())
        valid = {k: float(v) for k, v in stat.items() if v is not None and np.isfinite(v)}
        if not valid:
            invalid += 1
            continue
        rows.append(valid)
    ci: dict[str, list[float | None]] = {}
    if rows:
        samples = pd.DataFrame(rows)
        for col in samples.columns:
            vals = pd.to_numeric(samples[col], errors="coerce").dropna()
            ci[col] = [float(vals.quantile(0.025)), float(vals.quantile(0.975))] if len(vals) else [None, None]
    return {
        "ci95": ci,
        "valid_iterations": len(rows),
        "invalid_iterations": int(invalid),
        "iterations": int(iterations),
        "seed": int(seed),
        "cluster_unit": cluster_col,
    }


def cluster_bootstrap_spearman(
    frame: pd.DataFrame,
    *,
    x_col: str,
    y_col: str,
    cluster_col: str = "patient_group_id",
    iterations: int = 2000,
    seed: int = 2026,
) -> tuple[float | None, float | None, int]:
    from scipy.stats import rankdata

    work = frame[[cluster_col, x_col, y_col]].copy()
    work[x_col] = pd.to_numeric(work[x_col], errors="coerce")
    work[y_col] = pd.to_numeric(work[y_col], errors="coerce")
    work = work.dropna()
    if len(work) < 3:
        return None, None, 0
    work["_x_rank"] = rankdata(work[x_col].to_numpy(dtype=float), method="average")
    work["_y_rank"] = rankdata(work[y_col].to_numpy(dtype=float), method="average")
    cluster_codes, clusters = pd.factorize(work[cluster_col].astype(str), sort=False)
    grouped = [
        (
            work.loc[cluster_codes == code, "_x_rank"].to_numpy(dtype=float),
            work.loc[cluster_codes == code, "_y_rank"].to_numpy(dtype=float),
        )
        for code in range(len(clusters))
    ]
    rng = np.random.default_rng(seed)
    vals: list[float] = []
    n_clusters = len(grouped)
    for _ in range(iterations):
        picked = rng.integers(0, n_clusters, size=n_clusters)
        x = np.concatenate([grouped[int(code)][0] for code in picked])
        y = np.concatenate([grouped[int(code)][1] for code in picked])
        if x.size < 3 or np.unique(x).size < 2 or np.unique(y).size < 2:
            continue
        x_centered = x - x.mean()
        y_centered = y - y.mean()
        denom = float(np.sqrt(np.dot(x_centered, x_centered) * np.dot(y_centered, y_centered)))
        if denom > 0:
            vals.append(float(np.dot(x_centered, y_centered) / denom))
    if not vals:
        return None, None, 0
    return float(np.quantile(vals, 0.025)), float(np.quantile(vals, 0.975)), len(vals)
