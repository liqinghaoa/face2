from __future__ import annotations

import re

import numpy as np
import pandas as pd
from scipy import stats


def distribution_table(metrics: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for col in metrics.columns:
        if col == "sample_id":
            continue
        vals = pd.to_numeric(metrics[col], errors="coerce").dropna()
        if vals.empty:
            continue
        rows.append(
            {
                "metric": col,
                "valid_n": int(len(vals)),
                "min": float(vals.min()),
                "q25": float(vals.quantile(0.25)),
                "median": float(vals.median()),
                "q75": float(vals.quantile(0.75)),
                "max": float(vals.max()),
                "mean": float(vals.mean()),
                "std": float(vals.std()),
            }
        )
    return pd.DataFrame(rows)


def consistency(input_df: pd.DataFrame, raw_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    merged = input_df.merge(raw_df, on="sample_id", suffixes=("", "_rawframe"))
    rows = []
    pairs = []
    for col in input_df.columns:
        raw_col = None
        if col.startswith("bright_fraction_"):
            raw_col = "raw_" + col
        elif col.startswith("dark_fraction_"):
            raw_col = "raw_" + col
        elif col.startswith("input_specular_") and col.endswith("_fraction"):
            raw_col = col.replace("input_", "raw_", 1)
        if raw_col and raw_col in raw_df.columns:
            pairs.append((col, raw_col))
    for in_col, raw_col in pairs:
        x = pd.to_numeric(merged[in_col], errors="coerce")
        y = pd.to_numeric(merged[raw_col], errors="coerce")
        ok = x.notna() & y.notna()
        if ok.sum() >= 3 and x[ok].nunique() > 1 and y[ok].nunique() > 1:
            rho, p = stats.spearmanr(x[ok], y[ok])
        else:
            rho, p = np.nan, np.nan
        diff = (y - x).abs()
        rows.append(
            {
                "input_metric": in_col,
                "raw_metric": raw_col,
                "valid_n": int(ok.sum()),
                "spearman_r": float(rho) if np.isfinite(rho) else np.nan,
                "spearman_p": float(p) if np.isfinite(p) else np.nan,
                "median_absolute_difference": float(diff[ok].median()) if ok.any() else np.nan,
                "iqr_absolute_difference": float(diff[ok].quantile(0.75) - diff[ok].quantile(0.25)) if ok.any() else np.nan,
                "max_difference_sample_id": str(merged.loc[diff.idxmax(), "sample_id"]) if ok.any() else "",
                "max_absolute_difference": float(diff.max()) if ok.any() else np.nan,
            }
        )
    detail_rows = []
    for in_col, raw_col in pairs:
        temp = merged[["sample_id", in_col, raw_col]].copy()
        temp["metric_pair"] = f"{in_col}__vs__{raw_col}"
        temp["absolute_difference"] = (pd.to_numeric(temp[raw_col], errors="coerce") - pd.to_numeric(temp[in_col], errors="coerce")).abs()
        temp["relative_difference"] = temp["absolute_difference"] / (pd.to_numeric(temp[in_col], errors="coerce").abs() + 1e-8)
        detail_rows.append(temp.rename(columns={in_col: "input_fraction", raw_col: "raw_fraction"}))
    detail = pd.concat(detail_rows, ignore_index=True) if detail_rows else pd.DataFrame()
    return detail, pd.DataFrame(rows)


def threshold_summary(input_metrics: pd.DataFrame, raw_metrics: pd.DataFrame) -> pd.DataFrame:
    combined = input_metrics.merge(raw_metrics, on="sample_id", how="outer")
    cols = [c for c in combined.columns if any(token in c for token in ("bright_fraction", "dark_fraction", "shadow_ratio", "specular")) and c.endswith("fraction")]
    return distribution_table(combined[["sample_id", *cols]])
