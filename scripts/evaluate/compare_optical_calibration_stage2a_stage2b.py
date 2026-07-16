"""Fair ID/fold/target-aligned comparison of Stage 2A and Stage 2B."""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.optical_condition_calibration import (  # noqa: E402
    ALL_TARGETS,
    CONDITION_NAMES,
    camera_difference_metrics,
    fit_error_metrics,
    spearman_rho,
)
from utils.optical_condition_calibration_nn import write_csv  # noqa: E402


def _append_metric(
    rows: list[dict[str, Any]],
    fold: int,
    target: str,
    family: str,
    metric: str,
    a_value: float,
    b_value: float,
    better_direction: str,
    condition: str = "",
) -> None:
    delta = float(b_value - a_value)
    if better_direction == "lower":
        b_better, a_better = delta < 0, delta > 0
    elif better_direction == "higher":
        b_better, a_better = delta > 0, delta < 0
    else:
        b_better = a_better = False
    rows.append({
        "fold": int(fold), "target": target, "metric_family": family,
        "metric_name": metric, "condition": condition,
        "stage2a_value": float(a_value), "stage2b_value": float(b_value),
        "delta_b_minus_a": delta, "better_direction": better_direction,
        "stage2b_better": int(b_better), "stage2a_better": int(a_better),
        "tie": int(not b_better and not a_better and better_direction != "neutral"),
    })


def validate_alignment(stage2a: pd.DataFrame, stage2b: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    a = stage2a.sort_values(["ID", "fold"], kind="stable").reset_index(drop=True)
    b = stage2b.sort_values(["ID", "fold"], kind="stable").reset_index(drop=True)
    if len(a) != 500 or len(b) != 500 or a["ID"].nunique() != 500 or b["ID"].nunique() != 500:
        raise ValueError("Stage 2A and Stage 2B OOF must each contain 500 unique IDs")
    if not a[["ID", "fold"]].astype(str).equals(b[["ID", "fold"]].astype(str)):
        raise ValueError("Stage 2A and Stage 2B do not align on ID + fold")
    if not a["forehead_available"].astype(int).equals(b["forehead_available"].astype(int)):
        raise ValueError("Stage 2A and Stage 2B forehead availability differs")
    for target in ALL_TARGETS:
        if not np.allclose(
            a[f"raw_{target}"].to_numpy(float), b[f"raw_{target}"].to_numpy(float),
            rtol=0, atol=1e-12, equal_nan=True,
        ):
            raise ValueError(f"Raw target differs between Stage 2A and Stage 2B: {target}")
    return a, b


def compare_stage2a_stage2b(
    stage2a: pd.DataFrame,
    stage2b: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    a, b = validate_alignment(stage2a, stage2b)
    rows: list[dict[str, Any]] = []
    for fold in range(5):
        af = a.loc[a["fold"].astype(int).eq(fold)].reset_index(drop=True)
        bf = b.loc[b["fold"].astype(int).eq(fold)].reset_index(drop=True)
        for target in ALL_TARGETS:
            raw = bf[f"raw_{target}"]
            a_fit = fit_error_metrics(raw, af[f"predicted_acquisition_{target}"])
            b_fit = fit_error_metrics(raw, bf[f"predicted_condition_nn_{target}"])
            for metric, direction in (("mae", "lower"), ("rmse", "lower"), ("r2", "higher")):
                _append_metric(
                    rows, fold, target, "conditional_prediction", metric,
                    float(a_fit[metric]), float(b_fit[metric]), direction,
                )
            for condition in CONDITION_NAMES:
                _, a_rho = spearman_rho(af[f"calibrated_{target}"], af[condition])
                _, b_rho = spearman_rho(bf[f"calibrated_nn_{target}"], bf[condition])
                _append_metric(
                    rows, fold, target, "residual_exif", "absolute_spearman",
                    abs(a_rho), abs(b_rho), "lower", condition,
                )
            a_camera = camera_difference_metrics(af, f"calibrated_{target}")
            b_camera = camera_difference_metrics(bf, f"calibrated_nn_{target}")
            for source, metric in (
                ("mean_difference_honor_minus_xiaomi", "absolute_mean_difference"),
                ("median_difference_honor_minus_xiaomi", "absolute_median_difference"),
                ("standardized_mean_difference", "absolute_standardized_mean_difference"),
            ):
                _append_metric(
                    rows, fold, target, "camera_difference", metric,
                    abs(float(a_camera[source])), abs(float(b_camera[source])), "lower",
                )
            raw_values = raw.dropna().to_numpy(float)
            a_values = af[f"calibrated_{target}"].dropna().to_numpy(float)
            b_values = bf[f"calibrated_nn_{target}"].dropna().to_numpy(float)
            raw_variance = float(np.var(raw_values, ddof=0))
            a_retention = float(np.var(a_values, ddof=0) / raw_variance)
            b_retention = float(np.var(b_values, ddof=0) / raw_variance)
            _append_metric(
                rows, fold, target, "variance", "variance_retention",
                a_retention, b_retention, "neutral",
            )
    per_fold = pd.DataFrame(rows).sort_values(
        ["metric_family", "metric_name", "condition", "target", "fold"], kind="stable"
    ).reset_index(drop=True)
    summary_rows: list[dict[str, Any]] = []
    keys = ["target", "metric_family", "metric_name", "condition", "better_direction"]
    for key, group in per_fold.groupby(keys, sort=True, dropna=False):
        delta = group["delta_b_minus_a"].to_numpy(float)
        summary_rows.append({
            **dict(zip(keys, key)), "fold_valid_n": int(len(group)),
            "stage2b_better_fold_n": int(group["stage2b_better"].sum()),
            "stage2a_better_fold_n": int(group["stage2a_better"].sum()),
            "tie_fold_n": int(group["tie"].sum()),
            "mean_delta_b_minus_a": float(np.mean(delta)),
            "std_delta_b_minus_a": float(np.std(delta, ddof=0)),
            "median_delta_b_minus_a": float(np.median(delta)),
            "min_delta_b_minus_a": float(np.min(delta)),
            "max_delta_b_minus_a": float(np.max(delta)),
        })
    return per_fold, pd.DataFrame(summary_rows)


def camera_representation_comparison(stage2a: pd.DataFrame, stage2b: pd.DataFrame) -> pd.DataFrame:
    a, b = validate_alignment(stage2a, stage2b)
    rows: list[dict[str, Any]] = []
    for fold in range(5):
        af = a.loc[a["fold"].astype(int).eq(fold)]
        bf = b.loc[b["fold"].astype(int).eq(fold)]
        for target in ALL_TARGETS:
            for representation, frame, column in (
                ("raw", bf, f"raw_{target}"),
                ("stage2a_ridge_calibrated", af, f"calibrated_{target}"),
                ("stage2b_mlp_calibrated", bf, f"calibrated_nn_{target}"),
            ):
                rows.append({
                    "fold": fold, "target": target, "representation": representation,
                    **camera_difference_metrics(frame, column),
                })
    return pd.DataFrame(rows)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage2a-oof", type=Path, required=True)
    parser.add_argument("--stage2b-oof", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    a = pd.read_csv(args.stage2a_oof, dtype={"ID": str}, encoding="utf-8-sig")
    b = pd.read_csv(args.stage2b_oof, dtype={"ID": str}, encoding="utf-8-sig")
    per_fold, summary = compare_stage2a_stage2b(a, b)
    write_csv(per_fold, args.output_dir / "stage2a_vs_stage2b_per_fold.csv")
    write_csv(summary, args.output_dir / "stage2a_vs_stage2b_summary.csv")
    print(f"STAGE2A_VS_STAGE2B_COMPARISON=PASS ROWS={len(per_fold)} SUMMARY_ROWS={len(summary)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
