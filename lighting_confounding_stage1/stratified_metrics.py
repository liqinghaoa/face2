from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .features import fold_z_features
from .metrics_utils import cluster_bootstrap_metrics, compute_metrics


STRATIFY_VARS = {
    "brightness": "brightness_device_z",
    "iso": "log2_iso_device_z",
    "exposure_time": "log2_exposure_device_z",
}


def _assign_fold_quantile_strata(master: pd.DataFrame, var: str, q: tuple[float, float]) -> pd.Series:
    strata = pd.Series(index=master.index, dtype=object)
    for fold in sorted(master["fold"].astype(int).unique()):
        frame, _ = fold_z_features(master, int(fold))
        train = frame.loc[frame["fold"].astype(int) != int(fold), var].astype(float).dropna()
        test_idx = frame.index[frame["fold"].astype(int) == int(fold)]
        if len(train) < 3:
            strata.loc[test_idx] = "missing_threshold"
            continue
        low, high = train.quantile(list(q)).to_list()
        values = frame.loc[test_idx, var].astype(float)
        strata.loc[test_idx] = np.select(
            [values <= low, values > high, values.isna()],
            ["low", "high", "missing"],
            default="middle",
        )
    return strata


def _stratum_row(frame: pd.DataFrame, *, name: str, level: str, bootstrap_repeats: int, seed: int, unstable_total: int, unstable_per_class: int) -> dict[str, Any]:
    labels = frame["binary_label"].astype(int)
    base = {
        "stratification": name,
        "stratum": level,
        "n": int(len(frame)),
        "patient_group_n": int(frame["patient_group_id"].nunique()),
        "control_n": int((labels == 0).sum()),
        "patient_n": int((labels == 1).sum()),
        "unstable": bool(len(frame) < unstable_total or (labels == 0).sum() < unstable_per_class or (labels == 1).sum() < unstable_per_class),
        "auc_status": "available" if labels.nunique() == 2 else "single_class_auc_na",
    }
    metrics = compute_metrics(labels, frame["rgb_oof_probability_patient"].astype(float))
    base.update(metrics)
    if labels.nunique() == 2 and len(frame) > 1:
        boot_frame = frame.rename(columns={"rgb_oof_probability_patient": "probability_patient"})
        boot = cluster_bootstrap_metrics(boot_frame, repeats=bootstrap_repeats, seed=seed)
        for metric, ci in boot["ci95"].items():
            base[f"{metric}_ci95_low"] = ci[0]
            base[f"{metric}_ci95_high"] = ci[1]
        base["bootstrap_valid_iterations"] = boot["valid_iterations"]
    return base


def run_stratified_metrics(
    master: pd.DataFrame,
    out_dir: Path,
    *,
    bootstrap_repeats: int,
    seed: int,
    quantiles: tuple[float, float],
    unstable_total: int,
    unstable_per_class: int,
) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    work = master.copy()
    for name, var in STRATIFY_VARS.items():
        work[f"{name}_stratum"] = _assign_fold_quantile_strata(master, var, quantiles)

    rows = []
    for device, frame in work.groupby("camera_model", dropna=False):
        rows.append(_stratum_row(frame, name="camera_model", level=str(device), bootstrap_repeats=bootstrap_repeats, seed=seed, unstable_total=unstable_total, unstable_per_class=unstable_per_class))
    for name in STRATIFY_VARS:
        for level, frame in work.groupby(f"{name}_stratum", dropna=False):
            rows.append(_stratum_row(frame, name=name, level=str(level), bootstrap_repeats=bootstrap_repeats, seed=seed, unstable_total=unstable_total, unstable_per_class=unstable_per_class))

    metrics = pd.DataFrame(rows)
    metrics.to_csv(out_dir / "rgb_stratified_metrics.csv", index=False, encoding="utf-8-sig")
    worst: dict[str, Any] = {}
    for name, group in metrics.groupby("stratification"):
        valid = group.dropna(subset=["roc_auc"])
        if valid.empty:
            worst[name] = {"WorstGroupAUC": None, "DeltaAUC": None, "valid_strata": 0}
        else:
            worst[name] = {
                "WorstGroupAUC": float(valid["roc_auc"].min()),
                "DeltaAUC": float(valid["roc_auc"].max() - valid["roc_auc"].min()),
                "valid_strata": int(len(valid)),
                "worst_stratum": str(valid.loc[valid["roc_auc"].idxmin(), "stratum"]),
            }
    import json

    (out_dir / "rgb_worst_group_metrics.json").write_text(json.dumps(worst, ensure_ascii=False, indent=2), encoding="utf-8")
    return {"stratified": metrics, "worst": worst}
