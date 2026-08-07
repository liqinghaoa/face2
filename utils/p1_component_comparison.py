"""Paired comparison helpers for P1 component experiments against P1-RGB."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

import pandas as pd

from utils.p1_cluster_bootstrap import (
    paired_patient_cluster_visit_bootstrap,
    patient_cluster_bootstrap,
)
from utils.p1_rgb_audit import load_p1_frame


def load_p1_rgb_case_predictions(path: str | Path) -> pd.DataFrame:
    frame = pd.read_csv(path, dtype={"case_id": str, "patient_group_id": str})
    required = {
        "case_id",
        "patient_group_id",
        "fold",
        "label_original",
        "label_3class",
        "label_binary",
        "prob_control",
        "prob_patient",
        "pred_binary",
    }
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"P1-RGB case predictions missing columns: {sorted(missing)}")
    return frame


def compare_component_to_p1_rgb(
    component_case_frame: pd.DataFrame,
    p1_rgb_case_predictions: pd.DataFrame,
    *,
    iterations: int = 2000,
    seed: int = 2026,
) -> dict[str, Any]:
    component = component_case_frame.copy()
    baseline = p1_rgb_case_predictions.copy()
    component["case_id"] = component["case_id"].astype(str)
    baseline["case_id"] = baseline["case_id"].astype(str)
    merged = component.merge(
        baseline[["case_id", "patient_group_id", "prob_control", "prob_patient", "pred_binary"]],
        on="case_id",
        how="inner",
        suffixes=("", "_p1_rgb"),
    )
    if len(merged) != len(component) or merged["case_id"].nunique() != len(component):
        raise ValueError("P1-RGB case predictions must cover the same 500 case IDs")
    paired = paired_patient_cluster_visit_bootstrap(
        component_case_frame=component,
        historical_frame=baseline,
        iterations=iterations,
        seed=seed,
    )
    return {
        "status": "available",
        "pair_count": int(len(merged)),
        "unique_patient_groups": int(merged["patient_group_id"].astype(str).nunique()),
        "paired_bootstrap": paired,
        "p1_rgb_case_count": int(len(baseline)),
    }


def summarize_component_vs_p1_rgb(
    component_case_frame: pd.DataFrame,
    p1_rgb_case_predictions: pd.DataFrame,
    *,
    iterations: int = 2000,
    seed: int = 2026,
) -> dict[str, Any]:
    return compare_component_to_p1_rgb(
        component_case_frame,
        p1_rgb_case_predictions,
        iterations=iterations,
        seed=seed,
    )


def write_component_comparison_summary(
    output_path: str | Path,
    component_case_frame: pd.DataFrame,
    p1_rgb_case_predictions: pd.DataFrame,
    *,
    iterations: int = 2000,
    seed: int = 2026,
) -> dict[str, Any]:
    payload = compare_component_to_p1_rgb(
        component_case_frame,
        p1_rgb_case_predictions,
        iterations=iterations,
        seed=seed,
    )
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return payload

