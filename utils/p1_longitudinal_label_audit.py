"""Longitudinal patient-group label audit for P1."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from utils.p1_rgb_audit import load_p1_frame


def _joined_unique(values: pd.Series) -> str:
    unique = pd.unique(values.astype(str))
    return ";".join(sorted(unique.tolist()))


def audit_longitudinal_labels(
    manifest: Path,
    fixed_split: Path,
    output_dir: Path,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Audit longitudinal variation within each patient group.

    Same-group NYHA / binary changes are expected in longitudinal data and are
    recorded rather than treated as errors.
    """

    frame = load_p1_frame(manifest, fixed_split).copy()
    return audit_longitudinal_labels_from_frame(frame, output_dir)


def audit_longitudinal_labels_from_frame(
    frame: pd.DataFrame,
    output_dir: Path,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Audit a pre-merged P1 frame.

    This lower-level helper is convenient for unit tests and synthetic checks.
    """

    output_dir.mkdir(parents=True, exist_ok=True)

    case_label_cols = ["label_original", "label_3class", "label_binary", "fold"]
    case_conflicts = []
    duplicate_case_ids = []
    for case_id, group in frame.groupby("case_id", sort=False):
        if len(group) > 1:
            duplicate_case_ids.append(str(case_id))
        unique_counts = group[case_label_cols].nunique(dropna=False)
        if (unique_counts > 1).any():
            case_conflicts.append(str(case_id))
    if case_conflicts:
        raise ValueError(f"conflicting labels for case_id values: {case_conflicts[:10]}")
    if duplicate_case_ids:
        # Duplicate case rows are not expected in the authoritative P1 frame.
        raise ValueError(f"duplicate case_id entries are not allowed: {duplicate_case_ids[:10]}")

    group_fold_nunique = frame.groupby("patient_group_id", sort=False)["fold"].nunique()
    if int(group_fold_nunique.max()) != 1:
        raise ValueError("patient_group_id crosses folds")

    rows: list[dict[str, Any]] = []
    grouped = frame.groupby("patient_group_id", sort=False)
    for patient_group_id, group in grouped:
        rows.append(
            {
                "patient_group_id": str(patient_group_id),
                "n_cases": int(len(group)),
                "case_ids": _joined_unique(group["case_id"]),
                "unique_NYHA": _joined_unique(group["label_original"]),
                "unique_label_3class": _joined_unique(group["label_3class"]),
                "unique_label_binary": _joined_unique(group["label_binary"]),
                "fold_values": _joined_unique(group["fold"]),
                "has_nyha_change": bool(group["label_original"].nunique() > 1),
                "has_three_class_change": bool(group["label_3class"].nunique() > 1),
                "has_binary_label_change": bool(group["label_binary"].nunique() > 1),
            }
        )

    audit = pd.DataFrame(rows)
    audit = audit.sort_values(["n_cases", "patient_group_id"], ascending=[False, True]).reset_index(drop=True)
    audit.to_csv(output_dir / "patient_group_longitudinal_label_audit.csv", index=False, encoding="utf-8-sig")

    multi_case = audit[audit.n_cases > 1]
    summary: dict[str, Any] = {
        "status": "available",
        "unique_patient_group_count": int(audit.patient_group_id.nunique()),
        "multi_case_group_count": int(len(multi_case)),
        "groups_with_nyha_change": int(audit.has_nyha_change.sum()),
        "groups_with_three_class_change": int(audit.has_three_class_change.sum()),
        "groups_with_binary_label_change": int(audit.has_binary_label_change.sum()),
        "cases_in_binary_conflict_groups": int(
            audit.loc[audit.has_binary_label_change, "n_cases"].sum()
        ),
        "same_patient_visits_may_have_different_labels": True,
        "patient_group_id_used_only_for_split_and_resampling": True,
    }
    (output_dir / "patient_group_longitudinal_label_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return audit, summary
