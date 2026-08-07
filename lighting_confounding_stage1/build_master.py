from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .config import Stage1Config, json_safe
from .features import STANDARD_COLUMNS, add_oof_device_z, derive_metadata_features, feature_inventory
from .id_normalization import normalize_id
from .schema import (
    ALIASES,
    REQUIRED_METADATA,
    REQUIRED_OOF,
    REQUIRED_SPLIT,
    choose_metadata_sheet,
    detect_schema,
    ensure_unambiguous,
)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(json_safe(payload), ensure_ascii=False, indent=2), encoding="utf-8")


def ensure_output_dirs(output_dir: Path) -> dict[str, Path]:
    names = ("preflight", "metadata", "association", "metadata_only", "rgb_dependence", "stratified", "figures", "reports", "logs")
    dirs = {name: output_dir / name for name in names}
    for path in dirs.values():
        path.mkdir(parents=True, exist_ok=True)
    return dirs


def load_inputs(config: Stage1Config) -> dict[str, Any]:
    split = pd.read_csv(config.split_csv, dtype=str)
    oof = pd.read_csv(config.oof_predictions_csv, dtype=str)
    sheet, workbook_summary = choose_metadata_sheet(config.metadata_xlsx, config.metadata_sheet)
    metadata = pd.read_excel(config.metadata_xlsx, sheet_name=sheet, dtype=object)
    return {"split": split, "oof": oof, "metadata": metadata, "metadata_sheet": sheet, "workbook_summary": workbook_summary}


def _standardize_metadata_columns(metadata: pd.DataFrame, detection) -> pd.DataFrame:
    out = metadata.copy()
    for std, col in detection.mapping.items():
        if col is not None and std in STANDARD_COLUMNS:
            out[std] = out[col]
    for col in metadata.columns:
        out[f"metadata__{col}"] = metadata[col]
    return out


def build_master(config: Stage1Config) -> tuple[pd.DataFrame, dict[str, Any]]:
    dirs = ensure_output_dirs(config.output_dir)
    inputs = load_inputs(config)
    split = inputs["split"]
    oof = inputs["oof"]
    metadata = inputs["metadata"]

    split_detection = detect_schema(list(split.columns), list(ALIASES))
    metadata_detection = detect_schema(list(metadata.columns), list(ALIASES))
    oof_detection = detect_schema(list(oof.columns), list(ALIASES))
    ensure_unambiguous("split", split_detection, REQUIRED_SPLIT)
    ensure_unambiguous("metadata", metadata_detection, REQUIRED_METADATA)
    ensure_unambiguous("oof", oof_detection, REQUIRED_OOF)

    write_json(dirs["metadata"] / "detected_split_schema.json", split_detection.to_dict())
    write_json(dirs["metadata"] / "detected_metadata_schema.json", {**metadata_detection.to_dict(), "selected_sheet": inputs["metadata_sheet"], "workbook": inputs["workbook_summary"]})
    write_json(dirs["metadata"] / "detected_oof_schema.json", oof_detection.to_dict())

    s_map = split_detection.mapping
    m_map = metadata_detection.mapping
    o_map = oof_detection.mapping

    split_std = pd.DataFrame(
        {
            "sample_id": split[s_map["sample_id"]].map(normalize_id),
            "patient_group_id": split[s_map["patient_group_id"]].map(normalize_id),
            "original_nyha": pd.to_numeric(split[s_map["nyha"]], errors="raise").astype(int),
            "fold": pd.to_numeric(split[s_map["fold"]], errors="raise").astype(int),
        }
    )
    split_std["binary_label"] = (split_std["original_nyha"] >= 1).astype(int)
    if s_map.get("binary_label"):
        existing = pd.to_numeric(split[s_map["binary_label"]], errors="raise").astype(int)
        if not (existing.to_numpy() == split_std["binary_label"].to_numpy()).all():
            raise ValueError("split binary_label does not match fixed NYHA 0 vs 1-4 conversion")
    split_keep = split.copy()
    for col in split_keep.columns:
        split_std[f"split__{col}"] = split_keep[col]

    meta_std = _standardize_metadata_columns(metadata, metadata_detection)
    meta_std["sample_id"] = meta_std[m_map["sample_id"]].map(normalize_id)
    excluded = meta_std[~meta_std["sample_id"].isin(set(split_std["sample_id"]))].copy()
    excluded.to_csv(dirs["metadata"] / "excluded_metadata_rows.csv", index=False, encoding="utf-8-sig")
    meta_500 = meta_std[meta_std["sample_id"].isin(set(split_std["sample_id"]))].copy()

    oof_std = pd.DataFrame(
        {
            "sample_id": oof[o_map["sample_id"]].map(normalize_id),
            "oof_patient_group_id": oof[o_map["patient_group_id"]].map(normalize_id),
            "oof_binary_label": pd.to_numeric(oof[o_map["binary_label"]], errors="raise").astype(int),
            "rgb_oof_probability_patient": pd.to_numeric(oof[o_map["oof_probability_patient"]], errors="raise").astype(float),
            "rgb_oof_predicted_label": pd.to_numeric(oof[o_map["oof_predicted_label"]], errors="raise").astype(int),
        }
    )
    if o_map.get("fold"):
        oof_std["oof_fold"] = pd.to_numeric(oof[o_map["fold"]], errors="raise").astype(int)
    for col in oof.columns:
        oof_std[f"oof__{col}"] = oof[col]

    gates: dict[str, Any] = {}
    gates["fixed_cohort_rows"] = int(len(split_std))
    gates["fixed_cohort_unique_sample_id"] = int(split_std["sample_id"].nunique())
    gates["metadata_rows_for_500"] = int(len(meta_500))
    gates["metadata_unique_sample_id_for_500"] = int(meta_500["sample_id"].nunique())
    gates["oof_rows"] = int(len(oof_std))
    gates["oof_unique_sample_id"] = int(oof_std["sample_id"].nunique())
    gates["duplicate_metadata_ids"] = sorted(meta_500.loc[meta_500["sample_id"].duplicated(), "sample_id"].unique().tolist())
    gates["duplicate_oof_ids"] = sorted(oof_std.loc[oof_std["sample_id"].duplicated(), "sample_id"].unique().tolist())

    if len(split_std) != 500 or split_std["sample_id"].nunique() != 500:
        raise ValueError(f"fixed cohort gate failed: rows={len(split_std)}, unique={split_std['sample_id'].nunique()}")
    if gates["metadata_rows_for_500"] != 500 or gates["metadata_unique_sample_id_for_500"] != 500 or gates["duplicate_metadata_ids"]:
        raise ValueError(f"metadata 500/500 one-to-one gate failed: {gates}")
    if gates["oof_rows"] != 500 or gates["oof_unique_sample_id"] != 500 or gates["duplicate_oof_ids"]:
        raise ValueError(f"OOF 500/500 one-to-one gate failed: {gates}")

    master = split_std.merge(meta_500, on="sample_id", how="left", validate="one_to_one")
    master = master.merge(oof_std, on="sample_id", how="left", validate="one_to_one")
    gates["metadata_matched_500"] = int(master["camera_model"].notna().sum()) if "camera_model" in master else int(master[m_map["sample_id"]].notna().sum())
    gates["oof_matched_500"] = int(master["rgb_oof_probability_patient"].notna().sum())
    gates["label_conflicts"] = int((master["binary_label"].astype(int) != master["oof_binary_label"].astype(int)).sum())
    gates["patient_group_conflicts"] = int((master["patient_group_id"].astype(str) != master["oof_patient_group_id"].astype(str)).sum())
    gates["fold_conflicts"] = 0
    if "oof_fold" in master.columns:
        gates["fold_conflicts"] = int((master["fold"].astype(int) != master["oof_fold"].astype(int)).sum())
    gates["oof_fold_source"] = "oof_predictions.csv" if "oof_fold" in master.columns else "fixed_split_csv"
    gates["probability_min"] = float(master["rgb_oof_probability_patient"].min())
    gates["probability_max"] = float(master["rgb_oof_probability_patient"].max())
    gates["patient_group_cross_fold_count"] = int(master.groupby("patient_group_id")["fold"].nunique().gt(1).sum())

    hard_fail = [
        gates["metadata_matched_500"] != 500,
        gates["oof_matched_500"] != 500,
        gates["label_conflicts"] != 0,
        gates["patient_group_conflicts"] != 0,
        gates["fold_conflicts"] != 0,
        not (0 <= gates["probability_min"] <= gates["probability_max"] <= 1),
        gates["patient_group_cross_fold_count"] != 0,
    ]
    if any(hard_fail):
        write_json(dirs["preflight"] / "preflight_summary.json", {"gates": gates, "status": "failed"})
        raise ValueError(f"preflight merge gates failed: {gates}")

    master, invalid_counts = derive_metadata_features(master)
    master, z_diagnostics = add_oof_device_z(master)
    if any(v for v in z_diagnostics.get("unknown_test_devices", {}).values()):
        raise ValueError(f"device z gate failed: {z_diagnostics}")

    p = master["rgb_oof_probability_patient"].clip(1e-6, 1 - 1e-6)
    master["rgb_oof_correct"] = (master["rgb_oof_predicted_label"].astype(int) == master["binary_label"].astype(int)).astype(int)
    master["rgb_oof_error"] = 1 - master["rgb_oof_correct"]
    master["rgb_oof_residual"] = master["rgb_oof_probability_patient"] - master["binary_label"].astype(int)
    master["rgb_oof_logit"] = np.log(p / (1 - p))
    master = master.sort_values(["fold", "sample_id"]).reset_index(drop=True)
    master.to_csv(dirs["metadata"] / "stage1_master_500.csv", index=False, encoding="utf-8-sig")
    feature_inventory(master).to_csv(dirs["metadata"] / "metadata_feature_inventory.csv", index=False, encoding="utf-8-sig")

    summary = {
        "status": "passed",
        "input_files": {
            "split_csv": str(config.split_csv),
            "metadata_xlsx": str(config.metadata_xlsx),
            "metadata_sheet": inputs["metadata_sheet"],
            "oof_predictions_csv": str(config.oof_predictions_csv),
        },
        "gates": gates,
        "invalid_numeric_counts": invalid_counts,
        "device_z_diagnostics": z_diagnostics,
        "detected_columns": {
            "split": split_detection.mapping,
            "metadata": metadata_detection.mapping,
            "oof": oof_detection.mapping,
        },
        "cohort_counts": {
            "binary_label": master["binary_label"].value_counts().sort_index().to_dict(),
            "fold": master["fold"].value_counts().sort_index().to_dict(),
            "camera_model": master["camera_model"].value_counts(dropna=False).to_dict(),
        },
    }
    write_json(dirs["preflight"] / "preflight_summary.json", summary)
    report = [
        "# Preflight Report",
        "",
        f"- status: {summary['status']}",
        f"- split_csv: `{config.split_csv}`",
        f"- metadata_xlsx: `{config.metadata_xlsx}`",
        f"- selected_metadata_sheet: `{inputs['metadata_sheet']}`",
        f"- oof_predictions_csv: `{config.oof_predictions_csv}`",
        f"- fixed_cohort_rows: {gates['fixed_cohort_rows']}",
        f"- metadata_matched_500: {gates['metadata_matched_500']}",
        f"- oof_matched_500: {gates['oof_matched_500']}",
        f"- label_conflicts: {gates['label_conflicts']}",
        f"- fold_conflicts: {gates['fold_conflicts']}",
        f"- patient_group_cross_fold_count: {gates['patient_group_cross_fold_count']}",
        "",
        "## Detected Columns",
        "",
        "```json",
        json.dumps(json_safe(summary["detected_columns"]), ensure_ascii=False, indent=2),
        "```",
    ]
    (dirs["preflight"] / "preflight_report.md").write_text("\n".join(report), encoding="utf-8")
    return master, summary
