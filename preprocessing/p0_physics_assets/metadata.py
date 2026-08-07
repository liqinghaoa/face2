"""Split/EXIF preflight and metadata normalization with no image-pixel logic."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from .types import P0Config, SampleRecord

EXIF_COLUMNS = {"ID": "ID", "文件名": "filename", "SHA256": "source_sha256", "图像格式": "image_format", "宽度(px)": "width_px", "高度(px)": "height_px", "色彩模式": "color_mode", "通道数": "channels", "位深(每通道)": "bit_depth", "EXIF存在": "exif_present", "方向值": "orientation_value", "方向解释": "orientation_description", "厂商": "camera_make", "相机型号": "camera_model", "原始拍摄时间": "datetime_original", "曝光时间(s)": "exposure_time_s", "光圈F值": "f_number", "ISO": "iso", "亮度值(APEX)": "brightness_value", "焦距(mm)": "focal_length_mm", "白平衡": "white_balance", "闪光灯": "flash", "提取状态": "extraction_status"}


def validate_split(split: pd.DataFrame) -> pd.DataFrame:
    """Validate immutable P0 split composition and add mandated binary labels."""
    required = {"ID", "patient_group_id", "SEX", "sex_name", "NYHA", "label_3class", "label_3class_name", "fold"}
    missing = required - set(split.columns)
    if missing: raise ValueError(f"split missing columns: {sorted(missing)}")
    frame = split.copy(); frame["ID"] = frame.ID.astype(str).str.strip(); frame["patient_group_id"] = frame.patient_group_id.astype(str).str.strip()
    if len(frame) != 500 or frame.ID.nunique() != 500: raise ValueError("P0 split must contain exactly 500 unique IDs")
    if frame.groupby("patient_group_id").fold.nunique().gt(1).any(): raise ValueError("patient_group_id crosses folds")
    if frame.patient_group_id.nunique() != 483: raise ValueError("P0 split must contain exactly 483 patient groups")
    if set(pd.to_numeric(frame.fold)) != set(range(5)) or not (frame.groupby("fold").size() == 100).all(): raise ValueError("split folds must be 0..4 with 100 samples each")
    counts = frame.label_3class.value_counts().to_dict()
    if {int(k): int(v) for k, v in counts.items()} != {0:115, 1:237, 2:148}: raise ValueError("unexpected label_3class distribution")
    frame["binary_label"] = (pd.to_numeric(frame.label_3class) != 0).astype(int); frame["binary_name"] = frame.binary_label.map({0:"Control",1:"Patient"})
    if frame.binary_label.value_counts().to_dict() != {1:385,0:115}: raise ValueError("unexpected binary distribution")
    if frame.groupby("patient_group_id").binary_label.nunique().gt(1).any(): raise ValueError("patient_group_id has inconsistent binary labels")
    return frame


def load_exif_500(config: P0Config, split: pd.DataFrame) -> pd.DataFrame:
    """Read the source sheet and retain only requested IDs with known columns."""
    source = pd.read_excel(config.inputs.exif_workbook, sheet_name=config.inputs.exif_sheet_name, dtype={"ID": str})
    if "ID" not in source: raise ValueError("EXIF sheet has no ID column")
    source = source.copy(); source.ID = source.ID.astype(str).str.strip()
    if source.ID.duplicated().any(): raise ValueError("EXIF IDs must be unique")
    subset = source[source.ID.isin(split.ID)].copy()
    if len(subset) != 500 or subset.ID.nunique() != 500 or set(subset.ID) != set(split.ID): raise ValueError("EXIF must match split 500/500")
    output = subset[[name for name in EXIF_COLUMNS if name in subset.columns]].rename(columns=EXIF_COLUMNS)
    return output.merge(split[["ID"]], on="ID", how="right", validate="one_to_one")


def build_sample_records(config: P0Config, split: pd.DataFrame, exif: pd.DataFrame, assets: dict[str, dict[str, Path]]) -> list[SampleRecord]:
    """Join non-pixel metadata and verified asset paths by immutable ID."""
    merged = split.merge(exif, on="ID", how="left", validate="one_to_one")
    records: list[SampleRecord] = []
    for row in merged.to_dict("records"):
        image_id = str(row["ID"])
        records.append(SampleRecord(image_id, str(row["patient_group_id"]), int(row["SEX"]), str(row["sex_name"]), int(row["NYHA"]), int(row["label_3class"]), str(row["label_3class_name"]), int(row["fold"]), assets["raw"][image_id], assets["blackbg"][image_id], assets["meanbg"][image_id], row))
    return records
