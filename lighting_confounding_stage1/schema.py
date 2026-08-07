from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd


ALIASES: dict[str, tuple[str, ...]] = {
    "sample_id": ("sample_id", "case_id", "image_id", "ID", "id", "文件名", "相对路径", "绝对路径"),
    "patient_group_id": ("patient_group_id", "group_id", "patient_id", "患者ID"),
    "nyha": ("NYHA", "original_label", "nyha", "原始标签"),
    "fold": ("fold", "Fold", "fold_id"),
    "binary_label": ("binary_label", "label_binary", "label", "true_label"),
    "oof_probability_patient": ("prob_patient", "probability_patient", "patient_probability", "p_patient"),
    "oof_predicted_label": ("pred_class", "pred_binary", "predicted_label"),
    "camera_make": ("厂商", "camera_make", "make", "Make"),
    "camera_model": ("相机型号", "camera_model", "model", "Model"),
    "exposure_time_seconds": ("曝光时间(s)", "exposure_time_seconds", "ExposureTime", "exposure_time"),
    "iso": ("ISO", "iso", "ISOSpeedRatings"),
    "brightness_value_apex": ("亮度值(APEX)", "BrightnessValue", "brightness_value_apex"),
    "f_number": ("光圈F值", "FNumber", "f_number"),
    "exposure_bias_ev": ("曝光补偿(EV)", "ExposureBiasValue", "exposure_bias_ev"),
    "flash": ("闪光灯", "Flash", "flash"),
    "metering_mode": ("测光模式", "MeteringMode", "metering_mode"),
    "white_balance": ("白平衡", "WhiteBalance", "white_balance"),
    "exposure_mode": ("曝光模式", "ExposureMode", "exposure_mode"),
    "image_width": ("宽度(px)", "EXIF像素宽度", "ImageWidth", "image_width"),
    "image_height": ("高度(px)", "EXIF像素高度", "ImageLength", "image_height"),
    "capture_time": ("原始拍摄时间", "DateTimeOriginal", "capture_time", "拍摄时间", "文件内修改时间"),
    "software": ("软件", "Software", "software"),
}

REQUIRED_SPLIT = ("sample_id", "patient_group_id", "nyha", "fold", "binary_label")
REQUIRED_OOF = ("sample_id", "patient_group_id", "binary_label", "oof_probability_patient", "oof_predicted_label")
REQUIRED_METADATA = ("sample_id", "camera_model")


@dataclass(frozen=True)
class Detection:
    mapping: dict[str, str | None]
    candidates: dict[str, list[str]]
    ambiguous: dict[str, list[str]]
    columns: list[str]

    def to_dict(self) -> dict[str, Any]:
        return {
            "mapping": self.mapping,
            "candidates": self.candidates,
            "ambiguous": self.ambiguous,
            "columns": self.columns,
        }


def detect_schema(columns: list[str], semantic_fields: tuple[str, ...] | list[str]) -> Detection:
    mapping: dict[str, str | None] = {}
    candidates: dict[str, list[str]] = {}
    ambiguous: dict[str, list[str]] = {}
    normalized_lookup = {c.lower().replace(" ", "").replace("_", ""): c for c in columns}

    for semantic in semantic_fields:
        hits: list[str] = []
        for alias in ALIASES.get(semantic, (semantic,)):
            if alias in columns and alias not in hits:
                hits.append(alias)
            key = alias.lower().replace(" ", "").replace("_", "")
            if key in normalized_lookup and normalized_lookup[key] not in hits:
                hits.append(normalized_lookup[key])
        candidates[semantic] = hits
        if len(hits) >= 1:
            mapping[semantic] = hits[0]
        else:
            mapping[semantic] = None
    return Detection(mapping=mapping, candidates=candidates, ambiguous=ambiguous, columns=columns)


def choose_metadata_sheet(xlsx_path, requested: str = "auto") -> tuple[str, dict[str, Any]]:
    xl = pd.ExcelFile(xlsx_path)
    summary: dict[str, Any] = {"sheets": []}
    best_name = ""
    best_score = -1
    for sheet in xl.sheet_names:
        preview = pd.read_excel(xlsx_path, sheet_name=sheet, nrows=20)
        cols = [str(c) for c in preview.columns]
        score = 0
        for semantic in ("sample_id", "camera_model", "exposure_time_seconds", "iso", "brightness_value_apex"):
            score += int(bool(detect_schema(cols, [semantic]).candidates[semantic]))
        summary["sheets"].append({"sheet_name": sheet, "columns": cols, "score": score})
        if score > best_score:
            best_score = score
            best_name = sheet
    if requested != "auto":
        if requested not in xl.sheet_names:
            raise ValueError(f"metadata sheet {requested!r} not found; available={xl.sheet_names}")
        return requested, summary
    if not best_name:
        raise ValueError("could not identify a metadata sheet")
    return best_name, summary


def ensure_unambiguous(name: str, detection: Detection, required: tuple[str, ...]) -> None:
    missing = [field for field in required if detection.mapping.get(field) is None and field not in detection.ambiguous]
    if detection.ambiguous or missing:
        raise ValueError(
            f"{name} schema is not usable; ambiguous={detection.ambiguous}, missing={missing}, candidates={detection.candidates}"
        )
