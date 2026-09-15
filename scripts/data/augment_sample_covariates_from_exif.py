"""Append admission-time-aligned Height, Weight, and Age to image samples.

Each row in the target CSV represents an image sample.  A numeric suffix in an
ID (for example, ``100037382-1``) identifies another image of patient
``100037382``; it is not a separate hospital patient ID.  Covariates are
therefore selected from the patient's admission nearest to that image's EXIF
``DateTimeOriginal`` timestamp.
"""

from __future__ import annotations

import csv
import re
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

from openpyxl import load_workbook
from PIL import ExifTags, Image


ROOT = Path(__file__).resolve().parents[2]
SAMPLES_PATH = ROOT / "data/processed/global_face_R3DPR/nyha_2class_sex_stratified_group_5fold.csv"
IMAGE_DIR = ROOT / "data/processed/P0_Physics_Audit_v1/images/raw_scene"
COVARIATE_WORKBOOK = ROOT / "data/processed/clinical_covariate_selection.xlsx"
OUTPUT_PATH = ROOT / "data/processed/global_face_R3DPR/nyha_2class_sex_stratified_group_5fold_clinical_covariates.csv"
AUDIT_PATH = ROOT / "data/processed/global_face_R3DPR/nyha_2class_sex_stratified_group_5fold_clinical_covariates_audit.csv"

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp", ".webp", ".gif"}
OUTPUT_COLUMNS = [
    "Height_cm",
    "Weight_kg",
    "Age_y",
    "EXIF_DateTimeOriginal",
    "matched_admission_time",
    "admission_source_row",
    "admission_SEX",
    "admission_NYHA",
    "matching_gap_days",
    "Height_source",
    "clinical_match_status",
]


def base_patient_id(sample_id: str) -> str:
    """Strip only a terminal numeric image suffix, such as ``-1``."""
    return re.sub(r"-\d+$", "", sample_id)


def parse_datetime(value: object) -> datetime | None:
    if not value:
        return None
    try:
        return datetime.strptime(str(value).strip().replace(":", "-", 2), "%Y-%m-%d %H:%M:%S")
    except ValueError:
        return None


def read_exif_datetime(path: Path) -> datetime | None:
    with Image.open(path) as image:
        exif = image.getexif()
        exif_ifd = exif.get_ifd(getattr(ExifTags.IFD, "Exif", 34665))
        return parse_datetime(exif_ifd.get(36867) or exif.get(36867))


def main() -> None:
    with SAMPLES_PATH.open(encoding="utf-8-sig", newline="") as handle:
        samples = list(csv.DictReader(handle))
        source_columns = list(samples[0]) if samples else []
    if not samples or "ID" not in source_columns:
        raise ValueError("Target CSV must contain at least one row and an ID column.")

    image_paths = {
        path.stem: path
        for path in IMAGE_DIR.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    }
    if len(image_paths) != len([path for path in IMAGE_DIR.iterdir() if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES]):
        raise ValueError("Image directory has duplicate filename stems.")

    worksheet = load_workbook(COVARIATE_WORKBOOK, read_only=True, data_only=True)["住院明细_1068"]
    headers = [cell.value for cell in next(worksheet.iter_rows())]
    required = {"ID", "原表行号", "住院时间", "SEX", "NYHA", "Height_cm", "Weight_kg", "Age_y"}
    if missing := required.difference(headers):
        raise ValueError(f"Admission worksheet lacks columns: {sorted(missing)}")
    admissions: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in worksheet.iter_rows(min_row=2, values_only=True):
        record = dict(zip(headers, row))
        admissions[str(record["ID"])].append(record)

    output_rows: list[dict[str, object]] = []
    audit_rows: list[dict[str, object]] = []
    status_counts: dict[str, int] = defaultdict(int)

    for sample in samples:
        sample_id = sample["ID"].strip()
        patient_id = base_patient_id(sample_id)
        image_path = image_paths.get(sample_id)
        if image_path is None:
            raise FileNotFoundError(f"No source image found for sample {sample_id!r}")
        photo_time = read_exif_datetime(image_path)
        patient_admissions = admissions.get(patient_id, [])
        if not photo_time or not patient_admissions:
            raise ValueError(f"Cannot time-match sample {sample_id!r}: missing EXIF time or admissions.")

        gaps = [abs((record["住院时间"] - photo_time).total_seconds()) / 86400 for record in patient_admissions]
        minimum_gap = min(gaps)
        nearest = [record for record, gap in zip(patient_admissions, gaps) if gap == minimum_gap]
        if len(nearest) != 1:
            raise ValueError(f"Ambiguous nearest admission for sample {sample_id!r}")
        selected = nearest[0]

        height = selected["Height_cm"]
        height_source = "matched_admission"
        if height is None:
            patient_heights = {record["Height_cm"] for record in patient_admissions if record["Height_cm"] is not None}
            if len(patient_heights) == 1:
                height = patient_heights.pop()
                height_source = "same_patient_consistent_height"
            else:
                height_source = "missing"

        sex_matches = str(selected["SEX"]) == sample["SEX"].strip()
        nyha_matches = str(selected["NYHA"]) == sample["NYHA"].strip()
        if sex_matches and nyha_matches:
            status = "matched_by_patient_id_and_exif_time"
        elif sex_matches:
            status = "matched_by_patient_id_and_exif_time__nyha_mismatch"
        else:
            status = "matched_by_patient_id_and_exif_time__sex_and_nyha_mismatch"

        result = dict(sample)
        result.update(
            {
                "Height_cm": height,
                "Weight_kg": selected["Weight_kg"],
                "Age_y": selected["Age_y"],
                "EXIF_DateTimeOriginal": photo_time.strftime("%Y-%m-%d %H:%M:%S"),
                "matched_admission_time": selected["住院时间"].strftime("%Y-%m-%d %H:%M:%S"),
                "admission_source_row": selected["原表行号"],
                "admission_SEX": selected["SEX"],
                "admission_NYHA": selected["NYHA"],
                "matching_gap_days": round(minimum_gap, 2),
                "Height_source": height_source,
                "clinical_match_status": status,
            }
        )
        output_rows.append(result)
        status_counts[status] += 1
        if "mismatch" in status or minimum_gap > 30:
            audit_rows.append(result)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    output_columns = source_columns + [column for column in OUTPUT_COLUMNS if column not in source_columns]
    with OUTPUT_PATH.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=output_columns, extrasaction="raise")
        writer.writeheader()
        writer.writerows(output_rows)
    with AUDIT_PATH.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=output_columns, extrasaction="raise")
        writer.writeheader()
        writer.writerows(audit_rows)

    print(f"Wrote {OUTPUT_PATH}")
    print(f"Wrote {AUDIT_PATH}")
    print(f"Samples: {len(output_rows)}; audit rows: {len(audit_rows)}; statuses: {dict(status_counts)}")


if __name__ == "__main__":
    main()
