"""Build an auditable, image-date-aligned clinical covariate table.

The source workbook has one row per admission.  The image cohort has one row
per patient, so weight and age must be taken from the admission nearest to the
image capture date rather than from an arbitrary admission.
"""

from __future__ import annotations

import csv
import re
from collections import defaultdict
from datetime import datetime
from pathlib import Path

from openpyxl import Workbook, load_workbook
from openpyxl.styles import Alignment, Font, PatternFill
from openpyxl.utils import get_column_letter
from PIL import ExifTags, Image


ROOT = Path(__file__).resolve().parents[2]
BASELINE_PATH = ROOT / "data/raw/table/心功能患者数据-基础信息.xlsx"
EXIF_PATH = ROOT / "data/raw/EXIF/Image_Metadata_All.xlsx"
RAW_SCENE_DIR = ROOT / "data/processed/P0_Physics_Audit_v1/images/raw_scene"
COHORT_PATH = ROOT / "data/clinalData/心功能患者数据.csv"
OUTPUT_PATH = ROOT / "data/processed/clinical_covariate_selection.xlsx"


def numeric_value(value: object) -> float | None:
    if value is None:
        return None
    match = re.search(r"[0-9]+(?:[.][0-9]+)?", str(value))
    return float(match.group()) if match else None


def parse_exif_datetime(value: object) -> datetime | None:
    if not value:
        return None
    text = str(value).strip()
    try:
        return datetime.strptime(text.replace(":", "-", 2), "%Y-%m-%d %H:%M:%S")
    except ValueError:
        return None


def extract_raw_scene_dates(image_dir: Path) -> dict[str, datetime]:
    """Read DateTimeOriginal from the alternate raw-scene image set.

    These files use the ``ID-1.jpg`` naming convention and are not represented
    in the 522-row Image_Metadata_All.xlsx workbook.
    """
    dates: dict[str, datetime] = {}
    if not image_dir.exists():
        return dates
    for path in image_dir.glob("*.jpg"):
        patient_id = path.stem.removesuffix("-1")
        try:
            with Image.open(path) as image:
                exif = image.getexif()
                exif_ifd = exif.get_ifd(getattr(ExifTags.IFD, "Exif", 34665))
                captured_at = parse_exif_datetime(exif_ifd.get(36867))
            if captured_at:
                dates[patient_id] = captured_at
        except Exception:
            continue
    return dates


def write_sheet(ws, headers: list[str], rows: list[list[object]]) -> None:
    ws.append(headers)
    for row in rows:
        ws.append(row)
    for cell in ws[1]:
        cell.font = Font(bold=True, color="FFFFFF")
        cell.fill = PatternFill("solid", fgColor="1F4E78")
        cell.alignment = Alignment(horizontal="center", vertical="center", wrap_text=True)
    ws.freeze_panes = "A2"
    ws.auto_filter.ref = ws.dimensions
    for index, header in enumerate(headers, start=1):
        width = max(len(header) + 2, 12)
        for row in rows[:200]:
            width = max(width, min(len(str(row[index - 1] or "")) + 2, 28))
        ws.column_dimensions[get_column_letter(index)].width = width
    for row in ws.iter_rows(min_row=2):
        for cell in row:
            cell.alignment = Alignment(vertical="top")


def main() -> None:
    baseline_ws = load_workbook(BASELINE_PATH, read_only=True, data_only=True).worksheets[0]
    admissions: dict[str, list[dict[str, object]]] = defaultdict(list)
    for source_row, row in enumerate(baseline_ws.iter_rows(min_row=2, values_only=True), start=2):
        patient_id = str(row[1]).strip()
        admissions[patient_id].append(
            {
                "source_row": source_row,
                "id": patient_id,
                "admission_number": row[2],
                "admission_time": row[5],
                "sex": str(row[6]).strip(),
                "nyha": str(row[7]).strip(),
                "height_cm": numeric_value(row[8]),
                "weight_kg": numeric_value(row[9]),
                "age_y": numeric_value(row[10]),
            }
        )

    exif_ws = load_workbook(EXIF_PATH, read_only=True, data_only=True).worksheets[1]
    exif_headers = next(exif_ws.iter_rows(values_only=True))
    original_time_index = exif_headers.index("原始拍摄时间")
    image_metadata_ids: set[str] = set()
    image_dates: dict[str, datetime] = {}
    for row in exif_ws.iter_rows(min_row=2, values_only=True):
        patient_id = str(row[1]).strip()
        image_metadata_ids.add(patient_id)
        captured_at = parse_exif_datetime(row[original_time_index])
        if captured_at:
            image_dates[patient_id] = captured_at
    # Supplement the 522-image workbook with the alternate 500-image raw-scene
    # set. Existing workbook dates remain authoritative if an ID overlaps.
    raw_scene_dates = extract_raw_scene_dates(RAW_SCENE_DIR)
    for patient_id, captured_at in raw_scene_dates.items():
        image_metadata_ids.add(patient_id)
        image_dates.setdefault(patient_id, captured_at)

    with COHORT_PATH.open(encoding="utf-8-sig", newline="") as file:
        cohort = list(csv.DictReader(file))

    automatic_rows: list[list[object]] = []
    review_rows: list[list[object]] = []
    audit_rows: list[list[object]] = []
    selected_rows: dict[str, int] = {}

    for patient in cohort:
        patient_id = patient["ID"].strip()
        sex = patient["SEX"].strip()
        nyha = patient["NYHA"].strip()
        patient_admissions = admissions[patient_id]
        candidates = [item for item in patient_admissions if item["sex"] == sex and item["nyha"] == nyha]
        photo_time = image_dates.get(patient_id)
        height_values = {item["height_cm"] for item in patient_admissions if item["height_cm"] is not None}
        stable_height = next(iter(height_values), None) if len(height_values) == 1 else None

        selected = None
        gap_days = None
        if photo_time and candidates:
            distances = [abs((item["admission_time"] - photo_time).total_seconds()) / 86400 for item in candidates]
            smallest_gap = min(distances)
            nearest = [item for item, distance in zip(candidates, distances) if distance == smallest_gap]
            if len(nearest) == 1:
                selected = nearest[0]
                gap_days = smallest_gap
                selected_rows[patient_id] = int(selected["source_row"])

        height_at_selected = selected["height_cm"] if selected else None
        height = height_at_selected if height_at_selected is not None else stable_height
        height_source = (
            "匹配住院记录"
            if height_at_selected is not None
            else "同一患者其他住院记录（身高一致）"
            if height is not None
            else "缺失"
        )
        status = "自动匹配" if selected else "待人工核验"
        if gap_days is not None and gap_days <= 30:
            quality = "通过（间隔 <=30 天）"
        elif gap_days is not None:
            quality = "复核（间隔 >30 天）"
        elif patient_id not in image_metadata_ids:
            quality = "无法匹配：EXIF 元数据表无此 ID"
        elif not photo_time:
            quality = "无法匹配：无 EXIF 原始拍摄时间"
        else:
            quality = "无法匹配：无同 SEX、NYHA 的住院候选记录"
        common = [
            patient_id,
            sex,
            nyha,
            status,
            photo_time,
            selected["admission_time"] if selected else None,
            round(gap_days, 2) if gap_days is not None else None,
            selected["source_row"] if selected else None,
            len(patient_admissions),
            len(candidates),
            height,
            height_source,
            selected["weight_kg"] if selected else None,
            selected["age_y"] if selected else None,
            quality,
        ]
        audit_rows.append(common)
        if selected:
            automatic_rows.append(common)
        else:
            review_rows.append(
                common
                + [
                    min((item["admission_time"] for item in candidates), default=None),
                    max((item["admission_time"] for item in candidates), default=None),
                ]
            )

    admission_rows: list[list[object]] = []
    for patient_id in sorted(admissions):
        for item in sorted(admissions[patient_id], key=lambda value: value["admission_time"]):
            admission_rows.append(
                [
                    item["id"],
                    item["source_row"],
                    item["admission_number"],
                    item["admission_time"],
                    item["sex"],
                    item["nyha"],
                    item["height_cm"],
                    item["weight_kg"],
                    item["age_y"],
                    "是" if selected_rows.get(patient_id) == item["source_row"] else "否",
                ]
            )

    workbook = Workbook()
    guide = workbook.active
    guide.title = "使用说明"
    guide_rows = [
        ["项目", "内容"],
        ["推荐主表", "临床协变量_自动匹配：459 名患者，按 EXIF 原始拍摄时间最近的同 ID、同 SEX、同 NYHA 住院记录取 Weight 和 Age。"],
        ["复核规则", "匹配间隔 <=30 天标记为通过；31-39 天仍已匹配但建议核验。"],
        ["人工核验", "若仍有待核验记录：请补齐对应图片或用拍摄/采集/检查日期后再匹配。本次已同时检查 522 张主图片和 500 张 raw_scene 图片（文件名为 ID-1.jpg）。"],
        ["Height", "同一 ID 的已记录身高完全一致，因此可在匹配住院记录缺失时用同一患者其他住院记录补齐；仍缺失则保留空值。"],
        ["Weight 与 Age", "两者随住院时间变化。不可从其他住院记录补值，也不可按最早或最新住院记录任意替代。"],
        ["审计", "患者审计_483、住院明细_1068 保留每个取值的来源行、候选数和时间间隔，可用于复核和论文方法描述。"],
    ]
    write_sheet(guide, guide_rows[0], guide_rows[1:])
    headers = ["ID", "SEX", "NYHA", "选取状态", "EXIF原始拍摄时间", "匹配住院时间", "间隔_天", "原表行号", "该ID住院次数", "同SEX_NYHA候选数", "Height_cm", "Height来源", "Weight_kg", "Age_y", "质控结论"]
    write_sheet(workbook.create_sheet("临床协变量_自动匹配"), headers, automatic_rows)
    write_sheet(workbook.create_sheet(f"待人工核验_{len(review_rows)}"), headers + ["候选最早住院时间", "候选最晚住院时间"], review_rows)
    write_sheet(workbook.create_sheet("患者审计_483"), headers, audit_rows)
    write_sheet(workbook.create_sheet("住院明细_1068"), ["ID", "原表行号", "住院序号", "住院时间", "SEX", "NYHA", "Height_cm", "Weight_kg", "Age_y", "是否被选中"], admission_rows)
    for ws in workbook.worksheets:
        for row in ws.iter_rows():
            for cell in row:
                if isinstance(cell.value, datetime):
                    cell.number_format = "yyyy-mm-dd hh:mm:ss"
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    workbook.save(OUTPUT_PATH)
    print(f"Wrote {OUTPUT_PATH}")
    print(f"Automatically matched: {len(automatic_rows)}; manual review: {len(review_rows)}")


if __name__ == "__main__":
    main()
