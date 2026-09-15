"""Build the selected laboratory workbook with constant-memory XLSX writing.

``Workbook(write_only=True)`` never emits ``<cols>``, and drops ``freeze_panes``
and ``showGridLines`` along with it, so the column widths assigned to
``ws.column_dimensions`` are silently discarded. Rather than abandon streaming
(the detail sheet holds 230k rows and a normal in-memory workbook is not
affordable here), the missing blocks are spliced into the saved sheet XML and
the archive is rewritten -- still row-by-row, still constant memory.

Column positions are always resolved from header *names*, never hardcoded
indexes. Datetime formatting and column widths therefore cannot silently land
on the wrong column when ``--labels-csv`` inserts extra columns.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import shutil
import zipfile
from datetime import datetime
from pathlib import Path

from openpyxl import Workbook
from openpyxl.cell import WriteOnlyCell
from openpyxl.styles import Alignment, Font, PatternFill
from openpyxl.utils import get_column_letter


DATETIME_FORMAT = "%Y-%m-%d %H:%M:%S"
# Excel number format code -- must NOT reuse the strptime pattern above.
# Excel does not understand "%", so feeding it "%y-%m-%d %h:%m:%s" silently
# degrades the cell to its raw serial number (e.g. 45577.3564930556).
EXCEL_DATETIME_FORMAT = "yyyy-mm-dd hh:mm:ss"

# Datetime columns, identified by header name.
DETAIL_DATE_COLUMN_NAMES = {
    "EXIF原始拍摄时间",
    "住院锚点申请时间",
    "申请时间",
    "报告时间",
}
AUDIT_DATE_COLUMN_NAMES = {
    "EXIF原始拍摄时间",
    "住院1锚点申请时间",
    "住院1最晚申请时间",
    "住院2锚点申请时间",
    "住院2最晚申请时间",
}

# Column widths keyed by header name, so they survive column insertion.
DETAIL_COLUMN_WIDTHS = {
    "样本ID": 17,
    "患者ID": 15,
    "EXIF原始拍摄时间": 20,
    "选择规则": 25,
    "选择顺序": 14,
    "选择标签": 14,
    "相对EXIF位置": 14,
    "住院锚点申请时间": 20,
    "时间差_天": 12,
    "源表行号": 12,
    "SEX": 8,
    "sex_name": 10,
    "NYHA": 10,
    "binary_label": 13,
    "binary_name": 13,
}
AUDIT_COLUMN_WIDTHS = {
    "样本ID": 17,
    "患者ID": 17,
    "EXIF原始拍摄时间": 20,
    "可用住院次数": 16,
    "选择规则": 16,
    "选中住院数": 16,
}
DETAIL_DEFAULT_WIDTH = 15
AUDIT_DEFAULT_WIDTH = 18
AUDIT_FREEZE_PANES = "C2"

DEFAULT_LABEL_COLUMNS = ["SEX", "sex_name", "NYHA", "binary_label", "binary_name"]
LABEL_ID_COLUMN = "ID"


def parse_datetime(value: object) -> object:
    if not isinstance(value, str):
        return value
    try:
        return datetime.strptime(value, DATETIME_FORMAT)
    except ValueError:
        return value


def coerce_label(value: object) -> object:
    """Write integer-like labels as numbers, leave everything else as text."""
    if isinstance(value, str) and re.fullmatch(r"-?\d+", value.strip()):
        return int(value)
    return value


def load_labels(path: Path, columns: list[str]) -> dict[str, list[object]]:
    """Map sample ID -> label values, keeping the full ``-N`` suffix.

    Sample IDs such as ``201562603-1`` must NOT be trimmed to a patient ID:
    among the 17 patients holding two samples, 15 carry a different NYHA per
    sample, so a patient-level join would silently mislabel 34 rows.
    """
    with path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        headers = reader.fieldnames or []
        missing = [c for c in [LABEL_ID_COLUMN, *columns] if c not in headers]
        if missing:
            raise ValueError(f"Label CSV is missing columns: {missing}")
        labels: dict[str, list[object]] = {}
        for line_number, row in enumerate(reader, start=2):
            sample_id = (row[LABEL_ID_COLUMN] or "").strip()
            if not sample_id:
                raise ValueError(f"Label CSV line {line_number} has a blank {LABEL_ID_COLUMN}")
            if sample_id in labels:
                raise ValueError(f"Label CSV has a duplicate {LABEL_ID_COLUMN}: {sample_id}")
            labels[sample_id] = [coerce_label(row[c]) for c in columns]
    return labels


def styled_header(worksheet, headers: list[str], fill_color: str) -> list[WriteOnlyCell]:
    cells: list[WriteOnlyCell] = []
    for value in headers:
        cell = WriteOnlyCell(worksheet, value=value)
        cell.fill = PatternFill("solid", fgColor=fill_color)
        cell.font = Font(name="Arial", size=10, bold=True, color="FFFFFF")
        cell.alignment = Alignment(horizontal="center", vertical="center", wrap_text=True)
        cells.append(cell)
    return cells


def styled_row(worksheet, values: list[object], date_columns: set[int]) -> list[object]:
    output: list[object] = []
    for index, value in enumerate(values):
        if index in date_columns:
            parsed = parse_datetime(value)
            if isinstance(parsed, datetime):
                cell = WriteOnlyCell(worksheet, value=parsed)
                cell.number_format = EXCEL_DATETIME_FORMAT
                output.append(cell)
                continue
        output.append(value)
    return output


def date_indexes(headers: list[str], names: set[str]) -> set[int]:
    return {index for index, header in enumerate(headers) if header in names}


def widths_from_headers(headers: list[str], by_name: dict[str, int], default: int) -> dict[int, int]:
    return {index + 1: by_name.get(header, default) for index, header in enumerate(headers)}


def freeze_panes_for(headers: list[str], through_name: str | None, fallback_columns: int = 2) -> str:
    """Freeze the identifying/label block so it stays visible while scrolling.

    ``freeze_panes="C2"`` freezes columns A-B, i.e. the split letter is one
    *past* the last frozen column. Freezing through the column at 0-based
    ``index`` therefore needs letter ``index + 2``.
    """
    if through_name and through_name in headers:
        return f"{get_column_letter(headers.index(through_name) + 2)}2"
    return f"{get_column_letter(fallback_columns + 1)}2"


def build_cols_xml(widths: dict[int, int], column_count: int) -> str:
    """Collapse per-column widths into contiguous ``<col>`` spans."""
    resolved = {c: widths.get(c, DETAIL_DEFAULT_WIDTH) for c in range(1, column_count + 1)}
    spans: list[str] = []
    start = 1
    for column in range(2, column_count + 2):
        if column > column_count or resolved[column] != resolved[start]:
            width = resolved[start]
            spans.append(f'<col min="{start}" max="{column - 1}" width="{width}" customWidth="1"/>')
            start = column
    return "<cols>" + "".join(spans) + "</cols>"


def build_sheet_views_xml(freeze_panes: str | None, show_gridlines: bool) -> str:
    attrs = "" if show_gridlines else 'showGridLines="0" '
    pane = ""
    if freeze_panes:
        match = re.fullmatch(r"([A-Z]+)(\d+)", freeze_panes)
        if match:
            split_col, split_row = match.group(1), int(match.group(2))
            x_split = sum((ord(ch) - 64) * 26 ** i for i, ch in enumerate(reversed(split_col)))
            pane = (
                f'<pane xSplit="{x_split - 1}" ySplit="{split_row - 1}" '
                f'topLeftCell="{freeze_panes}" activePane="bottomRight" state="frozen"/>'
            )
    return f'<sheetViews><sheetView {attrs}workbookViewId="0">{pane}</sheetView></sheetViews>'


def patch_sheet_xml(
    path: Path,
    sheet_index: int,
    widths: dict[int, int],
    column_count: int,
    freeze_panes: str | None,
    show_gridlines: bool,
) -> None:
    """Splice ``<cols>`` and ``<sheetViews>`` into a write-only generated sheet.

    ``sheet_index`` is 1-based in workbook sheet order, matching ``sheetN.xml``.
    """
    target = f"xl/worksheets/sheet{sheet_index}.xml"
    cols_xml = build_cols_xml(widths, column_count)
    views_xml = build_sheet_views_xml(freeze_panes, show_gridlines)
    temp = path.with_name(path.name + ".patching")

    with zipfile.ZipFile(path) as source, zipfile.ZipFile(temp, "w", zipfile.ZIP_DEFLATED) as sink:
        for item in source.infolist():
            data = source.read(item.filename)
            if item.filename == target:
                text = data.decode("utf-8")
                if "<cols>" not in text:
                    text = text.replace("<sheetData>", cols_xml + "<sheetData>", 1)
                text = re.sub(r"<sheetViews>.*?</sheetViews>", views_xml, text, count=1, flags=re.S)
                if "<sheetViews>" not in text:
                    text = text.replace("<sheetFormatPr", views_xml + "<sheetFormatPr", 1)
                data = text.encode("utf-8")
            sink.writestr(item, data)

    shutil.move(str(temp), str(path))


def resolve_column_width(worksheet, column_index: int) -> float | None:
    """Read a column's width, resolving the span that covers it.

    openpyxl keys a ``<col min="5" max="7">`` span under the *first* letter only,
    so ``ws.column_dimensions['F']`` falls through to the default even though the
    span covers it. Match on each dimension's min/max instead.
    """
    for dimension in worksheet.column_dimensions.values():
        if dimension.min <= column_index <= dimension.max:
            return dimension.width
    return None


def verify_workbook(
    path: Path,
    sheet_name: str,
    date_columns: set[int],
    widths: dict[int, int],
    default_width: int,
    freeze_panes: str,
    column_count: int,
    label_indexes: list[int] | None = None,
    probe_rows: int = 60,
) -> list[str]:
    """Re-open the written sheet and confirm formatting survived the round-trip."""
    from openpyxl import load_workbook

    workbook = load_workbook(path, read_only=False)
    worksheet = workbook[sheet_name]
    problems: list[str] = []

    for row in worksheet.iter_rows(min_row=2, max_row=probe_rows + 1):
        for index in date_columns:
            if index >= len(row):
                continue
            cell = row[index]
            if cell.value is not None and not isinstance(cell.value, datetime):
                problems.append(f"{sheet_name}!{cell.coordinate} is {type(cell.value).__name__}, not datetime")
        for index in label_indexes or []:
            if index >= len(row):
                continue
            cell = row[index]
            if cell.value is None or str(cell.value).strip() == "":
                problems.append(f"empty label at {sheet_name}!{cell.coordinate}")

    for column in range(1, column_count + 1):
        expected = widths.get(column, default_width)
        actual = resolve_column_width(worksheet, column)
        if actual != expected:
            letter = get_column_letter(column)
            problems.append(f"{sheet_name}!{letter} width {actual} != {expected}")

    if worksheet.freeze_panes != freeze_panes:
        problems.append(f"{sheet_name} freeze_panes {worksheet.freeze_panes!r} != {freeze_panes!r}")

    workbook.close()
    return problems


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("detail_jsonl", type=Path)
    parser.add_argument("audit_json", type=Path)
    parser.add_argument("output_xlsx", type=Path)
    parser.add_argument("--labels-csv", type=Path, default=None,
                        help="CSV keyed by ID, e.g. nyha_2class_with_exif_time.csv")
    parser.add_argument("--label-columns", default=",".join(DEFAULT_LABEL_COLUMNS),
                        help="Comma-separated label columns to inject")
    parser.add_argument("--labels-target", choices=["detail", "audit", "both"], default="detail")
    parser.add_argument("--insert-after", default="样本ID",
                        help="Header name after which label columns are inserted")
    args = parser.parse_args()

    label_columns = [c.strip() for c in args.label_columns.split(",") if c.strip()]
    labels = load_labels(args.labels_csv, label_columns) if args.labels_csv else None
    enrich_detail = labels is not None and args.labels_target in {"detail", "both"}
    enrich_audit = labels is not None and args.labels_target in {"audit", "both"}

    audit = json.loads(args.audit_json.read_text(encoding="utf-8"))
    workbook = Workbook(write_only=True)

    detail_sheet = workbook.create_sheet("选中检验数据")
    detail_headers: list[str] | None = None
    detail_count = 0
    insert_at = 0
    missing_samples: list[str] = []
    with args.detail_jsonl.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle):
            values = json.loads(line)
            if line_number == 0:
                original_headers = values["headers"]
                if enrich_detail:
                    if args.insert_after not in original_headers:
                        raise ValueError(f"{args.insert_after!r} not found in detail headers")
                    insert_at = original_headers.index(args.insert_after) + 1
                    detail_headers = (
                        original_headers[:insert_at] + label_columns + original_headers[insert_at:]
                    )
                else:
                    detail_headers = list(original_headers)
                detail_date_columns = date_indexes(detail_headers, DETAIL_DATE_COLUMN_NAMES)
                detail_sheet.append(styled_header(detail_sheet, detail_headers, "1F4E78"))
                continue

            if enrich_detail:
                sample_id = str(values[0]).strip()
                label_values = labels.get(sample_id)
                if label_values is None:
                    missing_samples.append(sample_id)
                    label_values = [None] * len(label_columns)
                values = values[:insert_at] + label_values + values[insert_at:]

            detail_sheet.append(styled_row(detail_sheet, values, detail_date_columns))
            detail_count += 1

    if detail_headers is None:
        raise ValueError("Detail JSONL is empty")
    if missing_samples:
        unique = sorted(set(missing_samples))
        raise ValueError(f"{len(unique)} sample IDs have no label row, e.g. {unique[:5]}")

    detail_widths = widths_from_headers(detail_headers, DETAIL_COLUMN_WIDTHS, DETAIL_DEFAULT_WIDTH)
    detail_freeze = freeze_panes_for(
        detail_headers, label_columns[-1] if enrich_detail else None, fallback_columns=2
    )
    detail_sheet.freeze_panes = detail_freeze
    detail_sheet.sheet_view.showGridLines = False
    detail_sheet.auto_filter.ref = f"A1:{get_column_letter(len(detail_headers))}{detail_count + 1}"

    audit_sheet = workbook.create_sheet("匹配审计")
    audit_headers = list(audit["headers"])
    audit_insert_at = 0
    if enrich_audit:
        if args.insert_after not in audit_headers:
            raise ValueError(f"{args.insert_after!r} not found in audit headers")
        audit_insert_at = audit_headers.index(args.insert_after) + 1
        audit_headers = audit_headers[:audit_insert_at] + label_columns + audit_headers[audit_insert_at:]
    audit_date_columns = date_indexes(audit_headers, AUDIT_DATE_COLUMN_NAMES)
    audit_sheet.append(styled_header(audit_sheet, audit_headers, "4472C4"))

    audit_missing: list[str] = []
    for row in audit["rows"]:
        if enrich_audit:
            sample_id = str(row[0]).strip()
            label_values = labels.get(sample_id)
            if label_values is None:
                audit_missing.append(sample_id)
                label_values = [None] * len(label_columns)
            row = row[:audit_insert_at] + label_values + row[audit_insert_at:]
        audit_sheet.append(styled_row(audit_sheet, row, audit_date_columns))
    if audit_missing:
        unique = sorted(set(audit_missing))
        raise ValueError(f"{len(unique)} audit sample IDs have no label row, e.g. {unique[:5]}")

    audit_widths = widths_from_headers(audit_headers, AUDIT_COLUMN_WIDTHS, AUDIT_DEFAULT_WIDTH)
    audit_freeze = freeze_panes_for(
        audit_headers, label_columns[-1] if enrich_audit else None, fallback_columns=2
    )
    audit_sheet.freeze_panes = audit_freeze
    audit_sheet.sheet_view.showGridLines = False
    audit_sheet.auto_filter.ref = f"A1:{get_column_letter(len(audit_headers))}{len(audit['rows']) + 1}"

    args.output_xlsx.parent.mkdir(parents=True, exist_ok=True)
    workbook.save(args.output_xlsx)

    # write_only drops <cols>/<sheetViews>; splice them back in.
    patch_sheet_xml(args.output_xlsx, 1, detail_widths, len(detail_headers), detail_freeze, False)
    patch_sheet_xml(args.output_xlsx, 2, audit_widths, len(audit_headers), audit_freeze, False)

    problems = [
        *verify_workbook(
            args.output_xlsx, "选中检验数据", detail_date_columns, detail_widths,
            DETAIL_DEFAULT_WIDTH, detail_freeze, len(detail_headers),
            [detail_headers.index(c) for c in label_columns] if enrich_detail else None,
        ),
        *verify_workbook(
            args.output_xlsx, "匹配审计", audit_date_columns, audit_widths,
            AUDIT_DEFAULT_WIDTH, audit_freeze, len(audit_headers),
            [audit_headers.index(c) for c in label_columns] if enrich_audit else None,
        ),
    ]
    if problems:
        raise RuntimeError(f"Round-trip verification failed: {problems[:10]}")

    print(json.dumps({
        "output": str(args.output_xlsx),
        "detail_rows": detail_count,
        "detail_columns": len(detail_headers),
        "audit_rows": len(audit["rows"]),
        "audit_columns": len(audit_headers),
        "labels_injected": label_columns if labels else [],
        "labels_target": args.labels_target if labels else None,
        "detail_freeze": detail_freeze,
        "audit_freeze": audit_freeze,
        "summary": audit["summary"],
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
