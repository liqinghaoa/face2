"""Inject label columns from a sample CSV into an existing lab workbook.

Unlike ``build_selected_lab_visits_workbook_streaming.py`` (which rebuilds from
the JSONL intermediates), this script reads an existing XLSX so that manual
edits made in Excel -- deleted columns, tuned column widths -- are preserved.
Only the named sheet receives the new columns; every other sheet is copied
through unchanged.

The join key is the *full* sample ID including any ``-N`` suffix. Among the 17
patients holding two samples, 15 carry a different NYHA per sample, so joining
on a de-suffixed patient ID would silently mislabel 34 rows.

Output is written with a write-only workbook (constant memory) and then has
``<dimension>``/``<cols>``/``<sheetViews>`` spliced in, because write-only mode
omits all three.
"""

from __future__ import annotations

import argparse
import csv
import re
import shutil
import zipfile
from datetime import datetime
from pathlib import Path

from openpyxl import Workbook, load_workbook
from openpyxl.cell import WriteOnlyCell
from openpyxl.styles import Alignment, Font, PatternFill
from openpyxl.utils import get_column_letter


DATETIME_FORMAT = "%Y-%m-%d %H:%M:%S"
EXCEL_DATETIME_FORMAT = "yyyy-mm-dd hh:mm:ss"
DATE_COLUMN_NAMES = {
    "EXIF原始拍摄时间",
    "住院锚点申请时间",
    "申请时间",
    "报告时间",
    "住院1锚点申请时间",
    "住院1最晚申请时间",
    "住院2锚点申请时间",
    "住院2最晚申请时间",
}
DEFAULT_LABEL_COLUMNS = ["SEX", "sex_name", "NYHA", "binary_label", "binary_name"]
LABEL_COLUMN_WIDTHS = {"SEX": 8, "sex_name": 10, "NYHA": 10, "binary_label": 13, "binary_name": 13}
LABEL_ID_COLUMN = "ID"
DEFAULT_WIDTH = 13.0
DEFAULT_TARGET_SHEETS = ["选中检验数据"]


def fmt_width(value: float) -> str:
    return str(int(value)) if float(value).is_integer() else repr(float(value))


def parse_datetime(value: object) -> object:
    if isinstance(value, datetime):
        return value
    if isinstance(value, str):
        text = value.strip()
        for pattern in (DATETIME_FORMAT, "%Y-%m-%d %H:%M:%S", "%Y:%m:%d %H:%M:%S"):
            try:
                return datetime.strptime(text, pattern)
            except ValueError:
                continue
    return value


def coerce_label(value: object) -> object:
    if isinstance(value, str) and re.fullmatch(r"-?\d+", value.strip()):
        return int(value)
    return value


def load_labels(path: Path, columns: list[str]) -> dict[str, list[object]]:
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
                raise ValueError(f"Duplicate {LABEL_ID_COLUMN} in label CSV: {sample_id}")
            labels[sample_id] = [coerce_label(row[c]) for c in columns]
    return labels


def sheet_dimensions(zf: zipfile.ZipFile, sheet_path: str) -> tuple[dict[int, float], str | None]:
    """Read per-column widths and the freeze-pane split straight from the sheet XML."""
    xml = zf.read(sheet_path).decode("utf-8")
    widths: dict[int, float] = {}
    cols = re.search(r"<cols>(.*?)</cols>", xml, re.S)
    if cols:
        for match in re.finditer(r'<col min="(\d+)" max="(\d+)" width="([\d.]+)"', cols.group(1)):
            low, high, width = int(match.group(1)), int(match.group(2)), float(match.group(3))
            for index in range(low, high + 1):
                widths[index] = width
    freeze = None
    pane = re.search(r'<pane([^>]*)/>', xml)
    if pane:
        attrs = pane.group(1)
        x = re.search(r'xSplit="(\d+)"', attrs)
        y = re.search(r'ySplit="(\d+)"', attrs)
        split_col = int(x.group(1)) if x else 0
        split_row = int(y.group(1)) if y else 0
        if split_col or split_row:
            freeze = f"{get_column_letter(split_col + 1)}{split_row + 1}"
    return widths, freeze


def build_cols_xml(widths: dict[int, float], column_count: int) -> str:
    resolved = {c: widths.get(c, DEFAULT_WIDTH) for c in range(1, column_count + 1)}
    spans: list[str] = []
    start = 1
    for column in range(2, column_count + 2):
        if column > column_count or resolved[column] != resolved[start]:
            spans.append(
                f'<col min="{start}" max="{column - 1}" width="{fmt_width(resolved[start])}" customWidth="1"/>'
            )
            start = column
    return "<cols>" + "".join(spans) + "</cols>"


def build_sheet_views_xml(freeze_panes: str | None, show_gridlines: bool) -> str:
    attrs = "" if show_gridlines else 'showGridLines="0" '
    pane = ""
    if freeze_panes:
        match = re.fullmatch(r"([A-Z]+)(\d+)", freeze_panes)
        if match:
            letter, row = match.group(1), int(match.group(2))
            x_split = sum((ord(ch) - 64) * 26 ** i for i, ch in enumerate(reversed(letter)))
            pane = (
                f'<pane xSplit="{x_split - 1}" ySplit="{row - 1}" '
                f'topLeftCell="{freeze_panes}" activePane="bottomRight" state="frozen"/>'
            )
    return f'<sheetViews><sheetView {attrs}workbookViewId="0">{pane}</sheetView></sheetViews>'


def patch_sheet(path: Path, sheet_index: int, widths: dict[int, float], column_count: int,
                row_count: int, freeze_panes: str | None, show_gridlines: bool) -> None:
    target = f"xl/worksheets/sheet{sheet_index}.xml"
    cols_xml = build_cols_xml(widths, column_count)
    views_xml = build_sheet_views_xml(freeze_panes, show_gridlines)
    dimension = f'<dimension ref="A1:{get_column_letter(column_count)}{row_count + 1}"/>'
    temp = path.with_name(path.name + ".patching")

    with zipfile.ZipFile(path) as source, zipfile.ZipFile(temp, "w", zipfile.ZIP_DEFLATED) as sink:
        for item in source.infolist():
            data = source.read(item.filename)
            if item.filename == target:
                text = data.decode("utf-8")
                text = re.sub(r"<sheetViews>.*?</sheetViews>", views_xml, text, count=1, flags=re.S)
                if "<sheetViews>" not in text:
                    text = text.replace("<sheetFormatPr", views_xml + "<sheetFormatPr", 1)
                text = re.sub(r"<dimension[^>]*/>", "", text)
                text = text.replace("<sheetViews>", dimension + "<sheetViews>", 1)
                if "<cols>" in text:
                    text = re.sub(r"<cols>.*?</cols>", cols_xml, text, count=1, flags=re.S)
                else:
                    text = text.replace("<sheetData>", cols_xml + "<sheetData>", 1)
                data = text.encode("utf-8")
            sink.writestr(item, data)
    shutil.move(str(temp), str(path))


def styled_header(worksheet, headers: list[str], fill_color: str) -> list[WriteOnlyCell]:
    cells: list[WriteOnlyCell] = []
    for value in headers:
        cell = WriteOnlyCell(worksheet, value=value)
        cell.fill = PatternFill("solid", fgColor=fill_color)
        cell.font = Font(name="Arial", size=10, bold=True, color="FFFFFF")
        cell.alignment = Alignment(horizontal="center", vertical="center", wrap_text=True)
        cells.append(cell)
    return cells


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("source_xlsx", type=Path)
    parser.add_argument("labels_csv", type=Path)
    parser.add_argument("output_xlsx", type=Path)
    parser.add_argument("--sheet", action="append", dest="sheets",
                        help="目标 sheet，可重复传入以同时注入多个 sheet")
    parser.add_argument("--label-columns", default=",".join(DEFAULT_LABEL_COLUMNS))
    parser.add_argument("--insert-after", default="样本ID")
    args = parser.parse_args()

    label_columns = [c.strip() for c in args.label_columns.split(",") if c.strip()]
    labels = load_labels(args.labels_csv, label_columns)
    print(f"标签表: {len(labels)} 个样本, 注入 {len(label_columns)} 列")

    with zipfile.ZipFile(args.source_xlsx) as zf:
        sheet_paths = sorted(
            [n for n in zf.namelist() if re.fullmatch(r"xl/worksheets/sheet\d+\.xml", n)],
            key=lambda n: int(re.search(r"(\d+)", n).group(1)),
        )
        widths_by_sheet = {}
        freeze_by_sheet = {}
        for index, sheet_path in enumerate(sheet_paths, start=1):
            widths_by_sheet[index], freeze_by_sheet[index] = sheet_dimensions(zf, sheet_path)

    source = load_workbook(args.source_xlsx, read_only=True, data_only=True)
    sheet_names = list(source.sheetnames)
    requested = args.sheets or DEFAULT_TARGET_SHEETS
    unknown = [name for name in requested if name not in sheet_names]
    if unknown:
        raise ValueError(f"源表中不存在这些 sheet: {unknown} (可选: {sheet_names})")
    target_indexes = {sheet_names.index(name) + 1 for name in requested}
    print(f"源表 {len(sheet_names)} 个 sheet: {sheet_names}")
    print(f"目标 sheet {requested} -> 序号 {sorted(target_indexes)}")

    workbook = Workbook(write_only=True)
    target_headers: dict[int, list[str]] = {}
    missing: list[str] = []
    sheet_meta: dict[int, tuple[int, int]] = {}

    for index, sheet_name in enumerate(sheet_names, start=1):
        worksheet = workbook.create_sheet(sheet_name)
        source_sheet = source[sheet_name]
        rows = source_sheet.iter_rows(values_only=True)
        header = [("" if v is None else str(v)) for v in next(rows)]

        enrich = index in target_indexes
        insert_at = header.index(args.insert_after) + 1 if enrich else 0
        if enrich and args.insert_after not in header:
            raise ValueError(f"[{sheet_name}] 找不到插入锚点列 {args.insert_after!r}")
        out_header = header[:insert_at] + label_columns + header[insert_at:] if enrich else header
        if enrich:
            target_headers[index] = out_header
        # Parse by SOURCE position, format by OUTPUT position -- the two differ
        # once the label columns are inserted.
        source_date_indexes = {i for i, h in enumerate(header) if h in DATE_COLUMN_NAMES}
        output_date_indexes = {i for i, h in enumerate(out_header) if h in DATE_COLUMN_NAMES}

        worksheet.append(styled_header(worksheet, out_header, "1F4E78" if enrich else "4472C4"))

        count = 0
        for raw in rows:
            if all(v is None for v in raw):
                continue
            values = [parse_datetime(v) if i in source_date_indexes else v
                      for i, v in enumerate(raw)]
            if enrich:
                sample_id = str(raw[0]).strip()
                found = labels.get(sample_id)
                if found is None:
                    missing.append(sample_id)
                    found = [None] * len(label_columns)
                values = values[:insert_at] + found + values[insert_at:]

            # Datetimes must carry an explicit Excel format, or Excel renders the
            # raw serial number instead of a date.
            row_cells: list[object] = []
            for position, value in enumerate(values):
                if position in output_date_indexes and isinstance(value, datetime):
                    cell = WriteOnlyCell(worksheet, value=value)
                    cell.number_format = EXCEL_DATETIME_FORMAT
                    row_cells.append(cell)
                else:
                    row_cells.append(value)
            worksheet.append(row_cells)
            count += 1
        print(f"  [{sheet_name}] {count} 行 × {len(out_header)} 列{' (注入标签)' if enrich else ''}")

        widths = dict(widths_by_sheet.get(index, {}))
        if enrich:
            shifted = {k + len(label_columns): v for k, v in widths.items() if k >= insert_at + 1}
            widths = {k: v for k, v in widths.items() if k <= insert_at}
            widths.update(shifted)
            for offset, name in enumerate(label_columns, start=insert_at + 1):
                widths[offset] = LABEL_COLUMN_WIDTHS.get(name, DEFAULT_WIDTH)
        freeze = freeze_by_sheet.get(index)
        if enrich:
            freeze = f"{get_column_letter(insert_at + len(label_columns) + 1)}2"
        worksheet.freeze_panes = freeze
        worksheet.sheet_view.showGridLines = False
        worksheet.auto_filter.ref = f"A1:{get_column_letter(len(out_header))}{count + 1}"
        widths_by_sheet[index] = widths
        freeze_by_sheet[index] = freeze
        sheet_meta[index] = (len(out_header), count)

    source.close()
    if missing:
        unique = sorted(set(missing))
        raise ValueError(f"{len(unique)} 个样本ID在标签表中不存在, 例如 {unique[:5]}")

    args.output_xlsx.parent.mkdir(parents=True, exist_ok=True)
    workbook.save(args.output_xlsx)
    for index in range(1, len(sheet_names) + 1):
        column_count, row_count = sheet_meta[index]
        patch_sheet(args.output_xlsx, index, widths_by_sheet[index],
                    column_count, row_count, freeze_by_sheet[index], False)

    print(f"已写出 {args.output_xlsx}")
    for index in sorted(target_headers):
        print(f"  [{sheet_names[index - 1]}] 表头: {' | '.join(target_headers[index])}")


if __name__ == "__main__":
    main()
