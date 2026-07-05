"""Small XLSX writer fallback used by experiment summary scripts."""

from __future__ import annotations

import math
import zipfile
from pathlib import Path
from typing import Any
from xml.sax.saxutils import escape

import pandas as pd


def _excel_column(index: int) -> str:
    letters = ""
    index += 1
    while index:
        index, remainder = divmod(index - 1, 26)
        letters = chr(65 + remainder) + letters
    return letters


def _xlsx_cell(row_index: int, column_index: int, value: Any) -> str:
    ref = f"{_excel_column(column_index)}{row_index}"
    if value is None or pd.isna(value):
        return f'<c r="{ref}"/>'
    if isinstance(value, (int, float)) and math.isfinite(float(value)):
        return f'<c r="{ref}"><v>{value}</v></c>'
    text = escape(str(value))
    return f'<c r="{ref}" t="inlineStr"><is><t>{text}</t></is></c>'


def _sheet_xml(frame: pd.DataFrame) -> str:
    rows = []
    values = [list(frame.columns)] + frame.astype(object).where(pd.notna(frame), None).values.tolist()
    for row_index, row_values in enumerate(values, start=1):
        cells = [
            _xlsx_cell(row_index, column_index, value)
            for column_index, value in enumerate(row_values)
        ]
        rows.append(f'<row r="{row_index}">{"".join(cells)}</row>')
    return (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">'
        f'<sheetData>{"".join(rows)}</sheetData>'
        '</worksheet>'
    )


def _write_fallback_xlsx(path: Path, sheets: dict[str, pd.DataFrame]) -> None:
    sheet_names = list(sheets)
    content_types = [
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>',
        '<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">',
        '<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>',
        '<Default Extension="xml" ContentType="application/xml"/>',
        '<Override PartName="/xl/workbook.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet.main+xml"/>',
        '<Override PartName="/xl/styles.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.styles+xml"/>',
    ]
    for index in range(1, len(sheet_names) + 1):
        content_types.append(
            f'<Override PartName="/xl/worksheets/sheet{index}.xml" '
            'ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.worksheet+xml"/>'
        )
    content_types.append("</Types>")

    workbook_sheets = []
    workbook_rels = [
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>',
        '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">',
    ]
    for index, name in enumerate(sheet_names, start=1):
        safe_name = escape(name[:31], {'"': "&quot;"})
        workbook_sheets.append(f'<sheet name="{safe_name}" sheetId="{index}" r:id="rId{index}"/>')
        workbook_rels.append(
            f'<Relationship Id="rId{index}" '
            'Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/worksheet" '
            f'Target="worksheets/sheet{index}.xml"/>'
        )
    workbook_rels.append(
        f'<Relationship Id="rId{len(sheet_names) + 1}" '
        'Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/styles" '
        'Target="styles.xml"/>'
    )
    workbook_rels.append("</Relationships>")
    workbook = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<workbook xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main" '
        'xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships">'
        f'<sheets>{"".join(workbook_sheets)}</sheets>'
        '</workbook>'
    )
    root_rels = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
        '<Relationship Id="rId1" '
        'Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" '
        'Target="xl/workbook.xml"/>'
        '</Relationships>'
    )
    styles = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<styleSheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main"/>'
    )
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("[Content_Types].xml", "".join(content_types))
        archive.writestr("_rels/.rels", root_rels)
        archive.writestr("xl/workbook.xml", workbook)
        archive.writestr("xl/_rels/workbook.xml.rels", "".join(workbook_rels))
        archive.writestr("xl/styles.xml", styles)
        for index, name in enumerate(sheet_names, start=1):
            archive.writestr(f"xl/worksheets/sheet{index}.xml", _sheet_xml(sheets[name]))


def write_xlsx(path: str | Path, sheets: dict[str, pd.DataFrame]) -> None:
    """Write an XLSX workbook, falling back to a minimal XML writer if needed."""

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
            for name, frame in sheets.items():
                frame.to_excel(writer, sheet_name=name[:31], index=False)
    except ModuleNotFoundError:
        try:
            with pd.ExcelWriter(output_path, engine="xlsxwriter") as writer:
                for name, frame in sheets.items():
                    frame.to_excel(writer, sheet_name=name[:31], index=False)
        except ModuleNotFoundError:
            _write_fallback_xlsx(output_path, sheets)
