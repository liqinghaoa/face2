"""Minimal dependency-free XLSX writer for tabular experiment reports.

This module intentionally implements only the small subset of OOXML needed to
write pandas DataFrames as plain worksheets. It avoids optional Excel engines
such as openpyxl/xlsxwriter, which are not available in the training
environment and should not be installed for these experiments.
"""

from __future__ import annotations

import math
import re
import zipfile
from collections.abc import Mapping
from pathlib import Path
from typing import Any
from xml.sax.saxutils import escape

import pandas as pd


_INVALID_SHEET_CHARS = re.compile(r"[\[\]:*?/\\]")


def _column_name(index: int) -> str:
    if index < 1:
        raise ValueError(f"Excel column index must be positive, got {index}")
    letters: list[str] = []
    while index:
        index, remainder = divmod(index - 1, 26)
        letters.append(chr(ord("A") + remainder))
    return "".join(reversed(letters))


def _clean_sheet_name(name: str, used_names: set[str]) -> str:
    cleaned = _INVALID_SHEET_CHARS.sub("_", str(name)).strip().strip("'")
    if not cleaned:
        cleaned = "Sheet"
    cleaned = cleaned[:31]
    candidate = cleaned
    suffix = 1
    while candidate in used_names:
        marker = f"_{suffix}"
        candidate = f"{cleaned[:31 - len(marker)]}{marker}"
        suffix += 1
    used_names.add(candidate)
    return candidate


def _is_missing(value: Any) -> bool:
    try:
        return bool(pd.isna(value))
    except (TypeError, ValueError):
        return False


def _cell_xml(reference: str, value: Any) -> str:
    if _is_missing(value):
        return ""
    if isinstance(value, bool):
        return f'<c r="{reference}" t="b"><v>{1 if value else 0}</v></c>'
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        number = float(value)
        if math.isfinite(number):
            if isinstance(value, int) or number.is_integer():
                text = str(int(number))
            else:
                text = repr(number)
            return f'<c r="{reference}"><v>{text}</v></c>'
    text = str(value)
    escaped = escape(text)
    return (
        f'<c r="{reference}" t="inlineStr">'
        f'<is><t xml:space="preserve">{escaped}</t></is>'
        f"</c>"
    )


def _worksheet_xml(frame: pd.DataFrame) -> str:
    rows: list[str] = []
    header = [str(column) for column in frame.columns]
    all_rows = [header] + list(frame.itertuples(index=False, name=None))
    for row_index, row_values in enumerate(all_rows, start=1):
        cells: list[str] = []
        for column_index, value in enumerate(row_values, start=1):
            reference = f"{_column_name(column_index)}{row_index}"
            cell = _cell_xml(reference, value)
            if cell:
                cells.append(cell)
        rows.append(f'<row r="{row_index}">{"".join(cells)}</row>')
    return (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main" '
        'xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships">'
        f'<sheetData>{"".join(rows)}</sheetData>'
        "</worksheet>"
    )


def write_xlsx(path: str | Path, sheets: Mapping[str, pd.DataFrame]) -> Path:
    """Write DataFrames to a plain XLSX workbook using only stdlib zipfile."""

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    used_names: set[str] = set()
    normalized = [
        (_clean_sheet_name(name, used_names), frame.copy())
        for name, frame in sheets.items()
    ]
    if not normalized:
        normalized = [("Sheet1", pd.DataFrame())]

    workbook_sheets = []
    workbook_rels = []
    content_overrides = []
    for index, (sheet_name, _frame) in enumerate(normalized, start=1):
        escaped_name = escape(sheet_name, {'"': "&quot;"})
        workbook_sheets.append(
            f'<sheet name="{escaped_name}" sheetId="{index}" r:id="rId{index}"/>'
        )
        workbook_rels.append(
            f'<Relationship Id="rId{index}" '
            'Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/worksheet" '
            f'Target="worksheets/sheet{index}.xml"/>'
        )
        content_overrides.append(
            f'<Override PartName="/xl/worksheets/sheet{index}.xml" '
            'ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.worksheet+xml"/>'
        )

    styles_rel_id = len(normalized) + 1
    workbook_rels.append(
        f'<Relationship Id="rId{styles_rel_id}" '
        'Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/styles" '
        'Target="styles.xml"/>'
    )

    content_types = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">'
        '<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>'
        '<Default Extension="xml" ContentType="application/xml"/>'
        '<Override PartName="/xl/workbook.xml" '
        'ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet.main+xml"/>'
        '<Override PartName="/xl/styles.xml" '
        'ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.styles+xml"/>'
        f'{"".join(content_overrides)}'
        "</Types>"
    )
    root_rels = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
        '<Relationship Id="rId1" '
        'Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" '
        'Target="xl/workbook.xml"/>'
        "</Relationships>"
    )
    workbook = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<workbook xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main" '
        'xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships">'
        f'<sheets>{"".join(workbook_sheets)}</sheets>'
        "</workbook>"
    )
    workbook_relationships = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
        f'{"".join(workbook_rels)}'
        "</Relationships>"
    )
    styles = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<styleSheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">'
        '<fonts count="1"><font><sz val="11"/><name val="Calibri"/></font></fonts>'
        '<fills count="1"><fill><patternFill patternType="none"/></fill></fills>'
        '<borders count="1"><border><left/><right/><top/><bottom/><diagonal/></border></borders>'
        '<cellStyleXfs count="1"><xf numFmtId="0" fontId="0" fillId="0" borderId="0"/></cellStyleXfs>'
        '<cellXfs count="1"><xf numFmtId="0" fontId="0" fillId="0" borderId="0" xfId="0"/></cellXfs>'
        "</styleSheet>"
    )

    with zipfile.ZipFile(output_path, "w", compression=zipfile.ZIP_DEFLATED) as workbook_zip:
        workbook_zip.writestr("[Content_Types].xml", content_types)
        workbook_zip.writestr("_rels/.rels", root_rels)
        workbook_zip.writestr("xl/workbook.xml", workbook)
        workbook_zip.writestr("xl/_rels/workbook.xml.rels", workbook_relationships)
        workbook_zip.writestr("xl/styles.xml", styles)
        for index, (_sheet_name, frame) in enumerate(normalized, start=1):
            workbook_zip.writestr(f"xl/worksheets/sheet{index}.xml", _worksheet_xml(frame))
    return output_path
