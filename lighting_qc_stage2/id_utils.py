from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd


def normalize_id(value: Any) -> str:
    if pd.isna(value):
        return ""
    text = str(value).strip()
    if text.endswith(".0") and text[:-2].isdigit():
        text = text[:-2]
    name = Path(text).name
    if "." in name:
        stem = Path(name).stem
        if stem:
            name = stem
    return name.strip()
