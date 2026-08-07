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


def add_normalized_id(frame: pd.DataFrame, source_col: str, target_col: str = "sample_id_norm") -> pd.DataFrame:
    out = frame.copy()
    out[target_col] = out[source_col].map(normalize_id)
    return out
