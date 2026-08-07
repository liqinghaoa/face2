"""P1-RGB's manifest-backed, RGB-only binary dataset."""
from __future__ import annotations
from pathlib import Path
from typing import Any
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset

REQUIRED_COLUMNS = {"case_id", "patient_group_id", "fold", "label_original", "label_3class", "label_binary", "rgb_path"}


class P1RGBBinaryDataset(Dataset):
    """Loads only P0 aligned RGB.  DECA assets and EXIF fields are never opened."""
    def __init__(self, frame: pd.DataFrame, transform: Any) -> None:
        missing = REQUIRED_COLUMNS.difference(frame.columns)
        if missing: raise ValueError(f"P1 RGB frame missing columns: {sorted(missing)}")
        self.frame = frame.reset_index(drop=True).copy()
        self.frame["case_id"] = self.frame["case_id"].astype(str)
        self.frame["patient_group_id"] = self.frame["patient_group_id"].astype(str)
        self.transform = transform
        labels = pd.to_numeric(self.frame["label_binary"], errors="coerce")
        if labels.isna().any() or not labels.isin((0, 1)).all(): raise ValueError("P1 RGB labels must be binary")
        self.labels = labels.astype(int).tolist()

    def __len__(self) -> int: return len(self.frame)

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.frame.iloc[index]; path = Path(str(row["rgb_path"]))
        if not path.is_file(): raise FileNotFoundError(f"missing P1 RGB: {path}")
        try:
            with Image.open(path) as image: tensor = self.transform(image.convert("RGB"))
        except Exception as exc:
            raise RuntimeError(f"cannot decode P1 RGB for {row['case_id']}: {path}") from exc
        return {"image": tensor, "label_binary": torch.tensor(self.labels[index], dtype=torch.long),
                "case_id": str(row["case_id"]), "patient_group_id": str(row["patient_group_id"]),
                "fold": int(row["fold"]), "label_original": int(row["label_original"]),
                "label_3class": int(row["label_3class"]), "image_path": str(path)}
