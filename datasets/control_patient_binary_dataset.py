"""Dataset adapter for the fixed control-versus-patient E0B task."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset


def map_three_class_to_binary(label: object) -> int:
    """Map E0's Normal/Mild/Severe labels to Control/Patient explicitly."""
    try:
        value = int(label)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Unsupported original three-class label: {label!r}") from exc
    if value == 0:
        return 0
    if value in (1, 2):
        return 1
    raise ValueError(f"Unsupported original three-class label: {label!r}")


class ControlPatientFaceDataset(Dataset):
    """E0 image loading with a non-mutating binary label adapter."""

    required_columns = {"ID", "patient_group_id", "NYHA", "label_3class", "fold"}

    def __init__(
        self,
        csv_path: str | Path,
        transform: Any,
        image_root: str | Path,
        image_filename_template: str = "{ID}.png",
    ) -> None:
        self.csv_path = Path(csv_path)
        self.frame = pd.read_csv(
            self.csv_path,
            dtype={"ID": "string", "patient_group_id": "string"},
            encoding="utf-8-sig",
        )
        missing = self.required_columns.difference(self.frame.columns)
        if missing:
            raise ValueError(f"{self.csv_path} is missing columns: {sorted(missing)}")
        self.transform = transform
        self.image_root = Path(image_root)
        self.image_filename_template = image_filename_template
        self.labels = [map_three_class_to_binary(v) for v in self.frame["label_3class"]]

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.frame.iloc[index]
        image_path = self.image_root / self.image_filename_template.format(ID=str(row["ID"]))
        if not image_path.is_file():
            raise FileNotFoundError(f"Image does not exist: {image_path}")
        with Image.open(image_path) as image:
            tensor = self.transform(image.convert("RGB"))
        return {
            "image": tensor,
            "label": torch.tensor(self.labels[index], dtype=torch.long),
            "sample_id": str(row["ID"]),
            "patient_group_id": str(row["patient_group_id"]),
            "original_label": int(row["NYHA"]),
            "original_three_class_label": int(row["label_3class"]),
            "original_three_class_name": str(row.get("label_3class_name", "")),
            "image_path": str(image_path),
            "fold": int(row["fold"]),
        }
