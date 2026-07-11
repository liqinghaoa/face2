"""Dataset for global face + selected ROI feature-level fusion experiments."""

from __future__ import annotations

import random
from pathlib import Path
from typing import Any, Mapping, Sequence

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import functional as TF


REQUIRED_COLUMNS = {"ID", "patient_group_id", "label_3class"}
SUPPORTED_INPUTS = ("global", "eye", "cheek")


class GlobalROIFusionDataset(Dataset):
    """Load synchronized global/ROI inputs from fixed split CSV files."""

    def __init__(
        self,
        csv_path: str | Path,
        global_image_root: str | Path,
        roi_roots: Mapping[str, str | Path],
        enabled_inputs: Sequence[str],
        image_filename_template: str = "{ID}.png",
        image_size: int = 224,
        label_col: str = "label_3class",
        train: bool = False,
        horizontal_flip: bool = False,
        mean: Sequence[float] = (0.485, 0.456, 0.406),
        std: Sequence[float] = (0.229, 0.224, 0.225),
    ) -> None:
        self.csv_path = Path(csv_path).expanduser().resolve()
        self.global_image_root = Path(global_image_root).expanduser().resolve()
        self.roi_roots = {
            str(name).strip().lower(): Path(path).expanduser().resolve()
            for name, path in roi_roots.items()
        }
        self.enabled_inputs = [str(name).strip().lower() for name in enabled_inputs]
        self.image_filename_template = str(image_filename_template)
        self.image_size = int(image_size)
        self.label_col = str(label_col)
        self.train = bool(train)
        self.horizontal_flip = bool(horizontal_flip)
        self.mean = list(mean)
        self.std = list(std)

        if not self.csv_path.is_file():
            raise FileNotFoundError(f"Dataset CSV does not exist: {self.csv_path}")
        if not self.global_image_root.is_dir():
            raise FileNotFoundError(
                f"Global image root does not exist: {self.global_image_root}"
            )
        if "global" not in self.enabled_inputs:
            raise ValueError("enabled_inputs must include 'global'")
        if len(set(self.enabled_inputs)) != len(self.enabled_inputs):
            raise ValueError(f"enabled_inputs contain duplicates: {self.enabled_inputs}")
        unsupported = sorted(set(self.enabled_inputs).difference(SUPPORTED_INPUTS))
        if unsupported:
            raise ValueError(
                f"Unsupported enabled_inputs {unsupported}; choose from {list(SUPPORTED_INPUTS)}"
            )
        for roi_name in self.enabled_roi_names:
            if roi_name not in self.roi_roots:
                raise ValueError(f"Missing roi_roots entry for enabled ROI: {roi_name}")
            if not self.roi_roots[roi_name].is_dir():
                raise FileNotFoundError(
                    f"ROI image root for {roi_name!r} does not exist: {self.roi_roots[roi_name]}"
                )
        if self.image_size < 1:
            raise ValueError(f"image_size must be positive, got {self.image_size}")

        self.frame = pd.read_csv(
            self.csv_path,
            dtype={"ID": "string", "patient_group_id": "string"},
            encoding="utf-8-sig",
        )
        required = set(REQUIRED_COLUMNS)
        required.add(self.label_col)
        missing = sorted(required.difference(self.frame.columns))
        if missing:
            raise ValueError(
                f"Dataset CSV is missing required columns {missing}: {self.csv_path}"
            )
        if self.frame.empty:
            raise ValueError(f"Dataset CSV contains no samples: {self.csv_path}")

        labels = pd.to_numeric(self.frame[self.label_col], errors="coerce")
        if labels.isna().any() or not labels.isin([0, 1, 2]).all():
            bad_rows = self.frame.index[labels.isna() | ~labels.isin([0, 1, 2])].tolist()
            raise ValueError(
                f"Invalid {self.label_col} values at zero-based rows {bad_rows[:10]} "
                f"in {self.csv_path}"
            )
        self.frame[self.label_col] = labels.astype("int64")

    @property
    def enabled_roi_names(self) -> list[str]:
        return [name for name in self.enabled_inputs if name != "global"]

    @property
    def labels(self) -> list[int]:
        return self.frame[self.label_col].astype(int).tolist()

    def __len__(self) -> int:
        return len(self.frame)

    def _filename(self, identifier: str) -> str:
        return self.image_filename_template.format(ID=identifier)

    def global_image_path(self, identifier: str) -> Path:
        return self.global_image_root / self._filename(identifier)

    def roi_image_path(self, roi_name: str, identifier: str) -> Path:
        normalized = str(roi_name).strip().lower()
        return self.roi_roots[normalized] / self._filename(identifier)

    def _load_image(self, path: Path, identifier: str, input_name: str, do_flip: bool) -> torch.Tensor:
        if not path.is_file():
            raise FileNotFoundError(
                f"Missing {input_name} image for ID={identifier!r}: {path}"
            )
        try:
            with Image.open(path) as image:
                image = image.convert("RGB")
                image = TF.resize(image, [self.image_size, self.image_size])
                if do_flip:
                    image = TF.hflip(image)
                tensor = TF.to_tensor(image)
                return TF.normalize(tensor, mean=self.mean, std=self.std)
        except Exception as exc:
            raise RuntimeError(
                f"Failed to load {input_name} image for ID={identifier!r}, path={path}"
            ) from exc

    @staticmethod
    def _optional_int(row: pd.Series, key: str, default: int = -1) -> int:
        if key not in row or pd.isna(row[key]):
            return default
        return int(row[key])

    @staticmethod
    def _optional_str(row: pd.Series, key: str, default: str = "") -> str:
        if key not in row or pd.isna(row[key]):
            return default
        return str(row[key])

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.frame.iloc[index]
        identifier = str(row["ID"])
        do_flip = self.train and self.horizontal_flip and random.random() < 0.5

        global_path = self.global_image_path(identifier)
        item: dict[str, Any] = {
            "global_image": self._load_image(
                global_path, identifier, "global", do_flip
            ),
            "label": int(row[self.label_col]),
            "ID": identifier,
            "patient_group_id": str(row["patient_group_id"]),
            "global_image_path": str(global_path),
            "NYHA": self._optional_int(row, "NYHA"),
            "SEX": self._optional_int(row, "SEX"),
            "sex_name": self._optional_str(row, "sex_name"),
            "label_3class": int(row[self.label_col]),
            "label_3class_name": self._optional_str(row, "label_3class_name"),
            "fold": self._optional_int(row, "fold"),
        }

        image_path_parts = [str(global_path)]
        for roi_name in self.enabled_roi_names:
            path = self.roi_image_path(roi_name, identifier)
            item[f"{roi_name}_image"] = self._load_image(
                path, identifier, roi_name, do_flip
            )
            item[f"{roi_name}_image_path"] = str(path)
            image_path_parts.append(str(path))
        # Keep a generic image_path field for compatibility with existing
        # downstream utilities. It records every input path used by this sample.
        item["image_path"] = ";".join(image_path_parts)
        return item
