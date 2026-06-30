"""Multi-ROI dataset for NYHA three-class feature-level fusion experiments."""

from __future__ import annotations

import random
from pathlib import Path
from typing import Any, Sequence

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import functional as TF


REQUIRED_COLUMNS = {
    "ID",
    "patient_group_id",
    "NYHA",
    "label_3class",
    "SEX",
    "fold",
}


class NYHA3ClassMultiROIDataset(Dataset):
    """Load multiple ROI images per sample using IDs from a fixed split CSV."""

    def __init__(
        self,
        csv_path: str | Path,
        roi_root: str | Path,
        roi_names: Sequence[str],
        image_filename_template: str = "{ID}.png",
        image_size: int = 224,
        label_col: str = "label_3class",
        train: bool = False,
        horizontal_flip: bool = False,
        same_flip_for_all_rois: bool = True,
        mean: Sequence[float] = (0.485, 0.456, 0.406),
        std: Sequence[float] = (0.229, 0.224, 0.225),
    ) -> None:
        self.csv_path = Path(csv_path).expanduser().resolve()
        self.roi_root = Path(roi_root).expanduser().resolve()
        self.roi_names = list(roi_names)
        self.image_filename_template = image_filename_template
        self.image_size = int(image_size)
        self.label_col = label_col
        self.train = bool(train)
        self.horizontal_flip = bool(horizontal_flip)
        self.same_flip_for_all_rois = bool(same_flip_for_all_rois)
        self.mean = list(mean)
        self.std = list(std)

        if not self.csv_path.is_file():
            raise FileNotFoundError(f"Dataset CSV does not exist: {self.csv_path}")
        if not self.roi_root.is_dir():
            raise FileNotFoundError(f"ROI root does not exist: {self.roi_root}")
        if len(self.roi_names) < 2:
            raise ValueError("Multi-ROI dataset requires at least two ROI names")
        if len(set(self.roi_names)) != len(self.roi_names):
            raise ValueError(f"ROI names contain duplicates: {self.roi_names}")
        if self.label_col not in REQUIRED_COLUMNS:
            required = set(REQUIRED_COLUMNS)
            required.add(self.label_col)
        else:
            required = set(REQUIRED_COLUMNS)

        self.frame = pd.read_csv(
            self.csv_path,
            dtype={"ID": "string", "patient_group_id": "string"},
            encoding="utf-8-sig",
        )
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
    def labels(self) -> list[int]:
        return self.frame[self.label_col].astype(int).tolist()

    def __len__(self) -> int:
        return len(self.frame)

    def _roi_path(self, roi_name: str, identifier: str) -> Path:
        filename = self.image_filename_template.format(ID=identifier)
        return self.roi_root / roi_name / filename

    def _load_roi_image(self, path: Path, do_flip: bool) -> torch.Tensor:
        with Image.open(path) as image:
            image = image.convert("RGB")
            image = TF.resize(image, [self.image_size, self.image_size])
            if do_flip:
                image = TF.hflip(image)
            tensor = TF.to_tensor(image)
            return TF.normalize(tensor, mean=self.mean, std=self.std)

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.frame.iloc[index]
        identifier = str(row["ID"])
        do_flip = self.train and self.horizontal_flip and random.random() < 0.5
        if not self.same_flip_for_all_rois and self.train and self.horizontal_flip:
            raise ValueError(
                "same_flip_for_all_rois=false is not supported; all ROIs in a "
                "sample must share the same horizontal flip decision"
            )

        roi_tensors: list[torch.Tensor] = []
        image_paths: dict[str, str] = {}
        for roi_name in self.roi_names:
            path = self._roi_path(roi_name, identifier)
            if not path.is_file():
                raise FileNotFoundError(
                    f"Missing ROI image for ID={identifier!r}, roi={roi_name!r}: {path}"
                )
            try:
                roi_tensors.append(self._load_roi_image(path, do_flip))
            except Exception as exc:
                raise RuntimeError(
                    f"Failed to load ROI image for ID={identifier!r}, "
                    f"roi={roi_name!r}, path={path}"
                ) from exc
            image_paths[roi_name] = str(path)

        label = int(row[self.label_col])
        meta = {
            "ID": identifier,
            "patient_group_id": str(row["patient_group_id"]),
            "NYHA": int(row["NYHA"]),
            "label_3class": label,
            "SEX": int(row["SEX"]),
            "fold": int(row["fold"]),
            "roi_names": list(self.roi_names),
            "roi_image_paths": image_paths,
        }
        if "sex_name" in self.frame.columns:
            meta["sex_name"] = str(row["sex_name"])
        if "label_3class_name" in self.frame.columns:
            meta["label_3class_name"] = str(row["label_3class_name"])

        return {
            "image": torch.stack(roi_tensors, dim=0),
            "label": label,
            "ID": meta["ID"],
            "patient_group_id": meta["patient_group_id"],
            "image_path": ";".join(image_paths[roi] for roi in self.roi_names),
            "roi_image_paths": image_paths,
            "NYHA": meta["NYHA"],
            "SEX": meta["SEX"],
            "sex_name": meta.get("sex_name", ""),
            "label_3class_name": meta.get("label_3class_name", ""),
            "fold": meta["fold"],
            "meta": meta,
        }
