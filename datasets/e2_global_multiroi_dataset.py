"""Synchronized Global + five-ROI dataset with explicit ID/path auditing for E2."""

from __future__ import annotations

import random
from pathlib import Path
from typing import Any, Mapping, Sequence

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import functional as TF


ROI_ORDER = ("eye", "lip", "cheek", "forehead", "chin")
REQUIRED_COLUMNS = {
    "ID",
    "patient_group_id",
    "NYHA",
    "label_3class",
    "label_3class_name",
    "fold",
}
LABEL_NAMES = {0: "normal", 1: "mild", 2: "severe"}
NYHA_TO_LABEL = {0: 0, 1: 1, 2: 1, 3: 2, 4: 2}


class E2GlobalMultiROIDataset(Dataset):
    """Load six ID-aligned images, applying one shared geometric flip decision."""

    def __init__(
        self, csv_path: str | Path, global_root: str | Path,
        roi_roots: Mapping[str, str | Path], *, image_filename_template: str = "{ID}.png",
        image_size: int = 224, train: bool = False, horizontal_flip: bool = False,
        mean: Sequence[float] = (0.485, 0.456, 0.406),
        std: Sequence[float] = (0.229, 0.224, 0.225), validate_paths: bool = True,
    ) -> None:
        self.csv_path, self.global_root = Path(csv_path).resolve(), Path(global_root).resolve()
        missing_roi_keys = sorted(set(ROI_ORDER).difference(roi_roots))
        extra_roi_keys = sorted(set(roi_roots).difference(ROI_ORDER))
        if missing_roi_keys or extra_roi_keys:
            raise ValueError(
                "E2 ROI roots must contain exactly "
                f"{ROI_ORDER}; missing={missing_roi_keys}, extra={extra_roi_keys}"
            )
        self.roi_roots = {name: Path(roi_roots[name]).resolve() for name in ROI_ORDER}
        self.template, self.image_size, self.train, self.horizontal_flip = image_filename_template, int(image_size), bool(train), bool(horizontal_flip)
        self.mean, self.std = list(mean), list(std)
        if not self.csv_path.is_file() or not self.global_root.is_dir():
            raise FileNotFoundError(f"Missing E2 split or Global root: {self.csv_path}, {self.global_root}")
        missing_roots = [name for name, path in self.roi_roots.items() if not path.is_dir()]
        if missing_roots:
            raise FileNotFoundError(f"Missing E2 ROI roots: {[(name, str(self.roi_roots[name])) for name in missing_roots]}")
        self.frame = pd.read_csv(self.csv_path, dtype={"ID": "string", "patient_group_id": "string"}, encoding="utf-8-sig")
        missing = sorted(REQUIRED_COLUMNS.difference(self.frame.columns))
        if missing or self.frame.empty:
            raise ValueError(f"Invalid E2 split CSV {self.csv_path}; missing={missing}, empty={self.frame.empty}")
        labels = pd.to_numeric(self.frame["label_3class"], errors="coerce")
        if labels.isna().any() or not labels.isin([0, 1, 2]).all():
            raise ValueError(f"Invalid label_3class in {self.csv_path}")
        self.frame["label_3class"] = labels.astype(int)
        nyha = pd.to_numeric(self.frame["NYHA"], errors="coerce")
        expected_labels = nyha.map(NYHA_TO_LABEL)
        if expected_labels.isna().any() or not (
            self.frame["label_3class"] == expected_labels.astype(int)
        ).all():
            raise ValueError(
                f"NYHA to label_3class mapping is inconsistent in {self.csv_path}"
            )
        expected_names = self.frame["label_3class"].map(LABEL_NAMES)
        if not (self.frame["label_3class_name"].astype(str) == expected_names).all():
            raise ValueError(
                f"label_3class_name is inconsistent in {self.csv_path}"
            )
        if self.frame["ID"].isna().any() or self.frame["patient_group_id"].isna().any():
            raise ValueError(f"ID or patient_group_id is missing in {self.csv_path}")
        if self.frame["ID"].astype(str).duplicated().any():
            duplicate = self.frame.loc[self.frame["ID"].astype(str).duplicated(keep=False), "ID"].iloc[0]
            raise ValueError(f"Duplicate sample_id in E2 split: {duplicate}")
        if validate_paths:
            self.validate_paths()

    @property
    def labels(self) -> list[int]:
        return self.frame["label_3class"].tolist()

    def __len__(self) -> int:
        return len(self.frame)

    def _paths(self, identifier: str) -> dict[str, Path]:
        filename = self.template.format(ID=identifier)
        return {"global": self.global_root / filename, **{name: self.roi_roots[name] / filename for name in ROI_ORDER}}

    def manifest(self, split: str, run_fold: int | None = None) -> pd.DataFrame:
        """Return explicit input provenance for one outer-fold train/val role."""
        records = []
        for row in self.frame.itertuples(index=False):
            paths = self._paths(str(row.ID))
            records.append(
                {
                    "sample_id": str(row.ID),
                    "patient_group_id": str(row.patient_group_id),
                    "fold": int(row.fold),
                    "run_fold": int(row.fold if run_fold is None else run_fold),
                    "split": split,
                    "label": int(row.label_3class),
                    "label_name": str(row.label_3class_name),
                    "global_path": str(paths["global"]),
                    **{f"{name}_path": str(paths[name]) for name in ROI_ORDER},
                }
            )
        return pd.DataFrame(records)

    def validate_paths(self) -> None:
        for row in self.frame.itertuples(index=False):
            identifier, paths = str(row.ID), self._paths(str(row.ID))
            missing = [(name, str(path)) for name, path in paths.items() if not path.is_file()]
            if missing:
                raise FileNotFoundError(f"Missing E2 image(s) for ID={identifier}: {missing}")
            roi_paths = [str(paths[name]) for name in ROI_ORDER]
            if len(set(roi_paths)) != len(roi_paths):
                raise ValueError(f"Duplicate ROI paths for ID={identifier}: {roi_paths}")

    def _load(self, path: Path, flip: bool) -> torch.Tensor:
        with Image.open(path) as image:
            image = TF.resize(image.convert("RGB"), [self.image_size, self.image_size])
            if flip:
                image = TF.hflip(image)
            return TF.normalize(TF.to_tensor(image), mean=self.mean, std=self.std)

    def __getitem__(self, index: int) -> dict[str, Any]:
        row, identifier = self.frame.iloc[index], str(self.frame.iloc[index]["ID"])
        paths = self._paths(identifier)
        flip = self.train and self.horizontal_flip and random.random() < 0.5
        try:
            roi_images = torch.stack([self._load(paths[name], flip) for name in ROI_ORDER])
            global_image = self._load(paths["global"], flip)
        except Exception as error:
            raise RuntimeError(f"Failed to load synchronized E2 inputs for ID={identifier}: {paths}") from error
        return {"global_image": global_image, "roi_images": roi_images, "label": int(row["label_3class"]), "sample_id": identifier, "patient_group_id": str(row["patient_group_id"]), "fold": int(row["fold"]), "global_path": str(paths["global"]), **{f"{name}_path": str(paths[name]) for name in ROI_ORDER}}
