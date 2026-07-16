"""Global RGB plus masked Eye/Cheek optical inputs for P2-1."""

from __future__ import annotations

import random
from pathlib import Path
from typing import Any, Sequence

import cv2
import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import functional as TF

from utils.relative_optical_channels import (
    build_relative_optical_channels,
    normalize_optical_channels,
)


class RelativeEyeCheekDataset(Dataset):
    """Use one flip flag for global, both ROI images, and both masks."""

    def __init__(
        self,
        csv_path: str | Path,
        global_image_root: str | Path,
        eye_image_root: str | Path,
        cheek_image_root: str | Path,
        eye_mask_root: str | Path,
        cheek_mask_root: str | Path,
        optical_mean: Sequence[float],
        optical_std: Sequence[float],
        image_size: int = 224,
        train: bool = False,
        horizontal_flip: bool = True,
        epsilon: float = 1.0e-4,
        filename_template: str = "{ID}.png",
    ) -> None:
        self.csv_path = Path(csv_path)
        self.frame = pd.read_csv(
            self.csv_path, dtype={"ID": "string", "patient_group_id": "string"}, encoding="utf-8-sig"
        )
        required = {"ID", "patient_group_id", "NYHA", "SEX", "label_3class", "fold"}
        missing = sorted(required.difference(self.frame.columns))
        if missing:
            raise ValueError(f"split missing columns {missing}: {self.csv_path}")
        self.roots = {
            "global": Path(global_image_root),
            "eye": Path(eye_image_root),
            "cheek": Path(cheek_image_root),
            "eye_mask": Path(eye_mask_root),
            "cheek_mask": Path(cheek_mask_root),
        }
        self.mean = list(map(float, optical_mean))
        self.std = list(map(float, optical_std))
        self.image_size = int(image_size)
        self.train = bool(train)
        self.horizontal_flip = bool(horizontal_flip)
        self.epsilon = float(epsilon)
        self.filename_template = filename_template

    def __len__(self) -> int:
        return len(self.frame)

    def _path(self, root: str, identifier: str) -> Path:
        return self.roots[root] / self.filename_template.format(ID=identifier)

    def _load_rgb(self, path: Path) -> torch.Tensor:
        if not path.is_file():
            raise FileNotFoundError(path)
        image = Image.open(path).convert("RGB")
        image = TF.resize(image, [self.image_size, self.image_size], antialias=True)
        return TF.to_tensor(image)

    def _load_mask(self, path: Path) -> torch.Tensor:
        if not path.is_file():
            raise FileNotFoundError(path)
        mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise ValueError(f"cannot read mask: {path}")
        if mask.shape != (self.image_size, self.image_size):
            mask = cv2.resize(mask, (self.image_size, self.image_size), interpolation=cv2.INTER_NEAREST)
        return torch.from_numpy((mask > 0).astype(np.float32))[None]

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.frame.iloc[index]
        identifier = str(row["ID"])
        global_rgb = self._load_rgb(self._path("global", identifier))
        eye_rgb = self._load_rgb(self._path("eye", identifier))
        cheek_rgb = self._load_rgb(self._path("cheek", identifier))
        eye_mask = self._load_mask(self._path("eye_mask", identifier))
        cheek_mask = self._load_mask(self._path("cheek_mask", identifier))
        do_flip = self.train and self.horizontal_flip and random.random() < 0.5
        if do_flip:
            global_rgb = TF.hflip(global_rgb)
            eye_rgb = TF.hflip(eye_rgb)
            cheek_rgb = TF.hflip(cheek_rgb)
            eye_mask = TF.hflip(eye_mask)
            cheek_mask = TF.hflip(cheek_mask)
        eye_optical = normalize_optical_channels(
            build_relative_optical_channels(eye_rgb, eye_mask, self.epsilon),
            eye_mask,
            self.mean,
            self.std,
            self.epsilon,
        )
        cheek_optical = normalize_optical_channels(
            build_relative_optical_channels(cheek_rgb, cheek_mask, self.epsilon),
            cheek_mask,
            self.mean,
            self.std,
            self.epsilon,
        )
        global_image = TF.normalize(
            global_rgb,
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        )
        return {
            "global_image": global_image,
            "eye_optical": eye_optical,
            "cheek_optical": cheek_optical,
            "eye_mask": eye_mask,
            "cheek_mask": cheek_mask,
            "label": torch.tensor(int(row["label_3class"]), dtype=torch.long),
            "ID": identifier,
            "patient_group_id": str(row["patient_group_id"]),
            "NYHA": int(row["NYHA"]),
            "SEX": int(row["SEX"]),
            "fold": int(row["fold"]),
            "flip_applied": do_flip,
        }


def raw_roi_and_mask_samples(
    frame: pd.DataFrame,
    eye_root: Path,
    cheek_root: Path,
    eye_mask_root: Path,
    cheek_mask_root: Path,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Read raw [0,1] RGB and binary masks for train-only statistics."""
    samples: list[tuple[np.ndarray, np.ndarray]] = []
    for identifier in frame["ID"].astype(str):
        for image_root, mask_root in ((eye_root, eye_mask_root), (cheek_root, cheek_mask_root)):
            rgb = cv2.cvtColor(cv2.imread(str(image_root / f"{identifier}.png")), cv2.COLOR_BGR2RGB)
            mask = cv2.imread(str(mask_root / f"{identifier}.png"), cv2.IMREAD_GRAYSCALE)
            if rgb is None or mask is None:
                raise FileNotFoundError(f"missing ROI or mask for one split sample")
            if rgb.shape[:2] != (224, 224):
                rgb = cv2.resize(rgb, (224, 224), interpolation=cv2.INTER_AREA)
            if mask.shape != (224, 224):
                mask = cv2.resize(mask, (224, 224), interpolation=cv2.INTER_NEAREST)
            samples.append((np.transpose(rgb.astype(np.float32) / 255.0, (2, 0, 1)), mask > 0))
    return samples
