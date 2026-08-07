"""Lazy dataset and representation adapter for frozen P1 assets."""
from __future__ import annotations

import json
from collections import OrderedDict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset


LATENT_KEY_MAP = {
    "shape_code": "shape_code",
    "tex_code": "tex_code",
    "detail_code": "detail_code",
    "exp_code": "expression_code",
    "pose_code": "pose_code",
    "cam_code": "camera_code",
    "light_code": "light_code",
}

MAP_KEYS = (
    "input_aligned_rgb",
    "albedo_like",
    "normal_coarse",
    "shading_like",
    "reconstruction",
    "signed_residual",
    "absolute_residual",
)


class UnavailableRepresentationError(ValueError):
    """Raised when a requested P1 representation is not available."""


def _as_path(value: Any) -> Path:
    return value if isinstance(value, Path) else Path(str(value))


def _image_to_chw_float32(path: Path) -> torch.Tensor:
    with Image.open(path) as image:
        array = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
    return torch.from_numpy(np.transpose(array, (2, 0, 1)).copy())


def _mask_to_1hw(path: Path) -> torch.Tensor:
    with Image.open(path) as image:
        array = np.asarray(image.convert("L"), dtype=np.uint8) > 0
    return torch.from_numpy(array[None, :, :].astype(np.float32))


def _hwc_to_chw(array: np.ndarray) -> torch.Tensor:
    arr = np.asarray(array, dtype=np.float32)
    if arr.ndim != 3:
        raise ValueError(f"expected HWC array, got shape {arr.shape}")
    return torch.from_numpy(np.transpose(arr, (2, 0, 1)).copy())


class P1FrozenAssetDataset(Dataset):
    """Read frozen P1 assets lazily from a master manifest.

    The base dataset performs no clipping, resizing, z-scoring, ImageNet
    normalization, missing-value filling, or fold-specific fitting.
    """

    def __init__(self, manifest_path: str | Path, cache_size: int = 0):
        self.manifest_path = _as_path(manifest_path)
        self.rows = pd.read_csv(self.manifest_path, dtype={"case_id": str, "patient_id": str, "group_id": str})
        self.cache_size = int(cache_size)
        self._cache: OrderedDict[int, dict[str, Any]] = OrderedDict()

    def __len__(self) -> int:
        return int(len(self.rows))

    def _load_json(self, path: Path) -> dict[str, Any]:
        return json.loads(path.read_text(encoding="utf-8"))

    def _load_item_uncached(self, index: int) -> dict[str, Any]:
        row = self.rows.iloc[index]
        maps_path = Path(row["maps_path"])
        latents_path = Path(row["latents_path"])
        with np.load(maps_path, allow_pickle=False) as maps, np.load(latents_path, allow_pickle=False) as latents:
            item: dict[str, Any] = {
                "case_id": str(row["case_id"]),
                "patient_id": str(row["patient_id"]),
                "group_id": str(row["group_id"]),
                "fold": int(row["fold"]),
                "label_original": int(row["label_original"]),
                "label_3class": None if pd.isna(row.get("label_3class")) else int(row["label_3class"]),
                "label_binary": int(row["label_binary"]),
                "rgb": _image_to_chw_float32(Path(row["rgb_path"])),
                "face_valid": _mask_to_1hw(Path(row["face_valid_mask_path"])),
                "skin_strict": _mask_to_1hw(Path(row["skin_strict_mask_path"])),
                "physics_core_skin": _mask_to_1hw(Path(row["physics_core_skin_mask_path"])),
                "camera_exif": {
                    "camera_model": None if pd.isna(row.get("camera_model")) else row.get("camera_model"),
                    "exposure_time_raw": None if pd.isna(row.get("exposure_time_raw")) else row.get("exposure_time_raw"),
                    "exposure_time_seconds": None if pd.isna(row.get("exposure_time_seconds")) else row.get("exposure_time_seconds"),
                    "log_exposure_time": None if pd.isna(row.get("log_exposure_time")) else row.get("log_exposure_time"),
                    "fnumber": None if pd.isna(row.get("fnumber")) else row.get("fnumber"),
                    "iso_raw": None if pd.isna(row.get("iso_raw")) else row.get("iso_raw"),
                    "iso_numeric": None if pd.isna(row.get("iso_numeric")) else row.get("iso_numeric"),
                    "log_iso": None if pd.isna(row.get("log_iso")) else row.get("log_iso"),
                    "brightness_value": None if pd.isna(row.get("brightness_value")) else row.get("brightness_value"),
                    "datetime_original": None if pd.isna(row.get("datetime_original")) else row.get("datetime_original"),
                    "shooting_time_period": None if pd.isna(row.get("shooting_time_period")) else row.get("shooting_time_period"),
                },
                "quality": self._load_json(Path(row["quality_path"])),
            }
            for key in MAP_KEYS:
                if key in maps.files:
                    out_key = "rgb" if key == "input_aligned_rgb" else key
                    if out_key != "rgb":
                        item[out_key] = _hwc_to_chw(maps[key])
            for out_key, npz_key in LATENT_KEY_MAP.items():
                if npz_key in latents.files:
                    item[out_key] = torch.from_numpy(np.asarray(latents[npz_key], dtype=np.float32).copy()).squeeze(0)
        return item

    def __getitem__(self, index: int) -> dict[str, Any]:
        if self.cache_size <= 0:
            return self._load_item_uncached(index)
        if index in self._cache:
            self._cache.move_to_end(index)
            return self._cache[index]
        item = self._load_item_uncached(index)
        self._cache[index] = item
        while len(self._cache) > self.cache_size:
            self._cache.popitem(last=False)
        return item


class P1RepresentationAdapter:
    """Return canonical P1 representations from a dataset item."""

    supported = {"rgb", "albedo", "normal", "light", "shading", "residual", "rgb_albedo"}

    def __call__(self, item: dict[str, Any], representation: str) -> Any:
        if representation == "specular":
            raise UnavailableRepresentationError("unavailable_by_current_frontend")
        if representation not in self.supported:
            raise ValueError(f"unsupported representation: {representation}")
        face_valid = item["face_valid"].to(dtype=torch.float32)
        physics_core = item["physics_core_skin"].to(dtype=torch.float32)
        if representation == "rgb":
            return item["rgb"]
        if representation == "albedo":
            return item["albedo_like"] * physics_core
        if representation == "normal":
            return item["normal_coarse"] * face_valid
        if representation == "light":
            return item["light_code"]
        if representation == "shading":
            return item["shading_like"] * face_valid
        if representation == "residual":
            return item["signed_residual"] * face_valid
        if representation == "rgb_albedo":
            return {"rgb": item["rgb"], "albedo": item["albedo_like"] * physics_core}
        raise AssertionError("unreachable")
