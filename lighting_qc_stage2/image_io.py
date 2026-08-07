from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image, ImageOps


def read_rgb(path: Path) -> np.ndarray:
    with Image.open(path) as im:
        return np.asarray(im.convert("RGB"), dtype=np.uint8)


def read_gray(path: Path) -> np.ndarray:
    with Image.open(path) as im:
        return np.asarray(im.convert("L"), dtype=np.uint8)


def read_source_rgb_exif(path: Path) -> tuple[np.ndarray, bool]:
    with Image.open(path) as im:
        before = im.size
        transposed = ImageOps.exif_transpose(im)
        applied = transposed.size != before or transposed is not im
        return np.asarray(transposed.convert("RGB"), dtype=np.uint8), bool(applied)


def save_gray(path: Path, mask: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(mask.astype(np.uint8), mode="L").save(path)


def save_rgb(path: Path, rgb: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(rgb.astype(np.uint8), mode="RGB").save(path)


def find_raw(sample_id: str, raw_dir: Path) -> Path:
    for ext in (".jpg", ".jpeg", ".png", ".bmp"):
        path = raw_dir / f"{sample_id}{ext}"
        if path.is_file():
            return path
    raise FileNotFoundError(f"raw image not found for {sample_id}")
