"""Traceable ROI masks reconstructed with the original ROI geometry metadata."""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd


def _read_binary(path: Path) -> np.ndarray:
    value = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if value is None:
        raise FileNotFoundError(path)
    return (value > 0).astype(np.uint8) * 255


def _bbox(row: pd.Series, prefix: str = "") -> tuple[int, int, int, int]:
    def value(name: str) -> int:
        return int(round(float(row[f"{prefix}{name}"])))
    return value("bbox_x1"), value("bbox_y1"), value("bbox_x2") + 1, value("bbox_y2") + 1


def _crop_canvas(mask: np.ndarray, box: tuple[int, int, int, int], size: int = 224) -> np.ndarray:
    x1, y1, x2, y2 = box
    crop = mask[y1:y2, x1:x2]
    scale = min(size / crop.shape[1], size / crop.shape[0])
    width, height = max(1, round(crop.shape[1] * scale)), max(1, round(crop.shape[0] * scale))
    resized = cv2.resize(crop, (width, height), interpolation=cv2.INTER_NEAREST)
    canvas = np.zeros((size, size), dtype=np.uint8)
    left, top = (size - width) // 2, (size - height) // 2
    canvas[top : top + height, left : left + width] = resized
    return canvas


def _cheek_canvas(mask: np.ndarray, row: pd.Series, size: int = 224) -> np.ndarray:
    halves = []
    for prefix, width in (("left_cheek_", size // 2), ("right_cheek_", size - size // 2)):
        x1, y1, x2, y2 = _bbox(row, prefix)
        crop = mask[y1:y2, x1:x2]
        halves.append(cv2.resize(crop, (width, size), interpolation=cv2.INTER_NEAREST))
    return np.concatenate(halves, axis=1)


def build_traceable_roi_masks(
    ids: list[str],
    metadata_csv: Path,
    final_mask_root: Path,
    output_root: Path,
) -> dict:
    """Create new binary masks; never alters source masks or ROI images."""
    eye_dir, cheek_dir = output_root / "eye_mask", output_root / "cheek_mask"
    eye_dir.mkdir(parents=True, exist_ok=True)
    cheek_dir.mkdir(parents=True, exist_ok=True)
    metadata = pd.read_csv(metadata_csv, dtype={"ID": "string"})
    metadata["ID"] = metadata["ID"].astype(str)
    lookup = {(row.ID, row.roi_type): row for row in metadata.itertuples(index=False)}
    for identifier in ids:
        mask = _read_binary(final_mask_root / f"{identifier}.png")
        eye = pd.Series(lookup[(identifier, "eye_roi")]._asdict())
        cheek = pd.Series(lookup[(identifier, "cheek_roi")]._asdict())
        eye_mask = _crop_canvas(mask, _bbox(eye))
        cheek_mask = _cheek_canvas(mask, cheek)
        if not cv2.imwrite(str(eye_dir / f"{identifier}.png"), eye_mask):
            raise OSError("failed to write eye mask")
        if not cv2.imwrite(str(cheek_dir / f"{identifier}.png"), cheek_mask):
            raise OSError("failed to write cheek mask")
    payload = {
        "source_final_mask_root": str(final_mask_root.resolve()),
        "source_roi_metadata": str(metadata_csv.resolve()),
        "mask_type": "binary_uint8_0_255",
        "interpolation": "cv2.INTER_NEAREST",
        "geometry": "original ROI bbox/canvas rules from preprocess_global_aligned_face_parsing_roi_dataset_224_canvas.py",
        "feathered_edges": False,
        "sample_count": len(ids),
    }
    (output_root / "mask_manifest.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return payload
