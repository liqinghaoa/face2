"""Hyper-Skin VIS discovery, leakage checks, and region-spectrum extraction."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import h5py
import numpy as np
from PIL import Image


OFFICIAL_SPLITS = ("train", "valid", "test")
VIS_DIRECTORY_NAMES = ("VIS", "HSI_VIS")


@dataclass(frozen=True)
class HyperSkinSample:
    sample_id: str
    subject_id: str
    split: str
    hsi_path: Path
    mask_path: Path | None
    rgb_path: Path | None = None
    expression: str = ""
    direction: str = ""

    def record(self) -> dict[str, Any]:
        data = asdict(self)
        data["hsi_path"] = str(self.hsi_path)
        data["mask_path"] = str(self.mask_path) if self.mask_path else None
        data["rgb_path"] = str(self.rgb_path) if self.rgb_path else None
        return data


def _subject_id(stem: str) -> str:
    first = stem.split("_", maxsplit=1)[0]
    if not first.lower().startswith("p") or len(first) < 2:
        raise ValueError(f"Cannot parse Hyper-Skin subject ID from: {stem}")
    return first.lower()


def parse_hyperskin_sample_id(stem: str) -> tuple[str, str, str]:
    """Parse ``pNNN_expression_direction`` without silently accepting ambiguity."""

    parts = stem.lower().split("_")
    if len(parts) != 3:
        raise ValueError(f"Expected pNNN_expression_direction sample ID, got: {stem}")
    subject_id, expression, direction = parts
    _subject_id(stem)
    if expression not in {"neutral", "smile"}:
        raise ValueError(f"Unknown Hyper-Skin expression in: {stem}")
    if direction not in {"front", "left", "right"}:
        raise ValueError(f"Unknown Hyper-Skin direction in: {stem}")
    return subject_id, expression, direction


def _find_rgb(split_root: Path, stem: str) -> Path | None:
    rgb_root = split_root / "RGB"
    for suffix in (".jpg", ".jpeg", ".png", ".tif", ".tiff"):
        candidate = rgb_root / f"{stem}{suffix}"
        if candidate.is_file():
            return candidate.resolve()
    return None


def _find_mask(mask_root: Path | None, split: str, stem: str) -> Path | None:
    if mask_root is None:
        return None
    candidates: list[Path] = []
    for suffix in (".npy", ".png", ".tif", ".tiff"):
        candidates.extend((mask_root / split / f"{stem}{suffix}", mask_root / f"{stem}{suffix}"))
    return next((path.resolve() for path in candidates if path.is_file()), None)


def discover_hyperskin_vis(data_root: str | Path, mask_root: str | Path | None = None) -> list[HyperSkinSample]:
    root = Path(data_root).resolve()
    masks = Path(mask_root).resolve() if mask_root is not None else None
    samples: list[HyperSkinSample] = []
    for split in OFFICIAL_SPLITS:
        split_root = root / split
        vis_root = next((split_root / name for name in VIS_DIRECTORY_NAMES if (split_root / name).is_dir()), None)
        if vis_root is None:
            continue
        for hsi_path in sorted(vis_root.glob("*.mat")):
            subject_id, expression, direction = parse_hyperskin_sample_id(hsi_path.stem)
            samples.append(
                HyperSkinSample(
                    sample_id=hsi_path.stem,
                    subject_id=subject_id,
                    split=split,
                    hsi_path=hsi_path.resolve(),
                    mask_path=_find_mask(masks, split, hsi_path.stem),
                    rgb_path=_find_rgb(split_root, hsi_path.stem),
                    expression=expression,
                    direction=direction,
                )
            )
    return samples


def load_hyperskin_cube(path: str | Path, dataset_key: str = "cube", expected_bands: int = 31) -> np.ndarray:
    """Load the v7.3 MAT/HDF5 cube into [height, width, bands]."""

    with h5py.File(Path(path), "r") as archive:
        if dataset_key not in archive:
            raise ValueError(f"MAT file does not contain dataset key '{dataset_key}': {path}")
        cube = np.squeeze(np.asarray(archive[dataset_key], dtype=np.float32))
    if cube.ndim != 3:
        raise ValueError(f"Expected a three-dimensional HSI cube, got {cube.shape}")
    band_axes = [index for index, size in enumerate(cube.shape) if size == expected_bands]
    if len(band_axes) != 1:
        raise ValueError(f"Cannot identify the {expected_bands}-band axis in shape {cube.shape}")
    cube = np.moveaxis(cube, band_axes[0], -1)
    return np.ascontiguousarray(cube)


def load_binary_mask(path: str | Path, expected_shape: tuple[int, int]) -> np.ndarray:
    mask_path = Path(path)
    if mask_path.suffix.lower() == ".npy":
        mask = np.load(mask_path, allow_pickle=False)
    else:
        with Image.open(mask_path) as image:
            mask = np.asarray(image.convert("L"))
    mask = np.squeeze(mask)
    if mask.shape != expected_shape:
        raise ValueError(f"Mask shape {mask.shape} does not match cube shape {expected_shape}: {mask_path}")
    return np.asarray(mask > 0, dtype=bool)


def extract_mean_skin_spectrum(
    cube: np.ndarray,
    mask: np.ndarray,
    reflectance_epsilon: float,
    reflectance_max: float,
) -> tuple[np.ndarray, dict[str, Any]]:
    if cube.ndim != 3 or mask.shape != cube.shape[:2]:
        raise ValueError("Cube and mask shapes are incompatible")
    pixels = cube[mask]
    if pixels.size == 0:
        raise ValueError("Skin mask contains no valid pixels")
    valid = np.isfinite(pixels) & (pixels > reflectance_epsilon) & (pixels <= reflectance_max)
    spectrum = np.full(cube.shape[-1], np.nan, dtype=np.float64)
    for band in range(cube.shape[-1]):
        band_values = pixels[:, band]
        band_valid = valid[:, band]
        if np.any(band_valid):
            spectrum[band] = float(np.median(band_values[band_valid]))
    qc = {
        "mask_pixel_count": int(mask.sum()),
        "valid_value_fraction": float(valid.mean()),
        "valid_band_count": int(np.isfinite(spectrum).sum()),
        "aggregation": "per-band median over supplied skin mask",
    }
    return spectrum, qc


def audit_data_contract(
    samples: list[HyperSkinSample],
    expected_counts: dict[str, int] | None,
    require_masks: bool,
    require_rgb: bool = False,
) -> dict[str, Any]:
    issues: list[dict[str, str]] = []
    split_counts = {split: sum(sample.split == split for sample in samples) for split in OFFICIAL_SPLITS}
    if not samples:
        issues.append({"severity": "critical", "code": "NO_HSI", "message": "No Hyper-Skin VIS MAT files found"})
    if expected_counts is not None:
        for split, expected in expected_counts.items():
            if split_counts.get(split, 0) != int(expected):
                issues.append(
                    {
                        "severity": "critical",
                        "code": "COUNT_MISMATCH",
                        "message": f"{split}: expected {expected}, found {split_counts.get(split, 0)}",
                    }
                )
    subjects_by_split = {
        split: sorted({sample.subject_id for sample in samples if sample.split == split}) for split in OFFICIAL_SPLITS
    }
    for index, split_a in enumerate(OFFICIAL_SPLITS):
        for split_b in OFFICIAL_SPLITS[index + 1 :]:
            overlap = sorted(set(subjects_by_split[split_a]).intersection(subjects_by_split[split_b]))
            if overlap:
                issues.append(
                    {
                        "severity": "critical",
                        "code": "SUBJECT_LEAKAGE",
                        "message": f"Subjects overlap between {split_a} and {split_b}: {overlap}",
                    }
                )
    missing_masks = [sample.sample_id for sample in samples if sample.mask_path is None]
    if require_masks and missing_masks:
        issues.append(
            {
                "severity": "critical",
                "code": "MISSING_SKIN_MASKS",
                "message": f"Missing masks for {len(missing_masks)} samples",
            }
        )
    missing_rgb = [sample.sample_id for sample in samples if sample.rgb_path is None]
    if require_rgb and missing_rgb:
        issues.append(
            {
                "severity": "critical",
                "code": "MISSING_PAIRED_RGB",
                "message": f"Missing paired RGB files for {len(missing_rgb)} samples",
            }
        )
    return {
        "status": "PASS" if not any(issue["severity"] == "critical" for issue in issues) else "FAIL",
        "physical_quantity": "calibrated reflectance (dataset documentation; verify downloaded files)",
        "wavelength_contract": {"start_nm": 400, "stop_nm": 700, "step_nm": 10, "bands": 31},
        "sample_count": len(samples),
        "split_counts": split_counts,
        "subjects_by_split": subjects_by_split,
        "subject_counts": {key: len(value) for key, value in subjects_by_split.items()},
        "missing_mask_count": len(missing_masks),
        "missing_rgb_count": len(missing_rgb),
        "issues": issues,
    }
