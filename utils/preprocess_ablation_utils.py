"""Utilities for preprocessing ablation image generation."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Iterable, Sequence

import cv2
import numpy as np
import pandas as pd


IMAGENET_MEAN_RGB = np.array([0.485, 0.456, 0.406], dtype=np.float32) * 255.0


def _decode_image(path: str | Path, flags: int) -> np.ndarray:
    image_path = Path(path)
    data = np.fromfile(str(image_path), dtype=np.uint8)
    image = cv2.imdecode(data, flags)
    if image is None:
        raise ValueError(f"Cannot decode image: {image_path}")
    return image


def read_rgb(path: str | Path) -> np.ndarray:
    """Read an image as uint8 RGB."""
    bgr = _decode_image(path, cv2.IMREAD_COLOR)
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    return np.ascontiguousarray(rgb.astype(np.uint8, copy=False))


def save_rgb(path: str | Path, arr: np.ndarray) -> None:
    """Save a uint8 RGB PNG with Windows Unicode-safe IO."""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    image = ensure_uint8_rgb(arr)
    bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    ok, encoded = cv2.imencode(".png", bgr, [cv2.IMWRITE_PNG_COMPRESSION, 3])
    if not ok:
        raise ValueError(f"cv2.imencode('.png') failed for {output_path}")
    output_path.write_bytes(encoded.tobytes())


def read_mask(path: str | Path) -> np.ndarray:
    """Read a single-channel mask and normalize it to 0/255 uint8."""
    mask = _decode_image(path, cv2.IMREAD_GRAYSCALE)
    return ((mask > 0).astype(np.uint8) * 255)


def ensure_uint8_rgb(arr: np.ndarray) -> np.ndarray:
    image = np.asarray(arr)
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(f"Expected HxWx3 RGB image, got shape={image.shape}")
    if image.dtype != np.uint8:
        image = np.clip(np.rint(image), 0, 255).astype(np.uint8)
    return np.ascontiguousarray(image)


def feather_mask(mask: np.ndarray, kernel_size: int = 11) -> np.ndarray:
    """Convert a binary mask to a feathered float alpha in [0, 1]."""
    binary = (mask > 0).astype(np.uint8) * 255
    if kernel_size <= 1:
        return binary.astype(np.float32) / 255.0
    if kernel_size % 2 == 0:
        kernel_size += 1
    sigma = max(1.0, max(binary.shape[:2]) * 0.013)
    alpha = cv2.GaussianBlur(binary, (kernel_size, kernel_size), sigmaX=sigma)
    return np.clip(alpha.astype(np.float32) / 255.0, 0.0, 1.0)


def apply_background(
    image_rgb: np.ndarray,
    alpha: np.ndarray,
    mode: str | Sequence[float] = "black",
) -> np.ndarray:
    """Blend an RGB image over black or ImageNet-mean background."""
    image = ensure_uint8_rgb(image_rgb).astype(np.float32)
    alpha_3 = np.asarray(alpha, dtype=np.float32)[:, :, None]
    if isinstance(mode, str):
        normalized = mode.lower()
        if normalized in {"black", "black_rgb_0_0_0"}:
            background = np.zeros(3, dtype=np.float32)
        elif normalized in {"imagenet_mean", "imagenet_mean_rgb"}:
            background = IMAGENET_MEAN_RGB
        else:
            raise ValueError(f"Unsupported background mode: {mode}")
    else:
        background = np.asarray(mode, dtype=np.float32)
        if background.shape != (3,):
            raise ValueError(f"Background RGB must have shape (3,), got {background}")
    output = image * alpha_3 + background.reshape(1, 1, 3) * (1.0 - alpha_3)
    return ensure_uint8_rgb(output)


def _masked_pixels(channel: np.ndarray, mask: np.ndarray) -> np.ndarray:
    pixels = channel[mask > 0].astype(np.float32)
    if pixels.size == 0:
        raise ValueError("Mask contains no foreground pixels")
    return pixels


def lab_l_norm(image_rgb: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Mild robust L-channel normalization estimated inside the final mask."""
    image = ensure_uint8_rgb(image_rgb)
    lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB).astype(np.float32)
    l_channel = lab[:, :, 0]
    pixels = _masked_pixels(l_channel, mask)
    p5, p50, p95 = np.percentile(pixels, [5, 50, 95])
    robust_std = max((p95 - p5) / 3.29, 1.0)
    target_median = 145.0
    target_std = 42.0
    scale = float(np.clip(target_std / robust_std, 0.75, 1.25))
    shift = float(np.clip(target_median - p50, -18.0, 18.0))
    lab[:, :, 0] = np.clip((l_channel - p50) * scale + p50 + shift, 0, 255)
    return cv2.cvtColor(lab.astype(np.uint8), cv2.COLOR_LAB2RGB)


def clahe_l(
    image_rgb: np.ndarray,
    mask: np.ndarray | None = None,
    clip_limit: float = 2.0,
    tile_grid_size: tuple[int, int] = (8, 8),
) -> np.ndarray:
    """Apply CLAHE to the Lab L channel."""
    del mask
    image = ensure_uint8_rgb(image_rgb)
    lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
    clahe = cv2.createCLAHE(clipLimit=float(clip_limit), tileGridSize=tile_grid_size)
    lab[:, :, 0] = clahe.apply(lab[:, :, 0])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)


def masked_grayworld_wb(
    image_rgb: np.ndarray,
    mask: np.ndarray,
    clip_range: tuple[float, float] = (0.8, 1.25),
) -> np.ndarray:
    """Apply clipped Gray-World white balance using only foreground pixels."""
    image = ensure_uint8_rgb(image_rgb).astype(np.float32)
    foreground = mask > 0
    if not foreground.any():
        raise ValueError("Mask contains no foreground pixels")
    means = image[foreground].mean(axis=0)
    target = float(means.mean())
    scales = np.clip(target / np.maximum(means, 1.0), clip_range[0], clip_range[1])
    return ensure_uint8_rgb(image * scales.reshape(1, 1, 3))


def gray3ch(image_rgb: np.ndarray) -> np.ndarray:
    """Convert RGB to gray and repeat to three channels."""
    image = ensure_uint8_rgb(image_rgb)
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    return np.repeat(gray[:, :, None], 3, axis=2)


def retinex_msr(
    image_rgb: np.ndarray,
    scales: Sequence[float] = (15, 80, 250),
) -> np.ndarray:
    """Traditional Multi-Scale Retinex with robust per-channel normalization."""
    image = ensure_uint8_rgb(image_rgb).astype(np.float32) + 1.0
    result = np.zeros_like(image, dtype=np.float32)
    for sigma in scales:
        blurred = cv2.GaussianBlur(image, (0, 0), sigmaX=float(sigma), sigmaY=float(sigma))
        result += np.log(image) - np.log(np.maximum(blurred, 1.0))
    result /= float(len(scales))

    output = np.empty_like(result, dtype=np.float32)
    for channel in range(3):
        values = result[:, :, channel]
        lo, hi = np.percentile(values, [1, 99])
        if not math.isfinite(float(lo)) or not math.isfinite(float(hi)) or hi <= lo:
            output[:, :, channel] = 127.0
        else:
            output[:, :, channel] = (values - lo) * (255.0 / (hi - lo))
    return ensure_uint8_rgb(output)


def compute_region_color_stats(image_rgb: np.ndarray, mask: np.ndarray) -> dict[str, float]:
    """Compute RGB and Lab-L means inside mask."""
    image = ensure_uint8_rgb(image_rgb)
    foreground = mask > 0
    if not foreground.any():
        return {
            "mean_r": float("nan"),
            "mean_g": float("nan"),
            "mean_b": float("nan"),
            "mean_l": float("nan"),
        }
    rgb_means = image[foreground].astype(np.float32).mean(axis=0)
    lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
    mean_l = float(lab[:, :, 0][foreground].astype(np.float32).mean())
    return {
        "mean_r": float(rgb_means[0]),
        "mean_g": float(rgb_means[1]),
        "mean_b": float(rgb_means[2]),
        "mean_l": mean_l,
    }


def collect_required_ids_from_splits(split_dir: str | Path) -> list[str]:
    """Collect every ID referenced by fold train/val CSV files."""
    root = Path(split_dir)
    csv_paths = sorted(root.glob("fold_*_train.csv")) + sorted(root.glob("fold_*_val.csv"))
    if not csv_paths:
        raise FileNotFoundError(f"No fold train/val CSV files found in {root}")
    ids: set[str] = set()
    for csv_path in csv_paths:
        frame = pd.read_csv(csv_path, dtype={"ID": "string"}, encoding="utf-8-sig")
        if "ID" not in frame.columns:
            raise ValueError(f"{csv_path} is missing ID column")
        ids.update(frame["ID"].dropna().astype(str).tolist())
    return sorted(ids)


def make_qc_grid(
    samples: Iterable[dict[str, Any]],
    output_path: str | Path,
    columns: int = 8,
    tile_size: int = 112,
) -> Path | None:
    """Create a compact image grid from sample dictionaries with path/title keys."""
    sample_list = list(samples)
    if not sample_list:
        return None
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    columns = max(1, int(columns))
    rows = int(math.ceil(len(sample_list) / columns))
    header = 18
    canvas = np.full(
        (rows * (tile_size + header), columns * tile_size, 3),
        24,
        dtype=np.uint8,
    )
    for index, sample in enumerate(sample_list):
        row = index // columns
        col = index % columns
        x0 = col * tile_size
        y0 = row * (tile_size + header)
        try:
            image = read_rgb(sample["path"])
            image = cv2.resize(image, (tile_size, tile_size), interpolation=cv2.INTER_AREA)
        except Exception:
            image = np.full((tile_size, tile_size, 3), 80, dtype=np.uint8)
        canvas[y0 + header : y0 + header + tile_size, x0 : x0 + tile_size] = image
        title = str(sample.get("title", sample.get("ID", "")))[:24]
        cv2.putText(
            canvas,
            title,
            (x0 + 3, y0 + 13),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.36,
            (235, 235, 235),
            1,
            cv2.LINE_AA,
        )
    save_rgb(output, canvas)
    return output
