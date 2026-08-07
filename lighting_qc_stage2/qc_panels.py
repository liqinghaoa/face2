from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .color import luminance_y, rgb_uint8_to_linear
from .image_io import save_rgb


PALETTE = np.array(
    [
        [0, 0, 0],
        [220, 170, 130],
        [180, 90, 40],
        [180, 90, 40],
        [30, 120, 220],
        [30, 120, 220],
        [120, 200, 255],
        [220, 180, 120],
        [220, 180, 120],
        [220, 220, 80],
        [230, 130, 100],
        [180, 60, 120],
        [210, 70, 150],
        [210, 70, 150],
        [140, 100, 80],
        [150, 120, 90],
        [80, 180, 80],
        [40, 40, 40],
        [160, 160, 160],
    ],
    dtype=np.uint8,
)


def overlay(rgb: np.ndarray, mask: np.ndarray, color: tuple[int, int, int], alpha: float = 0.45) -> np.ndarray:
    out = rgb.copy()
    hit = mask > 0
    out[hit] = np.clip(out[hit].astype(float) * (1 - alpha) + np.asarray(color) * alpha, 0, 255).astype(np.uint8)
    return out


def heatmap_y(rgb: np.ndarray) -> np.ndarray:
    y = luminance_y(rgb_uint8_to_linear(rgb))
    v = np.clip((y - np.nanmin(y)) / (np.nanmax(y) - np.nanmin(y) + 1e-8), 0, 1)
    return np.stack([np.uint8(255 * v), np.uint8(255 * (1 - np.abs(v - 0.5) * 2)), np.uint8(255 * (1 - v))], axis=2)


def label_panel(arr: np.ndarray, title: str, size: tuple[int, int] = (256, 200)) -> Image.Image:
    im = Image.fromarray(arr.astype(np.uint8)).resize(size, Image.Resampling.BILINEAR)
    draw = ImageDraw.Draw(im)
    draw.rectangle([0, 0, size[0], 24], fill=(0, 0, 0))
    draw.text((6, 6), title[:32], fill=(255, 255, 255))
    return im


def parsing_color(parsing: np.ndarray) -> np.ndarray:
    idx = np.clip(parsing.astype(int), 0, len(PALETTE) - 1)
    return PALETTE[idx]


def make_qc_panel(path: Path, sample_id: str, raw_rgb: np.ndarray, aligned: np.ndarray, parsing: np.ndarray, masks: dict[str, np.ndarray], raw_masks: dict[str, np.ndarray], metrics: dict[str, Any], roundtrip: dict[str, Any]) -> None:
    raw_small = np.asarray(Image.fromarray(raw_rgb).resize((256, 200), Image.Resampling.BILINEAR))
    raw_core = overlay(raw_rgb, raw_masks["raw_core_skin"], (0, 220, 80))
    raw_roi = overlay(overlay(overlay(raw_rgb, raw_masks["raw_forehead"], (255, 220, 0)), raw_masks["raw_canvas_left_cheek"], (0, 120, 255)), raw_masks["raw_canvas_right_cheek"], (255, 90, 0))
    y = luminance_y(rgb_uint8_to_linear(aligned))
    bright = ((y >= 0.80) & (masks["core_skin_e2"] > 0)).astype(np.uint8) * 255
    dark = ((y <= 0.05) & (masks["core_skin_e2"] > 0)).astype(np.uint8) * 255
    shadow = ((y < 0.65 * np.median(y[masks["core_skin_e2"] > 0])) & (masks["core_skin_e2"] > 0)).astype(np.uint8) * 255
    spec = ((y >= 0.80) & (((rgb_uint8_to_linear(aligned).max(axis=2) - rgb_uint8_to_linear(aligned).min(axis=2)) <= 0.05)) & ((masks["core_skin_e2"] > 0) | (masks["nose"] > 0))).astype(np.uint8) * 255
    panels = [
        label_panel(raw_small, "EXIF-transposed raw"),
        label_panel(raw_core, "raw core skin"),
        label_panel(raw_roi, "raw forehead/cheek/nose"),
        label_panel(aligned, "aligned_srgb 256x320"),
        label_panel(parsing_color(parsing), "parsing_label"),
        label_panel(np.dstack([masks["face_geometry"]] * 3), "face/source valid"),
        label_panel(np.dstack([masks["core_skin_e0"], masks["core_skin_e2"], masks["core_skin_e4"]]), "core e0/e2/e4"),
        label_panel(overlay(overlay(overlay(aligned, masks["forehead"], (255, 220, 0)), masks["canvas_left_cheek"], (0, 120, 255)), masks["canvas_right_cheek"], (255, 90, 0)), "canvas ROI overlay"),
        label_panel(heatmap_y(aligned), "linear Y heatmap"),
        label_panel(overlay(aligned, bright, (255, 255, 255)), "bright y>=0.80"),
        label_panel(overlay(aligned, dark, (0, 0, 180)), "dark y<=0.05"),
        label_panel(overlay(aligned, shadow, (90, 0, 180)), "shadow ratio 0.65"),
        label_panel(overlay(aligned, spec, (255, 255, 0)), "specular candidate"),
        label_panel(overlay(raw_rgb, raw_masks["raw_core_skin"], (255, 255, 0)), "raw clipping/spec"),
        label_panel(overlay(aligned, roundtrip.get("roundtrip_core_skin", masks["core_skin_e2"]), (255, 0, 255)), "round-trip overlay"),
    ]
    text_img = Image.new("RGB", (256, 200), (245, 245, 245))
    draw = ImageDraw.Draw(text_img)
    text = [
        f"ID {sample_id}",
        f"skin median {metrics.get('skin_y_median', 'NA'):.4f}" if isinstance(metrics.get("skin_y_median"), float) else "skin median NA",
        f"bright>=.80 {metrics.get('bright_fraction_y_ge_0.80', 'NA')}",
        f"dark<=.05 {metrics.get('dark_fraction_y_le_0.05', 'NA')}",
        f"IoU {roundtrip.get('roundtrip_iou_core_skin', 'NA')}",
        "No thresholds frozen",
    ]
    for i, line in enumerate(text):
        draw.text((8, 12 + i * 22), line, fill=(0, 0, 0))
    panels.append(text_img)
    rows = []
    for start in range(0, len(panels), 4):
        chunk = panels[start : start + 4]
        while len(chunk) < 4:
            chunk.append(Image.new("RGB", (256, 200), (0, 0, 0)))
        rows.append(np.concatenate([np.asarray(p) for p in chunk], axis=1))
    save_rgb(path, np.concatenate(rows, axis=0))


def make_contact_sheet(paths: list[Path], output_path: Path, columns: int = 4) -> None:
    images = []
    for path in paths:
        if path.is_file():
            images.append(Image.open(path).convert("RGB").resize((320, 250), Image.Resampling.BILINEAR))
    if not images:
        return
    rows = []
    for start in range(0, len(images), columns):
        chunk = images[start : start + columns]
        while len(chunk) < columns:
            chunk.append(Image.new("RGB", (320, 250), (0, 0, 0)))
        rows.append(np.concatenate([np.asarray(x) for x in chunk], axis=1))
    save_rgb(output_path, np.concatenate(rows, axis=0))
