from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw

from .color import luminance_y, rgb_uint8_to_linear
from .frozen_metrics import AUXILIARY_SHADOW_RATIO, AUXILIARY_SPECULAR_CHROMA, AUXILIARY_SPECULAR_Y, SEVERE_BRIGHT_Y, SEVERE_DARK_Y
from .image_io import save_rgb
from .qc_panels import PALETTE, overlay


@dataclass(frozen=True)
class LetterboxResult:
    image: np.ndarray
    scale: float
    offset_x: int
    offset_y: int
    content_width: int
    content_height: int


def render_letterboxed(image: np.ndarray, target_width: int, target_height: int, pad_value: int | tuple[int, int, int] = 245, is_mask: bool = False) -> LetterboxResult:
    arr = np.asarray(image)
    original = arr.copy()
    if arr.ndim == 2:
        mode = "L"
        pil = Image.fromarray(arr.astype(np.uint8), mode=mode)
    elif arr.ndim == 3 and arr.shape[2] == 3:
        mode = "RGB"
        pil = Image.fromarray(arr.astype(np.uint8), mode=mode)
    else:
        raise ValueError("image must be HxW or HxWx3")
    h, w = arr.shape[:2]
    scale = min(target_width / w, target_height / h)
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    resample = Image.Resampling.NEAREST if is_mask or arr.ndim == 2 else Image.Resampling.BILINEAR
    resized = pil.resize((new_w, new_h), resample)
    if arr.ndim == 2:
        pad = int(pad_value) if not isinstance(pad_value, tuple) else int(pad_value[0])
        canvas = Image.new("L", (target_width, target_height), pad)
    else:
        pad_rgb = pad_value if isinstance(pad_value, tuple) else (int(pad_value), int(pad_value), int(pad_value))
        canvas = Image.new("RGB", (target_width, target_height), pad_rgb)
    off_x = (target_width - new_w) // 2
    off_y = (target_height - new_h) // 2
    canvas.paste(resized, (off_x, off_y))
    if not np.array_equal(arr, original):
        raise RuntimeError("render_letterboxed modified input array")
    return LetterboxResult(np.asarray(canvas, dtype=np.uint8), float(scale), off_x, off_y, new_w, new_h)


def parsing_color(parsing: np.ndarray) -> np.ndarray:
    idx = np.clip(parsing.astype(int), 0, len(PALETTE) - 1)
    return PALETTE[idx]


def heatmap_y(rgb: np.ndarray, mask: np.ndarray | None = None) -> np.ndarray:
    y = luminance_y(rgb_uint8_to_linear(rgb))
    valid = mask > 0 if mask is not None else np.isfinite(y)
    lo = float(np.nanmin(y[valid])) if valid.any() else 0.0
    hi = float(np.nanmax(y[valid])) if valid.any() else 1.0
    v = np.clip((y - lo) / (hi - lo + 1e-8), 0, 1)
    out = np.stack([np.uint8(255 * v), np.uint8(255 * (1 - np.abs(v - 0.5) * 2)), np.uint8(255 * (1 - v))], axis=2)
    out[~valid] = 0
    return out


def titled_panel(arr: np.ndarray, title: str, cell_size: tuple[int, int] = (260, 340), is_mask: bool = False) -> Image.Image:
    body_h = cell_size[1] - 42
    lb = render_letterboxed(arr, cell_size[0], body_h, pad_value=(238, 238, 238), is_mask=is_mask)
    panel = Image.new("RGB", cell_size, (255, 255, 255))
    if lb.image.ndim == 2:
        body = Image.fromarray(np.dstack([lb.image] * 3).astype(np.uint8), mode="RGB")
    else:
        body = Image.fromarray(lb.image.astype(np.uint8), mode="RGB")
    panel.paste(body, (0, 42))
    draw = ImageDraw.Draw(panel)
    draw.rectangle([0, 0, cell_size[0], 42], fill=(18, 18, 18))
    shape = "x".join(str(x) for x in arr.shape)
    draw.text((6, 5), title[:38], fill=(255, 255, 255))
    draw.text((6, 23), f"shape={shape} scale={lb.scale:.3f}", fill=(215, 215, 215))
    return panel


def combine_roi_overlay(aligned: np.ndarray, masks: dict[str, np.ndarray]) -> np.ndarray:
    out = aligned.copy()
    out = overlay(out, masks["forehead"], (255, 220, 0), 0.55)
    out = overlay(out, masks["canvas_left_cheek"], (0, 120, 255), 0.55)
    out = overlay(out, masks["canvas_right_cheek"], (255, 90, 0), 0.55)
    out = overlay(out, masks["nose"], (0, 210, 130), 0.55)
    return out


def make_qc_panel_fixed(path: Path, sample_id: str, raw_rgb: np.ndarray, aligned: np.ndarray, parsing: np.ndarray, masks: dict[str, np.ndarray], raw_masks: dict[str, np.ndarray], metrics: dict[str, Any], roundtrip: dict[str, Any]) -> None:
    linear = rgb_uint8_to_linear(aligned)
    y = luminance_y(linear)
    core = masks["core_skin_e2"] > 0
    median = float(np.median(y[core])) if core.any() else np.nan
    bright = ((y >= SEVERE_BRIGHT_Y) & core).astype(np.uint8) * 255
    dark = ((y <= SEVERE_DARK_Y) & core).astype(np.uint8) * 255
    shadow = ((y < AUXILIARY_SHADOW_RATIO * median) & core).astype(np.uint8) * 255
    chroma = linear.max(axis=2) - linear.min(axis=2)
    spec = ((y >= AUXILIARY_SPECULAR_Y) & (chroma <= AUXILIARY_SPECULAR_CHROMA) & (core | (masks["nose"] > 0))).astype(np.uint8) * 255
    raw_core = overlay(raw_rgb, raw_masks["raw_core_skin"], (0, 220, 80), 0.48)
    raw_roi = raw_rgb.copy()
    for name, color in [
        ("raw_forehead", (255, 220, 0)),
        ("raw_canvas_left_cheek", (0, 120, 255)),
        ("raw_canvas_right_cheek", (255, 90, 0)),
        ("raw_nose", (0, 210, 130)),
    ]:
        raw_roi = overlay(raw_roi, raw_masks[name], color, 0.5)
    flag_overlay = overlay(overlay(aligned, bright, (255, 255, 255), 0.75), dark, (0, 0, 190), 0.65)
    panels = [
        titled_panel(raw_rgb, "EXIF-transposed raw"),
        titled_panel(raw_core, "raw + raw core skin"),
        titled_panel(raw_roi, "raw + final raw ROI"),
        titled_panel(aligned, "aligned_srgb 256x320"),
        titled_panel(parsing_color(parsing), "parsing_label"),
        titled_panel(np.dstack([masks["core_skin_e2"]] * 3), "core skin erosion=2", is_mask=True),
        titled_panel(combine_roi_overlay(aligned, masks), "final forehead/canvas-cheek/nose ROI"),
        titled_panel(heatmap_y(aligned, masks["core_skin_e2"]), "linear Y heatmap"),
        titled_panel(flag_overlay, "severe bright / severe dark"),
        titled_panel(overlay(aligned, shadow, (90, 0, 180), 0.62), "relative dark-region candidate"),
        titled_panel(overlay(aligned, spec, (255, 255, 0), 0.70), "low-chroma high-luminance candidate"),
        titled_panel(overlay(aligned, roundtrip.get("roundtrip_core_skin", masks["core_skin_e2"]), (255, 0, 255), 0.5), "round-trip overlay"),
    ]
    text = Image.new("RGB", (1040, 110), (246, 246, 246))
    draw = ImageDraw.Draw(text)
    lines = [
        f"ID {sample_id}",
        f"skin_y_median={metrics.get('skin_y_median', 'NA')} dark_contrast_score={metrics.get('dark_contrast_score', 'NA')}",
        f"cheek_relative_difference={metrics.get('cheek_relative_difference', 'NA')} auxiliary_shadow_fraction={metrics.get('auxiliary_shadow_fraction', 'NA')}",
        f"IoU={roundtrip.get('roundtrip_iou_core_skin', 'NA')} centroid_distance={roundtrip.get('roundtrip_centroid_distance_core_skin', 'NA')}",
    ]
    for i, line in enumerate(lines):
        draw.text((10, 10 + i * 24), str(line)[:170], fill=(0, 0, 0))
    rows = []
    for start in range(0, len(panels), 4):
        rows.append(np.concatenate([np.asarray(p) for p in panels[start : start + 4]], axis=1))
    save_rgb(path, np.concatenate([*rows, np.asarray(text)], axis=0))


def make_fixed_contact_sheet(panel_paths: list[Path], output_path: Path, start: int, count: int = 8, columns: int = 4) -> None:
    selected = panel_paths[start : start + count]
    thumbs: list[Image.Image] = []
    for path in selected:
        if path.is_file():
            src = np.asarray(Image.open(path).convert("RGB"))
            thumb = render_letterboxed(src, 380, 270, pad_value=(245, 245, 245)).image
            thumbs.append(Image.fromarray(thumb, mode="RGB"))
    if not thumbs:
        return
    rows = []
    for i in range(0, len(thumbs), columns):
        chunk = thumbs[i : i + columns]
        while len(chunk) < columns:
            chunk.append(Image.new("RGB", (380, 270), (245, 245, 245)))
        rows.append(np.concatenate([np.asarray(x) for x in chunk], axis=1))
    save_rgb(output_path, np.concatenate(rows, axis=0))
