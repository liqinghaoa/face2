"""Build direct-warp, face-parsing hybrid forehead-repair RGB PNG images.

Key differences from the earlier regular-envelope version:

1. FaceMesh may run on a detection crop, but its landmarks are converted back
   to original-image coordinates. The final similarity transform directly
   warps the original RGB image to the output canvas, so crop boundaries cannot
   become rotated black cuts in the aligned image.
2. The final mask is primarily a lightly regularized semantic face mask. A
   regular ellipse is only OR-ed into a local forehead band when hair
   occlusion (or optionally contour jaggedness) triggers repair.

Saved images are ordinary RGB uint8 PNG files. ImageNet normalization here is
only used internally for BiSeNet inference and is not applied to saved images.
"""

from __future__ import annotations

import argparse
import math
import random
import shutil
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
import torch
import yaml

try:
    import build_global_face_oval_blackbg_png_simalign_strict as alignment
    import build_global_face_parsing_regularmask_blackbg_224_png_strict as parsing
except ImportError:
    from preprocessing import (  # type: ignore[no-redef]
        build_global_face_oval_blackbg_png_simalign_strict as alignment,
    )
    from preprocessing import (  # type: ignore[no-redef]
        build_global_face_parsing_regularmask_blackbg_224_png_strict as parsing,
    )


STATUS_VALUES = (
    "success",
    "failed_no_image",
    "failed_read_image",
    "failed_no_face",
    "failed_landmark_incomplete",
    "failed_alignment",
    "failed_parsing_model",
    "failed_empty_selected_mask",
    "failed_save",
    "failed_unexpected_error",
)

LOG_COLUMNS = (
    "ID", "NYHA", "extreme_label", "fold", "SEX",
    "input_path", "output_path", "status", "fail_reason",
    "num_faces_detected", "selected_face_index",
    "face_bbox_x", "face_bbox_y", "face_bbox_w", "face_bbox_h",
    "expanded_bbox_x", "expanded_bbox_y", "expanded_bbox_w", "expanded_bbox_h",
    "alignment_source", "align_success", "rotation_angle", "scale_factor",
    "translation_x", "translation_y", "eye_distance", "eye_y_after_align",
    "top_cut_warning", "top_invalid_area_ratio",
    "parsing_model", "parsing_checkpoint", "parsing_success",
    "selected_semantic_area_pixels", "selected_semantic_area_ratio",
    "semantic_regularized_area_pixels", "semantic_regularized_area_ratio",
    "regular_envelope_area_pixels", "regular_envelope_area_ratio",
    "final_mask_area_pixels", "final_mask_area_ratio",
    "mask_area_pixels", "mask_area_ratio", "mask_warning",
    "final_mask_mode", "forehead_repair_applied", "forehead_repair_reason",
    "forehead_band_ratio", "forehead_top_jaggedness",
    "hair_inside_candidate_envelope_ratio",
    "hair_inside_final_mask_pixels", "hair_inside_final_mask_ratio",
    "hair_warning_flag", "neck_inside_final_mask_ratio",
    "cloth_inside_final_mask_ratio", "background_inside_final_mask_ratio",
    "skin_area_ratio", "brow_area_ratio", "eye_area_ratio",
    "nose_area_ratio", "mouth_area_ratio", "lip_area_ratio",
    "hair_area_ratio", "ear_area_ratio", "neck_area_ratio",
    "cloth_area_ratio", "background_area_ratio",
    "forehead_expand_ratio", "side_expand_ratio", "chin_expand_ratio",
    "hair_repair_threshold", "jaggedness_threshold",
    "enable_jaggedness_trigger", "background_mode", "feather_enabled",
    "feather_kernel", "output_format", "image_size",
)

FAILED_COLUMNS = (
    "ID", "NYHA", "extreme_label", "fold", "SEX",
    "input_path", "status", "fail_reason",
)

SUMMARY_METRICS = (
    "count", "mean", "std", "min", "p1", "p5", "p25",
    "median", "p75", "p95", "p99", "max",
)

SUMMARY_VALUE_COLUMNS = (
    "mask_area_ratio",
    "selected_semantic_area_ratio",
    "semantic_regularized_area_ratio",
    "regular_envelope_area_ratio",
    "hair_inside_candidate_envelope_ratio",
    "hair_inside_final_mask_ratio",
    "neck_inside_final_mask_ratio",
    "cloth_inside_final_mask_ratio",
    "background_inside_final_mask_ratio",
    "forehead_top_jaggedness",
)


@dataclass
class PreviewImages:
    original: np.ndarray | None = None
    crop: np.ndarray | None = None
    aligned: np.ndarray | None = None
    parsing_map: np.ndarray | None = None
    selected_overlay: np.ndarray | None = None
    semantic_overlay: np.ndarray | None = None
    envelope_overlay: np.ndarray | None = None
    final_overlay: np.ndarray | None = None
    final: np.ndarray | None = None


def default_project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Build original-coordinate direct-warp face-parsing hybrid "
            "forehead-repair RGB PNG images."
        )
    )
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--project-root", type=Path, default=None)
    parser.add_argument("--split-csv", type=Path, default=None)
    parser.add_argument("--image-dir", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--image-size", type=int, default=None)
    parser.add_argument("--min-detection-confidence", type=float, default=None)
    parser.add_argument("--parsing-model", type=str, default=None)
    parser.add_argument("--parsing-checkpoint", type=Path, default=None)
    parser.add_argument("--parsing-device", type=str, default=None)
    parser.add_argument(
        "--final-mask-mode",
        choices=("semantic", "envelope", "hybrid"),
        default=None,
    )
    parser.add_argument("--forehead-band-ratio", type=float, default=None)
    parser.add_argument("--hair-repair-threshold", type=float, default=None)
    parser.add_argument("--jaggedness-threshold", type=float, default=None)
    parser.add_argument(
        "--enable-jaggedness-trigger",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    parser.add_argument("--hair-warning", type=float, default=None)
    parser.add_argument("--heavy-hair-warning", type=float, default=None)
    parser.add_argument("--mask-low-warning", type=float, default=None)
    parser.add_argument("--mask-high-warning", type=float, default=None)
    parser.add_argument("--forehead-expand-ratio", type=float, default=None)
    parser.add_argument("--side-expand-ratio", type=float, default=None)
    parser.add_argument("--chin-expand-ratio", type=float, default=None)
    parser.add_argument("--feather-kernel", type=int, default=None)
    parser.add_argument("--num-qc-preview", type=int, default=None)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        default=None,
        help="Remove only the selected output directory before rebuilding it.",
    )
    return parser


def _normalize_config_keys(config: dict[str, Any]) -> dict[str, Any]:
    return {str(key).strip().replace("-", "_"): value for key, value in config.items()}


def load_yaml_config(config_path: Path | None, project_root: Path) -> dict[str, Any]:
    if config_path is None:
        candidate = (
            project_root
            / "config"
            / "preprocess"
            / "global_face_parsing_hybrid_foreheadrepair_blackbg_224_png_strict.yaml"
        )
        config_path = candidate if candidate.is_file() else None
    elif not config_path.is_absolute():
        config_path = project_root / config_path
    if config_path is None:
        return {}
    if not config_path.is_file():
        raise FileNotFoundError(f"Config file does not exist: {config_path}")
    with config_path.open("r", encoding="utf-8") as handle:
        loaded = yaml.safe_load(handle) or {}
    if not isinstance(loaded, dict):
        raise ValueError(f"Config root must be a YAML mapping: {config_path}")
    return _normalize_config_keys(loaded)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = build_parser()
    cli = parser.parse_args(argv)
    inferred_root = (
        cli.project_root.expanduser().resolve()
        if cli.project_root is not None
        else default_project_root()
    )
    config = load_yaml_config(cli.config, inferred_root)
    config_root = config.get("project_root")
    project_root = (
        inferred_root
        if cli.project_root is not None or config_root in (None, "", ".")
        else Path(config_root).expanduser().resolve()
    )
    defaults: dict[str, Any] = {
        "split_csv": "data/processed/splits/extreme_5fold.csv",
        "image_dir": "data/raw/images",
        "output_dir": (
            "data/processed/global_face/"
            "global_face_parsing_hybrid_foreheadrepair_blackbg_224_png_strict"
        ),
        "image_size": 224,
        "min_detection_confidence": 0.5,
        "parsing_model": "bisenet",
        "parsing_checkpoint": (
            "preprocessing/checkpoints/face_parsing/79999_iter.pth"
        ),
        "parsing_device": "auto",
        "final_mask_mode": "hybrid",
        "forehead_band_ratio": 0.35,
        "hair_repair_threshold": 0.10,
        "jaggedness_threshold": 0.04,
        "enable_jaggedness_trigger": False,
        "hair_warning": 0.10,
        "heavy_hair_warning": 0.20,
        "mask_low_warning": 0.35,
        "mask_high_warning": 0.85,
        "forehead_expand_ratio": 0.18,
        "side_expand_ratio": 0.05,
        "chin_expand_ratio": 0.03,
        "feather_kernel": 11,
        "num_qc_preview": 20,
        "max_samples": None,
        "seed": 42,
        "overwrite": False,
    }
    values: dict[str, Any] = {"project_root": project_root, "config": cli.config}
    for key, default in defaults.items():
        cli_value = getattr(cli, key)
        values[key] = cli_value if cli_value is not None else config.get(key, default)
    args = argparse.Namespace(**values)
    validate_args(args)
    return args


def validate_args(args: argparse.Namespace) -> None:
    if int(args.image_size) <= 0:
        raise ValueError("--image-size must be greater than zero")
    if not 0.0 <= float(args.min_detection_confidence) <= 1.0:
        raise ValueError("--min-detection-confidence must be in [0, 1]")
    if str(args.parsing_model).lower() != "bisenet":
        raise ValueError("First version supports only --parsing-model bisenet")
    if str(args.parsing_device).lower() not in {"auto", "cpu", "cuda"}:
        raise ValueError("--parsing-device must be auto, cpu, or cuda")
    if str(args.final_mask_mode) not in {"semantic", "envelope", "hybrid"}:
        raise ValueError("--final-mask-mode must be semantic, envelope, or hybrid")
    for name in (
        "forehead_band_ratio",
        "hair_repair_threshold",
        "jaggedness_threshold",
        "hair_warning",
        "heavy_hair_warning",
        "mask_low_warning",
        "mask_high_warning",
        "forehead_expand_ratio",
        "side_expand_ratio",
        "chin_expand_ratio",
    ):
        if not 0.0 <= float(getattr(args, name)) <= 1.0:
            raise ValueError(f"--{name.replace('_', '-')} must be in [0, 1]")
    if float(args.hair_warning) >= float(args.heavy_hair_warning):
        raise ValueError("--hair-warning must be lower than --heavy-hair-warning")
    if float(args.mask_low_warning) >= float(args.mask_high_warning):
        raise ValueError("--mask-low-warning must be lower than --mask-high-warning")
    if int(args.feather_kernel) <= 0 or int(args.feather_kernel) % 2 == 0:
        raise ValueError("--feather-kernel must be a positive odd integer")
    if int(args.num_qc_preview) < 0:
        raise ValueError("--num-qc-preview must be non-negative")
    if args.max_samples is not None and int(args.max_samples) <= 0:
        raise ValueError("--max-samples must be greater than zero")


def resolve_project_root(args: argparse.Namespace) -> Path:
    root = Path(args.project_root).expanduser().resolve()
    if not root.is_dir():
        raise NotADirectoryError(f"Project root does not exist: {root}")
    return root


def resolve_under_project(path_value: str | Path, project_root: Path) -> Path:
    path = Path(path_value).expanduser()
    return path.resolve() if path.is_absolute() else (project_root / path).resolve()


def load_split_table(split_csv: Path, max_samples: int | None) -> pd.DataFrame:
    table = alignment.load_split_ids(split_csv)
    return table if max_samples is None else table.iloc[: int(max_samples)].copy()


def prepare_output_dirs(
    output_dir: Path, overwrite: bool, project_root: Path
) -> dict[str, Path]:
    output_dir = output_dir.resolve()
    protected = {
        project_root.resolve(),
        (project_root / "data").resolve(),
        (project_root / "data" / "processed").resolve(),
        (project_root / "data" / "processed" / "global_face").resolve(),
    }
    if output_dir in protected:
        raise ValueError(f"Refusing unsafe output directory: {output_dir}")
    if output_dir.exists():
        if not overwrite:
            raise FileExistsError(
                f"Output directory already exists: {output_dir}\n"
                "Use --overwrite only for this exact dataset directory."
            )
        shutil.rmtree(output_dir)
    names = (
        "images", "logs",
        "qc_preview/random_success",
        "qc_preview/semantic_only",
        "qc_preview/forehead_repair",
        "qc_preview/high_hair_inside_final_mask",
        "qc_preview/high_cloth_inside_final_mask",
        "qc_preview/high_neck_inside_final_mask",
        "qc_preview/low_mask_area",
        "qc_preview/high_mask_area",
        "qc_preview/failed_no_face",
        "qc_preview/failed_alignment",
        "qc_preview/failed_parsing",
        "qc_preview/failed_empty_mask",
        "qc_preview/.staging_success",
    )
    dirs = {"root": output_dir}
    for name in names:
        key = name.split("/")[-1]
        dirs[key] = output_dir / name
        dirs[key].mkdir(parents=True, exist_ok=True)
    return dirs


def _draw_detection_panel(
    image_rgb: np.ndarray,
    detections: Sequence[alignment.FaceDetection],
    selected: alignment.FaceDetection | None,
) -> np.ndarray:
    panel = image_rgb.copy()
    for detection in detections:
        x, y, width, height = detection.bbox
        color = (
            (0, 255, 0)
            if selected is not None and detection.index == selected.index
            else (255, 180, 0)
        )
        cv2.rectangle(panel, (x, y), (x + width, y + height), color, 3)
    return panel


def estimate_original_coordinate_alignment(
    image_rgb: np.ndarray,
    selected_face: alignment.FaceDetection,
    face_mesh: Any,
    image_size: int,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any], np.ndarray]:
    """Estimate and apply original-image -> output similarity alignment."""
    crop_x, crop_y, crop_width, crop_height = alignment.expand_bbox(
        selected_face.bbox, image_rgb.shape
    )
    if crop_width <= 1 or crop_height <= 1:
        raise alignment.SampleFailure(
            "failed_alignment", "expanded_face_bbox_is_empty"
        )
    crop_rgb = image_rgb[
        crop_y : crop_y + crop_height,
        crop_x : crop_x + crop_width,
    ].copy()
    landmarks = alignment.run_facemesh(crop_rgb, face_mesh)
    if landmarks is None:
        raise alignment.SampleFailure(
            "failed_landmark_incomplete", "mediapipe_facemesh_returned_none"
        )
    crop_points, _, eye_distance = alignment.extract_alignment_landmarks(
        landmarks, crop_rgb.shape
    )
    original_points = crop_points + np.array(
        [float(crop_x), float(crop_y)], dtype=np.float32
    )
    matrix, parameters = alignment.estimate_similarity_transform(
        original_points,
        alignment.canonical_alignment_template(image_size),
    )
    aligned_rgb = cv2.warpAffine(
        image_rgb,
        matrix,
        (image_size, image_size),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0),
    )
    if aligned_rgb.shape != (image_size, image_size, 3):
        raise alignment.SampleFailure(
            "failed_alignment", "direct original-coordinate warp failed"
        )
    transformed_points = cv2.transform(
        original_points.reshape(1, -1, 2), matrix
    )[0]
    eye_y_after_align = float(transformed_points[:2, 1].mean())

    source_valid = np.full(image_rgb.shape[:2], 255, dtype=np.uint8)
    aligned_valid = cv2.warpAffine(
        source_valid,
        matrix,
        (image_size, image_size),
        flags=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )
    top_band_height = max(4, int(round(image_size * 0.05)))
    top_invalid_ratio = float(
        (aligned_valid[:top_band_height] == 0).mean()
    )
    parameters.update(
        {
            "alignment_source": "original_coordinate",
            "eye_distance": float(eye_distance),
            "eye_y_after_align": eye_y_after_align,
            "top_invalid_area_ratio": top_invalid_ratio,
            "top_cut_warning": bool(top_invalid_ratio > 0.20),
            "expanded_bbox_x": crop_x,
            "expanded_bbox_y": crop_y,
            "expanded_bbox_w": crop_width,
            "expanded_bbox_h": crop_height,
        }
    )
    return aligned_rgb, crop_rgb, parameters, matrix


def build_semantic_regularized_mask(selected_mask: np.ndarray) -> np.ndarray:
    """Lightly regularize semantics without replacing the natural face shape."""
    binary = (selected_mask > 0).astype(np.uint8) * 255
    if int((binary > 0).sum()) < max(64, int(binary.size * 0.0025)):
        raise alignment.SampleFailure(
            "failed_empty_selected_mask", "selected_semantic_mask_too_small"
        )

    close_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    regularized = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, close_kernel)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(
        (regularized > 0).astype(np.uint8), connectivity=8
    )
    if count <= 1:
        raise alignment.SampleFailure(
            "failed_empty_selected_mask", "no_semantic_face_component"
        )
    largest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    regularized = (labels == largest).astype(np.uint8) * 255

    # Fill only small enclosed holes. Large hair-shaped forehead gaps remain
    # available for the explicit hybrid repair decision.
    inverse = (regularized == 0).astype(np.uint8)
    hole_count, hole_labels, hole_stats, _ = cv2.connectedComponentsWithStats(
        inverse, connectivity=8
    )
    max_hole_area = max(32, int(round(regularized.size * 0.003)))
    border_labels = set(
        np.unique(
            np.concatenate(
                [
                    hole_labels[0, :],
                    hole_labels[-1, :],
                    hole_labels[:, 0],
                    hole_labels[:, -1],
                ]
            )
        ).tolist()
    )
    for label_index in range(1, hole_count):
        if (
            label_index not in border_labels
            and int(hole_stats[label_index, cv2.CC_STAT_AREA]) <= max_hole_area
        ):
            regularized[hole_labels == label_index] = 255

    # A small blur/threshold smooths contour stair-stepping without dilation.
    smoothed = cv2.GaussianBlur(regularized, (5, 5), sigmaX=1.0)
    regularized = (smoothed >= 127).astype(np.uint8) * 255
    if int((regularized > 0).sum()) < max(64, int(regularized.size * 0.0025)):
        raise alignment.SampleFailure(
            "failed_empty_selected_mask", "semantic_regularized_mask_too_small"
        )
    return regularized


def compute_forehead_top_jaggedness(
    semantic_mask: np.ndarray,
    forehead_band_ratio: float,
) -> float:
    xmin, ymin, xmax, ymax = parsing._binary_bbox(semantic_mask)
    face_height = max(1, ymax - ymin)
    band_ymax = min(
        ymax, int(round(ymin + forehead_band_ratio * face_height))
    )
    top_values = np.full(xmax - xmin + 1, np.nan, dtype=np.float32)
    for offset, x in enumerate(range(xmin, xmax + 1)):
        ys = np.where(semantic_mask[ymin : band_ymax + 1, x] > 0)[0]
        if len(ys):
            top_values[offset] = float(ymin + ys.min())
    valid = np.isfinite(top_values)
    if int(valid.sum()) < 3:
        return 1.0
    indices = np.arange(len(top_values))
    top_values = np.interp(indices, indices[valid], top_values[valid])
    return float(np.mean(np.abs(np.diff(top_values))) / face_height)


def build_final_mask(
    semantic_mask: np.ndarray,
    candidate_envelope: np.ndarray,
    mode: str,
    forehead_band_ratio: float,
    hair_candidate_ratio: float,
    hair_repair_threshold: float,
    jaggedness: float,
    jaggedness_threshold: float,
    enable_jaggedness_trigger: bool,
) -> tuple[np.ndarray, bool, str]:
    if mode == "envelope":
        return candidate_envelope.copy(), False, "none"
    if mode == "semantic":
        return semantic_mask.copy(), False, "none"

    hair_trigger = hair_candidate_ratio > hair_repair_threshold
    jagged_trigger = (
        enable_jaggedness_trigger and jaggedness > jaggedness_threshold
    )
    if not hair_trigger and not jagged_trigger:
        return semantic_mask.copy(), False, "none"

    xmin, ymin, xmax, ymax = parsing._binary_bbox(semantic_mask)
    _, envelope_ymin, _, _ = parsing._binary_bbox(candidate_envelope)
    face_height = max(1, ymax - ymin)
    band_ymax = min(
        ymax, int(round(ymin + forehead_band_ratio * face_height))
    )
    # Start at the candidate envelope's rounded top instead of semantic ymin.
    # Starting exactly at semantic ymin can create an artificial horizontal
    # "flat-top" edge when a large hair notch triggered the repair.
    band_ymin = min(ymin, envelope_ymin)
    final_mask = semantic_mask.copy()
    final_mask[band_ymin : band_ymax + 1] = cv2.bitwise_or(
        semantic_mask[band_ymin : band_ymax + 1],
        candidate_envelope[band_ymin : band_ymax + 1],
    )
    if hair_trigger and jagged_trigger:
        reason = "hair_occlusion+jaggedness"
    elif hair_trigger:
        reason = "hair_occlusion"
    else:
        reason = "jaggedness"
    return final_mask, True, reason


def _ratio_inside_mask(
    label_map: np.ndarray,
    final_mask: np.ndarray,
    classes: Iterable[int],
) -> tuple[int, float]:
    foreground = final_mask > 0
    total = int(foreground.sum())
    if total == 0:
        raise alignment.SampleFailure(
            "failed_empty_selected_mask", "final_mask_is_empty"
        )
    pixels = int((np.isin(label_map, tuple(classes)) & foreground).sum())
    return pixels, pixels / float(total)


def _mask_overlay(
    image_rgb: np.ndarray,
    binary_mask: np.ndarray,
    color: tuple[int, int, int],
) -> np.ndarray:
    result = image_rgb.astype(np.float32)
    foreground = binary_mask > 0
    tint = np.zeros_like(result)
    tint[:] = color
    result[foreground] = result[foreground] * 0.68 + tint[foreground] * 0.32
    rendered = np.clip(result, 0, 255).astype(np.uint8)
    contours, _ = cv2.findContours(
        foreground.astype(np.uint8) * 255,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )
    cv2.drawContours(rendered, contours, -1, (255, 40, 40), 2, cv2.LINE_AA)
    return rendered


def _fit_preview_panel(
    image_rgb: np.ndarray | None,
    title: str,
    panel_size: int = 210,
) -> np.ndarray:
    label_height = 32
    canvas = np.full(
        (panel_size + label_height, panel_size, 3), 20, dtype=np.uint8
    )
    if image_rgb is not None and image_rgb.size:
        height, width = image_rgb.shape[:2]
        scale = min(panel_size / width, panel_size / height)
        out_width = max(1, int(round(width * scale)))
        out_height = max(1, int(round(height * scale)))
        resized = cv2.resize(
            image_rgb, (out_width, out_height), interpolation=cv2.INTER_AREA
        )
        x = (panel_size - out_width) // 2
        y = (panel_size - out_height) // 2
        canvas[y : y + out_height, x : x + out_width] = resized
    cv2.putText(
        canvas,
        title,
        (6, panel_size + 22),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.43,
        (240, 240, 240),
        1,
        cv2.LINE_AA,
    )
    return canvas


def make_qc_preview(
    output_path: Path,
    image_id: str,
    status: str,
    previews: PreviewImages,
    details: str,
) -> None:
    panels = [
        _fit_preview_panel(previews.original, "original"),
        _fit_preview_panel(previews.crop, "detection crop"),
        _fit_preview_panel(previews.aligned, "direct-warp aligned"),
        _fit_preview_panel(previews.parsing_map, "parsing labels"),
        _fit_preview_panel(previews.selected_overlay, "selected semantics"),
        _fit_preview_panel(previews.semantic_overlay, "semantic regularized"),
        _fit_preview_panel(previews.envelope_overlay, "candidate envelope"),
        _fit_preview_panel(previews.final_overlay, "final hybrid mask"),
        _fit_preview_panel(previews.final, "black-bg PNG"),
    ]
    strip = np.concatenate(panels, axis=1)
    header_height = 44
    canvas = np.full(
        (strip.shape[0] + header_height, strip.shape[1], 3),
        10,
        dtype=np.uint8,
    )
    canvas[header_height:] = strip
    cv2.putText(
        canvas,
        f"ID={image_id} status={status} {details}"[:260],
        (10, 29),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (245, 245, 245),
        1,
        cv2.LINE_AA,
    )
    bgr = cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR)
    ok, encoded = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, 90])
    if ok:
        try:
            output_path.write_bytes(encoded.tobytes())
        except OSError as exc:
            print(f"[warning] Failed to write QC preview {output_path}: {exc}")


def _initial_log_row(
    split_row: dict[str, Any], args: argparse.Namespace
) -> dict[str, Any]:
    row = dict(split_row)
    row.update(
        {
            "input_path": "", "output_path": "", "status": "", "fail_reason": "",
            "num_faces_detected": 0, "selected_face_index": np.nan,
            "face_bbox_x": np.nan, "face_bbox_y": np.nan,
            "face_bbox_w": np.nan, "face_bbox_h": np.nan,
            "expanded_bbox_x": np.nan, "expanded_bbox_y": np.nan,
            "expanded_bbox_w": np.nan, "expanded_bbox_h": np.nan,
            "alignment_source": "original_coordinate", "align_success": False,
            "rotation_angle": np.nan, "scale_factor": np.nan,
            "translation_x": np.nan, "translation_y": np.nan,
            "eye_distance": np.nan, "eye_y_after_align": np.nan,
            "top_cut_warning": False, "top_invalid_area_ratio": np.nan,
            "parsing_model": str(args.parsing_model),
            "parsing_checkpoint": str(args.parsing_checkpoint),
            "parsing_success": False,
            "selected_semantic_area_pixels": np.nan,
            "selected_semantic_area_ratio": np.nan,
            "semantic_regularized_area_pixels": np.nan,
            "semantic_regularized_area_ratio": np.nan,
            "regular_envelope_area_pixels": np.nan,
            "regular_envelope_area_ratio": np.nan,
            "final_mask_area_pixels": np.nan, "final_mask_area_ratio": np.nan,
            "mask_area_pixels": np.nan, "mask_area_ratio": np.nan,
            "mask_warning": "none", "final_mask_mode": str(args.final_mask_mode),
            "forehead_repair_applied": False, "forehead_repair_reason": "none",
            "forehead_band_ratio": float(args.forehead_band_ratio),
            "forehead_top_jaggedness": np.nan,
            "hair_inside_candidate_envelope_ratio": np.nan,
            "hair_inside_final_mask_pixels": np.nan,
            "hair_inside_final_mask_ratio": np.nan,
            "hair_warning_flag": "none",
            "neck_inside_final_mask_ratio": np.nan,
            "cloth_inside_final_mask_ratio": np.nan,
            "background_inside_final_mask_ratio": np.nan,
            "skin_area_ratio": np.nan, "brow_area_ratio": np.nan,
            "eye_area_ratio": np.nan, "nose_area_ratio": np.nan,
            "mouth_area_ratio": np.nan, "lip_area_ratio": np.nan,
            "hair_area_ratio": np.nan, "ear_area_ratio": np.nan,
            "neck_area_ratio": np.nan, "cloth_area_ratio": np.nan,
            "background_area_ratio": np.nan,
            "forehead_expand_ratio": float(args.forehead_expand_ratio),
            "side_expand_ratio": float(args.side_expand_ratio),
            "chin_expand_ratio": float(args.chin_expand_ratio),
            "hair_repair_threshold": float(args.hair_repair_threshold),
            "jaggedness_threshold": float(args.jaggedness_threshold),
            "enable_jaggedness_trigger": bool(args.enable_jaggedness_trigger),
            "background_mode": "black_rgb_0_0_0", "feather_enabled": True,
            "feather_kernel": int(args.feather_kernel),
            "output_format": "PNG", "image_size": int(args.image_size),
        }
    )
    return row


def process_one_sample(
    split_row: dict[str, Any],
    image_dir: Path,
    images_dir: Path,
    detector: Any,
    face_mesh: Any,
    parsing_model: torch.nn.Module,
    parsing_device: torch.device,
    args: argparse.Namespace,
) -> tuple[dict[str, Any], PreviewImages]:
    image_id = str(split_row["ID"])
    row = _initial_log_row(split_row, args)
    previews = PreviewImages()
    input_path = alignment.find_image_for_id(image_id, image_dir)
    output_path = images_dir / f"{image_id}.png"
    row["output_path"] = str(output_path)
    try:
        if input_path is None:
            raise alignment.SampleFailure(
                "failed_no_image", "cannot_find_image_by_ID"
            )
        row["input_path"] = str(input_path)
        image_rgb = alignment.read_image_rgb(input_path)
        if image_rgb is None:
            raise alignment.SampleFailure(
                "failed_read_image", "cannot_decode_image"
            )
        detections = alignment.detect_faces(image_rgb, detector)
        row["num_faces_detected"] = len(detections)
        selected_face = alignment.select_face(detections, image_rgb.shape)
        previews.original = _draw_detection_panel(
            image_rgb, detections, selected_face
        )
        if selected_face is None:
            raise alignment.SampleFailure(
                "failed_no_face", "mediapipe_face_detection_returned_none"
            )
        row["selected_face_index"] = selected_face.index
        x, y, width, height = selected_face.bbox
        row.update(
            {
                "face_bbox_x": x, "face_bbox_y": y,
                "face_bbox_w": width, "face_bbox_h": height,
            }
        )
        aligned_rgb, crop_rgb, align_stats, _ = (
            estimate_original_coordinate_alignment(
                image_rgb, selected_face, face_mesh, int(args.image_size)
            )
        )
        previews.crop = crop_rgb
        previews.aligned = aligned_rgb
        row.update(align_stats)
        row["align_success"] = True

        try:
            label_map = parsing.run_face_parsing(
                aligned_rgb, parsing_model, parsing_device
            )
        except Exception as exc:
            if parsing_device.type == "cuda":
                torch.cuda.empty_cache()
            raise alignment.SampleFailure(
                "failed_parsing_model", f"{type(exc).__name__}: {exc}"
            ) from exc
        row["parsing_success"] = True
        previews.parsing_map = parsing.colorize_parsing_label_map(label_map)

        selected_mask = parsing.build_selected_semantic_mask(label_map)
        selected_pixels = int((selected_mask > 0).sum())
        row["selected_semantic_area_pixels"] = selected_pixels
        row["selected_semantic_area_ratio"] = selected_pixels / float(label_map.size)
        previews.selected_overlay = _mask_overlay(
            aligned_rgb, selected_mask, (30, 220, 70)
        )

        semantic_mask = build_semantic_regularized_mask(selected_mask)
        semantic_pixels = int((semantic_mask > 0).sum())
        row["semantic_regularized_area_pixels"] = semantic_pixels
        row["semantic_regularized_area_ratio"] = semantic_pixels / float(label_map.size)
        previews.semantic_overlay = _mask_overlay(
            aligned_rgb, semantic_mask, (40, 200, 100)
        )

        candidate_envelope, _ = parsing.build_regularized_face_envelope_mask(
            semantic_mask,
            label_map,
            float(args.forehead_expand_ratio),
            float(args.side_expand_ratio),
            float(args.chin_expand_ratio),
        )
        envelope_pixels = int((candidate_envelope > 0).sum())
        row["regular_envelope_area_pixels"] = envelope_pixels
        row["regular_envelope_area_ratio"] = envelope_pixels / float(label_map.size)
        previews.envelope_overlay = _mask_overlay(
            aligned_rgb, candidate_envelope, (60, 120, 255)
        )

        _, hair_candidate_ratio = _ratio_inside_mask(
            label_map, candidate_envelope, (parsing.CLASS_HAIR,)
        )
        jaggedness = compute_forehead_top_jaggedness(
            semantic_mask, float(args.forehead_band_ratio)
        )
        row["hair_inside_candidate_envelope_ratio"] = hair_candidate_ratio
        row["forehead_top_jaggedness"] = jaggedness
        final_mask, repair_applied, repair_reason = build_final_mask(
            semantic_mask,
            candidate_envelope,
            str(args.final_mask_mode),
            float(args.forehead_band_ratio),
            hair_candidate_ratio,
            float(args.hair_repair_threshold),
            jaggedness,
            float(args.jaggedness_threshold),
            bool(args.enable_jaggedness_trigger),
        )
        final_pixels = int((final_mask > 0).sum())
        if final_pixels < max(64, int(round(final_mask.size * 0.01))):
            raise alignment.SampleFailure(
                "failed_empty_selected_mask", "final_mask_too_small"
            )
        final_ratio = final_pixels / float(label_map.size)
        row.update(
            {
                "final_mask_area_pixels": final_pixels,
                "final_mask_area_ratio": final_ratio,
                "mask_area_pixels": final_pixels,
                "mask_area_ratio": final_ratio,
                "forehead_repair_applied": repair_applied,
                "forehead_repair_reason": repair_reason,
            }
        )
        if final_ratio < float(args.mask_low_warning):
            row["mask_warning"] = "warning_low_mask_area"
        elif final_ratio > float(args.mask_high_warning):
            row["mask_warning"] = "warning_high_mask_area"

        hair_pixels, hair_final_ratio = _ratio_inside_mask(
            label_map, final_mask, (parsing.CLASS_HAIR,)
        )
        _, neck_ratio = _ratio_inside_mask(
            label_map, final_mask, (parsing.CLASS_NECK,)
        )
        _, cloth_ratio = _ratio_inside_mask(
            label_map, final_mask, (parsing.CLASS_CLOTH,)
        )
        _, background_ratio = _ratio_inside_mask(
            label_map, final_mask, (parsing.CLASS_BACKGROUND,)
        )
        row.update(
            {
                "hair_inside_final_mask_pixels": hair_pixels,
                "hair_inside_final_mask_ratio": hair_final_ratio,
                "neck_inside_final_mask_ratio": neck_ratio,
                "cloth_inside_final_mask_ratio": cloth_ratio,
                "background_inside_final_mask_ratio": background_ratio,
            }
        )
        if hair_final_ratio > float(args.heavy_hair_warning):
            row["hair_warning_flag"] = "warning_heavy_hair_occlusion"
        elif hair_final_ratio > float(args.hair_warning):
            row["hair_warning_flag"] = "warning_hair_occlusion"
        row.update(parsing.compute_semantic_area_ratios(label_map))

        previews.final_overlay = _mask_overlay(
            aligned_rgb, final_mask, (255, 90, 40)
        )
        alpha = parsing.feather_mask(final_mask, int(args.feather_kernel))
        final_rgb = alignment.apply_black_background(aligned_rgb, alpha)
        previews.final = final_rgb
        alignment.save_png(output_path, final_rgb)
        row["status"] = "success"
    except alignment.SampleFailure as exc:
        row["status"] = exc.status
        row["fail_reason"] = exc.reason
    except Exception as exc:
        row["status"] = "failed_unexpected_error"
        row["fail_reason"] = f"{type(exc).__name__}: {exc}"
    if row["status"] not in STATUS_VALUES:
        row["fail_reason"] = (
            f"invalid_status={row['status']}; {row['fail_reason']}"
        )
        row["status"] = "failed_unexpected_error"
    return row, previews


def _ordered_log_columns(
    log_df: pd.DataFrame, split_columns: Sequence[str]
) -> list[str]:
    preferred = list(LOG_COLUMNS)
    extras = [
        column for column in split_columns
        if column not in preferred and column in log_df.columns
    ]
    remaining = [
        column for column in log_df.columns
        if column not in preferred and column not in extras
    ]
    return preferred + extras + remaining


def _distribution(values: pd.Series) -> dict[str, Any]:
    numeric = pd.to_numeric(values, errors="coerce").dropna()
    if numeric.empty:
        return {metric: np.nan for metric in SUMMARY_METRICS}
    return {
        "count": int(numeric.count()),
        "mean": float(numeric.mean()),
        "std": float(numeric.std(ddof=1)),
        "min": float(numeric.min()),
        "p1": float(numeric.quantile(0.01)),
        "p5": float(numeric.quantile(0.05)),
        "p25": float(numeric.quantile(0.25)),
        "median": float(numeric.quantile(0.50)),
        "p75": float(numeric.quantile(0.75)),
        "p95": float(numeric.quantile(0.95)),
        "p99": float(numeric.quantile(0.99)),
        "max": float(numeric.max()),
    }


def build_hybrid_summary(log_df: pd.DataFrame) -> pd.DataFrame:
    success = log_df[log_df["status"] == "success"].copy()
    rows: list[dict[str, Any]] = []

    def append_group(group_type: str, group_value: Any, frame: pd.DataFrame) -> None:
        for metric in SUMMARY_VALUE_COLUMNS:
            row = {
                "group_type": group_type,
                "group_value": group_value,
                "metric": metric,
            }
            row.update(_distribution(frame[metric]))
            rows.append(row)

    append_group("overall", "all", success)
    for column in ("extreme_label", "fold"):
        if column in success.columns:
            for value, group in success.groupby(column, dropna=False, sort=True):
                append_group(column, value, group)
    return pd.DataFrame(
        rows,
        columns=("group_type", "group_value", "metric", *SUMMARY_METRICS),
    )


def _format_distribution(stats: dict[str, Any]) -> str:
    return ", ".join(
        f"{key}={value:.6f}"
        if isinstance(value, float) and math.isfinite(value)
        else f"{key}={value}"
        for key, value in stats.items()
    )


def summarize_logs(
    log_df: pd.DataFrame,
    args: argparse.Namespace,
    output_dir: Path,
    logs_dir: Path,
) -> str:
    log_df.to_csv(
        logs_dir / "preprocess_log.csv", index=False, encoding="utf-8-sig"
    )
    failed = log_df[log_df["status"] != "success"].copy()
    for column in FAILED_COLUMNS:
        if column not in failed.columns:
            failed[column] = ""
    failed[list(FAILED_COLUMNS)].to_csv(
        logs_dir / "failed_cases.csv", index=False, encoding="utf-8-sig"
    )
    build_hybrid_summary(log_df).to_csv(
        logs_dir / "parsing_hybrid_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )
    total = len(log_df)
    successes = int(log_df["status"].eq("success").sum())
    success_df = log_df[log_df["status"] == "success"]
    lines = [
        "Face parsing hybrid forehead-repair preprocessing summary",
        "=" * 72,
        f"Total IDs: {total}",
        f"Successes: {successes}",
        f"Failures: {total - successes}",
        f"Success rate: {successes / total if total else 0.0:.2%}",
        f"Forehead repairs: "
        f"{int(success_df['forehead_repair_applied'].astype(bool).sum())}",
        f"Top-cut warnings: "
        f"{int(success_df['top_cut_warning'].astype(bool).sum())}",
        "",
        "Failure statuses:",
    ]
    counts = Counter(failed["status"].astype(str))
    lines.extend(
        [f"  {status}: {count}" for status, count in sorted(counts.items())]
        or ["  none"]
    )
    for column in ("fold", "extreme_label"):
        lines.extend(["", f"Success/failure by {column}:"])
        if column not in log_df.columns:
            lines.append("  column not present")
            continue
        grouped = (
            log_df.assign(success=log_df["status"].eq("success"))
            .groupby(column, dropna=False)["success"]
            .agg(total="size", success="sum")
        )
        for value, group_row in grouped.iterrows():
            success_count = int(group_row["success"])
            lines.append(
                f"  {value}: total={int(group_row['total'])}, "
                f"success={success_count}, "
                f"failed={int(group_row['total']) - success_count}"
            )
    for metric in (
        "mask_area_ratio",
        "hair_inside_final_mask_ratio",
        "neck_inside_final_mask_ratio",
        "cloth_inside_final_mask_ratio",
        "background_inside_final_mask_ratio",
    ):
        lines.extend(
            [
                "",
                f"{metric} overall:",
                f"  {_format_distribution(_distribution(success_df[metric]))}",
            ]
        )
    lines.extend(
        [
            "",
            f"Output directory: {output_dir}",
            "Parameters:",
            *[f"  {key}: {value}" for key, value in vars(args).items()],
            "",
            "Alignment source: original-coordinate direct similarity warp.",
            "Strict failure policy: enabled; no resize or old-mask fallback.",
            "Final mask default: semantic body plus forehead-band-only repair.",
            "Saved output: RGB uint8 PNG; no training normalization is saved.",
        ]
    )
    summary = "\n".join(lines)
    (logs_dir / "preprocess_summary.txt").write_text(
        summary + "\n", encoding="utf-8"
    )
    return summary


def finalize_success_qc(
    log_df: pd.DataFrame,
    dirs: dict[str, Path],
    num_qc_preview: int,
    seed: int,
) -> None:
    success = log_df[log_df["status"] == "success"].copy()
    if success.empty or num_qc_preview <= 0:
        shutil.rmtree(dirs[".staging_success"], ignore_errors=True)
        return
    numeric_columns = (
        "mask_area_ratio", "hair_inside_final_mask_ratio",
        "cloth_inside_final_mask_ratio", "neck_inside_final_mask_ratio",
    )
    for column in numeric_columns:
        success[column] = pd.to_numeric(success[column], errors="coerce")
    selections = (
        (
            "random_success",
            success.sample(n=min(num_qc_preview, len(success)), random_state=seed),
        ),
        ("semantic_only", success[~success["forehead_repair_applied"].astype(bool)]),
        ("forehead_repair", success[success["forehead_repair_applied"].astype(bool)]),
        (
            "high_hair_inside_final_mask",
            success.nlargest(min(20, len(success)), "hair_inside_final_mask_ratio"),
        ),
        (
            "high_cloth_inside_final_mask",
            success.nlargest(min(20, len(success)), "cloth_inside_final_mask_ratio"),
        ),
        (
            "high_neck_inside_final_mask",
            success.nlargest(min(20, len(success)), "neck_inside_final_mask_ratio"),
        ),
        ("low_mask_area", success.nsmallest(min(20, len(success)), "mask_area_ratio")),
        ("high_mask_area", success.nlargest(min(20, len(success)), "mask_area_ratio")),
    )
    for category, frame in selections:
        frame = frame.head(20)
        for row in frame.itertuples(index=False):
            source = dirs[".staging_success"] / f"{row.ID}.jpg"
            if source.is_file():
                try:
                    shutil.copy2(source, dirs[category] / source.name)
                except OSError as exc:
                    print(f"[warning] Failed to copy QC preview: {exc}")
    shutil.rmtree(dirs[".staging_success"], ignore_errors=True)


def _failure_qc_category(status: str) -> str | None:
    return {
        "failed_no_face": "failed_no_face",
        "failed_landmark_incomplete": "failed_alignment",
        "failed_alignment": "failed_alignment",
        "failed_parsing_model": "failed_parsing",
        "failed_empty_selected_mask": "failed_empty_mask",
    }.get(status)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    project_root = resolve_project_root(args)
    args.project_root = project_root
    args.split_csv = resolve_under_project(args.split_csv, project_root)
    args.image_dir = resolve_under_project(args.image_dir, project_root)
    args.output_dir = resolve_under_project(args.output_dir, project_root)
    if args.parsing_checkpoint is None:
        raise FileNotFoundError("--parsing-checkpoint is required")
    args.parsing_checkpoint = resolve_under_project(
        args.parsing_checkpoint, project_root
    )
    if not args.image_dir.is_dir():
        raise NotADirectoryError(f"Image directory does not exist: {args.image_dir}")
    split_df = load_split_table(args.split_csv, args.max_samples)
    parsing_device = parsing.resolve_parsing_device(str(args.parsing_device))
    parsing_model = parsing.load_face_parsing_model(
        str(args.parsing_model), args.parsing_checkpoint, parsing_device
    )
    dirs = prepare_output_dirs(
        args.output_dir, bool(args.overwrite), project_root
    )

    random.seed(int(args.seed))
    np.random.seed(int(args.seed))
    torch.manual_seed(int(args.seed))
    detector = mp.solutions.face_detection.FaceDetection(
        model_selection=1,
        min_detection_confidence=float(args.min_detection_confidence),
    )
    face_mesh = mp.solutions.face_mesh.FaceMesh(
        static_image_mode=True,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=float(args.min_detection_confidence),
        min_tracking_confidence=0.5,
    )
    rows: list[dict[str, Any]] = []
    failed_preview_counts: Counter[str] = Counter()
    try:
        total = len(split_df)
        for index, split_row in enumerate(split_df.to_dict("records"), start=1):
            image_id = str(split_row["ID"])
            print(f"[{index:04d}/{total:04d}] {image_id}")
            row, previews = process_one_sample(
                split_row, args.image_dir, dirs["images"], detector, face_mesh,
                parsing_model, parsing_device, args,
            )
            rows.append(row)
            status = str(row["status"])
            if status == "success":
                details = (
                    f"mode={row['final_mask_mode']} "
                    f"repair={row['forehead_repair_applied']} "
                    f"mask={row['mask_area_ratio']:.4f} "
                    f"hair={row['hair_inside_final_mask_ratio']:.4f} "
                    f"top_cut={row['top_cut_warning']}"
                )
                make_qc_preview(
                    dirs[".staging_success"] / f"{image_id}.jpg",
                    image_id, status, previews, details,
                )
            else:
                category = _failure_qc_category(status)
                if category and failed_preview_counts[category] < 20:
                    make_qc_preview(
                        dirs[category] / f"{image_id}.jpg",
                        image_id, status, previews, str(row["fail_reason"]),
                    )
                    failed_preview_counts[category] += 1
    finally:
        detector.close()
        face_mesh.close()
    log_df = pd.DataFrame(rows)
    columns = _ordered_log_columns(log_df, split_df.columns.tolist())
    for column in columns:
        if column not in log_df.columns:
            log_df[column] = np.nan
    log_df = log_df[columns]
    finalize_success_qc(
        log_df, dirs, int(args.num_qc_preview), int(args.seed)
    )
    summary = summarize_logs(log_df, args, args.output_dir, dirs["logs"])
    print("\n" + summary)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (
        FileNotFoundError,
        FileExistsError,
        NotADirectoryError,
        RuntimeError,
        ValueError,
    ) as exc:
        print(f"[configuration error] {exc}", file=sys.stderr)
        raise SystemExit(2) from exc
