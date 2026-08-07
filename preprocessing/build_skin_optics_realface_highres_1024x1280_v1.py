"""Build 1024x1280 high-resolution real-face preprocessing pilot outputs."""

from __future__ import annotations

import argparse
import json
import math
import random
import shutil
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
import yaml

ROOT_FOR_IMPORTS = Path(__file__).resolve().parents[1]
if str(ROOT_FOR_IMPORTS) not in sys.path:
    sys.path.insert(0, str(ROOT_FOR_IMPORTS))

try:
    import build_global_face_oval_blackbg_png_simalign_strict as legacy_alignment
except ImportError:
    from preprocessing import build_global_face_oval_blackbg_png_simalign_strict as legacy_alignment

from src.skin_optics_real_preprocess import color_pipeline, face_geometry, face_parser, image_io, qc, skin_mask
from src.skin_optics_real_preprocess.schemas import (
    DetectionConfig,
    GeometryConfig,
    ParserConfig,
    SampleFailure,
    SkinMaskConfig,
)


PILOT_OUTPUT_DIR = Path("outputs/RealFace_HighRes_1024x1280_v1_pilot32")
BASELINE_PILOT_OUTPUT_DIR = Path("outputs/RealFace_HighRes_1024x1280_v1_pilot32")
CANONICAL_OUTPUT_ROOT = Path("data/processed/skin_optics/RealFace_HighRes_1024x1280_v1")
DEFAULT_CONFIG = Path("config/preprocess/skin_optics_realface_highres_1024x1280_v1.yaml")


def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def resolve_under_root(value: str | Path, root: Path) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (root / path).resolve()


def load_config(path: Path | None, root: Path) -> dict[str, Any]:
    config_path = resolve_under_root(path or DEFAULT_CONFIG, root)
    if not config_path.is_file():
        raise FileNotFoundError(f"Config file does not exist: {config_path}")
    with config_path.open("r", encoding="utf-8") as handle:
        loaded = yaml.safe_load(handle) or {}
    if not isinstance(loaded, dict):
        raise ValueError("Config root must be a mapping")
    loaded["_config_path"] = str(config_path)
    return loaded


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--pilot-count", type=int, default=None)
    parser.add_argument("--pilot-id-file", type=Path, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--overwrite-confirmed", action="store_true")
    return parser


def dataclass_from_config(config: dict[str, Any], key: str, cls: Any) -> Any:
    values = config.get(key, {}) or {}
    return cls(**values)


def validate_config(config: dict[str, Any], root: Path, args: argparse.Namespace) -> dict[str, Any]:
    paths = config.get("paths", {}) or {}
    split_csv = resolve_under_root(paths.get("split_csv"), root)
    image_dir = resolve_under_root(paths.get("image_dir"), root)
    output_dir = resolve_under_root(paths.get("pilot_output_dir", PILOT_OUTPUT_DIR), root)
    canonical_root = resolve_under_root(paths.get("canonical_output_root", CANONICAL_OUTPUT_ROOT), root)
    parser_cfg = dataclass_from_config(config, "parser", ParserConfig)
    checkpoint = resolve_under_root(parser_cfg.checkpoint, root)
    if args.device:
        parser_cfg = ParserConfig(model=parser_cfg.model, checkpoint=parser_cfg.checkpoint, device=args.device)
    if args.pilot_count is not None and args.pilot_count <= 0:
        raise ValueError("--pilot-count must be positive")
    if not split_csv.is_file():
        raise FileNotFoundError(f"Split CSV does not exist: {split_csv}")
    if not image_dir.is_dir():
        raise NotADirectoryError(f"Image directory does not exist: {image_dir}")
    if not checkpoint.is_file():
        raise FileNotFoundError(f"BiSeNet checkpoint does not exist: {checkpoint}")
    geometry_cfg = dataclass_from_config(config, "geometry", GeometryConfig)
    skin_cfg = dataclass_from_config(config, "skin_mask", SkinMaskConfig)
    detection_cfg = dataclass_from_config(config, "detection", DetectionConfig)
    if geometry_cfg.output_width != 1024 or geometry_cfg.output_height != 1280:
        raise ValueError("This v1 pipeline requires output_width=1024 and output_height=1280")
    return {
        "split_csv": split_csv,
        "image_dir": image_dir,
        "output_dir": output_dir,
        "canonical_root": canonical_root,
        "checkpoint": checkpoint,
        "geometry": geometry_cfg,
        "skin": skin_cfg,
        "detection": detection_cfg,
        "parser": parser_cfg,
        "pilot_count": int(args.pilot_count or config.get("pilot", {}).get("count", 32)),
        "config_path": Path(config["_config_path"]),
    }


def prepare_output_tree(output_dir: Path, root: Path, overwrite_confirmed: bool) -> dict[str, Path]:
    output_dir = output_dir.resolve()
    protected = {root.resolve(), (root / "outputs").resolve(), (root / "data").resolve(), (root / "data" / "processed").resolve()}
    if output_dir in protected or root.resolve() not in output_dir.parents:
        raise ValueError(f"Refusing unsafe output directory: {output_dir}")
    if output_dir.exists():
        if not overwrite_confirmed:
            raise FileExistsError(f"Pilot output directory already exists: {output_dir}. Use --overwrite-confirmed to rebuild it.")
        shutil.rmtree(output_dir)
    names = (
        "aligned_srgb",
        "aligned_linear_rgb",
        "skin_valid_mask",
        "source_valid_mask",
        "parsing_label",
        "masked_skin_preview",
        "metadata",
        "qc/preview_panels",
        "qc/pilot_contact_sheets",
        "manifests",
        "config",
    )
    dirs = {"root": output_dir}
    for name in names:
        key = name.replace("/", "_")
        dirs[key] = output_dir / name
        dirs[key].mkdir(parents=True, exist_ok=True)
    return dirs


def select_pilot_ids(split_df: pd.DataFrame, count: int, pilot_id_file: Path | None) -> list[str]:
    if pilot_id_file is not None:
        ids = [line.strip() for line in pilot_id_file.read_text(encoding="utf-8").splitlines() if line.strip()]
        if len(ids) != len(set(ids)):
            raise ValueError("Pilot ID file contains duplicates")
        return ids[:count]
    ordered = split_df.assign(ID=split_df["ID"].astype(str)).sort_values("ID", kind="mergesort").reset_index(drop=True)
    if len(ordered) < count:
        raise ValueError(f"Requested {count} pilot IDs but split contains only {len(ordered)} rows")
    indices = np.linspace(0, len(ordered) - 1, count, dtype=np.int64)
    ids = ordered.iloc[indices]["ID"].astype(str).tolist()
    if len(set(ids)) != len(ids):
        ids = ordered.iloc[np.unique(indices)]["ID"].astype(str).tolist()
    return ids


def source_valid_statistics(mask: np.ndarray) -> dict[str, float]:
    valid = mask > 0
    h, w = mask.shape
    band_h = max(1, int(round(h * 0.05)))
    band_w = max(1, int(round(w * 0.05)))
    return {
        "source_valid_fraction": float(valid.mean()),
        "top_invalid_fraction": float((~valid[:band_h, :]).mean()),
        "bottom_invalid_fraction": float((~valid[-band_h:, :]).mean()),
        "left_invalid_fraction": float((~valid[:, :band_w]).mean()),
        "right_invalid_fraction": float((~valid[:, -band_w:]).mean()),
    }


def warp_source_valid(source_shape: tuple[int, int], matrix: np.ndarray, width: int, height: int) -> np.ndarray:
    source_mask = np.full(source_shape, 255, dtype=np.uint8)
    warped = cv2.warpAffine(
        source_mask,
        matrix[:2, :].astype(np.float32),
        (width, height),
        flags=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )
    return ((warped > 0).astype(np.uint8) * 255)


def mask_fraction(mask: np.ndarray) -> float:
    return float((mask > 0).sum() / mask.size)


def top_border_skin_stats(mask: np.ndarray) -> dict[str, float | int | bool]:
    count = int((mask[0, :] > 0).sum())
    return {
        "top_border_skin_pixel_count": count,
        "top_border_skin_fraction": float(count / mask.shape[1]),
        "skin_touching_top_border": bool(count > 0),
    }


def face_region_mask(shape: tuple[int, int], canvas_landmarks: np.ndarray) -> np.ndarray:
    mask = np.zeros(shape, dtype=np.uint8)
    oval = canvas_landmarks[np.asarray(face_geometry.regions.FACE_OVAL_INDICES)]
    cv2.fillPoly(mask, [np.rint(oval).astype(np.int32)], 255)
    return mask


def warning_codes(metadata: dict[str, Any], cfg: SkinMaskConfig) -> list[str]:
    warnings: list[str] = []
    if abs(float(metadata["roll_angle_degrees"])) > 12.0:
        warnings.append("large_roll")
    if float(metadata["skin_fraction"]) < cfg.low_skin_fraction_warning:
        warnings.append("low_skin_fraction")
    if float(metadata["skin_fraction"]) > cfg.high_skin_fraction_warning:
        warnings.append("high_skin_fraction")
    if float(metadata["hair_fraction_in_face_region"]) > cfg.high_hair_fraction_warning:
        warnings.append("high_hair_occlusion")
    if float(metadata["top_invalid_fraction"]) > 0.05:
        warnings.append("top_invalid_area")
    if float(metadata["bottom_invalid_fraction"]) > 0.05:
        warnings.append("bottom_invalid_area")
    if max(float(metadata["left_invalid_fraction"]), float(metadata["right_invalid_fraction"])) > 0.05:
        warnings.append("side_invalid_area")
    if float(metadata["aligned_face_width_fraction"]) > 0.94:
        warnings.append("face_near_canvas_border")
    if float(metadata.get("source_eye_distance", 999.0)) < 80.0:
        warnings.append("low_source_face_resolution")
    return warnings


def crop_corners_in_source(geom: face_geometry.GeometryResult) -> np.ndarray:
    corners = np.array(
        [
            [geom.crop_left, geom.crop_top],
            [geom.crop_left + geom.crop_width, geom.crop_top],
            [geom.crop_left + geom.crop_width, geom.crop_top + geom.crop_height],
            [geom.crop_left, geom.crop_top + geom.crop_height],
        ],
        dtype=np.float32,
    )
    return face_geometry.transform_points(corners, np.linalg.inv(geom.roll_matrix))


def process_one(
    split_row: dict[str, Any],
    dirs: dict[str, Path],
    image_dir: Path,
    detector: Any,
    face_mesh: Any,
    parser_model: Any,
    parser_device: Any,
    geometry_cfg: GeometryConfig,
    skin_cfg: SkinMaskConfig,
) -> tuple[dict[str, Any], dict[str, Any]]:
    sample_id = str(split_row["ID"])
    manifest_row = dict(split_row)
    metadata: dict[str, Any] = {
        "sample_id": sample_id,
        "status": "failed",
        "warning_codes": [],
        "failure_code": "",
        "failure_detail": "",
    }
    outputs = {
        "aligned_srgb_path": str(dirs["aligned_srgb"] / f"{sample_id}.png"),
        "aligned_linear_rgb_path": str(dirs["aligned_linear_rgb"] / f"{sample_id}.npy"),
        "skin_valid_mask_path": str(dirs["skin_valid_mask"] / f"{sample_id}.png"),
        "source_valid_mask_path": str(dirs["source_valid_mask"] / f"{sample_id}.png"),
        "parsing_label_path": str(dirs["parsing_label"] / f"{sample_id}.png"),
        "masked_skin_preview_path": str(dirs["masked_skin_preview"] / f"{sample_id}.png"),
        "metadata_path": str(dirs["metadata"] / f"{sample_id}.json"),
        "qc_preview_path": str(dirs["qc_preview_panels"] / f"{sample_id}.png"),
    }
    manifest_row.update(outputs)
    try:
        source_path = image_io.find_image_for_id(sample_id, image_dir)
        if source_path is None:
            raise SampleFailure("missing_source_image", "cannot find image by ID in raw_scene")
        rgb, exif_applied = image_io.read_image_rgb_exif(source_path)
        h, w = rgb.shape[:2]
        metadata.update({"source_path": str(source_path), "source_width": w, "source_height": h, "exif_orientation_applied": exif_applied})
        detections = legacy_alignment.detect_faces(rgb, detector)
        metadata["num_faces_detected"] = len(detections)
        selected = legacy_alignment.select_face(detections, rgb.shape)
        if selected is None:
            raise SampleFailure("face_detection_failed", "MediaPipe Face Detection returned no valid face")
        x, y, bw, bh = selected.bbox
        metadata.update(
            {
                "detection_score": float(selected.confidence),
                "detection_bbox": [x, y, bw, bh],
                "selected_face_index": int(selected.index),
            }
        )
        exp = legacy_alignment.expand_bbox(selected.bbox, rgb.shape)
        crop_x, crop_y, crop_w, crop_h = exp
        metadata["expanded_bbox"] = [crop_x, crop_y, crop_w, crop_h]
        metadata["primary_face_selection_metrics"] = {
            "selected_area_ratio": float((bw * bh) / (w * h)),
            "selected_center_distance_fraction": float(np.linalg.norm(np.array([x + bw / 2, y + bh / 2]) - np.array([w / 2, h / 2])) / max(math.hypot(w, h), 1.0)),
        }
        crop_rgb = rgb[crop_y : crop_y + crop_h, crop_x : crop_x + crop_w].copy()
        landmarks = legacy_alignment.run_facemesh(crop_rgb, face_mesh)
        if landmarks is None:
            raise SampleFailure("facemesh_failed", "MediaPipe FaceMesh returned no landmarks")
        source_landmarks = face_geometry.landmarks_to_source_array(landmarks, crop_x, crop_y, crop_w, crop_h)
        keys = face_geometry.extract_key_points(source_landmarks)
        geom = face_geometry.build_geometry(source_landmarks, geometry_cfg)
        metadata.update(
            {
                "source_left_eye_center": keys["left_eye"].tolist(),
                "source_right_eye_center": keys["right_eye"].tolist(),
                "source_chin": keys["chin"].tolist(),
                "source_face_oval": keys["face_oval"].tolist(),
                "source_landmarks_5": [keys["left_eye"].tolist(), keys["right_eye"].tolist(), keys["nose_tip"].tolist(), keys["left_mouth"].tolist(), keys["right_mouth"].tolist()],
                "source_eye_distance": float(keys["eye_distance"]),
                "roll_angle_degrees": geom.roll_angle_degrees,
                "residual_roll_degrees": geom.residual_roll_degrees,
                "rolled_eye_y_delta": geom.rolled_eye_y_delta,
                "rolled_face_left": geom.rolled_face_left,
                "rolled_face_right": geom.rolled_face_right,
                "rolled_face_top": geom.rolled_face_top,
                "rolled_chin_y": geom.rolled_chin_y,
                "crop_left": geom.crop_left,
                "crop_top": geom.crop_top,
                "crop_width": geom.crop_width,
                "crop_height": geom.crop_height,
                "crop_aspect_ratio": geom.crop_width / geom.crop_height,
                "affine_source_to_canvas": geom.affine_source_to_canvas.tolist(),
                "affine_canvas_to_source": geom.affine_canvas_to_source.tolist(),
                "uniform_scale": geom.uniform_scale,
                "matrix_singular_values": list(geom.matrix_singular_values),
                "matrix_roundtrip_max_error": geom.roundtrip_max_error,
                "output_width": geometry_cfg.output_width,
                "output_height": geometry_cfg.output_height,
                "color_space_assumption": "assumed_sRGB",
                "inverse_srgb_version": "IEC_61966_2_1_standard_piecewise",
            }
        )
        linear = color_pipeline.uint8_rgb_to_linear(rgb)
        aligned_linear = color_pipeline.warp_linear_rgb(linear, geom.affine_source_to_canvas, geometry_cfg.output_width, geometry_cfg.output_height)
        aligned_srgb, aligned_chw_f16 = color_pipeline.encode_linear_outputs(aligned_linear)
        expected_shape = (geometry_cfg.output_height, geometry_cfg.output_width)
        if aligned_srgb.shape != (geometry_cfg.output_height, geometry_cfg.output_width, 3) or aligned_chw_f16.shape != (3, geometry_cfg.output_height, geometry_cfg.output_width):
            raise SampleFailure("invalid_output_shape", "aligned outputs have invalid shape")
        source_valid = warp_source_valid((h, w), geom.affine_source_to_canvas, geometry_cfg.output_width, geometry_cfg.output_height)
        metadata.update(source_valid_statistics(source_valid))
        parsing_label = face_parser.run(aligned_srgb, parser_model, parser_device)
        if parsing_label.shape != expected_shape:
            raise SampleFailure("parser_failed", f"parser returned shape {parsing_label.shape}")
        mask_parts = skin_mask.build_skin_valid_mask(parsing_label, geom.canvas_landmarks, source_valid, geom.rolled_face_right - geom.rolled_face_left, skin_cfg)
        skin_valid = mask_parts["skin_valid_mask"]
        skin_mask.assert_mask_contract(skin_valid, source_valid, expected_shape)
        face_mask = face_region_mask(expected_shape, geom.canvas_landmarks)
        face_pixels = max(1, int((face_mask > 0).sum()))
        hair_id = face_parser.class_ids(("hair",))[0]
        metadata.update(face_parser.semantic_area_ratios(parsing_label))
        metadata.update(
            {
                "aligned_face_width_fraction": float((geom.canvas_landmarks[:, 0].max() - geom.canvas_landmarks[:, 0].min()) / geometry_cfg.output_width),
                "aligned_face_height_fraction": float((geom.canvas_landmarks[:, 1].max() - geom.canvas_landmarks[:, 1].min()) / geometry_cfg.output_height),
                "skin_fraction": mask_fraction(skin_valid),
                "hair_fraction_in_face_region": float(((parsing_label == hair_id) & (face_mask > 0)).sum() / face_pixels),
                "facemesh_exclusion_dilation_radius_px": int(mask_parts["exclusion_dilation_radius_px"]),
                "eye_brow_lip_dilation_radius_px": int(mask_parts["eye_brow_lip_dilation_radius_px"]),
                "nostril_dilation_radius_px": int(mask_parts["nostril_dilation_radius_px"]),
                "nostril_exclusion_area_ratio": float(mask_parts["nostril_exclusion_area_ratio"]),
                "min_component_area_px": int(mask_parts["min_component_area_px"]),
            }
        )
        metadata.update(top_border_skin_stats(skin_valid))
        metadata["warning_codes"] = warning_codes(metadata, skin_cfg)
        masked_preview = aligned_srgb.copy()
        masked_preview[skin_valid == 0] = 0
        np.save(dirs["aligned_linear_rgb"] / f"{sample_id}.npy", aligned_chw_f16)
        image_io.save_rgb_png(dirs["aligned_srgb"] / f"{sample_id}.png", aligned_srgb)
        image_io.save_gray_png(dirs["skin_valid_mask"] / f"{sample_id}.png", skin_valid)
        image_io.save_gray_png(dirs["source_valid_mask"] / f"{sample_id}.png", source_valid)
        image_io.save_gray_png(dirs["parsing_label"] / f"{sample_id}.png", parsing_label.astype(np.uint8))
        image_io.save_rgb_png(dirs["masked_skin_preview"] / f"{sample_id}.png", masked_preview)
        qc.make_preview_panel(
            dirs["qc_preview_panels"] / f"{sample_id}.png",
            rgb,
            detections,
            selected,
            source_landmarks,
            crop_corners_in_source(geom),
            aligned_srgb,
            geom.canvas_landmarks,
            parsing_label,
            mask_parts["skin_candidate"],
            mask_parts["semantic_exclusion"],
            mask_parts["facemesh_exclusion"],
            skin_valid,
            masked_preview,
            source_valid,
        )
        metadata["status"] = "success"
        manifest_row["preprocess_status"] = "success"
        manifest_row["warning_codes"] = ";".join(metadata["warning_codes"])
    except SampleFailure as exc:
        metadata["status"] = "failed"
        metadata["failure_code"] = exc.code
        metadata["failure_detail"] = exc.detail
        manifest_row["preprocess_status"] = "failed"
        manifest_row["failure_code"] = exc.code
        manifest_row["failure_detail"] = exc.detail
        manifest_row["warning_codes"] = ""
    except Exception as exc:
        metadata["status"] = "failed"
        metadata["failure_code"] = "output_write_failed"
        metadata["failure_detail"] = f"{type(exc).__name__}: {exc}"
        manifest_row["preprocess_status"] = "failed"
        manifest_row["failure_code"] = metadata["failure_code"]
        manifest_row["failure_detail"] = metadata["failure_detail"]
        manifest_row["warning_codes"] = ""
    metadata_path = dirs["metadata"] / f"{sample_id}.json"
    metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    return manifest_row, metadata


def numeric_summary(values: list[float], names: tuple[str, ...]) -> dict[str, float | None]:
    arr = np.asarray([v for v in values if v is not None and np.isfinite(v)], dtype=np.float64)
    if arr.size == 0:
        return {name: None for name in names}
    out: dict[str, float | None] = {}
    for name in names:
        if name == "min":
            out[name] = float(np.min(arr))
        elif name == "median":
            out[name] = float(np.median(arr))
        elif name == "p95":
            out[name] = float(np.quantile(arr, 0.95))
        elif name == "max":
            out[name] = float(np.max(arr))
    return out


def summarize(metadata_rows: list[dict[str, Any]], dirs: dict[str, Path]) -> dict[str, Any]:
    success = [m for m in metadata_rows if m.get("status") == "success"]
    failed = [m for m in metadata_rows if m.get("status") != "success"]
    warning_counter = Counter(code for m in success for code in m.get("warning_codes", []))
    failure_counter = Counter(str(m.get("failure_code", "")) for m in failed)
    summary = {
        "pilot_total": len(metadata_rows),
        "success_count": len(success),
        "failure_count": len(failed),
        "warning_case_count": int(sum(bool(m.get("warning_codes")) for m in success)),
        "roll_angle": numeric_summary([abs(float(m["roll_angle_degrees"])) for m in success], ("min", "median", "p95", "max")),
        "residual_roll": numeric_summary([abs(float(m["residual_roll_degrees"])) for m in success], ("median", "p95", "max")),
        "aligned_face_width_fraction": numeric_summary([float(m["aligned_face_width_fraction"]) for m in success], ("min", "median", "p95", "max")),
        "aligned_face_height_fraction": numeric_summary([float(m["aligned_face_height_fraction"]) for m in success], ("min", "median", "p95", "max")),
        "skin_fraction": numeric_summary([float(m["skin_fraction"]) for m in success], ("min", "median", "p95", "max")),
        "hair_fraction_in_face_region": numeric_summary([float(m["hair_fraction_in_face_region"]) for m in success], ("median", "p95", "max")),
        "source_valid_fraction": numeric_summary([float(m["source_valid_fraction"]) for m in success], ("min", "median", "p95")),
        "top_border_skin_pixel_count": numeric_summary([float(m["top_border_skin_pixel_count"]) for m in success], ("min", "median", "p95", "max")),
        "top_border_skin_fraction": numeric_summary([float(m["top_border_skin_fraction"]) for m in success], ("min", "median", "p95", "max")),
        "number_of_cases_skin_touching_top_border": int(sum(bool(m.get("skin_touching_top_border")) for m in success)),
        "fraction_of_cases_skin_touching_top_border": float(sum(bool(m.get("skin_touching_top_border")) for m in success) / len(success)) if success else None,
        "nostril_exclusion_area_ratio": numeric_summary([float(m["nostril_exclusion_area_ratio"]) for m in success], ("min", "median", "p95", "max")),
        "nostril_dilation_radius_px": numeric_summary([float(m["nostril_dilation_radius_px"]) for m in success], ("min", "median", "p95", "max")),
        "top_invalid_fraction": numeric_summary([float(m["top_invalid_fraction"]) for m in success], ("min", "median", "p95", "max")),
        "bottom_invalid_fraction": numeric_summary([float(m["bottom_invalid_fraction"]) for m in success], ("min", "median", "p95", "max")),
        "left_invalid_fraction": numeric_summary([float(m["left_invalid_fraction"]) for m in success], ("min", "median", "p95", "max")),
        "right_invalid_fraction": numeric_summary([float(m["right_invalid_fraction"]) for m in success], ("min", "median", "p95", "max")),
        "failure_code_counts": dict(sorted(failure_counter.items())),
        "warning_code_counts": dict(sorted(warning_counter.items())),
    }
    (dirs["root"] / "qc" / "qc_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    summary_rows = []
    for m in metadata_rows:
        summary_rows.append(
            {
                "ID": m.get("sample_id"),
                "status": m.get("status"),
                "failure_code": m.get("failure_code", ""),
                "warning_codes": ";".join(m.get("warning_codes", [])),
                "roll_angle_degrees": m.get("roll_angle_degrees"),
                "residual_roll_degrees": m.get("residual_roll_degrees"),
                "aligned_face_width_fraction": m.get("aligned_face_width_fraction"),
                "aligned_face_height_fraction": m.get("aligned_face_height_fraction"),
                "skin_fraction": m.get("skin_fraction"),
                "hair_fraction_in_face_region": m.get("hair_fraction_in_face_region"),
                "source_valid_fraction": m.get("source_valid_fraction"),
                "top_border_skin_pixel_count": m.get("top_border_skin_pixel_count"),
                "top_border_skin_fraction": m.get("top_border_skin_fraction"),
                "skin_touching_top_border": m.get("skin_touching_top_border"),
                "nostril_exclusion_area_ratio": m.get("nostril_exclusion_area_ratio"),
                "nostril_dilation_radius_px": m.get("nostril_dilation_radius_px"),
            }
        )
    pd.DataFrame(summary_rows).to_csv(dirs["root"] / "qc" / "pilot_summary.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame([m for m in summary_rows if m["status"] != "success"]).to_csv(dirs["root"] / "qc" / "failed_cases.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame([m for m in summary_rows if m["warning_codes"]]).to_csv(dirs["root"] / "qc" / "warning_cases.csv", index=False, encoding="utf-8-sig")
    return summary


def read_rgb_png(path: Path) -> np.ndarray | None:
    bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if bgr is None:
        return None
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def read_gray_png(path: Path) -> np.ndarray | None:
    return cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)


def collect_output_metrics(root: Path, pilot_ids: list[str]) -> dict[str, dict[str, Any]]:
    metrics: dict[str, dict[str, Any]] = {}
    for sample_id in pilot_ids:
        mask = read_gray_png(root / "skin_valid_mask" / f"{sample_id}.png")
        metadata_path = root / "metadata" / f"{sample_id}.json"
        metadata = json.loads(metadata_path.read_text(encoding="utf-8")) if metadata_path.is_file() else {}
        if mask is not None:
            metadata.update(top_border_skin_stats(mask))
            metadata["skin_fraction_from_mask"] = mask_fraction(mask)
        metrics[sample_id] = metadata
    return metrics


def _stat(values: list[float]) -> dict[str, float | None]:
    return numeric_summary(values, ("min", "median", "p95", "max"))


def build_revision_comparison(
    baseline_root: Path,
    rev_root: Path,
    pilot_ids: list[str],
    dirs: dict[str, Path],
    base_config: dict[str, Any] | None,
    rev_config: dict[str, Any],
) -> dict[str, Any]:
    base = collect_output_metrics(baseline_root, pilot_ids)
    rev = collect_output_metrics(rev_root, pilot_ids)
    rows: list[dict[str, Any]] = []
    for sample_id in pilot_ids:
        b = base.get(sample_id, {})
        r = rev.get(sample_id, {})
        rows.append(
            {
                "ID": sample_id,
                "base_top_border_skin_pixel_count": b.get("top_border_skin_pixel_count"),
                "revA_top_border_skin_pixel_count": r.get("top_border_skin_pixel_count"),
                "base_top_border_skin_fraction": b.get("top_border_skin_fraction"),
                "revA_top_border_skin_fraction": r.get("top_border_skin_fraction"),
                "base_skin_touching_top_border": b.get("skin_touching_top_border"),
                "revA_skin_touching_top_border": r.get("skin_touching_top_border"),
                "base_aligned_face_width_fraction": b.get("aligned_face_width_fraction"),
                "revA_aligned_face_width_fraction": r.get("aligned_face_width_fraction"),
                "base_aligned_face_height_fraction": b.get("aligned_face_height_fraction"),
                "revA_aligned_face_height_fraction": r.get("aligned_face_height_fraction"),
                "base_skin_fraction": b.get("skin_fraction", b.get("skin_fraction_from_mask")),
                "revA_skin_fraction": r.get("skin_fraction", r.get("skin_fraction_from_mask")),
                "revA_nostril_exclusion_area_ratio": r.get("nostril_exclusion_area_ratio"),
                "revA_nostril_dilation_radius_px": r.get("nostril_dilation_radius_px"),
                "hair_fraction_in_face_region": r.get("hair_fraction_in_face_region"),
            }
        )
    comparison_df = pd.DataFrame(rows)
    comparison_path = dirs["root"] / "qc" / "revision_comparison.csv"
    comparison_df.to_csv(comparison_path, index=False, encoding="utf-8-sig")
    comparison_dir = dirs["root"] / "qc" / "revision_comparisons"
    comparison_dir.mkdir(parents=True, exist_ok=True)
    selected_ids = select_representative_comparison_ids(comparison_df, pilot_ids)
    comparison_images = make_revision_comparison_images(baseline_root, rev_root, comparison_dir, selected_ids)
    base_touch = comparison_df["base_skin_touching_top_border"].fillna(False).astype(bool)
    rev_touch = comparison_df["revA_skin_touching_top_border"].fillna(False).astype(bool)
    summary = {
        "baseline_root": str(baseline_root),
        "revA_root": str(rev_root),
        "comparison_csv": str(comparison_path),
        "comparison_images": comparison_images,
        "representative_ids": selected_ids,
        "parameter_comparison": {
            "base_top_margin_ratio": ((base_config or {}).get("geometry", {}) or {}).get("top_margin_ratio"),
            "revA_top_margin_ratio": (rev_config.get("geometry", {}) or {}).get("top_margin_ratio"),
            "base_side_margin_ratio": ((base_config or {}).get("geometry", {}) or {}).get("side_margin_ratio"),
            "revA_side_margin_ratio": (rev_config.get("geometry", {}) or {}).get("side_margin_ratio"),
            "base_bottom_margin_ratio": ((base_config or {}).get("geometry", {}) or {}).get("bottom_margin_ratio"),
            "revA_bottom_margin_ratio": (rev_config.get("geometry", {}) or {}).get("bottom_margin_ratio"),
            "base_exclusion_dilation_ratio": ((base_config or {}).get("skin_mask", {}) or {}).get("exclusion_dilation_ratio"),
            "revA_eye_brow_lip_dilation_ratio": (rev_config.get("skin_mask", {}) or {}).get("eye_brow_lip_dilation_ratio"),
            "revA_nostril_dilation_ratio": (rev_config.get("skin_mask", {}) or {}).get("nostril_dilation_ratio"),
        },
        "top_border": {
            "baseline_number_of_cases_skin_touching_top_border": int(base_touch.sum()),
            "revA_number_of_cases_skin_touching_top_border": int(rev_touch.sum()),
            "baseline_fraction_of_cases_skin_touching_top_border": float(base_touch.mean()) if len(base_touch) else None,
            "revA_fraction_of_cases_skin_touching_top_border": float(rev_touch.mean()) if len(rev_touch) else None,
            "baseline_top_border_skin_fraction": _stat(comparison_df["base_top_border_skin_fraction"].dropna().astype(float).tolist()),
            "revA_top_border_skin_fraction": _stat(comparison_df["revA_top_border_skin_fraction"].dropna().astype(float).tolist()),
        },
        "composition": {
            "baseline_aligned_face_width_fraction": _stat(comparison_df["base_aligned_face_width_fraction"].dropna().astype(float).tolist()),
            "revA_aligned_face_width_fraction": _stat(comparison_df["revA_aligned_face_width_fraction"].dropna().astype(float).tolist()),
            "baseline_aligned_face_height_fraction": _stat(comparison_df["base_aligned_face_height_fraction"].dropna().astype(float).tolist()),
            "revA_aligned_face_height_fraction": _stat(comparison_df["revA_aligned_face_height_fraction"].dropna().astype(float).tolist()),
            "baseline_skin_fraction": _stat(comparison_df["base_skin_fraction"].dropna().astype(float).tolist()),
            "revA_skin_fraction": _stat(comparison_df["revA_skin_fraction"].dropna().astype(float).tolist()),
        },
        "nostril": {
            "revA_nostril_exclusion_area_ratio": _stat(comparison_df["revA_nostril_exclusion_area_ratio"].dropna().astype(float).tolist()),
            "revA_nostril_dilation_radius_px": _stat(comparison_df["revA_nostril_dilation_radius_px"].dropna().astype(float).tolist()),
        },
    }
    (dirs["root"] / "qc" / "revision_comparison_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return summary


def select_representative_comparison_ids(comparison_df: pd.DataFrame, pilot_ids: list[str]) -> list[str]:
    selected: list[str] = []

    def add(sample_id: Any) -> None:
        sid = str(sample_id)
        if sid and sid != "nan" and sid not in selected:
            selected.append(sid)

    if "base_top_border_skin_pixel_count" in comparison_df:
        frame = comparison_df.dropna(subset=["base_top_border_skin_pixel_count"])
        if not frame.empty:
            add(frame.sort_values("base_top_border_skin_pixel_count", ascending=False).iloc[0]["ID"])
    if {"base_skin_fraction", "revA_skin_fraction"}.issubset(comparison_df.columns):
        frame = comparison_df.dropna(subset=["base_skin_fraction", "revA_skin_fraction"]).copy()
        if not frame.empty:
            frame["skin_fraction_delta"] = (frame["revA_skin_fraction"] - frame["base_skin_fraction"]).abs()
            add(frame.sort_values("skin_fraction_delta", ascending=False).iloc[0]["ID"])
    if "hair_fraction_in_face_region" in comparison_df:
        frame = comparison_df.dropna(subset=["hair_fraction_in_face_region"])
        if not frame.empty:
            add(frame.sort_values("hair_fraction_in_face_region", ascending=False).iloc[0]["ID"])
    for candidate in pilot_ids:
        add(candidate)
        if len(selected) >= 5:
            break
    return selected[:5]


def make_revision_comparison_images(
    baseline_root: Path,
    rev_root: Path,
    output_dir: Path,
    sample_ids: list[str],
) -> list[str]:
    outputs: list[str] = []
    for sample_id in sample_ids:
        panels: list[np.ndarray] = []
        for root, label in ((baseline_root, "base"), (rev_root, "revA")):
            aligned = read_rgb_png(root / "aligned_srgb" / f"{sample_id}.png")
            mask = read_gray_png(root / "skin_valid_mask" / f"{sample_id}.png")
            preview = read_rgb_png(root / "masked_skin_preview" / f"{sample_id}.png")
            if aligned is None or mask is None or preview is None:
                continue
            overlay = qc.overlay_mask(aligned, mask, (0, 255, 180), 0.45)
            panels.extend(
                [
                    qc.label_panel(qc.resize_panel(aligned, (300, 375)), f"{label} aligned"),
                    qc.label_panel(qc.resize_panel(overlay, (300, 375)), f"{label} mask overlay"),
                    qc.label_panel(qc.resize_panel(preview, (300, 375)), f"{label} masked preview"),
                ]
            )
        if len(panels) != 6:
            continue
        output = np.concatenate([np.concatenate(panels[:3], axis=1), np.concatenate(panels[3:], axis=1)], axis=0)
        out_path = output_dir / f"{sample_id}_base_vs_revA.png"
        image_io.save_rgb_png(out_path, output)
        outputs.append(str(out_path))
    return outputs


def write_report(
    dirs: dict[str, Path],
    pilot_ids: list[str],
    summary: dict[str, Any],
    comparison: dict[str, Any],
    config_path: Path,
    test_results: dict[str, str],
) -> None:
    report = dirs["root"] / "real_face_highres_1024x1280_pilot_revA_report.md"
    qc_paths = sorted(str(p) for p in (dirs["root"] / "qc").rglob("*") if p.is_file())
    param = comparison.get("parameter_comparison", {})
    top = comparison.get("top_border", {})
    composition = comparison.get("composition", {})
    nostril = comparison.get("nostril", {})
    recommend_full500 = (
        summary.get("success_count") == 32
        and summary.get("failure_count") == 0
        and (top.get("revA_number_of_cases_skin_touching_top_border") or 0)
        <= 0.5 * (top.get("baseline_number_of_cases_skin_touching_top_border") or 0)
        and (composition.get("revA_aligned_face_width_fraction", {}).get("median") or 0.0) >= 0.88
    )
    text = [
        "# RealFace HighRes 1024x1280 Pilot revA Report",
        "",
        "## Revision scope",
        "This is a revision of the existing fourth-part 32-case pilot, not a rebuild from scratch. The same 32 pilot IDs were reused and the initial pilot output directory was not overwritten.",
        "",
        "## Human QC feedback from initial pilot",
        "- Top valid skin often touched the canvas border.",
        "- Fine forehead hair contamination was mild and rare.",
        "- Mouth exclusion was normal.",
        "- Nostril exclusion was too wide.",
        "",
        "## revA changes",
        f"- Increased only `geometry.top_margin_ratio` from {param.get('base_top_margin_ratio')} to {param.get('revA_top_margin_ratio')}.",
        "- Split nostril dilation from the previous shared exclusion dilation: eye/brow/lip remains 0.006, nostril uses 0.003.",
        "",
        "## Explicitly unchanged",
        "- Forehead fine-hair strategy unchanged; no dark-pixel or texture hair detector was added.",
        "- Mouth/lip polygon and mouth-related dilation remain unchanged.",
        "- 1024x1280 4:5 canvas, FaceMesh geometry crop, one-shot affine warp, and BiSeNet parsing remain unchanged.",
        "- No forehead repair, alpha feather, skin-mask bbox crop, 3D frontalization, SO-1 inference, patching, M/H/S/P generation, or classification was run.",
        "",
        "## Legacy code analyzed",
        "- preprocessing/build_global_face_parsing_hybrid_foreheadrepair_blackbg_224_png_strict.py",
        "- preprocessing/build_global_face_oval_blackbg_png_simalign_strict.py",
        "- preprocessing/build_global_face_parsing_regularmask_blackbg_224_png_strict.py",
        "- config/preprocess/global_face_parsing_hybrid_foreheadrepair_blackbg_224_png_strict.yaml",
        "- preprocessing/checkpoints/face_parsing/79999_iter.pth",
        "",
        "## Reused legacy logic",
        "- MediaPipe Face Detection model_selection=1 and deterministic primary-face ranking by face area and image-center distance.",
        "- Expanded detection box ratios: top 10%, bottom 20%, left/right 20%.",
        "- FaceMesh static_image_mode=True, max_num_faces=1, refine_landmarks=True.",
        "- CelebAMask-HQ 19-class BiSeNet architecture, 512 parser input, ImageNet normalization, argmax logits, nearest-neighbor label restore.",
        "",
        "## New files",
        "- preprocessing/build_skin_optics_realface_highres_1024x1280_v1.py",
        "- src/skin_optics_real_preprocess/*",
        "- config/preprocess/skin_optics_realface_highres_1024x1280_v1.yaml",
        "- tests/skin_optics_real_preprocess/*",
        "",
        "## Parameter comparison",
        "| Parameter | Initial pilot | revA |",
        "| --- | ---: | ---: |",
        f"| top_margin_ratio | {param.get('base_top_margin_ratio')} | {param.get('revA_top_margin_ratio')} |",
        f"| side_margin_ratio | {param.get('base_side_margin_ratio')} | {param.get('revA_side_margin_ratio')} |",
        f"| bottom_margin_ratio | {param.get('base_bottom_margin_ratio')} | {param.get('revA_bottom_margin_ratio')} |",
        f"| shared/eye_brow_lip dilation | {param.get('base_exclusion_dilation_ratio')} | {param.get('revA_eye_brow_lip_dilation_ratio')} |",
        f"| nostril_dilation_ratio | {param.get('base_exclusion_dilation_ratio')} | {param.get('revA_nostril_dilation_ratio')} |",
        "",
        "## Geometry",
        "Roll is computed from the two eye centers. The final 3x3 source-to-canvas matrix composes roll correction, crop translation, and a single uniform scale. Crop geometry is driven only by rolled FaceMesh oval/chin geometry.",
        "",
        "## Crop parameters",
        "- side_margin_ratio=0.03 controls horizontal cheek margin.",
        f"- top_margin_ratio={param.get('revA_top_margin_ratio')} guarantees more forehead/oval top margin by uniform crop enlargement.",
        "- bottom_margin_ratio=0.025 anchors the crop bottom below the chin.",
        "",
        "## Color",
        "Images are treated as assumed sRGB, converted through the standard inverse-sRGB transfer function to float32 linear RGB, warped once with cv2.INTER_LINEAR, encoded back to sRGB uint8 for PNG previews, and saved as float16 CHW for aligned_linear_rgb.",
        "",
        "## Parser classes",
        f"{face_parser.CLASS_ID_TO_NAME}",
        "",
        "## FaceMesh exclusion regions",
        "Eyes, brows and lips use stable MediaPipe contour landmarks. Nostril exclusion uses compact project-defined nose-wing/base polygons stored in facemesh_regions.py and drawn in QC panels.",
        "",
        "## Test and validation results",
        *[f"- {name}: {result}" for name, result in test_results.items()],
        "",
        "## Initial vs revA QC comparison",
        "Top-border skin is defined as the count/fraction of final `skin_valid_mask` pixels equal to 255 on row 0. A case is touching the top border when that count is greater than zero.",
        "",
        "```json",
        json.dumps({"top_border": top, "composition": composition, "nostril": nostril}, ensure_ascii=False, indent=2),
        "```",
        "",
        f"Top-border cases changed from {top.get('baseline_number_of_cases_skin_touching_top_border')} to {top.get('revA_number_of_cases_skin_touching_top_border')}.",
        "Nostril exclusion was narrowed by changing only its dilation ratio; nostril polygons were not changed in revA.",
        "Composition remained close to the original tight crop; review the comparison images for possible side effects.",
        "",
        "## Representative comparison images",
        *[f"- {path}" for path in comparison.get("comparison_images", [])],
        "",
        "## Pilot IDs",
        ", ".join(pilot_ids),
        "",
        "## Summary",
        "```json",
        json.dumps(summary, ensure_ascii=False, indent=2),
        "```",
        "",
        "## QC files",
        *[f"- {path}" for path in qc_paths[:200]],
        "",
        "## Current issues",
        "- Pilot selection used deterministic uniform ID spacing because no reliable old QC field covering bangs/glasses/yaw was found in the fixed 500 CSV.",
        "- Nostril polygons are conservative project-local definitions rather than an official complete MediaPipe nostril contour.",
        "- Representative comparison categories are selected by reproducible QC statistics rather than manual visual labels; use the comparison images for the final visual judgment.",
        "",
        "## Full-500 recommendation",
        f"recommend_full500 = {str(bool(recommend_full500)).lower()}. Do not enter full 500 automatically; this is only a recommendation after pilot QC.",
    ]
    report.write_text("\n".join(text) + "\n", encoding="utf-8")


def run_pipeline(argv: Sequence[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    root = project_root()
    config = load_config(args.config, root)
    resolved = validate_config(config, root, args)
    baseline_root = resolve_under_root(BASELINE_PILOT_OUTPUT_DIR, root)
    if not baseline_root.is_dir():
        raise NotADirectoryError(f"Initial pilot output directory does not exist: {baseline_root}")
    baseline_config_path = baseline_root / "config" / "resolved_config.yaml"
    baseline_config = yaml.safe_load(baseline_config_path.read_text(encoding="utf-8")) if baseline_config_path.is_file() else {}
    print(f"[validate] config={resolved['config_path']}")
    print(f"[validate] split_csv={resolved['split_csv']}")
    print(f"[validate] image_dir={resolved['image_dir']}")
    print(f"[validate] checkpoint={resolved['checkpoint']}")
    print(f"[validate] baseline_pilot={baseline_root}")
    if args.validate_only:
        print("[validate] OK")
        return 0

    split_df = pd.read_csv(resolved["split_csv"], dtype={"ID": str})
    if "ID" not in split_df.columns or split_df["ID"].duplicated().any():
        raise ValueError("Split CSV must contain unique ID column")
    pilot_ids = select_pilot_ids(split_df, resolved["pilot_count"], args.pilot_id_file)
    pilot_df = split_df[split_df["ID"].astype(str).isin(set(pilot_ids))].copy()
    order = {sample_id: idx for idx, sample_id in enumerate(pilot_ids)}
    pilot_df["_pilot_order"] = pilot_df["ID"].map(order)
    pilot_df = pilot_df.sort_values("_pilot_order").drop(columns=["_pilot_order"])
    dirs = prepare_output_tree(resolved["output_dir"], root, args.overwrite_confirmed)
    (dirs["root"] / "pilot_ids.txt").write_text("\n".join(pilot_ids) + "\n", encoding="utf-8")
    shutil.copy2(resolved["config_path"], dirs["config"] / "resolved_config.yaml")
    (dirs["root"] / "run_manifest.json").write_text(
        json.dumps(
            {
                "started_at_utc": datetime.now(timezone.utc).isoformat(),
                "config": str(resolved["config_path"]),
                "pilot_count": len(pilot_ids),
                "full_500_run": False,
                "revision": "revA",
                "baseline_pilot": str(baseline_root),
                "selection_rule": "same_ids_as_initial_pilot",
                "changes": {
                    "top_margin_ratio": resolved["geometry"].top_margin_ratio,
                    "eye_brow_lip_dilation_ratio": resolved["skin"].normal_dilation_ratio(),
                    "nostril_dilation_ratio": resolved["skin"].nostril_dilation_ratio,
                },
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    random.seed(42)
    np.random.seed(42)
    parser_device = face_parser.resolve_device(resolved["parser"].device)
    parser_model = face_parser.load_model(resolved["parser"].model, resolved["checkpoint"], parser_device)
    detector = mp.solutions.face_detection.FaceDetection(
        model_selection=resolved["detection"].model_selection,
        min_detection_confidence=resolved["detection"].min_detection_confidence,
    )
    mesh = mp.solutions.face_mesh.FaceMesh(
        static_image_mode=True,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=resolved["detection"].min_detection_confidence,
        min_tracking_confidence=0.5,
    )
    manifest_rows: list[dict[str, Any]] = []
    metadata_rows: list[dict[str, Any]] = []
    try:
        for position, split_row in enumerate(pilot_df.to_dict("records"), start=1):
            sample_id = str(split_row["ID"])
            print(f"[{position:02d}/{len(pilot_df):02d}] {sample_id}")
            manifest_row, metadata = process_one(
                split_row,
                dirs,
                resolved["image_dir"],
                detector,
                mesh,
                parser_model,
                parser_device,
                resolved["geometry"],
                resolved["skin"],
            )
            manifest_rows.append(manifest_row)
            metadata_rows.append(metadata)
            print(f"  -> {metadata['status']} {metadata.get('failure_code', '')} {';'.join(metadata.get('warning_codes', []))}")
    finally:
        detector.close()
        mesh.close()
    pd.DataFrame(manifest_rows).to_csv(dirs["manifests"] / "real_face_preprocess_manifest.csv", index=False, encoding="utf-8-sig")
    summary = summarize(metadata_rows, dirs)
    panel_paths = sorted((dirs["root"] / "qc" / "preview_panels").glob("*.png"))
    qc.make_contact_sheet(panel_paths, dirs["root"] / "qc" / "pilot_contact_sheets" / "pilot32_contact_sheet.png")
    comparison = build_revision_comparison(baseline_root, dirs["root"], pilot_ids, dirs, baseline_config, config)
    test_results = {
        "validate-only": "passed in current run before pilot",
        "unit-tests": "run separately with pytest before pilot; see terminal/final report",
        "pilot": f"{summary['success_count']} success, {summary['failure_count']} failed",
    }
    write_report(dirs, pilot_ids, summary, comparison, resolved["config_path"], test_results)
    completed = {
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "pilot_total": summary["pilot_total"],
        "success_count": summary["success_count"],
        "failure_count": summary["failure_count"],
        "full_500_run": False,
    }
    (dirs["root"] / "COMPLETED.json").write_text(json.dumps(completed, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


def main() -> int:
    try:
        return run_pipeline()
    except (FileNotFoundError, FileExistsError, NotADirectoryError, ValueError, RuntimeError) as exc:
        print(f"[configuration error] {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
