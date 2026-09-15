"""Build high-resolution real-face skin-optics inputs and a fixed 4x5 patch grid.

The implementation deliberately reuses Scheme B's detector, FaceMesh, geometry,
linear-light resampling, and BiSeNet adapter.  Split columns other than ``ID``
are never passed to image-processing functions.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from preprocessing import build_global_face_oval_blackbg_png_simalign_strict as legacy_alignment
from preprocessing import build_realface_256x320_blackbg_from_raw_v1 as scheme_b
from src.skin_optics_real_preprocess import color_pipeline, face_geometry, face_parser, image_io, qc, skin_mask
from src.skin_optics_real_preprocess.schemas import DetectionConfig, GeometryConfig, ParserConfig, SampleFailure, SkinMaskConfig


DEFAULT_CONFIG = Path("config/preprocess/skinoptics_realface_979x1220_blackbg_v1.yaml")
TARGET_WIDTH, TARGET_HEIGHT = 979, 1220
PATCH_SIZE, PATCH_OVERLAP, PATCH_STRIDE = 256, 15, 241
PATCH_X = (0, 241, 482, 723)
PATCH_Y = (0, 241, 482, 723, 964)
PROCESSING_VERSION = "skinoptics_realface_979x1220_blackbg_v1"


def resolve(value: str | Path, root: Path = ROOT) -> Path:
    value = Path(value)
    return value.resolve() if value.is_absolute() else (root / value).resolve()


def load_config(path: Path | None) -> tuple[dict[str, Any], Path]:
    config_path = resolve(path or DEFAULT_CONFIG)
    data = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    if not isinstance(data, dict):
        raise ValueError("configuration root must be a mapping")
    return data, config_path


def fixed_patch_records() -> list[dict[str, int | str]]:
    records: list[dict[str, int | str]] = []
    for row, y0 in enumerate(PATCH_Y, start=1):
        for column, x0 in enumerate(PATCH_X, start=1):
            number = (row - 1) * len(PATCH_X) + column
            records.append({"patch_id": f"P{number:02d}", "row": row, "column": column, "x0": x0, "y0": y0, "x1": x0 + PATCH_SIZE, "y1": y0 + PATCH_SIZE})
    return records


PATCH_RECORDS = fixed_patch_records()


def validate_patch_protocol() -> None:
    if len(PATCH_RECORDS) != 20 or PATCH_X[-1] + PATCH_SIZE != TARGET_WIDTH or PATCH_Y[-1] + PATCH_SIZE != TARGET_HEIGHT:
        raise ValueError("fixed patch protocol does not tile the requested canvas")
    coverage = np.zeros((TARGET_HEIGHT, TARGET_WIDTH), dtype=np.uint8)
    for patch in PATCH_RECORDS:
        coverage[int(patch["y0"]):int(patch["y1"]), int(patch["x0"]):int(patch["x1"])] = 1
    if not coverage.all():
        raise ValueError("fixed patch protocol leaves canvas pixels uncovered")


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--full-500", action="store_true")
    parser.add_argument("--pilot-count", type=int)
    parser.add_argument("--pilot-id-file", type=Path)
    parser.add_argument("--device", type=str)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--validate-only", action="store_true")
    return parser.parse_args()


def validate_config(config: dict[str, Any], config_path: Path, args: argparse.Namespace) -> dict[str, Any]:
    paths = config.get("paths", {})
    image_dir, split_csv = resolve(paths["image_dir"]), resolve(paths["split_csv"])
    output_dir = resolve(paths["full_output_dir"] if args.full_500 else paths["pilot_output_dir"])
    expected_split = (ROOT / "data/processed/P0_Physics_Audit_v1/splits_500/nyha_2class_sex_stratified_group_5fold.csv").resolve()
    if split_csv != expected_split:
        raise ValueError(f"split_csv must be the fixed 2-class tracking index: {expected_split}")
    if not image_dir.is_dir() or not split_csv.is_file():
        raise FileNotFoundError("raw_scene directory or fixed split CSV is unavailable")
    allowed = (ROOT / "data/processed/skin_optics").resolve()
    if allowed not in output_dir.parents:
        raise ValueError("new assets must remain below data/processed/skin_optics")
    geometry = GeometryConfig(**(config.get("geometry") or {}))
    if (geometry.output_width, geometry.output_height) != (TARGET_WIDTH, TARGET_HEIGHT):
        raise ValueError("target canvas must be exactly 979x1220")
    patch = config.get("patches") or {}
    if (int(patch.get("patch_size", -1)), int(patch.get("overlap", -1)), int(patch.get("stride", -1)), int(patch.get("columns", -1)), int(patch.get("rows", -1))) != (PATCH_SIZE, PATCH_OVERLAP, PATCH_STRIDE, 4, 5):
        raise ValueError("patch configuration must remain the fixed 256/15/241 4x5 protocol")
    parser_cfg = ParserConfig(**(config.get("parser") or {}))
    if args.device:
        parser_cfg = ParserConfig(model=parser_cfg.model, checkpoint=parser_cfg.checkpoint, device=args.device)
    checkpoint = resolve(parser_cfg.checkpoint)
    if not checkpoint.is_file():
        raise FileNotFoundError(f"BiSeNet checkpoint unavailable: {checkpoint}")
    pilot_count = int(args.pilot_count or (config.get("pilot") or {}).get("count", 32))
    if pilot_count <= 0:
        raise ValueError("pilot count must be positive")
    return {"image_dir": image_dir, "split_csv": split_csv, "output_dir": output_dir, "checkpoint": checkpoint, "geometry": geometry, "detection": DetectionConfig(**(config.get("detection") or {})), "parser": parser_cfg, "skin": SkinMaskConfig(**(config.get("physics_core_skin") or {})), "pilot_count": pilot_count, "config_path": config_path}


def prepare_tree(root: Path, overwrite: bool) -> dict[str, Path]:
    if root.exists() and any(root.iterdir()):
        if not overwrite:
            raise FileExistsError(f"output exists: {root}; use --overwrite to replace this new-version output only")
        shutil.rmtree(root)
    names = ("images", "aligned_srgb", "aligned_linear_rgb", "blackbg_linear_rgb", "parsing_label", "physics_core_skin", "source_valid_mask", "patches", "metadata", "qc_preview", "logs", "config")
    dirs = {"root": root}
    for name in names:
        dirs[name] = root / name
        dirs[name].mkdir(parents=True, exist_ok=True)
    return dirs


def warp_source_valid(source_shape: tuple[int, int], affine: np.ndarray) -> np.ndarray:
    source = np.full(source_shape, 255, dtype=np.uint8)
    warped = cv2.warpAffine(source, affine[:2].astype(np.float32), (TARGET_WIDTH, TARGET_HEIGHT), flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    return (warped > 0).astype(np.uint8) * 255


def source_valid_stats(mask: np.ndarray) -> dict[str, float]:
    valid = mask > 0
    h, w = valid.shape
    bh, bw = max(1, round(h * .05)), max(1, round(w * .05))
    return {"source_valid_fraction": float(valid.mean()), "top_invalid_fraction": float((~valid[:bh]).mean()), "bottom_invalid_fraction": float((~valid[-bh:]).mean()), "left_invalid_fraction": float((~valid[:, :bw]).mean()), "right_invalid_fraction": float((~valid[:, -bw:]).mean())}


def crop_corners(geom: face_geometry.GeometryResult) -> np.ndarray:
    rolled = np.array([[geom.crop_left, geom.crop_top], [geom.crop_left + geom.crop_width, geom.crop_top], [geom.crop_left + geom.crop_width, geom.crop_top + geom.crop_height], [geom.crop_left, geom.crop_top + geom.crop_height]], dtype=np.float32)
    return face_geometry.transform_points(rolled, np.linalg.inv(geom.roll_matrix))


def black_background_linear(aligned_linear: np.ndarray, mask: np.ndarray) -> np.ndarray:
    result = np.asarray(aligned_linear, dtype=np.float32).copy()
    result[mask == 0] = 0.0
    return result


def patch_grid_preview(image: np.ndarray, fractions: list[float]) -> np.ndarray:
    panel = image.copy()
    for index, patch in enumerate(PATCH_RECORDS):
        x0, y0, x1, y1 = (int(patch[key]) for key in ("x0", "y0", "x1", "y1"))
        cv2.rectangle(panel, (x0, y0), (x1 - 1, y1 - 1), (255, 220, 20), 2)
        cv2.putText(panel, f"{patch['patch_id']} {fractions[index]:.2f}", (x0 + 5, min(y1 - 8, y0 + 28)), cv2.FONT_HERSHEY_SIMPLEX, .42, (255, 255, 0), 1, cv2.LINE_AA)
    return panel


def heatmap_preview(fractions: list[float]) -> np.ndarray:
    cells = np.asarray(fractions, dtype=np.float32).reshape(5, 4)
    heat = cv2.applyColorMap(np.rint(cells * 255).astype(np.uint8), cv2.COLORMAP_VIRIDIS)
    heat = cv2.cvtColor(cv2.resize(heat, (TARGET_WIDTH, TARGET_HEIGHT), interpolation=cv2.INTER_NEAREST), cv2.COLOR_BGR2RGB)
    for index, patch in enumerate(PATCH_RECORDS):
        x0, y0 = int(patch["x0"]), int(patch["y0"])
        cv2.putText(heat, f"{patch['patch_id']} {fractions[index]:.3f}", (x0 + 8, y0 + 35), cv2.FONT_HERSHEY_SIMPLEX, .52, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(heat, f"{patch['patch_id']} {fractions[index]:.3f}", (x0 + 8, y0 + 35), cv2.FONT_HERSHEY_SIMPLEX, .52, (0, 0, 0), 1, cv2.LINE_AA)
    return heat


def save_qc(path: Path, original: np.ndarray, detections: list[Any], selected: Any, source_landmarks: np.ndarray, geom: face_geometry.GeometryResult, aligned: np.ndarray, labels: np.ndarray, mask_parts: dict[str, Any], blackbg: np.ndarray, source_valid: np.ndarray, patch_fractions: list[float], metadata: dict[str, Any]) -> None:
    text = np.zeros_like(aligned)
    lines = [f"source: {metadata['source_width']}x{metadata['source_height']}", f"canvas: {TARGET_WIDTH}x{TARGET_HEIGHT}", f"roll: {metadata['roll_angle_deg']:.2f} deg", f"scale: {metadata['uniform_scale']:.5f}", f"valid: {metadata['source_valid_fraction']:.4f}", f"skin: {metadata['physics_core_skin_fraction']:.4f}"]
    for i, line in enumerate(lines): cv2.putText(text, line, (30, 70 + 55 * i), cv2.FONT_HERSHEY_SIMPLEX, 1.05, (255, 255, 255), 2, cv2.LINE_AA)
    panels = [
        qc.label_panel(qc.resize_panel(qc.draw_detection(original, detections, selected), (392, 488)), "1 source + selected face"),
        qc.label_panel(qc.resize_panel(qc.draw_source_geometry(original, source_landmarks, crop_corners(geom)), (392, 488)), "2 source + FaceMesh + crop"),
        qc.label_panel(qc.resize_panel(text, (392, 488)), "3 geometry audit"),
        qc.label_panel(qc.resize_panel(aligned, (392, 488)), "4 aligned_srgb"),
        qc.label_panel(qc.resize_panel(qc.draw_canvas_regions(aligned, geom.canvas_landmarks), (392, 488)), "5 aligned + oval / eye line"),
        qc.label_panel(qc.resize_panel(face_parser.colorize(labels), (392, 488)), "6 parsing_label"),
        qc.label_panel(qc.resize_panel(qc.overlay_mask(aligned, mask_parts['skin_valid_mask'], (0, 255, 180)), (392, 488)), "7 physics_core_skin overlay"),
        qc.label_panel(qc.resize_panel(blackbg, (392, 488)), "8 binary-mask black background"),
        qc.label_panel(qc.resize_panel(patch_grid_preview(aligned, patch_fractions), (392, 488)), "9 fixed 4x5 patch grid"),
        qc.label_panel(qc.resize_panel(heatmap_preview(patch_fractions), (392, 488)), "10 patch skin fraction"),
        qc.label_panel(qc.resize_panel(source_valid, (392, 488)), "11 source_valid_mask"),
        qc.label_panel(qc.resize_panel(qc.overlay_mask(aligned, 255 - source_valid, (255, 0, 0)), (392, 488)), "12 source boundary overlay"),
    ]
    image_io.save_rgb_png(path, np.concatenate([np.concatenate(panels[i:i + 4], axis=1) for i in range(0, 12, 4)], axis=0))


def write_patches(case_id: str, blackbg: np.ndarray, mask: np.ndarray, output_dir: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    case_dir = output_dir / case_id
    case_dir.mkdir(parents=True, exist_ok=True)
    for patch in PATCH_RECORDS:
        patch_id = str(patch["patch_id"]); x0, y0, x1, y1 = (int(patch[key]) for key in ("x0", "y0", "x1", "y1"))
        rgb, patch_mask = blackbg[y0:y1, x0:x1].copy(), mask[y0:y1, x0:x1].copy()
        if rgb.shape != (PATCH_SIZE, PATCH_SIZE, 3) or patch_mask.shape != (PATCH_SIZE, PATCH_SIZE):
            raise SampleFailure("invalid_output_shape", f"{patch_id} extraction is not 256x256")
        rgb_path, mask_path, metadata_path = case_dir / f"{patch_id}_rgb_linear.npy", case_dir / f"{patch_id}_mask.png", case_dir / f"{patch_id}_metadata.json"
        np.save(rgb_path, rgb.astype(np.float32))
        image_io.save_gray_png(mask_path, patch_mask.astype(np.uint8))
        valid_pixels = int((patch_mask > 0).sum())
        row = {"case_id": case_id, **patch, "patch_width": PATCH_SIZE, "patch_height": PATCH_SIZE, "valid_skin_pixels": valid_pixels, "valid_skin_fraction": valid_pixels / float(PATCH_SIZE * PATCH_SIZE), "rgb_path": str(rgb_path), "mask_path": str(mask_path), "case_status": "success"}
        metadata_path.write_text(json.dumps(row, ensure_ascii=False, indent=2), encoding="utf-8")
        rows.append(row)
    return rows


def process_one(case_id: str, dirs: dict[str, Path], image_dir: Path, detector: Any, mesh: Any, parser_model: Any, parser_device: Any, geometry_cfg: GeometryConfig, skin_cfg: SkinMaskConfig, config_hash: str) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    metadata: dict[str, Any] = {"case_id": case_id, "status": "failed", "failure_reason": "", "processing_version": PROCESSING_VERSION, "config_hash": config_hash}
    patches: list[dict[str, Any]] = []
    try:
        source_path = image_io.find_image_for_id(case_id, image_dir)
        if source_path is None: raise SampleFailure("missing_source_image", "cannot find raw_scene image by ID")
        rgb, exif_applied = image_io.read_image_rgb_exif(source_path); h, w = rgb.shape[:2]
        detections = legacy_alignment.detect_faces(rgb, detector); selected = legacy_alignment.select_face(detections, rgb.shape)
        if selected is None: raise SampleFailure("face_detection_failed", "MediaPipe Face Detection returned no valid face")
        x, y, bw, bh = selected.bbox
        crop_x, crop_y, crop_w, crop_h = legacy_alignment.expand_bbox(selected.bbox, rgb.shape)
        landmarks = legacy_alignment.run_facemesh(rgb[crop_y:crop_y + crop_h, crop_x:crop_x + crop_w].copy(), mesh)
        if landmarks is None: raise SampleFailure("facemesh_failed", "MediaPipe FaceMesh returned no landmarks")
        source_landmarks = face_geometry.landmarks_to_source_array(landmarks, crop_x, crop_y, crop_w, crop_h)
        keys, geom = face_geometry.extract_key_points(source_landmarks), face_geometry.build_geometry(source_landmarks, geometry_cfg)
        aligned_linear = color_pipeline.warp_linear_rgb(color_pipeline.uint8_rgb_to_linear(rgb), geom.affine_source_to_canvas, TARGET_WIDTH, TARGET_HEIGHT)
        aligned_srgb = color_pipeline.linear_to_uint8_rgb(aligned_linear)
        source_valid = warp_source_valid((h, w), geom.affine_source_to_canvas)
        labels = face_parser.run(aligned_srgb, parser_model, parser_device)
        if labels.shape != (TARGET_HEIGHT, TARGET_WIDTH): raise SampleFailure("parser_failed", f"parser label shape {labels.shape}")
        face_width = float(geom.canvas_landmarks[:, 0].max() - geom.canvas_landmarks[:, 0].min())
        mask_parts = skin_mask.build_skin_valid_mask(labels, geom.canvas_landmarks, source_valid, face_width, skin_cfg)
        physics_core = np.asarray(mask_parts["skin_valid_mask"], dtype=np.uint8)
        skin_mask.assert_mask_contract(physics_core, source_valid, (TARGET_HEIGHT, TARGET_WIDTH))
        blackbg_linear = black_background_linear(aligned_linear, physics_core)
        if np.any(blackbg_linear[physics_core == 0] != 0): raise SampleFailure("invalid_output_shape", "black background has non-zero values outside mask")
        blackbg_srgb = color_pipeline.linear_to_uint8_rgb(blackbg_linear)
        patches = write_patches(case_id, blackbg_linear, physics_core, dirs["patches"])
        metadata.update({"source_path": str(source_path), "source_width": w, "source_height": h, "exif_orientation_applied": exif_applied, "face_detection_bbox": [x, y, bw, bh], "face_detection_score": float(selected.confidence), "expanded_face_bbox": [crop_x, crop_y, crop_w, crop_h], "selected_face_index": int(selected.index), "num_faces_detected": len(detections), "landmarks_original": source_landmarks.tolist(), "left_eye_center": keys["left_eye"].tolist(), "right_eye_center": keys["right_eye"].tolist(), "nose": keys["nose_tip"].tolist(), "mouth_corners": [keys["left_mouth"].tolist(), keys["right_mouth"].tolist()], "chin": keys["chin"].tolist(), "face_oval": keys["face_oval"].tolist(), "roll_angle_deg": geom.roll_angle_degrees, "residual_roll_deg": geom.residual_roll_degrees, "crop_geometry": {"left": geom.crop_left, "top": geom.crop_top, "width": geom.crop_width, "height": geom.crop_height, "aspect_ratio": geom.crop_width / geom.crop_height, "face_quantile_low": geometry_cfg.face_quantile_low, "face_quantile_high": geometry_cfg.face_quantile_high, "top_margin_ratio": geometry_cfg.top_margin_ratio, "bottom_margin_ratio": geometry_cfg.bottom_margin_ratio}, "target_width": TARGET_WIDTH, "target_height": TARGET_HEIGHT, "target_aspect_ratio": TARGET_WIDTH / TARGET_HEIGHT, "affine_source_to_canvas": geom.affine_source_to_canvas.tolist(), "affine_canvas_to_source": geom.affine_canvas_to_source.tolist(), "uniform_scale": geom.uniform_scale, "affine_singular_values": list(geom.matrix_singular_values), "roundtrip_max_error": geom.roundtrip_max_error, "physics_core_skin_pixels": int((physics_core > 0).sum()), "physics_core_skin_fraction": float((physics_core > 0).mean()), "patch_count": len(patches), "patch_size": PATCH_SIZE, "overlap": PATCH_OVERLAP, "stride": PATCH_STRIDE})
        metadata.update(source_valid_stats(source_valid))
        warnings = []
        if metadata["source_valid_fraction"] < .98: warnings.append("source_boundary_visible")
        if abs(geom.roll_angle_degrees) > 12: warnings.append("large_roll")
        metadata["warning_codes"] = warnings
        np.save(dirs["aligned_linear_rgb"] / f"{case_id}.npy", aligned_linear.astype(np.float32)); np.save(dirs["blackbg_linear_rgb"] / f"{case_id}.npy", blackbg_linear.astype(np.float32))
        image_io.save_rgb_png(dirs["aligned_srgb"] / f"{case_id}.png", aligned_srgb); image_io.save_rgb_png(dirs["images"] / f"{case_id}.png", blackbg_srgb); image_io.save_gray_png(dirs["parsing_label"] / f"{case_id}.png", labels); image_io.save_gray_png(dirs["physics_core_skin"] / f"{case_id}.png", physics_core); image_io.save_gray_png(dirs["source_valid_mask"] / f"{case_id}.png", source_valid)
        save_qc(dirs["qc_preview"] / f"{case_id}.png", rgb, detections, selected, source_landmarks, geom, aligned_srgb, labels, mask_parts, blackbg_srgb, source_valid, [float(item["valid_skin_fraction"]) for item in patches], metadata)
        metadata["status"] = "success"
    except SampleFailure as exc:
        metadata["failure_reason"] = f"{exc.code}: {exc.detail}"
    except Exception as exc:  # Preserve a per-case reason; never silently fall back.
        metadata["failure_reason"] = f"unexpected_error: {type(exc).__name__}: {exc}"
    (dirs["metadata"] / f"{case_id}.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    return metadata, patches


def describe(values: list[float], include_mean: bool = False) -> dict[str, float | None]:
    a = np.asarray([x for x in values if np.isfinite(x)], dtype=float)
    if not len(a): return {key: None for key in (("min", "median", "mean", "p95", "max") if include_mean else ("min", "median", "p95", "max"))}
    result = {"min": float(a.min()), "median": float(np.median(a)), "p95": float(np.quantile(a, .95)), "max": float(a.max())}
    if include_mean: result["mean"] = float(a.mean())
    return result


def write_outputs(dirs: dict[str, Path], selected_ids: list[str], metadata: list[dict[str, Any]], patch_rows: list[dict[str, Any]], config_path: Path, full: bool) -> dict[str, Any]:
    success, failed = [m for m in metadata if m["status"] == "success"], [m for m in metadata if m["status"] != "success"]
    manifest = [{"case_id": m["case_id"], "status": m["status"], "failure_reason": m.get("failure_reason", ""), "warning_codes": ";".join(m.get("warning_codes", [])), "metadata_path": str(dirs["metadata"] / f"{m['case_id']}.json")} for m in metadata]
    pd.DataFrame(manifest).to_csv(dirs["root"] / "manifest.csv", index=False, encoding="utf-8-sig"); pd.DataFrame(manifest).to_csv(dirs["logs"] / "preprocess_log.csv", index=False, encoding="utf-8-sig"); pd.DataFrame(failed).to_csv(dirs["logs"] / "failed_cases.csv", index=False, encoding="utf-8-sig"); pd.DataFrame(patch_rows).to_csv(dirs["logs"] / "patch_manifest.csv", index=False, encoding="utf-8-sig")
    patch_by_id: dict[str, list[float]] = defaultdict(list)
    for row in patch_rows: patch_by_id[str(row["patch_id"])].append(float(row["valid_skin_fraction"]))
    summary = {"selected_cases": len(selected_ids), "success_cases": len(success), "failed_cases": len(failed), "full_500_run": full, "source_width": describe([float(m["source_width"]) for m in success]), "source_height": describe([float(m["source_height"]) for m in success]), "roll_angle_deg": describe([float(m["roll_angle_deg"]) for m in success]), "uniform_scale": describe([float(m["uniform_scale"]) for m in success]), "source_valid_fraction": describe([float(m["source_valid_fraction"]) for m in success]), "physics_core_skin_fraction": describe([float(m["physics_core_skin_fraction"]) for m in success], True), "patch_valid_skin_fraction_overall": describe([float(r["valid_skin_fraction"]) for r in patch_rows], True), "patch_valid_skin_fraction_by_id": {patch_id: describe(values, True) for patch_id, values in sorted(patch_by_id.items())}, "incomplete_success_cases": [m["case_id"] for m in success if sum(r["case_id"] == m["case_id"] for r in patch_rows) != 20], "config_path": str(config_path)}
    summary_rows = [{"metric": key, "value": json.dumps(value, ensure_ascii=False) if isinstance(value, (dict, list)) else value} for key, value in summary.items()]
    pd.DataFrame(summary_rows).to_csv(dirs["logs"] / "preprocess_summary.csv", index=False, encoding="utf-8-sig"); (dirs["logs"] / "preprocess_summary.txt").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def audit_output(root: Path) -> dict[str, Any]:
    manifest = pd.read_csv(root / "manifest.csv", dtype={"case_id": str}).fillna("")
    problems: list[str] = []
    for row in manifest.to_dict("records"):
        if row["status"] != "success": continue
        case_id = str(row["case_id"])
        linear, masked, mask = np.load(root / "aligned_linear_rgb" / f"{case_id}.npy"), np.load(root / "blackbg_linear_rgb" / f"{case_id}.npy"), cv2.imread(str(root / "physics_core_skin" / f"{case_id}.png"), cv2.IMREAD_GRAYSCALE)
        aligned_png = cv2.imread(str(root / "aligned_srgb" / f"{case_id}.png"), cv2.IMREAD_COLOR)
        blackbg_png = cv2.imread(str(root / "images" / f"{case_id}.png"), cv2.IMREAD_COLOR)
        labels = cv2.imread(str(root / "parsing_label" / f"{case_id}.png"), cv2.IMREAD_GRAYSCALE)
        source_valid = cv2.imread(str(root / "source_valid_mask" / f"{case_id}.png"), cv2.IMREAD_GRAYSCALE)
        if linear.shape != (TARGET_HEIGHT, TARGET_WIDTH, 3) or masked.shape != linear.shape: problems.append(f"{case_id}: canvas shape")
        for name, image in (("aligned_srgb", aligned_png), ("blackbg_srgb", blackbg_png), ("parsing_label", labels), ("source_valid_mask", source_valid)):
            if image is None or image.shape[:2] != (TARGET_HEIGHT, TARGET_WIDTH): problems.append(f"{case_id}: {name} shape")
        if mask is None or set(np.unique(mask).tolist()) - {0, 255}: problems.append(f"{case_id}: nonbinary mask")
        elif np.any(masked[mask == 0] != 0): problems.append(f"{case_id}: nonzero black background")
        for patch in PATCH_RECORDS:
            patch_id = str(patch["patch_id"]); x0, y0, x1, y1 = (int(patch[key]) for key in ("x0", "y0", "x1", "y1")); p_rgb, p_mask = np.load(root / "patches" / case_id / f"{patch_id}_rgb_linear.npy"), cv2.imread(str(root / "patches" / case_id / f"{patch_id}_mask.png"), cv2.IMREAD_GRAYSCALE)
            if p_rgb.shape != (256, 256, 3) or p_mask is None or p_mask.shape != (256, 256): problems.append(f"{case_id}:{patch_id} shape")
            elif not np.array_equal(p_rgb, masked[y0:y1, x0:x1]) or not np.array_equal(p_mask, mask[y0:y1, x0:x1]): problems.append(f"{case_id}:{patch_id} differs from full canvas")
    return {"passed": not problems, "problems": problems, "success_cases": int((manifest.status == "success").sum())}


def main() -> int:
    args = arguments(); validate_patch_protocol(); config, config_path = load_config(args.config); resolved = validate_config(config, config_path, args)
    if args.validate_only: print("configuration and fixed patch protocol are valid"); return 0
    split = pd.read_csv(resolved["split_csv"], dtype={"ID": str})
    if "ID" not in split or split.ID.duplicated().any(): raise ValueError("fixed split requires unique ID values")
    if args.full_500: selected_ids = split.ID.astype(str).tolist()
    else: selected_ids = scheme_b.select_pilot_ids(split[["ID"]], resolved["pilot_count"], args.pilot_id_file)
    dirs = prepare_tree(resolved["output_dir"], args.overwrite); (dirs["root"] / ("full_ids.txt" if args.full_500 else "pilot_ids.txt")).write_text("\n".join(selected_ids) + "\n", encoding="utf-8"); shutil.copy2(config_path, dirs["config"] / "resolved_config.yaml")
    config_hash = hashlib.sha256(config_path.read_bytes()).hexdigest(); parser_device = face_parser.resolve_device(resolved["parser"].device); parser_model = face_parser.load_model(resolved["parser"].model, resolved["checkpoint"], parser_device)
    detector = mp.solutions.face_detection.FaceDetection(model_selection=resolved["detection"].model_selection, min_detection_confidence=resolved["detection"].min_detection_confidence); mesh = mp.solutions.face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1, refine_landmarks=True, min_detection_confidence=resolved["detection"].min_detection_confidence, min_tracking_confidence=.5)
    metadata: list[dict[str, Any]] = []; patch_rows: list[dict[str, Any]] = []
    try:
        for index, case_id in enumerate(selected_ids, start=1):
            print(f"[{index:03d}/{len(selected_ids):03d}] {case_id}", flush=True)
            item, patches = process_one(case_id, dirs, resolved["image_dir"], detector, mesh, parser_model, parser_device, resolved["geometry"], resolved["skin"], config_hash); metadata.append(item); patch_rows.extend(patches); print(f"  -> {item['status']} {item.get('failure_reason', '')}", flush=True)
    finally:
        detector.close(); mesh.close()
    summary = write_outputs(dirs, selected_ids, metadata, patch_rows, config_path, args.full_500); audit = audit_output(dirs["root"]); (dirs["logs"] / "automatic_acceptance.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8"); (dirs["root"] / "COMPLETED.json").write_text(json.dumps({"completed_at_utc": datetime.now(timezone.utc).isoformat(), "summary": summary, "automatic_acceptance": audit}, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"summary": summary, "automatic_acceptance": audit}, ensure_ascii=False, indent=2)); return 0 if audit["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
