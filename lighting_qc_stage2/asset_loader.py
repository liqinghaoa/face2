from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from PIL import Image

from .affine_mapping import validate_inverse
from .config import Stage2Config, json_safe
from .id_utils import normalize_id
from .image_io import find_raw, read_gray, read_rgb, read_source_rgb_exif


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(json_safe(payload), ensure_ascii=False, indent=2), encoding="utf-8")


def output_dirs(output_dir: Path) -> dict[str, Path]:
    names = (
        "preflight",
        "pilot",
        "masks_canvas",
        "masks_raw",
        "metrics",
        "metrics_sensitivity",
        "qc_panels",
        "qc_panels_fixed",
        "contact_sheets",
        "contact_sheets_fixed",
        "comparison",
        "frozen_spec",
        "reports",
        "logs",
    )
    dirs = {name: output_dir / name for name in names}
    for path in dirs.values():
        path.mkdir(parents=True, exist_ok=True)
    return dirs


def load_split(config: Stage2Config) -> pd.DataFrame:
    split = pd.read_csv(config.split_csv, dtype=str)
    split["sample_id"] = split["ID"].map(normalize_id)
    split["fold"] = pd.to_numeric(split["fold"], errors="raise").astype(int)
    split["binary_label"] = pd.to_numeric(split["binary_label"], errors="raise").astype(int)
    if len(split) != 500 or split["sample_id"].nunique() != 500:
        raise ValueError("fixed split must contain 500 unique sample IDs")
    return split


def load_exif(config: Stage2Config) -> pd.DataFrame:
    exif = pd.read_excel(config.exif_xlsx, sheet_name="图片元数据", dtype=object)
    exif["sample_id"] = exif["ID"].map(normalize_id)
    keep = {
        "亮度值(APEX)": "brightness_value_apex",
        "ISO": "iso",
        "曝光时间(s)": "exposure_time",
    }
    out = exif[["sample_id", *keep.keys()]].rename(columns=keep).copy()
    for col in keep.values():
        out[col] = pd.to_numeric(out[col], errors="coerce")
    return out


def load_manifest(config: Stage2Config) -> pd.DataFrame:
    path = config.scheme_b_root / "manifests" / "realface_256x320_manifest.csv"
    frame = pd.read_csv(path, dtype=str)
    frame["sample_id"] = frame["ID"].map(normalize_id)
    return frame


def read_metadata(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def asset_paths(config: Stage2Config, sample_id: str) -> dict[str, Path]:
    root = config.scheme_b_root
    return {
        "images": root / "images" / f"{sample_id}.png",
        "aligned_srgb": root / "aligned_srgb" / f"{sample_id}.png",
        "face_valid_mask": root / "face_valid_mask" / f"{sample_id}.png",
        "source_valid_mask": root / "source_valid_mask" / f"{sample_id}.png",
        "parsing_label": root / "parsing_label" / f"{sample_id}.png",
        "metadata": root / "metadata" / f"{sample_id}.json",
        "raw_scene": find_raw(sample_id, config.raw_scene_dir),
    }


def preflight_assets(config: Stage2Config, class_map: dict[str, int], dirs: dict[str, Path]) -> dict[str, Any]:
    split = load_split(config)
    manifest = load_manifest(config)
    manifest_ids = set(manifest["sample_id"])
    rows = []
    failures = []
    for sample_id in split["sample_id"].tolist():
        row: dict[str, Any] = {"sample_id": sample_id}
        try:
            paths = asset_paths(config, sample_id)
            for key, path in paths.items():
                row[f"{key}_path"] = str(path)
                row[f"{key}_exists"] = bool(path.is_file())
                if key != "raw_scene" and not path.is_file():
                    raise FileNotFoundError(f"missing {key}: {path}")
            img = read_rgb(paths["images"])
            aligned = read_rgb(paths["aligned_srgb"])
            face = read_gray(paths["face_valid_mask"])
            source = read_gray(paths["source_valid_mask"])
            parsing = read_gray(paths["parsing_label"])
            if img.shape != (320, 256, 3) or aligned.shape != (320, 256, 3):
                raise ValueError("images/aligned_srgb must be 256x320 RGB")
            if face.shape != (320, 256) or source.shape != (320, 256) or parsing.shape != (320, 256):
                raise ValueError("masks/parsing must be 256x320")
            for name, mask in (("face_valid_mask", face), ("source_valid_mask", source)):
                values = set(np.unique(mask).astype(int).tolist())
                if not values.issubset({0, 255}):
                    raise ValueError(f"{name} is not binary 0/255: {values}")
            if not np.issubdtype(parsing.dtype, np.integer):
                raise ValueError("parsing label is not integer")
            if np.any((face > 0) & (source == 0)):
                raise ValueError("face_valid_mask is not source_valid_mask subset")
            if np.any(img[face == 0] != 0):
                raise ValueError("blackbg image is not black outside face_valid_mask")
            if not np.array_equal(img[face > 0], aligned[face > 0]):
                raise ValueError("images and aligned_srgb differ inside face_valid_mask")
            meta = read_metadata(paths["metadata"])
            a = np.asarray(meta["affine_source_to_canvas"], dtype=float)
            b = np.asarray(meta["affine_canvas_to_source"], dtype=float)
            row["affine_inverse_error"] = validate_inverse(a, b)
            raw_rgb, exif_applied = read_source_rgb_exif(paths["raw_scene"])
            row["raw_width"] = int(raw_rgb.shape[1])
            row["raw_height"] = int(raw_rgb.shape[0])
            row["metadata_source_width"] = int(meta["source_width"])
            row["metadata_source_height"] = int(meta["source_height"])
            row["exif_orientation_applied_read"] = bool(exif_applied)
            row["metadata_exif_orientation_applied"] = bool(meta.get("exif_orientation_applied"))
            if (raw_rgb.shape[1], raw_rgb.shape[0]) != (int(meta["source_width"]), int(meta["source_height"])):
                raise ValueError("raw source dimensions do not match metadata")
            row["status"] = "passed"
        except Exception as exc:
            row["status"] = "failed"
            row["failure"] = f"{type(exc).__name__}: {exc}"
            failures.append(row.copy())
        rows.append(row)
    inventory = pd.DataFrame(rows)
    inventory.to_csv(dirs["preflight"] / "asset_inventory.csv", index=False, encoding="utf-8-sig")
    schema = {
        "scheme_b_root": str(config.scheme_b_root),
        "manifest_shape": list(manifest.shape),
        "manifest_columns": list(manifest.columns),
        "run_manifest": json.loads((config.scheme_b_root / "run_manifest.json").read_text(encoding="utf-8")),
        "completed": json.loads((config.scheme_b_root / "COMPLETED.json").read_text(encoding="utf-8")),
        "directory_structure": sorted([p.name for p in config.scheme_b_root.iterdir()]),
    }
    write_json(dirs["preflight"] / "detected_scheme_b_schema.json", schema)
    meta0 = read_metadata(config.scheme_b_root / "metadata" / f"{split.iloc[0]['sample_id']}.json")
    write_json(dirs["preflight"] / "detected_metadata_schema.json", {"metadata_keys": sorted(meta0), "example_sample_id": split.iloc[0]["sample_id"], "example": meta0})
    summary = {
        "status": "passed" if not failures else "failed",
        "fixed_ids": int(len(split)),
        "unique_ids": int(split["sample_id"].nunique()),
        "manifest_rows": int(len(manifest)),
        "manifest_matches_split": bool(set(split["sample_id"]) == manifest_ids),
        "asset_rows_checked": int(len(inventory)),
        "failures": int(len(failures)),
        "aligned_srgb_coverage": int(inventory["aligned_srgb_exists"].sum()),
        "face_valid_mask_coverage": int(inventory["face_valid_mask_exists"].sum()),
        "source_valid_mask_coverage": int(inventory["source_valid_mask_exists"].sum()),
        "parsing_label_coverage": int(inventory["parsing_label_exists"].sum()),
        "metadata_coverage": int(inventory["metadata_exists"].sum()),
        "images_coverage": int(inventory["images_exists"].sum()),
    }
    write_json(dirs["preflight"] / "preflight_summary.json", summary)
    lines = [
        "# Stage2 Pilot32 Preflight",
        "",
        f"- status: {summary['status']}",
        f"- fixed_ids: {summary['fixed_ids']}",
        f"- manifest_matches_split: {summary['manifest_matches_split']}",
        f"- aligned_srgb: {summary['aligned_srgb_coverage']}/500",
        f"- face_valid_mask: {summary['face_valid_mask_coverage']}/500",
        f"- source_valid_mask: {summary['source_valid_mask_coverage']}/500",
        f"- parsing_label: {summary['parsing_label_coverage']}/500",
        f"- metadata_json: {summary['metadata_coverage']}/500",
        f"- images: {summary['images_coverage']}/500",
        f"- failures: {summary['failures']}",
    ]
    (dirs["preflight"] / "preflight_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    if failures:
        raise ValueError(f"preflight failed for {len(failures)} samples; see asset_inventory.csv")
    return summary
