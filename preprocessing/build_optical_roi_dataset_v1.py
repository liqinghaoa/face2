"""Build Optical ROI Dataset V1 from saved aligned labels, masks, and ROI bboxes.

This program deliberately does not run detection, alignment, face parsing, image
resizing, colour analysis, or model training. Historical inputs are read-only.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd
import yaml
from PIL import Image, ImageDraw, ImageFont


DEFAULT_CONFIG = "optical_roi_dataset_v1.yaml"
ROI_NAMES = ("forehead", "cheek_image_left", "cheek_image_right")
ROI_COLORS = {
    "forehead": (0, 200, 0),
    "cheek_image_left": (0, 102, 255),
    "cheek_image_right": (255, 0, 0),
}
ROI_BBOX_COLUMNS = {
    "forehead": ("bbox_x1", "bbox_y1", "bbox_x2", "bbox_y2"),
    "cheek_image_left": (
        "left_cheek_bbox_x1",
        "left_cheek_bbox_y1",
        "left_cheek_bbox_x2",
        "left_cheek_bbox_y2",
    ),
    "cheek_image_right": (
        "right_cheek_bbox_x1",
        "right_cheek_bbox_y1",
        "right_cheek_bbox_x2",
        "right_cheek_bbox_y2",
    ),
}
ROI_MANIFEST_PREFIX = {
    "forehead": "forehead",
    "cheek_image_left": "cheek_image_left",
    "cheek_image_right": "cheek_image_right",
}
EXIF_PARAMETERS = ("ExposureTime", "FNumber", "ISOSpeedRatings")


class BuildFailure(RuntimeError):
    def __init__(self, stage: str, errors: Sequence[str]):
        self.stage = stage
        self.errors = [str(error) for error in errors]
        super().__init__(f"{stage}: {len(self.errors)} error(s)")


@dataclass
class PreflightResult:
    ids: list[str]
    exif: pd.DataFrame
    boxes: dict[str, dict[str, tuple[int, int, int, int]]]
    skin_label: int
    historical_inputs: list[Path]
    historical_inventory_sha256: str
    bbox_log_sha256: str
    exif_source_sha256: dict[str, str]
    invalid_bbox_count: int
    missing_exif_count: int


def default_project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def project_relative(path: Path, project_root: Path) -> str:
    return path.resolve().relative_to(project_root.resolve()).as_posix()


def resolve_under_project(value: str | Path, project_root: Path) -> Path:
    path = Path(value).expanduser()
    resolved = path.resolve() if path.is_absolute() else (project_root / path).resolve()
    try:
        resolved.relative_to(project_root.resolve())
    except ValueError as exc:
        raise ValueError(f"Path is outside project root: {resolved}") from exc
    return resolved


def load_config(config_path: Path | None, project_root: Path) -> tuple[dict[str, Any], Path]:
    path = config_path
    if path is None:
        path = project_root / "config" / "preprocess" / DEFAULT_CONFIG
    elif not path.is_absolute():
        path = project_root / path
    path = path.resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Config file does not exist: {path}")
    loaded = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(loaded, dict):
        raise ValueError("Config root must be a mapping")
    return {str(key).replace("-", "_"): value for key, value in loaded.items()}, path


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build Optical ROI Dataset V1")
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--project-root", type=Path, default=None)
    parser.add_argument("--overwrite", action="store_true", default=None)
    parser.add_argument("--preflight-only", action="store_true")
    cli = parser.parse_args(argv)
    inferred_root = (
        cli.project_root.expanduser().resolve()
        if cli.project_root is not None
        else default_project_root()
    )
    config, config_path = load_config(cli.config, inferred_root)
    project_root_value = config.get("project_root", ".")
    project_root = (
        inferred_root
        if cli.project_root is not None or project_root_value in (None, "", ".")
        else Path(project_root_value).expanduser().resolve()
    )
    defaults: dict[str, Any] = {
        "dataset_name": "Optical ROI Dataset V1",
        "version": "1.0",
        "image_size": 224,
        "low_coverage_n": 10,
        "preview_columns": 2,
        "png_compress_level": 3,
        "overwrite": False,
        "manual_forehead_review_ids": ["9300518956", "206664979", "A001471135"],
    }
    required_paths = (
        "study_id_dir",
        "aligned_rgb_dir",
        "parsing_label_dir",
        "final_mask_dir",
        "bbox_log",
        "exif_identity_csv",
        "exif_parameter_long_csv",
        "skin_label_source_code",
        "dataset_output_dir",
        "report_output_dir",
    )
    values = {**defaults, **config}
    values["project_root"] = project_root.resolve()
    values["config_path"] = config_path
    for key in required_paths:
        if key not in values:
            raise ValueError(f"Missing config key: {key}")
        values[key] = resolve_under_project(values[key], project_root)
    if cli.overwrite is not None:
        values["overwrite"] = cli.overwrite
    values["preflight_only"] = bool(cli.preflight_only)
    args = argparse.Namespace(**values)
    if int(args.image_size) != 224:
        raise ValueError("Optical ROI Dataset V1 requires image_size=224")
    if not 0 <= int(args.png_compress_level) <= 9:
        raise ValueError("png_compress_level must be in [0, 9]")
    return args


def validate_unique_ids(ids: Sequence[str], expected_count: int | None = None) -> None:
    normalized = [str(value).strip() for value in ids]
    if any(not value for value in normalized):
        raise ValueError("ID list contains an empty ID")
    if len(set(normalized)) != len(normalized):
        raise ValueError("ID list contains exact duplicates")
    folded = [value.casefold() for value in normalized]
    if len(set(folded)) != len(folded):
        raise ValueError("ID list contains case-insensitive duplicates")
    if expected_count is not None and len(normalized) != expected_count:
        raise ValueError(f"ID count is {len(normalized)}, expected {expected_count}")


def bbox_area(bbox: tuple[int, int, int, int]) -> int:
    x1, y1, x2, y2 = bbox
    if not (0 <= x1 <= x2 and 0 <= y1 <= y2):
        raise ValueError(f"Invalid inclusive bbox: {bbox}")
    return (x2 - x1 + 1) * (y2 - y1 + 1)


def inclusive_bbox_region(
    shape: tuple[int, int], bbox: tuple[int, int, int, int]
) -> np.ndarray:
    height, width = shape
    x1, y1, x2, y2 = bbox
    if not (0 <= x1 <= x2 < width and 0 <= y1 <= y2 < height):
        raise ValueError(f"BBox {bbox} is outside image shape {shape}")
    region = np.zeros(shape, dtype=bool)
    region[y1 : y2 + 1, x1 : x2 + 1] = True
    return region


def build_effective_mask(
    parsing_label: np.ndarray,
    final_mask: np.ndarray,
    bbox: tuple[int, int, int, int],
    skin_label: int,
) -> np.ndarray:
    parsing = np.asarray(parsing_label)
    final = np.asarray(final_mask)
    if parsing.shape != (224, 224) or parsing.ndim != 2:
        raise ValueError(f"Invalid parsing label shape: {parsing.shape}")
    if final.shape != (224, 224) or final.ndim != 2:
        raise ValueError(f"Invalid final mask shape: {final.shape}")
    region = inclusive_bbox_region((224, 224), bbox)
    effective = (parsing == int(skin_label)) & (final > 0) & region
    return effective.astype(np.uint8) * 255


def valid_skin_fraction(pixel_count: int, area: int) -> float:
    if area <= 0:
        raise ValueError("bbox area must be positive")
    if pixel_count < 0 or pixel_count > area:
        raise ValueError("pixel count must be within [0, bbox area]")
    return float(pixel_count) / float(area)


def validate_positive_exif(exposure_time: Any, f_number: Any, iso: Any) -> tuple[float, float, float]:
    values = tuple(float(value) for value in (exposure_time, f_number, iso))
    if not all(math.isfinite(value) and value > 0 for value in values):
        raise ValueError(f"EXIF values must be finite and positive: {values}")
    return values


def derive_exif(exposure_time: Any, f_number: Any, iso: Any) -> tuple[float, float]:
    exposure, aperture, sensitivity = validate_positive_exif(exposure_time, f_number, iso)
    return (
        math.log2(exposure / (aperture**2)),
        math.log2(sensitivity / 100.0),
    )


def empty_mask_reason(mask: np.ndarray, image_id: str, roi_name: str) -> str | None:
    if int((np.asarray(mask) > 0).sum()) == 0:
        return f"empty_mask:{image_id}:{roi_name}"
    return None


def read_skin_label(source_path: Path) -> int:
    text = source_path.read_text(encoding="utf-8")
    match = re.search(r"^CLASS_SKIN\s*=\s*(\d+)\s*$", text, flags=re.MULTILINE)
    if not match:
        raise ValueError(f"CLASS_SKIN definition not found in {source_path}")
    return int(match.group(1))


def parse_bbox_values(values: Iterable[Any], image_id: str, roi_name: str) -> tuple[int, int, int, int]:
    numeric = np.asarray(list(values), dtype=np.float64)
    if numeric.shape != (4,) or not np.isfinite(numeric).all():
        raise ValueError(f"Non-finite bbox for {image_id}/{roi_name}: {numeric.tolist()}")
    rounded = np.rint(numeric)
    if not np.equal(numeric, rounded).all():
        raise ValueError(f"Non-integer bbox for {image_id}/{roi_name}: {numeric.tolist()}")
    bbox = tuple(int(value) for value in rounded)
    inclusive_bbox_region((224, 224), bbox)
    return bbox  # type: ignore[return-value]


def load_exif(args: argparse.Namespace, study_ids: set[str]) -> tuple[pd.DataFrame, list[str]]:
    errors: list[str] = []
    identity = pd.read_csv(
        args.exif_identity_csv,
        usecols=["ID", "Make", "Model"],
        dtype={"ID": str, "Make": str, "Model": str},
    )
    identity["ID"] = identity["ID"].astype(str).str.strip()
    identity = identity[identity["ID"].isin(study_ids)].copy()
    if identity.duplicated("ID").any():
        errors.append("duplicate_EXIF_identity_IDs:" + ",".join(identity.loc[identity.duplicated("ID", False), "ID"]))

    values = pd.read_csv(
        args.exif_parameter_long_csv,
        usecols=["ID", "parameter", "numeric_value"],
        dtype={"ID": str, "parameter": str},
    )
    values["ID"] = values["ID"].astype(str).str.strip()
    values = values[
        values["ID"].isin(study_ids) & values["parameter"].isin(EXIF_PARAMETERS)
    ].copy()
    if values.duplicated(["ID", "parameter"]).any():
        duplicate = values.loc[values.duplicated(["ID", "parameter"], False), ["ID", "parameter"]]
        errors.append("duplicate_EXIF_parameter_rows:" + duplicate.astype(str).agg("/".join, axis=1).str.cat(sep=","))
    wide = values.pivot(index="ID", columns="parameter", values="numeric_value").reset_index()
    exif = identity.merge(wide, on="ID", how="left", validate="one_to_one")
    missing_ids = sorted(study_ids - set(exif["ID"]))
    if missing_ids:
        errors.append("missing_EXIF_IDs:" + ",".join(missing_ids))
    for column in EXIF_PARAMETERS:
        exif[column] = pd.to_numeric(exif.get(column), errors="coerce")
        bad = ~np.isfinite(exif[column]) | (exif[column] <= 0)
        if bad.any():
            errors.append(f"invalid_{column}:" + ",".join(exif.loc[bad, "ID"].astype(str)))
    for column in ("Make", "Model"):
        bad = exif[column].isna() | exif[column].astype(str).str.strip().eq("")
        if bad.any():
            errors.append(f"blank_{column}:" + ",".join(exif.loc[bad, "ID"].astype(str)))
        exif[column] = exif[column].astype(str).str.strip()
    exif["camera_id"] = exif["Make"] + "/" + exif["Model"]
    return exif.sort_values("ID", kind="stable").reset_index(drop=True), errors


def load_boxes(args: argparse.Namespace, ids: list[str]) -> tuple[dict[str, dict[str, tuple[int, int, int, int]]], list[str], int]:
    errors: list[str] = []
    invalid_count = 0
    columns = [
        "ID",
        "roi_type",
        "roi_success",
        "bbox_x1",
        "bbox_y1",
        "bbox_x2",
        "bbox_y2",
        "left_cheek_bbox_x1",
        "left_cheek_bbox_y1",
        "left_cheek_bbox_x2",
        "left_cheek_bbox_y2",
        "right_cheek_bbox_x1",
        "right_cheek_bbox_y1",
        "right_cheek_bbox_x2",
        "right_cheek_bbox_y2",
    ]
    table = pd.read_csv(args.bbox_log, usecols=columns, dtype={"ID": str, "roi_type": str})
    table["ID"] = table["ID"].astype(str).str.strip()
    table = table[table["ID"].isin(ids) & table["roi_type"].isin(["forehead_roi", "cheek_roi"])].copy()
    indexed: dict[str, pd.DataFrame] = {}
    for roi_type in ("forehead_roi", "cheek_roi"):
        subset = table[table["roi_type"].eq(roi_type)].copy()
        if len(subset) != 500 or subset["ID"].nunique() != 500 or set(subset["ID"]) != set(ids):
            errors.append(f"{roi_type}_ID_match_failed:rows={len(subset)},unique={subset['ID'].nunique()}")
        if subset.duplicated("ID").any():
            errors.append(f"duplicate_{roi_type}_IDs")
        success = subset["roi_success"].astype(str).str.strip().str.lower().eq("true")
        if not success.all():
            errors.append(f"unsuccessful_{roi_type}:" + ",".join(subset.loc[~success, "ID"]))
        indexed[roi_type] = subset.set_index("ID")
    boxes: dict[str, dict[str, tuple[int, int, int, int]]] = {}
    if errors:
        return boxes, errors, invalid_count
    for image_id in ids:
        boxes[image_id] = {}
        forehead = indexed["forehead_roi"].loc[image_id]
        cheek = indexed["cheek_roi"].loc[image_id]
        for roi_name in ROI_NAMES:
            row = forehead if roi_name == "forehead" else cheek
            try:
                boxes[image_id][roi_name] = parse_bbox_values(
                    [row[column] for column in ROI_BBOX_COLUMNS[roi_name]], image_id, roi_name
                )
            except ValueError as exc:
                invalid_count += 1
                errors.append(str(exc))
    return boxes, errors, invalid_count


def historical_inventory_sha256(paths: Sequence[Path], project_root: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted((item.resolve() for item in paths), key=lambda item: str(item).casefold()):
        digest.update(project_relative(path, project_root).encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(path.stat().st_size).encode("ascii"))
        digest.update(b"\0")
        digest.update(str(path.stat().st_mtime_ns).encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest()


def run_preflight(args: argparse.Namespace) -> PreflightResult:
    errors: list[str] = []
    study_files = sorted(args.study_id_dir.glob("*.png"), key=lambda path: path.stem)
    ids = [path.stem for path in study_files]
    try:
        validate_unique_ids(ids, expected_count=500)
    except ValueError as exc:
        errors.append(str(exc))
    if len(study_files) != 500:
        errors.append(f"study_id_PNG_count={len(study_files)},expected=500")

    skin_label = read_skin_label(args.skin_label_source_code)
    if skin_label != 1:
        errors.append(f"unexpected_skin_label={skin_label},expected_code_audit_value=1")
    exif, exif_errors = load_exif(args, set(ids))
    errors.extend(exif_errors)
    boxes, bbox_errors, invalid_bbox_count = load_boxes(args, ids)
    errors.extend(bbox_errors)

    historical_inputs: list[Path] = []
    empty_masks: list[str] = []
    for image_id in ids:
        paths = {
            "aligned": args.aligned_rgb_dir / f"{image_id}.png",
            "parsing": args.parsing_label_dir / f"{image_id}.png",
            "final": args.final_mask_dir / f"{image_id}.png",
        }
        missing = [str(path) for path in paths.values() if not path.is_file()]
        if missing:
            errors.extend(f"missing_input:{path}" for path in missing)
            continue
        historical_inputs.extend(paths.values())
        try:
            aligned_image = Image.open(paths["aligned"])
            aligned = np.asarray(aligned_image)
            if (
                aligned_image.size != (224, 224)
                or aligned_image.mode != "RGB"
                or aligned.shape != (224, 224, 3)
                or aligned.dtype != np.uint8
            ):
                errors.append(
                    f"invalid_aligned:{image_id}:size={aligned_image.size},mode={aligned_image.mode},"
                    f"shape={aligned.shape},dtype={aligned.dtype}"
                )
            parsing_image = Image.open(paths["parsing"])
            parsing = np.asarray(parsing_image)
            if parsing.shape != (224, 224) or parsing.ndim != 2 or parsing.dtype.kind not in "ui":
                errors.append(
                    f"invalid_parsing:{image_id}:mode={parsing_image.mode},shape={parsing.shape},dtype={parsing.dtype}"
                )
            final_image = Image.open(paths["final"])
            final = np.asarray(final_image)
            if final.shape != (224, 224) or final.ndim != 2:
                errors.append(
                    f"invalid_final_mask:{image_id}:mode={final_image.mode},shape={final.shape},dtype={final.dtype}"
                )
            if image_id in boxes and parsing.shape == (224, 224) and final.shape == (224, 224):
                for roi_name in ROI_NAMES:
                    mask = build_effective_mask(parsing, final, boxes[image_id][roi_name], skin_label)
                    reason = empty_mask_reason(mask, image_id, roi_name)
                    if reason:
                        empty_masks.append(reason)
        except Exception as exc:
            errors.append(f"input_decode_error:{image_id}:{type(exc).__name__}:{exc}")
    errors.extend(empty_masks)
    if errors:
        raise BuildFailure("preflight", errors)
    inventory = historical_inventory_sha256(historical_inputs, args.project_root)
    return PreflightResult(
        ids=ids,
        exif=exif,
        boxes=boxes,
        skin_label=skin_label,
        historical_inputs=historical_inputs,
        historical_inventory_sha256=inventory,
        bbox_log_sha256=sha256_file(args.bbox_log),
        exif_source_sha256={
            project_relative(args.exif_identity_csv, args.project_root): sha256_file(args.exif_identity_csv),
            project_relative(args.exif_parameter_long_csv, args.project_root): sha256_file(args.exif_parameter_long_csv),
        },
        invalid_bbox_count=invalid_bbox_count,
        missing_exif_count=0,
    )


def verify_existing_generated_output(path: Path, is_dataset: bool) -> bool:
    if not path.exists() or not any(path.iterdir()):
        return True
    if is_dataset:
        manifest_path = path / "build_manifest.json"
        if not manifest_path.is_file():
            return False
        try:
            payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        except Exception:
            return False
        return payload.get("dataset_name") == "Optical ROI Dataset V1"
    return (path / "optical_roi_dataset_v1_build_report.md").is_file()


def prepare_output_dirs(args: argparse.Namespace) -> dict[str, Path]:
    dataset_root = args.dataset_output_dir.resolve()
    report_root = args.report_output_dir.resolve()
    expected = {
        (args.project_root / "data" / "processed" / "optical_roi_dataset_v1").resolve(),
        (args.project_root / "reports" / "optical_roi_dataset_v1").resolve(),
    }
    if {dataset_root, report_root} != expected:
        raise BuildFailure("output_safety", [f"Unexpected output paths: {dataset_root}, {report_root}"])
    nonempty = [path for path in (dataset_root, report_root) if path.exists() and any(path.iterdir())]
    if nonempty and not bool(args.overwrite):
        raise BuildFailure("output_safety", [f"Output directory is non-empty: {path}" for path in nonempty])
    if nonempty:
        if not verify_existing_generated_output(dataset_root, True):
            raise BuildFailure("output_safety", [f"Dataset output is not verified as program-generated: {dataset_root}"])
        if not verify_existing_generated_output(report_root, False):
            raise BuildFailure("output_safety", [f"Report output is not verified as program-generated: {report_root}"])
        for path in (dataset_root, report_root):
            if path.exists():
                shutil.rmtree(path)
    paths = {
        "dataset_root": dataset_root,
        "report_root": report_root,
        "manifest": dataset_root / "optical_roi_manifest.csv",
        "build_manifest": dataset_root / "build_manifest.json",
        "coverage": report_root / "mask_coverage_summary.csv",
        "low_coverage": report_root / "low_coverage_cases.csv",
        "previews": report_root / "previews",
        "logs": report_root / "logs",
        "report": report_root / "optical_roi_dataset_v1_build_report.md",
    }
    for roi_name in ROI_NAMES:
        paths[f"mask_{roi_name}"] = dataset_root / "masks" / roi_name
    for key in ("dataset_root", "report_root", "previews", "logs", *(f"mask_{roi}" for roi in ROI_NAMES)):
        paths[key].mkdir(parents=True, exist_ok=True)
    return paths


def save_mask_png(path: Path, mask: np.ndarray, compress_level: int) -> None:
    array = np.asarray(mask)
    if array.shape != (224, 224) or array.dtype != np.uint8:
        raise ValueError(f"Invalid mask before save: shape={array.shape}, dtype={array.dtype}")
    if not set(np.unique(array).tolist()).issubset({0, 255}):
        raise ValueError("Mask contains values outside {0,255}")
    Image.fromarray(array, mode="L").save(path, format="PNG", compress_level=int(compress_level))


def build_dataset(
    args: argparse.Namespace, preflight: PreflightResult, paths: dict[str, Path]
) -> pd.DataFrame:
    exif_index = preflight.exif.set_index("ID")
    rows: list[dict[str, Any]] = []
    for image_id in preflight.ids:
        aligned_path = args.aligned_rgb_dir / f"{image_id}.png"
        parsing_path = args.parsing_label_dir / f"{image_id}.png"
        final_path = args.final_mask_dir / f"{image_id}.png"
        parsing = np.asarray(Image.open(parsing_path))
        final = np.asarray(Image.open(final_path))
        exif = exif_index.loc[image_id]
        exposure, f_number, iso = validate_positive_exif(
            exif["ExposureTime"], exif["FNumber"], exif["ISOSpeedRatings"]
        )
        relative_exposure, log2_iso = derive_exif(exposure, f_number, iso)
        row: dict[str, Any] = {
            "ID": image_id,
            "aligned_rgb_path": project_relative(aligned_path, args.project_root),
            "parsing_label_path": project_relative(parsing_path, args.project_root),
            "final_mask_path": project_relative(final_path, args.project_root),
            "ExposureTime": exposure,
            "FNumber": f_number,
            "ISOSpeedRatings": iso,
            "relative_optical_exposure": relative_exposure,
            "log2_iso_condition": log2_iso,
            "Make": str(exif["Make"]),
            "Model": str(exif["Model"]),
            "camera_id": str(exif["camera_id"]),
            "build_status": "success",
            "failure_reason": "",
        }
        for roi_name in ROI_NAMES:
            prefix = ROI_MANIFEST_PREFIX[roi_name]
            bbox = preflight.boxes[image_id][roi_name]
            mask = build_effective_mask(parsing, final, bbox, preflight.skin_label)
            reason = empty_mask_reason(mask, image_id, roi_name)
            if reason:
                raise BuildFailure("mask_build", [reason])
            mask_path = paths[f"mask_{roi_name}"] / f"{image_id}.png"
            save_mask_png(mask_path, mask, int(args.png_compress_level))
            count = int((mask > 0).sum())
            area = bbox_area(bbox)
            x1, y1, x2, y2 = bbox
            row.update(
                {
                    f"{prefix}_mask_path": project_relative(mask_path, args.project_root),
                    f"{prefix}_x1": x1,
                    f"{prefix}_y1": y1,
                    f"{prefix}_x2": x2,
                    f"{prefix}_y2": y2,
                    f"{prefix}_bbox_area": area,
                    f"{prefix}_valid_skin_pixel_count": count,
                    f"{prefix}_valid_skin_fraction": valid_skin_fraction(count, area),
                    f"{prefix}_mask_sha256": sha256_file(mask_path),
                }
            )
        rows.append(row)
    manifest = pd.DataFrame(rows)
    ordered = [
        "ID",
        "aligned_rgb_path",
        "parsing_label_path",
        "final_mask_path",
        "forehead_mask_path",
        "cheek_image_left_mask_path",
        "cheek_image_right_mask_path",
        "forehead_x1",
        "forehead_y1",
        "forehead_x2",
        "forehead_y2",
        "forehead_bbox_area",
        "cheek_image_left_x1",
        "cheek_image_left_y1",
        "cheek_image_left_x2",
        "cheek_image_left_y2",
        "cheek_image_left_bbox_area",
        "cheek_image_right_x1",
        "cheek_image_right_y1",
        "cheek_image_right_x2",
        "cheek_image_right_y2",
        "cheek_image_right_bbox_area",
        "forehead_valid_skin_pixel_count",
        "forehead_valid_skin_fraction",
        "cheek_image_left_valid_skin_pixel_count",
        "cheek_image_left_valid_skin_fraction",
        "cheek_image_right_valid_skin_pixel_count",
        "cheek_image_right_valid_skin_fraction",
        "ExposureTime",
        "FNumber",
        "ISOSpeedRatings",
        "relative_optical_exposure",
        "log2_iso_condition",
        "Make",
        "Model",
        "camera_id",
        "forehead_mask_sha256",
        "cheek_image_left_mask_sha256",
        "cheek_image_right_mask_sha256",
        "build_status",
        "failure_reason",
    ]
    manifest = manifest[ordered].sort_values("ID", kind="stable").reset_index(drop=True)
    manifest.to_csv(paths["manifest"], index=False, encoding="utf-8-sig")
    return manifest


def summary_values(values: pd.Series, empty_n: int) -> dict[str, Any]:
    numeric = pd.to_numeric(values, errors="coerce")
    valid = numeric.dropna().astype(float)
    result: dict[str, Any] = {
        "valid_n": int(valid.size),
        "missing_n": int(numeric.isna().sum()),
        "empty_n": int(empty_n),
    }
    quantiles = {"p1": 0.01, "p5": 0.05, "p10": 0.10, "p25": 0.25, "median": 0.50, "p75": 0.75, "p90": 0.90, "p95": 0.95, "p99": 0.99}
    if valid.empty:
        result.update({key: math.nan for key in ("min", *quantiles, "max", "mean", "std", "iqr")})
        return result
    result["min"] = float(valid.min())
    for name, probability in quantiles.items():
        result[name] = float(valid.quantile(probability))
    result["max"] = float(valid.max())
    result["mean"] = float(valid.mean())
    result["std"] = float(valid.std(ddof=1)) if len(valid) > 1 else 0.0
    result["iqr"] = result["p75"] - result["p25"]
    return result


def build_coverage_summary(manifest: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    scopes: list[tuple[str, str, pd.DataFrame]] = [("overall", "ALL", manifest)]
    scopes.extend(("camera_id", str(camera), group.copy()) for camera, group in manifest.groupby("camera_id", sort=True))
    for scope_type, camera_id, group in scopes:
        for roi_name in ROI_NAMES:
            prefix = ROI_MANIFEST_PREFIX[roi_name]
            empty_n = int((group[f"{prefix}_valid_skin_pixel_count"] == 0).sum())
            metrics = {
                "valid_skin_pixel_count": f"{prefix}_valid_skin_pixel_count",
                "valid_skin_fraction": f"{prefix}_valid_skin_fraction",
                "bbox_area": f"{prefix}_bbox_area",
            }
            for metric, column in metrics.items():
                rows.append(
                    {
                        "scope_type": scope_type,
                        "camera_id": camera_id,
                        "roi_name": roi_name,
                        "metric": metric,
                        **summary_values(group[column], empty_n),
                    }
                )
    return pd.DataFrame(rows)


def build_low_coverage_cases(manifest: pd.DataFrame, n: int) -> pd.DataFrame:
    records: dict[tuple[str, str], set[str]] = {}
    for roi_name in ROI_NAMES:
        prefix = ROI_MANIFEST_PREFIX[roi_name]
        count_col = f"{prefix}_valid_skin_pixel_count"
        fraction_col = f"{prefix}_valid_skin_fraction"
        lowest_count = manifest.sort_values([count_col, "ID"], kind="stable").head(n)
        lowest_fraction = manifest.sort_values([fraction_col, "ID"], kind="stable").head(n)
        for image_id in lowest_count["ID"].astype(str):
            records.setdefault((roi_name, image_id), set()).add(f"lowest_{n}_valid_skin_pixel_count")
        for image_id in lowest_fraction["ID"].astype(str):
            records.setdefault((roi_name, image_id), set()).add(f"lowest_{n}_valid_skin_fraction")
    index = manifest.set_index("ID")
    rows: list[dict[str, Any]] = []
    for (roi_name, image_id), reasons in records.items():
        prefix = ROI_MANIFEST_PREFIX[roi_name]
        row = index.loc[image_id]
        rows.append(
            {
                "ID": image_id,
                "camera_id": row["camera_id"],
                "roi_name": roi_name,
                "valid_skin_pixel_count": int(row[f"{prefix}_valid_skin_pixel_count"]),
                "valid_skin_fraction": float(row[f"{prefix}_valid_skin_fraction"]),
                "bbox_area": int(row[f"{prefix}_bbox_area"]),
                "selection_reason": ";".join(sorted(reasons)),
            }
        )
    return pd.DataFrame(rows).sort_values(
        ["roi_name", "valid_skin_fraction", "valid_skin_pixel_count", "ID"], kind="stable"
    ).reset_index(drop=True)


def load_font(size: int, bold: bool = False) -> ImageFont.ImageFont:
    name = "arialbd.ttf" if bold else "arial.ttf"
    path = Path(os.environ.get("WINDIR", r"C:\Windows")) / "Fonts" / name
    try:
        return ImageFont.truetype(str(path), size=size)
    except OSError:
        return ImageFont.load_default()


def preview_panel(
    aligned_path: Path,
    mask_path: Path,
    image_id: str,
    camera_id: str,
    roi_name: str,
    bbox: tuple[int, int, int, int],
    pixel_count: int,
    fraction: float,
) -> Image.Image:
    aligned_image = Image.open(aligned_path)
    if aligned_image.mode != "RGB" or aligned_image.size != (224, 224):
        raise ValueError(f"Invalid aligned preview input: {aligned_path}")
    aligned = np.asarray(aligned_image).copy()
    mask = np.asarray(Image.open(mask_path)) > 0
    color = np.asarray(ROI_COLORS[roi_name], dtype=np.float32)
    blended = np.rint(aligned.astype(np.float32) * 0.57 + color * 0.43).astype(np.uint8)
    aligned[mask] = blended[mask]
    panel = Image.new("RGB", (560, 280), "white")
    origin = (8, 36)
    panel.paste(Image.fromarray(aligned, mode="RGB"), origin)
    draw = ImageDraw.Draw(panel)
    font = load_font(13)
    bold_font = load_font(14, bold=True)
    draw.text((8, 7), f"{roi_name} | ID {image_id}", fill=(0, 0, 0), font=bold_font)
    x1, y1, x2, y2 = bbox
    rgb = ROI_COLORS[roi_name]
    draw.rectangle((origin[0] + x1, origin[1] + y1, origin[0] + x2, origin[1] + y2), outline=rgb, width=2)
    details = [
        f"camera: {camera_id}",
        f"bbox: ({x1}, {y1}, {x2}, {y2})",
        f"valid pixels: {pixel_count}",
        f"skin fraction: {fraction:.6f}",
        "mask = skin & final & bbox",
    ]
    y = 48
    for line in details:
        draw.text((244, y), line, fill=(0, 0, 0), font=font)
        y += 25
    return panel


def save_contact_sheet(panels: list[Image.Image], output_path: Path, columns: int) -> None:
    if not panels:
        raise ValueError(f"No panels for {output_path}")
    width, height = panels[0].size
    rows = math.ceil(len(panels) / columns)
    sheet = Image.new("RGB", (width * columns, height * rows), (230, 230, 230))
    for index, panel in enumerate(panels):
        sheet.paste(panel, ((index % columns) * width, (index // columns) * height))
    sheet.save(output_path, format="PNG", compress_level=3)


def build_previews(
    args: argparse.Namespace,
    preflight: PreflightResult,
    manifest: pd.DataFrame,
    low: pd.DataFrame,
    paths: dict[str, Path],
) -> dict[str, Path]:
    index = manifest.set_index("ID")
    outputs: dict[str, Path] = {}
    filenames = {
        "forehead": "lowest_forehead_coverage.png",
        "cheek_image_left": "lowest_cheek_image_left_coverage.png",
        "cheek_image_right": "lowest_cheek_image_right_coverage.png",
    }
    for roi_name in ROI_NAMES:
        panels: list[Image.Image] = []
        subset = low[low["roi_name"].eq(roi_name)]
        for low_row in subset.itertuples(index=False):
            row = index.loc[str(low_row.ID)]
            panels.append(
                preview_panel(
                    args.project_root / row["aligned_rgb_path"],
                    args.project_root / row[f"{roi_name}_mask_path"],
                    str(low_row.ID),
                    str(row["camera_id"]),
                    roi_name,
                    preflight.boxes[str(low_row.ID)][roi_name],
                    int(row[f"{roi_name}_valid_skin_pixel_count"]),
                    float(row[f"{roi_name}_valid_skin_fraction"]),
                )
            )
        output_path = paths["previews"] / filenames[roi_name]
        save_contact_sheet(panels, output_path, int(args.preview_columns))
        outputs[roi_name] = output_path

    manual_panels: list[Image.Image] = []
    for image_id in [str(value) for value in args.manual_forehead_review_ids]:
        if image_id not in index.index:
            raise BuildFailure("preview", [f"Manual review ID is not in study cohort: {image_id}"])
        row = index.loc[image_id]
        manual_panels.append(
            preview_panel(
                args.project_root / row["aligned_rgb_path"],
                args.project_root / row["forehead_mask_path"],
                image_id,
                str(row["camera_id"]),
                "forehead",
                preflight.boxes[image_id]["forehead"],
                int(row["forehead_valid_skin_pixel_count"]),
                float(row["forehead_valid_skin_fraction"]),
            )
        )
    manual_path = paths["previews"] / "manually_checked_forehead_cases.png"
    save_contact_sheet(manual_panels, manual_path, int(args.preview_columns))
    outputs["manual"] = manual_path
    return outputs


def validate_disk_outputs(
    args: argparse.Namespace,
    preflight: PreflightResult,
    paths: dict[str, Path],
) -> dict[str, Any]:
    errors: list[str] = []
    manifest = pd.read_csv(paths["manifest"], dtype={"ID": str})
    if len(manifest) != 500:
        errors.append(f"manifest_rows={len(manifest)},expected=500")
    if manifest["ID"].nunique() != 500:
        errors.append(f"manifest_unique_IDs={manifest['ID'].nunique()},expected=500")
    if set(manifest["ID"]) != set(preflight.ids):
        errors.append("manifest_ID_set_mismatch")
    mask_counts: dict[str, int] = {}
    empty_counts: dict[str, int] = {}
    for roi_name in ROI_NAMES:
        files = sorted(paths[f"mask_{roi_name}"].glob("*.png"))
        mask_counts[roi_name] = len(files)
        empty_counts[roi_name] = 0
        if len(files) != 500:
            errors.append(f"{roi_name}_mask_count={len(files)},expected=500")
        if {path.stem for path in files} != set(preflight.ids):
            errors.append(f"{roi_name}_mask_ID_set_mismatch")
    for row in manifest.itertuples(index=False):
        image_id = str(row.ID)
        parsing = np.asarray(Image.open(args.parsing_label_dir / f"{image_id}.png"))
        final = np.asarray(Image.open(args.final_mask_dir / f"{image_id}.png"))
        base_skin = (parsing == preflight.skin_label) & (final > 0)
        for roi_name in ROI_NAMES:
            prefix = ROI_MANIFEST_PREFIX[roi_name]
            mask_path = args.project_root / getattr(row, f"{prefix}_mask_path")
            if not mask_path.is_file():
                errors.append(f"missing_mask:{image_id}:{roi_name}")
                continue
            image = Image.open(mask_path)
            mask = np.asarray(image)
            if image.size != (224, 224) or mask.shape != (224, 224) or mask.ndim != 2 or mask.dtype != np.uint8:
                errors.append(
                    f"invalid_mask_image:{image_id}:{roi_name}:size={image.size},mode={image.mode},shape={mask.shape},dtype={mask.dtype}"
                )
                continue
            if not set(np.unique(mask).tolist()).issubset({0, 255}):
                errors.append(f"invalid_mask_values:{image_id}:{roi_name}")
            positive = mask > 0
            if not positive.any():
                empty_counts[roi_name] += 1
                errors.append(f"empty_mask:{image_id}:{roi_name}")
            bbox = (
                int(getattr(row, f"{prefix}_x1")),
                int(getattr(row, f"{prefix}_y1")),
                int(getattr(row, f"{prefix}_x2")),
                int(getattr(row, f"{prefix}_y2")),
            )
            region = inclusive_bbox_region((224, 224), bbox)
            if np.any(positive & ~region):
                errors.append(f"mask_outside_bbox:{image_id}:{roi_name}")
            if np.any(positive & ~base_skin):
                errors.append(f"mask_outside_base_skin:{image_id}:{roi_name}")
            disk_count = int(positive.sum())
            manifest_count = int(getattr(row, f"{prefix}_valid_skin_pixel_count"))
            if disk_count != manifest_count:
                errors.append(f"pixel_count_mismatch:{image_id}:{roi_name}:{disk_count}!={manifest_count}")
            if sha256_file(mask_path) != str(getattr(row, f"{prefix}_mask_sha256")):
                errors.append(f"sha256_mismatch:{image_id}:{roi_name}")
        try:
            derived = derive_exif(row.ExposureTime, row.FNumber, row.ISOSpeedRatings)
            if not all(math.isfinite(value) for value in derived):
                errors.append(f"nonfinite_EXIF_derived:{image_id}")
            if not math.isclose(derived[0], float(row.relative_optical_exposure), rel_tol=0, abs_tol=1e-12):
                errors.append(f"relative_optical_exposure_mismatch:{image_id}")
            if not math.isclose(derived[1], float(row.log2_iso_condition), rel_tol=0, abs_tol=1e-12):
                errors.append(f"log2_iso_condition_mismatch:{image_id}")
        except Exception as exc:
            errors.append(f"invalid_EXIF_derived:{image_id}:{exc}")
        if str(row.camera_id) != f"{row.Make}/{row.Model}":
            errors.append(f"camera_id_mismatch:{image_id}")
    after_inventory = historical_inventory_sha256(preflight.historical_inputs, args.project_root)
    if after_inventory != preflight.historical_inventory_sha256:
        errors.append("historical_input_inventory_changed")
    return {
        "status": "PASS" if not errors else "FAIL",
        "errors": errors,
        "manifest_rows": int(len(manifest)),
        "unique_ids": int(manifest["ID"].nunique()),
        "mask_counts": mask_counts,
        "total_mask_count": int(sum(mask_counts.values())),
        "empty_counts": empty_counts,
        "nonfinite_exif_derived_count": int(sum("EXIF_derived" in error for error in errors)),
        "historical_input_inventory_before_sha256": preflight.historical_inventory_sha256,
        "historical_input_inventory_after_sha256": after_inventory,
        "historical_inputs_unchanged": after_inventory == preflight.historical_inventory_sha256,
    }


def get_git_commit(project_root: Path) -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=project_root, text=True, stderr=subprocess.DEVNULL
        ).strip()
    except Exception:
        return "unavailable"


def run_tests(project_root: Path) -> tuple[str, str]:
    command = [sys.executable, "-m", "pytest", "-q", "tests/test_optical_roi_dataset_v1.py"]
    result = subprocess.run(command, cwd=project_root, text=True, capture_output=True)
    output = (result.stdout + "\n" + result.stderr).strip()
    return ("PASS" if result.returncode == 0 else "FAIL", output)


def markdown_table(frame: pd.DataFrame, columns: Sequence[str]) -> str:
    view = frame.loc[:, list(columns)]
    header = "| " + " | ".join(columns) + " |"
    separator = "| " + " | ".join(["---"] * len(columns)) + " |"
    rows = [
        "| " + " | ".join(str(value).replace("|", "\\|") for value in values) + " |"
        for values in view.itertuples(index=False, name=None)
    ]
    return "\n".join([header, separator, *rows])


def write_report(
    args: argparse.Namespace,
    preflight: PreflightResult,
    paths: dict[str, Path],
    manifest: pd.DataFrame,
    coverage: pd.DataFrame,
    low: pd.DataFrame,
    validation: dict[str, Any],
    test_status: str,
    test_output: str,
    previews: dict[str, Path],
) -> None:
    overall = coverage[coverage["scope_type"].eq("overall")].copy()
    distribution = overall[overall["metric"].isin(["valid_skin_pixel_count", "valid_skin_fraction"])].copy()
    distribution = distribution[
        ["roi_name", "metric", "valid_n", "empty_n", "min", "p1", "p5", "median", "p95", "p99", "max", "mean", "std", "iqr"]
    ]
    device_fraction = coverage[
        coverage["scope_type"].eq("camera_id") & coverage["metric"].eq("valid_skin_fraction")
    ][["camera_id", "roi_name", "valid_n", "min", "median", "max", "mean", "std"]]
    manual = manifest[manifest["ID"].isin([str(value) for value in args.manual_forehead_review_ids])][
        ["ID", "camera_id", "forehead_valid_skin_pixel_count", "forehead_valid_skin_fraction", "forehead_bbox_area"]
    ]
    status = "COMPLETE" if validation["status"] == "PASS" and test_status == "PASS" else "INCOMPLETE"
    ready = "YES" if status == "COMPLETE" else "NO"
    lines = [
        "# Optical ROI Dataset V1 构建报告",
        "",
        "## 1. 完成状态",
        "",
        f"- `OPTICAL_ROI_DATASET_STATUS={status}`",
        f"- `READY_FOR_OPTICAL_FEATURE_EXTRACTION={ready}`",
        "- 本状态仅表示500例输入和三类有效皮肤mask构建、保存与完整性验证完成，不表示物理反演模型或物理参数有效。",
        "",
        "## 2. 新增文件",
        "",
        "- `preprocessing/build_optical_roi_dataset_v1.py`",
        "- `config/preprocess/optical_roi_dataset_v1.yaml`",
        "- `tests/test_optical_roi_dataset_v1.py`",
        "- `data/processed/optical_roi_dataset_v1/`：1500张mask、manifest与build manifest。",
        "- `reports/optical_roi_dataset_v1/`：覆盖统计、低覆盖清单、预览、日志与本报告。",
        "",
        "## 3. 未修改的历史数据声明",
        "",
        f"构建前后历史输入库存摘要一致：`{validation['historical_input_inventory_before_sha256']}`。程序的写入路径仅限两个新输出目录；没有修改meanbg、aligned、parsing label、final mask或ROI日志。",
        "",
        "## 4. 固定500例ID来源",
        "",
        f"唯一研究集合来自 `{project_relative(args.study_id_dir, args.project_root)}` 的500张PNG完整stem。manifest为500行、500个唯一ID，与该集合完全一致；没有加入上游多出的22例。",
        "",
        "## 5. 图像和mask来源",
        "",
        f"- aligned RGB：`{project_relative(args.aligned_rgb_dir, args.project_root)}`",
        f"- parsing label：`{project_relative(args.parsing_label_dir, args.project_root)}`",
        f"- final mask：`{project_relative(args.final_mask_dir, args.project_root)}`",
        "- aligned均为224×224 RGB uint8；parsing label为224×224单通道离散标签；final mask以`>0`定义有效区域。",
        "",
        "## 6. bbox来源与端点约定",
        "",
        f"bbox日志为 `{project_relative(args.bbox_log, args.project_root)}`，SHA256=`{preflight.bbox_log_sha256}`。forehead读取`forehead_roi`通用bbox字段；两侧cheek读取`cheek_roi`专用left/right字段。CSV为包含式`(x1,y1,x2,y2)`，NumPy切片使用`[y1:y2+1, x1:x2+1]`。",
        "",
        "## 7. skin标签代码依据",
        "",
        f"从 `{project_relative(args.skin_label_source_code, args.project_root)}` 的`CLASS_SKIN`定义读取到`skin_label={preflight.skin_label}`，不是凭经验指定。",
        "",
        "## 8. 三类mask精确定义",
        "",
        "`base_skin = (parsing_label == 1) AND (final_mask > 0)`；每类mask为`base_skin AND 对应包含式bbox区域`。保存为224×224、单通道uint8 PNG，背景0、有效像素255。没有resize、padding、羽化、腐蚀、膨胀、开闭运算、bbox移动或个例规则。",
        "",
        "## 9. EXIF来源和连接方式",
        "",
        f"设备字段来自 `{project_relative(args.exif_identity_csv, args.project_root)}` 的`ID/Make/Model`；数值字段来自 `{project_relative(args.exif_parameter_long_csv, args.project_root)}` 中参数名为ExposureTime、FNumber、ISOSpeedRatings的行。只加载这些非临床列，按完整字符串ID一对一连接。",
        "",
        "## 10. EXIF派生公式",
        "",
        "- `relative_optical_exposure = log2(ExposureTime / FNumber^2)`",
        "- `log2_iso_condition = log2(ISOSpeedRatings / 100)`",
        "- 未做全队列标准化、设备内中心化或camera数值编码。",
        "",
        "## 11. 500例完整性验证",
        "",
        f"磁盘复读状态：`{validation['status']}`；manifest={validation['manifest_rows']}行、唯一ID={validation['unique_ids']}；三类mask数={validation['mask_counts']}，总数={validation['total_mask_count']}。所有mask均为224×224单通道uint8、仅0/255、非空、位于bbox内且属于`parsing skin AND final mask`；manifest像素数和SHA256均与磁盘一致。",
        "",
        "## 12–13. ROI有效像素数和skin比例分布",
        "",
        markdown_table(distribution, list(distribution.columns)),
        "",
        "## 14. 两设备分层统计",
        "",
        markdown_table(device_fraction, list(device_fraction.columns)),
        "",
        "仅提供描述性统计，没有设备间显著性检验。",
        "",
        "## 15. 空mask、缺失和非法记录",
        "",
        f"空mask={validation['empty_counts']}；非法bbox={preflight.invalid_bbox_count}；缺失EXIF={preflight.missing_exif_count}；非有限EXIF派生值={validation['nonfinite_exif_derived_count']}；完整性错误={len(validation['errors'])}。",
        "",
        "## 16. 低覆盖病例清单",
        "",
        "低像素数和低比例病例只记录、不排除。完整清单位于`low_coverage_cases.csv`。",
        "",
        markdown_table(low, ["ID", "camera_id", "roi_name", "valid_skin_pixel_count", "valid_skin_fraction", "bbox_area", "selection_reason"]),
        "",
        "## 17. 三个已人工确认额部病例",
        "",
        markdown_table(manual, list(manual.columns)),
        "",
        f"记录图：`{project_relative(previews['manual'], args.project_root)}`。这些病例没有被自动标记为失败。",
        "",
        "## 18. 测试结果",
        "",
        f"专项测试状态：`{test_status}`。测试输出：`{test_output.replace(chr(10), ' | ')}`",
        "",
        "## 19. 是否满足继续提取区域光学量的条件",
        "",
        f"`READY_FOR_OPTICAL_FEATURE_EXTRACTION={ready}`。允许下一阶段读取aligned RGB并仅在这些mask有效像素上计算预先定义的区域光学量；本阶段没有计算任何RGB统计或物理反演。",
        "",
        "## 20. 已知限制",
        "",
        "1. saved aligned RGB已经经过上游双线性几何重采样，不是相机原始坐标像素。",
        "2. mask依赖BiSeNet离散标签和既有final mask；解析错误会传递到ROI。",
        "3. 部分额部有效像素很少，但V1不设置最低阈值，也不自动修改bbox。",
        "4. image-left/right仅指图像x坐标方向，不代表患者解剖学左右。",
        "5. 数据集完成不证明物理反演有效，也不证明跨设备无混杂。",
        "",
        "## 21. 下一步最小建议",
        "",
        "保持本V1 mask和manifest不变，新增独立的区域光学量提取步骤；任何标准化参数仅在后续每个训练折的训练子集内计算。",
        "",
    ]
    paths["report"].write_text("\n".join(lines), encoding="utf-8")


def write_build_manifest(
    args: argparse.Namespace,
    preflight: PreflightResult,
    paths: dict[str, Path],
    validation: dict[str, Any],
    test_status: str,
) -> None:
    payload = {
        "dataset_name": str(args.dataset_name),
        "version": str(args.version),
        "build_time": datetime.now().astimezone().isoformat(timespec="seconds"),
        "project_root": str(args.project_root),
        "study_id_source": project_relative(args.study_id_dir, args.project_root),
        "study_id_count": len(preflight.ids),
        "aligned_rgb_source": project_relative(args.aligned_rgb_dir, args.project_root),
        "parsing_label_source": project_relative(args.parsing_label_dir, args.project_root),
        "final_mask_source": project_relative(args.final_mask_dir, args.project_root),
        "bbox_log_source": project_relative(args.bbox_log, args.project_root),
        "bbox_log_sha256": preflight.bbox_log_sha256,
        "exif_source_files": list(preflight.exif_source_sha256),
        "exif_source_sha256": preflight.exif_source_sha256,
        "skin_label": preflight.skin_label,
        "skin_label_source": project_relative(args.skin_label_source_code, args.project_root),
        "bbox_endpoint_convention": "inclusive_x2_y2; numpy_slice_y1_y2_plus_1_x1_x2_plus_1",
        "mask_definition": "(parsing_label == skin_label) AND (final_mask > 0) AND inclusive_bbox_region",
        "morphology_applied": False,
        "resize_applied": False,
        "rgb_roi_images_generated": False,
        "output_mask_count": validation["total_mask_count"],
        "device_counts": preflight.exif["camera_id"].value_counts().sort_index().astype(int).to_dict(),
        "script_path": project_relative(Path(__file__), args.project_root),
        "config_path": project_relative(args.config_path, args.project_root),
        "script_sha256": sha256_file(Path(__file__)),
        "config_sha256": sha256_file(args.config_path),
        "git_commit": get_git_commit(args.project_root),
        "test_status": test_status,
        "integrity_validation_status": validation["status"],
        "historical_input_inventory_before_sha256": validation[
            "historical_input_inventory_before_sha256"
        ],
        "historical_input_inventory_after_sha256": validation[
            "historical_input_inventory_after_sha256"
        ],
        "optical_roi_dataset_status": (
            "COMPLETE" if validation["status"] == "PASS" and test_status == "PASS" else "INCOMPLETE"
        ),
    }
    paths["build_manifest"].write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def write_failure_log(args: argparse.Namespace, failure: BuildFailure) -> Path | None:
    report_root = args.report_output_dir
    try:
        if report_root.exists() and any(report_root.iterdir()) and not verify_existing_generated_output(report_root, False):
            return None
        logs = report_root / "logs"
        logs.mkdir(parents=True, exist_ok=True)
        path = logs / f"{failure.stage}_errors.txt"
        path.write_text("\n".join(failure.errors) + "\n", encoding="utf-8")
        return path
    except Exception:
        return None


def print_failure(args: argparse.Namespace, failure: BuildFailure, error_log: Path | None) -> None:
    command = (
        "python preprocessing/build_optical_roi_dataset_v1.py "
        "--config config/preprocess/optical_roi_dataset_v1.yaml"
    )
    if args.dataset_output_dir.exists() or args.report_output_dir.exists():
        command += " --overwrite"
    print("OPTICAL_ROI_DATASET_STATUS=FAILED")
    print(f"FAILED_STAGE={failure.stage}")
    print("FAILED_IDS=" + ",".join(sorted({token for error in failure.errors for token in re.findall(r"(?:^|:)([A-Za-z0-9]+)(?::|$)", error)})))
    print(f"ERROR_LOG_PATH={error_log if error_log else 'unavailable'}")
    print(f"EXACT_RESUME_OR_RERUN_COMMAND={command}")


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        preflight = run_preflight(args)
        if args.preflight_only:
            print("PREFLIGHT_STATUS=PASS")
            print(f"STUDY_ID_COUNT={len(preflight.ids)}")
            print(f"UNIQUE_ID_COUNT={len(set(preflight.ids))}")
            print(f"SKIN_LABEL={preflight.skin_label}")
            return 0
        test_status, test_output = run_tests(args.project_root)
        if test_status != "PASS":
            raise BuildFailure("tests", [test_output])
        paths = prepare_output_dirs(args)
        (paths["logs"] / "test_output.txt").write_text(test_output + "\n", encoding="utf-8")
        manifest = build_dataset(args, preflight, paths)
        coverage = build_coverage_summary(manifest)
        coverage.to_csv(paths["coverage"], index=False, encoding="utf-8-sig")
        low = build_low_coverage_cases(manifest, int(args.low_coverage_n))
        low.to_csv(paths["low_coverage"], index=False, encoding="utf-8-sig")
        previews = build_previews(args, preflight, manifest, low, paths)
        validation = validate_disk_outputs(args, preflight, paths)
        (paths["logs"] / "integrity_validation.json").write_text(
            json.dumps(validation, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        write_build_manifest(args, preflight, paths, validation, test_status)
        write_report(
            args,
            preflight,
            paths,
            manifest,
            coverage,
            low,
            validation,
            test_status,
            test_output,
            previews,
        )
        if validation["status"] != "PASS":
            raise BuildFailure("integrity_validation", validation["errors"])
        counts = preflight.exif["camera_id"].value_counts().to_dict()
        print("OPTICAL_ROI_DATASET_STATUS=COMPLETE")
        print(f"STUDY_ID_COUNT={len(preflight.ids)}")
        print(f"UNIQUE_ID_COUNT={len(set(preflight.ids))}")
        print(f"FOREHEAD_MASK_COUNT={validation['mask_counts']['forehead']}")
        print(f"CHEEK_IMAGE_LEFT_MASK_COUNT={validation['mask_counts']['cheek_image_left']}")
        print(f"CHEEK_IMAGE_RIGHT_MASK_COUNT={validation['mask_counts']['cheek_image_right']}")
        print(f"TOTAL_MASK_COUNT={validation['total_mask_count']}")
        print(f"EMPTY_FOREHEAD_MASKS={validation['empty_counts']['forehead']}")
        print(f"EMPTY_CHEEK_IMAGE_LEFT_MASKS={validation['empty_counts']['cheek_image_left']}")
        print(f"EMPTY_CHEEK_IMAGE_RIGHT_MASKS={validation['empty_counts']['cheek_image_right']}")
        print(f"INVALID_BBOX_COUNT={preflight.invalid_bbox_count}")
        print(f"MISSING_EXIF_COUNT={preflight.missing_exif_count}")
        print(f"NONFINITE_EXIF_DERIVED_COUNT={validation['nonfinite_exif_derived_count']}")
        print(f"HONOR_CASE_COUNT={counts.get('HONOR/BVL-AN00', 0)}")
        print(f"XIAOMI_CASE_COUNT={counts.get('Xiaomi/M2006J10C', 0)}")
        print(f"TEST_STATUS={test_status}")
        print("READY_FOR_OPTICAL_FEATURE_EXTRACTION=YES")
        print(f"DATASET_PATH={paths['dataset_root']}")
        print(f"MANIFEST_PATH={paths['manifest']}")
        print(f"REPORT_PATH={paths['report']}")
        print(f"LOW_COVERAGE_PREVIEW_PATH={previews['forehead']}")
        return 0
    except BuildFailure as failure:
        error_log = write_failure_log(args, failure)
        print_failure(args, failure, error_log)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
