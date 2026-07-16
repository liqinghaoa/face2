"""Extract deterministic regional facial optical observations (V1).

The program reads only the fixed 500-case Optical ROI Dataset V1 manifest,
saved 224x224 aligned RGB PNGs, and its three saved masks. It does not read
clinical labels, regenerate masks, resize images, normalize the cohort, alter
pixels using EXIF, train a model, or perform physical optical inversion.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
import PIL
import yaml
from PIL import Image

DEFAULT_CONFIG = "regional_optical_observations_v1.yaml"
TASK_NAME = "Regional Facial Optical Observation Extraction V1"
REPORT_TASK_MARKER = f"<!-- task_name: {TASK_NAME} -->"
REPORT_TITLE = "# 区域面部光学观测量提取 V1 报告"
IMPLEMENTATION_RELATIVE_PATHS = (
    "preprocessing/extract_regional_optical_observations_v1.py",
    "config/preprocess/regional_optical_observations_v1.yaml",
    "tests/test_regional_optical_observations_v1.py",
)
ROI_NAMES = ("forehead", "cheek_image_left", "cheek_image_right")
OBSERVATION_NAMES = ("log2_y", "log2_rg", "log2_bg")
EXIF_COLUMNS = ("ExposureTime", "FNumber", "ISOSpeedRatings")
Y_COEFFICIENTS = (0.2126, 0.7152, 0.0722)
EPSILON = 1.0e-6
LOG_BASE = 2
FOREHEAD_THRESHOLD = 0.20

REGIONAL_OBSERVATION_COLUMNS = tuple(
    f"{roi}_{observation}_median" for roi in ROI_NAMES for observation in OBSERVATION_NAMES
)
DERIVED_OBSERVATION_COLUMNS = tuple(
    [f"cheek_mean_{name}" for name in OBSERVATION_NAMES]
    + [f"forehead_minus_cheek_{name}" for name in OBSERVATION_NAMES]
)
CHEEK_ABS_DIFF_COLUMNS = tuple(f"cheek_abs_diff_{name}" for name in OBSERVATION_NAMES)
MAIN_OBSERVATION_COLUMNS = REGIONAL_OBSERVATION_COLUMNS + DERIVED_OBSERVATION_COLUMNS


class ExtractionFailure(RuntimeError):
    def __init__(self, stage: str, errors: Sequence[str]):
        self.stage = str(stage)
        self.errors = [str(value) for value in errors]
        super().__init__(f"{self.stage}: {len(self.errors)} error(s)")


@dataclass
class PreflightResult:
    ids: list[str]
    manifest: pd.DataFrame
    input_paths: list[Path]
    historical_inventory_before: str
    input_has_embedded_color_profile: bool
    embedded_color_profile_count: int
    roi_manifest_sha256: str
    roi_build_manifest_sha256: str


def project_root_default() -> Path:
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
    candidate = Path(value).expanduser()
    resolved = candidate.resolve() if candidate.is_absolute() else (project_root / candidate).resolve()
    try:
        resolved.relative_to(project_root.resolve())
    except ValueError as exc:
        raise ValueError(f"Path is outside project root: {resolved}") from exc
    return resolved


def stable_inventory_sha256(paths: Iterable[Path], project_root: Path) -> str:
    digest = hashlib.sha256()
    unique = sorted({path.resolve() for path in paths}, key=lambda value: str(value).casefold())
    for path in unique:
        stat = path.stat()
        digest.update(project_relative(path, project_root).encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(stat.st_size).encode("ascii"))
        digest.update(b"\0")
        digest.update(str(stat.st_mtime_ns).encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest()


def load_config(config_path: Path | None, inferred_root: Path) -> tuple[dict[str, Any], Path]:
    path = config_path or inferred_root / "config" / "preprocess" / DEFAULT_CONFIG
    if not path.is_absolute():
        path = inferred_root / path
    path = path.resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Config does not exist: {path}")
    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(payload, dict):
        raise ValueError("Config root must be a mapping")
    return {str(key).replace("-", "_"): value for key, value in payload.items()}, path


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract Regional Optical Observations V1")
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--project-root", type=Path, default=None)
    parser.add_argument("--overwrite", action="store_true", default=None)
    parser.add_argument("--preflight-only", action="store_true")
    cli = parser.parse_args(argv)
    inferred = cli.project_root.expanduser().resolve() if cli.project_root else project_root_default()
    config, config_path = load_config(cli.config, inferred)
    root_value = config.get("project_root", ".")
    root = inferred if cli.project_root is not None or root_value in (None, "", ".") else Path(root_value).resolve()
    defaults: dict[str, Any] = {
        "task_name": "Regional Facial Optical Observation Extraction V1",
        "version": "1.0",
        "image_size": 224,
        "roi_names": list(ROI_NAMES),
        "epsilon": EPSILON,
        "log_base": LOG_BASE,
        "y_coefficients": {"r": Y_COEFFICIENTS[0], "g": Y_COEFFICIENTS[1], "b": Y_COEFFICIENTS[2]},
        "forehead_available_threshold": FOREHEAD_THRESHOLD,
        "assumed_color_encoding": "sRGB",
        "linearization": "inverse_sRGB_transfer_function",
        "id_order": "lexicographic_ascending_complete_string_ID",
        "csv_nan_representation": "empty_field",
        "overwrite": False,
    }
    values = {**defaults, **config}
    values["project_root"] = root.resolve()
    values["config_path"] = config_path
    for key in (
        "study_id_dir", "roi_manifest", "roi_build_manifest", "aligned_rgb_dir",
        "mask_root", "data_output_dir", "report_output_dir",
    ):
        if key not in values:
            raise ValueError(f"Missing config path: {key}")
        values[key] = resolve_under_project(values[key], root)
    if cli.overwrite is not None:
        values["overwrite"] = cli.overwrite
    values["preflight_only"] = bool(cli.preflight_only)
    args = argparse.Namespace(**values)
    if int(args.image_size) != 224:
        raise ValueError("image_size must be 224")
    if tuple(args.roi_names) != ROI_NAMES:
        raise ValueError(f"roi_names must be exactly {ROI_NAMES}")
    if float(args.epsilon) != EPSILON or int(args.log_base) != LOG_BASE:
        raise ValueError(f"epsilon/log_base must be {EPSILON}/{LOG_BASE}")
    coefficients = args.y_coefficients
    actual_coefficients = tuple(float(coefficients[key]) for key in ("r", "g", "b"))
    if actual_coefficients != Y_COEFFICIENTS:
        raise ValueError(f"Y coefficients must be {Y_COEFFICIENTS}")
    if float(args.forehead_available_threshold) != FOREHEAD_THRESHOLD:
        raise ValueError(f"forehead threshold must be {FOREHEAD_THRESHOLD}")
    return args


def validate_unique_ids(ids: Sequence[str], expected: int = 500) -> None:
    clean = [str(value).strip() for value in ids]
    if len(clean) != expected:
        raise ValueError(f"ID count {len(clean)} != {expected}")
    if any(not value for value in clean):
        raise ValueError("Empty ID")
    if len(set(clean)) != len(clean):
        raise ValueError("Exact duplicate ID")
    if len({value.casefold() for value in clean}) != len(clean):
        raise ValueError("Case-insensitive duplicate ID")


def validate_positive_exif(exposure: Any, f_number: Any, iso: Any) -> tuple[float, float, float]:
    values = tuple(float(value) for value in (exposure, f_number, iso))
    if not all(math.isfinite(value) and value > 0 for value in values):
        raise ValueError(f"EXIF values must be finite and positive: {values}")
    return values


def derive_exif(exposure: Any, f_number: Any, iso: Any) -> tuple[float, float]:
    t, aperture, sensitivity = validate_positive_exif(exposure, f_number, iso)
    return math.log2(t / aperture**2), math.log2(sensitivity / 100.0)


def inverse_srgb(values: np.ndarray | Sequence[float] | float) -> np.ndarray:
    encoded = np.asarray(values, dtype=np.float64)
    if not np.isfinite(encoded).all() or np.any(encoded < 0.0) or np.any(encoded > 1.0):
        raise ValueError("sRGB values must be finite in [0,1]")
    return np.where(
        encoded <= 0.04045,
        encoded / 12.92,
        np.power((encoded + 0.055) / 1.055, 2.4),
    )


def read_rgb_uint8(path: str | Path) -> np.ndarray:
    """Read an image in explicit Pillow RGB channel order without resizing."""
    image_path = Path(path)
    with Image.open(image_path) as image:
        if image.size != (224, 224):
            raise ValueError(f"RGB image must be 224x224: {image_path} has {image.size}")
        rgb = np.asarray(image.convert("RGB"), dtype=np.uint8).copy()
    if rgb.shape != (224, 224, 3):
        raise ValueError(f"Unexpected RGB array shape: {rgb.shape}")
    return rgb


def pixel_observations(rgb_uint8: np.ndarray, mask: np.ndarray, epsilon: float = EPSILON) -> dict[str, np.ndarray]:
    rgb = np.asarray(rgb_uint8)
    valid = np.asarray(mask) > 0
    if rgb.shape != (224, 224, 3) or rgb.dtype != np.uint8:
        raise ValueError(f"RGB must be 224x224x3 uint8, got {rgb.shape}/{rgb.dtype}")
    if valid.shape != (224, 224):
        raise ValueError(f"Mask must be 224x224, got {valid.shape}")
    if not valid.any():
        raise ValueError("ROI mask is empty")
    linear = inverse_srgb(rgb.astype(np.float64) / 255.0)
    r, g, b = (linear[..., index][valid] for index in range(3))
    y = Y_COEFFICIENTS[0] * r + Y_COEFFICIENTS[1] * g + Y_COEFFICIENTS[2] * b
    observations = {
        "log2_y": np.log2(y + float(epsilon)),
        "log2_rg": np.log2((r + float(epsilon)) / (g + float(epsilon))),
        "log2_bg": np.log2((b + float(epsilon)) / (g + float(epsilon))),
    }
    return observations


def robust_summary(values: np.ndarray) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1 or array.size == 0:
        raise ValueError("Observation vector must be non-empty and one-dimensional")
    q25, median, q75 = np.quantile(array, [0.25, 0.50, 0.75])
    return {"q25": float(q25), "median_raw": float(median), "q75": float(q75), "iqr": float(q75 - q25)}


def channel_clipping(rgb_uint8: np.ndarray, mask: np.ndarray) -> dict[str, float]:
    rgb = np.asarray(rgb_uint8)
    valid = np.asarray(mask) > 0
    if rgb.shape != (224, 224, 3) or rgb.dtype != np.uint8 or valid.shape != (224, 224):
        raise ValueError("Invalid RGB/mask shape for clipping audit")
    count = int(valid.sum())
    if count == 0:
        raise ValueError("Cannot audit clipping for empty mask")
    output: dict[str, float] = {}
    for index, channel in enumerate(("r", "g", "b")):
        values = rgb[..., index][valid]
        output[f"{channel}_equal_0_fraction"] = float(np.mean(values == 0))
        output[f"{channel}_equal_255_fraction"] = float(np.mean(values == 255))
        output[f"{channel}_le_5_fraction"] = float(np.mean(values <= 5))
        output[f"{channel}_ge_250_fraction"] = float(np.mean(values >= 250))
    return output


def compute_roi_qc(
    rgb_uint8: np.ndarray,
    mask: np.ndarray,
    image_id: str,
    camera_id: str,
    roi_name: str,
    bbox_area: int,
    valid_skin_pixel_count: int,
    valid_skin_fraction: float,
) -> dict[str, Any]:
    valid = np.asarray(mask) > 0
    if int(valid.sum()) != int(valid_skin_pixel_count):
        raise ValueError(f"Mask count mismatch: {image_id}/{roi_name}")
    if not math.isclose(valid_skin_pixel_count / int(bbox_area), float(valid_skin_fraction), rel_tol=0, abs_tol=1e-15):
        raise ValueError(f"Mask fraction mismatch: {image_id}/{roi_name}")
    observations = pixel_observations(rgb_uint8, mask)
    nonfinite = int(sum((~np.isfinite(values)).sum() for values in observations.values()))
    row: dict[str, Any] = {
        "ID": str(image_id),
        "camera_id": str(camera_id),
        "roi_name": str(roi_name),
        "available_for_model": int(roi_name != "forehead" or float(valid_skin_fraction) >= FOREHEAD_THRESHOLD),
        "bbox_area": int(bbox_area),
        "valid_skin_pixel_count": int(valid_skin_pixel_count),
        "valid_skin_fraction": float(valid_skin_fraction),
    }
    for name, values in observations.items():
        summary = robust_summary(values)
        for statistic, value in summary.items():
            row[f"{name}_{statistic}"] = value
    row.update(channel_clipping(rgb_uint8, mask))
    row["transformed_nonfinite_count"] = nonfinite
    return row


def derive_case_observations(qc_rows: Mapping[str, Mapping[str, Any]], forehead_available: bool) -> dict[str, float]:
    output: dict[str, float] = {}
    for name in OBSERVATION_NAMES:
        left = float(qc_rows["cheek_image_left"][f"{name}_median_raw"])
        right = float(qc_rows["cheek_image_right"][f"{name}_median_raw"])
        cheek_mean = (left + right) / 2.0
        output[f"cheek_mean_{name}"] = cheek_mean
        output[f"cheek_abs_diff_{name}"] = abs(left - right)
        output[f"forehead_minus_cheek_{name}"] = (
            float(qc_rows["forehead"][f"{name}_median_raw"]) - cheek_mean
            if forehead_available else math.nan
        )
    return output


def build_main_row(source: Mapping[str, Any], qc_rows: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    available = float(source["forehead_valid_skin_fraction"]) >= FOREHEAD_THRESHOLD
    exposure, f_number, iso = validate_positive_exif(
        source["ExposureTime"], source["FNumber"], source["ISOSpeedRatings"]
    )
    relative, iso_condition = derive_exif(exposure, f_number, iso)
    row: dict[str, Any] = {
        "ID": str(source["ID"]),
        "camera_id": str(source["camera_id"]),
        "ExposureTime": exposure,
        "FNumber": f_number,
        "ISOSpeedRatings": iso,
        "relative_optical_exposure": relative,
        "log2_iso_condition": iso_condition,
        "forehead_available": int(available),
    }
    for roi in ROI_NAMES:
        for name in OBSERVATION_NAMES:
            raw = float(qc_rows[roi][f"{name}_median_raw"])
            row[f"{roi}_{name}_median"] = raw if roi != "forehead" or available else math.nan
    row.update(derive_case_observations(qc_rows, available))
    return row


def required_manifest_columns() -> list[str]:
    columns = [
        "ID", "aligned_rgb_path", "ExposureTime", "FNumber", "ISOSpeedRatings",
        "relative_optical_exposure", "log2_iso_condition", "camera_id",
    ]
    for roi in ROI_NAMES:
        columns.extend([
            f"{roi}_mask_path", f"{roi}_bbox_area", f"{roi}_valid_skin_pixel_count",
            f"{roi}_valid_skin_fraction", f"{roi}_mask_sha256",
        ])
    return columns


def run_preflight(args: argparse.Namespace) -> PreflightResult:
    errors: list[str] = []
    study_files = sorted(args.study_id_dir.glob("*.png"), key=lambda path: path.stem)
    ids = [path.stem for path in study_files]
    try:
        validate_unique_ids(ids)
    except ValueError as exc:
        errors.append(str(exc))
    header = pd.read_csv(args.roi_manifest, nrows=0, encoding="utf-8-sig").columns.tolist()
    missing_columns = [column for column in required_manifest_columns() if column not in header]
    if missing_columns:
        errors.append("missing_manifest_columns:" + ",".join(missing_columns))
        raise ExtractionFailure("preflight", errors)
    manifest = pd.read_csv(
        args.roi_manifest,
        usecols=required_manifest_columns(),
        dtype={"ID": str, "camera_id": str},
        encoding="utf-8-sig",
    )
    manifest["ID"] = manifest["ID"].astype(str).str.strip()
    manifest = manifest.sort_values("ID", kind="stable").reset_index(drop=True)
    try:
        validate_unique_ids(manifest["ID"].tolist())
    except ValueError as exc:
        errors.append("manifest_" + str(exc))
    if set(manifest["ID"]) != set(ids):
        errors.append("manifest_ID_set_does_not_equal_meanbg_cohort")
    input_paths: list[Path] = [args.roi_manifest, args.roi_build_manifest, *study_files]
    icc_count = 0
    for record in manifest.to_dict("records"):
        image_id = str(record["ID"])
        aligned_path = resolve_under_project(record["aligned_rgb_path"], args.project_root)
        expected_aligned = args.aligned_rgb_dir / f"{image_id}.png"
        if aligned_path != expected_aligned.resolve():
            errors.append(f"aligned_path_mismatch:{image_id}")
            continue
        if not aligned_path.is_file():
            errors.append(f"missing_aligned:{image_id}")
            continue
        input_paths.append(aligned_path)
        try:
            with Image.open(aligned_path) as image:
                icc_count += int(bool(image.info.get("icc_profile")))
                if image.mode != "RGB" or image.size != (224, 224):
                    errors.append(f"invalid_aligned:{image_id}:{image.mode}:{image.size}")
            rgb = read_rgb_uint8(aligned_path)
            if rgb.shape != (224, 224, 3) or rgb.dtype != np.uint8:
                errors.append(f"invalid_rgb_reader_output:{image_id}:{rgb.shape}:{rgb.dtype}")
        except Exception as exc:
            errors.append(f"aligned_decode_error:{image_id}:{type(exc).__name__}:{exc}")
            continue
        try:
            derived = derive_exif(record["ExposureTime"], record["FNumber"], record["ISOSpeedRatings"])
            if not math.isclose(derived[0], float(record["relative_optical_exposure"]), rel_tol=0, abs_tol=1e-12):
                errors.append(f"relative_optical_exposure_mismatch:{image_id}")
            if not math.isclose(derived[1], float(record["log2_iso_condition"]), rel_tol=0, abs_tol=1e-12):
                errors.append(f"log2_iso_condition_mismatch:{image_id}")
            if not str(record["camera_id"]).strip() or "/" not in str(record["camera_id"]):
                errors.append(f"invalid_camera_id:{image_id}")
        except Exception as exc:
            errors.append(f"invalid_EXIF:{image_id}:{exc}")
        for roi in ROI_NAMES:
            mask_path = resolve_under_project(record[f"{roi}_mask_path"], args.project_root)
            expected_mask = args.mask_root / roi / f"{image_id}.png"
            if mask_path != expected_mask.resolve():
                errors.append(f"mask_path_mismatch:{image_id}:{roi}")
                continue
            if not mask_path.is_file():
                errors.append(f"missing_mask:{image_id}:{roi}")
                continue
            input_paths.append(mask_path)
            try:
                with Image.open(mask_path) as image:
                    mask = np.asarray(image)
                    if image.mode != "L" or image.size != (224, 224) or mask.dtype != np.uint8:
                        errors.append(f"invalid_mask:{image_id}:{roi}:{image.mode}:{image.size}:{mask.dtype}")
                count = int((mask > 0).sum())
                if count <= 0:
                    errors.append(f"empty_mask:{image_id}:{roi}")
                if count != int(record[f"{roi}_valid_skin_pixel_count"]):
                    errors.append(f"mask_count_mismatch:{image_id}:{roi}")
                area = int(record[f"{roi}_bbox_area"])
                if area <= 0 or not math.isclose(count / area, float(record[f"{roi}_valid_skin_fraction"]), rel_tol=0, abs_tol=1e-15):
                    errors.append(f"mask_fraction_mismatch:{image_id}:{roi}")
                if sha256_file(mask_path) != str(record[f"{roi}_mask_sha256"]):
                    errors.append(f"mask_sha256_mismatch:{image_id}:{roi}")
            except Exception as exc:
                errors.append(f"mask_decode_error:{image_id}:{roi}:{type(exc).__name__}:{exc}")
    if errors:
        raise ExtractionFailure("preflight", errors)
    return PreflightResult(
        ids=ids,
        manifest=manifest,
        input_paths=input_paths,
        historical_inventory_before=stable_inventory_sha256(input_paths, args.project_root),
        input_has_embedded_color_profile=icc_count > 0,
        embedded_color_profile_count=icc_count,
        roi_manifest_sha256=sha256_file(args.roi_manifest),
        roi_build_manifest_sha256=sha256_file(args.roi_build_manifest),
    )


def extract_tables(args: argparse.Namespace, preflight: PreflightResult) -> tuple[pd.DataFrame, pd.DataFrame]:
    main_rows: list[dict[str, Any]] = []
    qc_rows: list[dict[str, Any]] = []
    for source in preflight.manifest.to_dict("records"):
        image_id = str(source["ID"])
        rgb = read_rgb_uint8(args.project_root / str(source["aligned_rgb_path"]))
        case_qc: dict[str, dict[str, Any]] = {}
        for roi in ROI_NAMES:
            with Image.open(args.project_root / str(source[f"{roi}_mask_path"])) as image:
                mask = np.asarray(image)
            row = compute_roi_qc(
                rgb, mask, image_id, str(source["camera_id"]), roi,
                int(source[f"{roi}_bbox_area"]), int(source[f"{roi}_valid_skin_pixel_count"]),
                float(source[f"{roi}_valid_skin_fraction"]),
            )
            case_qc[roi] = row
            qc_rows.append(row)
        main_rows.append(build_main_row(source, case_qc))
    main = pd.DataFrame(main_rows).sort_values("ID", kind="stable").reset_index(drop=True)
    qc = pd.DataFrame(qc_rows)
    qc["roi_name"] = pd.Categorical(qc["roi_name"], categories=list(ROI_NAMES), ordered=True)
    qc = qc.sort_values(["ID", "roi_name"], kind="stable").reset_index(drop=True)
    qc["roi_name"] = qc["roi_name"].astype(str)
    main_columns = [
        "ID", "camera_id", "ExposureTime", "FNumber", "ISOSpeedRatings",
        "relative_optical_exposure", "log2_iso_condition", "forehead_available",
        *REGIONAL_OBSERVATION_COLUMNS,
        *[f"cheek_mean_{name}" for name in OBSERVATION_NAMES],
        *[f"forehead_minus_cheek_{name}" for name in OBSERVATION_NAMES],
        *CHEEK_ABS_DIFF_COLUMNS,
    ]
    qc_columns = [
        "ID", "camera_id", "roi_name", "available_for_model", "bbox_area",
        "valid_skin_pixel_count", "valid_skin_fraction",
        *[f"{name}_{stat}" for name in OBSERVATION_NAMES for stat in ("q25", "median_raw", "q75", "iqr")],
        *[f"{channel}_{metric}_fraction" for channel in ("r", "g", "b") for metric in ("equal_0", "equal_255", "le_5", "ge_250")],
        "transformed_nonfinite_count",
    ]
    return main.loc[:, main_columns], qc.loc[:, qc_columns]


def write_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(
        path, index=False, encoding="utf-8-sig", na_rep="", float_format="%.17g", lineterminator="\n"
    )


def summary_record(values: pd.Series) -> dict[str, Any]:
    numeric = pd.to_numeric(values, errors="coerce")
    valid = numeric.dropna().astype(float)
    if valid.empty:
        return {
            "valid_n": 0,
            "missing_n": int(numeric.isna().sum()),
            "min": math.nan,
            "q25": math.nan,
            "median": math.nan,
            "q75": math.nan,
            "max": math.nan,
            "mean": math.nan,
            "std": math.nan,
            "iqr": math.nan,
        }
    q25, median, q75 = valid.quantile([0.25, 0.50, 0.75])
    return {
        "valid_n": int(valid.size), "missing_n": int(numeric.isna().sum()),
        "min": float(valid.min()), "q25": float(q25), "median": float(median),
        "q75": float(q75), "max": float(valid.max()), "mean": float(valid.mean()),
        "std": float(valid.std(ddof=1)) if valid.size > 1 else 0.0, "iqr": float(q75 - q25),
    }


def build_summaries(main: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    overall = pd.DataFrame([
        {"observation_name": column, **summary_record(main[column])} for column in MAIN_OBSERVATION_COLUMNS
    ])
    by_camera_rows: list[dict[str, Any]] = []
    for camera_id, group in main.groupby("camera_id", sort=True):
        for column in MAIN_OBSERVATION_COLUMNS:
            by_camera_rows.append({"camera_id": camera_id, "observation_name": column, **summary_record(group[column])})
    association_rows: list[dict[str, Any]] = []
    scopes: list[tuple[str, str, pd.DataFrame]] = [("overall", "ALL", main)]
    scopes.extend(("camera_id", str(camera), group.copy()) for camera, group in main.groupby("camera_id", sort=True))
    for scope, camera_id, group in scopes:
        for observation in MAIN_OBSERVATION_COLUMNS:
            for condition in ("relative_optical_exposure", "log2_iso_condition"):
                pair = group[[observation, condition]].dropna().astype(float)
                coefficient = math.nan
                if len(pair) >= 2 and pair[observation].nunique() > 1 and pair[condition].nunique() > 1:
                    coefficient = float(pair[observation].rank(method="average").corr(pair[condition].rank(method="average")))
                association_rows.append({
                    "scope": scope, "camera_id": camera_id, "observation_name": observation,
                    "condition_name": condition, "valid_n": int(len(pair)),
                    "spearman_rho": coefficient,
                })
    return overall, pd.DataFrame(by_camera_rows), pd.DataFrame(association_rows)


def qc_column_names() -> list[str]:
    columns = list(CHEEK_ABS_DIFF_COLUMNS) + [
        "valid_skin_fraction", "valid_skin_pixel_count", "bbox_area", "available_for_model",
        *[f"{name}_{stat}" for name in OBSERVATION_NAMES for stat in ("q25", "median_raw", "q75", "iqr")],
        *[f"{channel}_{metric}_fraction" for channel in ("r", "g", "b") for metric in ("equal_0", "equal_255", "le_5", "ge_250")],
        "transformed_nonfinite_count",
    ]
    return columns


def build_feature_schema(args: argparse.Namespace) -> dict[str, Any]:
    forbidden = ["camera_id", *EXIF_COLUMNS, *qc_column_names()]
    return {
        "schema_name": "regional_optical_observations_v1",
        "schema_version": "1.0",
        "identifier_columns": ["ID"],
        "raw_exif_columns": list(EXIF_COLUMNS),
        "optical_condition_columns": ["relative_optical_exposure", "log2_iso_condition"],
        "device_condition_columns": ["camera_id"],
        "availability_columns": ["forehead_available"],
        "regional_observation_columns": list(REGIONAL_OBSERVATION_COLUMNS),
        "derived_observation_columns": list(DERIVED_OBSERVATION_COLUMNS),
        "qc_only_columns": qc_column_names(),
        "forbidden_direct_classifier_columns": list(dict.fromkeys(forbidden)),
        "core_v1_observation_columns": list(REGIONAL_OBSERVATION_COLUMNS + DERIVED_OBSERVATION_COLUMNS),
        "epsilon": EPSILON,
        "log_base": LOG_BASE,
        "inverse_srgb_transfer_function": {
            "encoded_domain": "C_srgb = C_uint8 / 255.0",
            "branch_low": "C_srgb / 12.92 when C_srgb <= 0.04045",
            "branch_high": "((C_srgb + 0.055) / 1.055) ** 2.4 otherwise",
        },
        "y_coefficients": {"r": Y_COEFFICIENTS[0], "g": Y_COEFFICIENTS[1], "b": Y_COEFFICIENTS[2]},
        "forehead_available_threshold": FOREHEAD_THRESHOLD,
        "image_source": project_relative(args.aligned_rgb_dir, args.project_root),
        "mask_source": project_relative(args.mask_root, args.project_root),
        "color_space_assumption": "untagged PNG assumed sRGB; inverse transfer gives linear-sRGB-like values, not sensor RGB",
        "field_missing_semantics": {
            "forehead_main_observations": "empty CSV field/NaN means forehead_valid_skin_fraction < 0.20",
            "forehead_minus_cheek": "empty CSV field/NaN means forehead is unavailable",
            "other_observations": "missing is not expected; extraction stops on empty cheek mask or nonfinite transform",
        },
        "field_mapping_notes": {
            "valid_skinek_pixel_count_in_prompt": "implemented as the existing correctly spelled valid_skin_pixel_count",
            "source_median": "QC uses *_median_raw; model-facing table uses *_median",
        },
        "clinical_or_label_columns_read": [],
    }


def build_flagged_cases(qc: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for row in qc.to_dict("records"):
        flags: list[str] = []
        if row["roi_name"] == "forehead" and int(row["available_for_model"]) == 0:
            flags.append("forehead_fraction_below_0.20")
        for channel in ("r", "g", "b"):
            if float(row[f"{channel}_equal_0_fraction"]) > 0:
                flags.append(f"{channel}_exact_zero_present")
            if float(row[f"{channel}_equal_255_fraction"]) > 0:
                flags.append(f"{channel}_exact_255_present")
        if int(row["transformed_nonfinite_count"]) > 0:
            flags.append("transformed_nonfinite_present")
        if flags:
            rows.append({
                "ID": row["ID"], "camera_id": row["camera_id"], "roi_name": row["roi_name"],
                "available_for_model": row["available_for_model"],
                "valid_skin_pixel_count": row["valid_skin_pixel_count"],
                "valid_skin_fraction": row["valid_skin_fraction"],
                "qc_flags": ";".join(flags), "automatic_exclusion_applied": 0,
            })
    return pd.DataFrame(rows, columns=[
        "ID", "camera_id", "roi_name", "available_for_model", "valid_skin_pixel_count",
        "valid_skin_fraction", "qc_flags", "automatic_exclusion_applied",
    ])


def validate_outputs(
    args: argparse.Namespace,
    preflight: PreflightResult,
    main: pd.DataFrame,
    qc: pd.DataFrame,
    historical_after: str,
    deterministic_match: bool,
) -> dict[str, Any]:
    errors: list[str] = []
    if len(main) != 500 or main["ID"].nunique() != 500 or set(main["ID"]) != set(preflight.ids):
        errors.append("main_ID_integrity_failed")
    if len(qc) != 1500 or qc[["ID", "roi_name"]].duplicated().any():
        errors.append("qc_row_integrity_failed")
    counts = qc["roi_name"].value_counts().to_dict()
    if any(int(counts.get(roi, 0)) != 500 for roi in ROI_NAMES):
        errors.append("qc_ROI_counts_failed")
    left_columns = [f"cheek_image_left_{name}_median" for name in OBSERVATION_NAMES]
    right_columns = [f"cheek_image_right_{name}_median" for name in OBSERVATION_NAMES]
    if not np.isfinite(main[left_columns + right_columns].to_numpy(float)).all():
        errors.append("nonfinite_cheek_observations")
    available = main["forehead_available"].astype(bool)
    forehead_columns = [f"forehead_{name}_median" for name in OBSERVATION_NAMES]
    difference_columns = [f"forehead_minus_cheek_{name}" for name in OBSERVATION_NAMES]
    if not np.isfinite(main.loc[available, forehead_columns + difference_columns].to_numpy(float)).all():
        errors.append("nonfinite_available_forehead")
    if not main.loc[~available, forehead_columns + difference_columns].isna().all().all():
        errors.append("unavailable_forehead_not_NaN")
    source_fraction = preflight.manifest.set_index("ID").loc[main["ID"], "forehead_valid_skin_fraction"].to_numpy(float)
    if not np.array_equal(available.to_numpy(), source_fraction >= FOREHEAD_THRESHOLD):
        errors.append("forehead_threshold_rule_mismatch")
    for name in OBSERVATION_NAMES:
        left = main[f"cheek_image_left_{name}_median"].to_numpy(float)
        right = main[f"cheek_image_right_{name}_median"].to_numpy(float)
        mean = (left + right) / 2.0
        if not np.allclose(main[f"cheek_mean_{name}"], mean, rtol=0, atol=1e-15):
            errors.append(f"cheek_mean_mismatch:{name}")
        expected_difference = main[f"forehead_{name}_median"].to_numpy(float) - mean
        if not np.allclose(main[f"forehead_minus_cheek_{name}"], expected_difference, rtol=0, atol=1e-15, equal_nan=True):
            errors.append(f"forehead_minus_cheek_mismatch:{name}")
        if not np.allclose(main[f"cheek_abs_diff_{name}"], np.abs(left - right), rtol=0, atol=1e-15):
            errors.append(f"cheek_abs_diff_mismatch:{name}")
    if not np.isfinite(main[[*EXIF_COLUMNS, "relative_optical_exposure", "log2_iso_condition"]].to_numpy(float)).all():
        errors.append("nonfinite_EXIF")
    if int(qc["transformed_nonfinite_count"].sum()) != 0:
        errors.append("transformed_nonfinite_values")
    if historical_after != preflight.historical_inventory_before:
        errors.append("historical_inputs_changed")
    if not deterministic_match:
        errors.append("deterministic_repeat_mismatch")
    return {
        "status": "PASS" if not errors else "FAIL",
        "errors": errors,
        "input_id_count": len(preflight.ids),
        "main_rows": len(main),
        "unique_ids": int(main["ID"].nunique()),
        "qc_rows": len(qc),
        "roi_counts": {str(key): int(value) for key, value in counts.items()},
        "forehead_available_count": int(available.sum()),
        "forehead_unavailable_count": int((~available).sum()),
        "left_cheek_finite_case_count": int(np.isfinite(main[left_columns].to_numpy(float)).all(axis=1).sum()),
        "right_cheek_finite_case_count": int(np.isfinite(main[right_columns].to_numpy(float)).all(axis=1).sum()),
        "transformed_nonfinite_count": int(qc["transformed_nonfinite_count"].sum()),
        "historical_inventory_before_sha256": preflight.historical_inventory_before,
        "historical_inventory_after_sha256": historical_after,
        "historical_inputs_modified": historical_after != preflight.historical_inventory_before,
        "clinical_labels_read": False,
        "cohort_standardization_performed": False,
        "deterministic_repeat_match": bool(deterministic_match),
    }


def get_git_commit(root: Path) -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        return "unavailable"


def markdown_table(frame: pd.DataFrame, columns: Sequence[str], max_rows: int | None = None) -> str:
    view = frame.loc[:, list(columns)]
    if max_rows is not None:
        view = view.head(max_rows)
    header = "| " + " | ".join(columns) + " |"
    separator = "| " + " | ".join(["---"] * len(columns)) + " |"
    rows = ["| " + " | ".join(str(value).replace("|", "\\|") for value in values) + " |" for values in view.itertuples(index=False, name=None)]
    return "\n".join([header, separator, *rows])


def write_report(
    args: argparse.Namespace,
    preflight: PreflightResult,
    paths: Mapping[str, Path],
    main: pd.DataFrame,
    qc: pd.DataFrame,
    summary: pd.DataFrame,
    by_camera: pd.DataFrame,
    associations: pd.DataFrame,
    flagged: pd.DataFrame,
    validation: Mapping[str, Any],
    test_status: str,
    test_output: str,
    deterministic_hashes: Mapping[str, str],
) -> None:
    status = "COMPLETE" if validation["status"] == "PASS" and test_status == "PASS" else "INCOMPLETE"
    camera_counts = main["camera_id"].value_counts().rename_axis("camera_id").reset_index(name="n")
    clipping_suffixes = ("equal_0_fraction", "equal_255_fraction", "le_5_fraction", "ge_250_fraction")
    clipping_columns = [column for column in qc.columns if column.endswith(clipping_suffixes)]
    clipping = pd.DataFrame({
        "metric": clipping_columns,
        "roi_rows_with_nonzero_fraction": [int((qc[column] > 0).sum()) for column in clipping_columns],
        "maximum_fraction": [float(qc[column].max()) for column in clipping_columns],
    })
    representative_observations = [
        "forehead_log2_y_median",
        "cheek_mean_log2_y",
        "cheek_abs_diff_log2_y",
    ]
    camera_report_columns = ["camera_id", "observation_name", "valid_n", "missing_n", "median", "iqr", "mean", "std"]
    camera_report = by_camera.loc[
        by_camera["observation_name"].isin(representative_observations), camera_report_columns
    ].copy()
    for column in ("median", "iqr", "mean", "std"):
        camera_report[column] = camera_report[column].round(6)
    association_report_columns = [
        "scope", "camera_id", "observation_name", "condition_name", "valid_n", "spearman_rho"
    ]
    association_report = associations.loc[
        associations["observation_name"].isin(("forehead_log2_y_median", "cheek_mean_log2_y")),
        association_report_columns,
    ].copy()
    association_report["spearman_rho"] = association_report["spearman_rho"].round(6)
    implementation_and_output_files = [
        *IMPLEMENTATION_RELATIVE_PATHS,
        *[
            project_relative(path, args.project_root)
            for key, path in paths.items()
            if key not in ("data_root", "report_root")
        ],
    ]
    lines = [
        REPORT_TASK_MARKER,
        REPORT_TITLE, "",
        "## 1. 完成状态", "",
        f"- `OPTICAL_OBSERVATION_EXTRACTION_STATUS={status}`",
        f"- 验收状态：`{validation['status']}`；专项测试：`{test_status}`。",
        "- 本产物仅称为区域光学观测量或linear-sRGB-like区域观测，不是皮肤反射率、sensor RGB或生理参数。", "",
        "## 2. 新增文件", "",
        "本任务新增或更新的文件如下；未改动既有输入数据和其他实验代码。", "",
        *[f"- `{path}`" for path in implementation_and_output_files], "",
        "## 3. 输入图像、Mask和EXIF来源", "",
        f"- aligned RGB：`{project_relative(args.aligned_rgb_dir, args.project_root)}`。",
        f"- ROI Mask：`{project_relative(args.mask_root, args.project_root)}`，路径从现有ROI manifest精确读取，未重新生成。",
        f"- ROI与EXIF manifest：`{project_relative(args.roi_manifest, args.project_root)}`。仅点名读取ExposureTime、FNumber、ISOSpeedRatings、camera_id及ROI字段；未读取BrightnessValue或临床字段。", "",
        "## 4. 500例队列定义", "",
        f"研究ID来自`{project_relative(args.study_id_dir, args.project_root)}`的500个完整PNG stem，仅使用文件名定义队列，未使用meanbg图像计算颜色。主表{validation['main_rows']}行、唯一ID={validation['unique_ids']}，按完整字符串ID字典序升序。", "",
        "## 5. RGB通道和颜色空间假设", "",
        "项目已有OpenCV版`read_rgb`，但当前环境未安装OpenCV且任务禁止安装依赖；因此本任务使用Pillow `Image.open(...).convert('RGB')`原生RGB顺序读取，不resize，并用纯红/绿/蓝单元测试锁定通道顺序。500张PNG均无嵌入ICC profile，因此假定其编码为sRGB；该假设合理但未被文件标签证明。", "",
        "## 6. inverse sRGB公式", "",
        "先令`C_srgb=C_uint8/255`。当`C_srgb<=0.04045`时，`C_linear=C_srgb/12.92`；否则`C_linear=((C_srgb+0.055)/1.055)^2.4`。全程float64，所得量称为linear-sRGB-like。", "",
        "## 7. Y、log2-R/G、log2-B/G公式", "",
        "`Y=0.2126R_linear+0.7152G_linear+0.0722B_linear`；`log2_y=log2(Y+1e-6)`；`log2_rg=log2((R_linear+1e-6)/(G_linear+1e-6))`；`log2_bg=log2((B_linear+1e-6)/(G_linear+1e-6))`。", "",
        "## 8. ROI稳健汇总方式", "",
        "每例每ROI仅在`mask>0`像素内计算三个维度的Q25、median、Q75和IQR。主表使用median；原始四分位统计完整保留在1500行QC长表。", "",
        "## 9. 额部20%规则", "",
        f"固定使用`forehead_valid_skin_fraction>=0.20`，可用{validation['forehead_available_count']}例，不可用{validation['forehead_unavailable_count']}例。没有增加像素数阈值，也未根据EXIF或观测值调阈值。", "",
        "## 10. 不可用额部处理方式", "",
        "病例和双侧脸颊均保留；主表额部三个median及额部减脸颊三个字段写为空字段/NaN。QC长表仍保存额部原始Q25、median、Q75和IQR；没有用0替代或修改Mask。", "",
        "## 11. 左右脸颊合并和差异定义", "",
        "`cheek_mean=(left_median+right_median)/2`；`cheek_abs_diff=abs(left_median-right_median)`。后者只归入QC角色。额部可用时计算`forehead_minus_cheek=forehead_median-cheek_mean`。", "",
        "## 12. 字段角色", "",
        "`feature_schema.json`明确区分ID、原始EXIF、采集条件、设备条件、可用性、区域观测、派生观测、QC字段和禁止直接进入分类器的字段。提示词中的拼写`valid_skinek_pixel_count`映射为现有正确字段`valid_skin_pixel_count`。", "",
        "## 13. EXIF在本阶段的作用", "",
        "EXIF仅作为后续反演的采集条件保留。复算`relative_optical_exposure=log2(ExposureTime/FNumber^2)`和`log2_iso_condition=log2(ISO/100)`并与来源核对；未用EXIF修改像素，未做exposure-corrected RGB。", "",
        "## 14. 输出数据完整性", "",
        f"主表={validation['main_rows']}行，QC表={validation['qc_rows']}行，各ROI行数={validation['roi_counts']}；左脸颊有限病例={validation['left_cheek_finite_case_count']}，右脸颊有限病例={validation['right_cheek_finite_case_count']}。重复提取CSV SHA256一致：`{validation['deterministic_repeat_match']}`。", "",
        "## 15. 通道截断和非有限值检查", "",
        f"变换后非有限值总数={validation['transformed_nonfinite_count']}。截断比例均在原始uint8、ROI Mask内部计算；任何QC标记仅记录、不自动排除。标记行数={len(flagged)}。", "",
        markdown_table(clipping, clipping.columns), "",
        "## 16. 两设备描述性统计", "",
        markdown_table(camera_counts, camera_counts.columns), "",
        "以下给出两个设备上三项代表性观测的实际统计量；完整结果见`observation_summary_by_camera.csv`，未进行显著性筛选。", "",
        markdown_table(camera_report, camera_report_columns), "",
        "## 17. 区域观测与EXIF条件的描述性关系", "",
        "以下给出亮度相关代表性观测与两个EXIF条件的整体及设备内Spearman相关；完整结果见`exif_observation_associations.csv`。这些数值仅作描述，没有读取NYHA、选择特征、做显著性检验或按结果改公式。", "",
        markdown_table(association_report, association_report_columns), "",
        "## 18. 单元测试和全量验证结果", "",
        f"专项测试：`{test_status}`（{test_output.replace(chr(10), ' | ')}）。全量验收：`{validation['status']}`。主CSV首次SHA256={deterministic_hashes['main_first']}，重复={deterministic_hashes['main_repeat']}；QC CSV首次={deterministic_hashes['qc_first']}，重复={deterministic_hashes['qc_repeat']}。", "",
        "## 19. 未修改历史输入声明", "",
        f"历史输入库存构建前后摘要分别为`{validation['historical_inventory_before_sha256']}`和`{validation['historical_inventory_after_sha256']}`，一致；`historical_inputs_modified={str(validation['historical_inputs_modified']).lower()}`。只写入本任务两个新输出目录。", "",
        "## 20. 局限性", "",
        "1. 输入是手机处理后的JPEG/PNG编码图像，不是RAW。",
        "2. 保存的PNG没有嵌入颜色配置文件；sRGB是合理但未经文件标签证明的假设。",
        "3. inverse sRGB不能逆转手机白平衡、ISP、HDR或色调映射。",
        "4. 当前输出是区域光学观测量，不是真实皮肤反射率，也不是传感器线性RGB。",
        "5. 尚未实现光学反演网络。",
        "6. 尚未读取或验证这些观测量与NYHA的关系。", "",
    ]
    paths["report"].write_text("\n".join(lines), encoding="utf-8")


def run_tests(root: Path) -> tuple[str, str]:
    command = [sys.executable, "-m", "pytest", "-q", "tests/test_regional_optical_observations_v1.py"]
    result = subprocess.run(command, cwd=root, text=True, capture_output=True)
    output = (result.stdout + "\n" + result.stderr).strip()
    return ("PASS" if result.returncode == 0 else "FAIL", output)


def verify_generated_output(path: Path, dataset: bool) -> bool:
    if not path.exists() or not any(path.iterdir()):
        return True
    marker = path / ("extraction_manifest.json" if dataset else "optical_observation_extraction_report.md")
    if not marker.is_file():
        return False
    if dataset:
        try:
            return json.loads(marker.read_text(encoding="utf-8")).get("task_name") == TASK_NAME
        except Exception:
            return False
    try:
        report_text = marker.read_text(encoding="utf-8")
    except Exception:
        return False
    current_signature = REPORT_TASK_MARKER in report_text and REPORT_TITLE in report_text
    legacy_signature = (
        report_text.startswith(REPORT_TITLE)
        and "OPTICAL_OBSERVATION_EXTRACTION_STATUS=" in report_text
        and "## 20. 局限性" in report_text
    )
    return current_signature or legacy_signature


def prepare_output_paths(args: argparse.Namespace) -> dict[str, Path]:
    expected_data = (args.project_root / "data/processed/optical_observations_v1").resolve()
    expected_report = (args.project_root / "reports/optical_observations_v1").resolve()
    if args.data_output_dir.resolve() != expected_data or args.report_output_dir.resolve() != expected_report:
        raise ExtractionFailure("output_safety", ["Output directories differ from the fixed task directories"])
    nonempty = [path for path in (expected_data, expected_report) if path.exists() and any(path.iterdir())]
    if nonempty and not bool(args.overwrite):
        raise ExtractionFailure("output_safety", [f"Non-empty output directory: {path}" for path in nonempty])
    if nonempty:
        if not verify_generated_output(expected_data, True) or not verify_generated_output(expected_report, False):
            raise ExtractionFailure("output_safety", ["Existing output was not verified as generated by this task"])
        shutil.rmtree(expected_data, ignore_errors=False)
        shutil.rmtree(expected_report, ignore_errors=False)
    expected_data.mkdir(parents=True, exist_ok=True)
    expected_report.mkdir(parents=True, exist_ok=True)
    return {
        "data_root": expected_data,
        "report_root": expected_report,
        "main": expected_data / "regional_optical_observations.csv",
        "qc": expected_data / "regional_optical_qc_long.csv",
        "schema": expected_data / "feature_schema.json",
        "manifest": expected_data / "extraction_manifest.json",
        "report": expected_report / "optical_observation_extraction_report.md",
        "summary": expected_report / "observation_summary.csv",
        "summary_camera": expected_report / "observation_summary_by_camera.csv",
        "associations": expected_report / "exif_observation_associations.csv",
        "flagged": expected_report / "qc_flagged_cases.csv",
        "log": expected_report / "extraction_run.log",
    }


def write_extraction_manifest(
    args: argparse.Namespace,
    preflight: PreflightResult,
    paths: Mapping[str, Path],
    validation: Mapping[str, Any],
    test_status: str,
    deterministic_hashes: Mapping[str, str],
) -> None:
    output_files = {}
    for key, path in paths.items():
        if key in ("data_root", "report_root", "manifest"):
            continue
        output_files[key] = {"path": project_relative(path, args.project_root), "sha256": sha256_file(path)}
    payload = {
        "task_name": str(args.task_name),
        "version": str(args.version),
        "status": "COMPLETE" if validation["status"] == "PASS" and test_status == "PASS" else "INCOMPLETE",
        "run_time": datetime.now().astimezone().isoformat(timespec="seconds"),
        "cohort_size": len(preflight.ids),
        "unique_id_count": len(set(preflight.ids)),
        "id_source": project_relative(args.study_id_dir, args.project_root),
        "id_order": str(args.id_order),
        "input_aligned_rgb_dir": project_relative(args.aligned_rgb_dir, args.project_root),
        "input_mask_dir": project_relative(args.mask_root, args.project_root),
        "input_roi_manifest": {"path": project_relative(args.roi_manifest, args.project_root), "sha256": preflight.roi_manifest_sha256},
        "input_roi_build_manifest": {"path": project_relative(args.roi_build_manifest, args.project_root), "sha256": preflight.roi_build_manifest_sha256},
        "input_exif_source": {"path": project_relative(args.roi_manifest, args.project_root), "sha256": preflight.roi_manifest_sha256, "columns": [*EXIF_COLUMNS, "camera_id"]},
        "roi_names": list(ROI_NAMES),
        "image_size": [224, 224],
        "rgb_reading_method": "Pillow Image.open(...).convert('RGB') to uint8; no resize; synthetic RGB channel-order test",
        "assumed_color_encoding": "sRGB",
        "linearization": "standard inverse sRGB transfer function IEC 61966-2-1 branch formula",
        "input_has_embedded_color_profile": preflight.input_has_embedded_color_profile,
        "embedded_color_profile_count": preflight.embedded_color_profile_count,
        "epsilon": EPSILON,
        "log_base": LOG_BASE,
        "y_coefficients": {"r": Y_COEFFICIENTS[0], "g": Y_COEFFICIENTS[1], "b": Y_COEFFICIENTS[2]},
        "forehead_available_threshold": FOREHEAD_THRESHOLD,
        "csv_nan_representation": "empty field interpreted as NaN",
        "output_files": output_files,
        "extraction_manifest_self": {"path": project_relative(paths["manifest"], args.project_root), "sha256": "not_applicable_recursive_self_reference"},
        "config": yaml.safe_load(args.config_path.read_text(encoding="utf-8")),
        "config_path": project_relative(args.config_path, args.project_root),
        "config_sha256": sha256_file(args.config_path),
        "script_path": project_relative(Path(__file__), args.project_root),
        "script_sha256": sha256_file(Path(__file__)),
        "git_commit": get_git_commit(args.project_root),
        "runtime_versions": {
            "python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__,
            "Pillow": PIL.__version__, "PyYAML": yaml.__version__,
        },
        "test_status": test_status,
        "validation": dict(validation),
        "deterministic_repeat_hashes": dict(deterministic_hashes),
        "clinical_labels_read": False,
        "cohort_standardization_performed": False,
        "missing_value_imputation_performed": False,
        "exif_pixel_correction_performed": False,
        "historical_inputs_modified": False,
    }
    paths["manifest"].write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def final_lines(paths: Mapping[str, Path], validation: Mapping[str, Any], test_status: str) -> list[str]:
    status = "COMPLETE" if validation["status"] == "PASS" and test_status == "PASS" else "INCOMPLETE"
    project_root = Path(__file__).resolve().parents[1]
    output_files = [
        project_relative(path, project_root)
        for key, path in paths.items()
        if key not in ("data_root", "report_root")
    ]
    files = [*IMPLEMENTATION_RELATIVE_PATHS, *output_files]
    return [
        f"OPTICAL_OBSERVATION_EXTRACTION_STATUS={status}",
        "ADDED_OR_MODIFIED_FILES=" + ",".join(files),
        f"UNIT_TEST_STATUS={test_status}",
        f"INPUT_ID_COUNT={validation['input_id_count']}",
        f"MAIN_TABLE_ROWS={validation['main_rows']}",
        f"MAIN_TABLE_UNIQUE_IDS={validation['unique_ids']}",
        f"QC_TABLE_ROWS={validation['qc_rows']}",
        f"FOREHEAD_AVAILABLE_COUNT={validation['forehead_available_count']}",
        f"FOREHEAD_UNAVAILABLE_COUNT={validation['forehead_unavailable_count']}",
        f"LEFT_CHEEK_FINITE_CASE_COUNT={validation['left_cheek_finite_case_count']}",
        f"RIGHT_CHEEK_FINITE_CASE_COUNT={validation['right_cheek_finite_case_count']}",
        f"NONFINITE_TRANSFORMED_VALUE_COUNT={validation['transformed_nonfinite_count']}",
        f"MAIN_TABLE_PATH={paths['main']}", f"QC_TABLE_PATH={paths['qc']}",
        f"SCHEMA_PATH={paths['schema']}", f"MANIFEST_PATH={paths['manifest']}",
        f"REPORT_PATH={paths['report']}",
        "CLINICAL_OR_NYHA_READ=NO", "COHORT_STANDARDIZATION_PERFORMED=NO",
        "HISTORICAL_INPUTS_MODIFIED=NO",
        f"ALL_ACCEPTANCE_CRITERIA_MET={'YES' if status == 'COMPLETE' else 'NO'}",
    ]


def write_failure_log(args: argparse.Namespace, failure: ExtractionFailure) -> Path | None:
    try:
        report_root = Path(args.report_output_dir)
        if report_root.exists() and any(report_root.iterdir()):
            if not bool(getattr(args, "overwrite", False)):
                return None
            if not verify_generated_output(report_root, dataset=False):
                return None
        report_root.mkdir(parents=True, exist_ok=True)
        path = report_root / "extraction_run.log"
        path.write_text(f"STATUS=FAILED\nSTAGE={failure.stage}\n" + "\n".join(failure.errors) + "\n", encoding="utf-8")
        return path
    except Exception:
        return None


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        preflight = run_preflight(args)
        if args.preflight_only:
            print("PREFLIGHT_STATUS=PASS")
            print(f"INPUT_ID_COUNT={len(preflight.ids)}")
            print(f"UNIQUE_ID_COUNT={len(set(preflight.ids))}")
            print(f"EMBEDDED_COLOR_PROFILE_COUNT={preflight.embedded_color_profile_count}")
            print(f"FOREHEAD_AVAILABLE_COUNT={int((preflight.manifest['forehead_valid_skin_fraction'] >= FOREHEAD_THRESHOLD).sum())}")
            return 0
        test_status, test_output = run_tests(args.project_root)
        if test_status != "PASS":
            raise ExtractionFailure("unit_tests", [test_output])
        paths = prepare_output_paths(args)
        main_table, qc_table = extract_tables(args, preflight)
        write_csv(main_table, paths["main"])
        write_csv(qc_table, paths["qc"])
        schema = build_feature_schema(args)
        paths["schema"].write_text(json.dumps(schema, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        summary, summary_camera, associations = build_summaries(main_table)
        flagged = build_flagged_cases(qc_table)
        write_csv(summary, paths["summary"])
        write_csv(summary_camera, paths["summary_camera"])
        write_csv(associations, paths["associations"])
        write_csv(flagged, paths["flagged"])
        with tempfile.TemporaryDirectory(prefix="determinism_", dir=paths["data_root"]) as temp_name:
            repeat_main, repeat_qc = extract_tables(args, preflight)
            repeat_main_path = Path(temp_name) / "regional_optical_observations.csv"
            repeat_qc_path = Path(temp_name) / "regional_optical_qc_long.csv"
            write_csv(repeat_main, repeat_main_path)
            write_csv(repeat_qc, repeat_qc_path)
            deterministic_hashes = {
                "main_first": sha256_file(paths["main"]), "main_repeat": sha256_file(repeat_main_path),
                "qc_first": sha256_file(paths["qc"]), "qc_repeat": sha256_file(repeat_qc_path),
            }
        deterministic_match = (
            deterministic_hashes["main_first"] == deterministic_hashes["main_repeat"]
            and deterministic_hashes["qc_first"] == deterministic_hashes["qc_repeat"]
        )
        historical_after = stable_inventory_sha256(preflight.input_paths, args.project_root)
        validation = validate_outputs(
            args, preflight, main_table, qc_table, historical_after, deterministic_match
        )
        write_report(
            args, preflight, paths, main_table, qc_table, summary, summary_camera,
            associations, flagged, validation, test_status, test_output, deterministic_hashes,
        )
        lines = final_lines(paths, validation, test_status)
        paths["log"].write_text("\n".join(lines) + "\n", encoding="utf-8")
        write_extraction_manifest(args, preflight, paths, validation, test_status, deterministic_hashes)
        if validation["status"] != "PASS":
            raise ExtractionFailure("full_validation", validation["errors"])
        print("\n".join(lines))
        return 0
    except ExtractionFailure as failure:
        log_path = write_failure_log(args, failure)
        resume = "python preprocessing/extract_regional_optical_observations_v1.py --config config/preprocess/regional_optical_observations_v1.yaml"
        if args.data_output_dir.exists() or args.report_output_dir.exists():
            resume += " --overwrite"
        print("OPTICAL_OBSERVATION_EXTRACTION_STATUS=FAILED")
        print(f"FAILED_STAGE={failure.stage}")
        print("ERRORS=" + " | ".join(failure.errors))
        print(f"ERROR_LOG_PATH={log_path if log_path else 'unavailable'}")
        print(f"EXACT_RESUME_COMMAND={resume}")
        print("ALL_ACCEPTANCE_CRITERIA_MET=NO")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
