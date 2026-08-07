"""P1 data interface and frozen component statistics audit."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
import yaml
from PIL import Image, ImageDraw


REQUIRED_LATENT_KEYS = {
    "shape": "shape_code",
    "tex": "tex_code",
    "detail": "detail_code",
    "exp": "expression_code",
    "pose": "pose_code",
    "cam": "camera_code",
    "light": "light_code",
}

REQUIRED_MAP_KEYS = (
    "input_aligned_rgb",
    "reconstruction",
    "albedo_like",
    "normal_coarse",
    "shading_like",
    "signed_residual",
    "absolute_residual",
)

IMAGE_COMPONENT_TO_MAP_KEY = {
    "rgb": "input_aligned_rgb",
    "reconstruction": "reconstruction",
    "albedo_like": "albedo_like",
    "normal_coarse": "normal_coarse",
    "shading_like": "shading_like",
    "signed_residual": "signed_residual",
    "absolute_residual": "absolute_residual",
}


@dataclass(frozen=True)
class AuditConfig:
    root: Path
    p0_root: Path
    p0_master_index: Path
    p0_exif_index: Path
    split_csv: Path
    frozen_deca_root: Path
    frozen_contract: Path
    output_root: Path
    report_path: Path
    required_case_files: tuple[str, ...]
    image_components: tuple[str, ...]
    regions: tuple[str, ...]
    qc_iqr_multiplier: float
    montage_case_count: int
    raw: dict[str, Any]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_audit_config(config_path: Path) -> AuditConfig:
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    project_root = raw.get("project_root", ".")
    root = config_path.resolve().parents[2] if project_root == "." else (config_path.parent / project_root).resolve()

    def resolve(key: str) -> Path:
        return (root / raw[key]).resolve()

    return AuditConfig(
        root=root,
        p0_root=resolve("p0_unified_root"),
        p0_master_index=resolve("p0_master_index"),
        p0_exif_index=resolve("p0_exif_index"),
        split_csv=resolve("fixed_split_csv"),
        frozen_deca_root=resolve("frozen_deca_root"),
        frozen_contract=resolve("frozen_contract"),
        output_root=resolve("output_root"),
        report_path=resolve("report_path"),
        required_case_files=tuple(raw["required_case_files"]),
        image_components=tuple(raw["image_components"]),
        regions=tuple(raw["regions"]),
        qc_iqr_multiplier=float(raw.get("qc_iqr_multiplier", 3.0)),
        montage_case_count=int(raw.get("montage_case_count", 12)),
        raw=raw,
    )


def ensure_output_dirs(out: Path) -> None:
    for rel in (
        "manifests",
        "schema",
        "statistics",
        "acquisition",
        "qc/random_case_montages",
        "qc/flagged_case_montages",
        "metadata",
        "logs",
    ):
        (out / rel).mkdir(parents=True, exist_ok=True)


def _read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, dtype={"ID": str, "patient_group_id": str})


def _clean_str(value: Any) -> str:
    if pd.isna(value):
        return ""
    return str(value)


def _numeric(value: Any) -> float:
    val = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    if pd.isna(val):
        return math.nan
    return float(val)


def _log_positive(value: Any) -> float:
    val = _numeric(value)
    if not np.isfinite(val) or val <= 0:
        return math.nan
    return float(math.log(val))


def _shooting_period(datetime_original: Any) -> str:
    text = _clean_str(datetime_original)
    if not text:
        return ""
    for fmt in ("%Y:%m:%d %H:%M:%S", "%Y-%m-%d %H:%M:%S"):
        try:
            hour = datetime.strptime(text, fmt).hour
            if 5 <= hour < 12:
                return "morning"
            if 12 <= hour < 18:
                return "afternoon"
            if 18 <= hour < 22:
                return "evening"
            return "night"
        except ValueError:
            continue
    return ""


def _rel(path: Path, root: Path) -> str:
    try:
        return str(path.relative_to(root))
    except ValueError:
        return str(path)


def _path_from_rel(root: Path, relpath: Any) -> Path:
    return (root / _clean_str(relpath)).resolve()


def _nan_json(value: Any) -> Any:
    if isinstance(value, float) and not np.isfinite(value):
        return None
    if isinstance(value, dict):
        return {k: _nan_json(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_nan_json(v) for v in value]
    if isinstance(value, (np.integer, np.floating)):
        return _nan_json(value.item())
    return value


def _stats(values: np.ndarray) -> dict[str, float]:
    arr = np.asarray(values, dtype=np.float64).reshape(-1)
    if arr.size == 0:
        return {k: math.nan for k in ("min", "max", "mean", "std", "median", "p01", "p05", "p95", "p99")}
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return {k: math.nan for k in ("min", "max", "mean", "std", "median", "p01", "p05", "p95", "p99")}
    return {
        "min": float(np.min(finite)),
        "max": float(np.max(finite)),
        "mean": float(np.mean(finite)),
        "std": float(np.std(finite)),
        "median": float(np.median(finite)),
        "p01": float(np.percentile(finite, 1)),
        "p05": float(np.percentile(finite, 5)),
        "p95": float(np.percentile(finite, 95)),
        "p99": float(np.percentile(finite, 99)),
    }


def _robust(values: Iterable[Any]) -> dict[str, float]:
    arr = pd.to_numeric(pd.Series(list(values)), errors="coerce").dropna().to_numpy(dtype=np.float64)
    if arr.size == 0:
        return {k: math.nan for k in ("count", "mean", "std", "median", "IQR", "p01", "p05", "p95", "p99", "minimum", "maximum")}
    q25, q75 = np.percentile(arr, [25, 75])
    return {
        "count": int(arr.size),
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr)),
        "median": float(np.median(arr)),
        "IQR": float(q75 - q25),
        "p01": float(np.percentile(arr, 1)),
        "p05": float(np.percentile(arr, 5)),
        "p95": float(np.percentile(arr, 95)),
        "p99": float(np.percentile(arr, 99)),
        "minimum": float(np.min(arr)),
        "maximum": float(np.max(arr)),
    }


def _iqr_outliers(series: pd.Series, multiplier: float) -> set[str]:
    values = pd.to_numeric(series, errors="coerce")
    valid = values.dropna()
    if valid.empty:
        return set()
    q1, q3 = valid.quantile([0.25, 0.75])
    iqr = q3 - q1
    if not np.isfinite(iqr) or iqr == 0:
        return set()
    low, high = q1 - multiplier * iqr, q3 + multiplier * iqr
    return set(values[(values < low) | (values > high)].index.astype(str))


def _load_mask(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        return np.asarray(image.convert("L"), dtype=np.uint8) > 0


def _load_rgb(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        return np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0


def _build_manifest(cfg: AuditConfig) -> tuple[pd.DataFrame, dict[str, Any]]:
    p0 = _read_csv(cfg.p0_master_index)
    split = _read_csv(cfg.split_csv)
    p0["case_id"] = p0["ID"].astype(str)
    split["case_id"] = split["ID"].astype(str)
    join_rows: list[dict[str, Any]] = []

    def audit_pair(stage: str, left: pd.Series, right: pd.Series) -> None:
        left_ids, right_ids = set(left), set(right)
        for name, ids in (
            ("left_only", sorted(left_ids - right_ids)),
            ("right_only", sorted(right_ids - left_ids)),
            ("duplicate_left", sorted(left[left.duplicated()].unique())),
            ("duplicate_right", sorted(right[right.duplicated()].unique())),
            ("invalid_id_left", sorted(x for x in left_ids if not str(x).strip())),
            ("invalid_id_right", sorted(x for x in right_ids if not str(x).strip())),
        ):
            join_rows.append({"stage": stage, "issue": name, "count": len(ids), "ids": ";".join(map(str, ids[:50]))})

    audit_pair("p0_master_vs_fixed_split", p0["case_id"], split["case_id"])
    merged = p0.merge(split.add_prefix("split_"), left_on="case_id", right_on="split_case_id", how="outer", indicator=True)
    deca_case_ids = pd.Series([p.name for p in (cfg.frozen_deca_root / "cases").iterdir() if p.is_dir()], dtype=str)
    audit_pair("p0_master_vs_frozen_deca_cases", p0["case_id"], deca_case_ids)

    split_sha = sha256_file(cfg.split_csv)
    contract_sha = sha256_file(cfg.frozen_contract)
    p0_master_sha = sha256_file(cfg.p0_master_index)
    out_rows: list[dict[str, Any]] = []
    missing_required_assets = 0
    hash_mismatches = 0
    input_hash_mismatches = 0
    label_mismatches = 0
    fold_mismatches = 0
    rgb_input_match_count = 0

    for _, row in merged[merged["_merge"] == "both"].sort_values("case_id").iterrows():
        case_id = str(row["case_id"])
        case_dir = cfg.frozen_deca_root / "cases" / case_id
        p0_rgb = _path_from_rel(cfg.p0_root, row["aligned_scene_relpath"])
        final_face = _path_from_rel(cfg.p0_root, row["final_face_mask_relpath"])
        face_valid = _path_from_rel(cfg.p0_root, row["face_valid_mask_relpath"])
        skin_strict = _path_from_rel(cfg.p0_root, row["skin_strict_mask_relpath"])
        physics_core = _path_from_rel(cfg.p0_root, row["physics_core_skin_mask_relpath"])
        required_paths = {name: (case_dir / name).resolve() for name in cfg.required_case_files}
        missing = [name for name, path in required_paths.items() if not path.is_file()]
        missing_required_assets += len(missing)
        success_data: dict[str, Any] = {}
        if required_paths["_SUCCESS.json"].is_file():
            success_data = json.loads(required_paths["_SUCCESS.json"].read_text(encoding="utf-8"))
            for name, expected_hash in success_data.get("output_file_sha256", {}).items():
                path = case_dir / name
                if not path.is_file() or sha256_file(path) != expected_hash:
                    hash_mismatches += 1
        input_sha = sha256_file(p0_rgb) if p0_rgb.is_file() else ""
        if success_data.get("input_sha256") and success_data.get("input_sha256") != input_sha:
            input_hash_mismatches += 1
        else:
            rgb_input_match_count += 1
        label_original = int(row["NYHA"])
        label_binary = 0 if label_original == 0 else 1
        if label_binary != int(row["binary_label"]) or label_binary != int(row["split_binary_label"]):
            label_mismatches += 1
        if int(row["fold"]) != int(row["split_fold"]):
            fold_mismatches += 1
        exp_seconds = _numeric(row.get("exposure_time_s"))
        iso_numeric = _numeric(row.get("iso"))
        fnumber = _numeric(row.get("f_number"))
        brightness = _numeric(row.get("brightness_value"))
        out_rows.append({
            "case_id": case_id,
            "patient_id": str(row["patient_group_id"]),
            "group_id": str(row["patient_group_id"]),
            "fold": int(row["split_fold"]),
            "split_source_path": _rel(cfg.split_csv, cfg.root),
            "split_source_sha256": split_sha,
            "label_original": label_original,
            "label_3class": int(row["label_3class"]),
            "label_binary": label_binary,
            "p0_master_index_path": _rel(cfg.p0_master_index, cfg.root),
            "p0_master_index_sha256": p0_master_sha,
            "p0_quality_status": _clean_str(row.get("overall_status") or row.get("core_status")),
            "rgb_path": str(p0_rgb),
            "final_face_mask_path": str(final_face),
            "face_valid_mask_path": str(face_valid),
            "skin_strict_mask_path": str(skin_strict),
            "physics_core_skin_mask_path": str(physics_core),
            "latents_path": str(required_paths["latents.npz"]),
            "maps_path": str(required_paths["maps.npz"]),
            "relighting_path": str(required_paths["relighting.npz"]),
            "quality_path": str(required_paths["quality.json"]),
            "provenance_path": str(required_paths["provenance.json"]),
            "success_record_path": str(required_paths["_SUCCESS.json"]),
            "input_sha256": input_sha,
            "success_record_sha256": sha256_file(required_paths["_SUCCESS.json"]) if required_paths["_SUCCESS.json"].is_file() else "",
            "frozen_contract_sha256": contract_sha,
            "camera_model": _clean_str(row.get("camera_model")),
            "exposure_time_raw": _clean_str(row.get("exposure_time_s")),
            "exposure_time_seconds": exp_seconds,
            "log_exposure_time": _log_positive(exp_seconds),
            "fnumber": fnumber,
            "iso_raw": _clean_str(row.get("iso")),
            "iso_numeric": iso_numeric,
            "log_iso": _log_positive(iso_numeric),
            "brightness_value": brightness,
            "datetime_original": _clean_str(row.get("datetime_original")),
            "shooting_time_period": _shooting_period(row.get("datetime_original")),
            "has_rgb": p0_rgb.is_file(),
            "has_albedo_like": required_paths["maps.npz"].is_file(),
            "has_normal_coarse": required_paths["maps.npz"].is_file(),
            "has_light_code": required_paths["latents.npz"].is_file(),
            "has_shading_like": required_paths["maps.npz"].is_file(),
            "has_signed_residual": required_paths["maps.npz"].is_file(),
            "has_absolute_residual": required_paths["maps.npz"].is_file(),
            "has_relighting": required_paths["relighting.npz"].is_file(),
            "has_specular_like": False,
            "frozen_asset_dir": str(case_dir.resolve()),
            "deca_validation_passed": bool(success_data.get("validation_passed", False)),
        })

    asset_case_ids = set(deca_case_ids.astype(str))
    unmatched_frozen = sorted(set(p0["case_id"]) ^ asset_case_ids)
    asset_join_summary = {
        "row_count": len(out_rows),
        "unique_case_id": pd.Series([r["case_id"] for r in out_rows]).nunique(),
        "duplicate_case_id": len(out_rows) - pd.Series([r["case_id"] for r in out_rows]).nunique(),
        "unmatched_p0_record": int((merged["_merge"] == "right_only").sum()),
        "unmatched_split_record": int((merged["_merge"] == "left_only").sum()),
        "unmatched_frozen_asset": len(unmatched_frozen),
        "missing_required_asset": int(missing_required_assets),
        "asset_hash_mismatch": int(hash_mismatches),
        "input_hash_mismatch": int(input_hash_mismatches),
        "label_mapping_mismatch": int(label_mismatches),
        "fold_mismatch": int(fold_mismatches),
        "rgb_input_hash_match_count": int(rgb_input_match_count),
    }
    pd.DataFrame(join_rows).to_csv(cfg.output_root / "manifests/p1_asset_join_audit.csv", index=False)
    manifest = pd.DataFrame(out_rows)
    manifest.to_csv(cfg.output_root / "manifests/p1_master_manifest.csv", index=False)
    availability_cols = [
        "case_id",
        "has_rgb",
        "has_albedo_like",
        "has_normal_coarse",
        "has_light_code",
        "has_shading_like",
        "has_signed_residual",
        "has_absolute_residual",
        "has_relighting",
        "has_specular_like",
    ]
    manifest[availability_cols].to_csv(cfg.output_root / "manifests/p1_representation_availability.csv", index=False)
    return manifest, asset_join_summary


def _audit_splits(cfg: AuditConfig, manifest: pd.DataFrame) -> dict[str, Any]:
    split = _read_csv(cfg.split_csv)
    original_columns = list(split.columns)
    split["case_id"] = split["ID"].astype(str)
    fold_sets = split.groupby("patient_group_id")["fold"].nunique()
    summary = {
        "split_source": _rel(cfg.split_csv, cfg.root),
        "split_source_sha256": sha256_file(cfg.split_csv),
        "row_count": int(len(split)),
        "unique_case_id": int(split["case_id"].nunique()),
        "duplicate_case_id": int(split["case_id"].duplicated().sum()),
        "patient_or_group_cross_fold_count": int((fold_sets > 1).sum()),
        "fold_values": sorted(pd.to_numeric(split["fold"], errors="coerce").dropna().astype(int).unique().tolist()),
        "columns": original_columns,
    }
    schema = {col: str(dtype) for col, dtype in split.dtypes.items()}
    (cfg.output_root / "schema/split_schema.json").write_text(json.dumps({"columns": schema, "summary": summary}, indent=2), encoding="utf-8")
    rows = []
    for fold, group in split.groupby("fold", dropna=False):
        row: dict[str, Any] = {
            "fold": int(fold),
            "total_cases": int(len(group)),
            "control_count": int((pd.to_numeric(group["binary_label"], errors="coerce") == 0).sum()),
            "patient_count": int((pd.to_numeric(group["binary_label"], errors="coerce") == 1).sum()),
            "patient_or_group_count": int(group["patient_group_id"].nunique()),
        }
        for label, count in group["NYHA"].astype(str).value_counts().sort_index().items():
            row[f"label_original_{label}_count"] = int(count)
        for label, count in group["label_3class"].astype(str).value_counts().sort_index().items():
            row[f"label_3class_{label}_count"] = int(count)
        for sex, count in group["sex_name"].astype(str).value_counts().sort_index().items():
            row[f"sex_{sex}_count"] = int(count)
        rows.append(row)
    dist = pd.DataFrame(rows).fillna(0)
    dist.to_csv(cfg.output_root / "statistics/fold_label_distribution.csv", index=False)
    return summary


def _region_masks(row: pd.Series, shape: tuple[int, int]) -> dict[str, np.ndarray]:
    h, w = shape
    return {
        "full_image": np.ones((h, w), dtype=bool),
        "face_valid": _load_mask(Path(row["face_valid_mask_path"])),
        "skin_strict": _load_mask(Path(row["skin_strict_mask_path"])),
        "physics_core_skin": _load_mask(Path(row["physics_core_skin_mask_path"])),
    }


def _component_stats_for_values(values: np.ndarray, finite_total: int) -> dict[str, Any]:
    arr = np.asarray(values)
    flat = arr.reshape(-1)
    finite_fraction = float(np.isfinite(flat).sum() / finite_total) if finite_total else math.nan
    finite = flat[np.isfinite(flat)]
    stats = _stats(finite)
    zero_fraction = float((finite == 0).sum() / finite.size) if finite.size else math.nan
    negative_fraction = float((finite < 0).sum() / finite.size) if finite.size else math.nan
    return {
        "finite_fraction": finite_fraction,
        **stats,
        "zero_fraction": zero_fraction,
        "negative_fraction": negative_fraction,
    }


def _audit_components(cfg: AuditConfig, manifest: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    case_stat_rows: list[dict[str, Any]] = []
    case_metric_rows: list[dict[str, Any]] = []
    latent_case_rows: list[dict[str, Any]] = []
    relighting_rows: list[dict[str, Any]] = []
    latent_values: dict[str, list[np.ndarray]] = {name: [] for name in REQUIRED_LATENT_KEYS}
    shape_signatures: dict[str, set[str]] = {name: set() for name in cfg.image_components}
    dtype_signatures: dict[str, set[str]] = {name: set() for name in cfg.image_components}
    nonfinite_cases: set[str] = set()
    relighting_incomplete: set[str] = set()

    for _, row in manifest.iterrows():
        case_id = str(row["case_id"])
        with np.load(row["maps_path"], allow_pickle=False) as maps, np.load(row["latents_path"], allow_pickle=False) as latents:
            p0_rgb = _load_rgb(Path(row["rgb_path"]))
            component_arrays = {"rgb": p0_rgb}
            for component, key in IMAGE_COMPONENT_TO_MAP_KEY.items():
                if component != "rgb":
                    component_arrays[component] = np.asarray(maps[key], dtype=np.float32)
            masks = _region_masks(row, p0_rgb.shape[:2])
            metric_row: dict[str, Any] = {
                "case_id": case_id,
                "fold": int(row["fold"]),
                "label_binary": int(row["label_binary"]),
                "label_original": int(row["label_original"]),
                "label_3class": int(row["label_3class"]),
                "physics_core_coverage": float(masks["physics_core_skin"].mean()),
            }
            for component in cfg.image_components:
                array = component_arrays[component]
                shape_signatures[component].add(str(tuple(array.shape)))
                dtype_signatures[component].add(str(array.dtype))
                if not np.isfinite(array).all():
                    nonfinite_cases.add(case_id)
                for region, mask in masks.items():
                    for channel in range(array.shape[2]):
                        values = array[:, :, channel][mask]
                        stats = _component_stats_for_values(values, int(mask.sum()))
                        case_stat_rows.append({
                            "case_id": case_id,
                            "component": component,
                            "region": region,
                            "channel": channel,
                            "shape": str(tuple(array.shape)),
                            "dtype": str(array.dtype),
                            **stats,
                        })
                core = masks["physics_core_skin"]
                face = masks["face_valid"]
                if component == "albedo_like":
                    core_values = array[core]
                    metric_row["albedo_physics_core_mean"] = float(np.mean(core_values)) if core_values.size else math.nan
                    metric_row["albedo_physics_core_std"] = float(np.std(core_values)) if core_values.size else math.nan
                    metric_row["albedo_dynamic_range"] = float(np.nanmax(array) - np.nanmin(array))
                    metric_row["albedo_near_zero_fraction"] = float((core_values < 1e-4).mean()) if core_values.size else math.nan
                    metric_row["albedo_high_value_fraction"] = float((core_values > 1.0).mean()) if core_values.size else math.nan
                    channel_means = np.mean(array[core], axis=0) if core_values.size else np.array([math.nan] * 3)
                    metric_row["albedo_channel_balance_range"] = float(np.nanmax(channel_means) - np.nanmin(channel_means))
                elif component == "normal_coarse":
                    norm = np.linalg.norm(array, axis=2)
                    core_norm = norm[core]
                    metric_row["normal_physics_core_norm_mean"] = float(np.mean(core_norm)) if core_norm.size else math.nan
                    metric_row["normal_physics_core_norm_std"] = float(np.std(core_norm)) if core_norm.size else math.nan
                    metric_row["normal_near_zero_fraction"] = float((core_norm < 1e-4).mean()) if core_norm.size else math.nan
                    metric_row["normal_out_of_expected_range_fraction"] = float(((array < -1.05) | (array > 1.05)).mean())
                    dirs = array[core]
                    if dirs.size:
                        mean_dir = np.mean(dirs, axis=0)
                        metric_row["normal_physics_core_mean_direction_x"] = float(mean_dir[0])
                        metric_row["normal_physics_core_mean_direction_y"] = float(mean_dir[1])
                        metric_row["normal_physics_core_mean_direction_z"] = float(mean_dir[2])
                        metric_row["normal_physics_core_angular_dispersion"] = float(np.std(np.linalg.norm(dirs - mean_dir, axis=1)))
                elif component == "shading_like":
                    core_values = array[core]
                    metric_row["shading_physics_core_mean"] = float(np.mean(core_values)) if core_values.size else math.nan
                    metric_row["shading_physics_core_std"] = float(np.std(core_values)) if core_values.size else math.nan
                    metric_row["shading_dynamic_range"] = float(np.nanmax(array) - np.nanmin(array))
                    metric_row["shading_negative_fraction"] = float((core_values < 0).mean()) if core_values.size else math.nan
                    metric_row["shading_high_value_fraction"] = float((core_values > 1.0).mean()) if core_values.size else math.nan
                    metric_row["shading_physics_core_contrast"] = float(np.percentile(core_values, 95) - np.percentile(core_values, 5)) if core_values.size else math.nan
                elif component == "signed_residual":
                    for region_name, mask in (("full", np.ones_like(face)), ("face", face), ("physics_core", core)):
                        vals = array[mask]
                        if vals.size:
                            metric_row[f"residual_{region_name}_mae"] = float(np.mean(np.abs(vals)))
                            metric_row[f"residual_{region_name}_rmse"] = float(np.sqrt(np.mean(vals ** 2)))
                            metric_row[f"residual_{region_name}_p95_abs"] = float(np.percentile(np.abs(vals), 95))
                            metric_row[f"residual_{region_name}_p99_abs"] = float(np.percentile(np.abs(vals), 99))
                            metric_row[f"residual_{region_name}_positive_fraction"] = float((vals > 0).mean())
                            metric_row[f"residual_{region_name}_negative_fraction"] = float((vals < 0).mean())
                    channel_bias = np.mean(array[core], axis=0) if core.any() else np.array([math.nan] * 3)
                    for idx, value in enumerate(channel_bias):
                        metric_row[f"residual_physics_core_channel_{idx}_bias"] = float(value)
                    metric_row["residual_extreme_channel_bias"] = float(np.nanmax(np.abs(channel_bias)))
            for latent_name, key in REQUIRED_LATENT_KEYS.items():
                arr = np.asarray(latents[key], dtype=np.float32)
                if not np.isfinite(arr).all():
                    nonfinite_cases.add(case_id)
                flat = arr.reshape(-1)
                latent_values[latent_name].append(flat)
                l2 = float(np.linalg.norm(flat))
                metric_row[f"{latent_name}_code_l2_norm"] = l2
                latent_case_rows.append({
                    "case_id": case_id,
                    "latent": latent_name,
                    "shape": str(tuple(arr.shape)),
                    "dtype": str(arr.dtype),
                    "dimension": int(flat.size),
                    "finite_fraction": float(np.isfinite(flat).mean()),
                    "l2_norm": l2,
                    **_stats(flat),
                })
        with np.load(row["relighting_path"], allow_pickle=False) as relight:
            names = [str(x) for x in relight["preset_names"].tolist()] if "preset_names" in relight.files else []
            images = relight["relighted_images"] if "relighted_images" in relight.files else np.empty((0,))
            if len(names) != 6 or images.shape[0] != 6:
                relighting_incomplete.add(case_id)
            for idx, preset in enumerate(names):
                image = np.asarray(images[idx], dtype=np.float32)
                if not np.isfinite(image).all():
                    nonfinite_cases.add(case_id)
                relighting_rows.append({
                    "case_id": case_id,
                    "preset": preset,
                    "present": True,
                    "finite": bool(np.isfinite(image).all()),
                    "shape": str(tuple(image.shape)),
                    "mean": float(np.mean(image)),
                    "std": float(np.std(image)),
                    "black_fraction": float((image <= 1e-6).mean()),
                    "saturated_fraction": float((image >= 1.0).mean()),
                })
        case_metric_rows.append(metric_row)

    case_stats = pd.DataFrame(case_stat_rows)
    case_stats.to_csv(cfg.output_root / "statistics/image_component_case_stats.csv", index=False)
    global_rows: list[dict[str, Any]] = []
    stat_fields = ["finite_fraction", "min", "max", "mean", "std", "median", "p01", "p05", "p95", "p99", "zero_fraction", "negative_fraction"]
    for keys, group in case_stats.groupby(["component", "region", "channel"]):
        for field in stat_fields:
            global_rows.append({"component": keys[0], "region": keys[1], "channel": keys[2], "stat_field": field, **_robust(group[field])})
    global_stats = pd.DataFrame(global_rows)
    global_stats.to_csv(cfg.output_root / "statistics/image_component_global_stats.csv", index=False)

    case_metrics = pd.DataFrame(case_metric_rows).set_index("case_id")
    latent_cases = pd.DataFrame(latent_case_rows)
    latent_cases.to_csv(cfg.output_root / "statistics/latent_case_stats.csv", index=False)

    dimension_rows: list[dict[str, Any]] = []
    pca_summary: dict[str, Any] = {}
    for latent_name, arrays in latent_values.items():
        matrix = np.stack(arrays, axis=0).astype(np.float64)
        centered = matrix - matrix.mean(axis=0, keepdims=True)
        stds = matrix.std(axis=0)
        variances = matrix.var(axis=0)
        finite_fraction = np.isfinite(matrix).mean(axis=0)
        for idx in range(matrix.shape[1]):
            dimension_rows.append({
                "latent": latent_name,
                "dimension_index": idx,
                "finite_fraction": float(finite_fraction[idx]),
                "mean": float(matrix[:, idx].mean()),
                "std": float(stds[idx]),
                "near_zero_variance": bool(variances[idx] < 1e-12),
            })
        singular = np.linalg.svd(centered, full_matrices=False, compute_uv=False)
        variance = singular ** 2
        total = float(variance.sum())
        explained = (variance / total).tolist() if total > 0 else [0.0 for _ in variance]
        effective_rank = float(np.exp(-(np.array(explained) * np.log(np.array(explained) + 1e-12)).sum())) if explained else 0.0
        pca_summary[latent_name] = {
            "dimension": int(matrix.shape[1]),
            "near_zero_variance_dimensions": int((variances < 1e-12).sum()),
            "effective_rank": effective_rank,
            "pca_explained_variance_first_10": [float(x) for x in explained[:10]],
            "norm_summary": _robust(np.linalg.norm(matrix, axis=1)),
        }
    pd.DataFrame(dimension_rows).to_csv(cfg.output_root / "statistics/latent_dimension_stats.csv", index=False)
    pd.DataFrame(relighting_rows).to_csv(cfg.output_root / "statistics/relighting_stats.csv", index=False)

    group_metrics = [
        "albedo_physics_core_mean",
        "albedo_physics_core_std",
        "normal_physics_core_norm_mean",
        "shading_physics_core_mean",
        "shading_physics_core_std",
        "residual_physics_core_mae",
        "light_code_l2_norm",
        "shape_code_l2_norm",
        "tex_code_l2_norm",
    ]
    fold_rows = []
    for fold, group in case_metrics.groupby("fold"):
        for metric in group_metrics:
            fold_rows.append({"fold": int(fold), "metric": metric, **_robust(group[metric])})
    pd.DataFrame(fold_rows).to_csv(cfg.output_root / "statistics/component_stats_by_fold.csv", index=False)
    label_rows = []
    for by in ("label_binary", "label_original", "label_3class"):
        for label, group in case_metrics.groupby(by):
            for metric in group_metrics:
                label_rows.append({"grouping": by, "label_value": int(label), "metric": metric, **_robust(group[metric])})
    pd.DataFrame(label_rows).to_csv(cfg.output_root / "statistics/component_stats_by_label.csv", index=False)

    component_summary = {
        "shape_signatures": {k: sorted(v) for k, v in shape_signatures.items()},
        "dtype_signatures": {k: sorted(v) for k, v in dtype_signatures.items()},
        "nonfinite_case_count": len(nonfinite_cases),
        "nonfinite_cases": sorted(nonfinite_cases),
        "relighting_incomplete_count": len(relighting_incomplete),
        "relighting_incomplete_cases": sorted(relighting_incomplete),
        "latent_pca_summary": pca_summary,
    }
    return case_metrics, latent_cases, component_summary


def _audit_acquisition(cfg: AuditConfig, manifest: pd.DataFrame) -> dict[str, Any]:
    fields = {
        "camera_model": "string",
        "exposure_time_seconds": "positive_float",
        "fnumber": "positive_float",
        "iso_numeric": "positive_float",
        "brightness_value": "float",
        "datetime_original": "datetime",
        "shooting_time_period": "category",
    }
    rows = []
    for field, kind in fields.items():
        series = manifest[field]
        missing = series.isna() | (series.astype(str).str.len() == 0)
        invalid = pd.Series(False, index=series.index)
        parse_failure = pd.Series(False, index=series.index)
        if kind == "positive_float":
            numeric = pd.to_numeric(series, errors="coerce")
            parse_failure = (~missing) & numeric.isna()
            invalid = (~missing) & numeric.notna() & (numeric <= 0)
        elif kind == "float":
            numeric = pd.to_numeric(series, errors="coerce")
            parse_failure = (~missing) & numeric.isna()
        elif kind == "datetime":
            parse_failure = (~missing) & (manifest["shooting_time_period"].astype(str).str.len() == 0)
        elif kind == "category":
            valid = {"morning", "afternoon", "evening", "night"}
            invalid = (~missing) & ~series.astype(str).isin(valid)
        rows.append({
            "field": field,
            "available_count": int((~missing & ~invalid & ~parse_failure).sum()),
            "missing_count": int(missing.sum()),
            "missing_rate": float(missing.mean()),
            "invalid_count": int(invalid.sum()),
            "parse_failure_count": int(parse_failure.sum()),
            "unique_count": int(series[~missing].nunique()),
        })
    missingness = pd.DataFrame(rows)
    missingness.to_csv(cfg.output_root / "acquisition/metadata_missingness.csv", index=False)

    label_cross = pd.crosstab(manifest["camera_model"].replace("", "MISSING"), manifest["label_binary"])
    label_cross = label_cross.rename(columns={0: "Control", 1: "Patient"}).reset_index()
    label_cross["total"] = label_cross.get("Control", 0) + label_cross.get("Patient", 0)
    label_cross["single_label_camera"] = ((label_cross.get("Control", 0) == 0) | (label_cross.get("Patient", 0) == 0))
    label_cross["rare_camera"] = label_cross["total"] < 5
    label_cross.to_csv(cfg.output_root / "acquisition/camera_label_crosstab.csv", index=False)

    fold_cross = pd.crosstab(manifest["camera_model"].replace("", "MISSING"), manifest["fold"]).reset_index()
    fold_cols = [c for c in fold_cross.columns if c != "camera_model"]
    fold_cross["total"] = fold_cross[fold_cols].sum(axis=1)
    fold_cross["folds_present"] = (fold_cross[fold_cols] > 0).sum(axis=1)
    fold_cross["single_fold_camera"] = fold_cross["folds_present"] == 1
    fold_cross["rare_camera"] = fold_cross["total"] < 5
    fold_cross.to_csv(cfg.output_root / "acquisition/camera_fold_crosstab.csv", index=False)

    numeric_exif = ["exposure_time_seconds", "log_exposure_time", "fnumber", "iso_numeric", "log_iso", "brightness_value"]
    label_rows = []
    for label, group in manifest.groupby("label_binary"):
        for field in numeric_exif:
            label_rows.append({"label_binary": int(label), "field": field, **_robust(group[field])})
    pd.DataFrame(label_rows).to_csv(cfg.output_root / "acquisition/exif_summary_by_label.csv", index=False)
    fold_rows = []
    for fold, group in manifest.groupby("fold"):
        for field in numeric_exif:
            fold_rows.append({"fold": int(fold), "field": field, **_robust(group[field])})
    pd.DataFrame(fold_rows).to_csv(cfg.output_root / "acquisition/exif_summary_by_fold.csv", index=False)
    camera_rows = []
    for camera, group in manifest.groupby(manifest["camera_model"].replace("", "MISSING")):
        for field in numeric_exif:
            camera_rows.append({"camera_model": camera, "field": field, **_robust(group[field])})
    pd.DataFrame(camera_rows).to_csv(cfg.output_root / "acquisition/exif_summary_by_camera.csv", index=False)

    camera_complete = float((manifest["camera_model"].astype(str).str.len() > 0).mean())
    exif_fields = ["exposure_time_seconds", "fnumber", "iso_numeric", "brightness_value"]
    exif_complete = float((manifest[exif_fields].notna() & np.isfinite(manifest[exif_fields].astype(float))).all(axis=1).mean())
    label_status = "SUFFICIENT"
    if bool(label_cross["single_label_camera"].any()):
        label_status = "LIMITED"
    if camera_complete < 0.5:
        label_status = "INSUFFICIENT"
    fold_status = "SUFFICIENT"
    if bool(fold_cross["single_fold_camera"].any()):
        fold_status = "LIMITED"
    if camera_complete < 0.5:
        fold_status = "INSUFFICIENT"
    probe_status = "READY"
    if camera_complete < 0.95 or exif_complete < 0.95 or label_status != "SUFFICIENT" or fold_status != "SUFFICIENT":
        probe_status = "PARTIALLY_READY"
    if camera_complete < 0.5:
        probe_status = "NOT_READY"
    probe = {
        "camera_complete_fraction": camera_complete,
        "core_exif_complete_fraction": exif_complete,
        "camera_label_overlap_status": label_status,
        "camera_fold_overlap_status": fold_status,
        "acquisition_probe_status": probe_status,
        "camera_exif_probe_not_run": True,
    }
    (cfg.output_root / "acquisition/probe_readiness.json").write_text(json.dumps(_nan_json(probe), indent=2), encoding="utf-8")
    return probe


def _build_qc_flags(cfg: AuditConfig, manifest: pd.DataFrame, case_metrics: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    flags: dict[str, set[str]] = {cid: set() for cid in manifest["case_id"].astype(str)}
    indexed_manifest = manifest.set_index("case_id")

    metric_to_flag = {
        "albedo_dynamic_range": "extreme_albedo_range",
        "albedo_physics_core_std": "near_constant_albedo",
        "albedo_channel_balance_range": "abnormal_color_balance",
        "normal_physics_core_norm_mean": "abnormal_normal_norm",
        "shading_dynamic_range": "extreme_shading_range",
        "residual_physics_core_mae": "high_physics_core_residual",
        "residual_extreme_channel_bias": "extreme_channel_bias",
        "shape_code_l2_norm": "latent_norm_outlier",
        "tex_code_l2_norm": "latent_norm_outlier",
        "light_code_l2_norm": "latent_norm_outlier",
        "physics_core_coverage": "low_physics_core_coverage",
    }
    for metric, flag in metric_to_flag.items():
        if metric not in case_metrics.columns:
            continue
        if metric == "albedo_physics_core_std":
            values = pd.to_numeric(case_metrics[metric], errors="coerce")
            q1, q3 = values.quantile([0.25, 0.75])
            iqr = q3 - q1
            threshold = q1 - cfg.qc_iqr_multiplier * iqr
            for cid in values[values < threshold].index.astype(str):
                flags[cid].add(flag)
        elif metric == "physics_core_coverage":
            values = pd.to_numeric(case_metrics[metric], errors="coerce")
            q1, q3 = values.quantile([0.25, 0.75])
            iqr = q3 - q1
            threshold = q1 - cfg.qc_iqr_multiplier * iqr
            for cid in values[values < threshold].index.astype(str):
                flags[cid].add(flag)
        else:
            for cid in _iqr_outliers(case_metrics[metric], cfg.qc_iqr_multiplier):
                flags[cid].add(flag)
    for cid, row in indexed_manifest.iterrows():
        if not _clean_str(row.get("camera_model")):
            flags[str(cid)].add("missing_camera")
        if not np.isfinite(_numeric(row.get("exposure_time_seconds"))):
            flags[str(cid)].add("missing_exposure")
        if not np.isfinite(_numeric(row.get("iso_numeric"))):
            flags[str(cid)].add("missing_iso")
        if not np.isfinite(_numeric(row.get("fnumber"))):
            flags[str(cid)].add("missing_fnumber")
        if not np.isfinite(_numeric(row.get("brightness_value"))):
            flags[str(cid)].add("missing_brightness")
        if not _clean_str(row.get("shooting_time_period")):
            flags[str(cid)].add("invalid_shooting_time")
    rows = []
    for cid in sorted(flags):
        rows.append({"case_id": cid, "flag_count": len(flags[cid]), "flags": ";".join(sorted(flags[cid])), "case_retained": True})
    frame = pd.DataFrame(rows)
    frame.to_csv(cfg.output_root / "qc/p1_case_qc_flags.csv", index=False)
    summary = {
        "flagged_case_count": int((frame["flag_count"] > 0).sum()),
        "cases_removed": False,
        "flag_counts": frame["flags"].str.get_dummies(sep=";").sum().sort_values(ascending=False).to_dict() if not frame.empty else {},
    }
    (cfg.output_root / "qc/outlier_summary.json").write_text(json.dumps(_nan_json(summary), indent=2), encoding="utf-8")
    return frame, summary


def _write_montages(cfg: AuditConfig, manifest: pd.DataFrame, flags: pd.DataFrame) -> None:
    random_cases = manifest["case_id"].astype(str).sort_values().tolist()
    random.seed(20260728)
    random.shuffle(random_cases)
    flagged_cases = flags.loc[flags["flag_count"] > 0, "case_id"].astype(str).tolist()

    def make(case_ids: list[str], out_dir: Path, prefix: str) -> None:
        for cid in case_ids[: cfg.montage_case_count]:
            row = manifest.loc[manifest["case_id"].astype(str) == cid].iloc[0]
            panels = [
                ("rgb", _load_rgb(Path(row["rgb_path"]))),
            ]
            with np.load(row["maps_path"], allow_pickle=False) as maps:
                for key in ("albedo_like", "normal_coarse", "shading_like", "signed_residual"):
                    arr = np.asarray(maps[key], dtype=np.float32)
                    vis = arr
                    if key in {"normal_coarse", "signed_residual"}:
                        lo, hi = np.percentile(arr, [1, 99])
                        vis = (arr - lo) / (hi - lo + 1e-8)
                    panels.append((key, np.clip(vis, 0, 1)))
            tile_w, tile_h = 224, 248
            canvas = Image.new("RGB", (tile_w * len(panels), tile_h), "white")
            draw = ImageDraw.Draw(canvas)
            for idx, (name, arr) in enumerate(panels):
                image = Image.fromarray((np.clip(arr, 0, 1) * 255).astype(np.uint8))
                canvas.paste(image, (idx * tile_w, 0))
                draw.text((idx * tile_w + 4, 228), name, fill=(0, 0, 0))
            canvas.save(out_dir / f"{prefix}_{cid}.png")

    make(random_cases, cfg.output_root / "qc/random_case_montages", "random")
    make(flagged_cases, cfg.output_root / "qc/flagged_case_montages", "flagged")


def _write_schemas_and_sources(cfg: AuditConfig, manifest: pd.DataFrame, component_summary: dict[str, Any]) -> None:
    schema_dir = cfg.output_root / "schema"
    schema_dir.mkdir(parents=True, exist_ok=True)
    component_schema = {
        "image_components": cfg.image_components,
        "map_keys": REQUIRED_MAP_KEYS,
        "regions": cfg.regions,
        "shape_signatures": component_summary["shape_signatures"],
        "dtype_signatures": component_summary["dtype_signatures"],
        "specular_like_status": "unavailable_by_current_frontend",
    }
    (schema_dir / "component_schema.json").write_text(json.dumps(_nan_json(component_schema), indent=2), encoding="utf-8")
    latent_schema = {
        "latent_key_map": REQUIRED_LATENT_KEYS,
        "light_code_shape_preserved": "9x3",
        "flattening_forbidden_in_base_dataset": True,
        "pca_for_audit_only": True,
        "pca_summary": component_summary["latent_pca_summary"],
    }
    (schema_dir / "latent_schema.json").write_text(json.dumps(_nan_json(latent_schema), indent=2), encoding="utf-8")
    metadata_schema = {
        "manifest_columns": {col: str(dtype) for col, dtype in manifest.dtypes.items()},
        "label_binary_rule": "label_original == 0 -> 0 Control; label_original in {1,2,3,4} -> 1 Patient",
        "audit_statistics_are_not_training_normalization_parameters": True,
    }
    (schema_dir / "metadata_schema.json").write_text(json.dumps(_nan_json(metadata_schema), indent=2), encoding="utf-8")
    source_files = [
        cfg.p0_master_index,
        cfg.p0_exif_index,
        cfg.split_csv,
        cfg.frozen_deca_root / "metadata/run_manifest.json",
        cfg.frozen_deca_root / "qc/offline_integrity_audit.json",
        cfg.frozen_deca_root / "metadata/deca_output_inventory.json",
        cfg.frozen_contract,
    ]
    rows = [{"source_path": _rel(path, cfg.root), "sha256": sha256_file(path), "exists": path.is_file()} for path in source_files]
    pd.DataFrame(rows).to_csv(cfg.output_root / "manifests/p1_source_file_manifest.csv", index=False)


def _decide_readiness(
    cfg: AuditConfig,
    manifest: pd.DataFrame,
    asset_summary: dict[str, Any],
    split_summary: dict[str, Any],
    component_summary: dict[str, Any],
    qc_summary: dict[str, Any],
    acquisition_probe: dict[str, Any],
) -> dict[str, Any]:
    blockers: list[str] = []
    if asset_summary["row_count"] != 500 or asset_summary["unique_case_id"] != 500:
        blockers.append("manifest_not_500_unique")
    for key in ("duplicate_case_id", "unmatched_p0_record", "unmatched_split_record", "unmatched_frozen_asset", "missing_required_asset", "asset_hash_mismatch", "input_hash_mismatch", "label_mapping_mismatch", "fold_mismatch"):
        if asset_summary.get(key, 0):
            blockers.append(key)
    if split_summary["patient_or_group_cross_fold_count"]:
        blockers.append("patient_or_group_cross_fold")
    if component_summary["nonfinite_case_count"]:
        blockers.append("nonfinite_component")
    if component_summary["relighting_incomplete_count"]:
        blockers.append("incomplete_relighting")
    inconsistent_shapes = {k: v for k, v in component_summary["shape_signatures"].items() if len(v) != 1}
    if inconsistent_shapes:
        blockers.append("shape_inconsistent")

    warnings = []
    if qc_summary["flagged_case_count"]:
        warnings.append("qc_flags_present")
    warnings.append("specular_like_unavailable_by_current_frontend")
    if acquisition_probe["acquisition_probe_status"] != "READY":
        warnings.append("acquisition_probe_" + acquisition_probe["acquisition_probe_status"].lower())

    if blockers:
        component_status = "BLOCKED"
    elif warnings:
        component_status = "READY_WITH_WARNINGS"
    else:
        component_status = "READY"
    final_status = "BLOCKED_BY_DATA_INTERFACE_ERROR" if blockers else "READY_FOR_P1_COMPONENT_EXPERIMENTS"
    if not blockers and acquisition_probe["acquisition_probe_status"] != "READY":
        final_status = "READY_FOR_P1_COMPONENT_EXPERIMENTS_WITH_ACQUISITION_WARNINGS"
    decision = {
        "component_experiment_status": component_status,
        "acquisition_probe_status": acquisition_probe["acquisition_probe_status"],
        "next_step": "P1-RGB_E0B_BASELINE_REPRODUCTION" if not blockers else "FIX_DATA_INTERFACE_BLOCKERS",
        "final_status": final_status,
        "blockers": blockers,
        "warnings": warnings,
        "audit_statistics_are_not_training_normalization_parameters": True,
        "deca_inference_started": False,
        "classification_started": False,
        "folds_regenerated": False,
        "cases_removed": False,
    }
    (cfg.output_root / "metadata/readiness_decision.json").write_text(json.dumps(_nan_json(decision), indent=2), encoding="utf-8")
    return decision


def _write_metadata(cfg: AuditConfig, summaries: dict[str, Any]) -> None:
    (cfg.output_root / "metadata/audit_config.yaml").write_text(yaml.safe_dump(cfg.raw, sort_keys=False), encoding="utf-8")
    source_hashes = {
        "p0_master_index": sha256_file(cfg.p0_master_index),
        "p0_exif_index": sha256_file(cfg.p0_exif_index),
        "fixed_split_csv": sha256_file(cfg.split_csv),
        "frozen_contract": sha256_file(cfg.frozen_contract),
        "frozen_run_manifest": sha256_file(cfg.frozen_deca_root / "metadata/run_manifest.json"),
        "frozen_offline_integrity_audit": sha256_file(cfg.frozen_deca_root / "qc/offline_integrity_audit.json"),
        "frozen_deca_output_inventory": sha256_file(cfg.frozen_deca_root / "metadata/deca_output_inventory.json"),
    }
    (cfg.output_root / "metadata/source_hashes.json").write_text(json.dumps(source_hashes, indent=2), encoding="utf-8")
    audit_manifest = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "output_root": _rel(cfg.output_root, cfg.root),
        "summaries": summaries,
        "scope": {
            "deca_inference_started": False,
            "classification_started": False,
            "camera_exif_probe_started": False,
            "folds_regenerated": False,
            "cases_removed": False,
        },
    }
    (cfg.output_root / "metadata/audit_manifest.json").write_text(json.dumps(_nan_json(audit_manifest), indent=2), encoding="utf-8")


def _write_report(
    cfg: AuditConfig,
    manifest: pd.DataFrame,
    asset_summary: dict[str, Any],
    split_summary: dict[str, Any],
    component_summary: dict[str, Any],
    qc_summary: dict[str, Any],
    acquisition_probe: dict[str, Any],
    decision: dict[str, Any],
) -> None:
    fold_dist = pd.read_csv(cfg.output_root / "statistics/fold_label_distribution.csv")
    missing = pd.read_csv(cfg.output_root / "acquisition/metadata_missingness.csv")
    relighting = pd.read_csv(cfg.output_root / "statistics/relighting_stats.csv")
    latent_cases = pd.read_csv(cfg.output_root / "statistics/latent_case_stats.csv")
    camera_complete = acquisition_probe["camera_complete_fraction"]
    relighting_presets = sorted(relighting["preset"].unique().tolist()) if not relighting.empty else []
    latent_shapes = latent_cases.groupby("latent")["shape"].unique().map(lambda x: sorted(map(str, x))).to_dict() if not latent_cases.empty else {}
    lines = [
        "# P1 data interface and component statistics audit",
        "",
        "## Scope",
        "",
        "This audit connects the frozen P0 unified data entry, fixed 5-fold split, and Frozen500 DECA assets for P1/P2+ data access. It does not run DECA inference, train classifiers, run camera/EXIF probes, regenerate folds, remove cases, or modify labels.",
        "",
        "## Sources",
        "",
        f"- P0 unified source: `{_rel(cfg.p0_root, cfg.root)}`",
        f"- P0 master index: `{_rel(cfg.p0_master_index, cfg.root)}`",
        f"- Fixed split source: `{_rel(cfg.split_csv, cfg.root)}`",
        f"- Fixed split SHA256: `{split_summary['split_source_sha256']}`",
        f"- Frozen DECA root: `{_rel(cfg.frozen_deca_root, cfg.root)}`",
        f"- Frozen contract: `{_rel(cfg.frozen_contract, cfg.root)}`",
        "",
        "## Join Result",
        "",
        f"- Master manifest rows: `{asset_summary['row_count']}`",
        f"- Unique case IDs: `{asset_summary['unique_case_id']}`",
        f"- Duplicate case IDs: `{asset_summary['duplicate_case_id']}`",
        f"- Unmatched P0 records: `{asset_summary['unmatched_p0_record']}`",
        f"- Unmatched split records: `{asset_summary['unmatched_split_record']}`",
        f"- Unmatched frozen assets: `{asset_summary['unmatched_frozen_asset']}`",
        f"- Missing required assets: `{asset_summary['missing_required_asset']}`",
        f"- Asset hash mismatches: `{asset_summary['asset_hash_mismatch']}`",
        f"- P0 aligned RGB to Frozen input hash matches: `{asset_summary['rgb_input_hash_match_count']}/500`",
        "",
        "## Labels And Folds",
        "",
        "`label_original` is preserved from P0 `NYHA`; `label_binary` is derived only as `0 -> Control` and `1..4 -> Patient`. Existing `label_3class` is preserved and is not overwritten.",
        "",
        f"- Split columns: `{', '.join(split_summary['columns'])}`",
        f"- Patient/group cross-fold count: `{split_summary['patient_or_group_cross_fold_count']}`",
        "",
        "```text",
        fold_dist.to_string(index=False),
        "```",
        "",
        "## Dataset And Adapter",
        "",
        "`P1FrozenAssetDataset` lazy-loads per-case NPZ files with `np.load(..., allow_pickle=False)`, returns image-like tensors as float32 `C x H x W`, keeps masks in `1 x H x W`, preserves signed residual sign, preserves raw normal/shading ranges, and keeps `light_code` as `9 x 3`. `P1RepresentationAdapter` supports `rgb`, `albedo`, `normal`, `light`, `shading`, `residual`, and `rgb_albedo`; `specular` returns `unavailable_by_current_frontend`.",
        "",
        "## Component Statistics",
        "",
        f"- Image shape signatures: `{component_summary['shape_signatures']}`",
        f"- Image dtype signatures: `{component_summary['dtype_signatures']}`",
        f"- Non-finite component cases: `{component_summary['nonfinite_case_count']}`",
        f"- Incomplete relighting cases: `{component_summary['relighting_incomplete_count']}`",
        "",
        "The audit records descriptive image, latent, relighting, fold, and label statistics under `data/processed/P1_Component_Audit_v1/statistics/`. PCA summaries are audit-only and are not saved as training transformers.",
        "",
        "## Specialized Component Audits",
        "",
        "- Albedo-like: physics-core mean/std, dynamic range, near-zero fraction, high-value fraction, inter-case variation, and channel balance are recorded; flags include `near_constant_albedo`, `extreme_albedo_range`, and `abnormal_color_balance`.",
        "- Normal-coarse: values are treated as direction-coded normals, not visual RGB; physics-core norm, near-zero fraction, out-of-range fraction, mean direction, and angular dispersion are recorded.",
        "- Shading-like: raw unclipped `shading_like` is audited without forcing `[0,1]`; negative/high-value fractions and physics-core contrast are recorded.",
        "- Signed residual: sign is preserved; per-region MAE/RMSE/p95/p99 absolute residual, positive/negative fractions, and channel bias are recorded.",
        f"- Latents: shapes are `{latent_shapes}`; per-dimension finite fraction, mean/std, low-variance dimensions, L2 norm, effective rank, and audit-only PCA variance are recorded.",
        f"- Relighting: presets are `{relighting_presets}`; all six presets are checked for presence, finite values, shape, mean/std, black fraction, and saturated fraction.",
        "",
        "## Acquisition Metadata",
        "",
        f"- Camera complete fraction: `{camera_complete:.4f}`",
        f"- Core EXIF complete fraction: `{acquisition_probe['core_exif_complete_fraction']:.4f}`",
        f"- camera_label_overlap_status: `{acquisition_probe['camera_label_overlap_status']}`",
        f"- camera_fold_overlap_status: `{acquisition_probe['camera_fold_overlap_status']}`",
        "",
        "```text",
        missing.to_string(index=False),
        "```",
        "",
        "## QC Flags",
        "",
        f"- Flagged cases: `{qc_summary['flagged_case_count']}`",
        f"- Cases removed: `{qc_summary['cases_removed']}`",
        f"- Flag counts: `{qc_summary['flag_counts']}`",
        "",
        "QC flags are retained only for reporting, visualization, and later sensitivity analysis. They are not used to delete cases, regenerate assets, change labels, or change folds.",
        "",
        "## Readiness",
        "",
        f"- component_experiment_status: `{decision['component_experiment_status']}`",
        f"- acquisition_probe_status: `{decision['acquisition_probe_status']}`",
        f"- next_step: `{decision['next_step']}`",
        f"- final_status: `{decision['final_status']}`",
        "",
        "## Fixed Inputs For P1",
        "",
        "- P1-RGB: P0 `aligned_scene_224` RGB.",
        "- P1-A: `albedo_like * physics_core_skin`.",
        "- P1-N: `normal_coarse * face_valid`.",
        "- P1-L: raw `light_code` as `9 x 3`.",
        "- P1-S: raw unclipped `shading_like * face_valid`.",
        "- P1-R: signed `signed_residual * face_valid`.",
        "- P1-RGB+A: paired `rgb` and `albedo_like * physics_core_skin`.",
        "",
        "## Completion Flags",
        "",
        "```text",
        "p0_unified_source_used = true",
        "fixed_split_source =",
        "data/processed/P0_Physics_Audit_v1/splits_500/",
        "nyha_3class_sex_stratified_group_5fold.csv",
        "",
        "deca_inference_started = false",
        "classification_started = false",
        "folds_regenerated = false",
        "cases_removed = false",
        "```",
        "",
        f"Final status: `{decision['final_status']}`",
        "",
    ]
    cfg.report_path.parent.mkdir(parents=True, exist_ok=True)
    cfg.report_path.write_text("\n".join(lines), encoding="utf-8")


def run_audit(config_path: Path, render_qc: bool = True) -> dict[str, Any]:
    cfg = load_audit_config(config_path)
    ensure_output_dirs(cfg.output_root)
    manifest, asset_summary = _build_manifest(cfg)
    split_summary = _audit_splits(cfg, manifest)
    case_metrics, _latent_cases, component_summary = _audit_components(cfg, manifest)
    acquisition_probe = _audit_acquisition(cfg, manifest)
    flags, qc_summary = _build_qc_flags(cfg, manifest, case_metrics)
    if render_qc:
        _write_montages(cfg, manifest, flags)
    _write_schemas_and_sources(cfg, manifest, component_summary)
    decision = _decide_readiness(cfg, manifest, asset_summary, split_summary, component_summary, qc_summary, acquisition_probe)
    summaries = {
        "asset_join": asset_summary,
        "split": split_summary,
        "components": {k: v for k, v in component_summary.items() if k != "latent_pca_summary"},
        "qc": qc_summary,
        "acquisition": acquisition_probe,
        "readiness": decision,
    }
    _write_metadata(cfg, summaries)
    _write_report(cfg, manifest, asset_summary, split_summary, component_summary, qc_summary, acquisition_probe, decision)
    return summaries


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config/p1/p1_component_audit_v1.yaml")
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--build-manifest", action="store_true")
    parser.add_argument("--audit-components", action="store_true")
    parser.add_argument("--audit-splits", action="store_true")
    parser.add_argument("--audit-acquisition", action="store_true")
    parser.add_argument("--render-qc", action="store_true")
    parser.add_argument("--finalize", action="store_true")
    parser.add_argument("--all", action="store_true")
    args = parser.parse_args(argv)
    config_path = Path(args.config).resolve()
    cfg = load_audit_config(config_path)
    ensure_output_dirs(cfg.output_root)
    if args.validate_only:
        for path in (cfg.p0_root, cfg.p0_master_index, cfg.split_csv, cfg.frozen_deca_root, cfg.frozen_contract):
            if not path.exists():
                raise FileNotFoundError(path)
        return 0
    if args.all or not any((args.build_manifest, args.audit_components, args.audit_splits, args.audit_acquisition, args.render_qc, args.finalize)):
        result = run_audit(config_path, render_qc=True)
        print(json.dumps(_nan_json(result["readiness"]), indent=2))
        return 0
    manifest, asset_summary = _build_manifest(cfg)
    split_summary = _audit_splits(cfg, manifest) if args.audit_splits or args.finalize else {}
    case_metrics = pd.DataFrame()
    component_summary: dict[str, Any] = {"shape_signatures": {}, "dtype_signatures": {}, "nonfinite_case_count": 0, "relighting_incomplete_count": 0, "latent_pca_summary": {}}
    if args.audit_components or args.finalize:
        case_metrics, _latent_cases, component_summary = _audit_components(cfg, manifest)
    acquisition_probe = _audit_acquisition(cfg, manifest) if args.audit_acquisition or args.finalize else {"acquisition_probe_status": "PARTIALLY_READY", "camera_complete_fraction": math.nan, "core_exif_complete_fraction": math.nan, "camera_label_overlap_status": "LIMITED", "camera_fold_overlap_status": "LIMITED"}
    flags = pd.DataFrame()
    qc_summary = {"flagged_case_count": 0, "cases_removed": False, "flag_counts": {}}
    if args.render_qc or args.finalize:
        if case_metrics.empty:
            case_metrics, _latent_cases, component_summary = _audit_components(cfg, manifest)
        flags, qc_summary = _build_qc_flags(cfg, manifest, case_metrics)
        _write_montages(cfg, manifest, flags)
    if args.finalize:
        if not split_summary:
            split_summary = _audit_splits(cfg, manifest)
        _write_schemas_and_sources(cfg, manifest, component_summary)
        decision = _decide_readiness(cfg, manifest, asset_summary, split_summary, component_summary, qc_summary, acquisition_probe)
        _write_metadata(cfg, {"asset_join": asset_summary, "split": split_summary, "components": component_summary, "qc": qc_summary, "acquisition": acquisition_probe, "readiness": decision})
        _write_report(cfg, manifest, asset_summary, split_summary, component_summary, qc_summary, acquisition_probe, decision)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
