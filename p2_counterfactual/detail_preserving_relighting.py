from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
import time
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw

from p2_counterfactual.assets import (
    BOUNDARY_IDS,
    MASTER_MANIFEST,
    P0B_PILOT12_MAPPING,
    P0_ROOT,
    P1_AUDIT_ROOT,
    P1_FROZEN_ROOT,
    P1_QC_FLAGS,
    P2_ROOT as P2_SHADING_ROOT,
    PRESET_NAMES,
    ROOT,
    get_preset_index,
    json_safe,
    sha256_file,
    utc_now,
    write_json,
)
from p2_counterfactual.dataset import P2ASingleRGBDataset, load_p2_manifest
from p2_counterfactual.path_utils import resolve_project_path

OUTPUT_ROOT = ROOT / "data/processed/P2_DetailPreserving_Relighting500_v2"
MEANBG_ROOT = P0_ROOT / "images/e0b_meanbg_224"
P2_SHADING_MANIFEST = P2_SHADING_ROOT / "manifests/p2_training_manifest.csv"
EPSILON = 1e-4
VERSION = "P2_DetailPreserving_Relighting500_v2"


def digest_file(path: Path) -> str:
    return sha256_file(path)


def ensure_dirs() -> None:
    for path in (
        OUTPUT_ROOT / "cases",
        OUTPUT_ROOT / "manifests",
        OUTPUT_ROOT / "metadata",
        OUTPUT_ROOT / "qc/pilot_contact_sheets",
        OUTPUT_ROOT / "qc/full_contact_sheets",
        OUTPUT_ROOT / "reports",
        OUTPUT_ROOT / "logs",
    ):
        path.mkdir(parents=True, exist_ok=True)


def normalize_rgb_hwc(value: np.ndarray, *, name: str) -> np.ndarray:
    arr = np.asarray(value)
    if arr.ndim == 2:
        arr = np.repeat(arr[..., None], 3, axis=-1)
    elif arr.ndim == 3 and arr.shape[0] in {1, 3} and arr.shape[-1] not in {1, 3}:
        arr = np.transpose(arr, (1, 2, 0))
    if arr.ndim != 3 or arr.shape[-1] not in {1, 3}:
        raise ValueError(f"{name} must be RGB-like HWC/CHW, got shape {arr.shape}")
    if arr.shape[-1] == 1:
        arr = np.repeat(arr, 3, axis=-1)
    arr = arr.astype(np.float32, copy=False)
    if arr.size == 0 or not np.isfinite(arr).all():
        raise ValueError(f"{name} contains empty or non-finite values")
    if arr.max() > 2.0:
        arr = arr / 255.0
    return np.clip(arr, 0.0, 1.0).astype(np.float32)


def normalize_shading_hwc(value: np.ndarray, *, name: str, target_hw: tuple[int, int]) -> np.ndarray:
    arr = np.asarray(value)
    if arr.ndim == 2:
        arr = arr[..., None]
    elif arr.ndim == 3 and arr.shape[0] in {1, 3} and arr.shape[-1] not in {1, 3}:
        arr = np.transpose(arr, (1, 2, 0))
    if arr.ndim != 3 or arr.shape[-1] not in {1, 3}:
        raise ValueError(f"{name} must be shading-like HWC/CHW, got shape {arr.shape}")
    if arr.shape[-1] == 1:
        arr = np.repeat(arr, 3, axis=-1)
    arr = arr.astype(np.float32, copy=False)
    if arr.size == 0 or not np.isfinite(arr).all():
        raise ValueError(f"{name} contains empty or non-finite values")
    if arr.shape[:2] != target_hw:
        arr = resize_hwc_float(arr, target_hw, is_mask=False)
    return arr.astype(np.float32)


def normalize_alpha_hw1(value: np.ndarray, *, name: str, target_hw: tuple[int, int]) -> np.ndarray:
    arr = np.asarray(value)
    if arr.ndim == 3 and arr.shape[0] == 1 and arr.shape[-1] != 1:
        arr = np.transpose(arr, (1, 2, 0))
    if arr.ndim == 3:
        if arr.shape[-1] == 1:
            arr = arr[..., 0]
        elif arr.shape[-1] == 3:
            arr = arr.mean(axis=-1)
        else:
            raise ValueError(f"{name} has unsupported alpha shape {arr.shape}")
    if arr.ndim != 2:
        raise ValueError(f"{name} must be alpha/mask-like, got shape {arr.shape}")
    arr = arr.astype(np.float32, copy=False)
    if arr.size == 0 or not np.isfinite(arr).all():
        raise ValueError(f"{name} contains empty or non-finite values")
    if arr.max() > 1.0:
        arr = arr / 255.0
    arr = np.clip(arr, 0.0, 1.0)
    if arr.shape != target_hw:
        arr = resize_hw_float(arr, target_hw, is_mask=True)
    return arr[..., None].astype(np.float32)


def resize_hw_float(value: np.ndarray, target_hw: tuple[int, int], *, is_mask: bool) -> np.ndarray:
    mode = Image.Resampling.NEAREST if is_mask else Image.Resampling.BILINEAR
    image = Image.fromarray(value.astype(np.float32), mode="F")
    resized = image.resize((int(target_hw[1]), int(target_hw[0])), mode)
    return np.asarray(resized, dtype=np.float32)


def resize_hwc_float(value: np.ndarray, target_hw: tuple[int, int], *, is_mask: bool) -> np.ndarray:
    channels = [resize_hw_float(value[..., c], target_hw, is_mask=is_mask) for c in range(value.shape[-1])]
    return np.stack(channels, axis=-1).astype(np.float32)


def read_rgb_png(path: Path) -> np.ndarray:
    if not path.is_file():
        raise FileNotFoundError(path)
    return normalize_rgb_hwc(np.asarray(Image.open(path).convert("RGB")), name=str(path))


def generate_detail_preserving_images(
    meanbg_rgb: np.ndarray,
    original_shading: np.ndarray,
    relighted_shading: np.ndarray,
    alpha: np.ndarray,
    *,
    epsilon: float = EPSILON,
) -> np.ndarray:
    rgb = normalize_rgb_hwc(meanbg_rgb, name="meanbg_rgb")
    target_hw = tuple(rgb.shape[:2])
    s_orig = normalize_shading_hwc(original_shading, name="original_shading", target_hw=target_hw)
    rel = np.asarray(relighted_shading)
    if rel.ndim == 3 and rel.shape[0] == len(PRESET_NAMES):
        rel = rel[..., None]
    if rel.ndim != 4 or rel.shape[0] != len(PRESET_NAMES):
        raise ValueError(f"relighted_shading must be (6,H,W,C), got {rel.shape}")
    alpha_hw1 = normalize_alpha_hw1(alpha, name="alpha", target_hw=target_hw)
    output = []
    for index in range(rel.shape[0]):
        s_k = normalize_shading_hwc(rel[index], name=f"relighted_shading[{index}]", target_hw=target_hw)
        ratio = s_k / (s_orig + float(epsilon))
        face = np.clip(rgb * ratio, 0.0, 1.0)
        image = alpha_hw1 * face + (1.0 - alpha_hw1) * rgb
        image = np.clip(image, 0.0, 1.0).astype(np.float32)
        if not np.isfinite(image).all():
            raise ValueError(f"Generated non-finite relighted image at preset index {index}")
        output.append(image)
    return np.stack(output, axis=0).astype(np.float32)


def source_hashes(include_meanbg: bool = True) -> dict[str, Any]:
    files = {
        "p1_master_manifest": MASTER_MANIFEST,
        "p1_frozen_run_manifest": P1_FROZEN_ROOT / "metadata/run_manifest.json",
        "p1_frozen_asset_hashes": P1_FROZEN_ROOT / "metadata/asset_hashes.json",
        "p1_frozen_offline_integrity_audit": P1_FROZEN_ROOT / "qc/offline_integrity_audit.json",
        "p1_qc_flags": P1_QC_FLAGS,
        "p2_shading_manifest": P2_SHADING_MANIFEST,
        "p2_shading_frozen": P2_SHADING_ROOT / "metadata/P2_COUNTERFACTUAL_TRAINING_ASSETS_FROZEN.json",
        "p0_master_index": P0_ROOT / "metadata/master_index.csv",
        "p0_effective_config": P0_ROOT / "metadata/effective_config.yaml",
    }
    payload: dict[str, Any] = {name: digest_file(path) if path.is_file() else "missing" for name, path in files.items()}
    if include_meanbg:
        meanbg_hashes = {path.name: digest_file(path) for path in sorted(MEANBG_ROOT.glob("*.png"))}
        payload["meanbg_file_count"] = len(meanbg_hashes)
        payload["meanbg_combined_sha256"] = hashlib.sha256(
            json.dumps(meanbg_hashes, sort_keys=True).encode("utf-8")
        ).hexdigest()
    return payload


def write_source_inventory(stage: str) -> dict[str, Any]:
    ensure_dirs()
    path = OUTPUT_ROOT / "metadata/source_inventory.json"
    current = source_hashes()
    if path.is_file():
        existing = json.loads(path.read_text(encoding="utf-8"))
        baseline = existing.get("baseline_hashes", current)
    else:
        baseline = current
    data = {
        "stage": stage,
        "updated_at": utc_now(),
        "baseline_hashes": baseline,
        "current_hashes": current,
        "source_assets_unchanged": baseline == current,
        "protected_roots": [str(P1_FROZEN_ROOT), str(MEANBG_ROOT), str(P2_SHADING_ROOT)],
    }
    write_json(path, data)
    return data


def read_base_manifest() -> pd.DataFrame:
    frame = pd.read_csv(P2_SHADING_MANIFEST, dtype={"case_id": str, "patient_group_id": str})
    required = {
        "case_id",
        "patient_group_id",
        "fold",
        "binary_label",
        "maps_npz_path",
        "relighting_npz_path",
        "relighted_shading_path",
        "p1_qc_flag",
        "boundary_uncertain",
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"P2 shading manifest missing required columns: {missing}")
    if len(frame) != 500 or frame["case_id"].nunique() != 500:
        raise ValueError("P2 shading manifest must contain 500 unique cases")
    return frame.copy()


def meanbg_path(case_id: str) -> Path:
    path = MEANBG_ROOT / f"{case_id}.png"
    if not path.is_file():
        raise FileNotFoundError(f"Missing meanbg RGB for case {case_id}: {path}")
    return path


def case_paths(row: Mapping[str, Any]) -> dict[str, Path]:
    cid = str(row["case_id"])
    return {
        "meanbg": meanbg_path(cid),
        "maps": resolve_project_path(row["maps_npz_path"], ROOT, require_exists=True, case_id=cid, column_name="maps_npz_path"),
        "old_relighting": resolve_project_path(row["relighting_npz_path"], ROOT, require_exists=True, case_id=cid, column_name="relighting_npz_path"),
        "relighted_shading": resolve_project_path(row["relighted_shading_path"], ROOT, require_exists=True, case_id=cid, column_name="relighted_shading_path"),
    }


def success_valid(case_dir: Path, row: Mapping[str, Any]) -> bool:
    success = case_dir / "_SUCCESS.json"
    output = case_dir / "relighting.npz"
    if not success.is_file() or not output.is_file():
        return False
    try:
        paths = case_paths(row)
        data = json.loads(success.read_text(encoding="utf-8"))
        expected = {
            "source_meanbg_sha256": digest_file(paths["meanbg"]),
            "source_maps_sha256": digest_file(paths["maps"]),
            "source_relighted_shading_sha256": digest_file(paths["relighted_shading"]),
            "output_sha256": digest_file(output),
        }
        if any(data.get(key) != value for key, value in expected.items()):
            return False
        with np.load(output, allow_pickle=False) as archive:
            names = [str(x) for x in archive["preset_names"].tolist()]
            arr = archive["relighted_images"]
            return names == list(PRESET_NAMES) and arr.shape == (6, 224, 224, 3) and arr.dtype == np.float32 and np.isfinite(arr).all()
    except Exception:
        return False


def build_case(row: Mapping[str, Any], *, resume: bool, overwrite: bool) -> dict[str, Any]:
    cid = str(row["case_id"])
    case_dir = OUTPUT_ROOT / "cases" / cid
    output_path = case_dir / "relighting.npz"
    if resume and success_valid(case_dir, row):
        return {"case_id": cid, "status": "skipped_valid_resume", "output_path": str(output_path)}
    if output_path.exists() and not overwrite:
        return {"case_id": cid, "status": "failed", "stage": "preflight", "error": "existing output; pass --overwrite or --resume"}
    case_dir.mkdir(parents=True, exist_ok=True)
    tmp_dir = case_dir / f".tmp_{time.time_ns()}"
    tmp_dir.mkdir()
    try:
        paths = case_paths(row)
        meanbg = read_rgb_png(paths["meanbg"])
        with np.load(paths["maps"], allow_pickle=False) as maps:
            original_shading = maps["shading_like"]
            alpha = maps["alpha"]
        with np.load(paths["old_relighting"], allow_pickle=False) as old_rel:
            old_names = [str(x) for x in old_rel["preset_names"].tolist()]
        with np.load(paths["relighted_shading"], allow_pickle=False) as p2_shading:
            names = [str(x) for x in p2_shading["preset_names"].tolist()]
            relighted_shading = p2_shading["relighted_shading"]
        if old_names != names or names != list(PRESET_NAMES):
            raise ValueError(f"preset mismatch for case {cid}: old={old_names}, shading={names}, expected={list(PRESET_NAMES)}")
        new_images = generate_detail_preserving_images(meanbg, original_shading, relighted_shading, alpha, epsilon=EPSILON)
        np.savez_compressed(
            tmp_dir / "relighting.npz",
            preset_names=np.asarray(names),
            relighted_images=new_images,
            generation_method=np.asarray("DECA_guided_shading_ratio_detail_preserving"),
            epsilon=np.asarray(EPSILON, dtype=np.float32),
            source_meanbg_path=np.asarray(str(paths["meanbg"])),
            source_maps_path=np.asarray(str(paths["maps"])),
            source_relighted_shading_path=np.asarray(str(paths["relighted_shading"])),
        )
        if output_path.exists():
            output_path.unlink()
        shutil.move(str(tmp_dir / "relighting.npz"), str(output_path))
        success = {
            "case_id": cid,
            "method": "DECA_guided_shading_ratio_detail_preserving",
            "epsilon": EPSILON,
            "source_meanbg_path": str(paths["meanbg"]),
            "source_maps_path": str(paths["maps"]),
            "source_old_relighting_path": str(paths["old_relighting"]),
            "source_relighted_shading_path": str(paths["relighted_shading"]),
            "source_meanbg_sha256": digest_file(paths["meanbg"]),
            "source_maps_sha256": digest_file(paths["maps"]),
            "source_old_relighting_sha256": digest_file(paths["old_relighting"]),
            "source_relighted_shading_sha256": digest_file(paths["relighted_shading"]),
            "output_path": str(output_path),
            "output_sha256": digest_file(output_path),
            "preset_names": names,
            "shape": list(new_images.shape),
            "dtype": str(new_images.dtype),
            "finite": bool(np.isfinite(new_images).all()),
            "range": [float(new_images.min()), float(new_images.max())],
            "completed_at": utc_now(),
            "old_deca_render_rgb_used_in_synthesis": False,
        }
        write_json(case_dir / "_SUCCESS.json", success)
        return {"case_id": cid, "status": "success", "output_path": str(output_path)}
    except Exception as exc:
        shutil.rmtree(tmp_dir, ignore_errors=True)
        return {"case_id": cid, "status": "failed", "stage": "case_generation", "error": f"{type(exc).__name__}: {exc}"}


def select_rows(mode: str) -> pd.DataFrame:
    frame = read_base_manifest()
    if mode == "pilot":
        if P0B_PILOT12_MAPPING.is_file():
            pilot = pd.read_csv(P0B_PILOT12_MAPPING, dtype=str)
            column = "sample_id" if "sample_id" in pilot.columns else pilot.columns[0]
            selected = frame[frame["case_id"].isin([str(x) for x in pilot[column].tolist()])].copy()
            if len(selected) == 12:
                return selected.reset_index(drop=True)
        boundary = frame[frame["case_id"].isin(BOUNDARY_IDS)].copy()
        rest = frame[~frame["case_id"].isin(set(boundary["case_id"]))].head(12 - len(boundary))
        return pd.concat([boundary, rest], ignore_index=True)
    if mode == "full":
        return frame.reset_index(drop=True)
    raise ValueError(f"Unknown mode: {mode}")


def build_manifest() -> tuple[pd.DataFrame, pd.DataFrame]:
    ensure_dirs()
    base = read_base_manifest()
    rows: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    for row in base.to_dict(orient="records"):
        cid = str(row["case_id"])
        relighting_path = OUTPUT_ROOT / "cases" / cid / "relighting.npz"
        original_path = meanbg_path(cid)
        ready = False
        error = ""
        try:
            with np.load(relighting_path, allow_pickle=False) as archive:
                names = [str(x) for x in archive["preset_names"].tolist()]
                arr = archive["relighted_images"]
                ready = names == list(PRESET_NAMES) and arr.shape == (6, 224, 224, 3) and arr.dtype == np.float32 and np.isfinite(arr).all()
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
        if not ready:
            failures.append({"case_id": cid, "failure_stage": "manifest_validation", "error": error})
        rows.append(
            {
                **{key: row[key] for key in ["case_id", "patient_group_id", "fold", "binary_label"]},
                "original_rgb_path": str(original_path),
                "maps_npz_path": row["maps_npz_path"],
                "relighting_npz_path": str(relighting_path),
                "old_deca_relighting_npz_path": row["relighting_npz_path"],
                "relighted_shading_path": row["relighted_shading_path"],
                "original_shading_key": "shading_like",
                "alpha_key": "alpha",
                "relighted_rgb_key": "relighted_images",
                "relighted_shading_key": "relighted_shading",
                "preset_names": json.dumps(list(PRESET_NAMES), ensure_ascii=False),
                "relighting_generation_method": "DECA_guided_shading_ratio_detail_preserving",
                "epsilon": EPSILON,
                "p1_qc_flag": row["p1_qc_flag"],
                "boundary_uncertain": row["boundary_uncertain"],
                "source_asset_valid": bool(Path(row["maps_npz_path"]).is_file() and Path(row["relighted_shading_path"]).is_file() and original_path.is_file()),
                "detail_preserving_relighting_valid": bool(ready),
                "p2_training_ready": bool(ready),
            }
        )
    manifest = pd.DataFrame(rows)
    failure = pd.DataFrame(failures, columns=["case_id", "failure_stage", "error"])
    manifest.to_csv(OUTPUT_ROOT / "manifests/p2_training_manifest.csv", index=False, encoding="utf-8")
    failure.to_csv(OUTPUT_ROOT / "manifests/p2_failure_manifest.csv", index=False, encoding="utf-8")
    return manifest, failure


def write_contact_sheets(case_ids: Iterable[str], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for cid in case_ids:
        relighting_path = OUTPUT_ROOT / "cases" / str(cid) / "relighting.npz"
        if not relighting_path.is_file():
            continue
        meanbg = Image.open(meanbg_path(str(cid))).convert("RGB").resize((224, 224))
        with np.load(relighting_path, allow_pickle=False) as archive:
            names = [str(x) for x in archive["preset_names"].tolist()]
            images = archive["relighted_images"]
        panel = Image.new("RGB", (7 * 224, 252), "white")
        draw = ImageDraw.Draw(panel)
        draw.text((4, 4), f"{cid} | meanbg original", fill="black")
        panel.paste(meanbg, (0, 28))
        for idx, name in enumerate(names):
            x = (idx + 1) * 224
            arr = np.rint(np.clip(images[idx], 0.0, 1.0) * 255.0).astype(np.uint8)
            panel.paste(Image.fromarray(arr, "RGB"), (x, 28))
            draw.text((x + 4, 4), name, fill="black")
        draw.text((4, 232), "method: DECA-guided shading-ratio detail-preserving relighting", fill="black")
        panel.save(output_dir / f"{cid}.png")


def build(mode: str, *, resume: bool = True, overwrite: bool = False) -> dict[str, Any]:
    ensure_dirs()
    inventory = write_source_inventory(f"{mode}-pre")
    rows = select_rows(mode).to_dict(orient="records")
    outcomes = [build_case(row, resume=resume, overwrite=overwrite) for row in rows]
    failures = [item for item in outcomes if item["status"] == "failed"]
    summary = {
        "mode": mode,
        "expected_cases": len(rows),
        "success_or_resume": sum(item["status"] in {"success", "skipped_valid_resume"} for item in outcomes),
        "new_success": sum(item["status"] == "success" for item in outcomes),
        "skipped_valid_resume": sum(item["status"] == "skipped_valid_resume" for item in outcomes),
        "failed": len(failures),
        "failures": failures,
        "source_inventory_pre_unchanged": inventory["source_assets_unchanged"],
        "completed_at": utc_now(),
    }
    write_json(OUTPUT_ROOT / f"logs/{mode}_build_summary.json", summary)
    build_manifest()
    if mode == "pilot":
        write_contact_sheets([row["case_id"] for row in rows], OUTPUT_ROOT / "qc/pilot_contact_sheets")
    return summary


def validate_assets(*, freeze: bool = False) -> dict[str, Any]:
    ensure_dirs()
    manifest, failures = build_manifest()
    stats: list[dict[str, Any]] = []
    hard_failures: list[str] = []
    pair_errors: list[str] = []
    background_errors: list[str] = []
    old_path_refs: list[str] = []
    for row in manifest.to_dict(orient="records"):
        cid = str(row["case_id"])
        try:
            with np.load(row["relighting_npz_path"], allow_pickle=False) as rel, np.load(row["relighted_shading_path"], allow_pickle=False) as shade, np.load(row["maps_npz_path"], allow_pickle=False) as maps:
                names = [str(x) for x in rel["preset_names"].tolist()]
                shade_names = [str(x) for x in shade["preset_names"].tolist()]
                if names != shade_names or names != list(PRESET_NAMES):
                    pair_errors.append(cid)
                images = rel["relighted_images"]
                meanbg = read_rgb_png(Path(row["original_rgb_path"]))
                alpha = normalize_alpha_hw1(maps["alpha"], name=f"{cid}:alpha", target_hw=tuple(meanbg.shape[:2]))
                outside = alpha[..., 0] <= 1e-6
                if outside.any():
                    diff = np.abs(images[:, outside, :] - meanbg[outside][None, :, :])
                    bg_max = float(diff.max())
                    bg_mean = float(diff.mean())
                else:
                    bg_max = 0.0
                    bg_mean = 0.0
                if bg_max > 1e-6:
                    background_errors.append(cid)
                stat = {
                    "case_id": cid,
                    "shape": list(images.shape),
                    "dtype": str(images.dtype),
                    "finite": bool(np.isfinite(images).all()),
                    "min": float(images.min()),
                    "max": float(images.max()),
                    "mean": float(images.mean()),
                    "std": float(images.std()),
                    "background_max_abs_diff": bg_max,
                    "background_mean_abs_diff": bg_mean,
                    "all_black": bool(np.all(images <= 1e-6)),
                    "all_white": bool(np.all(images >= 1.0 - 1e-6)),
                }
                stats.append(stat)
                if images.shape != (6, 224, 224, 3) or images.dtype != np.float32 or not stat["finite"]:
                    hard_failures.append(f"{cid}:shape_dtype_finite")
                if stat["min"] < -1e-6 or stat["max"] > 1.0 + 1e-6:
                    hard_failures.append(f"{cid}:range")
                if stat["all_black"] or stat["all_white"] or stat["std"] <= 1e-8:
                    hard_failures.append(f"{cid}:empty_or_constant")
        except Exception as exc:
            hard_failures.append(f"{cid}:{type(exc).__name__}:{exc}")
    stats_df = pd.DataFrame(stats)
    stats_df.to_csv(OUTPUT_ROOT / "qc/detail_preserving_relighting_numeric_audit.csv", index=False, encoding="utf-8")
    source_inventory = write_source_inventory("freeze" if freeze else "validate")
    config_refs = scan_p2_config_refs()
    for ref in config_refs:
        if ref["experiment_id"] in {"p2_a2_relighting", "p2_a3_full_consistency"} and "P2_Counterfactual_Relighting500_v1" in ref["manifest_path"]:
            old_path_refs.append(ref["config_path"])
    dataset_smoke = dataset_smoke_test()
    checks = {
        "manifest_500_rows": len(manifest) == 500,
        "unique_500_cases": manifest["case_id"].nunique() == 500,
        "ready_500_cases": int(manifest["p2_training_ready"].sum()) == 500,
        "failure_manifest_empty": len(failures) == 0,
        "preset_pairing": not pair_errors,
        "numeric_integrity": not hard_failures,
        "background_integrity": not background_errors,
        "source_assets_unchanged": bool(source_inventory["source_assets_unchanged"]),
        "a2_a3_config_switched": not old_path_refs,
        "dataset_smoke_passed": dataset_smoke["status"] == "passed",
    }
    failed_checks = [key for key, ok in checks.items() if not ok]
    summary = {
        "status": "passed" if not failed_checks else "failed",
        "created_at": utc_now(),
        "case_count": int(len(manifest)),
        "unique_case_count": int(manifest["case_id"].nunique()),
        "ready_case_count": int(manifest["p2_training_ready"].sum()),
        "preset_count": len(PRESET_NAMES),
        "relighted_image_count": int(manifest["p2_training_ready"].sum()) * len(PRESET_NAMES),
        "failure_manifest_rows": int(len(failures)),
        "pair_error_count": len(pair_errors),
        "background_error_count": len(background_errors),
        "numeric_hard_failure_count": len(hard_failures),
        "checks": checks,
        "failed_checks": failed_checks,
        "dataset_smoke": dataset_smoke,
        "source_assets_unchanged": bool(source_inventory["source_assets_unchanged"]),
        "old_deca_render_rgb_used_in_synthesis": False,
        "classification_training_executed": False,
        "deca_encoder_executed": False,
        "epsilon": EPSILON,
        "preset_names": list(PRESET_NAMES),
        "warnings": [],
    }
    write_json(OUTPUT_ROOT / "metadata/validation_summary.json", summary)
    write_report(summary, config_refs)
    if freeze:
        write_freeze(summary)
    return summary


def scan_p2_config_refs() -> list[dict[str, str]]:
    refs = []
    for path in sorted((ROOT / "config/p2/p2_a").glob("p2_a*.yaml")):
        text = path.read_text(encoding="utf-8")
        experiment_id = ""
        manifest_path = ""
        for line in text.splitlines():
            if line.startswith("experiment_id:"):
                experiment_id = line.split(":", 1)[1].strip()
            if line.startswith("manifest_path:"):
                manifest_path = line.split(":", 1)[1].strip()
        refs.append({"config_path": str(path), "experiment_id": experiment_id, "manifest_path": manifest_path})
    return refs


def dataset_smoke_test() -> dict[str, Any]:
    manifest = OUTPUT_ROOT / "manifests/p2_training_manifest.csv"
    try:
        frame = load_p2_manifest(manifest)
        ds_a2 = P2ASingleRGBDataset(
            frame,
            fold=0,
            split="train",
            input_mode="relight_mix",
            training=True,
            original_probability=0.0,
            max_cases=4,
            horizontal_flip_probability=0.0,
            normalization_mean=(0.0, 0.0, 0.0),
            normalization_std=(1.0, 1.0, 1.0),
            original_rgb_override_dir=MEANBG_ROOT,
        )
        ds_a3 = P2ASingleRGBDataset(
            frame,
            fold=0,
            split="train",
            input_mode="paired",
            training=True,
            max_cases=4,
            horizontal_flip_probability=0.0,
            normalization_mean=(0.0, 0.0, 0.0),
            normalization_std=(1.0, 1.0, 1.0),
            original_rgb_override_dir=MEANBG_ROOT,
        )
        a2 = ds_a2[0]
        a3 = ds_a3[0]
        relighting_path = Path(ds_a2.frame.iloc[0]["relighting_npz_path"])
        old_path = Path(ds_a2.frame.iloc[0]["old_deca_relighting_npz_path"])
        return {
            "status": "passed",
            "a2_source_type": a2["source_type"],
            "a2_image_shape": list(a2["image"].shape),
            "a3_original_shape": list(a3["original_image"].shape),
            "a3_counterfactual_shape": list(a3["counterfactual_image"].shape),
            "a2_relighting_path": str(relighting_path),
            "old_deca_path_still_historical_only": str(old_path),
            "dataset_reads_new_relighting": OUTPUT_ROOT in relighting_path.parents,
        }
    except Exception as exc:
        return {"status": "failed", "error": f"{type(exc).__name__}: {exc}"}


def write_freeze(summary: Mapping[str, Any]) -> None:
    freeze = {
        "status": "P2_DETAIL_PRESERVING_RELIGHTING_ASSETS_FROZEN" if summary.get("status") == "passed" else "P2_DETAIL_PRESERVING_RELIGHTING_ASSETS_NOT_FROZEN",
        "created_at": utc_now(),
        "version": VERSION,
        "case_count": summary.get("case_count"),
        "preset_count": summary.get("preset_count"),
        "relighted_image_count": summary.get("relighted_image_count"),
        "manifest_sha256": digest_file(OUTPUT_ROOT / "manifests/p2_training_manifest.csv"),
        "validation_summary_sha256": digest_file(OUTPUT_ROOT / "metadata/validation_summary.json"),
        "source_inventory_sha256": digest_file(OUTPUT_ROOT / "metadata/source_inventory.json"),
        "failed_checks": summary.get("failed_checks", []),
        "source_assets_unchanged": summary.get("source_assets_unchanged"),
        "old_deca_render_rgb_used_in_synthesis": False,
        "classification_training_executed": False,
        "deca_encoder_executed": False,
    }
    write_json(OUTPUT_ROOT / "metadata/P2_DETAIL_PRESERVING_RELIGHTING_ASSETS_FROZEN.json", freeze)


def write_report(summary: Mapping[str, Any], config_refs: list[dict[str, str]]) -> None:
    report = [
        "# P2 Detail-Preserving Relighting Asset Generation Report",
        "",
        "## 1. 任务目标",
        "旧版 `relighted_images` 来自 DECA 直接渲染 RGB，视觉上会抹除真实皮肤纹理、眼周、唇部和面颊高频信息。本任务生成以 meanbg 原图为基础、只改变人脸区域 shading ratio 的 P2-A2/P2-A3 输入。",
        "",
        "## 2. 实际项目代码审计",
        f"- 权威 500 例 manifest：`{P2_SHADING_MANIFEST}`。",
        f"- meanbg 原图目录：`{MEANBG_ROOT}`，文件名为 `<case_id>.png`。",
        "- 原始 shading：`maps.npz` / `shading_like`。",
        "- alpha：`maps.npz` / `alpha`。",
        "- 六套目标 shading：P2-0 `relit_shading.npz` / `relighted_shading`。",
        "- 旧 DECA render RGB：P1 `relighting.npz` / `relighted_images`，仅用于审计 preset，不参与新图合成。",
        "- P2 Dataset：`p2_counterfactual.dataset.P2ASingleRGBDataset`，通过 manifest 的 `relighting_npz_path` 读取 `relighted_images`。",
        "- 路径兼容：复用 `p2_counterfactual.path_utils.resolve_project_path`。",
        "",
        "## 3. 新生成方法",
        f"- `epsilon = {EPSILON}`。",
        "- `R_k = S_k / (S_orig + epsilon)`",
        "- `I_relight_k = alpha * clip(I_meanbg * R_k, 0, 1) + (1 - alpha) * I_meanbg`",
        "",
        "## 4. 输入与输出资产",
        f"- 输出目录：`{OUTPUT_ROOT}`。",
        "- 每例：`cases/<case_id>/relighting.npz`，字段 `preset_names`、`relighted_images`。",
        "- 新 manifest：`manifests/p2_training_manifest.csv`。",
        "- 数组格式：`relighted_images` 为 `(6,224,224,3)`、`float32`、RGB、范围 `[0,1]`。",
        f"- preset 顺序：`{list(PRESET_NAMES)}`。",
        "",
        "## 5. Pilot结果",
        f"- Pilot 对比图目录：`{OUTPUT_ROOT / 'qc/pilot_contact_sheets'}`。",
        "- Pilot 生成使用固定 P0B Pilot12 病例；图像以 meanbg 原图为基础，保留原图身份、几何和纹理细节。",
        "",
        "## 6. Full-500生成结果",
        f"- 病例数：`{summary.get('case_count')}`。",
        f"- preset数：`{summary.get('preset_count')}`。",
        f"- 总图像数：`{summary.get('relighted_image_count')}`。",
        f"- failure rows：`{summary.get('failure_manifest_rows')}`。",
        "",
        "## 7. 完整性审计",
        f"- finite/shape/dtype/range hard failures：`{summary.get('numeric_hard_failure_count')}`。",
        f"- preset pairing errors：`{summary.get('pair_error_count')}`。",
        f"- background consistency errors：`{summary.get('background_error_count')}`。",
        f"- source assets unchanged：`{summary.get('source_assets_unchanged')}`。",
        "",
        "## 8. P2读取链切换",
    ]
    for ref in config_refs:
        report.append(f"- `{ref['experiment_id']}` manifest: `{ref['manifest_path']}`")
    report.extend(
        [
            "- A2 relighted 输入读取 v2 manifest 中的新 `relighting_npz_path`。",
            "- A3 relighted 分支读取 v2 manifest 中的新 `relighting_npz_path`。",
            f"- original 分支仍读取 `{MEANBG_ROOT}`。",
            "",
            "## 9. 修改文件清单",
            "- `p2_counterfactual/detail_preserving_relighting.py`",
            "- `scripts/p2/run_detail_preserving_relighting_v2.py`",
            "- `config/p2/p2_a/p2_a2_relighting.yaml`",
            "- `config/p2/p2_a/p2_a3_full_consistency.yaml`",
            "- `tests/p2/test_detail_preserving_relighting.py`",
            "",
            "## 10. 最终结论",
            f"- 新资产可供 P2-A2 使用：`{summary.get('status') == 'passed'}`。",
            f"- 新资产可供 P2-A3 使用：`{summary.get('status') == 'passed'}`。",
            f"- 是否已替代旧版 DECA render RGB 正式读取：`{summary.get('checks', {}).get('a2_a3_config_switched')}`。",
            "- 是否运行正式分类训练：`false`。",
        ]
    )
    (OUTPUT_ROOT / "reports/p2_detail_preserving_relighting_report.md").write_text("\n".join(report) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build P2 detail-preserving shading-ratio relighted_images v2.")
    parser.add_argument("--pilot", action="store_true")
    parser.add_argument("--full", action="store_true")
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--freeze", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)
    modes = sum(bool(x) for x in (args.pilot, args.full, args.validate_only, args.freeze))
    if modes != 1:
        parser.error("select exactly one of --pilot, --full, --validate-only, --freeze")
    if args.pilot:
        result = build("pilot", resume=args.resume or True, overwrite=args.overwrite)
    elif args.full:
        result = build("full", resume=args.resume or True, overwrite=args.overwrite)
    elif args.validate_only:
        result = validate_assets(freeze=False)
    else:
        result = validate_assets(freeze=True)
    print(json.dumps(json_safe(result), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
