from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import shutil
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np
import pandas as pd
import yaml
from PIL import Image, ImageDraw
from p2_counterfactual.path_utils import resolve_project_path

ROOT = Path(__file__).resolve().parents[1]
P0_ROOT = ROOT / "data/processed/P0_Physics_Audit_v1"
P1_FROZEN_ROOT = ROOT / "data/processed/P1_DECA_Frozen500_v1"
P1_AUDIT_ROOT = ROOT / "data/processed/P1_Component_Audit_v1"
P2_ROOT = ROOT / "data/processed/P2_Counterfactual_Relighting500_v1"
MASTER_MANIFEST = P1_AUDIT_ROOT / "manifests/p1_master_manifest.csv"
P0_MASTER_INDEX = P0_ROOT / "metadata/master_index.csv"
P1_QC_FLAGS = P1_AUDIT_ROOT / "qc/p1_case_qc_flags.csv"
P0B_PILOT12_MAPPING = ROOT / "data/processed/P0B_DECA_Pilot12_v1/pilot_manifest/p0b_pilot12_id_mapping.csv"

PRESET_NAMES = ("neutral_front", "left", "right", "top", "dim_front", "bright_front")
BOUNDARY_IDS = ("A001917272-1", "A002081031-1")
RENDERER_SOURCE = ROOT / "third_party/DECA/decalib/utils/renderer.py"
P1_GENERATOR_SOURCE = ROOT / "p0b_deca/p1_generation.py"
P0B_SH_SOURCE = ROOT / "p0b_deca/sh_lighting.py"
P2_PROTOCOL_VERSION = "v1"


def utc_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def write_json(path: Path, data: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(json_safe(data), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def json_safe(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    return value


def ensure_dirs() -> None:
    for path in (
        P2_ROOT / "cases",
        P2_ROOT / "manifests",
        P2_ROOT / "metadata",
        P2_ROOT / "qc/sample_contact_sheets",
        P2_ROOT / "reports",
        P2_ROOT / "logs",
    ):
        path.mkdir(parents=True, exist_ok=True)


def read_master_manifest() -> pd.DataFrame:
    df = pd.read_csv(MASTER_MANIFEST, dtype=str)
    required = {
        "case_id",
        "group_id",
        "fold",
        "label_binary",
        "rgb_path",
        "maps_path",
        "relighting_path",
        "quality_path",
        "provenance_path",
        "success_record_path",
    }
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"P1 master manifest missing required columns: {missing}")
    if len(df) != 500 or df["case_id"].nunique() != 500:
        raise ValueError("P1 master manifest must contain 500 unique case_id rows")
    return df


def read_p0_master() -> pd.DataFrame:
    df = pd.read_csv(P0_MASTER_INDEX, dtype=str)
    if "ID" not in df.columns:
        raise ValueError("P0 master_index.csv missing ID")
    return df


def read_qc_flags() -> pd.DataFrame:
    df = pd.read_csv(P1_QC_FLAGS, dtype=str)
    if {"case_id", "flag_count", "case_retained"} - set(df.columns):
        raise ValueError("P1 QC flags file has unexpected schema")
    return df


def as_path(value: str | Path) -> Path:
    return resolve_project_path(value, ROOT)


def npz_schema(path: Path) -> dict[str, Any]:
    result: dict[str, Any] = {}
    with np.load(path, allow_pickle=False) as archive:
        for key in archive.files:
            value = archive[key]
            item: dict[str, Any] = {"shape": list(value.shape), "dtype": str(value.dtype)}
            if value.dtype.kind in "biufc":
                item.update(
                    {
                        "min": float(np.nanmin(value)),
                        "max": float(np.nanmax(value)),
                        "mean": float(np.nanmean(value)),
                        "std": float(np.nanstd(value)),
                        "finite": bool(np.isfinite(value).all()),
                    }
                )
            else:
                item["sample"] = value.tolist() if value.size <= 20 else value.reshape(-1)[:20].tolist()
            result[key] = item
    return result


def get_preset_index(preset_names: Iterable[Any], requested_name: str) -> int:
    names = [str(x) for x in preset_names]
    if len(names) != len(set(names)):
        raise ValueError(f"Duplicate preset names: {names}")
    try:
        return names.index(str(requested_name))
    except ValueError as exc:
        raise KeyError(f"Preset {requested_name!r} is absent from {names}") from exc


def renderer_constant_factor() -> np.ndarray:
    pi = math.pi
    values = [
        1 / math.sqrt(4 * pi),
        ((2 * pi) / 3) * math.sqrt(3 / (4 * pi)),
        ((2 * pi) / 3) * math.sqrt(3 / (4 * pi)),
        ((2 * pi) / 3) * math.sqrt(3 / (4 * pi)),
        (pi / 4) * 3 * math.sqrt(5 / (12 * pi)),
        (pi / 4) * 3 * math.sqrt(5 / (12 * pi)),
        (pi / 4) * 3 * math.sqrt(5 / (12 * pi)),
        (pi / 4) * (3 / 2) * math.sqrt(5 / (12 * pi)),
        (pi / 4) * (1 / 2) * math.sqrt(5 / (4 * pi)),
    ]
    return np.asarray(values, dtype=np.float32)


def deca_add_shlight_numpy(normal_hwc: np.ndarray, sh_coefficients: np.ndarray) -> np.ndarray:
    """Numpy equivalent of DECA ``SRenderY.add_SHlight`` for saved P1 coefficients."""
    normal = np.asarray(normal_hwc, dtype=np.float32)
    coeff = np.asarray(sh_coefficients, dtype=np.float32)
    if normal.shape != (224, 224, 3):
        raise ValueError(f"normal_coarse must be (224,224,3), got {normal.shape}")
    if coeff.shape != (9, 3):
        raise ValueError(f"SH coefficients must be (9,3), got {coeff.shape}")
    x, y, z = np.moveaxis(normal, -1, 0)
    sh = np.stack(
        (
            x * 0.0 + 1.0,
            x,
            y,
            z,
            x * y,
            x * z,
            y * z,
            x * x - y * y,
            3.0 * (z * z) - 1.0,
        ),
        axis=-1,
    ).astype(np.float32)
    sh = sh * renderer_constant_factor()[None, None, :]
    return np.einsum("hwk,kc->hwc", sh, coeff, optimize=True).astype(np.float32)


def derive_relighted_shading(normal_coarse: np.ndarray, sh_coefficients: np.ndarray) -> np.ndarray:
    coeff = np.asarray(sh_coefficients, dtype=np.float32)
    if coeff.shape != (6, 9, 3):
        raise ValueError(f"sh_coefficients must be (6,9,3), got {coeff.shape}")
    return np.stack([deca_add_shlight_numpy(normal_coarse, coeff[index]) for index in range(coeff.shape[0])], axis=0).astype(np.float32)


def source_hashes() -> dict[str, str]:
    files = {
        "p1_master_manifest": MASTER_MANIFEST,
        "p1_deca_output_manifest": P1_FROZEN_ROOT / "manifests/p1_deca_output_manifest.csv",
        "p1_deca_quality_manifest": P1_FROZEN_ROOT / "manifests/p1_deca_quality_manifest.csv",
        "p1_effective_config": P1_FROZEN_ROOT / "metadata/effective_config.yaml",
        "p1_environment": P1_FROZEN_ROOT / "metadata/environment.json",
        "p1_asset_hashes": P1_FROZEN_ROOT / "metadata/asset_hashes.json",
        "p1_deca_output_inventory": P1_FROZEN_ROOT / "metadata/deca_output_inventory.json",
        "p1_offline_integrity_audit": P1_FROZEN_ROOT / "qc/offline_integrity_audit.json",
        "p1_component_source_hashes": P1_AUDIT_ROOT / "metadata/source_hashes.json",
        "p1_qc_flags": P1_QC_FLAGS,
        "p0_master_index": P0_MASTER_INDEX,
        "p0_effective_config": P0_ROOT / "metadata/effective_config.yaml",
        "renderer_py": RENDERER_SOURCE,
        "p1_generation_py": P1_GENERATOR_SOURCE,
        "p0b_sh_lighting_py": P0B_SH_SOURCE,
        "relighting_preset_config": ROOT / "config/p0b/p0b_relighting_presets_v1.yaml",
    }
    return {name: sha256_file(path) if path.is_file() else "missing" for name, path in files.items()}


def write_source_inventory(stage: str) -> dict[str, Any]:
    ensure_dirs()
    path = P2_ROOT / "metadata/source_inventory.json"
    current = source_hashes()
    if path.exists():
        data = json.loads(path.read_text(encoding="utf-8"))
        baseline = data.get("baseline_hashes", current)
    else:
        baseline = current
    unchanged = baseline == current
    data = {
        "stage": stage,
        "created_or_updated_at": utc_now(),
        "baseline_hashes": baseline,
        "current_hashes": current,
        "source_p1_unchanged": unchanged,
        "note": "Hashes are restricted to key metadata/manifests/code files; P2 does not hash all large P1 case NPZ files.",
    }
    write_json(path, data)
    return data


def inspect_protocol() -> dict[str, Any]:
    ensure_dirs()
    df = read_master_manifest()
    first = df.iloc[0]
    maps_path = as_path(first["maps_path"])
    relighting_path = as_path(first["relighting_path"])
    maps = npz_schema(maps_path)
    relighting = npz_schema(relighting_path)
    with np.load(relighting_path, allow_pickle=False) as rel:
        presets = [str(x) for x in rel["preset_names"].tolist()]
    with np.load(maps_path, allow_pickle=False) as m, np.load(relighting_path, allow_pickle=False) as r:
        generated = derive_relighted_shading(m["normal_coarse"], r["sh_coefficients"])
        generated_schema = {
            "shape": list(generated.shape),
            "dtype": str(generated.dtype),
            "min": float(generated.min()),
            "max": float(generated.max()),
            "mean": float(generated.mean()),
            "std": float(generated.std()),
            "finite": bool(np.isfinite(generated).all()),
        }
    code_sha = sha256_text(
        sha256_file(P1_GENERATOR_SOURCE) + sha256_file(RENDERER_SOURCE) + sha256_file(ROOT / "config/p0b/p0b_relighting_presets_v1.yaml")
    )
    protocol = {
        "protocol_version": P2_PROTOCOL_VERSION,
        "source_project_root": str(ROOT),
        "source_generator_script": str(P1_GENERATOR_SOURCE),
        "source_renderer_module": str(RENDERER_SOURCE),
        "source_renderer_function": "SRenderY.add_SHlight and SRenderY.forward",
        "source_code_sha256": code_sha,
        "input_mode": "direct_p0_aligned",
        "normal_source": "normal_coarse",
        "use_detail_normal": False,
        "sh_basis_description": "DECA renderer order: [1, x, y, z, x*y, x*z, y*z, x^2-y^2, 3*z^2-1], multiplied by renderer.constant_factor before summing coefficients.",
        "sh_coefficient_order": "Saved P1 sh_coefficients are renderer-scaled coefficients in DECA add_SHlight order, shape (preset, 9, RGB channel).",
        "channel_order": "RGB",
        "preset_names": presets,
        "albedo_source": "albedo_like",
        "use_signed_residual_in_relighted_rgb": False,
        "use_alpha_composition": True,
        "background_source": "zero background in DECA renderer forward because P1 called model.render without background; images and albedo_images are multiplied by alpha_images.",
        "clip_range": [0.0, 1.0],
        "stored_dtype": "float32",
        "stored_color_space": "linear-like float RGB arrays from DECA/P0 pipeline, not gamma-normalized by P2",
        "relighted_images_training_ready": True,
        "relighted_images_key": "relighted_images",
        "relighted_images_complete_rgb": True,
        "relighted_images_contains_original_background": False,
        "alpha_already_applied_to_relighted_images": True,
        "double_alpha_composition_risk": True,
        "relighted_rgb_uses_signed_residual": False,
        "original_shading_like_same_function": True,
        "output_clip_in_p1_npz": False,
        "output_rgb_or_bgr": "RGB",
        "array_storage_space": "NHWC float32; relighted_images stored raw/unclipped except zero background alpha multiplication",
        "inspected_case_id": str(first["case_id"]),
        "maps_npz_schema": maps,
        "relighting_npz_schema": relighting,
        "p2_relighted_shading_schema_from_inspected_case": generated_schema,
        "evidence_files": [
            str(P1_GENERATOR_SOURCE),
            str(RENDERER_SOURCE),
            str(ROOT / "config/p0b/p0b_relighting_presets_v1.yaml"),
            str(P1_FROZEN_ROOT / "metadata/effective_config.yaml"),
            str(P1_FROZEN_ROOT / "metadata/deca_output_inventory.json"),
            str(maps_path),
            str(relighting_path),
        ],
    }
    write_json(P2_ROOT / "metadata/P2_Relighting_Protocol_v1.json", protocol)
    write_json(
        P2_ROOT / "metadata/preset_registry.json",
        {
            "preset_names": list(PRESET_NAMES),
            "source": str(ROOT / "config/p0b/p0b_relighting_presets_v1.yaml"),
            "source_sha256": sha256_file(ROOT / "config/p0b/p0b_relighting_presets_v1.yaml"),
            "index_policy": "Use get_preset_index(preset_names, requested_name); do not hard-code numeric positions.",
        },
    )
    config = {
        "project_root": str(ROOT),
        "p0_root": str(P0_ROOT),
        "p1_frozen_root": str(P1_FROZEN_ROOT),
        "p1_audit_root": str(P1_AUDIT_ROOT),
        "p2_output_root": str(P2_ROOT),
        "master_manifest": str(MASTER_MANIFEST),
        "case_id_column": "case_id",
        "patient_group_column": "group_id",
        "fold_column": "fold",
        "binary_label_column": "label_binary",
        "preset_names": list(PRESET_NAMES),
        "source_read_only": True,
        "overwrite": False,
        "resume": True,
        "seed": 2026,
    }
    (P2_ROOT / "metadata/derivation_config.yaml").write_text(yaml.safe_dump(config, sort_keys=False, allow_unicode=True), encoding="utf-8")
    write_source_inventory("inspect-only")
    return protocol


def success_valid(case_dir: Path, maps_path: Path, relighting_path: Path) -> bool:
    success = case_dir / "_SUCCESS.json"
    output = case_dir / "relit_shading.npz"
    if not success.is_file() or not output.is_file():
        return False
    try:
        data = json.loads(success.read_text(encoding="utf-8"))
        if data.get("source_maps_sha256") != sha256_file(maps_path):
            return False
        if data.get("source_relighting_sha256") != sha256_file(relighting_path):
            return False
        if data.get("output_sha256") != sha256_file(output):
            return False
        with np.load(output, allow_pickle=False) as archive:
            arr = archive["relighted_shading"]
            return arr.shape == (6, 224, 224, 3) and arr.dtype == np.float32 and bool(np.isfinite(arr).all())
    except Exception:
        return False


def build_case(row: Mapping[str, Any], *, resume: bool = True) -> dict[str, Any]:
    cid = str(row["case_id"])
    maps_path = as_path(str(row["maps_path"]))
    relighting_path = as_path(str(row["relighting_path"]))
    case_dir = P2_ROOT / "cases" / cid
    output_path = case_dir / "relit_shading.npz"
    if resume and success_valid(case_dir, maps_path, relighting_path):
        return {"case_id": cid, "status": "skipped_valid_resume", "output_path": str(output_path)}
    case_dir.mkdir(parents=True, exist_ok=True)
    tmp = case_dir / f".tmp_relit_shading_{time.time_ns()}.npz"
    try:
        with np.load(maps_path, allow_pickle=False) as maps, np.load(relighting_path, allow_pickle=False) as rel:
            normal = np.asarray(maps["normal_coarse"], dtype=np.float32)
            preset_names = np.asarray([str(x) for x in rel["preset_names"].tolist()])
            coefficients = np.asarray(rel["sh_coefficients"], dtype=np.float32)
            if tuple(preset_names.tolist()) != tuple(PRESET_NAMES):
                if sorted(preset_names.tolist()) != sorted(PRESET_NAMES):
                    raise ValueError(f"{cid}: preset names do not match registry: {preset_names.tolist()}")
            shading = derive_relighted_shading(normal, coefficients)
        if shading.shape != (6, 224, 224, 3) or shading.dtype != np.float32 or not np.isfinite(shading).all():
            raise ValueError(f"{cid}: invalid relighted_shading")
        np.savez_compressed(tmp, preset_names=preset_names, sh_coefficients=coefficients, relighted_shading=shading)
        output_path.replace(output_path.with_suffix(".npz.bak")) if output_path.exists() else None
        tmp.replace(output_path)
        backup = output_path.with_suffix(".npz.bak")
        if backup.exists():
            backup.unlink()
        success = {
            "case_id": cid,
            "source_maps_path": str(maps_path),
            "source_relighting_path": str(relighting_path),
            "source_maps_sha256": sha256_file(maps_path),
            "source_relighting_sha256": sha256_file(relighting_path),
            "output_path": str(output_path),
            "output_sha256": sha256_file(output_path),
            "preset_names": preset_names.tolist(),
            "shape": list(shading.shape),
            "dtype": str(shading.dtype),
            "finite": bool(np.isfinite(shading).all()),
            "completed_at": utc_now(),
        }
        write_json(case_dir / "_SUCCESS.json", success)
        return {"case_id": cid, "status": "success", "output_path": str(output_path)}
    except Exception as exc:
        if tmp.exists():
            tmp.unlink()
        return {"case_id": cid, "status": "failed", "error": f"{type(exc).__name__}: {exc}"}


def select_rows(mode: str) -> pd.DataFrame:
    df = read_master_manifest()
    if mode == "pilot12":
        if P0B_PILOT12_MAPPING.is_file():
            pilot = pd.read_csv(P0B_PILOT12_MAPPING, dtype=str)
            column = "sample_id" if "sample_id" in pilot.columns else pilot.columns[0]
            ids = [str(x) for x in pilot[column].tolist()]
            selected = df[df["case_id"].isin(ids)].copy()
            if len(selected) == 12:
                return selected
        rng = np.random.default_rng(2026)
        boundary = df[df["case_id"].isin(BOUNDARY_IDS)]
        rest = df[~df["case_id"].isin(set(boundary["case_id"]))].copy()
        take = rest.iloc[rng.choice(len(rest), size=12 - len(boundary), replace=False)]
        return pd.concat([boundary, take], ignore_index=True)
    if mode == "full-500":
        return df
    raise ValueError(f"Unknown selection mode: {mode}")


def build_shading(mode: str, *, resume: bool = True) -> dict[str, Any]:
    ensure_dirs()
    rows = select_rows(mode).to_dict(orient="records")
    outcomes = [build_case(row, resume=resume) for row in rows]
    failures = [item for item in outcomes if item["status"] == "failed"]
    summary = {
        "mode": mode,
        "expected_cases": len(rows),
        "success_or_resume": sum(item["status"] in {"success", "skipped_valid_resume"} for item in outcomes),
        "new_success": sum(item["status"] == "success" for item in outcomes),
        "failed": len(failures),
        "failures": failures,
        "completed_at": utc_now(),
    }
    write_json(P2_ROOT / f"logs/{mode}_build_summary.json", summary)
    return summary


def build_manifest() -> tuple[pd.DataFrame, pd.DataFrame]:
    ensure_dirs()
    df = read_master_manifest().copy()
    p0 = read_p0_master()[["ID", "jaw_neck_boundary_ambiguous", "boundary_ambiguity_status"]].copy()
    p0 = p0.rename(columns={"ID": "case_id"})
    qc = read_qc_flags()[["case_id", "flag_count", "flags", "case_retained"]].copy()
    merged = df.merge(p0, on="case_id", how="left").merge(qc, on="case_id", how="left")
    rows: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    for row in merged.to_dict(orient="records"):
        cid = str(row["case_id"])
        maps_path = as_path(row["maps_path"])
        relighting_path = as_path(row["relighting_path"])
        relighted_shading_path = P2_ROOT / "cases" / cid / "relit_shading.npz"
        source_valid = maps_path.is_file() and relighting_path.is_file()
        shading_valid = False
        preset_names: list[str] = []
        error = ""
        try:
            with np.load(relighting_path, allow_pickle=False) as rel:
                preset_names = [str(x) for x in rel["preset_names"].tolist()]
            if relighted_shading_path.is_file():
                with np.load(relighted_shading_path, allow_pickle=False) as p2:
                    shading = p2["relighted_shading"]
                    p2_names = [str(x) for x in p2["preset_names"].tolist()]
                    shading_valid = (
                        shading.shape == (6, 224, 224, 3)
                        and shading.dtype == np.float32
                        and np.isfinite(shading).all()
                        and p2_names == preset_names
                    )
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
        ready = bool(source_valid and shading_valid)
        if not ready:
            failures.append({"case_id": cid, "source_asset_valid": source_valid, "relighted_shading_valid": shading_valid, "error": error})
        rows.append(
            {
                "case_id": cid,
                "patient_group_id": str(row["group_id"]),
                "fold": int(row["fold"]),
                "binary_label": int(row["label_binary"]),
                "original_rgb_path": str(as_path(row["rgb_path"])),
                "maps_npz_path": str(maps_path),
                "relighting_npz_path": str(relighting_path),
                "relighted_shading_path": str(relighted_shading_path),
                "original_shading_key": "shading_like",
                "original_residual_key": "signed_residual",
                "relighted_rgb_key": "relighted_images",
                "relighted_shading_key": "relighted_shading",
                "preset_names": json.dumps(preset_names, ensure_ascii=False),
                "p1_qc_flag": int(float(row["flag_count"])) > 0 if str(row.get("flag_count", "")).strip() not in {"", "nan"} else False,
                "boundary_uncertain": str(row.get("jaw_neck_boundary_ambiguous", "")).lower() in {"1", "true", "yes"}
                or str(row.get("boundary_ambiguity_status", "")) == "jaw_neck_ambiguous",
                "source_asset_valid": bool(source_valid),
                "relighted_shading_valid": bool(shading_valid),
                "p2_training_ready": ready,
            }
        )
    manifest = pd.DataFrame(rows)
    failures_df = pd.DataFrame(failures, columns=["case_id", "source_asset_valid", "relighted_shading_valid", "error"])
    manifest.to_csv(P2_ROOT / "manifests/p2_training_manifest.csv", index=False, encoding="utf-8")
    failures_df.to_csv(P2_ROOT / "manifests/p2_failure_manifest.csv", index=False, encoding="utf-8")
    return manifest, failures_df


@dataclass
class P2CounterfactualDataset:
    manifest: pd.DataFrame
    mode: str = "original"
    preset_name: str = "neutral_front"

    @classmethod
    def from_csv(cls, path: Path, mode: str = "original", preset_name: str = "neutral_front") -> "P2CounterfactualDataset":
        return cls(pd.read_csv(path, dtype={"case_id": str}), mode=mode, preset_name=preset_name)

    def __len__(self) -> int:
        return len(self.manifest)

    @staticmethod
    def _chw(value: np.ndarray) -> np.ndarray:
        array = np.asarray(value, dtype=np.float32)
        if array.shape != (224, 224, 3):
            raise ValueError(f"Expected HWC RGB-like array, got {array.shape}")
        return np.transpose(array, (2, 0, 1)).astype(np.float32)

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.manifest.iloc[int(index)]
        maps_path = resolve_project_path(row["maps_npz_path"], ROOT, require_exists=True, case_id=str(row["case_id"]), column_name="maps_npz_path")
        relighting_path = resolve_project_path(row["relighting_npz_path"], ROOT, require_exists=True, case_id=str(row["case_id"]), column_name="relighting_npz_path")
        relighted_shading_path = resolve_project_path(row["relighted_shading_path"], ROOT, require_exists=True, case_id=str(row["case_id"]), column_name="relighted_shading_path")
        with np.load(maps_path, allow_pickle=False) as maps:
            residual = self._chw(maps["signed_residual"])
            if self.mode == "original":
                return {
                    "case_id": str(row["case_id"]),
                    "original_rgb": self._chw(maps["input_aligned_rgb"]),
                    "original_shading": self._chw(maps["shading_like"]),
                    "original_residual": residual,
                    "label": int(row["binary_label"]),
                    "fold": int(row["fold"]),
                }
        if self.mode != "counterfactual":
            raise ValueError(f"Unknown dataset mode: {self.mode}")
        with np.load(relighting_path, allow_pickle=False) as rel, np.load(relighted_shading_path, allow_pickle=False) as shade:
            rel_names = [str(x) for x in rel["preset_names"].tolist()]
            shade_names = [str(x) for x in shade["preset_names"].tolist()]
            if rel_names != shade_names:
                raise ValueError(f"Preset mismatch for case {row['case_id']}")
            idx = get_preset_index(rel_names, self.preset_name)
            return {
                "case_id": str(row["case_id"]),
                "relighted_rgb_k": self._chw(rel["relighted_images"][idx]),
                "relighted_shading_k": self._chw(shade["relighted_shading"][idx]),
                "original_residual": residual,
                "label": int(row["binary_label"]),
                "fold": int(row["fold"]),
                "preset_name": rel_names[idx],
            }


def contact_sheet(case_ids: list[str], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = P2_ROOT / "manifests/p2_training_manifest.csv"
    if not manifest_path.is_file():
        return
    frame = pd.read_csv(manifest_path, dtype={"case_id": str})
    by_id = {str(row.case_id): row for row in frame.itertuples()}
    for cid in case_ids:
        if cid not in by_id:
            continue
        row = by_id[cid]
        with np.load(row.relighting_npz_path, allow_pickle=False) as rel, np.load(row.relighted_shading_path, allow_pickle=False) as shade:
            names = [str(x) for x in rel["preset_names"].tolist()]
            rgb = rel["relighted_images"]
            shading = shade["relighted_shading"]
        panel = Image.new("RGB", (7 * 224, 2 * 244), "white")
        draw = ImageDraw.Draw(panel)
        raw = Image.open(row.original_rgb_path).convert("RGB").resize((224, 224))
        panel.paste(raw, (0, 20))
        draw.text((4, 3), "original", fill="black")
        for idx, name in enumerate(names):
            x = (idx + 1) * 224
            rgb_img = Image.fromarray(np.rint(np.clip(rgb[idx], 0, 1) * 255).astype(np.uint8), "RGB")
            sh = shading[idx]
            sh_norm = (sh - np.nanmin(sh)) / max(float(np.nanmax(sh) - np.nanmin(sh)), 1e-6)
            sh_img = Image.fromarray(np.rint(np.clip(sh_norm, 0, 1) * 255).astype(np.uint8), "RGB")
            panel.paste(rgb_img, (x, 20))
            panel.paste(sh_img, (x, 264))
            draw.text((x + 4, 3), name[:22], fill="black")
        panel.save(output_dir / f"{cid}.png")


def validate_assets(*, write_report: bool = True, freeze: bool = False) -> dict[str, Any]:
    ensure_dirs()
    if not (P2_ROOT / "metadata/P2_Relighting_Protocol_v1.json").is_file():
        inspect_protocol()
    manifest, failures = build_manifest()
    stats_rows: list[dict[str, Any]] = []
    pair_errors: list[str] = []
    preset_errors: list[str] = []
    numeric_errors: list[str] = []
    directional_rows: list[dict[str, Any]] = []
    for row in manifest.to_dict(orient="records"):
        cid = str(row["case_id"])
        try:
            with np.load(row["relighting_npz_path"], allow_pickle=False) as rel, np.load(row["relighted_shading_path"], allow_pickle=False) as shade:
                rel_names = [str(x) for x in rel["preset_names"].tolist()]
                shade_names = [str(x) for x in shade["preset_names"].tolist()]
                if rel_names != shade_names:
                    pair_errors.append(cid)
                if set(rel_names) != set(PRESET_NAMES) or len(rel_names) != 6 or len(set(rel_names)) != 6:
                    preset_errors.append(cid)
                arr = shade["relighted_shading"]
                finite = bool(np.isfinite(arr).all())
                all_zero = bool(np.all(arr == 0))
                all_constant = bool(np.nanstd(arr) == 0.0)
                stat = {
                    "case_id": cid,
                    "shape": list(arr.shape),
                    "dtype": str(arr.dtype),
                    "finite": finite,
                    "min": float(np.nanmin(arr)),
                    "max": float(np.nanmax(arr)),
                    "mean": float(np.nanmean(arr)),
                    "std": float(np.nanstd(arr)),
                    "empty": bool(arr.size == 0),
                    "all_zero": all_zero,
                    "all_constant": all_constant,
                }
                stats_rows.append(stat)
                if arr.shape != (6, 224, 224, 3) or arr.dtype != np.float32 or not finite or arr.size == 0 or all_zero or all_constant:
                    numeric_errors.append(cid)
                lookup = {name: get_preset_index(rel_names, name) for name in PRESET_NAMES}
                means = {name: float(arr[idx].mean()) for name, idx in lookup.items()}
                left = arr[lookup["left"]]
                right = arr[lookup["right"]]
                left_half_mean = float(left[:, :112, :].mean())
                left_right_half_mean = float(left[:, 112:, :].mean())
                right_left_half_mean = float(right[:, :112, :].mean())
                right_half_mean = float(right[:, 112:, :].mean())
                directional_rows.append(
                    {
                        "case_id": cid,
                        **{f"mean_{name}": means[name] for name in PRESET_NAMES},
                        "dim_lt_neutral_lt_bright": bool(means["dim_front"] < means["neutral_front"] < means["bright_front"]),
                        "left_half_gt_right_half_for_left": bool(left_half_mean > left_right_half_mean),
                        "right_half_gt_left_half_for_right": bool(right_half_mean > right_left_half_mean),
                        "left_preset_left_half_mean": left_half_mean,
                        "left_preset_right_half_mean": left_right_half_mean,
                        "right_preset_left_half_mean": right_left_half_mean,
                        "right_preset_right_half_mean": right_half_mean,
                    }
                )
        except Exception as exc:
            numeric_errors.append(cid)
            stats_rows.append({"case_id": cid, "error": f"{type(exc).__name__}: {exc}"})
    stats_df = pd.DataFrame(stats_rows)
    direction_df = pd.DataFrame(directional_rows)
    stats_df.to_csv(P2_ROOT / "qc/relighted_shading_numeric_stats.csv", index=False, encoding="utf-8")
    direction_df.to_csv(P2_ROOT / "qc/directionality_smoke_stats.csv", index=False, encoding="utf-8")
    dataset_smoke = dataset_smoke_test(manifest)
    source = write_source_inventory("freeze" if freeze else "validate-only")
    qc_flag_count = int(manifest["p1_qc_flag"].sum())
    boundary_count = int(manifest["boundary_uncertain"].sum())
    warnings: list[str] = []
    if not direction_df.empty:
        dim_rate = float(direction_df["dim_lt_neutral_lt_bright"].mean())
        left_rate = float(direction_df["left_half_gt_right_half_for_left"].mean())
        right_rate = float(direction_df["right_half_gt_left_half_for_right"].mean())
        if dim_rate < 0.95:
            warnings.append(f"dim/neutral/bright trend pass rate {dim_rate:.3f}")
        if left_rate < 0.50 or right_rate < 0.50:
            warnings.append(f"possible left/right systematic reversal: left_rate={left_rate:.3f}, right_rate={right_rate:.3f}")
    else:
        dim_rate = left_rate = right_rate = 0.0
        warnings.append("No directionality rows available")
    hard_failures = []
    checks = {
        "manifest_500_rows": len(manifest) == 500,
        "unique_500_cases": manifest["case_id"].nunique() == 500,
        "relit_shading_500_valid": int(manifest["relighted_shading_valid"].sum()) == 500,
        "relighted_shading_3000_complete": int(manifest["relighted_shading_valid"].sum()) * 6 == 3000,
        "preset_complete_unique": not preset_errors,
        "rgb_shading_preset_names_match": not pair_errors,
        "all_new_arrays_finite_shape_dtype_nonconstant": not numeric_errors,
        "dataset_smoke_passed": dataset_smoke["status"] == "passed",
        "source_p1_unchanged": bool(source["source_p1_unchanged"]),
        "qc_flagged_cases_retained": qc_flag_count == 16,
        "boundary_uncertain_cases_retained": boundary_count == 2,
        "classification_training_executed": False,
        "deca_encoder_executed": False,
    }
    for key, ok in checks.items():
        if key in {"classification_training_executed", "deca_encoder_executed"}:
            if ok:
                hard_failures.append(key)
        elif not ok:
            hard_failures.append(key)
    if not direction_df.empty and (left_rate < 0.25 or right_rate < 0.25):
        hard_failures.append("systematic_left_right_direction_reversal")
    summary = {
        "status": "passed" if not hard_failures else "failed",
        "created_at": utc_now(),
        "case_count": int(len(manifest)),
        "unique_case_count": int(manifest["case_id"].nunique()),
        "ready_case_count": int(manifest["p2_training_ready"].sum()),
        "relighted_shading_count": int(manifest["relighted_shading_valid"].sum()) * 6,
        "failure_manifest_rows": int(len(failures)),
        "pair_error_count": len(pair_errors),
        "numeric_error_count": len(numeric_errors),
        "preset_error_count": len(preset_errors),
        "directionality": {
            "dim_lt_neutral_lt_bright_case_rate": dim_rate,
            "left_half_gt_right_half_for_left_case_rate": left_rate,
            "right_half_gt_left_half_for_right_case_rate": right_rate,
        },
        "dataset_smoke": dataset_smoke,
        "qc_flagged_case_count": qc_flag_count,
        "boundary_uncertain_case_count": boundary_count,
        "source_p1_unchanged": bool(source["source_p1_unchanged"]),
        "classification_training_executed": False,
        "deca_encoder_executed": False,
        "hard_failures": hard_failures,
        "warnings": warnings,
    }
    write_json(P2_ROOT / "metadata/validation_summary.json", summary)
    if write_report:
        write_report_md(summary)
    if freeze:
        write_freeze(summary)
    contact_ids = [str(x) for x in manifest["case_id"].tolist()[:6]]
    contact_sheet(contact_ids, P2_ROOT / "qc/sample_contact_sheets")
    return summary


def dataset_smoke_test(manifest: pd.DataFrame) -> dict[str, Any]:
    try:
        sample = manifest[manifest["p2_training_ready"] == True].head(3).copy()  # noqa: E712
        if sample.empty:
            sample = manifest.head(3).copy()
        original = P2CounterfactualDataset(sample, mode="original")[0]
        counter = P2CounterfactualDataset(sample, mode="counterfactual", preset_name="left")[0]
        checks = {
            "original_rgb_shape": list(original["original_rgb"].shape),
            "original_shading_shape": list(original["original_shading"].shape),
            "original_residual_shape": list(original["original_residual"].shape),
            "relighted_rgb_shape": list(counter["relighted_rgb_k"].shape),
            "relighted_shading_shape": list(counter["relighted_shading_k"].shape),
            "counter_residual_shape": list(counter["original_residual"].shape),
            "preset_name": counter["preset_name"],
            "residual_uses_signed_residual": True,
            "rgb_shading_same_preset": counter["preset_name"] == "left",
        }
        shape_ok = all(value == [3, 224, 224] for key, value in checks.items() if key.endswith("_shape"))
        status = "passed" if shape_ok and checks["rgb_shading_same_preset"] else "failed"
        return {"status": status, "checked_cases": int(len(sample)), "checks": checks}
    except Exception as exc:
        return {"status": "failed", "error": f"{type(exc).__name__}: {exc}"}


def write_report_md(summary: Mapping[str, Any]) -> None:
    protocol_path = P2_ROOT / "metadata/P2_Relighting_Protocol_v1.json"
    protocol = json.loads(protocol_path.read_text(encoding="utf-8")) if protocol_path.is_file() else {}
    lines = [
        "# P2-0 Asset Preparation Report",
        "",
        f"Generated at: `{utc_now()}`",
        "",
        "## Checked Existing Code And Data",
        "",
        f"- P1 generator: `{P1_GENERATOR_SOURCE}`",
        f"- DECA renderer: `{RENDERER_SOURCE}`",
        f"- P1 master manifest: `{MASTER_MANIFEST}`",
        f"- P1 frozen root: `{P1_FROZEN_ROOT}`",
        "",
        "## Real Relighting Interface",
        "",
        f"- Renderer function: `{protocol.get('source_renderer_function', 'unknown')}`",
        f"- SH basis: {protocol.get('sh_basis_description', 'unknown')}",
        "- Saved `relighting.npz` fields: `preset_names`, `sh_coefficients`, `relighted_images`, `preset_quality_json`.",
        "- `relighted_images` is complete RGB from DECA renderer `images`, stored NHWC float32 and training-ready for RGB relighting reads.",
        "- P1 called `model.render` without a background, so alpha was applied against zero background; it does not contain original RGB background.",
        "- P1 relighted RGB did not add `signed_residual`.",
        "",
        "## Generated Shading",
        "",
        "- P2 generated `relighted_shading` from `maps.npz/normal_coarse` and `relighting.npz/sh_coefficients` using a numpy equivalent of DECA `add_SHlight`.",
        f"- Ready cases: `{summary.get('ready_case_count')}/500`; relighted shading views: `{summary.get('relighted_shading_count')}/3000`.",
        "",
        "## Validation",
        "",
        f"- Preset pair errors: `{summary.get('pair_error_count')}`",
        f"- Numeric errors: `{summary.get('numeric_error_count')}`",
        f"- Preset registry errors: `{summary.get('preset_error_count')}`",
        f"- DataLoader smoke: `{summary.get('dataset_smoke', {}).get('status')}`",
        f"- Directionality dim<neutral<bright rate: `{summary.get('directionality', {}).get('dim_lt_neutral_lt_bright_case_rate')}`",
        f"- Directionality left preset left-half rate: `{summary.get('directionality', {}).get('left_half_gt_right_half_for_left_case_rate')}`",
        f"- Directionality right preset right-half rate: `{summary.get('directionality', {}).get('right_half_gt_left_half_for_right_case_rate')}`",
        f"- P1 source unchanged: `{summary.get('source_p1_unchanged')}`",
        f"- QC flagged retained: `{summary.get('qc_flagged_case_count')}`",
        f"- Boundary uncertain retained: `{summary.get('boundary_uncertain_case_count')}`",
        "",
        "## Warnings",
        "",
    ]
    warnings = list(summary.get("warnings", []))
    lines.extend([f"- {warning}" for warning in warnings] or ["- none"])
    lines.extend(
        [
            "",
            "## Freeze Decision",
            "",
            f"- Status: `{summary.get('status')}`",
            f"- Hard failures: `{summary.get('hard_failures')}`",
            "- DECA encoder executed: `false`",
            "- Classification training executed: `false`",
        ]
    )
    (P2_ROOT / "reports/p2_0_asset_preparation_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_freeze(validation_summary: Mapping[str, Any]) -> dict[str, Any]:
    manifest_path = P2_ROOT / "manifests/p2_training_manifest.csv"
    protocol_path = P2_ROOT / "metadata/P2_Relighting_Protocol_v1.json"
    registry_path = P2_ROOT / "metadata/preset_registry.json"
    inventory_path = P2_ROOT / "metadata/source_inventory.json"
    status = "P2_COUNTERFACTUAL_TRAINING_ASSETS_FROZEN" if validation_summary.get("status") == "passed" else "P2_COUNTERFACTUAL_TRAINING_ASSETS_NOT_FROZEN"
    freeze = {
        "status": status,
        "created_at": utc_now(),
        "protocol_version": P2_PROTOCOL_VERSION,
        "case_count": int(validation_summary.get("case_count", 0)),
        "preset_count": 6,
        "relighted_shading_count": int(validation_summary.get("relighted_shading_count", 0)),
        "manifest_sha256": sha256_file(manifest_path) if manifest_path.is_file() else "missing",
        "protocol_sha256": sha256_file(protocol_path) if protocol_path.is_file() else "missing",
        "preset_registry_sha256": sha256_file(registry_path) if registry_path.is_file() else "missing",
        "source_inventory_sha256": sha256_file(inventory_path) if inventory_path.is_file() else "missing",
        "hard_failures": list(validation_summary.get("hard_failures", [])),
        "warnings": list(validation_summary.get("warnings", [])),
        "source_p1_unchanged": bool(validation_summary.get("source_p1_unchanged")),
        "classification_training_executed": False,
        "deca_encoder_executed": False,
    }
    write_json(P2_ROOT / "metadata/P2_COUNTERFACTUAL_TRAINING_ASSETS_FROZEN.json", freeze)
    return freeze


def run_all() -> dict[str, Any]:
    inspect_protocol()
    pilot = build_shading("pilot12", resume=True)
    pilot_validation = validate_assets(write_report=True, freeze=False)
    if pilot.get("failed"):
        raise RuntimeError(f"Pilot12 failed: {pilot['failures'][:3]}")
    full = build_shading("full-500", resume=True)
    validation = validate_assets(write_report=True, freeze=False)
    frozen_validation = validate_assets(write_report=True, freeze=True)
    return {"pilot": pilot, "pilot_validation": pilot_validation, "full": full, "validation": validation, "freeze_validation": frozen_validation}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Prepare P2-0 counterfactual relighting training assets.")
    parser.add_argument("--inspect-only", action="store_true")
    parser.add_argument("--pilot12", action="store_true")
    parser.add_argument("--full-500", action="store_true")
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--freeze", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--all", action="store_true")
    args = parser.parse_args(argv)
    selected = sum(bool(x) for x in (args.inspect_only, args.pilot12, args.full_500, args.validate_only, args.freeze, args.all))
    if selected != 1:
        parser.error("select exactly one mode")
    if args.inspect_only:
        result = inspect_protocol()
    elif args.pilot12:
        result = build_shading("pilot12", resume=args.resume or True)
        build_manifest()
    elif args.full_500:
        result = build_shading("full-500", resume=args.resume or True)
        build_manifest()
    elif args.validate_only:
        result = validate_assets(write_report=True, freeze=False)
    elif args.freeze:
        result = validate_assets(write_report=True, freeze=True)
    else:
        result = run_all()
    print(json.dumps(json_safe(result), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
