"""Read-only validation helpers for the P1 component framework."""

from __future__ import annotations

import json
import csv
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np
import pandas as pd
import yaml
from PIL import Image

from datasets.p1_component_dataset import P1ComponentDataset, build_component_dataset
from models.p1_component_models import build_p1_component_model
from utils.p1_component_registry import (
    EXPERIMENT_ORDER,
    get_component_spec,
    get_component_registry,
    runnable_component_keys,
    validate_component_registry,
)
from utils.p1_representation_normalization import (
    build_training_collate,
    fit_component_normalization,
    inspect_normal_flip_convention,
)
from utils.p1_rgb_audit import load_p1_frame, local_path, sha


ROOT = Path(__file__).resolve().parents[1]
MANIFEST_PATH = ROOT / "data/processed/P1_Component_Audit_v1/manifests/p1_master_manifest.csv"
FIXED_SPLIT_PATH = ROOT / "data/processed/P0_Physics_Audit_v1/splits_500/nyha_3class_sex_stratified_group_5fold.csv"
BASELINE_ROOT = ROOT / "experiments/500Data/P1_RGB_P0Aligned_ResNet18_5fold_v1"
FROZEN_CONTRACT_PATH = ROOT / "data/processed/P0B_cross_process_production_gate_v2_1/metadata/FROZEN.json"
CORRECTED_PROTOCOL_PATH = BASELINE_ROOT / "metadata/evaluation_protocol_correction.json"
LONGITUDINAL_SUMMARY_PATH = BASELINE_ROOT / "metadata/patient_group_longitudinal_label_summary.json"
NORMAL_CONVENTION_FILENAME = "normal_coordinate_convention.json"
NORMAL_REAL_ASSET_VALIDATION_FILENAME = "normal_flip_real_asset_validation.csv"
NORMAL_REAL_ASSET_SUMMARY_FILENAME = "normal_flip_real_asset_summary.json"


def _component_flip_policy(config: Mapping[str, Any], experiment_key: str) -> dict[str, Any]:
    experiments = config.get("experiments", {})
    exp_cfg = experiments.get(experiment_key, {}) if isinstance(experiments, Mapping) else {}
    flip_cfg = exp_cfg.get("horizontal_flip", {}) if isinstance(exp_cfg, Mapping) else {}
    enabled = bool(flip_cfg.get("enabled", False))
    mode = str(flip_cfg.get("mode", "disabled_due_to_unverified_coordinate_convention"))
    coordinate_status = str(flip_cfg.get("coordinate_status", "unverified")).upper()
    return {
        "enabled": enabled,
        "mode": mode,
        "coordinate_status": coordinate_status,
    }


def load_sweep_config(config_path: str | Path) -> dict[str, Any]:
    path = Path(config_path)
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("sweep config must be a mapping")
    return payload


def sha256_file(path: str | Path) -> str:
    return sha(Path(path))


def _resolve_path(value: str | Path | None, default: Path) -> Path:
    if value is None or value == "":
        return default
    path = Path(value)
    return path if path.is_absolute() else (ROOT / path)


def _finite_tensor(value: Any) -> bool:
    array = np.asarray(value)
    return bool(np.isfinite(array).all())


def _load_frame(config: Mapping[str, Any]) -> pd.DataFrame:
    manifest = _resolve_path(config.get("manifest_path"), MANIFEST_PATH)
    split = _resolve_path(config.get("fixed_split_path"), FIXED_SPLIT_PATH)
    return load_p1_frame(manifest, split)


def _check_common_contracts(config: Mapping[str, Any], frame: pd.DataFrame) -> list[str]:
    blockers: list[str] = []
    if len(frame) != 500 or frame.case_id.nunique() != 500:
        blockers.append("manifest_rows_not_500")
    if frame.groupby("patient_group_id")["fold"].nunique().gt(1).any():
        blockers.append("patient_group_crosses_folds")
    if not FROZEN_CONTRACT_PATH.is_file():
        blockers.append("frozen_contract_missing")
    if not CORRECTED_PROTOCOL_PATH.is_file():
        blockers.append("corrected_protocol_missing")
    if not LONGITUDINAL_SUMMARY_PATH.is_file():
        blockers.append("longitudinal_summary_missing")
    if sha256_file(_resolve_path(config.get("fixed_split_path"), FIXED_SPLIT_PATH)) != config.get(
        "fixed_split_sha256",
        "d5a20fb56c96e657dd7902b6d829bed78b6d43d3e58ec47e6bd3542ec34378cb",
    ):
        blockers.append("fixed_split_sha256_mismatch")
    return blockers


def _check_component_contract(
    config: Mapping[str, Any],
    frame: pd.DataFrame,
    experiment_key: str,
    *,
    sample_count: int = 3,
    root: str | Path | None = None,
) -> dict[str, Any]:
    spec = get_component_spec(experiment_key)
    if spec.key == "p1_spec":
        return {
            "experiment_key": spec.key,
            "display_name": spec.display_name,
            "representation": spec.representation,
            "input_type": spec.input_type,
            "model_type": spec.model_type,
            "status": "SKIPPED_UNAVAILABLE_BY_CURRENT_FRONTEND",
            "reason": spec.skip_reason,
            "availability": spec.availability,
            "normalization": None,
            "normal_flip": {"status": "UNVERIFIED", "reason": spec.skip_reason or ""},
        }
    subset = frame.head(sample_count).copy()
    dataset = build_component_dataset(subset, spec.key)
    normalization = fit_component_normalization(subset, spec.key, fold=None, root=root)
    flip_policy = _component_flip_policy(config, spec.key)
    samples = [dataset[index] for index in range(len(dataset))]
    normalized = [build_training_collate(spec.key, normalization)([sample]) for sample in samples]
    validation_rows = []
    for sample, batch in zip(samples, normalized, strict=True):
        rep = sample["representation"]
        mask = sample.get("valid_mask")
        if spec.key == "p1_rgb_a":
            rgb = batch["representation"]["rgb"]
            albedo = batch["representation"]["albedo"]
            validation_rows.append(
                {
                    "case_id": sample["case_id"],
                    "representation_shape": [list(rgb.shape), list(albedo.shape)],
                    "mask_shape": None if mask is None else list(mask.shape),
                    "finite": _finite_tensor(rgb) and _finite_tensor(albedo),
                }
            )
        else:
            tensor = batch["representation"]
            validation_rows.append(
                {
                    "case_id": sample["case_id"],
                    "representation_shape": list(tensor.shape),
                    "mask_shape": None if mask is None else list(mask.shape),
                    "finite": _finite_tensor(tensor),
                }
            )
    normal_flip = inspect_normal_flip_convention(root)
    normal_flip_contract = {
        "status": normalization.normal_flip_status if spec.key == "p1_n" else normal_flip.status,
        "coordinate_status": normalization.normal_coordinate_status if spec.key == "p1_n" else normal_flip.status,
        "mode": normalization.normal_flip_mode if spec.key == "p1_n" else "N/A",
        "enabled": flip_policy["enabled"] if spec.key == "p1_n" else None,
        "policy_mode": flip_policy["mode"] if spec.key == "p1_n" else None,
        "policy_coordinate_status": flip_policy["coordinate_status"] if spec.key == "p1_n" else None,
        "reason": normalization.normal_flip_reason if spec.key == "p1_n" else normal_flip.reason,
        "evidence_paths": list(normal_flip.evidence_paths),
        "renderer_basis_verified": False,
    }
    return {
        "experiment_key": spec.key,
        "display_name": spec.display_name,
        "representation": spec.representation,
        "input_type": spec.input_type,
        "model_type": spec.model_type,
        "availability": spec.availability,
        "normalization": normalization.to_dict(),
        "samples": validation_rows,
        "normal_flip": normal_flip_contract,
        "status": "passed" if all(row["finite"] for row in validation_rows) else "failed",
        "requires_phase2_guard": spec.stage_two_status == "RUNNABLE",
    }


def build_framework_manifest(config: Mapping[str, Any]) -> dict[str, Any]:
    registry = validate_component_registry()
    frame = _load_frame(config)
    component_contracts = [_check_component_contract(config, frame, key, root=config.get("root")) for key in EXPERIMENT_ORDER]
    blockers = _check_common_contracts(config, frame)
    normal_flip = next(
        contract["normal_flip"]
        for contract in component_contracts
        if contract["experiment_key"] == "p1_n"
    )
    spec_contract = next(
        contract for contract in component_contracts if contract["experiment_key"] == "p1_spec"
    )
    if normal_flip["enabled"] and normal_flip["status"] != "VERIFIED":
        blockers.append("normal_flip_status_unverified")
    if spec_contract["status"] != "SKIPPED_UNAVAILABLE_BY_CURRENT_FRONTEND":
        blockers.append("p1_spec_not_explicitly_skipped")

    status = "P1_COMPONENT_FRAMEWORK_READY_FOR_PHASE2" if not blockers else "BLOCKED_BY_FRAMEWORK_IMPLEMENTATION_ERROR"
    return {
        "status": status,
        "blockers": blockers,
        "registry": registry,
        "component_contracts": component_contracts,
        "baseline_root": str(BASELINE_ROOT),
        "manifest_rows": int(len(frame)),
        "unique_case_ids": int(frame.case_id.nunique()),
        "patient_group_count": int(frame.patient_group_id.nunique()),
        "fixed_split_sha256": sha256_file(_resolve_path(config.get("fixed_split_path"), FIXED_SPLIT_PATH)),
        "baseline_protocol": json.loads(CORRECTED_PROTOCOL_PATH.read_text(encoding="utf-8")),
        "normal_flip_policy": {
            "normal_coordinate_status": normal_flip["coordinate_status"],
            "normal_flip_mode": normal_flip["mode"],
            "unverified_normal_flip_enabled": bool(normal_flip["enabled"]),
        },
    }


def _normal_coordinate_convention_payload(manifest: dict[str, Any]) -> dict[str, Any]:
    normal_contract = next(
        contract for contract in manifest["component_contracts"] if contract["experiment_key"] == "p1_n"
    )
    flip = normal_contract["normal_flip"]
    normalization = normal_contract["normalization"]
    return {
        "status": normalization.get("normal_coordinate_status", "UNVERIFIED"),
        "normal_source_variable": "rendered['normal_images']",
        "generation_function": "p0b_deca.p1_generation._write_case",
        "source_files": [
            "third_party/DECA/decalib/utils/renderer.py",
            "third_party/DECA/decalib/deca.py",
            "p0b_deca/p1_generation.py",
            "p0b_deca/p1_component_audit.py",
            "scripts/p0b/validate_p0b_sh_presets.py",
        ],
        "source_line_ranges": [
            "third_party/DECA/decalib/utils/renderer.py:262-304",
            "third_party/DECA/decalib/deca.py:206-216, 221-233",
            "p0b_deca/p1_generation.py:197-199, 299-311",
            "p0b_deca/p1_component_audit.py:524-544, 1110-1114",
            "scripts/p0b/validate_p0b_sh_presets.py:69-72",
        ],
        "coordinate_space": "renderer normal_coarse stored as raw float32 in maps.npz",
        "channel_order": None,
        "raw_value_range": "unverified_from_source",
        "saved_value_range": "float32 npz maps.npz::normal_coarse",
        "horizontal_image_axis": "positive right",
        "horizontal_normal_component": "unverified",
        "spatial_flip_required": False,
        "component_sign_flip_required": False,
        "sign_flip_channel_index": None,
        "evidence_types": [
            "source_trace",
            "schema_audit",
            "real_asset_read_only_validation",
        ],
        "renderer_basis_verified": False,
        "normal_coordinate_status": normalization.get("normal_coordinate_status", "UNVERIFIED"),
        "normal_flip_mode": normalization.get("normal_flip_mode", "DISABLED_SAFE_FALLBACK"),
        "unverified_normal_flip_enabled": bool(normal_contract["normal_flip"].get("enabled", False)),
        "decision": "DISABLED_SAFE_FALLBACK",
        "note": "Evidence traces the renderer output and storage contract, but does not prove normal channel order or the horizontal mirror sign convention. P1-N flip is therefore disabled.",
    }


def _validate_real_normal_assets(frame: pd.DataFrame, *, sample_count: int = 3) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    sample = frame.head(sample_count).copy()
    for _, record in sample.iterrows():
        maps_path = local_path(record["maps_path"])
        mask_path = local_path(record["face_valid_mask_path"])
        with np.load(maps_path, allow_pickle=False) as archive:
            normal = np.asarray(archive["normal_coarse"], dtype=np.float32)
        with Image.open(mask_path) as image:
            face_mask = np.asarray(image, dtype=np.uint8)
        face_mask = face_mask > 0
        rows.append(
            {
                "case_id": str(record["case_id"]),
                "maps_path": str(maps_path),
                "normal_shape": json.dumps(list(normal.shape)),
                "mask_shape": json.dumps(list(face_mask.shape)),
                "finite": bool(np.isfinite(normal).all()),
                "raw_min": float(np.nanmin(normal)),
                "raw_max": float(np.nanmax(normal)),
                "mask_coverage": float(face_mask.mean()),
                "horizontal_flip_path_called": False,
            }
        )
    summary = {
        "status": "passed" if rows and all(row["finite"] for row in rows) else "failed",
        "sample_count": len(rows),
        "horizontal_flip_path_called": False,
        "flip_mode": "DISABLED_SAFE_FALLBACK",
        "normal_coordinate_status": "UNVERIFIED",
        "notes": "Read-only validation of raw frozen normal assets only; no horizontal flip path was invoked.",
    }
    return rows, summary


def build_execution_plan(config: Mapping[str, Any]) -> dict[str, Any]:
    registry = validate_component_registry()
    runnable = list(runnable_component_keys())
    folds = [0, 1, 2, 3, 4]
    experiments = []
    total_jobs = 0
    for key in EXPERIMENT_ORDER:
        spec = get_component_spec(key)
        if not spec.enabled or spec.availability != "runnable":
            skip_reason = spec.skip_reason or "disabled"
            estimated_jobs = 0
        else:
            skip_reason = ""
            estimated_jobs = len(folds)
            total_jobs += estimated_jobs
        experiments.append(
            {
                "experiment_key": key,
                "display_name": spec.display_name,
                "availability": spec.availability,
                "input_type": spec.input_type,
                "model_type": spec.model_type,
                "representation": spec.representation,
                "mask": spec.mask,
                "folds": folds,
                "normalization_strategy": spec.normalization_key,
                "expected_outputs": [
                    "experiment_status.json",
                    "oof/oof_predictions_case.csv",
                    "oof/oof_metrics_visit.json",
                    "summary/p1_component_visit_metrics_cluster_bootstrap.json",
                ]
                if spec.enabled and spec.availability == "runnable"
                else [],
                "estimated_training_jobs": estimated_jobs,
                "dependencies": [
                    "manifest",
                    "fixed_split",
                    "frozen_deca_assets",
                    "corrected_p1_rgb_baseline",
                ],
                "skip_reason": skip_reason,
            }
        )
    return {
        "status": "planned",
        "experiment_order": list(EXPERIMENT_ORDER),
        "registry": registry,
        "experiments": experiments,
        "estimated_training_jobs": total_jobs,
        "runnable_experiments": runnable,
        "phase2_guard_required": True,
        "baseline_root": str(BASELINE_ROOT),
        "frozen_contract": str(FROZEN_CONTRACT_PATH),
    }


def write_validation_outputs(output_root: str | Path, manifest: dict[str, Any], config: Mapping[str, Any]) -> dict[str, Path]:
    root = Path(output_root)
    metadata = root / "metadata"
    validation = root / "validation"
    logs = root / "logs"
    metadata.mkdir(parents=True, exist_ok=True)
    validation.mkdir(parents=True, exist_ok=True)
    logs.mkdir(parents=True, exist_ok=True)
    paths = {
        "framework_manifest": metadata / "framework_manifest.json",
        "source_hashes": metadata / "source_hashes.json",
        "representation_contracts": metadata / "representation_contracts.json",
        "model_contracts": metadata / "model_contracts.json",
        "experiment_status": metadata / "experiment_status.json",
        "preflight_validation": metadata / "preflight_validation.json",
        "normal_coordinate_convention": metadata / NORMAL_CONVENTION_FILENAME,
        "dataset_validation": validation / "dataset_validation.json",
        "model_forward_validation": validation / "model_forward_validation.json",
        "normalization_validation": validation / "normalization_validation.json",
        "normal_flip_real_asset_validation": validation / NORMAL_REAL_ASSET_VALIDATION_FILENAME,
        "normal_flip_real_asset_summary": validation / NORMAL_REAL_ASSET_SUMMARY_FILENAME,
    }
    paths["framework_manifest"].write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    paths["source_hashes"].write_text(
        json.dumps(
            {
                "manifest_sha256": sha256_file(MANIFEST_PATH),
                "fixed_split_sha256": sha256_file(FIXED_SPLIT_PATH),
                "frozen_contract_sha256": sha256_file(FROZEN_CONTRACT_PATH),
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    paths["representation_contracts"].write_text(
        json.dumps({item["experiment_key"]: item["normalization"] for item in manifest["component_contracts"]}, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    paths["model_contracts"].write_text(
        json.dumps({item["experiment_key"]: {"model_type": item["model_type"], "representation": item["representation"]} for item in manifest["component_contracts"]}, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    paths["experiment_status"].write_text(
        json.dumps(
            {
                key: ("SKIPPED_UNAVAILABLE" if key == "p1_spec" else "FRAMEWORK_VALIDATED" if not manifest["blockers"] else "PENDING")
                for key in manifest["registry"]["experiment_order"]
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    paths["preflight_validation"].write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    paths["normal_coordinate_convention"].write_text(
        json.dumps(_normal_coordinate_convention_payload(manifest), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    real_asset_rows, real_asset_summary = _validate_real_normal_assets(_load_frame(config))
    with paths["normal_flip_real_asset_validation"].open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=sorted({key for row in real_asset_rows for key in row}))
        writer.writeheader()
        writer.writerows(real_asset_rows)
    paths["normal_flip_real_asset_summary"].write_text(
        json.dumps(real_asset_summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return paths


def validate_framework(config: Mapping[str, Any], output_root: str | Path) -> dict[str, Any]:
    manifest = build_framework_manifest(config)
    write_validation_outputs(output_root, manifest, config)
    return manifest
