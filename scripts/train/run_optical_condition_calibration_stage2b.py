"""Train Stage 2B label-free nonlinear EXIF/device condition-response models."""

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
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
import torch
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models.exif_conditioned_response_mlp import (  # noqa: E402
    ARCHITECTURE,
    EXPECTED_PARAMETER_COUNT,
    restore_model_from_checkpoint,
)
from scripts.evaluate.compare_optical_calibration_stage2a_stage2b import (  # noqa: E402
    camera_representation_comparison,
    compare_stage2a_stage2b,
)
from utils.optical_condition_calibration import (  # noqa: E402
    ALL_TARGETS,
    CHEEK_TARGETS,
    CONDITION_NAMES,
    DESIGN_FEATURE_NAMES,
    EXPECTED_CAMERAS,
    FOREHEAD_CHEEK_TARGETS,
    REFERENCE_CAMERA,
    build_design_matrix,
    transform_conditions,
    validate_camera_values,
)
from utils.optical_condition_calibration_nn import (  # noqa: E402
    build_diagnostics,
    combined_diagnostics,
    combined_file_sha256,
    condition_range_audit,
    deterministic_inner_split,
    predict_original_scale,
    sha256_file,
    sha256_ids,
    standardize_targets,
    train_epoch_selection,
    train_final_model,
    transform_feature_frame,
    write_csv,
    write_json,
)

TASK = "optical_condition_calibration_stage2b"
TASK_TITLE = "Stage 2B：EXIF与设备条件化的非线性区域光学表型校准网络"
REPORT_MARKER = f"<!-- task: {TASK} -->"
DEFAULT_CONFIG = PROJECT_ROOT / "config/train/optical_condition_calibration_stage2b.yaml"
EXPECTED_SPLIT_SHA256 = "fe5102c02890c546f323b0a94ebc5b125ebcfeb50e62d2d43f0564b4b383f24b"
FIRST_STAGE_COLUMNS = (
    "ID", "camera_id", "relative_optical_exposure", "log2_iso_condition",
    "forehead_available", *ALL_TARGETS,
)
STAGE2A_BASE_COLUMNS = (
    "ID", "fold", "split_role", "camera_id", "forehead_available",
    *CONDITION_NAMES, "z_relative_optical_exposure", "z_log2_iso_condition",
)
STAGE2A_TARGET_COLUMNS = tuple(
    f"{representation}_{target}"
    for target in ALL_TARGETS
    for representation in ("raw", "predicted_acquisition", "residual", "calibrated")
)
STAGE2A_COLUMNS = STAGE2A_BASE_COLUMNS + STAGE2A_TARGET_COLUMNS
IMPLEMENTATION_FILES = (
    "models/exif_conditioned_response_mlp.py",
    "utils/optical_condition_calibration_nn.py",
    "scripts/train/run_optical_condition_calibration_stage2b.py",
    "scripts/evaluate/compare_optical_calibration_stage2a_stage2b.py",
    "config/train/optical_condition_calibration_stage2b.yaml",
    "tests/test_exif_conditioned_response_mlp.py",
    "tests/test_optical_condition_calibration_stage2b.py",
    "tests/test_optical_calibration_stage2b_protocol.py",
)


class Stage2BFailure(RuntimeError):
    def __init__(self, stage: str, errors: Sequence[str]):
        self.stage = str(stage)
        self.errors = [str(value) for value in errors]
        super().__init__(f"{self.stage}: {' | '.join(self.errors)}")


def project_relative(path: Path, root: Path = PROJECT_ROOT) -> str:
    return path.resolve().relative_to(root.resolve()).as_posix()


def resolve_under_project(value: str | Path, root: Path) -> Path:
    candidate = Path(value).expanduser()
    resolved = candidate.resolve() if candidate.is_absolute() else (root / candidate).resolve()
    try:
        resolved.relative_to(root.resolve())
    except ValueError as exc:
        raise ValueError(f"Path outside project root: {resolved}") from exc
    return resolved


def get_git_commit(root: Path) -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=root, text=True, stderr=subprocess.DEVNULL
        ).strip()
    except Exception:
        return "unavailable"


def load_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=TASK_TITLE)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--fold", default="all", choices=["all", "0", "1", "2", "3", "4"])
    parser.add_argument("--protocol-only", action="store_true")
    parser.add_argument("--summarize-only", action="store_true")
    parser.add_argument("--compare-only", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--skip-completed", action="store_true")
    parser.add_argument("--overwrite", action="store_true", default=None)
    cli = parser.parse_args(argv)
    config_path = cli.config.resolve() if cli.config.is_absolute() else (PROJECT_ROOT / cli.config).resolve()
    config = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    root_value = config.get("project_root", ".")
    root = PROJECT_ROOT if root_value in (None, "", ".") else Path(root_value).resolve()
    values = dict(config)
    values.update({"project_root": root, "config_path": config_path})
    for key in (
        "first_stage_csv", "first_stage_schema", "first_stage_manifest",
        "stage2a_experiment_dir", "stage2a_report", "stage2a_run_manifest", "stage2a_oof",
        "split_dir", "experiment_output_dir", "report_output_dir",
    ):
        values[key] = resolve_under_project(values[key], root)
    values.update({
        "fold": cli.fold, "protocol_only": bool(cli.protocol_only),
        "summarize_only": bool(cli.summarize_only), "compare_only": bool(cli.compare_only),
        "resume": bool(cli.resume), "skip_completed": bool(cli.skip_completed),
    })
    if cli.overwrite is not None:
        values["overwrite"] = bool(cli.overwrite)
    args = argparse.Namespace(**values)
    validate_config(args)
    return args


def validate_config(args: argparse.Namespace) -> None:
    errors: list[str] = []
    fixed = {
        "task": TASK, "n_folds": 5, "reference_camera": REFERENCE_CAMERA,
        "device": "cpu", "dtype": "float32", "optimizer": "AdamW",
        "learning_rate": 1.0e-2, "weight_decay": 1.0e-4,
        "batch_mode": "full_batch", "max_epochs": 500, "minimum_epochs": 50,
        "early_stopping_patience": 50, "minimum_improvement": 1.0e-6,
        "gradient_clip_max_norm": 5.0, "scheduler": "none", "amp": False,
        "target_std_epsilon": 1.0e-8, "condition_std_epsilon": 1.0e-8,
        "inner_val_fraction": 0.20, "base_seed": 2026,
    }
    for key, expected in fixed.items():
        if getattr(args, key) != expected:
            errors.append(f"{key} must equal fixed value {expected!r}")
    if dict(args.camera_mapping) != {EXPECTED_CAMERAS[0]: 0, EXPECTED_CAMERAS[1]: 1}:
        errors.append("camera mapping differs from Stage 2A")
    if tuple(args.condition_names) != CONDITION_NAMES or tuple(args.condition_feature_names) != DESIGN_FEATURE_NAMES:
        errors.append("condition field order differs from Stage 2A")
    if tuple(args.cheek_targets) != CHEEK_TARGETS or tuple(args.forehead_cheek_targets) != FOREHEAD_CHEEK_TARGETS:
        errors.append("target order differs from Stage 2A")
    if dict(args.architecture) != {
        "input_dim": 5, "hidden_dims": [8, 8], "output_dim": 3,
        "activation": "Tanh", "parameter_count": 147,
    }:
        errors.append("architecture differs from fixed 5-8-8-3 Tanh MLP")
    if sum((args.protocol_only, args.summarize_only, args.compare_only)) > 1:
        errors.append("protocol-only, summarize-only and compare-only are mutually exclusive")
    if (args.summarize_only or args.compare_only) and args.fold != "all":
        errors.append("summarize-only and compare-only require --fold all")
    expected_exp = (args.project_root / "experiments/optical_condition_calibration_stage2b").resolve()
    expected_report = (args.project_root / "reports/optical_condition_calibration_stage2b").resolve()
    if args.experiment_output_dir != expected_exp or args.report_output_dir != expected_report:
        errors.append("output directories differ from fixed Stage 2B directories")
    if errors:
        raise Stage2BFailure("config", errors)


def load_first_stage(args: argparse.Namespace) -> pd.DataFrame:
    for path in (args.first_stage_csv, args.first_stage_schema, args.first_stage_manifest):
        if not path.is_file():
            raise Stage2BFailure("preflight", [f"Missing first-stage input: {path}"])
    schema = json.loads(args.first_stage_schema.read_text(encoding="utf-8"))
    manifest = json.loads(args.first_stage_manifest.read_text(encoding="utf-8"))
    if manifest.get("status") != "COMPLETE" or manifest.get("clinical_labels_read") is not False:
        raise Stage2BFailure("preflight", ["First-stage manifest is not completed and label-free"])
    if manifest.get("output_files", {}).get("main", {}).get("sha256") != sha256_file(args.first_stage_csv):
        raise Stage2BFailure("preflight", ["First-stage CSV SHA256 differs from extraction manifest"])
    if not set(ALL_TARGETS).issubset(set(schema.get("derived_observation_columns", []))):
        raise Stage2BFailure("preflight", ["First-stage schema lacks the fixed six targets"])
    try:
        frame = pd.read_csv(
            args.first_stage_csv, usecols=list(FIRST_STAGE_COLUMNS),
            dtype={"ID": str, "camera_id": str}, encoding="utf-8-sig",
        )
    except ValueError as exc:
        raise Stage2BFailure("preflight", [f"First-stage whitelist read failed: {exc}"]) from exc
    frame["ID"] = frame["ID"].astype(str).str.strip()
    errors: list[str] = []
    if len(frame) != 500 or frame["ID"].nunique() != 500 or frame["ID"].str.casefold().nunique() != 500:
        errors.append("First-stage input must have 500 exact unique IDs")
    try:
        validate_camera_values(frame["camera_id"], require_both=True)
    except ValueError as exc:
        errors.append(str(exc))
    conditions = frame[list(CONDITION_NAMES)].apply(pd.to_numeric, errors="coerce").to_numpy(float)
    if not np.isfinite(conditions).all():
        errors.append("Condition inputs contain nonfinite values")
    available = pd.to_numeric(frame["forehead_available"], errors="coerce")
    if not available.isin([0, 1]).all():
        errors.append("forehead_available must contain only 0/1")
    if not np.isfinite(frame[list(CHEEK_TARGETS)].to_numpy(float)).all():
        errors.append("Cheek targets contain nonfinite values")
    forehead = frame[list(FOREHEAD_CHEEK_TARGETS)]
    if not np.isfinite(forehead.loc[available.eq(1)].to_numpy(float)).all():
        errors.append("Available forehead targets contain nonfinite values")
    if not forehead.loc[available.eq(0)].isna().all().all():
        errors.append("Unavailable forehead targets must be NaN")
    if errors:
        raise Stage2BFailure("preflight", errors)
    return frame.sort_values("ID", kind="stable").reset_index(drop=True)


def load_stage2a_inputs(args: argparse.Namespace) -> dict[str, Any]:
    required = (args.stage2a_run_manifest, args.stage2a_oof, args.stage2a_report)
    if not all(path.is_file() for path in required):
        raise Stage2BFailure("preflight", ["Stage 2A manifest, OOF or report is missing"])
    manifest = json.loads(args.stage2a_run_manifest.read_text(encoding="utf-8"))
    errors: list[str] = []
    if manifest.get("task") != "optical_condition_calibration_stage2a" or manifest.get("status") != "COMPLETE":
        errors.append("Stage 2A run manifest is not COMPLETE")
    if manifest.get("clinical_labels_loaded") is not False or manifest.get("nyha_used") is not False:
        errors.append("Stage 2A manifest is not label-free")
    if manifest.get("first_stage_csv", {}).get("sha256") != sha256_file(args.first_stage_csv):
        errors.append("Stage 2A first-stage SHA256 differs from current first-stage input")
    if manifest.get("split", {}).get("sha256") != EXPECTED_SPLIT_SHA256:
        errors.append("Stage 2A split SHA256 differs from the fixed split")
    if manifest.get("oof_output_sha256") != sha256_file(args.stage2a_oof):
        errors.append("Stage 2A OOF SHA256 differs from its run manifest")
    if tuple(manifest.get("target_fields", [])) != ALL_TARGETS:
        errors.append("Stage 2A target order differs")
    if tuple(manifest.get("design_matrix_fields", [])) != DESIGN_FEATURE_NAMES:
        errors.append("Stage 2A condition feature order differs")
    if errors:
        raise Stage2BFailure("preflight", errors)
    try:
        oof = pd.read_csv(
            args.stage2a_oof, usecols=list(STAGE2A_COLUMNS),
            dtype={"ID": str, "camera_id": str}, encoding="utf-8-sig",
        )
    except ValueError as exc:
        raise Stage2BFailure("preflight", [f"Stage 2A OOF whitelist read failed: {exc}"]) from exc
    if len(oof) != 500 or oof["ID"].nunique() != 500:
        raise Stage2BFailure("preflight", ["Stage 2A OOF is not 500 unique IDs"])
    return {"manifest": manifest, "oof": oof.sort_values("ID", kind="stable").reset_index(drop=True)}


def audit_splits(args: argparse.Namespace, first_stage: pd.DataFrame) -> dict[str, Any]:
    splits: dict[int, dict[str, pd.DataFrame]] = {}
    files: list[Path] = []
    rows: list[dict[str, Any]] = []
    errors: list[str] = []
    all_ids = set(first_stage["ID"])
    all_val: list[str] = []
    per_fold_hashes: dict[str, Any] = {}
    for fold in range(5):
        frames: dict[str, pd.DataFrame] = {}
        hashes: dict[str, str] = {}
        for role, pattern in (("train", args.train_csv_pattern), ("val", args.val_csv_pattern)):
            path = args.split_dir / str(pattern).format(fold=fold)
            files.append(path)
            try:
                frame = pd.read_csv(
                    path, usecols=["ID", "fold"], dtype={"ID": str}, encoding="utf-8-sig"
                )
            except Exception as exc:
                errors.append(f"Cannot read split ID/fold whitelist: {path}: {exc}")
                continue
            frame["ID"] = frame["ID"].astype(str).str.strip()
            frame["fold"] = pd.to_numeric(frame["fold"], errors="coerce")
            frames[role] = frame
            hashes[role] = sha256_file(path)
            rows.append({
                "fold": fold, "split_role": role, "n_rows": len(frame),
                "unique_id_n": frame["ID"].nunique(), "file_sha256": hashes[role],
                "id_sha256": sha256_ids(frame["ID"].tolist()),
            })
        if set(frames) != {"train", "val"}:
            continue
        train, val = frames["train"], frames["val"]
        train_ids, val_ids = set(train["ID"]), set(val["ID"])
        if len(train) != 400 or train["ID"].nunique() != 400:
            errors.append(f"Fold {fold} train is not 400 unique IDs")
        if len(val) != 100 or val["ID"].nunique() != 100:
            errors.append(f"Fold {fold} val is not 100 unique IDs")
        if train_ids & val_ids or train_ids | val_ids != all_ids:
            errors.append(f"Fold {fold} train/val is not a disjoint complete partition")
        if not val["fold"].eq(fold).all() or train["fold"].eq(fold).any():
            errors.append(f"Fold {fold} fold assignment is inconsistent")
        outer_train = first_stage.set_index("ID").loc[sorted(train_ids)]
        try:
            validate_camera_values(outer_train["camera_id"], require_both=True)
        except ValueError as exc:
            errors.append(f"Fold {fold}: {exc}")
        all_val.extend(val["ID"].tolist())
        per_fold_hashes[str(fold)] = {
            "train_file_sha256": hashes["train"], "val_file_sha256": hashes["val"],
            "train_id_sha256": sha256_ids(train["ID"].tolist()),
            "val_id_sha256": sha256_ids(val["ID"].tolist()),
        }
        splits[fold] = frames
    split_sha = combined_file_sha256(files) if all(path.is_file() for path in files) else "unavailable"
    if split_sha != EXPECTED_SPLIT_SHA256:
        errors.append(f"Split SHA256 {split_sha} differs from fixed {EXPECTED_SPLIT_SHA256}")
    if len(all_val) != 500 or len(set(all_val)) != 500 or set(all_val) != all_ids:
        errors.append("Five validation folds do not form exact one-time OOF coverage")
    if errors:
        raise Stage2BFailure("split_protocol", errors)
    return {
        "splits": splits, "split_files": files, "split_sha256": split_sha,
        "audit": pd.DataFrame(rows).sort_values(["fold", "split_role"], kind="stable"),
        "per_fold_hashes": per_fold_hashes,
    }


def select_rows(first_stage: pd.DataFrame, ids: Sequence[str]) -> pd.DataFrame:
    return first_stage.set_index("ID", drop=False).loc[
        sorted(str(value) for value in ids)
    ].reset_index(drop=True)


def load_stage2a_fold(
    args: argparse.Namespace,
    fold: int,
    roles: Sequence[str] = ("train", "val"),
) -> dict[str, Any]:
    fold_dir = args.stage2a_experiment_dir / f"fold_{fold}"
    scaler_path = fold_dir / "condition_scaler.json"
    if not scaler_path.is_file():
        raise Stage2BFailure("preflight", [f"Missing Stage 2A condition scaler for fold {fold}"])
    scaler = json.loads(scaler_path.read_text(encoding="utf-8"))
    frames: dict[str, pd.DataFrame] = {}
    for role in roles:
        if role not in {"train", "val"}:
            raise Stage2BFailure("preflight", [f"Unsupported Stage 2A fold role: {role}"])
        path = fold_dir / f"{role}_calibrated_features.csv"
        try:
            frames[role] = pd.read_csv(
                path, usecols=list(STAGE2A_COLUMNS), dtype={"ID": str, "camera_id": str},
                encoding="utf-8-sig",
            ).sort_values("ID", kind="stable").reset_index(drop=True)
        except Exception as exc:
            raise Stage2BFailure("preflight", [f"Cannot read Stage 2A fold {fold} {role}: {exc}"]) from exc
    return {"scaler": scaler, "scaler_path": scaler_path, **frames}


def validate_stage2a_fold_alignment(
    fold: int,
    outer_train: pd.DataFrame,
    outer_val: pd.DataFrame | None,
    stage2a_fold: Mapping[str, Any],
) -> dict[str, float]:
    errors: list[str] = []
    max_z_error = 0.0
    role_sources = [("train", outer_train)]
    if outer_val is not None:
        role_sources.append(("val", outer_val))
    for role, source in role_sources:
        a = stage2a_fold[role]
        source = source.sort_values("ID", kind="stable").reset_index(drop=True)
        if not source["ID"].astype(str).equals(a["ID"].astype(str)):
            errors.append(f"Fold {fold} {role} IDs differ from Stage 2A")
            continue
        transformed = transform_conditions(source, stage2a_fold["scaler"])
        for column in ("z_relative_optical_exposure", "z_log2_iso_condition"):
            error = float(np.max(np.abs(transformed[column].to_numpy(float) - a[column].to_numpy(float))))
            max_z_error = max(max_z_error, error)
        for target in ALL_TARGETS:
            if not np.allclose(
                source[target].to_numpy(float), a[f"raw_{target}"].to_numpy(float),
                rtol=0, atol=1e-12, equal_nan=True,
            ):
                errors.append(f"Fold {fold} {role} raw {target} differs from Stage 2A")
    if max_z_error > 1e-12:
        errors.append(f"Fold {fold} Stage 2B z values differ from Stage 2A: {max_z_error}")
    if errors:
        raise Stage2BFailure("stage2a_alignment", errors)
    return {"maximum_z_absolute_error": max_z_error}


def input_hash_inventory(args: argparse.Namespace, split_info: Mapping[str, Any]) -> dict[str, str]:
    paths = [
        args.first_stage_csv, args.first_stage_schema, args.first_stage_manifest,
        args.stage2a_run_manifest, args.stage2a_oof, args.stage2a_report,
        *split_info["split_files"],
    ]
    for fold in range(5):
        fold_dir = args.stage2a_experiment_dir / f"fold_{fold}"
        paths.extend([
            fold_dir / "condition_scaler.json", fold_dir / "train_calibrated_features.csv",
            fold_dir / "val_calibrated_features.csv",
        ])
    return {project_relative(path): sha256_file(path) for path in paths}


def verify_owned_output(path: Path, report: bool = False) -> bool:
    if not path.exists() or not any(path.iterdir()):
        return True
    owner = path / ".stage2b_owner.json"
    if owner.is_file():
        try:
            return json.loads(owner.read_text(encoding="utf-8")).get("task") == TASK
        except Exception:
            return False
    if report:
        report_path = path / "optical_condition_calibration_stage2b_report.md"
        return report_path.is_file() and REPORT_MARKER in report_path.read_text(encoding="utf-8")
    manifest = path / "summary/run_manifest.json"
    try:
        return manifest.is_file() and json.loads(manifest.read_text(encoding="utf-8")).get("task") == TASK
    except Exception:
        return False


def prepare_outputs(args: argparse.Namespace) -> None:
    roots = (args.experiment_output_dir, args.report_output_dir)
    nonempty = [path for path in roots if path.exists() and any(path.iterdir())]
    reuse = args.resume or args.skip_completed or args.summarize_only or args.compare_only
    if nonempty and not reuse and not bool(args.overwrite):
        raise Stage2BFailure("output_safety", [f"Non-empty output directory: {path}" for path in nonempty])
    if nonempty and not all(verify_owned_output(path, path == args.report_output_dir) for path in nonempty):
        raise Stage2BFailure("output_safety", ["Existing outputs are not verified Stage 2B artifacts"])
    if nonempty and bool(args.overwrite) and not reuse:
        for path in roots:
            if path.exists():
                shutil.rmtree(path)
    for path in roots:
        path.mkdir(parents=True, exist_ok=True)
        write_json({"task": TASK}, path / ".stage2b_owner.json")


def write_protocol(
    args: argparse.Namespace,
    first_stage: pd.DataFrame,
    stage2a: Mapping[str, Any],
    split_info: Mapping[str, Any],
) -> None:
    protocol = args.experiment_output_dir / "protocol"
    inputs = pd.DataFrame([
        {"input_role": "first_stage_csv", "path": project_relative(args.first_stage_csv), "sha256": sha256_file(args.first_stage_csv), "status": "PASS"},
        {"input_role": "stage2a_run_manifest", "path": project_relative(args.stage2a_run_manifest), "sha256": sha256_file(args.stage2a_run_manifest), "status": stage2a["manifest"]["status"]},
        {"input_role": "stage2a_oof", "path": project_relative(args.stage2a_oof), "sha256": sha256_file(args.stage2a_oof), "status": "PASS"},
        {"input_role": "fixed_split_combined", "path": project_relative(args.split_dir), "sha256": split_info["split_sha256"], "status": "PASS"},
    ])
    write_csv(inputs, protocol / "input_audit.csv")
    write_csv(split_info["audit"], protocol / "split_audit.csv")
    write_json({
        "task": TASK, "status": "PASS", "first_stage_rows": len(first_stage),
        "first_stage_unique_ids": int(first_stage["ID"].nunique()),
        "stage2a_status": stage2a["manifest"]["status"],
        "stage2a_oof_rows": len(stage2a["oof"]),
        "stage2a_oof_unique_ids": int(stage2a["oof"]["ID"].nunique()),
        "split_source": project_relative(args.split_dir), "split_sha256": split_info["split_sha256"],
        "split_regenerated": False, "clinical_columns_loaded": [],
        "allowed_input_columns": list(FIRST_STAGE_COLUMNS),
        "pytorch_version": torch.__version__, "cuda_available": torch.cuda.is_available(),
        "execution_device": "cpu", "per_fold_split_hashes": split_info["per_fold_hashes"],
    }, protocol / "protocol_manifest.json")


def _inner_split_output(frame: pd.DataFrame, fold: int, network_type: str, role: str) -> pd.DataFrame:
    output = frame.loc[:, ["ID", "camera_id", "forehead_available"]].copy()
    output.insert(1, "fold", int(fold))
    output.insert(2, "network_type", network_type)
    output.insert(3, "split_role", role)
    return output.sort_values("ID", kind="stable").reset_index(drop=True)


def run_fold(
    args: argparse.Namespace,
    fold: int,
    first_stage: pd.DataFrame,
    split_info: Mapping[str, Any],
    stage2a_context: Mapping[str, Any],
    output_root: Path,
) -> dict[str, Any]:
    fold_dir = output_root / f"fold_{fold}"
    fold_dir.mkdir(parents=True, exist_ok=True)
    outer_train = select_rows(first_stage, split_info["splits"][fold]["train"]["ID"].tolist())
    outer_val_ids = [str(value) for value in split_info["splits"][fold]["val"]["ID"].tolist()]
    stage2a_fold = load_stage2a_fold(args, fold, roles=("train",))
    train_z_audit = validate_stage2a_fold_alignment(fold, outer_train, None, stage2a_fold)
    eligible = {
        "cheek": outer_train,
        "forehead_cheek": outer_train.loc[
            outer_train["forehead_available"].astype(int).eq(1)
            & np.isfinite(outer_train[list(FOREHEAD_CHEEK_TARGETS)].to_numpy(float)).all(axis=1)
        ].copy(),
    }
    target_groups = {"cheek": CHEEK_TARGETS, "forehead_cheek": FOREHEAD_CHEEK_TARGETS}
    seeds = {
        "inner_split": int(args.base_seed) + fold * 100,
        "cheek": int(args.base_seed) + fold * 100 + 1,
        "forehead_cheek": int(args.base_seed) + fold * 100 + 2,
    }
    inner_dir = fold_dir / "inner_split"
    inner_manifest: dict[str, Any] = {
        "fold": fold, "seed": seeds["inner_split"], "stratification_fields": ["camera_id"],
        "outer_validation_id_sha256": sha256_ids(outer_val_ids), "networks": {},
        "clinical_columns_loaded": [],
    }
    inner_splits: dict[str, tuple[pd.DataFrame, pd.DataFrame]] = {}
    inner_paths: dict[str, tuple[Path, Path]] = {}
    for network_type in ("cheek", "forehead_cheek"):
        inner_train, inner_val, manifest = deterministic_inner_split(
            eligible[network_type], seeds["inner_split"], float(args.inner_val_fraction)
        )
        if network_type == "forehead_cheek" and (
            not inner_train["forehead_available"].eq(1).all() or not inner_val["forehead_available"].eq(1).all()
        ):
            raise Stage2BFailure("inner_split", ["Forehead inner split contains unavailable cases"])
        outer_overlap = len((set(inner_train["ID"]) | set(inner_val["ID"])) & set(outer_val_ids))
        if outer_overlap:
            raise Stage2BFailure("inner_split", [f"Fold {fold} {network_type} overlaps outer validation"])
        train_path = inner_dir / f"{network_type}_inner_train.csv"
        val_path = inner_dir / f"{network_type}_inner_val.csv"
        write_csv(_inner_split_output(inner_train, fold, network_type, "inner_train"), train_path)
        write_csv(_inner_split_output(inner_val, fold, network_type, "inner_val"), val_path)
        manifest.update({
            "network_type": network_type, "fold": fold,
            "outer_validation_overlap_n": outer_overlap,
            "train_file_sha256": sha256_file(train_path), "val_file_sha256": sha256_file(val_path),
            "combined_split_sha256": combined_file_sha256([train_path, val_path]),
        })
        inner_manifest["networks"][network_type] = manifest
        inner_splits[network_type] = (inner_train, inner_val)
        inner_paths[network_type] = (train_path, val_path)
    write_json(inner_manifest, inner_dir / "inner_split_manifest.json")
    selections: dict[str, Any] = {}
    finals: dict[str, Any] = {}
    for network_type in ("cheek", "forehead_cheek"):
        train_inner, val_inner = inner_splits[network_type]
        selection_dir = fold_dir / f"{network_type}_epoch_selection"
        selection_context = {
            "inner_split_sha256": inner_manifest["networks"][network_type]["combined_split_sha256"],
            "config_sha256": sha256_file(args.config_path),
        }
        selection = train_epoch_selection(
            train_inner, val_inner, target_groups[network_type], fold, network_type,
            seeds[network_type], selection_dir, vars(args), selection_context,
        )
        selections[network_type] = selection
        prefix = "cheek" if network_type == "cheek" else "forehead_cheek"
        checkpoint_path = fold_dir / f"{prefix}_final_model.pth"
        target_scaler_path = fold_dir / f"{prefix}_target_scaler.json"
        manifest_path = fold_dir / f"{prefix}_final_model_manifest.json"
        final_context = {
            "stage2a_condition_scaler_path": project_relative(stage2a_fold["scaler_path"]),
            "stage2a_condition_scaler_sha256": sha256_file(stage2a_fold["scaler_path"]),
            "split_sha256": split_info["split_sha256"], "config_sha256": sha256_file(args.config_path),
            "first_stage_sha256": stage2a_context["first_stage_sha256"],
            "stage2a_manifest_sha256": stage2a_context["stage2a_manifest_sha256"],
            "git_commit": stage2a_context["git_commit"],
            "checkpoint_relative_path": checkpoint_path.relative_to(output_root).as_posix(),
            "target_scaler_relative_path": target_scaler_path.relative_to(output_root).as_posix(),
        }
        finals[network_type] = train_final_model(
            eligible[network_type], target_groups[network_type], stage2a_fold["scaler"],
            fold, network_type, selection["selected"]["selected_epoch"], seeds[network_type],
            checkpoint_path, target_scaler_path, manifest_path, vars(args), final_context,
        )
    # The outer validation feature rows and Stage 2A validation file are deliberately
    # materialized only after epoch selection and fresh full-outer refits are complete.
    outer_val = select_rows(first_stage, outer_val_ids)
    stage2a_fold.update(load_stage2a_fold(args, fold, roles=("val",)))
    z_audit = validate_stage2a_fold_alignment(fold, outer_train, outer_val, stage2a_fold)
    z_audit["inner_selection_train_maximum_z_absolute_error"] = train_z_audit[
        "maximum_z_absolute_error"
    ]
    train_features = transform_feature_frame(
        outer_train, fold, "train", stage2a_fold["scaler"],
        finals["cheek"]["model"], finals["forehead_cheek"]["model"],
        finals["cheek"]["target_scaler"], finals["forehead_cheek"]["target_scaler"],
    )
    val_features = transform_feature_frame(
        outer_val, fold, "val", stage2a_fold["scaler"],
        finals["cheek"]["model"], finals["forehead_cheek"]["model"],
        finals["cheek"]["target_scaler"], finals["forehead_cheek"]["target_scaler"],
    )
    write_csv(train_features, fold_dir / "train_nn_calibrated_features.csv")
    write_csv(val_features, fold_dir / "val_nn_calibrated_features.csv")
    train_diag = build_diagnostics(train_features, fold, "train")
    val_diag = build_diagnostics(val_features, fold, "val")
    diagnostics = {
        family: pd.concat([train_diag[family], val_diag[family]], ignore_index=True)
        for family in train_diag
    }
    generalization_rows = []
    for network_type, targets in target_groups.items():
        mask_train = np.ones(len(train_features), dtype=bool) if network_type == "cheek" else train_features["forehead_available"].eq(1).to_numpy()
        mask_val = np.ones(len(val_features), dtype=bool) if network_type == "cheek" else val_features["forehead_available"].eq(1).to_numpy()
        scaler = finals[network_type]["target_scaler"]
        train_raw = train_features.loc[mask_train, [f"raw_{target}" for target in targets]].to_numpy(float)
        train_pred = train_features.loc[mask_train, [f"predicted_condition_nn_{target}" for target in targets]].to_numpy(float)
        val_raw = val_features.loc[mask_val, [f"raw_{target}" for target in targets]].to_numpy(float)
        val_pred = val_features.loc[mask_val, [f"predicted_condition_nn_{target}" for target in targets]].to_numpy(float)
        std = np.asarray(scaler["population_std"], dtype=float)
        train_mse = float(np.mean(((train_raw - train_pred) / std) ** 2))
        val_mse = float(np.mean(((val_raw - val_pred) / std) ** 2))
        selected = selections[network_type]["selected"]
        generalization_rows.append({
            "fold": fold, "network_type": network_type,
            "selected_epoch": selected["selected_epoch"],
            "inner_train_best_epoch_loss": selected["best_inner_train_loss"],
            "inner_val_best_epoch_loss": selected["best_inner_val_loss"],
            "final_outer_train_loss": train_mse, "outer_val_mse": val_mse,
            "outer_val_minus_train_mse": val_mse - train_mse,
            "outer_validation_used_for_training_decision": 0,
        })
    diagnostics["generalization"] = pd.DataFrame(generalization_rows)
    write_csv(combined_diagnostics(diagnostics), fold_dir / "diagnostics.csv")
    range_audit = condition_range_audit(outer_train, outer_val, stage2a_fold["scaler"], fold)
    write_csv(range_audit, fold_dir / "condition_range_audit.csv")
    summary = {
        "fold": fold, "status": "PASS", "outer_train_n": len(outer_train), "outer_val_n": len(outer_val),
        "cheek_final_train_n": len(eligible["cheek"]),
        "forehead_cheek_final_train_n": len(eligible["forehead_cheek"]),
        "val_forehead_available_n": int(outer_val["forehead_available"].sum()),
        "cheek_selected_epoch": selections["cheek"]["selected"]["selected_epoch"],
        "forehead_cheek_selected_epoch": selections["forehead_cheek"]["selected"]["selected_epoch"],
        "condition_range_outside_val_n": int(range_audit["any_condition_outside_train_range"].sum()),
        "both_conditions_outside_val_n": int(range_audit["both_conditions_outside_train_range"].sum()),
        "stage2a_z_max_absolute_error": z_audit["maximum_z_absolute_error"],
        "outer_train_id_sha256": sha256_ids(outer_train["ID"].tolist()),
        "outer_val_id_sha256": sha256_ids(outer_val["ID"].tolist()),
    }
    write_json(summary, fold_dir / "fold_summary.json")
    return {
        "summary": summary, "val_features": val_features, "diagnostics": diagnostics,
        "range_audit": range_audit, "selections": selections, "finals": finals,
    }


def completed_fold(output_root: Path, fold: int) -> bool:
    fold_dir = output_root / f"fold_{fold}"
    required = [
        "cheek_final_model.pth", "forehead_cheek_final_model.pth",
        "cheek_final_model_manifest.json", "forehead_cheek_final_model_manifest.json",
        "train_nn_calibrated_features.csv", "val_nn_calibrated_features.csv",
        "diagnostics.csv", "condition_range_audit.csv", "fold_summary.json",
        "inner_split/inner_split_manifest.json",
        "cheek_epoch_selection/selected_epoch.json",
        "forehead_cheek_epoch_selection/selected_epoch.json",
    ]
    if not all((fold_dir / name).is_file() for name in required):
        return False
    try:
        summary = json.loads((fold_dir / "fold_summary.json").read_text(encoding="utf-8"))
        val = pd.read_csv(fold_dir / "val_nn_calibrated_features.csv", usecols=["ID"], dtype={"ID": str})
        return summary.get("status") == "PASS" and len(val) == 100 and val["ID"].nunique() == 100
    except Exception:
        return False


def load_fold_result(output_root: Path, fold: int) -> dict[str, Any]:
    if not completed_fold(output_root, fold):
        raise Stage2BFailure("summarize", [f"Fold {fold} is incomplete"])
    fold_dir = output_root / f"fold_{fold}"
    summary = json.loads((fold_dir / "fold_summary.json").read_text(encoding="utf-8"))
    val = pd.read_csv(fold_dir / "val_nn_calibrated_features.csv", dtype={"ID": str}, encoding="utf-8-sig")
    union = pd.read_csv(fold_dir / "diagnostics.csv", encoding="utf-8-sig")
    diagnostics = {
        family: union.loc[union["metric_family"].eq(family)]
        .drop(columns="metric_family").dropna(axis=1, how="all").reset_index(drop=True)
        for family in ("fit", "correlation", "camera", "variance", "generalization")
    }
    range_audit = pd.read_csv(fold_dir / "condition_range_audit.csv", dtype={"ID": str}, encoding="utf-8-sig")
    selections = {
        network: {"selected": json.loads((fold_dir / f"{network}_epoch_selection/selected_epoch.json").read_text(encoding="utf-8"))}
        for network in ("cheek", "forehead_cheek")
    }
    return {"summary": summary, "val_features": val, "diagnostics": diagnostics, "range_audit": range_audit, "selections": selections}


def build_feature_schema() -> dict[str, Any]:
    predicted = [f"predicted_condition_nn_{target}" for target in ALL_TARGETS]
    residual = [f"residual_nn_{target}" for target in ALL_TARGETS]
    calibrated = [f"calibrated_nn_{target}" for target in ALL_TARGETS]
    qc = ["valid_skin_fraction", "valid_skin_pixel_count", "bbox_area", "cheek_abs_diff", "IQR", "channel_clipping_fractions"]
    forbidden = [
        "camera_id", *CONDITION_NAMES, "z_relative_optical_exposure", "z_log2_iso_condition",
        *predicted, *qc, "inner_split", "target_scaler",
    ]
    return {
        "schema_name": TASK, "identifier_columns": ["ID", "fold", "split_role"],
        "availability_columns": ["forehead_available"],
        "raw_observation_columns": [f"raw_{target}" for target in ALL_TARGETS],
        "condition_columns": [*CONDITION_NAMES, "z_relative_optical_exposure", "z_log2_iso_condition"],
        "device_condition_columns": ["camera_id"],
        "predicted_condition_nn_columns": predicted, "residual_nn_columns": residual,
        "calibrated_nn_optical_feature_columns": calibrated, "qc_only_columns": qc,
        "forbidden_direct_nyha_classifier_columns": forbidden,
        "clinical_or_label_columns_loaded": [],
        "usage_note": "Use calibrated_nn optical features only in later controlled comparisons; acquisition conditions remain diagnostic.",
    }


def validate_oof(
    oof: pd.DataFrame,
    first_stage: pd.DataFrame,
    stage2a_oof: pd.DataFrame,
    split_info: Mapping[str, Any],
) -> dict[str, Any]:
    errors: list[str] = []
    if len(oof) != 500 or oof["ID"].nunique() != 500:
        errors.append("OOF is not 500 rows / 500 unique IDs")
    expected_fold = {
        str(identifier): fold for fold in range(5)
        for identifier in split_info["splits"][fold]["val"]["ID"]
    }
    actual_fold = dict(zip(oof["ID"].astype(str), oof["fold"].astype(int)))
    if actual_fold != expected_fold:
        errors.append("OOF ID-to-fold mapping differs from fixed split")
    a = stage2a_oof.sort_values("ID", kind="stable").reset_index(drop=True)
    b = oof.sort_values("ID", kind="stable").reset_index(drop=True)
    if not a["ID"].astype(str).equals(b["ID"].astype(str)):
        errors.append("Stage 2B OOF IDs differ from Stage 2A")
    if not a["forehead_available"].astype(int).equals(b["forehead_available"].astype(int)):
        errors.append("Stage 2B forehead availability differs from Stage 2A")
    for target in ALL_TARGETS:
        if not np.allclose(a[f"raw_{target}"], b[f"raw_{target}"], rtol=0, atol=1e-12, equal_nan=True):
            errors.append(f"Stage 2B raw target differs from Stage 2A: {target}")
    cheek_columns = [
        f"{representation}_{target}" for target in CHEEK_TARGETS
        for representation in ("raw", "predicted_condition_nn", "residual_nn", "calibrated_nn")
    ]
    forehead_columns = [
        f"{representation}_{target}" for target in FOREHEAD_CHEEK_TARGETS
        for representation in ("raw", "predicted_condition_nn", "residual_nn", "calibrated_nn")
    ]
    available = b["forehead_available"].astype(int).eq(1)
    nonfinite_cheek = int((~np.isfinite(b[cheek_columns].to_numpy(float))).sum())
    illegal_forehead = int(b.loc[~available, forehead_columns].notna().sum().sum())
    if nonfinite_cheek:
        errors.append("OOF contains nonfinite cheek outputs")
    if not np.isfinite(b.loc[available, forehead_columns].to_numpy(float)).all():
        errors.append("OOF contains nonfinite available-forehead outputs")
    if illegal_forehead:
        errors.append("OOF contains illegal unavailable-forehead outputs")
    forbidden_tokens = ("nyha", "label", "sex", "bnp", "pred_class", "brightnessvalue", "path")
    illegal_columns = [column for column in b.columns if any(token in column.casefold() for token in forbidden_tokens)]
    if illegal_columns:
        errors.append(f"OOF contains forbidden columns: {illegal_columns}")
    return {
        "status": "PASS" if not errors else "FAIL", "errors": errors,
        "oof_rows": len(b), "oof_unique_ids": int(b["ID"].nunique()),
        "forehead_available_n": int(available.sum()), "forehead_unavailable_n": int((~available).sum()),
        "nonfinite_cheek_output_n": nonfinite_cheek, "illegal_forehead_output_n": illegal_forehead,
    }


def summarize_results(
    args: argparse.Namespace,
    first_stage: pd.DataFrame,
    stage2a: Mapping[str, Any],
    split_info: Mapping[str, Any],
    fold_results: Mapping[int, Mapping[str, Any]],
    experiment_root: Path,
) -> dict[str, Any]:
    summary_dir = experiment_root / "summary"
    oof = pd.concat([fold_results[fold]["val_features"] for fold in range(5)], ignore_index=True)
    oof = oof.sort_values("ID", kind="stable").reset_index(drop=True)
    validation = validate_oof(oof, first_stage, stage2a["oof"], split_info)
    if validation["status"] != "PASS":
        raise Stage2BFailure("oof_validation", validation["errors"])
    write_csv(oof, summary_dir / "oof_nn_calibrated_features.csv")
    diagnostics = {
        family: pd.concat([fold_results[fold]["diagnostics"][family] for fold in range(5)], ignore_index=True)
        for family in ("fit", "correlation", "camera", "variance", "generalization")
    }
    write_csv(combined_diagnostics(diagnostics), summary_dir / "fold_diagnostics_all.csv")
    range_audit = pd.concat([fold_results[fold]["range_audit"] for fold in range(5)], ignore_index=True)
    write_csv(range_audit, summary_dir / "condition_range_audit_all.csv")
    per_fold, comparison_summary = compare_stage2a_stage2b(stage2a["oof"], oof)
    write_csv(per_fold, summary_dir / "stage2a_vs_stage2b_per_fold.csv")
    write_csv(comparison_summary, summary_dir / "stage2a_vs_stage2b_summary.csv")
    write_json(build_feature_schema(), summary_dir / "calibration_stage2b_feature_schema.json")
    return {
        "fold_results": dict(fold_results), "oof": oof, "validation": validation,
        "diagnostics": diagnostics, "range_audit": range_audit,
        "comparison_per_fold": per_fold, "comparison_summary": comparison_summary,
    }


def markdown_table(frame: pd.DataFrame, columns: Sequence[str], max_rows: int = 30) -> str:
    view = frame.loc[:, list(columns)].head(max_rows).copy()
    for column in view.select_dtypes(include=[np.number]).columns:
        view[column] = view[column].map(lambda value: round(float(value), 6) if pd.notna(value) else "NaN")
    header = "| " + " | ".join(columns) + " |"
    divider = "| " + " | ".join(["---"] * len(columns)) + " |"
    rows = ["| " + " | ".join(str(value).replace("|", "\\|") for value in row) + " |" for row in view.itertuples(index=False, name=None)]
    return "\n".join([header, divider, *rows])


def write_report_outputs(
    args: argparse.Namespace,
    result: Mapping[str, Any],
    stage2a: Mapping[str, Any],
) -> None:
    report = args.report_output_dir
    write_csv(result["comparison_summary"], report / "stage2a_vs_stage2b_summary.csv")
    write_csv(result["diagnostics"]["fit"], report / "conditional_prediction_metrics.csv")
    write_csv(result["diagnostics"]["correlation"], report / "residual_exif_correlations.csv")
    write_csv(
        camera_representation_comparison(stage2a["oof"], result["oof"]),
        report / "raw_ridge_mlp_camera_differences.csv",
    )
    variance = result["comparison_per_fold"].loc[
        result["comparison_per_fold"]["metric_family"].eq("variance")
    ].rename(columns={
        "stage2a_value": "variance_retention_A",
        "stage2b_value": "variance_retention_B",
        "delta_b_minus_a": "delta_variance_retention_B_minus_A",
    })
    write_csv(variance, report / "variance_retention_comparison.csv")
    write_csv(result["range_audit"], report / "condition_range_audit.csv")
    curves = report / "training_curves"
    curves.mkdir(parents=True, exist_ok=True)
    for fold in range(5):
        for network in ("cheek", "forehead_cheek"):
            source = args.experiment_output_dir / f"fold_{fold}/{network}_epoch_selection/selection_curve.png"
            shutil.copy2(source, curves / f"fold_{fold}_{network}_selection_curve.png")


def write_report(
    args: argparse.Namespace,
    result: Mapping[str, Any],
    stage2a: Mapping[str, Any],
    split_info: Mapping[str, Any],
    unit_status: str,
    protocol_status: str,
    deterministic_match: bool,
) -> Path:
    fold_summary = pd.DataFrame([result["fold_results"][fold]["summary"] for fold in range(5)])
    fit_val = result["diagnostics"]["fit"].loc[result["diagnostics"]["fit"]["split_role"].eq("val")]
    corr_val = result["diagnostics"]["correlation"].loc[
        result["diagnostics"]["correlation"]["split_role"].eq("val")
        & result["diagnostics"]["correlation"]["scope"].eq("overall")
        & result["diagnostics"]["correlation"]["representation"].isin(["raw", "calibrated_nn"])
    ]
    generalization = result["diagnostics"]["generalization"]
    comparison = result["comparison_per_fold"]
    rmse = comparison.loc[(comparison["metric_family"] == "conditional_prediction") & (comparison["metric_name"] == "rmse")]
    r2 = comparison.loc[(comparison["metric_family"] == "conditional_prediction") & (comparison["metric_name"] == "r2")]
    b_wins = int(comparison.loc[comparison["better_direction"].ne("neutral"), "stage2b_better"].sum())
    a_wins = int(comparison.loc[comparison["better_direction"].ne("neutral"), "stage2a_better"].sum())
    extrapolation_by_fold = result["range_audit"].groupby("fold", as_index=False).agg(
        outside_n=("any_condition_outside_train_range", "sum"),
        both_outside_n=("both_conditions_outside_train_range", "sum"),
    )
    variance = comparison.loc[comparison["metric_family"].eq("variance")]
    lower_variance_n = int((variance["delta_b_minus_a"] < 0).sum())
    overfit_n = int((generalization["outer_val_minus_train_mse"] > 0).sum())
    selected = fold_summary[["fold", "cheek_selected_epoch", "forehead_cheek_selected_epoch"]]
    comparison_fold_summary = result["comparison_summary"].loc[
        result["comparison_summary"]["metric_family"].eq("conditional_prediction")
    ]
    fold4_device = comparison.loc[
        comparison["metric_family"].eq("camera_difference")
        & comparison["metric_name"].eq("absolute_standardized_mean_difference")
        & comparison["fold"].eq(4)
    ]
    fold4_device_a = float(fold4_device["stage2a_value"].mean())
    fold4_device_b = float(fold4_device["stage2b_value"].mean())
    files = [*IMPLEMENTATION_FILES]
    files.extend(
        project_relative(path) for root in (args.experiment_output_dir, args.report_output_dir)
        for path in sorted(root.rglob("*")) if path.is_file() and not path.name.startswith(".")
    )
    files.extend([
        project_relative(args.experiment_output_dir / "summary/run_manifest.json"),
        project_relative(args.report_output_dir / "optical_condition_calibration_stage2b_report.md"),
        project_relative(args.report_output_dir / "run.log"),
    ])
    lines = [
        REPORT_MARKER, f"# {TASK_TITLE}", "",
        "## 1. 完成状态", "", "- `OPTICAL_CALIBRATION_STAGE2B_STATUS=COMPLETE`",
        "- 本产物是EXIF-conditioned nonlinear acquisition response calibration / 非线性采集条件校准后的区域光学表型，也可称physics-inspired inverse representation；不是皮肤真实反射率、传感器线性RGB或医学指标。", "",
        "## 2. 新增/修改文件", "", *[f"- `{path}`" for path in dict.fromkeys(files)], "",
        "## 3. 第一阶段和Stage 2A输入来源", "",
        f"第一阶段：`{project_relative(args.first_stage_csv)}`，SHA256 `{sha256_file(args.first_stage_csv)}`；Stage 2A manifest SHA256 `{sha256_file(args.stage2a_run_manifest)}`；Stage 2A OOF SHA256 `{sha256_file(args.stage2a_oof)}`。", "",
        "## 4. split和SHA256", "", f"固定普通五折split SHA256为`{split_info['split_sha256']}`，未生成或修改外层split。", "",
        markdown_table(fold_summary, ["fold", "outer_train_n", "outer_val_n", "cheek_final_train_n", "forehead_cheek_final_train_n"]), "",
        "## 5. 未读取NYHA声明", "", "仅白名单读取ID、fold、camera、两个派生EXIF条件、forehead_available、六个观测和Stage 2A校准/诊断字段；`clinical_labels_loaded=false`、`nyha_used=false`、`global_features_used=false`。", "",
        "## 6. Stage 2B数学定义", "", "`predicted_condition_nn=MLP(c)`（先从target-z恢复原尺度）；`residual_nn=raw-predicted_condition_nn`；`calibrated_nn=residual_nn+outer_train_target_mean`。网络预测条件期望观测，不解释为真实曝光成分或真实设备响应。", "",
        "## 7. MLP输入和输出", "", "输入固定为`camera_xiaomi,z_exposure,z_iso,camera_xiaomi*z_exposure,camera_xiaomi*z_iso`；输出为三个标准化目标。原始区域观测、Stage 2A预测、QC、图像和临床字段均不进入网络。", "",
        "## 8. 两个独立网络", "", "每fold分别训练Cheek与Forehead-cheek网络；不共享参数。后者仅使用forehead_available=1且三目标有限的病例。", "",
        "## 9. 网络结构及参数量", "", "固定`Linear(5,8)-Tanh-Linear(8,8)-Tanh-Linear(8,3)`，单网络147参数、每fold两网络294参数；无BatchNorm、LayerNorm、Dropout、attention、卷积或自编码器。", "",
        "## 10. 条件标准化", "", "内部epoch选择仅由inner-train按设备拟合condition scaler；完整重拟合严格复用Stage 2A外层condition scaler。Stage 2B复算z与Stage 2A最大绝对误差见下表。", "",
        markdown_table(fold_summary, ["fold", "stage2a_z_max_absolute_error"]), "",
        "## 11. 目标标准化", "", "Cheek与Forehead-cheek分别在当前训练子集逐目标计算population mean/std（ddof=0）；std<1e-8会停止该fold。内部选择和最终重拟合分别重新拟合target scaler。", "",
        "## 12. 内部epoch选择协议", "", "按camera分层，以`SHA256(seed|ID)`确定80/20 inner split；外层val未加载。指标为inner-val标准化MSE，改善阈值1e-6，并列保留更早epoch。", "",
        "## 13. 完整训练折重拟合", "", "选定epoch后丢弃inner checkpoint参数，以同一预设seed全新初始化；Cheek用400例，Forehead-cheek用全部额部可用outer-train病例，固定训练到selected epoch，无validation loader。", "",
        "## 14. 固定训练配置", "", "CPU/float32、AdamW、lr=1e-2、weight_decay=1e-4、full-batch、max_epochs=500、minimum_epochs=50、patience=50、gradient clip=5、无scheduler/AMP/搜索。", "",
        "## 15. 额部缺失处理", "", "额部不可用病例保留；三个Forehead-cheek目标的raw/predicted/residual/calibrated均为NaN，不参与内部split、scaler、训练或loss；cheek输出保留。", "",
        "## 16. 五折selected epoch", "", markdown_table(selected, selected.columns), "",
        "## 17. inner train/val训练曲线", "", "每fold、每网络的完整training_log、best checkpoint、selected_epoch和曲线均已保存；报告目录`training_curves/`包含10张选择曲线。", "",
        markdown_table(generalization, ["fold", "network_type", "selected_epoch", "inner_train_best_epoch_loss", "inner_val_best_epoch_loss", "final_outer_train_loss", "outer_val_mse", "outer_val_minus_train_mse"]), "",
        "## 18. 外层条件预测MAE、RMSE和R²", "", markdown_table(fit_val, ["fold", "target", "valid_n", "mae", "rmse", "r2"], 30), "",
        "## 19. EXIF残余相关性", "", markdown_table(corr_val, ["fold", "target", "representation", "condition", "valid_n", "spearman_rho"], 40), "",
        "## 20. 校准前后设备差异", "", "完整raw、Ridge和MLP设备差异见`raw_ridge_mlp_camera_differences.csv`；设备差异降低不自动代表方法更优。", "",
        "## 21. 方差保留", "", f"30个target/fold中，Stage 2B方差保留低于Stage 2A的有{lower_variance_n}个；方差降低可能代表删除个体变化，不定义单向优劣。", "",
        "## 22. 条件范围外推病例", "", markdown_table(extrapolation_by_fold, ["fold", "outside_n", "both_outside_n"]), "",
        f"fold 4范围外病例={int(extrapolation_by_fold.loc[extrapolation_by_fold['fold'].eq(4), 'outside_n'].iloc[0])}；六目标绝对标准化设备均值差的fold内平均为Stage 2A={fold4_device_a:.6g}、Stage 2B={fold4_device_b:.6g}（B−A={fold4_device_b - fold4_device_a:.6g}）。因此fold 4并非一致改善，报告设备差异时需同时参考这一外推负担；未据此重训、clamp或调参。", "",
        "## 23. Stage 2A与Stage 2B逐折比较", "", markdown_table(rmse, ["fold", "target", "stage2a_value", "stage2b_value", "delta_b_minus_a", "stage2b_better", "stage2a_better"], 30), "",
        "## 24. Stage 2B优于Stage 2A的fold数量", "", f"所有有方向指标的target/fold条目中，B优于A={b_wins}，A优于B={a_wins}。条件预测RMSE平均B−A={rmse['delta_b_minus_a'].mean():.6g}；R²平均B−A={r2['delta_b_minus_a'].mean():.6g}。逐目标、逐指标fold胜负如下；结果存在fold间不一致，不只报告B占优结果。", "",
        markdown_table(comparison_fold_summary, ["target", "metric_name", "stage2b_better_fold_n", "stage2a_better_fold_n", "tie_fold_n", "mean_delta_b_minus_a"], 30), "",
        "## 25. OOF完整性", "", f"OOF={result['validation']['oof_rows']}行、唯一ID={result['validation']['oof_unique_ids']}；额部可用={result['validation']['forehead_available_n']}、不可用={result['validation']['forehead_unavailable_n']}；非有限cheek={result['validation']['nonfinite_cheek_output_n']}；非法额部输出={result['validation']['illegal_forehead_output_n']}。", "",
        "## 26. 单元和协议测试", "", f"单元测试=`{unit_status}`；协议测试=`{protocol_status}`。", "",
        "## 27. 确定性验证", "", f"在临时目录重复完整五折核心流程，fold/summary的CSV、JSON和PTH SHA256一致：`{deterministic_match}`。", "",
        "## 28. 历史输入未修改声明", "", "第一阶段、Stage 2A、十份固定split运行前后SHA256一致；`historical_inputs_modified=false`。", "",
        "## 29. 局限性", "",
        "1. 小型MLP可能捕获非线性，也可能过拟合；本次10个网络中外层val MSE高于完整train MSE的有" + str(overfit_n) + "个。",
        "2. 两设备与有限条件范围不支持未知设备或广泛外推；范围外病例未删除或截断。",
        "3. 若A/B接近，应优先保留更简单、可解释的Stage 2A。",
        "4. 是否进入NYHA模型必须等待后续Raw、A、B和Global融合对照；本任务不提前决定。",
        "5. 输出不是皮肤真实反射率、传感器线性RGB、生理参数或强意义物理反演。", "",
        "## 30. 后续可用于NYHA实验的六个Stage 2B字段", "",
        *[f"- `calibrated_nn_{target}`" for target in ALL_TARGETS], "",
    ]
    path = args.report_output_dir / "optical_condition_calibration_stage2b_report.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def run_test_file(root: Path, filename: str) -> tuple[str, str]:
    result = subprocess.run([sys.executable, "-m", "pytest", "-q", filename], cwd=root, text=True, capture_output=True)
    return ("PASS" if result.returncode == 0 else "FAIL", (result.stdout + "\n" + result.stderr).strip())


def deterministic_files(experiment_root: Path) -> dict[str, str]:
    output: dict[str, str] = {}
    for path in sorted(experiment_root.rglob("*")):
        if not path.is_file() or path.name.startswith(".") or path.suffix.lower() not in (".csv", ".json", ".pth"):
            continue
        relative = path.relative_to(experiment_root).as_posix()
        if not (relative.startswith("fold_") or relative.startswith("summary/")):
            continue
        if relative == "summary/run_manifest.json":
            continue
        output[relative] = sha256_file(path)
    return output


def write_run_manifest(
    args: argparse.Namespace,
    result: Mapping[str, Any],
    split_info: Mapping[str, Any],
    unit_status: str,
    protocol_status: str,
    deterministic_match: bool,
    input_hashes: Mapping[str, str],
) -> None:
    folds: dict[str, Any] = {}
    for fold in range(5):
        fold_dir = args.experiment_output_dir / f"fold_{fold}"
        folds[str(fold)] = {
            "inner_split_manifest_sha256": sha256_file(fold_dir / "inner_split/inner_split_manifest.json"),
            "cheek_selected_epoch": result["fold_results"][fold]["summary"]["cheek_selected_epoch"],
            "forehead_cheek_selected_epoch": result["fold_results"][fold]["summary"]["forehead_cheek_selected_epoch"],
            "cheek_final_model_sha256": sha256_file(fold_dir / "cheek_final_model.pth"),
            "forehead_cheek_final_model_sha256": sha256_file(fold_dir / "forehead_cheek_final_model.pth"),
            "cheek_final_train_n": result["fold_results"][fold]["summary"]["cheek_final_train_n"],
            "forehead_cheek_final_train_n": result["fold_results"][fold]["summary"]["forehead_cheek_final_train_n"],
        }
    manifest = {
        "task": TASK, "status": "COMPLETE",
        "method": "EXIF-conditioned nonlinear acquisition response calibration",
        "representation_name": "physics-inspired inverse representation",
        "architecture": ARCHITECTURE, "parameter_count_per_network": 147,
        "parameter_count_two_networks_per_fold": 294, "activation": "Tanh",
        "optimizer": "AdamW", "learning_rate": 1.0e-2, "weight_decay": 1.0e-4,
        "max_epochs": 500, "minimum_epochs": 50, "patience": 50,
        "minimum_improvement": 1.0e-6, "gradient_clip_max_norm": 5.0,
        "batch_mode": "full_batch", "device": "cpu", "dtype": "float32",
        "seed_rule": {
            "inner_split": "2026 + outer_fold * 100",
            "cheek": "2026 + outer_fold * 100 + 1",
            "forehead_cheek": "2026 + outer_fold * 100 + 2",
        },
        "first_stage_input": {"path": project_relative(args.first_stage_csv), "sha256": sha256_file(args.first_stage_csv)},
        "stage2a_run_manifest": {"path": project_relative(args.stage2a_run_manifest), "sha256": sha256_file(args.stage2a_run_manifest)},
        "stage2a_oof": {"path": project_relative(args.stage2a_oof), "sha256": sha256_file(args.stage2a_oof)},
        "split": {"path": project_relative(args.split_dir), "sha256": split_info["split_sha256"]},
        "per_fold": folds,
        "oof_output_sha256": sha256_file(args.experiment_output_dir / "summary/oof_nn_calibrated_features.csv"),
        "config": {"path": project_relative(args.config_path), "sha256": sha256_file(args.config_path)},
        "code_sha256": {path: sha256_file(args.project_root / path) for path in IMPLEMENTATION_FILES[:4]},
        "tests": {"unit": unit_status, "protocol": protocol_status},
        "deterministic_repeat_match": deterministic_match,
        "pytorch_version": torch.__version__, "python_version": platform.python_version(),
        "numpy_version": np.__version__, "pandas_version": pd.__version__,
        "git_commit": get_git_commit(args.project_root),
        "clinical_labels_loaded": False, "nyha_used": False, "global_features_used": False,
        "outer_validation_tuning": False, "architecture_search": False,
        "hyperparameter_search": False, "historical_inputs_modified": False,
        "input_sha256_before_and_after": dict(input_hashes),
    }
    write_json(manifest, args.experiment_output_dir / "summary/run_manifest.json")


def final_lines(args: argparse.Namespace, result: Mapping[str, Any], split_info: Mapping[str, Any]) -> list[str]:
    folds = [result["fold_results"][fold]["summary"] for fold in range(5)]
    comparison = result["comparison_per_fold"]
    rmse = comparison.loc[(comparison["metric_family"] == "conditional_prediction") & (comparison["metric_name"] == "rmse")]
    r2 = comparison.loc[(comparison["metric_family"] == "conditional_prediction") & (comparison["metric_name"] == "r2")]
    directional = comparison.loc[comparison["better_direction"].ne("neutral")]
    files = [*IMPLEMENTATION_FILES]
    files.extend(
        project_relative(path) for root in (args.experiment_output_dir, args.report_output_dir)
        for path in sorted(root.rglob("*")) if path.is_file() and not path.name.startswith(".")
    )
    files.extend([
        project_relative(args.experiment_output_dir / "summary/run_manifest.json"),
        project_relative(args.report_output_dir / "optical_condition_calibration_stage2b_report.md"),
        project_relative(args.report_output_dir / "run.log"),
    ])
    return [
        "OPTICAL_CALIBRATION_STAGE2B_STATUS=COMPLETE",
        "ADDED_OR_MODIFIED_FILES=" + ",".join(dict.fromkeys(files)),
        f"FIRST_STAGE_INPUT_SHA256={sha256_file(args.first_stage_csv)}",
        f"STAGE2A_RUN_MANIFEST_SHA256={sha256_file(args.stage2a_run_manifest)}",
        f"STAGE2A_OOF_SHA256={sha256_file(args.stage2a_oof)}",
        f"SPLIT_SHA256={split_info['split_sha256']}", "PREFLIGHT_STATUS=PASS",
        "UNIT_TEST_STATUS=PASS", "PROTOCOL_TEST_STATUS=PASS", "COMPLETED_FOLD_COUNT=5",
        "CHEEK_SELECTED_EPOCHS=" + ",".join(f"{row['fold']}:{row['cheek_selected_epoch']}" for row in folds),
        "FOREHEAD_CHEEK_SELECTED_EPOCHS=" + ",".join(f"{row['fold']}:{row['forehead_cheek_selected_epoch']}" for row in folds),
        "CHEEK_FINAL_TRAIN_COUNTS=" + ",".join(f"{row['fold']}:{row['cheek_final_train_n']}" for row in folds),
        "FOREHEAD_CHEEK_FINAL_TRAIN_COUNTS=" + ",".join(f"{row['fold']}:{row['forehead_cheek_final_train_n']}" for row in folds),
        f"OOF_ROWS={result['validation']['oof_rows']}", f"OOF_UNIQUE_IDS={result['validation']['oof_unique_ids']}",
        f"FOREHEAD_AVAILABLE_COUNT={result['validation']['forehead_available_n']}",
        f"FOREHEAD_UNAVAILABLE_COUNT={result['validation']['forehead_unavailable_n']}",
        f"NONFINITE_CHEEK_OUTPUT_COUNT={result['validation']['nonfinite_cheek_output_n']}",
        f"ILLEGAL_FOREHEAD_OUTPUT_COUNT={result['validation']['illegal_forehead_output_n']}",
        f"CONDITION_RANGE_OUTSIDE_VAL_COUNT={int(result['range_audit']['any_condition_outside_train_range'].sum())}",
        f"MEAN_DELTA_RMSE_B_MINUS_A={float(rmse['delta_b_minus_a'].mean()):.17g}",
        f"MEAN_DELTA_R2_B_MINUS_A={float(r2['delta_b_minus_a'].mean()):.17g}",
        f"STAGE2B_BETTER_TARGET_FOLD_METRIC_COUNT={int(directional['stage2b_better'].sum())}",
        f"OOF_PATH={args.experiment_output_dir / 'summary/oof_nn_calibrated_features.csv'}",
        f"FEATURE_SCHEMA_PATH={args.experiment_output_dir / 'summary/calibration_stage2b_feature_schema.json'}",
        f"STAGE2A_VS_STAGE2B_PATH={args.experiment_output_dir / 'summary/stage2a_vs_stage2b_per_fold.csv'}",
        f"REPORT_PATH={args.report_output_dir / 'optical_condition_calibration_stage2b_report.md'}",
        "CLINICAL_LABELS_LOADED=false", "NYHA_USED=false", "OUTER_VALIDATION_TUNING=false",
        "ARCHITECTURE_SEARCH=false", "HYPERPARAMETER_SEARCH=false",
        "HISTORICAL_INPUTS_MODIFIED=false", "ALL_ACCEPTANCE_CRITERIA_MET=YES",
    ]


def write_failure_log_safely(args: argparse.Namespace | None, failure: Stage2BFailure) -> Path | None:
    if args is None:
        return None
    root = args.report_output_dir
    try:
        if root.exists() and any(root.iterdir()) and not verify_owned_output(root, report=True):
            return None
        root.mkdir(parents=True, exist_ok=True)
        write_json({"task": TASK}, root / ".stage2b_owner.json")
        path = root / "run_failure.log"
        path.write_text(
            f"TASK={TASK}\nSTATUS=FAILED\nSTAGE={failure.stage}\n" + "\n".join(failure.errors) + "\n",
            encoding="utf-8",
        )
        return path
    except Exception:
        return None


def main(argv: Sequence[str] | None = None) -> int:
    args: argparse.Namespace | None = None
    try:
        args = load_args(argv)
        first_stage = load_first_stage(args)
        stage2a = load_stage2a_inputs(args)
        split_info = audit_splits(args, first_stage)
        historical_before = input_hash_inventory(args, split_info)
        print(f"FIRST_STAGE_PATH={args.first_stage_csv} SHA256={sha256_file(args.first_stage_csv)}")
        print(f"STAGE2A_RUN_MANIFEST_PATH={args.stage2a_run_manifest} SHA256={sha256_file(args.stage2a_run_manifest)}")
        print(f"STAGE2A_OOF_PATH={args.stage2a_oof} SHA256={sha256_file(args.stage2a_oof)}")
        print(f"SPLIT_SHA256={split_info['split_sha256']}")
        print(f"STAGE2A_STATUS={stage2a['manifest']['status']} PYTORCH_VERSION={torch.__version__} CUDA_AVAILABLE={torch.cuda.is_available()} DEVICE=cpu")
        print("PROPOSED_INPUT_READS_CLINICAL_OR_NYHA=false")
        prepare_outputs(args)
        write_protocol(args, first_stage, stage2a, split_info)
        if args.protocol_only:
            print("OPTICAL_CALIBRATION_STAGE2B_STATUS=PROTOCOL_ONLY_COMPLETE")
            print("PREFLIGHT_STATUS=PASS")
            return 0
        unit_files = ["tests/test_exif_conditioned_response_mlp.py", "tests/test_optical_condition_calibration_stage2b.py"]
        unit_outputs = []
        for filename in unit_files:
            status, output = run_test_file(args.project_root, filename)
            unit_outputs.append(output)
            if status != "PASS":
                raise Stage2BFailure("unit_tests", [f"{filename}: {output}"])
        protocol_status, protocol_output = run_test_file(
            args.project_root, "tests/test_optical_calibration_stage2b_protocol.py"
        )
        if protocol_status != "PASS":
            raise Stage2BFailure("protocol_tests", [protocol_output])
        context = {
            "first_stage_sha256": sha256_file(args.first_stage_csv),
            "stage2a_manifest_sha256": sha256_file(args.stage2a_run_manifest),
            "git_commit": get_git_commit(args.project_root),
        }
        if args.summarize_only or args.compare_only:
            fold_results = {fold: load_fold_result(args.experiment_output_dir, fold) for fold in range(5)}
            result = summarize_results(args, first_stage, stage2a, split_info, fold_results, args.experiment_output_dir)
            existing_manifest = json.loads((args.experiment_output_dir / "summary/run_manifest.json").read_text(encoding="utf-8"))
            deterministic_match = bool(existing_manifest.get("deterministic_repeat_match", False))
        else:
            selected_folds = tuple(range(5)) if args.fold == "all" else (int(args.fold),)
            fold_results: dict[int, dict[str, Any]] = {}
            for fold in selected_folds:
                if (args.resume or args.skip_completed) and completed_fold(args.experiment_output_dir, fold):
                    fold_results[fold] = load_fold_result(args.experiment_output_dir, fold)
                    print(f"FOLD={fold} STATUS=SKIPPED_COMPLETED", flush=True)
                else:
                    fold_results[fold] = run_fold(args, fold, first_stage, split_info, context, args.experiment_output_dir)
                    print(f"FOLD={fold} STATUS=COMPLETE", flush=True)
            if set(selected_folds) != set(range(5)):
                print("OPTICAL_CALIBRATION_STAGE2B_STATUS=PARTIAL")
                print(f"COMPLETED_FOLD_COUNT={len(selected_folds)}")
                print("ALL_ACCEPTANCE_CRITERIA_MET=NO")
                return 0
            result = summarize_results(args, first_stage, stage2a, split_info, fold_results, args.experiment_output_dir)
            official_hashes = deterministic_files(args.experiment_output_dir)
            with tempfile.TemporaryDirectory(prefix=".stage2b_determinism_", dir=args.experiment_output_dir) as temp_name:
                temp_root = Path(temp_name) / "experiment"
                repeat_folds = {
                    fold: run_fold(args, fold, first_stage, split_info, context, temp_root)
                    for fold in range(5)
                }
                repeat_result = summarize_results(args, first_stage, stage2a, split_info, repeat_folds, temp_root)
                repeat_hashes = deterministic_files(temp_root)
                deterministic_match = official_hashes == repeat_hashes
                if not deterministic_match:
                    missing = sorted(set(official_hashes).symmetric_difference(repeat_hashes))
                    changed = sorted(
                        key for key in set(official_hashes) & set(repeat_hashes)
                        if official_hashes[key] != repeat_hashes[key]
                    )
                    raise Stage2BFailure("determinism", [f"missing={missing}", f"changed={changed}"])
                if not result["oof"].equals(repeat_result["oof"]):
                    raise Stage2BFailure("determinism", ["Repeated OOF DataFrame differs"])
        historical_after = input_hash_inventory(args, split_info)
        if historical_before != historical_after:
            raise Stage2BFailure("historical_input_integrity", ["Historical input SHA256 changed during run"])
        write_report_outputs(args, result, stage2a)
        write_report(args, result, stage2a, split_info, "PASS", protocol_status, deterministic_match)
        write_run_manifest(
            args, result, split_info, "PASS", protocol_status, deterministic_match, historical_before
        )
        lines = final_lines(args, result, split_info)
        (args.report_output_dir / "run.log").write_text("\n".join(lines) + "\n", encoding="utf-8")
        print("\n".join(lines))
        return 0
    except Exception as exc:
        failure = exc if isinstance(exc, Stage2BFailure) else Stage2BFailure(
            "unexpected_error", [f"{type(exc).__name__}: {exc}"]
        )
        failure_path = write_failure_log_safely(args, failure)
        print("OPTICAL_CALIBRATION_STAGE2B_STATUS=FAILED")
        print(f"FAILED_STAGE={failure.stage}")
        print("ERRORS=" + " | ".join(failure.errors))
        print(f"ERROR_LOG_PATH={failure_path if failure_path else 'unavailable'}")
        print("EXACT_RESUME_COMMAND=python scripts/train/run_optical_condition_calibration_stage2b.py --config config/train/optical_condition_calibration_stage2b.yaml --fold all --resume --skip-completed")
        print("ALL_ACCEPTANCE_CRITERIA_MET=NO")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
