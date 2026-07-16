"""Preflight and safe orchestration for the five-variant fusion experiment."""

from __future__ import annotations

import argparse
import json
import os
import platform
import re
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
import torch
import torchvision


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.experiment_utils import load_yaml  # noqa: E402
from utils.optical_feature_preprocessor import (  # noqa: E402
    AVAILABILITY_COLUMN,
    RAW_FEATURE_COLUMNS,
    STAGE2A_FEATURE_COLUMNS,
    STAGE2B_FEATURE_COLUMNS,
    VARIANTS,
    VARIANT_AUX_DIM,
    assert_classifier_feature_path,
    code_sha256,
    relative_path,
    sha256_file,
    sha256_ids,
    validate_feature_schema,
)


EXPECTED_TRAIN = {
    "device": "auto", "batch_size": 16, "epochs": 50, "optimizer": "AdamW",
    "learning_rate": 0.0001, "weight_decay": 0.0001, "scheduler": "none",
    "warmup": "none", "gradient_clipping": "none", "loss": "weighted_cross_entropy",
    "label_smoothing": 0, "amp": False, "early_stopping_patience": 10,
    "monitor_metric": "macro_auc", "monitor_mode": "max", "minimum_improvement": 0,
    "tie_breaking": "earlier_epoch", "seed": 2026, "num_workers": 0,
    "pin_memory": False,
}
EXPECTED_EXPERIMENT = {
    "name": "GlobalResNet18_OpticalFusion_NYHA3Class_5Fold",
    "output_root": "experiments/global_resnet18_optical_fusion",
    "variants": list(VARIANTS),
}
EXPECTED_DATA = {
    "image_root": "data/processed/global_face/preprocess_ablation/hybrid_imagenet_meanbg/images",
    "image_filename_template": "{ID}.png",
    "label_path": "data/raw/label_raw_nyha2_remove22_sex_balanced_500.csv",
    "split_root": "data/processed/splits_500",
    "master_split": "data/processed/splits_500/nyha_3class_sex_stratified_group_5fold.csv",
    "train_csv_pattern": "fold_{fold}_train.csv",
    "val_csv_pattern": "fold_{fold}_val.csv",
    "folds": [0, 1, 2, 3, 4],
    "expected_num_samples": 500,
    "expected_patient_groups": 483,
    "expected_multi_image_patient_groups": 17,
    "expected_class_counts": [115, 237, 148],
}
EXPECTED_FEATURES = {
    "raw_source": "data/processed/optical_observations_v1/regional_optical_observations.csv",
    "raw_schema": "data/processed/optical_observations_v1/feature_schema.json",
    "raw_manifest": "data/processed/optical_observations_v1/extraction_manifest.json",
    "stage2a_root": "experiments/optical_condition_calibration_stage2a",
    "stage2a_schema": "experiments/optical_condition_calibration_stage2a/summary/calibration_feature_schema.json",
    "stage2a_manifest": "experiments/optical_condition_calibration_stage2a/summary/run_manifest.json",
    "stage2a_oof": "experiments/optical_condition_calibration_stage2a/summary/oof_calibrated_features.csv",
    "stage2b_root": "experiments/optical_condition_calibration_stage2b",
    "stage2b_schema": "experiments/optical_condition_calibration_stage2b/summary/calibration_stage2b_feature_schema.json",
    "stage2b_manifest": "experiments/optical_condition_calibration_stage2b/summary/run_manifest.json",
    "stage2b_oof": "experiments/optical_condition_calibration_stage2b/summary/oof_nn_calibrated_features.csv",
    "forbid_oof_as_classifier_input": True,
    "strict_positive_allowlist": True,
}
EXPECTED_MODEL = {
    "backbone": "resnet18", "pretrained": "imagenet", "num_classes": 3,
    "global_feature_dim": 512, "direct_concat": True, "freeze_backbone": False,
}
EXPECTED_TRANSFORMS = {
    "image_size": 224, "resize": [224, 224],
    "horizontal_flip_probability": 0.5, "imagenet_normalization": True,
    "mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225],
    "color_jitter": False, "random_crop": False,
}
EXPECTED_STANDARDIZATION = {
    "ddof": 0, "std_epsilon": 1e-8,
    "missing_fill_after_standardization": 0, "train_only": True,
}
EXPECTED_SUMMARY = {
    "bootstrap_repetitions": 2000, "bootstrap_seed": 2026,
    "bootstrap_minimum_valid_repetitions": 1900,
    "bootstrap_ci": [0.025, 0.975], "expected_oof_rows": 500,
}
RAW_STAGE_COLUMNS = tuple(f"raw_{name}" for name in RAW_FEATURE_COLUMNS)
REQUIRED_TEST_FILES = (
    "tests/test_global_optical_fusion_model.py",
    "tests/test_global_optical_fusion_dataset.py",
    "tests/test_global_optical_fusion_protocol.py",
    "tests/test_global_optical_fusion_checkpoint.py",
    "tests/test_global_optical_fusion_summary.py",
    "tests/test_optical_roi_dataset_v1.py",
    "tests/test_regional_optical_observations_v1.py",
    "tests/test_optical_condition_calibration_stage2a.py",
    "tests/test_optical_calibration_fivefold_protocol.py",
    "tests/test_optical_condition_calibration_stage2b.py",
    "tests/test_optical_calibration_stage2b_protocol.py",
)
TRAINING_IMPLEMENTATION_FILES = (
    "models/resnet18_optical_fusion.py",
    "datasets/global_optical_fusion_dataset.py",
    "utils/optical_feature_preprocessor.py",
    "trainers/global_optical_fusion_trainer.py",
    "evaluators/global_optical_fusion_evaluator.py",
    "scripts/train/train_global_optical_fusion_5fold.py",
    "scripts/run/run_global_optical_fusion_5fold.py",
    "config/train/global_optical_fusion/global_resnet18_optical_fusion.yaml",
)
IMPLEMENTATION_FILES = (
    *TRAINING_IMPLEMENTATION_FILES,
    "scripts/evaluate/summarize_global_optical_fusion_5fold.py",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--variant", action="append", default=[])
    parser.add_argument("--fold", action="append", default=[])
    parser.add_argument("--protocol-only", action="store_true")
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--summarize-only", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--skip-completed", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--allow-cpu-training", action="store_true")
    return parser.parse_args()


def project_path(value: str | Path) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _write_json(payload: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, default=str)


def _collect_git_metadata() -> dict[str, Any]:
    """Collect reproducibility metadata without making Git a runtime dependency."""
    metadata: dict[str, Any] = {
        "git_available": False,
        "git_repository": False,
        "git_branch": None,
        "git_commit": None,
        "git_error": None,
    }
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            check=False,
            timeout=10,
        )
    except FileNotFoundError:
        metadata["git_error"] = "Git executable was not found on PATH"
        return metadata
    except (OSError, subprocess.SubprocessError) as exc:
        metadata["git_error"] = f"Git metadata unavailable: {type(exc).__name__}: {exc}"
        return metadata

    metadata["git_available"] = True
    if commit.returncode != 0:
        detail = commit.stderr.strip() or commit.stdout.strip() or f"exit code {commit.returncode}"
        metadata["git_error"] = f"Project Git commit unavailable: {detail}"
        return metadata

    metadata["git_repository"] = True
    metadata["git_commit"] = commit.stdout.strip() or None
    try:
        branch = subprocess.run(
            ["git", "branch", "--show-current"],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            check=False,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        metadata["git_error"] = f"Git branch unavailable: {type(exc).__name__}: {exc}"
        return metadata

    if branch.returncode == 0:
        # An empty branch is valid when HEAD is detached.
        metadata["git_branch"] = branch.stdout.strip() or None
    else:
        detail = branch.stderr.strip() or branch.stdout.strip() or f"exit code {branch.returncode}"
        metadata["git_error"] = f"Git branch unavailable: {detail}"
    return metadata


def implementation_signature() -> str:
    return code_sha256(
        [PROJECT_ROOT / path for path in (*IMPLEMENTATION_FILES, *REQUIRED_TEST_FILES)],
        PROJECT_ROOT,
    )


def training_code_signature() -> str:
    return code_sha256(
        [PROJECT_ROOT / path for path in TRAINING_IMPLEMENTATION_FILES], PROJECT_ROOT
    )


def run_required_tests(config_path: Path, output_root: Path) -> dict[str, Any]:
    """Run and persist the exact implementation and upstream regression suite."""
    audit_path = output_root / "protocol/test_audit.json"
    signature = implementation_signature()
    config_digest = sha256_file(config_path)
    if audit_path.is_file():
        existing = _read_json(audit_path)
        if (
            existing.get("status") == "PASS"
            and existing.get("implementation_signature") == signature
            and existing.get("config_sha256") == config_digest
        ):
            print(f"TEST_STATUS=PASS_CACHED ({existing.get('passed_count')} passed)")
            return existing
    command = [
        sys.executable, "-B", "-m", "pytest", "-p", "no:cacheprovider",
        *REQUIRED_TEST_FILES, "-q",
    ]
    environment = os.environ.copy()
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    started = time.perf_counter()
    completed = subprocess.run(
        command, cwd=PROJECT_ROOT, capture_output=True, text=True,
        encoding="utf-8", errors="replace", env=environment,
    )
    output = (completed.stdout or "") + (completed.stderr or "")
    matches = re.findall(r"(\d+) passed", output)
    audit = {
        "task": "global_optical_fusion_required_tests",
        "status": "PASS" if completed.returncode == 0 else "FAIL",
        "returncode": int(completed.returncode),
        "passed_count": int(matches[-1]) if matches else None,
        "test_files": list(REQUIRED_TEST_FILES),
        "command": command,
        "implementation_signature": signature,
        "config_sha256": config_digest,
        "elapsed_seconds": time.perf_counter() - started,
        "stdout": completed.stdout,
        "stderr": completed.stderr,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    _write_json(audit, audit_path)
    if completed.returncode != 0:
        raise RuntimeError(
            f"Required tests failed; see {audit_path}. Last output:\n{output[-4000:]}"
        )
    print(f"TEST_STATUS=PASS ({audit['passed_count']} passed)")
    return audit


def smoke_evidence(output_root: Path) -> dict[str, Any]:
    runs: list[dict[str, Any]] = []
    for variant in VARIANTS:
        path = output_root / "smoke" / variant / "fold_0/fold_manifest.json"
        if not path.is_file():
            raise FileNotFoundError(f"Missing smoke manifest: {path}")
        manifest = _read_json(path)
        if (
            manifest.get("status") != "SMOKE_COMPLETE"
            or manifest.get("formal_result") is not False
            or manifest.get("full_training_executed") is not False
            or manifest.get("variant") != variant
            or int(manifest.get("prediction_rows", -1)) != 3
            or manifest.get("implementation_signature") != training_code_signature()
        ):
            raise ValueError(f"Invalid smoke evidence: {path}")
        runs.append({
            "variant": variant,
            "manifest_path": relative_path(path, PROJECT_ROOT),
            "manifest_sha256": sha256_file(path),
            "best_checkpoint_sha256": manifest.get("best_checkpoint_sha256"),
        })
    return {"status": "PASS", "completed_variants": len(runs), "runs": runs}


def ensure_smoke_suite(config_path: Path, output_root: Path) -> dict[str, Any]:
    try:
        evidence = smoke_evidence(output_root)
        print("SMOKE_TEST_STATUS=PASS_CACHED")
        return evidence
    except (FileNotFoundError, ValueError):
        pass
    for variant in VARIANTS:
        command = [
            sys.executable,
            str(PROJECT_ROOT / "scripts/train/train_global_optical_fusion_5fold.py"),
            "--config", str(config_path), "--variant", variant, "--fold", "0",
            "--output-root", str(output_root), "--smoke-test", "--overwrite",
        ]
        subprocess.run(command, cwd=PROJECT_ROOT, check=True)
    evidence = smoke_evidence(output_root)
    print("SMOKE_TEST_STATUS=PASS")
    return evidence


def _ids(frame: pd.DataFrame, path: Path) -> list[str]:
    if "ID" not in frame.columns:
        raise ValueError(f"Missing ID column: {path}")
    values = frame["ID"].astype(str).tolist()
    if len(values) != len(set(values)):
        raise ValueError(f"Duplicate IDs in {path}")
    return values


def validate_locked_config(config: dict[str, Any]) -> None:
    expected_sections = {
        "experiment": EXPECTED_EXPERIMENT,
        "data": EXPECTED_DATA,
        "features": EXPECTED_FEATURES,
        "model": EXPECTED_MODEL,
        "train": EXPECTED_TRAIN,
        "transforms": EXPECTED_TRANSFORMS,
        "feature_standardization": EXPECTED_STANDARDIZATION,
        "summary": EXPECTED_SUMMARY,
    }
    missing_sections = sorted(set(expected_sections).difference(config))
    extra_sections = sorted(set(config).difference(expected_sections))
    if missing_sections or extra_sections:
        raise ValueError(
            f"Locked config sections differ; missing={missing_sections}, extra={extra_sections}"
        )
    for section, expected in expected_sections.items():
        actual = config[section]
        if actual != expected:
            differing_keys = sorted(
                key for key in set(actual) | set(expected)
                if actual.get(key) != expected.get(key)
            )
            raise ValueError(
                f"Locked config mismatch in section {section!r}; "
                f"differing_keys={differing_keys}"
            )
    for oof_key in ("stage2a_oof", "stage2b_oof"):
        try:
            assert_classifier_feature_path(project_path(config["features"][oof_key]))
        except ValueError:
            pass
        else:
            raise ValueError(f"OOF path guard did not reject configured audit-only {oof_key}")


def _maximum_raw_difference(
    raw: pd.DataFrame, calibrated: pd.DataFrame, path: Path
) -> float:
    joined = raw.set_index("ID").loc[calibrated["ID"].astype(str)]
    left = joined.loc[:, list(RAW_FEATURE_COLUMNS)].to_numpy(float)
    right = calibrated.loc[:, list(RAW_STAGE_COLUMNS)].to_numpy(float)
    if not np.array_equal(np.isnan(left), np.isnan(right)):
        raise ValueError(f"Raw feature NaN pattern differs in {path}")
    finite = np.isfinite(left) & np.isfinite(right)
    if not finite.any():
        return 0.0
    maximum = float(np.max(np.abs(left[finite] - right[finite])))
    if maximum > 1e-10:
        raise ValueError(f"Raw feature mismatch {maximum} exceeds tolerance in {path}")
    return maximum


def run_preflight(
    config_path: str | Path, output_root: str | Path | None = None
) -> dict[str, Any]:
    """Execute all real-metadata protocol checks and write audit artifacts."""
    config_path = project_path(config_path)
    config = load_yaml(config_path)
    validate_locked_config(config)
    root = project_path(output_root or config["experiment"]["output_root"])
    protocol_dir = root / "protocol"
    protocol_dir.mkdir(parents=True, exist_ok=True)
    data, features = config["data"], config["features"]

    image_root = project_path(data["image_root"])
    image_ids = [path.stem for path in sorted(image_root.glob("*.png"))]
    if len(image_ids) != 500 or len(set(image_ids)) != 500:
        raise ValueError("Formal meanbg image root must contain exactly 500 unique PNG IDs")
    label_path = project_path(data["label_path"])
    labels = pd.read_csv(label_path, dtype={"ID": "string"}, encoding="utf-8-sig")
    label_ids = _ids(labels, label_path)
    master_path = project_path(data["master_split"])
    master = pd.read_csv(
        master_path, dtype={"ID": "string", "patient_group_id": "string"}, encoding="utf-8-sig"
    )
    master_ids = _ids(master, master_path)
    raw_path = project_path(features["raw_source"])
    raw = pd.read_csv(raw_path, dtype={"ID": "string"}, encoding="utf-8-sig")
    raw_ids = _ids(raw, raw_path)
    a_oof_path = project_path(features["stage2a_oof"])
    b_oof_path = project_path(features["stage2b_oof"])
    a_oof = pd.read_csv(a_oof_path, dtype={"ID": "string"}, encoding="utf-8-sig")
    b_oof = pd.read_csv(b_oof_path, dtype={"ID": "string"}, encoding="utf-8-sig")
    a_oof_ids, b_oof_ids = _ids(a_oof, a_oof_path), _ids(b_oof, b_oof_path)
    id_sets = [set(values) for values in (image_ids, label_ids, master_ids, raw_ids, a_oof_ids, b_oof_ids)]
    if any(values != id_sets[0] for values in id_sets[1:]):
        raise ValueError("Image, label, split, Stage 1, Stage 2A and Stage 2B ID sets differ")

    class_counts = master["label_3class"].astype(int).value_counts().reindex([0, 1, 2], fill_value=0).tolist()
    if class_counts != list(data["expected_class_counts"]):
        raise ValueError(f"Class counts differ from 115/237/148: {class_counts}")
    group_sizes = master.groupby("patient_group_id", sort=False).size()
    if group_sizes.size != int(data["expected_patient_groups"]):
        raise ValueError("patient_group_id count differs from 483")
    if int((group_sizes > 1).sum()) != int(data["expected_multi_image_patient_groups"]):
        raise ValueError("Multi-image patient-group count differs from 17")

    availability = pd.to_numeric(raw[AVAILABILITY_COLUMN], errors="coerce").to_numpy()
    if int((availability == 1).sum()) != 486 or int((availability == 0).sum()) != 14:
        raise ValueError("Stage 1 availability counts differ from 486/14")
    raw_values = raw.loc[:, list(RAW_FEATURE_COLUMNS)].to_numpy(float)
    if not np.isfinite(raw_values[:, :3]).all():
        raise ValueError("Stage 1 cheek features are not all finite")
    if not np.isfinite(raw_values[availability == 1, 3:]).all():
        raise ValueError("Available Stage 1 forehead features are not finite")
    if not np.isnan(raw_values[availability == 0, 3:]).all():
        raise ValueError("Unavailable Stage 1 forehead features are not all NaN")

    schema_rows: dict[str, Any] = {}
    for variant, schema_key in (
        ("global_raw", "raw_schema"), ("global_stage2a", "stage2a_schema"),
        ("global_stage2b", "stage2b_schema"),
    ):
        schema_path = project_path(features[schema_key])
        schema = validate_feature_schema(schema_path, variant)
        schema_rows[variant] = {
            "path": relative_path(schema_path, PROJECT_ROOT), "sha256": sha256_file(schema_path),
            "feature_columns": list({
                "global_raw": RAW_FEATURE_COLUMNS,
                "global_stage2a": STAGE2A_FEATURE_COLUMNS,
                "global_stage2b": STAGE2B_FEATURE_COLUMNS,
            }[variant]), "status": "PASS", "schema_name": schema.get("schema_name"),
        }
    allowed_model_fields = {
        "global_only": [],
        "global_mask": [AVAILABILITY_COLUMN],
        "global_raw": [*RAW_FEATURE_COLUMNS, AVAILABILITY_COLUMN],
        "global_stage2a": [*STAGE2A_FEATURE_COLUMNS, AVAILABILITY_COLUMN],
        "global_stage2b": [*STAGE2B_FEATURE_COLUMNS, AVAILABILITY_COLUMN],
    }
    forbidden_tokens = ("camera", "exif", "condition", "predicted", "residual", "qc", "nyha", "sex")
    for variant, columns in allowed_model_fields.items():
        bad = [column for column in columns if any(token in column.lower() for token in forbidden_tokens)]
        if bad:
            raise ValueError(f"Forbidden fields entered the {variant} positive allowlist: {bad}")
    schema_rows["classifier_positive_allowlists"] = {
        "status": "PASS", "columns_by_variant": allowed_model_fields,
        "camera_used": False, "exif_used": False, "condition_used": False,
        "predicted_or_residual_used": False, "clinical_or_label_used": False,
    }
    for manifest_key in ("raw_manifest", "stage2a_manifest", "stage2b_manifest"):
        manifest = _read_json(project_path(features[manifest_key]))
        if manifest.get("status") != "COMPLETE":
            raise ValueError(f"Upstream manifest is not COMPLETE: {features[manifest_key]}")

    input_rows = []
    for name, path, ids in (
        ("meanbg_images", image_root, image_ids), ("labels", label_path, label_ids),
        ("master_split", master_path, master_ids), ("stage1", raw_path, raw_ids),
        ("stage2a_oof_audit_only", a_oof_path, a_oof_ids),
        ("stage2b_oof_audit_only", b_oof_path, b_oof_ids),
    ):
        input_rows.append({
            "input": name, "path": relative_path(path, PROJECT_ROOT), "rows": len(ids),
            "unique_ids": len(set(ids)), "id_sha256": sha256_ids(ids),
            "file_sha256": sha256_file(path) if path.is_file() else None, "status": "PASS",
        })

    alignment_rows: list[dict[str, Any]] = []
    source_rows: list[dict[str, Any]] = [
        {
            "variant": "global_only", "fold": "all", "split_role": "none",
            "path": None, "sha256": None, "rows": 0, "unique_ids": 0,
            "oof_classifier_input": False, "status": "PASS_NO_OPTICAL_READ",
        },
        {
            "variant": "global_mask", "fold": "all", "split_role": "train_and_val_join",
            "path": relative_path(raw_path, PROJECT_ROOT), "sha256": sha256_file(raw_path),
            "rows": len(raw), "unique_ids": len(raw_ids), "oof_classifier_input": False,
            "selected_columns": f"ID,{AVAILABILITY_COLUMN}", "status": "PASS",
        },
        {
            "variant": "global_raw", "fold": "all", "split_role": "train_and_val_join",
            "path": relative_path(raw_path, PROJECT_ROOT), "sha256": sha256_file(raw_path),
            "rows": len(raw), "unique_ids": len(raw_ids), "oof_classifier_input": False,
            "selected_columns": ",".join(["ID", *RAW_FEATURE_COLUMNS, AVAILABILITY_COLUMN]),
            "status": "PASS",
        },
    ]
    all_val_ids: list[str] = []
    for fold in data["folds"]:
        split_root = project_path(data["split_root"])
        train_path = split_root / data["train_csv_pattern"].format(fold=fold)
        val_path = split_root / data["val_csv_pattern"].format(fold=fold)
        train = pd.read_csv(train_path, dtype={"ID": "string", "patient_group_id": "string"}, encoding="utf-8-sig")
        val = pd.read_csv(val_path, dtype={"ID": "string", "patient_group_id": "string"}, encoding="utf-8-sig")
        train_ids, val_ids = _ids(train, train_path), _ids(val, val_path)
        if len(train_ids) != 400 or len(val_ids) != 100 or set(train_ids) & set(val_ids):
            raise ValueError(f"Fold {fold} does not have disjoint 400/100 sample IDs")
        if set(train["patient_group_id"].astype(str)) & set(val["patient_group_id"].astype(str)):
            raise ValueError(f"Fold {fold} leaks patient_group_id")
        all_val_ids.extend(val_ids)
        fold_row: dict[str, Any] = {
            "fold": fold, "train_n": 400, "val_n": 100, "id_overlap": 0,
            "patient_group_overlap": 0, "train_id_sha256": sha256_ids(train_ids),
            "val_id_sha256": sha256_ids(val_ids), "status": "PASS",
        }
        for stage, root_key, train_name, val_name in (
            ("stage2a", "stage2a_root", "train_calibrated_features.csv", "val_calibrated_features.csv"),
            ("stage2b", "stage2b_root", "train_nn_calibrated_features.csv", "val_nn_calibrated_features.csv"),
        ):
            stage_root = project_path(features[root_key])
            for role, filename, expected_ids in (
                ("train", train_name, train_ids), ("val", val_name, val_ids),
            ):
                path = stage_root / f"fold_{fold}" / filename
                frame = pd.read_csv(path, dtype={"ID": "string"}, encoding="utf-8-sig")
                ids = _ids(frame, path)
                if set(ids) != set(expected_ids) or len(ids) != len(expected_ids):
                    raise ValueError(f"{stage} fold {fold} {role} IDs differ from classification split")
                if not (pd.to_numeric(frame["fold"]) == fold).all():
                    raise ValueError(f"{stage} fold field mismatch in {path}")
                if not (frame["split_role"].astype(str).str.lower() == role).all():
                    raise ValueError(f"{stage} split_role mismatch in {path}")
                maximum = _maximum_raw_difference(raw, frame, path)
                calibrated_columns = (
                    STAGE2A_FEATURE_COLUMNS if stage == "stage2a" else STAGE2B_FEATURE_COLUMNS
                )
                aligned_raw = raw.set_index("ID").loc[frame["ID"].astype(str)]
                stage_availability = pd.to_numeric(frame[AVAILABILITY_COLUMN], errors="coerce").to_numpy()
                raw_availability = pd.to_numeric(
                    aligned_raw[AVAILABILITY_COLUMN], errors="coerce"
                ).to_numpy()
                if not np.array_equal(stage_availability, raw_availability):
                    raise ValueError(f"{stage} availability differs from Stage 1 in {path}")
                calibrated_values = frame.loc[:, list(calibrated_columns)].to_numpy(float)
                if not np.isfinite(calibrated_values[:, :3]).all():
                    raise ValueError(f"{stage} cheek features are not all finite in {path}")
                if not np.isfinite(calibrated_values[stage_availability == 1, 3:]).all():
                    raise ValueError(f"{stage} available forehead features are not finite in {path}")
                if not np.isnan(calibrated_values[stage_availability == 0, 3:]).all():
                    raise ValueError(f"{stage} unavailable forehead features are not NaN in {path}")
                fold_row[f"{stage}_{role}_raw_max_abs_diff"] = maximum
                source_rows.append({
                    "variant": "global_stage2a" if stage == "stage2a" else "global_stage2b",
                    "fold": fold, "split_role": role,
                    "path": relative_path(path, PROJECT_ROOT), "sha256": sha256_file(path),
                    "rows": len(frame), "unique_ids": len(ids), "oof_classifier_input": False,
                    "status": "PASS",
                })
        alignment_rows.append(fold_row)
    if len(all_val_ids) != 500 or len(set(all_val_ids)) != 500 or set(all_val_ids) != set(master_ids):
        raise ValueError("Five validation folds do not cover all 500 IDs exactly once")

    environment = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "project_root": str(PROJECT_ROOT), "python": platform.python_version(),
        "pytorch": torch.__version__, "torchvision": torchvision.__version__,
        "cuda_available": torch.cuda.is_available(), "cuda_version": torch.version.cuda,
        "device_count": torch.cuda.device_count(), "platform": platform.platform(),
        **_collect_git_metadata(),
    }
    pd.DataFrame(input_rows).to_csv(protocol_dir / "input_audit.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(source_rows).to_csv(protocol_dir / "feature_source_audit.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(alignment_rows).to_csv(protocol_dir / "fold_alignment_audit.csv", index=False, encoding="utf-8-sig")
    _write_json(environment, protocol_dir / "environment_audit.json")
    _write_json(schema_rows, protocol_dir / "schema_audit.json")
    manifest = {
        "task": "global_resnet18_optical_fusion_preflight", "status": "PASS",
        "critical_checks": 27, "cohort_size": 500, "unique_ids": 500,
        "class_counts": class_counts, "patient_group_count": int(group_sizes.size),
        "multi_image_patient_group_count": int((group_sizes > 1).sum()),
        "availability_counts": {"available": 486, "unavailable": 14},
        "variants": list(VARIANTS), "auxiliary_input_dims": dict(VARIANT_AUX_DIM),
        "config_sha256": sha256_file(config_path), "split_sha256": sha256_file(master_path),
        "oof_used_as_classifier_train": False, "camera_used": False, "exif_used": False,
        "clinical_features_used": False, "full_training_executed": False,
        "historical_inputs_modified": False, "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    _write_json(manifest, protocol_dir / "protocol_manifest.json")
    return manifest


def _expand(values: Iterable[str], allowed: tuple[str, ...], label: str) -> list[Any]:
    values = list(values)
    if not values or "all" in values:
        return list(allowed)
    invalid = [value for value in values if value not in allowed]
    if invalid:
        raise ValueError(f"Invalid {label}: {invalid}")
    return list(dict.fromkeys(values))


def main() -> None:
    args = parse_args()
    config_path = project_path(args.config)
    config = load_yaml(config_path)
    output_root = project_path(config["experiment"]["output_root"])
    variants = _expand(args.variant, VARIANTS, "variant")
    fold_strings = _expand(args.fold, tuple(str(value) for value in config["data"]["folds"]), "fold")
    folds = [int(value) for value in fold_strings]

    if args.summarize_only:
        command = [
            sys.executable, str(PROJECT_ROOT / "scripts/evaluate/summarize_global_optical_fusion_5fold.py"),
            "--config", str(config_path), "--experiment-dir", str(output_root),
        ]
        subprocess.run(command, cwd=PROJECT_ROOT, check=True)
        return

    manifest = run_preflight(config_path, output_root)
    print(f"PROTOCOL_STATUS={manifest['status']}")
    if args.protocol_only:
        print(f"PROTOCOL_MANIFEST={output_root / 'protocol' / 'protocol_manifest.json'}")
        return

    formal_count = len(variants) * len(folds)
    if not args.smoke_test and formal_count == 25 and not torch.cuda.is_available() and not args.allow_cpu_training:
        raise RuntimeError(
            "CUDA is unavailable; refusing all variants x all folds formal training. "
            "Use a CUDA server or explicitly pass --allow-cpu-training."
        )
    if not args.smoke_test:
        run_required_tests(config_path, output_root)
        ensure_smoke_suite(config_path, output_root)
    for variant in variants:
        for fold in folds:
            run_dir = (
                output_root / "smoke" / variant / f"fold_{fold}"
                if args.smoke_test else output_root / variant / f"fold_{fold}"
            )
            manifest_path = run_dir / "fold_manifest.json"
            if args.skip_completed and manifest_path.is_file():
                status = _read_json(manifest_path).get("status")
                expected = "SMOKE_COMPLETE" if args.smoke_test else "COMPLETE"
                if status == expected:
                    print(f"SKIP_COMPLETED={run_dir}")
                    continue
            command = [
                sys.executable, str(PROJECT_ROOT / "scripts/train/train_global_optical_fusion_5fold.py"),
                "--config", str(config_path), "--variant", variant, "--fold", str(fold),
                "--output-root", str(output_root),
            ]
            for enabled, flag in (
                (args.resume, "--resume"), (args.overwrite, "--overwrite"),
                (args.smoke_test, "--smoke-test"),
            ):
                if enabled:
                    command.append(flag)
            subprocess.run(command, cwd=PROJECT_ROOT, check=True)
    if args.smoke_test:
        print("SMOKE_TEST_STATUS=PASS")
        print("FULL_TRAINING_EXECUTED=false")
        return
    if set(variants) == set(VARIANTS) and set(folds) == set(config["data"]["folds"]):
        subprocess.run(
            [sys.executable, str(PROJECT_ROOT / "scripts/evaluate/summarize_global_optical_fusion_5fold.py"),
             "--config", str(config_path), "--experiment-dir", str(output_root)],
            cwd=PROJECT_ROOT, check=True,
        )


if __name__ == "__main__":
    main()
