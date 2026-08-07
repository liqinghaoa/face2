"""Stage3-B0 Xiaomi-only ResNet18 control-vs-patient experiment.

This module builds the fixed Xiaomi-only cohort/splits, trains one ResNet18
per outer group fold, evaluates pooled OOF predictions, compares simple
lighting/EXIF baselines, and writes the audit/report artifacts required by
``prompt.md``.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import logging
import os
import platform
import random
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
import torch
import torchvision
from PIL import Image
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from datasets.nyha_3class_face_dataset import build_transforms
from models.nyha_backbone_factory import build_nyha_classification_model, count_parameters
from utils.experiment_utils import load_yaml, seed_worker, set_random_seed
from utils.resnet18_anti_overfit import (
    apply_train_mode,
    build_optimizer,
    build_trainability_audit,
    classifier_module,
    configure_trainability,
    normalize_strategy,
)

LOGGER = logging.getLogger("stage3_b0_xiaomi")

EXPERIMENT_NAME = "Stage3_B0_XiaomiOnly_ResNet18_ControlVsPatient_Group5Fold_v1"
OUTPUT_DIR = PROJECT_ROOT / "experiments" / "lighting_confounding" / EXPERIMENT_NAME
DATA_ASSET_DIR = PROJECT_ROOT / "data" / "processed" / "Stage3_B0_XiaomiOnly_v1"
STAGE1_MASTER = PROJECT_ROOT / "experiments" / "lighting_confounding" / "Lighting_Confounding_Audit_Stage1_v1" / "metadata" / "stage1_master_500.csv"
ORIGINAL_SPLIT = PROJECT_ROOT / "data" / "processed" / "P0_Physics_Audit_v1" / "splits_500" / "nyha_3class_sex_stratified_group_5fold.csv"
IMAGE_ROOT = PROJECT_ROOT / "data" / "processed" / "global_face" / "realface_256x320_blackbg_from_raw_v1" / "images"
ORIGINAL_E0B_DIR = PROJECT_ROOT / "experiments" / "500Data" / "E0B_Global_ResNet18_ControlVsPatient_Binary_realface_256x320_blackbg_from_raw_v1_anti_overfit_v3_bn_eval_full"
ORIGINAL_E0B_CONFIG = PROJECT_ROOT / "config" / "train" / "e0b_global_resnet18_control_patient_binary_realface_256x320_blackbg_from_raw_v1_anti_overfit_v3_bn_eval_full.yaml"
ORIGINAL_E0B_TRAIN_SCRIPT = PROJECT_ROOT / "scripts" / "train" / "train_e0b_global_resnet18_control_patient_binary_5fold.py"
ORIGINAL_E0B_OOF = ORIGINAL_E0B_DIR / "oof_predictions.csv"
STAGE2_FEATURES = PROJECT_ROOT / "experiments" / "lighting_confounding" / "Lighting_QC_Stage2_Full500_v1" / "features" / "stage2_lighting_features_500.csv"
STAGE2_R2_EXIF = PROJECT_ROOT / "experiments" / "lighting_confounding" / "Lighting_QC_Stage2_R2_Statistical_Repair_and_EXIF_Decomposition_v1" / "exif" / "exif_crossfit_predictions_500.csv"
STAGE3_A0_DIR = PROJECT_ROOT / "experiments" / "lighting_confounding" / "Stage3_A0_FrozenRGB_Paired_ExposureGamma_Stress_v1"
XIAOMI_MODEL = "M2006J10C"
EXPECTED_TOTAL = 233
EXPECTED_CONTROL = 115
EXPECTED_PATIENT = 118
N_FOLDS = 5
SPLIT_SEED = 2026
BOOTSTRAP_ITERATIONS = 2000


def ensure_dirs(root: Path) -> None:
    for name in (
        "cohort",
        "splits/inner",
        "preflight",
        "training_contract",
        "oof",
        "metrics",
        "bootstrap",
        "stability",
        "baselines",
        "comparison",
        "brightness_audit",
        "figures",
        "reports",
        "logs",
    ):
        (root / name).mkdir(parents=True, exist_ok=True)
    DATA_ASSET_DIR.mkdir(parents=True, exist_ok=True)


def write_json(data: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, default=_json_default), encoding="utf-8")


def _json_default(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.ndarray,)):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    return str(value)


def write_text(text: str, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def normalize_camera(value: object) -> str:
    if pd.isna(value):
        return ""
    return str(value).strip().upper()


def read_csv_str(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, dtype={"sample_id": "string", "patient_group_id": "string", "ID": "string"}, encoding="utf-8-sig")


def setup_logging(root: Path) -> None:
    root.mkdir(parents=True, exist_ok=True)
    handlers: list[logging.Handler] = [
        logging.StreamHandler(),
        logging.FileHandler(root / "logs" / "run.log", encoding="utf-8"),
    ]
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        handlers=handlers,
        force=True,
    )


def save_runtime_environment(root: Path) -> dict[str, Any]:
    env = {
        "started_at": datetime.now().isoformat(timespec="seconds"),
        "python_executable": sys.executable,
        "python_version": sys.version,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "torchvision": torchvision.__version__,
        "cuda_available": bool(torch.cuda.is_available()),
        "cuda_version": torch.version.cuda,
        "cudnn_version": torch.backends.cudnn.version(),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "numpy": np.__version__,
        "pillow": Image.__version__,
        "opencv": _optional_version("cv2"),
        "sklearn": sklearn.__version__,
        "pandas": pd.__version__,
        "git_commit": _git(["rev-parse", "HEAD"]),
        "working_directory": str(PROJECT_ROOT),
    }
    lines = [f"{key}: {value}" for key, value in env.items()]
    write_text("\n".join(lines) + "\n", root / "logs" / "runtime_environment.txt")
    write_text(_git(["status", "--short"]) + "\n", root / "logs" / "git_status.txt")
    write_json(env, root / "logs" / "runtime_environment.json")
    return env


def _optional_version(module_name: str) -> str | None:
    try:
        module = __import__(module_name)
    except Exception:
        return None
    return getattr(module, "__version__", None)


def _git(args: list[str]) -> str:
    try:
        return subprocess.check_output(["git", *args], cwd=PROJECT_ROOT, text=True, stderr=subprocess.STDOUT).strip()
    except Exception as exc:
        return f"unavailable: {exc}"


def extract_e0b_contract(root: Path) -> dict[str, Any]:
    config = load_yaml(ORIGINAL_E0B_CONFIG)
    contract = {
        "source_config": str(ORIGINAL_E0B_CONFIG),
        "source_training_script": str(ORIGINAL_E0B_TRAIN_SCRIPT),
        "source_experiment_dir": str(ORIGINAL_E0B_DIR),
        "model": config["model"],
        "data": config["data"],
        "train": config["train"],
        "augmentation": config["augmentation"],
        "normalize": config["normalize"],
        "task": config["task"],
        "b0_required_modifications": {
            "cohort": "restrict to exact normalized camera_model M2006J10C",
            "outer_split": "new Xiaomi-only StratifiedGroupKFold with random_state=2026",
            "inner_split": "first split from StratifiedGroupKFold within outer development, random_state=12026+outer_fold",
            "loss": "BCEWithLogitsLoss on patient-minus-control logit margin with pos_weight=1.0",
            "training_seed": "22026+outer_fold",
            "outer_test_use": "single inference after best checkpoint selected by inner validation only",
        },
    }
    write_json(contract, root / "training_contract" / "stage3_b0_training_contract.json")
    comparison_rows = []
    for section in ("data", "model", "train", "augmentation", "normalize", "task"):
        comparison_rows.append(
            {
                "section": section,
                "original_e0b": json.dumps(config.get(section), ensure_ascii=False, sort_keys=True),
                "stage3_b0": json.dumps(contract["b0_required_modifications"] if section == "train" else config.get(section), ensure_ascii=False, sort_keys=True),
                "status": "reused_except_loss_seed_split" if section == "train" else "reused",
            }
        )
    pd.DataFrame(comparison_rows).to_csv(root / "training_contract" / "original_e0b_contract_comparison.csv", index=False, encoding="utf-8-sig")
    return contract


def build_xiaomi_cohort(root: Path) -> pd.DataFrame:
    (root / "cohort").mkdir(parents=True, exist_ok=True)
    stage1 = read_csv_str(STAGE1_MASTER)
    stage1["camera_model_raw"] = stage1["camera_model"].astype("string")
    stage1["camera_model_normalized"] = stage1["camera_model_raw"].map(normalize_camera)
    inventory = (
        stage1.groupby(["camera_model_raw", "camera_model_normalized", "binary_label"], dropna=False)
        .size()
        .reset_index(name="n")
        .sort_values(["camera_model_normalized", "binary_label"])
    )
    inventory.to_csv(root / "cohort" / "camera_label_inventory.csv", index=False, encoding="utf-8-sig")

    xiaomi = stage1.loc[stage1["camera_model_normalized"].eq(XIAOMI_MODEL)].copy()
    xiaomi["sample_id"] = xiaomi["sample_id"].astype(str)
    xiaomi["patient_group_id"] = xiaomi["patient_group_id"].astype(str)
    xiaomi["binary_label"] = pd.to_numeric(xiaomi["binary_label"], errors="raise").astype(int)
    xiaomi["original_nyha_label"] = pd.to_numeric(xiaomi["original_nyha"], errors="raise").astype(int)
    xiaomi["sex"] = xiaomi["split__sex_name"].astype("string").fillna("")
    xiaomi["sex_code"] = pd.to_numeric(xiaomi["split__SEX"], errors="coerce")
    xiaomi["image_path"] = xiaomi["sample_id"].map(lambda value: str(IMAGE_ROOT / f"{value}.png"))
    xiaomi["image_exists"] = xiaomi["image_path"].map(lambda value: Path(value).is_file())
    xiaomi["image_sha256"] = xiaomi["image_path"].map(lambda value: sha256_file(Path(value)) if Path(value).is_file() else "")

    cohort = xiaomi.loc[
        :,
        [
            "sample_id",
            "patient_group_id",
            "binary_label",
            "original_nyha_label",
            "sex",
            "sex_code",
            "camera_model_raw",
            "camera_model_normalized",
            "image_path",
            "image_sha256",
            "fold",
            "brightness_value_apex",
            "log2_exposure_time",
            "log2_iso",
        ],
    ].sort_values("sample_id", kind="stable")

    cohort.to_csv(root / "cohort" / "xiaomi_cohort_233.csv", index=False, encoding="utf-8-sig")
    cohort.to_csv(DATA_ASSET_DIR / "xiaomi_cohort_233.csv", index=False, encoding="utf-8-sig")
    group_inventory = (
        cohort.groupby("patient_group_id")
        .agg(n=("sample_id", "size"), labels=("binary_label", lambda s: ",".join(map(str, sorted(set(s))))), sexes=("sex", lambda s: ",".join(map(str, sorted(set(s))))))
        .reset_index()
    )
    group_inventory.to_csv(root / "cohort" / "patient_group_inventory.csv", index=False, encoding="utf-8-sig")
    hash_inventory = cohort.groupby("image_sha256", dropna=False).agg(n=("sample_id", "size"), sample_ids=("sample_id", lambda s: ",".join(map(str, s)))).reset_index()
    hash_inventory.to_csv(root / "cohort" / "image_hash_inventory.csv", index=False, encoding="utf-8-sig")

    audit = cohort_audit(cohort)
    write_json(audit, root / "cohort" / "xiaomi_cohort_audit.json")
    write_text(cohort_report(audit), root / "cohort" / "xiaomi_cohort_report.md")
    if audit["core_status"] != "passed":
        write_json({"status": "failed", "errors": audit["errors"]}, root / "preflight" / "stage3_b0_preflight_summary.json")
        raise RuntimeError(f"Xiaomi cohort gate failed: {audit['errors']}")
    return cohort


def cohort_audit(cohort: pd.DataFrame) -> dict[str, Any]:
    errors: list[str] = []
    warnings: list[str] = []
    counts = cohort["binary_label"].value_counts().to_dict()
    group_label_conflicts = []
    group_sex_conflicts = []
    for group_id, group in cohort.groupby("patient_group_id"):
        if group["binary_label"].nunique() > 1:
            group_label_conflicts.append(str(group_id))
        if group["sex"].nunique(dropna=False) > 1:
            group_sex_conflicts.append(str(group_id))
    duplicate_hashes = cohort.loc[cohort["image_sha256"].duplicated(keep=False) & cohort["image_sha256"].ne(""), "image_sha256"].unique().tolist()
    checks = {
        "total_233": len(cohort) == EXPECTED_TOTAL,
        "control_115": int(counts.get(0, 0)) == EXPECTED_CONTROL,
        "patient_118": int(counts.get(1, 0)) == EXPECTED_PATIENT,
        "sample_id_unique": not cohort["sample_id"].duplicated().any(),
        "images_233_exist": bool(cohort["image_path"].map(lambda p: Path(p).is_file()).all()),
        "binary_label_complete": not cohort["binary_label"].isna().any(),
        "patient_group_complete": not cohort["patient_group_id"].isna().any() and (cohort["patient_group_id"].astype(str).str.len() > 0).all(),
        "camera_model_unique_xiaomi": cohort["camera_model_normalized"].nunique() == 1 and cohort["camera_model_normalized"].iloc[0] == XIAOMI_MODEL,
        "no_honor": not cohort["camera_model_normalized"].astype(str).str.contains("BVL-AN00|HONOR", case=False, regex=True).any(),
        "no_group_label_conflict": len(group_label_conflicts) == 0,
        "no_group_sex_conflict": len(group_sex_conflicts) == 0,
        "no_duplicate_image_path": not cohort["image_path"].duplicated().any(),
        "no_duplicate_image_hash": len(duplicate_hashes) == 0,
    }
    if cohort["sex"].isna().any() or (cohort["sex"].astype(str).str.len() == 0).any():
        warnings.append("sex field contains missing/empty values")
    for key, passed in checks.items():
        if not passed:
            errors.append(key)
    return {
        "core_status": "passed" if not errors else "failed",
        "checks": checks,
        "errors": errors,
        "warnings": warnings,
        "n_samples": int(len(cohort)),
        "n_control": int(counts.get(0, 0)),
        "n_patient": int(counts.get(1, 0)),
        "n_patient_groups": int(cohort["patient_group_id"].nunique()),
        "camera_models": sorted(cohort["camera_model_normalized"].unique().tolist()),
        "group_label_conflicts": group_label_conflicts,
        "group_sex_conflicts": group_sex_conflicts,
        "duplicate_image_hashes": duplicate_hashes,
    }


def cohort_report(audit: dict[str, Any]) -> str:
    checks = "\n".join(f"- {key}: {'PASS' if value else 'FAIL'}" for key, value in audit["checks"].items())
    return f"""# Stage3-B0 Xiaomi Cohort Audit

Status: {audit['core_status']}

- Samples: {audit['n_samples']}
- Control: {audit['n_control']}
- Patient: {audit['n_patient']}
- Patient groups: {audit['n_patient_groups']}
- Camera models: {', '.join(audit['camera_models'])}

## Checks

{checks}

Warnings: {audit['warnings']}
Errors: {audit['errors']}
"""


def _stratification_for_groups(groups: pd.DataFrame) -> tuple[np.ndarray, str]:
    composite = groups["binary_label"].astype(str) + "_" + groups["sex"].astype(str)
    if composite.value_counts().min() >= N_FOLDS:
        return composite.to_numpy(), "label_x_sex"
    return groups["binary_label"].to_numpy(), "label_only_degraded_from_label_x_sex"


def build_outer_splits(root: Path, cohort: pd.DataFrame) -> pd.DataFrame:
    group_table = (
        cohort.groupby("patient_group_id")
        .agg(
            binary_label=("binary_label", "first"),
            sex=("sex", "first"),
            n=("sample_id", "size"),
        )
        .reset_index()
        .sort_values("patient_group_id", kind="stable")
    )
    strata, strategy = _stratification_for_groups(group_table)
    splitter = StratifiedGroupKFold(n_splits=N_FOLDS, shuffle=True, random_state=SPLIT_SEED)
    fold_by_group: dict[str, int] = {}
    dummy_x = np.zeros(len(group_table))
    groups = group_table["patient_group_id"].to_numpy()
    for fold, (_, test_idx) in enumerate(splitter.split(dummy_x, strata, groups=groups)):
        for group_id in group_table.iloc[test_idx]["patient_group_id"].astype(str):
            fold_by_group[group_id] = fold
    split = cohort.copy()
    split["outer_fold"] = split["patient_group_id"].map(fold_by_group).astype(int)
    split["camera_model"] = split["camera_model_normalized"]
    split = split.loc[:, ["sample_id", "patient_group_id", "binary_label", "sex", "outer_fold", "camera_model", "image_path"]].sort_values(["outer_fold", "sample_id"], kind="stable")
    split.to_csv(root / "splits" / "xiaomi_control_patient_group5fold_v1.csv", index=False, encoding="utf-8-sig")
    split.to_csv(DATA_ASSET_DIR / "xiaomi_control_patient_group5fold_v1.csv", index=False, encoding="utf-8-sig")
    digest = sha256_file(root / "splits" / "xiaomi_control_patient_group5fold_v1.csv")
    write_text(digest + "\n", root / "splits" / "xiaomi_control_patient_group5fold_v1.sha256")

    audit_rows = []
    for fold in range(N_FOLDS):
        test = split[split["outer_fold"].eq(fold)]
        dev = split[~split["outer_fold"].eq(fold)]
        audit_rows.append(_split_count_row(test, fold, "outer_test") | {"dev_patient_groups": int(dev["patient_group_id"].nunique()), "group_overlap": int(len(set(test["patient_group_id"]) & set(dev["patient_group_id"])))})
    audit = pd.DataFrame(audit_rows)
    audit.to_csv(root / "splits" / "xiaomi_group5fold_audit.csv", index=False, encoding="utf-8-sig")
    distribution = {
        "strategy": strategy,
        "seed": SPLIT_SEED,
        "folds": audit.to_dict(orient="records"),
        "coverage_rows": int(len(split)),
        "unique_samples": int(split["sample_id"].nunique()),
        "unique_groups": int(split["patient_group_id"].nunique()),
    }
    write_json(distribution, root / "splits" / "xiaomi_group5fold_distribution.json")
    write_text(split_report("Outer Xiaomi-only group 5-fold split", strategy, audit), root / "splits" / "xiaomi_group5fold_report.md")
    outer_errors = validate_outer_split(split)
    if outer_errors:
        raise RuntimeError(f"Outer split gate failed: {outer_errors}")
    return split


def _split_count_row(frame: pd.DataFrame, fold: int, split_name: str) -> dict[str, Any]:
    return {
        "fold": int(fold),
        "split": split_name,
        "n": int(len(frame)),
        "control": int((frame["binary_label"].astype(int) == 0).sum()),
        "patient": int((frame["binary_label"].astype(int) == 1).sum()),
        "patient_groups": int(frame["patient_group_id"].nunique()),
        "female": int((frame["sex"].astype(str).str.lower() == "female").sum()),
        "male": int((frame["sex"].astype(str).str.lower() == "male").sum()),
    }


def validate_outer_split(split: pd.DataFrame) -> list[str]:
    errors: list[str] = []
    if len(split) != EXPECTED_TOTAL or split["sample_id"].nunique() != EXPECTED_TOTAL:
        errors.append("outer split does not cover exactly 233 unique samples")
    if set(split["outer_fold"].unique()) != set(range(N_FOLDS)):
        errors.append("outer folds are incomplete")
    for fold in range(N_FOLDS):
        test = split[split["outer_fold"].eq(fold)]
        dev = split[~split["outer_fold"].eq(fold)]
        labels = set(test["binary_label"].astype(int))
        if labels != {0, 1}:
            errors.append(f"fold {fold} lacks both classes")
        if (test["binary_label"].astype(int) == 0).sum() < 15:
            errors.append(f"fold {fold} has <15 control samples")
        if (test["binary_label"].astype(int) == 1).sum() < 15:
            errors.append(f"fold {fold} has <15 patient samples")
        if set(test["patient_group_id"]) & set(dev["patient_group_id"]):
            errors.append(f"fold {fold} has patient-group leakage")
    return errors


def split_report(title: str, strategy: str, audit: pd.DataFrame) -> str:
    rows = "\n".join(
        f"| {int(r.fold)} | {r.split} | {int(r.n)} | {int(r.control)} | {int(r.patient)} | {int(r.patient_groups)} | {int(r.female)} | {int(r.male)} |"
        for r in audit.itertuples()
    )
    return f"""# {title}

Strategy: `{strategy}`

| Fold | Split | N | Control | Patient | Patient groups | Female | Male |
|---:|---|---:|---:|---:|---:|---:|---:|
{rows}
"""


def build_inner_splits(root: Path, outer_split: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    audit_rows: list[dict[str, Any]] = []
    for outer_fold in range(N_FOLDS):
        dev = outer_split[~outer_split["outer_fold"].eq(outer_fold)].copy()
        group_table = (
            dev.groupby("patient_group_id")
            .agg(binary_label=("binary_label", "first"), sex=("sex", "first"), n=("sample_id", "size"))
            .reset_index()
            .sort_values("patient_group_id", kind="stable")
        )
        strata, strategy = _stratification_for_groups(group_table)
        splitter = StratifiedGroupKFold(n_splits=N_FOLDS, shuffle=True, random_state=12026 + outer_fold)
        _, val_idx = next(splitter.split(np.zeros(len(group_table)), strata, groups=group_table["patient_group_id"].to_numpy()))
        val_groups = set(group_table.iloc[val_idx]["patient_group_id"].astype(str))
        fold_rows = dev.copy()
        fold_rows["inner_split"] = np.where(fold_rows["patient_group_id"].isin(val_groups), "inner_val", "inner_train")
        fold_rows["outer_fold"] = outer_fold
        fold_rows.loc[:, ["sample_id", "patient_group_id", "binary_label", "sex", "outer_fold", "inner_split", "camera_model", "image_path"]].to_csv(
            root / "splits" / "inner" / f"fold_{outer_fold}_inner_split.csv",
            index=False,
            encoding="utf-8-sig",
        )
        for split_name in ("inner_train", "inner_val"):
            frame = fold_rows[fold_rows["inner_split"].eq(split_name)]
            audit_rows.append(_split_count_row(frame, outer_fold, split_name) | {"strategy": strategy})
        for row in fold_rows.to_dict(orient="records"):
            rows.append(row)
    audit = pd.DataFrame(audit_rows)
    audit.to_csv(root / "splits" / "inner" / "inner_split_audit.csv", index=False, encoding="utf-8-sig")
    write_text(split_report("Inner development split audit", "per outer fold; label_x_sex or degraded label-only", audit), root / "splits" / "inner" / "inner_split_report.md")
    errors = validate_inner_splits(outer_split, pd.DataFrame(rows))
    if errors:
        raise RuntimeError(f"Inner split gate failed: {errors}")
    return pd.DataFrame(rows)


def validate_inner_splits(outer_split: pd.DataFrame, inner_all: pd.DataFrame) -> list[str]:
    errors: list[str] = []
    for fold in range(N_FOLDS):
        inner = inner_all[inner_all["outer_fold"].eq(fold)]
        outer_test = outer_split[outer_split["outer_fold"].eq(fold)]
        if set(inner["sample_id"]) & set(outer_test["sample_id"]):
            errors.append(f"fold {fold} inner contains outer test samples")
        train = inner[inner["inner_split"].eq("inner_train")]
        val = inner[inner["inner_split"].eq("inner_val")]
        if set(train["patient_group_id"]) & set(val["patient_group_id"]):
            errors.append(f"fold {fold} inner train/val group leakage")
        for name, frame in (("inner_train", train), ("inner_val", val)):
            if set(frame["binary_label"].astype(int)) != {0, 1}:
                errors.append(f"fold {fold} {name} lacks both classes")
        val_counts = val["binary_label"].astype(int).value_counts()
        if int(val_counts.get(0, 0)) < 10 or int(val_counts.get(1, 0)) < 10:
            errors.append(f"fold {fold} inner validation has <10 samples in a class")
    return errors


def write_preflight(root: Path, env: dict[str, Any], contract: dict[str, Any], cohort: pd.DataFrame, outer_split: pd.DataFrame, inner_all: pd.DataFrame) -> dict[str, Any]:
    input_inventory = pd.DataFrame(
        [
            {"name": "stage1_master_500", "path": STAGE1_MASTER, "exists": STAGE1_MASTER.is_file(), "sha256": sha256_file(STAGE1_MASTER) if STAGE1_MASTER.is_file() else ""},
            {"name": "original_e0b_config", "path": ORIGINAL_E0B_CONFIG, "exists": ORIGINAL_E0B_CONFIG.is_file(), "sha256": sha256_file(ORIGINAL_E0B_CONFIG) if ORIGINAL_E0B_CONFIG.is_file() else ""},
            {"name": "original_e0b_train_script", "path": ORIGINAL_E0B_TRAIN_SCRIPT, "exists": ORIGINAL_E0B_TRAIN_SCRIPT.is_file(), "sha256": sha256_file(ORIGINAL_E0B_TRAIN_SCRIPT) if ORIGINAL_E0B_TRAIN_SCRIPT.is_file() else ""},
            {"name": "original_e0b_oof", "path": ORIGINAL_E0B_OOF, "exists": ORIGINAL_E0B_OOF.is_file(), "sha256": sha256_file(ORIGINAL_E0B_OOF) if ORIGINAL_E0B_OOF.is_file() else ""},
            {"name": "stage2_features", "path": STAGE2_FEATURES, "exists": STAGE2_FEATURES.is_file(), "sha256": sha256_file(STAGE2_FEATURES) if STAGE2_FEATURES.is_file() else ""},
            {"name": "stage2_r2_exif", "path": STAGE2_R2_EXIF, "exists": STAGE2_R2_EXIF.is_file(), "sha256": sha256_file(STAGE2_R2_EXIF) if STAGE2_R2_EXIF.is_file() else ""},
            {"name": "stage3_a0_reference_dir", "path": STAGE3_A0_DIR, "exists": STAGE3_A0_DIR.exists(), "sha256": ""},
            {"name": "image_root", "path": IMAGE_ROOT, "exists": IMAGE_ROOT.is_dir(), "sha256": ""},
        ]
    )
    input_inventory.to_csv(root / "preflight" / "input_inventory.csv", index=False, encoding="utf-8-sig")
    checks = {
        "face2_python_environment": Path(sys.executable).as_posix().lower().endswith("/envs/face2/python.exe") or str(sys.executable).lower().endswith("\\envs\\face2\\python.exe"),
        "cuda_available": bool(torch.cuda.is_available()),
        "xiaomi_total_233": len(cohort) == EXPECTED_TOTAL,
        "control_115": int((cohort["binary_label"] == 0).sum()) == EXPECTED_CONTROL,
        "patient_118": int((cohort["binary_label"] == 1).sum()) == EXPECTED_PATIENT,
        "camera_model_consistent": cohort["camera_model_normalized"].nunique() == 1 and cohort["camera_model_normalized"].iloc[0] == XIAOMI_MODEL,
        "images_exist": bool(cohort["image_path"].map(lambda p: Path(p).is_file()).all()),
        "patient_group_complete": not cohort["patient_group_id"].isna().any(),
        "no_group_label_conflict": not cohort.groupby("patient_group_id")["binary_label"].nunique().gt(1).any(),
        "outer_no_group_leakage": not validate_outer_split(outer_split),
        "inner_valid": not validate_inner_splits(outer_split, inner_all),
        "e0b_contract_parsed": bool(contract.get("model")),
        "output_not_original_e0b": root.resolve() != ORIGINAL_E0B_DIR.resolve(),
    }
    errors = [key for key, value in checks.items() if not value]
    summary = {
        "status": "passed" if not errors else "failed",
        "checks": checks,
        "errors": errors,
        "environment": env,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
    }
    write_json(summary, root / "preflight" / "stage3_b0_preflight_summary.json")
    check_lines = "\n".join(f"- {key}: {'PASS' if value else 'FAIL'}" for key, value in checks.items())
    write_text(f"# Stage3-B0 Preflight\n\nStatus: {summary['status']}\n\n{check_lines}\n\nErrors: {errors}\n", root / "preflight" / "stage3_b0_preflight_report.md")
    if errors:
        raise RuntimeError(f"Preflight failed: {errors}")
    return summary


class XiaomiB0Dataset(Dataset):
    def __init__(self, frame: pd.DataFrame, transform: Any) -> None:
        self.frame = frame.reset_index(drop=True).copy()
        self.transform = transform
        self.labels = self.frame["binary_label"].astype(int).tolist()

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.frame.iloc[index]
        image_path = Path(str(row["image_path"]))
        with Image.open(image_path) as image:
            tensor = self.transform(image.convert("RGB"))
        return {
            "image": tensor,
            "label": torch.tensor(int(row["binary_label"]), dtype=torch.float32),
            "sample_id": str(row["sample_id"]),
            "patient_group_id": str(row["patient_group_id"]),
            "outer_fold": int(row["outer_fold"]),
            "sex": str(row["sex"]),
            "camera_model": str(row["camera_model"]),
            "image_path": str(image_path),
        }


def make_loader(dataset: Dataset, batch_size: int, shuffle: bool, workers: int, seed: int, pin_memory: bool) -> DataLoader:
    generator = torch.Generator()
    generator.manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=workers,
        pin_memory=pin_memory,
        persistent_workers=workers > 0,
        worker_init_fn=seed_worker,
        generator=generator,
    )


@torch.no_grad()
def predict_frame(model: torch.nn.Module, data_loader: DataLoader, device: torch.device, selected_epoch: int, checkpoint_path: Path, checkpoint_sha256: str) -> pd.DataFrame:
    model.eval()
    rows: list[dict[str, Any]] = []
    for batch in data_loader:
        logits2 = model(batch["image"].to(device, non_blocking=True))
        margin = logits2[:, 1] - logits2[:, 0]
        prob_patient = torch.sigmoid(margin).detach().cpu().numpy()
        logits_np = logits2.detach().cpu().numpy()
        labels = batch["label"].detach().cpu().numpy().astype(int)
        for i, label in enumerate(labels):
            probability = float(prob_patient[i])
            pred = int(probability >= 0.5)
            rows.append(
                {
                    "sample_id": batch["sample_id"][i],
                    "patient_group_id": batch["patient_group_id"][i],
                    "outer_fold": int(batch["outer_fold"][i]),
                    "binary_label": int(label),
                    "sex": batch["sex"][i],
                    "camera_model": batch["camera_model"][i],
                    "logit_control": float(logits_np[i, 0]),
                    "logit_patient_raw": float(logits_np[i, 1]),
                    "logit_patient": float(margin.detach().cpu().numpy()[i]),
                    "probability_patient": probability,
                    "prediction_threshold_05": pred,
                    "correct": int(pred == int(label)),
                    "image_path": batch["image_path"][i],
                    "checkpoint_path": str(checkpoint_path),
                    "checkpoint_sha256": checkpoint_sha256,
                    "selected_epoch": int(selected_epoch),
                }
            )
    return pd.DataFrame(rows)


def binary_metrics(y_true: Iterable[Any], prob_patient: Iterable[Any]) -> dict[str, Any]:
    y = np.asarray(list(y_true), dtype=int)
    p = np.asarray(list(prob_patient), dtype=float)
    pred = (p >= 0.5).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    auc = float(roc_auc_score(y, p)) if len(np.unique(y)) == 2 else float("nan")
    sensitivity = float(tp / (tp + fn)) if (tp + fn) else float("nan")
    specificity = float(tn / (tn + fp)) if (tn + fp) else float("nan")
    return {
        "auc": auc,
        "accuracy": float(accuracy_score(y, pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
        "macro_precision": float(precision_score(y, pred, average="macro", zero_division=0)),
        "macro_recall": float(recall_score(y, pred, average="macro", zero_division=0)),
        "macro_f1": float(f1_score(y, pred, average="macro", zero_division=0)),
        "sensitivity": sensitivity,
        "specificity": specificity,
        "brier": float(brier_score_loss(y, p)),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
        "n": int(len(y)),
        "control": int((y == 0).sum()),
        "patient": int((y == 1).sum()),
    }


def calibration_metrics(y_true: Iterable[Any], prob_patient: Iterable[Any]) -> dict[str, float]:
    y = np.asarray(list(y_true), dtype=int)
    p = np.clip(np.asarray(list(prob_patient), dtype=float), 1e-6, 1 - 1e-6)
    logit = np.log(p / (1 - p)).reshape(-1, 1)
    try:
        model = LogisticRegression(C=1e6, solver="lbfgs", max_iter=1000)
        model.fit(logit, y)
        return {"calibration_intercept": float(model.intercept_[0]), "calibration_slope": float(model.coef_[0, 0])}
    except Exception:
        return {"calibration_intercept": float("nan"), "calibration_slope": float("nan")}


def train_one_fold(root: Path, contract: dict[str, Any], outer_split: pd.DataFrame, fold: int, device: torch.device, resume: bool = True) -> tuple[pd.DataFrame, dict[str, Any]]:
    cfg = contract
    e0b = load_yaml(ORIGINAL_E0B_CONFIG)
    train_cfg = e0b["train"]
    data_cfg = e0b["data"]
    model_cfg = e0b["model"]
    fold_dir = root / f"fold_{fold}"
    checkpoint_dir = fold_dir / "checkpoints"
    history_dir = fold_dir / "history"
    figure_dir = fold_dir / "figures"
    for path in (checkpoint_dir, history_dir, figure_dir):
        path.mkdir(parents=True, exist_ok=True)
    output_predictions = fold_dir / "outer_test_predictions.csv"
    output_metrics = fold_dir / "outer_test_metrics.csv"
    if resume and output_predictions.is_file() and output_metrics.is_file() and (checkpoint_dir / "best_macro_auc.pth").is_file():
        LOGGER.info("fold=%d already complete; loading existing predictions", fold)
        return pd.read_csv(output_predictions, dtype={"sample_id": "string", "patient_group_id": "string"}), pd.read_csv(output_metrics).iloc[0].to_dict()

    set_random_seed(22026 + fold)
    train_tf = build_transforms("train", data_cfg["image_size"], e0b["normalize"]["mean"], e0b["normalize"]["std"], bool(e0b["augmentation"]["horizontal_flip"]))
    val_tf = build_transforms("val", data_cfg["image_size"], e0b["normalize"]["mean"], e0b["normalize"]["std"], False)
    inner = pd.read_csv(root / "splits" / "inner" / f"fold_{fold}_inner_split.csv", dtype={"sample_id": "string", "patient_group_id": "string"}, encoding="utf-8-sig")
    inner_train = inner[inner["inner_split"].eq("inner_train")].copy()
    inner_val = inner[inner["inner_split"].eq("inner_val")].copy()
    outer_test = outer_split[outer_split["outer_fold"].eq(fold)].copy()
    train_loader = make_loader(XiaomiB0Dataset(inner_train, train_tf), int(train_cfg["batch_size"]), True, int(train_cfg["num_workers"]), 22026 + fold, bool(train_cfg["pin_memory"]))
    val_loader = make_loader(XiaomiB0Dataset(inner_val, val_tf), int(train_cfg["batch_size"]), False, int(train_cfg["num_workers"]), 23026 + fold, bool(train_cfg["pin_memory"]))
    test_loader = make_loader(XiaomiB0Dataset(outer_test, val_tf), int(train_cfg["batch_size"]), False, int(train_cfg["num_workers"]), 24026 + fold, bool(train_cfg["pin_memory"]))

    strategy = normalize_strategy(model_cfg.get("trainability_strategy", "full_finetune"))
    model = build_nyha_classification_model("resnet18", num_classes=2, pretrained=model_cfg["pretrained"], freeze_backbone=False, dropout=model_cfg.get("dropout")).to(device)
    configure_trainability(model, strategy)
    criterion = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor([1.0], device=device))
    optimizer = build_optimizer(
        model,
        strategy,
        full_lr=float(train_cfg["lr"]),
        classifier_lr=float(train_cfg.get("classifier_lr", train_cfg["lr"])),
        layer4_lr=float(train_cfg.get("layer4_lr", float(train_cfg.get("classifier_lr", train_cfg["lr"])) * 0.1)),
        weight_decay=float(train_cfg["weight_decay"]),
    )
    gradient_clip_max_norm = train_cfg.get("gradient_clip_max_norm", None)
    trainability_audit = build_trainability_audit(
        model,
        optimizer,
        strategy,
        fold=fold,
        dropout=model_cfg.get("dropout"),
        gradient_clip_max_norm=None if gradient_clip_max_norm is None else float(gradient_clip_max_norm),
    )
    fold_contract = {
        "fold": fold,
        "training_seed": 22026 + fold,
        "python_seed": 22026 + fold,
        "numpy_seed": 22026 + fold,
        "torch_seed": 22026 + fold,
        "cuda_seed": 22026 + fold if torch.cuda.is_available() else None,
        "dataloader_worker_seed": "torch.initial_seed() % 2**32 via seed_worker",
        "loss": "BCEWithLogitsLoss",
        "pos_weight": 1.0,
        "weighted_random_sampler": False,
        "outer_test_used_for_checkpoint_selection": False,
        "outer_test_inference_count": 1,
        "trainability_audit": trainability_audit,
        "parameter_counts": count_parameters(model),
    }
    write_json(fold_contract, fold_dir / "training_parameters.json")
    write_json(trainability_audit, fold_dir / "trainability_audit.json")

    best_auc = float("-inf")
    best_loss = float("inf")
    best_epoch = 0
    patience = 0
    history: list[dict[str, Any]] = []
    started = time.perf_counter()
    for epoch in range(1, int(train_cfg["epochs"]) + 1):
        apply_train_mode(model, strategy)
        train_loss, train_seen = 0.0, 0
        train_probs: list[float] = []
        train_labels: list[int] = []
        for batch in train_loader:
            image = batch["image"].to(device, non_blocking=True)
            label = batch["label"].to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            logits2 = model(image)
            margin = logits2[:, 1] - logits2[:, 0]
            loss = criterion(margin, label)
            loss.backward()
            if gradient_clip_max_norm is not None:
                torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], float(gradient_clip_max_norm))
            optimizer.step()
            train_loss += float(loss.detach()) * len(label)
            train_seen += len(label)
            train_probs.extend(torch.sigmoid(margin).detach().cpu().numpy().tolist())
            train_labels.extend(label.detach().cpu().numpy().astype(int).tolist())
        train_metrics = binary_metrics(train_labels, train_probs)
        val_eval, val_loss = evaluate_for_selection(model, val_loader, device, criterion)
        inner_val_auc = float(val_eval["auc"])
        inner_val_loss = float(val_loss)
        row = {
            "epoch": epoch,
            "train_loss": float(train_loss / train_seen),
            "inner_val_loss": inner_val_loss,
            "train_auc": float(train_metrics["auc"]),
            "inner_val_auc": inner_val_auc,
            "learning_rate": float(optimizer.param_groups[0]["lr"]),
        }
        history.append(row)
        improved = (inner_val_auc > best_auc) or (np.isclose(inner_val_auc, best_auc) and (inner_val_loss < best_loss or (np.isclose(inner_val_loss, best_loss) and epoch < best_epoch)))
        if improved:
            best_auc = inner_val_auc
            best_loss = inner_val_loss
            best_epoch = epoch
            patience = 0
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "epoch": epoch,
                    "best_macro_auc": inner_val_auc,
                    "best_inner_val_loss": inner_val_loss,
                    "config": e0b,
                    "stage3_b0_contract": cfg,
                    "fold_contract": fold_contract,
                },
                checkpoint_dir / "best_macro_auc.pth",
            )
        else:
            patience += 1
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "epoch": epoch,
                "best_macro_auc": best_auc,
                "best_inner_val_loss": best_loss,
                "config": e0b,
                "stage3_b0_contract": cfg,
                "fold_contract": fold_contract,
            },
            checkpoint_dir / "last.pth",
        )
        LOGGER.info("fold=%d epoch=%d train_loss=%.5f train_auc=%.4f inner_val_loss=%.5f inner_val_auc=%.4f best=%.4f patience=%d/%d", fold, epoch, row["train_loss"], row["train_auc"], inner_val_loss, inner_val_auc, best_auc, patience, int(train_cfg["early_stopping_patience"]))
        if patience >= int(train_cfg["early_stopping_patience"]):
            break

    history_frame = pd.DataFrame(history)
    history_frame.to_csv(history_dir / "training_history.csv", index=False, encoding="utf-8-sig")
    history_frame.to_csv(fold_dir / "training_history.csv", index=False, encoding="utf-8-sig")
    plot_fold_history(history_frame, figure_dir, fold)
    checkpoint_path = checkpoint_dir / "best_macro_auc.pth"
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    checkpoint_digest = sha256_file(checkpoint_path)
    test_pred = predict_frame(model, test_loader, device, int(checkpoint["epoch"]), checkpoint_path, checkpoint_digest)
    test_pred.to_csv(output_predictions, index=False, encoding="utf-8-sig")
    test_metrics = binary_metrics(test_pred["binary_label"], test_pred["probability_patient"])
    selected_history = history_frame.loc[history_frame["epoch"].eq(int(checkpoint["epoch"]))].iloc[0].to_dict()
    metric_row = {
        "fold": fold,
        "test_n": int(len(test_pred)),
        "test_control": int((test_pred["binary_label"] == 0).sum()),
        "test_patient": int((test_pred["binary_label"] == 1).sum()),
        "test_patient_groups": int(test_pred["patient_group_id"].nunique()),
        "selected_epoch": int(checkpoint["epoch"]),
        "train_auc_at_selected_epoch": float(selected_history["train_auc"]),
        "inner_val_auc_at_selected_epoch": float(selected_history["inner_val_auc"]),
        "train_loss_at_selected_epoch": float(selected_history["train_loss"]),
        "inner_val_loss_at_selected_epoch": float(selected_history["inner_val_loss"]),
        "training_seconds": float(time.perf_counter() - started),
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_sha256": checkpoint_digest,
        **{f"test_{k}": v for k, v in test_metrics.items() if k not in {"tn", "fp", "fn", "tp"}},
        "tn": test_metrics["tn"],
        "fp": test_metrics["fp"],
        "fn": test_metrics["fn"],
        "tp": test_metrics["tp"],
    }
    pd.DataFrame([metric_row]).to_csv(output_metrics, index=False, encoding="utf-8-sig")
    inventory = pd.DataFrame(
        [
            {"checkpoint": "best_macro_auc", "path": checkpoint_path, "sha256": checkpoint_digest, "selected": True, "epoch": int(checkpoint["epoch"])},
            {"checkpoint": "last", "path": checkpoint_dir / "last.pth", "sha256": sha256_file(checkpoint_dir / "last.pth"), "selected": False, "epoch": int(torch.load(checkpoint_dir / "last.pth", map_location="cpu", weights_only=False)["epoch"])},
        ]
    )
    inventory.to_csv(checkpoint_dir / "checkpoint_inventory.csv", index=False, encoding="utf-8-sig")
    return test_pred, metric_row


@torch.no_grad()
def evaluate_for_selection(model: torch.nn.Module, data_loader: DataLoader, device: torch.device, criterion: torch.nn.Module) -> tuple[dict[str, Any], float]:
    model.eval()
    probs: list[float] = []
    labels: list[int] = []
    total_loss = 0.0
    seen = 0
    for batch in data_loader:
        image = batch["image"].to(device, non_blocking=True)
        label = batch["label"].to(device, non_blocking=True)
        logits2 = model(image)
        margin = logits2[:, 1] - logits2[:, 0]
        loss = criterion(margin, label)
        total_loss += float(loss.detach()) * len(label)
        seen += len(label)
        probs.extend(torch.sigmoid(margin).detach().cpu().numpy().tolist())
        labels.extend(label.detach().cpu().numpy().astype(int).tolist())
    return binary_metrics(labels, probs), float(total_loss / seen)


def plot_fold_history(history: pd.DataFrame, figure_dir: Path, fold: int) -> None:
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(history["epoch"], history["train_loss"], label="train")
    ax.plot(history["epoch"], history["inner_val_loss"], label="inner val")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("BCE loss")
    ax.set_title(f"Fold {fold} loss")
    ax.legend()
    fig.tight_layout()
    fig.savefig(figure_dir / "loss_curve.png", dpi=160)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(history["epoch"], history["train_auc"], label="train")
    ax.plot(history["epoch"], history["inner_val_auc"], label="inner val")
    ax.axhline(0.5, color="gray", linestyle="--", linewidth=1)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("AUC")
    ax.set_title(f"Fold {fold} AUC")
    ax.legend()
    fig.tight_layout()
    fig.savefig(figure_dir / "auc_curve.png", dpi=160)
    plt.close(fig)


def combine_oof_and_metrics(root: Path, predictions: list[pd.DataFrame], fold_metrics: list[dict[str, Any]]) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    oof = pd.concat(predictions, ignore_index=True).sort_values(["outer_fold", "sample_id"], kind="stable")
    validate_oof(root, oof)
    oof.to_csv(root / "oof" / "xiaomi_b0_oof_predictions.csv", index=False, encoding="utf-8-sig")
    metrics = binary_metrics(oof["binary_label"], oof["probability_patient"])
    metrics.update(calibration_metrics(oof["binary_label"], oof["probability_patient"]))
    write_json(metrics, root / "metrics" / "xiaomi_b0_oof_metrics.json")
    pd.DataFrame([metrics]).to_csv(root / "metrics" / "xiaomi_b0_oof_metrics.csv", index=False, encoding="utf-8-sig")
    matrix = np.array([[metrics["tn"], metrics["fp"]], [metrics["fn"], metrics["tp"]]], dtype=int)
    pd.DataFrame(matrix, index=["control", "patient"], columns=["control", "patient"]).to_csv(root / "metrics" / "xiaomi_b0_confusion_matrix.csv", index_label="true_pred", encoding="utf-8-sig")
    cal = pd.DataFrame([{"calibration_intercept": metrics["calibration_intercept"], "calibration_slope": metrics["calibration_slope"], "brier": metrics["brier"]}])
    cal.to_csv(root / "metrics" / "xiaomi_b0_calibration.csv", index=False, encoding="utf-8-sig")
    fold_frame = pd.DataFrame(fold_metrics).sort_values("fold")
    fold_frame.to_csv(root / "metrics" / "xiaomi_b0_fold_metrics.csv", index=False, encoding="utf-8-sig")
    return oof, fold_frame, metrics


def validate_oof(root: Path, oof: pd.DataFrame) -> None:
    split = pd.read_csv(root / "splits" / "xiaomi_control_patient_group5fold_v1.csv", dtype={"sample_id": "string", "patient_group_id": "string"}, encoding="utf-8-sig")
    errors: list[str] = []
    if len(oof) != EXPECTED_TOTAL:
        errors.append(f"OOF rows {len(oof)} != {EXPECTED_TOTAL}")
    if oof["sample_id"].nunique() != EXPECTED_TOTAL or oof["sample_id"].duplicated().any():
        errors.append("OOF sample_id uniqueness failed")
    if set(oof["sample_id"].astype(str)) != set(split["sample_id"].astype(str)):
        errors.append("OOF sample_id set differs from split")
    if not np.isfinite(oof["probability_patient"].astype(float)).all() or not np.isfinite(oof["logit_patient"].astype(float)).all():
        errors.append("OOF probability/logit has non-finite values")
    if not np.array_equal((oof["probability_patient"].astype(float) >= 0.5).astype(int), oof["prediction_threshold_05"].astype(int)):
        errors.append("OOF threshold is not fixed at 0.5")
    merged = oof.merge(split[["sample_id", "outer_fold"]], on="sample_id", suffixes=("", "_split"))
    if not np.array_equal(merged["outer_fold"].astype(int), merged["outer_fold_split"].astype(int)):
        errors.append("OOF fold assignment differs from frozen split")
    if errors:
        raise RuntimeError(f"OOF validation failed: {errors}")


def cluster_bootstrap(frame: pd.DataFrame, metrics_fn, iterations: int = BOOTSTRAP_ITERATIONS, seed: int = SPLIT_SEED) -> tuple[pd.DataFrame, dict[str, Any]]:
    rng = np.random.default_rng(seed)
    group_ids = np.array(sorted(frame["patient_group_id"].astype(str).unique()))
    by_group = {gid: frame[frame["patient_group_id"].astype(str).eq(gid)] for gid in group_ids}
    rows: list[dict[str, Any]] = []
    invalid = 0
    for i in range(iterations):
        chosen = rng.choice(group_ids, size=len(group_ids), replace=True)
        sample = pd.concat([by_group[gid] for gid in chosen], ignore_index=True)
        if sample["binary_label"].nunique() < 2:
            invalid += 1
            continue
        m = metrics_fn(sample)
        m["iteration"] = i
        rows.append(m)
    boot = pd.DataFrame(rows)
    ci_rows = []
    for metric in ["auc", "accuracy", "balanced_accuracy", "macro_f1", "sensitivity", "specificity", "brier"]:
        values = boot[metric].dropna().astype(float)
        ci_rows.append({"metric": metric, "lower_95": float(values.quantile(0.025)), "upper_95": float(values.quantile(0.975)), "mean": float(values.mean()), "valid_n": int(len(values))})
    return pd.DataFrame(ci_rows), {"iterations": iterations, "valid": int(len(boot)), "invalid": int(invalid), "seed": seed}


def run_bootstrap(root: Path, oof: pd.DataFrame) -> pd.DataFrame:
    ci, diag = cluster_bootstrap(oof, lambda sample: binary_metrics(sample["binary_label"], sample["probability_patient"]))
    ci.to_csv(root / "bootstrap" / "xiaomi_b0_cluster_bootstrap_ci.csv", index=False, encoding="utf-8-sig")
    write_json(diag, root / "bootstrap" / "xiaomi_b0_bootstrap_diagnostics.json")
    return ci


def run_stability(root: Path, fold_metrics: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    rows = []
    for r in fold_metrics.itertuples():
        rows.append(
            {
                "fold": int(r.fold),
                "test_n": int(r.test_n),
                "control": int(r.test_control),
                "patient": int(r.test_patient),
                "patient_groups": int(r.test_patient_groups),
                "auc": float(r.test_auc),
                "accuracy": float(r.test_accuracy),
                "balanced_accuracy": float(r.test_balanced_accuracy),
                "macro_f1": float(r.test_macro_f1),
                "sensitivity": float(r.test_sensitivity),
                "specificity": float(r.test_specificity),
                "brier": float(r.test_brier),
                "selected_epoch": int(r.selected_epoch),
            }
        )
    stability = pd.DataFrame(rows)
    stability.to_csv(root / "stability" / "fold_stability_summary.csv", index=False, encoding="utf-8-sig")
    summary = {
        "auc_gt_0_5_folds": int((stability["auc"] > 0.5).sum()),
        "auc_min": float(stability["auc"].min()),
        "auc_max": float(stability["auc"].max()),
        "auc_range": float(stability["auc"].max() - stability["auc"].min()),
        "auc_sd": float(stability["auc"].std(ddof=1)),
        "balanced_accuracy_range": float(stability["balanced_accuracy"].max() - stability["balanced_accuracy"].min()),
        "sensitivity_range": float(stability["sensitivity"].max() - stability["sensitivity"].min()),
        "specificity_range": float(stability["specificity"].max() - stability["specificity"].min()),
    }
    write_json(summary, root / "stability" / "fold_stability_summary.json")
    return stability, summary


def run_overfitting_audit(root: Path, fold_metrics: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for fold in range(N_FOLDS):
        hist = pd.read_csv(root / f"fold_{fold}" / "history" / "training_history.csv")
        fm = fold_metrics[fold_metrics["fold"].eq(fold)].iloc[0]
        row = {
            "fold": fold,
            "best_train_auc": float(hist["train_auc"].max()),
            "best_inner_val_auc": float(hist["inner_val_auc"].max()),
            "train_val_auc_gap": float(hist["train_auc"].max() - hist["inner_val_auc"].max()),
            "best_train_loss": float(hist["train_loss"].min()),
            "best_inner_val_loss": float(hist["inner_val_loss"].min()),
            "selected_epoch": int(fm["selected_epoch"]),
            "early_stopping_epoch": int(hist["epoch"].max()),
            "warning_train_val_auc_gap_gt_0_20": bool(hist["train_auc"].max() - hist["inner_val_auc"].max() > 0.20),
            "warning_train_auc_near_1_val_near_random": bool(hist["train_auc"].max() > 0.95 and hist["inner_val_auc"].max() < 0.60),
            "warning_selected_epoch_early": bool(int(fm["selected_epoch"]) <= 2),
        }
        rows.append(row)
    audit = pd.DataFrame(rows)
    audit.to_csv(root / "stability" / "overfitting_audit.csv", index=False, encoding="utf-8-sig")
    write_json({"folds": audit.to_dict(orient="records"), "any_warning": bool(audit.filter(like="warning_").any().any())}, root / "stability" / "overfitting_audit.json")
    return audit


def _prepare_feature_frame(cohort: pd.DataFrame) -> pd.DataFrame:
    exif = read_csv_str(STAGE2_R2_EXIF)
    stage2 = read_csv_str(STAGE2_FEATURES)
    features = cohort[["sample_id", "patient_group_id", "binary_label"]].copy()
    exif_cols = ["sample_id", "log2_exposure_time", "log2_iso", "brightness_value"]
    features = features.merge(exif[exif_cols], on="sample_id", how="left")
    stage2_cols = ["sample_id", "skin_y_median", "skin_y_p05", "dark_contrast_score", "cheek_relative_difference"]
    features = features.merge(stage2[stage2_cols], on="sample_id", how="left")
    return features


def run_baselines(root: Path, cohort: pd.DataFrame, outer_split: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    feature_frame = _prepare_feature_frame(cohort)
    split = outer_split[["sample_id", "outer_fold"]].copy()
    feature_frame = feature_frame.merge(split, on="sample_id", how="left")
    predictions: list[pd.DataFrame] = []
    contract: dict[str, Any] = {
        "shared_outer_folds": str(root / "splits" / "xiaomi_control_patient_group5fold_v1.csv"),
        "preprocessing": "fit median imputer and standard scaler only on outer development; apply to outer test",
        "models": {},
    }
    old = pd.read_csv(ORIGINAL_E0B_OOF, dtype={"sample_id": "string", "patient_group_id": "string"}, encoding="utf-8-sig")
    old_subset = cohort[["sample_id", "patient_group_id", "binary_label"]].merge(old[["sample_id", "prob_patient", "logit_patient", "pred_class"]], on="sample_id", how="left")
    old_subset["outer_fold"] = old_subset["sample_id"].map(outer_split.set_index("sample_id")["outer_fold"])
    old_subset["model"] = "M0_full500_model_xiaomi_subset"
    old_subset["probability_patient"] = old_subset["prob_patient"].astype(float)
    old_subset["prediction_threshold_05"] = (old_subset["probability_patient"] >= 0.5).astype(int)
    predictions.append(old_subset.loc[:, ["model", "sample_id", "patient_group_id", "outer_fold", "binary_label", "probability_patient", "prediction_threshold_05"]])
    contract["models"]["M0_full500_model_xiaomi_subset"] = {"source": str(ORIGINAL_E0B_OOF), "training": "none; historical OOF restricted to Xiaomi subset"}

    baseline_specs = {
        "M1_BrightnessValue": ["brightness_value"],
        "M2_EXIF_three_variables": ["log2_exposure_time", "log2_iso", "brightness_value"],
        "M3_Stage2_lighting_four_metrics": ["skin_y_median", "skin_y_p05", "dark_contrast_score", "cheek_relative_difference"],
    }
    for model_name, cols in baseline_specs.items():
        contract["models"][model_name] = {"features": cols, "classifier": "LogisticRegression(C=1.0, solver=lbfgs)", "no_hyperparameter_tuning": True}
        fold_rows = []
        for fold in range(N_FOLDS):
            dev = feature_frame[~feature_frame["outer_fold"].eq(fold)].copy()
            test = feature_frame[feature_frame["outer_fold"].eq(fold)].copy()
            pipe = Pipeline(
                [
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler()),
                    ("logistic", LogisticRegression(C=1.0, solver="lbfgs", max_iter=1000)),
                ]
            )
            pipe.fit(dev[cols].astype(float), dev["binary_label"].astype(int))
            prob = pipe.predict_proba(test[cols].astype(float))[:, 1]
            for sid, gid, label, p in zip(test["sample_id"], test["patient_group_id"], test["binary_label"], prob):
                fold_rows.append(
                    {
                        "model": model_name,
                        "sample_id": sid,
                        "patient_group_id": gid,
                        "outer_fold": fold,
                        "binary_label": int(label),
                        "probability_patient": float(p),
                        "prediction_threshold_05": int(float(p) >= 0.5),
                    }
                )
        predictions.append(pd.DataFrame(fold_rows))
    pred = pd.concat(predictions, ignore_index=True)
    pred.to_csv(root / "baselines" / "xiaomi_baseline_oof_predictions.csv", index=False, encoding="utf-8-sig")
    metric_rows = []
    for model_name, group in pred.groupby("model"):
        metric_rows.append({"model": model_name, **binary_metrics(group["binary_label"], group["probability_patient"])})
    metrics = pd.DataFrame(metric_rows).sort_values("model")
    metrics.to_csv(root / "baselines" / "xiaomi_baseline_metrics.csv", index=False, encoding="utf-8-sig")
    write_json(contract, root / "baselines" / "xiaomi_baseline_contract.json")
    return pred, metrics


def run_comparison(root: Path, oof: pd.DataFrame, baseline_pred: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    b0 = oof[["sample_id", "patient_group_id", "binary_label", "probability_patient"]].rename(columns={"probability_patient": "b0_probability_patient"})
    rows = []
    boot_rows = []
    for model_name, base in baseline_pred.groupby("model"):
        merged = b0.merge(base[["sample_id", "probability_patient"]].rename(columns={"probability_patient": "baseline_probability_patient"}), on="sample_id")
        b0_m = binary_metrics(merged["binary_label"], merged["b0_probability_patient"])
        base_m = binary_metrics(merged["binary_label"], merged["baseline_probability_patient"])
        row = {
            "baseline": model_name,
            "auc_b0": b0_m["auc"],
            "auc_baseline": base_m["auc"],
            "delta_auc": b0_m["auc"] - base_m["auc"],
            "balanced_accuracy_b0": b0_m["balanced_accuracy"],
            "balanced_accuracy_baseline": base_m["balanced_accuracy"],
            "delta_balanced_accuracy": b0_m["balanced_accuracy"] - base_m["balanced_accuracy"],
            "brier_b0": b0_m["brier"],
            "brier_baseline": base_m["brier"],
            "delta_brier": b0_m["brier"] - base_m["brier"],
        }
        rows.append(row)
        ci = paired_cluster_bootstrap_ci(merged)
        ci["baseline"] = model_name
        boot_rows.append(ci)
    comparison = pd.DataFrame(rows).sort_values("baseline")
    boot = pd.concat(boot_rows, ignore_index=True)
    comparison.to_csv(root / "comparison" / "xiaomi_b0_vs_baselines.csv", index=False, encoding="utf-8-sig")
    boot.to_csv(root / "comparison" / "xiaomi_b0_vs_baselines_bootstrap.csv", index=False, encoding="utf-8-sig")
    comparison[comparison["baseline"].eq("M0_full500_model_xiaomi_subset")].to_csv(root / "comparison" / "xiaomi_b0_vs_fullmodel_xiaomi_subset.csv", index=False, encoding="utf-8-sig")
    return comparison, boot


def paired_delta_metrics(sample: pd.DataFrame) -> dict[str, float]:
    b0_m = binary_metrics(sample["binary_label"], sample["b0_probability_patient"])
    base_m = binary_metrics(sample["binary_label"], sample["baseline_probability_patient"])
    return {
        "auc": b0_m["auc"] - base_m["auc"],
        "accuracy": b0_m["accuracy"] - base_m["accuracy"],
        "balanced_accuracy": b0_m["balanced_accuracy"] - base_m["balanced_accuracy"],
        "macro_f1": b0_m["macro_f1"] - base_m["macro_f1"],
        "sensitivity": b0_m["sensitivity"] - base_m["sensitivity"],
        "specificity": b0_m["specificity"] - base_m["specificity"],
        "brier": b0_m["brier"] - base_m["brier"],
    }


def _fast_auc(y: np.ndarray, p: np.ndarray) -> float:
    y = y.astype(int)
    n_pos = int((y == 1).sum())
    n_neg = int((y == 0).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = np.argsort(p, kind="mergesort")
    sorted_p = p[order]
    ranks_sorted = np.empty(len(p), dtype=float)
    start = 0
    while start < len(p):
        end = start + 1
        while end < len(p) and sorted_p[end] == sorted_p[start]:
            end += 1
        avg_rank = (start + 1 + end) / 2.0
        ranks_sorted[start:end] = avg_rank
        start = end
    ranks = np.empty(len(p), dtype=float)
    ranks[order] = ranks_sorted
    sum_pos_ranks = float(ranks[y == 1].sum())
    return float((sum_pos_ranks - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def _fast_binary_metrics(y: np.ndarray, p: np.ndarray) -> dict[str, float]:
    y = y.astype(int)
    pred = (p >= 0.5).astype(int)
    tp = int(((y == 1) & (pred == 1)).sum())
    tn = int(((y == 0) & (pred == 0)).sum())
    fp = int(((y == 0) & (pred == 1)).sum())
    fn = int(((y == 1) & (pred == 0)).sum())
    sensitivity = tp / (tp + fn) if (tp + fn) else float("nan")
    specificity = tn / (tn + fp) if (tn + fp) else float("nan")
    precision_pos = tp / (tp + fp) if (tp + fp) else 0.0
    precision_neg = tn / (tn + fn) if (tn + fn) else 0.0
    recall_pos = sensitivity if np.isfinite(sensitivity) else 0.0
    recall_neg = specificity if np.isfinite(specificity) else 0.0
    f1_pos = 2 * precision_pos * recall_pos / (precision_pos + recall_pos) if (precision_pos + recall_pos) else 0.0
    f1_neg = 2 * precision_neg * recall_neg / (precision_neg + recall_neg) if (precision_neg + recall_neg) else 0.0
    return {
        "auc": _fast_auc(y, p),
        "accuracy": float((pred == y).mean()),
        "balanced_accuracy": float(np.nanmean([sensitivity, specificity])),
        "macro_f1": float((f1_pos + f1_neg) / 2.0),
        "sensitivity": float(sensitivity),
        "specificity": float(specificity),
        "brier": float(np.mean((p - y) ** 2)),
    }


def paired_cluster_bootstrap_ci(merged: pd.DataFrame, iterations: int = BOOTSTRAP_ITERATIONS, seed: int = SPLIT_SEED) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    group_ids = np.array(sorted(merged["patient_group_id"].astype(str).unique()))
    group_series = merged["patient_group_id"].astype(str).reset_index(drop=True)
    group_indices = [np.flatnonzero(group_series.to_numpy() == gid) for gid in group_ids]
    y = merged["binary_label"].to_numpy(dtype=int)
    b0 = merged["b0_probability_patient"].to_numpy(dtype=float)
    base = merged["baseline_probability_patient"].to_numpy(dtype=float)
    rows: list[dict[str, float]] = []
    for _ in range(iterations):
        chosen = rng.integers(0, len(group_indices), size=len(group_indices))
        idx = np.concatenate([group_indices[i] for i in chosen])
        yy = y[idx]
        if len(np.unique(yy)) < 2:
            continue
        b0_m = _fast_binary_metrics(yy, b0[idx])
        base_m = _fast_binary_metrics(yy, base[idx])
        rows.append({metric: b0_m[metric] - base_m[metric] for metric in ("auc", "accuracy", "balanced_accuracy", "macro_f1", "sensitivity", "specificity", "brier")})
    boot = pd.DataFrame(rows)
    ci_rows = []
    for metric in ["auc", "accuracy", "balanced_accuracy", "macro_f1", "sensitivity", "specificity", "brier"]:
        values = boot[metric].dropna().astype(float)
        ci_rows.append({"metric": metric, "lower_95": float(values.quantile(0.025)), "upper_95": float(values.quantile(0.975)), "mean": float(values.mean()), "valid_n": int(len(values))})
    return pd.DataFrame(ci_rows)


def run_brightness_audit(root: Path, oof: pd.DataFrame, cohort: pd.DataFrame, outer_split: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    features = _prepare_feature_frame(cohort).merge(outer_split[["sample_id", "outer_fold"]], on="sample_id", how="left")
    variables = ["skin_y_median", "skin_y_p05", "brightness_value"]
    cut_rows = []
    assignments = []
    for fold in range(N_FOLDS):
        dev = features[~features["outer_fold"].eq(fold)]
        test = features[features["outer_fold"].eq(fold)]
        for variable in variables:
            q33 = float(dev[variable].astype(float).quantile(1 / 3))
            q67 = float(dev[variable].astype(float).quantile(2 / 3))
            cut_rows.append({"outer_fold": fold, "variable": variable, "q33": q33, "q67": q67})
            for row in test.itertuples():
                value = float(getattr(row, variable))
                band = "low" if value <= q33 else "middle" if value <= q67 else "high"
                assignments.append({"sample_id": row.sample_id, "outer_fold": fold, "variable": variable, "value": value, "brightness_band": band})
    cut = pd.DataFrame(cut_rows)
    assign = pd.DataFrame(assignments).merge(oof[["sample_id", "patient_group_id", "binary_label", "probability_patient"]], on="sample_id", how="left")
    metric_rows = []
    for (variable, band), group in assign.groupby(["variable", "brightness_band"]):
        metric_rows.append({"variable": variable, "brightness_band": band, **binary_metrics(group["binary_label"], group["probability_patient"])})
    metrics = pd.DataFrame(metric_rows).sort_values(["variable", "brightness_band"])
    worst_rows = []
    for variable, group in metrics.groupby("variable"):
        worst_rows.append(
            {
                "variable": variable,
                "worst_group_auc": float(group["auc"].min()),
                "best_group_auc": float(group["auc"].max()),
                "delta_auc": float(group["auc"].max() - group["auc"].min()),
                "worst_group_sensitivity": float(group["sensitivity"].min()),
            }
        )
    worst = pd.DataFrame(worst_rows)
    cut.to_csv(root / "brightness_audit" / "b0_foldwise_brightness_cutpoints.csv", index=False, encoding="utf-8-sig")
    assign.to_csv(root / "brightness_audit" / "b0_foldwise_brightness_assignments.csv", index=False, encoding="utf-8-sig")
    metrics.to_csv(root / "brightness_audit" / "b0_brightness_stratified_metrics.csv", index=False, encoding="utf-8-sig")
    worst.to_csv(root / "brightness_audit" / "b0_brightness_worst_group_summary.csv", index=False, encoding="utf-8-sig")
    return metrics, worst


def make_figures(root: Path, oof: pd.DataFrame, fold_stability: pd.DataFrame, ci: pd.DataFrame, baseline_metrics: pd.DataFrame, brightness_metrics: pd.DataFrame) -> None:
    fig_dir = root / "figures"
    y = oof["binary_label"].astype(int)
    p = oof["probability_patient"].astype(float)
    fpr, tpr, _ = roc_curve(y, p)
    auc = roc_auc_score(y, p)
    auc_ci = ci[ci["metric"].eq("auc")].iloc[0]
    fig, ax = plt.subplots(figsize=(5.5, 5))
    ax.plot(fpr, tpr, label=f"B0 AUC {auc:.3f} [{auc_ci.lower_95:.3f}, {auc_ci.upper_95:.3f}], n={len(oof)}")
    ax.plot([0, 1], [0, 1], color="gray", linestyle="--", linewidth=1)
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    fig.savefig(fig_dir / "xiaomi_b0_roc_curve.png", dpi=180)
    plt.close(fig)

    _bar_plot(fig_dir / "xiaomi_b0_fold_auc.png", fold_stability["fold"].astype(str), fold_stability["auc"], "Fold", "AUC", "Fold outer-test AUC")
    cm = confusion_matrix(y, (p >= 0.5).astype(int), labels=[0, 1])
    fig, ax = plt.subplots(figsize=(4.5, 4))
    ax.imshow(cm, cmap="Blues")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center", color="black")
    ax.set_xticks([0, 1], ["Control", "Patient"])
    ax.set_yticks([0, 1], ["Control", "Patient"])
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    fig.tight_layout()
    fig.savefig(fig_dir / "xiaomi_b0_confusion_matrix.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(oof.loc[y.eq(0), "probability_patient"], bins=20, alpha=0.65, label=f"Control n={(y==0).sum()}")
    ax.hist(oof.loc[y.eq(1), "probability_patient"], bins=20, alpha=0.65, label=f"Patient n={(y==1).sum()}")
    ax.axvline(0.5, color="gray", linestyle="--", linewidth=1)
    ax.set_xlabel("Patient probability")
    ax.set_ylabel("Count")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(fig_dir / "xiaomi_b0_probability_distribution.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(5, 4))
    bins = pd.qcut(p.rank(method="first"), q=5, labels=False)
    cal = pd.DataFrame({"bin": bins, "p": p, "y": y}).groupby("bin").agg(mean_p=("p", "mean"), frac_y=("y", "mean"), n=("y", "size")).reset_index()
    ax.plot([0, 1], [0, 1], color="gray", linestyle="--", linewidth=1)
    ax.plot(cal["mean_p"], cal["frac_y"], marker="o")
    for row in cal.itertuples():
        ax.text(row.mean_p, row.frac_y, f"n={row.n}", fontsize=7)
    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Observed patient fraction")
    fig.tight_layout()
    fig.savefig(fig_dir / "xiaomi_b0_calibration_curve.png", dpi=180)
    plt.close(fig)

    histories = []
    for fold in range(N_FOLDS):
        h = pd.read_csv(root / f"fold_{fold}" / "history" / "training_history.csv")
        h["fold"] = fold
        histories.append(h)
    hist = pd.concat(histories, ignore_index=True)
    for metric, filename, ylabel in (("auc", "xiaomi_b0_training_validation_auc.png", "AUC"), ("loss", "xiaomi_b0_training_validation_loss.png", "BCE loss")):
        fig, ax = plt.subplots(figsize=(7, 4))
        for fold, h in hist.groupby("fold"):
            ax.plot(h["epoch"], h[f"train_{metric}"], color="#2b6cb0", alpha=0.35)
            ax.plot(h["epoch"], h[f"inner_val_{metric}"], color="#c2410c", alpha=0.35)
        ax.plot([], [], color="#2b6cb0", label="train")
        ax.plot([], [], color="#c2410c", label="inner val")
        ax.set_xlabel("Epoch")
        ax.set_ylabel(ylabel)
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(fig_dir / filename, dpi=180)
        plt.close(fig)

    _bar_plot(fig_dir / "xiaomi_b0_vs_baseline_auc.png", baseline_metrics["model"], baseline_metrics["auc"], "Model", "AUC", "Baseline AUC")
    for metric, filename, ylabel in (("auc", "xiaomi_b0_brightness_stratified_auc.png", "AUC"), ("sensitivity", "xiaomi_b0_brightness_stratified_sensitivity.png", "Sensitivity")):
        fig, ax = plt.subplots(figsize=(8, 4))
        labels = brightness_metrics["variable"] + ":" + brightness_metrics["brightness_band"]
        ax.bar(labels, brightness_metrics[metric])
        ax.set_ylabel(ylabel)
        ax.tick_params(axis="x", labelrotation=45)
        fig.tight_layout()
        fig.savefig(fig_dir / filename, dpi=180)
        plt.close(fig)


def _bar_plot(path: Path, labels: Iterable[Any], values: Iterable[Any], xlabel: str, ylabel: str, title: str) -> None:
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar([str(x) for x in labels], list(values))
    ax.axhline(0.5, color="gray", linestyle="--", linewidth=1)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.tick_params(axis="x", labelrotation=30)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def classify_signal(metrics: dict[str, Any], ci: pd.DataFrame, stability_summary: dict[str, Any], comparison: pd.DataFrame) -> dict[str, Any]:
    auc_ci = ci[ci["metric"].eq("auc")].iloc[0]
    b0_auc = float(metrics["auc"])
    ba = float(metrics["balanced_accuracy"])
    sens = float(metrics["sensitivity"])
    spec = float(metrics["specificity"])
    auc_gt = int(stability_summary["auc_gt_0_5_folds"])
    main_baselines = comparison[comparison["baseline"].ne("M0_full500_model_xiaomi_subset")]
    beats_simple = bool((main_baselines["delta_auc"] > 0).sum() >= 2)
    if b0_auc > 0.70 and auc_ci.lower_95 > 0.5 and auc_gt >= 4 and ba > 0.60 and min(sens, spec) > 0.45 and beats_simple:
        level = "strong"
    elif b0_auc > 0.62 and auc_ci.lower_95 > 0.5 and auc_gt >= 4 and ba > 0.55 and min(sens, spec) > 0.35:
        level = "moderate"
    elif b0_auc > 0.55 and auc_gt >= 3:
        level = "weak"
    elif abs(b0_auc - 0.5) <= 0.05 or ba < 0.53:
        level = "none"
    else:
        level = "indeterminate"
    decision = {
        "same_camera_signal_level": level,
        "stage3_b0_a0_eligible": level in {"moderate", "strong"} or ("conditional" if level == "weak" else False),
        "stage3_b1_eligible": "conditional" if level in {"weak", "moderate", "strong"} else False,
        "recommend_stage3_b0_a0": level in {"moderate", "strong"},
        "recommend_stage3_b1": False,
        "rationale": "Stage3-B1 should wait until B0-A0 confirms brightness sensitivity on B0 checkpoints.",
    }
    if decision["stage3_b0_a0_eligible"] == "conditional":
        decision["recommend_stage3_b0_a0"] = "conditional_after_seed_stability"
    return decision


def write_reports(root: Path, env: dict[str, Any], contract: dict[str, Any], cohort: pd.DataFrame, outer_split: pd.DataFrame, inner_all: pd.DataFrame, fold_metrics: pd.DataFrame, oof_metrics: dict[str, Any], ci: pd.DataFrame, stability_summary: dict[str, Any], overfit: pd.DataFrame, baseline_metrics: pd.DataFrame, comparison: pd.DataFrame, brightness_metrics: pd.DataFrame, brightness_worst: pd.DataFrame, decision: dict[str, Any]) -> None:
    write_json(decision, root / "reports" / "stage3_b0_signal_classification.json")
    write_text(
        f"""# Stage3-B0 Next Stage Decision

- same_camera_signal_level: {decision['same_camera_signal_level']}
- stage3_b0_a0_eligible: {decision['stage3_b0_a0_eligible']}
- stage3_b1_eligible: {decision['stage3_b1_eligible']}
- recommend_stage3_b0_a0: {decision['recommend_stage3_b0_a0']}
- recommend_stage3_b1: {decision['recommend_stage3_b1']}

{decision['rationale']}
""",
        root / "reports" / "stage3_b0_next_stage_decision.md",
    )
    machine = {
        "environment": env,
        "cohort": {"n": len(cohort), "control": int((cohort["binary_label"] == 0).sum()), "patient": int((cohort["binary_label"] == 1).sum()), "patient_groups": int(cohort["patient_group_id"].nunique())},
        "outer_fold_distribution": pd.read_csv(root / "splits" / "xiaomi_group5fold_audit.csv").to_dict(orient="records"),
        "inner_split_distribution": pd.read_csv(root / "splits" / "inner" / "inner_split_audit.csv").to_dict(orient="records"),
        "fold_metrics": fold_metrics.to_dict(orient="records"),
        "oof_metrics": oof_metrics,
        "bootstrap_ci": ci.to_dict(orient="records"),
        "stability_summary": stability_summary,
        "baseline_metrics": baseline_metrics.to_dict(orient="records"),
        "comparison": comparison.to_dict(orient="records"),
        "brightness_worst": brightness_worst.to_dict(orient="records"),
        "decision": decision,
    }
    write_json(machine, root / "reports" / "stage3_b0_machine_summary.json")
    inventory = [{"path": str(p.relative_to(root)), "bytes": p.stat().st_size} for p in sorted(root.rglob("*")) if p.is_file()]
    write_json({"root": str(root), "files": inventory}, root / "reports" / "stage3_b0_output_inventory.json")
    auc_ci = ci[ci["metric"].eq("auc")].iloc[0]
    baseline_rows = "\n".join(f"| {r.model} | {r.auc:.4f} | {r.balanced_accuracy:.4f} | {r.macro_f1:.4f} | {r.sensitivity:.4f} | {r.specificity:.4f} | {r.brier:.4f} |" for r in baseline_metrics.itertuples())
    fold_rows = "\n".join(f"| {int(r.fold)} | {int(r.test_n)} | {int(r.test_control)} | {int(r.test_patient)} | {r.test_auc:.4f} | {r.test_balanced_accuracy:.4f} | {r.test_macro_f1:.4f} | {int(r.selected_epoch)} | {r.inner_val_auc_at_selected_epoch:.4f} |" for r in fold_metrics.itertuples())
    comparison_rows = "\n".join(f"| {r.baseline} | {r.auc_b0:.4f} | {r.auc_baseline:.4f} | {r.delta_auc:.4f} | {r.delta_balanced_accuracy:.4f} | {r.delta_brier:.4f} |" for r in comparison.itertuples())
    brightness_rows = "\n".join(f"| {r.variable} | {r.brightness_band} | {int(r.n)} | {int(r.control)} | {int(r.patient)} | {r.auc:.4f} | {r.balanced_accuracy:.4f} | {r.sensitivity:.4f} | {r.specificity:.4f} | {r.brier:.4f} |" for r in brightness_metrics.itertuples())
    report = f"""# Stage3-B0 Xiaomi-only ResNet18 Report

## Purpose

This experiment retrained an RGB ResNet18 after restricting the cohort to the exact same camera model, Xiaomi `M2006J10C`, to test for a same-camera discriminative signal between Control and Patient. It does not test cross-camera generalization and does not establish a pure medical signal.

## Cohort

- Xiaomi samples: {len(cohort)}
- Control: {int((cohort['binary_label'] == 0).sum())}
- Patient: {int((cohort['binary_label'] == 1).sum())}
- Patient groups: {int(cohort['patient_group_id'].nunique())}
- Camera model after normalization: {', '.join(cohort['camera_model_normalized'].unique())}

No HONOR samples were used for training or testing.

## Splits and Training Contract

Outer folds were rebuilt with `StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=2026)` at patient-group level. Inner validation used the first split from `StratifiedGroupKFold` within each outer development set with seed `12026 + outer_fold`.

The model reused the E0B ResNet18/ImageNet/dropout/input-size/normalization/horizontal-flip/AdamW/BN-eval contract. The necessary B0 change was the loss: `BCEWithLogitsLoss(pos_weight=1.0)` applied to the patient-minus-control logit margin. EXIF, camera model, Stage3-A0 counterfactual images, exposure/gamma augmentation, and Stage3-B1 procedures were not used as ResNet inputs or training objectives.

## Pooled OOF Metrics

- ROC-AUC: {oof_metrics['auc']:.4f} (patient-group cluster bootstrap 95% CI {auc_ci.lower_95:.4f} to {auc_ci.upper_95:.4f})
- Accuracy: {oof_metrics['accuracy']:.4f}
- Balanced Accuracy: {oof_metrics['balanced_accuracy']:.4f}
- Macro-F1: {oof_metrics['macro_f1']:.4f}
- Sensitivity: {oof_metrics['sensitivity']:.4f}
- Specificity: {oof_metrics['specificity']:.4f}
- Brier: {oof_metrics['brier']:.4f}
- Calibration intercept: {oof_metrics['calibration_intercept']:.4f}
- Calibration slope: {oof_metrics['calibration_slope']:.4f}

## Fold Results

| Fold | N | Control | Patient | Test AUC | BA | Macro-F1 | Best epoch | Inner val AUC |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
{fold_rows}

## Stability and Overfitting

- Folds with AUC > 0.5: {stability_summary['auc_gt_0_5_folds']}/5
- AUC range: {stability_summary['auc_min']:.4f} to {stability_summary['auc_max']:.4f} (range {stability_summary['auc_range']:.4f}, SD {stability_summary['auc_sd']:.4f})
- Any overfitting warning: {bool(overfit.filter(like='warning_').any().any())}

## Baselines

| Model | AUC | BA | Macro-F1 | Sensitivity | Specificity | Brier |
|---|---:|---:|---:|---:|---:|---:|
{baseline_rows}

## Paired Comparison

| Baseline | B0 AUC | Baseline AUC | Delta AUC | Delta BA | Delta Brier |
|---|---:|---:|---:|---:|---:|
{comparison_rows}

## Brightness Stratification

| Variable | Band | N | Control | Patient | AUC | BA | Sensitivity | Specificity | Brier |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
{brightness_rows}

## Signal Classification

- same_camera_signal_level: `{decision['same_camera_signal_level']}`
- stage3_b0_a0_eligible: `{decision['stage3_b0_a0_eligible']}`
- stage3_b1_eligible: `{decision['stage3_b1_eligible']}`

The result should be described as a same-camera discriminative signal only. It must not be described as pure medical signal or as evidence that all acquisition shortcuts have been removed.

## Limitations

Fixed camera model does not remove collection time, operator, environment, ISP/HDR mode, lighting, workflow, or population-composition confounding. Stage3-B1 was not run.
"""
    write_text(report, root / "reports" / "stage3_b0_report.md")


def run_pipeline(output_dir: Path = OUTPUT_DIR, resume: bool = True, stop_after_preflight: bool = False) -> Path:
    root = output_dir.resolve()
    ensure_dirs(root)
    setup_logging(root)
    LOGGER.info("Stage3-B0 pipeline root=%s", root)
    env = save_runtime_environment(root)
    contract = extract_e0b_contract(root)
    cohort = build_xiaomi_cohort(root)
    outer_split = build_outer_splits(root, cohort)
    inner_all = build_inner_splits(root, outer_split)
    write_preflight(root, env, contract, cohort, outer_split, inner_all)
    if stop_after_preflight:
        LOGGER.info("Stopping after preflight by request")
        return root
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    predictions: list[pd.DataFrame] = []
    fold_metrics: list[dict[str, Any]] = []
    for fold in range(N_FOLDS):
        pred, met = train_one_fold(root, contract, outer_split, fold, device, resume=resume)
        predictions.append(pred)
        fold_metrics.append(met)
    oof, fold_frame, oof_metrics = combine_oof_and_metrics(root, predictions, fold_metrics)
    ci = run_bootstrap(root, oof)
    fold_stability, stability_summary = run_stability(root, fold_frame)
    overfit = run_overfitting_audit(root, fold_frame)
    baseline_pred, baseline_metrics = run_baselines(root, cohort, outer_split)
    comparison, comparison_boot = run_comparison(root, oof, baseline_pred)
    brightness_metrics, brightness_worst = run_brightness_audit(root, oof, cohort, outer_split)
    decision = classify_signal(oof_metrics, ci, stability_summary, comparison)
    make_figures(root, oof, fold_stability, ci, baseline_metrics, brightness_metrics)
    write_reports(root, env, contract, cohort, outer_split, inner_all, fold_frame, oof_metrics, ci, stability_summary, overfit, baseline_metrics, comparison, brightness_metrics, brightness_worst, decision)
    write_text(datetime.now().isoformat(timespec="seconds") + "\n", root / "run_finished_at.txt")
    LOGGER.info("Stage3-B0 complete: %s", root)
    return root


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--stop-after-preflight", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = run_pipeline(args.output_dir, resume=not args.no_resume, stop_after_preflight=args.stop_after_preflight)
    print(f"STAGE3_B0_DIR={root}")


if __name__ == "__main__":
    main()
