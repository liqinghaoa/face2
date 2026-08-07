"""Protocol guards for the fixed P2-B0 RGB fold-0 baseline.

This module deliberately contains no image/model/training implementation.
P2-B0 reuses E0B for those operations and only adds immutable-protocol and
P1-ID alignment checks around it.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from datasets.control_patient_binary_dataset import map_three_class_to_binary
from utils.experiment_utils import load_yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
E0B_CONFIG = PROJECT_ROOT / "config" / "train" / "e0b_global_resnet18_control_patient_binary_5fold.yaml"

_REUSED_E0B_SECTIONS = ("data", "model", "train", "augmentation", "normalize")
_P2_DISABLED_FLAGS = (
    "use_p1_features",
    "use_relighting",
    "use_auxiliary_features",
    "use_reliability_gate",
)


def project_path(value: str | Path) -> Path:
    """Resolve a project-relative path without changing the current directory."""

    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def e0b_protocol_differences(config: dict[str, Any]) -> list[str]:
    """Return differing E0B protocol sections; an empty list means exact reuse."""

    baseline = load_yaml(E0B_CONFIG)
    differences: list[str] = []
    for section in _REUSED_E0B_SECTIONS:
        if config.get(section) != baseline.get(section):
            differences.append(section)
    return differences


def validate_p2_b0_config(config: dict[str, Any], *, allow_smoke_epoch_override: bool = False) -> None:
    """Fail closed unless a configuration is the fixed E0B-compatible P2-B0 task."""

    experiment = config.get("experiment", {})
    task = config.get("task", {})
    p2 = config.get("p2", {})
    if experiment.get("name") != "P2_B0_BinaryRGB_Fold0_v1":
        raise ValueError("P2-B0 experiment.name must be P2_B0_BinaryRGB_Fold0_v1")
    if experiment.get("stage") != "P2" or experiment.get("role") != "fixed_single_fold_baseline":
        raise ValueError("P2-B0 must declare stage=P2 and role=fixed_single_fold_baseline")
    if task.get("type") != "binary_control_vs_patient" or int(task.get("num_classes", -1)) != 2:
        raise ValueError("P2-B0 requires binary_control_vs_patient with two classes")
    if int(config.get("model", {}).get("num_classes", -1)) != 2:
        raise ValueError("P2-B0 model.num_classes must be 2")
    if int(p2.get("fixed_fold", -1)) != 0:
        raise ValueError("P2-B0 is fixed to fold 0")
    enabled = [name for name in _P2_DISABLED_FLAGS if p2.get(name) is not False]
    if enabled:
        raise ValueError(f"P2-B0 must disable all P1/relighting/auxiliary/gate features: {enabled}")
    if not p2.get("p1_ready_cases_csv"):
        raise ValueError("P2-B0 requires p2.p1_ready_cases_csv for the read-only ID alignment check")
    differences = e0b_protocol_differences(config)
    if allow_smoke_epoch_override and differences == ["train"]:
        baseline_epochs = int(load_yaml(E0B_CONFIG)["train"]["epochs"])
        actual_train = dict(config["train"])
        actual_epochs = int(actual_train.pop("epochs"))
        baseline_train = dict(load_yaml(E0B_CONFIG)["train"])
        baseline_train.pop("epochs")
        if 1 <= actual_epochs < baseline_epochs and actual_train == baseline_train:
            differences = []
    if differences:
        raise ValueError(f"P2-B0 must exactly reuse E0B sections; differences: {differences}")


def verify_fold0_split_protocol(config: dict[str, Any]) -> dict[str, int]:
    """Check the fixed fold-0 counts and patient-group isolation before training."""

    data = config["data"]
    split_dir = project_path(data["split_dir"])
    train = pd.read_csv(
        split_dir / data["train_csv_pattern"].format(fold=0),
        dtype={"ID": "string", "patient_group_id": "string"},
        encoding="utf-8-sig",
    )
    val = pd.read_csv(
        split_dir / data["val_csv_pattern"].format(fold=0),
        dtype={"ID": "string", "patient_group_id": "string"},
        encoding="utf-8-sig",
    )
    required = {"ID", "patient_group_id", "label_3class", "fold"}
    for name, frame in (("train", train), ("val", val)):
        missing = required.difference(frame.columns)
        if missing:
            raise ValueError(f"fold 0 {name} CSV lacks columns: {sorted(missing)}")
        if frame["ID"].isna().any() or frame["ID"].duplicated().any():
            raise ValueError(f"fold 0 {name} IDs must be non-null and unique")
        frame["binary_label"] = frame["label_3class"].map(map_three_class_to_binary)
    # In this project's fixed-split convention, train rows retain the fold in
    # which they are validated (1--4 for fold_0_train.csv).  Only the held-out
    # validation table is expected to carry fold=0 throughout.
    if not (pd.to_numeric(val["fold"], errors="coerce") == 0).all():
        raise ValueError("fold_0_val.csv has a non-zero fold field")
    overlap = set(train["patient_group_id"].astype(str)).intersection(val["patient_group_id"].astype(str))
    if overlap:
        raise ValueError(f"fold 0 train/val patient_group_id leakage: {sorted(overlap)[:10]}")
    counts = {
        "train_total": int(len(train)),
        "train_control": int((train["binary_label"] == 0).sum()),
        "train_patient": int((train["binary_label"] == 1).sum()),
        "val_total": int(len(val)),
        "val_control": int((val["binary_label"] == 0).sum()),
        "val_patient": int((val["binary_label"] == 1).sum()),
        "train_patient_groups": int(train["patient_group_id"].nunique()),
        "val_patient_groups": int(val["patient_group_id"].nunique()),
    }
    expected = {
        "train_total": 400,
        "train_control": 92,
        "train_patient": 308,
        "val_total": 100,
        "val_control": 23,
        "val_patient": 77,
    }
    mismatches = {key: {"expected": value, "observed": counts[key]} for key, value in expected.items() if counts[key] != value}
    if mismatches:
        raise ValueError(f"P2-B0 fold-0 split differs from the fixed E0B protocol: {mismatches}")
    return counts


def check_p1_ready_case_alignment(config: dict[str, Any], output_dir: Path) -> dict[str, Any]:
    """Perform exact, read-only P1 case-ID alignment and persist its audit record.

    Only the P1 CSV column ``case_id`` is read.  No P1 latent, map, relighting,
    quality, or RGB artifact path is opened or used to select a sample.
    """

    data, p2 = config["data"], config["p2"]
    p1_path = project_path(p2["p1_ready_cases_csv"])
    split_dir = project_path(data["split_dir"])
    errors: list[str] = []
    if not p1_path.is_file():
        raise FileNotFoundError(f"P1 ready-case index does not exist: {p1_path}")
    p1 = pd.read_csv(p1_path, usecols=["case_id"], dtype={"case_id": "string"}, encoding="utf-8-sig")
    p1_ids = p1["case_id"].astype(str)
    if len(p1) != 500:
        errors.append(f"P1 ready cases must contain 500 rows, observed {len(p1)}")
    if p1_ids.isna().any() or p1_ids.duplicated().any():
        errors.append("P1 case_id values must be non-null and unique")

    validation_ids: list[str] = []
    for fold in range(int(data["n_folds"])):
        val = pd.read_csv(
            split_dir / data["val_csv_pattern"].format(fold=fold),
            usecols=["ID"], dtype={"ID": "string"}, encoding="utf-8-sig",
        )
        validation_ids.extend(val["ID"].astype(str).tolist())
    validation_set = set(validation_ids)
    p1_set = set(p1_ids.tolist())
    if len(validation_ids) != 500 or len(validation_set) != 500:
        errors.append("The fixed five-fold validation IDs must consist of 500 unique cases")
    if p1_set != validation_set:
        errors.append("P1 case_id set does not exactly equal the fixed five-fold validation ID set")

    fold0_ids: dict[str, set[str]] = {}
    for split_name, pattern in (("train", data["train_csv_pattern"]), ("val", data["val_csv_pattern"])):
        frame = pd.read_csv(
            split_dir / pattern.format(fold=0), usecols=["ID"], dtype={"ID": "string"}, encoding="utf-8-sig",
        )
        fold0_ids[split_name] = set(frame["ID"].astype(str).tolist())
        if not fold0_ids[split_name].issubset(p1_set):
            errors.append(f"fold 0 {split_name} contains IDs absent from P1 ready cases")

    report: dict[str, Any] = {
        "check": "P2-B0 read-only exact P1 case-ID alignment",
        "status": "passed" if not errors else "failed",
        "p1_ready_cases_csv": str(p1_path),
        "p1_fields_read": ["case_id"],
        "p1_artifacts_read": [],
        "p1_rows": int(len(p1)),
        "p1_unique_case_ids": int(len(p1_set)),
        "fixed_oof_rows": int(len(validation_ids)),
        "fixed_oof_unique_ids": int(len(validation_set)),
        "full_id_set_match": p1_set == validation_set,
        "fold0_train_cases": int(len(fold0_ids["train"])),
        "fold0_val_cases": int(len(fold0_ids["val"])),
        "fold0_train_missing_from_p1": sorted(fold0_ids["train"] - p1_set),
        "fold0_val_missing_from_p1": sorted(fold0_ids["val"] - p1_set),
        "errors": errors,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "p1_ready_case_alignment.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    if errors:
        raise ValueError("; ".join(errors))
    return report
