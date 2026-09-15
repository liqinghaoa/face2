"""Frozen SO-1D formal-training protocol checks and audit helpers."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from typing import Any

import pandas as pd
import torch

from .checkpoint import load_checkpoint
from .contract import canonical_json_hash, sha256_file
from .unet_decomposer import SO1UNetDecomposer


FORMAL_PROTOCOL_VERSION = "SO1D_Formal_Train_v1.0"
FORMAL_RUN_NAME = "SO1D_Formal_Train_v1"
TARGET_ORDER = ["M_norm", "H_norm", "S_norm", "P_norm"]
CONTRACT_TARGET_ORDER = ["M", "H", "S_norm", "P_norm"]
NORMALIZATION = {
    "M": "identity",
    "H": "identity",
    "S_norm": "(S - 0.25) / 1.75",
    "P_norm": "P / 0.10",
}
FORBIDDEN_SPLITS = ["id_test", "camera_ood", "light_ood", "joint_ood"]
EXPECTED_GENERATOR_CONFIG_HASH = (
    "78256a4f18072257d407c46f311c5903a66e0242ee945ff62e1d1a18afcc5cdf"
)
EXPECTED_SO0_VERSION = "SO0_Forward_Model_v1.1"
EXPECTED_SO0_CONFIG_HASH = (
    "32b7304a4ed2132bb578a51a4b8647c36934ae3c347e77137dfc42aaca4d5fea"
)


def file_sha256(path: str | Path) -> str:
    return sha256_file(path)


def formal_resume_signature(config: dict[str, Any]) -> str:
    frozen = {
        "protocol_version": config["protocol_version"],
        "seed": config["seed"],
        "model": config["model"],
        "loss": config["loss"],
        "optimizer": config["optimizer"],
        "scheduler": config["scheduler"],
        "training": config["training"],
        "early_stopping": config["early_stopping"],
        "checkpoint": config["checkpoint"],
        "dataloader": config["dataloader"],
        "augmentation": config["augmentation"],
    }
    return canonical_json_hash(frozen)


def validate_formal_config(config: dict[str, Any]) -> None:
    required = {
        "phase": "SO-1D",
        "protocol_version": FORMAL_PROTOCOL_VERSION,
        "seed": 20260801,
    }
    for key, expected in required.items():
        if config.get(key) != expected:
            raise ValueError(f"Formal config mismatch for {key}: {config.get(key)!r} != {expected!r}")
    checks = {
        "data.train_split": config["data"]["train_split"] == "train",
        "data.validation_split": config["data"]["validation_split"] == "validation",
        "data.train_samples": config["data"]["train_samples"] == 20_000,
        "data.validation_samples": config["data"]["validation_samples"] == 2_000,
        "data.forbidden_splits": config["data"]["forbidden_splits"] == FORBIDDEN_SPLITS,
        "model": config["model"] == {
            "architecture": "standard_unet",
            "encoder_channels": [64, 128, 256, 512],
            "bottleneck_channels": 1024,
            "decoder_channels": [512, 256, 128, 64],
            "output_channels": 4,
            "output_order": TARGET_ORDER,
            "normalization": "BatchNorm",
            "activation": "LeakyReLU",
            "final_activation": "Sigmoid",
        },
        "loss": config["loss"] == {
            "type": "masked_smooth_l1", "beta": 0.1,
            "channel_weights": [0.25, 0.25, 0.25, 0.25],
        },
        "optimizer": config["optimizer"] == {
            "type": "AdamW", "lr": 0.001, "weight_decay": 0.0001,
        },
        "scheduler": config["scheduler"] == {
            "type": "CosineAnnealingLR", "T_max": 30,
            "eta_min": 0.000001, "step_unit": "epoch",
        },
        "training": config["training"] == {"batch_size": 8, "amp": True, "max_epochs": 30},
        "early_stopping": config["early_stopping"] == {
            "enabled": True, "monitor": "val_M_MAE_plus_val_H_MAE",
            "mode": "min", "patience": 5, "min_delta": 0.0,
        },
        "checkpoint.selection_metric": (
            config["checkpoint"]["selection_metric"] == "val_M_MAE_plus_val_H_MAE"
        ),
        "checkpoint.milestone_epochs": (
            config["checkpoint"]["milestone_epochs"] == [5, 10, 15, 20, 25, 30]
        ),
        "dataloader": config["dataloader"] == {
            "num_workers": 2, "pin_memory": True,
            "persistent_workers": True, "prefetch_factor": 2,
        },
        "augmentation.enabled": config["augmentation"]["enabled"] is False,
    }
    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        raise ValueError(f"Frozen SO-1D config mismatch: {failed}")


def _unique_column(frame: pd.DataFrame, column: str) -> str:
    values = sorted(set(frame[column].dropna().astype(str)))
    if len(values) != 1:
        raise RuntimeError(f"Expected one {column} value, found {values[:5]}")
    return values[0]


def validate_dataset_contract(
    *,
    data_root: Path,
    so1c_contract_path: Path,
    so1c_checkpoint_path: Path,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    completed = json.loads((data_root / "COMPLETED.json").read_text(encoding="utf-8"))
    if completed.get("status") != "COMPLETED" or int(completed.get("rows", -1)) != 27_000:
        raise RuntimeError("COMPLETED.json status/rows mismatch")
    contract = json.loads(so1c_contract_path.read_text(encoding="utf-8"))
    if contract.get("schema_version") != "SO1_Decomposition_DatasetContract_v1":
        raise RuntimeError("SO-1C dataset contract schema mismatch")
    if contract.get("split_counts", {}).get("train") != 20_000:
        raise RuntimeError("Dataset contract train count mismatch")
    if contract.get("split_counts", {}).get("validation") != 2_000:
        raise RuntimeError("Dataset contract validation count mismatch")
    if contract.get("target", {}).get("channel_order") != CONTRACT_TARGET_ORDER:
        raise RuntimeError("Dataset contract target order mismatch")
    if contract.get("target", {}).get("normalization") != NORMALIZATION:
        raise RuntimeError("Dataset contract normalization mismatch")
    contract_hash = canonical_json_hash(contract)
    checkpoint = torch.load(so1c_checkpoint_path, map_location="cpu", weights_only=False)
    if checkpoint.get("dataset_contract_hash") != contract_hash:
        raise RuntimeError("SO-1C checkpoint dataset contract hash mismatch")
    if checkpoint.get("generator_config_hash") != EXPECTED_GENERATOR_CONFIG_HASH:
        raise RuntimeError("SO-1C checkpoint generator config hash mismatch")
    if checkpoint.get("target_order") != TARGET_ORDER:
        raise RuntimeError("SO-1C checkpoint target order mismatch")
    if checkpoint.get("model_architecture") != SO1UNetDecomposer.architecture_id:
        raise RuntimeError("SO-1C checkpoint model architecture mismatch")

    train_metadata = pd.read_csv(data_root / "train" / "metadata.csv")
    validation_metadata = pd.read_csv(data_root / "validation" / "metadata.csv")
    if len(train_metadata) != 20_000 or len(validation_metadata) != 2_000:
        raise RuntimeError("Formal train/validation metadata counts mismatch")
    for split, frame in (("train", train_metadata), ("validation", validation_metadata)):
        if set(frame["split"].astype(str)) != {split}:
            raise RuntimeError(f"{split} metadata split mismatch")
        if frame.sort_values("split_index")["split_index"].astype(int).tolist() != list(range(len(frame))):
            raise RuntimeError(f"{split} split_index is not dense 0-based")
    combined = pd.concat([train_metadata, validation_metadata], ignore_index=True)
    so0_version = _unique_column(combined, "so0_version")
    so0_config_hash = _unique_column(combined, "so0_config_hash")
    generator_hash = _unique_column(combined, "generator_config_hash")
    if so0_version != EXPECTED_SO0_VERSION:
        raise RuntimeError("Synthetic metadata SO-0 version mismatch")
    if so0_config_hash != EXPECTED_SO0_CONFIG_HASH:
        raise RuntimeError("Synthetic metadata SO-0 config hash mismatch")
    if generator_hash != EXPECTED_GENERATOR_CONFIG_HASH:
        raise RuntimeError("Synthetic metadata generator config hash mismatch")
    fingerprint = {
        "contract_hash": contract_hash,
        "data_root": str(data_root),
        "train_metadata_sha256": sha256_file(data_root / "train" / "metadata.csv"),
        "validation_metadata_sha256": sha256_file(data_root / "validation" / "metadata.csv"),
    }
    audit = {
        "status": "PASS",
        "completed_status": completed["status"],
        "completed_rows": completed["rows"],
        "train_samples": len(train_metadata),
        "validation_samples": len(validation_metadata),
        "target_order": TARGET_ORDER,
        "normalization": NORMALIZATION,
        "dataset_contract_hash": contract_hash,
        "generator_config_hash": generator_hash,
        "SO0_version": so0_version,
        "SO0_config_hash": so0_config_hash,
        "so1c_checkpoint_path": str(so1c_checkpoint_path),
        "so1c_legacy_SO0_version": checkpoint.get("SO0_version"),
        "so1c_legacy_SO0_config_hash": checkpoint.get("SO0_config_hash"),
        "so1c_legacy_missing_SO0_fields_resolved_from_frozen_metadata": True,
    }
    return contract, fingerprint, audit


def build_formal_identity(
    *, contract_hash: str, config: dict[str, Any]
) -> dict[str, Any]:
    return {
        "target_order": TARGET_ORDER,
        "normalization": NORMALIZATION,
        "dataset_contract_hash": contract_hash,
        "generator_config_hash": EXPECTED_GENERATOR_CONFIG_HASH,
        "SO0_version": EXPECTED_SO0_VERSION,
        "SO0_config_hash": EXPECTED_SO0_CONFIG_HASH,
        "model_architecture": SO1UNetDecomposer.architecture_id,
        "formal_resume_signature": formal_resume_signature(config),
    }


def split_access_payload(*, train_accessed: bool, validation_accessed: bool) -> dict[str, Any]:
    return {
        "train": {"accessed": bool(train_accessed)},
        "validation": {"accessed": bool(validation_accessed)},
        **{split: {"accessed": False} for split in FORBIDDEN_SPLITS},
        "forbidden_splits_accessed": [],
        "status": "PASS" if train_accessed and validation_accessed else "IN_PROGRESS",
    }


def verify_history_continuity(path: Path, *, checkpoint_epoch: int) -> list[int]:
    if not path.is_file():
        raise RuntimeError("training_history.csv is required for resume")
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    epochs = [int(float(row["epoch"])) for row in rows]
    if epochs != list(range(1, checkpoint_epoch + 1)):
        raise RuntimeError(f"REFUSE_RESUME: history discontinuity {epochs}")
    return epochs


def audit_formal_resume(
    *, checkpoint_path: Path, model: SO1UNetDecomposer,
    identity: dict[str, Any], config_sha256: str, history_path: Path,
) -> dict[str, Any]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if checkpoint.get("config_sha256") != config_sha256:
        raise RuntimeError("REFUSE_RESUME: config SHA256 mismatch")
    for key, expected in identity.items():
        if checkpoint.get(key) != expected:
            raise RuntimeError(f"REFUSE_RESUME: {key} mismatch")
    epochs = verify_history_continuity(history_path, checkpoint_epoch=int(checkpoint["epoch"]))
    return {
        "status": "PASS",
        "checkpoint_path": str(checkpoint_path),
        "resume_epoch": int(checkpoint["epoch"]),
        "resume_global_step": int(checkpoint["global_step"]),
        "history_epochs": epochs,
        "config_match": True,
        "contract_match": True,
        "optimizer_present": checkpoint.get("optimizer_state_dict") is not None,
        "scheduler_present": checkpoint.get("scheduler_state_dict") is not None,
        "scaler_present": checkpoint.get("grad_scaler_state_dict") is not None,
        "rng_present": checkpoint.get("rng_state") is not None,
    }
