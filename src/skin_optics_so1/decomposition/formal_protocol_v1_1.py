"""Frozen protocol and overflow gates for SO-1D formal training v1.1."""

from __future__ import annotations

import csv
from collections import deque
from pathlib import Path
from typing import Any

import torch

from .contract import canonical_json_hash
from .formal_protocol import FORBIDDEN_SPLITS, TARGET_ORDER
from .unet_decomposer import SO1UNetDecomposer, count_parameters


PROTOCOL_VERSION = "SO1D_Formal_Train_v1.1"
RUN_NAME = "SO1D_Formal_Train_v1_1"
NUMERICAL_POLICY = {
    "global_amp": True,
    "mode": "localized_fp32",
    "fp32_blocks": ["up1.conv"],
}


def validate_config(config: dict[str, Any]) -> None:
    exact = {
        "phase": "SO-1D",
        "protocol_version": PROTOCOL_VERSION,
        "seed": 20260801,
        "data": {
            "train_split": "train", "validation_split": "validation",
            "train_samples": 20_000, "validation_samples": 2_000,
            "forbidden_splits": FORBIDDEN_SPLITS,
        },
        "model": {
            "architecture": "standard_unet",
            "encoder_channels": [64, 128, 256, 512],
            "bottleneck_channels": 1024,
            "decoder_channels": [512, 256, 128, 64],
            "output_channels": 4, "output_order": TARGET_ORDER,
            "normalization": "BatchNorm", "activation": "LeakyReLU",
            "final_activation": "Sigmoid",
        },
        "numerical_precision": NUMERICAL_POLICY,
        "loss": {"type": "masked_smooth_l1", "beta": 0.1, "channel_weights": [0.25] * 4},
        "optimizer": {"type": "AdamW", "lr": 0.001, "weight_decay": 0.0001},
        "scheduler": {"type": "CosineAnnealingLR", "T_max": 30, "eta_min": 0.000001, "step_unit": "epoch"},
        "training": {
            "batch_size": 8, "max_epochs": 30,
            "gradient_accumulation": False, "gradient_clipping": False,
        },
        "early_stopping": {
            "enabled": True, "monitor": "val_M_MAE_plus_val_H_MAE",
            "mode": "min", "patience": 5, "min_delta": 0.0,
        },
        "checkpoint": {
            "selection_metric": "val_M_MAE_plus_val_H_MAE",
            "save_best": True, "save_last": True,
            "milestone_epochs": [5, 10, 15, 20, 25, 30],
        },
        "dataloader": {
            "num_workers": 2, "pin_memory": True,
            "persistent_workers": True, "prefetch_factor": 2,
        },
        "augmentation": {"enabled": False},
        "amp_overflow": {
            "recoverable_gradient_overflow_allowed": True,
            "fail_on_forward_nonfinite": True,
            "fail_on_loss_nonfinite": True,
            "fail_on_state_contamination": True,
            "max_consecutive_overflows": 5,
            "max_overflows_per_100_batches": 10,
        },
    }
    failed = [key for key, expected in exact.items() if config.get(key) != expected]
    if failed:
        raise ValueError(f"Frozen SO-1D v1.1 config mismatch: {failed}")


def resume_signature(config: dict[str, Any]) -> str:
    validate_config(config)
    return canonical_json_hash(config)


def build_identity(base_identity: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    return {
        **base_identity,
        "formal_resume_signature": resume_signature(config),
        "protocol_version": PROTOCOL_VERSION,
        "numerical_precision_policy": NUMERICAL_POLICY,
    }


class OverflowFrequencyGate:
    def __init__(self, *, max_consecutive: int = 5, max_per_100: int = 10) -> None:
        self.max_consecutive = int(max_consecutive)
        self.max_per_100 = int(max_per_100)
        self.consecutive = 0
        self.window: deque[bool] = deque(maxlen=100)
        self.maximum_consecutive = 0
        self.maximum_per_100 = 0

    def update(self, overflow: bool) -> bool:
        self.consecutive = self.consecutive + 1 if overflow else 0
        self.window.append(bool(overflow))
        self.maximum_consecutive = max(self.maximum_consecutive, self.consecutive)
        self.maximum_per_100 = max(self.maximum_per_100, sum(self.window))
        return (
            self.consecutive >= self.max_consecutive
            or (len(self.window) == 100 and sum(self.window) >= self.max_per_100)
        )


def audit_resume(
    checkpoint_path: Path, *, expected_path: Path, config_sha256: str,
    identity: dict[str, Any], history_path: Path,
) -> dict[str, Any]:
    if checkpoint_path.resolve() != expected_path.resolve():
        raise RuntimeError("REFUSE_RESUME: only formal_train_v1_1/checkpoints/last.pt is allowed")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if checkpoint.get("config_sha256") != config_sha256:
        raise RuntimeError("REFUSE_RESUME: config SHA256 mismatch")
    for key, expected in identity.items():
        if checkpoint.get(key) != expected:
            raise RuntimeError(f"REFUSE_RESUME: {key} mismatch")
    if checkpoint.get("protocol_version") != PROTOCOL_VERSION:
        raise RuntimeError("REFUSE_RESUME: protocol mismatch")
    if count_parameters(SO1UNetDecomposer(numerical_precision=NUMERICAL_POLICY))["parameter_count"] != 31_037_828:
        raise RuntimeError("REFUSE_RESUME: parameter count mismatch")
    with history_path.open("r", encoding="utf-8-sig", newline="") as handle:
        epochs = [int(float(row["epoch"])) for row in csv.DictReader(handle)]
    if epochs != list(range(1, int(checkpoint["epoch"]) + 1)):
        raise RuntimeError("REFUSE_RESUME: history discontinuity")
    return {"status": "PASS", "resume_epoch": checkpoint["epoch"], "history_epochs": epochs}
