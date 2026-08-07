"""Checkpoint helpers for SO-1 decomposition training."""

from __future__ import annotations

import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn


IDENTITY_FIELDS = (
    "target_order",
    "dataset_contract_hash",
    "generator_config_hash",
    "SO0_version",
    "SO0_config_hash",
    "model_architecture",
)


def capture_rng_state() -> dict[str, Any]:
    state: dict[str, Any] = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["torch_cuda"] = torch.cuda.get_rng_state_all()
    return state


def restore_rng_state(state: dict[str, Any]) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch_cpu"])
    if torch.cuda.is_available() and state.get("torch_cuda") is not None:
        torch.cuda.set_rng_state_all(state["torch_cuda"])


def validate_checkpoint_identity(checkpoint: dict[str, Any], expected: dict[str, Any]) -> None:
    for field in IDENTITY_FIELDS:
        if checkpoint.get(field) != expected.get(field):
            raise ValueError(
                f"Checkpoint identity mismatch for {field}: "
                f"checkpoint={checkpoint.get(field)!r}, expected={expected.get(field)!r}"
            )


def save_checkpoint(
    path: str | Path,
    *,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    grad_scaler: Any,
    epoch: int,
    global_step: int,
    best_selection_metric: float,
    early_stopping_counter: int,
    identity: dict[str, Any],
    resolved_config: dict[str, Any],
    dataset_contract: dict[str, Any],
    dataset_fingerprint: dict[str, Any],
    seed: int,
) -> None:
    payload = {
        **identity,
        "epoch": int(epoch),
        "global_step": int(global_step),
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
        "grad_scaler_state_dict": grad_scaler.state_dict() if grad_scaler is not None else None,
        "best_selection_metric": float(best_selection_metric),
        "early_stopping_counter": int(early_stopping_counter),
        "resolved_config": resolved_config,
        "dataset_contract": dataset_contract,
        "dataset_fingerprint": dataset_fingerprint,
        "seed": int(seed),
        "rng_state": capture_rng_state(),
    }
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)


def load_checkpoint(
    path: str | Path,
    *,
    model: nn.Module,
    optimizer: torch.optim.Optimizer | None,
    scheduler: Any,
    grad_scaler: Any,
    expected_identity: dict[str, Any],
    map_location: str | torch.device,
    restore_rng: bool = True,
) -> dict[str, Any]:
    try:
        checkpoint = torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        checkpoint = torch.load(path, map_location=map_location)
    validate_checkpoint_identity(checkpoint, expected_identity)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    if scheduler is not None and checkpoint.get("scheduler_state_dict") is not None:
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
    if grad_scaler is not None and checkpoint.get("grad_scaler_state_dict") is not None:
        grad_scaler.load_state_dict(checkpoint["grad_scaler_state_dict"])
    if restore_rng:
        restore_rng_state(checkpoint["rng_state"])
    return checkpoint
