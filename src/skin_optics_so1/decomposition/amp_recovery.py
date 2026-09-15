"""Small, testable helpers for AMP overflow recovery diagnostics."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import torch
from torch import nn


RECOVERY_HISTORY_FIELDS = [
    "batch_index", "global_step", "scale_before", "scale_after",
    "forward_finite", "loss_finite", "gradient_finite", "overflow_detected",
    "optimizer_step_skipped", "optimizer_step_executed", "parameters_finite",
    "BN_finite", "Adam_finite",
]


def _update_tensor(digest: Any, value: torch.Tensor) -> None:
    tensor = value.detach().cpu().contiguous()
    digest.update(str(tensor.dtype).encode("ascii"))
    digest.update(str(tuple(tensor.shape)).encode("ascii"))
    digest.update(tensor.numpy().tobytes())


def parameter_state_sha256(model: nn.Module) -> str:
    digest = hashlib.sha256()
    for name, parameter in model.named_parameters():
        digest.update(name.encode("utf-8"))
        _update_tensor(digest, parameter)
    return digest.hexdigest()


def adam_state_sha256(
    optimizer: torch.optim.Optimizer, model: nn.Module
) -> str:
    digest = hashlib.sha256()
    names = {id(parameter): name for name, parameter in model.named_parameters()}
    for parameter, state in sorted(
        optimizer.state.items(), key=lambda item: names.get(id(item[0]), "")
    ):
        digest.update(names.get(id(parameter), "UNKNOWN").encode("utf-8"))
        for key in sorted(state):
            digest.update(str(key).encode("utf-8"))
            value = state[key]
            if torch.is_tensor(value):
                _update_tensor(digest, value)
            else:
                digest.update(repr(value).encode("utf-8"))
    return digest.hexdigest()


def optimizer_group_sha256(optimizer: torch.optim.Optimizer) -> str:
    groups = []
    for group in optimizer.param_groups:
        groups.append({key: value for key, value in group.items() if key != "params"})
    payload = json.dumps(groups, sort_keys=True, default=str, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def scaler_step_and_update(scaler: Any, optimizer: torch.optim.Optimizer) -> dict[str, Any]:
    """Invoke the standard AMP chain while observing whether optimizer.step ran."""
    scale_before = float(scaler.get_scale())
    original_step = optimizer.step
    optimizer_step_calls = 0

    def observed_step(*args: Any, **kwargs: Any) -> Any:
        nonlocal optimizer_step_calls
        optimizer_step_calls += 1
        return original_step(*args, **kwargs)

    optimizer.step = observed_step  # type: ignore[method-assign]
    try:
        scaler.step(optimizer)
    finally:
        optimizer.step = original_step  # type: ignore[method-assign]
    scaler.update()
    return {
        "scaler_step_called": True,
        "scaler_update_called": True,
        "optimizer_step_calls": optimizer_step_calls,
        "optimizer_step_executed": optimizer_step_calls > 0,
        "optimizer_step_skipped": optimizer_step_calls == 0,
        "scale_before": scale_before,
        "scale_after": float(scaler.get_scale()),
    }


def grad_scaler_found_inf(scaler: Any, optimizer: torch.optim.Optimizer) -> bool:
    """Read GradScaler's authoritative non-finite flag after ``unscale_``.

    PyTorch's ``unscale_`` already checks every gradient with the fused AMP
    kernel. Reusing that result avoids a second per-parameter GPU sync on every
    finite training batch.
    """
    optimizer_state = scaler._per_optimizer_states.get(id(optimizer))
    if not optimizer_state:
        raise RuntimeError("GradScaler has no optimizer state; call unscale_ first")
    per_device = optimizer_state.get("found_inf_per_device")
    if not per_device:
        raise RuntimeError("GradScaler found_inf state is unavailable after unscale_")
    return bool(torch.stack(tuple(per_device.values())).sum().item())


def adjudicate_recovery_window(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if len(rows) != 100:
        return {
            "completed": len(rows),
            "recoverable_overflow_events": sum(bool(row["overflow_detected"]) for row in rows),
            "last_50_overflow_free": False,
            "AMP_RECOVERY_STABLE": False,
        }
    overflow_count = sum(bool(row["overflow_detected"]) for row in rows)
    last_50_clear = not any(bool(row["overflow_detected"]) for row in rows[-50:])
    return {
        "completed": 100,
        "recoverable_overflow_events": overflow_count,
        "last_50_overflow_free": last_50_clear,
        "AMP_RECOVERY_STABLE": overflow_count <= 2 and last_50_clear,
    }


def diagnostic_path_isolated(diagnostic: Path, formal_checkpoints: Path) -> bool:
    diagnostic = diagnostic.resolve()
    formal_checkpoints = formal_checkpoints.resolve()
    return (
        formal_checkpoints not in diagnostic.parents
        and diagnostic not in formal_checkpoints.parents
        and diagnostic != formal_checkpoints
    )
