"""Reusable finite-value and replay helpers for SO-1D-R1 diagnostics."""

from __future__ import annotations

import csv
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Iterable

import torch
from torch import nn


MODULE_GROUPS = {
    "inc": "encoder1",
    "down1": "encoder2",
    "down2": "encoder3",
    "down3": "encoder4",
    "down4": "bottleneck",
    "up1": "decoder4",
    "up2": "decoder3",
    "up3": "decoder2",
    "up4": "decoder1",
    "outc": "output_head",
}


def module_group(name: str) -> str:
    root = name.split(".", 1)[0]
    return MODULE_GROUPS.get(root, root or "model")


def tensor_health(value: torch.Tensor) -> dict[str, Any]:
    tensor = value.detach().float()
    finite = torch.isfinite(tensor)
    finite_values = tensor[finite]
    result: dict[str, Any] = {
        "finite": bool(finite.all().item()),
        "num_nan": int(torch.isnan(tensor).sum().item()),
        "num_inf": int(torch.isinf(tensor).sum().item()),
        "dtype": str(value.dtype),
        "numel": int(tensor.numel()),
    }
    if finite_values.numel():
        result.update({
            "min": float(finite_values.min().cpu()),
            "max": float(finite_values.max().cpu()),
            "mean": float(finite_values.mean().cpu()),
            "std": float(finite_values.std(unbiased=False).cpu()),
            "max_abs": float(finite_values.abs().max().cpu()),
        })
    else:
        result.update({key: None for key in ("min", "max", "mean", "std", "max_abs")})
    return result


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = fieldnames or list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def atomic_torch_save(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def model_parameter_health(model: nn.Module) -> list[dict[str, Any]]:
    rows = []
    for name, parameter in model.named_parameters():
        rows.append({"name": name, "module_group": module_group(name), **tensor_health(parameter)})
    return rows


def batchnorm_health(model: nn.Module) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for name, module in model.named_modules():
        if not isinstance(module, nn.BatchNorm2d):
            continue
        for tensor_name in ("weight", "bias", "running_mean", "running_var", "num_batches_tracked"):
            value = getattr(module, tensor_name)
            health = tensor_health(value)
            rows.append({
                "module": name,
                "module_group": module_group(name),
                "tensor": tensor_name,
                "nonnegative": bool((value >= 0).all().item()) if tensor_name == "running_var" else None,
                **health,
            })
    return rows


def optimizer_state_health(
    optimizer: torch.optim.Optimizer, model: nn.Module
) -> list[dict[str, Any]]:
    names = {id(parameter): name for name, parameter in model.named_parameters()}
    rows: list[dict[str, Any]] = []
    for group in optimizer.param_groups:
        for parameter in group["params"]:
            state = optimizer.state.get(parameter, {})
            for state_name in ("exp_avg", "exp_avg_sq"):
                if state_name in state:
                    rows.append({
                        "parameter": names.get(id(parameter), "UNKNOWN"),
                        "module_group": module_group(names.get(id(parameter), "UNKNOWN")),
                        "state": state_name,
                        **tensor_health(state[state_name]),
                    })
    return rows


def all_rows_finite(rows: Iterable[dict[str, Any]]) -> bool:
    return all(bool(row["finite"]) for row in rows)


def epoch7_order_from_generator_state(
    *, sample_count: int, batch_size: int, generator_state: torch.Tensor
) -> list[int]:
    """Reproduce persistent-worker DataLoader reset and RandomSampler randperm."""
    generator = torch.Generator()
    generator.set_state(generator_state.detach().cpu())
    return torch.randperm(sample_count, generator=generator).tolist()


def sample_order_sha256(rows: list[dict[str, Any]]) -> str:
    fields = (
        "batch_index", "position_in_batch", "split_index", "sample_id",
        "base_latent_id", "acquisition_variant_id", "camera", "light", "camera_light_pair",
    )
    text = "\n".join("|".join(str(row[field]) for field in fields) for row in rows) + "\n"
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def gradient_health(model: nn.Module) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    squared_norm = 0.0
    global_max = 0.0
    for name, parameter in model.named_parameters():
        if parameter.grad is None:
            continue
        gradient = parameter.grad.detach().float()
        finite = torch.isfinite(gradient)
        norm = float(gradient.norm().cpu())
        health = {
            "finite": bool(finite.all().item()),
            "num_nan": int(torch.isnan(gradient).sum().item()),
            "num_inf": int(torch.isinf(gradient).sum().item()),
            "max_abs": float(gradient[finite].abs().max().cpu()) if bool(finite.any()) else None,
            "dtype": str(parameter.grad.dtype),
            "numel": int(gradient.numel()),
        }
        rows.append({"name": name, "module_group": module_group(name), "grad_norm": norm, **health})
        squared_norm += norm * norm
        if health["max_abs"] is not None:
            global_max = max(global_max, float(health["max_abs"]))
    summary = {
        "finite": all_rows_finite(rows),
        "global_grad_norm": math.sqrt(squared_norm),
        "global_max_abs_grad": global_max,
        "first_nonfinite_parameter": next((row["name"] for row in rows if not row["finite"]), None),
        "first_nonfinite_module": next((row["module_group"] for row in rows if not row["finite"]), None),
    }
    return rows, summary


def state_finite(model: nn.Module, optimizer: torch.optim.Optimizer) -> dict[str, Any]:
    first_parameter = next(
        (name for name, value in model.named_parameters() if not bool(torch.isfinite(value).all())), None
    )
    first_optimizer = None
    parameter_names = {id(parameter): name for name, parameter in model.named_parameters()}
    for parameter, state in optimizer.state.items():
        for state_name in ("exp_avg", "exp_avg_sq"):
            value = state.get(state_name)
            if value is not None and not bool(torch.isfinite(value).all()):
                first_optimizer = f"{parameter_names.get(id(parameter), 'UNKNOWN')}:{state_name}"
                break
        if first_optimizer:
            break
    first_bn = None
    for name, module in model.named_modules():
        if isinstance(module, nn.BatchNorm2d):
            for tensor_name in ("running_mean", "running_var"):
                if not bool(torch.isfinite(getattr(module, tensor_name)).all()):
                    first_bn = f"{name}:{tensor_name}"
                    break
        if first_bn:
            break
    return {
        "parameters_finite": first_parameter is None,
        "optimizer_state_finite": first_optimizer is None,
        "batchnorm_buffers_finite": first_bn is None,
        "first_nonfinite_parameter": first_parameter,
        "first_nonfinite_optimizer_state": first_optimizer,
        "first_nonfinite_batchnorm": first_bn,
    }


def parameter_digest(model: nn.Module) -> str:
    digest = hashlib.sha256()
    for name, parameter in model.named_parameters():
        digest.update(name.encode("utf-8"))
        digest.update(parameter.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()
