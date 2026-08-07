"""Shared ResNet18 anti-overfitting trainability helpers.

The project has two ResNet18 layouts in active use:

- torchvision classifiers with ``conv1``/``layer4``/``fc`` at the top level
- wrapper modules with ``backbone`` and ``classifier``

This module keeps the anti-overfit protocol explicit and auditable for both
layouts without changing the legacy full-finetuning default.
"""

from __future__ import annotations

from typing import Any

import torch
from torch import nn


TRAINABILITY_STRATEGIES = (
    "full_finetune",
    "full_finetune_bn_eval",
    "frozen_backbone",
    "partial_layer4",
)


def normalize_strategy(strategy: str | None) -> str:
    value = str(strategy or "full_finetune").strip().lower()
    if value not in TRAINABILITY_STRATEGIES:
        raise ValueError(f"trainability_strategy must be one of {TRAINABILITY_STRATEGIES}, got {strategy!r}")
    return value


def _layout(model: nn.Module) -> str:
    if hasattr(model, "backbone") and hasattr(model, "classifier"):
        backbone = model.backbone
        if all(hasattr(backbone, name) for name in ("conv1", "bn1", "layer1", "layer2", "layer3", "layer4")):
            return "wrapped"
    if all(hasattr(model, name) for name in ("conv1", "bn1", "layer1", "layer2", "layer3", "layer4", "fc")):
        return "torchvision"
    raise TypeError("Expected a ResNet18-like model with either .backbone/.classifier or top-level .fc layout")


def _backbone(model: nn.Module) -> nn.Module:
    return model.backbone if _layout(model) == "wrapped" else model


def classifier_module(model: nn.Module) -> nn.Module:
    return model.classifier if _layout(model) == "wrapped" else model.fc


def frozen_modules(model: nn.Module, strategy: str | None) -> dict[str, nn.Module]:
    strategy = normalize_strategy(strategy)
    backbone = _backbone(model)
    if strategy in {"full_finetune", "full_finetune_bn_eval"}:
        return {}
    if strategy == "frozen_backbone":
        return {
            "conv1": backbone.conv1,
            "bn1": backbone.bn1,
            "layer1": backbone.layer1,
            "layer2": backbone.layer2,
            "layer3": backbone.layer3,
            "layer4": backbone.layer4,
        }
    return {
        "conv1": backbone.conv1,
        "bn1": backbone.bn1,
        "layer1": backbone.layer1,
        "layer2": backbone.layer2,
        "layer3": backbone.layer3,
    }


def configure_trainability(model: nn.Module, strategy: str | None) -> None:
    strategy = normalize_strategy(strategy)
    backbone = _backbone(model)
    classifier = classifier_module(model)
    if strategy in {"full_finetune", "full_finetune_bn_eval"}:
        for parameter in model.parameters():
            parameter.requires_grad = True
        return
    for parameter in model.parameters():
        parameter.requires_grad = False
    for parameter in classifier.parameters():
        parameter.requires_grad = True
    if strategy == "partial_layer4":
        for parameter in backbone.layer4.parameters():
            parameter.requires_grad = True


def apply_train_mode(model: nn.Module, strategy: str | None) -> None:
    strategy = normalize_strategy(strategy)
    backbone = _backbone(model)
    classifier = classifier_module(model)
    if strategy == "full_finetune":
        model.train()
        return
    if strategy == "full_finetune_bn_eval":
        model.train()
        for module in model.modules():
            if isinstance(module, nn.modules.batchnorm._BatchNorm):
                module.eval()
        return
    model.train()
    if strategy == "frozen_backbone":
        backbone.eval()
    else:
        backbone.conv1.eval()
        backbone.bn1.eval()
        backbone.layer1.eval()
        backbone.layer2.eval()
        backbone.layer3.eval()
        backbone.layer4.train()
    classifier.train()


def trainable_named_parameters(model: nn.Module) -> list[tuple[str, nn.Parameter]]:
    return [(name, parameter) for name, parameter in model.named_parameters() if parameter.requires_grad]


def build_optimizer(
    model: nn.Module,
    strategy: str | None,
    *,
    full_lr: float,
    classifier_lr: float | None = None,
    layer4_lr: float | None = None,
    weight_decay: float,
) -> torch.optim.AdamW:
    strategy = normalize_strategy(strategy)
    backbone = _backbone(model)
    classifier = classifier_module(model)
    if strategy in {"full_finetune", "full_finetune_bn_eval"}:
        return torch.optim.AdamW(model.parameters(), lr=float(full_lr), weight_decay=float(weight_decay))
    classifier_lr_value = float(classifier_lr if classifier_lr is not None else full_lr)
    groups: list[dict[str, Any]] = []
    if strategy == "partial_layer4":
        layer4_lr_value = float(layer4_lr if layer4_lr is not None else classifier_lr_value * 0.1)
        groups.append(
            {
                "name": "layer4",
                "params": [parameter for parameter in backbone.layer4.parameters() if parameter.requires_grad],
                "lr": layer4_lr_value,
                "weight_decay": float(weight_decay),
            }
        )
    groups.append(
        {
            "name": "classifier",
            "params": [parameter for parameter in classifier.parameters() if parameter.requires_grad],
            "lr": classifier_lr_value,
            "weight_decay": float(weight_decay),
        }
    )
    return torch.optim.AdamW(groups)


def optimizer_group_summary(model: nn.Module, optimizer: torch.optim.Optimizer) -> list[dict[str, Any]]:
    name_by_id = {id(parameter): name for name, parameter in model.named_parameters()}
    rows: list[dict[str, Any]] = []
    for index, group in enumerate(optimizer.param_groups):
        names = [name_by_id.get(id(parameter), "<unknown>") for parameter in group["params"]]
        rows.append(
            {
                "index": int(index),
                "name": str(group.get("name", f"group_{index}")),
                "learning_rate": float(group["lr"]),
                "weight_decay": float(group["weight_decay"]),
                "tensor_count": int(len(group["params"])),
                "parameter_count": int(sum(parameter.numel() for parameter in group["params"])),
                "parameter_names": [str(name) for name in names],
            }
        )
    return rows


def build_trainability_audit(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    strategy: str | None,
    *,
    fold: int,
    dropout: float | None,
    gradient_clip_max_norm: float | None,
) -> dict[str, Any]:
    strategy = normalize_strategy(strategy)
    trainable = [name for name, parameter in model.named_parameters() if parameter.requires_grad]
    frozen = [name for name, parameter in model.named_parameters() if not parameter.requires_grad]
    named = dict(model.named_parameters())
    modules = frozen_modules(model, strategy)
    batchnorm_training_modes = {
        name: bool(module.training)
        for name, module in model.named_modules()
        if isinstance(module, nn.modules.batchnorm._BatchNorm)
    }
    return {
        "schema_version": "resnet18_anti_overfit_audit_v1",
        "strategy": strategy,
        "fold": int(fold),
        "dropout": None if dropout is None else float(dropout),
        "gradient_clip_max_norm": None if gradient_clip_max_norm is None else float(gradient_clip_max_norm),
        "layout": _layout(model),
        "trainable_parameter_names": trainable,
        "frozen_parameter_names": frozen,
        "trainable_parameter_count": int(sum(named[name].numel() for name in trainable)),
        "frozen_parameter_count": int(sum(named[name].numel() for name in frozen)),
        "total_parameter_count": int(sum(parameter.numel() for parameter in model.parameters())),
        "frozen_module_names": list(modules.keys()),
        "batchnorm_training_modes": batchnorm_training_modes,
        "optimizer_groups": optimizer_group_summary(model, optimizer),
    }
