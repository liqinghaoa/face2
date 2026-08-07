"""Locked ResNet18 trainability strategies for the overfitting-control experiment."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch
from torch import nn


TRAINABILITY_STRATEGIES = ("frozen_backbone", "partial_layer4")
FROZEN_BACKBONE_MODULES = (
    "backbone.conv1",
    "backbone.bn1",
    "backbone.layer1",
    "backbone.layer2",
    "backbone.layer3",
    "backbone.layer4",
)
PARTIAL_FROZEN_MODULES = (
    "backbone.conv1",
    "backbone.bn1",
    "backbone.layer1",
    "backbone.layer2",
    "backbone.layer3",
)


def validate_trainability_strategy(strategy: str) -> str:
    value = str(strategy).strip().lower()
    if value not in TRAINABILITY_STRATEGIES:
        raise ValueError(
            f"strategy must be one of {TRAINABILITY_STRATEGIES}, got {strategy!r}"
        )
    return value


def _require_model_layout(model: nn.Module) -> None:
    if not hasattr(model, "backbone") or not hasattr(model, "classifier"):
        raise TypeError("Model must expose .backbone and .classifier modules")
    backbone = model.backbone
    for name in ("conv1", "bn1", "layer1", "layer2", "layer3", "layer4"):
        if not hasattr(backbone, name):
            raise TypeError(f"ResNet18 backbone is missing module {name!r}")


def configure_trainability_strategy(model: nn.Module, strategy: str) -> None:
    """Set the exact parameter-level contract for one locked strategy."""
    strategy = validate_trainability_strategy(strategy)
    _require_model_layout(model)
    for parameter in model.parameters():
        parameter.requires_grad = False
    for parameter in model.classifier.parameters():
        parameter.requires_grad = True
    if strategy == "partial_layer4":
        for parameter in model.backbone.layer4.parameters():
            parameter.requires_grad = True


def enforce_trainability_modes(
    model: nn.Module, strategy: str, *, training: bool
) -> None:
    """Re-apply module modes immediately after every top-level ``model.train`` call."""
    strategy = validate_trainability_strategy(strategy)
    _require_model_layout(model)
    if not training:
        model.eval()
        return
    model.train()
    if strategy == "frozen_backbone":
        model.backbone.eval()
    else:
        model.backbone.conv1.eval()
        model.backbone.bn1.eval()
        model.backbone.layer1.eval()
        model.backbone.layer2.eval()
        model.backbone.layer3.eval()
        model.backbone.layer4.train()
    model.classifier.train()


def expected_trainable_parameter_names(model: nn.Module, strategy: str) -> list[str]:
    strategy = validate_trainability_strategy(strategy)
    _require_model_layout(model)
    prefixes = ("classifier.",)
    if strategy == "partial_layer4":
        prefixes = ("backbone.layer4.", "classifier.")
    return [name for name, _ in model.named_parameters() if name.startswith(prefixes)]


def build_optimizer(
    model: nn.Module,
    strategy: str,
    *,
    classifier_lr: float = 1.0e-4,
    layer4_lr: float = 1.0e-5,
    weight_decay: float = 1.0e-4,
) -> torch.optim.AdamW:
    """Build AdamW with the exact, named parameter groups required by the protocol."""
    strategy = validate_trainability_strategy(strategy)
    assert_trainability_contract(model, strategy)
    groups: list[dict[str, Any]] = []
    if strategy == "partial_layer4":
        groups.append(
            {
                "name": "layer4",
                "params": list(model.backbone.layer4.parameters()),
                "lr": float(layer4_lr),
                "weight_decay": float(weight_decay),
            }
        )
    groups.append(
        {
            "name": "classifier",
            "params": list(model.classifier.parameters()),
            "lr": float(classifier_lr),
            "weight_decay": float(weight_decay),
        }
    )
    optimizer = torch.optim.AdamW(groups)
    assert_optimizer_contract(
        model,
        optimizer,
        strategy,
        classifier_lr=classifier_lr,
        layer4_lr=layer4_lr,
        weight_decay=weight_decay,
    )
    return optimizer


def _parameter_name_by_id(model: nn.Module) -> dict[int, str]:
    return {id(parameter): name for name, parameter in model.named_parameters()}


def optimizer_group_records(
    model: nn.Module, optimizer: torch.optim.Optimizer
) -> list[dict[str, Any]]:
    name_by_id = _parameter_name_by_id(model)
    records: list[dict[str, Any]] = []
    for index, group in enumerate(optimizer.param_groups):
        names = [name_by_id.get(id(parameter)) for parameter in group["params"]]
        if any(name is None for name in names):
            raise ValueError("Optimizer contains a parameter that does not belong to the model")
        records.append(
            {
                "index": index,
                "name": str(group.get("name", f"group_{index}")),
                "learning_rate": float(group["lr"]),
                "weight_decay": float(group["weight_decay"]),
                "parameter_names": [str(name) for name in names],
                "parameter_count": int(sum(parameter.numel() for parameter in group["params"])),
                "tensor_count": len(group["params"]),
            }
        )
    return records


def assert_trainability_contract(model: nn.Module, strategy: str) -> None:
    strategy = validate_trainability_strategy(strategy)
    expected = set(expected_trainable_parameter_names(model, strategy))
    actual = {name for name, parameter in model.named_parameters() if parameter.requires_grad}
    if actual != expected:
        raise AssertionError(
            "Trainable parameter set violates the locked strategy; "
            f"missing={sorted(expected - actual)}, unexpected={sorted(actual - expected)}"
        )


def assert_optimizer_contract(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    strategy: str,
    *,
    classifier_lr: float = 1.0e-4,
    layer4_lr: float = 1.0e-5,
    weight_decay: float = 1.0e-4,
) -> None:
    strategy = validate_trainability_strategy(strategy)
    records = optimizer_group_records(model, optimizer)
    expected_group_names = (
        ["classifier"] if strategy == "frozen_backbone" else ["layer4", "classifier"]
    )
    if [record["name"] for record in records] != expected_group_names:
        raise AssertionError("Optimizer group names/order violate the locked protocol")
    flattened = [name for record in records for name in record["parameter_names"]]
    if len(flattened) != len(set(flattened)):
        raise AssertionError("A parameter appears in more than one optimizer group")
    expected = set(expected_trainable_parameter_names(model, strategy))
    if set(flattened) != expected:
        raise AssertionError("Optimizer parameters do not exactly equal trainable parameters")
    for record in records:
        expected_lr = layer4_lr if record["name"] == "layer4" else classifier_lr
        if float(record["learning_rate"]) != float(expected_lr):
            raise AssertionError(f"Unexpected learning rate in {record['name']} group")
        if float(record["weight_decay"]) != float(weight_decay):
            raise AssertionError(f"Unexpected weight decay in {record['name']} group")


def _module_training_records(model: nn.Module) -> dict[str, bool]:
    selected = {
        "model": model,
        "backbone": model.backbone,
        "backbone.conv1": model.backbone.conv1,
        "backbone.bn1": model.backbone.bn1,
        "backbone.layer1": model.backbone.layer1,
        "backbone.layer2": model.backbone.layer2,
        "backbone.layer3": model.backbone.layer3,
        "backbone.layer4": model.backbone.layer4,
        "classifier": model.classifier,
    }
    return {name: bool(module.training) for name, module in selected.items()}


def batchnorm_training_records(model: nn.Module) -> dict[str, bool]:
    return {
        name: bool(module.training)
        for name, module in model.named_modules()
        if isinstance(module, nn.modules.batchnorm._BatchNorm)
    }


def assert_training_mode_contract(model: nn.Module, strategy: str) -> None:
    strategy = validate_trainability_strategy(strategy)
    records = _module_training_records(model)
    if not records["model"] or not records["classifier"]:
        raise AssertionError("Top-level model and classifier must be in training mode")
    if strategy == "frozen_backbone":
        if records["backbone"] or any(batchnorm_training_records(model).values()):
            raise AssertionError("Frozen backbone and every backbone BatchNorm must stay in eval mode")
    else:
        for name in PARTIAL_FROZEN_MODULES:
            if records[name]:
                raise AssertionError(f"Frozen module {name} must stay in eval mode")
        if not records["backbone.layer4"]:
            raise AssertionError("layer4 must be in training mode")
        bn_modes = batchnorm_training_records(model)
        for name, training in bn_modes.items():
            expected = name.startswith("backbone.layer4.")
            if training != expected:
                raise AssertionError(f"BatchNorm mode mismatch for {name}")


def build_trainability_audit(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    strategy: str,
    *,
    variant: str,
    fold: int,
) -> dict[str, Any]:
    """Return a JSON-ready, exhaustive audit after enforcing the training mode."""
    strategy = validate_trainability_strategy(strategy)
    assert_trainability_contract(model, strategy)
    assert_optimizer_contract(model, optimizer, strategy)
    assert_training_mode_contract(model, strategy)
    trainable = [name for name, parameter in model.named_parameters() if parameter.requires_grad]
    frozen = [name for name, parameter in model.named_parameters() if not parameter.requires_grad]
    named = dict(model.named_parameters())
    return {
        "schema_version": "resnet18_overfit_trainability_audit_v1",
        "strategy": strategy,
        "variant": str(variant),
        "fold": int(fold),
        "trainable_parameter_names": trainable,
        "frozen_parameter_names": frozen,
        "trainable_parameter_count": int(sum(named[name].numel() for name in trainable)),
        "frozen_parameter_count": int(sum(named[name].numel() for name in frozen)),
        "total_parameter_count": int(sum(parameter.numel() for parameter in model.parameters())),
        "optimizer_groups": optimizer_group_records(model, optimizer),
        "module_training_modes": _module_training_records(model),
        "batchnorm_training_modes": batchnorm_training_records(model),
        "frozen_module_names": list(
            FROZEN_BACKBONE_MODULES
            if strategy == "frozen_backbone"
            else PARTIAL_FROZEN_MODULES
        ),
        "trainable_module_names": (
            ["classifier"]
            if strategy == "frozen_backbone"
            else ["backbone.layer4", "classifier"]
        ),
    }


def validate_audit_against_checkpoint(
    audit: Mapping[str, Any], checkpoint: Mapping[str, Any]
) -> None:
    for key in (
        "strategy",
        "trainable_parameter_names",
        "frozen_parameter_names",
        "trainable_parameter_count",
        "frozen_parameter_count",
        "optimizer_groups",
        "batchnorm_training_modes",
    ):
        checkpoint_key = f"trainability_{key}" if key != "strategy" else key
        if checkpoint.get(checkpoint_key) != audit.get(key):
            raise ValueError(f"Checkpoint trainability field does not match audit: {checkpoint_key}")
