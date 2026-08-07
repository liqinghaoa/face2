"""Factory for the unified P1 component experiment models."""

from __future__ import annotations

from collections import OrderedDict
from typing import Any

import torch
from torch import nn
from torchvision.models import ResNet18_Weights, resnet18

from models.p1_light_linear_probe import P1LightLinearProbe, build_light_linear_probe
from models.p1_shared_dual_resnet18 import SharedResNet18DualBranch, count_parameters


def _resnet18_backbone(pretrained: bool = True) -> nn.Module:
    return resnet18(weights=ResNet18_Weights.IMAGENET1K_V1 if pretrained else None)


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    if isinstance(value, (int, float)):
        return bool(value)
    return str(value).strip().lower() in {"1", "true", "yes", "y", "imagenet", "default"}


def _ensure_two_class_head(model: nn.Module, out_features: int = 2) -> nn.Module:
    if not hasattr(model, "fc") or not isinstance(model.fc, nn.Linear):
        raise ValueError("model does not expose an nn.Linear fc head")
    model.fc = nn.Linear(model.fc.in_features, out_features)
    return model


def build_single_resnet18_component_model(
    *,
    pretrained: bool = True,
    num_classes: int = 2,
) -> nn.Module:
    model = _ensure_two_class_head(_resnet18_backbone(pretrained=pretrained), num_classes)
    for parameter in model.parameters():
        parameter.requires_grad = True
    return model


def build_p1_component_model(model_type: str, config: dict[str, Any] | None = None) -> nn.Module:
    config = dict(config or {})
    normalized = str(model_type).strip().lower()
    pretrained = _as_bool(config.get("pretrained", True))
    num_classes = int(config.get("num_classes", 2))

    if normalized in {"resnet18_single", "single_resnet18", "resnet18"}:
        return build_single_resnet18_component_model(pretrained=pretrained, num_classes=num_classes)

    if normalized in {"linear_probe", "light_linear_probe"}:
        return build_light_linear_probe(num_classes=num_classes)

    if normalized in {"shared_resnet18_dual_branch", "shared_dual_resnet18", "shared_dual"}:
        return SharedResNet18DualBranch(pretrained=pretrained, num_classes=num_classes)

    raise ValueError(f"unsupported P1 component model_type: {model_type!r}")


def count_component_parameters(model: nn.Module) -> dict[str, int]:
    return count_parameters(model)


def summarize_component_model_contract() -> dict[str, Any]:
    single = build_single_resnet18_component_model(pretrained=False)
    probe = build_light_linear_probe()
    dual = SharedResNet18DualBranch(pretrained=False)
    return {
        "single_resnet18": count_parameters(single),
        "light_probe": count_parameters(probe),
        "shared_dual": count_parameters(dual),
        "shared_dual_audit": count_parameters(dual),
    }
