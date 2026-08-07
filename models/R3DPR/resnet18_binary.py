"""ImageNet-pretrained ResNet18 binary classifier for R3DPR."""

from __future__ import annotations

import torch
from torch import nn
from torchvision.models import ResNet18_Weights, resnet18


def build_r3dpr_resnet18_binary(
    pretrained: bool | str = "imagenet",
    dropout: float | None = None,
) -> nn.Module:
    """Build ResNet18 with logits ordered as [Control, Patient]."""
    enabled = str(pretrained).strip().lower() in {"imagenet", "default", "true", "1", "yes"}
    disabled = str(pretrained).strip().lower() in {"none", "false", "0", "no", "random"}
    if not enabled and not disabled:
        raise ValueError(f"Unsupported pretrained setting: {pretrained!r}")
    model = resnet18(weights=ResNet18_Weights.DEFAULT if enabled else None)
    linear = nn.Linear(model.fc.in_features, 2)
    if dropout is None:
        model.fc = linear
    else:
        probability = float(dropout)
        if not 0.0 <= probability < 1.0:
            raise ValueError(f"dropout must be in [0, 1), got {dropout}")
        model.fc = nn.Sequential(nn.Dropout(p=probability), linear)
    return model


def count_parameters(model: nn.Module) -> dict[str, int]:
    return {
        "total_params": int(sum(parameter.numel() for parameter in model.parameters())),
        "trainable_params": int(sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)),
    }
