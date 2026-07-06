"""Torchvision backbone factory for NYHA three-class image classification."""

from __future__ import annotations

from collections import OrderedDict

import torch
from torch import nn
from torchvision import models

from models.resnet_nyha_3class import build_resnet_nyha_model


_RESNET_BACKBONES = ("resnet18", "resnet34", "resnet50")
_NON_RESNET_BACKBONES = (
    "densenet121",
    "efficientnet_b0",
    "convnext_tiny",
    "swin_t",
    "mobilenet_v3_large",
)
_SUPPORTED_BACKBONES = _RESNET_BACKBONES + _NON_RESNET_BACKBONES


def get_supported_backbones() -> list[str]:
    """Return the backbone names accepted by this factory."""

    return list(_SUPPORTED_BACKBONES)


def count_parameters(model: torch.nn.Module) -> dict[str, int]:
    """Count total and trainable parameters for any torch module."""

    total_params = sum(parameter.numel() for parameter in model.parameters())
    trainable_params = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
    return {"total_params": int(total_params), "trainable_params": int(trainable_params)}


def _pretrained_enabled(value: bool | str | None) -> bool:
    if value is None:
        return False
    if isinstance(value, bool):
        return value
    normalized = str(value).strip().lower()
    if normalized in {"imagenet", "default", "true", "yes", "1"}:
        return True
    if normalized in {"none", "false", "no", "0", "random"}:
        return False
    raise ValueError(
        "pretrained must be True/False, None, or one of "
        "{'imagenet', 'default', 'true', 'false', 'none', 'random'}; "
        f"got {value!r}"
    )


def _default_weights(weights_enum_name: str, pretrained: bool | str | None):
    """Resolve torchvision DEFAULT weights, falling back to None when disabled."""

    if not _pretrained_enabled(pretrained):
        return None
    weights_enum = getattr(models, weights_enum_name, None)
    if weights_enum is None:
        raise ValueError(
            f"Current torchvision does not expose {weights_enum_name}; "
            "cannot load ImageNet weights for this backbone."
        )
    return weights_enum.DEFAULT


def _make_linear_head(
    in_features: int,
    num_classes: int,
    dropout: float | None,
) -> nn.Module:
    if dropout is None:
        return nn.Linear(in_features, num_classes)
    dropout_value = float(dropout)
    if not 0.0 <= dropout_value < 1.0:
        raise ValueError(f"dropout must be in [0, 1), got {dropout}")
    return nn.Sequential(
        OrderedDict(
            [
                ("dropout", nn.Dropout(p=dropout_value)),
                ("linear", nn.Linear(in_features, num_classes)),
            ]
        )
    )


def _replace_last_linear(
    module: nn.Module,
    num_classes: int,
    dropout: float | None,
) -> nn.Module:
    """Replace the last nested ``nn.Linear`` under ``module``.

    ConvNeXt classifiers have changed slightly across torchvision versions, so
    this helper searches children in reverse order instead of hard-coding an
    index. It also works for other Sequential-style classifier heads.
    """

    children = list(module.named_children())
    for child_name, child in reversed(children):
        if isinstance(child, nn.Linear):
            setattr(
                module,
                child_name,
                _make_linear_head(child.in_features, num_classes, dropout),
            )
            return module
        if list(child.named_children()):
            try:
                _replace_last_linear(child, num_classes, dropout)
                return module
            except ValueError:
                pass
    raise ValueError(
        f"Could not find an nn.Linear classifier head under {module.__class__.__name__}"
    )


def _final_head(model: nn.Module, backbone: str) -> nn.Module:
    """Return the final trainable classifier/head module for freezing support."""

    if backbone in _RESNET_BACKBONES:
        return model.fc
    if backbone == "densenet121":
        return model.classifier
    if backbone in {"efficientnet_b0", "convnext_tiny", "mobilenet_v3_large"}:
        return model.classifier
    if backbone == "swin_t":
        return model.head
    raise ValueError(f"Unsupported backbone for final head lookup: {backbone!r}")


def _apply_freeze_backbone(model: nn.Module, backbone: str) -> None:
    for parameter in model.parameters():
        parameter.requires_grad = False
    for parameter in _final_head(model, backbone).parameters():
        parameter.requires_grad = True


def build_nyha_classification_model(
    backbone: str,
    num_classes: int = 3,
    pretrained: bool | str | None = True,
    freeze_backbone: bool = False,
    dropout: float | None = None,
) -> torch.nn.Module:
    """Build a torchvision classifier with a NYHA ``num_classes`` output head.

    ResNet18/34/50 are delegated to the existing project builder to preserve
    old experiment behavior. Non-ResNet models replace their official
    torchvision classifier heads as follows:

    - DenseNet121: ``model.classifier``
    - EfficientNet-B0: final linear layer in ``model.classifier``
    - ConvNeXt-Tiny: last nested ``nn.Linear`` in ``model.classifier``
    - Swin-Tiny: ``model.head``
    - MobileNetV3-Large: final linear layer in ``model.classifier``
    """

    normalized = str(backbone).strip().lower()
    if normalized not in _SUPPORTED_BACKBONES:
        raise ValueError(
            f"Unsupported backbone {backbone!r}; choose from {list(_SUPPORTED_BACKBONES)}"
        )

    num_classes = int(num_classes)
    if num_classes < 2:
        raise ValueError(f"num_classes must be at least 2, got {num_classes}")

    if normalized in _RESNET_BACKBONES:
        model = build_resnet_nyha_model(
            backbone=normalized,
            num_classes=num_classes,
            pretrained=_pretrained_enabled(pretrained),
        )
        if dropout is not None:
            model.fc = _make_linear_head(model.fc.in_features, num_classes, dropout)

    elif normalized == "densenet121":
        model = models.densenet121(
            weights=_default_weights("DenseNet121_Weights", pretrained)
        )
        model.classifier = _make_linear_head(
            model.classifier.in_features, num_classes, dropout
        )

    elif normalized == "efficientnet_b0":
        model = models.efficientnet_b0(
            weights=_default_weights("EfficientNet_B0_Weights", pretrained)
        )
        if not hasattr(model, "classifier"):
            raise ValueError("efficientnet_b0 model is missing classifier")
        model.classifier = _replace_last_linear(model.classifier, num_classes, dropout)

    elif normalized == "convnext_tiny":
        model = models.convnext_tiny(
            weights=_default_weights("ConvNeXt_Tiny_Weights", pretrained)
        )
        if not hasattr(model, "classifier"):
            raise ValueError("convnext_tiny model is missing classifier")
        model.classifier = _replace_last_linear(model.classifier, num_classes, dropout)

    elif normalized == "swin_t":
        model = models.swin_t(weights=_default_weights("Swin_T_Weights", pretrained))
        model.head = _make_linear_head(model.head.in_features, num_classes, dropout)

    elif normalized == "mobilenet_v3_large":
        model = models.mobilenet_v3_large(
            weights=_default_weights("MobileNet_V3_Large_Weights", pretrained)
        )
        if not hasattr(model, "classifier"):
            raise ValueError("mobilenet_v3_large model is missing classifier")
        model.classifier = _replace_last_linear(model.classifier, num_classes, dropout)

    else:  # pragma: no cover - guarded above.
        raise ValueError(f"Unsupported backbone {backbone!r}")

    if freeze_backbone:
        _apply_freeze_backbone(model, normalized)
    else:
        for parameter in model.parameters():
            parameter.requires_grad = True

    return model
