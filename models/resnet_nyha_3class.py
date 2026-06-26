"""ImageNet-pretrained ResNet classifiers for NYHA three-class prediction."""

from __future__ import annotations

from torch import nn
from torchvision import models


def _build_torchvision_resnet(backbone: str, pretrained: bool) -> nn.Module:
    constructors = {
        "resnet18": (models.resnet18, "ResNet18_Weights"),
        "resnet34": (models.resnet34, "ResNet34_Weights"),
        "resnet50": (models.resnet50, "ResNet50_Weights"),
    }
    if backbone not in constructors:
        raise ValueError(
            f"Unsupported backbone {backbone!r}; choose from {sorted(constructors)}"
        )

    constructor, weights_name = constructors[backbone]
    weights_enum = getattr(models, weights_name, None)
    if weights_enum is not None:
        weights = weights_enum.IMAGENET1K_V1 if pretrained else None
        return constructor(weights=weights)

    # Compatibility with older torchvision releases.
    return constructor(pretrained=pretrained)


def build_resnet_nyha_model(
    backbone: str = "resnet18",
    num_classes: int = 3,
    pretrained: bool = True,
) -> nn.Module:
    """Build a fully trainable ResNet that returns unnormalized logits."""
    model = _build_torchvision_resnet(backbone.lower(), pretrained)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    for parameter in model.parameters():
        parameter.requires_grad = True
    return model
