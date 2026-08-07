from __future__ import annotations

import torch
from torch import nn
from torchvision.models import ResNet18_Weights, resnet18


def _resnet18_encoder(*, pretrained: bool = True) -> tuple[nn.Module, int]:
    model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1 if pretrained else None)
    feature_dim = int(model.fc.in_features)
    model.fc = nn.Identity()
    return model, feature_dim


class IndependentDualResNet18(nn.Module):
    """Independent two-encoder ResNet18 fusion model for P1 extension stage B."""

    def __init__(self, *, pretrained: bool = True, num_classes: int = 2) -> None:
        super().__init__()
        if int(num_classes) != 2:
            raise ValueError("P1 extension stage B is fixed to binary classification")
        self.encoder_primary, feature_dim_1 = _resnet18_encoder(pretrained=pretrained)
        self.encoder_secondary, feature_dim_2 = _resnet18_encoder(pretrained=pretrained)
        if feature_dim_1 != 512 or feature_dim_2 != 512:
            raise ValueError("ResNet18 encoders must produce 512-dimensional features")
        self.classifier = nn.Linear(feature_dim_1 + feature_dim_2, int(num_classes))
        self.feature_dim = feature_dim_1
        self.fused_dim = feature_dim_1 + feature_dim_2

    def forward(self, input_1: torch.Tensor | dict[str, torch.Tensor], input_2: torch.Tensor | None = None) -> torch.Tensor:
        if isinstance(input_1, dict):
            input_2 = input_1.get("secondary", input_2)
            input_1 = input_1.get("primary")
        if input_1 is None or input_2 is None:
            raise ValueError("IndependentDualResNet18 requires primary and secondary inputs")
        feature_1 = self.encoder_primary(input_1)
        feature_2 = self.encoder_secondary(input_2)
        return self.classifier(torch.cat([feature_1, feature_2], dim=1))


def count_parameters(model: nn.Module) -> dict[str, int]:
    total = sum(parameter.numel() for parameter in model.parameters())
    trainable = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    return {"total_params": int(total), "trainable_params": int(trainable)}

