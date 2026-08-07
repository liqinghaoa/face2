from __future__ import annotations

import torch
from torch import nn
from torchvision.models import ResNet18_Weights, resnet18


def _resnet18_encoder(*, pretrained: bool = True) -> tuple[nn.Module, int]:
    model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1 if pretrained else None)
    feature_dim = int(model.fc.in_features)
    model.fc = nn.Identity()
    return model, feature_dim


class IndependentTripleResNet18(nn.Module):
    """Three independent ResNet18 encoders with direct 1536 -> 2 fusion."""

    def __init__(self, *, pretrained: bool = True, num_classes: int = 2) -> None:
        super().__init__()
        if int(num_classes) != 2:
            raise ValueError("Stage D v1 is fixed to binary Control/Patient classification")
        self.encoder_rgb, rgb_dim = _resnet18_encoder(pretrained=pretrained)
        self.encoder_shading, shading_dim = _resnet18_encoder(pretrained=pretrained)
        self.encoder_residual, residual_dim = _resnet18_encoder(pretrained=pretrained)
        if (rgb_dim, shading_dim, residual_dim) != (512, 512, 512):
            raise ValueError("Each ResNet18 encoder must output 512-dimensional features")
        self.classifier = nn.Linear(rgb_dim + shading_dim + residual_dim, int(num_classes))
        self.feature_dim = 512
        self.fused_dim = 1536

    def forward(
        self,
        rgb: torch.Tensor | dict[str, torch.Tensor],
        shading: torch.Tensor | None = None,
        residual: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if isinstance(rgb, dict):
            shading = rgb.get("shading", shading)
            residual = rgb.get("residual", residual)
            rgb = rgb.get("rgb")
        if rgb is None or shading is None or residual is None:
            raise ValueError("IndependentTripleResNet18 requires rgb, shading, and residual inputs")
        rgb_feature = self.encoder_rgb(rgb)
        shading_feature = self.encoder_shading(shading)
        residual_feature = self.encoder_residual(residual)
        fused = torch.cat([rgb_feature, shading_feature, residual_feature], dim=1)
        return self.classifier(fused)


def count_parameters(model: nn.Module) -> dict[str, int]:
    total = sum(parameter.numel() for parameter in model.parameters())
    trainable = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    classifier = sum(parameter.numel() for parameter in getattr(model, "classifier").parameters())
    return {"total_params": int(total), "trainable_params": int(trainable), "classifier_params": int(classifier)}
