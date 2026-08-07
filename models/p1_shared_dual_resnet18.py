"""Shared-encoder dual-branch ResNet18 for P1-RGB+A."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import nn
from torchvision.models import ResNet18_Weights, resnet18


def _resnet18_encoder(pretrained: bool = True) -> tuple[nn.Module, int]:
    model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1 if pretrained else None)
    feature_dim = int(model.fc.in_features)
    model.fc = nn.Identity()
    return model, feature_dim


class SharedResNet18DualBranch(nn.Module):
    """One encoder reused for both RGB and albedo inputs."""

    def __init__(
        self,
        *,
        pretrained: bool = True,
        num_classes: int = 2,
    ) -> None:
        super().__init__()
        if int(num_classes) != 2:
            raise ValueError("P1-RGB+A is a fixed binary task")
        self.encoder, feature_dim = _resnet18_encoder(pretrained=pretrained)
        self.classifier = nn.Linear(feature_dim * 2, num_classes)
        self.feature_dim = feature_dim
        self.num_classes = int(num_classes)

    def encode(self, tensor: torch.Tensor) -> torch.Tensor:
        return self.encoder(tensor)

    def forward(
        self,
        batch_or_rgb: dict[str, Any] | torch.Tensor,
        albedo: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if isinstance(batch_or_rgb, dict):
            rgb = batch_or_rgb.get("rgb")
            albedo = batch_or_rgb.get("albedo", albedo)
            if rgb is None or albedo is None:
                representation = batch_or_rgb.get("representation")
                if isinstance(representation, dict):
                    rgb = representation.get("rgb", rgb)
                    albedo = representation.get("albedo", albedo)
        else:
            rgb = batch_or_rgb
        if rgb is None or albedo is None:
            raise ValueError("both rgb and albedo inputs are required")
        rgb_feature = self.encode(rgb)
        albedo_feature = self.encode(albedo)
        fused = torch.cat([rgb_feature, albedo_feature], dim=1)
        return self.classifier(fused)


@dataclass(frozen=True)
class DualBranchParameterAudit:
    single_resnet_trainable_parameters: int
    shared_dual_trainable_parameters: int
    independent_dual_reference_parameters: int

    def to_dict(self) -> dict[str, int]:
        return {
            "single_resnet_trainable_parameters": int(self.single_resnet_trainable_parameters),
            "shared_dual_trainable_parameters": int(self.shared_dual_trainable_parameters),
            "independent_dual_reference_parameters": int(self.independent_dual_reference_parameters),
        }


def count_parameters(model: nn.Module) -> dict[str, int]:
    total = sum(parameter.numel() for parameter in model.parameters())
    trainable = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    return {"total_params": int(total), "trainable_params": int(trainable)}


def audit_dual_branch_parameters(pretrained: bool = True) -> DualBranchParameterAudit:
    single, feature_dim = _resnet18_encoder(pretrained=pretrained)
    shared = SharedResNet18DualBranch(pretrained=pretrained)
    reference_encoder_a, _ = _resnet18_encoder(pretrained=pretrained)
    reference_encoder_b, _ = _resnet18_encoder(pretrained=pretrained)

    class _IndependentReference(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.rgb = reference_encoder_a
            self.albedo = reference_encoder_b
            self.classifier = nn.Linear(feature_dim * 2, 2)

        def forward(self, rgb: torch.Tensor, albedo: torch.Tensor) -> torch.Tensor:
            return self.classifier(torch.cat([self.rgb(rgb), self.albedo(albedo)], dim=1))

    reference = _IndependentReference()
    return DualBranchParameterAudit(
        single_resnet_trainable_parameters=count_parameters(single)["trainable_params"],
        shared_dual_trainable_parameters=count_parameters(shared)["trainable_params"],
        independent_dual_reference_parameters=count_parameters(reference)["trainable_params"],
    )
