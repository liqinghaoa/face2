"""Direct-concatenation ResNet18 models for the optical fusion experiment."""

from __future__ import annotations

import torch
from torch import nn

from models.resnet_nyha_3class import _build_torchvision_resnet
from utils.optical_feature_preprocessor import VARIANT_AUX_DIM, validate_variant


class ResNet18OpticalFusion(nn.Module):
    """Fully trainable ResNet18 GAP features plus an optional fixed-width aux vector."""

    global_feature_dim = 512

    def __init__(
        self, variant: str, num_classes: int = 3, pretrained: bool = True
    ) -> None:
        super().__init__()
        self.variant = validate_variant(variant)
        if int(num_classes) != 3:
            raise ValueError("This locked experiment requires num_classes=3")
        self.num_classes = int(num_classes)
        self.auxiliary_input_dim = VARIANT_AUX_DIM[self.variant]
        self.fused_input_dim = self.global_feature_dim + self.auxiliary_input_dim
        self.backbone = _build_torchvision_resnet("resnet18", bool(pretrained))
        if int(self.backbone.fc.in_features) != self.global_feature_dim:
            raise ValueError("Unexpected torchvision ResNet18 feature dimension")
        self.backbone.fc = nn.Identity()
        self.classifier = nn.Linear(self.fused_input_dim, self.num_classes)
        for parameter in self.parameters():
            parameter.requires_grad = True

    def forward_features(self, images: torch.Tensor) -> torch.Tensor:
        features = self.backbone(images)
        if features.ndim != 2 or features.shape[1] != self.global_feature_dim:
            raise RuntimeError(f"Expected [B,512] backbone features, got {features.shape}")
        return features

    def _validate_aux(
        self, features: torch.Tensor, aux_features: torch.Tensor | None
    ) -> torch.Tensor | None:
        batch_size = features.shape[0]
        expected = self.auxiliary_input_dim
        if expected == 0:
            if aux_features is None:
                return None
            if aux_features.ndim != 2 or tuple(aux_features.shape) != (batch_size, 0):
                raise ValueError("global_only accepts only None or aux with shape [B,0]")
            return None
        if aux_features is None:
            raise ValueError(f"{self.variant} requires aux_features with shape [B,{expected}]")
        if aux_features.ndim != 2:
            raise ValueError(f"aux_features must be 2D, got {aux_features.shape}")
        if aux_features.shape[0] != batch_size:
            raise ValueError("Image and auxiliary batch sizes do not match")
        if aux_features.shape[1] != expected:
            raise ValueError(
                f"{self.variant} requires aux width {expected}, got {aux_features.shape[1]}"
            )
        if not torch.is_floating_point(aux_features):
            raise TypeError("aux_features must use a floating-point dtype")
        if not torch.isfinite(aux_features).all():
            raise ValueError("aux_features contain NaN or infinity")
        availability = aux_features[:, -1]
        if not torch.logical_or(availability == 0, availability == 1).all():
            raise ValueError("The last auxiliary value, forehead_available, must be 0 or 1")
        return aux_features.to(device=features.device, dtype=features.dtype)

    def forward(
        self, images: torch.Tensor, aux_features: torch.Tensor | None = None
    ) -> torch.Tensor:
        features = self.forward_features(images)
        aux = self._validate_aux(features, aux_features)
        fused = features if aux is None else torch.cat((features, aux), dim=1)
        return self.classifier(fused)

    @property
    def classifier_head_parameter_count(self) -> int:
        return int(sum(parameter.numel() for parameter in self.classifier.parameters()))


def build_resnet18_optical_fusion(
    variant: str, num_classes: int = 3, pretrained: bool = True
) -> ResNet18OpticalFusion:
    return ResNet18OpticalFusion(variant, num_classes=num_classes, pretrained=pretrained)
