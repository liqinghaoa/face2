"""Configurable shared-backbone Multi-ROI fusion models for NYHA prediction."""

from __future__ import annotations

import torch
from torch import nn

from models.resnet_nyha_3class import _build_torchvision_resnet


class ConfigurableMultiROIFusionResNet(nn.Module):
    """Shared ResNet ROI feature extractor with concat fusion classifier."""

    def __init__(
        self,
        backbone: str = "resnet34",
        pretrained: str | bool = "imagenet",
        num_rois: int = 4,
        num_classes: int = 3,
        shared_backbone: bool = True,
        fusion_method: str = "concat",
        hidden_dim: int = 512,
        dropout: float = 0.3,
        use_batchnorm: bool = True,
        freeze_backbone: bool = False,
    ) -> None:
        super().__init__()
        self.backbone_name = str(backbone).lower()
        self.num_rois = int(num_rois)
        self.num_classes = int(num_classes)
        self.shared_backbone = bool(shared_backbone)
        self.fusion_method = str(fusion_method).lower()

        if self.backbone_name not in {"resnet18", "resnet34", "resnet50"}:
            raise ValueError(
                "Unsupported backbone for Multi-ROI fusion: "
                f"{backbone!r}; choose resnet18, resnet34, or resnet50"
            )
        if not self.shared_backbone:
            raise ValueError("Only shared_backbone=true is supported in this version")
        if self.fusion_method != "concat":
            raise ValueError("Only fusion_method='concat' is supported in this version")
        if self.num_rois < 2:
            raise ValueError(f"num_rois must be >= 2, got {self.num_rois}")
        if self.num_classes != 3:
            raise ValueError(f"num_classes must be 3, got {self.num_classes}")
        if bool(freeze_backbone):
            raise ValueError("freeze_backbone=true is not supported for this experiment")

        pretrained_enabled = (
            pretrained
            if isinstance(pretrained, bool)
            else str(pretrained).lower() in {"imagenet", "true", "yes", "1"}
        )
        self.backbone = _build_torchvision_resnet(
            self.backbone_name, bool(pretrained_enabled)
        )
        self.feature_dim = int(self.backbone.fc.in_features)
        self.backbone.fc = nn.Identity()
        for parameter in self.backbone.parameters():
            parameter.requires_grad = True

        self.fusion_dim = self.num_rois * self.feature_dim
        layers: list[nn.Module] = [nn.Linear(self.fusion_dim, int(hidden_dim))]
        if bool(use_batchnorm):
            layers.append(nn.BatchNorm1d(int(hidden_dim)))
        layers.extend(
            [
                nn.ReLU(inplace=True),
                nn.Dropout(float(dropout)),
                nn.Linear(int(hidden_dim), self.num_classes),
            ]
        )
        self.classifier = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 5:
            raise ValueError(f"Multi-ROI input must have shape [B, R, C, H, W], got {x.shape}")
        batch_size, num_rois, channels, height, width = x.shape
        if num_rois != self.num_rois:
            raise ValueError(
                f"Input ROI dimension {num_rois} != configured num_rois {self.num_rois}"
            )
        x = x.reshape(batch_size * num_rois, channels, height, width)
        features = self.backbone(x)
        features = features.reshape(batch_size, num_rois, self.feature_dim)
        fused = features.reshape(batch_size, self.fusion_dim)
        return self.classifier(fused)
