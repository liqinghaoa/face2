"""Global ResNet18 with two conditional NYHA prediction heads for E1."""

from __future__ import annotations

import torch
from torch import nn

from models.resnet_nyha_3class import build_resnet_nyha_model


class GlobalResNet18HierarchicalConditional(nn.Module):
    """ImageNet ResNet18 encoder with Normal/Abnormal and Mild/Severe heads."""

    def __init__(self, pretrained: bool = True) -> None:
        super().__init__()
        base = build_resnet_nyha_model(
            backbone="resnet18", num_classes=3, pretrained=pretrained
        )
        feature_dim = int(base.fc.in_features)
        if feature_dim != 512:
            raise RuntimeError(f"Expected ResNet18 feature dim 512, got {feature_dim}")
        base.fc = nn.Identity()
        self.backbone = base
        self.abnormal_head = nn.Linear(512, 1)
        self.severe_cond_head = nn.Linear(512, 1)
        for parameter in self.parameters():
            parameter.requires_grad = True

    def forward(self, images: torch.Tensor) -> dict[str, torch.Tensor]:
        h_global = self.backbone(images)
        logit_abnormal = self.abnormal_head(h_global)
        logit_severe_cond = self.severe_cond_head(h_global)
        return {
            "logit_abnormal": logit_abnormal,
            "logit_severe_cond": logit_severe_cond,
        }
