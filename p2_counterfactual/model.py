from __future__ import annotations

import torch
from torch import nn
from torchvision.models import ResNet18_Weights, resnet18


class P2ASingleRGBResNet18(nn.Module):
    """ImageNet ResNet18 with a direct 512 -> 2 classifier for P2-A single RGB experiments."""

    def __init__(self, *, pretrained: bool = True, num_classes: int = 2) -> None:
        super().__init__()
        if int(num_classes) != 2:
            raise ValueError("P2-A is fixed to binary Control/Patient classification")
        backbone = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1 if pretrained else None)
        feature_dim = int(backbone.fc.in_features)
        if feature_dim != 512:
            raise ValueError(f"ResNet18 feature dimension must be 512, got {feature_dim}")
        backbone.fc = nn.Identity()
        self.backbone = backbone
        self.classifier = nn.Linear(feature_dim, int(num_classes))
        self.feature_dim = feature_dim
        self.num_classes = int(num_classes)

    def forward(self, images: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if images.ndim != 4 or images.size(1) != 3:
            raise ValueError(f"P2-A expects [B,3,H,W] RGB tensor, got {tuple(images.shape)}")
        features = self.backbone(images)
        logits = self.classifier(features)
        return logits, features
