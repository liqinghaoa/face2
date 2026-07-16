"""Strictly monotonic cumulative-link ResNet model for three ordered classes."""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn

from models.resnet_nyha_3class import build_resnet_nyha_model


def inverse_softplus(value: float) -> float:
    if value <= 0:
        raise ValueError("inverse_softplus requires a positive value")
    return value + math.log(-math.expm1(-value))


class MonotonicCumulativeLinkHead(nn.Module):
    """One severity score and two structurally ordered cutpoints."""

    def __init__(
        self,
        in_features: int,
        *,
        min_gap: float = 1e-4,
        initial_c0: float = -1.0,
        initial_c1: float = 1.0,
    ) -> None:
        super().__init__()
        if initial_c1 - initial_c0 <= min_gap:
            raise ValueError("initial cutpoint gap must exceed min_gap")
        self.min_gap = float(min_gap)
        self.severity = nn.Linear(int(in_features), 1, bias=False)
        self.theta0 = nn.Parameter(torch.tensor(float(initial_c0)))
        desired_softplus = float(initial_c1 - initial_c0 - min_gap)
        self.raw_delta = nn.Parameter(torch.tensor(inverse_softplus(desired_softplus)))

    def cutpoints(self) -> torch.Tensor:
        c0 = self.theta0
        c1 = self.theta0 + F.softplus(self.raw_delta) + self.min_gap
        return torch.stack((c0, c1))

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        score = self.severity(features)
        cutpoints = self.cutpoints()
        return score - cutpoints.unsqueeze(0)


class MonotonicCumulativeResNet18(nn.Module):
    """ImageNet-pretrained ResNet18 with a monotonic cumulative-link head."""

    def __init__(self, *, pretrained: bool = True, min_gap: float = 1e-4) -> None:
        super().__init__()
        backbone = build_resnet_nyha_model(
            backbone="resnet18", num_classes=3, pretrained=pretrained
        )
        in_features = int(backbone.fc.in_features)
        backbone.fc = nn.Identity()
        self.backbone = backbone
        self.ordinal_head = MonotonicCumulativeLinkHead(
            in_features, min_gap=min_gap, initial_c0=-1.0, initial_c1=1.0
        )

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.ordinal_head(self.backbone(images))

    def cutpoints(self) -> torch.Tensor:
        return self.ordinal_head.cutpoints()
