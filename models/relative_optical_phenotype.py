"""P2-1 Global + Eye-Cheek relative optical phenotype model."""

from __future__ import annotations

from typing import Any

import torch
from torch import nn
from torchvision.models import ResNet18_Weights, resnet18


def _resnet18_features(pretrained: bool | str) -> tuple[nn.Module, int]:
    use_weights = bool(pretrained) and str(pretrained).lower() not in {"false", "none", "0"}
    model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1 if use_weights else None)
    dim = int(model.fc.in_features)
    model.fc = nn.Identity()
    return model, dim


class OpticalInputAdapter(nn.Conv2d):
    def __init__(self) -> None:
        super().__init__(7, 3, kernel_size=1, bias=False)
        with torch.no_grad():
            self.weight.zero_()
            self.weight[0, 0, 0, 0] = 1.0
            self.weight[1, 1, 0, 0] = 1.0
            self.weight[2, 2, 0, 0] = 1.0


class RelativeOpticalPhenotypeModel(nn.Module):
    """One global encoder and one parameter-shared ROI encoder."""

    def __init__(
        self,
        pretrained: bool | str = True,
        projection_dim: int = 256,
        dropout: float = 0.3,
        num_classes: int = 3,
    ) -> None:
        super().__init__()
        if num_classes != 3:
            raise ValueError("P2-1 is fixed to three classes")
        self.global_encoder, feature_dim = _resnet18_features(pretrained)
        self.roi_encoder, roi_feature_dim = _resnet18_features(pretrained)
        if feature_dim != roi_feature_dim:
            raise RuntimeError("encoder feature dimensions differ")
        self.optical_adapter = OpticalInputAdapter()
        self.global_projection = nn.Sequential(
            nn.Linear(feature_dim, projection_dim), nn.BatchNorm1d(projection_dim), nn.ReLU(True), nn.Dropout(dropout)
        )
        self.roi_projection = nn.Sequential(
            nn.Linear(feature_dim, projection_dim), nn.BatchNorm1d(projection_dim), nn.ReLU(True), nn.Dropout(dropout)
        )
        self.feature_dim = feature_dim
        self.projection_dim = projection_dim
        self.fusion_dim = projection_dim * 3
        self.classifier = nn.Sequential(
            nn.Linear(self.fusion_dim, 256), nn.BatchNorm1d(256), nn.ReLU(True), nn.Dropout(dropout), nn.Linear(256, 3)
        )

    def encode_roi(self, optical: torch.Tensor) -> torch.Tensor:
        return self.roi_projection(self.roi_encoder(self.optical_adapter(optical)))

    def relative_features(
        self, global_image: torch.Tensor, eye_optical: torch.Tensor, cheek_optical: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        global_feature = self.global_projection(self.global_encoder(global_image))
        eye_feature = self.encode_roi(eye_optical)
        cheek_feature = self.encode_roi(cheek_optical)
        signed = eye_feature - cheek_feature
        absolute = torch.abs(signed)
        return global_feature, signed, absolute, torch.cat([global_feature, signed, absolute], dim=1)

    def forward(
        self,
        batch_or_inputs: dict[str, Any] | None = None,
        *,
        global_image: torch.Tensor | None = None,
        eye_optical: torch.Tensor | None = None,
        cheek_optical: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if batch_or_inputs is not None:
            global_image = batch_or_inputs.get("global_image", global_image)
            eye_optical = batch_or_inputs.get("eye_optical", eye_optical)
            cheek_optical = batch_or_inputs.get("cheek_optical", cheek_optical)
        if global_image is None or eye_optical is None or cheek_optical is None:
            raise ValueError("global_image, eye_optical and cheek_optical are required")
        _, _, _, fusion = self.relative_features(global_image, eye_optical, cheek_optical)
        return self.classifier(fusion)


def count_parameters(model: nn.Module) -> dict[str, int]:
    return {
        "total_params": sum(p.numel() for p in model.parameters()),
        "trainable_params": sum(p.numel() for p in model.parameters() if p.requires_grad),
    }
