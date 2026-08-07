"""E2: Global ResNet18 plus one shared ROI ResNet18 with fixed mean fusion."""

from __future__ import annotations

import torch
from torch import nn

from models.resnet_nyha_3class import build_resnet_nyha_model


ROI_ORDER = ("eye", "lip", "cheek", "forehead", "chin")


def _resnet18_features(pretrained: bool) -> nn.Module:
    model = build_resnet_nyha_model("resnet18", num_classes=3, pretrained=pretrained)
    if int(model.fc.in_features) != 512:
        raise RuntimeError("E2 requires ResNet18 512-dimensional pooled features")
    model.fc = nn.Identity()
    return model


class E2GlobalMultiROIEqualFusionResNet18(nn.Module):
    """Two independent ResNet18s; five ROI views share the ROI encoder.

    ``roi_images`` is [B, 5, 3, H, W].  The only ROI aggregation is the
    parameter-free arithmetic mean along the fixed ROI dimension.
    """

    roi_order = ROI_ORDER

    def __init__(self, pretrained: bool = True) -> None:
        super().__init__()
        self.global_encoder = _resnet18_features(pretrained)
        self.roi_encoder = _resnet18_features(pretrained)
        self.classifier = nn.Linear(1024, 3)
        for parameter in self.parameters():
            parameter.requires_grad = True

    @staticmethod
    def aggregate_roi_features(roi_features: torch.Tensor) -> torch.Tensor:
        """Return the prescribed non-learnable equal mean, [B, 5, 512] -> [B, 512]."""
        if roi_features.ndim != 3 or roi_features.shape[1:] != (5, 512):
            raise ValueError(f"roi_features must be [B,5,512], got {tuple(roi_features.shape)}")
        return roi_features.mean(dim=1)

    def forward(
        self,
        global_image: torch.Tensor,
        roi_images: torch.Tensor,
        *,
        return_features: bool = False,
    ) -> torch.Tensor | dict[str, torch.Tensor]:
        if global_image.ndim != 4 or global_image.shape[1] != 3:
            raise ValueError("global_image must be [B,3,H,W]")
        if roi_images.ndim != 5 or roi_images.shape[1:3] != (5, 3):
            raise ValueError("roi_images must be [B,5,3,H,W] in the fixed E2 ROI order")
        batch_size = global_image.shape[0]
        if roi_images.shape[0] != batch_size:
            raise ValueError("global_image and roi_images must have the same batch size")
        f_global = self.global_encoder(global_image)
        flat = roi_images.reshape(batch_size * 5, *roi_images.shape[2:])
        roi_features = self.roi_encoder(flat).reshape(batch_size, 5, 512)
        f_roi = self.aggregate_roi_features(roi_features)
        fused_features = torch.cat([f_global, f_roi], dim=1)
        logits = self.classifier(fused_features)
        if return_features:
            return {
                "logits": logits,
                "global_features": f_global,
                "roi_features": roi_features,
                "roi_mean_features": f_roi,
                "fused_features": fused_features,
            }
        return logits


def count_parameters(model: nn.Module) -> dict[str, int]:
    return {
        "total_params": sum(parameter.numel() for parameter in model.parameters()),
        "trainable_params": sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad),
    }
