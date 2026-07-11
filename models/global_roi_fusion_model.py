"""Global face + selected ROI feature-level fusion model for NYHA prediction."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import torch
from torch import nn
from torchvision import models


SUPPORTED_INPUTS = ("global", "eye", "cheek")
SUPPORTED_BACKBONES = ("resnet18", "resnet34", "resnet50")


def _pretrained_enabled(value: bool | str | None) -> bool:
    if value is None:
        return False
    if isinstance(value, bool):
        return value
    normalized = str(value).strip().lower()
    if normalized in {"imagenet", "default", "true", "yes", "1"}:
        return True
    if normalized in {"none", "false", "no", "0", "random"}:
        return False
    raise ValueError(
        "pretrained must be True/False or one of "
        "{'imagenet', 'default', 'true', 'false', 'none', 'random'}; "
        f"got {value!r}"
    )


def _resnet_constructor(backbone: str):
    constructors = {
        "resnet18": (models.resnet18, "ResNet18_Weights"),
        "resnet34": (models.resnet34, "ResNet34_Weights"),
        "resnet50": (models.resnet50, "ResNet50_Weights"),
    }
    normalized = str(backbone).strip().lower()
    if normalized not in constructors:
        raise ValueError(
            f"Unsupported backbone {backbone!r}; choose from {list(constructors)}"
        )
    return constructors[normalized]


def _build_resnet_feature_extractor(
    backbone: str,
    pretrained: bool | str | None,
    freeze_backbone: bool,
) -> tuple[nn.Module, int]:
    constructor, weights_name = _resnet_constructor(backbone)
    weights = None
    if _pretrained_enabled(pretrained):
        weights_enum = getattr(models, weights_name, None)
        if weights_enum is not None:
            weights = weights_enum.DEFAULT
            model = constructor(weights=weights)
        else:
            model = constructor(pretrained=True)
    else:
        model = constructor(weights=None)

    feature_dim = int(model.fc.in_features)
    model.fc = nn.Identity()
    for parameter in model.parameters():
        parameter.requires_grad = not bool(freeze_backbone)
    return model, feature_dim


def count_parameters(model: nn.Module) -> dict[str, int]:
    """Return total and trainable parameter counts."""

    total_params = sum(parameter.numel() for parameter in model.parameters())
    trainable_params = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
    return {"total_params": int(total_params), "trainable_params": int(trainable_params)}


class GlobalROIFusionModel(nn.Module):
    """Independent-backbone feature fusion for global face and selected ROIs.

    Each enabled input owns one independent ResNet feature extractor. Extracted
    features are projected branch-wise, concatenated, and classified by a small
    fusion head.
    """

    def __init__(
        self,
        backbone: str = "resnet18",
        num_classes: int = 3,
        pretrained: bool | str = True,
        enabled_inputs: Sequence[str] = ("global", "eye", "cheek"),
        projection_dim: int = 256,
        dropout: float = 0.3,
        freeze_backbone: bool = False,
    ) -> None:
        super().__init__()
        self.backbone_name = str(backbone).strip().lower()
        self.num_classes = int(num_classes)
        self.enabled_inputs = [str(name).strip().lower() for name in enabled_inputs]
        self.projection_dim = int(projection_dim)
        self.dropout = float(dropout)

        if self.backbone_name not in SUPPORTED_BACKBONES:
            raise ValueError(
                f"Unsupported backbone {backbone!r}; choose from {list(SUPPORTED_BACKBONES)}"
            )
        if self.num_classes != 3:
            raise ValueError(f"num_classes must be 3 for this experiment, got {num_classes}")
        if "global" not in self.enabled_inputs:
            raise ValueError("enabled_inputs must include 'global'")
        if len(set(self.enabled_inputs)) != len(self.enabled_inputs):
            raise ValueError(f"enabled_inputs contain duplicates: {self.enabled_inputs}")
        unsupported = sorted(set(self.enabled_inputs).difference(SUPPORTED_INPUTS))
        if unsupported:
            raise ValueError(
                f"Unsupported enabled_inputs {unsupported}; choose from {list(SUPPORTED_INPUTS)}"
            )
        if self.projection_dim < 1:
            raise ValueError(f"projection_dim must be positive, got {projection_dim}")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError(f"dropout must be in [0, 1), got {dropout}")

        self.extractors = nn.ModuleDict()
        self.projections = nn.ModuleDict()
        feature_dim: int | None = None
        for input_name in self.enabled_inputs:
            extractor, current_feature_dim = _build_resnet_feature_extractor(
                self.backbone_name,
                pretrained=pretrained,
                freeze_backbone=bool(freeze_backbone),
            )
            if feature_dim is None:
                feature_dim = current_feature_dim
            elif feature_dim != current_feature_dim:
                raise RuntimeError("All enabled branches must have the same feature dim")
            self.extractors[input_name] = extractor
            self.projections[input_name] = nn.Sequential(
                nn.Linear(current_feature_dim, self.projection_dim),
                nn.BatchNorm1d(self.projection_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(self.dropout),
            )

        self.feature_dim = int(feature_dim or 0)
        self.fusion_dim = self.projection_dim * len(self.enabled_inputs)
        self.classifier = nn.Sequential(
            nn.Linear(self.fusion_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(self.dropout),
            nn.Linear(256, self.num_classes),
        )

    def _input_from_batch(
        self,
        batch_or_inputs: dict[str, Any] | None,
        input_name: str,
        explicit_value: torch.Tensor | None,
    ) -> torch.Tensor:
        if explicit_value is not None:
            return explicit_value
        if batch_or_inputs is None:
            raise ValueError(f"Missing required input tensor: {input_name}_image")
        key = f"{input_name}_image"
        value = batch_or_inputs.get(key)
        if value is None:
            raise ValueError(f"Missing required input tensor in batch: {key}")
        return value

    def forward(
        self,
        batch_or_inputs: dict[str, Any] | None = None,
        *,
        global_image: torch.Tensor | None = None,
        eye_image: torch.Tensor | None = None,
        cheek_image: torch.Tensor | None = None,
    ) -> torch.Tensor:
        explicit_inputs = {
            "global": global_image,
            "eye": eye_image,
            "cheek": cheek_image,
        }
        projected_features: list[torch.Tensor] = []
        for input_name in self.enabled_inputs:
            image = self._input_from_batch(
                batch_or_inputs,
                input_name,
                explicit_inputs[input_name],
            )
            if image is None:
                raise ValueError(f"Input tensor for {input_name!r} is None")
            features = self.extractors[input_name](image)
            projected_features.append(self.projections[input_name](features))
        fused = torch.cat(projected_features, dim=1)
        return self.classifier(fused)
