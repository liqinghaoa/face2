from __future__ import annotations

import types

import numpy as np  # Preload NumPy before torch for the local Anaconda/MKL runtime.
import pytest
import torch
from torch import nn

from models.resnet18_optical_fusion import ResNet18OpticalFusion
from utils.optical_feature_preprocessor import VARIANT_AUX_DIM, VARIANTS

assert np.__version__


@pytest.mark.parametrize(
    ("variant", "aux_dim", "head_in", "head_parameters"),
    [
        ("global_only", 0, 512, 1539),
        ("global_mask", 1, 513, 1542),
        ("global_raw", 7, 519, 1560),
        ("global_stage2a", 7, 519, 1560),
        ("global_stage2b", 7, 519, 1560),
    ],
)
def test_model_shapes_and_locked_head(variant, aux_dim, head_in, head_parameters):
    model = ResNet18OpticalFusion(variant, pretrained=False).eval()
    images = torch.randn(2, 3, 64, 64)
    aux = torch.zeros(2, aux_dim)
    if aux_dim:
        aux[:, -1] = torch.tensor([0.0, 1.0])
    with torch.no_grad():
        features = model.forward_features(images)
        logits = model(images, None if aux_dim == 0 else aux)
    assert features.shape == (2, 512)
    assert logits.shape == (2, 3)
    assert model.classifier.in_features == head_in
    assert model.classifier_head_parameter_count == head_parameters
    assert [module for module in model.modules() if isinstance(module, nn.Linear)] == [model.classifier]
    assert not any(isinstance(module, nn.Dropout) for module in model.modules())
    assert all(parameter.requires_grad for parameter in model.parameters())


def _fast_model(variant: str) -> ResNet18OpticalFusion:
    model = ResNet18OpticalFusion(variant, pretrained=False)
    model.forward_features = types.MethodType(
        lambda self, images: torch.zeros(images.shape[0], 512, dtype=images.dtype), model
    )
    return model


@pytest.mark.parametrize("variant", VARIANTS)
def test_variant_aux_width_is_immutable(variant):
    assert _fast_model(variant).auxiliary_input_dim == VARIANT_AUX_DIM[variant]


def test_global_only_accepts_none_or_empty_only():
    model = _fast_model("global_only")
    images = torch.zeros(2, 3, 8, 8)
    assert model(images).shape == (2, 3)
    assert model(images, torch.empty(2, 0)).shape == (2, 3)
    with pytest.raises(ValueError, match="global_only"):
        model(images, torch.zeros(2, 1))


@pytest.mark.parametrize("variant", ["global_mask", "global_raw", "global_stage2a", "global_stage2b"])
def test_aux_validation_errors(variant):
    model = _fast_model(variant)
    images = torch.zeros(2, 3, 8, 8)
    width = VARIANT_AUX_DIM[variant]
    valid = torch.zeros(2, width)
    valid[:, -1] = 1
    assert model(images, valid).shape == (2, 3)
    with pytest.raises(ValueError, match="requires"):
        model(images, None)
    with pytest.raises(ValueError, match="batch"):
        model(images, valid[:1])
    with pytest.raises(ValueError, match="width"):
        model(images, valid[:, :-1])
    invalid = valid.clone()
    invalid[0, 0] = float("nan")
    with pytest.raises(ValueError, match="NaN"):
        model(images, invalid)
    invalid = valid.clone()
    invalid[0, -1] = 0.5
    with pytest.raises(ValueError, match="must be 0 or 1"):
        model(images, invalid)
    with pytest.raises(TypeError, match="floating"):
        model(images, valid.long())


def test_illegal_variant_fails():
    with pytest.raises(ValueError, match="Unknown variant"):
        ResNet18OpticalFusion("not_a_variant", pretrained=False)
