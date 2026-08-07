from __future__ import annotations

import torch
from torch import nn

from p2_counterfactual.model import P2ASingleRGBResNet18


def test_p2_a_model_shapes_and_simple_head() -> None:
    model = P2ASingleRGBResNet18(pretrained=False)
    logits, features = model(torch.randn(2, 3, 224, 224))
    assert logits.shape == (2, 2)
    assert features.shape == (2, 512)
    assert isinstance(model.classifier, nn.Linear)
    assert model.classifier.in_features == 512
    assert model.classifier.out_features == 2


def test_p2_a3_uses_shared_model_instance() -> None:
    model = P2ASingleRGBResNet18(pretrained=False)
    logits_o, feat_o = model(torch.randn(1, 3, 224, 224))
    logits_c, feat_c = model(torch.randn(1, 3, 224, 224))
    assert logits_o.shape == logits_c.shape == (1, 2)
    assert feat_o.shape == feat_c.shape == (1, 512)
    assert len({id(parameter) for parameter in model.parameters()}) == len(list(model.parameters()))
