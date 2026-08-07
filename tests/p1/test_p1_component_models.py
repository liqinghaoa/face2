from __future__ import annotations

import torch

from models.p1_component_models import build_p1_component_model, count_component_parameters
from models.p1_shared_dual_resnet18 import SharedResNet18DualBranch, audit_dual_branch_parameters


def test_model_factory_builds_required_heads():
    light = build_p1_component_model("linear_probe", {"num_classes": 2})
    assert light.classifier.out_features == 2
    single = build_p1_component_model("resnet18_single", {"pretrained": False, "num_classes": 2})
    assert single.fc.out_features == 2


def test_shared_dual_branch_uses_one_encoder_and_parameter_audit_exists():
    model = SharedResNet18DualBranch(pretrained=False)
    audit = audit_dual_branch_parameters(pretrained=False)
    assert audit.shared_dual_trainable_parameters < audit.independent_dual_reference_parameters
    batch = {
        "rgb": torch.zeros(2, 3, 224, 224),
        "albedo": torch.zeros(2, 3, 224, 224),
    }
    logits = model(batch)
    assert tuple(logits.shape) == (2, 2)
    assert count_component_parameters(model)["trainable_params"] == audit.shared_dual_trainable_parameters

