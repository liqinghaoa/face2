"""Architecture and checkpoint tests for the fixed Stage 2B MLP."""

from __future__ import annotations

import numpy as np  # noqa: F401  # Initialize NumPy/MKL before PyTorch in this environment.
import torch
from torch import nn

from models.exif_conditioned_response_mlp import (
    ARCHITECTURE,
    EXPECTED_PARAMETER_COUNT,
    EXIFConditionedResponseMLP,
    restore_model_from_checkpoint,
)


def test_fixed_architecture_shape_parameter_count_and_modules() -> None:
    model = EXIFConditionedResponseMLP()
    x = torch.zeros(7, 5)
    assert model(x).shape == (7, 3)
    assert model.parameter_count == EXPECTED_PARAMETER_COUNT == 147
    modules = list(model.modules())
    assert sum(isinstance(module, nn.Tanh) for module in modules) == 2
    assert not any(isinstance(module, (nn.BatchNorm1d, nn.LayerNorm, nn.Dropout)) for module in modules)
    assert isinstance(model.network[-1], nn.Linear)
    assert ARCHITECTURE == {
        "name": "EXIFConditionedResponseMLP", "input_dim": 5,
        "hidden_dims": [8, 8], "output_dim": 3,
        "activation": "Tanh", "output_activation": None,
    }


def test_forward_accepts_only_five_condition_features() -> None:
    model = EXIFConditionedResponseMLP()
    for invalid in (torch.zeros(5), torch.zeros(2, 4), torch.zeros(2, 6)):
        try:
            model(invalid)
        except ValueError as exc:
            assert "shape [batch, 5]" in str(exc)
        else:
            raise AssertionError("Invalid condition shape was accepted")
    try:
        model(torch.zeros(2, 5), torch.zeros(2, 3))  # type: ignore[call-arg]
    except TypeError:
        pass
    else:
        raise AssertionError("Model unexpectedly accepted a raw-observation argument")


def test_checkpoint_roundtrip_reproduces_predictions(tmp_path) -> None:
    torch.manual_seed(123)
    model = EXIFConditionedResponseMLP().eval()
    x = torch.randn(12, 5)
    expected = model(x).detach().clone()
    path = tmp_path / "model.pth"
    payload = {
        "model_state_dict": model.state_dict(), "architecture": ARCHITECTURE,
        "parameter_count": 147, "target_names": ["a", "b", "c"],
        "condition_feature_names": ["d", "e", "i", "de", "di"],
        "target_scaler": {"mean": [1, 2, 3], "population_std": [4, 5, 6]},
    }
    torch.save(payload, path)
    restored, loaded = restore_model_from_checkpoint(path)
    assert torch.equal(expected, restored(x))
    assert loaded["target_names"] == payload["target_names"]
    assert loaded["target_scaler"] == payload["target_scaler"]
