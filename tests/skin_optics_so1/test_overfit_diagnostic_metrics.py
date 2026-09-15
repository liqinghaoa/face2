from __future__ import annotations

import math

import torch

from skin_optics_so1.decomposition.diagnostic_metrics import (
    constant_baseline,
    cross_channel_matrices,
    diagnostic_metrics,
    final_output_gradient_norms,
    pair_prediction_drift,
)
from skin_optics_so1.decomposition.unet_decomposer import SO1UNetDecomposer


def test_constant_baseline_and_p_active_metrics() -> None:
    target = torch.zeros(1, 4, 1, 4)
    target[:, 0] = torch.tensor([[[0.0, 0.0, 1.0, 1.0]]])
    target[:, 3, 0, 2:] = 0.5
    prediction = target.clone()
    result = diagnostic_metrics(prediction, target, torch.ones(1, 1, 1, 4))
    baseline = constant_baseline(prediction, target, torch.ones(1, 1, 1, 4))
    assert result["channels"]["P"]["active_region_recall"] == 1.0
    assert result["channels"]["P"]["active_region_MAE"] == 0.0
    assert baseline["M"]["relative_improvement"] == 1.0


def test_cross_channel_pearson_returns_nan_for_constant_vectors() -> None:
    target = torch.zeros(1, 4, 1, 3)
    prediction = torch.zeros_like(target)
    mae, pearson, counts = cross_channel_matrices(prediction, target, torch.ones(1, 1, 1, 3))
    assert mae.shape == (4, 4)
    assert math.isnan(pearson[0, 0])
    assert counts[0, 0] == 3


def test_pair_drift_and_output_gradient_norms_are_channel_separate() -> None:
    prediction = torch.zeros(2, 4, 2, 2)
    prediction[1, 1] = 2.0
    drift = pair_prediction_drift(prediction, torch.ones(2, 1, 2, 2))
    assert drift["M_pair_prediction_drift"] == 0.0
    assert drift["H_pair_prediction_drift"] == 2.0
    model = SO1UNetDecomposer()
    model.outc.weight.grad = torch.ones_like(model.outc.weight)
    norms = final_output_gradient_norms(model)
    assert set(norms) == {"M_output_gradient_norm", "H_output_gradient_norm", "S_output_gradient_norm", "P_output_gradient_norm"}
    assert len(set(norms.values())) == 1
