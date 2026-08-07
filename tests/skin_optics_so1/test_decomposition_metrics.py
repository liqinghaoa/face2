from __future__ import annotations

import torch

from skin_optics_so1.decomposition.metrics import MaskedMAEAggregator, compute_masked_mae


def test_masked_mae_ignores_outside_mask_and_selection_metric() -> None:
    prediction = torch.zeros(1, 4, 2, 2)
    target = torch.zeros(1, 4, 2, 2)
    target[:, 0] = 1.0
    target[:, 1] = 2.0
    prediction[:, :, 0, :] = 10.0
    mask = torch.zeros(1, 1, 2, 2)
    mask[:, :, 1, :] = 1.0
    result = compute_masked_mae(prediction, target, mask)
    assert result["M_MAE"] == 1.0
    assert result["H_MAE"] == 2.0
    assert result["S_MAE"] == 0.0
    assert result["P_MAE"] == 0.0
    assert result["selection_metric"] == 3.0


def test_masked_mae_aggregates_by_total_valid_pixels_not_batch_mean() -> None:
    aggregator = MaskedMAEAggregator()
    aggregator.update(torch.ones(1, 4, 1, 1), torch.zeros(1, 4, 1, 1), torch.ones(1, 1, 1, 1))
    aggregator.update(torch.zeros(1, 4, 1, 3), torch.zeros(1, 4, 1, 3), torch.ones(1, 1, 1, 3))
    result = aggregator.compute()
    assert result["M_MAE"] == 0.25
