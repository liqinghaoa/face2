from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

from skin_optics_so1.decomposition.losses import SMOOTH_L1_BETA, masked_smooth_l1_loss


def test_masked_smooth_l1_zero_when_prediction_matches_target() -> None:
    target = torch.rand(2, 4, 8, 8)
    mask = torch.ones(2, 1, 8, 8)
    losses = masked_smooth_l1_loss(target, target, mask)
    assert float(losses["total"]) == 0.0


def test_masked_smooth_l1_full_mask_matches_plain_smooth_l1_per_channel() -> None:
    prediction = torch.zeros(1, 4, 4, 4)
    target = torch.ones(1, 4, 4, 4) * 0.2
    mask = torch.ones(1, 1, 4, 4)
    losses = masked_smooth_l1_loss(prediction, target, mask)
    expected = F.smooth_l1_loss(
        prediction[:, 0:1], target[:, 0:1], reduction="mean", beta=SMOOTH_L1_BETA
    )
    assert torch.allclose(losses["M"], expected)
    assert torch.allclose(losses["total"], expected)


def test_masked_smooth_l1_ignores_outside_mask_and_rejects_empty_mask() -> None:
    target = torch.zeros(1, 4, 4, 4)
    prediction = target.clone()
    prediction[:, :, 0:2, :] = 1.0
    mask = torch.zeros(1, 1, 4, 4)
    mask[:, :, 2:, :] = 1.0
    losses = masked_smooth_l1_loss(prediction, target, mask)
    assert float(losses["total"]) == 0.0
    with pytest.raises(ValueError, match="valid_mask.sum"):
        masked_smooth_l1_loss(prediction, target, torch.zeros_like(mask))


def test_masked_smooth_l1_gradient_and_amp_finite() -> None:
    prediction = torch.rand(1, 4, 8, 8, requires_grad=True)
    target = torch.rand(1, 4, 8, 8)
    mask = torch.ones(1, 1, 8, 8)
    with torch.amp.autocast("cpu", enabled=True):
        losses = masked_smooth_l1_loss(prediction, target, mask)
    losses["total"].backward()
    assert prediction.grad is not None
    assert torch.isfinite(prediction.grad).all()
