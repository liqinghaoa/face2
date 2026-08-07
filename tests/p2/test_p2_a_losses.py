from __future__ import annotations

import torch
from torch import nn

from losses.classification_losses import compute_class_weights
from p2_counterfactual.losses import (
    consistency_warmup_factor,
    feature_cosine_loss,
    p2_a_full6_loss,
    p2_a_loss,
    p2_a_pairwise_loss,
    symmetric_js_divergence,
)


def test_weighted_ce_and_non_consistency_loss_are_finite() -> None:
    logits = torch.randn(4, 2, requires_grad=True)
    labels = torch.tensor([0, 1, 1, 0])
    criterion = nn.CrossEntropyLoss(weight=compute_class_weights(labels.tolist(), 2))
    out = p2_a_loss(criterion=criterion, labels=labels, logits_original=logits)
    assert torch.isfinite(out.loss_total)
    assert out.loss_pred.item() == 0.0
    assert out.loss_feat.item() == 0.0
    out.loss_total.backward()
    assert logits.grad is not None


def test_js_divergence_is_symmetric_and_zero_for_same_logits() -> None:
    a = torch.randn(5, 2)
    b = torch.randn(5, 2)
    assert torch.allclose(symmetric_js_divergence(a, b), symmetric_js_divergence(b, a), atol=1e-7)
    assert symmetric_js_divergence(a, a).item() < 1e-7


def test_feature_loss_and_warmup() -> None:
    features = torch.randn(3, 512)
    assert feature_cosine_loss(features, features).item() < 1e-6
    assert [consistency_warmup_factor(i, 5) for i in range(1, 6)] == [0.2, 0.4, 0.6, 0.8, 1.0]


def test_p2_a3_full_loss_backpropagates() -> None:
    logits_o = torch.randn(4, 2, requires_grad=True)
    logits_c = torch.randn(4, 2, requires_grad=True)
    feat_o = torch.randn(4, 512, requires_grad=True)
    feat_c = torch.randn(4, 512, requires_grad=True)
    labels = torch.tensor([0, 1, 1, 0])
    criterion = nn.CrossEntropyLoss(weight=compute_class_weights(labels.tolist(), 2))
    out = p2_a_loss(
        criterion=criterion,
        labels=labels,
        logits_original=logits_o,
        features_original=feat_o,
        logits_counterfactual=logits_c,
        features_counterfactual=feat_c,
        consistency_enabled=True,
        epoch=1,
    )
    assert torch.isfinite(out.loss_total)
    assert out.consistency_warmup_factor == 0.2
    out.loss_total.backward()
    assert logits_o.grad is not None and logits_c.grad is not None


def test_p2_a3_v2_full6_loss_averages_six_relights_and_uses_fixed_weights() -> None:
    labels = torch.tensor([0, 1, 1, 0])
    logits_o = torch.randn(4, 2, requires_grad=True)
    logits_r = torch.randn(4, 6, 2, requires_grad=True)
    feat_o = torch.randn(4, 512, requires_grad=True)
    feat_r = torch.randn(4, 6, 512, requires_grad=True)
    criterion = nn.CrossEntropyLoss(weight=compute_class_weights(labels.tolist(), 2))

    out = p2_a_full6_loss(
        criterion=criterion,
        labels=labels,
        logits_original=logits_o,
        features_original=feat_o,
        logits_relighted=logits_r,
        features_relighted=feat_r,
        epoch=5,
        prediction_weight=0.5,
        feature_weight=0.1,
        warmup_epochs=5,
    )

    relight_ce = torch.stack([criterion(logits_r[:, index, :], labels) for index in range(6)]).mean()
    expected_cls = 0.5 * criterion(logits_o, labels) + 0.5 * relight_ce
    assert torch.allclose(out.loss_cls, expected_cls, atol=1e-7)
    assert out.relight_ce_by_preset is not None and out.relight_ce_by_preset.shape == (6,)
    assert out.pred_by_preset is not None and out.pred_by_preset.shape == (6,)
    assert out.feature_cosine_by_preset is not None and out.feature_cosine_by_preset.shape == (6,)
    expected_total = out.loss_cls + 0.5 * out.loss_pred + 0.1 * out.loss_feat
    assert torch.allclose(out.loss_total, expected_total, atol=1e-7)
    assert torch.isfinite(out.loss_total)
    out.loss_total.backward()
    assert logits_o.grad is not None and logits_r.grad is not None
    assert feat_o.grad is not None and feat_r.grad is not None


def test_p2_a3_v3_pairwise_loss_uses_unified_batch_ce_and_one_scalar() -> None:
    labels = torch.tensor([0, 1, 1, 0])
    logits_o = torch.randn(4, 2, requires_grad=True)
    logits_r = torch.randn(4, 2, requires_grad=True)
    feat_o = torch.randn(4, 512, requires_grad=True)
    feat_r = torch.randn(4, 512, requires_grad=True)
    criterion = nn.CrossEntropyLoss(weight=compute_class_weights(labels.tolist(), 2))
    out = p2_a_pairwise_loss(
        criterion=criterion,
        labels=labels,
        logits_original=logits_o,
        features_original=feat_o,
        logits_relighted=logits_r,
        features_relighted=feat_r,
        epoch=3,
        prediction_weight=0.5,
        feature_weight=0.1,
        warmup_epochs=5,
    )
    expected_cls = criterion(torch.cat([logits_o, logits_r], dim=0), torch.cat([labels, labels], dim=0))
    expected_total = expected_cls + 0.6 * (0.5 * out.loss_pred + 0.1 * out.loss_feat)
    assert torch.allclose(out.loss_cls, expected_cls, atol=1e-7)
    assert torch.allclose(out.loss_total, expected_total, atol=1e-7)
    assert out.consistency_warmup_factor == 0.6
    assert out.loss_total.ndim == 0
    out.loss_total.backward()
    assert logits_o.grad is not None and logits_r.grad is not None
    assert feat_o.grad is not None and feat_r.grad is not None
