from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn

from losses.classification_losses import compute_class_weights


@dataclass
class P2ALossOutput:
    loss_total: torch.Tensor
    loss_cls: torch.Tensor
    loss_pred: torch.Tensor
    loss_feat: torch.Tensor
    consistency_warmup_factor: float
    loss_orig_cls: torch.Tensor | None = None
    loss_relight_cls: torch.Tensor | None = None
    relight_ce_by_preset: torch.Tensor | None = None
    pred_by_preset: torch.Tensor | None = None
    feature_loss_by_preset: torch.Tensor | None = None
    feature_cosine_by_preset: torch.Tensor | None = None
    mean_original_confidence: torch.Tensor | None = None
    mean_relighted_confidence: torch.Tensor | None = None


def consistency_warmup_factor(epoch: int, warmup_epochs: int = 5) -> float:
    if int(warmup_epochs) <= 0:
        return 1.0
    return min(1.0, max(0.0, float(epoch) / float(warmup_epochs)))


def symmetric_js_divergence(logits_a: torch.Tensor, logits_b: torch.Tensor, eps: float = 1e-7) -> torch.Tensor:
    log_p = F.log_softmax(logits_a, dim=1)
    log_q = F.log_softmax(logits_b, dim=1)
    p = log_p.exp()
    q = log_q.exp()
    m = (0.5 * (p + q)).clamp_min(float(eps))
    log_m = m.log()
    kl_pm = (p * (log_p - log_m)).sum(dim=1)
    kl_qm = (q * (log_q - log_m)).sum(dim=1)
    return 0.5 * (kl_pm + kl_qm).mean()


def feature_cosine_loss(features_a: torch.Tensor, features_b: torch.Tensor) -> torch.Tensor:
    return (1.0 - F.cosine_similarity(features_a, features_b, dim=1)).mean()


def _confidence(logits: torch.Tensor) -> torch.Tensor:
    return torch.softmax(logits, dim=-1).amax(dim=-1).mean()


def p2_a_full6_loss(
    *,
    criterion: nn.Module,
    labels: torch.Tensor,
    logits_original: torch.Tensor,
    features_original: torch.Tensor,
    logits_relighted: torch.Tensor,
    features_relighted: torch.Tensor,
    epoch: int = 1,
    prediction_weight: float = 0.5,
    feature_weight: float = 0.1,
    warmup_epochs: int = 5,
) -> P2ALossOutput:
    if logits_relighted.ndim != 3 or logits_relighted.size(1) != 6 or logits_relighted.size(2) != logits_original.size(1):
        raise ValueError(f"full-six logits must be [B,6,C], got {tuple(logits_relighted.shape)}")
    if features_relighted.ndim != 3 or features_relighted.size(1) != 6 or features_relighted.size(2) != features_original.size(1):
        raise ValueError(f"full-six features must be [B,6,F], got {tuple(features_relighted.shape)}")
    labels = labels.long()
    orig_cls = criterion(logits_original, labels)
    relight_ce = torch.stack([criterion(logits_relighted[:, index, :], labels) for index in range(6)], dim=0)
    relight_cls = relight_ce.mean()
    cls = 0.5 * orig_cls + 0.5 * relight_cls

    pred_values = torch.stack([symmetric_js_divergence(logits_original, logits_relighted[:, index, :]) for index in range(6)], dim=0)
    pred = pred_values.mean()
    cosine_values = torch.stack(
        [F.cosine_similarity(features_original, features_relighted[:, index, :], dim=1).mean() for index in range(6)],
        dim=0,
    )
    feature_losses = 1.0 - cosine_values
    feat = feature_losses.mean()
    warmup = consistency_warmup_factor(epoch, warmup_epochs)
    total = cls + float(warmup) * (float(prediction_weight) * pred + float(feature_weight) * feat)
    return P2ALossOutput(
        loss_total=total,
        loss_cls=cls,
        loss_pred=pred,
        loss_feat=feat,
        consistency_warmup_factor=warmup,
        loss_orig_cls=orig_cls,
        loss_relight_cls=relight_cls,
        relight_ce_by_preset=relight_ce,
        pred_by_preset=pred_values,
        feature_loss_by_preset=feature_losses,
        feature_cosine_by_preset=cosine_values,
        mean_original_confidence=_confidence(logits_original),
        mean_relighted_confidence=_confidence(logits_relighted),
    )


def p2_a_pairwise_loss(
    *,
    criterion: nn.Module,
    labels: torch.Tensor,
    logits_original: torch.Tensor,
    features_original: torch.Tensor,
    logits_relighted: torch.Tensor,
    features_relighted: torch.Tensor,
    epoch: int = 1,
    prediction_weight: float = 0.5,
    feature_weight: float = 0.1,
    warmup_epochs: int = 5,
) -> P2ALossOutput:
    if logits_original.shape != logits_relighted.shape:
        raise ValueError(f"pairwise logits shape mismatch: {tuple(logits_original.shape)} vs {tuple(logits_relighted.shape)}")
    if features_original.shape != features_relighted.shape:
        raise ValueError(f"pairwise feature shape mismatch: {tuple(features_original.shape)} vs {tuple(features_relighted.shape)}")
    labels = labels.long()
    all_logits = torch.cat([logits_original, logits_relighted], dim=0)
    all_labels = torch.cat([labels, labels], dim=0)
    cls = criterion(all_logits, all_labels)
    orig_cls = criterion(logits_original, labels)
    relight_cls = criterion(logits_relighted, labels)
    pred = symmetric_js_divergence(logits_original, logits_relighted)
    cosine = F.cosine_similarity(features_original, features_relighted, dim=1)
    feat = (1.0 - cosine).mean()
    warmup = consistency_warmup_factor(epoch, warmup_epochs)
    total = cls + float(warmup) * (float(prediction_weight) * pred + float(feature_weight) * feat)
    return P2ALossOutput(
        loss_total=total,
        loss_cls=cls,
        loss_pred=pred,
        loss_feat=feat,
        consistency_warmup_factor=warmup,
        loss_orig_cls=orig_cls,
        loss_relight_cls=relight_cls,
        feature_cosine_by_preset=cosine.mean().reshape(1),
        mean_original_confidence=_confidence(logits_original),
        mean_relighted_confidence=_confidence(logits_relighted),
    )


def p2_a_loss(
    *,
    criterion: nn.Module,
    labels: torch.Tensor,
    logits_original: torch.Tensor,
    features_original: torch.Tensor | None = None,
    logits_counterfactual: torch.Tensor | None = None,
    features_counterfactual: torch.Tensor | None = None,
    consistency_enabled: bool = False,
    epoch: int = 1,
    prediction_weight: float = 0.5,
    feature_weight: float = 0.1,
    warmup_epochs: int = 5,
) -> P2ALossOutput:
    zero = logits_original.new_tensor(0.0)
    if not consistency_enabled:
        cls = criterion(logits_original, labels.long())
        return P2ALossOutput(cls, cls, zero, zero, 0.0, loss_orig_cls=cls, loss_relight_cls=zero)
    if logits_counterfactual is None or features_original is None or features_counterfactual is None:
        raise ValueError("P2-A3 consistency loss requires original/counterfactual logits and features")
    cls = 0.5 * criterion(logits_original, labels.long()) + 0.5 * criterion(logits_counterfactual, labels.long())
    pred = symmetric_js_divergence(logits_original, logits_counterfactual)
    feat = feature_cosine_loss(features_original, features_counterfactual)
    warmup = consistency_warmup_factor(epoch, warmup_epochs)
    total = cls + float(warmup) * (float(prediction_weight) * pred + float(feature_weight) * feat)
    return P2ALossOutput(
        total,
        cls,
        pred,
        feat,
        warmup,
        loss_orig_cls=criterion(logits_original, labels.long()),
        loss_relight_cls=criterion(logits_counterfactual, labels.long()),
        mean_original_confidence=_confidence(logits_original),
        mean_relighted_confidence=_confidence(logits_counterfactual),
    )


__all__ = [
    "P2ALossOutput",
    "compute_class_weights",
    "consistency_warmup_factor",
    "feature_cosine_loss",
    "p2_a_full6_loss",
    "p2_a_pairwise_loss",
    "p2_a_loss",
    "symmetric_js_divergence",
]
