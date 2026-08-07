"""Fold-specific weighted conditional BCE loss used by E1."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import asdict, dataclass

import torch
import torch.nn.functional as F
from torch import nn


@dataclass(frozen=True)
class E1ClassWeights:
    class_counts: dict[str, int]
    w_abnormal_neg: float
    w_abnormal_pos: float
    w_severe_neg: float
    w_severe_pos: float

    def as_dict(self) -> dict[str, float | dict[str, int]]:
        return asdict(self)


def compute_e1_class_weights(labels: Iterable[int]) -> E1ClassWeights:
    """Compute the prescribed fold-training-set-only E1 binary weights."""
    values = torch.as_tensor(list(labels), dtype=torch.long)
    if values.numel() == 0 or not torch.isin(values, torch.tensor([0, 1, 2])).all():
        raise ValueError("E1 class weights require a non-empty sequence of labels in {0,1,2}")
    counts = torch.bincount(values, minlength=3)
    n0, n1, n2 = (int(value) for value in counts.tolist())
    if min(n0, n1, n2) == 0:
        raise ValueError(
            "E1 fold-specific weighted BCE is undefined because a training class is "
            f"missing: n0={n0}, n1={n1}, n2={n2}"
        )
    total = n0 + n1 + n2
    abnormal = n1 + n2
    return E1ClassWeights(
        class_counts={"normal": n0, "mild": n1, "severe": n2},
        w_abnormal_neg=total / (2.0 * n0),
        w_abnormal_pos=total / (2.0 * abnormal),
        w_severe_neg=abnormal / (2.0 * n1),
        w_severe_pos=abnormal / (2.0 * n2),
    )


class HierarchicalConditionalWeightedBCELoss(nn.Module):
    """Weighted BCE where Normal samples are masked from the severity task."""

    def __init__(self, weights: E1ClassWeights, severity_loss_weight: float = 1.0) -> None:
        super().__init__()
        if float(severity_loss_weight) != 1.0:
            raise ValueError("E1 fixes severity_loss_weight to 1.0; do not tune it")
        self.severity_loss_weight = 1.0
        self.weights = weights

    def forward(
        self, outputs: dict[str, torch.Tensor], labels: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        logit_abnormal = outputs["logit_abnormal"]
        logit_severe_cond = outputs["logit_severe_cond"]
        if logit_abnormal.ndim != 2 or logit_abnormal.shape[1] != 1:
            raise ValueError(f"logit_abnormal must be [B,1], got {tuple(logit_abnormal.shape)}")
        if logit_severe_cond.shape != logit_abnormal.shape:
            raise ValueError("logit_severe_cond must have the same [B,1] shape")
        labels = labels.to(device=logit_abnormal.device, dtype=torch.long)
        if labels.ndim != 1 or labels.numel() != logit_abnormal.shape[0]:
            raise ValueError("labels must be [B] and align with both E1 logits")

        target_abnormal = (labels >= 1).float()
        target_severe = (labels == 2).float()
        severity_mask = labels >= 1
        raw_abnormal = F.binary_cross_entropy_with_logits(
            logit_abnormal.squeeze(1), target_abnormal, reduction="none"
        )
        abnormal_weight = torch.where(
            target_abnormal > 0.5,
            raw_abnormal.new_tensor(self.weights.w_abnormal_pos),
            raw_abnormal.new_tensor(self.weights.w_abnormal_neg),
        )
        loss_abnormal = (raw_abnormal * abnormal_weight).mean()
        if severity_mask.any():
            valid_logits = logit_severe_cond.squeeze(1)[severity_mask]
            valid_targets = target_severe[severity_mask]
            raw_severe = F.binary_cross_entropy_with_logits(
                valid_logits, valid_targets, reduction="none"
            )
            severe_weight = torch.where(
                valid_targets > 0.5,
                raw_severe.new_tensor(self.weights.w_severe_pos),
                raw_severe.new_tensor(self.weights.w_severe_neg),
            )
            loss_severe = (raw_severe * severe_weight).mean()
        else:
            # Preserves the computation graph while contributing exactly zero.
            loss_severe = logit_severe_cond.sum() * 0.0
        loss_total = loss_abnormal + loss_severe
        return {
            "loss_total": loss_total,
            "loss_abnormal": loss_abnormal,
            "loss_severe": loss_severe,
        }
