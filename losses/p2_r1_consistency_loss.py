"""Numerically stable Jensen--Shannon prediction consistency for P2-R1."""
from __future__ import annotations
import torch

def js_divergence(raw_logits: torch.Tensor, relight_logits: torch.Tensor, eps: float = 1e-7) -> torch.Tensor:
    if raw_logits.shape != relight_logits.shape:
        raise ValueError("raw and relight logits must have the same shape")
    p = torch.softmax(raw_logits, 1).clamp_min(eps); q = torch.softmax(relight_logits, 1).clamp_min(eps); m = .5 * (p + q)
    return .5 * ((p * (p.log() - m.log())).sum(1) + (q * (q.log() - m.log())).sum(1)).mean()

def consistency_weight(epoch_index: int, start_epoch_one_based: int = 4, weight: float = .2) -> float:
    return 0.0 if int(epoch_index) < int(start_epoch_one_based) - 1 else float(weight)
