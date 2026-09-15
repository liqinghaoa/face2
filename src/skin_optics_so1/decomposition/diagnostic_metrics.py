"""Metrics used only by the SO-1C-R1 overfit failure diagnostic."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from .losses import CHANNEL_NAMES, SMOOTH_L1_BETA


def _masked_vectors(
    prediction: torch.Tensor, target: torch.Tensor, valid_mask: torch.Tensor
) -> tuple[np.ndarray, np.ndarray]:
    if prediction.shape != target.shape or prediction.ndim != 4 or prediction.shape[1] != 4:
        raise ValueError("prediction and target must have shape [N,4,H,W]")
    if valid_mask.shape != (prediction.shape[0], 1, prediction.shape[2], prediction.shape[3]):
        raise ValueError("valid_mask must have shape [N,1,H,W]")
    keep = valid_mask.detach().cpu().numpy().astype(bool)[:, 0]
    pred = prediction.detach().float().cpu().numpy()
    true = target.detach().float().cpu().numpy()
    return pred.transpose(1, 0, 2, 3)[:, keep], true.transpose(1, 0, 2, 3)[:, keep]


def channel_summary(values: np.ndarray) -> dict[str, float]:
    if values.size == 0:
        raise ValueError("cannot summarize an empty vector")
    percentile = np.percentile(values, [1, 5, 50, 95, 99])
    return {
        "min": float(values.min()), "max": float(values.max()), "mean": float(values.mean()),
        "std": float(values.std()), "p01": float(percentile[0]), "p05": float(percentile[1]),
        "p50": float(percentile[2]), "p95": float(percentile[3]), "p99": float(percentile[4]),
    }


def diagnostic_metrics(
    prediction: torch.Tensor, target: torch.Tensor, valid_mask: torch.Tensor, *, p_active_threshold: float = 0.01
) -> dict[str, Any]:
    """Return JSON-ready channel diagnostics over only valid pixels."""
    pred, true = _masked_vectors(prediction, target, valid_mask)
    result: dict[str, Any] = {"p_active_threshold": float(p_active_threshold), "channels": {}}
    for index, name in enumerate(CHANNEL_NAMES):
        error = np.abs(pred[index] - true[index])
        active = true[index] > p_active_threshold if name == "P" else np.ones_like(true[index], dtype=bool)
        channel: dict[str, Any] = {
            "MAE": float(error.mean()),
            "SmoothL1": float(F.smooth_l1_loss(
                torch.from_numpy(pred[index]), torch.from_numpy(true[index]), reduction="mean", beta=SMOOTH_L1_BETA
            ).item()),
            "true": channel_summary(true[index]),
            "pred": channel_summary(pred[index]),
            "pred_le_0_01_fraction": float((pred[index] <= 0.01).mean()),
            "pred_ge_0_99_fraction": float((pred[index] >= 0.99).mean()),
            "pred_to_true_std_ratio": float(pred[index].std() / true[index].std())
            if float(true[index].std()) > 1.0e-12 else float("nan"),
        }
        if name == "P":
            channel.update({
                "true_nonzero_fraction": float((true[index] > 0.0).mean()),
                "pred_nonzero_fraction": float((pred[index] > 0.0).mean()),
                "pred_p99": float(np.percentile(pred[index], 99)),
                "pred_max": float(pred[index].max()),
                "active_region_MAE": float(error[active].mean()) if active.any() else float("nan"),
                "active_region_recall": float((pred[index][active] > p_active_threshold).mean()) if active.any() else float("nan"),
                "active_pixel_count": int(active.sum()),
            })
        result["channels"][name] = channel
    return result


def constant_baseline(prediction: torch.Tensor, target: torch.Tensor, valid_mask: torch.Tensor) -> dict[str, Any]:
    pred, true = _masked_vectors(prediction, target, valid_mask)
    result: dict[str, Any] = {}
    for index, name in enumerate(CHANNEL_NAMES):
        constant = float(true[index].mean())
        baseline_mae = float(np.abs(true[index] - constant).mean())
        model_mae = float(np.abs(pred[index] - true[index]).mean())
        result[name] = {
            "constant": constant,
            "constant_MAE": baseline_mae,
            "model_MAE": model_mae,
            "relative_improvement": float(1.0 - model_mae / baseline_mae) if baseline_mae > 1.0e-12 else float("nan"),
        }
    return result


def cross_channel_matrices(
    prediction: torch.Tensor, target: torch.Tensor, valid_mask: torch.Tensor
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    pred, true = _masked_vectors(prediction, target, valid_mask)
    mae = np.empty((4, 4), dtype=np.float64)
    pearson = np.full((4, 4), np.nan, dtype=np.float64)
    counts = np.zeros((4, 4), dtype=np.int64)
    for i in range(4):
        for j in range(4):
            mae[i, j] = np.abs(pred[i] - true[j]).mean()
            usable = np.isfinite(pred[i]) & np.isfinite(true[j])
            counts[i, j] = int(usable.sum())
            if usable.sum() > 1 and np.std(pred[i][usable]) > 1.0e-12 and np.std(true[j][usable]) > 1.0e-12:
                pearson[i, j] = float(np.corrcoef(pred[i][usable], true[j][usable])[0, 1])
    return mae, pearson, counts


def pair_prediction_drift(prediction: torch.Tensor, valid_mask: torch.Tensor) -> dict[str, float]:
    if prediction.shape[0] != 2:
        raise ValueError("pair drift requires exactly two samples")
    common = (valid_mask[0, 0] > 0) & (valid_mask[1, 0] > 0)
    if not bool(common.any()):
        raise ValueError("pair has no common valid pixels")
    return {
        f"{name}_pair_prediction_drift": float(
            torch.abs(prediction[0, index][common] - prediction[1, index][common]).mean().detach().cpu()
        )
        for index, name in enumerate(CHANNEL_NAMES)
    }


def final_output_gradient_norms(model: torch.nn.Module) -> dict[str, float]:
    out_layer = getattr(model, "outc", None)
    if not isinstance(out_layer, torch.nn.Conv2d) or out_layer.weight.grad is None:
        raise ValueError("model.outc Conv2d gradients are required")
    return {
        f"{name}_output_gradient_norm": float(out_layer.weight.grad[index].detach().norm().cpu())
        for index, name in enumerate(CHANNEL_NAMES)
    }
