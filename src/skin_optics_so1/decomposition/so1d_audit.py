"""Pure analysis helpers for the read-only SO-1D-A diagnostic audit."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import numpy as np
import torch
from scipy.stats import rankdata
from sklearn.neighbors import NearestNeighbors


CHANNELS = ("M", "H", "S", "P")
ALLOWED_SPLITS = ("train", "validation")
FORBIDDEN_SPLITS = ("id_test", "camera_ood", "light_ood", "joint_ood")
AUDIT_SEED = 20260801


def require_audit_split(split: str) -> None:
    if split not in ALLOWED_SPLITS:
        raise ValueError(f"SO-1D-A forbids split {split!r}; allowed={ALLOWED_SPLITS}")


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def model_state_sha256(model: torch.nn.Module) -> str:
    digest = hashlib.sha256()
    for name, value in model.state_dict().items():
        digest.update(name.encode("utf-8"))
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def configure_read_only_fp32(model: torch.nn.Module) -> torch.nn.Module:
    model.float().eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model


def assert_fp32_eval_read_only(model: torch.nn.Module) -> None:
    if model.training:
        raise RuntimeError("SO-1D-A model must be in eval mode")
    for name, parameter in model.named_parameters():
        if parameter.requires_grad:
            raise RuntimeError(f"SO-1D-A parameter requires grad: {name}")
        if parameter.dtype != torch.float32:
            raise RuntimeError(f"SO-1D-A parameter is not FP32: {name}")


def pearson_from_sufficient(
    count: np.ndarray | float,
    sum_x: np.ndarray | float,
    sum_y: np.ndarray | float,
    sum_x2: np.ndarray | float,
    sum_y2: np.ndarray | float,
    sum_xy: np.ndarray | float,
) -> np.ndarray:
    n = np.asarray(count, dtype=np.float64)
    sx = np.asarray(sum_x, dtype=np.float64)
    sy = np.asarray(sum_y, dtype=np.float64)
    numerator = n * np.asarray(sum_xy, dtype=np.float64) - sx * sy
    denominator = np.sqrt(
        np.maximum(n * np.asarray(sum_x2, dtype=np.float64) - sx * sx, 0.0)
        * np.maximum(n * np.asarray(sum_y2, dtype=np.float64) - sy * sy, 0.0)
    )
    return np.divide(
        numerator, denominator, out=np.full_like(numerator, np.nan), where=denominator > 0
    )


def spearman_columns(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    if x.shape != y.shape or x.ndim != 2:
        raise ValueError("x and y must have matching [N,C] shape")
    result = np.empty(x.shape[1], dtype=np.float64)
    for channel in range(x.shape[1]):
        rx = rankdata(x[:, channel], method="average")
        ry = rankdata(y[:, channel], method="average")
        result[channel] = float(np.corrcoef(rx, ry)[0, 1])
    return result


def cross_talk_correlation(
    prediction: np.ndarray, target: np.ndarray, *, method: str
) -> np.ndarray:
    if prediction.shape != target.shape or prediction.ndim != 2 or prediction.shape[1] != 4:
        raise ValueError("prediction/target must have matching [N,4] shape in M/H/S/P order")
    matrix = np.empty((4, 4), dtype=np.float64)
    for pred_channel in range(4):
        for true_channel in range(4):
            x = prediction[:, pred_channel]
            y = target[:, true_channel]
            if method == "pearson":
                matrix[pred_channel, true_channel] = np.corrcoef(x, y)[0, 1]
            elif method == "spearman":
                matrix[pred_channel, true_channel] = np.corrcoef(
                    rankdata(x, method="average"), rankdata(y, method="average")
                )[0, 1]
            else:
                raise ValueError(f"Unsupported correlation method: {method}")
    return matrix


def masked_error_sums(
    prediction: torch.Tensor, target: torch.Tensor, mask: torch.Tensor
) -> dict[str, np.ndarray]:
    expanded = mask.expand_as(prediction).bool()
    difference = prediction.double() - target.double()
    count = mask.double().sum(dim=(1, 2, 3)).cpu().numpy()
    absolute = torch.where(expanded, difference.abs(), 0.0).sum(dim=(2, 3)).cpu().numpy()
    squared = torch.where(expanded, difference.square(), 0.0).sum(dim=(2, 3)).cpu().numpy()
    signed = torch.where(expanded, difference, 0.0).sum(dim=(2, 3)).cpu().numpy()
    return {"count": count, "absolute": absolute, "squared": squared, "signed": signed}


def p_active_counts(
    prediction: torch.Tensor, target: torch.Tensor, mask: torch.Tensor, threshold: float
) -> dict[str, np.ndarray]:
    pred = prediction[:, 3]
    true = target[:, 3]
    valid = mask[:, 0].bool()
    target_active = (true > threshold) & valid
    pred_active = (pred > threshold) & valid
    inactive = (~target_active) & valid
    difference = pred.double() - true.double()
    return {
        "valid": valid.sum(dim=(1, 2)).double().cpu().numpy(),
        "target_active": target_active.sum(dim=(1, 2)).double().cpu().numpy(),
        "pred_active": pred_active.sum(dim=(1, 2)).double().cpu().numpy(),
        "tp": (target_active & pred_active).sum(dim=(1, 2)).double().cpu().numpy(),
        "fp": ((~target_active) & pred_active & valid).sum(dim=(1, 2)).double().cpu().numpy(),
        "fn": (target_active & (~pred_active)).sum(dim=(1, 2)).double().cpu().numpy(),
        "active_abs": torch.where(target_active, difference.abs(), 0.0).sum(dim=(1, 2)).cpu().numpy(),
        "inactive_abs": torch.where(inactive, difference.abs(), 0.0).sum(dim=(1, 2)).cpu().numpy(),
        "active_signed": torch.where(target_active, difference, 0.0).sum(dim=(1, 2)).cpu().numpy(),
        "inactive_count": inactive.sum(dim=(1, 2)).double().cpu().numpy(),
    }


def summarize_active_counts(counts: dict[str, np.ndarray]) -> dict[str, float]:
    sums = {key: float(np.sum(value, dtype=np.float64)) for key, value in counts.items()}
    precision = sums["tp"] / (sums["tp"] + sums["fp"]) if sums["tp"] + sums["fp"] else np.nan
    recall = sums["tp"] / sums["target_active"] if sums["target_active"] else np.nan
    return {
        "target_active_prevalence": sums["target_active"] / sums["valid"],
        "pred_active_fraction": sums["pred_active"] / sums["valid"],
        "precision": precision,
        "recall": recall,
        "f1": 2 * precision * recall / (precision + recall) if precision + recall else np.nan,
        "active_region_MAE": sums["active_abs"] / sums["target_active"],
        "inactive_region_MAE": sums["inactive_abs"] / sums["inactive_count"],
        "active_region_bias": sums["active_signed"] / sums["target_active"],
    }


def zero_baseline_mae(target: np.ndarray, mask: np.ndarray) -> float:
    valid = np.asarray(mask, dtype=bool)
    values = np.asarray(target, dtype=np.float64)[valid]
    return float(np.mean(np.abs(values)))


def fit_train_affine(train_prediction: np.ndarray, train_target: np.ndarray) -> dict[str, float]:
    x = np.asarray(train_prediction, dtype=np.float64)
    y = np.asarray(train_target, dtype=np.float64)
    design = np.column_stack((x, np.ones_like(x)))
    slope, intercept = np.linalg.lstsq(design, y, rcond=None)[0]
    return {"a": float(slope), "b": float(intercept), "fit_source": "train_only"}


def find_rgb_neighbors(
    train_descriptors: np.ndarray, validation_descriptors: np.ndarray, *, top_k: int = 5
) -> tuple[np.ndarray, np.ndarray]:
    """Find descriptors only; target arrays are intentionally not accepted."""
    if top_k < 1:
        raise ValueError("top_k must be positive")
    train = np.asarray(train_descriptors, dtype=np.float32)
    validation = np.asarray(validation_descriptors, dtype=np.float32)
    model = NearestNeighbors(n_neighbors=top_k, metric="euclidean", algorithm="brute", n_jobs=-1)
    model.fit(train)
    distances, indices = model.kneighbors(validation, return_distance=True)
    return indices.astype(np.int32), distances.astype(np.float32)


def bootstrap_ci(values: np.ndarray, statistic: Any, *, replicates: int = 1000) -> list[float]:
    values = np.asarray(values)
    rng = np.random.default_rng(AUDIT_SEED)
    estimates = np.empty(replicates, dtype=np.float64)
    for index in range(replicates):
        sample = rng.integers(0, len(values), size=len(values))
        estimates[index] = statistic(values[sample])
    low, high = np.nanpercentile(estimates, [2.5, 97.5])
    return [float(low), float(high)]


def output_is_isolated(output_dir: Path, formal_dir: Path) -> bool:
    output = output_dir.resolve()
    formal = formal_dir.resolve()
    return output != formal and output not in formal.parents and formal not in output.parents
