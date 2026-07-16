"""Deterministic masked optical channels for the P2-1 experiment."""

from __future__ import annotations

from typing import Sequence

import cv2
import numpy as np
import torch


CHANNEL_NAMES = (
    "chrom_r",
    "chrom_g",
    "chrom_b",
    "log_rg",
    "log_bg",
    "lab_a",
    "lab_b",
)


def _as_numpy_rgb(rgb: np.ndarray | torch.Tensor) -> tuple[np.ndarray, bool, torch.device | None]:
    is_tensor = torch.is_tensor(rgb)
    device = rgb.device if is_tensor else None
    value = rgb.detach().cpu().numpy() if is_tensor else np.asarray(rgb)
    if value.ndim == 3:
        value = value[None, ...]
        squeeze = True
    elif value.ndim == 4:
        squeeze = False
    else:
        raise ValueError(f"rgb must be [3,H,W] or [B,3,H,W], got {value.shape}")
    if value.shape[1] != 3:
        raise ValueError(f"rgb channel dimension must be 3, got {value.shape}")
    return value.astype(np.float32, copy=False), squeeze, device


def _as_numpy_mask(mask: np.ndarray | torch.Tensor, batch: int, height: int, width: int) -> np.ndarray:
    value = mask.detach().cpu().numpy() if torch.is_tensor(mask) else np.asarray(mask)
    if value.ndim == 2:
        value = value[None, None, ...]
    elif value.ndim == 3:
        if value.shape[0] == batch and value.shape[1:] == (height, width):
            value = value[:, None, ...]
        elif batch == 1 and value.shape[0] == 1:
            value = value[None, ...]
        else:
            raise ValueError(f"ambiguous mask shape {value.shape}")
    elif value.ndim != 4:
        raise ValueError(f"mask must be [H,W], [1,H,W], [B,H,W], or [B,1,H,W], got {value.shape}")
    if value.shape != (batch, 1, height, width):
        raise ValueError(f"mask shape mismatch: expected {(batch,1,height,width)}, got {value.shape}")
    return value > 0


def build_relative_optical_channels(
    rgb: np.ndarray | torch.Tensor,
    valid_mask: np.ndarray | torch.Tensor,
    epsilon: float = 1.0e-4,
) -> np.ndarray | torch.Tensor:
    """Build [r,g,b,log(R/G),log(B/G),Lab-a,Lab-b] from raw RGB [0,1].

    OpenCV float32 ``COLOR_RGB2LAB`` is used, so a* and b* are already signed
    physical Lab coordinates and do not carry the uint8 +128 offset.
    """
    if epsilon <= 0:
        raise ValueError("epsilon must be positive")
    value, squeeze, device = _as_numpy_rgb(rgb)
    if not np.isfinite(value).all() or value.min() < 0.0 or value.max() > 1.0:
        raise ValueError("rgb must contain finite raw values in [0,1]")
    batch, _, height, width = value.shape
    mask = _as_numpy_mask(valid_mask, batch, height, width)
    output = np.zeros((batch, 7, height, width), dtype=np.float32)
    for index in range(batch):
        r, g, b = value[index]
        total = r + g + b + float(epsilon)
        lab = cv2.cvtColor(
            np.transpose(value[index], (1, 2, 0)).astype(np.float32),
            cv2.COLOR_RGB2LAB,
        )
        channels = np.stack(
            [
                r / total,
                g / total,
                b / total,
                np.log((r + epsilon) / (g + epsilon)),
                np.log((b + epsilon) / (g + epsilon)),
                lab[..., 1],
                lab[..., 2],
            ],
            axis=0,
        ).astype(np.float32)
        channels[:, ~mask[index, 0]] = 0.0
        output[index] = channels
    if not np.isfinite(output).all():
        raise FloatingPointError("optical channels contain NaN or Inf")
    output = output[0] if squeeze else output
    if torch.is_tensor(rgb):
        return torch.from_numpy(output).to(device=device, dtype=rgb.dtype)
    return output


def normalize_optical_channels(
    channels: np.ndarray | torch.Tensor,
    valid_mask: np.ndarray | torch.Tensor,
    mean: Sequence[float],
    std: Sequence[float],
    epsilon: float = 1.0e-4,
) -> np.ndarray | torch.Tensor:
    """Apply fold-train statistics and restore an exact zero background."""
    is_tensor = torch.is_tensor(channels)
    device = channels.device if is_tensor else None
    dtype = channels.dtype if is_tensor else None
    value = channels.detach().cpu().numpy() if is_tensor else np.asarray(channels)
    squeeze = value.ndim == 3
    if squeeze:
        value = value[None]
    if value.ndim != 4 or value.shape[1] != 7:
        raise ValueError(f"channels must be [7,H,W] or [B,7,H,W], got {value.shape}")
    mask = _as_numpy_mask(valid_mask, value.shape[0], value.shape[2], value.shape[3])
    means = np.asarray(mean, dtype=np.float32).reshape(1, 7, 1, 1)
    stds = np.asarray(std, dtype=np.float32).reshape(1, 7, 1, 1)
    if means.size != 7 or stds.size != 7 or np.any(stds < 0):
        raise ValueError("mean/std must contain seven valid values")
    result = (value.astype(np.float32) - means) / (stds + float(epsilon))
    result *= mask.astype(np.float32)
    if not np.isfinite(result).all():
        raise FloatingPointError("normalized optical channels contain NaN or Inf")
    result = result[0] if squeeze else result
    if is_tensor:
        return torch.from_numpy(result).to(device=device, dtype=dtype)
    return result
