"""Torch color-space utilities."""

from __future__ import annotations

import torch


XYZ_TO_LINEAR_SRGB = torch.tensor(
    [[3.2404542, -1.5371385, -0.4985314], [-0.9692660, 1.8760108, 0.0415560], [0.0556434, -0.2040259, 1.0572252]],
    dtype=torch.float64,
)
LINEAR_SRGB_TO_XYZ = torch.tensor(
    [[0.4124564, 0.3575761, 0.1804375], [0.2126729, 0.7151522, 0.0721750], [0.0193339, 0.1191920, 0.9503041]],
    dtype=torch.float64,
)
BRADFORD = torch.tensor(
    [[0.8951, 0.2664, -0.1614], [-0.7502, 1.7135, 0.0367], [0.0389, -0.0685, 1.0296]],
    dtype=torch.float64,
)


def srgb_encode(linear: torch.Tensor) -> torch.Tensor:
    """Apply exact sRGB OETF with negatives on the linear branch."""

    high = 1.055 * torch.pow(torch.clamp(linear, min=0.0031308), 1.0 / 2.4) - 0.055
    return torch.where(linear <= 0.0031308, 12.92 * linear, high)


def srgb_decode(srgb: torch.Tensor) -> torch.Tensor:
    """Apply exact inverse sRGB OETF."""

    base = torch.clamp((srgb + 0.055) / 1.055, min=0.0)
    return torch.where(srgb <= 0.04045, srgb / 12.92, torch.pow(base, 2.4))


def xyz_to_linear_srgb(xyz: torch.Tensor) -> torch.Tensor:
    """Convert XYZ to linear sRGB."""

    mat = XYZ_TO_LINEAR_SRGB.to(device=xyz.device, dtype=xyz.dtype)
    return xyz @ mat.T


def bradford_adapt(xyz: torch.Tensor, source_white: torch.Tensor, target_white: torch.Tensor) -> torch.Tensor:
    """Bradford adaptation in torch."""

    b = BRADFORD.to(device=xyz.device, dtype=xyz.dtype)
    b_inv = torch.linalg.inv(b)
    src = b @ source_white
    dst = b @ target_white
    transform = b_inv @ torch.diag(dst / src) @ b
    return xyz @ transform.T
