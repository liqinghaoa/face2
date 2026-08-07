"""Color-space and chromatic-adaptation utilities."""

from __future__ import annotations

import numpy as np


XYZ_TO_LINEAR_SRGB = np.array(
    [[3.2404542, -1.5371385, -0.4985314], [-0.9692660, 1.8760108, 0.0415560], [0.0556434, -0.2040259, 1.0572252]],
    dtype=np.float64,
)
LINEAR_SRGB_TO_XYZ = np.array(
    [[0.4124564, 0.3575761, 0.1804375], [0.2126729, 0.7151522, 0.0721750], [0.0193339, 0.1191920, 0.9503041]],
    dtype=np.float64,
)
BRADFORD = np.array(
    [[0.8951, 0.2664, -0.1614], [-0.7502, 1.7135, 0.0367], [0.0389, -0.0685, 1.0296]],
    dtype=np.float64,
)
BRADFORD_INV = np.linalg.inv(BRADFORD)


def srgb_encode(linear: np.ndarray) -> np.ndarray:
    """Apply the exact piecewise sRGB OETF, with negatives on the linear branch."""

    linear = np.asarray(linear, dtype=np.float64)
    out = np.empty_like(linear, dtype=np.float64)
    low = linear <= 0.0031308
    out[low] = 12.92 * linear[low]
    out[~low] = 1.055 * np.power(linear[~low], 1.0 / 2.4) - 0.055
    return out


def srgb_decode(srgb: np.ndarray) -> np.ndarray:
    """Apply the exact piecewise inverse sRGB OETF."""

    srgb = np.asarray(srgb, dtype=np.float64)
    out = np.empty_like(srgb, dtype=np.float64)
    low = srgb <= 0.04045
    out[low] = srgb[low] / 12.92
    out[~low] = np.power((srgb[~low] + 0.055) / 1.055, 2.4)
    return out


def xyz_to_linear_srgb(xyz: np.ndarray) -> np.ndarray:
    """Convert XYZ to linear sRGB."""

    return np.asarray(xyz, dtype=np.float64) @ XYZ_TO_LINEAR_SRGB.T


def linear_srgb_to_xyz(rgb: np.ndarray) -> np.ndarray:
    """Convert linear sRGB to XYZ."""

    return np.asarray(rgb, dtype=np.float64) @ LINEAR_SRGB_TO_XYZ.T


def bradford_adapt(xyz: np.ndarray, source_white: np.ndarray, target_white: np.ndarray) -> np.ndarray:
    """Bradford-adapt XYZ values from source white to target white."""

    xyz = np.asarray(xyz, dtype=np.float64)
    source_white = np.asarray(source_white, dtype=np.float64)
    target_white = np.asarray(target_white, dtype=np.float64)
    src_cone = BRADFORD @ source_white
    dst_cone = BRADFORD @ target_white
    if np.any(np.abs(src_cone) < 1e-14):
        raise ValueError("Source white has near-zero Bradford cone response")
    transform = BRADFORD_INV @ np.diag(dst_cone / src_cone) @ BRADFORD
    return xyz @ transform.T


def xyz_to_lab_d65(xyz: np.ndarray, white: np.ndarray | None = None) -> np.ndarray:
    """Convert XYZ to CIELAB under D65 using the standard formula."""

    white = np.asarray(white if white is not None else [0.95047, 1.0, 1.08883], dtype=np.float64)
    xyz_n = np.asarray(xyz, dtype=np.float64) / white
    delta = 6.0 / 29.0
    f = np.where(xyz_n > delta**3, np.cbrt(xyz_n), xyz_n / (3.0 * delta**2) + 4.0 / 29.0)
    lstar = 116.0 * f[..., 1] - 16.0
    astar = 500.0 * (f[..., 0] - f[..., 1])
    bstar = 200.0 * (f[..., 1] - f[..., 2])
    return np.stack([lstar, astar, bstar], axis=-1)


def deltae00(lab1: np.ndarray, lab2: np.ndarray) -> np.ndarray:
    """Compute CIEDE2000 with colour-science."""

    import colour

    return np.asarray(colour.delta_E(np.asarray(lab1), np.asarray(lab2), method="CIE 2000"), dtype=np.float64)
