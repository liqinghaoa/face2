"""Shared types for the SO-0 forward model."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray


FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class RenderResult:
    """Container for CIE reference rendering outputs."""

    xyz_source: FloatArray
    xyz_d65: FloatArray
    linear_srgb_unclipped: FloatArray
    srgb_unclipped: FloatArray
    srgb_display_clipped: FloatArray


@dataclass(frozen=True)
class CameraRenderResult:
    """Container for camera rendering outputs."""

    camera_rgb_raw: FloatArray
    camera_rgb_wb: FloatArray
    xyz_source: FloatArray
    xyz_d65: FloatArray
    linear_srgb_unclipped: FloatArray
    srgb_unclipped: FloatArray
    srgb_display_clipped: FloatArray


@dataclass(frozen=True)
class PathConfig:
    """Resolved project paths."""

    workspace_root: Path
    standardized_asset_root: Path
    output_dir: Path


ConfigDict = dict[str, Any]
