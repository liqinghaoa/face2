"""Label-free P0 QC contact sheets."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from .asset_io import save_png


def _overlay(image: np.ndarray, mask: np.ndarray, color: tuple[int, int, int]) -> np.ndarray:
    result = image.copy(); hit = mask > 0; result[hit] = np.clip(.55 * result[hit] + .45 * np.asarray(color), 0, 255).astype(np.uint8); return result


def save_qc(path: Path, artifacts: dict[str, Any]) -> None:
    """Save a no-label contact sheet containing only image/mask diagnostics."""
    import cv2
    aligned = artifacts["aligned"]; labels = artifacts["labels"]; colors = np.zeros_like(aligned); palette=np.array([[0,0,0],[210,160,130],[80,190,80],[70,120,210],[180,60,180],[255,200,40]],np.uint8); colors[:] = palette[np.asarray(labels)%len(palette)]
    panels = [artifacts["raw"], aligned, colors, _overlay(aligned, artifacts["final"], (255,80,30)), _overlay(aligned, artifacts["skin"], (30,220,80)), _overlay(aligned, artifacts["cheeks"], (80,120,255)), _overlay(aligned, artifacts["forehead_lip"], (255,220,40)), artifacts["existing"], artifacts["reconstructed"], cv2.cvtColor(artifacts["heat"],cv2.COLOR_GRAY2RGB)]
    resized = [cv2.resize(panel,(224,224),interpolation=cv2.INTER_AREA) for panel in panels]
    # Ten diagnostic panels are arranged as two equal five-panel rows.
    sheet=np.vstack([np.hstack(resized[:5]),np.hstack(resized[5:])]); save_png(path,sheet)
