"""P0-coordinate raster mapping helpers with correct interpolation semantics."""
from __future__ import annotations
import cv2
import numpy as np


def map_to_p0(value: np.ndarray, deca_to_p0: np.ndarray, interpolation: str, output_size: int = 224) -> np.ndarray:
    """Map a DECA-space raster to P0 224-space with an explicit interpolation rule."""
    methods = {"bilinear": cv2.INTER_LINEAR, "nearest": cv2.INTER_NEAREST}
    if interpolation not in methods:
        raise ValueError("interpolation must be bilinear or nearest")
    matrix = np.asarray(deca_to_p0, dtype=np.float32)
    if matrix.shape != (3, 3):
        raise ValueError("deca_to_p0 must be 3x3")
    return cv2.warpPerspective(np.asarray(value), matrix, (output_size, output_size), flags=methods[interpolation], borderMode=cv2.BORDER_CONSTANT, borderValue=0)


def map_normal_to_p0(normal: np.ndarray) -> np.ndarray:
    """Renormalize bilinearly mapped normals; zero vectors remain zero."""
    value=np.asarray(normal,np.float32); length=np.linalg.norm(value,axis=-1,keepdims=True); return np.divide(value,length,out=np.zeros_like(value),where=length>1e-8)
