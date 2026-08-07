"""Fixed-light reflectance-like rendering arithmetic; no residual compensation."""
from __future__ import annotations
import numpy as np
def relight(albedo_like: np.ndarray, shading_like: np.ndarray, alpha: np.ndarray, aligned_scene: np.ndarray) -> tuple[np.ndarray,np.ndarray]:
    """Return raw unclipped face-only and original-scene composite floats."""
    face=np.asarray(albedo_like,np.float32)*np.asarray(shading_like,np.float32); a=np.asarray(alpha,np.float32)[...,None]; return face,a*face+(1-a)*np.asarray(aligned_scene,np.float32)
