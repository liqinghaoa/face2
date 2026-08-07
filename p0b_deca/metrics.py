"""P0-coordinate residual and coverage metrics with no clinical interpretation."""
from __future__ import annotations
import numpy as np

def residual_p0(scene: np.ndarray, reconstruction: np.ndarray) -> tuple[np.ndarray,np.ndarray]:
    """Recompute signed and absolute residual only after mapping to P0 coordinates."""
    signed=np.asarray(scene,np.float32)-np.asarray(reconstruction,np.float32); return signed,np.abs(signed)

def coverage(mask: np.ndarray, visibility: np.ndarray) -> float:
    selected=np.asarray(mask)>0; return float((selected & (np.asarray(visibility)>0)).sum()/max(1,selected.sum()))

def finite_fraction(value: np.ndarray) -> float: return float(np.isfinite(value).mean())
