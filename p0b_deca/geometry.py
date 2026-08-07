"""Geometry checks independent of a particular DECA release."""
from __future__ import annotations
import numpy as np
def normal_unit_length_error(normals: np.ndarray, valid: np.ndarray | None=None) -> float:
    lengths=np.linalg.norm(np.asarray(normals),axis=-1); values=lengths[np.asarray(valid)>0] if valid is not None else lengths.reshape(-1); return float(np.mean(np.abs(values-1))) if values.size else float('nan')
