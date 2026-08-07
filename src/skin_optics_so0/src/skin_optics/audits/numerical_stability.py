"""Numerical stability audits."""

from __future__ import annotations

import numpy as np

from skin_optics.assets import SpectralAssets
from skin_optics.numpy_backend.reflectance import compute_skin_reflectance


def audit_numerical_stability(assets: SpectralAssets) -> dict[str, float]:
    """Evaluate finite state and reflectance bounds on parameter corners."""

    vals = []
    for m in [0.0, 1.0]:
        for h in [0.0, 1.0]:
            vals.append(compute_skin_reflectance(assets, m, h))
    arr = np.stack(vals)
    return {"finite_rate": float(np.mean(np.isfinite(arr))), "min_reflectance": float(arr.min()), "max_reflectance": float(arr.max())}
