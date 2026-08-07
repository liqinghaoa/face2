"""NumPy/PyTorch parity audits."""

from __future__ import annotations

import numpy as np
import torch

from skin_optics.assets import SpectralAssets
from skin_optics.numpy_backend.colorchecker import solve_camera_to_xyz_matrix
from skin_optics.numpy_backend.reflectance import compute_skin_reflectance as np_reflectance
from skin_optics.torch_backend.forward_model import SO0TorchForwardModel


def audit_backend_parity(assets: SpectralAssets) -> dict[str, float]:
    """Compare NumPy and PyTorch reflectance on the same 5 nm grid."""

    m = np.linspace(0.0, 1.0, 13)
    h = np.linspace(0.0, 1.0, 13)
    mm, hh = np.meshgrid(m, h, indexing="ij")
    np_r = np_reflectance(assets, mm, hh)
    model64 = SO0TorchForwardModel(assets, matrix_3x3=solve_camera_to_xyz_matrix(assets, "Canon 5DMarkII", "D65")[0], dtype=torch.float64)
    with torch.no_grad():
        torch64 = model64.compute_skin_reflectance(torch.as_tensor(mm, dtype=torch.float64), torch.as_tensor(hh, dtype=torch.float64)).numpy()
    model32 = SO0TorchForwardModel(assets, dtype=torch.float32)
    with torch.no_grad():
        torch32 = model32.compute_skin_reflectance(torch.as_tensor(mm, dtype=torch.float32), torch.as_tensor(hh, dtype=torch.float32)).double().numpy()
    return {
        "numpy_torch_float64_reflectance_max_abs": float(np.max(np.abs(np_r - torch64))),
        "torch_float32_float64_reflectance_max_abs": float(np.max(np.abs(np_r - torch32))),
    }
