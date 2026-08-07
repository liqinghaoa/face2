import numpy as np
import torch

from skin_optics.numpy_backend.reflectance import compute_skin_reflectance, compute_skin_reflectance_bchw
from skin_optics.torch_backend.forward_model import SO0TorchForwardModel


def test_numpy_batch_shapes(assets5):
    assert compute_skin_reflectance(assets5, 0.5, 0.5).shape == (65,)
    assert compute_skin_reflectance(assets5, np.array([0.4, 0.6]), np.array([0.5, 0.5])).shape == (2, 65)
    m = np.ones((2, 3, 4)) * 0.5
    h = np.ones((2, 3, 4)) * 0.5
    assert compute_skin_reflectance(assets5, m, h).shape == (2, 3, 4, 65)
    mb = np.ones((2, 1, 3, 4)) * 0.5
    hb = np.ones((2, 1, 3, 4)) * 0.5
    assert compute_skin_reflectance_bchw(assets5, mb, hb).shape == (2, 65, 3, 4)


def test_torch_bchw_wrapper(assets5):
    model = SO0TorchForwardModel(assets5, dtype=torch.float32)
    m = torch.ones((2, 1, 3, 4)) * 0.5
    h = torch.ones((2, 1, 3, 4)) * 0.5
    assert model.compute_skin_reflectance(m, h).shape == (2, 1, 3, 4, 65)
    assert model.forward_bchw(m, h).shape == (2, 3, 3, 4)
