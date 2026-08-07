import numpy as np
import pytest

from skin_optics.numpy_backend.reflectance import compute_skin_reflectance


def test_reflectance_bounds_and_shapes(assets5):
    m = np.zeros((2, 3))
    h = np.ones((2, 3)) * 0.5
    r = compute_skin_reflectance(assets5, m, h)
    assert r.shape == (2, 3, 65)
    assert np.all(np.isfinite(r))
    assert r.min() >= -1e-12
    assert r.max() <= 1 + 1e-12


def test_reflectance_rejects_out_of_range(assets5):
    with pytest.raises(ValueError):
        compute_skin_reflectance(assets5, -0.1, 0.5)
