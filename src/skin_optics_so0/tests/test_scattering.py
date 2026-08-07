import numpy as np

from skin_optics.numpy_backend.scattering import reduced_scattering


def test_scattering_positive(assets5):
    mus = reduced_scattering(assets5)
    assert mus.shape == (65,)
    assert np.all(np.isfinite(mus))
    assert np.all(mus > 0)
