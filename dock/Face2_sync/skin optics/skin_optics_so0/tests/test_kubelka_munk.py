import numpy as np

from skin_optics.numpy_backend.kubelka_munk import finite_dermis_reflectance


def test_kubelka_munk_limits():
    mus = np.array([0.0, 10.0])
    mua = np.array([0.0, 0.0])
    r = finite_dermis_reflectance(mua, mus, 0.2)
    assert r[0] == 0.0
    assert np.isclose(r[1], 2.0 / 3.0)


def test_kubelka_munk_finite(assets5):
    mua = assets5.derived["mua_blood_y075_cm1"]
    mus = assets5.derived["musp_total_cm1"]
    r = finite_dermis_reflectance(mua, mus, 0.2)
    assert np.all(np.isfinite(r))
    assert np.all((r >= 0) & (r <= 1))
