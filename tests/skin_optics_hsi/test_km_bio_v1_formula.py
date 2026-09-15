import numpy as np
import pytest
import torch

from src.skin_optics_hsi.km_bio_v1 import (
    WAVELENGTH_NM,
    _km_layer_numpy,
    forward_numpy,
    forward_preloaded_numpy,
    forward_torch,
    load_optical_numpy,
)


ASSET = "E:/projects/face2/data/external/SO0_Spectral_Assets_v1/processed/SO0_Spectral_Standardized_v1/production_10nm/derived_optics_10nm.npz"


def test_finite_layer_limits():
    # d -> 0, pure absorption, and pure scattering limits.
    r, t = _km_layer_numpy(np.array([1.0, 1.0, 0.0]), np.array([2.0, 0.0, 2.0]), 0.060)
    assert np.isclose(r[1], 0.0)
    assert np.isclose(t[1], np.exp(-0.060))
    assert np.isclose(r[2], 0.12 / 1.12)
    assert np.isclose(t[2], 1.0 / 1.12)
    r0, t0 = _km_layer_numpy(np.array([1.0]), np.array([2.0]), 0.0)
    assert np.isclose(r0[0], 0.0) and np.isclose(t0[0], 1.0)


def test_random_physical_bounds_and_finite():
    rng = np.random.default_rng(20260909)
    k = 10.0 ** rng.uniform(-3, 2, 10000)
    s = 10.0 ** rng.uniform(-3, 2, 10000)
    r, t = _km_layer_numpy(k, s, 0.060)
    assert np.all(np.isfinite(r)) and np.all(np.isfinite(t))
    assert np.min(r) >= -1e-12 and np.min(t) >= -1e-12
    assert np.max(r + t) <= 1.0 + 1e-10


def test_forward_numpy_and_torch_parity_and_gradient():
    theta = np.array([0.18, 0.025, 0.72], dtype=np.float64)
    np_out = forward_numpy(theta, WAVELENGTH_NM, ASSET)
    with np.load(ASSET, allow_pickle=False) as z:
        w = z["wavelength_nm"]
        optical = {k: torch.as_tensor(np.interp(WAVELENGTH_NM, w, z[field] / 10.0), dtype=torch.float64)
                   for k, field in [("mua_mel", "mua_mel_alternative_cm1"), ("mua_bg", "mua_base_cm1"),
                                    ("mua_hbo2", "mua_hbo2_whole_blood_150gL_cm1"), ("mua_hb", "mua_hb_whole_blood_150gL_cm1"),
                                    ("musp", "musp_total_cm1")]}
    theta_t = torch.tensor(theta, dtype=torch.float64, requires_grad=True)
    torch_out = forward_torch(theta_t, optical)
    assert np.max(np.abs(np_out - torch_out.detach().numpy())) < 1e-10
    torch_out.sum().backward()
    assert torch.all(torch.isfinite(theta_t.grad))


def test_forward_rejects_out_of_bounds():
    with pytest.raises(ValueError):
        forward_numpy(np.array([0.44, 0.02, 0.5]), WAVELENGTH_NM, ASSET)


def test_preloaded_forward_matches_file_path_and_supports_fixed_sensitivities():
    theta = np.array([0.18, 0.025, 0.72], dtype=np.float64)
    optical = load_optical_numpy(WAVELENGTH_NM, ASSET)
    expected = forward_numpy(theta, WAVELENGTH_NM, ASSET)
    np.testing.assert_allclose(forward_preloaded_numpy(theta, optical), expected, atol=1e-14, rtol=0)
    variants = [
        forward_preloaded_numpy(theta, optical, epidermis_thickness_mm=0.050),
        forward_preloaded_numpy(theta, optical, scattering_scale=0.8),
        forward_preloaded_numpy(theta, optical, whole_blood_hb_g_l=120.0),
    ]
    assert all(np.isfinite(value).all() and np.max(np.abs(value - expected)) > 0 for value in variants)
