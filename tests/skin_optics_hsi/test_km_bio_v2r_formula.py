import numpy as np
import pytest
import torch

from skin_optics_hsi.km_bio_v1 import forward_preloaded_numpy as v1_forward
from skin_optics_hsi.km_bio_v2r import (
    AS_BOUNDS,
    DELTA_BS_BOUNDS,
    G0_BOUNDS,
    WAVELENGTH_NM,
    _km_layer_numpy,
    blood_packaging_factor_numpy,
    estimate_global_gain,
    estimate_shape_scales,
    forward_preloaded_numpy,
    forward_torch,
    load_optical_numpy,
    observation_scale_audit,
    profile_identifiability,
)


ASSET = "E:/projects/face2/data/external/SO0_Spectral_Assets_v1/processed/SO0_Spectral_Standardized_v1/production_10nm/derived_optics_10nm.npz"


def _optical():
    return load_optical_numpy(WAVELENGTH_NM, ASSET)


def test_packaging_zero_limit_and_unit_conversion():
    mu = np.array([0.0, 1.0, 100.0])
    assert np.array_equal(blood_packaging_factor_numpy(mu, 0.0), np.ones(3))
    # 15 um = 0.015 mm; packaging is a dimensionless factor in (0, 1].
    packed = blood_packaging_factor_numpy(np.array([10.0]), 15.0)[0]
    expected = -np.expm1(-10.0 * 0.015) / (10.0 * 0.015)
    assert np.isclose(packed, expected)


def test_v2r_zero_packaging_recovers_v1_and_two_parameter_s0():
    optical = _optical()
    theta = np.array([0.18, 0.025, 0.72])
    v1 = v1_forward(theta, optical)
    v2 = forward_preloaded_numpy(theta, optical, wavelength_nm=WAVELENGTH_NM, diameter_um=0.0)
    base = forward_preloaded_numpy(theta[:2], optical, wavelength_nm=WAVELENGTH_NM, s0=theta[2], diameter_um=0.0)
    np.testing.assert_allclose(v2, v1, atol=1e-12, rtol=0)
    np.testing.assert_allclose(base, v2, atol=1e-12, rtol=0)


def test_scattering_and_gain_identity_and_bounds():
    optical = _optical()
    theta = np.array([0.2, 0.03, 0.65])
    expected = forward_preloaded_numpy(theta, optical, wavelength_nm=WAVELENGTH_NM)
    np.testing.assert_array_equal(expected, forward_preloaded_numpy(theta, optical, wavelength_nm=WAVELENGTH_NM, scattering_amplitude=1.0, delta_bs=0.0, g0=1.0))
    with pytest.raises(ValueError):
        forward_preloaded_numpy(theta, optical, wavelength_nm=WAVELENGTH_NM, scattering_amplitude=AS_BOUNDS[0] - 0.01)
    with pytest.raises(ValueError):
        forward_preloaded_numpy(theta, optical, wavelength_nm=WAVELENGTH_NM, delta_bs=DELTA_BS_BOUNDS[1] + 0.01)
    with pytest.raises(ValueError):
        forward_preloaded_numpy(theta, optical, wavelength_nm=WAVELENGTH_NM, g0=G0_BOUNDS[0] - 0.01)
    high_gain = forward_preloaded_numpy(np.array([0.0, 0.0]), optical, wavelength_nm=WAVELENGTH_NM, g0=1.5)
    scale_audit = observation_scale_audit(high_gain)
    assert not scale_audit["pass"] and scale_audit["above_one_count"] > 0 and not scale_audit["clipped"]


def test_numpy_torch_parity_and_gradient():
    optical = _optical()
    optical_t = {key: torch.as_tensor(value, dtype=torch.float64) for key, value in optical.items()}
    theta = np.array([0.18, 0.025, 0.72], dtype=np.float64)
    np_out = forward_preloaded_numpy(theta, optical, wavelength_nm=WAVELENGTH_NM, diameter_um=15.0, scattering_amplitude=1.1, delta_bs=0.12, g0=0.97)
    theta_t = torch.tensor(theta, dtype=torch.float64, requires_grad=True)
    torch_out = forward_torch(theta_t, optical_t, wavelength_nm=WAVELENGTH_NM, diameter_um=15.0, scattering_amplitude=1.1, delta_bs=0.12, g0=0.97)
    np.testing.assert_allclose(np_out, torch_out.detach().numpy(), atol=1e-10, rtol=0)
    torch_out.sum().backward()
    assert torch.isfinite(theta_t.grad).all()
    theta_zero = torch.tensor(theta, dtype=torch.float64, requires_grad=True)
    forward_torch(theta_zero, optical_t, wavelength_nm=WAVELENGTH_NM, diameter_um=0.0).sum().backward()
    assert torch.isfinite(theta_zero.grad).all()


def test_km_layer_physical_random_bounds():
    rng = np.random.default_rng(20260910)
    k = 10 ** rng.uniform(-3, 2, 2000)
    s = 10 ** rng.uniform(-3, 2, 2000)
    r, t = _km_layer_numpy(k, s, 0.060)
    assert np.isfinite(r).all() and np.isfinite(t).all()
    assert r.min() >= -1e-12 and t.min() >= -1e-12 and (r + t).max() <= 1 + 1e-10


def test_sequential_global_scale_interfaces():
    optical = _optical()
    theta_set = [np.array([0.18, 0.025, 0.72]), np.array([0.28, 0.04, 0.62])]
    amplitudes = np.array([1.0, 1.1])
    deltas = np.array([0.0, 0.12])
    candidates = np.asarray([
        [[forward_preloaded_numpy(theta, optical, wavelength_nm=WAVELENGTH_NM, scattering_amplitude=amplitude, delta_bs=delta) for theta in theta_set] for delta in deltas]
        for amplitude in amplitudes
    ])
    observed = candidates[1, 1]
    result = estimate_shape_scales(observed, candidates, amplitudes, deltas)
    assert result["A_s"] == 1.1 and result["delta_bs"] == 0.12
    gain = estimate_global_gain(observed, candidates[0, 0])
    assert G0_BOUNDS[0] <= gain["g0"] <= G0_BOUNDS[1]
    assert gain["estimation"] == "analytic_bounded_raw_logrmse" and len(gain["profile"]) >= 201


def test_profile_identifiability_rejects_wide_platform():
    narrow = [{"g0": value, "raw_logrmse": abs(value - 1.0)} for value in np.linspace(0.5, 1.5, 101)]
    wide = [{"g0": value, "raw_logrmse": 0.0} for value in np.linspace(0.5, 1.5, 101)]
    assert profile_identifiability(narrow, "g0", "raw_logrmse", G0_BOUNDS)["identifiable"]
    assert not profile_identifiability(wide, "g0", "raw_logrmse", G0_BOUNDS)["identifiable"]
