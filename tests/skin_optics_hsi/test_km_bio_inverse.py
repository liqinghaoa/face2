from pathlib import Path

import numpy as np

from src.skin_optics_hsi.km_bio_inverse import FitSettings, acceptable_intervals, fit_bounded_spectrum, profile_parameter
from src.skin_optics_hsi.km_bio_v1 import WAVELENGTH_NM, forward_preloaded_numpy, load_optical_numpy


ASSET = Path("E:/projects/face2/data/external/SO0_Spectral_Assets_v1/processed/SO0_Spectral_Standardized_v1/production_10nm/derived_optics_10nm.npz")


def test_registered_multistart_recovers_noiseless_spectrum():
    optical = load_optical_numpy(WAVELENGTH_NM, ASSET)
    truth = np.array([0.17, 0.035, 0.68])
    observed = forward_preloaded_numpy(truth, optical)
    result = fit_bounded_spectrum(
        observed,
        lambda theta: forward_preloaded_numpy(theta, optical),
        np.zeros(3),
        np.array([0.43, 0.10, 1.0]),
        FitSettings(sobol_starts=8, max_nfev=500),
    )
    assert result["success"]
    assert len(result["starts"]) == 9
    assert result["metrics"]["logrmse"] < 1e-7
    np.testing.assert_allclose(result["theta"], truth, atol=2e-4)


def test_profile_keeps_noncontiguous_interval_structure():
    profile = [
        {"fixed_u": 0.0, "logrmse": 0.004, "success": True},
        {"fixed_u": 0.2, "logrmse": 0.020, "success": True},
        {"fixed_u": 0.8, "logrmse": 0.003, "success": True},
        {"fixed_u": 1.0, "logrmse": 0.030, "success": True},
    ]
    result = acceptable_intervals(profile, optimum=0.0, delta=0.005, scale=0.10)
    assert result["intervals"] == [[0.0, 0.0], [0.08000000000000002, 0.08000000000000002]]


def test_profile_runs_51_grid_points_plus_optimum():
    observed = np.exp(-np.array([0.2, 0.4, 0.6]))

    def forward_u(u):
        return np.exp(-np.array([u[0], u[1], 0.5 * (u[0] + u[1])]))

    settings = FitSettings(sobol_starts=8, max_nfev=200)
    profile, starts = profile_parameter(observed, forward_u, np.array([0.2, 0.4, 0.5]), 0, settings, 17)
    assert len(profile) >= 51
    assert len(starts) >= 51 * 9

