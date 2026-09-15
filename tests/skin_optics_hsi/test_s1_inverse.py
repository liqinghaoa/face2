from __future__ import annotations

import numpy as np
import pytest

from skin_optics_hsi.model_registry import load_model_registry
from skin_optics_hsi.s1_inverse import fit_registered_spectrum, profile_registered_spectrum
from skin_optics_hsi.skin_forward import RegisteredSkinForwardModel


FIT = {
    "reflectance_epsilon": 1e-6,
    "minimum_valid_bands": 24,
    "log_pseudo_huber_delta": 0.05,
    "sam_weight": 0.25,
    "boundary_fraction": 0.01,
    "solution_cluster_normalized_distance": 0.03,
    "jacobian_relative_step": 1e-4,
    "max_function_evaluations": 180,
    "convergence_xtol": 1e-10,
    "convergence_ftol": 1e-10,
    "convergence_gtol": 1e-10,
}


@pytest.mark.parametrize(
    ("model_id", "theta"),
    [
        ("B1", [0.17]),
        ("P2", [0.17, 0.16]),
        ("P3-S", [0.17, 0.16, 2.2]),
        ("P3-O", [0.17, 0.16, 0.62]),
        ("P4", [0.17, 0.16, 2.2, 0.62]),
    ],
)
def test_variable_dimension_inverse_recovers_synthetic_parameters(model_id: str, theta: list[float]) -> None:
    registry = load_model_registry()
    model = RegisteredSkinForwardModel(model_id, registry)
    wavelength = np.asarray(registry.wavelength_nm)
    expected = np.asarray(theta)
    observed = np.asarray(model.forward_flat_numpy(expected, wavelength))
    fit = fit_registered_spectrum(observed, wavelength, model, FIT, n_starts=5)
    np.testing.assert_allclose(fit.theta, expected, rtol=0.015, atol=0.002)
    assert fit.metrics["log_rmse"] < 1e-5
    assert fit.identifiability["jacobian_rank"] == len(theta)
    assert len(fit.starts) == 5


def test_variable_inverse_band_gate_and_b0_rejection() -> None:
    registry = load_model_registry()
    wavelength = np.asarray(registry.wavelength_nm)
    p2 = RegisteredSkinForwardModel("P2", registry)
    observed = np.asarray(p2.forward_flat_numpy(np.asarray([0.17, 0.16]), wavelength))
    too_few = np.zeros(31, dtype=bool)
    too_few[:23] = True
    with pytest.raises(ValueError, match="too few"):
        fit_registered_spectrum(observed, wavelength, p2, FIT, n_starts=3, band_mask=too_few)
    with pytest.raises(ValueError, match="no free parameters"):
        fit_registered_spectrum(
            observed,
            wavelength,
            RegisteredSkinForwardModel("B0", registry),
            FIT,
            n_starts=3,
        )


def test_profile_likelihood_keeps_fixed_parameter_and_reports_delta() -> None:
    registry = load_model_registry()
    wavelength = np.asarray(registry.wavelength_nm)
    model = RegisteredSkinForwardModel("P2", registry)
    expected = np.asarray([0.17, 0.16])
    observed = np.asarray(model.forward_flat_numpy(expected, wavelength))
    fit = fit_registered_spectrum(observed, wavelength, model, FIT, n_starts=4)
    rows = profile_registered_spectrum(
        observed,
        wavelength,
        model,
        fit.theta,
        FIT,
        [0.2, 0.5, 0.8],
    )
    assert len(rows) == 6
    assert all(row["delta_objective"] >= -1e-10 for row in rows)
    for row in rows:
        index = model.parameter_names.index(row["profiled_parameter"])
        assert row["conditional_theta"][index] == pytest.approx(row["fixed_value"])
