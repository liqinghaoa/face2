from __future__ import annotations

import numpy as np
import torch

from skin_optics_hsi.config import load_stage1_config
from skin_optics_hsi.hsi_inverse_fit import fit_spectrum
from skin_optics_hsi.parameterization import logits_to_theta, theta_to_logits
from skin_optics_hsi.skin_forward import SkinForwardModel


def test_parameterization_round_trip() -> None:
    config = load_stage1_config()
    theta = np.asarray([0.17, 0.21])
    recovered = logits_to_theta(theta_to_logits(theta, config.bounds), config.bounds)
    np.testing.assert_allclose(recovered, theta, rtol=0, atol=1e-12)


def test_forward_is_finite_monotonic_and_has_torch_parity() -> None:
    config = load_stage1_config()
    model = SkinForwardModel(config)
    wavelength = np.asarray(config.wavelength_nm)
    low = np.asarray([0.08, 0.08])
    high_m = np.asarray([0.30, 0.08])
    high_h = np.asarray([0.08, 0.30])
    low_spectrum = np.asarray(model.forward_numpy(low, wavelength))
    assert np.all(np.isfinite(low_spectrum))
    assert np.all((low_spectrum >= 0) & (low_spectrum <= 1))
    assert np.mean(model.forward_numpy(high_m, wavelength)) < np.mean(low_spectrum)
    assert np.mean(model.forward_numpy(high_h, wavelength)) < np.mean(low_spectrum)

    theta_torch = torch.tensor(low, dtype=torch.float64, requires_grad=True)
    torch_spectrum = model.forward_torch(theta_torch, wavelength)
    np.testing.assert_allclose(torch_spectrum.detach().numpy(), low_spectrum, rtol=1e-12, atol=1e-12)
    torch_spectrum.sum().backward()
    assert theta_torch.grad is not None
    assert torch.all(torch.isfinite(theta_torch.grad))
    assert torch.all(theta_torch.grad < 0)


def test_multistart_inverse_recovers_synthetic_theta() -> None:
    config = load_stage1_config()
    model = SkinForwardModel(config)
    wavelength = np.asarray(config.wavelength_nm)
    expected = np.asarray([0.19, 0.16])
    observed = np.asarray(model.forward_numpy(expected, wavelength))
    result = fit_spectrum(observed, wavelength, model, config, run_noise_audit=False)
    np.testing.assert_allclose(result.theta, expected, rtol=0.02, atol=0.003)
    assert result.metrics["log_rmse"] < 1e-3
    assert len(result.starts) == config.raw["fit"]["n_starts"]
    assert len(result.identifiability["jacobian_singular_values"]) == 2

