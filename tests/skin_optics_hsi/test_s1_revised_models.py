import numpy as np
import pytest
import torch

from src.skin_optics_hsi.s1_proxy_inverse import fit_group_isolated_pca, fit_proxy_wls
from src.skin_optics_hsi.s1_revised_forward import (
    RevisedSkinForwardModel, centered_log_ratio, load_revised_registry, packaging_factor, revised_skin_forward,
)


@pytest.fixture(scope="module")
def registry():
    return load_revised_registry()


def test_registry_contract(registry):
    assert registry.wavelength_nm.tolist() == list(np.arange(400.0, 701.0, 10.0))
    assert set(registry.models) == {"B0-R", "B0-S", "B2-PCA", "B2-RPCA", "D1-M", "D2-MH", "D3-MHG", "D3-MHO", "K2-MH-KM", "K3-MHS-KM", "T2-MH-RTE"}
    assert registry.raw["wavelength_contract"]["effective_srf_status"] == "missing"


def test_packaging_limits_and_monotonicity():
    out = packaging_factor(np.array([0.0, 0.1, 1.0]), 15.0)
    assert np.isclose(out[0], 1.0)
    assert np.all((out > 0) & (out <= 1))
    assert np.all(np.diff(out) < 0)
    assert np.array_equal(packaging_factor(np.array([0.1, 10.0]), 0.0), np.ones(2))
    with pytest.raises(ValueError):
        packaging_factor(np.array([-0.1]), 15.0)


def test_d2_exact_recovery(registry):
    wave = registry.wavelength_nm
    ref = 0.48 + 0.035 * np.cos((wave - 550) / 90)
    model = RevisedSkinForwardModel("D2-MH", registry)
    theta = np.array([0.31, -0.19])
    observed = model.forward_numpy(theta, wave, {"reference_reflectance": ref, "a_obs": 0.25})
    fit = fit_proxy_wls("D2-MH", observed, ref, registry=registry)
    assert np.max(np.abs(fit.theta - theta)) < 1e-7
    assert fit.weighted_rmse < 1e-9


def test_torch_numpy_parity_and_grad(registry):
    wave = registry.wavelength_nm
    model = RevisedSkinForwardModel("D2-MH", registry)
    theta = torch.tensor([0.2, -0.1], dtype=torch.float64, requires_grad=True)
    np_out = model.forward_numpy(theta.detach().numpy(), wave, {"reference_reflectance": np.full(31, 0.5)})
    out = model.forward_torch(theta, wave, {"reference_reflectance": np.full(31, 0.5)})
    assert np.max(np.abs(np_out - out.detach().numpy())) < 1e-10
    out.sum().backward()
    assert torch.all(torch.isfinite(theta.grad))


def test_k2_positive_and_rte_deferred(registry):
    wave = registry.wavelength_nm
    model = RevisedSkinForwardModel("K2-MH-KM", registry)
    theta_np = np.array([0.13, 0.011])
    out = model.forward_numpy(theta_np, wave)
    assert np.all(np.isfinite(out)) and np.all(out > 0)
    theta_t = torch.tensor(theta_np, dtype=torch.float64, requires_grad=True)
    out_t = model.forward_torch(theta_t, wave)
    assert np.max(np.abs(out - out_t.detach().numpy())) < 1e-10
    out_t.sum().backward()
    assert torch.all(torch.isfinite(theta_t.grad))
    unpackaged = model.forward_numpy(theta_np, wave, {"vessel_diameter_um": 0.0})
    assert np.all(np.isfinite(unpackaged)) and np.all(unpackaged > 0)
    with pytest.raises(RuntimeError):
        RevisedSkinForwardModel("T2-MH-RTE", registry).forward_numpy(np.empty(0), wave)


def test_centered_log_ratio_rejects_nonpositive():
    with pytest.raises(ValueError):
        centered_log_ratio(np.zeros(31), np.ones(31))


def test_public_interface_and_unbounded_proxy(registry):
    wave = registry.wavelength_nm
    result = revised_skin_forward(theta_bio=np.array([2.5, -2.5]), theta_nuisance=None,
                                  wavelength_nm=wave, global_params={"reference_reflectance": np.full(31, 0.5)},
                                  model_id="D2-MH", registry=registry)
    assert result["reflectance"].shape == (31,)
    assert result["centered_log_ratio"].shape == (31,)
    assert result["provenance"]["model_id"] == "D2-MH"


def test_group_isolated_pca_excludes_holdout():
    x = np.arange(8 * 31, dtype=float).reshape(8, 31)
    ids = np.array(["a", "a", "b", "b", "c", "c", "hold", "hold"])
    result = fit_group_isolated_pca(x, ids, held_out_subject_id="hold")
    assert result.components.shape == (31, 2)
    assert result.training_subject_ids == ("a", "b", "c")


def test_registered_band_subset_forward_and_inverse(registry):
    wave = registry.wavelength_nm[1:-1]
    ref = np.full(wave.size, 0.5)
    theta = np.array([0.2, -0.12])
    model = RevisedSkinForwardModel("D2-MH", registry)
    observed = model.forward_numpy(theta, wave, {"reference_reflectance": ref})
    fit = fit_proxy_wls("D2-MH", observed, ref, registry=registry, wavelength_nm=wave)
    assert np.max(np.abs(fit.theta - theta)) < 1e-7


def test_torch_rejects_invalid_km_parameters(registry):
    with pytest.raises(ValueError):
        RevisedSkinForwardModel("K2-MH-KM", registry).forward_torch(
            torch.tensor([1.1, 0.01], dtype=torch.float64), registry.wavelength_nm
        )
