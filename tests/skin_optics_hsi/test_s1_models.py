from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch

from skin_optics_hsi.model_registry import load_model_registry
from skin_optics_hsi.s1_model_audit import build_s1_4_model_audit
from skin_optics_hsi.s1_spectral_sensitivity import (
    GaussianSRFModelAdapter,
    band_mask,
    gaussian_srf_reflectance,
    shifted_centers,
)
from skin_optics_hsi.skin_forward import RegisteredSkinForwardModel, skin_forward


def _reference_groups(model: RegisteredSkinForwardModel) -> tuple[np.ndarray | None, np.ndarray | None]:
    bio = (
        np.asarray([model.registry.parameter(name).reference_value for name in model.bio_parameter_names])
        if model.bio_parameter_names
        else None
    )
    nuisance = (
        np.asarray([model.registry.parameter(name).reference_value for name in model.nuisance_parameter_names])
        if model.nuisance_parameter_names
        else None
    )
    return bio, nuisance


def test_s1_4_registry_has_frozen_parameter_order_and_activation_policy() -> None:
    registry = load_model_registry()
    assert tuple(registry.models) == ("B0", "B1", "P2", "P3-S", "P3-O", "P4")
    assert registry.model("P2").flat_parameter_names == ("M_absorbance", "Hb_absorbance_proxy")
    assert registry.model("P3-S").theta_nuisance == ("S_amp",)
    assert registry.model("P3-O").is_activated is False
    assert registry.model("P4").is_activated is False
    assert registry.raw["wavelength_contract"]["effective_srf_status"] == "missing"
    assert registry.raw["observation_policy"]["per_spectrum_exposure_scale_allowed"] is False


@pytest.mark.parametrize("model_id", ["B1", "P2", "P3-S", "P3-O", "P4"])
def test_registered_physical_models_have_numpy_torch_parity_and_auditable_components(model_id: str) -> None:
    registry = load_model_registry()
    model = RegisteredSkinForwardModel(model_id, registry)
    wavelength = np.asarray(registry.wavelength_nm)
    bio, nuisance = _reference_groups(model)
    predicted, components = model.forward_numpy(bio, wavelength, nuisance, return_components=True)
    bio_t = None if bio is None else torch.tensor(bio, dtype=torch.float64, requires_grad=True)
    nuisance_t = None if nuisance is None else torch.tensor(nuisance, dtype=torch.float64, requires_grad=True)
    predicted_t, components_t = model.forward_torch(
        bio_t, wavelength, nuisance_t, return_components=True
    )
    np.testing.assert_allclose(predicted_t.detach().numpy(), predicted, rtol=1e-11, atol=1e-12)
    assert np.all(np.isfinite(predicted))
    assert np.all(predicted >= 0)
    np.testing.assert_allclose(components.reflectance, predicted, rtol=0, atol=0)
    np.testing.assert_allclose(
        components.total_dermal_absorption_mm1,
        components.baseline_absorption_mm1 + components.hemoglobin_absorption_mm1,
        rtol=1e-12,
        atol=1e-12,
    )
    assert components.physical_model_applied is True
    assert components_t.physical_model_applied is True
    predicted_t.sum().backward()
    if bio_t is not None:
        assert bio_t.grad is not None and torch.all(torch.isfinite(bio_t.grad))
    if nuisance_t is not None:
        assert nuisance_t.grad is not None and torch.all(torch.isfinite(nuisance_t.grad))


@pytest.mark.parametrize("model_id", ["B1", "P2", "P3-S", "P3-O", "P4"])
def test_all_free_parameter_gradients_match_central_finite_difference(model_id: str) -> None:
    registry = load_model_registry()
    model = RegisteredSkinForwardModel(model_id, registry)
    wavelength = np.asarray(registry.wavelength_nm)
    theta = np.asarray(model.registry.reference_values_for(model_id), dtype=np.float64)
    theta_t = torch.tensor(theta, dtype=torch.float64, requires_grad=True)
    torch.sum(model.forward_flat_torch(theta_t, wavelength)).backward()
    assert theta_t.grad is not None
    analytic = theta_t.grad.detach().numpy()
    numeric = np.zeros_like(theta)
    for index, (lower, upper) in enumerate(model.bounds):
        step = (upper - lower) * 1e-6
        lo = theta.copy()
        hi = theta.copy()
        lo[index] -= step
        hi[index] += step
        numeric[index] = (
            np.sum(model.forward_flat_numpy(hi, wavelength))
            - np.sum(model.forward_flat_numpy(lo, wavelength))
        ) / (2.0 * step)
    np.testing.assert_allclose(analytic, numeric, rtol=2e-5, atol=1e-7)


def test_b0_requires_explicit_train_only_reference_and_has_no_free_parameters() -> None:
    registry = load_model_registry()
    model = RegisteredSkinForwardModel("B0", registry)
    wavelength = np.asarray(registry.wavelength_nm)
    with pytest.raises(ValueError, match="Train-only"):
        model.forward_numpy(None, wavelength)
    train_mean = np.linspace(0.2, 0.6, len(wavelength))
    predicted, components = model.forward_numpy(
        None,
        wavelength,
        global_params={"train_mean_reflectance": train_mean},
        return_components=True,
    )
    np.testing.assert_allclose(predicted, train_mean)
    assert model.parameter_names == ()
    assert components.physical_model_applied is False


def test_bounds_and_free_per_spectrum_exposure_are_hard_rejections() -> None:
    registry = load_model_registry()
    model = RegisteredSkinForwardModel("P3-S", registry)
    wavelength = np.asarray(registry.wavelength_nm)
    with pytest.raises(ValueError, match="M_absorbance outside"):
        model.forward_numpy(np.asarray([1.0, 0.18]), wavelength, np.asarray([1.99]))
    with pytest.raises(ValueError, match="Per-spectrum"):
        model.forward_numpy(
            np.asarray([0.13, 0.18]),
            wavelength,
            np.asarray([1.99]),
            global_params={"exposure_scale": 1.1},
        )


def test_side_effect_is_fixed_global_context_not_a_free_theta() -> None:
    registry = load_model_registry()
    model = RegisteredSkinForwardModel("P2", registry)
    wavelength = np.asarray(registry.wavelength_nm)
    bio = np.asarray([0.13, 0.18])
    fixed = {
        "side_log_gain_by_region": {"image_left": 0.1, "image_right": -0.1, "other": 0.0}
    }
    left = model.forward_numpy(
        bio, wavelength, global_params=fixed, observation_context={"region_side": "image_left"}
    )
    right = model.forward_numpy(
        bio, wavelength, global_params=fixed, observation_context={"region_side": "image_right"}
    )
    np.testing.assert_allclose(left / right, np.exp(0.2), rtol=1e-12, atol=1e-12)
    assert "side_log_gain" not in model.parameter_names


def test_public_interface_and_prespecified_sensitivities() -> None:
    registry = load_model_registry()
    wavelength = np.asarray(registry.wavelength_nm)
    result, components = skin_forward(
        np.asarray([0.13, 0.18]), wavelength, model_id="P2", registry=registry
    )
    assert result.shape == (31,)
    assert components.model_id == "P2"
    assert int(band_mask(wavelength, "full_31", registry).sum()) == 31
    assert int(band_mask(wavelength, "remove_endpoints", registry).sum()) == 29
    assert int(band_mask(wavelength, "remove_high_curvature_candidates", registry).sum()) == 24
    np.testing.assert_allclose(shifted_centers(wavelength[1:-1], -5.0, registry), wavelength[1:-1] - 5.0)

    model = RegisteredSkinForwardModel("P2", registry)
    point = gaussian_srf_reflectance(model, np.asarray([0.13, 0.18]), wavelength, 0.0)
    blurred = gaussian_srf_reflectance(model, np.asarray([0.13, 0.18]), wavelength, 10.0)
    np.testing.assert_allclose(point, result)
    assert blurred.shape == (31,)
    assert np.all(np.isfinite(blurred))
    assert not np.allclose(blurred, point)
    adapter = GaussianSRFModelAdapter(model, wavelength, 10.0)
    adapter_blurred = adapter.forward_flat_numpy(np.asarray([0.13, 0.18]), wavelength)
    np.testing.assert_allclose(adapter_blurred, blurred, rtol=2e-4, atol=2e-5)


def test_s1_4_artifact_writer_requires_s1_3_gate_and_refuses_overwrite(tmp_path: Path) -> None:
    registry = load_model_registry()
    s1_3_path = tmp_path / "s1_3_final_decision.json"
    contract_path = tmp_path / "data_contract.json"
    s1_3_path.write_text(
        json.dumps(
            {
                "status": "PASS_FOR_S1_4",
                "next_stage_allowed": True,
                "authorized_next_stage": "S1-4",
                "test_access_count": 0,
            }
        ),
        encoding="utf-8",
    )
    contract_path.write_text(
        json.dumps(
            {
                "wavelength": {
                    "centers_nm": {
                        "status": "confirmed_from_official_code",
                        "value": list(registry.wavelength_nm),
                    },
                    "release_effective_bandwidth_or_srf": {"status": "missing"},
                }
            }
        ),
        encoding="utf-8",
    )
    output = tmp_path / "s1_4"
    decision = build_s1_4_model_audit(s1_3_path, contract_path, output)
    assert decision["status"] == "PASS_FOR_S1_5"
    assert decision["conditional_not_activated"] == ["P3-O", "P4"]
    assert decision["test_access_count"] == 0
    assert {
        "parameter_contract.yaml",
        "model_registry.json",
        "model_unit_audit.json",
        "MODEL_CARDS.md",
        "s1_4_decision.json",
    } == {path.name for path in output.iterdir()}
    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        build_s1_4_model_audit(s1_3_path, contract_path, output)

    blocked = tmp_path / "blocked.json"
    blocked.write_text(
        json.dumps(
            {
                "status": "PASS_FOR_S1_4",
                "next_stage_allowed": True,
                "authorized_next_stage": "S1-4",
                "test_access_count": 1,
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(RuntimeError, match="Test-isolation"):
        build_s1_4_model_audit(blocked, contract_path, tmp_path / "blocked_output")
