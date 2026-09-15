"""S1-4 registered reduced-order skin-reflectance candidate models.

The physical candidates are screening models. They intentionally separate
biological proxy parameters from nuisance and global parameters, and they are
not a reproduction of the Jonasson three-layer inverse Monte Carlo model.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np
import torch

from .config import Stage1Config
from .model_registry import Stage1ModelRegistry, load_model_registry
from .spectral_assets import HemoglobinSpectra, load_hemoglobin_spectra


@dataclass(frozen=True)
class SpectralComponents:
    model_id: str
    physical_model_applied: bool
    melanin_shape: Any
    hemoglobin_shape: Any
    melanin_absorbance: Any
    hemoglobin_absorption_mm1: Any
    baseline_absorption_mm1: Any
    total_dermal_absorption_mm1: Any
    reduced_scattering_mm1: Any
    epidermal_transmission_squared: Any
    dermal_reflectance: Any
    observation_gain: Any
    reflectance: Any


def _km_infinite_numpy(mu_a: np.ndarray, mu_s_prime: np.ndarray) -> np.ndarray:
    ratio = mu_a / np.maximum(mu_s_prime, 1e-12)
    coefficient = 1.0 + ratio
    return coefficient - np.sqrt(np.maximum(coefficient * coefficient - 1.0, 0.0))


def _km_infinite_torch(mu_a: torch.Tensor, mu_s_prime: torch.Tensor) -> torch.Tensor:
    coefficient = 1.0 + mu_a / torch.clamp(mu_s_prime, min=1e-12)
    return coefficient - torch.sqrt(torch.clamp(coefficient.square() - 1.0, min=0.0))


class RegisteredSkinForwardModel:
    """One model selected from the validated S1-4 registry."""

    _ALLOWED_GLOBAL_OVERRIDES = {
        "scattering_slope",
        "rayleigh_fraction",
        "global_reflectance_scale",
        "side_log_gain_by_region",
        "train_mean_reflectance",
    }
    _FORBIDDEN_PER_SPECTRUM_KEYS = {
        "exposure_scale",
        "per_spectrum_scale",
        "per_image_scale",
        "free_observation_gain",
    }

    def __init__(
        self,
        model_id: str,
        registry: Stage1ModelRegistry | None = None,
        hemoglobin: HemoglobinSpectra | None = None,
    ) -> None:
        self.registry = registry or load_model_registry()
        self.model_spec = self.registry.model(model_id)
        self.model_id = model_id
        self.hemoglobin = hemoglobin or load_hemoglobin_spectra(self.registry.spectral_asset_path)

    @property
    def bio_parameter_names(self) -> tuple[str, ...]:
        return self.model_spec.theta_bio

    @property
    def nuisance_parameter_names(self) -> tuple[str, ...]:
        return self.model_spec.theta_nuisance

    @property
    def parameter_names(self) -> tuple[str, ...]:
        return self.model_spec.flat_parameter_names

    @property
    def bounds(self) -> tuple[tuple[float, float], ...]:
        return self.registry.bounds_for(self.model_id)

    def _validate_wavelength(self, wavelength_nm: np.ndarray) -> np.ndarray:
        wavelength = np.asarray(wavelength_nm, dtype=np.float64)
        if wavelength.ndim != 1 or len(wavelength) < 1:
            raise ValueError("wavelength_nm must be a non-empty one-dimensional array")
        if np.any(~np.isfinite(wavelength)) or (len(wavelength) > 1 and np.any(np.diff(wavelength) <= 0)):
            raise ValueError("wavelength_nm must be finite and strictly increasing")
        if wavelength.min() < self.hemoglobin.wavelength_nm.min() or wavelength.max() > self.hemoglobin.wavelength_nm.max():
            raise ValueError("wavelength_nm exceeds the audited hemoglobin asset coverage")
        return wavelength

    def _validate_numpy_group(
        self,
        values: np.ndarray | None,
        names: tuple[str, ...],
        label: str,
    ) -> np.ndarray:
        if values is None:
            if names:
                raise ValueError(f"{label} is required for {self.model_id}")
            return np.empty((0,), dtype=np.float64)
        result = np.asarray(values, dtype=np.float64)
        if result.ndim == 0 or result.shape[-1] != len(names):
            raise ValueError(f"{label} must have final dimension {len(names)} for {self.model_id}")
        if np.any(~np.isfinite(result)):
            raise ValueError(f"{label} contains non-finite values")
        for index, name in enumerate(names):
            parameter = self.registry.parameter(name)
            if np.any((result[..., index] < parameter.minimum) | (result[..., index] > parameter.maximum)):
                raise ValueError(f"{name} outside [{parameter.minimum}, {parameter.maximum}]")
        return result

    def _validate_torch_group(
        self,
        values: torch.Tensor | None,
        names: tuple[str, ...],
        label: str,
        *,
        dtype: torch.dtype,
        device: torch.device,
    ) -> torch.Tensor:
        if values is None:
            if names:
                raise ValueError(f"{label} is required for {self.model_id}")
            return torch.empty((0,), dtype=dtype, device=device)
        if values.ndim == 0 or values.shape[-1] != len(names):
            raise ValueError(f"{label} must have final dimension {len(names)} for {self.model_id}")
        if not bool(torch.all(torch.isfinite(values)).detach().cpu()):
            raise ValueError(f"{label} contains non-finite values")
        for index, name in enumerate(names):
            parameter = self.registry.parameter(name)
            in_range = torch.all((values[..., index] >= parameter.minimum) & (values[..., index] <= parameter.maximum))
            if not bool(in_range.detach().cpu()):
                raise ValueError(f"{name} outside [{parameter.minimum}, {parameter.maximum}]")
        return values

    def _global_parameters(self, overrides: Mapping[str, Any] | None) -> dict[str, Any]:
        supplied = dict(overrides or {})
        forbidden = self._FORBIDDEN_PER_SPECTRUM_KEYS.intersection(supplied)
        if forbidden:
            raise ValueError(f"Per-spectrum observation scaling is forbidden: {sorted(forbidden)}")
        unknown = set(supplied).difference(self._ALLOWED_GLOBAL_OVERRIDES)
        if unknown:
            raise ValueError(f"Unknown global parameter override(s): {sorted(unknown)}")
        values: dict[str, Any] = {
            "scattering_slope": float(self.registry.global_value("scattering_slope")),
            "rayleigh_fraction": float(self.registry.global_value("rayleigh_fraction")),
            "global_reflectance_scale": float(self.registry.global_value("global_reflectance_scale")),
            "side_log_gain_by_region": dict(self.registry.global_value("side_log_gain_by_region")),
        }
        values.update(supplied)
        for name in ("scattering_slope", "rayleigh_fraction", "global_reflectance_scale"):
            value = float(values[name])
            contract = self.registry.raw["global_parameters"][name]
            if not np.isfinite(value) or not float(contract["allowed_min"]) <= value <= float(contract["allowed_max"]):
                raise ValueError(f"Global parameter {name} outside its audited range")
            values[name] = value
        gains = values["side_log_gain_by_region"]
        if not isinstance(gains, Mapping):
            raise ValueError("side_log_gain_by_region must be a mapping")
        allowed_sides = set(self.registry.raw["observation_policy"]["allowed_region_sides"])
        if set(gains) != allowed_sides:
            raise ValueError("side_log_gain_by_region must contain exactly image_left/image_right/other")
        clean_gains = {str(key): float(value) for key, value in gains.items()}
        if any(not np.isfinite(value) for value in clean_gains.values()):
            raise ValueError("side_log_gain_by_region contains non-finite values")
        values["side_log_gain_by_region"] = clean_gains
        return values

    def _observation_gain(
        self,
        global_values: Mapping[str, Any],
        observation_context: Mapping[str, Any] | None,
    ) -> float:
        side = "other" if observation_context is None else str(observation_context.get("region_side", "other"))
        allowed = set(self.registry.raw["observation_policy"]["allowed_region_sides"])
        if side not in allowed:
            raise ValueError(f"Unknown region_side: {side}")
        return float(np.exp(global_values["side_log_gain_by_region"][side]))

    def _numpy_hb_shape(self, wavelength: np.ndarray, oxygenation: np.ndarray) -> np.ndarray:
        hbo2 = np.interp(wavelength, self.hemoglobin.wavelength_nm, self.hemoglobin.epsilon_hbo2)
        hb = np.interp(wavelength, self.hemoglobin.wavelength_nm, self.hemoglobin.epsilon_hb)
        reference_nm = float(self.registry.raw["fixed_physics"]["reference_wavelength_nm"])
        hbo2_ref = float(np.interp(reference_nm, self.hemoglobin.wavelength_nm, self.hemoglobin.epsilon_hbo2))
        hb_ref = float(np.interp(reference_nm, self.hemoglobin.wavelength_nm, self.hemoglobin.epsilon_hb))
        mixed = oxygenation[..., None] * hbo2 + (1.0 - oxygenation[..., None]) * hb
        reference = oxygenation * hbo2_ref + (1.0 - oxygenation) * hb_ref
        return mixed / reference[..., None]

    def _torch_hb_shape(
        self,
        wavelength: np.ndarray,
        oxygenation: torch.Tensor,
        *,
        dtype: torch.dtype,
        device: torch.device,
    ) -> torch.Tensor:
        hbo2 = torch.as_tensor(
            np.interp(wavelength, self.hemoglobin.wavelength_nm, self.hemoglobin.epsilon_hbo2),
            dtype=dtype,
            device=device,
        )
        hb = torch.as_tensor(
            np.interp(wavelength, self.hemoglobin.wavelength_nm, self.hemoglobin.epsilon_hb),
            dtype=dtype,
            device=device,
        )
        reference_nm = float(self.registry.raw["fixed_physics"]["reference_wavelength_nm"])
        hbo2_ref = torch.as_tensor(
            np.interp(reference_nm, self.hemoglobin.wavelength_nm, self.hemoglobin.epsilon_hbo2),
            dtype=dtype,
            device=device,
        )
        hb_ref = torch.as_tensor(
            np.interp(reference_nm, self.hemoglobin.wavelength_nm, self.hemoglobin.epsilon_hb),
            dtype=dtype,
            device=device,
        )
        mixed = oxygenation[..., None] * hbo2 + (1.0 - oxygenation[..., None]) * hb
        reference = oxygenation * hbo2_ref + (1.0 - oxygenation) * hb_ref
        return mixed / reference[..., None]

    @staticmethod
    def _parameter_numpy(values: np.ndarray, names: tuple[str, ...], name: str, default: float) -> np.ndarray:
        if name not in names:
            return np.asarray(default, dtype=np.float64)
        return values[..., names.index(name)]

    @staticmethod
    def _parameter_torch(values: torch.Tensor, names: tuple[str, ...], name: str, default: float) -> torch.Tensor:
        if name not in names:
            return torch.as_tensor(default, dtype=values.dtype, device=values.device)
        return values[..., names.index(name)]

    def forward_numpy(
        self,
        theta_bio: np.ndarray | None,
        wavelength_nm: np.ndarray,
        theta_nuisance: np.ndarray | None = None,
        global_params: Mapping[str, Any] | None = None,
        *,
        observation_context: Mapping[str, Any] | None = None,
        return_components: bool = False,
    ) -> np.ndarray | tuple[np.ndarray, SpectralComponents]:
        wavelength = self._validate_wavelength(wavelength_nm)
        bio = self._validate_numpy_group(theta_bio, self.bio_parameter_names, "theta_bio")
        nuisance = self._validate_numpy_group(theta_nuisance, self.nuisance_parameter_names, "theta_nuisance")
        globals_ = self._global_parameters(global_params)
        gain = self._observation_gain(globals_, observation_context)

        if self.model_spec.family == "train_mean_nonphysical":
            if "train_mean_reflectance" not in globals_:
                raise ValueError("B0 requires a Train-only train_mean_reflectance global input")
            mean = np.asarray(globals_["train_mean_reflectance"], dtype=np.float64)
            if mean.shape != wavelength.shape or np.any(~np.isfinite(mean)) or np.any(mean < 0):
                raise ValueError("train_mean_reflectance must be finite, non-negative, and wavelength-aligned")
            reflectance = float(globals_["global_reflectance_scale"]) * gain * mean
            zeros = np.zeros_like(reflectance)
            ones = np.ones_like(reflectance)
            components = SpectralComponents(
                self.model_id, False, zeros, zeros, zeros, zeros, zeros, zeros, zeros,
                ones, mean, np.asarray(gain), reflectance,
            )
            return (reflectance, components) if return_components else reflectance

        try:
            batch_shape = np.broadcast_shapes(bio.shape[:-1], nuisance.shape[:-1])
        except ValueError as exc:
            raise ValueError("theta_bio and theta_nuisance batch dimensions are not broadcastable") from exc
        bio = np.broadcast_to(bio, batch_shape + (len(self.bio_parameter_names),))
        nuisance = np.broadcast_to(nuisance, batch_shape + (len(self.nuisance_parameter_names),))

        physics = self.registry.raw["fixed_physics"]
        reference_nm = float(physics["reference_wavelength_nm"])
        melanin_shape = (wavelength / reference_nm) ** (-float(physics["melanin_power"]))
        m = self._parameter_numpy(bio, self.bio_parameter_names, "M_absorbance", 0.0)
        h = self._parameter_numpy(bio, self.bio_parameter_names, "Hb_absorbance_proxy", 0.0)
        oxygenation = self._parameter_numpy(nuisance, self.nuisance_parameter_names, "sO2", float(physics["fixed_oxygenation"]))
        scattering_amplitude = self._parameter_numpy(
            nuisance, self.nuisance_parameter_names, "S_amp", float(physics["fixed_scattering_amplitude_mm1_at_600nm"])
        )
        hemoglobin_shape = self._numpy_hb_shape(wavelength, oxygenation)
        baseline = (0.244 + 85.3 * np.exp(-(wavelength - 154.0) / 66.2)) / 10.0
        ratio = wavelength / float(physics["scattering_reference_wavelength_nm"])
        scattering_shape = (
            (1.0 - float(globals_["rayleigh_fraction"])) * ratio ** (-float(globals_["scattering_slope"]))
            + float(globals_["rayleigh_fraction"]) * ratio ** (-4.0)
        )
        melanin_absorbance = m[..., None] * melanin_shape
        hemoglobin_absorption = h[..., None] * hemoglobin_shape
        total_absorption = baseline + hemoglobin_absorption
        scattering = scattering_amplitude[..., None] * scattering_shape
        transmission_squared = np.exp(-2.0 * melanin_absorbance)
        dermal = _km_infinite_numpy(total_absorption, scattering)
        reflectance = float(globals_["global_reflectance_scale"]) * gain * transmission_squared * dermal
        if np.any(~np.isfinite(reflectance)) or np.any(reflectance < 0):
            raise ValueError("Forward model produced invalid reflectance")
        components = SpectralComponents(
            self.model_id,
            True,
            np.broadcast_to(melanin_shape, reflectance.shape),
            np.broadcast_to(hemoglobin_shape, reflectance.shape),
            np.broadcast_to(melanin_absorbance, reflectance.shape),
            np.broadcast_to(hemoglobin_absorption, reflectance.shape),
            np.broadcast_to(baseline, reflectance.shape),
            np.broadcast_to(total_absorption, reflectance.shape),
            np.broadcast_to(scattering, reflectance.shape),
            np.broadcast_to(transmission_squared, reflectance.shape),
            np.broadcast_to(dermal, reflectance.shape),
            np.asarray(gain),
            reflectance,
        )
        return (reflectance, components) if return_components else reflectance

    def forward_torch(
        self,
        theta_bio: torch.Tensor | None,
        wavelength_nm: np.ndarray,
        theta_nuisance: torch.Tensor | None = None,
        global_params: Mapping[str, Any] | None = None,
        *,
        observation_context: Mapping[str, Any] | None = None,
        return_components: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, SpectralComponents]:
        wavelength = self._validate_wavelength(wavelength_nm)
        template = theta_bio if theta_bio is not None else theta_nuisance
        dtype = torch.float64 if template is None else template.dtype
        device = torch.device("cpu") if template is None else template.device
        bio = self._validate_torch_group(theta_bio, self.bio_parameter_names, "theta_bio", dtype=dtype, device=device)
        nuisance = self._validate_torch_group(
            theta_nuisance, self.nuisance_parameter_names, "theta_nuisance", dtype=dtype, device=device
        )
        globals_ = self._global_parameters(global_params)
        gain = torch.as_tensor(self._observation_gain(globals_, observation_context), dtype=dtype, device=device)

        if self.model_spec.family == "train_mean_nonphysical":
            if "train_mean_reflectance" not in globals_:
                raise ValueError("B0 requires a Train-only train_mean_reflectance global input")
            mean_np = np.asarray(globals_["train_mean_reflectance"], dtype=np.float64)
            if mean_np.shape != wavelength.shape or np.any(~np.isfinite(mean_np)) or np.any(mean_np < 0):
                raise ValueError("train_mean_reflectance must be finite, non-negative, and wavelength-aligned")
            mean = torch.as_tensor(mean_np, dtype=dtype, device=device)
            reflectance = float(globals_["global_reflectance_scale"]) * gain * mean
            zeros = torch.zeros_like(reflectance)
            ones = torch.ones_like(reflectance)
            components = SpectralComponents(
                self.model_id, False, zeros, zeros, zeros, zeros, zeros, zeros, zeros,
                ones, mean, gain, reflectance,
            )
            return (reflectance, components) if return_components else reflectance

        try:
            batch_shape = torch.broadcast_shapes(bio.shape[:-1], nuisance.shape[:-1])
        except RuntimeError as exc:
            raise ValueError("theta_bio and theta_nuisance batch dimensions are not broadcastable") from exc
        bio = torch.broadcast_to(bio, batch_shape + (len(self.bio_parameter_names),))
        nuisance = torch.broadcast_to(nuisance, batch_shape + (len(self.nuisance_parameter_names),))
        wave = torch.as_tensor(wavelength, dtype=dtype, device=device)
        physics = self.registry.raw["fixed_physics"]
        melanin_shape = (wave / float(physics["reference_wavelength_nm"])) ** (-float(physics["melanin_power"]))
        m = self._parameter_torch(bio, self.bio_parameter_names, "M_absorbance", 0.0)
        h = self._parameter_torch(bio, self.bio_parameter_names, "Hb_absorbance_proxy", 0.0)
        oxygenation = self._parameter_torch(nuisance, self.nuisance_parameter_names, "sO2", float(physics["fixed_oxygenation"]))
        scattering_amplitude = self._parameter_torch(
            nuisance, self.nuisance_parameter_names, "S_amp", float(physics["fixed_scattering_amplitude_mm1_at_600nm"])
        )
        hemoglobin_shape = self._torch_hb_shape(wavelength, oxygenation, dtype=dtype, device=device)
        baseline = (0.244 + 85.3 * torch.exp(-(wave - 154.0) / 66.2)) / 10.0
        ratio = wave / float(physics["scattering_reference_wavelength_nm"])
        scattering_shape = (
            (1.0 - float(globals_["rayleigh_fraction"])) * ratio ** (-float(globals_["scattering_slope"]))
            + float(globals_["rayleigh_fraction"]) * ratio ** (-4.0)
        )
        melanin_absorbance = m[..., None] * melanin_shape
        hemoglobin_absorption = h[..., None] * hemoglobin_shape
        total_absorption = baseline + hemoglobin_absorption
        scattering = scattering_amplitude[..., None] * scattering_shape
        transmission_squared = torch.exp(-2.0 * melanin_absorbance)
        dermal = _km_infinite_torch(total_absorption, scattering)
        reflectance = float(globals_["global_reflectance_scale"]) * gain * transmission_squared * dermal
        if not bool(torch.all(torch.isfinite(reflectance)).detach().cpu()) or bool(torch.any(reflectance < 0).detach().cpu()):
            raise ValueError("Forward model produced invalid reflectance")
        components = SpectralComponents(
            self.model_id,
            True,
            torch.broadcast_to(melanin_shape, reflectance.shape),
            torch.broadcast_to(hemoglobin_shape, reflectance.shape),
            torch.broadcast_to(melanin_absorbance, reflectance.shape),
            torch.broadcast_to(hemoglobin_absorption, reflectance.shape),
            torch.broadcast_to(baseline, reflectance.shape),
            torch.broadcast_to(total_absorption, reflectance.shape),
            torch.broadcast_to(scattering, reflectance.shape),
            torch.broadcast_to(transmission_squared, reflectance.shape),
            torch.broadcast_to(dermal, reflectance.shape),
            gain,
            reflectance,
        )
        return (reflectance, components) if return_components else reflectance

    def _split_flat_numpy(self, theta: np.ndarray) -> tuple[np.ndarray | None, np.ndarray | None]:
        values = np.asarray(theta, dtype=np.float64)
        if values.ndim == 0 or values.shape[-1] != len(self.parameter_names):
            raise ValueError(f"theta must have final dimension {len(self.parameter_names)} for {self.model_id}")
        n_bio = len(self.bio_parameter_names)
        return values[..., :n_bio] if n_bio else None, values[..., n_bio:] if self.nuisance_parameter_names else None

    def _split_flat_torch(self, theta: torch.Tensor) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        if theta.ndim == 0 or theta.shape[-1] != len(self.parameter_names):
            raise ValueError(f"theta must have final dimension {len(self.parameter_names)} for {self.model_id}")
        n_bio = len(self.bio_parameter_names)
        return theta[..., :n_bio] if n_bio else None, theta[..., n_bio:] if self.nuisance_parameter_names else None

    def forward_flat_numpy(self, theta: np.ndarray, wavelength_nm: np.ndarray, **kwargs: Any) -> Any:
        bio, nuisance = self._split_flat_numpy(theta)
        return self.forward_numpy(bio, wavelength_nm, nuisance, **kwargs)

    def forward_flat_torch(self, theta: torch.Tensor, wavelength_nm: np.ndarray, **kwargs: Any) -> Any:
        bio, nuisance = self._split_flat_torch(theta)
        return self.forward_torch(bio, wavelength_nm, nuisance, **kwargs)

    def provenance(self) -> dict[str, Any]:
        return {
            "model_id": self.model_id,
            "family": self.model_spec.family,
            "status": self.model_spec.status,
            "activation": self.model_spec.activation,
            "theta_bio": list(self.bio_parameter_names),
            "theta_nuisance": list(self.nuisance_parameter_names),
            "interpretation": self.registry.raw["interpretation_boundary"],
            "hemoglobin_asset": str(self.hemoglobin.source_path),
            "hemoglobin_asset_sha256": self.hemoglobin.source_sha256,
            "registry_path": str(self.registry.path),
            "references": list(self.registry.raw["references"]),
        }


def skin_forward(
    theta_bio: np.ndarray | torch.Tensor | None,
    wavelength_nm: np.ndarray,
    theta_nuisance: np.ndarray | torch.Tensor | None = None,
    global_params: Mapping[str, Any] | None = None,
    model_id: str = "P2",
    *,
    backend: str = "numpy",
    observation_context: Mapping[str, Any] | None = None,
    return_components: bool = True,
    registry: Stage1ModelRegistry | None = None,
) -> Any:
    """Unified public S1-4 forward interface."""

    model = RegisteredSkinForwardModel(model_id=model_id, registry=registry)
    if backend == "numpy":
        return model.forward_numpy(
            theta_bio, wavelength_nm, theta_nuisance, global_params,
            observation_context=observation_context, return_components=return_components,
        )
    if backend == "torch":
        return model.forward_torch(
            theta_bio, wavelength_nm, theta_nuisance, global_params,
            observation_context=observation_context, return_components=return_components,
        )
    raise ValueError("backend must be numpy or torch")


class SkinForwardModel:
    """Backward-compatible wrapper for the original two-parameter P2 path."""

    parameter_names = ("M_absorbance", "Hb_absorbance_proxy")

    def __init__(self, config: Stage1Config, hemoglobin: HemoglobinSpectra | None = None) -> None:
        self.config = config
        self.hemoglobin = hemoglobin or load_hemoglobin_spectra(config.spectral_asset_path)
        self.model_config = config.raw["forward_model"]
        self._registered = RegisteredSkinForwardModel("P2", hemoglobin=self.hemoglobin)

    def _legacy_globals(self) -> dict[str, float]:
        return {
            "scattering_slope": float(self.model_config["scattering_mie_decay"]),
            "rayleigh_fraction": float(self.model_config["scattering_rayleigh_fraction"]),
            "global_reflectance_scale": float(self.model_config["global_reflectance_scale"]),
        }

    @staticmethod
    def _km_infinite_reflectance(mu_a: np.ndarray, mu_s_prime: np.ndarray) -> np.ndarray:
        return _km_infinite_numpy(mu_a, mu_s_prime)

    def forward_numpy(self, theta: np.ndarray, wavelength_nm: np.ndarray, return_components: bool = False) -> Any:
        return self._registered.forward_numpy(
            theta, wavelength_nm, global_params=self._legacy_globals(), return_components=return_components
        )

    def forward_torch(self, theta: torch.Tensor, wavelength_nm: np.ndarray) -> torch.Tensor:
        return self._registered.forward_torch(
            theta, wavelength_nm, global_params=self._legacy_globals(), return_components=False
        )  # type: ignore[return-value]

    def provenance(self) -> dict[str, Any]:
        provenance = self._registered.provenance()
        provenance.update(
            {
                "model_id": str(self.model_config["model_id"]),
                "legacy_wrapper_for": "P2",
                "references": list(self.model_config["references"]),
            }
        )
        return provenance
