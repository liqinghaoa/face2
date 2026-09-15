"""S1-4R revised forward models and auditable optical basis functions.

This module is deliberately independent of the frozen S1-4 v1 registry.  The
new proxy models operate in centered log-reflectance space; the K--M model is
an explicitly labelled semi-mechanistic bridge and never claims absolute
physiology.
"""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_REVISED_REGISTRY_PATH = PROJECT_ROOT / "configs/skin_optics_hsi/s1_4_revised_registry_v3.yaml"


@dataclass(frozen=True)
class RevisedParameter:
    name: str
    category: str
    minimum: float
    maximum: float
    reference_value: float
    unit: str
    interpretation: str

    @classmethod
    def from_mapping(cls, name: str, raw: Mapping[str, Any]) -> "RevisedParameter":
        result = cls(
            name=name,
            category=str(raw["category"]),
            minimum=float(raw["min"]),
            maximum=float(raw["max"]),
            reference_value=float(raw["reference_value"]),
            unit=str(raw["unit"]),
            interpretation=str(raw["interpretation"]),
        )
        if result.category not in {"bio", "nuisance", "latent"}:
            raise ValueError(f"Unknown revised parameter category: {result.category}")
        if not result.minimum < result.maximum or not result.minimum <= result.reference_value <= result.maximum:
            raise ValueError(f"Invalid revised parameter range: {name}")
        return result

    @property
    def bounds(self) -> tuple[float, float]:
        return self.minimum, self.maximum


@dataclass(frozen=True)
class RevisedModelSpec:
    model_id: str
    family: str
    theta_bio: tuple[str, ...]
    theta_nuisance: tuple[str, ...]
    status: str
    activation: str
    purpose: str

    @property
    def flat_parameter_names(self) -> tuple[str, ...]:
        return self.theta_bio + self.theta_nuisance


@dataclass(frozen=True)
class RevisedRegistry:
    raw: dict[str, Any]
    path: Path
    parameters: dict[str, RevisedParameter]
    models: dict[str, RevisedModelSpec]
    asset_path: Path
    srf_sensitivity_path: Path

    @property
    def wavelength_nm(self) -> np.ndarray:
        return np.asarray(self.raw["wavelength_contract"]["centers_nm"], dtype=np.float64)

    def model(self, model_id: str) -> RevisedModelSpec:
        try:
            return self.models[model_id]
        except KeyError as exc:
            raise KeyError(f"Unknown revised model: {model_id}") from exc

    def parameter(self, name: str) -> RevisedParameter:
        try:
            return self.parameters[name]
        except KeyError as exc:
            raise KeyError(f"Unknown revised parameter: {name}") from exc

    def bounds_for(self, model_id: str) -> tuple[tuple[float, float], ...]:
        return tuple(self.parameter(name).bounds for name in self.model(model_id).flat_parameter_names)

    def reference_values_for(self, model_id: str) -> np.ndarray:
        return np.asarray(
            [self.parameter(name).reference_value for name in self.model(model_id).flat_parameter_names],
            dtype=np.float64,
        )

    def global_value(self, name: str) -> Any:
        return self.raw["global_parameters"][name]["value"]

    def validate(self) -> None:
        if int(self.raw.get("schema_version", -1)) != 3 or self.raw.get("stage") != "S1-4R":
            raise ValueError("Expected revised S1-4R schema version 3")
        expected_wavelength = np.arange(400.0, 701.0, 10.0)
        if not np.array_equal(self.wavelength_nm, expected_wavelength):
            raise ValueError("Revised wavelength contract must be official 400:10:700 nm")
        contract = self.raw["wavelength_contract"]
        if contract.get("ordering") != "ascending" or contract.get("centers_status") != "confirmed_from_official_code":
            raise ValueError("Revised wavelength ordering/confirmation is invalid")
        if contract.get("effective_srf_status") != "missing":
            raise ValueError("effective_srf_missing must remain explicit")
        expected_models = {
            "B0-R", "B0-S", "B2-PCA", "B2-RPCA", "D1-M", "D2-MH", "D3-MHG", "D3-MHO",
            "K2-MH-KM", "K3-MHS-KM", "T2-MH-RTE",
        }
        if set(self.models) != expected_models:
            raise ValueError(f"Revised model registry mismatch: {sorted(set(self.models) ^ expected_models)}")
        for model in self.models.values():
            for name in model.flat_parameter_names:
                self.parameter(name)
        if not self.asset_path.is_file() or not self.srf_sensitivity_path.is_file():
            raise FileNotFoundError("Revised optical assets are missing")
        if bool(self.raw["observation_policy"]["per_spectrum_exposure_scale_allowed"]):
            raise ValueError("Per-spectrum scale must remain disabled")


def _resolve(root: Path, value: str) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (root / path).resolve()


def file_sha256(path: str | Path) -> str:
    digest = sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_revised_registry(path: str | Path | None = None) -> RevisedRegistry:
    registry_path = Path(path or DEFAULT_REVISED_REGISTRY_PATH).resolve()
    with registry_path.open("r", encoding="utf-8") as stream:
        raw = yaml.safe_load(stream)
    if not isinstance(raw, dict):
        raise ValueError("Revised registry must be a YAML mapping")
    parameters = {name: RevisedParameter.from_mapping(name, value) for name, value in raw["parameters"].items()}
    models = {
        model_id: RevisedModelSpec(
            model_id=model_id,
            family=str(value["family"]),
            theta_bio=tuple(str(v) for v in value["theta_bio"]),
            theta_nuisance=tuple(str(v) for v in value["theta_nuisance"]),
            status=str(value["status"]),
            activation=str(value["activation"]),
            purpose=str(value["purpose"]),
        )
        for model_id, value in raw["models"].items()
    }
    result = RevisedRegistry(
        raw=raw,
        path=registry_path,
        parameters=parameters,
        models=models,
        asset_path=_resolve(PROJECT_ROOT, str(raw["spectral_assets"]["hemoglobin_npz"])),
        srf_sensitivity_path=_resolve(PROJECT_ROOT, str(raw["spectral_assets"]["srf_sensitivity_npz"])),
    )
    result.validate()
    expected = raw["spectral_assets"].get("hemoglobin_sha256")
    expected_srf = raw["spectral_assets"].get("srf_sensitivity_sha256")
    if expected and file_sha256(result.asset_path) != str(expected).lower():
        raise ValueError("Revised 10 nm optical asset SHA-256 mismatch")
    if expected_srf and file_sha256(result.srf_sensitivity_path) != str(expected_srf).lower():
        raise ValueError("Revised 1 nm optical asset SHA-256 mismatch")
    return result


@dataclass(frozen=True)
class RevisedOpticalAssets:
    wavelength_nm: np.ndarray
    hbo2_mm1: np.ndarray
    hb_mm1: np.ndarray
    baseline_mm1: np.ndarray
    source_path: Path
    source_sha256: str

    @classmethod
    def load(cls, path: str | Path) -> "RevisedOpticalAssets":
        source = Path(path).resolve()
        with np.load(source, allow_pickle=False) as archive:
            required = {"wavelength_nm", "mua_hbo2_whole_blood_150gL_cm1", "mua_hb_whole_blood_150gL_cm1", "mua_base_cm1"}
            missing = required.difference(archive.files)
            if missing:
                raise ValueError(f"Optical asset missing keys: {sorted(missing)}")
            wavelength = np.asarray(archive["wavelength_nm"], dtype=np.float64)
            hbo2 = np.asarray(archive["mua_hbo2_whole_blood_150gL_cm1"], dtype=np.float64) / 10.0
            hb = np.asarray(archive["mua_hb_whole_blood_150gL_cm1"], dtype=np.float64) / 10.0
            baseline = np.asarray(archive["mua_base_cm1"], dtype=np.float64) / 10.0
        if wavelength.ndim != 1 or any(arr.shape != wavelength.shape for arr in (hbo2, hb, baseline)):
            raise ValueError("Optical asset arrays must be aligned one-dimensional arrays")
        if np.any(np.diff(wavelength) <= 0) or np.any(~np.isfinite(np.concatenate((hbo2, hb, baseline)))):
            raise ValueError("Optical asset contains invalid values")
        return cls(wavelength, hbo2, hb, baseline, source, file_sha256(source))

    def interp(self, values: np.ndarray, target_wavelength_nm: np.ndarray) -> np.ndarray:
        target = np.asarray(target_wavelength_nm, dtype=np.float64)
        if target.min() < self.wavelength_nm.min() or target.max() > self.wavelength_nm.max():
            raise ValueError("Requested wavelengths exceed revised optical asset coverage")
        return np.interp(target, self.wavelength_nm, values)

    def blood_mm1(self, target_wavelength_nm: np.ndarray, sO2: float) -> np.ndarray:
        if not 0.0 <= float(sO2) <= 1.0:
            raise ValueError("sO2 must be in [0, 1]")
        return float(sO2) * self.interp(self.hbo2_mm1, target_wavelength_nm) + (1.0 - float(sO2)) * self.interp(self.hb_mm1, target_wavelength_nm)

    def baseline(self, target_wavelength_nm: np.ndarray) -> np.ndarray:
        return self.interp(self.baseline_mm1, target_wavelength_nm)


def _validate_wavelength(wavelength_nm: np.ndarray) -> np.ndarray:
    result = np.asarray(wavelength_nm, dtype=np.float64)
    if result.ndim != 1 or result.size == 0 or np.any(~np.isfinite(result)) or np.any(np.diff(result) <= 0):
        raise ValueError("wavelength_nm must be a non-empty, strictly increasing finite array")
    return result


def center_weighted(values: np.ndarray, weights: np.ndarray | None = None) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    w = np.ones(values.shape[-1], dtype=np.float64) if weights is None else np.asarray(weights, dtype=np.float64)
    if values.shape[-1] != w.size or np.any(~np.isfinite(w)) or np.any(w <= 0):
        raise ValueError("weights must be finite and positive and match the wavelength dimension")
    return values - np.sum(values * w, axis=-1, keepdims=True) / np.sum(w)


def centered_log_ratio(
    observed_reflectance: np.ndarray,
    reference_reflectance: np.ndarray,
    weights: np.ndarray | None = None,
    epsilon: float = 1e-8,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    observed = np.asarray(observed_reflectance, dtype=np.float64)
    reference = np.asarray(reference_reflectance, dtype=np.float64)
    if observed.shape != reference.shape or observed.ndim != 1 or np.any(observed <= 0) or np.any(reference <= 0):
        raise ValueError("observed and reference reflectance must be aligned positive vectors")
    d = np.log(observed + epsilon) - np.log(reference + epsilon)
    w = np.ones(d.size, dtype=np.float64) if weights is None else np.asarray(weights, dtype=np.float64)
    if w.shape != d.shape or np.any(~np.isfinite(w)) or np.any(w <= 0):
        raise ValueError("weights must be finite, positive and aligned")
    a = float(np.sum(w * d) / np.sum(w))
    return d, a, center_weighted(d, w)


def packaging_factor(mu_a_blood_mm1: np.ndarray, vessel_diameter_um: float) -> np.ndarray:
    mu = np.asarray(mu_a_blood_mm1, dtype=np.float64)
    if np.any(~np.isfinite(mu)) or np.any(mu < 0):
        raise ValueError("mu_a_blood_mm1 must be finite and non-negative")
    if not np.isfinite(vessel_diameter_um) or vessel_diameter_um < 0:
        raise ValueError("vessel_diameter_um must be non-negative")
    if vessel_diameter_um == 0:
        return np.ones_like(mu)
    x = mu * float(vessel_diameter_um) / 1000.0
    result = np.empty_like(x)
    small = np.abs(x) < 1e-7
    result[small] = 1.0 - x[small] / 2.0 + x[small] * x[small] / 6.0
    result[~small] = -np.expm1(-x[~small]) / x[~small]
    return result


def revised_basis(
    wavelength_nm: np.ndarray,
    assets: RevisedOpticalAssets,
    *,
    weights: np.ndarray | None = None,
    melanin_power: float = 4.3,
    sO2_ref: float = 0.50,
    vessel_diameter_um: float = 15.0,
) -> dict[str, np.ndarray]:
    wavelength = _validate_wavelength(wavelength_nm)
    melanin = (wavelength / 570.0) ** (-float(melanin_power))
    blood = assets.blood_mm1(wavelength, sO2_ref)
    packaged = packaging_factor(blood, vessel_diameter_um) * blood
    ref = float(np.interp(570.0, wavelength, packaged))
    if ref <= 0 or not np.isfinite(ref):
        raise ValueError("Invalid packaged Hb reference")
    hb_shape = packaged / ref
    oxygen_low = packaging_factor(assets.blood_mm1(wavelength, 0.40), vessel_diameter_um) * assets.blood_mm1(wavelength, 0.40)
    oxygen_high = packaging_factor(assets.blood_mm1(wavelength, 0.60), vessel_diameter_um) * assets.blood_mm1(wavelength, 0.60)
    oxygen = oxygen_high - oxygen_low
    return {
        "phi_M": center_weighted(melanin, weights),
        "phi_H": center_weighted(hb_shape, weights),
        "phi_G": center_weighted(np.log(wavelength / 600.0), weights),
        "phi_O": center_weighted(oxygen, weights),
        "melanin_shape": melanin,
        "hb_shape": hb_shape,
    }


class RevisedSkinForwardModel:
    """Forward implementation for one model in the S1-4R v3 registry."""

    _ALLOWED_OVERRIDES = {
        "weights", "vessel_diameter_um", "sO2_ref", "epidermis_thickness_mm",
        "scattering_amplitude_mm1_at_600nm", "scattering_slope", "rayleigh_fraction",
        "global_reflectance_scale", "g_system", "side_log_gain_by_region",
        "train_reference_reflectance", "reference_reflectance", "a_obs", "pca_basis", "pca_mean",
    }

    def __init__(self, model_id: str, registry: RevisedRegistry | None = None, assets: RevisedOpticalAssets | None = None) -> None:
        self.registry = registry or load_revised_registry()
        self.spec = self.registry.model(model_id)
        self.model_id = model_id
        self.assets = assets or RevisedOpticalAssets.load(self.registry.asset_path)

    @property
    def parameter_names(self) -> tuple[str, ...]:
        return self.spec.flat_parameter_names

    @property
    def bounds(self) -> tuple[tuple[float, float], ...]:
        return self.registry.bounds_for(self.model_id)

    def _validate_theta(self, theta: np.ndarray | None) -> np.ndarray:
        n = len(self.parameter_names)
        if n == 0:
            if theta is None:
                return np.empty((0,), dtype=np.float64)
            arr = np.asarray(theta, dtype=np.float64)
            if arr.size != 0:
                raise ValueError(f"{self.model_id} has no free parameters")
            return arr.reshape(0)
        if theta is None:
            raise ValueError(f"theta is required for {self.model_id}")
        arr = np.asarray(theta, dtype=np.float64)
        if arr.ndim == 0 or arr.shape[-1] != n or np.any(~np.isfinite(arr)):
            raise ValueError(f"theta must have final dimension {n} for {self.model_id}")
        # Proxy coefficients are intentionally unbounded in the primary
        # analytic fit. Registry ranges for them are audit ranges, not clamps.
        if self.spec.family == "semi_mechanistic_km":
            for idx, name in enumerate(self.parameter_names):
                p = self.registry.parameter(name)
                if np.any((arr[..., idx] < p.minimum) | (arr[..., idx] > p.maximum)):
                    raise ValueError(f"{name} outside [{p.minimum}, {p.maximum}]")
        return arr

    def _globals(self, values: Mapping[str, Any] | None, wavelength_count: int = 31) -> dict[str, Any]:
        supplied = dict(values or {})
        unknown = set(supplied).difference(self._ALLOWED_OVERRIDES)
        if unknown:
            raise ValueError(f"Unknown revised global override(s): {sorted(unknown)}")
        result = {
            "weights": np.asarray(supplied.get("weights", np.ones(wavelength_count)), dtype=np.float64),
            "vessel_diameter_um": float(supplied.get("vessel_diameter_um", self.registry.global_value("vessel_diameter_um"))),
            "sO2_ref": float(supplied.get("sO2_ref", self.registry.global_value("sO2_ref"))),
            "epidermis_thickness_mm": float(supplied.get("epidermis_thickness_mm", self.registry.global_value("epidermis_thickness_mm"))),
            "scattering_amplitude_mm1_at_600nm": float(supplied.get("scattering_amplitude_mm1_at_600nm", self.registry.global_value("scattering_amplitude_mm1_at_600nm"))),
            "scattering_slope": float(supplied.get("scattering_slope", self.registry.global_value("scattering_slope"))),
            "rayleigh_fraction": float(supplied.get("rayleigh_fraction", self.registry.global_value("rayleigh_fraction"))),
            "global_reflectance_scale": float(supplied.get("global_reflectance_scale", self.registry.global_value("global_reflectance_scale"))),
            "g_system": float(supplied.get("g_system", 1.0)),
            "side_log_gain_by_region": dict(supplied.get("side_log_gain_by_region", self.registry.global_value("side_log_gain_by_region"))),
            "train_reference_reflectance": supplied.get("train_reference_reflectance"),
            "reference_reflectance": supplied.get("reference_reflectance"),
            "a_obs": float(supplied.get("a_obs", 0.0)),
            "pca_basis": supplied.get("pca_basis"),
            "pca_mean": supplied.get("pca_mean"),
        }
        if result["weights"].shape != (wavelength_count,) or np.any(~np.isfinite(result["weights"])) or np.any(result["weights"] <= 0):
            raise ValueError("weights must be positive, finite and wavelength aligned")
        if result["vessel_diameter_um"] < 0 or not 0.0 <= result["sO2_ref"] <= 1.0 or result["epidermis_thickness_mm"] <= 0:
            raise ValueError("Invalid revised global optical parameter")
        if result["g_system"] <= 0 or result["global_reflectance_scale"] <= 0:
            raise ValueError("Global gains must be positive")
        return result

    @staticmethod
    def _side_gain(values: Mapping[str, Any], context: Mapping[str, Any] | None) -> float:
        side = "other" if context is None else str(context.get("region_side", "other"))
        gains = values["side_log_gain_by_region"]
        if side not in gains:
            raise ValueError(f"Unknown region_side: {side}")
        return float(np.exp(float(gains[side])))

    def _raw_numpy(self, theta: np.ndarray, wavelength: np.ndarray, values: Mapping[str, Any], context: Mapping[str, Any] | None) -> tuple[np.ndarray, dict[str, Any]]:
        family = self.spec.family
        if family == "deferred_rte":
            raise RuntimeError("T2-MH-RTE is deferred because the Hyper-Skin observation operator is unavailable")
        if family == "raw_reference_baseline":
            ref = values.get("train_reference_reflectance")
            if ref is None:
                raise ValueError("B0-R requires train_reference_reflectance")
            result = np.asarray(ref, dtype=np.float64)
            if result.shape != wavelength.shape:
                raise ValueError("train_reference_reflectance is not wavelength aligned")
            return result, {"family": family}
        if family in {"centered_log_reference_baseline", "proxy_linear", "proxy_linear_oxygenation"}:
            basis = revised_basis(wavelength, self.assets, weights=values["weights"], sO2_ref=values["sO2_ref"], vessel_diameter_um=values["vessel_diameter_um"])
            y = np.zeros_like(wavelength)
            for idx, name in enumerate(self.parameter_names):
                if name == "delta_M_OD":
                    y = y - theta[idx] * basis["phi_M"]
                elif name == "delta_Hb_OD":
                    y = y - theta[idx] * basis["phi_H"]
                elif name == "q_tilt":
                    y = y + theta[idx] * basis["phi_G"]
                elif name == "delta_sO2":
                    y = y + theta[idx] * basis["phi_O"]
            return y, {"family": family, "basis": basis}
        if family in {"centered_log_pca", "raw_log_pca"}:
            basis = np.asarray(values.get("pca_basis"), dtype=np.float64)
            mean = np.asarray(values.get("pca_mean", np.zeros(wavelength.size)), dtype=np.float64)
            if basis.shape != (wavelength.size, len(theta)) or mean.shape != wavelength.shape:
                raise ValueError("PCA basis/mean are not aligned to theta and wavelength")
            return mean + basis @ theta, {"family": family, "pca_basis": basis, "pca_mean": mean}
        if family == "semi_mechanistic_km":
            bg = self.assets.baseline(wavelength)
            blood = self.assets.blood_mm1(wavelength, values["sO2_ref"])
            packaged = packaging_factor(blood, values["vessel_diameter_um"])
            m_idx = self.parameter_names.index("M_epi_OD")
            f_idx = self.parameter_names.index("f_blood_proxy")
            m_od = float(theta[m_idx])
            f_blood = float(theta[f_idx])
            tau_epi = values["epidermis_thickness_mm"] * bg + m_od * (wavelength / 570.0) ** (-float(self.registry.global_value("melanin_power")))
            mua_dermis = (1.0 - f_blood) * bg + f_blood * packaged * blood
            s_amp = float(values["scattering_amplitude_mm1_at_600nm"])
            if "S_amp" in self.parameter_names:
                s_amp = float(theta[self.parameter_names.index("S_amp")])
            ratio = wavelength / float(self.registry.global_value("scattering_reference_wavelength_nm"))
            musp = s_amp * ((1.0 - values["rayleigh_fraction"]) * ratio ** (-values["scattering_slope"]) + values["rayleigh_fraction"] * ratio ** (-4.0))
            K = 2.0 * mua_dermis
            S = 0.75 * musp - 0.25 * mua_dermis
            if np.any(S <= 0) or np.any(~np.isfinite(S)):
                raise ValueError("K--M S_KM is non-positive or invalid")
            x = K / S
            A = 1.0 + x
            dermal = A - np.sqrt(np.maximum(A * A - 1.0, 0.0))
            gain = values["global_reflectance_scale"] * values["g_system"] * self._side_gain(values, context)
            reflectance = gain * np.exp(-2.0 * tau_epi) * dermal
            if np.any(~np.isfinite(reflectance)) or np.any(reflectance <= 0):
                raise ValueError("Semi-mechanistic forward model produced invalid reflectance")
            return reflectance, {
                "family": family, "baseline_mm1": bg, "blood_mm1": blood, "packaging_factor": packaged,
                "mua_dermis_mm1": mua_dermis, "musp_mm1": musp, "K_KM": K, "S_KM": S,
                "tau_epi": tau_epi, "dermal_reflectance": dermal, "gain": gain,
            }
        raise ValueError(f"Unsupported revised model family: {family}")

    def forward_numpy(
        self,
        theta: np.ndarray | None,
        wavelength_nm: np.ndarray,
        global_params: Mapping[str, Any] | None = None,
        *,
        observation_space: str = "raw_reflectance",
        observation_context: Mapping[str, Any] | None = None,
        return_components: bool = False,
    ) -> np.ndarray | tuple[np.ndarray, dict[str, Any]]:
        wavelength = _validate_wavelength(wavelength_nm)
        values = self._globals(global_params, wavelength.size)
        if values["weights"].size != wavelength.size:
            raise ValueError("weights must match wavelength_nm")
        clean_theta = self._validate_theta(theta)
        raw, components = self._raw_numpy(clean_theta, wavelength, values, observation_context)
        if self.spec.family in {"centered_log_reference_baseline", "proxy_linear", "proxy_linear_oxygenation", "centered_log_pca"}:
            if observation_space == "centered_log_ratio":
                result = raw
            elif observation_space == "raw_reflectance":
                reference = values.get("reference_reflectance")
                if reference is None:
                    raise ValueError(f"{self.model_id} raw output requires reference_reflectance")
                reference = np.asarray(reference, dtype=np.float64)
                a_obs = float(values.get("a_obs", 0.0))
                if reference.shape != wavelength.shape or np.any(reference <= 0):
                    raise ValueError("reference_reflectance is not aligned positive")
                result = reference * np.exp(a_obs + raw)
            else:
                raise ValueError("observation_space must be raw_reflectance or centered_log_ratio")
        elif self.spec.family == "raw_log_pca":
            reference = values.get("reference_reflectance")
            if observation_space == "raw_reflectance":
                if reference is None:
                    raise ValueError("B2-RPCA raw output requires reference_reflectance")
                result = np.asarray(reference, dtype=np.float64) * np.exp(raw)
            elif observation_space == "centered_log_ratio":
                result = center_weighted(raw, values["weights"])
            else:
                raise ValueError("observation_space must be raw_reflectance or centered_log_ratio")
        elif observation_space == "raw_reflectance":
            result = raw
        elif observation_space == "centered_log_ratio":
            reference = values.get("reference_reflectance")
            if reference is None:
                raise ValueError("centered_log_ratio output requires reference_reflectance")
            reference = np.asarray(reference, dtype=np.float64)
            if reference.shape != wavelength.shape or np.any(reference <= 0):
                raise ValueError("reference_reflectance is not aligned positive")
            result = center_weighted(np.log(raw + 1e-8) - np.log(reference + 1e-8), values["weights"])
        else:
            raise ValueError("observation_space must be raw_reflectance or centered_log_ratio")
        if np.any(~np.isfinite(result)):
            raise ValueError("Revised forward output is non-finite")
        return (result, components) if return_components else result

    def forward_flat_numpy(self, theta: np.ndarray | None, wavelength_nm: np.ndarray, **kwargs: Any) -> Any:
        return self.forward_numpy(theta, wavelength_nm, **kwargs)

    def forward_torch(
        self,
        theta: torch.Tensor | None,
        wavelength_nm: np.ndarray,
        global_params: Mapping[str, Any] | None = None,
        *,
        observation_space: str = "raw_reflectance",
        observation_context: Mapping[str, Any] | None = None,
        return_components: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, dict[str, Any]]:
        # The torch path mirrors the NumPy equations while treating fixed
        # asset spectra as constants; this preserves gradients through theta.
        wavelength = _validate_wavelength(wavelength_nm)
        values = self._globals(global_params, wavelength.size)
        if theta is None:
            theta_t = torch.empty((0,), dtype=torch.float64)
        else:
            theta_t = theta
        if theta_t.ndim == 0 or theta_t.shape[-1] != len(self.parameter_names):
            raise ValueError(f"theta must have final dimension {len(self.parameter_names)} for {self.model_id}")
        if not bool(torch.all(torch.isfinite(theta_t)).detach().cpu()):
            raise ValueError("theta must be finite")
        if self.spec.family == "semi_mechanistic_km":
            for idx, name in enumerate(self.parameter_names):
                parameter = self.registry.parameter(name)
                outside = (theta_t[..., idx] < parameter.minimum) | (theta_t[..., idx] > parameter.maximum)
                if bool(torch.any(outside).detach().cpu()):
                    raise ValueError(f"{name} outside [{parameter.minimum}, {parameter.maximum}]")
        dtype, device = theta_t.dtype, theta_t.device
        wave = torch.as_tensor(wavelength, dtype=dtype, device=device)
        if self.spec.family == "semi_mechanistic_km":
            bg = torch.as_tensor(self.assets.baseline(wavelength), dtype=dtype, device=device)
            blood = torch.as_tensor(self.assets.blood_mm1(wavelength, values["sO2_ref"]), dtype=dtype, device=device)
            xpack = blood * float(values["vessel_diameter_um"]) / 1000.0
            pack = torch.where(torch.abs(xpack) < 1e-7, 1.0 - xpack / 2.0 + xpack.square() / 6.0, -torch.expm1(-xpack) / xpack)
            m_od = theta_t[..., self.parameter_names.index("M_epi_OD")]
            f_blood = theta_t[..., self.parameter_names.index("f_blood_proxy")]
            mel = (wave / 570.0) ** (-float(self.registry.global_value("melanin_power")))
            tau = float(values["epidermis_thickness_mm"]) * bg + m_od[..., None] * mel
            mua = (1.0 - f_blood[..., None]) * bg + f_blood[..., None] * pack * blood
            s_amp = torch.as_tensor(
                float(values["scattering_amplitude_mm1_at_600nm"]),
                dtype=dtype,
                device=device,
            )
            if "S_amp" in self.parameter_names:
                s_amp = theta_t[..., self.parameter_names.index("S_amp")]
            ratio = wave / float(self.registry.global_value("scattering_reference_wavelength_nm"))
            musp = s_amp[..., None] * ((1.0 - values["rayleigh_fraction"]) * ratio ** (-values["scattering_slope"]) + values["rayleigh_fraction"] * ratio ** -4.0)
            K, S = 2.0 * mua, 0.75 * musp - 0.25 * mua
            if bool(torch.any(S <= 0).detach().cpu()):
                raise ValueError("K--M S_KM is non-positive")
            A = 1.0 + K / S
            dermal = A - torch.sqrt(torch.clamp(A.square() - 1.0, min=0.0))
            gain = values["global_reflectance_scale"] * values["g_system"] * self._side_gain(values, observation_context)
            raw = gain * torch.exp(-2.0 * tau) * dermal
            components: dict[str, Any] = {"K_KM": K, "S_KM": S, "mua_dermis_mm1": mua, "musp_mm1": musp}
        elif self.spec.family in {"centered_log_reference_baseline", "proxy_linear", "proxy_linear_oxygenation"}:
            basis_np = revised_basis(wavelength, self.assets, weights=values["weights"], sO2_ref=values["sO2_ref"], vessel_diameter_um=values["vessel_diameter_um"])
            y = torch.zeros(theta_t.shape[:-1] + (wavelength.size,), dtype=dtype, device=device)
            for idx, name in enumerate(self.parameter_names):
                basis = torch.as_tensor(basis_np[{"delta_M_OD": "phi_M", "delta_Hb_OD": "phi_H", "q_tilt": "phi_G", "delta_sO2": "phi_O"}[name]], dtype=dtype, device=device)
                if name in {"delta_M_OD", "delta_Hb_OD"}:
                    y = y - theta_t[..., idx, None] * basis
                else:
                    y = y + theta_t[..., idx, None] * basis
            raw = y
            components = {"centered_log_ratio": y}
        elif self.spec.family in {"centered_log_pca", "raw_log_pca"}:
            basis = torch.as_tensor(np.asarray(values.get("pca_basis")), dtype=dtype, device=device)
            mean = torch.as_tensor(np.asarray(values.get("pca_mean", np.zeros(wavelength.size))), dtype=dtype, device=device)
            raw = mean + theta_t @ basis.T
            components = {"pca_basis": basis, "pca_mean": mean}
        elif self.spec.family == "raw_reference_baseline":
            ref = values.get("train_reference_reflectance")
            if ref is None:
                raise ValueError("B0-R requires train_reference_reflectance")
            raw = torch.as_tensor(np.asarray(ref), dtype=dtype, device=device)
            components = {"family": self.spec.family}
        else:
            raise RuntimeError("T2-MH-RTE is deferred or model family unsupported")
        if self.spec.family in {"centered_log_reference_baseline", "proxy_linear", "proxy_linear_oxygenation", "centered_log_pca"}:
            if observation_space == "centered_log_ratio":
                result = raw
            elif observation_space == "raw_reflectance":
                reference = values.get("reference_reflectance")
                if reference is None:
                    raise ValueError(f"{self.model_id} raw output requires reference_reflectance")
                ref = torch.as_tensor(np.asarray(reference), dtype=dtype, device=device)
                result = ref * torch.exp(float(values.get("a_obs", 0.0)) + raw)
            else:
                raise ValueError("observation_space must be raw_reflectance or centered_log_ratio")
        elif self.spec.family == "raw_log_pca":
            reference = values.get("reference_reflectance")
            if observation_space == "raw_reflectance":
                if reference is None:
                    raise ValueError("B2-RPCA raw output requires reference_reflectance")
                ref = torch.as_tensor(np.asarray(reference), dtype=dtype, device=device)
                result = ref * torch.exp(raw)
            elif observation_space == "centered_log_ratio":
                result = center_weighted_torch(raw, torch.as_tensor(values["weights"], dtype=dtype, device=device))
            else:
                raise ValueError("observation_space must be raw_reflectance or centered_log_ratio")
        elif observation_space == "centered_log_ratio":
            reference = values.get("reference_reflectance")
            if reference is None:
                raise ValueError("centered_log_ratio output requires reference_reflectance")
            ref = torch.as_tensor(np.asarray(reference), dtype=dtype, device=device)
            result = center_weighted_torch(torch.log(raw + 1e-8) - torch.log(ref + 1e-8), torch.as_tensor(values["weights"], dtype=dtype, device=device))
        elif observation_space == "raw_reflectance":
            result = raw
        else:
            raise ValueError("observation_space must be raw_reflectance or centered_log_ratio")
        return (result, components) if return_components else result

    def provenance(self) -> dict[str, Any]:
        """Return a serializable audit record for this model and its assets."""
        return {
            "registry_path": str(self.registry.path),
            "registry_id": self.registry.raw.get("registry_id"),
            "model_id": self.model_id,
            "family": self.spec.family,
            "parameter_names": list(self.parameter_names),
            "wavelength_nm": self.registry.wavelength_nm.tolist(),
            "optical_asset_path": str(self.assets.source_path),
            "optical_asset_sha256": self.assets.source_sha256,
        }


def revised_skin_forward(
    *,
    theta_bio: np.ndarray | None,
    wavelength_nm: np.ndarray,
    theta_nuisance: np.ndarray | None = None,
    global_params: Mapping[str, Any] | None = None,
    model_id: str,
    observation_context: Mapping[str, Any] | None = None,
    registry: RevisedRegistry | None = None,
) -> dict[str, Any]:
    """Return both observation spaces and their auditable model components."""
    model = RevisedSkinForwardModel(model_id, registry=registry)
    bio = np.empty(0) if theta_bio is None else np.asarray(theta_bio, dtype=np.float64).reshape(-1)
    nuisance = np.empty(0) if theta_nuisance is None else np.asarray(theta_nuisance, dtype=np.float64).reshape(-1)
    if bio.size != len(model.spec.theta_bio) or nuisance.size != len(model.spec.theta_nuisance):
        raise ValueError("theta_bio/theta_nuisance do not match the registered model contract")
    by_name = dict(zip(model.spec.theta_bio, bio)) | dict(zip(model.spec.theta_nuisance, nuisance))
    theta = np.asarray([by_name[name] for name in model.parameter_names], dtype=np.float64)
    values = dict(global_params or {})
    raw, components = model.forward_numpy(
        theta, wavelength_nm, values, observation_space="raw_reflectance",
        observation_context=observation_context, return_components=True,
    )
    if model.spec.family in {"centered_log_reference_baseline", "proxy_linear", "proxy_linear_oxygenation", "centered_log_pca"}:
        shape = model.forward_numpy(theta, wavelength_nm, values, observation_space="centered_log_ratio")
    else:
        reference = values.get("reference_reflectance")
        if reference is None:
            shape = None
        else:
            shape = model.forward_numpy(
                theta, wavelength_nm, values, observation_space="centered_log_ratio",
                observation_context=observation_context,
            )
    tier = "semi_mechanistic" if model.spec.family == "semi_mechanistic_km" else "proxy_or_baseline"
    return {
        "reflectance": raw,
        "centered_log_ratio": shape,
        "a_obs": float(values.get("a_obs", 0.0)),
        "components": components,
        "model_id": model_id,
        "evidence_tier": tier,
        "global_params": values,
        "provenance": model.provenance(),
    }


def center_weighted_torch(values: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    return values - torch.sum(values * weights, dim=-1, keepdim=True) / torch.sum(weights)


__all__ = [
    "RevisedRegistry", "RevisedOpticalAssets", "RevisedSkinForwardModel", "center_weighted",
    "centered_log_ratio", "file_sha256", "load_revised_registry", "packaging_factor", "revised_basis",
    "revised_skin_forward",
]
