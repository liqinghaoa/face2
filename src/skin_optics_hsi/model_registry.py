"""Validated S1-4 candidate-model and parameter contracts."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MODEL_REGISTRY_PATH = (
    PROJECT_ROOT / "configs" / "skin_optics_hsi" / "s1_4_model_registry_v1.yaml"
)


def _require(mapping: dict[str, Any], key: str, context: str) -> Any:
    if key not in mapping:
        raise ValueError(f"Missing configuration field: {context}.{key}")
    return mapping[key]


@dataclass(frozen=True)
class RegisteredParameter:
    name: str
    category: str
    unit: str
    minimum: float
    maximum: float
    reference_value: float
    interpretation: str
    range_status: str
    evidence: str

    @classmethod
    def from_mapping(cls, name: str, raw: dict[str, Any]) -> "RegisteredParameter":
        result = cls(
            name=name,
            category=str(_require(raw, "category", name)),
            unit=str(_require(raw, "unit", name)),
            minimum=float(_require(raw, "min", name)),
            maximum=float(_require(raw, "max", name)),
            reference_value=float(_require(raw, "reference_value", name)),
            interpretation=str(_require(raw, "interpretation", name)),
            range_status=str(_require(raw, "range_status", name)),
            evidence=str(_require(raw, "evidence", name)),
        )
        if result.category not in {"bio", "nuisance"}:
            raise ValueError(f"Unknown parameter category for {name}: {result.category}")
        if not result.minimum < result.maximum:
            raise ValueError(f"Invalid bounds for {name}")
        if not result.minimum <= result.reference_value <= result.maximum:
            raise ValueError(f"Reference value outside bounds for {name}")
        return result

    @property
    def bounds(self) -> tuple[float, float]:
        return self.minimum, self.maximum


@dataclass(frozen=True)
class RegisteredModel:
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

    @property
    def is_activated(self) -> bool:
        return self.activation == "always"


@dataclass(frozen=True)
class Stage1ModelRegistry:
    raw: dict[str, Any]
    path: Path
    parameters: dict[str, RegisteredParameter]
    models: dict[str, RegisteredModel]
    spectral_asset_path: Path

    @property
    def wavelength_nm(self) -> tuple[float, ...]:
        values = self.raw["wavelength_contract"]["centers_nm"]
        return tuple(float(value) for value in values)

    def parameter(self, name: str) -> RegisteredParameter:
        try:
            return self.parameters[name]
        except KeyError as exc:
            raise KeyError(f"Unknown S1-4 parameter: {name}") from exc

    def model(self, model_id: str) -> RegisteredModel:
        try:
            return self.models[model_id]
        except KeyError as exc:
            raise KeyError(f"Unknown S1-4 model: {model_id}") from exc

    def bounds_for(self, model_id: str, category: str | None = None) -> tuple[tuple[float, float], ...]:
        model = self.model(model_id)
        if category == "bio":
            names = model.theta_bio
        elif category == "nuisance":
            names = model.theta_nuisance
        elif category is None:
            names = model.flat_parameter_names
        else:
            raise ValueError("category must be bio, nuisance, or None")
        return tuple(self.parameter(name).bounds for name in names)

    def reference_values_for(self, model_id: str) -> tuple[float, ...]:
        return tuple(self.parameter(name).reference_value for name in self.model(model_id).flat_parameter_names)

    def global_value(self, name: str) -> Any:
        return self.raw["global_parameters"][name]["value"]

    def validate(self) -> None:
        expected_parameters = {
            "M_absorbance": "bio",
            "Hb_absorbance_proxy": "bio",
            "S_amp": "nuisance",
            "sO2": "nuisance",
        }
        if set(self.parameters) != set(expected_parameters):
            raise ValueError("S1-4 parameter registry must contain exactly M, H, S_amp, and sO2")
        for name, category in expected_parameters.items():
            if self.parameters[name].category != category:
                raise ValueError(f"Incorrect category for {name}")

        expected_models = {
            "B0": ((), ()),
            "B1": (("M_absorbance",), ()),
            "P2": (("M_absorbance", "Hb_absorbance_proxy"), ()),
            "P3-S": (("M_absorbance", "Hb_absorbance_proxy"), ("S_amp",)),
            "P3-O": (("M_absorbance", "Hb_absorbance_proxy"), ("sO2",)),
            "P4": (("M_absorbance", "Hb_absorbance_proxy"), ("S_amp", "sO2")),
        }
        if set(self.models) != set(expected_models):
            raise ValueError("S1-4 model registry must contain exactly B0/B1/P2/P3-S/P3-O/P4")
        for model_id, (bio, nuisance) in expected_models.items():
            model = self.models[model_id]
            if model.theta_bio != bio or model.theta_nuisance != nuisance:
                raise ValueError(f"Incorrect parameter order for {model_id}")
            for name in model.theta_bio:
                if self.parameter(name).category != "bio":
                    raise ValueError(f"{name} is not a biological proxy")
            for name in model.theta_nuisance:
                if self.parameter(name).category != "nuisance":
                    raise ValueError(f"{name} is not a nuisance parameter")

        wavelength = self.wavelength_nm
        if len(wavelength) != 31 or wavelength != tuple(float(value) for value in range(400, 701, 10)):
            raise ValueError("S1-4 requires the frozen official 400:10:700 nm wavelength centers")
        contract = self.raw["wavelength_contract"]
        if contract["ordering"] != "ascending" or contract["centers_status"] != "confirmed_from_official_code":
            raise ValueError("S1-4 requires confirmed ascending wavelength order")
        if contract["effective_srf_status"] != "missing":
            raise ValueError("The missing effective SRF status must remain explicit")
        if not self.spectral_asset_path.is_file():
            raise FileNotFoundError(f"Missing audited spectral asset: {self.spectral_asset_path}")

        observation = self.raw["observation_policy"]
        if bool(observation["per_spectrum_exposure_scale_allowed"]):
            raise ValueError("Per-spectrum exposure scale must remain disabled")
        if set(observation["allowed_region_sides"]) != {"image_left", "image_right", "other"}:
            raise ValueError("Unexpected region-side categories")
        if set(self.global_value("side_log_gain_by_region")) != {"image_left", "image_right", "other"}:
            raise ValueError("Fixed side-gain mapping must cover all allowed region sides")

        band_sets = self.raw["sensitivity"]["band_sets"]
        if set(band_sets) != {"full_31", "remove_endpoints", "remove_high_curvature_candidates"}:
            raise ValueError("Required pre-specified band sensitivities are missing")
        if [float(value) for value in band_sets["full_31"]["include_nm"]] != list(wavelength):
            raise ValueError("full_31 sensitivity does not match the wavelength contract")
        references = self.raw.get("references")
        if not isinstance(references, list) or not references or not all(isinstance(item, str) for item in references):
            raise ValueError("Every model reference must be a non-empty YAML string")

    def parameter_contract_payload(self) -> dict[str, Any]:
        return {
            "schema_version": int(self.raw["schema_version"]),
            "stage": "S1-4",
            "registry_id": self.raw["registry_id"],
            "status": self.raw["status"],
            "parameters": self.raw["parameters"],
            "global_parameters": self.raw["global_parameters"],
            "observation_policy": self.raw["observation_policy"],
            "interpretation_boundary": self.raw["interpretation_boundary"],
        }

    def model_registry_payload(self) -> dict[str, Any]:
        return {
            "schema_version": int(self.raw["schema_version"]),
            "stage": "S1-4",
            "registry_id": self.raw["registry_id"],
            "status": self.raw["status"],
            "wavelength_contract": self.raw["wavelength_contract"],
            "models": self.raw["models"],
            "fixed_physics": self.raw["fixed_physics"],
            "sensitivity": self.raw["sensitivity"],
            "references": self.raw["references"],
            "interpretation_boundary": self.raw["interpretation_boundary"],
        }


def load_model_registry(path: str | Path | None = None) -> Stage1ModelRegistry:
    registry_path = Path(path) if path is not None else DEFAULT_MODEL_REGISTRY_PATH
    registry_path = registry_path.resolve()
    with registry_path.open("r", encoding="utf-8") as stream:
        raw = yaml.safe_load(stream)
    if not isinstance(raw, dict):
        raise ValueError(f"Expected a YAML mapping: {registry_path}")

    parameter_raw = _require(raw, "parameters", "root")
    model_raw = _require(raw, "models", "root")
    if not isinstance(parameter_raw, dict) or not isinstance(model_raw, dict):
        raise ValueError("parameters and models must be mappings")
    parameters = {
        name: RegisteredParameter.from_mapping(name, value)
        for name, value in parameter_raw.items()
    }
    models = {
        model_id: RegisteredModel(
            model_id=model_id,
            family=str(_require(value, "family", model_id)),
            theta_bio=tuple(str(name) for name in _require(value, "theta_bio", model_id)),
            theta_nuisance=tuple(str(name) for name in _require(value, "theta_nuisance", model_id)),
            status=str(_require(value, "status", model_id)),
            activation=str(_require(value, "activation", model_id)),
            purpose=str(_require(value, "purpose", model_id)),
        )
        for model_id, value in model_raw.items()
    }
    asset_value = Path(str(raw["spectral_assets"]["hemoglobin_npz"]))
    asset_path = asset_value if asset_value.is_absolute() else PROJECT_ROOT / asset_value
    result = Stage1ModelRegistry(
        raw=raw,
        path=registry_path,
        parameters=parameters,
        models=models,
        spectral_asset_path=asset_path.resolve(),
    )
    result.validate()
    return result
