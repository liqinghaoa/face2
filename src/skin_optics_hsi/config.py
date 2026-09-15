"""Configuration loading and validation for stage-one HSI inversion."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "configs" / "skin_optics_hsi" / "stage1_candidate_v1.yaml"


def _require(mapping: dict[str, Any], key: str, context: str) -> Any:
    if key not in mapping:
        raise ValueError(f"Missing configuration field: {context}.{key}")
    return mapping[key]


@dataclass(frozen=True)
class ParameterSpec:
    name: str
    unit: str
    minimum: float
    maximum: float
    prior_center: float
    prior_scale: float

    @classmethod
    def from_mapping(cls, name: str, data: dict[str, Any]) -> "ParameterSpec":
        spec = cls(
            name=name,
            unit=str(_require(data, "unit", name)),
            minimum=float(_require(data, "min", name)),
            maximum=float(_require(data, "max", name)),
            prior_center=float(_require(data, "prior_center", name)),
            prior_scale=float(_require(data, "prior_scale", name)),
        )
        if not spec.minimum < spec.maximum:
            raise ValueError(f"Invalid bounds for {name}")
        if not spec.minimum <= spec.prior_center <= spec.maximum:
            raise ValueError(f"Prior center outside bounds for {name}")
        if spec.prior_scale <= 0:
            raise ValueError(f"Prior scale must be positive for {name}")
        return spec


@dataclass(frozen=True)
class Stage1Config:
    raw: dict[str, Any]
    path: Path
    parameters: tuple[ParameterSpec, ParameterSpec]
    spectral_asset_path: Path

    @property
    def wavelength_nm(self) -> list[float]:
        data = self.raw["data"]
        start = float(data["wavelength_start_nm"])
        stop = float(data["wavelength_stop_nm"])
        step = float(data["wavelength_step_nm"])
        count = int(round((stop - start) / step)) + 1
        return [start + i * step for i in range(count)]

    @property
    def bounds(self) -> tuple[tuple[float, float], tuple[float, float]]:
        return tuple((p.minimum, p.maximum) for p in self.parameters)  # type: ignore[return-value]

    @property
    def prior_centers(self) -> tuple[float, float]:
        return tuple(p.prior_center for p in self.parameters)  # type: ignore[return-value]

    @property
    def prior_scales(self) -> tuple[float, float]:
        return tuple(p.prior_scale for p in self.parameters)  # type: ignore[return-value]

    def validate(self) -> None:
        if [p.name for p in self.parameters] != ["M_absorbance", "Hb_absorbance_proxy"]:
            raise ValueError("Stage one v1 requires exactly M_absorbance and Hb_absorbance_proxy")
        wavelengths = self.wavelength_nm
        if wavelengths != sorted(wavelengths) or len(wavelengths) < 3:
            raise ValueError("Invalid wavelength grid")
        if wavelengths[0] < 400 or wavelengths[-1] > 720:
            raise ValueError("Candidate-v1 hemoglobin basis only supports 400-720 nm")
        if not self.spectral_asset_path.is_file():
            raise FileNotFoundError(f"Missing frozen spectral asset: {self.spectral_asset_path}")
        fit = self.raw["fit"]
        if int(fit["n_starts"]) < 2:
            raise ValueError("n_starts must be at least 2")
        if int(fit["noise_repeats"]) < 0:
            raise ValueError("noise_repeats cannot be negative")
        if float(fit["reflectance_epsilon"]) <= 0:
            raise ValueError("reflectance_epsilon must be positive")


def load_stage1_config(path: str | Path | None = None) -> Stage1Config:
    config_path = Path(path) if path is not None else DEFAULT_CONFIG_PATH
    config_path = config_path.resolve()
    with config_path.open("r", encoding="utf-8") as stream:
        raw = yaml.safe_load(stream)
    if not isinstance(raw, dict):
        raise ValueError(f"Expected a YAML mapping: {config_path}")
    parameter_map = _require(raw, "parameters", "root")
    if not isinstance(parameter_map, dict):
        raise ValueError("parameters must be a mapping")
    parameters = tuple(
        ParameterSpec.from_mapping(name, parameter_map[name])
        for name in ("M_absorbance", "Hb_absorbance_proxy")
    )
    asset_value = Path(str(raw["spectral_assets"]["source_standardized_npz"]))
    asset_path = asset_value if asset_value.is_absolute() else PROJECT_ROOT / asset_value
    config = Stage1Config(
        raw=raw,
        path=config_path,
        parameters=parameters,  # type: ignore[arg-type]
        spectral_asset_path=asset_path.resolve(),
    )
    config.validate()
    return config

