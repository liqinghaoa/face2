"""YAML configuration helpers."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml


PACKAGE_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_DIR = PACKAGE_ROOT / "configs" / "so0"


def load_yaml(path: str | Path) -> dict[str, Any]:
    """Load a YAML file and require a mapping at the top level."""

    with Path(path).open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Expected YAML mapping: {path}")
    return data


def load_default_config() -> dict[str, Any]:
    """Load the frozen SO-0 MVP configuration."""

    return load_yaml(DEFAULT_CONFIG_DIR / "forward_model_mvp.yaml")


def load_thresholds() -> dict[str, Any]:
    """Load acceptance thresholds."""

    return load_yaml(DEFAULT_CONFIG_DIR / "acceptance_thresholds.yaml")


def load_audit_protocol() -> dict[str, Any]:
    """Load the SO-0 v1.1 audit protocol."""

    return load_yaml(DEFAULT_CONFIG_DIR / "audit_protocol_v1_1.yaml")


def load_formula_registry() -> dict[str, Any]:
    """Load the formula registry and ensure no empty fields are present."""

    registry = load_yaml(DEFAULT_CONFIG_DIR / "forward_formula_registry.yaml")

    def walk(value: Any, path: str) -> None:
        if value is None or value == "":
            raise ValueError(f"Empty formula registry field: {path}")
        if isinstance(value, dict):
            for key, child in value.items():
                walk(child, f"{path}.{key}")
        elif isinstance(value, list):
            for idx, child in enumerate(value):
                walk(child, f"{path}[{idx}]")

    walk(registry, "registry")
    return registry


@dataclass(frozen=True)
class SO0Config:
    """Strictly validated SO-0 configuration bundle.

    The object is intentionally lightweight: it wraps the three YAML files that
    define paths, physical parameters, thresholds, and audit grids. Numerical
    code receives this object instead of duplicating those constants.
    """

    model: dict[str, Any]
    thresholds: dict[str, Any]
    audit_protocol: dict[str, Any]
    config_dir: Path = DEFAULT_CONFIG_DIR

    @classmethod
    def load(cls, config_dir: str | Path | None = None) -> "SO0Config":
        """Load and validate SO-0 YAML configuration files."""

        root = Path(config_dir) if config_dir is not None else DEFAULT_CONFIG_DIR
        cfg = cls(
            model=load_yaml(root / "forward_model_mvp.yaml"),
            thresholds=load_yaml(root / "acceptance_thresholds.yaml"),
            audit_protocol=load_yaml(root / "audit_protocol_v1_1.yaml"),
            config_dir=root,
        )
        cfg.validate()
        return cfg

    @property
    def workspace_root(self) -> Path:
        """Project workspace root."""

        return Path(self.model["project"]["workspace_root"])

    @property
    def standardized_asset_root(self) -> Path:
        """Frozen standardized spectral asset root."""

        return self.workspace_root / self.model["project"]["standardized_asset_root"]

    @property
    def raw_asset_root(self) -> Path:
        """Frozen raw spectral asset root."""

        return self.workspace_root / self.model["project"]["raw_asset_root"]

    @property
    def output_dir(self) -> Path:
        """SO-0 v1.1 output directory."""

        return self.workspace_root / self.model["project"]["output_dir"]

    @property
    def reference_step_nm(self) -> int:
        """Reference wavelength grid step in nm."""

        return int(self.model["assets"]["reference_step_nm"])

    @property
    def production_step_nm(self) -> int:
        """Production wavelength grid step in nm."""

        return int(self.model["assets"]["production_step_nm"])

    def parameter(self, name: str) -> dict[str, Any]:
        """Return one parameter definition."""

        return self.model["parameters"][name]

    def primary_value(self, name: str) -> float:
        """Return a primary scalar parameter value."""

        return float(self.parameter(name)["primary"])

    def range_minmax(self, name: str, key: str = "primary_range") -> tuple[float, float]:
        """Return a configured min/max pair."""

        values = self.parameter(name)[key]
        return float(values[0]), float(values[1])

    def validate(self) -> None:
        """Validate required fields, ranges, and the production grid contract."""

        required_model_paths = [
            ("project", "workspace_root"),
            ("project", "standardized_asset_root"),
            ("project", "raw_asset_root"),
            ("project", "output_dir"),
            ("project", "random_seed"),
            ("assets", "required_decision_status"),
            ("assets", "reference_step_nm"),
            ("assets", "production_step_nm"),
            ("assets", "forbid_production_10nm"),
        ]
        for section, key in required_model_paths:
            if section not in self.model or key not in self.model[section]:
                raise ValueError(f"Missing configuration field: {section}.{key}")
        if self.reference_step_nm != 1:
            raise ValueError("reference_step_nm must be 1")
        if self.production_step_nm != 5:
            raise ValueError("production_step_nm must be 5; 10 nm production is forbidden")
        if not bool(self.model["assets"]["forbid_production_10nm"]):
            raise ValueError("forbid_production_10nm must be true")

        for name in [
            "melanin_fraction",
            "blood_fraction",
            "oxygenation",
            "epidermis_thickness_cm",
            "dermis_thickness_cm",
            "melanin_control",
            "hemoglobin_control",
            "shading",
            "specular",
            "exposure",
        ]:
            if name not in self.model["parameters"]:
                raise ValueError(f"Missing parameter configuration: {name}")
            values = self.model["parameters"][name]
            if "unit" not in values or values["unit"] in (None, ""):
                raise ValueError(f"Missing unit for parameter: {name}")
            if values.get("trainable") not in (True, False):
                raise ValueError(f"Missing trainable flag for parameter: {name}")

        for name in ["melanin_fraction", "blood_fraction", "melanin_control", "hemoglobin_control"]:
            values = self.parameter(name)
            lo = float(values["min"])
            hi = float(values["max"])
            if not lo < hi:
                raise ValueError(f"Invalid range for {name}: min must be less than max")
        for name in ["shading", "specular", "exposure"]:
            lo, hi = self.range_minmax(name, "primary_range")
            if not lo <= hi:
                raise ValueError(f"Invalid primary_range for {name}")

        required_threshold_sections = [
            "reflectance",
            "backend_parity",
            "color",
            "colorchecker",
            "separability",
            "resolution",
            "linearity",
        ]
        for section in required_threshold_sections:
            if section not in self.thresholds:
                raise ValueError(f"Missing threshold section: {section}")
            self._check_no_empty(self.thresholds[section], f"thresholds.{section}")
        self._check_no_empty(self.audit_protocol, "audit_protocol")

    def _check_no_empty(self, value: Any, path: str) -> None:
        if value is None or value == "":
            raise ValueError(f"Empty configuration field: {path}")
        if isinstance(value, dict):
            for key, child in value.items():
                self._check_no_empty(child, f"{path}.{key}")
        elif isinstance(value, list):
            if not value:
                raise ValueError(f"Empty configuration list: {path}")
            for idx, child in enumerate(value):
                self._check_no_empty(child, f"{path}[{idx}]")


@lru_cache(maxsize=8)
def _load_so0_config_cached(config_dir: str) -> SO0Config:
    return SO0Config.load(config_dir)


def load_so0_config(config_dir: str | Path | None = None) -> SO0Config:
    """Load the strict SO-0 configuration object."""

    root = Path(config_dir) if config_dir is not None else DEFAULT_CONFIG_DIR
    return _load_so0_config_cached(str(root.resolve()))
