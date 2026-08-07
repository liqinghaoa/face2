"""Frozen spectral asset loading and validation."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from skin_optics.config import load_default_config
from skin_optics.types import FloatArray


SOURCE_KEYS = {
    "wavelength_nm",
    "integration_weights_nm",
    "cie_xyz_2deg",
    "core_illuminants_raw",
    "core_illuminants_y1",
    "core_illuminant_names",
    "camera_ssf",
    "camera_names",
    "camera_channel_names",
    "colorchecker_reflectance",
    "colorchecker_patch_names",
    "epsilon_hbo2_cm1_per_mol_L",
    "epsilon_hb_cm1_per_mol_L",
}
DERIVED_KEYS = {
    "mua_hbo2_whole_blood_150gL_cm1",
    "mua_hb_whole_blood_150gL_cm1",
    "mua_blood_y060_cm1",
    "mua_blood_y075_cm1",
    "mua_blood_y090_cm1",
    "mua_mel_primary_cm1",
    "mua_mel_alternative_cm1",
    "mua_base_cm1",
    "musp_rayleigh_cm1",
    "musp_mie_cm1",
    "musp_total_cm1",
}


@dataclass(frozen=True)
class SpectralAssets:
    """Validated frozen SO-0 spectral assets for one wavelength grid."""

    step_nm: int
    root: Path
    source: dict[str, np.ndarray]
    derived: dict[str, FloatArray]

    @property
    def wavelength_nm(self) -> FloatArray:
        return self.source["wavelength_nm"].astype(np.float64, copy=False)

    @property
    def weights_nm(self) -> FloatArray:
        return self.source["integration_weights_nm"].astype(np.float64, copy=False)

    @property
    def n_lambda(self) -> int:
        return int(self.wavelength_nm.shape[0])

    def illuminant_index(self, name: str) -> int:
        names = [str(x) for x in self.source["core_illuminant_names"]]
        if name not in names:
            raise ValueError(f"Unknown illuminant {name!r}; available={names}")
        return names.index(name)

    def camera_index(self, name: str) -> int:
        names = [str(x) for x in self.source["camera_names"]]
        if name not in names:
            raise ValueError(f"Unknown camera {name!r}; available={names}")
        return names.index(name)


def standardized_root(workspace_root: str | Path | None = None) -> Path:
    """Return the frozen standardized asset root from configuration."""

    cfg = load_default_config()
    workspace = Path(workspace_root or cfg["project"]["workspace_root"])
    return workspace / cfg["project"]["standardized_asset_root"]


def read_standardization_decision(root: str | Path | None = None) -> dict[str, object]:
    """Read and validate the standardized asset decision file."""

    asset_root = Path(root) if root is not None else standardized_root()
    path = asset_root / "audits" / "standardization_decision.json"
    with path.open("r", encoding="utf-8") as f:
        decision = json.load(f)
    if decision.get("status") != "PASS_WITH_5NM_FALLBACK":
        raise ValueError(f"Unexpected standardization status: {decision.get('status')}")
    if int(decision.get("production_step_nm")) != 5:
        raise ValueError(f"Unexpected production_step_nm: {decision.get('production_step_nm')}")
    return decision


def _npz_to_dict(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as data:
        return {key: data[key] for key in data.files}


def _require_finite(name: str, array: np.ndarray) -> None:
    if np.issubdtype(array.dtype, np.number) and not np.all(np.isfinite(array)):
        raise ValueError(f"Non-finite values in {name}")


def _validate_assets(step_nm: int, source: dict[str, np.ndarray], derived: dict[str, np.ndarray]) -> None:
    missing_source = SOURCE_KEYS.difference(source)
    missing_derived = DERIVED_KEYS.difference(derived)
    if missing_source or missing_derived:
        raise ValueError(f"Missing source={missing_source}, derived={missing_derived}")
    expected_n = 321 if step_nm == 1 else 65 if step_nm == 5 else None
    if expected_n is None:
        raise ValueError("Only 1 nm reference and 5 nm production grids are accepted")
    if source["wavelength_nm"].shape != (expected_n,):
        raise ValueError(f"Unexpected wavelength shape for {step_nm} nm")
    if source["camera_ssf"].shape != (28, expected_n, 3):
        raise ValueError(f"Unexpected camera_ssf shape for {step_nm} nm")
    if source["colorchecker_reflectance"].shape != (24, expected_n):
        raise ValueError(f"Unexpected ColorChecker shape for {step_nm} nm")
    for key, value in source.items():
        _require_finite(f"source.{key}", value)
        if np.issubdtype(value.dtype, np.floating) and value.dtype != np.float64:
            raise ValueError(f"Expected float64 source array for {key}, got {value.dtype}")
    for key, value in derived.items():
        _require_finite(f"derived.{key}", value)
        if value.shape != (expected_n,):
            raise ValueError(f"Unexpected derived shape for {key}: {value.shape}")
        if value.dtype != np.float64:
            raise ValueError(f"Expected float64 derived array for {key}, got {value.dtype}")


def load_assets(step_nm: int = 5, workspace_root: str | Path | None = None) -> SpectralAssets:
    """Load validated reference 1 nm or production 5 nm assets."""

    if step_nm not in (1, 5):
        raise ValueError("Only 1 nm reference and 5 nm production assets are allowed")
    root = standardized_root(workspace_root)
    read_standardization_decision(root)
    subdir = "reference_1nm" if step_nm == 1 else "production_5nm"
    if step_nm == 5 and (root / "production_10nm").exists():
        # The directory may exist as an audited candidate, but it is never used.
        pass
    source_path = root / subdir / f"source_standardized_{step_nm}nm.npz"
    derived_path = root / subdir / f"derived_optics_{step_nm}nm.npz"
    source = _npz_to_dict(source_path)
    derived = _npz_to_dict(derived_path)
    _validate_assets(step_nm, source, derived)
    return SpectralAssets(step_nm=step_nm, root=root, source=source, derived=derived)
