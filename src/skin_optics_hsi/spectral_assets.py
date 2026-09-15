"""Read-only access to audited hemoglobin spectra used by candidate-v1."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class HemoglobinSpectra:
    wavelength_nm: np.ndarray
    epsilon_hbo2: np.ndarray
    epsilon_hb: np.ndarray
    source_path: Path
    source_sha256: str

    def mixed_normalized(self, target_wavelength_nm: np.ndarray, oxygenation: float, reference_nm: float) -> np.ndarray:
        if not 0.0 <= oxygenation <= 1.0:
            raise ValueError("oxygenation must be in [0, 1]")
        target = np.asarray(target_wavelength_nm, dtype=np.float64)
        if target.min() < self.wavelength_nm.min() or target.max() > self.wavelength_nm.max():
            raise ValueError("Requested wavelengths exceed hemoglobin asset coverage")
        mixed = oxygenation * self.epsilon_hbo2 + (1.0 - oxygenation) * self.epsilon_hb
        interpolated = np.interp(target, self.wavelength_nm, mixed)
        reference = float(np.interp(reference_nm, self.wavelength_nm, mixed))
        if reference <= 0 or not np.isfinite(reference):
            raise ValueError("Invalid hemoglobin normalization reference")
        return interpolated / reference


def _file_sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_hemoglobin_spectra(path: str | Path) -> HemoglobinSpectra:
    """Load the frozen OMLC-derived spectra without importing legacy SO-0 code."""

    source_path = Path(path).resolve()
    with np.load(source_path, allow_pickle=False) as archive:
        required = {"wavelength_nm", "epsilon_hbo2_cm1_per_mol_L", "epsilon_hb_cm1_per_mol_L"}
        missing = required.difference(archive.files)
        if missing:
            raise ValueError(f"Spectral asset is missing keys: {sorted(missing)}")
        wavelength = np.asarray(archive["wavelength_nm"], dtype=np.float64)
        hbo2 = np.asarray(archive["epsilon_hbo2_cm1_per_mol_L"], dtype=np.float64)
        hb = np.asarray(archive["epsilon_hb_cm1_per_mol_L"], dtype=np.float64)
    if wavelength.ndim != 1 or hbo2.shape != wavelength.shape or hb.shape != wavelength.shape:
        raise ValueError("Hemoglobin asset arrays must be one-dimensional and aligned")
    if np.any(np.diff(wavelength) <= 0) or np.any(hbo2 <= 0) or np.any(hb <= 0):
        raise ValueError("Hemoglobin spectra must be positive on a strictly increasing grid")
    return HemoglobinSpectra(wavelength, hbo2, hb, source_path, _file_sha256(source_path))

