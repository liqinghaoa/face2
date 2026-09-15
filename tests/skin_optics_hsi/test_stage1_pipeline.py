from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
import yaml

from skin_optics_hsi.config import DEFAULT_CONFIG_PATH, load_stage1_config
from skin_optics_hsi.skin_forward import SkinForwardModel
from skin_optics_hsi.stage1_pipeline import run_stage1


def _write_cube(root: Path, masks: Path, split: str, stem: str, spectrum: np.ndarray) -> None:
    hsi_dir = root / split / "VIS"
    mask_dir = masks / split
    hsi_dir.mkdir(parents=True, exist_ok=True)
    mask_dir.mkdir(parents=True, exist_ok=True)
    stored = np.broadcast_to(spectrum[:, None, None], (31, 4, 5)).astype(np.float32)
    with h5py.File(hsi_dir / f"{stem}.mat", "w") as archive:
        archive.create_dataset("cube", data=stored)
    np.save(mask_dir / f"{stem}.npy", np.ones((4, 5), dtype=np.uint8))


def test_development_pipeline_writes_full_artifact_contract(tmp_path: Path) -> None:
    with DEFAULT_CONFIG_PATH.open("r", encoding="utf-8") as stream:
        raw = yaml.safe_load(stream)
    raw["fit"]["n_starts"] = 3
    raw["fit"]["noise_repeats"] = 2
    config_path = tmp_path / "stage1_smoke.yaml"
    config_path.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")
    config = load_stage1_config(config_path)
    model = SkinForwardModel(config)
    wavelength = np.asarray(config.wavelength_nm)

    data_root = tmp_path / "Hyper-Skin(RGB, VIS)"
    mask_root = tmp_path / "masks"
    _write_cube(data_root, mask_root, "train", "p001_neutral_front", model.forward_numpy(np.array([0.14, 0.12]), wavelength))
    _write_cube(data_root, mask_root, "valid", "p002_neutral_front", model.forward_numpy(np.array([0.22, 0.20]), wavelength))
    # This valid-shape Test file is discovered for leakage auditing but its data
    # must not be opened by a development run.
    _write_cube(data_root, mask_root, "test", "p003_neutral_front", np.full(31, np.nan))
    output = tmp_path / "stage1_output"
    decision = run_stage1(data_root, mask_root, output, config_path=config_path, mode="development")
    assert decision["decision"] == "NOT_EVALUATED"
    required = {
        "data_contract.json",
        "split_manifest.csv",
        "parameter_contract.yaml",
        "forward_model_config.yaml",
        "global_calibration.json",
        "per_spectrum_fit.parquet",
        "multistart_stability.parquet",
        "wavelength_residuals.csv",
        "identifiability_report.json",
        "STAGE1_DECISION.md",
        "resolved_stage1_config.yaml",
    }
    assert required.issubset({path.name for path in output.iterdir()})
    assert (output / "figures" / "stage1_diagnostics.svg").is_file()
    assert (output / "figures" / "stage1_diagnostics.png").is_file()

