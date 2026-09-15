from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np

from skin_optics_hsi.data_contracts import (
    audit_data_contract,
    discover_hyperskin_vis,
    extract_mean_skin_spectrum,
    load_binary_mask,
    load_hyperskin_cube,
)


def _write_sample(root: Path, masks: Path, split: str, stem: str) -> None:
    vis = root / split / "VIS"
    vis.mkdir(parents=True, exist_ok=True)
    (masks / split).mkdir(parents=True, exist_ok=True)
    stored = np.linspace(0.1, 0.7, 31 * 5 * 4, dtype=np.float32).reshape(31, 5, 4)
    with h5py.File(vis / f"{stem}.mat", "w") as archive:
        archive.create_dataset("cube", data=stored)
    np.save(masks / split / f"{stem}.npy", np.ones((5, 4), dtype=np.uint8))


def test_hyperskin_discovery_loading_and_contract(tmp_path: Path) -> None:
    root = tmp_path / "Hyper-Skin(RGB, VIS)"
    masks = tmp_path / "masks"
    _write_sample(root, masks, "train", "p001_neutral_front")
    _write_sample(root, masks, "valid", "p002_neutral_front")
    _write_sample(root, masks, "test", "p003_neutral_front")
    samples = discover_hyperskin_vis(root, masks)
    assert len(samples) == 3
    audit = audit_data_contract(samples, {"train": 1, "valid": 1, "test": 1}, require_masks=True)
    assert audit["status"] == "PASS"
    cube = load_hyperskin_cube(samples[0].hsi_path)
    assert cube.shape == (5, 4, 31)
    mask = load_binary_mask(samples[0].mask_path, cube.shape[:2])
    spectrum, qc = extract_mean_skin_spectrum(cube, mask, 1e-6, 1.2)
    assert spectrum.shape == (31,)
    assert qc["valid_band_count"] == 31


def test_subject_leakage_is_critical(tmp_path: Path) -> None:
    root = tmp_path / "dataset"
    _write_sample(root, tmp_path / "masks", "train", "p001_neutral_front")
    _write_sample(root, tmp_path / "masks", "test", "p001_smile_left")
    samples = discover_hyperskin_vis(root, tmp_path / "masks")
    audit = audit_data_contract(samples, None, require_masks=True)
    assert audit["status"] == "FAIL"
    assert any(issue["code"] == "SUBJECT_LEAKAGE" for issue in audit["issues"])

