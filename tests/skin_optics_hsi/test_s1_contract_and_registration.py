from __future__ import annotations

import json
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
from PIL import Image

from skin_optics_hsi.s1_contract import build_s1_0_contract
from skin_optics_hsi.s1_registration import finalize_s1_1_manual_review, run_s1_1_registration


def _pattern(size: int = 64) -> np.ndarray:
    yy, xx = np.mgrid[:size, :size]
    image = 0.15 + 0.55 * (xx / size) + 0.15 * (yy / size)
    image += 0.25 * (((xx - 18) ** 2 + (yy - 43) ** 2) < 8**2)
    image -= 0.20 * (((xx - 47) ** 2 + (yy - 17) ** 2) < 5**2)
    return np.clip(image, 0.02, 0.98)


def _write_pair(root: Path, split: str, stem: str, brightness: float = 1.0) -> None:
    rgb_dir = root / split / "RGB"
    vis_dir = root / split / "VIS"
    rgb_dir.mkdir(parents=True, exist_ok=True)
    vis_dir.mkdir(parents=True, exist_ok=True)
    rgb_gray = np.clip(_pattern() * brightness, 0.0, 1.0)
    rgb = np.repeat(np.rint(rgb_gray[..., None] * 255).astype(np.uint8), 3, axis=-1)
    Image.fromarray(rgb).save(rgb_dir / f"{stem}.jpg", quality=100, subsampling=0)
    native_hsi = rgb_gray.T
    cube = np.stack([native_hsi * (0.8 + 0.004 * band) for band in range(31)], axis=0)
    with h5py.File(vis_dir / f"{stem}.mat", "w") as archive:
        archive.create_dataset("cube", data=cube.astype(np.float32))


def test_s1_0_writes_evidence_manifest_and_hashes(tmp_path: Path) -> None:
    root = tmp_path / "Hyper-Skin(RGB, VIS)"
    _write_pair(root, "train", "p001_neutral_front")
    _write_pair(root, "valid", "p002_neutral_front")
    _write_pair(root, "test", "p003_neutral_front")
    output = tmp_path / "processed"
    contract = build_s1_0_contract(
        root,
        output,
        expected_counts={"train": 1, "valid": 1, "test": 1},
        expected_spatial_shape=(64, 64),
        value_scan="full",
    )
    assert contract["status"] == "PASS"
    wavelength = contract["wavelength"]
    assert wavelength["centers_nm"]["status"] == "confirmed_from_official_code"
    assert wavelength["centers_nm"]["value"] == [float(value) for value in range(400, 701, 10)]
    assert wavelength["ordering"] == "ascending"
    assert wavelength["ordering_status"] == "confirmed_from_official_code"
    assert wavelength["release_effective_bandwidth_or_srf"]["status"] == "missing"
    issue_codes = {issue["code"] for issue in contract["issues"]}
    assert "WAVELENGTH_CENTERS_INFERRED" not in issue_codes
    assert "EFFECTIVE_SRF_MISSING" in issue_codes
    assert contract["hsi_storage"]["axis_contract"]["rgb_spatial_mapping"]["status"] == "pending"
    manifest = pd.read_csv(output / "manifests" / "split_manifest.csv")
    assert len(manifest) == 3
    assert manifest["rgb_sha256"].str.len().eq(64).all()
    assert manifest["hsi_sha256"].str.len().eq(64).all()
    assert (output / "contracts" / "contract_evidence.md").is_file()


def test_s1_1_finds_transpose_without_touching_non_train(tmp_path: Path) -> None:
    root = tmp_path / "Hyper-Skin(RGB, VIS)"
    subject = 1
    for expression in ("neutral", "smile"):
        for direction in ("front", "left", "right"):
            for brightness in (0.7, 1.0, 1.2):
                _write_pair(root, "train", f"p{subject:03d}_{expression}_{direction}", brightness)
                subject += 1
    _write_pair(root, "valid", "p100_neutral_front")
    _write_pair(root, "test", "p101_neutral_front")
    output = tmp_path / "processed"
    build_s1_0_contract(
        root,
        output,
        expected_counts={"train": 18, "valid": 1, "test": 1},
        expected_spatial_shape=(64, 64),
        value_scan="full",
    )
    decision = run_s1_1_registration(output / "contracts" / "data_contract.json", output, working_size=64)
    assert decision["data_scope"] == "train_only"
    assert decision["selection"]["sample_count"] == 18
    assert decision["automatic_candidate_transform"] == "transpose"
    assert decision["automatic_decision"] == "PASS"
    assert decision["coordinate_mapping_frozen"] is False
    assert decision["manual_review"]["status"] == "pending"
    assert (output / "registration_qc" / "registration_qc.parquet").is_file()
    assert (output / "registration_qc" / "registration_contact_sheet.png").is_file()
    saved = json.loads((output / "registration_qc" / "registration_decision.json").read_text(encoding="utf-8"))
    assert saved["next_stage_allowed"] is False
    frozen = finalize_s1_1_manual_review(
        output / "registration_qc",
        reviewer="unit-test researcher",
        approve=True,
        notes="All anatomical edges align in the synthetic review panels.",
    )
    assert frozen["status"] == "PASS"
    assert frozen["coordinate_mapping_frozen"] is True
    assert frozen["frozen_transform"] == "transpose"
    assert frozen["next_stage_allowed"] is True
