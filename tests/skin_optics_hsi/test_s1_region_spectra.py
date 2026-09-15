from __future__ import annotations

import json
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import pytest

from skin_optics_hsi.s1_region_spectra import (
    _validate_wavelength_contract,
    extract_s1_3_region_spectra,
    sha256_file,
    summarize_region,
)


def test_summarize_region_computes_robust_and_sensitivity_statistics() -> None:
    values = np.asarray(
        [
            [0.1, 0.2, 0.3],
            [0.2, 0.3, 0.4],
            [0.3, 0.4, 0.5],
            [0.9, 0.9, 0.9],
        ],
        dtype=np.float32,
    )
    result = summarize_region(values, trim_fraction=0.25)
    np.testing.assert_allclose(result["median"], [0.25, 0.35, 0.45])
    np.testing.assert_allclose(result["mad"], [0.1, 0.1, 0.1], atol=1e-7)
    np.testing.assert_allclose(result["trimmed_mean"], [0.25, 0.35, 0.45])
    assert np.all(result["iqr"] > 0)


def test_wavelength_contract_requires_official_code_confirmation_and_retains_missing_srf() -> None:
    contract = {
        "wavelength": {
            "centers_nm": {
                "value": [float(value) for value in range(400, 701, 10)],
                "status": "confirmed_from_official_code",
            },
            "ordering": "ascending",
            "release_effective_bandwidth_or_srf": {"value": None, "status": "missing"},
        }
    }
    config = {
        "wavelength_contract": {
            "required_centers_status": "confirmed_from_official_code",
            "required_ordering": "ascending",
            "allow_missing_effective_srf": True,
        }
    }
    values = _validate_wavelength_contract(contract, config)
    assert values.shape == (31,)
    contract["wavelength"]["centers_nm"]["status"] = "inferred"
    with pytest.raises(ValueError, match="not frozen"):
        _validate_wavelength_contract(contract, config)


def test_s1_3_extracts_train_valid_and_never_reads_test(tmp_path: Path) -> None:
    stage_root = tmp_path / "HyperSkin_Stage1_v1"
    contracts = stage_root / "contracts"
    manifests = stage_root / "manifests"
    freeze = stage_root / "freeze"
    registration = stage_root / "registration_qc"
    masks = stage_root / "masks"
    raw = tmp_path / "raw"
    for path in (contracts, manifests, freeze, registration, masks, raw):
        path.mkdir(parents=True, exist_ok=True)

    split_rows = []
    mask_rows = []
    for split, sample_id, role in (
        ("train", "p001_neutral_front", "primary_development"),
        ("valid", "p002_neutral_front", "primary_validation"),
    ):
        cube_path = raw / f"{sample_id}.mat"
        spectrum = np.linspace(0.2, 0.8, 31, dtype=np.float32)
        stored = np.broadcast_to(spectrum[:, None, None], (31, 8, 8)).copy()
        with h5py.File(cube_path, "w") as archive:
            archive.create_dataset("cube", data=stored)
        split_rows.append(
            {
                "sample_id": sample_id,
                "split": split,
                "hsi_path": str(cube_path),
                "hsi_sha256": sha256_file(cube_path),
            }
        )
        sample_mask_root = masks / sample_id
        sample_mask_root.mkdir()
        paths = {}
        for region in ("left_cheek", "right_cheek", "forehead", "whole_skin"):
            mask_path = sample_mask_root / f"{region}.npy"
            np.save(mask_path, np.ones((8, 8), dtype=bool), allow_pickle=False)
            paths[f"{region}_path"] = str(mask_path)
        mask_rows.append(
            {
                "sample_id": sample_id,
                "subject_id": sample_id[:4],
                "split": split,
                "expression": "neutral",
                "direction": "front",
                "status": "PASS",
                "failure_codes": "",
                "review_codes": "",
                "s1_2_analysis_role": role,
                "s1_2_usable_flag": True,
                "anatomical_skin_pixels": 64,
                "radiometric_exclusion_fraction": 0.0,
                "illumination_exclusion_fraction": 0.0,
                **paths,
            }
        )
    forbidden_test_path = raw / "TEST_MUST_NOT_BE_OPENED.mat"
    split_rows.append(
        {
            "sample_id": "p003_neutral_front",
            "split": "test",
            "hsi_path": str(forbidden_test_path),
            "hsi_sha256": "not-computed-by-s1-3",
        }
    )
    split_manifest = manifests / "split_manifest.csv"
    pd.DataFrame(split_rows).to_csv(split_manifest, index=False)
    mask_manifest = manifests / "mask_manifest.parquet"
    pd.DataFrame(mask_rows).to_parquet(mask_manifest, index=False)
    contract = {
        "status": "PASS",
        "manifest": {"path": str(split_manifest), "sha256": sha256_file(split_manifest)},
        "hsi_storage": {"dataset_key": "cube"},
        "wavelength": {
            "centers_nm": {
                "value": [float(value) for value in range(400, 701, 10)],
                "status": "confirmed_from_official_code",
            },
            "ordering": "ascending",
            "release_effective_bandwidth_or_srf": {"value": None, "status": "missing"},
        },
    }
    contract_path = contracts / "data_contract.json"
    contract_path.write_text(json.dumps(contract), encoding="utf-8")
    target = {
        "status": "PASS_FOR_DEVELOPMENT",
        "next_stage_allowed": True,
        "authorized_next_stage": "S1-3",
        "test_access_count": 0,
        "manifest_path": str(mask_manifest),
        "manifest_sha256": sha256_file(mask_manifest),
    }
    (freeze / "s1_2_target_domain_final_decision.json").write_text(json.dumps(target), encoding="utf-8")
    (registration / "registration_decision.json").write_text(
        json.dumps({"status": "PASS", "frozen_transform": "transpose"}), encoding="utf-8"
    )

    decision = extract_s1_3_region_spectra(
        contract_path,
        mask_manifest,
        stage_root / "region_spectra",
        progress=False,
    )
    assert decision["status"] == "PASS_FOR_S1_4"
    assert decision["sample_count"] == 2
    assert decision["row_count"] == 8
    assert decision["test_access_count"] == 0
    assert not forbidden_test_path.exists()
    output = pd.read_parquet(stage_root / "region_spectra" / "region_spectra.parquet")
    assert set(output["split"]) == {"train", "valid"}
    assert output["extraction_status"].eq("USABLE").all()
