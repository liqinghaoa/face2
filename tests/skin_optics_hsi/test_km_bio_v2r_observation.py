from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from skin_optics_hsi.km_bio_v2r_observation import build_symmetric_observation, symmetric_log_reflectance


ROOT = Path(__file__).resolve().parents[2]


def test_symmetric_log_reflectance_is_geometric_mean():
    left = np.array([0.2, 0.4, 0.8])
    right = np.array([0.8, 0.4, 0.2])
    np.testing.assert_allclose(symmetric_log_reflectance(left, right), np.sqrt(left * right), atol=1e-15, rtol=0)


def test_symmetric_log_reflectance_rejects_nonpositive():
    try:
        symmetric_log_reflectance(np.array([0.2, 0.0]), np.array([0.3, 0.4]))
    except ValueError as exc:
        assert "strictly positive" in str(exc)
    else:
        raise AssertionError("nonpositive reflectance must be rejected")


def test_r_b_contract_builds_44_traceable_subject_spectra(tmp_path):
    source_contract = ROOT / "configs/skin_optics_hsi/km_bio_v2r_observation_contract.yaml"
    contract = yaml.safe_load(source_contract.read_text(encoding="utf-8"))
    contract["outputs"]["directory"] = str(tmp_path / "r_b_audit")
    temporary_contract = tmp_path / "km_bio_v2r_observation_contract.yaml"
    temporary_contract.write_text(yaml.safe_dump(contract, sort_keys=False), encoding="utf-8")

    summary = build_symmetric_observation(temporary_contract, ROOT)
    manifest = pd.read_parquet(summary["outputs"]["symmetric_manifest_parquet"]["path"])
    assert summary["status"] == "PASS_FOR_V2R_TRAIN_INVERSION"
    assert summary["hsi_content_reads"] == 0
    assert summary["validation_hsi_content_reads"] == 0
    assert summary["test_hsi_content_reads"] == 0
    assert summary["clinical_500_content_reads"] == 0
    assert summary["r_c0_train_inversion_allowed"]
    assert len(manifest) == manifest["subject_id"].nunique() == 44
    assert manifest["side_gain_applied"].eq(False).all()
    assert manifest["normalization_applied"].eq("none").all()
