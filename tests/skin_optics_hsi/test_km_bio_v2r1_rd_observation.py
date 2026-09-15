from pathlib import Path

import numpy as np
import yaml

from src.skin_optics_hsi.km_bio_v2r1_rd_observation import _resolve


ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "configs/skin_optics_hsi/km_bio_v2r1_rd_validation_observation_contract.yaml"
TRAIN_CONFIG = ROOT / "configs/skin_optics_hsi/km_bio_v2r_observation_contract.yaml"


def _config():
    return yaml.safe_load(CONFIG.read_text(encoding="utf-8"))


def test_rd_observation_contract_is_validation_only():
    config = _config()
    assert config["contract_id"] == "km_bio_v2r1_rd_validation_observation_contract"
    assert config["scope"]["split"] == "valid"
    assert config["scope"]["expected_subjects"] == 3
    assert config["scope"]["expected_captures"] == 3
    assert config["scope"]["expected_input_region_spectra"] == 6
    assert config["scope"]["expected_output_symmetric_spectra"] == 3
    assert config["scope"]["expected_subject_ids"] == ["p001", "p016", "p030"]
    for key in ("train_content_allowed", "test_content_allowed", "clinical_500_content_allowed"):
        assert config["scope"][key] is False


def test_rd_observation_inputs_exist_and_point_at_the_valid_split():
    config = _config()
    for value in config["inputs"].values():
        assert _resolve(ROOT, value).exists()
    for value in config["implementation"].values():
        assert _resolve(ROOT, value).exists()


def test_rd_observation_mirrors_the_train_wavelength_and_observation_contract():
    config = _config()
    train = yaml.safe_load(TRAIN_CONFIG.read_text(encoding="utf-8"))
    assert config["wavelength"]["centers_nm"] == train["wavelength"]["centers_nm"]
    assert config["wavelength"]["fit_centers_nm"] == train["wavelength"]["fit_centers_nm"]
    assert config["wavelength"]["fit_index_zero_based"] == train["wavelength"]["fit_index_zero_based"]
    assert config["wavelength"]["edge_diagnostic_index_zero_based"] == train["wavelength"]["edge_diagnostic_index_zero_based"]
    assert config["observation"]["formula"] == train["observation"]["formula"]
    assert config["observation"]["input_statistic"] == train["observation"]["input_statistic"]
    assert config["observation"]["side_gain_applied"] is False
    assert config["observation"]["per_spectrum_gain_fitted"] is False
    assert config["observation"]["normalization_applied"] == "none"


def test_rd_observation_spatial_contract_is_the_frozen_transpose_reader():
    config = _config()
    assert config["spatial"]["hsi_dataset_key"] == "cube"
    assert config["spatial"]["expected_stored_shape"] == [31, 1024, 1024]
    assert config["spatial"]["expected_stored_dtype"] == "float64"
    assert config["spatial"]["frozen_rgb_hsi_transform"] == "transpose"
    assert config["spatial"]["minimum_region_pixels"] == 1000
    assert config["spatial"]["required_path_fragment"] == "valid"


def test_rd_observation_declares_the_full_check_set():
    config = _config()
    required = set(config["audit"]["required_checks"])
    checks = {
        "v2r_train_observation_contract_pass",
        "source_manifest_rows_are_valid_split_only",
        "no_train_or_test_path_is_opened",
        "exact_left_right_pairing",
        "raw_float64_vs_float32_cache_within_tolerance",
        "symmetric_formula_reconstruction",
        "all_symmetric_values_finite_positive",
    }
    assert checks.issubset(required)
    assert config["audit"]["expected_validation_hsi_content_reads"] == 3
    assert config["audit"]["expected_train_hsi_content_reads"] == 0
    assert config["audit"]["expected_test_hsi_content_reads"] == 0


def test_frozen_valid_split_contains_the_declared_subjects():
    import pandas as pd

    split = pd.read_csv(ROOT / "data/processed/HyperSkin_Stage1_v1/manifests/split_manifest.csv")
    valid_front = split.loc[
        split["split"].eq("valid") & split["expression"].eq("neutral") & split["direction"].eq("front")
    ]
    assert sorted(valid_front["subject_id"].astype(str)) == ["p001", "p016", "p030"]
    assert valid_front["hsi_path"].str.contains("valid").all()
    assert np.all(valid_front["hsi_stored_shape"].astype(str).str.contains("31", regex=False))
