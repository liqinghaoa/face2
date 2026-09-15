from pathlib import Path

import numpy as np
import yaml

from src.skin_optics_hsi.km_bio_v2r1_stage_c1r import (
    BEST_O_HI,
    BEST_O_LO,
    BASE_PARAMETER_NAMES,
    BEST_O_PARAMETER_NAMES,
    FIT_HI,
    FIT_LO,
    _base_variant,
    _classify,
    _profile_envelope,
    _resolve,
    _variants,
)


ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "configs/skin_optics_hsi/km_bio_v2r1_stage_c1r.yaml"


def _config():
    return yaml.safe_load(CONFIG.read_text(encoding="utf-8"))


def test_c1r_config_is_train_only_and_runs_best_o():
    config = _config()
    assert config["task_id"] == "R-C1R_COMPLETE_IDENTIFIABILITY_AND_BEST_O"
    assert config["execution"]["train_only"] is True
    assert config["execution"]["run_best_o"] is True
    assert config["execution"]["stop_on_diagnostic_flags"] is False
    for key in ("raw_hsi_content_allowed", "rgb_content_allowed", "validation_content_allowed", "test_content_allowed", "clinical_500_content_allowed"):
        assert config["access_policy"][key] is False
    for key in ("run_validation", "run_test", "run_clinical_500", "run_rgb_encoder_training"):
        assert config["execution"][key] is False


def test_c1r_inputs_and_frozen_fold_are_declared():
    config = _config()
    for value in config["inputs"].values():
        assert _resolve(ROOT, value).exists()
    assert config["folds"]["expected_sha256"] == "e763a9f3dc7f58babe264f7056699c3d5b3722187fbb558f83f2fdb5d6b67370"
    assert config["solver"]["profile_grid_points"] == 51
    assert config["best_o"]["parameter_names"] == ["f_mel", "f_blood", "s"]


def test_c1r_identifiability_rule_is_the_registered_one():
    config = _config()
    ident = config["identifiability"]
    assert ident["acceptable_loss_delta_logrmse"] == 0.005
    assert ident["maximum_acceptable_normalized_envelope_span"] == 0.20
    assert ident["reliable_min_identifiable_fraction"] == 0.80
    assert ident["conditional_min_identifiable_fraction"] == 0.50


def test_profile_envelope_uses_acceptable_loss_delta():
    rows = [
        {"grid_value_u": 0.20, "logrmse": 0.30, "solver_success": True},
        {"grid_value_u": 0.40, "logrmse": 0.10, "solver_success": True},
        {"grid_value_u": 0.50, "logrmse": 0.101, "solver_success": True},
        {"grid_value_u": 0.60, "logrmse": 0.106, "solver_success": True},
        {"grid_value_u": 0.80, "logrmse": 0.40, "solver_success": True},
    ]
    envelope = _profile_envelope(rows, 0.005)
    assert envelope["optimum_logrmse"] == 0.10
    assert envelope["envelope_low_u"] == 0.40
    assert envelope["envelope_high_u"] == 0.50
    assert np.isclose(envelope["envelope_span_u"], 0.10)
    assert envelope["all_grid_points_converged"] is True


def test_parameter_bounds_match_the_v2r_formula_contract():
    assert list(FIT_LO) == [0.0, 0.0]
    assert list(FIT_HI) == [0.43, 0.10]
    assert list(BEST_O_LO) == [0.0, 0.0, 0.0]
    assert list(BEST_O_HI) == [0.43, 0.10, 1.0]
    assert BASE_PARAMETER_NAMES == ("f_mel", "f_blood")
    assert BEST_O_PARAMETER_NAMES == ("f_mel", "f_blood", "s")


def test_sensitivity_grid_matches_the_registered_contract():
    config = _config()
    globals_row = {"A_s": 1.40, "delta_bs": 0.50, "g0": 1.05}
    base = _base_variant(config, globals_row)
    base_variants = _variants(config, base, include_s0_setting=True)
    settings = [row["setting"] for row in base_variants]
    assert settings[0] == "baseline"
    assert settings.count("diameter_um") == 4
    assert settings.count("s0") == 3
    assert settings.count("epidermis_thickness_mm") == 3
    assert settings.count("whole_blood_hb_g_l") == 3
    assert settings.count("scattering_amplitude_multiplier") == 3
    assert settings.count("delta_bs_shift") == 2
    assert settings.count("bandwidth_430_670") == 1
    assert settings.count("bandwidth_440_660") == 1
    best_o_variants = _variants(config, base, include_s0_setting=False)
    best_o_settings = [row["setting"] for row in best_o_variants]
    assert "s0" not in best_o_settings
    assert set(best_o_settings) == set(settings) - {"s0"}
    assert max(row["scattering_amplitude"] for row in base_variants) <= 1.6
    assert max(row["delta_bs"] for row in base_variants) <= 0.5


def test_reliable_conditional_unreliable_layering():
    config = _config()
    rules = config["identifiability"]
    assert _classify(0.95, 0.05, "f_mel", rules) == "reliable"
    assert _classify(0.95, 0.30, "f_blood", rules) == "conditional"
    assert _classify(0.95, 0.90, "f_blood", rules) == "unreliable"
    assert _classify(0.60, 0.10, "f_mel", rules) == "conditional"
    assert _classify(0.40, 0.10, "f_mel", rules) == "unreliable"
    assert _classify(0.85, 0.15, "s", rules) == "reliable"
