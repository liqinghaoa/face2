from pathlib import Path

import json

import numpy as np
import pandas as pd
import yaml

from src.skin_optics_hsi.km_bio_v2r1_stage_rd import (
    FIT_HI,
    FIT_LO,
    FIT_MASK,
    WAVELENGTH,
    _columns,
    _edge_median_abs,
    _resolve,
)


ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "configs/skin_optics_hsi/km_bio_v2r1_stage_rd.yaml"


def _config():
    return yaml.safe_load(CONFIG.read_text(encoding="utf-8"))


def test_rd_config_freezes_ps_and_is_locked_beyond_validation():
    config = _config()
    assert config["task_id"] == "R-D_FREEZE_AND_DEVELOPMENTAL_VALIDATION"
    assert config["frozen_candidate"] == "V2R-PS"
    assert config["freeze"]["candidate"] == "V2R-PS"
    assert config["execution"]["run_freeze"] is True
    assert config["execution"]["run_validation"] is True
    assert config["execution"]["require_freeze_before_validation"] is True
    for key in ("run_test", "run_clinical_500", "run_rgb_encoder_training"):
        assert config["execution"][key] is False
    assert config["access_policy"]["validation_content_allowed"] is True
    for key in ("test_content_allowed", "clinical_500_content_allowed", "raw_hsi_content_allowed", "rgb_content_allowed"):
        assert config["access_policy"][key] is False


def test_rd_inputs_and_frozen_fold_are_declared():
    config = _config()
    for value in config["inputs"].values():
        assert _resolve(ROOT, value).exists()
    for value in config["implementation"].values():
        assert _resolve(ROOT, value).exists()
    assert config["folds"]["expected_sha256"] == "e763a9f3dc7f58babe264f7056699c3d5b3722187fbb558f83f2fdb5d6b67370"


def test_rd_uses_the_registered_v2r_bands_and_bounds():
    config = _config()
    assert config["analysis"]["fit_centers_nm"] == list(range(420, 690, 10))
    assert config["analysis"]["edge_diagnostic_centers_nm"] == [400, 410, 690, 700]
    assert FIT_MASK.sum() == 27
    assert list(WAVELENGTH[FIT_MASK]) == [float(value) for value in config["analysis"]["fit_centers_nm"]]
    assert list(FIT_LO) == config["parameters"]["lower"]
    assert list(FIT_HI) == config["parameters"]["upper"]
    assert _columns() == [f"observed_reflectance_{int(value)}nm" for value in WAVELENGTH]


def test_rd_frozen_quantities_match_the_registered_freeze():
    config = _config()
    frozen_quantities = config["freeze"]["frozen_quantities"]
    assert frozen_quantities["fixed_s0"] == 0.70
    assert frozen_quantities["fixed_diameter_um"] == 15.0
    assert frozen_quantities["fixed_g0"] == 1.0
    assert frozen_quantities["fixed_epidermis_thickness_mm"] == 0.060
    assert config["freeze"]["estimator"] == "full_train_joint_centered_log_shape"
    assert config["freeze"]["train_global_estimation"] == ["A_s", "delta_bs"]
    # The reused joint-shape estimator reads the frozen scalars from ``parameters``.
    assert config["parameters"]["fixed_s0"] == frozen_quantities["fixed_s0"]
    assert config["parameters"]["fixed_diameter_um"] == frozen_quantities["fixed_diameter_um"]
    assert config["parameters"]["fixed_g0"] == frozen_quantities["fixed_g0"]


def test_rd_parameter_upgrade_is_forbidden_after_r_c1r():
    config = _config()
    assert config["decision"]["parameter_upgrade_forbidden"] is True
    assert config["decision"]["parameter_upgrade_forbidden_reason"] == "f_mel_f_blood_s_all_unreliable_in_R_C1R"
    assert config["decision"]["retained_development_state"] == "V2R_DEVELOPMENT_SPECTRAL_CANDIDATE"
    assert config["decision"]["spectral_only_state"] == "V2R_SPECTRAL_ONLY"
    assert config["scope"]["validation_subject_count"] == 3
    assert config["scope"]["validation_is_developmental_only"] is True


def test_rd_direction_thresholds_are_preregistered():
    config = _config()
    thresholds = config["direction_consistency"]["thresholds"]
    assert thresholds["median_logrmse_absolute_regression_max"] == 0.03
    assert thresholds["median_sam_deg_absolute_regression_max"] == 3.0
    assert thresholds["validation_better_than_reference_fraction_min"] == 0.5
    assert thresholds["parameter_median_margin_fraction_of_train_range"] == 0.20
    assert config["direction_consistency"]["reference"] == "full_train_individual_refit_with_frozen_globals"
    assert config["direction_consistency"]["action"] == "report_only_no_re_tuning"


def test_edge_median_abs_uses_the_worst_edge_band():
    residuals = pd.DataFrame([
        {"wavelength_nm": 400, "band_role": "edge_diagnostic", "signed_residual": 0.02},
        {"wavelength_nm": 400, "band_role": "edge_diagnostic", "signed_residual": 0.04},
        {"wavelength_nm": 410, "band_role": "edge_diagnostic", "signed_residual": -0.15},
        {"wavelength_nm": 690, "band_role": "edge_diagnostic", "signed_residual": 0.01},
        {"wavelength_nm": 700, "band_role": "edge_diagnostic", "signed_residual": 0.03},
        {"wavelength_nm": 500, "band_role": "fit", "signed_residual": 0.90},
    ])
    worst, medians = _edge_median_abs(residuals)
    assert np.isclose(medians["400"], 0.03)
    assert np.isclose(medians["410"], -0.15)
    assert np.isclose(worst, 0.15)
    assert "500" not in medians


def test_frozen_train_manifest_is_a_pure_train_split():
    config = _config()
    train = pd.read_parquet(_resolve(ROOT, config["inputs"]["r_b_observation_manifest"]))
    assert len(train) == 44
    assert set(train["split"].astype(str)) == {"train"}
    assert train["input_quality_status"].eq("PASS").all()
    validation = pd.read_parquet(_resolve(ROOT, config["inputs"]["rd_validation_observation_manifest"]))
    assert len(validation) == 3
    assert set(validation["split"].astype(str)) == {"valid"}
    assert set(train["subject_id"]).isdisjoint(set(validation["subject_id"]))


def test_emitted_r_d_artifacts_are_self_consistent_when_present():
    """Reproducibility gate for the emitted batch; skipped before the batch runs."""

    config = _config()
    output = _resolve(ROOT, config["output_directory"])
    frozen_path = output / "frozen_global_parameters.json"
    if not frozen_path.is_file():
        return
    from src.skin_optics_hsi.km_bio_observation import sha256_file

    frozen = json.loads(frozen_path.read_text(encoding="utf-8"))
    assert frozen["candidate"] == config["frozen_candidate"]
    assert frozen["validation_used_in_freeze"] is False
    assert frozen["fitted_from_splits"] == ["train"]
    assert frozen["train_subject_count"] == 44
    sidecar = (output / "frozen_global_parameters.sha256").read_text(encoding="ascii").split()[0]
    assert sidecar == sha256_file(frozen_path)

    manifest = pd.read_csv(output / "artifact_hash_manifest.csv", encoding="utf-8-sig")
    recorded = manifest.loc[manifest["name"].eq("frozen_global_parameters.json"), "sha256"]
    assert len(recorded) == 1 and str(recorded.iloc[0]) == sha256_file(frozen_path)

    for name in ("train_frozen_refit_subject_metrics.csv", "validation_subject_metrics.csv"):
        table = pd.read_csv(output / name, encoding="utf-8-sig")
        expected = 44 if name.startswith("train") else 3
        assert len(table) == expected
        assert table["solver_converged"].all()
        assert np.isclose(table["A_s"].iloc[0], frozen["A_s"])
        assert np.isclose(table["delta_bs"].iloc[0], frozen["delta_bs"])
        assert np.isclose(table["g0"].iloc[0], frozen["g0"])
        assert np.isclose(table["diameter_um"].iloc[0], frozen["diameter_um"])
        assert np.isclose(table["s0"].iloc[0], frozen["s0"])
    decision = json.loads((output / "r_d_decision.json").read_text(encoding="utf-8"))
    assert decision["validation_used_in_freeze"] is False
    assert decision["test_executed"] is False
    assert decision["parameter_reliability_claim"] == "none"
