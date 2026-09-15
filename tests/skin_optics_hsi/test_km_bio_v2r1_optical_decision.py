from pathlib import Path

import json

import pandas as pd
import yaml

from src.skin_optics_hsi.km_bio_v2r1_optical_decision import (
    MILESTONE_STATES,
    PER_SAMPLE_OUTPUT_SCHEMA,
    TERMINAL_STATES,
    STATE_MEANING_ZH,
    ContractError,
    _compare,
    _ladder_audit,
    _observation_blocker_analysis,
    _parameter_reliability_audit,
    _preflight,
    _resolve,
    _strong_spectral_audit,
)


ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "configs/skin_optics_hsi/km_bio_v2r1_stage1_optical_decision.yaml"
OUTPUT = ROOT / "outputs/skin_optics_hsi_v2r1/stage1_hsi_physics/km_bio_v2r1_optical_decision"


def _config():
    return yaml.safe_load(CONFIG.read_text(encoding="utf-8"))


def test_decision_config_is_decision_only_and_reads_no_new_content():
    config = _config()
    assert config["task_id"] == "STAGE1_OPTICAL_LAYER_DECISION"
    assert config["selected_candidate"] == "V2R-PS"
    for key in ("run_test", "run_clinical_500", "run_rgb_encoder_training", "run_stage2_design"):
        assert config["execution"][key] is False
    assert config["execution"]["run_decision_gate"] is True
    assert config["execution"]["require_no_new_data_content_reads"] is True
    for key in (
        "raw_hsi_content_allowed",
        "rgb_content_allowed",
        "validation_content_allowed",
        "test_content_allowed",
        "clinical_500_content_allowed",
    ):
        assert config["access_policy"][key] is False
    assert config["access_policy"]["summary_artifacts_only"] is True
    assert config["access_policy"]["refuse_existing_output"] is True
    assert config["locked_stages"]["parameter_upgrade_allowed"] is False
    for key in ("test_executed", "clinical_500_executed", "rgb_encoder_training_executed", "stage2_design_executed"):
        assert config["locked_stages"][key] is False


def test_decision_inputs_exist_and_preflight_passes():
    config = _config()
    paths, context = _preflight(ROOT, config)
    assert paths
    for path in paths.values():
        assert path.exists(), path
    assert context["checks"]
    assert all(context["checks"].values())
    assert context["upstream"]["r_d_decision"]["next_registered_stage"] == "STAGE1_OPTICAL_LAYER_DECISION_REQUIRED"


def test_strong_targets_match_section_7_1():
    config = _config()
    targets = {str(item["key"]): item for item in config["strong_spectral_targets"]}
    assert set(targets) == {
        "median_logrmse",
        "p90_logrmse",
        "median_rmse",
        "median_sam_deg",
        "maximum_abs_median_signed_band_residual",
        "reference_better_fraction",
        "model_to_reference_median_error_ratio",
    }
    assert float(targets["median_logrmse"]["target"]) == 0.06
    assert float(targets["p90_logrmse"]["target"]) == 0.10
    assert float(targets["median_rmse"]["target"]) == 0.03
    assert float(targets["median_sam_deg"]["target"]) == 3.0
    assert float(targets["maximum_abs_median_signed_band_residual"]["target"]) == 0.03
    assert abs(float(targets["reference_better_fraction"]["target"]) - 2.0 / 3.0) < 1e-12
    assert str(targets["reference_better_fraction"]["operator"]) == "ge"
    assert float(targets["model_to_reference_median_error_ratio"]["target"]) == 0.90


def test_strong_target_audit_agrees_with_the_registered_ladder_engine():
    config = _config()
    paths, context = _preflight(ROOT, config)
    frame = _strong_spectral_audit(config, context["upstream"]["r_c0r_decision"])
    assert len(frame) == 7
    assert bool(frame["audit_agrees"].all())
    assert frame.loc[frame["target_key"] == "median_rmse", "recomputed_pass"].iloc[0]
    assert not frame.loc[frame["target_key"] == "median_logrmse", "recomputed_pass"].iloc[0]
    assert int(frame["recomputed_pass"].sum()) == 3


def test_reliability_audit_carries_r_c1r_verdicts_without_upgrading():
    config = _config()
    _, context = _preflight(ROOT, config)
    frame = _parameter_reliability_audit(config, context["upstream"])
    r_c1r = frame[(frame["source_stage"] == "R-C1R") & (frame["candidate"] == "V2R-PS")]
    assert set(r_c1r["parameter"]) == {"f_mel", "f_blood"}
    assert list(r_c1r["classification"]) == ["unreliable", "unreliable"]
    assert not bool(frame[frame["source_stage"] == "R-C1R"]["usable_as_stage2_theta"].any())
    v1 = frame[frame["source_stage"] == "v1_C"]
    assert set(v1["parameter"]) == {"f_mel", "f_blood", "s"}
    assert bool((v1["identifiable_fraction"] == 0.0).all())


def test_observation_blocker_analysis_measures_both_blockers():
    config = _config()
    paths, context = _preflight(ROOT, config)
    frame, flags = _observation_blocker_analysis(config, paths, context["upstream"])
    assert set(frame["blocker_key"]) == {"unified_scale", "fixed_side_difference", "per_spectrum_high_capacity_correction"}
    assert flags["unified_scale_blocker"] is False
    assert flags["fixed_side_difference_coupling_present"] is True
    assert flags["fixed_side_difference_blocker"] is False
    assert flags["observation_level_blocker_present"] is False
    assert flags["per_spectrum_high_capacity_correction_required"] is False
    assert flags["primary_unit_excludes_side_difference"] is True


def test_ladder_covers_every_state_and_picks_the_first_satisfied_terminal_state():
    config = _config()
    paths, context = _preflight(ROOT, config)
    strong = _strong_spectral_audit(config, context["upstream"]["r_c0r_decision"])
    reliability = _parameter_reliability_audit(config, context["upstream"])
    _, blocker_flags = _observation_blocker_analysis(config, paths, context["upstream"])
    ladder, outcome, flags = _ladder_audit(config, strong, reliability, blocker_flags, context["upstream"])
    assert set(ladder["state"]) == set(TERMINAL_STATES) | set(MILESTONE_STATES)
    assert set(STATE_MEANING_ZH) == set(TERMINAL_STATES) | set(MILESTONE_STATES)
    terminal = ladder[ladder["kind"] == "terminal"].sort_values("priority")
    assert list(terminal["state"]) == list(TERMINAL_STATES)
    satisfied = dict(zip(ladder["state"], ladder["satisfied"]))
    assert satisfied["V2R_STOP_KM"] is False
    assert satisfied["V2R_STRONG_THETA_READY"] is False
    assert satisfied["V2R_READY_WITH_S"] is False
    assert satisfied["V2R_PARTIAL_THETA_CANDIDATE"] is False
    assert satisfied["V2R_REVISE_OBSERVATION"] is False
    assert satisfied["V2R_SPECTRAL_ONLY"] is True
    assert satisfied["V2R_DEVELOPMENT_SPECTRAL_CANDIDATE"] is True
    assert outcome == "V2R_SPECTRAL_ONLY"
    assert flags["development_candidate_oof_met"] is True
    assert flags["validation_confirms_direction"] is True
    assert flags["spectral_reconstruction_useful"] is True
    assert abs(float(flags["oof_model_to_reference_ratio"]) - 0.4431593301842419) < 1e-12
    assert abs(float(flags["oof_better_than_reference_fraction"]) - 0.9318181818181818) < 1e-12


def test_compare_operator_rejects_unknown_operators():
    assert _compare(0.05, "le", 0.06)
    assert _compare(0.7, "ge", 2.0 / 3.0)
    assert not _compare(0.07, "le", 0.06)
    try:
        _compare(1.0, "eq", 1.0)
    except ContractError:
        pass
    else:
        raise AssertionError("unsupported operator must raise ContractError")


def test_emitted_decision_artifacts_are_self_consistent_when_present():
    config = _config()
    if not (OUTPUT / "stage1_km_bio_decision.json").exists():
        return
    decision = json.loads((OUTPUT / "stage1_km_bio_decision.json").read_text(encoding="utf-8"))
    audit = json.loads((OUTPUT / "audit_summary.json").read_text(encoding="utf-8"))
    interface = json.loads((OUTPUT / "stage2_interface_freeze.json").read_text(encoding="utf-8"))
    assert decision["status"] == "STAGE1_OPTICAL_LAYER_DECISION_COMPLETE"
    assert decision["decision_state"] == "V2R_SPECTRAL_ONLY"
    assert decision["stage1_optical_layer_outcome"] == "SPECTRAL_RECONSTRUCTION_ONLY_NO_THETA_TARGET"
    assert decision["parameter_reliability_claim"] == "none"
    assert decision["parameter_upgrade_forbidden"] is True
    assert decision["strong_spectral_targets"]["passed_count"] == 3
    assert decision["strong_spectral_targets"]["all_targets_pass"] is False
    assert decision["strong_spectral_targets"]["engine_agreement"] is True
    assert decision["development_candidate_oof_met"] is True
    assert decision["validation_confirms_direction"] is True
    assert decision["spectral_reconstruction_useful"] is True
    assert (
        abs(decision["development_candidate_oof_evidence"]["model_to_reference_median_error_ratio"] - 0.4431593301842419)
        < 1e-12
    )
    assert decision["stage2_theta_usable_as_supervised_target"] is False
    assert interface["stage2_usage"]["theta_usable_as_supervised_target"] is False
    assert interface["inversion_config"]["reliable_parameter_list"] == []
    assert interface["all_file_hashes"]["optical_asset_10nm"]
    assert audit["preflight_all_passed"] is True
    assert audit["data_access"]["new_data_content_reads"] == 0
    assert audit["authorized_stages"] == []
    assert decision["next_registered_stage"] == "NONE_REGISTERED"
    for item in config["required_outputs"]:
        assert (OUTPUT / item).exists(), item
    manifest = pd.read_csv(OUTPUT / "artifact_hash_manifest.csv")
    assert {"role", "name", "path", "sha256"}.issubset(set(manifest.columns))
    assert set(manifest["role"]) == {"input", "config", "implementation", "output"}
    ladder = pd.read_csv(OUTPUT / "decision_ladder_audit.csv")
    assert len(ladder) == len(TERMINAL_STATES) + len(MILESTONE_STATES)
    risks = pd.read_csv(OUTPUT / "residual_risk_register.csv")
    assert len(risks) == 8
    assert not bool(risks["would_justify_revise_observation"].any())
    access = pd.read_csv(OUTPUT / "evidence_access_log.csv")
    forbidden = access[access["content_kind"] == "forbidden_content"]
    assert len(forbidden) == 5
    assert int(forbidden["content_reads"].sum()) == 0


def test_per_sample_schema_matches_section_3_6():
    assert PER_SAMPLE_OUTPUT_SCHEMA == [
        "subject_id",
        "capture_id",
        "roi",
        "model_version",
        "theta_hat[3]",
        "reflectance_hat[31]",
        "logrmse",
        "rmse",
        "sam_deg",
        "residual[31]",
        "parameter_tolerance_intervals",
        "boundary_flags",
        "parameter_reliability_flags",
        "sensitivity_flags",
        "solver_status",
        "input_quality_status",
        "source_hashes",
    ]


def test_resolve_prefers_absolute_paths():
    absolute = str(ROOT / "configs/skin_optics_hsi/km_bio_v2r1_stage1_optical_decision.yaml")
    assert _resolve(ROOT, absolute) == Path(absolute).resolve()
    assert _resolve(ROOT, "rel/file") == (ROOT / "rel/file").resolve()
