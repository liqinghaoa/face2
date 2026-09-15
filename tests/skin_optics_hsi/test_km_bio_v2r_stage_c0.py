from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from src.skin_optics_hsi.km_bio_v2r_stage_c0 import (
    _spectral_gate,
    compare_candidates,
    validate_fold_assignments,
)


ROOT = Path(__file__).resolve().parents[2]
CONFIG = yaml.safe_load((ROOT / "configs/skin_optics_hsi/km_bio_v2r_candidate_ladder.yaml").read_text(encoding="utf-8"))


def _summary(logrmse: float, soret_score: float = 0.1) -> dict:
    summary = {
        "median_logrmse": logrmse,
        "p90_logrmse": logrmse,
        "median_rmse": 0.01,
        "median_sam_deg": 1.0,
        "maximum_abs_median_signed_band_residual": 0.01,
        "model_better_than_reference_fraction": 1.0,
        "median_model_to_reference_error_ratio": 0.5,
        "f_mel_any_boundary_fraction": 0.0,
        "f_blood_upper_boundary_fraction": 0.0,
        "median_centered_logrmse": logrmse,
        "absolute_global_median_subject_mean_log_residual": logrmse,
        "median_signed_residual_by_wavelength": {
            "420": soret_score,
            "540": 0.0,
            "550": 0.0,
            "570": 0.0,
            "580": 0.0,
        },
    }
    summary["spectral_gate_checks"] = _spectral_gate(summary, CONFIG["spectral_gates"])
    return summary


def test_frozen_fold_manifest_exactly_covers_44_subjects():
    subjects = [subject for values in CONFIG["folds"]["assignments"].values() for subject in values]
    result = validate_fold_assignments(subjects, CONFIG["folds"]["assignments"])
    assert len(result) == 44
    assert result["subject_id"].nunique() == 44
    assert sorted(result.groupby("outer_fold").size().tolist()) == [8, 9, 9, 9, 9]


def test_fold_validator_rejects_duplicate_or_missing_subject():
    assignments = {1: ["p001", "p002"], 2: ["p002"]}
    try:
        validate_fold_assignments(["p001", "p002"], assignments)
    except ValueError as error:
        assert "more than one" in str(error)
    else:
        raise AssertionError("duplicate frozen subject was accepted")


def test_candidate_upgrade_requires_every_frozen_gate():
    ids = [f"p{i:03d}" for i in range(12)]
    previous = pd.DataFrame({"subject_id": ids, "logrmse": np.full(12, 0.060)})
    current = pd.DataFrame({"subject_id": ids, "logrmse": np.full(12, 0.050)})
    result = compare_candidates("V2R-0", "V2R-P", previous, current, _summary(0.060, 0.10), _summary(0.050, 0.05), CONFIG)
    assert result["cheap_upgrade_gate_pass"]
    assert all(result["checks"].values())

    failed_summary = _summary(0.061, 0.05)
    failed_summary["spectral_gate_checks"] = _spectral_gate(failed_summary, CONFIG["spectral_gates"])
    failed = compare_candidates("V2R-0", "V2R-P", previous, current, _summary(0.070, 0.10), failed_summary, CONFIG)
    assert not failed["checks"]["all_final_spectral_gates"]
    assert not failed["cheap_upgrade_gate_pass"]


def test_spectral_gate_uses_all_seven_registered_limits():
    passed = _summary(0.05)
    assert all(passed["spectral_gate_checks"].values())
    passed["maximum_abs_median_signed_band_residual"] = 0.031
    checks = _spectral_gate(passed, CONFIG["spectral_gates"])
    assert not checks["maximum_abs_median_signed_band_residual"]
    assert sum(not value for value in checks.values()) == 1
