from copy import deepcopy
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from src.skin_optics_hsi.km_bio_v2r1_stage_c0r import (
    ContractError,
    _select_candidate,
    _validate_reflectance,
    validate_fold_assignments,
    verify_historical_r_c0,
)


ROOT = Path(__file__).resolve().parents[2]
CONFIG = yaml.safe_load((ROOT / "configs/skin_optics_hsi/km_bio_v2r1_candidate_ladder_resume.yaml").read_text(encoding="utf-8"))


def test_resume_contract_keeps_scope_and_disables_early_stop():
    assert CONFIG["task_id"] == "R-C0R_RESUME_REGISTERED_CANDIDATE_LADDER"
    assert CONFIG["protocol_version"] == "KM-BIO-v2R.1"
    assert CONFIG["execution"]["stop_after_first_failed_upgrade"] is False
    assert CONFIG["historical_r_c0"]["reuse_candidates"] == ["V2R-0", "V2R-P"]
    assert CONFIG["candidates"]["run"] == ["V2R-PS", "V2R-PSG"]
    assert not any(CONFIG["execution"][name] for name in ("run_r_c1", "run_best_o", "run_validation", "run_test", "run_clinical_500"))


def test_frozen_fold_assignment_exactly_covers_44_subjects():
    subjects = [subject for group in CONFIG["folds"]["assignments"].values() for subject in group]
    frame = validate_fold_assignments(subjects, CONFIG["folds"]["assignments"])
    assert len(frame) == 44
    assert frame["subject_id"].nunique() == 44
    assert sorted(frame.groupby("outer_fold").size().tolist()) == [8, 9, 9, 9, 9]


def test_fold_assignment_rejects_leakage_by_duplicate_subject():
    with pytest.raises(ContractError, match="duplicate"):
        validate_fold_assignments(["p001", "p002"], {1: ["p001", "p002"], 2: ["p002"]})


def test_historical_r_c0_hashes_and_reusable_candidate_artifacts_verify():
    frames, verification = verify_historical_r_c0(ROOT, CONFIG)
    assert verification["matches"].all()
    assert set(frames["metrics"]["candidate"]) == {"V2R-0", "V2R-P"}
    assert len(frames["predictions"]) == 44 * 31 * 2
    assert len(frames["residuals"]) == 44 * 31 * 2


def test_selection_prefers_less_complex_candidate_within_registered_tolerance():
    summary = pd.DataFrame([
        {"candidate": "V2R-0", "median_logrmse": 0.1050},
        {"candidate": "V2R-P", "median_logrmse": 0.1000},
        {"candidate": "V2R-PS", "median_logrmse": 0.0980},
        {"candidate": "V2R-PSG", "median_logrmse": 0.0970},
    ])
    result = _select_candidate(summary, CONFIG)
    assert result["selected_candidate"] == "V2R-P"
    assert result["selection_reason"] == "within_0p005_of_best_select_lower_complexity"


def test_reflectance_audit_rejects_negative_nonfinite_and_above_one_without_clipping():
    _validate_reflectance(np.array([0.0, 0.5, 1.0]), "test", require_unit_interval=True)
    with pytest.raises(ContractError, match="Negative"):
        _validate_reflectance(np.array([-1e-9]), "test", require_unit_interval=True)
    with pytest.raises(ContractError, match="Non-finite"):
        _validate_reflectance(np.array([np.nan]), "test", require_unit_interval=True)
    with pytest.raises(ContractError, match="above one"):
        _validate_reflectance(np.array([1.000001]), "test", require_unit_interval=True)
