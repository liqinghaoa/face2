from __future__ import annotations
import numpy as np
from p0b_deca.p1_cross_process_contract import PRESETS, RUN_NAMES, VERSION, _thresholds


def test_v2_calibration_run_set_excludes_holdout():
    assert RUN_NAMES == ("run_01", "run_02", "run_03")
    assert "run_04" not in RUN_NAMES and VERSION == "p0b_cross_process_v2"


def test_threshold_formula_is_data_derived_and_has_hard_margin():
    rows,index=_thresholds([{"category":"latent","output":"shape_code","metric":"MAE","value":v} for v in (.1,.2,.3)])
    row=rows[0]
    assert row["warning_limit"] == .3 and np.isclose(row["hard_limit"], .45)
    assert row["count"] == 3 and "run_01" in row["source_runs"]


def test_six_presets_are_fixed_and_complete():
    assert PRESETS == ("neutral_front","left","right","top","dim_front","bright_front")


def test_thresholds_do_not_use_case_specific_keys():
    rows,_=_thresholds([{"category":"normal_coarse","output":"normal_coarse","metric":"physics_core_MAE","value":.01}])
    assert "case_id" not in rows[0]
