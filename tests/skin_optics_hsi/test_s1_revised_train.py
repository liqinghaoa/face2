from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.skin_optics_hsi.s1_revised_forward import load_revised_registry
from src.skin_optics_hsi.s1_revised_train import _assert_physically_train_only, _center, _metrics, _new_output, _side_gains


def test_center_removes_log_amplitude():
    x = np.linspace(-1, 2, 31)
    assert np.allclose(_center(x + 7.2), _center(x))
    assert abs(float(_center(x).mean())) < 1e-14


def test_metrics_identical_prediction_is_zero():
    observed = np.linspace(0.1, 0.8, 31)
    y = _center(np.log(observed))
    result = _metrics(observed, observed, y, y)
    assert result["raw_log_rmse"] == 0.0
    assert result["shape_log_rmse"] == 0.0


def test_side_gain_excludes_held_subject():
    rows = []
    for subject, delta in [("a", 0.2), ("b", 0.4), ("held", 4.0)]:
        for region, sign in [("left_cheek", 0.5), ("right_cheek", -0.5)]:
            row = {"subject_id": subject, "region": region}
            for wave in range(400, 701, 10):
                row[f"reflectance_median_{wave}nm"] = np.exp(sign * delta)
            rows.append(row)
    gains = _side_gains(pd.DataFrame(rows), "held")
    assert np.isclose(gains["image_left"], 0.15)
    assert np.isclose(gains["image_right"], -0.15)


def test_output_refuses_overwrite(tmp_path: Path):
    output = tmp_path / "run"
    output.mkdir()
    (output / "evidence.txt").write_text("keep")
    with pytest.raises(FileExistsError):
        _new_output(output)


def test_revised_upstream_decision_contract():
    registry = load_revised_registry()
    assert registry.raw["stage"] == "S1-4R"


def test_physical_train_only_parquet_gate(tmp_path: Path):
    good = tmp_path / "train.parquet"
    pd.DataFrame({"split": ["train", "train"], "x": [1, 2]}).to_parquet(good, index=False)
    _assert_physically_train_only(good)
    mixed = tmp_path / "mixed.parquet"
    pd.DataFrame({"split": ["train", "valid"], "x": [1, 2]}).to_parquet(mixed, index=False)
    with pytest.raises(RuntimeError):
        _assert_physically_train_only(mixed)
