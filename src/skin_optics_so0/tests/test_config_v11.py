"""SO-0 v1.1 configuration and gate regression tests."""

from __future__ import annotations

import importlib.util
import shutil
from pathlib import Path

import numpy as np
import pytest
import yaml

from skin_optics.assets import load_assets
from skin_optics.config import DEFAULT_CONFIG_DIR, SO0Config
from skin_optics.numpy_backend.reflectance import compute_skin_reflectance


def _copy_config_dir(tmp_path: Path) -> Path:
    target = tmp_path / "so0"
    shutil.copytree(DEFAULT_CONFIG_DIR, target)
    return target


def test_epidermis_thickness_yaml_changes_output(tmp_path: Path) -> None:
    """A physical parameter edited in YAML must change the computed output."""

    cfg_dir = _copy_config_dir(tmp_path)
    cfg = SO0Config.load(cfg_dir)
    assets = load_assets(5)
    baseline = compute_skin_reflectance(assets, 0.5, 0.5, config=cfg)

    path = cfg_dir / "forward_model_mvp.yaml"
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    data["parameters"]["epidermis_thickness_cm"]["primary"] = 0.010
    path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
    changed = compute_skin_reflectance(assets, 0.5, 0.5, config=SO0Config.load(cfg_dir))
    assert np.max(np.abs(baseline - changed)) > 0.0


def test_10nm_production_grid_is_rejected(tmp_path: Path) -> None:
    """The v1.1 configuration gate must reject 10 nm production."""

    cfg_dir = _copy_config_dir(tmp_path)
    path = cfg_dir / "forward_model_mvp.yaml"
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    data["assets"]["production_step_nm"] = 10
    path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
    with pytest.raises(ValueError, match="production_step_nm"):
        SO0Config.load(cfg_dir)


def test_missing_required_config_field_fails(tmp_path: Path) -> None:
    """Missing required fields are not silently defaulted."""

    cfg_dir = _copy_config_dir(tmp_path)
    path = cfg_dir / "forward_model_mvp.yaml"
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    del data["project"]["output_dir"]
    path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
    with pytest.raises(ValueError, match="project.output_dir"):
        SO0Config.load(cfg_dir)


def test_required_artifacts_gate_rejects_empty_run(tmp_path: Path) -> None:
    """A current run with missing artifacts cannot pass completeness."""

    script = Path(__file__).resolve().parents[1] / "scripts" / "run_so0_all.py"
    spec = importlib.util.spec_from_file_location("so0_run_gate", script)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    ok, missing = module.required_artifacts_present(tmp_path)
    assert not ok
    assert "pytest_report.xml" in missing


def test_pytest_failure_status_forces_decision_fail(tmp_path: Path) -> None:
    """A nonzero pytest result is a hard final gate failure."""

    script = Path(__file__).resolve().parents[1] / "scripts" / "run_so0_all.py"
    spec = importlib.util.spec_from_file_location("so0_run_decide", script)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    cfg = SO0Config.load()
    (tmp_path / "frozen_config.yaml").write_text("x: 1\n", encoding="utf-8")
    metrics = {
        "colorchecker": {"status": "PASS"},
        "resolution": {"status": "PASS"},
        "monotonicity": {"status": "PASS"},
        "backend_parity": {"status": "PASS"},
        "shading_specular_exposure": {"status": "PASS"},
        "autograd": {"status": "PASS"},
        "spectral_separability": {"status": "PASS"},
        "observation_separability": {"status": "PASS"},
        "sensitivity": {},
        "physics_regression": {"status": "PASS"},
        "config_valid": True,
        "formula_registry_valid": True,
    }
    decision = module.decide(
        "unit_test_run",
        tmp_path,
        cfg,
        {"return_code": 1, "tests": 1, "failures": 0, "errors": 0, "skipped": 0, "duration_seconds": 0.0, "status": "FAIL"},
        metrics,
        True,
        True,
    )
    assert decision["status"] == "FAIL"
    assert "pytest" in decision["hard_failures"]
