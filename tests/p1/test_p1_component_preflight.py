from __future__ import annotations

from pathlib import Path

from utils.p1_component_preflight import load_sweep_config, validate_framework


def test_real_data_preflight_allows_safe_flip_disabled_fallback():
    root = Path(__file__).resolve().parents[2]
    config = load_sweep_config(root / "config/p1/p1_component_sweep_v1.yaml")
    config["root"] = str(root)
    result = validate_framework(config, root / "experiments/500Data/P1_Component_Sweep_v1_test")
    assert result["status"] == "P1_COMPONENT_FRAMEWORK_READY_FOR_PHASE2"
    assert "normal_flip_status_unverified" not in result["blockers"]
    assert result["normal_flip_policy"]["normal_flip_mode"] == "DISABLED_SAFE_FALLBACK"
    assert result["normal_flip_policy"]["unverified_normal_flip_enabled"] is False
    assert any(contract["experiment_key"] == "p1_spec" and contract["status"] == "SKIPPED_UNAVAILABLE_BY_CURRENT_FRONTEND" for contract in result["component_contracts"])
