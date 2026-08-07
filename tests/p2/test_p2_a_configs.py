from __future__ import annotations

from pathlib import Path

from p2_counterfactual.config import resolve_p2_a_config

CONFIGS = [
    Path("config/p2/p2_a/p2_a1_colorjitter.yaml"),
    Path("config/p2/p2_a/p2_a2_relighting.yaml"),
    Path("config/p2/p2_a/p2_a3_full_consistency.yaml"),
]


def test_remaining_configs_share_fixed_training_protocol() -> None:
    configs = [resolve_p2_a_config(path) for path in CONFIGS]
    common_keys = ["model", "training", "evaluation", "output_root"]
    baseline = {key: configs[0][key] for key in common_keys}
    for config in configs[1:]:
        for key, value in baseline.items():
            assert config[key] == value
        data_without_source = {key: value for key, value in config["data"].items() if key not in {"original_rgb_source", "original_rgb_override_dir"}}
        baseline_data_without_source = {
            key: value for key, value in configs[0]["data"].items() if key not in {"original_rgb_source", "original_rgb_override_dir"}
        }
        assert data_without_source == baseline_data_without_source
    assert [config["input_mode"] for config in configs] == ["original", "relight_mix", "paired"]
    for config in configs:
        assert config["data"]["original_rgb_source"] == "p0_e0b_meanbg_224"
        assert config["data"]["original_rgb_override_dir"].replace("\\", "/").endswith("data/processed/P0_Physics_Audit_v1/images/e0b_meanbg_224")
        assert Path(config["data"]["original_rgb_override_dir"]).is_dir()
    assert "P2_Counterfactual_Relighting500_v1" in configs[0]["manifest_path"]
    assert "P2_DetailPreserving_Relighting500_v2" in configs[1]["manifest_path"]
    assert "P2_DetailPreserving_Relighting500_v2" in configs[2]["manifest_path"]
    for config in configs:
        assert Path(config["manifest_path"]).is_file()
    assert configs[0]["color_jitter"] == {"brightness": 0.3, "contrast": 0.3, "saturation": 0.2, "hue": 0.05}
    assert configs[2]["consistency_loss_weights"] == {"prediction": 0.5, "feature": 0.1}
