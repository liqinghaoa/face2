from skin_optics.config import load_default_config, load_formula_registry, load_thresholds


def test_config_and_registry_strict():
    cfg = load_default_config()
    thresholds = load_thresholds()
    registry = load_formula_registry()
    assert cfg["assets"]["production_step_nm"] == 5
    assert cfg["color"]["camera_to_xyz"]["neural_mapping"] is False
    assert thresholds["reflectance"]["hemoglobin_band_spearman_min"] == 0.995
    assert len(registry["formulas"]) >= 10
    assert all(formula["trainable"] is False for formula in registry["formulas"])
