from skin_optics.audits.monotonicity import audit_monotonicity


def test_monotonicity_hard_metrics(assets5):
    metrics = audit_monotonicity(assets5)
    assert metrics["melanin_max_reflectance_increase"] <= 1e-10
    assert metrics["hemoglobin_band_spearman"] >= 0.995
    assert metrics["melanin_lstar_spearman"] <= -0.995
