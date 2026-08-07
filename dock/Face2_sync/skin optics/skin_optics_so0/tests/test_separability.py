from skin_optics.audits.separability import audit_spectral_separability


def test_spectral_separability(assets5):
    metrics = audit_spectral_separability(assets5)
    assert metrics["median"] <= 0.95
    assert metrics["p95"] <= 0.98
