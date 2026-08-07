from skin_optics.audits.backend_parity import audit_backend_parity


def test_backend_parity_reflectance(assets5):
    metrics = audit_backend_parity(assets5)
    assert metrics["numpy_torch_float64_reflectance_max_abs"] <= 1e-9
    assert metrics["torch_float32_float64_reflectance_max_abs"] <= 1e-5
