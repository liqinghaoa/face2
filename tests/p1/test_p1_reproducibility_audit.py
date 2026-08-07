from __future__ import annotations
import numpy as np
from p0b_deca.p1_reproducibility_audit import ALPHA_THRESHOLD, RUN_NAMES, _map_metrics, _pair_metrics, _regions


def test_audit_has_exactly_three_fixed_independent_run_names():
    assert RUN_NAMES == ("run_01", "run_02", "run_03")


def test_alpha_regions_use_the_fixed_p1_threshold_and_erode():
    alpha = np.zeros((9, 9), np.float32); alpha[1:8, 1:8] = ALPHA_THRESHOLD + .1
    result = _regions(alpha, alpha, np.ones((9, 9), np.uint8), np.ones((9, 9), np.uint8))
    assert result["alpha_intersection"].sum() == 49
    assert result["alpha_intersection_erode_1"].sum() < 49
    assert result["alpha_intersection_erode_5"].sum() == 0


def test_pair_metrics_expose_numerical_contract_without_gate_change():
    metrics = _pair_metrics(np.array([1.0], np.float32), np.array([1.0002], np.float32))
    assert metrics["shape_equal"] and metrics["dtype_equal"]
    assert metrics["max_abs"] > 1e-4 and not metrics["allclose_at_1e-4"]


def test_normal_angular_error_uses_normal_vectors_not_display_rgb():
    a = np.zeros((3, 3, 3), np.float32); b = a.copy(); a[..., 2] = 1; b[..., 0] = 1
    metrics = _map_metrics(a, b, np.ones((3, 3), bool), normal=True)
    assert 89.9 < metrics["mean_angular_error_deg"] < 90.1


def test_map_metrics_are_region_limited():
    a = np.zeros((5, 5, 3), np.float32); b = a.copy(); b[0] = 1
    mask = np.zeros((5, 5), bool); mask[1:, 1:] = np.eye(4, dtype=bool)
    metrics = _map_metrics(a, b, mask)
    assert metrics["valid_pixel_count"] == 4
    assert metrics["MAE"] == 0
