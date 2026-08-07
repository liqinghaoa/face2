from __future__ import annotations

import cv2
import numpy as np

from src.skin_optics_real_preprocess import color_pipeline


def test_srgb_linear_roundtrip_error_is_quantization_bounded() -> None:
    rng = np.random.default_rng(42)
    rgb = rng.integers(0, 256, size=(32, 24, 3), dtype=np.uint8)
    linear = color_pipeline.uint8_rgb_to_linear(rgb)
    encoded = color_pipeline.linear_to_uint8_rgb(linear)
    assert np.abs(encoded.astype(np.int16) - rgb.astype(np.int16)).max() <= 1


def test_warp_linear_rgb_is_finite_and_in_range() -> None:
    rgb = np.full((20, 30, 3), 128, dtype=np.uint8)
    linear = color_pipeline.uint8_rgb_to_linear(rgb)
    matrix = np.array([[2.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)
    warped = color_pipeline.warp_linear_rgb(linear, matrix, 60, 40)
    assert warped.shape == (40, 60, 3)
    assert np.isfinite(warped).all()
    assert 0.0 <= float(warped.min()) <= float(warped.max()) <= 1.0
    encoded, chw = color_pipeline.encode_linear_outputs(warped)
    assert encoded.dtype == np.uint8
    assert chw.dtype == np.float16
    assert chw.shape == (3, 40, 60)


def test_png_decode_matches_encoded_uint8_expectation() -> None:
    linear = np.linspace(0, 1, 16 * 16 * 3, dtype=np.float32).reshape(16, 16, 3)
    encoded = color_pipeline.linear_to_uint8_rgb(linear)
    ok, data = cv2.imencode(".png", cv2.cvtColor(encoded, cv2.COLOR_RGB2BGR))
    assert ok
    decoded = cv2.cvtColor(cv2.imdecode(data, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
    assert np.array_equal(decoded, encoded)
