from __future__ import annotations

import numpy as np

from src.skin_optics_real_preprocess import face_geometry, facemesh_regions as regions
from src.skin_optics_real_preprocess.schemas import GeometryConfig


def synthetic_landmarks() -> np.ndarray:
    pts = np.zeros((478, 2), dtype=np.float32)
    center = np.array([500.0, 640.0], dtype=np.float32)
    rx, ry = 210.0, 320.0
    for i, idx in enumerate(regions.FACE_OVAL_INDICES):
        angle = -np.pi / 2 + 2 * np.pi * i / len(regions.FACE_OVAL_INDICES)
        pts[idx] = center + np.array([np.cos(angle) * rx, np.sin(angle) * ry], dtype=np.float32)
    for idx in regions.IMAGE_LEFT_EYE_INDICES:
        pts[idx] = [420.0, 500.0]
    for idx in regions.IMAGE_RIGHT_EYE_INDICES:
        pts[idx] = [600.0, 535.0]
    pts[regions.NOSE_TIP_INDEX] = [510.0, 650.0]
    pts[regions.IMAGE_LEFT_MOUTH_INDEX] = [440.0, 760.0]
    pts[regions.IMAGE_RIGHT_MOUTH_INDEX] = [580.0, 785.0]
    pts[regions.CHIN_INDEX] = [500.0, 960.0]
    pts[pts.sum(axis=1) == 0] = center
    return pts


def test_crop_is_4_to_5_and_matrix_is_similarity() -> None:
    cfg = GeometryConfig()
    geom = face_geometry.build_geometry(synthetic_landmarks(), cfg)
    assert abs((geom.crop_height / geom.crop_width) - 1.25) < 1.0e-6
    s0, s1 = geom.matrix_singular_values
    assert abs(s0 - s1) < 1.0e-5
    linear = geom.affine_source_to_canvas[:2, :2]
    assert np.allclose(linear.T @ linear, np.eye(2) * s0**2, atol=1.0e-5)


def test_roll_reduces_eye_y_delta_and_geometry_is_deterministic() -> None:
    pts = synthetic_landmarks()
    before = abs(float(pts[regions.IMAGE_RIGHT_EYE_INDICES, 1].mean() - pts[regions.IMAGE_LEFT_EYE_INDICES, 1].mean()))
    geom1 = face_geometry.build_geometry(pts, GeometryConfig())
    geom2 = face_geometry.build_geometry(pts, GeometryConfig())
    assert abs(geom1.rolled_eye_y_delta) < before
    assert abs(geom1.residual_roll_degrees) < 1.0e-4
    assert np.allclose(geom1.affine_source_to_canvas, geom2.affine_source_to_canvas)
    assert geom1.roundtrip_max_error < 1.0e-3


def test_geometry_does_not_depend_on_parsing_or_hair_mask() -> None:
    pts = synthetic_landmarks()
    geom1 = face_geometry.build_geometry(pts, GeometryConfig())
    fake_parsing = np.zeros((1280, 1024), dtype=np.uint8)
    fake_parsing[:100, :] = 17
    geom2 = face_geometry.build_geometry(pts, GeometryConfig())
    assert fake_parsing.sum() > 0
    assert np.allclose(geom1.affine_source_to_canvas, geom2.affine_source_to_canvas)


def test_top_margin_revision_enlarges_crop_without_breaking_similarity() -> None:
    pts = synthetic_landmarks()
    base = face_geometry.build_geometry(pts, GeometryConfig(top_margin_ratio=0.02))
    rev = face_geometry.build_geometry(pts, GeometryConfig(top_margin_ratio=0.035))
    assert rev.crop_height > base.crop_height
    assert rev.crop_width > base.crop_width
    assert abs((rev.crop_height / rev.crop_width) - 1.25) < 1.0e-6
    s0, s1 = rev.matrix_singular_values
    assert abs(s0 - s1) < 1.0e-5
