import numpy as np

from skin_optics.numpy_backend.colorchecker import solve_camera_to_xyz_matrix
from skin_optics.numpy_backend.image_formation import render_camera, render_cie_reference
from skin_optics.numpy_backend.reflectance import compute_skin_reflectance


def test_shading_specular_exposure_linearity(assets5):
    r = compute_skin_reflectance(assets5, 0.4, 0.6)
    matrix = solve_camera_to_xyz_matrix(assets5, "Canon 5DMarkII", "D65")[0]
    a = render_cie_reference(assets5, r, shading=0.5, specular=0.0, exposure=1.0).xyz_source
    b = render_cie_reference(assets5, r, shading=1.5, specular=0.0, exposure=1.0).xyz_source
    assert np.max(np.abs(b - 3.0 * a) / np.maximum(np.abs(3.0 * a), 1e-12)) <= 1e-6
    s0 = render_camera(assets5, r, matrix, specular=0.0).camera_rgb_raw
    s1 = render_camera(assets5, r, matrix, specular=0.03).camera_rgb_raw
    s2 = render_camera(assets5, r, matrix, specular=0.06).camera_rgb_raw
    assert np.all(s1 > s0)
    assert np.max(np.abs((s2 - s0) - 2.0 * (s1 - s0))) <= 1e-6
    e2 = render_camera(assets5, r, matrix, exposure=2.0).camera_rgb_raw
    e1 = render_camera(assets5, r, matrix, exposure=1.0).camera_rgb_raw
    assert np.max(np.abs(e2 - 2.0 * e1)) <= 1e-6
