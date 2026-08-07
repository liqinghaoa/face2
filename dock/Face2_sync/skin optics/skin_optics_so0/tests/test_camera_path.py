import numpy as np

from skin_optics.numpy_backend.colorchecker import solve_camera_to_xyz_matrix
from skin_optics.numpy_backend.image_formation import render_camera
from skin_optics.numpy_backend.reflectance import compute_skin_reflectance


def test_camera_path_outputs(assets5):
    matrix, rank, _ = solve_camera_to_xyz_matrix(assets5, "Canon 5DMarkII", "D65")
    out = render_camera(assets5, compute_skin_reflectance(assets5, 0.5, 0.5), matrix, "Canon 5DMarkII", "D65")
    assert rank == 3
    assert out.camera_rgb_raw.shape == (3,)
    assert np.all(np.isfinite(out.linear_srgb_unclipped))
