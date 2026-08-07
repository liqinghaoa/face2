import numpy as np

from skin_optics.numpy_backend.color_spaces import bradford_adapt, linear_srgb_to_xyz, srgb_decode, srgb_encode, xyz_to_linear_srgb


def test_srgb_roundtrip_and_negative_branch():
    linear = np.array([-0.1, 0.0, 0.001, 0.01, 0.5])
    srgb = srgb_encode(linear)
    assert np.all(np.isfinite(srgb))
    assert np.max(np.abs(srgb_decode(srgb) - linear)) <= 1e-12


def test_xyz_rgb_matrices_and_bradford():
    rgb = np.array([[1.0, 1.0, 1.0]])
    xyz = linear_srgb_to_xyz(rgb)
    assert np.allclose(xyz[0, 1], 1.0, atol=1e-7)
    assert np.allclose(xyz_to_linear_srgb(xyz), rgb, atol=1e-6)
    src = np.array([0.95047, 1.0, 1.08883])
    dst = np.array([1.0985, 1.0, 0.35585])
    x = np.array([[0.2, 0.3, 0.4]])
    y = bradford_adapt(x, src, dst)
    z = bradford_adapt(y, dst, src)
    assert np.max(np.abs(z - x)) <= 1e-10
    assert np.max(np.abs(bradford_adapt(x, src, src) - x)) <= 1e-10
