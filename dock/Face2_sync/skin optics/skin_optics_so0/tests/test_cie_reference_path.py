import numpy as np

from skin_optics.numpy_backend.color_spaces import deltae00, xyz_to_lab_d65
from skin_optics.numpy_backend.image_formation import d65_white_xyz, integrate_xyz, render_cie_reference
from skin_optics.numpy_backend.reflectance import compute_skin_reflectance


def test_cie_reference_outputs_and_white(assets5):
    r = compute_skin_reflectance(assets5, 0.5, 0.5)
    out = render_cie_reference(assets5, r)
    assert np.all(np.isfinite(out.xyz_source))
    assert np.all(np.isfinite(out.srgb_unclipped))
    white = d65_white_xyz(assets5)
    assert np.isclose(white[1], 1.0)


def test_1nm_5nm_cie_skin_deltae(assets1, assets5):
    m = np.linspace(0, 1, 7)
    h = np.linspace(0, 1, 7)
    mm, hh = np.meshgrid(m, h, indexing="ij")
    c1 = render_cie_reference(assets1, compute_skin_reflectance(assets1, mm, hh), "D65")
    c5 = render_cie_reference(assets5, compute_skin_reflectance(assets5, mm, hh), "D65")
    de = deltae00(xyz_to_lab_d65(c1.xyz_d65, d65_white_xyz(assets1)), xyz_to_lab_d65(c5.xyz_d65, d65_white_xyz(assets5)))
    assert np.median(de) <= 1.0
    assert np.percentile(de, 95) <= 2.5
