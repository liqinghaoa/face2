import numpy as np

from skin_optics.numpy_backend.absorption import blood_fraction, dermis_absorption, epidermis_absorption, melanin_fraction


def test_control_fraction_mappings(assets5):
    assert melanin_fraction(0.0) == 0.013
    assert melanin_fraction(1.0) == 0.43
    assert blood_fraction(0.0) == 0.02
    assert blood_fraction(1.0) == 0.07
    epi = epidermis_absorption(assets5, np.array([0.0, 1.0]))
    derm = dermis_absorption(assets5, np.array([0.0, 1.0]))
    assert epi.shape == (2, 65)
    assert derm.shape == (2, 65)
    assert np.all(np.isfinite(epi))
    assert np.all(np.isfinite(derm))
