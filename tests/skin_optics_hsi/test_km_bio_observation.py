from pathlib import Path

import h5py
import numpy as np

from src.skin_optics_hsi.km_bio_observation import _load_raw_cube, _region_statistics


def test_raw_cube_orientation_and_region_median_preserve_float64(tmp_path: Path):
    stored = np.zeros((31, 2, 3), dtype=np.float64)
    for band in range(31):
        stored[band] = band + np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]])
    hsi = tmp_path / "sample.mat"
    with h5py.File(hsi, "w") as archive:
        archive.create_dataset("cube", data=stored)
    cube = _load_raw_cube(hsi, "cube", (31, 2, 3), "float64")
    assert cube.shape == (3, 2, 31)
    mask = np.zeros((3, 2), dtype=bool)
    mask[0, 0] = True
    mask[2, 1] = True
    mask_path = tmp_path / "mask.npy"
    np.save(mask_path, mask)
    result = _region_statistics(cube, mask_path)
    np.testing.assert_allclose(result["raw_median"], np.arange(31) + 0.35)
    assert result["pixel_count"] == 2


def test_region_statistics_reports_clip_effect_without_mutating_raw(tmp_path: Path):
    cube = np.full((2, 2, 31), 0.5, dtype=np.float64)
    cube[0, 0, 0] = 1.0 + 1e-8
    cube[0, 1, 0] = 1.0 + 1e-8
    mask_path = tmp_path / "mask.npy"
    np.save(mask_path, np.ones((2, 2), dtype=bool))
    result = _region_statistics(cube, mask_path)
    assert result["above_one_pixel_band_values"] == 2
    assert result["raw_median"][0] > result["clipped_median"][0]
    assert cube[0, 0, 0] > 1.0
