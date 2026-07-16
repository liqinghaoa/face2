"""Focused rule tests for Optical ROI Dataset V1."""

from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from preprocessing.build_optical_roi_dataset_v1 import (
    bbox_area,
    build_effective_mask,
    derive_exif,
    empty_mask_reason,
    historical_inventory_sha256,
    inclusive_bbox_region,
    sha256_file,
    valid_skin_fraction,
    validate_positive_exif,
    validate_unique_ids,
)


def test_inclusive_bbox_converts_to_right_open_slice() -> None:
    region = inclusive_bbox_region((224, 224), (10, 20, 12, 22))
    assert int(region.sum()) == 9
    assert region[20:23, 10:13].all()
    assert not region[19, 10]
    assert not region[23, 12]


def test_bbox_boundary_223_does_not_overflow() -> None:
    region = inclusive_bbox_region((224, 224), (222, 222, 223, 223))
    assert int(region.sum()) == 4
    assert region[223, 223]


def test_mask_is_bbox_skin_and_final_intersection() -> None:
    parsing = np.zeros((224, 224), dtype=np.uint8)
    final = np.zeros((224, 224), dtype=np.uint8)
    parsing[5:10, 5:10] = 1
    parsing[6, 6] = 17
    final[7:12, 7:12] = 255
    mask = build_effective_mask(parsing, final, (6, 6, 8, 8), skin_label=1)
    expected = np.zeros((224, 224), dtype=np.uint8)
    expected[7:9, 7:9] = 255
    assert np.array_equal(mask, expected)


def test_output_mask_contains_only_zero_and_255() -> None:
    parsing = np.ones((224, 224), dtype=np.uint8)
    final = np.ones((224, 224), dtype=np.uint8)
    mask = build_effective_mask(parsing, final, (0, 0, 1, 1), skin_label=1)
    assert mask.dtype == np.uint8
    assert set(np.unique(mask).tolist()) == {0, 255}


def test_bbox_area_includes_last_row_and_column() -> None:
    assert bbox_area((10, 20, 12, 22)) == 9
    assert bbox_area((223, 223, 223, 223)) == 1


def test_valid_skin_fraction() -> None:
    assert valid_skin_fraction(3, 12) == 0.25
    with pytest.raises(ValueError):
        valid_skin_fraction(13, 12)


def test_exif_derived_formulas() -> None:
    relative, iso_condition = derive_exif(1 / 100, 2.0, 400)
    assert relative == pytest.approx(np.log2((1 / 100) / (2.0**2)))
    assert iso_condition == pytest.approx(2.0)


@pytest.mark.parametrize(
    "values",
    [(0, 2.0, 100), (0.01, 0, 100), (0.01, 2.0, 0), (float("nan"), 2.0, 100)],
)
def test_invalid_or_empty_exif_fails(values: tuple[float, float, float]) -> None:
    with pytest.raises(ValueError):
        validate_positive_exif(*values)


def test_duplicate_ids_fail() -> None:
    with pytest.raises(ValueError):
        validate_unique_ids(["A001", "A001"])
    with pytest.raises(ValueError):
        validate_unique_ids(["A001", "a001"])


def test_empty_mask_is_reported_not_repaired() -> None:
    parsing = np.zeros((224, 224), dtype=np.uint8)
    final = np.ones((224, 224), dtype=np.uint8) * 255
    mask = build_effective_mask(parsing, final, (0, 0, 20, 20), skin_label=1)
    assert empty_mask_reason(mask, "ID1", "forehead") == "empty_mask:ID1:forehead"
    assert int(mask.sum()) == 0


def test_historical_input_file_is_not_modified(tmp_path: Path) -> None:
    source = tmp_path / "source.png"
    Image.fromarray(np.full((224, 224, 3), 127, dtype=np.uint8), mode="RGB").save(source)
    before_hash = sha256_file(source)
    before_inventory = historical_inventory_sha256([source], tmp_path)
    before_stat = (source.stat().st_size, source.stat().st_mtime_ns)
    with Image.open(source) as image:
        array = np.asarray(image).copy()
    assert array.shape == (224, 224, 3)
    assert sha256_file(source) == before_hash
    assert historical_inventory_sha256([source], tmp_path) == before_inventory
    assert (source.stat().st_size, source.stat().st_mtime_ns) == before_stat
