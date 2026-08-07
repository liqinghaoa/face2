from __future__ import annotations

import csv
from pathlib import Path

import cv2
import numpy as np

from preprocessing.p0_physics_assets.finalization import AMBIGUOUS_IDS, physics_core_skin_from_effective_masks


ROOT = Path("data/processed/P0_Physics_Audit_v1")


def _gray(path: Path) -> np.ndarray:
    value = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
    assert value is not None
    return value


def test_physics_core_skin_is_exact_effective_union_and_excludes_chin_only() -> None:
    left = np.zeros((224, 224), np.uint8); right = left.copy(); forehead = left.copy(); chin = left.copy()
    left[10, 10] = 255; right[20, 20] = 255; forehead[30, 30] = 255; chin[40, 40] = 255
    core = physics_core_skin_from_effective_masks(left, right, forehead)
    assert np.array_equal(core, np.maximum(np.maximum(left, right), forehead))
    assert core[40, 40] == 0  # chin-only pixels cannot enter the physics core.


def test_finalized_p0_output_statuses_and_physics_masks() -> None:
    assert ROOT.is_dir(), "run finalize_p0_physics_audit_assets_v1.py before this integration test"
    with (ROOT / "metadata/master_index.csv").open(encoding="utf-8-sig", newline="") as handle:
        master = list(csv.DictReader(handle))
    with (ROOT / "metadata/build_status.csv").open(encoding="utf-8-sig", newline="") as handle:
        status = list(csv.DictReader(handle))
    assert len(master) == len(status) == 500
    assert [row["ID"] for row in master] == [row["ID"] for row in status]
    by_id = {row["ID"]: row for row in master}
    assert set(by_id) >= AMBIGUOUS_IDS
    for image_id in AMBIGUOUS_IDS:
        row = by_id[image_id]
        assert row["core_asset_status"] == "success"
        assert row["legacy_regression_status"] == "localized_jaw_neck_mismatch"
        assert row["legacy_regression_strict_pass"] == "0"
        assert row["boundary_ambiguity_status"] == "jaw_neck_ambiguous"
        assert row["p0_usable"] == "1"
        assert row["overall_status"] == "success_with_boundary_ambiguity"
        assert row["legacy_mismatch_affects_physics_core"] == "0"
        assert row["chin_usage"] == "exploratory_only"
        assert row["chin_boundary_warning"] == "1"
        assert row["chin_roi_status"] == "usable_with_boundary_warning"
    for image_id, row in by_id.items():
        core = _gray(ROOT / row["physics_core_skin_mask_relpath"])
        left = _gray(ROOT / row["left_cheek_effective_relpath"])
        right = _gray(ROOT / row["right_cheek_effective_relpath"])
        forehead = _gray(ROOT / row["forehead_effective_relpath"])
        chin = _gray(ROOT / row["chin_effective_relpath"])
        assert core.shape == (224, 224) and core.dtype == np.uint8
        assert set(np.unique(core)).issubset({0, 255})
        assert np.array_equal(core, physics_core_skin_from_effective_masks(left, right, forehead))
        chin_only = (chin > 0) & ~(np.maximum(np.maximum(left, right), forehead) > 0)
        assert not np.any((core > 0) & chin_only)
        if image_id not in AMBIGUOUS_IDS:
            assert row["core_asset_status"] == "success"
            assert row["legacy_regression_status"] == "passed"
            assert row["boundary_ambiguity_status"] == "none"
            assert row["p0_usable"] == "1"
            assert row["overall_status"] == "success"
