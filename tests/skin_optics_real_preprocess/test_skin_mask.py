from __future__ import annotations

import numpy as np

from src.skin_optics_real_preprocess import facemesh_regions as regions, skin_mask
from src.skin_optics_real_preprocess.schemas import SkinMaskConfig


def landmarks_for_mask() -> np.ndarray:
    pts = np.zeros((478, 2), dtype=np.float32)
    pts[:] = [512.0, 640.0]
    assignments = {
        regions.LEFT_EYE_POLYGON: [(360, 470), (430, 470), (430, 520), (360, 520)],
        regions.RIGHT_EYE_POLYGON: [(610, 470), (690, 470), (690, 520), (610, 520)],
        regions.LEFT_BROW_POLYGON: [(350, 420), (440, 420), (440, 455), (350, 455)],
        regions.RIGHT_BROW_POLYGON: [(600, 420), (700, 420), (700, 455), (600, 455)],
        regions.LIPS_POLYGON: [(440, 800), (590, 800), (590, 870), (440, 870)],
        regions.LEFT_NOSTRIL_POLYGON: [(460, 650), (500, 650), (500, 690), (460, 690)],
        regions.RIGHT_NOSTRIL_POLYGON: [(530, 650), (570, 650), (570, 690), (530, 690)],
    }
    for indices, box in assignments.items():
        coords = np.array(box, dtype=np.float32)
        for i, idx in enumerate(indices):
            pts[idx] = coords[i % len(coords)]
    return pts


def test_strict_skin_mask_binary_subset_and_exclusions() -> None:
    labels = np.ones((1280, 1024), dtype=np.uint8)
    labels[100:200, 100:300] = 17
    source_valid = np.full_like(labels, 255, dtype=np.uint8)
    result = skin_mask.build_skin_valid_mask(labels, landmarks_for_mask(), source_valid, 300.0, SkinMaskConfig())
    mask = result["skin_valid_mask"]
    assert mask.shape == (1280, 1024)
    assert set(np.unique(mask).tolist()).issubset({0, 255})
    skin_mask.assert_mask_contract(mask, source_valid, (1280, 1024))
    assert int(((labels == 17) & (mask > 0)).sum()) == 0
    for poly_mask in result["polygon_masks"].values():
        assert int(((poly_mask > 0) & (mask > 0)).sum()) == 0


def test_source_valid_mask_limits_skin_mask() -> None:
    labels = np.ones((1280, 1024), dtype=np.uint8)
    source_valid = np.full_like(labels, 255, dtype=np.uint8)
    source_valid[:, :512] = 0
    result = skin_mask.build_skin_valid_mask(labels, landmarks_for_mask(), source_valid, 300.0, SkinMaskConfig())
    assert int((result["skin_valid_mask"][:, :512] > 0).sum()) == 0


def test_nostril_dilation_is_decoupled_from_eye_brow_lip_exclusion() -> None:
    labels = np.ones((1280, 1024), dtype=np.uint8)
    source_valid = np.full_like(labels, 255, dtype=np.uint8)
    landmarks = landmarks_for_mask()
    wide = skin_mask.build_skin_valid_mask(
        labels,
        landmarks,
        source_valid,
        300.0,
        SkinMaskConfig(eye_brow_lip_dilation_ratio=0.006, nostril_dilation_ratio=0.006),
    )
    narrow = skin_mask.build_skin_valid_mask(
        labels,
        landmarks,
        source_valid,
        300.0,
        SkinMaskConfig(eye_brow_lip_dilation_ratio=0.006, nostril_dilation_ratio=0.003),
    )
    assert np.array_equal(wide["facemesh_exclusion_without_nostril"], narrow["facemesh_exclusion_without_nostril"])
    assert int((narrow["nostril_exclusion"] > 0).sum()) < int((wide["nostril_exclusion"] > 0).sum())
    assert narrow["nostril_dilation_radius_px"] < wide["nostril_dilation_radius_px"]


def test_scheme_b_face_mask_keeps_features_and_removes_hair_background() -> None:
    labels = np.zeros((320, 256), dtype=np.uint8)
    labels[40:260, 50:210] = 1  # skin
    labels[80:95, 70:100] = 2  # brow
    labels[100:120, 70:105] = 4  # eye
    labels[145:180, 115:145] = 10  # nose
    labels[220:245, 95:165] = 11  # mouth
    labels[0:60, :] = 17  # hair removes the top area
    source_valid = np.full_like(labels, 255, dtype=np.uint8)
    result = skin_mask.build_face_valid_mask(labels, source_valid)
    mask = result["face_valid_mask"]
    assert mask.shape == (320, 256)
    assert set(np.unique(mask).tolist()).issubset({0, 255})
    assert int(((labels == 17) & (mask > 0)).sum()) == 0
    assert int(((labels == 0) & (mask > 0)).sum()) == 0
    assert int(((labels == 4) & (mask > 0)).sum()) > 0
    assert int(((labels == 11) & (mask > 0)).sum()) > 0
