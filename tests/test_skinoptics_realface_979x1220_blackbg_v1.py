from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = ROOT / "preprocessing" / "build_skinoptics_realface_979x1220_blackbg_v1.py"
spec = importlib.util.spec_from_file_location("skinoptics_979", SCRIPT_PATH)
assert spec and spec.loader
pipeline = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pipeline)


def test_target_and_fixed_patch_coordinates() -> None:
    assert (pipeline.TARGET_WIDTH, pipeline.TARGET_HEIGHT) == (979, 1220)
    assert pipeline.PATCH_X == (0, 241, 482, 723)
    assert pipeline.PATCH_Y == (0, 241, 482, 723, 964)
    assert len(pipeline.PATCH_RECORDS) == 20
    pipeline.validate_patch_protocol()


def test_fixed_patches_cover_canvas_with_15_pixel_overlap() -> None:
    coverage = np.zeros((1220, 979), dtype=np.uint8)
    for patch in pipeline.PATCH_RECORDS:
        x0, y0, x1, y1 = (int(patch[key]) for key in ("x0", "y0", "x1", "y1"))
        assert (x1 - x0, y1 - y0) == (256, 256)
        coverage[y0:y1, x0:x1] += 1
    assert coverage.min() >= 1
    assert 256 - (pipeline.PATCH_X[1] - pipeline.PATCH_X[0]) == 15
    assert 256 - (pipeline.PATCH_Y[1] - pipeline.PATCH_Y[0]) == 15


def test_similarity_transform_and_black_background_contract() -> None:
    from src.skin_optics_real_preprocess import face_geometry
    from src.skin_optics_real_preprocess.schemas import GeometryConfig
    from tests.skin_optics_real_preprocess.test_face_geometry import synthetic_landmarks

    geom = face_geometry.build_geometry(synthetic_landmarks(), GeometryConfig(output_width=979, output_height=1220, top_margin_ratio=.08))
    assert np.isclose(geom.crop_width / geom.crop_height, 979 / 1220)
    singular = np.linalg.svd(geom.affine_source_to_canvas[:2, :2], compute_uv=False)
    assert np.allclose(singular[0], singular[1], atol=1e-5)
    rgb = np.ones((1220, 979, 3), dtype=np.float32)
    mask = np.zeros((1220, 979), dtype=np.uint8); mask[100:300, 100:300] = 255
    blackbg = pipeline.black_background_linear(rgb, mask)
    assert np.all(blackbg[mask == 0] == 0)
    assert np.all(blackbg[mask > 0] == 1)


def test_patch_extraction_is_exact_and_masks_are_binary(tmp_path: Path) -> None:
    full = np.arange(1220 * 979 * 3, dtype=np.float32).reshape(1220, 979, 3)
    mask = np.zeros((1220, 979), dtype=np.uint8); mask[200:900, 100:800] = 255
    rows = pipeline.write_patches("case", full, mask, tmp_path)
    assert len(rows) == 20
    for row in rows:
        x0, y0, x1, y1 = (int(row[key]) for key in ("x0", "y0", "x1", "y1"))
        assert np.array_equal(np.load(tmp_path / "case" / f"{row['patch_id']}_rgb_linear.npy"), full[y0:y1, x0:x1])
        assert row["valid_skin_pixels"] == int((mask[y0:y1, x0:x1] > 0).sum())
        import cv2
        saved_mask = cv2.imread(str(tmp_path / "case" / f"{row['patch_id']}_mask.png"), cv2.IMREAD_GRAYSCALE)
        assert set(np.unique(saved_mask).tolist()).issubset({0, 255})


def test_config_uses_fixed_2class_tracking_index_and_no_legacy_output() -> None:
    config = (ROOT / "config/preprocess/skinoptics_realface_979x1220_blackbg_v1.yaml").read_text(encoding="utf-8")
    assert "P0_Physics_Audit_v1/splits_500/nyha_2class_sex_stratified_group_5fold.csv" in config
    assert "nyha_3class" not in config
    assert "realface_256x320_blackbg_from_raw_v1" not in config


def test_pixel_processing_entrypoint_does_not_accept_labels_or_fold() -> None:
    source = SCRIPT_PATH.read_text(encoding="utf-8")
    start, end = source.index("def process_one("), source.index("\ndef describe(")
    pixel_function = source[start:end].lower()
    assert "nyha" not in pixel_function
    assert "fold" not in pixel_function
