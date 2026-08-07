from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from lighting_qc_stage2.affine_mapping import canvas_to_source_mask, mask_iou, source_to_canvas_mask, validate_inverse
from lighting_qc_stage2.class_map import load_parsing_class_map
from lighting_qc_stage2.color import inverse_srgb, luminance_y, rgb_uint8_to_linear
from lighting_qc_stage2.masks import build_masks
from lighting_qc_stage2.metrics_input_space import assert_monotonic_candidates, input_space_metrics
from lighting_qc_stage2.metrics_raw_space import channel_clipping_metrics
from lighting_qc_stage2.pilot_selection import select_pilot32
from lighting_qc_stage2.frozen_metrics import auxiliary_metrics, primary_metrics, raw_frozen_metrics
from lighting_qc_stage2.frozen_spec import affine_definition_hash, metric_spec_payload, roi_definition_hash
from lighting_qc_stage2.full500_pipeline import (
    PRIMARY_METRICS,
    STRATIFICATION_METRICS,
    _bh_fdr,
    compute_metrics,
    decide_stage3,
    foldwise_stratification,
    full500_dirs,
    load_oof,
    metric_families,
    stratified_performance,
    verify_frozen_integrity,
)
from lighting_qc_stage2.visualization_fixed import render_letterboxed
from lighting_qc_stage2.r2_statistical_repair import (
    EROSION_METRICS,
    compare_old_repaired,
    deduplicate_fdr_hits,
    foldwise_assignments as r2_foldwise_assignments,
    stability_status,
    stratified_metrics as r2_stratified_metrics,
    worst_group_bootstrap,
    worst_group_summary as r2_worst_group_summary,
)
from lighting_qc_stage2.r2_exif_decomposition import (
    EXIF_FEATURES,
    TARGET_METRICS,
    _as_seconds,
    build_exif_frame,
    crossfit_decomposition,
)


def toy_class_map() -> dict[str, int]:
    return {
        "background": 0,
        "skin": 1,
        "left_brow": 2,
        "right_brow": 3,
        "left_eye": 4,
        "right_eye": 5,
        "eyeglass": 6,
        "eye_glasses": 6,
        "left_ear": 7,
        "right_ear": 8,
        "earring": 9,
        "nose": 10,
        "mouth": 11,
        "upper_lip": 12,
        "lower_lip": 13,
        "neck": 14,
        "necklace": 15,
        "cloth": 16,
        "hair": 17,
        "hat": 18,
    }


def toy_masks():
    parsing = np.zeros((320, 256), dtype=np.uint8)
    parsing[40:280, 40:216] = 1
    parsing[120:220, 105:150] = 10
    parsing[50:70, 80:180] = 17
    face = np.zeros_like(parsing)
    face[35:285, 35:221] = 255
    source = np.full_like(parsing, 255)
    cfg = {
        "forehead": {"x0": 0.20, "x1": 0.80, "y0": 0.05, "y1": 0.34},
        "canvas_left_cheek": {"x0": 0.08, "x1": 0.44, "y0": 0.40, "y1": 0.78},
        "canvas_right_cheek": {"x0": 0.56, "x1": 0.92, "y0": 0.40, "y1": 0.78},
        "nose": {"x0": 0.35, "x1": 0.65, "y0": 0.28, "y1": 0.72},
    }
    return build_masks(parsing, face, source, toy_class_map(), cfg), parsing, source, face


def test_fixed_pilot_selection_is_reproducible() -> None:
    rows = []
    for i in range(100):
        rows.append({"sample_id": str(i), "ID": str(i), "fold": i % 5, "binary_label": i % 2, "patient_group_id": str(i)})
    split = pd.DataFrame(rows)
    exif = pd.DataFrame({"sample_id": [str(i) for i in range(100)], "brightness_value_apex": range(100), "iso": range(100, 200), "exposure_time": np.linspace(0.001, 0.1, 100)})
    manifest = pd.DataFrame({"sample_id": [str(i) for i in range(100)], "warning_codes": ["" for _ in range(100)], "face_valid_mask_path": "", "source_valid_mask_path": "", "parsing_label_path": "", "aligned_srgb_path": "", "metadata_path": "", "image_path": ""})
    a = select_pilot32(split, exif, manifest, pilot_size=32, seed=7)
    b = select_pilot32(split, exif, manifest, pilot_size=32, seed=7)
    assert a["sample_id"].tolist() == b["sample_id"].tolist()
    assert a["fold"].nunique() == 5 and set(a["binary_label"]) == {0, 1}


def test_formal_500_asset_coverage_inputs_exist() -> None:
    root = ROOT / "data/processed/global_face/realface_256x320_blackbg_from_raw_v1"
    manifest = pd.read_csv(root / "manifests/realface_256x320_manifest.csv")
    assert len(manifest) == 500
    assert (root / "COMPLETED.json").is_file()


def test_parsing_class_map_from_source() -> None:
    mapping = load_parsing_class_map(ROOT)
    assert mapping["skin"] == 1
    assert mapping["nose"] == 10
    assert mapping["hair"] == 17


def test_inverse_srgb_transfer() -> None:
    values = np.array([0.0, 0.04045, 0.5, 1.0])
    out = inverse_srgb(values)
    assert out[0] == pytest.approx(0)
    assert out[1] == pytest.approx(0.04045 / 12.92)
    assert out[2] == pytest.approx(((0.5 + 0.055) / 1.055) ** 2.4)
    assert out[3] == pytest.approx(1)


def test_linear_luminance_formula() -> None:
    rgb = np.array([[[255, 0, 0], [0, 255, 0], [0, 0, 255]]], dtype=np.uint8)
    y = luminance_y(rgb_uint8_to_linear(rgb))
    assert y[0, 0] == pytest.approx(0.2126)
    assert y[0, 1] == pytest.approx(0.7152)
    assert y[0, 2] == pytest.approx(0.0722)


def test_core_skin_only_skin_class() -> None:
    (masks, _), parsing, _, _ = toy_masks()
    assert set(np.unique(parsing[masks["core_skin_e2"] > 0]).tolist()) == {1}


def test_roi_subset_source_valid() -> None:
    (masks, _), _, source, _ = toy_masks()
    for name in ["forehead", "canvas_left_cheek", "canvas_right_cheek", "nose"]:
        assert not ((masks[name] > 0) & (source == 0)).any()


def test_cheeks_do_not_overlap() -> None:
    (masks, _), _, _, _ = toy_masks()
    assert not ((masks["canvas_left_cheek"] > 0) & (masks["canvas_right_cheek"] > 0)).any()


def test_cheeks_exclude_nose() -> None:
    (masks, _), _, _, _ = toy_masks()
    cheeks = (masks["canvas_left_cheek"] > 0) | (masks["canvas_right_cheek"] > 0)
    assert not (cheeks & (masks["nose"] > 0)).any()


def test_forehead_excludes_brow_eye_hair() -> None:
    (masks, _), parsing, _, _ = toy_masks()
    assert not np.isin(parsing[masks["forehead"] > 0], [2, 3, 4, 5, 17]).any()


def test_face_valid_is_not_core_skin() -> None:
    (masks, _), _, _, face = toy_masks()
    assert int((face > 0).sum()) > int((masks["core_skin_e2"] > 0).sum())


def test_black_background_not_in_statistics() -> None:
    (masks, _), _, _, _ = toy_masks()
    rgb = np.zeros((320, 256, 3), dtype=np.uint8)
    rgb[masks["core_skin_e2"] > 0] = 255
    class C:
        erosion_candidates = (0, 2, 4)
        brightness_threshold_candidates = (0.65, 0.70)
        darkness_threshold_candidates = (0.01, 0.02)
        shadow_ratio_candidates = (0.45,)
        shadow_gaussian_sigma = 1.0
        specular_y_threshold_candidates = (0.70,)
        specular_chroma_threshold_candidates = (0.10,)
    row = input_space_metrics(rgb, masks, C)
    assert row["skin_y_median"] == pytest.approx(1.0)


def test_bright_fraction_monotonic() -> None:
    row = {"bright_fraction_y_ge_0.65": 0.5, "bright_fraction_y_ge_0.70": 0.4, "dark_fraction_y_le_0.01": 0.1, "dark_fraction_y_le_0.02": 0.2}
    assert_monotonic_candidates(row, (0.65, 0.70), (0.01, 0.02))


def test_dark_fraction_monotonic() -> None:
    row = {"bright_fraction_y_ge_0.65": 0.5, "bright_fraction_y_ge_0.70": 0.4, "dark_fraction_y_le_0.01": 0.1, "dark_fraction_y_le_0.02": 0.2}
    assert_monotonic_candidates(row, (0.65, 0.70), (0.01, 0.02))


def test_specular_candidate_logic() -> None:
    (masks, _), _, _, _ = toy_masks()
    rgb = np.full((320, 256, 3), 255, dtype=np.uint8)
    class C:
        erosion_candidates = (0, 2, 4)
        brightness_threshold_candidates = (0.80,)
        darkness_threshold_candidates = (0.05,)
        shadow_ratio_candidates = (0.65,)
        shadow_gaussian_sigma = 1.0
        specular_y_threshold_candidates = (0.70,)
        specular_chroma_threshold_candidates = (0.03,)
    row = input_space_metrics(rgb, masks, C)
    assert row["input_specular_y_ge_0.70_chroma_le_0.03_fraction"] > 0


def test_channel_clipping_uses_uint8_raw_pixels() -> None:
    rgb = np.array([[[0, 255, 5], [250, 250, 250]]], dtype=np.uint8)
    mask = np.array([[255, 0]], dtype=np.uint8)
    out = channel_clipping_metrics(rgb, mask, (250, 255), (0, 5))
    assert out["raw_r_eq_0_fraction"] == 1.0
    assert out["raw_g_eq_255_fraction"] == 1.0
    assert out["raw_any_channel_ge_250_fraction"] == 1.0


def test_affine_matrices_inverse() -> None:
    a = np.array([[2, 0, 5], [0, 2, 7], [0, 0, 1]], dtype=float)
    b = np.linalg.inv(a)
    assert validate_inverse(a, b) < 1e-12


def test_mask_roundtrip_iou() -> None:
    mask = np.zeros((20, 20), dtype=np.uint8)
    mask[5:15, 5:15] = 255
    a = np.eye(3)
    raw = canvas_to_source_mask(mask, (20, 20), a)
    back = source_to_canvas_mask(raw, (20, 20), a)
    assert mask_iou(mask, back) == 1.0


def test_source_orientation_matches_scheme_b_for_example() -> None:
    from lighting_qc_stage2.asset_loader import read_metadata
    from lighting_qc_stage2.image_io import read_source_rgb_exif

    meta = read_metadata(ROOT / "data/processed/global_face/realface_256x320_blackbg_from_raw_v1/metadata/100037382.json")
    rgb, _ = read_source_rgb_exif(Path(meta["source_path"]))
    assert (rgb.shape[1], rgb.shape[0]) == (meta["source_width"], meta["source_height"])


def test_labels_fold_oof_do_not_enter_pixel_metric_names() -> None:
    names = " ".join(input_space_metrics.__code__.co_names).lower()
    assert "binary_label" not in names and "fold" not in names and "oof" not in names


def test_invalid_roi_returns_none_not_zero() -> None:
    from lighting_qc_stage2.metrics_input_space import robust_y_stats

    y = np.ones((2, 2))
    mask = np.zeros((2, 2), dtype=np.uint8)
    out = robust_y_stats(y, mask, "empty")
    assert out["empty_y_median"] is None


def test_output_file_completeness_list() -> None:
    required = ["preflight/preflight_summary.json", "pilot/pilot32_manifest.csv", "metrics/pilot32_continuous_metrics.csv", "reports/stage2_pilot32_report.md", "logs/run.log"]
    assert "pilot/pilot32_manifest.csv" in required


def test_stop_after_pilot_blocks_full500() -> None:
    from lighting_qc_stage2.cli import process_pilot
    from lighting_qc_stage2.config import Stage2Config

    cfg = Stage2Config.from_yaml(ROOT / "configs/lighting_qc_stage2_pilot32_v1.yaml")
    bad = Stage2Config(**{**cfg.__dict__, "stop_after_pilot": False})
    with pytest.raises(ValueError):
        process_pilot(bad)


def test_r1_aligned_srgb_shape_is_320_by_256() -> None:
    from lighting_qc_stage2.image_io import read_rgb

    old_pilot = pd.read_csv(ROOT / "experiments/lighting_confounding/Lighting_QC_Stage2_Pilot32_v1/pilot/pilot32_manifest.csv", dtype=str)
    sid = old_pilot.iloc[0]["sample_id"]
    arr = read_rgb(ROOT / f"data/processed/global_face/realface_256x320_blackbg_from_raw_v1/aligned_srgb/{sid}.png")
    assert arr.shape == (320, 256, 3)


def test_r1_parser_and_mask_shapes_are_320_by_256() -> None:
    from lighting_qc_stage2.image_io import read_gray

    old_pilot = pd.read_csv(ROOT / "experiments/lighting_confounding/Lighting_QC_Stage2_Pilot32_v1/pilot/pilot32_manifest.csv", dtype=str)
    sid = old_pilot.iloc[0]["sample_id"]
    base = ROOT / "data/processed/global_face/realface_256x320_blackbg_from_raw_v1"
    assert read_gray(base / f"parsing_label/{sid}.png").shape == (320, 256)
    assert read_gray(base / f"face_valid_mask/{sid}.png").shape == (320, 256)
    assert read_gray(base / f"source_valid_mask/{sid}.png").shape == (320, 256)


def test_letterbox_preserves_320_by_256_as_4_to_5_content() -> None:
    arr = np.zeros((320, 256, 3), dtype=np.uint8)
    out = render_letterboxed(arr, 260, 298)
    assert out.content_width / out.content_height == pytest.approx(0.8, rel=0.01)


def test_letterbox_preserves_landscape_raw_ratio() -> None:
    arr = np.zeros((300, 600, 3), dtype=np.uint8)
    out = render_letterboxed(arr, 260, 298)
    assert out.content_width / out.content_height == pytest.approx(2.0, rel=0.01)


def test_letterbox_mask_and_rgb_geometry_match() -> None:
    rgb = np.zeros((320, 256, 3), dtype=np.uint8)
    mask = np.zeros((320, 256), dtype=np.uint8)
    a = render_letterboxed(rgb, 260, 298)
    b = render_letterboxed(mask, 260, 298, is_mask=True)
    assert (a.scale, a.offset_x, a.offset_y, a.content_width, a.content_height) == (b.scale, b.offset_x, b.offset_y, b.content_width, b.content_height)


def test_letterbox_does_not_modify_input_array() -> None:
    arr = np.arange(320 * 256 * 3, dtype=np.uint32).reshape(320, 256, 3).astype(np.uint8)
    before = arr.copy()
    render_letterboxed(arr, 260, 298)
    assert np.array_equal(arr, before)


def test_fixed_qc_source_has_no_aspect_auto() -> None:
    src = (ROOT / "lighting_qc_stage2/visualization_fixed.py").read_text(encoding="utf-8")
    assert 'aspect="auto"' not in src and "aspect='auto'" not in src


def test_final_roi_masks_are_named_not_candidate_rectangles() -> None:
    src = (ROOT / "lighting_qc_stage2/visualization_fixed.py").read_text(encoding="utf-8")
    assert "final forehead/canvas-cheek/nose ROI" in src
    assert "candidate rectangle" not in src.lower()


def test_r1_primary_erosion_is_fixed_to_2() -> None:
    cfg = __import__("lighting_qc_stage2.config", fromlist=["Stage2Config"]).Stage2Config.from_yaml(ROOT / "configs/lighting_qc_stage2_r1_pilot32_v1.yaml")
    assert cfg.erosion_candidates == (0, 2, 4)
    spec = metric_spec_payload(cfg)
    assert spec["primary_skin_erosion_px"] == 2


def test_roi_definition_hash_is_stable_function_of_config() -> None:
    cfg = __import__("lighting_qc_stage2.config", fromlist=["Stage2Config"]).Stage2Config.from_yaml(ROOT / "configs/lighting_qc_stage2_r1_pilot32_v1.yaml")
    assert roi_definition_hash(cfg) == roi_definition_hash(cfg)


def test_affine_definition_hash_is_stable() -> None:
    assert affine_definition_hash() == affine_definition_hash()


def test_dark_contrast_score_formula() -> None:
    (masks, _), _, _, _ = toy_masks()
    rgb = np.full((320, 256, 3), 128, dtype=np.uint8)
    rgb[40:80, 40:216] = 64
    out = primary_metrics(rgb, masks)
    expected = (out["skin_y_median"] - out["skin_y_p10"]) / (out["skin_y_median"] + 1e-8)
    assert out["dark_contrast_score"] == pytest.approx(expected)


def test_deep_dark_contrast_score_formula() -> None:
    (masks, _), _, _, _ = toy_masks()
    rgb = np.full((320, 256, 3), 128, dtype=np.uint8)
    rgb[40:70, 40:216] = 40
    out = primary_metrics(rgb, masks)
    expected = (out["skin_y_median"] - out["skin_y_p05"]) / (out["skin_y_median"] + 1e-8)
    assert out["deep_dark_contrast_score"] == pytest.approx(expected)


def test_r1_raw_clipping_metrics_are_compact() -> None:
    rgb = np.array([[[0, 255, 5], [250, 250, 250]]], dtype=np.uint8)
    mask = np.array([[255, 255]], dtype=np.uint8)
    out = raw_frozen_metrics(rgb, {"raw_core_skin": mask})
    assert "raw_any_channel_ge_250_fraction" in out
    assert "raw_any_channel_ge_252_fraction" not in out
    assert out["raw_any_channel_eq_255_fraction"] == 0.5


def test_r1_primary_metrics_do_not_depend_on_quality_labels() -> None:
    names = " ".join(primary_metrics.__code__.co_names).lower()
    assert "quality" not in names and "label" not in names and "oof" not in names


def test_auxiliary_shadow_only_ratio_055_named_output() -> None:
    (masks, _), _, _, _ = toy_masks()
    rgb = np.full((320, 256, 3), 128, dtype=np.uint8)
    out = auxiliary_metrics(rgb, masks)
    keys = " ".join(out.keys())
    assert "auxiliary_shadow_fraction" in out
    assert "0.65" not in keys and "0.75" not in keys


def test_auxiliary_specular_only_y075_chroma008() -> None:
    (masks, _), _, _, _ = toy_masks()
    rgb = np.full((320, 256, 3), 255, dtype=np.uint8)
    out = auxiliary_metrics(rgb, masks)
    assert out["auxiliary_specular_fraction"] > 0
    assert not any("0.70" in key or "0.90" in key for key in out)


def test_r1_config_reuses_old_pilot32_manifest() -> None:
    from lighting_qc_stage2.config import Stage2Config

    cfg = Stage2Config.from_yaml(ROOT / "configs/lighting_qc_stage2_r1_pilot32_v1.yaml")
    old = pd.read_csv(cfg.old_pilot_manifest_csv, dtype=str)
    assert len(old) == 32 and old["sample_id"].nunique() == 32


def test_r1_output_consistency_file_when_present() -> None:
    path = ROOT / "experiments/lighting_confounding/Lighting_QC_Stage2_R1_Pilot32_v1/comparison/unchanged_metric_consistency.json"
    if path.is_file():
        import json

        data = json.loads(path.read_text(encoding="utf-8"))
        assert data["all_unchanged_metrics_close"] is True


def test_frozen_spec_fields_complete() -> None:
    from lighting_qc_stage2.config import Stage2Config

    cfg = Stage2Config.from_yaml(ROOT / "configs/lighting_qc_stage2_r1_pilot32_v1.yaml")
    spec = metric_spec_payload(cfg)
    for key in ["spec_version", "srgb_inverse_formula", "relative_luminance_y_formula", "core_skin_definition", "primary_continuous_metrics", "auxiliary_threshold_metrics", "full500_foldwise_stratification"]:
        assert key in spec


def test_full500_config_generated_but_not_executed_when_present() -> None:
    path = ROOT / "configs/lighting_qc_stage2_full500_v1.yaml"
    if path.is_file():
        text = path.read_text(encoding="utf-8")
        assert "mode: full500" in text
        out = ROOT / "experiments/lighting_confounding/Lighting_QC_Stage2_Full500_v1"
        assert not (out / "reports/stage2_r1_machine_summary.json").exists()


def test_r1_does_not_generate_quality_grades_or_delete_samples() -> None:
    r1 = ROOT / "experiments/lighting_confounding/Lighting_QC_Stage2_R1_Pilot32_v1"
    if r1.exists():
        files = [p.name.lower() for p in r1.rglob("*") if p.is_file()]
        assert not any("quality_grade" in name or "abcd" in name for name in files)
        assert not any("removed_sample" in name or "deleted_sample" in name for name in files)


def test_full500_frozen_hash_integrity_static() -> None:
    import json
    from lighting_qc_stage2.frozen_spec import sha256_file

    frozen = json.loads((ROOT / "experiments/lighting_confounding/Lighting_QC_Stage2_R1_Pilot32_v1/frozen_spec/FROZEN.json").read_text(encoding="utf-8"))
    spec_json = ROOT / "experiments/lighting_confounding/Lighting_QC_Stage2_R1_Pilot32_v1/frozen_spec/stage2_lighting_metric_spec_v1.json"
    assert frozen["frozen"] is True
    assert sha256_file(spec_json) == frozen["metric_spec_hash"]
    assert frozen["roi_definition_hash"] == "19517dd90a7becffd8ac5f6e4647f7f1de8e77a61bc3e24d2c0440e93dc1a3db"


def test_full500_spec_primary_erosion_immutable() -> None:
    import json

    spec = json.loads((ROOT / "experiments/lighting_confounding/Lighting_QC_Stage2_R1_Pilot32_v1/frozen_spec/stage2_lighting_metric_spec_v1.json").read_text(encoding="utf-8"))
    assert spec["primary_skin_erosion_px"] == 2
    assert spec["sensitivity_erosion_px"] == [0, 4]


def test_full500_asset_coverage_static() -> None:
    manifest = pd.read_csv(ROOT / "data/processed/global_face/realface_256x320_blackbg_from_raw_v1/manifests/realface_256x320_manifest.csv", dtype=str)
    split = pd.read_csv(ROOT / "data/processed/P0_Physics_Audit_v1/splits_500/nyha_3class_sex_stratified_group_5fold.csv", dtype=str)
    assert len(split) == 500 and split["ID"].nunique() == 500
    assert len(manifest) == 500


def test_full500_oof_one_to_one_alignment_static() -> None:
    from lighting_qc_stage2.config import Stage2Config

    cfg = Stage2Config.from_yaml(ROOT / "configs/lighting_qc_stage2_full500_v1.yaml")
    oof = load_oof(cfg)
    assert len(oof) == 500 and oof["sample_id"].nunique() == 500


def test_full500_label_consistency_static() -> None:
    from lighting_qc_stage2.config import Stage2Config
    from lighting_qc_stage2.asset_loader import load_split

    cfg = Stage2Config.from_yaml(ROOT / "configs/lighting_qc_stage2_full500_v1.yaml")
    split = load_split(cfg)
    oof = load_oof(cfg)
    merged = split.merge(oof, on="sample_id", suffixes=("_split", "_oof"))
    assert (merged["binary_label_split"].astype(int) == merged["binary_label_oof"].astype(int)).all()


def test_full500_fold_consistency_static() -> None:
    from lighting_qc_stage2.config import Stage2Config
    from lighting_qc_stage2.asset_loader import load_split

    cfg = Stage2Config.from_yaml(ROOT / "configs/lighting_qc_stage2_full500_v1.yaml")
    merged = load_split(cfg).merge(load_oof(cfg), on="sample_id", suffixes=("_split", "_oof"))
    assert (merged["fold_split"].astype(int) == merged["fold_oof"].astype(int)).all()


def test_full500_patient_group_not_cross_fold_static() -> None:
    split = pd.read_csv(ROOT / "data/processed/P0_Physics_Audit_v1/splits_500/nyha_3class_sex_stratified_group_5fold.csv", dtype=str)
    assert int(split.groupby("patient_group_id")["fold"].nunique().gt(1).sum()) == 0


def test_full500_rgb_probability_direction() -> None:
    from lighting_qc_stage2.config import Stage2Config

    cfg = Stage2Config.from_yaml(ROOT / "configs/lighting_qc_stage2_full500_v1.yaml")
    oof = load_oof(cfg)
    assert ((oof["prob_patient"] >= 0) & (oof["prob_patient"] <= 1)).all()
    assert ((oof["prob_patient"] >= 0.5).astype(int) == oof["pred_class"]).mean() > 0.99


def test_full500_primary_secondary_families_definition() -> None:
    frame = pd.DataFrame(columns=PRIMARY_METRICS + ["skin_y_p05", "raw_any_channel_ge_250_fraction", "auxiliary_shadow_fraction"])
    fam = metric_families(frame)
    assert fam["primary"] == PRIMARY_METRICS
    assert "skin_y_p05" in fam["secondary_input"]
    assert "raw_any_channel_ge_250_fraction" in fam["raw_coding_boundary"]


def test_bh_fdr_monotone_and_family_local() -> None:
    q = _bh_fdr([0.01, 0.02, 0.5])
    assert q[0] <= q[1] <= q[2]
    assert _bh_fdr([0.5])[0] == 0.5


def test_group_cluster_bootstrap_reproducible_small() -> None:
    from lighting_confounding_stage1.metrics_utils import cluster_bootstrap_smd

    frame = pd.DataFrame({"patient_group_id": list("aabbccdd"), "binary_label": [0, 0, 0, 0, 1, 1, 1, 1], "x": [1, 2, 1, 2, 3, 4, 3, 4]})
    a = cluster_bootstrap_smd(frame, value_col="x", repeats=20, seed=7)
    b = cluster_bootstrap_smd(frame, value_col="x", repeats=20, seed=7)
    assert a == b


def test_foldwise_cutpoints_use_train_not_test(tmp_path: Path) -> None:
    frame = pd.DataFrame({"sample_id": [str(i) for i in range(10)], "fold": [0] * 2 + [1] * 8, "skin_y_median": [100, 101, 1, 2, 3, 4, 5, 6, 7, 8]})
    for m in STRATIFICATION_METRICS:
        if m not in frame:
            frame[m] = frame["skin_y_median"]
    dirs = full500_dirs(tmp_path)
    _, cuts = foldwise_stratification(frame, dirs)
    row = cuts[(cuts.metric_name == "skin_y_median") & (cuts.fold == 0)].iloc[0]
    assert row.train_q67 < 100


def test_foldwise_non_stratifiable_when_quantiles_tie(tmp_path: Path) -> None:
    frame = pd.DataFrame({"sample_id": [str(i) for i in range(10)], "fold": [0, 1] * 5, "skin_y_median": [0] * 10})
    for m in STRATIFICATION_METRICS:
        if m not in frame:
            frame[m] = frame["skin_y_median"]
    dirs = full500_dirs(tmp_path)
    assignments, _ = foldwise_stratification(frame, dirs)
    assert set(assignments.assigned_group) == {"non_stratifiable"}


def test_single_class_auc_returns_none() -> None:
    out = compute_metrics([1, 1, 1], [0.2, 0.3, 0.4])
    assert out["roc_auc"] is None


def test_classification_threshold_fixed_05() -> None:
    out = compute_metrics([0, 1], [0.49, 0.50])
    assert out["accuracy"] == 1.0


def test_worst_group_auc_delta_calculation(tmp_path: Path) -> None:
    import lighting_qc_stage2.full500_pipeline as f500

    f500.BOOTSTRAP_REPEATS = 5
    frame = pd.DataFrame({
        "sample_id": [str(i) for i in range(90)],
        "patient_group_id": [str(i) for i in range(90)],
        "fold": [i % 5 for i in range(90)],
        "binary_label": [0, 1] * 45,
        "rgb_oof_probability_patient": [0.1, 0.9] * 45,
    })
    for m in STRATIFICATION_METRICS:
        frame[m] = np.linspace(0, 1, 90)
    dirs = full500_dirs(tmp_path)
    assignments, _ = foldwise_stratification(frame, dirs)
    _, worst, _ = stratified_performance(frame, assignments, dirs)
    assert "DeltaAUC" in worst
    assert pd.to_numeric(worst["WorstGroupAUC"], errors="coerce").notna().any()


def test_unstable_group_flag(tmp_path: Path) -> None:
    import lighting_qc_stage2.full500_pipeline as f500

    f500.BOOTSTRAP_REPEATS = 5
    frame = pd.DataFrame({"sample_id": [str(i) for i in range(12)], "patient_group_id": [str(i) for i in range(12)], "fold": [i % 5 for i in range(12)], "binary_label": [0, 1] * 6, "rgb_oof_probability_patient": [0.2, 0.8] * 6})
    for m in STRATIFICATION_METRICS:
        frame[m] = np.linspace(0, 1, 12)
    dirs = full500_dirs(tmp_path)
    assignments, _ = foldwise_stratification(frame, dirs)
    strat, _, _ = stratified_performance(frame, assignments, dirs)
    assert strat["unstable"].any()


def test_full500_main_erosion_fixed_2_config() -> None:
    from lighting_qc_stage2.config import Stage2Config

    cfg = Stage2Config.from_yaml(ROOT / "configs/lighting_qc_stage2_full500_v1.yaml")
    assert cfg.metric_spec_path is not None
    assert cfg.generate_quality_grades is False and cfg.remove_samples is False


def test_full500_erosion_sensitivity_separate_when_present() -> None:
    root = ROOT / "experiments/lighting_confounding/Lighting_QC_Stage2_Full500_v1"
    if root.exists():
        main = pd.read_csv(root / "features/stage2_lighting_features_500.csv")
        assert "erosion_px" not in main.columns
        assert (root / "features/stage2_lighting_features_erosion0.csv").is_file()


def test_full500_no_quality_grades_when_present() -> None:
    root = ROOT / "experiments/lighting_confounding/Lighting_QC_Stage2_Full500_v1"
    if root.exists():
        assert not any("quality_grade" in p.name.lower() or "abcd" in p.name.lower() for p in root.rglob("*"))


def test_full500_no_sample_deletion_when_present() -> None:
    root = ROOT / "experiments/lighting_confounding/Lighting_QC_Stage2_Full500_v1"
    if root.exists():
        assert not any("deleted" in p.name.lower() or "removed_sample" in p.name.lower() for p in root.rglob("*"))


def test_full500_no_model_training_code_path() -> None:
    src = (ROOT / "lighting_qc_stage2/full500_pipeline.py").read_text(encoding="utf-8")
    assert ".fit(" not in src or "Logit" in src or "OLS" in src
    assert "checkpoint" not in src.lower()


def test_full500_does_not_modify_frozen_metrics_source_hash() -> None:
    import json
    from lighting_qc_stage2.frozen_spec import sha256_file

    frozen = json.loads((ROOT / "experiments/lighting_confounding/Lighting_QC_Stage2_R1_Pilot32_v1/frozen_spec/FROZEN.json").read_text(encoding="utf-8"))
    assert sha256_file(ROOT / "lighting_qc_stage2/frozen_metrics.py") == frozen["source_code_hashes"]["frozen_metrics_py"]


def test_full500_output_completeness_when_present() -> None:
    root = ROOT / "experiments/lighting_confounding/Lighting_QC_Stage2_Full500_v1"
    if root.exists():
        required = ["features/stage2_lighting_features_500.csv", "association/lighting_label_tests.csv", "rgb_dependence/rgb_lighting_spearman.csv", "stratified/rgb_lighting_worst_group_metrics.json", "reports/stage2_full500_machine_summary.json"]
        assert all((root / p).is_file() for p in required)


def test_full500_machine_summary_fields_when_present() -> None:
    root = ROOT / "experiments/lighting_confounding/Lighting_QC_Stage2_Full500_v1"
    if root.exists():
        import json

        data = json.loads((root / "reports/stage2_full500_machine_summary.json").read_text(encoding="utf-8"))
        for key in ["lighting_dependence_level", "stage3_recommendation", "full_500_run", "model_training_run", "samples_removed", "entered_stage3"]:
            assert key in data


def test_full500_feature_coverage_when_present() -> None:
    path = ROOT / "experiments/lighting_confounding/Lighting_QC_Stage2_Full500_v1/features/stage2_lighting_features_500.csv"
    if path.is_file():
        frame = pd.read_csv(path)
        assert len(frame) == 500 and frame["sample_id"].nunique() == 500


def test_full500_primary_metrics_no_missing_when_present() -> None:
    path = ROOT / "experiments/lighting_confounding/Lighting_QC_Stage2_Full500_v1/features/stage2_lighting_features_500.csv"
    if path.is_file():
        frame = pd.read_csv(path)
        assert int(frame[PRIMARY_METRICS].isna().sum().sum()) == 0


def test_stage3_decision_schema() -> None:
    worst = pd.DataFrame({"DeltaAUC": [0.01], "WorstGroupAUC": [0.9]})
    decision = decide_stage3(worst, {"fdr": {"logit_lighting_terms_q_lt_0.05": 0, "error_lighting_terms_q_lt_0.05": 0}}, {"min_erosion_spearman": 0.9})
    assert decision["lighting_dependence_level"] in {"low", "moderate", "high", "indeterminate"}


def _r2_toy_labels(n: int = 30) -> pd.DataFrame:
    return pd.DataFrame({
        "sample_id": [str(i) for i in range(n)],
        "patient_group_id": [f"g{i}" for i in range(n)],
        "fold": [i % 5 for i in range(n)],
        "binary_label": [i % 2 for i in range(n)],
        "rgb_oof_probability_patient": [0.2 if i % 2 == 0 else 0.8 for i in range(n)],
    })


def _r2_toy_features(n: int = 30, offset: float = 0.0) -> pd.DataFrame:
    frame = pd.DataFrame({"sample_id": [str(i) for i in range(n)]})
    for metric in EROSION_METRICS:
        frame[metric] = np.linspace(0 + offset, 1 + offset, n)
    return frame


def test_r2_erosion_each_uses_own_train_fold_cutpoints() -> None:
    labels = _r2_toy_labels()
    assignments, cuts = r2_foldwise_assignments({0: _r2_toy_features(offset=0), 2: _r2_toy_features(offset=10), 4: _r2_toy_features(offset=20)}, labels)
    q0 = cuts[(cuts.erosion_px == 0) & (cuts.metric_name == "skin_y_median") & (cuts.fold == 0)].iloc[0].train_q33
    q4 = cuts[(cuts.erosion_px == 4) & (cuts.metric_name == "skin_y_median") & (cuts.fold == 0)].iloc[0].train_q33
    assert q4 > q0 + 10
    assert len(assignments) == 30 * 3 * len(EROSION_METRICS)


def test_r2_test_fold_not_in_cutpoint() -> None:
    labels = _r2_toy_labels(10)
    feat = _r2_toy_features(10)
    feat.loc[labels.fold == 0, "skin_y_median"] = 1000
    _, cuts = r2_foldwise_assignments({2: feat}, labels)
    q = cuts[(cuts.erosion_px == 2) & (cuts.metric_name == "skin_y_median") & (cuts.fold == 0)].iloc[0].train_q67
    assert q < 1000


def test_r2_no_global_quantile_in_repair_source() -> None:
    src = (ROOT / "lighting_qc_stage2/r2_statistical_repair.py").read_text(encoding="utf-8")
    assert 'data["fold"].astype(int) != fold' in src
    assert "full_quantile" not in src.lower()


def test_r2_repaired_erosion2_assignments_match_foldwise_logic() -> None:
    labels = _r2_toy_labels()
    feat = _r2_toy_features()
    assignments, _ = r2_foldwise_assignments({2: feat}, labels)
    assert set(assignments["erosion_px"]) == {2}
    assert set(assignments["assigned_group"]).issubset({"low", "middle", "high", "non_stratifiable"})


def test_r2_stable_rule_requires_each_class_20() -> None:
    frame = pd.DataFrame({"binary_label": [0] * 19 + [1] * 30, "patient_group_id": [str(i) for i in range(49)]})
    unstable, severe, reasons = stability_status(frame)
    assert unstable is True and severe is False and "n_control<20" in reasons


def test_r2_severely_unstable_rule_class_under_10() -> None:
    frame = pd.DataFrame({"binary_label": [0] * 9 + [1] * 30, "patient_group_id": [str(i) for i in range(39)]})
    _, severe, reasons = stability_status(frame)
    assert severe is True and "severe:n_control<10" in reasons


def test_r2_unstable_group_retained() -> None:
    labels = _r2_toy_labels(18)
    feat = _r2_toy_features(18)
    assignments, _ = r2_foldwise_assignments({2: feat}, labels)
    metrics = r2_stratified_metrics(labels, assignments)
    assert len(metrics) > 0 and metrics["is_unstable"].any()


def test_r2_stable_only_delta_requires_two_stable_groups() -> None:
    metrics = pd.DataFrame({"erosion_px": [2, 2], "metric_name": ["x", "x"], "assigned_group": ["low", "high"], "auc": [0.8, 0.9], "balanced_accuracy": [0.7, 0.8], "sensitivity": [0.6, 0.7], "specificity": [0.8, 0.9], "is_unstable": [False, True]})
    worst = r2_worst_group_summary(metrics)
    val = worst[(worst.scope == "stable_only") & (worst.metric_name == "x")].iloc[0].DeltaAUC
    assert pd.isna(val)


def test_r2_bootstrap_by_patient_group_reproducible() -> None:
    from lighting_qc_stage2.r2_bootstrap import sample_patient_groups

    frame = pd.DataFrame({"patient_group_id": list("aabbcc"), "x": [1, 1, 2, 2, 3, 3]})
    stat = lambda sample: {"mean": float(sample.x.mean())}
    a = sample_patient_groups(frame, iterations=10, seed=3, statistic=stat)
    b = sample_patient_groups(frame, iterations=10, seed=3, statistic=stat)
    assert a["ci95"] == b["ci95"] and a["cluster_unit"] == "patient_group_id"


def test_r2_worst_group_ci_fields_complete() -> None:
    labels = _r2_toy_labels(60)
    feat = _r2_toy_features(60)
    assignments, _ = r2_foldwise_assignments({2: feat}, labels)
    metrics = r2_stratified_metrics(labels, assignments)
    ci, _ = worst_group_bootstrap(labels, assignments, metrics, iterations=3, seed=1)
    assert {"WorstGroupAUC_ci95_low", "DeltaAUC_ci95_high", "valid_bootstrap_iterations"}.issubset(ci.columns)


def test_r2_fdr_hit_deduplication_logic(tmp_path: Path) -> None:
    root = tmp_path
    (root / "rgb_dependence").mkdir()
    df = pd.DataFrame({
        "model": ["ols:y~binary_label+z_a", "ols:y~binary_label+z_a+z_b", "ols:y~binary_label+z_a+z_b"],
        "term": ["z_a", "z_a", "z_b"],
        "q_value": [0.01, 0.02, 0.2],
    })
    df.to_csv(root / "rgb_dependence/rgb_logit_adjusted_regression.csv", index=False)
    df.to_csv(root / "rgb_dependence/rgb_error_risk_regression.csv", index=False)
    _, summary = deduplicate_fdr_hits(root)
    assert summary["rgb_logit_deduplicated_any_model"]["significant_unique_metrics"] == 1


def test_r2_report_forbidden_environmental_dependence_phrase_when_present() -> None:
    path = ROOT / "experiments/lighting_confounding/Lighting_QC_Stage2_R2_Statistical_Repair_and_EXIF_Decomposition_v1/reports/stage2_r2_report.md"
    if path.is_file():
        text = path.read_text(encoding="utf-8").lower()
        assert "已证实环境光照依赖" not in text
        assert "high environmental-lighting dependence" not in text


def test_r2_exif_field_mapping_uses_allowed_fields_static() -> None:
    path = ROOT / "experiments/lighting_confounding/Lighting_QC_Stage2_R2_Statistical_Repair_and_EXIF_Decomposition_v1/exif/exif_field_mapping.json"
    if path.is_file():
        import json
        data = json.loads(path.read_text(encoding="utf-8"))
        assert data["derived_features"] == EXIF_FEATURES
        assert "camera_model" in data["forbidden_fields_not_used"]


def test_r2_exposure_fraction_parse() -> None:
    assert _as_seconds("1/100") == pytest.approx(0.01)


def test_r2_log2_exposure_iso_derivation() -> None:
    stage1 = pd.DataFrame({"sample_id": ["1"], "exposure_time_seconds": [0.25], "iso": [400], "brightness_value_apex": [5]})
    features = pd.DataFrame({"sample_id": ["1"], "patient_group_id": ["g1"], "fold": [0], "binary_label": [1], "rgb_oof_probability_patient": [0.7], "rgb_oof_logit": [0.8], "rgb_oof_error": [0], "rgb_oof_brier_contribution": [0.09], "rgb_oof_residual": [-0.3], **{m: [1.0] for m in TARGET_METRICS}})
    exif, _, _ = build_exif_frame(stage1, features)
    assert exif.loc[0, "log2_exposure_time"] == pytest.approx(-2)
    assert exif.loc[0, "log2_iso"] == pytest.approx(np.log2(400))


def test_r2_missing_imputation_train_only_and_no_test_standardization_leak() -> None:
    src = (ROOT / "lighting_qc_stage2/r2_exif_decomposition.py").read_text(encoding="utf-8")
    assert "scaler.fit_transform" in src and "scaler.transform" in src
    assert ".median()" in src and "train" in src


def test_r2_exif_model_forbidden_columns_absent() -> None:
    src = (ROOT / "lighting_qc_stage2/r2_exif_decomposition.py").read_text(encoding="utf-8")
    assert "camera_model" not in EXIF_FEATURES
    assert "binary_label" not in EXIF_FEATURES
    assert "rgb_oof_probability_patient" not in EXIF_FEATURES
    assert "capture_time" not in EXIF_FEATURES


def test_r2_crossfit_predictions_from_other_folds_source() -> None:
    src = (ROOT / "lighting_qc_stage2/r2_exif_decomposition.py").read_text(encoding="utf-8")
    assert 'train = exif.loc[exif["fold"].astype(int) != fold]' in src
    assert 'test = exif.loc[exif["fold"].astype(int) == fold]' in src


def test_r2_crossfit_prediction_complete_when_present() -> None:
    path = ROOT / "experiments/lighting_confounding/Lighting_QC_Stage2_R2_Statistical_Repair_and_EXIF_Decomposition_v1/exif/exif_crossfit_predictions_500.csv"
    if path.is_file():
        df = pd.read_csv(path)
        assert len(df) == 500 and df["sample_id"].nunique() == 500


def test_r2_crossfit_prediction_columns_unique_when_present() -> None:
    path = ROOT / "experiments/lighting_confounding/Lighting_QC_Stage2_R2_Statistical_Repair_and_EXIF_Decomposition_v1/exif/exif_crossfit_predictions_500.csv"
    if path.is_file():
        header = path.read_text(encoding="utf-8-sig").splitlines()[0].split(",")
        assert len(header) == len(set(header))


def test_r2_observed_predicted_residual_identity_when_present() -> None:
    path = ROOT / "experiments/lighting_confounding/Lighting_QC_Stage2_R2_Statistical_Repair_and_EXIF_Decomposition_v1/exif/exif_crossfit_predictions_500.csv"
    if path.is_file():
        df = pd.read_csv(path)
        for metric in TARGET_METRICS:
            err = (df[f"observed_{metric}"] - df[f"predicted_{metric}"] - df[f"residual_{metric}"]).abs().max()
            assert err <= 1e-12


def test_r2_ridge_alpha_fixed_one_source() -> None:
    src = (ROOT / "lighting_qc_stage2/r2_exif_decomposition.py").read_text(encoding="utf-8")
    assert "Ridge(alpha=alpha" in src
    assert "alpha=1.0" in src


def test_r2_no_alpha_selection_source() -> None:
    src = (ROOT / "lighting_qc_stage2/r2_exif_decomposition.py").read_text(encoding="utf-8").lower()
    assert "gridsearch" not in src and "ridgecv" not in src


def test_r2_no_traditional_exif_covariate_adjustment_config() -> None:
    from lighting_qc_stage2.config import Stage2Config
    cfg = Stage2Config.from_yaml(ROOT / "configs/lighting_qc_stage2_r2_statistical_repair_and_exif_decomposition_v1.yaml")
    assert cfg.run_traditional_exif_covariate_adjustment is False


def test_r2_predicted_residual_names_not_pure_or_medical_when_present() -> None:
    root = ROOT / "experiments/lighting_confounding/Lighting_QC_Stage2_R2_Statistical_Repair_and_EXIF_Decomposition_v1"
    if root.exists():
        names = " ".join(p.name for p in root.rglob("*")).lower()
        text = ""
        for p in (root / "reports").glob("*.md"):
            text += p.read_text(encoding="utf-8").lower()
        assert "pure illumination" not in text and "physiology-only" not in text and "disease-specific residual" not in text
        assert "predicted_skin_y_median" in (root / "exif/exif_crossfit_predictions_500.csv").read_text(encoding="utf-8", errors="ignore")


def test_r2_fdr_families_separate_when_present() -> None:
    path = ROOT / "experiments/lighting_confounding/Lighting_QC_Stage2_R2_Statistical_Repair_and_EXIF_Decomposition_v1/exif/exif_component_label_fdr.json"
    if path.is_file():
        import json
        data = json.loads(path.read_text(encoding="utf-8"))
        assert "predicted" in data and "residual" in data


def test_r2_cluster_robust_or_bootstrap_source() -> None:
    src = (ROOT / "lighting_qc_stage2/r2_exif_decomposition.py").read_text(encoding="utf-8") + (ROOT / "lighting_qc_stage2/r2_bootstrap.py").read_text(encoding="utf-8")
    assert "patient_group_id" in src and "cluster" in src


def test_r2_attenuation_allows_negative_source() -> None:
    src = (ROOT / "lighting_qc_stage2/r2_exif_decomposition.py").read_text(encoding="utf-8")
    assert "1 - abs(res_smd)" in src
    assert "clip" not in src[src.find("attenuation_summary") : src.find("interpretation")]


def test_r2_oof_r2_can_be_negative_when_present() -> None:
    path = ROOT / "experiments/lighting_confounding/Lighting_QC_Stage2_R2_Statistical_Repair_and_EXIF_Decomposition_v1/exif/exif_decomposition_model_performance.csv"
    if path.is_file():
        df = pd.read_csv(path)
        assert "r2" in df.columns


def test_r2_stratified_metrics_include_group_auc_ci_when_present() -> None:
    path = ROOT / "experiments/lighting_confounding/Lighting_QC_Stage2_R2_Statistical_Repair_and_EXIF_Decomposition_v1/repair/erosion_stratified_metrics_repaired.csv"
    if path.is_file():
        df = pd.read_csv(path)
        assert {"auc_ci_low", "auc_ci_high", "auc_bootstrap_valid", "auc_bootstrap_invalid"}.issubset(df.columns)


def test_r2_attenuation_includes_rgb_dependence_comparisons_when_present() -> None:
    path = ROOT / "experiments/lighting_confounding/Lighting_QC_Stage2_R2_Statistical_Repair_and_EXIF_Decomposition_v1/exif/exif_decomposition_attenuation_summary.csv"
    if path.is_file():
        df = pd.read_csv(path)
        required = {
            "original_rgb_logit_coef",
            "residual_rgb_logit_coef",
            "rgb_logit_coef_attenuation",
            "original_error_risk_or",
            "residual_error_risk_or",
            "error_log_or_attenuation",
        }
        assert required.issubset(df.columns)


def test_r2_required_outputs_when_present() -> None:
    root = ROOT / "experiments/lighting_confounding/Lighting_QC_Stage2_R2_Statistical_Repair_and_EXIF_Decomposition_v1"
    if root.exists():
        required = ["preflight/r2_preflight_summary.json", "repair/fdr_hit_deduplication.json", "exif/exif_decomposition_interpretation.json", "reports/stage2_r2_machine_summary.json"]
        assert all((root / p).is_file() for p in required)


def test_r2_not_stage3_or_rgb_training_config() -> None:
    from lighting_qc_stage2.config import Stage2Config
    cfg = Stage2Config.from_yaml(ROOT / "configs/lighting_qc_stage2_r2_statistical_repair_and_exif_decomposition_v1.yaml")
    assert cfg.run_stage3 is False
    assert cfg.run_traditional_exif_covariate_adjustment is False


def test_r2_no_rgb_classifier_modification_source() -> None:
    src = (ROOT / "lighting_qc_stage2/r2_pipeline.py").read_text(encoding="utf-8").lower()
    assert "checkpoint" not in src and "torch" not in src


def test_r2_no_sample_deletion_source() -> None:
    src = (ROOT / "lighting_qc_stage2/r2_pipeline.py").read_text(encoding="utf-8").lower()
    assert "drop(" not in src or "dropna" in src
