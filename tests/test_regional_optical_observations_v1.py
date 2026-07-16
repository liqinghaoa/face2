"""Focused tests for Regional Optical Observations V1."""

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from PIL import Image

from preprocessing.extract_regional_optical_observations_v1 import (
    EPSILON,
    FOREHEAD_THRESHOLD,
    REPORT_TASK_MARKER,
    REPORT_TITLE,
    ExtractionFailure,
    build_feature_schema,
    build_main_row,
    channel_clipping,
    compute_roi_qc,
    derive_case_observations,
    derive_exif,
    inverse_srgb,
    pixel_observations,
    read_rgb_uint8,
    required_manifest_columns,
    robust_summary,
    summary_record,
    validate_positive_exif,
    verify_generated_output,
    write_failure_log,
)


def full_mask() -> np.ndarray:
    return np.ones((224, 224), dtype=np.uint8) * 255


def test_inverse_srgb_endpoints_breakpoint_and_shape() -> None:
    values = np.array([[0.0, 0.04045, 1.0]], dtype=np.float64)
    result = inverse_srgb(values)
    assert result.shape == values.shape
    assert result[0, 0] == 0.0
    assert result[0, 1] == pytest.approx(0.04045 / 12.92)
    assert result[0, 2] == pytest.approx(1.0)


def test_rgb_reader_does_not_swap_bgr(tmp_path: Path) -> None:
    array = np.zeros((224, 224, 3), dtype=np.uint8)
    array[0, 0] = [255, 0, 0]
    array[0, 1] = [0, 255, 0]
    array[0, 2] = [0, 0, 255]
    path = tmp_path / "rgb.png"
    Image.fromarray(array, mode="RGB").save(path)
    loaded = read_rgb_uint8(path)
    assert np.array_equal(loaded[0, :3], array[0, :3])


def test_optical_formula_for_pure_channels_is_correct_and_finite() -> None:
    rgb = np.zeros((224, 224, 3), dtype=np.uint8)
    rgb[0, 0] = [255, 0, 0]
    rgb[0, 1] = [0, 255, 0]
    rgb[0, 2] = [0, 0, 255]
    mask = np.zeros((224, 224), dtype=np.uint8)
    mask[0, :3] = 255
    result = pixel_observations(rgb, mask)
    expected_y = np.log2(np.array([0.2126, 0.7152, 0.0722]) + EPSILON)
    assert np.allclose(result["log2_y"], expected_y)
    assert result["log2_rg"][0] == pytest.approx(np.log2((1 + EPSILON) / EPSILON))
    assert result["log2_bg"][2] == pytest.approx(np.log2((1 + EPSILON) / EPSILON))
    assert all(np.isfinite(values).all() for values in result.values())


def test_roi_statistics_exclude_mask_outside_and_preserve_input() -> None:
    rgb = np.full((224, 224, 3), 50, dtype=np.uint8)
    rgb[0, 0] = [10, 20, 30]
    rgb[0, 1] = [40, 50, 60]
    rgb[0, 2] = [250, 250, 250]
    before = rgb.copy()
    mask = np.zeros((224, 224), dtype=np.uint8)
    mask[0, :2] = 255
    observations = pixel_observations(rgb, mask)
    summary = robust_summary(observations["log2_y"])
    assert summary["q25"] == pytest.approx(np.quantile(observations["log2_y"], 0.25))
    assert summary["median_raw"] == pytest.approx(np.median(observations["log2_y"]))
    assert summary["q75"] == pytest.approx(np.quantile(observations["log2_y"], 0.75))
    assert summary["iqr"] == pytest.approx(summary["q75"] - summary["q25"])
    assert np.array_equal(rgb, before)
    assert len(observations["log2_y"]) == 2


def test_channel_clipping_uses_raw_uint8_inside_mask_only() -> None:
    rgb = np.full((224, 224, 3), 100, dtype=np.uint8)
    rgb[0, 0] = [0, 255, 5]
    rgb[0, 1] = [255, 0, 250]
    rgb[0, 2] = [0, 0, 0]
    mask = np.zeros((224, 224), dtype=np.uint8)
    mask[0, :2] = 255
    result = channel_clipping(rgb, mask)
    assert result["r_equal_0_fraction"] == 0.5
    assert result["r_equal_255_fraction"] == 0.5
    assert result["g_equal_0_fraction"] == 0.5
    assert result["b_le_5_fraction"] == 0.5
    assert result["b_ge_250_fraction"] == 0.5


def qc_rows() -> dict[str, dict[str, float]]:
    rows = {}
    for roi, base in (("forehead", 3.0), ("cheek_image_left", 1.0), ("cheek_image_right", 2.0)):
        rows[roi] = {f"{name}_median_raw": base + index for index, name in enumerate(("log2_y", "log2_rg", "log2_bg"))}
    return rows


def source_row(fraction: float) -> dict[str, float | str]:
    return {
        "ID": "ID1", "camera_id": "Make/Model", "ExposureTime": 0.01,
        "FNumber": 2.0, "ISOSpeedRatings": 200.0,
        "forehead_valid_skin_fraction": fraction,
    }


def test_forehead_rule_boundary_and_unavailable_main_nan() -> None:
    unavailable = build_main_row(source_row(0.199999), qc_rows())
    available = build_main_row(source_row(FOREHEAD_THRESHOLD), qc_rows())
    assert unavailable["forehead_available"] == 0
    assert np.isnan(unavailable["forehead_log2_y_median"])
    assert np.isnan(unavailable["forehead_minus_cheek_log2_y"])
    assert unavailable["cheek_image_left_log2_y_median"] == 1.0
    assert available["forehead_available"] == 1
    assert available["forehead_log2_y_median"] == 3.0


def test_unavailable_forehead_qc_still_contains_raw_statistics() -> None:
    rgb = np.full((224, 224, 3), 100, dtype=np.uint8)
    row = compute_roi_qc(rgb, full_mask(), "ID1", "M/M", "forehead", 501760, 50176, 0.1)
    assert row["available_for_model"] == 0
    assert np.isfinite(row["log2_y_median_raw"])
    assert np.isfinite(row["log2_rg_q25"])


def test_derived_observations() -> None:
    result = derive_case_observations(qc_rows(), forehead_available=True)
    assert result["cheek_mean_log2_y"] == 1.5
    assert result["cheek_abs_diff_log2_y"] == 1.0
    assert result["forehead_minus_cheek_log2_y"] == 1.5
    result_unavailable = derive_case_observations(qc_rows(), forehead_available=False)
    assert np.isnan(result_unavailable["forehead_minus_cheek_log2_y"])


def test_exif_formulas_and_nonpositive_rejection() -> None:
    relative, iso = derive_exif(0.01, 2.0, 400)
    assert relative == pytest.approx(np.log2(0.01 / 4.0))
    assert iso == pytest.approx(2.0)
    for values in ((0, 2.0, 100), (0.01, 0, 100), (0.01, 2.0, 0), (np.nan, 2.0, 100)):
        with pytest.raises(ValueError):
            validate_positive_exif(*values)


def test_schema_roles_exclude_device_exif_qc_and_clinical_labels() -> None:
    class Args:
        project_root = Path(".").resolve()
        aligned_rgb_dir = project_root / "aligned"
        mask_root = project_root / "masks"

    schema = build_feature_schema(Args())
    core = set(schema["core_v1_observation_columns"])
    assert "camera_id" not in core
    assert "valid_skin_fraction" not in core
    assert "cheek_abs_diff_log2_y" not in core
    assert "camera_id" in schema["forbidden_direct_classifier_columns"]
    all_names = " ".join(str(value) for value in schema.values()).lower()
    assert "nyha" not in all_names
    assert "sex" not in all_names


def test_brightness_value_is_not_a_required_or_core_field() -> None:
    assert "BrightnessValue" not in required_manifest_columns()
    schema = build_feature_schema(SimpleNamespace(
        project_root=Path(".").resolve(),
        aligned_rgb_dir=Path(".").resolve() / "aligned",
        mask_root=Path(".").resolve() / "masks",
    ))
    assert "BrightnessValue" not in schema["core_v1_observation_columns"]


def test_all_missing_summary_counts_every_missing_value() -> None:
    result = summary_record(pd.Series([np.nan, None, "not-numeric"]))
    assert result["valid_n"] == 0
    assert result["missing_n"] == 3
    for field in ("min", "q25", "median", "q75", "max", "mean", "std", "iqr"):
        assert np.isnan(result[field])


def test_no_resize_or_morphology_code_path_in_core_statistics() -> None:
    rgb = np.full((224, 224, 3), 127, dtype=np.uint8)
    rgb[73, 91] = [10, 20, 30]
    mask = np.zeros((224, 224), dtype=np.uint8)
    mask[73, 91] = 255
    rgb_before = rgb.copy()
    mask_before = mask.copy()
    result = compute_roi_qc(rgb, mask, "ID", "M/M", "cheek_image_left", 1, 1, 1.0)
    assert result["valid_skin_pixel_count"] == 1
    assert result["transformed_nonfinite_count"] == 0
    expected = pixel_observations(rgb_before, mask_before)["log2_y"][0]
    assert result["log2_y_median_raw"] == pytest.approx(expected)
    assert np.array_equal(rgb, rgb_before)
    assert np.array_equal(mask, mask_before)
    with pytest.raises(ValueError, match="Mask must be 224x224"):
        pixel_observations(rgb, mask[:100])


def test_report_output_verification_requires_task_signature(tmp_path: Path) -> None:
    report = tmp_path / "optical_observation_extraction_report.md"
    report.write_text("# unrelated report\n", encoding="utf-8")
    assert not verify_generated_output(tmp_path, dataset=False)
    report.write_text(f"{REPORT_TASK_MARKER}\n{REPORT_TITLE}\n", encoding="utf-8")
    assert verify_generated_output(tmp_path, dataset=False)


def test_failure_log_does_not_overwrite_without_authorized_output(tmp_path: Path) -> None:
    existing = tmp_path / "extraction_run.log"
    existing.write_text("UNRELATED\n", encoding="utf-8")
    args = SimpleNamespace(report_output_dir=tmp_path, overwrite=False)
    result = write_failure_log(args, ExtractionFailure("test", ["expected failure"]))
    assert result is None
    assert existing.read_text(encoding="utf-8") == "UNRELATED\n"


def test_failure_log_rejects_unrelated_directory_even_with_overwrite(tmp_path: Path) -> None:
    report = tmp_path / "optical_observation_extraction_report.md"
    report.write_text("# unrelated report\n", encoding="utf-8")
    args = SimpleNamespace(report_output_dir=tmp_path, overwrite=True)
    result = write_failure_log(args, ExtractionFailure("test", ["expected failure"]))
    assert result is None
    assert report.read_text(encoding="utf-8") == "# unrelated report\n"
