from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from .asset_loader import write_json
from .config import Stage2Config, json_safe


SPEC_VERSION = "stage2_lighting_metric_spec_v1"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def roi_definition_hash(config: Stage2Config) -> str:
    payload = {
        "roi_geometry": config.roi_geometry,
        "core_skin": "parsing_label == skin AND source_valid_mask AND face_valid_mask; remove components <16; ellipse erosion",
        "nose": "parsing_label == nose AND source_valid_mask AND normalized nose candidate",
        "cheek_excludes_nose": True,
        "forehead_intersects_core_skin_e2": True,
    }
    return sha256_text(json.dumps(payload, sort_keys=True, ensure_ascii=False))


def affine_definition_hash() -> str:
    return sha256_text(
        "metadata.affine_canvas_to_source; PIL nearest-neighbor mask transform; source read with PIL ImageOps.exif_transpose; roundtrip IoU>=0.95 centroid<=2px"
    )


def metric_spec_payload(config: Stage2Config) -> dict[str, Any]:
    return {
        "spec_version": SPEC_VERSION,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "scheme_b_root": str(config.scheme_b_root),
        "scheme_b_manifest": str(config.scheme_b_root / "manifests" / "realface_256x320_manifest.csv"),
        "srgb_inverse_formula": "c<=0.04045: c/12.92; else ((c+0.055)/1.055)^2.4",
        "relative_luminance_y_formula": "Y = 0.2126 R_linear + 0.7152 G_linear + 0.0722 B_linear",
        "core_skin_definition": "parsing_label == skin_class AND source_valid_mask AND face_valid_mask",
        "primary_skin_erosion_px": 2,
        "sensitivity_erosion_px": [0, 4],
        "roi_definition": {
            "coordinate_system": "normalized to face_geometry bbox on 256x320 canvas",
            "geometry": config.roi_geometry,
            "final_masks": "forehead/canvas cheek masks intersect core_skin_e2; cheeks exclude nose; nose uses parser nose class and source_valid",
            "canvas_left_right_note": "canvas left/right are not patient anatomical left/right",
        },
        "raw_mapping_definition": "canvas masks mapped to EXIF-transposed source coordinates using frozen Scheme B affine_canvas_to_source and nearest-neighbor interpolation",
        "primary_continuous_metrics": [
            "skin_y_mean",
            "skin_y_median",
            "skin_y_std",
            "skin_y_iqr",
            "skin_y_mad",
            "skin_y_p01",
            "skin_y_p05",
            "skin_y_p10",
            "skin_y_p25",
            "skin_y_p75",
            "skin_y_p90",
            "skin_y_p95",
            "skin_y_p99",
            "skin_y_p90_minus_p10",
            "skin_y_mad_over_median",
            "dark_contrast_score",
            "deep_dark_contrast_score",
            "canvas_left_cheek_y_median",
            "canvas_right_cheek_y_median",
            "cheek_signed_difference",
            "cheek_absolute_difference",
            "cheek_relative_difference",
            "forehead_y_median",
            "mean_cheek_y",
            "forehead_cheek_signed_difference",
            "forehead_cheek_absolute_difference",
            "nose_y_median",
            "nose_y_p90",
            "nose_y_p95",
            "nose_y_p99",
            "forehead_y_p90",
            "forehead_y_p95",
            "forehead_y_p99",
        ],
        "auxiliary_threshold_metrics": {
            "severe_bright_fraction": "core_skin_e2 Y >= 0.80",
            "severe_dark_fraction": "core_skin_e2 Y <= 0.05",
            "deep_dark_fraction": "core_skin_e2 Y <= 0.03",
            "auxiliary_shadow_fraction": "Gaussian-smoothed Y < 0.55 * skin_y_median within core_skin_e2; relative dark-region candidate",
            "deep_relative_dark_fraction": "Gaussian-smoothed Y < 0.45 * skin_y_median within core_skin_e2",
            "auxiliary_specular_fraction": "Y >= 0.75 AND max(linear_rgb)-min(linear_rgb)<=0.08 in core_skin_e2 OR nose; low-chroma high-luminance candidate",
        },
        "raw_coding_boundary_metrics": [
            "raw_r_eq_255_fraction",
            "raw_g_eq_255_fraction",
            "raw_b_eq_255_fraction",
            "raw_any_channel_eq_255_fraction",
            "raw_all_channels_eq_255_fraction",
            "raw_r_eq_0_fraction",
            "raw_g_eq_0_fraction",
            "raw_b_eq_0_fraction",
            "raw_any_channel_eq_0_fraction",
            "raw_all_channels_eq_0_fraction",
            "raw_any_channel_ge_250_fraction",
            "raw_all_channels_ge_250_fraction",
            "raw_any_channel_le_5_fraction",
            "raw_all_channels_le_5_fraction",
        ],
        "missing_value_rule": "empty or invalid ROI returns NA/None; no fabricated zero",
        "invalid_roi_rule": "engineering failure below 32 pixels; report valid_ge_64/128/256 but do not freeze area threshold",
        "forbidden_actions": ["remove_samples", "generate_A_B_C_D_quality_grades", "modify_scheme_b_assets"],
        "full500_foldwise_stratification": {
            "rule": "within each outer fold, fit 33.33% and 66.67% quantiles on train portion only, apply to that fold test portion, then merge test predictions",
            "metrics": [
                "skin_y_median",
                "skin_y_p05",
                "skin_y_p99",
                "skin_y_iqr",
                "dark_contrast_score",
                "cheek_relative_difference",
                "forehead_cheek_absolute_difference",
                "raw_any_channel_ge_250_fraction",
                "raw_any_channel_le_5_fraction",
            ],
            "not_clinical_quality_grade": True,
        },
    }


def metric_dictionary(spec: dict[str, Any]) -> pd.DataFrame:
    rows = []
    for name in spec["primary_continuous_metrics"]:
        rows.append({"metric": name, "category": "primary_continuous", "formula_or_definition": "see spec", "range_or_unit": "continuous Y-derived fraction/score or luminance", "missing_rule": spec["missing_value_rule"]})
    for name, formula in spec["auxiliary_threshold_metrics"].items():
        rows.append({"metric": name, "category": "auxiliary_threshold", "formula_or_definition": formula, "range_or_unit": "fraction [0,1] or count for paired component metrics", "missing_rule": spec["missing_value_rule"]})
    for name in spec["raw_coding_boundary_metrics"]:
        rows.append({"metric": name, "category": "raw_coding_boundary", "formula_or_definition": "computed on raw uint8 pixels inside raw_core_skin", "range_or_unit": "fraction [0,1]", "missing_rule": spec["missing_value_rule"]})
    return pd.DataFrame(rows)


def write_frozen_spec(config: Stage2Config, output_dir: Path, config_path: Path, pilot_manifest_path: Path) -> dict[str, Any]:
    frozen_dir = output_dir / "frozen_spec"
    frozen_dir.mkdir(parents=True, exist_ok=True)
    spec = metric_spec_payload(config)
    yaml_path = frozen_dir / "stage2_lighting_metric_spec_v1.yaml"
    json_path = frozen_dir / "stage2_lighting_metric_spec_v1.json"
    yaml_path.write_text(yaml.safe_dump(json_safe(spec), sort_keys=False, allow_unicode=True), encoding="utf-8")
    write_json(json_path, spec)
    metric_dictionary(spec).to_csv(frozen_dir / "stage2_lighting_metric_dictionary.csv", index=False, encoding="utf-8-sig")
    frozen = {
        "spec_version": SPEC_VERSION,
        "source_code_commit": "not_committed",
        "source_code_hashes": {
            "masks_py": sha256_file(config.project_root / "lighting_qc_stage2" / "masks.py"),
            "affine_mapping_py": sha256_file(config.project_root / "lighting_qc_stage2" / "affine_mapping.py"),
            "frozen_metrics_py": sha256_file(config.project_root / "lighting_qc_stage2" / "frozen_metrics.py"),
        },
        "config_hash": sha256_file(config_path),
        "scheme_b_manifest_hash": sha256_file(config.scheme_b_root / "manifests" / "realface_256x320_manifest.csv"),
        "pilot_manifest_hash": sha256_file(pilot_manifest_path),
        "metric_spec_hash": sha256_file(json_path),
        "roi_definition_hash": roi_definition_hash(config),
        "affine_definition_hash": affine_definition_hash(),
        "frozen": True,
    }
    write_json(frozen_dir / "FROZEN.json", frozen)
    return frozen
