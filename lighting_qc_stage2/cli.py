from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .affine_mapping import canvas_to_source_mask, centroid_distance, mask_iou, source_to_canvas_mask
from .asset_loader import asset_paths, load_exif, load_manifest, load_split, output_dirs, preflight_assets, read_metadata, write_json
from .class_map import load_parsing_class_map, write_class_map
from .config import Stage2Config, json_safe
from .frozen_metrics import auxiliary_metrics, erosion_sensitivity_metrics, primary_metrics, raw_frozen_metrics
from .frozen_spec import affine_definition_hash, roi_definition_hash, write_frozen_spec
from .full500_config_builder import write_full500_config_template
from .full500_pipeline import run_full500
from .image_io import read_gray, read_rgb, read_source_rgb_exif, save_gray
from .masks import build_masks
from .metrics_input_space import assert_monotonic_candidates, input_space_metrics
from .metrics_raw_space import raw_space_metrics
from .pilot_selection import select_pilot32
from .qc_panels import make_contact_sheet, make_qc_panel
from .reporting import manual_review_template, write_report
from .threshold_candidates import consistency, distribution_table, threshold_summary
from .visualization_fixed import make_fixed_contact_sheet, make_qc_panel_fixed
from .r2_pipeline import run_r2


def process_pilot(config: Stage2Config) -> dict[str, Any]:
    if not config.stop_after_pilot or config.pilot_size != 32:
        raise ValueError("Stage2 Pilot requires stop_after_pilot=true and pilot_size=32")
    dirs = output_dirs(config.output_dir)
    log_lines = ["Stage2 Pilot32 started"]
    warnings: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    class_map = load_parsing_class_map(config.project_root)
    write_class_map(dirs["preflight"] / "detected_parsing_class_map.json", class_map)
    preflight = preflight_assets(config, class_map, dirs)
    split = load_split(config)
    exif = load_exif(config)
    manifest = load_manifest(config)
    pilot = select_pilot32(split, exif, manifest, pilot_size=config.pilot_size, seed=config.random_seed)
    pilot["raw_scene_path"] = pilot["sample_id"].map(lambda sid: str(asset_paths(config, sid)["raw_scene"]))
    pilot.to_csv(dirs["pilot"] / "pilot32_manifest.csv", index=False, encoding="utf-8-sig")

    input_rows = []
    raw_rows = []
    roi_rows = []
    roundtrip_rows = []
    panel_paths = []
    for n, row in enumerate(pilot.to_dict("records"), start=1):
        sid = str(row["sample_id"])
        if n > 32:
            raise RuntimeError("attempted to process more than 32 pilot samples")
        paths = asset_paths(config, sid)
        try:
            aligned = read_rgb(paths["aligned_srgb"])
            parsing = read_gray(paths["parsing_label"])
            face = read_gray(paths["face_valid_mask"])
            source_valid = read_gray(paths["source_valid_mask"])
            meta = read_metadata(paths["metadata"])
            raw_rgb, exif_applied = read_source_rgb_exif(paths["raw_scene"])
            masks, diag = build_masks(parsing, face, source_valid, class_map, config.roi_geometry)
            if config.save_intermediate_masks:
                sample_dir = dirs["masks_canvas"] / sid
                for name in ["core_skin_e0", "core_skin_e2", "core_skin_e4", "forehead", "canvas_left_cheek", "canvas_right_cheek", "nose"]:
                    save_gray(sample_dir / f"{name}.png", masks[name])
            input_metric = {"sample_id": sid, **input_space_metrics(aligned, masks, config)}
            assert_monotonic_candidates(input_metric, config.brightness_threshold_candidates, config.darkness_threshold_candidates)
            source_to_canvas = np.asarray(meta["affine_source_to_canvas"], dtype=float)
            canvas_to_source = np.asarray(meta["affine_canvas_to_source"], dtype=float)
            raw_masks = {}
            for cname, rname in [
                ("core_skin_e2", "raw_core_skin"),
                ("forehead", "raw_forehead"),
                ("canvas_left_cheek", "raw_canvas_left_cheek"),
                ("canvas_right_cheek", "raw_canvas_right_cheek"),
                ("nose", "raw_nose"),
            ]:
                raw_masks[rname] = canvas_to_source_mask(masks[cname], (raw_rgb.shape[1], raw_rgb.shape[0]), source_to_canvas)
            if config.save_raw_space_masks:
                sample_raw_dir = dirs["masks_raw"] / sid
                for name, mask in raw_masks.items():
                    save_gray(sample_raw_dir / f"{name}.png", mask)
            roundtrip_core = source_to_canvas_mask(raw_masks["raw_core_skin"], (256, 320), canvas_to_source)
            iou = mask_iou(masks["core_skin_e2"], roundtrip_core)
            cd = centroid_distance(masks["core_skin_e2"], roundtrip_core)
            area_ratio = float((roundtrip_core > 0).sum() / max(1, (masks["core_skin_e2"] > 0).sum()))
            if iou < config.roundtrip_iou_threshold or (cd is not None and cd > config.centroid_distance_threshold):
                raise ValueError(f"roundtrip failed iou={iou} centroid_distance={cd}")
            roundtrip_row = {"sample_id": sid, "roundtrip_iou_core_skin": iou, "roundtrip_centroid_distance_core_skin": cd, "roundtrip_area_ratio_core_skin": area_ratio}
            raw_metric = {"sample_id": sid, **raw_space_metrics(raw_rgb, raw_masks, config)}
            raw_metric = {**raw_metric, "source_width": raw_rgb.shape[1], "source_height": raw_rgb.shape[0], "exif_orientation_applied": bool(exif_applied)}
            for key in ["core_skin_e0", "core_skin_e2", "core_skin_e4", "forehead", "canvas_left_cheek", "canvas_right_cheek", "nose"]:
                pixels = int((masks[key] > 0).sum())
                roi_rows.append(
                    {
                        "sample_id": sid,
                        "roi": key,
                        "pixels": pixels,
                        "bbox_fraction": pixels / max(1, diag["face_width"] * diag["face_height"]),
                        "core_skin_fraction": pixels / max(1, int((masks["core_skin_e2"] > 0).sum())),
                        "valid_ge_64": pixels >= 64,
                        "valid_ge_128": pixels >= 128,
                        "valid_ge_256": pixels >= 256,
                        "status": "valid" if pixels >= 32 else "invalid_too_small",
                    }
                )
            input_rows.append({**row, **diag, **input_metric})
            raw_rows.append({**row, **raw_metric})
            roundtrip_rows.append(roundtrip_row)
            if config.generate_qc_panels:
                panel_path = dirs["qc_panels"] / f"{sid}_stage2_pilot.png"
                make_qc_panel(panel_path, sid, raw_rgb, aligned, parsing, masks, {**raw_masks, "roundtrip_core_skin": roundtrip_core}, input_metric, {**roundtrip_row, "roundtrip_core_skin": roundtrip_core})
                panel_paths.append(panel_path)
        except Exception as exc:
            failures.append({"sample_id": sid, "failure": f"{type(exc).__name__}: {exc}"})
            raise

    input_df = pd.DataFrame(input_rows)
    raw_df = pd.DataFrame(raw_rows)
    roi_df = pd.DataFrame(roi_rows)
    roundtrip_df = pd.DataFrame(roundtrip_rows)
    input_df.to_csv(dirs["metrics"] / "pilot32_continuous_metrics.csv", index=False, encoding="utf-8-sig")
    roi_df.to_csv(dirs["metrics"] / "pilot32_roi_validity.csv", index=False, encoding="utf-8-sig")
    raw_df.to_csv(dirs["metrics"] / "pilot32_raw_space_metrics.csv", index=False, encoding="utf-8-sig")
    detail, consist_summary = consistency(input_df, raw_df)
    detail.to_csv(dirs["metrics"] / "pilot32_raw_input_consistency.csv", index=False, encoding="utf-8-sig")
    consist_summary.to_csv(dirs["metrics"] / "pilot32_consistency_summary.csv", index=False, encoding="utf-8-sig")
    dist = distribution_table(input_df)
    dist.to_csv(dirs["metrics"] / "pilot32_metric_distributions.csv", index=False, encoding="utf-8-sig")
    threshold_summary(input_df, raw_df).to_csv(dirs["reports"] / "threshold_candidate_summary.csv", index=False, encoding="utf-8-sig")
    extreme_cols = ["skin_y_median", "bright_fraction_y_ge_0.80", "dark_fraction_y_le_0.05", "input_shadow_ratio_0.65_fraction"]
    extreme = input_df[["sample_id", "selection_reason", *[c for c in extreme_cols if c in input_df.columns]]].copy()
    extreme.to_csv(dirs["reports"] / "pilot32_extreme_cases.csv", index=False, encoding="utf-8-sig")
    manual_review_template(pilot).to_csv(dirs["reports"] / "pilot32_manual_review_template.csv", index=False, encoding="utf-8-sig")
    contact_names = [
        "core_skin_erosion_comparison.png",
        "bright_threshold_comparison.png",
        "dark_threshold_comparison.png",
        "shadow_ratio_comparison.png",
        "specular_threshold_comparison.png",
        "raw_input_consistency_extremes.png",
        "roi_validity_overview.png",
    ]
    for name in contact_names:
        make_contact_sheet(panel_paths, dirs["contact_sheets"] / name)
    (dirs["logs"] / "failures.csv").write_text(pd.DataFrame(failures).to_csv(index=False), encoding="utf-8")
    pd.DataFrame(warnings).to_csv(dirs["logs"] / "warnings.csv", index=False, encoding="utf-8-sig")
    (dirs["logs"] / "run.log").write_text("\n".join(log_lines + ["Stage2 Pilot32 completed; stopped before Full-500"]), encoding="utf-8")
    summary = {
        "experiment_name": config.experiment_name,
        "inputs": {
            "scheme_b_root": str(config.scheme_b_root),
            "raw_scene_dir": str(config.raw_scene_dir),
            "split_csv": str(config.split_csv),
            "exif_xlsx": str(config.exif_xlsx),
            "stage1_master_csv": str(config.stage1_master_csv),
            "rgb_oof_csv": str(config.rgb_oof_csv),
        },
        "preflight": preflight,
        "pilot": {"n": int(len(pilot)), "folds": sorted(pilot["fold"].astype(int).unique().tolist()), "label_counts": pilot["binary_label"].value_counts().sort_index().to_dict(), "ids": pilot["sample_id"].tolist()},
        "class_map": class_map,
        "roi": {
            "invalid_roi_rows": int((roi_df["status"] != "valid").sum()),
            "min_core_skin_e2_pixels": int(roi_df.loc[roi_df["roi"] == "core_skin_e2", "pixels"].min()),
            "min_roi_pixels": int(roi_df["pixels"].min()),
        },
        "roundtrip": {
            "min_iou": float(roundtrip_df["roundtrip_iou_core_skin"].min()),
            "max_centroid_distance": float(roundtrip_df["roundtrip_centroid_distance_core_skin"].max()),
            "all_passed": bool((roundtrip_df["roundtrip_iou_core_skin"] >= config.roundtrip_iou_threshold).all()),
        },
        "stop_after_pilot": True,
        "full_500_run": False,
        "thresholds_frozen": False,
    }
    write_report(config.output_dir, summary)
    return summary


def _copy_r1_preflight_files(dirs: dict[str, Path]) -> None:
    src_summary = dirs["preflight"] / "preflight_summary.json"
    src_report = dirs["preflight"] / "preflight_report.md"
    if src_summary.is_file():
        shutil.copyfile(src_summary, dirs["preflight"] / "r1_preflight_summary.json")
    if src_report.is_file():
        text = src_report.read_text(encoding="utf-8").replace("Stage2 Pilot32 Preflight", "Stage2 R1 Pilot32 Preflight")
        (dirs["preflight"] / "r1_preflight_report.md").write_text(text, encoding="utf-8")


def _strict_shape_inventory(config: Stage2Config, pilot: pd.DataFrame, dirs: dict[str, Path]) -> pd.DataFrame:
    rows = []
    for sid in pilot["sample_id"].astype(str).tolist():
        paths = asset_paths(config, sid)
        aligned = read_rgb(paths["aligned_srgb"])
        image = read_rgb(paths["images"])
        parsing = read_gray(paths["parsing_label"])
        face = read_gray(paths["face_valid_mask"])
        source = read_gray(paths["source_valid_mask"])
        row = {
            "sample_id": sid,
            "aligned_srgb_shape": str(tuple(aligned.shape)),
            "images_shape": str(tuple(image.shape)),
            "parsing_label_shape": str(tuple(parsing.shape)),
            "face_valid_mask_shape": str(tuple(face.shape)),
            "source_valid_mask_shape": str(tuple(source.shape)),
            "shape_ok": bool(
                aligned.shape == (320, 256, 3)
                and image.shape == (320, 256, 3)
                and parsing.shape == (320, 256)
                and face.shape == (320, 256)
                and source.shape == (320, 256)
            ),
        }
        rows.append(row)
    inventory = pd.DataFrame(rows)
    inventory.to_csv(dirs["preflight"] / "image_shape_inventory.csv", index=False, encoding="utf-8-sig")
    if not bool(inventory["shape_ok"].all()):
        raise ValueError("one or more Pilot32 Scheme B arrays are not shape (320,256)")
    return inventory


def _roi_rows(sample_id: str, masks: dict[str, np.ndarray], diag: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    core_pixels = int((masks["core_skin_e2"] > 0).sum())
    for key in ["core_skin_e0", "core_skin_e2", "core_skin_e4", "forehead", "canvas_left_cheek", "canvas_right_cheek", "nose"]:
        pixels = int((masks[key] > 0).sum())
        rows.append(
            {
                "sample_id": sample_id,
                "roi": key,
                "pixels": pixels,
                "bbox_fraction": pixels / max(1, diag["face_width"] * diag["face_height"]),
                "core_skin_fraction": pixels / max(1, core_pixels),
                "valid_ge_64": pixels >= 64,
                "valid_ge_128": pixels >= 128,
                "valid_ge_256": pixels >= 256,
                "status": "valid" if pixels >= 32 else "invalid_too_small",
            }
        )
    return rows


def _compare_old_and_r1(config: Stage2Config, r1_primary: pd.DataFrame, r1_raw: pd.DataFrame, dirs: dict[str, Path]) -> dict[str, Any]:
    old_dir = config.project_root / "experiments" / "lighting_confounding" / "Lighting_QC_Stage2_Pilot32_v1"
    old_input = pd.read_csv(old_dir / "metrics" / "pilot32_continuous_metrics.csv")
    old_raw = pd.read_csv(old_dir / "metrics" / "pilot32_raw_space_metrics.csv")
    old = old_input.merge(old_raw, on="sample_id", how="outer", suffixes=("", "_rawtable"))
    new = r1_primary.merge(r1_raw, on="sample_id", how="outer")
    metric_cols = []
    protected_prefixes = ("skin_y_", "forehead_y_", "canvas_left_cheek_y_", "canvas_right_cheek_y_", "nose_y_", "cheek_", "mean_cheek_y", "raw_")
    for col in new.columns:
        if col == "sample_id" or col not in old.columns:
            continue
        if col.startswith(protected_prefixes):
            metric_cols.append(col)
    rows = []
    merged = old[["sample_id", *metric_cols]].merge(new[["sample_id", *metric_cols]], on="sample_id", suffixes=("_old", "_r1"))
    for col in metric_cols:
        old_v = pd.to_numeric(merged[f"{col}_old"], errors="coerce")
        new_v = pd.to_numeric(merged[f"{col}_r1"], errors="coerce")
        diff = (old_v - new_v).abs()
        rows.append(
            {
                "metric": col,
                "valid_n": int((old_v.notna() & new_v.notna()).sum()),
                "max_abs_difference": float(diff.max(skipna=True)) if diff.notna().any() else None,
                "all_close_1e_10": bool((diff.fillna(0) <= 1e-10).all()),
            }
        )
    comp = pd.DataFrame(rows)
    comp.to_csv(dirs["comparison"] / "old_vs_r1_metric_comparison.csv", index=False, encoding="utf-8-sig")
    summary = {
        "compared_metric_count": int(len(comp)),
        "all_unchanged_metrics_close": bool(comp["all_close_1e_10"].all()) if len(comp) else True,
        "max_abs_difference_overall": float(comp["max_abs_difference"].max()) if len(comp) and comp["max_abs_difference"].notna().any() else 0.0,
        "tolerance": 1e-10,
        "note": "Only metrics whose definitions were not changed and are present in both old Pilot and R1 tables were compared.",
    }
    write_json(dirs["comparison"] / "unchanged_metric_consistency.json", summary)
    return summary


def _write_r1_reports(config: Stage2Config, summary: dict[str, Any], dirs: dict[str, Path]) -> None:
    write_json(dirs["reports"] / "stage2_r1_machine_summary.json", summary)
    lines = [
        "# Lighting QC Stage2 R1 Pilot32 Report",
        "",
        "## Purpose",
        "R1 fixes QC-panel display geometry, freezes already reviewed skin/ROI/raw mapping definitions, and switches Stage2 toward continuous lighting metrics plus a compact auxiliary threshold set.",
        "",
        "## QC Display Fix",
        "The old Pilot panels resized all images to a 256x200 cell, which horizontally stretched 320x256 canvas images. R1 uses letterboxed rendering with shared geometry for RGB, mask, heatmap, and overlays.",
        "",
        "## Frozen Definitions",
        f"- roi_definition_hash: `{summary['frozen']['roi_definition_hash']}`",
        f"- affine_definition_hash: `{summary['frozen']['affine_definition_hash']}`",
        "- primary erosion: 2 px",
        "",
        "## Pilot32",
        f"- reused_old_pilot_manifest: {summary['pilot']['reused_old_pilot_manifest']}",
        f"- pilot_n: {summary['pilot']['n']}",
        f"- folds: {summary['pilot']['folds']}",
        "",
        "## Shape Gate",
        f"- aligned/image/mask/parser shape ok: {summary['shape_gate']['all_ok']}",
        "- expected canvas shape: aligned/images (320,256,3); masks/parser (320,256)",
        "",
        "## Metrics",
        "Primary continuous metrics are in `metrics/pilot32_frozen_primary_metrics.csv`; auxiliary metrics are in `metrics/pilot32_auxiliary_metrics.csv`; raw coding-boundary metrics are in `metrics/pilot32_raw_metrics.csv`.",
        "",
        "## Old vs R1 Consistency",
        f"- all unchanged metrics close: {summary['comparison']['all_unchanged_metrics_close']}",
        f"- max abs difference: {summary['comparison']['max_abs_difference_overall']}",
        "",
        "## Full-500 Readiness",
        "Frozen metric spec and Full-500 config template were generated. The Full-500 config was not executed in this run.",
        "",
        "## Explicit Non-actions",
        "No Full-500 run, no RGB model training, no sample deletion, and no A/B/C/D quality grades were produced.",
    ]
    (dirs["reports"] / "stage2_r1_pilot32_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    checklist = [
        "# Full-500 Readiness Checklist",
        "",
        "- [x] Pilot32 IDs reused exactly",
        "- [x] Scheme B assets passed coverage preflight",
        "- [x] Canvas shape gate passed for Pilot32",
        "- [x] ROI and affine definitions frozen by hash",
        "- [x] Primary continuous metrics generated",
        "- [x] Auxiliary threshold metrics reduced to severe bright/dark, ratio 0.55 relative dark-region candidate, and Y0.75/chroma0.08 low-chroma high-luminance candidate",
        "- [x] Frozen metric spec generated",
        "- [x] Full-500 config template generated",
        "- [x] Full-500 not executed",
        "",
        "Conclusion: ready for human review of R1 Pilot32 outputs before any Full-500 execution.",
    ]
    (dirs["reports"] / "full500_readiness_checklist.md").write_text("\n".join(checklist) + "\n", encoding="utf-8")


def process_r1_pilot(config: Stage2Config, config_path: Path) -> dict[str, Any]:
    if config.mode != "r1_pilot32":
        raise ValueError("R1 processing requires mode: r1_pilot32")
    if not config.stop_after_pilot or config.pilot_size != 32:
        raise ValueError("R1 Pilot requires stop_after_pilot=true and pilot_size=32")
    if config.old_pilot_manifest_csv is None:
        raise ValueError("R1 requires old_pilot_manifest_csv")
    dirs = output_dirs(config.output_dir)
    class_map = load_parsing_class_map(config.project_root)
    write_class_map(dirs["preflight"] / "detected_parsing_class_map.json", class_map)
    preflight = preflight_assets(config, class_map, dirs)
    _copy_r1_preflight_files(dirs)
    pilot = pd.read_csv(config.old_pilot_manifest_csv, dtype={"sample_id": str})
    if len(pilot) != 32 or pilot["sample_id"].nunique() != 32:
        raise ValueError("old Pilot manifest must contain 32 unique sample IDs")
    pilot.to_csv(dirs["pilot"] / "pilot32_manifest.csv", index=False, encoding="utf-8-sig")
    shape_inventory = _strict_shape_inventory(config, pilot, dirs)

    primary_rows: list[dict[str, Any]] = []
    aux_rows: list[dict[str, Any]] = []
    raw_rows: list[dict[str, Any]] = []
    roi_rows: list[dict[str, Any]] = []
    roundtrip_rows: list[dict[str, Any]] = []
    sensitivity_rows: list[dict[str, Any]] = []
    panel_paths: list[Path] = []
    failures: list[dict[str, Any]] = []

    for row in pilot.to_dict("records"):
        sid = str(row["sample_id"])
        paths = asset_paths(config, sid)
        try:
            aligned = read_rgb(paths["aligned_srgb"])
            parsing = read_gray(paths["parsing_label"])
            face = read_gray(paths["face_valid_mask"])
            source_valid = read_gray(paths["source_valid_mask"])
            meta = read_metadata(paths["metadata"])
            raw_rgb, exif_applied = read_source_rgb_exif(paths["raw_scene"])
            masks, diag = build_masks(parsing, face, source_valid, class_map, config.roi_geometry)
            if config.save_intermediate_masks:
                sample_dir = dirs["masks_canvas"] / sid
                for name in ["core_skin_e0", "core_skin_e2", "core_skin_e4", "forehead", "canvas_left_cheek", "canvas_right_cheek", "nose"]:
                    save_gray(sample_dir / f"{name}.png", masks[name])
            source_to_canvas = np.asarray(meta["affine_source_to_canvas"], dtype=float)
            canvas_to_source = np.asarray(meta["affine_canvas_to_source"], dtype=float)
            raw_masks = {}
            for cname, rname in [
                ("core_skin_e2", "raw_core_skin"),
                ("forehead", "raw_forehead"),
                ("canvas_left_cheek", "raw_canvas_left_cheek"),
                ("canvas_right_cheek", "raw_canvas_right_cheek"),
                ("nose", "raw_nose"),
            ]:
                raw_masks[rname] = canvas_to_source_mask(masks[cname], (raw_rgb.shape[1], raw_rgb.shape[0]), source_to_canvas)
            if config.save_raw_space_masks:
                sample_raw_dir = dirs["masks_raw"] / sid
                for name, mask in raw_masks.items():
                    save_gray(sample_raw_dir / f"{name}.png", mask)
            roundtrip_core = source_to_canvas_mask(raw_masks["raw_core_skin"], (aligned.shape[1], aligned.shape[0]), canvas_to_source)
            iou = mask_iou(masks["core_skin_e2"], roundtrip_core)
            cd = centroid_distance(masks["core_skin_e2"], roundtrip_core)
            area_ratio = float((roundtrip_core > 0).sum() / max(1, (masks["core_skin_e2"] > 0).sum()))
            if iou < config.roundtrip_iou_threshold or (cd is not None and cd > config.centroid_distance_threshold):
                raise ValueError(f"roundtrip failed iou={iou} centroid_distance={cd}")
            primary = {"sample_id": sid, **primary_metrics(aligned, masks)}
            aux = {"sample_id": sid, **auxiliary_metrics(aligned, masks, config.shadow_gaussian_sigma)}
            raw = {"sample_id": sid, **raw_frozen_metrics(raw_rgb, raw_masks), "source_width": raw_rgb.shape[1], "source_height": raw_rgb.shape[0], "exif_orientation_applied": bool(exif_applied)}
            sensitivity = {"sample_id": sid, **erosion_sensitivity_metrics(aligned, masks)}
            roundtrip = {"sample_id": sid, "roundtrip_iou_core_skin": iou, "roundtrip_centroid_distance_core_skin": cd, "roundtrip_area_ratio_core_skin": area_ratio}
            primary_rows.append(primary)
            aux_rows.append(aux)
            raw_rows.append(raw)
            sensitivity_rows.append(sensitivity)
            roi_rows.extend(_roi_rows(sid, masks, diag))
            roundtrip_rows.append(roundtrip)
            if config.generate_qc_panels:
                panel_path = dirs["qc_panels_fixed"] / f"{sid}_stage2_r1.png"
                make_qc_panel_fixed(panel_path, sid, raw_rgb, aligned, parsing, masks, {**raw_masks, "roundtrip_core_skin": roundtrip_core}, {**primary, **aux}, {**roundtrip, "roundtrip_core_skin": roundtrip_core})
                panel_paths.append(panel_path)
        except Exception as exc:
            failures.append({"sample_id": sid, "failure": f"{type(exc).__name__}: {exc}"})
            raise

    primary_df = pd.DataFrame(primary_rows)
    aux_df = pd.DataFrame(aux_rows)
    raw_df = pd.DataFrame(raw_rows)
    roi_df = pd.DataFrame(roi_rows)
    roundtrip_df = pd.DataFrame(roundtrip_rows)
    sensitivity_df = pd.DataFrame(sensitivity_rows)
    primary_df.to_csv(dirs["metrics"] / "pilot32_frozen_primary_metrics.csv", index=False, encoding="utf-8-sig")
    aux_df.to_csv(dirs["metrics"] / "pilot32_auxiliary_metrics.csv", index=False, encoding="utf-8-sig")
    raw_df.to_csv(dirs["metrics"] / "pilot32_raw_metrics.csv", index=False, encoding="utf-8-sig")
    roi_df.to_csv(dirs["metrics"] / "pilot32_roi_validity.csv", index=False, encoding="utf-8-sig")
    roundtrip_df.to_csv(dirs["metrics"] / "pilot32_roundtrip_metrics.csv", index=False, encoding="utf-8-sig")
    sensitivity_df.to_csv(dirs["metrics_sensitivity"] / "pilot32_erosion_sensitivity.csv", index=False, encoding="utf-8-sig")
    comparison = _compare_old_and_r1(config, primary_df, raw_df, dirs)
    frozen = write_frozen_spec(config, config.output_dir, config_path, dirs["pilot"] / "pilot32_manifest.csv")
    full500_config = write_full500_config_template(config, config.project_root, dirs["frozen_spec"] / "stage2_lighting_metric_spec_v1.yaml")
    for group in ["roi_review", "luminance_review", "auxiliary_flags_review"]:
        for idx in range(4):
            make_fixed_contact_sheet(panel_paths, dirs["contact_sheets_fixed"] / f"{group}_{idx + 1:02d}.png", start=idx * 8, count=8)
    pd.DataFrame(failures).to_csv(dirs["logs"] / "failures.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame([]).to_csv(dirs["logs"] / "warnings.csv", index=False, encoding="utf-8-sig")
    (dirs["logs"] / "run.log").write_text("Stage2 R1 Pilot32 completed; Full-500 config generated but not executed.\n", encoding="utf-8")
    summary = {
        "experiment_name": config.experiment_name,
        "inputs": {
            "scheme_b_root": str(config.scheme_b_root),
            "old_pilot_manifest_csv": str(config.old_pilot_manifest_csv),
            "output_dir": str(config.output_dir),
        },
        "preflight": preflight,
        "pilot": {
            "n": int(len(pilot)),
            "ids": pilot["sample_id"].astype(str).tolist(),
            "folds": sorted(pd.to_numeric(pilot["fold"]).astype(int).unique().tolist()) if "fold" in pilot else [],
            "reused_old_pilot_manifest": True,
        },
        "shape_gate": {
            "all_ok": bool(shape_inventory["shape_ok"].all()),
            "expected_aligned_shape": [320, 256, 3],
            "checked_n": int(len(shape_inventory)),
        },
        "roi": {
            "invalid_roi_rows": int((roi_df["status"] != "valid").sum()),
            "min_core_skin_e2_pixels": int(roi_df.loc[roi_df["roi"] == "core_skin_e2", "pixels"].min()),
            "min_roi_pixels": int(roi_df["pixels"].min()),
            "primary_skin_erosion_px": 2,
        },
        "roundtrip": {
            "min_iou": float(roundtrip_df["roundtrip_iou_core_skin"].min()),
            "max_centroid_distance": float(roundtrip_df["roundtrip_centroid_distance_core_skin"].max()),
            "all_passed": bool((roundtrip_df["roundtrip_iou_core_skin"] >= config.roundtrip_iou_threshold).all()),
        },
        "comparison": comparison,
        "frozen": frozen,
        "outputs": {
            "fixed_qc_panels": len(panel_paths),
            "fixed_contact_sheets": 12,
            "full500_config_path": str(full500_config),
        },
        "stop_after_pilot": True,
        "full_500_run": False,
        "model_training_run": False,
        "samples_removed": False,
        "quality_grades_generated": False,
    }
    _write_r1_reports(config, summary, dirs)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    config_path = Path(args.config)
    config = Stage2Config.from_yaml(config_path)
    if config.mode == "full500":
        summary = run_full500(config)
    elif config.mode == "stage2_r2":
        summary = run_r2(config)
    else:
        summary = process_r1_pilot(config, config_path) if config.mode == "r1_pilot32" else process_pilot(config)
    print(json.dumps(json_safe(summary), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
