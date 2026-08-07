from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .asset_loader import load_split, write_json
from .config import Stage2Config
from .frozen_spec import affine_definition_hash, roi_definition_hash, sha256_file
from .r2_exif_decomposition import TARGET_METRICS, run_exif_decomposition
from .r2_figures import write_r2_figures
from .r2_reporting import markdown_table, write_r2_reports
from .r2_statistical_repair import run_statistical_repair


def r2_dirs(output_dir: Path) -> dict[str, Path]:
    dirs = {name: output_dir / name for name in ["preflight", "repair", "exif", "figures", "reports", "logs"]}
    for path in dirs.values():
        path.mkdir(parents=True, exist_ok=True)
    return dirs


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def r2_preflight(config: Stage2Config, dirs: dict[str, Path]) -> dict[str, Any]:
    if config.full500_output_dir is None:
        raise ValueError("R2 requires full500_output_dir")
    full = config.full500_output_dir
    features = pd.read_csv(full / "features" / "stage2_lighting_features_500.csv", dtype={"sample_id": str, "patient_group_id": str})
    e0 = pd.read_csv(full / "features" / "stage2_lighting_features_erosion0.csv", dtype={"sample_id": str})
    e4 = pd.read_csv(full / "features" / "stage2_lighting_features_erosion4.csv", dtype={"sample_id": str})
    split = load_split(config)
    stage1 = pd.read_csv(config.stage1_master_csv, dtype={"sample_id": str, "patient_group_id": str})
    frozen_path = config.metric_spec_path.parent / "FROZEN.json" if config.metric_spec_path else Path("")
    frozen = _load_json(frozen_path)
    spec_json = config.metric_spec_path.with_suffix(".json") if config.metric_spec_path else Path("")
    integrity = {
        "frozen_integrity_pass": bool(frozen.get("frozen") is True),
        "metric_spec_hash_matches": sha256_file(spec_json) == frozen.get("metric_spec_hash"),
        "roi_definition_hash_matches": roi_definition_hash(config) == frozen.get("roi_definition_hash"),
        "affine_definition_hash_matches": affine_definition_hash() == frozen.get("affine_definition_hash"),
    }
    merged = features[["sample_id", "patient_group_id", "fold", "binary_label"]].merge(split[["sample_id", "patient_group_id", "fold", "binary_label"]], on="sample_id", suffixes=("_features", "_split"), how="outer")
    merged["label_match"] = merged["binary_label_features"].astype(str) == merged["binary_label_split"].astype(str)
    merged["fold_match"] = merged["fold_features"].astype(str) == merged["fold_split"].astype(str)
    merged["patient_group_match"] = merged["patient_group_id_features"].astype(str) == merged["patient_group_id_split"].astype(str)
    stage1_ids = set(stage1["sample_id"].astype(str))
    merged["stage1_present"] = merged["sample_id"].astype(str).isin(stage1_ids)
    merged.to_csv(dirs["preflight"] / "r2_id_alignment_audit.csv", index=False, encoding="utf-8-sig")
    exif_cols = [c for c in ["exposure_time_seconds", "iso", "brightness_value_apex", "log2_exposure_time", "log2_iso"] if c in stage1.columns]
    exif_coverage = pd.DataFrame([{"field": c, "non_missing": int(pd.to_numeric(stage1[c], errors="coerce").notna().sum()), "missing": int(pd.to_numeric(stage1[c], errors="coerce").isna().sum())} for c in exif_cols])
    exif_coverage.to_csv(dirs["preflight"] / "r2_exif_coverage_audit.csv", index=False, encoding="utf-8-sig")
    inventory = {
        "full500_root": str(full),
        "feature_table": str(full / "features" / "stage2_lighting_features_500.csv"),
        "erosion0_table": str(full / "features" / "stage2_lighting_features_erosion0.csv"),
        "erosion4_table": str(full / "features" / "stage2_lighting_features_erosion4.csv"),
        "stage1_master": str(config.stage1_master_csv),
        "split_csv": str(config.split_csv),
        "frozen_json": str(frozen_path),
        "metric_spec": str(config.metric_spec_path),
    }
    write_json(dirs["preflight"] / "r2_input_inventory.json", inventory)
    summary = {
        **integrity,
        "features_rows": int(len(features)),
        "features_unique_ids": int(features["sample_id"].nunique()),
        "patient_group_complete": int(features["patient_group_id"].notna().sum()),
        "fold_complete": int(features["fold"].notna().sum()),
        "binary_label_complete": int(features["binary_label"].notna().sum()),
        "rgb_oof_probability_complete": int(features["rgb_oof_probability_patient"].notna().sum()),
        "erosion0_rows": int(len(e0)),
        "erosion4_rows": int(len(e4)),
        "erosion0_id_match": set(e0["sample_id"].astype(str)) == set(features["sample_id"].astype(str)),
        "erosion4_id_match": set(e4["sample_id"].astype(str)) == set(features["sample_id"].astype(str)),
        "stage1_alignment": int(merged["stage1_present"].sum()),
        "label_conflicts": int((~merged["label_match"]).sum()),
        "fold_conflicts": int((~merged["fold_match"]).sum()),
        "patient_group_cross_fold": int(features.groupby("patient_group_id")["fold"].nunique().gt(1).sum()),
        "reextracted_image_features": False,
        "traditional_exif_covariate_adjustment": bool(config.run_traditional_exif_covariate_adjustment),
        "run_stage3": bool(config.run_stage3),
    }
    summary["pass"] = bool(
        all(integrity.values())
        and summary["features_rows"] == 500
        and summary["features_unique_ids"] == 500
        and summary["rgb_oof_probability_complete"] == 500
        and summary["erosion0_id_match"]
        and summary["erosion4_id_match"]
        and summary["stage1_alignment"] == 500
        and summary["label_conflicts"] == 0
        and summary["fold_conflicts"] == 0
        and summary["patient_group_cross_fold"] == 0
        and not summary["traditional_exif_covariate_adjustment"]
        and not summary["run_stage3"]
    )
    write_json(dirs["preflight"] / "r2_preflight_summary.json", summary)
    lines = ["# R2 Preflight", "", *[f"- {k}: {v}" for k, v in summary.items()]]
    (dirs["preflight"] / "r2_preflight_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    if not summary["pass"]:
        raise ValueError(f"R2 preflight failed: {summary}")
    return summary


def run_r2(config: Stage2Config) -> dict[str, Any]:
    if config.mode != "stage2_r2":
        raise ValueError("run_r2 requires mode: stage2_r2")
    if config.run_traditional_exif_covariate_adjustment or config.run_stage3:
        raise ValueError("R2 forbids traditional EXIF covariate adjustment and Stage3 execution")
    dirs = r2_dirs(config.output_dir)
    preflight = r2_preflight(config, dirs)
    full = config.full500_output_dir
    features = pd.read_csv(full / "features" / "stage2_lighting_features_500.csv", dtype={"sample_id": str, "patient_group_id": str})
    labels = features[["sample_id", "patient_group_id", "fold", "binary_label", "rgb_oof_probability_patient"]].copy()
    repair = run_statistical_repair(full, labels, config.output_dir, iterations=config.bootstrap_iterations, seed=config.random_seed)
    stage1 = pd.read_csv(config.stage1_master_csv, dtype={"sample_id": str, "patient_group_id": str})
    exif = run_exif_decomposition(features, stage1, config.output_dir, iterations=config.bootstrap_iterations, seed=config.random_seed)
    write_r2_figures(config.output_dir)
    worst = repair["worst"]
    revised = repair["revised_evidence"]
    stable = worst[worst["scope"] == "stable_only"].copy()
    all_scope = worst[worst["scope"] == "all_groups"].copy()
    fdr = repair["fdr"]
    perf = exif["performance"]
    performance_markdown = markdown_table(perf, ["target_metric", "r2", "mae", "rmse", "spearman_r"])
    exif_summary = {
        "performance_markdown": performance_markdown,
        "model_performance": perf.to_dict("records"),
        "interpretation": exif["interpretation"],
        "label_fdr": exif["label_fdr"],
        "rgb_logit_fdr": exif["logit_fdr"],
        "error_risk_fdr": exif["error_fdr"],
    }
    stage3 = _stage3_recommendation(revised, exif["interpretation"])
    repair_summary = {
        "all_group_max_delta_auc": float(pd.to_numeric(all_scope["DeltaAUC"], errors="coerce").max()),
        "stable_only_max_delta_auc": revised["stable_only_max_delta_auc"],
        "stable_only_max_delta_auc_ci": revised["stable_only_max_delta_auc_ci"],
        "dependence_term": revised["dependence_term"],
        "dependence_level": revised["dependence_level"],
        "rgb_logit_single_sig": fdr["rgb_logit_single_metric_models"]["significant_unique_metrics"],
        "rgb_error_single_sig": fdr["rgb_error_single_metric_models"]["significant_unique_metrics"],
        "rgb_error_multi_sig": fdr["rgb_error_multivariable_model"]["significant_unique_metrics"],
    }
    summary = {
        "experiment_name": config.experiment_name,
        "preflight": preflight,
        "repair": repair_summary,
        "exif": exif_summary,
        "stage3_recommendation": stage3,
        "traditional_exif_covariate_adjustment_performed": False,
        "rgb_model_trained_or_modified": False,
        "stage3_executed": False,
        "samples_deleted": False,
        "image_features_reextracted": False,
        "bootstrap_iterations": config.bootstrap_iterations,
        "random_seed": config.random_seed,
    }
    write_r2_reports(config.output_dir, summary)
    pd.DataFrame([]).to_csv(dirs["logs"] / "warnings.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame([]).to_csv(dirs["logs"] / "failures.csv", index=False, encoding="utf-8-sig")
    (dirs["logs"] / "run.log").write_text("Stage2 R2 completed; no traditional EXIF covariate adjustment, no RGB training, no Stage3.\n", encoding="utf-8")
    return summary


def _stage3_recommendation(revised: dict[str, Any], interpretation: dict[str, Any]) -> str:
    categories = {v["interpretation"] for v in interpretation.values()}
    if revised["dependence_level"] == "high":
        if "limited_exif_explanation" in categories or "mixed_recorded_exposure_and_unexplained_component" in categories:
            return "Proceed to Stage3, prioritizing constrained exposure/gamma augmentation plus local luminance/appearance perturbations; interpret EXIF residuals only as variation unexplained by the recorded EXIF exposure fields."
        return "Proceed to Stage3 with constrained exposure/gamma/white-balance augmentation focused on recorded-exposure-related luminance sensitivity."
    if revised["dependence_level"] == "moderate":
        return "Stage3 should run lightweight constrained augmentation and monitor stable-only WorstGroupAUC."
    return "Stage3 may be limited to stability validation unless downstream review requires stronger stress testing."
