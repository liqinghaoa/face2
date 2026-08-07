from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from scipy import stats
from sklearn.metrics import brier_score_loss

from lighting_confounding_stage1.metrics_utils import cluster_bootstrap_metrics, cluster_bootstrap_smd, compute_metrics, smd

from .affine_mapping import canvas_to_source_mask, centroid_distance, mask_iou, source_to_canvas_mask
from .asset_loader import asset_paths, load_manifest, load_split, preflight_assets, read_metadata, write_json
from .class_map import load_parsing_class_map, write_class_map
from .config import Stage2Config
from .frozen_metrics import auxiliary_metrics, erosion_sensitivity_metrics, primary_metrics, raw_frozen_metrics
from .frozen_spec import affine_definition_hash, roi_definition_hash, sha256_file
from .image_io import read_gray, read_rgb, read_source_rgb_exif
from .masks import build_masks


PRIMARY_METRICS = ["skin_y_median", "dark_contrast_score", "cheek_relative_difference", "skin_y_p99"]
STRATIFICATION_METRICS = [
    "skin_y_median",
    "skin_y_p05",
    "skin_y_p99",
    "skin_y_iqr",
    "dark_contrast_score",
    "cheek_relative_difference",
    "forehead_cheek_absolute_difference",
    "raw_any_channel_ge_250_fraction",
    "raw_any_channel_le_5_fraction",
]
SENSITIVITY_METRICS = ["skin_y_median", "skin_y_p05", "skin_y_p99", "skin_y_iqr", "dark_contrast_score", "cheek_relative_difference"]
SECONDARY_INPUT_METRICS = [
    "skin_y_p05",
    "skin_y_p10",
    "skin_y_p95",
    "skin_y_iqr",
    "skin_y_p90_minus_p10",
    "deep_dark_contrast_score",
    "forehead_cheek_absolute_difference",
    "nose_y_p99",
    "forehead_y_p99",
]
BOOTSTRAP_REPEATS = 2000
SEED = 2026


def full500_dirs(output_dir: Path) -> dict[str, Path]:
    names = ["preflight", "features", "association", "rgb_dependence", "stratified", "sensitivity", "figures", "reports", "logs"]
    dirs = {name: output_dir / name for name in names}
    for path in dirs.values():
        path.mkdir(parents=True, exist_ok=True)
    return dirs


def _bh_fdr(p_values: Iterable[float | None]) -> list[float | None]:
    vals = np.asarray([np.nan if p is None else float(p) for p in p_values], dtype=float)
    out = np.full(vals.shape, np.nan, dtype=float)
    ok = np.isfinite(vals)
    if not ok.any():
        return [None for _ in vals]
    p = vals[ok]
    order = np.argsort(p)
    ranked = p[order]
    m = len(ranked)
    q = ranked * m / np.arange(1, m + 1)
    q = np.minimum.accumulate(q[::-1])[::-1]
    q = np.clip(q, 0, 1)
    temp = np.empty_like(q)
    temp[order] = q
    out[ok] = temp
    return [None if not np.isfinite(x) else float(x) for x in out]


def _z(series: pd.Series) -> pd.Series:
    x = pd.to_numeric(series, errors="coerce").astype(float)
    sd = x.std(ddof=1)
    if not np.isfinite(sd) or sd == 0:
        return pd.Series(np.nan, index=series.index)
    return (x - x.mean()) / sd


def _clip_logit(prob: pd.Series) -> pd.Series:
    p = pd.to_numeric(prob, errors="coerce").clip(1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def verify_frozen_integrity(config: Stage2Config, dirs: dict[str, Path]) -> dict[str, Any]:
    if config.metric_spec_path is None:
        raise ValueError("Full-500 requires metric_spec_path")
    frozen_path = config.metric_spec_path.parent / "FROZEN.json"
    spec_json_path = config.metric_spec_path.with_suffix(".json")
    frozen = json.loads(frozen_path.read_text(encoding="utf-8"))
    config_text = Path("configs/lighting_qc_stage2_full500_v1.yaml")
    checks = {
        "frozen_json_exists": frozen_path.is_file(),
        "frozen_true": bool(frozen.get("frozen") is True),
        "metric_spec_hash_matches": sha256_file(spec_json_path) == frozen.get("metric_spec_hash"),
        "roi_definition_hash_matches": roi_definition_hash(config) == frozen.get("roi_definition_hash") == "19517dd90a7becffd8ac5f6e4647f7f1de8e77a61bc3e24d2c0440e93dc1a3db",
        "affine_definition_hash_matches": affine_definition_hash() == frozen.get("affine_definition_hash") == "7f0c8172751e7cad1cdcc3fb901d18de8bb06341e599570bfe259436b9ef4345",
        "scheme_b_manifest_hash_matches": sha256_file(config.scheme_b_root / "manifests" / "realface_256x320_manifest.csv") == frozen.get("scheme_b_manifest_hash"),
        "spec_primary_erosion_is_2": False,
        "generate_quality_grades_false": config.generate_quality_grades is False,
        "remove_samples_false": config.remove_samples is False,
        "full500_config_exists": config_text.is_file(),
    }
    spec = json.loads(spec_json_path.read_text(encoding="utf-8"))
    checks["spec_primary_erosion_is_2"] = int(spec.get("primary_skin_erosion_px")) == 2
    checks["pass"] = bool(all(checks.values()))
    checks["frozen_path"] = str(frozen_path)
    checks["metric_spec_json_path"] = str(spec_json_path)
    write_json(dirs["preflight"] / "frozen_integrity_check.json", checks)
    lines = ["# Frozen Integrity Check", "", *[f"- {k}: {v}" for k, v in checks.items()]]
    (dirs["preflight"] / "frozen_integrity_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    if not checks["pass"]:
        raise ValueError("frozen integrity check failed")
    return checks


def load_oof(config: Stage2Config) -> pd.DataFrame:
    from .id_utils import normalize_id

    oof = pd.read_csv(config.rgb_oof_csv, dtype=str)
    oof["sample_id"] = oof["sample_id"].map(normalize_id)
    oof["patient_group_id"] = oof["patient_group_id"].map(normalize_id)
    oof["fold"] = pd.to_numeric(oof["fold"], errors="raise").astype(int)
    oof["binary_label"] = pd.to_numeric(oof["binary_label"], errors="raise").astype(int)
    oof["prob_patient"] = pd.to_numeric(oof["prob_patient"], errors="raise").astype(float)
    oof["pred_class"] = pd.to_numeric(oof["pred_class"], errors="raise").astype(int)
    return oof


def full500_preflight(config: Stage2Config, dirs: dict[str, Path], class_map: dict[str, int]) -> dict[str, Any]:
    preflight = preflight_assets(config, class_map, dirs)
    split = load_split(config)
    oof = load_oof(config)
    manifest = load_manifest(config)
    inventory = pd.read_csv(dirs["preflight"] / "asset_inventory.csv")
    inventory.to_csv(dirs["preflight"] / "full500_asset_inventory.csv", index=False, encoding="utf-8-sig")
    merged = split.merge(oof, on="sample_id", suffixes=("_split", "_oof"), how="left")
    label_conflicts = int((merged["binary_label_split"].astype(int) != merged["binary_label_oof"].astype(int)).sum())
    fold_conflicts = int((merged["fold_split"].astype(int) != merged["fold_oof"].astype(int)).sum())
    group_cross_fold = int(split.groupby("patient_group_id")["fold"].nunique().gt(1).sum())
    duplicate_assets = int(manifest["sample_id"].duplicated().sum())
    summary = {
        **preflight,
        "rgb_oof_coverage": int(merged["prob_patient"].notna().sum()),
        "label_conflicts": label_conflicts,
        "fold_conflicts": fold_conflicts,
        "patient_group_cross_fold": group_cross_fold,
        "duplicate_oof": int(oof["sample_id"].duplicated().sum()),
        "duplicate_image_asset": duplicate_assets,
        "shape_all_ok": bool(inventory["status"].eq("passed").all()),
    }
    summary["pass"] = bool(
        summary["fixed_ids"] == 500
        and summary["unique_ids"] == 500
        and summary["rgb_oof_coverage"] == 500
        and label_conflicts == 0
        and fold_conflicts == 0
        and group_cross_fold == 0
        and summary["duplicate_oof"] == 0
        and duplicate_assets == 0
        and preflight["failures"] == 0
    )
    write_json(dirs["preflight"] / "full500_preflight_summary.json", summary)
    lines = ["# Full-500 Preflight", "", *[f"- {k}: {v}" for k, v in summary.items()]]
    (dirs["preflight"] / "full500_preflight_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    if not summary["pass"]:
        raise ValueError("Full-500 preflight failed")
    return summary


def extract_lighting_features(config: Stage2Config, dirs: dict[str, Path], class_map: dict[str, int]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    split = load_split(config)
    primary_rows: list[dict[str, Any]] = []
    erosion0_rows: list[dict[str, Any]] = []
    erosion4_rows: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    for sample_id in split["sample_id"].astype(str).tolist():
        try:
            paths = asset_paths(config, sample_id)
            aligned = read_rgb(paths["aligned_srgb"])
            parsing = read_gray(paths["parsing_label"])
            face = read_gray(paths["face_valid_mask"])
            source_valid = read_gray(paths["source_valid_mask"])
            raw_rgb, exif_applied = read_source_rgb_exif(paths["raw_scene"])
            meta = read_metadata(paths["metadata"])
            masks, diag = build_masks(parsing, face, source_valid, class_map, config.roi_geometry)
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
            roundtrip_core = source_to_canvas_mask(raw_masks["raw_core_skin"], (aligned.shape[1], aligned.shape[0]), canvas_to_source)
            iou = mask_iou(masks["core_skin_e2"], roundtrip_core)
            cd = centroid_distance(masks["core_skin_e2"], roundtrip_core)
            pm = primary_metrics(aligned, masks)
            aux = auxiliary_metrics(aligned, masks, config.shadow_gaussian_sigma)
            raw = raw_frozen_metrics(raw_rgb, raw_masks)
            roi = {
                "core_skin_pixel_count": int((masks["core_skin_e2"] > 0).sum()),
                "forehead_pixel_count": int((masks["forehead"] > 0).sum()),
                "canvas_left_cheek_pixel_count": int((masks["canvas_left_cheek"] > 0).sum()),
                "canvas_right_cheek_pixel_count": int((masks["canvas_right_cheek"] > 0).sum()),
                "nose_pixel_count": int((masks["nose"] > 0).sum()),
                "roi_valid": bool(min(int((masks[k] > 0).sum()) for k in ["core_skin_e2", "forehead", "canvas_left_cheek", "canvas_right_cheek", "nose"]) >= 32),
            }
            roundtrip = {
                "roundtrip_iou_core_skin": iou,
                "roundtrip_centroid_distance_core_skin": cd,
                "roundtrip_pass": bool(iou >= config.roundtrip_iou_threshold and (cd is None or cd <= config.centroid_distance_threshold)),
                "source_width": int(raw_rgb.shape[1]),
                "source_height": int(raw_rgb.shape[0]),
                "exif_orientation_applied": bool(exif_applied),
            }
            primary_rows.append({"sample_id": sample_id, **pm, **aux, **raw, **roi, **roundtrip})
            sens = erosion_sensitivity_metrics(aligned, masks)
            for erosion, rows in [(0, erosion0_rows), (4, erosion4_rows)]:
                prefix = f"skin_e{erosion}"
                median = sens[f"{prefix}_y_median"]
                p10 = sens[f"{prefix}_y_p10"]
                p05 = sens[f"{prefix}_y_p05"]
                rows.append(
                    {
                        "sample_id": sample_id,
                        "erosion_px": erosion,
                        "skin_y_median": median,
                        "skin_y_p05": p05,
                        "skin_y_p99": sens[f"{prefix}_y_p99"],
                        "skin_y_iqr": sens[f"{prefix}_y_iqr"],
                        "dark_contrast_score": (median - p10) / (median + 1e-8) if median is not None and p10 is not None else None,
                        "deep_dark_contrast_score": (median - p05) / (median + 1e-8) if median is not None and p05 is not None else None,
                        "cheek_relative_difference": pm["cheek_relative_difference"],
                    }
                )
        except Exception as exc:
            failures.append({"sample_id": sample_id, "failure": f"{type(exc).__name__}: {exc}"})
            raise
    pd.DataFrame(failures).to_csv(dirs["logs"] / "failures.csv", index=False, encoding="utf-8-sig")
    return pd.DataFrame(primary_rows), pd.DataFrame(erosion0_rows), pd.DataFrame(erosion4_rows)


def build_feature_table(config: Stage2Config, lighting: pd.DataFrame, erosion0: pd.DataFrame, erosion4: pd.DataFrame, dirs: dict[str, Path]) -> pd.DataFrame:
    split = load_split(config)
    oof = load_oof(config)
    base = split[["sample_id", "patient_group_id", "fold", "NYHA", "binary_label"]].rename(columns={"NYHA": "original_nyha"})
    frame = base.merge(lighting, on="sample_id", how="left").merge(
        oof[["sample_id", "prob_patient", "pred_class", "fold", "binary_label"]].rename(
            columns={"prob_patient": "rgb_oof_probability_patient", "pred_class": "rgb_oof_predicted_label", "fold": "oof_fold", "binary_label": "oof_binary_label"}
        ),
        on="sample_id",
        how="left",
    )
    frame["rgb_oof_predicted_label"] = (frame["rgb_oof_probability_patient"].astype(float) >= 0.5).astype(int)
    frame["rgb_oof_correct"] = (frame["rgb_oof_predicted_label"].astype(int) == frame["binary_label"].astype(int)).astype(int)
    frame["rgb_oof_error"] = 1 - frame["rgb_oof_correct"]
    frame["rgb_oof_residual"] = frame["rgb_oof_probability_patient"].astype(float) - frame["binary_label"].astype(float)
    frame["rgb_oof_logit"] = _clip_logit(frame["rgb_oof_probability_patient"])
    frame["rgb_oof_brier_contribution"] = frame["rgb_oof_residual"] ** 2
    main_cols = PRIMARY_METRICS + ["skin_y_mean", "skin_y_std", "skin_y_iqr", "skin_y_mad", "skin_y_p01", "skin_y_p05", "skin_y_p10", "skin_y_p25", "skin_y_p75", "skin_y_p90", "skin_y_p95", "skin_y_p90_minus_p10", "skin_y_mad_over_median"]
    gates = {
        "rows": int(len(frame)),
        "unique_ids": int(frame["sample_id"].nunique()),
        "primary_missing": int(frame[main_cols].isna().sum().sum()),
        "oof_probability_missing": int(frame["rgb_oof_probability_patient"].isna().sum()),
        "label_conflicts": int((frame["binary_label"].astype(int) != frame["oof_binary_label"].astype(int)).sum()),
        "fold_conflicts": int((frame["fold"].astype(int) != frame["oof_fold"].astype(int)).sum()),
        "roundtrip_failures": int((~frame["roundtrip_pass"].astype(bool)).sum()),
    }
    if gates["rows"] != 500 or gates["unique_ids"] != 500 or gates["primary_missing"] != 0 or gates["oof_probability_missing"] != 0 or gates["label_conflicts"] != 0 or gates["fold_conflicts"] != 0 or gates["roundtrip_failures"] != 0:
        raise ValueError(f"feature table gates failed: {gates}")
    frame.to_csv(dirs["features"] / "stage2_lighting_features_500.csv", index=False, encoding="utf-8-sig")
    erosion0.to_csv(dirs["features"] / "stage2_lighting_features_erosion0.csv", index=False, encoding="utf-8-sig")
    erosion4.to_csv(dirs["features"] / "stage2_lighting_features_erosion4.csv", index=False, encoding="utf-8-sig")
    dictionary = pd.DataFrame([{"feature": col, "dtype": str(frame[col].dtype), "role": "id_or_label" if col in ["sample_id", "patient_group_id", "fold", "binary_label"] else "stage2_feature"} for col in frame.columns])
    dictionary.to_csv(dirs["features"] / "stage2_lighting_feature_dictionary.csv", index=False, encoding="utf-8-sig")
    write_json(dirs["features"] / "stage2_lighting_feature_summary.json", {**gates, "success_rate": 1.0, "rgb_oof_auc": compute_metrics(frame["binary_label"], frame["rgb_oof_probability_patient"])["roc_auc"]})
    return frame


def metric_families(frame: pd.DataFrame) -> dict[str, list[str]]:
    aux = [c for c in frame.columns if c.startswith(("severe_", "deep_dark_fraction", "auxiliary_", "deep_relative_", "left_cheek_auxiliary", "right_cheek_auxiliary"))]
    raw = [c for c in frame.columns if c.startswith("raw_") and c.endswith("_fraction")]
    secondary = [c for c in SECONDARY_INPUT_METRICS if c in frame.columns]
    return {"primary": PRIMARY_METRICS, "secondary_input": secondary, "raw_coding_boundary": raw, "auxiliary": aux}


def run_label_association(frame: pd.DataFrame, dirs: dict[str, Path]) -> dict[str, Any]:
    families = metric_families(frame)
    metrics = [m for fam in families.values() for m in fam]
    desc_rows = []
    effect_rows = []
    test_rows = []
    for metric in metrics:
        x = pd.to_numeric(frame[metric], errors="coerce")
        c = x[frame["binary_label"] == 0].dropna()
        p = x[frame["binary_label"] == 1].dropna()
        desc_rows.append(
            {
                "metric": metric,
                "control_n": int(c.size),
                "patient_n": int(p.size),
                "control_mean": float(c.mean()),
                "control_sd": float(c.std(ddof=1)),
                "patient_mean": float(p.mean()),
                "patient_sd": float(p.std(ddof=1)),
                "control_median": float(c.median()),
                "control_iqr": float(c.quantile(0.75) - c.quantile(0.25)),
                "patient_median": float(p.median()),
                "patient_iqr": float(p.quantile(0.75) - p.quantile(0.25)),
                "min": float(x.min()),
                "max": float(x.max()),
                "missing_n": int(x.isna().sum()),
                "zero_fraction": float((x == 0).mean()),
            }
        )
        ci_low, ci_high, valid = cluster_bootstrap_smd(frame[["patient_group_id", "binary_label", metric]].dropna(), value_col=metric, repeats=BOOTSTRAP_REPEATS, seed=SEED)
        effect_rows.append({"metric": metric, "smd_patient_minus_control": smd(c, p), "median_difference_patient_minus_control": float(p.median() - c.median()), "smd_ci95_low": ci_low, "smd_ci95_high": ci_high, "bootstrap_valid": valid, "bootstrap_repeats": BOOTSTRAP_REPEATS})
        try:
            mw_p = float(stats.mannwhitneyu(c, p, alternative="two-sided").pvalue)
        except Exception:
            mw_p = None
        family = next(name for name, vals in families.items() if metric in vals)
        test_rows.append({"metric": metric, "family": family, "mannwhitney_p": mw_p})
    tests = pd.DataFrame(test_rows)
    q_parts = []
    for fam, sub in tests.groupby("family", sort=False):
        sub = sub.copy()
        sub["q_value"] = _bh_fdr(sub["mannwhitney_p"].tolist())
        sub["q_lt_0.05"] = sub["q_value"].fillna(1) < 0.05
        q_parts.append(sub)
    tests = pd.concat(q_parts, ignore_index=True)
    pd.DataFrame(desc_rows).to_csv(dirs["association"] / "lighting_label_descriptive.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(effect_rows).to_csv(dirs["association"] / "lighting_label_effect_sizes.csv", index=False, encoding="utf-8-sig")
    tests.to_csv(dirs["association"] / "lighting_label_tests.csv", index=False, encoding="utf-8-sig")
    regress = _label_regression(frame)
    regress.to_csv(dirs["association"] / "lighting_label_regression.csv", index=False, encoding="utf-8-sig")
    summary = {fam: {"n": int((tests["family"] == fam).sum()), "q_lt_0.05": int(((tests["family"] == fam) & tests["q_lt_0.05"]).sum())} for fam in families}
    write_json(dirs["association"] / "lighting_label_fdr_summary.json", summary)
    return {"tests": tests, "regression": regress, "fdr": summary}


def _fit_model(frame: pd.DataFrame, y_col: str, x_cols: list[str], model_type: str) -> pd.DataFrame:
    try:
        import statsmodels.api as sm
        from statsmodels.stats.outliers_influence import variance_inflation_factor

        work = frame[["patient_group_id", y_col, *x_cols]].dropna().copy()
        y = work[y_col].astype(float)
        x = sm.add_constant(work[x_cols].astype(float), has_constant="add")
        if model_type == "logit":
            model = sm.Logit(y, x).fit(disp=False, maxiter=200, cov_type="cluster", cov_kwds={"groups": work["patient_group_id"].astype(str)})
            cov_type = "cluster"
        else:
            model = sm.OLS(y, x).fit(cov_type="cluster", cov_kwds={"groups": work["patient_group_id"].astype(str)})
            cov_type = "cluster"
        ci = model.conf_int()
        rows = []
        vif = {}
        if len(x_cols) > 1:
            for i, col in enumerate(x.columns):
                if col != "const":
                    try:
                        vif[col] = float(variance_inflation_factor(x.values, i))
                    except Exception:
                        vif[col] = None
        for term in model.params.index:
            coef = float(model.params[term])
            row = {
                "model": f"{model_type}:{y_col}~{'+'.join(x_cols)}",
                "term": term,
                "coefficient": coef,
                "ci95_low": float(ci.loc[term, 0]),
                "ci95_high": float(ci.loc[term, 1]),
                "p_value": float(model.pvalues[term]),
                "or": float(np.exp(coef)) if model_type == "logit" else None,
                "or_ci95_low": float(np.exp(ci.loc[term, 0])) if model_type == "logit" else None,
                "or_ci95_high": float(np.exp(ci.loc[term, 1])) if model_type == "logit" else None,
                "vif": vif.get(term),
                "n": int(len(work)),
                "r_squared": float(getattr(model, "rsquared", np.nan)) if model_type == "ols" else None,
                "fit_status": "ok",
                "cov_type": cov_type,
            }
            rows.append(row)
        return pd.DataFrame(rows)
    except Exception as exc:
        return pd.DataFrame([{"model": f"{model_type}:{y_col}~{'+'.join(x_cols)}", "term": "model_failed", "fit_status": f"{type(exc).__name__}: {exc}"}])


def _label_regression(frame: pd.DataFrame) -> pd.DataFrame:
    work = frame.copy()
    for m in PRIMARY_METRICS:
        work[f"z_{m}"] = _z(work[m])
    rows = []
    for m in PRIMARY_METRICS:
        rows.append(_fit_model(work, "binary_label", [f"z_{m}"], "logit"))
    rows.append(_fit_model(work, "binary_label", [f"z_{m}" for m in PRIMARY_METRICS], "logit"))
    return pd.concat(rows, ignore_index=True)


def run_rgb_dependence(frame: pd.DataFrame, dirs: dict[str, Path]) -> dict[str, Any]:
    work = frame.copy()
    work["absolute_residual"] = work["rgb_oof_residual"].abs()
    for m in PRIMARY_METRICS:
        work[f"z_{m}"] = _z(work[m])
    spearman_rows = []
    targets = ["rgb_oof_probability_patient", "rgb_oof_logit", "rgb_oof_residual", "rgb_oof_brier_contribution"]
    for metric in PRIMARY_METRICS:
        for target in targets:
            for subset_name, sub in [("all", work), ("control", work[work["binary_label"] == 0]), ("patient", work[work["binary_label"] == 1])]:
                ok = pd.to_numeric(sub[metric], errors="coerce").notna() & pd.to_numeric(sub[target], errors="coerce").notna()
                rho, p = stats.spearmanr(sub.loc[ok, metric], sub.loc[ok, target]) if ok.sum() >= 3 else (np.nan, np.nan)
                spearman_rows.append({"metric": metric, "target": target, "subset": subset_name, "n": int(ok.sum()), "spearman_r": float(rho) if np.isfinite(rho) else None, "p_value": float(p) if np.isfinite(p) else None})
    spearman = pd.DataFrame(spearman_rows)
    spearman.to_csv(dirs["rgb_dependence"] / "rgb_lighting_spearman.csv", index=False, encoding="utf-8-sig")
    logit_rows = []
    error_rows = []
    magnitude_rows = []
    for metric in PRIMARY_METRICS:
        logit_rows.append(_fit_model(work, "rgb_oof_logit", ["binary_label", f"z_{metric}"], "ols"))
        error_rows.append(_fit_model(work, "rgb_oof_error", ["binary_label", f"z_{metric}"], "logit"))
        magnitude_rows.append(_fit_model(work, "rgb_oof_brier_contribution", ["binary_label", f"z_{metric}"], "ols"))
        magnitude_rows.append(_fit_model(work, "absolute_residual", ["binary_label", f"z_{metric}"], "ols"))
    logit_rows.append(_fit_model(work, "rgb_oof_logit", ["binary_label", *[f"z_{m}" for m in PRIMARY_METRICS]], "ols"))
    error_rows.append(_fit_model(work, "rgb_oof_error", ["binary_label", *[f"z_{m}" for m in PRIMARY_METRICS]], "logit"))
    logit = pd.concat(logit_rows, ignore_index=True)
    error = pd.concat(error_rows, ignore_index=True)
    magnitude = pd.concat(magnitude_rows, ignore_index=True)
    z_terms = [f"z_{m}" for m in PRIMARY_METRICS]
    for df in [logit, error, magnitude]:
        mask = df["term"].isin(z_terms) if "term" in df else []
        df["q_value"] = None
        if len(df) and any(mask):
            df.loc[mask, "q_value"] = _bh_fdr(df.loc[mask, "p_value"].tolist())
    logit.to_csv(dirs["rgb_dependence"] / "rgb_logit_adjusted_regression.csv", index=False, encoding="utf-8-sig")
    error.to_csv(dirs["rgb_dependence"] / "rgb_error_risk_regression.csv", index=False, encoding="utf-8-sig")
    magnitude.to_csv(dirs["rgb_dependence"] / "rgb_error_magnitude_analysis.csv", index=False, encoding="utf-8-sig")
    summary = {
        "logit_lighting_terms_q_lt_0.05": int(pd.to_numeric(logit.loc[logit["term"].isin(z_terms), "q_value"], errors="coerce").lt(0.05).sum()),
        "error_lighting_terms_q_lt_0.05": int(pd.to_numeric(error.loc[error["term"].isin(z_terms), "q_value"], errors="coerce").lt(0.05).sum()),
    }
    write_json(dirs["rgb_dependence"] / "rgb_lighting_dependence_fdr.json", summary)
    return {"spearman": spearman, "logit": logit, "error": error, "magnitude": magnitude, "fdr": summary}


def foldwise_stratification(frame: pd.DataFrame, dirs: dict[str, Path]) -> tuple[pd.DataFrame, pd.DataFrame]:
    assignments = []
    cutpoints = []
    for metric in STRATIFICATION_METRICS:
        for fold in sorted(frame["fold"].astype(int).unique()):
            train = frame.loc[frame["fold"].astype(int) != fold, metric].astype(float).dropna()
            test = frame.loc[frame["fold"].astype(int) == fold, ["sample_id", "fold", metric]].copy()
            q33, q67 = train.quantile([0.3333, 0.6667]).tolist()
            non = bool(not np.isfinite(q33) or not np.isfinite(q67) or q33 == q67)
            cutpoints.append({"metric_name": metric, "fold": fold, "train_q33": q33, "train_q67": q67, "train_n": int(train.size), "non_stratifiable": non})
            for _, row in test.iterrows():
                val = float(row[metric]) if pd.notna(row[metric]) else np.nan
                if non or not np.isfinite(val):
                    group = "non_stratifiable"
                elif val <= q33:
                    group = "low"
                elif val > q67:
                    group = "high"
                else:
                    group = "middle"
                assignments.append({"sample_id": row["sample_id"], "fold": int(fold), "metric_name": metric, "train_q33": q33, "train_q67": q67, "assigned_group": group})
    a = pd.DataFrame(assignments)
    c = pd.DataFrame(cutpoints)
    a.to_csv(dirs["stratified"] / "foldwise_stratification_assignments.csv", index=False, encoding="utf-8-sig")
    c.to_csv(dirs["stratified"] / "foldwise_cutpoints.csv", index=False, encoding="utf-8-sig")
    return a, c


def _calibration(y: pd.Series, prob: pd.Series) -> tuple[float | None, float | None]:
    try:
        import statsmodels.api as sm

        x = sm.add_constant(_clip_logit(prob), has_constant="add")
        model = sm.Logit(y.astype(int), x).fit(disp=False, maxiter=100)
        return float(model.params.iloc[0]), float(model.params.iloc[1])
    except Exception:
        return None, None


def stratified_performance(frame: pd.DataFrame, assignments: pd.DataFrame, dirs: dict[str, Path]) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    rows = []
    for metric in STRATIFICATION_METRICS:
        for group in ["low", "middle", "high", "non_stratifiable"]:
            ids = assignments.loc[(assignments["metric_name"] == metric) & (assignments["assigned_group"] == group), "sample_id"]
            sub = frame[frame["sample_id"].isin(ids)].copy()
            if sub.empty:
                continue
            metrics = compute_metrics(sub["binary_label"], sub["rgb_oof_probability_patient"])
            cal_i, cal_s = _calibration(sub["binary_label"], sub["rgb_oof_probability_patient"])
            unstable = bool(len(sub) < 30 or (sub["binary_label"] == 0).sum() < 10 or (sub["binary_label"] == 1).sum() < 10 or sub["patient_group_id"].nunique() < 25)
            row = {
                "metric_name": metric,
                "assigned_group": group,
                "n": int(len(sub)),
                "patient_group_n": int(sub["patient_group_id"].nunique()),
                "control_n": int((sub["binary_label"] == 0).sum()),
                "patient_n": int((sub["binary_label"] == 1).sum()),
                "unstable": unstable,
                "mean_patient_probability": float(sub["rgb_oof_probability_patient"].mean()),
                "calibration_intercept": cal_i,
                "calibration_slope": cal_s,
                **metrics,
            }
            if sub["binary_label"].nunique() == 2:
                boot = cluster_bootstrap_metrics(sub.rename(columns={"rgb_oof_probability_patient": "probability_patient"}), repeats=BOOTSTRAP_REPEATS, seed=SEED)
                for key, ci in boot["ci95"].items():
                    row[f"{key}_ci95_low"] = ci[0]
                    row[f"{key}_ci95_high"] = ci[1]
                row["bootstrap_valid_iterations"] = boot["valid_iterations"]
                row["bootstrap_failed_iterations"] = boot["failed_iterations"]
            else:
                row["auc_status"] = "single_class_auc_na"
            rows.append(row)
    strat = pd.DataFrame(rows)
    strat.to_csv(dirs["stratified"] / "rgb_lighting_stratified_metrics.csv", index=False, encoding="utf-8-sig")
    worst_rows = []
    worst_json: dict[str, Any] = {}
    for metric, sub in strat[strat["assigned_group"].isin(["low", "middle", "high"])].groupby("metric_name"):
        valid = sub.dropna(subset=["roc_auc"])
        if valid.empty:
            payload = {"metric_name": metric, "WorstGroupAUC": None, "BestGroupAUC": None, "DeltaAUC": None, "valid_group_n": 0}
        else:
            payload = {
                "metric_name": metric,
                "WorstGroupAUC": float(valid["roc_auc"].min()),
                "BestGroupAUC": float(valid["roc_auc"].max()),
                "DeltaAUC": float(valid["roc_auc"].max() - valid["roc_auc"].min()),
                "WorstGroupBalancedAccuracy": float(valid["balanced_accuracy"].min()),
                "DeltaBalancedAccuracy": float(valid["balanced_accuracy"].max() - valid["balanced_accuracy"].min()),
                "WorstGroupSensitivity": float(valid["sensitivity"].min()),
                "WorstGroupSpecificity": float(valid["specificity"].min()),
                "valid_group_n": int(len(valid)),
                "worst_auc_group": str(valid.loc[valid["roc_auc"].idxmin(), "assigned_group"]),
            }
        worst_rows.append(payload)
        worst_json[metric] = payload
    worst = pd.DataFrame(worst_rows)
    worst.to_csv(dirs["stratified"] / "rgb_lighting_worst_group_metrics.csv", index=False, encoding="utf-8-sig")
    write_json(dirs["stratified"] / "rgb_lighting_worst_group_metrics.json", worst_json)
    return strat, worst, worst_json


def run_sensitivity(frame: pd.DataFrame, e0: pd.DataFrame, e4: pd.DataFrame, dirs: dict[str, Path]) -> dict[str, Any]:
    e2 = frame[["sample_id", "patient_group_id", "binary_label", "rgb_oof_logit", "rgb_oof_error", "rgb_oof_probability_patient", *SENSITIVITY_METRICS]].copy()
    e2["erosion_px"] = 2
    e0m = e0.merge(frame[["sample_id", "patient_group_id", "binary_label", "rgb_oof_logit", "rgb_oof_error", "rgb_oof_probability_patient", "fold"]], on="sample_id")
    e4m = e4.merge(frame[["sample_id", "patient_group_id", "binary_label", "rgb_oof_logit", "rgb_oof_error", "rgb_oof_probability_patient", "fold"]], on="sample_id")
    corr_rows = []
    for metric in SENSITIVITY_METRICS:
        for erosion, other in [(0, e0m), (4, e4m)]:
            rho, p = stats.spearmanr(frame[metric], other[metric])
            corr_rows.append({"metric": metric, "comparison": f"erosion2_vs_erosion{erosion}", "spearman_r": float(rho), "p_value": float(p)})
    pd.DataFrame(corr_rows).to_csv(dirs["sensitivity"] / "erosion_metric_correlations.csv", index=False, encoding="utf-8-sig")
    label_rows = []
    rgb_rows = []
    strat_rows = []
    for erosion, data in [(0, e0m), (2, frame), (4, e4m)]:
        for metric in SENSITIVITY_METRICS:
            c = data.loc[data["binary_label"] == 0, metric]
            p = data.loc[data["binary_label"] == 1, metric]
            label_rows.append({"erosion_px": erosion, "metric": metric, "smd_patient_minus_control": smd(c, p), "median_difference_patient_minus_control": float(pd.to_numeric(p).median() - pd.to_numeric(c).median())})
            tmp = data.copy()
            tmp[f"z_{metric}"] = _z(tmp[metric])
            reg = _fit_model(tmp, "rgb_oof_logit", ["binary_label", f"z_{metric}"], "ols")
            zrow = reg[reg["term"] == f"z_{metric}"].head(1)
            if not zrow.empty:
                rgb_rows.append({"erosion_px": erosion, "metric": metric, "rgb_logit_coef": zrow.iloc[0].get("coefficient"), "p_value": zrow.iloc[0].get("p_value")})
            # compact within-erosion tertiles on the full frame for sensitivity only
            q1, q2 = pd.to_numeric(data[metric]).quantile([0.3333, 0.6667]).tolist()
            groups = pd.cut(pd.to_numeric(data[metric]), bins=[-np.inf, q1, q2, np.inf], labels=["low", "middle", "high"], include_lowest=True)
            aucs = []
            for g in ["low", "middle", "high"]:
                sub = data.loc[groups == g]
                aucs.append(compute_metrics(sub["binary_label"], sub["rgb_oof_probability_patient"])["roc_auc"] if len(sub) and sub["binary_label"].nunique() == 2 else np.nan)
            strat_rows.append({"erosion_px": erosion, "metric": metric, "WorstGroupAUC": float(np.nanmin(aucs)) if np.isfinite(aucs).any() else None, "DeltaAUC": float(np.nanmax(aucs) - np.nanmin(aucs)) if np.isfinite(aucs).any() else None})
    pd.DataFrame(label_rows).to_csv(dirs["sensitivity"] / "erosion_label_association.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(rgb_rows).to_csv(dirs["sensitivity"] / "erosion_rgb_dependence.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(strat_rows).to_csv(dirs["sensitivity"] / "erosion_stratified_summary.csv", index=False, encoding="utf-8-sig")
    return {"min_erosion_spearman": float(pd.DataFrame(corr_rows)["spearman_r"].min())}


def make_figures(frame: pd.DataFrame, assoc: dict[str, Any], rgb: dict[str, Any], strat: pd.DataFrame, worst: pd.DataFrame, dirs: dict[str, Path]) -> None:
    plt.switch_backend("Agg")
    fig_paths = [
        "primary_metric_distributions_by_label.png",
        "primary_metric_effect_sizes.png",
        "rgb_logit_vs_primary_metrics.png",
        "rgb_error_rate_vs_primary_metrics.png",
        "stratified_auc_by_lighting_metric.png",
        "worst_group_auc_summary.png",
        "raw_coding_boundary_distributions.png",
    ]
    for name in fig_paths:
        plt.figure(figsize=(9, 5))
        if name == "primary_metric_distributions_by_label.png":
            data = [frame.loc[frame["binary_label"] == lab, PRIMARY_METRICS[0]] for lab in [0, 1]]
            plt.boxplot(data, labels=["Control", "Patient"])
            plt.title("skin_y_median by label")
        elif name == "primary_metric_effect_sizes.png":
            eff = pd.read_csv(dirs["association"] / "lighting_label_effect_sizes.csv")
            sub = eff[eff["metric"].isin(PRIMARY_METRICS)]
            plt.bar(sub["metric"], sub["smd_patient_minus_control"])
            plt.xticks(rotation=25, ha="right")
        elif name == "rgb_logit_vs_primary_metrics.png":
            plt.scatter(frame[PRIMARY_METRICS[0]], frame["rgb_oof_logit"], s=12, alpha=0.7)
            plt.xlabel(PRIMARY_METRICS[0]); plt.ylabel("rgb_oof_logit")
        elif name == "rgb_error_rate_vs_primary_metrics.png":
            ordered = frame.sort_values(PRIMARY_METRICS[0])
            roll = ordered["rgb_oof_error"].rolling(50, center=True, min_periods=20).mean()
            plt.plot(ordered[PRIMARY_METRICS[0]], roll)
            plt.xlabel(PRIMARY_METRICS[0]); plt.ylabel("moving-window error rate")
        elif name == "stratified_auc_by_lighting_metric.png":
            sub = strat[strat["assigned_group"].isin(["low", "middle", "high"])]
            for group, g in sub.groupby("assigned_group"):
                plt.scatter(g["metric_name"], g["roc_auc"], label=group)
            plt.xticks(rotation=35, ha="right"); plt.legend()
        elif name == "worst_group_auc_summary.png":
            plt.bar(worst["metric_name"], worst["WorstGroupAUC"])
            plt.xticks(rotation=35, ha="right")
        else:
            raw_cols = [c for c in frame.columns if c.startswith("raw_") and c.endswith("_fraction")][:6]
            frame[raw_cols].plot(kind="box", ax=plt.gca())
            plt.xticks(rotation=25, ha="right")
        plt.tight_layout()
        plt.savefig(dirs["figures"] / name, dpi=180)
        plt.close()


def decide_stage3(worst: pd.DataFrame, rgb: dict[str, Any], sensitivity: dict[str, Any]) -> dict[str, Any]:
    max_delta = float(pd.to_numeric(worst["DeltaAUC"], errors="coerce").max())
    logit_hits = int(rgb["fdr"]["logit_lighting_terms_q_lt_0.05"])
    error_hits = int(rgb["fdr"]["error_lighting_terms_q_lt_0.05"])
    min_auc = float(pd.to_numeric(worst["WorstGroupAUC"], errors="coerce").min())
    if error_hits >= 2 or max_delta > 0.10 or min_auc < 0.60:
        level = "high"
        rec = "Stage3 must proceed with constrained exposure/gamma/white-balance/local-shadow augmentation; consider lighting consistency loss after augmentation baselines."
    elif logit_hits >= 1 or error_hits >= 1 or max_delta >= 0.05:
        level = "moderate"
        rec = "Stage3 should implement constrained exposure, gamma, white-balance and local-shadow augmentation, prioritizing Worst-group AUC stability."
    else:
        level = "low"
        rec = "Stage3 can be a lightweight constrained augmentation stability validation; complex adversarial deconfounding is not required by Stage2 evidence."
    return {"lighting_dependence_level": level, "stage3_recommendation": rec, "evidence": {"max_DeltaAUC": max_delta, "min_WorstGroupAUC": min_auc, "rgb_logit_fdr_hits": logit_hits, "rgb_error_fdr_hits": error_hits, **sensitivity}}


def write_reports(frame: pd.DataFrame, assoc: dict[str, Any], rgb: dict[str, Any], strat: pd.DataFrame, worst: pd.DataFrame, decision: dict[str, Any], summary: dict[str, Any], dirs: dict[str, Path]) -> None:
    primary_tests = assoc["tests"][assoc["tests"]["metric"].isin(PRIMARY_METRICS)][["metric", "mannwhitney_p", "q_value"]]
    lines = [
        "# Lighting QC Stage2 Full500 Report",
        "",
        "## 1-4. Purpose and Frozen Protocol",
        "This audit applies the frozen Stage2-R1 image-level lighting metric specification to the fixed 500-case OOF cohort. Metrics are image presentation / capture-quality proxies, not real environmental illumination or full physical light decomposition.",
        "",
        "## 5. Asset Integrity",
        f"- Full-500 rows: {len(frame)}",
        f"- unique sample IDs: {frame['sample_id'].nunique()}",
        f"- OOF AUC: {summary['rgb_oof_overall']['roc_auc']:.4f}",
        "",
        "## 6-9. Core Skin, ROI and Metrics",
        "Core skin, ROI and raw affine mapping are unchanged from R1. Primary metrics: skin_y_median, dark_contrast_score, cheek_relative_difference, skin_y_p99. Canvas left/right are not anatomical left/right.",
        "",
        "## 10-11. Label Association",
        primary_tests.to_markdown(index=False),
        "",
        "## 12-15. RGB OOF Dependence",
        f"- label-adjusted RGB logit lighting FDR hits: {rgb['fdr']['logit_lighting_terms_q_lt_0.05']}",
        f"- RGB error-risk lighting FDR hits: {rgb['fdr']['error_lighting_terms_q_lt_0.05']}",
        "",
        "## 16-18. Foldwise Stratified Performance",
        f"- minimum WorstGroupAUC: {pd.to_numeric(worst['WorstGroupAUC'], errors='coerce').min():.4f}",
        f"- maximum DeltaAUC: {pd.to_numeric(worst['DeltaAUC'], errors='coerce').max():.4f}",
        "",
        "## 19-20. Raw Boundary and Erosion Sensitivity",
        "Raw coding-boundary metrics were computed only inside raw_core_skin. Erosion 0/4 sensitivity outputs are separate from the primary table.",
        "",
        "## 21-22. Stage3 Decision",
        f"- lighting_dependence_level: {decision['lighting_dependence_level']}",
        f"- recommendation: {decision['stage3_recommendation']}",
        "",
        "## 23-24. Limits and Non-interpretation",
        "Associations are not causal; no samples were deleted; no model was retrained; only internal fixed-fivefold OOF predictions were used; no A/B/C/D quality grades were generated.",
    ]
    (dirs["reports"] / "stage2_full500_report.md").write_text("\n\n".join(lines) + "\n", encoding="utf-8")
    (dirs["reports"] / "stage2_to_stage3_decision.md").write_text(f"# Stage2 to Stage3 Decision\n\n- level: {decision['lighting_dependence_level']}\n- recommendation: {decision['stage3_recommendation']}\n- evidence: `{json.dumps(decision['evidence'], ensure_ascii=False)}`\n", encoding="utf-8")
    write_json(dirs["reports"] / "stage2_full500_machine_summary.json", summary)
    inventory = {str(p.relative_to(dirs["reports"].parents[0])): p.stat().st_size for p in dirs["reports"].parents[0].rglob("*") if p.is_file()}
    write_json(dirs["reports"] / "stage2_full500_output_inventory.json", inventory)


def run_full500(config: Stage2Config) -> dict[str, Any]:
    if config.mode != "full500":
        raise ValueError("run_full500 requires mode: full500")
    if config.generate_quality_grades or config.remove_samples:
        raise ValueError("Full-500 config must not generate grades or remove samples")
    dirs = full500_dirs(config.output_dir)
    warnings: list[dict[str, Any]] = []
    class_map = load_parsing_class_map(config.project_root)
    write_class_map(dirs["preflight"] / "detected_parsing_class_map.json", class_map)
    frozen = verify_frozen_integrity(config, dirs)
    preflight = full500_preflight(config, dirs, class_map)
    lighting, erosion0, erosion4 = extract_lighting_features(config, dirs, class_map)
    frame = build_feature_table(config, lighting, erosion0, erosion4, dirs)
    rgb_oof_overall = compute_metrics(frame["binary_label"], frame["rgb_oof_probability_patient"])
    assoc = run_label_association(frame, dirs)
    rgb = run_rgb_dependence(frame, dirs)
    assignments, cutpoints = foldwise_stratification(frame, dirs)
    strat, worst, worst_json = stratified_performance(frame, assignments, dirs)
    sensitivity = run_sensitivity(frame, erosion0, erosion4, dirs)
    make_figures(frame, assoc, rgb, strat, worst, dirs)
    decision = decide_stage3(worst, rgb, sensitivity)
    summary = {
        "experiment_name": config.experiment_name,
        "frozen_integrity_pass": bool(frozen["pass"]),
        "preflight_pass": bool(preflight["pass"]),
        "feature_rows": int(len(frame)),
        "feature_success_rate": 1.0,
        "primary_metric_missing": int(frame[PRIMARY_METRICS].isna().sum().sum()),
        "rgb_oof_overall": rgb_oof_overall,
        "primary_label_tests": assoc["tests"][assoc["tests"]["metric"].isin(PRIMARY_METRICS)].to_dict("records"),
        "rgb_dependence_fdr": rgb["fdr"],
        "worst_group": worst_json,
        **decision,
        "bootstrap_repeats": BOOTSTRAP_REPEATS,
        "bootstrap_seed": SEED,
        "full_500_run": True,
        "model_training_run": False,
        "samples_removed": False,
        "frozen_metrics_modified": False,
        "quality_grades_generated": False,
        "entered_stage3": False,
    }
    write_reports(frame, assoc, rgb, strat, worst, decision, summary, dirs)
    pd.DataFrame(warnings).to_csv(dirs["logs"] / "warnings.csv", index=False, encoding="utf-8-sig")
    if not (dirs["logs"] / "failures.csv").is_file():
        pd.DataFrame([]).to_csv(dirs["logs"] / "failures.csv", index=False, encoding="utf-8-sig")
    (dirs["logs"] / "run.log").write_text("Stage2 Full500 completed; no model training, no sample deletion, no Stage3 execution.\n", encoding="utf-8")
    return summary
