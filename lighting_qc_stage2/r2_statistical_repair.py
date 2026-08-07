from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from lighting_confounding_stage1.metrics_utils import compute_metrics

EROSION_METRICS = ["skin_y_median", "skin_y_p05", "skin_y_p99", "skin_y_iqr", "dark_contrast_score", "cheek_relative_difference"]
GROUPS = ["low", "middle", "high"]


def stability_status(frame: pd.DataFrame) -> tuple[bool, bool, str]:
    n_total = len(frame)
    n_control = int((frame["binary_label"].astype(int) == 0).sum())
    n_patient = int((frame["binary_label"].astype(int) == 1).sum())
    n_groups = int(frame["patient_group_id"].astype(str).nunique())
    reasons = []
    if n_total < 30:
        reasons.append("n_total<30")
    if n_control < 20:
        reasons.append("n_control<20")
    if n_patient < 20:
        reasons.append("n_patient<20")
    if n_groups < 25:
        reasons.append("n_patient_groups<25")
    severe = n_control < 10 or n_patient < 10
    if n_control < 10:
        reasons.append("severe:n_control<10")
    if n_patient < 10:
        reasons.append("severe:n_patient<10")
    return bool(reasons), bool(severe), ";".join(reasons)


def foldwise_assignments(features_by_erosion: dict[int, pd.DataFrame], labels: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    assign_rows = []
    cut_rows = []
    for erosion, feat in features_by_erosion.items():
        data = labels.merge(feat[["sample_id", *EROSION_METRICS]], on="sample_id", how="inner")
        for metric in EROSION_METRICS:
            for fold in sorted(data["fold"].astype(int).unique()):
                train = pd.to_numeric(data.loc[data["fold"].astype(int) != fold, metric], errors="coerce").dropna()
                test = data.loc[data["fold"].astype(int) == fold, ["sample_id", "fold", metric]].copy()
                q33, q67 = train.quantile([0.3333, 0.6667]).tolist()
                non = bool(not np.isfinite(q33) or not np.isfinite(q67) or q33 == q67)
                cut_rows.append({"erosion_px": erosion, "metric_name": metric, "fold": fold, "train_q33": q33, "train_q67": q67, "train_n": int(len(train)), "non_stratifiable": non})
                for _, row in test.iterrows():
                    value = float(row[metric]) if pd.notna(row[metric]) else np.nan
                    if non or not np.isfinite(value):
                        group = "non_stratifiable"
                    elif value <= q33:
                        group = "low"
                    elif value > q67:
                        group = "high"
                    else:
                        group = "middle"
                    assign_rows.append({"sample_id": row["sample_id"], "fold": int(fold), "erosion_px": erosion, "metric_name": metric, "train_q33": q33, "train_q67": q67, "assigned_group": group})
    return pd.DataFrame(assign_rows), pd.DataFrame(cut_rows)


def stratified_metrics(frame: pd.DataFrame, assignments: pd.DataFrame, *, iterations: int = 2000, seed: int = 2026) -> pd.DataFrame:
    rows = []
    base_cols = ["sample_id", "patient_group_id", "binary_label", "rgb_oof_probability_patient"]
    for i, ((erosion, metric, group), sub_assign) in enumerate(assignments.groupby(["erosion_px", "metric_name", "assigned_group"], sort=False)):
        if group not in GROUPS:
            continue
        sub = frame[base_cols].merge(sub_assign[["sample_id"]], on="sample_id", how="inner")
        labels = sub["binary_label"].astype(int)
        met = compute_metrics(labels, sub["rgb_oof_probability_patient"].astype(float)) if len(sub) else {}
        auc_low, auc_high, auc_valid, auc_invalid = _cluster_auc_ci(sub, iterations=iterations, seed=seed + i)
        unstable, severe, reasons = stability_status(sub)
        rows.append(
            {
                "erosion_px": int(erosion),
                "metric_name": metric,
                "assigned_group": group,
                "n_total": int(len(sub)),
                "n_control": int((labels == 0).sum()),
                "n_patient": int((labels == 1).sum()),
                "n_patient_groups": int(sub["patient_group_id"].astype(str).nunique()),
                "is_unstable": unstable,
                "is_severely_unstable": severe,
                "instability_reasons": reasons,
                "auc": met.get("roc_auc"),
                "auc_ci_low": auc_low,
                "auc_ci_high": auc_high,
                "auc_bootstrap_valid": auc_valid,
                "auc_bootstrap_invalid": auc_invalid,
                "balanced_accuracy": met.get("balanced_accuracy"),
                "sensitivity": met.get("sensitivity"),
                "specificity": met.get("specificity"),
                "brier_score": met.get("brier_score"),
            }
        )
    return pd.DataFrame(rows)


def _cluster_auc_ci(frame: pd.DataFrame, *, iterations: int, seed: int) -> tuple[float | None, float | None, int, int]:
    work = frame[["patient_group_id", "binary_label", "rgb_oof_probability_patient"]].dropna().copy()
    if work.empty:
        return None, None, 0, iterations
    cluster_codes, _ = pd.factorize(work["patient_group_id"].astype(str), sort=False)
    grouped = [
        (
            work.loc[cluster_codes == code, "binary_label"].to_numpy(dtype=int),
            work.loc[cluster_codes == code, "rgb_oof_probability_patient"].to_numpy(dtype=float),
        )
        for code in range(int(cluster_codes.max()) + 1)
    ]
    rng = np.random.default_rng(seed)
    vals: list[float] = []
    invalid = 0
    n_clusters = len(grouped)
    for _ in range(iterations):
        picked = rng.integers(0, n_clusters, size=n_clusters)
        labels = np.concatenate([grouped[int(code)][0] for code in picked])
        probs = np.concatenate([grouped[int(code)][1] for code in picked])
        auc = _fast_auc(labels, probs)
        if auc is None or not np.isfinite(auc):
            invalid += 1
            continue
        vals.append(float(auc))
    if not vals:
        return None, None, 0, invalid
    return float(np.quantile(vals, 0.025)), float(np.quantile(vals, 0.975)), len(vals), invalid


def _worst_stats_from_metrics(metrics: pd.DataFrame, stable_only: bool) -> dict[str, float | None]:
    sub = metrics.copy()
    if stable_only:
        sub = sub.loc[~sub["is_unstable"].astype(bool)]
    valid = sub[sub["assigned_group"].isin(GROUPS)].dropna(subset=["auc"])
    if len(valid) == 0:
        return {
            "WorstGroupAUC": None,
            "BestGroupAUC": None,
            "DeltaAUC": None,
            "WorstGroupBalancedAccuracy": None,
            "DeltaBalancedAccuracy": None,
            "WorstGroupSensitivity": None,
            "WorstGroupSpecificity": None,
        }
    delta_auc = float(valid["auc"].max() - valid["auc"].min()) if len(valid) >= 2 else None
    delta_ba = float(valid["balanced_accuracy"].max() - valid["balanced_accuracy"].min()) if len(valid) >= 2 else None
    return {
        "WorstGroupAUC": float(valid["auc"].min()),
        "BestGroupAUC": float(valid["auc"].max()),
        "DeltaAUC": delta_auc,
        "WorstGroupBalancedAccuracy": float(valid["balanced_accuracy"].min()),
        "DeltaBalancedAccuracy": delta_ba,
        "WorstGroupSensitivity": float(valid["sensitivity"].min()),
        "WorstGroupSpecificity": float(valid["specificity"].min()),
    }


def worst_group_summary(metrics: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (erosion, metric), sub in metrics.groupby(["erosion_px", "metric_name"], sort=False):
        for stable_only in [False, True]:
            stats = _worst_stats_from_metrics(sub, stable_only)
            rows.append(
                {
                    "erosion_px": int(erosion),
                    "metric_name": metric,
                    "scope": "stable_only" if stable_only else "all_groups",
                    "stable_group_count": int((~sub["is_unstable"].astype(bool)).sum()),
                    "valid_auc_group_count": int(sub.dropna(subset=["auc"]).shape[0]),
                    **stats,
                }
            )
    return pd.DataFrame(rows)


def worst_group_bootstrap(frame: pd.DataFrame, assignments: pd.DataFrame, metrics: pd.DataFrame, *, iterations: int, seed: int) -> tuple[pd.DataFrame, dict[str, Any]]:
    rows = []
    diagnostics: dict[str, Any] = {}
    base = frame[["sample_id", "patient_group_id", "binary_label", "rgb_oof_probability_patient"]]
    stable_lookup = metrics.set_index(["erosion_px", "metric_name", "assigned_group"])["is_unstable"].to_dict()
    assign_base = assignments[assignments["assigned_group"].isin(GROUPS)][["sample_id", "erosion_px", "metric_name", "assigned_group"]]
    joined = base.merge(assign_base, on="sample_id", how="inner")
    for (erosion, metric), sub in joined.groupby(["erosion_px", "metric_name"], sort=False):
        for scope in ["all_groups", "stable_only"]:
            stable_only = scope == "stable_only"
            boot = _bootstrap_worst_group_auc(sub, erosion=erosion, metric=metric, stable_only=stable_only, stable_lookup=stable_lookup, iterations=iterations, seed=seed)
            ci = boot["ci95"]
            rows.append(
                {
                    "erosion_px": int(erosion),
                    "metric_name": metric,
                    "scope": scope,
                    "WorstGroupAUC_ci95_low": ci.get("WorstGroupAUC", [None, None])[0],
                    "WorstGroupAUC_ci95_high": ci.get("WorstGroupAUC", [None, None])[1],
                    "BestGroupAUC_ci95_low": ci.get("BestGroupAUC", [None, None])[0],
                    "BestGroupAUC_ci95_high": ci.get("BestGroupAUC", [None, None])[1],
                    "DeltaAUC_ci95_low": ci.get("DeltaAUC", [None, None])[0],
                    "DeltaAUC_ci95_high": ci.get("DeltaAUC", [None, None])[1],
                    "valid_bootstrap_iterations": boot["valid_iterations"],
                    "invalid_bootstrap_iterations": boot["invalid_iterations"],
                }
            )
            diagnostics[f"erosion{erosion}_{metric}_{scope}"] = {k: v for k, v in boot.items() if k != "ci95"}
    return pd.DataFrame(rows), diagnostics


def _bootstrap_worst_group_auc(
    sub: pd.DataFrame,
    *,
    erosion: int,
    metric: str,
    stable_only: bool,
    stable_lookup: dict[tuple[int, str, str], bool],
    iterations: int,
    seed: int,
) -> dict[str, Any]:
    group_code_lookup = {name: code for code, name in enumerate(GROUPS)}
    active_codes = [
        code
        for code, name in enumerate(GROUPS)
        if not stable_only or not bool(stable_lookup.get((erosion, metric, name), True))
    ]
    work = sub.copy()
    work["_group_code"] = work["assigned_group"].map(group_code_lookup)
    work = work.dropna(subset=["_group_code", "binary_label", "rgb_oof_probability_patient"])
    if work.empty or not active_codes:
        return _empty_bootstrap_result(iterations, seed, iterations)
    cluster_codes, _ = pd.factorize(work["patient_group_id"].astype(str), sort=False)
    grouped = [
        (
            work.loc[cluster_codes == code, "_group_code"].to_numpy(dtype=int),
            work.loc[cluster_codes == code, "binary_label"].to_numpy(dtype=int),
            work.loc[cluster_codes == code, "rgb_oof_probability_patient"].to_numpy(dtype=float),
        )
        for code in range(int(cluster_codes.max()) + 1)
    ]
    rng = np.random.default_rng(seed)
    samples: dict[str, list[float]] = {"WorstGroupAUC": [], "BestGroupAUC": [], "DeltaAUC": []}
    invalid = 0
    n_clusters = len(grouped)
    for _ in range(iterations):
        picked = rng.integers(0, n_clusters, size=n_clusters)
        group_codes = np.concatenate([grouped[int(code)][0] for code in picked])
        labels = np.concatenate([grouped[int(code)][1] for code in picked])
        probs = np.concatenate([grouped[int(code)][2] for code in picked])
        stats = _auc_extrema(group_codes, labels, probs, active_codes)
        valid = {key: value for key, value in stats.items() if value is not None and np.isfinite(value)}
        if not valid:
            invalid += 1
            continue
        for key, value in valid.items():
            samples[key].append(float(value))
    ci = {
        key: [float(np.quantile(vals, 0.025)), float(np.quantile(vals, 0.975))] if vals else [None, None]
        for key, vals in samples.items()
    }
    return {
        "ci95": ci,
        "valid_iterations": int(iterations - invalid),
        "invalid_iterations": int(invalid),
        "iterations": int(iterations),
        "seed": int(seed),
        "cluster_unit": "patient_group_id",
    }


def _empty_bootstrap_result(iterations: int, seed: int, invalid: int) -> dict[str, Any]:
    return {
        "ci95": {"WorstGroupAUC": [None, None], "BestGroupAUC": [None, None], "DeltaAUC": [None, None]},
        "valid_iterations": 0,
        "invalid_iterations": int(invalid),
        "iterations": int(iterations),
        "seed": int(seed),
        "cluster_unit": "patient_group_id",
    }


def _auc_extrema(group_codes: np.ndarray, labels: np.ndarray, probs: np.ndarray, active_codes: list[int]) -> dict[str, float | None]:
    aucs = []
    for code in active_codes:
        mask = group_codes == code
        if not mask.any():
            continue
        auc = _fast_auc(labels[mask], probs[mask])
        if auc is not None and np.isfinite(auc):
            aucs.append(float(auc))
    if not aucs:
        return {"WorstGroupAUC": None, "BestGroupAUC": None, "DeltaAUC": None}
    worst = float(np.min(aucs))
    best = float(np.max(aucs))
    delta = float(best - worst) if len(aucs) >= 2 else None
    return {"WorstGroupAUC": worst, "BestGroupAUC": best, "DeltaAUC": delta}


def _fast_auc(labels: np.ndarray, scores: np.ndarray) -> float | None:
    y = np.asarray(labels, dtype=int)
    s = np.asarray(scores, dtype=float)
    finite = np.isfinite(s)
    y = y[finite]
    s = s[finite]
    n_pos = int((y == 1).sum())
    n_neg = int((y == 0).sum())
    if n_pos == 0 or n_neg == 0:
        return None
    ranks = _average_ranks(s)
    pos_rank_sum = float(ranks[y == 1].sum())
    return (pos_rank_sum - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)


def _average_ranks(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    sorted_values = values[order]
    ranks_sorted = np.empty(len(values), dtype=float)
    start = 0
    while start < len(values):
        end = start + 1
        while end < len(values) and sorted_values[end] == sorted_values[start]:
            end += 1
        ranks_sorted[start:end] = (start + end - 1) / 2.0 + 1.0
        start = end
    ranks = np.empty(len(values), dtype=float)
    ranks[order] = ranks_sorted
    return ranks


def compare_old_repaired(full500_root: Path, repaired_worst: pd.DataFrame) -> pd.DataFrame:
    old_path = full500_root / "sensitivity" / "erosion_stratified_summary.csv"
    rows = []
    if old_path.is_file():
        old = pd.read_csv(old_path)
        rows.append({"item": "old_file_present", "old_rows": len(old), "repaired_rows": len(repaired_worst), "confirmed_issue": "old sensitivity stratification used global quantiles; R2 repaired with foldwise train cutpoints"})
    else:
        rows.append({"item": "old_file_missing", "old_rows": 0, "repaired_rows": len(repaired_worst), "confirmed_issue": "old file unavailable, R2 regenerated repaired foldwise stratification"})
    return pd.DataFrame(rows)


def deduplicate_fdr_hits(full500_root: Path) -> tuple[pd.DataFrame, dict[str, Any]]:
    out_rows = []
    summary: dict[str, Any] = {}
    for analysis, rel in [
        ("rgb_logit", "rgb_dependence/rgb_logit_adjusted_regression.csv"),
        ("rgb_error", "rgb_dependence/rgb_error_risk_regression.csv"),
    ]:
        df = pd.read_csv(full500_root / rel)
        lighting = df[df["term"].astype(str).str.startswith("z_")].copy()
        lighting["metric_name"] = lighting["term"].str.replace("^z_", "", regex=True)
        single = lighting[lighting["model"].astype(str).str.count("\\+") == 1]
        multi = lighting[lighting["model"].astype(str).str.count("\\+") > 1]
        for subset_name, subset in [("single_metric_models", single), ("multivariable_model", multi)]:
            tested = sorted(subset["metric_name"].dropna().unique().tolist())
            significant = sorted(subset.loc[pd.to_numeric(subset["q_value"], errors="coerce") < 0.05, "metric_name"].dropna().unique().tolist())
            payload = {
                "analysis": analysis,
                "model_family": subset_name,
                "tested_unique_metrics": len(tested),
                "significant_unique_metrics": len(significant),
                "significant_metric_names": ";".join(significant),
            }
            out_rows.append(payload)
            summary[f"{analysis}_{subset_name}"] = payload
        union_sig = sorted(lighting.loc[pd.to_numeric(lighting["q_value"], errors="coerce") < 0.05, "metric_name"].dropna().unique().tolist())
        summary[f"{analysis}_deduplicated_any_model"] = {
            "tested_unique_metrics": int(lighting["metric_name"].nunique()),
            "significant_unique_metrics": len(union_sig),
            "significant_metric_names": union_sig,
            "note": "single and multivariable appearances are not counted as independent evidence rows",
        }
    return pd.DataFrame(out_rows), summary


def revised_evidence_summary(worst: pd.DataFrame, boot_ci: pd.DataFrame, fdr_dedup: dict[str, Any]) -> dict[str, Any]:
    all_rows = worst[worst["scope"] == "all_groups"]
    stable_rows = worst[worst["scope"] == "stable_only"]
    stable_delta = pd.to_numeric(stable_rows["DeltaAUC"], errors="coerce")
    if stable_delta.notna().any():
        idx = stable_delta.idxmax()
        row = stable_rows.loc[idx]
        ci_row = boot_ci[(boot_ci["erosion_px"] == row["erosion_px"]) & (boot_ci["metric_name"] == row["metric_name"]) & (boot_ci["scope"] == "stable_only")]
        ci = [None, None] if ci_row.empty else [ci_row.iloc[0]["DeltaAUC_ci95_low"], ci_row.iloc[0]["DeltaAUC_ci95_high"]]
        stable_max = float(row["DeltaAUC"])
    else:
        ci = [None, None]
        stable_max = None
    error_multi = fdr_dedup.get("rgb_error_multivariable_model", {})
    significant_terms = error_multi.get("significant_metric_names", "")
    independent_terms = [x for x in str(significant_terms).split(";") if x]
    all_max = float(pd.to_numeric(all_rows["DeltaAUC"], errors="coerce").max())
    if independent_terms and stable_max is not None and stable_max > 0.10:
        level = "high"
    elif independent_terms or (stable_max is not None and stable_max >= 0.05):
        level = "moderate"
    elif stable_max is None:
        level = "indeterminate"
    else:
        level = "low"
    return {
        "dependence_term": "image_luminance_associated_dependence",
        "dependence_label_zh": "高图像亮度相关依赖风险" if level == "high" else "图像亮度相关依赖风险",
        "dependence_level": level,
        "all_group_max_delta_auc": all_max,
        "stable_only_max_delta_auc": stable_max,
        "stable_only_max_delta_auc_ci": ci,
        "independent_error_risk_terms": independent_terms,
        "evidence_notes": [
            "Stable-only DeltaAUC is prioritized over unstable best-group contrasts.",
            "FDR hits are deduplicated by unique metric and separated into single-metric and multivariable models.",
            "Terminology is restricted to image luminance-associated dependence risk.",
        ],
        "limitations": [
            "Image luminance metrics mix illumination, exposure, camera ISP, skin tone/appearance, phenotype, and unrecorded acquisition factors.",
            "Stage2 remains observational and cannot prove environmental-light causality.",
        ],
    }


def run_statistical_repair(full500_root: Path, labels: pd.DataFrame, out_dir: Path, *, iterations: int, seed: int) -> dict[str, Any]:
    repair = out_dir / "repair"
    repair.mkdir(parents=True, exist_ok=True)
    main = pd.read_csv(full500_root / "features" / "stage2_lighting_features_500.csv")
    e0 = pd.read_csv(full500_root / "features" / "stage2_lighting_features_erosion0.csv")
    e4 = pd.read_csv(full500_root / "features" / "stage2_lighting_features_erosion4.csv")
    e2 = main[["sample_id", *EROSION_METRICS]].copy()
    features_by_erosion = {0: e0, 2: e2, 4: e4}
    assignments, cuts = foldwise_assignments(features_by_erosion, labels)
    assignments.to_csv(repair / "erosion_foldwise_assignments_repaired.csv", index=False, encoding="utf-8-sig")
    cuts.to_csv(repair / "erosion_foldwise_cutpoints_repaired.csv", index=False, encoding="utf-8-sig")
    metrics = stratified_metrics(main, assignments, iterations=iterations, seed=seed)
    metrics.to_csv(repair / "erosion_stratified_metrics_repaired.csv", index=False, encoding="utf-8-sig")
    worst = worst_group_summary(metrics)
    boot_ci, diagnostics = worst_group_bootstrap(main, assignments, metrics, iterations=iterations, seed=seed)
    worst = worst.merge(boot_ci, on=["erosion_px", "metric_name", "scope"], how="left")
    worst.to_csv(repair / "erosion_worst_group_metrics_repaired.csv", index=False, encoding="utf-8-sig")
    boot_ci.to_csv(repair / "worst_group_bootstrap_ci.csv", index=False, encoding="utf-8-sig")
    (repair / "worst_group_bootstrap_diagnostics.json").write_text(json.dumps(diagnostics, ensure_ascii=False, indent=2), encoding="utf-8")
    compare_old_repaired(full500_root, worst).to_csv(repair / "old_vs_repaired_erosion_stratification.csv", index=False, encoding="utf-8-sig")
    fdr_df, fdr_json = deduplicate_fdr_hits(full500_root)
    fdr_df.to_csv(repair / "fdr_hit_deduplication.csv", index=False, encoding="utf-8-sig")
    (repair / "fdr_hit_deduplication.json").write_text(json.dumps(fdr_json, ensure_ascii=False, indent=2), encoding="utf-8")
    revised = revised_evidence_summary(worst, boot_ci, fdr_json)
    (repair / "stage2_revised_evidence_summary.json").write_text(json.dumps(revised, ensure_ascii=False, indent=2), encoding="utf-8")
    return {
        "assignments": assignments,
        "cutpoints": cuts,
        "metrics": metrics,
        "worst": worst,
        "fdr": fdr_json,
        "revised_evidence": revised,
    }
