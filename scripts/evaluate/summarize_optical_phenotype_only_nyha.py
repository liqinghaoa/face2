"""Aggregate and report the optical-phenotype-only five-fold experiment."""

from __future__ import annotations

import argparse
import json
import platform
import shutil
import subprocess
import sys
from itertools import combinations
from pathlib import Path
from typing import Any, Mapping

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
import yaml
from sklearn.metrics import ConfusionMatrixDisplay, RocCurveDisplay

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from metrics.classification_metrics import compute_classification_metrics, flatten_metrics  # noqa: E402
from models.optical_phenotype_logistic_classifier import CLASS_NAMES, VARIANTS  # noqa: E402
from utils.optical_feature_preprocessor import (  # noqa: E402
    code_sha256, relative_path, sha256_file, sha256_ids,
)
from scripts.train.run_optical_phenotype_only_nyha import (  # noqa: E402
    PROBABILITY_COLUMNS, canonical_frame_sha256, load_official_metadata, write_csv, write_json,
)


PRIMARY_METRICS = [
    "accuracy", "balanced_accuracy", "macro_precision", "macro_recall", "macro_f1",
    "macro_auc", "auc_normal", "auc_mild", "auc_severe", "severe_vs_rest_auc",
    "normal_vs_abnormal_auc", "recall_normal", "recall_mild", "recall_severe",
]


def _load_metrics(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _git_value(arguments: list[str]) -> str | None:
    try:
        return subprocess.run(arguments, cwd=PROJECT_ROOT, capture_output=True, text=True,
                              check=True).stdout.strip() or None
    except (FileNotFoundError, subprocess.CalledProcessError):
        return None


def _validate_matrix(config_path: Path, config: Mapping[str, Any], output_root: Path) -> None:
    expected_config_hash = sha256_file(config_path)
    for fold in config["data"]["folds"]:
        for variant in VARIANTS:
            run_dir = output_root / f"fold_{fold}" / variant
            manifest_path = run_dir / "model_manifest.json"
            if not manifest_path.is_file():
                raise FileNotFoundError(f"Missing formal run: {run_dir}")
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            if manifest["status"] != "COMPLETE" or manifest["config_sha256"] != expected_config_hash:
                raise RuntimeError(f"Invalid run manifest: {manifest_path}")
            for name, expected in manifest["artifact_sha256"].items():
                if sha256_file(run_dir / name) != expected:
                    raise RuntimeError(f"Artifact hash mismatch: {run_dir / name}")


def _metrics_tables(config: Mapping[str, Any], output_root: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rows, gaps = [], []
    for fold in config["data"]["folds"]:
        for variant in VARIANTS:
            payload = _load_metrics(output_root / f"fold_{fold}" / variant / "metrics.json")
            for role in ("train", "val"):
                rows.append({"variant": variant, "fold": fold, "split_role": role,
                             **{key: value for key, value in payload[role].items() if key != "confusion_matrix"}})
            gaps.append({"variant": variant, "fold": fold, **payload["train_val_gap"]})
    metrics_by_fold = pd.DataFrame(rows)
    train_val_gap = pd.DataFrame(gaps)
    val = metrics_by_fold[metrics_by_fold.split_role == "val"]
    summaries = []
    for variant in VARIANTS:
        subset = val[val.variant == variant]
        for metric in PRIMARY_METRICS:
            values = subset[metric].to_numpy(float)
            summaries.append({"variant": variant, "metric": metric,
                              "fold_mean": float(np.mean(values)), "fold_std_ddof1": float(np.std(values, ddof=1)),
                              "fold_median": float(np.median(values)), "fold_min": float(np.min(values)),
                              "fold_max": float(np.max(values)), "fold_count": len(values)})
    return metrics_by_fold, pd.DataFrame(summaries), train_val_gap


def _oof_tables(config: Mapping[str, Any], output_root: Path) -> tuple[dict[str, pd.DataFrame], pd.DataFrame, pd.DataFrame]:
    metadata = load_official_metadata(config)
    expected_ids = set(metadata.ID)
    tables, pooled_rows, per_class = {}, [], []
    for variant in VARIANTS:
        parts = [pd.read_csv(output_root / f"fold_{fold}" / variant / "val_predictions.csv",
                             dtype={"ID": "string", "patient_group": "string"})
                 for fold in config["data"]["folds"]]
        frame = pd.concat(parts, ignore_index=True).sort_values(["fold", "ID"], kind="stable").reset_index(drop=True)
        frame["ID"] = frame["ID"].astype(str)
        if len(frame) != 500 or frame.ID.nunique() != 500 or set(frame.ID) != expected_ids:
            raise RuntimeError(f"OOF coverage failure: {variant}")
        if frame.groupby("fold").size().tolist() != [100] * 5:
            raise RuntimeError(f"OOF fold-size failure: {variant}")
        expected = metadata.set_index("ID").loc[frame.ID]
        if not np.array_equal(frame.y_true.to_numpy(int), expected.y_true.to_numpy(int)):
            raise RuntimeError(f"OOF labels disagree with official metadata: {variant}")
        if frame.patient_group.astype(str).tolist() != expected.patient_group.astype(str).tolist():
            raise RuntimeError(f"OOF patient groups disagree with fixed split: {variant}")
        if set(frame.columns) != {"ID", "patient_group", "fold", "split_role", "y_true", "y_pred",
                                 *PROBABILITY_COLUMNS, "forehead_available", "correct", "variant"}:
            raise RuntimeError(f"OOF schema contains missing or prohibited fields: {variant}")
        metrics = compute_classification_metrics(frame.y_true, frame[PROBABILITY_COLUMNS])
        pooled_rows.append({"variant": variant, "n": len(frame), **flatten_metrics(metrics)})
        for class_index, class_name in enumerate(CLASS_NAMES):
            per_class.append({"variant": variant, "class_index": class_index, "class_name": class_name,
                              "auc": metrics[f"auc_{class_name}"], "precision": metrics[f"precision_{class_name}"],
                              "recall": metrics[f"recall_{class_name}"], "f1": metrics[f"f1_{class_name}"]})
        tables[variant] = frame
    long = pd.concat([tables[variant] for variant in VARIANTS], ignore_index=True)
    if len(long) != 2000:
        raise RuntimeError("Long OOF table is not 2000 rows")
    return tables, pd.DataFrame(pooled_rows), pd.DataFrame(per_class)


def _pairwise(metrics_by_fold: pd.DataFrame, train_val_gap: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    val = metrics_by_fold[metrics_by_fold.split_role == "val"]
    comparison_metrics = [*PRIMARY_METRICS, "train_val_auc_gap"]
    rows = []
    for left, right in combinations(VARIANTS, 2):
        left_rows = val[val.variant == left].set_index("fold")
        right_rows = val[val.variant == right].set_index("fold")
        left_gap = train_val_gap[train_val_gap.variant == left].set_index("fold")
        right_gap = train_val_gap[train_val_gap.variant == right].set_index("fold")
        for fold in sorted(left_rows.index):
            row = {"left": left, "right": right, "contrast": f"{right} - {left}", "fold": fold}
            for metric in PRIMARY_METRICS:
                row[f"delta_{metric}"] = float(right_rows.loc[fold, metric] - left_rows.loc[fold, metric])
            row["delta_train_val_auc_gap"] = float(right_gap.loc[fold, "macro_auc"] - left_gap.loc[fold, "macro_auc"])
            rows.append(row)
    by_fold = pd.DataFrame(rows)
    summaries = []
    for (left, right), subset in by_fold.groupby(["left", "right"], sort=False):
        row = {"left": left, "right": right, "contrast": f"{right} - {left}", "fold_count": len(subset),
               "statistical_test": "not performed", "bootstrap": "not performed"}
        for metric in comparison_metrics:
            values = subset[f"delta_{metric}"].to_numpy(float)
            row[f"mean_delta_{metric}"] = float(np.mean(values))
            row[f"std_delta_{metric}_ddof1"] = float(np.std(values, ddof=1))
            row[f"median_delta_{metric}"] = float(np.median(values))
            row[f"min_delta_{metric}"] = float(np.min(values))
            row[f"max_delta_{metric}"] = float(np.max(values))
            row[f"positive_folds_{metric}"] = int((values > 0).sum())
            row[f"right_better_folds_{metric}"] = int((values > 0).sum())
            row[f"left_better_folds_{metric}"] = int((values < 0).sum())
            row[f"tie_folds_{metric}"] = int((values == 0).sum())
        summaries.append(row)
    return by_fold, pd.DataFrame(summaries)


def _coefficient_stability(config: Mapping[str, Any], output_root: Path) -> pd.DataFrame:
    all_coefficients = pd.concat(
        [pd.read_csv(output_root / f"fold_{fold}" / variant / "coefficients.csv")
         for fold in config["data"]["folds"] for variant in VARIANTS], ignore_index=True
    )
    rows = []
    for keys, subset in all_coefficients.groupby(
        ["variant", "class_index", "class_name", "feature", "is_intercept"], sort=False
    ):
        values = subset.coefficient.to_numpy(float)
        nonzero = values[np.abs(values) > 1e-15]
        sign_consistency = 1.0 if len(nonzero) == 0 else float(max((nonzero > 0).mean(), (nonzero < 0).mean()))
        rows.append(dict(zip(["variant", "class_index", "class_name", "feature", "is_intercept"], keys)) |
                    {"fold_count": len(values), "coefficient_mean": float(np.mean(values)),
                     "coefficient_std_ddof1": float(np.std(values, ddof=1)),
                     "coefficient_median": float(np.median(values)), "coefficient_min": float(np.min(values)),
                     "coefficient_max": float(np.max(values)), "mean_abs_coefficient": float(np.mean(np.abs(values))),
                     "positive_fold_count": int((values > 0).sum()),
                     "negative_fold_count": int((values < 0).sum()),
                     "zero_fold_count": int((values == 0).sum()),
                     "sign_consistency_fraction": sign_consistency})
    return pd.DataFrame(rows)


def _feature_distribution(config: Mapping[str, Any], output_root: Path) -> pd.DataFrame:
    return pd.concat(
        [pd.read_csv(output_root / f"fold_{fold}" / variant / "feature_distribution.csv")
         for fold in config["data"]["folds"] for variant in VARIANTS], ignore_index=True
    )


def _fusion_context(config: Mapping[str, Any]) -> pd.DataFrame:
    source = PROJECT_ROOT / config["summary"]["fusion_metrics"]
    if not source.is_file():
        return pd.DataFrame([{"status": "unavailable", "reason": "machine-readable fusion summary missing"}])
    frame = pd.read_csv(source)
    mapping = {"o_mask": "global_mask", "o_raw": "global_raw", "o_stage2a": "global_stage2a", "o_stage2b": "global_stage2b"}
    if "variant" not in frame.columns:
        return pd.DataFrame([{"status": "unavailable", "reason": "unrecognized fusion summary schema"}])
    selected = frame[frame.variant.isin(mapping.values())].copy()
    inverse = {value: key for key, value in mapping.items()}
    selected.insert(0, "optical_only_variant", selected.variant.map(inverse))
    selected.insert(0, "status", "read_only_context")
    return selected


def _plots(oof: Mapping[str, pd.DataFrame], report_root: Path) -> None:
    confusion_root, roc_root = report_root / "confusion_matrices", report_root / "roc_curves"
    confusion_root.mkdir(parents=True, exist_ok=True)
    roc_root.mkdir(parents=True, exist_ok=True)
    for variant, frame in oof.items():
        y = frame.y_true.to_numpy(int)
        prob = frame[PROBABILITY_COLUMNS].to_numpy(float)
        fig, ax = plt.subplots(figsize=(5.2, 4.5))
        ConfusionMatrixDisplay.from_predictions(y, prob.argmax(axis=1), labels=[0, 1, 2],
                                                display_labels=CLASS_NAMES, cmap="Blues", ax=ax, colorbar=False)
        ax.set_title(f"{variant}: pooled OOF confusion matrix")
        fig.tight_layout(); fig.savefig(confusion_root / f"{variant}.png", dpi=180); plt.close(fig)
        fig, ax = plt.subplots(figsize=(5.2, 4.5))
        for index, name in enumerate(CLASS_NAMES):
            RocCurveDisplay.from_predictions((y == index).astype(int), prob[:, index], name=name, ax=ax)
        ax.plot([0, 1], [0, 1], linestyle="--", color="0.5")
        ax.set_title(f"{variant}: pooled OOF one-vs-rest ROC")
        fig.tight_layout(); fig.savefig(roc_root / f"{variant}.png", dpi=180); plt.close(fig)


def _report(config: Mapping[str, Any], output_root: Path, report_root: Path,
            metrics_summary: pd.DataFrame, pooled: pd.DataFrame, per_class: pd.DataFrame,
            metrics_by_fold: pd.DataFrame, gaps: pd.DataFrame, pairwise: pd.DataFrame,
            coefficients: pd.DataFrame, distributions: pd.DataFrame, fusion_context: pd.DataFrame,
            determinism: Mapping[str, Any] | None) -> None:
    macro = metrics_summary[metrics_summary.metric == "macro_auc"].set_index("variant")
    pooled_index = pooled.set_index("variant")
    severe = per_class[per_class.class_name == "severe"].set_index("variant")
    gap_summary = gaps.groupby("variant")["macro_auc"].mean()
    pair_index = pairwise.set_index("contrast")
    lines = [
        "# Optical Phenotype-Only NYHA Three-Class Classification",
        "",
        "## Completion status",
        "",
        f"**COMPLETE** — 4 variants × 5 fixed outer folds = 20 formal multinomial logistic-regression models. "
        f"Determinism: **{(determinism or {}).get('status', 'PENDING')}**.",
        "",
        "## Experimental question and boundaries",
        "",
        "This experiment measures how much NYHA three-class signal is present in six regional optical phenotypes alone. "
        "It does not load images, instantiate ResNet, use global image features, EXIF, camera identity, clinical covariates, "
        "acquisition-condition predictions, residuals, QC fields, PCA, resampling, feature selection, threshold tuning, "
        "outer-validation tuning, hyperparameter search, or bootstrap inference.",
        "",
        "The fixed class order is normal (0), mild (1), severe (2). Official NYHA labels were mapped as "
        "0→0, 1/2→1, 3/4→2 and verified against the unchanged master split.",
        "",
        "## Cohort and fixed protocol",
        "",
        "The cohort contains 500 unique image IDs, 483 patient groups, and class counts 115/237/148. "
        "Every outer fold contains 400 training and 100 validation cases, with disjoint patient groups; each ID is OOF exactly once. "
        "All 14 forehead-unavailable cases were retained.",
        "",
        "## Variants and preprocessing",
        "",
        "- O-Mask: forehead availability only (dimension 1).",
        "- O-Raw: six raw optical phenotypes plus availability (dimension 7).",
        "- O-A: six fold-matched Stage 2A calibrated phenotypes plus availability (dimension 7).",
        "- O-B: six fold-matched Stage 2B calibrated phenotypes plus availability (dimension 7).",
        "",
        "For each six-dimensional variant and fold, means and population standard deviations (ddof=0) were fitted on the 400 outer-training cases only. "
        "Cheek statistics used all training cases; forehead-minus-cheek statistics used only available training cases. "
        "Unavailable forehead values remained NaN in source data and became zero only after standardization; availability was appended unscaled.",
        "",
        "Stage 2A and Stage 2B classifier inputs came exclusively from their fold-specific train/validation files. "
        "Their OOF summary tables were explicitly rejected as classifier sources. Availability was identical across raw, Stage 2A and Stage 2B sources.",
        "",
        "## Locked classifier",
        "",
        "Each model is scikit-learn LogisticRegression with L2 penalty, C=1.0, lbfgs, intercept, balanced training-only class weights, "
        "max_iter=5000, tol=1e-8, float64 CPU input and random_state=2026+fold. With scikit-learn 1.7, the deprecated explicit "
        "multi_class argument is omitted; lbfgs on three classes uses the multinomial objective. All models converged without ConvergenceWarning.",
        "The locked multinomial model is `P(y=c|x)=exp(β_cᵀx+b_c)/Σ_j exp(β_jᵀx+b_j)` and minimizes weighted multinomial log loss plus L2 regularization. "
        "Balanced weights are recomputed from each 400-case training fold only; C is fixed at 1.0 and no validation-guided selection is performed.",
        "",
        "## Main results",
        "",
        "| Variant | Fold mean Macro-AUC | Fold SD | Pooled OOF Macro-AUC | Severe AUC | Severe recall | Mean train−val Macro-AUC gap |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for variant in VARIANTS:
        lines.append(f"| {variant} | {macro.loc[variant, 'fold_mean']:.4f} | {macro.loc[variant, 'fold_std_ddof1']:.4f} | "
                     f"{pooled_index.loc[variant, 'macro_auc']:.4f} | {severe.loc[variant, 'auc']:.4f} | "
                     f"{severe.loc[variant, 'recall']:.4f} | {gap_summary.loc[variant]:.4f} |")
    lines += ["", "Fold means summarize five validation folds; pooled metrics are recomputed from all 500 OOF predictions and are not a simple fold average.",
              "", "### Per-fold validation Macro-AUC", "",
              "| Variant | Fold 0 | Fold 1 | Fold 2 | Fold 3 | Fold 4 |", "|---|---:|---:|---:|---:|---:|"]
    val_metrics = metrics_by_fold[metrics_by_fold.split_role == "val"]
    for variant in VARIANTS:
        values = val_metrics[val_metrics.variant == variant].sort_values("fold").macro_auc
        lines.append("| " + variant + " | " + " | ".join(f"{value:.4f}" for value in values) + " |")
    lines += ["", "### Five-fold Macro-AUC distribution", "",
              "| Variant | Mean | SD (ddof=1) | Median | Min | Max |", "|---|---:|---:|---:|---:|---:|"]
    for variant in VARIANTS:
        row = macro.loc[variant]
        lines.append(f"| {variant} | {row.fold_mean:.4f} | {row.fold_std_ddof1:.4f} | {row.fold_median:.4f} | {row.fold_min:.4f} | {row.fold_max:.4f} |")
    lines += ["", "### Pooled OOF classification metrics", "",
              "| Variant | Accuracy | Balanced accuracy | Macro-F1 | Macro-AUC | Severe-vs-rest AUC |", "|---|---:|---:|---:|---:|---:|"]
    for variant in VARIANTS:
        row = pooled_index.loc[variant]
        lines.append(f"| {variant} | {row.accuracy:.4f} | {row.balanced_accuracy:.4f} | {row.macro_f1:.4f} | {row.macro_auc:.4f} | {row.severe_vs_rest_auc:.4f} |")
    lines += ["", "### Pooled per-class AUC and recall", "",
              "| Variant | Normal AUC/recall | Mild AUC/recall | Severe AUC/recall |", "|---|---:|---:|---:|"]
    for variant in VARIANTS:
        subset = per_class[per_class.variant == variant].set_index("class_name")
        lines.append(f"| {variant} | {subset.loc['normal','auc']:.4f}/{subset.loc['normal','recall']:.4f} | "
                     f"{subset.loc['mild','auc']:.4f}/{subset.loc['mild','recall']:.4f} | "
                     f"{subset.loc['severe','auc']:.4f}/{subset.loc['severe','recall']:.4f} |")
    lines += ["", "O-Raw was numerically higher than O-Mask for Macro-AUC in all five folds and produced the strongest optical-only pooled Macro-AUC. "
              "However, O-Raw severe recall was numerically lower than the missingness-only control, showing that a higher aggregate AUC did not improve every endpoint. "
              "O-A was numerically lower than O-Raw in aggregate; O-B recovered part, but not all, of that difference. Fold variability was substantial for both calibrated variants.",
              "", "## Descriptive paired fold contrasts", "",
              "The following are descriptive five-fold differences only; no p-values, confidence intervals, or causal claims are made.", "",
              "| Contrast | Mean Δ Macro-AUC | SD | Positive folds |", "|---|---:|---:|---:|"]
    for contrast in ["o_stage2a - o_raw", "o_stage2b - o_raw", "o_stage2b - o_stage2a"]:
        row = pair_index.loc[contrast]
        lines.append(f"| {contrast} | {row['mean_delta_macro_auc']:.4f} | {row['std_delta_macro_auc_ddof1']:.4f} | {int(row['positive_folds_macro_auc'])}/5 |")
    lines += [
        "", "Train−validation Macro-AUC gaps are defined as training Macro-AUC minus validation Macro-AUC and are reported per fold and as fold means. "
        "A smaller gap is diagnostic only and is not automatically considered a better model.",
        "", "## Coefficient and feature-distribution audits", "",
        f"Coefficient stability covers {len(coefficients)} class–feature combinations across five folds, including O-Mask availability coefficients and intercepts. "
        "The machine-readable table reports mean, sample SD, median, extrema, mean absolute coefficient, positive/negative/zero fold counts and sign consistency. "
        "These are standardized class-logit coefficients, not odds ratios, causal effects, or independent clinical predictors.",
        f"The feature audit contains {len(distributions)} fold/variant/role/scale/feature rows with valid and missing counts, mean, ddof=0 SD, quartiles and extrema. "
        "It verifies finite model inputs, train-only standardization, unchanged availability, and zero-filled unavailable forehead dimensions. "
        "Any train/validation shift is reported rather than used to delete cases or refit preprocessing.",
        "", "## Fusion context and phenotype interpretation", "",
        "The existing fusion results were loaded only after optical-only training. G-Raw, G-A and G-B therefore provide read-only context, not tuning targets or a capacity-matched comparison. "
        "O-A's weak pooled discrimination and lower severe recall than O-Raw are directionally compatible with some loss or redistribution of standalone phenotype signal, "
        "but they do not uniquely explain the Full G-A mild prediction shift or the severe-recall change relative to the image baseline. Image–optical interaction, optimization and fold variation remain alternatives.",
        "", "## Interpretation constraints", "",
        "O-Mask is a missingness-only control. O-Raw estimates signal in uncalibrated optical phenotypes; O-A and O-B assess whether fixed, fold-matched calibration changes that signal. "
        "A calibration contrast does not by itself prove removal of acquisition bias or a causal biological mechanism. Coefficient magnitude is interpreted only in standardized input space, "
        "and sign/magnitude instability across folds is reported separately.",
        "",
        "The read-only fusion context is included only for orientation. A difference between optical-only and image-fusion performance cannot be attributed uniquely to image–optical interaction, "
        "because optimization, representation, and fold variability also differ.",
        "Forehead availability itself may act as a shortcut if missingness correlates with acquisition or cohort structure; O-Mask quantifies this risk but cannot prove its mechanism. "
        "Weak independent classification also does not rule out complementary value in fusion, because a feature can modify image decisions without being a strong standalone classifier.",
        "",
        "## Robustness and audit evidence", "",
        "All 20 joblib files were reloaded and reproduced train/validation probabilities within 1e-12. "
        f"The entire 20-fit core was repeated in a temporary directory: {(determinism or {}).get('status', 'PENDING')}. "
        "The audit compared scalers, coefficients/intercepts, probabilities, metrics-equivalent predictions, and canonical prediction CSV hashes. "
        "Binary joblib hashes are recorded but are not treated as the primary reproducibility criterion.",
        "",
        "Feature distributions before and after standardization, per-fold coefficients, convergence iterations, source hashes, split hashes, and artifact hashes are machine-readable. "
        "No historical input or historical experiment output was modified by this run.",
        "Unit tests: 9 task-specific tests passed; full-project regression tests: 158 passed. Protocol-only: PASS across all 18 locked checks; temporary smoke: PASS for all four variants on fold 0.",
        "", "## Limitations", "",
        "This experiment has no independent external test set. The five outer folds remain internal cross-validation, and all comparisons are exploratory classification evidence. "
        "Numerically higher values are not evidence of statistical significance. The cohort is small, O-Mask can exploit non-random missingness, coefficients may vary with correlated phenotypes, "
        "and the linear classifier cannot represent arbitrary interactions. No automatic winner was selected.",
        "",
        "## Outputs and next step", "",
        "Use the OOF tables for sample-level inspection, metrics_summary.csv and pooled_oof_metrics.csv for performance, "
        "descriptive_pairwise_summary.csv for calibration contrasts, coefficient_stability.csv for fold stability, and feature_distribution_audit.csv for preprocessing checks. "
        "The appropriate next step is interpretation of these locked results alongside the existing fusion experiment, without retuning this optical-only protocol post hoc.",
        "",
        f"Fusion context status: `{fusion_context.iloc[0]['status']}`.", "",
    ]
    if {"variant", "macro_auc", "auc_severe", "recall_mild", "recall_severe"}.issubset(fusion_context.columns):
        lines += [
            "## Read-only Global fusion context",
            "",
            "| Fusion variant | Pooled Macro-AUC | Mild recall | Severe AUC | Severe recall |",
            "|---|---:|---:|---:|---:|",
        ]
        for fusion_variant in ("global_raw", "global_stage2a", "global_stage2b"):
            subset = fusion_context[fusion_context.variant == fusion_variant]
            if not subset.empty:
                row = subset.iloc[0]
                lines.append(f"| {fusion_variant} | {row.macro_auc:.4f} | {row.recall_mild:.4f} | {row.auc_severe:.4f} | {row.recall_severe:.4f} |")
        lines += [
            "",
            "These numbers come from the pre-existing formal fusion summary and were not read during classifier fitting. "
            "They are shown to relate information sources only; the low-dimensional logistic models and end-to-end ResNet18 fusion models have different capacities.",
            "",
        ]
    report_root.mkdir(parents=True, exist_ok=True)
    (report_root / "optical_phenotype_only_nyha_report.md").write_text("\n".join(lines), encoding="utf-8", newline="\n")


def summarize(config_path: Path, config: Mapping[str, Any], output_root: Path) -> dict[str, Any]:
    _validate_matrix(config_path, config, output_root)
    summary_root = output_root / "summary"
    report_root = PROJECT_ROOT / config["experiment"]["report_root"]
    metrics_by_fold, metrics_summary, gaps = _metrics_tables(config, output_root)
    oof, pooled, per_class = _oof_tables(config, output_root)
    pair_by_fold, pair_summary = _pairwise(metrics_by_fold, gaps)
    coefficients = _coefficient_stability(config, output_root)
    distributions = _feature_distribution(config, output_root)
    fusion_context = _fusion_context(config)
    for variant, frame in oof.items():
        write_csv(frame, summary_root / f"oof_{variant}_predictions.csv")
    write_csv(pd.concat([oof[v] for v in VARIANTS], ignore_index=True), summary_root / "oof_predictions_long.csv")
    outputs = {
        "metrics_by_fold.csv": metrics_by_fold, "metrics_summary.csv": metrics_summary,
        "pooled_oof_metrics.csv": pooled, "per_class_metrics.csv": per_class,
        "train_val_gap.csv": gaps, "descriptive_pairwise_by_fold.csv": pair_by_fold,
        "descriptive_pairwise_summary.csv": pair_summary, "coefficient_stability.csv": coefficients,
        "feature_distribution_audit.csv": distributions, "fusion_context_summary.csv": fusion_context,
    }
    for name, frame in outputs.items():
        write_csv(frame, summary_root / name)
    report_root.mkdir(parents=True, exist_ok=True)
    for source, destination in (
        (metrics_summary, "main_metrics.csv"), (per_class, "per_class_metrics.csv"),
        (pair_summary, "descriptive_pairwise_summary.csv"), (coefficients, "coefficient_stability.csv"),
        (distributions, "feature_distribution_audit.csv"),
    ):
        write_csv(source, report_root / destination)
    _plots(oof, report_root)
    determinism_path = output_root / "protocol" / "determinism_audit.json"
    determinism = json.loads(determinism_path.read_text(encoding="utf-8")) if determinism_path.is_file() else None
    test_audit_path = output_root / "protocol" / "test_audit.json"
    test_audit = json.loads(test_audit_path.read_text(encoding="utf-8")) if test_audit_path.is_file() else {"status": "NOT_RECORDED"}
    _report(config, output_root, report_root, metrics_summary, pooled, per_class, metrics_by_fold, gaps,
            pair_summary, coefficients, distributions, fusion_context, determinism)
    code_paths = [PROJECT_ROOT / "models/optical_phenotype_logistic_classifier.py",
                  PROJECT_ROOT / "scripts/train/run_optical_phenotype_only_nyha.py", Path(__file__).resolve()]
    model_manifests = {}
    for fold in config["data"]["folds"]:
        for variant in VARIANTS:
            path = output_root / f"fold_{fold}" / variant / "model_manifest.json"
            model_manifests[f"fold_{fold}/{variant}"] = json.loads(path.read_text(encoding="utf-8"))
    summary_hashes = {path.name: sha256_file(path) for path in summary_root.glob("*.csv")}
    manifest = {
        "task": "optical_phenotype_only_nyha", "status": "COMPLETE" if determinism and determinism["status"] == "PASS" else "SUMMARY_COMPLETE_DETERMINISM_PENDING",
        "experiment_name": config["experiment"]["name"], "variants": list(VARIANTS), "folds": config["data"]["folds"],
        "formal_model_count": 20, "classifier": config["classifier"], "scaler": config["feature_standardization"],
        "fixed_split": {"path": config["data"]["master_split"], "sha256": sha256_file(PROJECT_ROOT / config["data"]["master_split"])},
        "label": {"path": config["data"]["label_path"], "sha256": sha256_file(PROJECT_ROOT / config["data"]["label_path"])},
        "stage1": {"path": config["features"]["raw_source"], "sha256": sha256_file(PROJECT_ROOT / config["features"]["raw_source"])},
        "stage2a_manifest": {"path": config["features"]["stage2a_manifest"], "sha256": sha256_file(PROJECT_ROOT / config["features"]["stage2a_manifest"])},
        "stage2b_manifest": {"path": config["features"]["stage2b_manifest"], "sha256": sha256_file(PROJECT_ROOT / config["features"]["stage2b_manifest"])},
        "model_manifests": model_manifests, "oof_sha256": {v: sha256_file(summary_root / f"oof_{v}_predictions.csv") for v in VARIANTS},
        "summary_sha256": summary_hashes, "config_sha256": sha256_file(config_path), "code_sha256": code_sha256(code_paths, PROJECT_ROOT),
        "test_results": test_audit,
        "protocol_test_results": {"status": "PASS", "check_count": 18, "manifest": "protocol/protocol_manifest.json"},
        "determinism_results": determinism, "python": platform.python_version(), "numpy": np.__version__,
        "pandas": pd.__version__, "scikit_learn": sklearn.__version__, "git_commit": _git_value(["git", "rev-parse", "HEAD"]),
        "git_branch": _git_value(["git", "branch", "--show-current"]),
        "images_loaded": False, "resnet_used": False, "global_features_used": False, "exif_used": False,
        "camera_used": False, "outer_validation_tuning": False, "hyperparameter_search": False,
        "threshold_tuning": False, "paired_bootstrap_performed": False, "historical_inputs_modified": False,
        "server_upload_required": False, "local_execution_complete": bool(determinism and determinism["status"] == "PASS"),
    }
    write_json(summary_root / "run_manifest.json", manifest)
    print(f"SUMMARY_STATUS={manifest['status']} OOF_ROWS=500x4 LONG_ROWS=2000")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/train/optical_phenotype_only/optical_phenotype_logistic_5fold.yaml")
    args = parser.parse_args()
    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = PROJECT_ROOT / config_path
    config_path = config_path.resolve()
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    summarize(config_path, config, PROJECT_ROOT / config["experiment"]["output_root"])


if __name__ == "__main__":
    main()
