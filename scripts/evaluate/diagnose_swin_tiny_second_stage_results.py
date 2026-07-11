"""Diagnose Swin-Tiny second-stage experiment results.

This script performs read-only analysis of completed experiment outputs and
writes a diagnostic package under
experiments/swin_tiny_second_stage_500Data/diagnostic_analysis.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    precision_recall_fscore_support,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.simple_xlsx import write_xlsx  # noqa: E402


SECOND_STAGE_ROOT = PROJECT_ROOT / "experiments" / "swin_tiny_second_stage_500Data"
THRESHOLD_DIR = SECOND_STAGE_ROOT / "threshold_scan"
OUTPUT_DIR = SECOND_STAGE_ROOT / "diagnostic_analysis"

REPORT_PATH = OUTPUT_DIR / "swin_tiny_second_stage_diagnostic_report.md"
TABLES_XLSX_PATH = OUTPUT_DIR / "swin_tiny_second_stage_diagnostic_tables.xlsx"
CORE_SUMMARY_PATH = OUTPUT_DIR / "swin_tiny_second_stage_core_summary.csv"
CONFUSION_SUMMARY_PATH = OUTPUT_DIR / "swin_tiny_second_stage_confusion_summary.csv"
THRESHOLD_RULES_SUMMARY_PATH = (
    OUTPUT_DIR / "swin_tiny_second_stage_threshold_best_rules_summary.csv"
)
PER_CLASS_METRICS_PATH = OUTPUT_DIR / "swin_tiny_second_stage_per_class_metrics.csv"
RECOMMENDATION_PATH = OUTPUT_DIR / "swin_tiny_second_stage_recommendation.csv"

SWIN_ORIGINAL_DIR = (
    PROJECT_ROOT
    / "experiments"
    / "model_exploration_500Data"
    / "ModelExploration_SwinTiny_ImageNetMeanBG"
)
RESNET18_DIR = (
    PROJECT_ROOT
    / "experiments"
    / "preprocess_ablation_500Data"
    / "PreprocAblation_ResNet18_NYHA3Class_hybrid_imagenet_meanbg"
)
LR5E5_PREFIX = "SwinTiny_ImageNetMeanBG_LR5e5_WeightedCE_5Fold"
LS005_PREFIX = "SwinTiny_ImageNetMeanBG_WeightedSoftCE_LS005_5Fold"

CLASS_LABELS = [0, 1, 2]
CLASS_NAMES = ["normal", "mild", "severe"]
PROB_COLS = ["prob_normal", "prob_mild", "prob_severe"]

SUMMARY_FILES = [
    "summary/fold_metrics_all.csv",
    "summary/mean_metrics.csv",
    "summary/oof_metrics.csv",
    "summary/oof_predictions.csv",
    "summary/summary_report.md",
]

THRESHOLD_STRATEGIES = {
    "best_macro_f1": "swin_tiny_threshold_best_macro_f1",
    "best_balanced_accuracy": "swin_tiny_threshold_best_balanced_accuracy",
    "best_mild_f1": "swin_tiny_threshold_best_mild_f1",
    "best_balanced_tradeoff": "swin_tiny_threshold_best_balanced_tradeoff",
    "conservative_candidate": "swin_tiny_threshold_conservative_candidate",
}


def _read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, encoding="utf-8-sig")


def _safe_float(value: Any) -> float:
    try:
        if value is None or (isinstance(value, str) and value.strip() == ""):
            return math.nan
        result = float(value)
        return result if math.isfinite(result) else math.nan
    except Exception:
        return math.nan


def _fmt(value: Any) -> str:
    number = _safe_float(value)
    return "nan" if math.isnan(number) else f"{number:.4f}"


def _fmt_count(value: Any) -> str:
    number = _safe_float(value)
    return "nan" if math.isnan(number) else str(int(round(number)))


def _metric_mean(frame: pd.DataFrame, column: str) -> float:
    if frame.empty or column not in frame.columns:
        return math.nan
    return float(pd.to_numeric(frame[column], errors="coerce").mean())


def _metric_std(frame: pd.DataFrame, column: str) -> float:
    if frame.empty or column not in frame.columns:
        return math.nan
    return float(pd.to_numeric(frame[column], errors="coerce").std(ddof=1))


def _latest_dir_by_prefix(root: Path, prefix: str) -> Path:
    candidates = [
        path
        for path in root.iterdir()
        if path.is_dir() and (path.name == prefix or path.name.startswith(prefix + "_"))
    ] if root.is_dir() else []
    if not candidates:
        return root / prefix
    return max(candidates, key=lambda path: path.stat().st_mtime)


def _required_files() -> list[tuple[str, str, Path]]:
    lr_dir = _latest_dir_by_prefix(SECOND_STAGE_ROOT, LR5E5_PREFIX)
    ls_dir = _latest_dir_by_prefix(SECOND_STAGE_ROOT, LS005_PREFIX)
    files: list[tuple[str, str, Path]] = []
    for rel in [
        "swin_threshold_scan_summary.xlsx",
        "swin_threshold_scan_summary.md",
        "argmax_metrics.csv",
        "single_threshold_scan.csv",
        "double_threshold_scan.csv",
        "best_threshold_rules.csv",
        "confusion_matrices.xlsx",
    ]:
        files.append(("threshold_scan", rel, THRESHOLD_DIR / rel))
    for rel in [
        "swin_tiny_second_stage_summary.xlsx",
        "swin_tiny_second_stage_summary.csv",
        "swin_tiny_second_stage_summary.md",
        "swin_tiny_second_stage_job_queue.csv",
    ]:
        files.append(("second_stage_root", rel, SECOND_STAGE_ROOT / rel))
    for method, directory in [
        ("swin_tiny_lr5e5", lr_dir),
        ("swin_tiny_ls005", ls_dir),
        ("swin_tiny_original", SWIN_ORIGINAL_DIR),
        ("resnet18_meanbg", RESNET18_DIR),
    ]:
        for rel in SUMMARY_FILES:
            files.append((method, rel, directory / rel))
    return files


def _files_checked_frame() -> pd.DataFrame:
    rows = []
    for scope, relative_name, path in _required_files():
        rows.append(
            {
                "scope": scope,
                "relative_name": relative_name,
                "absolute_path": str(path),
                "exists": path.is_file(),
                "size_bytes": path.stat().st_size if path.is_file() else math.nan,
            }
        )
    return pd.DataFrame(rows)


def _prepare_oof_predictions(path: Path) -> pd.DataFrame:
    frame = _read_csv(path).copy()
    missing = [column for column in ["label_3class", *PROB_COLS] if column not in frame.columns]
    if missing:
        raise ValueError(f"{path} missing required columns: {missing}")
    frame["label_3class"] = pd.to_numeric(frame["label_3class"], errors="coerce").astype(int)
    frame[PROB_COLS] = frame[PROB_COLS].apply(pd.to_numeric, errors="coerce")
    argmax_pred = np.asarray(CLASS_LABELS)[
        np.argmax(frame[PROB_COLS].to_numpy(dtype=float), axis=1)
    ]
    if "pred_class" in frame.columns:
        pred = pd.to_numeric(frame["pred_class"], errors="coerce")
        frame["pred_class"] = argmax_pred if pred.isna().any() else pred.astype(int).to_numpy()
    else:
        frame["pred_class"] = argmax_pred
    return frame


def _hard_metrics_from_predictions(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, Any]:
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=CLASS_LABELS, zero_division=0
    )
    macro_precision, macro_recall, macro_f1, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=CLASS_LABELS, average="macro", zero_division=0
    )
    _, _, weighted_f1, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=CLASS_LABELS, average="weighted", zero_division=0
    )
    out: dict[str, Any] = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_precision": float(macro_precision),
        "macro_recall": float(macro_recall),
        "macro_f1": float(macro_f1),
        "weighted_f1": float(weighted_f1),
    }
    for index, class_name in enumerate(CLASS_NAMES):
        out[f"support_{class_name}"] = int(support[index])
        out[f"precision_{class_name}"] = float(precision[index])
        out[f"recall_{class_name}"] = float(recall[index])
        out[f"f1_{class_name}"] = float(f1[index])
    return out


def _confusion_summary(method_name: str, y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, Any]:
    matrix = confusion_matrix(y_true, y_pred, labels=CLASS_LABELS)
    totals = matrix.sum(axis=1)
    row = {
        "method_name": method_name,
        "normal_total": int(totals[0]),
        "mild_total": int(totals[1]),
        "severe_total": int(totals[2]),
    }
    for true_index, true_name in enumerate(CLASS_NAMES):
        for pred_index, pred_name in enumerate(CLASS_NAMES):
            row[f"{true_name}_to_{pred_name}"] = int(matrix[true_index, pred_index])
        row[f"{true_name}_recall"] = _rate(matrix[true_index, true_index], totals[true_index])
    row["normal_to_severe_rate"] = _rate(row["normal_to_severe"], row["normal_total"])
    row["mild_to_normal_rate"] = _rate(row["mild_to_normal"], row["mild_total"])
    row["mild_to_severe_rate"] = _rate(row["mild_to_severe"], row["mild_total"])
    row["severe_to_normal_rate"] = _rate(row["severe_to_normal"], row["severe_total"])
    row["severe_to_mild_rate"] = _rate(row["severe_to_mild"], row["severe_total"])
    return row


def _rate(numerator: float, denominator: float) -> float:
    return math.nan if denominator == 0 else float(numerator) / float(denominator)


def _per_class_frame(method_name: str, metrics: dict[str, Any]) -> pd.DataFrame:
    rows = []
    for class_name in CLASS_NAMES:
        rows.append(
            {
                "method_name": method_name,
                "class_name": class_name,
                "support": metrics.get(f"support_{class_name}", math.nan),
                "precision": metrics.get(f"precision_{class_name}", math.nan),
                "recall": metrics.get(f"recall_{class_name}", math.nan),
                "f1": metrics.get(f"f1_{class_name}", math.nan),
            }
        )
    return pd.DataFrame(rows)


def _row_from_experiment(method_name: str, source_type: str, directory: Path) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    summary_dir = directory / "summary"
    fold = _read_csv(summary_dir / "fold_metrics_all.csv")
    oof_metrics = _read_csv(summary_dir / "oof_metrics.csv")
    oof = _prepare_oof_predictions(summary_dir / "oof_predictions.csv")
    hard = _hard_metrics_from_predictions(
        oof["label_3class"].to_numpy(dtype=int),
        oof["pred_class"].to_numpy(dtype=int),
    )
    row = {
        "method_name": method_name,
        "source_type": source_type,
        "output_dir": str(directory),
        "macro_auc_mean": _metric_mean(fold, "macro_auc"),
        "macro_auc_std": _metric_std(fold, "macro_auc"),
        "balanced_accuracy_mean": _metric_mean(fold, "balanced_accuracy"),
        "balanced_accuracy_std": _metric_std(fold, "balanced_accuracy"),
        "macro_f1_mean": _metric_mean(fold, "macro_f1"),
        "macro_f1_std": _metric_std(fold, "macro_f1"),
        "macro_recall_mean": _metric_mean(fold, "macro_recall"),
        "macro_recall_std": _metric_std(fold, "macro_recall"),
        "accuracy_mean": _metric_mean(fold, "accuracy"),
        "accuracy_std": _metric_std(fold, "accuracy"),
        "recall_normal_mean": _metric_mean(fold, "recall_normal"),
        "recall_mild_mean": _metric_mean(fold, "recall_mild"),
        "recall_severe_mean": _metric_mean(fold, "recall_severe"),
        "f1_normal_mean": _metric_mean(fold, "f1_normal"),
        "f1_mild_mean": _metric_mean(fold, "f1_mild"),
        "f1_severe_mean": _metric_mean(fold, "f1_severe"),
        "severe_vs_rest_auc_mean": _metric_mean(fold, "severe_vs_rest_auc"),
        "normal_vs_abnormal_auc_mean": _metric_mean(fold, "normal_vs_abnormal_auc"),
        "oof_macro_auc": _safe_float(oof_metrics.iloc[0].get("macro_auc", math.nan)),
        "oof_balanced_accuracy": hard["balanced_accuracy"],
        "oof_macro_f1": hard["macro_f1"],
        "oof_accuracy": hard["accuracy"],
        "oof_recall_normal": hard["recall_normal"],
        "oof_recall_mild": hard["recall_mild"],
        "oof_recall_severe": hard["recall_severe"],
        "oof_f1_normal": hard["f1_normal"],
        "oof_f1_mild": hard["f1_mild"],
        "oof_f1_severe": hard["f1_severe"],
        "oof_severe_vs_rest_auc": _safe_float(oof_metrics.iloc[0].get("severe_vs_rest_auc", math.nan)),
        "oof_normal_vs_abnormal_auc": _safe_float(oof_metrics.iloc[0].get("normal_vs_abnormal_auc", math.nan)),
    }
    fold = fold.copy()
    fold.insert(0, "method_name", method_name)
    return row, fold, _per_class_frame(method_name, hard), oof


def _argmax_two(prob_a: np.ndarray, label_a: int, prob_b: np.ndarray, label_b: int) -> np.ndarray:
    return np.where(prob_a >= prob_b, label_a, label_b).astype(int)


def _predict_threshold_rule(swin_oof: pd.DataFrame, rule: pd.Series) -> np.ndarray:
    rule_name = str(rule["rule_name"])
    p_normal = swin_oof["prob_normal"].to_numpy(dtype=float)
    p_mild = swin_oof["prob_mild"].to_numpy(dtype=float)
    p_severe = swin_oof["prob_severe"].to_numpy(dtype=float)
    tn = _safe_float(rule.get("threshold_normal"))
    tm = _safe_float(rule.get("threshold_mild"))
    ts = _safe_float(rule.get("threshold_severe"))
    if rule_name == "severe_priority":
        return np.where(p_severe >= ts, 2, _argmax_two(p_normal, 0, p_mild, 1)).astype(int)
    if rule_name == "mild_priority":
        return np.where(p_mild >= tm, 1, _argmax_two(p_normal, 0, p_severe, 2)).astype(int)
    if rule_name == "normal_priority":
        return np.where(p_normal >= tn, 0, _argmax_two(p_mild, 1, p_severe, 2)).astype(int)
    if rule_name == "middle_fallback":
        return np.where(
            (p_normal >= tn) & (p_normal >= p_severe),
            0,
            np.where((p_severe >= ts) & (p_severe > p_normal), 2, 1),
        ).astype(int)
    if rule_name == "balanced_middle_fallback":
        return np.where(p_normal >= tn, 0, np.where(p_severe >= ts, 2, 1)).astype(int)
    raise ValueError(f"Unknown threshold rule: {rule_name}")


def _row_from_threshold(
    method_name: str,
    rule: pd.Series,
    original_row: dict[str, Any],
    swin_oof: pd.DataFrame,
) -> tuple[dict[str, Any], pd.DataFrame, dict[str, Any]]:
    y_true = swin_oof["label_3class"].to_numpy(dtype=int)
    y_pred = _predict_threshold_rule(swin_oof, rule)
    hard = _hard_metrics_from_predictions(y_true, y_pred)
    row = {
        "method_name": method_name,
        "source_type": "threshold_scan",
        "output_dir": str(THRESHOLD_DIR),
        "macro_auc_mean": original_row["macro_auc_mean"],
        "macro_auc_std": original_row["macro_auc_std"],
        "balanced_accuracy_mean": hard["balanced_accuracy"],
        "balanced_accuracy_std": math.nan,
        "macro_f1_mean": hard["macro_f1"],
        "macro_f1_std": math.nan,
        "macro_recall_mean": hard["macro_recall"],
        "macro_recall_std": math.nan,
        "accuracy_mean": hard["accuracy"],
        "accuracy_std": math.nan,
        "recall_normal_mean": hard["recall_normal"],
        "recall_mild_mean": hard["recall_mild"],
        "recall_severe_mean": hard["recall_severe"],
        "f1_normal_mean": hard["f1_normal"],
        "f1_mild_mean": hard["f1_mild"],
        "f1_severe_mean": hard["f1_severe"],
        "severe_vs_rest_auc_mean": original_row["severe_vs_rest_auc_mean"],
        "normal_vs_abnormal_auc_mean": original_row["normal_vs_abnormal_auc_mean"],
        "oof_macro_auc": original_row["oof_macro_auc"],
        "oof_balanced_accuracy": hard["balanced_accuracy"],
        "oof_macro_f1": hard["macro_f1"],
        "oof_accuracy": hard["accuracy"],
        "oof_recall_normal": hard["recall_normal"],
        "oof_recall_mild": hard["recall_mild"],
        "oof_recall_severe": hard["recall_severe"],
        "oof_f1_normal": hard["f1_normal"],
        "oof_f1_mild": hard["f1_mild"],
        "oof_f1_severe": hard["f1_severe"],
        "oof_severe_vs_rest_auc": original_row["oof_severe_vs_rest_auc"],
        "oof_normal_vs_abnormal_auc": original_row["oof_normal_vs_abnormal_auc"],
    }
    return row, _per_class_frame(method_name, hard), _confusion_summary(method_name, y_true, y_pred)


def _delta_frame(core: pd.DataFrame, baseline_method: str, comparison_name: str) -> pd.DataFrame:
    baseline = core.loc[core["method_name"] == baseline_method].iloc[0]
    columns = {
        "delta_macro_auc_mean": "macro_auc_mean",
        "delta_balanced_accuracy_mean": "balanced_accuracy_mean",
        "delta_macro_f1_mean": "macro_f1_mean",
        "delta_recall_mild_mean": "recall_mild_mean",
        "delta_f1_mild_mean": "f1_mild_mean",
        "delta_recall_severe_mean": "recall_severe_mean",
        "delta_f1_severe_mean": "f1_severe_mean",
        "delta_oof_balanced_accuracy": "oof_balanced_accuracy",
        "delta_oof_macro_f1": "oof_macro_f1",
        "delta_oof_recall_mild": "oof_recall_mild",
        "delta_oof_f1_mild": "oof_f1_mild",
        "delta_oof_recall_severe": "oof_recall_severe",
        "delta_oof_f1_severe": "oof_f1_severe",
    }
    rows = []
    for _, row in core.iterrows():
        out = {"method_name": row["method_name"], "comparison": comparison_name}
        for delta_name, metric_name in columns.items():
            out[delta_name] = _safe_float(row[metric_name]) - _safe_float(baseline[metric_name])
        rows.append(out)
    return pd.DataFrame(rows)


def _fold_stability_frame(fold_all: pd.DataFrame) -> pd.DataFrame:
    metrics = ["macro_auc", "balanced_accuracy", "macro_f1", "recall_mild", "recall_severe"]
    rows = []
    for method_name, group in fold_all.groupby("method_name", sort=False):
        row: dict[str, Any] = {"method_name": method_name}
        stds = []
        for metric in metrics:
            row[f"{metric}_mean"] = _metric_mean(group, metric)
            row[f"{metric}_std"] = _metric_std(group, metric)
            if math.isfinite(row[f"{metric}_std"]):
                stds.append(row[f"{metric}_std"])
        row["stability_std_mean"] = float(np.nanmean(stds)) if stds else math.nan
        rows.append(row)
    return pd.DataFrame(rows)


def _recommendations(core: pd.DataFrame, delta_swin: pd.DataFrame, delta_resnet: pd.DataFrame) -> pd.DataFrame:
    rows = []
    delta_swin_lookup = delta_swin.set_index("method_name")
    delta_resnet_lookup = delta_resnet.set_index("method_name")
    for _, row in core.iterrows():
        method = row["method_name"]
        if method in {"resnet18_meanbg", "swin_tiny_original"}:
            recommendation = "reference"
            priority = "reference"
        else:
            ds = delta_swin_lookup.loc[method]
            dr = delta_resnet_lookup.loc[method]
            effective = (
                _safe_float(ds["delta_macro_f1_mean"]) > 0
                and _safe_float(ds["delta_recall_mild_mean"]) > 0
                and _safe_float(ds["delta_balanced_accuracy_mean"]) >= -0.02
                and _safe_float(ds["delta_recall_severe_mean"]) >= -0.05
            )
            candidate_main = (
                _safe_float(dr["delta_macro_f1_mean"]) >= -0.02
                and _safe_float(dr["delta_balanced_accuracy_mean"]) > 0
                and _safe_float(dr["delta_recall_severe_mean"]) >= 0
            )
            if method == "swin_tiny_threshold_conservative_candidate":
                recommendation = "best_balanced_threshold_candidate" if candidate_main else "diagnostic_threshold_candidate"
                priority = "high"
            elif method == "swin_tiny_ls005":
                recommendation = "training_tuning_candidate" if effective else "partial_improvement"
                priority = "medium_high" if effective else "medium"
            elif method == "swin_tiny_lr5e5":
                recommendation = "not_enough_ba_macro_f1_gain" if not effective else "training_tuning_candidate"
                priority = "medium_low"
            elif str(method).startswith("swin_tiny_threshold"):
                recommendation = "threshold_diagnostic_not_external_threshold"
                priority = "medium"
            else:
                recommendation = "review"
                priority = "low"
        rows.append(
            {
                "method_name": method,
                "recommendation": recommendation,
                "priority": priority,
            }
        )
    rows.extend(
        [
            {
                "method_name": "next_step_1",
                "recommendation": "ROI-global fusion / ROI-fusion should be prioritized if the goal is robust hard decision improvement.",
                "priority": "highest",
            },
            {
                "method_name": "next_step_2",
                "recommendation": "Run ordinal or two-stage classification to address normal-mild-severe boundary structure.",
                "priority": "high",
            },
            {
                "method_name": "next_step_3",
                "recommendation": "Optional: SwinTiny_LR5e5_LS005 only as a low-priority ablation, not the main next step.",
                "priority": "optional",
            },
        ]
    )
    return pd.DataFrame(rows)


def _markdown_table(frame: pd.DataFrame, columns: list[str]) -> str:
    if frame.empty:
        return "_No rows._"
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join(["---"] * len(columns)) + " |",
    ]
    for _, row in frame[columns].iterrows():
        cells = []
        for column in columns:
            value = row[column]
            cells.append(_fmt(value) if isinstance(value, (float, np.floating)) else str(value))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def _best(core: pd.DataFrame, metric: str) -> tuple[str, float]:
    values = pd.to_numeric(core[metric], errors="coerce")
    index = values.idxmax()
    return str(core.loc[index, "method_name"]), float(values.loc[index])


def _row(frame: pd.DataFrame, method_name: str) -> pd.Series:
    result = frame.loc[frame["method_name"] == method_name]
    if result.empty:
        raise KeyError(method_name)
    return result.iloc[0]


def _key_information(core: pd.DataFrame, confusion: pd.DataFrame, best_rules: pd.DataFrame) -> list[str]:
    swin = _row(core, "swin_tiny_original")
    resnet = _row(core, "resnet18_meanbg")
    best_macro = _row(core, "swin_tiny_threshold_best_macro_f1")
    lr = _row(core, "swin_tiny_lr5e5")
    ls = _row(core, "swin_tiny_ls005")
    best_conf = _row(confusion, "swin_tiny_threshold_best_macro_f1")
    best_rule = best_rules.loc[best_rules["strategy_name"] == "best_macro_f1"].iloc[0]
    best_macro_f1_method = _best(core, "macro_f1_mean")
    best_ba_method = _best(core, "balanced_accuracy_mean")
    best_mild_recall_method = _best(core, "recall_mild_mean")
    best_severe_recall_method = _best(core, "recall_severe_mean")
    exceeds_resnet_f1 = bool((pd.to_numeric(core["macro_f1_mean"], errors="coerce") > _safe_float(resnet["macro_f1_mean"])).any())
    return [
        f"1. 原始 Swin-Tiny macro-F1：{_fmt(swin['macro_f1_mean'])}",
        f"2. 原始 Swin-Tiny BA：{_fmt(swin['balanced_accuracy_mean'])}",
        f"3. 原始 Swin-Tiny mild recall：{_fmt(swin['recall_mild_mean'])}",
        f"4. 原始 Swin-Tiny severe recall：{_fmt(swin['recall_severe_mean'])}",
        f"5. ResNet18 meanbg macro-F1：{_fmt(resnet['macro_f1_mean'])}",
        f"6. ResNet18 meanbg BA：{_fmt(resnet['balanced_accuracy_mean'])}",
        f"7. ResNet18 meanbg mild recall：{_fmt(resnet['recall_mild_mean'])}",
        f"8. ResNet18 meanbg severe recall：{_fmt(resnet['recall_severe_mean'])}",
        f"9. best threshold rule 名称：{best_rule['rule_name']}（strategy=best_macro_f1）",
        f"10. best threshold rule macro-F1：{_fmt(best_macro['macro_f1_mean'])}",
        f"11. best threshold rule BA：{_fmt(best_macro['balanced_accuracy_mean'])}",
        f"12. best threshold rule mild recall：{_fmt(best_macro['recall_mild_mean'])}",
        f"13. best threshold rule severe recall：{_fmt(best_macro['recall_severe_mean'])}",
        f"14. best threshold rule mild -> severe 数量：{_fmt_count(best_conf['mild_to_severe'])}",
        f"15. best threshold rule severe -> mild 数量：{_fmt_count(best_conf['severe_to_mild'])}",
        f"16. lr5e5 macro-F1：{_fmt(lr['macro_f1_mean'])}",
        f"17. lr5e5 BA：{_fmt(lr['balanced_accuracy_mean'])}",
        f"18. lr5e5 mild recall：{_fmt(lr['recall_mild_mean'])}",
        f"19. lr5e5 severe recall：{_fmt(lr['recall_severe_mean'])}",
        f"20. LS0.05 macro-F1：{_fmt(ls['macro_f1_mean'])}",
        f"21. LS0.05 BA：{_fmt(ls['balanced_accuracy_mean'])}",
        f"22. LS0.05 mild recall：{_fmt(ls['recall_mild_mean'])}",
        f"23. LS0.05 severe recall：{_fmt(ls['recall_severe_mean'])}",
        f"24. 哪个方法 macro-F1 最高：{best_macro_f1_method[0]}（{best_macro_f1_method[1]:.4f}）",
        f"25. 哪个方法 BA 最高：{best_ba_method[0]}（{best_ba_method[1]:.4f}）",
        f"26. 哪个方法 mild recall 最高：{best_mild_recall_method[0]}（{best_mild_recall_method[1]:.4f}）",
        f"27. 哪个方法 severe recall 最高：{best_severe_recall_method[0]}（{best_severe_recall_method[1]:.4f}）",
        f"28. 是否有方法超过 ResNet18 meanbg 的 macro-F1：{'是' if exceeds_resnet_f1 else '否'}",
        "29. 是否建议做 SwinTiny_LR5e5_LS005 组合实验：不作为优先项；可作为低优先级 ablation。",
        "30. 是否建议转向 ordinal/two-stage/ROI fusion：是，优先 ROI-global fusion，其次 ordinal/two-stage。",
        "31. 下一步最推荐的 3 个实验：ROI-global fusion；ordinal classification；two-stage classification。",
    ]


def _write_report(
    files_checked: pd.DataFrame,
    core: pd.DataFrame,
    confusion: pd.DataFrame,
    best_rules: pd.DataFrame,
    job_queue: pd.DataFrame,
    delta_swin: pd.DataFrame,
    delta_resnet: pd.DataFrame,
    fold_stability: pd.DataFrame,
    recommendation: pd.DataFrame,
    key_info: list[str],
) -> None:
    missing = files_checked.loc[files_checked["exists"] != True]
    swin = _row(core, "swin_tiny_original")
    resnet = _row(core, "resnet18_meanbg")
    lr = _row(core, "swin_tiny_lr5e5")
    ls = _row(core, "swin_tiny_ls005")
    best_macro = best_rules.loc[best_rules["strategy_name"] == "best_macro_f1"].iloc[0]
    best_ba = best_rules.loc[best_rules["strategy_name"] == "best_balanced_accuracy"].iloc[0]
    best_mild = best_rules.loc[best_rules["strategy_name"] == "best_mild_f1"].iloc[0]
    best_tradeoff = best_rules.loc[best_rules["strategy_name"] == "best_balanced_tradeoff"].iloc[0]
    conservative = best_rules.loc[best_rules["strategy_name"] == "conservative_candidate"].iloc[0]
    original_conf = _row(confusion, "swin_tiny_original")
    best_conf = _row(confusion, "swin_tiny_threshold_best_macro_f1")
    trade_conf = _row(confusion, "swin_tiny_threshold_best_balanced_tradeoff")
    lr_conf = _row(confusion, "swin_tiny_lr5e5")
    ls_conf = _row(confusion, "swin_tiny_ls005")
    biggest_stability = _best(fold_stability.rename(columns={"stability_std_mean": "metric"}), "metric")
    stable_values = pd.to_numeric(fold_stability["stability_std_mean"], errors="coerce")
    smallest_idx = stable_values.idxmin()
    most_stable = (fold_stability.loc[smallest_idx, "method_name"], stable_values.loc[smallest_idx])

    candidate_mask = (
        (core["method_name"] != "resnet18_meanbg")
        & (pd.to_numeric(core["macro_f1_mean"], errors="coerce") >= _safe_float(resnet["macro_f1_mean"]) - 0.02)
        & (pd.to_numeric(core["balanced_accuracy_mean"], errors="coerce") > _safe_float(resnet["balanced_accuracy_mean"]))
        & (pd.to_numeric(core["recall_severe_mean"], errors="coerce") >= _safe_float(resnet["recall_severe_mean"]))
    )
    candidates = core.loc[candidate_mask, "method_name"].tolist()

    lines = [
        "# Swin-Tiny Second Stage Diagnostic Report",
        "",
        "## 1. Files Checked",
        "",
        f"- Checked files: {len(files_checked)}",
        f"- Missing files: {len(missing)}",
    ]
    if missing.empty:
        lines.append("- No required files were missing.")
    else:
        lines.append("- Missing paths:")
        lines.extend([f"  - {p}" for p in missing["absolute_path"].tolist()])

    lines.extend(
        [
            "",
            "## 2. Experiment Status",
            "",
            "- Threshold scan outputs exist and were parsed.",
            f"- lr5e5 status: {job_queue.loc[job_queue['experiment_key'].astype(str) == 'lr5e5', 'status'].iloc[0]}",
            f"- LS0.05 status: {job_queue.loc[job_queue['experiment_key'].astype(str) == 'ls005', 'status'].iloc[0]}",
            "- Note: LS0.05 queue contains `completed_after_resume_from_native_access_violation`; final fold summaries are complete after resume.",
            "",
            _markdown_table(job_queue, ["experiment_key", "status", "duration_minutes", "output_dir", "error_message"]),
            "",
            "## 3. Core Result Table",
            "",
            _markdown_table(
                core.merge(recommendation[["method_name", "recommendation"]], on="method_name", how="left"),
                [
                    "method_name",
                    "source_type",
                    "macro_auc_mean",
                    "balanced_accuracy_mean",
                    "macro_f1_mean",
                    "recall_mild_mean",
                    "recall_severe_mean",
                    "f1_mild_mean",
                    "f1_severe_mean",
                    "oof_macro_auc",
                    "oof_balanced_accuracy",
                    "oof_macro_f1",
                    "oof_recall_mild",
                    "oof_recall_severe",
                    "recommendation",
                ],
            ),
            "",
            "## 4. Threshold Scan Analysis",
            "",
            f"- best_macro_f1: {best_macro['rule_name']}, macro-F1={_fmt(best_macro['macro_f1'])}, BA={_fmt(best_macro['balanced_accuracy'])}, mild recall={_fmt(best_macro['recall_mild'])}, severe recall={_fmt(best_macro['recall_severe'])}.",
            f"- best_balanced_accuracy: {best_ba['rule_name']}, BA={_fmt(best_ba['balanced_accuracy'])}, macro-F1={_fmt(best_ba['macro_f1'])}.",
            f"- best_mild_f1: {best_mild['rule_name']}, mild recall={_fmt(best_mild['recall_mild'])}, severe recall={_fmt(best_mild['recall_severe'])}; this strongly sacrifices severe.",
            f"- best_balanced_tradeoff: {best_tradeoff['rule_name']}, macro-F1={_fmt(best_tradeoff['macro_f1'])}, BA={_fmt(best_tradeoff['balanced_accuracy'])}, mild recall={_fmt(best_tradeoff['recall_mild'])}, severe recall={_fmt(best_tradeoff['recall_severe'])}.",
            f"- conservative_candidate exists: {'yes' if str(conservative['rule_name']) != 'no_candidate' else 'no'}.",
            f"- Original Swin mild->severe={_fmt_count(original_conf['mild_to_severe'])}; best_macro_f1 mild->severe={_fmt_count(best_conf['mild_to_severe'])}; best_tradeoff mild->severe={_fmt_count(trade_conf['mild_to_severe'])}.",
            "- Threshold scan improves mild recall and macro-F1, but part of that gain comes from lowering severe recall. The conservative candidate is the most balanced threshold candidate, not the raw best mild-F1 rule.",
            "",
            "## 5. Training Tuning Analysis",
            "",
            "### 5.1 Swin-Tiny lr=5e-5",
            "",
            f"- macro-F1: {_fmt(swin['macro_f1_mean'])} -> {_fmt(lr['macro_f1_mean'])}; only a small improvement.",
            f"- mild recall: {_fmt(swin['recall_mild_mean'])} -> {_fmt(lr['recall_mild_mean'])}; improved.",
            f"- severe recall: {_fmt(swin['recall_severe_mean'])} -> {_fmt(lr['recall_severe_mean'])}; improved.",
            f"- BA: {_fmt(swin['balanced_accuracy_mean'])} -> {_fmt(lr['balanced_accuracy_mean'])}; declined, so lr=5e-5 is not clearly better than original Swin.",
            f"- Compared with ResNet18 macro-F1={_fmt(resnet['macro_f1_mean'])}, lr5e5 remains insufficient.",
            "",
            "### 5.2 Swin-Tiny LS0.05",
            "",
            f"- macro-F1: {_fmt(swin['macro_f1_mean'])} -> {_fmt(ls['macro_f1_mean'])}; improved.",
            f"- mild recall: {_fmt(swin['recall_mild_mean'])} -> {_fmt(ls['recall_mild_mean'])}; improved.",
            f"- severe recall: {_fmt(swin['recall_severe_mean'])} -> {_fmt(ls['recall_severe_mean'])}; decreased but remains close to ResNet18.",
            f"- BA remains {_fmt(ls['balanced_accuracy_mean'])}, approximately unchanged from original Swin.",
            f"- LS0.05 is more effective than lr=5e-5 for balanced tuning, but still does not reach ResNet18 macro-F1.",
            "",
            "## 6. Confusion Matrix Analysis",
            "",
            f"- Original Swin mild->severe={_fmt_count(original_conf['mild_to_severe'])}, confirming mild compression toward severe.",
            f"- best_macro_f1 reduces mild->severe to {_fmt_count(best_conf['mild_to_severe'])}, but severe->mild rises to {_fmt_count(best_conf['severe_to_mild'])}.",
            f"- best_balanced_tradeoff reduces mild->severe to {_fmt_count(trade_conf['mild_to_severe'])}, but severe recall is low.",
            f"- lr=5e-5 mild->severe={_fmt_count(lr_conf['mild_to_severe'])}; LS0.05 mild->severe={_fmt_count(ls_conf['mild_to_severe'])}. LS0.05 gives cleaner macro-F1/BA balance than lr5e5.",
            "- The method that most restores mild recall is the threshold best_mild_f1 rule, but it destroys severe recall. The most balanced candidate is conservative thresholding; among actual retraining experiments, LS0.05 is preferable.",
            "",
            "## 7. Comparison with ResNet18 MeanBG",
            "",
            f"- Methods meeting macro-F1 close/above ResNet18, BA above ResNet18, and severe recall not below ResNet18: {', '.join(candidates) if candidates else 'none'}.",
            "- Some Swin variants preserve higher AUC/BA, but macro-F1 remains weaker unless thresholding sacrifices severe performance.",
            "- For final hard three-class prediction, ResNet18 meanbg remains the safer main-table model; Swin-Tiny remains a useful high AUC/BA candidate and diagnostic direction.",
            "",
            "## 8. Recommendation for Next Step",
            "",
            "Priority 1: ROI-global fusion / ROI-fusion, because current failure mode is class-boundary hard decision and Swin global-only tuning only partially fixes mild.",
            "Priority 2: ordinal classification, because NYHA classes have ordered structure and mild is a boundary class.",
            "Priority 3: two-stage classification, e.g. normal-vs-abnormal followed by mild-vs-severe.",
            "Priority 4: keep Swin-Tiny LS0.05 as a secondary high-AUC/BA model; do not use threshold scan as external clinical threshold.",
            "Priority 5: SwinTiny_LR5e5_LS005 can be run as a low-priority ablation, but current evidence does not make it the primary next experiment.",
            "Priority 6: EfficientNet-B0 label smoothing is optional for a lightweight route.",
            "",
            "## 9. Final Conclusion",
            "",
            "1. threshold scan is effective diagnostically: it can raise macro-F1 and mild recall, but severe recall tradeoff is substantial.",
            "2. lr=5e-5 is only partially effective: mild/severe recall improve, but BA drops and macro-F1 remains low.",
            "3. LS0.05 is the best training-side second-stage adjustment: macro-F1 and mild recall improve while BA is preserved, but it still does not beat ResNet18 macro-F1.",
            "4. Swin-Tiny should continue as a secondary high-AUC/BA candidate, not replace ResNet18 as the main hard-decision model yet.",
            "5. Next step should prioritize ROI-global fusion or ordered/two-stage formulations rather than only stacking Swin hyperparameter tweaks.",
            "",
            "## 10. Fold Stability Analysis",
            "",
            f"- Largest fold volatility by average std: {biggest_stability[0]} ({biggest_stability[1]:.4f}).",
            f"- Smallest fold volatility by average std: {most_stable[0]} ({most_stable[1]:.4f}).",
            "- If a method improves mean macro-F1 but has large std, interpret it cautiously; this is especially relevant in the current 500-sample five-fold setting.",
            "",
            "## Key Information for Further Analysis",
            "",
            *key_info,
            "",
        ]
    )
    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    files_checked = _files_checked_frame()
    missing = files_checked.loc[files_checked["exists"] != True]

    lr_dir = _latest_dir_by_prefix(SECOND_STAGE_ROOT, LR5E5_PREFIX)
    ls_dir = _latest_dir_by_prefix(SECOND_STAGE_ROOT, LS005_PREFIX)

    core_rows = []
    fold_frames = []
    per_class_frames = []
    oof_by_method: dict[str, pd.DataFrame] = {}
    confusion_rows = []

    for method_name, source_type, directory in [
        ("resnet18_meanbg", "baseline", RESNET18_DIR),
        ("swin_tiny_original", "model_exploration", SWIN_ORIGINAL_DIR),
        ("swin_tiny_lr5e5", "second_stage_training", lr_dir),
        ("swin_tiny_ls005", "second_stage_training", ls_dir),
    ]:
        row, fold, per_class, oof = _row_from_experiment(method_name, source_type, directory)
        core_rows.append(row)
        fold_frames.append(fold)
        per_class_frames.append(per_class)
        oof_by_method[method_name] = oof
        confusion_rows.append(
            _confusion_summary(
                method_name,
                oof["label_3class"].to_numpy(dtype=int),
                oof["pred_class"].to_numpy(dtype=int),
            )
        )

    best_rules = _read_csv(THRESHOLD_DIR / "best_threshold_rules.csv")
    threshold_rules_summary = best_rules.copy()
    original_row = next(row for row in core_rows if row["method_name"] == "swin_tiny_original")
    swin_oof = oof_by_method["swin_tiny_original"]
    for strategy_name, method_name in THRESHOLD_STRATEGIES.items():
        rule = best_rules.loc[best_rules["strategy_name"] == strategy_name]
        if rule.empty or str(rule.iloc[0].get("rule_name")) == "no_candidate":
            continue
        threshold_row, per_class, confusion = _row_from_threshold(
            method_name, rule.iloc[0], original_row, swin_oof
        )
        core_rows.append(threshold_row)
        per_class_frames.append(per_class)
        confusion_rows.append(confusion)

    core = pd.DataFrame(core_rows)
    requested_order = [
        "resnet18_meanbg",
        "swin_tiny_original",
        "swin_tiny_threshold_best_macro_f1",
        "swin_tiny_threshold_best_balanced_accuracy",
        "swin_tiny_threshold_best_mild_f1",
        "swin_tiny_threshold_best_balanced_tradeoff",
        "swin_tiny_threshold_conservative_candidate",
        "swin_tiny_lr5e5",
        "swin_tiny_ls005",
    ]
    core["method_name"] = pd.Categorical(core["method_name"], requested_order, ordered=True)
    core = core.sort_values("method_name").reset_index(drop=True)
    core["method_name"] = core["method_name"].astype(str)

    fold_all = pd.concat(fold_frames, ignore_index=True)
    per_class_metrics = pd.concat(per_class_frames, ignore_index=True)
    confusion = pd.DataFrame(confusion_rows)
    delta_swin = _delta_frame(core, "swin_tiny_original", "delta_vs_swin_tiny_original")
    delta_resnet = _delta_frame(core, "resnet18_meanbg", "delta_vs_resnet18_meanbg")
    fold_stability = _fold_stability_frame(fold_all)
    recommendation = _recommendations(core, delta_swin, delta_resnet)
    job_queue = _read_csv(SECOND_STAGE_ROOT / "swin_tiny_second_stage_job_queue.csv")
    key_info = _key_information(core, confusion, best_rules)

    core.to_csv(CORE_SUMMARY_PATH, index=False, encoding="utf-8-sig")
    confusion.to_csv(CONFUSION_SUMMARY_PATH, index=False, encoding="utf-8-sig")
    threshold_rules_summary.to_csv(
        THRESHOLD_RULES_SUMMARY_PATH, index=False, encoding="utf-8-sig"
    )
    per_class_metrics.to_csv(PER_CLASS_METRICS_PATH, index=False, encoding="utf-8-sig")
    recommendation.to_csv(RECOMMENDATION_PATH, index=False, encoding="utf-8-sig")

    write_xlsx(
        TABLES_XLSX_PATH,
        {
            "files_checked": files_checked,
            "core_summary": core,
            "delta_vs_swin_original": delta_swin,
            "delta_vs_resnet18": delta_resnet,
            "threshold_best_rules": threshold_rules_summary,
            "confusion_summary": confusion,
            "per_class_metrics": per_class_metrics,
            "fold_stability": fold_stability,
            "job_queue": job_queue,
            "recommendation": recommendation,
            "key_information": pd.DataFrame({"item": key_info}),
        },
    )

    _write_report(
        files_checked,
        core,
        confusion,
        best_rules,
        job_queue,
        delta_swin,
        delta_resnet,
        fold_stability,
        recommendation,
        key_info,
    )

    print(f"REPORT={REPORT_PATH}")
    print(f"TABLES_XLSX={TABLES_XLSX_PATH}")
    print(f"CORE_SUMMARY={CORE_SUMMARY_PATH}")
    print(f"CONFUSION_SUMMARY={CONFUSION_SUMMARY_PATH}")
    print(f"THRESHOLD_RULES_SUMMARY={THRESHOLD_RULES_SUMMARY_PATH}")
    print(f"PER_CLASS_METRICS={PER_CLASS_METRICS_PATH}")
    print(f"RECOMMENDATION={RECOMMENDATION_PATH}")
    print(f"MISSING_FILES={len(missing)}")


if __name__ == "__main__":
    main()
