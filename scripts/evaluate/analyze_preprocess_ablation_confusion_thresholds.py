"""Analyze ResNet18 preprocessing ablation confusion matrices and thresholds."""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path
from typing import Any, Sequence

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


DEFAULT_EXPERIMENTS_ROOT = PROJECT_ROOT / "experiments" / "preprocess_ablation_500Data"
DEFAULT_OUTPUT_DIR = DEFAULT_EXPERIMENTS_ROOT / "decision_analysis"
DEFAULT_VARIANTS = (
    "hybrid_black_baseline",
    "hybrid_imagenet_meanbg",
    "hybrid_black_clahe_l",
)
REQUIRED_COLUMNS = (
    "label_3class",
    "pred_class",
    "prob_normal",
    "prob_mild",
    "prob_severe",
)
CLASS_LABELS = (0, 1, 2)
CLASS_NAMES = {0: "normal", 1: "mild", 2: "severe"}


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiments-root", type=Path, default=DEFAULT_EXPERIMENTS_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--variants", type=str, default=",".join(DEFAULT_VARIANTS))
    parser.add_argument("--threshold-min", type=float, default=0.05)
    parser.add_argument("--threshold-max", type=float, default=0.95)
    parser.add_argument("--threshold-step", type=float, default=0.01)
    parser.add_argument("--make-figures", action="store_true")
    return parser.parse_args(argv)


def _resolve(path: str | Path) -> Path:
    value = Path(path).expanduser()
    return value.resolve() if value.is_absolute() else (PROJECT_ROOT / value).resolve()


def _thresholds(start: float, stop: float, step: float) -> list[float]:
    if step <= 0:
        raise ValueError("--threshold-step must be positive")
    if stop < start:
        raise ValueError("--threshold-max must be >= --threshold-min")
    count = int(round((stop - start) / step)) + 1
    return [round(start + index * step, 10) for index in range(count)]


def _variant_oof_path(experiments_root: Path, variant_name: str) -> Path:
    return (
        experiments_root
        / f"PreprocAblation_ResNet18_NYHA3Class_{variant_name}"
        / "summary"
        / "oof_predictions.csv"
    )


def _load_predictions(experiments_root: Path, variant_name: str) -> pd.DataFrame:
    path = _variant_oof_path(experiments_root, variant_name)
    if not path.is_file():
        raise FileNotFoundError(f"OOF predictions not found for {variant_name}: {path}")
    frame = pd.read_csv(path, dtype=str, encoding="utf-8-sig").fillna("")
    missing = [column for column in REQUIRED_COLUMNS if column not in frame.columns]
    if missing:
        raise ValueError(f"{path} missing required columns: {missing}")
    for column in REQUIRED_COLUMNS:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    if frame[list(REQUIRED_COLUMNS)].isna().any().any():
        bad = frame[list(REQUIRED_COLUMNS)].isna().sum()
        raise ValueError(f"{path} contains non-numeric required values: {bad.to_dict()}")
    frame["label_3class"] = frame["label_3class"].astype(int)
    frame["pred_class"] = frame["pred_class"].astype(int)
    frame["variant_name"] = variant_name
    return frame


def _safe_div(numerator: float, denominator: float) -> float:
    return float(numerator / denominator) if denominator else 0.0


def _classification_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=list(CLASS_LABELS),
        zero_division=0,
    )
    macro_precision, macro_recall, macro_f1, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=list(CLASS_LABELS),
        average="macro",
        zero_division=0,
    )
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_precision": float(macro_precision),
        "macro_recall": float(macro_recall),
        "macro_f1": float(macro_f1),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "recall_normal": float(recall[0]),
        "recall_mild": float(recall[1]),
        "recall_severe": float(recall[2]),
        "f1_normal": float(f1[0]),
        "f1_mild": float(f1[1]),
        "f1_severe": float(f1[2]),
        "precision_normal": float(precision[0]),
        "precision_mild": float(precision[1]),
        "precision_severe": float(precision[2]),
    }


def _confusion_error_row(variant_name: str, cm: np.ndarray) -> dict[str, Any]:
    true_normal_total = int(cm[0].sum())
    true_mild_total = int(cm[1].sum())
    true_severe_total = int(cm[2].sum())
    severe_to_normal = int(cm[2, 0])
    severe_to_mild = int(cm[2, 1])
    severe_to_severe = int(cm[2, 2])
    normal_to_severe = int(cm[0, 2])
    mild_to_severe = int(cm[1, 2])
    return {
        "variant_name": variant_name,
        "true_severe_total": true_severe_total,
        "severe_to_normal": severe_to_normal,
        "severe_to_mild": severe_to_mild,
        "severe_to_severe": severe_to_severe,
        "severe_recall": _safe_div(severe_to_severe, true_severe_total),
        "normal_to_severe": normal_to_severe,
        "mild_to_severe": mild_to_severe,
        "severe_false_positive_total": normal_to_severe + mild_to_severe,
        "normal_to_severe_rate": _safe_div(normal_to_severe, true_normal_total),
        "mild_to_severe_rate": _safe_div(mild_to_severe, true_mild_total),
    }


def _confusion_long(variant_name: str, cm: np.ndarray, matrix_type: str) -> pd.DataFrame:
    rows = []
    for true_index, true_class in enumerate(CLASS_LABELS):
        for pred_index, pred_class in enumerate(CLASS_LABELS):
            rows.append(
                {
                    "variant_name": variant_name,
                    "matrix_type": matrix_type,
                    "true_class": true_class,
                    "true_class_name": CLASS_NAMES[true_class],
                    "pred_class": pred_class,
                    "pred_class_name": CLASS_NAMES[pred_class],
                    "value": float(cm[true_index, pred_index]),
                }
            )
    return pd.DataFrame(rows)


def _argmax_analysis(predictions: dict[str, pd.DataFrame]) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, np.ndarray], dict[str, np.ndarray], pd.DataFrame]:
    metric_rows = []
    error_rows = []
    count_matrices: dict[str, np.ndarray] = {}
    normalized_matrices: dict[str, np.ndarray] = {}
    long_frames = []
    for variant_name, frame in predictions.items():
        y_true = frame["label_3class"].to_numpy(dtype=int)
        y_pred = frame["pred_class"].to_numpy(dtype=int)
        metrics = _classification_metrics(y_true, y_pred)
        cm = confusion_matrix(y_true, y_pred, labels=list(CLASS_LABELS))
        row_sum = cm.sum(axis=1, keepdims=True)
        row_norm = np.divide(cm, row_sum, out=np.zeros_like(cm, dtype=float), where=row_sum != 0)
        count_matrices[variant_name] = cm
        normalized_matrices[variant_name] = row_norm
        error_rows.append(_confusion_error_row(variant_name, cm))
        metric_rows.append({"variant_name": variant_name, **metrics})
        long_frames.append(_confusion_long(variant_name, cm, "count"))
        long_frames.append(_confusion_long(variant_name, row_norm, "row_normalized"))
    return (
        pd.DataFrame(metric_rows),
        pd.DataFrame(error_rows),
        count_matrices,
        normalized_matrices,
        pd.concat(long_frames, ignore_index=True),
    )


def _binary_threshold_row(variant_name: str, y_true_binary: np.ndarray, score: np.ndarray, threshold: float) -> dict[str, Any]:
    y_pred = (score >= threshold).astype(int)
    tp = int(((y_true_binary == 1) & (y_pred == 1)).sum())
    fp = int(((y_true_binary == 0) & (y_pred == 1)).sum())
    tn = int(((y_true_binary == 0) & (y_pred == 0)).sum())
    fn = int(((y_true_binary == 1) & (y_pred == 0)).sum())
    sensitivity = _safe_div(tp, tp + fn)
    specificity = _safe_div(tn, tn + fp)
    precision = _safe_div(tp, tp + fp)
    npv = _safe_div(tn, tn + fn)
    f1 = _safe_div(2 * precision * sensitivity, precision + sensitivity)
    f2 = _safe_div(5 * precision * sensitivity, 4 * precision + sensitivity)
    accuracy = _safe_div(tp + tn, tp + fp + tn + fn)
    balanced_accuracy = (sensitivity + specificity) / 2.0
    return {
        "variant_name": variant_name,
        "threshold": threshold,
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "sensitivity": sensitivity,
        "recall_severe": sensitivity,
        "specificity": specificity,
        "precision": precision,
        "ppv": precision,
        "npv": npv,
        "f1": f1,
        "f2": f2,
        "accuracy": accuracy,
        "balanced_accuracy": balanced_accuracy,
        "youden_index": sensitivity + specificity - 1.0,
    }


def _binary_scan(predictions: dict[str, pd.DataFrame], thresholds: list[float]) -> pd.DataFrame:
    rows = []
    for variant_name, frame in predictions.items():
        y_true_binary = (frame["label_3class"].to_numpy(dtype=int) == 2).astype(int)
        score = frame["prob_severe"].to_numpy(dtype=float)
        for threshold in thresholds:
            rows.append(_binary_threshold_row(variant_name, y_true_binary, score, threshold))
    return pd.DataFrame(rows)


def _pick_best(frame: pd.DataFrame, strategy: str, sort_columns: list[tuple[str, bool]]) -> dict[str, Any]:
    if frame.empty:
        return {"strategy": strategy, "selection_status": "no_candidate"}
    ascending = [not descending for _, descending in sort_columns]
    sorted_frame = frame.sort_values([column for column, _ in sort_columns], ascending=ascending)
    row = sorted_frame.iloc[0].to_dict()
    row["strategy"] = strategy
    row["selection_status"] = "selected"
    return row


def _best_binary_thresholds(scan: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for variant_name, group in scan.groupby("variant_name", sort=False):
        rows.append(_pick_best(group, "best_youden", [("youden_index", True), ("balanced_accuracy", True), ("f1", True)]))
        rows.append(_pick_best(group, "best_binary_f1", [("f1", True), ("balanced_accuracy", True), ("specificity", True)]))
        rows.append(_pick_best(group, "best_binary_balanced_accuracy", [("balanced_accuracy", True), ("f1", True), ("specificity", True)]))
        for target in (0.50, 0.60, 0.70):
            eligible = group[group["recall_severe"] >= target]
            rows.append(
                _pick_best(
                    eligible,
                    f"recall_ge_{target:.2f}_best_specificity",
                    [("specificity", True), ("balanced_accuracy", True), ("f1", True)],
                )
            )
    return pd.DataFrame(rows)


def _severe_priority_predictions(frame: pd.DataFrame, threshold: float) -> np.ndarray:
    prob_severe = frame["prob_severe"].to_numpy(dtype=float)
    prob_normal = frame["prob_normal"].to_numpy(dtype=float)
    prob_mild = frame["prob_mild"].to_numpy(dtype=float)
    normal_or_mild = np.where(prob_normal >= prob_mild, 0, 1)
    return np.where(prob_severe >= threshold, 2, normal_or_mild).astype(int)


def _multiclass_scan(predictions: dict[str, pd.DataFrame], thresholds: list[float]) -> pd.DataFrame:
    rows = []
    for variant_name, frame in predictions.items():
        y_true = frame["label_3class"].to_numpy(dtype=int)
        for threshold in thresholds:
            y_pred = _severe_priority_predictions(frame, threshold)
            metrics = _classification_metrics(y_true, y_pred)
            cm = confusion_matrix(y_true, y_pred, labels=list(CLASS_LABELS))
            rows.append(
                {
                    "variant_name": variant_name,
                    "threshold": threshold,
                    **metrics,
                    "normal_to_severe": int(cm[0, 2]),
                    "mild_to_severe": int(cm[1, 2]),
                    "severe_to_normal": int(cm[2, 0]),
                    "severe_to_mild": int(cm[2, 1]),
                }
            )
    return pd.DataFrame(rows)


def _best_multiclass_thresholds(scan: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for variant_name, group in scan.groupby("variant_name", sort=False):
        rows.append(_pick_best(group, "best_multiclass_macro_f1", [("macro_f1", True), ("balanced_accuracy", True), ("recall_severe", True)]))
        rows.append(_pick_best(group, "best_multiclass_balanced_accuracy", [("balanced_accuracy", True), ("macro_f1", True), ("recall_severe", True)]))
        rows.append(_pick_best(group[group["recall_severe"] >= 0.50], "severe_recall_ge_0.50_best_macro_f1", [("macro_f1", True), ("balanced_accuracy", True)]))
        rows.append(_pick_best(group[group["recall_severe"] >= 0.60], "severe_recall_ge_0.60_best_macro_f1", [("macro_f1", True), ("balanced_accuracy", True)]))
        rows.append(_pick_best(group[group["recall_severe"] >= 0.60], "severe_recall_ge_0.60_best_balanced_accuracy", [("balanced_accuracy", True), ("macro_f1", True)]))
        rows.append(_pick_best(group[group["recall_severe"] >= 0.70], "severe_recall_ge_0.70_best_macro_f1", [("macro_f1", True), ("balanced_accuracy", True)]))
        rows.append(_pick_best(group[group["recall_severe"] >= 0.70], "severe_recall_ge_0.70_best_balanced_accuracy", [("balanced_accuracy", True), ("macro_f1", True)]))
    return pd.DataFrame(rows)


def _write_adjusted_predictions(
    predictions: dict[str, pd.DataFrame],
    best_multiclass: pd.DataFrame,
    output_dir: Path,
) -> None:
    adjusted_dir = output_dir / "adjusted_predictions"
    adjusted_dir.mkdir(parents=True, exist_ok=True)
    file_map = {
        "best_multiclass_macro_f1": "best_macro_f1",
        "best_multiclass_balanced_accuracy": "best_balanced_accuracy",
    }
    selected = best_multiclass[best_multiclass.get("strategy", "").isin(file_map)]
    for _, row in selected.iterrows():
        if row.get("selection_status") != "selected":
            continue
        variant_name = str(row["variant_name"])
        threshold = float(row["threshold"])
        frame = predictions[variant_name].copy()
        frame["threshold_strategy"] = str(row["strategy"])
        frame["threshold"] = threshold
        frame["pred_adjusted"] = _severe_priority_predictions(frame, threshold)
        frame["pred_adjusted_name"] = frame["pred_adjusted"].map(CLASS_NAMES)
        frame["adjusted_correct"] = (frame["pred_adjusted"].astype(int) == frame["label_3class"].astype(int)).astype(int)
        output_name = f"{variant_name}_{file_map[str(row['strategy'])]}_predictions.csv"
        frame.to_csv(adjusted_dir / output_name, index=False, encoding="utf-8-sig")


def _recommendations(argmax_metrics: pd.DataFrame, argmax_errors: pd.DataFrame, best_multiclass: pd.DataFrame) -> pd.DataFrame:
    rows = []
    argmax_lookup = argmax_metrics.set_index("variant_name")
    error_lookup = argmax_errors.set_index("variant_name")
    selected = best_multiclass[best_multiclass.get("selection_status", "") == "selected"].copy()
    for _, row in selected.iterrows():
        variant = str(row["variant_name"])
        arg = argmax_lookup.loc[variant]
        err = error_lookup.loc[variant]
        delta_macro_f1 = float(row["macro_f1"]) - float(arg["macro_f1"])
        delta_ba = float(row["balanced_accuracy"]) - float(arg["balanced_accuracy"])
        delta_recall_severe = float(row["recall_severe"]) - float(arg["recall_severe"])
        delta_severe_fp = int(row["normal_to_severe"]) + int(row["mild_to_severe"]) - int(err["severe_false_positive_total"])
        if (delta_macro_f1 > 0 or delta_ba > 0) and delta_recall_severe >= -1e-12:
            label = "candidate decision rule"
        elif delta_recall_severe >= 0.10 and delta_macro_f1 < -0.02 and delta_ba < -0.02:
            label = "high-sensitivity exploratory threshold"
        else:
            label = "not recommended as primary threshold"
        rows.append(
            {
                "variant_name": variant,
                "strategy": row["strategy"],
                "threshold": row["threshold"],
                "macro_f1": row["macro_f1"],
                "balanced_accuracy": row["balanced_accuracy"],
                "recall_severe": row["recall_severe"],
                "delta_macro_f1_vs_argmax": delta_macro_f1,
                "delta_balanced_accuracy_vs_argmax": delta_ba,
                "delta_recall_severe_vs_argmax": delta_recall_severe,
                "delta_severe_false_positive_vs_argmax": delta_severe_fp,
                "recommendation": label,
            }
        )
    return pd.DataFrame(rows)


def _plot_confusion(cm: np.ndarray, title: str, path: Path, normalized: bool) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(5.2, 4.6), dpi=160)
    image = ax.imshow(cm, cmap="Blues")
    ax.set_title(title)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_xticks(range(3), [CLASS_NAMES[index] for index in CLASS_LABELS])
    ax.set_yticks(range(3), [CLASS_NAMES[index] for index in CLASS_LABELS])
    for i in range(3):
        for j in range(3):
            value = f"{cm[i, j]:.2f}" if normalized else f"{int(cm[i, j])}"
            ax.text(j, i, value, ha="center", va="center", color="black", fontsize=9)
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def _plot_curves(frame: pd.DataFrame, variant_name: str, columns: list[str], title: str, path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7.2, 4.6), dpi=160)
    for column in columns:
        ax.plot(frame["threshold"], frame[column], label=column)
    ax.set_title(title)
    ax.set_xlabel("threshold")
    ax.set_ylabel("metric")
    ax.set_ylim(0, 1.02)
    ax.grid(alpha=0.25)
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def _make_figures(
    output_dir: Path,
    count_matrices: dict[str, np.ndarray],
    normalized_matrices: dict[str, np.ndarray],
    binary_scan: pd.DataFrame,
    multiclass_scan: pd.DataFrame,
) -> None:
    figure_dir = output_dir / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    for variant_name, cm in count_matrices.items():
        _plot_confusion(cm, f"{variant_name} confusion count", figure_dir / f"{variant_name}_confusion_count.png", normalized=False)
        _plot_confusion(
            normalized_matrices[variant_name],
            f"{variant_name} row-normalized confusion",
            figure_dir / f"{variant_name}_confusion_row_normalized.png",
            normalized=True,
        )
    for variant_name, group in binary_scan.groupby("variant_name", sort=False):
        _plot_curves(
            group,
            variant_name,
            ["sensitivity", "specificity", "precision", "f1", "balanced_accuracy"],
            f"{variant_name} binary severe-vs-rest threshold curves",
            figure_dir / f"{variant_name}_binary_threshold_curves.png",
        )
    for variant_name, group in multiclass_scan.groupby("variant_name", sort=False):
        _plot_curves(
            group,
            variant_name,
            ["macro_f1", "balanced_accuracy", "recall_severe", "recall_mild", "recall_normal"],
            f"{variant_name} severe-priority multiclass threshold curves",
            figure_dir / f"{variant_name}_multiclass_threshold_curves.png",
        )


def _fmt(value: Any) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "NA"
    return "NA" if not math.isfinite(number) else f"{number:.4f}"


def _best_strategy_text(recommendations: pd.DataFrame, variant_name: str) -> str:
    subset = recommendations[
        (recommendations["variant_name"] == variant_name)
        & (recommendations["recommendation"] == "candidate decision rule")
    ].copy()
    if subset.empty:
        return f"- `{variant_name}`: no primary severe-priority threshold is recommended."
    subset = subset.sort_values(["macro_f1", "balanced_accuracy", "recall_severe"], ascending=False)
    row = subset.iloc[0]
    return (
        f"- `{variant_name}`: `{row['strategy']}` at threshold={_fmt(row['threshold'])} "
        f"(macro-F1={_fmt(row['macro_f1'])}, BA={_fmt(row['balanced_accuracy'])}, "
        f"severe recall={_fmt(row['recall_severe'])})."
    )


def _write_summary_md(
    path: Path,
    argmax_metrics: pd.DataFrame,
    argmax_errors: pd.DataFrame,
    best_binary: pd.DataFrame,
    best_multiclass: pd.DataFrame,
    recommendations: pd.DataFrame,
) -> None:
    err = argmax_errors.set_index("variant_name")
    baseline_missed = int(err.loc["hybrid_black_baseline", "severe_to_normal"] + err.loc["hybrid_black_baseline", "severe_to_mild"]) if "hybrid_black_baseline" in err.index else 0
    lines = [
        "# Decision Analysis Summary",
        "",
        "## Purpose",
        "",
        "This analysis uses existing ResNet18 OOF predictions to inspect confusion patterns and evaluate internal severe-priority threshold rules without retraining.",
        "",
        "## Argmax Confusion Overview",
        "",
        "| variant | severe recall | severe->normal | severe->mild | normal->severe | mild->severe |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in argmax_errors.itertuples(index=False):
        lines.append(
            f"| {row.variant_name} | {_fmt(row.severe_recall)} | {row.severe_to_normal} | "
            f"{row.severe_to_mild} | {row.normal_to_severe} | {row.mild_to_severe} |"
        )
    lines.extend(["", "## Severe Misclassification Direction", ""])
    for row in argmax_errors.itertuples(index=False):
        missed = int(row.severe_to_normal + row.severe_to_mild)
        main_direction = "mild" if row.severe_to_mild >= row.severe_to_normal else "normal"
        lines.append(f"- `{row.variant_name}`: {missed} true severe cases were missed, mainly to `{main_direction}`.")
    lines.extend(["", "## Meanbg vs Baseline", ""])
    if {"hybrid_black_baseline", "hybrid_imagenet_meanbg"}.issubset(err.index):
        meanbg_missed = int(err.loc["hybrid_imagenet_meanbg", "severe_to_normal"] + err.loc["hybrid_imagenet_meanbg", "severe_to_mild"])
        lines.append(
            f"`hybrid_imagenet_meanbg` changes severe misses from {baseline_missed} to {meanbg_missed} "
            f"(reduction={baseline_missed - meanbg_missed})."
        )
    lines.extend(["", "## CLAHE vs Baseline", ""])
    if {"hybrid_black_baseline", "hybrid_black_clahe_l"}.issubset(err.index):
        clahe_missed = int(err.loc["hybrid_black_clahe_l", "severe_to_normal"] + err.loc["hybrid_black_clahe_l", "severe_to_mild"])
        lines.append(
            f"`hybrid_black_clahe_l` changes severe misses from {baseline_missed} to {clahe_missed} "
            f"(reduction={baseline_missed - clahe_missed})."
        )
    lines.extend(["", "## Binary Severe-vs-Rest Threshold Scan", ""])
    binary_focus = best_binary[best_binary.get("strategy", "") == "best_binary_balanced_accuracy"]
    if binary_focus.empty:
        lines.append("No binary threshold candidate was selected.")
    else:
        for row in binary_focus.itertuples(index=False):
            lines.append(
                f"- `{row.variant_name}` best binary BA threshold={_fmt(row.threshold)}, "
                f"sensitivity={_fmt(row.sensitivity)}, specificity={_fmt(row.specificity)}, "
                f"BA={_fmt(row.balanced_accuracy)}."
            )
    lines.extend(["", "## Severe-Priority Multiclass Threshold Scan", ""])
    multiclass_focus = best_multiclass[best_multiclass.get("strategy", "") == "best_multiclass_macro_f1"]
    if multiclass_focus.empty:
        lines.append("No multiclass threshold candidate was selected.")
    else:
        for row in multiclass_focus.itertuples(index=False):
            lines.append(
                f"- `{row.variant_name}` best macro-F1 threshold={_fmt(row.threshold)}, "
                f"macro-F1={_fmt(row.macro_f1)}, BA={_fmt(row.balanced_accuracy)}, "
                f"severe recall={_fmt(row.recall_severe)}."
            )
    lines.extend(["", "## Recommended Threshold Strategy", ""])
    for variant_name in DEFAULT_VARIANTS:
        if variant_name in set(argmax_metrics["variant_name"]):
            lines.append(_best_strategy_text(recommendations, variant_name))
    use_threshold = (recommendations.get("recommendation", pd.Series(dtype=str)) == "candidate decision rule").any()
    lines.extend(
        [
            "",
            "## Severe-Priority Recommendation",
            "",
            (
                "A severe-priority threshold can be considered as a candidate decision rule for variants with improved macro-F1 or balanced accuracy and non-decreased severe recall."
                if use_threshold
                else "No severe-priority threshold is recommended as a primary decision rule from this OOF scan."
            ),
            "",
            "## Caution",
            "",
            "OOF threshold scanning is an internal cross-validation analysis. Any selected threshold should be validated on an independent test set or external validation cohort before clinical interpretation.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    experiments_root = _resolve(args.experiments_root)
    output_dir = _resolve(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    variants = tuple(item.strip() for item in args.variants.split(",") if item.strip())
    thresholds = _thresholds(args.threshold_min, args.threshold_max, args.threshold_step)

    predictions = {variant: _load_predictions(experiments_root, variant) for variant in variants}
    argmax_metrics, argmax_errors, count_matrices, normalized_matrices, confusion_long = _argmax_analysis(predictions)
    binary_scan = _binary_scan(predictions, thresholds)
    best_binary = _best_binary_thresholds(binary_scan)
    multiclass_scan = _multiclass_scan(predictions, thresholds)
    best_multiclass = _best_multiclass_thresholds(multiclass_scan)
    recommendations = _recommendations(argmax_metrics, argmax_errors, best_multiclass)

    argmax_metrics.to_csv(output_dir / "argmax_metrics.csv", index=False, encoding="utf-8-sig")
    argmax_errors.to_csv(output_dir / "confusion_error_summary.csv", index=False, encoding="utf-8-sig")
    confusion_long.to_csv(output_dir / "confusion_matrices_long.csv", index=False, encoding="utf-8-sig")
    binary_scan.to_csv(output_dir / "threshold_scan_binary_severe_vs_rest.csv", index=False, encoding="utf-8-sig")
    best_binary.to_csv(output_dir / "best_binary_thresholds.csv", index=False, encoding="utf-8-sig")
    multiclass_scan.to_csv(output_dir / "threshold_scan_multiclass_severe_priority.csv", index=False, encoding="utf-8-sig")
    best_multiclass.to_csv(output_dir / "best_multiclass_thresholds.csv", index=False, encoding="utf-8-sig")
    recommendations.to_csv(output_dir / "recommendation.csv", index=False, encoding="utf-8-sig")
    recommendations.to_csv(output_dir / "threshold_recommendations.csv", index=False, encoding="utf-8-sig")

    confusion_sheets = {"confusion_long": confusion_long}
    for variant_name, cm in count_matrices.items():
        confusion_sheets[f"{variant_name[:20]}_count"] = pd.DataFrame(
            cm,
            index=[CLASS_NAMES[index] for index in CLASS_LABELS],
            columns=[CLASS_NAMES[index] for index in CLASS_LABELS],
        ).reset_index(names="true_class")
        confusion_sheets[f"{variant_name[:18]}_rownorm"] = pd.DataFrame(
            normalized_matrices[variant_name],
            index=[CLASS_NAMES[index] for index in CLASS_LABELS],
            columns=[CLASS_NAMES[index] for index in CLASS_LABELS],
        ).reset_index(names="true_class")
    write_xlsx(output_dir / "confusion_matrices.xlsx", confusion_sheets)
    write_xlsx(
        output_dir / "decision_analysis_summary.xlsx",
        {
            "argmax_metrics": argmax_metrics,
            "confusion_error_summary": argmax_errors,
            "binary_threshold_scan": binary_scan,
            "best_binary_thresholds": best_binary,
            "multiclass_threshold_scan": multiclass_scan,
            "best_multiclass_thresholds": best_multiclass,
            "recommendation": recommendations,
        },
    )
    _write_adjusted_predictions(predictions, best_multiclass, output_dir)
    if args.make_figures:
        _make_figures(output_dir, count_matrices, normalized_matrices, binary_scan, multiclass_scan)
    _write_summary_md(
        output_dir / "decision_analysis_summary.md",
        argmax_metrics,
        argmax_errors,
        best_binary,
        best_multiclass,
        recommendations,
    )
    (output_dir / "summary.md").write_text(
        (output_dir / "decision_analysis_summary.md").read_text(encoding="utf-8"),
        encoding="utf-8",
    )

    print(f"Decision analysis written to: {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
