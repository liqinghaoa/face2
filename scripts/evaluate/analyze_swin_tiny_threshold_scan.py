"""OOF threshold scan for the completed Swin-Tiny NYHA three-class experiment.

Experiment A is diagnostic only: it reads existing OOF predictions and never
trains or mutates previous experiment results.
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.metrics import (  # noqa: E402
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    precision_recall_fscore_support,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.simple_xlsx import write_xlsx  # noqa: E402


CLASS_LABELS = [0, 1, 2]
CLASS_NAMES = ["normal", "mild", "severe"]
PROB_COLS = ["prob_normal", "prob_mild", "prob_severe"]

DEFAULT_SWIN_OOF = (
    PROJECT_ROOT
    / "experiments"
    / "model_exploration_500Data"
    / "ModelExploration_SwinTiny_ImageNetMeanBG"
    / "summary"
    / "oof_predictions.csv"
)
DEFAULT_RESNET18_OOF = (
    PROJECT_ROOT
    / "experiments"
    / "preprocess_ablation_500Data"
    / "PreprocAblation_ResNet18_NYHA3Class_hybrid_imagenet_meanbg"
    / "summary"
    / "oof_predictions.csv"
)
DEFAULT_EFFICIENTNET_OOF = (
    PROJECT_ROOT
    / "experiments"
    / "model_exploration_500Data"
    / "ModelExploration_EfficientNetB0_ImageNetMeanBG"
    / "summary"
    / "oof_predictions.csv"
)
DEFAULT_OUTPUT_DIR = (
    PROJECT_ROOT
    / "experiments"
    / "swin_tiny_second_stage_500Data"
    / "threshold_scan"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--swin-oof", type=Path, default=DEFAULT_SWIN_OOF)
    parser.add_argument("--resnet18-oof", type=Path, default=DEFAULT_RESNET18_OOF)
    parser.add_argument("--efficientnet-oof", type=Path, default=DEFAULT_EFFICIENTNET_OOF)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--single-threshold-min", type=float, default=0.05)
    parser.add_argument("--single-threshold-max", type=float, default=0.95)
    parser.add_argument("--single-threshold-step", type=float, default=0.01)
    parser.add_argument("--double-threshold-min", type=float, default=0.05)
    parser.add_argument("--double-threshold-max", type=float, default=0.95)
    parser.add_argument("--double-threshold-step", type=float, default=0.05)
    parser.add_argument("--make-figures", action="store_true")
    return parser.parse_args()


def _resolve(path: Path) -> Path:
    path = path.expanduser()
    return path.resolve() if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def _threshold_values(minimum: float, maximum: float, step: float) -> list[float]:
    if step <= 0:
        raise ValueError("threshold step must be positive")
    values = []
    value = minimum
    while value <= maximum + step / 2:
        values.append(round(float(value), 10))
        value += step
    return values


def _read_oof(path: Path) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(f"OOF predictions not found: {path}")
    frame = pd.read_csv(path, encoding="utf-8-sig")
    missing = [column for column in ["label_3class", *PROB_COLS] if column not in frame.columns]
    if missing:
        raise ValueError(f"{path} is missing required columns: {missing}")
    frame = frame.copy()
    frame["label_3class"] = pd.to_numeric(frame["label_3class"], errors="coerce").astype(int)
    frame[PROB_COLS] = frame[PROB_COLS].apply(pd.to_numeric, errors="coerce")
    if frame[["label_3class", *PROB_COLS]].isna().any().any():
        raise ValueError(f"{path} contains non-numeric label/probability values")
    argmax_pred = np.asarray(CLASS_LABELS)[
        np.argmax(frame[PROB_COLS].to_numpy(dtype=float), axis=1)
    ]
    if "pred_class" in frame.columns:
        pred_class = pd.to_numeric(frame["pred_class"], errors="coerce")
        frame["pred_class"] = (
            argmax_pred if pred_class.isna().any() else pred_class.astype(int).to_numpy()
        )
    else:
        frame["pred_class"] = argmax_pred
    frame["pred_class_argmax"] = argmax_pred
    return frame


def _argmax_two(prob_a: np.ndarray, label_a: int, prob_b: np.ndarray, label_b: int) -> np.ndarray:
    return np.where(prob_a >= prob_b, label_a, label_b).astype(int)


def _metric_row(
    *,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    model_name: str = "swin_t",
    rule_name: str = "argmax",
    threshold_normal: float | None = None,
    threshold_mild: float | None = None,
    threshold_severe: float | None = None,
) -> dict[str, Any]:
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=CLASS_LABELS, zero_division=0
    )
    macro_precision, macro_recall, macro_f1, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=CLASS_LABELS, average="macro", zero_division=0
    )
    _, _, weighted_f1, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=CLASS_LABELS, average="weighted", zero_division=0
    )
    matrix = confusion_matrix(y_true, y_pred, labels=CLASS_LABELS)
    row_sums = matrix.sum(axis=1)

    row: dict[str, Any] = {
        "model_name": model_name,
        "rule_name": rule_name,
        "threshold_normal": threshold_normal,
        "threshold_mild": threshold_mild,
        "threshold_severe": threshold_severe,
        "n_total": int(len(y_true)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_precision": float(macro_precision),
        "macro_recall": float(macro_recall),
        "macro_f1": float(macro_f1),
        "weighted_f1": float(weighted_f1),
    }
    for index, class_name in enumerate(CLASS_NAMES):
        row[f"support_{class_name}"] = int(support[index])
        row[f"precision_{class_name}"] = float(precision[index])
        row[f"recall_{class_name}"] = float(recall[index])
        row[f"f1_{class_name}"] = float(f1[index])
    for true_index, true_name in enumerate(CLASS_NAMES):
        for pred_index, pred_name in enumerate(CLASS_NAMES):
            row[f"{true_name}_to_{pred_name}"] = int(matrix[true_index, pred_index])
    row["mild_to_normal_rate"] = _rate(row["mild_to_normal"], row_sums[1])
    row["mild_to_severe_rate"] = _rate(row["mild_to_severe"], row_sums[1])
    row["severe_to_normal_rate"] = _rate(row["severe_to_normal"], row_sums[2])
    row["severe_to_mild_rate"] = _rate(row["severe_to_mild"], row_sums[2])
    return row


def _rate(numerator: float, denominator: float) -> float:
    return math.nan if denominator == 0 else float(numerator) / float(denominator)


def _confusion_frames(model_name: str, y_true: np.ndarray, y_pred: np.ndarray) -> tuple[pd.DataFrame, pd.DataFrame]:
    matrix = confusion_matrix(y_true, y_pred, labels=CLASS_LABELS)
    row_sums = matrix.sum(axis=1, keepdims=True)
    normalized = np.divide(
        matrix,
        row_sums,
        out=np.full_like(matrix, np.nan, dtype=float),
        where=row_sums != 0,
    )
    count_rows = []
    norm_rows = []
    for true_index, true_name in enumerate(CLASS_NAMES):
        for pred_index, pred_name in enumerate(CLASS_NAMES):
            count_rows.append(
                {
                    "model_name": model_name,
                    "true_label": true_name,
                    "pred_label": pred_name,
                    "count": int(matrix[true_index, pred_index]),
                }
            )
            norm_rows.append(
                {
                    "model_name": model_name,
                    "true_label": true_name,
                    "pred_label": pred_name,
                    "row_normalized": float(normalized[true_index, pred_index]),
                }
            )
    return pd.DataFrame(count_rows), pd.DataFrame(norm_rows)


def _single_threshold_scan(swin: pd.DataFrame, thresholds: list[float]) -> pd.DataFrame:
    y_true = swin["label_3class"].to_numpy(dtype=int)
    p_normal = swin["prob_normal"].to_numpy(dtype=float)
    p_mild = swin["prob_mild"].to_numpy(dtype=float)
    p_severe = swin["prob_severe"].to_numpy(dtype=float)
    rows: list[dict[str, Any]] = []
    for threshold in thresholds:
        severe_else = _argmax_two(p_normal, 0, p_mild, 1)
        pred = np.where(p_severe >= threshold, 2, severe_else).astype(int)
        rows.append(
            _metric_row(
                y_true=y_true,
                y_pred=pred,
                rule_name="severe_priority",
                threshold_severe=threshold,
            )
        )

        mild_else = _argmax_two(p_normal, 0, p_severe, 2)
        pred = np.where(p_mild >= threshold, 1, mild_else).astype(int)
        rows.append(
            _metric_row(
                y_true=y_true,
                y_pred=pred,
                rule_name="mild_priority",
                threshold_mild=threshold,
            )
        )

        normal_else = _argmax_two(p_mild, 1, p_severe, 2)
        pred = np.where(p_normal >= threshold, 0, normal_else).astype(int)
        rows.append(
            _metric_row(
                y_true=y_true,
                y_pred=pred,
                rule_name="normal_priority",
                threshold_normal=threshold,
            )
        )
    return pd.DataFrame(rows)


def _double_threshold_scan(swin: pd.DataFrame, thresholds: list[float]) -> pd.DataFrame:
    y_true = swin["label_3class"].to_numpy(dtype=int)
    p_normal = swin["prob_normal"].to_numpy(dtype=float)
    p_severe = swin["prob_severe"].to_numpy(dtype=float)
    rows: list[dict[str, Any]] = []
    for threshold_normal in thresholds:
        for threshold_severe in thresholds:
            pred = np.where(
                (p_normal >= threshold_normal) & (p_normal >= p_severe),
                0,
                np.where(
                    (p_severe >= threshold_severe) & (p_severe > p_normal),
                    2,
                    1,
                ),
            ).astype(int)
            rows.append(
                _metric_row(
                    y_true=y_true,
                    y_pred=pred,
                    rule_name="middle_fallback",
                    threshold_normal=threshold_normal,
                    threshold_severe=threshold_severe,
                )
            )

            pred = np.where(
                p_normal >= threshold_normal,
                0,
                np.where(p_severe >= threshold_severe, 2, 1),
            ).astype(int)
            rows.append(
                _metric_row(
                    y_true=y_true,
                    y_pred=pred,
                    rule_name="balanced_middle_fallback",
                    threshold_normal=threshold_normal,
                    threshold_severe=threshold_severe,
                )
            )
    return pd.DataFrame(rows)


def _select_best_rules(
    all_rules: pd.DataFrame,
    swin_original: pd.Series,
    resnet18_original: pd.Series,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []

    def add(strategy_name: str, candidates: pd.DataFrame, metric: str, recommendation: str) -> None:
        if candidates.empty:
            row = {column: math.nan for column in all_rules.columns}
            row.update(
                {
                    "strategy_name": strategy_name,
                    "rule_name": "no_candidate",
                    "recommendation": "no_candidate",
                }
            )
            rows.append(row)
            return
        index = pd.to_numeric(candidates[metric], errors="coerce").idxmax()
        row = candidates.loc[index].to_dict()
        row["strategy_name"] = strategy_name
        row["recommendation"] = recommendation
        rows.append(row)

    add("best_macro_f1", all_rules, "macro_f1", "highest macro_f1 among threshold rules")
    add(
        "best_balanced_accuracy",
        all_rules,
        "balanced_accuracy",
        "highest balanced_accuracy among threshold rules",
    )
    add("best_mild_f1", all_rules, "f1_mild", "highest mild F1 among threshold rules")
    add(
        "best_mild_recall_with_macro_f1_constraint",
        all_rules.loc[all_rules["macro_f1"] >= float(swin_original["macro_f1"])],
        "recall_mild",
        "max mild recall while preserving original Swin macro_f1",
    )
    add(
        "best_macro_f1_with_ba_constraint",
        all_rules.loc[
            all_rules["balanced_accuracy"]
            >= float(swin_original["balanced_accuracy"]) - 0.02
        ],
        "macro_f1",
        "max macro_f1 with balanced_accuracy no more than 0.02 below original Swin",
    )

    scored = all_rules.copy()
    scored["score"] = (
        scored["macro_f1"]
        + scored["balanced_accuracy"]
        + 0.5 * scored["recall_mild"]
        + 0.3 * scored["recall_severe"]
    )
    add(
        "best_balanced_tradeoff",
        scored,
        "score",
        "max macro_f1 + balanced_accuracy + 0.5*mild_recall + 0.3*severe_recall",
    )

    conservative = all_rules.loc[
        (all_rules["macro_f1"] > float(swin_original["macro_f1"]))
        & (
            all_rules["balanced_accuracy"]
            >= float(swin_original["balanced_accuracy"]) - 0.02
        )
        & (all_rules["recall_mild"] > float(swin_original["recall_mild"]))
        & (
            all_rules["recall_severe"]
            >= float(resnet18_original["recall_severe"]) - 0.03
        )
    ]
    add(
        "conservative_candidate",
        conservative,
        "macro_f1",
        "passes macro_f1/BA/mild/severe conservative constraints",
    )

    columns = [
        "strategy_name",
        "rule_name",
        "threshold_normal",
        "threshold_mild",
        "threshold_severe",
        "accuracy",
        "balanced_accuracy",
        "macro_f1",
        "weighted_f1",
        "recall_normal",
        "recall_mild",
        "recall_severe",
        "f1_normal",
        "f1_mild",
        "f1_severe",
        "mild_to_normal",
        "mild_to_severe",
        "severe_to_normal",
        "severe_to_mild",
        "recommendation",
    ]
    return pd.DataFrame(rows).reindex(columns=columns)


def _predict_rule(swin: pd.DataFrame, rule: pd.Series) -> np.ndarray:
    rule_name = str(rule["rule_name"])
    if rule_name == "no_candidate":
        raise ValueError("Cannot predict no_candidate")
    p_normal = swin["prob_normal"].to_numpy(dtype=float)
    p_mild = swin["prob_mild"].to_numpy(dtype=float)
    p_severe = swin["prob_severe"].to_numpy(dtype=float)
    threshold_normal = _float_or_nan(rule.get("threshold_normal"))
    threshold_mild = _float_or_nan(rule.get("threshold_mild"))
    threshold_severe = _float_or_nan(rule.get("threshold_severe"))
    if rule_name == "severe_priority":
        return np.where(
            p_severe >= threshold_severe,
            2,
            _argmax_two(p_normal, 0, p_mild, 1),
        ).astype(int)
    if rule_name == "mild_priority":
        return np.where(
            p_mild >= threshold_mild,
            1,
            _argmax_two(p_normal, 0, p_severe, 2),
        ).astype(int)
    if rule_name == "normal_priority":
        return np.where(
            p_normal >= threshold_normal,
            0,
            _argmax_two(p_mild, 1, p_severe, 2),
        ).astype(int)
    if rule_name == "middle_fallback":
        return np.where(
            (p_normal >= threshold_normal) & (p_normal >= p_severe),
            0,
            np.where(
                (p_severe >= threshold_severe) & (p_severe > p_normal),
                2,
                1,
            ),
        ).astype(int)
    if rule_name == "balanced_middle_fallback":
        return np.where(
            p_normal >= threshold_normal,
            0,
            np.where(p_severe >= threshold_severe, 2, 1),
        ).astype(int)
    raise ValueError(f"Unknown rule_name: {rule_name}")


def _float_or_nan(value: Any) -> float:
    try:
        return float(value)
    except Exception:
        return math.nan


def _write_adjusted_predictions(
    swin: pd.DataFrame,
    best_rules: pd.DataFrame,
    output_dir: Path,
) -> None:
    target_strategies = {
        "best_macro_f1",
        "best_balanced_accuracy",
        "best_mild_f1",
        "best_balanced_tradeoff",
        "conservative_candidate",
    }
    adjusted_dir = output_dir / "adjusted_predictions"
    adjusted_dir.mkdir(parents=True, exist_ok=True)
    base_columns = [
        "ID",
        "patient_group_id",
        "NYHA",
        "label_3class",
        "label_3class_name",
    ]
    available_base_columns = [column for column in base_columns if column in swin.columns]
    for _, rule in best_rules.iterrows():
        strategy_name = str(rule["strategy_name"])
        if strategy_name not in target_strategies or str(rule["rule_name"]) == "no_candidate":
            continue
        pred_adjusted = _predict_rule(swin, rule)
        frame = swin[available_base_columns + PROB_COLS].copy()
        frame["pred_class_original"] = swin["pred_class"].astype(int).to_numpy()
        frame["pred_class_adjusted"] = pred_adjusted
        labels = swin["label_3class"].astype(int).to_numpy()
        frame["correct_original"] = frame["pred_class_original"].to_numpy() == labels
        frame["correct_adjusted"] = pred_adjusted == labels
        frame["rule_name"] = rule["rule_name"]
        frame["threshold_normal"] = rule.get("threshold_normal")
        frame["threshold_mild"] = rule.get("threshold_mild")
        frame["threshold_severe"] = rule.get("threshold_severe")
        frame["strategy_name"] = strategy_name
        ordered = (
            available_base_columns
            + [
                "pred_class_original",
                "pred_class_adjusted",
                *PROB_COLS,
                "correct_original",
                "correct_adjusted",
                "rule_name",
                "threshold_normal",
                "threshold_mild",
                "threshold_severe",
                "strategy_name",
            ]
        )
        frame[ordered].to_csv(
            adjusted_dir / f"{strategy_name}.csv",
            index=False,
            encoding="utf-8-sig",
        )


def _plot_confusion(matrix: np.ndarray, path: Path, title: str, normalized: bool) -> None:
    fig, ax = plt.subplots(figsize=(5.2, 4.5), dpi=160)
    image = ax.imshow(matrix, cmap="Blues", vmin=0)
    ax.set_xticks(np.arange(3), labels=CLASS_NAMES)
    ax.set_yticks(np.arange(3), labels=CLASS_NAMES)
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_title(title)
    fmt = ".2f" if normalized else ".0f"
    max_value = np.nanmax(matrix) if np.isfinite(matrix).any() else 0
    for i in range(3):
        for j in range(3):
            value = matrix[i, j]
            text = "" if not math.isfinite(float(value)) else format(value, fmt)
            color = "white" if math.isfinite(float(value)) and value > max_value / 2 else "black"
            ax.text(j, i, text, ha="center", va="center", color=color, fontsize=9)
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)


def _make_figures(
    swin: pd.DataFrame,
    single_scan: pd.DataFrame,
    double_scan: pd.DataFrame,
    output_dir: Path,
) -> None:
    figure_dir = output_dir / "figures"
    y_true = swin["label_3class"].to_numpy(dtype=int)
    y_pred = swin["pred_class"].to_numpy(dtype=int)
    count_matrix = confusion_matrix(y_true, y_pred, labels=CLASS_LABELS)
    row_sums = count_matrix.sum(axis=1, keepdims=True)
    norm_matrix = np.divide(
        count_matrix,
        row_sums,
        out=np.full_like(count_matrix, np.nan, dtype=float),
        where=row_sums != 0,
    )
    _plot_confusion(
        count_matrix.astype(float),
        figure_dir / "swin_argmax_confusion_count.png",
        "Swin-Tiny argmax confusion matrix (count)",
        normalized=False,
    )
    _plot_confusion(
        norm_matrix,
        figure_dir / "swin_argmax_confusion_row_normalized.png",
        "Swin-Tiny argmax confusion matrix (row-normalized)",
        normalized=True,
    )

    fig, ax = plt.subplots(figsize=(7.2, 4.8), dpi=160)
    for rule_name, group in single_scan.groupby("rule_name", sort=False):
        threshold_column = {
            "severe_priority": "threshold_severe",
            "mild_priority": "threshold_mild",
            "normal_priority": "threshold_normal",
        }[rule_name]
        ordered = group.sort_values(threshold_column)
        ax.plot(ordered[threshold_column], ordered["macro_f1"], label=f"{rule_name} macro_f1")
        if rule_name == "mild_priority":
            ax.plot(
                ordered[threshold_column],
                ordered["recall_mild"],
                linestyle="--",
                label="mild_priority recall_mild",
            )
    ax.set_xlabel("threshold")
    ax.set_ylabel("metric")
    ax.set_title("Swin-Tiny single-threshold curves")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(figure_dir / "swin_single_threshold_curves.png")
    plt.close(fig)

    middle = double_scan.loc[double_scan["rule_name"] == "middle_fallback"].copy()
    for metric, filename in [
        ("macro_f1", "swin_double_threshold_heatmap_macro_f1.png"),
        ("balanced_accuracy", "swin_double_threshold_heatmap_balanced_accuracy.png"),
        ("recall_mild", "swin_double_threshold_heatmap_mild_recall.png"),
    ]:
        pivot = middle.pivot(
            index="threshold_severe", columns="threshold_normal", values=metric
        ).sort_index(ascending=True)
        fig, ax = plt.subplots(figsize=(7.0, 5.8), dpi=160)
        image = ax.imshow(pivot.to_numpy(), origin="lower", aspect="auto", cmap="viridis")
        ax.set_xticks(np.arange(len(pivot.columns)))
        ax.set_yticks(np.arange(len(pivot.index)))
        ax.set_xticklabels([f"{x:.2f}" for x in pivot.columns], rotation=90, fontsize=6)
        ax.set_yticklabels([f"{y:.2f}" for y in pivot.index], fontsize=6)
        ax.set_xlabel("threshold_normal")
        ax.set_ylabel("threshold_severe")
        ax.set_title(f"middle_fallback {metric}")
        fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
        fig.tight_layout()
        fig.savefig(figure_dir / filename)
        plt.close(fig)


def _format(value: Any) -> str:
    try:
        number = float(value)
    except Exception:
        return str(value)
    if math.isnan(number):
        return "nan"
    return f"{number:.4f}"


def _write_summary_md(
    path: Path,
    argmax_metrics: pd.DataFrame,
    best_rules: pd.DataFrame,
    single_scan: pd.DataFrame,
    double_scan: pd.DataFrame,
) -> None:
    swin = argmax_metrics.loc[argmax_metrics["model_name"] == "swin_t"].iloc[0]
    resnet = argmax_metrics.loc[argmax_metrics["model_name"] == "resnet18_meanbg"].iloc[0]
    best_macro = best_rules.loc[best_rules["strategy_name"] == "best_macro_f1"].iloc[0]
    best_ba = best_rules.loc[best_rules["strategy_name"] == "best_balanced_accuracy"].iloc[0]
    conservative = best_rules.loc[
        best_rules["strategy_name"] == "conservative_candidate"
    ].iloc[0]
    mild_best = single_scan.loc[
        single_scan["rule_name"] == "mild_priority"
    ].sort_values("f1_mild", ascending=False).iloc[0]
    middle_best = double_scan.loc[
        double_scan["rule_name"] == "middle_fallback"
    ].sort_values("macro_f1", ascending=False).iloc[0]

    threshold_beats_resnet = bool(float(best_macro["macro_f1"]) >= float(resnet["macro_f1"]))
    lines = [
        "# Swin-Tiny OOF Threshold Scan Summary",
        "",
        "## Purpose",
        "",
        "This analysis checks whether Swin-Tiny's OOF probability ranking can be converted into better hard NYHA three-class predictions by threshold rules. No model training was performed.",
        "",
        "## 1. Original Swin-Tiny problem",
        "",
        (
            f"Original Swin-Tiny has high balanced accuracy ({_format(swin['balanced_accuracy'])}) "
            f"but lower macro-F1 ({_format(swin['macro_f1'])}) because mild hard classification is weak: "
            f"mild recall={_format(swin['recall_mild'])}, mild F1={_format(swin['f1_mild'])}."
        ),
        "",
        "## 2. Does mild-priority recover mild?",
        "",
        (
            f"Best mild-priority mild F1 row: threshold_mild={_format(mild_best['threshold_mild'])}, "
            f"macro-F1={_format(mild_best['macro_f1'])}, BA={_format(mild_best['balanced_accuracy'])}, "
            f"mild recall={_format(mild_best['recall_mild'])}, mild F1={_format(mild_best['f1_mild'])}."
        ),
        "",
        "## 3. Does middle-fallback improve macro-F1?",
        "",
        (
            f"Best middle-fallback macro-F1 row: threshold_normal={_format(middle_best['threshold_normal'])}, "
            f"threshold_severe={_format(middle_best['threshold_severe'])}, "
            f"macro-F1={_format(middle_best['macro_f1'])}, BA={_format(middle_best['balanced_accuracy'])}, "
            f"mild recall={_format(middle_best['recall_mild'])}."
        ),
        "",
        "## 4. Best macro-F1 threshold rule",
        "",
        (
            f"{best_macro['rule_name']} with threshold_normal={_format(best_macro['threshold_normal'])}, "
            f"threshold_mild={_format(best_macro['threshold_mild'])}, "
            f"threshold_severe={_format(best_macro['threshold_severe'])}: "
            f"macro-F1={_format(best_macro['macro_f1'])}, BA={_format(best_macro['balanced_accuracy'])}."
        ),
        "",
        "## 5. Best BA threshold rule",
        "",
        (
            f"{best_ba['rule_name']} with threshold_normal={_format(best_ba['threshold_normal'])}, "
            f"threshold_mild={_format(best_ba['threshold_mild'])}, "
            f"threshold_severe={_format(best_ba['threshold_severe'])}: "
            f"BA={_format(best_ba['balanced_accuracy'])}, macro-F1={_format(best_ba['macro_f1'])}."
        ),
        "",
        "## 6. Conservative candidate",
        "",
        (
            "Exists: no."
            if str(conservative["rule_name"]) == "no_candidate"
            else (
                f"Exists: yes. {conservative['rule_name']} "
                f"(tn={_format(conservative['threshold_normal'])}, "
                f"tm={_format(conservative['threshold_mild'])}, "
                f"ts={_format(conservative['threshold_severe'])}) "
                f"macro-F1={_format(conservative['macro_f1'])}, "
                f"BA={_format(conservative['balanced_accuracy'])}, "
                f"mild recall={_format(conservative['recall_mild'])}, "
                f"severe recall={_format(conservative['recall_severe'])}."
            )
        ),
        "",
        "## 7. Does thresholding approach or exceed ResNet18 macro-F1?",
        "",
        (
            f"ResNet18 meanbg macro-F1={_format(resnet['macro_f1'])}. "
            f"Best threshold macro-F1={_format(best_macro['macro_f1'])}. "
            f"Approach/exceed ResNet18: {'yes' if threshold_beats_resnet else 'no'}."
        ),
        "",
        "## 8. Continue Swin-Tiny training tuning?",
        "",
        "Yes. Threshold scan is useful diagnostically, but lr=5e-5 and label smoothing=0.05 are still needed to check whether the hard-decision problem can be improved at training time.",
        "",
        "## 9. Caution",
        "",
        "This threshold scan is based on OOF predictions from the same five-fold training protocol. It is an internal diagnostic and must not be treated as an externally validated clinical threshold.",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    output_dir = _resolve(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    swin = _read_oof(_resolve(args.swin_oof))
    resnet18 = _read_oof(_resolve(args.resnet18_oof))
    model_oofs = {
        "swin_t": swin,
        "resnet18_meanbg": resnet18,
    }
    efficientnet_path = _resolve(args.efficientnet_oof)
    if efficientnet_path.is_file():
        model_oofs["efficientnet_b0"] = _read_oof(efficientnet_path)

    argmax_rows = []
    confusion_count_frames = []
    confusion_norm_frames = []
    for model_name, frame in model_oofs.items():
        y_true = frame["label_3class"].to_numpy(dtype=int)
        y_pred = frame["pred_class"].to_numpy(dtype=int)
        argmax_rows.append(
            _metric_row(y_true=y_true, y_pred=y_pred, model_name=model_name)
        )
        count_frame, norm_frame = _confusion_frames(model_name, y_true, y_pred)
        confusion_count_frames.append(count_frame)
        confusion_norm_frames.append(norm_frame)

    argmax_metrics = pd.DataFrame(argmax_rows)
    confusion_count = pd.concat(confusion_count_frames, ignore_index=True)
    confusion_row_normalized = pd.concat(confusion_norm_frames, ignore_index=True)

    single_thresholds = _threshold_values(
        args.single_threshold_min,
        args.single_threshold_max,
        args.single_threshold_step,
    )
    double_thresholds = _threshold_values(
        args.double_threshold_min,
        args.double_threshold_max,
        args.double_threshold_step,
    )
    single_scan = _single_threshold_scan(swin, single_thresholds)
    double_scan = _double_threshold_scan(swin, double_thresholds)
    all_rules = pd.concat([single_scan, double_scan], ignore_index=True)

    swin_original = argmax_metrics.loc[argmax_metrics["model_name"] == "swin_t"].iloc[0]
    resnet18_original = argmax_metrics.loc[
        argmax_metrics["model_name"] == "resnet18_meanbg"
    ].iloc[0]
    best_rules = _select_best_rules(all_rules, swin_original, resnet18_original)
    _write_adjusted_predictions(swin, best_rules, output_dir)

    recommendation = pd.DataFrame(
        [
            {
                "item": "main_finding",
                "value": (
                    "Thresholding can recover mild only if macro-F1/BA tradeoffs are "
                    "acceptable; this is OOF diagnostic evidence, not a clinical threshold."
                ),
            },
            {
                "item": "continue_training_tuning",
                "value": "yes: run lr=5e-5 and label_smoothing=0.05.",
            },
        ]
    )

    argmax_metrics.to_csv(output_dir / "argmax_metrics.csv", index=False, encoding="utf-8-sig")
    single_scan.to_csv(
        output_dir / "single_threshold_scan.csv", index=False, encoding="utf-8-sig"
    )
    double_scan.to_csv(
        output_dir / "double_threshold_scan.csv", index=False, encoding="utf-8-sig"
    )
    best_rules.to_csv(
        output_dir / "best_threshold_rules.csv", index=False, encoding="utf-8-sig"
    )
    write_xlsx(
        output_dir / "confusion_matrices.xlsx",
        {
            "confusion_count": confusion_count,
            "confusion_row_normalized": confusion_row_normalized,
        },
    )
    write_xlsx(
        output_dir / "swin_threshold_scan_summary.xlsx",
        {
            "argmax_metrics": argmax_metrics,
            "single_threshold_scan": single_scan,
            "double_threshold_scan": double_scan,
            "best_threshold_rules": best_rules,
            "confusion_count": confusion_count,
            "confusion_row_normalized": confusion_row_normalized,
            "recommendation": recommendation,
        },
    )
    _write_summary_md(
        output_dir / "swin_threshold_scan_summary.md",
        argmax_metrics,
        best_rules,
        single_scan,
        double_scan,
    )
    if args.make_figures:
        _make_figures(swin, single_scan, double_scan, output_dir)

    print(f"ARGMAX_METRICS={output_dir / 'argmax_metrics.csv'}")
    print(f"SINGLE_THRESHOLD_SCAN={output_dir / 'single_threshold_scan.csv'}")
    print(f"DOUBLE_THRESHOLD_SCAN={output_dir / 'double_threshold_scan.csv'}")
    print(f"BEST_THRESHOLD_RULES={output_dir / 'best_threshold_rules.csv'}")
    print(f"SUMMARY_XLSX={output_dir / 'swin_threshold_scan_summary.xlsx'}")
    print(f"SUMMARY_MD={output_dir / 'swin_threshold_scan_summary.md'}")


if __name__ == "__main__":
    main()
