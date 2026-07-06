"""Diagnose completed backbone model-exploration experiments.

This script is intentionally read-only with respect to training results: it reads
existing fold/OOF summaries and writes a separate diagnostic package under
experiments/model_exploration_500Data/diagnostic_analysis.
"""

from __future__ import annotations

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
    roc_auc_score,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.simple_xlsx import write_xlsx  # noqa: E402


EXPERIMENT_ROOT = PROJECT_ROOT / "experiments" / "model_exploration_500Data"
BASELINE_DIR = (
    PROJECT_ROOT
    / "experiments"
    / "preprocess_ablation_500Data"
    / "PreprocAblation_ResNet18_NYHA3Class_hybrid_imagenet_meanbg"
)
OUTPUT_DIR = EXPERIMENT_ROOT / "diagnostic_analysis"
FIGURE_DIR = OUTPUT_DIR / "figures"

REPORT_PATH = OUTPUT_DIR / "model_exploration_diagnostic_report.md"
TABLES_XLSX_PATH = OUTPUT_DIR / "model_exploration_diagnostic_tables.xlsx"
CONFUSION_CSV_PATH = OUTPUT_DIR / "model_exploration_confusion_summary.csv"
ARGMAX_CSV_PATH = OUTPUT_DIR / "model_exploration_argmax_metrics.csv"
PER_CLASS_CSV_PATH = OUTPUT_DIR / "model_exploration_per_class_metrics.csv"
AUC_CSV_PATH = OUTPUT_DIR / "model_exploration_oof_auc_summary.csv"

SUMMARY_CSV_PATH = EXPERIMENT_ROOT / "model_exploration_summary.csv"
SUMMARY_MD_PATH = EXPERIMENT_ROOT / "model_exploration_summary.md"
QUEUE_CSV_PATH = EXPERIMENT_ROOT / "model_exploration_job_queue.csv"

CLASS_LABELS = [0, 1, 2]
CLASS_NAMES = ["normal", "mild", "severe"]
LABEL_TO_NAME = dict(zip(CLASS_LABELS, CLASS_NAMES))
PROB_COLS = ["prob_normal", "prob_mild", "prob_severe"]

MODEL_DIRS: dict[str, Path] = {
    "resnet18_meanbg": BASELINE_DIR,
    "densenet121": EXPERIMENT_ROOT / "ModelExploration_DenseNet121_ImageNetMeanBG",
    "efficientnet_b0": EXPERIMENT_ROOT / "ModelExploration_EfficientNetB0_ImageNetMeanBG",
    "convnext_tiny": EXPERIMENT_ROOT / "ModelExploration_ConvNeXtTiny_ImageNetMeanBG",
    "swin_t": EXPERIMENT_ROOT / "ModelExploration_SwinTiny_ImageNetMeanBG",
    "mobilenet_v3_large": EXPERIMENT_ROOT
    / "ModelExploration_MobileNetV3Large_ImageNetMeanBG",
}

REQUIRED_SUMMARY_FILES = [
    "summary/fold_metrics_all.csv",
    "summary/mean_metrics.csv",
    "summary/oof_metrics.csv",
    "summary/oof_predictions.csv",
    "summary/summary_report.md",
]


def _read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, encoding="utf-8-sig")


def _safe_float(value: Any) -> float:
    try:
        if value is None:
            return math.nan
        if isinstance(value, str) and value.strip() == "":
            return math.nan
        result = float(value)
        return result if math.isfinite(result) else math.nan
    except Exception:
        return math.nan


def _safe_int(value: Any) -> int | None:
    try:
        if pd.isna(value):
            return None
        return int(float(value))
    except Exception:
        return None


def _fmt(value: Any, digits: int = 4) -> str:
    number = _safe_float(value)
    if math.isfinite(number):
        return f"{number:.{digits}f}"
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return ""
    return str(value)


def _fmt_int(value: Any) -> str:
    intval = _safe_int(value)
    return "" if intval is None else str(intval)


def _rate(numerator: float, denominator: float) -> float:
    if denominator == 0:
        return math.nan
    return float(numerator) / float(denominator)


def _metric_mean(frame: pd.DataFrame, column: str) -> float:
    if frame.empty or column not in frame.columns:
        return math.nan
    return float(pd.to_numeric(frame[column], errors="coerce").mean())


def _metric_std(frame: pd.DataFrame, column: str) -> float:
    if frame.empty or column not in frame.columns:
        return math.nan
    return float(pd.to_numeric(frame[column], errors="coerce").std(ddof=1))


def _metric_values(frame: pd.DataFrame, column: str) -> list[float]:
    if frame.empty or column not in frame.columns:
        return []
    values = pd.to_numeric(frame[column], errors="coerce").dropna()
    return [float(v) for v in values.tolist()]


def _load_summary_lookup() -> dict[str, dict[str, Any]]:
    if not SUMMARY_CSV_PATH.is_file():
        return {}
    frame = _read_csv(SUMMARY_CSV_PATH)
    if "backbone" not in frame.columns:
        return {}
    return {
        str(row["backbone"]): row.to_dict()
        for _, row in frame.iterrows()
        if str(row.get("backbone", "")).strip()
    }


def _load_queue_lookup() -> dict[str, dict[str, Any]]:
    if not QUEUE_CSV_PATH.is_file():
        return {}
    frame = _read_csv(QUEUE_CSV_PATH)
    if "backbone" not in frame.columns:
        return {}
    return {
        str(row["backbone"]): row.to_dict()
        for _, row in frame.iterrows()
        if str(row.get("backbone", "")).strip()
    }


def _read_model_summary_params(model_dir: Path) -> tuple[float, float]:
    summary_path = model_dir / "model_summary.txt"
    if not summary_path.is_file():
        return math.nan, math.nan
    values: dict[str, str] = {}
    for line in summary_path.read_text(encoding="utf-8", errors="replace").splitlines():
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        values[key.strip()] = value.strip()
    return _safe_float(values.get("total_params")), _safe_float(
        values.get("trainable_params")
    )


def _resolve_params(
    model_name: str,
    model_dir: Path,
    summary_lookup: dict[str, dict[str, Any]],
    queue_lookup: dict[str, dict[str, Any]],
) -> tuple[float, float]:
    for lookup in (summary_lookup, queue_lookup):
        row = lookup.get(model_name)
        if not row:
            continue
        total = _safe_float(row.get("total_params"))
        trainable = _safe_float(row.get("trainable_params"))
        if math.isfinite(total):
            return total, trainable if math.isfinite(trainable) else total
    total, trainable = _read_model_summary_params(model_dir)
    return total, trainable


def _check_files() -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for path in [SUMMARY_CSV_PATH, SUMMARY_MD_PATH, QUEUE_CSV_PATH]:
        rows.append(
            {
                "scope": "model_exploration_root",
                "model_name": "",
                "relative_path": str(path.relative_to(PROJECT_ROOT)),
                "absolute_path": str(path),
                "exists": path.is_file(),
                "size_bytes": path.stat().st_size if path.is_file() else math.nan,
            }
        )
    for model_name, model_dir in MODEL_DIRS.items():
        for rel in REQUIRED_SUMMARY_FILES:
            path = model_dir / rel
            rows.append(
                {
                    "scope": "model",
                    "model_name": model_name,
                    "relative_path": str(path.relative_to(PROJECT_ROOT))
                    if path.is_absolute()
                    else str(path),
                    "absolute_path": str(path),
                    "exists": path.is_file(),
                    "size_bytes": path.stat().st_size if path.is_file() else math.nan,
                }
            )
    return pd.DataFrame(rows)


def _prepare_oof_predictions(path: Path) -> pd.DataFrame:
    frame = _read_csv(path)
    required = ["label_3class", *PROB_COLS]
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise ValueError(f"{path} missing required columns: {missing}")

    frame = frame.copy()
    labels = pd.to_numeric(frame["label_3class"], errors="coerce")
    if labels.isna().any():
        raise ValueError(f"{path} has non-numeric label_3class values.")
    frame["label_3class"] = labels.astype(int)

    prob_values = frame[PROB_COLS].apply(pd.to_numeric, errors="coerce")
    if prob_values.isna().any().any():
        raise ValueError(f"{path} has non-numeric probability values.")
    frame[PROB_COLS] = prob_values

    argmax_pred = np.asarray(CLASS_LABELS, dtype=int)[
        np.argmax(frame[PROB_COLS].to_numpy(dtype=float), axis=1)
    ]
    if "pred_class" not in frame.columns:
        frame["pred_class"] = argmax_pred
        frame["pred_class_source"] = "computed_argmax"
    else:
        pred = pd.to_numeric(frame["pred_class"], errors="coerce")
        if pred.isna().any():
            frame["pred_class"] = argmax_pred
            frame["pred_class_source"] = "computed_argmax_missing_values"
        else:
            frame["pred_class"] = pred.astype(int)
            frame["pred_class_source"] = "existing_pred_class"
    frame["pred_class_argmax"] = argmax_pred
    return frame


def _safe_binary_auc(y_true: np.ndarray, score: np.ndarray) -> float:
    if len(np.unique(y_true)) < 2:
        return math.nan
    try:
        return float(roc_auc_score(y_true, score))
    except Exception:
        return math.nan


def _safe_ovr_auc(y_true: np.ndarray, scores: np.ndarray, average: str) -> float:
    try:
        return float(
            roc_auc_score(
                y_true,
                scores,
                labels=CLASS_LABELS,
                multi_class="ovr",
                average=average,
            )
        )
    except Exception:
        return math.nan


def _compute_argmax_metrics(model_name: str, oof: pd.DataFrame) -> dict[str, Any]:
    y_true = oof["label_3class"].to_numpy(dtype=int)
    y_pred = oof["pred_class"].to_numpy(dtype=int)

    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=CLASS_LABELS, zero_division=0
    )
    macro_precision, macro_recall, macro_f1, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=CLASS_LABELS, average="macro", zero_division=0
    )
    _, _, weighted_f1, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=CLASS_LABELS, average="weighted", zero_division=0
    )
    row: dict[str, Any] = {
        "model_name": model_name,
        "n_total": int(len(oof)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_precision": float(macro_precision),
        "macro_recall": float(macro_recall),
        "macro_f1": float(macro_f1),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "weighted_f1": float(weighted_f1),
    }
    for index, class_name in enumerate(CLASS_NAMES):
        row[f"support_{class_name}"] = int(support[index])
        row[f"recall_{class_name}"] = float(recall[index])
        row[f"precision_{class_name}"] = float(precision[index])
        row[f"f1_{class_name}"] = float(f1[index])
    return row


def _compute_per_class_metrics(model_name: str, oof: pd.DataFrame) -> pd.DataFrame:
    y_true = oof["label_3class"].to_numpy(dtype=int)
    y_pred = oof["pred_class"].to_numpy(dtype=int)
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=CLASS_LABELS, zero_division=0
    )
    rows = []
    for index, class_name in enumerate(CLASS_NAMES):
        rows.append(
            {
                "model_name": model_name,
                "class_id": CLASS_LABELS[index],
                "class_name": class_name,
                "support": int(support[index]),
                "precision": float(precision[index]),
                "recall": float(recall[index]),
                "f1": float(f1[index]),
            }
        )
    return pd.DataFrame(rows)


def _compute_oof_auc(model_name: str, oof: pd.DataFrame) -> dict[str, Any]:
    y_true = oof["label_3class"].to_numpy(dtype=int)
    scores = oof[PROB_COLS].to_numpy(dtype=float)
    row: dict[str, Any] = {
        "model_name": model_name,
        "n_total": int(len(oof)),
        "macro_auc_ovr": _safe_ovr_auc(y_true, scores, "macro"),
        "weighted_auc_ovr": _safe_ovr_auc(y_true, scores, "weighted"),
    }
    for index, class_name in enumerate(CLASS_NAMES):
        row[f"auc_{class_name}"] = _safe_binary_auc(
            (y_true == CLASS_LABELS[index]).astype(int), scores[:, index]
        )
    row["normal_vs_abnormal_auc"] = _safe_binary_auc(
        (y_true == 0).astype(int), oof["prob_normal"].to_numpy(dtype=float)
    )
    row["severe_vs_rest_auc"] = _safe_binary_auc(
        (y_true == 2).astype(int), oof["prob_severe"].to_numpy(dtype=float)
    )
    return row


def _compute_confusion(model_name: str, oof: pd.DataFrame) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame]:
    y_true = oof["label_3class"].to_numpy(dtype=int)
    y_pred = oof["pred_class"].to_numpy(dtype=int)
    counts = confusion_matrix(y_true, y_pred, labels=CLASS_LABELS)
    row_sums = counts.sum(axis=1, keepdims=True)
    row_norm = np.divide(
        counts,
        row_sums,
        out=np.full_like(counts, np.nan, dtype=float),
        where=row_sums != 0,
    )

    summary: dict[str, Any] = {
        "model_name": model_name,
        "n_total": int(counts.sum()),
        "normal_total": int(row_sums[0][0]),
        "mild_total": int(row_sums[1][0]),
        "severe_total": int(row_sums[2][0]),
    }
    for true_idx, true_name in enumerate(CLASS_NAMES):
        for pred_idx, pred_name in enumerate(CLASS_NAMES):
            summary[f"{true_name}_to_{pred_name}"] = int(counts[true_idx, pred_idx])
        summary[f"{true_name}_recall"] = float(row_norm[true_idx, true_idx])

    summary["normal_to_severe_rate"] = _rate(
        summary["normal_to_severe"], summary["normal_total"]
    )
    summary["mild_to_severe_rate"] = _rate(
        summary["mild_to_severe"], summary["mild_total"]
    )
    summary["severe_to_normal_rate"] = _rate(
        summary["severe_to_normal"], summary["severe_total"]
    )
    summary["severe_to_mild_rate"] = _rate(
        summary["severe_to_mild"], summary["severe_total"]
    )

    count_long = _matrix_to_long(model_name, counts, "count")
    row_norm_long = _matrix_to_long(model_name, row_norm, "row_normalized")
    return summary, count_long, row_norm_long


def _matrix_to_long(model_name: str, matrix: np.ndarray, value_name: str) -> pd.DataFrame:
    rows = []
    for true_idx, true_name in enumerate(CLASS_NAMES):
        for pred_idx, pred_name in enumerate(CLASS_NAMES):
            rows.append(
                {
                    "model_name": model_name,
                    "true_label": true_name,
                    "pred_label": pred_name,
                    value_name: float(matrix[true_idx, pred_idx]),
                }
            )
    return pd.DataFrame(rows)


def _plot_confusion_matrix(
    model_name: str,
    matrix: np.ndarray,
    *,
    normalized: bool,
    output_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(5.2, 4.6), dpi=160)
    image = ax.imshow(matrix, cmap="Blues", vmin=0)
    ax.set_xticks(np.arange(len(CLASS_NAMES)), labels=CLASS_NAMES)
    ax.set_yticks(np.arange(len(CLASS_NAMES)), labels=CLASS_NAMES)
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    title_suffix = "row-normalized" if normalized else "count"
    ax.set_title(f"{model_name} confusion matrix ({title_suffix})")
    fmt = ".2f" if normalized else ".0f"
    max_value = np.nanmax(matrix) if np.isfinite(matrix).any() else 0
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            value = matrix[i, j]
            text = "" if not math.isfinite(float(value)) else format(value, fmt)
            color = "white" if math.isfinite(float(value)) and value > max_value / 2 else "black"
            ax.text(j, i, text, ha="center", va="center", color=color, fontsize=9)
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path)
    plt.close(fig)


def _write_figures(model_name: str, oof: pd.DataFrame) -> None:
    y_true = oof["label_3class"].to_numpy(dtype=int)
    y_pred = oof["pred_class"].to_numpy(dtype=int)
    counts = confusion_matrix(y_true, y_pred, labels=CLASS_LABELS)
    row_sums = counts.sum(axis=1, keepdims=True)
    row_norm = np.divide(
        counts,
        row_sums,
        out=np.full_like(counts, np.nan, dtype=float),
        where=row_sums != 0,
    )
    _plot_confusion_matrix(
        model_name,
        counts.astype(float),
        normalized=False,
        output_path=FIGURE_DIR / f"{model_name}_confusion_count.png",
    )
    _plot_confusion_matrix(
        model_name,
        row_norm,
        normalized=True,
        output_path=FIGURE_DIR / f"{model_name}_confusion_row_normalized.png",
    )


def _compute_fold_stability(model_name: str, fold_metrics: pd.DataFrame) -> dict[str, Any]:
    metrics = ["macro_auc", "balanced_accuracy", "macro_f1", "recall_severe"]
    row: dict[str, Any] = {"model_name": model_name, "n_folds": int(len(fold_metrics))}
    stds = []
    for metric in metrics:
        values = _metric_values(fold_metrics, metric)
        row[f"{metric}_mean"] = float(np.nanmean(values)) if values else math.nan
        row[f"{metric}_std"] = float(np.nanstd(values, ddof=1)) if len(values) > 1 else math.nan
        row[f"{metric}_min"] = float(np.nanmin(values)) if values else math.nan
        row[f"{metric}_max"] = float(np.nanmax(values)) if values else math.nan
        row[f"{metric}_values"] = ", ".join(f"{value:.4f}" for value in values)
        if len(values) > 1:
            stds.append(float(np.nanstd(values, ddof=1)))
    row["stability_std_mean"] = float(np.nanmean(stds)) if stds else math.nan
    return row


def _recommendation(model_name: str) -> str:
    if model_name == "resnet18_meanbg":
        return "baseline_reference"
    if model_name == "swin_t":
        return "main_candidate_threshold_scan_and_light_tuning"
    if model_name == "efficientnet_b0":
        return "lightweight_candidate"
    if model_name == "densenet121":
        return "auc_improved_but_hard_classification_not_improved"
    if model_name == "convnext_tiny":
        return "severe_sensitive_but_unbalanced_not_main_model"
    if model_name == "mobilenet_v3_large":
        return "stop_advancing"
    return "review_manually"


def _markdown_table(frame: pd.DataFrame, columns: list[str], digits: int = 4) -> str:
    if frame.empty:
        return "_No rows._"
    header = "| " + " | ".join(columns) + " |"
    separator = "| " + " | ".join(["---"] * len(columns)) + " |"
    rows = [header, separator]
    for _, row in frame[columns].iterrows():
        cells = []
        for column in columns:
            value = row[column]
            if isinstance(value, (float, np.floating)):
                cells.append(_fmt(value, digits))
            elif isinstance(value, (int, np.integer)):
                cells.append(str(value))
            else:
                cells.append(str(value).replace("|", "\\|"))
        rows.append("| " + " | ".join(cells) + " |")
    return "\n".join(rows)


def _best_model(frame: pd.DataFrame, column: str) -> tuple[str, float]:
    values = pd.to_numeric(frame[column], errors="coerce")
    if values.notna().sum() == 0:
        return "", math.nan
    idx = values.idxmax()
    return str(frame.loc[idx, "model_name"]), float(values.loc[idx])


def _get_row(frame: pd.DataFrame, model_name: str) -> pd.Series:
    rows = frame.loc[frame["model_name"] == model_name]
    if rows.empty:
        raise KeyError(model_name)
    return rows.iloc[0]


def _get_per_class(per_class: pd.DataFrame, model_name: str, class_name: str) -> pd.Series:
    rows = per_class.loc[
        (per_class["model_name"] == model_name)
        & (per_class["class_name"] == class_name)
    ]
    if rows.empty:
        raise KeyError(f"{model_name}/{class_name}")
    return rows.iloc[0]


def _delta_frame(overall: pd.DataFrame) -> pd.DataFrame:
    baseline = _get_row(overall, "resnet18_meanbg")
    metric_pairs = {
        "delta_macro_auc": "macro_auc_mean",
        "delta_balanced_accuracy": "balanced_accuracy_mean",
        "delta_macro_f1": "macro_f1_mean",
        "delta_recall_severe": "recall_severe_mean",
        "delta_oof_macro_auc": "oof_macro_auc",
        "delta_oof_balanced_accuracy": "oof_balanced_accuracy",
        "delta_oof_macro_f1": "oof_macro_f1",
    }
    rows = []
    for _, row in overall.iterrows():
        out = {"model_name": row["model_name"]}
        for delta_column, metric_column in metric_pairs.items():
            out[delta_column] = _safe_float(row[metric_column]) - _safe_float(
                baseline[metric_column]
            )
        rows.append(out)
    return pd.DataFrame(rows)


def _experiment_status_frame(
    summary_lookup: dict[str, dict[str, Any]],
    queue_lookup: dict[str, dict[str, Any]],
) -> pd.DataFrame:
    rows = []
    for model_name, model_dir in MODEL_DIRS.items():
        total_params, trainable_params = _resolve_params(
            model_name, model_dir, summary_lookup, queue_lookup
        )
        queue_row = queue_lookup.get(model_name, {})
        rows.append(
            {
                "model_name": model_name,
                "status": "BASELINE_REFERENCE"
                if model_name == "resnet18_meanbg"
                else queue_row.get("status", "UNKNOWN"),
                "duration_minutes": queue_row.get("duration_minutes", ""),
                "total_params": total_params,
                "trainable_params": trainable_params,
                "output_dir": str(model_dir),
            }
        )
    return pd.DataFrame(rows)


def _build_key_information(
    overall: pd.DataFrame,
    per_class: pd.DataFrame,
    confusion: pd.DataFrame,
    auc_summary: pd.DataFrame,
) -> list[str]:
    best_macro_auc = _best_model(overall, "macro_auc_mean")
    best_ba = _best_model(overall, "balanced_accuracy_mean")
    best_f1 = _best_model(overall, "macro_f1_mean")
    best_severe = _best_model(overall, "recall_severe_mean")
    swin_mild = _get_per_class(per_class, "swin_t", "mild")
    swin_conf = _get_row(confusion, "swin_t")
    swin_auc = _get_row(auc_summary, "swin_t")
    resnet_auc = _get_row(auc_summary, "resnet18_meanbg")

    mild_total = _safe_float(swin_conf["mild_total"])
    severe_total = _safe_float(swin_conf["severe_total"])
    lines = [
        f"1. 最优 macro-AUC 模型：{best_macro_auc[0]}（{best_macro_auc[1]:.4f}）。",
        f"2. 最优 BA 模型：{best_ba[0]}（{best_ba[1]:.4f}）。",
        f"3. 最优 macro-F1 模型：{best_f1[0]}（{best_f1[1]:.4f}）。",
        f"4. 最优 severe recall 模型：{best_severe[0]}（{best_severe[1]:.4f}）。",
        f"5. Swin-Tiny mild recall：{_safe_float(swin_mild['recall']):.4f}。",
        f"6. Swin-Tiny mild F1：{_safe_float(swin_mild['f1']):.4f}。",
        (
            "7. Swin-Tiny mild -> normal 数量和比例："
            f"{_fmt_int(swin_conf['mild_to_normal'])}/"
            f"{_fmt_int(swin_conf['mild_total'])}"
            f"（{_rate(_safe_float(swin_conf['mild_to_normal']), mild_total):.4f}）。"
        ),
        (
            "8. Swin-Tiny mild -> severe 数量和比例："
            f"{_fmt_int(swin_conf['mild_to_severe'])}/"
            f"{_fmt_int(swin_conf['mild_total'])}"
            f"（{_rate(_safe_float(swin_conf['mild_to_severe']), mild_total):.4f}）。"
        ),
        (
            "9. Swin-Tiny severe -> mild 数量和比例："
            f"{_fmt_int(swin_conf['severe_to_mild'])}/"
            f"{_fmt_int(swin_conf['severe_total'])}"
            f"（{_rate(_safe_float(swin_conf['severe_to_mild']), severe_total):.4f}）。"
        ),
        (
            "10. Swin-Tiny severe -> normal 数量和比例："
            f"{_fmt_int(swin_conf['severe_to_normal'])}/"
            f"{_fmt_int(swin_conf['severe_total'])}"
            f"（{_rate(_safe_float(swin_conf['severe_to_normal']), severe_total):.4f}）。"
        ),
        (
            "11. Swin-Tiny severe-vs-rest AUC："
            f"{_safe_float(swin_auc['severe_vs_rest_auc']):.4f}。"
        ),
        (
            "12. ResNet18 meanbg severe-vs-rest AUC："
            f"{_safe_float(resnet_auc['severe_vs_rest_auc']):.4f}。"
        ),
        "13. 是否推荐 Swin-Tiny 进入二轮调参：是，优先做 OOF threshold scan、lr=5e-5、label smoothing=0.05。",
        "14. 是否推荐 EfficientNet-B0 作为轻量候选：是，参数量明显低于 ResNet18，BA 接近或略高，但 macro-F1 需要继续确认。",
        "15. 是否建议停止 MobileNetV3-Large：是，整体指标和 severe recall 都不占优。",
        "16. 下一步最推荐的 3 个实验：Swin-Tiny OOF threshold scan；Swin-Tiny lr=5e-5；Swin-Tiny label smoothing=0.05。",
    ]
    return lines


def _write_report(
    files_checked: pd.DataFrame,
    status: pd.DataFrame,
    overall: pd.DataFrame,
    comparison: pd.DataFrame,
    per_class: pd.DataFrame,
    confusion: pd.DataFrame,
    auc_summary: pd.DataFrame,
    fold_stability: pd.DataFrame,
    key_lines: list[str],
) -> None:
    missing = files_checked.loc[files_checked["exists"] != True]
    all_success = status.loc[status["model_name"] != "resnet18_meanbg", "status"].eq(
        "SUCCESS"
    ).all()

    swin_overall = _get_row(overall, "swin_t")
    resnet_overall = _get_row(overall, "resnet18_meanbg")
    efficient_overall = _get_row(overall, "efficientnet_b0")
    densenet_overall = _get_row(overall, "densenet121")
    convnext_overall = _get_row(overall, "convnext_tiny")
    mobile_overall = _get_row(overall, "mobilenet_v3_large")

    swin_mild = _get_per_class(per_class, "swin_t", "mild")
    swin_severe = _get_per_class(per_class, "swin_t", "severe")
    swin_conf = _get_row(confusion, "swin_t")
    convnext_conf = _get_row(confusion, "convnext_tiny")
    mobile_conf = _get_row(confusion, "mobilenet_v3_large")
    swin_auc = _get_row(auc_summary, "swin_t")
    resnet_auc = _get_row(auc_summary, "resnet18_meanbg")

    largest_volatility = _best_model(fold_stability, "stability_std_mean")
    smallest_idx = pd.to_numeric(
        fold_stability["stability_std_mean"], errors="coerce"
    ).idxmin()
    most_stable = (
        str(fold_stability.loc[smallest_idx, "model_name"]),
        float(fold_stability.loc[smallest_idx, "stability_std_mean"]),
    )
    convnext_stability = _get_row(fold_stability, "convnext_tiny")
    efficient_stability = _get_row(fold_stability, "efficientnet_b0")
    swin_stability = _get_row(fold_stability, "swin_t")

    full_win_mask = (
        (overall["model_name"] != "resnet18_meanbg")
        & (pd.to_numeric(overall["macro_auc_mean"], errors="coerce") > _safe_float(resnet_overall["macro_auc_mean"]))
        & (pd.to_numeric(overall["balanced_accuracy_mean"], errors="coerce") > _safe_float(resnet_overall["balanced_accuracy_mean"]))
        & (pd.to_numeric(overall["macro_f1_mean"], errors="coerce") > _safe_float(resnet_overall["macro_f1_mean"]))
        & (pd.to_numeric(overall["recall_severe_mean"], errors="coerce") > _safe_float(resnet_overall["recall_severe_mean"]))
    )
    full_win_models = overall.loc[full_win_mask, "model_name"].tolist()

    lines: list[str] = [
        "# Model Exploration Diagnostic Report",
        "",
        "## 1. Files Checked",
        "",
        f"- Checked files: {len(files_checked)}.",
        f"- Missing files: {len(missing)}.",
    ]
    if missing.empty:
        lines.append("- No required files were missing.")
    else:
        lines.append("- Missing file paths:")
        lines.extend([f"  - {path}" for path in missing["absolute_path"].tolist()])

    lines.extend(
        [
            "",
            _markdown_table(
                files_checked.assign(exists=files_checked["exists"].astype(str)),
                ["scope", "model_name", "relative_path", "exists"],
                digits=4,
            ),
            "",
            "## 2. Experiment Status",
            "",
            f"- Non-baseline queue all SUCCESS: {'yes' if all_success else 'no'}.",
            "",
            _markdown_table(
                status.assign(
                    total_params=status["total_params"].map(_fmt_int),
                    trainable_params=status["trainable_params"].map(_fmt_int),
                ),
                [
                    "model_name",
                    "status",
                    "duration_minutes",
                    "total_params",
                    "trainable_params",
                ],
            ),
            "",
            "## 3. Overall Result Table",
            "",
            _markdown_table(
                overall.assign(total_params=overall["total_params"].map(_fmt_int)),
                [
                    "model_name",
                    "total_params",
                    "macro_auc_mean",
                    "balanced_accuracy_mean",
                    "macro_f1_mean",
                    "recall_severe_mean",
                    "oof_macro_auc",
                    "oof_balanced_accuracy",
                    "oof_macro_f1",
                    "oof_recall_severe",
                    "recommendation",
                ],
            ),
            "",
            "## 4. Comparison with ResNet18 MeanBG",
            "",
            _markdown_table(
                comparison,
                [
                    "model_name",
                    "delta_macro_auc",
                    "delta_balanced_accuracy",
                    "delta_macro_f1",
                    "delta_recall_severe",
                    "delta_oof_macro_auc",
                    "delta_oof_balanced_accuracy",
                    "delta_oof_macro_f1",
                ],
            ),
            "",
            "## 5. Per-class Performance",
            "",
            _markdown_table(
                per_class,
                ["model_name", "class_name", "support", "precision", "recall", "f1"],
            ),
            "",
            "## 6. Confusion Matrix Analysis",
            "",
            (
                "- Swin-Tiny 的 mild 类主要被误判为 "
                f"{'normal' if _safe_float(swin_conf['mild_to_normal']) >= _safe_float(swin_conf['mild_to_severe']) else 'severe'}："
                f"mild->normal={_fmt_int(swin_conf['mild_to_normal'])}，"
                f"mild->severe={_fmt_int(swin_conf['mild_to_severe'])}。"
            ),
            (
                "- ConvNeXt-Tiny 增加 severe 预测倾向："
                f"severe recall={_fmt(convnext_overall['recall_severe_mean'])}，"
                f"mild->severe={_fmt_int(convnext_conf['mild_to_severe'])}，"
                f"normal->severe={_fmt_int(convnext_conf['normal_to_severe'])}。"
            ),
            (
                "- MobileNetV3-Large 漏判 severe 较明显："
                f"severe recall={_fmt(mobile_overall['recall_severe_mean'])}，"
                f"severe->normal={_fmt_int(mobile_conf['severe_to_normal'])}，"
                f"severe->mild={_fmt_int(mobile_conf['severe_to_mild'])}。"
            ),
            "",
            _markdown_table(
                confusion,
                [
                    "model_name",
                    "normal_total",
                    "mild_total",
                    "severe_total",
                    "normal_to_normal",
                    "normal_to_mild",
                    "normal_to_severe",
                    "mild_to_normal",
                    "mild_to_mild",
                    "mild_to_severe",
                    "severe_to_normal",
                    "severe_to_mild",
                    "severe_to_severe",
                ],
            ),
            "",
            "## 7. Swin-Tiny Focused Analysis",
            "",
            (
                f"1. Swin-Tiny macro-AUC={_fmt(swin_overall['macro_auc_mean'])}，"
                f"是否最高：{'是' if _best_model(overall, 'macro_auc_mean')[0] == 'swin_t' else '否'}。"
            ),
            (
                f"2. Swin-Tiny balanced accuracy={_fmt(swin_overall['balanced_accuracy_mean'])}，"
                f"是否最高：{'是' if _best_model(overall, 'balanced_accuracy_mean')[0] == 'swin_t' else '否'}。"
            ),
            (
                "3. Swin-Tiny macro-F1 不高的主因是 mild 类 hard decision 较弱："
                f"mild recall={_fmt(swin_mild['recall'])}，mild F1={_fmt(swin_mild['f1'])}；"
                f"相比之下 severe recall={_fmt(swin_severe['recall'])}。"
            ),
            (
                f"4. Swin-Tiny mild recall={_fmt(swin_mild['recall'])}，"
                f"mild F1={_fmt(swin_mild['f1'])}。"
            ),
            (
                "5. Swin-Tiny mild 类主要被误判为 "
                f"{'normal' if _safe_float(swin_conf['mild_to_normal']) >= _safe_float(swin_conf['mild_to_severe']) else 'severe'}。"
            ),
            (
                "6. 目前看 Swin-Tiny 的确牺牲了部分 mild hard decision，换取更好的整体排序能力和较高 severe/normal 区分能力。"
            ),
            (
                f"7. Swin-Tiny severe recall={_fmt(swin_overall['recall_severe_mean'])}，"
                f"ResNet18={_fmt(resnet_overall['recall_severe_mean'])}，"
                f"是否优于 ResNet18：{'是' if _safe_float(swin_overall['recall_severe_mean']) > _safe_float(resnet_overall['recall_severe_mean']) else '否'}。"
            ),
            (
                f"8. Swin-Tiny severe-vs-rest AUC={_fmt(swin_auc['severe_vs_rest_auc'])}，"
                f"ResNet18={_fmt(resnet_auc['severe_vs_rest_auc'])}，"
                f"是否优于 ResNet18：{'是' if _safe_float(swin_auc['severe_vs_rest_auc']) > _safe_float(resnet_auc['severe_vs_rest_auc']) else '否'}。"
            ),
            "9. Swin-Tiny 值得进入第二轮调参，但重点不应只是继续换 backbone，而是围绕阈值和 mild 边界优化。",
            "10. 优先参数：OOF threshold scan、lr=5e-5、label smoothing=0.05；可随后评估 weight_decay/dropout。",
            "",
            "## 8. Candidate Models for Next Stage",
            "",
            "1. 主线候选：Swin-Tiny。理由是 macro-AUC、BA、OOF AUC、OOF BA 均领先，但需要修复 mild hard decision。",
            "2. 轻量候选：EfficientNet-B0。理由是参数量明显更小，BA 接近或略高于 ResNet18，适合作为轻量路线。",
            "3. CNN 对照：ResNet18 meanbg 继续作为强基线；DenseNet121 可作为 AUC 排序提升但 hard classification 未改善的对照。",
            "4. 不推荐继续推进：MobileNetV3-Large；ConvNeXt-Tiny 不作为主模型，只保留 severe-sensitive 分析价值。",
            "",
            "## 9. Suggested Next Experiments",
            "",
            "- Swin-Tiny OOF threshold scan，优先优化 macro-F1、balanced accuracy、severe recall 与 mild recall 的折中。",
            "- Swin-Tiny lr=5e-5，验证 transformer backbone 在较小学习率下是否减少 mild 类误判。",
            "- Swin-Tiny label smoothing 0.05，检查 hard decision 和概率校准是否改善。",
            "- EfficientNet-B0 label smoothing 0.05，作为轻量候选的低成本二轮实验。",
            "- 如果 backbone 改进有限，转向 ordinal classification / two-stage classification / ROI-global fusion。",
            "",
            "## 10. Final Conclusion",
            "",
            (
                "1. 是否有模型全面超过 ResNet18 meanbg："
                f"{'有，' + ', '.join(full_win_models) if full_win_models else '没有。Swin-Tiny 在 AUC/BA/severe recall 上更好，但 macro-F1 低于 ResNet18。'}"
            ),
            "2. Swin-Tiny 值得继续，但应定位为二轮阈值扫描和轻量调参候选，而不是直接当作最终模型。",
            "3. 当前瓶颈仍主要在 hard decision 和 mild 类边界；AUC 提升说明概率排序有改进，但 argmax 分类没有同步全面改善。",
            "4. 下一步优先做 Swin-Tiny OOF threshold scan，其次做 lr=5e-5 和 label smoothing=0.05。",
            "",
            "## 11. 5-fold Stability Analysis",
            "",
            (
                "按 macro-AUC、BA、macro-F1、severe recall 四项 std 的均值衡量，"
                f"fold 间波动最大的是 {largest_volatility[0]}（{largest_volatility[1]:.4f}），"
                f"最稳定的是 {most_stable[0]}（{most_stable[1]:.4f}）。"
            ),
            (
                f"Swin-Tiny macro-AUC fold values=[{swin_stability['macro_auc_values']}]，"
                f"std={_fmt(swin_stability['macro_auc_std'])}；提升不是单个指标表里的孤立最高值，但仍需通过阈值扫描验证 hard decision。"
            ),
            (
                f"ConvNeXt-Tiny severe recall fold values=[{convnext_stability['recall_severe_values']}]，"
                f"std={_fmt(convnext_stability['recall_severe_std'])}，说明 severe 敏感性存在但稳定性需要谨慎解释。"
            ),
            (
                f"EfficientNet-B0 stability score={_fmt(efficient_stability['stability_std_mean'])}，"
                "整体较稳但上限有限，适合轻量候选而非主线性能突破模型。"
            ),
            "",
            "## Key Information for Further Analysis",
            "",
            *key_lines,
            "",
        ]
    )

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)

    files_checked = _check_files()
    missing = files_checked.loc[files_checked["exists"] != True]
    if not missing.empty:
        missing_paths = "\n".join(missing["absolute_path"].tolist())
        raise FileNotFoundError(f"Required files are missing:\n{missing_paths}")

    summary_lookup = _load_summary_lookup()
    queue_lookup = _load_queue_lookup()
    status = _experiment_status_frame(summary_lookup, queue_lookup)

    argmax_rows: list[dict[str, Any]] = []
    per_class_frames: list[pd.DataFrame] = []
    auc_rows: list[dict[str, Any]] = []
    confusion_rows: list[dict[str, Any]] = []
    confusion_count_frames: list[pd.DataFrame] = []
    confusion_norm_frames: list[pd.DataFrame] = []
    fold_rows: list[dict[str, Any]] = []

    for model_name, model_dir in MODEL_DIRS.items():
        summary_dir = model_dir / "summary"
        oof = _prepare_oof_predictions(summary_dir / "oof_predictions.csv")
        fold_metrics = _read_csv(summary_dir / "fold_metrics_all.csv")

        argmax_rows.append(_compute_argmax_metrics(model_name, oof))
        per_class_frames.append(_compute_per_class_metrics(model_name, oof))
        auc_rows.append(_compute_oof_auc(model_name, oof))
        confusion_row, count_long, norm_long = _compute_confusion(model_name, oof)
        confusion_rows.append(confusion_row)
        confusion_count_frames.append(count_long)
        confusion_norm_frames.append(norm_long)
        fold_rows.append(_compute_fold_stability(model_name, fold_metrics))
        _write_figures(model_name, oof)

    argmax_metrics = pd.DataFrame(argmax_rows)
    per_class_metrics = pd.concat(per_class_frames, ignore_index=True)
    auc_summary = pd.DataFrame(auc_rows)
    confusion_summary = pd.DataFrame(confusion_rows)
    confusion_count_long = pd.concat(confusion_count_frames, ignore_index=True)
    confusion_norm_long = pd.concat(confusion_norm_frames, ignore_index=True)
    fold_stability = pd.DataFrame(fold_rows)

    params_lookup = status.set_index("model_name")
    overall_rows = []
    for model_name in MODEL_DIRS:
        fold_row = _get_row(fold_stability, model_name)
        arg_row = _get_row(argmax_metrics, model_name)
        auc_row = _get_row(auc_summary, model_name)
        status_row = params_lookup.loc[model_name]
        overall_rows.append(
            {
                "model_name": model_name,
                "total_params": _safe_float(status_row["total_params"]),
                "macro_auc_mean": _safe_float(fold_row["macro_auc_mean"]),
                "balanced_accuracy_mean": _safe_float(
                    fold_row["balanced_accuracy_mean"]
                ),
                "macro_f1_mean": _safe_float(fold_row["macro_f1_mean"]),
                "recall_severe_mean": _safe_float(fold_row["recall_severe_mean"]),
                "oof_macro_auc": _safe_float(auc_row["macro_auc_ovr"]),
                "oof_balanced_accuracy": _safe_float(arg_row["balanced_accuracy"]),
                "oof_macro_f1": _safe_float(arg_row["macro_f1"]),
                "oof_recall_severe": _safe_float(arg_row["recall_severe"]),
                "recommendation": _recommendation(model_name),
            }
        )
    overall = pd.DataFrame(overall_rows)
    comparison = _delta_frame(overall)
    key_lines = _build_key_information(
        overall, per_class_metrics, confusion_summary, auc_summary
    )

    ARGMAX_CSV_PATH.parent.mkdir(parents=True, exist_ok=True)
    argmax_metrics.to_csv(ARGMAX_CSV_PATH, index=False, encoding="utf-8-sig")
    per_class_metrics.to_csv(PER_CLASS_CSV_PATH, index=False, encoding="utf-8-sig")
    auc_summary.to_csv(AUC_CSV_PATH, index=False, encoding="utf-8-sig")
    confusion_summary.to_csv(CONFUSION_CSV_PATH, index=False, encoding="utf-8-sig")

    key_info_frame = pd.DataFrame({"key_information": key_lines})
    write_xlsx(
        TABLES_XLSX_PATH,
        {
            "files_checked": files_checked,
            "experiment_status": status,
            "overall_results": overall,
            "comparison_vs_resnet18": comparison,
            "argmax_metrics": argmax_metrics,
            "per_class_metrics": per_class_metrics,
            "oof_auc_summary": auc_summary,
            "confusion_summary": confusion_summary,
            "confusion_count_long": confusion_count_long,
            "confusion_rownorm_long": confusion_norm_long,
            "fold_stability": fold_stability,
            "key_information": key_info_frame,
        },
    )

    _write_report(
        files_checked,
        status,
        overall,
        comparison,
        per_class_metrics,
        confusion_summary,
        auc_summary,
        fold_stability,
        key_lines,
    )

    print(f"REPORT={REPORT_PATH}")
    print(f"TABLES_XLSX={TABLES_XLSX_PATH}")
    print(f"ARGMAX_CSV={ARGMAX_CSV_PATH}")
    print(f"PER_CLASS_CSV={PER_CLASS_CSV_PATH}")
    print(f"AUC_CSV={AUC_CSV_PATH}")
    print(f"CONFUSION_CSV={CONFUSION_CSV_PATH}")
    print(f"FIGURE_DIR={FIGURE_DIR}")


if __name__ == "__main__":
    main()
