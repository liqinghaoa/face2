"""Summarize Swin-Tiny second-stage threshold and training experiments."""

from __future__ import annotations

import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score, precision_recall_fscore_support


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.simple_xlsx import write_xlsx  # noqa: E402


OUTPUT_ROOT = PROJECT_ROOT / "experiments" / "swin_tiny_second_stage_500Data"
THRESHOLD_BEST_RULES = OUTPUT_ROOT / "threshold_scan" / "best_threshold_rules.csv"
SUMMARY_CSV = OUTPUT_ROOT / "swin_tiny_second_stage_summary.csv"
SUMMARY_XLSX = OUTPUT_ROOT / "swin_tiny_second_stage_summary.xlsx"
SUMMARY_MD = OUTPUT_ROOT / "swin_tiny_second_stage_summary.md"

CLASS_LABELS = [0, 1, 2]
CLASS_NAMES = ["normal", "mild", "severe"]

BASELINE_DIR = (
    PROJECT_ROOT
    / "experiments"
    / "preprocess_ablation_500Data"
    / "PreprocAblation_ResNet18_NYHA3Class_hybrid_imagenet_meanbg"
)
SWIN_ORIGINAL_DIR = (
    PROJECT_ROOT
    / "experiments"
    / "model_exploration_500Data"
    / "ModelExploration_SwinTiny_ImageNetMeanBG"
)
TRAINED_EXPERIMENTS = {
    "lr5e5": "SwinTiny_ImageNetMeanBG_LR5e5_WeightedCE_5Fold",
    "ls005": "SwinTiny_ImageNetMeanBG_WeightedSoftCE_LS005_5Fold",
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


def _metric_mean(frame: pd.DataFrame, column: str) -> float:
    if frame.empty or column not in frame.columns:
        return math.nan
    return float(pd.to_numeric(frame[column], errors="coerce").mean())


def _metric_std(frame: pd.DataFrame, column: str) -> float:
    if frame.empty or column not in frame.columns:
        return math.nan
    return float(pd.to_numeric(frame[column], errors="coerce").std(ddof=1))


def _oof_value(oof_metrics: pd.DataFrame, column: str) -> float:
    if oof_metrics.empty or column not in oof_metrics.columns:
        return math.nan
    return _safe_float(oof_metrics.iloc[0][column])


def _latest_dir_by_prefix(root: Path, prefix: str) -> Path | None:
    if not root.is_dir():
        return None
    candidates = [
        path
        for path in root.iterdir()
        if path.is_dir() and (path.name == prefix or path.name.startswith(f"{prefix}_"))
    ]
    if not candidates:
        return None
    return max(candidates, key=lambda path: path.stat().st_mtime)


def _oof_hard_metrics(oof_predictions: pd.DataFrame) -> dict[str, float]:
    if oof_predictions.empty:
        return {}
    y_true = pd.to_numeric(oof_predictions["label_3class"], errors="coerce").astype(int).to_numpy()
    if "pred_class" in oof_predictions.columns:
        y_pred = pd.to_numeric(oof_predictions["pred_class"], errors="coerce").astype(int).to_numpy()
    else:
        y_pred = np.asarray(CLASS_LABELS)[
            np.argmax(
                oof_predictions[["prob_normal", "prob_mild", "prob_severe"]].to_numpy(dtype=float),
                axis=1,
            )
        ]
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=CLASS_LABELS, zero_division=0
    )
    _, _, macro_f1, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=CLASS_LABELS, average="macro", zero_division=0
    )
    return {
        "oof_balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "oof_macro_f1": float(macro_f1),
        "oof_recall_normal": float(recall[0]),
        "oof_recall_mild": float(recall[1]),
        "oof_recall_severe": float(recall[2]),
        "oof_f1_normal": float(f1[0]),
        "oof_f1_mild": float(f1[1]),
        "oof_f1_severe": float(f1[2]),
    }


def _experiment_row(
    *,
    experiment_key: str,
    experiment_name: str,
    source_type: str,
    output_dir: Path,
) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame]:
    summary_dir = output_dir / "summary"
    fold = _read_csv(summary_dir / "fold_metrics_all.csv") if (summary_dir / "fold_metrics_all.csv").is_file() else pd.DataFrame()
    oof_metrics = _read_csv(summary_dir / "oof_metrics.csv") if (summary_dir / "oof_metrics.csv").is_file() else pd.DataFrame()
    oof_predictions = _read_csv(summary_dir / "oof_predictions.csv") if (summary_dir / "oof_predictions.csv").is_file() else pd.DataFrame()
    hard = _oof_hard_metrics(oof_predictions)
    row: dict[str, Any] = {
        "experiment_key": experiment_key,
        "experiment_name": experiment_name,
        "source_type": source_type,
        "output_dir": str(output_dir),
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
        "oof_macro_auc": _oof_value(oof_metrics, "macro_auc"),
        "oof_balanced_accuracy": hard.get(
            "oof_balanced_accuracy", _oof_value(oof_metrics, "balanced_accuracy")
        ),
        "oof_macro_f1": hard.get("oof_macro_f1", _oof_value(oof_metrics, "macro_f1")),
        "oof_accuracy": _oof_value(oof_metrics, "accuracy"),
        "oof_recall_normal": hard.get("oof_recall_normal", _oof_value(oof_metrics, "recall_normal")),
        "oof_recall_mild": hard.get("oof_recall_mild", _oof_value(oof_metrics, "recall_mild")),
        "oof_recall_severe": hard.get("oof_recall_severe", _oof_value(oof_metrics, "recall_severe")),
        "oof_f1_normal": hard.get("oof_f1_normal", _oof_value(oof_metrics, "f1_normal")),
        "oof_f1_mild": hard.get("oof_f1_mild", _oof_value(oof_metrics, "f1_mild")),
        "oof_f1_severe": hard.get("oof_f1_severe", _oof_value(oof_metrics, "f1_severe")),
        "oof_severe_vs_rest_auc": _oof_value(oof_metrics, "severe_vs_rest_auc"),
        "oof_normal_vs_abnormal_auc": _oof_value(oof_metrics, "normal_vs_abnormal_auc"),
    }
    if not fold.empty:
        fold = fold.copy()
        fold.insert(0, "experiment_key", experiment_key)
    if not oof_metrics.empty:
        oof_metrics = oof_metrics.copy()
        oof_metrics.insert(0, "experiment_key", experiment_key)
    return row, fold, oof_metrics


def _threshold_rows(original_swin: pd.Series) -> tuple[pd.DataFrame, pd.DataFrame]:
    if not THRESHOLD_BEST_RULES.is_file():
        return pd.DataFrame(), pd.DataFrame()
    best = _read_csv(THRESHOLD_BEST_RULES)
    rows = []
    for _, rule in best.iterrows():
        if str(rule.get("rule_name")) == "no_candidate":
            continue
        key = f"threshold_{rule['strategy_name']}"
        rows.append(
            {
                "experiment_key": key,
                "experiment_name": f"Swin-Tiny threshold {rule['strategy_name']}",
                "source_type": "threshold_scan",
                "output_dir": str(THRESHOLD_BEST_RULES.parent),
                "macro_auc_mean": _safe_float(original_swin["macro_auc_mean"]),
                "macro_auc_std": _safe_float(original_swin["macro_auc_std"]),
                "balanced_accuracy_mean": _safe_float(rule["balanced_accuracy"]),
                "balanced_accuracy_std": math.nan,
                "macro_f1_mean": _safe_float(rule["macro_f1"]),
                "macro_f1_std": math.nan,
                "macro_recall_mean": math.nan,
                "macro_recall_std": math.nan,
                "accuracy_mean": _safe_float(rule["accuracy"]),
                "accuracy_std": math.nan,
                "recall_normal_mean": _safe_float(rule["recall_normal"]),
                "recall_mild_mean": _safe_float(rule["recall_mild"]),
                "recall_severe_mean": _safe_float(rule["recall_severe"]),
                "f1_normal_mean": _safe_float(rule["f1_normal"]),
                "f1_mild_mean": _safe_float(rule["f1_mild"]),
                "f1_severe_mean": _safe_float(rule["f1_severe"]),
                "severe_vs_rest_auc_mean": _safe_float(original_swin["severe_vs_rest_auc_mean"]),
                "normal_vs_abnormal_auc_mean": _safe_float(original_swin["normal_vs_abnormal_auc_mean"]),
                "oof_macro_auc": _safe_float(original_swin["oof_macro_auc"]),
                "oof_balanced_accuracy": _safe_float(rule["balanced_accuracy"]),
                "oof_macro_f1": _safe_float(rule["macro_f1"]),
                "oof_accuracy": _safe_float(rule["accuracy"]),
                "oof_recall_normal": _safe_float(rule["recall_normal"]),
                "oof_recall_mild": _safe_float(rule["recall_mild"]),
                "oof_recall_severe": _safe_float(rule["recall_severe"]),
                "oof_f1_normal": _safe_float(rule["f1_normal"]),
                "oof_f1_mild": _safe_float(rule["f1_mild"]),
                "oof_f1_severe": _safe_float(rule["f1_severe"]),
                "oof_severe_vs_rest_auc": _safe_float(original_swin["oof_severe_vs_rest_auc"]),
                "oof_normal_vs_abnormal_auc": _safe_float(original_swin["oof_normal_vs_abnormal_auc"]),
            }
        )
    return pd.DataFrame(rows), best


def _delta_frame(summary: pd.DataFrame, baseline_key: str, sheet_name: str) -> pd.DataFrame:
    if baseline_key not in set(summary["experiment_key"]):
        return pd.DataFrame()
    base = summary.loc[summary["experiment_key"] == baseline_key].iloc[0]
    metrics = {
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
        "delta_oof_recall_severe": "oof_recall_severe",
    }
    rows = []
    for _, row in summary.iterrows():
        out = {
            "experiment_key": row["experiment_key"],
            "comparison": sheet_name,
        }
        for delta, metric in metrics.items():
            out[delta] = _safe_float(row[metric]) - _safe_float(base[metric])
        rows.append(out)
    return pd.DataFrame(rows)


def _recommendation(summary: pd.DataFrame) -> pd.DataFrame:
    original = summary.loc[summary["experiment_key"] == "swin_original"].iloc[0]
    resnet = summary.loc[summary["experiment_key"] == "resnet18_meanbg"].iloc[0]
    rows = []
    lr_effective = False
    ls_effective = False
    any_effective = False
    for _, row in summary.iterrows():
        key = row["experiment_key"]
        if key in {"resnet18_meanbg", "swin_original"}:
            rows.append(
                {
                    "experiment_key": key,
                    "effective_for_mild_recovery": False,
                    "candidate_main_model": False,
                    "recommendation": "reference",
                }
            )
            continue
        effective = (
            _safe_float(row["macro_f1_mean"]) > _safe_float(original["macro_f1_mean"])
            and _safe_float(row["recall_mild_mean"]) > _safe_float(original["recall_mild_mean"])
            and _safe_float(row["balanced_accuracy_mean"])
            >= _safe_float(original["balanced_accuracy_mean"]) - 0.02
            and _safe_float(row["recall_severe_mean"])
            >= _safe_float(original["recall_severe_mean"]) - 0.05
        )
        candidate = (
            _safe_float(row["macro_f1_mean"]) >= _safe_float(resnet["macro_f1_mean"]) - 0.02
            and _safe_float(row["balanced_accuracy_mean"]) > _safe_float(resnet["balanced_accuracy_mean"])
            and _safe_float(row["recall_severe_mean"]) >= _safe_float(resnet["recall_severe_mean"])
        )
        any_effective = any_effective or effective
        if key == "lr5e5":
            lr_effective = effective
        if key == "ls005":
            ls_effective = effective
        recommendation = []
        if str(key).startswith("threshold_"):
            recommendation.append("threshold_scan_effective" if effective else "threshold_scan_diagnostic_only")
        elif key == "lr5e5":
            recommendation.append("lr5e5_effective" if effective else "lr5e5_not_effective")
        elif key == "ls005":
            recommendation.append("ls005_effective" if effective else "ls005_not_effective")
        if candidate:
            recommendation.append("candidate_main_model")
        if not candidate:
            recommendation.append("still_not_better_than_resnet18")
        rows.append(
            {
                "experiment_key": key,
                "effective_for_mild_recovery": effective,
                "candidate_main_model": candidate,
                "recommendation": ";".join(recommendation),
            }
        )
    rows.append(
        {
            "experiment_key": "next_step",
            "effective_for_mild_recovery": lr_effective and ls_effective,
            "candidate_main_model": False,
            "recommendation": (
                "candidate_for_next_combined_lr5e5_ls005"
                if lr_effective and ls_effective
                else (
                    "move_to_ordinal_or_two_stage"
                    if not any_effective
                    else "review_best_single_method_before_combination"
                )
            ),
        }
    )
    return pd.DataFrame(rows)


def _markdown_table(frame: pd.DataFrame, columns: list[str]) -> str:
    if frame.empty:
        return "_No rows available._"
    rows = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join(["---"] * len(columns)) + " |",
    ]
    for _, row in frame[columns].iterrows():
        cells = []
        for column in columns:
            value = row[column]
            cells.append(_fmt(value) if isinstance(value, (float, np.floating)) else str(value))
        rows.append("| " + " | ".join(cells) + " |")
    return "\n".join(rows)


def _write_md(
    summary: pd.DataFrame,
    threshold_best: pd.DataFrame,
    delta_original: pd.DataFrame,
    delta_resnet: pd.DataFrame,
    recommendation: pd.DataFrame,
) -> None:
    swin = summary.loc[summary["experiment_key"] == "swin_original"].iloc[0]
    resnet = summary.loc[summary["experiment_key"] == "resnet18_meanbg"].iloc[0]
    lr = summary.loc[summary["experiment_key"] == "lr5e5"] if "lr5e5" in set(summary["experiment_key"]) else pd.DataFrame()
    ls = summary.loc[summary["experiment_key"] == "ls005"] if "ls005" in set(summary["experiment_key"]) else pd.DataFrame()
    next_step = recommendation.loc[recommendation["experiment_key"] == "next_step"].iloc[0]
    lines = [
        "# Swin-Tiny Second-Stage Summary",
        "",
        "## 1. 实验目的",
        "",
        "本轮实验评估 Swin-Tiny 的 OOF 阈值策略、lr=5e-5、label smoothing=0.05 是否能把较好的概率排序能力转化为更好的三分类 hard prediction。",
        "",
        "## 2. 为什么选择 Swin-Tiny",
        "",
        f"原始 Swin-Tiny macro-AUC={_fmt(swin['macro_auc_mean'])}，BA={_fmt(swin['balanced_accuracy_mean'])}，是多 backbone 探索中排序和 BA 最好的模型，但 macro-F1={_fmt(swin['macro_f1_mean'])} 低于 ResNet18 meanbg 的 {_fmt(resnet['macro_f1_mean'])}。",
        "",
        "## 3. Threshold scan 结论",
        "",
        _markdown_table(
            threshold_best,
            [
                "strategy_name",
                "rule_name",
                "threshold_normal",
                "threshold_mild",
                "threshold_severe",
                "balanced_accuracy",
                "macro_f1",
                "recall_mild",
                "recall_severe",
            ],
        ),
        "",
        "## 4. lr=5e-5 结论",
        "",
        (
            "lr=5e-5 结果暂不可用。"
            if lr.empty
            else f"lr=5e-5 macro-F1={_fmt(lr.iloc[0]['macro_f1_mean'])}，BA={_fmt(lr.iloc[0]['balanced_accuracy_mean'])}，mild recall={_fmt(lr.iloc[0]['recall_mild_mean'])}，severe recall={_fmt(lr.iloc[0]['recall_severe_mean'])}。"
        ),
        "",
        "## 5. LS0.05 结论",
        "",
        (
            "LS0.05 结果暂不可用。"
            if ls.empty
            else f"LS0.05 macro-F1={_fmt(ls.iloc[0]['macro_f1_mean'])}，BA={_fmt(ls.iloc[0]['balanced_accuracy_mean'])}，mild recall={_fmt(ls.iloc[0]['recall_mild_mean'])}，severe recall={_fmt(ls.iloc[0]['recall_severe_mean'])}。"
        ),
        "",
        "## 6. 与 original Swin 对比",
        "",
        _markdown_table(
            delta_original,
            [
                "experiment_key",
                "delta_macro_f1_mean",
                "delta_balanced_accuracy_mean",
                "delta_recall_mild_mean",
                "delta_recall_severe_mean",
                "delta_oof_macro_f1",
            ],
        ),
        "",
        "## 7. 与 ResNet18 meanbg 对比",
        "",
        _markdown_table(
            delta_resnet,
            [
                "experiment_key",
                "delta_macro_f1_mean",
                "delta_balanced_accuracy_mean",
                "delta_recall_mild_mean",
                "delta_recall_severe_mean",
                "delta_oof_macro_f1",
            ],
        ),
        "",
        "## 8. 是否推荐 Swin-Tiny 继续作为主线",
        "",
        "如果二轮结果能提升 mild recall/macro-F1 且 BA 与 severe recall 不明显下降，则 Swin-Tiny 可继续作为主线；否则应把 Swin-Tiny 保留为排序能力较强的候选，而不是唯一主线。",
        "",
        "## 9. 是否推荐组合 lr5e-5 + LS0.05",
        "",
        str(next_step["recommendation"]),
        "",
        "## 10. 是否应转向 ordinal/two-stage/ROI fusion",
        "",
        "如果阈值、lr5e-5、LS0.05 仍无法恢复 mild 类 hard decision，则建议转向 ordinal classification / two-stage classification / ROI-global fusion。",
        "",
    ]
    SUMMARY_MD.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    rows = []
    fold_frames = []
    oof_metric_frames = []

    for key, name, source_type, directory in [
        ("resnet18_meanbg", "ResNet18 meanbg baseline", "baseline", BASELINE_DIR),
        ("swin_original", "Swin-Tiny original", "model_exploration", SWIN_ORIGINAL_DIR),
    ]:
        row, fold, oof_metrics = _experiment_row(
            experiment_key=key,
            experiment_name=name,
            source_type=source_type,
            output_dir=directory,
        )
        rows.append(row)
        if not fold.empty:
            fold_frames.append(fold)
        if not oof_metrics.empty:
            oof_metric_frames.append(oof_metrics)

    base_summary = pd.DataFrame(rows)
    threshold_rows, threshold_best = _threshold_rows(
        base_summary.loc[base_summary["experiment_key"] == "swin_original"].iloc[0]
    )
    if not threshold_rows.empty:
        rows.extend(threshold_rows.to_dict("records"))

    for key, prefix in TRAINED_EXPERIMENTS.items():
        directory = _latest_dir_by_prefix(OUTPUT_ROOT, prefix)
        if directory is None or not (directory / "summary" / "fold_metrics_all.csv").is_file():
            continue
        row, fold, oof_metrics = _experiment_row(
            experiment_key=key,
            experiment_name=prefix,
            source_type="second_stage_training",
            output_dir=directory,
        )
        rows.append(row)
        if not fold.empty:
            fold_frames.append(fold)
        if not oof_metrics.empty:
            oof_metric_frames.append(oof_metrics)

    summary = pd.DataFrame(rows)
    delta_original = _delta_frame(summary, "swin_original", "delta_vs_swin_original")
    delta_resnet = _delta_frame(summary, "resnet18_meanbg", "delta_vs_resnet18_meanbg")
    recommendation = _recommendation(summary)
    fold_all = pd.concat(fold_frames, ignore_index=True) if fold_frames else pd.DataFrame()
    oof_all = pd.concat(oof_metric_frames, ignore_index=True) if oof_metric_frames else pd.DataFrame()

    summary.to_csv(SUMMARY_CSV, index=False, encoding="utf-8-sig")
    write_xlsx(
        SUMMARY_XLSX,
        {
            "experiment_summary": summary,
            "threshold_best_rules": threshold_best,
            "fold_metrics_all": fold_all,
            "oof_metrics_all": oof_all,
            "delta_vs_swin_original": delta_original,
            "delta_vs_resnet18_meanbg": delta_resnet,
            "recommendation": recommendation,
        },
    )
    _write_md(summary, threshold_best, delta_original, delta_resnet, recommendation)

    print(f"SUMMARY_CSV={SUMMARY_CSV}")
    print(f"SUMMARY_XLSX={SUMMARY_XLSX}")
    print(f"SUMMARY_MD={SUMMARY_MD}")


if __name__ == "__main__":
    main()
