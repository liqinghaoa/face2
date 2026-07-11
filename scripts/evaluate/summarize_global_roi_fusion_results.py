"""Summarize Global + selected ROI fusion experiments and baselines."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.simple_xlsx_writer import write_xlsx  # noqa: E402


OUTPUT_ROOT = PROJECT_ROOT / "experiments" / "global_roi_fusion_500Data"
SUMMARY_CSV = OUTPUT_ROOT / "global_roi_fusion_summary.csv"
SUMMARY_XLSX = OUTPUT_ROOT / "global_roi_fusion_summary.xlsx"
SUMMARY_MD = OUTPUT_ROOT / "global_roi_fusion_summary.md"

REQUIRED_SUMMARY_FILES = [
    "fold_metrics_all.csv",
    "mean_metrics.csv",
    "oof_metrics.csv",
    "oof_predictions.csv",
    "summary_report.md",
]
CLASS_NAMES = ["normal", "mild", "severe"]
NEW_EXPERIMENT_KEYS = ["global_eye", "global_cheek", "global_eye_cheek"]


EXPERIMENT_SPECS = [
    {
        "experiment_key": "global_resnet18_meanbg",
        "experiment_name": "PreprocAblation_ResNet18_NYHA3Class_hybrid_imagenet_meanbg",
        "source_type": "global_baseline",
        "path": PROJECT_ROOT
        / "experiments"
        / "preprocess_ablation_500Data"
        / "PreprocAblation_ResNet18_NYHA3Class_hybrid_imagenet_meanbg",
    },
    {
        "experiment_key": "roi_single_eye_resnet18",
        "experiment_name": "ROI_Eye_ImageNetResNet18",
        "source_type": "roi_single",
        "scan_root": PROJECT_ROOT / "experiments" / "ROI_500",
        "include": ["ROI_Eye", "ResNet18"],
    },
    {
        "experiment_key": "roi_single_cheek_resnet34",
        "experiment_name": "ROI_Cheek_ImageNetResNet34",
        "source_type": "roi_single",
        "scan_root": PROJECT_ROOT / "experiments" / "ROI_500",
        "include": ["ROI_Cheek", "ResNet34"],
    },
    {
        "experiment_key": "roi_fusion_multiroi5_resnet18",
        "experiment_name": "MultiROI5_ImageNetResNet18",
        "source_type": "roi_fusion",
        "scan_root": PROJECT_ROOT / "experiments" / "ROI_Fusion_500",
        "include": ["MultiROI5", "ResNet18"],
    },
    {
        "experiment_key": "global_eye",
        "experiment_name": "GlobalROIFusion_GlobalEye_ResNet18_WeightedCE_5Fold",
        "source_type": "global_roi_fusion",
        "path": OUTPUT_ROOT / "GlobalROIFusion_GlobalEye_ResNet18_WeightedCE_5Fold",
        "prefix": "GlobalROIFusion_GlobalEye_ResNet18_WeightedCE_5Fold",
    },
    {
        "experiment_key": "global_cheek",
        "experiment_name": "GlobalROIFusion_GlobalCheek_ResNet18_WeightedCE_5Fold",
        "source_type": "global_roi_fusion",
        "path": OUTPUT_ROOT / "GlobalROIFusion_GlobalCheek_ResNet18_WeightedCE_5Fold",
        "prefix": "GlobalROIFusion_GlobalCheek_ResNet18_WeightedCE_5Fold",
    },
    {
        "experiment_key": "global_eye_cheek",
        "experiment_name": "GlobalROIFusion_GlobalEyeCheek_ResNet18_WeightedCE_5Fold",
        "source_type": "global_roi_fusion",
        "path": OUTPUT_ROOT / "GlobalROIFusion_GlobalEyeCheek_ResNet18_WeightedCE_5Fold",
        "prefix": "GlobalROIFusion_GlobalEyeCheek_ResNet18_WeightedCE_5Fold",
    },
]


def _summary_complete(path: Path) -> bool:
    return all((path / "summary" / name).is_file() for name in REQUIRED_SUMMARY_FILES)


def _find_latest_by_prefix(root: Path, prefix: str) -> Path | None:
    if not root.is_dir():
        return None
    candidates = [
        path
        for path in root.iterdir()
        if path.is_dir() and path.name.startswith(prefix) and _summary_complete(path)
    ]
    if not candidates:
        return None
    return max(candidates, key=lambda p: p.stat().st_mtime)


def _find_by_tokens(root: Path, include: list[str]) -> Path | None:
    if not root.is_dir():
        return None
    tokens = [token.lower() for token in include]
    candidates = []
    for path in root.iterdir():
        if not path.is_dir() or not _summary_complete(path):
            continue
        name = path.name.lower()
        if all(token in name for token in tokens):
            candidates.append(path)
    if not candidates:
        return None
    return max(candidates, key=lambda p: p.stat().st_mtime)


def _resolve_experiment_dir(spec: dict[str, Any]) -> Path | None:
    path = spec.get("path")
    if path is not None:
        path = Path(path)
        if _summary_complete(path):
            return path
    if spec.get("prefix"):
        found = _find_latest_by_prefix(OUTPUT_ROOT, spec["prefix"])
        if found is not None:
            return found
    if spec.get("scan_root") and spec.get("include"):
        return _find_by_tokens(Path(spec["scan_root"]), list(spec["include"]))
    return None


def _metric_lookup(mean_frame: pd.DataFrame, metric: str, field: str) -> float:
    if mean_frame.empty or "metric" not in mean_frame.columns:
        return float("nan")
    row = mean_frame[mean_frame["metric"] == metric]
    if row.empty or field not in row.columns:
        return float("nan")
    return float(pd.to_numeric(row.iloc[0][field], errors="coerce"))


def _fold_metric_lookup(fold_frame: pd.DataFrame, metric: str, field: str) -> float:
    if fold_frame.empty or metric not in fold_frame.columns:
        return float("nan")
    values = pd.to_numeric(fold_frame[metric], errors="coerce").dropna()
    if values.empty:
        return float("nan")
    if field == "mean":
        return float(values.mean())
    if field == "std":
        return float(values.std(ddof=1))
    return float("nan")


def _summary_metric(
    mean_frame: pd.DataFrame,
    fold_frame: pd.DataFrame,
    metric: str,
    field: str,
) -> float:
    value = _metric_lookup(mean_frame, metric, field)
    if not pd.isna(value):
        return value
    return _fold_metric_lookup(fold_frame, metric, field)


def _read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, encoding="utf-8-sig")


def _oof_value(oof_metrics: pd.DataFrame, metric: str) -> float:
    if oof_metrics.empty or metric not in oof_metrics.columns:
        return float("nan")
    return float(pd.to_numeric(oof_metrics.iloc[0][metric], errors="coerce"))


def _experiment_row(spec: dict[str, Any], directory: Path) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    summary = directory / "summary"
    fold = _read_csv(summary / "fold_metrics_all.csv")
    mean = _read_csv(summary / "mean_metrics.csv")
    oof_metrics = _read_csv(summary / "oof_metrics.csv")
    oof_predictions = _read_csv(summary / "oof_predictions.csv")
    row = {
        "experiment_key": spec["experiment_key"],
        "experiment_name": directory.name,
        "source_type": spec["source_type"],
        "output_dir": str(directory),
        "macro_auc_mean": _summary_metric(mean, fold, "macro_auc", "mean"),
        "macro_auc_std": _summary_metric(mean, fold, "macro_auc", "std"),
        "balanced_accuracy_mean": _summary_metric(mean, fold, "balanced_accuracy", "mean"),
        "balanced_accuracy_std": _summary_metric(mean, fold, "balanced_accuracy", "std"),
        "macro_f1_mean": _summary_metric(mean, fold, "macro_f1", "mean"),
        "macro_f1_std": _summary_metric(mean, fold, "macro_f1", "std"),
        "macro_recall_mean": _summary_metric(mean, fold, "macro_recall", "mean"),
        "macro_recall_std": _summary_metric(mean, fold, "macro_recall", "std"),
        "accuracy_mean": _summary_metric(mean, fold, "accuracy", "mean"),
        "accuracy_std": _summary_metric(mean, fold, "accuracy", "std"),
        "recall_normal_mean": _summary_metric(mean, fold, "recall_normal", "mean"),
        "recall_mild_mean": _summary_metric(mean, fold, "recall_mild", "mean"),
        "recall_severe_mean": _summary_metric(mean, fold, "recall_severe", "mean"),
        "f1_normal_mean": _summary_metric(mean, fold, "f1_normal", "mean"),
        "f1_mild_mean": _summary_metric(mean, fold, "f1_mild", "mean"),
        "f1_severe_mean": _summary_metric(mean, fold, "f1_severe", "mean"),
        "severe_vs_rest_auc_mean": _summary_metric(mean, fold, "severe_vs_rest_auc", "mean"),
        "normal_vs_abnormal_auc_mean": _summary_metric(mean, fold, "normal_vs_abnormal_auc", "mean"),
        "oof_macro_auc": _oof_value(oof_metrics, "macro_auc"),
        "oof_balanced_accuracy": _oof_value(oof_metrics, "balanced_accuracy"),
        "oof_macro_f1": _oof_value(oof_metrics, "macro_f1"),
        "oof_accuracy": _oof_value(oof_metrics, "accuracy"),
        "oof_recall_normal": _oof_value(oof_metrics, "recall_normal"),
        "oof_recall_mild": _oof_value(oof_metrics, "recall_mild"),
        "oof_recall_severe": _oof_value(oof_metrics, "recall_severe"),
        "oof_f1_normal": _oof_value(oof_metrics, "f1_normal"),
        "oof_f1_mild": _oof_value(oof_metrics, "f1_mild"),
        "oof_f1_severe": _oof_value(oof_metrics, "f1_severe"),
        "oof_severe_vs_rest_auc": _oof_value(oof_metrics, "severe_vs_rest_auc"),
        "oof_normal_vs_abnormal_auc": _oof_value(oof_metrics, "normal_vs_abnormal_auc"),
    }
    fold = fold.copy()
    fold.insert(0, "experiment_key", spec["experiment_key"])
    oof_metrics = oof_metrics.copy()
    oof_metrics.insert(0, "experiment_key", spec["experiment_key"])
    return row, fold, oof_metrics, oof_predictions


def _confusion_row(summary_row: dict[str, Any], oof_predictions: pd.DataFrame) -> dict[str, Any]:
    true_col = "label_3class" if "label_3class" in oof_predictions.columns else "y_true"
    pred_col = "pred_class" if "pred_class" in oof_predictions.columns else "y_pred"
    true = pd.to_numeric(oof_predictions[true_col], errors="coerce").astype(int)
    pred = pd.to_numeric(oof_predictions[pred_col], errors="coerce").astype(int)
    cm = confusion_matrix(true, pred, labels=[0, 1, 2])
    totals = cm.sum(axis=1)
    row = {
        "experiment_key": summary_row["experiment_key"],
        "experiment_name": summary_row["experiment_name"],
    }
    for i, true_name in enumerate(CLASS_NAMES):
        for j, pred_name in enumerate(CLASS_NAMES):
            row[f"{true_name}_to_{pred_name}"] = int(cm[i, j])
    row["mild_to_normal_rate"] = cm[1, 0] / totals[1] if totals[1] else np.nan
    row["mild_to_severe_rate"] = cm[1, 2] / totals[1] if totals[1] else np.nan
    row["severe_to_normal_rate"] = cm[2, 0] / totals[2] if totals[2] else np.nan
    row["severe_to_mild_rate"] = cm[2, 1] / totals[2] if totals[2] else np.nan
    return row


def _delta_vs_global(summary: pd.DataFrame) -> pd.DataFrame:
    base = summary[summary["experiment_key"] == "global_resnet18_meanbg"]
    if base.empty:
        return pd.DataFrame()
    base_row = base.iloc[0]
    rows = []
    for _, row in summary[summary["experiment_key"].isin(NEW_EXPERIMENT_KEYS)].iterrows():
        rows.append(
            {
                "experiment_key": row["experiment_key"],
                "experiment_name": row["experiment_name"],
                "delta_macro_auc_mean": row["macro_auc_mean"] - base_row["macro_auc_mean"],
                "delta_balanced_accuracy_mean": row["balanced_accuracy_mean"] - base_row["balanced_accuracy_mean"],
                "delta_macro_f1_mean": row["macro_f1_mean"] - base_row["macro_f1_mean"],
                "delta_recall_mild_mean": row["recall_mild_mean"] - base_row["recall_mild_mean"],
                "delta_f1_mild_mean": row["f1_mild_mean"] - base_row["f1_mild_mean"],
                "delta_recall_severe_mean": row["recall_severe_mean"] - base_row["recall_severe_mean"],
                "delta_f1_severe_mean": row["f1_severe_mean"] - base_row["f1_severe_mean"],
                "delta_oof_balanced_accuracy": row["oof_balanced_accuracy"] - base_row["oof_balanced_accuracy"],
                "delta_oof_macro_f1": row["oof_macro_f1"] - base_row["oof_macro_f1"],
                "delta_oof_recall_mild": row["oof_recall_mild"] - base_row["oof_recall_mild"],
                "delta_oof_recall_severe": row["oof_recall_severe"] - base_row["oof_recall_severe"],
            }
        )
    return pd.DataFrame(rows)


def _recommendation(summary: pd.DataFrame) -> pd.DataFrame:
    base = summary[summary["experiment_key"] == "global_resnet18_meanbg"]
    best_single = summary[
        summary["experiment_key"].isin(["roi_single_eye_resnet18", "roi_single_cheek_resnet34"])
    ]
    old_fusion = summary[summary["experiment_key"] == "roi_fusion_multiroi5_resnet18"]
    if base.empty:
        return pd.DataFrame()
    base_row = base.iloc[0]
    best_single_macro_f1 = best_single["macro_f1_mean"].max() if not best_single.empty else np.nan
    old_fusion_macro_f1 = old_fusion.iloc[0]["macro_f1_mean"] if not old_fusion.empty else np.nan
    rows = []
    for _, row in summary[summary["experiment_key"].isin(NEW_EXPERIMENT_KEYS)].iterrows():
        improves_mild = row["recall_mild_mean"] >= base_row["recall_mild_mean"]
        improves_severe = row["recall_severe_mean"] >= base_row["recall_severe_mean"]
        candidate_main = (
            row["macro_f1_mean"] > base_row["macro_f1_mean"]
            and row["balanced_accuracy_mean"] >= base_row["balanced_accuracy_mean"]
            and improves_mild
            and improves_severe
        )
        candidate_secondary = (
            row["macro_f1_mean"] >= base_row["macro_f1_mean"] - 0.02
            and row["balanced_accuracy_mean"] > base_row["balanced_accuracy_mean"]
            and (
                (improves_mild and row["recall_severe_mean"] >= base_row["recall_severe_mean"] - 0.03)
                or (improves_severe and row["recall_mild_mean"] >= base_row["recall_mild_mean"] - 0.03)
            )
        )
        fusion_not_helpful = (
            row["macro_f1_mean"] < base_row["macro_f1_mean"]
            and (np.isnan(best_single_macro_f1) or row["macro_f1_mean"] < best_single_macro_f1)
            and (np.isnan(old_fusion_macro_f1) or row["macro_f1_mean"] < old_fusion_macro_f1)
        )
        overfit_or_unbalanced = abs(row["recall_mild_mean"] - row["recall_severe_mean"]) > 0.20
        if candidate_main:
            next_step = "candidate_main_model; repeat with statistical testing and independent validation"
        elif candidate_secondary:
            next_step = "candidate_secondary_model; inspect class trade-off and consider threshold/ordinal analysis"
        elif fusion_not_helpful:
            next_step = "fusion_not_helpful; do not expand before error analysis"
        else:
            next_step = "needs diagnostic analysis before expansion"
        rows.append(
            {
                "experiment_key": row["experiment_key"],
                "candidate_main_model": bool(candidate_main),
                "candidate_secondary_model": bool(candidate_secondary),
                "improves_mild_only": bool(improves_mild and not improves_severe),
                "improves_severe_only": bool(improves_severe and not improves_mild),
                "fusion_not_helpful": bool(fusion_not_helpful),
                "overfit_or_unbalanced": bool(overfit_or_unbalanced),
                "needs_threshold_scan": bool(candidate_secondary or overfit_or_unbalanced),
                "next_step": next_step,
            }
        )
    return pd.DataFrame(rows)


def _fmt(value: Any) -> str:
    try:
        if pd.isna(value):
            return ""
        return f"{float(value):.4f}"
    except Exception:
        return str(value)


def _md_table(frame: pd.DataFrame, columns: list[str]) -> str:
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join(["---"] * len(columns)) + " |",
    ]
    for _, row in frame.iterrows():
        lines.append("| " + " | ".join(_fmt(row.get(col, "")) for col in columns) + " |")
    return "\n".join(lines)


def _write_markdown(summary: pd.DataFrame, delta: pd.DataFrame, recommendation: pd.DataFrame, missing: list[str]) -> None:
    cols = [
        "experiment_key",
        "macro_auc_mean",
        "balanced_accuracy_mean",
        "macro_f1_mean",
        "recall_mild_mean",
        "recall_severe_mean",
        "oof_macro_f1",
    ]
    lines = [
        "# Global ROI Fusion Summary",
        "",
        "## 1. 实验目的",
        "",
        "验证 global face 与 selected ROI（eye、cheek）在特征级融合后，是否比 global-only、ROI single 和旧 ROI-only fusion 更好。",
        "",
        "## 2. 为什么选择 eye 和 cheek",
        "",
        "- eye ROI：此前 ROI single 中 ResNet18 对 mild recall 较有价值。",
        "- cheek ROI：此前 ROI single 中 cheek/ResNet34 接近 global baseline，并可能补充肤色与 severe 相关信息。",
        "",
        "## 3. 核心结果",
        "",
        _md_table(summary, cols) if not summary.empty else "No readable summaries.",
        "",
    ]
    if missing:
        lines.extend(["## Missing experiments", "", *[f"- {item}" for item in missing], ""])
    if not delta.empty:
        lines.extend(
            [
                "## 4. 新实验相对 Global ResNet18 MeanBG 的差值",
                "",
                _md_table(
                    delta,
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
            ]
        )
    if not recommendation.empty:
        lines.extend(
            [
                "## 5. 自动推荐",
                "",
                _md_table(
                    recommendation,
                    [
                        "experiment_key",
                        "candidate_main_model",
                        "candidate_secondary_model",
                        "fusion_not_helpful",
                        "needs_threshold_scan",
                        "next_step",
                    ],
                ),
                "",
            ]
        )
    lines.extend(
        [
            "## 6. 需要进一步回答的问题",
            "",
            "1. Global + Eye 是否改善 mild？",
            "2. Global + Cheek 是否改善 severe / BA？",
            "3. Global + Eye + Cheek 是否优于 global-only？",
            "4. 是否优于最佳单 ROI 与旧 ROI-only fusion？",
            "5. 是否建议继续加入 lip/forehead、做 ResNet34 版本，或转向 ordinal/two-stage？",
            "",
            "所有结论仍基于 5-fold CV 和 OOF，不是独立测试集结论。",
        ]
    )
    SUMMARY_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> Path:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    fold_frames: list[pd.DataFrame] = []
    oof_metric_frames: list[pd.DataFrame] = []
    confusion_rows: list[dict[str, Any]] = []
    missing: list[str] = []

    for spec in EXPERIMENT_SPECS:
        directory = _resolve_experiment_dir(spec)
        if directory is None:
            missing.append(spec["experiment_key"])
            continue
        row, fold, oof_metrics, oof_predictions = _experiment_row(spec, directory)
        rows.append(row)
        fold_frames.append(fold)
        oof_metric_frames.append(oof_metrics)
        confusion_rows.append(_confusion_row(row, oof_predictions))

    summary = pd.DataFrame(rows)
    fold_all = pd.concat(fold_frames, ignore_index=True) if fold_frames else pd.DataFrame()
    oof_all = pd.concat(oof_metric_frames, ignore_index=True) if oof_metric_frames else pd.DataFrame()
    confusion = pd.DataFrame(confusion_rows)
    delta = _delta_vs_global(summary) if not summary.empty else pd.DataFrame()
    recommendation = _recommendation(summary) if not summary.empty else pd.DataFrame()

    summary.to_csv(SUMMARY_CSV, index=False, encoding="utf-8-sig")
    write_xlsx(
        SUMMARY_XLSX,
        {
            "experiment_summary": summary,
            "fold_metrics_all": fold_all,
            "oof_metrics_all": oof_all,
            "confusion_summary": confusion,
            # Excel worksheet names are limited to 31 characters.
            "delta_vs_global_r18_meanbg": delta,
            "recommendation": recommendation,
        },
    )
    _write_markdown(summary, delta, recommendation, missing)

    print(f"SUMMARY_CSV={SUMMARY_CSV}")
    print(f"SUMMARY_XLSX={SUMMARY_XLSX}")
    print(f"SUMMARY_MD={SUMMARY_MD}")
    if missing:
        print("MISSING=" + ",".join(missing))
    return SUMMARY_CSV


if __name__ == "__main__":
    main()
