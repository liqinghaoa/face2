"""Summarize model-exploration experiments against the ResNet18 mean-bg baseline."""

from __future__ import annotations

import math
import sys
from pathlib import Path
from typing import Any

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models.nyha_backbone_factory import (  # noqa: E402
    build_nyha_classification_model,
    count_parameters,
)
from utils.experiment_utils import load_yaml  # noqa: E402
from utils.simple_xlsx import write_xlsx  # noqa: E402


EXPERIMENT_ROOT = PROJECT_ROOT / "experiments" / "model_exploration_500Data"
BASELINE_DIR = (
    PROJECT_ROOT
    / "experiments"
    / "preprocess_ablation_500Data"
    / "PreprocAblation_ResNet18_NYHA3Class_hybrid_imagenet_meanbg"
)
MANIFEST_PATH = (
    PROJECT_ROOT
    / "config"
    / "train"
    / "model_exploration_imagenet_meanbg"
    / "model_exploration_config_manifest.csv"
)
QUEUE_PATH = EXPERIMENT_ROOT / "model_exploration_job_queue.csv"

SUMMARY_CSV = EXPERIMENT_ROOT / "model_exploration_summary.csv"
SUMMARY_XLSX = EXPERIMENT_ROOT / "model_exploration_summary.xlsx"
SUMMARY_MD = EXPERIMENT_ROOT / "model_exploration_summary.md"

BASELINE_KEY = "resnet18_meanbg"
BASELINE_FALLBACK = {
    "macro_auc_mean": 0.7094,
    "balanced_accuracy_mean": 0.5353,
    "macro_f1_mean": 0.5111,
    "recall_severe_mean": 0.4333,
}

EXPERIMENT_SUMMARY_COLUMNS = [
    "backbone",
    "experiment_name",
    "output_dir",
    "batch_size",
    "total_params",
    "trainable_params",
    "macro_auc_mean",
    "macro_auc_std",
    "balanced_accuracy_mean",
    "balanced_accuracy_std",
    "macro_f1_mean",
    "macro_f1_std",
    "macro_recall_mean",
    "macro_recall_std",
    "accuracy_mean",
    "accuracy_std",
    "severe_vs_rest_auc_mean",
    "normal_vs_abnormal_auc_mean",
    "recall_normal_mean",
    "recall_mild_mean",
    "recall_severe_mean",
    "f1_normal_mean",
    "f1_mild_mean",
    "f1_severe_mean",
    "oof_macro_auc",
    "oof_balanced_accuracy",
    "oof_macro_f1",
    "oof_macro_recall",
    "oof_accuracy",
    "oof_severe_vs_rest_auc",
    "oof_normal_vs_abnormal_auc",
]

DELTA_COLUMNS = [
    "delta_macro_auc_mean",
    "delta_balanced_accuracy_mean",
    "delta_macro_f1_mean",
    "delta_recall_severe_mean",
    "delta_f1_severe_mean",
    "delta_severe_vs_rest_auc_mean",
    "delta_normal_vs_abnormal_auc_mean",
    "delta_oof_macro_auc",
    "delta_oof_balanced_accuracy",
    "delta_oof_macro_f1",
    "delta_oof_severe_vs_rest_auc",
    "delta_oof_normal_vs_abnormal_auc",
]


def _read_optional_csv(path: Path) -> pd.DataFrame:
    if not path.is_file():
        return pd.DataFrame()
    return pd.read_csv(path, encoding="utf-8-sig")


def _summary_complete(experiment_dir: Path) -> bool:
    summary_dir = experiment_dir / "summary"
    return all(
        (summary_dir / filename).is_file()
        for filename in [
            "fold_metrics_all.csv",
            "mean_metrics.csv",
            "oof_metrics.csv",
            "oof_predictions.csv",
            "summary_report.md",
        ]
    )


def _read_model_summary(path: Path) -> dict[str, str]:
    summary_path = path / "model_summary.txt"
    if not summary_path.is_file():
        return {}
    parsed: dict[str, str] = {}
    for line in summary_path.read_text(encoding="utf-8", errors="replace").splitlines():
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        parsed[key.strip()] = value.strip()
    return parsed


def _safe_float(value: Any) -> float:
    try:
        if value in {None, ""}:
            return math.nan
        return float(value)
    except Exception:
        return math.nan


def _numeric_mean(frame: pd.DataFrame, column: str) -> float:
    if frame.empty or column not in frame.columns:
        return math.nan
    return float(pd.to_numeric(frame[column], errors="coerce").mean())


def _numeric_std(frame: pd.DataFrame, column: str) -> float:
    if frame.empty or column not in frame.columns:
        return math.nan
    return float(pd.to_numeric(frame[column], errors="coerce").std(ddof=1))


def _oof_value(oof_metrics: pd.DataFrame, column: str) -> float:
    if oof_metrics.empty or column not in oof_metrics.columns:
        return math.nan
    return _safe_float(oof_metrics.iloc[0][column])


def _load_config(experiment_dir: Path) -> dict[str, Any]:
    config_path = experiment_dir / "config.yaml"
    if not config_path.is_file():
        return {}
    return load_yaml(config_path)


def _parameter_counts(backbone: str, experiment_dir: Path) -> tuple[float, float]:
    parsed = _read_model_summary(experiment_dir)
    total = _safe_float(parsed.get("total_params"))
    trainable = _safe_float(parsed.get("trainable_params"))
    if math.isfinite(total) and math.isfinite(trainable):
        return total, trainable
    try:
        model_backbone = "resnet18" if backbone == BASELINE_KEY else backbone
        model = build_nyha_classification_model(
            model_backbone,
            num_classes=3,
            pretrained=False,
        )
        counts = count_parameters(model)
        return float(counts["total_params"]), float(counts["trainable_params"])
    except Exception:
        return math.nan, math.nan


def _row_from_experiment(
    experiment_dir: Path,
    *,
    backbone_override: str | None = None,
    name_override: str | None = None,
    batch_size_override: int | None = None,
    baseline: bool = False,
) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame]:
    summary_dir = experiment_dir / "summary"
    config = _load_config(experiment_dir)
    fold_metrics = _read_optional_csv(summary_dir / "fold_metrics_all.csv")
    oof_metrics = _read_optional_csv(summary_dir / "oof_metrics.csv")

    model_config = config.get("model", {}) if config else {}
    train_config = config.get("train", {}) if config else {}
    experiment_config = config.get("experiment", {}) if config else {}
    backbone = backbone_override or str(model_config.get("backbone", "unknown"))
    experiment_name = name_override or str(
        experiment_config.get("name", experiment_dir.name)
    )
    batch_size = batch_size_override
    if batch_size is None:
        batch_size = train_config.get("batch_size", "")

    total_params, trainable_params = _parameter_counts(backbone, experiment_dir)
    row: dict[str, Any] = {
        "backbone": backbone,
        "experiment_name": experiment_name,
        "output_dir": str(experiment_dir),
        "batch_size": batch_size,
        "total_params": total_params,
        "trainable_params": trainable_params,
        "macro_auc_mean": _numeric_mean(fold_metrics, "macro_auc"),
        "macro_auc_std": _numeric_std(fold_metrics, "macro_auc"),
        "balanced_accuracy_mean": _numeric_mean(fold_metrics, "balanced_accuracy"),
        "balanced_accuracy_std": _numeric_std(fold_metrics, "balanced_accuracy"),
        "macro_f1_mean": _numeric_mean(fold_metrics, "macro_f1"),
        "macro_f1_std": _numeric_std(fold_metrics, "macro_f1"),
        "macro_recall_mean": _numeric_mean(fold_metrics, "macro_recall"),
        "macro_recall_std": _numeric_std(fold_metrics, "macro_recall"),
        "accuracy_mean": _numeric_mean(fold_metrics, "accuracy"),
        "accuracy_std": _numeric_std(fold_metrics, "accuracy"),
        "severe_vs_rest_auc_mean": _numeric_mean(fold_metrics, "severe_vs_rest_auc"),
        "normal_vs_abnormal_auc_mean": _numeric_mean(
            fold_metrics, "normal_vs_abnormal_auc"
        ),
        "recall_normal_mean": _numeric_mean(fold_metrics, "recall_normal"),
        "recall_mild_mean": _numeric_mean(fold_metrics, "recall_mild"),
        "recall_severe_mean": _numeric_mean(fold_metrics, "recall_severe"),
        "f1_normal_mean": _numeric_mean(fold_metrics, "f1_normal"),
        "f1_mild_mean": _numeric_mean(fold_metrics, "f1_mild"),
        "f1_severe_mean": _numeric_mean(fold_metrics, "f1_severe"),
        "oof_macro_auc": _oof_value(oof_metrics, "macro_auc"),
        "oof_balanced_accuracy": _oof_value(oof_metrics, "balanced_accuracy"),
        "oof_macro_f1": _oof_value(oof_metrics, "macro_f1"),
        "oof_macro_recall": _oof_value(oof_metrics, "macro_recall"),
        "oof_accuracy": _oof_value(oof_metrics, "accuracy"),
        "oof_severe_vs_rest_auc": _oof_value(oof_metrics, "severe_vs_rest_auc"),
        "oof_normal_vs_abnormal_auc": _oof_value(
            oof_metrics, "normal_vs_abnormal_auc"
        ),
    }
    if baseline:
        for key, value in BASELINE_FALLBACK.items():
            if not math.isfinite(_safe_float(row.get(key))):
                row[key] = value
    return row, fold_metrics, oof_metrics


def _candidate_dirs_from_queue() -> dict[str, Path]:
    rows = _read_optional_csv(QUEUE_PATH)
    result: dict[str, Path] = {}
    if rows.empty:
        return result
    for _, row in rows.iterrows():
        output_dir = str(row.get("output_dir", "")).strip()
        experiment_name = str(row.get("experiment_name", "")).strip()
        if not output_dir or not experiment_name:
            continue
        path = Path(output_dir).expanduser()
        if path.is_dir() and _summary_complete(path):
            result[experiment_name] = path.resolve()
    return result


def _candidate_dirs_from_scan() -> dict[str, Path]:
    if not EXPERIMENT_ROOT.is_dir():
        return {}
    candidates: dict[str, Path] = {}
    for path in EXPERIMENT_ROOT.iterdir():
        if not path.is_dir() or not path.name.startswith("ModelExploration_"):
            continue
        if not _summary_complete(path):
            continue
        config = _load_config(path)
        experiment_name = str(config.get("experiment", {}).get("name", path.name))
        previous = candidates.get(experiment_name)
        if previous is None or path.stat().st_mtime > previous.stat().st_mtime:
            candidates[experiment_name] = path
    return candidates


def _manifest_metadata() -> dict[str, dict[str, Any]]:
    frame = _read_optional_csv(MANIFEST_PATH)
    if frame.empty:
        return {}
    return {
        str(row["experiment_name"]): row.to_dict()
        for _, row in frame.iterrows()
        if "experiment_name" in row
    }


def _build_experiment_frames() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, bool]:
    rows: list[dict[str, Any]] = []
    fold_frames: list[pd.DataFrame] = []
    oof_frames: list[pd.DataFrame] = []
    baseline_used_fallback = False

    baseline_row, baseline_fold, baseline_oof = _row_from_experiment(
        BASELINE_DIR,
        backbone_override=BASELINE_KEY,
        name_override="ResNet18_ImageNetMeanBG_baseline",
        batch_size_override=16,
        baseline=True,
    )
    for key, value in BASELINE_FALLBACK.items():
        if key in baseline_row and _safe_float(baseline_row[key]) == value:
            baseline_used_fallback = not _summary_complete(BASELINE_DIR)
    rows.append(baseline_row)
    if not baseline_fold.empty:
        baseline_fold = baseline_fold.copy()
        baseline_fold.insert(0, "backbone", BASELINE_KEY)
        baseline_fold.insert(1, "experiment_name", baseline_row["experiment_name"])
        baseline_fold.insert(2, "output_dir", baseline_row["output_dir"])
        fold_frames.append(baseline_fold)
    if not baseline_oof.empty:
        baseline_oof = baseline_oof.copy()
        baseline_oof.insert(0, "backbone", BASELINE_KEY)
        baseline_oof.insert(1, "experiment_name", baseline_row["experiment_name"])
        baseline_oof.insert(2, "output_dir", baseline_row["output_dir"])
        oof_frames.append(baseline_oof)

    metadata = _manifest_metadata()
    candidate_dirs = _candidate_dirs_from_scan()
    candidate_dirs.update(_candidate_dirs_from_queue())
    for experiment_name, experiment_dir in sorted(candidate_dirs.items()):
        meta = metadata.get(experiment_name, {})
        batch_size = meta.get("batch_size", None) if meta else None
        try:
            batch_size = int(batch_size) if batch_size not in {None, ""} else None
        except Exception:
            batch_size = None
        row, fold_metrics, oof_metrics = _row_from_experiment(
            experiment_dir,
            backbone_override=str(meta.get("backbone", "")).strip() or None,
            name_override=experiment_name,
            batch_size_override=batch_size,
        )
        rows.append(row)
        if not fold_metrics.empty:
            frame = fold_metrics.copy()
            frame.insert(0, "backbone", row["backbone"])
            frame.insert(1, "experiment_name", row["experiment_name"])
            frame.insert(2, "output_dir", row["output_dir"])
            fold_frames.append(frame)
        if not oof_metrics.empty:
            frame = oof_metrics.copy()
            frame.insert(0, "backbone", row["backbone"])
            frame.insert(1, "experiment_name", row["experiment_name"])
            frame.insert(2, "output_dir", row["output_dir"])
            oof_frames.append(frame)

    experiment_summary = pd.DataFrame(rows)
    for column in EXPERIMENT_SUMMARY_COLUMNS:
        if column not in experiment_summary.columns:
            experiment_summary[column] = math.nan
    experiment_summary = experiment_summary[EXPERIMENT_SUMMARY_COLUMNS]
    fold_metrics_all = (
        pd.concat(fold_frames, ignore_index=True) if fold_frames else pd.DataFrame()
    )
    oof_metrics_all = (
        pd.concat(oof_frames, ignore_index=True) if oof_frames else pd.DataFrame()
    )
    return experiment_summary, fold_metrics_all, oof_metrics_all, baseline_used_fallback


def _delta_frame(experiment_summary: pd.DataFrame) -> pd.DataFrame:
    baseline = experiment_summary.loc[
        experiment_summary["backbone"] == BASELINE_KEY
    ].iloc[0]
    rows: list[dict[str, Any]] = []
    metric_pairs = {
        "delta_macro_auc_mean": ("macro_auc_mean", "macro_auc_mean"),
        "delta_balanced_accuracy_mean": (
            "balanced_accuracy_mean",
            "balanced_accuracy_mean",
        ),
        "delta_macro_f1_mean": ("macro_f1_mean", "macro_f1_mean"),
        "delta_recall_severe_mean": ("recall_severe_mean", "recall_severe_mean"),
        "delta_f1_severe_mean": ("f1_severe_mean", "f1_severe_mean"),
        "delta_severe_vs_rest_auc_mean": (
            "severe_vs_rest_auc_mean",
            "severe_vs_rest_auc_mean",
        ),
        "delta_normal_vs_abnormal_auc_mean": (
            "normal_vs_abnormal_auc_mean",
            "normal_vs_abnormal_auc_mean",
        ),
        "delta_oof_macro_auc": ("oof_macro_auc", "oof_macro_auc"),
        "delta_oof_balanced_accuracy": (
            "oof_balanced_accuracy",
            "oof_balanced_accuracy",
        ),
        "delta_oof_macro_f1": ("oof_macro_f1", "oof_macro_f1"),
        "delta_oof_severe_vs_rest_auc": (
            "oof_severe_vs_rest_auc",
            "oof_severe_vs_rest_auc",
        ),
        "delta_oof_normal_vs_abnormal_auc": (
            "oof_normal_vs_abnormal_auc",
            "oof_normal_vs_abnormal_auc",
        ),
    }
    for _, row in experiment_summary.iterrows():
        delta = {
            "backbone": row["backbone"],
            "experiment_name": row["experiment_name"],
            "output_dir": row["output_dir"],
        }
        for delta_key, (metric_key, baseline_key) in metric_pairs.items():
            value = _safe_float(row.get(metric_key))
            base = _safe_float(baseline.get(baseline_key))
            delta[delta_key] = value - base if math.isfinite(value) and math.isfinite(base) else math.nan
        rows.append(delta)
    return pd.DataFrame(rows)[
        ["backbone", "experiment_name", "output_dir", *DELTA_COLUMNS]
    ]


def _ranking_frame(experiment_summary: pd.DataFrame) -> pd.DataFrame:
    frame = experiment_summary[
        [
            "backbone",
            "experiment_name",
            "macro_auc_mean",
            "balanced_accuracy_mean",
            "macro_f1_mean",
            "recall_severe_mean",
            "oof_macro_auc",
            "oof_balanced_accuracy",
            "oof_macro_f1",
        ]
    ].copy()
    rank_map = {
        "rank_by_macro_auc_mean": "macro_auc_mean",
        "rank_by_balanced_accuracy_mean": "balanced_accuracy_mean",
        "rank_by_macro_f1_mean": "macro_f1_mean",
        "rank_by_recall_severe_mean": "recall_severe_mean",
        "rank_by_oof_macro_auc": "oof_macro_auc",
        "rank_by_oof_balanced_accuracy": "oof_balanced_accuracy",
        "rank_by_oof_macro_f1": "oof_macro_f1",
    }
    for rank_column, metric_column in rank_map.items():
        frame[rank_column] = (
            pd.to_numeric(frame[metric_column], errors="coerce")
            .rank(ascending=False, method="min")
            .astype("Int64")
        )
    return frame


def _complexity_frame(experiment_summary: pd.DataFrame) -> pd.DataFrame:
    baseline_params = _safe_float(
        experiment_summary.loc[
            experiment_summary["backbone"] == BASELINE_KEY, "total_params"
        ].iloc[0]
    )
    rows = []
    for _, row in experiment_summary.iterrows():
        params = _safe_float(row.get("total_params"))
        ratio = params / baseline_params if math.isfinite(params) and baseline_params else math.nan
        note = "baseline"
        if row["backbone"] != BASELINE_KEY:
            if math.isfinite(ratio) and ratio < 0.75:
                note = "fewer parameters than ResNet18 baseline"
            elif math.isfinite(ratio) and ratio > 2.0:
                note = "substantially larger than ResNet18 baseline"
            else:
                note = "similar parameter scale to ResNet18 baseline"
        rows.append(
            {
                "backbone": row["backbone"],
                "total_params": row["total_params"],
                "trainable_params": row["trainable_params"],
                "batch_size": row["batch_size"],
                "params_vs_resnet18_meanbg": ratio,
                "performance_note": note,
            }
        )
    return pd.DataFrame(rows)


def _recommendation_frame(
    experiment_summary: pd.DataFrame,
    delta_vs_baseline: pd.DataFrame,
    complexity: pd.DataFrame,
) -> pd.DataFrame:
    delta_lookup = delta_vs_baseline.set_index("backbone")
    complexity_lookup = complexity.set_index("backbone")
    rows = []
    for _, row in experiment_summary.iterrows():
        backbone = row["backbone"]
        if backbone == BASELINE_KEY:
            rows.append(
                {
                    "backbone": backbone,
                    "experiment_name": row["experiment_name"],
                    "recommendation": "baseline_reference",
                    "candidate_for_tuning": False,
                    "reason": "Reference ResNet18 + hybrid_imagenet_meanbg baseline.",
                }
            )
            continue
        delta = delta_lookup.loc[backbone]
        d_auc = _safe_float(delta.get("delta_macro_auc_mean"))
        d_ba = _safe_float(delta.get("delta_balanced_accuracy_mean"))
        d_f1 = _safe_float(delta.get("delta_macro_f1_mean"))
        d_severe = _safe_float(delta.get("delta_recall_severe_mean"))
        d_oof_ba = _safe_float(delta.get("delta_oof_balanced_accuracy"))
        d_oof_f1 = _safe_float(delta.get("delta_oof_macro_f1"))
        ratio = _safe_float(
            complexity_lookup.loc[backbone].get("params_vs_resnet18_meanbg")
            if backbone in complexity_lookup.index
            else math.nan
        )

        if d_severe > 0.05 and d_ba < -0.03 and d_f1 < -0.03:
            recommendation = "high_severe_recall_but_unbalanced"
            candidate = False
            reason = (
                "Severe recall improved, but balanced accuracy and macro-F1 declined "
                "materially."
            )
        elif (d_auc >= -0.02 and d_ba >= 0.02) or (d_auc >= -0.02 and d_f1 >= 0.02):
            recommendation = "candidate_for_tuning"
            candidate = True
            reason = "Core fold-level metrics improved without a large macro-AUC drop."
        elif d_severe > 0 and d_ba > -0.03 and d_f1 > -0.03:
            recommendation = "candidate_for_tuning"
            candidate = True
            reason = (
                "Severe recall improved while macro-F1 and balanced accuracy did not "
                "show a material decline."
            )
        elif d_oof_ba >= 0.02 or d_oof_f1 >= 0.02:
            recommendation = "candidate_for_tuning"
            candidate = True
            reason = "OOF balanced accuracy or OOF macro-F1 improved materially."
        elif ratio < 0.75 and d_auc > -0.02 and d_ba > -0.02 and d_f1 > -0.02:
            recommendation = "lightweight_candidate"
            candidate = True
            reason = "Performance is close to baseline with a smaller parameter count."
        elif d_auc < 0 and d_ba < 0 and d_f1 < 0:
            recommendation = "not_recommended"
            candidate = False
            reason = "macro-AUC, balanced accuracy and macro-F1 all declined."
        else:
            recommendation = "not_recommended"
            candidate = False
            reason = "No predefined improvement criterion over ResNet18 was met."
        rows.append(
            {
                "backbone": backbone,
                "experiment_name": row["experiment_name"],
                "recommendation": recommendation,
                "candidate_for_tuning": candidate,
                "reason": reason,
            }
        )
    return pd.DataFrame(rows)


def _best_line(frame: pd.DataFrame, metric: str) -> str:
    values = pd.to_numeric(frame[metric], errors="coerce")
    if values.notna().sum() == 0:
        return f"{metric}: unavailable"
    index = values.idxmax()
    row = frame.loc[index]
    return f"{metric}: {row['backbone']} ({values.loc[index]:.4f})"


def _write_markdown(
    experiment_summary: pd.DataFrame,
    delta_vs_baseline: pd.DataFrame,
    recommendation: pd.DataFrame,
    baseline_used_fallback: bool,
) -> None:
    candidates = recommendation.loc[recommendation["candidate_for_tuning"] == True]
    nonbaseline = experiment_summary.loc[experiment_summary["backbone"] != BASELINE_KEY]
    deltas = delta_vs_baseline.loc[delta_vs_baseline["backbone"] != BASELINE_KEY]
    any_clear_win = False
    if not deltas.empty:
        any_clear_win = bool(
            (
                (pd.to_numeric(deltas["delta_macro_auc_mean"], errors="coerce") > 0)
                | (
                    pd.to_numeric(
                        deltas["delta_balanced_accuracy_mean"], errors="coerce"
                    )
                    > 0
                )
                | (pd.to_numeric(deltas["delta_macro_f1_mean"], errors="coerce") > 0)
            ).any()
        )

    lines = [
        "# Model exploration summary",
        "",
        "## 实验目的",
        "",
        (
            "在固定 hybrid_imagenet_meanbg 预处理、固定 splits_500 五折、固定 "
            "Weighted CE/AdamW/lr/weight decay/epoch/early stopping/augmentation 的前提下，"
            "比较非 ResNet backbone 是否优于当前 ResNet18 meanbg 基线。"
        ),
        "",
        "## 固定变量",
        "",
        "- 数据划分：data/processed/splits_500",
        "- 图像目录：data/processed/global_face/preprocess_ablation/hybrid_imagenet_meanbg/images",
        "- 输入尺寸：224×224",
        "- loss：weighted_cross_entropy；optimizer：AdamW；lr=1e-4；weight_decay=1e-4",
        "- epochs=50；early_stopping_patience=10；monitor_metric=macro_auc",
        "- 不加入 label smoothing、focal loss、ColorJitter、ROI fusion 或阈值优化训练结果。",
        "",
        "## 模型清单",
        "",
        *[
            f"- {row['backbone']}: {row['experiment_name']}"
            for _, row in experiment_summary.iterrows()
        ],
        "",
        "## 与 ResNet18 meanbg 基线对比",
        "",
    ]
    baseline = experiment_summary.loc[experiment_summary["backbone"] == BASELINE_KEY].iloc[0]
    lines.extend(
        [
            f"- 基线 macro-AUC: {_safe_float(baseline['macro_auc_mean']):.4f}",
            (
                "- 基线 balanced accuracy: "
                f"{_safe_float(baseline['balanced_accuracy_mean']):.4f}"
            ),
            f"- 基线 macro-F1: {_safe_float(baseline['macro_f1_mean']):.4f}",
            (
                "- 基线 severe recall: "
                f"{_safe_float(baseline['recall_severe_mean']):.4f}"
            ),
        ]
    )
    if baseline_used_fallback:
        lines.append("- 注意：基线 summary 读取失败，以上基线关键指标使用 prompt fallback 值。")
    lines.extend(
        [
            "",
            "## 最优指标",
            "",
            f"- {_best_line(experiment_summary, 'macro_auc_mean')}",
            f"- {_best_line(experiment_summary, 'balanced_accuracy_mean')}",
            f"- {_best_line(experiment_summary, 'macro_f1_mean')}",
            f"- {_best_line(experiment_summary, 'recall_severe_mean')}",
            "",
            "## 二轮调参建议",
            "",
        ]
    )
    if candidates.empty:
        lines.append("- 当前没有模型满足自动进入二轮调参的规则。")
    else:
        for _, row in candidates.iterrows():
            lines.append(
                f"- {row['backbone']}: {row['recommendation']}；{row['reason']}"
            )

    if not nonbaseline.empty and not any_clear_win:
        lines.extend(
            [
                "",
                "## 研究解释",
                "",
                (
                    "当前若所有非 ResNet backbone 均未明显优于 ResNet18 meanbg，"
                    "说明性能瓶颈可能不主要来自 backbone，而可能来自 mild/severe 标签边界、"
                    "类别不平衡、样本量、决策阈值或 ROI 信息融合不足。"
                ),
            ]
        )

    lines.extend(
        [
            "",
            "## 下一步建议",
            "",
            "- 对 top 2 模型做 lr/weight_decay/dropout/label smoothing 轻量调参。",
            "- 对 top 模型做混淆矩阵和 OOF 阈值扫描。",
            "- 若 backbone 改进有限，转向 ordinal classification、two-stage classification 或 ROI/global fusion。",
            "",
            "本轮实验是模型架构消融，不应写成最终临床模型结论。",
        ]
    )
    SUMMARY_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> Path:
    EXPERIMENT_ROOT.mkdir(parents=True, exist_ok=True)
    (
        experiment_summary,
        fold_metrics_all,
        oof_metrics_all,
        baseline_used_fallback,
    ) = _build_experiment_frames()
    delta_vs_baseline = _delta_frame(experiment_summary)
    ranking = _ranking_frame(experiment_summary)
    complexity = _complexity_frame(experiment_summary)
    recommendation = _recommendation_frame(
        experiment_summary, delta_vs_baseline, complexity
    )

    experiment_summary.to_csv(SUMMARY_CSV, index=False, encoding="utf-8-sig")
    write_xlsx(
        SUMMARY_XLSX,
        {
            "experiment_summary": experiment_summary,
            "fold_metrics_all": fold_metrics_all,
            "oof_metrics_all": oof_metrics_all,
            "delta_vs_resnet18_meanbg": delta_vs_baseline,
            "ranking": ranking,
            "model_complexity": complexity,
            "recommendation": recommendation,
        },
    )
    _write_markdown(
        experiment_summary,
        delta_vs_baseline,
        recommendation,
        baseline_used_fallback,
    )
    print(f"SUMMARY_CSV={SUMMARY_CSV}")
    print(f"SUMMARY_XLSX={SUMMARY_XLSX}")
    print(f"SUMMARY_MD={SUMMARY_MD}")
    return SUMMARY_XLSX


if __name__ == "__main__":
    main()
