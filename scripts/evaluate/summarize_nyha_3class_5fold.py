"""Summarize fold-level and out-of-fold NYHA three-class results."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from metrics.classification_metrics import (  # noqa: E402
    compute_classification_metrics,
    flatten_metrics,
)
from utils.experiment_utils import load_yaml, resolve_project_path  # noqa: E402


MAIN_METRICS = [
    "macro_auc",
    "accuracy",
    "macro_precision",
    "macro_recall",
    "macro_f1",
    "balanced_accuracy",
]
AUXILIARY_METRICS = [
    "auc_normal",
    "auc_mild",
    "auc_severe",
    "precision_normal",
    "precision_mild",
    "precision_severe",
    "recall_normal",
    "recall_mild",
    "recall_severe",
    "f1_normal",
    "f1_mild",
    "f1_severe",
    "severe_vs_rest_auc",
    "normal_vs_abnormal_auc",
]


BACKBONE_FEATURE_DIMS = {"resnet18": 512, "resnet34": 512, "resnet50": 2048}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment-dir", type=Path, required=True)
    return parser.parse_args()


def _format_metric(value: float) -> str:
    return "nan" if pd.isna(value) else f"{value:.4f}"


def _mean_std_rows(
    fold_metrics: pd.DataFrame, metric_names: list[str]
) -> pd.DataFrame:
    rows = []
    for metric in metric_names:
        if metric not in fold_metrics.columns:
            raise ValueError(f"Fold metrics are missing required column: {metric}")
        values = pd.to_numeric(fold_metrics[metric], errors="coerce")
        rows.append(
            {
                "metric": metric,
                "mean": values.mean(),
                "std": values.std(ddof=1),
            }
        )
    return pd.DataFrame(rows)


def _loss_report_lines(config: dict[str, Any]) -> list[str]:
    if "loss" in config and isinstance(config["loss"], dict):
        loss_config = config["loss"]
        label_smoothing = loss_config.get("label_smoothing", {}) or {}
        class_weight = (
            "fold-specific class weights"
            if bool(loss_config.get("class_weight", True))
            else "disabled"
        )
        lines = [
            f"- Loss: {loss_config.get('name')}",
            f"- Class weight: {class_weight}",
            (
                "- Label smoothing: "
                f"{'enabled' if bool(label_smoothing.get('enabled', False)) else 'disabled'}"
            ),
        ]
        if bool(label_smoothing.get("enabled", False)):
            alpha = float(label_smoothing.get("alpha", 0.0))
            num_classes = int(config["data"]["num_classes"])
            other_value = alpha / (num_classes - 1)
            true_value = 1.0 - alpha
            lines.extend(
                [
                    f"- Label smoothing alpha: {alpha:g}",
                    f"- Label smoothing mode: {label_smoothing.get('mode')}",
                    (
                        "- Smoothed target rule: true class = 1 - alpha, "
                        "other classes = alpha / (num_classes - 1)"
                    ),
                    (
                        f"- For alpha = {alpha:g} and num_classes = {num_classes}: "
                        f"true class = {true_value:.2f}, other classes = {other_value:.2f}"
                    ),
                ]
            )
        return lines

    return [
        (
            f"- Loss: {config['train'].get('loss', 'weighted_cross_entropy')} "
            "with fold-specific class weights"
        )
    ]


def _model_report_lines(config: dict[str, Any]) -> list[str]:
    model_config = config["model"]
    data_config = config["data"]
    model_type = str(model_config.get("type", "single_image"))
    lines = [
        f"- Model type: {model_type}",
        f"- Backbone: {model_config['backbone']}",
        f"- Pretrained weights: {model_config['pretrained']}",
    ]
    if model_type == "multi_roi_fusion":
        roi_names = list(data_config["roi_names"])
        feature_dim = BACKBONE_FEATURE_DIMS[str(model_config["backbone"]).lower()]
        fusion_dim = feature_dim * len(roi_names)
        fusion_head = model_config.get("fusion_head", {}) or {}
        lines.extend(
            [
                "- Dataset: ROI_Fusion_500",
                f"- Split dir: `{data_config['split_dir']}`",
                f"- ROI root: `{data_config['roi_root']}`",
                f"- ROI names: {', '.join(roi_names)}",
                f"- Number of ROIs: {len(roi_names)}",
                f"- Shared backbone: {str(model_config['shared_backbone']).lower()}",
                f"- Fusion method: {model_config['fusion_method']}",
                f"- Feature dim per ROI: {feature_dim}",
                f"- Fusion dim: {fusion_dim}",
                f"- Fusion head hidden dim: {fusion_head.get('hidden_dim')}",
                f"- Fusion head dropout: {fusion_head.get('dropout')}",
                f"- Fusion head batchnorm: {str(fusion_head.get('use_batchnorm')).lower()}",
            ]
        )
    return lines


def _write_report(
    path: Path,
    config: dict[str, Any],
    mean_frame: pd.DataFrame,
    auxiliary_frame: pd.DataFrame,
    oof_metrics: dict[str, Any],
) -> None:
    lookup = mean_frame.set_index("metric")
    auxiliary_lookup = auxiliary_frame.set_index("metric")
    image_size = config["data"]["image_size"]
    lines = [
        f"# {config['experiment']['name']}",
        "",
        "## Experiment setup",
        "",
        *_model_report_lines(config),
        f"- Input: RGB {image_size}\u00d7{image_size}",
        f"- Fixed fold files: `{config['data']['split_dir']}`",
        *(
            [f"- Image root override: `{config['data']['image_root']}`"]
            if config["data"].get("image_root")
            else []
        ),
        (
            f"- Training: AdamW, lr={config['train']['lr']}, "
            f"weight_decay={config['train']['weight_decay']}, "
            f"epochs={config['train']['epochs']}, "
            f"early stopping patience={config['train']['early_stopping_patience']}"
        ),
        *_loss_report_lines(config),
        (
            "- Class weight rule: fold-specific N / (num_classes * class_count)"
        ),
        (
            "- Augmentation: resize + horizontal flip (train only); "
            "no crop or color jitter"
        ),
        *(
            [
                "- same_flip_for_all_rois: "
                f"{str(config['augmentation'].get('same_flip_for_all_rois')).lower()}"
            ]
            if config["model"].get("type") == "multi_roi_fusion"
            else []
        ),
        "",
        "## Fold-level mean \u00b1 std",
        "",
        "| Metric | Mean \u00b1 std |",
        "|---|---:|",
    ]
    for metric in MAIN_METRICS:
        lines.append(
            f"| {metric} | {_format_metric(lookup.loc[metric, 'mean'])} "
            f"\u00b1 {_format_metric(lookup.loc[metric, 'std'])} |"
        )

    lines.extend(
        [
            "",
            "## Fold-level auxiliary metrics",
            "",
            "| Metric | Mean \u00b1 std |",
            "|---|---:|",
        ]
    )
    for metric in AUXILIARY_METRICS:
        lines.append(
            f"| {metric} | "
            f"{_format_metric(auxiliary_lookup.loc[metric, 'mean'])} "
            f"\u00b1 {_format_metric(auxiliary_lookup.loc[metric, 'std'])} |"
        )

    lines.extend(["", "## OOF metrics", "", "| Metric | Value |", "|---|---:|"])
    for metric, value in flatten_metrics(oof_metrics).items():
        lines.append(f"| {metric} | {_format_metric(value)} |")

    matrix = np.asarray(oof_metrics["confusion_matrix"])
    lines.extend(
        [
            "",
            "## OOF confusion matrix",
            "",
            "| True \\ Pred | normal | mild | severe |",
            "|---|---:|---:|---:|",
            f"| normal | {matrix[0, 0]} | {matrix[0, 1]} | {matrix[0, 2]} |",
            f"| mild | {matrix[1, 0]} | {matrix[1, 1]} | {matrix[1, 2]} |",
            f"| severe | {matrix[2, 0]} | {matrix[2, 1]} | {matrix[2, 2]} |",
            "",
            "All fold results are held-out validation results, not an independent test set.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> Path:
    args = parse_args()
    experiment_dir = resolve_project_path(args.experiment_dir)
    if experiment_dir is None:
        raise ValueError("--experiment-dir must not be empty")
    config = load_yaml(experiment_dir / "config.yaml")
    summary_dir = experiment_dir / "summary"
    summary_dir.mkdir(parents=True, exist_ok=True)
    n_folds = int(config["data"]["n_folds"])

    metric_frames: list[pd.DataFrame] = []
    prediction_frames: list[pd.DataFrame] = []
    for fold in range(n_folds):
        metric_path = experiment_dir / f"fold_{fold}" / "metrics" / "fold_metrics.csv"
        prediction_path = (
            experiment_dir
            / f"fold_{fold}"
            / "predictions"
            / "val_predictions.csv"
        )
        if not metric_path.is_file() or not prediction_path.is_file():
            raise FileNotFoundError(
                f"Fold {fold} artifacts are incomplete: {metric_path}, {prediction_path}"
            )
        metric_frames.append(pd.read_csv(metric_path))
        prediction_frames.append(
            pd.read_csv(
                prediction_path,
                dtype={"ID": "string", "patient_group_id": "string"},
            )
        )

    fold_metrics = pd.concat(metric_frames, ignore_index=True).sort_values("fold")
    fold_metrics.to_csv(
        summary_dir / "fold_metrics_all.csv", index=False, encoding="utf-8-sig"
    )
    mean_frame = _mean_std_rows(fold_metrics, MAIN_METRICS)
    mean_frame.to_csv(
        summary_dir / "mean_metrics.csv", index=False, encoding="utf-8-sig"
    )
    auxiliary_frame = _mean_std_rows(fold_metrics, AUXILIARY_METRICS)

    oof = pd.concat(prediction_frames, ignore_index=True).sort_values(
        ["fold", "ID"], kind="stable"
    )
    if oof["ID"].duplicated().any():
        duplicates = oof.loc[oof["ID"].duplicated(keep=False), "ID"].tolist()
        raise ValueError(
            f"OOF predictions contain duplicate sample IDs: {duplicates[:10]}"
        )
    oof.to_csv(
        summary_dir / "oof_predictions.csv", index=False, encoding="utf-8-sig"
    )
    oof_metrics = compute_classification_metrics(
        oof["label_3class"].to_numpy(),
        oof[["prob_normal", "prob_mild", "prob_severe"]].to_numpy(),
    )
    pd.DataFrame([flatten_metrics(oof_metrics)]).to_csv(
        summary_dir / "oof_metrics.csv", index=False, encoding="utf-8-sig"
    )
    _write_report(
        summary_dir / "summary_report.md",
        config,
        mean_frame,
        auxiliary_frame,
        oof_metrics,
    )
    print(f"SUMMARY_DIR={summary_dir}")
    return summary_dir


if __name__ == "__main__":
    main()
