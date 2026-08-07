"""Create complete five-fold E1 summaries from held-out fold artifacts."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from metrics.classification_metrics import flatten_metrics  # noqa: E402
from metrics.e1_hierarchical_metrics import compute_e1_metrics  # noqa: E402
from utils.experiment_utils import load_yaml, resolve_project_path  # noqa: E402


SUMMARY_METRICS = [
    "macro_auc", "accuracy", "macro_precision", "macro_recall", "macro_f1", "balanced_accuracy",
    "auc_normal", "auc_mild", "auc_severe", "precision_normal", "precision_mild", "precision_severe",
    "recall_normal", "recall_mild", "recall_severe", "f1_normal", "f1_mild", "f1_severe",
    "auc_normal_vs_abnormal", "auc_mild_vs_severe_cond", "ordinal_mae", "two_step_error_rate",
    "qwk", "severe_recall", "multiclass_brier_score", "ece_15_equal_width",
    "abnormal_binary_brier_score", "severe_cond_binary_brier_score",
]


def _mean_std(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for metric in SUMMARY_METRICS:
        if metric not in frame:
            raise ValueError(f"Fold metrics are missing required E1 metric: {metric}")
        values = pd.to_numeric(frame[metric], errors="coerce")
        rows.append({"metric": metric, "mean": values.mean(), "std": values.std(ddof=1)})
    return pd.DataFrame(rows)


def _expected_image_ids(config: dict[str, Any]) -> list[str]:
    split_dir = resolve_project_path(config["data"]["split_dir"])
    if split_dir is None:
        raise ValueError("data.split_dir cannot be empty")
    ids: list[str] = []
    for fold in range(int(config["data"]["n_folds"])):
        frame = pd.read_csv(split_dir / str(config["data"]["val_csv_pattern"]).format(fold=fold), dtype={"ID": "string"}, encoding="utf-8-sig")
        if "ID" not in frame:
            raise ValueError(f"E1 validation split {fold} has no image-level ID column")
        ids.extend(frame["ID"].astype(str).tolist())
    if pd.Series(ids, dtype="string").duplicated().any():
        raise ValueError("Fixed validation folds contain duplicate image-level IDs; cannot make OOF assertions")
    return ids


def _jsonable(metrics: dict[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in metrics.items():
        if isinstance(value, np.ndarray):
            result[key] = value.tolist()
        elif isinstance(value, np.generic):
            result[key] = value.item()
        else:
            result[key] = value
    return result


def _write_report(path: Path, config: dict, mean_std: pd.DataFrame, oof: dict[str, Any]) -> None:
    lookup = mean_std.set_index("metric")
    lines = [
        f"# {config['experiment']['name']}", "",
        "All values below are produced from fixed-fold held-out predictions. E1 uses joint restored probabilities: P(N)=1-P(A), P(M)=P(A)(1-P(S|A)), P(S)=P(A)P(S|A).", "",
        "## Fold mean ± std", "", "| Metric | Mean ± std |", "|---|---:|",
    ]
    for metric in SUMMARY_METRICS:
        row = lookup.loc[metric]
        lines.append(f"| {metric} | {row['mean']:.4f} ± {row['std']:.4f} |")
    lines.extend(["", "## Pooled OOF", "", "| Metric | Value |", "|---|---:|"])
    for key, value in flatten_metrics(oof).items():
        lines.append(f"| {key} | {value:.4f} |")
    matrix = np.asarray(oof["confusion_matrix"])
    lines.extend(["", "## Interpretation guardrail", "", "Do not declare E1 successful from Macro-AUC alone. Interpret it together with Macro-F1, balanced accuracy, severe recall, ordinal MAE, two-step errors, QWK, and both boundary AUCs.", "", "## ID mapping", "", "`image_id` is the project image-level `ID`; `patient_id` is `patient_group_id`. No IDs were fabricated.", "", "## OOF confusion matrix", "", "| True \\ Pred | Normal | Mild | Severe |", "|---|---:|---:|---:|", f"| Normal | {matrix[0,0]} | {matrix[0,1]} | {matrix[0,2]} |", f"| Mild | {matrix[1,0]} | {matrix[1,1]} | {matrix[1,2]} |", f"| Severe | {matrix[2,0]} | {matrix[2,1]} | {matrix[2,2]} |"])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def summarize(experiment_dir: str | Path) -> Path:
    directory = resolve_project_path(experiment_dir)
    if directory is None:
        raise ValueError("experiment_dir cannot be empty")
    config, summary_dir = load_yaml(directory / "config.yaml"), directory / "summary"
    summary_dir.mkdir(parents=True, exist_ok=True)
    metrics_frames, prediction_frames = [], []
    for fold in range(int(config["data"]["n_folds"])):
        metrics_path = directory / f"fold_{fold}" / "metrics" / "fold_metrics.csv"
        prediction_path = directory / f"fold_{fold}" / "predictions" / "val_predictions.csv"
        if not metrics_path.is_file() or not prediction_path.is_file():
            raise FileNotFoundError(f"E1 fold {fold} is incomplete and cannot be included in OOF summaries")
        metrics_frames.append(pd.read_csv(metrics_path))
        prediction_frames.append(pd.read_csv(prediction_path, dtype={"patient_id": "string", "image_id": "string"}))
    fold_metrics = pd.concat(metrics_frames, ignore_index=True).sort_values("fold")
    oof = pd.concat(prediction_frames, ignore_index=True).sort_values(["fold", "image_id"], kind="stable")
    expected = _expected_image_ids(config)
    if oof["image_id"].duplicated().any() or set(oof["image_id"].astype(str)) != set(expected) or len(oof) != len(expected):
        raise ValueError("E1 OOF image IDs are duplicated, missing, or unexpected relative to the fixed validation folds")
    fold_metrics.to_csv(summary_dir / "fold_metrics_all.csv", index=False, encoding="utf-8-sig")
    mean_std = _mean_std(fold_metrics)
    mean_std.to_csv(summary_dir / "mean_std_metrics.csv", index=False, encoding="utf-8-sig")
    oof.to_csv(summary_dir / "oof_predictions.csv", index=False, encoding="utf-8-sig")
    metrics = compute_e1_metrics(oof["true_label"], oof[["prob_normal", "prob_mild", "prob_severe"]], oof["p_abnormal"], oof["p_severe_cond"], int(config.get("metrics", {}).get("ece_bins", 15)))
    (summary_dir / "oof_metrics.json").write_text(json.dumps(_jsonable(metrics), indent=2, allow_nan=True), encoding="utf-8")
    pd.DataFrame(metrics["confusion_matrix"], index=["normal", "mild", "severe"], columns=["normal", "mild", "severe"]).to_csv(summary_dir / "confusion_matrix.csv", index_label="true\\pred", encoding="utf-8-sig")
    pd.DataFrame([{key: metrics[key] for key in ["auc_normal_vs_abnormal", "auc_mild_vs_severe_cond", "abnormal_binary_brier_score", "severe_cond_binary_brier_score"]}]).to_csv(summary_dir / "boundary_metrics.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame([{key: metrics[key] for key in ["ordinal_mae", "normal_to_severe_count", "severe_to_normal_count", "two_step_error_count", "two_step_error_rate", "qwk", "severe_recall"]}]).to_csv(summary_dir / "ordinal_metrics.csv", index=False, encoding="utf-8-sig")
    _write_report(summary_dir / "summary_report.md", config, mean_std, metrics)
    return summary_dir


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment-dir", type=Path, required=True)
    print(f"SUMMARY_DIR={summarize(parser.parse_args().experiment_dir)}")
