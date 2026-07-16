"""Validate and summarize outer-test predictions for one nested-CV method."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import confusion_matrix


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from metrics.classification_metrics import (  # noqa: E402
    CLASS_NAMES,
    compute_classification_metrics,
    flatten_metrics,
)
from utils.ordinal_utils import compute_ordinal_metrics  # noqa: E402


OUTPUT_ROOT = PROJECT_ROOT / "experiments/ordinal_stage1_nested5x5_500Data"
CONFIGS = {
    "ce": PROJECT_ROOT / "config/train/ordinal/nyha_3class_resnet18_meanbg_nested5x5_weighted_ce.yaml",
    "ordinal": PROJECT_ROOT / "config/train/ordinal/nyha_3class_resnet18_meanbg_nested5x5_monotonic_cumulative.yaml",
}
PROBABILITY_COLUMNS = ["prob_normal", "prob_mild", "prob_severe"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", choices=["ce", "ordinal"], required=True)
    parser.add_argument("--smoke-test", action="store_true")
    return parser.parse_args()


def experiment_dir(method: str, smoke_test: bool) -> Path:
    config = yaml.safe_load(CONFIGS[method].read_text(encoding="utf-8"))
    parent = OUTPUT_ROOT / "smoke_tests" if smoke_test else OUTPUT_ROOT
    name = config["experiment"]["name"] + ("_SMOKE" if smoke_test else "")
    return parent / name


def validate_predictions(frame: pd.DataFrame, expected_rows: int) -> None:
    required = {
        "ID",
        "patient_group_id",
        "NYHA",
        "SEX",
        "label_3class",
        "label_3class_name",
        "pred_class",
        "pred_class_name",
        *PROBABILITY_COLUMNS,
        "correct",
        "fold",
    }
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"OOF predictions missing columns: {missing}")
    if len(frame) != expected_rows or frame["ID"].nunique() != expected_rows:
        raise ValueError(
            f"OOF coverage invalid: rows={len(frame)}, unique={frame['ID'].nunique()}, expected={expected_rows}"
        )
    probabilities = frame[PROBABILITY_COLUMNS].to_numpy(dtype=float)
    if not np.isfinite(probabilities).all():
        raise ValueError("OOF probabilities contain non-finite values")
    if (probabilities < -1e-8).any() or (probabilities > 1 + 1e-8).any():
        raise ValueError("OOF probabilities are outside [0,1]")
    if not np.allclose(probabilities.sum(axis=1), 1.0, atol=1e-6):
        raise ValueError("OOF probability rows do not sum to one")
    if not np.array_equal(probabilities.argmax(axis=1), frame["pred_class"].astype(int).to_numpy()):
        raise ValueError("pred_class does not equal probability argmax")


def all_metrics(frame: pd.DataFrame) -> dict[str, Any]:
    y_true = frame["label_3class"].astype(int).to_numpy()
    probabilities = frame[PROBABILITY_COLUMNS].to_numpy(dtype=float)
    classification = compute_classification_metrics(y_true, probabilities)
    predicted = probabilities.argmax(axis=1)
    return {
        **flatten_metrics(classification),
        **compute_ordinal_metrics(y_true, predicted),
        "confusion_matrix": classification["confusion_matrix"],
    }


def main() -> Path:
    args = parse_args()
    root = experiment_dir(args.method, args.smoke_test)
    outer_folds = [0] if args.smoke_test else list(range(5))
    predictions = []
    metric_rows = []
    selected_rows = []
    for outer_fold in outer_folds:
        prediction_path = root / f"outer_fold_{outer_fold}/predictions/outer_test_predictions.csv"
        metric_path = root / f"outer_fold_{outer_fold}/metrics/outer_test_metrics.csv"
        selected_path = root / f"outer_fold_{outer_fold}/selection/selected_epoch.json"
        if not prediction_path.is_file() or not metric_path.is_file() or not selected_path.is_file():
            raise FileNotFoundError(f"outer fold {outer_fold} is incomplete under {root}")
        predictions.append(
            pd.read_csv(
                prediction_path,
                dtype={"ID": "string", "patient_group_id": "string"},
                encoding="utf-8-sig",
            )
        )
        metric_rows.append(pd.read_csv(metric_path, encoding="utf-8-sig"))
        selected_rows.append(json.loads(selected_path.read_text(encoding="utf-8")))
    oof = pd.concat(predictions, ignore_index=True).sort_values(["fold", "ID"], kind="stable")
    expected_rows = 100 if args.smoke_test else 500
    validate_predictions(oof, expected_rows)
    if not args.smoke_test and set(oof["fold"].astype(int)) != set(range(5)):
        raise ValueError("formal OOF does not cover folds 0..4")
    if args.method == "ordinal":
        violations = sum(
            int(frame["monotonic_violation_count"].iloc[0]) for frame in metric_rows
        )
        if violations != 0:
            raise ValueError(f"ordinal OOF has {violations} monotonic violations")
    summary_dir = root / "summary"
    summary_dir.mkdir(parents=True, exist_ok=True)
    oof.to_csv(summary_dir / "oof_predictions.csv", index=False, encoding="utf-8-sig")
    fold_metrics = pd.concat(metric_rows, ignore_index=True).sort_values("outer_fold")
    fold_metrics.to_csv(summary_dir / "outer_fold_metrics.csv", index=False, encoding="utf-8-sig")
    metric_names = [
        "macro_auc",
        "accuracy",
        "balanced_accuracy",
        "macro_precision",
        "macro_recall",
        "macro_f1",
        "auc_normal",
        "auc_mild",
        "auc_severe",
        "normal_vs_abnormal_auc",
        "severe_vs_rest_auc",
        "ordinal_mae",
        "within_one_accuracy",
        "extreme_error_rate",
        "quadratic_weighted_kappa",
    ]
    fold_summary = pd.DataFrame(
        [
            {
                "metric": metric,
                "fold_mean": pd.to_numeric(fold_metrics[metric], errors="coerce").mean(),
                "fold_std": pd.to_numeric(fold_metrics[metric], errors="coerce").std(ddof=1),
                "valid_folds": pd.to_numeric(fold_metrics[metric], errors="coerce").notna().sum(),
            }
            for metric in metric_names
        ]
    )
    metrics = all_metrics(oof)
    oof_row = {key: value for key, value in metrics.items() if key != "confusion_matrix"}
    pd.DataFrame([oof_row]).to_csv(summary_dir / "oof_metrics.csv", index=False, encoding="utf-8-sig")
    fold_summary.to_csv(summary_dir / "fold_metric_summary.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(selected_rows).to_csv(summary_dir / "selected_epochs.csv", index=False, encoding="utf-8-sig")
    matrix = np.asarray(metrics["confusion_matrix"])
    matrix_frame = pd.DataFrame(
        matrix,
        index=[CLASS_NAMES[i] for i in range(3)],
        columns=[CLASS_NAMES[i] for i in range(3)],
    )
    matrix_frame.to_csv(summary_dir / "oof_confusion_matrix.csv", index_label="true\\pred", encoding="utf-8-sig")
    plt.figure(figsize=(5, 4))
    plt.imshow(matrix, cmap="Blues")
    plt.colorbar()
    plt.xticks(range(3), [CLASS_NAMES[i] for i in range(3)])
    plt.yticks(range(3), [CLASS_NAMES[i] for i in range(3)])
    for i in range(3):
        for j in range(3):
            plt.text(j, i, str(int(matrix[i, j])), ha="center", va="center")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    plt.savefig(summary_dir / "oof_confusion_matrix.png", dpi=180)
    plt.close()
    report_lines = [
        f"# {args.method} nested-CV summary",
        "",
        f"- Rows: {len(oof)}",
        f"- Unique IDs: {oof['ID'].nunique()}",
        f"- Outer folds: {sorted(oof['fold'].astype(int).unique().tolist())}",
        "- Outer test predictions were produced only by final full-outer-train refit models.",
        "",
        "## OOF metrics",
        "",
        "| metric | value |",
        "|---|---:|",
    ]
    for metric in metric_names:
        report_lines.append(f"| {metric} | {float(oof_row[metric]):.6f} |")
    (summary_dir / "summary_report.md").write_text("\n".join(report_lines), encoding="utf-8")
    print(f"SUMMARY_DIR={summary_dir}")
    return summary_dir


if __name__ == "__main__":
    main()
