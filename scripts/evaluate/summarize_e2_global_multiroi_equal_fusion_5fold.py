"""Strict standalone five-fold E2 OOF summary generation."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from metrics.e2_metrics import compute_e2_metrics  # noqa: E402
from utils.experiment_utils import load_yaml, resolve_project_path  # noqa: E402


METRICS = [
    "macro_auc",
    "accuracy",
    "macro_precision",
    "macro_recall",
    "macro_f1",
    "balanced_accuracy",
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
    "ordinal_mae",
    "qwk",
    "severe_recall",
    "normal_to_severe_count",
    "severe_to_normal_count",
    "two_step_error_count",
    "two_step_error_rate",
    "multiclass_brier_score",
    "ece_15_equal_width",
    "nll",
]
REQUIRED_PREDICTION_COLUMNS = {
    "sample_id",
    "patient_group_id",
    "fold",
    "true_label",
    "pred_label",
    "prob_normal",
    "prob_mild",
    "prob_severe",
    "global_path",
    "eye_path",
    "lip_path",
    "cheek_path",
    "forehead_path",
    "chin_path",
}


def _load_fold(
    root: Path, fold: int
) -> tuple[pd.DataFrame, pd.DataFrame]:
    fold_dir = root / f"fold_{fold}"
    metric_path = fold_dir / "metrics" / "fold_metrics.csv"
    prediction_path = fold_dir / "predictions" / "val_predictions.csv"
    checkpoint_path = fold_dir / "checkpoints" / "best_macro_auc.pth"
    if not (
        metric_path.is_file()
        and prediction_path.is_file()
        and checkpoint_path.is_file()
    ):
        raise FileNotFoundError(f"Fold {fold} is incomplete and cannot enter E2 OOF")
    fold_metrics = pd.read_csv(metric_path)
    predictions = pd.read_csv(
        prediction_path,
        dtype={"sample_id": "string", "patient_group_id": "string"},
    )
    if (
        len(fold_metrics) != 1
        or "fold" not in fold_metrics
        or int(fold_metrics["fold"].iloc[0]) != fold
    ):
        raise ValueError(f"Fold {fold} metrics are malformed")
    missing = sorted(REQUIRED_PREDICTION_COLUMNS.difference(predictions.columns))
    if missing:
        raise ValueError(
            f"Fold {fold} prediction export is missing required columns: {missing}"
        )
    if predictions.empty or not (predictions["fold"].astype(int) == fold).all():
        raise ValueError(f"Fold {fold} prediction export has invalid fold values")
    return fold_metrics, predictions


def _expected_fold_targets(config: dict) -> dict[tuple[int, str], tuple[int, str]]:
    split_dir = resolve_project_path(config["data"]["split_dir"])
    if split_dir is None:
        raise ValueError("E2 data.split_dir is empty")
    expected: dict[tuple[int, str], tuple[int, str]] = {}
    for fold in range(5):
        path = split_dir / str(config["data"]["val_csv_pattern"]).format(fold=fold)
        frame = pd.read_csv(path, dtype={"ID": "string"}, encoding="utf-8-sig")
        for row in frame.itertuples(index=False):
            key = (fold, str(row.ID))
            if key in expected:
                raise ValueError(f"Duplicate expected fold/ID key: {key}")
            expected[key] = (int(row.label_3class), str(row.patient_group_id))
    return expected


def _validate_oof(
    config: dict,
    oof: pd.DataFrame,
    probabilities: np.ndarray,
) -> None:
    expected = _expected_fold_targets(config)
    exported = {
        (int(row.fold), str(row.sample_id)): (
            int(row.true_label),
            str(row.patient_group_id),
        )
        for row in oof.itertuples(index=False)
    }
    group_to_folds = (
        oof.groupby("patient_group_id", dropna=False)["fold"].nunique().to_numpy()
    )
    valid = (
        len(oof) == int(config["data"]["expected_num_samples"])
        and not oof["sample_id"].duplicated().any()
        and set(oof["fold"].astype(int)) == set(range(5))
        and expected == exported
        and np.all(group_to_folds == 1)
        and np.isfinite(probabilities).all()
        and not (probabilities < 0).any()
        and not (probabilities > 1).any()
        and np.allclose(probabilities.sum(axis=1), 1.0, atol=1e-5)
    )
    if not valid:
        raise ValueError(
            "E2 OOF integrity failure: count, IDs/groups, true labels, folds, "
            "or probabilities are invalid"
        )


def summarize(experiment_dir: str | Path) -> Path:
    root = Path(experiment_dir).resolve()
    config = load_yaml(root / "resolved_config.yaml")
    if (
        int(config["data"]["n_folds"]) != 5
        or int(config["data"]["expected_num_samples"]) != 500
    ):
        raise ValueError("E2 summary requires the fixed 500-sample five-fold protocol")

    metrics_frames, prediction_frames = zip(
        *(_load_fold(root, fold) for fold in range(5))
    )
    fold_metrics = pd.concat(metrics_frames, ignore_index=True).sort_values("fold")
    missing_metrics = sorted(set(METRICS).difference(fold_metrics.columns))
    if missing_metrics:
        raise ValueError(f"E2 fold metrics are missing: {missing_metrics}")
    oof = pd.concat(prediction_frames, ignore_index=True).sort_values(
        ["fold", "sample_id"], kind="stable"
    )
    probabilities = oof[["prob_normal", "prob_mild", "prob_severe"]].to_numpy(float)
    _validate_oof(config, oof, probabilities)

    summary = root / "summary"
    summary.mkdir(parents=True, exist_ok=True)
    fold_metrics.to_csv(summary / "fold_metrics.csv", index=False, encoding="utf-8-sig")
    mean_std = pd.DataFrame(
        [
            {
                "metric": metric,
                "mean": pd.to_numeric(fold_metrics[metric], errors="coerce").mean(),
                "std": pd.to_numeric(
                    fold_metrics[metric], errors="coerce"
                ).std(ddof=1),
            }
            for metric in METRICS
        ]
    )
    mean_std.to_csv(summary / "mean_std_metrics.csv", index=False, encoding="utf-8-sig")
    oof.to_csv(summary / "oof_predictions.csv", index=False, encoding="utf-8-sig")

    pooled = compute_e2_metrics(
        oof["true_label"],
        probabilities,
        int(config.get("metrics", {}).get("ece_bins", 15)),
    )
    serializable: dict[str, Any] = {
        key: value.tolist() if isinstance(value, np.ndarray) else value
        for key, value in pooled.items()
    }
    (summary / "pooled_oof_metrics.json").write_text(
        json.dumps(serializable, indent=2, allow_nan=True),
        encoding="utf-8",
    )
    pd.DataFrame(
        pooled["confusion_matrix"],
        index=["normal", "mild", "severe"],
        columns=["normal", "mild", "severe"],
    ).to_csv(
        summary / "pooled_confusion_matrix.csv",
        index_label="true\\pred",
        encoding="utf-8-sig",
    )
    report = [
        f"# {config['experiment']['name']}",
        "",
        (
            "This report is descriptive. Compare fold and pooled Macro-AUC with "
            "Macro-F1, balanced accuracy, severe recall, QWK, ordinal MAE, and "
            "two-step errors against E0 before drawing a conclusion."
        ),
        "",
        "## Pooled OOF metrics",
        "",
        "| Metric | Value |",
        "|---|---:|",
    ]
    report.extend(
        f"| {key} | {value:.4f} |"
        for key, value in pooled.items()
        if np.isscalar(value)
    )
    (summary / "summary_report.md").write_text(
        "\n".join(report) + "\n", encoding="utf-8"
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate the strict E2 five-fold OOF summary."
    )
    parser.add_argument("--experiment-dir", required=True, type=Path)
    args = parser.parse_args()
    summary = summarize(args.experiment_dir)
    print(f"SUMMARY_DIR={summary}")


if __name__ == "__main__":
    main()
