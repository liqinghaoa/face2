"""Validate and summarize pooled out-of-fold predictions for the R3DPR OneCycle variant."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from metrics.R3DPR.binary_classification_metrics import compute_binary_metrics, flatten_metrics
from scripts.evaluate.R3DPR.summarize_r3dpr_resnet18_binary_5fold import expected_table, validate_oof
from utils.experiment_utils import load_yaml


METRICS = ["macro_auc", "accuracy", "macro_precision", "macro_recall", "macro_f1", "balanced_accuracy"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment-dir", type=Path, required=True)
    return parser.parse_args()


def make_report(config: dict, fold_metrics: pd.DataFrame, metrics: dict, matrix: np.ndarray) -> str:
    data = config["data"]
    strategy = config["model"].get("trainability_strategy", "full_finetune")
    dropout = config["model"].get("dropout")
    train = config["train"]
    aug = config.get("augmentation", {})
    fold_rows = "\n".join(
        f"| {int(row.fold)} | {row.macro_auc:.4f} | {row.accuracy:.4f} | {row.macro_f1:.4f} | {row.balanced_accuracy:.4f} | {int(row.best_epoch)} |"
        for row in fold_metrics.itertuples()
    )
    metric_rows = "\n".join(f"| {name} | {metrics[name]:.4f} |" for name in METRICS)
    return f"""# R3DPR ResNet18 Control vs Patient Binary Baseline: ColorJitter + OneCycleLR

## Protocol

- Direct binary label column: `{data['label_column']}`. No three-class label mapping was used.
- One-table five-fold split: training fold != k; validation fold == k.
- Input resize: height x width = {data['image_height']} x {data['image_width']}.
- Training-only augmentation: horizontal flip={bool(aug.get('horizontal_flip', True))}, brightness={float(aug.get('brightness', 0.05))}, contrast={float(aug.get('contrast', 0.05))}. Validation and OOF evaluation remain deterministic.
- ResNet18 ImageNet pretrained, `{strategy}`, dropout={dropout}, BatchNorm mode=`{train.get('batchnorm_mode', 'train')}`, weighted cross entropy calculated within each training fold.
- AdamW (lr={train['lr']}, weight_decay={train['weight_decay']}), OneCycleLR(max_lr={float(train.get('onecycle_max_lr', train['lr']))}, div_factor={float(train.get('onecycle_div_factor', 10.0))}, final_div_factor={float(train.get('onecycle_final_div_factor', 100.0))}, pct_start={float(train.get('onecycle_pct_start', 0.3))}, cycle_momentum={bool(train.get('onecycle_cycle_momentum', False))}), max_epochs={int(train.get('max_epochs', train.get('epochs', 50)))}.
- No early stopping was used. Each fold trained for the full budget, and the best checkpoint was chosen by validation macro-AUC.
- Hard labels are softmax argmax. No threshold search was performed.

## Pooled OOF Metrics

| Metric | Value |
|---|---:|
{metric_rows}

## Pooled OOF Confusion Matrix

| True class | Predicted Control | Predicted Patient |
|---|---:|---:|
| Control | {int(matrix[0, 0])} | {int(matrix[0, 1])} |
| Patient | {int(matrix[1, 0])} | {int(matrix[1, 1])} |

## Fold Metrics

| Fold | Macro-AUC | Accuracy | Macro-F1 | Balanced Accuracy | Best epoch |
|---:|---:|---:|---:|---:|---:|
{fold_rows}

The primary result is the pooled OOF metric set, not the best individual fold or training-set metric.
"""


def main() -> None:
    args = parse_args()
    experiment_dir = args.experiment_dir if args.experiment_dir.is_absolute() else PROJECT_ROOT / args.experiment_dir
    config = load_yaml(experiment_dir / "config_snapshot.yaml")
    n_folds = int(config["data"]["n_folds"])
    prediction_frames: list[pd.DataFrame] = []
    metric_frames: list[pd.DataFrame] = []
    for fold in range(n_folds):
        fold_dir = experiment_dir / f"fold_{fold}"
        prediction_path = fold_dir / "val_predictions.csv"
        metric_path = fold_dir / "metrics.csv"
        if not prediction_path.is_file() or not metric_path.is_file():
            raise FileNotFoundError(f"Missing completed artifacts for fold {fold}: {fold_dir}")
        prediction_frames.append(pd.read_csv(prediction_path, dtype={"sample_id": "string", "patient_group_id": "string"}))
        metric_frames.append(pd.read_csv(metric_path))
    oof = pd.concat(prediction_frames, ignore_index=True)
    expected = expected_table(config)
    validate_oof(oof, expected, n_folds)
    metrics = compute_binary_metrics(oof["binary_label"].to_numpy(int), oof[["prob_control", "prob_patient"]].to_numpy(float))
    fold_metrics = pd.concat(metric_frames, ignore_index=True).sort_values("fold")
    oof.to_csv(experiment_dir / "oof_predictions.csv", index=False, encoding="utf-8-sig")
    fold_metrics.to_csv(experiment_dir / "fold_metrics.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame([flatten_metrics(metrics)]).to_csv(experiment_dir / "oof_metrics.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(metrics["confusion_matrix"], index=["control", "patient"], columns=["control", "patient"]).to_csv(
        experiment_dir / "oof_confusion_matrix.csv", index_label="true\\pred", encoding="utf-8-sig"
    )
    (experiment_dir / "binary_experiment_results.md").write_text(make_report(config, fold_metrics, metrics, metrics["confusion_matrix"]), encoding="utf-8")
    print(f"OOF_SUMMARY_COMPLETED={experiment_dir}")


if __name__ == "__main__":
    main()
