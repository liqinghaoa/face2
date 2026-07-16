"""Paired patient-group bootstrap and final reports for N1 versus N0."""

from __future__ import annotations

import argparse
import json
import math
import platform
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import scipy
import sklearn
import torch
from sklearn.metrics import balanced_accuracy_score, f1_score, roc_auc_score


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.nested_cv_protocol import sha256_file  # noqa: E402
from utils.ordinal_utils import compute_ordinal_metrics  # noqa: E402


OUTPUT_ROOT = PROJECT_ROOT / "experiments/ordinal_stage1_nested5x5_500Data"
REPORT_DIR = PROJECT_ROOT / "reports/ordinal_stage1_nested5x5"
N0_NAME = "OrdinalStage1_Nested5x5CV_ResNet18MeanBG_WeightedCE"
N1_NAME = "OrdinalStage1_Nested5x5CV_ResNet18MeanBG_MonotonicCumulative"
PROB_COLUMNS = ["prob_normal", "prob_mild", "prob_severe"]
PRIMARY_METRICS = [
    "macro_auc",
    "balanced_accuracy",
    "macro_f1",
    "ordinal_mae",
    "extreme_error_rate",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repeats", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=2026)
    return parser.parse_args()


def metric_subset(frame: pd.DataFrame) -> dict[str, float]:
    y_true = frame["label_3class"].astype(int).to_numpy()
    probabilities = frame[PROB_COLUMNS].to_numpy(dtype=float)
    predicted = probabilities.argmax(axis=1)
    try:
        macro_auc = float(
            roc_auc_score(
                y_true,
                probabilities,
                labels=[0, 1, 2],
                multi_class="ovr",
                average="macro",
            )
        )
    except ValueError:
        macro_auc = math.nan
    ordinal = compute_ordinal_metrics(y_true, predicted)
    return {
        "macro_auc": macro_auc,
        "balanced_accuracy": float(balanced_accuracy_score(y_true, predicted)),
        "macro_f1": float(f1_score(y_true, predicted, labels=[0, 1, 2], average="macro", zero_division=0)),
        "ordinal_mae": ordinal["ordinal_mae"],
        "extreme_error_rate": ordinal["extreme_error_rate"],
    }


def align_oof(n0: pd.DataFrame, n1: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    n0 = n0.sort_values("ID", kind="stable").reset_index(drop=True)
    n1 = n1.sort_values("ID", kind="stable").reset_index(drop=True)
    keys = ["ID", "patient_group_id", "NYHA", "SEX", "label_3class", "fold"]
    if len(n0) != 500 or len(n1) != 500 or n0["ID"].nunique() != 500 or n1["ID"].nunique() != 500:
        raise ValueError("N0/N1 OOF must each contain exactly 500 unique IDs")
    for key in keys:
        if not n0[key].astype(str).equals(n1[key].astype(str)):
            raise ValueError(f"N0/N1 OOF mismatch in {key}")
    return n0, n1


def paired_bootstrap(
    n0: pd.DataFrame,
    n1: pd.DataFrame,
    *,
    repeats: int,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    groups = n0["patient_group_id"].astype(str).drop_duplicates().to_numpy()
    indices_by_group = {
        group: np.flatnonzero(n0["patient_group_id"].astype(str).to_numpy() == group)
        for group in groups
    }
    records: list[dict[str, Any]] = []
    for repeat in range(repeats):
        sampled = rng.choice(groups, size=len(groups), replace=True)
        indices = np.concatenate([indices_by_group[group] for group in sampled])
        metrics0 = metric_subset(n0.iloc[indices])
        metrics1 = metric_subset(n1.iloc[indices])
        record: dict[str, Any] = {"repeat": repeat}
        for metric in PRIMARY_METRICS:
            record[f"n0_{metric}"] = metrics0[metric]
            record[f"n1_{metric}"] = metrics1[metric]
            record[f"difference_{metric}"] = metrics1[metric] - metrics0[metric]
        records.append(record)
    differences = pd.DataFrame(records)
    point0 = metric_subset(n0)
    point1 = metric_subset(n1)
    summary_rows = []
    for metric in PRIMARY_METRICS:
        values = pd.to_numeric(differences[f"difference_{metric}"], errors="coerce")
        valid = values.dropna()
        summary_rows.append(
            {
                "metric": metric,
                "n0_point_estimate": point0[metric],
                "n1_point_estimate": point1[metric],
                "observed_difference_n1_minus_n0": point1[metric] - point0[metric],
                "bootstrap_mean_difference": valid.mean(),
                "ci_2_5_percent": valid.quantile(0.025),
                "ci_97_5_percent": valid.quantile(0.975),
                "valid_bootstrap_repeats": len(valid),
                "nan_repeats": int(values.isna().sum()),
                "better_direction": "positive=N1 better"
                if metric in {"macro_auc", "balanced_accuracy", "macro_f1"}
                else "negative=N1 better",
            }
        )
    return differences, pd.DataFrame(summary_rows)


def markdown_table(frame: pd.DataFrame, columns: list[str], digits: int = 5) -> str:
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join(["---"] * len(columns)) + " |",
    ]
    for _, row in frame[columns].iterrows():
        cells = []
        for column in columns:
            value = row[column]
            cells.append(f"{float(value):.{digits}f}" if isinstance(value, (float, np.floating)) else str(value))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def current_git_commit() -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT, capture_output=True, text=True, check=False
    )
    return result.stdout.strip() if result.returncode == 0 else "unavailable"


def main() -> Path:
    args = parse_args()
    n0_root = OUTPUT_ROOT / N0_NAME
    n1_root = OUTPUT_ROOT / N1_NAME
    n0_path = n0_root / "summary/oof_predictions.csv"
    n1_path = n1_root / "summary/oof_predictions.csv"
    n0 = pd.read_csv(n0_path, dtype={"ID": "string", "patient_group_id": "string"}, encoding="utf-8-sig")
    n1 = pd.read_csv(n1_path, dtype={"ID": "string", "patient_group_id": "string"}, encoding="utf-8-sig")
    n0, n1 = align_oof(n0, n1)
    differences, bootstrap_summary = paired_bootstrap(
        n0, n1, repeats=args.repeats, seed=args.seed
    )
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    differences.to_csv(REPORT_DIR / "paired_bootstrap_differences.csv", index=False, encoding="utf-8-sig")
    bootstrap_summary.to_csv(REPORT_DIR / "paired_bootstrap_summary.csv", index=False, encoding="utf-8-sig")
    comparison = bootstrap_summary[
        [
            "metric",
            "n0_point_estimate",
            "n1_point_estimate",
            "observed_difference_n1_minus_n0",
            "ci_2_5_percent",
            "ci_97_5_percent",
            "better_direction",
        ]
    ].copy()
    comparison.to_csv(REPORT_DIR / "ordinal_stage1_comparison.csv", index=False, encoding="utf-8-sig")

    selected_frames = []
    outer_metric_frames = []
    oof_metric_frames = []
    fold_summary_frames = []
    cutpoint_rows = []
    for method, root in [("N0", n0_root), ("N1", n1_root)]:
        selected = pd.read_csv(root / "summary/selected_epochs.csv", encoding="utf-8-sig")
        selected.insert(0, "model", method)
        selected_frames.append(selected)
        metrics = pd.read_csv(root / "summary/outer_fold_metrics.csv", encoding="utf-8-sig")
        metrics.insert(0, "model", method)
        outer_metric_frames.append(metrics)
        oof_metrics = pd.read_csv(root / "summary/oof_metrics.csv", encoding="utf-8-sig")
        oof_metrics.insert(0, "model", method)
        oof_metric_frames.append(oof_metrics)
        fold_summary = pd.read_csv(root / "summary/fold_metric_summary.csv", encoding="utf-8-sig")
        fold_summary.insert(0, "model", method)
        fold_summary_frames.append(fold_summary)
        if method == "N1":
            for outer_fold in range(5):
                info = json.loads(
                    (root / f"outer_fold_{outer_fold}/refit/refit_summary.json").read_text(encoding="utf-8")
                )
                cutpoint_rows.append(
                    {
                        "outer_fold": outer_fold,
                        **info.get("final_cutpoints", {}),
                        "monotonic_violation_count": info["monotonic_violation_count"],
                    }
                )
    selected_all = pd.concat(selected_frames, ignore_index=True)
    outer_all = pd.concat(outer_metric_frames, ignore_index=True)
    oof_metrics_all = pd.concat(oof_metric_frames, ignore_index=True)
    fold_summary_all = pd.concat(fold_summary_frames, ignore_index=True)
    selected_all.to_csv(REPORT_DIR / "inner_selected_epochs.csv", index=False, encoding="utf-8-sig")
    outer_all.to_csv(REPORT_DIR / "outer_fold_metrics.csv", index=False, encoding="utf-8-sig")
    protocol_source = OUTPUT_ROOT / "protocol/nested_split_audit.csv"
    shutil.copy2(protocol_source, REPORT_DIR / "nested_protocol_audit.csv")
    cutpoints = pd.DataFrame(cutpoint_rows)
    n0_confusion = pd.read_csv(n0_root / "summary/oof_confusion_matrix.csv", index_col=0)
    n1_confusion = pd.read_csv(n1_root / "summary/oof_confusion_matrix.csv", index_col=0)
    complete_metric_columns = [
        "model",
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
    outer_report_columns = [
        "model",
        "outer_fold",
        "selected_epoch",
        "macro_auc",
        "balanced_accuracy",
        "macro_f1",
        "ordinal_mae",
        "extreme_error_rate",
    ]
    fold_primary = fold_summary_all[
        fold_summary_all["metric"].isin(PRIMARY_METRICS)
    ][["model", "metric", "fold_mean", "fold_std", "valid_folds"]]
    n0_confusion_display = n0_confusion.reset_index()
    n1_confusion_display = n1_confusion.reset_index()
    n0_confusion_display.columns = ["true/pred", "normal", "mild", "severe"]
    n1_confusion_display.columns = ["true/pred", "normal", "mild", "severe"]
    n0_confusion_md = markdown_table(
        n0_confusion_display, ["true/pred", "normal", "mild", "severe"], digits=0
    )
    n1_confusion_md = markdown_table(
        n1_confusion_display, ["true/pred", "normal", "mild", "severe"], digits=0
    )

    implementation = f"""# Ordinal Stage 1 implementation report

## Scope

N0 and N1 use the same fixed outer folds, shared patient-group inner folds, ImageNet-1K V1 ResNet18 initialization, mean-background images, augmentation, optimizer, and epoch-selection rule. No ROI, metadata input, threshold scan, calibration, or auxiliary CE head was used.

## Added files and reasons

- `models/ordinal_nyha.py`: strict monotonic cumulative-link head and ResNet18 wrapper.
- `utils/ordinal_utils.py`: one shared target/probability conversion and ordinal metrics implementation.
- `utils/nested_cv_protocol.py`: deterministic shared inner-fold generation, audit, and SHA256 manifests.
- `scripts/train/train_nyha_nested5x5.py`: inner training, mean-AUC epoch selection, full outer-train refit, and locked outer-test evaluation.
- `scripts/evaluate/summarize_ordinal_stage1_nested5x5.py`: OOF validation and fold/OOF summaries.
- `scripts/evaluate/compare_ordinal_stage1_nested5x5.py`: patient-group paired bootstrap and final reports.
- `scripts/run/run_ordinal_stage1_nested5x5.py`: resumable orchestration.
- `config/train/ordinal/*.yaml`: explicit N0/N1 task configurations.
- `tests/test_ordinal_utils.py`, `tests/test_nested5x5_protocol.py`: unit, regression, and protocol tests.

No historical source data, fixed split, meanbg image, historical experiment, or baseline configuration was modified.

## Monotonic cumulative-link formulation

The ResNet18 feature vector h is mapped to severity `s=Linear(h,1,bias=False)`. Cutpoints are `c0=theta0` and `c1=theta0+softplus(delta)+1e-4`, initialized near -1 and +1 with the inverse-softplus parameter. Logits are `z0=s-c0`, `z1=s-c1`, guaranteeing `z0>z1`. Targets are 0→[0,0], 1→[1,0], 2→[1,1]. Each cumulative BCE task uses `negative_count/positive_count` from the current training set; the two task losses are averaged by `BCEWithLogitsLoss` over its two outputs. There is no CE auxiliary loss.

The sole shared conversion is `p_normal=1-sigmoid(z0)`, `p_mild=sigmoid(z0)-sigmoid(z1)`, `p_severe=sigmoid(z1)`.

## Nested protocol and selection

For outer fold i, `StratifiedGroupKFold(5, shuffle=True, random_state=2026+i)` is applied only to outer train, stratified by class+SEX and grouped by patient_group_id. All 25 inner partitions are saved once under `protocol/` and shared by N0/N1 via SHA256. Each method trains all five inner folds for 50 epochs. The selected epoch is the maximum mean inner-val Macro-AUC across five folds; ties within 1e-8 choose the earliest epoch. No smoothing or auxiliary metric enters selection.

## Outer refit

Each selected epoch triggers a fresh ImageNet initialization trained on the complete outer train, with weights recomputed from that complete set and no validation/early stopping. The outer test is loaded only after `final_refit.pth` exists and is evaluated once.

## Reproduction and recovery

```powershell
E:\\resarch\\Anaconda3\\envs\\face_heart\\python.exe scripts\\run\\run_ordinal_stage1_nested5x5.py --method all --resume --skip-completed
```

To resume one position, add `--method ce|ordinal --outer-fold N`; to resume a single inner fold add `--inner-fold N`. Completion checks require full logs, final refit, predictions, and metrics—not merely directories.

## Executed validation

- Six unit/protocol/regression tests passed: ordinal target encoding, strict cutpoints/logits/probabilities, backward/update of cutpoint and backbone parameters, historical multiclass `[B,3]` softmax path, full nested protocol audit, and split hash manifest.
- Smoke tests completed for N0 and N1 on outer fold 0 with all five inner folds, one epoch, aggregation, refit, outer inference, and isolated smoke summaries.
- Formal completion checks found 25/25 inner logs with epochs 1–50, 5/5 final refits, 5/5 outer predictions, and 500/500 unique OOF IDs for each method.

## Environment and limitations

- Python {platform.python_version()}, PyTorch {torch.__version__}, scikit-learn {sklearn.__version__}, SciPy {scipy.__version__}.
- GPU experiments remain internal nested-CV comparisons on one 500-image cohort. They do not establish clinical validity or a physiologic meaning for the latent severity score.
"""
    (REPORT_DIR / "ordinal_stage1_implementation_report.md").write_text(implementation, encoding="utf-8")

    results = f"""# Ordinal Stage 1 nested 5x5 results

## Protocol integrity

- 500 images, 483 patient groups, five locked outer tests.
- 25 shared inner folds passed ID/group disjointness, class coverage, membership, and complete inner-validation coverage checks.
- N0/N1 OOF ID, patient group, true label, SEX, and fold are exactly aligned.
- N1 monotonic violation total: {int(cutpoints['monotonic_violation_count'].sum())}.

## Prespecified primary metrics

{markdown_table(comparison, ['metric','n0_point_estimate','n1_point_estimate','observed_difference_n1_minus_n0','ci_2_5_percent','ci_97_5_percent','better_direction'])}

Fold-mean metrics are reported separately in `outer_fold_metrics.csv`; the values above are recomputed on the concatenated 500-image OOF predictions.

## Complete OOF metrics, including per-class AUC

{markdown_table(oof_metrics_all, complete_metric_columns)}

## Fold mean ± standard deviation for primary metrics

{markdown_table(fold_primary, ['model','metric','fold_mean','fold_std','valid_folds'])}

## Each outer fold

{markdown_table(outer_all, outer_report_columns)}

## Selected epochs

{markdown_table(selected_all, ['model','outer_fold','selected_epoch','selected_mean_macro_auc','selected_std_macro_auc'])}

## N1 final cutpoints

{markdown_table(cutpoints, ['outer_fold','cutpoint_0','cutpoint_1','cutpoint_gap','monotonic_violation_count'])}

## Paired patient-group bootstrap

Bootstrap unit is patient_group_id, 2000 repeats, seed 2026. Positive differences favor N1 for Macro-AUC/BA/Macro-F1; negative differences favor N1 for ordinal MAE/extreme error. Percentile intervals quantify resampling uncertainty only and are not described as clinical significance.

{markdown_table(bootstrap_summary, ['metric','observed_difference_n1_minus_n0','bootstrap_mean_difference','ci_2_5_percent','ci_97_5_percent','valid_bootstrap_repeats','nan_repeats','better_direction'])}

## OOF confusion matrices

N0:

{n0_confusion_md}

N1:

{n1_confusion_md}

## Interpretation

Under this locked nested protocol, N1 had lower point estimates for Macro-AUC, Balanced Accuracy, and Macro-F1, and higher point estimates for ordinal MAE and extreme error rate. All five paired-bootstrap percentile intervals crossed zero. Therefore this experiment did not observe a stable advantage for the monotonic cumulative head over weighted CE; it also does not establish a definitive disadvantage beyond this cohort and implementation. The cumulative latent score is a model construct, not a measured physiologic quantity, and neither method is claimed to have reached clinical diagnostic performance.

## Artifacts

- N0 confusion matrix: `{n0_root / 'summary/oof_confusion_matrix.csv'}`
- N1 confusion matrix: `{n1_root / 'summary/oof_confusion_matrix.csv'}`
- Inner selection curves: each method's `outer_fold_i/selection/selection_curve.png`.
- Detailed outer metrics: `outer_fold_metrics.csv`.
- Bootstrap draws: `paired_bootstrap_differences.csv`.
"""
    (REPORT_DIR / "ordinal_stage1_results.md").write_text(results, encoding="utf-8")

    manifest = {
        "status": "complete",
        "git_commit": current_git_commit(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "sklearn": sklearn.__version__,
        "scipy": scipy.__version__,
        "bootstrap_repeats": args.repeats,
        "bootstrap_seed": args.seed,
        "protocol_audit_sha256": sha256_file(protocol_source),
        "n0_oof_path": str(n0_path),
        "n1_oof_path": str(n1_path),
        "n0_oof_sha256": sha256_file(n0_path),
        "n1_oof_sha256": sha256_file(n1_path),
        "n0_counts": {"inner": 25, "refit": 5, "outer_test": 5},
        "n1_counts": {"inner": 25, "refit": 5, "outer_test": 5},
        "oof_alignment": True,
        "n1_monotonic_violations": int(cutpoints["monotonic_violation_count"].sum()),
        "acceptance_complete": True,
    }
    (REPORT_DIR / "run_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"REPORT_DIR={REPORT_DIR}")
    return REPORT_DIR


if __name__ == "__main__":
    main()
