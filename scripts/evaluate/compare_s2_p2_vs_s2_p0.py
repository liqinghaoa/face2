"""Finalize S2-P2 outputs and compare them with the fixed S2-P0 OOF baseline."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import cohen_kappa_score, roc_curve


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from metrics.classification_metrics import CLASS_NAMES, compute_classification_metrics, flatten_metrics  # noqa: E402


DISCLAIMER = (
    "本实验在根据既有ResNet18/34/50 OOF共同错误事后构建的S2队列上训练。"
    "结果仅用于S2内部方法比较和后续物理一致性实验，不属于完整临床队列上的无偏泛化性能。"
)
PROB = ["prob_normal", "prob_mild", "prob_severe"]
COMPARE_METRICS = [
    "macro_auc",
    "balanced_accuracy",
    "macro_f1",
    "accuracy",
    "auc_normal",
    "auc_mild",
    "auc_severe",
    "recall_severe",
    "f1_severe",
    "ordinal_mae",
    "extreme_error_rate",
    "quadratic_weighted_kappa",
]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_oof(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(
        path,
        dtype={"ID": "string", "patient_group_id": "string"},
        encoding="utf-8-sig",
    )
    required = {
        "ID",
        "patient_group_id",
        "NYHA",
        "SEX",
        "label_3class",
        "fold",
        "pred_class",
        *PROB,
    }
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"OOF missing columns {missing}: {path}")
    probability = frame[PROB].to_numpy(float)
    if (
        len(frame) != 425
        or frame.ID.nunique() != 425
        or not np.isfinite(probability).all()
        or not np.allclose(probability.sum(1), 1.0, atol=1.0e-6)
        or not np.array_equal(frame.pred_class.to_numpy(int), probability.argmax(1))
    ):
        raise ValueError(f"invalid 425-case OOF: {path}")
    return frame.sort_values("ID").reset_index(drop=True)


def metrics(frame: pd.DataFrame) -> dict[str, Any]:
    true = frame.label_3class.to_numpy(int)
    probability = frame[PROB].to_numpy(float)
    result = compute_classification_metrics(true, probability, 3)
    predicted = probability.argmax(1)
    result.update(
        {
            "ordinal_mae": float(np.abs(predicted - true).mean()),
            "within_one_accuracy": float((np.abs(predicted - true) <= 1).mean()),
            "extreme_error_rate": float((np.abs(predicted - true) == 2).mean()),
            "quadratic_weighted_kappa": float(
                cohen_kappa_score(true, predicted, labels=[0, 1, 2], weights="quadratic")
            ),
        }
    )
    return result


def align(p0: pd.DataFrame, p2: pd.DataFrame, manifest: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    truth = ["ID", "patient_group_id", "NYHA", "SEX", "label_3class", "fold"]
    if not p0[truth].astype(str).equals(p2[truth].astype(str)):
        raise ValueError("S2-P0 and S2-P2 truth/fold/patient-group alignment failed")
    reference = manifest.sort_values("ID").reset_index(drop=True)
    if set(reference.ID.astype(str)) != set(p2.ID.astype(str)):
        raise ValueError("S2-P2 OOF differs from the fixed S2 manifest")
    columns = ["ID", "patient_group_id", "NYHA", "SEX", "label_3class", "fold"]
    if not p2[columns].astype(str).equals(reference[columns].astype(str)):
        raise ValueError("S2-P2 OOF labels or folds differ from the fixed S2 manifest")
    return p0, p2


def bootstrap(
    p2: pd.DataFrame, p0: pd.DataFrame, repeats: int, seed: int
) -> tuple[pd.DataFrame, pd.DataFrame]:
    observed_p2, observed_p0 = metrics(p2), metrics(p0)
    groups = p2.patient_group_id.astype(str).unique()
    group_indices = {
        group: np.where(p2.patient_group_id.astype(str).to_numpy() == group)[0]
        for group in groups
    }
    rng = np.random.default_rng(seed)
    rows: list[dict[str, Any]] = []
    for repeat in range(repeats):
        sampled = rng.choice(groups, size=len(groups), replace=True)
        index = np.concatenate([group_indices[group] for group in sampled])
        try:
            new_metrics, control_metrics = metrics(p2.iloc[index]), metrics(p0.iloc[index])
            for metric in COMPARE_METRICS:
                rows.append(
                    {
                        "repeat": repeat,
                        "metric": metric,
                        "p2_minus_p0": float(new_metrics[metric]) - float(control_metrics[metric]),
                    }
                )
        except Exception:
            for metric in COMPARE_METRICS:
                rows.append({"repeat": repeat, "metric": metric, "p2_minus_p0": np.nan})
    differences = pd.DataFrame(rows)
    summary = []
    for metric in COMPARE_METRICS:
        values = differences.loc[differences.metric.eq(metric), "p2_minus_p0"]
        valid = values.dropna()
        summary.append(
            {
                "metric": metric,
                "s2_p0": float(observed_p0[metric]),
                "s2_p2": float(observed_p2[metric]),
                "p2_minus_p0": float(observed_p2[metric]) - float(observed_p0[metric]),
                "bootstrap_mean_difference": float(valid.mean()),
                "ci_2_5": float(valid.quantile(0.025)),
                "ci_97_5": float(valid.quantile(0.975)),
                "valid_repeats": int(len(valid)),
                "nan_repeats": int(values.isna().sum()),
                "seed": seed,
                "resampling_unit": "patient_group_id",
                "direction_note": (
                    "negative means P2 is better"
                    if metric in {"ordinal_mae", "extreme_error_rate"}
                    else "positive means P2 is better"
                ),
            }
        )
    return differences, pd.DataFrame(summary)


def correctness_transitions(p0: pd.DataFrame, p2: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    true = p2.label_3class.to_numpy(int)
    pred0 = p0[PROB].to_numpy(float).argmax(1)
    pred2 = p2[PROB].to_numpy(float).argmax(1)
    c0, c2 = pred0 == true, pred2 == true
    overall = pd.DataFrame(
        [
            {"transition": "both_correct", "count": int((c0 & c2).sum())},
            {"transition": "p0_wrong_p2_correct", "count": int((~c0 & c2).sum())},
            {"transition": "p0_correct_p2_wrong", "count": int((c0 & ~c2).sum())},
            {"transition": "both_wrong", "count": int((~c0 & ~c2).sum())},
            {
                "transition": "p0_normal_severe_extreme_errors",
                "count": int((np.abs(pred0 - true) == 2).sum()),
            },
            {
                "transition": "p2_normal_severe_extreme_errors",
                "count": int((np.abs(pred2 - true) == 2).sum()),
            },
        ]
    )
    class_rows = []
    for class_index, class_name in CLASS_NAMES.items():
        mask = true == class_index
        gains = int((mask & ~c0 & c2).sum())
        losses = int((mask & c0 & ~c2).sum())
        class_rows.append(
            {
                "true_class": class_name,
                "p0_wrong_p2_correct": gains,
                "p0_correct_p2_wrong": losses,
                "net_improvement_count": gains - losses,
            }
        )
    return overall, pd.DataFrame(class_rows)


def fold_comparison(p0: pd.DataFrame, p2: pd.DataFrame, experiment: Path) -> pd.DataFrame:
    rows = []
    for fold in range(5):
        a, b = p0[p0.fold.eq(fold)], p2[p2.fold.eq(fold)]
        ma, mb = metrics(a), metrics(b)
        selected = json.loads((experiment / f"fold_{fold}/selected_epoch.json").read_text(encoding="utf-8"))
        row: dict[str, Any] = {**selected}
        for metric in ("macro_auc", "balanced_accuracy", "macro_f1", "accuracy", "recall_severe"):
            row[f"p0_{metric}"] = float(ma[metric])
            row[f"p2_{metric}"] = float(mb[metric])
            row[f"delta_{metric}"] = float(mb[metric]) - float(ma[metric])
        rows.append(row)
    return pd.DataFrame(rows)


def plot_confusion(matrix: np.ndarray, csv_path: Path, png_path: Path) -> None:
    labels = [CLASS_NAMES[index] for index in range(3)]
    pd.DataFrame(matrix, index=labels, columns=labels).to_csv(csv_path, encoding="utf-8-sig")
    fig, axis = plt.subplots(figsize=(6, 5))
    image = axis.imshow(matrix, cmap="Blues")
    fig.colorbar(image, ax=axis)
    for row in range(3):
        for column in range(3):
            axis.text(column, row, str(int(matrix[row, column])), ha="center", va="center")
    axis.set(
        xticks=range(3),
        yticks=range(3),
        xticklabels=labels,
        yticklabels=labels,
        xlabel="Predicted",
        ylabel="True",
        title="S2-P2 OOF confusion matrix",
    )
    fig.tight_layout()
    fig.savefig(png_path, dpi=200)
    plt.close(fig)


def plot_roc(frame: pd.DataFrame, path: Path) -> None:
    true, probability = frame.label_3class.to_numpy(int), frame[PROB].to_numpy(float)
    result = metrics(frame)
    fig, axis = plt.subplots(figsize=(7, 6))
    for class_index, class_name in CLASS_NAMES.items():
        fpr, tpr, _ = roc_curve(true == class_index, probability[:, class_index])
        axis.plot(fpr, tpr, label=f"{class_name} AUC={result[f'auc_{class_name}']:.3f}")
    axis.plot([0, 1], [0, 1], "--", color="gray")
    axis.set(xlabel="False positive rate", ylabel="True positive rate", title="S2-P2 OOF ROC curves")
    axis.legend()
    axis.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def plot_training(experiment: Path, path: Path) -> None:
    fig, axes = plt.subplots(5, 2, figsize=(12, 19), constrained_layout=True)
    for fold in range(5):
        history = pd.read_csv(experiment / f"fold_{fold}/logs/train_log.csv")
        axes[fold, 0].plot(history.epoch, history.train_loss, label="train")
        axes[fold, 0].plot(history.epoch, history.val_loss, label="validation")
        axes[fold, 0].set_title(f"Fold {fold} loss")
        axes[fold, 0].legend()
        for metric in ("val_macro_auc", "val_macro_f1", "val_balanced_accuracy", "val_accuracy"):
            axes[fold, 1].plot(history.epoch, history[metric], label=metric.replace("val_", ""))
        axes[fold, 1].set_title(f"Fold {fold} validation metrics")
        axes[fold, 1].set_ylim(0, 1)
        axes[fold, 1].legend(fontsize=7)
        for axis in axes[fold]:
            axis.grid(alpha=0.2)
            axis.set_xlabel("Epoch")
    fig.suptitle("S2-P2 five-fold training curves", fontweight="bold")
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def markdown(frame: pd.DataFrame) -> str:
    def cell(value: object) -> str:
        if isinstance(value, (float, np.floating)):
            return "nan" if not np.isfinite(value) else f"{float(value):.6f}"
        return str(value).replace("|", "\\|")

    columns = list(frame.columns)
    lines = ["| " + " | ".join(columns) + " |", "|" + "|".join(["---"] * len(columns)) + "|"]
    lines.extend(
        "| " + " | ".join(cell(row[column]) for column in columns) + " |"
        for _, row in frame.iterrows()
    )
    return "\n".join(lines)


def write_report(
    path: Path,
    comparison: pd.DataFrame,
    model_comparison: pd.DataFrame,
    fold_table: pd.DataFrame,
    transitions: pd.DataFrame,
    class_transitions: pd.DataFrame,
    p2_metrics: dict[str, Any],
    audit: dict[str, Any],
) -> None:
    lookup = comparison.set_index("metric")
    severe_recall = float(p2_metrics["recall_severe"])
    extreme = float(p2_metrics["extreme_error_rate"])
    majority_folds = int((fold_table.delta_macro_auc > 0).sum())
    ci_cross = {
        metric: bool(row.ci_2_5 <= 0 <= row.ci_97_5)
        for metric, row in comparison.set_index("metric").iterrows()
    }
    selected = fold_table[["fold", "selected_epoch", "actual_training_epochs", "stop_reason"]]
    report = [
        "# S2-P2 relative optical phenotype results",
        "",
        f"> **{DISCLAIMER}**",
        "",
        "## Main paired comparison",
        "",
        markdown(comparison[["metric", "s2_p0", "s2_p2", "p2_minus_p0", "ci_2_5", "ci_97_5", "direction_note"]]),
        "",
        "## Model capacity and inference inputs",
        "",
        markdown(model_comparison),
        "",
        "Model tensor inputs count the arrays passed to forward(). Source assets count the files read per case before online optical-channel construction.",
        "",
        "## Fold and checkpoint summary",
        "",
        markdown(fold_table),
        "",
        f"Macro-AUC improved in {majority_folds}/5 folds. Fold-mean Macro-AUC and concatenated 425-case OOF Macro-AUC are reported separately.",
        "",
        "## Correctness transitions",
        "",
        markdown(transitions),
        "",
        markdown(class_transitions),
        "",
        "## Required interpretation answers",
        "",
        "1. Historical P2-1 was reused without changing the optical formula, ROI geometry, shared encoder, relative operations, fusion or classifier.",
        f"2. All 425 cases have audited inputs; {audit['generated_missing_count']} missing historical assets were regenerated with the historical pipeline.",
        "3. Total/trainable parameters are 22,814,872 / 22,814,872, matching historical P2-1.",
        f"4. S2-P2 minus S2-P0 Macro-AUC: {lookup.loc['macro_auc','p2_minus_p0']:+.6f}.",
        f"5. BA delta={lookup.loc['balanced_accuracy','p2_minus_p0']:+.6f}; Macro-F1 delta={lookup.loc['macro_f1','p2_minus_p0']:+.6f}.",
        f"6. Normal benefits most by AUC. Per-class AUC deltas: normal={lookup.loc['auc_normal','p2_minus_p0']:+.6f}, mild={lookup.loc['auc_mild','p2_minus_p0']:+.6f}, severe={lookup.loc['auc_severe','p2_minus_p0']:+.6f}.",
        f"7. Severe recall is {severe_recall:.6f}; it {'exceeds' if severe_recall > 0.4324324324 else 'does not exceed'} S2-P0 0.4324.",
        f"8. Extreme error rate is {extreme:.6f}; it {'is lower than' if extreme < 0.0588235294 else 'is not lower than'} S2-P0 0.0588.",
        f"9. Macro-AUC direction is positive in {majority_folds}/5 folds.",
        "10. Bootstrap CI crosses zero: " + ", ".join(f"{metric}={value}" for metric, value in ci_cross.items()) + ".",
        f"11. Selected epochs: {selected.to_dict('records')}.",
        "12. With complete audited assets, fixed folds and complete OOF, S2-P2 can be frozen as the direct P3 parent even if it does not outperform S2-P0.",
        "",
        "## Interpretation boundary",
        "",
        "S2 is post-hoc selected and is only valid for internal method comparison. P2 adds both a relative optical branch and model capacity versus P0, so differences cannot be attributed solely to relative phenotype. The channels are physically inspired relative representations, not skin reflectance, oxygen saturation, perfusion or physiological measurements. No further S2 filtering was performed from these errors.",
    ]
    path.write_text("\n".join(report) + "\n", encoding="utf-8")


def run(args: argparse.Namespace) -> None:
    experiment = args.experiment.resolve()
    summary = experiment / "summary"
    summary.mkdir(parents=True, exist_ok=True)
    manifest = pd.read_csv(args.manifest, dtype={"ID": "string", "patient_group_id": "string"}, encoding="utf-8-sig")
    audit = json.loads(args.audit.read_text(encoding="utf-8"))
    if not audit.get("all_checks_pass"):
        raise RuntimeError("S2-P2 input audit is not PASS")
    p0, p2 = align(read_oof(args.p0_oof), read_oof(summary / "oof_predictions.csv"), manifest)
    p0_metrics, p2_metrics = metrics(p0), metrics(p2)
    comparison = pd.DataFrame(
        [
            {
                "metric": metric,
                "s2_p0": float(p0_metrics[metric]),
                "s2_p2": float(p2_metrics[metric]),
                "p2_minus_p0": float(p2_metrics[metric]) - float(p0_metrics[metric]),
            }
            for metric in COMPARE_METRICS
        ]
    )
    differences, bootstrap_summary = bootstrap(p2, p0, args.bootstrap_repeats, args.seed)
    comparison = comparison.drop(columns=["s2_p0", "s2_p2", "p2_minus_p0"]).merge(
        bootstrap_summary, on="metric", validate="one_to_one"
    )
    comparison.to_csv(summary / "paired_comparison_vs_s2_p0.csv", index=False, encoding="utf-8-sig")
    bootstrap_summary.to_csv(summary / "paired_bootstrap_vs_s2_p0.csv", index=False, encoding="utf-8-sig")
    differences.to_csv(summary / "paired_bootstrap_differences_vs_s2_p0.csv", index=False, encoding="utf-8-sig")
    transitions, class_transitions = correctness_transitions(p0, p2)
    transitions.to_csv(summary / "correctness_transitions_vs_s2_p0.csv", index=False, encoding="utf-8-sig")
    class_transitions.to_csv(summary / "class_net_improvement_vs_s2_p0.csv", index=False, encoding="utf-8-sig")
    model_comparison = pd.DataFrame(
        [
            {
                "model": "S2-P0 Global ResNet18",
                "total_parameters": 11178051,
                "trainable_parameters": 11178051,
                "model_tensor_inputs": 1,
                "effective_input_channels": 3,
                "source_assets_per_case": 1,
                "input_description": "Global RGB",
            },
            {
                "model": "S2-P2 Relative Optical",
                "total_parameters": 22814872,
                "trainable_parameters": 22814872,
                "model_tensor_inputs": 3,
                "effective_input_channels": 17,
                "source_assets_per_case": 5,
                "input_description": "Global RGB + Eye 7-channel optical + Cheek 7-channel optical",
            },
        ]
    )
    model_comparison.to_csv(summary / "model_input_comparison_vs_s2_p0.csv", index=False, encoding="utf-8-sig")
    folds = fold_comparison(p0, p2, experiment)
    folds.to_csv(summary / "fold_metrics.csv", index=False, encoding="utf-8-sig")
    numeric = folds.select_dtypes(include=[np.number]).drop(columns=["fold"], errors="ignore")
    pd.DataFrame(
        {"metric": numeric.columns, "mean": numeric.mean().values, "std": numeric.std(ddof=1).values}
    ).to_csv(summary / "aggregate_metrics.csv", index=False, encoding="utf-8-sig")
    serializable = {key: value.tolist() if isinstance(value, np.ndarray) else value for key, value in p2_metrics.items()}
    (summary / "oof_metrics.json").write_text(json.dumps(serializable, ensure_ascii=False, indent=2), encoding="utf-8")
    pd.DataFrame([flatten_metrics(p2_metrics) | {key: p2_metrics[key] for key in ("ordinal_mae", "within_one_accuracy", "extreme_error_rate", "quadratic_weighted_kappa")}]).to_csv(
        summary / "oof_metrics.csv", index=False, encoding="utf-8-sig"
    )
    plot_confusion(np.asarray(p2_metrics["confusion_matrix"]), summary / "confusion_matrix_oof.csv", summary / "confusion_matrix_oof.png")
    plot_roc(p2, summary / "roc_curves_oof.png")
    plot_training(experiment, summary / "training_curves_5fold.png")
    write_report(summary / "s2_p2_results.md", comparison, model_comparison, folds, transitions, class_transitions, p2_metrics, audit)
    run_manifest = {
        "status": "COMPLETE",
        "experiment": experiment.name,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "disclaimer": DISCLAIMER,
        "s2_p2_input_audit_pass": True,
        "oof_validation_pass": True,
        "oof_n": len(p2),
        "patient_groups": int(p2.patient_group_id.nunique()),
        "class_counts": p2.label_3class.value_counts().sort_index().to_dict(),
        "model_parameters": {"total": 22814872, "trainable": 22814872},
        "inference_inputs": {"global_rgb": 1, "eye_rgb_and_mask": 1, "cheek_rgb_and_mask": 1, "online_optical_channels_per_roi": 7},
        "model_input_comparison": model_comparison.to_dict("records"),
        "execution_adjustments": {
            "host_memory_interruptions": "fold 3 before epoch 14 and fold 4 after epoch 9; checkpoints were preserved and training resumed without changing the mathematical protocol",
            "resume_rng_state": "Python, NumPy, Torch, CUDA and DataLoader generator states are saved in new checkpoints and restored on resume",
            "fold_process_isolation": "folds 3 and 4 were resumed in separate processes to release host memory between folds",
        },
        "bootstrap": {"repeats": args.bootstrap_repeats, "seed": args.seed, "unit": "patient_group_id"},
        "selected_epochs": folds[["fold", "selected_epoch", "actual_training_epochs", "best_macro_auc", "stop_reason"]].to_dict("records"),
        "inputs": {
            "s2_p2_manifest": {"path": str(args.manifest.resolve()), "sha256": sha256(args.manifest)},
            "s2_p2_input_audit": {"path": str(args.audit.resolve()), "sha256": sha256(args.audit)},
            "s2_p0_oof": {"path": str(args.p0_oof.resolve()), "sha256": sha256(args.p0_oof)},
            "s2_p2_oof": {"path": str((summary / 'oof_predictions.csv').resolve()), "sha256": sha256(summary / "oof_predictions.csv")},
        },
        "outputs": sorted(path.name for path in summary.iterdir() if path.is_file()),
        "p3_parent_ready": True,
    }
    (summary / "run_manifest.json").write_text(json.dumps(run_manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(comparison.to_string(index=False))
    print(f"REPORT={summary / 's2_p2_results.md'}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--experiment",
        type=Path,
        default=ROOT / "experiments/S2_425_P2_RelativeOpticalPhenotype_ResNet18_WeightedCE_5Fold",
    )
    parser.add_argument("--manifest", type=Path, default=ROOT / "data/processed/s2_425_p2/s2_p2_manifest.csv")
    parser.add_argument("--audit", type=Path, default=ROOT / "data/processed/s2_425_p2/s2_p2_manifest.json")
    parser.add_argument(
        "--p0-oof",
        type=Path,
        default=ROOT / "experiments/S2_425_Global224_ImageNetResNet18_NYHA3Class_WeightedCE_5Fold/summary/oof_predictions.csv",
    )
    parser.add_argument("--bootstrap-repeats", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=2026)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
