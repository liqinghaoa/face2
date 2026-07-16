"""Finalize S2 ResNet18 outputs and compare original, filtered, and retrained OOF."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import shutil
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


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from losses.classification_losses import compute_class_weights  # noqa: E402
from metrics.classification_metrics import CLASS_NAMES, compute_classification_metrics, flatten_metrics  # noqa: E402
from utils.experiment_utils import load_yaml, save_yaml  # noqa: E402


SEED = 2026
DISCLAIMER = "本实验使用根据既有ResNet18/34/50 OOF共同错误事后构建的S2队列。结果用于S2内部方法比较和数据敏感性分析，不属于完整临床队列上的无偏泛化性能，也不能替代原始522例五折OOF结果。"
PROB_COLUMNS = ["prob_normal", "prob_mild", "prob_severe"]
BOOTSTRAP_METRICS = ["macro_auc", "balanced_accuracy", "macro_f1", "accuracy", "ordinal_mae", "extreme_error_rate", "quadratic_weighted_kappa"]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def save_csv(frame: pd.DataFrame, path: Path) -> None:
    frame.to_csv(path, index=False, encoding="utf-8-sig")


def metrics_from_arrays(true: np.ndarray, prob: np.ndarray) -> dict[str, Any]:
    result = compute_classification_metrics(true, prob, num_classes=3)
    predicted = prob.argmax(axis=1)
    result["ordinal_mae"] = float(np.mean(np.abs(true - predicted)))
    result["within_one_accuracy"] = float(np.mean(np.abs(true - predicted) <= 1))
    result["extreme_error_rate"] = float(np.mean(np.abs(true - predicted) == 2))
    result["quadratic_weighted_kappa"] = float(cohen_kappa_score(true, predicted, labels=[0, 1, 2], weights="quadratic"))
    return result


def frame_metrics(frame: pd.DataFrame) -> dict[str, Any]:
    return metrics_from_arrays(frame["label_3class"].to_numpy(int), frame[PROB_COLUMNS].to_numpy(float))


def validate_oof(oof: pd.DataFrame, manifest: pd.DataFrame) -> list[str]:
    errors: list[str] = []
    if len(oof) != 425 or oof["ID"].nunique() != 425:
        errors.append(f"OOF row/unique count is {len(oof)}/{oof['ID'].nunique()}, expected 425/425")
    if set(oof["ID"].astype(str)) != set(manifest["ID"].astype(str)):
        errors.append("OOF ID set differs from s2_manifest")
    merged = oof[["ID", "label_3class", "fold"]].merge(
        manifest[["ID", "label_3class", "fold"]], on="ID", suffixes=("_oof", "_manifest"), how="outer", indicator=True
    )
    bad = merged.loc[
        (merged["_merge"] != "both")
        | (merged["label_3class_oof"] != merged["label_3class_manifest"])
        | (merged["fold_oof"] != merged["fold_manifest"]), "ID"
    ].astype(str).tolist()
    if bad:
        errors.append(f"OOF label/fold mismatch IDs: {bad[:20]}")
    prob = oof[PROB_COLUMNS].apply(pd.to_numeric, errors="coerce").to_numpy(float)
    if not np.isfinite(prob).all():
        errors.append("OOF probabilities contain NaN/Inf")
    if ((prob < 0) | (prob > 1)).any() or not np.allclose(prob.sum(axis=1), 1.0, atol=1e-5):
        errors.append("OOF probabilities are illegal or do not sum to one")
    if not np.array_equal(oof["pred_class"].to_numpy(int), prob.argmax(axis=1)):
        errors.append("OOF pred_class differs from probability argmax")
    if oof["fold"].value_counts().size != 5:
        errors.append("OOF does not contain all five folds")
    return errors


def paired_bootstrap(true: np.ndarray, prob_b: np.ndarray, prob_c: np.ndarray, iterations: int) -> pd.DataFrame:
    rng = np.random.default_rng(SEED)
    observed_b = metrics_from_arrays(true, prob_b)
    observed_c = metrics_from_arrays(true, prob_c)
    values = {metric: [] for metric in BOOTSTRAP_METRICS}
    n = len(true)
    for _ in range(iterations):
        index = rng.integers(0, n, n)
        sample_true = true[index]
        if np.unique(sample_true).size < 3:
            continue
        b = metrics_from_arrays(sample_true, prob_b[index])
        c = metrics_from_arrays(sample_true, prob_c[index])
        for metric in BOOTSTRAP_METRICS:
            values[metric].append(float(c[metric]) - float(b[metric]))
    rows = []
    for metric in BOOTSTRAP_METRICS:
        low, high = np.quantile(values[metric], [0.025, 0.975])
        rows.append(
            {
                "metric": metric,
                "original_filtered_S2_B": observed_b[metric],
                "retrained_S2_C": observed_c[metric],
                "C_minus_B": float(observed_c[metric]) - float(observed_b[metric]),
                "paired_bootstrap_ci_2.5%": low,
                "paired_bootstrap_ci_97.5%": high,
                "iterations": iterations,
                "seed": SEED,
            }
        )
    return pd.DataFrame(rows)


def plot_confusion(matrix: np.ndarray, path: Path) -> None:
    labels = [CLASS_NAMES[index] for index in range(3)]
    fig, ax = plt.subplots(figsize=(6, 5))
    image = ax.imshow(matrix, cmap="Blues")
    fig.colorbar(image, ax=ax)
    for row in range(3):
        for column in range(3):
            ax.text(column, row, str(int(matrix[row, column])), ha="center", va="center")
    ax.set(xticks=range(3), yticks=range(3), xticklabels=labels, yticklabels=labels, xlabel="Predicted", ylabel="True", title="S2 retrained ResNet18 OOF confusion matrix")
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def plot_roc(oof: pd.DataFrame, path: Path) -> None:
    true = oof["label_3class"].to_numpy(int)
    prob = oof[PROB_COLUMNS].to_numpy(float)
    fig, ax = plt.subplots(figsize=(7, 6))
    for class_index in range(3):
        fpr, tpr, _ = roc_curve(true == class_index, prob[:, class_index])
        auc_value = compute_classification_metrics(true, prob)[f"auc_{CLASS_NAMES[class_index]}"]
        ax.plot(fpr, tpr, label=f"{CLASS_NAMES[class_index]} AUC={auc_value:.3f}")
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray")
    ax.set(xlabel="False positive rate", ylabel="True positive rate", title="S2 retrained ResNet18 OOF ROC curves")
    ax.legend()
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def plot_training_curves(experiment_dir: Path, path: Path) -> None:
    fig, axes = plt.subplots(5, 2, figsize=(12, 19), constrained_layout=True)
    for fold in range(5):
        history = pd.read_csv(experiment_dir / f"fold_{fold}" / "logs" / "train_log.csv")
        axes[fold, 0].plot(history["epoch"], history["train_loss"], label="train")
        axes[fold, 0].plot(history["epoch"], history["val_loss"], label="validation")
        axes[fold, 0].set_title(f"Fold {fold} loss")
        axes[fold, 0].legend()
        for metric in ("val_macro_auc", "val_macro_f1", "val_balanced_accuracy", "val_accuracy"):
            axes[fold, 1].plot(history["epoch"], history[metric], label=metric.replace("val_", ""))
        axes[fold, 1].set_title(f"Fold {fold} validation metrics")
        axes[fold, 1].set_ylim(0, 1)
        axes[fold, 1].legend(fontsize=7)
        for ax in axes[fold]:
            ax.grid(alpha=0.2)
            ax.set_xlabel("Epoch")
    fig.suptitle("S2 ResNet18 five-fold training curves", fontweight="bold")
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def selected_epoch_and_fold_outputs(experiment_dir: Path, config: dict[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame]:
    fold_rows: list[dict[str, Any]] = []
    distribution_rows: list[dict[str, Any]] = []
    split_dir = Path(config["data"]["split_dir"])
    if not split_dir.is_absolute():
        split_dir = PROJECT_ROOT / split_dir
    for fold in range(5):
        fold_dir = experiment_dir / f"fold_{fold}"
        history_path = fold_dir / "logs" / "train_log.csv"
        prediction_path = fold_dir / "predictions" / "val_predictions.csv"
        history = pd.read_csv(history_path)
        predictions = pd.read_csv(prediction_path, dtype={"ID": "string", "patient_group_id": "string"})
        best = history.loc[pd.to_numeric(history["val_macro_auc"]).idxmax()]
        last_epoch = int(history["epoch"].max())
        best_epoch = int(best["epoch"])
        stopped_early = last_epoch < int(config["train"]["epochs"])
        stop_reason = "early_stopping_patience_reached" if stopped_early else "max_epochs_reached"
        train_csv = split_dir / config["data"]["train_csv_pattern"].format(fold=fold)
        val_csv = split_dir / config["data"]["val_csv_pattern"].format(fold=fold)
        train = pd.read_csv(train_csv)
        val = pd.read_csv(val_csv)
        weights = compute_class_weights(train["label_3class"].astype(int).tolist(), 3).tolist()
        extended = frame_metrics(predictions)
        row = {"fold": fold, **flatten_metrics(extended), "ordinal_mae": extended["ordinal_mae"], "within_one_accuracy": extended["within_one_accuracy"], "extreme_error_rate": extended["extreme_error_rate"], "quadratic_weighted_kappa": extended["quadratic_weighted_kappa"], "selected_epoch": best_epoch, "actual_training_epochs": last_epoch, "best_macro_auc": float(best["val_macro_auc"]), "early_stopped": stopped_early, "stop_reason": stop_reason}
        fold_rows.append(row)

        save_yaml(config, fold_dir / "resolved_config.yaml")
        shutil.copy2(history_path, fold_dir / "train_history.csv")
        shutil.copy2(prediction_path, fold_dir / "fold_predictions.csv")
        selected = {"fold": fold, "selected_epoch": best_epoch, "actual_training_epochs": last_epoch, "best_macro_auc": float(best["val_macro_auc"]), "early_stopped": stopped_early, "stop_reason": stop_reason, "best_checkpoint": str((fold_dir / 'checkpoints' / 'best_macro_auc.pth').resolve()), "class_weights_from_training_fold_only": {CLASS_NAMES[i]: float(weights[i]) for i in range(3)}}
        (fold_dir / "selected_epoch.json").write_text(json.dumps(selected, ensure_ascii=False, indent=2), encoding="utf-8")
        (fold_dir / "fold_metrics.json").write_text(json.dumps({key: value.tolist() if isinstance(value, np.ndarray) else value for key, value in extended.items()}, ensure_ascii=False, indent=2), encoding="utf-8")

        for split_name, frame in (("train", train), ("validation", val)):
            for class_index in range(3):
                distribution_rows.append({"fold": fold, "split": split_name, "class_label": class_index, "class_name": CLASS_NAMES[class_index], "n": int((frame["label_3class"] == class_index).sum()), "weighted_ce_weight": float(weights[class_index]) if split_name == "train" else math.nan})
            for sex_name, count in frame["sex_name"].value_counts().items():
                distribution_rows.append({"fold": fold, "split": split_name, "class_label": math.nan, "class_name": f"sex:{sex_name}", "n": int(count), "weighted_ce_weight": math.nan})
    return pd.DataFrame(fold_rows), pd.DataFrame(distribution_rows)


def markdown_table(frame: pd.DataFrame) -> str:
    display = frame.copy()
    for column in display.select_dtypes(include=[np.number]).columns:
        display[column] = display[column].map(lambda value: "NA" if pd.isna(value) else f"{value:.4f}")
    lines = ["| " + " | ".join(display.columns.astype(str)) + " |", "| " + " | ".join(["---"] * len(display.columns)) + " |"]
    lines.extend("| " + " | ".join(map(str, row)) + " |" for row in display.itertuples(index=False, name=None))
    return "\n".join(lines)


def write_report(path: Path, abc: pd.DataFrame, paired: pd.DataFrame, fold_metrics: pd.DataFrame, oof_metrics: dict[str, Any], audit_manifest: dict[str, Any], paths: dict[str, Path]) -> None:
    report_metrics = ["macro_auc", "balanced_accuracy", "macro_f1", "accuracy", "ordinal_mae", "extreme_error_rate", "quadratic_weighted_kappa"]
    abc_view = abc[abc["metric"].isin(report_metrics)].copy()
    lines = [
        "# S2 425-case Global ResNet18 Weighted CE five-fold baseline",
        "",
        f"> [!CAUTION]  ",
        f"> **{DISCLAIMER}**",
        "",
        "## Data and protocol",
        "",
        f"- S2: {audit_manifest['actual_counts']['retained']} cases; normal={audit_manifest['actual_counts']['normal']}, mild={audit_manifest['actual_counts']['mild']}, severe={audit_manifest['actual_counts']['severe']}.",
        "- Original fold assignments were retained without re-randomization. Each held-out fold was used for checkpoint selection and evaluation, so this is internal five-fold OOF evaluation rather than an independent test.",
        "- Weighted CE weights were computed separately from each four-fold training set only. No sampler, downsampling, label smoothing, ordinal loss, scheduler, or AMP was added.",
        "- Operational execution adjustment: the historical config used num_workers=4, but the first launch exited before epoch 1 with a Windows system-resource error while spawning workers. Training was safely restarted with num_workers=0. This changes data-loading concurrency only, not samples, shuffle seed, augmentation, optimization, loss, model, or checkpoint selection.",
        "- The original 522 experiment OOF records show that the actual images came from the strict black-background global-face directory. The historical experiment config omitted image_root, so the S2 resolved config states that OOF-confirmed image root explicitly.",
        "",
        "## A/B/C comparison",
        "",
        markdown_table(abc_view),
        "",
        "A is the original ResNet18 trained on original four-fold training partitions and evaluated across all 522 OOF cases. B filters those original predictions to the 425 S2 cases post hoc; its training folds still contained some of the 97 excluded cases. C is retrained without any of the 97 excluded cases. B and C are not external validation results.",
        "",
        "## Paired bootstrap: C minus B on the same 425 cases",
        "",
        markdown_table(paired),
        "",
        "The paired bootstrap uses 2,000 patient-level resamples with seed 2026. Intervals describe numerical uncertainty conditional on this post-hoc S2 cohort; no independent-validation significance is claimed.",
        "",
        "## Fold training and checkpoint selection",
        "",
        markdown_table(fold_metrics[["fold", "selected_epoch", "actual_training_epochs", "best_macro_auc", "early_stopped", "stop_reason", "macro_auc", "balanced_accuracy", "macro_f1"]]),
        "",
        "## Retrained S2 OOF metrics",
        "",
        "| Metric | Value |",
        "|---|---:|",
    ]
    for metric, value in {**flatten_metrics(oof_metrics), "ordinal_mae": oof_metrics["ordinal_mae"], "within_one_accuracy": oof_metrics["within_one_accuracy"], "extreme_error_rate": oof_metrics["extreme_error_rate"], "quadratic_weighted_kappa": oof_metrics["quadratic_weighted_kappa"]}.items():
        lines.append(f"| {metric} | {float(value):.4f} |")
    lines += ["", "## Input paths", ""]
    for name, value in paths.items():
        lines.append(f"- {name}: `{value.resolve()}`")
    lines += ["", "## Use as downstream baseline", "", "If all integrity checks remain PASS, this retrained S2 ResNet18 experiment is the fixed direct baseline for later S2-P3/P4 comparisons. Those later comparisons must reuse the same S2 cases, original folds, training protocol, and reporting caveats."]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(args: argparse.Namespace) -> None:
    experiment_dir = args.experiment_dir.resolve()
    summary_dir = experiment_dir / "summary"
    summary_dir.mkdir(parents=True, exist_ok=True)
    config = load_yaml(experiment_dir / "config.yaml")
    manifest = pd.read_csv(args.s2_manifest, dtype={"ID": "string", "patient_group_id": "string"}, encoding="utf-8-sig")
    audit_manifest = json.loads(args.s2_data_manifest.read_text(encoding="utf-8"))
    if not audit_manifest.get("all_checks_pass"):
        raise RuntimeError("S2 data audit did not pass")
    retrained = pd.read_csv(summary_dir / "oof_predictions.csv", dtype={"ID": "string", "patient_group_id": "string"}, encoding="utf-8-sig")
    errors = validate_oof(retrained, manifest)
    if errors:
        raise RuntimeError("; ".join(errors))

    original = pd.read_csv(args.original_resnet18_oof, dtype={"ID": "string", "patient_group_id": "string"}, encoding="utf-8-sig")
    retained_ids = set(manifest["ID"].astype(str))
    filtered = original[original["ID"].astype(str).isin(retained_ids)].copy()
    if len(filtered) != 425:
        raise RuntimeError(f"Original filtered-to-S2 count is {len(filtered)}, expected 425")
    filtered = filtered.set_index("ID").loc[retrained["ID"].astype(str)].reset_index()
    if not np.array_equal(filtered["label_3class"].to_numpy(int), retrained["label_3class"].to_numpy(int)):
        raise RuntimeError("B/C true labels differ after ID alignment")

    metric_sets = {
        "Original-522 ResNet18": frame_metrics(original),
        "Original-ResNet18 filtered-to-S2": frame_metrics(filtered),
        "Retrained-S2 ResNet18": frame_metrics(retrained),
    }
    metric_order = ["macro_auc", "accuracy", "balanced_accuracy", "macro_precision", "macro_recall", "macro_f1", "auc_normal", "auc_mild", "auc_severe", "normal_vs_abnormal_auc", "severe_vs_rest_auc", "precision_normal", "precision_mild", "precision_severe", "recall_normal", "recall_mild", "recall_severe", "f1_normal", "f1_mild", "f1_severe", "ordinal_mae", "within_one_accuracy", "extreme_error_rate", "quadratic_weighted_kappa"]
    abc = pd.DataFrame([{"metric": metric, **{name: values[metric] for name, values in metric_sets.items()}} for metric in metric_order])
    save_csv(abc, summary_dir / "abc_comparison.csv")

    true = retrained["label_3class"].to_numpy(int)
    paired = paired_bootstrap(true, filtered[PROB_COLUMNS].to_numpy(float), retrained[PROB_COLUMNS].to_numpy(float), args.bootstrap_iterations)
    save_csv(paired, summary_dir / "paired_bootstrap_C_minus_B.csv")

    fold_metrics, distribution = selected_epoch_and_fold_outputs(experiment_dir, config)
    save_csv(fold_metrics, summary_dir / "fold_metrics.csv")
    numeric_metrics = [column for column in fold_metrics.columns if column not in {"fold", "selected_epoch", "actual_training_epochs", "early_stopped", "stop_reason"}]
    aggregate = pd.DataFrame([{"metric": metric, "mean": pd.to_numeric(fold_metrics[metric], errors="coerce").mean(), "std": pd.to_numeric(fold_metrics[metric], errors="coerce").std(ddof=1)} for metric in numeric_metrics])
    save_csv(aggregate, summary_dir / "aggregate_metrics.csv")
    save_csv(distribution, summary_dir / "class_fold_distribution.csv")

    retrained_metrics = metric_sets["Retrained-S2 ResNet18"]
    serializable_metrics = {key: value.tolist() if isinstance(value, np.ndarray) else value for key, value in retrained_metrics.items()}
    (summary_dir / "oof_metrics.json").write_text(json.dumps(serializable_metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    plot_confusion(np.asarray(retrained_metrics["confusion_matrix"]), summary_dir / "confusion_matrix_oof.png")
    plot_roc(retrained, summary_dir / "roc_curves_oof.png")
    plot_training_curves(experiment_dir, summary_dir / "training_curves_5fold.png")

    paths = {"source_label": args.label_csv, "source_master_split": args.source_master_split, "s2_manifest": args.s2_manifest, "s2_retained_ids": args.retained_ids, "s2_excluded_ids": args.excluded_ids, "image_root": Path(config["data"]["image_root"]), "original_resnet18_oof": args.original_resnet18_oof, "retrained_s2_oof": summary_dir / "oof_predictions.csv"}
    write_report(summary_dir / "s2_resnet18_results.md", abc, paired, fold_metrics, retrained_metrics, audit_manifest, paths)
    run_manifest = {
        "experiment": config["experiment"]["name"],
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "disclaimer": DISCLAIMER,
        "random_seed": SEED,
        "paired_bootstrap_iterations": args.bootstrap_iterations,
        "execution_adjustments": {
            "num_workers": "4 -> 0 after Windows worker-spawn resource failure before epoch 1; mathematical training protocol unchanged",
            "image_root": "explicitly resolved from the original 522 OOF image_path because the historical config omitted image_root",
        },
        "s2_data_audit_pass": True,
        "oof_validation_pass": True,
        "oof_n": len(retrained),
        "class_counts": retrained["label_3class"].value_counts().sort_index().to_dict(),
        "selected_epochs": fold_metrics[["fold", "selected_epoch", "actual_training_epochs", "best_macro_auc", "stop_reason"]].to_dict("records"),
        "inputs": {name: {"path": str(path.resolve()), "sha256": sha256(path.resolve()) if path.is_file() else None} for name, path in paths.items()},
        "outputs": sorted(path.name for path in summary_dir.iterdir() if path.is_file()),
    }
    (summary_dir / "run_manifest.json").write_text(json.dumps(run_manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(abc[abc["metric"].isin(BOOTSTRAP_METRICS)].to_string(index=False))
    print(paired.to_string(index=False))
    print(f"REPORT={summary_dir / 's2_resnet18_results.md'}")


def parse_args() -> argparse.Namespace:
    report_dir = PROJECT_ROOT / "reports" / "posthoc_oracle_data_adjustment_522"
    s2_dir = PROJECT_ROOT / "data" / "processed" / "s2_425"
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment-dir", type=Path, default=PROJECT_ROOT / "experiments" / "S2_425_Global224_ImageNetResNet18_NYHA3Class_WeightedCE_5Fold")
    parser.add_argument("--s2-manifest", type=Path, default=s2_dir / "s2_manifest.csv")
    parser.add_argument("--s2-data-manifest", type=Path, default=s2_dir / "s2_data_manifest.json")
    parser.add_argument("--retained-ids", type=Path, default=report_dir / "retained_ids_S2.csv")
    parser.add_argument("--excluded-ids", type=Path, default=report_dir / "excluded_ids_S2.csv")
    parser.add_argument("--label-csv", type=Path, default=PROJECT_ROOT / "data" / "raw" / "label_raw.csv")
    parser.add_argument("--source-master-split", type=Path, default=PROJECT_ROOT / "data" / "processed" / "splits" / "nyha_3class_sex_stratified_group_5fold.csv")
    parser.add_argument("--original-resnet18-oof", type=Path, default=PROJECT_ROOT / "experiments" / "Global224_ImageNetResNet18_NYHA3Class_WeightedCE_5Fold" / "summary" / "oof_predictions.csv")
    parser.add_argument("--bootstrap-iterations", type=int, default=2000)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
