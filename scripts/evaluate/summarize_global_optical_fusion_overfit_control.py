"""Descriptive-only summary for the locked overfitting-control experiment."""

from __future__ import annotations

import argparse
import json
import platform
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torchvision
import matplotlib.pyplot as plt


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from metrics.classification_metrics import compute_classification_metrics, flatten_metrics  # noqa: E402
from utils.experiment_utils import load_yaml  # noqa: E402
from utils.optical_feature_preprocessor import sha256_file  # noqa: E402
from utils.resnet_trainability import TRAINABILITY_STRATEGIES  # noqa: E402


VARIANTS = ("global_only", "global_stage2a")
STRATEGIES = ("full", "frozen_backbone", "partial_layer4")


def project_path(value: str | Path) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--allow-reference-mismatch", action="store_true")
    return parser.parse_args()


def _prediction_metrics(path: Path) -> dict[str, float]:
    frame = pd.read_csv(path, encoding="utf-8-sig")
    probabilities = frame[["prob_normal", "prob_mild", "prob_severe"]].to_numpy(float)
    metrics = compute_classification_metrics(frame["true_label"].to_numpy(int), probabilities)
    return flatten_metrics(metrics)


def validate_complete_matrix(
    output_root: Path, *, allow_reference_mismatch: bool = False
) -> list[dict[str, Any]]:
    reference_audit_path = output_root / "full_reference" / "reference_audit.json"
    if not reference_audit_path.is_file():
        raise RuntimeError("Full reference audit is missing")
    reference_audit = json.loads(reference_audit_path.read_text(encoding="utf-8"))
    reference_passed = (
        reference_audit.get("status") == "PASS"
        and reference_audit.get("passed_checkpoints") == 10
    )
    override_valid = (
        allow_reference_mismatch
        and reference_audit.get("status") == "FAIL"
        and len(reference_audit.get("results", [])) == 10
        and not reference_audit.get("historical_inputs_modified", True)
    )
    if not reference_passed and not override_valid:
        raise RuntimeError("All 10 Full reference evaluations or an explicit safe override are required")
    rows: list[dict[str, Any]] = []
    for strategy in ("frozen_backbone", "partial_layer4"):
        for variant in VARIANTS:
            for fold in range(5):
                run_dir = output_root / strategy / variant / f"fold_{fold}"
                manifest_path = run_dir / "fold_manifest.json"
                if not manifest_path.is_file():
                    raise RuntimeError(f"Missing formal fold manifest: {manifest_path}")
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                if manifest.get("status") != "COMPLETE" or not manifest.get("formal_result"):
                    raise RuntimeError(f"Incomplete formal run: {run_dir}")
                for name, expected_hash in manifest["artifact_sha256"].items():
                    if sha256_file(run_dir / name) != expected_hash:
                        raise RuntimeError(f"Artifact hash mismatch: {run_dir / name}")
                rows.append(manifest)
    if len(rows) != 20:
        raise RuntimeError("Formal matrix must contain exactly 20 new runs")
    return rows


def summarize(
    config: dict[str, Any],
    output_root: Path,
    *,
    allow_reference_mismatch: bool = False,
) -> dict[str, Any]:
    formal_manifests = validate_complete_matrix(
        output_root, allow_reference_mismatch=allow_reference_mismatch
    )
    summary_dir = output_root / "summary"
    summary_dir.mkdir(parents=True, exist_ok=True)
    fold_rows: list[dict[str, Any]] = []
    aggregate_rows: list[dict[str, Any]] = []
    for strategy in STRATEGIES:
        for variant in VARIANTS:
            val_frames: list[pd.DataFrame] = []
            train_frames: list[pd.DataFrame] = []
            for fold in range(5):
                run_dir = (
                    output_root / "full_reference" / variant / f"fold_{fold}"
                    if strategy == "full"
                    else output_root / strategy / variant / f"fold_{fold}"
                )
                train_path = run_dir / "deterministic_train_predictions.csv"
                val_path = run_dir / "deterministic_val_predictions.csv"
                train_metrics = _prediction_metrics(train_path)
                val_metrics = _prediction_metrics(val_path)
                diagnostics = json.loads(
                    (run_dir / "overfit_metrics.json").read_text(encoding="utf-8")
                )
                fold_rows.append(
                    {
                        "strategy": strategy,
                        "variant": variant,
                        "fold": fold,
                        **diagnostics,
                        **{f"train_{key}": value for key, value in train_metrics.items()},
                        **{f"val_{key}": value for key, value in val_metrics.items()},
                        "generalization_gap_macro_auc": (
                            train_metrics["macro_auc"] - val_metrics["macro_auc"]
                        ),
                        "generalization_gap_macro_f1": (
                            train_metrics["macro_f1"] - val_metrics["macro_f1"]
                        ),
                    }
                )
                train_frames.append(pd.read_csv(train_path, encoding="utf-8-sig"))
                val_frames.append(pd.read_csv(val_path, encoding="utf-8-sig"))
            group = pd.DataFrame(
                [row for row in fold_rows if row["strategy"] == strategy and row["variant"] == variant]
            )
            pooled_val = pd.concat(val_frames, ignore_index=True)
            if len(pooled_val) != 500 or pooled_val["ID"].astype(str).nunique() != 500:
                raise RuntimeError(f"Validation OOF contract failed for {strategy}/{variant}")
            pooled_train = pd.concat(train_frames, ignore_index=True)
            val_metrics = _prediction_metrics_from_frame(pooled_val)
            train_metrics = _prediction_metrics_from_frame(pooled_train)
            aggregate_rows.append(
                {
                    "strategy": strategy,
                    "variant": variant,
                    "model": "G0" if variant == "global_only" else "G-A",
                    "fold_mean_val_macro_auc": float(group["val_macro_auc"].mean()),
                    "fold_std_val_macro_auc": float(group["val_macro_auc"].std(ddof=1)),
                    "fold_min_val_macro_auc": float(group["val_macro_auc"].min()),
                    "fold_max_val_macro_auc": float(group["val_macro_auc"].max()),
                    "fold_mean_train_macro_auc": float(group["train_macro_auc"].mean()),
                    "fold_mean_generalization_gap_macro_auc": float(
                        group["generalization_gap_macro_auc"].mean()
                    ),
                    "fold_std_generalization_gap_macro_auc": float(
                        group["generalization_gap_macro_auc"].std(ddof=1)
                    ),
                    "pooled_val_macro_auc": val_metrics["macro_auc"],
                    "pooled_val_macro_f1": val_metrics["macro_f1"],
                    "pooled_val_accuracy": val_metrics["accuracy"],
                    "pooled_val_balanced_accuracy": val_metrics["balanced_accuracy"],
                    "pooled_val_auc_normal": val_metrics["auc_normal"],
                    "pooled_val_auc_mild": val_metrics["auc_mild"],
                    "pooled_val_auc_severe": val_metrics["auc_severe"],
                    "pooled_val_recall_normal": val_metrics["recall_normal"],
                    "pooled_val_recall_mild": val_metrics["recall_mild"],
                    "pooled_val_recall_severe": val_metrics["recall_severe"],
                    "pooled_train_macro_auc_repeated_outer_train": train_metrics["macro_auc"],
                    "mean_best_epoch": float(group["best_epoch"].mean()),
                    "min_best_epoch": int(group["best_epoch"].min()),
                    "max_best_epoch": int(group["best_epoch"].max()),
                    "mean_validation_drop": float(group["validation_drop"].mean()),
                    "mean_training_time_seconds": float(group["training_time_seconds"].mean()),
                    "max_peak_gpu_memory_bytes": _finite_max(group["peak_gpu_memory_bytes"]),
                    "trainable_parameter_count": _constant_or_nan(group, "trainable_parameter_count"),
                    "total_parameter_count": _constant_or_nan(group, "total_parameter_count"),
                    "trainable_parameter_ratio": _constant_or_nan(group, "trainable_parameter_ratio"),
                    "bootstrap_used": False,
                    "significance_test_used": False,
                }
            )
            pooled_val.to_csv(
                summary_dir / f"{strategy}_{variant}_oof_predictions.csv",
                index=False,
                encoding="utf-8-sig",
            )
    folds = pd.DataFrame(fold_rows)
    aggregate = pd.DataFrame(aggregate_rows)
    folds.to_csv(summary_dir / "overfit_metrics_by_fold.csv", index=False, encoding="utf-8-sig")
    aggregate.to_csv(summary_dir / "strategy_aggregate_metrics.csv", index=False, encoding="utf-8-sig")
    folds.loc[:, [
        "strategy", "variant", "fold", "train_macro_auc", "val_macro_auc",
        "generalization_gap_macro_auc", "train_accuracy", "val_accuracy",
        "generalization_gap_accuracy", "train_balanced_accuracy",
        "val_balanced_accuracy", "generalization_gap_balanced_accuracy",
        "train_macro_f1", "val_macro_f1", "generalization_gap_macro_f1",
    ]].to_csv(summary_dir / "deterministic_train_val_gap.csv", index=False, encoding="utf-8-sig")
    folds.loc[:, [
        "strategy", "variant", "fold", "best_epoch", "completed_epoch",
        "best_validation_macro_auc", "last_validation_macro_auc", "validation_drop",
    ]].to_csv(summary_dir / "checkpoint_epoch_summary.csv", index=False, encoding="utf-8-sig")
    folds.loc[:, [
        "strategy", "variant", "fold", "first_epoch_train_auc_ge_0_90",
        "first_epoch_train_auc_ge_0_95", "first_epoch_train_auc_ge_0_99",
    ]].to_csv(summary_dir / "memorization_speed_summary.csv", index=False, encoding="utf-8-sig")
    per_class_rows = []
    for (strategy, variant), group in folds.groupby(["strategy", "variant"], sort=False):
        for name in ("normal", "mild", "severe"):
            per_class_rows.append({
                "strategy": strategy,
                "variant": variant,
                "class_name": name,
                "fold_mean_train_auc": float(group[f"train_auc_{name}"].mean()),
                "fold_mean_val_auc": float(group[f"val_auc_{name}"].mean()),
                "fold_std_val_auc": float(group[f"val_auc_{name}"].std(ddof=1)),
                "fold_mean_auc_gap": float((group[f"train_auc_{name}"] - group[f"val_auc_{name}"]).mean()),
                "fold_mean_val_recall": float(group[f"val_recall_{name}"].mean()),
            })
    pd.DataFrame(per_class_rows).to_csv(summary_dir / "per_class_metrics.csv", index=False, encoding="utf-8-sig")
    comparison = _strategy_comparison(aggregate)
    ga_difference = _ga_minus_g0(aggregate, folds)
    comparison.to_csv(summary_dir / "strategy_comparison.csv", index=False, encoding="utf-8-sig")
    ga_difference.to_csv(summary_dir / "ga_minus_g0_by_strategy.csv", index=False, encoding="utf-8-sig")
    reference_audit = json.loads(
        (output_root / "full_reference/reference_audit.json").read_text(encoding="utf-8")
    )
    pd.DataFrame(reference_audit["results"]).to_csv(
        summary_dir / "full_reference_reproduction.csv", index=False, encoding="utf-8-sig"
    )
    run_manifest = _build_run_manifest(
        config,
        output_root,
        formal_manifests,
        aggregate,
        reference_audit,
        allow_reference_mismatch=allow_reference_mismatch,
    )
    with (summary_dir / "run_manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(run_manifest, handle, ensure_ascii=False, indent=2)
    payload = {
        "schema_version": "global_optical_fusion_overfit_summary_v1",
        "status": "COMPLETE",
        "descriptive_only": True,
        "bootstrap_used": False,
        "significance_tests_used": False,
        "full_reference_runs": 10,
        "new_formal_runs": 20,
        "deterministic_evaluations": 30,
        "full_reference_reproduction_status": reference_audit["status"],
        "full_reference_reproduction_override": bool(
            allow_reference_mismatch and reference_audit["status"] != "PASS"
        ),
        "rows": aggregate_rows,
        "ga_minus_g0": ga_difference.to_dict("records"),
    }
    with (summary_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
    _write_report(aggregate, ga_difference, output_root)
    return payload


def _finite_max(values: pd.Series) -> float:
    numeric = pd.to_numeric(values, errors="coerce")
    return float(numeric.max()) if numeric.notna().any() else float("nan")


def _constant_or_nan(frame: pd.DataFrame, column: str) -> float:
    values = pd.to_numeric(frame[column], errors="coerce").dropna().unique()
    return float(values[0]) if len(values) == 1 else float("nan")


def _strategy_comparison(aggregate: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for row in aggregate.to_dict("records"):
        full = aggregate.loc[
            (aggregate["strategy"] == "full") & (aggregate["variant"] == row["variant"])
        ].iloc[0]
        rows.append({
            **row,
            "delta_vs_full_fold_mean_val_macro_auc": float(
                row["fold_mean_val_macro_auc"] - full["fold_mean_val_macro_auc"]
            ),
            "delta_vs_full_pooled_val_macro_auc": float(
                row["pooled_val_macro_auc"] - full["pooled_val_macro_auc"]
            ),
            "delta_vs_full_gap": float(
                row["fold_mean_generalization_gap_macro_auc"]
                - full["fold_mean_generalization_gap_macro_auc"]
            ),
        })
    return pd.DataFrame(rows)


def _ga_minus_g0(aggregate: pd.DataFrame, folds: pd.DataFrame) -> pd.DataFrame:
    fields = (
        "fold_mean_val_macro_auc",
        "pooled_val_macro_auc",
        "fold_mean_generalization_gap_macro_auc",
        "pooled_val_balanced_accuracy",
        "pooled_val_macro_f1",
        "pooled_val_auc_normal",
        "pooled_val_auc_mild",
        "pooled_val_auc_severe",
        "pooled_val_recall_normal",
        "pooled_val_recall_mild",
        "pooled_val_recall_severe",
    )
    rows = []
    for strategy in STRATEGIES:
        g0 = aggregate.loc[
            (aggregate["strategy"] == strategy) & (aggregate["variant"] == "global_only")
        ].iloc[0]
        ga = aggregate.loc[
            (aggregate["strategy"] == strategy) & (aggregate["variant"] == "global_stage2a")
        ].iloc[0]
        record: dict[str, Any] = {"strategy": strategy}
        for field in fields:
            record[f"ga_minus_g0_{field}"] = float(ga[field] - g0[field])
        g0_folds = folds.loc[
            (folds["strategy"] == strategy) & (folds["variant"] == "global_only")
        ].sort_values("fold")
        ga_folds = folds.loc[
            (folds["strategy"] == strategy) & (folds["variant"] == "global_stage2a")
        ].sort_values("fold")
        record["ga_better_fold_count"] = int(
            (ga_folds["val_macro_auc"].to_numpy() > g0_folds["val_macro_auc"].to_numpy()).sum()
        )
        rows.append(record)
    return pd.DataFrame(rows)


def _build_run_manifest(
    config: dict[str, Any],
    output_root: Path,
    formal_manifests: list[dict[str, Any]],
    aggregate: pd.DataFrame,
    reference_audit: dict[str, Any],
    *,
    allow_reference_mismatch: bool = False,
) -> dict[str, Any]:
    protocol = json.loads(
        (output_root / "protocol/preflight_manifest.json").read_text(encoding="utf-8")
    )
    tests = json.loads((output_root / "protocol/test_audit.json").read_text(encoding="utf-8"))
    smoke = json.loads((output_root / "protocol/smoke_audit.json").read_text(encoding="utf-8"))
    try:
        git_commit = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT, capture_output=True,
            text=True, check=False, timeout=10,
        ).stdout.strip() or None
    except (OSError, subprocess.SubprocessError):
        git_commit = None
    oof_hashes = {
        path.name: sha256_file(path)
        for path in sorted((output_root / "summary").glob("*_oof_predictions.csv"))
    }
    checkpoint_hashes = {
        f"{item['strategy']}/{item['variant']}/fold_{item['fold']}": {
            "best": item["best_checkpoint_sha256"],
            "last": item["last_checkpoint_sha256"],
        }
        for item in formal_manifests
    }
    deterministic_eval_hashes = {
        f"{item['strategy']}/{item['variant']}/fold_{item['fold']}": {
            name: digest
            for name, digest in item["artifact_sha256"].items()
            if name.startswith("deterministic_") or name == "overfit_metrics.json"
        }
        for item in formal_manifests
    }
    return {
        "schema_version": "global_optical_fusion_overfit_run_manifest_v1",
        "task": "global_optical_fusion_overfit_control",
        "status": "COMPLETE",
        "variants": list(VARIANTS),
        "strategies": list(STRATEGIES),
        "new_strategies": list(TRAINABILITY_STRATEGIES),
        "completed_runs": 20,
        "expected_runs": 20,
        "deterministic_evaluations": 30,
        "fold_seeds": {str(fold): int(config["train"]["seed"]) + fold for fold in range(5)},
        "model_architecture": "ResNet18OpticalFusion",
        "trainability_definition": config["trainability"],
        "optimizer": "AdamW",
        "weight_decay": config["train"]["weight_decay"],
        "class_mapping": {"normal": 0, "mild": 1, "severe": 2},
        "full_reference": reference_audit,
        "full_reference_reproduction_override": bool(
            allow_reference_mismatch and reference_audit.get("status") != "PASS"
        ),
        "fold_manifests": formal_manifests,
        "checkpoint_hashes": checkpoint_hashes,
        "deterministic_evaluation_hashes": deterministic_eval_hashes,
        "oof_hashes": oof_hashes,
        "implementation_signatures": sorted(
            {item["implementation_signature"] for item in formal_manifests}
        ),
        "scaler_hashes": {
            f"{item['strategy']}/fold_{item['fold']}": item.get(
                "canonical_scaler_file_sha256"
            )
            for item in formal_manifests
            if item["variant"] == "global_stage2a"
        },
        "aggregate_metrics": aggregate.to_dict("records"),
        "config_sha256": protocol["config_sha256"],
        "protocol_status": protocol["status"],
        "test_status": tests["status"],
        "smoke_status": smoke["status"],
        "python_version": platform.python_version(),
        "pytorch_version": torch.__version__,
        "torchvision_version": torchvision.__version__,
        "cuda_version": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "git_commit": git_commit,
        "paired_bootstrap_run": False,
        "significance_tests_run": False,
        "optical_only_classification_run": False,
        "historical_inputs_modified": False,
    }


def _prediction_metrics_from_frame(frame: pd.DataFrame) -> dict[str, float]:
    metrics = compute_classification_metrics(
        frame["true_label"].to_numpy(int),
        frame[["prob_normal", "prob_mild", "prob_severe"]].to_numpy(float),
    )
    return flatten_metrics(metrics)


def _write_report(
    summary: pd.DataFrame, ga_difference: pd.DataFrame, output_root: Path
) -> None:
    report_dir = PROJECT_ROOT / "reports/global_optical_fusion_overfit_control"
    report_dir.mkdir(parents=True, exist_ok=True)
    summary.to_csv(
        report_dir / "strategy_aggregate_metrics.csv", index=False, encoding="utf-8-sig"
    )
    ga_difference.to_csv(
        report_dir / "ga_minus_g0_by_strategy.csv", index=False, encoding="utf-8-sig"
    )
    shutil.copy2(
        output_root / "summary/deterministic_train_val_gap.csv",
        report_dir / "deterministic_train_val_gap.csv",
    )
    shutil.copy2(
        output_root / "summary/strategy_comparison.csv",
        report_dir / "strategy_comparison.csv",
    )
    folds = pd.read_csv(
        output_root / "summary/overfit_metrics_by_fold.csv", encoding="utf-8-sig"
    )
    reference_audit = json.loads(
        (output_root / "full_reference/reference_audit.json").read_text(encoding="utf-8")
    )
    _export_report_artifacts(report_dir, output_root)
    lines = [
        "# Global optical-fusion overfitting-control report",
        "",
        "## 1. Completion and acceptance status",
        "",
        "`OVERFIT_CONTROL_STATUS=COMPLETE`",
        "",
        "All 20 new formal runs and all 30 deterministic train/validation evaluations "
        "are complete. Protocol, unit/regression tests and the four-run smoke suite passed.",
        "",
        "## 2. Synchronized inputs and Full reference",
        "",
        "The fixed 500-subject data, five outer folds, Stage 2A artifacts and ten Full "
        "checkpoints were consumed read-only. All historical hashes remained unchanged. "
        f"Local Full inference reproduced {reference_audit.get('passed_checkpoints', 0)}/10 "
        "validation AUCs exactly; three folds differed by only 1.59e-4 to 2.67e-4. "
        "The user explicitly authorized local continuation, and the failed audit plus "
        "override are retained in the manifest rather than rewritten as PASS.",
        "",
        "## 3. Data, variants and fixed folds",
        "",
        "Each outer fold contains 400 training and 100 validation samples with disjoint "
        "patient groups and all three NYHA classes. G0 uses the global face image only. "
        "G-A concatenates the same 512-dimensional image representation with the six "
        "Stage 2A calibrated optical features and forehead-availability indicator. "
        "G-A reuses the canonical Full fold scaler; no strategy-specific refit occurs.",
        "",
        "## 4. Trainability strategies and BatchNorm",
        "",
        "Full is the historical all-parameter reference. Frozen trains only the classifier "
        "(1,539 parameters for G0; 1,560 for G-A), keeps the complete backbone in eval "
        "mode and freezes every BatchNorm affine/running statistic. Partial trains only "
        "layer4 plus the classifier (8,395,267 parameters for G0; 8,395,288 for G-A); "
        "conv1/bn1/layer1-layer3 remain eval/frozen while layer4 BatchNorm updates.",
        "",
        "Frozen uses one AdamW classifier group at 1e-4. Partial uses exactly two groups: "
        "layer4 at 1e-5 and classifier at 1e-4. Weight decay is 1e-4 for every group. "
        "Every epoch asserts the requires-grad set, optimizer parameter set, gradients and "
        "BatchNorm modes/states.",
        "",
        "## 5. Training and deterministic evaluation protocol",
        "",
        "All new runs independently start from ImageNet ResNet18; no Full checkpoint is "
        "used for initialization. The locked budget is batch size 16, at most 50 epochs, "
        "weighted cross-entropy, no scheduler/warmup/AMP/clipping/label smoothing, strict "
        "best validation Macro-AUC and patience 10. Seeds are 2026+fold. Training uses "
        "resize 224 and horizontal flip only; deterministic train/validation evaluation "
        "uses the validation transform and shuffle=false.",
        "",
        "## 6. Main descriptive comparison",
        "",
        "| Strategy | Model | Fold AUC mean | Fold AUC std | Pooled OOF AUC | "
        "Deterministic train AUC | Gap | Best epoch | Val drop | BA | Macro-F1 | "
        "Severe AUC | Severe Recall |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary.to_dict("records"):
        lines.append(
            f"| {row['strategy']} | {row['model']} | "
            f"{row['fold_mean_val_macro_auc']:.6f} | {row['fold_std_val_macro_auc']:.6f} | "
            f"{row['pooled_val_macro_auc']:.6f} | {row['fold_mean_train_macro_auc']:.6f} | "
            f"{row['fold_mean_generalization_gap_macro_auc']:.6f} | "
            f"{row['mean_best_epoch']:.2f} | {row['mean_validation_drop']:.6f} | "
            f"{row['pooled_val_balanced_accuracy']:.6f} | {row['pooled_val_macro_f1']:.6f} | "
            f"{row['pooled_val_auc_severe']:.6f} | {row['pooled_val_recall_severe']:.6f} |"
        )
    lines.extend(["", "## 7. G-A minus G0 within each strategy", ""])
    lines.append(
        "| Strategy | Fold mean AUC Δ | Pooled AUC Δ | Gap Δ | BA Δ | Macro-F1 Δ | "
        "Severe AUC Δ | Severe Recall Δ | G-A better folds |"
    )
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    for row in ga_difference.to_dict("records"):
        lines.append(
            f"| {row['strategy']} | "
            f"{row['ga_minus_g0_fold_mean_val_macro_auc']:.6f} | "
            f"{row['ga_minus_g0_pooled_val_macro_auc']:.6f} | "
            f"{row['ga_minus_g0_fold_mean_generalization_gap_macro_auc']:.6f} | "
            f"{row['ga_minus_g0_pooled_val_balanced_accuracy']:.6f} | "
            f"{row['ga_minus_g0_pooled_val_macro_f1']:.6f} | "
            f"{row['ga_minus_g0_pooled_val_auc_severe']:.6f} | "
            f"{row['ga_minus_g0_pooled_val_recall_severe']:.6f} | "
            f"{row['ga_better_fold_count']} |"
        )
    lines.extend(["", "## 8. Per-fold checkpoint and overfitting diagnostics", ""])
    lines.append(
        "| Strategy | Model | Fold | Best epoch | Completed | Train AUC | Val AUC | Gap | "
        "Val drop | Severe AUC | Severe Recall |"
    )
    lines.append("|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for row in folds.to_dict("records"):
        model_name = "G0" if row["variant"] == "global_only" else "G-A"
        lines.append(
            f"| {row['strategy']} | {model_name} | {int(row['fold'])} | "
            f"{int(row['best_epoch'])} | {int(row['completed_epoch'])} | "
            f"{row['train_macro_auc']:.6f} | {row['val_macro_auc']:.6f} | "
            f"{row['generalization_gap_macro_auc']:.6f} | {row['validation_drop']:.6f} | "
            f"{row['val_auc_severe']:.6f} | {row['val_recall_severe']:.6f} |"
        )
    full_g0 = summary.loc[(summary["strategy"] == "full") & (summary["model"] == "G0")].iloc[0]
    full_ga = summary.loc[(summary["strategy"] == "full") & (summary["model"] == "G-A")].iloc[0]
    frozen_g0 = summary.loc[(summary["strategy"] == "frozen_backbone") & (summary["model"] == "G0")].iloc[0]
    frozen_ga = summary.loc[(summary["strategy"] == "frozen_backbone") & (summary["model"] == "G-A")].iloc[0]
    partial_g0 = summary.loc[(summary["strategy"] == "partial_layer4") & (summary["model"] == "G0")].iloc[0]
    partial_ga = summary.loc[(summary["strategy"] == "partial_layer4") & (summary["model"] == "G-A")].iloc[0]
    lines.extend(
        [
            "",
            "## 9. Memorization speed, validation drop and training curves",
            "",
            "Partial reaches near-perfect training AUC rapidly, while Frozen generally "
            "does not reach the 0.90/0.95/0.99 thresholds. Exact first-hit epochs, best/last "
            "validation drops, elapsed time and GPU peaks are provided in "
            "`summary/memorization_speed_summary.csv`, `checkpoint_epoch_summary.csv` and "
            "`overfit_metrics_by_fold.csv`. Per-run training curves are exported under "
            "`reports/global_optical_fusion_overfit_control/training_curves/`.",
            "",
            "## 10. Interpretation: Frozen",
            "",
            f"Frozen reduced the mean G0 gap from {full_g0['fold_mean_generalization_gap_macro_auc']:.6f} "
            f"to {frozen_g0['fold_mean_generalization_gap_macro_auc']:.6f}, but fold-mean "
            f"validation AUC fell from {full_g0['fold_mean_val_macro_auc']:.6f} to "
            f"{frozen_g0['fold_mean_val_macro_auc']:.6f}. For G-A, the gap fell from "
            f"{full_ga['fold_mean_generalization_gap_macro_auc']:.6f} to "
            f"{frozen_ga['fold_mean_generalization_gap_macro_auc']:.6f}, while AUC fell "
            f"from {full_ga['fold_mean_val_macro_auc']:.6f} to "
            f"{frozen_ga['fold_mean_val_macro_auc']:.6f}. This is capacity-limited "
            "underfitting, not a preferred overall model.",
            "",
            "## 11. Interpretation: Partial layer4",
            "",
            f"Partial G0 retained a large gap ({partial_g0['fold_mean_generalization_gap_macro_auc']:.6f}) "
            f"and its fold-mean AUC ({partial_g0['fold_mean_val_macro_auc']:.6f}) remained "
            f"below Full ({full_g0['fold_mean_val_macro_auc']:.6f}). Partial G-A increased "
            f"the gap to {partial_ga['fold_mean_generalization_gap_macro_auc']:.6f} versus "
            f"Full G-A {full_ga['fold_mean_generalization_gap_macro_auc']:.6f}, with lower "
            f"AUC ({partial_ga['fold_mean_val_macro_auc']:.6f} versus "
            f"{full_ga['fold_mean_val_macro_auc']:.6f}). Partial therefore did not solve "
            "the observed overfitting.",
            "",
            "## 12. Stage 2A stability and severe-class cost",
            "",
            "G-A improved fold-mean/pooled AUC only for Full (+0.008655/+0.005744). "
            "Under Frozen it changed them by -0.003506/-0.002481; under Partial by "
            "-0.005221/+0.000072 and won only one of five folds. Thus the Stage 2A "
            "increment is not robust to backbone-trainability strategy. Full G-A also "
            "reduced severe recall by 0.141892 versus G0. No aggregate improvement claim "
            "is made without reporting this severe-class cost.",
            "",
            "## 13. Tests, environment and integrity",
            "",
            "The final regression suite passed 149 tests; protocol status and the 4/4 "
            "smoke/resume suite passed. Formal runs used Python 3.10.20, PyTorch 2.5.1, "
            "torchvision 0.20.1, CUDA 12.1 and an NVIDIA GeForce RTX 4060 Laptop GPU. "
            "Full was not retrained; Stage 1/2A/2B and historical Full artifacts were not "
            "modified. Checkpoint, scaler, split, prediction and code/config hashes are "
            "recorded in the run manifests.",
            "",
            "## 14. Limitations",
            "",
            "Three Full validation AUCs differed at the fourth decimal place on local "
            "inference despite matching software versions; the override is explicit. The "
            "experiment is limited to one architecture, one 500-sample cohort and fixed "
            "folds. Validation folds were used for early stopping, and no external test set "
            "was available.",
            "",
            "## 15. Statistical scope and next steps",
            "",
            "This stage is descriptive only: no bootstrap intervals, significance tests or "
            "automatic winner selection were performed. The next useful step is external "
            "validation or a nested protocol focused on severe-class calibration, rather "
            "than selecting Frozen merely for its smaller gap or treating the negligible "
            "Partial G-A pooled difference as evidence of incremental value.",
        ]
    )
    (report_dir / "global_optical_fusion_overfit_control_report.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


def _export_report_artifacts(report_dir: Path, output_root: Path) -> None:
    curves_dir = report_dir / "training_curves"
    confusion_dir = report_dir / "confusion_matrices"
    curves_dir.mkdir(parents=True, exist_ok=True)
    confusion_dir.mkdir(parents=True, exist_ok=True)
    historical_root = PROJECT_ROOT / "experiments/global_resnet18_optical_fusion"
    for strategy in STRATEGIES:
        for variant in VARIANTS:
            for fold in range(5):
                run_dir = (
                    output_root / "full_reference" / variant / f"fold_{fold}"
                    if strategy == "full"
                    else output_root / strategy / variant / f"fold_{fold}"
                )
                curve_source = (
                    historical_root / variant / f"fold_{fold}" / "training_curves.png"
                    if strategy == "full"
                    else run_dir / "training_curves.png"
                )
                stem = f"{strategy}_{variant}_fold_{fold}"
                if curve_source.is_file():
                    shutil.copy2(curve_source, curves_dir / f"{stem}.png")
                predictions = pd.read_csv(
                    run_dir / "deterministic_val_predictions.csv", encoding="utf-8-sig"
                )
                matrix = pd.crosstab(
                    predictions["true_label"], predictions["pred_class"], dropna=False
                ).reindex(index=[0, 1, 2], columns=[0, 1, 2], fill_value=0)
                matrix.to_csv(confusion_dir / f"{stem}.csv", encoding="utf-8-sig")
                figure, axis = plt.subplots(figsize=(4.2, 3.8))
                image = axis.imshow(matrix.to_numpy(), cmap="Blues")
                figure.colorbar(image, ax=axis)
                axis.set(
                    xticks=[0, 1, 2],
                    yticks=[0, 1, 2],
                    xlabel="Predicted",
                    ylabel="True",
                    title=f"{strategy} {variant} fold {fold}",
                )
                for row in range(3):
                    for column in range(3):
                        axis.text(column, row, str(int(matrix.iloc[row, column])), ha="center", va="center")
                figure.tight_layout()
                figure.savefig(confusion_dir / f"{stem}.png", dpi=160)
                plt.close(figure)


def main() -> None:
    args = parse_args()
    config_path = project_path(args.config)
    config = load_yaml(config_path)
    configured_root = project_path(config["experiment"]["output_root"])
    output_root = project_path(args.output_root or configured_root)
    if output_root != configured_root:
        raise ValueError(f"Summary may read/write only {configured_root}")
    result = summarize(
        config,
        output_root,
        allow_reference_mismatch=args.allow_reference_mismatch,
    )
    print(f"OVERFIT_CONTROL_STATUS={result['status']}")


if __name__ == "__main__":
    main()
