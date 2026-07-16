"""Five-variant OOF summaries and paired patient-group bootstrap comparisons."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from metrics.classification_metrics import CLASS_NAMES, compute_classification_metrics, flatten_metrics  # noqa: E402
from scripts.run.run_global_optical_fusion_5fold import (  # noqa: E402
    implementation_signature,
    smoke_evidence,
    training_code_signature,
    validate_locked_config,
)
from utils.experiment_utils import load_yaml  # noqa: E402
from utils.optical_feature_preprocessor import (  # noqa: E402
    AVAILABILITY_COLUMN,
    VARIANTS,
    VARIANT_FEATURE_COLUMNS,
    FeatureScaler,
    sha256_file,
    sha256_json,
)


PROBABILITY_COLUMNS = ["prob_normal", "prob_mild", "prob_severe"]
ALIGNMENT_COLUMNS = ["ID", "patient_group_id", "fold", "true_label"]
PAIRWISE_COMPARISONS = (
    ("global_mask", "global_only"),
    ("global_raw", "global_mask"),
    ("global_stage2a", "global_mask"),
    ("global_stage2b", "global_mask"),
    ("global_stage2a", "global_raw"),
    ("global_stage2b", "global_raw"),
    ("global_stage2b", "global_stage2a"),
)
DELTA_METRICS = (
    "macro_auc", "accuracy", "balanced_accuracy", "macro_f1",
    "auc_normal", "auc_mild", "auc_severe",
    "recall_normal", "recall_mild", "recall_severe",
    "severe_vs_rest_auc", "normal_vs_abnormal_auc",
)
BOOTSTRAP_METRICS = (
    "macro_auc", "accuracy", "balanced_accuracy", "macro_f1",
    "auc_normal", "auc_mild", "auc_severe",
)
BASE_FOLD_ARTIFACTS = (
    "resolved_config.yaml",
    "training_log.csv",
    "training_curves.png",
    "best_macro_auc.pth",
    "last_checkpoint.pth",
    "val_predictions.csv",
    "metrics.json",
    "confusion_matrix.csv",
    "confusion_matrix.png",
    "feature_distribution.csv",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--experiment-dir", type=Path, required=True)
    parser.add_argument("--expected-rows", type=int)
    parser.add_argument("--bootstrap-repetitions", type=int)
    return parser.parse_args()


def _project_path(value: str | Path) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def _json_dump(payload: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, default=str)


def validate_oof_frame(
    frame: pd.DataFrame,
    *,
    expected_rows: int,
    expected_mapping: pd.DataFrame | None = None,
) -> pd.DataFrame:
    required = set(ALIGNMENT_COLUMNS + PROBABILITY_COLUMNS)
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"OOF predictions are missing columns: {missing}")
    result = frame.copy()
    result["ID"] = result["ID"].astype(str)
    result["patient_group_id"] = result["patient_group_id"].astype(str)
    if len(result) != int(expected_rows):
        raise ValueError(f"OOF row count {len(result)} != expected {expected_rows}")
    if result["ID"].duplicated().any() or result["ID"].nunique() != expected_rows:
        raise ValueError("OOF predictions must contain one unique row per ID")
    probabilities = result[PROBABILITY_COLUMNS].to_numpy(float)
    if not np.isfinite(probabilities).all() or (probabilities < 0).any() or (probabilities > 1).any():
        raise ValueError("OOF probabilities must be finite values in [0,1]")
    if not np.allclose(probabilities.sum(axis=1), 1.0, atol=1e-5):
        raise ValueError("OOF probability rows must sum to one")
    folds = pd.to_numeric(result["fold"], errors="coerce")
    labels = pd.to_numeric(result["true_label"], errors="coerce")
    if folds.isna().any() or not folds.astype(int).isin(range(5)).all():
        raise ValueError("OOF fold values must be 0..4")
    if labels.isna().any() or not labels.astype(int).isin([0, 1, 2]).all():
        raise ValueError("OOF true labels must be 0,1,2")
    if expected_mapping is not None:
        mapping = expected_mapping.loc[:, ALIGNMENT_COLUMNS].copy()
        for column in ("ID", "patient_group_id"):
            mapping[column] = mapping[column].astype(str)
        if result[ALIGNMENT_COLUMNS].to_dict("records") != mapping.to_dict("records"):
            raise ValueError("OOF ID/group/fold/label mapping differs from the fixed split order")
    return result


def validate_oof_alignment(
    candidate: pd.DataFrame, reference: pd.DataFrame
) -> None:
    """Require identical existing row order; never sort to hide a mismatch."""
    if len(candidate) != len(reference):
        raise ValueError("Paired OOF variants have different row counts")
    left = candidate.loc[:, ALIGNMENT_COLUMNS].astype(
        {"ID": str, "patient_group_id": str, "fold": int, "true_label": int}
    )
    right = reference.loc[:, ALIGNMENT_COLUMNS].astype(
        {"ID": str, "patient_group_id": str, "fold": int, "true_label": int}
    )
    if left.to_dict("records") != right.to_dict("records"):
        raise ValueError("Paired OOF alignment differs by ID, patient group, fold or true label")


def scalar_metrics(frame: pd.DataFrame) -> dict[str, float]:
    return flatten_metrics(
        compute_classification_metrics(
            frame["true_label"].to_numpy(int), frame[PROBABILITY_COLUMNS].to_numpy(float)
        )
    )


def _patient_group_bootstrap_plan(
    reference: pd.DataFrame,
) -> tuple[tuple[tuple[np.ndarray, ...], ...], dict[str, Any]]:
    """Build label-composition strata while keeping every patient group intact."""
    groups = reference["patient_group_id"].astype(str).to_numpy()
    label_values = pd.to_numeric(reference["true_label"], errors="coerce")
    if label_values.isna().any() or not label_values.isin([0, 1, 2]).all():
        raise ValueError("Bootstrap true labels must be 0,1,2")
    labels = label_values.to_numpy(dtype=np.int64)

    # A patient can have images carrying different labels.  Assigning such a
    # patient to only one label stratum would either split the cluster or discard
    # a true label.  Stratifying by the complete (n0, n1, n2) label composition
    # preserves whole clusters and the exact image-level class counts.
    strata: dict[tuple[int, int, int], list[np.ndarray]] = {}
    mixed_group_count = 0
    for group in pd.unique(groups):
        indices = np.flatnonzero(groups == group).astype(np.int64)
        composition = tuple(
            int(value) for value in np.bincount(labels[indices], minlength=3)
        )
        if sum(value > 0 for value in composition) > 1:
            mixed_group_count += 1
        strata.setdefault(composition, []).append(indices)

    class_counts = np.bincount(labels, minlength=3)
    for label, count in enumerate(class_counts):
        if int(count) == 0:
            raise ValueError(f"No observations exist for class {label}")
    plan = tuple(tuple(population) for population in strata.values())
    return plan, {
        "stratification_rule": "patient_group_true_label_count_composition",
        "stratum_count": len(plan),
        "patient_group_count": int(sum(len(population) for population in plan)),
        "mixed_true_label_patient_group_count": int(mixed_group_count),
        "preserves_exact_image_class_counts": True,
    }


def _sample_patient_group_plan(
    plan: tuple[tuple[np.ndarray, ...], ...], rng: np.random.Generator
) -> np.ndarray:
    sampled: list[np.ndarray] = []
    for population in plan:
        draws = rng.integers(0, len(population), size=len(population))
        sampled.extend(population[int(draw)] for draw in draws)
    if not sampled:
        raise ValueError("Patient-group bootstrap plan is empty")
    return np.concatenate(sampled).astype(np.int64, copy=False)


def _bootstrap_indices_by_patient_group(
    reference: pd.DataFrame, rng: np.random.Generator
) -> np.ndarray:
    plan, _ = _patient_group_bootstrap_plan(reference)
    return _sample_patient_group_plan(plan, rng)


def paired_cluster_bootstrap(
    candidate: pd.DataFrame,
    reference: pd.DataFrame,
    *,
    repetitions: int = 2000,
    seed: int = 2026,
    minimum_valid_repetitions: int | None = None,
    metrics: tuple[str, ...] = BOOTSTRAP_METRICS,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Stratified paired bootstrap that always samples whole patient groups."""
    validate_oof_alignment(candidate, reference)
    if repetitions < 1:
        raise ValueError("Bootstrap repetitions must be positive")
    minimum = repetitions if minimum_valid_repetitions is None else int(minimum_valid_repetitions)
    rng = np.random.default_rng(int(seed))
    plan, plan_audit = _patient_group_bootstrap_plan(reference)
    deltas: dict[str, list[float]] = {metric: [] for metric in metrics}
    failures: Counter[str] = Counter()
    for _ in range(int(repetitions)):
        try:
            indices = _sample_patient_group_plan(plan, rng)
            reference_metrics = scalar_metrics(reference.iloc[indices])
            candidate_metrics = scalar_metrics(candidate.iloc[indices])
            values = {
                metric: float(candidate_metrics[metric] - reference_metrics[metric])
                for metric in metrics
            }
            if not all(np.isfinite(value) for value in values.values()):
                raise ValueError("nonfinite bootstrap metric")
            for metric, value in values.items():
                deltas[metric].append(value)
        except Exception as exc:  # each invalid resample is explicitly audited
            failures[f"{type(exc).__name__}: {exc}"] += 1
    valid = min((len(values) for values in deltas.values()), default=0)
    if valid < minimum:
        raise RuntimeError(
            f"Only {valid}/{repetitions} bootstrap repetitions were valid; minimum={minimum}; "
            f"failures={dict(failures)}"
        )
    rows = []
    for metric in metrics:
        values = np.asarray(deltas[metric], dtype=np.float64)
        rows.append({
            "metric": metric, "delta_mean": float(values.mean()),
            "delta_median": float(np.median(values)),
            "ci_lower_2_5": float(np.quantile(values, 0.025)),
            "ci_upper_97_5": float(np.quantile(values, 0.975)),
            "valid_repetitions": int(values.size), "requested_repetitions": int(repetitions),
            "bootstrap_seed": int(seed), "sampling_unit": "patient_group_id",
            "stratified_by_true_label": True,
            "stratification_rule": plan_audit["stratification_rule"],
            "mixed_true_label_patient_group_count": plan_audit[
                "mixed_true_label_patient_group_count"
            ],
        })
    audit = {
        "requested_repetitions": int(repetitions), "valid_repetitions": int(valid),
        "failed_repetitions": int(repetitions - valid), "failure_reasons": dict(failures),
        "seed": int(seed), "paired": True, "sampling_unit": "patient_group_id",
        "stratified_by_true_label": True,
        **plan_audit,
    }
    return pd.DataFrame(rows), audit


def _save_confusion(matrix: np.ndarray, csv_path: Path, png_path: Path) -> None:
    names = [CLASS_NAMES[index] for index in range(3)]
    pd.DataFrame(matrix, index=names, columns=names).to_csv(
        csv_path, index_label="true\\pred", encoding="utf-8-sig"
    )
    figure, axis = plt.subplots(figsize=(5.5, 5))
    image = axis.imshow(matrix, cmap="Blues")
    figure.colorbar(image, ax=axis)
    axis.set(xticks=range(3), yticks=range(3), xticklabels=names, yticklabels=names,
             xlabel="Predicted", ylabel="True", title="Pooled OOF confusion matrix")
    for row in range(3):
        for column in range(3):
            axis.text(column, row, str(int(matrix[row, column])), ha="center", va="center")
    figure.tight_layout()
    figure.savefig(png_path, dpi=180)
    plt.close(figure)


def _aggregate_fold_metrics(fold_metrics: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for column in fold_metrics.columns:
        if column in {"variant", "fold", "best_epoch", "confusion_matrix"}:
            continue
        values = pd.to_numeric(fold_metrics[column], errors="coerce").dropna()
        if values.empty:
            continue
        rows.append({
            "metric": column, "mean": values.mean(), "sample_std_ddof1": values.std(ddof=1),
            "median": values.median(), "min": values.min(), "max": values.max(),
            "valid_fold_count": int(values.size),
        })
    return pd.DataFrame(rows)


def _dataframe_markdown(frame: pd.DataFrame, columns: list[str] | None = None) -> str:
    """Render a small table without requiring the optional tabulate package."""
    selected = frame.loc[:, columns] if columns is not None else frame
    headers = [str(column) for column in selected.columns]
    lines = ["| " + " | ".join(headers) + " |", "|" + "|".join(["---"] * len(headers)) + "|"]
    for row in selected.itertuples(index=False, name=None):
        formatted = []
        for value in row:
            if isinstance(value, (float, np.floating)):
                formatted.append("nan" if not np.isfinite(value) else f"{float(value):.6f}")
            else:
                formatted.append(str(value))
        lines.append("| " + " | ".join(formatted) + " |")
    return "\n".join(lines)


def _save_training_curve_summary(fold_dirs: list[Path], path: Path, variant: str) -> None:
    figure, axis = plt.subplots(figsize=(8, 5))
    for fold, fold_dir in enumerate(fold_dirs):
        log = pd.read_csv(fold_dir / "training_log.csv")
        axis.plot(log["epoch"], log["val_macro_auc"], label=f"fold {fold}")
    axis.set(xlabel="Epoch", ylabel="Validation macro-AUC", ylim=(0, 1), title=variant)
    axis.legend()
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)


def _expected_mapping(config: dict[str, Any]) -> pd.DataFrame:
    data = config["data"]
    split_root = _project_path(data["split_root"])
    frames = []
    for fold in data["folds"]:
        path = split_root / data["val_csv_pattern"].format(fold=fold)
        frame = pd.read_csv(
            path, dtype={"ID": "string", "patient_group_id": "string"}, encoding="utf-8-sig"
        )
        frames.append(pd.DataFrame({
            "ID": frame["ID"].astype(str),
            "patient_group_id": frame["patient_group_id"].astype(str),
            "fold": frame["fold"].astype(int),
            "true_label": frame["label_3class"].astype(int),
        }))
    return pd.concat(frames, ignore_index=True)


def validate_formal_fold_artifacts(
    fold_dir: Path,
    variant: str,
    fold: int,
    manifest: Mapping[str, Any],
    *,
    expected_implementation_signature: str | None = None,
) -> None:
    """Validate every required formal artifact and its recorded digest."""
    if manifest.get("status") != "COMPLETE" or manifest.get("formal_result") is not True:
        raise ValueError(f"Fold is not a complete formal result: {fold_dir}")
    if manifest.get("variant") != variant or int(manifest.get("fold", -1)) != int(fold):
        raise ValueError(f"Fold manifest identity mismatch: {fold_dir}")
    expected_signature = expected_implementation_signature or training_code_signature()
    if manifest.get("implementation_signature") != expected_signature:
        raise ValueError(f"Fold was produced by a stale implementation: {fold_dir}")
    scaler_required = bool(VARIANT_FEATURE_COLUMNS[variant])
    required = list(BASE_FOLD_ARTIFACTS)
    if scaler_required:
        required.append("feature_scaler.json")
    missing = [name for name in required if not (fold_dir / name).is_file()]
    empty = [
        name for name in required
        if (fold_dir / name).is_file() and (fold_dir / name).stat().st_size == 0
    ]
    if missing or empty:
        raise FileNotFoundError(
            f"Formal fold artifacts are incomplete in {fold_dir}; missing={missing}, empty={empty}"
        )
    if not scaler_required and (fold_dir / "feature_scaler.json").exists():
        raise ValueError(f"{variant} must not have a six-dimensional feature scaler")
    recorded_hashes = manifest.get("artifact_sha256")
    if not isinstance(recorded_hashes, dict):
        raise ValueError(f"Fold manifest has no artifact_sha256 mapping: {fold_dir}")
    if set(recorded_hashes) != set(required):
        raise ValueError(
            f"Artifact hash inventory differs in {fold_dir}; "
            f"recorded={sorted(recorded_hashes)}, required={sorted(required)}"
        )
    for name in required:
        actual = sha256_file(fold_dir / name)
        if recorded_hashes[name] != actual:
            raise ValueError(f"Artifact SHA256 mismatch for {fold_dir / name}")
    if manifest.get("best_checkpoint_sha256") != recorded_hashes["best_macro_auc.pth"]:
        raise ValueError(f"Best-checkpoint digest mismatch in {fold_dir}")
    if manifest.get("last_checkpoint_sha256") != recorded_hashes["last_checkpoint.pth"]:
        raise ValueError(f"Last-checkpoint digest mismatch in {fold_dir}")
    if manifest.get("val_predictions_sha256") != recorded_hashes["val_predictions.csv"]:
        raise ValueError(f"Prediction digest mismatch in {fold_dir}")

    resolved = load_yaml(fold_dir / "resolved_config.yaml")
    resolved_run = resolved.get("resolved_run", {})
    if (
        resolved_run.get("variant") != variant
        or int(resolved_run.get("fold", -1)) != int(fold)
        or resolved_run.get("smoke_test") is not False
    ):
        raise ValueError(f"Resolved run identity mismatch in {fold_dir}")
    training_log = pd.read_csv(fold_dir / "training_log.csv")
    required_log_columns = {
        "epoch", "train_loss", "val_loss", "train_accuracy", "val_accuracy",
        "train_macro_auc", "val_macro_auc", "val_balanced_accuracy",
        "val_macro_f1", "learning_rate", "elapsed_seconds", "is_best",
        "patience_counter",
    }
    if training_log.empty or not required_log_columns.issubset(training_log.columns):
        raise ValueError(f"Training log is incomplete in {fold_dir}")
    if training_log["epoch"].duplicated().any() or not training_log["epoch"].is_monotonic_increasing:
        raise ValueError(f"Training epochs are duplicated or unordered in {fold_dir}")
    with (fold_dir / "metrics.json").open("r", encoding="utf-8") as handle:
        metrics = json.load(handle)
    if metrics.get("variant") != variant or int(metrics.get("fold", -1)) != int(fold):
        raise ValueError(f"Metric identity mismatch in {fold_dir}")
    best_epoch = int(metrics.get("best_epoch", -1))
    best_rows = training_log.loc[training_log["is_best"].astype(int) == 1, "epoch"]
    if best_epoch not in set(best_rows.astype(int)):
        raise ValueError(f"Selected best epoch is not marked is_best in {fold_dir}")
    predictions = pd.read_csv(
        fold_dir / "val_predictions.csv",
        dtype={"ID": "string", "patient_group_id": "string"},
        encoding="utf-8-sig",
    )
    if len(predictions) != 100 or predictions["ID"].nunique() != 100:
        raise ValueError(f"Formal fold predictions must contain 100 unique IDs: {fold_dir}")
    if not (pd.to_numeric(predictions["fold"], errors="coerce") == int(fold)).all():
        raise ValueError(f"Prediction fold values differ in {fold_dir}")
    confusion = pd.read_csv(fold_dir / "confusion_matrix.csv", index_col=0)
    if confusion.shape != (3, 3):
        raise ValueError(f"Confusion matrix must be 3x3 in {fold_dir}")
    distribution = pd.read_csv(fold_dir / "feature_distribution.csv")
    expected_distribution_rows = 6 if scaler_required else 1
    if len(distribution) != expected_distribution_rows:
        raise ValueError(f"Feature distribution row count differs in {fold_dir}")
    if scaler_required:
        scaler = FeatureScaler.load_json(fold_dir / "feature_scaler.json")
        if scaler.variant != variant or scaler.fold != int(fold):
            raise ValueError(f"Feature scaler identity mismatch in {fold_dir}")
        if scaler.payload_sha256 != manifest.get("feature_scaler_sha256"):
            raise ValueError(f"Feature scaler payload hash mismatch in {fold_dir}")
        if recorded_hashes["feature_scaler.json"] != manifest.get("feature_scaler_file_sha256"):
            raise ValueError(f"Feature scaler file hash mismatch in {fold_dir}")


def load_execution_evidence(
    experiment_dir: Path, config_path: Path
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    protocol_path = experiment_dir / "protocol/protocol_manifest.json"
    test_path = experiment_dir / "protocol/test_audit.json"
    if not protocol_path.is_file() or not test_path.is_file():
        raise FileNotFoundError(
            "Formal summary requires protocol_manifest.json and test_audit.json"
        )
    with protocol_path.open("r", encoding="utf-8") as handle:
        protocol = json.load(handle)
    with test_path.open("r", encoding="utf-8") as handle:
        tests = json.load(handle)
    config_digest = sha256_file(config_path)
    if protocol.get("status") != "PASS" or protocol.get("config_sha256") != config_digest:
        raise ValueError("Protocol evidence is missing, failed, or belongs to another config")
    if (
        tests.get("status") != "PASS"
        or tests.get("config_sha256") != config_digest
        or tests.get("implementation_signature") != implementation_signature()
        or not isinstance(tests.get("passed_count"), int)
    ):
        raise ValueError("Test evidence is missing, failed, or stale for the current implementation")
    smoke = smoke_evidence(experiment_dir)
    return protocol, tests, smoke


def summarize_experiment(
    config_path: str | Path,
    experiment_dir: str | Path,
    *,
    expected_rows: int | None = None,
    bootstrap_repetitions: int | None = None,
) -> Path:
    config_path = _project_path(config_path)
    experiment_dir = _project_path(experiment_dir)
    config = load_yaml(config_path)
    validate_locked_config(config)
    protocol_evidence, test_evidence, smoke_test_evidence = load_execution_evidence(
        experiment_dir, config_path
    )
    summary_config = config["summary"]
    expected_rows = int(expected_rows or summary_config["expected_oof_rows"])
    repetitions = int(bootstrap_repetitions or summary_config["bootstrap_repetitions"])
    minimum_valid = min(
        repetitions, int(summary_config["bootstrap_minimum_valid_repetitions"])
    )
    expected_mapping = _expected_mapping(config)
    if len(expected_mapping) != expected_rows:
        raise ValueError("Configured expected OOF rows differ from validation splits")

    variant_oof: dict[str, pd.DataFrame] = {}
    variant_fold_metrics: dict[str, pd.DataFrame] = {}
    loaded: dict[str, list[tuple[pd.DataFrame, dict[str, Any], Path]]] = {}
    recorded_fold_implementation_signature: str | None = None
    for variant in VARIANTS:
        loaded[variant] = []
        for fold in config["data"]["folds"]:
            fold_dir = experiment_dir / variant / f"fold_{fold}"
            manifest_path = fold_dir / "fold_manifest.json"
            if not manifest_path.is_file():
                raise FileNotFoundError(f"Incomplete fold: {manifest_path}")
            with manifest_path.open("r", encoding="utf-8") as handle:
                manifest = json.load(handle)
            manifest_signature = manifest.get("implementation_signature")
            if not isinstance(manifest_signature, str) or not manifest_signature:
                raise ValueError(f"Fold manifest has no implementation signature: {fold_dir}")
            if recorded_fold_implementation_signature is None:
                recorded_fold_implementation_signature = manifest_signature
            elif manifest_signature != recorded_fold_implementation_signature:
                raise ValueError(
                    "Formal folds were produced by inconsistent training implementations: "
                    f"{fold_dir}"
                )
            validate_formal_fold_artifacts(
                fold_dir, variant, fold, manifest,
                expected_implementation_signature=recorded_fold_implementation_signature,
            )
            prediction_path, metric_path = fold_dir / "val_predictions.csv", fold_dir / "metrics.json"
            frame = pd.read_csv(
                prediction_path, dtype={"ID": "string", "patient_group_id": "string"}, encoding="utf-8-sig"
            )
            with metric_path.open("r", encoding="utf-8") as handle:
                metrics = json.load(handle)
            loaded[variant].append((frame, metrics, fold_dir))

    # Validate every formal artifact before creating any summary result.
    for variant in VARIANTS:
        frames = [item[0] for item in loaded[variant]]
        oof = validate_oof_frame(
            pd.concat(frames, ignore_index=True), expected_rows=expected_rows,
            expected_mapping=expected_mapping,
        )
        variant_oof[variant] = oof
        variant_fold_metrics[variant] = pd.DataFrame([item[1] for item in loaded[variant]])
    for variant in VARIANTS[1:]:
        validate_oof_alignment(variant_oof[variant], variant_oof[VARIANTS[0]])

    summary_dir = experiment_dir / "summary"
    summary_dir.mkdir(parents=True, exist_ok=True)
    all_oof_metric_rows = []
    distribution_frames = []
    for variant in VARIANTS:
        output = summary_dir / variant
        output.mkdir(parents=True, exist_ok=True)
        fold_metrics = variant_fold_metrics[variant]
        fold_metrics.to_csv(output / "fold_metrics.csv", index=False, encoding="utf-8-sig")
        _aggregate_fold_metrics(fold_metrics).to_csv(
            output / "aggregate_fold_metrics.csv", index=False, encoding="utf-8-sig"
        )
        oof = variant_oof[variant]
        oof.to_csv(output / "oof_predictions.csv", index=False, encoding="utf-8-sig")
        metrics = compute_classification_metrics(
            oof["true_label"].to_numpy(int), oof[PROBABILITY_COLUMNS].to_numpy(float)
        )
        scalar = flatten_metrics(metrics)
        all_oof_metric_rows.append({"variant": variant, **scalar})
        _json_dump(
            {**scalar, "confusion_matrix": np.asarray(metrics["confusion_matrix"]).tolist()},
            output / "oof_metrics.json",
        )
        _save_confusion(
            np.asarray(metrics["confusion_matrix"]), output / "oof_confusion_matrix.csv",
            output / "oof_confusion_matrix.png",
        )
        _save_training_curve_summary(
            [item[2] for item in loaded[variant]], output / "training_curve_summary.png", variant
        )
        for _, _, fold_dir in loaded[variant]:
            distribution_frames.append(pd.read_csv(fold_dir / "feature_distribution.csv"))
    oof_metrics_all = pd.DataFrame(all_oof_metric_rows)
    oof_metrics_all.to_csv(summary_dir / "oof_metrics_all_variants.csv", index=False, encoding="utf-8-sig")

    foldwise_rows, comparison_rows, bootstrap_frames = [], [], []
    bootstrap_audits: dict[str, Any] = {}
    for candidate, reference in PAIRWISE_COMPARISONS:
        candidate_fold = variant_fold_metrics[candidate].sort_values("fold", kind="stable")
        reference_fold = variant_fold_metrics[reference].sort_values("fold", kind="stable")
        if candidate_fold["fold"].tolist() != reference_fold["fold"].tolist():
            raise ValueError(f"Fold alignment differs for {candidate} vs {reference}")
        candidate_oof_metrics = scalar_metrics(variant_oof[candidate])
        reference_oof_metrics = scalar_metrics(variant_oof[reference])
        for metric in DELTA_METRICS:
            deltas = candidate_fold[metric].to_numpy(float) - reference_fold[metric].to_numpy(float)
            for fold, delta in zip(candidate_fold["fold"], deltas):
                foldwise_rows.append({
                    "candidate": candidate, "reference": reference,
                    "metric": metric, "fold": int(fold), "delta_candidate_minus_reference": delta,
                })
            comparison_rows.append({
                "candidate": candidate, "reference": reference, "metric": metric,
                "oof_delta_candidate_minus_reference": candidate_oof_metrics[metric] - reference_oof_metrics[metric],
                "fold_delta_mean": float(np.mean(deltas)),
                "fold_delta_sample_std_ddof1": float(np.std(deltas, ddof=1)),
                "fold_delta_median": float(np.median(deltas)), "fold_delta_min": float(np.min(deltas)),
                "fold_delta_max": float(np.max(deltas)), "candidate_better_folds": int((deltas > 0).sum()),
                "reference_better_folds": int((deltas < 0).sum()), "equal_folds": int((deltas == 0).sum()),
            })
        bootstrap, audit = paired_cluster_bootstrap(
            variant_oof[candidate], variant_oof[reference], repetitions=repetitions,
            seed=int(summary_config["bootstrap_seed"]), minimum_valid_repetitions=minimum_valid,
        )
        bootstrap.insert(0, "reference", reference)
        bootstrap.insert(0, "candidate", candidate)
        bootstrap_frames.append(bootstrap)
        bootstrap_audits[f"{candidate}_minus_{reference}"] = audit

    foldwise = pd.DataFrame(foldwise_rows)
    comparison = pd.DataFrame(comparison_rows)
    bootstraps = pd.concat(bootstrap_frames, ignore_index=True)
    foldwise.to_csv(summary_dir / "foldwise_metric_deltas.csv", index=False, encoding="utf-8-sig")
    comparison.to_csv(summary_dir / "pairwise_comparison.csv", index=False, encoding="utf-8-sig")
    bootstraps.to_csv(summary_dir / "pairwise_bootstrap_deltas.csv", index=False, encoding="utf-8-sig")
    distributions = pd.concat(distribution_frames, ignore_index=True)
    distributions.to_csv(summary_dir / "feature_distribution_audit.csv", index=False, encoding="utf-8-sig")
    fold_manifests = {
        variant: [
            json.loads((experiment_dir / variant / f"fold_{fold}" / "fold_manifest.json").read_text(encoding="utf-8"))
            for fold in config["data"]["folds"]
        ]
        for variant in VARIANTS
    }

    reports_dir = PROJECT_ROOT / "reports/global_resnet18_optical_fusion"
    reports_dir.mkdir(parents=True, exist_ok=True)
    report_curve_dir = reports_dir / "training_curves"
    report_confusion_dir = reports_dir / "confusion_matrices"
    report_curve_dir.mkdir(parents=True, exist_ok=True)
    report_confusion_dir.mkdir(parents=True, exist_ok=True)
    for name in (
        "oof_metrics_all_variants.csv", "pairwise_comparison.csv",
        "pairwise_bootstrap_deltas.csv", "foldwise_metric_deltas.csv",
        "feature_distribution_audit.csv",
    ):
        shutil.copy2(summary_dir / name, reports_dir / name)
    for variant in VARIANTS:
        shutil.copy2(
            summary_dir / variant / "training_curve_summary.png",
            report_curve_dir / f"{variant}_training_curve_summary.png",
        )
        shutil.copy2(
            summary_dir / variant / "oof_confusion_matrix.png",
            report_confusion_dir / f"{variant}_oof_confusion_matrix.png",
        )
    main_metric_columns = [
        "variant", "macro_auc", "accuracy", "balanced_accuracy", "macro_f1",
        "auc_normal", "auc_mild", "auc_severe",
    ]
    main_comparison = comparison.loc[comparison["metric"].isin(
        ["macro_auc", "accuracy", "balanced_accuracy", "macro_f1"]
    )]
    bootstrap_main = bootstraps.loc[bootstraps["metric"] == "macro_auc"]
    best_epoch_rows: list[dict[str, Any]] = []
    aggregate_rows: list[dict[str, Any]] = []
    overfit_rows: list[dict[str, Any]] = []
    class_weight_rows: list[dict[str, Any]] = []
    for variant in VARIANTS:
        fold_metrics = variant_fold_metrics[variant].sort_values("fold", kind="stable")
        for metric in ("macro_auc", "accuracy", "balanced_accuracy", "macro_f1"):
            values = pd.to_numeric(fold_metrics[metric], errors="raise")
            aggregate_rows.append({
                "variant": variant, "metric": metric, "mean": values.mean(),
                "sample_std_ddof1": values.std(ddof=1), "median": values.median(),
                "min": values.min(), "max": values.max(),
            })
        for fold, (_, metrics, fold_dir), manifest in zip(
            config["data"]["folds"], loaded[variant], fold_manifests[variant]
        ):
            log = pd.read_csv(fold_dir / "training_log.csv")
            best_epoch = int(metrics["best_epoch"])
            best_row = log.loc[log["epoch"].astype(int) == best_epoch].iloc[0]
            last_row = log.iloc[-1]
            best_epoch_rows.append({
                "variant": variant, "fold": fold, "best_epoch": best_epoch,
                "best_val_macro_auc": float(metrics["macro_auc"]),
                "best_val_macro_f1": float(metrics["macro_f1"]),
                "best_val_balanced_accuracy": float(metrics["balanced_accuracy"]),
                "completed_epoch": int(last_row["epoch"]),
            })
            overfit_rows.append({
                "variant": variant, "fold": fold,
                "best_epoch": best_epoch,
                "train_macro_auc_at_best": float(best_row["train_macro_auc"]),
                "val_macro_auc_at_best": float(best_row["val_macro_auc"]),
                "train_minus_val_auc_at_best": float(
                    best_row["train_macro_auc"] - best_row["val_macro_auc"]
                ),
                "last_train_macro_auc": float(last_row["train_macro_auc"]),
                "last_val_macro_auc": float(last_row["val_macro_auc"]),
            })
            weights = manifest["class_weights"]
            class_weight_rows.append({
                "variant": variant, "fold": fold,
                "weight_normal": weights[0], "weight_mild": weights[1],
                "weight_severe": weights[2],
            })
    aggregate_report = pd.DataFrame(aggregate_rows)
    best_epoch_report = pd.DataFrame(best_epoch_rows)
    overfit_report = pd.DataFrame(overfit_rows)
    class_weight_report = pd.DataFrame(class_weight_rows)
    per_class_changes = comparison.loc[
        comparison["metric"].isin([
            "auc_normal", "auc_mild", "auc_severe",
            "recall_normal", "recall_mild", "recall_severe",
        ])
    ]
    numeric_smd = pd.to_numeric(
        distributions.get("standardized_mean_difference"), errors="coerce"
    )
    shift_frame = distributions.assign(abs_smd=numeric_smd.abs()).dropna(subset=["abs_smd"])
    shift_summary = shift_frame.groupby("variant", as_index=False).agg(
        mean_absolute_smd=("abs_smd", "mean"),
        maximum_absolute_smd=("abs_smd", "max"),
    )
    oof_completeness = pd.DataFrame([
        {
            "variant": variant, "rows": len(frame), "unique_ids": frame["ID"].nunique(),
            "folds": ",".join(map(str, sorted(frame["fold"].unique()))),
            "probabilities_valid": True,
        }
        for variant, frame in variant_oof.items()
    ])
    pooled_lookup = oof_metrics_all.set_index("variant")
    delta_b_a = float(
        pooled_lookup.loc["global_stage2b", "macro_auc"]
        - pooled_lookup.loc["global_stage2a", "macro_auc"]
    )
    b_a_fold_row = comparison.loc[
        (comparison["candidate"] == "global_stage2b")
        & (comparison["reference"] == "global_stage2a")
        & (comparison["metric"] == "macro_auc")
    ].iloc[0]
    if abs(delta_b_a) < 0.01:
        recommendation = (
            "G-A and G-B have close pooled macro-AUC (absolute delta < 0.01); "
            "the provisional candidate is G-A because Ridge is simpler and Stage 2B has greater shift risk."
        )
    elif (
        delta_b_a > 0
        and int(b_a_fold_row["candidate_better_folds"]) >= 3
        and pooled_lookup.loc["global_stage2b", "macro_f1"]
        >= pooled_lookup.loc["global_stage2a", "macro_f1"]
        and pooled_lookup.loc["global_stage2b", "balanced_accuracy"]
        >= pooled_lookup.loc["global_stage2a", "balanced_accuracy"]
    ):
        recommendation = (
            "G-B is the provisional candidate over G-A because its pooled macro-AUC is higher, "
            "at least 3/5 folds agree, and pooled macro-F1/balanced accuracy do not decrease; "
            "per-class recall, bootstrap CI, and Stage 2B shift still require human review."
        )
    else:
        recommendation = (
            "The pre-registered conditions do not support G-B over G-A. Select the final candidate "
            "only after jointly reviewing pooled/fold metrics, availability control, per-class recall, "
            "bootstrap intervals, and feature shift; do not use one metric alone."
        )
    report_lines = [
        "# Global ResNet18 + Optical Phenotype Fusion Experiment", "",
        "EXPERIMENT_STATUS=COMPLETE", "",
        "## Completion, data, and provenance", "",
        "All five variants and all 25 fold runs completed on the formal 500-case cohort (normal 115, mild 237, severe 148; 483 patient groups). Images came from the fixed hybrid ImageNet mean-background source and labels/splits were not regenerated.", "",
        f"Image source: `{config['data']['image_root']}`. Split source: `{config['data']['master_split']}`. Label provenance: `{config['data']['label_path']}`.", "",
        f"Stage 1 source: `{config['features']['raw_source']}`; Stage 2A root: `{config['features']['stage2a_root']}`; Stage 2B root: `{config['features']['stage2b_root']}`. All were verified by schema, manifest, ID, fold, split role, and SHA256.", "",
        "## Variants and model", "",
        "G0 uses 512 image features; G-Mask uses 512+1; G-Raw/G-A/G-B use 512+6+1. Every model is a fully trainable ImageNet ResNet18 followed only by direct concatenation and one 3-class Linear head (1,539 / 1,542 / 1,560 head parameters).", "",
        "## Training protocol", "",
        "All variants use the same patient-group folds, seed streams, flip-only ImageNet-normalized transforms, fold-specific weighted cross-entropy, AdamW (lr 1e-4, weight decay 1e-4), 50-epoch budget, patience 10, and strict earliest best validation macro-AUC checkpoint.", "",
        "Stage 2A/2B use the matching per-fold train/validation files. Their 500-row OOF tables were audit-only and were never classifier training inputs.", "",
        "Six-dimensional scalers were fit only on each outer training fold (ddof=0); unavailable forehead differences were filled with zero after standardization and the availability mask was appended last.", "",
        "Fixed six-dimensional orders:", "",
        f"- G-Raw: {', '.join(VARIANT_FEATURE_COLUMNS['global_raw'])}",
        f"- G-A: {', '.join(VARIANT_FEATURE_COLUMNS['global_stage2a'])}",
        f"- G-B: {', '.join(VARIANT_FEATURE_COLUMNS['global_stage2b'])}", "",
        "Fold-specific class weights use N_train / (3 × class_count):", "",
        _dataframe_markdown(class_weight_report), "",
        "## Per-fold checkpoint selection and training curves", "",
        _dataframe_markdown(best_epoch_report), "",
        "Per-variant and per-fold training curves are retained under the experiment fold directories and copied summaries are under `reports/global_resnet18_optical_fusion/training_curves/`.", "",
        "## Five-fold mean, sample standard deviation, median, minimum and maximum", "",
        _dataframe_markdown(aggregate_report), "",
        "## Pooled OOF results for all variants", "",
        _dataframe_markdown(oof_metrics_all, main_metric_columns), "",
        "Five-fold mean±sample-std metrics are preserved separately in each variant's `aggregate_fold_metrics.csv`; pooled OOF and fold means are not conflated.", "",
        "## Pairwise fold and OOF changes", "",
        _dataframe_markdown(main_comparison, [
            "candidate", "reference", "metric", "oof_delta_candidate_minus_reference",
            "fold_delta_mean", "fold_delta_sample_std_ddof1", "candidate_better_folds",
            "reference_better_folds", "equal_folds",
        ]), "",
        "## Paired patient-group bootstrap macro-AUC deltas", "",
        _dataframe_markdown(bootstrap_main, [
            "candidate", "reference", "delta_mean", "ci_lower_2_5", "ci_upper_97_5",
            "valid_repetitions",
        ]), "",
        "All comparisons use the same stratified patient-group resample for candidate and reference, retain all images from each selected group, and report negative as well as positive changes.", "",
        "All paired bootstrap metrics and confidence intervals:", "",
        _dataframe_markdown(bootstraps, [
            "candidate", "reference", "metric", "delta_mean", "delta_median",
            "ci_lower_2_5", "ci_upper_97_5", "valid_repetitions",
        ]), "",
        "## Per-class behavior, confusion, and distribution shift", "",
        _dataframe_markdown(per_class_changes, [
            "candidate", "reference", "metric", "oof_delta_candidate_minus_reference",
            "fold_delta_mean", "candidate_better_folds", "reference_better_folds",
        ]), "",
        "All OOF confusion matrices are under `reports/global_resnet18_optical_fusion/confusion_matrices/`. `feature_distribution_audit.csv` retains every fold/feature statistic.", "",
        "Train/validation feature-shift summary:", "",
        _dataframe_markdown(shift_summary), "",
        "Stage 2B shift must be interpreted as an in-sample train versus out-of-sample validation risk and must not trigger post-hoc feature editing.", "",
        "## Overfitting audit", "",
        _dataframe_markdown(overfit_report), "",
        "Train-minus-validation AUC gaps and best-versus-last behavior are descriptive only; outer validation was already used for checkpoint selection.", "",
        "## OOF completeness", "",
        _dataframe_markdown(oof_completeness), "",
        "## Interpretation and limitations", "",
        "Outer validation was used every epoch for early stopping, so estimates may be optimistic and are not independent-test results. The availability-only control must be considered before attributing gains to optical phenotype values.", "",
        "No single metric automatically declares a winner. Candidate selection must consider pooled and fold-average macro-AUC, at least 3/5 fold direction, macro-F1, balanced accuracy, per-class recall, confidence intervals, outlier-fold sensitivity, complexity, and Stage 2B shift. When G-A and G-B are close, the simpler Ridge-calibrated G-A is preferred.", "",
        "## Tests, reproducibility, and integrity", "",
        f"Required test audit: {test_evidence['passed_count']} passed; protocol status: {protocol_evidence['status']}; smoke status: {smoke_test_evidence['status']} ({smoke_test_evidence['completed_variants']}/5 variants).", "",
        "Model/data/augmentation random streams were isolated by fold using base_seed=2026 and fold_seed=base_seed+fold, and were recorded in every checkpoint. Source hashes, scaler hashes, checkpoint hashes, and OOF hashes are recorded in the manifests. Stage 1/2A/2B, labels, splits, and historical image inputs were not modified.", "",
        "## Final candidate guidance", "",
        recommendation,
    ]
    (reports_dir / "global_resnet18_optical_fusion_report.md").write_text(
        "\n".join(report_lines) + "\n", encoding="utf-8"
    )
    formal_report_path = reports_dir / "global_resnet18_optical_fusion_report.md"
    environment_path = experiment_dir / "protocol/environment_audit.json"
    environment = json.loads(environment_path.read_text(encoding="utf-8"))
    code_paths = [
        PROJECT_ROOT / "models/resnet18_optical_fusion.py",
        PROJECT_ROOT / "datasets/global_optical_fusion_dataset.py",
        PROJECT_ROOT / "utils/optical_feature_preprocessor.py",
        PROJECT_ROOT / "trainers/global_optical_fusion_trainer.py",
        PROJECT_ROOT / "evaluators/global_optical_fusion_evaluator.py",
        PROJECT_ROOT / "scripts/train/train_global_optical_fusion_5fold.py",
        PROJECT_ROOT / "scripts/run/run_global_optical_fusion_5fold.py",
        Path(__file__).resolve(),
    ]
    run_manifest = {
        "task": "global_resnet18_optical_fusion", "status": "COMPLETE",
        "variants": list(VARIANTS), "completed_variants": list(VARIANTS),
        "completed_folds": {variant: [0, 1, 2, 3, 4] for variant in VARIANTS},
        "model_architecture": "ResNet18OpticalFusion direct concat + one Linear",
        "backbone": "resnet18", "pretrained_weights": "IMAGENET1K_V1",
        "feature_dimensions": {"global": 512, "global_only_aux": 0, "global_mask_aux": 1, "fusion_aux": 7},
        "parameter_counts": {
            variant: fold_manifests[variant][0]["parameter_count"] for variant in VARIANTS
        },
        "classifier_head_parameter_counts": {
            variant: fold_manifests[variant][0]["classifier_head_parameter_count"] for variant in VARIANTS
        },
        "training_config": config["train"], "class_mapping": {"normal": 0, "mild": 1, "severe": 2},
        "class_counts": config["data"]["expected_class_counts"],
        "split_sha256": fold_manifests["global_only"][0]["split_sha256"],
        "stage1_sha256": sha256_file(_project_path(config["features"]["raw_source"])),
        "stage2a_manifest_sha256": sha256_file(_project_path(config["features"]["stage2a_manifest"])),
        "stage2a_schema_sha256": sha256_file(_project_path(config["features"]["stage2a_schema"])),
        "stage2b_manifest_sha256": sha256_file(_project_path(config["features"]["stage2b_manifest"])),
        "stage2b_schema_sha256": sha256_file(_project_path(config["features"]["stage2b_schema"])),
        "config_sha256": sha256_file(config_path),
        "recorded_fold_implementation_signature": recorded_fold_implementation_signature,
        "current_training_code_signature": training_code_signature(),
        "all_fold_implementation_signatures_identical": True,
        "fold_signature_matches_current_training_code": (
            recorded_fold_implementation_signature == training_code_signature()
        ),
        "summary_code_is_excluded_from_training_signature": True,
        "code_sha256": sha256_json({
            path.relative_to(PROJECT_ROOT).as_posix(): sha256_file(path) for path in code_paths
        }),
        "fold_artifacts": {
            variant: [
                {
                    "fold": item["fold"], "feature_source_sha256": item["feature_source_sha256"],
                    "train_id_sha256": item["train_id_sha256"], "val_id_sha256": item["val_id_sha256"],
                    "feature_scaler_sha256": item["feature_scaler_sha256"],
                    "best_checkpoint_sha256": item["best_checkpoint_sha256"],
                }
                for item in fold_manifests[variant]
            ]
            for variant in VARIANTS
        },
        "variant_oof_sha256": {
            variant: sha256_file(summary_dir / variant / "oof_predictions.csv") for variant in VARIANTS
        },
        "formal_report_sha256": sha256_file(formal_report_path),
        "bootstrap_audits": bootstrap_audits, "seed_rule": "base_seed + fold; isolated model/shuffle/augmentation streams",
        "software_versions": {
            "python": environment.get("python"), "pytorch": environment.get("pytorch"),
            "torchvision": environment.get("torchvision"), "cuda": environment.get("cuda_version"),
        },
        "device": "cuda" if environment.get("cuda_available") else "cpu",
        "git_branch": environment.get("git_branch"), "git_commit": environment.get("git_commit"),
        "tests": {
            "status": test_evidence["status"],
            "passed_count": test_evidence["passed_count"],
            "test_files": test_evidence["test_files"],
            "implementation_signature": test_evidence["implementation_signature"],
            "audit_sha256": sha256_file(experiment_dir / "protocol/test_audit.json"),
        },
        "protocol_status": protocol_evidence["status"],
        "protocol_manifest_sha256": sha256_file(
            experiment_dir / "protocol/protocol_manifest.json"
        ),
        "smoke_test_status": smoke_test_evidence["status"],
        "smoke_test_evidence": smoke_test_evidence,
        "full_training_executed": True, "outer_validation_tuning": True,
        "camera_used": False, "exif_used": False, "clinical_features_used": False,
        "oof_used_as_classifier_train": False, "stage1_modified": False,
        "stage2a_modified": False, "stage2b_modified": False, "split_modified": False,
        "labels_modified": False, "historical_inputs_modified": False,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    _json_dump(run_manifest, summary_dir / "run_manifest.json")
    required_summary_outputs = [
        summary_dir / "oof_metrics_all_variants.csv",
        summary_dir / "foldwise_metric_deltas.csv",
        summary_dir / "pairwise_comparison.csv",
        summary_dir / "pairwise_bootstrap_deltas.csv",
        summary_dir / "feature_distribution_audit.csv",
        summary_dir / "run_manifest.json",
        formal_report_path,
    ]
    for variant in VARIANTS:
        required_summary_outputs.extend([
            summary_dir / variant / "fold_metrics.csv",
            summary_dir / variant / "aggregate_fold_metrics.csv",
            summary_dir / variant / "oof_predictions.csv",
            summary_dir / variant / "oof_metrics.json",
            summary_dir / variant / "oof_confusion_matrix.csv",
            summary_dir / variant / "oof_confusion_matrix.png",
            summary_dir / variant / "training_curve_summary.png",
        ])
    incomplete = [
        str(path) for path in required_summary_outputs
        if not path.is_file() or path.stat().st_size == 0
    ]
    if incomplete:
        raise RuntimeError(f"Formal summary outputs are incomplete: {incomplete}")
    experiment_summary = {
        "status": "COMPLETE", "variants": list(VARIANTS), "completed_fold_runs": 25,
        "oof_rows_per_variant": {variant: len(frame) for variant, frame in variant_oof.items()},
        "bootstrap_audits": bootstrap_audits,
        "outer_validation_tuning": True,
        "protocol_status": protocol_evidence["status"],
        "test_status": test_evidence["status"],
        "smoke_test_status": smoke_test_evidence["status"],
        "formal_report_sha256": sha256_file(formal_report_path),
        "run_manifest_sha256": sha256_file(summary_dir / "run_manifest.json"),
        "interpretation_warning": "Outer validation selected checkpoints each epoch; results are validation, not an independent test.",
    }
    _json_dump(experiment_summary, summary_dir / "experiment_summary.json")
    return summary_dir


def print_completion_summary(summary_dir: Path) -> None:
    metrics = pd.read_csv(summary_dir / "oof_metrics_all_variants.csv")
    comparison = pd.read_csv(summary_dir / "pairwise_comparison.csv")
    bootstrap = pd.read_csv(summary_dir / "pairwise_bootstrap_deltas.csv")
    distributions = pd.read_csv(summary_dir / "feature_distribution_audit.csv")
    run_manifest = json.loads(
        (summary_dir / "run_manifest.json").read_text(encoding="utf-8")
    )
    experiment_summary = json.loads(
        (summary_dir / "experiment_summary.json").read_text(encoding="utf-8")
    )
    print("GLOBAL_OPTICAL_FUSION_EXPERIMENT_STATUS=COMPLETE")
    print(f"COMPLETED_VARIANTS={len(run_manifest['completed_variants'])}")
    print("COMPLETED_FOLD_RUNS=25")
    for variant in VARIANTS:
        row = metrics.loc[metrics["variant"] == variant].iloc[0]
        oof = pd.read_csv(
            summary_dir / variant / "oof_predictions.csv", dtype={"ID": "string"}
        )
        fold_metrics = pd.read_csv(summary_dir / variant / "fold_metrics.csv")
        print(
            f"{variant.upper()}_OOF_ROWS={len(oof)};UNIQUE_IDS={oof['ID'].nunique()}"
        )
        print(
            f"{variant.upper()}_METRICS="
            f"pooled_macro_auc={row['macro_auc']:.6f};"
            f"fold_mean_macro_auc={fold_metrics['macro_auc'].mean():.6f};"
            f"accuracy={row['accuracy']:.6f};"
            f"balanced_accuracy={row['balanced_accuracy']:.6f};"
            f"macro_f1={row['macro_f1']:.6f}"
        )
        print(f"{variant.upper()}_OOF_PATH={summary_dir / variant / 'oof_predictions.csv'}")
    for candidate, reference in PAIRWISE_COMPARISONS:
        row = comparison.loc[
            (comparison["candidate"] == candidate)
            & (comparison["reference"] == reference)
            & (comparison["metric"] == "macro_auc")
        ].iloc[0]
        boot = bootstrap.loc[
            (bootstrap["candidate"] == candidate)
            & (bootstrap["reference"] == reference)
            & (bootstrap["metric"] == "macro_auc")
        ].iloc[0]
        label = f"{candidate.upper()}_MINUS_{reference.upper()}"
        print(
            f"{label}=oof_macro_auc_delta={row['oof_delta_candidate_minus_reference']:.6f};"
            f"candidate_better_folds={int(row['candidate_better_folds'])};"
            f"reference_better_folds={int(row['reference_better_folds'])};"
            f"equal_folds={int(row['equal_folds'])};"
            f"bootstrap_valid_repetitions={int(boot['valid_repetitions'])};"
            f"ci95=[{boot['ci_lower_2_5']:.6f},{boot['ci_upper_97_5']:.6f}]"
        )
    stage2b = distributions.loc[distributions["variant"] == "global_stage2b"].copy()
    stage2b_smd = pd.to_numeric(
        stage2b["standardized_mean_difference"], errors="coerce"
    ).abs().dropna()
    print(
        "STAGE2B_DISTRIBUTION_SHIFT="
        f"mean_absolute_smd={stage2b_smd.mean():.6f};"
        f"maximum_absolute_smd={stage2b_smd.max():.6f}"
    )
    print(
        f"TEST_STATUS={run_manifest['tests']['status']};"
        f"passed={run_manifest['tests']['passed_count']}"
    )
    print(f"PAIRWISE_COMPARISON_PATH={summary_dir / 'pairwise_comparison.csv'}")
    print(f"BOOTSTRAP_PATH={summary_dir / 'pairwise_bootstrap_deltas.csv'}")
    print(
        "FORMAL_REPORT_PATH="
        f"{PROJECT_ROOT / 'reports/global_resnet18_optical_fusion/global_resnet18_optical_fusion_report.md'}"
    )
    print(f"RUN_MANIFEST_PATH={summary_dir / 'run_manifest.json'}")
    print(
        "ALL_ACCEPTANCE_CONDITIONS_MET="
        f"{str(experiment_summary.get('status') == 'COMPLETE').lower()}"
    )


def main() -> None:
    args = parse_args()
    summary = summarize_experiment(
        args.config, args.experiment_dir, expected_rows=args.expected_rows,
        bootstrap_repetitions=args.bootstrap_repetitions,
    )
    print(f"SUMMARY_DIR={summary}")
    print_completion_summary(summary)


if __name__ == "__main__":
    main()
