from __future__ import annotations

import argparse
import json
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from _bootstrap import ROOT, resolve_output_dir
from utils.p1_component_preflight import FIXED_SPLIT_PATH, MANIFEST_PATH, sha256_file
from utils.p1_component_registry import get_component_spec


CORRECTION_VERSION = "P1_PHASE2_SUMMARY_CORRECTION_V1"
EXPERIMENTS = ["p1_a", "p1_n", "p1_l", "p1_s", "p1_r", "p1_rgb_a"]
EXPECTED_COUNTS = {
    "p1_a": (139, 361),
    "p1_n": (130, 370),
    "p1_l": (211, 289),
    "p1_s": (164, 336),
    "p1_r": (149, 351),
    "p1_rgb_a": (103, 397),
}
SUMMARY_FILES = [
    "p1_component_main_results.csv",
    "p1_component_fold_results.csv",
    "p1_component_paired_comparisons.csv",
    "p1_component_status.csv",
    "p1_component_result_matrix.md",
    "p1_component_fold_distribution_summary.csv",
    "p1_phase2_fold_metric_recalculation_audit.csv",
    "p1_component_fold_confusion_matrices.csv",
]
TOL = 1e-10


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment-root", type=Path, default=ROOT / "experiments/500Data/P1_Component_Sweep_v1")
    parser.add_argument("--report", type=Path, default=ROOT / "reports/p1_component_sweep_phase2_results.md")
    parser.add_argument("--audit-only", action="store_true")
    parser.add_argument("--apply-corrections", action="store_true")
    return parser.parse_args()


def _json_safe(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    if isinstance(value, tuple):
        return [_json_safe(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    return value


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_json_safe(payload), ensure_ascii=False, indent=2), encoding="utf-8")


def _git_commit() -> str:
    try:
        result = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True, check=True)
        return result.stdout.strip()
    except Exception:
        return "UNKNOWN"


def _hash_existing(paths: list[Path]) -> dict[str, str | None]:
    return {str(path): (sha256_file(path) if path.is_file() else None) for path in paths}


def _fmt(value: Any, digits: int = 4) -> str:
    if value is None or pd.isna(value):
        return "NA"
    if isinstance(value, (float, np.floating)):
        return f"{float(value):.{digits}f}"
    if isinstance(value, (int, np.integer)) and not isinstance(value, bool):
        return str(int(value))
    return str(value)


def _markdown_table(frame: pd.DataFrame, columns: list[str], digits: int = 4) -> str:
    lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join(["---"] * len(columns)) + " |"]
    for _, row in frame.iterrows():
        lines.append("| " + " | ".join(_fmt(row.get(column), digits) for column in columns) + " |")
    return "\n".join(lines)


def _load_manifest_and_split() -> tuple[pd.DataFrame, pd.DataFrame]:
    manifest = pd.read_csv(MANIFEST_PATH)
    split = pd.read_csv(FIXED_SPLIT_PATH)
    manifest["case_id"] = manifest["case_id"].astype(str)
    manifest["patient_group_id"] = manifest["group_id"].astype(str)
    split["case_id"] = split["ID"].astype(str)
    split["patient_group_id"] = split["patient_group_id"].astype(str)
    return manifest, split


def _prediction_path(experiment_root: Path, experiment: str, fold: int) -> Path:
    visit = experiment_root / experiment / f"fold_{fold}" / "val_predictions_visit.csv"
    if visit.is_file():
        return visit
    case = experiment_root / experiment / f"fold_{fold}" / "val_predictions_case.csv"
    if case.is_file():
        return case
    raise FileNotFoundError(f"missing fold prediction file for {experiment} fold {fold}")


def _compute_fold_metrics(frame: pd.DataFrame) -> dict[str, Any]:
    y_true = frame["label_binary"].astype(int).to_numpy()
    probs = frame[["prob_control", "prob_patient"]].astype(float).to_numpy()
    pred = probs.argmax(axis=1)
    cm = confusion_matrix(y_true, pred, labels=[0, 1])
    tn, fp, fn, tp = [int(x) for x in cm.ravel()]
    return {
        "macro_auc": float(roc_auc_score(y_true, probs[:, 1])),
        "accuracy": float(accuracy_score(y_true, pred)),
        "macro_precision": float(precision_score(y_true, pred, average="macro", zero_division=0)),
        "macro_recall": float(recall_score(y_true, pred, average="macro", zero_division=0)),
        "macro_f1": float(f1_score(y_true, pred, average="macro", zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, pred)),
        "patient_sensitivity": float(tp / (tp + fn)) if (tp + fn) else np.nan,
        "control_specificity": float(tn / (tn + fp)) if (tn + fp) else np.nan,
        "predicted_control_count": int((pred == 0).sum()),
        "predicted_patient_count": int((pred == 1).sum()),
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp": tp,
    }


def _validate_predictions(frame: pd.DataFrame, *, experiment: str, fold: int) -> list[str]:
    errors: list[str] = []
    if len(frame) != 100:
        errors.append(f"{experiment}/fold_{fold}: prediction rows != 100")
    probs = frame[["prob_control", "prob_patient"]].astype(float).to_numpy()
    if not np.isfinite(probs).all():
        errors.append(f"{experiment}/fold_{fold}: non-finite probabilities")
    if (probs < 0).any() or (probs > 1).any():
        errors.append(f"{experiment}/fold_{fold}: probability outside [0, 1]")
    if not np.allclose(probs.sum(axis=1), 1.0, atol=1e-6):
        errors.append(f"{experiment}/fold_{fold}: probabilities do not sum to 1")
    pred = probs.argmax(axis=1)
    if not np.array_equal(frame["pred_binary"].astype(int).to_numpy(), pred):
        errors.append(f"{experiment}/fold_{fold}: pred_binary does not match argmax")
    return errors


def _read_all_predictions(experiment_root: Path) -> tuple[dict[str, pd.DataFrame], list[dict[str, Any]], list[str]]:
    manifest, split = _load_manifest_and_split()
    manifest_ref = manifest[["case_id", "patient_group_id", "fold", "label_original", "label_3class", "label_binary"]].copy()
    split_ref = split[["case_id", "patient_group_id", "fold", "label_3class", "binary_label"]].copy()
    predictions: dict[str, pd.DataFrame] = {}
    integrity_rows: list[dict[str, Any]] = []
    errors: list[str] = []
    expected_cases = set(manifest_ref["case_id"])

    for experiment in EXPERIMENTS:
        fold_frames = []
        fold_counts = []
        for fold in range(5):
            frame = pd.read_csv(_prediction_path(experiment_root, experiment, fold))
            frame["case_id"] = frame["case_id"].astype(str)
            frame["patient_group_id"] = frame["patient_group_id"].astype(str)
            errors.extend(_validate_predictions(frame, experiment=experiment, fold=fold))
            fold_frames.append(frame)
            fold_counts.append(int(len(frame)))
        oof = pd.concat(fold_frames, ignore_index=True).sort_values(["fold", "case_id"]).reset_index(drop=True)
        predictions[experiment] = oof
        merged_manifest = oof.merge(manifest_ref, on="case_id", how="left", suffixes=("", "_manifest"))
        merged_split = oof.merge(split_ref, on="case_id", how="left", suffixes=("", "_split"))
        label_mismatch = int(
            (
                (merged_manifest["label_original"].astype(int) != merged_manifest["label_original_manifest"].astype(int))
                | (merged_manifest["label_3class"].astype(int) != merged_manifest["label_3class_manifest"].astype(int))
                | (merged_manifest["label_binary"].astype(int) != merged_manifest["label_binary_manifest"].astype(int))
            ).sum()
        )
        fold_mismatch = int((merged_manifest["fold"].astype(int) != merged_manifest["fold_manifest"].astype(int)).sum())
        split_fold_mismatch = int((merged_split["fold"].astype(int) != merged_split["fold_split"].astype(int)).sum())
        patient_group_mismatch = int((merged_manifest["patient_group_id"] != merged_manifest["patient_group_id_manifest"]).sum())
        duplicate_case_id = int(oof["case_id"].duplicated().sum())
        missing_case_id = len(expected_cases - set(oof["case_id"]))
        unexpected_case_id = len(set(oof["case_id"]) - expected_cases)
        pred = oof[["prob_control", "prob_patient"]].astype(float).to_numpy().argmax(axis=1)
        predicted_control_count = int((pred == 0).sum())
        predicted_patient_count = int((pred == 1).sum())
        expected_control, expected_patient = EXPECTED_COUNTS[experiment]
        if (predicted_control_count, predicted_patient_count) != (expected_control, expected_patient):
            errors.append(
                f"{experiment}: predicted class counts {(predicted_control_count, predicted_patient_count)} != expected {(expected_control, expected_patient)}"
            )
        integrity_rows.append(
            {
                "experiment": experiment,
                "fold_count": 5,
                "each_fold_prediction_rows": json.dumps(fold_counts),
                "total_oof_rows": int(len(oof)),
                "unique_case_id": int(oof["case_id"].nunique()),
                "duplicate_case_id": duplicate_case_id,
                "missing_case_id": int(missing_case_id),
                "unexpected_case_id": int(unexpected_case_id),
                "label_mismatch": label_mismatch,
                "fold_mismatch": fold_mismatch + split_fold_mismatch,
                "patient_group_mismatch": patient_group_mismatch,
                "predicted_control_count": predicted_control_count,
                "predicted_patient_count": predicted_patient_count,
                "status": "PASSED",
            }
        )
    return predictions, integrity_rows, errors


def _recompute_fold_table(experiment_root: Path, stored_fold: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    audit_rows: list[dict[str, Any]] = []
    corrected_rows: list[dict[str, Any]] = []
    confusion_rows: list[dict[str, Any]] = []
    compare_metrics = [
        "macro_auc",
        "accuracy",
        "macro_precision",
        "macro_recall",
        "macro_f1",
        "balanced_accuracy",
        "patient_sensitivity",
        "control_specificity",
        "predicted_control_count",
        "predicted_patient_count",
    ]
    for experiment in EXPERIMENTS:
        for fold in range(5):
            pred_frame = pd.read_csv(_prediction_path(experiment_root, experiment, fold))
            metrics = _compute_fold_metrics(pred_frame)
            stored = stored_fold[(stored_fold["experiment"] == experiment) & (stored_fold["fold"].astype(int) == fold)]
            if len(stored) != 1:
                raise RuntimeError(f"stored fold row missing/duplicated: {experiment} fold {fold}")
            stored_row = stored.iloc[0].to_dict()
            corrected_row = dict(stored_row)
            for metric in ["macro_precision", "macro_recall", "patient_sensitivity", "control_specificity"]:
                if pd.isna(corrected_row.get(metric)) or str(corrected_row.get(metric)).strip() == "":
                    corrected_row[metric] = metrics[metric]
            corrected_rows.append(corrected_row)
            confusion_rows.append(
                {
                    "experiment": experiment,
                    "fold": fold,
                    "tn": metrics["tn"],
                    "fp": metrics["fp"],
                    "fn": metrics["fn"],
                    "tp": metrics["tp"],
                }
            )
            for metric in compare_metrics:
                stored_value = stored_row.get(metric, np.nan)
                recomputed_value = metrics[metric]
                if metric not in stored_row:
                    status = "ADDED_RECOMPUTED_FIELD" if metric in {"macro_precision", "macro_recall"} else "MISSING_REQUIRES_REVIEW"
                    difference = np.nan
                elif pd.isna(stored_value):
                    status = "KNOWN_P1_L_MISSING_FIELD" if experiment == "p1_l" and metric in {"patient_sensitivity", "control_specificity"} else "MISSING_REQUIRES_REVIEW"
                    difference = np.nan
                else:
                    stored_float = float(stored_value)
                    difference = abs(stored_float - float(recomputed_value))
                    status = "MATCH" if difference <= TOL else "MISMATCH_REQUIRES_REVIEW"
                audit_rows.append(
                    {
                        "experiment": experiment,
                        "fold": fold,
                        "metric": metric,
                        "stored_value": stored_value,
                        "recomputed_value": recomputed_value,
                        "absolute_difference": difference,
                        "status": status,
                    }
                )
    return pd.DataFrame(corrected_rows), pd.DataFrame(confusion_rows), pd.DataFrame(audit_rows)


def _fold_distribution_summary(fold_df: pd.DataFrame, main_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for experiment in EXPERIMENTS:
        subset = fold_df[fold_df["experiment"] == experiment].copy()
        main = main_df[main_df["experiment"] == experiment].iloc[0]
        rows.append(
            {
                "experiment": experiment,
                "completed_folds": int(len(subset)),
                "fold_macro_auc_mean": float(subset["macro_auc"].astype(float).mean()),
                "fold_macro_auc_std": float(subset["macro_auc"].astype(float).std(ddof=1)),
                "fold_macro_auc_min": float(subset["macro_auc"].astype(float).min()),
                "fold_macro_auc_max": float(subset["macro_auc"].astype(float).max()),
                "pooled_oof_macro_auc": float(main["macro_auc"]),
                "pooled_minus_fold_mean_auc": float(main["macro_auc"]) - float(subset["macro_auc"].astype(float).mean()),
                "fold_macro_f1_mean": float(subset["macro_f1"].astype(float).mean()),
                "fold_macro_f1_std": float(subset["macro_f1"].astype(float).std(ddof=1)),
                "pooled_oof_macro_f1": float(main["macro_f1"]),
                "fold_balanced_accuracy_mean": float(subset["balanced_accuracy"].astype(float).mean()),
                "fold_balanced_accuracy_std": float(subset["balanced_accuracy"].astype(float).std(ddof=1)),
                "pooled_oof_balanced_accuracy": float(main["balanced_accuracy"]),
            }
        )
    return pd.DataFrame(rows)


def _paired_integrity(summary_dir: Path) -> tuple[pd.DataFrame, list[str]]:
    paired = pd.read_csv(summary_dir / "p1_component_paired_comparisons.csv")
    errors = []
    for _, row in paired.iterrows():
        if int(row["pair_count"]) != 500 or int(row["matched_rows"]) != 500:
            errors.append(f"{row['experiment']}: paired row count != 500")
        for column in ("label_mismatch", "fold_mismatch", "patient_group_mismatch"):
            if int(row[column]) != 0:
                errors.append(f"{row['experiment']}: {column} != 0")
    return paired, errors


def _group_artifact_audit(experiment_root: Path) -> tuple[list[dict[str, Any]], list[str]]:
    forbidden = {"oof_predictions_group.csv", "metrics_group.json", "confusion_matrix_group.csv"}
    rows = []
    errors = []
    for path in experiment_root.rglob("*"):
        if path.is_file() and path.name in forbidden:
            deprecated = "deprecated" in [part.lower() for part in path.parts]
            rows.append({"path": str(path), "deprecated": deprecated, "status": "ALLOWED_DEPRECATED" if deprecated else "FORBIDDEN_ACTIVE"})
            if not deprecated:
                errors.append(f"active forbidden group artifact: {path}")
    if not rows:
        rows.append({"path": "", "deprecated": False, "status": "NO_GROUP_ARTIFACTS_FOUND"})
    return rows, errors


def _write_report(
    report_path: Path,
    *,
    integrity_df: pd.DataFrame,
    corrected_fold_df: pd.DataFrame,
    confusion_df: pd.DataFrame,
    main_df: pd.DataFrame,
    fold_distribution: pd.DataFrame,
    paired_df: pd.DataFrame,
    audit_df: pd.DataFrame,
) -> None:
    completion = integrity_df[["experiment", "fold_count", "total_oof_rows", "predicted_control_count", "predicted_patient_count"]].copy()
    p1_l = corrected_fold_df[corrected_fold_df["experiment"] == "p1_l"][
        ["experiment", "fold", "patient_sensitivity", "control_specificity"]
    ]
    p1_l_confusion = confusion_df[confusion_df["experiment"] == "p1_l"]
    p1_l_total = p1_l_confusion[["tn", "fp", "fn", "tp"]].sum().to_dict()
    fold_auc = fold_distribution.copy()
    fold_auc["fold_auc_mean_sd"] = fold_auc.apply(
        lambda row: f"{row['fold_macro_auc_mean']:.4f} ± {row['fold_macro_auc_std']:.4f}", axis=1
    )
    fold_auc["fold_auc_range"] = fold_auc.apply(lambda row: f"{row['fold_macro_auc_min']:.4f}-{row['fold_macro_auc_max']:.4f}", axis=1)
    lines = [
        "# P1 phase-two report",
        "",
        "- final_status: P1_COMPONENT_SWEEP_COMPLETE",
        "- integrity_status: P1_PHASE2_RESULTS_INTEGRITY_VERIFIED",
        f"- correction_version: {CORRECTION_VERSION}",
        "- stage_one_status: P1_COMPONENT_FRAMEWORK_READY_FOR_PHASE2",
        "- smoke: 6/6 passed",
        "- formal_folds: 30/30",
        "- P1-Spec: SKIPPED_UNAVAILABLE_BY_FRONTEND",
        "- no_group_aggregation: true",
        "- no_threshold_search: true",
        "- no_hyperparameter_search: true",
        "- no_model_retraining: true",
        "- checkpoints_modified: false",
        "- oof_probabilities_modified: false",
        "",
        "## 1. Correction scope",
        "",
        "This correction audits existing P1 phase-two outputs and fixes summary/report integrity fields only. No model was retrained. No checkpoint was modified. No OOF probability, prediction label, fixed split, or case label was modified.",
        "",
        "## 2. Formal fold completion",
        "",
        _markdown_table(completion, ["experiment", "fold_count", "total_oof_rows", "predicted_control_count", "predicted_patient_count"]),
        "",
        "All six completed experiments contain 500 unique visit/case-level OOF predictions. The previously reported values 139, 211, 130, 149, 103, and 164 represented the total number of predicted Control cases rather than OOF row counts.",
        "",
        "## 3. P1-L fold sensitivity/specificity",
        "",
        _markdown_table(p1_l, ["experiment", "fold", "patient_sensitivity", "control_specificity"]),
        "",
        "## 4. P1-L fold confusion matrices",
        "",
        _markdown_table(p1_l_confusion, ["experiment", "fold", "tn", "fp", "fn", "tp"]),
        "",
        f"P1-L summed confusion matrix: TN={int(p1_l_total['tn'])}, FP={int(p1_l_total['fp'])}, FN={int(p1_l_total['fn'])}, TP={int(p1_l_total['tp'])}. This matches the pooled OOF confusion matrix.",
        "",
        "## 5. Main pooled OOF results",
        "",
        _markdown_table(
            main_df[main_df["status"] == "COMPLETED"],
            ["experiment", "macro_auc", "macro_f1", "balanced_accuracy", "patient_sensitivity", "control_specificity"],
        ),
        "",
        "## 6. Fold distribution and pooled OOF AUC",
        "",
        _markdown_table(
            fold_auc,
            ["experiment", "fold_auc_mean_sd", "fold_auc_range", "pooled_oof_macro_auc", "pooled_minus_fold_mean_auc"],
        ),
        "",
        "Fold Macro-AUC mean ± SD first computes AUC within each 100-case validation fold, then computes the sample mean and sample standard deviation across the five fold AUCs.",
        "",
        "Pooled OOF Macro-AUC concatenates the 500 OOF probabilities produced by five independent models and computes one AUC over all 500 records.",
        "",
        "These two values can differ because each fold uses a different model, fold probability scales can differ, and AUC is not an additive quantity.",
        "",
        "## 7. Pairwise comparison integrity",
        "",
        _markdown_table(paired_df, ["experiment", "pair_count", "matched_rows", "label_mismatch", "fold_mismatch", "patient_group_mismatch"]),
        "",
        "## 8. Fold metric recalculation audit",
        "",
        f"- rows checked: {len(audit_df)}",
        f"- mismatches requiring review: {int((audit_df['status'] == 'MISMATCH_REQUIRES_REVIEW').sum())}",
        f"- known missing P1-L fields corrected: {int((audit_df['status'] == 'KNOWN_P1_L_MISSING_FIELD').sum())}",
        "",
        "## 9. Guardrails",
        "",
        "- No group aggregation result was used.",
        "- No checkpoint was modified.",
        "- No OOF probability was modified.",
        "- No model was retrained.",
        "- Stage A/B/C were not started.",
        "",
        "## 10. Final integrity status",
        "",
        "P1_PHASE2_RESULTS_INTEGRITY_VERIFIED",
        "",
    ]
    report_path.write_text("\n".join(lines), encoding="utf-8")


def _write_integrity_report(
    report_path: Path,
    *,
    integrity_df: pd.DataFrame,
    corrected_fold_df: pd.DataFrame,
    confusion_df: pd.DataFrame,
    main_df: pd.DataFrame,
    fold_distribution: pd.DataFrame,
    paired_df: pd.DataFrame,
    audit_df: pd.DataFrame,
) -> None:
    p1_l = corrected_fold_df[corrected_fold_df["experiment"] == "p1_l"][
        ["fold", "patient_sensitivity", "control_specificity"]
    ]
    p1_l_confusion = confusion_df[confusion_df["experiment"] == "p1_l"]
    p1_l_total = p1_l_confusion[["tn", "fp", "fn", "tp"]].sum()
    lines = [
        "# P1 phase-two result integrity correction report",
        "",
        f"- correction_version: {CORRECTION_VERSION}",
        "- final_status: P1_PHASE2_RESULTS_INTEGRITY_VERIFIED",
        "- training_rerun: false",
        "- checkpoints_modified: false",
        "- oof_probabilities_modified: false",
        "- fold_predictions_modified: false",
        "- labels_modified: false",
        "- fixed_split_modified: false",
        "- stage_a_started: false",
        "- stage_b_started: false",
        "- stage_c_started: false",
        "",
        "## 1. Scope",
        "",
        "The task audited existing phase-two outputs, corrected report row-count wording, filled missing P1-L fold sensitivity/specificity from existing fold predictions, and added fold distribution summaries.",
        "",
        "## 2. OOF integrity",
        "",
        _markdown_table(integrity_df, ["experiment", "fold_count", "total_oof_rows", "unique_case_id", "duplicate_case_id", "missing_case_id", "unexpected_case_id"]),
        "",
        "## 3. Corrected prediction counts",
        "",
        _markdown_table(integrity_df, ["experiment", "total_oof_rows", "predicted_control_count", "predicted_patient_count"]),
        "",
        "## 4. Original oof_rows error cause",
        "",
        "`scripts/p1/rebuild_p1_component_sweep_summary.py` previously aggregated `oof_rows` from `predicted_control_count`. The correction computes OOF rows independently as predicted Control plus predicted Patient count, and displays both prediction counts separately.",
        "",
        "## 5. P1-L recalculation",
        "",
        _markdown_table(p1_l, ["fold", "patient_sensitivity", "control_specificity"]),
        "",
        _markdown_table(p1_l_confusion, ["fold", "tn", "fp", "fn", "tp"]),
        "",
        f"P1-L summed confusion matrix: TN={int(p1_l_total['tn'])}, FP={int(p1_l_total['fp'])}, FN={int(p1_l_total['fn'])}, TP={int(p1_l_total['tp'])}.",
        "",
        "## 6. Fold metric recalculation",
        "",
        f"- checked rows: {len(audit_df)}",
        f"- MATCH: {int((audit_df['status'] == 'MATCH').sum())}",
        f"- KNOWN_P1_L_MISSING_FIELD: {int((audit_df['status'] == 'KNOWN_P1_L_MISSING_FIELD').sum())}",
        f"- MISMATCH_REQUIRES_REVIEW: {int((audit_df['status'] == 'MISMATCH_REQUIRES_REVIEW').sum())}",
        "",
        "## 7. Fold AUC mean±SD and pooled OOF AUC",
        "",
        _markdown_table(fold_distribution, ["experiment", "fold_macro_auc_mean", "fold_macro_auc_std", "pooled_oof_macro_auc", "pooled_minus_fold_mean_auc"]),
        "",
        "Fold Macro-AUC mean ± SD is computed from five per-fold AUC values. Pooled OOF Macro-AUC is computed once from all 500 OOF predictions. They are both reported and not substituted for each other.",
        "",
        "## 8. Pairwise comparison integrity",
        "",
        _markdown_table(paired_df, ["experiment", "pair_count", "matched_rows", "label_mismatch", "fold_mismatch", "patient_group_mismatch"]),
        "",
        "## 9. Group aggregation",
        "",
        "No active group-level OOF, metrics, or confusion-matrix artifact was used by the corrected summary.",
        "",
        "## 10. Final status",
        "",
        "P1_PHASE2_RESULTS_INTEGRITY_VERIFIED",
        "",
    ]
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(lines), encoding="utf-8")


def audit_and_optionally_correct(experiment_root: Path, report_path: Path, *, apply: bool) -> dict[str, Any]:
    experiment_root = resolve_output_dir(experiment_root)
    report_path = report_path if report_path.is_absolute() else ROOT / report_path
    summary_dir = experiment_root / "summary"
    metadata_dir = experiment_root / "metadata"
    report_archive = ROOT / "reports" / "archive" / "p1_component_sweep_phase2_results_before_integrity_correction.md"
    integrity_report = ROOT / "reports" / "p1_phase2_result_integrity_correction_report.md"

    mutable_paths = [summary_dir / name for name in SUMMARY_FILES] + [
        report_path,
        integrity_report,
        report_archive,
    ]
    before_hashes = _hash_existing(mutable_paths)
    correction_path = metadata_dir / "p1_phase2_summary_integrity_correction.json"
    if correction_path.is_file():
        previous_record = _read_json(correction_path)
        previous_before = previous_record.get("source_file_hashes_before_correction")
        if isinstance(previous_before, dict) and previous_record.get("correction_version") == CORRECTION_VERSION:
            tracked_keys = {str(path) for path in mutable_paths}
            before_hashes = {key: value for key, value in previous_before.items() if key in tracked_keys}
    main_df = pd.read_csv(summary_dir / "p1_component_main_results.csv")
    stored_fold_df = pd.read_csv(summary_dir / "p1_component_fold_results.csv")
    paired_df, pair_errors = _paired_integrity(summary_dir)
    predictions, integrity_rows, prediction_errors = _read_all_predictions(experiment_root)
    corrected_fold_df, confusion_df, audit_df = _recompute_fold_table(experiment_root, stored_fold_df)
    fold_distribution = _fold_distribution_summary(corrected_fold_df, main_df)
    group_rows, group_errors = _group_artifact_audit(experiment_root)
    integrity_df = pd.DataFrame(integrity_rows)

    p1_l_confusion_sum = confusion_df[confusion_df["experiment"] == "p1_l"][["tn", "fp", "fn", "tp"]].sum()
    p1_l_oof = _read_json(experiment_root / "p1_l" / "oof" / "oof_metrics_visit.json")
    p1_l_oof_cm = np.asarray(p1_l_oof["confusion_matrix"], dtype=int)
    p1_l_confusion_matches = bool(
        int(p1_l_confusion_sum["tn"]) == int(p1_l_oof_cm[0, 0])
        and int(p1_l_confusion_sum["fp"]) == int(p1_l_oof_cm[0, 1])
        and int(p1_l_confusion_sum["fn"]) == int(p1_l_oof_cm[1, 0])
        and int(p1_l_confusion_sum["tp"]) == int(p1_l_oof_cm[1, 1])
    )
    if not p1_l_confusion_matches:
        prediction_errors.append("P1-L fold confusion matrix sum does not match pooled OOF confusion matrix")

    audit_bad = audit_df[audit_df["status"].isin(["MISMATCH_REQUIRES_REVIEW", "MISSING_REQUIRES_REVIEW"])]
    known_missing = audit_df[audit_df["status"] == "KNOWN_P1_L_MISSING_FIELD"]
    errors = prediction_errors + pair_errors + group_errors
    if not audit_bad.empty:
        errors.append("fold metric recalculation found mismatches requiring review")
    if len(known_missing) not in {0, 10}:
        errors.append(f"expected 0 or 10 known P1-L missing fold sensitivity/specificity fields, found {len(known_missing)}")
    if not all(integrity_df["total_oof_rows"].astype(int) == 500):
        errors.append("at least one experiment does not have 500 OOF rows")
    if not all(integrity_df["unique_case_id"].astype(int) == 500):
        errors.append("at least one experiment does not have 500 unique case_id")

    status = "P1_PHASE2_RESULTS_INTEGRITY_VERIFIED" if not errors else "BLOCKED_BY_P1_PHASE2_RESULT_INTEGRITY_ERROR"

    if apply:
        if status != "P1_PHASE2_RESULTS_INTEGRITY_VERIFIED":
            raise RuntimeError("; ".join(errors))
        report_archive.parent.mkdir(parents=True, exist_ok=True)
        if report_path.is_file() and not report_archive.is_file():
            shutil.copy2(report_path, report_archive)
        corrected_fold_df.to_csv(summary_dir / "p1_component_fold_results.csv", index=False, encoding="utf-8-sig")
        confusion_df.to_csv(summary_dir / "p1_component_fold_confusion_matrices.csv", index=False, encoding="utf-8-sig")
        audit_df.to_csv(summary_dir / "p1_phase2_fold_metric_recalculation_audit.csv", index=False, encoding="utf-8-sig")
        fold_distribution.to_csv(summary_dir / "p1_component_fold_distribution_summary.csv", index=False, encoding="utf-8-sig")

        status_df = pd.read_csv(summary_dir / "p1_component_status.csv")
        status_df["result_integrity_status"] = "VERIFIED"
        status_df["summary_correction_version"] = CORRECTION_VERSION
        status_df.to_csv(summary_dir / "p1_component_status.csv", index=False, encoding="utf-8-sig")

        _write_report(
            report_path,
            integrity_df=integrity_df,
            corrected_fold_df=corrected_fold_df,
            confusion_df=confusion_df,
            main_df=main_df,
            fold_distribution=fold_distribution,
            paired_df=paired_df,
            audit_df=audit_df,
        )
        _write_integrity_report(
            integrity_report,
            integrity_df=integrity_df,
            corrected_fold_df=corrected_fold_df,
            confusion_df=confusion_df,
            main_df=main_df,
            fold_distribution=fold_distribution,
            paired_df=paired_df,
            audit_df=audit_df,
        )
        after_hashes = _hash_existing(mutable_paths)
        correction_record = {
            "correction_version": CORRECTION_VERSION,
            "training_rerun": False,
            "checkpoints_modified": False,
            "oof_probabilities_modified": False,
            "fold_predictions_modified": False,
            "labels_modified": False,
            "fixed_split_modified": False,
            "corrected_report_oof_rows": True,
            "corrected_p1_l_fold_sensitivity_specificity": True,
            "added_fold_mean_sd_summary": True,
            "stage_a_started": False,
            "stage_b_started": False,
            "stage_c_started": False,
            "source_file_hashes_before_correction": before_hashes,
            "source_file_hashes_after_correction": after_hashes,
            "correction_timestamp": datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds"),
            "git_commit": _git_commit(),
            "final_status": status,
        }
        _write_json(correction_path, correction_record)
    else:
        after_hashes = _hash_existing(mutable_paths)

    result = {
        "final_status": status,
        "apply_corrections": bool(apply),
        "errors": errors,
        "oof_integrity": integrity_rows,
        "p1_l_fold_sensitivity_specificity": corrected_fold_df[corrected_fold_df["experiment"] == "p1_l"][
            ["fold", "patient_sensitivity", "control_specificity"]
        ].to_dict(orient="records"),
        "p1_l_fold_confusion": confusion_df[confusion_df["experiment"] == "p1_l"].to_dict(orient="records"),
        "p1_l_confusion_matches_oof": p1_l_confusion_matches,
        "fold_metric_audit_status_counts": audit_df["status"].value_counts().to_dict(),
        "fold_distribution_summary": fold_distribution.to_dict(orient="records"),
        "paired_integrity_rows": paired_df[["experiment", "pair_count", "matched_rows", "label_mismatch", "fold_mismatch", "patient_group_mismatch"]].to_dict(orient="records"),
        "group_artifact_audit": group_rows,
        "source_file_hashes_before_correction": before_hashes,
        "source_file_hashes_after_correction": after_hashes,
    }
    return result


def main() -> None:
    args = parse_args()
    if args.audit_only == args.apply_corrections:
        raise SystemExit("Specify exactly one of --audit-only or --apply-corrections")
    result = audit_and_optionally_correct(args.experiment_root, args.report, apply=bool(args.apply_corrections))
    print(json.dumps(_json_safe(result), ensure_ascii=False, indent=2))
    if result["final_status"] != "P1_PHASE2_RESULTS_INTEGRITY_VERIFIED":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
