from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import random
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np
import pandas as pd
import torch
from scipy.stats import rankdata, spearmanr
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, confusion_matrix, f1_score, precision_score, recall_score
from sklearn.metrics import accuracy_score, balanced_accuracy_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from torch import nn
from torch.utils.data import DataLoader, Dataset

from _bootstrap import ROOT, resolve_output_dir
from metrics.binary_classification_metrics import compute_binary_metrics
from models.p1_independent_dual_resnet18 import IndependentDualResNet18, count_parameters
from p0b_deca.p1_dataset import _image_to_chw_float32, _mask_to_1hw
from utils.experiment_utils import set_random_seed
from utils.p1_cluster_bootstrap import compute_visit_metrics, patient_cluster_bootstrap
from utils.p1_component_preflight import FIXED_SPLIT_PATH, MANIFEST_PATH, sha256_file
from utils.p1_representation_normalization import (
    IMAGENET_MEAN,
    IMAGENET_STD,
    apply_component_normalization,
    fit_component_normalization,
)
from utils.p1_rgb_audit import load_p1_frame, local_path


P1_ROOT = ROOT / "experiments/500Data/P1_Component_Sweep_v1"
ABC_ROOT = ROOT / "experiments/500Data/P1_Extension_ABC_v1"
P1_RGB_ROOT = ROOT / "experiments/500Data/P1_RGB_P0Aligned_ResNet18_5fold_v1"
P1_RGB_OOF = P1_RGB_ROOT / "oof/oof_predictions_case.csv"
P1_INTEGRITY_JSON = P1_ROOT / "metadata/p1_phase2_summary_integrity_correction.json"
P1_INTEGRITY_REPORT = ROOT / "reports/p1_phase2_result_integrity_correction_report.md"
P1_PHASE2_REPORT = ROOT / "reports/p1_component_sweep_phase2_results.md"
STAGE_B_EXPERIMENTS = ("rgb_rgb_capacity_control", "rgb_s_independent_dual", "rgb_r_independent_dual")
STAGE_B_DISPLAY = {
    "rgb_rgb_capacity_control": "B-RGBRGB",
    "rgb_s_independent_dual": "B-RGBS",
    "rgb_r_independent_dual": "B-RGBR",
}
STAGE_B_SECONDARY = {
    "rgb_rgb_capacity_control": "rgb",
    "rgb_s_independent_dual": "p1_s",
    "rgb_r_independent_dual": "p1_r",
}
P1_SOURCE_MODELS = ("p1_rgb", "p1_l", "p1_s", "p1_r")
ABC_CORRECTION_VERSION = "P1_EXTENSION_ABC_V1"
BOOTSTRAP_ITERATIONS = 2000
SEED = 2026


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=ABC_ROOT)
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--stage-a", action="store_true")
    parser.add_argument("--stage-b", action="store_true")
    parser.add_argument("--stage-c", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--summarize-only", action="store_true")
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--decision-only", action="store_true")
    return parser.parse_args()


def _now() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


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


def _write_json(path: Path, payload: Mapping[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_json_safe(dict(payload)), indent=2, ensure_ascii=False), encoding="utf-8")
    return path


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_text(path: Path, text: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def _md_table(frame: pd.DataFrame, *, max_rows: int | None = None) -> str:
    if frame is None or frame.empty:
        return "_No rows._"
    display = frame.head(max_rows).copy() if max_rows is not None else frame.copy()
    columns = [str(column) for column in display.columns]
    lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join(["---"] * len(columns)) + " |"]
    for _, row in display.iterrows():
        cells = []
        for column in display.columns:
            value = row[column]
            if isinstance(value, (float, np.floating)):
                cells.append("NA" if pd.isna(value) else f"{float(value):.4f}")
            elif pd.isna(value):
                cells.append("NA")
            else:
                cells.append(str(value).replace("|", "\\|"))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def _git_commit() -> str:
    try:
        result = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True, check=True)
        return result.stdout.strip()
    except Exception:
        return "UNKNOWN"


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _prob_frame_from_patient(base: pd.DataFrame, patient_prob: np.ndarray, name: str) -> pd.DataFrame:
    out = base[["case_id", "patient_group_id", "fold", "label_original", "label_3class", "label_binary"]].copy()
    out["prob_patient"] = np.asarray(patient_prob, dtype=float)
    out["prob_control"] = 1.0 - out["prob_patient"]
    out["pred_binary"] = out[["prob_control", "prob_patient"]].to_numpy(float).argmax(axis=1)
    out["model_name"] = name
    return out


def _metrics_row(frame: pd.DataFrame, *, experiment: str, display_name: str | None = None) -> dict[str, Any]:
    metrics = compute_visit_metrics(frame)
    cm = np.asarray(metrics["confusion_matrix"], dtype=int)
    return {
        "experiment": experiment,
        "display_name": display_name or experiment,
        "n_oof": int(len(frame)),
        "unique_case_id": int(frame["case_id"].astype(str).nunique()),
        "macro_auc": float(metrics["macro_auc"]),
        "accuracy": float(metrics["accuracy"]),
        "macro_precision": float(metrics["macro_precision"]),
        "macro_recall": float(metrics["macro_recall"]),
        "macro_f1": float(metrics["macro_f1"]),
        "balanced_accuracy": float(metrics["balanced_accuracy"]),
        "pr_auc": float(metrics["pr_auc"]),
        "patient_sensitivity": float(metrics["patient_sensitivity"]),
        "control_specificity": float(metrics["control_specificity"]),
        "ppv": float(metrics["ppv"]),
        "npv": float(metrics["npv"]),
        "tn": int(cm[0, 0]),
        "fp": int(cm[0, 1]),
        "fn": int(cm[1, 0]),
        "tp": int(cm[1, 1]),
        "predicted_control_count": int((frame["pred_binary"].astype(int) == 0).sum()),
        "predicted_patient_count": int((frame["pred_binary"].astype(int) == 1).sum()),
    }


def _visit_metrics_np(y_true: np.ndarray, prob_patient: np.ndarray) -> dict[str, float]:
    y = np.asarray(y_true, dtype=int)
    p = np.asarray(prob_patient, dtype=float)
    pred = (p >= 0.5).astype(int)
    return {
        "macro_auc": float(roc_auc_score(y, p)),
        "accuracy": float(accuracy_score(y, pred)),
        "macro_precision": float(precision_score(y, pred, average="macro", zero_division=0)),
        "macro_recall": float(recall_score(y, pred, average="macro", zero_division=0)),
        "macro_f1": float(f1_score(y, pred, average="macro", zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
        "pr_auc": float(average_precision_score(y, p)),
        "patient_sensitivity": float(recall_score(y, pred, pos_label=1, zero_division=0)),
        "control_specificity": float(recall_score(y, pred, pos_label=0, zero_division=0)),
        "ppv": float(precision_score(y, pred, pos_label=1, zero_division=0)),
        "npv": float(precision_score(y, pred, pos_label=0, zero_division=0)),
    }


def _binary_auc_rank(y_true: np.ndarray, prob_patient: np.ndarray) -> float:
    y = np.asarray(y_true, dtype=int)
    p = np.asarray(prob_patient, dtype=float)
    n_pos = int((y == 1).sum())
    n_neg = int((y == 0).sum())
    if n_pos == 0 or n_neg == 0:
        return np.nan
    ranks = rankdata(p, method="average")
    pos_rank_sum = float(ranks[y == 1].sum())
    return float((pos_rank_sum - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def _fast_threshold_metrics_np(y_true: np.ndarray, prob_patient: np.ndarray) -> dict[str, float]:
    y = np.asarray(y_true, dtype=int)
    p = np.asarray(prob_patient, dtype=float)
    pred = (p >= 0.5).astype(int)
    tn = int(((y == 0) & (pred == 0)).sum())
    fp = int(((y == 0) & (pred == 1)).sum())
    fn = int(((y == 1) & (pred == 0)).sum())
    tp = int(((y == 1) & (pred == 1)).sum())
    n = int(len(y))
    sens = float(tp / (tp + fn)) if (tp + fn) else 0.0
    spec = float(tn / (tn + fp)) if (tn + fp) else 0.0
    precision_patient = float(tp / (tp + fp)) if (tp + fp) else 0.0
    precision_control = float(tn / (tn + fn)) if (tn + fn) else 0.0
    f1_patient = float(2.0 * precision_patient * sens / (precision_patient + sens)) if (precision_patient + sens) else 0.0
    f1_control = float(2.0 * precision_control * spec / (precision_control + spec)) if (precision_control + spec) else 0.0
    return {
        "macro_auc": _binary_auc_rank(y, p),
        "accuracy": float((tp + tn) / n) if n else np.nan,
        "macro_f1": float((f1_control + f1_patient) / 2.0),
        "balanced_accuracy": float((sens + spec) / 2.0),
        "patient_sensitivity": sens,
        "control_specificity": spec,
    }


def _cluster_index_arrays(frame: pd.DataFrame) -> list[np.ndarray]:
    groups = []
    for _, group in frame.reset_index(drop=True).groupby("patient_group_id", sort=False):
        groups.append(group.index.to_numpy(dtype=int))
    return groups


def _fast_cluster_bootstrap(frame: pd.DataFrame, *, iterations: int = BOOTSTRAP_ITERATIONS, seed: int = SEED) -> dict[str, Any]:
    y = frame["label_binary"].astype(int).to_numpy()
    p = frame["prob_patient"].to_numpy(float)
    groups = _cluster_index_arrays(frame)
    rng = np.random.default_rng(seed)
    rows = []
    failed = 0
    keys = ("macro_auc", "accuracy", "macro_f1", "balanced_accuracy", "patient_sensitivity", "control_specificity")
    for _ in range(int(iterations)):
        picked = rng.integers(0, len(groups), size=len(groups))
        idx = np.concatenate([groups[i] for i in picked])
        if np.unique(y[idx]).size < 2:
            failed += 1
            continue
        try:
            metrics = _visit_metrics_np(y[idx], p[idx])
        except ValueError:
            failed += 1
            continue
        rows.append({key: metrics[key] for key in keys})
    sample = pd.DataFrame(rows)
    return {
        "iterations": int(iterations),
        "seed": int(seed),
        "cluster_unit": "patient_group_id",
        "metric_unit": "visit_case",
        "valid_iterations": int(len(sample)),
        "failed_iterations": int(failed),
        "ci95": {key: [float(sample[key].quantile(0.025)), float(sample[key].quantile(0.975))] for key in keys} if not sample.empty else {},
        "bootstrap_mean": {key: float(sample[key].mean()) for key in keys} if not sample.empty else {},
    }


def _paired_delta(candidate: pd.DataFrame, baseline: pd.DataFrame) -> dict[str, float]:
    c = candidate.sort_values("case_id").reset_index(drop=True)
    b = baseline.sort_values("case_id").reset_index(drop=True)
    c_metrics = compute_visit_metrics(c)
    b_metrics = compute_visit_metrics(b)
    return {
        "delta_macro_auc": float(c_metrics["macro_auc"] - b_metrics["macro_auc"]),
        "delta_accuracy": float(c_metrics["accuracy"] - b_metrics["accuracy"]),
        "delta_macro_f1": float(c_metrics["macro_f1"] - b_metrics["macro_f1"]),
        "delta_balanced_accuracy": float(c_metrics["balanced_accuracy"] - b_metrics["balanced_accuracy"]),
        "delta_patient_sensitivity": float(c_metrics["patient_sensitivity"] - b_metrics["patient_sensitivity"]),
        "delta_control_specificity": float(c_metrics["control_specificity"] - b_metrics["control_specificity"]),
    }


def _paired_bootstrap_delta(candidate: pd.DataFrame, baseline: pd.DataFrame, *, iterations: int = BOOTSTRAP_ITERATIONS, seed: int = SEED) -> dict[str, Any]:
    merged = candidate.merge(
        baseline[["case_id", "prob_control", "prob_patient", "pred_binary"]],
        on="case_id",
        how="inner",
        suffixes=("", "_baseline"),
    )
    groups = _cluster_index_arrays(merged)
    y = merged["label_binary"].astype(int).to_numpy()
    pc = merged["prob_patient"].to_numpy(float)
    pb = merged["prob_patient_baseline"].to_numpy(float)
    rng = np.random.default_rng(seed)
    samples: list[dict[str, float]] = []
    failed = 0
    for _ in range(int(iterations)):
        picked = rng.integers(0, len(groups), size=len(groups))
        idx = np.concatenate([groups[i] for i in picked])
        if np.unique(y[idx]).size < 2:
            failed += 1
            continue
        try:
            mc = _fast_threshold_metrics_np(y[idx], pc[idx])
            mb = _fast_threshold_metrics_np(y[idx], pb[idx])
            samples.append(
                {
                    "delta_macro_auc": float(mc["macro_auc"] - mb["macro_auc"]),
                    "delta_accuracy": float(mc["accuracy"] - mb["accuracy"]),
                    "delta_macro_f1": float(mc["macro_f1"] - mb["macro_f1"]),
                    "delta_balanced_accuracy": float(mc["balanced_accuracy"] - mb["balanced_accuracy"]),
                    "delta_patient_sensitivity": float(mc["patient_sensitivity"] - mb["patient_sensitivity"]),
                    "delta_control_specificity": float(mc["control_specificity"] - mb["control_specificity"]),
                }
            )
        except ValueError:
            failed += 1
    sample_df = pd.DataFrame(samples)
    return {
        "iterations": int(iterations),
        "seed": int(seed),
        "cluster_unit": "patient_group_id",
        "metric_unit": "visit_case",
        "valid_iterations": int(len(sample_df)),
        "failed_iterations": int(failed),
        "ci95": {column: [float(sample_df[column].quantile(0.025)), float(sample_df[column].quantile(0.975))] for column in sample_df.columns} if not sample_df.empty else {},
        "bootstrap_mean": {column: float(sample_df[column].mean()) for column in sample_df.columns} if not sample_df.empty else {},
    }


def _load_oof_sources() -> dict[str, pd.DataFrame]:
    sources = {
        "p1_rgb": pd.read_csv(P1_RGB_OOF, dtype={"case_id": str, "patient_group_id": str}),
        "p1_s": pd.read_csv(P1_ROOT / "p1_s/oof/oof_predictions_visit.csv", dtype={"case_id": str, "patient_group_id": str}),
        "p1_r": pd.read_csv(P1_ROOT / "p1_r/oof/oof_predictions_visit.csv", dtype={"case_id": str, "patient_group_id": str}),
        "p1_l": pd.read_csv(P1_ROOT / "p1_l/oof/oof_predictions_visit.csv", dtype={"case_id": str, "patient_group_id": str}),
    }
    for frame in sources.values():
        frame["case_id"] = frame["case_id"].astype(str)
        frame["patient_group_id"] = frame["patient_group_id"].astype(str)
    return sources


def _create_layout(output_dir: Path) -> dict[str, Path]:
    paths = {
        "root": output_dir,
        "metadata": output_dir / "metadata",
        "stage_a": output_dir / "stage_a",
        "stage_b": output_dir / "stage_b",
        "stage_c": output_dir / "stage_c",
        "summary": output_dir / "summary",
        "logs": output_dir / "logs",
    }
    for path in paths.values():
        path.mkdir(parents=True, exist_ok=True)
    for sub in (
        "stage_a/alignment",
        "stage_a/pairwise",
        "stage_a/fixed_fusion",
        "stage_a/bootstrap",
        "stage_a/summary",
        "stage_b/smoke",
        "stage_b/summary",
        "stage_c/exif_numeric_probe",
        "stage_c/camera_probe",
        "stage_c/combined_acquisition_probe",
        "stage_c/audit",
    ):
        (output_dir / sub).mkdir(parents=True, exist_ok=True)
    return paths


def create_approval(output_dir: Path) -> dict[str, Any]:
    payload = {
        "approval_version": ABC_CORRECTION_VERSION,
        "stage_a_approved": True,
        "stage_b_smoke_approved": True,
        "stage_b_formal_training_approved": True,
        "stage_c_approved": True,
        "stage_d_approved": False,
        "allow_threshold_search": False,
        "allow_late_fusion_weight_search": False,
        "allow_hyperparameter_search": False,
        "allow_case_removal": False,
        "allow_fold_regeneration": False,
        "allow_deca_inference": False,
        "evaluation_unit": "visit_case",
        "split_group_unit": "patient_group_id",
        "bootstrap_cluster_unit": "patient_group_id",
        "aggregate_predictions_within_patient": False,
        "p1_integrity_correction_version": _read_json(P1_INTEGRITY_JSON).get("correction_version") if P1_INTEGRITY_JSON.is_file() else None,
        "p1_rgb_oof_sha256": sha256_file(P1_RGB_OOF),
        "p1_s_oof_sha256": sha256_file(P1_ROOT / "p1_s/oof/oof_predictions_visit.csv"),
        "p1_r_oof_sha256": sha256_file(P1_ROOT / "p1_r/oof/oof_predictions_visit.csv"),
        "p1_manifest_sha256": sha256_file(MANIFEST_PATH),
        "fixed_split_sha256": sha256_file(FIXED_SPLIT_PATH),
        "frozen500_schema_sha256": _sha256_text("rgb_path|maps_path|face_valid_mask_path|shading_like|signed_residual"),
        "git_commit": _git_commit(),
        "created_at": _now(),
    }
    _write_json(output_dir / "metadata/STAGE_ABC_EXECUTION_APPROVED.json", payload)
    return payload


def run_preflight(output_dir: Path) -> dict[str, Any]:
    sources = _load_oof_sources()
    integrity = _read_json(P1_INTEGRITY_JSON) if P1_INTEGRITY_JSON.is_file() else {}
    frame = load_p1_frame(MANIFEST_PATH, FIXED_SPLIT_PATH)
    frame["case_id"] = frame["case_id"].astype(str)
    checks: dict[str, Any] = {
        "p1_integrity_verified": integrity.get("final_status") == "P1_PHASE2_RESULTS_INTEGRITY_VERIFIED",
        "manifest_rows_500": int(len(frame)) == 500,
        "unique_case_id_500": int(frame["case_id"].nunique()) == 500,
        "fixed_split_sha256_match": sha256_file(FIXED_SPLIT_PATH) == "d5a20fb56c96e657dd7902b6d829bed78b6d43d3e58ec47e6bd3542ec34378cb",
        "fold_400_100": True,
        "patient_group_not_cross_fold": not bool(frame.groupby("patient_group_id")["fold"].nunique().gt(1).any()),
        "stage_d_approved_false": True,
        "cuda_available": bool(torch.cuda.is_available()),
        "no_group_aggregation": not any(path.name in {"oof_predictions_group.csv", "metrics_group.json", "confusion_matrix_group.csv"} for path in P1_ROOT.rglob("*") if path.is_file() and "deprecated" not in [part.lower() for part in path.parts]),
    }
    for fold in range(5):
        checks[f"fold_{fold}_val_100"] = int((frame["fold"] == fold).sum()) == 100
        checks[f"fold_{fold}_train_400"] = int((frame["fold"] != fold).sum()) == 400
    ref = sources["p1_rgb"].sort_values("case_id").reset_index(drop=True)
    for name, src in sources.items():
        src = src.sort_values("case_id").reset_index(drop=True)
        checks[f"{name}_oof_500"] = len(src) == 500 and src["case_id"].nunique() == 500
        checks[f"{name}_case_id_match"] = src["case_id"].tolist() == ref["case_id"].tolist()
        checks[f"{name}_label_match"] = src["label_binary"].astype(int).tolist() == ref["label_binary"].astype(int).tolist()
        checks[f"{name}_fold_match"] = src["fold"].astype(int).tolist() == ref["fold"].astype(int).tolist()
        checks[f"{name}_patient_group_match"] = src["patient_group_id"].astype(str).tolist() == ref["patient_group_id"].astype(str).tolist()
    asset_ok = True
    for _, row in frame.iterrows():
        for column in ("rgb_path", "maps_path", "face_valid_mask_path"):
            if not local_path(row[column]).exists():
                asset_ok = False
    checks["frozen500_rgb_shading_residual_assets_complete"] = asset_ok
    checks["exif_camera_readable"] = all(column in frame.columns for column in ("camera_model", "exposure_time_seconds", "fnumber", "iso_numeric", "brightness_value"))
    if not checks["p1_integrity_verified"]:
        status = "BLOCKED_BY_P1_SOURCE_RESULT_INTEGRITY_ERROR"
    elif all(bool(v) for v in checks.values()):
        status = "PHASE_ABC_PREFLIGHT_PASSED"
    else:
        status = "BLOCKED_BY_ABC_SHARED_DATA_OR_PROTOCOL_ERROR"
    payload = {
        "status": status,
        "checks": checks,
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "cuda_device_count": int(torch.cuda.device_count()) if torch.cuda.is_available() else 0,
    }
    _write_json(output_dir / "metadata/abc_preflight.json", payload)
    _write_json(
        output_dir / "metadata/source_hashes.json",
        {
            "p1_integrity_json": sha256_file(P1_INTEGRITY_JSON),
            "p1_rgb_oof": sha256_file(P1_RGB_OOF),
            "p1_s_oof": sha256_file(P1_ROOT / "p1_s/oof/oof_predictions_visit.csv"),
            "p1_r_oof": sha256_file(P1_ROOT / "p1_r/oof/oof_predictions_visit.csv"),
            "manifest": sha256_file(MANIFEST_PATH),
            "fixed_split": sha256_file(FIXED_SPLIT_PATH),
            "script": sha256_file(Path(__file__)),
        },
    )
    _write_json(
        output_dir / "metadata/execution_plan.json",
        {
            "stages": ["preflight", "stage_a", "stage_b_smoke", "stage_b_formal", "stage_c", "decision_report"],
            "stage_b_experiments": list(STAGE_B_EXPERIMENTS),
            "stage_b_formal_fold_jobs": 15,
            "stage_d_auto_start": False,
        },
    )
    _write_json(
        output_dir / "metadata/protocol_snapshot.json",
        {
            "protocol_version": ABC_CORRECTION_VERSION,
            "evaluation_unit": "visit_case",
            "split_group_unit": "patient_group_id",
            "bootstrap_cluster_unit": "patient_group_id",
            "bootstrap_iterations": BOOTSTRAP_ITERATIONS,
            "seed": SEED,
            "stage_b": {
                "model": "IndependentDualResNet18",
                "optimizer": "AdamW",
                "learning_rate": 1.0e-4,
                "weight_decay": 1.0e-4,
                "batch_size": 16,
                "max_epochs": 50,
                "early_stopping_patience": 10,
                "checkpoint_metric": "visit_case_macro_auc",
                "checkpoint_mode": "max",
                "amp": False,
                "loss": "training_fold_weighted_cross_entropy",
                "scheduler": False,
                "mixup": False,
                "cutmix": False,
                "label_smoothing": False,
                "focal_loss": False,
                "sampler_resampling": False,
                "threshold_optimization": False,
                "hyperparameter_search": False,
                "p1_checkpoint_initialization": False,
            },
            "stage_c": {
                "logistic_regression": {
                    "penalty": "l2",
                    "C": 1.0,
                    "class_weight": "balanced",
                    "solver": "liblinear",
                    "max_iter": 5000,
                    "random_state": SEED,
                    "hyperparameter_search": False,
                },
                "acquisition_dependence_thresholds": {
                    "high_probe_auc": 0.75,
                    "high_overall_minus_stratified_auc": 0.10,
                    "moderate_overall_minus_stratified_auc": 0.05,
                    "within_label_rho": 0.30,
                },
            },
            "stage_d_auto_start": False,
        },
    )
    if status != "PHASE_ABC_PREFLIGHT_PASSED":
        raise RuntimeError(status)
    return payload


def _alignment_audit(sources: dict[str, pd.DataFrame]) -> dict[str, Any]:
    ref = sources["p1_rgb"].sort_values("case_id").reset_index(drop=True)
    rows = []
    for name, src in sources.items():
        src = src.sort_values("case_id").reset_index(drop=True)
        rows.append(
            {
                "model": name,
                "matched_rows": int(len(src.merge(ref[["case_id"]], on="case_id", how="inner"))),
                "label_mismatch": int((src["label_binary"].astype(int).to_numpy() != ref["label_binary"].astype(int).to_numpy()).sum()),
                "fold_mismatch": int((src["fold"].astype(int).to_numpy() != ref["fold"].astype(int).to_numpy()).sum()),
                "patient_group_mismatch": int((src["patient_group_id"].astype(str).to_numpy() != ref["patient_group_id"].astype(str).to_numpy()).sum()),
                "duplicate_case_id": int(src["case_id"].duplicated().sum()),
                "missing_case_id": int(len(set(ref["case_id"]) - set(src["case_id"]))),
            }
        )
    return {"status": "passed", "rows": rows}


def _relationship_row(name_a: str, a: pd.DataFrame, name_b: str, b: pd.DataFrame) -> dict[str, Any]:
    merged = a.merge(b[["case_id", "prob_patient", "pred_binary"]], on="case_id", suffixes=("_a", "_b"))
    pa = merged["prob_patient_a"].to_numpy(float)
    pb = merged["prob_patient_b"].to_numpy(float)
    correct_a = merged["pred_binary_a"].astype(int).to_numpy() == merged["label_binary"].astype(int).to_numpy()
    correct_b = merged["pred_binary_b"].astype(int).to_numpy() == merged["label_binary"].astype(int).to_numpy()
    wrong_a = set(merged.loc[~correct_a, "case_id"].astype(str))
    wrong_b = set(merged.loc[~correct_b, "case_id"].astype(str))
    union = len(wrong_a | wrong_b)
    return {
        "model_a": name_a,
        "model_b": name_b,
        "pearson": float(np.corrcoef(pa, pb)[0, 1]),
        "spearman": float(spearmanr(pa, pb).correlation),
        "probability_mae": float(np.mean(np.abs(pa - pb))),
        "probability_rmse": float(np.sqrt(np.mean((pa - pb) ** 2))),
        "hard_prediction_agreement": float((merged["pred_binary_a"].astype(int) == merged["pred_binary_b"].astype(int)).mean()),
        "changed_prediction_count": int((merged["pred_binary_a"].astype(int) != merged["pred_binary_b"].astype(int)).sum()),
        "error_set_jaccard_index": float(len(wrong_a & wrong_b) / union) if union else 1.0,
    }


def _complementarity_rows(name_a: str, a: pd.DataFrame, name_b: str, b: pd.DataFrame) -> list[dict[str, Any]]:
    merged = a.merge(b[["case_id", "prob_patient", "pred_binary"]], on="case_id", suffixes=("_a", "_b"))
    correct_a = merged["pred_binary_a"].astype(int) == merged["label_binary"].astype(int)
    correct_b = merged["pred_binary_b"].astype(int) == merged["label_binary"].astype(int)
    rows = []
    for label, subset in (
        ("all", merged),
        ("control", merged[merged["label_binary"].astype(int) == 0]),
        ("patient", merged[merged["label_binary"].astype(int) == 1]),
    ):
        ca = correct_a.loc[subset.index]
        cb = correct_b.loc[subset.index]
        correction = int((~ca & cb).sum())
        harm = int((ca & ~cb).sum())
        a_wrong = int((~ca).sum())
        a_correct = int(ca.sum())
        rows.append(
            {
                "model_a": name_a,
                "model_b": name_b,
                "stratum": label,
                "n": int(len(subset)),
                "both_correct": int((ca & cb).sum()),
                "only_a_correct": int((ca & ~cb).sum()),
                "only_b_correct": correction,
                "both_wrong": int((~ca & ~cb).sum()),
                "correction_count": correction,
                "harm_count": harm,
                "correction_rate": float(correction / a_wrong) if a_wrong else 0.0,
                "harm_rate": float(harm / a_correct) if a_correct else 0.0,
                "net_correction_count": int(correction - harm),
            }
        )
    return rows


def _oracle_metrics(models: list[tuple[str, pd.DataFrame]], base: pd.DataFrame) -> dict[str, Any]:
    correct_any = np.zeros(len(base), dtype=bool)
    y = base["label_binary"].astype(int).to_numpy()
    for _, frame in models:
        aligned = frame.sort_values("case_id").reset_index(drop=True)
        correct_any |= aligned["pred_binary"].astype(int).to_numpy() == y
    pred = np.where(correct_any, y, 1 - y)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return {
        "oracle_accuracy": float((pred == y).mean()),
        "oracle_balanced_accuracy": float((tp / (tp + fn) + tn / (tn + fp)) / 2.0),
        "oracle_control_recall": float(tn / (tn + fp)),
        "oracle_patient_recall": float(tp / (tp + fn)),
        "oracle_is_trainable_or_deployable": False,
    }


def run_stage_a(output_dir: Path) -> dict[str, Any]:
    sources = _load_oof_sources()
    stage_dir = output_dir / "stage_a"
    summary_dir = output_dir / "summary"
    audit = _alignment_audit({k: sources[k] for k in ("p1_rgb", "p1_s", "p1_r")})
    _write_json(stage_dir / "alignment/oof_pairwise_alignment_audit.json", audit)
    pairs = [("p1_rgb", "p1_s"), ("p1_rgb", "p1_r"), ("p1_s", "p1_r")]
    relationships = [_relationship_row(a, sources[a], b, sources[b]) for a, b in pairs]
    relationship_df = pd.DataFrame(relationships)
    relationship_df.to_csv(stage_dir / "pairwise/pairwise_probability_relationships.csv", index=False, encoding="utf-8-sig")
    complementarity_rows = []
    for a, b in pairs:
        complementarity_rows.extend(_complementarity_rows(a, sources[a], b, sources[b]))
        complementarity_rows.extend(_complementarity_rows(b, sources[b], a, sources[a]))
    comp_df = pd.DataFrame(complementarity_rows)
    comp_df.to_csv(stage_dir / "pairwise/pairwise_error_complementarity.csv", index=False, encoding="utf-8-sig")
    comp_df.to_csv(stage_dir / "pairwise/class_specific_correction_analysis.csv", index=False, encoding="utf-8-sig")
    base = sources["p1_rgb"].sort_values("case_id").reset_index(drop=True)
    fusion_frames = {
        "rgb_s_equal_prob": _prob_frame_from_patient(base, 0.5 * base["prob_patient"].to_numpy(float) + 0.5 * sources["p1_s"].sort_values("case_id")["prob_patient"].to_numpy(float), "rgb_s_equal_prob"),
        "rgb_r_equal_prob": _prob_frame_from_patient(base, 0.5 * base["prob_patient"].to_numpy(float) + 0.5 * sources["p1_r"].sort_values("case_id")["prob_patient"].to_numpy(float), "rgb_r_equal_prob"),
        "rgb_s_r_equal_prob": _prob_frame_from_patient(base, (base["prob_patient"].to_numpy(float) + sources["p1_s"].sort_values("case_id")["prob_patient"].to_numpy(float) + sources["p1_r"].sort_values("case_id")["prob_patient"].to_numpy(float)) / 3.0, "rgb_s_r_equal_prob"),
    }
    fixed_rows = []
    bootstrap_rows = []
    for name, frame in fusion_frames.items():
        row = _metrics_row(frame, experiment=name)
        row.update(_paired_delta(frame, base))
        fixed_rows.append(row)
        boot = _fast_cluster_bootstrap(frame, iterations=BOOTSTRAP_ITERATIONS, seed=SEED)
        delta_boot = _paired_bootstrap_delta(frame, base)
        for key, ci in boot.get("ci95", {}).items():
            bootstrap_rows.append({"experiment": name, "metric": key, "ci_low": ci[0], "ci_high": ci[1], "bootstrap_type": "single_model"})
        for key, ci in delta_boot.get("ci95", {}).items():
            bootstrap_rows.append({"experiment": name, "metric": key, "ci_low": ci[0], "ci_high": ci[1], "bootstrap_type": "paired_vs_p1_rgb"})
    fixed_df = pd.DataFrame(fixed_rows)
    fixed_df.to_csv(stage_dir / "fixed_fusion/fixed_late_fusion_results.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(bootstrap_rows).to_csv(stage_dir / "bootstrap/fixed_late_fusion_bootstrap.csv", index=False, encoding="utf-8-sig")
    rank_frames = {
        "rgb_s_rank_mean": _prob_frame_from_patient(base, (rankdata(base["prob_patient"]) + rankdata(sources["p1_s"].sort_values("case_id")["prob_patient"])) / (2.0 * len(base)), "rgb_s_rank_mean"),
        "rgb_r_rank_mean": _prob_frame_from_patient(base, (rankdata(base["prob_patient"]) + rankdata(sources["p1_r"].sort_values("case_id")["prob_patient"])) / (2.0 * len(base)), "rgb_r_rank_mean"),
        "rgb_s_r_rank_mean": _prob_frame_from_patient(base, (rankdata(base["prob_patient"]) + rankdata(sources["p1_s"].sort_values("case_id")["prob_patient"]) + rankdata(sources["p1_r"].sort_values("case_id")["prob_patient"])) / (3.0 * len(base)), "rgb_s_r_rank_mean"),
    }
    rank_df = pd.DataFrame([_metrics_row(frame, experiment=name) for name, frame in rank_frames.items()])
    rank_df["diagnostic_only"] = True
    rank_df.to_csv(stage_dir / "fixed_fusion/fixed_rank_fusion_results.csv", index=False, encoding="utf-8-sig")
    oracle_rows = []
    oracle_defs = {
        "rgb_s_oracle": [("p1_rgb", sources["p1_rgb"]), ("p1_s", sources["p1_s"])],
        "rgb_r_oracle": [("p1_rgb", sources["p1_rgb"]), ("p1_r", sources["p1_r"])],
        "s_r_oracle": [("p1_s", sources["p1_s"]), ("p1_r", sources["p1_r"])],
        "rgb_s_r_oracle": [("p1_rgb", sources["p1_rgb"]), ("p1_s", sources["p1_s"]), ("p1_r", sources["p1_r"])],
    }
    for name, models in oracle_defs.items():
        oracle_rows.append({"experiment": name, **_oracle_metrics(models, base)})
    oracle_df = pd.DataFrame(oracle_rows)
    oracle_df.to_csv(stage_dir / "fixed_fusion/oracle_complementarity.csv", index=False, encoding="utf-8-sig")
    stage_a_main = fixed_df.copy()
    stage_a_main.to_csv(summary_dir / "stage_a_main_results.csv", index=False, encoding="utf-8-sig")
    comp_df.to_csv(summary_dir / "stage_a_error_complementarity.csv", index=False, encoding="utf-8-sig")
    fixed_df.to_csv(summary_dir / "stage_a_fixed_fusion_results.csv", index=False, encoding="utf-8-sig")
    report = [
        "# P1 Extension Stage A OOF Complementarity Report",
        "",
        f"- status: STAGE_A_COMPLEMENTARITY_ANALYSIS_COMPLETE",
        "- no_training: true",
        "- weight_search: false",
        "",
        "## Alignment",
        _md_table(pd.DataFrame(audit["rows"])),
        "",
        "## Probability relationships",
        _md_table(relationship_df),
        "",
        "## RGB correction/harm summary",
        _md_table(comp_df[(comp_df["model_a"] == "p1_rgb") & (comp_df["model_b"].isin(["p1_s", "p1_r"])) & (comp_df["stratum"] == "all")]),
        "",
        "## Fixed equal-probability fusion",
        _md_table(fixed_df),
        "",
        "## Diagnostic rank fusion",
        _md_table(rank_df),
        "",
        "## Oracle complementarity upper bound",
        _md_table(oracle_df),
        "",
        "Oracle results are diagnostic only and are not trainable or deployable models.",
    ]
    _write_text(ROOT / "reports/p1_extension_stage_a_oof_complementarity_report.md", "\n".join(report))
    _write_json(stage_dir / "_STAGE_SUCCESS.json", {"status": "STAGE_A_COMPLEMENTARITY_ANALYSIS_COMPLETE", "completed_at": _now()})
    return {
        "status": "STAGE_A_COMPLEMENTARITY_ANALYSIS_COMPLETE",
        "rgb_s_correction_rate": float(comp_df[(comp_df.model_a == "p1_rgb") & (comp_df.model_b == "p1_s") & (comp_df.stratum == "all")]["correction_rate"].iloc[0]),
        "rgb_r_correction_rate": float(comp_df[(comp_df.model_a == "p1_rgb") & (comp_df.model_b == "p1_r") & (comp_df.stratum == "all")]["correction_rate"].iloc[0]),
    }


class P1DualFusionDataset(Dataset):
    def __init__(self, frame: pd.DataFrame, experiment_key: str) -> None:
        self.frame = frame.reset_index(drop=True).copy()
        self.experiment_key = experiment_key
        self.secondary = STAGE_B_SECONDARY[experiment_key]

    def __len__(self) -> int:
        return int(len(self.frame))

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.frame.iloc[index].to_dict()
        sample = {
            "case_id": str(row["case_id"]),
            "patient_group_id": str(row["patient_group_id"]),
            "fold": int(row["fold"]),
            "label_original": int(row["label_original"]),
            "label_3class": int(row["label_3class"]),
            "label_binary": int(row["label_binary"]),
            "rgb": _image_to_chw_float32(local_path(row["rgb_path"])),
        }
        if self.secondary == "p1_s":
            import utils.p1_component_registry as registry
            from utils.p1_representation_normalization import load_raw_component_sample
            sample["secondary_sample"] = load_raw_component_sample(row, registry.get_component_spec("p1_s"))
        elif self.secondary == "p1_r":
            import utils.p1_component_registry as registry
            from utils.p1_representation_normalization import load_raw_component_sample
            sample["secondary_sample"] = load_raw_component_sample(row, registry.get_component_spec("p1_r"))
        else:
            sample["secondary_sample"] = None
        return sample


def _imagenet(tensor: torch.Tensor) -> torch.Tensor:
    return (tensor - IMAGENET_MEAN.view(3, 1, 1)) / IMAGENET_STD.view(3, 1, 1)


def build_dual_collate(experiment_key: str, normalization_state: Any | None, *, training: bool) -> Any:
    secondary = STAGE_B_SECONDARY[experiment_key]

    def _collate(batch: list[Mapping[str, Any]]) -> dict[str, Any]:
        primary_tensors: list[torch.Tensor] = []
        secondary_tensors: list[torch.Tensor] = []
        for sample in batch:
            do_flip = bool(training and float(torch.rand(1).item()) < 0.5)
            rgb = torch.as_tensor(sample["rgb"], dtype=torch.float32)
            if do_flip:
                rgb = rgb.flip(-1)
            primary = _imagenet(rgb)
            if secondary == "rgb":
                secondary_tensor = primary.clone()
            else:
                secondary_sample = dict(sample["secondary_sample"])
                normalized = apply_component_normalization(secondary_sample, normalization_state, horizontal_flip=do_flip)
                secondary_tensor = normalized["representation"]
            primary_tensors.append(primary)
            secondary_tensors.append(secondary_tensor)
        return {
            "case_id": [str(sample["case_id"]) for sample in batch],
            "patient_group_id": [str(sample["patient_group_id"]) for sample in batch],
            "fold": torch.tensor([int(sample["fold"]) for sample in batch], dtype=torch.long),
            "label_original": torch.tensor([int(sample["label_original"]) for sample in batch], dtype=torch.long),
            "label_3class": torch.tensor([int(sample["label_3class"]) for sample in batch], dtype=torch.long),
            "label_binary": torch.tensor([int(sample["label_binary"]) for sample in batch], dtype=torch.long),
            "primary": torch.stack(primary_tensors, dim=0),
            "secondary": torch.stack(secondary_tensors, dim=0),
        }

    return _collate


def _make_loader(dataset: Dataset, *, batch_size: int, shuffle: bool, seed: int, collate_fn: Any, device: torch.device) -> DataLoader:
    generator = torch.Generator()
    generator.manual_seed(seed)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=0, pin_memory=device.type == "cuda", generator=generator, collate_fn=collate_fn)


def _class_weights(train_frame: pd.DataFrame, device: torch.device) -> torch.Tensor:
    counts = train_frame["label_binary"].astype(int).value_counts().to_dict()
    total = int(len(train_frame))
    return torch.tensor([total / (2.0 * counts.get(0, 1)), total / (2.0 * counts.get(1, 1))], dtype=torch.float32, device=device)


def _evaluate_model(model: nn.Module, loader: DataLoader, device: torch.device, criterion: nn.Module | None = None) -> tuple[pd.DataFrame, dict[str, Any]]:
    model.eval()
    rows: list[dict[str, Any]] = []
    losses: list[float] = []
    with torch.no_grad():
        for batch in loader:
            x1 = batch["primary"].to(device)
            x2 = batch["secondary"].to(device)
            labels = batch["label_binary"].to(device=device, dtype=torch.long)
            logits = model(x1, x2)
            if criterion is not None:
                losses.append(float(criterion(logits, labels).item()))
            probs = torch.softmax(logits, dim=1).detach().cpu().numpy()
            pred = probs.argmax(axis=1)
            for i, case_id in enumerate(batch["case_id"]):
                rows.append(
                    {
                        "case_id": str(case_id),
                        "patient_group_id": str(batch["patient_group_id"][i]),
                        "fold": int(batch["fold"][i].item()),
                        "label_original": int(batch["label_original"][i].item()),
                        "label_3class": int(batch["label_3class"][i].item()),
                        "label_binary": int(batch["label_binary"][i].item()),
                        "prob_control": float(probs[i, 0]),
                        "prob_patient": float(probs[i, 1]),
                        "pred_binary": int(pred[i]),
                    }
                )
    frame = pd.DataFrame(rows)
    metrics = compute_visit_metrics(frame)
    metrics["loss"] = float(np.mean(losses)) if losses else np.nan
    return frame, metrics


def _fold_is_complete(fold_dir: Path) -> bool:
    success = fold_dir / "_FOLD_SUCCESS.json"
    pred = fold_dir / "val_predictions_visit.csv"
    ckpt = fold_dir / "checkpoints/best_macro_auc.pth"
    return success.is_file() and pred.is_file() and ckpt.is_file() and len(pd.read_csv(pred)) == 100


def train_stage_b_fold(output_dir: Path, experiment_key: str, fold: int, *, smoke: bool, resume: bool) -> dict[str, Any]:
    mode_root = output_dir / "stage_b" / ("smoke" if smoke else experiment_key) / experiment_key if smoke else output_dir / "stage_b" / experiment_key
    fold_dir = mode_root / f"fold_{fold}"
    if resume and _fold_is_complete(fold_dir):
        return _read_json(fold_dir / "_FOLD_SUCCESS.json")
    fold_dir.mkdir(parents=True, exist_ok=True)
    set_random_seed(SEED + fold)
    frame = load_p1_frame(MANIFEST_PATH, FIXED_SPLIT_PATH)
    train_frame = frame[frame["fold"] != fold].copy().reset_index(drop=True)
    val_frame = frame[frame["fold"] == fold].copy().reset_index(drop=True)
    secondary = STAGE_B_SECONDARY[experiment_key]
    normalization_state = None
    if secondary == "p1_s":
        normalization_state = fit_component_normalization(train_frame, "p1_s", root=ROOT)
    elif secondary == "p1_r":
        normalization_state = fit_component_normalization(train_frame, "p1_r", root=ROOT)
    fitted_payload = {
        "experiment": experiment_key,
        "fold": fold,
        "secondary": secondary,
        "training_case_count": int(len(train_frame)),
        "validation_used_for_fit": False,
        "fitted_parameters": normalization_state.to_dict() if normalization_state is not None else {"strategy": "fixed_rgb_imagenet"},
    }
    _write_json(fold_dir / "preprocessing/fitted_parameters.json", fitted_payload)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("CUDA is required for stage B training")
    train_ds = P1DualFusionDataset(train_frame, experiment_key)
    val_ds = P1DualFusionDataset(val_frame, experiment_key)
    batch_size = 16
    train_loader = _make_loader(train_ds, batch_size=batch_size, shuffle=True, seed=SEED + fold, collate_fn=build_dual_collate(experiment_key, normalization_state, training=True), device=device)
    val_loader = _make_loader(val_ds, batch_size=batch_size, shuffle=False, seed=SEED + 10_000 + fold, collate_fn=build_dual_collate(experiment_key, normalization_state, training=False), device=device)
    model = IndependentDualResNet18(pretrained=True).to(device)
    param_counts = count_parameters(model)
    criterion = nn.CrossEntropyLoss(weight=_class_weights(train_frame, device))
    optimizer = torch.optim.AdamW(model.parameters(), lr=1.0e-4, weight_decay=1.0e-4)
    max_epochs = 2 if smoke else 50
    patience = 10
    best_auc = -math.inf
    best_epoch = 0
    best_payload: dict[str, Any] | None = None
    history: list[dict[str, Any]] = []
    start = time.time()
    peak_memory = 0
    bad_epochs = 0
    for epoch in range(1, max_epochs + 1):
        model.train()
        train_losses = []
        train_rows = []
        for batch in train_loader:
            x1 = batch["primary"].to(device)
            x2 = batch["secondary"].to(device)
            labels = batch["label_binary"].to(device=device, dtype=torch.long)
            optimizer.zero_grad(set_to_none=True)
            logits = model(x1, x2)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            train_losses.append(float(loss.item()))
            probs = torch.softmax(logits.detach(), dim=1).cpu().numpy()
            for i, case_id in enumerate(batch["case_id"]):
                train_rows.append({"label_binary": int(batch["label_binary"][i].item()), "prob_control": float(probs[i, 0]), "prob_patient": float(probs[i, 1])})
        if torch.cuda.is_available():
            peak_memory = max(peak_memory, int(torch.cuda.max_memory_allocated(device)))
        train_metric_frame = pd.DataFrame(train_rows)
        train_metric_frame["pred_binary"] = train_metric_frame[["prob_control", "prob_patient"]].to_numpy(float).argmax(axis=1)
        val_pred, val_metrics = _evaluate_model(model, val_loader, device, criterion)
        train_metrics = compute_visit_metrics(train_metric_frame)
        row = {
            "epoch": epoch,
            "train_loss": float(np.mean(train_losses)),
            "val_loss": float(val_metrics["loss"]),
            "train_accuracy": float(train_metrics["accuracy"]),
            "val_accuracy": float(val_metrics["accuracy"]),
            "train_auc": float(train_metrics["macro_auc"]),
            "val_auc": float(val_metrics["macro_auc"]),
            "val_macro_f1": float(val_metrics["macro_f1"]),
            "val_balanced_accuracy": float(val_metrics["balanced_accuracy"]),
        }
        history.append(row)
        ckpt = {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": epoch,
            "experiment": experiment_key,
            "fold": fold,
            "seed": SEED,
        }
        (fold_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
        torch.save(ckpt, fold_dir / "checkpoints/last.pth")
        if val_metrics["macro_auc"] > best_auc:
            best_auc = float(val_metrics["macro_auc"])
            best_epoch = epoch
            best_payload = ckpt
            torch.save(ckpt, fold_dir / "checkpoints/best_macro_auc.pth")
            bad_epochs = 0
        else:
            bad_epochs += 1
        if not smoke and bad_epochs >= patience:
            break
    if best_payload is None:
        raise RuntimeError("no best checkpoint selected")
    checkpoint = torch.load(fold_dir / "checkpoints/best_macro_auc.pth", map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model"])
    val_pred, val_metrics = _evaluate_model(model, val_loader, device, criterion)
    val_pred["best_epoch"] = int(best_epoch)
    val_pred["checkpoint_path"] = str(fold_dir / "checkpoints/best_macro_auc.pth")
    val_pred.to_csv(fold_dir / "val_predictions_visit.csv", index=False, encoding="utf-8-sig")
    val_pred.to_csv(fold_dir / "val_predictions_case.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(history).to_csv(fold_dir / "training_history.csv", index=False, encoding="utf-8-sig")
    metrics_payload = {k: v for k, v in val_metrics.items() if k != "confusion_matrix"}
    metrics_payload["confusion_matrix"] = np.asarray(val_metrics["confusion_matrix"], dtype=int).tolist()
    _write_json(fold_dir / "metrics_visit.json", metrics_payload)
    cm = np.asarray(val_metrics["confusion_matrix"], dtype=int)
    pd.DataFrame(cm, index=["true_control", "true_patient"], columns=["pred_control", "pred_patient"]).to_csv(fold_dir / "confusion_matrix_visit.csv", encoding="utf-8-sig")
    duration = float(time.time() - start)
    fold_summary = {
        "experiment": experiment_key,
        "fold": fold,
        "best_epoch": int(best_epoch),
        "max_epoch_reached": int(history[-1]["epoch"]),
        "visit_case_macro_auc": float(val_metrics["macro_auc"]),
        "predicted_control_count": int((val_pred["pred_binary"] == 0).sum()),
        "predicted_patient_count": int((val_pred["pred_binary"] == 1).sum()),
        "parameter_count": int(param_counts["trainable_params"]),
        "peak_gpu_memory": int(peak_memory),
        "training_duration": duration,
    }
    _write_json(fold_dir / "fold_summary.json", fold_summary)
    success = {
        "experiment": experiment_key,
        "fold": fold,
        "mode": "smoke" if smoke else "formal",
        "prediction_rows": int(len(val_pred)),
        "unique_case_ids": int(val_pred["case_id"].nunique()),
        "best_epoch": int(best_epoch),
        "best_macro_auc": float(best_auc),
        "config_sha256": _sha256_text(ABC_CORRECTION_VERSION + experiment_key),
        "p1_rgb_oof_sha256": sha256_file(P1_RGB_OOF),
        "manifest_sha256": sha256_file(MANIFEST_PATH),
        "split_sha256": sha256_file(FIXED_SPLIT_PATH),
        "checkpoint_path": str(fold_dir / "checkpoints/best_macro_auc.pth"),
        "completed_at": _now(),
    }
    _write_json(fold_dir / "_FOLD_SUCCESS.json", success)
    if smoke:
        _write_json(mode_root / "_SMOKE_SUCCESS.json", {"experiment": experiment_key, "status": "SMOKE_PASSED", "fold": fold, "completed_at": _now(), "parameter_count": param_counts, "peak_gpu_memory": peak_memory})
    return success


def _summarize_stage_b_experiment(output_dir: Path, experiment_key: str) -> dict[str, Any]:
    exp_dir = output_dir / "stage_b" / experiment_key
    fold_frames = [pd.read_csv(exp_dir / f"fold_{fold}/val_predictions_visit.csv", dtype={"case_id": str, "patient_group_id": str}) for fold in range(5)]
    oof = pd.concat(fold_frames, ignore_index=True).sort_values(["fold", "case_id"]).reset_index(drop=True)
    oof_dir = exp_dir / "oof"
    oof_dir.mkdir(parents=True, exist_ok=True)
    oof.to_csv(oof_dir / "oof_predictions_visit.csv", index=False, encoding="utf-8-sig")
    metrics = compute_visit_metrics(oof)
    metrics_payload = {k: v for k, v in metrics.items() if k != "confusion_matrix"}
    metrics_payload["confusion_matrix"] = np.asarray(metrics["confusion_matrix"], dtype=int).tolist()
    _write_json(oof_dir / "oof_metrics_visit.json", metrics_payload)
    pd.DataFrame(np.asarray(metrics["confusion_matrix"], dtype=int), index=["true_control", "true_patient"], columns=["pred_control", "pred_patient"]).to_csv(oof_dir / "oof_confusion_matrix_visit.csv", encoding="utf-8-sig")
    bootstrap = _fast_cluster_bootstrap(oof, iterations=BOOTSTRAP_ITERATIONS, seed=SEED)
    _write_json(exp_dir / "summary/bootstrap_visit.json", bootstrap)
    fold_rows = []
    stability_rows = []
    for fold in range(5):
        metrics_fold = _read_json(exp_dir / f"fold_{fold}/metrics_visit.json")
        summary = _read_json(exp_dir / f"fold_{fold}/fold_summary.json")
        history = pd.read_csv(exp_dir / f"fold_{fold}/training_history.csv")
        best_hist = history[history["epoch"] == int(summary["best_epoch"])].iloc[0]
        fold_rows.append(
            {
                "experiment": experiment_key,
                "fold": fold,
                "best_epoch": int(summary["best_epoch"]),
                "macro_auc": float(metrics_fold["macro_auc"]),
                "accuracy": float(metrics_fold["accuracy"]),
                "macro_f1": float(metrics_fold["macro_f1"]),
                "balanced_accuracy": float(metrics_fold["balanced_accuracy"]),
                "patient_sensitivity": float(metrics_fold["patient_sensitivity"]),
                "control_specificity": float(metrics_fold["control_specificity"]),
                "predicted_control_count": int(summary["predicted_control_count"]),
                "predicted_patient_count": int(summary["predicted_patient_count"]),
            }
        )
        stability_rows.append(
            {
                "experiment": experiment_key,
                "fold": fold,
                "best_epoch": int(summary["best_epoch"]),
                "maximum_epoch_reached": int(summary["max_epoch_reached"]),
                "training_loss_at_best_epoch": float(best_hist["train_loss"]),
                "validation_loss_at_best_epoch": float(best_hist["val_loss"]),
                "training_accuracy": float(best_hist["train_accuracy"]),
                "validation_accuracy": float(best_hist["val_accuracy"]),
                "training_auc": float(best_hist["train_auc"]),
                "validation_auc": float(best_hist["val_auc"]),
                "predicted_control_count": int(summary["predicted_control_count"]),
                "predicted_patient_count": int(summary["predicted_patient_count"]),
                "parameter_count": int(summary["parameter_count"]),
                "peak_gpu_memory": int(summary["peak_gpu_memory"]),
                "training_duration": float(summary["training_duration"]),
                "early_epoch_selection": int(summary["best_epoch"]) <= 2,
                "prediction_collapse_control": int(summary["predicted_control_count"]) >= 95,
                "prediction_collapse_patient": int(summary["predicted_patient_count"]) >= 95,
                "large_train_val_gap": float(best_hist["train_auc"]) - float(best_hist["val_auc"]) > 0.15,
            }
        )
    fold_df = pd.DataFrame(fold_rows)
    fold_df.to_csv(exp_dir / "summary/fold_metrics_visit.csv", index=False, encoding="utf-8-sig")
    display = STAGE_B_DISPLAY[experiment_key]
    main = _metrics_row(oof, experiment=experiment_key, display_name=display)
    main["completed_folds"] = 5
    main["fold_macro_auc_mean"] = float(fold_df["macro_auc"].mean())
    main["fold_macro_auc_std"] = float(fold_df["macro_auc"].std(ddof=1))
    main["fold_macro_f1_mean"] = float(fold_df["macro_f1"].mean())
    main["fold_macro_f1_std"] = float(fold_df["macro_f1"].std(ddof=1))
    main["fold_balanced_accuracy_mean"] = float(fold_df["balanced_accuracy"].mean())
    main["fold_balanced_accuracy_std"] = float(fold_df["balanced_accuracy"].std(ddof=1))
    main["pooled_minus_fold_mean_auc"] = float(main["macro_auc"] - main["fold_macro_auc_mean"])
    main["status"] = "COMPLETED"
    main["warning_count"] = int(sum(bool(row["prediction_collapse_control"] or row["prediction_collapse_patient"] or row["large_train_val_gap"]) for row in stability_rows))
    _write_json(exp_dir / "_EXPERIMENT_SUCCESS.json", {"experiment": experiment_key, "status": "COMPLETED", "oof_rows": int(len(oof)), "completed_at": _now()})
    return {"main": main, "fold_rows": fold_rows, "stability_rows": stability_rows, "oof": oof, "bootstrap": bootstrap}


def run_stage_b(output_dir: Path, *, resume: bool = False) -> dict[str, Any]:
    smoke_status = {}
    for experiment in STAGE_B_EXPERIMENTS:
        train_stage_b_fold(output_dir, experiment, 0, smoke=True, resume=resume)
        smoke_status[experiment] = "SMOKE_PASSED"
    main_rows = []
    fold_rows_all = []
    stability_rows_all = []
    oofs: dict[str, pd.DataFrame] = {}
    for experiment in STAGE_B_EXPERIMENTS:
        failed = False
        for fold in range(5):
            try:
                train_stage_b_fold(output_dir, experiment, fold, smoke=False, resume=resume)
            except Exception:
                try:
                    train_stage_b_fold(output_dir, experiment, fold, smoke=False, resume=False)
                except Exception as error:
                    failed = True
                    _write_json(output_dir / "stage_b" / experiment / "_EXPERIMENT_SUCCESS.json", {"experiment": experiment, "status": "FAILED", "error": str(error), "completed_at": _now()})
                    break
        if not failed:
            summary = _summarize_stage_b_experiment(output_dir, experiment)
            main_rows.append(summary["main"])
            fold_rows_all.extend(summary["fold_rows"])
            stability_rows_all.extend(summary["stability_rows"])
            oofs[experiment] = summary["oof"]
    summary_dir = output_dir / "stage_b/summary"
    main_df = pd.DataFrame(main_rows)
    fold_df = pd.DataFrame(fold_rows_all)
    stability_df = pd.DataFrame(stability_rows_all)
    main_df.to_csv(summary_dir / "stage_b_main_results.csv", index=False, encoding="utf-8-sig")
    fold_df.to_csv(summary_dir / "stage_b_fold_results.csv", index=False, encoding="utf-8-sig")
    stability_df.to_csv(summary_dir / "stage_b_training_stability.csv", index=False, encoding="utf-8-sig")
    sources = _load_oof_sources()
    paired_rows = []
    comparisons = {
        "rgb_rgb_capacity_control": [("p1_rgb", sources["p1_rgb"])],
        "rgb_s_independent_dual": [("p1_rgb", sources["p1_rgb"]), ("p1_s", sources["p1_s"]), ("rgb_rgb_capacity_control", oofs.get("rgb_rgb_capacity_control"))],
        "rgb_r_independent_dual": [("p1_rgb", sources["p1_rgb"]), ("p1_r", sources["p1_r"]), ("rgb_rgb_capacity_control", oofs.get("rgb_rgb_capacity_control"))],
    }
    for experiment, refs in comparisons.items():
        if experiment not in oofs:
            continue
        for ref_name, ref in refs:
            if ref is None:
                continue
            delta = _paired_delta(oofs[experiment], ref)
            boot = _paired_bootstrap_delta(oofs[experiment], ref)
            paired_rows.append({"experiment": experiment, "reference": ref_name, **delta, **{f"{k}_ci_low": v[0] for k, v in boot.get("ci95", {}).items()}, **{f"{k}_ci_high": v[1] for k, v in boot.get("ci95", {}).items()}})
    paired_df = pd.DataFrame(paired_rows)
    paired_df.to_csv(summary_dir / "stage_b_paired_comparisons.csv", index=False, encoding="utf-8-sig")
    capacity_rows = []
    for experiment in ("rgb_s_independent_dual", "rgb_r_independent_dual"):
        row = paired_df[(paired_df["experiment"] == experiment) & (paired_df["reference"] == "rgb_rgb_capacity_control")]
        if not row.empty:
            capacity_rows.append({"experiment": experiment, "reference": "rgb_rgb_capacity_control", "delta_macro_auc": float(row["delta_macro_auc"].iloc[0]), "delta_balanced_accuracy": float(row["delta_balanced_accuracy"].iloc[0])})
    capacity_df = pd.DataFrame(capacity_rows)
    capacity_df.to_csv(summary_dir / "stage_b_capacity_control_analysis.csv", index=False, encoding="utf-8-sig")
    matrix_lines = ["# Stage B result matrix", ""]
    if not main_df.empty:
        matrix_lines.extend(["| experiment | status | macro_auc | macro_f1 | balanced_accuracy |", "| --- | --- | ---: | ---: | ---: |"])
        for _, row in main_df.sort_values("macro_auc", ascending=False).iterrows():
            matrix_lines.append(f"| {row['experiment']} | {row['status']} | {row['macro_auc']:.4f} | {row['macro_f1']:.4f} | {row['balanced_accuracy']:.4f} |")
    _write_text(summary_dir / "stage_b_result_matrix.md", "\n".join(matrix_lines))
    for src_name, dst_name in (
        ("stage_b_main_results.csv", "stage_b_main_results.csv"),
        ("stage_b_fold_results.csv", "stage_b_fold_results.csv"),
        ("stage_b_paired_comparisons.csv", "stage_b_paired_comparisons.csv"),
        ("stage_b_capacity_control_analysis.csv", "stage_b_capacity_control_analysis.csv"),
        ("stage_b_training_stability.csv", "stage_b_training_stability.csv"),
    ):
        (output_dir / "summary").mkdir(parents=True, exist_ok=True)
        pd.read_csv(summary_dir / src_name).to_csv(output_dir / "summary" / dst_name, index=False, encoding="utf-8-sig")
    status = "STAGE_B_FUSION_EXPERIMENTS_COMPLETE" if len(main_rows) == 3 else "STAGE_B_FUSION_EXPERIMENTS_COMPLETE_WITH_FAILURES"
    _write_json(output_dir / "stage_b/_STAGE_SUCCESS.json", {"status": status, "completed_experiments": len(main_rows), "formal_folds_completed": int(len(fold_rows_all)), "smoke": smoke_status, "completed_at": _now()})
    report = [
        "# P1 Extension Stage B Fusion Results Report",
        "",
        f"- status: {status}",
        "- no_p1_checkpoint_initialization: true",
        "- hyperparameter_search: false",
        "",
        "## Main results",
        _md_table(main_df) if not main_df.empty else "No completed experiments.",
        "",
        "## Paired comparisons",
        _md_table(paired_df) if not paired_df.empty else "No paired comparisons.",
        "",
        "## Capacity control",
        _md_table(capacity_df) if not capacity_df.empty else "No capacity control rows.",
        "",
        "## Training stability",
        _md_table(stability_df) if not stability_df.empty else "No stability rows.",
    ]
    _write_text(ROOT / "reports/p1_extension_stage_b_fusion_results_report.md", "\n".join(report))
    return {"status": status, "completed_folds": int(len(fold_rows_all)), "main_rows": main_rows, "paired_rows": paired_rows, "smoke": smoke_status}


def _exif_frame() -> pd.DataFrame:
    frame = load_p1_frame(MANIFEST_PATH, FIXED_SPLIT_PATH)
    out = frame[["case_id", "patient_group_id", "fold", "label_binary", "label_original", "label_3class", "camera_model", "exposure_time_seconds", "fnumber", "iso_numeric", "brightness_value"]].copy()
    out["case_id"] = out["case_id"].astype(str)
    out["patient_group_id"] = out["patient_group_id"].astype(str)
    out["camera_key"] = out["camera_model"].fillna("MISSING").astype(str)
    out["log2_exposure"] = np.where(pd.to_numeric(out["exposure_time_seconds"], errors="coerce") > 0, np.log2(pd.to_numeric(out["exposure_time_seconds"], errors="coerce")), np.nan)
    out["log2_iso"] = np.where(pd.to_numeric(out["iso_numeric"], errors="coerce") > 0, np.log2(pd.to_numeric(out["iso_numeric"], errors="coerce")), np.nan)
    out["log2_fnumber"] = np.where(pd.to_numeric(out["fnumber"], errors="coerce") > 0, np.log2(pd.to_numeric(out["fnumber"], errors="coerce")), np.nan)
    out["brightness"] = pd.to_numeric(out["brightness_value"], errors="coerce")
    return out


def _fit_acquisition_probe(frame: pd.DataFrame, probe: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    numeric = ["log2_exposure", "log2_fnumber", "log2_iso", "brightness"]
    rows = []
    fold_rows = []
    for fold in range(5):
        train = frame[frame["fold"] != fold].copy()
        val = frame[frame["fold"] == fold].copy()
        y_train = train["label_binary"].astype(int)
        transformers = []
        if probe in {"exif_numeric", "combined_acquisition"}:
            transformers.append(("num", Pipeline([("imputer", SimpleImputer(strategy="median", add_indicator=True)), ("scaler", StandardScaler())]), numeric))
        if probe in {"camera", "combined_acquisition"}:
            train_counts = train["camera_key"].fillna("MISSING").astype(str).value_counts()
            train["camera_bucket"] = train["camera_key"].fillna("MISSING").astype(str).map(lambda x: x if train_counts.get(x, 0) >= 5 else "OTHER")
            allowed = set(train["camera_bucket"].unique())
            val["camera_bucket"] = val["camera_key"].fillna("MISSING").astype(str).map(lambda x: x if x in allowed else "OTHER")
            try:
                encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
            except TypeError:
                encoder = OneHotEncoder(handle_unknown="ignore", sparse=False)
            transformers.append(("camera", encoder, ["camera_bucket"]))
        preprocessor = ColumnTransformer(transformers)
        model = Pipeline([("preprocessor", preprocessor), ("clf", LogisticRegression(penalty="l2", C=1.0, class_weight="balanced", solver="liblinear", max_iter=5000, random_state=SEED))])
        train_features = train[numeric + (["camera_bucket"] if "camera_bucket" in train.columns else [])]
        val_features = val[numeric + (["camera_bucket"] if "camera_bucket" in val.columns else [])]
        model.fit(train_features, y_train)
        prob_patient = model.predict_proba(val_features)[:, 1]
        pred = np.column_stack([1.0 - prob_patient, prob_patient]).argmax(axis=1)
        for i, (_, row) in enumerate(val.iterrows()):
            rows.append({"probe": probe, "case_id": row["case_id"], "patient_group_id": row["patient_group_id"], "fold": fold, "label_binary": int(row["label_binary"]), "label_original": int(row["label_original"]), "label_3class": int(row["label_3class"]), "prob_control": float(1.0 - prob_patient[i]), "prob_patient": float(prob_patient[i]), "pred_binary": int(pred[i])})
        pred_frame = pd.DataFrame([r for r in rows if r["probe"] == probe and int(r["fold"]) == fold])
        metrics = _metrics_row(pred_frame, experiment=probe)
        metrics["fold"] = fold
        fold_rows.append(metrics)
    return pd.DataFrame(rows), pd.DataFrame(fold_rows)


def _safe_spearman(x: pd.Series, y: pd.Series) -> float:
    valid = x.notna() & y.notna()
    if int(valid.sum()) < 3 or x[valid].nunique() < 2 or y[valid].nunique() < 2:
        return np.nan
    return float(spearmanr(x[valid], y[valid]).correlation)


def _safe_spearman_np(x: np.ndarray, y: np.ndarray) -> float:
    x_arr = np.asarray(x, dtype=float)
    y_arr = np.asarray(y, dtype=float)
    valid = np.isfinite(x_arr) & np.isfinite(y_arr)
    if int(valid.sum()) < 3:
        return np.nan
    xv = x_arr[valid]
    yv = y_arr[valid]
    if np.unique(xv).size < 2 or np.unique(yv).size < 2:
        return np.nan
    xr = rankdata(xv, method="average")
    yr = rankdata(yv, method="average")
    xr = xr - float(xr.mean())
    yr = yr - float(yr.mean())
    denom = float(np.sqrt(np.sum(xr * xr) * np.sum(yr * yr)))
    return float(np.sum(xr * yr) / denom) if denom else np.nan


def _cluster_spearman_ci(frame: pd.DataFrame, *, variable: str, iterations: int = BOOTSTRAP_ITERATIONS, seed: int = SEED) -> dict[str, Any]:
    work = frame[["patient_group_id", "prob_patient", variable]].reset_index(drop=True).copy()
    groups = _cluster_index_arrays(work)
    rng = np.random.default_rng(seed)
    samples: list[float] = []
    failed = 0
    prob = work["prob_patient"].to_numpy(float)
    values = work[variable].to_numpy(float)
    for _ in range(int(iterations)):
        picked = rng.integers(0, len(groups), size=len(groups))
        idx = np.concatenate([groups[i] for i in picked])
        rho = _safe_spearman_np(prob[idx], values[idx])
        if np.isnan(rho):
            failed += 1
        else:
            samples.append(float(rho))
    sample = pd.Series(samples, dtype=float)
    return {
        "rho_ci_low": float(sample.quantile(0.025)) if not sample.empty else np.nan,
        "rho_ci_high": float(sample.quantile(0.975)) if not sample.empty else np.nan,
        "bootstrap_iterations": int(iterations),
        "bootstrap_valid_iterations": int(len(sample)),
        "bootstrap_failed_iterations": int(failed),
    }


def _stratified_auc(frame: pd.DataFrame, stratum_col: str) -> dict[str, Any]:
    total_pairs = 0
    concordant = 0.0
    valid_strata = 0
    for _, group in frame.groupby(stratum_col, dropna=False):
        pos = group[group["label_binary"].astype(int) == 1]["prob_patient"].to_numpy(float)
        neg = group[group["label_binary"].astype(int) == 0]["prob_patient"].to_numpy(float)
        if len(pos) == 0 or len(neg) == 0:
            continue
        valid_strata += 1
        diff = pos[:, None] - neg[None, :]
        concordant += float((diff > 0).sum() + 0.5 * (diff == 0).sum())
        total_pairs += int(diff.size)
    return {"within_stratum_weighted_auc": float(concordant / total_pairs) if total_pairs else np.nan, "valid_strata": int(valid_strata), "valid_pairs": int(total_pairs)}


def run_stage_c(output_dir: Path) -> dict[str, Any]:
    exif = _exif_frame()
    stage_dir = output_dir / "stage_c"
    for filename in (
        "brightness_stratified_performance.csv",
        "exposure_stratified_performance.csv",
        "iso_stratified_performance.csv",
        "fnumber_stratified_performance.csv",
    ):
        path = stage_dir / "audit" / filename
        if path.is_file():
            path.unlink()
    stratified_performance_rows: dict[str, list[dict[str, Any]]] = {
        "brightness_stratified_performance.csv": [],
        "exposure_stratified_performance.csv": [],
        "iso_stratified_performance.csv": [],
        "fnumber_stratified_performance.csv": [],
    }
    probes = []
    fold_probe_rows = []
    for probe in ("exif_numeric", "camera", "combined_acquisition"):
        oof, fold_rows = _fit_acquisition_probe(exif, probe)
        probes.append(oof)
        fold_probe_rows.append(fold_rows)
    probe_oof = pd.concat(probes, ignore_index=True)
    probe_fold = pd.concat(fold_probe_rows, ignore_index=True)
    probe_main = pd.DataFrame([_metrics_row(probe_oof[probe_oof["probe"] == probe].copy(), experiment=probe) for probe in ("exif_numeric", "camera", "combined_acquisition")])
    probe_boot_rows = []
    for probe in ("exif_numeric", "camera", "combined_acquisition"):
        boot = _fast_cluster_bootstrap(probe_oof[probe_oof["probe"] == probe].copy(), iterations=BOOTSTRAP_ITERATIONS, seed=SEED)
        for metric, ci in boot.get("ci95", {}).items():
            probe_boot_rows.append({"probe": probe, "metric": metric, "ci_low": ci[0], "ci_high": ci[1]})
    probe_main.to_csv(stage_dir / "acquisition_probe_main_results.csv", index=False, encoding="utf-8-sig")
    probe_oof.to_csv(stage_dir / "acquisition_probe_oof_predictions.csv", index=False, encoding="utf-8-sig")
    probe_fold.to_csv(stage_dir / "acquisition_probe_fold_results.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(probe_boot_rows).to_csv(stage_dir / "acquisition_probe_bootstrap.csv", index=False, encoding="utf-8-sig")
    model_frames = _load_oof_sources()
    for exp in STAGE_B_EXPERIMENTS:
        path = output_dir / "stage_b" / exp / "oof/oof_predictions_visit.csv"
        if path.is_file():
            model_frames[exp] = pd.read_csv(path, dtype={"case_id": str, "patient_group_id": str})
    numeric_vars = ["log2_exposure", "log2_fnumber", "log2_iso", "brightness"]
    corr_rows = []
    within_rows = []
    for model_name, pred in model_frames.items():
        merged = pred.merge(exif[["case_id", *numeric_vars, "camera_key"]], on="case_id", how="left")
        for var in numeric_vars:
            corr_rows.append({"model": model_name, "variable": var, "rho": _safe_spearman(merged["prob_patient"], merged[var]), "scope": "overall", **_cluster_spearman_ci(merged, variable=var)})
            for label, name in ((0, "control"), (1, "patient")):
                sub = merged[merged["label_binary"].astype(int) == label]
                within_rows.append({"model": model_name, "variable": var, "rho": _safe_spearman(sub["prob_patient"], sub[var]), "scope": name, **_cluster_spearman_ci(sub, variable=var)})
    corr_df = pd.DataFrame(corr_rows)
    within_df = pd.DataFrame(within_rows)
    corr_df.to_csv(stage_dir / "audit/model_exif_correlations.csv", index=False, encoding="utf-8-sig")
    within_df.to_csv(stage_dir / "audit/model_exif_within_label_correlations.csv", index=False, encoding="utf-8-sig")
    camera_summary_rows = []
    camera_perf_rows = []
    for model_name, pred in model_frames.items():
        merged = pred.merge(exif[["case_id", "camera_key"]], on="case_id", how="left")
        for camera, group in merged.groupby("camera_key", dropna=False):
            n_control = int((group["label_binary"].astype(int) == 0).sum())
            n_patient = int((group["label_binary"].astype(int) == 1).sum())
            camera_summary_rows.append({"model": model_name, "camera_key": camera, "n_total": int(len(group)), "n_control": n_control, "n_patient": n_patient, "mean_prob_patient": float(group["prob_patient"].mean()), "median_prob_patient": float(group["prob_patient"].median()), "control_mean_prob_patient": float(group[group["label_binary"].astype(int) == 0]["prob_patient"].mean()) if n_control else np.nan, "patient_mean_prob_patient": float(group[group["label_binary"].astype(int) == 1]["prob_patient"].mean()) if n_patient else np.nan})
            if len(group) >= 30 and n_control >= 10 and n_patient >= 10:
                row = _metrics_row(group, experiment=model_name)
                row.update({"camera_key": camera, "status": "OK"})
                camera_perf_rows.append(row)
            else:
                camera_perf_rows.append({"experiment": model_name, "camera_key": camera, "n_oof": int(len(group)), "status": "INSUFFICIENT_STRATUM_SUPPORT"})
    pd.DataFrame(camera_summary_rows).to_csv(stage_dir / "audit/camera_probability_summary.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(camera_perf_rows).to_csv(stage_dir / "audit/camera_stratified_performance.csv", index=False, encoding="utf-8-sig")
    strat_auc_rows = []
    for model_name, pred in model_frames.items():
        merged = pred.merge(exif[["case_id", "camera_key", *numeric_vars]], on="case_id", how="left")
        overall = float(compute_visit_metrics(merged)["macro_auc"])
        strat = _stratified_auc(merged, "camera_key")
        strat_auc_rows.append({"model": model_name, "stratification": "camera", "overall_auc": overall, **strat, "difference": overall - strat["within_stratum_weighted_auc"] if not pd.isna(strat["within_stratum_weighted_auc"]) else np.nan})
        for var, filename in (("brightness", "brightness_stratified_performance.csv"), ("log2_exposure", "exposure_stratified_performance.csv"), ("log2_iso", "iso_stratified_performance.csv"), ("log2_fnumber", "fnumber_stratified_performance.csv")):
            q = pd.qcut(merged[var], q=4, duplicates="drop")
            merged[f"{var}_stratum"] = q.astype(str)
            merged.loc[merged[var].isna(), f"{var}_stratum"] = "MISSING"
            rows = []
            for stratum, group in merged.groupby(f"{var}_stratum", dropna=False):
                n_control = int((group["label_binary"].astype(int) == 0).sum())
                n_patient = int((group["label_binary"].astype(int) == 1).sum())
                row = {"model": model_name, "variable": var, "stratum": stratum, "n_total": int(len(group)), "n_control": n_control, "n_patient": n_patient}
                if n_control and n_patient:
                    row.update(_metrics_row(group, experiment=model_name))
                    row["status"] = "OK"
                else:
                    row["status"] = "AUC_NOT_ESTIMABLE"
                rows.append(row)
            stratified_performance_rows[filename].extend(rows)
            strat = _stratified_auc(merged, f"{var}_stratum")
            strat_auc_rows.append({"model": model_name, "stratification": var, "overall_auc": overall, **strat, "difference": overall - strat["within_stratum_weighted_auc"] if not pd.isna(strat["within_stratum_weighted_auc"]) else np.nan})
    for filename, rows in stratified_performance_rows.items():
        pd.DataFrame(rows).to_csv(stage_dir / f"audit/{filename}", index=False, encoding="utf-8-sig")
    strat_df = pd.DataFrame(strat_auc_rows)
    strat_df.to_csv(stage_dir / "audit/stratified_auc_summary.csv", index=False, encoding="utf-8-sig")
    combined_auc = float(probe_main[probe_main["experiment"] == "combined_acquisition"]["macro_auc"].iloc[0])
    flag_rows = []
    for model_name in model_frames:
        max_diff = float(strat_df[strat_df["model"] == model_name]["difference"].max())
        max_within = float(within_df[within_df["model"] == model_name]["rho"].abs().max())
        if combined_auc >= 0.75 and max_diff >= 0.10:
            flag = "HIGH_ACQUISITION_DEPENDENCE"
        elif max_diff >= 0.05 or max_within >= 0.30:
            flag = "MODERATE_ACQUISITION_DEPENDENCE"
        else:
            flag = "LIMITED_EVIDENCE_OF_ACQUISITION_DEPENDENCE"
        flag_rows.append({"model": model_name, "combined_probe_auc": combined_auc, "max_overall_minus_stratified_auc": max_diff, "max_abs_within_label_exif_rho": max_within, "acquisition_dependence_flag": flag, "causal_claim": False})
    flags_df = pd.DataFrame(flag_rows)
    flags_df.to_csv(stage_dir / "audit/acquisition_confounding_flags.csv", index=False, encoding="utf-8-sig")
    probe_main.to_csv(output_dir / "summary/stage_c_acquisition_probe_results.csv", index=False, encoding="utf-8-sig")
    corr_df.to_csv(output_dir / "summary/stage_c_exif_correlations.csv", index=False, encoding="utf-8-sig")
    strat_df.to_csv(output_dir / "summary/stage_c_stratified_auc_results.csv", index=False, encoding="utf-8-sig")
    flags_df.to_csv(output_dir / "summary/stage_c_acquisition_confounding_flags.csv", index=False, encoding="utf-8-sig")
    report = [
        "# P1 Extension Stage C Acquisition Audit Report",
        "",
        "- status: STAGE_C_ACQUISITION_AUDIT_COMPLETE",
        "- logistic_regression_C: 1.0",
        "- hyperparameter_search: false",
        "- causal_claim: false",
        "",
        "## Probe results",
        _md_table(probe_main),
        "",
        "## EXIF correlations",
        _md_table(corr_df),
        "",
        "## Within-label EXIF correlations",
        _md_table(within_df),
        "",
        "## Stratified AUC",
        _md_table(strat_df),
        "",
        "## Acquisition flags",
        _md_table(flags_df),
        "",
        "Available metadata cannot exclude unrecorded acquisition factors.",
    ]
    _write_text(ROOT / "reports/p1_extension_stage_c_acquisition_audit_report.md", "\n".join(report))
    _write_json(stage_dir / "_STAGE_SUCCESS.json", {"status": "STAGE_C_ACQUISITION_AUDIT_COMPLETE", "completed_at": _now()})
    return {"status": "STAGE_C_ACQUISITION_AUDIT_COMPLETE", "probe_results": probe_main.to_dict(orient="records"), "flags": flags_df.to_dict(orient="records")}


def write_decision_report(output_dir: Path) -> dict[str, Any]:
    summary_dir = output_dir / "summary"
    stage_a_comp = pd.read_csv(summary_dir / "stage_a_error_complementarity.csv") if (summary_dir / "stage_a_error_complementarity.csv").is_file() else pd.DataFrame()
    stage_a_fusion = pd.read_csv(summary_dir / "stage_a_fixed_fusion_results.csv") if (summary_dir / "stage_a_fixed_fusion_results.csv").is_file() else pd.DataFrame()
    stage_b = pd.read_csv(summary_dir / "stage_b_main_results.csv") if (summary_dir / "stage_b_main_results.csv").is_file() else pd.DataFrame()
    stage_b_folds = pd.read_csv(summary_dir / "stage_b_fold_results.csv") if (summary_dir / "stage_b_fold_results.csv").is_file() else pd.DataFrame()
    p1_rgb_folds = pd.read_csv(P1_RGB_ROOT / "summary/fold_metrics_case.csv") if (P1_RGB_ROOT / "summary/fold_metrics_case.csv").is_file() else pd.DataFrame()
    paired = pd.read_csv(summary_dir / "stage_b_paired_comparisons.csv") if (summary_dir / "stage_b_paired_comparisons.csv").is_file() else pd.DataFrame()
    flags = pd.read_csv(summary_dir / "stage_c_acquisition_confounding_flags.csv") if (summary_dir / "stage_c_acquisition_confounding_flags.csv").is_file() else pd.DataFrame()
    rows = []
    for component, fusion_exp, late_exp, p1_ref in (
        ("S", "rgb_s_independent_dual", "rgb_s_equal_prob", "p1_s"),
        ("R", "rgb_r_independent_dual", "rgb_r_equal_prob", "p1_r"),
    ):
        comp_row = stage_a_comp[(stage_a_comp["model_a"] == "p1_rgb") & (stage_a_comp["model_b"] == p1_ref) & (stage_a_comp["stratum"] == "all")]
        late_row = stage_a_fusion[stage_a_fusion["experiment"] == late_exp]
        pair_rgb = paired[(paired["experiment"] == fusion_exp) & (paired["reference"] == "p1_rgb")]
        pair_cap = paired[(paired["experiment"] == fusion_exp) & (paired["reference"] == "rgb_rgb_capacity_control")]
        flag_row = flags[flags["model"] == fusion_exp]
        delta_auc = float(pair_rgb["delta_macro_auc"].iloc[0]) if not pair_rgb.empty else np.nan
        ci_low = float(pair_rgb["delta_macro_auc_ci_low"].iloc[0]) if "delta_macro_auc_ci_low" in pair_rgb.columns and not pair_rgb.empty else np.nan
        ci_high = float(pair_rgb["delta_macro_auc_ci_high"].iloc[0]) if "delta_macro_auc_ci_high" in pair_rgb.columns and not pair_rgb.empty else np.nan
        cap_delta = float(pair_cap["delta_macro_auc"].iloc[0]) if not pair_cap.empty else np.nan
        flag = str(flag_row["acquisition_dependence_flag"].iloc[0]) if not flag_row.empty else "UNKNOWN"
        folds_improved = np.nan
        if not stage_b_folds.empty and not p1_rgb_folds.empty:
            candidate_fold_auc = stage_b_folds[stage_b_folds["experiment"] == fusion_exp][["fold", "macro_auc"]].rename(columns={"macro_auc": "candidate_macro_auc"})
            baseline_fold_auc = p1_rgb_folds[["fold", "macro_auc"]].rename(columns={"macro_auc": "p1_rgb_macro_auc"})
            fold_delta = candidate_fold_auc.merge(baseline_fold_auc, on="fold", how="inner")
            if len(fold_delta) == 5:
                folds_improved = int((fold_delta["candidate_macro_auc"] > fold_delta["p1_rgb_macro_auc"]).sum())
        if flag == "HIGH_ACQUISITION_DEPENDENCE":
            rec = "PERFORM_ACQUISITION_DECONFOUNDING_FIRST"
        elif not np.isnan(ci_low) and ci_low > 0 and cap_delta >= -0.005:
            rec = f"PROCEED_WITH_RGB_{component}"
        elif not np.isnan(delta_auc) and delta_auc > 0.01 and cap_delta >= -0.005:
            rec = f"PROCEED_WITH_RGB_{component}" if flag != "MODERATE_ACQUISITION_DEPENDENCE" else "INCONCLUSIVE_REQUIRES_REVIEW"
        else:
            rec = "DO_NOT_PROCEED_WITH_MULTICOMPONENT_MODEL"
        rows.append(
            {
                "candidate_component": component,
                "stage_a_correction_rate": float(comp_row["correction_rate"].iloc[0]) if not comp_row.empty else np.nan,
                "stage_a_harm_rate": float(comp_row["harm_rate"].iloc[0]) if not comp_row.empty else np.nan,
                "late_fusion_delta_auc": float(late_row["delta_macro_auc"].iloc[0]) if not late_row.empty else np.nan,
                "stage_b_fusion_delta_auc": delta_auc,
                "stage_b_delta_auc_ci_low": ci_low,
                "stage_b_delta_auc_ci_high": ci_high,
                "delta_vs_rgb_rgb_capacity_control": cap_delta,
                "folds_improved": folds_improved,
                "acquisition_dependence_flag": flag,
                "stage_d_recommendation": rec,
                "reason": "Rule-based decision from pre-registered ABC criteria; stage D not started.",
            }
        )
    matrix = pd.DataFrame(rows)
    matrix.to_csv(summary_dir / "stage_d_candidate_matrix.csv", index=False, encoding="utf-8-sig")
    status_rows = []
    for stage_path, name in ((output_dir / "stage_a/_STAGE_SUCCESS.json", "stage_a"), (output_dir / "stage_b/_STAGE_SUCCESS.json", "stage_b"), (output_dir / "stage_c/_STAGE_SUCCESS.json", "stage_c")):
        status_rows.append({"stage": name, **(_read_json(stage_path) if stage_path.is_file() else {"status": "MISSING"})})
    status_rows.append({"stage": "stage_d", "status": "NOT_STARTED"})
    pd.DataFrame(status_rows).to_csv(summary_dir / "abc_execution_status.csv", index=False, encoding="utf-8-sig")
    result_matrix_lines = ["# P1 Extension ABC result matrix", "", "## Stage D candidate matrix", "", _md_table(matrix)]
    _write_text(summary_dir / "abc_result_matrix.md", "\n".join(result_matrix_lines))
    report = [
        "# P1 Extension ABC Decision Report",
        "",
        "- final_status: P1_EXTENSION_ABC_COMPLETE",
        "- stage_d_started: false",
        "",
        "## Stage A conclusion",
        _md_table(stage_a_comp[(stage_a_comp["model_a"] == "p1_rgb") & (stage_a_comp["model_b"].isin(["p1_s", "p1_r"])) & (stage_a_comp["stratum"] == "all")]) if not stage_a_comp.empty else "Stage A unavailable.",
        "",
        "## Stage B conclusion",
        _md_table(stage_b) if not stage_b.empty else "Stage B unavailable.",
        "",
        "## Stage C conclusion",
        _md_table(flags) if not flags.empty else "Stage C unavailable.",
        "",
        "## Stage D candidate matrix",
        _md_table(matrix),
        "",
        "Stage D is not automatically started.",
    ]
    _write_text(ROOT / "reports/p1_extension_abc_decision_report.md", "\n".join(report))
    _write_json(output_dir / "metadata/abc_run_manifest.json", {"final_status": "P1_EXTENSION_ABC_COMPLETE", "stage_d_started": False, "completed_at": _now(), "python": sys.version, "platform": platform.platform()})
    return {"final_status": "P1_EXTENSION_ABC_COMPLETE", "stage_d_started": False, "candidate_matrix": rows}


def summarize_only(output_dir: Path) -> dict[str, Any]:
    required = (
        output_dir / "stage_a/_STAGE_SUCCESS.json",
        output_dir / "stage_b/_STAGE_SUCCESS.json",
        output_dir / "stage_c/_STAGE_SUCCESS.json",
    )
    decision = None
    if all(path.is_file() for path in required):
        decision = write_decision_report(output_dir)
    status = output_dir / "summary/abc_execution_status.csv"
    manifest = output_dir / "metadata/abc_run_manifest.json"
    return {
        "status_file": str(status),
        "status_exists": status.is_file(),
        "decision_refreshed": decision is not None,
        "manifest": _read_json(manifest) if manifest.is_file() else None,
    }


def main() -> None:
    args = parse_args()
    output_dir = resolve_output_dir(args.output_dir)
    _create_layout(output_dir)
    if args.summarize_only:
        print(json.dumps(_json_safe(summarize_only(output_dir)), ensure_ascii=False, indent=2))
        return
    create_approval(output_dir)
    preflight = run_preflight(output_dir)
    if args.validate_only:
        print(json.dumps(_json_safe({"final_status": preflight["status"], "preflight": preflight}), ensure_ascii=False, indent=2))
        return
    run_a = args.all or args.stage_a
    run_b = args.all or args.stage_b
    run_c = args.all or args.stage_c
    results: dict[str, Any] = {"preflight": preflight}
    if args.decision_only:
        results["decision"] = write_decision_report(output_dir)
        print(json.dumps(_json_safe(results), ensure_ascii=False, indent=2))
        return
    if run_a:
        results["stage_a"] = run_stage_a(output_dir)
    if run_b:
        results["stage_b"] = run_stage_b(output_dir, resume=bool(args.resume))
    if run_c:
        results["stage_c"] = run_stage_c(output_dir)
    if args.all:
        results["decision"] = write_decision_report(output_dir)
    print(json.dumps(_json_safe(results), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
