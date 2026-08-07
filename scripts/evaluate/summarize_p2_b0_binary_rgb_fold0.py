"""Validate and summarize the fixed P2-B0 RGB fold-0 result."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from datasets.control_patient_binary_dataset import map_three_class_to_binary
from metrics.binary_classification_metrics import compute_binary_metrics, flatten_metrics
from utils.experiment_utils import load_yaml
from utils.p2_b0_protocol import e0b_protocol_differences, project_path, validate_p2_b0_config


METRIC_NAMES = (
    "macro_auc",
    "accuracy",
    "macro_precision",
    "macro_recall",
    "macro_f1",
    "balanced_accuracy",
)
REQUIRED_PREDICTION_COLUMNS = {
    "sample_id", "patient_group_id", "fold", "original_label", "original_three_class_label",
    "binary_label", "logit_normal", "logit_patient", "prob_normal", "prob_patient",
    "pred_class", "image_path", "selected_epoch", "checkpoint_path",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment-dir", type=Path, required=True)
    return parser.parse_args()


def _expected_fold0(config: dict[str, Any]) -> pd.DataFrame:
    data = config["data"]
    path = project_path(data["split_dir"]) / data["val_csv_pattern"].format(fold=0)
    expected = pd.read_csv(
        path, dtype={"ID": "string", "patient_group_id": "string"}, encoding="utf-8-sig"
    )
    required = {"ID", "patient_group_id", "NYHA", "label_3class", "fold"}
    missing = required.difference(expected.columns)
    if missing:
        raise ValueError(f"fold_0_val.csv lacks columns: {sorted(missing)}")
    if len(expected) != 100 or expected["ID"].duplicated().any():
        raise ValueError("P2-B0 requires exactly 100 unique fold-0 validation IDs")
    if not (pd.to_numeric(expected["fold"], errors="coerce") == 0).all():
        raise ValueError("fold_0_val.csv has a non-zero fold field")
    expected = expected.assign(
        sample_id=expected["ID"].astype(str),
        expected_binary_label=expected["label_3class"].map(map_three_class_to_binary),
        expected_patient_group_id=expected["patient_group_id"].astype(str),
    )
    return expected.loc[:, ["sample_id", "expected_patient_group_id", "NYHA", "label_3class", "expected_binary_label"]]


def validate_fold0_predictions(predictions: pd.DataFrame, expected: pd.DataFrame) -> dict[str, Any]:
    """Validate prediction identity/semantics, then recompute binary metrics."""

    missing = REQUIRED_PREDICTION_COLUMNS.difference(predictions.columns)
    if missing:
        raise ValueError(f"val_predictions.csv lacks fields: {sorted(missing)}")
    threshold_fields = [name for name in predictions.columns if "threshold" in name.lower()]
    if threshold_fields:
        raise ValueError(f"P2-B0 does not permit threshold-optimization fields: {threshold_fields}")
    frame = predictions.copy()
    frame["sample_id"] = frame["sample_id"].astype(str)
    frame["patient_group_id"] = frame["patient_group_id"].astype(str)
    if len(frame) != 100 or frame["sample_id"].duplicated().any():
        raise ValueError("P2-B0 predictions must contain exactly 100 unique sample IDs")
    if set(frame["sample_id"]) != set(expected["sample_id"]):
        raise ValueError("Prediction ID set does not exactly match fold_0_val.csv")
    merged = frame.merge(expected, on="sample_id", how="left", validate="one_to_one")
    if merged["expected_binary_label"].isna().any():
        raise ValueError("Predictions contain IDs not present in fold_0_val.csv")
    if not (pd.to_numeric(merged["fold"], errors="coerce") == 0).all():
        raise ValueError("Predictions must be from fold 0 only")
    if (merged["patient_group_id"] != merged["expected_patient_group_id"]).any():
        raise ValueError("patient_group_id is not traceable to fold_0_val.csv")
    if not np.array_equal(merged["binary_label"].to_numpy(int), merged["expected_binary_label"].to_numpy(int)):
        raise ValueError("Prediction binary labels do not match the fixed fold-0 mapping")
    if not np.array_equal(merged["original_three_class_label"].to_numpy(int), merged["label_3class"].to_numpy(int)):
        raise ValueError("Prediction original three-class labels do not match fold_0_val.csv")
    if not np.array_equal(merged["original_label"].to_numpy(int), merged["NYHA"].to_numpy(int)):
        raise ValueError("Prediction original NYHA labels do not match fold_0_val.csv")
    probabilities = merged[["prob_normal", "prob_patient"]].to_numpy(dtype=float)
    if not np.isfinite(probabilities).all() or (probabilities < 0).any() or (probabilities > 1).any():
        raise ValueError("Prediction probabilities must be finite and in [0, 1]")
    if not np.allclose(probabilities.sum(axis=1), 1.0, atol=1e-5):
        raise ValueError("Prediction probability rows must sum to one")
    if not np.array_equal(merged["pred_class"].to_numpy(int), probabilities.argmax(axis=1)):
        raise ValueError("pred_class must be the softmax probability argmax")
    metrics = compute_binary_metrics(merged["binary_label"].to_numpy(int), probabilities)
    return {"frame": frame, "metrics": metrics}


def _validate_training_artifacts(exp: Path, metrics: dict[str, Any]) -> dict[str, Any]:
    fold_dir = exp / "fold_0"
    metric_frame = pd.read_csv(fold_dir / "metrics.csv")
    if len(metric_frame) != 1 or int(metric_frame.loc[0, "fold"]) != 0:
        raise ValueError("fold_0/metrics.csv must have exactly one fold-0 row")
    for name, value in flatten_metrics(metrics).items():
        if name not in metric_frame.columns or not np.isclose(float(metric_frame.loc[0, name]), value, rtol=1e-7, atol=1e-8):
            raise ValueError(f"Recomputed {name} does not match fold_0/metrics.csv")
    matrix = metrics["confusion_matrix"]
    saved_matrix = pd.read_csv(fold_dir / "confusion_matrix.csv", index_col=0)
    if list(saved_matrix.index) != ["control", "patient"] or list(saved_matrix.columns) != ["control", "patient"]:
        raise ValueError("confusion_matrix.csv must use [control, patient] row/column order")
    if not np.array_equal(saved_matrix.to_numpy(dtype=int), matrix):
        raise ValueError("Recomputed confusion matrix does not match fold_0/confusion_matrix.csv")
    history = pd.read_csv(fold_dir / "training_history.csv")
    params = json.loads((fold_dir / "training_parameters.json").read_text(encoding="utf-8"))
    checkpoint_path = fold_dir / "checkpoints" / "best_macro_auc.pth"
    if history.empty or not checkpoint_path.is_file():
        raise ValueError("P2-B0 requires non-empty training history and the best checkpoint")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    required_checkpoint = {"model_state_dict", "epoch", "best_macro_auc", "config"}
    if not required_checkpoint.issubset(checkpoint):
        raise ValueError("best_macro_auc.pth has an unexpected checkpoint format")
    if int(checkpoint["epoch"]) != int(metric_frame.loc[0, "best_epoch"]):
        raise ValueError("Checkpoint epoch and metrics.csv best_epoch differ")
    if not np.isclose(float(checkpoint["best_macro_auc"]), float(metric_frame.loc[0, "macro_auc"]), rtol=1e-7, atol=1e-8):
        raise ValueError("Checkpoint best_macro_auc and metrics.csv macro_auc differ")
    return {
        "fold_metrics": metric_frame.loc[0].to_dict(),
        "training_parameters": params,
        "history_epochs": int(len(history)),
        "checkpoint_epoch": int(checkpoint["epoch"]),
        "checkpoint_keys": sorted(checkpoint.keys()),
    }


def _rate(numerator: int, denominator: int) -> float:
    if denominator == 0:
        raise ValueError("Confusion-matrix denominator must be non-zero")
    return float(numerator / denominator)


def _reproduction_check(config: dict[str, Any], current_metrics: dict[str, Any], current_best_epoch: int) -> dict[str, Any]:
    p2 = config["p2"]
    historical_dir = project_path(p2["historical_experiment_dir"])
    historical_csv = historical_dir / f"fold_{int(p2['historical_fold'])}" / "metrics.csv"
    if not historical_csv.is_file():
        raise FileNotFoundError(f"Historical E0B fold-0 metrics are missing: {historical_csv}")
    historical = pd.read_csv(historical_csv)
    if len(historical) != 1:
        raise ValueError("Historical fold-0 metrics must have exactly one row")
    historical_auc = float(historical.loc[0, "macro_auc"])
    historical_epoch = int(historical.loc[0, "best_epoch"])
    current_auc = float(current_metrics["macro_auc"])
    difference = abs(current_auc - historical_auc)
    tolerance = float(p2["soft_reproduction_auc_tolerance"])
    protocol_differences = e0b_protocol_differences(config)
    warnings: list[str] = []
    if current_best_epoch != historical_epoch:
        warnings.append("Current best epoch differs from historical fold 0; epoch equality is not required for soft reproduction.")
    if difference > tolerance:
        warnings.append(
            "AUC difference exceeds tolerance. Inspect CUDA/PyTorch environment, ImageNet weights, image root, fixed fold, seed, DataLoader seeds, and macro-AUC checkpoint selection; do not retune automatically."
        )
    return {
        "historical_fold0_macro_auc": historical_auc,
        "current_fold0_macro_auc": current_auc,
        "absolute_auc_difference": difference,
        "historical_fold0_best_epoch": historical_epoch,
        "current_fold0_best_epoch": int(current_best_epoch),
        "protocol_match": not protocol_differences,
        "protocol_differences": protocol_differences,
        "data_match": config["data"] == load_yaml(PROJECT_ROOT / "config" / "train" / "e0b_global_resnet18_control_patient_binary_5fold.yaml")["data"],
        "image_root_match": config["data"]["image_root"] == load_yaml(PROJECT_ROOT / "config" / "train" / "e0b_global_resnet18_control_patient_binary_5fold.yaml")["data"]["image_root"],
        "training_config_match": not any(item in protocol_differences for item in ("model", "train", "augmentation", "normalize")),
        "within_soft_reproduction_tolerance": difference <= tolerance,
        "soft_reproduction_auc_tolerance": tolerance,
        "warnings": warnings,
    }


def _summary_markdown(metrics: dict[str, Any], matrix: np.ndarray, artifacts: dict[str, Any], check: dict[str, Any], p1: dict[str, Any]) -> str:
    values = flatten_metrics(metrics)
    metric_rows = "\n".join(f"| {name} | {values[name]:.6f} |" for name in METRIC_NAMES)
    warning_text = "；".join(check["warnings"]) if check["warnings"] else "无"
    return f"""# P2-B0：固定 RGB 二分类 fold 0 基线汇总

## 定位

本结果仅复现 E0B 的预定义 fold 0，作为 P2-R1 与 P2-A1 的固定公平对照；不是新的五折正式性能，也不用于诊断或替代临床 NYHA 评估。类别为 Control（无病正常对照，0）与 Patient（NYHA I–IV，1）。

## 复现状态

- 协议与 E0B 一致：`{check['protocol_match']}`
- 固定数据一致：`{check['data_match']}`
- Global 图像根目录一致：`{check['image_root_match']}`
- 训练设置一致：`{check['training_config_match']}`
- P1 ready cases 仅做 ID 对齐且通过：`{p1.get('status') == 'passed'}`（未读取 P1 特征/NPZ/重光照图）
- 历史 fold 0 Macro-AUC：`{check['historical_fold0_macro_auc']:.6f}`
- 当前 fold 0 Macro-AUC：`{check['current_fold0_macro_auc']:.6f}`
- AUC 绝对差：`{check['absolute_auc_difference']:.6f}`，容差 `{check['soft_reproduction_auc_tolerance']:.6f}`，满足：`{check['within_soft_reproduction_tolerance']}`
- 历史/当前最佳 epoch：`{check['historical_fold0_best_epoch']}` / `{check['current_fold0_best_epoch']}`
- 警告：{warning_text}

## Fold 0 指标（100 个固定验证样本）

| Metric | Value |
|---|---:|
{metric_rows}
| patient_sensitivity | {artifacts['patient_sensitivity']:.6f} |
| control_specificity | {artifacts['control_specificity']:.6f} |
| patient_ppv | {artifacts['patient_ppv']:.6f} |
| patient_npv | {artifacts['patient_npv']:.6f} |

## 混淆矩阵

行是真实类别，列是预测类别；顺序固定为 `[Control, Patient]`。

| True \\ Pred | Control | Patient |
|---|---:|---:|
| Control | {int(matrix[0, 0])} | {int(matrix[0, 1])} |
| Patient | {int(matrix[1, 0])} | {int(matrix[1, 1])} |

## 可复查性

本汇总重新读取固定 `fold_0_val.csv`，校验 100 个 ID、患者组、标签映射、概率和、argmax 预测、指标、混淆矩阵、训练历史与最佳 checkpoint 格式。预测由 `prob_patient` 计算 ROC-AUC；未做阈值搜索、校准或测试时增强。
"""


def main() -> Path:
    args = parse_args()
    exp = project_path(args.experiment_dir)
    config = load_yaml(exp / "config_snapshot.yaml")
    validate_p2_b0_config(config, allow_smoke_epoch_override=exp.name.endswith("_smoke"))
    fold_dir = exp / "fold_0"
    expected = _expected_fold0(config)
    predictions = pd.read_csv(fold_dir / "val_predictions.csv", dtype={"sample_id": "string", "patient_group_id": "string"}, encoding="utf-8-sig")
    validated = validate_fold0_predictions(predictions, expected)
    metrics = validated["metrics"]
    artifacts = _validate_training_artifacts(exp, metrics)
    matrix = metrics["confusion_matrix"]
    artifacts.update({
        "patient_sensitivity": _rate(int(matrix[1, 1]), int(matrix[1, 0] + matrix[1, 1])),
        "control_specificity": _rate(int(matrix[0, 0]), int(matrix[0, 0] + matrix[0, 1])),
        "patient_ppv": _rate(int(matrix[1, 1]), int(matrix[0, 1] + matrix[1, 1])),
        "patient_npv": _rate(int(matrix[0, 0]), int(matrix[0, 0] + matrix[1, 0])),
    })
    check = _reproduction_check(config, metrics, int(artifacts["fold_metrics"]["best_epoch"]))
    p1_path = exp / "p1_ready_case_alignment.json"
    if not p1_path.is_file():
        raise FileNotFoundError("P1 ID alignment record is required before P2-B0 summary")
    p1 = json.loads(p1_path.read_text(encoding="utf-8"))
    if p1.get("status") != "passed":
        raise ValueError("P1 ID alignment record is not successful")

    metric_payload = {**flatten_metrics(metrics), **{key: artifacts[key] for key in ("patient_sensitivity", "control_specificity", "patient_ppv", "patient_npv")}}
    (exp / "p2_b0_metrics.json").write_text(json.dumps(metric_payload, ensure_ascii=False, indent=2), encoding="utf-8")
    validated["frame"].to_csv(exp / "p2_b0_predictions.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(matrix, index=["control", "patient"], columns=["control", "patient"]).to_csv(
        exp / "p2_b0_confusion_matrix.csv", index_label="true\\pred", encoding="utf-8-sig"
    )
    (exp / "p2_b0_reproduction_check.json").write_text(json.dumps(check, ensure_ascii=False, indent=2), encoding="utf-8")
    (exp / "p2_b0_summary.md").write_text(_summary_markdown(metrics, matrix, artifacts, check, p1), encoding="utf-8")
    print(f"P2_B0_SUMMARY_DIR={exp}")
    return exp


if __name__ == "__main__":
    main()
