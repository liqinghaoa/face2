"""Protocol, reference evaluation and run orchestration for overfitting controls."""

from __future__ import annotations

import argparse
import json
import math
import platform
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torchvision
from torch.utils.data import DataLoader


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from datasets.global_optical_fusion_dataset import GlobalOpticalFusionDataset  # noqa: E402
from datasets.nyha_3class_face_dataset import build_transforms  # noqa: E402
from metrics.classification_metrics import compute_classification_metrics, flatten_metrics  # noqa: E402
from models.resnet18_optical_fusion import ResNet18OpticalFusion  # noqa: E402
from scripts.train.train_global_optical_fusion_overfit_control_5fold import (  # noqa: E402
    ALLOWED_VARIANTS,
    prepare_features,
    project_path,
)
from trainers.global_optical_fusion_trainer import load_torch_checkpoint  # noqa: E402
from utils.experiment_utils import load_yaml, set_random_seed  # noqa: E402
from utils.optical_feature_preprocessor import sha256_file  # noqa: E402
from utils.resnet_trainability import TRAINABILITY_STRATEGIES  # noqa: E402


TEST_FILES = (
    "tests/test_resnet18_trainability_strategy.py",
    "tests/test_overfit_control_optimizer_groups.py",
    "tests/test_frozen_batchnorm_behavior.py",
    "tests/test_partial_layer4_behavior.py",
    "tests/test_overfit_control_checkpoint.py",
    "tests/test_overfit_control_deterministic_eval.py",
    "tests/test_overfit_control_protocol.py",
    "tests/test_overfit_control_summary.py",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument(
        "--strategy", choices=[*TRAINABILITY_STRATEGIES, "all"], default="all"
    )
    parser.add_argument("--variant", choices=[*ALLOWED_VARIANTS, "G0", "G-A", "all"], default="all")
    parser.add_argument("--fold", choices=["0", "1", "2", "3", "4", "all"], default="all")
    parser.add_argument("--protocol-only", action="store_true")
    parser.add_argument("--reference-eval-only", action="store_true")
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--train-only", action="store_true")
    parser.add_argument("--summarize-only", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--skip-completed", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--allow-cpu-training", action="store_true")
    parser.add_argument(
        "--allow-reference-mismatch",
        action="store_true",
        help="Explicitly continue local formal training while preserving a failed Full reproduction audit.",
    )
    return parser.parse_args()


def _variant_values(value: str) -> list[str]:
    mapping = {"G0": "global_only", "G-A": "global_stage2a"}
    return list(ALLOWED_VARIANTS) if value == "all" else [mapping.get(value, value)]


def _strategy_values(value: str) -> list[str]:
    return list(TRAINABILITY_STRATEGIES) if value == "all" else [value]


def _fold_values(value: str) -> list[int]:
    return list(range(5)) if value == "all" else [int(value)]


def _json_write(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)


def _historical_paths(config: dict[str, Any]) -> list[Path]:
    root = project_path(config["full_reference"]["read_only_root"])
    paths: list[Path] = []
    for variant in ALLOWED_VARIANTS:
        for fold in range(5):
            run = root / variant / f"fold_{fold}"
            paths.extend(
                [
                    run / "best_macro_auc.pth",
                    run / "fold_manifest.json",
                    run / "metrics.json",
                    run / "val_predictions.csv",
                    run / "training_log.csv",
                    run / "resolved_config.yaml",
                ]
            )
            if variant == "global_stage2a":
                paths.append(run / "feature_scaler.json")
    return paths


def historical_hash_inventory(config: dict[str, Any]) -> dict[str, str]:
    return {
        str(path.relative_to(PROJECT_ROOT)).replace("\\", "/"): sha256_file(path)
        for path in _historical_paths(config)
    }


def run_preflight(config_path: Path, output_root: Path) -> dict[str, Any]:
    config = load_yaml(config_path)
    checks: list[dict[str, Any]] = []

    def check(name: str, condition: bool, detail: Any) -> None:
        checks.append({"name": name, "passed": bool(condition), "detail": detail})

    check("dedicated_output_root", output_root == project_path(config["experiment"]["output_root"]), str(output_root))
    check("variant_allowlist", tuple(config["experiment"]["variants"]) == ALLOWED_VARIANTS, config["experiment"]["variants"])
    check("strategy_allowlist", tuple(config["experiment"]["strategies"]) == TRAINABILITY_STRATEGIES, config["experiment"]["strategies"])
    check("formal_matrix_size", 2 * 2 * len(config["data"]["folds"]) == 20, 20)
    check("imagenet_initialization", config["model"]["pretrained"] == "imagenet" and config["model"]["independent_initialization_per_run"], config["model"])
    check("forbid_full_initialization", bool(config["model"]["forbid_full_checkpoint_initialization"]), True)
    train = config["train"]
    locked_train = (
        train["batch_size"] == 16
        and train["epochs"] == 50
        and train["optimizer"] == "AdamW"
        and float(train["weight_decay"]) == 1.0e-4
        and train["scheduler"] == "none"
        and train["warmup"] == "none"
        and train["gradient_clipping"] == "none"
        and train["loss"] == "weighted_cross_entropy"
        and float(train["label_smoothing"]) == 0
        and not train["amp"]
        and train["early_stopping_patience"] == 10
        and train["seed"] == 2026
        and train["num_workers"] == 0
        and not train["pin_memory"]
    )
    check("locked_training_budget", locked_train, train)
    transforms = config["transforms"]
    check(
        "locked_transforms",
        transforms["resize"] == [224, 224]
        and float(transforms["horizontal_flip_probability"]) == 0.5
        and not transforms["color_jitter"]
        and not transforms["random_crop"],
        transforms,
    )
    trainability = config["trainability"]
    check("frozen_optimizer", float(trainability["frozen_backbone"]["classifier_learning_rate"]) == 1.0e-4, trainability["frozen_backbone"])
    check(
        "partial_optimizer",
        float(trainability["partial_layer4"]["layer4_learning_rate"]) == 1.0e-5
        and float(trainability["partial_layer4"]["classifier_learning_rate"]) == 1.0e-4,
        trainability["partial_layer4"],
    )
    check("descriptive_summary_only", config["summary"]["descriptive_only"] and not config["summary"]["bootstrap"] and not config["summary"]["significance_tests"], config["summary"])
    split_records = []
    split_ok = True
    patient_disjoint = True
    class_ok = True
    for fold in range(5):
        train_path = project_path(config["data"]["split_root"]) / str(config["data"]["train_csv_pattern"]).format(fold=fold)
        val_path = project_path(config["data"]["split_root"]) / str(config["data"]["val_csv_pattern"]).format(fold=fold)
        train_frame = pd.read_csv(train_path, dtype={"ID": "string", "patient_group_id": "string"}, encoding="utf-8-sig")
        val_frame = pd.read_csv(val_path, dtype={"ID": "string", "patient_group_id": "string"}, encoding="utf-8-sig")
        split_ok &= len(train_frame) == 400 and len(val_frame) == 100 and set(train_frame["ID"]).isdisjoint(set(val_frame["ID"]))
        patient_disjoint &= set(train_frame["patient_group_id"]).isdisjoint(set(val_frame["patient_group_id"]))
        class_ok &= set(train_frame["label_3class"].astype(int)) == {0, 1, 2} and set(val_frame["label_3class"].astype(int)) == {0, 1, 2}
        split_records.append({"fold": fold, "train": len(train_frame), "val": len(val_frame)})
    check("outer_split_400_100_id_disjoint", split_ok, split_records)
    check("outer_patient_group_disjoint", patient_disjoint, True)
    check("all_classes_per_split", class_ok, True)
    historical = project_path(config["full_reference"]["read_only_root"])
    check("historical_root_separate", historical != output_root and historical not in output_root.parents, str(historical))
    historical_ok = True
    historical_detail = []
    for variant in ALLOWED_VARIANTS:
        for fold in range(5):
            run = historical / variant / f"fold_{fold}"
            manifest = json.loads((run / "fold_manifest.json").read_text(encoding="utf-8"))
            actual = sha256_file(run / "best_macro_auc.pth")
            matched = actual == manifest["best_checkpoint_sha256"]
            historical_ok &= matched
            historical_detail.append({"variant": variant, "fold": fold, "checkpoint_hash_match": matched})
    check("ten_full_checkpoints_and_hashes", historical_ok and len(historical_detail) == 10, historical_detail)
    scaler_ok = True
    scaler_hashes = []
    for fold in range(5):
        scaler = historical / "global_stage2a" / f"fold_{fold}" / "feature_scaler.json"
        scaler_ok &= scaler.is_file()
        scaler_hashes.append(sha256_file(scaler))
    check("canonical_full_scalers", scaler_ok and len(set(scaler_hashes)) == 5, scaler_hashes)
    check("stage2a_classifier_sources_not_oof", bool(config["features"]["forbid_oof_as_classifier_input"]), config["features"])
    check("deterministic_eval_contract", config["summary"]["expected_full_runs"] + config["summary"]["expected_new_runs"] == 30, 30)
    check("required_test_files", all((PROJECT_ROOT / path).is_file() for path in TEST_FILES), list(TEST_FILES))
    check("no_forbidden_variants", not any(name in config["experiment"]["variants"] for name in ("optical_only", "global_mask", "global_raw", "global_stage2b")), config["experiment"]["variants"])
    before = historical_hash_inventory(config)
    manifest = {
        "schema_version": "global_optical_fusion_overfit_protocol_v1",
        "status": "PASS" if all(item["passed"] for item in checks) else "FAIL",
        "config": str(config_path.relative_to(PROJECT_ROOT)).replace("\\", "/"),
        "config_sha256": sha256_file(config_path),
        "checks": checks,
        "historical_hash_inventory": before,
        "historical_inputs_modified": False,
        "formal_training_gate": "requires full_reference/reference_audit.json status PASS",
    }
    _json_write(output_root / "protocol" / "preflight_manifest.json", manifest)
    if manifest["status"] != "PASS":
        failed = [item["name"] for item in checks if not item["passed"]]
        raise RuntimeError(f"Protocol preflight failed: {failed}")
    return manifest


def _runtime_matches(checkpoint: dict[str, Any]) -> tuple[bool, dict[str, Any]]:
    expected = {
        "python_version": checkpoint.get("python_version"),
        "pytorch_version": checkpoint.get("pytorch_version"),
        "torchvision_version": checkpoint.get("torchvision_version"),
        "cuda_version": checkpoint.get("cuda_version"),
        "device_type": str(checkpoint.get("device", "")).split(":")[0],
    }
    actual = {
        "python_version": platform.python_version(),
        "pytorch_version": torch.__version__,
        "torchvision_version": torchvision.__version__,
        "cuda_version": torch.version.cuda,
        "device_type": "cuda" if torch.cuda.is_available() else "cpu",
    }
    return expected == actual, {"expected": expected, "actual": actual}


def _weighted_cross_entropy_from_predictions(
    frame: pd.DataFrame, class_weights: list[float]
) -> float:
    labels = frame["true_label"].to_numpy(int)
    probabilities = frame[["prob_normal", "prob_mild", "prob_severe"]].to_numpy(float)
    weights = np.asarray(class_weights, dtype=float)[labels]
    selected = np.clip(probabilities[np.arange(len(labels)), labels], 1.0e-15, 1.0)
    return float(np.sum(-np.log(selected) * weights) / np.sum(weights))


@torch.inference_mode()
def _predict_full(
    model: ResNet18OpticalFusion,
    loader: DataLoader,
    device: torch.device,
    *,
    variant: str,
    fold: int,
    split_role: str,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    model.eval()
    rows: list[dict[str, Any]] = []
    for batch in loader:
        probabilities = torch.softmax(
            model(
                batch["image"].to(device),
                batch["aux_features"].to(device),
            ),
            dim=1,
        ).cpu().numpy()
        labels = batch["label"].numpy().astype(int)
        for index, label in enumerate(labels):
            rows.append(
                {
                    "ID": str(batch["ID"][index]),
                    "patient_group_id": str(batch["patient_group_id"][index]),
                    "fold": fold,
                    "split_role": split_role,
                    "strategy": "full",
                    "variant": variant,
                    "true_label": int(label),
                    "prob_normal": float(probabilities[index, 0]),
                    "prob_mild": float(probabilities[index, 1]),
                    "prob_severe": float(probabilities[index, 2]),
                    "pred_class": int(probabilities[index].argmax()),
                }
            )
    frame = pd.DataFrame(rows)
    metrics = compute_classification_metrics(
        frame["true_label"].to_numpy(int),
        frame[["prob_normal", "prob_mild", "prob_severe"]].to_numpy(float),
    )
    serializable: dict[str, Any] = flatten_metrics(metrics)
    serializable["confusion_matrix"] = np.asarray(metrics["confusion_matrix"]).tolist()
    serializable.update({"rows": len(frame), "variant": variant, "fold": fold, "split_role": split_role, "strategy": "full"})
    return frame, serializable


def run_full_reference_evaluation(
    config_path: Path,
    output_root: Path,
    *,
    overwrite: bool,
) -> dict[str, Any]:
    config = load_yaml(config_path)
    historical_root = project_path(config["full_reference"]["read_only_root"])
    before = historical_hash_inventory(config)
    runtime_records: list[dict[str, Any]] = []
    checkpoint_runs: list[tuple[str, int, Path, Path]] = []
    for variant in ALLOWED_VARIANTS:
        for fold in range(5):
            run = historical_root / variant / f"fold_{fold}"
            checkpoint_path = run / "best_macro_auc.pth"
            checkpoint = load_torch_checkpoint(checkpoint_path, "cpu")
            matches, runtime = _runtime_matches(checkpoint)
            runtime_records.append({"variant": variant, "fold": fold, "matches": matches, **runtime})
            checkpoint_runs.append((variant, fold, run, checkpoint_path))
            del checkpoint
    if not all(item["matches"] for item in runtime_records):
        audit = {
            "schema_version": "global_optical_fusion_full_reference_audit_v1",
            "status": "RUNTIME_MISMATCH",
            "passed_checkpoints": 0,
            "required_checkpoints": 10,
            "tolerance": float(config["full_reference"]["validation_auc_tolerance"]),
            "runtime_records": runtime_records,
            "historical_hashes_before": before,
            "historical_hashes_after": historical_hash_inventory(config),
            "historical_inputs_modified": False,
        }
        _json_write(output_root / "full_reference" / "reference_audit.json", audit)
        return audit
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    results: list[dict[str, Any]] = []
    tolerance = float(config["full_reference"]["validation_auc_tolerance"])
    transform_config = config["transforms"]
    transform = build_transforms(
        "val",
        int(transform_config["image_size"]),
        transform_config["mean"],
        transform_config["std"],
        False,
    )
    for variant, fold, historical_run, checkpoint_path in checkpoint_runs:
        set_random_seed(int(config["train"]["seed"]) + fold)
        checkpoint = load_torch_checkpoint(checkpoint_path, device)
        output = output_root / "full_reference" / variant / f"fold_{fold}"
        output.mkdir(parents=True, exist_ok=True)
        train_path = project_path(config["data"]["split_root"]) / str(config["data"]["train_csv_pattern"]).format(fold=fold)
        val_path = project_path(config["data"]["split_root"]) / str(config["data"]["val_csv_pattern"]).format(fold=fold)
        train_frame = pd.read_csv(train_path, dtype={"ID": "string"}, encoding="utf-8-sig")
        val_frame = pd.read_csv(val_path, dtype={"ID": "string"}, encoding="utf-8-sig")
        train_features, val_features, scaler, _ = prepare_features(
            config,
            variant,
            fold,
            train_frame["ID"].astype(str).tolist(),
            val_frame["ID"].astype(str).tolist(),
            allow_source_superset=False,
        )
        kwargs = {
            "variant": variant,
            "fold": fold,
            "transform": transform,
            "image_root": project_path(config["data"]["image_root"]),
            "image_filename_template": config["data"]["image_filename_template"],
            "scaler": scaler,
        }
        train_dataset = GlobalOpticalFusionDataset(train_path, split_role="train", feature_frame=train_features, **kwargs)
        val_dataset = GlobalOpticalFusionDataset(val_path, split_role="val", feature_frame=val_features, **kwargs)
        train_loader = DataLoader(train_dataset, batch_size=16, shuffle=False, num_workers=0, pin_memory=False)
        val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False, num_workers=0, pin_memory=False)
        model = ResNet18OpticalFusion(variant, pretrained=False).to(device)
        model.load_state_dict(checkpoint["model_state_dict"], strict=True)
        train_predictions, train_metrics = _predict_full(model, train_loader, device, variant=variant, fold=fold, split_role="train")
        val_predictions, val_metrics = _predict_full(model, val_loader, device, variant=variant, fold=fold, split_role="val")
        train_metrics["weighted_cross_entropy"] = _weighted_cross_entropy_from_predictions(
            train_predictions, checkpoint["class_weights"]
        )
        val_metrics["weighted_cross_entropy"] = _weighted_cross_entropy_from_predictions(
            val_predictions, checkpoint["class_weights"]
        )
        recorded_metrics = json.loads((historical_run / "metrics.json").read_text(encoding="utf-8"))
        stored_predictions = pd.read_csv(historical_run / "val_predictions.csv", encoding="utf-8-sig")
        stored_auc = float(compute_classification_metrics(
            stored_predictions["true_label"].to_numpy(int),
            stored_predictions[["prob_normal", "prob_mild", "prob_severe"]].to_numpy(float),
        )["macro_auc"])
        recorded_auc = float(recorded_metrics["macro_auc"])
        reproduced_auc = float(val_metrics["macro_auc"])
        delta = abs(reproduced_auc - recorded_auc)
        passed = abs(stored_auc - recorded_auc) <= 1.0e-12 and delta <= tolerance
        train_predictions.to_csv(output / "train_predictions.csv", index=False, encoding="utf-8-sig")
        val_predictions.to_csv(output / "val_predictions.csv", index=False, encoding="utf-8-sig")
        train_predictions.to_csv(output / "deterministic_train_predictions.csv", index=False, encoding="utf-8-sig")
        val_predictions.to_csv(output / "deterministic_val_predictions.csv", index=False, encoding="utf-8-sig")
        _json_write(output / "train_metrics.json", train_metrics)
        _json_write(output / "val_metrics.json", val_metrics)
        _json_write(output / "deterministic_train_metrics.json", train_metrics)
        _json_write(output / "deterministic_val_metrics.json", val_metrics)
        history = pd.read_csv(historical_run / "training_log.csv")

        def first_threshold(threshold: float) -> float:
            reached = history.loc[history["train_macro_auc"] >= threshold, "epoch"]
            return float(reached.iloc[0]) if not reached.empty else float("nan")

        last_val_auc = float(history.iloc[-1]["val_macro_auc"])
        reference_metrics = {
            "strategy": "full",
            "variant": variant,
            "fold": fold,
            "best_epoch": int(checkpoint["best_epoch"]),
            "completed_epoch": int(history["epoch"].max()),
            "recorded_val_macro_auc": recorded_auc,
            "stored_prediction_val_macro_auc": stored_auc,
            "deterministic_train_macro_auc": float(train_metrics["macro_auc"]),
            "deterministic_val_macro_auc": reproduced_auc,
            "reproduced_val_macro_auc": reproduced_auc,
            "absolute_delta": delta,
            "tolerance": tolerance,
            "passed": passed,
            "train_macro_auc": float(train_metrics["macro_auc"]),
            "val_macro_auc": reproduced_auc,
            "generalization_gap_macro_auc": float(train_metrics["macro_auc"] - reproduced_auc),
            "macro_auc_generalization_gap": float(train_metrics["macro_auc"] - reproduced_auc),
            "best_validation_macro_auc": float(checkpoint["best_macro_auc"]),
            "last_validation_macro_auc": last_val_auc,
            "validation_drop": float(checkpoint["best_macro_auc"] - last_val_auc),
            "first_epoch_train_auc_ge_0_90": first_threshold(0.90),
            "first_epoch_train_auc_ge_0_95": first_threshold(0.95),
            "first_epoch_train_auc_ge_0_99": first_threshold(0.99),
            "deterministic_train_accuracy": float(train_metrics["accuracy"]),
            "deterministic_val_accuracy": float(val_metrics["accuracy"]),
            "generalization_gap_accuracy": float(train_metrics["accuracy"] - val_metrics["accuracy"]),
            "deterministic_train_balanced_accuracy": float(train_metrics["balanced_accuracy"]),
            "deterministic_val_balanced_accuracy": float(val_metrics["balanced_accuracy"]),
            "generalization_gap_balanced_accuracy": float(train_metrics["balanced_accuracy"] - val_metrics["balanced_accuracy"]),
            "deterministic_train_macro_f1": float(train_metrics["macro_f1"]),
            "deterministic_val_macro_f1": float(val_metrics["macro_f1"]),
            "generalization_gap_macro_f1": float(train_metrics["macro_f1"] - val_metrics["macro_f1"]),
            "train_weighted_cross_entropy": float(train_metrics["weighted_cross_entropy"]),
            "val_weighted_cross_entropy": float(val_metrics["weighted_cross_entropy"]),
            "loss_generalization_gap": float(
                val_metrics["weighted_cross_entropy"]
                - train_metrics["weighted_cross_entropy"]
            ),
            **{
                f"{name}_auc_generalization_gap": float(
                    train_metrics[f"auc_{name}"] - val_metrics[f"auc_{name}"]
                )
                for name in ("normal", "mild", "severe")
            },
            **{
                f"deterministic_val_{name}_{metric}": float(val_metrics[f"{metric}_{name}"])
                for name in ("normal", "mild", "severe")
                for metric in ("auc", "recall")
            },
            "trainable_parameter_count": int(checkpoint["trainable_parameter_count"]),
            "total_parameter_count": int(checkpoint["parameter_count"]),
            "trainable_parameter_ratio": float(checkpoint["trainable_parameter_count"] / checkpoint["parameter_count"]),
            "peak_gpu_memory_bytes": None,
            "training_time_seconds": float(history["elapsed_seconds"].sum()),
            "historical_checkpoint_sha256": sha256_file(historical_run / "best_macro_auc.pth"),
        }
        _json_write(output / "overfit_metrics.json", reference_metrics)
        results.append(reference_metrics)
        print(f"FULL_REFERENCE variant={variant} fold={fold} passed={passed} delta={delta:.3g}", flush=True)
        del checkpoint, model
    after = historical_hash_inventory(config)
    unchanged = before == after
    status = "PASS" if unchanged and len(results) == 10 and all(item["passed"] for item in results) else "FAIL"
    audit = {
        "schema_version": "global_optical_fusion_full_reference_audit_v1",
        "status": status,
        "passed_checkpoints": sum(bool(item["passed"]) for item in results),
        "required_checkpoints": 10,
        "tolerance": tolerance,
        "results": results,
        "runtime_records": runtime_records,
        "historical_hashes_before": before,
        "historical_hashes_after": after,
        "historical_inputs_modified": not unchanged,
    }
    _json_write(output_root / "full_reference" / "reference_audit.json", audit)
    return audit


def run_required_tests(config_path: Path, output_root: Path) -> None:
    command = [sys.executable, "-m", "pytest", "-p", "no:cacheprovider", "tests", "-q"]
    completed = subprocess.run(command, cwd=PROJECT_ROOT, capture_output=True, text=True)
    payload = {
        "status": "PASS" if completed.returncode == 0 else "FAIL",
        "returncode": completed.returncode,
        "output": (completed.stdout + completed.stderr)[-20000:],
        "config_sha256": sha256_file(config_path),
    }
    _json_write(output_root / "protocol" / "test_audit.json", payload)
    if completed.returncode != 0:
        raise RuntimeError(f"Required tests failed; see {output_root / 'protocol/test_audit.json'}")
    print(f"TEST_STATUS=PASS ({completed.stdout.strip().splitlines()[-1]})")


def _train_command(
    config_path: Path,
    output_root: Path,
    strategy: str,
    variant: str,
    fold: int,
    *,
    smoke: bool,
    resume: bool,
    overwrite: bool,
    allow_cpu_training: bool,
    allow_reference_mismatch: bool = False,
) -> list[str]:
    command = [
        sys.executable,
        str(PROJECT_ROOT / "scripts/train/train_global_optical_fusion_overfit_control_5fold.py"),
        "--config",
        str(config_path),
        "--strategy",
        strategy,
        "--variant",
        variant,
        "--fold",
        str(fold),
        "--output-root",
        str(output_root),
    ]
    if smoke:
        command.append("--smoke-test")
    if resume:
        command.append("--resume")
    if overwrite:
        command.append("--overwrite")
    if allow_cpu_training:
        command.append("--allow-cpu-training")
    if allow_reference_mismatch:
        command.append("--allow-reference-mismatch")
    return command


def run_smoke_suite(config_path: Path, output_root: Path, *, overwrite: bool) -> None:
    records = []
    for strategy in TRAINABILITY_STRATEGIES:
        for variant in ALLOWED_VARIANTS:
            run = output_root / "smoke" / strategy / variant / "fold_0"
            if run.exists() and not overwrite:
                manifest_path = run / "fold_manifest.json"
                if manifest_path.is_file() and json.loads(manifest_path.read_text(encoding="utf-8")).get("status") == "SMOKE_COMPLETE":
                    records.append({"strategy": strategy, "variant": variant, "status": "REUSED"})
                    continue
            subprocess.run(
                _train_command(config_path, output_root, strategy, variant, 0, smoke=True, resume=False, overwrite=True, allow_cpu_training=True, allow_reference_mismatch=False),
                cwd=PROJECT_ROOT,
                check=True,
            )
            subprocess.run(
                _train_command(config_path, output_root, strategy, variant, 0, smoke=True, resume=True, overwrite=False, allow_cpu_training=True, allow_reference_mismatch=False),
                cwd=PROJECT_ROOT,
                check=True,
            )
            manifest = json.loads((run / "fold_manifest.json").read_text(encoding="utf-8"))
            audit = json.loads((run / "trainability_audit.json").read_text(encoding="utf-8"))
            required = (
                "best_macro_auc.pth",
                "last_checkpoint.pth",
                "deterministic_train_predictions.csv",
                "deterministic_val_predictions.csv",
                "deterministic_train_metrics.json",
                "deterministic_val_metrics.json",
                "overfit_metrics.json",
                "training_log.csv",
            )
            valid = manifest.get("status") == "SMOKE_COMPLETE" and all((run / name).is_file() for name in required) and audit.get("strategy") == strategy
            if not valid:
                raise RuntimeError(f"Smoke artifact validation failed: {run}")
            records.append(
                {
                    "strategy": strategy,
                    "variant": variant,
                    "status": "PASS",
                    "resume_validated": True,
                }
            )
    _json_write(output_root / "protocol" / "smoke_audit.json", {"status": "PASS", "runs": records})
    print("SMOKE_STATUS=PASS (4/4)")


def run_formal_matrix(args: argparse.Namespace, config_path: Path, output_root: Path) -> None:
    for strategy in _strategy_values(args.strategy):
        for variant in _variant_values(args.variant):
            for fold in _fold_values(args.fold):
                run = output_root / strategy / variant / f"fold_{fold}"
                if args.skip_completed and (run / "fold_manifest.json").is_file():
                    manifest = json.loads((run / "fold_manifest.json").read_text(encoding="utf-8"))
                    if manifest.get("status") == "COMPLETE" and manifest.get("formal_result"):
                        print(f"SKIP_COMPLETED={strategy}/{variant}/fold_{fold}")
                        continue
                subprocess.run(
                    _train_command(
                        config_path,
                        output_root,
                        strategy,
                        variant,
                        fold,
                        smoke=False,
                        resume=args.resume,
                        overwrite=args.overwrite,
                        allow_cpu_training=args.allow_cpu_training,
                        allow_reference_mismatch=args.allow_reference_mismatch,
                    ),
                    cwd=PROJECT_ROOT,
                    check=True,
                )


def run_summary(
    config_path: Path,
    output_root: Path,
    *,
    allow_reference_mismatch: bool = False,
) -> None:
    command = [
            sys.executable,
            str(PROJECT_ROOT / "scripts/evaluate/summarize_global_optical_fusion_overfit_control.py"),
            "--config",
            str(config_path),
            "--output-root",
            str(output_root),
        ]
    if allow_reference_mismatch:
        command.append("--allow-reference-mismatch")
    subprocess.run(
        command,
        cwd=PROJECT_ROOT,
        check=True,
    )


def final_status(output_root: Path) -> str:
    reference = output_root / "full_reference" / "reference_audit.json"
    reference_status = (
        json.loads(reference.read_text(encoding="utf-8")).get("status")
        if reference.is_file()
        else "NOT_RUN"
    )
    complete = 0
    for strategy in TRAINABILITY_STRATEGIES:
        for variant in ALLOWED_VARIANTS:
            for fold in range(5):
                manifest = output_root / strategy / variant / f"fold_{fold}" / "fold_manifest.json"
                if manifest.is_file() and json.loads(manifest.read_text(encoding="utf-8")).get("status") == "COMPLETE":
                    complete += 1
    summary_path = output_root / "summary" / "summary.json"
    if complete == 20 and summary_path.is_file():
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        reference_accepted = reference_status == "PASS" or bool(
            summary.get("full_reference_reproduction_override")
        )
        if summary.get("status") == "COMPLETE" and reference_accepted:
            return "COMPLETE"
    if complete > 0:
        return "PARTIAL"
    smoke_path = output_root / "protocol" / "smoke_audit.json"
    test_path = output_root / "protocol" / "test_audit.json"
    smoke_passed = smoke_path.is_file() and json.loads(
        smoke_path.read_text(encoding="utf-8")
    ).get("status") == "PASS"
    tests_passed = test_path.is_file() and json.loads(
        test_path.read_text(encoding="utf-8")
    ).get("status") == "PASS"
    if smoke_passed and tests_passed and reference_status in {"PASS", "RUNTIME_MISMATCH"}:
        return "READY_FOR_LOCAL_GPU_TRAINING"
    return "PARTIAL"


def main() -> None:
    args = parse_args()
    config_path = project_path(args.config)
    config = load_yaml(config_path)
    output_root = project_path(config["experiment"]["output_root"])
    mutually_exclusive = sum(
        bool(value)
        for value in (
            args.protocol_only,
            args.reference_eval_only,
            args.smoke_test,
            args.train_only,
            args.summarize_only,
        )
    )
    if mutually_exclusive > 1:
        raise ValueError("Select at most one workflow-only flag")
    run_preflight(config_path, output_root)
    print("PROTOCOL_STATUS=PASS")
    if args.protocol_only:
        print(f"OVERFIT_CONTROL_STATUS={final_status(output_root)}")
        return
    if args.reference_eval_only:
        audit = run_full_reference_evaluation(config_path, output_root, overwrite=args.overwrite)
        print(f"FULL_REFERENCE_STATUS={audit['status']}")
        print(f"OVERFIT_CONTROL_STATUS={final_status(output_root)}")
        return
    if args.summarize_only:
        run_summary(
            config_path,
            output_root,
            allow_reference_mismatch=args.allow_reference_mismatch,
        )
        print(f"OVERFIT_CONTROL_STATUS={final_status(output_root)}")
        return
    if args.smoke_test:
        run_required_tests(config_path, output_root)
        run_smoke_suite(config_path, output_root, overwrite=args.overwrite)
        print(f"OVERFIT_CONTROL_STATUS={final_status(output_root)}")
        return
    if args.train_only:
        run_formal_matrix(args, config_path, output_root)
        print(f"OVERFIT_CONTROL_STATUS={final_status(output_root)}")
        return
    reference = run_full_reference_evaluation(config_path, output_root, overwrite=args.overwrite)
    if reference["status"] != "PASS" and not args.allow_reference_mismatch:
        print(f"FULL_REFERENCE_STATUS={reference['status']}")
        print(f"OVERFIT_CONTROL_STATUS={final_status(output_root)}")
        return
    if reference["status"] != "PASS":
        print(
            "FULL_REFERENCE_OVERRIDE=true "
            f"status={reference['status']} passed={reference.get('passed_checkpoints', 0)}/10"
        )
    run_required_tests(config_path, output_root)
    run_smoke_suite(config_path, output_root, overwrite=args.overwrite)
    if not torch.cuda.is_available() and not args.allow_cpu_training:
        print("OVERFIT_CONTROL_STATUS=READY_FOR_LOCAL_GPU_TRAINING")
        return
    run_formal_matrix(args, config_path, output_root)
    if args.strategy == "all" and args.variant == "all" and args.fold == "all":
        run_summary(
            config_path,
            output_root,
            allow_reference_mismatch=args.allow_reference_mismatch,
        )
    print(f"OVERFIT_CONTROL_STATUS={final_status(output_root)}")


if __name__ == "__main__":
    main()
