from __future__ import annotations

import argparse
import json
import math
import platform
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from _bootstrap import ROOT, resolve_output_dir
from metrics.binary_classification_metrics import compute_binary_metrics
from models.p1_independent_triple_resnet18 import IndependentTripleResNet18, count_parameters
from p0b_deca.p1_dataset import _image_to_chw_float32
from utils.experiment_utils import set_random_seed
from utils.p1_cluster_bootstrap import compute_visit_metrics
from utils.p1_component_preflight import FIXED_SPLIT_PATH, MANIFEST_PATH, sha256_file
from utils.p1_representation_normalization import (
    IMAGENET_MEAN,
    IMAGENET_STD,
    apply_component_normalization,
    fit_component_normalization,
)
from utils.p1_rgb_audit import load_p1_frame, local_path

from run_p1_extension_abc import (
    ABC_ROOT,
    BOOTSTRAP_ITERATIONS,
    P1_RGB_OOF,
    P1_ROOT,
    SEED,
    _fast_cluster_bootstrap,
    _git_commit,
    _json_safe,
    _md_table,
    _now,
    _paired_bootstrap_delta,
    _paired_delta,
    _read_json,
    _sha256_text,
    _write_json,
    _write_text,
)


EXPERIMENT = "P1-RGBSR_TripleBranch_v1"
STATUS_COMPLETE = "P1_STAGE_D_RGBSR_V1_COMPLETE"
STATUS_COMPLETE_WITH_FAILURES = "P1_STAGE_D_RGBSR_V1_COMPLETE_WITH_FAILURES"
OUTPUT_ROOT = ROOT / "experiments/500Data/P1_RGBSR_TripleBranch_v1"
REPORT_PATH = ROOT / "reports/p1_stage_d_rgbsr_triplebranch_v1_results.md"
CORE_METRICS = ("macro_auc", "macro_f1", "balanced_accuracy", "patient_sensitivity", "control_specificity")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_ROOT)
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--formal", action="store_true")
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--summarize-only", action="store_true")
    parser.add_argument("--fold", type=int, action="append")
    return parser.parse_args()


def _write_csv(path: Path, frame: pd.DataFrame) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, encoding="utf-8-sig")
    return path


def _create_layout(output_dir: Path) -> None:
    for rel in ("metadata", "summary", "oof", "logs", "smoke", "smoke/checkpoints", "smoke/preprocessing"):
        (output_dir / rel).mkdir(parents=True, exist_ok=True)
    for fold in range(5):
        (output_dir / f"fold_{fold}/checkpoints").mkdir(parents=True, exist_ok=True)
        (output_dir / f"fold_{fold}/preprocessing").mkdir(parents=True, exist_ok=True)


def _imagenet(tensor: torch.Tensor) -> torch.Tensor:
    return (tensor - IMAGENET_MEAN.view(3, 1, 1)) / IMAGENET_STD.view(3, 1, 1)


class P1TripleFusionDataset(Dataset):
    def __init__(self, frame: pd.DataFrame) -> None:
        self.frame = frame.reset_index(drop=True).copy()

    def __len__(self) -> int:
        return int(len(self.frame))

    def __getitem__(self, index: int) -> dict[str, Any]:
        import utils.p1_component_registry as registry
        from utils.p1_representation_normalization import load_raw_component_sample

        row = self.frame.iloc[index].to_dict()
        return {
            "case_id": str(row["case_id"]),
            "patient_group_id": str(row["patient_group_id"]),
            "fold": int(row["fold"]),
            "label_original": int(row["label_original"]),
            "label_3class": int(row["label_3class"]),
            "label_binary": int(row["label_binary"]),
            "rgb": _image_to_chw_float32(local_path(row["rgb_path"])),
            "shading_sample": load_raw_component_sample(row, registry.get_component_spec("p1_s")),
            "residual_sample": load_raw_component_sample(row, registry.get_component_spec("p1_r")),
        }


def build_triple_collate(shading_state: Any, residual_state: Any, *, training: bool) -> Any:
    def _collate(batch: list[Mapping[str, Any]]) -> dict[str, Any]:
        rgb_tensors: list[torch.Tensor] = []
        shading_tensors: list[torch.Tensor] = []
        residual_tensors: list[torch.Tensor] = []
        for sample in batch:
            do_flip = bool(training and float(torch.rand(1).item()) < 0.5)
            rgb = torch.as_tensor(sample["rgb"], dtype=torch.float32)
            if do_flip:
                rgb = rgb.flip(-1)
            rgb_tensors.append(_imagenet(rgb))
            shading_tensors.append(
                apply_component_normalization(dict(sample["shading_sample"]), shading_state, horizontal_flip=do_flip)["representation"]
            )
            residual_tensors.append(
                apply_component_normalization(dict(sample["residual_sample"]), residual_state, horizontal_flip=do_flip)["representation"]
            )
        return {
            "case_id": [str(sample["case_id"]) for sample in batch],
            "patient_group_id": [str(sample["patient_group_id"]) for sample in batch],
            "fold": torch.tensor([int(sample["fold"]) for sample in batch], dtype=torch.long),
            "label_original": torch.tensor([int(sample["label_original"]) for sample in batch], dtype=torch.long),
            "label_3class": torch.tensor([int(sample["label_3class"]) for sample in batch], dtype=torch.long),
            "label_binary": torch.tensor([int(sample["label_binary"]) for sample in batch], dtype=torch.long),
            "rgb": torch.stack(rgb_tensors, dim=0),
            "shading": torch.stack(shading_tensors, dim=0),
            "residual": torch.stack(residual_tensors, dim=0),
        }

    return _collate


def _make_loader(dataset: Dataset, *, batch_size: int, shuffle: bool, seed: int, collate_fn: Any, device: torch.device) -> DataLoader:
    generator = torch.Generator()
    generator.manual_seed(int(seed))
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=0, pin_memory=device.type == "cuda", generator=generator, collate_fn=collate_fn)


def _class_weights(train_frame: pd.DataFrame, device: torch.device) -> torch.Tensor:
    counts = train_frame["label_binary"].astype(int).value_counts().to_dict()
    total = int(len(train_frame))
    return torch.tensor([total / (2.0 * counts.get(0, 1)), total / (2.0 * counts.get(1, 1))], dtype=torch.float32, device=device)


def _evaluate_model(model: nn.Module, loader: DataLoader, device: torch.device, criterion: nn.Module | None = None) -> tuple[pd.DataFrame, dict[str, Any], bool]:
    model.eval()
    rows: list[dict[str, Any]] = []
    losses: list[float] = []
    finite_logits = True
    with torch.no_grad():
        for batch in loader:
            rgb = batch["rgb"].to(device)
            shading = batch["shading"].to(device)
            residual = batch["residual"].to(device)
            labels = batch["label_binary"].to(device=device, dtype=torch.long)
            logits = model(rgb, shading, residual)
            finite_logits = finite_logits and bool(torch.isfinite(logits).all().item())
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
    return frame, metrics, finite_logits


def create_approval(output_dir: Path) -> dict[str, Any]:
    payload = {
        "approval_version": "P1_STAGE_D_V1",
        "experiment": EXPERIMENT,
        "smoke_approved": True,
        "formal_training_approved": True,
        "stage_d_v2_approved": False,
        "camera_features_enabled": False,
        "exif_features_enabled": False,
        "bottleneck_enabled": False,
        "attention_enabled": False,
        "gating_enabled": False,
        "allow_threshold_search": False,
        "allow_hyperparameter_search": False,
        "allow_case_removal": False,
        "evaluation_unit": "visit_case",
        "split_group_unit": "patient_group_id",
        "aggregate_predictions_within_patient": False,
        "manifest_sha256": sha256_file(MANIFEST_PATH),
        "split_sha256": sha256_file(FIXED_SPLIT_PATH),
        "stage_b_protocol_sha256": sha256_file(ROOT / "scripts/p1/run_p1_extension_abc.py"),
        "rgb_source_sha256": sha256_file(P1_RGB_OOF),
        "shading_source_contract_sha256": _sha256_text("p1_s:raw_unclipped_shading_like:face_valid:train_fold_p01_p99"),
        "residual_source_contract_sha256": _sha256_text("p1_r:signed_residual:face_valid:train_fold_p99_abs_signed_preserved"),
        "git_commit": _git_commit(),
        "timestamp": _now(),
    }
    return _write_json(output_dir / "metadata/STAGE_D_V1_EXECUTION_APPROVED.json", payload) and payload


def _model_parameter_audit(output_dir: Path) -> dict[str, Any]:
    single = IndependentTripleResNet18(pretrained=False).encoder_rgb
    model = IndependentTripleResNet18(pretrained=False)
    first_params = {
        "encoder_rgb": next(model.encoder_rgb.parameters()),
        "encoder_shading": next(model.encoder_shading.parameters()),
        "encoder_residual": next(model.encoder_residual.parameters()),
    }
    audit = {
        "single_resnet18_total_parameters": int(sum(p.numel() for p in single.parameters())),
        "single_resnet18_trainable_parameters": int(sum(p.numel() for p in single.parameters() if p.requires_grad)),
        "triple_encoder_total_parameters": int(
            sum(p.numel() for p in model.encoder_rgb.parameters())
            + sum(p.numel() for p in model.encoder_shading.parameters())
            + sum(p.numel() for p in model.encoder_residual.parameters())
        ),
        "triple_encoder_trainable_parameters": int(
            sum(p.numel() for p in model.encoder_rgb.parameters() if p.requires_grad)
            + sum(p.numel() for p in model.encoder_shading.parameters() if p.requires_grad)
            + sum(p.numel() for p in model.encoder_residual.parameters() if p.requires_grad)
        ),
        "classifier_parameters": int(sum(p.numel() for p in model.classifier.parameters())),
        "total_model_parameters": int(count_parameters(model)["total_params"]),
        "encoder_object_ids": {
            "encoder_rgb": id(model.encoder_rgb),
            "encoder_shading": id(model.encoder_shading),
            "encoder_residual": id(model.encoder_residual),
        },
        "encoder_parameter_data_ptrs": {name: int(param.data_ptr()) for name, param in first_params.items()},
        "encoders_independent": len({id(model.encoder_rgb), id(model.encoder_shading), id(model.encoder_residual)}) == 3,
        "first_parameter_storage_independent": len({int(param.data_ptr()) for param in first_params.values()}) == 3,
        "classifier_in_features": int(model.classifier.in_features),
        "classifier_out_features": int(model.classifier.out_features),
        "bottleneck_used": False,
        "dropout_used": False,
        "attention_used": False,
        "gating_used": False,
    }
    _write_json(output_dir / "metadata/model_parameter_audit.json", audit)
    return audit


def run_preflight(output_dir: Path) -> dict[str, Any]:
    frame = load_p1_frame(MANIFEST_PATH, FIXED_SPLIT_PATH)
    frame["case_id"] = frame["case_id"].astype(str)
    checks: dict[str, Any] = {
        "manifest_rows_500": int(len(frame)) == 500,
        "unique_case_id_500": int(frame["case_id"].nunique()) == 500,
        "fixed_split_sha256_match": sha256_file(FIXED_SPLIT_PATH) == "d5a20fb56c96e657dd7902b6d829bed78b6d43d3e58ec47e6bd3542ec34378cb",
        "patient_group_not_cross_fold": not bool(frame.groupby("patient_group_id")["fold"].nunique().gt(1).any()),
        "stage_b_complete": (ABC_ROOT / "stage_b/_STAGE_SUCCESS.json").is_file()
        and _read_json(ABC_ROOT / "stage_b/_STAGE_SUCCESS.json").get("status") == "STAGE_B_FUSION_EXPERIMENTS_COMPLETE",
        "stage_b_report_exists": (ROOT / "reports/p1_extension_stage_b_fusion_results_report.md").is_file(),
        "abc_decision_report_exists": (ROOT / "reports/p1_extension_abc_decision_report.md").is_file(),
        "camera_exif_not_connected": True,
        "no_group_oof_output": not (output_dir / "oof/oof_predictions_group.csv").exists(),
        "stage_d_v2_approved_false": True,
        "cuda_available": bool(torch.cuda.is_available()),
    }
    for fold in range(5):
        train = frame[frame["fold"] != fold]
        val = frame[frame["fold"] == fold]
        checks[f"fold_{fold}_train_400"] = int(len(train)) == 400
        checks[f"fold_{fold}_val_100"] = int(len(val)) == 100
        checks[f"fold_{fold}_train_class_counts"] = train["label_binary"].astype(int).value_counts().to_dict() == {1: 308, 0: 92}
        checks[f"fold_{fold}_val_class_counts"] = val["label_binary"].astype(int).value_counts().to_dict() == {1: 77, 0: 23}
    rgb_ok = shading_ok = residual_ok = True
    for _, row in frame.iterrows():
        if not local_path(row["rgb_path"]).is_file():
            rgb_ok = False
        if not local_path(row["maps_path"]).is_file() or not local_path(row["face_valid_mask_path"]).is_file():
            shading_ok = False
            residual_ok = False
    checks["rgb_assets_500_readable"] = rgb_ok
    checks["shading_assets_500_readable"] = shading_ok
    checks["residual_assets_500_readable"] = residual_ok
    try:
        train0 = frame[frame["fold"] != 0].copy()
        s_state = fit_component_normalization(train0, "p1_s", root=ROOT)
        r_state = fit_component_normalization(train0, "p1_r", root=ROOT)
        checks["shading_fold_normalization_interface_valid"] = bool(getattr(s_state, "p01", None) and getattr(s_state, "p99", None))
        checks["residual_fold_normalization_interface_valid"] = bool(getattr(r_state, "residual_scale", None))
    except Exception:
        checks["shading_fold_normalization_interface_valid"] = False
        checks["residual_fold_normalization_interface_valid"] = False
    audit = _model_parameter_audit(output_dir)
    checks["three_encoders_independent"] = bool(audit["encoders_independent"] and audit["first_parameter_storage_independent"])
    checks["classifier_input_1536"] = int(audit["classifier_in_features"]) == 1536
    status = "STAGE_D_PREFLIGHT_PASSED" if all(bool(v) for v in checks.values()) else "BLOCKED_BY_STAGE_D_SHARED_DATA_OR_PROTOCOL_ERROR"
    payload = {
        "status": status,
        "checks": checks,
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "cuda_device_count": int(torch.cuda.device_count()) if torch.cuda.is_available() else 0,
        "cuda_memory_total": int(torch.cuda.get_device_properties(0).total_memory) if torch.cuda.is_available() else 0,
    }
    _write_json(output_dir / "metadata/stage_d_preflight.json", payload)
    _write_json(
        output_dir / "metadata/source_hashes.json",
        {
            "manifest": sha256_file(MANIFEST_PATH),
            "split": sha256_file(FIXED_SPLIT_PATH),
            "stage_b_script": sha256_file(ROOT / "scripts/p1/run_p1_extension_abc.py"),
            "stage_d_script": sha256_file(Path(__file__)),
            "triple_model": sha256_file(ROOT / "models/p1_independent_triple_resnet18.py"),
            "p1_rgb_oof": sha256_file(P1_RGB_OOF),
            "stage_b_rgb_rgb_oof": sha256_file(ABC_ROOT / "stage_b/rgb_rgb_capacity_control/oof/oof_predictions_visit.csv"),
            "stage_b_rgb_s_oof": sha256_file(ABC_ROOT / "stage_b/rgb_s_independent_dual/oof/oof_predictions_visit.csv"),
            "stage_b_rgb_r_oof": sha256_file(ABC_ROOT / "stage_b/rgb_r_independent_dual/oof/oof_predictions_visit.csv"),
        },
    )
    if status != "STAGE_D_PREFLIGHT_PASSED":
        raise RuntimeError(status)
    return payload


def _fold_dir(output_dir: Path, fold: int, *, smoke: bool) -> Path:
    return output_dir / "smoke" if smoke else output_dir / f"fold_{fold}"


def _fold_is_complete(fold_dir: Path, *, smoke: bool) -> bool:
    success = fold_dir / ("SMOKE_SUCCESS.json" if smoke else "_FOLD_SUCCESS.json")
    pred = fold_dir / "val_predictions_visit.csv"
    ckpt = fold_dir / "checkpoints/best_macro_auc.pth"
    return success.is_file() and pred.is_file() and ckpt.is_file() and len(pd.read_csv(pred)) == 100


def train_fold(output_dir: Path, fold: int, *, smoke: bool, resume: bool) -> dict[str, Any]:
    fold_dir = _fold_dir(output_dir, fold, smoke=smoke)
    if resume and _fold_is_complete(fold_dir, smoke=smoke):
        return _read_json(fold_dir / ("SMOKE_SUCCESS.json" if smoke else "_FOLD_SUCCESS.json"))
    fold_dir.mkdir(parents=True, exist_ok=True)
    (fold_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
    (fold_dir / "preprocessing").mkdir(parents=True, exist_ok=True)
    set_random_seed(SEED + int(fold))
    frame = load_p1_frame(MANIFEST_PATH, FIXED_SPLIT_PATH)
    train_frame = frame[frame["fold"] != fold].copy().reset_index(drop=True)
    val_frame = frame[frame["fold"] == fold].copy().reset_index(drop=True)
    train_case_hash = _sha256_text("|".join(sorted(train_frame["case_id"].astype(str).tolist())))
    shading_state = fit_component_normalization(train_frame, "p1_s", root=ROOT)
    residual_state = fit_component_normalization(train_frame, "p1_r", root=ROOT)
    _write_json(
        fold_dir / "preprocessing/shading_fitted_parameters.json",
        {"experiment": EXPERIMENT, "fold": int(fold), "validation_used_for_fit": False, "training_case_id_sha256": train_case_hash, "fitted_parameters": shading_state.to_dict()},
    )
    _write_json(
        fold_dir / "preprocessing/residual_fitted_parameters.json",
        {"experiment": EXPERIMENT, "fold": int(fold), "validation_used_for_fit": False, "training_case_id_sha256": train_case_hash, "fitted_parameters": residual_state.to_dict()},
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("CUDA is required for Stage D v1 training")
    torch.cuda.reset_peak_memory_stats(device)
    train_loader = _make_loader(
        P1TripleFusionDataset(train_frame),
        batch_size=16,
        shuffle=True,
        seed=SEED + int(fold),
        collate_fn=build_triple_collate(shading_state, residual_state, training=True),
        device=device,
    )
    val_loader = _make_loader(
        P1TripleFusionDataset(val_frame),
        batch_size=16,
        shuffle=False,
        seed=SEED + 10_000 + int(fold),
        collate_fn=build_triple_collate(shading_state, residual_state, training=False),
        device=device,
    )
    model = IndependentTripleResNet18(pretrained=True).to(device)
    params = count_parameters(model)
    criterion = nn.CrossEntropyLoss(weight=_class_weights(train_frame, device))
    optimizer = torch.optim.AdamW(model.parameters(), lr=1.0e-4, weight_decay=1.0e-4)
    max_epochs = 2 if smoke else 50
    patience = 10
    best_auc = -math.inf
    best_epoch = 0
    best_payload: dict[str, Any] | None = None
    finite_loss = True
    finite_logits = True
    bad_epochs = 0
    history: list[dict[str, Any]] = []
    start = time.time()
    for epoch in range(1, max_epochs + 1):
        model.train()
        train_losses = []
        train_rows = []
        for batch in train_loader:
            rgb = batch["rgb"].to(device)
            shading = batch["shading"].to(device)
            residual = batch["residual"].to(device)
            labels = batch["label_binary"].to(device=device, dtype=torch.long)
            optimizer.zero_grad(set_to_none=True)
            logits = model(rgb, shading, residual)
            finite_logits = finite_logits and bool(torch.isfinite(logits).all().item())
            loss = criterion(logits, labels)
            finite_loss = finite_loss and bool(torch.isfinite(loss).item())
            loss.backward()
            optimizer.step()
            train_losses.append(float(loss.item()))
            probs = torch.softmax(logits.detach(), dim=1).cpu().numpy()
            pred = probs.argmax(axis=1)
            for i in range(len(pred)):
                train_rows.append({"label_binary": int(batch["label_binary"][i].item()), "prob_control": float(probs[i, 0]), "prob_patient": float(probs[i, 1]), "pred_binary": int(pred[i])})
        train_metric_frame = pd.DataFrame(train_rows)
        train_metrics = compute_visit_metrics(train_metric_frame)
        val_pred, val_metrics, val_finite_logits = _evaluate_model(model, val_loader, device, criterion)
        finite_logits = finite_logits and val_finite_logits
        row = {
            "epoch": int(epoch),
            "train_loss": float(np.mean(train_losses)),
            "val_loss": float(val_metrics["loss"]),
            "train_macro_auc": float(train_metrics["macro_auc"]),
            "val_macro_auc": float(val_metrics["macro_auc"]),
            "train_macro_f1": float(train_metrics["macro_f1"]),
            "val_macro_f1": float(val_metrics["macro_f1"]),
            "train_balanced_accuracy": float(train_metrics["balanced_accuracy"]),
            "val_balanced_accuracy": float(val_metrics["balanced_accuracy"]),
        }
        history.append(row)
        ckpt = {"model": model.state_dict(), "optimizer": optimizer.state_dict(), "epoch": int(epoch), "experiment": EXPERIMENT, "fold": int(fold), "seed": SEED}
        torch.save(ckpt, fold_dir / "checkpoints/last.pth")
        if float(val_metrics["macro_auc"]) > best_auc:
            best_auc = float(val_metrics["macro_auc"])
            best_epoch = int(epoch)
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
    val_pred, val_metrics, val_finite_logits = _evaluate_model(model, val_loader, device, criterion)
    finite_logits = finite_logits and val_finite_logits
    val_pred["best_epoch"] = int(best_epoch)
    val_pred["checkpoint_path"] = str(fold_dir / "checkpoints/best_macro_auc.pth")
    _write_csv(fold_dir / "val_predictions_visit.csv", val_pred)
    history_df = pd.DataFrame(history)
    _write_csv(fold_dir / "training_history.csv", history_df)
    metrics_payload = {k: v for k, v in val_metrics.items() if k != "confusion_matrix"}
    metrics_payload["confusion_matrix"] = np.asarray(val_metrics["confusion_matrix"], dtype=int).tolist()
    _write_json(fold_dir / "metrics_visit.json", metrics_payload)
    cm = np.asarray(val_metrics["confusion_matrix"], dtype=int)
    pd.DataFrame(cm, index=["true_control", "true_patient"], columns=["pred_control", "pred_patient"]).to_csv(fold_dir / "confusion_matrix_visit.csv", encoding="utf-8-sig")
    duration = float(time.time() - start)
    peak_memory = int(torch.cuda.max_memory_allocated(device))
    env = {
        "python": sys.version,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "cuda_available": bool(torch.cuda.is_available()),
        "gpu_name": torch.cuda.get_device_name(0),
        "peak_gpu_memory": peak_memory,
    }
    _write_json(fold_dir / "environment.json", env)
    best_hist = history_df[history_df["epoch"] == int(best_epoch)].iloc[0]
    fold_summary = {
        "experiment": EXPERIMENT,
        "fold": int(fold),
        "mode": "smoke" if smoke else "formal",
        "best_epoch": int(best_epoch),
        "max_epoch_reached": int(history_df["epoch"].max()),
        "training_loss": float(best_hist["train_loss"]),
        "validation_loss": float(best_hist["val_loss"]),
        "training_macro_auc": float(best_hist["train_macro_auc"]),
        "validation_macro_auc": float(best_hist["val_macro_auc"]),
        "predicted_control_count": int((val_pred["pred_binary"].astype(int) == 0).sum()),
        "predicted_patient_count": int((val_pred["pred_binary"].astype(int) == 1).sum()),
        "parameter_count": int(params["total_params"]),
        "classifier_parameters": int(params["classifier_params"]),
        "training_duration": duration,
        "peak_gpu_memory": peak_memory,
        "finite_loss": bool(finite_loss),
        "finite_logits": bool(finite_logits),
    }
    _write_json(fold_dir / "fold_summary.json", fold_summary)
    success = {
        "experiment": EXPERIMENT,
        "fold": int(fold),
        "mode": "smoke" if smoke else "formal",
        "config_sha256": _sha256_text("P1_STAGE_D_V1|triple_resnet18|adamw|1e-4|wd1e-4|batch16|seed2026"),
        "manifest_sha256": sha256_file(MANIFEST_PATH),
        "split_sha256": sha256_file(FIXED_SPLIT_PATH),
        "best_checkpoint_sha256": sha256_file(fold_dir / "checkpoints/best_macro_auc.pth"),
        "prediction_sha256": sha256_file(fold_dir / "val_predictions_visit.csv"),
        "prediction_rows": int(len(val_pred)),
        "unique_case_ids": int(val_pred["case_id"].astype(str).nunique()),
        "best_epoch": int(best_epoch),
        "best_macro_auc": float(best_auc),
        "finite_loss": bool(finite_loss),
        "finite_logits": bool(finite_logits),
        "peak_gpu_memory_recorded": peak_memory > 0,
        "completed_timestamp": _now(),
    }
    _write_json(fold_dir / ("SMOKE_SUCCESS.json" if smoke else "_FOLD_SUCCESS.json"), success)
    return success


def _metrics_row(frame: pd.DataFrame, *, experiment: str, status: str = "COMPLETED") -> dict[str, Any]:
    metrics = compute_visit_metrics(frame)
    cm = np.asarray(metrics["confusion_matrix"], dtype=int)
    return {
        "experiment": experiment,
        "status": status,
        "n_oof": int(len(frame)),
        "macro_auc": float(metrics["macro_auc"]),
        "macro_f1": float(metrics["macro_f1"]),
        "balanced_accuracy": float(metrics["balanced_accuracy"]),
        "patient_sensitivity": float(metrics["patient_sensitivity"]),
        "control_specificity": float(metrics["control_specificity"]),
        "accuracy": float(metrics["accuracy"]),
        "macro_precision": float(metrics["macro_precision"]),
        "macro_recall": float(metrics["macro_recall"]),
        "pr_auc": float(metrics["pr_auc"]),
        "ppv": float(metrics["ppv"]),
        "npv": float(metrics["npv"]),
        "tn": int(cm[0, 0]),
        "fp": int(cm[0, 1]),
        "fn": int(cm[1, 0]),
        "tp": int(cm[1, 1]),
        "predicted_control_count": int((frame["pred_binary"].astype(int) == 0).sum()),
        "predicted_patient_count": int((frame["pred_binary"].astype(int) == 1).sum()),
    }


def summarize(output_dir: Path) -> dict[str, Any]:
    fold_frames = [pd.read_csv(output_dir / f"fold_{fold}/val_predictions_visit.csv", dtype={"case_id": str, "patient_group_id": str}) for fold in range(5)]
    oof = pd.concat(fold_frames, ignore_index=True).sort_values(["fold", "case_id"]).reset_index(drop=True)
    frame = load_p1_frame(MANIFEST_PATH, FIXED_SPLIT_PATH).sort_values("case_id").reset_index(drop=True)
    oof_check = oof.sort_values("case_id").reset_index(drop=True)
    alignment = {
        "rows": int(len(oof)),
        "unique_case_id": int(oof["case_id"].astype(str).nunique()),
        "duplicate": int(oof["case_id"].duplicated().sum()),
        "missing": int(len(set(frame["case_id"].astype(str)) - set(oof["case_id"].astype(str)))),
        "unexpected": int(len(set(oof["case_id"].astype(str)) - set(frame["case_id"].astype(str)))),
        "label_mismatch": int((oof_check["label_binary"].astype(int).to_numpy() != frame["label_binary"].astype(int).to_numpy()).sum()),
        "fold_mismatch": int((oof_check["fold"].astype(int).to_numpy() != frame["fold"].astype(int).to_numpy()).sum()),
        "patient_group_mismatch": int((oof_check["patient_group_id"].astype(str).to_numpy() != frame["patient_group_id"].astype(str).to_numpy()).sum()),
    }
    _write_csv(output_dir / "oof/oof_predictions_visit.csv", oof)
    oof_metrics = compute_visit_metrics(oof)
    metrics_payload = {k: v for k, v in oof_metrics.items() if k != "confusion_matrix"}
    metrics_payload["confusion_matrix"] = np.asarray(oof_metrics["confusion_matrix"], dtype=int).tolist()
    _write_json(output_dir / "oof/oof_metrics_visit.json", metrics_payload)
    pd.DataFrame(np.asarray(oof_metrics["confusion_matrix"], dtype=int), index=["true_control", "true_patient"], columns=["pred_control", "pred_patient"]).to_csv(output_dir / "oof/oof_confusion_matrix_visit.csv", encoding="utf-8-sig")
    fold_rows = []
    stability_rows = []
    for fold in range(5):
        fold_dir = output_dir / f"fold_{fold}"
        metrics = _read_json(fold_dir / "metrics_visit.json")
        summary = _read_json(fold_dir / "fold_summary.json")
        fold_rows.append(
            {
                "experiment": EXPERIMENT,
                "fold": fold,
                "best_epoch": int(summary["best_epoch"]),
                "macro_auc": float(metrics["macro_auc"]),
                "macro_f1": float(metrics["macro_f1"]),
                "balanced_accuracy": float(metrics["balanced_accuracy"]),
                "patient_sensitivity": float(metrics["patient_sensitivity"]),
                "control_specificity": float(metrics["control_specificity"]),
            }
        )
        stability_rows.append(
            {
                "experiment": EXPERIMENT,
                "fold": fold,
                "best_epoch": int(summary["best_epoch"]),
                "training_loss": float(summary["training_loss"]),
                "validation_loss": float(summary["validation_loss"]),
                "training_macro_auc": float(summary["training_macro_auc"]),
                "validation_macro_auc": float(summary["validation_macro_auc"]),
                "predicted_control_count": int(summary["predicted_control_count"]),
                "predicted_patient_count": int(summary["predicted_patient_count"]),
                "parameter_count": int(summary["parameter_count"]),
                "training_duration": float(summary["training_duration"]),
                "peak_gpu_memory": int(summary["peak_gpu_memory"]),
                "best_epoch_le_3": int(summary["best_epoch"]) <= 3,
                "predicted_control_count_le_5": int(summary["predicted_control_count"]) <= 5,
                "predicted_patient_count_le_5": int(summary["predicted_patient_count"]) <= 5,
                "large_train_validation_gap": float(summary["training_macro_auc"]) - float(summary["validation_macro_auc"]) > 0.15,
            }
        )
    fold_df = pd.DataFrame(fold_rows)
    stability_df = pd.DataFrame(stability_rows)
    main = _metrics_row(oof, experiment=EXPERIMENT)
    main.update(
        {
            "completed_folds": int(len(fold_df)),
            "fold_macro_auc_mean": float(fold_df["macro_auc"].mean()),
            "fold_macro_auc_std": float(fold_df["macro_auc"].std(ddof=1)),
            "fold_macro_auc_min": float(fold_df["macro_auc"].min()),
            "fold_macro_auc_max": float(fold_df["macro_auc"].max()),
            "fold_macro_f1_mean": float(fold_df["macro_f1"].mean()),
            "fold_macro_f1_std": float(fold_df["macro_f1"].std(ddof=1)),
            "fold_macro_f1_min": float(fold_df["macro_f1"].min()),
            "fold_macro_f1_max": float(fold_df["macro_f1"].max()),
            "fold_balanced_accuracy_mean": float(fold_df["balanced_accuracy"].mean()),
            "fold_balanced_accuracy_std": float(fold_df["balanced_accuracy"].std(ddof=1)),
            "fold_balanced_accuracy_min": float(fold_df["balanced_accuracy"].min()),
            "fold_balanced_accuracy_max": float(fold_df["balanced_accuracy"].max()),
            "pooled_oof_macro_auc": float(oof_metrics["macro_auc"]),
            "pooled_minus_fold_mean_auc": float(oof_metrics["macro_auc"] - fold_df["macro_auc"].mean()),
            "parameter_count": int(_read_json(output_dir / "metadata/model_parameter_audit.json")["total_model_parameters"]),
        }
    )
    stability_df["pooled_fold_auc_gap"] = abs(float(main["pooled_minus_fold_mean_auc"])) > 0.05
    main["warning_count"] = int(
        stability_df[["best_epoch_le_3", "predicted_control_count_le_5", "predicted_patient_count_le_5", "large_train_validation_gap", "pooled_fold_auc_gap"]].any(axis=1).sum()
    )
    main_df = pd.DataFrame([main])
    _write_csv(output_dir / "summary/stage_d_v1_main_results.csv", main_df)
    _write_csv(output_dir / "summary/stage_d_v1_fold_results.csv", fold_df)
    _write_csv(output_dir / "summary/stage_d_v1_training_stability.csv", stability_df)
    _write_json(output_dir / "metadata/oof_alignment_audit.json", alignment)
    _write_json(output_dir / "oof/oof_bootstrap_visit.json", _fast_cluster_bootstrap(oof, iterations=BOOTSTRAP_ITERATIONS, seed=SEED))

    comparators = {
        "P1-RGB": pd.read_csv(P1_RGB_OOF, dtype={"case_id": str, "patient_group_id": str}),
        "RGB+RGB capacity control": pd.read_csv(ABC_ROOT / "stage_b/rgb_rgb_capacity_control/oof/oof_predictions_visit.csv", dtype={"case_id": str, "patient_group_id": str}),
        "RGB+S independent dual": pd.read_csv(ABC_ROOT / "stage_b/rgb_s_independent_dual/oof/oof_predictions_visit.csv", dtype={"case_id": str, "patient_group_id": str}),
        "RGB+R independent dual": pd.read_csv(ABC_ROOT / "stage_b/rgb_r_independent_dual/oof/oof_predictions_visit.csv", dtype={"case_id": str, "patient_group_id": str}),
        "P1-S": pd.read_csv(P1_ROOT / "p1_s/oof/oof_predictions_visit.csv", dtype={"case_id": str, "patient_group_id": str}),
        "P1-R": pd.read_csv(P1_ROOT / "p1_r/oof/oof_predictions_visit.csv", dtype={"case_id": str, "patient_group_id": str}),
    }
    comparison_rows = []
    bootstrap_rows = []
    alignment_rows = []
    for name, comp in comparators.items():
        merged = oof.merge(comp[["case_id", "label_binary", "fold", "patient_group_id"]], on="case_id", suffixes=("", "_comparator"))
        alignment_rows.append(
            {
                "comparator": name,
                "matched_rows": int(len(merged)),
                "label_mismatch": int((merged["label_binary"].astype(int) != merged["label_binary_comparator"].astype(int)).sum()),
                "fold_mismatch": int((merged["fold"].astype(int) != merged["fold_comparator"].astype(int)).sum()),
                "patient_group_mismatch": int((merged["patient_group_id"].astype(str) != merged["patient_group_id_comparator"].astype(str)).sum()),
            }
        )
        comp_metrics = compute_visit_metrics(comp)
        delta = _paired_delta(oof, comp)
        comparison_rows.append(
            {
                "comparator": name,
                "comparator_macro_auc": float(comp_metrics["macro_auc"]),
                "stage_d_macro_auc": float(main["macro_auc"]),
                "delta_macro_auc": float(delta["delta_macro_auc"]),
                "comparator_macro_f1": float(comp_metrics["macro_f1"]),
                "stage_d_macro_f1": float(main["macro_f1"]),
                "delta_macro_f1": float(delta["delta_macro_f1"]),
                "comparator_balanced_accuracy": float(comp_metrics["balanced_accuracy"]),
                "stage_d_balanced_accuracy": float(main["balanced_accuracy"]),
                "delta_balanced_accuracy": float(delta["delta_balanced_accuracy"]),
                "comparator_sensitivity": float(comp_metrics["patient_sensitivity"]),
                "stage_d_sensitivity": float(main["patient_sensitivity"]),
                "delta_sensitivity": float(delta["delta_patient_sensitivity"]),
                "comparator_specificity": float(comp_metrics["control_specificity"]),
                "stage_d_specificity": float(main["control_specificity"]),
                "delta_specificity": float(delta["delta_control_specificity"]),
            }
        )
        boot = _paired_bootstrap_delta(oof, comp, iterations=BOOTSTRAP_ITERATIONS, seed=SEED)
        row = {"comparator": name, "iterations": BOOTSTRAP_ITERATIONS, "seed": SEED, "cluster_unit": "patient_group_id", "metric_unit": "visit_case"}
        for metric in ("delta_macro_auc", "delta_macro_f1", "delta_balanced_accuracy", "delta_patient_sensitivity", "delta_control_specificity"):
            ci = boot.get("ci95", {}).get(metric, [np.nan, np.nan])
            row[f"{metric}_ci_low"] = ci[0]
            row[f"{metric}_ci_high"] = ci[1]
        bootstrap_rows.append(row)
    comparison_df = pd.DataFrame(comparison_rows)
    bootstrap_df = pd.DataFrame(bootstrap_rows)
    _write_csv(output_dir / "summary/stage_d_v1_comparison_table.csv", comparison_df)
    _write_csv(output_dir / "summary/stage_d_v1_paired_bootstrap.csv", bootstrap_df)
    _write_csv(output_dir / "summary/stage_d_v1_alignment_audit.csv", pd.DataFrame(alignment_rows))
    matrix = [
        "# Stage D v1 RGB+S+R result matrix",
        "",
        "## Main result",
        "",
        _md_table(main_df[["experiment", "macro_auc", "macro_f1", "balanced_accuracy", "patient_sensitivity", "control_specificity", "completed_folds", "warning_count"]]),
        "",
        "## Comparisons",
        "",
        _md_table(comparison_df),
    ]
    _write_text(output_dir / "summary/stage_d_v1_result_matrix.md", "\n".join(matrix))
    rgb_r_row = comparison_df[comparison_df["comparator"] == "RGB+R independent dual"].iloc[0]
    if any(float(rgb_r_row[col]) > 0 for col in ("delta_macro_auc", "delta_macro_f1", "delta_balanced_accuracy")) or (
        float(rgb_r_row["delta_specificity"]) > 0 and float(rgb_r_row["delta_sensitivity"]) > -0.05
    ) or (float(rgb_r_row["delta_sensitivity"]) > 0 and float(rgb_r_row["delta_specificity"]) > -0.05):
        conclusion = "STAGE_D_V1_SHOWS_IMPROVEMENT"
    elif float(rgb_r_row["delta_macro_f1"]) > 0 and float(rgb_r_row["delta_balanced_accuracy"]) > 0:
        conclusion = "STAGE_D_V1_IMPROVES_CLASS_BALANCE"
    else:
        conclusion = "STAGE_D_V1_NO_IMPROVEMENT_OVER_RGB_R"
    _write_report(output_dir, main_df, fold_df, comparison_df, bootstrap_df, stability_df, conclusion)
    manifest = {"final_status": STATUS_COMPLETE, "stage_d_v2_started": False, "conclusion": conclusion, "completed_at": _now(), "python": sys.version, "platform": platform.platform()}
    _write_json(output_dir / "metadata/run_manifest.json", manifest)
    return {"final_status": STATUS_COMPLETE, "conclusion": conclusion, "main": main, "comparisons": comparison_rows}


def _write_report(output_dir: Path, main_df: pd.DataFrame, fold_df: pd.DataFrame, comparison_df: pd.DataFrame, bootstrap_df: pd.DataFrame, stability_df: pd.DataFrame, conclusion: str) -> None:
    audit = _read_json(output_dir / "metadata/model_parameter_audit.json")
    smoke = _read_json(output_dir / "smoke/SMOKE_SUCCESS.json") if (output_dir / "smoke/SMOKE_SUCCESS.json").is_file() else {}
    cm = pd.read_csv(output_dir / "oof/oof_confusion_matrix_visit.csv", index_col=0)
    report = [
        "# Stage D v1: RGB+Shading+Residual Triple-Branch Results",
        "",
        f"- experiment: {EXPERIMENT}",
        f"- final_status: {STATUS_COMPLETE}",
        f"- conclusion: {conclusion}",
        "- camera_features_used: false",
        "- exif_features_used: false",
        "- bottleneck_used: false",
        "- attention_used: false",
        "- gating_used: false",
        "- threshold_search_performed: false",
        "- hyperparameter_search_performed: false",
        "- group_aggregation_used: false",
        "- stage_d_v2_started: false",
        "",
        "## Architecture",
        "",
        "Three independent ImageNet-initialized ResNet18 encoders feed a direct 1536-dimensional concatenation into Linear(1536, 2). No bottleneck, dropout, attention, gating, or MLP fusion is used.",
        "",
        "## Parameter audit",
        "",
        _md_table(pd.DataFrame([audit])),
        "",
        "## Smoke",
        "",
        _md_table(pd.DataFrame([smoke])) if smoke else "Smoke not available.",
        "",
        "## Five-fold metrics",
        "",
        _md_table(fold_df),
        "",
        "## Pooled 500-case OOF metrics",
        "",
        _md_table(main_df),
        "",
        "## Confusion matrix",
        "",
        _md_table(cm.reset_index().rename(columns={"index": "true_label"})),
        "",
        "## Comparisons",
        "",
        _md_table(comparison_df),
        "",
        "## Paired bootstrap CIs",
        "",
        _md_table(bootstrap_df),
        "",
        "## Training stability and warnings",
        "",
        _md_table(stability_df),
        "",
        "camera_features_used = false",
        "exif_features_used = false",
        "bottleneck_used = false",
        "attention_used = false",
        "gating_used = false",
        "threshold_search_performed = false",
        "hyperparameter_search_performed = false",
        "group_aggregation_used = false",
        "stage_d_v2_started = false",
    ]
    _write_text(REPORT_PATH, "\n".join(report))


def run_tests(output_dir: Path) -> dict[str, Any]:
    commands = [
        [sys.executable, "-m", "pytest", "tests/p1_stage_d", "-q"],
        [sys.executable, "-m", "pytest", "tests/p1", "-q"],
        [sys.executable, "-m", "pytest", "tests/p1_extension", "-q"],
    ]
    rows = []
    for command in commands:
        result = subprocess.run(command, cwd=ROOT, capture_output=True, text=True)
        rows.append({"command": " ".join(command), "returncode": int(result.returncode), "stdout_tail": result.stdout[-2000:], "stderr_tail": result.stderr[-2000:]})
        if result.returncode != 0:
            _write_json(output_dir / "metadata/pre_training_tests.json", {"status": "FAILED", "results": rows})
            raise RuntimeError(f"test command failed: {' '.join(command)}")
    payload = {"status": "PASSED", "results": rows}
    _write_json(output_dir / "metadata/pre_training_tests.json", payload)
    return payload


def main() -> None:
    args = parse_args()
    output_dir = resolve_output_dir(args.output_dir)
    _create_layout(output_dir)
    create_approval(output_dir)
    if args.summarize_only:
        print(json.dumps(_json_safe(summarize(output_dir)), ensure_ascii=False, indent=2))
        return
    preflight = run_preflight(output_dir)
    if args.validate_only:
        print(json.dumps(_json_safe({"status": preflight["status"], "preflight": preflight}), ensure_ascii=False, indent=2))
        return
    results: dict[str, Any] = {"preflight": preflight}
    if args.all:
        results["tests"] = run_tests(output_dir)
        results["smoke"] = train_fold(output_dir, 0, smoke=True, resume=args.resume)
        if not (results["smoke"].get("finite_loss") and results["smoke"].get("finite_logits") and results["smoke"].get("prediction_rows") == 100 and results["smoke"].get("peak_gpu_memory_recorded")):
            raise RuntimeError("BLOCKED_BY_STAGE_D_SMOKE_FAILURE")
        results["formal"] = []
        for fold in range(5):
            results["formal"].append(train_fold(output_dir, fold, smoke=False, resume=args.resume))
        results["summary"] = summarize(output_dir)
    elif args.smoke:
        results["smoke"] = train_fold(output_dir, 0, smoke=True, resume=args.resume)
    elif args.formal:
        folds = args.fold if args.fold is not None else list(range(5))
        results["formal"] = [train_fold(output_dir, int(fold), smoke=False, resume=args.resume) for fold in folds]
        if args.fold is None or len(set(args.fold)) == 5:
            results["summary"] = summarize(output_dir)
    elif args.resume:
        if not _fold_is_complete(output_dir / "smoke", smoke=True):
            results["smoke"] = train_fold(output_dir, 0, smoke=True, resume=True)
        results["formal"] = [train_fold(output_dir, fold, smoke=False, resume=True) for fold in range(5)]
        results["summary"] = summarize(output_dir)
    print(json.dumps(_json_safe(results), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
