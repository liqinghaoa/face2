"""Stage-two trainer for the unified P1 component experiments."""

from __future__ import annotations

import json
import math
import platform
import random
import shutil
import time
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader

from datasets.p1_component_dataset import P1ComponentDataset
from metrics.binary_classification_metrics import compute_binary_metrics
from models.p1_component_models import build_p1_component_model
from utils.experiment_utils import choose_device, set_random_seed
from utils.p1_component_preflight import FIXED_SPLIT_PATH, MANIFEST_PATH, load_sweep_config, sha256_file
from utils.p1_component_registry import get_component_spec
from utils.p1_cluster_bootstrap import compute_visit_metrics
from utils.p1_representation_normalization import build_training_collate, fit_component_normalization, summarize_normalization_state
from utils.p1_rgb_audit import load_p1_frame
from utils.p1_sweep_state import P1ComponentSweepState, save_state


class Phase2ExecutionNotApprovedError(RuntimeError):
    """Raised when a stage-two execution flag is used without approval."""


def _approval_path(output_dir: str | Path) -> Path:
    root = Path(output_dir)
    return root / "metadata" / "PHASE2_EXECUTION_APPROVED.json"


def _phase2_approved(output_dir: str | Path) -> bool:
    root = Path(output_dir)
    for candidate_root in (root, *root.parents):
        if (candidate_root / "metadata" / "PHASE2_EXECUTION_APPROVED.json").is_file():
            return True
    return False


def _resolve_config(experiment_config: Mapping[str, Any] | str | Path) -> dict[str, Any]:
    if isinstance(experiment_config, (str, Path)):
        return load_sweep_config(experiment_config)
    return dict(experiment_config)


def _load_frame_from_config(config: Mapping[str, Any]) -> pd.DataFrame:
    if "frame" in config and isinstance(config["frame"], pd.DataFrame):
        return config["frame"].copy()
    manifest = Path(config.get("manifest_path", MANIFEST_PATH))
    split = Path(config.get("fixed_split_path", FIXED_SPLIT_PATH))
    return load_p1_frame(manifest, split)


def _make_loader(
    dataset: P1ComponentDataset,
    *,
    batch_size: int,
    shuffle: bool,
    seed: int,
    collate_fn,
    device: torch.device,
) -> DataLoader:
    generator = torch.Generator()
    generator.manual_seed(int(seed))
    return DataLoader(
        dataset,
        batch_size=int(batch_size),
        shuffle=bool(shuffle),
        num_workers=0,
        pin_memory=device.type == "cuda",
        generator=generator,
        collate_fn=collate_fn,
    )


def _forward_inputs(model: nn.Module, batch: Mapping[str, Any]) -> torch.Tensor:
    representation = batch["representation"]
    if isinstance(representation, dict):
        return model(representation)
    return model(representation)


def _json_safe(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    return value


def _to_device_batch(batch: Mapping[str, Any], device: torch.device) -> dict[str, Any]:
    moved: dict[str, Any] = {}
    for key, value in batch.items():
        if torch.is_tensor(value):
            moved[key] = value.to(device)
        elif isinstance(value, dict):
            moved[key] = {
                inner_key: inner_value.to(device) if torch.is_tensor(inner_value) else inner_value
                for inner_key, inner_value in value.items()
            }
        else:
            moved[key] = value
    return moved


def _labels_from_batch(batch: Mapping[str, Any]) -> torch.Tensor:
    return batch["label_binary"].to(dtype=torch.long)


def _loss_normalizer(criterion: nn.Module, labels: torch.Tensor) -> float:
    if (
        isinstance(criterion, nn.CrossEntropyLoss)
        and criterion.reduction == "mean"
        and criterion.weight is not None
    ):
        weights = criterion.weight.to(labels.device)
        return float(weights.index_select(0, labels.long()).sum().item())
    return float(labels.size(0))


def _capture_rng_state(train_loader: DataLoader) -> dict[str, Any]:
    generator = getattr(train_loader, "generator", None)
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        "train_loader_generator": generator.get_state() if generator is not None else None,
    }


def _restore_rng_state(state: Mapping[str, Any] | None, train_loader: DataLoader) -> None:
    if not state:
        return
    if state.get("python") is not None:
        random.setstate(state["python"])
    if state.get("numpy") is not None:
        np.random.set_state(state["numpy"])
    if state.get("torch_cpu") is not None:
        torch.set_rng_state(state["torch_cpu"].cpu())
    cuda_states = state.get("torch_cuda")
    if cuda_states is not None and torch.cuda.is_available():
        for device_index, cuda_state in enumerate(cuda_states[: torch.cuda.device_count()]):
            torch.cuda.set_rng_state(cuda_state.cpu(), device=device_index)
    generator = getattr(train_loader, "generator", None)
    generator_state = state.get("train_loader_generator")
    if generator is not None and generator_state is not None:
        generator.set_state(generator_state.cpu())


def _load_torch_checkpoint(path: str | Path, device: torch.device) -> dict[str, Any]:
    checkpoint_path = Path(path)
    try:
        return torch.load(checkpoint_path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(checkpoint_path, map_location=device)


def _hash_file(path: Path) -> str:
    return sha256_file(path)


def _sha256_text_lines(lines: list[str]) -> str:
    import hashlib

    digest = hashlib.sha256()
    for line in lines:
        digest.update(line.encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def _metric_counts(frame: pd.DataFrame) -> dict[str, int]:
    counts = frame["pred_binary"].value_counts().to_dict()
    return {
        "predicted_control_count": int(counts.get(0, 0)),
        "predicted_patient_count": int(counts.get(1, 0)),
    }


def _write_confusion_matrix_csv(path: Path, confusion_matrix: Any) -> None:
    matrix = np.asarray(confusion_matrix, dtype=int)
    df = pd.DataFrame(matrix, index=["true_control", "true_patient"], columns=["pred_control", "pred_patient"])
    df.to_csv(path, encoding="utf-8-sig")


class P1ComponentTrainer:
    """Train and validate one P1 component fold with strict protocol checks."""

    def __init__(self, device: torch.device | None = None) -> None:
        self.device = device or choose_device()

    def _resolve_experiment_key(self, config: Mapping[str, Any]) -> str:
        experiment_key = (
            config.get("experiment_key")
            or config.get("component_key")
            or config.get("experiment", {}).get("key")
            or ""
        )
        if experiment_key:
            return str(experiment_key)
        representation = str(config.get("representation", "")).strip().lower()
        model_type = str(config.get("model_type", "")).strip().lower()
        for candidate in ("p1_a", "p1_n", "p1_l", "p1_s", "p1_r", "p1_rgb_a"):
            spec = get_component_spec(candidate)
            if spec.representation == representation or spec.model_type == model_type:
                return candidate
        raise ValueError("unable to resolve P1 component experiment key from config")

    def _validate_only_fold(
        self,
        config: Mapping[str, Any],
        spec_key: str,
        fold: int,
        output_path: Path,
        mode: str,
    ) -> dict[str, Any]:
        spec = get_component_spec(spec_key)
        frame = _load_frame_from_config(config)
        train_frame = frame[frame["fold"] != int(fold)].copy().reset_index(drop=True)
        val_frame = frame[frame["fold"] == int(fold)].copy().reset_index(drop=True)
        if train_frame.empty or val_frame.empty:
            raise ValueError("fold split produced an empty train/validation set")

        normalization = fit_component_normalization(train_frame, spec.key, fold=None, root=config.get("root"))
        train_dataset = P1ComponentDataset(train_frame, spec.key)
        val_dataset = P1ComponentDataset(val_frame, spec.key)
        collate_fn = build_training_collate(spec.key, normalization)
        train_loader = _make_loader(
            train_dataset,
            batch_size=int(config.get("batch_size", config.get("training", {}).get("batch_size", 16))),
            shuffle=True,
            seed=int(config.get("seed", 2026)) + int(fold),
            collate_fn=collate_fn,
            device=self.device,
        )
        val_loader = _make_loader(
            val_dataset,
            batch_size=int(config.get("batch_size", config.get("training", {}).get("batch_size", 16))),
            shuffle=False,
            seed=int(config.get("seed", 2026)) + 1000 + int(fold),
            collate_fn=collate_fn,
            device=self.device,
        )

        model = build_p1_component_model(spec.model_type, {"pretrained": True, "num_classes": 2}).to(self.device)
        first_batch = next(iter(train_loader))
        first_batch_device = _to_device_batch(first_batch, self.device)
        with torch.no_grad():
            logits = _forward_inputs(model, first_batch_device)
            if logits.shape[-1] != 2:
                raise ValueError("component model must output two logits")

        validation_frame, validation_metrics = self._evaluate_loader(model, val_loader)
        validation_dir = output_path / "validation"
        validation_dir.mkdir(parents=True, exist_ok=True)
        validation_report = {
            "status": "FRAMEWORK_VALIDATED",
            "execution_mode": mode,
            "experiment_key": spec.key,
            "display_name": spec.display_name,
            "fold": int(fold),
            "train_rows": int(len(train_frame)),
            "val_rows": int(len(val_frame)),
            "train_patient_groups": int(train_frame["patient_group_id"].nunique()),
            "val_patient_groups": int(val_frame["patient_group_id"].nunique()),
            "representation_shape": list(first_batch["representation"].shape)
            if not isinstance(first_batch["representation"], dict)
            else {
                "rgb": list(first_batch["representation"]["rgb"].shape),
                "albedo": list(first_batch["representation"]["albedo"].shape),
            },
            "normalization": normalization.to_dict(),
            "validation_metrics": _json_safe(validation_metrics),
        }
        (validation_dir / f"fold_{fold}_validate_only.json").write_text(
            json.dumps(validation_report, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        validation_frame.to_csv(validation_dir / f"fold_{fold}_validate_only_oof.csv", index=False, encoding="utf-8-sig")

        state = P1ComponentSweepState(
            experiment_key=spec.key,
            status="FRAMEWORK_VALIDATED",
            contract_version="p1_component_stage1",
            config_sha256=sha256_file(Path(config["config_path"])) if config.get("config_path") else None,
            split_sha256=sha256_file(FIXED_SPLIT_PATH),
            manifest_sha256=sha256_file(MANIFEST_PATH),
            completed_folds=(int(fold),),
            oof_rows=int(len(validation_frame)),
        )
        save_state(state, output_path / "metadata" / "experiment_status.json")
        return validation_report

    def _build_training_components(
        self,
        spec_key: str,
        train_frame: pd.DataFrame,
        config: Mapping[str, Any],
        fold: int,
    ) -> tuple[P1ComponentDataset, P1ComponentDataset, DataLoader, DataLoader, nn.Module, nn.Module, torch.optim.Optimizer, Any]:
        spec = get_component_spec(spec_key)
        normalization = fit_component_normalization(train_frame, spec.key, fold=None, root=config.get("root"))
        training_cfg = config.get("training", {})
        batch_size = int(config.get("batch_size", training_cfg.get("batch_size", 16)))
        train_dataset = P1ComponentDataset(train_frame, spec.key)
        val_dataset = P1ComponentDataset(config["_val_frame"], spec.key)
        horizontal_flip_enabled = spec.key in {"p1_a", "p1_s", "p1_r", "p1_rgb_a"}
        train_collate = build_training_collate(
            spec.key,
            normalization,
            training=True,
            horizontal_flip=horizontal_flip_enabled,
            flip_probability=0.5,
        )
        val_collate = build_training_collate(spec.key, normalization)
        train_loader = _make_loader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            seed=int(config.get("seed", 2026)) + int(fold),
            collate_fn=train_collate,
            device=self.device,
        )
        val_loader = _make_loader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            seed=int(config.get("seed", 2026)) + 1000 + int(fold),
            collate_fn=val_collate,
            device=self.device,
        )

        model = build_p1_component_model(spec.model_type, {"pretrained": True, "num_classes": 2}).to(self.device)
        train_counts = train_frame["label_binary"].value_counts().to_dict()
        n_train = int(len(train_frame))
        class_weights = torch.tensor(
            [
                n_train / max(2.0 * float(train_counts.get(0, 1)), 1.0),
                n_train / max(2.0 * float(train_counts.get(1, 1)), 1.0),
            ],
            dtype=torch.float32,
            device=self.device,
        )
        criterion = nn.CrossEntropyLoss(weight=class_weights)
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=float(training_cfg.get("optimizer", {}).get("learning_rate", 1.0e-4)),
            weight_decay=float(training_cfg.get("optimizer", {}).get("weight_decay", 1.0e-4)),
        )
        return train_dataset, val_dataset, train_loader, val_loader, model, criterion, optimizer, normalization

    @torch.no_grad()
    def _evaluate_loader(
        self,
        model: nn.Module,
        loader: DataLoader,
    ) -> tuple[pd.DataFrame, dict[str, Any]]:
        model.eval()
        rows: list[dict[str, Any]] = []
        labels: list[np.ndarray] = []
        probs: list[np.ndarray] = []
        for batch in loader:
            batch_on_device = _to_device_batch(batch, self.device)
            logits = _forward_inputs(model, batch_on_device)
            probability = torch.softmax(logits, dim=1).detach().cpu().numpy()
            label = batch["label_binary"].detach().cpu().numpy().astype(int)
            labels.append(label)
            probs.append(probability)
            pred = probability.argmax(axis=1)
            for index in range(len(label)):
                rows.append(
                    {
                        "case_id": str(batch["case_id"][index]),
                        "patient_group_id": str(batch["patient_group_id"][index]),
                        "fold": int(batch["fold"][index]),
                        "label_original": int(batch["label_original"][index]),
                        "label_3class": int(batch["label_3class"][index]),
                        "label_binary": int(label[index]),
                        "prob_control": float(probability[index, 0]),
                        "prob_patient": float(probability[index, 1]),
                        "pred_binary": int(pred[index]),
                    }
                )
        frame = pd.DataFrame(rows)
        metrics = compute_visit_metrics(frame)
        metrics["visit_case_macro_auc"] = float(metrics["macro_auc"])
        metrics["evaluation_unit"] = "visit_case"
        metrics["case_count"] = int(len(frame))
        metrics["unique_case_ids"] = int(frame["case_id"].astype(str).nunique()) if not frame.empty else 0
        metrics["predicted_control_count"] = int((frame["pred_binary"] == 0).sum())
        metrics["predicted_patient_count"] = int((frame["pred_binary"] == 1).sum())
        return frame, metrics

    def _save_fold_artifacts(
        self,
        *,
        fold_root: Path,
        fold: int,
        spec_key: str,
        config: Mapping[str, Any],
        normalization,
        training_history: pd.DataFrame,
        val_frame: pd.DataFrame,
        metrics: Mapping[str, Any],
        checkpoint: Mapping[str, Any],
        predictions: pd.DataFrame,
        best_epoch: int,
        best_checkpoint_path: Path,
        mode: str,
        training_case_ids: list[str],
    ) -> dict[str, Any]:
        fold_root.mkdir(parents=True, exist_ok=True)
        (fold_root / "checkpoints").mkdir(parents=True, exist_ok=True)
        (fold_root / "preprocessing").mkdir(parents=True, exist_ok=True)

        preprocessing_payload = summarize_normalization_state(normalization)
        preprocessing_payload.update(
            {
                "training_case_count": int(len(training_case_ids)),
                "training_case_ids_sha256": _sha256_text_lines(training_case_ids),
                "validation_used_for_fit": False,
                "parameter_sha256": "",
            }
        )
        preprocessing_path = fold_root / "preprocessing" / "fitted_parameters.json"
        preprocessing_path.write_text(
            json.dumps(preprocessing_payload, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        preprocessing_payload["parameter_sha256"] = sha256_file(preprocessing_path)
        preprocessing_path.write_text(
            json.dumps(preprocessing_payload, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

        training_history = training_history.copy()
        training_history.to_csv(fold_root / "training_history.csv", index=False, encoding="utf-8-sig")

        predictions = predictions.copy()
        predictions["best_epoch"] = int(best_epoch)
        predictions["checkpoint_path"] = str(best_checkpoint_path)
        predictions.to_csv(fold_root / "val_predictions_visit.csv", index=False, encoding="utf-8-sig")
        predictions.to_csv(fold_root / "val_predictions_case.csv", index=False, encoding="utf-8-sig")

        metrics_payload = {
            **{key: _json_safe(value) for key, value in metrics.items()},
            "visit_case_macro_auc": float(metrics["visit_case_macro_auc"]),
            "best_epoch": int(best_epoch),
            "checkpoint_path": str(best_checkpoint_path),
            "prediction_rows": int(len(predictions)),
            "unique_case_ids": int(predictions["case_id"].nunique()),
            "evaluation_unit": "visit_case",
        }
        (fold_root / "metrics_visit.json").write_text(
            json.dumps(metrics_payload, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        (fold_root / "metrics_case.json").write_text(
            json.dumps(metrics_payload, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        _write_confusion_matrix_csv(fold_root / "confusion_matrix_visit.csv", metrics["confusion_matrix"])
        _write_confusion_matrix_csv(fold_root / "confusion_matrix_case.csv", metrics["confusion_matrix"])

        fold_summary = {
            "experiment_key": spec_key,
            "mode": mode,
            "fold": int(fold),
            "best_epoch": int(best_epoch),
            "best_macro_auc": float(metrics["macro_auc"]),
            "visit_case_macro_auc": float(metrics["visit_case_macro_auc"]),
            "prediction_rows": int(len(predictions)),
            "unique_case_ids": int(predictions["case_id"].nunique()),
            "predicted_control_count": int((predictions["pred_binary"] == 0).sum()),
            "predicted_patient_count": int((predictions["pred_binary"] == 1).sum()),
        }
        (fold_root / "fold_summary.json").write_text(
            json.dumps(fold_summary, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

        fold_success = {
            "experiment": spec_key,
            "fold": int(fold),
            "config_sha256": sha256_file(Path(config["config_path"])) if config.get("config_path") else None,
            "manifest_sha256": sha256_file(MANIFEST_PATH),
            "split_sha256": sha256_file(FIXED_SPLIT_PATH),
            "framework_sha256": sha256_file(Path(config["framework_manifest_path"]))
            if config.get("framework_manifest_path")
            else None,
            "preprocessing_sha256": sha256_file(preprocessing_path),
            "best_checkpoint_sha256": sha256_file(best_checkpoint_path),
            "prediction_sha256": sha256_file(fold_root / "val_predictions_visit.csv"),
            "prediction_rows": int(len(predictions)),
            "unique_case_ids": int(predictions["case_id"].nunique()),
            "best_epoch": int(best_epoch),
            "best_macro_auc": float(metrics["macro_auc"]),
            "completed_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "evaluation_unit": "visit_case",
            "checkpoint_path": str(best_checkpoint_path),
            "mode": mode,
        }
        (fold_root / "_FOLD_SUCCESS.json").write_text(
            json.dumps(fold_success, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

        return {
            "fold_summary": fold_summary,
            "metrics": metrics_payload,
            "fold_success": fold_success,
            "preprocessing_path": str(preprocessing_path),
            "predictions_path": str(fold_root / "val_predictions_visit.csv"),
            "checkpoint_path": str(best_checkpoint_path),
        }

    def _train_fold(
        self,
        config: Mapping[str, Any],
        spec_key: str,
        fold: int,
        output_path: Path,
        mode: str,
        resume_from: str | Path | None = None,
    ) -> dict[str, Any]:
        if self.device.type != "cuda":
            raise RuntimeError("P1 phase-two training requires CUDA; CPU execution is not permitted")
        spec = get_component_spec(spec_key)
        if mode == "smoke":
            epochs = 2
            patience = int(config.get("training", {}).get("early_stopping_patience", 10))
        else:
            epochs = int(config.get("training", {}).get("max_epochs", 50))
            patience = int(config.get("training", {}).get("early_stopping_patience", 10))

        frame = _load_frame_from_config(config)
        train_frame = frame[frame["fold"] != int(fold)].copy().reset_index(drop=True)
        val_frame = frame[frame["fold"] == int(fold)].copy().reset_index(drop=True)
        if train_frame.empty or val_frame.empty:
            raise ValueError("fold split produced an empty train/validation set")

        config = dict(config)
        config["_val_frame"] = val_frame

        train_dataset, val_dataset, train_loader, val_loader, model, criterion, optimizer, normalization = self._build_training_components(
            spec.key,
            train_frame,
            config,
            fold,
        )

        best_epoch = 0
        best_macro_auc = float("-inf")
        patience_counter = 0
        history_records: list[dict[str, Any]] = []
        start_epoch = 1

        if resume_from is not None:
            checkpoint = _load_torch_checkpoint(resume_from, self.device)
            model.load_state_dict(checkpoint["model_state_dict"])
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            best_epoch = int(checkpoint.get("best_epoch", checkpoint.get("epoch", 0)))
            best_macro_auc = float(checkpoint.get("best_macro_auc", float("-inf")))
            patience_counter = int(checkpoint.get("patience_counter", 0))
            start_epoch = int(checkpoint.get("epoch", 0)) + 1
            history_path = output_path / f"fold_{fold}" / "training_history.csv"
            if history_path.is_file():
                history_records = pd.read_csv(history_path).to_dict("records")
            _restore_rng_state(checkpoint.get("rng_state"), train_loader)

        fold_root = output_path / f"fold_{fold}"
        fold_root.mkdir(parents=True, exist_ok=True)
        (fold_root / "checkpoints").mkdir(parents=True, exist_ok=True)
        (fold_root / "preprocessing").mkdir(parents=True, exist_ok=True)

        set_random_seed(int(config.get("seed", 2026)))
        if self.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(self.device)

        # Save a lightweight experiment status snapshot before training starts.
        save_state(
            P1ComponentSweepState(
                experiment_key=spec.key,
                status="RUNNING" if mode == "formal" else "SMOKE_PASSED",
                contract_version="p1_component_phase2",
                config_sha256=sha256_file(Path(config["config_path"])) if config.get("config_path") else None,
                split_sha256=sha256_file(FIXED_SPLIT_PATH),
                manifest_sha256=sha256_file(MANIFEST_PATH),
                completed_folds=(),
                note=f"mode={mode}",
            ),
            output_path / "metadata" / "experiment_status.json",
        )

        for epoch in range(start_epoch, epochs + 1):
            started = time.perf_counter()
            model.train()
            running_loss = 0.0
            normalizer_total = 0.0
            train_labels: list[np.ndarray] = []
            train_probs: list[np.ndarray] = []
            for batch in train_loader:
                batch_on_device = _to_device_batch(batch, self.device)
                labels = _labels_from_batch(batch_on_device)
                optimizer.zero_grad(set_to_none=True)
                logits = _forward_inputs(model, batch_on_device)
                loss = criterion(logits, labels)
                loss.backward()
                optimizer.step()
                normalizer = _loss_normalizer(criterion, labels)
                running_loss += float(loss.detach().item()) * normalizer
                normalizer_total += normalizer
                train_labels.append(labels.detach().cpu().numpy())
                train_probs.append(torch.softmax(logits.detach(), dim=1).cpu().numpy())
            train_metrics = compute_binary_metrics(np.concatenate(train_labels), np.concatenate(train_probs))
            train_metrics["visit_case_macro_auc"] = float(train_metrics["macro_auc"])

            val_loss_total = 0.0
            val_normalizer_total = 0.0
            model.eval()
            val_labels: list[np.ndarray] = []
            val_probs: list[np.ndarray] = []
            with torch.no_grad():
                for batch in val_loader:
                    batch_on_device = _to_device_batch(batch, self.device)
                    labels = _labels_from_batch(batch_on_device)
                    logits = _forward_inputs(model, batch_on_device)
                    loss = criterion(logits, labels)
                    normalizer = _loss_normalizer(criterion, labels)
                    val_loss_total += float(loss.detach().item()) * normalizer
                    val_normalizer_total += normalizer
                    val_labels.append(labels.detach().cpu().numpy())
                    val_probs.append(torch.softmax(logits.detach(), dim=1).cpu().numpy())
            val_metrics = compute_binary_metrics(np.concatenate(val_labels), np.concatenate(val_probs))
            val_metrics["visit_case_macro_auc"] = float(val_metrics["macro_auc"])

            macro_auc = float(val_metrics["macro_auc"])
            improved = math.isfinite(macro_auc) and macro_auc > best_macro_auc
            if improved:
                best_macro_auc = macro_auc
                best_epoch = epoch
                patience_counter = 0
            else:
                patience_counter += 1

            checkpoint_payload = {
                "epoch": int(epoch),
                "best_epoch": int(best_epoch),
                "best_macro_auc": float(best_macro_auc),
                "patience_counter": int(patience_counter),
                "fold": int(fold),
                "experiment_key": spec.key,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "rng_state": _capture_rng_state(train_loader),
                "device": str(self.device),
                "mode": mode,
                "normalization": normalization.to_dict(),
            }
            torch.save(checkpoint_payload, fold_root / "checkpoints" / "last.pth")
            if improved or not (fold_root / "checkpoints" / "best_macro_auc.pth").is_file():
                torch.save(checkpoint_payload, fold_root / "checkpoints" / "best_macro_auc.pth")

            history_records.append(
                {
                    "epoch": int(epoch),
                    "train_loss": running_loss / max(normalizer_total, 1.0),
                    "val_loss": val_loss_total / max(val_normalizer_total, 1.0),
                    "train_macro_auc": float(train_metrics["macro_auc"]),
                    "train_accuracy": float(train_metrics["accuracy"]),
                    "train_macro_f1": float(train_metrics["macro_f1"]),
                    "train_balanced_accuracy": float(train_metrics["balanced_accuracy"]),
                    "val_macro_auc": float(val_metrics["macro_auc"]),
                    "val_accuracy": float(val_metrics["accuracy"]),
                    "val_macro_f1": float(val_metrics["macro_f1"]),
                    "val_balanced_accuracy": float(val_metrics["balanced_accuracy"]),
                    "patience_counter": int(patience_counter),
                    "is_best": int(improved),
                    "elapsed_seconds": float(time.perf_counter() - started),
                }
            )
            pd.DataFrame(history_records).to_csv(fold_root / "training_history.csv", index=False, encoding="utf-8-sig")
            if patience_counter >= patience:
                break

        best_checkpoint = fold_root / "checkpoints" / "best_macro_auc.pth"
        if not best_checkpoint.is_file():
            raise RuntimeError("best_macro_auc.pth was not produced")
        checkpoint = _load_torch_checkpoint(best_checkpoint, self.device)
        model.load_state_dict(checkpoint["model_state_dict"])
        predictions, metrics = self._evaluate_loader(model, val_loader)
        metrics["best_epoch"] = int(checkpoint.get("best_epoch", best_epoch))
        metrics["checkpoint_path"] = str(best_checkpoint)
        metrics["prediction_rows"] = int(len(predictions))
        metrics["unique_case_ids"] = int(predictions["case_id"].nunique())

        result = self._save_fold_artifacts(
            fold_root=fold_root,
            fold=fold,
            spec_key=spec.key,
            config=config,
            normalization=normalization,
            training_history=pd.DataFrame(history_records),
            val_frame=val_frame,
            metrics=metrics,
            checkpoint=checkpoint,
            predictions=predictions,
            best_epoch=int(metrics["best_epoch"]),
            best_checkpoint_path=best_checkpoint,
            mode=mode,
            training_case_ids=[str(case_id) for case_id in train_frame["case_id"].tolist()],
        )

        save_state(
            P1ComponentSweepState(
                experiment_key=spec.key,
                status="SMOKE_PASSED" if mode == "smoke" else "FOLDS_COMPLETE",
                contract_version="p1_component_phase2",
                config_sha256=sha256_file(Path(config["config_path"])) if config.get("config_path") else None,
                split_sha256=sha256_file(FIXED_SPLIT_PATH),
                manifest_sha256=sha256_file(MANIFEST_PATH),
                completed_folds=(int(fold),),
                oof_rows=int(len(predictions)),
            ),
            output_path / "metadata" / "experiment_status.json",
        )
        result["status"] = "SMOKE_PASSED" if mode == "smoke" else "FOLD_COMPLETED"
        return result

    def fit_fold(
        self,
        experiment_config: Mapping[str, Any] | str | Path,
        fold: int,
        output_dir: str | Path,
        execution_mode: str = "validate_only",
        *,
        resume_from: str | Path | None = None,
    ) -> dict[str, Any]:
        config = _resolve_config(experiment_config)
        if isinstance(experiment_config, (str, Path)):
            config["config_path"] = str(Path(experiment_config))
        experiment_key = self._resolve_experiment_key(config)
        mode = str(execution_mode).strip().lower()
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        set_random_seed(int(config.get("seed", 2026)))
        if mode in {"smoke", "formal", "all"} and not _phase2_approved(output_path):
            raise Phase2ExecutionNotApprovedError("PHASE2_EXECUTION_NOT_APPROVED")
        if mode == "validate_only":
            return self._validate_only_fold(config, experiment_key, int(fold), output_path, mode)
        return self._train_fold(config, experiment_key, int(fold), output_path, mode, resume_from=resume_from)
