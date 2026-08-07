"""Deterministic train/validation evaluator for overfitting-control checkpoints."""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader

from metrics.classification_metrics import CLASS_NAMES, compute_classification_metrics, flatten_metrics
from trainers.global_optical_fusion_overfit_trainer import (
    validate_overfit_checkpoint_metadata,
)
from trainers.global_optical_fusion_trainer import load_torch_checkpoint
from utils.optical_feature_preprocessor import FeatureScaler


class GlobalOpticalFusionOverfitEvaluator:
    """Evaluate a selected checkpoint once on deterministic train and validation data."""

    def __init__(
        self,
        model: nn.Module,
        criterion: nn.Module,
        device: torch.device,
        output_dir: str | Path,
        *,
        expected_metadata: Mapping[str, Any],
        feature_scaler: FeatureScaler | None,
        forehead_available_by_id: Mapping[str, int] | None = None,
    ) -> None:
        self.model = model.to(device)
        self.criterion = criterion
        self.device = device
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.expected_metadata = dict(expected_metadata)
        self.feature_scaler = feature_scaler
        self.availability = {
            str(key): int(value) for key, value in (forehead_available_by_id or {}).items()
        }

    @staticmethod
    def _value(batch: Mapping[str, Any], key: str, index: int) -> Any:
        value = batch[key]
        return value[index].item() if torch.is_tensor(value) else value[index]

    def load_checkpoint(self, path: str | Path) -> dict[str, Any]:
        checkpoint = load_torch_checkpoint(path, self.device)
        validate_overfit_checkpoint_metadata(checkpoint, self.expected_metadata)
        actual_scaler_hash = self.feature_scaler.payload_sha256 if self.feature_scaler else None
        if actual_scaler_hash != self.expected_metadata["feature_scaler_sha256"]:
            raise ValueError("Evaluator scaler does not match checkpoint metadata")
        self.model.load_state_dict(checkpoint["model_state_dict"], strict=True)
        self.model.eval()
        return checkpoint

    def _loss_normalizer(self, labels: torch.Tensor) -> float:
        if (
            isinstance(self.criterion, nn.CrossEntropyLoss)
            and self.criterion.reduction == "mean"
            and self.criterion.weight is not None
        ):
            return float(
                self.criterion.weight.to(labels.device)
                .index_select(0, labels.long())
                .sum()
                .item()
            )
        return float(labels.shape[0])

    @torch.inference_mode()
    def _evaluate_loader(
        self, loader: DataLoader, *, split_name: str, checkpoint: Mapping[str, Any]
    ) -> tuple[pd.DataFrame, dict[str, Any]]:
        if split_name not in {"train", "val"}:
            raise ValueError("split_name must be train or val")
        self.model.eval()
        rows: list[dict[str, Any]] = []
        labels_all: list[np.ndarray] = []
        probabilities_all: list[np.ndarray] = []
        loss_total = 0.0
        normalizer_total = 0.0
        for batch in loader:
            images = batch["image"].to(self.device, non_blocking=True)
            aux = batch["aux_features"].to(self.device, non_blocking=True)
            labels = batch["label"].to(self.device, non_blocking=True).long()
            logits = self.model(images, aux)
            loss = self.criterion(logits, labels)
            normalizer = self._loss_normalizer(labels)
            loss_total += float(loss.item()) * normalizer
            normalizer_total += normalizer
            probabilities = torch.softmax(logits, dim=1).cpu().numpy()
            labels_np = labels.cpu().numpy().astype(int)
            predictions = probabilities.argmax(axis=1)
            labels_all.append(labels_np)
            probabilities_all.append(probabilities)
            for index, label in enumerate(labels_np):
                identifier = str(self._value(batch, "ID", index))
                availability = self.availability.get(identifier)
                if availability is None and aux.shape[1] > 0:
                    availability = int(aux[index, -1].item())
                predicted = int(predictions[index])
                rows.append(
                    {
                        "ID": identifier,
                        "patient_group_id": str(
                            self._value(batch, "patient_group_id", index)
                        ),
                        "fold": int(checkpoint["fold"]),
                        "split_role": split_name,
                        "strategy": str(checkpoint["strategy"]),
                        "variant": str(checkpoint["variant"]),
                        "true_label": int(label),
                        "true_class_name": CLASS_NAMES[int(label)],
                        "prob_normal": float(probabilities[index, 0]),
                        "prob_mild": float(probabilities[index, 1]),
                        "prob_severe": float(probabilities[index, 2]),
                        "pred_class": predicted,
                        "pred_class_name": CLASS_NAMES[predicted],
                        "correct": int(predicted == int(label)),
                        "NYHA": int(self._value(batch, "NYHA", index)),
                        "SEX": int(self._value(batch, "SEX", index)),
                        "sex_name": str(self._value(batch, "sex_name", index)),
                        "forehead_available": availability,
                    }
                )
        true = np.concatenate(labels_all)
        probabilities = np.concatenate(probabilities_all)
        metrics = compute_classification_metrics(true, probabilities)
        serializable: dict[str, Any] = flatten_metrics(metrics)
        serializable.update(
            {
                "weighted_cross_entropy": loss_total / max(normalizer_total, 1.0),
                "rows": int(len(true)),
                "strategy": str(checkpoint["strategy"]),
                "variant": str(checkpoint["variant"]),
                "fold": int(checkpoint["fold"]),
                "best_epoch": int(checkpoint["best_epoch"]),
                "confusion_matrix": np.asarray(metrics["confusion_matrix"]).tolist(),
            }
        )
        predictions = pd.DataFrame(rows)
        deterministic_predictions_path = (
            self.output_dir / f"deterministic_{split_name}_predictions.csv"
        )
        predictions.to_csv(
            deterministic_predictions_path,
            index=False,
            encoding="utf-8-sig",
        )
        with (self.output_dir / f"deterministic_{split_name}_metrics.json").open(
            "w", encoding="utf-8"
        ) as handle:
            json.dump(serializable, handle, ensure_ascii=False, indent=2)
        # Keep the conventional fold-level aliases while making the deterministic
        # protocol explicit in the canonical filenames above.
        predictions.to_csv(
            self.output_dir / f"{split_name}_predictions.csv",
            index=False,
            encoding="utf-8-sig",
        )
        with (self.output_dir / f"{split_name}_metrics.json").open(
            "w", encoding="utf-8"
        ) as handle:
            json.dump(serializable, handle, ensure_ascii=False, indent=2)
        if split_name == "val":
            with (self.output_dir / "metrics.json").open("w", encoding="utf-8") as handle:
                json.dump(serializable, handle, ensure_ascii=False, indent=2)
        return predictions, serializable

    def evaluate(
        self,
        deterministic_train_loader: DataLoader,
        val_loader: DataLoader,
        checkpoint_path: str | Path,
    ) -> dict[str, Any]:
        checkpoint = self.load_checkpoint(checkpoint_path)
        train_predictions, train_metrics = self._evaluate_loader(
            deterministic_train_loader, split_name="train", checkpoint=checkpoint
        )
        val_predictions, val_metrics = self._evaluate_loader(
            val_loader, split_name="val", checkpoint=checkpoint
        )
        history_path = self.output_dir / "training_log.csv"
        history = pd.read_csv(history_path) if history_path.is_file() else pd.DataFrame()

        def first_threshold(threshold: float) -> float:
            if history.empty or "train_macro_auc" not in history:
                return float("nan")
            reached = history.loc[history["train_macro_auc"] >= threshold, "epoch"]
            return float(reached.iloc[0]) if not reached.empty else float("nan")

        last_val_auc = (
            float(history.iloc[-1]["val_macro_auc"])
            if not history.empty and "val_macro_auc" in history
            else float("nan")
        )
        per_class_gap = {
            f"{name}_auc_generalization_gap": float(
                train_metrics[f"auc_{name}"] - val_metrics[f"auc_{name}"]
            )
            for name in CLASS_NAMES.values()
        }
        gap_metrics = {
            "schema_version": "global_optical_fusion_overfit_metrics_v1",
            "strategy": str(checkpoint["strategy"]),
            "variant": str(checkpoint["variant"]),
            "fold": int(checkpoint["fold"]),
            "best_epoch": int(checkpoint["best_epoch"]),
            "completed_epoch": int(history["epoch"].max()) if not history.empty else int(checkpoint["epoch"]),
            "train_rows": len(train_predictions),
            "val_rows": len(val_predictions),
            "deterministic_train_macro_auc": float(train_metrics["macro_auc"]),
            "deterministic_val_macro_auc": float(val_metrics["macro_auc"]),
            "generalization_gap_macro_auc": float(
                train_metrics["macro_auc"] - val_metrics["macro_auc"]
            ),
            "best_validation_macro_auc": float(checkpoint["best_macro_auc"]),
            "last_validation_macro_auc": last_val_auc,
            "validation_drop": float(checkpoint["best_macro_auc"] - last_val_auc),
            "first_epoch_train_auc_ge_0_90": first_threshold(0.90),
            "first_epoch_train_auc_ge_0_95": first_threshold(0.95),
            "first_epoch_train_auc_ge_0_99": first_threshold(0.99),
            "deterministic_train_macro_f1": float(train_metrics["macro_f1"]),
            "deterministic_val_macro_f1": float(val_metrics["macro_f1"]),
            "generalization_gap_macro_f1": float(
                train_metrics["macro_f1"] - val_metrics["macro_f1"]
            ),
            "deterministic_train_accuracy": float(train_metrics["accuracy"]),
            "deterministic_val_accuracy": float(val_metrics["accuracy"]),
            "generalization_gap_accuracy": float(
                train_metrics["accuracy"] - val_metrics["accuracy"]
            ),
            "deterministic_train_balanced_accuracy": float(
                train_metrics["balanced_accuracy"]
            ),
            "deterministic_val_balanced_accuracy": float(
                val_metrics["balanced_accuracy"]
            ),
            "generalization_gap_balanced_accuracy": float(
                train_metrics["balanced_accuracy"] - val_metrics["balanced_accuracy"]
            ),
            "train_weighted_cross_entropy": float(
                train_metrics["weighted_cross_entropy"]
            ),
            "val_weighted_cross_entropy": float(val_metrics["weighted_cross_entropy"]),
            "loss_generalization_gap": float(
                val_metrics["weighted_cross_entropy"]
                - train_metrics["weighted_cross_entropy"]
            ),
            **per_class_gap,
            **{
                f"deterministic_val_{name}_{metric}": float(
                    val_metrics[f"{metric}_{name}"]
                )
                for name in CLASS_NAMES.values()
                for metric in ("auc", "recall")
            },
            "trainable_parameter_count": int(
                checkpoint["trainability_trainable_parameter_count"]
            ),
            "total_parameter_count": int(checkpoint["parameter_count"]),
            "trainable_parameter_ratio": float(
                checkpoint["trainability_trainable_parameter_count"]
                / checkpoint["parameter_count"]
            ),
            "peak_gpu_memory_bytes": int(
                history["peak_gpu_memory_bytes"].max()
                if not history.empty and "peak_gpu_memory_bytes" in history
                else 0
            ),
            "training_time_seconds": float(
                history["elapsed_seconds"].sum()
                if not history.empty and "elapsed_seconds" in history
                else 0.0
            ),
        }
        # Compatibility aliases used by earlier tooling in this repository.
        gap_metrics.update(
            {
                "train_macro_auc": gap_metrics["deterministic_train_macro_auc"],
                "val_macro_auc": gap_metrics["deterministic_val_macro_auc"],
                "macro_auc_generalization_gap": gap_metrics[
                    "generalization_gap_macro_auc"
                ],
                "train_macro_f1": gap_metrics["deterministic_train_macro_f1"],
                "val_macro_f1": gap_metrics["deterministic_val_macro_f1"],
                "macro_f1_generalization_gap": gap_metrics[
                    "generalization_gap_macro_f1"
                ],
                "train_accuracy": gap_metrics["deterministic_train_accuracy"],
                "val_accuracy": gap_metrics["deterministic_val_accuracy"],
                "accuracy_generalization_gap": gap_metrics[
                    "generalization_gap_accuracy"
                ],
            }
        )
        with (self.output_dir / "overfit_metrics.json").open(
            "w", encoding="utf-8"
        ) as handle:
            json.dump(gap_metrics, handle, ensure_ascii=False, indent=2)
        return gap_metrics
