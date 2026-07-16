"""Held-out validation evaluator for global optical fusion checkpoints."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader

from metrics.classification_metrics import CLASS_NAMES, compute_classification_metrics, flatten_metrics
from trainers.global_optical_fusion_trainer import (
    load_torch_checkpoint,
    validate_checkpoint_metadata,
)
from utils.optical_feature_preprocessor import FeatureScaler


class GlobalOpticalFusionEvaluator:
    """Load the selected checkpoint and export one fold's final predictions."""

    def __init__(
        self,
        model: nn.Module,
        device: torch.device,
        output_dir: str | Path,
        *,
        expected_metadata: Mapping[str, Any],
        feature_scaler: FeatureScaler | None,
        forehead_available_by_id: Mapping[str, int] | None = None,
    ) -> None:
        self.model = model.to(device)
        self.device = device
        self.output_dir = Path(output_dir)
        self.expected_metadata = dict(expected_metadata)
        self.feature_scaler = feature_scaler
        self.availability = {str(k): int(v) for k, v in (forehead_available_by_id or {}).items()}

    def load_checkpoint(self, path: str | Path) -> dict[str, Any]:
        checkpoint = load_torch_checkpoint(path, self.device)
        validate_checkpoint_metadata(checkpoint, self.expected_metadata)
        expected_scaler_hash = self.expected_metadata["feature_scaler_sha256"]
        actual_scaler_hash = self.feature_scaler.payload_sha256 if self.feature_scaler else None
        if actual_scaler_hash != expected_scaler_hash:
            raise ValueError("Evaluator scaler does not match the checkpoint scaler hash")
        self.model.load_state_dict(checkpoint["model_state_dict"])
        return checkpoint

    @staticmethod
    def _value(batch: Mapping[str, Any], key: str, index: int) -> Any:
        value = batch[key]
        return value[index].item() if torch.is_tensor(value) else value[index]

    @torch.no_grad()
    def evaluate(
        self, loader: DataLoader, checkpoint_path: str | Path
    ) -> tuple[pd.DataFrame, dict[str, Any]]:
        checkpoint = self.load_checkpoint(checkpoint_path)
        self.model.eval()
        rows: list[dict[str, Any]] = []
        labels_all: list[int] = []
        probabilities_all: list[np.ndarray] = []
        for batch in loader:
            images = batch["image"].to(self.device, non_blocking=True)
            aux = batch["aux_features"].to(self.device, non_blocking=True)
            probabilities = torch.softmax(self.model(images, aux), dim=1).cpu().numpy()
            labels = batch["label"].cpu().numpy().astype(int)
            predictions = probabilities.argmax(axis=1)
            labels_all.extend(labels.tolist())
            probabilities_all.append(probabilities)
            for index, label in enumerate(labels):
                identifier = str(self._value(batch, "ID", index))
                predicted = int(predictions[index])
                availability = self.availability.get(identifier)
                if availability is None and aux.shape[1] > 0:
                    availability = int(aux[index, -1].item())
                rows.append({
                    "ID": identifier,
                    "patient_group_id": str(self._value(batch, "patient_group_id", index)),
                    "fold": int(self._value(batch, "fold", index)),
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
                })
        predictions = pd.DataFrame(rows)
        predictions.to_csv(self.output_dir / "val_predictions.csv", index=False, encoding="utf-8-sig")
        metrics = compute_classification_metrics(
            np.asarray(labels_all), np.concatenate(probabilities_all, axis=0)
        )
        serializable = flatten_metrics(metrics)
        serializable.update({
            "fold": int(checkpoint["fold"]), "variant": checkpoint["variant"],
            "best_epoch": int(checkpoint["best_epoch"]),
            "confusion_matrix": np.asarray(metrics["confusion_matrix"]).tolist(),
        })
        with (self.output_dir / "metrics.json").open("w", encoding="utf-8") as handle:
            json.dump(serializable, handle, ensure_ascii=False, indent=2)
        self._save_confusion(np.asarray(metrics["confusion_matrix"]))
        return predictions, metrics

    def _save_confusion(self, matrix: np.ndarray) -> None:
        names = [CLASS_NAMES[index] for index in range(3)]
        pd.DataFrame(matrix, index=names, columns=names).to_csv(
            self.output_dir / "confusion_matrix.csv", index_label="true\\pred", encoding="utf-8-sig"
        )
        figure, axis = plt.subplots(figsize=(5.5, 5))
        image = axis.imshow(matrix, cmap="Blues")
        figure.colorbar(image, ax=axis)
        axis.set(xticks=range(3), yticks=range(3), xticklabels=names, yticklabels=names,
                 xlabel="Predicted", ylabel="True", title="Validation confusion matrix")
        threshold = matrix.max() / 2 if matrix.size else 0
        for row in range(3):
            for column in range(3):
                axis.text(column, row, str(int(matrix[row, column])), ha="center", va="center",
                          color="white" if matrix[row, column] > threshold else "black")
        figure.tight_layout()
        figure.savefig(self.output_dir / "confusion_matrix.png", dpi=180)
        plt.close(figure)
