"""Single-fold evaluation and artifact export."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader

from metrics.classification_metrics import (
    CLASS_NAMES,
    compute_classification_metrics,
    flatten_metrics,
)


LOGGER = logging.getLogger(__name__)


def _load_checkpoint(path: Path, device: torch.device) -> dict[str, Any]:
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


class NYHA3ClassEvaluator:
    """Evaluate a checkpoint on its held-out validation fold."""

    def __init__(
        self,
        model: nn.Module,
        device: torch.device,
        output_dir: str | Path,
    ) -> None:
        self.model = model.to(device)
        self.device = device
        self.output_dir = Path(output_dir)
        self.prediction_dir = self.output_dir / "predictions"
        self.metric_dir = self.output_dir / "metrics"
        self.prediction_dir.mkdir(parents=True, exist_ok=True)
        self.metric_dir.mkdir(parents=True, exist_ok=True)

    def load_checkpoint(self, checkpoint_path: str | Path) -> dict[str, Any]:
        path = Path(checkpoint_path)
        if not path.is_file():
            raise FileNotFoundError(f"Checkpoint does not exist: {path}")
        checkpoint = _load_checkpoint(path, self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        LOGGER.info("Loaded checkpoint: %s", path)
        return checkpoint

    @staticmethod
    def _batch_value(batch: dict[str, Any], key: str, index: int) -> Any:
        value = batch[key]
        if torch.is_tensor(value):
            return value[index].item()
        return value[index]

    @torch.no_grad()
    def evaluate(
        self, loader: DataLoader, checkpoint_path: str | Path
    ) -> tuple[pd.DataFrame, dict[str, Any]]:
        self.load_checkpoint(checkpoint_path)
        self.model.eval()
        rows: list[dict[str, Any]] = []
        probabilities_all: list[np.ndarray] = []
        labels_all: list[int] = []

        for batch in loader:
            images = batch["image"].to(self.device, non_blocking=True)
            logits = self.model(images)
            probabilities = torch.softmax(logits, dim=1).cpu().numpy()
            predictions = probabilities.argmax(axis=1)
            labels = batch["label"].cpu().numpy()

            probabilities_all.append(probabilities)
            labels_all.extend(labels.astype(int).tolist())
            for index in range(len(labels)):
                label = int(labels[index])
                predicted = int(predictions[index])
                rows.append(
                    {
                        "ID": self._batch_value(batch, "ID", index),
                        "patient_group_id": self._batch_value(
                            batch, "patient_group_id", index
                        ),
                        "image_path": self._batch_value(
                            batch, "image_path", index
                        ),
                        "NYHA": int(self._batch_value(batch, "NYHA", index)),
                        "SEX": int(self._batch_value(batch, "SEX", index)),
                        "sex_name": self._batch_value(batch, "sex_name", index),
                        "label_3class": label,
                        "label_3class_name": self._batch_value(
                            batch, "label_3class_name", index
                        ),
                        "pred_class": predicted,
                        "pred_class_name": CLASS_NAMES[predicted],
                        "prob_normal": float(probabilities[index, 0]),
                        "prob_mild": float(probabilities[index, 1]),
                        "prob_severe": float(probabilities[index, 2]),
                        "correct": int(predicted == label),
                        "fold": int(self._batch_value(batch, "fold", index)),
                    }
                )

        prediction_frame = pd.DataFrame(rows)
        prediction_frame.to_csv(
            self.prediction_dir / "val_predictions.csv",
            index=False,
            encoding="utf-8-sig",
        )
        metrics = compute_classification_metrics(
            np.asarray(labels_all),
            np.concatenate(probabilities_all, axis=0),
        )
        metric_row = flatten_metrics(metrics)
        metric_row["fold"] = int(prediction_frame["fold"].iloc[0])
        pd.DataFrame([metric_row]).to_csv(
            self.metric_dir / "fold_metrics.csv",
            index=False,
            encoding="utf-8-sig",
        )
        self._save_confusion_matrix(metrics["confusion_matrix"])
        return prediction_frame, metrics

    def _save_confusion_matrix(self, matrix: np.ndarray) -> None:
        labels = [CLASS_NAMES[index] for index in range(3)]
        frame = pd.DataFrame(matrix, index=labels, columns=labels)
        frame.to_csv(
            self.metric_dir / "confusion_matrix.csv",
            index_label="true\\pred",
            encoding="utf-8-sig",
        )

        figure, axis = plt.subplots(figsize=(6, 5))
        image = axis.imshow(matrix, interpolation="nearest", cmap="Blues")
        figure.colorbar(image, ax=axis)
        axis.set(
            xticks=np.arange(3),
            yticks=np.arange(3),
            xticklabels=labels,
            yticklabels=labels,
            xlabel="Predicted class",
            ylabel="True class",
            title="Validation confusion matrix",
        )
        threshold = matrix.max() / 2.0 if matrix.size else 0
        for row in range(3):
            for column in range(3):
                axis.text(
                    column,
                    row,
                    str(int(matrix[row, column])),
                    ha="center",
                    va="center",
                    color="white" if matrix[row, column] > threshold else "black",
                )
        figure.tight_layout()
        figure.savefig(self.metric_dir / "confusion_matrix.png", dpi=180)
        plt.close(figure)
