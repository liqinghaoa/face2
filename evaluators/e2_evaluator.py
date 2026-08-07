"""Best-checkpoint held-out E2 evaluation and required prediction export."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader

from metrics.classification_metrics import flatten_metrics
from metrics.e2_metrics import compute_e2_metrics, softmax_probabilities


class E2Evaluator:
    def __init__(
        self, model: nn.Module, device: torch.device, fold_dir: Path, ece_bins: int = 15
    ) -> None:
        self.model, self.device, self.fold_dir = model.to(device), device, fold_dir
        self.ece_bins = int(ece_bins)

    @torch.no_grad()
    def evaluate(self, loader: DataLoader, checkpoint_path: Path) -> tuple[pd.DataFrame, dict[str, Any]]:
        try:
            checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        except TypeError:
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"]); self.model.eval()
        rows: list[dict[str, Any]] = []; labels_all: list[np.ndarray] = []; probs_all: list[np.ndarray] = []
        for batch in loader:
            logits = self.model(batch["global_image"].to(self.device), batch["roi_images"].to(self.device)); probs = softmax_probabilities(logits).cpu().numpy(); labels = batch["label"].numpy().astype(int); pred = probs.argmax(1)
            labels_all.append(labels); probs_all.append(probs)
            for index, label in enumerate(labels):
                rows.append({"sample_id": str(batch["sample_id"][index]), "patient_group_id": str(batch["patient_group_id"][index]), "fold": int(batch["fold"][index]), "true_label": int(label), "pred_label": int(pred[index]), "prob_normal": float(probs[index, 0]), "prob_mild": float(probs[index, 1]), "prob_severe": float(probs[index, 2]), "global_path": str(batch["global_path"][index]), **{f"{name}_path": str(batch[f"{name}_path"][index]) for name in self.model.roi_order}})
        frame = pd.DataFrame(rows); metrics = compute_e2_metrics(
            np.concatenate(labels_all), np.concatenate(probs_all), self.ece_bins
        )
        prediction_dir, metric_dir = self.fold_dir / "predictions", self.fold_dir / "metrics"; prediction_dir.mkdir(parents=True, exist_ok=True); metric_dir.mkdir(parents=True, exist_ok=True)
        frame.to_csv(prediction_dir / "val_predictions.csv", index=False, encoding="utf-8-sig")
        row = {**flatten_metrics(metrics), "fold": int(frame["fold"].iloc[0])}; pd.DataFrame([row]).to_csv(metric_dir / "fold_metrics.csv", index=False, encoding="utf-8-sig")
        pd.DataFrame(metrics["confusion_matrix"], index=["normal", "mild", "severe"], columns=["normal", "mild", "severe"]).to_csv(metric_dir / "confusion_matrix.csv", index_label="true\\pred", encoding="utf-8-sig")
        (metric_dir / "metrics.json").write_text(json.dumps(row, indent=2, allow_nan=True), encoding="utf-8")
        return frame, metrics
