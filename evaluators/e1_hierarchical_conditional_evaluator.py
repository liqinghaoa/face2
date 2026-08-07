"""Held-out fold evaluation and prediction export for E1."""

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
from metrics.e1_hierarchical_metrics import compute_e1_metrics, restore_three_class_probabilities


class E1HierarchicalConditionalEvaluator:
    def __init__(self, model: nn.Module, device: torch.device, output_dir: str | Path, ece_bins: int = 15) -> None:
        self.model, self.device, self.output_dir = model.to(device), device, Path(output_dir)
        self.prediction_dir, self.metric_dir = self.output_dir / "predictions", self.output_dir / "metrics"
        self.ece_bins = int(ece_bins)
        self.prediction_dir.mkdir(parents=True, exist_ok=True)
        self.metric_dir.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _load(path: Path, device: torch.device) -> dict[str, Any]:
        try:
            return torch.load(path, map_location=device, weights_only=False)
        except TypeError:
            return torch.load(path, map_location=device)

    @torch.no_grad()
    def evaluate(self, loader: DataLoader, checkpoint_path: str | Path) -> tuple[pd.DataFrame, dict[str, Any]]:
        checkpoint = self._load(Path(checkpoint_path), self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.eval()
        rows, labels_all, probs_all, abnormal_all, severe_all = [], [], [], [], []
        for batch in loader:
            images = batch["image"].to(self.device, non_blocking=True)
            outputs = self.model(images)
            restored = restore_three_class_probabilities(outputs["logit_abnormal"], outputs["logit_severe_cond"])
            labels, probs = batch["label"].numpy().astype(int), restored["probabilities"].cpu().numpy()
            abnormal, severe = restored["p_abnormal"].cpu().numpy().reshape(-1), restored["p_severe_cond"].cpu().numpy().reshape(-1)
            pred = probs.argmax(axis=1)
            labels_all.append(labels); probs_all.append(probs); abnormal_all.append(abnormal); severe_all.append(severe)
            for i, label in enumerate(labels):
                error = abs(int(pred[i]) - int(label))
                rows.append({
                    "fold": int(batch["fold"][i]), "patient_id": str(batch["patient_group_id"][i]),
                    "image_id": str(batch["ID"][i]), "true_label": int(label),
                    "logit_abnormal": float(outputs["logit_abnormal"][i, 0].cpu()),
                    "logit_severe_cond": float(outputs["logit_severe_cond"][i, 0].cpu()),
                    "p_abnormal": float(abnormal[i]), "p_severe_cond": float(severe[i]),
                    "prob_normal": float(probs[i, 0]), "prob_mild": float(probs[i, 1]), "prob_severe": float(probs[i, 2]),
                    "pred_class": int(pred[i]), "correct": int(pred[i] == label),
                    "absolute_grade_error": error, "is_two_step_error": int(error == 2),
                })
        frame = pd.DataFrame(rows)
        metrics = compute_e1_metrics(np.concatenate(labels_all), np.concatenate(probs_all), np.concatenate(abnormal_all), np.concatenate(severe_all), self.ece_bins)
        frame.to_csv(self.prediction_dir / "val_predictions.csv", index=False, encoding="utf-8-sig")
        metric_row = {**flatten_metrics(metrics), "fold": int(frame["fold"].iloc[0])}
        pd.DataFrame([metric_row]).to_csv(self.metric_dir / "fold_metrics.csv", index=False, encoding="utf-8-sig")
        pd.DataFrame(metrics["confusion_matrix"], index=["normal", "mild", "severe"], columns=["normal", "mild", "severe"]).to_csv(self.metric_dir / "confusion_matrix.csv", index_label="true\\pred", encoding="utf-8-sig")
        (self.metric_dir / "metrics.json").write_text(json.dumps(metric_row, indent=2, allow_nan=True), encoding="utf-8")
        return frame, metrics
