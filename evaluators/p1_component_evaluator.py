"""Evaluator for unified P1 component experiments."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader

from utils.p1_cluster_bootstrap import (
    compute_visit_metrics,
    paired_patient_cluster_visit_bootstrap,
    patient_cluster_bootstrap,
)


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


def _forward(model: nn.Module, representation: Any) -> torch.Tensor:
    if isinstance(representation, dict):
        return model(representation)
    return model(representation)


class P1ComponentEvaluator:
    """Produce visit/case OOF outputs and patient-cluster bootstrap summaries."""

    def __init__(
        self,
        model: nn.Module,
        device: torch.device | None = None,
        output_dir: str | Path | None = None,
    ) -> None:
        self.model = model
        self.device = device or torch.device("cpu")
        self.output_dir = Path(output_dir) if output_dir is not None else None
        self.model.to(self.device)

    @torch.no_grad()
    def evaluate_loader(
        self,
        loader: DataLoader,
        *,
        baseline_case_frame: pd.DataFrame | None = None,
        iterations: int = 2000,
        seed: int = 2026,
    ) -> tuple[pd.DataFrame, dict[str, Any]]:
        self.model.eval()
        rows: list[dict[str, Any]] = []
        for batch in loader:
            representation = batch["representation"]
            if isinstance(representation, dict):
                representation = {key: value.to(self.device) for key, value in representation.items()}
            else:
                representation = representation.to(self.device)
            logits = _forward(self.model, representation)
            probabilities = torch.softmax(logits, dim=1).cpu().numpy()
            predictions = probabilities.argmax(axis=1)
            for index in range(len(predictions)):
                rows.append(
                    {
                        "case_id": str(batch["case_id"][index]),
                        "patient_group_id": str(batch["patient_group_id"][index]),
                        "fold": int(batch["fold"][index]),
                        "label_original": int(batch["label_original"][index]),
                        "label_3class": int(batch["label_3class"][index]),
                        "label_binary": int(batch["label_binary"][index]),
                        "prob_control": float(probabilities[index, 0]),
                        "prob_patient": float(probabilities[index, 1]),
                        "pred_binary": int(predictions[index]),
                    }
                )
        case_frame = pd.DataFrame(rows)
        metrics = compute_visit_metrics(case_frame)
        payload: dict[str, Any] = {
            "status": "available",
            "metric_unit": "visit_case",
            "cluster_unit": "patient_group_id",
            "visit_count": int(len(case_frame)),
            "unique_clusters": int(case_frame["patient_group_id"].astype(str).nunique()),
            "point_estimates": {key: float(value) if np.isscalar(value) else _json_safe(value) for key, value in metrics.items()},
        }
        bootstrap = patient_cluster_bootstrap(case_frame, iterations=iterations, seed=seed)
        payload["bootstrap"] = bootstrap
        if baseline_case_frame is not None:
            payload["paired_bootstrap"] = paired_patient_cluster_visit_bootstrap(
                case_frame,
                baseline_case_frame,
                iterations=iterations,
                seed=seed,
            )
        if self.output_dir is not None:
            oof_dir = self.output_dir / "oof"
            summary_dir = self.output_dir / "summary"
            oof_dir.mkdir(parents=True, exist_ok=True)
            summary_dir.mkdir(parents=True, exist_ok=True)
            case_frame.to_csv(oof_dir / "oof_predictions_visit.csv", index=False, encoding="utf-8-sig")
            case_frame.to_csv(oof_dir / "oof_predictions_case.csv", index=False, encoding="utf-8-sig")
            visit_metrics = dict(metrics)
            visit_metrics["confusion_matrix"] = _json_safe(visit_metrics["confusion_matrix"])
            (oof_dir / "oof_metrics_visit.json").write_text(
                json.dumps(_json_safe(visit_metrics), indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
            (summary_dir / "cluster_bootstrap_visit.json").write_text(
                json.dumps(_json_safe(bootstrap), indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
            (summary_dir / "p1_component_visit_metrics_cluster_bootstrap.json").write_text(
                json.dumps(_json_safe(bootstrap), indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
            if baseline_case_frame is not None:
                (summary_dir / "paired_patient_cluster_visit_bootstrap.json").write_text(
                    json.dumps(_json_safe(payload["paired_bootstrap"]), indent=2, ensure_ascii=False),
                    encoding="utf-8",
                )
                (summary_dir / "paired_cluster_bootstrap_vs_p1_rgb.json").write_text(
                    json.dumps(_json_safe(payload["paired_bootstrap"]), indent=2, ensure_ascii=False),
                    encoding="utf-8",
                )
        return case_frame, payload

    def evaluate_frame(
        self,
        case_frame: pd.DataFrame,
        *,
        iterations: int = 2000,
        seed: int = 2026,
    ) -> dict[str, Any]:
        metrics = compute_visit_metrics(case_frame)
        bootstrap = patient_cluster_bootstrap(case_frame, iterations=iterations, seed=seed)
        return {
            "status": "available",
            "metric_unit": "visit_case",
            "cluster_unit": "patient_group_id",
            "visit_count": int(len(case_frame)),
            "unique_clusters": int(case_frame["patient_group_id"].astype(str).nunique()),
            "point_estimates": _json_safe(metrics),
            "bootstrap": _json_safe(bootstrap),
        }
