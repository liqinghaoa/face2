from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader

from p2_counterfactual.assets import PRESET_NAMES
from p2_counterfactual.dataset import P2ASingleRGBDataset, p2_a_collate
from p2_counterfactual.metrics import compute_p2_a_metrics, json_safe, scalar_metrics


def _string_array(values: Any, *, max_chars: int = 128) -> np.ndarray:
    return np.asarray([str(value) for value in values], dtype=f"<U{int(max_chars)}")


def write_confusion_matrix_csv(path: Path, matrix: Any) -> None:
    pd.DataFrame(np.asarray(matrix, dtype=int), index=["true_control", "true_patient"], columns=["pred_control", "pred_patient"]).to_csv(
        path, encoding="utf-8-sig"
    )


class P2AEvaluator:
    def __init__(self, model: nn.Module, config: dict[str, Any], *, fold: int, device: torch.device, output_dir: Path) -> None:
        self.model = model
        self.config = config
        self.fold = int(fold)
        self.device = device
        self.output_dir = Path(output_dir)
        self.model.to(self.device)

    def _loader(self, dataset: P2ASingleRGBDataset, num_workers: int = 0) -> DataLoader:
        return DataLoader(
            dataset,
            batch_size=int(self.config["training"]["batch_size"]),
            shuffle=False,
            num_workers=int(num_workers),
            pin_memory=self.device.type == "cuda",
            collate_fn=p2_a_collate,
        )

    @torch.no_grad()
    def _predict_loader(self, loader: DataLoader, *, preset_name: str | None = None) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
        self.model.eval()
        rows: list[dict[str, Any]] = []
        features: list[np.ndarray] = []
        labels: list[int] = []
        for batch in loader:
            images = batch["image"].to(self.device, non_blocking=True)
            logits, feats = self.model(images)
            probabilities = torch.softmax(logits, dim=1).detach().cpu().numpy()
            feats_np = feats.detach().cpu().numpy()
            predictions = probabilities.argmax(axis=1)
            features.append(feats_np)
            labels.extend([int(x) for x in batch["label"]])
            for index in range(len(predictions)):
                row = {
                    "case_id": str(batch["case_id"][index]),
                    "patient_group_id": str(batch["patient_group_id"][index]),
                    "fold": int(batch["fold"][index]),
                    "label": int(batch["label"][index]),
                    "prob_control": float(probabilities[index, 0]),
                    "prob_patient": float(probabilities[index, 1]),
                    "predicted_label": int(predictions[index]),
                }
                if "image_path" in batch:
                    row["image_path"] = str(batch["image_path"][index])
                if "original_path" in batch:
                    row["original_path"] = str(batch["original_path"][index])
                if "preset_index" in batch:
                    row["preset_index"] = int(batch["preset_index"][index])
                if preset_name is not None:
                    row["preset_name"] = str(preset_name)
                rows.append(row)
        return pd.DataFrame(rows), np.concatenate(features, axis=0), np.asarray(labels, dtype=np.int64)

    def evaluate_original(
        self,
        *,
        max_cases: int | None = None,
        num_workers: int = 0,
        save_features: bool | None = None,
        sample_seed: int | None = None,
    ) -> dict[str, Any]:
        data_cfg = self.config["data"]
        eval_cfg = self.config["evaluation"]
        dataset = P2ASingleRGBDataset(
            self.config["manifest_path"],
            fold=self.fold,
            split="val",
            input_mode="original",
            training=False,
            evaluation_source="original",
            image_size=int(data_cfg["image_size"]),
            horizontal_flip_probability=float(data_cfg["horizontal_flip_probability"]),
            normalization_mean=data_cfg["normalization_mean"],
            normalization_std=data_cfg["normalization_std"],
            original_rgb_override_dir=data_cfg.get("original_rgb_override_dir"),
            max_cases=max_cases,
            seed=int(sample_seed) if sample_seed is not None else int(self.config["training"]["seed"]) + 5000 + self.fold,
            project_root=self.config["project_root"],
            path_resolution=self.config.get("path_resolution", {}),
        )
        frame, features, labels = self._predict_loader(self._loader(dataset, num_workers=num_workers))
        probs = frame[["prob_control", "prob_patient"]].to_numpy(dtype=np.float64)
        metrics = compute_p2_a_metrics(labels, probs)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        frame.to_csv(self.output_dir / "val_predictions_original.csv", index=False, encoding="utf-8-sig")
        (self.output_dir / "metrics_original.json").write_text(json.dumps(json_safe(metrics), ensure_ascii=False, indent=2), encoding="utf-8")
        write_confusion_matrix_csv(self.output_dir / "confusion_matrix_original.csv", metrics["confusion_matrix"])
        if bool(eval_cfg.get("save_validation_features", True) if save_features is None else save_features):
            np.savez_compressed(
                self.output_dir / "val_features_original.npz",
                case_ids=_string_array(frame["case_id"], max_chars=128),
                features=features.astype(np.float32),
            )
        return {"rows": int(len(frame)), "metrics": scalar_metrics(metrics), "feature_shape": list(features.shape)}

    def evaluate_relighted(
        self,
        *,
        max_cases: int | None = None,
        num_workers: int = 0,
        save_features: bool | None = None,
        sample_seed: int | None = None,
    ) -> dict[str, Any]:
        data_cfg = self.config["data"]
        eval_cfg = self.config["evaluation"]
        all_frames: list[pd.DataFrame] = []
        all_features: list[np.ndarray] = []
        case_ids: np.ndarray | None = None
        for preset in PRESET_NAMES:
            dataset = P2ASingleRGBDataset(
                self.config["manifest_path"],
                fold=self.fold,
                split="val",
                input_mode="original",
                training=False,
                evaluation_source="relighted",
                evaluation_preset_name=preset,
                image_size=int(data_cfg["image_size"]),
                horizontal_flip_probability=float(data_cfg["horizontal_flip_probability"]),
                normalization_mean=data_cfg["normalization_mean"],
                normalization_std=data_cfg["normalization_std"],
                original_rgb_override_dir=data_cfg.get("original_rgb_override_dir"),
                max_cases=max_cases,
                seed=int(sample_seed) if sample_seed is not None else int(self.config["training"]["seed"]) + 6000 + self.fold,
                project_root=self.config["project_root"],
                path_resolution=self.config.get("path_resolution", {}),
            )
            frame, features, _ = self._predict_loader(self._loader(dataset, num_workers=num_workers), preset_name=preset)
            if case_ids is None:
                case_ids = _string_array(frame["case_id"], max_chars=128)
            all_frames.append(frame)
            all_features.append(features.astype(np.float32))
        out = pd.concat(all_frames, ignore_index=True)
        out.to_csv(self.output_dir / "val_predictions_relighted.csv", index=False, encoding="utf-8-sig")
        stacked = np.stack(all_features, axis=1)
        if bool(eval_cfg.get("save_validation_features", True) if save_features is None else save_features):
            np.savez_compressed(
                self.output_dir / "val_features_relighted.npz",
                case_ids=case_ids,
                preset_names=_string_array(PRESET_NAMES, max_chars=64),
                features=stacked,
            )
        return {"rows": int(len(out)), "expected_rows": int(stacked.shape[0] * len(PRESET_NAMES)), "feature_shape": list(stacked.shape)}
