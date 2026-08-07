from __future__ import annotations

import json
import math
import random
import time
from functools import partial
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader

from losses.classification_losses import compute_class_weights
from p2_counterfactual.assets import PRESET_NAMES
from p2_counterfactual.config import config_fingerprint
from p2_counterfactual.dataset import P2ASingleRGBDataset, load_p2_manifest, p2_a_collate
from p2_counterfactual.evaluator import P2AEvaluator
from p2_counterfactual.losses import p2_a_full6_loss, p2_a_loss, p2_a_pairwise_loss, symmetric_js_divergence
from p2_counterfactual.metrics import compute_p2_a_metrics, json_safe, scalar_metrics
from p2_counterfactual.model import P2ASingleRGBResNet18
from utils.experiment_utils import save_yaml, set_random_seed


def _load_torch(path: Path, device: torch.device) -> dict[str, Any]:
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


def _to_device(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    return {key: value.to(device, non_blocking=True) if torch.is_tensor(value) else value for key, value in batch.items()}


def _seed_worker(worker_id: int, *, base_seed: int) -> None:
    worker_seed = int(base_seed) + int(worker_id) + 10000
    np.random.seed(worker_seed)
    random.seed(worker_seed)


class P2AFoldTrainer:
    def __init__(
        self,
        config: dict[str, Any],
        *,
        fold: int,
        device: torch.device,
        smoke: bool = False,
        num_workers: int | None = None,
        max_batches: int | None = None,
    ) -> None:
        self.config = config
        self.fold = int(fold)
        self.device = device
        self.smoke = bool(smoke)
        self.max_batches = max_batches
        self.num_workers = int(num_workers if num_workers is not None else config["data"].get("num_workers", 0))
        root = Path(config["smoke_output_root"] if self.smoke else config["output_root"])
        self.output_dir = root / str(config["experiment_id"]) / f"fold_{self.fold}"
        self.checkpoint_dir = self.output_dir / "checkpoints"
        self.log_dir = self.output_dir / "logs"
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.history_path = self.output_dir / "training_history.csv"
        self.fingerprint = config_fingerprint(config)

    def _frame(self) -> pd.DataFrame:
        return load_p2_manifest(self.config["manifest_path"])

    def _dataset(self, split: str) -> P2ASingleRGBDataset:
        data_cfg = self.config["data"]
        train_cfg = self.config["training"]
        smoke_cfg = self.config.get("smoke", {})
        max_cases = None
        if self.smoke:
            max_cases = int(smoke_cfg.get("train_cases" if split == "train" else "val_cases", 0)) or None
        return P2ASingleRGBDataset(
            self.config["manifest_path"],
            fold=self.fold,
            split=split,
            input_mode=str(self.config["input_mode"]) if split == "train" else "original",
            training=split == "train",
            color_jitter_enabled=bool(self.config.get("color_jitter_enabled", False)),
            color_jitter_probability=float(self.config.get("color_jitter_probability", 0.5)),
            color_jitter_config=self.config.get("color_jitter", {}),
            original_probability=float(self.config.get("original_probability", 0.5)),
            image_size=int(data_cfg["image_size"]),
            horizontal_flip_probability=float(data_cfg["horizontal_flip_probability"]),
            normalization_mean=data_cfg["normalization_mean"],
            normalization_std=data_cfg["normalization_std"],
            original_rgb_override_dir=data_cfg.get("original_rgb_override_dir"),
            max_cases=max_cases,
            seed=int(train_cfg["seed"]) + self.fold + (0 if split == "train" else 1000),
            project_root=self.config["project_root"],
            path_resolution=self.config.get("path_resolution", {}),
        )

    def _loader(self, dataset: P2ASingleRGBDataset, *, shuffle: bool) -> DataLoader:
        generator = torch.Generator()
        base_seed = int(self.config["training"]["seed"]) + self.fold
        generator.manual_seed(base_seed)
        return DataLoader(
            dataset,
            batch_size=int(self.config["training"]["batch_size"]),
            shuffle=bool(shuffle),
            num_workers=self.num_workers,
            pin_memory=self.device.type == "cuda",
            generator=generator,
            worker_init_fn=partial(_seed_worker, base_seed=base_seed),
            collate_fn=p2_a_collate,
        )

    def _make_model(self) -> P2ASingleRGBResNet18:
        return P2ASingleRGBResNet18(
            pretrained=str(self.config["model"].get("pretrained", "imagenet")).lower() not in {"false", "none", "random"},
            num_classes=int(self.config["model"]["num_classes"]),
        ).to(self.device)

    def _metadata(self, train_dataset: P2ASingleRGBDataset, val_dataset: P2ASingleRGBDataset, class_weights: torch.Tensor) -> dict[str, Any]:
        train_frame = train_dataset.frame
        val_frame = val_dataset.frame
        payload = {
            "fold": self.fold,
            "experiment_id": self.config["experiment_id"],
            "input_mode": self.config["input_mode"],
            "train_cases": int(len(train_frame)),
            "val_cases": int(len(val_frame)),
            "train_class_counts": {str(k): int(v) for k, v in train_frame["binary_label"].value_counts().sort_index().to_dict().items()},
            "val_class_counts": {str(k): int(v) for k, v in val_frame["binary_label"].value_counts().sort_index().to_dict().items()},
            "class_weights_from_training_fold_only": {"control": float(class_weights[0]), "patient": float(class_weights[1])},
            "checkpoint_metric": "original_validation_macro_auc",
            "tie_break": "higher_macro_auc_then_higher_macro_f1_then_earlier_epoch",
            "original_rgb_override_dir": self.config["data"].get("original_rgb_override_dir"),
            "smoke": self.smoke,
        }
        (self.output_dir / "fold_metadata.json").write_text(json.dumps(json_safe(payload), ensure_ascii=False, indent=2), encoding="utf-8")
        return payload

    @torch.no_grad()
    def _validate_original(self, model: nn.Module, val_loader: DataLoader, criterion: nn.Module) -> tuple[float, dict[str, Any]]:
        model.eval()
        losses: list[float] = []
        labels: list[int] = []
        probs: list[np.ndarray] = []
        for batch_index, batch in enumerate(val_loader):
            if self.max_batches is not None and batch_index >= int(self.max_batches):
                break
            batch = _to_device(batch, self.device)
            logits, _ = model(batch["image"])
            loss = criterion(logits, batch["label"].long())
            losses.append(float(loss.detach()) * int(batch["label"].size(0)))
            labels.extend(batch["label"].detach().cpu().numpy().astype(int).tolist())
            probs.append(torch.softmax(logits, dim=1).detach().cpu().numpy())
        y_prob = np.concatenate(probs, axis=0)
        metrics = compute_p2_a_metrics(np.asarray(labels, dtype=np.int64), y_prob)
        return float(np.sum(losses) / max(1, len(labels))), metrics

    def _checkpoint_payload(
        self,
        *,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        epoch: int,
        best_metric: float,
        best_macro_f1: float,
        wait: int,
        class_weights: torch.Tensor,
    ) -> dict[str, Any]:
        return {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "epoch": int(epoch),
            "best_metric": float(best_metric),
            "best_macro_auc": float(best_metric),
            "best_macro_f1": float(best_macro_f1),
            "epochs_without_improvement": int(wait),
            "config": self.config,
            "config_fingerprint": self.fingerprint,
            "class_weights": class_weights.detach().cpu(),
            "fold": self.fold,
            "seed": int(self.config["training"]["seed"]),
            "smoke": self.smoke,
        }

    def _resume(self, model: nn.Module, optimizer: torch.optim.Optimizer) -> tuple[int, float, float, int, list[dict[str, Any]]]:
        last_path = self.checkpoint_dir / "last.pth"
        if not last_path.is_file():
            return 1, float("-inf"), float("-inf"), 0, []
        checkpoint = _load_torch(last_path, self.device)
        if checkpoint.get("config_fingerprint") != self.fingerprint:
            raise RuntimeError("Refusing resume because resolved config fingerprint differs from last.pth")
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        records = pd.read_csv(self.history_path).to_dict("records") if self.history_path.is_file() else []
        return (
            int(checkpoint["epoch"]) + 1,
            float(checkpoint.get("best_metric", checkpoint.get("best_macro_auc", float("-inf")))),
            float(checkpoint.get("best_macro_f1", float("-inf"))),
            int(checkpoint.get("epochs_without_improvement", 0)),
            records,
        )

    def train(self, *, resume: bool = False) -> dict[str, Any]:
        seed = int(self.config["training"]["seed"]) + self.fold
        set_random_seed(seed)
        if self.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(self.device)
        save_yaml(self.config, self.output_dir / "config_resolved.yaml")
        train_dataset = self._dataset("train")
        val_dataset = self._dataset("val")
        train_loader = self._loader(train_dataset, shuffle=True)
        val_loader = self._loader(val_dataset, shuffle=False)
        class_weights = compute_class_weights(train_dataset.labels, 2).to(self.device)
        self._metadata(train_dataset, val_dataset, class_weights.detach().cpu())
        model = self._make_model()
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=float(self.config["training"]["learning_rate"]),
            weight_decay=float(self.config["training"]["weight_decay"]),
        )
        criterion = nn.CrossEntropyLoss(weight=class_weights)
        max_epochs = int(self.config["training"]["max_epochs"])
        if self.smoke:
            max_epochs = int(self.config.get("smoke", {}).get("max_epochs", 1))
        start_epoch, best_auc, best_f1, wait, records = self._resume(model, optimizer) if resume else (1, float("-inf"), float("-inf"), 0, [])
        patience = int(self.config["training"]["early_stopping_patience"])
        accumulation_steps = max(1, int(self.config["training"].get("gradient_accumulation_steps", 1)))
        started = time.perf_counter()
        for epoch in range(start_epoch, max_epochs + 1):
            if hasattr(train_dataset, "set_epoch"):
                train_dataset.set_epoch(epoch)
            model.train()
            totals = {
                "loss_total": 0.0,
                "loss_cls": 0.0,
                "loss_orig_cls": 0.0,
                "loss_relight_cls": 0.0,
                "loss_pred": 0.0,
                "loss_feat": 0.0,
                "original_confidence": 0.0,
                "relighted_confidence": 0.0,
                "correct": 0,
                "seen": 0,
            }
            preset_totals = {
                name: {"ce": 0.0, "js": 0.0, "feature_cosine": 0.0, "count": 0}
                for name in PRESET_NAMES
            }
            sample_counts = {"original": 0, "relighted": 0, **{name: 0 for name in PRESET_NAMES}}
            seen_case_ids: set[str] = set()
            duplicate_case_within_epoch = False
            warmup_factor = 0.0
            optimizer.zero_grad(set_to_none=True)
            for batch_index, batch in enumerate(train_loader):
                if self.max_batches is not None and batch_index >= int(self.max_batches):
                    break
                batch = _to_device(batch, self.device)
                labels = batch["label"].long()
                if str(self.config["input_mode"]) == "pairwise_consistency":
                    original = batch["original_image"]
                    relighted = batch["relighted_image"]
                    bsz = int(labels.size(0))
                    all_images = torch.cat([original, relighted], dim=0)
                    all_logits, all_features = model(all_images)
                    logits_o = all_logits[:bsz]
                    logits_r = all_logits[bsz:]
                    features_o = all_features[:bsz]
                    features_r = all_features[bsz:]
                    loss_out = p2_a_pairwise_loss(
                        criterion=criterion,
                        labels=labels,
                        logits_original=logits_o,
                        features_original=features_o,
                        logits_relighted=logits_r,
                        features_relighted=features_r,
                        epoch=epoch,
                        prediction_weight=float(self.config["consistency_loss_weights"]["prediction"]),
                        feature_weight=float(self.config["consistency_loss_weights"]["feature"]),
                        warmup_epochs=int(self.config["consistency_warmup_epochs"]),
                    )
                    logits_for_acc = logits_o
                    sample_counts["original"] += bsz
                    sample_counts["relighted"] += bsz
                    preset_names = [str(name) for name in batch.get("sampled_preset_name", batch["preset_name"])]
                    for case_id in batch["case_id"]:
                        case_id_text = str(case_id)
                        if case_id_text in seen_case_ids:
                            duplicate_case_within_epoch = True
                        seen_case_ids.add(case_id_text)
                    cosine_per_pair = torch.nn.functional.cosine_similarity(features_o, features_r, dim=1)
                    for preset in PRESET_NAMES:
                        indices = [idx for idx, name in enumerate(preset_names) if name == preset]
                        if not indices:
                            continue
                        idx_tensor = torch.as_tensor(indices, dtype=torch.long, device=self.device)
                        sample_counts[preset] += int(idx_tensor.numel())
                        preset_totals[preset]["ce"] += float(criterion(logits_r.index_select(0, idx_tensor), labels.index_select(0, idx_tensor)).detach()) * int(idx_tensor.numel())
                        preset_totals[preset]["js"] += float(symmetric_js_divergence(logits_o.index_select(0, idx_tensor), logits_r.index_select(0, idx_tensor)).detach()) * int(idx_tensor.numel())
                        preset_totals[preset]["feature_cosine"] += float(cosine_per_pair.index_select(0, idx_tensor).mean().detach()) * int(idx_tensor.numel())
                        preset_totals[preset]["count"] += int(idx_tensor.numel())
                elif str(self.config["input_mode"]) == "full6_consistency":
                    original = batch["original_image"]
                    relighted = batch["relighted_images"]
                    bsz = int(labels.size(0))
                    views = torch.cat([original.unsqueeze(1), relighted], dim=1)
                    logits_all, features_all = model(views.reshape(bsz * 7, 3, views.size(-2), views.size(-1)))
                    logits_all = logits_all.reshape(bsz, 7, -1)
                    features_all = features_all.reshape(bsz, 7, -1)
                    loss_out = p2_a_full6_loss(
                        criterion=criterion,
                        labels=labels,
                        logits_original=logits_all[:, 0, :],
                        features_original=features_all[:, 0, :],
                        logits_relighted=logits_all[:, 1:, :],
                        features_relighted=features_all[:, 1:, :],
                        epoch=epoch,
                        prediction_weight=float(self.config["consistency_loss_weights"]["prediction"]),
                        feature_weight=float(self.config["consistency_loss_weights"]["feature"]),
                        warmup_epochs=int(self.config["consistency_warmup_epochs"]),
                    )
                    logits_for_acc = logits_all[:, 0, :]
                    sample_counts["original"] += bsz
                    sample_counts["relighted"] += bsz * len(PRESET_NAMES)
                    for preset_index, name in enumerate(PRESET_NAMES):
                        sample_counts[name] += bsz
                        if loss_out.relight_ce_by_preset is not None:
                            preset_totals[name]["ce"] += float(loss_out.relight_ce_by_preset[preset_index].detach()) * bsz
                        if loss_out.pred_by_preset is not None:
                            preset_totals[name]["js"] += float(loss_out.pred_by_preset[preset_index].detach()) * bsz
                        if loss_out.feature_cosine_by_preset is not None:
                            preset_totals[name]["feature_cosine"] += float(loss_out.feature_cosine_by_preset[preset_index].detach()) * bsz
                        preset_totals[name]["count"] += bsz
                elif str(self.config["input_mode"]) == "paired":
                    logits_o, features_o = model(batch["original_image"])
                    logits_c, features_c = model(batch["counterfactual_image"])
                    loss_out = p2_a_loss(
                        criterion=criterion,
                        labels=labels,
                        logits_original=logits_o,
                        features_original=features_o,
                        logits_counterfactual=logits_c,
                        features_counterfactual=features_c,
                        consistency_enabled=True,
                        epoch=epoch,
                        prediction_weight=float(self.config["consistency_loss_weights"]["prediction"]),
                        feature_weight=float(self.config["consistency_loss_weights"]["feature"]),
                        warmup_epochs=int(self.config["consistency_warmup_epochs"]),
                    )
                    logits_for_acc = logits_o
                    sample_counts["original"] += int(labels.size(0))
                    sample_counts["relighted"] += int(labels.size(0))
                    for name in batch["preset_name"]:
                        sample_counts[str(name)] += 1
                else:
                    logits, features = model(batch["image"])
                    loss_out = p2_a_loss(criterion=criterion, labels=labels, logits_original=logits, features_original=features)
                    logits_for_acc = logits
                    for source in batch["source_type"]:
                        sample_counts[str(source)] += 1
                    for name in batch["preset_name"]:
                        if name is not None and str(name) in PRESET_NAMES:
                            sample_counts[str(name)] += 1
                if not torch.isfinite(loss_out.loss_total):
                    raise RuntimeError("non-finite P2-A training loss")
                bsz = int(labels.size(0))
                (loss_out.loss_total / float(accumulation_steps)).backward()
                next_batch_index = batch_index + 1
                reached_loader_end = next_batch_index == len(train_loader)
                reached_max_batches = self.max_batches is not None and next_batch_index >= int(self.max_batches)
                should_step = (next_batch_index % accumulation_steps == 0) or reached_loader_end or reached_max_batches
                if should_step:
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)
                totals["loss_total"] += float(loss_out.loss_total.detach()) * bsz
                totals["loss_cls"] += float(loss_out.loss_cls.detach()) * bsz
                orig_cls_value = loss_out.loss_orig_cls if loss_out.loss_orig_cls is not None else loss_out.loss_cls
                relight_cls_value = loss_out.loss_relight_cls if loss_out.loss_relight_cls is not None else loss_out.loss_total.new_tensor(0.0)
                totals["loss_orig_cls"] += float(orig_cls_value.detach()) * bsz
                totals["loss_relight_cls"] += float(relight_cls_value.detach()) * bsz
                totals["loss_pred"] += float(loss_out.loss_pred.detach()) * bsz
                totals["loss_feat"] += float(loss_out.loss_feat.detach()) * bsz
                if loss_out.mean_original_confidence is not None:
                    totals["original_confidence"] += float(loss_out.mean_original_confidence.detach()) * bsz
                if loss_out.mean_relighted_confidence is not None:
                    totals["relighted_confidence"] += float(loss_out.mean_relighted_confidence.detach()) * bsz
                totals["correct"] += int((logits_for_acc.argmax(dim=1) == labels).sum().item())
                totals["seen"] += bsz
                warmup_factor = float(loss_out.consistency_warmup_factor)
            val_loss, val_metrics = self._validate_original(model, val_loader, criterion)
            macro_auc = float(val_metrics["macro_auc"])
            macro_f1 = float(val_metrics["macro_f1"])
            improved = math.isfinite(macro_auc) and (
                macro_auc > best_auc or (math.isclose(macro_auc, best_auc, abs_tol=1e-12) and macro_f1 > best_f1)
            )
            if improved:
                best_auc = macro_auc
                best_f1 = macro_f1
                wait = 0
            else:
                wait += 1
            record = {
                "epoch": int(epoch),
                "train_total_loss": totals["loss_total"] / max(1, totals["seen"]),
                "train_cls_loss": totals["loss_cls"] / max(1, totals["seen"]),
                "train_orig_cls_loss": totals["loss_orig_cls"] / max(1, totals["seen"]),
                "train_relight_cls_loss": totals["loss_relight_cls"] / max(1, totals["seen"]),
                "train_pred_loss": totals["loss_pred"] / max(1, totals["seen"]),
                "train_feat_loss": totals["loss_feat"] / max(1, totals["seen"]),
                "mean_original_confidence": totals["original_confidence"] / max(1, totals["seen"]),
                "mean_relighted_confidence": totals["relighted_confidence"] / max(1, totals["seen"]),
                "train_accuracy": totals["correct"] / max(1, totals["seen"]),
                "val_loss": val_loss,
                "val_macro_auc": macro_auc,
                "val_macro_f1": macro_f1,
                "val_balanced_accuracy": float(val_metrics["balanced_accuracy"]),
                "val_sensitivity": float(val_metrics["patient_sensitivity"]),
                "val_specificity": float(val_metrics["control_specificity"]),
                "learning_rate": float(optimizer.param_groups[0]["lr"]),
                "warmup_factor": warmup_factor,
                "physical_case_batch_size": int(self.config["training"]["batch_size"]),
                "gradient_accumulation_steps": int(accumulation_steps),
                "effective_case_batch_size": int(self.config["training"]["batch_size"]) * int(accumulation_steps),
                "images_per_step": int(self.config["training"]["batch_size"]) * (7 if str(self.config["input_mode"]) == "full6_consistency" else (2 if str(self.config["input_mode"]) == "pairwise_consistency" else 1)),
                "original_sample_count": int(sample_counts["original"]),
                "relighted_sample_count": int(sample_counts["relighted"]),
                **{f"preset_{name}_count": int(sample_counts[name]) for name in PRESET_NAMES},
                **{
                    f"preset_{name}_mean_ce": preset_totals[name]["ce"] / max(1, preset_totals[name]["count"])
                    for name in PRESET_NAMES
                },
                **{
                    f"preset_{name}_mean_js": preset_totals[name]["js"] / max(1, preset_totals[name]["count"])
                    for name in PRESET_NAMES
                },
                **{
                    f"preset_{name}_mean_feature_cosine": preset_totals[name]["feature_cosine"] / max(1, preset_totals[name]["count"])
                    for name in PRESET_NAMES
                },
                "is_best": bool(improved),
            }
            records.append(record)
            pd.DataFrame(records).to_csv(self.history_path, index=False, encoding="utf-8-sig")
            if str(self.config["input_mode"]) == "pairwise_consistency":
                coverage_row = {
                    "epoch": int(epoch),
                    "case_count": int(totals["seen"]),
                    "unique_case_count": int(len(seen_case_ids)),
                    "duplicate_case_within_epoch": bool(duplicate_case_within_epoch),
                    "cycle_index": int((epoch - 1) // len(PRESET_NAMES)),
                    "cycle_position": int((epoch - 1) % len(PRESET_NAMES)),
                    **{f"preset_{name}_count": int(sample_counts[name]) for name in PRESET_NAMES},
                }
                coverage_path = self.output_dir / "pair_coverage_log.csv"
                previous = pd.read_csv(coverage_path).to_dict("records") if coverage_path.is_file() else []
                previous = [row for row in previous if int(row["epoch"]) != int(epoch)]
                pd.DataFrame([*previous, coverage_row]).sort_values("epoch").to_csv(coverage_path, index=False, encoding="utf-8-sig")
            payload = self._checkpoint_payload(
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                best_metric=best_auc,
                best_macro_f1=best_f1,
                wait=wait,
                class_weights=class_weights,
            )
            torch.save(payload, self.checkpoint_dir / "last.pth")
            if improved or not (self.checkpoint_dir / "best_macro_auc.pth").is_file():
                torch.save(payload, self.checkpoint_dir / "best_macro_auc.pth")
            print(
                f"fold={self.fold} exp={self.config['experiment_id']} epoch={epoch}/{max_epochs} "
                f"train_loss={record['train_total_loss']:.5f} val_macro_auc={macro_auc:.4f} "
                f"val_macro_f1={macro_f1:.4f} best={best_auc:.4f} patience={wait}/{patience}",
                flush=True,
            )
            if wait >= patience:
                break
        return {
            "status": "trained",
            "experiment_id": self.config["experiment_id"],
            "fold": self.fold,
            "output_dir": str(self.output_dir),
            "epochs_completed": int(records[-1]["epoch"]) if records else 0,
            "best_macro_auc": float(best_auc),
            "training_seconds": float(time.perf_counter() - started),
            "max_cuda_memory_allocated_mb": float(torch.cuda.max_memory_allocated(self.device) / (1024**2)) if self.device.type == "cuda" else None,
            "physical_case_batch_size": int(self.config["training"]["batch_size"]),
            "gradient_accumulation_steps": int(accumulation_steps),
            "effective_case_batch_size": int(self.config["training"]["batch_size"]) * int(accumulation_steps),
        }

    def evaluate(self) -> dict[str, Any]:
        model = self._make_model()
        checkpoint = _load_torch(self.checkpoint_dir / "best_macro_auc.pth", self.device)
        model.load_state_dict(checkpoint["model_state_dict"])
        smoke_val = None
        if self.smoke:
            smoke_val = int(self.config.get("smoke", {}).get("val_cases", 0)) or None
        evaluator = P2AEvaluator(model, self.config, fold=self.fold, device=self.device, output_dir=self.output_dir)
        sample_seed = int(self.config["training"]["seed"]) + 5000 + self.fold if smoke_val is not None else None
        original = evaluator.evaluate_original(max_cases=smoke_val, num_workers=self.num_workers, sample_seed=sample_seed)
        relighted = evaluator.evaluate_relighted(max_cases=smoke_val, num_workers=self.num_workers, sample_seed=sample_seed)
        summary = {
            "status": "evaluated",
            "experiment_id": self.config["experiment_id"],
            "fold": self.fold,
            "output_dir": str(self.output_dir),
            "original": original,
            "relighted": relighted,
            "checkpoint": str(self.checkpoint_dir / "best_macro_auc.pth"),
        }
        (self.output_dir / "fold_summary.json").write_text(json.dumps(json_safe(summary), ensure_ascii=False, indent=2), encoding="utf-8")
        return summary
