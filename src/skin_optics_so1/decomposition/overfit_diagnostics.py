"""Reusable evaluation, BatchNorm, preview, and training helpers for SO-1C-R1."""

from __future__ import annotations

import copy
import csv
import json
import math
import time
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from .diagnostic_metrics import (
    constant_baseline,
    cross_channel_matrices,
    diagnostic_metrics,
    final_output_gradient_norms,
    pair_prediction_drift,
)
from .losses import CHANNEL_NAMES, masked_smooth_l1_loss
from .unet_decomposer import SO1UNetDecomposer


def _device_batch(batch: dict[str, Any], device: torch.device) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    return tuple(batch[key].to(device, non_blocking=True) for key in ("linear_rgb", "target_mhsp", "valid_mask"))  # type: ignore[return-value]


@torch.inference_mode()
def collect_predictions(model: nn.Module, loader: DataLoader, device: torch.device, *, train_mode: bool = False) -> dict[str, Any]:
    model.train(train_mode)
    predictions: list[torch.Tensor] = []
    rgbs: list[torch.Tensor] = []
    targets: list[torch.Tensor] = []
    masks: list[torch.Tensor] = []
    ids: list[str] = []
    metadata: list[dict[str, Any]] = []
    for batch in loader:
        rgb, target, mask = _device_batch(batch, device)
        prediction = model(rgb)
        predictions.append(prediction.cpu())
        rgbs.append(rgb.cpu())
        targets.append(target.cpu())
        masks.append(mask.cpu())
        ids.extend([str(item) for item in batch["sample_id"]])
        for index in range(len(batch["sample_id"])):
            metadata.append({
                "sample_id": str(batch["sample_id"][index]),
                "base_latent_id": int(batch["base_latent_id"][index]),
                "acquisition_variant_id": int(batch["acquisition_variant_id"][index]),
            })
    return {"prediction": torch.cat(predictions), "target": torch.cat(targets), "mask": torch.cat(masks), "rgb": torch.cat(rgbs), "sample_ids": ids, "metadata": metadata}


def summarize_collection(collection: dict[str, Any]) -> dict[str, Any]:
    prediction, target, mask = collection["prediction"], collection["target"], collection["mask"]
    summary = diagnostic_metrics(prediction, target, mask)
    summary["constant_baseline"] = constant_baseline(prediction, target, mask)
    mae, pearson, counts = cross_channel_matrices(prediction, target, mask)
    summary["cross_channel_mae"] = mae.tolist()
    summary["cross_channel_pearson"] = pearson.tolist()
    summary["cross_channel_pearson_valid_counts"] = counts.tolist()
    if prediction.shape[0] == 2:
        summary["pair_prediction_drift"] = pair_prediction_drift(prediction, mask)
    per_sample: list[dict[str, Any]] = []
    for index, sample_id in enumerate(collection["sample_ids"]):
        single = diagnostic_metrics(prediction[index : index + 1], target[index : index + 1], mask[index : index + 1])
        row = {**collection["metadata"][index], "sample_id": sample_id}
        row.update({f"{name}_MAE": single["channels"][name]["MAE"] for name in CHANNEL_NAMES})
        per_sample.append(row)
    summary["per_sample"] = per_sample
    return summary


def write_matrix_csv(path: Path, matrix: list[list[float]], *, value_name: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow([value_name, *[f"True {name}" for name in CHANNEL_NAMES]])
        for name, row in zip(CHANNEL_NAMES, matrix):
            writer.writerow([f"Pred {name}", *row])


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=True) + "\n", encoding="utf-8")


def save_diagnostic_summary(output_dir: Path, summary: dict[str, Any], *, stem: str) -> None:
    write_json(output_dir / f"{stem}_metrics.json", summary)
    write_matrix_csv(output_dir / f"{stem}_cross_channel_mae.csv", summary["cross_channel_mae"], value_name="MAE")
    write_matrix_csv(output_dir / f"{stem}_cross_channel_pearson.csv", summary["cross_channel_pearson"], value_name="Pearson")
    pd_rows = summary["per_sample"]
    if pd_rows:
        with (output_dir / f"{stem}_per_sample_metrics.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(pd_rows[0].keys()))
            writer.writeheader(); writer.writerows(pd_rows)


def _preview_indices(summary: dict[str, Any]) -> list[int]:
    rows = summary["per_sample"]
    if not rows:
        return []
    sorted_h = sorted(range(len(rows)), key=lambda i: rows[i]["H_MAE"])
    p_index = max(range(len(rows)), key=lambda i: rows[i].get("P_MAE", -math.inf))
    return list(dict.fromkeys([sorted_h[0], sorted_h[len(sorted_h) // 2], sorted_h[-1], p_index]))


def save_previews(output_dir: Path, collection: dict[str, Any], summary: dict[str, Any], *, step: int, all_samples: bool = False) -> None:
    indices = list(range(len(collection["sample_ids"]))) if all_samples else _preview_indices(summary)
    preview_dir = output_dir / "previews"
    preview_dir.mkdir(parents=True, exist_ok=True)
    pred, true, mask = collection["prediction"].numpy(), collection["target"].numpy(), collection["mask"].numpy()
    for index in indices:
        figure, axes = plt.subplots(5, 4, figsize=(13, 15), constrained_layout=True)
        rgb = np.transpose(np.clip(collection["rgb"][index].numpy(), 0, 1), (1, 2, 0))
        axes[0, 0].imshow(rgb); axes[0, 0].set_title("linear RGB")
        axes[0, 1].imshow(mask[index, 0], cmap="gray", vmin=0, vmax=1); axes[0, 1].set_title("valid mask")
        axes[0, 2].axis("off"); axes[0, 3].axis("off")
        row = summary["per_sample"][index]
        for channel, name in enumerate(CHANNEL_NAMES):
            vmax = max(1.0e-6, float(max(true[index, channel].max(), pred[index, channel].max())))
            axes[channel + 1, 0].imshow(true[index, channel], cmap="viridis", vmin=0, vmax=vmax); axes[channel + 1, 0].set_title(f"{name} true")
            axes[channel + 1, 1].imshow(pred[index, channel], cmap="viridis", vmin=0, vmax=vmax); axes[channel + 1, 1].set_title(f"{name} pred")
            axes[channel + 1, 2].imshow(np.abs(true[index, channel] - pred[index, channel]), cmap="magma"); axes[channel + 1, 2].set_title(f"{name} abs error")
            axes[channel + 1, 3].axis("off")
        for axis in axes.ravel(): axis.set_xticks([]); axis.set_yticks([])
        figure.suptitle(f"{row['sample_id']} | base={row['base_latent_id']} | step={step} | " + " ".join(f"{n}_MAE={row[f'{n}_MAE']:.5f}" for n in CHANNEL_NAMES), fontsize=10)
        figure.savefig(preview_dir / f"step_{step:04d}_{row['sample_id']}.png", dpi=140)
        plt.close(figure)


def batchnorm_diagnostic(model: nn.Module, loader: DataLoader, device: torch.device) -> dict[str, Any]:
    base = collect_predictions(model, loader, device, train_mode=False)
    train_copy = copy.deepcopy(model).to(device)
    train_mode = collect_predictions(train_copy, loader, device, train_mode=True)
    recalibrated = copy.deepcopy(model).to(device)
    for parameter in recalibrated.parameters(): parameter.requires_grad_(False)
    for module in recalibrated.modules():
        if isinstance(module, nn.modules.batchnorm._BatchNorm): module.reset_running_stats()
    recalibrated.train()
    with torch.no_grad():
        for batch in loader:
            rgb, _, _ = _device_batch(batch, device); recalibrated(rgb)
    recal = collect_predictions(recalibrated, loader, device, train_mode=False)
    def maes(value: dict[str, Any]) -> dict[str, float]:
        summary = diagnostic_metrics(value["prediction"], value["target"], value["mask"])
        return {f"{name}_MAE": summary["channels"][name]["MAE"] for name in CHANNEL_NAMES}
    eval_mae, train_mae, recal_mae = maes(base), maes(train_mode), maes(recal)
    return {
        "eval": eval_mae, "train_mode_no_grad": train_mae, "bn_recalibrated_eval": recal_mae,
        "trainmode_vs_eval_relative": {name: (train_mae[name] - eval_mae[name]) / max(eval_mae[name], 1e-12) for name in eval_mae},
        "recalibrated_vs_eval_relative": {name: (recal_mae[name] - eval_mae[name]) / max(eval_mae[name], 1e-12) for name in eval_mae},
        "weights_unchanged": True,
    }


def diagnostic_train(
    *, output_dir: Path, train_loader: DataLoader, eval_loader: DataLoader, device: torch.device,
    seed: int, max_steps: int, evaluation_interval: int, learning_rate: float = 1e-3, weight_decay: float = 1e-4,
) -> dict[str, Any]:
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    model = SO1UNetDecomposer().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    output_dir.mkdir(parents=True, exist_ok=True)
    if device.type == "cuda": torch.cuda.reset_peak_memory_stats(device)
    records: list[dict[str, Any]] = []
    best: dict[str, Any] | None = None
    initial_loss: float | None = None
    iterator = iter(train_loader)
    started = time.perf_counter()
    for step in range(1, max_steps + 1):
        try: batch = next(iterator)
        except StopIteration: iterator = iter(train_loader); batch = next(iterator)
        model.train(); rgb, target, mask = _device_batch(batch, device)
        optimizer.zero_grad(set_to_none=True)
        prediction = model(rgb); losses = masked_smooth_l1_loss(prediction, target, mask)
        losses["total"].backward()
        gradients = final_output_gradient_norms(model)
        total_gradient_norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=float("inf")).detach().cpu())
        optimizer.step()
        if initial_loss is None: initial_loss = float(losses["total"].detach().cpu())
        if step % evaluation_interval == 0 or step == max_steps:
            collection = collect_predictions(model, eval_loader, device)
            summary = summarize_collection(collection)
            channel = summary["channels"]
            record = {
                "step": step, "learning_rate": float(optimizer.param_groups[0]["lr"]),
                "total_loss": float(losses["total"].detach().cpu()),
                **{f"{name}_loss": float(losses[name].detach().cpu()) for name in CHANNEL_NAMES},
                **{f"{name}_MAE": channel[name]["MAE"] for name in CHANNEL_NAMES},
                "selection_metric": channel["M"]["MAE"] + channel["H"]["MAE"],
                "total_gradient_norm": total_gradient_norm, **gradients,
                "elapsed_seconds": time.perf_counter() - started,
            }
            records.append(record)
            if best is None or record["selection_metric"] < best["record"]["selection_metric"]:
                best = {"record": record, "summary": summary, "collection": collection, "state_dict": {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}}
            if step in {1000, 2000, 3000}:
                write_json(output_dir / f"step_{step}_metrics.json", record)
    assert best is not None and initial_loss is not None
    model.load_state_dict(best["state_dict"])
    best_collection = collect_predictions(model, eval_loader, device)
    best_summary = summarize_collection(best_collection)
    bn = batchnorm_diagnostic(model, eval_loader, device)
    write_json(output_dir / "best_metrics.json", best_summary)
    write_json(output_dir / "bn_diagnostic.json", bn)
    with (output_dir / "training_history.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0].keys())); writer.writeheader(); writer.writerows(records)
    torch.save({"model_state_dict": best["state_dict"], "step": best["record"]["step"], "seed": seed}, output_dir / "best_diagnostic_weights.pt")
    peak = int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else 0
    result = {
        "initial_loss": initial_loss, "best": best["record"], "final": records[-1], "loss_reduction": 1.0 - records[-1]["total_loss"] / max(initial_loss, 1e-12),
        "best_summary": best_summary, "bn": bn, "gpu_peak_memory_bytes": peak,
        "steps_per_second": max_steps / max(time.perf_counter() - started, 1e-12),
    }
    save_previews(output_dir, best_collection, best_summary, step=int(best["record"]["step"]), all_samples=len(best_collection["sample_ids"]) <= 2)
    write_json(output_dir / "result.json", result)
    return result


def load_trusted_checkpoint(path: Path, device: torch.device) -> tuple[SO1UNetDecomposer, dict[str, Any]]:
    """The caller has explicitly authorized local trusted legacy checkpoint deserialization."""
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    model = SO1UNetDecomposer().to(device)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    return model, checkpoint
