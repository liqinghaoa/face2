"""Reproduce and localize the SO-1D epoch-7 numerical failure."""

from __future__ import annotations

import argparse
import copy
import csv
import json
import math
import random
import sys
import traceback
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from skin_optics_so1.decomposition.checkpoint import load_checkpoint  # noqa: E402
from skin_optics_so1.decomposition.formal_protocol import (  # noqa: E402
    build_formal_identity,
    validate_dataset_contract,
)
from skin_optics_so1.decomposition.losses import CHANNEL_NAMES, masked_smooth_l1_loss  # noqa: E402
from skin_optics_so1.decomposition.numerical_diagnostics import (  # noqa: E402
    all_rows_finite,
    atomic_torch_save,
    batchnorm_health,
    epoch7_order_from_generator_state,
    gradient_health,
    model_parameter_health,
    optimizer_state_health,
    sample_order_sha256,
    state_finite,
    tensor_health,
    write_csv,
    write_json,
)
from skin_optics_so1.decomposition.synthetic_dataset import SO1DecompositionDataset  # noqa: E402
from skin_optics_so1.decomposition.unet_decomposer import SO1UNetDecomposer  # noqa: E402


FORMAL_DIR = PROJECT_ROOT / "experiments/skin_optics_so1/SO1_Decomposition_UNet_v1/formal_train"
LAST_PATH = FORMAL_DIR / "checkpoints/last.pt"
BEST_PATH = FORMAL_DIR / "checkpoints/best.pt"
DATA_ROOT = PROJECT_ROOT / "data/processed/SO1_Synthetic_Generator_v1_localfull"
OUTPUT_DIR = (
    PROJECT_ROOT
    / "experiments/skin_optics_so1/SO1_Decomposition_UNet_v1/diagnostics/so1d_r1_epoch7_numerical_failure"
)
CONFIG = {
    "protocol_version": "SO1D_Formal_Train_v1.0",
    "seed": 20260801,
    "model": {"architecture": "standard_unet", "encoder_channels": [64, 128, 256, 512], "bottleneck_channels": 1024, "decoder_channels": [512, 256, 128, 64], "output_channels": 4, "output_order": ["M_norm", "H_norm", "S_norm", "P_norm"], "normalization": "BatchNorm", "activation": "LeakyReLU", "final_activation": "Sigmoid"},
    "loss": {"type": "masked_smooth_l1", "beta": 0.1, "channel_weights": [0.25] * 4},
    "optimizer": {"type": "AdamW", "lr": 0.001, "weight_decay": 0.0001},
    "scheduler": {"type": "CosineAnnealingLR", "T_max": 30, "eta_min": 0.000001, "step_unit": "epoch"},
    "training": {"batch_size": 8, "amp": True, "max_epochs": 30},
    "early_stopping": {"enabled": True, "monitor": "val_M_MAE_plus_val_H_MAE", "mode": "min", "patience": 5, "min_delta": 0.0},
    "checkpoint": {"selection_metric": "val_M_MAE_plus_val_H_MAE", "save_best": True, "save_last": True, "milestone_epochs": [5, 10, 15, 20, 25, 30]},
    "dataloader": {"num_workers": 2, "pin_memory": True, "persistent_workers": True, "prefetch_factor": 2},
    "augmentation": {"enabled": False},
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--audit-only", action="store_true")
    parser.add_argument("--max-batches", type=int, default=2500)
    parser.add_argument("--complete-evidence-only", action="store_true")
    return parser.parse_args()


def set_determinism() -> None:
    random.seed(20260801)
    np.random.seed(20260801)
    torch.manual_seed(20260801)
    torch.cuda.manual_seed_all(20260801)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def build_runtime(checkpoint_path: Path, *, amp: bool) -> tuple[Any, ...]:
    device = torch.device("cuda")
    model = SO1UNetDecomposer().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.0001)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=30, eta_min=0.000001)
    scaler = torch.amp.GradScaler("cuda", enabled=amp)
    contract, fingerprint, _ = validate_dataset_contract(
        data_root=DATA_ROOT,
        so1c_contract_path=FORMAL_DIR / "dataset_contract.json",
        so1c_checkpoint_path=FORMAL_DIR / "checkpoints/last.pt",
    )
    identity = build_formal_identity(contract_hash=fingerprint["contract_hash"], config=CONFIG)
    checkpoint = load_checkpoint(
        checkpoint_path, model=model, optimizer=optimizer, scheduler=scheduler,
        grad_scaler=scaler if amp else None, expected_identity=identity,
        map_location=device, restore_rng=True,
    )
    if not amp:
        scaler = torch.amp.GradScaler("cuda", enabled=False)
    return model, optimizer, scheduler, scaler, checkpoint


def checkpoint_audit() -> dict[str, Any]:
    health_dir = OUTPUT_DIR / "checkpoint_health"
    model6, optimizer6, _, scaler6, checkpoint6 = build_runtime(LAST_PATH, amp=True)
    params6 = model_parameter_health(model6)
    bn6 = batchnorm_health(model6)
    opt6 = optimizer_state_health(optimizer6, model6)
    write_csv(health_dir / "model_parameter_health.csv", params6)
    write_csv(health_dir / "batchnorm_health.csv", bn6)
    write_csv(health_dir / "optimizer_state_health.csv", opt6)
    scaler_payload = {
        "epoch": 6, "global_step": int(checkpoint6["global_step"]),
        "scale": float(scaler6.get_scale()),
        "growth_tracker": int(checkpoint6["grad_scaler_state_dict"]["_growth_tracker"]),
        "history": "NOT_AVAILABLE",
    }
    write_json(health_dir / "gradscaler_health.json", scaler_payload)
    write_csv(health_dir / "gradscaler_history.csv", [{
        "epoch": 6, "global_step": checkpoint6["global_step"],
        "scale": scaler6.get_scale(), "source": "checkpoint_only",
    }])

    model5, optimizer5, _, scaler5, checkpoint5 = build_runtime(BEST_PATH, amp=True)
    p5 = {row["name"]: row for row in model_parameter_health(model5)}
    p6 = {row["name"]: row for row in params6}
    b5 = {(row["module"], row["tensor"]): row for row in batchnorm_health(model5)}
    b6 = {(row["module"], row["tensor"]): row for row in bn6}
    o5 = {(row["parameter"], row["state"]): row for row in optimizer_state_health(optimizer5, model5)}
    o6 = {(row["parameter"], row["state"]): row for row in opt6}
    comparison: list[dict[str, Any]] = []
    for category, left, right in (("parameter", p5, p6), ("batchnorm", b5, b6), ("optimizer", o5, o6)):
        for key in sorted(set(left).intersection(right), key=str):
            for metric in ("max_abs", "std"):
                before, after = left[key].get(metric), right[key].get(metric)
                ratio = float(after / before) if before not in (None, 0.0) and after is not None else None
                comparison.append({
                    "category": category, "tensor": str(key), "metric": metric,
                    "epoch5": before, "epoch6": after, "ratio": ratio,
                    "gt_2x": ratio is not None and ratio > 2,
                    "gt_5x": ratio is not None and ratio > 5,
                    "gt_10x": ratio is not None and ratio > 10,
                    "gt_100x": ratio is not None and ratio > 100,
                })
    comparison.append({
        "category": "gradscaler", "tensor": "scale", "metric": "value",
        "epoch5": float(scaler5.get_scale()), "epoch6": float(scaler6.get_scale()),
        "ratio": float(scaler6.get_scale() / scaler5.get_scale()),
        "gt_2x": scaler6.get_scale() / scaler5.get_scale() > 2,
        "gt_5x": False, "gt_10x": False, "gt_100x": False,
    })
    write_csv(health_dir / "epoch5_vs_epoch6_state_comparison.csv", comparison)
    summary = {
        "checkpoint_epoch": checkpoint6["epoch"], "global_step": checkpoint6["global_step"],
        "model_finite": all_rows_finite(params6),
        "batchnorm_finite": all_rows_finite(bn6),
        "batchnorm_running_var_nonnegative": all(
            row["nonnegative"] is not False for row in bn6
        ),
        "optimizer_finite": all_rows_finite(opt6),
        "gradscaler_scale": float(scaler6.get_scale()),
        "gradscaler_growth_tracker": scaler_payload["growth_tracker"],
        "minimum_running_var": min(row["min"] for row in bn6 if row["tensor"] == "running_var"),
        "maximum_running_var": max(row["max"] for row in bn6 if row["tensor"] == "running_var"),
        "maximum_abs_running_mean": max(row["max_abs"] for row in bn6 if row["tensor"] == "running_mean"),
        "abnormal_growth_gt_10x": any(row["gt_10x"] for row in comparison),
        "epoch5": checkpoint5["epoch"],
    }
    write_json(health_dir / "checkpoint_health_summary.json", summary)
    del model5, model6, optimizer5, optimizer6
    torch.cuda.empty_cache()
    return summary


def build_order(dataset: SO1DecompositionDataset, checkpoint: dict[str, Any]) -> tuple[list[int], list[dict[str, Any]], str]:
    state = checkpoint["dataloader_rng_state"]["train"]
    order = epoch7_order_from_generator_state(
        sample_count=len(dataset), batch_size=8, generator_state=state
    )
    metadata = dataset.metadata.set_index("split_index", drop=False)
    rows: list[dict[str, Any]] = []
    for position, split_index in enumerate(order):
        row = metadata.loc[split_index]
        rows.append({
            "batch_index": position // 8,
            "position_in_batch": position % 8,
            "split_index": int(split_index),
            "sample_id": str(row["sample_id"]),
            "base_latent_id": int(row["base_latent_id"]),
            "acquisition_variant_id": int(row["acquisition_variant_id"]),
            "camera": str(row["camera_name"]),
            "light": str(row["light_name"]),
            "camera_light_pair": str(row["camera_light_pair"]),
        })
    digest = sample_order_sha256(rows)
    write_csv(OUTPUT_DIR / "epoch7_sample_order.csv", rows)
    (OUTPUT_DIR / "epoch7_sample_order_sha256.txt").write_text(digest + "\n", encoding="ascii")
    audit = {
        "random_sampler_rng": "explicit torch.Generator",
        "persistent_worker_reset_draws_new_base_seed": False,
        "dataloader_iterator_base_seed_draw_reproduced": "NOT_APPLICABLE_AFTER_FIRST_EPOCH",
        "sampler_generator_state_in_checkpoint": True,
        "worker_seed_rule": "(torch.initial_seed() + worker_id) % 2**32",
        "dataset_has_random_transforms": False,
        "saved_order_count": len(order),
        "reproduction_level": "EXACT",
        "epoch7_order_sha256": digest,
    }
    write_json(OUTPUT_DIR / "sampler_reproducibility_audit.json", audit)
    return order, rows, digest


def collate_fixed_batch(dataset: SO1DecompositionDataset, indices: list[int]) -> dict[str, Any]:
    samples = [dataset[index] for index in indices]
    tensor_keys = ("linear_rgb", "target_mhsp", "valid_mask")
    result: dict[str, Any] = {key: torch.stack([sample[key] for sample in samples]) for key in tensor_keys}
    for key in ("sample_id", "split_index", "base_latent_id", "acquisition_variant_id"):
        result[key] = [sample[key] for sample in samples]
    return result


def coarse_hooks(model: SO1UNetDecomposer, trace: list[dict[str, Any]], context: dict[str, int]) -> list[Any]:
    names = {
        "inc": "encoder1", "down1": "encoder2", "down2": "encoder3", "down3": "encoder4",
        "down4": "bottleneck", "up1": "decoder4", "up2": "decoder3", "up3": "decoder2",
        "up4": "decoder1", "outc": "final_logits", "out_activation": "sigmoid_output",
    }
    handles = []
    for module_name, label in names.items():
        module = getattr(model, module_name)
        def hook(_module: Any, _inputs: Any, output: torch.Tensor, *, name: str = label) -> None:
            trace.append({
                "batch_index": context["batch_index"], "global_step": context["global_step"],
                "module": name, "autocast_enabled": torch.is_autocast_enabled(),
                **tensor_health(output),
            })
        handles.append(module.register_forward_hook(hook))
    return handles


def bn_snapshot(model: SO1UNetDecomposer) -> dict[str, tuple[bool, bool]]:
    return {
        name: (bool(torch.isfinite(module.running_mean).all()), bool(torch.isfinite(module.running_var).all()))
        for name, module in model.named_modules() if isinstance(module, torch.nn.BatchNorm2d)
    }


def first_nonfinite_trace(rows: list[dict[str, Any]], batch_index: int) -> dict[str, Any] | None:
    return next((row for row in rows if row["batch_index"] == batch_index and not row["finite"]), None)


def save_failure_batch(batch: dict[str, Any], metadata: pd.DataFrame, event: dict[str, Any]) -> None:
    rows = metadata.set_index("split_index", drop=False).loc[batch["split_index"]]
    payload = {
        "linear_rgb": batch["linear_rgb"], "target_mhsp": batch["target_mhsp"],
        "valid_mask": batch["valid_mask"], "sample_ids": batch["sample_id"],
        "split_indices": batch["split_index"], "base_latent_ids": batch["base_latent_id"],
        "acquisition_variant_ids": batch["acquisition_variant_id"],
        "camera": rows["camera_name"].astype(str).tolist(),
        "light": rows["light_name"].astype(str).tolist(),
        "camera_light_pair": rows["camera_light_pair"].astype(str).tolist(),
        "metadata": rows.to_dict("records"), "event": event,
    }
    atomic_torch_save(payload, OUTPUT_DIR / "failing_batch/failing_batch_snapshot.pt")


def save_anchor(
    *, model: Any, optimizer: Any, scheduler: Any, scaler: Any,
    batch_index: int, global_step: int, path: Path,
) -> None:
    atomic_torch_save({
        "model_state_dict": model.state_dict(), "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(), "grad_scaler_state_dict": scaler.state_dict(),
        "python_rng_state": random.getstate(), "numpy_rng_state": np.random.get_state(),
        "torch_cpu_rng_state": torch.get_rng_state(), "torch_cuda_rng_state": torch.cuda.get_rng_state_all(),
        "batch_index": batch_index, "global_step": global_step,
    }, path)


def replay_track_a(dataset: SO1DecompositionDataset, order: list[int], *, max_batches: int) -> dict[str, Any]:
    track = OUTPUT_DIR / "replay_track_a_amp_original"
    model, optimizer, scheduler, scaler, checkpoint = build_runtime(LAST_PATH, amp=True)
    model.train()
    device = torch.device("cuda")
    trace: list[dict[str, Any]] = []
    bn_trace: list[dict[str, Any]] = []
    scaler_trace: list[dict[str, Any]] = []
    gradient_rows: list[dict[str, Any]] = []
    context = {"batch_index": -1, "global_step": int(checkpoint["global_step"])}
    handles = coarse_hooks(model, trace, context)
    first_failure: dict[str, Any] | None = None
    metadata = dataset.metadata
    for batch_index in range(min(max_batches, math.ceil(len(order) / 8))):
        indices = order[batch_index * 8:(batch_index + 1) * 8]
        batch = collate_fixed_batch(dataset, indices)
        context.update(batch_index=batch_index, global_step=int(checkpoint["global_step"]) + batch_index + 1)
        if batch_index % 100 == 0:
            save_anchor(
                model=model, optimizer=optimizer, scheduler=scheduler, scaler=scaler,
                batch_index=batch_index, global_step=context["global_step"] - 1,
                path=OUTPUT_DIR / "anchors/latest_safe_anchor.pt",
            )
        for tensor_name in ("linear_rgb", "target_mhsp", "valid_mask"):
            health = tensor_health(batch[tensor_name])
            if not health["finite"] or (tensor_name == "valid_mask" and float(batch[tensor_name].sum()) <= 0):
                first_failure = {"stage": "DATA", "module": "DataLoader", "tensor_name": tensor_name, **health}
                break
        if first_failure:
            first_failure.update(batch_index=batch_index, global_step=context["global_step"])
            save_failure_batch(batch, metadata, first_failure)
            break
        x = batch["linear_rgb"].to(device)
        target = batch["target_mhsp"].to(device)
        mask = batch["valid_mask"].to(device)
        optimizer.zero_grad(set_to_none=True)
        before_bn = bn_snapshot(model)
        with torch.amp.autocast("cuda", enabled=True):
            prediction = model(x)
            losses = masked_smooth_l1_loss(prediction, target, mask, beta=0.1)
        after_bn = bn_snapshot(model)
        for name in before_bn:
            bn_trace.append({
                "batch_index": batch_index, "global_step": context["global_step"], "bn_module_name": name,
                "before_running_mean_finite": before_bn[name][0], "after_running_mean_finite": after_bn[name][0],
                "before_running_var_finite": before_bn[name][1], "after_running_var_finite": after_bn[name][1],
            })
        failed_trace = first_nonfinite_trace(trace, batch_index)
        if failed_trace:
            first_failure = {"stage": "FORWARD", "tensor_name": "activation", **failed_trace}
        elif not all(torch.isfinite(value).all() for value in losses.values()):
            key = next(name for name, value in losses.items() if not torch.isfinite(value).all())
            first_failure = {"stage": "LOSS", "module": "masked_smooth_l1", "tensor_name": f"{key}_loss", **tensor_health(losses[key])}
        if first_failure:
            first_failure.update(
                batch_index=batch_index, global_step=context["global_step"],
                bn_contamination=any(before_bn[name] != after_bn[name] and not all(after_bn[name]) for name in before_bn),
                parameters_finite=state_finite(model, optimizer)["parameters_finite"],
                optimizer_state_finite=state_finite(model, optimizer)["optimizer_state_finite"],
                gradients_finite=None, scale_before=float(scaler.get_scale()), scale_after=None,
            )
            save_failure_batch(batch, metadata, first_failure)
            break
        scale_before = float(scaler.get_scale())
        scaler.scale(losses["total"]).backward()
        scaler.unscale_(optimizer)
        grad_rows, grad_summary = gradient_health(model)
        for row in grad_rows:
            row.update(batch_index=batch_index, global_step=context["global_step"])
        gradient_rows.extend(grad_rows)
        pre = state_finite(model, optimizer)
        if not grad_summary["finite"]:
            first_failure = {
                "batch_index": batch_index, "global_step": context["global_step"],
                "stage": "BACKWARD", "module": grad_summary["first_nonfinite_module"],
                "tensor_name": grad_summary["first_nonfinite_parameter"], "tensor_dtype": "gradient",
                "num_nan": sum(row["num_nan"] for row in grad_rows),
                "num_inf": sum(row["num_inf"] for row in grad_rows),
                "min": None, "max": None, "mean": None, "std": None,
                "max_abs": grad_summary["global_max_abs_grad"],
                "global_grad_norm": grad_summary["global_grad_norm"],
                "scale_before": scale_before, "scale_after": None,
                "bn_contamination": any(not all(value) for value in after_bn.values()),
                "parameters_finite": pre["parameters_finite"],
                "optimizer_state_finite": pre["optimizer_state_finite"], "gradients_finite": False,
            }
            save_failure_batch(batch, metadata, first_failure)
            break
        output_head_before = model.outc.weight.detach().clone()
        scaler.step(optimizer)
        scaler.update()
        scale_after = float(scaler.get_scale())
        optimizer_step_skipped = bool(torch.equal(output_head_before, model.outc.weight.detach()))
        post = state_finite(model, optimizer)
        scaler_trace.append({
            "batch_index": batch_index, "global_step": context["global_step"],
            "scale_before": scale_before, "scale_after": scale_after,
            "overflow_detected": scale_after < scale_before,
            "optimizer_step_skipped": optimizer_step_skipped,
            "global_grad_norm": grad_summary["global_grad_norm"],
            "global_max_abs_grad": grad_summary["global_max_abs_grad"],
        })
        if not all((post["parameters_finite"], post["optimizer_state_finite"], post["batchnorm_buffers_finite"])):
            first_failure = {
                "batch_index": batch_index, "global_step": context["global_step"],
                "stage": "OPTIMIZER_STEP", "module": post["first_nonfinite_parameter"] or post["first_nonfinite_optimizer_state"] or post["first_nonfinite_batchnorm"],
                "tensor_name": post["first_nonfinite_parameter"] or post["first_nonfinite_optimizer_state"] or post["first_nonfinite_batchnorm"],
                "tensor_dtype": "state", "num_nan": None, "num_inf": None,
                "min": None, "max": None, "mean": None, "std": None, "max_abs": None,
                "scale_before": scale_before, "scale_after": scale_after,
                "bn_contamination": not post["batchnorm_buffers_finite"],
                "parameters_finite": post["parameters_finite"],
                "optimizer_state_finite": post["optimizer_state_finite"], "gradients_finite": True,
                "optimizer_step_skipped": optimizer_step_skipped,
            }
            save_failure_batch(batch, metadata, first_failure)
            break
    for handle in handles:
        handle.remove()
    write_csv(track / "coarse_forward_trace.csv", trace)
    write_csv(track / "batchnorm_before_after.csv", bn_trace)
    write_csv(track / "gradscaler_trace.csv", scaler_trace)
    write_csv(track / "gradient_trace.csv", gradient_rows)
    if first_failure:
        first_failure.update({
            "sample_ids": batch["sample_id"], "split_indices": batch["split_index"],
            "base_latent_ids": batch["base_latent_id"],
            "first_failure_sample_ids": batch["sample_id"],
        })
        write_json(track / "first_failure_event.json", first_failure)
        write_json(OUTPUT_DIR / "first_failure_event.json", first_failure)
    result = {
        "status": "REPRODUCED" if first_failure else "NOT_REPRODUCED",
        "batches_completed_before_failure": first_failure["batch_index"] if first_failure else min(max_batches, 2500),
        "first_failure": first_failure,
    }
    write_json(track / "result.json", result)
    return result


def load_anchor(path: Path, *, amp: bool) -> tuple[Any, ...]:
    model = SO1UNetDecomposer().cuda()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.0001)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=30, eta_min=0.000001)
    scaler = torch.amp.GradScaler("cuda", enabled=amp)
    anchor = torch.load(path, map_location="cuda", weights_only=False)
    model.load_state_dict(anchor["model_state_dict"])
    optimizer.load_state_dict(anchor["optimizer_state_dict"])
    scheduler.load_state_dict(anchor["scheduler_state_dict"])
    if amp:
        scaler.load_state_dict(anchor["grad_scaler_state_dict"])
    return model, optimizer, scheduler, scaler, anchor


def controlled_track(name: str, *, amp: bool, backward: bool, step: bool) -> dict[str, Any]:
    snapshot = torch.load(OUTPUT_DIR / "failing_batch/failing_batch_snapshot.pt", map_location="cpu", weights_only=False)
    model, optimizer, _, scaler, anchor = load_anchor(OUTPUT_DIR / "anchors/latest_safe_anchor.pt", amp=True)
    model.train()
    # Replay finite batches from the safe anchor up to the failing batch to recover identical state.
    dataset = SO1DecompositionDataset(DATA_ROOT, "train")
    order_rows = pd.read_csv(OUTPUT_DIR / "epoch7_sample_order.csv")
    failing_index = int(snapshot["event"]["batch_index"])
    start = int(anchor["batch_index"])
    result: dict[str, Any] = {"track": name, "amp": amp, "backward": backward, "optimizer_step": step}
    for batch_index in range(start, failing_index + 1):
        indices = order_rows.loc[order_rows["batch_index"] == batch_index, "split_index"].astype(int).tolist()
        batch = collate_fixed_batch(dataset, indices)
        x, target, mask = (batch[key].cuda() for key in ("linear_rgb", "target_mhsp", "valid_mask"))
        is_failing_batch = batch_index == failing_index
        current_amp = amp if is_failing_batch else True
        current_backward = backward if is_failing_batch else True
        current_step = step if is_failing_batch else True
        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda", enabled=current_amp):
            prediction = model(x)
            losses = masked_smooth_l1_loss(prediction, target, mask, beta=0.1)
        forward_finite = bool(torch.isfinite(prediction).all())
        loss_finite = all(bool(torch.isfinite(value).all()) for value in losses.values())
        backward_finite: bool | None = None
        if current_backward and forward_finite and loss_finite:
            if current_amp:
                scaler.scale(losses["total"]).backward()
                scaler.unscale_(optimizer)
            else:
                losses["total"].backward()
            _, grad = gradient_health(model)
            backward_finite = bool(grad["finite"])
        if current_step and backward_finite:
            if current_amp:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
        if is_failing_batch:
            state = state_finite(model, optimizer)
            result.update({
                "batch_index": batch_index, "forward_finite": forward_finite,
                "loss_finite": loss_finite, "backward_finite": backward_finite,
                "parameters_finite": state["parameters_finite"],
                "optimizer_state_finite": state["optimizer_state_finite"],
                "batchnorm_buffers_finite": state["batchnorm_buffers_finite"],
                "status": "PASS" if forward_finite and loss_finite and (not backward or backward_finite) and state["parameters_finite"] else "FAIL",
            })
            break
    write_json(OUTPUT_DIR / name / "result.json", result)
    return result


def fine_localization() -> dict[str, Any]:
    snapshot = torch.load(
        OUTPUT_DIR / "failing_batch/failing_batch_snapshot.pt", map_location="cpu", weights_only=False
    )
    model, optimizer, _, scaler, anchor = load_anchor(
        OUTPUT_DIR / "anchors/latest_safe_anchor.pt", amp=True
    )
    model.train()
    dataset = SO1DecompositionDataset(DATA_ROOT, "train")
    order_rows = pd.read_csv(OUTPUT_DIR / "epoch7_sample_order.csv")
    failing_index = int(snapshot["event"]["batch_index"])
    hook_start = max(int(anchor["batch_index"]), failing_index - 9)
    layer_rows: list[dict[str, Any]] = []
    handles: list[Any] = []
    context = {"batch_index": -1, "global_step": 15000}

    def add_output_hook(module_name: str, operation: str) -> Callable[..., None]:
        def hook(_module: Any, _inputs: Any, output: Any) -> None:
            value = output[0] if isinstance(output, tuple) else output
            if torch.is_tensor(value):
                layer_rows.append({
                    "batch_index": context["batch_index"],
                    "global_step": context["global_step"],
                    "module": module_name, "operation": operation,
                    "tensor": "output", **tensor_health(value),
                })
        return hook

    def add_input_hook(module_name: str, operation: str, tensor_name: str = "input") -> Callable[..., None]:
        def hook(_module: Any, inputs: Any) -> None:
            value = inputs[0]
            if torch.is_tensor(value):
                layer_rows.append({
                    "batch_index": context["batch_index"],
                    "global_step": context["global_step"],
                    "module": module_name, "operation": operation,
                    "tensor": tensor_name, **tensor_health(value),
                })
        return hook

    def install_hooks() -> None:
        for name, module in model.named_modules():
            if isinstance(module, torch.nn.Conv2d):
                handles.append(module.register_forward_hook(add_output_hook(name, "Conv2d")))
            elif isinstance(module, torch.nn.BatchNorm2d):
                handles.append(module.register_forward_pre_hook(add_input_hook(name, "BatchNorm2d", "input")))
                handles.append(module.register_forward_hook(add_output_hook(name, "BatchNorm2d")))
            elif isinstance(module, torch.nn.LeakyReLU):
                handles.append(module.register_forward_hook(add_output_hook(name, "LeakyReLU")))
            elif isinstance(module, torch.nn.MaxPool2d):
                handles.append(module.register_forward_hook(add_output_hook(name, "MaxPool2d")))
            elif isinstance(module, torch.nn.ConvTranspose2d):
                handles.append(module.register_forward_hook(add_output_hook(name, "ConvTranspose2d")))
        for up_name in ("up1", "up2", "up3", "up4"):
            up = getattr(model, up_name)
            handles.append(up.register_forward_pre_hook(add_input_hook(up_name, "UpBlock", "upsample_input")))
            def skip_hook(_module: Any, inputs: Any, *, name: str = up_name) -> None:
                layer_rows.append({
                    "batch_index": context["batch_index"], "global_step": context["global_step"],
                    "module": name, "operation": "skip", "tensor": "skip_tensor",
                    **tensor_health(inputs[1]),
                })
            handles.append(up.register_forward_pre_hook(skip_hook))
            handles.append(up.conv.register_forward_pre_hook(add_input_hook(f"{up_name}.conv", "concat", "concat_tensor")))

    hooks_installed = False
    failure_stage = None
    anomaly_text = None
    for batch_index in range(int(anchor["batch_index"]), failing_index + 1):
        if batch_index >= hook_start and not hooks_installed:
            install_hooks()
            hooks_installed = True
        indices = order_rows.loc[order_rows["batch_index"] == batch_index, "split_index"].astype(int).tolist()
        batch = collate_fixed_batch(dataset, indices)
        context.update(batch_index=batch_index, global_step=15000 + batch_index + 1)
        x, target, mask = (batch[key].cuda() for key in ("linear_rgb", "target_mhsp", "valid_mask"))
        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda", enabled=True):
            prediction = model(x)
            losses = masked_smooth_l1_loss(prediction, target, mask, beta=0.1)
        forward_finite = bool(torch.isfinite(prediction).all())
        loss_finite = all(bool(torch.isfinite(value).all()) for value in losses.values())
        if not forward_finite:
            failure_stage = "FORWARD"
            break
        if not loss_finite:
            failure_stage = "LOSS"
            break
        try:
            anomaly_enabled = batch_index >= failing_index - 1
            with torch.autograd.set_detect_anomaly(anomaly_enabled):
                scaler.scale(losses["total"]).backward()
            scaler.unscale_(optimizer)
        except RuntimeError:
            failure_stage = "BACKWARD"
            anomaly_text = traceback.format_exc()
            break
        _, gradients = gradient_health(model)
        if not gradients["finite"]:
            failure_stage = "BACKWARD"
            break
        scaler.step(optimizer)
        scaler.update()
    for handle in handles:
        handle.remove()
    write_csv(OUTPUT_DIR / "fine_localization/layer_trace.csv", layer_rows)
    if anomaly_text:
        (OUTPUT_DIR / "fine_localization/autograd_anomaly_trace.txt").write_text(
            anomaly_text, encoding="utf-8"
        )
    first_layer = next((row for row in layer_rows if not row["finite"]), None)
    result = {
        "hook_start_batch": hook_start, "failing_batch": failing_index,
        "batches_with_fine_hooks": failing_index - hook_start + 1,
        "failure_stage": failure_stage,
        "first_nonfinite_module": first_layer["module"] if first_layer else None,
        "first_nonfinite_operation": first_layer["operation"] if first_layer else None,
        "first_nonfinite_tensor": first_layer["tensor"] if first_layer else None,
        "autograd_anomaly_trace_saved": anomaly_text is not None,
    }
    write_json(OUTPUT_DIR / "fine_localization/result.json", result)
    return result


def failing_batch_offline_audit() -> dict[str, Any]:
    snapshot = torch.load(OUTPUT_DIR / "failing_batch/failing_batch_snapshot.pt", map_location="cpu", weights_only=False)
    metadata = pd.DataFrame(snapshot["metadata"])
    data = {
        "linear_rgb": tensor_health(snapshot["linear_rgb"]),
        "target_mhsp": tensor_health(snapshot["target_mhsp"]),
        "valid_mask": tensor_health(snapshot["valid_mask"]),
        "valid_pixel_count": float(snapshot["valid_mask"].sum()),
        "mask_fraction": float(snapshot["valid_mask"].mean()),
        "sample_ids": snapshot["sample_ids"],
        "camera": snapshot["camera"], "light": snapshot["light"],
        "camera_light_pair": snapshot["camera_light_pair"],
        "low_clip_fraction": metadata["low_clip_fraction"].astype(float).tolist(),
        "high_clip_fraction": metadata["high_clip_fraction"].astype(float).tolist(),
        "retry_count": metadata["retry_count"].astype(int).tolist(),
    }
    full = pd.read_csv(DATA_ROOT / "train/metadata.csv", usecols=[
        "rgb_max", "h_std", "p_nonzero_fraction", "actual_valid_fraction"
    ])
    percentiles = {}
    mapping = {
        "rgb_max": float(snapshot["linear_rgb"].max()),
        "h_std": float(snapshot["target_mhsp"][:, 1].std()),
        "p_nonzero_fraction": float((snapshot["target_mhsp"][:, 3] > 0).float().mean()),
        "actual_valid_fraction": float(snapshot["valid_mask"].mean()),
    }
    for column, value in mapping.items():
        percentiles[column] = {"batch_value": value, "train_percentile": float((full[column] <= value).mean())}
    data["train_percentile_positions"] = percentiles
    first_inf_sample = metadata.loc[metadata["split_index"].astype(int) == 2356]
    if not first_inf_sample.empty:
        row = first_inf_sample.iloc[0]
        data["first_inf_sample"] = {
            "sample_id": str(row["sample_id"]), "split_index": int(row["split_index"]),
            "camera": str(row["camera_name"]), "light": str(row["light_name"]),
            "camera_light_pair": str(row["camera_light_pair"]),
        }
    write_json(OUTPUT_DIR / "failing_batch/failing_batch_data_audit.json", data)
    return data


def complete_failure_evidence() -> dict[str, Any]:
    snapshot = torch.load(
        OUTPUT_DIR / "failing_batch/failing_batch_snapshot.pt", map_location="cpu", weights_only=False
    )
    model, optimizer, _, scaler, anchor = load_anchor(
        OUTPUT_DIR / "anchors/latest_safe_anchor.pt", amp=True
    )
    model.train()
    dataset = SO1DecompositionDataset(DATA_ROOT, "train")
    order_rows = pd.read_csv(OUTPUT_DIR / "epoch7_sample_order.csv")
    failing_index = int(snapshot["event"]["batch_index"])
    captured: dict[str, torch.Tensor] = {}
    current_batch = {"value": -1}

    def capture_conv(_module: Any, _inputs: Any, output: torch.Tensor) -> None:
        if current_batch["value"] == failing_index:
            captured["up1_conv0_output"] = output.detach().cpu()

    handle = model.up1.conv.net[0].register_forward_hook(capture_conv)
    head_rows: list[dict[str, Any]] = []
    optimizer_pre_step: dict[str, Any] = {}
    for batch_index in range(int(anchor["batch_index"]), failing_index + 1):
        current_batch["value"] = batch_index
        indices = order_rows.loc[
            order_rows["batch_index"] == batch_index, "split_index"
        ].astype(int).tolist()
        batch = collate_fixed_batch(dataset, indices)
        x, target, mask = (batch[key].cuda() for key in ("linear_rgb", "target_mhsp", "valid_mask"))
        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda", enabled=True):
            prediction = model(x)
            losses = masked_smooth_l1_loss(prediction, target, mask, beta=0.1)
        if batch_index == failing_index:
            break
        scaler.scale(losses["total"]).backward()
        scaler.unscale_(optimizer)
        if batch_index == failing_index - 1:
            weight_grad = model.outc.weight.grad.detach().float()
            for channel_index, channel in enumerate(CHANNEL_NAMES):
                channel_grad = weight_grad[channel_index]
                head_rows.append({
                    "batch_index": batch_index, "global_step": 15000 + batch_index + 1,
                    "channel": channel, "grad_norm": float(channel_grad.norm().cpu()),
                    "grad_max_abs": float(channel_grad.abs().max().cpu()),
                    "finite": bool(torch.isfinite(channel_grad).all()),
                })
            optimizer_pre_step = {
                "batch_index": batch_index, "global_step": 15000 + batch_index + 1,
                **state_finite(model, optimizer),
                "gradients_finite": gradient_health(model)[1]["finite"],
                "scale": float(scaler.get_scale()),
            }
        scaler.step(optimizer)
        scaler.update()
    handle.remove()
    write_csv(OUTPUT_DIR / "replay_track_a_amp_original/output_head_channel_gradients.csv", head_rows)
    write_json(OUTPUT_DIR / "replay_track_a_amp_original/optimizer_pre_step_health.json", optimizer_pre_step)
    output = captured["up1_conv0_output"]
    inf_locations = torch.nonzero(torch.isinf(output), as_tuple=False).tolist()
    locations = []
    for sample_index, channel_index, y, x_coord in inf_locations:
        locations.append({
            "sample_position": sample_index,
            "sample_id": snapshot["sample_ids"][sample_index],
            "split_index": snapshot["split_indices"][sample_index],
            "output_channel": channel_index,
            "y": y,
            "x": x_coord,
            "value": str(output[sample_index, channel_index, y, x_coord].item()),
        })
    evidence = {
        "module": "up1.conv.net.0", "operation": "Conv2d",
        "tensor": "output", "dtype": str(output.dtype),
        "num_nan": int(torch.isnan(output).sum()), "num_inf": int(torch.isinf(output).sum()),
        "inf_locations": locations, "output_head_gradient_batch": failing_index - 1,
        "output_head_channel_gradients": head_rows,
        "optimizer_pre_step_health": optimizer_pre_step,
    }
    write_json(OUTPUT_DIR / "fine_localization/first_nonfinite_tensor_detail.json", evidence)
    health = tensor_health(output)
    for event_path in (
        OUTPUT_DIR / "first_failure_event.json",
        OUTPUT_DIR / "replay_track_a_amp_original/first_failure_event.json",
    ):
        event = json.loads(event_path.read_text(encoding="utf-8"))
        event.update({
            "coarse_failure_module": event.get("module"),
            "coarse_failure_tensor": event.get("tensor_name"),
            "module": "up1.conv.net.0", "tensor_name": "output",
            **health,
        })
        write_json(event_path, event)
    result_path = OUTPUT_DIR / "replay_track_a_amp_original/result.json"
    result = json.loads(result_path.read_text(encoding="utf-8"))
    result["first_failure"] = json.loads(
        (OUTPUT_DIR / "first_failure_event.json").read_text(encoding="utf-8")
    )
    write_json(result_path, result)
    return evidence


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for SO-1D-R1")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    set_determinism()
    if args.complete_evidence_only:
        evidence = complete_failure_evidence()
        final_path = OUTPUT_DIR / "final_status.json"
        final = json.loads(final_path.read_text(encoding="utf-8"))
        final["first_failure"] = json.loads(
            (OUTPUT_DIR / "first_failure_event.json").read_text(encoding="utf-8")
        )
        final["first_nonfinite_tensor_detail"] = evidence
        write_json(final_path, final)
        return
    audit = checkpoint_audit()
    if not all((audit["model_finite"], audit["batchnorm_finite"], audit["optimizer_finite"])):
        write_json(OUTPUT_DIR / "final_status.json", {"SO-1D-R1": "FAIL", "primary_diagnosis": "CHECKPOINT_ALREADY_CORRUPTED"})
        return
    model, _, _, _, checkpoint = build_runtime(LAST_PATH, amp=True)
    dataset = SO1DecompositionDataset(DATA_ROOT, "train")
    order, _, digest = build_order(dataset, checkpoint)
    del model
    torch.cuda.empty_cache()
    if args.audit_only:
        return
    track_a = replay_track_a(dataset, order, max_batches=args.max_batches)
    if track_a["status"] != "REPRODUCED":
        write_json(OUTPUT_DIR / "final_status.json", {
            "SO-1D-R1": "PARTIAL", "primary_diagnosis": "NON_REPRODUCIBLE_UNDER_REPLAY",
            "epoch7_order_sha256": digest, "allow_automatic_SO1D_resume": False,
        })
        return
    fine = fine_localization()
    track_b = controlled_track("replay_track_b_fp32", amp=False, backward=True, step=True)
    track_c = controlled_track("replay_track_c_amp_forward_only", amp=True, backward=False, step=False)
    track_d = controlled_track("replay_track_d_amp_no_optimizer", amp=True, backward=True, step=False)
    data_audit = failing_batch_offline_audit()
    evidence = complete_failure_evidence()
    event = track_a["first_failure"]
    if event["stage"] == "FORWARD" and track_b["status"] == "PASS":
        primary = "FORWARD_FP16_OVERFLOW"
    elif event.get("bn_contamination"):
        primary = "BATCHNORM_BUFFER_CONTAMINATION"
    elif event["stage"] == "BACKWARD" and track_b["status"] == "PASS":
        primary = "BACKWARD_FP16_OVERFLOW"
    elif event["stage"] == "BACKWARD":
        primary = "BACKWARD_NUMERICAL_FAILURE"
    elif event["stage"] == "OPTIMIZER_STEP":
        primary = "OPTIMIZER_UPDATE_FAILURE"
    elif event["stage"] == "LOSS":
        primary = "LOSS_NUMERICAL_FAILURE"
    elif event["stage"] == "DATA":
        primary = "DATA_PIPELINE_FAILURE"
    else:
        primary = "FORWARD_NUMERICAL_FAILURE"
    final = {
        "SO-1D-R1": "PASS", "track_a": track_a["status"],
        "reproduction_level": "EXACT", "epoch7_order_sha256": digest,
        "primary_diagnosis": primary,
        "secondary_diagnosis": (
            "BATCHNORM_BUFFER_CONTAMINATION_AFTER_NONFINITE_FORWARD"
            if event.get("bn_contamination") else None
        ),
        "contributing_condition": (
            "AMP_NUMERICAL_INSTABILITY" if track_b["status"] == "PASS" else None
        ),
        "first_failure": event, "fine_localization": fine,
        "first_nonfinite_tensor_detail": evidence,
        "track_b": track_b, "track_c": track_c, "track_d": track_d,
        "allow_automatic_SO1D_resume": False, "allow_SO1E": False,
        "candidate_minimal_fix": (
            "For a separately authorized repair protocol, keep the frozen model/data/loss/optimizer "
            "and run the numerically vulnerable decoder convolution path in float32 while preserving AMP elsewhere; "
            "re-validate BatchNorm buffers and resume semantics before any formal continuation."
        ),
    }
    write_json(OUTPUT_DIR / "final_status.json", final)
    write_report(audit, final, data_audit)


def write_report(audit: dict[str, Any], final: dict[str, Any], data_audit: dict[str, Any]) -> None:
    event = final["first_failure"]
    lines = [
        "# SO-1D-R1 Epoch-7 Numerical Failure Report", "",
        "## Background", "",
        "Formal SO-1D stopped at epoch 7 because all epoch-level losses and validation metrics became non-finite. This R1 task reproduced and localized the failure without changing the formal protocol.", "",
        "## Checkpoint Health", "",
        f"- Epoch 6 model finite: {audit['model_finite']}",
        f"- Optimizer finite: {audit['optimizer_finite']}",
        f"- BatchNorm finite: {audit['batchnorm_finite']}",
        f"- GradScaler scale/tracker: {audit['gradscaler_scale']} / {audit['gradscaler_growth_tracker']}",
        f"- Epoch5 to epoch6 >10x state growth detected: {audit['abnormal_growth_gt_10x']}", "",
        "## Reproduction", "",
        f"- Reproduction level: {final['reproduction_level']}",
        f"- Epoch 7 order SHA256: `{final['epoch7_order_sha256']}`",
        f"- Track A: {final['track_a']}", "",
        "## First Failure", "",
        f"- Batch index: {event['batch_index']}",
        f"- Global step: {event['global_step']}",
        f"- Sample IDs: {event['sample_ids']}",
        f"- Stage/module/tensor: {event['stage']} / {event.get('module')} / {event.get('tensor_name')}",
        f"- Dtype: {event.get('dtype', event.get('tensor_dtype'))}",
        f"- NaN / Inf: {event.get('num_nan')} / {event.get('num_inf')}",
        f"- BN contamination: {event.get('bn_contamination')}",
        f"- Scale before/after: {event.get('scale_before')} / {event.get('scale_after')}", "",
        "## Controlled Replay", "",
        f"- Track B AMP=false: {final['track_b']['status']}",
        f"- Track C AMP forward-only: {final['track_c']['status']}",
        f"- Track D AMP backward-no-step: {final['track_d']['status']}", "",
        "## Failing Batch Data", "",
        f"- RGB finite/range: {data_audit['linear_rgb']['finite']} / {data_audit['linear_rgb']['min']} to {data_audit['linear_rgb']['max']}",
        f"- Target finite/range: {data_audit['target_mhsp']['finite']} / {data_audit['target_mhsp']['min']} to {data_audit['target_mhsp']['max']}",
        f"- Mask fraction: {data_audit['mask_fraction']}", "",
        "## Diagnosis", "",
        f"- Primary: **{final['primary_diagnosis']}**",
        f"- Secondary: {final['secondary_diagnosis']}",
        f"- Contributing condition: {final['contributing_condition']}",
        f"- Candidate minimal fix: {final['candidate_minimal_fix']}", "",
        "## Boundaries", "",
        "No formal resume, hyperparameter change, gradient clipping, normalization change, SO-1E evaluation, or new frozen checkpoint was performed.", "",
        "- Allow automatic SO-1D resume: NO",
        "- Allow SO-1E: NO", "",
    ]
    (OUTPUT_DIR / "SO1D_R1_epoch7_numerical_failure_report.md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        (OUTPUT_DIR / "diagnostic_exception.txt").write_text(traceback.format_exc(), encoding="utf-8")
        raise
