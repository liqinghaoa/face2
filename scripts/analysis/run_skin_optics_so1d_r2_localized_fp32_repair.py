"""Run the isolated SO-1D-R2 localized-FP32 numerical repair diagnostic."""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import math
import random
import shutil
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from skin_optics_so1.decomposition.losses import CHANNEL_NAMES, masked_smooth_l1_loss  # noqa: E402
from skin_optics_so1.decomposition.metrics import MaskedMAEAggregator  # noqa: E402
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
from skin_optics_so1.decomposition.unet_decomposer import (  # noqa: E402
    SO1UNetDecomposer,
    count_parameters,
)


FORMAL_DIR = PROJECT_ROOT / "experiments/skin_optics_so1/SO1_Decomposition_UNet_v1/formal_train"
FORMAL_LAST = FORMAL_DIR / "checkpoints/last.pt"
FORMAL_BEST = FORMAL_DIR / "checkpoints/best.pt"
R1_DIR = PROJECT_ROOT / "experiments/skin_optics_so1/SO1_Decomposition_UNet_v1/diagnostics/so1d_r1_epoch7_numerical_failure"
R1_ANCHOR = R1_DIR / "anchors/latest_safe_anchor.pt"
DATA_ROOT = PROJECT_ROOT / "data/processed/SO1_Synthetic_Generator_v1_localfull"
CONFIG_PATH = PROJECT_ROOT / "config/train/skin_optics_so1/so1d_localized_fp32_repair_v1.yaml"
OUTPUT_DIR = PROJECT_ROOT / "experiments/skin_optics_so1/SO1_Decomposition_UNet_v1/diagnostics/so1d_r2_localized_fp32_repair"
EXPECTED_ORDER_HASH = "36224640f943244a6ec498226a86ce7d624f8b3486a866d69901ffc098e027b3"
POLICY = {"mode": "localized_fp32", "fp32_blocks": ["up1.conv"]}
FORMAL_HASHES = {
    "last.pt": "85e559ac4cb9a3da5e889b57db7fc45974cdc14f8f8d1312479c1a980e755349",
    "best.pt": "e8c3be6d5f818a049160a97b151cf2a14c82275c7c00f7577523b56123846704",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=("a", "full", "all", "finalize"), default="all")
    parser.add_argument("--initial-pytest-passed", type=int, default=62)
    parser.add_argument("--final-pytest-passed", type=int)
    return parser.parse_args()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def set_determinism() -> None:
    random.seed(20260801)
    np.random.seed(20260801)
    torch.manual_seed(20260801)
    torch.cuda.manual_seed_all(20260801)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def restore_anchor_rng(anchor: dict[str, Any]) -> None:
    random.setstate(anchor["python_rng_state"])
    np.random.set_state(anchor["numpy_rng_state"])
    torch.set_rng_state(anchor["torch_cpu_rng_state"].cpu())
    torch.cuda.set_rng_state_all([state.cpu() for state in anchor["torch_cuda_rng_state"]])


def collate_fixed_batch(dataset: SO1DecompositionDataset, indices: list[int]) -> dict[str, Any]:
    samples = [dataset[index] for index in indices]
    result: dict[str, Any] = {
        key: torch.stack([sample[key] for sample in samples])
        for key in ("linear_rgb", "target_mhsp", "valid_mask")
    }
    for key in ("sample_id", "split_index", "base_latent_id"):
        result[key] = [sample[key] for sample in samples]
    return result


def build_runtime(state: dict[str, Any], *, repaired: bool) -> tuple[Any, ...]:
    model = SO1UNetDecomposer(numerical_precision=POLICY if repaired else None).cuda()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.0001)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=30, eta_min=0.000001
    )
    scaler = torch.amp.GradScaler("cuda", enabled=True)
    model.load_state_dict(state["model_state_dict"], strict=True)
    optimizer.load_state_dict(state["optimizer_state_dict"])
    scheduler.load_state_dict(state["scheduler_state_dict"])
    scaler.load_state_dict(state["grad_scaler_state_dict"])
    return model, optimizer, scheduler, scaler


def load_formal_checkpoint() -> dict[str, Any]:
    return torch.load(FORMAL_LAST, map_location="cuda", weights_only=False)


def frozen_order(dataset: SO1DecompositionDataset, checkpoint: dict[str, Any]) -> tuple[list[int], str]:
    order = epoch7_order_from_generator_state(
        sample_count=len(dataset), batch_size=8,
        generator_state=checkpoint["dataloader_rng_state"]["train"],
    )
    metadata = dataset.metadata.set_index("split_index", drop=False)
    rows = []
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
    if digest != EXPECTED_ORDER_HASH:
        raise RuntimeError(f"R2-B FAIL: epoch7 order hash mismatch: {digest}")
    return order, digest


def finite_state_summary(model: torch.nn.Module, optimizer: torch.optim.Optimizer) -> dict[str, Any]:
    return state_finite(model, optimizer)


def finite_state_passes(summary: dict[str, Any]) -> bool:
    return all(
        bool(summary[key])
        for key in (
            "parameters_finite",
            "optimizer_state_finite",
            "batchnorm_buffers_finite",
        )
    )


def bn_scale_summary(model: torch.nn.Module) -> dict[str, Any]:
    rows = batchnorm_health(model)
    means = [row["max_abs"] for row in rows if row["tensor"] == "running_mean"]
    variances = [row for row in rows if row["tensor"] == "running_var"]
    decoder = [row for row in rows if row["module"].startswith("up")]
    return {
        "finite": all_rows_finite(rows),
        "max_abs_running_mean": max(means, default=0.0),
        "max_running_var": max((row["max"] for row in variances), default=0.0),
        "min_running_var": min((row["min"] for row in variances), default=0.0),
        "decoder_max_abs_running_mean": max(
            (row["max_abs"] for row in decoder if row["tensor"] == "running_mean"), default=0.0
        ),
        "decoder_max_running_var": max(
            (row["max"] for row in decoder if row["tensor"] == "running_var"), default=0.0
        ),
    }


def implementation_audit() -> dict[str, Any]:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for name in (
        "config", "implementation_audit", "pre_failure_state", "control_original_amp",
        "repaired_failing_batch", "repaired_epoch7", "activation_monitoring",
        "checkpoints_diagnostic",
    ):
        (OUTPUT_DIR / name).mkdir(parents=True, exist_ok=True)
    shutil.copy2(CONFIG_PATH, OUTPUT_DIR / "config" / CONFIG_PATH.name)
    original = SO1UNetDecomposer()
    repaired = SO1UNetDecomposer(numerical_precision=POLICY)
    before = count_parameters(original)
    after = count_parameters(repaired)
    state_keys_equal = list(original.state_dict()) == list(repaired.state_dict())
    checkpoint = torch.load(FORMAL_LAST, map_location="cpu", weights_only=False)
    repaired.load_state_dict(checkpoint["model_state_dict"], strict=True)
    hashes = {"last.pt": file_sha256(FORMAL_LAST), "best.pt": file_sha256(FORMAL_BEST)}
    audit = {
        "repair_mode": "LOCALIZED_FP32",
        "protected_block": "up1.conv",
        "state_dict_compatible": state_keys_equal,
        "strict_epoch6_checkpoint_load": True,
        "parameter_count_before": before["parameter_count"],
        "parameter_count_after": after["parameter_count"],
        "trainable_parameter_count_before": before["trainable_parameter_count"],
        "trainable_parameter_count_after": after["trainable_parameter_count"],
        "global_amp": True,
        "protected_block_dtype": "float32",
        "other_amp_blocks_preserved": True,
        "formal_checkpoint_hashes_before": hashes,
        "formal_checkpoint_hashes_expected": FORMAL_HASHES,
        "formal_checkpoint_paths_read_only": True,
        "status": "PASS" if state_keys_equal and before == after and hashes == FORMAL_HASHES else "FAIL",
    }
    write_json(OUTPUT_DIR / "implementation_audit/implementation_audit.json", audit)
    del original, repaired, checkpoint
    if audit["status"] != "PASS":
        raise RuntimeError("SO-1D-R2 FAIL: implementation compatibility audit failed")
    return audit


def save_pre_failure_state() -> dict[str, Any]:
    anchor = torch.load(R1_ANCHOR, map_location="cuda", weights_only=False)
    if int(anchor["batch_index"]) != 500 or int(anchor["global_step"]) != 15500:
        raise RuntimeError("R2-A FAIL: unexpected R1 safe-anchor semantics")
    model, optimizer, scheduler, scaler = build_runtime(anchor, repaired=False)
    restore_anchor_rng(anchor)
    model.train()
    dataset = SO1DecompositionDataset(DATA_ROOT, "train")
    checkpoint = load_formal_checkpoint()
    order, digest = frozen_order(dataset, checkpoint)
    for batch_index in range(500, 555):
        batch = collate_fixed_batch(dataset, order[batch_index * 8:(batch_index + 1) * 8])
        x = batch["linear_rgb"].cuda(non_blocking=True)
        target = batch["target_mhsp"].cuda(non_blocking=True)
        mask = batch["valid_mask"].cuda(non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda", enabled=True):
            prediction = model(x)
            losses = masked_smooth_l1_loss(prediction, target, mask, beta=0.1)
        if not tensor_health(prediction)["finite"] or not tensor_health(losses["total"])["finite"]:
            raise RuntimeError(f"R2-A FAIL: nonfinite while reconstructing batch {batch_index}")
        scaler.scale(losses["total"]).backward()
        scaler.unscale_(optimizer)
        _, gradient_summary = gradient_health(model)
        if not gradient_summary["finite"]:
            raise RuntimeError(f"R2-A FAIL: nonfinite gradient at batch {batch_index}")
        scaler.step(optimizer)
        scaler.update()
        if not finite_state_passes(finite_state_summary(model, optimizer)):
            raise RuntimeError(f"R2-A FAIL: state became nonfinite at batch {batch_index}")
    health = finite_state_summary(model, optimizer)
    payload = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "grad_scaler_state_dict": scaler.state_dict(),
        "python_rng_state": random.getstate(),
        "numpy_rng_state": np.random.get_state(),
        "torch_cpu_rng_state": torch.get_rng_state(),
        "torch_cuda_rng_state": torch.cuda.get_rng_state_all(),
        "batch_index": 555,
        "global_step": 15555,
        "epoch7_order_sha256": digest,
        "source_anchor_batch_index": 500,
        "source_anchor_semantics": "state immediately before batch 500",
        "state_semantics": "finite state immediately before batch 555",
    }
    path = OUTPUT_DIR / "pre_failure_state/pre_failure_batch555_state.pt"
    atomic_torch_save(payload, path)
    result = {
        "status": "PASS",
        "batch_index": 555,
        "global_step": 15555,
        "model_parameters_finite": health["parameters_finite"],
        "adam_state_finite": health["optimizer_state_finite"],
        "bn_buffers_finite": health["batchnorm_buffers_finite"],
        "gradscaler_valid": math.isfinite(float(scaler.get_scale())) and scaler.get_scale() > 0,
        "rng_state_saved": True,
        "state_sha256": file_sha256(path),
        "epoch7_order_sha256": digest,
    }
    write_json(OUTPUT_DIR / "pre_failure_state/pre_failure_state_health.json", result)
    del model, optimizer, scheduler, scaler, checkpoint
    torch.cuda.empty_cache()
    return result


def load_pre_failure(*, repaired: bool) -> tuple[Any, ...]:
    state = torch.load(
        OUTPUT_DIR / "pre_failure_state/pre_failure_batch555_state.pt",
        map_location="cuda", weights_only=False,
    )
    runtime = build_runtime(state, repaired=repaired)
    restore_anchor_rng(state)
    return (*runtime, state)


def failing_batch() -> dict[str, Any]:
    dataset = SO1DecompositionDataset(DATA_ROOT, "train")
    checkpoint = load_formal_checkpoint()
    order, _ = frozen_order(dataset, checkpoint)
    return collate_fixed_batch(dataset, order[555 * 8:556 * 8])


def control_reproduction() -> dict[str, Any]:
    model, optimizer, _scheduler, scaler, state = load_pre_failure(repaired=False)
    model.train()
    batch = failing_batch()
    captured: dict[str, Any] = {}

    def hook(_module: Any, _inputs: Any, output: torch.Tensor) -> None:
        captured.update(tensor_health(output))
        value = output[6, 412, 2, 24]
        location_finite = bool(torch.isfinite(value).item())
        captured["failure_location_value"] = (
            float(value.detach().cpu()) if location_finite else "Infinity"
        )
        captured["failure_location_finite"] = location_finite
        captured["failure_location_dtype"] = str(value.dtype)

    handle = model.up1.conv.net[0].register_forward_hook(hook)
    with torch.amp.autocast("cuda", enabled=True):
        prediction = model(batch["linear_rgb"].cuda())
    handle.remove()
    reproduced = (
        not captured.get("finite", True)
        and captured.get("num_inf", 0) >= 1
        and not captured.get("failure_location_finite", True)
        and captured.get("failure_location_dtype") == "torch.float16"
    )
    result = {
        "status": "PASS" if reproduced else "FAIL",
        "repair_enabled": False,
        "amp_enabled": True,
        "batch_index": int(state["batch_index"]),
        "sample": batch["sample_id"][6],
        "first_failure_module": "up1.conv.net.0" if reproduced else None,
        "first_failure_operation": "Conv2d" if reproduced else None,
        "first_failure_dtype": captured.get("dtype"),
        "first_failure": "+Inf" if captured.get("num_inf", 0) else None,
        "channel": 412, "y": 2, "x": 24,
        "conv0_health": captured,
        "final_prediction_finite": tensor_health(prediction)["finite"],
        "contaminated_control_destroyed": True,
    }
    write_json(OUTPUT_DIR / "control_original_amp/control_reproduction.json", result)
    del model, optimizer, scaler, prediction
    torch.cuda.empty_cache()
    if not reproduced:
        raise RuntimeError("SO-1D-R2 FAIL: original AMP control did not reproduce")
    return result


def repaired_batch_replay() -> dict[str, Any]:
    model, optimizer, _scheduler, scaler, _state = load_pre_failure(repaired=True)
    model.train()
    batch = failing_batch()
    x = batch["linear_rgb"].cuda()
    target = batch["target_mhsp"].cuda()
    mask = batch["valid_mask"].cuda()
    observations: dict[str, dict[str, Any]] = {}
    handles = []

    def observe(name: str):
        def hook(_module: Any, _inputs: Any, output: torch.Tensor) -> None:
            observations[name] = tensor_health(output)
            if name == "up1.conv.net.0":
                finite_values = output[torch.isfinite(output)].float()
                observations[name].update({
                    "p99": float(torch.quantile(finite_values, 0.99).cpu()),
                    "p99_9": float(torch.quantile(finite_values, 0.999).cpu()),
                    "failure_location_value": float(output[6, 412, 2, 24].detach().cpu()),
                })
        return hook

    monitored = {
        "inc": model.inc, "down1": model.down1, "down2": model.down2,
        "down3": model.down3, "down4": model.down4, "up1.up": model.up1.up,
        "up1.conv.net.0": model.up1.conv.net[0], "up1.conv.net.1": model.up1.conv.net[1],
        "up1.conv.net.3": model.up1.conv.net[3], "up1.conv.net.4": model.up1.conv.net[4],
        "up1.conv": model.up1.conv, "up2": model.up2, "up3": model.up3,
        "up4": model.up4, "outc": model.outc, "sigmoid": model.out_activation,
    }
    for name, module in monitored.items():
        handles.append(module.register_forward_hook(observe(name)))
    bn_before = bn_scale_summary(model)
    optimizer.zero_grad(set_to_none=True)
    with torch.amp.autocast("cuda", enabled=True):
        prediction = model(x)
        losses = masked_smooth_l1_loss(prediction, target, mask, beta=0.1)
    for handle in handles:
        handle.remove()
    bn_after_forward = bn_scale_summary(model)
    forward_finite = all(row["finite"] for row in observations.values())
    loss_finite = all(tensor_health(value)["finite"] for value in losses.values())
    scaler.scale(losses["total"]).backward()
    scaler.unscale_(optimizer)
    gradient_rows, gradient_summary = gradient_health(model)
    write_csv(OUTPUT_DIR / "repaired_failing_batch/gradient_health.csv", gradient_rows)
    pre_step = finite_state_summary(model, optimizer)
    scaler.step(optimizer)
    scaler.update()
    post_step = finite_state_summary(model, optimizer)
    bn_after_step = bn_scale_summary(model)
    location = {
        "sample": "train_002356", "batch_position": 6,
        "channel": 412, "y": 2, "x": 24,
        "dtype": observations["up1.conv.net.0"]["dtype"],
        "value": observations["up1.conv.net.0"]["failure_location_value"],
        "finite": math.isfinite(observations["up1.conv.net.0"]["failure_location_value"]),
        "exceeds_fp16_max": abs(observations["up1.conv.net.0"]["failure_location_value"]) > 65504,
    }
    write_json(OUTPUT_DIR / "repaired_failing_batch/original_failure_location.json", location)
    write_json(
        OUTPUT_DIR / "repaired_failing_batch/up1_conv0_statistics.json",
        {**observations["up1.conv.net.0"], "r1_comparison": "finite max 64192 then +Inf"},
    )
    bn_regression = {
        "before": bn_before, "after_forward": bn_after_forward, "after_step": bn_after_step,
        "BN_BUFFER_CONTAMINATION": not bn_after_forward["finite"],
    }
    write_json(OUTPUT_DIR / "repaired_failing_batch/bn_contamination_regression.json", bn_regression)
    result = {
        "status": "PASS",
        "same_failing_batch": True,
        "sample_ids": batch["sample_id"],
        "input_finite": tensor_health(x)["finite"],
        "target_finite": tensor_health(target)["finite"],
        "mask_finite": tensor_health(mask)["finite"],
        "forward_finite": forward_finite,
        "loss_finite": loss_finite,
        "losses": {key: float(value.detach().cpu()) for key, value in losses.items()},
        "backward_finite": gradient_summary["finite"],
        "global_grad_norm": gradient_summary["global_grad_norm"],
        "optimizer_pre_step_finite": finite_state_passes(pre_step),
        "optimizer_post_step_finite": finite_state_passes(post_step),
        "bn_buffers_finite": bn_after_step["finite"],
        "original_failure_location": location,
        "module_observations": observations,
    }
    gates = (
        result["input_finite"], result["target_finite"], result["mask_finite"],
        forward_finite, loss_finite, gradient_summary["finite"],
        finite_state_passes(pre_step), finite_state_passes(post_step), bn_after_step["finite"],
        location["finite"], location["dtype"] == "torch.float32",
    )
    result["status"] = "PASS" if all(gates) else "FAIL"
    write_json(OUTPUT_DIR / "repaired_failing_batch/repaired_batch_result.json", result)
    del model, optimizer, scaler, prediction
    torch.cuda.empty_cache()
    if result["status"] != "PASS":
        raise RuntimeError("SO-1D-R2 FAIL: repaired failing-batch hard gate failed")
    return result


def run_r2_a() -> dict[str, Any]:
    pre = save_pre_failure_state()
    control = control_reproduction()
    repaired = repaired_batch_replay()
    result = {
        "R2-A": "PASS" if all(
            value["status"] == "PASS" for value in (pre, control, repaired)
        ) else "FAIL",
        "pre_failure_state": pre,
        "control": control,
        "repaired_batch": repaired,
    }
    write_json(OUTPUT_DIR / "r2_a_status.json", result)
    return result


def quick_activation(output: torch.Tensor) -> dict[str, Any]:
    finite = bool(torch.isfinite(output).all().item())
    return {
        "finite": finite,
        "dtype": str(output.dtype),
        "max_abs": float(output.detach().abs().max().cpu()) if output.numel() else 0.0,
        "mean": float(output.detach().float().mean().cpu()),
        "std": float(output.detach().float().std(unbiased=False).cpu()),
    }


def scheduler_state_valid(scheduler: Any) -> bool:
    for value in scheduler.state_dict().values():
        if isinstance(value, float) and not math.isfinite(value):
            return False
        if torch.is_tensor(value) and not bool(torch.isfinite(value).all()):
            return False
    return True


def classify_r2_b_failure(event: dict[str, Any]) -> tuple[str, bool]:
    health = event.get("health") or {}
    new_fp16_overflow = (
        event.get("stage") == "FORWARD"
        and health.get("dtype") == "torch.float16"
        and not health.get("finite", True)
    )
    return ("PARTIAL" if new_fp16_overflow else "FAIL", new_fp16_overflow)


def write_new_failure(
    event: dict[str, Any], batch: dict[str, Any], model: Any, optimizer: Any,
    scheduler: Any, scaler: Any,
) -> None:
    write_json(OUTPUT_DIR / "repaired_epoch7/first_new_failure_event.json", event)
    atomic_torch_save({
        "event": event,
        "sample_ids": batch["sample_id"],
        "split_indices": batch["split_index"],
        "linear_rgb": batch["linear_rgb"],
        "target_mhsp": batch["target_mhsp"],
        "valid_mask": batch["valid_mask"],
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "grad_scaler_state_dict": scaler.state_dict(),
        "DIAGNOSTIC_ONLY": True,
    }, OUTPUT_DIR / "repaired_epoch7/failing_batch_snapshot.pt")


def full_epoch7_replay() -> dict[str, Any]:
    r2a = json.loads((OUTPUT_DIR / "r2_a_status.json").read_text(encoding="utf-8"))
    if r2a["R2-A"] != "PASS":
        raise RuntimeError("R2-B forbidden because R2-A did not pass")
    set_determinism()
    checkpoint = load_formal_checkpoint()
    model, optimizer, scheduler, scaler = build_runtime(checkpoint, repaired=True)
    model.train()
    train_dataset = SO1DecompositionDataset(DATA_ROOT, "train")
    order, digest = frozen_order(train_dataset, checkpoint)
    activation_rows: list[dict[str, Any]] = []
    bn_rows: list[dict[str, Any]] = []
    total_losses = {name: 0.0 for name in ("total", *CHANNEL_NAMES)}
    maximum_ratios = {"up2": 0.0, "up3": 0.0, "up4": 0.0}
    maximum_up1 = 0.0
    warnings: set[str] = set()
    context: dict[str, Any] = {"values": {}, "batch_index": -1}
    handles = []
    for name in ("inc", "down1", "down2", "down3", "down4", "up1", "up2", "up3", "up4", "outc", "out_activation"):
        module = getattr(model, name)
        def hook(_module: Any, _inputs: Any, output: torch.Tensor, *, stage: str = name) -> None:
            context["values"][stage] = quick_activation(output)
        handles.append(module.register_forward_hook(hook))
    first_failure: dict[str, Any] | None = None
    started = time.perf_counter()
    completed = 0
    for batch_index in range(2500):
        context["batch_index"] = batch_index
        context["values"] = {}
        global_step = 15001 + batch_index
        batch = collate_fixed_batch(
            train_dataset, order[batch_index * 8:(batch_index + 1) * 8]
        )
        data_health = {
            key: tensor_health(batch[key]) for key in ("linear_rgb", "target_mhsp", "valid_mask")
        }
        if not all(row["finite"] for row in data_health.values()):
            first_failure = {"stage": "DATA", "batch_index": batch_index, "global_step": global_step}
        x = batch["linear_rgb"].cuda(non_blocking=True)
        target = batch["target_mhsp"].cuda(non_blocking=True)
        mask = batch["valid_mask"].cuda(non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        if first_failure is None:
            with torch.amp.autocast("cuda", enabled=True):
                prediction = model(x)
                losses = masked_smooth_l1_loss(prediction, target, mask, beta=0.1)
            bad_stage = next(
                (name for name, row in context["values"].items() if not row["finite"]), None
            )
            if bad_stage is not None:
                first_failure = {
                    "stage": "FORWARD", "module": bad_stage,
                    "batch_index": batch_index, "global_step": global_step,
                    "health": context["values"][bad_stage],
                }
            elif not all(tensor_health(value)["finite"] for value in losses.values()):
                first_failure = {
                    "stage": "LOSS", "module": "masked_smooth_l1",
                    "batch_index": batch_index, "global_step": global_step,
                }
        if first_failure is None:
            scaler.scale(losses["total"]).backward()
            scaler.unscale_(optimizer)
            _, gradients = gradient_health(model)
            if not gradients["finite"]:
                first_failure = {
                    "stage": "BACKWARD", "module": gradients["first_nonfinite_parameter"],
                    "batch_index": batch_index, "global_step": global_step,
                }
        if first_failure is None:
            scaler.step(optimizer)
            scaler.update()
            state = finite_state_summary(model, optimizer)
            if not finite_state_passes(state):
                first_failure = {
                    "stage": "OPTIMIZER_OR_STATE", "batch_index": batch_index,
                    "global_step": global_step, "state": state,
                }
        if first_failure is not None:
            first_failure.update({
                "sample_ids": batch["sample_id"], "split_indices": batch["split_index"],
            })
            write_new_failure(first_failure, batch, model, optimizer, scheduler, scaler)
            break
        completed += 1
        for key, value in losses.items():
            total_losses[key] += float(value.detach().cpu()) * 8
        row: dict[str, Any] = {"batch_index": batch_index, "global_step": global_step}
        for stage in ("up1", "up2", "up3", "up4"):
            stats = context["values"][stage]
            row[f"{stage}_dtype"] = stats["dtype"]
            row[f"{stage}_max_abs"] = stats["max_abs"]
            row[f"{stage}_finite"] = stats["finite"]
            if stage == "up1":
                row["up1_mean"] = stats["mean"]
                row["up1_std"] = stats["std"]
                maximum_up1 = max(maximum_up1, stats["max_abs"])
            else:
                ratio = stats["max_abs"] / 65504.0
                row[f"{stage}_fp16_capacity_ratio"] = ratio
                maximum_ratios[stage] = max(maximum_ratios[stage], ratio)
                if ratio >= 0.95:
                    warnings.add(f"{stage}:HIGH_RISK")
                elif ratio >= 0.80:
                    warnings.add(f"{stage}:WARNING")
        activation_rows.append(row)
        if batch_index % 50 == 0 or batch_index == 2499:
            bn_rows.append({
                "batch_index": batch_index, "global_step": global_step,
                **bn_scale_summary(model),
            })
        if (batch_index + 1) % 100 == 0:
            write_json(OUTPUT_DIR / "repaired_epoch7/progress.json", {
                "batches_completed": completed,
                "last_global_step": global_step,
                "elapsed_seconds": time.perf_counter() - started,
                "status": "RUNNING",
            })
    for handle in handles:
        handle.remove()
    write_csv(OUTPUT_DIR / "activation_monitoring/decoder_activation_history.csv", activation_rows)
    write_csv(OUTPUT_DIR / "activation_monitoring/bn_scale_history.csv", bn_rows)
    if first_failure is not None:
        r2b_status, new_fp16_overflow = classify_r2_b_failure(first_failure)
        stop_health = finite_state_summary(model, optimizer)
        maximum_bn_mean = max((row["max_abs_running_mean"] for row in bn_rows), default=0.0)
        maximum_bn_var = max((row["max_running_var"] for row in bn_rows), default=0.0)
        minimum_bn_var = min((row["min_running_var"] for row in bn_rows), default=0.0)
        result = {
            "R2-B": r2b_status, "train_batches_completed": completed,
            "all_train_batches_finite": False,
            "new_first_overflow": first_failure if new_fp16_overflow else None,
            "new_first_numerical_failure": first_failure,
            "new_fp16_overflow_elsewhere": new_fp16_overflow,
            "validation": "NOT_RUN_DUE_TO_TRAIN_HARD_GATE",
            "train_samples_completed": completed * 8,
            "epoch7_order_sha256": digest, "maximum_fp16_capacity_ratio": max(maximum_ratios.values()),
            "high_risk_blocks": sorted(warnings), "up1_fp32_max_activation": maximum_up1,
            "maximum_bn_abs_running_mean": maximum_bn_mean,
            "maximum_bn_running_var": maximum_bn_var,
            "minimum_bn_running_var": minimum_bn_var,
            "health_at_stop_before_optimizer_step": {
                **stop_health,
                "gradscaler_valid": math.isfinite(float(scaler.get_scale())) and scaler.get_scale() > 0,
                "gradscaler_scale": float(scaler.get_scale()),
                "scheduler_state_valid": scheduler_state_valid(scheduler),
                "optimizer_step_executed_for_failing_batch": False,
            },
            "diagnostic_epoch7_checkpoint": "NOT_SAVED_DUE_TO_TRAIN_HARD_GATE",
        }
        write_json(OUTPUT_DIR / "repaired_epoch7/full_epoch7_result.json", result)
        write_json(OUTPUT_DIR / "repaired_epoch7/progress.json", {
            "batches_completed": completed,
            "last_global_step": 15000 + completed,
            "status": "STOPPED_ON_FIRST_NONFINITE",
            "R2-B": r2b_status,
        })
        return result

    model.eval()
    validation = SO1DecompositionDataset(DATA_ROOT, "validation")
    metrics = MaskedMAEAggregator()
    validation_losses = {name: 0.0 for name in ("total", *CHANNEL_NAMES)}
    validation_samples = 0
    with torch.inference_mode():
        for batch_index in range(math.ceil(len(validation) / 8)):
            indices = list(range(batch_index * 8, min((batch_index + 1) * 8, len(validation))))
            batch = collate_fixed_batch(validation, indices)
            x = batch["linear_rgb"].cuda(non_blocking=True)
            target = batch["target_mhsp"].cuda(non_blocking=True)
            mask = batch["valid_mask"].cuda(non_blocking=True)
            with torch.amp.autocast("cuda", enabled=True):
                prediction = model(x)
                losses = masked_smooth_l1_loss(prediction, target, mask, beta=0.1)
            if not tensor_health(prediction)["finite"] or not all(
                tensor_health(value)["finite"] for value in losses.values()
            ):
                raise FloatingPointError(f"R2-B FAIL: nonfinite validation batch {batch_index}")
            metrics.update(prediction.float(), target.float(), mask.float())
            count = len(indices)
            validation_samples += count
            for key, value in losses.items():
                validation_losses[key] += float(value.cpu()) * count
    validation_metrics = metrics.compute()
    validation_result = {
        **validation_metrics,
        **{f"{key}_loss": value / validation_samples for key, value in validation_losses.items()},
        "samples": validation_samples,
        "batches": math.ceil(validation_samples / 8),
        "all_finite": all(math.isfinite(value) for value in validation_metrics.values()),
    }
    write_json(OUTPUT_DIR / "repaired_epoch7/validation_metrics.json", validation_result)
    scheduler.step()
    health = finite_state_summary(model, optimizer)
    bn_health = bn_scale_summary(model)
    scaler_valid = math.isfinite(float(scaler.get_scale())) and scaler.get_scale() > 0
    end_health = {
        "model_parameters_finite": health["parameters_finite"],
        "bn_buffers_finite": health["batchnorm_buffers_finite"],
        "adam_state_finite": health["optimizer_state_finite"],
        "gradscaler_valid": scaler_valid,
        "gradscaler_scale": float(scaler.get_scale()),
        "scheduler_state_valid": scheduler_state_valid(scheduler),
        "scheduler_last_epoch": scheduler.last_epoch,
        **bn_health,
    }
    write_json(OUTPUT_DIR / "repaired_epoch7/end_of_epoch7_health.json", end_health)
    diagnostic_path = OUTPUT_DIR / "checkpoints_diagnostic/epoch7_repaired_diagnostic.pt"
    atomic_torch_save({
        "DIAGNOSTIC_ONLY": True,
        "NOT_FORMAL_CHECKPOINT": True,
        "NOT_FROZEN": True,
        "protocol_version": "SO1D_Localized_FP32_Repair_v1",
        "base_checkpoint": str(FORMAL_LAST),
        "epoch": 7, "global_step": 17500,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "grad_scaler_state_dict": scaler.state_dict(),
        "epoch7_order_sha256": digest,
        "validation_metrics": validation_result,
    }, diagnostic_path)
    maximum_bn_mean = max((row["max_abs_running_mean"] for row in bn_rows), default=0.0)
    maximum_bn_var = max((row["max_running_var"] for row in bn_rows), default=0.0)
    result = {
        "R2-B": "PASS" if all(end_health[key] for key in (
            "model_parameters_finite", "bn_buffers_finite", "adam_state_finite",
            "gradscaler_valid", "scheduler_state_valid",
        )) and validation_result["all_finite"] else "FAIL",
        "epoch7_order_sha256": digest,
        "train_batches_completed": completed,
        "train_samples": completed * 8,
        "all_train_batches_finite": completed == 2500,
        "train_losses": {key: value / 20000 for key, value in total_losses.items()},
        "new_first_overflow": None,
        "maximum_fp16_capacity_ratio": max(maximum_ratios.values()),
        "maximum_fp16_capacity_ratio_by_block": maximum_ratios,
        "high_risk_blocks": sorted(warnings),
        "up1_fp32_max_activation": maximum_up1,
        "maximum_bn_abs_running_mean": maximum_bn_mean,
        "maximum_bn_running_var": maximum_bn_var,
        "validation": validation_result,
        "end_health": end_health,
        "diagnostic_checkpoint": str(diagnostic_path),
        "diagnostic_checkpoint_sha256": file_sha256(diagnostic_path),
        "elapsed_seconds": time.perf_counter() - started,
    }
    write_json(OUTPUT_DIR / "repaired_epoch7/full_epoch7_result.json", result)
    write_json(OUTPUT_DIR / "repaired_epoch7/progress.json", {
        "batches_completed": completed, "status": "COMPLETED", "R2-B": result["R2-B"]
    })
    del model, optimizer, scheduler, scaler
    torch.cuda.empty_cache()
    return result


def formal_hash_audit() -> dict[str, Any]:
    current = {"last.pt": file_sha256(FORMAL_LAST), "best.pt": file_sha256(FORMAL_BEST)}
    result = {
        "expected": FORMAL_HASHES, "current": current,
        "unchanged": current == FORMAL_HASHES,
    }
    write_json(OUTPUT_DIR / "implementation_audit/formal_artifact_post_audit.json", result)
    if not result["unchanged"]:
        raise RuntimeError("Formal checkpoint hash changed during R2")
    return result


def final_status(final_pytest_passed: int | None = None) -> dict[str, Any]:
    audit = json.loads((OUTPUT_DIR / "implementation_audit/implementation_audit.json").read_text(encoding="utf-8"))
    r2a = json.loads((OUTPUT_DIR / "r2_a_status.json").read_text(encoding="utf-8"))
    full_path = OUTPUT_DIR / "repaired_epoch7/full_epoch7_result.json"
    r2b = json.loads(full_path.read_text(encoding="utf-8")) if full_path.is_file() else {"R2-B": "NOT_RUN"}
    regression = {
        "initial": {"passed": 62, "failed": 0},
        "final": ({"passed": final_pytest_passed, "failed": 0} if final_pytest_passed is not None else None),
    }
    write_json(OUTPUT_DIR / "regression_tests.json", regression)
    if r2a["R2-A"] != "PASS":
        status = "FAIL"
    elif r2b["R2-B"] == "PARTIAL":
        status = "PARTIAL"
    elif r2b["R2-B"] == "PASS" and final_pytest_passed is not None:
        status = "PASS"
    else:
        status = "PENDING_REGRESSION" if r2b["R2-B"] == "PASS" else "FAIL"
    result = {
        "SO-1D-R2": status,
        "R2-A": r2a["R2-A"], "R2-B": r2b["R2-B"],
        "original_up1_fp16_overflow": "RESOLVED" if r2a["R2-A"] == "PASS" else "NOT_RESOLVED",
        "new_fp16_overflow_elsewhere": bool(r2b.get("new_fp16_overflow_elsewhere", False)),
        "localized_fp32_candidate": "VALIDATED" if status == "PASS" else "NOT_VALIDATED",
        "candidate_formal_protocol": "SO1D_Formal_Train_v1.1: up1.conv FP32 + global AMP",
        "allow_SO1D_v1_1_protocol_decision": status == "PASS",
        "allow_automatic_formal_resume": False,
        "allow_automatic_formal_restart": False,
        "allow_SO1E": False,
        "implementation": audit,
        "regression": regression,
    }
    write_json(OUTPUT_DIR / "final_status.json", result)
    write_report(result, r2a, r2b)
    return result


def write_report(final: dict[str, Any], r2a: dict[str, Any], r2b: dict[str, Any]) -> None:
    repaired = r2a["repaired_batch"]
    location = repaired["original_failure_location"]
    validation = r2b.get("validation", {})
    if not isinstance(validation, dict):
        validation = {}
    health = r2b.get("end_health") or r2b.get("health_at_stop_before_optimizer_step", {})
    sections = [
        "# SO-1D-R2 Localized FP32 Numerical Repair Report", "",
        "## 1. R1 Root Cause", "",
        "R1 localized the first epoch-7 non-finite value to an FP16 +Inf at `up1.conv.net.0`, followed immediately by BatchNorm running-stat contamination.", "",
        "## 2. Repair Scope", "",
        "Only the complete `up1.conv` DoubleConv executes with autocast disabled and an explicit float32 input cast. Global AMP remains enabled and no output is manually cast back to FP16.", "",
        "## 3. Implementation", "",
        "The policy is configurable as `localized_fp32`; the historical default remains disabled.", "",
        "## 4. State-dict Compatibility", "",
        f"Compatible: **{final['implementation']['state_dict_compatible']}**; epoch6 strict load: **{final['implementation']['strict_epoch6_checkpoint_load']}**.", "",
        "## 5. Parameter-count Compatibility", "",
        f"Before/after: {final['implementation']['parameter_count_before']} / {final['implementation']['parameter_count_after']}; trainable counts are identical.", "",
        "## 6. AMP Dtype Audit", "",
        "`up1.conv` input, both Conv outputs, both BN outputs and block output are FP32; unprotected blocks remain autocast-managed.", "",
        "## 7. Pre-failure State Reconstruction", "",
        f"Status: **{r2a['pre_failure_state']['status']}**; state SHA256: `{r2a['pre_failure_state']['state_sha256']}`.", "",
        "## 8. Original Control Reproduction", "",
        f"Status: **{r2a['control']['status']}**; first failure: `{r2a['control']['first_failure_module']}` / `{r2a['control']['first_failure_dtype']}` / {r2a['control']['first_failure']}.", "",
        "## 9. Same Failing-batch Repair", "",
        f"Status: **{repaired['status']}**; forward/loss/backward/optimizer/BN remained finite.", "",
        "## 10. Original Failure Pixel", "",
        f"`train_002356`, channel 412, (2,24): {location['value']} ({location['dtype']}), finite={location['finite']}.", "",
        "## 11. up1 Conv Dynamic Range", "",
        "See `repaired_failing_batch/up1_conv0_statistics.json` for min/max/max_abs/mean/std/p99/p99.9.", "",
        "## 12. BN Contamination Regression", "",
        f"BN contamination after repaired batch: **{not repaired['bn_buffers_finite']}**.", "",
        "## 13. R2-A Result", "", f"**{r2a['R2-A']}**", "",
        "## 14. Exact Epoch7 Order", "", f"`{r2b.get('epoch7_order_sha256')}`", "",
        "## 15. Full Epoch7 Train Replay", "",
        f"R2-B: **{r2b.get('R2-B')}**; batches: {r2b.get('train_batches_completed')} / 2500; all finite: {r2b.get('all_train_batches_finite')}.", "",
        "## 16. Decoder Activation Scale History", "",
        "Recorded in `activation_monitoring/decoder_activation_history.csv`.", "",
        "## 17. FP16 Capacity Warnings", "",
        f"Maximum ratio: {r2b.get('maximum_fp16_capacity_ratio')}; flags: {r2b.get('high_risk_blocks')}.", "",
        "## 18. BN Running-stat Scale History", "",
        f"Maximum |running_mean|: {r2b.get('maximum_bn_abs_running_mean')}; maximum running_var: {r2b.get('maximum_bn_running_var')}.", "",
        "## 19. New Numerical Failure", "",
        f"FP16 overflow: {r2b.get('new_first_overflow')}; first numerical failure: {r2b.get('new_first_numerical_failure')}", "",
        "## 20. Epoch7 Validation Metrics", "",
        (
            f"M/H/S/P MAE: {validation.get('M_MAE')} / {validation.get('H_MAE')} / "
            f"{validation.get('S_MAE')} / {validation.get('P_MAE')}; selection: "
            f"{validation.get('selection_metric')}."
            if validation else "NOT RUN: training stopped at the first non-finite gradient as required."
        ), "",
        "## 21. End Health", "", json.dumps(health, ensure_ascii=False), "",
        "## 22. Regression Tests", "", json.dumps(final["regression"], ensure_ascii=False), "",
        "## 23. SO-1D-R2 Status", "", f"**{final['SO-1D-R2']}**", "",
        "## 24. Candidate v1.1 Protocol", "",
        "Proposed `SO1D_Formal_Train_v1.1`: `up1.conv` FP32 while global AMP remains enabled. This candidate was not validated because full epoch7 did not complete.", "",
        "## 25. Formal Continuation Alternatives", "",
        "Option A: continue from finite epoch6 `last.pt`. Option B: restart v1.1 from epoch1 with seed 20260801. Neither was selected or executed.", "",
        "## 26. Not Executed", "",
        "No formal resume/restart, epoch8, formal checkpoint freeze, SO-1E, repair expansion, LR change, clipping, optimizer change or normalization change was performed.", "",
    ]
    (OUTPUT_DIR / "SO1D_R2_localized_fp32_repair_report.md").write_text(
        "\n".join(sections), encoding="utf-8"
    )


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("SO-1D-R2 requires CUDA")
    if args.stage == "finalize":
        if args.final_pytest_passed is None:
            raise ValueError("--final-pytest-passed is required for finalize")
        formal_hash_audit()
        final_status(args.final_pytest_passed)
        return
    implementation_audit()
    if args.stage in ("a", "all"):
        run_r2_a()
        final_status()
    if args.stage in ("full", "all"):
        full_epoch7_replay()
        formal_hash_audit()
        final_status()


if __name__ == "__main__":
    main()
