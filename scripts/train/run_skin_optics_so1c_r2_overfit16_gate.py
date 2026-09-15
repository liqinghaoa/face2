"""Controlled SO-1C-R2 rerun of the frozen original overfit16 gate."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import platform
import random
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import yaml
from torch.utils.data import DataLoader, Subset

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from skin_optics_so1.decomposition.diagnostic_metrics import diagnostic_metrics, final_output_gradient_norms
from skin_optics_so1.decomposition.overfit_diagnostics import (
    collect_predictions, save_previews, summarize_collection, write_json,
)
from skin_optics_so1.decomposition.losses import CHANNEL_NAMES, masked_smooth_l1_loss
from skin_optics_so1.decomposition.synthetic_dataset import SO1DecompositionDataset
from skin_optics_so1.decomposition.unet_decomposer import SO1UNetDecomposer

SEED = 20260801
BATCH_SIZE = 8
MAX_STEPS = 3000
EVALUATION_INTERVAL = 50
MILESTONES = {1000, 1500, 2000, 2500, 3000}
DATA_ROOT = PROJECT_ROOT / "data/processed/SO1_Synthetic_Generator_v1_localfull"
ORIGINAL_IDS_PATH = PROJECT_ROOT / "experiments/skin_optics_so1/SO1_Decomposition_UNet_v1/preflight/overfit16_ids.txt"
ORIGINAL_CONFIG = PROJECT_ROOT / "config/train/skin_optics_so1/so1_decomposition_overfit_v1.yaml"
OUTPUT_DIR = PROJECT_ROOT / "experiments/skin_optics_so1/SO1_Decomposition_UNet_v1/so1c_r2_overfit16_gate"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def set_seed() -> None:
    random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.benchmark = False; torch.backends.cudnn.deterministic = True


def subset_for_ids(dataset: SO1DecompositionDataset, sample_ids: list[str]) -> Subset:
    lookup = {str(row.sample_id): index for index, row in enumerate(dataset.metadata.itertuples(index=False))}
    missing = [sample_id for sample_id in sample_ids if sample_id not in lookup]
    if missing:
        raise RuntimeError(f"Original overfit16 IDs missing from train metadata: {missing}")
    return Subset(dataset, [lookup[sample_id] for sample_id in sample_ids])


def model_fingerprint(model: SO1UNetDecomposer) -> dict[str, Any]:
    state = model.state_dict()
    digest = hashlib.sha256()
    for key, value in state.items():
        digest.update(key.encode("utf-8")); digest.update(value.detach().cpu().numpy().tobytes())
    def stats(name: str) -> dict[str, float]:
        value = state[name].detach().float()
        return {"mean": float(value.mean()), "std": float(value.std()), "min": float(value.min()), "max": float(value.max())}
    return {"state_sha256": digest.hexdigest(), "initialization": "SO1UNetDecomposer.reset_parameters: Kaiming normal Conv/ConvTranspose; BatchNorm weight=1,bias=0", "first_layer": stats("inc.net.0.weight"), "output_layer": stats("outc.weight")}


def gpu_stats(device: torch.device) -> dict[str, int]:
    return {
        "allocated_bytes": int(torch.cuda.memory_allocated(device)),
        "reserved_bytes": int(torch.cuda.memory_reserved(device)),
        "peak_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
        "peak_reserved_bytes": int(torch.cuda.max_memory_reserved(device)),
    }


def gate(summary: dict[str, Any], loss_reduction: float) -> bool:
    channels = summary["channels"]
    return bool(
        loss_reduction >= .90 and channels["M"]["MAE"] <= .02 and channels["H"]["MAE"] <= .02
        and channels["S"]["MAE"] <= .03 and all(np.isfinite(channels[name]["MAE"]) for name in CHANNEL_NAMES)
        and channels["P"]["pred"]["std"] > 0 and channels["P"]["active_region_recall"] > 0
    )


def save_checkpoint(path: Path, model: SO1UNetDecomposer, step: int, record: dict[str, Any], fingerprint: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model_state_dict": model.state_dict(), "optimizer_step": step, "record": record, "seed": SEED, "initial_state_sha256": fingerprint["state_sha256"]}, path)


def read_page_file() -> dict[str, Any] | None:
    try:
        import ctypes
        class MemoryStatus(ctypes.Structure):
            _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong), ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong), ("ullTotalPageFile", ctypes.c_ulonglong), ("ullAvailPageFile", ctypes.c_ulonglong), ("ullTotalVirtual", ctypes.c_ulonglong), ("ullAvailVirtual", ctypes.c_ulonglong), ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]
        status = MemoryStatus(); status.dwLength = ctypes.sizeof(MemoryStatus)
        ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status))
        return {"system_ram_total_bytes": int(status.ullTotalPhys), "system_ram_available_bytes": int(status.ullAvailPhys), "page_file_total_bytes": int(status.ullTotalPageFile), "page_file_available_bytes": int(status.ullAvailPageFile)}
    except Exception as exc:
        return {"unavailable": str(exc)}


def main() -> None:
    args = parse_args(); output = args.output_dir.resolve(); metrics_dir = output / "metrics"; checkpoints = output / "checkpoints"
    if output.exists() and any(output.iterdir()):
        raise RuntimeError(f"R2 output directory must be new and empty: {output}")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable in the Face2 environment")
    output.mkdir(parents=True); metrics_dir.mkdir(); checkpoints.mkdir()
    device = torch.device("cuda")
    original_ids = [line.strip() for line in ORIGINAL_IDS_PATH.read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(original_ids) != 16 or len(set(original_ids)) != 16:
        raise RuntimeError("original overfit16_ids.txt must contain 16 unique IDs")
    original_hash = sha256(ORIGINAL_IDS_PATH)
    original_ids_copy = output / "original_overfit16_ids.txt"; original_ids_copy.write_text("\n".join(original_ids) + "\n", encoding="utf-8")
    if sha256(original_ids_copy) != original_hash:
        raise RuntimeError("Copied original overfit16 IDs hash mismatch")
    (output / "original_overfit16_ids_sha256.txt").write_text(original_hash + "\n", encoding="utf-8")
    metadata = pd.read_csv(DATA_ROOT / "train/metadata.csv").set_index("sample_id")
    id_audit = metadata.loc[original_ids, ["base_latent_id", "camera_name", "light_name", "camera_light_pair"]].reset_index()
    id_audit.to_csv(output / "original_overfit16_id_audit.csv", index=False, encoding="utf-8")
    config = yaml.safe_load(ORIGINAL_CONFIG.read_text(encoding="utf-8"))
    config["train"]["batch_size"] = BATCH_SIZE; config["train"]["max_steps"] = MAX_STEPS
    config["train"]["seed"] = SEED; config["train"]["amp"] = False; config["train"]["scheduler"] = "none"; config["train"]["early_stopping_patience"] = None
    config["r2_protocol"] = {"batch_size": BATCH_SIZE, "max_steps": MAX_STEPS, "evaluation_interval": EVALUATION_INTERVAL, "original_ids_sha256": original_hash}
    (output / "config_resolved.yaml").write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    set_seed(); dataset = SO1DecompositionDataset(DATA_ROOT, "train"); subset = subset_for_ids(dataset, original_ids)
    train_loader = DataLoader(subset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0, pin_memory=True, generator=torch.Generator().manual_seed(SEED))
    eval_loader = DataLoader(subset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=True)
    model = SO1UNetDecomposer().to(device); fingerprint = model_fingerprint(model)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    torch.cuda.reset_peak_memory_stats(device); started = time.perf_counter(); started_at = datetime.now(timezone.utc).isoformat()
    initial_loss: float | None = None; history: list[dict[str, Any]] = []; gate_history: list[dict[str, Any]] = []; best: dict[str, Any] | None = None
    iterator = iter(train_loader); step_times: list[float] = []
    for step in range(1, MAX_STEPS + 1):
        try: batch = next(iterator)
        except StopIteration: iterator = iter(train_loader); batch = next(iterator)
        step_started = time.perf_counter(); model.train()
        rgb = batch["linear_rgb"].to(device, non_blocking=True); target = batch["target_mhsp"].to(device, non_blocking=True); mask = batch["valid_mask"].to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True); prediction = model(rgb); losses = masked_smooth_l1_loss(prediction, target, mask); losses["total"].backward()
        gradients = final_output_gradient_norms(model); optimizer.step(); step_times.append(time.perf_counter() - step_started)
        if initial_loss is None: initial_loss = float(losses["total"].detach().cpu())
        if step % EVALUATION_INTERVAL != 0:
            continue
        collection = collect_predictions(model, eval_loader, device); summary = summarize_collection(collection); channels = summary["channels"]
        current_loss = float(losses["total"].detach().cpu()); reduction = 1.0 - current_loss / max(initial_loss, 1e-12)
        record = {"step": step, "total_loss": current_loss, **{f"{name}_loss": float(losses[name].detach().cpu()) for name in CHANNEL_NAMES}, **{f"{name}_MAE": channels[name]["MAE"] for name in CHANNEL_NAMES}, "P_pred_std": channels["P"]["pred"]["std"], "P_active_recall": channels["P"]["active_region_recall"], "learning_rate": float(optimizer.param_groups[0]["lr"]), "wall_clock_seconds": time.perf_counter() - started, "seconds_per_step": float(np.mean(step_times[-EVALUATION_INTERVAL:])), "samples_per_second": BATCH_SIZE / max(float(np.mean(step_times[-EVALUATION_INTERVAL:])), 1e-12), "loss_reduction": reduction, **gpu_stats(device), **gradients}
        record["selection_metric"] = record["M_MAE"] + record["H_MAE"]; record["formal_gate_pass"] = gate(summary, reduction)
        history.append(record); gate_history.append({key: record[key] for key in ("step", "formal_gate_pass", "loss_reduction", "M_MAE", "H_MAE", "S_MAE", "P_MAE", "P_pred_std", "P_active_recall", "selection_metric")})
        write_json(metrics_dir / f"step_{step}_metrics.json", {"record": record, "summary": summary})
        if step in MILESTONES:
            save_checkpoint(checkpoints / f"step_{step}.pt", model, step, record, fingerprint); save_previews(output / "prediction_panels" / f"step_{step}", collection, summary, step=step)
        if best is None or record["selection_metric"] < best["record"]["selection_metric"]:
            best = {"record": record, "summary": summary, "collection": collection, "state": {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}}
            save_checkpoint(checkpoints / "best.pt", model, step, record, fingerprint)
    assert initial_loss is not None and best is not None
    final_collection = collect_predictions(model, eval_loader, device); final_summary = summarize_collection(final_collection); last = history[-1]
    save_checkpoint(checkpoints / "last.pt", model, MAX_STEPS, last, fingerprint)
    write_json(metrics_dir / "best_metrics.json", {"record": best["record"], "summary": best["summary"]}); write_json(metrics_dir / "last_metrics.json", {"record": last, "summary": final_summary})
    save_previews(output / "prediction_panels" / "last", final_collection, final_summary, step=MAX_STEPS)
    model.load_state_dict(best["state"])
    best_collection = collect_predictions(model, eval_loader, device)
    save_previews(output / "prediction_panels" / "best", best_collection, best["summary"], step=int(best["record"]["step"]))
    with (output / "training_history.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(history[0].keys())); writer.writeheader(); writer.writerows(history)
    with (output / "gate_history.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(gate_history[0].keys())); writer.writeheader(); writer.writerows(gate_history)
    pass_steps = [row["step"] for row in gate_history if row["formal_gate_pass"]]; first = pass_steps[0] if pass_steps else None
    intervals: list[list[int]] = []
    for step in pass_steps:
        if not intervals or step != intervals[-1][1] + EVALUATION_INTERVAL: intervals.append([step, step])
        else: intervals[-1][1] = step
    stable = any((end - start) // EVALUATION_INTERVAL + 1 >= 3 for start, end in intervals)
    finished_at = datetime.now(timezone.utc).isoformat(); elapsed = time.perf_counter() - started
    runtime = {"run_start_time": started_at, "run_end_time": finished_at, "wall_clock_seconds": elapsed, "optimizer_steps": MAX_STEPS, "average_seconds_per_step": elapsed / MAX_STEPS, "median_seconds_per_step": float(np.median(step_times)), "samples_per_second": BATCH_SIZE / max(float(np.mean(step_times)), 1e-12), "gpu_name": torch.cuda.get_device_name(device), **gpu_stats(device), "page_file": read_page_file(), "initialization_fingerprint": fingerprint, "first_gate_pass_step": first, "gate_pass_intervals": intervals, "formal_gate_pass": bool(pass_steps), "stable_gate_pass": stable}
    write_json(output / "runtime.json", runtime)


if __name__ == "__main__":
    main()
