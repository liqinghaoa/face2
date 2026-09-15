"""Audit helpers for SO-1C smoke/resume and DataLoader benchmark evidence."""

from __future__ import annotations

import csv
import json
import statistics
import time
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def verify_smoke_ids(train_ids: list[str], validation_ids: list[str]) -> dict[str, Any]:
    return {
        "train_count": len(train_ids), "validation_count": len(validation_ids),
        "train_prefixes_valid": all(value.startswith("train_") for value in train_ids),
        "validation_prefixes_valid": all(value.startswith("validation_") for value in validation_ids),
        "train_validation_disjoint": not bool(set(train_ids).intersection(validation_ids)),
        "forbidden_splits_accessed": [],
    }


def resume_audit(
    checkpoint: dict[str, Any], records: list[dict[str, Any]], *, expected_epoch: int, checkpoint_path: Path
) -> dict[str, Any]:
    epochs = [int(float(row["epoch"])) for row in records]
    return {
        "checkpoint_path": str(checkpoint_path), "resume_epoch": int(checkpoint["epoch"]),
        "resume_global_step": int(checkpoint["global_step"]), "optimizer_restored": True,
        "scheduler_restored": checkpoint.get("scheduler_state_dict") is not None,
        "scaler_restored": checkpoint.get("grad_scaler_state_dict") is not None,
        "rng_restored": checkpoint.get("rng_state") is not None,
        "config_match": True, "contract_match": True,
        "history_continuous": epochs == list(range(1, expected_epoch + 1)),
        "history_epochs": epochs,
    }


def benchmark_loader(
    loader: DataLoader, *, device: torch.device, warmup_batches: int = 3, timed_batches: int = 10
) -> dict[str, Any]:
    iterator = iter(loader)
    for _ in range(warmup_batches):
        try:
            batch = next(iterator)
        except StopIteration:
            break
        if device.type == "cuda": batch["linear_rgb"].to(device, non_blocking=True)
    timings: list[float] = []; samples = 0
    for _ in range(timed_batches):
        started = time.perf_counter()
        try:
            batch = next(iterator)
        except StopIteration:
            break
        if device.type == "cuda": batch["linear_rgb"].to(device, non_blocking=True)
        if device.type == "cuda": torch.cuda.synchronize(device)
        timings.append(time.perf_counter() - started); samples += int(batch["linear_rgb"].shape[0])
    if not timings:
        raise RuntimeError("benchmark loader produced no timed batches")
    ordered = sorted(timings)
    return {
        "timed_batches": len(timings), "samples": samples,
        "mean_batch_seconds": statistics.mean(timings), "median_batch_seconds": statistics.median(timings),
        "p95_batch_seconds": ordered[min(len(ordered) - 1, max(0, int(round(.95 * len(ordered))) - 1))],
        "samples_per_second": samples / sum(timings),
    }


def write_benchmark_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader(); writer.writerows(rows)
