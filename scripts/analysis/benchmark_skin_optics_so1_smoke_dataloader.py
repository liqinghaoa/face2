"""Windows-spawn-safe SO-1 smoke Train128 DataLoader benchmark."""

from __future__ import annotations

import gc
import json
import os
import random
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from skin_optics_so1.decomposition.smoke_audit import benchmark_loader, write_benchmark_csv, write_json
from skin_optics_so1.decomposition.synthetic_dataset import SO1DecompositionDataset

DATA_ROOT = PROJECT_ROOT / "data/processed/SO1_Synthetic_Generator_v1_localfull"
SMOKE_ROOT = PROJECT_ROOT / "experiments/skin_optics_so1/SO1_Decomposition_UNet_v1/smoke"
SEED = 20260801


def worker_init_fn(worker_id: int) -> None:
    seed = (torch.initial_seed() + worker_id) % (2**32)
    random.seed(seed); np.random.seed(seed)


def process_memory() -> int | None:
    try:
        import ctypes
        class Counters(ctypes.Structure):
            _fields_ = [("cb", ctypes.c_ulong), ("PageFaultCount", ctypes.c_ulong), ("PeakWorkingSetSize", ctypes.c_size_t), ("WorkingSetSize", ctypes.c_size_t), ("QuotaPeakPagedPoolUsage", ctypes.c_size_t), ("QuotaPagedPoolUsage", ctypes.c_size_t), ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t), ("QuotaNonPagedPoolUsage", ctypes.c_size_t), ("PagefileUsage", ctypes.c_size_t), ("PeakPagefileUsage", ctypes.c_size_t)]
        counters = Counters(); counters.cb = ctypes.sizeof(Counters)
        ok = ctypes.windll.psapi.GetProcessMemoryInfo(ctypes.windll.kernel32.GetCurrentProcess(), ctypes.byref(counters), counters.cb)
        return int(counters.PeakWorkingSetSize) if ok else None
    except Exception:
        return None


def make_subset() -> Subset:
    ids = [line.strip() for line in (SMOKE_ROOT / "smoke_train128_ids.txt").read_text(encoding="utf-8").splitlines() if line.strip()]
    dataset = SO1DecompositionDataset(DATA_ROOT, "train")
    lookup = {str(row.sample_id): index for index, row in enumerate(dataset.metadata.itertuples(index=False))}
    return Subset(dataset, [lookup[value] for value in ids])


def loader_for_workers(subset: Subset, workers: int) -> DataLoader:
    kwargs: dict[str, Any] = {
        "batch_size": 8, "shuffle": False, "num_workers": workers, "pin_memory": True,
        "worker_init_fn": worker_init_fn, "generator": torch.Generator().manual_seed(SEED),
    }
    if workers > 0:
        kwargs.update({"persistent_workers": True, "prefetch_factor": 2})
    return DataLoader(subset, **kwargs)


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda": raise RuntimeError("CUDA is required for the prescribed end-to-end benchmark")
    rows: list[dict[str, Any]] = []
    for workers in (0, 2, 4):
        torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats(device)
        subset = make_subset(); started = time.perf_counter(); loader = loader_for_workers(subset, workers)
        startup_seconds = time.perf_counter() - started
        try:
            stats = benchmark_loader(loader, device=device, warmup_batches=3, timed_batches=10)
            row = {"num_workers": workers, "batch_size": 8, "warmup_batches": 3, "startup_seconds": startup_seconds, **stats,
                   "peak_system_RAM": process_memory(), "peak_GPU_allocated": int(torch.cuda.max_memory_allocated(device)),
                   "peak_GPU_reserved": int(torch.cuda.max_memory_reserved(device)), "worker_error": None, "status": "PASS"}
        except Exception as exc:
            row = {"num_workers": workers, "batch_size": 8, "warmup_batches": 3, "timed_batches": 0, "startup_seconds": startup_seconds,
                   "mean_batch_seconds": None, "median_batch_seconds": None, "p95_batch_seconds": None, "samples_per_second": None,
                   "peak_system_RAM": process_memory(), "peak_GPU_allocated": int(torch.cuda.max_memory_allocated(device)),
                   "peak_GPU_reserved": int(torch.cuda.max_memory_reserved(device)), "worker_error": repr(exc), "status": "FAIL"}
        rows.append(row); del loader, subset; gc.collect(); torch.cuda.empty_cache()
    stable = [row for row in rows if row["status"] == "PASS"]
    recommended = 2 if any(row["num_workers"] == 2 and row["status"] == "PASS" for row in rows) else max(stable, key=lambda row: float(row["samples_per_second"]))["num_workers"]
    write_benchmark_csv(SMOKE_ROOT / "dataloader_benchmark.csv", rows)
    write_json(SMOKE_ROOT / "dataloader_benchmark.json", {"rows": rows, "recommended_num_workers": recommended, "selection_rule": "stability > throughput > startup; prefer workers=2 when stable"})


if __name__ == "__main__":
    main()
