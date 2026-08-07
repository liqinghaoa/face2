"""Numpy memmap storage for SO-1 synthetic datasets."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np


def prepare_split_storage(output_root: str | Path, split_counts: dict[str, int], size: int, overwrite: bool = False) -> dict[str, dict[str, np.memmap]]:
    root = Path(output_root)
    stores: dict[str, dict[str, np.memmap]] = {}
    for split, n in split_counts.items():
        split_dir = root / split
        split_dir.mkdir(parents=True, exist_ok=True)
        specs = {
            "linear_rgb": ("linear_rgb.f16.npy", np.float16, (n, 3, size, size)),
            "target_mhsp": ("target_mhsp.f16.npy", np.float16, (n, 4, size, size)),
            "valid_mask": ("valid_mask.u8.npy", np.uint8, (n, 1, size, size)),
            "written": ("written.u8.npy", np.uint8, (n,)),
        }
        stores[split] = {}
        for key, (name, dtype, shape) in specs.items():
            path = split_dir / name
            if path.exists() and not overwrite:
                arr = np.lib.format.open_memmap(path, mode="r+", dtype=dtype, shape=shape)
                if arr.shape != shape or arr.dtype != dtype:
                    raise ValueError(f"Existing storage mismatch: {path}")
            else:
                arr = np.lib.format.open_memmap(path, mode="w+", dtype=dtype, shape=shape)
                arr[:] = 0
                arr.flush()
            stores[split][key] = arr
    return stores


def write_sample(stores: dict[str, dict[str, np.memmap]], split: str, split_index: int, linear_rgb: np.ndarray, target_mhsp: np.ndarray, mask: np.ndarray, resume: bool = False) -> None:
    store = stores[split]
    if resume and int(store["written"][split_index]) == 1:
        return
    store["linear_rgb"][split_index] = linear_rgb.astype(np.float16)
    store["target_mhsp"][split_index] = target_mhsp.astype(np.float16)
    store["valid_mask"][split_index, 0] = mask.astype(np.uint8)
    store["written"][split_index] = np.uint8(1)
    for arr in store.values():
        arr.flush()


def write_metadata(output_root: str | Path, split: str, rows: list[dict]) -> None:
    path = Path(output_root) / split / "metadata.csv"
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_run_hashes(output_root: str | Path, payload: dict) -> None:
    Path(output_root).mkdir(parents=True, exist_ok=True)
    (Path(output_root) / "run_hashes.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

