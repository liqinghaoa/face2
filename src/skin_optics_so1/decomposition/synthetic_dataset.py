"""PyTorch dataset for frozen SO-1 synthetic decomposition arrays."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from .contract import ARRAY_SPECS, EXPECTED_SPLIT_COUNTS, SPLITS


class SO1DecompositionDataset(Dataset[dict[str, Any]]):
    """Read SO-1 decomposition samples from per-split memmap arrays.

    Arrays are opened lazily so DataLoader workers do not inherit open memmap
    handles from the parent process.
    """

    def __init__(
        self,
        data_root: str | Path,
        split: str,
        *,
        require_full_count: bool = True,
        metadata_columns: list[str] | None = None,
    ) -> None:
        if split not in SPLITS:
            raise ValueError(f"Unknown SO-1 split {split!r}; expected one of {SPLITS}")
        self.data_root = Path(data_root)
        self.split = split
        self.split_dir = self.data_root / split
        if not self.split_dir.is_dir():
            raise FileNotFoundError(f"SO-1 split directory not found: {self.split_dir}")
        self._arrays: dict[str, np.ndarray] | None = None

        metadata_path = self.split_dir / "metadata.csv"
        if not metadata_path.is_file():
            raise FileNotFoundError(f"SO-1 split metadata not found: {metadata_path}")
        metadata = pd.read_csv(metadata_path)
        required = {"sample_id", "split", "split_index", "base_latent_id"}
        missing = required.difference(metadata.columns)
        if missing:
            raise ValueError(f"SO-1 metadata is missing columns: {sorted(missing)}")
        if require_full_count and len(metadata) != EXPECTED_SPLIT_COUNTS[split]:
            raise ValueError(
                f"{split} metadata row count mismatch: "
                f"{len(metadata)} != {EXPECTED_SPLIT_COUNTS[split]}"
            )
        if set(metadata["split"].astype(str)) != {split}:
            raise ValueError(f"{split} metadata contains rows from other splits")
        expected_index = list(range(len(metadata)))
        sorted_metadata = metadata.sort_values("split_index", kind="mergesort").reset_index(drop=True)
        if sorted_metadata["split_index"].astype(int).tolist() != expected_index:
            raise ValueError(f"{split} split_index must be a dense 0-based sequence")

        if metadata_columns is not None:
            keep = list(dict.fromkeys(["sample_id", "split", "split_index", *metadata_columns]))
            missing_keep = set(keep).difference(sorted_metadata.columns)
            if missing_keep:
                raise ValueError(f"Requested metadata columns are absent: {sorted(missing_keep)}")
            sorted_metadata = sorted_metadata[keep]
        self.metadata = sorted_metadata

    def __len__(self) -> int:
        return len(self.metadata)

    def _open_arrays(self) -> dict[str, np.ndarray]:
        if self._arrays is None:
            arrays = {}
            for name, spec in ARRAY_SPECS.items():
                path = self.split_dir / spec.filename
                if not path.is_file():
                    raise FileNotFoundError(f"SO-1 array file not found: {path}")
                arrays[name] = np.load(path, mmap_mode="r")
            self._arrays = arrays
        return self._arrays

    def get_metadata(self, index: int) -> dict[str, Any]:
        row = self.metadata.iloc[int(index)]
        return row.to_dict()

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.metadata.iloc[int(index)]
        split_index = int(row["split_index"])
        arrays = self._open_arrays()
        linear_rgb = np.array(arrays["linear_rgb"][split_index], dtype=np.float32, copy=True)
        target_mhsp = np.array(arrays["target_mhsp"][split_index], dtype=np.float32, copy=True)
        valid_mask = np.array(arrays["valid_mask"][split_index], dtype=np.float32, copy=True)
        written = int(arrays["written"][split_index])
        if written != 1:
            raise RuntimeError(f"SO-1 sample is not marked written: {self.split}[{split_index}]")
        return {
            "linear_rgb": torch.from_numpy(linear_rgb),
            "target_mhsp": torch.from_numpy(target_mhsp),
            "valid_mask": torch.from_numpy(valid_mask),
            "input": torch.from_numpy(linear_rgb),
            "target": torch.from_numpy(target_mhsp),
            "mask": torch.from_numpy(valid_mask),
            "sample_id": str(row["sample_id"]),
            "split": str(row["split"]),
            "split_index": split_index,
            "base_latent_id": int(row["base_latent_id"]),
            "acquisition_variant_id": int(row["acquisition_variant_id"])
            if "acquisition_variant_id" in row.index
            else -1,
        }
