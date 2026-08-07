"""Lazy P1 component dataset backed by the frozen 500-case manifest."""

from __future__ import annotations

from collections import OrderedDict
from pathlib import Path
from typing import Any, Mapping

import pandas as pd
from torch.utils.data import Dataset

from utils.p1_component_registry import P1ComponentSpec, get_component_spec
from utils.p1_representation_normalization import load_raw_component_sample
from utils.p1_rgb_audit import load_p1_frame


REQUIRED_COLUMNS = {
    "case_id",
    "patient_group_id",
    "fold",
    "label_original",
    "label_3class",
    "label_binary",
    "rgb_path",
    "face_valid_mask_path",
    "physics_core_skin_mask_path",
    "maps_path",
    "latents_path",
}


def _ensure_frame(frame: pd.DataFrame) -> pd.DataFrame:
    missing = REQUIRED_COLUMNS.difference(frame.columns)
    if missing:
        raise ValueError(f"P1 component frame missing required columns: {sorted(missing)}")
    result = frame.reset_index(drop=True).copy()
    result["case_id"] = result["case_id"].astype(str)
    result["patient_group_id"] = result["patient_group_id"].astype(str)
    if "group_id" not in result.columns:
        result["group_id"] = result["patient_group_id"]
    result["group_id"] = result["group_id"].astype(str)
    return result


class P1ComponentDataset(Dataset):
    """Return one raw P1 component sample at a time.

    The dataset performs no training normalization.  It only loads the minimum
    raw assets required by the requested component, preserving labels and fold
    assignments exactly as recorded in the frozen manifest.
    """

    def __init__(
        self,
        frame_or_manifest: pd.DataFrame | str | Path,
        experiment_key: str,
        *,
        fixed_split: str | Path | None = None,
        cache_size: int = 0,
    ) -> None:
        self.spec: P1ComponentSpec = get_component_spec(experiment_key)
        if isinstance(frame_or_manifest, pd.DataFrame):
            frame = _ensure_frame(frame_or_manifest)
        else:
            manifest = Path(frame_or_manifest)
            if fixed_split is None:
                raise ValueError("fixed_split is required when constructing from a manifest path")
            frame = load_p1_frame(manifest, Path(fixed_split))
            frame = _ensure_frame(frame)
        self.frame = frame
        self.cache_size = int(cache_size)
        self._cache: OrderedDict[int, dict[str, Any]] = OrderedDict()

    @classmethod
    def from_manifest(
        cls,
        manifest: str | Path,
        fixed_split: str | Path,
        experiment_key: str,
        *,
        cache_size: int = 0,
    ) -> "P1ComponentDataset":
        frame = load_p1_frame(Path(manifest), Path(fixed_split))
        return cls(frame, experiment_key, cache_size=cache_size)

    def __len__(self) -> int:
        return int(len(self.frame))

    def _load_item(self, index: int) -> dict[str, Any]:
        row = self.frame.iloc[index].to_dict()
        sample = load_raw_component_sample(row, self.spec)
        sample["component_key"] = self.spec.key
        sample["display_name"] = self.spec.display_name
        sample["representation_name"] = self.spec.representation
        sample["input_type"] = self.spec.input_type
        sample["model_type"] = self.spec.model_type
        sample["valid_mask_name"] = self.spec.mask
        return sample

    def __getitem__(self, index: int) -> dict[str, Any]:
        if self.cache_size <= 0:
            return self._load_item(index)
        if index in self._cache:
            self._cache.move_to_end(index)
            return self._cache[index]
        item = self._load_item(index)
        self._cache[index] = item
        while len(self._cache) > self.cache_size:
            self._cache.popitem(last=False)
        return item


def build_component_dataset(
    frame_or_manifest: pd.DataFrame | str | Path,
    experiment_key: str,
    *,
    fixed_split: str | Path | None = None,
    cache_size: int = 0,
) -> P1ComponentDataset:
    return P1ComponentDataset(frame_or_manifest, experiment_key, fixed_split=fixed_split, cache_size=cache_size)
