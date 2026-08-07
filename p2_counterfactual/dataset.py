from __future__ import annotations

import json
import random
import hashlib
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset

from p2_counterfactual.assets import PRESET_NAMES, ROOT, get_preset_index
from p2_counterfactual.path_utils import DEFAULT_KNOWN_PROJECT_ROOTS, resolve_project_path
from p2_counterfactual.transforms import flip_preset_name, transform_full_six_rgb, transform_paired_rgb, transform_single_rgb


REQUIRED_BASE_MANIFEST_COLUMNS = {
    "case_id",
    "patient_group_id",
    "fold",
    "binary_label",
    "p1_qc_flag",
    "boundary_uncertain",
}

REQUIRED_NPZ_MANIFEST_COLUMNS = {
    "original_rgb_path",
    "maps_npz_path",
    "relighting_npz_path",
    "relighted_shading_path",
}

REQUIRED_IMAGE_MANIFEST_COLUMNS = {
    "original_path",
    *{f"relight_{preset}_path" for preset in PRESET_NAMES},
}

REQUIRED_PAIR_MANIFEST_COLUMNS = {
    "pair_id",
    "case_id",
    "patient_group_id",
    "binary_label",
    "fold_id",
    "original_path",
    "relighted_path",
    "preset_id",
    "preset_index",
}


def parse_preset_names(value: Any) -> list[str]:
    if isinstance(value, str):
        parsed = json.loads(value)
    elif isinstance(value, np.ndarray):
        parsed = value.tolist()
    else:
        parsed = value
    return [str(item) for item in parsed]


def load_p2_manifest(path: str | Path) -> pd.DataFrame:
    frame = pd.read_csv(path, dtype={"case_id": str, "patient_group_id": str})
    if REQUIRED_PAIR_MANIFEST_COLUMNS.issubset(frame.columns) and not REQUIRED_IMAGE_MANIFEST_COLUMNS.issubset(frame.columns):
        pair_frame = frame.copy()
        expected_pairs = len(PRESET_NAMES)
        if not (pair_frame.groupby("case_id").size() == expected_pairs).all():
            raise ValueError("P2 pair manifest must contain exactly six pairs per case")
        rows = []
        for case_id, group in pair_frame.groupby("case_id", sort=False):
            if group["preset_id"].astype(str).nunique() != expected_pairs:
                raise ValueError(f"P2 pair manifest missing unique presets for case_id={case_id}")
            first = group.iloc[0]
            row: dict[str, Any] = {
                "case_id": str(case_id),
                "patient_group_id": str(first["patient_group_id"]),
                "fold": int(first["fold_id"]),
                "binary_label": int(first["binary_label"]),
                "original_path": str(first["original_path"]),
                "p1_qc_flag": bool(first.get("p1_qc_flag", False)),
                "boundary_uncertain": bool(first.get("boundary_uncertain", False)),
                "pair_manifest_source": str(path),
            }
            for preset in PRESET_NAMES:
                preset_group = group[group["preset_id"].astype(str) == preset]
                if len(preset_group) != 1:
                    raise ValueError(f"P2 pair manifest must contain one row for preset={preset}, case_id={case_id}")
                row[f"relight_{preset}_path"] = str(preset_group.iloc[0]["relighted_path"])
            rows.append(row)
        frame = pd.DataFrame(rows)
    base_missing = sorted(REQUIRED_BASE_MANIFEST_COLUMNS - set(frame.columns))
    if base_missing:
        raise ValueError(f"P2 manifest missing required base columns: {base_missing}")
    has_npz = REQUIRED_NPZ_MANIFEST_COLUMNS.issubset(frame.columns)
    has_images = REQUIRED_IMAGE_MANIFEST_COLUMNS.issubset(frame.columns)
    if not has_npz and not has_images:
        raise ValueError(
            "P2 manifest must contain either old NPZ relighting columns "
            f"{sorted(REQUIRED_NPZ_MANIFEST_COLUMNS)} or image relighting columns "
            f"{sorted(REQUIRED_IMAGE_MANIFEST_COLUMNS)}"
        )
    frame = frame.copy()
    frame["fold"] = frame["fold"].astype(int)
    frame["binary_label"] = frame["binary_label"].astype(int)
    return frame


def _limit_balanced(frame: pd.DataFrame, max_cases: int | None, seed: int) -> pd.DataFrame:
    if max_cases is None or int(max_cases) <= 0 or len(frame) <= int(max_cases):
        return frame.reset_index(drop=True)
    max_cases = int(max_cases)
    rng = np.random.default_rng(int(seed))
    parts = []
    remaining = max_cases
    for label in sorted(frame["binary_label"].unique()):
        label_frame = frame[frame["binary_label"] == label]
        take = min(len(label_frame), max(1, max_cases // 2))
        idx = rng.choice(label_frame.index.to_numpy(), size=take, replace=False)
        parts.append(frame.loc[idx])
        remaining -= take
    if remaining > 0:
        used = pd.concat(parts).index if parts else []
        rest = frame.drop(index=used)
        if len(rest):
            idx = rng.choice(rest.index.to_numpy(), size=min(remaining, len(rest)), replace=False)
            parts.append(frame.loc[idx])
    return pd.concat(parts).sort_values(["fold", "case_id"], kind="stable").reset_index(drop=True)


def p2_a_collate(batch: list[dict[str, Any]]) -> dict[str, Any]:
    if not batch:
        raise ValueError("cannot collate an empty P2-A batch")
    output: dict[str, Any] = {}
    for key in batch[0]:
        values = [item[key] for item in batch]
        if torch.is_tensor(values[0]):
            output[key] = torch.stack(values, dim=0)
        elif key == "label" or all(isinstance(value, (int, np.integer)) for value in values):
            output[key] = torch.as_tensor(values, dtype=torch.long)
        elif key == "was_flipped" or all(isinstance(value, (bool, np.bool_)) for value in values):
            output[key] = torch.as_tensor(values, dtype=torch.bool)
        else:
            output[key] = values
    return output


class P2ASingleRGBDataset(Dataset):
    """Unified P2-A dataset for original RGB, relighting mix, and paired consistency modes."""

    def __init__(
        self,
        manifest: str | Path | pd.DataFrame,
        *,
        fold: int,
        split: str,
        input_mode: str,
        training: bool,
        color_jitter_enabled: bool = False,
        color_jitter_probability: float = 0.5,
        color_jitter_config: dict[str, Any] | None = None,
        original_probability: float = 0.5,
        presets: Iterable[str] = PRESET_NAMES,
        evaluation_source: str = "original",
        evaluation_preset_name: str | None = None,
        image_size: int = 224,
        horizontal_flip_probability: float = 0.5,
        normalization_mean: Iterable[float] = (0.485, 0.456, 0.406),
        normalization_std: Iterable[float] = (0.229, 0.224, 0.225),
        original_rgb_override_dir: str | Path | None = None,
        max_cases: int | None = None,
        seed: int = 2026,
        project_root: str | Path = ROOT,
        path_resolution: dict[str, Any] | None = None,
    ) -> None:
        frame = load_p2_manifest(manifest) if not isinstance(manifest, pd.DataFrame) else manifest.copy()
        if split == "train":
            frame = frame[frame["fold"].astype(int) != int(fold)].copy()
        elif split in {"val", "validation"}:
            frame = frame[frame["fold"].astype(int) == int(fold)].copy()
        elif split != "all":
            raise ValueError(f"Unsupported split: {split}")
        if frame.empty:
            raise ValueError(f"P2-A dataset split {split!r} for fold {fold} is empty")
        self.frame = _limit_balanced(frame, max_cases, int(seed)).reset_index(drop=True)
        self.fold = int(fold)
        self.split = str(split)
        self.input_mode = str(input_mode)
        self.training = bool(training)
        self.color_jitter_enabled = bool(color_jitter_enabled)
        self.color_jitter_probability = float(color_jitter_probability)
        self.color_jitter_config = dict(color_jitter_config or {})
        self.original_probability = float(original_probability)
        self.presets = tuple(str(x) for x in presets)
        self.evaluation_source = str(evaluation_source)
        self.evaluation_preset_name = evaluation_preset_name
        self.image_size = int(image_size)
        self.horizontal_flip_probability = float(horizontal_flip_probability)
        self.normalization_mean = tuple(float(x) for x in normalization_mean)
        self.normalization_std = tuple(float(x) for x in normalization_std)
        self.original_rgb_override_dir = Path(original_rgb_override_dir) if original_rgb_override_dir else None
        self.seed = int(seed)
        self.labels = self.frame["binary_label"].astype(int).tolist()
        self.current_epoch = 1
        self.project_root = Path(project_root)
        path_cfg = dict(path_resolution or {})
        self.path_resolution_enabled = bool(path_cfg.get("enabled", True))
        self.known_project_roots = tuple(path_cfg.get("known_project_roots", DEFAULT_KNOWN_PROJECT_ROOTS))
        self.require_path_exists = bool(path_cfg.get("require_exists", True))
        if bool(path_cfg.get("rewrite_manifest", False)):
            raise ValueError("P2-A Dataset refuses path_resolution.rewrite_manifest=true")

    def __len__(self) -> int:
        return len(self.frame)

    def set_epoch(self, epoch: int) -> None:
        self.current_epoch = max(1, int(epoch))

    def _pair_cycle_preset(self, row: pd.Series) -> tuple[str, int, int, list[str]]:
        cycle = (int(self.current_epoch) - 1) // len(self.presets)
        cycle_position = (int(self.current_epoch) - 1) % len(self.presets)
        key = f"{self.seed}:{str(row['case_id'])}:{cycle}".encode("utf-8")
        seed = int(hashlib.sha256(key).hexdigest()[:16], 16) % (2**32)
        rng = np.random.default_rng(seed)
        order = [self.presets[index] for index in rng.permutation(len(self.presets)).tolist()]
        return order[cycle_position], cycle, cycle_position, order

    def _base_meta(self, row: pd.Series) -> dict[str, Any]:
        return {
            "case_id": str(row["case_id"]),
            "patient_group_id": str(row["patient_group_id"]),
            "fold": int(row["fold"]),
            "label": int(row["binary_label"]),
            "p1_qc_flag": bool(row["p1_qc_flag"]),
            "boundary_uncertain": bool(row["boundary_uncertain"]),
        }

    def _resolve_row_path(self, row: pd.Series, column_name: str) -> Path:
        stored = row[column_name]
        if not self.path_resolution_enabled:
            path = Path(str(stored))
            resolved = path if path.is_absolute() else self.project_root / path
            if self.require_path_exists and not resolved.exists():
                raise FileNotFoundError(
                    "P2 manifest path does not exist with path resolution disabled: "
                    f"case_id={str(row['case_id'])!r}, column_name={column_name!r}, "
                    f"stored_path={str(stored)!r}, resolved_path={str(resolved)!r}"
                )
            return resolved
        return resolve_project_path(
            stored,
            self.project_root,
            known_project_roots=self.known_project_roots,
            require_exists=self.require_path_exists,
            case_id=str(row["case_id"]),
            column_name=column_name,
        )

    def _load_original_rgb(self, row: pd.Series) -> np.ndarray:
        if self.original_rgb_override_dir is not None:
            rgb_path = (self.original_rgb_override_dir / f"{str(row['case_id'])}.png").resolve()
            if not rgb_path.is_file():
                raise FileNotFoundError(
                    "Configured original_rgb_override_dir is missing a case image: "
                    f"case_id={str(row['case_id'])!r}, resolved_path={str(rgb_path)!r}"
                )
            return np.asarray(Image.open(rgb_path).convert("RGB"), dtype=np.float32) / 255.0
        original_column = "original_path" if "original_path" in row.index else "original_rgb_path"
        rgb_path = self._resolve_row_path(row, original_column)
        if rgb_path.is_file():
            return np.asarray(Image.open(rgb_path).convert("RGB"), dtype=np.float32) / 255.0
        maps_path = self._resolve_row_path(row, "maps_npz_path")
        with np.load(maps_path, allow_pickle=False) as archive:
            return np.asarray(archive["input_aligned_rgb"], dtype=np.float32)

    def _load_relighted_rgb(self, row: pd.Series, preset_name: str) -> np.ndarray:
        image_column = f"relight_{preset_name}_path"
        if image_column in row.index and str(row[image_column]).strip() not in {"", "nan", "None"}:
            path = self._resolve_row_path(row, image_column)
            return np.asarray(Image.open(path).convert("RGB"), dtype=np.float32) / 255.0
        relighting_path = self._resolve_row_path(row, "relighting_npz_path")
        self._resolve_row_path(row, "relighted_shading_path")
        with np.load(relighting_path, allow_pickle=False) as archive:
            names = parse_preset_names(archive["preset_names"])
            index = get_preset_index(names, preset_name)
            return np.asarray(archive["relighted_images"][index], dtype=np.float32)

    def _original_path_text(self, row: pd.Series) -> str:
        if self.original_rgb_override_dir is not None:
            return str((self.original_rgb_override_dir / f"{str(row['case_id'])}.png").resolve())
        column = "original_path" if "original_path" in row.index else "original_rgb_path"
        return str(self._resolve_row_path(row, column))

    def _relighted_path_text(self, row: pd.Series, preset_name: str) -> str:
        image_column = f"relight_{preset_name}_path"
        if image_column in row.index and str(row[image_column]).strip() not in {"", "nan", "None"}:
            return str(self._resolve_row_path(row, image_column))
        return str(self._resolve_row_path(row, "relighting_npz_path")) + f"::{preset_name}"

    def _sample_preset(self) -> str:
        return random.choice(self.presets)

    def _single_transform(self, image: np.ndarray, *, allow_color_jitter: bool, force_flip: bool | None = None):
        return transform_single_rgb(
            image,
            training=self.training,
            image_size=self.image_size,
            horizontal_flip_probability=self.horizontal_flip_probability,
            color_jitter_enabled=allow_color_jitter,
            color_jitter_probability=self.color_jitter_probability,
            color_jitter_config=self.color_jitter_config,
            mean=self.normalization_mean,
            std=self.normalization_std,
            force_flip=force_flip,
        )

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.frame.iloc[int(index)]
        meta = self._base_meta(row)

        if not self.training and self.evaluation_source == "relighted":
            preset = str(self.evaluation_preset_name)
            image, flipped = self._single_transform(self._load_relighted_rgb(row, preset), allow_color_jitter=False, force_flip=False)
            return {
                **meta,
                "image": image,
                "source_type": "relighted",
                "preset_name": preset,
                "preset_index": get_preset_index(self.presets, preset),
                "image_path": self._relighted_path_text(row, preset),
                "original_path": self._original_path_text(row),
                "was_flipped": bool(flipped),
            }

        if self.input_mode == "paired":
            preset = self._sample_preset()
            original = self._load_original_rgb(row)
            counter = self._load_relighted_rgb(row, preset)
            original_tensor, counter_tensor, flipped = transform_paired_rgb(
                original,
                counter,
                training=self.training,
                image_size=self.image_size,
                horizontal_flip_probability=self.horizontal_flip_probability,
                mean=self.normalization_mean,
                std=self.normalization_std,
            )
            return {
                **meta,
                "original_image": original_tensor,
                "counterfactual_image": counter_tensor,
                "preset_name": flip_preset_name(preset) if flipped else preset,
                "was_flipped": bool(flipped),
            }

        if self.input_mode == "pairwise_consistency":
            preset, cycle, cycle_position, cycle_order = self._pair_cycle_preset(row)
            original = self._load_original_rgb(row)
            relighted = self._load_relighted_rgb(row, preset)
            original_tensor, relighted_tensor, flipped = transform_paired_rgb(
                original,
                relighted,
                training=self.training,
                image_size=self.image_size,
                horizontal_flip_probability=self.horizontal_flip_probability,
                mean=self.normalization_mean,
                std=self.normalization_std,
            )
            effective_preset = flip_preset_name(preset) if flipped else preset
            pair_id = f"{str(row['case_id'])}__{preset}"
            return {
                **meta,
                "original_image": original_tensor,
                "relighted_image": relighted_tensor,
                "sampled_preset_name": preset,
                "preset_name": effective_preset,
                "preset_index": get_preset_index(self.presets, preset),
                "pair_id": pair_id,
                "cycle_index": int(cycle),
                "cycle_position": int(cycle_position),
                "cycle_order": cycle_order,
                "original_path": self._original_path_text(row),
                "relighted_path": self._relighted_path_text(row, preset),
                "was_flipped": bool(flipped),
            }

        if self.input_mode == "full6_consistency":
            original = self._load_original_rgb(row)
            relighted = [self._load_relighted_rgb(row, preset) for preset in self.presets]
            original_tensor, relighted_tensor, flipped = transform_full_six_rgb(
                original,
                relighted,
                training=self.training,
                image_size=self.image_size,
                horizontal_flip_probability=self.horizontal_flip_probability,
                mean=self.normalization_mean,
                std=self.normalization_std,
            )
            effective_presets = [flip_preset_name(preset) if flipped else preset for preset in self.presets]
            return {
                **meta,
                "original_image": original_tensor,
                "relighted_images": relighted_tensor,
                "preset_names": effective_presets,
                "preset_ids": list(range(len(self.presets))),
                "relighted_paths": [self._relighted_path_text(row, preset) for preset in self.presets],
                "original_path": self._original_path_text(row),
                "was_flipped": bool(flipped),
            }

        use_relighted = self.training and self.input_mode == "relight_mix" and random.random() >= self.original_probability
        if use_relighted:
            preset = self._sample_preset()
            image, flipped = self._single_transform(self._load_relighted_rgb(row, preset), allow_color_jitter=False)
            effective_preset = flip_preset_name(preset) if flipped else preset
            return {
                **meta,
                "image": image,
                "source_type": "relighted",
                "preset_name": effective_preset,
                "preset_index": get_preset_index(self.presets, preset),
                "image_path": self._relighted_path_text(row, preset),
                "original_path": self._original_path_text(row),
                "was_flipped": bool(flipped),
            }

        image, flipped = self._single_transform(
            self._load_original_rgb(row),
            allow_color_jitter=self.training and self.color_jitter_enabled,
        )
        original_path = self._original_path_text(row)
        return {
            **meta,
            "image": image,
            "source_type": "original",
            "preset_name": None,
            "preset_index": -1,
            "image_path": original_path,
            "original_path": original_path,
            "was_flipped": bool(flipped),
        }
