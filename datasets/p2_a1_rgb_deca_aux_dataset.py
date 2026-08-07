"""Read-only P0-A RGB plus exactly three frozen P1 latent codes for P2-A1."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset

from datasets.control_patient_binary_dataset import map_three_class_to_binary


FEATURE_KEYS = ("tex_key", "shape_key", "detail_key")
FEATURE_ORDER = ("tex_code", "shape_code", "detail_code")
FEATURE_DIMENSIONS = (50, 100, 128)
TOTAL_FEATURE_DIM = sum(FEATURE_DIMENSIONS)


def _read_split_table(table: str | Path) -> pd.DataFrame:
    frame = pd.read_csv(
        table,
        dtype={"ID": "string", "patient_group_id": "string"},
        encoding="utf-8-sig",
    )
    required = {"ID", "patient_group_id", "NYHA", "label_3class", "fold"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"P0-A split table missing fields: {sorted(missing)}")
    if frame.ID.isna().any() or frame.ID.duplicated().any() or len(frame) != 500:
        raise ValueError("P0-A split table must contain 500 unique non-null IDs")
    return frame


def split_fold_zero(table: str | Path, train: bool) -> pd.DataFrame:
    """Use the immutable five-fold table: fold != 0 for train, fold == 0 for val."""
    return split_fold(table, 0, train)


def split_fold(table: str | Path, fold: int, train: bool) -> pd.DataFrame:
    """Use the immutable five-fold table for one requested validation fold."""
    if fold not in range(5):
        raise ValueError(f"P2-A1 fold must be in [0, 1, 2, 3, 4], got {fold}")
    frame = _read_split_table(table)
    if set(frame.fold.unique()) != set(range(5)):
        raise ValueError("P0-A split table must contain exactly folds 0-4")
    selected = frame[frame.fold.ne(fold) if train else frame.fold.eq(fold)].reset_index(drop=True)
    expected = 400 if train else 100
    if len(selected) != expected:
        raise ValueError(f"fixed fold-{fold} {'train' if train else 'validation'} count is not {expected}")
    return selected


def read_frozen_auxiliary_vector(p1_root: str | Path, p1: dict, case_id: str) -> np.ndarray:
    """Read *only* configured tex, shape, detail fields in the mandated order."""
    path = Path(p1_root) / p1["cases_subdir"] / str(case_id) / p1["latent_filename"]
    if not path.is_file():
        raise FileNotFoundError(f"missing frozen P1 latent file: {path}")
    with np.load(path, allow_pickle=False) as archive:
        values = []
        for config_key, expected_dim in zip(FEATURE_KEYS, FEATURE_DIMENSIONS, strict=True):
            source_key = p1[config_key]
            if source_key not in archive.files:
                raise KeyError(f"{path}: missing configured latent key {source_key}")
            value = np.asarray(archive[source_key], dtype=np.float32).reshape(-1)
            if value.shape != (expected_dim,):
                raise ValueError(f"{path}: {source_key} dimension {value.shape}, expected {(expected_dim,)}")
            values.append(value)
    vector = np.concatenate(values, axis=0).astype(np.float32, copy=False)
    if vector.shape != (TOTAL_FEATURE_DIM,) or not np.isfinite(vector).all():
        raise ValueError(f"invalid frozen auxiliary vector for case {case_id}")
    return vector


def audit_p0a_p1_assets(config: dict, output_dir: str | Path) -> dict:
    """Validate immutable P0-A/P1 inputs and write the required provenance record."""
    data, assets, p1 = config["data"], config["assets"], config["p1"]
    frame = _read_split_table(data["split_table"])
    master = pd.read_csv(data["master_index"], dtype={"ID": "string"}, encoding="utf-8-sig")
    if "ID" not in master.columns or set(master.ID.dropna().astype(str)) != set(frame.ID.astype(str)):
        raise ValueError("P0-A master_index IDs do not exactly match fixed split IDs")

    train, val = split_fold_zero(data["split_table"], True), split_fold_zero(data["split_table"], False)
    overlap = set(train.patient_group_id.astype(str)) & set(val.patient_group_id.astype(str))
    if overlap:
        raise ValueError(f"patient-group leakage across fold 0: {sorted(overlap)[:5]}")

    image_root, p1_root = Path(data["image_root"]), Path(assets["p1_root"])
    missing_images, missing_latents = [], []
    for case_id in frame.ID.astype(str):
        if not (image_root / f"{case_id}.png").is_file():
            missing_images.append(case_id)
        try:
            read_frozen_auxiliary_vector(p1_root, p1, case_id)
        except (FileNotFoundError, KeyError, ValueError):
            missing_latents.append(case_id)
    if missing_images or missing_latents:
        raise FileNotFoundError(
            f"P2-A1 asset check failed: missing_images={len(missing_images)}, "
            f"invalid_or_missing_latents={len(missing_latents)}"
        )

    ready_csv = p1_root / "indexes" / "p1_ready_cases.csv"
    ready_ids = None
    if ready_csv.is_file():
        ready = pd.read_csv(ready_csv, dtype={"ID": "string"}, encoding="utf-8-sig")
        id_column = "ID" if "ID" in ready.columns else "case_id"
        if id_column not in ready.columns:
            raise ValueError("P1 ready-case index has neither ID nor case_id")
        ready_ids = set(ready[id_column].dropna().astype(str))
        if ready_ids != set(frame.ID.astype(str)):
            raise ValueError("P1 ready-case IDs do not exactly match P0-A split IDs")

    def counts(part: pd.DataFrame) -> dict:
        labels = part.label_3class.map(map_three_class_to_binary)
        return {"total": int(len(part)), "control": int((labels == 0).sum()), "patient": int((labels == 1).sum())}

    record = {
        "p0a_root": str(Path(assets["p0a_root"])),
        "split_table": str(Path(data["split_table"])),
        "master_index": str(Path(data["master_index"])),
        "image_root": str(image_root),
        "p1_root": str(p1_root),
        "p1_ready_cases_csv": str(ready_csv) if ready_ids is not None else None,
        "latent_filename": p1["latent_filename"],
        "latent_keys": {k: p1[k] for k in FEATURE_KEYS},
        "feature_order": list(FEATURE_ORDER),
        "feature_dimensions": dict(zip(FEATURE_ORDER, FEATURE_DIMENSIONS, strict=True)),
        "total_feature_dim": TOTAL_FEATURE_DIM,
        "total_cases": int(len(frame)),
        "train": counts(train),
        "validation": counts(val),
        "patient_group_overlap_count": 0,
        "all_images_present": True,
        "all_configured_latents_present_and_finite": True,
    }
    destination = Path(output_dir) / "p0a_p1_asset_record.json"
    destination.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
    return record


class P2A1Dataset(Dataset):
    def __init__(self, table, image_root, p1_root, p1, fold, train, transform, stats=None):
        self.fold = int(fold)
        self.frame = split_fold(table, self.fold, train)
        self.root = Path(image_root)
        self.p1root, self.p1, self.transform, self.stats = Path(p1_root), p1, transform, stats
        self.vectors = np.stack([read_frozen_auxiliary_vector(self.p1root, p1, case_id) for case_id in self.frame.ID])

    def set_stats(self, stats):
        if int(stats["train_case_count"]) != 400:
            raise ValueError("A1 normalization statistics must be fitted on the 400-case train split")
        self.stats = stats

    def __len__(self):
        return len(self.frame)

    def __getitem__(self, index):
        row = self.frame.iloc[index]
        case_id = str(row.ID)
        image_path = self.root / f"{case_id}.png"
        with Image.open(image_path) as source:
            image = self.transform(source.convert("RGB"))
        vector = self.vectors[index]
        if self.stats is not None:
            vector = (vector - self.stats["mean"]) / self.stats["std"]
        return {
            "image": image,
            "aux_vector": torch.tensor(vector, dtype=torch.float32),
            "label": torch.tensor(map_three_class_to_binary(row.label_3class), dtype=torch.long),
            "sample_id": case_id,
            "patient_group_id": str(row.patient_group_id),
            "original_nyha": int(row.NYHA),
            "original_three_class_label": int(row.label_3class),
        }


def train_stats(dataset: P2A1Dataset, epsilon: float = 1e-6, validation_case_count: int | None = None) -> dict:
    if len(dataset) != 400:
        raise ValueError("normalization may only fit the fixed 400-case training set")
    return {
        "mean": dataset.vectors.mean(axis=0).astype(np.float32),
        "std": np.maximum(dataset.vectors.std(axis=0), epsilon).astype(np.float32),
        "feature_dim": TOTAL_FEATURE_DIM,
        "tex_dim": FEATURE_DIMENSIONS[0],
        "shape_dim": FEATURE_DIMENSIONS[1],
        "detail_dim": FEATURE_DIMENSIONS[2],
        "feature_order": np.asarray(FEATURE_ORDER),
        "train_case_count": len(dataset),
        "validation_case_count": 100 if validation_case_count is None else int(validation_case_count),
        "fold": dataset.fold,
    }
