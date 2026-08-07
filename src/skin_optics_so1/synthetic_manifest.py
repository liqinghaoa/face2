"""Manifest-first construction and validation for SO-1 synthetic data."""

from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import numpy as np

from skin_optics_so1.camera_light_split import CameraLightSplit, FAILED_PAIRS, shuffled_balanced_pairs
from skin_optics_so1.random_fields import sobol_base_values, stable_seed
from skin_optics_so1.synthetic_config import SyntheticGenerationConfig
from skin_optics_so1.synthetic_masks import generate_valid_mask, sample_mask_type
from skin_optics_so1.random_fields import rng_from_seed


SPLITS = ("train", "validation", "id_test", "camera_ood", "light_ood", "joint_ood")
SPLIT_BASE_OFFSETS = {
    "train": 0,
    "validation": 1_000_000,
    "id_test": 2_000_000,
    "camera_ood": 3_000_000,
    "light_ood": 4_000_000,
    "joint_ood": 5_000_000,
}


@dataclass
class ManifestRow:
    row_index: int
    sample_id: str
    split: str
    split_index: int
    base_latent_id: int
    acquisition_variant_id: int
    m_base: float
    h_base: float
    m_seed: int
    h_seed: int
    s_seed: int
    p_seed: int
    mask_seed: int
    acquisition_seed: int
    camera_light_seed: int
    camera_name: str
    light_name: str
    camera_light_pair: str
    exposure: float
    valid_mask_type: str
    expected_valid_fraction: float
    so0_version: str
    so0_config_hash: str
    spectral_asset_hash: str
    generator_config_hash: str
    actual_valid_fraction: float = float("nan")
    low_clip_fraction: float = float("nan")
    high_clip_fraction: float = float("nan")
    total_clip_fraction: float = float("nan")
    per_channel_clip_fraction: str = ""
    retry_count: int = 0
    generation_status: str = "PENDING"
    rgb_min: float = float("nan")
    rgb_max: float = float("nan")
    rgb_mean: float = float("nan")
    rgb_std: float = float("nan")
    m_mean: float = float("nan")
    m_std: float = float("nan")
    h_mean: float = float("nan")
    h_std: float = float("nan")
    s_mean: float = float("nan")
    s_std: float = float("nan")
    p_mean: float = float("nan")
    p_std: float = float("nan")
    p_nonzero_fraction: float = float("nan")
    p_blob_count: int = -1
    initial_camera_name: str = ""
    initial_light_name: str = ""
    initial_camera_light_pair: str = ""
    final_camera_name: str = ""
    final_light_name: str = ""
    final_camera_light_pair: str = ""
    retry_pair_history: str = "[]"


def _so0_hash(path: str | Path) -> str:
    import hashlib

    h = hashlib.sha256()
    p = Path(path)
    for rel in ("frozen_config.yaml", "formula_registry.yaml", "FROZEN.json", "tables/colorchecker_calibration_metrics.csv"):
        target = p / rel
        h.update(rel.encode("utf-8"))
        h.update(target.read_bytes())
    return h.hexdigest()


def _asset_hash(path: str | Path) -> str:
    import hashlib

    p = Path(path) / "frozen_asset_hash_after.txt"
    return hashlib.sha256(p.read_bytes()).hexdigest() if p.exists() else ""


def _bases_for_split(cfg: SyntheticGenerationConfig, split: str, n_base: int) -> np.ndarray:
    seed = stable_seed(cfg.global_seed, split, "sobol", 0, "m_field")
    values = sobol_base_values(n_base, seed)
    values[:, 0] = cfg.controls.m_base_min + (cfg.controls.m_base_max - cfg.controls.m_base_min) * (
        (values[:, 0] - 0.05) / 0.90
    )
    values[:, 1] = cfg.controls.h_base_min + (cfg.controls.h_base_max - cfg.controls.h_base_min) * (
        (values[:, 1] - 0.05) / 0.90
    )
    return values.astype(np.float32)


def build_manifest(
    cfg: SyntheticGenerationConfig,
    split_cfg: CameraLightSplit,
    so0_output_dir: str | Path = "outputs/SO0_Forward_Model_v1.1",
) -> list[ManifestRow]:
    rows: list[ManifestRow] = []
    so0_hash = _so0_hash(so0_output_dir)
    asset_hash = _asset_hash(so0_output_dir)
    config_hash = cfg.canonical_hash()
    row_index = 0
    for split in SPLITS:
        count = cfg.counts.split_counts()[split]
        n_base = cfg.counts.train_base_latents if split == "train" else count
        bases = _bases_for_split(cfg, split, n_base)
        pairs = shuffled_balanced_pairs(
            split_cfg.allowed_pairs(split),
            count,
            stable_seed(cfg.global_seed, split, "pair_assignment", 0, "camera_light"),
        )
        for split_index in range(count):
            if split == "train":
                base_idx = split_index // cfg.counts.train_variants_per_latent
                variant = split_index % cfg.counts.train_variants_per_latent
            else:
                base_idx = split_index
                variant = 0
            base_latent_id = SPLIT_BASE_OFFSETS[split] + base_idx
            pair = pairs[split_index]
            if split == "train" and variant == 1 and pairs[split_index].key == pairs[split_index - 1].key:
                pair = pairs[(split_index + 1) % len(pairs)]
            m_seed = stable_seed(cfg.global_seed, split, base_latent_id, 0, "m_field")
            h_seed = stable_seed(cfg.global_seed, split, base_latent_id, 0, "h_field")
            mask_seed = stable_seed(cfg.global_seed, split, base_latent_id, 0, "mask")
            rng = rng_from_seed(mask_seed)
            mask_type = sample_mask_type(rng, cfg.mask.probabilities())
            _, valid_fraction, _ = generate_valid_mask(mask_seed, cfg.patch.size, mask_type, cfg.mask.min_valid_fraction)
            row = ManifestRow(
                row_index=row_index,
                sample_id=f"{split}_{split_index:06d}",
                split=split,
                split_index=split_index,
                base_latent_id=base_latent_id,
                acquisition_variant_id=variant,
                m_base=float(bases[base_idx, 0]),
                h_base=float(bases[base_idx, 1]),
                m_seed=m_seed,
                h_seed=h_seed,
                s_seed=stable_seed(cfg.global_seed, split, base_latent_id, variant, "s_field"),
                p_seed=stable_seed(cfg.global_seed, split, base_latent_id, variant, "p_field"),
                mask_seed=mask_seed,
                acquisition_seed=stable_seed(cfg.global_seed, split, base_latent_id, variant, "acquisition"),
                camera_light_seed=stable_seed(cfg.global_seed, split, base_latent_id, variant, "camera_light"),
                camera_name=pair.camera_name,
                light_name=pair.light_name,
                camera_light_pair=pair.key,
                initial_camera_name=pair.camera_name,
                initial_light_name=pair.light_name,
                initial_camera_light_pair=pair.key,
                final_camera_name=pair.camera_name,
                final_light_name=pair.light_name,
                final_camera_light_pair=pair.key,
                exposure=cfg.controls.exposure,
                valid_mask_type=mask_type,
                expected_valid_fraction=valid_fraction,
                so0_version="SO0_Forward_Model_v1.1",
                so0_config_hash=so0_hash,
                spectral_asset_hash=asset_hash,
                generator_config_hash=config_hash,
            )
            rows.append(row)
            row_index += 1
    validate_manifest(rows, cfg, split_cfg)
    return rows


def validate_manifest(rows: list[ManifestRow], cfg: SyntheticGenerationConfig, split_cfg: CameraLightSplit) -> None:
    if len(rows) != cfg.total_samples:
        raise ValueError(f"Manifest row count mismatch: {len(rows)} != {cfg.total_samples}")
    if [r.row_index for r in rows] != list(range(len(rows))):
        raise ValueError("Manifest row_index must be contiguous")
    if len({r.sample_id for r in rows}) != len(rows):
        raise ValueError("sample_id must be unique")
    for split, count in cfg.counts.split_counts().items():
        split_rows = [r for r in rows if r.split == split]
        if len(split_rows) != count:
            raise ValueError(f"{split} count mismatch")
        allowed = {p.key for p in split_cfg.allowed_pairs(split)}
        assigned = [r.camera_light_pair for r in split_rows]
        if any(p not in allowed for p in assigned):
            raise ValueError(f"{split} contains pair outside allowed rules")
        counts = {p: assigned.count(p) for p in set(assigned)}
        if counts and max(counts.values()) - min(counts.values()) > 1:
            raise ValueError(f"{split} pair assignment is not balanced")
    bases_by_split = {s: {r.base_latent_id for r in rows if r.split == s} for s in SPLITS}
    for i, a in enumerate(SPLITS):
        for b in SPLITS[i + 1 :]:
            if bases_by_split[a] & bases_by_split[b]:
                raise ValueError(f"base_latent_id leakage between {a} and {b}")
    train = [r for r in rows if r.split == "train"]
    by_base: dict[int, list[ManifestRow]] = {}
    for r in train:
        by_base.setdefault(r.base_latent_id, []).append(r)
    for base, group in by_base.items():
        if len(group) != 2 or {r.acquisition_variant_id for r in group} != {0, 1}:
            raise ValueError(f"Train base {base} does not have variants 0/1")
        a, b = sorted(group, key=lambda r: r.acquisition_variant_id)
        if not (a.m_seed == b.m_seed and a.h_seed == b.h_seed and a.mask_seed == b.mask_seed):
            raise ValueError("Train paired variants must share M/H/mask seeds")
        if a.camera_light_pair == b.camera_light_pair:
            raise ValueError("Train paired variants must use different camera-light pairs")
    for r in rows:
        if (r.camera_name, r.light_name) in FAILED_PAIRS:
            raise ValueError(f"Failed SO-0 pair present in manifest: {r.camera_light_pair}")


def write_manifest(rows: Iterable[ManifestRow], output_dir: str | Path) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = list(rows)
    if not rows:
        raise ValueError("Cannot write empty manifest")
    fieldnames = list(asdict(rows[0]).keys())
    csv_path = output_dir / "synthetic_manifest.csv"
    jsonl_path = output_dir / "synthetic_manifest.jsonl"
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            writer.writerow(asdict(r))
    with jsonl_path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(asdict(r), sort_keys=True, ensure_ascii=False) + "\n")
