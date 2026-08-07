"""Strict SO-1 synthetic-generation configuration."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any, TypeVar, get_origin, get_type_hints

import yaml


T = TypeVar("T")


def _require_keys(data: dict[str, Any], cls: type) -> None:
    expected = {f.name for f in fields(cls)}
    actual = set(data)
    extra = actual - expected
    missing = expected - actual
    if extra:
        raise ValueError(f"Unknown config field(s) for {cls.__name__}: {sorted(extra)}")
    if missing:
        raise ValueError(f"Missing config field(s) for {cls.__name__}: {sorted(missing)}")


def _coerce(cls: type[T], data: Any) -> T:
    if not hasattr(cls, "__dataclass_fields__"):
        return data
    if not isinstance(data, dict):
        raise ValueError(f"Expected mapping for {cls.__name__}")
    _require_keys(data, cls)
    hints = get_type_hints(cls)
    kwargs: dict[str, Any] = {}
    for f in fields(cls):
        value = data[f.name]
        typ = hints[f.name]
        origin = get_origin(typ)
        if hasattr(typ, "__dataclass_fields__"):
            kwargs[f.name] = _coerce(typ, value)
        elif origin in (list, tuple):
            if not isinstance(value, list):
                raise ValueError(f"{f.name} must be a list")
            kwargs[f.name] = value
        else:
            kwargs[f.name] = value
    return cls(**kwargs)  # type: ignore[misc]


def _canonical_json(data: Any) -> str:
    return json.dumps(data, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def sha256_canonical(data: Any) -> str:
    return hashlib.sha256(_canonical_json(data).encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class PatchConfig:
    size: int


@dataclass(frozen=True)
class CountsConfig:
    train_base_latents: int
    train_variants_per_latent: int
    validation: int
    id_test: int
    camera_ood: int
    light_ood: int
    joint_ood: int

    @property
    def train(self) -> int:
        return self.train_base_latents * self.train_variants_per_latent

    def split_counts(self) -> dict[str, int]:
        return {
            "train": self.train,
            "validation": self.validation,
            "id_test": self.id_test,
            "camera_ood": self.camera_ood,
            "light_ood": self.light_ood,
            "joint_ood": self.joint_ood,
        }


@dataclass(frozen=True)
class ControlsConfig:
    m_min: float
    m_max: float
    h_min: float
    h_max: float
    m_base_min: float
    m_base_max: float
    h_base_min: float
    h_base_max: float
    shading_min: float
    shading_max: float
    specular_min: float
    specular_max: float
    exposure: float


@dataclass(frozen=True)
class MaskConfig:
    full_probability: float
    boundary_probability: float
    holes_probability: float
    mixed_probability: float
    min_valid_fraction: float

    def probabilities(self) -> dict[str, float]:
        return {
            "full": self.full_probability,
            "boundary": self.boundary_probability,
            "holes": self.holes_probability,
            "mixed": self.mixed_probability,
        }


@dataclass(frozen=True)
class LightsConfig:
    seen: list[str]
    unseen: list[str]


@dataclass(frozen=True)
class RenderConfig:
    batch_size: int
    oom_fallback_batch_sizes: list[int]
    compute_dtype: str
    max_total_clip_fraction: float
    max_channel_clip_fraction: float
    max_retries: int


@dataclass(frozen=True)
class StorageConfig:
    rgb_dtype: str
    target_dtype: str
    mask_dtype: str
    format: str
    save_full_srgb_dataset: bool


@dataclass(frozen=True)
class SyntheticGenerationConfig:
    version: str
    global_seed: int
    patch: PatchConfig
    counts: CountsConfig
    controls: ControlsConfig
    mask: MaskConfig
    lights: LightsConfig
    render: RenderConfig
    storage: StorageConfig

    @property
    def total_samples(self) -> int:
        return sum(self.counts.split_counts().values())

    def canonical_hash(self) -> str:
        return sha256_canonical(to_plain_dict(self))


def to_plain_dict(obj: Any) -> Any:
    if hasattr(obj, "__dataclass_fields__"):
        return {f.name: to_plain_dict(getattr(obj, f.name)) for f in fields(obj)}
    if isinstance(obj, list):
        return [to_plain_dict(v) for v in obj]
    return obj


def load_config(path: str | Path) -> SyntheticGenerationConfig:
    with Path(path).open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    if not isinstance(raw, dict):
        raise ValueError(f"Expected YAML mapping: {path}")
    cfg = _coerce(SyntheticGenerationConfig, raw)
    validate_config(cfg)
    return cfg


def validate_config(cfg: SyntheticGenerationConfig) -> None:
    if cfg.version != "SO1_Synthetic_v1":
        raise ValueError("version must be SO1_Synthetic_v1")
    if cfg.global_seed <= 0:
        raise ValueError("global_seed must be positive")
    if cfg.patch.size != 256:
        raise ValueError("patch.size must be 256 for SO-1 v1")
    if cfg.counts.train_variants_per_latent != 2:
        raise ValueError("train_variants_per_latent must be 2")
    for name, count in cfg.counts.split_counts().items():
        if count <= 0:
            raise ValueError(f"{name} count must be positive")
    c = cfg.controls
    if (c.m_min, c.m_max, c.h_min, c.h_max) != (0.0, 1.0, 0.0, 1.0):
        raise ValueError("M/H control ranges must be [0,1]")
    if not (0.0 <= c.m_base_min < c.m_base_max <= 1.0 and 0.0 <= c.h_base_min < c.h_base_max <= 1.0):
        raise ValueError("M/H base ranges are invalid")
    if not (0.25 <= c.shading_min < c.shading_max <= 2.0):
        raise ValueError("Shading range must stay inside SO-0 primary [0.25,2.0]")
    if not (0.0 <= c.specular_min < c.specular_max <= 0.10):
        raise ValueError("Specular range must stay inside SO-0 primary [0,0.10]")
    if c.exposure != 1.0:
        raise ValueError("exposure must be fixed at 1.0")
    probs = list(cfg.mask.probabilities().values())
    if any(p < 0 for p in probs) or abs(sum(probs) - 1.0) > 1e-8:
        raise ValueError("Mask probabilities must be non-negative and sum to 1")
    if not (0.20 <= cfg.mask.min_valid_fraction <= 1.0):
        raise ValueError("mask.min_valid_fraction must be in [0.20,1.00]")
    if cfg.lights.seen != ["D65", "A", "FL2"] or cfg.lights.unseen != ["FL11"]:
        raise ValueError("SO-1 v1 lights must be seen=[D65,A,FL2], unseen=[FL11]")
    if cfg.render.compute_dtype != "float32":
        raise ValueError("render.compute_dtype must be float32")
    if cfg.render.batch_size <= 0 or 1 not in cfg.render.oom_fallback_batch_sizes:
        raise ValueError("OOM fallback batch sizes must include 1")
    if cfg.render.max_retries < 0:
        raise ValueError("max_retries must be non-negative")
    if cfg.storage.rgb_dtype != "float16" or cfg.storage.target_dtype != "float16" or cfg.storage.mask_dtype != "uint8":
        raise ValueError("Storage dtypes must be rgb=float16,target=float16,mask=uint8")
    if cfg.storage.format != "numpy_open_memmap":
        raise ValueError("Only numpy_open_memmap storage is supported")
    if cfg.storage.save_full_srgb_dataset:
        raise ValueError("save_full_srgb_dataset must remain false for SO-1 v1")


def assert_so0_range_compatible(cfg: SyntheticGenerationConfig, so0_config: Any) -> list[str]:
    """Check SO-1 configured ranges against actual SO-0 validation ranges."""

    notes: list[str] = []
    c = cfg.controls
    sh_min, sh_max = so0_config.range_minmax("shading", "primary_range")
    sp_min, sp_max = so0_config.range_minmax("specular", "primary_range")
    m = so0_config.parameter("melanin_control")
    h = so0_config.parameter("hemoglobin_control")
    expected = {
        "m": (c.m_min, c.m_max, float(m["min"]), float(m["max"])),
        "h": (c.h_min, c.h_max, float(h["min"]), float(h["max"])),
        "shading": (c.shading_min, c.shading_max, sh_min, sh_max),
        "specular": (c.specular_min, c.specular_max, sp_min, sp_max),
    }
    for name, (lo, hi, slo, shi) in expected.items():
        if lo < slo or hi > shi:
            raise ValueError(f"{name} range [{lo},{hi}] exceeds SO-0 actual range [{slo},{shi}]")
        if (lo, hi) != (slo, shi):
            notes.append(f"{name}: SO-1 uses [{lo},{hi}] within SO-0 [{slo},{shi}]")
    return notes
