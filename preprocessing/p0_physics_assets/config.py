"""Configuration loading and strict P0-A invariant validation."""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Any

import yaml

from .types import GlobalMaskConfig, InputConfig, P0Config, QcConfig, RegressionConfig, RoiConfig, RuntimeConfig


def _resolve(root: Path, value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (root / path).resolve()


def parse_sample_ids(value: str | list[str] | tuple[str, ...] | None) -> tuple[str, ...]:
    """Normalize comma-separated or YAML sample IDs without coercing numeric IDs."""
    if value is None: return ()
    values = value.split(",") if isinstance(value, str) else value
    return tuple(item.strip() for item in map(str, values) if item.strip())


def load_config(path: Path, project_root_override: Path | None = None, overrides: dict[str, Any] | None = None) -> P0Config:
    """Load a P0 config; all path fields are resolved relative to project root."""
    with path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}
    if not isinstance(raw, dict): raise ValueError("P0 config root must be a mapping")
    raw = {str(k).replace("-", "_"): v for k, v in raw.items()}
    raw.update(overrides or {})
    root = (project_root_override or _resolve(path.parent.parent.parent, raw.get("project_root", "."))).resolve()
    inputs = InputConfig(root, *(_resolve(root, raw[name]) for name in ("split_csv", "raw_image_dir", "existing_blackbg_dir", "existing_meanbg_dir", "exif_workbook")), str(raw.get("exif_sheet_name", "图片元数据")), _resolve(root, raw["parsing_checkpoint"]), _resolve(root, raw["output_dir"]))
    runtime = RuntimeConfig(**{key: raw[key] for key in RuntimeConfig.__dataclass_fields__ if key in raw and key != "sample_ids"}, sample_ids=parse_sample_ids(raw.get("sample_ids")))
    mask = GlobalMaskConfig(**{key: raw[key] for key in GlobalMaskConfig.__dataclass_fields__ if key in raw})
    roi_values = {key: raw[key] for key in RoiConfig.__dataclass_fields__ if key in raw}
    if "roi_types" in roi_values: roi_values["roi_types"] = tuple(roi_values["roi_types"])
    config = P0Config(inputs, runtime, mask, RoiConfig(**roi_values), RegressionConfig(**{key: raw[key] for key in RegressionConfig.__dataclass_fields__ if key in raw}), QcConfig(**{key: raw[key] for key in QcConfig.__dataclass_fields__ if key in raw}))
    validate_config(config)
    return config


def validate_config(config: P0Config) -> None:
    """Reject settings that could silently diverge from the Global standard."""
    r, g, roi = config.runtime, config.global_mask, config.roi
    if r.image_size != 224: raise ValueError("P0-A requires image_size=224")
    if not 0 <= r.min_detection_confidence <= 1: raise ValueError("min_detection_confidence must be in [0,1]")
    if r.parsing_device not in {"auto", "cpu", "cuda"}: raise ValueError("parsing_device must be auto/cpu/cuda")
    if r.max_samples is not None and r.max_samples <= 0: raise ValueError("max_samples must be positive")
    if g.final_mask_mode != "hybrid": raise ValueError("P0-A requires Global final_mask_mode=hybrid")
    if (g.forehead_band_ratio, g.hair_repair_threshold, g.jaggedness_threshold, g.forehead_expand_ratio, g.side_expand_ratio, g.chin_expand_ratio) != (0.35, 0.10, 0.04, 0.18, 0.05, 0.03) or g.enable_jaggedness_trigger:
        raise ValueError("P0-A Global mask parameters must match the hybrid specification")
    if g.feather_kernel <= 0 or g.feather_kernel % 2 == 0: raise ValueError("feather_kernel must be positive and odd")
    if roi.min_roi_width <= 0 or roi.min_roi_height <= 0: raise ValueError("minimum ROI dimensions must be positive")
    if not 0 <= roi.forehead_valid_skin_threshold <= 1: raise ValueError("forehead_valid_skin_threshold must be in [0,1]")


def config_as_dict(config: P0Config) -> dict[str, Any]:
    """Return a YAML-serializable effective configuration."""
    def coerce(value: Any) -> Any:
        if isinstance(value, Path): return str(value)
        if isinstance(value, tuple): return list(value)
        if isinstance(value, dict): return {k: coerce(v) for k, v in value.items()}
        return value
    return coerce(asdict(config))
