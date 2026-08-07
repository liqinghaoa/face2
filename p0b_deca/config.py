"""P0-B config loading with guarded, fixed-scope paths."""
from __future__ import annotations
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any
import yaml

@dataclass(frozen=True)
class P0BConfig:
    project_root: Path; p0a_root: Path; output_root: Path; deca_root: Path; device: str; seed: int; pilot_size: int; license_status: str; input_modes: tuple[str, ...]; fixed_input_mode: str | None; excluded_ids: tuple[str, ...]; selection_groups: dict[str, int]; acquisition_group_rules: dict[str, dict[str, str]]; mask_bbox_expand_ratio: float; relighting_config: Path | None = None

def load_config(path: Path, project_root: Path | None = None) -> P0BConfig:
    """Load YAML paths relative to the actual project root."""
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    root = (project_root or Path(raw.get("project_root", path.parents[2]))).resolve()
    def resolve(value: str) -> Path: return Path(value).resolve() if Path(value).is_absolute() else (root / value).resolve()
    rules = {str(group): {str(key): str(value) for key, value in values.items()} for group, values in (raw.get("acquisition_group_rules") or {}).items()}
    cfg = P0BConfig(root, resolve(raw["p0a_root"]), resolve(raw["output_root"]), resolve(raw.get("deca_root", "third_party/DECA")), str(raw.get("device", "cuda")), int(raw.get("seed", 42)), int(raw.get("pilot_size", 12)), str(raw.get("license_status", "not_confirmed")), tuple(raw.get("input_modes", ["direct_p0_aligned", "mask_bbox_crop"])), raw.get("fixed_input_mode"), tuple(map(str, raw.get("excluded_ids", []))), {str(k): int(v) for k,v in raw.get("selection_groups", {}).items()}, rules, float(raw.get("mask_bbox_expand_ratio", .15)), resolve(raw["relighting_config"]) if raw.get("relighting_config") else None)
    if cfg.device not in {"cuda", "cpu", "auto"}: raise ValueError("device must be cuda/cpu/auto")
    if cfg.license_status not in {"confirmed_by_user", "not_confirmed", "not_applicable"}: raise ValueError("invalid license_status")
    if cfg.pilot_size != 12 or cfg.selection_groups != {"Xiaomi Control":4,"Xiaomi Patient":4,"HONOR Patient":4}: raise ValueError("P0-B1 requires fixed 4/4/4 pilot groups")
    required_rules = {"Xiaomi Control": {"camera_make": "Xiaomi", "camera_model": "M2006J10C", "binary_name": "Control"}, "Xiaomi Patient": {"camera_make": "Xiaomi", "camera_model": "M2006J10C", "binary_name": "Patient"}, "HONOR Patient": {"camera_make": "HONOR", "camera_model": "BVL-AN00", "binary_name": "Patient"}}
    if cfg.acquisition_group_rules != required_rules: raise ValueError("P0-B1 requires verified make/model/binary acquisition rules")
    return cfg

def as_dict(config: P0BConfig) -> dict[str, Any]:
    return {key: (str(value) if isinstance(value, Path) else list(value) if isinstance(value, tuple) else value) for key,value in asdict(config).items()}
