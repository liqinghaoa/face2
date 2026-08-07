"""Typed P0-B records; none of these claim biological ground truth."""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Any
import numpy as np

@dataclass(frozen=True)
class DecaEnvironmentInfo:
    python_version: str; torch_version: str; torchvision_version: str; pytorch3d_version: str; cuda_runtime: str; cuda_driver: str; gpu_name: str; deca_commit: str; deca_dirty: bool; rasterizer_type: str
@dataclass(frozen=True)
class DecaAssetRecord:
    name: str; path: Path; exists: bool; size_bytes: int | None; sha256: str | None; load_status: str; required: bool
@dataclass(frozen=True)
class PilotSample:
    audit_id: str; sample_id: str; patient_group_id: str; acquisition_group: str; sex: int; brightness_value: float | None; iso: float | None; exposure_time: float | None; forehead_available: bool; aligned_scene_path: Path; face_valid_mask_path: Path; skin_strict_mask_path: Path; physics_core_mask_path: Path
@dataclass
class DecaCodes:
    shape: np.ndarray; texture: np.ndarray; expression: np.ndarray; pose: np.ndarray; camera: np.ndarray; lighting: np.ndarray; detail: np.ndarray
@dataclass
class DecaGeometry:
    vertices: np.ndarray; projected_vertices: np.ndarray; landmarks_2d: np.ndarray; landmarks_3d: np.ndarray | None; coarse_normals: np.ndarray; detail_normals: np.ndarray | None
@dataclass
class PhysicalMaps:
    albedo_like: np.ndarray; shading_like: np.ndarray; reconstruction: np.ndarray; alpha_mask: np.ndarray; visibility_mask: np.ndarray; normal_coarse: np.ndarray; normal_detail: np.ndarray | None; residual_signed: np.ndarray; residual_abs: np.ndarray
@dataclass
class RelightingOutput:
    preset_name: str; sh_coefficients: np.ndarray; face_only_float: np.ndarray; composite_float: np.ndarray; display_face_png: np.ndarray; display_composite_png: np.ndarray
@dataclass
class PilotCaseResult:
    audit_id: str; sample_id: str; status: str; warnings: list[str]; error_code: str | None; runtime_metrics: dict[str, Any]; geometry_metrics: dict[str, Any]; decomposition_metrics: dict[str, Any]; relighting_metrics: dict[str, Any]; output_paths: dict[str, str]
