"""Typed records shared by the P0-A asset builder."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np


@dataclass(frozen=True)
class InputConfig:
    """Immutable locations for source assets and the isolated P0 output."""
    project_root: Path; split_csv: Path; raw_image_dir: Path; existing_blackbg_dir: Path
    existing_meanbg_dir: Path; exif_workbook: Path; exif_sheet_name: str; parsing_checkpoint: Path
    output_dir: Path


@dataclass(frozen=True)
class RuntimeConfig:
    """Deterministic runtime controls; labels never appear here as pixel inputs."""
    image_size: int = 224; min_detection_confidence: float = 0.5; parsing_device: str = "auto"
    seed: int = 42; max_samples: int | None = None; sample_ids: tuple[str, ...] = ()
    overwrite: bool = False; resume: bool = False


@dataclass(frozen=True)
class GlobalMaskConfig:
    """Exact hybrid Global-mask constants copied from the authoritative YAML."""
    final_mask_mode: str = "hybrid"; forehead_band_ratio: float = 0.35
    hair_repair_threshold: float = 0.10; jaggedness_threshold: float = 0.04
    enable_jaggedness_trigger: bool = False; forehead_expand_ratio: float = 0.18
    side_expand_ratio: float = 0.05; chin_expand_ratio: float = 0.03; feather_kernel: int = 11


@dataclass(frozen=True)
class RoiConfig:
    """Full-canvas ROI geometry constraints and forehead availability threshold."""
    roi_types: tuple[str, ...] = ("left_cheek", "right_cheek", "combined_cheek", "forehead", "lip", "eye", "chin")
    min_roi_width: int = 8; min_roi_height: int = 8; forehead_valid_skin_threshold: float = 0.20


@dataclass(frozen=True)
class RegressionConfig:
    """Pixel-level reconstruction acceptance limits for legacy black backgrounds."""
    enabled: bool = True; mae_max: float = 1.0; p99_abs_diff_max: float = 2.0
    ssim_min: float = 0.995; large_diff_threshold: int = 5; large_diff_fraction_max: float = 0.001


@dataclass(frozen=True)
class QcConfig:
    """Label-free QC preview selection settings."""
    save_qc: bool = True; num_random_qc: int = 20


@dataclass(frozen=True)
class P0Config:
    """Complete typed P0-A configuration assembled from YAML and CLI overrides."""
    inputs: InputConfig; runtime: RuntimeConfig = field(default_factory=RuntimeConfig)
    global_mask: GlobalMaskConfig = field(default_factory=GlobalMaskConfig)
    roi: RoiConfig = field(default_factory=RoiConfig)
    regression: RegressionConfig = field(default_factory=RegressionConfig)
    qc: QcConfig = field(default_factory=QcConfig)


@dataclass(frozen=True)
class SampleRecord:
    """One fixed-split sample plus its verified source paths and non-pixel metadata."""
    image_id: str; patient_group_id: str; sex: int; sex_name: str; nyha: int
    label_3class: int; label_3class_name: str; fold: int; source_path: Path
    blackbg_path: Path; meanbg_path: Path; metadata: dict[str, Any]


@dataclass
class AlignmentArtifacts:
    """Original-to-224 similarity geometry and the aligned scene/source-valid raster."""
    aligned_rgb: np.ndarray; source_valid: np.ndarray; matrix_2x3: np.ndarray
    original_keypoints_5: np.ndarray; aligned_keypoints_5: np.ndarray
    original_landmarks_468: np.ndarray; aligned_landmarks_468: np.ndarray
    expanded_bbox_xywh: tuple[int, int, int, int]; source_image_hw: tuple[int, int]
    parameters: dict[str, Any]


@dataclass
class ParsingArtifacts:
    """One BiSeNet label map and the Global semantic intermediate masks."""
    label_map: np.ndarray; selected_semantic_mask: np.ndarray
    semantic_regularized_mask: np.ndarray; candidate_envelope_mask: np.ndarray


@dataclass
class MaskArtifacts:
    """P0 base masks plus full-canvas ROI geometry/effective masks and measures."""
    final_face_mask: np.ndarray; face_valid_mask: np.ndarray; skin_strict_mask: np.ndarray
    feather_alpha: np.ndarray; roi_geometry: dict[str, np.ndarray] = field(default_factory=dict)
    roi_effective: dict[str, np.ndarray] = field(default_factory=dict); roi_metrics: dict[str, dict[str, Any]] = field(default_factory=dict)


@dataclass(frozen=True)
class RegressionResult:
    """Black-background reconstruction metrics and optional absolute-difference heatmap."""
    status: str; mae: float; rmse: float; max_abs_diff: float; p99_abs_diff: float
    rgb_ssim: float; large_diff_fraction: float; difference_heatmap: np.ndarray | None = None


@dataclass
class SampleBuildResult:
    """Traceable core/overall outcome for one requested split record."""
    image_id: str; core_status: str; overall_status: str; failure_reason: str = ""
    warnings: list[str] = field(default_factory=list); fields: dict[str, Any] = field(default_factory=dict)
    core_asset_status: str = ""; legacy_regression_status: str = ""
    boundary_ambiguity_status: str = ""; p0_usable: int = 0
