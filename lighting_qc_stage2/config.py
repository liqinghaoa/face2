from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class Stage2Config:
    experiment_name: str
    project_root: Path
    scheme_b_root: Path
    raw_scene_dir: Path
    split_csv: Path
    exif_xlsx: Path
    stage1_master_csv: Path
    rgb_oof_csv: Path
    output_dir: Path
    pilot_size: int
    random_seed: int
    roi_geometry: dict[str, dict[str, float]]
    erosion_candidates: tuple[int, ...]
    brightness_threshold_candidates: tuple[float, ...]
    darkness_threshold_candidates: tuple[float, ...]
    specular_y_threshold_candidates: tuple[float, ...]
    specular_chroma_threshold_candidates: tuple[float, ...]
    shadow_ratio_candidates: tuple[float, ...]
    clip_high_threshold_candidates: tuple[int, ...]
    clip_low_threshold_candidates: tuple[int, ...]
    roundtrip_iou_threshold: float
    centroid_distance_threshold: float
    shadow_gaussian_sigma: float
    save_intermediate_masks: bool
    save_raw_space_masks: bool
    generate_qc_panels: bool
    stop_after_pilot: bool
    mode: str = "pilot32"
    old_pilot_manifest_csv: Path | None = None
    expected_samples: int = 500
    metric_spec_path: Path | None = None
    generate_quality_grades: bool = False
    remove_samples: bool = False
    run_rgb_association: bool = True
    run_label_association: bool = True
    run_foldwise_stratification: bool = True
    stop_after_feature_extraction: bool = False
    full500_output_dir: Path | None = None
    stage1_master_csv_r2: Path | None = None
    run_statistical_repair: bool = False
    run_exif_decomposition: bool = False
    run_traditional_exif_covariate_adjustment: bool = False
    run_stage3: bool = False
    bootstrap_iterations: int = 2000

    @classmethod
    def from_yaml(cls, path: str | Path) -> "Stage2Config":
        data = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
        spec_data: dict[str, Any] = {}
        if data.get("metric_spec_path") and Path(data["metric_spec_path"]).is_file():
            spec_data = yaml.safe_load(Path(data["metric_spec_path"]).read_text(encoding="utf-8")) or {}
        roi_geometry = data.get("roi_geometry") or spec_data.get("roi_definition", {}).get("geometry")
        if roi_geometry is None:
            raise KeyError("roi_geometry is required unless metric_spec_path provides roi_definition.geometry")
        return cls(
            experiment_name=str(data["experiment_name"]),
            project_root=Path(data["project_root"]),
            scheme_b_root=Path(data["scheme_b_root"]),
            raw_scene_dir=Path(data["raw_scene_dir"]),
            split_csv=Path(data["split_csv"]),
            exif_xlsx=Path(data["exif_xlsx"]),
            stage1_master_csv=Path(data["stage1_master_csv"]),
            rgb_oof_csv=Path(data["rgb_oof_csv"]),
            output_dir=Path(data["output_dir"]),
            pilot_size=int(data.get("pilot_size", 32)),
            random_seed=int(data.get("random_seed", 2026)),
            roi_geometry=dict(roi_geometry),
            erosion_candidates=tuple(int(x) for x in data.get("erosion_candidates", [0, 2, 4])),
            brightness_threshold_candidates=tuple(float(x) for x in data.get("brightness_threshold_candidates", [0.80])),
            darkness_threshold_candidates=tuple(float(x) for x in data.get("darkness_threshold_candidates", [0.03, 0.05])),
            specular_y_threshold_candidates=tuple(float(x) for x in data.get("specular_threshold_candidates", {"y": [0.75]})["y"]),
            specular_chroma_threshold_candidates=tuple(float(x) for x in data.get("specular_threshold_candidates", {"chroma": [0.08]})["chroma"]),
            shadow_ratio_candidates=tuple(float(x) for x in data.get("shadow_ratio_candidates", [0.55])),
            clip_high_threshold_candidates=tuple(int(x) for x in data.get("clip_threshold_candidates", {"high": [250]})["high"]),
            clip_low_threshold_candidates=tuple(int(x) for x in data.get("clip_threshold_candidates", {"low": [5]})["low"]),
            roundtrip_iou_threshold=float(data.get("roundtrip_iou_threshold", 0.95)),
            centroid_distance_threshold=float(data.get("centroid_distance_threshold", 2.0)),
            shadow_gaussian_sigma=float(data.get("shadow_gaussian_sigma", 1.2)),
            save_intermediate_masks=bool(data.get("save_intermediate_masks", True)),
            save_raw_space_masks=bool(data.get("save_raw_space_masks", True)),
            generate_qc_panels=bool(data.get("generate_qc_panels", True)),
            stop_after_pilot=bool(data.get("stop_after_pilot", True)),
            mode=str(data.get("mode", "pilot32")),
            old_pilot_manifest_csv=Path(data["old_pilot_manifest_csv"]) if data.get("old_pilot_manifest_csv") else None,
            expected_samples=int(data.get("expected_samples", 500)),
            metric_spec_path=Path(data["metric_spec_path"]) if data.get("metric_spec_path") else None,
            generate_quality_grades=bool(data.get("generate_quality_grades", False)),
            remove_samples=bool(data.get("remove_samples", False)),
            run_rgb_association=bool(data.get("run_rgb_association", True)),
            run_label_association=bool(data.get("run_label_association", True)),
            run_foldwise_stratification=bool(data.get("run_foldwise_stratification", True)),
            stop_after_feature_extraction=bool(data.get("stop_after_feature_extraction", False)),
            full500_output_dir=Path(data["full500_output_dir"]) if data.get("full500_output_dir") else None,
            stage1_master_csv_r2=Path(data["stage1_master_csv"]) if data.get("stage1_master_csv") else None,
            run_statistical_repair=bool(data.get("run_statistical_repair", False)),
            run_exif_decomposition=bool(data.get("run_exif_decomposition", False)),
            run_traditional_exif_covariate_adjustment=bool(data.get("run_traditional_exif_covariate_adjustment", False)),
            run_stage3=bool(data.get("run_stage3", False)),
            bootstrap_iterations=int(data.get("bootstrap_iterations", 2000)),
        )


def json_safe(value: Any) -> Any:
    import math
    import numpy as np
    import pandas as pd

    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        f = float(value)
        return None if not math.isfinite(f) else f
    if isinstance(value, np.ndarray):
        return json_safe(value.tolist())
    try:
        if pd.isna(value) and not isinstance(value, (str, bytes)):
            return None
    except Exception:
        pass
    return value
