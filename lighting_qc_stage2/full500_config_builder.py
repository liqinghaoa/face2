from __future__ import annotations

from pathlib import Path

import yaml

from .config import Stage2Config


def write_full500_config_template(config: Stage2Config, project_root: Path, frozen_spec_path: Path) -> Path:
    output_path = project_root / "configs" / "lighting_qc_stage2_full500_v1.yaml"
    payload = {
        "experiment_name": "Lighting_QC_Stage2_Full500_v1",
        "mode": "full500",
        "project_root": str(config.project_root).replace("\\", "/"),
        "scheme_b_root": str(config.scheme_b_root).replace("\\", "/"),
        "raw_scene_dir": str(config.raw_scene_dir).replace("\\", "/"),
        "split_csv": str(config.split_csv).replace("\\", "/"),
        "exif_xlsx": str(config.exif_xlsx).replace("\\", "/"),
        "stage1_master_csv": str(config.stage1_master_csv).replace("\\", "/"),
        "rgb_oof_csv": str(config.rgb_oof_csv).replace("\\", "/"),
        "output_dir": "E:/projects/face2/experiments/lighting_confounding/Lighting_QC_Stage2_Full500_v1",
        "expected_samples": 500,
        "metric_spec_path": str(frozen_spec_path).replace("\\", "/"),
        "primary_skin_erosion_px": 2,
        "sensitivity_erosion_px": [0, 4],
        "generate_quality_grades": False,
        "remove_samples": False,
        "run_rgb_association": True,
        "run_label_association": True,
        "run_foldwise_stratification": True,
        "stop_after_feature_extraction": False,
        "foldwise_stratification": {
            "quantiles": [0.3333, 0.6667],
            "fit_quantiles_on": "train_split_only_within_each_outer_fold",
            "apply_quantiles_to": "heldout_test_split_within_same_outer_fold",
            "merge_after_foldwise_assignment": True,
            "not_clinical_quality_grade": True,
        },
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(yaml.safe_dump(payload, sort_keys=False, allow_unicode=True), encoding="utf-8")
    return output_path
