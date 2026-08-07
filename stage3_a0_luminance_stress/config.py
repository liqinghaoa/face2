from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class Stage3A0Config:
    experiment_name: str
    project_root: Path
    rgb_experiment_root: Path
    rgb_oof_csv: Path
    input_image_dir: Path
    face_valid_mask_dir: Path
    aligned_srgb_dir: Path
    parsing_label_dir: Path
    split_csv: Path
    stage1_master_csv: Path
    stage2_feature_csv: Path
    stage2_r2_root: Path
    output_dir: Path
    exposure_ev_levels: list[float]
    gamma_levels: list[float]
    classification_threshold: float
    bootstrap_iterations: int
    random_seed: int
    save_all_counterfactual_images: bool
    save_qc_examples: bool
    run_training: bool
    stop_after_stress_test: bool

    @classmethod
    def from_yaml(cls, path: str | Path) -> "Stage3A0Config":
        with Path(path).open("r", encoding="utf-8") as handle:
            data: dict[str, Any] = yaml.safe_load(handle)
        def p(key: str) -> Path:
            return Path(str(data[key]))
        cfg = cls(
            experiment_name=str(data["experiment_name"]),
            project_root=p("project_root"),
            rgb_experiment_root=p("rgb_experiment_root"),
            rgb_oof_csv=p("rgb_oof_csv"),
            input_image_dir=p("input_image_dir"),
            face_valid_mask_dir=p("face_valid_mask_dir"),
            aligned_srgb_dir=p("aligned_srgb_dir"),
            parsing_label_dir=p("parsing_label_dir"),
            split_csv=p("split_csv"),
            stage1_master_csv=p("stage1_master_csv"),
            stage2_feature_csv=p("stage2_feature_csv"),
            stage2_r2_root=p("stage2_r2_root"),
            output_dir=p("output_dir"),
            exposure_ev_levels=[float(x) for x in data["exposure_ev_levels"]],
            gamma_levels=[float(x) for x in data["gamma_levels"]],
            classification_threshold=float(data["classification_threshold"]),
            bootstrap_iterations=int(data["bootstrap_iterations"]),
            random_seed=int(data["random_seed"]),
            save_all_counterfactual_images=bool(data["save_all_counterfactual_images"]),
            save_qc_examples=bool(data["save_qc_examples"]),
            run_training=bool(data["run_training"]),
            stop_after_stress_test=bool(data["stop_after_stress_test"]),
        )
        if cfg.run_training:
            raise ValueError("Stage3-A0 forbids training")
        if cfg.classification_threshold != 0.5:
            raise ValueError("Stage3-A0 classification threshold must remain 0.5")
        if cfg.exposure_ev_levels != [-1.0, -0.5, 0.0, 0.5, 1.0]:
            raise ValueError("Stage3-A0 exposure levels are fixed")
        if cfg.gamma_levels != [0.8, 1.2]:
            raise ValueError("Stage3-A0 gamma levels are fixed")
        return cfg


def ensure_dirs(root: Path) -> dict[str, Path]:
    names = [
        "preflight", "reproduction", "qc", "qc_panels", "contact_sheets",
        "predictions", "paired", "metrics", "subgroups", "bootstrap",
        "sensitivity", "figures", "reports", "logs",
    ]
    out = {name: root / name for name in names}
    for path in out.values():
        path.mkdir(parents=True, exist_ok=True)
    return out
