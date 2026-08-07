from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class Stage3A0R0Config:
    mode: str
    experiment_name: str
    project_root: Path
    rgb_experiment_root: Path
    rgb_oof_csv: Path
    stage3_a0_root: Path
    input_image_dir: Path
    split_csv: Path
    stage1_master_csv: Path
    output_dir: Path
    run_counterfactual_inference: bool
    run_training: bool
    modify_checkpoints: bool
    classification_threshold: float
    random_seed: int
    strict_probability_tolerance: float
    numerical_equivalence_max_abs_diff: float
    numerical_equivalence_mean_abs_diff: float
    numerical_equivalence_p99_abs_diff: float
    numerical_equivalence_min_pearson: float
    numerical_equivalence_min_spearman: float
    stop_after_r0: bool

    @classmethod
    def from_yaml(cls, path: str | Path) -> "Stage3A0R0Config":
        payload = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
        config = cls(
            mode=str(payload["mode"]),
            experiment_name=str(payload["experiment_name"]),
            project_root=Path(payload["project_root"]),
            rgb_experiment_root=Path(payload["rgb_experiment_root"]),
            rgb_oof_csv=Path(payload["rgb_oof_csv"]),
            stage3_a0_root=Path(payload["stage3_a0_root"]),
            input_image_dir=Path(payload["input_image_dir"]),
            split_csv=Path(payload["split_csv"]),
            stage1_master_csv=Path(payload["stage1_master_csv"]),
            output_dir=Path(payload["output_dir"]),
            run_counterfactual_inference=bool(payload["run_counterfactual_inference"]),
            run_training=bool(payload["run_training"]),
            modify_checkpoints=bool(payload["modify_checkpoints"]),
            classification_threshold=float(payload["classification_threshold"]),
            random_seed=int(payload["random_seed"]),
            strict_probability_tolerance=float(payload["strict_probability_tolerance"]),
            numerical_equivalence_max_abs_diff=float(payload["numerical_equivalence_max_abs_diff"]),
            numerical_equivalence_mean_abs_diff=float(payload["numerical_equivalence_mean_abs_diff"]),
            numerical_equivalence_p99_abs_diff=float(payload["numerical_equivalence_p99_abs_diff"]),
            numerical_equivalence_min_pearson=float(payload["numerical_equivalence_min_pearson"]),
            numerical_equivalence_min_spearman=float(payload["numerical_equivalence_min_spearman"]),
            stop_after_r0=bool(payload["stop_after_r0"]),
        )
        config.validate()
        return config

    def validate(self) -> None:
        if self.mode != "stage3_a0_r0":
            raise ValueError(f"R0 config requires mode=stage3_a0_r0, got {self.mode!r}")
        if self.run_counterfactual_inference or self.run_training or self.modify_checkpoints:
            raise ValueError("R0 is audit-only: counterfactual, training, and checkpoint modification must be disabled")
        if self.classification_threshold != 0.5:
            raise ValueError("R0 must retain classification_threshold=0.5")
        if not self.stop_after_r0:
            raise ValueError("R0 must stop after the discrepancy audit")
        for key in ("project_root", "rgb_experiment_root", "rgb_oof_csv", "stage3_a0_root", "input_image_dir"):
            value = getattr(self, key)
            if not value.exists():
                raise FileNotFoundError(f"{key} does not exist: {value}")


def r0_dirs(output_dir: Path) -> dict[str, Path]:
    names = ["preflight", "discrepancy", "provenance", "contract", "runtime", "environment", "decision", "figures", "reports", "logs"]
    dirs = {name: output_dir / name for name in names}
    for path in dirs.values():
        path.mkdir(parents=True, exist_ok=True)
    return dirs


def as_jsonable_config(config: Stage3A0R0Config) -> dict[str, Any]:
    result = {}
    for key, value in config.__dict__.items():
        result[key] = str(value) if isinstance(value, Path) else value
    return result
