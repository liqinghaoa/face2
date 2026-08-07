from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class Stage1Config:
    project_root: Path
    split_csv: Path
    metadata_xlsx: Path
    oof_predictions_csv: Path
    output_dir: Path
    metadata_sheet: str
    random_seed: int
    bootstrap_repeats: int
    permutation_repeats: int
    inner_cv_splits: int
    logistic_C_grid: tuple[float, ...]
    stratification_quantiles: tuple[float, float]
    unstable_group_min_total: int
    unstable_group_min_per_class: int
    smoke_mode: bool

    @classmethod
    def from_yaml(cls, path: str | Path) -> "Stage1Config":
        config_path = Path(path)
        data = yaml.safe_load(config_path.read_text(encoding="utf-8"))

        def p(key: str) -> Path:
            return Path(data[key])

        return cls(
            project_root=p("project_root"),
            split_csv=p("split_csv"),
            metadata_xlsx=p("metadata_xlsx"),
            oof_predictions_csv=p("oof_predictions_csv"),
            output_dir=p("output_dir"),
            metadata_sheet=str(data.get("metadata_sheet", "auto")),
            random_seed=int(data.get("random_seed", 2026)),
            bootstrap_repeats=int(data.get("bootstrap_repeats", 2000)),
            permutation_repeats=int(data.get("permutation_repeats", 1000)),
            inner_cv_splits=int(data.get("inner_cv_splits", 3)),
            logistic_C_grid=tuple(float(x) for x in data.get("logistic_C_grid", [1.0])),
            stratification_quantiles=tuple(float(x) for x in data.get("stratification_quantiles", [1 / 3, 2 / 3])),  # type: ignore[arg-type]
            unstable_group_min_total=int(data.get("unstable_group_min_total", 30)),
            unstable_group_min_per_class=int(data.get("unstable_group_min_per_class", 10)),
            smoke_mode=bool(data.get("smoke_mode", False)),
        )

    def effective(self, *, smoke: bool | None = None) -> "Stage1Config":
        if smoke is None:
            smoke = self.smoke_mode
        if not smoke:
            return self
        return Stage1Config(
            project_root=self.project_root,
            split_csv=self.split_csv,
            metadata_xlsx=self.metadata_xlsx,
            oof_predictions_csv=self.oof_predictions_csv,
            output_dir=self.output_dir,
            metadata_sheet=self.metadata_sheet,
            random_seed=self.random_seed,
            bootstrap_repeats=min(self.bootstrap_repeats, 50),
            permutation_repeats=min(self.permutation_repeats, 20),
            inner_cv_splits=self.inner_cv_splits,
            logistic_C_grid=self.logistic_C_grid,
            stratification_quantiles=self.stratification_quantiles,
            unstable_group_min_total=self.unstable_group_min_total,
            unstable_group_min_per_class=self.unstable_group_min_per_class,
            smoke_mode=True,
        )


def json_safe(value: Any) -> Any:
    import math
    import numpy as np
    import pandas as pd

    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return None if not math.isfinite(float(value)) else float(value)
    if isinstance(value, float):
        return None if not math.isfinite(value) else value
    if isinstance(value, (np.ndarray,)):
        return json_safe(value.tolist())
    if isinstance(value, (pd.Timestamp,)):
        return value.isoformat()
    if isinstance(value, Path):
        return str(value)
    if pd.isna(value) if not isinstance(value, (str, bytes, dict, list, tuple)) else False:
        return None
    return value
