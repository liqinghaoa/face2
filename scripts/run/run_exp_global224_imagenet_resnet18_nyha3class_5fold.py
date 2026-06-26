"""One-command entry point for the complete NYHA three-class experiment."""

from __future__ import annotations

import subprocess
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = (
    PROJECT_ROOT
    / "config"
    / "train"
    / "roi"
    / "nyha_3class_lip_roi_resnet18_weightedce.yaml"
)
REQUIRED_COLUMNS = {
    "ID",
    "patient_group_id",
    "image_path",
    "NYHA",
    "label_3class",
    "label_3class_name",
    "SEX",
    "sex_name",
    "fold",
}


def load_config(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(f"Config file does not exist: {path}")
    with path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError(f"Config root must be a mapping: {path}")
    return config


def preflight(config_path: Path, config: dict) -> None:
    split_dir = Path(config["data"]["split_dir"]).expanduser().resolve()
    if not split_dir.is_dir():
        raise FileNotFoundError(f"Split directory does not exist: {split_dir}")

    n_folds = int(config["data"]["n_folds"])
    missing_images: list[tuple[str, str]] = []
    validation_ids: list[str] = []
    for fold in range(n_folds):
        fold_frames: dict[str, pd.DataFrame] = {}
        for split_name, pattern_key in (
            ("train", "train_csv_pattern"),
            ("val", "val_csv_pattern"),
        ):
            csv_path = split_dir / config["data"][pattern_key].format(fold=fold)
            if not csv_path.is_file():
                raise FileNotFoundError(
                    f"Missing fold file for fold={fold}, split={split_name}: {csv_path}"
                )
            frame = pd.read_csv(
                csv_path,
                dtype={"ID": "string", "patient_group_id": "string"},
                encoding="utf-8-sig",
            )
            missing_columns = sorted(REQUIRED_COLUMNS.difference(frame.columns))
            if missing_columns:
                raise ValueError(
                    f"{csv_path} is missing required columns: {missing_columns}"
                )
            if frame.empty:
                raise ValueError(f"Fold CSV contains no rows: {csv_path}")
            labels = pd.to_numeric(frame["label_3class"], errors="coerce")
            if labels.isna().any() or not labels.isin([0, 1, 2]).all():
                raise ValueError(f"Invalid label_3class values in {csv_path}")
            if split_name == "val" and not (frame["fold"] == fold).all():
                raise ValueError(
                    f"Validation CSV contains rows not assigned to fold {fold}: {csv_path}"
                )
            if split_name == "train" and (frame["fold"] == fold).any():
                raise ValueError(
                    f"Training CSV leaks held-out validation fold {fold}: {csv_path}"
                )
            expected_labels = pd.to_numeric(frame["NYHA"], errors="coerce").map(
                {0: 0, 1: 1, 2: 1, 3: 2, 4: 2}
            )
            if expected_labels.isna().any() or not (
                labels.astype("int64") == expected_labels.astype("int64")
            ).all():
                raise ValueError(
                    f"NYHA to label_3class mapping is inconsistent in {csv_path}"
                )
            expected_class_names = labels.astype("int64").map(
                {0: "normal", 1: "mild", 2: "severe"}
            )
            if not (
                frame["label_3class_name"].astype(str) == expected_class_names
            ).all():
                raise ValueError(
                    f"label_3class_name is inconsistent in {csv_path}"
                )
            expected_sex_names = pd.to_numeric(
                frame["SEX"], errors="coerce"
            ).map({0: "female", 1: "male"})
            if expected_sex_names.isna().any() or not (
                frame["sex_name"].astype(str) == expected_sex_names
            ).all():
                raise ValueError(f"SEX/sex_name mapping is inconsistent in {csv_path}")
            fold_frames[split_name] = frame
            for row in frame.itertuples(index=False):
                image_path = Path(str(row.image_path)).expanduser()
                if not image_path.is_file():
                    missing_images.append((str(row.ID), str(image_path)))
        train_groups = set(fold_frames["train"]["patient_group_id"].astype(str))
        val_groups = set(fold_frames["val"]["patient_group_id"].astype(str))
        overlap = sorted(train_groups.intersection(val_groups))
        if overlap:
            raise ValueError(
                f"Patient-group leakage in fold {fold}: {overlap[:20]}"
            )
        validation_ids.extend(fold_frames["val"]["ID"].astype(str).tolist())

    if missing_images:
        preview = "\n".join(
            f"  ID={identifier}, path={path}"
            for identifier, path in missing_images[:20]
        )
        raise FileNotFoundError(
            f"{len(missing_images)} referenced images are missing. First entries:\n"
            f"{preview}"
        )
    duplicated_validation_ids = pd.Series(validation_ids)[
        pd.Series(validation_ids).duplicated(keep=False)
    ].unique()
    if len(duplicated_validation_ids):
        raise ValueError(
            "Samples occur in more than one validation fold: "
            f"{duplicated_validation_ids[:20].tolist()}"
        )
    print(f"Preflight passed: config={config_path}, folds={n_folds}")


def select_output_dir(config: dict) -> Path:
    root = Path(config["experiment"]["output_dir"]).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    name = str(config["experiment"]["name"])
    candidate = root / name
    if candidate.exists():
        candidate = root / f"{name}_{datetime.now():%Y%m%d_%H%M%S}"
    candidate.mkdir(parents=False, exist_ok=False)
    return candidate


def run_checked(command: list[str]) -> None:
    print("Running:", subprocess.list2cmdline(command))
    subprocess.run(command, cwd=PROJECT_ROOT, check=True)


def main() -> None:
    config_path = DEFAULT_CONFIG.resolve()
    config = load_config(config_path)
    preflight(config_path, config)
    experiment_dir = select_output_dir(config)

    train_script = PROJECT_ROOT / "scripts" / "train" / "train_nyha_3class_5fold.py"
    summary_script = (
        PROJECT_ROOT
        / "scripts"
        / "evaluate"
        / "summarize_nyha_3class_5fold.py"
    )
    run_checked(
        [
            sys.executable,
            str(train_script),
            "--config",
            str(config_path),
            "--output-dir",
            str(experiment_dir),
        ]
    )
    run_checked(
        [
            sys.executable,
            str(summary_script),
            "--experiment-dir",
            str(experiment_dir),
        ]
    )
    print(f"Experiment completed: {experiment_dir}")


if __name__ == "__main__":
    main()
