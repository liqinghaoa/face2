"""One-command entry point for ROI NYHA three-class experiments."""

from __future__ import annotations

import argparse
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_DIR = PROJECT_ROOT / "config" / "train" / "roi"
ROI_ROOT = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "roi_dataset"
    / "global_aligned_face_parsing_roi_final5_224_canvas_500"
    / "roi_masked"
)
SPLIT_DIR = PROJECT_ROOT / "data" / "processed" / "splits_500"
ROI_CHOICES = ("cheek_roi", "chin_roi", "eye_roi", "forehead_roi", "lip_roi")
BACKBONE_CHOICES = ("resnet18", "resnet34", "resnet50")
REQUIRED_COLUMNS = {
    "ID",
    "patient_group_id",
    "NYHA",
    "label_3class",
    "label_3class_name",
    "SEX",
    "sex_name",
    "fold",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--roi", choices=ROI_CHOICES, default=None)
    parser.add_argument("--backbone", choices=BACKBONE_CHOICES, default=None)
    return parser.parse_args()


def infer_config_path(args: argparse.Namespace) -> Path:
    if args.config is not None:
        return args.config.expanduser().resolve()
    if args.roi is None or args.backbone is None:
        raise ValueError("Provide --config or both --roi and --backbone")
    return (
        DEFAULT_CONFIG_DIR
        / f"nyha_3class_{args.roi}_{args.backbone}_weightedce.yaml"
    ).resolve()


def load_config(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(f"ROI config file does not exist: {path}")
    with path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError(f"Config root must be a mapping: {path}")
    return config


def _resolve_image_path(config: dict, identifier: str) -> Path:
    image_root = Path(config["data"]["image_root"]).expanduser().resolve()
    template = str(config["data"].get("image_filename_template", "{ID}.png"))
    return image_root / template.format(ID=identifier)


def validate_roi_config(config: dict) -> None:
    data_config = config["data"]
    model_config = config["model"]
    train_config = config["train"]

    split_dir = Path(data_config["split_dir"]).expanduser().resolve()
    if split_dir != SPLIT_DIR.resolve():
        raise ValueError(f"ROI experiments must use {SPLIT_DIR}, got {split_dir}")
    image_root = Path(data_config["image_root"]).expanduser().resolve()
    if image_root.parent != ROI_ROOT.resolve():
        raise ValueError(f"ROI image_root must be under {ROI_ROOT}, got {image_root}")
    if image_root.name not in ROI_CHOICES:
        raise ValueError(f"Unsupported ROI directory: {image_root.name}")
    if str(data_config.get("image_filename_template", "{ID}.png")) != "{ID}.png":
        raise ValueError("ROI experiments require image_filename_template='{ID}.png'")
    if int(data_config["n_folds"]) != 5:
        raise ValueError("ROI experiments require n_folds=5")
    if int(data_config["image_size"]) != 224:
        raise ValueError("ROI experiments require image_size=224")
    if int(data_config["num_classes"]) != 3:
        raise ValueError("ROI experiments require num_classes=3")

    backbone = str(model_config["backbone"]).lower()
    if backbone not in BACKBONE_CHOICES:
        raise ValueError(f"Unsupported ROI backbone: {backbone}")
    if str(model_config["pretrained"]).lower() != "imagenet":
        raise ValueError("ROI experiments require ImageNet pretrained weights")
    if int(model_config["num_classes"]) != 3:
        raise ValueError("ROI experiments require model.num_classes=3")
    if bool(model_config.get("freeze_backbone", False)):
        raise ValueError("ROI experiments require freeze_backbone=false")

    expected_train_values = {
        "batch_size": 16,
        "epochs": 50,
        "optimizer": "AdamW",
        "lr": 0.0001,
        "weight_decay": 0.0001,
        "loss": "weighted_cross_entropy",
        "early_stopping_patience": 10,
        "monitor_metric": "macro_auc",
        "random_seed": 2026,
        "num_workers": 0,
        "pin_memory": False,
        "use_amp": False,
    }
    for key, expected in expected_train_values.items():
        actual = train_config.get(key)
        if actual != expected:
            raise ValueError(
                f"ROI controlled experiment requires train.{key}={expected!r}, "
                f"got {actual!r}"
            )

    expected_augmentation = {
        "horizontal_flip": True,
        "color_jitter": False,
        "random_crop": False,
    }
    for key, expected in expected_augmentation.items():
        actual = config["augmentation"].get(key)
        if actual is not expected:
            raise ValueError(
                f"ROI controlled experiment requires augmentation.{key}="
                f"{expected!r}, got {actual!r}"
            )


def preflight(config_path: Path, config: dict) -> None:
    validate_roi_config(config)
    split_dir = Path(config["data"]["split_dir"]).expanduser().resolve()
    image_root = Path(config["data"]["image_root"]).expanduser().resolve()
    if not split_dir.is_dir():
        raise FileNotFoundError(f"Split directory does not exist: {split_dir}")
    if not image_root.is_dir():
        raise FileNotFoundError(f"ROI image root does not exist: {image_root}")

    n_folds = int(config["data"]["n_folds"])
    missing_images: list[tuple[str, str]] = []
    validation_ids: list[str] = []
    all_ids: set[str] = set()
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
                identifier = str(row.ID)
                image_path = _resolve_image_path(config, identifier)
                all_ids.add(identifier)
                if not image_path.is_file():
                    missing_images.append((identifier, str(image_path)))

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
            f"{len(missing_images)} referenced ROI images are missing. "
            f"First entries:\n{preview}"
        )

    validation_id_series = pd.Series(validation_ids)
    duplicated_validation_ids = validation_id_series[
        validation_id_series.duplicated(keep=False)
    ].unique()
    if len(duplicated_validation_ids):
        raise ValueError(
            "Samples occur in more than one validation fold: "
            f"{duplicated_validation_ids[:20].tolist()}"
        )

    roi_file_ids = {path.stem for path in image_root.glob("*.png")}
    extra = sorted(roi_file_ids.difference(all_ids))
    if extra:
        raise ValueError(
            f"ROI directory contains {len(extra)} files not present in splits_500: "
            f"{extra[:20]}"
        )

    print(
        "ROI preflight passed: "
        f"config={config_path}, roi={image_root.name}, "
        f"backbone={config['model']['backbone']}, folds={n_folds}, samples={len(all_ids)}"
    )


def select_output_dir(config: dict) -> Path:
    root = Path(config["experiment"]["output_dir"]).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    candidate = root / str(config["experiment"]["name"])
    if candidate.exists():
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        candidate = root / f"{candidate.name}_{timestamp}"
    candidate.mkdir(parents=False, exist_ok=False)
    return candidate


def run_checked(command: list[str]) -> None:
    print("Running:", subprocess.list2cmdline(command))
    subprocess.run(command, cwd=PROJECT_ROOT, check=True)


def main() -> None:
    args = parse_args()
    config_path = infer_config_path(args)
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
    print(f"ROI experiment completed: {experiment_dir}")


if __name__ == "__main__":
    main()
