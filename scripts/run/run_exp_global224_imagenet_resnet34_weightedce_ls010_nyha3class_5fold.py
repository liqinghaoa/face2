"""One-command entry point for ResNet34 weighted CE label-smoothing experiment."""

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
    / "nyha_3class_global224_imagenet_resnet34_weightedce_ls010.yaml"
)
EXPECTED_EXPERIMENT_NAME = (
    "Global224_ImageNetResNet34_NYHA3Class_WeightedCE_LabelSmoothing010_5Fold"
)
REQUIRED_METADATA_COLUMNS = {
    "ID",
    "patient_group_id",
    "NYHA",
    "label_3class",
    "label_3class_name",
    "SEX",
    "sex_name",
    "fold",
}
IMAGE_PATH_COLUMN = "image_path"


def resolve_project_path(value: str | Path | None) -> Path | None:
    if value in {None, ""}:
        return None
    path = Path(value).expanduser()
    if path.is_absolute():
        return path.resolve()
    return (PROJECT_ROOT / path).resolve()


def resolve_image_path(config: dict, row: object) -> Path:
    image_root = resolve_project_path(config["data"].get("image_root"))
    if image_root is None:
        return Path(str(getattr(row, IMAGE_PATH_COLUMN))).expanduser()
    template = str(config["data"].get("image_filename_template", "{ID}.png"))
    return image_root / template.format(ID=str(getattr(row, "ID")))


def load_config(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(f"Label-smoothing config file does not exist: {path}")
    with path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError(f"Config root must be a mapping: {path}")
    return config


def validate_label_smoothing_config(config: dict) -> None:
    if config["experiment"]["name"] != EXPECTED_EXPERIMENT_NAME:
        raise ValueError(
            "Unexpected experiment name in label-smoothing config: "
            f"{config['experiment']['name']!r}"
        )
    expected_split_dir = (
        PROJECT_ROOT / "data" / "processed" / "splits"
    ).resolve()
    actual_split_dir = resolve_project_path(config["data"]["split_dir"])
    if actual_split_dir != expected_split_dir:
        raise ValueError(
            "This controlled experiment must use the fixed 522Data splits: "
            f"{expected_split_dir}; got {actual_split_dir}"
        )
    if str(config["model"]["backbone"]).lower() != "resnet34":
        raise ValueError(
            f"Label-smoothing runner requires backbone=resnet34, got "
            f"{config['model']['backbone']!r}"
        )
    if str(config["model"]["pretrained"]).lower() != "imagenet":
        raise ValueError(
            "Label-smoothing runner requires ImageNet pretrained weights, got "
            f"{config['model']['pretrained']!r}"
        )
    if int(config["model"]["num_classes"]) != 3:
        raise ValueError("Label-smoothing runner requires num_classes=3")
    if bool(config["model"].get("freeze_backbone", False)):
        raise ValueError("This controlled experiment requires freeze_backbone=false")

    loss_config = config.get("loss", {})
    label_smoothing = loss_config.get("label_smoothing", {}) or {}
    if loss_config.get("name") != "weighted_soft_cross_entropy":
        raise ValueError(
            "This experiment requires loss.name='weighted_soft_cross_entropy'"
        )
    if loss_config.get("class_weight") is not True:
        raise ValueError("This experiment requires fold-specific class weights")
    if label_smoothing.get("enabled") is not True:
        raise ValueError("This experiment requires label smoothing enabled")
    if float(label_smoothing.get("alpha")) != 0.1:
        raise ValueError("This experiment requires label_smoothing.alpha=0.1")
    if label_smoothing.get("mode") != "exclude_true_class":
        raise ValueError(
            "This experiment requires label_smoothing.mode='exclude_true_class'"
        )
    num_classes = int(config["data"]["num_classes"])
    true_value = 1.0 - float(label_smoothing.get("alpha"))
    other_value = float(label_smoothing.get("alpha")) / (num_classes - 1)
    if round(true_value, 8) != 0.9 or round(other_value, 8) != 0.05:
        raise ValueError(
            "Label smoothing must produce true class=0.90 and other classes=0.05 "
            f"for alpha=0.1, num_classes=3; got {true_value}, {other_value}"
        )

    expected_train_values = {
        "batch_size": 16,
        "epochs": 50,
        "optimizer": "AdamW",
        "lr": 0.0001,
        "weight_decay": 0.0001,
        "early_stopping_patience": 10,
        "monitor_metric": "macro_auc",
        "random_seed": 2026,
        "num_workers": 0,
        "pin_memory": False,
        "use_amp": False,
    }
    for key, expected in expected_train_values.items():
        actual = config["train"].get(key)
        if actual != expected:
            raise ValueError(
                f"Label-smoothing experiment requires train.{key}={expected!r}, "
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
                f"Label-smoothing experiment requires augmentation.{key}="
                f"{expected!r}, got {actual!r}"
            )


def preflight(config_path: Path, config: dict) -> None:
    validate_label_smoothing_config(config)
    split_dir = resolve_project_path(config["data"]["split_dir"])
    if split_dir is None:
        raise ValueError("data.split_dir must not be empty")
    if not split_dir.is_dir():
        raise FileNotFoundError(f"Split directory does not exist: {split_dir}")
    image_root = resolve_project_path(config["data"].get("image_root"))
    if image_root is not None and not image_root.is_dir():
        raise FileNotFoundError(f"Image root does not exist: {image_root}")

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
            required_columns = set(REQUIRED_METADATA_COLUMNS)
            if image_root is None:
                required_columns.add(IMAGE_PATH_COLUMN)
            missing_columns = sorted(required_columns.difference(frame.columns))
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
                image_path = resolve_image_path(config, row)
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

    validation_id_series = pd.Series(validation_ids)
    duplicated_validation_ids = validation_id_series[
        validation_id_series.duplicated(keep=False)
    ].unique()
    if len(duplicated_validation_ids):
        raise ValueError(
            "Samples occur in more than one validation fold: "
            f"{duplicated_validation_ids[:20].tolist()}"
        )
    print(
        "ResNet34 label-smoothing preflight passed: "
        f"config={config_path}, backbone={config['model']['backbone']}, "
        f"loss={config['loss']['name']}, alpha="
        f"{config['loss']['label_smoothing']['alpha']}, folds={n_folds}"
    )


def select_output_dir(config: dict) -> Path:
    root = resolve_project_path(config["experiment"]["output_dir"])
    if root is None:
        raise ValueError("experiment.output_dir must not be empty")
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
    print(f"ResNet34 label-smoothing experiment completed: {experiment_dir}")


if __name__ == "__main__":
    main()
