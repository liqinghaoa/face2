"""One-command entry point for configurable Multi-ROI fusion experiments."""

from __future__ import annotations

import argparse
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.experiment_utils import resolve_project_path  # noqa: E402


ALLOWED_ROIS = {"cheek_roi", "chin_roi", "eye_roi", "forehead_roi", "lip_roi"}
ALLOWED_BACKBONES = {"resnet18", "resnet34", "resnet50"}
REQUIRED_COLUMNS = {"ID", "patient_group_id", "NYHA", "label_3class", "SEX", "fold"}
EXPECTED_SPLIT_DIR = (PROJECT_ROOT / "data" / "processed" / "splits_500").resolve()
EXPECTED_ROI_ROOT = (
    PROJECT_ROOT / "data" / "processed" / "roi_dataset" / "manual_shift_data"
).resolve()
EXPECTED_SAMPLE_COUNT = 500


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    return parser.parse_args()


def load_config(path: Path) -> dict:
    config_path = resolve_project_path(path)
    if config_path is None or not config_path.is_file():
        raise FileNotFoundError(f"ROI fusion config file does not exist: {path}")
    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError(f"Config root must be a mapping: {config_path}")
    return config


def _resolve_image_path(config: dict, roi_name: str, identifier: str) -> Path:
    roi_root = resolve_project_path(config["data"]["roi_root"])
    if roi_root is None:
        raise ValueError("data.roi_root must not be empty")
    template = str(config["data"].get("image_filename_template", "{ID}.png"))
    return roi_root / roi_name / template.format(ID=identifier)


def validate_config(config: dict) -> None:
    if not str(config["experiment"].get("name", "")).strip():
        raise ValueError("experiment.name must not be empty")

    data_config = config["data"]
    split_dir = resolve_project_path(data_config["split_dir"])
    roi_root = resolve_project_path(data_config["roi_root"])
    if split_dir != EXPECTED_SPLIT_DIR:
        raise ValueError(f"data.split_dir must be {EXPECTED_SPLIT_DIR}, got {split_dir}")
    if roi_root != EXPECTED_ROI_ROOT:
        raise ValueError(f"data.roi_root must be {EXPECTED_ROI_ROOT}, got {roi_root}")
    roi_names = list(data_config.get("roi_names", []))
    if len(roi_names) < 2:
        raise ValueError("data.roi_names must contain at least two ROIs")
    if len(set(roi_names)) != len(roi_names):
        raise ValueError(f"data.roi_names contains duplicates: {roi_names}")
    unsupported = sorted(set(roi_names).difference(ALLOWED_ROIS))
    if unsupported:
        raise ValueError(f"Unsupported ROI names: {unsupported}")
    if str(data_config.get("image_filename_template")) != "{ID}.png":
        raise ValueError("data.image_filename_template must be '{ID}.png'")
    if int(data_config["n_folds"]) != 5:
        raise ValueError("data.n_folds must be 5")
    if int(data_config["image_size"]) != 224:
        raise ValueError("data.image_size must be 224")
    if int(data_config["num_classes"]) != 3:
        raise ValueError("data.num_classes must be 3")
    if int(data_config.get("expected_num_samples", -1)) != EXPECTED_SAMPLE_COUNT:
        raise ValueError(
            f"data.expected_num_samples must be {EXPECTED_SAMPLE_COUNT}"
        )

    model_config = config["model"]
    if str(model_config.get("type")) != "multi_roi_fusion":
        raise ValueError("model.type must be 'multi_roi_fusion'")
    if str(model_config["backbone"]).lower() not in ALLOWED_BACKBONES:
        raise ValueError(f"Unsupported backbone: {model_config['backbone']}")
    if str(model_config["pretrained"]).lower() != "imagenet":
        raise ValueError("model.pretrained must be 'imagenet'")
    if model_config.get("shared_backbone") is not True:
        raise ValueError("model.shared_backbone must be true")
    if str(model_config.get("fusion_method")).lower() != "concat":
        raise ValueError("model.fusion_method must be 'concat'")
    if int(model_config["num_classes"]) != 3:
        raise ValueError("model.num_classes must be 3")
    if bool(model_config.get("freeze_backbone", False)):
        raise ValueError("model.freeze_backbone must be false")

    loss_config = config.get("loss", {})
    if loss_config.get("name") != "weighted_cross_entropy":
        raise ValueError("loss.name must be 'weighted_cross_entropy'")
    if loss_config.get("class_weight") is not True:
        raise ValueError("loss.class_weight must be true")

    if int(config["train"].get("num_workers", -1)) != 0:
        raise ValueError(
            "train.num_workers must be 0 so synchronized ROI augmentation and "
            "checkpoint RNG restoration remain exactly reproducible on Windows"
        )

    expected_aug = {
        "horizontal_flip": True,
        "same_flip_for_all_rois": True,
        "color_jitter": False,
        "random_crop": False,
    }
    for key, expected in expected_aug.items():
        actual = config["augmentation"].get(key)
        if actual is not expected:
            raise ValueError(f"augmentation.{key} must be {expected!r}, got {actual!r}")


def preflight(config: dict) -> None:
    validate_config(config)
    split_dir = resolve_project_path(config["data"]["split_dir"])
    roi_root = resolve_project_path(config["data"]["roi_root"])
    if split_dir is None or not split_dir.is_dir():
        raise FileNotFoundError(f"Split directory does not exist: {split_dir}")
    if roi_root is None or not roi_root.is_dir():
        raise FileNotFoundError(f"ROI root does not exist: {roi_root}")

    roi_names = list(config["data"]["roi_names"])
    for roi_name in roi_names:
        roi_dir = roi_root / roi_name
        if not roi_dir.is_dir():
            raise FileNotFoundError(f"ROI directory does not exist: {roi_dir}")

    n_folds = int(config["data"]["n_folds"])
    validation_ids: list[str] = []
    split_ids: set[str] = set()
    missing_images: list[tuple[str, str, str]] = []
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
                raise ValueError(f"{csv_path} is missing required columns: {missing_columns}")
            if frame.empty:
                raise ValueError(f"Fold CSV contains no rows: {csv_path}")
            duplicated_ids = frame.loc[
                frame["ID"].duplicated(keep=False), "ID"
            ].astype(str)
            if not duplicated_ids.empty:
                raise ValueError(
                    f"Duplicate IDs in {csv_path}: "
                    f"{duplicated_ids.unique()[:20].tolist()}"
                )
            labels = pd.to_numeric(frame["label_3class"], errors="coerce")
            if labels.isna().any() or not labels.isin([0, 1, 2]).all():
                raise ValueError(f"Invalid label_3class values in {csv_path}")
            expected_labels = pd.to_numeric(frame["NYHA"], errors="coerce").map(
                {0: 0, 1: 1, 2: 1, 3: 2, 4: 2}
            )
            if expected_labels.isna().any() or not (
                labels.astype("int64") == expected_labels.astype("int64")
            ).all():
                raise ValueError(f"NYHA to label_3class mapping is inconsistent in {csv_path}")
            if "label_3class_name" in frame.columns:
                expected_names = labels.astype("int64").map(
                    {0: "normal", 1: "mild", 2: "severe"}
                )
                if not (frame["label_3class_name"].astype(str) == expected_names).all():
                    raise ValueError(f"label_3class_name is inconsistent in {csv_path}")
            if "sex_name" in frame.columns:
                expected_sex_names = pd.to_numeric(frame["SEX"], errors="coerce").map(
                    {0: "female", 1: "male"}
                )
                if expected_sex_names.isna().any() or not (
                    frame["sex_name"].astype(str) == expected_sex_names
                ).all():
                    raise ValueError(f"SEX/sex_name mapping is inconsistent in {csv_path}")
            if split_name == "val" and not (frame["fold"] == fold).all():
                raise ValueError(
                    f"Validation CSV contains rows not assigned to fold {fold}: {csv_path}"
                )
            if split_name == "train" and (frame["fold"] == fold).any():
                raise ValueError(
                    f"Training CSV leaks held-out validation fold {fold}: {csv_path}"
                )

            fold_frames[split_name] = frame
            for row in frame.itertuples(index=False):
                identifier = str(row.ID)
                split_ids.add(identifier)
                for roi_name in roi_names:
                    image_path = _resolve_image_path(config, roi_name, identifier)
                    if not image_path.is_file():
                        missing_images.append((identifier, roi_name, str(image_path)))

        train_ids = set(fold_frames["train"]["ID"].astype(str))
        val_ids = set(fold_frames["val"]["ID"].astype(str))
        overlap_ids = sorted(train_ids.intersection(val_ids))
        if overlap_ids:
            raise ValueError(f"ID leakage in fold {fold}: {overlap_ids[:20]}")
        train_groups = set(fold_frames["train"]["patient_group_id"].astype(str))
        val_groups = set(fold_frames["val"]["patient_group_id"].astype(str))
        overlap_groups = sorted(train_groups.intersection(val_groups))
        if overlap_groups:
            raise ValueError(f"Patient-group leakage in fold {fold}: {overlap_groups[:20]}")
        validation_ids.extend(fold_frames["val"]["ID"].astype(str).tolist())

    if missing_images:
        preview = "\n".join(
            f"  ID={identifier}, roi={roi_name}, path={path}"
            for identifier, roi_name, path in missing_images[:20]
        )
        raise FileNotFoundError(
            f"{len(missing_images)} ROI images are missing. First entries:\n{preview}"
        )

    duplicated_validation_ids = pd.Series(validation_ids)[
        pd.Series(validation_ids).duplicated(keep=False)
    ].unique()
    if len(duplicated_validation_ids):
        raise ValueError(
            "Samples occur in more than one validation fold: "
            f"{duplicated_validation_ids[:20].tolist()}"
        )

    if len(split_ids) != EXPECTED_SAMPLE_COUNT:
        raise ValueError(
            f"Split files contain {len(split_ids)} unique IDs; "
            f"expected {EXPECTED_SAMPLE_COUNT}"
        )
    if len(validation_ids) != EXPECTED_SAMPLE_COUNT:
        raise ValueError(
            f"OOF validation rows total {len(validation_ids)}; "
            f"expected {EXPECTED_SAMPLE_COUNT}"
        )
    validation_id_set = set(validation_ids)
    missing_validation_ids = sorted(split_ids.difference(validation_id_set))
    unexpected_validation_ids = sorted(validation_id_set.difference(split_ids))
    if missing_validation_ids or unexpected_validation_ids:
        raise ValueError(
            "OOF validation IDs do not exactly match all split IDs. "
            f"Missing={missing_validation_ids[:20]}, "
            f"unexpected={unexpected_validation_ids[:20]}"
        )

    for roi_name in roi_names:
        roi_file_ids = {path.stem for path in (roi_root / roi_name).glob("*.png")}
        missing = sorted(split_ids.difference(roi_file_ids))
        extra = sorted(roi_file_ids.difference(split_ids))
        if missing:
            raise FileNotFoundError(
                f"{roi_name} is missing {len(missing)} split IDs: {missing[:20]}"
            )
        if extra:
            raise ValueError(
                f"{roi_name} contains {len(extra)} PNG files outside splits_500: {extra[:20]}"
            )

    print(
        "ROI fusion preflight passed: "
        f"experiment={config['experiment']['name']}, rois={roi_names}, "
        f"backbone={config['model']['backbone']}, samples={len(split_ids)}"
    )


def select_output_dir(config: dict) -> Path:
    root = resolve_project_path(config["experiment"]["output_dir"])
    if root is None:
        raise ValueError("experiment.output_dir must not be empty")
    root.mkdir(parents=True, exist_ok=True)
    name = str(config["experiment"]["name"])
    candidate = root / name
    if candidate.exists():
        candidate = root / f"{name}_{datetime.now():%Y%m%d_%H%M%S}"
    candidate.mkdir(parents=False, exist_ok=False)
    return candidate


def run_checked(command: list[str], log_path: Path) -> None:
    line = "Running: " + subprocess.list2cmdline(command)
    print(line)
    with log_path.open("a", encoding="utf-8") as handle:
        handle.write(line + "\n")
        handle.flush()
        subprocess.run(
            command,
            cwd=PROJECT_ROOT,
            check=True,
            stdout=handle,
            stderr=subprocess.STDOUT,
        )


def main() -> None:
    args = parse_args()
    config_path = resolve_project_path(args.config)
    if config_path is None:
        raise ValueError("--config must not be empty")
    config = load_config(config_path)
    preflight(config)
    experiment_dir = select_output_dir(config)
    log_path = experiment_dir / "run_5fold.log"

    train_script = PROJECT_ROOT / "scripts" / "train" / "train_nyha_3class_5fold.py"
    summary_script = (
        PROJECT_ROOT / "scripts" / "evaluate" / "summarize_nyha_3class_5fold.py"
    )
    run_checked(
        [
            sys.executable,
            str(train_script),
            "--config",
            str(config_path),
            "--output-dir",
            str(experiment_dir),
        ],
        log_path,
    )
    run_checked(
        [
            sys.executable,
            str(summary_script),
            "--experiment-dir",
            str(experiment_dir),
        ],
        log_path,
    )
    print(f"ROI fusion experiment completed: {experiment_dir}")


if __name__ == "__main__":
    main()
