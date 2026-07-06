"""Generate model-exploration configs for the fixed ImageNet mean-bg dataset."""

from __future__ import annotations

import copy
import csv
import sys
from pathlib import Path

import yaml
from torchvision import models


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


TEMPLATE_CONFIG = (
    PROJECT_ROOT / "config" / "train" / "nyha_3class_global224_imagenet_resnet18.yaml"
)
OUTPUT_DIR = PROJECT_ROOT / "config" / "train" / "model_exploration_imagenet_meanbg"
MANIFEST_PATH = OUTPUT_DIR / "model_exploration_config_manifest.csv"
RUN_SCRIPT = PROJECT_ROOT / "scripts" / "run" / "run_nyha3class_5fold_with_config.py"

SPLIT_DIR = "data/processed/splits_500"
IMAGE_ROOT = (
    "data/processed/global_face/preprocess_ablation/"
    "hybrid_imagenet_meanbg/images"
)
EXPERIMENT_OUTPUT_ROOT = "experiments/model_exploration_500Data"

JOBS = [
    {
        "job_id": "01_densenet121",
        "backbone": "densenet121",
        "experiment_name": "ModelExploration_DenseNet121_ImageNetMeanBG",
        "config_filename": "nyha_3class_densenet121_imagenet_meanbg.yaml",
        "batch_size": 16,
    },
    {
        "job_id": "02_efficientnet_b0",
        "backbone": "efficientnet_b0",
        "experiment_name": "ModelExploration_EfficientNetB0_ImageNetMeanBG",
        "config_filename": "nyha_3class_efficientnet_b0_imagenet_meanbg.yaml",
        "batch_size": 16,
    },
    {
        "job_id": "03_convnext_tiny",
        "backbone": "convnext_tiny",
        "experiment_name": "ModelExploration_ConvNeXtTiny_ImageNetMeanBG",
        "config_filename": "nyha_3class_convnext_tiny_imagenet_meanbg.yaml",
        "batch_size": 8,
    },
    {
        "job_id": "04_swin_t",
        "backbone": "swin_t",
        "experiment_name": "ModelExploration_SwinTiny_ImageNetMeanBG",
        "config_filename": "nyha_3class_swin_t_imagenet_meanbg.yaml",
        "batch_size": 8,
    },
    {
        "job_id": "05_mobilenet_v3_large",
        "backbone": "mobilenet_v3_large",
        "experiment_name": "ModelExploration_MobileNetV3Large_ImageNetMeanBG",
        "config_filename": "nyha_3class_mobilenet_v3_large_imagenet_meanbg.yaml",
        "batch_size": 16,
    },
]

FIELDNAMES = [
    "job_id",
    "backbone",
    "config_path",
    "image_root",
    "experiment_name",
    "output_root",
    "batch_size",
    "supported",
    "status",
    "error_message",
]


def _project_relative(path: Path) -> str:
    return path.resolve().relative_to(PROJECT_ROOT).as_posix()


def _load_template() -> dict:
    if not TEMPLATE_CONFIG.is_file():
        raise FileNotFoundError(f"Template config does not exist: {TEMPLATE_CONFIG}")
    with TEMPLATE_CONFIG.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError(f"Template config root must be a mapping: {TEMPLATE_CONFIG}")
    return config


def _check_common_inputs() -> None:
    split_dir = PROJECT_ROOT / SPLIT_DIR
    image_root = PROJECT_ROOT / IMAGE_ROOT
    if not split_dir.is_dir():
        raise FileNotFoundError(f"Split directory does not exist: {split_dir}")
    if not image_root.is_dir():
        raise FileNotFoundError(f"Image root does not exist: {image_root}")
    if not RUN_SCRIPT.is_file():
        raise FileNotFoundError(f"Run script does not exist: {RUN_SCRIPT}")


def _torchvision_supports(backbone: str) -> tuple[bool, str]:
    if getattr(models, backbone, None) is None:
        return False, f"torchvision.models has no constructor named {backbone}"
    return True, ""


def _build_config(template: dict, job: dict) -> dict:
    config = copy.deepcopy(template)
    config.pop("loss", None)

    config.setdefault("experiment", {})
    config["experiment"]["name"] = job["experiment_name"]
    config["experiment"]["output_dir"] = EXPERIMENT_OUTPUT_ROOT

    config.setdefault("data", {})
    config["data"].update(
        {
            "split_dir": SPLIT_DIR,
            "image_root": IMAGE_ROOT,
            "image_filename_template": "{ID}.png",
            "n_folds": 5,
            "image_size": 224,
            "num_classes": 3,
            "label_col": "label_3class",
            "train_csv_pattern": "fold_{fold}_train.csv",
            "val_csv_pattern": "fold_{fold}_val.csv",
        }
    )

    config.setdefault("model", {})
    config["model"].update(
        {
            "backbone": job["backbone"],
            "pretrained": "imagenet",
            "num_classes": 3,
            "freeze_backbone": False,
        }
    )
    config["model"].pop("type", None)
    config["model"].pop("fusion_method", None)
    config["model"].pop("shared_backbone", None)
    config["model"].pop("fusion_head", None)

    config.setdefault("train", {})
    config["train"].update(
        {
            "batch_size": int(job["batch_size"]),
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
    )

    config["augmentation"] = {
        "horizontal_flip": True,
        "color_jitter": False,
        "random_crop": False,
    }
    config["normalize"] = {
        "mean": [0.485, 0.456, 0.406],
        "std": [0.229, 0.224, 0.225],
    }
    config["metrics"] = {
        "main": [
            "macro_auc",
            "accuracy",
            "macro_precision",
            "macro_recall",
            "macro_f1",
            "balanced_accuracy",
        ],
        "auxiliary": [
            "per_class_auc",
            "per_class_precision",
            "per_class_recall",
            "per_class_f1",
            "severe_vs_rest_auc",
            "normal_vs_abnormal_auc",
            "confusion_matrix",
        ],
    }
    return config


def main() -> Path:
    _check_common_inputs()
    template = _load_template()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, object]] = []
    for job in JOBS:
        config_path = OUTPUT_DIR / job["config_filename"]
        supported, error_message = _torchvision_supports(str(job["backbone"]))
        if supported:
            config = _build_config(template, job)
            with config_path.open("w", encoding="utf-8") as handle:
                yaml.safe_dump(config, handle, sort_keys=False, allow_unicode=True)
            status = "PENDING"
        else:
            status = "UNSUPPORTED"

        rows.append(
            {
                "job_id": job["job_id"],
                "backbone": job["backbone"],
                "config_path": _project_relative(config_path),
                "image_root": IMAGE_ROOT,
                "experiment_name": job["experiment_name"],
                "output_root": EXPERIMENT_OUTPUT_ROOT,
                "batch_size": job["batch_size"],
                "supported": str(bool(supported)).lower(),
                "status": status,
                "error_message": error_message,
            }
        )

    with MANIFEST_PATH.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Generated manifest: {MANIFEST_PATH}")
    for row in rows:
        print(
            f"{row['job_id']},{row['backbone']},{row['status']},"
            f"config={row['config_path']}"
        )
    return MANIFEST_PATH


if __name__ == "__main__":
    main()
