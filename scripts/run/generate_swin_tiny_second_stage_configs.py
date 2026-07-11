"""Generate Swin-Tiny second-stage training configs.

The generated configs are derived from the completed Swin-Tiny model
exploration config. Existing experiment results are not modified.
"""

from __future__ import annotations

import copy
import sys
from pathlib import Path
from typing import Any

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.experiment_utils import load_yaml, save_yaml  # noqa: E402


SOURCE_CONFIG = (
    PROJECT_ROOT
    / "config"
    / "train"
    / "model_exploration_imagenet_meanbg"
    / "nyha_3class_swin_t_imagenet_meanbg.yaml"
)
CONFIG_DIR = PROJECT_ROOT / "config" / "train" / "swin_tiny_second_stage"
OUTPUT_ROOT = PROJECT_ROOT / "experiments" / "swin_tiny_second_stage_500Data"
MANIFEST_PATH = CONFIG_DIR / "swin_tiny_second_stage_config_manifest.csv"

SPLIT_DIR = "data/processed/splits_500"
IMAGE_ROOT = "data/processed/global_face/preprocess_ablation/hybrid_imagenet_meanbg/images"


def _find_source_config() -> Path:
    if SOURCE_CONFIG.is_file():
        return SOURCE_CONFIG
    candidates = sorted(
        (PROJECT_ROOT / "config" / "train").rglob("*swin*t*imagenet*meanbg*.yaml")
    )
    if not candidates:
        raise FileNotFoundError(
            "Could not find original Swin-Tiny model-exploration config."
        )
    return candidates[0]


def _normalize_common(config: dict[str, Any], *, experiment_name: str) -> dict[str, Any]:
    cfg = copy.deepcopy(config)
    cfg.setdefault("experiment", {})
    cfg["experiment"]["name"] = experiment_name
    cfg["experiment"]["output_dir"] = "experiments/swin_tiny_second_stage_500Data"

    cfg.setdefault("data", {})
    cfg["data"]["split_dir"] = SPLIT_DIR
    cfg["data"]["image_root"] = IMAGE_ROOT
    cfg["data"]["image_filename_template"] = "{ID}.png"
    cfg["data"]["n_folds"] = 5
    cfg["data"]["image_size"] = 224
    cfg["data"]["num_classes"] = 3
    cfg["data"]["label_col"] = "label_3class"
    cfg["data"]["train_csv_pattern"] = "fold_{fold}_train.csv"
    cfg["data"]["val_csv_pattern"] = "fold_{fold}_val.csv"

    cfg.setdefault("model", {})
    cfg["model"]["backbone"] = "swin_t"
    cfg["model"]["pretrained"] = "imagenet"
    cfg["model"]["num_classes"] = 3
    cfg["model"]["freeze_backbone"] = False

    cfg.setdefault("train", {})
    cfg["train"]["batch_size"] = 8
    cfg["train"]["epochs"] = 50
    cfg["train"]["optimizer"] = "AdamW"
    cfg["train"]["weight_decay"] = 0.0001
    cfg["train"]["early_stopping_patience"] = 10
    cfg["train"]["monitor_metric"] = "macro_auc"
    cfg["train"]["random_seed"] = 2026
    cfg["train"]["num_workers"] = 0
    cfg["train"]["pin_memory"] = False
    cfg["train"]["use_amp"] = False

    cfg.setdefault("augmentation", {})
    cfg["augmentation"]["horizontal_flip"] = True
    cfg["augmentation"]["color_jitter"] = False
    cfg["augmentation"]["random_crop"] = False

    cfg.setdefault("normalize", {})
    cfg["normalize"]["mean"] = [0.485, 0.456, 0.406]
    cfg["normalize"]["std"] = [0.229, 0.224, 0.225]
    return cfg


def _lr5e5_config(source: dict[str, Any]) -> dict[str, Any]:
    cfg = _normalize_common(
        source,
        experiment_name="SwinTiny_ImageNetMeanBG_LR5e5_WeightedCE_5Fold",
    )
    cfg.pop("loss", None)
    cfg["train"]["lr"] = 0.00005
    cfg["train"]["loss"] = "weighted_cross_entropy"
    return cfg


def _ls005_config(source: dict[str, Any]) -> dict[str, Any]:
    cfg = _normalize_common(
        source,
        experiment_name="SwinTiny_ImageNetMeanBG_WeightedSoftCE_LS005_5Fold",
    )
    cfg["train"]["lr"] = 0.0001
    cfg["train"].pop("loss", None)
    cfg["loss"] = {
        "name": "weighted_soft_cross_entropy",
        "class_weight": True,
        "label_smoothing": {
            "enabled": True,
            "alpha": 0.05,
            "mode": "exclude_true_class",
        },
    }
    return cfg


def _manifest_rows(configs: list[tuple[str, str, Path, dict[str, Any]]]) -> pd.DataFrame:
    rows = []
    for job_id, experiment_key, path, cfg in configs:
        loss_name = cfg.get("loss", {}).get("name") if isinstance(cfg.get("loss"), dict) else cfg["train"].get("loss")
        label_smoothing = ""
        if isinstance(cfg.get("loss"), dict):
            label_smoothing = cfg["loss"].get("label_smoothing", {}).get("alpha", "")
        rows.append(
            {
                "job_id": job_id,
                "experiment_key": experiment_key,
                "config_path": str(path.relative_to(PROJECT_ROOT)).replace("\\", "/"),
                "experiment_name": cfg["experiment"]["name"],
                "output_root": cfg["experiment"]["output_dir"],
                "backbone": cfg["model"]["backbone"],
                "lr": cfg["train"]["lr"],
                "loss_name": loss_name,
                "label_smoothing": label_smoothing,
                "batch_size": cfg["train"]["batch_size"],
                "status": "PENDING",
                "error_message": "",
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

    source_path = _find_source_config()
    source = load_yaml(source_path)

    lr_path = CONFIG_DIR / "swin_tiny_imagenet_meanbg_lr5e5.yaml"
    ls_path = CONFIG_DIR / "swin_tiny_imagenet_meanbg_weightedsoftce_ls005.yaml"
    lr_cfg = _lr5e5_config(source)
    ls_cfg = _ls005_config(source)
    save_yaml(lr_cfg, lr_path)
    save_yaml(ls_cfg, ls_path)

    manifest = _manifest_rows(
        [
            ("01_lr5e5", "lr5e5", lr_path, lr_cfg),
            ("02_ls005", "ls005", ls_path, ls_cfg),
        ]
    )
    manifest.to_csv(MANIFEST_PATH, index=False, encoding="utf-8-sig")

    print(f"SOURCE_CONFIG={source_path}")
    print(f"LR5E5_CONFIG={lr_path}")
    print(f"LS005_CONFIG={ls_path}")
    print(f"MANIFEST={MANIFEST_PATH}")


if __name__ == "__main__":
    main()
