"""Generate configs for Global + selected ROI fusion experiments."""

from __future__ import annotations

import csv
import sys
from pathlib import Path
from typing import Any

import pandas as pd
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.experiment_utils import resolve_project_path  # noqa: E402


CONFIG_DIR = PROJECT_ROOT / "config" / "train" / "global_roi_fusion"
MANIFEST_PATH = CONFIG_DIR / "global_roi_fusion_config_manifest.csv"
GLOBAL_IMAGE_ROOT = (
    "data/processed/global_face/preprocess_ablation/hybrid_imagenet_meanbg/images"
)
SPLIT_DIR = "data/processed/splits_500"
OUTPUT_ROOT = "experiments/global_roi_fusion_500Data"
ROI_SEARCH_DIRS = [
    PROJECT_ROOT / "config" / "train" / "roi",
    PROJECT_ROOT / "config" / "train" / "roi_single",
    PROJECT_ROOT / "config" / "train" / "roi_fusion",
]
EXPERIMENT_CONFIG_GLOBS = [
    PROJECT_ROOT / "experiments" / "ROI_500",
    PROJECT_ROOT / "experiments" / "ROI_Fusion_500",
]


def _load_yaml(path: Path) -> dict[str, Any] | None:
    try:
        with path.open("r", encoding="utf-8") as handle:
            data = yaml.safe_load(handle)
        return data if isinstance(data, dict) else None
    except Exception:
        return None


def _relpath(path: Path) -> str:
    try:
        return path.resolve().relative_to(PROJECT_ROOT.resolve()).as_posix()
    except ValueError:
        return str(path.resolve())


def _roi_key_from_text(text: str) -> str | None:
    normalized = text.lower()
    if "eye_roi" in normalized or "roi_eye" in normalized or "eye" in normalized:
        return "eye"
    if "cheek_roi" in normalized or "roi_cheek" in normalized or "cheek" in normalized:
        return "cheek"
    return None


def _candidate_config_paths() -> list[Path]:
    paths: list[Path] = []
    for directory in ROI_SEARCH_DIRS:
        if directory.is_dir():
            paths.extend(sorted(directory.glob("*.yaml")))
            paths.extend(sorted(directory.glob("*.yml")))
    for root in EXPERIMENT_CONFIG_GLOBS:
        if root.is_dir():
            paths.extend(sorted(root.glob("*/config.yaml")))
    return paths


def locate_roi_roots() -> dict[str, Path]:
    """Locate eye/cheek ROI roots from existing ROI configs and experiments."""

    found: dict[str, Path] = {}
    for path in _candidate_config_paths():
        config = _load_yaml(path)
        if not config:
            continue
        data = config.get("data", {}) or {}
        experiment_name = str((config.get("experiment", {}) or {}).get("name", ""))

        image_root = data.get("image_root")
        if image_root:
            key = _roi_key_from_text(" ".join([path.name, experiment_name, str(image_root)]))
            if key in {"eye", "cheek"} and key not in found:
                resolved = resolve_project_path(image_root)
                if resolved is not None and resolved.is_dir():
                    found[key] = resolved

        roi_root = data.get("roi_root")
        roi_names = data.get("roi_names") or []
        if roi_root and roi_names:
            root = resolve_project_path(roi_root)
            if root is not None:
                for roi_name in roi_names:
                    key = _roi_key_from_text(str(roi_name))
                    if key in {"eye", "cheek"} and key not in found:
                        candidate = root / str(roi_name)
                        if candidate.is_dir():
                            found[key] = candidate.resolve()

    missing = sorted({"eye", "cheek"}.difference(found))
    if missing:
        raise FileNotFoundError(
            "Could not locate eye/cheek ROI image root. Missing: "
            + ", ".join(missing)
        )
    return found


def _base_config(
    experiment_name: str,
    enabled_inputs: list[str],
    roi_roots: dict[str, Path],
) -> dict[str, Any]:
    return {
        "experiment": {
            "name": experiment_name,
            "output_dir": OUTPUT_ROOT,
        },
        "data": {
            "split_dir": SPLIT_DIR,
            "global_image_root": GLOBAL_IMAGE_ROOT,
            "roi_roots": {
                "eye": _relpath(roi_roots["eye"]),
                "cheek": _relpath(roi_roots["cheek"]),
            },
            "enabled_inputs": enabled_inputs,
            "image_filename_template": "{ID}.png",
            "n_folds": 5,
            "image_size": 224,
            "num_classes": 3,
            "label_col": "label_3class",
            "train_csv_pattern": "fold_{fold}_train.csv",
            "val_csv_pattern": "fold_{fold}_val.csv",
        },
        "model": {
            "type": "global_roi_fusion",
            "backbone": "resnet18",
            "pretrained": "imagenet",
            "num_classes": 3,
            "freeze_backbone": False,
            "fusion_type": "concat",
            "projection_dim": 256,
            "dropout": 0.3,
        },
        "train": {
            "batch_size": 8,
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
        },
        "augmentation": {
            "horizontal_flip": True,
            "synchronized_horizontal_flip": True,
            "color_jitter": False,
            "random_crop": False,
            "random_rotation": False,
        },
        "normalize": {
            "mean": [0.485, 0.456, 0.406],
            "std": [0.229, 0.224, 0.225],
        },
        "metrics": {
            "main": [
                "macro_auc",
                "accuracy",
                "macro_precision",
                "macro_recall",
                "macro_f1",
                "weighted_f1",
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
        },
    }


def _preflight_static_paths(roi_roots: dict[str, Path]) -> None:
    global_root = resolve_project_path(GLOBAL_IMAGE_ROOT)
    split_root = resolve_project_path(SPLIT_DIR)
    if global_root is None or not global_root.is_dir():
        raise FileNotFoundError(f"Global image root does not exist: {global_root}")
    if split_root is None or not split_root.is_dir():
        raise FileNotFoundError(f"Split directory does not exist: {split_root}")
    for fold in range(5):
        for suffix in ("train", "val"):
            path = split_root / f"fold_{fold}_{suffix}.csv"
            if not path.is_file():
                raise FileNotFoundError(f"Missing split CSV: {path}")
            frame = pd.read_csv(path, nrows=5, encoding="utf-8-sig")
            missing = {"ID", "patient_group_id", "label_3class"}.difference(frame.columns)
            if missing:
                raise ValueError(f"Split CSV missing columns {sorted(missing)}: {path}")
    for key, path in roi_roots.items():
        if not path.is_dir():
            raise FileNotFoundError(f"{key} ROI root does not exist: {path}")


def main() -> Path:
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    roi_roots = locate_roi_roots()
    _preflight_static_paths(roi_roots)

    jobs = [
        {
            "job_id": "01_global_eye",
            "experiment_key": "global_eye",
            "filename": "global_roi_fusion_global_eye_resnet18.yaml",
            "experiment_name": "GlobalROIFusion_GlobalEye_ResNet18_WeightedCE_5Fold",
            "enabled_inputs": ["global", "eye"],
        },
        {
            "job_id": "02_global_cheek",
            "experiment_key": "global_cheek",
            "filename": "global_roi_fusion_global_cheek_resnet18.yaml",
            "experiment_name": "GlobalROIFusion_GlobalCheek_ResNet18_WeightedCE_5Fold",
            "enabled_inputs": ["global", "cheek"],
        },
        {
            "job_id": "03_global_eye_cheek",
            "experiment_key": "global_eye_cheek",
            "filename": "global_roi_fusion_global_eye_cheek_resnet18.yaml",
            "experiment_name": "GlobalROIFusion_GlobalEyeCheek_ResNet18_WeightedCE_5Fold",
            "enabled_inputs": ["global", "eye", "cheek"],
        },
    ]
    manifest_rows = []
    for job in jobs:
        config = _base_config(
            job["experiment_name"],
            job["enabled_inputs"],
            roi_roots,
        )
        config_path = CONFIG_DIR / job["filename"]
        with config_path.open("w", encoding="utf-8") as handle:
            yaml.safe_dump(config, handle, sort_keys=False, allow_unicode=True)
        manifest_rows.append(
            {
                "job_id": job["job_id"],
                "experiment_key": job["experiment_key"],
                "config_path": _relpath(config_path),
                "experiment_name": job["experiment_name"],
                "output_root": OUTPUT_ROOT,
                "enabled_inputs": ",".join(job["enabled_inputs"]),
                "global_image_root": GLOBAL_IMAGE_ROOT,
                "eye_root": _relpath(roi_roots["eye"]),
                "cheek_root": _relpath(roi_roots["cheek"]),
                "backbone": "resnet18",
                "batch_size": 8,
                "status": "PENDING",
                "error_message": "",
            }
        )

    with MANIFEST_PATH.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(manifest_rows[0].keys()))
        writer.writeheader()
        writer.writerows(manifest_rows)

    print(f"CONFIG_DIR={CONFIG_DIR}")
    print(f"MANIFEST={MANIFEST_PATH}")
    print(f"eye_root={roi_roots['eye']}")
    print(f"cheek_root={roi_roots['cheek']}")
    return MANIFEST_PATH


if __name__ == "__main__":
    main()
