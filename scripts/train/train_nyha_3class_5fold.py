"""Train and evaluate all five fixed NYHA validation folds."""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

# In the bundled Windows environment, importing scikit-learn first ensures its
# shared OpenMP runtime is initialized before PyTorch/torchvision.
import sklearn  # noqa: F401
import torch
from torch.utils.data import DataLoader


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from datasets.nyha_3class_face_dataset import (  # noqa: E402
    NYHA3ClassFaceDataset,
    build_transforms,
)
from datasets.nyha_3class_multi_roi_dataset import (  # noqa: E402
    NYHA3ClassMultiROIDataset,
)
from evaluators.nyha_3class_evaluator import NYHA3ClassEvaluator  # noqa: E402
from losses.classification_losses import (  # noqa: E402
    build_criterion,
    compute_class_weights,
)
from models.resnet_nyha_3class import build_resnet_nyha_model  # noqa: E402
from models.multi_roi_fusion_nyha_3class import (  # noqa: E402
    ConfigurableMultiROIFusionResNet,
)
from trainers.nyha_3class_trainer import NYHA3ClassTrainer  # noqa: E402
from utils.experiment_utils import (  # noqa: E402
    choose_device,
    configure_logging,
    create_experiment_dir,
    load_yaml,
    resolve_project_path,
    save_yaml,
    seed_worker,
    set_random_seed,
)


LOGGER = logging.getLogger("train_nyha_3class_5fold")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Use an already selected experiment directory.",
    )
    parser.add_argument(
        "--fold",
        type=int,
        action="append",
        dest="folds",
        help="Train only this fold; repeat to select multiple folds.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Override epochs for a controlled smoke test.",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=None,
        help="Override DataLoader workers.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Override DataLoader batch size.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume incomplete folds from fold_x/checkpoints/last.pth and skip completed folds.",
    )
    return parser.parse_args()


def _pretrained_enabled(value: object) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).lower() in {"imagenet", "true", "yes", "1"}


def _resolve_loss_settings(config: dict) -> dict:
    """Return a normalized loss config while preserving legacy train.loss configs."""
    if "loss" in config and isinstance(config["loss"], dict):
        loss_config = config["loss"]
        label_smoothing = loss_config.get("label_smoothing", {}) or {}
        return {
            "name": str(loss_config.get("name", "weighted_cross_entropy")),
            "class_weight": bool(loss_config.get("class_weight", True)),
            "label_smoothing_enabled": bool(label_smoothing.get("enabled", False)),
            "label_smoothing_alpha": float(label_smoothing.get("alpha", 0.0)),
            "label_smoothing_mode": str(
                label_smoothing.get("mode", "exclude_true_class")
            ),
        }

    return {
        "name": str(config["train"].get("loss", "weighted_cross_entropy")),
        "class_weight": True,
        "label_smoothing_enabled": False,
        "label_smoothing_alpha": 0.0,
        "label_smoothing_mode": "none",
    }


def _build_loader(
    dataset: NYHA3ClassFaceDataset | NYHA3ClassMultiROIDataset,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
    seed: int,
    pin_memory: bool,
) -> DataLoader:
    generator = torch.Generator()
    generator.manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=num_workers > 0,
        worker_init_fn=seed_worker,
        generator=generator,
    )


def _is_multi_roi_config(config: dict) -> bool:
    return str(config.get("model", {}).get("type", "")).lower() == "multi_roi_fusion"


def _build_datasets(
    config: dict,
    fold: int,
    split_dir: Path,
    train_transform,
    val_transform,
) -> tuple[NYHA3ClassFaceDataset | NYHA3ClassMultiROIDataset, NYHA3ClassFaceDataset | NYHA3ClassMultiROIDataset]:
    data_config = config["data"]
    train_csv = split_dir / data_config["train_csv_pattern"].format(fold=fold)
    val_csv = split_dir / data_config["val_csv_pattern"].format(fold=fold)
    if _is_multi_roi_config(config):
        roi_root = resolve_project_path(data_config["roi_root"])
        if roi_root is None:
            raise ValueError("data.roi_root must not be empty for multi_roi_fusion")
        return (
            NYHA3ClassMultiROIDataset(
                train_csv,
                roi_root=roi_root,
                roi_names=data_config["roi_names"],
                image_filename_template=str(
                    data_config.get("image_filename_template", "{ID}.png")
                ),
                image_size=int(data_config["image_size"]),
                label_col=str(data_config.get("label_col", "label_3class")),
                train=True,
                horizontal_flip=bool(config["augmentation"]["horizontal_flip"]),
                same_flip_for_all_rois=bool(
                    config["augmentation"].get("same_flip_for_all_rois", True)
                ),
                mean=config["normalize"]["mean"],
                std=config["normalize"]["std"],
            ),
            NYHA3ClassMultiROIDataset(
                val_csv,
                roi_root=roi_root,
                roi_names=data_config["roi_names"],
                image_filename_template=str(
                    data_config.get("image_filename_template", "{ID}.png")
                ),
                image_size=int(data_config["image_size"]),
                label_col=str(data_config.get("label_col", "label_3class")),
                train=False,
                horizontal_flip=False,
                same_flip_for_all_rois=True,
                mean=config["normalize"]["mean"],
                std=config["normalize"]["std"],
            ),
        )

    image_root = resolve_project_path(data_config.get("image_root"))
    image_filename_template = str(data_config.get("image_filename_template", "{ID}.png"))
    return (
        NYHA3ClassFaceDataset(
            train_csv,
            train_transform,
            image_root=image_root,
            image_filename_template=image_filename_template,
        ),
        NYHA3ClassFaceDataset(
            val_csv,
            val_transform,
            image_root=image_root,
            image_filename_template=image_filename_template,
        ),
    )


def _build_model(config: dict) -> torch.nn.Module:
    model_config = config["model"]
    if _is_multi_roi_config(config):
        fusion_head = model_config.get("fusion_head", {}) or {}
        return ConfigurableMultiROIFusionResNet(
            backbone=model_config["backbone"],
            pretrained=model_config["pretrained"],
            num_rois=len(config["data"]["roi_names"]),
            num_classes=int(model_config["num_classes"]),
            shared_backbone=bool(model_config["shared_backbone"]),
            fusion_method=str(model_config["fusion_method"]),
            hidden_dim=int(fusion_head.get("hidden_dim", 512)),
            dropout=float(fusion_head.get("dropout", 0.3)),
            use_batchnorm=bool(fusion_head.get("use_batchnorm", True)),
            freeze_backbone=bool(model_config.get("freeze_backbone", False)),
        )
    return build_resnet_nyha_model(
        backbone=model_config["backbone"],
        num_classes=int(model_config["num_classes"]),
        pretrained=_pretrained_enabled(model_config["pretrained"]),
    )


def main() -> Path:
    args = parse_args()
    config = load_yaml(args.config)
    if args.epochs is not None:
        if args.epochs < 1:
            raise ValueError("--epochs must be at least 1")
        config["train"]["epochs"] = args.epochs
    if args.num_workers is not None:
        if args.num_workers < 0:
            raise ValueError("--num-workers cannot be negative")
        config["train"]["num_workers"] = args.num_workers
    if args.batch_size is not None:
        if args.batch_size < 1:
            raise ValueError("--batch-size must be at least 1")
        config["train"]["batch_size"] = args.batch_size
    if bool(config["model"].get("freeze_backbone", False)):
        raise ValueError("This baseline requires freeze_backbone=false")
    if str(config["train"].get("monitor_metric", "macro_auc")) != "macro_auc":
        raise ValueError("This baseline requires monitor_metric=macro_auc")
    seed = int(config["train"]["random_seed"])
    set_random_seed(seed)
    device = choose_device()

    if args.output_dir is None:
        experiment_dir = create_experiment_dir(
            config["experiment"]["output_dir"],
            config["experiment"]["name"],
        )
    else:
        experiment_dir = resolve_project_path(args.output_dir)
        if experiment_dir is None:
            raise ValueError("--output-dir must not be empty")
        experiment_dir.mkdir(parents=True, exist_ok=True)

    configure_logging(experiment_dir / "experiment.log")
    save_yaml(config, experiment_dir / "config.yaml")
    LOGGER.info("Experiment directory: %s", experiment_dir)
    LOGGER.info("Training device: %s", device)
    LOGGER.info("TORCH_HOME: %s", os.environ.get("TORCH_HOME"))
    LOGGER.info("Torch hub directory: %s", torch.hub.get_dir())

    split_dir = resolve_project_path(config["data"]["split_dir"])
    if split_dir is None:
        raise ValueError("data.split_dir must not be empty")
    LOGGER.info("Split directory: %s", split_dir)
    if _is_multi_roi_config(config):
        LOGGER.info(
            "Multi-ROI input: roi_root=%s, roi_names=%s, filename template=%s",
            resolve_project_path(config["data"]["roi_root"]),
            config["data"]["roi_names"],
            config["data"].get("image_filename_template", "{ID}.png"),
        )
    else:
        image_root = resolve_project_path(config["data"].get("image_root"))
        LOGGER.info(
            "Image root override: %s; filename template: %s",
            image_root,
            config["data"].get("image_filename_template", "{ID}.png"),
        )
    image_size = int(config["data"]["image_size"])
    mean = config["normalize"]["mean"]
    std = config["normalize"]["std"]
    horizontal_flip = bool(config["augmentation"]["horizontal_flip"])
    train_transform = build_transforms(
        "train", image_size, mean, std, horizontal_flip
    )
    val_transform = build_transforms("val", image_size, mean, std, False)

    train_config = config["train"]
    model_config = config["model"]
    loss_settings = _resolve_loss_settings(config)
    batch_size = int(train_config["batch_size"])
    num_workers = int(train_config["num_workers"])
    pin_memory = bool(train_config.get("pin_memory", device.type == "cuda"))
    n_folds = int(config["data"]["n_folds"])
    folds = args.folds if args.folds is not None else list(range(n_folds))
    invalid_folds = sorted(set(folds).difference(range(n_folds)))
    if invalid_folds:
        raise ValueError(
            f"Requested folds are outside [0, {n_folds - 1}]: {invalid_folds}"
        )
    folds = list(dict.fromkeys(folds))

    for fold in folds:
        fold_dir = experiment_dir / f"fold_{fold}"
        fold_metrics_path = fold_dir / "metrics" / "fold_metrics.csv"
        best_checkpoint_path = fold_dir / "checkpoints" / "best_macro_auc.pth"
        if args.resume and fold_metrics_path.is_file() and best_checkpoint_path.is_file():
            LOGGER.info(
                "Skipping fold %d because resume mode found completed artifacts: %s",
                fold,
                fold_metrics_path,
            )
            continue

        LOGGER.info("Preparing held-out validation fold %d/%d", fold, n_folds - 1)
        LOGGER.info(
            "Model configuration: backbone=%s, pretrained=%s, num_classes=%s, "
            "freeze_backbone=%s",
            model_config["backbone"],
            model_config["pretrained"],
            model_config["num_classes"],
            model_config.get("freeze_backbone", False),
        )
        train_dataset, val_dataset = _build_datasets(
            config,
            fold,
            split_dir,
            train_transform,
            val_transform,
        )
        train_loader = _build_loader(
            train_dataset,
            batch_size,
            True,
            num_workers,
            seed + fold,
            pin_memory,
        )
        val_loader = _build_loader(
            val_dataset,
            batch_size,
            False,
            num_workers,
            seed + 1000 + fold,
            pin_memory,
        )

        model = _build_model(config)
        class_weights = compute_class_weights(
            train_dataset.labels, int(config["data"]["num_classes"])
        )
        LOGGER.info("fold=%d class weights=%s", fold, class_weights.tolist())
        LOGGER.info(
            "Loss configuration: name=%s, class_weight=%s, "
            "label_smoothing_enabled=%s, alpha=%s, mode=%s",
            loss_settings["name"],
            loss_settings["class_weight"],
            loss_settings["label_smoothing_enabled"],
            loss_settings["label_smoothing_alpha"],
            loss_settings["label_smoothing_mode"],
        )
        if loss_settings["label_smoothing_enabled"] and (
            loss_settings["label_smoothing_mode"] != "exclude_true_class"
        ):
            raise ValueError(
                "This experiment only supports label_smoothing.mode="
                "exclude_true_class"
            )
        criterion = build_criterion(
            loss_settings["name"],
            class_weights if loss_settings["class_weight"] else None,
            device=device,
            smoothing=loss_settings["label_smoothing_alpha"],
            num_classes=int(config["data"]["num_classes"]),
        )
        optimizer_name = str(train_config["optimizer"]).lower()
        if optimizer_name != "adamw":
            raise ValueError(
                f"This baseline requires AdamW, got {train_config['optimizer']!r}"
            )
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=float(train_config["lr"]),
            weight_decay=float(train_config["weight_decay"]),
        )

        resume_checkpoint = None
        last_checkpoint_path = fold_dir / "checkpoints" / "last.pth"
        if args.resume and last_checkpoint_path.is_file():
            resume_checkpoint = last_checkpoint_path

        trainer = NYHA3ClassTrainer(
            model=model,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            output_dir=fold_dir,
            epochs=int(train_config["epochs"]),
            early_stopping_patience=int(
                train_config["early_stopping_patience"]
            ),
            use_amp=bool(train_config.get("use_amp", False)),
            fold=fold,
            config=config,
            resume_from=resume_checkpoint,
        )
        trainer.fit(train_loader, val_loader)

        evaluator = NYHA3ClassEvaluator(model, device, fold_dir)
        evaluator.evaluate(
            val_loader, fold_dir / "checkpoints" / "best_macro_auc.pth"
        )
        LOGGER.info("Completed held-out validation fold %d", fold)

    print(f"EXPERIMENT_DIR={experiment_dir}")
    return experiment_dir


if __name__ == "__main__":
    main()
