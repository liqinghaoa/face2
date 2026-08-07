"""Run E1 on the existing fixed Global ResNet18 five-fold protocol."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import sklearn  # noqa: F401  # initialize OpenMP before torch on the bundled Windows env
import torch
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from datasets.nyha_3class_face_dataset import NYHA3ClassFaceDataset, build_transforms  # noqa: E402
from evaluators.e1_hierarchical_conditional_evaluator import E1HierarchicalConditionalEvaluator  # noqa: E402
from losses.e1_hierarchical_conditional_loss import (  # noqa: E402
    HierarchicalConditionalWeightedBCELoss,
    compute_e1_class_weights,
)
from models.e1_global_resnet18_hierarchical import GlobalResNet18HierarchicalConditional  # noqa: E402
from models.nyha_backbone_factory import count_parameters  # noqa: E402
from scripts.evaluate.summarize_e1_global_resnet18_hierarchical_conditional_5fold import summarize  # noqa: E402
from scripts.run.run_exp_global224_imagenet_resnet18_nyha3class_5fold import preflight  # noqa: E402
from trainers.e1_hierarchical_conditional_trainer import E1HierarchicalConditionalTrainer  # noqa: E402
from utils.experiment_utils import (  # noqa: E402
    choose_device, configure_logging, load_yaml, resolve_project_path, save_yaml,
    seed_worker, set_random_seed,
)

LOGGER = logging.getLogger("train_e1_global_resnet18")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--fold", action="append", type=int, dest="folds", help="Optional explicit fold selection; default runs all five fixed folds.")
    parser.add_argument("--resume", action="store_true", help="Resume incomplete folds from last.pth and skip completed folds.")
    parser.add_argument("--allow-overwrite", action="store_true", help="Allow writing only to an existing empty output directory.")
    return parser.parse_args()


def _pretrained_enabled(value: object) -> bool:
    return value if isinstance(value, bool) else str(value).strip().lower() in {"imagenet", "default", "true", "yes", "1"}


def _loader(dataset: NYHA3ClassFaceDataset, batch_size: int, shuffle: bool, num_workers: int, seed: int, pin_memory: bool) -> DataLoader:
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers,
                      pin_memory=pin_memory, persistent_workers=num_workers > 0,
                      worker_init_fn=seed_worker, generator=generator)


def _validate_config(config: dict) -> None:
    if config["model"].get("backbone") != "resnet18" or bool(config["model"].get("freeze_backbone", False)):
        raise ValueError("E1 requires the fully fine-tuned ResNet18 backbone")
    if config["train"].get("optimizer") != "AdamW" or config["train"].get("monitor_metric") != "macro_auc":
        raise ValueError("E1 must preserve E0 AdamW and restored-probability macro_auc monitoring")
    loss = config.get("loss", {})
    if loss.get("name") != "hierarchical_conditional_weighted_bce" or float(loss.get("severity_loss_weight", -1)) != 1.0 or not bool(loss.get("mask_normal_from_severity")) or loss.get("class_weights") != "fold_specific":
        raise ValueError("E1 loss configuration must be the specified fold-specific conditional weighted BCE")
    if int(config.get("metrics", {}).get("ece_bins", 15)) != 15:
        raise ValueError("E1 fixes ECE to 15 equal-width confidence bins")


def main() -> Path:
    args = parse_args()
    config = load_yaml(args.config)
    _validate_config(config)
    default_dir = resolve_project_path(config["experiment"]["output_dir"]) / config["experiment"]["name"]
    experiment_dir = resolve_project_path(args.output_dir) if args.output_dir else default_dir
    if experiment_dir is None:
        raise ValueError("output directory cannot be empty")
    if experiment_dir.exists() and any(experiment_dir.iterdir()) and not args.resume:
        raise FileExistsError(f"E1 output directory already contains results: {experiment_dir}. Choose --output-dir; E1 never overwrites results.")
    if experiment_dir.exists() and not args.allow_overwrite and not args.resume:
        raise FileExistsError(f"E1 output directory already exists: {experiment_dir}. Pass --allow-overwrite only if it is empty.")
    experiment_dir.mkdir(parents=True, exist_ok=True)
    configure_logging(experiment_dir / "experiment.log")
    save_yaml(config, experiment_dir / "config.yaml")
    # Reuse E0's complete split, metadata, patient-leakage, and image preflight.
    preflight(args.config.resolve(), config)
    seed = int(config["train"]["random_seed"])
    set_random_seed(seed)
    device, data, train = choose_device(), config["data"], config["train"]
    split_dir, image_root = resolve_project_path(data["split_dir"]), resolve_project_path(data["image_root"])
    if split_dir is None or image_root is None:
        raise ValueError("E1 requires concrete data.split_dir and data.image_root")
    train_transform = build_transforms("train", int(data["image_size"]), config["normalize"]["mean"], config["normalize"]["std"], bool(config["augmentation"]["horizontal_flip"]))
    val_transform = build_transforms("val", int(data["image_size"]), config["normalize"]["mean"], config["normalize"]["std"], False)
    folds = args.folds if args.folds is not None else list(range(int(data["n_folds"])))
    if any(fold not in range(int(data["n_folds"])) for fold in folds):
        raise ValueError(f"E1 folds must be in [0, {int(data['n_folds']) - 1}]")
    for fold in dict.fromkeys(folds):
        fold_seed = seed + fold
        fold_dir = experiment_dir / f"fold_{fold}"
        completed_metrics = fold_dir / "metrics" / "fold_metrics.csv"
        best_checkpoint = fold_dir / "checkpoints" / "best_macro_auc.pth"
        if args.resume and completed_metrics.is_file() and best_checkpoint.is_file():
            LOGGER.info("Skipping completed fold=%d in resume mode", fold)
            continue
        train_set = NYHA3ClassFaceDataset(split_dir / data["train_csv_pattern"].format(fold=fold), train_transform, image_root, data["image_filename_template"])
        val_set = NYHA3ClassFaceDataset(split_dir / data["val_csv_pattern"].format(fold=fold), val_transform, image_root, data["image_filename_template"])
        weights = compute_e1_class_weights(train_set.labels)
        (fold_dir / "metrics").mkdir(parents=True, exist_ok=True)
        (fold_dir / "metrics" / "class_weights.json").write_text(json.dumps(weights.as_dict(), indent=2), encoding="utf-8")
        model = GlobalResNet18HierarchicalConditional(pretrained=_pretrained_enabled(config["model"]["pretrained"]))
        if not all(parameter.requires_grad for parameter in model.parameters()):
            raise RuntimeError("E1 requires all backbone and heads to be trainable")
        params = count_parameters(model)
        LOGGER.info("fold=%d full fine-tuning; parameters total=%d trainable=%d; class_counts=%s weights=%s", fold, params["total_params"], params["trainable_params"], weights.class_counts, weights.as_dict())
        optimizer = torch.optim.AdamW(model.parameters(), lr=float(train["lr"]), weight_decay=float(train["weight_decay"]))
        criterion = HierarchicalConditionalWeightedBCELoss(weights, float(config["loss"]["severity_loss_weight"]))
        resume_checkpoint = fold_dir / "checkpoints" / "last.pth" if args.resume and (fold_dir / "checkpoints" / "last.pth").is_file() else None
        trainer = E1HierarchicalConditionalTrainer(model, criterion, optimizer, device, fold_dir, int(train["epochs"]), int(train["early_stopping_patience"]), bool(train.get("use_amp", False)), fold, config, weights.as_dict(), int(config.get("metrics", {}).get("ece_bins", 15)), resume_checkpoint)
        trainer.fit(_loader(train_set, int(train["batch_size"]), True, int(train["num_workers"]), fold_seed, bool(train.get("pin_memory", False))), _loader(val_set, int(train["batch_size"]), False, int(train["num_workers"]), seed + 1000 + fold, bool(train.get("pin_memory", False))))
        E1HierarchicalConditionalEvaluator(model, device, fold_dir, int(config.get("metrics", {}).get("ece_bins", 15))).evaluate(_loader(val_set, int(train["batch_size"]), False, int(train["num_workers"]), seed + 1000 + fold, bool(train.get("pin_memory", False))), fold_dir / "checkpoints" / "best_macro_auc.pth")
    if args.folds is None:
        summarize(experiment_dir)
    print(f"EXPERIMENT_DIR={experiment_dir}")
    return experiment_dir


if __name__ == "__main__":
    main()
