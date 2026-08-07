"""Formal E2 five-fold entry point; intentionally no local smoke-training mode."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import pandas as pd
import sklearn  # noqa: F401  # initialize OpenMP before torch on the Windows env
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from datasets.e2_global_multiroi_dataset import (  # noqa: E402
    E2GlobalMultiROIDataset,
    ROI_ORDER,
)
from evaluators.e2_evaluator import E2Evaluator  # noqa: E402
from losses.classification_losses import build_criterion, compute_class_weights  # noqa: E402
from models.e2_global_multiroi_equal_fusion import (  # noqa: E402
    E2GlobalMultiROIEqualFusionResNet18,
    count_parameters,
)
from scripts.evaluate.summarize_e2_global_multiroi_equal_fusion_5fold import (  # noqa: E402
    summarize,
)
from trainers.e2_trainer import E2Trainer  # noqa: E402
from utils.experiment_utils import (  # noqa: E402
    choose_device,
    configure_logging,
    load_yaml,
    resolve_project_path,
    save_yaml,
    seed_worker,
    set_random_seed,
)

LOGGER = logging.getLogger("train_e2_equal_fusion")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Formal E2 five-fold training. This command performs a full "
            "metadata/path preflight and intentionally has no local smoke mode."
        )
    )
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument(
        "--fold",
        type=int,
        action="append",
        dest="folds",
        help="Optional explicit fold selection. Omit to run all five folds.",
    )
    parser.add_argument(
        "--allow-existing-empty",
        action="store_true",
        help="Permit only a pre-created empty output directory.",
    )
    return parser.parse_args()


def _path(value: str | Path) -> Path:
    resolved = resolve_project_path(value)
    if resolved is None:
        raise ValueError("Required E2 configuration path is empty")
    return resolved


def _pretrained_enabled(value: object) -> bool:
    if isinstance(value, bool):
        return value
    normalized = str(value).strip().lower()
    if normalized in {"imagenet", "default", "true", "yes", "1"}:
        return True
    if normalized in {"none", "false", "no", "0", "random"}:
        return False
    raise ValueError(f"Unsupported E2 model.pretrained value: {value!r}")


def _loader(
    dataset: E2GlobalMultiROIDataset,
    batch_size: int,
    shuffle: bool,
    workers: int,
    seed: int,
    pin_memory: bool,
) -> DataLoader:
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=workers,
        pin_memory=pin_memory,
        persistent_workers=workers > 0,
        worker_init_fn=seed_worker,
        generator=generator,
    )


def _validate_config(config: dict) -> None:
    data, model, train, loss = (
        config["data"],
        config["model"],
        config["train"],
        config["loss"],
    )
    if tuple(data["roi_order"]) != ROI_ORDER:
        raise ValueError(f"E2 ROI order must be {ROI_ORDER}")
    if set(data["roi_roots"]) != set(ROI_ORDER):
        raise ValueError("E2 must configure exactly the five fixed ROI roots")
    if int(data["expected_num_samples"]) != 500 or int(data["n_folds"]) != 5:
        raise ValueError("E2 requires exactly the fixed splits_500 five-fold protocol")
    if int(model.get("num_classes", -1)) != 3:
        raise ValueError("E2 requires direct three-class logits")
    required_model = {
        "backbone": "resnet18",
        "global_encoder": "independent_resnet18",
        "roi_encoder": "one_shared_resnet18",
        "roi_aggregation": "fixed_arithmetic_mean",
        "fusion": "concat_512_plus_512",
        "classifier": "linear_1024_to_3",
    }
    if any(model.get(key) != value for key, value in required_model.items()):
        raise ValueError("E2 model configuration differs from its fixed equal-fusion design")
    if bool(model.get("freeze_backbone", False)):
        raise ValueError("E2 requires full fine-tuning of both encoders")
    if (
        int(train["batch_size"]) != 16
        or train["optimizer"] != "AdamW"
        or train["monitor_metric"] != "macro_auc"
    ):
        raise ValueError(
            "E2 must preserve E0 batch_size=16, AdamW, and Macro-AUC monitoring"
        )
    if (
        loss.get("name") != "weighted_cross_entropy"
        or loss.get("class_weight") != "fold_specific"
        or bool(loss.get("label_smoothing", False))
    ):
        raise ValueError("E2 only permits E0 weighted cross entropy without smoothing")
    augmentation = config["augmentation"]
    if (
        not bool(augmentation.get("synchronized_horizontal_flip"))
        or bool(augmentation.get("color_jitter"))
        or bool(augmentation.get("random_crop"))
        or bool(augmentation.get("random_rotation"))
    ):
        raise ValueError("E2 only permits E0's synchronized flip-only augmentation")


def _preflight(config: dict) -> pd.DataFrame:
    """Check splits_500 and all configured paths, returning a 2,500-row manifest.

    This is part of the explicit formal command only. It verifies metadata and
    exact filename mappings without image decoding or feature extraction.
    """

    data = config["data"]
    split_dir = _path(data["split_dir"])
    global_root = _path(data["global_root"])
    roi_roots = {name: _path(path) for name, path in data["roi_roots"].items()}
    if not split_dir.is_dir():
        raise FileNotFoundError(f"E2 split root does not exist: {split_dir}")
    for image_root in [global_root, *roi_roots.values()]:
        if not image_root.is_dir():
            raise FileNotFoundError(f"E2 image root does not exist: {image_root}")

    validation_ids: list[str] = []
    validation_group_folds: dict[str, set[int]] = {}
    all_cohort_ids: set[str] | None = None
    manifests: list[pd.DataFrame] = []
    for fold in range(5):
        train_set = E2GlobalMultiROIDataset(
            split_dir / data["train_csv_pattern"].format(fold=fold),
            global_root,
            roi_roots,
            image_filename_template=data["image_filename_template"],
            image_size=int(data["image_size"]),
            validate_paths=True,
        )
        val_set = E2GlobalMultiROIDataset(
            split_dir / data["val_csv_pattern"].format(fold=fold),
            global_root,
            roi_roots,
            image_filename_template=data["image_filename_template"],
            image_size=int(data["image_size"]),
            validate_paths=True,
        )
        train_ids = set(train_set.frame["ID"].astype(str))
        val_ids = set(val_set.frame["ID"].astype(str))
        train_groups = set(train_set.frame["patient_group_id"].astype(str))
        val_groups = set(val_set.frame["patient_group_id"].astype(str))
        if train_ids.intersection(val_ids) or train_groups.intersection(val_groups):
            raise ValueError(f"Sample or patient-group leakage in fold {fold}")
        cohort_ids = train_ids.union(val_ids)
        if len(cohort_ids) != 500:
            raise ValueError(f"Fold {fold} does not contain exactly 500 unique IDs")
        if all_cohort_ids is None:
            all_cohort_ids = cohort_ids
        elif cohort_ids != all_cohort_ids:
            raise ValueError(
                f"Fold {fold} is not drawn from the same fixed 500-sample cohort"
            )
        if not (val_set.frame["fold"] == fold).all() or (
            train_set.frame["fold"] == fold
        ).any():
            raise ValueError(f"Fold assignment inconsistency in fold {fold}")
        validation_ids.extend(val_ids)
        # Repeated patient groups can have multiple images, but all such images
        # must remain inside the same held-out fold.
        for group in val_groups:
            validation_group_folds.setdefault(group, set()).add(fold)
        manifests.extend(
            [train_set.manifest("train", fold), val_set.manifest("val", fold)]
        )

    repeated_ids = pd.Series(validation_ids).duplicated().any()
    groups_in_multiple_folds = [
        group for group, group_folds in validation_group_folds.items() if len(group_folds) > 1
    ]
    if len(validation_ids) != 500 or repeated_ids or groups_in_multiple_folds:
        raise ValueError(
            "E2 validation folds do not form a unique 500-sample OOF cohort "
            "with patient groups confined to one fold"
        )
    return pd.concat(manifests, ignore_index=True)


def main() -> Path:
    args = parse_args()
    config = load_yaml(args.config)
    _validate_config(config)

    default_dir = _path(config["experiment"]["output_dir"]) / config["experiment"]["name"]
    experiment_dir = _path(args.output_dir) if args.output_dir else default_dir
    if experiment_dir.exists() and any(experiment_dir.iterdir()):
        raise FileExistsError(
            f"E2 output exists and will not be overwritten: {experiment_dir}"
        )
    if experiment_dir.exists() and not args.allow_existing_empty:
        raise FileExistsError(
            f"E2 output directory exists: {experiment_dir}; use "
            "--allow-existing-empty only when it is empty."
        )

    manifest = _preflight(config)
    experiment_dir.mkdir(parents=True, exist_ok=True)
    configure_logging(experiment_dir / "experiment.log")
    save_yaml(config, experiment_dir / "resolved_config.yaml")
    manifest.to_csv(experiment_dir / "data_manifest.csv", index=False, encoding="utf-8-sig")

    seed = int(config["train"]["random_seed"])
    set_random_seed(seed)
    device = choose_device()
    data, train = config["data"], config["train"]
    split_dir = _path(data["split_dir"])
    roi_roots = {name: _path(path) for name, path in data["roi_roots"].items()}
    common_dataset_args = {
        "global_root": _path(data["global_root"]),
        "roi_roots": roi_roots,
        "image_filename_template": data["image_filename_template"],
        "image_size": int(data["image_size"]),
        "horizontal_flip": bool(config["augmentation"]["horizontal_flip"]),
        "mean": config["normalize"]["mean"],
        "std": config["normalize"]["std"],
        "validate_paths": False,
    }
    LOGGER.info(
        "Global root=%s; split root=%s; ROI roots=%s; ROI order=%s",
        data["global_root"],
        data["split_dir"],
        data["roi_roots"],
        list(ROI_ORDER),
    )
    LOGGER.info(
        "Two independent backbones; one shared ROI backbone; fixed equal ROI "
        "mean; full fine-tuning; batch_size=16; monitor=macro_auc"
    )

    folds = args.folds if args.folds is not None else list(range(5))
    if not folds or any(fold not in range(5) for fold in folds):
        raise ValueError("E2 folds must be in 0..4")
    for fold in dict.fromkeys(folds):
        train_set = E2GlobalMultiROIDataset(
            split_dir / data["train_csv_pattern"].format(fold=fold),
            train=True,
            **common_dataset_args,
        )
        val_set = E2GlobalMultiROIDataset(
            split_dir / data["val_csv_pattern"].format(fold=fold),
            train=False,
            **common_dataset_args,
        )
        weights = compute_class_weights(train_set.labels, num_classes=3)
        class_counts = torch.bincount(
            torch.tensor(train_set.labels), minlength=3
        ).tolist()
        fold_dir = experiment_dir / f"fold_{fold}"
        (fold_dir / "metrics").mkdir(parents=True, exist_ok=True)
        (fold_dir / "metrics" / "class_weights.json").write_text(
            json.dumps(
                {
                    "class_counts": class_counts,
                    "class_weights": weights.tolist(),
                    "formula": "N_train / (3 * n_c)",
                    "source_split": "train",
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        model = E2GlobalMultiROIEqualFusionResNet18(
            pretrained=_pretrained_enabled(config["model"]["pretrained"])
        )
        parameter_counts = count_parameters(model)
        if not all(parameter.requires_grad for parameter in model.parameters()):
            raise RuntimeError("E2 requires every model parameter to be trainable")
        LOGGER.info(
            "fold=%d n_train=%d n_val=%d counts=%s weights=%s "
            "total_params=%d trainable_params=%d",
            fold,
            len(train_set),
            len(val_set),
            class_counts,
            weights.tolist(),
            parameter_counts["total_params"],
            parameter_counts["trainable_params"],
        )
        criterion = build_criterion(
            "weighted_cross_entropy", weights, device=device, num_classes=3
        )
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=float(train["lr"]),
            weight_decay=float(train["weight_decay"]),
        )
        trainer = E2Trainer(
            model,
            criterion,
            optimizer,
            device,
            fold_dir,
            config,
            fold,
            weights.tolist(),
            class_counts,
        )
        train_loader = _loader(
            train_set,
            int(train["batch_size"]),
            True,
            int(train["num_workers"]),
            seed + fold,
            bool(train["pin_memory"]),
        )
        val_loader = _loader(
            val_set,
            int(train["batch_size"]),
            False,
            int(train["num_workers"]),
            seed + 1000 + fold,
            bool(train["pin_memory"]),
        )
        trainer.fit(train_loader, val_loader)
        E2Evaluator(
            model,
            device,
            fold_dir,
            int(config.get("metrics", {}).get("ece_bins", 15)),
        ).evaluate(
            val_loader, fold_dir / "checkpoints" / "best_macro_auc.pth"
        )

    if args.folds is None:
        summarize(experiment_dir)
    print(f"EXPERIMENT_DIR={experiment_dir}")
    return experiment_dir


if __name__ == "__main__":
    main()
