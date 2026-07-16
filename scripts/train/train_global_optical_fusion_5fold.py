"""Train one or more fixed folds of one global optical-fusion variant."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from datasets.global_optical_fusion_dataset import GlobalOpticalFusionDataset  # noqa: E402
from datasets.nyha_3class_face_dataset import build_transforms  # noqa: E402
from evaluators.global_optical_fusion_evaluator import GlobalOpticalFusionEvaluator  # noqa: E402
from losses.classification_losses import build_criterion, compute_class_weights  # noqa: E402
from models.resnet18_optical_fusion import ResNet18OpticalFusion  # noqa: E402
from trainers.global_optical_fusion_trainer import (  # noqa: E402
    GlobalOpticalFusionTrainer,
    load_torch_checkpoint,
    make_data_generator,
    seed_payload,
    validate_checkpoint_metadata,
)
from utils.experiment_utils import load_yaml, save_yaml, set_random_seed  # noqa: E402
from utils.optical_feature_preprocessor import (  # noqa: E402
    AVAILABILITY_COLUMN,
    VARIANT_FEATURE_COLUMNS,
    FeatureScaler,
    code_sha256,
    feature_distribution_rows,
    fit_feature_scaler,
    load_feature_frame,
    relative_path,
    resolve_feature_source,
    sha256_file,
    sha256_ids,
    validate_feature_scaler_provenance,
    validate_variant,
)


CLASS_MAPPING = {"normal": 0, "mild": 1, "severe": 2}
IMPLEMENTATION_PATHS = [
    PROJECT_ROOT / "models/resnet18_optical_fusion.py",
    PROJECT_ROOT / "datasets/global_optical_fusion_dataset.py",
    PROJECT_ROOT / "utils/optical_feature_preprocessor.py",
    PROJECT_ROOT / "trainers/global_optical_fusion_trainer.py",
    PROJECT_ROOT / "evaluators/global_optical_fusion_evaluator.py",
    PROJECT_ROOT / "scripts/train/train_global_optical_fusion_5fold.py",
    PROJECT_ROOT / "scripts/run/run_global_optical_fusion_5fold.py",
    PROJECT_ROOT / "config/train/global_optical_fusion/global_resnet18_optical_fusion.yaml",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--variant", required=True)
    parser.add_argument("--fold", type=int, action="append", required=True)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--smoke-test", action="store_true")
    return parser.parse_args()


def _project_path(value: str | Path) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def _optional_git_commit() -> str | None:
    """Return the current commit when available; Git is not required for training."""
    try:
        completed = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            check=False,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if completed.returncode != 0:
        return None
    return completed.stdout.strip() or None


def _balanced_smoke_subset(frame: pd.DataFrame) -> pd.DataFrame:
    selected = []
    for label in (0, 1, 2):
        rows = frame.loc[pd.to_numeric(frame["label_3class"]) == label]
        if rows.empty:
            raise ValueError(f"Smoke split is missing class {label}")
        selected.append(rows.iloc[[0]])
    return pd.concat(selected, ignore_index=True)


def _write_smoke_split(frame: pd.DataFrame, output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output_path, index=False, encoding="utf-8-sig")
    return output_path


def _load_split(config: dict[str, Any], fold: int, role: str) -> tuple[Path, pd.DataFrame]:
    data = config["data"]
    source = _project_path(data["split_root"]) / str(data[f"{role}_csv_pattern"]).format(fold=fold)
    frame = pd.read_csv(
        source, dtype={"ID": "string", "patient_group_id": "string"}, encoding="utf-8-sig"
    )
    return source, frame


def _availability_lookup(config: dict[str, Any]) -> dict[str, int]:
    path = _project_path(config["features"]["raw_source"])
    frame = pd.read_csv(
        path, usecols=["ID", AVAILABILITY_COLUMN], dtype={"ID": "string"}, encoding="utf-8-sig"
    )
    return dict(zip(frame["ID"].astype(str), frame[AVAILABILITY_COLUMN].astype(int)))


def _prepare_features(
    config: dict[str, Any], variant: str, fold: int,
    train_ids: list[str], val_ids: list[str], train_split: Path,
    config_path: Path, *, allow_split_subset: bool = False,
    existing_scaler_path: Path | None = None,
) -> tuple[pd.DataFrame | None, pd.DataFrame | None, FeatureScaler | None, dict[str, Any]]:
    if variant == "global_only":
        return None, None, None, {
            "feature_names": [], "feature_source_relative_path": None,
            "feature_source_sha256": None, "feature_schema_sha256": None,
            "upstream_manifest_sha256": None,
        }
    train_source, schema, manifest = resolve_feature_source(
        config, variant, fold, "train", PROJECT_ROOT
    )
    val_source, val_schema, val_manifest = resolve_feature_source(
        config, variant, fold, "val", PROJECT_ROOT
    )
    assert train_source and val_source and schema and manifest and val_schema and val_manifest
    allow_superset = variant in {"global_mask", "global_raw"} or allow_split_subset
    train_frame = load_feature_frame(
        train_source, variant, train_ids, fold=fold, split_role="train",
        schema_path=schema, allow_source_superset=allow_superset,
    )
    val_frame = load_feature_frame(
        val_source, variant, val_ids, fold=fold, split_role="val",
        schema_path=val_schema, allow_source_superset=allow_superset,
    )
    scaler = None
    if VARIANT_FEATURE_COLUMNS[variant]:
        code_paths = [
            PROJECT_ROOT / "utils/optical_feature_preprocessor.py",
            PROJECT_ROOT / "datasets/global_optical_fusion_dataset.py",
            PROJECT_ROOT / "models/resnet18_optical_fusion.py",
            PROJECT_ROOT / "trainers/global_optical_fusion_trainer.py",
            PROJECT_ROOT / "evaluators/global_optical_fusion_evaluator.py",
            Path(__file__).resolve(),
        ]
        if existing_scaler_path is not None:
            if not existing_scaler_path.is_file():
                raise FileNotFoundError(
                    f"Resume requires the persisted feature scaler: {existing_scaler_path}"
                )
            scaler = FeatureScaler.load_json(existing_scaler_path)
            validate_feature_scaler_provenance(
                scaler, train_frame, variant, fold, source_path=train_source,
                schema_path=schema, upstream_manifest_path=manifest,
                split_path=train_split, code_paths=code_paths,
                config_path=config_path, project_root=PROJECT_ROOT,
            )
        else:
            scaler = fit_feature_scaler(
                train_frame, variant, fold, source_path=train_source, schema_path=schema,
                upstream_manifest_path=manifest, split_path=train_split,
                code_paths=code_paths, config_path=config_path, project_root=PROJECT_ROOT,
                std_epsilon=float(config["feature_standardization"]["std_epsilon"]),
            )
    feature_names = list(VARIANT_FEATURE_COLUMNS[variant]) + [AVAILABILITY_COLUMN]
    return train_frame, val_frame, scaler, {
        "feature_names": feature_names,
        "feature_source_relative_path": {
            "train": relative_path(train_source, PROJECT_ROOT),
            "val": relative_path(val_source, PROJECT_ROOT),
        },
        "feature_source_sha256": {
            "train": sha256_file(train_source), "val": sha256_file(val_source)
        },
        "feature_schema_sha256": sha256_file(schema),
        "upstream_manifest_sha256": sha256_file(manifest),
    }


def train_fold(
    config: dict[str, Any], config_path: Path, variant: str, fold: int,
    output_root: Path, *, resume: bool, overwrite: bool, smoke_test: bool,
) -> Path:
    variant = validate_variant(variant)
    if fold not in config["data"]["folds"]:
        raise ValueError(f"Fold {fold} is not configured")
    run_dir = (
        output_root / "smoke" / variant / f"fold_{fold}"
        if smoke_test else output_root / variant / f"fold_{fold}"
    )
    run_dir_existed = run_dir.exists()
    if run_dir_existed and overwrite:
        resolved_root, resolved_run = output_root.resolve(), run_dir.resolve()
        if resolved_root not in resolved_run.parents:
            raise ValueError("Refusing overwrite outside this experiment root")
        shutil.rmtree(resolved_run)
        run_dir_existed = False
    if run_dir_existed and not resume:
        raise FileExistsError(f"Run directory already exists; use --resume or --overwrite: {run_dir}")
    run_dir.mkdir(parents=True, exist_ok=True)
    resume_path = run_dir / "last_checkpoint.pth" if resume and run_dir_existed else None
    if resume_path is not None and not resume_path.is_file():
        raise FileNotFoundError(
            f"Resume was requested for an existing run without last_checkpoint.pth: {run_dir}"
        )

    train_split, train_frame_meta = _load_split(config, fold, "train")
    val_split, val_frame_meta = _load_split(config, fold, "val")
    if smoke_test:
        train_frame_meta = _balanced_smoke_subset(train_frame_meta)
        val_frame_meta = _balanced_smoke_subset(val_frame_meta)
        train_split = _write_smoke_split(train_frame_meta, run_dir / "smoke_train.csv")
        val_split = _write_smoke_split(val_frame_meta, run_dir / "smoke_val.csv")
    train_ids = train_frame_meta["ID"].astype(str).tolist()
    val_ids = val_frame_meta["ID"].astype(str).tolist()

    seeds = seed_payload(int(config["train"]["seed"]), fold)
    set_random_seed(seeds["model_seed"])
    pretrained = False if smoke_test else str(config["model"]["pretrained"]).lower() == "imagenet"
    model = ResNet18OpticalFusion(variant, pretrained=pretrained)
    set_random_seed(seeds["augmentation_seed"])

    train_features, val_features, scaler, feature_meta = _prepare_features(
        config, variant, fold, train_ids, val_ids, train_split, config_path,
        allow_split_subset=smoke_test,
        existing_scaler_path=(run_dir / "feature_scaler.json") if (
            resume_path is not None and VARIANT_FEATURE_COLUMNS[variant]
        ) else None,
    )
    if scaler is not None and resume_path is None:
        scaler.save_json(run_dir / "feature_scaler.json")
    image_size = 64 if smoke_test else int(config["transforms"]["image_size"])
    transform_config = config["transforms"]
    train_transform = build_transforms(
        "train", image_size, transform_config["mean"], transform_config["std"], True
    )
    val_transform = build_transforms(
        "val", image_size, transform_config["mean"], transform_config["std"], False
    )
    dataset_kwargs = {
        "variant": variant, "fold": fold,
        "image_root": _project_path(config["data"]["image_root"]),
        "image_filename_template": config["data"]["image_filename_template"],
        "scaler": scaler,
    }
    train_dataset = GlobalOpticalFusionDataset(
        train_split, split_role="train", transform=train_transform,
        feature_frame=train_features, **dataset_kwargs,
    )
    val_dataset = GlobalOpticalFusionDataset(
        val_split, split_role="val", transform=val_transform,
        feature_frame=val_features, **dataset_kwargs,
    )
    batch_size = len(train_dataset) if smoke_test else int(config["train"]["batch_size"])
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=0, pin_memory=False, generator=make_data_generator(seeds["shuffle_seed"]),
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        num_workers=0, pin_memory=False, generator=make_data_generator(seeds["validation_seed"]),
    )
    device = torch.device("cuda" if torch.cuda.is_available() and not smoke_test else "cpu")
    class_weights = compute_class_weights(train_dataset.labels, 3)
    criterion = build_criterion(
        "weighted_cross_entropy", class_weights, device=device, reduction="mean"
    )
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=float(config["train"]["learning_rate"]),
        weight_decay=float(config["train"]["weight_decay"]),
    )
    parameter_count = int(sum(parameter.numel() for parameter in model.parameters()))
    trainable_parameter_count = int(
        sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    )
    scaler_payload = scaler.to_dict() if scaler else None
    metadata = {
        "variant": variant, "fold": fold, "architecture": "ResNet18OpticalFusion",
        "backbone": "resnet18", "pretrained_weights": "none_smoke" if smoke_test else "IMAGENET1K_V1",
        "num_classes": 3, "global_feature_dim": 512,
        "auxiliary_input_dim": model.auxiliary_input_dim,
        "fused_input_dim": model.fused_input_dim,
        "classifier_head": f"Linear({model.fused_input_dim},3)",
        "classifier_head_parameter_count": model.classifier_head_parameter_count,
        "parameter_count": parameter_count, "trainable_parameter_count": trainable_parameter_count,
        "feature_names": feature_meta["feature_names"], "availability_position": (
            len(feature_meta["feature_names"]) - 1 if feature_meta["feature_names"] else None
        ),
        "feature_scaler_payload": scaler_payload,
        "feature_scaler_sha256": scaler.payload_sha256 if scaler else None,
        "feature_source_relative_path": feature_meta["feature_source_relative_path"],
        "feature_source_sha256": feature_meta["feature_source_sha256"],
        "feature_schema_sha256": feature_meta["feature_schema_sha256"],
        "upstream_manifest_sha256": feature_meta["upstream_manifest_sha256"],
        "train_id_sha256": sha256_ids(train_ids), "val_id_sha256": sha256_ids(val_ids),
        "split_sha256": sha256_file(_project_path(config["data"]["master_split"])),
        "config": config, "config_sha256": sha256_file(config_path),
        "implementation_signature": code_sha256(IMPLEMENTATION_PATHS, PROJECT_ROOT),
        "git_commit": _optional_git_commit(),
        "class_mapping": CLASS_MAPPING, "class_weights": class_weights.tolist(),
        "transform_definition": {
            "image_size": image_size, "train_horizontal_flip": True,
            "mean": transform_config["mean"], "std": transform_config["std"],
        },
    }
    if resume_path is not None:
        checkpoint = load_torch_checkpoint(resume_path, device)
        validate_checkpoint_metadata(checkpoint, metadata)
    resolved = json.loads(json.dumps(config))
    resolved["resolved_run"] = {
        "variant": variant, "fold": fold, "auxiliary_input_dim": model.auxiliary_input_dim,
        "fused_input_dim": model.fused_input_dim, "seed_info": seeds,
        "smoke_test": smoke_test,
    }
    save_yaml(resolved, run_dir / "resolved_config.yaml")
    if variant in {"global_raw", "global_stage2a", "global_stage2b"}:
        distribution = feature_distribution_rows(train_features, val_features, variant, fold)
    elif variant == "global_mask":
        distribution = [{
            "variant": variant, "fold": fold, "feature": AVAILABILITY_COLUMN,
            "train_valid_n": len(train_features), "val_valid_n": len(val_features),
            "train_availability_ratio": float(train_features[AVAILABILITY_COLUMN].mean()),
            "val_availability_ratio": float(val_features[AVAILABILITY_COLUMN].mean()),
        }]
    else:
        distribution = [{"variant": variant, "fold": fold, "feature": "none"}]
    pd.DataFrame(distribution).to_csv(run_dir / "feature_distribution.csv", index=False, encoding="utf-8-sig")
    trainer = GlobalOpticalFusionTrainer(
        model, criterion, optimizer, device, run_dir, variant=variant, fold=fold,
        metadata=metadata, seed_info=seeds,
        epochs=1 if smoke_test else int(config["train"]["epochs"]),
        patience=int(config["train"]["early_stopping_patience"]),
        minimum_improvement=float(config["train"]["minimum_improvement"]),
        resume_from=resume_path,
    )
    history = trainer.fit(train_loader, val_loader)
    evaluator = GlobalOpticalFusionEvaluator(
        model, device, run_dir, expected_metadata=metadata, feature_scaler=scaler,
        forehead_available_by_id=_availability_lookup(config),
    )
    predictions, _ = evaluator.evaluate(val_loader, run_dir / "best_macro_auc.pth")
    selected = history.loc[history["is_best"].astype(int) == 1].iloc[-1]
    artifact_names = [
        "resolved_config.yaml", "training_log.csv", "training_curves.png",
        "best_macro_auc.pth", "last_checkpoint.pth", "val_predictions.csv",
        "metrics.json", "confusion_matrix.csv", "confusion_matrix.png",
        "feature_distribution.csv",
    ]
    if scaler is not None:
        artifact_names.append("feature_scaler.json")
    artifact_sha256 = {
        name: sha256_file(run_dir / name) for name in artifact_names
    }
    manifest = {
        "task": "global_resnet18_optical_fusion_fold",
        "status": "SMOKE_COMPLETE" if smoke_test else "COMPLETE",
        "formal_result": not smoke_test, "variant": variant, "fold": fold,
        "architecture": "ResNet18OpticalFusion", "backbone": "resnet18",
        "pretrained_weights": "none_smoke" if smoke_test else "IMAGENET1K_V1",
        "global_feature_dim": 512, "auxiliary_input_dim": model.auxiliary_input_dim,
        "fused_input_dim": model.fused_input_dim,
        "classifier_head_parameter_count": model.classifier_head_parameter_count,
        "parameter_count": parameter_count,
        "trainable_parameter_count": trainable_parameter_count,
        "feature_names": feature_meta["feature_names"],
        "feature_source_relative_path": feature_meta["feature_source_relative_path"],
        "feature_source_sha256": feature_meta["feature_source_sha256"],
        "feature_schema_sha256": feature_meta["feature_schema_sha256"],
        "upstream_manifest_sha256": feature_meta["upstream_manifest_sha256"],
        "feature_scaler_sha256": scaler.payload_sha256 if scaler else None,
        "feature_scaler_file_sha256": (
            sha256_file(run_dir / "feature_scaler.json") if scaler else None
        ),
        "train_id_sha256": sha256_ids(train_ids), "val_id_sha256": sha256_ids(val_ids),
        "split_sha256": metadata["split_sha256"], "config_sha256": sha256_file(config_path),
        "class_weights": class_weights.tolist(),
        "best_epoch": int(selected["epoch"]),
        "best_macro_auc": float(selected["val_macro_auc"]),
        "completed_epoch": int(history["epoch"].max()),
        "seed_info": seeds,
        "implementation_signature": metadata["implementation_signature"],
        "prediction_rows": int(len(predictions)), "unique_ids": int(predictions["ID"].nunique()),
        "best_checkpoint_sha256": sha256_file(run_dir / "best_macro_auc.pth"),
        "last_checkpoint_sha256": sha256_file(run_dir / "last_checkpoint.pth"),
        "val_predictions_sha256": sha256_file(run_dir / "val_predictions.csv"),
        "artifact_sha256": artifact_sha256,
        "full_training_executed": not smoke_test,
        "outer_validation_tuning": True, "camera_used": False, "exif_used": False,
        "clinical_features_used": False, "oof_used_as_classifier_train": False,
        "historical_inputs_modified": False,
    }
    with (run_dir / "fold_manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, ensure_ascii=False, indent=2)
    return run_dir


def main() -> None:
    args = parse_args()
    config_path = _project_path(args.config)
    config = load_yaml(config_path)
    variant = validate_variant(args.variant)
    configured_output_root = _project_path(config["experiment"]["output_root"])
    output_root = _project_path(args.output_root or configured_output_root)
    if output_root != configured_output_root:
        raise ValueError(
            "This locked experiment may write only to its dedicated output root: "
            f"{configured_output_root}; got {output_root}"
        )
    for fold in dict.fromkeys(args.fold):
        path = train_fold(
            config, config_path, variant, fold, output_root,
            resume=args.resume, overwrite=args.overwrite, smoke_test=args.smoke_test,
        )
        print(f"RUN_DIR={path}")


if __name__ == "__main__":
    main()
