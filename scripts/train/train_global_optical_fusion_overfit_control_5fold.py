"""Train locked frozen-backbone or partial-layer4 optical-fusion controls."""

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
from evaluators.global_optical_fusion_overfit_evaluator import (  # noqa: E402
    GlobalOpticalFusionOverfitEvaluator,
)
from losses.classification_losses import build_criterion, compute_class_weights  # noqa: E402
from models.resnet18_optical_fusion import ResNet18OpticalFusion  # noqa: E402
from trainers.global_optical_fusion_overfit_trainer import (  # noqa: E402
    GlobalOpticalFusionOverfitTrainer,
)
from trainers.global_optical_fusion_trainer import make_data_generator, seed_payload  # noqa: E402
from utils.experiment_utils import load_yaml, save_yaml, set_random_seed  # noqa: E402
from utils.optical_feature_preprocessor import (  # noqa: E402
    AVAILABILITY_COLUMN,
    VARIANT_FEATURE_COLUMNS,
    FeatureScaler,
    code_sha256,
    feature_distribution_rows,
    load_feature_frame,
    relative_path,
    resolve_feature_source,
    sha256_file,
    sha256_ids,
)
from utils.resnet_trainability import (  # noqa: E402
    build_optimizer,
    build_trainability_audit,
    configure_trainability_strategy,
    enforce_trainability_modes,
    validate_trainability_strategy,
)


ALLOWED_VARIANTS = ("global_only", "global_stage2a")
CLASS_MAPPING = {"normal": 0, "mild": 1, "severe": 2}
IMPLEMENTATION_PATHS = [
    PROJECT_ROOT / "models/resnet18_optical_fusion.py",
    PROJECT_ROOT / "datasets/global_optical_fusion_dataset.py",
    PROJECT_ROOT / "utils/optical_feature_preprocessor.py",
    PROJECT_ROOT / "utils/resnet_trainability.py",
    PROJECT_ROOT / "trainers/global_optical_fusion_overfit_trainer.py",
    PROJECT_ROOT / "evaluators/global_optical_fusion_overfit_evaluator.py",
    Path(__file__).resolve(),
    PROJECT_ROOT / "scripts/run/run_global_optical_fusion_overfit_control.py",
    PROJECT_ROOT / "scripts/evaluate/summarize_global_optical_fusion_overfit_control.py",
    PROJECT_ROOT
    / "config/train/global_optical_fusion_overfit_control/global_resnet18_overfit_control.yaml",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--strategy", required=True)
    parser.add_argument("--variant", required=True)
    parser.add_argument("--fold", type=int, action="append", required=True)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--allow-cpu-training", action="store_true")
    parser.add_argument("--allow-reference-mismatch", action="store_true")
    return parser.parse_args()


def project_path(value: str | Path) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def optional_git_commit() -> str | None:
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
    return completed.stdout.strip() or None if completed.returncode == 0 else None


def load_split(config: dict[str, Any], fold: int, role: str) -> tuple[Path, pd.DataFrame]:
    source = project_path(config["data"]["split_root"]) / str(
        config["data"][f"{role}_csv_pattern"]
    ).format(fold=fold)
    frame = pd.read_csv(
        source,
        dtype={"ID": "string", "patient_group_id": "string"},
        encoding="utf-8-sig",
    )
    return source, frame


def availability_lookup(config: dict[str, Any]) -> dict[str, int]:
    frame = pd.read_csv(
        project_path(config["features"]["raw_source"]),
        usecols=["ID", AVAILABILITY_COLUMN],
        dtype={"ID": "string"},
        encoding="utf-8-sig",
    )
    return dict(zip(frame["ID"].astype(str), frame[AVAILABILITY_COLUMN].astype(int)))


def balanced_smoke_subset(frame: pd.DataFrame) -> pd.DataFrame:
    selected = []
    for label in (0, 1, 2):
        rows = frame.loc[pd.to_numeric(frame["label_3class"]) == label]
        if rows.empty:
            raise ValueError(f"Smoke split is missing class {label}")
        selected.append(rows.iloc[[0]])
    return pd.concat(selected, ignore_index=True)


def verify_canonical_scaler(
    scaler: FeatureScaler,
    train_features: pd.DataFrame,
    *,
    fold: int,
    train_ids: list[str],
    source_path: Path,
) -> dict[str, Any]:
    if scaler.variant != "global_stage2a" or scaler.fold != fold:
        raise ValueError("Canonical scaler variant/fold mismatch")
    if scaler.train_id_sha256 != sha256_ids(train_ids):
        raise ValueError("Canonical scaler was not fitted on this outer-train ID list")
    if scaler.source_sha256 != sha256_file(source_path):
        raise ValueError("Canonical scaler feature-source hash mismatch")
    columns = list(VARIANT_FEATURE_COLUMNS["global_stage2a"])
    availability = train_features[AVAILABILITY_COLUMN].astype(int).to_numpy()
    recomputed_mean: list[float] = []
    recomputed_std: list[float] = []
    recomputed_n: list[int] = []
    for index, column in enumerate(columns):
        values = pd.to_numeric(train_features[column], errors="coerce").to_numpy(float)
        selected = values if index < 3 else values[availability == 1]
        if not np.isfinite(selected).all():
            raise ValueError(f"Non-finite canonical train values in {column}")
        recomputed_mean.append(float(np.mean(selected)))
        recomputed_std.append(float(np.std(selected, ddof=0)))
        recomputed_n.append(int(len(selected)))
    mean_delta = float(np.max(np.abs(np.asarray(recomputed_mean) - np.asarray(scaler.mean))))
    std_delta = float(np.max(np.abs(np.asarray(recomputed_std) - np.asarray(scaler.std))))
    if recomputed_n != scaler.valid_n or mean_delta > 1.0e-12 or std_delta > 1.0e-12:
        raise ValueError(
            "Canonical Full scaler cannot be reproduced exactly from the outer train split; "
            f"mean_delta={mean_delta}, std_delta={std_delta}"
        )
    return {
        "canonical_scaler_recomputed": True,
        "mean_max_abs_delta": mean_delta,
        "std_max_abs_delta": std_delta,
        "valid_n": recomputed_n,
    }


def prepare_features(
    config: dict[str, Any],
    variant: str,
    fold: int,
    train_ids: list[str],
    val_ids: list[str],
    *,
    allow_source_superset: bool,
) -> tuple[
    pd.DataFrame | None,
    pd.DataFrame | None,
    FeatureScaler | None,
    dict[str, Any],
]:
    if variant == "global_only":
        return None, None, None, {
            "feature_names": [],
            "feature_source_relative_path": None,
            "feature_source_sha256": None,
            "feature_schema_sha256": None,
            "upstream_manifest_sha256": None,
            "canonical_scaler_path": None,
            "canonical_scaler_file_sha256": None,
            "canonical_scaler_verification": None,
        }
    train_source, train_schema, train_manifest = resolve_feature_source(
        config, variant, fold, "train", PROJECT_ROOT
    )
    val_source, val_schema, val_manifest = resolve_feature_source(
        config, variant, fold, "val", PROJECT_ROOT
    )
    if not all((train_source, train_schema, train_manifest, val_source, val_schema, val_manifest)):
        raise ValueError("Stage 2A feature provenance is incomplete")
    train_features = load_feature_frame(
        train_source,
        variant,
        train_ids,
        fold=fold,
        split_role="train",
        schema_path=train_schema,
        allow_source_superset=allow_source_superset,
    )
    val_features = load_feature_frame(
        val_source,
        variant,
        val_ids,
        fold=fold,
        split_role="val",
        schema_path=val_schema,
        allow_source_superset=allow_source_superset,
    )
    canonical_path = project_path(
        str(config["features"]["canonical_scaler_pattern"]).format(fold=fold)
    )
    scaler = FeatureScaler.load_json(canonical_path)
    full_train_ids = (
        pd.read_csv(
            project_path(config["data"]["split_root"])
            / str(config["data"]["train_csv_pattern"]).format(fold=fold),
            usecols=["ID"],
            dtype={"ID": "string"},
            encoding="utf-8-sig",
        )["ID"]
        .astype(str)
        .tolist()
    )
    full_train_features = load_feature_frame(
        train_source,
        variant,
        full_train_ids,
        fold=fold,
        split_role="train",
        schema_path=train_schema,
        allow_source_superset=False,
    )
    verification = verify_canonical_scaler(
        scaler,
        full_train_features,
        fold=fold,
        train_ids=full_train_ids,
        source_path=train_source,
    )
    return train_features, val_features, scaler, {
        "feature_names": list(VARIANT_FEATURE_COLUMNS[variant]) + [AVAILABILITY_COLUMN],
        "feature_source_relative_path": {
            "train": relative_path(train_source, PROJECT_ROOT),
            "val": relative_path(val_source, PROJECT_ROOT),
        },
        "feature_source_sha256": {
            "train": sha256_file(train_source),
            "val": sha256_file(val_source),
        },
        "feature_schema_sha256": sha256_file(train_schema),
        "upstream_manifest_sha256": sha256_file(train_manifest),
        "canonical_scaler_path": relative_path(canonical_path, PROJECT_ROOT),
        "canonical_scaler_file_sha256": sha256_file(canonical_path),
        "canonical_scaler_verification": verification,
    }


def _remove_scoped_run(run_dir: Path, output_root: Path) -> None:
    resolved_root = output_root.resolve()
    resolved_run = run_dir.resolve()
    if resolved_root not in resolved_run.parents:
        raise ValueError("Refusing overwrite outside the dedicated experiment root")
    shutil.rmtree(resolved_run)


def _require_reference_gate(
    output_root: Path, *, allow_reference_mismatch: bool = False
) -> None:
    path = output_root / "full_reference" / "reference_audit.json"
    if not path.is_file():
        raise RuntimeError("Formal training requires full_reference/reference_audit.json")
    audit = json.loads(path.read_text(encoding="utf-8"))
    historical_unchanged = not bool(audit.get("historical_inputs_modified", True))
    if not historical_unchanged:
        raise RuntimeError("Formal training is forbidden because historical Full inputs changed")
    if audit.get("status") != "PASS" or int(audit.get("passed_checkpoints", 0)) != 10:
        if allow_reference_mismatch and audit.get("status") in {"FAIL", "RUNTIME_MISMATCH"}:
            print(
                "REFERENCE_REPRODUCTION_OVERRIDE=true "
                f"status={audit.get('status')} passed={audit.get('passed_checkpoints', 0)}/10",
                flush=True,
            )
            return
        raise RuntimeError("Formal training is forbidden until all 10 Full checkpoints reproduce")


def train_fold(
    config: dict[str, Any],
    config_path: Path,
    strategy: str,
    variant: str,
    fold: int,
    output_root: Path,
    *,
    resume: bool,
    overwrite: bool,
    smoke_test: bool,
    allow_cpu_training: bool,
    allow_reference_mismatch: bool = False,
) -> Path:
    strategy = validate_trainability_strategy(strategy)
    if variant not in ALLOWED_VARIANTS:
        raise ValueError(f"variant must be one of {ALLOWED_VARIANTS}")
    if fold not in config["data"]["folds"]:
        raise ValueError(f"Fold {fold} is not configured")
    if not smoke_test:
        _require_reference_gate(
            output_root, allow_reference_mismatch=allow_reference_mismatch
        )
        if not torch.cuda.is_available() and not allow_cpu_training:
            raise RuntimeError(
                "CUDA is unavailable; formal training requires CUDA or --allow-cpu-training"
            )
    run_dir = (
        output_root / "smoke" / strategy / variant / f"fold_{fold}"
        if smoke_test
        else output_root / strategy / variant / f"fold_{fold}"
    )
    existed = run_dir.exists()
    if existed and overwrite:
        _remove_scoped_run(run_dir, output_root)
        existed = False
    if existed and not resume:
        raise FileExistsError(f"Run exists; use --resume or --overwrite: {run_dir}")
    run_dir.mkdir(parents=True, exist_ok=True)
    resume_path = run_dir / "last_checkpoint.pth" if resume and existed else None
    if resume_path is not None and not resume_path.is_file():
        raise FileNotFoundError("Resume requested but last_checkpoint.pth is missing")

    train_split, train_frame = load_split(config, fold, "train")
    val_split, val_frame = load_split(config, fold, "val")
    if not smoke_test:
        expected_train = int(config["data"]["expected_train_rows"])
        expected_val = int(config["data"]["expected_val_rows"])
        if (len(train_frame), len(val_frame)) != (expected_train, expected_val):
            raise ValueError("Outer fold must contain exactly 400 train and 100 validation rows")
    if smoke_test:
        train_frame = balanced_smoke_subset(train_frame)
        val_frame = balanced_smoke_subset(val_frame)
        train_split = run_dir / "smoke_train.csv"
        val_split = run_dir / "smoke_val.csv"
        train_frame.to_csv(train_split, index=False, encoding="utf-8-sig")
        val_frame.to_csv(val_split, index=False, encoding="utf-8-sig")
    train_ids = train_frame["ID"].astype(str).tolist()
    val_ids = val_frame["ID"].astype(str).tolist()

    seed_info = seed_payload(int(config["train"]["seed"]), fold)
    set_random_seed(seed_info["model_seed"])
    model = ResNet18OpticalFusion(
        variant,
        num_classes=3,
        pretrained=(False if smoke_test else True),
    )
    configure_trainability_strategy(model, strategy)
    optimizer = build_optimizer(
        model,
        strategy,
        classifier_lr=1.0e-4,
        layer4_lr=1.0e-5,
        weight_decay=float(config["train"]["weight_decay"]),
    )
    enforce_trainability_modes(model, strategy, training=True)
    audit = build_trainability_audit(
        model, optimizer, strategy, variant=variant, fold=fold
    )
    if strategy == "frozen_backbone":
        expected_head_count = 1539 if variant == "global_only" else 1560
        if audit["trainable_parameter_count"] != expected_head_count:
            raise AssertionError("Frozen-backbone trainable parameter count is incorrect")
    with (run_dir / "trainability_audit.json").open("w", encoding="utf-8") as handle:
        json.dump(audit, handle, ensure_ascii=False, indent=2)

    train_features, val_features, scaler, feature_meta = prepare_features(
        config,
        variant,
        fold,
        train_ids,
        val_ids,
        allow_source_superset=smoke_test,
    )
    if scaler is not None:
        source = project_path(feature_meta["canonical_scaler_path"])
        shutil.copy2(source, run_dir / "feature_scaler.json")
        shutil.copy2(source, run_dir / "feature_scaler_reference.json")
        if sha256_file(source) != sha256_file(run_dir / "feature_scaler.json"):
            raise RuntimeError("Copied canonical scaler differs from the Full scaler")

    image_size = 64 if smoke_test else int(config["transforms"]["image_size"])
    transforms_config = config["transforms"]
    train_transform = build_transforms(
        "train",
        image_size,
        transforms_config["mean"],
        transforms_config["std"],
        True,
    )
    val_transform = build_transforms(
        "val",
        image_size,
        transforms_config["mean"],
        transforms_config["std"],
        False,
    )
    dataset_kwargs = {
        "variant": variant,
        "fold": fold,
        "image_root": project_path(config["data"]["image_root"]),
        "image_filename_template": config["data"]["image_filename_template"],
        "scaler": scaler,
    }
    train_dataset = GlobalOpticalFusionDataset(
        train_split,
        split_role="train",
        transform=train_transform,
        feature_frame=train_features,
        **dataset_kwargs,
    )
    deterministic_train_dataset = GlobalOpticalFusionDataset(
        train_split,
        split_role="train",
        transform=val_transform,
        feature_frame=train_features,
        **dataset_kwargs,
    )
    val_dataset = GlobalOpticalFusionDataset(
        val_split,
        split_role="val",
        transform=val_transform,
        feature_frame=val_features,
        **dataset_kwargs,
    )
    batch_size = len(train_dataset) if smoke_test else int(config["train"]["batch_size"])
    loader_kwargs = {
        "batch_size": batch_size,
        "num_workers": 0,
        "pin_memory": False,
    }
    train_loader = DataLoader(
        train_dataset,
        shuffle=True,
        generator=make_data_generator(seed_info["shuffle_seed"]),
        **loader_kwargs,
    )
    deterministic_train_loader = DataLoader(
        deterministic_train_dataset, shuffle=False, **loader_kwargs
    )
    val_loader = DataLoader(val_dataset, shuffle=False, **loader_kwargs)
    device = torch.device("cpu" if smoke_test else ("cuda" if torch.cuda.is_available() else "cpu"))
    class_weights = compute_class_weights(train_dataset.labels, 3)
    criterion = build_criterion(
        "weighted_cross_entropy", class_weights, device=device, reduction="mean"
    )
    parameter_count = int(sum(parameter.numel() for parameter in model.parameters()))
    implementation_paths = [path for path in IMPLEMENTATION_PATHS if path.is_file()]
    metadata: dict[str, Any] = {
        "task": "global_optical_fusion_overfit_control",
        "strategy": strategy,
        "variant": variant,
        "fold": fold,
        "architecture": "ResNet18OpticalFusion",
        "backbone": "resnet18",
        "pretrained_weights": "none_smoke" if smoke_test else "IMAGENET1K_V1",
        "initialization_source": "independent_imagenet_resnet18",
        "initialized_from_full_checkpoint": False,
        "num_classes": 3,
        "global_feature_dim": 512,
        "auxiliary_input_dim": model.auxiliary_input_dim,
        "fused_input_dim": model.fused_input_dim,
        "classifier_head_parameter_count": model.classifier_head_parameter_count,
        "parameter_count": parameter_count,
        "trainable_parameter_count": audit["trainable_parameter_count"],
        "frozen_parameter_count": audit["frozen_parameter_count"],
        "feature_names": feature_meta["feature_names"],
        "feature_scaler_payload": scaler.to_dict() if scaler else None,
        "feature_scaler_sha256": scaler.payload_sha256 if scaler else None,
        "feature_source_relative_path": feature_meta["feature_source_relative_path"],
        "feature_source_sha256": feature_meta["feature_source_sha256"],
        "feature_schema_sha256": feature_meta["feature_schema_sha256"],
        "upstream_manifest_sha256": feature_meta["upstream_manifest_sha256"],
        "canonical_scaler_path": feature_meta["canonical_scaler_path"],
        "canonical_scaler_file_sha256": feature_meta["canonical_scaler_file_sha256"],
        "canonical_scaler_verification": feature_meta["canonical_scaler_verification"],
        "train_id_sha256": sha256_ids(train_ids),
        "val_id_sha256": sha256_ids(val_ids),
        "split_sha256": sha256_file(project_path(config["data"]["master_split"])),
        "config": config,
        "config_sha256": sha256_file(config_path),
        "implementation_signature": code_sha256(implementation_paths, PROJECT_ROOT),
        "git_commit": optional_git_commit(),
        "class_mapping": CLASS_MAPPING,
        "class_weights": class_weights.tolist(),
        "transform_definition": {
            "image_size": image_size,
            "train_horizontal_flip_probability": 0.5,
            "deterministic_train_transform": "validation_transform",
            "mean": transforms_config["mean"],
            "std": transforms_config["std"],
        },
        "trainability_trainable_parameter_names": audit["trainable_parameter_names"],
        "trainability_frozen_parameter_names": audit["frozen_parameter_names"],
        "trainability_trainable_parameter_count": audit["trainable_parameter_count"],
        "trainability_frozen_parameter_count": audit["frozen_parameter_count"],
        "trainability_optimizer_groups": audit["optimizer_groups"],
        "trainability_batchnorm_training_modes": audit["batchnorm_training_modes"],
        "frozen_module_names": audit["frozen_module_names"],
        "trainable_module_names": audit["trainable_module_names"],
    }
    resolved = json.loads(json.dumps(config))
    resolved["resolved_run"] = {
        "strategy": strategy,
        "variant": variant,
        "fold": fold,
        "seed_info": seed_info,
        "device": str(device),
        "smoke_test": smoke_test,
        "trainability_audit_sha256": sha256_file(run_dir / "trainability_audit.json"),
    }
    save_yaml(resolved, run_dir / "resolved_config.yaml")
    distribution = (
        feature_distribution_rows(train_features, val_features, variant, fold)
        if variant == "global_stage2a"
        else [{"variant": variant, "fold": fold, "feature": "none"}]
    )
    pd.DataFrame(distribution).to_csv(
        run_dir / "feature_distribution.csv", index=False, encoding="utf-8-sig"
    )
    trainer = GlobalOpticalFusionOverfitTrainer(
        model,
        criterion,
        optimizer,
        device,
        run_dir,
        strategy=strategy,
        variant=variant,
        fold=fold,
        metadata=metadata,
        seed_info=seed_info,
        epochs=(1 if smoke_test else int(config["train"]["epochs"])),
        patience=int(config["train"]["early_stopping_patience"]),
        minimum_improvement=float(config["train"]["minimum_improvement"]),
        resume_from=resume_path,
    )
    history = trainer.fit(train_loader, val_loader)
    evaluator = GlobalOpticalFusionOverfitEvaluator(
        model,
        criterion,
        device,
        run_dir,
        expected_metadata=metadata,
        feature_scaler=scaler,
        forehead_available_by_id=availability_lookup(config),
    )
    overfit_metrics = evaluator.evaluate(
        deterministic_train_loader, val_loader, run_dir / "best_macro_auc.pth"
    )
    artifact_names = [
        "resolved_config.yaml",
        "trainability_audit.json",
        "feature_distribution.csv",
        "training_log.csv",
        "training_curves.png",
        "best_macro_auc.pth",
        "last_checkpoint.pth",
        "train_predictions.csv",
        "val_predictions.csv",
        "train_metrics.json",
        "val_metrics.json",
        "metrics.json",
        "deterministic_train_predictions.csv",
        "deterministic_val_predictions.csv",
        "deterministic_train_metrics.json",
        "deterministic_val_metrics.json",
        "overfit_metrics.json",
    ]
    if scaler is not None:
        artifact_names.extend(["feature_scaler.json", "feature_scaler_reference.json"])
    manifest = {
        "schema_version": "global_optical_fusion_overfit_fold_manifest_v1",
        "task": "global_optical_fusion_overfit_control",
        "status": "SMOKE_COMPLETE" if smoke_test else "COMPLETE",
        "formal_result": not smoke_test,
        "strategy": strategy,
        "variant": variant,
        "fold": fold,
        "best_epoch": int(overfit_metrics["best_epoch"]),
        "best_macro_auc": float(overfit_metrics["val_macro_auc"]),
        "completed_epoch": int(history["epoch"].max()),
        "parameter_count": parameter_count,
        "trainable_parameter_count": audit["trainable_parameter_count"],
        "frozen_parameter_count": audit["frozen_parameter_count"],
        "train_rows": int(overfit_metrics["train_rows"]),
        "val_rows": int(overfit_metrics["val_rows"]),
        "feature_scaler_sha256": scaler.payload_sha256 if scaler else None,
        "canonical_scaler_file_sha256": feature_meta["canonical_scaler_file_sha256"],
        "seed_info": seed_info,
        "config_sha256": metadata["config_sha256"],
        "implementation_signature": metadata["implementation_signature"],
        "initialized_from_full_checkpoint": False,
        "historical_inputs_modified": False,
        "best_checkpoint_sha256": sha256_file(run_dir / "best_macro_auc.pth"),
        "last_checkpoint_sha256": sha256_file(run_dir / "last_checkpoint.pth"),
        "artifact_sha256": {
            name: sha256_file(run_dir / name) for name in artifact_names
        },
    }
    with (run_dir / "fold_manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, ensure_ascii=False, indent=2)
    return run_dir


def main() -> None:
    args = parse_args()
    config_path = project_path(args.config)
    config = load_yaml(config_path)
    strategy = validate_trainability_strategy(args.strategy)
    if args.variant not in ALLOWED_VARIANTS:
        raise ValueError(f"variant must be one of {ALLOWED_VARIANTS}")
    configured_root = project_path(config["experiment"]["output_root"])
    output_root = project_path(args.output_root or configured_root)
    if output_root != configured_root:
        raise ValueError(f"Experiment may write only to {configured_root}")
    for fold in dict.fromkeys(args.fold):
        path = train_fold(
            config,
            config_path,
            strategy,
            args.variant,
            fold,
            output_root,
            resume=args.resume,
            overwrite=args.overwrite,
            smoke_test=args.smoke_test,
            allow_cpu_training=args.allow_cpu_training,
            allow_reference_mismatch=args.allow_reference_mismatch,
        )
        print(f"RUN_DIR={path}")


if __name__ == "__main__":
    main()
