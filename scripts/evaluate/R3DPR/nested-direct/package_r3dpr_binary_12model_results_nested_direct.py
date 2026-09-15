"""Package validated nested-direct pooled OOF results for the prespecified 12-model analysis."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[4]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from metrics.R3DPR.binary_classification_metrics import compute_binary_metrics, flatten_metrics
from scripts.evaluate.R3DPR.summarize_r3dpr_resnet18_binary_5fold import (
    METRICS,
    expected_table,
    validate_oof,
)


DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "dock" / "Face2_sync" / "paper" / "R3DPR_binary_12model_results_nested_direct"
PROTOCOL = "outer_5fold_inner_holdout_direct_test"
FOLD_METRICS = ["macro_auc", "balanced_accuracy", "macro_f1", "sensitivity", "specificity"]
OOF_KEY_COLUMNS = ["sample_id", "patient_group_id", "fold", "binary_label"]

# This is intentionally explicit: eye2_roi is exploratory and is not one of the prespecified four input views.
MAIN_CONFIG_NAMES = [
    "r3dpr_resnet18_binary_256x320_relight_neutral_front_fullbaseline_ls005_nested_direct_5fold.yaml",
    "r3dpr_resnet34_binary_256x320_relight_neutral_front_fullbaseline_ls005_nested_direct_5fold.yaml",
    "r3dpr_resnet50_binary_256x320_relight_neutral_front_fullbaseline_ls005_nested_direct_5fold.yaml",
    "r3dpr_resnet18_binary_224x224_manual_shift_eye_roi_fullbaseline_ls005_nested_direct_5fold.yaml",
    "r3dpr_resnet34_binary_224x224_manual_shift_eye_roi_fullbaseline_ls005_nested_direct_5fold.yaml",
    "r3dpr_resnet50_binary_224x224_manual_shift_eye_roi_fullbaseline_ls005_nested_direct_5fold.yaml",
    "r3dpr_resnet18_binary_224x224_manual_shift_cheek_roi_fullbaseline_ls005_nested_direct_5fold.yaml",
    "r3dpr_resnet34_binary_224x224_manual_shift_cheek_roi_fullbaseline_ls005_nested_direct_5fold.yaml",
    "r3dpr_resnet50_binary_224x224_manual_shift_cheek_roi_fullbaseline_ls005_nested_direct_5fold.yaml",
    "r3dpr_resnet18_binary_224x224_manual_shift_lip_roi_fullbaseline_ls005_nested_direct_5fold.yaml",
    "r3dpr_resnet34_binary_224x224_manual_shift_lip_roi_fullbaseline_ls005_nested_direct_5fold.yaml",
    "r3dpr_resnet50_binary_224x224_manual_shift_lip_roi_fullbaseline_ls005_nested_direct_5fold.yaml",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def project_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def read_yaml(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as stream:
        return yaml.safe_load(stream)


def main_config_paths() -> list[Path]:
    config_dir = PROJECT_ROOT / "config" / "train" / "R3DPR" / "nested-direct"
    paths = [config_dir / name for name in MAIN_CONFIG_NAMES]
    missing = [path for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing prespecified nested-direct configs: {missing}")
    return paths


def model_metadata(config_path: Path, config: dict, experiment_dir: Path) -> dict[str, object]:
    image_root = Path(config["data"]["image_root"])
    backbone = str(config["model"]["backbone"]).lower()
    if image_root.name == "SH093" or image_root.parent.name == "SH093":
        input_view, display_prefix = "whole_face_sh93", "Whole-face SH93"
    else:
        input_view = image_root.name
        display_prefix = f"{image_root.name.removesuffix('_roi').replace('_', ' ').title()} ROI"
    return {
        "model_id": f"{input_view}_{backbone}",
        "model_display_name": f"{display_prefix} {backbone.replace('resnet', 'ResNet')}",
        "input_view": input_view,
        "backbone": backbone,
        "input_width": int(config["data"]["image_width"]),
        "input_height": int(config["data"]["image_height"]),
        "source_config": str(config_path.relative_to(PROJECT_ROOT)),
        "source_experiment_dir": str(experiment_dir.relative_to(PROJECT_ROOT)),
        "protocol": str(config["nested_direct"]["protocol"]),
        "loss": str(config["train"]["loss"]),
        "label_smoothing_alpha": float(config["train"]["label_smoothing_alpha"]),
        "trainability_strategy": str(config["model"]["trainability_strategy"]),
    }


def prepend_metadata(frame: pd.DataFrame, metadata: dict[str, object]) -> pd.DataFrame:
    """Add source metadata without silently replacing a same-named experiment column."""
    result = frame.copy()
    for key, value in reversed(list(metadata.items())):
        if key in result.columns:
            if not result[key].astype(str).eq(str(value)).all():
                raise ValueError(f"Existing {key} column does not match registered metadata")
        else:
            result.insert(0, key, value)
    return result


def read_confusion(path: Path) -> np.ndarray:
    matrix = pd.read_csv(path, index_col=0).to_numpy(dtype=int)
    if matrix.shape != (2, 2):
        raise ValueError(f"Expected a 2x2 confusion matrix: {path}")
    return matrix


def validate_fold_metrics(oof: pd.DataFrame, fold_metrics: pd.DataFrame, n_folds: int) -> None:
    required = {"fold", "selected_epoch", "inner_best_macro_auc", *FOLD_METRICS}
    missing = sorted(required.difference(fold_metrics.columns))
    if missing:
        raise ValueError(f"nested-direct fold_metrics.csv lacks required columns: {missing}")
    if len(fold_metrics) != n_folds or set(fold_metrics["fold"].astype(int)) != set(range(n_folds)):
        raise ValueError("fold_metrics.csv must contain exactly one row for every outer fold")
    indexed = fold_metrics.set_index("fold")
    for fold in range(n_folds):
        frame = oof.loc[oof["fold"].astype(int) == fold]
        computed = flatten_metrics(
            compute_binary_metrics(
                frame["binary_label"].to_numpy(int),
                frame[["prob_control", "prob_patient"]].to_numpy(float),
            )
        )
        for metric in FOLD_METRICS:
            if not np.isclose(float(indexed.loc[fold, metric]), float(computed[metric]), atol=1e-10):
                raise ValueError(f"fold={fold} {metric} differs from OOF prediction-derived value")


def write_readme(output_dir: Path, inventory: pd.DataFrame) -> None:
    model_rows = "\n".join(
        f"| {row.model_id} | {row.input_view} | {row.backbone} | {row.input_width}x{row.input_height} |"
        for row in inventory.itertuples(index=False)
    )
    content = f"""# R3DPR Binary 12-model Nested-Direct Results Package

This package is a derived, read-only organization of the completed prespecified nested-direct experiments. Original experiment directories remain the source of record.

## Protocol

- Each outer fold contains 400 development cases and an untouched 100-case outer test fold.
- The development cases are split using label x sex stratification into 320 internal training and 80 internal validation cases.
- Internal validation Macro-AUC selects the checkpoint. That same checkpoint directly predicts the outer test fold; no 400-case refitting stage is used.
- The primary results are pooled outer-held-out OOF metrics. Fold mean +/- SD is stability description only.
- This package contains the prespecified four input views x three backbones = 12 models. Exploratory `eye2_roi` experiments are excluded.

## Contents

- `model_inventory.csv`: source paths and protocol metadata for all 12 models.
- `pooled_oof_metrics.csv`: five pooled OOF primary metrics for each model.
- `foldwise_metrics_long.csv`: 60 unrounded outer-test fold rows, including selected epochs.
- `foldwise_metrics_mean_sd.csv`: fold mean and sample SD for stability description only.
- `pooled_oof_predictions/`: 12 complete 500-row OOF prediction files.
- `pooled_oof_predictions_long.csv`: all 6,000 model-sample OOF rows.
- `pooled_confusion_matrices.csv` and `confusion_matrices/`: matrices recomputed from OOF labels and hard predictions.
- `integrity_checks.csv`: ID, fold, metric, probability, protocol, and cross-model alignment checks.

## Model Inventory

| Model ID | Input view | Backbone | Input width x height |
|---|---|---|---:|
{model_rows}
"""
    (output_dir / "README.md").write_text(content, encoding="utf-8")


def package_results(output_dir: Path) -> Path:
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Output directory is not empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    oof_dir = output_dir / "pooled_oof_predictions"
    matrix_dir = output_dir / "confusion_matrices"
    oof_dir.mkdir()
    matrix_dir.mkdir()

    inventory_rows: list[dict[str, object]] = []
    pooled_rows: list[dict[str, object]] = []
    fold_rows: list[pd.DataFrame] = []
    oof_rows: list[pd.DataFrame] = []
    confusion_rows: list[dict[str, object]] = []
    integrity_rows: list[dict[str, object]] = []
    reference_keys: pd.DataFrame | None = None

    for config_path in main_config_paths():
        declared_config = read_yaml(config_path)
        experiment_dir = project_path(declared_config["experiment"]["output_dir"])
        required_paths = {
            "config_snapshot": experiment_dir / "config_snapshot.yaml",
            "run_finished": experiment_dir / "run_finished_at.txt",
            "fold_metrics": experiment_dir / "fold_metrics.csv",
            "oof_predictions": experiment_dir / "oof_predictions.csv",
            "oof_confusion": experiment_dir / "oof_confusion_matrix.csv",
        }
        missing = [name for name, path in required_paths.items() if not path.is_file()]
        if missing:
            raise FileNotFoundError(f"Incomplete experiment {experiment_dir}: missing {missing}")
        config = read_yaml(required_paths["config_snapshot"])
        if config.get("nested_direct", {}).get("protocol") != PROTOCOL:
            raise ValueError(f"Experiment is not nested-direct: {experiment_dir}")
        if config["experiment"]["output_dir"] != declared_config["experiment"]["output_dir"]:
            raise ValueError(f"Config snapshot output directory differs from registered config: {config_path}")
        metadata = model_metadata(config_path, config, experiment_dir)
        n_folds = int(config["data"]["n_folds"])
        oof = pd.read_csv(required_paths["oof_predictions"], dtype={"sample_id": "string", "patient_group_id": "string"})
        validate_oof(oof, expected_table(config), n_folds)
        if set(oof.get("protocol", [])) != {PROTOCOL}:
            raise ValueError(f"OOF predictions do not declare the nested-direct protocol: {experiment_dir}")
        fold_metrics = pd.read_csv(required_paths["fold_metrics"])
        validate_fold_metrics(oof, fold_metrics, n_folds)
        computed = compute_binary_metrics(
            oof["binary_label"].to_numpy(int), oof[["prob_control", "prob_patient"]].to_numpy(float)
        )
        saved_matrix = read_confusion(required_paths["oof_confusion"])
        if not np.array_equal(saved_matrix, computed["confusion_matrix"]):
            raise ValueError(f"Saved pooled confusion matrix differs from OOF prediction-derived value: {experiment_dir}")
        keys = oof.loc[:, OOF_KEY_COLUMNS].sort_values("sample_id").reset_index(drop=True)
        if reference_keys is None:
            reference_keys = keys
        elif not np.array_equal(keys.to_numpy(), reference_keys.to_numpy()):
            raise ValueError(f"OOF rows are not aligned with the other main models: {metadata['model_id']}")

        inventory_rows.append({**metadata, "n_oof_samples": len(oof), "n_folds": n_folds})
        pooled_rows.append({**metadata, "n_oof_samples": len(oof), **flatten_metrics(computed)})
        fold_export = prepend_metadata(
            fold_metrics.loc[:, ["fold", "protocol", "selected_epoch", "inner_best_macro_auc", *FOLD_METRICS]], metadata
        )
        fold_rows.append(fold_export)
        oof_export = prepend_metadata(oof, metadata)
        oof_export.to_csv(oof_dir / f"{metadata['model_id']}_oof_predictions.csv", index=False, encoding="utf-8-sig")
        oof_rows.append(oof_export)
        matrix = computed["confusion_matrix"]
        matrix_frame = pd.DataFrame(matrix, index=["Control", "Patient"], columns=["Predicted_Control", "Predicted_Patient"])
        matrix_frame.index.name = "True_class"
        matrix_frame.to_csv(matrix_dir / f"{metadata['model_id']}_confusion_matrix.csv", encoding="utf-8-sig")
        confusion_rows.append({**metadata, "tn": int(matrix[0, 0]), "fp": int(matrix[0, 1]), "fn": int(matrix[1, 0]), "tp": int(matrix[1, 1]), "n_samples": int(matrix.sum())})
        integrity_rows.append(
            {
                **metadata,
                "run_finished": True,
                "nested_direct_protocol": True,
                "oof_rows": len(oof),
                "oof_ids_unique": not oof["sample_id"].duplicated().any(),
                "fold_metrics_rows": len(fold_metrics),
                "fold_metrics_match_oof": True,
                "pooled_confusion_matches_oof": True,
                "cross_model_oof_alignment": True,
            }
        )

    inventory = pd.DataFrame(inventory_rows).sort_values(["input_view", "backbone"])
    pooled = pd.DataFrame(pooled_rows).sort_values(["input_view", "backbone"])
    foldwise = pd.concat(fold_rows, ignore_index=True).sort_values(["input_view", "backbone", "fold"])
    all_oof = pd.concat(oof_rows, ignore_index=True).sort_values(["input_view", "backbone", "sample_id"])
    confusion = pd.DataFrame(confusion_rows).sort_values(["input_view", "backbone"])
    integrity = pd.DataFrame(integrity_rows).sort_values(["input_view", "backbone"])
    summary_rows: list[dict[str, object]] = []
    for metadata in inventory.to_dict(orient="records"):
        subset = foldwise.loc[foldwise["model_id"] == metadata["model_id"]]
        row = {key: metadata[key] for key in metadata if key != "n_oof_samples"}
        row["n_folds"] = len(subset)
        for metric in FOLD_METRICS:
            row[f"{metric}_mean"] = float(subset[metric].mean())
            row[f"{metric}_sd"] = float(subset[metric].std(ddof=1))
        summary_rows.append(row)

    inventory.to_csv(output_dir / "model_inventory.csv", index=False, encoding="utf-8-sig")
    pooled.to_csv(output_dir / "pooled_oof_metrics.csv", index=False, encoding="utf-8-sig")
    foldwise.to_csv(output_dir / "foldwise_metrics_long.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(summary_rows).sort_values(["input_view", "backbone"]).to_csv(
        output_dir / "foldwise_metrics_mean_sd.csv", index=False, encoding="utf-8-sig"
    )
    all_oof.to_csv(output_dir / "pooled_oof_predictions_long.csv", index=False, encoding="utf-8-sig")
    confusion.to_csv(output_dir / "pooled_confusion_matrices.csv", index=False, encoding="utf-8-sig")
    integrity.to_csv(output_dir / "integrity_checks.csv", index=False, encoding="utf-8-sig")
    write_readme(output_dir, inventory)
    return output_dir


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir if args.output_dir.is_absolute() else PROJECT_ROOT / args.output_dir
    packaged = package_results(output_dir)
    print(f"R3DPR_NESTED_DIRECT_12MODEL_PACKAGE={packaged}")


if __name__ == "__main__":
    main()
