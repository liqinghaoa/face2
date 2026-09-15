"""Package validated fold-wise, pooled OOF, and confusion-matrix results for 12 R3DPR models."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from metrics.R3DPR.binary_classification_metrics import compute_binary_metrics, flatten_metrics
from scripts.evaluate.R3DPR.summarize_r3dpr_resnet18_binary_5fold import (
    METRIC_LABELS,
    METRICS,
    expected_table,
    validate_oof,
)


DEFAULT_OUTPUT_DIR = (
    PROJECT_ROOT / "dock" / "Face2_sync" / "paper" / "R3DPR_binary_12model_results"
)
FOLD_METRIC_COLUMNS = ["macro_auc", "balanced_accuracy", "macro_f1", "sensitivity", "specificity"]
OOF_KEY_COLUMNS = ["sample_id", "patient_group_id", "fold", "binary_label"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def discover_configs() -> list[Path]:
    config_dir = PROJECT_ROOT / "config" / "train" / "R3DPR"
    whole_face = list(
        config_dir.glob(
            "r3dpr_resnet*_binary_256x320_relight_neutral_front_fullbaseline_ls005_5fold.yaml"
        )
    )
    roi = list(
        config_dir.glob(
            "r3dpr_resnet*_binary_224x224_manual_shift_*_fullbaseline_ls005_5fold.yaml"
        )
    )
    paths = sorted(whole_face + roi)
    if len(paths) != 12:
        raise ValueError(f"Expected exactly 12 configured experiments, found {len(paths)}: {paths}")
    return paths


def read_config(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as stream:
        return yaml.safe_load(stream)


def project_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def model_metadata(config_path: Path, config: dict, experiment_dir: Path) -> dict[str, object]:
    image_root = Path(config["data"]["image_root"])
    backbone = str(config["model"]["backbone"]).lower()
    if image_root.name == "SH093" or image_root.parent.name == "SH093":
        input_view = "whole_face_sh93"
        display_name = f"Whole-face SH93 {backbone.replace('resnet', 'ResNet')}"
    else:
        input_view = image_root.name
        roi_name = image_root.name.removesuffix("_roi").replace("_", " ").title()
        display_name = f"{roi_name} ROI {backbone.replace('resnet', 'ResNet')}"
    return {
        "model_id": f"{input_view}_{backbone}",
        "model_display_name": display_name,
        "input_view": input_view,
        "backbone": backbone,
        "input_width": int(config["data"]["image_width"]),
        "input_height": int(config["data"]["image_height"]),
        "source_config": str(config_path.relative_to(PROJECT_ROOT)),
        "source_experiment_dir": str(experiment_dir.relative_to(PROJECT_ROOT)),
        "loss": str(config["train"]["loss"]),
        "label_smoothing_alpha": float(config["train"]["label_smoothing_alpha"]),
        "trainability_strategy": str(config["model"]["trainability_strategy"]),
    }


def read_saved_confusion(path: Path) -> np.ndarray:
    saved = pd.read_csv(path, index_col=0).to_numpy(dtype=int)
    if saved.shape != (2, 2):
        raise ValueError(f"Expected a 2x2 confusion matrix: {path}")
    return saved


def validate_fold_metrics(oof: pd.DataFrame, fold_metrics: pd.DataFrame, n_folds: int) -> None:
    required = {"fold", "best_epoch", *FOLD_METRIC_COLUMNS}
    missing = sorted(required.difference(fold_metrics.columns))
    if missing:
        raise ValueError(f"fold_metrics.csv lacks required columns: {missing}")
    if len(fold_metrics) != n_folds or set(fold_metrics["fold"].astype(int)) != set(range(n_folds)):
        raise ValueError("fold_metrics.csv does not contain one row for every fold")
    indexed = fold_metrics.set_index("fold")
    for fold in range(n_folds):
        frame = oof.loc[oof["fold"].astype(int) == fold]
        derived = flatten_metrics(
            compute_binary_metrics(
                frame["binary_label"].to_numpy(int),
                frame[["prob_control", "prob_patient"]].to_numpy(float),
            )
        )
        for metric in FOLD_METRIC_COLUMNS:
            if not np.isclose(float(indexed.loc[fold, metric]), float(derived[metric]), atol=1e-10):
                raise ValueError(f"fold={fold} {metric} differs from the OOF prediction-derived value")


def write_readme(output_dir: Path, inventory: pd.DataFrame) -> None:
    model_rows = "\n".join(
        f"| {row.model_id} | {row.input_view} | {row.backbone} | {row.input_width}x{row.input_height} |"
        for row in inventory.itertuples(index=False)
    )
    content = f"""# R3DPR Binary 12-model Results Package

This package is a derived, read-only organization of the completed R3DPR binary experiments. Original experiment directories remain the source of record.

## Contents

- `model_inventory.csv`: configuration and source-path registry for all 12 models.
- `foldwise_metrics_long.csv`: 60 unrounded rows (12 models x 5 folds).
- `foldwise_metrics_mean_sd.csv`: fold mean and sample SD (`ddof=1`) for stability description only.
- `pooled_oof_predictions/`: 12 complete 500-row OOF prediction files retaining source prediction columns plus model metadata.
- `pooled_oof_predictions_long.csv`: all 6,000 model-sample OOF rows in long format.
- `pooled_confusion_matrices.csv` and `confusion_matrices/`: pooled OOF confusion matrices recomputed from `binary_label` and `pred_class`.
- `integrity_checks.csv`: validation results recorded during packaging.

## Interpretation Rules

- Pooled OOF metrics are the primary results; fold mean +/- SD describes fold stability only.
- Each model's OOF file has one held-out prediction per participant.
- Confusion matrices are recomputed from unrounded OOF labels and predictions, never reconstructed from displayed sensitivity or specificity.
- The 12 OOF files share identical `sample_id`, `patient_group_id`, fold, and true-label assignments, permitting paired prediction-level comparisons.

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
    fold_rows: list[pd.DataFrame] = []
    oof_rows: list[pd.DataFrame] = []
    confusion_rows: list[dict[str, object]] = []
    integrity_rows: list[dict[str, object]] = []
    reference_keys: pd.DataFrame | None = None

    for config_path in discover_configs():
        config = read_config(config_path)
        experiment_dir = project_path(config["experiment"]["output_dir"])
        n_folds = int(config["data"]["n_folds"])
        metadata = model_metadata(config_path, config, experiment_dir)
        required_paths = {
            "config_snapshot": experiment_dir / "config_snapshot.yaml",
            "run_finished": experiment_dir / "run_finished_at.txt",
            "fold_metrics": experiment_dir / "fold_metrics.csv",
            "oof_predictions": experiment_dir / "oof_predictions.csv",
            "oof_confusion": experiment_dir / "oof_confusion_matrix.csv",
        }
        missing = [name for name, path in required_paths.items() if not path.is_file()]
        if missing:
            raise FileNotFoundError(f"{metadata['model_id']} is incomplete: missing {missing}")

        oof = pd.read_csv(
            required_paths["oof_predictions"],
            dtype={"sample_id": "string", "patient_group_id": "string"},
        )
        validate_oof(oof, expected_table(config), n_folds)
        fold_metrics = pd.read_csv(required_paths["fold_metrics"])
        validate_fold_metrics(oof, fold_metrics, n_folds)

        saved_matrix = read_saved_confusion(required_paths["oof_confusion"])
        computed_matrix = compute_binary_metrics(
            oof["binary_label"].to_numpy(int),
            oof[["prob_control", "prob_patient"]].to_numpy(float),
        )["confusion_matrix"]
        if not np.array_equal(saved_matrix, computed_matrix):
            raise ValueError(f"{metadata['model_id']} saved pooled confusion matrix does not match OOF")

        keys = oof.loc[:, OOF_KEY_COLUMNS].sort_values("sample_id").reset_index(drop=True)
        if reference_keys is None:
            reference_keys = keys
        elif not np.array_equal(keys.to_numpy(), reference_keys.to_numpy()):
            raise ValueError(f"{metadata['model_id']} OOF rows are not aligned with the other models")

        inventory_rows.append({**metadata, "n_oof_samples": len(oof), "n_folds": n_folds})
        fold_export = fold_metrics.loc[:, ["fold", "best_epoch", *FOLD_METRIC_COLUMNS]].copy()
        for key, value in metadata.items():
            fold_export.insert(0, key, value)
        fold_rows.append(fold_export)

        oof_export = oof.copy()
        for key, value in reversed(list(metadata.items())):
            oof_export.insert(0, key, value)
        oof_export.to_csv(oof_dir / f"{metadata['model_id']}_oof_predictions.csv", index=False, encoding="utf-8-sig")
        oof_rows.append(oof_export)

        matrix_frame = pd.DataFrame(
            computed_matrix,
            index=["Control", "Patient"],
            columns=["Predicted_Control", "Predicted_Patient"],
        )
        matrix_frame.index.name = "True_class"
        matrix_frame.to_csv(matrix_dir / f"{metadata['model_id']}_confusion_matrix.csv", encoding="utf-8-sig")
        confusion_rows.append(
            {
                **metadata,
                "tn": int(computed_matrix[0, 0]),
                "fp": int(computed_matrix[0, 1]),
                "fn": int(computed_matrix[1, 0]),
                "tp": int(computed_matrix[1, 1]),
                "n_samples": int(computed_matrix.sum()),
            }
        )
        integrity_rows.append(
            {
                **metadata,
                "run_finished": True,
                "oof_rows": len(oof),
                "oof_ids_unique": not oof["sample_id"].duplicated().any(),
                "fold_metrics_rows": len(fold_metrics),
                "fold_metrics_match_oof": True,
                "pooled_confusion_matches_oof": True,
                "cross_model_oof_alignment": True,
            }
        )

    inventory = pd.DataFrame(inventory_rows).sort_values(["input_view", "backbone"])
    foldwise = pd.concat(fold_rows, ignore_index=True).sort_values(["input_view", "backbone", "fold"])
    all_oof = pd.concat(oof_rows, ignore_index=True).sort_values(["input_view", "backbone", "sample_id"])
    confusion = pd.DataFrame(confusion_rows).sort_values(["input_view", "backbone"])
    integrity = pd.DataFrame(integrity_rows).sort_values(["input_view", "backbone"])

    summary_rows: list[dict[str, object]] = []
    for metadata in inventory.to_dict(orient="records"):
        subset = foldwise.loc[foldwise["model_id"] == metadata["model_id"]]
        row = {key: metadata[key] for key in metadata if key != "n_oof_samples"}
        row["n_folds"] = len(subset)
        for metric in FOLD_METRIC_COLUMNS:
            row[f"{metric}_mean"] = float(subset[metric].mean())
            row[f"{metric}_sd"] = float(subset[metric].std(ddof=1))
        summary_rows.append(row)
    summary = pd.DataFrame(summary_rows).sort_values(["input_view", "backbone"])

    inventory.to_csv(output_dir / "model_inventory.csv", index=False, encoding="utf-8-sig")
    foldwise.to_csv(output_dir / "foldwise_metrics_long.csv", index=False, encoding="utf-8-sig")
    summary.to_csv(output_dir / "foldwise_metrics_mean_sd.csv", index=False, encoding="utf-8-sig")
    all_oof.to_csv(output_dir / "pooled_oof_predictions_long.csv", index=False, encoding="utf-8-sig")
    confusion.to_csv(output_dir / "pooled_confusion_matrices.csv", index=False, encoding="utf-8-sig")
    integrity.to_csv(output_dir / "integrity_checks.csv", index=False, encoding="utf-8-sig")
    write_readme(output_dir, inventory)
    return output_dir


def main() -> Path:
    args = parse_args()
    output_dir = args.output_dir if args.output_dir.is_absolute() else PROJECT_ROOT / args.output_dir
    packaged = package_results(output_dir)
    print(f"R3DPR_12MODEL_PACKAGE={packaged}")
    return packaged


if __name__ == "__main__":
    main()
