"""Generate Figure 2 from pooled out-of-fold predictions.

The script intentionally calculates ROC curves and AUC values from
``binary_label`` and ``prob_patient`` only.  It does not use fold-wise metrics
or hard class predictions for any Figure 2 calculation.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np
import pandas as pd
from sklearn.metrics import auc, roc_curve


# Keep all input and output locations together for reproducibility.
DATA_DIR = Path(__file__).resolve().parent
PREDICTIONS_PATH = DATA_DIR / "pooled_oof_predictions_long.csv"
INTEGRITY_CHECKS_PATH = DATA_DIR / "integrity_checks.csv"
MODEL_INVENTORY_PATH = DATA_DIR / "model_inventory.csv"
OUTPUT_DIR = DATA_DIR
PNG_PATH = OUTPUT_DIR / "Figure2_ROC_curves.png"
PDF_PATH = OUTPUT_DIR / "Figure2_ROC_curves.pdf"
SVG_PATH = OUTPUT_DIR / "Figure2_ROC_curves.svg"
TIFF_PATH = OUTPUT_DIR / "Figure2_ROC_curves.tiff"
AUC_CHECK_PATH = OUTPUT_DIR / "Figure2_AUC_check.csv"

VIEW_CONFIG = {
    "whole_face_sh93": ("A", "Whole-face"),
    "eye_roi": ("B", "Eye"),
    "cheek_roi": ("C", "Cheek"),
    "lip_roi": ("D", "Lip"),
}
BACKBONE_CONFIG = {
    "resnet18": ("ResNet-18", "#0072B2"),
    "resnet34": ("ResNet-34", "#009E73"),
    "resnet50": ("ResNet-50", "#D55E00"),
}
EXPECTED_AUCS = {
    ("whole_face_sh93", "resnet18"): 0.8207,
    ("whole_face_sh93", "resnet34"): 0.7980,
    ("whole_face_sh93", "resnet50"): 0.8233,
    ("eye_roi", "resnet18"): 0.8177,
    ("eye_roi", "resnet34"): 0.7737,
    ("eye_roi", "resnet50"): 0.8116,
    ("cheek_roi", "resnet18"): 0.8065,
    ("cheek_roi", "resnet34"): 0.8091,
    ("cheek_roi", "resnet50"): 0.8268,
    ("lip_roi", "resnet18"): 0.7824,
    ("lip_roi", "resnet34"): 0.7861,
    ("lip_roi", "resnet50"): 0.7484,
}
EXPECTED_MODEL_KEYS = set(EXPECTED_AUCS)
REQUIRED_PREDICTION_COLUMNS = {
    "model_id",
    "model_display_name",
    "input_view",
    "backbone",
    "sample_id",
    "binary_label",
    "prob_patient",
    "pred_class",
    "fold",
}
INTEGRITY_BOOLEAN_COLUMNS = (
    "run_finished",
    "oof_ids_unique",
    "fold_metrics_match_oof",
    "pooled_confusion_matches_oof",
    "cross_model_oof_alignment",
)


def choose_font() -> str:
    """Prefer Arial, while keeping a deterministic sans-serif fallback."""
    available = {font.name for font in font_manager.fontManager.ttflist}
    for font_name in ("Arial", "Helvetica", "DejaVu Sans"):
        if font_name in available:
            return font_name
    return "sans-serif"


def normalise_bool(value: Any) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, str):
        return value.strip().lower() == "true"
    return bool(value)


def read_inputs() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    predictions = pd.read_csv(PREDICTIONS_PATH, dtype={"sample_id": str})
    integrity = pd.read_csv(INTEGRITY_CHECKS_PATH)
    inventory = pd.read_csv(MODEL_INVENTORY_PATH)
    missing_columns = REQUIRED_PREDICTION_COLUMNS - set(predictions.columns)
    if missing_columns:
        raise ValueError(f"Missing prediction columns: {sorted(missing_columns)}")
    return predictions, integrity, inventory


def calculate_roc_metrics(group: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, float]:
    labels = pd.to_numeric(group["binary_label"], errors="raise").astype(int)
    probabilities = pd.to_numeric(group["prob_patient"], errors="raise").astype(float)
    if set(labels.unique()) != {0, 1}:
        raise ValueError(f"binary_label must contain Control=0 and Patient=1, got {sorted(labels.unique())}")
    if not np.isfinite(probabilities).all() or not probabilities.between(0.0, 1.0).all():
        raise ValueError("prob_patient contains non-finite values or values outside [0, 1]")
    fpr, tpr, _ = roc_curve(labels, probabilities, pos_label=1)
    return fpr, tpr, float(auc(fpr, tpr))


def validate_integrity_table(integrity: pd.DataFrame, inventory: pd.DataFrame) -> None:
    required_integrity = {"model_id", "oof_rows", "fold_metrics_rows", *INTEGRITY_BOOLEAN_COLUMNS}
    missing_columns = required_integrity - set(integrity.columns)
    if missing_columns:
        raise ValueError(f"Missing integrity-check columns: {sorted(missing_columns)}")

    integrity_ids = set(integrity["model_id"])
    inventory_ids = set(inventory["model_id"])
    if integrity_ids != inventory_ids:
        raise ValueError("model_inventory.csv and integrity_checks.csv have different model IDs")
    if len(integrity_ids) != 12:
        raise ValueError(f"Expected 12 integrity-check rows, found {len(integrity_ids)}")

    failures: list[str] = []
    for _, row in integrity.iterrows():
        model_id = row["model_id"]
        if int(row["oof_rows"]) != 500 or int(row["fold_metrics_rows"]) != 5:
            failures.append(f"{model_id}: oof_rows={row['oof_rows']}, fold_metrics_rows={row['fold_metrics_rows']}")
        false_flags = [column for column in INTEGRITY_BOOLEAN_COLUMNS if not normalise_bool(row[column])]
        if false_flags:
            failures.append(f"{model_id}: failed flags={', '.join(false_flags)}")
    if failures:
        raise ValueError("Integrity-check table failed:\n" + "\n".join(failures))


def validate_and_calculate(predictions: pd.DataFrame) -> tuple[dict[tuple[str, str], tuple[np.ndarray, np.ndarray]], pd.DataFrame]:
    observed_keys = set(zip(predictions["input_view"], predictions["backbone"]))
    if observed_keys != EXPECTED_MODEL_KEYS:
        missing = EXPECTED_MODEL_KEYS - observed_keys
        unexpected = observed_keys - EXPECTED_MODEL_KEYS
        raise ValueError(f"Unexpected input/model mapping. Missing={sorted(missing)}, unexpected={sorted(unexpected)}")

    curves: dict[tuple[str, str], tuple[np.ndarray, np.ndarray]] = {}
    records: list[dict[str, object]] = []
    failures: list[str] = []

    for input_view in VIEW_CONFIG:
        for backbone in BACKBONE_CONFIG:
            key = (input_view, backbone)
            group = predictions.loc[(predictions["input_view"] == input_view) & (predictions["backbone"] == backbone)].copy()
            model_name, _ = BACKBONE_CONFIG[backbone]
            input_name = VIEW_CONFIG[input_view][1]
            n_samples = len(group)
            control_count = int((group["binary_label"] == 0).sum())
            patient_count = int((group["binary_label"] == 1).sum())
            unique_ids = group["sample_id"].nunique()
            model_ids = group["model_id"].nunique()
            n_folds = group["fold"].nunique()

            if n_samples != 500 or control_count != 115 or patient_count != 385 or unique_ids != 500 or model_ids != 1 or n_folds != 5:
                failures.append(
                    f"{input_name} {model_name}: N={n_samples}, Control={control_count}, Patient={patient_count}, "
                    f"unique sample_id={unique_ids}, model_id count={model_ids}, fold count={n_folds}"
                )
                continue

            fpr, tpr, auc_value = calculate_roc_metrics(group)
            expected_auc = EXPECTED_AUCS[key]
            difference = auc_value - expected_auc
            if not np.isclose(auc_value, expected_auc, atol=0.00005, rtol=0.0):
                failures.append(
                    f"{input_name} {model_name}: calculated AUC={auc_value:.6f}, expected AUC={expected_auc:.4f}, "
                    f"difference={difference:+.6f}, N={n_samples}, Control={control_count}, Patient={patient_count}"
                )
                continue

            curves[key] = (fpr, tpr)
            records.append(
                {
                    "Input": input_name,
                    "Model": model_name,
                    "N": n_samples,
                    "Control": control_count,
                    "Patient": patient_count,
                    "AUC": auc_value,
                }
            )

    if failures:
        print("## Figure 2 QC failures")
        print("\n".join(failures))
        raise RuntimeError("Figure 2 was not generated because pooled OOF QC failed.")

    checks = pd.DataFrame(records)
    return curves, checks


def plot_figure(curves: dict[tuple[str, str], tuple[np.ndarray, np.ndarray]], checks: pd.DataFrame) -> None:
    plt.rcParams.update(
        {
            "font.family": choose_font(),
            "font.size": 9,
            "axes.linewidth": 0.8,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
        }
    )
    # Compact two-column layout sized for a single manuscript column (about 89 mm).
    figure, axes = plt.subplots(2, 2, figsize=(3.5, 5.1), facecolor="white")
    figure.subplots_adjust(left=0.14, right=0.98, bottom=0.08, top=0.95, wspace=0.10, hspace=0.05)
    ticks = np.linspace(0.0, 1.0, 6)

    for panel_index, (axis, input_view) in enumerate(zip(axes.flat, VIEW_CONFIG)):
        letter, input_name = VIEW_CONFIG[input_view]
        axis.plot([0, 1], [0, 1], color="#A6A6A6", linestyle="--", linewidth=0.75, zorder=1)
        for backbone in BACKBONE_CONFIG:
            model_name, color = BACKBONE_CONFIG[backbone]
            fpr, tpr = curves[(input_view, backbone)]
            auc_value = checks.loc[(checks["Input"] == input_name) & (checks["Model"] == model_name), "AUC"].iloc[0]
            axis.plot(fpr, tpr, color=color, linewidth=1.35, label=f"{model_name} (AUC = {auc_value:.4f})", zorder=2)

        axis.set_xlim(0.0, 1.0)
        axis.set_ylim(0.0, 1.0)
        axis.set_xticks(ticks)
        axis.set_yticks(ticks)
        axis.set_xlabel("1 - Specificity", fontsize=7)
        axis.tick_params(labelsize=6)
        if panel_index % 2 == 0:
            axis.set_ylabel("Sensitivity", fontsize=7)
        else:
            axis.tick_params(labelleft=False)
        axis.set_box_aspect(1)
        axis.set_title(input_name, fontsize=8, pad=6)
        label_x = -0.14 if panel_index % 2 == 0 else -0.09
        axis.text(label_x, 1.10, letter, transform=axis.transAxes, fontsize=8, fontweight="bold", va="top", clip_on=False)
        axis.legend(loc="lower right", frameon=False, fontsize=5.5, handlelength=1.2)

    figure.savefig(PNG_PATH, dpi=600, facecolor="white")
    figure.savefig(PDF_PATH, format="pdf", facecolor="white")
    figure.savefig(SVG_PATH, format="svg", facecolor="white")
    figure.savefig(TIFF_PATH, dpi=600, format="tiff", facecolor="white")
    plt.close(figure)


def print_qc_summary(checks: pd.DataFrame) -> None:
    print("## Figure 2 QC")
    for _, row in checks.iterrows():
        print(
            f"{row['Input']} {row['Model']}: N={row['N']}, Control={row['Control']}, "
            f"Patient={row['Patient']}, AUC={row['AUC']:.4f} PASS"
        )
    print("All 12 pooled OOF datasets passed integrity checks.")
    print(f"PNG: {PNG_PATH}")
    print(f"PDF: {PDF_PATH}")
    print(f"AUC check: {AUC_CHECK_PATH}")


def main() -> None:
    predictions, integrity, inventory = read_inputs()
    validate_integrity_table(integrity, inventory)
    curves, checks = validate_and_calculate(predictions)
    checks.to_csv(AUC_CHECK_PATH, index=False, float_format="%.4f")
    plot_figure(curves, checks)
    print_qc_summary(checks)


if __name__ == "__main__":
    main()
