"""Generate Figure 4 from formal pooled out-of-fold predictions.

Each panel uses a prespecified displayed model for one facial input. Cell colour
encodes row-normalized proportion only, while the annotation reports both the
raw count and row-normalized percentage.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
from matplotlib import colormaps, font_manager
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, roc_auc_score


# Keep all source and output paths in one place for a reproducible figure.
DATA_DIR = Path(__file__).resolve().parent
POOLED_OOF_PATH = DATA_DIR / "pooled_oof_predictions_long.csv"
POOLED_CONFUSION_PATH = DATA_DIR / "pooled_confusion_matrices.csv"
MODEL_INVENTORY_PATH = DATA_DIR / "model_inventory.csv"
INTEGRITY_CHECKS_PATH = DATA_DIR / "integrity_checks.csv"
OUTPUT_DIR = DATA_DIR
PNG_PATH = OUTPUT_DIR / "Figure4_confusion_matrices.png"
PDF_PATH = OUTPUT_DIR / "Figure4_confusion_matrices.pdf"
CHECK_PATH = OUTPUT_DIR / "Figure4_confusion_matrix_check.csv"

VIEW_CONFIG = {
    "whole_face_sh93": ("A", "Whole-face"),
    "eye_roi": ("B", "Eye"),
    "cheek_roi": ("C", "Cheek"),
    "lip_roi": ("D", "Lip"),
}
BACKBONE_CONFIG = {
    "resnet18": "ResNet-18",
    "resnet34": "ResNet-34",
    "resnet50": "ResNet-50",
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
DISPLAYED_BACKBONES = {
    "whole_face_sh93": "resnet18",
    "eye_roi": "resnet18",
    "cheek_roi": "resnet50",
    "lip_roi": "resnet34",
}
EXPECTED_CONFUSIONS = {
    ("whole_face_sh93", "resnet18"): np.array([[80, 35], [82, 303]]),
    ("eye_roi", "resnet18"): np.array([[74, 41], [83, 302]]),
    ("cheek_roi", "resnet50"): np.array([[90, 25], [105, 280]]),
    ("lip_roi", "resnet34"): np.array([[79, 36], [104, 281]]),
}
EXPECTED_KEYS = set(EXPECTED_AUCS)
REQUIRED_PREDICTION_COLUMNS = {
    "model_id",
    "model_display_name",
    "input_view",
    "backbone",
    "sample_id",
    "binary_label",
    "prob_patient",
    "pred_class",
}
REQUIRED_CONFUSION_COLUMNS = {"model_id", "input_view", "backbone", "tn", "fp", "fn", "tp", "n_samples"}
INTEGRITY_BOOLEAN_COLUMNS = (
    "run_finished",
    "oof_ids_unique",
    "fold_metrics_match_oof",
    "pooled_confusion_matches_oof",
    "cross_model_oof_alignment",
)


def choose_font() -> str:
    """Prefer Arial while retaining a clear sans-serif fallback."""
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


def require_columns(frame: pd.DataFrame, required: set[str], name: str) -> None:
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"{name} is missing required columns: {sorted(missing)}")


def read_inputs() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    predictions = pd.read_csv(POOLED_OOF_PATH, dtype={"sample_id": str})
    pooled_confusions = pd.read_csv(POOLED_CONFUSION_PATH)
    inventory = pd.read_csv(MODEL_INVENTORY_PATH)
    integrity = pd.read_csv(INTEGRITY_CHECKS_PATH)
    require_columns(predictions, REQUIRED_PREDICTION_COLUMNS, "pooled_oof_predictions_long.csv")
    require_columns(pooled_confusions, REQUIRED_CONFUSION_COLUMNS, "pooled_confusion_matrices.csv")
    require_columns(inventory, {"model_id", "n_oof_samples", "n_folds"}, "model_inventory.csv")
    require_columns(
        integrity,
        {"model_id", "oof_rows", "fold_metrics_rows", *INTEGRITY_BOOLEAN_COLUMNS},
        "integrity_checks.csv",
    )
    return predictions, pooled_confusions, inventory, integrity


def validate_metadata(inventory: pd.DataFrame, integrity: pd.DataFrame) -> None:
    if set(inventory["model_id"]) != set(integrity["model_id"]) or inventory["model_id"].nunique() != 12:
        raise ValueError("model_inventory.csv and integrity_checks.csv must contain the same 12 models")
    failures: list[str] = []
    for _, row in inventory.iterrows():
        if int(row["n_oof_samples"]) != 500 or int(row["n_folds"]) != 5:
            failures.append(f"{row['model_id']}: n_oof_samples={row['n_oof_samples']}, n_folds={row['n_folds']}")
    for _, row in integrity.iterrows():
        if int(row["oof_rows"]) != 500 or int(row["fold_metrics_rows"]) != 5:
            failures.append(f"{row['model_id']}: oof_rows={row['oof_rows']}, fold_metrics_rows={row['fold_metrics_rows']}")
        false_flags = [column for column in INTEGRITY_BOOLEAN_COLUMNS if not normalise_bool(row[column])]
        if false_flags:
            failures.append(f"{row['model_id']}: failed integrity flags={', '.join(false_flags)}")
    if failures:
        raise ValueError("Metadata and integrity QC failed:\n" + "\n".join(failures))


def validate_prediction_groups(predictions: pd.DataFrame) -> dict[tuple[str, str], float]:
    keys = set(zip(predictions["input_view"], predictions["backbone"]))
    if keys != EXPECTED_KEYS:
        raise ValueError(
            "Unexpected input/model mappings. "
            f"Missing={sorted(EXPECTED_KEYS - keys)}, unexpected={sorted(keys - EXPECTED_KEYS)}"
        )

    pooled_aucs: dict[tuple[str, str], float] = {}
    failures: list[str] = []
    for input_view, backbone in EXPECTED_AUCS:
        group = predictions.loc[(predictions["input_view"] == input_view) & (predictions["backbone"] == backbone)].copy()
        labels = pd.to_numeric(group["binary_label"], errors="raise").astype(int)
        probabilities = pd.to_numeric(group["prob_patient"], errors="raise").astype(float)
        hard_predictions = pd.to_numeric(group["pred_class"], errors="raise").astype(int)
        if (
            len(group) != 500
            or group["sample_id"].nunique() != 500
            or int((labels == 0).sum()) != 115
            or int((labels == 1).sum()) != 385
            or group["model_id"].nunique() != 1
            or set(labels.unique()) != {0, 1}
            or set(hard_predictions.unique()) - {0, 1}
            or not np.isfinite(probabilities).all()
            or not probabilities.between(0.0, 1.0).all()
        ):
            failures.append(
                f"{input_view} {backbone}: N={len(group)}, unique sample_id={group['sample_id'].nunique()}, "
                f"Control={(labels == 0).sum()}, Patient={(labels == 1).sum()}"
            )
            continue
        calculated_auc = float(roc_auc_score(labels, probabilities))
        expected_auc = EXPECTED_AUCS[(input_view, backbone)]
        if not np.isclose(calculated_auc, expected_auc, atol=0.00005, rtol=0.0):
            failures.append(
                f"{input_view} {backbone}: calculated AUC={calculated_auc:.6f}, expected={expected_auc:.4f}, "
                f"difference={calculated_auc - expected_auc:+.6f}, N={len(group)}, "
                f"Control={(labels == 0).sum()}, Patient={(labels == 1).sum()}"
            )
            continue
        pooled_aucs[(input_view, backbone)] = calculated_auc

    if failures:
        print("## Figure 4 pooled OOF AUC QC failures")
        print("\n".join(failures))
        raise RuntimeError("Figure 4 was not generated because pooled OOF AUC QC failed.")
    return pooled_aucs


def select_display_models(pooled_aucs: dict[tuple[str, str], float]) -> dict[str, str]:
    """Return the models explicitly chosen for the Figure 4 display panels."""
    if set(DISPLAYED_BACKBONES) != set(VIEW_CONFIG):
        raise ValueError("DISPLAYED_BACKBONES must specify exactly the four Figure 4 input views")
    invalid = [
        f"{input_view}={backbone}"
        for input_view, backbone in DISPLAYED_BACKBONES.items()
        if backbone not in BACKBONE_CONFIG or (input_view, backbone) not in pooled_aucs
    ]
    if invalid:
        raise ValueError("Invalid Figure 4 display models: " + ", ".join(invalid))
    return dict(DISPLAYED_BACKBONES)


def calculate_confusion(labels: pd.Series, hard_predictions: pd.Series) -> np.ndarray:
    return confusion_matrix(labels.astype(int), hard_predictions.astype(int), labels=[0, 1])


def build_representative_data(
    predictions: pd.DataFrame,
    pooled_confusions: pd.DataFrame,
    pooled_aucs: dict[tuple[str, str], float],
    selected: dict[str, str],
) -> tuple[dict[str, np.ndarray], pd.DataFrame]:
    matrices: dict[str, np.ndarray] = {}
    records: list[dict[str, object]] = []
    failures: list[str] = []

    for input_view, backbone in selected.items():
        input_name = VIEW_CONFIG[input_view][1]
        model_name = BACKBONE_CONFIG[backbone]
        group = predictions.loc[(predictions["input_view"] == input_view) & (predictions["backbone"] == backbone)].copy()
        matrix = calculate_confusion(group["binary_label"], group["pred_class"])
        tn, fp, fn, tp = (int(matrix[0, 0]), int(matrix[0, 1]), int(matrix[1, 0]), int(matrix[1, 1]))
        model_id = group["model_id"].iloc[0]
        row_sums = matrix.sum(axis=1)
        total = int(matrix.sum())
        if row_sums.tolist() != [115, 385] or total != 500:
            failures.append(
                f"{input_name} {model_name}: TN={tn}, FP={fp}, FN={fn}, TP={tp}, row sums={row_sums.tolist()}, N={total}"
            )
            continue
        expected_matrix = EXPECTED_CONFUSIONS[(input_view, backbone)]
        if not np.array_equal(matrix, expected_matrix):
            failures.append(
                f"{input_name} {model_name}: TN={tn}, FP={fp}, FN={fn}, TP={tp}, row sums={row_sums.tolist()}, N={total}; "
                f"expected={expected_matrix.tolist()}"
            )
            continue
        source_row = pooled_confusions.loc[pooled_confusions["model_id"] == model_id]
        if len(source_row) != 1:
            failures.append(f"{input_name} {model_name}: expected one pooled_confusion_matrices.csv row, found {len(source_row)}")
            continue
        source = source_row.iloc[0]
        source_values = np.array([[int(source["tn"]), int(source["fp"])], [int(source["fn"]), int(source["tp"])]])
        if not np.array_equal(matrix, source_values) or int(source["n_samples"]) != 500:
            failures.append(
                f"{input_name} {model_name}: recomputed={matrix.tolist()}, source={source_values.tolist()}, "
                f"source N={source['n_samples']}"
            )
            continue

        sensitivity = tp / (tp + fn)
        specificity = tn / (tn + fp)
        balanced_accuracy = (sensitivity + specificity) / 2.0
        matrices[input_view] = matrix
        records.append(
            {
                "Input": input_name,
                "Model": model_name,
                "Pooled_OOF_AUC": pooled_aucs[(input_view, backbone)],
                "N": total,
                "Control": int(row_sums[0]),
                "Patient": int(row_sums[1]),
                "TN": tn,
                "FP": fp,
                "FN": fn,
                "TP": tp,
                "Sensitivity": sensitivity,
                "Specificity": specificity,
                "Balanced_Accuracy": balanced_accuracy,
            }
        )

    if failures:
        print("## Figure 4 confusion-matrix QC failures")
        print("\n".join(failures))
        raise RuntimeError("Figure 4 was not generated because confusion-matrix QC failed.")
    return matrices, pd.DataFrame(records)


def row_normalize(matrix: np.ndarray) -> np.ndarray:
    row_sums = matrix.sum(axis=1, keepdims=True)
    if (row_sums == 0).any():
        raise ValueError("A confusion-matrix row has zero samples and cannot be normalized")
    return matrix / row_sums


def plot_figure(matrices: dict[str, np.ndarray], selected: dict[str, str]) -> None:
    plt.rcParams.update(
        {
            "font.family": choose_font(),
            "font.size": 9,
            "axes.linewidth": 0.8,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    figure, axes = plt.subplots(2, 2, figsize=(7.1, 5.5), facecolor="white", layout="constrained")
    colormap = colormaps["Blues"]
    image = None

    for axis, input_view in zip(axes.flat, VIEW_CONFIG):
        letter, input_name = VIEW_CONFIG[input_view]
        model_name = BACKBONE_CONFIG[selected[input_view]]
        matrix = matrices[input_view]
        proportions = row_normalize(matrix)
        image = axis.imshow(proportions, cmap=colormap, vmin=0.0, vmax=1.0, interpolation="nearest")

        axis.set_xticks([0, 1], ["Normal", "Abnormal"], fontsize=9)
        axis.set_yticks([0, 1], ["Normal", "Abnormal"], fontsize=9)
        axis.set_xlabel("Predicted class", fontsize=10)
        axis.set_ylabel("True class", fontsize=10)
        axis.tick_params(length=0)
        axis.set_title(f"{input_name}\n{model_name}", fontsize=10.5, pad=11, linespacing=1.45)
        axis.text(-0.20, 1.18, letter, transform=axis.transAxes, fontsize=12, fontweight="bold", va="top", clip_on=False)

        for row_index in range(2):
            for column_index in range(2):
                proportion = proportions[row_index, column_index]
                text_color = "white" if proportion >= 0.58 else "#1A1A1A"
                axis.text(
                    column_index,
                    row_index - 0.10,
                    f"{matrix[row_index, column_index]:d}",
                    ha="center",
                    va="center",
                    fontsize=10,
                    fontweight="bold",
                    color=text_color,
                )
                axis.text(
                    column_index,
                    row_index + 0.13,
                    f"({proportion * 100:.1f}%)",
                    ha="center",
                    va="center",
                    fontsize=8.6,
                    color=text_color,
                )
        axis.set_box_aspect(1)

    colorbar = figure.colorbar(image, ax=axes.ravel().tolist(), location="right", shrink=0.88, pad=0.035)
    colorbar.set_label("Row-normalized proportion", fontsize=9.5)
    colorbar.set_ticks(np.linspace(0.0, 1.0, 6))
    colorbar.ax.tick_params(labelsize=8.5)
    figure.savefig(PNG_PATH, dpi=600, facecolor="white")
    figure.savefig(PDF_PATH, format="pdf", facecolor="white")
    plt.close(figure)


def print_qc_summary(checks: pd.DataFrame) -> None:
    print("## Figure 4 QC")
    for _, row in checks.iterrows():
        print(
            f"{row['Input']}: selected model = {row['Model']}; pooled OOF AUC = {row['Pooled_OOF_AUC']:.4f}; "
            f"TN={row['TN']} FP={row['FP']} FN={row['FN']} TP={row['TP']}; "
            f"Sensitivity={row['Sensitivity']:.4f}; Specificity={row['Specificity']:.4f}; PASS"
        )
    print("All four configured pooled OOF confusion matrices passed integrity checks.")
    print(f"PNG: {PNG_PATH}")
    print(f"PDF: {PDF_PATH}")
    print(f"CSV: {CHECK_PATH}")


def main() -> None:
    predictions, pooled_confusions, inventory, integrity = read_inputs()
    validate_metadata(inventory, integrity)
    pooled_aucs = validate_prediction_groups(predictions)
    selected = select_display_models(pooled_aucs)
    matrices, checks = build_representative_data(predictions, pooled_confusions, pooled_aucs, selected)
    checks.to_csv(CHECK_PATH, index=False, float_format="%.6f")
    plot_figure(matrices, selected)
    print_qc_summary(checks)


if __name__ == "__main__":
    main()
