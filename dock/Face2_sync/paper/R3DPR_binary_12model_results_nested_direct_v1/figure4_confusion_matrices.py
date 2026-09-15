"""Generate Figure 4 pooled OOF confusion matrices for the primary model of each input view."""

from __future__ import annotations

import matplotlib as mpl
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix

from figure_common import BACKBONE_CONFIG, DATA_DIR, VIEW_CONFIG, configure_matplotlib, require_columns, validate_integrity, validate_prediction_groups
import matplotlib.pyplot as plt


mpl.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"], "svg.fonttype": "none", "pdf.fonttype": 42})


PNG_PATH = DATA_DIR / "Figure4_confusion_matrices.png"
CHECK_PATH = DATA_DIR / "Figure4_confusion_matrix_check.csv"
DISPLAYED_BACKBONE_OVERRIDES = {"whole_face_sh93": "resnet18"}


def main() -> None:
    configure_matplotlib()
    predictions = pd.read_csv(DATA_DIR / "pooled_oof_predictions_long.csv", dtype={"sample_id": "string"})
    pooled_metrics = pd.read_csv(DATA_DIR / "pooled_oof_metrics.csv")
    pooled_confusions = pd.read_csv(DATA_DIR / "pooled_confusion_matrices.csv")
    inventory = pd.read_csv(DATA_DIR / "model_inventory.csv")
    integrity = pd.read_csv(DATA_DIR / "integrity_checks.csv")
    validate_integrity(inventory, integrity)
    validate_prediction_groups(predictions)
    require_columns(pooled_metrics, {"model_id", "input_view", "backbone", "macro_auc"}, "pooled_oof_metrics.csv")
    require_columns(pooled_confusions, {"model_id", "tn", "fp", "fn", "tp", "n_samples"}, "pooled_confusion_matrices.csv")

    selected_rows = []
    matrices = {}
    for view, (_, view_label) in VIEW_CONFIG.items():
        candidates = pooled_metrics.loc[pooled_metrics["input_view"] == view].sort_values(["macro_auc", "backbone"], ascending=[False, True])
        if len(candidates) != 3:
            raise ValueError(f"Expected three backbones for {view}, found {len(candidates)}")
        override = DISPLAYED_BACKBONE_OVERRIDES.get(view)
        selected = candidates.loc[candidates["backbone"] == override].iloc[0] if override else candidates.iloc[0]
        model_id = str(selected["model_id"])
        group = predictions.loc[predictions["model_id"] == model_id]
        matrix = confusion_matrix(group["binary_label"].astype(int), group["pred_class"].astype(int), labels=[0, 1])
        source = pooled_confusions.loc[pooled_confusions["model_id"] == model_id]
        if len(source) != 1:
            raise ValueError(f"Expected one pooled confusion-matrix row for {model_id}, found {len(source)}")
        source_matrix = np.array([[int(source.iloc[0].tn), int(source.iloc[0].fp)], [int(source.iloc[0].fn), int(source.iloc[0].tp)]])
        if not np.array_equal(matrix, source_matrix) or int(matrix.sum()) != 500:
            raise ValueError(f"Pooled confusion-matrix validation failed for {model_id}")
        matrices[view] = matrix
        selected_rows.append({"Input": view_label, "Model_ID": model_id, "Backbone": BACKBONE_CONFIG[str(selected.backbone)][0], "Pooled_OOF_Macro_AUC": float(selected.macro_auc), "TN": int(matrix[0, 0]), "FP": int(matrix[0, 1]), "FN": int(matrix[1, 0]), "TP": int(matrix[1, 1]), "N": int(matrix.sum())})
    checks = pd.DataFrame(selected_rows)
    checks.to_csv(CHECK_PATH, index=False, float_format="%.6f")

    figure, axes = plt.subplots(2, 2, figsize=(7.2, 5.7), facecolor="white", layout="constrained")
    image = None
    for axis, (view, (letter, view_label)) in zip(axes.flat, VIEW_CONFIG.items()):
        matrix = matrices[view]
        proportions = matrix / matrix.sum(axis=1, keepdims=True)
        row = checks.loc[checks["Input"] == view_label].iloc[0]
        image = axis.imshow(proportions, cmap="Blues", vmin=0.0, vmax=1.0, interpolation="nearest")
        axis.set_xticks([0, 1], ["Normal", "Abnormal"])
        axis.set_yticks([0, 1], ["Normal", "Abnormal"])
        axis.set_xlabel("Predicted class")
        axis.set_ylabel("True class")
        axis.tick_params(length=0)
        axis.set_title(f"{view_label}: {row.Backbone}\npooled Macro-AUC={row.Pooled_OOF_Macro_AUC:.3f}", fontsize=9.3, pad=8)
        axis.text(-0.17, 1.13, letter, transform=axis.transAxes, fontsize=12, fontweight="bold", va="top", clip_on=False)
        for i in range(2):
            for j in range(2):
                value = proportions[i, j]
                color = "white" if value >= 0.58 else "#1A1A1A"
                axis.text(j, i - 0.09, f"{matrix[i, j]:d}", ha="center", va="center", fontsize=10, fontweight="bold", color=color)
                axis.text(j, i + 0.14, f"({value * 100:.1f}%)", ha="center", va="center", fontsize=8, color=color)
        axis.set_box_aspect(1)
    colorbar = figure.colorbar(image, ax=axes.ravel().tolist(), shrink=0.88, pad=0.025)
    colorbar.set_label("Row-normalized proportion")
    colorbar.ax.tick_params(labelsize=8)
    figure.savefig(PNG_PATH, dpi=600, facecolor="white", bbox_inches="tight")
    plt.close(figure)
    print(f"FIGURE4_PNG={PNG_PATH}")
    print(f"FIGURE4_CHECK={CHECK_PATH}")


if __name__ == "__main__":
    main()
