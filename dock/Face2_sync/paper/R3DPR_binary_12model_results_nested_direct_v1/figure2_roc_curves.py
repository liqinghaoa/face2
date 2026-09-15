"""Generate Figure 2 ROC curves from nested-direct pooled OOF predictions."""

from __future__ import annotations

import matplotlib as mpl
import numpy as np
import pandas as pd
from sklearn.metrics import auc, roc_curve

from figure_common import BACKBONE_CONFIG, DATA_DIR, VIEW_CONFIG, configure_matplotlib, validate_integrity, validate_prediction_groups
import matplotlib.pyplot as plt


mpl.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"], "svg.fonttype": "none", "pdf.fonttype": 42})


PNG_PATH = DATA_DIR / "Figure2_ROC_curves.png"
CHECK_PATH = DATA_DIR / "Figure2_AUC_check.csv"


def main() -> None:
    configure_matplotlib()
    predictions = pd.read_csv(DATA_DIR / "pooled_oof_predictions_long.csv", dtype={"sample_id": "string"})
    inventory = pd.read_csv(DATA_DIR / "model_inventory.csv")
    integrity = pd.read_csv(DATA_DIR / "integrity_checks.csv")
    validate_integrity(inventory, integrity)
    validate_prediction_groups(predictions)
    records = []
    curves = {}
    for view, (_, view_label) in VIEW_CONFIG.items():
        for backbone, (backbone_label, _) in BACKBONE_CONFIG.items():
            group = predictions.loc[(predictions["input_view"] == view) & (predictions["backbone"] == backbone)]
            labels = group["binary_label"].to_numpy(int)
            probabilities = group["prob_patient"].to_numpy(float)
            fpr, tpr, _ = roc_curve(labels, probabilities, pos_label=1)
            value = float(auc(fpr, tpr))
            curves[(view, backbone)] = (fpr, tpr, value)
            records.append({"Input": view_label, "Backbone": backbone_label, "Pooled_OOF_Macro_AUC": value, "N": len(group)})
    checks = pd.DataFrame(records)
    checks.to_csv(CHECK_PATH, index=False, float_format="%.6f")

    figure, axes = plt.subplots(2, 2, figsize=(7.2, 6.0), facecolor="white")
    figure.subplots_adjust(left=0.09, right=0.99, bottom=0.08, top=0.95, wspace=0.03, hspace=0.48)
    for axis, (view, (letter, view_label)) in zip(axes.flat, VIEW_CONFIG.items()):
        axis.plot([0, 1], [0, 1], color="#9E9E9E", linewidth=0.8, linestyle="--", zorder=1)
        for backbone, (backbone_label, color) in BACKBONE_CONFIG.items():
            fpr, tpr, value = curves[(view, backbone)]
            axis.plot(fpr, tpr, color=color, linewidth=1.6, label=f"{backbone_label} (AUC={value:.3f})", zorder=2)
        axis.set_xlim(-0.01, 1.01)
        axis.set_ylim(-0.01, 1.01)
        axis.set_aspect("equal", adjustable="box")
        axis.set_xlabel("1 - Specificity")
        axis.set_ylabel("Sensitivity")
        axis.set_title(view_label, fontsize=10, pad=8)
        axis.text(-0.15, 1.12, letter, transform=axis.transAxes, fontsize=12, fontweight="bold", va="top", clip_on=False)
        axis.grid(color="#E5E5E5", linewidth=0.6, zorder=0)
        axis.legend(loc="lower right", fontsize=7.1, handlelength=1.6)
    figure.savefig(PNG_PATH, dpi=600, facecolor="white", bbox_inches="tight")
    plt.close(figure)
    print(f"FIGURE2_PNG={PNG_PATH}")
    print(f"FIGURE2_CHECK={CHECK_PATH}")


if __name__ == "__main__":
    main()
