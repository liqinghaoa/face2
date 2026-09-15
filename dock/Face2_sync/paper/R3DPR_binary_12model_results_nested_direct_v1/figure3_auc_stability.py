"""Generate Figure 3 outer-fold AUC stability curves for nested-direct experiments."""

from __future__ import annotations

import matplotlib as mpl
import numpy as np
import pandas as pd

from figure_common import BACKBONE_CONFIG, DATA_DIR, VIEW_CONFIG, configure_matplotlib, require_columns, validate_integrity
import matplotlib.pyplot as plt


mpl.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"], "svg.fonttype": "none", "pdf.fonttype": 42})


PNG_PATH = DATA_DIR / "Figure3_AUC_stability.png"
CHECK_PATH = DATA_DIR / "Figure3_fold_auc_check.csv"
SUMMARY_PATH = DATA_DIR / "Figure3_AUC_summary_check.csv"


def main() -> None:
    configure_matplotlib()
    foldwise = pd.read_csv(DATA_DIR / "foldwise_metrics_long.csv")
    inventory = pd.read_csv(DATA_DIR / "model_inventory.csv")
    integrity = pd.read_csv(DATA_DIR / "integrity_checks.csv")
    validate_integrity(inventory, integrity)
    require_columns(foldwise, {"model_id", "input_view", "backbone", "fold", "selected_epoch", "macro_auc"}, "foldwise_metrics_long.csv")
    if len(foldwise) != 60:
        raise ValueError(f"Expected 60 fold rows for 12 models, found {len(foldwise)}")
    records = []
    for view, (_, view_label) in VIEW_CONFIG.items():
        for backbone, (backbone_label, _) in BACKBONE_CONFIG.items():
            group = foldwise.loc[(foldwise["input_view"] == view) & (foldwise["backbone"] == backbone)].sort_values("fold")
            if len(group) != 5 or set(group["fold"].astype(int)) != set(range(5)):
                raise ValueError(f"{view}/{backbone} does not have exactly one row for outer folds 0-4")
            for row in group.itertuples(index=False):
                records.append({"Input": view_label, "Backbone": backbone_label, "Fold": int(row.fold) + 1, "Outer_Test_Macro_AUC": float(row.macro_auc), "Selected_Epoch": int(row.selected_epoch)})
    checks = pd.DataFrame(records)
    checks.to_csv(CHECK_PATH, index=False, float_format="%.6f")
    summary = (
        checks.groupby(["Input", "Backbone"])["Outer_Test_Macro_AUC"]
        .agg(Fold_Mean_Macro_AUC="mean", Fold_SD_Macro_AUC="std")
        .reset_index()
    )
    summary.to_csv(SUMMARY_PATH, index=False, float_format="%.6f")

    figure, axes = plt.subplots(2, 2, figsize=(7.2, 5.7), facecolor="white", layout="constrained")
    for axis, (view, (letter, view_label)) in zip(axes.flat, VIEW_CONFIG.items()):
        for backbone, (backbone_label, color) in BACKBONE_CONFIG.items():
            group = checks.loc[(checks["Input"] == view_label) & (checks["Backbone"] == backbone_label)].sort_values("Fold")
            axis.plot(group["Fold"], group["Outer_Test_Macro_AUC"], color=color, linewidth=1.5, marker="o", markersize=4.5, markeredgecolor="white", markeredgewidth=0.6, label=backbone_label, zorder=2)
        axis.set_xlim(0.8, 5.2)
        axis.set_ylim(0.0, 1.0)
        axis.set_xticks(range(1, 6))
        axis.set_yticks(np.arange(0.0, 1.01, 0.2))
        axis.set_xlabel("Outer fold")
        axis.set_ylabel("Macro-AUC")
        axis.set_title(view_label, fontsize=10, pad=8)
        axis.text(-0.15, 1.12, letter, transform=axis.transAxes, fontsize=12, fontweight="bold", va="top", clip_on=False)
        axis.grid(axis="y", color="#E5E5E5", linewidth=0.6, zorder=0)
        axis.legend(loc="lower right", fontsize=7.3, handlelength=1.7)
    figure.savefig(PNG_PATH, dpi=600, facecolor="white", bbox_inches="tight")
    plt.close(figure)
    print(f"FIGURE3_PNG={PNG_PATH}")
    print(f"FIGURE3_CHECK={CHECK_PATH}")


if __name__ == "__main__":
    main()
