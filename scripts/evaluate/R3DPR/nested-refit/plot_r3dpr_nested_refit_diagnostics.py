"""Plot internal selection diagnostics for R3DPR nested-refit experiments."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd


OUTPUT_SUBDIR = "figures/internal_selection_diagnostics"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment-dir", type=Path, required=True)
    parser.add_argument("--output-subdir", default=OUTPUT_SUBDIR)
    return parser.parse_args()


def configure_matplotlib() -> None:
    mpl.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"], "font.size": 8, "legend.frameon": False, "figure.dpi": 150})


def plot_fold(experiment_dir: Path, output_dir: Path, fold: int, selected_epoch: int, best_auc: float) -> None:
    history = pd.read_csv(experiment_dir / f"fold_{fold}" / "inner_selection_history.csv")
    required = {"epoch", "train_loss", "inner_val_loss", "train_macro_auc", "inner_val_macro_auc"}
    missing = sorted(required.difference(history.columns))
    if missing:
        raise ValueError(f"Fold {fold} inner history lacks columns: {missing}")
    epochs = history["epoch"].astype(int)
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.0), constrained_layout=True)
    fig.suptitle(f"Nested-refit | Outer fold {fold} | selected epoch {selected_epoch} | internal val Macro-AUC {best_auc:.3f}", fontsize=9)
    colors = {"train": "#3C5488", "validation": "#00A087", "selected": "#D55E00"}
    axes[0].plot(epochs, history["train_loss"], color=colors["train"], marker="o", markersize=3, linewidth=1.4, label="Internal training loss")
    axes[0].plot(epochs, history["inner_val_loss"], color=colors["validation"], marker="o", markersize=3, linewidth=1.4, label="Internal validation loss")
    axes[0].axvline(selected_epoch, color=colors["selected"], linestyle="--", linewidth=1.0, label="Selected epoch")
    axes[0].set_title("(a) Internal Training / Validation Loss", loc="left", fontweight="bold")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Cross-entropy loss")
    axes[1].plot(epochs, history["train_macro_auc"], color=colors["train"], marker="o", markersize=3, linewidth=1.4, label="Internal training Macro-AUC")
    axes[1].plot(epochs, history["inner_val_macro_auc"], color=colors["validation"], marker="o", markersize=3, linewidth=1.4, label="Internal validation Macro-AUC")
    axes[1].axvline(selected_epoch, color=colors["selected"], linestyle="--", linewidth=1.0, label="Selected epoch")
    axes[1].set_title("(b) Internal Training / Validation AUC", loc="left", fontweight="bold")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Macro-AUC")
    for axis in axes:
        axis.grid(axis="y", color="#E6E6E6", linewidth=0.6)
        axis.legend(loc="best")
    output_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_dir / f"fold_{fold}_internal_selection_diagnostics.png", dpi=600, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    experiment_dir = args.experiment_dir.resolve()
    metrics = pd.read_csv(experiment_dir / "fold_metrics.csv").sort_values("fold")
    required = {"fold", "selected_epoch", "inner_best_macro_auc"}
    missing = sorted(required.difference(metrics.columns))
    if missing:
        raise ValueError(f"fold_metrics.csv lacks columns: {missing}")
    configure_matplotlib()
    output_dir = experiment_dir / args.output_subdir
    for row in metrics.itertuples(index=False):
        plot_fold(experiment_dir, output_dir, int(row.fold), int(row.selected_epoch), float(row.inner_best_macro_auc))
    print(f"NESTED_REFIT_DIAGNOSTIC_DIR={output_dir}")


if __name__ == "__main__":
    main()
