"""Plot E0B per-fold overfitting diagnostic curves from existing histories."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd


METRIC_AUC = "macro_auc"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment-dir", type=Path, required=True)
    parser.add_argument("--output-subdir", default="figures/overfit_diagnostics")
    return parser.parse_args()


def configure_matplotlib() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "font.size": 8,
            "axes.spines.right": False,
            "axes.spines.top": False,
            "axes.linewidth": 0.8,
            "legend.frameon": False,
            "figure.dpi": 150,
        }
    )


def save_figure(fig: plt.Figure, stem: Path) -> None:
    fig.savefig(stem.with_suffix(".png"), dpi=600, bbox_inches="tight")
    fig.savefig(stem.with_suffix(".svg"), bbox_inches="tight")
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")


def load_fold_metrics(experiment_dir: Path) -> pd.DataFrame:
    path = experiment_dir / "fold_metrics.csv"
    if not path.is_file():
        raise FileNotFoundError(f"Missing fold_metrics.csv: {path}")
    frame = pd.read_csv(path)
    required = {"fold", "best_epoch", METRIC_AUC}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"fold_metrics.csv lacks required columns: {sorted(missing)}")
    return frame


def plot_fold(
    experiment_dir: Path,
    output_dir: Path,
    fold: int,
    best_epoch: int,
    best_auc: float,
) -> dict[str, float | int | str]:
    history_path = experiment_dir / f"fold_{fold}" / "training_history.csv"
    if not history_path.is_file():
        raise FileNotFoundError(f"Missing training history: {history_path}")
    history = pd.read_csv(history_path)
    required = {"epoch", "train_loss", METRIC_AUC}
    missing = required.difference(history.columns)
    if missing:
        raise ValueError(f"{history_path} lacks required columns: {sorted(missing)}")

    epochs = history["epoch"].astype(int)
    final_auc = float(history[METRIC_AUC].iloc[-1])
    final_train_loss = float(history["train_loss"].iloc[-1])
    best_row = history.loc[history["epoch"].astype(int) == int(best_epoch)]
    train_loss_at_best = float(best_row["train_loss"].iloc[0]) if not best_row.empty else float("nan")

    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.7), constrained_layout=True)
    fig.suptitle(
        f"Fold {fold} overfitting diagnostic: best epoch {best_epoch}, validation macro-AUC {best_auc:.3f}",
        y=1.04,
        fontsize=9,
    )

    axes[0].plot(
        epochs,
        history["train_loss"],
        color="#3C5488",
        marker="o",
        markersize=3.0,
        linewidth=1.4,
        label="Training loss",
    )
    axes[0].axvline(best_epoch, color="#D55E00", linestyle="--", linewidth=1.0, label="Best epoch")
    axes[0].set_title("(a) Training / validation loss", loc="left", fontweight="bold")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Cross-entropy loss")
    axes[0].grid(axis="y", color="#E6E6E6", linewidth=0.6)
    axes[0].text(
        0.02,
        0.96,
        "Validation loss not logged",
        transform=axes[0].transAxes,
        va="top",
        ha="left",
        fontsize=7,
        color="#666666",
    )
    axes[0].legend(loc="upper right")

    axes[1].plot(
        epochs,
        history[METRIC_AUC],
        color="#009E73",
        marker="o",
        markersize=3.0,
        linewidth=1.4,
        label="Validation macro-AUC",
    )
    axes[1].axvline(best_epoch, color="#D55E00", linestyle="--", linewidth=1.0, label="Best epoch")
    axes[1].set_title("(b) Training / validation AUC", loc="left", fontweight="bold")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Macro-AUC")
    axes[1].set_ylim(0.70, 0.92)
    axes[1].grid(axis="y", color="#E6E6E6", linewidth=0.6)
    axes[1].text(
        0.02,
        0.96,
        "Training AUC not logged",
        transform=axes[1].transAxes,
        va="top",
        ha="left",
        fontsize=7,
        color="#666666",
    )
    axes[1].legend(loc="lower right")

    for ax in axes:
        ax.set_xticks(list(epochs))
        if len(epochs) > 12:
            ax.set_xticks([int(epochs.iloc[0]), int(best_epoch), int(epochs.iloc[-1])])

    stem = output_dir / f"fold_{fold}_overfit_diagnostic"
    save_figure(fig, stem)
    plt.close(fig)

    return {
        "fold": fold,
        "epochs_run": int(epochs.iloc[-1]),
        "best_epoch": int(best_epoch),
        "best_val_macro_auc": float(best_auc),
        "final_val_macro_auc": final_auc,
        "auc_drop_after_best": float(best_auc - final_auc),
        "train_loss_at_best": train_loss_at_best,
        "final_train_loss": final_train_loss,
        "png": str(stem.with_suffix(".png")),
        "svg": str(stem.with_suffix(".svg")),
        "pdf": str(stem.with_suffix(".pdf")),
    }


def write_report(output_dir: Path, summary: pd.DataFrame) -> None:
    rows = "\n".join(
        "| {fold} | {epochs_run} | {best_epoch} | {best_val_macro_auc:.4f} | "
        "{final_val_macro_auc:.4f} | {auc_drop_after_best:.4f} | "
        "{train_loss_at_best:.4f} | {final_train_loss:.4f} |".format(**row)
        for row in summary.to_dict(orient="records")
    )
    report = f"""# E0B overfitting diagnostic figures

These figures use the existing per-epoch logs without retraining. The current
training history records training loss and validation metrics only; validation
loss and training AUC were not logged.

| Fold | Epochs run | Best epoch | Best val macro-AUC | Final val macro-AUC | AUC drop after best | Train loss at best | Final train loss |
|---:|---:|---:|---:|---:|---:|---:|---:|
{rows}
"""
    (output_dir / "overfit_diagnostic_summary.md").write_text(report, encoding="utf-8")


def main() -> Path:
    args = parse_args()
    experiment_dir = args.experiment_dir.resolve()
    output_dir = experiment_dir / args.output_subdir
    output_dir.mkdir(parents=True, exist_ok=True)
    configure_matplotlib()

    fold_metrics = load_fold_metrics(experiment_dir)
    records = []
    for row in fold_metrics.sort_values("fold").itertuples(index=False):
        records.append(
            plot_fold(
                experiment_dir=experiment_dir,
                output_dir=output_dir,
                fold=int(row.fold),
                best_epoch=int(row.best_epoch),
                best_auc=float(getattr(row, METRIC_AUC)),
            )
        )

    summary = pd.DataFrame(records)
    summary.to_csv(output_dir / "overfit_diagnostic_summary.csv", index=False, encoding="utf-8-sig")
    write_report(output_dir, summary)
    print(f"OVERFIT_DIAGNOSTIC_DIR={output_dir}")
    return output_dir


if __name__ == "__main__":
    main()
