"""Create per-fold two-panel training diagnostics for the R3DPR OneCycle variant."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd


OUTPUT_SUBDIR = "figures/onecycle_diagnostics"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment-dir", type=Path, required=True)
    parser.add_argument("--output-subdir", default=OUTPUT_SUBDIR)
    return parser.parse_args()


def configure_matplotlib() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
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


def load_fold_metrics(experiment_dir: Path) -> pd.DataFrame:
    path = experiment_dir / "fold_metrics.csv"
    if not path.is_file():
        raise FileNotFoundError(f"Missing fold_metrics.csv: {path}")
    frame = pd.read_csv(path)
    required = {"fold", "best_epoch", "macro_auc"}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"{path} lacks required columns: {missing}")
    return frame.sort_values("fold").reset_index(drop=True)


def _set_loss_limits(axis: plt.Axes, values: pd.Series) -> None:
    low = float(values.min())
    high = float(values.max())
    if low == high:
        margin = max(abs(low) * 0.1, 0.05)
    else:
        margin = (high - low) * 0.08
    axis.set_ylim(max(0.0, low - margin), high + margin)


def _set_auc_limits(axis: plt.Axes, values: pd.Series) -> None:
    low = max(0.0, float(values.min()) - 0.05)
    high = min(1.0, float(values.max()) + 0.05)
    if high - low < 0.2:
        center = (high + low) / 2.0
        low = max(0.0, center - 0.1)
        high = min(1.0, center + 0.1)
    axis.set_ylim(low, high)


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
    required = {"epoch", "train_loss", "val_loss", "train_macro_auc", "val_macro_auc"}
    missing = sorted(required.difference(history.columns))
    if missing:
        raise ValueError(f"{history_path} lacks required columns: {missing}")
    if history.empty:
        raise ValueError(f"{history_path} contains no epochs")

    epochs = history["epoch"].astype(int)
    all_loss = pd.concat([history["train_loss"], history["val_loss"]], ignore_index=True)
    all_auc = pd.concat([history["train_macro_auc"], history["val_macro_auc"]], ignore_index=True)
    final_auc = float(history["val_macro_auc"].iloc[-1])
    final_train_loss = float(history["train_loss"].iloc[-1])
    best_rows = history.loc[epochs == int(best_epoch)]
    if best_rows.empty:
        raise ValueError(f"best_epoch={best_epoch} is absent from {history_path}")
    train_loss_at_best = float(best_rows["train_loss"].iloc[0])

    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.0), constrained_layout=True)
    fig.suptitle(
        f"R3DPR ResNet18 binary OneCycle variant | Fold {fold} | best epoch {best_epoch} | val macro-AUC {best_auc:.3f}",
        fontsize=9,
    )
    colors = {"train": "#3C5488", "val": "#00A087", "best": "#D55E00"}

    axes[0].plot(epochs, history["train_loss"], color=colors["train"], marker="o", markersize=3.0, linewidth=1.4, label="Training loss")
    axes[0].plot(epochs, history["val_loss"], color=colors["val"], marker="o", markersize=3.0, linewidth=1.4, label="Validation loss")
    axes[0].axvline(best_epoch, color=colors["best"], linestyle="--", linewidth=1.0, label="Best epoch")
    axes[0].set_title("(a) Training / Validation Loss", loc="left", fontweight="bold")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Cross-entropy loss")
    axes[0].grid(axis="y", color="#E6E6E6", linewidth=0.6)
    axes[0].legend(loc="best")
    _set_loss_limits(axes[0], all_loss)

    axes[1].plot(epochs, history["train_macro_auc"], color=colors["train"], marker="o", markersize=3.0, linewidth=1.4, label="Training macro-AUC")
    axes[1].plot(epochs, history["val_macro_auc"], color=colors["val"], marker="o", markersize=3.0, linewidth=1.4, label="Validation macro-AUC")
    axes[1].axvline(best_epoch, color=colors["best"], linestyle="--", linewidth=1.0, label="Best epoch")
    axes[1].set_title("(b) Training / Validation AUC", loc="left", fontweight="bold")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Macro-AUC")
    axes[1].grid(axis="y", color="#E6E6E6", linewidth=0.6)
    axes[1].legend(loc="best")
    _set_auc_limits(axes[1], all_auc)

    for axis in axes:
        if len(epochs) <= 12:
            axis.set_xticks(list(epochs))
        else:
            axis.set_xticks(sorted(set([int(epochs.iloc[0]), int(best_epoch), int(epochs.iloc[-1])])))

    output_dir.mkdir(parents=True, exist_ok=True)
    stem = output_dir / f"fold_{fold}_training_diagnostics"
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
    }


def write_report(output_dir: Path, summary: pd.DataFrame) -> None:
    rows = "\n".join(
        "| {fold} | {epochs_run} | {best_epoch} | {best_val_macro_auc:.4f} | "
        "{final_val_macro_auc:.4f} | {auc_drop_after_best:.4f} | "
        "{train_loss_at_best:.4f} | {final_train_loss:.4f} |".format(**row)
        for row in summary.to_dict(orient="records")
    )
    report = f"""# R3DPR OneCycle training diagnostic figures

Each fold has one PNG figure with two panels: (a) training and validation loss;
(b) training and validation macro-AUC. The orange dashed line marks the
validation macro-AUC checkpoint selected for that fold. Figures are generated
from existing logs after training and do not retrain the model.

| Fold | Epochs run | Best epoch | Best val macro-AUC | Final val macro-AUC | AUC drop after best | Train loss at best | Final train loss |
|---:|---:|---:|---:|---:|---:|---:|
{rows}
"""
    (output_dir / "training_diagnostic_summary.md").write_text(report, encoding="utf-8")


def plot_experiment(experiment_dir: Path, output_subdir: str = OUTPUT_SUBDIR) -> Path:
    experiment_dir = experiment_dir.resolve()
    output_dir = experiment_dir / output_subdir
    configure_matplotlib()
    fold_metrics = load_fold_metrics(experiment_dir)
    records = []
    for row in fold_metrics.itertuples(index=False):
        records.append(plot_fold(experiment_dir, output_dir, int(row.fold), int(row.best_epoch), float(row.macro_auc)))
    summary = pd.DataFrame(records)
    summary.to_csv(output_dir / "training_diagnostic_summary.csv", index=False, encoding="utf-8-sig")
    write_report(output_dir, summary)
    return output_dir


def main() -> Path:
    args = parse_args()
    output_dir = plot_experiment(args.experiment_dir, args.output_subdir)
    print(f"TRAINING_DIAGNOSTIC_DIR={output_dir}")
    return output_dir


if __name__ == "__main__":
    main()
