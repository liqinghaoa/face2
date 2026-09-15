"""Generate Figure 3 as five-fold AUC trajectories for all 12 models.

Each panel shows the observed AUC in Fold 1--5. Lines connect folds only to
make fold-to-fold stability easier to inspect; they do not imply a temporal or
continuous measurement between independent cross-validation folds.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np
import pandas as pd


# Input and output paths are deliberately centralized for reproducibility.
DATA_DIR = Path(__file__).resolve().parent
FOLDWISE_LONG_PATH = DATA_DIR / "foldwise_metrics_long.csv"
MODEL_INVENTORY_PATH = DATA_DIR / "model_inventory.csv"
INTEGRITY_CHECKS_PATH = DATA_DIR / "integrity_checks.csv"
OUTPUT_DIR = DATA_DIR
PNG_PATH = OUTPUT_DIR / "Figure3_AUC_stability.png"
PDF_PATH = OUTPUT_DIR / "Figure3_AUC_stability.pdf"
SVG_PATH = OUTPUT_DIR / "Figure3_AUC_stability.svg"
TIFF_PATH = OUTPUT_DIR / "Figure3_AUC_stability.tiff"
CHECK_PATH = OUTPUT_DIR / "Figure3_fold_auc_check.csv"

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
EXPECTED_KEYS = {(input_view, backbone) for input_view in VIEW_CONFIG for backbone in BACKBONE_CONFIG}
REQUIRED_LONG_COLUMNS = {"model_id", "input_view", "backbone", "fold", "best_epoch", "macro_auc"}
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


def require_columns(frame: pd.DataFrame, expected: set[str], frame_name: str) -> None:
    missing = expected - set(frame.columns)
    if missing:
        raise ValueError(f"{frame_name} is missing required columns: {sorted(missing)}")


def read_inputs() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    fold_long = pd.read_csv(FOLDWISE_LONG_PATH)
    inventory = pd.read_csv(MODEL_INVENTORY_PATH)
    integrity = pd.read_csv(INTEGRITY_CHECKS_PATH)
    require_columns(fold_long, REQUIRED_LONG_COLUMNS, "foldwise_metrics_long.csv")
    return fold_long, inventory, integrity


def validate_metadata(inventory: pd.DataFrame, integrity: pd.DataFrame) -> None:
    required_integrity = {"model_id", "oof_rows", "fold_metrics_rows", *INTEGRITY_BOOLEAN_COLUMNS}
    require_columns(inventory, {"model_id", "n_oof_samples", "n_folds"}, "model_inventory.csv")
    require_columns(integrity, required_integrity, "integrity_checks.csv")
    inventory_ids = set(inventory["model_id"])
    integrity_ids = set(integrity["model_id"])
    if inventory_ids != integrity_ids or len(inventory_ids) != 12:
        raise ValueError("model_inventory.csv and integrity_checks.csv do not contain the same 12 models")

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
        raise ValueError("Metadata or integrity checks failed:\n" + "\n".join(failures))


def build_plot_data(fold_long: pd.DataFrame) -> pd.DataFrame:
    keys = set(zip(fold_long["input_view"], fold_long["backbone"]))
    if keys != EXPECTED_KEYS:
        raise ValueError(
            "foldwise_metrics_long.csv has unexpected input/model mappings. "
            f"Missing={sorted(EXPECTED_KEYS - keys)}, unexpected={sorted(keys - EXPECTED_KEYS)}"
        )

    records: list[dict[str, object]] = []
    failures: list[str] = []
    for input_view, (_, input_name) in VIEW_CONFIG.items():
        for backbone, (model_name, _) in BACKBONE_CONFIG.items():
            group = fold_long.loc[
                (fold_long["input_view"] == input_view) & (fold_long["backbone"] == backbone)
            ].copy()
            folds = pd.to_numeric(group["fold"], errors="raise").astype(int)
            aucs = pd.to_numeric(group["macro_auc"], errors="raise").astype(float)
            if len(group) != 5 or set(folds) != {0, 1, 2, 3, 4} or group["model_id"].nunique() != 1:
                failures.append(
                    f"{input_name} {model_name}: expected 5 rows with folds 0-4; "
                    f"found rows={len(group)}, folds={sorted(folds.tolist())}"
                )
                continue
            if not np.isfinite(aucs).all() or ((aucs < 0.0) | (aucs > 1.0)).any():
                failures.append(f"{input_name} {model_name}: invalid fold AUC values")
                continue
            for _, row in group.sort_values("fold").iterrows():
                records.append(
                    {
                        "Input": input_name,
                        "Model": model_name,
                        "Fold": int(row["fold"]) + 1,
                        "Fold_AUC": float(row["macro_auc"]),
                    }
                )
    if failures:
        print("## Figure 3 QC failures")
        print("\n".join(failures))
        raise RuntimeError("Figure 3 was not generated because fold-AUC QC failed.")
    return pd.DataFrame(records)


def y_axis_limits(_fold_checks: pd.DataFrame) -> tuple[float, float, np.ndarray]:
    """Use the complete AUC scale rather than a data-derived zoomed range."""
    return 0.0, 1.0, np.arange(0.0, 1.01, 0.2)


def plot_figure(fold_checks: pd.DataFrame) -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": [choose_font(), "Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
            "font.size": 9,
            "axes.linewidth": 0.8,
            "axes.spines.right": False,
            "axes.spines.top": False,
            "legend.frameon": False,
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
        }
    )
    lower, upper, ticks = y_axis_limits(fold_checks)
    figure, axes = plt.subplots(2, 2, figsize=(7.1, 5.5), facecolor="white")
    figure.subplots_adjust(left=0.10, right=0.985, bottom=0.15, top=0.92, wspace=0.28, hspace=0.42)

    for axis, input_view in zip(axes.flat, VIEW_CONFIG):
        letter, input_name = VIEW_CONFIG[input_view]
        for backbone, (model_name, color) in BACKBONE_CONFIG.items():
            data = fold_checks.loc[
                (fold_checks["Input"] == input_name) & (fold_checks["Model"] == model_name)
            ].sort_values("Fold")
            axis.plot(
                data["Fold"],
                data["Fold_AUC"],
                color=color,
                linewidth=1.5,
                marker="o",
                markersize=4.5,
                markeredgecolor="white",
                markeredgewidth=0.65,
                label=model_name,
                zorder=3,
            )
        axis.set_xlim(0.8, 5.2)
        axis.set_ylim(lower, upper)
        axis.set_xticks(range(1, 6))
        axis.set_yticks(ticks)
        axis.set_xlabel("Fold", fontsize=10)
        axis.set_ylabel("AUC", fontsize=10)
        axis.tick_params(labelsize=9)
        axis.grid(axis="y", color="#D9D9D9", linewidth=0.6, zorder=0)
        axis.set_title(input_name, fontsize=11, pad=9)
        axis.text(-0.14, 1.13, letter, transform=axis.transAxes, fontsize=12, fontweight="bold", va="top", clip_on=False)

    handles, labels = axes.flat[0].get_legend_handles_labels()
    figure.legend(
        handles,
        labels,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.012),
        ncol=3,
        fontsize=8.5,
        handlelength=2.2,
        columnspacing=2.1,
    )
    figure.savefig(PNG_PATH, dpi=600, facecolor="white")
    figure.savefig(PDF_PATH, format="pdf", facecolor="white")
    figure.savefig(SVG_PATH, format="svg", facecolor="white")
    figure.savefig(TIFF_PATH, dpi=600, format="tiff", facecolor="white")
    plt.close(figure)


def print_qc_summary(fold_checks: pd.DataFrame) -> None:
    print("## Figure 3 QC")
    for input_view, (_, input_name) in VIEW_CONFIG.items():
        for backbone, (model_name, _) in BACKBONE_CONFIG.items():
            data = fold_checks.loc[
                (fold_checks["Input"] == input_name) & (fold_checks["Model"] == model_name)
            ].sort_values("Fold")
            values = ", ".join(f"Fold {int(row.Fold)}={row.Fold_AUC:.4f}" for row in data.itertuples())
            print(f"{input_name} {model_name}: {values}; PASS")
    print("All 12 models passed five-fold AUC checks.")
    print(f"PNG: {PNG_PATH}")
    print(f"PDF: {PDF_PATH}")
    print(f"SVG: {SVG_PATH}")
    print(f"TIFF: {TIFF_PATH}")
    print(f"Fold-AUC check: {CHECK_PATH}")


def main() -> None:
    fold_long, inventory, integrity = read_inputs()
    validate_metadata(inventory, integrity)
    fold_checks = build_plot_data(fold_long)
    fold_checks.to_csv(CHECK_PATH, index=False, float_format="%.6f")
    plot_figure(fold_checks)
    print_qc_summary(fold_checks)


if __name__ == "__main__":
    main()
