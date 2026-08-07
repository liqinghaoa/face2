from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from .r2_exif_decomposition import TARGET_METRICS


def write_r2_figures(r2_root: Path) -> None:
    figures = r2_root / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    plt.switch_backend("Agg")
    worst = pd.read_csv(r2_root / "repair" / "erosion_worst_group_metrics_repaired.csv")
    perf = pd.read_csv(r2_root / "exif" / "exif_decomposition_model_performance.csv")
    pred = pd.read_csv(r2_root / "exif" / "exif_crossfit_predictions_500.csv")
    label_assoc = pd.read_csv(r2_root / "exif" / "exif_component_label_association.csv")
    error = pd.read_csv(r2_root / "exif" / "exif_component_error_risk_regression.csv")
    attenuation = pd.read_csv(r2_root / "exif" / "exif_decomposition_attenuation_summary.csv")

    _bar(worst[(worst.scope == "stable_only") & (worst.erosion_px == 2)], "metric_name", "WorstGroupAUC", figures / "repaired_worst_group_auc_summary.png", "Stable-only WorstGroupAUC")
    _bar(worst[(worst.scope == "stable_only") & (worst.erosion_px == 2)], "metric_name", "DeltaAUC", figures / "repaired_delta_auc_with_ci.png", "Stable-only DeltaAUC")
    _bar(worst[worst.metric_name == "skin_y_median"], "erosion_px", "DeltaAUC", figures / "erosion_repaired_sensitivity.png", "DeltaAUC by erosion for skin_y_median")

    plt.figure(figsize=(8, 6))
    for target in TARGET_METRICS:
        plt.scatter(pred[f"observed_{target}"], pred[f"predicted_{target}"], s=10, alpha=0.5, label=target)
    plt.xlabel("observed metric")
    plt.ylabel("EXIF-predicted component")
    plt.legend(fontsize=7)
    plt.tight_layout()
    plt.savefig(figures / "exif_observed_vs_predicted.png", dpi=180)
    plt.close()

    _bar(perf, "target_metric", "r2", figures / "exif_model_oof_r2_summary.png", "EXIF Ridge OOF R2")
    _component_box(pred, "predicted", figures / "exif_predicted_component_by_label.png")
    _component_box(pred, "residual", figures / "exif_residual_component_by_label.png")
    _bar(label_assoc, "component", "label_auc", figures / "exif_component_label_auc.png", "Component label AUC")
    err_terms = error[error.term == "z_component"].copy()
    _bar(err_terms, "target_metric", "or", figures / "exif_component_rgb_error_or.png", "Component RGB error OR")
    _bar(attenuation, "target_metric", "smd_attenuation", figures / "exif_attenuation_summary.png", "SMD attenuation, residual vs observed")


def _bar(frame: pd.DataFrame, x_col: str, y_col: str, path: Path, title: str) -> None:
    plt.figure(figsize=(9, 4.8))
    if len(frame):
        x = frame[x_col].astype(str)
        y = pd.to_numeric(frame[y_col], errors="coerce")
        plt.bar(range(len(frame)), y)
        plt.xticks(range(len(frame)), x, rotation=35, ha="right")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()


def _component_box(pred: pd.DataFrame, component_type: str, path: Path) -> None:
    plt.figure(figsize=(9, 5))
    target = TARGET_METRICS[0]
    col = f"{component_type}_{target}"
    data = [pred.loc[pred.binary_label == lab, col] for lab in [0, 1]]
    plt.boxplot(data, labels=["Control", "Patient"])
    plt.title(f"EXIF-{component_type} component by label: {target}")
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()
