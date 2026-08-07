from __future__ import annotations

from pathlib import Path

import pandas as pd


def make_plots(master: pd.DataFrame, metadata_oof: pd.DataFrame | None, stratified: pd.DataFrame | None, figures_dir: Path) -> None:
    figures_dir.mkdir(parents=True, exist_ok=True)
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    def savefig(name: str) -> None:
        plt.tight_layout()
        plt.savefig(figures_dir / name, dpi=160)
        plt.close()

    pd.crosstab(master["camera_model"], master["binary_label"]).plot(kind="bar", stacked=True, figsize=(9, 5))
    plt.xlabel("camera_model")
    plt.ylabel("count")
    savefig("camera_model_by_label.png")

    for var, filename in [
        ("brightness_device_z", "brightness_by_label_and_device.png"),
        ("log2_iso_device_z", "iso_by_label_and_device.png"),
        ("log2_exposure_device_z", "exposure_by_label_and_device.png"),
    ]:
        if var in master.columns:
            master.boxplot(column=var, by=["camera_model", "binary_label"], rot=90, figsize=(11, 5))
            plt.suptitle("")
            plt.title(var)
            savefig(filename)

    pd.crosstab(master.get("capture_year_month", pd.Series(["missing"] * len(master))), master["binary_label"]).plot(kind="bar", stacked=True, figsize=(10, 5))
    plt.xlabel("capture_year_month")
    plt.ylabel("count")
    savefig("capture_batch_by_label.png")

    if stratified is not None and not stratified.empty:
        for strat, filename in [("camera_model", "rgb_performance_by_device.png"), ("brightness", "rgb_performance_by_brightness.png")]:
            sub = stratified[stratified["stratification"] == strat].copy()
            if not sub.empty:
                sub.plot(kind="bar", x="stratum", y="roc_auc", legend=False, figsize=(9, 4))
                plt.ylabel("ROC-AUC")
                savefig(filename)

    for var in ["brightness_device_z", "log2_iso_device_z", "log2_exposure_device_z"]:
        if var in master.columns:
            plt.figure(figsize=(6, 4))
            plt.scatter(master[var], master["rgb_oof_residual"], s=16, alpha=0.7)
            plt.xlabel(var)
            plt.ylabel("rgb_oof_residual")
            savefig("rgb_residual_vs_metadata.png")
            break

    if metadata_oof is not None and not metadata_oof.empty:
        plt.figure(figsize=(6, 5))
        for model_name, frame in metadata_oof.groupby("model_name"):
            from sklearn.metrics import roc_curve

            if frame["binary_label"].nunique() == 2:
                fpr, tpr, _ = roc_curve(frame["binary_label"].astype(int), frame["probability_patient"].astype(float))
                plt.plot(fpr, tpr, label=model_name)
        plt.plot([0, 1], [0, 1], linestyle="--", color="gray", linewidth=1)
        plt.xlabel("False positive rate")
        plt.ylabel("True positive rate")
        plt.legend()
        savefig("metadata_only_roc_curves.png")
