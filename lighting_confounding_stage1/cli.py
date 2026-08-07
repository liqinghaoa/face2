from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil

import pandas as pd

from .association_analysis import run_association_analysis
from .build_master import build_master, ensure_output_dirs, write_json
from .config import Stage1Config, json_safe
from .metadata_models import run_metadata_models
from .plotting import make_plots
from .reporting import write_reports
from .rgb_dependence import run_rgb_dependence
from .stratified_metrics import run_stratified_metrics


def run(config: Stage1Config) -> dict:
    dirs = ensure_output_dirs(config.output_dir)
    master, preflight = build_master(config)
    association = run_association_analysis(master, dirs["association"], bootstrap_repeats=config.bootstrap_repeats, seed=config.random_seed)
    metadata_results = run_metadata_models(
        master,
        dirs["metadata_only"],
        bootstrap_repeats=config.bootstrap_repeats,
        permutation_repeats=config.permutation_repeats,
        seed=config.random_seed,
        c_grid=config.logistic_C_grid,
        inner_cv_splits=config.inner_cv_splits,
    )
    rgb_results = run_rgb_dependence(master, dirs["rgb_dependence"])
    strat_results = run_stratified_metrics(
        master,
        dirs["stratified"],
        bootstrap_repeats=config.bootstrap_repeats,
        seed=config.random_seed,
        quantiles=config.stratification_quantiles,
        unstable_total=config.unstable_group_min_total,
        unstable_per_class=config.unstable_group_min_per_class,
    )
    summary_df = pd.DataFrame(
        [
            {
                "model": "RGB",
                "roc_auc": rgb_results["rgb_metrics"].get("roc_auc"),
                "accuracy": rgb_results["rgb_metrics"].get("accuracy"),
                "balanced_accuracy": rgb_results["rgb_metrics"].get("balanced_accuracy"),
            }
        ]
        + [
            {"model": row["model_name"], "roc_auc": row["roc_auc"], "accuracy": row["accuracy"], "balanced_accuracy": row["balanced_accuracy"]}
            for row in metadata_results["metrics"].to_dict("records")
        ]
    )
    summary_df.to_csv(dirs["stratified"] / "rgb_vs_metadata_summary.csv", index=False, encoding="utf-8-sig")
    make_plots(master, metadata_results["oof"], strat_results["stratified"], dirs["figures"])
    roc_plot = dirs["figures"] / "metadata_only_roc_curves.png"
    if roc_plot.is_file():
        shutil.copy2(roc_plot, dirs["metadata_only"] / "metadata_only_roc_curves.png")
    report_summary = write_reports(
        config.output_dir,
        preflight=preflight,
        association=association,
        metadata_results=metadata_results,
        rgb_results=rgb_results,
        strat_results=strat_results,
    )
    write_json(dirs["logs"] / "run_config_effective.json", config.__dict__)
    status = {
        "status": "completed",
        "output_dir": str(config.output_dir),
        "preflight": preflight["gates"],
        "metadata_only_metrics": metadata_results["metrics"].to_dict("records"),
        "rgb_metrics": rgb_results["rgb_metrics"],
        "worst_group": strat_results["worst"],
        "gates": report_summary["gates"],
    }
    write_json(dirs["logs"] / "run_status.json", status)
    return status


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    config = Stage1Config.from_yaml(args.config)
    if args.smoke:
        config = config.effective(smoke=True)
        config = Stage1Config(
            project_root=config.project_root,
            split_csv=config.split_csv,
            metadata_xlsx=config.metadata_xlsx,
            oof_predictions_csv=config.oof_predictions_csv,
            output_dir=config.output_dir / "_smoke",
            metadata_sheet=config.metadata_sheet,
            random_seed=config.random_seed,
            bootstrap_repeats=config.bootstrap_repeats,
            permutation_repeats=config.permutation_repeats,
            inner_cv_splits=config.inner_cv_splits,
            logistic_C_grid=config.logistic_C_grid,
            stratification_quantiles=config.stratification_quantiles,
            unstable_group_min_total=config.unstable_group_min_total,
            unstable_group_min_per_class=config.unstable_group_min_per_class,
            smoke_mode=True,
        )
    status = run(config)
    print(json.dumps(json_safe(status), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
