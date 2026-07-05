"""Summarize ResNet34/50 preprocessing backbone check experiments."""

from __future__ import annotations

import math
import sys
from pathlib import Path
from typing import Any, Sequence

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.simple_xlsx import write_xlsx  # noqa: E402


OUTPUT_ROOT = PROJECT_ROOT / "experiments" / "preprocess_ablation_500Data" / "backbone_check"
MANIFEST_PATH = (
    PROJECT_ROOT
    / "config"
    / "train"
    / "preprocess_ablation_backbone_check"
    / "backbone_check_config_manifest.csv"
)
SUMMARY_REQUIRED = (
    "summary/fold_metrics_all.csv",
    "summary/mean_metrics.csv",
    "summary/oof_metrics.csv",
    "summary/oof_predictions.csv",
    "summary/summary_report.md",
)
EXPERIMENT_COLUMNS = (
    "backbone",
    "variant_name",
    "experiment_name",
    "output_dir",
    "macro_auc_mean",
    "macro_auc_std",
    "balanced_accuracy_mean",
    "balanced_accuracy_std",
    "macro_f1_mean",
    "macro_f1_std",
    "macro_recall_mean",
    "macro_recall_std",
    "accuracy_mean",
    "accuracy_std",
    "severe_vs_rest_auc_mean",
    "normal_vs_abnormal_auc_mean",
    "recall_normal_mean",
    "recall_mild_mean",
    "recall_severe_mean",
    "f1_normal_mean",
    "f1_mild_mean",
    "f1_severe_mean",
    "oof_macro_auc",
    "oof_balanced_accuracy",
    "oof_macro_f1",
    "oof_macro_recall",
    "oof_accuracy",
    "oof_severe_vs_rest_auc",
    "oof_normal_vs_abnormal_auc",
)


def _resolve(path: str | Path) -> Path:
    value = Path(path).expanduser()
    return value.resolve() if value.is_absolute() else (PROJECT_ROOT / value).resolve()


def _load_manifest(path: Path = MANIFEST_PATH) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(f"Backbone check manifest not found: {path}")
    frame = pd.read_csv(path, dtype=str, encoding="utf-8-sig").fillna("")
    required = {"job_id", "backbone", "variant_name", "experiment_name", "output_root"}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"Manifest missing columns: {missing}")
    return frame


def _completed_experiment_dir(output_root: Path, experiment_name: str) -> Path | None:
    if not output_root.is_dir():
        return None
    candidates = sorted(
        [path for path in output_root.glob(f"{experiment_name}*") if path.is_dir()],
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    for candidate in candidates:
        if all((candidate / item).is_file() for item in SUMMARY_REQUIRED):
            return candidate
    return None


def _metric_mean_std(frame: pd.DataFrame, metric: str) -> tuple[float, float]:
    if metric not in frame.columns:
        return float("nan"), float("nan")
    values = pd.to_numeric(frame[metric], errors="coerce")
    return float(values.mean()), float(values.std(ddof=1))


def _oof_metric(oof: pd.DataFrame, metric: str) -> float:
    if oof.empty or metric not in oof.columns:
        return float("nan")
    return float(pd.to_numeric(oof.iloc[0].get(metric), errors="coerce"))


def _read_experiment(row: pd.Series) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame] | None:
    output_root = _resolve(str(row["output_root"]))
    experiment_dir = _completed_experiment_dir(output_root, str(row["experiment_name"]))
    if experiment_dir is None:
        return None
    fold = pd.read_csv(experiment_dir / "summary" / "fold_metrics_all.csv")
    mean = pd.read_csv(experiment_dir / "summary" / "mean_metrics.csv")
    oof = pd.read_csv(experiment_dir / "summary" / "oof_metrics.csv")
    if {"metric", "mean", "std"}.issubset(mean.columns):
        mean_lookup = mean.set_index("metric").to_dict("index")
    else:
        mean_lookup = {}

    def metric_mean(metric: str) -> float:
        if metric in mean_lookup:
            return float(pd.to_numeric(mean_lookup[metric].get("mean"), errors="coerce"))
        return _metric_mean_std(fold, metric)[0]

    def metric_std(metric: str) -> float:
        if metric in mean_lookup:
            return float(pd.to_numeric(mean_lookup[metric].get("std"), errors="coerce"))
        return _metric_mean_std(fold, metric)[1]

    summary_row: dict[str, Any] = {
        "backbone": row["backbone"],
        "variant_name": row["variant_name"],
        "experiment_name": row["experiment_name"],
        "output_dir": str(experiment_dir),
        "macro_auc_mean": metric_mean("macro_auc"),
        "macro_auc_std": metric_std("macro_auc"),
        "balanced_accuracy_mean": metric_mean("balanced_accuracy"),
        "balanced_accuracy_std": metric_std("balanced_accuracy"),
        "macro_f1_mean": metric_mean("macro_f1"),
        "macro_f1_std": metric_std("macro_f1"),
        "macro_recall_mean": metric_mean("macro_recall"),
        "macro_recall_std": metric_std("macro_recall"),
        "accuracy_mean": metric_mean("accuracy"),
        "accuracy_std": metric_std("accuracy"),
        "severe_vs_rest_auc_mean": _metric_mean_std(fold, "severe_vs_rest_auc")[0],
        "normal_vs_abnormal_auc_mean": _metric_mean_std(fold, "normal_vs_abnormal_auc")[0],
        "recall_normal_mean": _metric_mean_std(fold, "recall_normal")[0],
        "recall_mild_mean": _metric_mean_std(fold, "recall_mild")[0],
        "recall_severe_mean": _metric_mean_std(fold, "recall_severe")[0],
        "f1_normal_mean": _metric_mean_std(fold, "f1_normal")[0],
        "f1_mild_mean": _metric_mean_std(fold, "f1_mild")[0],
        "f1_severe_mean": _metric_mean_std(fold, "f1_severe")[0],
        "oof_macro_auc": _oof_metric(oof, "macro_auc"),
        "oof_balanced_accuracy": _oof_metric(oof, "balanced_accuracy"),
        "oof_macro_f1": _oof_metric(oof, "macro_f1"),
        "oof_macro_recall": _oof_metric(oof, "macro_recall"),
        "oof_accuracy": _oof_metric(oof, "accuracy"),
        "oof_severe_vs_rest_auc": _oof_metric(oof, "severe_vs_rest_auc"),
        "oof_normal_vs_abnormal_auc": _oof_metric(oof, "normal_vs_abnormal_auc"),
    }
    fold = fold.assign(
        job_id=row["job_id"],
        backbone=row["backbone"],
        variant_name=row["variant_name"],
        experiment_name=row["experiment_name"],
        output_dir=str(experiment_dir),
    )
    oof = oof.assign(
        job_id=row["job_id"],
        backbone=row["backbone"],
        variant_name=row["variant_name"],
        experiment_name=row["experiment_name"],
        output_dir=str(experiment_dir),
    )
    return summary_row, fold, oof


def _num(row: pd.Series, column: str) -> float:
    return float(pd.to_numeric(row.get(column), errors="coerce"))


def _delta_row(backbone: str, baseline: pd.Series, meanbg: pd.Series) -> dict[str, Any]:
    delta = {
        "backbone": backbone,
        "delta_macro_auc_mean": _num(meanbg, "macro_auc_mean") - _num(baseline, "macro_auc_mean"),
        "delta_balanced_accuracy_mean": _num(meanbg, "balanced_accuracy_mean") - _num(baseline, "balanced_accuracy_mean"),
        "delta_macro_f1_mean": _num(meanbg, "macro_f1_mean") - _num(baseline, "macro_f1_mean"),
        "delta_recall_severe_mean": _num(meanbg, "recall_severe_mean") - _num(baseline, "recall_severe_mean"),
        "delta_f1_severe_mean": _num(meanbg, "f1_severe_mean") - _num(baseline, "f1_severe_mean"),
        "delta_severe_vs_rest_auc_mean": _num(meanbg, "severe_vs_rest_auc_mean") - _num(baseline, "severe_vs_rest_auc_mean"),
        "delta_normal_vs_abnormal_auc_mean": _num(meanbg, "normal_vs_abnormal_auc_mean") - _num(baseline, "normal_vs_abnormal_auc_mean"),
        "delta_oof_macro_auc": _num(meanbg, "oof_macro_auc") - _num(baseline, "oof_macro_auc"),
        "delta_oof_balanced_accuracy": _num(meanbg, "oof_balanced_accuracy") - _num(baseline, "oof_balanced_accuracy"),
        "delta_oof_macro_f1": _num(meanbg, "oof_macro_f1") - _num(baseline, "oof_macro_f1"),
        "delta_oof_recall_severe": _num(meanbg, "oof_recall_severe") - _num(baseline, "oof_recall_severe")
        if "oof_recall_severe" in meanbg.index and "oof_recall_severe" in baseline.index
        else float("nan"),
    }
    auc_ok = delta["delta_macro_auc_mean"] >= -0.02
    hard_ok = delta["delta_balanced_accuracy_mean"] > 0 or delta["delta_macro_f1_mean"] > 0
    severe_ok = delta["delta_recall_severe_mean"] >= 0
    delta["conclusion"] = "meanbg_effective" if auc_ok and hard_ok and severe_ok else "meanbg_not_confirmed"
    return delta


def _paired_delta(summary: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for backbone in sorted(summary["backbone"].dropna().unique()):
        subset = summary[summary["backbone"] == backbone]
        baseline = subset[subset["variant_name"] == "hybrid_black_baseline"]
        meanbg = subset[subset["variant_name"] == "hybrid_imagenet_meanbg"]
        if baseline.empty or meanbg.empty:
            rows.append({"backbone": backbone, "conclusion": "incomplete_pair"})
            continue
        rows.append(_delta_row(backbone, baseline.iloc[0], meanbg.iloc[0]))
    return pd.DataFrame(rows)


def _fmt(value: Any) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "NA"
    return "NA" if not math.isfinite(number) else f"{number:.4f}"


def _overall_conclusion(paired: pd.DataFrame) -> str:
    if paired.empty:
        return "No complete backbone pair is available."
    effective = int((paired.get("conclusion") == "meanbg_effective").sum())
    complete = int(paired["conclusion"].isin(["meanbg_effective", "meanbg_not_confirmed"]).sum())
    if complete >= 2 and effective == complete:
        return "hybrid_imagenet_meanbg shows consistent improvement in hard classification stability across larger backbones."
    if effective >= 1:
        return "hybrid_imagenet_meanbg shows partial backbone consistency."
    return "hybrid_imagenet_meanbg improvement may be specific to ResNet18 or unstable across backbones."


def _write_markdown(path: Path, summary: pd.DataFrame, paired: pd.DataFrame) -> None:
    lines = [
        "# Backbone Check Summary",
        "",
        "## Purpose",
        "",
        "This analysis compares `hybrid_black_baseline` and `hybrid_imagenet_meanbg` under ResNet34 and ResNet50.",
        "",
        "## Experiment Summary",
        "",
    ]
    if summary.empty:
        lines.append("No completed backbone check experiment was found.")
    else:
        lines.extend(
            [
                "| backbone | variant | macro-AUC | BA | macro-F1 | severe recall |",
                "|---|---|---:|---:|---:|---:|",
            ]
        )
        for row in summary.itertuples(index=False):
            lines.append(
                f"| {row.backbone} | {row.variant_name} | {_fmt(row.macro_auc_mean)} | "
                f"{_fmt(row.balanced_accuracy_mean)} | {_fmt(row.macro_f1_mean)} | "
                f"{_fmt(row.recall_severe_mean)} |"
            )
    lines.extend(["", "## Paired Delta", ""])
    if paired.empty:
        lines.append("No paired comparison is available.")
    else:
        lines.extend(
            [
                "| backbone | Δmacro-AUC | ΔBA | Δmacro-F1 | Δsevere recall | conclusion |",
                "|---|---:|---:|---:|---:|---|",
            ]
        )
        for row in paired.itertuples(index=False):
            lines.append(
                f"| {row.backbone} | {_fmt(getattr(row, 'delta_macro_auc_mean', float('nan')))} | "
                f"{_fmt(getattr(row, 'delta_balanced_accuracy_mean', float('nan')))} | "
                f"{_fmt(getattr(row, 'delta_macro_f1_mean', float('nan')))} | "
                f"{_fmt(getattr(row, 'delta_recall_severe_mean', float('nan')))} | "
                f"{row.conclusion} |"
            )
    lines.extend(["", "## Overall Conclusion", "", _overall_conclusion(paired)])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv: Sequence[str] | None = None) -> int:
    del argv
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    manifest = _load_manifest()
    summary_rows: list[dict[str, Any]] = []
    fold_frames: list[pd.DataFrame] = []
    oof_frames: list[pd.DataFrame] = []
    skipped: list[str] = []

    for _, manifest_row in manifest.iterrows():
        result = _read_experiment(manifest_row)
        if result is None:
            skipped.append(str(manifest_row["experiment_name"]))
            continue
        summary_row, fold, oof = result
        summary_rows.append(summary_row)
        fold_frames.append(fold)
        oof_frames.append(oof)

    summary = pd.DataFrame(summary_rows)
    for column in EXPERIMENT_COLUMNS:
        if column not in summary.columns:
            summary[column] = pd.NA
    summary = summary[list(EXPERIMENT_COLUMNS)]
    fold_all = pd.concat(fold_frames, ignore_index=True) if fold_frames else pd.DataFrame()
    oof_all = pd.concat(oof_frames, ignore_index=True) if oof_frames else pd.DataFrame()
    paired = _paired_delta(summary) if not summary.empty else pd.DataFrame()

    summary.to_csv(OUTPUT_ROOT / "backbone_check_summary.csv", index=False, encoding="utf-8-sig")
    fold_all.to_csv(OUTPUT_ROOT / "backbone_check_fold_metrics_all.csv", index=False, encoding="utf-8-sig")
    oof_all.to_csv(OUTPUT_ROOT / "backbone_check_oof_metrics_all.csv", index=False, encoding="utf-8-sig")
    paired.to_csv(OUTPUT_ROOT / "backbone_check_paired_delta.csv", index=False, encoding="utf-8-sig")
    write_xlsx(
        OUTPUT_ROOT / "backbone_check_summary.xlsx",
        {
            "experiment_summary": summary,
            "fold_metrics_all": fold_all,
            "oof_metrics_all": oof_all,
            "paired_delta": paired,
        },
    )
    _write_markdown(OUTPUT_ROOT / "backbone_check_summary.md", summary, paired)

    print(f"Summarized {len(summary)} backbone check experiments: {OUTPUT_ROOT}")
    if skipped:
        print(f"Skipped incomplete/missing experiments: {skipped}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
