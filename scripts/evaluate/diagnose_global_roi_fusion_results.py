"""Diagnostic analysis for Global + selected ROI fusion experiments."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.simple_xlsx_writer import write_xlsx  # noqa: E402


OUTPUT_ROOT = PROJECT_ROOT / "experiments" / "global_roi_fusion_500Data"
SUMMARY_CSV = OUTPUT_ROOT / "global_roi_fusion_summary.csv"
JOB_QUEUE = OUTPUT_ROOT / "global_roi_fusion_job_queue.csv"
DIAG_DIR = OUTPUT_ROOT / "diagnostic_analysis"
REPORT_MD = DIAG_DIR / "global_roi_fusion_diagnostic_report.md"
TABLES_XLSX = DIAG_DIR / "global_roi_fusion_diagnostic_tables.xlsx"
CORE_CSV = DIAG_DIR / "global_roi_fusion_core_comparison.csv"
CONFUSION_CSV = DIAG_DIR / "global_roi_fusion_confusion_summary.csv"
NEW_KEYS = ["global_eye", "global_cheek", "global_eye_cheek"]
CLASS_NAMES = ["normal", "mild", "severe"]


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.is_file():
        return pd.DataFrame()
    return pd.read_csv(path, encoding="utf-8-sig")


def _fmt(value: Any) -> str:
    try:
        if pd.isna(value):
            return ""
        return f"{float(value):.4f}"
    except Exception:
        return str(value)


def _md_table(frame: pd.DataFrame, columns: list[str]) -> str:
    if frame.empty:
        return "No data."
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join(["---"] * len(columns)) + " |",
    ]
    for _, row in frame.iterrows():
        lines.append("| " + " | ".join(_fmt(row.get(col, "")) for col in columns) + " |")
    return "\n".join(lines)


def _prediction_path(output_dir: str) -> Path:
    return Path(output_dir) / "summary" / "oof_predictions.csv"


def _confusion_from_predictions(key: str, name: str, output_dir: str) -> dict[str, Any] | None:
    path = _prediction_path(output_dir)
    if not path.is_file():
        return None
    pred = pd.read_csv(path, encoding="utf-8-sig")
    true_col = "label_3class" if "label_3class" in pred.columns else "y_true"
    pred_col = "pred_class" if "pred_class" in pred.columns else "y_pred"
    true = pd.to_numeric(pred[true_col], errors="coerce").astype(int)
    y_pred = pd.to_numeric(pred[pred_col], errors="coerce").astype(int)
    cm = confusion_matrix(true, y_pred, labels=[0, 1, 2])
    totals = cm.sum(axis=1)
    row: dict[str, Any] = {
        "experiment_key": key,
        "experiment_name": name,
        "oof_rows": len(pred),
    }
    for i, true_name in enumerate(CLASS_NAMES):
        row[f"{true_name}_total"] = int(totals[i])
        for j, pred_name in enumerate(CLASS_NAMES):
            row[f"{true_name}_to_{pred_name}"] = int(cm[i, j])
    row["mild_to_normal_rate"] = cm[1, 0] / totals[1] if totals[1] else np.nan
    row["mild_to_severe_rate"] = cm[1, 2] / totals[1] if totals[1] else np.nan
    row["severe_to_normal_rate"] = cm[2, 0] / totals[2] if totals[2] else np.nan
    row["severe_to_mild_rate"] = cm[2, 1] / totals[2] if totals[2] else np.nan
    return row


def _core_view(summary: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "experiment_key",
        "source_type",
        "macro_auc_mean",
        "balanced_accuracy_mean",
        "macro_f1_mean",
        "recall_mild_mean",
        "recall_severe_mean",
        "f1_mild_mean",
        "f1_severe_mean",
        "oof_macro_f1",
        "oof_balanced_accuracy",
        "oof_recall_mild",
        "oof_recall_severe",
        "output_dir",
    ]
    existing = [col for col in columns if col in summary.columns]
    return summary[existing].copy() if not summary.empty else pd.DataFrame(columns=columns)


def _delta_frame(summary: pd.DataFrame) -> pd.DataFrame:
    base = summary[summary["experiment_key"] == "global_resnet18_meanbg"]
    if base.empty:
        return pd.DataFrame()
    base = base.iloc[0]
    rows = []
    for _, row in summary[summary["experiment_key"].isin(NEW_KEYS)].iterrows():
        rows.append(
            {
                "experiment_key": row["experiment_key"],
                "delta_macro_f1_mean": row["macro_f1_mean"] - base["macro_f1_mean"],
                "delta_balanced_accuracy_mean": row["balanced_accuracy_mean"] - base["balanced_accuracy_mean"],
                "delta_recall_mild_mean": row["recall_mild_mean"] - base["recall_mild_mean"],
                "delta_recall_severe_mean": row["recall_severe_mean"] - base["recall_severe_mean"],
                "delta_oof_macro_f1": row["oof_macro_f1"] - base["oof_macro_f1"],
            }
        )
    return pd.DataFrame(rows)


def _status_frame(summary: pd.DataFrame, queue: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for key in NEW_KEYS:
        summary_row = summary[summary["experiment_key"] == key]
        queue_row = queue[queue["experiment_key"] == key] if not queue.empty and "experiment_key" in queue.columns else pd.DataFrame()
        rows.append(
            {
                "experiment_key": key,
                "summary_present": not summary_row.empty,
                "queue_status": queue_row.iloc[-1]["status"] if not queue_row.empty and "status" in queue_row.columns else "",
                "output_dir": summary_row.iloc[0]["output_dir"] if not summary_row.empty else "",
            }
        )
    return pd.DataFrame(rows)


def _key_info(summary: pd.DataFrame) -> list[str]:
    lookup = {row["experiment_key"]: row for _, row in summary.iterrows()}

    def get(key: str, metric: str) -> str:
        if key not in lookup:
            return "NA"
        return _fmt(lookup[key].get(metric, np.nan))

    new_available = summary[summary["experiment_key"].isin(NEW_KEYS)]
    best_key = ""
    if not new_available.empty:
        best_key = str(
            new_available.sort_values("macro_f1_mean", ascending=False).iloc[0][
                "experiment_key"
            ]
        )
    base_f1 = lookup.get("global_resnet18_meanbg", {}).get("macro_f1_mean", np.nan)
    best_single_f1 = summary[
        summary["experiment_key"].isin(["roi_single_eye_resnet18", "roi_single_cheek_resnet34"])
    ]["macro_f1_mean"].max()
    old_fusion_f1 = lookup.get("roi_fusion_multiroi5_resnet18", {}).get("macro_f1_mean", np.nan)
    best_new_f1 = lookup.get(best_key, {}).get("macro_f1_mean", np.nan) if best_key else np.nan

    return [
        f"1. ResNet18 meanbg macro-F1: {get('global_resnet18_meanbg', 'macro_f1_mean')}",
        f"2. ResNet18 meanbg BA: {get('global_resnet18_meanbg', 'balanced_accuracy_mean')}",
        f"3. ResNet18 meanbg mild recall: {get('global_resnet18_meanbg', 'recall_mild_mean')}",
        f"4. ResNet18 meanbg severe recall: {get('global_resnet18_meanbg', 'recall_severe_mean')}",
        f"5. Eye single ResNet18 macro-F1: {get('roi_single_eye_resnet18', 'macro_f1_mean')}",
        f"6. Eye single ResNet18 mild recall: {get('roi_single_eye_resnet18', 'recall_mild_mean')}",
        f"7. Cheek single ResNet34 macro-F1: {get('roi_single_cheek_resnet34', 'macro_f1_mean')}",
        f"8. Cheek single ResNet34 severe recall: {get('roi_single_cheek_resnet34', 'recall_severe_mean')}",
        f"9. Old ROI-only fusion ResNet18 macro-F1: {get('roi_fusion_multiroi5_resnet18', 'macro_f1_mean')}",
        f"10. Global+Eye macro-F1: {get('global_eye', 'macro_f1_mean')}",
        f"11. Global+Eye BA: {get('global_eye', 'balanced_accuracy_mean')}",
        f"12. Global+Eye mild recall: {get('global_eye', 'recall_mild_mean')}",
        f"13. Global+Eye severe recall: {get('global_eye', 'recall_severe_mean')}",
        f"14. Global+Cheek macro-F1: {get('global_cheek', 'macro_f1_mean')}",
        f"15. Global+Cheek BA: {get('global_cheek', 'balanced_accuracy_mean')}",
        f"16. Global+Cheek mild recall: {get('global_cheek', 'recall_mild_mean')}",
        f"17. Global+Cheek severe recall: {get('global_cheek', 'recall_severe_mean')}",
        f"18. Global+Eye+Cheek macro-F1: {get('global_eye_cheek', 'macro_f1_mean')}",
        f"19. Global+Eye+Cheek BA: {get('global_eye_cheek', 'balanced_accuracy_mean')}",
        f"20. Global+Eye+Cheek mild recall: {get('global_eye_cheek', 'recall_mild_mean')}",
        f"21. Global+Eye+Cheek severe recall: {get('global_eye_cheek', 'recall_severe_mean')}",
        f"22. Best Global+ROI model: {best_key or 'NA'}",
        f"23. Exceeds ResNet18 meanbg: {bool(best_new_f1 > base_f1) if pd.notna(best_new_f1) and pd.notna(base_f1) else 'NA'}",
        f"24. Exceeds best single ROI: {bool(best_new_f1 > best_single_f1) if pd.notna(best_new_f1) and pd.notna(best_single_f1) else 'NA'}",
        f"25. Exceeds old ROI-only fusion: {bool(best_new_f1 > old_fusion_f1) if pd.notna(best_new_f1) and pd.notna(old_fusion_f1) else 'NA'}",
        "26. Add lip/forehead: only if selected ROI fusion shows complementary benefit or error analysis supports it.",
        "27. ResNet34 version: consider only if ResNet18 fusion is competitive with baseline.",
        "28. Recommended next experiment: depends on best Global+ROI model and confusion pattern.",
    ]


def _write_report(
    summary: pd.DataFrame,
    status: pd.DataFrame,
    core: pd.DataFrame,
    confusion: pd.DataFrame,
    delta: pd.DataFrame,
    files_checked: list[str],
    missing_files: list[str],
) -> None:
    columns = [
        "experiment_key",
        "macro_f1_mean",
        "balanced_accuracy_mean",
        "recall_mild_mean",
        "recall_severe_mean",
        "oof_macro_f1",
    ]
    lines = [
        "# Global ROI Fusion Diagnostic Report",
        "",
        "## 1. Files Checked",
        "",
        *[f"- {item}" for item in files_checked],
        "",
    ]
    if missing_files:
        lines.extend(["Missing files:", "", *[f"- {item}" for item in missing_files], ""])
    lines.extend(
        [
            "## 2. Experiment Status",
            "",
            _md_table(status, ["experiment_key", "summary_present", "queue_status", "output_dir"]),
            "",
            "## 3. Core Result Table",
            "",
            _md_table(core, columns),
            "",
            "## 4. Comparison with Global ResNet18 MeanBG",
            "",
            _md_table(
                delta,
                [
                    "experiment_key",
                    "delta_macro_f1_mean",
                    "delta_balanced_accuracy_mean",
                    "delta_recall_mild_mean",
                    "delta_recall_severe_mean",
                    "delta_oof_macro_f1",
                ],
            ),
            "",
            "## 5. Comparison with ROI Single and ROI-only Fusion",
            "",
            "Use the core table to compare new Global+ROI rows with `roi_single_eye_resnet18`, `roi_single_cheek_resnet34`, and `roi_fusion_multiroi5_resnet18`.",
            "",
            "## 6. Per-class Analysis",
            "",
            "Focus on mild/severe recall and F1 columns in the core table.",
            "",
            "## 7. Confusion Matrix Analysis",
            "",
            _md_table(
                confusion,
                [
                    "experiment_key",
                    "mild_to_normal_rate",
                    "mild_to_severe_rate",
                    "severe_to_normal_rate",
                    "severe_to_mild_rate",
                ],
            ),
            "",
            "## 8. Recommendation for Next Step",
            "",
            "If no Global+ROI model exceeds the global baseline on macro-F1 and balanced accuracy, prioritize error analysis or ordinal/two-stage modeling over adding more ROI branches.",
            "",
            "## Key Information for Further Analysis",
            "",
            *_key_info(summary),
            "",
        ]
    )
    REPORT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> Path:
    DIAG_DIR.mkdir(parents=True, exist_ok=True)
    files_checked = [str(SUMMARY_CSV), str(JOB_QUEUE)]
    missing_files = [path for path in files_checked if not Path(path).is_file()]
    summary = _read_csv(SUMMARY_CSV)
    queue = _read_csv(JOB_QUEUE)
    status = _status_frame(summary, queue)
    core = _core_view(summary)
    confusion_rows = []
    if not summary.empty:
        for _, row in summary.iterrows():
            confusion = _confusion_from_predictions(
                str(row["experiment_key"]),
                str(row["experiment_name"]),
                str(row["output_dir"]),
            )
            if confusion is not None:
                confusion_rows.append(confusion)
    confusion_df = pd.DataFrame(confusion_rows)
    delta = _delta_frame(summary) if not summary.empty else pd.DataFrame()

    core.to_csv(CORE_CSV, index=False, encoding="utf-8-sig")
    confusion_df.to_csv(CONFUSION_CSV, index=False, encoding="utf-8-sig")
    write_xlsx(
        TABLES_XLSX,
        {
            "experiment_status": status,
            "core_comparison": core,
            "confusion_summary": confusion_df,
            "delta_vs_global": delta,
        },
    )
    _write_report(summary, status, core, confusion_df, delta, files_checked, missing_files)

    print(f"DIAGNOSTIC_REPORT={REPORT_MD}")
    print(f"DIAGNOSTIC_TABLES={TABLES_XLSX}")
    print(f"CORE_COMPARISON={CORE_CSV}")
    print(f"CONFUSION_SUMMARY={CONFUSION_CSV}")
    return REPORT_MD


if __name__ == "__main__":
    main()
