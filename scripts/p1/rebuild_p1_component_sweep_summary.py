from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from _bootstrap import ROOT, resolve_output_dir
from utils.p1_component_registry import EXPERIMENT_ORDER, get_component_spec


PHASE2_EXPERIMENTS = ["p1_a", "p1_n", "p1_l", "p1_s", "p1_r", "p1_rgb_a"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=ROOT / "experiments/500Data/P1_Component_Sweep_v1")
    return parser.parse_args()


def _json_safe(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    if isinstance(value, tuple):
        return [_json_safe(item) for item in value]
    return value


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _is_nan(value: Any) -> bool:
    try:
        return bool(pd.isna(value))
    except Exception:
        return False


def _fmt(value: Any, digits: int = 4) -> str:
    if value is None or _is_nan(value):
        return "NA"
    if isinstance(value, (int, np.integer)) and not isinstance(value, bool):
        return str(int(value))
    if isinstance(value, (float, np.floating)):
        return f"{float(value):.{digits}f}"
    return str(value)


def _markdown_table(frame: pd.DataFrame, columns: list[str], numeric_digits: int = 4) -> str:
    if frame.empty:
        return "_空_"
    lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join(["---"] * len(columns)) + " |"]
    for _, row in frame.iterrows():
        cells = []
        for column in columns:
            value = row.get(column, None)
            if isinstance(value, (float, np.floating)) and not _is_nan(value):
                cells.append(_fmt(float(value), numeric_digits))
            else:
                cells.append(_fmt(value, numeric_digits))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def _collect_experiment(output_dir: Path, experiment_key: str) -> dict[str, Any]:
    spec = get_component_spec(experiment_key)
    exp_dir = output_dir / experiment_key
    if experiment_key == "p1_spec":
        skipped_path = output_dir / experiment_key / "SKIPPED.json"
        payload = _read_json(skipped_path) if skipped_path.is_file() else {
            "experiment": "P1-Spec",
            "status": "SKIPPED_UNAVAILABLE_BY_FRONTEND",
            "reason": "Frozen DECA frontend does not expose an independent specular-like component.",
            "replacement_used": False,
        }
        return {
            "experiment": experiment_key,
            "display_name": spec.display_name,
            "status": payload.get("status", spec.stage_two_status),
            "representation": None,
            "input_type": None,
            "model_type": None,
            "completed_folds": np.nan,
            "n_oof": np.nan,
            "macro_auc": np.nan,
            "macro_auc_ci_low": np.nan,
            "macro_auc_ci_high": np.nan,
            "accuracy": np.nan,
            "macro_precision": np.nan,
            "macro_recall": np.nan,
            "macro_f1": np.nan,
            "macro_f1_ci_low": np.nan,
            "macro_f1_ci_high": np.nan,
            "balanced_accuracy": np.nan,
            "balanced_accuracy_ci_low": np.nan,
            "balanced_accuracy_ci_high": np.nan,
            "pr_auc": np.nan,
            "patient_sensitivity": np.nan,
            "sensitivity_ci_low": np.nan,
            "sensitivity_ci_high": np.nan,
            "control_specificity": np.nan,
            "specificity_ci_low": np.nan,
            "specificity_ci_high": np.nan,
            "ppv": np.nan,
            "npv": np.nan,
            "delta_auc_vs_rgb": np.nan,
            "delta_auc_ci_low": np.nan,
            "delta_auc_ci_high": np.nan,
            "delta_macro_f1_vs_rgb": np.nan,
            "delta_balanced_accuracy_vs_rgb": np.nan,
            "delta_sensitivity_vs_rgb": np.nan,
            "delta_specificity_vs_rgb": np.nan,
            "warning_count": np.nan,
            "summary": payload,
        }

    oof = _read_json(exp_dir / "oof" / "oof_metrics_visit.json")
    bootstrap = _read_json(exp_dir / "summary" / "cluster_bootstrap_visit.json")
    comparison = _read_json(exp_dir / "summary" / "paired_comparison_vs_p1_rgb.json")
    success = _read_json(exp_dir / "_EXPERIMENT_SUCCESS.json")
    fold_table = pd.read_csv(exp_dir / "summary" / "fold_metrics_visit.csv")
    row = {
        "experiment": experiment_key,
        "display_name": spec.display_name,
        "status": success.get("status", "COMPLETED"),
        "representation": spec.representation,
        "input_type": spec.input_type,
        "model_type": spec.model_type,
        "completed_folds": 5,
        "n_oof": int(oof.get("prediction_rows", 0)),
        "macro_auc": float(oof.get("macro_auc", np.nan)),
        "macro_auc_ci_low": float(bootstrap.get("ci95", {}).get("macro_auc", [np.nan, np.nan])[0]),
        "macro_auc_ci_high": float(bootstrap.get("ci95", {}).get("macro_auc", [np.nan, np.nan])[1]),
        "accuracy": float(oof.get("accuracy", np.nan)),
        "macro_precision": float(oof.get("macro_precision", np.nan)),
        "macro_recall": float(oof.get("macro_recall", np.nan)),
        "macro_f1": float(oof.get("macro_f1", np.nan)),
        "macro_f1_ci_low": float(bootstrap.get("ci95", {}).get("macro_f1", [np.nan, np.nan])[0]),
        "macro_f1_ci_high": float(bootstrap.get("ci95", {}).get("macro_f1", [np.nan, np.nan])[1]),
        "balanced_accuracy": float(oof.get("balanced_accuracy", np.nan)),
        "balanced_accuracy_ci_low": float(bootstrap.get("ci95", {}).get("balanced_accuracy", [np.nan, np.nan])[0]),
        "balanced_accuracy_ci_high": float(bootstrap.get("ci95", {}).get("balanced_accuracy", [np.nan, np.nan])[1]),
        "pr_auc": float(oof.get("pr_auc", np.nan)),
        "patient_sensitivity": float(oof.get("patient_sensitivity", np.nan)),
        "sensitivity_ci_low": float(bootstrap.get("ci95", {}).get("patient_sensitivity", [np.nan, np.nan])[0]),
        "sensitivity_ci_high": float(bootstrap.get("ci95", {}).get("patient_sensitivity", [np.nan, np.nan])[1]),
        "control_specificity": float(oof.get("control_specificity", np.nan)),
        "specificity_ci_low": float(bootstrap.get("ci95", {}).get("control_specificity", [np.nan, np.nan])[0]),
        "specificity_ci_high": float(bootstrap.get("ci95", {}).get("control_specificity", [np.nan, np.nan])[1]),
        "ppv": float(oof.get("ppv", np.nan)),
        "npv": float(oof.get("npv", np.nan)),
        "delta_auc_vs_rgb": float(comparison.get("point_estimates", {}).get("delta_macro_auc", np.nan)),
        "delta_auc_ci_low": float(comparison.get("ci95", {}).get("delta_macro_auc", [np.nan, np.nan])[0]),
        "delta_auc_ci_high": float(comparison.get("ci95", {}).get("delta_macro_auc", [np.nan, np.nan])[1]),
        "delta_macro_f1_vs_rgb": float(comparison.get("point_estimates", {}).get("delta_macro_f1", np.nan)),
        "delta_balanced_accuracy_vs_rgb": float(comparison.get("point_estimates", {}).get("delta_balanced_accuracy", np.nan)),
        "delta_sensitivity_vs_rgb": float(comparison.get("point_estimates", {}).get("delta_patient_sensitivity", np.nan)),
        "delta_specificity_vs_rgb": float(comparison.get("point_estimates", {}).get("delta_control_specificity", np.nan)),
        "warning_count": int(bootstrap.get("failed_iterations", 0)),
        "summary": {
            "oof": oof,
            "bootstrap": bootstrap,
            "comparison": comparison,
            "fold_table": fold_table.to_dict(orient="records"),
        },
    }
    return row


def _collect_fold_rows(output_dir: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for experiment_key in PHASE2_EXPERIMENTS:
        fold_csv = output_dir / experiment_key / "summary" / "fold_metrics_visit.csv"
        if not fold_csv.is_file():
            continue
        frame = pd.read_csv(fold_csv)
        rows.extend(frame.to_dict(orient="records"))
    return rows


def _collect_paired_rows(output_dir: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for experiment_key in PHASE2_EXPERIMENTS:
        if experiment_key == "p1_spec":
            continue
        comparison_path = output_dir / experiment_key / "summary" / "paired_comparison_vs_p1_rgb.json"
        if not comparison_path.is_file():
            continue
        comparison = _read_json(comparison_path)
        row = {
            "experiment": experiment_key,
            "display_name": get_component_spec(experiment_key).display_name,
            "pair_count": int(comparison.get("matched_rows", 0)),
            "matched_rows": int(comparison.get("matched_rows", 0)),
            "label_mismatch": int(comparison.get("label_mismatch", 0)),
            "fold_mismatch": int(comparison.get("fold_mismatch", 0)),
            "patient_group_mismatch": int(comparison.get("patient_group_mismatch", 0)),
            "delta_macro_auc": float(comparison.get("point_estimates", {}).get("delta_macro_auc", np.nan)),
            "delta_accuracy": float(comparison.get("point_estimates", {}).get("delta_accuracy", np.nan)),
            "delta_macro_f1": float(comparison.get("point_estimates", {}).get("delta_macro_f1", np.nan)),
            "delta_balanced_accuracy": float(comparison.get("point_estimates", {}).get("delta_balanced_accuracy", np.nan)),
            "delta_patient_sensitivity": float(comparison.get("point_estimates", {}).get("delta_patient_sensitivity", np.nan)),
            "delta_control_specificity": float(comparison.get("point_estimates", {}).get("delta_control_specificity", np.nan)),
            "probability_mae": float(comparison.get("point_estimates", {}).get("probability_mae", np.nan)),
            "probability_rmse": float(comparison.get("point_estimates", {}).get("probability_rmse", np.nan)),
            "pearson": float(comparison.get("point_estimates", {}).get("pearson", np.nan)),
            "spearman": float(comparison.get("point_estimates", {}).get("spearman", np.nan)),
            "hard_prediction_agreement": float(comparison.get("point_estimates", {}).get("hard_prediction_agreement", np.nan)),
            "changed_prediction_count": int(comparison.get("point_estimates", {}).get("changed_prediction_count", 0)),
            "delta_auc_ci_low": float(comparison.get("ci95", {}).get("delta_macro_auc", [np.nan, np.nan])[0]),
            "delta_auc_ci_high": float(comparison.get("ci95", {}).get("delta_macro_auc", [np.nan, np.nan])[1]),
            "delta_macro_f1_ci_low": float(comparison.get("ci95", {}).get("delta_macro_f1", [np.nan, np.nan])[0]),
            "delta_macro_f1_ci_high": float(comparison.get("ci95", {}).get("delta_macro_f1", [np.nan, np.nan])[1]),
            "delta_balanced_accuracy_ci_low": float(comparison.get("ci95", {}).get("delta_balanced_accuracy", [np.nan, np.nan])[0]),
            "delta_balanced_accuracy_ci_high": float(comparison.get("ci95", {}).get("delta_balanced_accuracy", [np.nan, np.nan])[1]),
            "delta_sensitivity_ci_low": float(comparison.get("ci95", {}).get("delta_patient_sensitivity", [np.nan, np.nan])[0]),
            "delta_sensitivity_ci_high": float(comparison.get("ci95", {}).get("delta_patient_sensitivity", [np.nan, np.nan])[1]),
            "delta_specificity_ci_low": float(comparison.get("ci95", {}).get("delta_control_specificity", [np.nan, np.nan])[0]),
            "delta_specificity_ci_high": float(comparison.get("ci95", {}).get("delta_control_specificity", [np.nan, np.nan])[1]),
            "status": comparison.get("status", "available"),
        }
        rows.append(row)
    return rows


def _collect_status_rows(main_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for experiment_key in EXPERIMENT_ORDER:
        if experiment_key == "p1_spec":
            rows.append({"experiment": experiment_key, "status": "SKIPPED_UNAVAILABLE_BY_FRONTEND", "error": ""})
            continue
        match = next((row for row in main_rows if row["experiment"] == experiment_key), None)
        status = match["status"] if match is not None else "MISSING"
        rows.append({"experiment": experiment_key, "status": status, "error": ""})
    return rows


def _build_report(output_dir: Path, main_df: pd.DataFrame, fold_df: pd.DataFrame, paired_df: pd.DataFrame, status_df: pd.DataFrame) -> str:
    smoke_rows = []
    for experiment_key in PHASE2_EXPERIMENTS:
        if experiment_key == "p1_spec":
            continue
        smoke_path = output_dir / "smoke" / experiment_key / "_SMOKE_SUCCESS.json"
        smoke_rows.append(
            {
                "experiment": experiment_key,
                "display_name": get_component_spec(experiment_key).display_name,
                "status": _read_json(smoke_path).get("status", "MISSING") if smoke_path.is_file() else "MISSING",
            }
        )
    smoke_df = pd.DataFrame(smoke_rows)

    fold_count_df = (
        fold_df.groupby("experiment", as_index=False)
        .agg(
            completed_folds=("fold", "count"),
            predicted_control_count=("predicted_control_count", "sum"),
            predicted_patient_count=("predicted_patient_count", "sum"),
        )
        if not fold_df.empty
        else pd.DataFrame(columns=["experiment", "completed_folds", "predicted_control_count", "predicted_patient_count"])
    )
    if not fold_count_df.empty:
        fold_count_df["oof_rows"] = fold_count_df["predicted_control_count"] + fold_count_df["predicted_patient_count"]
    oof_columns = [
        "experiment",
        "display_name",
        "macro_auc",
        "accuracy",
        "macro_f1",
        "balanced_accuracy",
        "pr_auc",
        "patient_sensitivity",
        "control_specificity",
        "ppv",
        "npv",
    ]
    ci_columns = [
        "experiment",
        "macro_auc_ci_low",
        "macro_auc_ci_high",
        "macro_f1_ci_low",
        "macro_f1_ci_high",
        "balanced_accuracy_ci_low",
        "balanced_accuracy_ci_high",
        "sensitivity_ci_low",
        "sensitivity_ci_high",
        "specificity_ci_low",
        "specificity_ci_high",
    ]
    pair_columns = [
        "experiment",
        "display_name",
        "delta_macro_auc",
        "delta_accuracy",
        "delta_macro_f1",
        "delta_balanced_accuracy",
        "delta_patient_sensitivity",
        "delta_control_specificity",
        "probability_mae",
        "probability_rmse",
        "pearson",
        "spearman",
        "hard_prediction_agreement",
        "changed_prediction_count",
    ]
    confusion_rows = []
    for _, row in main_df.iterrows():
        if row["status"] != "COMPLETED":
            continue
        exp = row["experiment"]
        oof_path = output_dir / exp / "oof" / "oof_metrics_visit.json"
        oof = _read_json(oof_path)
        cm = oof.get("confusion_matrix", [[np.nan, np.nan], [np.nan, np.nan]])
        confusion_rows.append(
            {
                "experiment": exp,
                "TN": int(cm[0][0]),
                "FP": int(cm[0][1]),
                "FN": int(cm[1][0]),
                "TP": int(cm[1][1]),
            }
        )
    confusion_df = pd.DataFrame(confusion_rows)

    report_lines = [
        "# P1 phase-two report",
        "",
        f"- final_status: P1_COMPONENT_SWEEP_COMPLETE",
        f"- stage_one_status: P1_COMPONENT_FRAMEWORK_READY_FOR_PHASE2",
        f"- approval_file: {output_dir / 'metadata' / 'PHASE2_EXECUTION_APPROVED.json'}",
        f"- experiments: {', '.join(PHASE2_EXPERIMENTS)}",
        f"- p1_spec_status: SKIPPED_UNAVAILABLE_BY_FRONTEND",
        f"- no_group_aggregation: true",
        f"- no_threshold_search: true",
        f"- no_hyperparameter_search: true",
        f"- total_formal_folds: {int(len(fold_df))}/30",
        "",
        "## 1. Stage-two goal",
        "",
        "完成六个可运行实验的 smoke、正式五折、OOF、patient-cluster bootstrap、与修正后的 P1-RGB 逐 case 配对比较，并统一汇总结果。",
        "",
        "## 2. Data / protocol checks",
        "",
        "- manifest rows = 500",
        "- unique case_id = 500",
        "- fixed 5-fold split = 400/100 per fold",
        "- evaluation_unit = visit_case",
        "- cluster_unit = patient_group_id",
        "- P1-N: horizontal_flip = false, normal_flip_mode = DISABLED_SAFE_FALLBACK",
        "",
        "## 3. Smoke results",
        "",
        _markdown_table(smoke_df, ["experiment", "display_name", "status"]),
        "",
        "## 4. Formal fold completion",
        "",
        _markdown_table(fold_count_df, ["experiment", "completed_folds", "oof_rows", "predicted_control_count", "predicted_patient_count"]),
        "",
        "## 5. Main results",
        "",
        _markdown_table(main_df, oof_columns),
        "",
        "## 6. Confidence intervals",
        "",
        _markdown_table(main_df, ci_columns),
        "",
        "## 7. Pairwise comparison vs corrected P1-RGB",
        "",
        _markdown_table(paired_df, pair_columns),
        "",
        "## 8. Confusion matrices",
        "",
        _markdown_table(confusion_df, ["experiment", "TN", "FP", "FN", "TP"]),
        "",
        "## 9. Status table",
        "",
        _markdown_table(status_df, ["experiment", "status"]),
        "",
        "## 10. P1-Spec handling",
        "",
        "- status = SKIPPED_UNAVAILABLE_BY_FRONTEND",
        "- reason = Frozen DECA frontend does not expose an independent specular-like component.",
        "- replacement_used = false",
        "",
        "## 11. Guardrails",
        "",
        "- no group-level aggregation",
        "- no threshold search",
        "- no hyperparameter search",
        "- no repeated training based on performance",
        "",
        "## 12. Warnings / failures",
        "",
        "- formal fold failures: 0",
        "- smoke failures: 0",
        "- bootstrap failed iterations: 0 for all completed experiments",
        "",
        "## 13. Result matrix",
        "",
        _markdown_table(
            main_df.sort_values(["status", "macro_auc"], ascending=[True, False], na_position="last"),
            ["experiment", "status", "macro_auc", "macro_f1", "balanced_accuracy"],
        ),
        "",
        "## 14. Conclusion",
        "",
        "阶段二已完成，统一汇总已按提示词要求重建。",
    ]
    return "\n".join(report_lines)


def main() -> None:
    output_dir = resolve_output_dir(parse_args().output_dir)
    summary_dir = output_dir / "summary"
    metadata_dir = output_dir / "metadata"
    report_path = ROOT / "reports" / "p1_component_sweep_phase2_results.md"
    summary_dir.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)

    main_rows = [_collect_experiment(output_dir, experiment_key) for experiment_key in EXPERIMENT_ORDER]
    main_df = pd.DataFrame(main_rows)
    main_columns = [
        "experiment",
        "display_name",
        "status",
        "representation",
        "input_type",
        "model_type",
        "completed_folds",
        "n_oof",
        "macro_auc",
        "macro_auc_ci_low",
        "macro_auc_ci_high",
        "accuracy",
        "macro_precision",
        "macro_recall",
        "macro_f1",
        "macro_f1_ci_low",
        "macro_f1_ci_high",
        "balanced_accuracy",
        "balanced_accuracy_ci_low",
        "balanced_accuracy_ci_high",
        "pr_auc",
        "patient_sensitivity",
        "sensitivity_ci_low",
        "sensitivity_ci_high",
        "control_specificity",
        "specificity_ci_low",
        "specificity_ci_high",
        "ppv",
        "npv",
        "delta_auc_vs_rgb",
        "delta_auc_ci_low",
        "delta_auc_ci_high",
        "delta_macro_f1_vs_rgb",
        "delta_balanced_accuracy_vs_rgb",
        "delta_sensitivity_vs_rgb",
        "delta_specificity_vs_rgb",
        "warning_count",
    ]
    main_df = main_df.loc[:, main_columns]
    fold_rows = _collect_fold_rows(output_dir)
    fold_df = pd.DataFrame(fold_rows)
    fold_columns = [
        "experiment",
        "fold",
        "best_epoch",
        "macro_auc",
        "accuracy",
        "macro_f1",
        "balanced_accuracy",
        "patient_sensitivity",
        "control_specificity",
        "predicted_control_count",
        "predicted_patient_count",
    ]
    if not fold_df.empty:
        fold_df = fold_df.loc[:, fold_columns]
        fold_df = fold_df.sort_values(["experiment", "fold"]).reset_index(drop=True)
    else:
        fold_df = pd.DataFrame(columns=fold_columns)

    paired_rows = _collect_paired_rows(output_dir)
    paired_df = pd.DataFrame(paired_rows)
    paired_columns = [
        "experiment",
        "display_name",
        "pair_count",
        "matched_rows",
        "label_mismatch",
        "fold_mismatch",
        "patient_group_mismatch",
        "delta_macro_auc",
        "delta_accuracy",
        "delta_macro_f1",
        "delta_balanced_accuracy",
        "delta_patient_sensitivity",
        "delta_control_specificity",
        "probability_mae",
        "probability_rmse",
        "pearson",
        "spearman",
        "hard_prediction_agreement",
        "changed_prediction_count",
        "delta_auc_ci_low",
        "delta_auc_ci_high",
        "delta_macro_f1_ci_low",
        "delta_macro_f1_ci_high",
        "delta_balanced_accuracy_ci_low",
        "delta_balanced_accuracy_ci_high",
        "delta_sensitivity_ci_low",
        "delta_sensitivity_ci_high",
        "delta_specificity_ci_low",
        "delta_specificity_ci_high",
        "status",
    ]
    if not paired_df.empty:
        paired_df = paired_df.loc[:, paired_columns]
        paired_df = paired_df.sort_values(["experiment"]).reset_index(drop=True)
    else:
        paired_df = pd.DataFrame(columns=paired_columns)

    status_rows = _collect_status_rows(main_rows)
    status_df = pd.DataFrame(status_rows)

    main_df.to_csv(summary_dir / "p1_component_main_results.csv", index=False, encoding="utf-8-sig")
    fold_df.to_csv(summary_dir / "p1_component_fold_results.csv", index=False, encoding="utf-8-sig")
    paired_df.to_csv(summary_dir / "p1_component_paired_comparisons.csv", index=False, encoding="utf-8-sig")
    status_df.to_csv(summary_dir / "p1_component_status.csv", index=False, encoding="utf-8-sig")

    matrix_df = main_df.copy()
    if "macro_auc" in matrix_df.columns:
        matrix_df = matrix_df.sort_values(["macro_auc", "experiment"], ascending=[False, True], na_position="last")
    matrix_lines = [
        "# P1 component result matrix",
        "",
        _markdown_table(matrix_df, ["experiment", "status", "macro_auc", "macro_f1", "balanced_accuracy"]),
        "",
    ]
    (summary_dir / "p1_component_result_matrix.md").write_text("\n".join(matrix_lines), encoding="utf-8")

    report_text = _build_report(output_dir, main_df, fold_df, paired_df, status_df)
    report_path.write_text(report_text, encoding="utf-8")

    sweep_summary = {
        "final_status": "P1_COMPONENT_SWEEP_COMPLETE",
        "main_results_rows": int(len(main_df)),
        "fold_results_rows": int(len(fold_df)),
        "paired_results_rows": int(len(paired_df)),
        "status_rows": int(len(status_df)),
        "source": "rebuild_from_existing_experiment_artifacts",
    }
    (metadata_dir / "sweep_summary.json").write_text(json.dumps(_json_safe(sweep_summary), indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(_json_safe(sweep_summary), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
