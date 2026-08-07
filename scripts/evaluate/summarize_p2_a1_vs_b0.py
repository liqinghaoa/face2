"""Paired fixed-fold P2-B0/P2-A1 comparison and P2-A1 final metric enrichment."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
import sys
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from metrics.binary_classification_metrics import compute_binary_metrics, flatten_metrics


def class_metrics(confusion) -> dict[str, float]:
    tn, fp, fn, tp = [int(value) for value in confusion.reshape(-1)]
    return {
        "patient_sensitivity": tp / (tp + fn) if tp + fn else 0.0,
        "control_specificity": tn / (tn + fp) if tn + fp else 0.0,
        "patient_ppv": tp / (tp + fp) if tp + fp else 0.0,
        "patient_npv": tn / (tn + fn) if tn + fn else 0.0,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment-dir", required=True, type=Path)
    args = parser.parse_args()
    output_dir = args.experiment_dir if args.experiment_dir.is_absolute() else ROOT / args.experiment_dir
    b0_dir = ROOT / "experiments/500Data/P2_B0_BinaryRGB_Fold0_v1"
    b0 = pd.read_csv(b0_dir / "p2_b0_predictions.csv", dtype={"sample_id": "string"})
    a1 = pd.read_csv(output_dir / "fold_0/val_predictions.csv", dtype={"sample_id": "string"})
    if len(b0) != 100 or len(a1) != 100 or set(b0.sample_id) != set(a1.sample_id):
        raise ValueError("B0/A1 must contain exactly the same 100 validation IDs")

    raw_a1_metrics = compute_binary_metrics(a1.binary_label.to_numpy(), a1[["prob_normal", "prob_patient"]].to_numpy())
    a1_metrics = {"best_epoch": json.loads((output_dir / "fold_0/metrics.json").read_text(encoding="utf-8"))["best_epoch"], **flatten_metrics(raw_a1_metrics), **class_metrics(raw_a1_metrics["confusion_matrix"])}
    (output_dir / "fold_0/metrics.json").write_text(json.dumps(a1_metrics, ensure_ascii=False, indent=2), encoding="utf-8")

    merged = b0.merge(a1, on="sample_id", suffixes=("_b0", "_a1"), validate="one_to_one")
    rows = []
    for row in merged.itertuples():
        b0_correct = row.pred_class_b0 == row.binary_label_b0
        a1_correct = row.pred_class_a1 == row.binary_label_a1
        transition = "both_correct" if b0_correct and a1_correct else "b0_wrong_a1_correct" if a1_correct else "b0_correct_a1_wrong" if b0_correct else "both_wrong"
        rows.append({
            "sample_id": row.sample_id, "patient_group_id": row.patient_group_id_a1, "true_label": row.binary_label_a1,
            "original_nyha": row.original_label_a1, "b0_prob_patient": row.prob_patient_b0, "a1_prob_patient": row.prob_patient_a1,
            "b0_pred": row.pred_class_b0, "a1_pred": row.pred_class_a1, "b0_correct": b0_correct,
            "a1_correct": a1_correct, "correctness_transition": transition,
        })
    cases = pd.DataFrame(rows)
    cases.to_csv(output_dir / "p2_a1_vs_b0_case_comparison.csv", index=False, encoding="utf-8-sig")

    b0_metrics = json.loads((b0_dir / "p2_b0_metrics.json").read_text(encoding="utf-8"))
    pd.DataFrame([{"model": "B0", **b0_metrics}, {"model": "A1", **a1_metrics}]).to_csv(output_dir / "p2_a1_vs_b0_metrics.csv", index=False)
    details = {
        "b0_wrong_a1_correct": int((cases.correctness_transition == "b0_wrong_a1_correct").sum()),
        "b0_correct_a1_wrong": int((cases.correctness_transition == "b0_correct_a1_wrong").sum()),
        "b0_patient_fn_corrected": int(((cases.true_label == 1) & (cases.b0_pred == 0) & (cases.a1_pred == 1)).sum()),
        "b0_control_fp_corrected": int(((cases.true_label == 0) & (cases.b0_pred == 1) & (cases.a1_pred == 0)).sum()),
        "a1_patient_fn": int(((cases.true_label == 1) & (cases.a1_pred == 0)).sum()),
        "a1_control_fp": int(((cases.true_label == 0) & (cases.a1_pred == 1)).sum()),
        "a1_patient_predictions": int((cases.a1_pred == 1).sum()), "a1_control_predictions": int((cases.a1_pred == 0).sum()),
    }
    mode_a = a1_metrics["macro_auc"] > b0_metrics["macro_auc"] and (a1_metrics["macro_f1"] > b0_metrics["macro_f1"] or a1_metrics["balanced_accuracy"] > b0_metrics["balanced_accuracy"]) and a1_metrics["patient_sensitivity"] >= b0_metrics["patient_sensitivity"] and a1_metrics["control_specificity"] >= b0_metrics["control_specificity"] and details["b0_wrong_a1_correct"] >= details["b0_correct_a1_wrong"]
    mode_b = a1_metrics["macro_auc"] >= b0_metrics["macro_auc"] - 0.01 and (a1_metrics["macro_f1"] > b0_metrics["macro_f1"] or a1_metrics["balanced_accuracy"] > b0_metrics["balanced_accuracy"] or abs(a1_metrics["patient_sensitivity"] - a1_metrics["control_specificity"]) < abs(b0_metrics["patient_sensitivity"] - b0_metrics["control_specificity"]) or details["b0_wrong_a1_correct"] > details["b0_correct_a1_wrong"])
    specificity_drop = b0_metrics["control_specificity"] - a1_metrics["control_specificity"]
    sensitivity_drop = b0_metrics["patient_sensitivity"] - a1_metrics["patient_sensitivity"]
    vetoes = {
        "macro_auc_drop_over_0_02": a1_metrics["macro_auc"] < b0_metrics["macro_auc"] - 0.02,
        "macro_f1_and_balanced_accuracy_both_drop": a1_metrics["macro_f1"] < b0_metrics["macro_f1"] and a1_metrics["balanced_accuracy"] < b0_metrics["balanced_accuracy"],
        "control_specificity_drop_over_0_05": specificity_drop > 0.05,
        "patient_sensitivity_drop_over_0_05": sensitivity_drop > 0.05,
        "new_errors_exceed_corrections": details["b0_correct_a1_wrong"] > details["b0_wrong_a1_correct"],
    }
    recommendation = (mode_a or mode_b) and not any(vetoes.values())
    decision = {
        "recommend_p3": recommendation, "mode_a_met": mode_a, "mode_b_met": mode_b,
        "vetoes": vetoes, "specificity_change_a1_minus_b0": -specificity_drop,
        "sensitivity_change_a1_minus_b0": -sensitivity_drop, "details": details,
        "recommendation": "proceed_to_p3" if recommendation else "freeze_a1_do_not_start_formal_five_fold",
    }
    (output_dir / "p2_a1_decision.json").write_text(json.dumps(decision, ensure_ascii=False, indent=2), encoding="utf-8")
    (output_dir / "p2_a1_summary.md").write_text(
        "# P2-A1 vs P2-B0\n\n"
        f"建议进入 P3：{recommendation}\n\n"
        f"Mode A：{mode_a}；Mode B：{mode_b}；Control specificity 变化：{-specificity_drop:.4f}\n\n"
        f"逐病例变化：{details}\n",
        encoding="utf-8",
    )
    print(f"P2_A1_SUMMARY_DIR={output_dir}")


if __name__ == "__main__":
    main()
