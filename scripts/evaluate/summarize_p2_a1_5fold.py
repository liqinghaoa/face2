"""OOF integrity, frozen E0B pairing, and decision for P2-A1 full five-fold validation."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from datasets.control_patient_binary_dataset import map_three_class_to_binary
from metrics.binary_classification_metrics import compute_binary_metrics, flatten_metrics
from utils.experiment_utils import load_yaml


CORE_METRICS = ["macro_auc", "accuracy", "macro_precision", "macro_recall", "macro_f1", "balanced_accuracy", "patient_sensitivity", "control_specificity", "patient_ppv", "patient_npv"]


def class_metrics(confusion: np.ndarray) -> dict[str, float]:
    tn, fp, fn, tp = [int(value) for value in confusion.reshape(-1)]
    return {"patient_sensitivity": tp / (tp + fn) if tp + fn else 0.0, "control_specificity": tn / (tn + fp) if tn + fp else 0.0, "patient_ppv": tp / (tp + fp) if tp + fp else 0.0, "patient_npv": tn / (tn + fn) if tn + fn else 0.0}


def all_metrics(frame: pd.DataFrame) -> tuple[dict, np.ndarray]:
    metrics = compute_binary_metrics(frame["true_label"].to_numpy(int), frame[["prob_control", "prob_patient"]].to_numpy(float))
    result = {**flatten_metrics(metrics), **class_metrics(metrics["confusion_matrix"])}
    return result, metrics["confusion_matrix"]


def validate_oof(frame: pd.DataFrame, config: dict) -> pd.DataFrame:
    required = {"sample_id", "patient_group_id", "fold", "true_label", "original_nyha", "prob_control", "prob_patient", "pred_label", "correct"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"OOF lacks required columns: {sorted(missing)}")
    split = pd.read_csv(ROOT / config["data"]["split_csv"], dtype={"ID": "string", "patient_group_id": "string"}, encoding="utf-8-sig")
    master = pd.read_csv(ROOT / config["data"]["master_index"], dtype={"ID": "string", "patient_group_id": "string"}, encoding="utf-8-sig")
    expected_ids, observed_ids = set(split.ID.astype(str)), set(frame.sample_id.astype(str))
    if len(frame) != 500 or frame.sample_id.duplicated().any() or observed_ids != expected_ids:
        raise ValueError("OOF must contain exactly the 500 unique fixed-split sample IDs")
    if set(frame.fold.unique()) != {0, 1, 2, 3, 4} or frame.groupby("fold").size().to_dict() != {0: 100, 1: 100, 2: 100, 3: 100, 4: 100}:
        raise ValueError("OOF fold coverage must be exactly 100 examples in folds 0-4")
    lookup = split.set_index("ID")
    aligned = frame.assign(sample_id=frame.sample_id.astype(str)).merge(lookup[["patient_group_id", "fold", "label_3class"]], left_on="sample_id", right_index=True, suffixes=("", "_fixed"), validate="one_to_one")
    if not (aligned.patient_group_id.astype(str) == aligned.patient_group_id_fixed.astype(str)).all() or not (aligned.fold == aligned.fold_fixed).all():
        raise ValueError("OOF patient-group or fold assignment differs from immutable P0-A split")
    expected_labels = aligned.label_3class.map(map_three_class_to_binary).to_numpy(int)
    if not np.array_equal(expected_labels, aligned.true_label.to_numpy(int)):
        raise ValueError("OOF labels differ from fixed P0-A labels")
    master_ids = set(master.ID.astype(str))
    if expected_ids != master_ids:
        raise ValueError("P0-A master index and split IDs differ")
    probabilities = aligned[["prob_control", "prob_patient"]].to_numpy(float)
    if not np.isfinite(probabilities).all() or not np.allclose(probabilities.sum(axis=1), 1.0, atol=1e-5):
        raise ValueError("OOF probabilities are invalid")
    if not np.array_equal(aligned.pred_label.to_numpy(int), probabilities.argmax(axis=1)):
        raise ValueError("OOF predictions are not softmax argmax")
    return aligned.drop(columns=["patient_group_id_fixed", "fold_fixed", "label_3class"])


def fold0_reproduction(output_dir: Path) -> dict:
    frozen = ROOT / "experiments/500Data/P2_A1_Binary_RGB_DECAAux_Fold0_v1"
    current_metrics = json.loads((output_dir / "fold_0" / "metrics.json").read_text(encoding="utf-8"))
    frozen_metrics = json.loads((frozen / "fold_0" / "metrics.json").read_text(encoding="utf-8"))
    current_aux = json.loads((output_dir / "fold_0" / "aux_behavior.json").read_text(encoding="utf-8"))
    frozen_aux = json.loads((frozen / "p2_a1_aux_behavior.json").read_text(encoding="utf-8"))
    current_cm = pd.read_csv(output_dir / "fold_0" / "confusion_matrix.csv", index_col=0).to_numpy(int).tolist()
    frozen_cm = pd.read_csv(frozen / "fold_0" / "confusion_matrix.csv", index_col=0).to_numpy(int).tolist()
    metric_keys = ["best_epoch", "macro_auc", "accuracy", "macro_f1", "balanced_accuracy"]
    metric_delta = {key: float(current_metrics[key]) - float(frozen_metrics[key]) for key in metric_keys}
    aux_delta = {"best_epoch_alpha": float(current_aux["best_epoch_alpha"]) - float(frozen_aux["best_epoch_alpha"]), "best_epoch_norm_ratio": float(current_aux["best_epoch_norm_ratio"]) - float(frozen_aux["best_epoch_norm_ratio"])}
    matches = current_cm == frozen_cm and all(abs(value) <= 1e-6 for value in metric_delta.values()) and all(abs(value) <= 1e-6 for value in aux_delta.values())
    result = {"reference_dir": str(frozen), "matches_within_1e-6": matches, "metric_delta_new_minus_frozen": metric_delta, "aux_delta_new_minus_frozen": aux_delta, "current_confusion_matrix": current_cm, "frozen_confusion_matrix": frozen_cm}
    (output_dir / "fold_0_reproduction_check.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return result


def paired_comparison(output_dir: Path, a1: pd.DataFrame, config: dict) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    reference = ROOT / config["evaluation"]["reference_b0_5fold_dir"]
    b0 = pd.read_csv(reference / "oof_predictions.csv", dtype={"sample_id": "string", "patient_group_id": "string"})
    b0 = b0.rename(columns={"binary_label": "true_label", "pred_class": "pred_label", "prob_normal": "prob_control", "is_correct": "correct"})
    checks = [len(b0) == 500, set(b0.sample_id) == set(a1.sample_id), (b0.set_index("sample_id").loc[a1.sample_id, "patient_group_id"].astype(str).to_numpy() == a1.patient_group_id.astype(str).to_numpy()).all(), (b0.set_index("sample_id").loc[a1.sample_id, "true_label"].to_numpy(int) == a1.true_label.to_numpy(int)).all(), (b0.set_index("sample_id").loc[a1.sample_id, "fold"].to_numpy(int) == a1.fold.to_numpy(int)).all()]
    if not all(checks):
        raise ValueError("B0 five-fold OOF is not exactly paired to P2-A1 OOF")
    b0_metrics, _ = all_metrics(b0)
    a1_metrics, _ = all_metrics(a1)
    metric_rows = [{"metric": key, "b0": b0_metrics[key], "a1": a1_metrics[key], "a1_minus_b0": a1_metrics[key] - b0_metrics[key]} for key in CORE_METRICS]
    metrics_frame = pd.DataFrame(metric_rows)
    merged = b0.merge(a1, on="sample_id", suffixes=("_b0", "_a1"), validate="one_to_one")
    rows = []
    for row in merged.itertuples():
        b0_correct, a1_correct = bool(row.correct_b0), bool(row.correct_a1)
        transition = "both_correct" if b0_correct and a1_correct else "b0_wrong_a1_correct" if a1_correct else "b0_correct_a1_wrong" if b0_correct else "both_wrong"
        rows.append({"sample_id": row.sample_id, "patient_group_id": row.patient_group_id_a1, "fold": int(row.fold_a1), "true_label": int(row.true_label_a1), "original_nyha": int(row.original_nyha), "b0_prob_patient": float(row.prob_patient_b0), "a1_prob_patient": float(row.prob_patient_a1), "b0_pred": int(row.pred_label_b0), "a1_pred": int(row.pred_label_a1), "b0_correct": b0_correct, "a1_correct": a1_correct, "correctness_transition": transition, "probability_shift": float(row.prob_patient_a1 - row.prob_patient_b0)})
    cases = pd.DataFrame(rows)
    per_fold_rows = []
    for fold in range(5):
        b0_fold = b0[b0.fold == fold]
        a1_fold = a1[a1.fold == fold]
        bm, _ = all_metrics(b0_fold)
        am, _ = all_metrics(a1_fold)
        per_fold_rows.append({"fold": fold, "auc_delta": am["macro_auc"] - bm["macro_auc"], "macro_f1_delta": am["macro_f1"] - bm["macro_f1"], "balanced_accuracy_delta": am["balanced_accuracy"] - bm["balanced_accuracy"], "sensitivity_delta": am["patient_sensitivity"] - bm["patient_sensitivity"], "specificity_delta": am["control_specificity"] - bm["control_specificity"]})
    return metrics_frame, pd.DataFrame(per_fold_rows), {"b0_metrics": b0_metrics, "a1_metrics": a1_metrics, "cases": cases}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment-dir", required=True, type=Path)
    args = parser.parse_args()
    output_dir = args.experiment_dir if args.experiment_dir.is_absolute() else ROOT / args.experiment_dir
    config = load_yaml(output_dir / "config_snapshot.yaml")
    frames, fold_rows, aux_rows = [], [], []
    for fold in config["data"]["folds"]:
        fold_dir = output_dir / f"fold_{fold}"
        if not (fold_dir / "_SUCCESS.json").is_file():
            raise FileNotFoundError(f"fold {fold} is not explicitly successful")
        frame = pd.read_csv(fold_dir / "val_predictions.csv", dtype={"sample_id": "string", "patient_group_id": "string"})
        if len(frame) != 100 or set(frame.fold) != {fold}:
            raise ValueError(f"fold {fold} prediction contract failed")
        frames.append(frame)
        fold_rows.append(json.loads((fold_dir / "metrics.json").read_text(encoding="utf-8")))
        aux_rows.append(json.loads((fold_dir / "aux_behavior.json").read_text(encoding="utf-8")))
    oof = validate_oof(pd.concat(frames, ignore_index=True), config).sort_values(["fold", "sample_id"], kind="stable")
    oof.to_csv(output_dir / "oof_predictions.csv", index=False, encoding="utf-8-sig")
    pooled, matrix = all_metrics(oof)
    (output_dir / "oof_metrics.json").write_text(json.dumps(pooled, ensure_ascii=False, indent=2), encoding="utf-8")
    pd.DataFrame(matrix, index=["control", "patient"], columns=["control", "patient"]).to_csv(output_dir / "oof_confusion_matrix.csv", index_label="true\\pred", encoding="utf-8-sig")
    per_fold = pd.DataFrame(fold_rows).sort_values("fold")
    per_fold.to_csv(output_dir / "per_fold_metrics.csv", index=False, encoding="utf-8-sig")
    summary = {key: {"mean": float(per_fold[key].mean()), "std": float(per_fold[key].std(ddof=1)), "min": float(per_fold[key].min()), "max": float(per_fold[key].max())} for key in ["macro_auc", "macro_f1", "balanced_accuracy", "patient_sensitivity", "control_specificity"]}
    aux = pd.DataFrame(aux_rows).sort_values("fold")
    aux.to_csv(output_dir / "p2_a1_5fold_aux_behavior.csv", index=False, encoding="utf-8-sig")
    aux_summary = {key: {"mean": float(aux[key].mean()), "std": float(aux[key].std(ddof=1))} for key in ["best_epoch_alpha", "final_epoch_alpha", "best_epoch_norm_ratio", "final_epoch_norm_ratio"]}
    reproduction = fold0_reproduction(output_dir)
    comparison, comparison_per_fold, paired = paired_comparison(output_dir, oof, config)
    comparison.to_csv(output_dir / "p2_a1_5fold_vs_b0_metrics.csv", index=False, encoding="utf-8-sig")
    comparison_per_fold.to_csv(output_dir / "p2_a1_5fold_vs_b0_per_fold.csv", index=False, encoding="utf-8-sig")
    cases = paired["cases"]
    cases.to_csv(output_dir / "p2_a1_5fold_vs_b0_case_comparison.csv", index=False, encoding="utf-8-sig")
    details = {"b0_wrong_a1_correct": int((cases.correctness_transition == "b0_wrong_a1_correct").sum()), "b0_correct_a1_wrong": int((cases.correctness_transition == "b0_correct_a1_wrong").sum()), "net_corrections": int((cases.correctness_transition == "b0_wrong_a1_correct").sum() - (cases.correctness_transition == "b0_correct_a1_wrong").sum()), "b0_patient_fn_corrected": int(((cases.true_label == 1) & (cases.b0_pred == 0) & (cases.a1_pred == 1)).sum()), "b0_control_fp_corrected": int(((cases.true_label == 0) & (cases.b0_pred == 1) & (cases.a1_pred == 0)).sum()), "a1_patient_fn": int(((cases.true_label == 1) & (cases.a1_pred == 0)).sum()), "a1_control_fp": int(((cases.true_label == 0) & (cases.a1_pred == 1)).sum()), "a1_patient_predictions": int((cases.a1_pred == 1).sum()), "a1_control_predictions": int((cases.a1_pred == 0).sum())}
    nyha = cases.groupby("original_nyha").agg(n=("sample_id", "size"), mean_probability_shift=("probability_shift", "mean"), b0_wrong_a1_correct=("correctness_transition", lambda x: int((x == "b0_wrong_a1_correct").sum())), b0_correct_a1_wrong=("correctness_transition", lambda x: int((x == "b0_correct_a1_wrong").sum()))).reset_index()
    nyha.to_csv(output_dir / "p2_a1_5fold_vs_b0_nyha_summary.csv", index=False, encoding="utf-8-sig")
    b0m, a1m = paired["b0_metrics"], paired["a1_metrics"]
    improve_counts = {"auc_improved_folds": int((comparison_per_fold.auc_delta >= 0).sum()), "macro_f1_improved_folds": int((comparison_per_fold.macro_f1_delta > 0).sum()), "balanced_accuracy_improved_folds": int((comparison_per_fold.balanced_accuracy_delta > 0).sum()), "specificity_improved_folds": int((comparison_per_fold.specificity_delta > 0).sum())}
    sensitivity_drop, specificity_drop = b0m["patient_sensitivity"] - a1m["patient_sensitivity"], b0m["control_specificity"] - a1m["control_specificity"]
    collapsed = min(details["a1_patient_predictions"], details["a1_control_predictions"]) == 0
    go = a1m["macro_auc"] > b0m["macro_auc"] and (a1m["macro_f1"] > b0m["macro_f1"] or a1m["balanced_accuracy"] > b0m["balanced_accuracy"]) and improve_counts["auc_improved_folds"] >= 3 and sensitivity_drop <= .10 and specificity_drop <= .10 and details["b0_wrong_a1_correct"] >= details["b0_correct_a1_wrong"] and not collapsed
    stop = a1m["macro_auc"] <= b0m["macro_auc"] or (a1m["macro_f1"] <= b0m["macro_f1"] and a1m["balanced_accuracy"] <= b0m["balanced_accuracy"]) or details["b0_correct_a1_wrong"] > details["b0_wrong_a1_correct"] or improve_counts["auc_improved_folds"] <= 2 or collapsed
    decision = "GO_P3" if go else "STOP_A1" if stop else "HOLD_REVIEW"
    decision_payload = {"decision": decision, "go_conditions_met": go, "stop_conditions_met": stop, "pooled_sensitivity_drop": sensitivity_drop, "pooled_specificity_drop": specificity_drop, "fold_direction_counts": improve_counts, "pair_details": details, "single_class_prediction_collapse": collapsed, "auxiliary_behavior_summary": aux_summary}
    (output_dir / "p2_a1_5fold_decision.json").write_text(json.dumps(decision_payload, ensure_ascii=False, indent=2), encoding="utf-8")
    fivefold = {"pooled_oof_metrics": pooled, "fold_metric_summary": summary, "auxiliary_behavior_summary": aux_summary, "fold_0_reproduction": reproduction, "decision": decision_payload}
    (output_dir / "fivefold_summary.json").write_text(json.dumps(fivefold, ensure_ascii=False, indent=2), encoding="utf-8")
    (output_dir / "fivefold_summary.md").write_text(f"# P2-A1 完整五折 OOF 汇总\n\n主结果为 500 例 pooled OOF：Macro-AUC={pooled['macro_auc']:.6f}，Macro-F1={pooled['macro_f1']:.6f}，Balanced Accuracy={pooled['balanced_accuracy']:.6f}。\n\n最终决策：**{decision}**。\n", encoding="utf-8")
    print(f"P2_A1_5FOLD_SUMMARY_DIR={output_dir}")


if __name__ == "__main__":
    main()
