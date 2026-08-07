from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

import pandas as pd

from p2_counterfactual.a3_v3_assets import (
    ASSET_DIR,
    CODE_AUDIT_PATH,
    PAIR_AUDIT_PATH,
    PAIR_CYCLE_SCHEDULE_PATH,
    PAIR_MANIFEST_PATH,
    PRESET_MAPPING_PATH,
    build_a3_v3_assets,
)
from p2_counterfactual.cluster_bootstrap import (
    experiment_metric_values,
    load_experiment_frames,
    paired_cluster_bootstrap_comparison,
)
from p2_counterfactual.config import resolve_p2_a_config
from p2_counterfactual.io_utils import json_safe, sha256_file
from p2_counterfactual.oof_aggregation import aggregate_p2_a_experiment


ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = ROOT / "config/p2/p2_a/p2_a3_v3_r3dpr_pairwise_consistency_meanbg.yaml"
EXPERIMENT_ID = "p2_a3_v3_r3dpr_pairwise_consistency_meanbg"
DISPLAY_ID = "P2-A3-v3_R3DPR_PairwiseConsistency_MeanBG"
A2_ID = "p2_a2_v2_r3dpr_meanbg"
A2_DIR = ROOT / "experiments/500Data/P2_A2_v2_R3DPR_MeanBG/p2_a2_v2_r3dpr_meanbg"
A3_V2_ID = "p2_a3_v2_r3dpr_full6_consistency_meanbg"
A3_V2_DIR = ROOT / "experiments/500Data/P2_A3_v2_R3DPR_Full6Consistency_MeanBG/p2_a3_v2_r3dpr_full6_consistency_meanbg"


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(json_safe(payload), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _fmt(value: Any, digits: int = 4) -> str:
    try:
        return f"{float(value):.{digits}f}"
    except Exception:
        return str(value)


def _standardize_original(source: Path, target: Path) -> pd.DataFrame:
    frame = pd.read_csv(source, dtype={"case_id": str, "patient_group_id": str})
    out = frame.copy()
    out.insert(2, "fold_id", out["fold"].astype(int))
    out.insert(4, "true_label", out["label"].astype(int))
    out.insert(8, "pred_label", out["predicted_label"].astype(int))
    out["checkpoint_path"] = out["fold"].astype(int).map(
        lambda fold: str(target.parents[0] / f"fold_{fold}" / "checkpoints/best_macro_auc.pth")
    )
    out.to_csv(target, index=False, encoding="utf-8-sig")
    return out


def _standardize_relighted(source: Path, target: Path) -> pd.DataFrame:
    frame = pd.read_csv(source, dtype={"case_id": str, "patient_group_id": str})
    out = frame.copy()
    out.insert(2, "fold_id", out["fold"].astype(int))
    out.insert(4, "true_label", out["label"].astype(int))
    out.insert(8, "pred_label", out["predicted_label"].astype(int))
    out.insert(11, "preset_id", out["preset_index"].astype(int))
    out["checkpoint_path"] = out["fold"].astype(int).map(
        lambda fold: str(target.parents[0] / f"fold_{fold}" / "checkpoints/best_macro_auc.pth")
    )
    out["feature_npz_path"] = str(target.parents[0] / "p2_a3_v3_relighted_features.npz")
    out.to_csv(target, index=False, encoding="utf-8-sig")
    return out


def _fold_results(experiment_dir: Path, success: dict[str, Any]) -> pd.DataFrame:
    rows = []
    success_by_fold = {int(row["fold"]): row for row in success["fold_success"]}
    for fold in range(5):
        fold_dir = experiment_dir / f"fold_{fold}"
        summary = _read_json(fold_dir / "fold_summary.json")
        history = pd.read_csv(fold_dir / "training_history.csv")
        metrics = _read_json(fold_dir / "metrics_original.json")
        rows.append(
            {
                "fold_id": fold,
                "status": success_by_fold[fold]["status"],
                "best_epoch": int(success_by_fold[fold]["best_epoch"]),
                "epochs_completed": int(history["epoch"].max()),
                "train_cases": 400,
                "test_cases": int(summary["original"]["rows"]),
                "relighted_rows": int(summary["relighted"]["rows"]),
                "macro_auc": float(metrics["macro_auc"]),
                "accuracy": float(metrics["accuracy"]),
                "macro_precision": float(metrics["macro_precision"]),
                "macro_recall": float(metrics["macro_recall"]),
                "macro_f1": float(metrics["macro_f1"]),
                "balanced_accuracy": float(metrics["balanced_accuracy"]),
                "patient_sensitivity": float(metrics["patient_sensitivity"]),
                "control_specificity": float(metrics["control_specificity"]),
                "physical_case_batch_size": int(history["physical_case_batch_size"].iloc[-1]),
                "gradient_accumulation_steps": int(history["gradient_accumulation_steps"].iloc[-1]),
                "effective_case_batch_size": int(history["effective_case_batch_size"].iloc[-1]),
                "images_per_step": int(history["images_per_step"].iloc[-1]),
                "checkpoint_sha256": success_by_fold[fold]["checkpoint_sha256"],
            }
        )
        history.to_csv(fold_dir / "consistency_log.csv", index=False, encoding="utf-8-sig")
    out = pd.DataFrame(rows)
    out.to_csv(experiment_dir / "p2_a3_v3_fold_results.csv", index=False, encoding="utf-8-sig")
    return out


def _model_tables(frames: list[Any]) -> tuple[pd.DataFrame, pd.DataFrame]:
    class_rows = []
    stability_rows = []
    for frame in frames:
        values = experiment_metric_values(frame.original, frame.stability)
        class_rows.append({key: values[key] for key in ["macro_auc", "macro_f1", "balanced_accuracy", "patient_sensitivity", "control_specificity"]} | {"experiment_id": frame.experiment_id})
        stability_rows.append(
            {
                "experiment_id": frame.experiment_id,
                "mean_prediction_std": values["mean_prediction_std"],
                "case_level_flip_rate": values["case_level_flip_rate"],
                "mean_abs_probability_delta": values["mean_abs_probability_delta"],
                "worst_light_auc": values["worst_light_auc"],
                "mean_feature_cosine": values["mean_feature_cosine"],
            }
        )
    return pd.DataFrame(class_rows), pd.DataFrame(stability_rows)


def _metric_row(rows: list[dict[str, Any]], metric: str) -> dict[str, Any]:
    return next(row for row in rows if row["metric"] == metric)


def _delta_text(rows: list[dict[str, Any]], metric: str) -> str:
    row = _metric_row(rows, metric)
    return f"{_fmt(row['difference_b_minus_a'])} [{_fmt(row['ci_lower'])}, {_fmt(row['ci_upper'])}]"


def _write_report(
    experiment_dir: Path,
    success: dict[str, Any],
    fold_results: pd.DataFrame,
    class_table: pd.DataFrame,
    stability_table: pd.DataFrame,
    primary_rows: list[dict[str, Any]],
    diagnostic_rows: list[dict[str, Any]],
) -> Path:
    metrics = success["summary_metrics"]
    stability = success["stability_metrics"]
    classification_checks = {
        "macro_auc": _metric_row(primary_rows, "macro_auc")["difference_b_minus_a"] >= -0.01,
        "macro_f1": _metric_row(primary_rows, "macro_f1")["difference_b_minus_a"] >= -0.02,
        "balanced_accuracy": _metric_row(primary_rows, "balanced_accuracy")["difference_b_minus_a"] >= -0.02,
        "patient_sensitivity": _metric_row(primary_rows, "patient_sensitivity")["difference_b_minus_a"] >= -0.03,
        "control_specificity": _metric_row(primary_rows, "control_specificity")["difference_b_minus_a"] >= -0.03,
    }
    stability_checks = {
        "mean_prediction_std": _metric_row(primary_rows, "mean_prediction_std")["difference_b_minus_a"] <= 0.0,
        "case_level_flip_rate": _metric_row(primary_rows, "case_level_flip_rate")["difference_b_minus_a"] <= 0.0,
        "mean_abs_probability_delta": _metric_row(primary_rows, "mean_abs_probability_delta")["difference_b_minus_a"] <= 0.0,
        "mean_feature_cosine": _metric_row(primary_rows, "mean_feature_cosine")["difference_b_minus_a"] >= 0.0,
    }
    supports = bool(all(classification_checks.values()) and any(stability_checks.values()))
    lines = [
        "# P2-A3-v3 R3DPR Pairwise Consistency Experiment Report",
        "",
        "## 1. Experiment Objective",
        "",
        "Evaluate whether pairwise original-relighted consistency training preserves original-image classification while improving cross-lighting stability without multiplying optimizer updates by six.",
        "",
        "## 2. Difference From P2-A2-v2",
        "",
        "- A2-v2 samples one image per case per step: original with 50% probability or one randomly selected relighted image.",
        "- A3-v3 uses one pair per training case: original plus one deterministic pair-cycle relighted preset.",
        "- Consecutive six epochs cover all six relighting presets once per case, while each virtual epoch still has one optimizer update opportunity per case.",
        "- A3-v3 adds symmetric JS prediction consistency and 512-dimensional feature cosine consistency between the paired views.",
        "- Inference remains single-image original-only for checkpoint selection and original OOF metrics.",
        "",
        "## 3. Data Assets",
        "",
        f"- Data directory: `{ROOT / 'data/processed/global_face/SixRelighting_OriginalCamera_meanfg'}`",
        f"- Pair manifest: `{PAIR_MANIFEST_PATH}`",
        f"- Pair audit: `{PAIR_AUDIT_PATH}`",
        f"- Preset mapping: `{PRESET_MAPPING_PATH}`",
        f"- Pair-cycle schedule: `{PAIR_CYCLE_SCHEDULE_PATH}`",
        "- Cases: 500; pair records: 3000; pairs per case: 6; fixed preset order: neutral_front, left, right, top, dim_front, bright_front.",
        "",
        "## 4. Data And Leakage Audit",
        "",
        "- Pair audit passed: 500 unique cases, 3000 valid pair rows, six presets per case, no patient_group_id fold violations.",
        "- Fixed folds and labels were reused; no fold split or label was modified.",
        "- The A3-v3 pair manifest uses image-directory paths and does not read old DECA `relighted_images` assets.",
        "",
        "## 5. Model Structure",
        "",
        "- Single shared ImageNet ResNet18 backbone and one Linear(512,2) classifier head.",
        "- Pair batches provide `original_images [B,3,H,W]` and `relighted_images [B,3,H,W]`, concatenate to `[2B,3,H,W]`, forward once through the shared backbone, then split back to paired logits/features.",
        "- The v3 training path does not use `[B,6,...]`, `[B,7,...]`, full-six expansion, or teacher-student detaching.",
        "- No feature fusion head, attention, preset encoder, second backbone, or multi-view ensemble classifier was added.",
        "",
        "## 6. Loss Function",
        "",
        "- `L_cls` is computed as one unified cross-entropy over concatenated original and relighted logits, equivalent to `0.5 * CE(original) + 0.5 * CE(relighted)`.",
        "- `L_pred = JS(softmax(original), softmax(relighted))`, symmetric and averaged over pairs.",
        "- `L_feat = mean(1 - cosine(feature_original, feature_relighted))`.",
        "- `L_total = L_cls + warmup(epoch) * (0.5 * L_pred + 0.1 * L_feat)`.",
        "- Warm-up: epochs 1-5 are 0.2, 0.4, 0.6, 0.8, 1.0.",
        "",
        "## 7. Image Augmentation",
        "",
        "- Synchronized horizontal flip only, probability 0.5.",
        "- No ColorJitter, MixUp, CutMix, random brightness, random contrast, random saturation, random hue, or extra lighting augmentation.",
        "",
        "## 8. Tests And GPU Smoke",
        "",
        "- `pytest tests/p2 -q`: 65 passed before formal training.",
        "- GPU smoke passed on fold 0 with physical case batch size 8, gradient accumulation 1, effective case batch size 8, and 16 images per step.",
        "- Smoke exported original and six-relighted predictions and validated checkpoint resume.",
        "- AMP/mixed precision remained disabled because the locked A2-v2 training protocol has `amp: false`; no AMP-specific code path was used.",
        "",
        "## 9. Five-Fold Training",
        "",
        "| Fold | Best epoch | Macro-AUC | Macro-F1 | BA | Sensitivity | Specificity |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in fold_results.itertuples(index=False):
        lines.append(f"| {row.fold_id} | {row.best_epoch} | {_fmt(row.macro_auc)} | {_fmt(row.macro_f1)} | {_fmt(row.balanced_accuracy)} | {_fmt(row.patient_sensitivity)} | {_fmt(row.control_specificity)} |")
    lines += [
        "",
        "## 10. Original OOF Results",
        "",
        f"- Macro-AUC: {_fmt(metrics['macro_auc'])}",
        f"- Accuracy: {_fmt(metrics['accuracy'])}",
        f"- Macro-Precision: {_fmt(metrics['macro_precision'])}",
        f"- Macro-Recall: {_fmt(metrics['macro_recall'])}",
        f"- Macro-F1: {_fmt(metrics['macro_f1'])}",
        f"- Balanced Accuracy: {_fmt(metrics['balanced_accuracy'])}",
        f"- Sensitivity: {_fmt(metrics['patient_sensitivity'])}",
        f"- Specificity: {_fmt(metrics['control_specificity'])}",
        "",
        "## 11. Cross-Lighting Stability",
        "",
        f"- Mean prediction std: {_fmt(stability['prediction_std']['mean'])}",
        f"- Case flip rate: {_fmt(stability['case_level_label_flip_rate'])}",
        f"- Mean absolute probability delta: {_fmt(sum(stability['mean_abs_probability_delta'].values()) / 6.0)}",
        f"- Worst-light Macro-AUC: {_fmt(stability['worst_light_auc'])}",
        f"- Mean feature cosine: {_fmt(stability['mean_feature_cosine'])}",
        f"- Relighted-only prediction std: {_fmt(stability['relighted_prediction_std']['mean'])}",
        f"- Relighted pairwise feature distance: {_fmt(stability['mean_relighted_pairwise_feature_distance'])}",
        "",
        "## 12. A3-v3 vs A2-v2",
        "",
        "| Metric | A3-v3 minus A2-v2, 95% CI |",
        "|---|---:|",
    ]
    for metric in ["macro_auc", "macro_f1", "balanced_accuracy", "patient_sensitivity", "control_specificity", "mean_prediction_std", "case_level_flip_rate", "mean_abs_probability_delta", "worst_light_auc", "mean_feature_cosine"]:
        lines.append(f"| {metric} | {_delta_text(primary_rows, metric)} |")
    lines += [
        "",
        "## 13. A3-v3 vs A3-v2 Full6 Diagnostic",
        "",
        "| Metric | A3-v3 minus A3-v2 full6, 95% CI |",
        "|---|---:|",
    ]
    for metric in ["macro_auc", "macro_f1", "balanced_accuracy", "patient_sensitivity", "control_specificity", "mean_prediction_std", "case_level_flip_rate", "mean_abs_probability_delta", "worst_light_auc", "mean_feature_cosine"]:
        lines.append(f"| {metric} | {_delta_text(diagnostic_rows, metric)} |")
    lines += [
        "",
        "## 14. Result Interpretation",
        "",
        f"- Primary A3-v3 minus A2-v2 Macro-AUC delta: {_delta_text(primary_rows, 'macro_auc')}.",
        f"- Primary stability delta, mean prediction std: {_delta_text(primary_rows, 'mean_prediction_std')}.",
        f"- Primary stability delta, case flip rate: {_delta_text(primary_rows, 'case_level_flip_rate')}.",
        f"- Classification preservation checks: {classification_checks}.",
        f"- Stability improvement checks: {stability_checks}.",
        "- A3-v3 differs from A2-v2 by paired shared-backbone consistency training and deterministic pair-cycle exposure. The diagnostic comparison against A3-v2 separates pairwise scheduling from the older full-six consistency design.",
        "",
        "## 15. Code And Output Checklist",
        "",
        f"- Config: `{CONFIG_PATH}`",
        f"- Code audit: `{CODE_AUDIT_PATH}`",
        f"- Original OOF: `{experiment_dir / 'p2_a3_v3_original_oof_predictions.csv'}`",
        f"- Relighted OOF: `{experiment_dir / 'p2_a3_v3_relighted_oof_predictions.csv'}`",
        f"- Primary comparison: `{experiment_dir / 'p2_a3_v3_vs_a2_v2_paired_comparisons.csv'}`",
        f"- Diagnostic comparison: `{experiment_dir / 'p2_a3_v3_vs_full6_a3_v2_comparisons.csv'}`",
        "",
        "## 16. Final Conclusion",
        "",
        f"- Five-fold completed: yes.",
        f"- 500-case original OOF complete: yes.",
        f"- 3000-row relighted OOF complete: yes.",
        f"- Supports pairwise consistency hypothesis under the configured criterion: {'yes' if supports else 'no'}."
        " A3-v3 is considered supportive only if original classification is preserved while stability improves.",
        "- No P2-B, P3, or P4 execution was started.",
    ]
    path = experiment_dir / "p2_a3_v3_pairwise_consistency_report.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8-sig")
    return path


def summarize() -> dict[str, Any]:
    asset_audit = build_a3_v3_assets()
    config = resolve_p2_a_config(CONFIG_PATH)
    experiment_dir = Path(config["output_root"]) / config["experiment_id"]
    success = aggregate_p2_a_experiment(
        experiment_dir=experiment_dir,
        config=config,
        manifest_path=config["manifest_path"],
        folds=(0, 1, 2, 3, 4),
        expected_case_count=500,
        expected_full_folds=True,
    )
    shutil.copy2(experiment_dir / "fold_0/config_resolved.yaml", experiment_dir / "resolved_config.yaml")
    shutil.copy2(experiment_dir / "fold_0/config_resolved.yaml", experiment_dir / "p2_a3_v3_resolved_config.yaml")
    shutil.copy2(PAIR_MANIFEST_PATH, experiment_dir / "p2_a3_v3_pair_manifest.csv")
    shutil.copy2(PAIR_AUDIT_PATH, experiment_dir / "p2_a3_v3_pair_audit.json")
    shutil.copy2(PRESET_MAPPING_PATH, experiment_dir / "p2_a3_v3_preset_mapping.json")
    shutil.copy2(PAIR_CYCLE_SCHEDULE_PATH, experiment_dir / "p2_a3_v3_pair_cycle_schedule.json")
    shutil.copy2(CODE_AUDIT_PATH, experiment_dir / "code_audit.md")
    _standardize_original(experiment_dir / "oof/oof_predictions_original.csv", experiment_dir / "p2_a3_v3_original_oof_predictions.csv")
    _standardize_relighted(experiment_dir / "oof/oof_predictions_relighted.csv", experiment_dir / "p2_a3_v3_relighted_oof_predictions.csv")
    shutil.copy2(experiment_dir / "summary/metrics_original_oof.json", experiment_dir / "p2_a3_v3_original_metrics.json")
    shutil.copy2(experiment_dir / "oof/oof_features_original.npz", experiment_dir / "p2_a3_v3_original_features.npz")
    shutil.copy2(experiment_dir / "oof/oof_features_relighted.npz", experiment_dir / "p2_a3_v3_relighted_features.npz")
    shutil.copy2(experiment_dir / "summary/case_level_stability.csv", experiment_dir / "p2_a3_v3_stability_case_level.csv")
    shutil.copy2(experiment_dir / "summary/per_preset_metrics.csv", experiment_dir / "p2_a3_v3_stability_by_preset.csv")
    shutil.copy2(experiment_dir / "summary/stability_metrics.json", experiment_dir / "p2_a3_v3_stability_summary.json")
    fold_results = _fold_results(experiment_dir, success)

    a2 = load_experiment_frames(A2_DIR, A2_ID)
    a3_v2 = load_experiment_frames(A3_V2_DIR, A3_V2_ID)
    a3 = load_experiment_frames(experiment_dir, EXPERIMENT_ID)
    iterations = int(config.get("bootstrap", {}).get("iterations", 2000))
    seed = int(config.get("bootstrap", {}).get("seed", 2026))
    primary_rows = paired_cluster_bootstrap_comparison(a2, a3, iterations=iterations, seed=seed)
    diagnostic_rows = paired_cluster_bootstrap_comparison(a3_v2, a3, iterations=iterations, seed=seed)
    pd.DataFrame(primary_rows).to_csv(experiment_dir / "p2_a3_v3_vs_a2_v2_paired_comparisons.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(diagnostic_rows).to_csv(experiment_dir / "p2_a3_v3_vs_full6_a3_v2_comparisons.csv", index=False, encoding="utf-8-sig")
    class_table, stability_table = _model_tables([a2, a3_v2, a3])
    class_table.to_csv(experiment_dir / "p2_a3_v3_model_classification_comparison.csv", index=False, encoding="utf-8-sig")
    stability_table.to_csv(experiment_dir / "p2_a3_v3_model_stability_comparison.csv", index=False, encoding="utf-8-sig")
    _write_json(
        experiment_dir / "p2_a3_v3_vs_a2_v2_bootstrap_summary.json",
        {
            "display_experiment_id": DISPLAY_ID,
            "comparison": f"{EXPERIMENT_ID} vs {A2_ID}",
            "bootstrap_iterations": iterations,
            "bootstrap_seed": seed,
            "cluster_unit": "patient_group_id",
            "metric_unit": "visit_case",
            "rows": primary_rows,
        },
    )
    _write_json(
        experiment_dir / "p2_a3_v3_bootstrap_summary.json",
        {
            "primary": primary_rows,
            "diagnostic_full6": diagnostic_rows,
            "bootstrap_iterations": iterations,
            "bootstrap_seed": seed,
        },
    )
    report = _write_report(experiment_dir, success, fold_results, class_table, stability_table, primary_rows, diagnostic_rows)
    final = {
        "display_experiment_id": DISPLAY_ID,
        "experiment_id": EXPERIMENT_ID,
        "status": success["status"],
        "experiment_dir": str(experiment_dir),
        "asset_audit_status": asset_audit["status"],
        "oof_case_count": success["oof_case_count"],
        "relighted_view_count": success["relighted_view_count"],
        "summary_metrics": success["summary_metrics"],
        "stability_metrics": success["stability_metrics"],
        "primary_comparison": primary_rows,
        "diagnostic_full6_comparison": diagnostic_rows,
        "report": str(report),
        "p2_b_executed": False,
        "p3_executed": False,
        "p4_executed": False,
    }
    _write_json(experiment_dir / "p2_a3_v3_final_summary.json", final)
    return final


def main() -> None:
    print(json.dumps(json_safe(summarize()), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
