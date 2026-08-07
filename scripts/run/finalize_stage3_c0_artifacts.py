"""Finalize non-training Stage3-C0 figures, calibration and report inventory."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

ROOT = Path(__file__).resolve().parents[2] / "experiments" / "lighting_confounding" / "Stage3_C0_SH093_224_SameCameraSignal_and_DeviceDomain_Audit_v1"


def read(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, dtype={"sample_id": str, "patient_group_id": str}, encoding="utf-8-sig")


def dump(value: object, path: Path) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str), encoding="utf-8")


def calibration(probability: pd.Series, target: pd.Series) -> tuple[float, float]:
    p = np.clip(probability.to_numpy(float), 1e-6, 1 - 1e-6)
    logit = np.log(p / (1 - p)).reshape(-1, 1)
    fit = LogisticRegression(C=1e6, solver="lbfgs", max_iter=2000).fit(logit, target.to_numpy(int))
    return float(fit.intercept_[0]), float(fit.coef_[0, 0])


def calibration_plot(oof: pd.DataFrame, path: Path, title: str) -> None:
    ranked = pd.qcut(oof["probability"].rank(method="first"), q=5, labels=False)
    x = pd.DataFrame({"bin": ranked, "p": oof["probability"], "y": oof["task_label"]}).groupby("bin").mean()
    fig, ax = plt.subplots(figsize=(5, 4)); ax.plot([0, 1], [0, 1], "--", color="gray"); ax.plot(x.p, x.y, marker="o"); ax.set(xlabel="Mean predicted probability", ylabel="Observed positive fraction", title=title); fig.tight_layout(); fig.savefig(path, dpi=180); plt.close(fig)


def main() -> None:
    a = read(ROOT / "c0a_disease/oof/c0a_xiaomi_sh093_oof_predictions.csv")
    b = read(ROOT / "c0b_camera/oof/c0b_patient_camera_oof_predictions.csv")
    ai, aslope = calibration(a.probability, a.task_label); bi, bslope = calibration(b.probability, b.task_label)
    for task, oof, intercept, slope, stem in (("c0a_disease", a, ai, aslope, "c0a"), ("c0b_camera", b, bi, bslope, "c0b_camera")):
        metrics_path = ROOT / task / "metrics" / ("c0a_oof_metrics.json" if task == "c0a_disease" else "c0b_oof_metrics.json")
        metrics = json.loads(metrics_path.read_text(encoding="utf-8")); metrics.update({"calibration_intercept": intercept, "calibration_slope": slope}); dump(metrics, metrics_path)
        calibration_plot(oof, ROOT / task / "figures" / f"{stem}_calibration_curve.png", f"{stem} calibration")
        histories = []
        for fold in range(5):
            h = read(ROOT / task / f"fold_{fold}/history/training_history.csv")
            h["fold"] = fold
            histories.append(h)
        hist = pd.concat(histories, ignore_index=True)
        fig, ax = plt.subplots(figsize=(7, 4))
        for _, h in hist.groupby("fold"):
            ax.plot(h.epoch, h.train_loss, color="#4c78a8", alpha=.35)
            ax.plot(h.epoch, h.inner_val_loss, color="#e45756", alpha=.35)
        ax.plot([], [], color="#4c78a8", label="train"); ax.plot([], [], color="#e45756", label="inner validation"); ax.set(xlabel="Epoch", ylabel="BCE loss", title=f"{stem} train/validation loss"); ax.legend(); fig.tight_layout(); fig.savefig(ROOT / task / "figures" / f"{stem}_training_validation_loss.png", dpi=180); plt.close(fig)
    decision_path = ROOT / "joint/reports/stage3_c0_final_decision.json"; decision = json.loads(decision_path.read_text(encoding="utf-8")); decision["c0a_metrics"].update({"calibration_intercept": ai, "calibration_slope": aslope}); decision["c0b_metrics"].update({"calibration_intercept": bi, "calibration_slope": bslope}); dump(decision, decision_path)
    for task, metrics_name, summary_name in (("c0a_disease", "c0a_oof_metrics.json", "c0a_machine_summary.json"), ("c0b_camera", "c0b_oof_metrics.json", "c0b_machine_summary.json")):
        summary_path = ROOT / task / "reports" / summary_name
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        summary["oof_metrics"] = json.loads((ROOT / task / "metrics" / metrics_name).read_text(encoding="utf-8"))
        dump(summary, summary_path)

    b0 = read(ROOT / "c0a_disease/comparison/c0a_vs_stage3_b0.csv").iloc[0]
    fig, ax = plt.subplots(figsize=(5, 4)); ax.bar(["C0-A SH093-224", "B0 RGB 256x320"], [b0.auc_c0a, b0.auc_b0], color=["#4c78a8", "#f58518"]); ax.axhline(.5, color="gray", ls="--"); ax.set(ylabel="Pooled OOF AUC", title="Paired descriptive comparison"); fig.tight_layout(); fig.savefig(ROOT / "c0a_disease/figures/c0a_vs_b0_paired_comparison.png", dpi=180); plt.close(fig)

    cov = read(ROOT / "c0b_camera/baselines/covariate_baseline_metrics.csv").iloc[0]
    low = read(ROOT / "c0b_camera/baselines/lowlevel_metrics.csv").iloc[0]
    matched = read(ROOT / "c0b_camera/matched/matched_metrics.csv").iloc[0]
    fig, ax = plt.subplots(figsize=(6, 4)); ax.bar(["Covariates", "Low-level image", "ResNet SH093"], [cov.auc, low.auc, decision["c0b_metrics"]["auc"]], color=["#72b7b2", "#eeca3b", "#e45756"]); ax.axhline(.5, color="gray", ls="--"); ax.set(ylabel="Camera-domain AUC", title="C0-B baseline comparison"); fig.tight_layout(); fig.savefig(ROOT / "c0b_camera/figures/c0b_covariate_vs_image_baselines.png", dpi=180); plt.close(fig)
    features = read(ROOT / "c0b_camera/baselines/lowlevel_feature_table.csv"); numeric = [c for c in features if c not in {"sample_id", "patient_group_id", "task_label"}]; gaps = [(c, abs(features.loc[features.task_label == 1, c].mean() - features.loc[features.task_label == 0, c].mean())) for c in numeric]; top = sorted(gaps, key=lambda x: x[1], reverse=True)[:12]
    fig, ax = plt.subplots(figsize=(8, 4)); ax.bar([x[0] for x in top], [x[1] for x in top], color="#54a24b"); ax.tick_params(axis="x", rotation=45); ax.set(ylabel="Absolute class mean difference", title="Low-level SH093 domain differences"); fig.tight_layout(); fig.savefig(ROOT / "c0b_camera/figures/c0b_lowlevel_feature_importance.png", dpi=180); plt.close(fig)
    fig, ax = plt.subplots(figsize=(5, 4)); ax.bar(["Full patient-only", "1:1 group matched"], [decision["c0b_metrics"]["auc"], matched.auc], color=["#e45756", "#b279a2"]); ax.axhline(.5, color="gray", ls="--"); ax.set(ylabel="Camera-domain AUC", title="C0-B full vs matched"); fig.tight_layout(); fig.savefig(ROOT / "c0b_camera/figures/c0b_matched_vs_full_auc.png", dpi=180); plt.close(fig)
    fig, ax = plt.subplots(figsize=(7, 3)); ax.axis("off"); ax.text(.03, .70, f"C0-A same-camera disease AUC: {decision['c0a_metrics']['auc']:.3f} ({decision['c0a_same_camera_signal_level']})", fontsize=12); ax.text(.03, .40, f"C0-B patient-only domain AUC: {decision['c0b_metrics']['auc']:.3f} ({decision['c0b_camera_domain_predictability_level']})", fontsize=12); ax.text(.03, .10, f"Decision: {decision['full500_sh093_interpretation']}", fontsize=12); fig.tight_layout(); fig.savefig(ROOT / "joint/figures/joint_evidence_matrix.png", dpi=180); plt.close(fig)

    af = read(ROOT / "c0a_disease/metrics/c0a_fold_metrics.csv"); bf = read(ROOT / "c0b_camera/metrics/c0b_fold_metrics.csv")
    def rows(frame: pd.DataFrame) -> str:
        return "\n".join(f"| {int(r.fold)} | {int(r.selected_epoch)} | {r.inner_val_auc:.4f} | {r.test_auc:.4f} | {r.test_balanced_accuracy:.4f} |" for r in frame.itertuples())
    joint = f"""# Stage3-C0 Joint Report

## Purpose and asset contract

SH093 fixes the explicit spherical-harmonic illumination at the final generated-image stage. It is not a camera-independent reflectance map and does not establish removal of ISP, collection workflow, environment, or generator-domain effects. `fixedSH_origcam_menafg` was audited as 500 unique, decodable RGB uint8 PNG images at 224x224; no labels, NYHA, folds, or predictions were supplied to C0 pixel processing.

The historical 500-case SH093 experiment was located at `{ROOT.parent.parent / '500Data' if False else 'experiments/500Data/E0B_Global_ResNet18_ControlVsPatient_Binary_fixedSH_origcam_menafg_5fold'}`. Its reported pooled OOF macro-AUC was 0.8281, but the historical outer validation fold selected the checkpoint and no independent inner validation was found. It is therefore historical context, not a valid nested comparator.

## C0-A: Xiaomi-only disease task

C0-A reused the exact B0 cohort, outer folds and inner folds: 233 samples, 115 Control, 118 Patient, 232 patient groups, all `M2006J10C`. It used ImageNet-initialized ResNet18, dropout 0.3, 224x224 RGB, ImageNet normalization, horizontal flip only, AdamW (1e-4, 1e-4), batch size 16, maximum 30 epochs, patience 5, frozen-eval BN, no AMP, BCEWithLogitsLoss(pos_weight=1.0), inner-validation AUC checkpoint selection, and a fixed 0.5 threshold.

- Pooled OOF AUC: {decision['c0a_metrics']['auc']:.4f}; 95% CI {decision['c0a_ci'][0]['lower_95']:.4f}-{decision['c0a_ci'][0]['upper_95']:.4f}
- Accuracy / BA / Macro-F1: {decision['c0a_metrics']['accuracy']:.4f} / {decision['c0a_metrics']['balanced_accuracy']:.4f} / {decision['c0a_metrics']['macro_f1']:.4f}
- Sensitivity / specificity / Brier: {decision['c0a_metrics']['sensitivity']:.4f} / {decision['c0a_metrics']['specificity']:.4f} / {decision['c0a_metrics']['brier']:.4f}
- Calibration intercept / slope: {ai:.4f} / {aslope:.4f}
- C0-A vs B0 descriptive Delta AUC: {b0.auc:.4f}; matched Original-RGB-224 control: unavailable (no uniquely verified matched asset)

| Fold | Best epoch | Inner AUC | Outer AUC | Outer BA |
|---:|---:|---:|---:|---:|
{rows(af)}

## C0-B: Patient-only camera-associated domain task

C0-B excluded every Control and used 385 Patient samples: 118 Xiaomi and 267 HONOR. It used a separate group-safe outer/inner nested split, with class weight calculated only from each inner training fold. C0-B measures camera-associated domain predictability, not a pure device fingerprint.

- Pooled camera AUC: {decision['c0b_metrics']['auc']:.4f}; 95% CI {decision['c0b_ci'][0]['lower_95']:.4f}-{decision['c0b_ci'][0]['upper_95']:.4f}
- BA / Macro-F1 / Xiaomi recall / HONOR recall: {decision['c0b_metrics']['balanced_accuracy']:.4f} / {decision['c0b_metrics']['macro_f1']:.4f} / {decision['c0b_metrics']['specificity']:.4f} / {decision['c0b_metrics']['sensitivity']:.4f}
- Covariate-only AUC: {cov.auc:.4f}; low-level image AUC: {low.auc:.4f}
- Group-level 1:1 matching: {int(matched.n_pairs)} pairs; matched camera AUC: {matched.auc:.4f}

| Fold | Best epoch | Inner AUC | Outer AUC | Outer BA |
|---:|---:|---:|---:|
{rows(bf)}

## Joint conclusion

`c0a_same_camera_signal_level={decision['c0a_same_camera_signal_level']}` and `c0b_camera_domain_predictability_level={decision['c0b_camera_domain_predictability_level']}`. The 500-case 0.8281 result is currently classified as `{decision['full500_sh093_interpretation']}`. `sh093_main_binary_candidate={decision['sh093_main_binary_candidate']}` and `next_stage_recommendation={decision['next_stage_recommendation']}`.

No HONOR samples entered C0-A. No Control samples entered C0-B. No multi-SH, fixedcam, Exposure/Gamma, ROI, or Stage3-B1 experiment was run.
"""
    (ROOT / "joint/reports/stage3_c0_joint_report.md").write_text(joint, encoding="utf-8")
    pd.DataFrame(columns=["warning"]).to_csv(ROOT / "logs/warnings.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(columns=["failure"]).to_csv(ROOT / "logs/failures.csv", index=False, encoding="utf-8-sig")
    files = [{"path": str(p.relative_to(ROOT)), "bytes": p.stat().st_size} for p in sorted(ROOT.rglob("*")) if p.is_file()]
    dump({"root": str(ROOT), "files": files}, ROOT / "joint/reports/stage3_c0_output_inventory.json")


if __name__ == "__main__":
    main()
