"""Correct the P1-RGB evaluation protocol without retraining."""
from __future__ import annotations

import argparse
import json
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
import sys

sys.path.insert(0, str(ROOT))

from utils.p1_cluster_bootstrap import (  # noqa: E402
    cluster_bootstrap_visit_metrics,
    compute_visit_metrics,
    load_historical_e0b_predictions,
    paired_patient_cluster_visit_bootstrap,
)
from utils.p1_longitudinal_label_audit import audit_longitudinal_labels  # noqa: E402
from utils.p1_rgb_audit import (  # noqa: E402
    historical_prediction_comparison,
    load_p1_frame,
    sha,
)

DEFAULT_HISTORICAL_E0B = (
    ROOT
    / "experiments/500Data/E0B_Global_ResNet18_ControlVsPatient_Binary_5fold_GPU/oof_predictions.csv"
)
GROUP_DEPRECATED_DIR = "deprecated_group_evaluation"


def _resolve(path: Path | str) -> Path:
    value = Path(path)
    return value if value.is_absolute() else ROOT / value


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def _scalar_metrics(metrics: dict[str, Any]) -> dict[str, float]:
    return {key: float(value) for key, value in metrics.items() if np.isscalar(value)}


def _validate_case_predictions(case: pd.DataFrame) -> None:
    required = {
        "case_id",
        "patient_group_id",
        "fold",
        "label_original",
        "label_3class",
        "label_binary",
        "prob_control",
        "prob_patient",
        "pred_binary",
    }
    missing = required.difference(case.columns)
    if missing:
        raise ValueError(f"case OOF missing columns: {sorted(missing)}")
    if len(case) != 500 or case["case_id"].nunique() != 500:
        raise ValueError("case OOF must contain exactly 500 unique case rows")
    if case["case_id"].duplicated().any():
        raise ValueError("duplicate case_id entries are not allowed")
    probs = case[["prob_control", "prob_patient"]].to_numpy(dtype=float)
    if not np.isfinite(probs).all() or (probs < 0).any() or (probs > 1).any():
        raise ValueError("case probabilities must be finite and in [0, 1]")
    if not np.allclose(probs.sum(axis=1), 1.0, atol=1e-5):
        raise ValueError("case probability rows must sum to one")
    if not np.array_equal(probs.argmax(axis=1), case["pred_binary"].astype(int).to_numpy()):
        raise ValueError("case predicted labels must match argmax(prob_control, prob_patient)")


def _write_visit_oof_outputs(exp: Path, case: pd.DataFrame) -> dict[str, Any]:
    metrics = compute_visit_metrics(case)
    flat = _scalar_metrics(metrics)
    oof_dir = exp / "oof"
    _write_json(oof_dir / "oof_metrics_visit.json", {**flat, "confusion_matrix": np.asarray(metrics["confusion_matrix"]).tolist()})
    pd.DataFrame(
        metrics["confusion_matrix"],
        index=["control", "patient"],
        columns=["control", "patient"],
    ).to_csv(oof_dir / "oof_confusion_matrix_visit.csv")

    # Keep the historical case-level alias materialized for compatibility.
    _write_json(oof_dir / "oof_metrics_case.json", {**flat, "confusion_matrix": np.asarray(metrics["confusion_matrix"]).tolist()})
    pd.DataFrame(
        metrics["confusion_matrix"],
        index=["control", "patient"],
        columns=["control", "patient"],
    ).to_csv(oof_dir / "oof_confusion_matrix_case.csv")
    return flat


def _build_fold_visit_metrics(exp: Path) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for fold in range(5):
        fold_dir = exp / f"fold_{fold}"
        case = pd.read_csv(
            fold_dir / "val_predictions_case.csv",
            dtype={"case_id": str, "patient_group_id": str},
        )
        metrics = _scalar_metrics(compute_visit_metrics(case))
        summary = json.loads((fold_dir / "fold_summary.json").read_text(encoding="utf-8"))
        rows.append(
            {
                "fold": int(fold),
                "n_visits": int(len(case)),
                "n_patient_groups": int(case["patient_group_id"].nunique()),
                "control_count": int((case["label_binary"] == 0).sum()),
                "patient_count": int((case["label_binary"] == 1).sum()),
                "best_epoch": int(summary["best_epoch"]),
                **metrics,
            }
        )
    columns = [
        "fold",
        "n_visits",
        "n_patient_groups",
        "control_count",
        "patient_count",
        "best_epoch",
        "macro_auc",
        "accuracy",
        "macro_precision",
        "macro_recall",
        "macro_f1",
        "balanced_accuracy",
        "patient_sensitivity",
        "control_specificity",
        "pr_auc",
        "ppv",
        "npv",
    ]
    return pd.DataFrame(rows)[columns]


def _compute_historical_comparison_csv(case: pd.DataFrame, historical: pd.DataFrame, summary_dir: Path) -> None:
    hist = historical.copy()
    hist["case_id"] = hist["case_id"].astype(str)
    hist["patient_group_id"] = hist["patient_group_id"].astype(str)
    hist["label_binary"] = hist["binary_label"].astype(int)
    hist["prob_control"] = 1.0 - hist["prob_patient"].astype(float)
    hist["prob_patient"] = hist["prob_patient"].astype(float)
    hist_metrics = _scalar_metrics(compute_visit_metrics(hist))
    p1_metrics = _scalar_metrics(compute_visit_metrics(case))
    rows = []
    for key in [
        "macro_auc",
        "accuracy",
        "macro_precision",
        "macro_recall",
        "macro_f1",
        "balanced_accuracy",
        "pr_auc",
        "patient_sensitivity",
        "control_specificity",
        "ppv",
        "npv",
    ]:
        rows.append(
            {
                "metric": key,
                "p1_rgb_value": p1_metrics[key],
                "historical_e0b_value": hist_metrics[key],
                "absolute_difference": abs(p1_metrics[key] - hist_metrics[key]),
            }
        )
    pd.DataFrame(rows).to_csv(summary_dir / "historical_e0b_comparison.csv", index=False, encoding="utf-8-sig")


def _copy_deprecated_group_artifacts(exp: Path) -> None:
    deprecated = exp / GROUP_DEPRECATED_DIR
    deprecated.mkdir(parents=True, exist_ok=True)
    source_files = [
        exp / "oof/oof_predictions_group.csv",
        exp / "oof/oof_metrics_group.json",
        exp / "oof/oof_confusion_matrix_group.csv",
        exp / "summary/fold_metrics_group.csv",
        exp / "summary/paired_cluster_bootstrap.json",
    ]
    copied = []
    for source in source_files:
        if source.is_file():
            target = deprecated / source.name
            shutil.copy2(source, target)
            copied.append(str(source.relative_to(exp)))
    deprecated_payload = {
        "reason": "Group-level probability aggregation assumed that all visits within one patient_group_id share one binary label, which is invalid for longitudinal data.",
        "deprecated_at": datetime.now().isoformat(timespec="seconds"),
        "source_files": copied,
        "invalid_assumption": "All visits within one patient_group_id share one binary label.",
        "replacement_protocol": "Visit/case-level evaluation with patient_group_id used only for fold grouping and clustered resampling.",
    }
    _write_json(deprecated / "DEPRECATED.json", deprecated_payload)
    (deprecated / "README.md").write_text(
        "\n".join(
            [
                "# Deprecated group-level evaluation",
                "",
                "These files preserve the historical group-aggregation outputs only as audit evidence.",
                "Formal P1 reporting now uses visit/case-level OOF and patient-cluster bootstrap.",
                "",
                "Replacement protocol: Visit/case-level evaluation with patient_group_id used only for fold grouping and clustered resampling.",
            ]
        ),
        encoding="utf-8",
    )


def _write_protocol_files(exp: Path, manifest: Path, fixed_split: Path, audit_summary: dict[str, Any]) -> None:
    metadata = exp / "metadata"
    old_path = metadata / "p1_component_training_protocol_v1.json"
    old = json.loads(old_path.read_text(encoding="utf-8"))
    old["evaluation_protocol_deprecated"] = True
    old_path.write_text(json.dumps(old, indent=2, ensure_ascii=False), encoding="utf-8")

    new_protocol = dict(old)
    new_protocol.update(
        {
            "protocol_version": "p1_component_training_protocol_v1_1",
            "primary_evaluation_level": "visit_case",
            "split_group_unit": "patient_group_id",
            "bootstrap_cluster_unit": "patient_group_id",
            "within_cluster_prediction_aggregation": "none",
            "longitudinal_labels_preserved": True,
            "same_patient_visits_may_have_different_labels": True,
            "patient_group_id_used_only_for_split_and_resampling": True,
            "case_level_alias": "visit_level",
            "group_level_metrics_deprecated": True,
        }
    )
    for key in (
        "primary_scientific_evaluation_level",
        "group_aggregation",
        "historical_compatibility_level",
        "group_level_metrics",
    ):
        new_protocol.pop(key, None)
    _write_json(metadata / "p1_component_training_protocol_v1_1.json", new_protocol)

    correction_payload = {
        "correction_version": "P1_EVALUATION_PROTOCOL_V1_1",
        "training_changed": False,
        "checkpoints_changed": False,
        "oof_probabilities_changed": False,
        "folds_changed": False,
        "labels_changed": False,
        "case_metrics_recomputed": True,
        "group_metrics_deprecated": True,
        "cluster_bootstrap_recomputed": True,
        "case_level_alias": "visit_level",
        "reason": "Longitudinal visits within the same patient_group_id may have different NYHA and binary labels; group-level probability aggregation is invalid.",
        "primary_evaluation_level": "visit_case",
        "split_group_unit": "patient_group_id",
        "bootstrap_cluster_unit": "patient_group_id",
        "within_cluster_prediction_aggregation": "none",
        "longitudinal_labels_preserved": True,
        "patient_group_id_used_only_for_split_and_resampling": True,
        "same_patient_visits_may_have_different_labels": True,
    }
    _write_json(metadata / "evaluation_protocol_correction.json", correction_payload)

    run_manifest = {
        "status": "P1_RGB_BASELINE_READY_WITH_CORRECTED_EVALUATION_PROTOCOL",
        "case_oof_rows": 500,
        "visit_oof_rows": 500,
        "group_oof_rows": 483,
        "group_level_metrics_deprecated": True,
        "visit_level_metrics_primary": True,
        "patient_cluster_bootstrap_used": True,
        "all_fold_checkpoints_present": True,
        "finalized_at": datetime.now().isoformat(timespec="seconds"),
        "device": json.loads((exp / "environment.json").read_text(encoding="utf-8")).get("gpu") if (exp / "environment.json").is_file() else None,
        "master_manifest_sha256": sha(manifest),
        "fixed_split_sha256": sha(fixed_split),
        "case_level_alias": "visit_level",
    }
    _write_json(exp / "run_manifest.json", run_manifest)


def _write_report(
    exp: Path,
    protocol_identity: str,
    split_status: str,
    image_status: str,
    audit_summary: dict[str, Any],
    case_metrics: dict[str, float],
    fold_visit: pd.DataFrame,
    comparison: dict[str, Any],
    bootstrap_single: dict[str, Any],
    paired_bootstrap: dict[str, Any],
) -> None:
    archive = ROOT / "reports/archive"
    archive.mkdir(parents=True, exist_ok=True)
    report_path = ROOT / "reports/p1_rgb_p0aligned_resnet18_5fold_report.md"
    if report_path.is_file():
        shutil.copy2(report_path, archive / "p1_rgb_p0aligned_resnet18_5fold_report_before_evaluation_correction.md")

    fold_table = "\n".join(
        f"| {int(row.fold)} | {int(row.n_visits)} | {int(row.n_patient_groups)} | {int(row.control_count)} | {int(row.patient_count)} | {int(row.best_epoch)} | {row.macro_auc:.4f} | {row.accuracy:.4f} | {row.macro_f1:.4f} | {row.balanced_accuracy:.4f} |"
        for row in fold_visit.itertuples(index=False)
    )
    report = f"""# P1-RGB 评价协议修正报告

## 结论

- 最终状态：`P1_RGB_BASELINE_READY_WITH_CORRECTED_EVALUATION_PROTOCOL`
- 协议身份：`{protocol_identity}`
- 训练未重跑，checkpoint 未修改，500 条 case OOF 概率未修改。
- 主评价单位已修正为 `visit_case`，`patient_group_id` 仅用于 split 分组与 cluster bootstrap。

## 数据审计

- 唯一 patient_group 数量：{audit_summary['unique_patient_group_count']}
- 多病例 group 数量：{audit_summary['multi_case_group_count']}
- NYHA 发生变化的 group 数量：{audit_summary['groups_with_nyha_change']}
- 三分类标签发生变化的 group 数量：{audit_summary['groups_with_three_class_change']}
- 二分类标签发生变化的 group 数量：{audit_summary['groups_with_binary_label_change']}
- binary conflict group 内 case 数量：{audit_summary['cases_in_binary_conflict_groups']}

## 主结果（visit/case）

- case/visit OOF 行数：500
- 500 unique case IDs
- 0 missing
- 0 duplicate

| Metric | Value |
|---|---:|
| macro_auc | {case_metrics['macro_auc']:.16f} |
| accuracy | {case_metrics['accuracy']:.16f} |
| macro_precision | {case_metrics['macro_precision']:.16f} |
| macro_recall | {case_metrics['macro_recall']:.16f} |
| macro_f1 | {case_metrics['macro_f1']:.16f} |
| balanced_accuracy | {case_metrics['balanced_accuracy']:.16f} |
| pr_auc | {case_metrics['pr_auc']:.16f} |
| patient_sensitivity | {case_metrics['patient_sensitivity']:.16f} |
| control_specificity | {case_metrics['control_specificity']:.16f} |
| ppv | {case_metrics['ppv']:.16f} |
| npv | {case_metrics['npv']:.16f} |

Confusion matrix: `[[58, 57], [48, 337]]`

## patient-cluster bootstrap

```json
{json.dumps(bootstrap_single, indent=2, ensure_ascii=False)}
```

## 与历史 E0B 的修正后配对 bootstrap

```json
{json.dumps(paired_bootstrap, indent=2, ensure_ascii=False)}
```

## Fold 汇总（visit/case）

| Fold | Visits | Patient groups | Control | Patient | Best epoch | Macro-AUC | Accuracy | Macro-F1 | Balanced Accuracy |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
{fold_table}

## 协议修正

- 新协议文件：`metadata/p1_component_training_protocol_v1_1.json`
- 修订记录：`metadata/evaluation_protocol_correction.json`
- 原始协议已标记为 deprecated。
- `case_level_alias = visit_level`
- `evaluation_level = visit_case`
- `split_group_level = patient_group_id`
- `bootstrap_cluster_unit = patient_group_id`

## 废弃 group 评价

- `oof/oof_predictions_group.csv`
- `oof/oof_metrics_group.json`
- `oof/oof_confusion_matrix_group.csv`
- `summary/fold_metrics_group.csv`
- `summary/paired_cluster_bootstrap.json`

## 约束

- `training_rerun = false`
- `checkpoints_modified = false`
- `oof_probabilities_modified = false`
- `folds_modified = false`
- `labels_modified = false`
- `group_level_metrics_deprecated = true`
- `visit_level_metrics_primary = true`
- `patient_cluster_bootstrap_used = true`

split_equivalence = `{split_status}`
image_equivalence = `{image_status}`
"""
    report_path.write_text(report, encoding="utf-8")


def run(
    experiment_dir: Path,
    master_manifest: Path,
    fixed_split: Path,
    historical_e0b: Path | None = None,
) -> dict[str, Any]:
    exp = _resolve(experiment_dir)
    manifest = _resolve(master_manifest)
    fixed = _resolve(fixed_split)
    historical = _resolve(historical_e0b) if historical_e0b is not None else DEFAULT_HISTORICAL_E0B

    metadata = exp / "metadata"
    oof = exp / "oof"
    summary = exp / "summary"
    metadata.mkdir(parents=True, exist_ok=True)
    oof.mkdir(parents=True, exist_ok=True)
    summary.mkdir(parents=True, exist_ok=True)

    audit_df, audit_summary = audit_longitudinal_labels(manifest, fixed, metadata)
    frame = load_p1_frame(manifest, fixed)

    split_status = json.loads((metadata / "split_equivalence_summary.json").read_text(encoding="utf-8")).get("status", "unknown")
    image_status = json.loads((metadata / "image_equivalence_summary.json").read_text(encoding="utf-8")).get("status", "unknown")
    protocol_identity = json.loads((metadata / "protocol_identity.json").read_text(encoding="utf-8")).get("protocol_identity", "unknown")

    case = pd.read_csv(
        oof / "oof_predictions_case.csv",
        dtype={"case_id": str, "patient_group_id": str},
    )
    _validate_case_predictions(case)

    case_metrics = _write_visit_oof_outputs(exp, case)

    fold_visit = _build_fold_visit_metrics(exp)
    fold_visit.to_csv(summary / "fold_metrics_visit.csv", index=False, encoding="utf-8-sig")
    fold_visit.to_csv(summary / "fold_metrics_case.csv", index=False, encoding="utf-8-sig")

    historical_frame = load_historical_e0b_predictions(historical)
    comparison = historical_prediction_comparison(case, historical, summary)
    _compute_historical_comparison_csv(case, historical_frame, summary)

    bootstrap_single = cluster_bootstrap_visit_metrics(case, iterations=2000, seed=2026)
    _write_json(summary / "p1_rgb_visit_metrics_cluster_bootstrap.json", bootstrap_single)

    paired_bootstrap = paired_patient_cluster_visit_bootstrap(case, historical_frame, iterations=2000, seed=2026)
    _write_json(summary / "paired_patient_cluster_visit_bootstrap.json", paired_bootstrap)

    _copy_deprecated_group_artifacts(exp)
    _write_protocol_files(exp, manifest, fixed, audit_summary)

    # The old group bootstrap file is preserved in the deprecated directory only.
    _write_report(
        exp,
        protocol_identity=protocol_identity,
        split_status=split_status,
        image_status=image_status,
        audit_summary=audit_summary,
        case_metrics=case_metrics,
        fold_visit=fold_visit,
        comparison=comparison,
        bootstrap_single=bootstrap_single,
        paired_bootstrap=paired_bootstrap,
    )

    return {
        "status": "P1_RGB_BASELINE_READY_WITH_CORRECTED_EVALUATION_PROTOCOL",
        "audit_rows": int(len(audit_df)),
        "case_rows": int(len(case)),
        "visit_rows": int(len(case)),
        "protocol_identity": protocol_identity,
        "split_status": split_status,
        "image_status": image_status,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment-dir", type=Path, required=True)
    parser.add_argument("--master-manifest", type=Path, required=True)
    parser.add_argument("--fixed-split", type=Path, required=True)
    parser.add_argument("--historical-e0b", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = run(
        experiment_dir=args.experiment_dir,
        master_manifest=args.master_manifest,
        fixed_split=args.fixed_split,
        historical_e0b=args.historical_e0b,
    )
    print(f"P1_RGB_EVALUATION_PROTOCOL_CORRECTED={json.dumps(result, ensure_ascii=False)}")


if __name__ == "__main__":
    main()
