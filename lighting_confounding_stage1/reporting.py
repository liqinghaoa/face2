from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from .config import json_safe


def _fmt(x: Any) -> str:
    if x is None or pd.isna(x):
        return "NA"
    if isinstance(x, float):
        return f"{x:.4f}"
    return str(x)


def gate_summary(metadata_metrics: pd.DataFrame, rgb_corr: pd.DataFrame, strat_worst: dict[str, Any], association: dict[str, pd.DataFrame]) -> dict[str, Any]:
    aucs = dict(zip(metadata_metrics["model_name"], metadata_metrics["roc_auc"]))
    camera = association["by_device"]
    device_assoc = bool((camera.get("p_value", pd.Series(dtype=float)).dropna() < 0.05).any()) if not camera.empty else False
    exposure_assoc = False
    cont = association["continuous"]
    if not cont.empty:
        exposure_assoc = bool((cont["variable"].isin(["brightness_device_z", "log2_iso_device_z", "log2_exposure_device_z"]) & (cont["mannwhitney_p"].fillna(1) < 0.05)).any())
    rgb_exposure = False
    if not rgb_corr.empty:
        rgb_exposure = bool((rgb_corr["scope"].isin(["overall", "label"]) & (rgb_corr["p_value"].fillna(1) < 0.05) & (rgb_corr["spearman_r"].abs().fillna(0) >= 0.15)).any())
    time_auc = aucs.get("META-T")
    device_auc = aucs.get("META-D")
    light_auc = aucs.get("META-L")
    return {
        "device_bias_risk": bool(device_assoc or (device_auc is not None and device_auc >= 0.60) or (strat_worst.get("camera_model", {}).get("DeltaAUC") or 0) >= 0.10),
        "exposure_proxy_risk": bool(exposure_assoc or rgb_exposure or (light_auc is not None and light_auc >= 0.60)),
        "capture_batch_risk": bool(time_auc is not None and time_auc >= 0.60),
        "metadata_auc": {k: float(v) for k, v in aucs.items()},
        "interpretation_note": "Gates are evidence summaries, not causal decisions and not single-threshold proof.",
    }


def write_reports(
    output_dir: Path,
    *,
    preflight: dict[str, Any],
    association: dict[str, pd.DataFrame],
    metadata_results: dict[str, Any],
    rgb_results: dict[str, Any],
    strat_results: dict[str, Any],
) -> dict[str, Any]:
    reports = output_dir / "reports"
    reports.mkdir(parents=True, exist_ok=True)
    metadata_metrics = metadata_results["metrics"]
    gates = gate_summary(metadata_metrics, rgb_results["correlations"], strat_results["worst"], association)
    rgb_metrics = rgb_results["rgb_metrics"]

    summary = {
        "experiment": "Lighting_Confounding_Audit_Stage1_v1",
        "preflight": preflight,
        "metadata_only_metrics": metadata_metrics.to_dict("records"),
        "rgb_metrics": rgb_metrics,
        "worst_group": strat_results["worst"],
        "gates": gates,
        "limitations": [
            "EXIF BrightnessValue, ISO and ExposureTime are acquisition-condition or lighting-proxy variables, not direct ambient illumination measurements.",
            "Associations are not interpreted as causal effects.",
            "The analysis uses existing OOF predictions and does not retrain or alter the RGB model.",
        ],
    }
    (reports / "stage1_machine_readable_summary.json").write_text(json.dumps(json_safe(summary), ensure_ascii=False, indent=2), encoding="utf-8")

    lines = [
        "# Lighting Confounding Audit Stage 1",
        "",
        "## 1. 研究目的",
        "审计固定 500 例 Control vs Patient 二分类任务中，设备、曝光代理变量和采集时间是否与标签或当前 RGB OOF 输出相关。",
        "",
        "## 2. 阶段一定位",
        "本阶段只量化采集条件混杂风险，不读取人脸像素，不重新训练 RGB 模型，也不声称消除光照影响。",
        "",
        "## 3. 输入数据和冻结约束",
        f"- split: `{preflight['input_files']['split_csv']}`",
        f"- metadata: `{preflight['input_files']['metadata_xlsx']}` / sheet `{preflight['input_files']['metadata_sheet']}`",
        f"- RGB OOF: `{preflight['input_files']['oof_predictions_csv']}`",
        "",
        "## 4. 500例对齐结果",
        f"- 固定队列: {preflight['gates']['fixed_cohort_rows']} rows, unique sample_id={preflight['gates']['fixed_cohort_unique_sample_id']}",
        f"- metadata matched: {preflight['gates']['metadata_matched_500']}/500",
        f"- RGB OOF matched: {preflight['gates']['oof_matched_500']}/500",
        f"- label conflicts: {preflight['gates']['label_conflicts']}",
        f"- fold conflicts: {preflight['gates']['fold_conflicts']}",
        f"- patient-group cross-fold: {preflight['gates']['patient_group_cross_fold_count']}",
        "",
        "## 5. 元数据字段及预处理",
        "字段识别结果已写入 `metadata/detected_*_schema.json`。曝光时间和 ISO 在取 log2 前只接受 >0 数值；设备内 z 值按外层 fold 训练部分计算。",
        "",
        "## 6-8. 设备和曝光参数分布",
        "设备、连续曝光代理变量、类别曝光变量与采集时间的关联结果见 `association/` 目录。设备内曝光差异优先参考 `metadata_label_association_by_device.csv`。",
        "",
        "## 9. Metadata-only 五折结果",
        "| Model | ROC-AUC | Accuracy | Balanced Acc | Macro-F1 | Brier |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in metadata_metrics.to_dict("records"):
        lines.append(f"| {row['model_name']} | {_fmt(row.get('roc_auc'))} | {_fmt(row.get('accuracy'))} | {_fmt(row.get('balanced_accuracy'))} | {_fmt(row.get('macro_f1'))} | {_fmt(row.get('brier_score'))} |")
    lines += [
        "",
        "## 10. 标签置换检验",
        "置换在 patient_group_id 层面进行，结果见 `metadata_only/metadata_permutation_results.csv`。",
        "",
        "## 11. 当前 RGB 模型总体 OOF 结果",
        f"- ROC-AUC: {_fmt(rgb_metrics.get('roc_auc'))}",
        f"- Accuracy: {_fmt(rgb_metrics.get('accuracy'))}",
        f"- Macro-F1: {_fmt(rgb_metrics.get('macro_f1'))}",
        f"- Balanced Accuracy: {_fmt(rgb_metrics.get('balanced_accuracy'))}",
        f"- Sensitivity: {_fmt(rgb_metrics.get('sensitivity'))}",
        f"- Specificity: {_fmt(rgb_metrics.get('specificity'))}",
        "",
        "## 12-13. RGB预测与元数据关联及错误风险",
        "Spearman 相关、标签调整后的 logit 回归和错误风险回归见 `rgb_dependence/` 目录。解释限于相关性与风险审计。",
        "",
        "## 14-15. 分层性能和 Worst-group",
        "分层性能见 `stratified/rgb_stratified_metrics.csv`。",
    ]
    for name, payload in strat_results["worst"].items():
        lines.append(f"- {name}: WorstGroupAUC={_fmt(payload.get('WorstGroupAUC'))}, DeltaAUC={_fmt(payload.get('DeltaAUC'))}, worst_stratum={payload.get('worst_stratum', 'NA')}")
    lines += [
        "",
        "## 16. 设备偏倚判断",
        "存在设备或采集来源偏倚风险。" if gates["device_bias_risk"] else "当前未观察到强设备偏倚证据。",
        "",
        "## 17. 曝光或光照代理偏倚判断",
        "存在曝光或光照代理相关混杂风险。" if gates["exposure_proxy_risk"] else "当前未观察到强曝光代理混杂证据。",
        "",
        "## 18. 采集批次偏倚判断",
        "存在采集批次或数据来源 shortcut 风险。" if gates["capture_batch_risk"] else "当前未观察到强采集批次 shortcut 证据。",
        "",
        "## 19. 研究限制",
        "EXIF 的 BrightnessValue、ISO 和 ExposureTime 只是拍摄条件或光照代理变量，不能代表真实环境照度。所有结果均为内部固定五折与现有 RGB OOF 的相关性审计，不构成因果结论。",
        "",
        "## 20. 下一阶段建议",
        "优先结合风险门控结果做设备/曝光/批次分层复核；若曝光代理风险持续存在，再进入图像级光照质量控制、受约束光照增强或重加权评估。",
    ]
    (reports / "stage1_lighting_confounding_report.md").write_text("\n".join(lines), encoding="utf-8")
    return summary
