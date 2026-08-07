from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd


def _fmt(value: Any, digits: int = 4) -> str:
    try:
        return f"{float(value):.{digits}f}"
    except Exception:
        return str(value)


def _final_judgment_lines(gate_decision: dict[str, Any], a3_increment: dict[str, Any] | None) -> list[str]:
    gate_pass = gate_decision.get("gate_pass")
    deltas = gate_decision.get("metric_deltas") or {}
    positive = gate_decision.get("positive_signals") or {}
    no_collapse = gate_decision.get("no_performance_collapse")
    stability_improved = bool(
        positive.get("mean_prediction_std_lower")
        or positive.get("case_level_flip_rate_lower")
        or positive.get("worst_light_auc_improved")
    )
    if gate_pass is True:
        relighting_vs_colorjitter = "是；A2 通过预设 gate，可认为在本协议下优于 ColorJitter。"
        preserves_original = "是；A2 满足预设的原图分类性能保持阈值。"
        p2b = "可作为 P2-B 的候选输入，但仍需结合 P2-B 提示词单独启动。"
    elif gate_pass is False:
        relighting_vs_colorjitter = "否；A2 未通过 A2-vs-A1 gate，不能判定物理重光照优于 ColorJitter。"
        preserves_original = "否；A2 在至少一个核心分类指标上超过预设性能跌幅阈值。"
        p2b = "否；本阶段不建议自动进入 P2-B。"
    else:
        relighting_vs_colorjitter = "未评估；gate 未产生正式结论。"
        preserves_original = "未评估。"
        p2b = "否；缺少正式 gate 结论。"
    a3_triggered = "是。" if gate_decision.get("a3_should_run") else "否。"
    if a3_increment is None:
        consistency_value = "未评估；A3 未运行。"
    else:
        consistency_value = "是。" if a3_increment.get("recommended_for_p2_b") else "否；A3 未显示安全的额外增益。"
    return [
        "## 最终中文判断",
        "",
        f"- 物理重光照是否优于 ColorJitter：{relighting_vs_colorjitter}",
        f"- 是否保持原图分类能力：{preserves_original}",
        f"- 是否提高跨光照稳定性：{'是，至少一个稳定性指标改善；但该改善不能替代分类性能 gate。' if stability_improved else '否，未见预设稳定性正向信号。'}",
        f"- A3 是否由 gate 触发：{a3_triggered}",
        f"- 完整一致性是否有额外价值：{consistency_value}",
        f"- 是否建议进入 P2-B：{p2b}",
        f"- A2-A1 主要分类差值：Macro-AUC={_fmt(deltas.get('macro_auc'))}, Macro-F1={_fmt(deltas.get('macro_f1'))}, BA={_fmt(deltas.get('balanced_accuracy'))}; no_performance_collapse={no_collapse}.",
    ]


def write_final_report(
    *,
    summary_dir: str | Path,
    model_comparison: pd.DataFrame,
    stability_comparison: pd.DataFrame,
    paired_comparisons: pd.DataFrame,
    gate_decision: dict[str, Any],
    a3_increment: dict[str, Any] | None,
    manifest_sha256: str,
    experiment_statuses: dict[str, Any],
) -> Path:
    summary = Path(summary_dir)
    summary.mkdir(parents=True, exist_ok=True)
    lines = [
        "# P2-A formal result report",
        "",
        "## Experiment status",
        "",
        f"- Manifest SHA256: `{manifest_sha256}`",
        f"- A3 triggered by gate: `{gate_decision.get('a3_should_run')}`",
        "- Formal five-fold status:",
    ]
    for experiment_id, status in experiment_statuses.items():
        lines.append(f"  - `{experiment_id}`: `{status}`")
    lines += [
        "",
        "## Original-image classification metrics",
        "",
        "| Model | Macro-AUC | Macro-F1 | Balanced Accuracy | Sensitivity | Specificity |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in model_comparison.itertuples(index=False):
        lines.append(
            f"| {row.experiment_id} | {_fmt(row.macro_auc)} | {_fmt(row.macro_f1)} | {_fmt(row.balanced_accuracy)} | {_fmt(row.patient_sensitivity)} | {_fmt(row.control_specificity)} |"
        )
    lines += [
        "",
        "## Cross-lighting stability metrics",
        "",
        "| Model | Mean prediction std | Case flip rate | Worst-light AUC | Mean feature cosine |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in stability_comparison.itertuples(index=False):
        lines.append(
            f"| {row.experiment_id} | {_fmt(row.mean_prediction_std)} | {_fmt(row.case_level_flip_rate)} | {_fmt(row.worst_light_auc)} | {_fmt(row.mean_feature_cosine)} |"
        )
    lines += [
        "",
        "## Gate decision",
        "",
        f"- Candidate: `{gate_decision.get('candidate')}`",
        f"- Reference: `{gate_decision.get('reference')}`",
        f"- Gate pass: `{gate_decision.get('gate_pass')}`",
        f"- Decision reason: {gate_decision.get('decision_reason')}",
        "",
        "## Paired patient-cluster bootstrap comparisons",
        "",
        "The paired comparisons use the same OOF cases and patient-group cluster resampling. Visit/case remains the evaluation unit; no patient-level averaging is performed.",
        "",
    ]
    if paired_comparisons.empty:
        lines.append("- No paired comparisons available.")
    else:
        important = paired_comparisons[
            paired_comparisons["metric"].isin(["macro_auc", "macro_f1", "balanced_accuracy", "mean_prediction_std", "case_level_flip_rate", "worst_light_auc"])
        ].copy()
        lines += [
            "| Comparison | Metric | Delta | 95% CI |",
            "|---|---|---:|---:|",
        ]
        for row in important.itertuples(index=False):
            lines.append(f"| {row.comparison} | {row.metric} | {_fmt(row.difference_b_minus_a)} | [{_fmt(row.ci_lower)}, {_fmt(row.ci_upper)}] |")
    if a3_increment is not None:
        lines += [
            "",
            "## A3 increment decision",
            "",
            f"- Status: `{a3_increment.get('status')}`",
            f"- Recommended for P2-B: `{a3_increment.get('recommended_for_p2_b')}`",
            f"- Reason: {a3_increment.get('reason')}",
        ]
    lines += ["", *_final_judgment_lines(gate_decision, a3_increment)]
    lines += [
        "",
        "## Boundary statements",
        "",
        "- No P2-B execution was started.",
        "- No P3/P4 execution was started.",
        "- No patient-level prediction averaging was used.",
        "- Relighting results are interpreted as physics-inspired counterfactual augmentations, not true physiological parameter recovery.",
    ]
    path = summary / "p2_a_final_report.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path
