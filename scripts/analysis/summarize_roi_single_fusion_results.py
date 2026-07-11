from pathlib import Path
import re

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    recall_score,
)


ROOT = Path(r"E:\projects\face2")
SINGLE_ROOT = ROOT / "experiments" / "ROI_500"
FUSION_ROOT = ROOT / "experiments" / "ROI_Fusion_500"
OUT_MD = ROOT / "experiments" / "roi_single_fusion_results_summary_for_chatgpt.md"
OUT_CSV = ROOT / "experiments" / "roi_single_fusion_core_summary.csv"

CLASS_NAMES = ["normal", "mild", "severe"]
CLASS_LABELS = [0, 1, 2]
METRIC_COLS = [
    "macro_auc",
    "balanced_accuracy",
    "macro_f1",
    "accuracy",
    "recall_normal",
    "recall_mild",
    "recall_severe",
    "f1_normal",
    "f1_mild",
    "f1_severe",
    "severe_vs_rest_auc",
    "normal_vs_abnormal_auc",
]


def parse_exp(exp_dir: Path, kind: str):
    name = exp_dir.name
    match = re.search(r"ResNet(18|34|50)", name, re.I)
    backbone = f"resnet{match.group(1)}" if match else "unknown"
    if kind == "roi_single":
        roi_match = re.match(r"ROI_([^_]+)_", name)
        input_name = roi_match.group(1).lower() if roi_match else "unknown"
    else:
        input_name = "multiroi5_fusion"
    return input_name, backbone


def safe_float(value):
    try:
        if pd.isna(value):
            return np.nan
        return float(value)
    except Exception:
        return np.nan


def read_experiment(exp_dir: Path, kind: str):
    input_name, backbone = parse_exp(exp_dir, kind)
    summary = exp_dir / "summary"
    fold = pd.read_csv(summary / "fold_metrics_all.csv")
    oof_metrics = pd.read_csv(summary / "oof_metrics.csv")
    oof_row = oof_metrics.iloc[0].to_dict() if len(oof_metrics) else {}
    pred = pd.read_csv(summary / "oof_predictions.csv")

    true_candidates = ["true_label", "label", "label_3class", "target", "y_true"]
    true_col = next((c for c in true_candidates if c in pred.columns), None)
    if true_col is None:
        raise ValueError(
            f"No true label column in {summary / 'oof_predictions.csv'}; "
            f"columns={list(pred.columns)}"
        )
    y_true = pred[true_col].to_numpy()

    if "pred_class" in pred.columns:
        y_pred = pred["pred_class"].to_numpy()
    elif "pred_label" in pred.columns:
        y_pred = pred["pred_label"].to_numpy()
    else:
        prob_cols = [c for c in pred.columns if c.startswith("prob_")]
        ordered = [
            c
            for c in [
                "prob_normal",
                "prob_mild",
                "prob_severe",
                "prob_0",
                "prob_1",
                "prob_2",
            ]
            if c in prob_cols
        ]
        if len(ordered) >= 3:
            prob_cols = ordered[:3]
        if len(prob_cols) < 3:
            raise ValueError(
                f"No prediction column in {summary / 'oof_predictions.csv'}; "
                f"columns={list(pred.columns)}"
            )
        y_pred = pred[prob_cols].to_numpy().argmax(axis=1)

    cm = confusion_matrix(y_true, y_pred, labels=CLASS_LABELS)
    row_sums = cm.sum(axis=1)
    cm_rates = np.divide(
        cm,
        row_sums[:, None],
        out=np.zeros_like(cm, dtype=float),
        where=row_sums[:, None] != 0,
    )

    rec = {
        "kind": kind,
        "roi_or_input": input_name,
        "backbone": backbone,
        "experiment": exp_dir.name,
        "experiment_dir": str(exp_dir),
        "n_oof": int(len(pred)),
        "n_fold_rows": int(len(fold)),
    }
    for col in METRIC_COLS:
        rec[f"fold_mean_{col}"] = (
            safe_float(fold[col].mean()) if col in fold.columns else np.nan
        )
        rec[f"fold_std_{col}"] = (
            safe_float(fold[col].std(ddof=1)) if col in fold.columns else np.nan
        )
        rec[f"oof_reported_{col}"] = safe_float(oof_row.get(col, np.nan))

    rec["oof_accuracy_hard"] = accuracy_score(y_true, y_pred)
    rec["oof_balanced_accuracy_hard"] = balanced_accuracy_score(y_true, y_pred)
    rec["oof_macro_f1_hard"] = f1_score(
        y_true, y_pred, labels=CLASS_LABELS, average="macro", zero_division=0
    )
    recalls = recall_score(
        y_true, y_pred, labels=CLASS_LABELS, average=None, zero_division=0
    )
    f1s = f1_score(
        y_true, y_pred, labels=CLASS_LABELS, average=None, zero_division=0
    )
    for i, cname in enumerate(CLASS_NAMES):
        rec[f"oof_recall_{cname}_hard"] = recalls[i]
        rec[f"oof_f1_{cname}_hard"] = f1s[i]
        rec[f"n_true_{cname}"] = int(row_sums[i])
        for j, pname in enumerate(CLASS_NAMES):
            rec[f"cm_{cname}_to_{pname}"] = int(cm[i, j])
            rec[f"cm_rate_{cname}_to_{pname}"] = float(cm_rates[i, j])
    return rec


def fmt_num(value, digits=4):
    try:
        if pd.isna(value):
            return ""
        return f"{float(value):.{digits}f}"
    except Exception:
        return str(value)


def cell_text(value):
    if isinstance(value, (float, np.floating)):
        return fmt_num(value)
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    try:
        if pd.isna(value):
            return ""
    except Exception:
        pass
    return str(value)


def md_table(frame: pd.DataFrame, cols, rename=None):
    rename = rename or {}
    headers = [rename.get(c, c) for c in cols]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(cols)) + " |",
    ]
    for _, row in frame.iterrows():
        cells = [cell_text(row.get(c, "")).replace("|", "\\|") for c in cols]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def collect_results():
    records = []
    for path in sorted(SINGLE_ROOT.iterdir()):
        if path.is_dir() and (path / "summary" / "fold_metrics_all.csv").exists():
            records.append(read_experiment(path, "roi_single"))
    for path in sorted(FUSION_ROOT.iterdir()):
        if path.is_dir() and (path / "summary" / "fold_metrics_all.csv").exists():
            records.append(read_experiment(path, "roi_fusion"))
    if not records:
        raise RuntimeError("No ROI single/fusion experiments found.")
    return pd.DataFrame(records)


def main():
    df = collect_results()
    roi_order = {
        "cheek": 0,
        "eye": 1,
        "forehead": 2,
        "lip": 3,
        "multiroi5_fusion": 4,
    }
    bb_order = {"resnet18": 0, "resnet34": 1, "resnet50": 2}
    df["_roi_order"] = df["roi_or_input"].map(roi_order).fillna(99)
    df["_bb_order"] = df["backbone"].map(bb_order).fillna(99)
    df = df.sort_values(["kind", "_roi_order", "_bb_order"]).reset_index(drop=True)

    core_cols = [
        "kind",
        "roi_or_input",
        "backbone",
        "experiment",
        "n_oof",
        "fold_mean_macro_auc",
        "fold_std_macro_auc",
        "fold_mean_balanced_accuracy",
        "fold_std_balanced_accuracy",
        "fold_mean_macro_f1",
        "fold_std_macro_f1",
        "fold_mean_accuracy",
        "fold_std_accuracy",
        "fold_mean_recall_normal",
        "fold_mean_recall_mild",
        "fold_mean_recall_severe",
        "fold_mean_f1_normal",
        "fold_mean_f1_mild",
        "fold_mean_f1_severe",
        "fold_mean_severe_vs_rest_auc",
        "fold_mean_normal_vs_abnormal_auc",
        "oof_accuracy_hard",
        "oof_balanced_accuracy_hard",
        "oof_macro_f1_hard",
        "oof_recall_normal_hard",
        "oof_recall_mild_hard",
        "oof_recall_severe_hard",
        "oof_f1_normal_hard",
        "oof_f1_mild_hard",
        "oof_f1_severe_hard",
        "cm_mild_to_normal",
        "cm_mild_to_mild",
        "cm_mild_to_severe",
        "cm_rate_mild_to_normal",
        "cm_rate_mild_to_mild",
        "cm_rate_mild_to_severe",
        "cm_severe_to_normal",
        "cm_severe_to_mild",
        "cm_severe_to_severe",
        "cm_rate_severe_to_normal",
        "cm_rate_severe_to_mild",
        "cm_rate_severe_to_severe",
        "experiment_dir",
    ]
    df[[c for c in core_cols if c in df.columns]].to_csv(
        OUT_CSV, index=False, encoding="utf-8-sig"
    )

    rank_cols = [
        "kind",
        "roi_or_input",
        "backbone",
        "fold_mean_macro_auc",
        "fold_mean_balanced_accuracy",
        "fold_mean_macro_f1",
        "fold_mean_accuracy",
        "fold_mean_recall_normal",
        "fold_mean_recall_mild",
        "fold_mean_recall_severe",
        "fold_mean_severe_vs_rest_auc",
        "oof_macro_f1_hard",
    ]
    ranked = df.sort_values("fold_mean_macro_f1", ascending=False).reset_index(
        drop=True
    )
    single = df[df.kind == "roi_single"].copy()
    fusion = df[df.kind == "roi_fusion"].copy()

    best_overall = ranked.iloc[0]
    best_single = single.sort_values("fold_mean_macro_f1", ascending=False).iloc[0]
    best_fusion = fusion.sort_values("fold_mean_macro_f1", ascending=False).iloc[0]
    best_backbone_per_roi = (
        single.sort_values("fold_mean_macro_f1", ascending=False)
        .groupby("roi_or_input", as_index=False)
        .head(1)
        .sort_values("_roi_order")
    )
    best_roi_per_backbone = (
        single.sort_values("fold_mean_macro_f1", ascending=False)
        .groupby("backbone", as_index=False)
        .head(1)
        .sort_values("_bb_order")
    )

    fusion_vs_single = []
    for _, fusion_row in fusion.sort_values("_bb_order").iterrows():
        same = (
            single[single.backbone == fusion_row.backbone]
            .sort_values("fold_mean_macro_f1", ascending=False)
            .iloc[0]
        )
        fusion_vs_single.append(
            {
                "backbone": fusion_row.backbone,
                "fusion_macro_f1": fusion_row.fold_mean_macro_f1,
                "best_single_roi_same_backbone": same.roi_or_input,
                "best_single_macro_f1_same_backbone": same.fold_mean_macro_f1,
                "delta_fusion_minus_best_single": fusion_row.fold_mean_macro_f1
                - same.fold_mean_macro_f1,
                "fusion_macro_auc": fusion_row.fold_mean_macro_auc,
                "best_single_macro_auc_same_backbone": same.fold_mean_macro_auc,
                "delta_auc": fusion_row.fold_mean_macro_auc
                - same.fold_mean_macro_auc,
            }
        )
    fusion_vs_single_df = pd.DataFrame(fusion_vs_single)

    cm_cols = [
        "kind",
        "roi_or_input",
        "backbone",
        "fold_mean_macro_f1",
        "cm_rate_mild_to_normal",
        "cm_rate_mild_to_mild",
        "cm_rate_mild_to_severe",
        "cm_rate_severe_to_normal",
        "cm_rate_severe_to_mild",
        "cm_rate_severe_to_severe",
    ]
    cm_view = ranked[cm_cols]

    obs = [
        "按 5-fold fold 平均 macro-F1 排序，当前最佳实验是 "
        f"{best_overall.kind} / {best_overall.roi_or_input} / "
        f"{best_overall.backbone}，macro-F1={fmt_num(best_overall.fold_mean_macro_f1)}，"
        f"macro-AUC={fmt_num(best_overall.fold_mean_macro_auc)}。",
        f"最佳单 ROI 是 {best_single.roi_or_input} / {best_single.backbone}，"
        f"macro-F1={fmt_num(best_single.fold_mean_macro_f1)}。最佳 ROI-fusion 是 "
        f"{best_fusion.backbone}，macro-F1={fmt_num(best_fusion.fold_mean_macro_f1)}。",
    ]
    pos = fusion_vs_single_df[
        fusion_vs_single_df.delta_fusion_minus_best_single > 0
    ]
    if len(pos):
        obs.append(
            "在相同 backbone 下，ROI-fusion 的 macro-F1 高于最佳单 ROI 的 backbone 包括："
            + ", ".join(pos.backbone.tolist())
            + "。"
        )
    else:
        obs.append(
            "在相同 backbone 下，ROI-fusion 的 macro-F1 均未超过对应 backbone 的最佳单 ROI。"
        )
    worst_mild_to_normal = cm_view.sort_values(
        "cm_rate_mild_to_normal", ascending=False
    ).iloc[0]
    worst_severe_to_mild = cm_view.sort_values(
        "cm_rate_severe_to_mild", ascending=False
    ).iloc[0]
    obs.append(
        "从 OOF 硬预测混淆矩阵看，mild 被判为 normal 比例最高的是 "
        f"{worst_mild_to_normal.roi_or_input} / {worst_mild_to_normal.backbone}，"
        f"比例={fmt_num(worst_mild_to_normal.cm_rate_mild_to_normal)}。"
    )
    obs.append(
        "severe 被判为 mild 比例最高的是 "
        f"{worst_severe_to_mild.roi_or_input} / {worst_severe_to_mild.backbone}，"
        f"比例={fmt_num(worst_severe_to_mild.cm_rate_severe_to_mild)}。"
    )

    rename = {
        "kind": "实验类型",
        "roi_or_input": "ROI/输入",
        "backbone": "Backbone",
        "fold_mean_macro_auc": "Macro-AUC",
        "fold_mean_balanced_accuracy": "Balanced Acc",
        "fold_mean_macro_f1": "Macro-F1",
        "fold_mean_accuracy": "Acc",
        "fold_mean_recall_normal": "Recall normal",
        "fold_mean_recall_mild": "Recall mild",
        "fold_mean_recall_severe": "Recall severe",
        "fold_mean_severe_vs_rest_auc": "Severe-vs-rest AUC",
        "oof_macro_f1_hard": "OOF Macro-F1",
        "best_single_roi_same_backbone": "同backbone最佳单ROI",
        "best_single_macro_f1_same_backbone": "最佳单ROI Macro-F1",
        "delta_fusion_minus_best_single": "Fusion-单ROI Macro-F1差值",
        "fusion_macro_f1": "Fusion Macro-F1",
        "fusion_macro_auc": "Fusion Macro-AUC",
        "best_single_macro_auc_same_backbone": "最佳单ROI Macro-AUC",
        "delta_auc": "Fusion-单ROI AUC差值",
        "cm_rate_mild_to_normal": "mild→normal",
        "cm_rate_mild_to_mild": "mild→mild",
        "cm_rate_mild_to_severe": "mild→severe",
        "cm_rate_severe_to_normal": "severe→normal",
        "cm_rate_severe_to_mild": "severe→mild",
        "cm_rate_severe_to_severe": "severe→severe",
    }

    lines = [
        "# ROI single 与 ROI fusion 实验结果汇总\n",
        "本文档由现有训练输出自动汇总，用于复制给 ChatGPT 或其他分析工具进一步解读。"
        "指标主要来自每个实验 `summary/fold_metrics_all.csv` 的 5-fold 平均值；"
        "OOF 硬预测指标和混淆矩阵由 `summary/oof_predictions.csv` 重新计算。\n",
        "## 1. 数据范围与读取情况\n",
        f"- 项目目录：`{ROOT}`",
        f"- ROI single 结果目录：`{SINGLE_ROOT}`",
        f"- ROI fusion 结果目录：`{FUSION_ROOT}`",
        f"- 已汇总实验数：{len(df)} 个，其中 ROI single {len(single)} 个，ROI fusion {len(fusion)} 个。",
        f"- 每个实验 OOF 样本数：{sorted(df['n_oof'].unique().tolist())}",
        f"- 核心汇总 CSV：`{OUT_CSV}`\n",
        "## 2. 全部实验核心指标\n",
        md_table(ranked, rank_cols, rename),
        "",
        "## 3. 总体最佳结果\n",
        md_table(pd.DataFrame([best_overall]), rank_cols, rename),
        "",
        "- 排名标准：`fold_mean_macro_f1`。",
        "- Macro-F1 对三分类不均衡更敏感，因此这里优先用于模型比较；Macro-AUC 和 balanced accuracy 作为辅助判断。\n",
        "## 4. ROI single 内部比较\n",
        "### 4.1 每个 ROI 的最佳 backbone\n",
        md_table(best_backbone_per_roi, rank_cols, rename),
        "\n### 4.2 每个 backbone 的最佳单 ROI\n",
        md_table(best_roi_per_backbone, rank_cols, rename),
        "\n### 4.3 ROI single 按 Macro-F1 排名\n",
        md_table(single.sort_values("fold_mean_macro_f1", ascending=False), rank_cols, rename),
        "",
        "## 5. ROI fusion 结果\n",
        md_table(fusion.sort_values("_bb_order"), rank_cols, rename),
        "",
        "## 6. ROI fusion 与同 backbone 最佳单 ROI 对比\n",
        md_table(
            fusion_vs_single_df,
            [
                "backbone",
                "fusion_macro_f1",
                "best_single_roi_same_backbone",
                "best_single_macro_f1_same_backbone",
                "delta_fusion_minus_best_single",
                "fusion_macro_auc",
                "best_single_macro_auc_same_backbone",
                "delta_auc",
            ],
            rename,
        ),
        "",
        "## 7. OOF 混淆矩阵关键误判比例\n",
        "下表只列 mild 和 severe 两类的预测流向，便于判断模型是否主要发生相邻等级误判，还是轻重程度方向性偏移。\n",
        md_table(cm_view, cm_cols, rename),
        "",
        "## 8. 初步观察\n",
    ]
    lines.extend(f"- {item}" for item in obs)
    lines.extend(
        [
            "",
            "## 9. 建议交给 ChatGPT 进一步分析的问题\n",
            "可以把本文档连同核心 CSV 一起提供给 ChatGPT，并要求重点分析：\n",
            "1. 单 ROI 中哪个区域最稳定，是否存在 backbone 增大后收益下降。",
            "2. ROI fusion 是否真正优于单 ROI，优势体现在哪些指标，而不是只看 accuracy。",
            "3. mild 与 severe 的误判方向是否符合临床等级相邻混淆的预期。",
            "4. 如果后续要写论文结果，主表应该选择 Macro-F1、Macro-AUC、balanced accuracy 中哪些指标。",
            "5. 下一步是否需要做统计检验、置信区间、DeLong/Bootstrap、或 patient-level error analysis。\n",
            "## 10. 可直接引用的路径\n",
            f"- Markdown 汇总：`{OUT_MD}`",
            f"- CSV 汇总：`{OUT_CSV}`",
            f"- ROI single 实验根目录：`{SINGLE_ROOT}`",
            f"- ROI fusion 实验根目录：`{FUSION_ROOT}`",
        ]
    )

    OUT_MD.write_text("\n".join(lines), encoding="utf-8-sig")
    print(f"Wrote: {OUT_MD}")
    print(f"Wrote: {OUT_CSV}")
    print("Top 5 by fold_mean_macro_f1:")
    print(
        ranked[
            [
                "kind",
                "roi_or_input",
                "backbone",
                "fold_mean_macro_f1",
                "fold_mean_macro_auc",
                "fold_mean_balanced_accuracy",
            ]
        ]
        .head(5)
        .to_string(index=False)
    )


if __name__ == "__main__":
    main()
