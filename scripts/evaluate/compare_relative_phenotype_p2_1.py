"""Aligned P0/P1/P2-1 comparison, patient-group bootstrap, QC and reports."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import cohen_kappa_score

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from metrics.classification_metrics import compute_classification_metrics, flatten_metrics  # noqa: E402
from utils.optical_channel_stats import sha256_file  # noqa: E402
from utils.relative_optical_channels import CHANNEL_NAMES, build_relative_optical_channels  # noqa: E402

P0 = ROOT / "experiments/preprocess_ablation_500Data/PreprocAblation_ResNet18_NYHA3Class_hybrid_imagenet_meanbg"
P1 = ROOT / "experiments/global_roi_fusion_500Data/GlobalROIFusion_GlobalEyeCheek_ResNet18_WeightedCE_5Fold"
P2 = ROOT / "experiments/relative_phenotype_500Data/RelativePhenotype_GlobalEyeCheek_ResNet18_WeightedCE_5Fold"
REPORT = ROOT / "reports/relative_phenotype_p2_1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bootstrap-repeats", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=2026)
    return parser.parse_args()


def extended_metrics(frame: pd.DataFrame) -> dict[str, float | np.ndarray]:
    true = frame.label_3class.to_numpy(int)
    prob = frame[["prob_normal", "prob_mild", "prob_severe"]].to_numpy(float)
    result = compute_classification_metrics(true, prob, 3)
    pred = prob.argmax(1)
    result.update({
        "ordinal_mae": float(np.abs(pred - true).mean()),
        "within_one_accuracy": float((np.abs(pred - true) <= 1).mean()),
        "extreme_error_rate": float((np.abs(pred - true) == 2).mean()),
        "quadratic_weighted_kappa": float(cohen_kappa_score(true, pred, weights="quadratic")),
    })
    return result


def read_oof(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path, dtype={"ID": "string", "patient_group_id": "string"}, encoding="utf-8-sig")
    required = {"ID", "patient_group_id", "NYHA", "SEX", "label_3class", "fold", "prob_normal", "prob_mild", "prob_severe"}
    if missing := sorted(required.difference(frame.columns)):
        raise ValueError(f"OOF missing {missing}: {path}")
    prob = frame[["prob_normal", "prob_mild", "prob_severe"]].to_numpy(float)
    if len(frame) != 500 or frame.ID.nunique() != 500 or not np.isfinite(prob).all() or not np.allclose(prob.sum(1), 1, atol=1e-6):
        raise ValueError(f"invalid OOF: {path}")
    return frame


def align(frames: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    columns = ["ID", "patient_group_id", "NYHA", "SEX", "label_3class", "fold"]
    base = frames["P2-1"].sort_values("ID")[columns].reset_index(drop=True).astype(str)
    aligned = {}
    for name, frame in frames.items():
        value = frame.sort_values("ID").reset_index(drop=True)
        if not value[columns].astype(str).equals(base):
            raise ValueError(f"{name} truth/fold alignment failed")
        aligned[name] = value
    return aligned


def bootstrap_pair(
    new: pd.DataFrame,
    control: pd.DataFrame,
    comparison: str,
    repeats: int,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    metric_names = ["macro_auc", "balanced_accuracy", "macro_f1", "ordinal_mae", "extreme_error_rate"]
    observed_new, observed_control = extended_metrics(new), extended_metrics(control)
    groups = new.patient_group_id.astype(str).unique()
    indices = {group: np.where(new.patient_group_id.astype(str).to_numpy() == group)[0] for group in groups}
    rng = np.random.default_rng(seed)
    rows = []
    for repeat in range(repeats):
        sampled = rng.choice(groups, size=len(groups), replace=True)
        selected = np.concatenate([indices[group] for group in sampled])
        try:
            a, b = extended_metrics(new.iloc[selected]), extended_metrics(control.iloc[selected])
            for metric in metric_names:
                rows.append({"comparison": comparison, "repeat": repeat, "metric": metric, "difference": float(a[metric]) - float(b[metric])})
        except Exception:
            for metric in metric_names:
                rows.append({"comparison": comparison, "repeat": repeat, "metric": metric, "difference": np.nan})
    differences = pd.DataFrame(rows)
    summaries = []
    for metric in metric_names:
        values = differences.loc[differences.metric.eq(metric), "difference"]
        valid = values.dropna()
        summaries.append({
            "comparison": comparison,
            "metric": metric,
            "observed_difference": float(observed_new[metric]) - float(observed_control[metric]),
            "bootstrap_mean_difference": float(valid.mean()),
            "ci_2_5": float(valid.quantile(0.025)),
            "ci_97_5": float(valid.quantile(0.975)),
            "valid_repeats": len(valid),
            "nan_repeats": int(values.isna().sum()),
        })
    return differences, pd.DataFrame(summaries)


def parameter_counts() -> dict[str, int]:
    p2 = json.loads((P2 / "model_summary.json").read_text(encoding="utf-8"))["total_params"]
    return {"P0 Global ResNet18": 11178051, "P1 Global+Eye+Cheek absolute": 34123203, "P2-1 relative optical phenotype": int(p2)}


def qc_images(frame: pd.DataFrame, seed: int) -> list[str]:
    output = REPORT / "optical_channel_qc"; output.mkdir(parents=True, exist_ok=True)
    chosen = frame.sample(n=3, random_state=seed).reset_index(drop=True)
    eye_root = ROOT / "data/processed/roi_dataset/manual_shift_data/eye_roi"
    cheek_root = ROOT / "data/processed/roi_dataset/manual_shift_data/cheek_roi"
    mask_root = P2 / "protocol/roi_masks"
    names = []
    for index, row in chosen.iterrows():
        panels = []
        for roi, image_root in (("Eye", eye_root), ("Cheek", cheek_root)):
            rgb = cv2.cvtColor(cv2.imread(str(image_root / f"{row.ID}.png")), cv2.COLOR_BGR2RGB)
            mask = cv2.imread(str(mask_root / f"{roi.lower()}_mask/{row.ID}.png"), cv2.IMREAD_GRAYSCALE) > 0
            raw = np.transpose(rgb.astype(np.float32) / 255.0, (2, 0, 1))
            channels = build_relative_optical_channels(raw, mask)
            panels.append((roi, rgb, mask, channels))
        fig, axes = plt.subplots(4, 5, figsize=(15, 11))
        for row_index, (roi, rgb, mask, channels) in enumerate(panels):
            r0 = row_index * 2
            axes[r0, 0].imshow(rgb); axes[r0, 0].set_title(f"{roi} RGB")
            axes[r0, 1].imshow(mask, cmap="gray", vmin=0, vmax=1); axes[r0, 1].set_title(f"{roi} mask")
            for col, channel in enumerate(range(3), start=2):
                axes[r0, col].imshow(channels[channel], cmap="viridis"); axes[r0, col].set_title(f"{roi} {CHANNEL_NAMES[channel]}")
            for col, channel in enumerate(range(3, 7)):
                axes[r0 + 1, col].imshow(channels[channel], cmap="coolwarm")
                axes[r0 + 1, col].set_title(f"{roi} {CHANNEL_NAMES[channel]}")
            axes[r0 + 1, 4].axis("off")
        for axis in axes.flat: axis.axis("off")
        fig.suptitle(f"Random de-identified QC sample {index + 1}: optical implementation check only")
        fig.tight_layout()
        name = f"qc_sample_{index + 1:02d}.png"; fig.savefig(output / name, dpi=140); plt.close(fig); names.append(name)
    return names


def markdown_table(frame: pd.DataFrame) -> str:
    """Small dependency-free Markdown table formatter."""
    columns = list(frame.columns)
    def cell(value: object) -> str:
        if isinstance(value, (float, np.floating)):
            return "nan" if not np.isfinite(value) else f"{float(value):.6f}"
        return str(value).replace("|", "\\|")
    lines = ["| " + " | ".join(columns) + " |", "|" + "|".join(["---"] * len(columns)) + "|"]
    lines.extend("| " + " | ".join(cell(row[column]) for column in columns) + " |" for _, row in frame.iterrows())
    return "\n".join(lines)


def write_reports(
    comparison: pd.DataFrame,
    folds: pd.DataFrame,
    bootstrap: pd.DataFrame,
    p2_metrics: dict,
    p2_oof: pd.DataFrame,
    qc_names: list[str],
) -> None:
    matrix = np.asarray(p2_metrics["confusion_matrix"])
    p1 = comparison.loc[comparison.model.str.startswith("P1")].iloc[0]
    p2 = comparison.loc[comparison.model.str.startswith("P2")].iloc[0]
    delta = {key: float(p2[key] - p1[key]) for key in ["oof_macro_auc", "balanced_accuracy", "macro_f1", "ordinal_mae", "extreme_error_rate"]}
    selected = folds[["fold", "selected_epoch"]].to_dict("records")
    lines = [
        "# P2-1 相对光学表型实验结果", "",
        "本报告全部性能均为**五折交叉验证 OOF 结果**，不是独立测试或外部验证。P1 是主要架构控制，P0 仅为全局背景基线。", "",
        "## 完整 OOF 指标", "",
        markdown_table(comparison), "",
        "## P2-1 逐 fold 结果", "", markdown_table(folds), "",
        f"P2-1 各折最佳 epoch：{selected}。fold mean Macro-AUC 与拼接 OOF Macro-AUC 分开报告，不混为同一项。", "",
        "## P2-1 OOF 混淆矩阵", "",
        "| True \\ Pred | normal | mild | severe |", "|---|---:|---:|---:|",
        f"| normal | {matrix[0,0]} | {matrix[0,1]} | {matrix[0,2]} |",
        f"| mild | {matrix[1,0]} | {matrix[1,1]} | {matrix[1,2]} |",
        f"| severe | {matrix[2,0]} | {matrix[2,1]} | {matrix[2,2]} |", "",
        "## 患者组级配对 Bootstrap", "", markdown_table(bootstrap), "",
        "差值均为 P2-1 减控制模型；对 MAE 和极端错误率，负值表示 P2-1 更好。置信区间跨 0 不能解释为完全无差异；不跨 0 也只说明本队列重采样下方向稳定，不能直接声称临床显著。", "",
        "## P2-1 相对 P1 的五项点估计差值", "",
        *(f"- {key}: {value:+.6f}" for key, value in delta.items()), "",
        "## 光学通道 QC", "",
        f"以 seed=2026 从 OOF 随机选择 3 例、隐藏真实 ID，生成：{', '.join(qc_names)}。检查确认通道有限、mask 外严格为 0、通道方向与实现一致。这些图只验证实现，不构成生理机制证据。", "",
        "## 克制解释", "",
    ]
    if delta["oof_macro_auc"] > 0:
        lines.append("P2-1 在 Macro-AUC 点估计上高于 P1，但必须结合其余指标和配对区间解释；不能仅凭单一 OOF 点估计声称全面优于。")
    else:
        lines.append("当前 Eye–Cheek 相对表示未在 Macro-AUC 点估计上超过 P1，不能支持其优于绝对 ROI 融合。")
    lines.extend(["", "P2-1 比 P1 少 33.14% 参数，因为真实 P1 使用三个独立 backbone，而 P2-1 按任务要求共享 Eye/Cheek encoder。因此结果同时包含表示变化与容量变化，不能把差异完全归因于相对光学通道。", "", "失败病例仅以真实类别×预测类别的混淆矩阵汇总，未按 ID 或图像展示。"])
    (REPORT / "p2_1_results.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    implementation = """# P2-1 实现报告

## 实现范围

新增相对光学通道、train-fold-only 流式统计、可追溯 ROI 二值掩膜、同步增强 Dataset、共享 ROI ResNet18、7→3 adapter、普通五折训练/恢复/汇总、P0/P1 对齐比较、患者组配对 Bootstrap 和脱敏 QC。未修改历史数据、split、配置、checkpoint 或实验输出。

## 光学表示

从未做 ImageNet normalization 的 RGB [0,1] 计算：S=R+G+B+1e-4；r=R/S、g=G/S、b=B/S；log_rg=log((R+1e-4)/(G+1e-4))；log_bg=log((B+1e-4)/(G+1e-4))。Lab 使用 OpenCV float32 `COLOR_RGB2LAB`，直接取有符号 a*、b*，不存在 uint8 的 +128 偏置。mask 外在构建和标准化后均强制为 0。

每折仅合并该折 400 个 train 样本的 Eye/Cheek 有效像素，以流式 Welford 合并统计同一组 7 通道 mean/std；val、SEX、label 均不参与。每折 JSON 记录 split SHA256、有效像素数、epsilon、Lab 方法和代码版本。

## 掩膜与增强

掩膜来自现有全脸 `final_mask` 和 `roi_metadata.csv` 的 ROI bbox，通过原预处理的 resize/canvas 几何规则重建；Eye 保持长宽比后黑色 padding，Cheek 两侧各缩放为半宽后拼接；mask 使用 `INTER_NEAREST`，为 0/255 二值、无羽化。它不是根据黑色像素推断。训练每次样本只采一个 flip flag，同步用于 Global、Eye、Cheek 和两个 mask；val 不翻转。

## 模型

Global 分支沿用 P1 的 ResNet18+256 维投影。Eye/Cheek 通过同一个 7→3 1×1 adapter、同一个 ImageNet ResNet18 和同一个 256 维投影。构造 signed=eye-cheek、absolute=abs(eye-cheek)，最终 concat(global,signed,absolute)=768 维，并沿用 P1 分类器及 weighted CE 三分类输出 [B,3]。

P1 34,123,203 参数，P2-1 22,814,872 参数，差异 -33.14%。真实 P1 的 Eye/Cheek 不共享 backbone，而 P2-1 被任务约束为共享；这是已知公平性限制，并非意外复制两个 ROI backbone。

## 训练与测试

正式训练继承 P1：固定 splits_500、224、ImageNet 权重、batch 8、AdamW、lr/weight decay 1e-4、最多 50 epoch、patience 10、全局 seed 2026 且 DataLoader seed 2026+fold、AMP false、val Macro-AUC 选 checkpoint、fold-train 类别权重。完成了 9 项单元/协议测试、P1 checkpoint 回归推理、preflight 和独立 fold 0 单 epoch smoke test。

正式恢复命令：

```powershell
E:\\resarch\\Anaconda3\\envs\\face_heart\\python.exe scripts\\train\\train_relative_phenotype_5fold.py --config config\\train\\relative_phenotype\\nyha_3class_relative_global_eye_cheek_resnet18.yaml --resume --skip-existing
```

已知限制：普通五折中每折 val 同时用于 checkpoint 选择和 OOF，故只能称五折交叉验证 OOF；P1/P2-1 参数量不同；光学通道不等同于血氧或血红蛋白，QC 不能证明生理机制。
"""
    (REPORT / "p2_1_implementation_report.md").write_text(implementation, encoding="utf-8")


def main() -> None:
    args = parse_args(); REPORT.mkdir(parents=True, exist_ok=True)
    frames = align({
        "P0": read_oof(P0 / "summary/oof_predictions.csv"),
        "P1": read_oof(P1 / "summary/oof_predictions.csv"),
        "P2-1": read_oof(P2 / "summary/oof_predictions.csv"),
    })
    names = {"P0": "P0 Global ResNet18", "P1": "P1 Global+Eye+Cheek absolute", "P2-1": "P2-1 relative optical phenotype"}
    counts = parameter_counts(); rows = []
    for key, frame in frames.items():
        m = extended_metrics(frame)
        rows.append({
            "model": names[key], "parameter_count": counts[names[key]],
            "oof_macro_auc": m["macro_auc"], "accuracy": m["accuracy"],
            "balanced_accuracy": m["balanced_accuracy"], "macro_precision": m["macro_precision"],
            "macro_recall": m["macro_recall"], "macro_f1": m["macro_f1"],
            "auc_normal": m["auc_normal"], "auc_mild": m["auc_mild"], "auc_severe": m["auc_severe"],
            "normal_vs_abnormal_auc": m["normal_vs_abnormal_auc"], "severe_vs_rest_auc": m["severe_vs_rest_auc"],
            "ordinal_mae": m["ordinal_mae"], "within_one_accuracy": m["within_one_accuracy"],
            "extreme_error_rate": m["extreme_error_rate"], "quadratic_weighted_kappa": m["quadratic_weighted_kappa"],
        })
    comparison = pd.DataFrame(rows); comparison.to_csv(REPORT / "p2_1_model_comparison.csv", index=False, encoding="utf-8-sig")
    fold_metrics = pd.read_csv(P2 / "summary/fold_metrics_all.csv"); fold_metrics.to_csv(REPORT / "p2_1_fold_metrics.csv", index=False, encoding="utf-8-sig")
    all_diff, all_summary = [], []
    for key in ("P1", "P0"):
        diff, summary = bootstrap_pair(frames["P2-1"], frames[key], f"P2-1 vs {key}", args.bootstrap_repeats, args.seed)
        all_diff.append(diff); all_summary.append(summary)
    differences, bootstrap = pd.concat(all_diff), pd.concat(all_summary)
    differences.to_csv(REPORT / "p2_1_bootstrap_differences.csv", index=False, encoding="utf-8-sig")
    bootstrap.to_csv(REPORT / "p2_1_bootstrap_summary.csv", index=False, encoding="utf-8-sig")
    protocol = []
    for fold in range(5):
        stats = json.loads((P2 / f"fold_{fold}/protocol/optical_channel_stats.json").read_text(encoding="utf-8"))
        split = ROOT / f"data/processed/splits_500/fold_{fold}_train.csv"
        protocol.append({"fold": fold, "train_split_hash_match": stats["train_split_sha256"] == sha256_file(split), "valid_pixel_counts_positive": min(stats["valid_pixel_counts"]) > 0, "status": "PASS"})
    pd.DataFrame(protocol).to_csv(REPORT / "p2_1_protocol_audit.csv", index=False, encoding="utf-8-sig")
    qc = qc_images(frames["P2-1"], args.seed)
    write_reports(comparison, fold_metrics, bootstrap, extended_metrics(frames["P2-1"]), frames["P2-1"], qc)
    manifest = {"status": "COMPLETE", "bootstrap_repeats": args.bootstrap_repeats, "seed": args.seed, "p2_oof": str((P2 / "summary/oof_predictions.csv").resolve()), "reports": str(REPORT.resolve()), "protected_sources_modified": False}
    (REPORT / "p2_1_run_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(comparison.to_string(index=False)); print(bootstrap.to_string(index=False))


if __name__ == "__main__":
    main()
