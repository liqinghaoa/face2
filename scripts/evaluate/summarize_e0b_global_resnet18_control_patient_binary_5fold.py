"""OOF integrity checks and Chinese reports for the completed E0B experiment."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from datasets.control_patient_binary_dataset import map_three_class_to_binary
from metrics.binary_classification_metrics import compute_binary_metrics, flatten_metrics
from utils.experiment_utils import load_yaml

METRICS = ["macro_auc", "accuracy", "macro_precision", "macro_recall", "macro_f1", "balanced_accuracy"]
GROUP_NAMES = {0: "Normal", 1: "Mild", 2: "Severe"}


def _format_image_size(image_size: object) -> str:
    if isinstance(image_size, (list, tuple)) and len(image_size) == 2:
        height, width = int(image_size[0]), int(image_size[1])
        return f"{width}×{height}"
    size = int(image_size)
    return f"{size}×{size}"


def _training_setting_sentence(config: dict) -> str:
    data = config["data"]
    model = config["model"]
    train = config["train"]
    strategy = model.get("trainability_strategy", "full_finetune")
    dropout = model.get("dropout", None)
    gradient_clip = train.get("gradient_clip_max_norm", None)
    bn_mode = train.get("batchnorm_mode", "default")
    extras = []
    if dropout is not None:
        extras.append(f"dropout={dropout}")
    if gradient_clip is not None:
        extras.append(f"gradient clipping max_norm={gradient_clip}")
    if bn_mode != "default":
        extras.append(f"BatchNorm 模式：{bn_mode}")
    extras_text = "，" + "，".join(extras) if extras else ""
    return (
        f"复用 E0 的 Global { _format_image_size(data['image_size']) } RGB、"
        f"ImageNet 预训练 ResNet18、训练策略 `{strategy}`、ImageNet 标准化、"
        f"训练期随机水平翻转、AdamW（lr={train['lr']}，"
        f"weight_decay={train['weight_decay']}）、batch size {train['batch_size']}、"
        f"最多 {train['epochs']} epoch、patience {train['early_stopping_patience']}、"
        f"seed {train['random_seed']}、AMP={train.get('use_amp', False)}{extras_text}。"
        "分类头为 Linear(512,2)，并以各训练折自身标签计算加权交叉熵。"
        "模型选择指标为验证集 macro-AUC；推理为 softmax 后 argmax，不作阈值搜索。"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment-dir", type=Path, required=True)
    return parser.parse_args()


def _expected_labels(config: dict) -> pd.DataFrame:
    root = PROJECT_ROOT / config["data"]["split_dir"]
    frames: list[pd.DataFrame] = []
    for fold in range(int(config["data"]["n_folds"])):
        val = pd.read_csv(root / config["data"]["val_csv_pattern"].format(fold=fold), dtype={"ID": "string"})
        frames.append(val.loc[:, ["ID", "label_3class"]].assign(binary_label=lambda x: x["label_3class"].map(map_three_class_to_binary)))
    expected = pd.concat(frames, ignore_index=True).rename(columns={"ID": "sample_id"})
    expected["sample_id"] = expected["sample_id"].astype(str)
    if expected["sample_id"].duplicated().any():
        raise ValueError("Fixed validation split IDs are not unique")
    return expected.loc[:, ["sample_id", "binary_label"]]


def validate_oof(oof: pd.DataFrame, config: dict) -> None:
    expected_labels = _expected_labels(config); expected = set(expected_labels["sample_id"]); observed = set(oof["sample_id"].astype(str))
    required = {"sample_id", "patient_group_id", "fold", "binary_label", "logit_normal", "logit_patient", "prob_normal", "prob_patient", "pred_class", "image_path", "selected_epoch", "checkpoint_path"}
    missing = required.difference(oof.columns)
    if missing: raise ValueError(f"OOF lacks required fields: {sorted(missing)}")
    if len(oof) != len(expected) or oof["sample_id"].duplicated().any() or observed != expected:
        raise ValueError(f"OOF ID integrity failed: rows={len(oof)}, expected={len(expected)}, missing={sorted(expected-observed)[:5]}, extra={sorted(observed-expected)[:5]}")
    if set(oof["binary_label"].unique()) - {0, 1}: raise ValueError("OOF labels are not binary")
    probs = oof[["prob_normal", "prob_patient"]].to_numpy(float)
    if not np.isfinite(probs).all() or (probs < 0).any() or (probs > 1).any() or not np.allclose(probs.sum(axis=1), 1, atol=1e-5): raise ValueError("OOF probabilities invalid")
    if not np.array_equal(oof["pred_class"].to_numpy(int), probs.argmax(axis=1)): raise ValueError("OOF predictions are not probability argmax")
    if set(oof["fold"].unique()) != set(range(int(config["data"]["n_folds"]))): raise ValueError("OOF fold coverage is incomplete")
    mapped = oof["original_three_class_label"].map(map_three_class_to_binary).to_numpy(int)
    if not np.array_equal(mapped, oof["binary_label"].to_numpy(int)): raise ValueError("OOF binary label mapping is inconsistent")
    expected_lookup = expected_labels.set_index("sample_id")["binary_label"]
    expected_binary = oof["sample_id"].astype(str).map(expected_lookup).to_numpy(int)
    if not np.array_equal(expected_binary, oof["binary_label"].to_numpy(int)):
        raise ValueError("OOF binary label distribution does not match the audited fixed splits")


def _probability_groups(oof: pd.DataFrame) -> pd.DataFrame:
    records = []
    for label, name in GROUP_NAMES.items():
        values = oof.loc[oof["original_three_class_label"] == label, "prob_patient"].astype(float)
        records.append({"original_group": name, "N": len(values), "mean": values.mean(), "std": values.std(ddof=1), "median": values.median(), "Q1": values.quantile(.25), "Q3": values.quantile(.75), "min": values.min(), "max": values.max()})
    return pd.DataFrame(records)


def _results_report(config: dict, fold_metrics: pd.DataFrame, metrics: dict, matrix: np.ndarray, groups: pd.DataFrame) -> str:
    rows = "\n".join(f"| {int(r.fold)} | {r.macro_auc:.4f} | {r.accuracy:.4f} | {r.macro_precision:.4f} | {r.macro_recall:.4f} | {r.macro_f1:.4f} | {r.balanced_accuracy:.4f} | {int(r.best_epoch)} | {int(r.train_normal_count)} | {int(r.train_patient_count)} |" for r in fold_metrics.itertuples())
    result_rows = "\n".join(f"| {key} | {metrics[key]:.4f} |" for key in METRICS)
    group_rows = "\n".join(
        f"| {row['original_group']} | {int(row['N'])} | {row['mean']:.4f} | "
        f"{row['std']:.4f} | {row['median']:.4f} | {row['Q1']:.4f} | "
        f"{row['Q3']:.4f} | {row['min']:.4f} | {row['max']:.4f} |"
        for _, row in groups.iterrows()
    )
    split_dir = config["data"]["split_dir"]
    image_root = config["data"]["image_root"]
    settings = _training_setting_sentence(config)
    return f"""# E0B：对照与 NYHA I–IV 患者二分类结果\n\n## 目的与医学任务\n\n本实验评估单张普通面部照片对**无病正常对照**与 **NYHA I–IV 患者**的区分能力。标签 0 是额外设置的无病正常对照，并不表示“NYHA 0 级”。\n\n## 数据范围与标签\n\n仅使用 `{split_dir}` 固定患者级五折划分及 `{image_root}` 图像。原 E0 三分类标签映射为 Normal(0)→Control(0)，Mild(1)→Patient(1)，Severe(2)→Patient(1)；未纳入额外 22 例。\n\n## 代码复用与固定训练设置\n\n{settings}\n\n## 各折结果\n\n| Fold | Macro-AUC | Accuracy | Macro-Precision | Macro-Recall | Macro-F1 | Balanced Accuracy | Best Epoch | Normal N | Patient N |\n|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|\n{rows}\n\n## 完整 OOF 指标（500 例合并后直接计算）\n\n| Metric | Value |\n|---|---:|\n{result_rows}\n\n## OOF 混淆矩阵\n\n行是真实标签、列是预测标签，顺序为 [0=Control, 1=Patient]。\n\n| True \\ Pred | Control | Patient |\n|---|---:|---:|\n| Control | {matrix[0,0]} | {matrix[0,1]} |\n| Patient | {matrix[1,0]} | {matrix[1,1]} |\n\n## 原始三分类组的患者概率（描述性）\n\n| Group | N | Mean | SD | Median | Q1 | Q3 | Min | Max |\n|---|---:|---:|---:|---:|---:|---:|---:|---:|\n{group_rows}\n\n## 稳定性、类别倾向与限制\n\n各折指标的离散程度见 `fold_metrics.csv`；应结合 macro-F1 与 balanced accuracy 判断是否存在患者多数类偏向，不能只依据 accuracy。结果为固定 500 例的内部五折 OOF 评估，患者组占多数，未做独立外部验证、阈值优化、校准、PR-AUC 或置信区间分析。\n\n## 结论\n\n本结果仅用于描述模型对**无病正常对照与 NYHA I–IV 患者的区分能力**，不代表能够诊断心力衰竭、识别 NYHA 严重程度或替代临床 NYHA 评估。\n"""


def _implementation_report(config: dict, exp: Path) -> str:
    experiment_name = config.get("experiment", {}).get("name", exp.name)
    image_size = _format_image_size(config["data"]["image_size"])
    image_root = config["data"]["image_root"]
    strategy = config["model"].get("trainability_strategy", "full_finetune")
    return f"""# E0B 实现报告\n\n## 已审阅的 E0 实现\n\n- 训练入口：`scripts/train/train_nyha_3class_5fold.py`\n- 配置：`config/train/nyha_3class_global224_imagenet_resnet18.yaml`\n- Dataset/变换：`datasets/nyha_3class_face_dataset.py`\n- 模型工厂：`models/nyha_backbone_factory.py` 与 `models/resnet_nyha_3class.py`\n- 类别权重/loss：`losses/classification_losses.py`\n- Trainer/Evaluator：`trainers/nyha_3class_trainer.py`、`evaluators/nyha_3class_evaluator.py`\n- 五折汇总：`scripts/evaluate/summarize_nyha_3class_5fold.py`\n\n## 本次 E0B 文件及目的\n\n- `config_snapshot.yaml`：本次实验 `{experiment_name}` 的运行时配置快照，输入为 `{image_root}`，尺寸为 {image_size}，训练策略为 `{strategy}`。\n- `datasets/control_patient_binary_dataset.py`：不修改 CSV 的三分类→二分类动态映射及元数据保留。\n- `metrics/binary_classification_metrics.py`：以患者概率计算二分类 ROC-AUC 和六项指标。\n- `utils/e0b_binary_audit.py`：训练前图像、分组泄漏、OOF 覆盖和标签冲突审计。\n- `scripts/train/train_e0b_global_resnet18_control_patient_binary_5fold.py`：独立训练、checkpoint、预测和每折产物。\n- `scripts/evaluate/summarize_e0b_global_resnet18_control_patient_binary_5fold.py`：OOF 完整性检查、汇总和中文报告。\n- `tests/test_e0b_control_patient_binary.py`：映射、非法标签、二分类指标、概率、权重和矩形输入变换测试。\n\nE0 三分类源码与既有结果未被修改。二分类映射位于 `map_three_class_to_binary`，分类头经 E0 的模型工厂传入 `num_classes=2` 构建，类别权重由 E0 的 `compute_class_weights(..., num_classes=2)` 基于各训练折单独计算。\n\n## 验证与运行\n\n已执行配置解析、数据审计、Dataset/模型/指标测试、矩形输入变换测试和独立单折 1 epoch 烟雾测试。正式运行命令：\n\n```powershell\npython scripts/train/train_e0b_global_resnet18_control_patient_binary_5fold.py --config path/to/config.yaml --output-dir {exp}\npython scripts/evaluate/summarize_e0b_global_resnet18_control_patient_binary_5fold.py --experiment-dir {exp}\n```\n\n正式结果目录：`{exp}`。训练环境、开始/完成时间保存在 `environment.json` 和 `run_finished_at.txt`。\n"""


def main() -> Path:
    args = parse_args(); exp = args.experiment_dir.resolve(); config = load_yaml(exp / "config_snapshot.yaml")
    prediction_frames, metric_frames = [], []
    for fold in range(int(config["data"]["n_folds"])):
        prediction_frames.append(pd.read_csv(exp / f"fold_{fold}" / "val_predictions.csv", dtype={"sample_id": "string", "patient_group_id": "string"}))
        metric_frames.append(pd.read_csv(exp / f"fold_{fold}" / "metrics.csv"))
    oof = pd.concat(prediction_frames, ignore_index=True).sort_values(["fold", "sample_id"], kind="stable"); validate_oof(oof, config)
    oof.to_csv(exp / "oof_predictions.csv", index=False, encoding="utf-8-sig")
    metrics = compute_binary_metrics(oof["binary_label"].to_numpy(), oof[["prob_normal", "prob_patient"]].to_numpy())
    pd.DataFrame([flatten_metrics(metrics)]).to_csv(exp / "oof_metrics.csv", index=False, encoding="utf-8-sig")
    matrix = metrics["confusion_matrix"]; pd.DataFrame(matrix, index=["control", "patient"], columns=["control", "patient"]).to_csv(exp / "oof_confusion_matrix.csv", index_label="true\\pred", encoding="utf-8-sig")
    fold_metrics = pd.concat(metric_frames, ignore_index=True).sort_values("fold"); fold_metrics.to_csv(exp / "fold_metrics.csv", index=False, encoding="utf-8-sig")
    groups = _probability_groups(oof); groups.to_csv(exp / "prob_patient_by_original_group.csv", index=False, encoding="utf-8-sig")
    (exp / "binary_experiment_results.md").write_text(_results_report(config, fold_metrics, metrics, matrix, groups), encoding="utf-8")
    implementation = _implementation_report(config, exp)
    implementation += (
        "\n\n## 一键正式运行入口\n\n"
        "也可使用 `scripts/run/run_e0b_global_resnet18_control_patient_binary_5fold.py`，"
        "它会依次执行训练和 OOF 汇总，避免遗漏最终报告步骤。\n"
    )
    (exp / "implementation_report.md").write_text(implementation, encoding="utf-8")
    print(f"SUMMARY_DIR={exp}"); return exp


if __name__ == "__main__": main()
