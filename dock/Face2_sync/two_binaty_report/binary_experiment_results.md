# E0B：对照与 NYHA I–IV 患者二分类结果

## 目的与医学任务

本实验评估单张普通面部照片对**无病正常对照**与 **NYHA I–IV 患者**的区分能力。标签 0 是额外设置的无病正常对照，并不表示“NYHA 0 级”。

## 数据范围与标签

仅使用 `data/processed/splits_500` 固定患者级五折划分及 mean-background Global 图像。原 E0 三分类标签映射为 Normal(0)→Control(0)，Mild(1)→Patient(1)，Severe(2)→Patient(1)；未纳入额外 22 例。

## 代码复用与固定训练设置

复用 E0 的 Global 224×224 RGB、ImageNet 预训练 ResNet18、全量微调、ImageNet 标准化、训练期随机水平翻转、AdamW（lr=1e-4，weight_decay=1e-4）、batch size 16、最多 50 epoch、patience 10、seed 2026、无 AMP。仅将分类头改为 Linear(512,2)，并以各训练折自身标签计算加权交叉熵。模型选择指标为验证集 macro-AUC；推理为 softmax 后 argmax，不作阈值搜索。

## 各折结果

| Fold | Macro-AUC | Accuracy | Macro-Precision | Macro-Recall | Macro-F1 | Balanced Accuracy | Best Epoch | Normal N | Patient N |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 0.8458 | 0.7300 | 0.6917 | 0.7637 | 0.6923 | 0.7637 | 4 | 92 | 308 |
| 1 | 0.8876 | 0.8400 | 0.7733 | 0.7894 | 0.7807 | 0.7894 | 20 | 92 | 308 |
| 2 | 0.8577 | 0.8100 | 0.7344 | 0.7547 | 0.7432 | 0.7547 | 9 | 92 | 308 |
| 3 | 0.8577 | 0.8300 | 0.7625 | 0.7372 | 0.7482 | 0.7372 | 9 | 92 | 308 |
| 4 | 0.8837 | 0.8200 | 0.7484 | 0.7764 | 0.7600 | 0.7764 | 7 | 92 | 308 |

## 完整 OOF 指标（500 例合并后直接计算）

| Metric | Value |
|---|---:|
| macro_auc | 0.8550 |
| accuracy | 0.8060 |
| macro_precision | 0.7321 |
| macro_recall | 0.7643 |
| macro_f1 | 0.7447 |
| balanced_accuracy | 0.7643 |

## OOF 混淆矩阵

行是真实标签、列是预测标签，顺序为 [0=Control, 1=Patient]。

| True \ Pred | Control | Patient |
|---|---:|---:|
| Control | 79 | 36 |
| Patient | 61 | 324 |

## 原始三分类组的患者概率（描述性）

| Group | N | Mean | SD | Median | Q1 | Q3 | Min | Max |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Normal | 115 | 0.3511 | 0.3401 | 0.2328 | 0.0444 | 0.6747 | 0.0004 | 0.9977 |
| Mild | 237 | 0.8097 | 0.3127 | 0.9852 | 0.7372 | 0.9988 | 0.0004 | 1.0000 |
| Severe | 148 | 0.8242 | 0.3117 | 0.9928 | 0.8309 | 0.9989 | 0.0017 | 1.0000 |

## 稳定性、类别倾向与限制

各折指标的离散程度见 `fold_metrics.csv`；应结合 macro-F1 与 balanced accuracy 判断是否存在患者多数类偏向，不能只依据 accuracy。结果为固定 500 例的内部五折 OOF 评估，患者组占多数，未做独立外部验证、阈值优化、校准、PR-AUC 或置信区间分析。

## 结论

本结果仅用于描述模型对**无病正常对照与 NYHA I–IV 患者的区分能力**，不代表能够诊断心力衰竭、识别 NYHA 严重程度或替代临床 NYHA 评估。
