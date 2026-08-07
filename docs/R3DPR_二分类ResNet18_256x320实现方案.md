# R3DPR 二分类 ResNet18 256x320 实现方案

## 1. 目的与边界

本方案用于重新建立一套独立、干净、可追溯的 R3DPR 二分类训练管线。新管线不修改旧 E0B 的代码、配置或结果，也不复用旧 E0B 的三分类到二分类标签映射逻辑。

任务定义为 Control vs Patient 二分类。输入为 R3DPR 已整理的人脸 RGB 图像，尺寸为 256x320；模型为 ImageNet 预训练 ResNet18。

## 2. 固定数据协议

五折划分表：

`E:\projects\face2\data\processed\global_face_R3DPR\nyha_2class_sex_stratified_group_5fold.csv`

该表已删除三分类标签列，仅保留二分类标签。因此新 Dataset 必须直接读取现有二分类标签，禁止：

- 根据 NYHA 或其他列重新推断标签；
- 使用 `label_3class`；
- 调用旧 E0B 的 `map_three_class_to_binary`；
- 修改原始 CSV。

正式实现前必须读取 CSV 表头，并在配置中明确指定二分类标签列名。Dataset 最低要求的字段为：

| 字段 | 用途 |
|---|---|
| `ID` | 图像文件名定位和 OOF 样本标识 |
| `patient_group_id` | 患者级泄漏审计 |
| `fold` | 五折划分标识 |
| 二分类标签列 | 直接作为 0/1 监督标签 |

若实际列名不同，应在新配置中显式声明，不应在代码中硬编码猜测。

## 3. 单表五折训练协议

新表为一个包含全部样本及 `fold` 列的单一 CSV，不再要求旧管线使用的 `fold_{k}_train.csv` 与 `fold_{k}_val.csv`。

对每个 `k in {0, 1, 2, 3, 4}`：

```text
训练集：fold != k
验证集：fold == k
```

正式训练前必须执行数据审计：

1. `ID` 非空且唯一，图像文件存在。
2. `fold` 仅包含 0--4，每个样本只出现一次。
3. 每一折训练集和验证集均同时具备两类样本。
4. 同一 `patient_group_id` 不得同时存在于某折训练集和验证集。
5. 记录各折二分类数量和性别分布，验证其与性别分层分组五折设计一致。
6. 记录数据表总样本数及每折验证样本数，作为 OOF 覆盖基准。

## 4. 建议的新代码目录

在每类程序原有上一级目录下创建 `R3DPR` 子目录：

```text
config/train/R3DPR/
  r3dpr_resnet18_binary_256x320_5fold.yaml

datasets/R3DPR/
  binary_face_dataset.py

models/R3DPR/
  resnet18_binary.py

metrics/R3DPR/
  binary_classification_metrics.py

utils/R3DPR/
  binary_data_audit.py

scripts/train/R3DPR/
  train_r3dpr_resnet18_binary_5fold.py

scripts/evaluate/R3DPR/
  summarize_r3dpr_resnet18_binary_5fold.py

scripts/run/R3DPR/
  run_r3dpr_resnet18_binary_5fold.py

tests/R3DPR/
  test_r3dpr_binary_pipeline.py
```

R3DPR 代码应独立维护，以免旧 E0B 的数据格式、标签映射或历史实验策略混入新实验。可继续使用项目通用的配置读取、日志和随机种子工具；不需要重复实现 torchvision 的 ResNet18 基础模块。

## 5. 256x320 图像支持

图像根目录由配置文件显式指定，不能沿用旧 E0B 的输入目录。配置建议明确写为：

```yaml
data:
  image_root: <R3DPR 256x320 image directory>
  image_filename_template: "{ID}.png"
  image_height: 256
  image_width: 320
```

训练和验证预处理均使用 `Resize((256, 320))`，使送入模型的张量固定为 `[3, 256, 320]`。训练期可使用随机水平翻转，验证期不使用随机增强；两者均采用 ImageNet mean/std 标准化。

标准 torchvision ResNet18 支持非正方形输入：其卷积层保留空间结构，最终通过自适应全局平均池化得到固定长度特征。因此无需为了 256x320 修改 ResNet18 的卷积或分类层结构。

## 6. 模型与 baseline 训练定义

新模型文件应清晰实现：

```text
torchvision ResNet18 (ImageNet-1K pretrained)
  -> replace fc: Linear(512, 1000) to Linear(512, 2)
  -> logits for [Control, Patient]
```

新 baseline 的默认策略：

- 全参数微调（full finetune）；
- `CrossEntropyLoss`，类别权重仅基于当前训练折计算；
- AdamW；
- 以验证集 Macro-AUC 选择最佳 checkpoint；
- early stopping；
- 不做阈值搜索；
- 推理使用 softmax 后的 `argmax` 得到硬分类。

若后续评估 Dropout、BN 冻结、梯度裁剪、学习率调度或其他抗过拟合策略，应通过新配置创建独立实验，不得改变 baseline 的默认含义。

## 7. GPU 和可复现性要求

训练程序启动后必须检查 CUDA 可用性。若 CUDA 不可用，程序应明确报错退出，禁止静默回退至 CPU。

训练环境为：

`E:\resarch\Anaconda3\envs\face2\python.exe`

每次运行应写入：

- 配置快照；
- Python、PyTorch、torchvision、CUDA 与 GPU 信息；
- 随机种子；
- 每折类别权重和参数数量；
- 每 epoch 训练损失和验证指标；
- 最佳 checkpoint 所在 epoch。

## 8. OOF 汇总和评价协议

每一折只使用该折验证集预测。五折完成后，合并为覆盖全部样本且无重复 ID 的 pooled OOF 预测表。

主评价固定为 pooled OOF 指标：

| 指标 | 定义 |
|---|---|
| Macro-AUC | 以 Patient 概率计算 ROC-AUC |
| Macro-F1 | 两类 F1 的宏平均 |
| Balanced Accuracy | Sensitivity 和 Specificity 的平均 |
| Sensitivity | Patient 为阳性类，TP / (TP + FN) |
| Specificity | Control 为阴性类，TN / (TN + FP) |

汇总程序必须先验证：OOF ID 完整覆盖原始单表、无重复样本、各折覆盖完整、标签与 CSV 一致、概率合法且两类概率之和为 1。之后输出：

```text
fold_0 ... fold_4/
  training_history.csv
  metrics.csv
  val_predictions.csv
  checkpoints/best_macro_auc.pth

config_snapshot.yaml
environment.json
data_audit_report.md
fold_metrics.csv
oof_predictions.csv
oof_metrics.csv
oof_confusion_matrix.csv
binary_experiment_results.md
```

论文或横向比较以 pooled OOF 为主，不以单折最高值或训练集指标作为主结论。

## 9. 建议的结果目录与启动入口

建议 baseline 结果目录：

`E:\projects\face2\experiments\500Data\R3DPR_ResNet18_ControlVsPatient_Binary_256x320_5fold`

`scripts/run/R3DPR/run_r3dpr_resnet18_binary_5fold.py` 应是唯一的一键正式入口，按顺序调用训练和 OOF 汇总，避免遗漏最终评价步骤。

## 10. 实现前待确认项

代码实现前需要从实际文件确认并写入配置的仅有两项：

1. 五折 CSV 中二分类标签列的精确列名。
2. R3DPR 256x320 图像输入根目录及文件扩展名/命名模板。

在这两项确认后，即可按照本文档建立独立的 R3DPR 二分类 baseline 管线。
