# Face2 二分类实验接力文档：E0B 与方案B（anti-v3）

本文档用于新 Thread 继续 Face2 二分类实验研究。当前 Thread 上下文较长，因此这里把已经实现、运行和确认过的 E0B 二分类实验入口、配置、训练逻辑、方案B反过拟合策略、结果目录与注意事项集中整理。

更新时间：2026-08-05  
项目根目录：`E:\projects\face2`  
训练环境：`E:\resarch\Anaconda3\envs\face2\python.exe`  
训练要求：必须使用 GPU，不要使用 CPU 训练。本机环境已确认支持 CUDA，Torch 为 CUDA 版本，GPU 为 NVIDIA GeForce RTX 4060 Laptop GPU。

---

## 1. 当前保留的二分类实验入口

目前只保留 E0B 作为二分类实验入口。此前 P2-A0 相关二分类实验代码和结果已按要求删除，不再作为后续二分类实验入口。

核心入口文件：

- 训练脚本：`E:\projects\face2\scripts\train\train_e0b_global_resnet18_control_patient_binary_5fold.py`
- 运行封装：`E:\projects\face2\scripts\run\run_e0b_global_resnet18_control_patient_binary_5fold.py`
- 汇总脚本：`E:\projects\face2\scripts\evaluate\summarize_e0b_global_resnet18_control_patient_binary_5fold.py`
- 数据集定义：`E:\projects\face2\datasets\control_patient_binary_dataset.py`
- 二分类指标：`E:\projects\face2\metrics\binary_classification_metrics.py`
- 反过拟合工具：`E:\projects\face2\utils\resnet18_anti_overfit.py`
- 数据审计工具：`E:\projects\face2\utils\e0b_binary_audit.py`

推荐后续新增二分类实验时：复制一个 E0B 配置文件，修改 `experiment.name`、`data.image_root`、必要时修改 `data.image_size`，然后仍调用同一个 E0B 训练/汇总入口。

---

## 2. E0B 二分类任务定义

任务：Control vs Patient 二分类。

标签映射逻辑在 `datasets/control_patient_binary_dataset.py`：

- 原三分类标签 `label_3class = 0` → 二分类 `control = 0`
- 原三分类标签 `label_3class = 1` 或 `2` → 二分类 `patient = 1`
- 其他标签会直接报错

固定划分：

- split 目录：`E:\projects\face2\data\processed\splits_500`
- 五折训练：`fold_0` 到 `fold_4`
- 训练 CSV 模板：`fold_{fold}_train.csv`
- 验证 CSV 模板：`fold_{fold}_val.csv`
- 每折训练集统计：control 92，patient 308
- 每折验证集合计组成 500 例 OOF 结果

主要评价协议：

- 每折训练，取验证集 `macro_auc` 最优 checkpoint
- 汇总五折验证集预测为 OOF
- 主结果以 pooled OOF 指标为准
- 不进行 group-level 聚合
- 不进行阈值搜索作为主结果
- 默认硬分类阈值来自 `argmax(prob_control, prob_patient)`

---

## 3. baseline E0B 配置

配置文件：

`E:\projects\face2\config\train\e0b_global_resnet18_control_patient_binary_5fold.yaml`

结果目录：

`E:\projects\face2\experiments\500Data\E0B_Global_ResNet18_ControlVsPatient_Binary_5fold`

输入数据：

`E:\projects\face2\data\processed\global_face\preprocess_ablation\hybrid_imagenet_meanbg\images`

关键设置：

- 输入分辨率：224 × 224
- 背景：meanbg / ImageNet mean 背景预处理版本
- 模型：ImageNet 预训练 ResNet18
- 分类头：2 类
- 训练方式：full finetune，全模型可训练
- loss：weighted cross entropy
- optimizer：AdamW
- learning rate：1e-4
- weight decay：1e-4
- batch size：16
- epochs：50
- early stopping patience：10
- monitor metric：macro_auc
- augmentation：horizontal flip
- normalize：ImageNet mean/std
- AMP：false

baseline 的作用：作为 E0B 原始 full-finetune 对照结果。它主指标较高，但训练曲线存在更明显过拟合风险，因此后来增加了 anti-overfit 方案。

---

## 4. anti-overfit v1 配置

配置文件：

`E:\projects\face2\config\train\e0b_global_resnet18_control_patient_binary_anti_overfit_v1.yaml`

结果目录：

`E:\projects\face2\experiments\500Data\E0B_Global_ResNet18_ControlVsPatient_Binary_anti_overfit_v1`

输入数据同 baseline：

`E:\projects\face2\data\processed\global_face\preprocess_ablation\hybrid_imagenet_meanbg\images`

关键设置：

- 模型：ImageNet 预训练 ResNet18
- 训练策略：`partial_layer4`
- 可训练部分：layer4 + classifier
- 冻结部分：conv1、bn1、layer1、layer2、layer3
- dropout：0.5
- epochs：30
- early stopping patience：5
- optimizer：AdamW
- classifier lr：1e-4
- layer4 lr：1e-5
- weight decay：1e-3
- gradient clipping：max_norm = 1.0
- loss：weighted cross entropy
- monitor metric：macro_auc

v1 的定位：更强地抑制过拟合，但因为冻结了较多特征层，泛化主指标下降较明显。它适合作为“强正则/低容量”对照，不建议直接作为当前最优展示方案。

---

## 5. 方案B / anti-v3 配置

方案B在代码和结果目录中对应：

`anti_overfit_v3_bn_eval_full`

配置文件：

`E:\projects\face2\config\train\e0b_global_resnet18_control_patient_binary_anti_overfit_v3_bn_eval_full.yaml`

结果目录：

`E:\projects\face2\experiments\500Data\E0B_Global_ResNet18_ControlVsPatient_Binary_anti_overfit_v3_bn_eval_full`

输入数据同 baseline：

`E:\projects\face2\data\processed\global_face\preprocess_ablation\hybrid_imagenet_meanbg\images`

关键设置：

- 模型：ImageNet 预训练 ResNet18
- 训练策略：`full_finetune_bn_eval`
- 参数训练：全模型参数仍然可训练
- BatchNorm：训练时强制 eval 模式，冻结 BN running mean / running var 统计
- dropout：0.3
- epochs：30
- early stopping patience：5
- optimizer：AdamW
- lr：1e-4
- classifier lr：1e-4
- weight decay：1e-4
- gradient clipping：max_norm = 1.0
- loss：weighted cross entropy
- monitor metric：macro_auc
- AMP：false

方案B的核心思想：

小样本医学图像训练中，BatchNorm 的 running statistics 容易被小 batch / 折内分布扰动带偏。方案B保留 full finetune 的表达能力，但冻结 BN 统计，配合中等 dropout 和梯度裁剪，试图在“不过度牺牲指标”和“降低训练不稳定/过拟合风险”之间折中。

注意：方案B不是冻结 backbone，也不是只训练分类头；它是 full finetune，只是 BN 统计在训练阶段保持 eval 状态。

---

## 6. 反过拟合工具的实现细节

实现文件：

`E:\projects\face2\utils\resnet18_anti_overfit.py`

当前支持的训练策略：

- `full_finetune`
- `full_finetune_bn_eval`
- `frozen_backbone`
- `partial_layer4`

关键函数：

- `configure_trainability()`：按策略设置哪些参数可训练
- `apply_train_mode()`：每个 epoch 训练前设置模型/模块的 train/eval 状态
- `build_optimizer()`：按策略构建 AdamW 参数组
- `build_trainability_audit()`：输出训练可审计信息

训练审计输出：

每折会生成：

`fold_{fold}\trainability_audit.json`

方案B审计中应看到：

- `strategy = full_finetune_bn_eval`
- `dropout = 0.3`
- `gradient_clip_max_norm = 1.0`
- `trainable_parameter_count = 11177538`
- `frozen_parameter_count = 0`
- `batchnorm_training_modes` 中各 BN 层均为 `false`

这说明方案B是“全参数可训练 + BN统计冻结”，实现符合预期。

---

## 7. 训练脚本输出结构

每个正式实验目录通常包含：

- `config_snapshot.yaml`：运行时配置快照
- `environment.json`：环境信息，包括 torch、torchvision、CUDA、GPU
- `data_audit_report.md`：数据审计报告
- `fold_0` 到 `fold_4`：五折结果
- `fold_metrics.csv`：每折指标汇总
- `oof_predictions.csv`：五折 OOF 预测
- `oof_metrics.csv`：OOF 主指标
- `oof_confusion_matrix.csv`：OOF 混淆矩阵
- `prob_patient_by_original_group.csv`：按原三分类组别汇总 patient 概率
- `binary_experiment_results.md`：实验结果报告
- `implementation_report.md`：实现报告
- `run_finished_at.txt`：完成时间

每折目录中包含：

- `training_parameters.json`
- `trainability_audit.json`
- `training_history.csv`
- `val_predictions.csv`
- `metrics.csv`
- `confusion_matrix.csv`
- `checkpoints\best_macro_auc.pth`
- `checkpoints\last.pth`

---

## 8. 已运行核心实验结果

以下结果均为 pooled OOF 指标，建议作为论文/报告里的主要横向比较依据。

Sensitivity 以 patient 为阳性类：`TP / (TP + FN)`  
Specificity 以 control 为阴性类：`TN / (TN + FP)`

| 实验 | 结果目录 | Macro-AUC | Macro-F1 | Balanced Accuracy | Sensitivity | Specificity |
|---|---|---:|---:|---:|---:|---:|
| E0B baseline full-finetune 224 meanbg | `E0B_Global_ResNet18_ControlVsPatient_Binary_5fold` | 0.855020 | 0.744703 | 0.764257 | 0.841558 | 0.686957 |
| E0B anti-overfit v1 partial-layer4 | `E0B_Global_ResNet18_ControlVsPatient_Binary_anti_overfit_v1` | 0.811632 | 0.689727 | 0.720045 | 0.779221 | 0.660870 |
| E0B 方案B / anti-v3 full-finetune-BN-eval | `E0B_Global_ResNet18_ControlVsPatient_Binary_anti_overfit_v3_bn_eval_full` | 0.858882 | 0.731066 | 0.755167 | 0.823377 | 0.686957 |
| E0B 256 meanbg resolution ablation | `E0B_Global_ResNet18_ControlVsPatient_Binary_256MeanBG_5fold` | 0.845308 | 0.703742 | 0.723885 | 0.812987 | 0.634783 |

核心解读：

- baseline 的 Macro-F1 和 Balanced Accuracy 更高，但训练过程更容易表现出过拟合风险。
- anti-v1 明显抑制模型容量，但主指标下降较多。
- 方案B的 Macro-AUC 略高于 baseline，同时保留较多 full-finetune 表达能力；但 Macro-F1 / Balanced Accuracy 低于 baseline。
- 256 分辨率没有提升，反而下降。说明当前任务不一定受限于 224 分辨率，可能更多受样本量、预处理噪声、折间校准、背景/颜色信息稳定性影响。

---

## 9. 224 vs 256 分辨率实验

256 数据配置文件：

`E:\projects\face2\config\train\e0b_global_resnet18_control_patient_binary_256_meanbg_5fold.yaml`

256 输入数据：

`E:\projects\face2\data\processed\global_face\preprocess_ablation_256\hybrid_imagenet_meanbg\images`

256 结果目录：

`E:\projects\face2\experiments\500Data\E0B_Global_ResNet18_ControlVsPatient_Binary_256MeanBG_5fold`

224 vs 256 对比报告：

`E:\projects\face2\experiments\500Data\E0B_Resolution_224_vs_256_MeanBG_Comparison\resolution_comparison_report.md`

与 224 baseline 相比，256 的 pooled OOF 变化：

- Macro-AUC：-0.009712
- Macro-F1：-0.040961
- Balanced Accuracy：-0.040373
- Sensitivity：-0.028571
- Specificity：-0.052174

当前结论：256 分辨率不应自动视为更优。若继续做高分辨率实验，建议先检查：

- 256 预处理是否与 224 完全同源
- face crop / mask / meanbg 是否存在边缘伪影
- ResNet18 默认结构是否适合更高输入
- batch size 是否因分辨率升高而变化
- OOF 概率在各折间是否校准一致

---

## 10. 正式运行命令模板

### PowerShell

```powershell
& "E:\resarch\Anaconda3\envs\face2\python.exe" "E:\projects\face2\scripts\run\run_e0b_global_resnet18_control_patient_binary_5fold.py" --config "E:\projects\face2\config\train\e0b_global_resnet18_control_patient_binary_anti_overfit_v3_bn_eval_full.yaml" --output-dir "E:\projects\face2\experiments\500Data\E0B_Global_ResNet18_ControlVsPatient_Binary_anti_overfit_v3_bn_eval_full"
```

### cmd.exe

cmd 中不要加 PowerShell 的 `&`。

```cmd
"E:\resarch\Anaconda3\envs\face2\python.exe" "E:\projects\face2\scripts\run\run_e0b_global_resnet18_control_patient_binary_5fold.py" --config "E:\projects\face2\config\train\e0b_global_resnet18_control_patient_binary_anti_overfit_v3_bn_eval_full.yaml" --output-dir "E:\projects\face2\experiments\500Data\E0B_Global_ResNet18_ControlVsPatient_Binary_anti_overfit_v3_bn_eval_full"
```

如需观察实时输出，可以使用 `-u`：

```cmd
"E:\resarch\Anaconda3\envs\face2\python.exe" -u "E:\projects\face2\scripts\run\run_e0b_global_resnet18_control_patient_binary_5fold.py" --config "E:\projects\face2\config\train\e0b_global_resnet18_control_patient_binary_anti_overfit_v3_bn_eval_full.yaml" --output-dir "E:\projects\face2\experiments\500Data\E0B_Global_ResNet18_ControlVsPatient_Binary_anti_overfit_v3_bn_eval_full"
```

---

## 11. 新增二分类实验的推荐流程

如果新 Thread 要继续实现其他二分类实验，建议按以下流程：

1. 复制一个现有 E0B 配置文件。
2. 修改 `experiment.name`，确保新实验名称唯一。
3. 修改 `data.image_root` 指向新输入图像目录。
4. 如分辨率不同，修改 `data.image_size`。
5. 保持 `split_dir = data/processed/splits_500`，除非明确要换数据划分。
6. 保持二分类标签映射逻辑不变：control vs patient。
7. 使用 E0B run wrapper 启动五折训练。
8. 训练完成后检查 `oof_metrics.csv`、`oof_confusion_matrix.csv`、`binary_experiment_results.md`。
9. 横向比较时使用 pooled OOF 指标，不要只看单折最优。

新增配置时最小修改示例：

```yaml
experiment:
  name: E0B_NewInput_ResNet18_ControlVsPatient_Binary_5fold
  output_dir: experiments/500Data

data:
  split_dir: data/processed/splits_500
  image_root: data/processed/path/to/new/images
  image_filename_template: "{ID}.png"
  n_folds: 5
  image_size: 224
  train_csv_pattern: fold_{fold}_train.csv
  val_csv_pattern: fold_{fold}_val.csv
```

---

## 12. 后续实验解释口径

关于“过拟合”和“论文展示”的建议口径：

- 不能为了让训练曲线好看而牺牲主验证/OOF结果过多。
- 也不能只展示训练集很好、验证集不稳定的结果。
- 更合适的方式是：保留 baseline 作为性能上限/原始 full-finetune 对照，同时保留方案B作为反过拟合/稳定性改进对照。
- 如果方案B在 AUC 接近或略优、同时训练行为更稳定，可以作为更稳健方案讨论。
- 如果论文主张是“某种输入/预处理/物理约束是否提高分类效果”，则必须保持训练协议一致，避免把输入差异和训练策略差异混在一起。

推荐主指标集合：

- Macro-AUC
- Macro-F1
- Balanced Accuracy
- Sensitivity
- Specificity

辅助指标：

- Accuracy
- Macro Precision
- Macro Recall
- Confusion matrix
- Fold-level metrics
- Training/validation loss and AUC curves

---

## 13. 给新 Thread 的关键提醒

- 运行训练必须使用 `face2` 环境和 GPU。
- 在 Codex 沙箱中直接调用 Anaconda 环境有时可能遇到 DLL / `_bz2` 等访问问题；如出现，应使用已授权/升级权限方式运行，而不是改成 CPU。
- E0B 的 `oof_metrics.csv` 默认没有直接写出 Sensitivity / Specificity，需要从 `oof_confusion_matrix.csv` 计算。
- 后续比较实验时优先比较 pooled OOF，不要被某些 fold-level AUC 的升降误导。
- 方案B目录名是 `anti_overfit_v3_bn_eval_full`，不要和 v1 混淆。
- 256 分辨率实验已经跑过，当前没有提升；若继续高分辨率方向，应先解释为何 256 可能引入更多噪声或校准问题。

