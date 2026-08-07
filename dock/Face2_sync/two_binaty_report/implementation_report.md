# E0B 实现报告

## 已审阅的 E0 实现

- 训练入口：`scripts/train/train_nyha_3class_5fold.py`
- 配置：`config/train/nyha_3class_global224_imagenet_resnet18.yaml`
- Dataset/变换：`datasets/nyha_3class_face_dataset.py`
- 模型工厂：`models/nyha_backbone_factory.py` 与 `models/resnet_nyha_3class.py`
- 类别权重/loss：`losses/classification_losses.py`
- Trainer/Evaluator：`trainers/nyha_3class_trainer.py`、`evaluators/nyha_3class_evaluator.py`
- 五折汇总：`scripts/evaluate/summarize_nyha_3class_5fold.py`

## 新增 E0B 文件及目的

- `config/train/e0b_global_resnet18_control_patient_binary_5fold.yaml`：独立配置，固定 `splits_500` 与 mean-background 图像。
- `datasets/control_patient_binary_dataset.py`：不修改 CSV 的三分类→二分类动态映射及元数据保留。
- `metrics/binary_classification_metrics.py`：以患者概率计算二分类 ROC-AUC 和六项指标。
- `utils/e0b_binary_audit.py`：训练前图像、分组泄漏、OOF 覆盖和标签冲突审计。
- `scripts/train/train_e0b_global_resnet18_control_patient_binary_5fold.py`：独立训练、checkpoint、预测和每折产物。
- `scripts/evaluate/summarize_e0b_global_resnet18_control_patient_binary_5fold.py`：OOF 完整性检查、汇总和中文报告。
- `tests/test_e0b_control_patient_binary.py`：映射、非法标签、二分类指标、概率和权重测试。

E0 三分类源码与既有结果未被修改。二分类映射位于 `map_three_class_to_binary`，分类头经 E0 的模型工厂传入 `num_classes=2` 构建，类别权重由 E0 的 `compute_class_weights(..., num_classes=2)` 基于各训练折单独计算。

## 验证与运行

已执行静态导入、配置解析、数据审计、Dataset/模型/指标轻量测试和独立单折 1 epoch 烟雾测试。正式运行命令：

```powershell
python scripts/train/train_e0b_global_resnet18_control_patient_binary_5fold.py --config config/train/e0b_global_resnet18_control_patient_binary_5fold.yaml --output-dir E:\projects\face2\experiments\500Data\E0B_Global_ResNet18_ControlVsPatient_Binary_5fold_GPU
python scripts/evaluate/summarize_e0b_global_resnet18_control_patient_binary_5fold.py --experiment-dir E:\projects\face2\experiments\500Data\E0B_Global_ResNet18_ControlVsPatient_Binary_5fold_GPU
```

正式结果目录：`E:\projects\face2\experiments\500Data\E0B_Global_ResNet18_ControlVsPatient_Binary_5fold_GPU`。训练环境、开始/完成时间保存在 `environment.json` 和 `run_finished_at.txt`。


## 一键正式运行入口

也可使用 `scripts/run/run_e0b_global_resnet18_control_patient_binary_5fold.py`，它会依次执行训练和 OOF 汇总，避免遗漏最终报告步骤。
