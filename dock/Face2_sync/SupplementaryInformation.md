# Supplementary Information

## 上一步数据预处理消融实验信息补充

本节回答 `prompt.md` 中关于上一轮 ResNet18 数据预处理消融实验的问题，并给出后续 ResNet34/ResNet50 复核与阈值扫描的建议。

### 1. 三个 ResNet18 预处理实验的实际输出目录

- baseline = `E:\projects\face2\experiments\preprocess_ablation_500Data\PreprocAblation_ResNet18_NYHA3Class_hybrid_black_baseline`
- meanbg = `E:\projects\face2\experiments\preprocess_ablation_500Data\PreprocAblation_ResNet18_NYHA3Class_hybrid_imagenet_meanbg`
- CLAHE = `E:\projects\face2\experiments\preprocess_ablation_500Data\PreprocAblation_ResNet18_NYHA3Class_hybrid_black_clahe_l`

对应汇总文件：

- `E:\projects\face2\experiments\preprocess_ablation_500Data\summary_all.csv`
- `E:\projects\face2\experiments\preprocess_ablation_500Data\summary_all.xlsx`
- `E:\projects\face2\experiments\preprocess_ablation_500Data\summary_all.md`

### 2. 上一轮是否已有通用 config 运行脚本

有。

通用脚本为：

- `E:\projects\face2\scripts\run\run_nyha3class_5fold_with_config.py`

该脚本可根据指定 YAML config 运行 NYHA 三分类五折训练。上一轮预处理消融还生成了批量运行脚本：

- `E:\projects\face2\scripts\run\run_preprocess_ablation_resnet18_nyha3class_5fold.py`
- `E:\projects\face2\scripts\run\run_full_preprocess_ablation_pipeline.py`

### 3. ResNet34/50 当前配置是否已跑通过

已跑通过。

当前 ResNet34 配置对应的历史输出目录为：

- config: `E:\projects\face2\config\train\nyha_3class_global224_imagenet_resnet34.yaml`
- output: `E:\projects\face2\experiments\500Data\Global224_ImageNetResNet34_NYHA3Class_WeightedCE_5Fold`

该目录下存在 `fold_0` 至 `fold_4` 的五折结果，并已生成：

- `summary\fold_metrics_all.csv`
- `summary\mean_metrics.csv`
- `summary\oof_metrics.csv`
- `summary\oof_predictions.csv`

当前 ResNet50 配置对应的历史输出目录为：

- config: `E:\projects\face2\config\train\nyha_3class_global224_imagenet_resnet50.yaml`
- output: `E:\projects\face2\experiments\500Data\Global224_ImageNetResNet50_NYHA3Class_WeightedCE_5Fold`

该目录同样存在五折结果和 `summary` 汇总文件。

### 4. Step 1 是否加入 CLAHE 的 ResNet34/50 复核

暂不加入主线复核。

建议 Step 1 优先复核：

1. `hybrid_black_baseline`
2. `hybrid_imagenet_meanbg`

理由：ResNet18 消融中，baseline 的 macro-AUC 最高；ImageNet mean background 在 macro-AUC 基本不下降的前提下，取得最佳 balanced accuracy 和 macro-F1，并明显提高 severe recall。CLAHE 虽然优于 baseline 的 balanced accuracy、macro-F1 和 severe recall，但整体不超过 meanbg，因此建议作为低优先级备选，而不是第一轮 ResNet34/50 主线复核项。

### 5. oof_predictions.csv 是否包含 prob_normal、prob_mild、prob_severe

是。

上一轮 8 个预处理消融实验的 `summary\oof_predictions.csv` 均包含以下概率列：

- `prob_normal`
- `prob_mild`
- `prob_severe`

这些列可直接用于后续阈值扫描、类别阈值重标定、severe-vs-rest 分析和 OOF 层面的决策规则搜索。

### 6. 阈值扫描重点

选择 C：两者都要，先看整体 BA/F1，再看 severe recall。

理由：单独追求 severe recall 可能导致整体判别能力下降。上一轮中 `hybrid_black_retinex_msr` 的 severe recall 最高，但 macro-AUC、balanced accuracy 和 macro-F1 均明显下降，说明它更像是把模型推向 severe 判定偏置，而不是带来更好的三分类表征。后续阈值扫描应优先保证 macro-F1 与 balanced accuracy 不明显下降，再在可接受范围内提高 severe recall。

### 上一轮 ResNet18 预处理消融关键结果

| Variant | macro-AUC | BA | macro-F1 | severe recall | 结论 |
|---|---:|---:|---:|---:|---|
| `hybrid_black_baseline` | 0.7128 | 0.5099 | 0.4852 | 0.2363 | macro-AUC 最优，作为强基线保留 |
| `hybrid_imagenet_meanbg` | 0.7094 | 0.5353 | 0.5111 | 0.4333 | BA 和 macro-F1 最优，是下一步首选候选 |
| `hybrid_black_clahe_l` | 0.7091 | 0.5224 | 0.5008 | 0.4216 | 可作为低优先级备选 |
| `hybrid_black_gray3ch` | 0.7073 | 0.5349 | 0.4857 | 0.3664 | 说明颜色信息不是唯一来源，但不建议直接替代 RGB |
| `hybrid_black_retinex_msr` | 0.6694 | 0.4852 | 0.4578 | 0.4598 | severe recall 高但整体损失明显，不建议优先推进 |

总体结论：下一步模型容量复核建议先在 ResNet34/ResNet50 上比较 baseline 与 ImageNet mean background。阈值扫描使用 OOF 概率列进行，优化目标以 macro-F1/BA 为主，severe recall 为约束或次级目标。

## 下一轮多 Backbone 实验前置信息补充

本节回答当前 `prompt.md` 中提出的 8 项补充信息，用于后续 DenseNet、EfficientNet、ConvNeXt、Swin 和 MobileNetV3 等非 ResNet backbone 实验设计。

### 1. 当前 PyTorch / torchvision 环境

在项目环境 `E:\resarch\Anaconda3\envs\face_heart\python.exe` 中已确认：

```text
torch = 2.3.0
torchvision = 0.18.0
cuda = True
```

`torchvision.models` 当前支持下一轮候选模型：

| Model | torchvision support |
|---|---|
| `densenet121` | yes |
| `efficientnet_b0` | yes |
| `convnext_tiny` | yes |
| `swin_t` | yes |
| `mobilenet_v3_large` | yes |

因此第一轮不需要引入 `timm`，优先使用 `torchvision.models` 即可。

### 2. 通用 YAML 配置运行脚本状态

通用入口脚本已存在：

- `E:\projects\face2\scripts\run\run_nyha3class_5fold_with_config.py`

该脚本已经在正式 Backbone Check 中实际跑通过多个 YAML 配置。最近一次串行全量五折复核中，以下 4 个作业均为 `SUCCESS`：

| Job | Backbone | Preprocess | Status |
|---|---|---|---|
| B1 | `resnet34` | `hybrid_black_baseline` | `SUCCESS` |
| B2 | `resnet34` | `hybrid_imagenet_meanbg` | `SUCCESS` |
| B3 | `resnet50` | `hybrid_black_baseline` | `SUCCESS` |
| B4 | `resnet50` | `hybrid_imagenet_meanbg` | `SUCCESS` |

因此下一轮 Codex 可继续复用该脚本，不需要重新实现五折训练入口。

### 3. 当前训练脚本的模型构建调用位置

当前 `scripts\train\train_nyha_3class_5fold.py` 中，单图像分支仍直接调用：

```text
models/resnet_nyha_3class.py::build_resnet_nyha_model
```

具体位置在 `_build_model(config)` 中：

```python
return build_resnet_nyha_model(
    backbone=model_config["backbone"],
    num_classes=int(model_config["num_classes"]),
    pretrained=_pretrained_enabled(model_config["pretrained"]),
)
```

当前尚未改成通用模型 factory。多 ROI 分支使用 `ConfigurableMultiROIFusionResNet`，也仍限定在 ResNet 系列。

### 4. 是否允许小幅修改训练脚本

建议允许 Codex 小幅修改 `scripts\train\train_nyha_3class_5fold.py`，但修改范围应限制为模型构建入口：

1. 新增或扩展通用模型 factory。
2. 让 `_build_model(config)` 根据 `model.backbone` 调用通用 factory。
3. 保持旧 ResNet 配置完全兼容。
4. 不修改 loss、optimizer、学习率、epoch、early stopping、数据增强和五折划分。

### 5. 是否允许新增 torchvision 以外依赖

建议第一轮不允许新增依赖，不安装 `timm`。

如果某个模型在当前 `torchvision==0.18.0` 中不可用，应在 manifest 或 job queue 中标记为 `unsupported`，不要临时安装新包。当前已确认 5 个候选模型均可由 `torchvision.models` 提供。

### 6. 第一轮候选模型清单

建议支持完整 5 个候选模型：

```text
densenet121
efficientnet_b0
convnext_tiny
swin_t
mobilenet_v3_large
```

正式训练时可通过 `--only` 先跑前三个更稳的模型：

```text
densenet121
efficientnet_b0
convnext_tiny
```

### 7. Batch size 策略

建议采用保守 batch size 配置：

| Model | batch size |
|---|---:|
| `densenet121` | 16 |
| `efficientnet_b0` | 16 |
| `convnext_tiny` | 8 |
| `swin_t` | 8 |
| `mobilenet_v3_large` | 16 |

理由：当前设备为笔记本 RTX 4060，`convnext_tiny` 和 `swin_t` 的显存占用更高，优先使用 batch size 8 可降低正式五折训练中断风险。

### 8. ResNet18 meanbg 基线目录

下一轮可继续使用默认 ResNet18 meanbg 基线目录：

- `E:\projects\face2\experiments\preprocess_ablation_500Data\PreprocAblation_ResNet18_NYHA3Class_hybrid_imagenet_meanbg`

该目录已确认存在，并包含：

- `summary\fold_metrics_all.csv`
- `summary\mean_metrics.csv`
- `summary\oof_metrics.csv`
- `summary\oof_predictions.csv`
- `summary\summary_report.md`

因此可作为后续多 backbone 实验的 ResNet18 meanbg 参考基线。
