# Global ResNet18 + 六维光学表型融合实验预实现审计

## 1. 审计结论与完成状态

**PREIMPLEMENTATION_AUDIT_STATUS：`READY_WITH_EXPLICIT_ADAPTATIONS`**

本次审计已完成提示词要求的仓库、代码、配置、历史产物、Stage 1/2A/2B、固定五折、标签、ID、缺失值、逐折文件选择、模型/Trainer/Evaluator 复用、颜色增强和公平性核验。当前数据与协议信息足以进入正式编码，但不能直接把现有代码原样拼接后开跑，必须落实以下适配：

1. 使用正式 meanbg 配置训练新的 G0；未找到已完成的 meanbg ResNet18 G0 产物。
2. 2A/2B 必须读取与分类 fold 同号的逐折 train/val 文件，禁止用 500 行 OOF 再拆分。
3. 对每个 variant 使用顺序固定的正向字段白名单，禁止按“所有数值列”自动选择。
4. 新增 outer-train-only 六维标准化与缺失处理，并保存/恢复 scaler。
5. 给 Dataset、Trainer、Evaluator 和模型增加图像+辅助向量的数据通路，同时保持历史单图代码不变。
6. 隔离每个 `(variant, fold)` 的随机流，确保五个模型的图像增强和训练预算公平。
7. 正式训练前恢复 CUDA 可用的 PyTorch 环境；本次没有安装依赖或训练模型。

审计期间只新增本目录下五个报告文件，没有修改任何历史代码、配置、split、标签、图像或三阶段产物，没有启动正式训练或完整五折实验。

## 2. 实际项目与仓库状态

| 项目 | 审计值 |
|---|---|
| 实际 project root | `E:/projects/face2` |
| Git branch | `main` |
| Git commit | `b58ad3fafb5bc992aa8e63e94996ef04a7a38b55` |
| 工作区是否有未提交内容 | 是 |
| 未提交内容是否可能与本任务相关 | 是；Global runner/trainer/summary、Stage 1/2A/2B 代码/配置/测试/报告均有在途修改或未跟踪内容 |
| Python | 3.13.9 |
| PyTorch | 2.12.0+cpu |
| `torch.cuda.is_available()` | `False` |
| `torch.version.cuda` | `None` |
| GPU/驱动可见性 | `nvidia-smi` 可见 NVIDIA GeForce RTX 4060 Laptop GPU，驱动 581.42 |

环境说明：直接先导入 torch 会触发重复 OpenMP 运行库错误；项目训练脚本通过先导入 sklearn 规避。本次用相同顺序完成只读 torch/checkpoint 检查。当前安装是 CPU-only PyTorch，因此不能据 GPU 硬件可见就声称 CUDA 可用。

本次没有清理或改写用户未提交内容。开始审计时相关在途修改已存在；尤其 `trainers/nyha_3class_trainer.py`、`scripts/train/train_nyha_3class_5fold.py` 和 `scripts/evaluate/summarize_nyha_3class_5fold.py` 处于修改状态，正式编码应避免覆盖。

## 3. 相关目录结构

```text
config/train/preprocess_ablation_resnet18/
  nyha_3class_resnet18_preproc_hybrid_imagenet_meanbg.yaml
data/processed/global_face/preprocess_ablation/hybrid_imagenet_meanbg/images/
data/processed/splits_500/
data/processed/optical_observations_v1/
  regional_optical_observations.csv
  feature_schema.json
  extraction_manifest.json
experiments/optical_condition_calibration_stage2a/
  fold_0 ... fold_4/
  summary/{calibration_feature_schema.json,oof_calibrated_features.csv,run_manifest.json}
experiments/optical_condition_calibration_stage2b/
  fold_0 ... fold_4/
  summary/{calibration_stage2b_feature_schema.json,oof_nn_calibrated_features.csv,run_manifest.json}
models/  datasets/  trainers/  evaluators/  losses/  metrics/
scripts/{run,train,evaluate,preprocess}/
tests/
```

直接相关文件的逐项作用、复用性和风险见 `relevant_file_inventory.csv`；特征源与白名单见 `feature_source_inventory.csv`；逐折对齐结果见 `fold_alignment_audit.csv`。

## 4. 真正的 Global ResNet18 基线选择

### 4.1 候选比较

| 候选 | 图像/队列 | 证据 | 已完成产物 | 是否推荐为 G0 |
|---|---|---|---|---|
| 正式 meanbg 配置 | `hybrid_imagenet_meanbg/images` + `splits_500` | 配置、既有 EXIF/预处理审计、Stage 1 study ID 均明确指向 | **未找到** `experiments/preprocess_ablation_500Data/` | **推荐**；需在新实验中训练 |
| 历史 500Data ResNet18 | 保存配置指向 `splits_500`，但缺 `image_root` | 完整五折 checkpoint/log/metrics/OOF；fold 0 checkpoint `fc.weight=[3,512]` | 有，OOF macro-AUC=0.7025 | 不可充当 meanbg G0；仅用于训练协议证据 |
| 当前 strict-blackbg Global 配置 | 522 张 strict-blackbg 目录按 500 split 取队列 | `config/train/nyha_3class_global224_imagenet_resnet18.yaml` | 历史候选产物存在，但并非正式 meanbg 条件 | 不推荐 |
| 历史 522Data ResNet18 | 522 队列/旧 split | 历史实验目录 | 有 | 不推荐；队列与三阶段 500 ID 不同 |

推荐结论：

- **推荐基线代码**：`models/resnet_nyha_3class.py`，由 `scripts/train/train_nyha_3class_5fold.py` 的 `_build_model` 经 `models/nyha_backbone_factory.py` 调用。
- **推荐基线配置**：`config/train/preprocess_ablation_resnet18/nyha_3class_resnet18_preproc_hybrid_imagenet_meanbg.yaml`。
- **推荐实验语义**：`PreprocAblation_ResNet18_NYHA3Class_hybrid_imagenet_meanbg` 的协议，在新统一融合 runner 中作为 `global_only` 重训。
- **图像目录**：`data/processed/global_face/preprocess_ablation/hybrid_imagenet_meanbg/images/`。
- **固定 split**：`data/processed/splits_500/`。

未找到已完成 meanbg G0 的详细搜索记录见 `unresolved_items.md`。历史 500Data 保存配置没有 image root，当前 split 又没有 `image_path`，因此不能原样复现；它的运行时间也早于正式 meanbg 链，不能用文件名推定成 meanbg。

## 5. Global 基线模型结构

1. `models/resnet_nyha_3class.py::_build_torchvision_resnet` 创建 torchvision ResNet18；`pretrained=imagenet` 对 ResNet18 解析为 `ResNet18_Weights.IMAGENET1K_V1`。
2. `model.fc.in_features` 为 512，随后替换为 `nn.Linear(512, 3)`；当前正式配置不设置 dropout，因此没有额外 Dropout 或 MLP。
3. torchvision ResNet 的最后 `AdaptiveAvgPool2d((1,1))` 后展平得到 512 维 Global 特征；当前项目没有显式暴露 `forward_features` 或 `return_features`。
4. `forward` 直接返回未归一化 logits `[B,3]`；softmax 位于 Trainer 验证和 Evaluator。
5. 当前 backbone 全部可训练；不冻结、不分阶段解冻、不设置不同参数组学习率。
6. 当前线性头参数量为 `512×3+3=1,539`；checkpoint fold 0 证实 `fc.weight=(3,512)`、`fc.bias=(3,)`。
7. checkpoint 通过 `torch.load(..., map_location=device)` 读取，并严格 `load_state_dict(checkpoint["model_state_dict"])`。Trainer resume 同时恢复 optimizer；当前工作树版本还会恢复 AMP scaler/RNG，但历史 checkpoint 只有 `epoch/fold/model_state_dict/optimizer_state_dict/best_macro_auc/config`。

扩展判断：512+7 直接拼接完全可行，但不应修改 torchvision ResNet 的历史对象或原地改写 `models/resnet_nyha_3class.py`。建议新建包装器，构建相同 ResNet18、将 `fc` 设为 Identity 获取 512 维，再连接一个维度由 variant 决定的单层 `Linear`。历史 checkpoint 无需强行兼容，因为新的 meanbg G0 应一起重训；若未来要导入历史 G0，必须另写明确的键映射测试，不能静默 `strict=False`。

## 6. Global 基线完整训练协议

### 6.1 代码默认、正式配置和历史实际值

| 项目 | 代码默认 | 推荐 meanbg 配置 | 已完成 500Data 历史实际证据 |
|---|---|---|---|
| backbone | resnet18 | resnet18 | resnet18 |
| pretrained | `True` | imagenet | imagenet |
| image size | 224 | 224 | 224 |
| batch size | Trainer 不定义；入口读取配置 | 16 | 16 |
| epochs | Trainer 默认 50 | 50 | 最多 50；各折因早停运行 11/20/15/11/14 epochs |
| optimizer | 入口限制 AdamW | AdamW | AdamW |
| lr | 配置必填 | 1e-4 | 1e-4 |
| weight decay | 配置必填 | 1e-4 | 1e-4 |
| scheduler | 未实现 | 无 | 无 |
| warmup | 未实现 | 无 | 无 |
| gradient clipping | 未实现 | 无 | 无 |
| AMP | Trainer 支持 | false | false |
| seed | 入口读取 | 2026 | 2026 |
| deterministic | cuDNN deterministic=true、benchmark=false | 同代码 | 同协议；未调用 `torch.use_deterministic_algorithms` |
| num_workers | 配置必填 | 0 | 0 |
| pin_memory | 配置可选 | false | false |
| early stopping | Trainer 默认 patience=10 | 10 | 10 |
| checkpoint metric | 宏 OvR AUC | macro_auc | macro_auc |
| 并列规则 | 仅 `>` 视为改进 | 较早 epoch 保留 | 较早 epoch 保留 |

历史五折 best epoch 分别为 1、10、5、1、4。outer val 每一 epoch 都参与 checkpoint/早停选择，因此这 100 例是 held-out validation，不是独立 test。模型不固定训练 100 epochs，也不使用最后 epoch；使用 `best_macro_auc.pth`。

### 6.2 数据输入与 transform

- 正式 meanbg 根：`data/processed/global_face/preprocess_ablation/hybrid_imagenet_meanbg/images/`，500 个 `.png`，模板 `{ID}.png`。
- Pillow `Image.open(...).convert("RGB")`，所以模型输入是 RGB，不是 OpenCV BGR。
- Dataset 对 meanbg 成品不再做背景移除或背景合成，只做 transform。
- train：`Resize(224,224)` → `RandomHorizontalFlip(p=0.5)` → `ToTensor` → ImageNet Normalize。
- val：`Resize(224,224)` → `ToTensor` → 同一 ImageNet Normalize。
- mean=`[0.485,0.456,0.406]`，std=`[0.229,0.224,0.225]`。
- 无随机裁剪、旋转、ColorJitter、brightness/contrast/saturation/hue/gamma、通道扰动、RandomErasing、MixUp 或 CutMix。
- `augmentation.color_jitter=false` 和 `random_crop=false` 写在配置中，但当前 `build_transforms` 本身也没有实现这两类操作；最终实际 transform 只有水平翻转这一项随机增强。

### 6.3 固定五折

- fold 编号为 0–4。
- 每折 400 train / 100 val，总计 500；五个 val 恰好覆盖每个 ID 一次。
- `data/processed/splits_500/nyha_3class_sex_stratified_group_5fold.csv` 有 `fold` 列；逐折 train/val 文件没有 `split_role` 列，角色由文件名表达。
- 分层依据为 `label_3class × SEX`；患者组 `patient_group_id` 由去除 `-数字` 后缀构造并整体分配。
- 500 个样本对应 483 个患者组；17 个患者组含多图，0 个患者组跨 fold。
- 当前普通 Global 协议没有 inner train/val。Stage 2B 的 inner split 只用于无标签校准网络选 epoch，不是分类器 inner validation。
- 仓库存在 522 例历史数据/实验；正式 meanbg/光学三阶段队列是 500，不应混用。

### 6.4 标签

- 标签血缘：`data/raw/label_raw_nyha2_remove22_sex_balanced_500.csv`，字段 `ID/SEX/NYHA`，500 行、500 唯一 ID、无缺标签/重复 ID。
- 分类训练时，Dataset 实际读取 fold CSV 内的 `label_3class`，不会再次读取原始标签表。
- 映射：NYHA 0→normal/0；NYHA 1、2→mild/1；NYHA 3、4→severe/2。
- 实际三类数量：normal=115、mild=237、severe=148。
- 原始 NYHA 0–4 数量：115、68、169、129、19。
- ID 以 pandas string dtype 读取并 `str()` 保留；500 队列有 41 个 `-数字` 后缀 ID，无前导零、科学计数法样式或首尾空格。

### 6.5 损失

- 正式配置为 fold-specific weighted cross entropy，无 label smoothing。
- 类别权重公式：`N / (num_classes × class_count)`，只用当前 fold 的 400 个训练标签计算。
- `nn.CrossEntropyLoss(weight=..., reduction='mean')`；权重和标签平滑不同时使用。
- 历史 fold 0/1 权重约 `[1.449275,0.705467,1.120448]`，fold 2–4 约 `[1.449275,0.701754,1.129943]`。
- 代码另有 exclude-true-class 平滑实现：真类 `1-α`、其他类 `α/(C-1)`；但不属于推荐 meanbg G0 的实际协议，α 实际为 0/未启用。

### 6.6 指标

- Evaluator 先 softmax，传概率给 `compute_classification_metrics`。
- macro-AUC 为 `roc_auc_score(..., multi_class="ovr", average="macro")`；同时计算三类 per-class OvR AUC。
- 另有 Accuracy、Balanced Accuracy、Macro Precision/Recall/F1、per-class Precision/Recall/F1、severe-vs-rest、normal-vs-abnormal 和 confusion matrix。
- 任一类缺失时对应 binary AUC 返回 NaN，macro-AUC 也设为 NaN；precision/recall/F1 用 `zero_division=0`。
- summary 拼接五个 val prediction，检查 ID 不重复并输出五折均值±样本标准差（ddof=1）和 500 行 OOF。
- 当前通用 Global evaluator 未找到 ROC 坐标表、ROC 图或 bootstrap 置信区间实现。

## 7. Dataset 数据流与扩展建议

`NYHA3ClassFaceDataset.__init__` 参数为 `csv_path, transform, image_root, image_filename_template`。它在初始化时以 string dtype 读取 ID/患者组并验证标签；`__getitem__` 时按 ID 构建图像路径、打开并转 RGB、执行 transform，返回 `image/label/ID/patient_group_id/image_path/NYHA/SEX/sex_name/label_3class_name/fold`。

当前 Dataset 没有光学表型、availability、variant 或 fold-specific 特征文件概念。推荐新增 `GlobalOpticalFusionDataset`，内部组合一个 `NYHA3ClassFaceDataset` 并按其 frame 顺序做一对一 ID join：

- G0：不读取任何光学表。
- G-Mask：只从 Stage 1 正向读取 `ID,forehead_available`。
- G-Raw：Stage 1 正向读取 `ID,forehead_available` 和六个 Raw 字段。
- G-A/G-B：构建 classification fold k 时分别读取该 fold 的 train/val 文件；禁止使用 OOF。
- 所有 variant 以同一 split CSV 的行序作为样本序；特征表仅按 ID 对齐，不得改变/排序 Dataset frame。
- join 后断言行数、唯一 ID、无缺失/额外 ID、fold/role 一致，再把预处理后的辅助向量作为 `aux_features` 返回。

与直接扩展历史 Dataset 相比，新 Dataset 更安全，能保持历史 G0/ROI 行为不变。为保证新 G0 与融合组完全同队列，可让新 runner 的五个 variant 都使用这个新 Dataset 包装层；G0 的 `aux_features` 为形状 `[0]` 或根本不返回，模型明确走 512→3。

## 8. Trainer 数据流与复用判断

当前 Trainer：

- 假设 batch 至少有 `image` 和 `label`；训练/验证均调用 `model(images)`。
- 支持 AMP、weighted loss、五折外部入口、best/last checkpoint、早停、训练 CSV 和曲线。
- 不支持 `aux_features`、variant、feature order 或 classifier scaler。
- Trainer 不输出 ID 预测；ID 预测由 Evaluator 完成。
- checkpoint 保存 model/optimizer/config；当前工作树版本另存 AMP scaler/RNG，但不保存光学 feature scaler。

因此 Trainer 的协议和辅助函数可复用，但类不能“原样”用于融合。为避免影响历史实验及覆盖用户在途修改，推荐新增薄的 fusion trainer/evaluator，或在正式变更获确认后以默认保持 `model(images)` 的 hook 做向后兼容扩展。无论哪种方式，都必须把 variant、feature order、source hashes 和 `feature_scaler` 写入 checkpoint/manifest。

## 9. Evaluator 与 OOF 流程

当前 Evaluator 恢复 `best_macro_auc.pth`，逐 batch softmax，保存 ID、patient_group_id、NYHA、SEX、fold、三类概率、预测类与正确性；输出逐折 metrics 和 confusion matrix。summary 再拼接五折预测。

融合实现可复用类别顺序 `normal/mild/severe = 0/1/2`、概率列和指标函数，但必须增加：

1. `model(image, aux_features)` 通路；G0 明确无 aux。
2. checkpoint 中的 variant/feature order/scaler 与当前 Dataset 的一致性检查。
3. OOF 恰好 500 行/500 唯一 ID/每 ID 一次，fold 映射等于固定 split。
4. 每行概率有限、范围 `[0,1]` 且和为 1。
5. 五个 variant 输出同构 OOF 文件，便于配对比较。

## 10. 图像增强与颜色一致性风险

当前正式 Global transform **没有颜色或强度增强**。配置中的 `color_jitter=false` 与代码实际一致；brightness、contrast、saturation、hue、gamma、通道扰动均未实现/未启用。因此不存在“Global 颜色被改、六维表型仍是原图”的训练时跨模态颜色冲突。

水平翻转可以保留：Raw/2A/2B 使用左右脸颊的均值与额部减脸颊，不携带左右方向；翻转不会改变六维表型语义。Resize 对所有样本固定，val 与 train 的 Normalize 完全一致。train/val 唯一有意差异是 train 的水平翻转，没有额外颜色预处理不一致。

结论：不需要因颜色增强关闭而重训一个“颜色匹配 G0”，因为当前颜色增强本来就是关闭的；但仍需要训练新的 meanbg G0，原因是正式 meanbg G0 历史产物不存在，且公平对照应由同一新 runner 产生，而不是颜色冲突。

## 11. 固定 split 与标签完整性

`splits_500` 的五个 val 均为 100 例，train 为其余 400 例，train/val ID 交集为 0，患者组交集为 0。五折 val 拼接为 500 唯一 ID；patient_group_id 跨折数为 0。标签源与 split 的 ID、NYHA、SEX 全部一致，NYHA/SEX 不匹配数均为 0。

注意：split 文件没有 `split_role` 列。正式实现必须由明确的 `train_csv_pattern/val_csv_pattern` 决定角色，不能从行内容猜测。Stage 2A/2B 输出则有 `split_role`，应断言 train 文件全部为 `train`、val 文件全部为 `val`。

## 12. Stage 1 Raw 六维审计

### 12.1 正式来源与字段

- 正式目录：`data/processed/optical_observations_v1/`。
- 正式文件：`regional_optical_observations.csv`。
- schema：`feature_schema.json`；manifest：`extraction_manifest.json`，状态 COMPLETE/PASS。
- ID 字段：`ID`；**无 fold 字段**；availability：`forehead_available`。
- 六维顺序与提示词一致：
  1. `cheek_mean_log2_y`
  2. `cheek_mean_log2_rg`
  3. `cheek_mean_log2_bg`
  4. `forehead_minus_cheek_log2_y`
  5. `forehead_minus_cheek_log2_rg`
  6. `forehead_minus_cheek_log2_bg`

ID 读为 string；availability 为 int64；六维均为 float64。总行数 500、唯一 ID 500、重复 ID 0。availability=1 为 486，=0 为 14。前三个 cheek 字段非有限数均为 0；后三个额部差值各有 14 个 NaN、无 ±Inf。14 例 availability=0 的后三维全为 NaN，486 例 availability=1 的后三维全有限，严格一致。

表中不含 NYHA，但含 `camera_id`、ExposureTime/FNumber/ISOSpeedRatings、两个派生条件、区域中间量和 cheek QC。schema 声明了 `derived_observation_columns` 六维顺序和一组 forbidden 字段，但 forbidden 列表未覆盖所有非目标数值列；实现必须正向 `usecols`。

Stage 1 的 study ID 来自 meanbg 500 图的完整 stem，但实际光学计算使用 meanbg 生成前的 aligned RGB 与 mask；因此与 Global 使用相同 ID 体系，不是相同像素输入。这正是合理的数据血缘：Global 使用 meanbg，光学表型使用颜色更接近原始的 aligned ROI。

## 13. Stage 2A 六维审计

### 13.1 正式来源

- 根：`experiments/optical_condition_calibration_stage2a/`。
- run manifest：`summary/run_manifest.json`，状态 COMPLETE。
- schema：`summary/calibration_feature_schema.json`。
- OOF：`summary/oof_calibrated_features.csv`，仅供审计/汇总。
- 每折：`fold_k/train_calibrated_features.csv` 和 `fold_k/val_calibrated_features.csv`，k=0..4，全部存在。

六个允许进入 G-A 的字段顺序：

1. `calibrated_cheek_mean_log2_y`
2. `calibrated_cheek_mean_log2_rg`
3. `calibrated_cheek_mean_log2_bg`
4. `calibrated_forehead_minus_cheek_log2_y`
5. `calibrated_forehead_minus_cheek_log2_rg`
6. `calibrated_forehead_minus_cheek_log2_bg`

每折 train=400/400 唯一 ID、val=100/100 唯一 ID，交集 0；与同号分类 split 完全一致。五个 val 拼接 500/500 唯一 ID，fold=0..4 各 100。availability 总计 486/14；cheek calibrated 非有限数 0；后三维仅 14 个 unavailable 行为 NaN。

文件同时含 raw_*、predicted_acquisition_*、residual_*、camera、条件 z 值。G-A 只允许 calibrated 六维和 availability；所有 raw/predicted/residual/camera/EXIF/condition/QC 都不得进入此 variant。Raw 与 Stage 1 逐 ID 数值一致，CSV 往返产生的最大绝对浮点差为 8.88e-16。

condition scaler 仅在对应 outer train 内按设备拟合 population mean/std，val 复用 train 参数；Ridge 只拟合 outer train。Stage 2A 白名单读取中没有 NYHA/临床标签，manifest 明确 `nyha_used=false`。

**分类 fold k 的唯一正确读取方式：**

- train：`experiments/optical_condition_calibration_stage2a/fold_k/train_calibrated_features.csv`
- val：`experiments/optical_condition_calibration_stage2a/fold_k/val_calibrated_features.csv`

## 14. Stage 2B 六维审计

### 14.1 正式来源

- 根：`experiments/optical_condition_calibration_stage2b/`。
- run manifest：`summary/run_manifest.json`，状态 COMPLETE。
- schema：`summary/calibration_stage2b_feature_schema.json`。
- OOF：`summary/oof_nn_calibrated_features.csv`，仅供审计/汇总。
- 每折：`fold_k/train_nn_calibrated_features.csv` 和 `fold_k/val_nn_calibrated_features.csv`，k=0..4，全部存在。

六个允许进入 G-B 的字段顺序：

1. `calibrated_nn_cheek_mean_log2_y`
2. `calibrated_nn_cheek_mean_log2_rg`
3. `calibrated_nn_cheek_mean_log2_bg`
4. `calibrated_nn_forehead_minus_cheek_log2_y`
5. `calibrated_nn_forehead_minus_cheek_log2_rg`
6. `calibrated_nn_forehead_minus_cheek_log2_bg`

逐折数量、唯一 ID、train/val 交集、split 对齐均与 Stage 2A 相同；五折 OOF 为 500/500；availability 486/14；cheek 输出全有限，后三维只在 14 个 unavailable 行为 NaN。Raw 与 Stage 1/2A 数值一致到最大绝对差 8.88e-16。

文件含 raw_*、predicted_condition_nn_*、residual_nn_*、camera、条件 z 值。只允许 calibrated_nn 六维和 availability。manifest/协议白名单没有 NYHA、SEX、临床或 Global 特征；`clinical_labels_loaded=false`、`nyha_used=false`。

2B 的 inner 80/20 仅从 outer train 内按 camera 确定 epoch。选定 epoch 后丢弃 inner 参数，重新初始化并在完整 outer train 拟合；最终 train 特征是该完整 outer-train 模型的 **in-sample** 转换，val 特征是同一模型的严格 outer-val 转换。因此文件边界正确，但存在训练/验证校准误差分布差：现有报告 10 个网络中 7 个 outer-val MSE 高于 train，最大差 0.324196；本次六维均值差审计的最大绝对 train-std 单位为 0.386622。此风险要披露，但不构成标签泄漏。

**分类 fold k 的唯一正确读取方式：**

- train：`experiments/optical_condition_calibration_stage2b/fold_k/train_nn_calibrated_features.csv`
- val：`experiments/optical_condition_calibration_stage2b/fold_k/val_nn_calibrated_features.csv`

## 15. Global、标签、split、三阶段 ID 一致性

| 比较 | 左侧数量 | 右侧数量 | 交集 | 仅左侧 | 仅右侧 | 完全一致 |
|---|---:|---:|---:|---:|---:|---|
| meanbg 图像 vs 标签 | 500 | 500 | 500 | 0 | 0 | 是 |
| meanbg 图像 vs 固定 split | 500 | 500 | 500 | 0 | 0 | 是 |
| 固定 split vs Stage 1 | 500 | 500 | 500 | 0 | 0 | 是 |
| Stage 1 vs Stage 2A OOF | 500 | 500 | 500 | 0 | 0 | 是 |
| Stage 1 vs Stage 2B OOF | 500 | 500 | 500 | 0 | 0 | 是 |
| Stage 2A OOF vs Stage 2B OOF | 500 | 500 | 500 | 0 | 0 | 是 |
| strict-blackbg 目录 vs 500 split | 522 | 500 | 500 | 22 | 0 | 否；目录是超集 |

未发现前导零丢失、科学计数法、空格、大小写、扩展名残留、重复 ID 或排序导致的集合差异。41 个 `-数字` 后缀 ID 保持完整；patient_group_id 只用于分组，不替代样本 ID。逐 fold 对齐全部 PASS，详情见 `fold_alignment_audit.csv`。

## 16. 为什么不能把 500 行 Stage 2A/2B OOF 再拆成分类 400/100

对分类 fold k，正确校准器必须只在该 fold 的 400 个分类 train ID 上拟合，并同时转换这 400 个 train 与 100 个 val。逐折文件正好满足该条件。

若读 500 行 OOF 再按分类 fold k 拆：

- 分类 val 的 100 行恰好来自 fold k 校准器，这部分看似正确。
- 分类 train 的每个 ID 来自“以该 ID 所属 validation fold j 为外折”的另一个校准器；这些校准器通常训练时看过分类 fold k 的 val ID。
- 因而分类 train 特征的预处理参数间接使用了分类 val 病例的无标签条件/光学数据，而分类 val 特征又来自另一个校准器；同一分类 fold 内 train/val 不共享同一个预处理拟合边界。

这不是 NYHA 标签泄漏，因为 2A/2B 没有读取 NYHA；但它是 **外折预处理边界不严格/跨折信息混用**，并制造多校准器混合的训练分布。现有逐折 train/val 文件足够完成正确实现，不需要重跑 Stage 2A 或 Stage 2B。

## 17. 六维标准化与缺失处理方案

提示词拟定方案可行，推荐以独立 preprocessor 实现，而不是放在 collate_fn 或每次 `__getitem__` 动态拟合：

1. 每个 outer fold、每个 `global_raw/global_stage2a/global_stage2b` 独立拟合。
2. 只读取当前 fold 的 400 个 train 行；val 永不参与 fit。
3. 前三维 cheek 用全部 400 个 train 计算 population mean/std（ddof=0）。
4. 后三维只用 `forehead_available==1` 且有限的 train 行计算 mean/std。
5. 任一 std `<1e-8` 立即报错，不静默设零。
6. val 使用 train mean/std；不可用病例在 **标准化后** 将后三维填 0。
7. 14 个 unavailable 病例保留，前三维 cheek 保留；availability 作为独立 0/1 输入。
8. G-Mask 不拟合六维 scaler，只读取 availability；G0 不读取 availability。
9. feature order 由 variant 常量与 schema 交叉核验，禁止依赖 CSV 原始列顺序或正则自动收集。
10. 每 fold/variant 保存 `feature_scaler.json`，至少含 schema version、variant、fold、六维顺序、mean/std、eligible counts、ddof、epsilon、missing fill、source path/hash、train ID hash。
11. checkpoint 内嵌同一 scaler payload 或其不可变哈希；恢复时验证外部 JSON 与 checkpoint 一致，Evaluator 只做 transform，绝不 refit。

现有 Stage 2A condition scaler/Stage 2B target scaler可复用 JSON、ddof=0、std 阈值和测试风格，但语义不同，不能直接当最终 classifier feature scaler。

## 18. 512+7 直接拼接与五个 variant

推荐模型为单个轻量包装器，不修改 torchvision ResNet 历史实现：

| variant | Global 特征 | 六维表型 | mask | 线性头输入 | 线性头参数 |
|---|---:|---:|---:|---:|---:|
| G0 / `global_only` | 512 | 0 | 0 | 512 | 1,539 |
| G-Mask / `global_mask` | 512 | 0 | 1 | 513 | 1,542 |
| G-Raw / `global_raw` | 512 | 6 | 1 | 519 | 1,560 |
| G-A / `global_stage2a` | 512 | 6 | 1 | 519 | 1,560 |
| G-B / `global_stage2b` | 512 | 6 | 1 | 519 | 1,560 |

包装器应提供 `forward_features(image)->[B,512]`，`forward(image,aux)->[B,3]`。只做 `torch.cat([global_features, aux], dim=1)` 后 `nn.Linear(fused_dim,3)`；不加隐藏 MLP、BN、注意力、门控、FiLM、投影层或额外 dropout。forward 应断言 batch 对齐、aux 维度、dtype、有限值和 availability 为 0/1；完成标准化/缺失填充后任何 NaN/Inf 都应报错。

G0 与融合组用相同 backbone 构造、ImageNet 权重、trainability 和图像 transform。正式比较必须训练新的 G0；不能只训练 G-Raw/G-A/G-B 再与旧 checkpoint 比较。

## 19. 现有 ROI/多区域融合代码的复用价值

`models/global_roi_fusion_model.py` 是 Global+ROI **多图像、独立多 backbone**，还有分支投影和额外 MLP；`models/multi_roi_fusion_nyha_3class.py` 是多 ROI 共享 backbone，但同样有隐藏层/BN/ReLU/Dropout。它们不是图像+数值特征模型，不能直接支持 512+7 且会引入不需要的复杂分支。

可复用的只有：

- `fc=Identity` 获取 ResNet GAP 特征的做法；
- 多输入 Dataset 的 ID 预检、统一样本顺序和同步翻转思想；
- ResNet18/34/50 特征维度映射和配置校验方式。

结论：新建轻量包装器比强行套用 ROI 框架更合理。

## 20. 模型选择与公平比较

保留历史 Global 协议：outer val 选择 best macro-AUC，patience=10，最多 50 epochs。必须明确它是五折 validation，不是独立 test；宏 AUC/早停会对 outer-val 性能产生选择乐观性，但只要五个 variant 完全一致，就能保留与历史协议的可比性。

不建议本次临时改成分类 inner validation：这会要求五个 variant 全部重训，并重新设计与 2A/2B inner/outer 特征边界，破坏和既有 Global 协议的直接可比性。若未来决定改协议，必须预先注册并对全部五个模型同步实施。

公平性必须额外修复 RNG：当前单图入口只在五折循环前 seed 一次。不同线性头维度会改变 RNG 消耗，进而可能使随机水平翻转序列不完全相同。新 runner 应在每个 `(fold,variant)` 重置相同 fold seed，并把模型初始化、DataLoader shuffle、augmentation 各自使用独立可复现随机流；同一 ID/epoch 的翻转决策在五个 variant 间应一致。

## 21. 最小实现文件计划

| 建议文件 | 新增/修改 | 作用 | 复用模块 | 影响历史实验 |
|---|---|---|---|---|
| `models/resnet18_optical_fusion.py` | 新增 | 统一五 variant、512/513/519→3 直接线性头 | `models.resnet_nyha_3class._build_torchvision_resnet` 的构造逻辑 | 否 |
| `datasets/global_optical_fusion_dataset.py` | 新增 | 包装单图 Dataset、按 ID 合并逐折辅助特征 | `NYHA3ClassFaceDataset`、`build_transforms` | 否 |
| `utils/optical_feature_preprocessor.py` | 新增 | 正向字段白名单、train-only scaler、缺失填充、JSON/哈希 | Stage 2 scaler 的校验/序列化风格 | 否 |
| `trainers/global_optical_fusion_trainer.py` | 新增 | 接收 image+aux、保存 scaler/variant/feature order | 现有 Trainer 协议与 metrics | 否 |
| `evaluators/global_optical_fusion_evaluator.py` | 新增 | 恢复 scaler/checkpoint、输出同构逐折预测 | 现有 Evaluator/metrics | 否 |
| `scripts/train/train_global_optical_fusion_5fold.py` | 新增 | 五 variant×五折统一训练、逐折源选择与公平 seed | 现有 loader/loss/model summary 逻辑 | 否 |
| `scripts/run/run_global_optical_fusion_5fold.py` | 新增 | preflight、resume/skip/overwrite、统一输出布局 | 当前 runner 风格 | 否 |
| `scripts/evaluate/summarize_global_optical_fusion_5fold.py` | 新增或薄包装 | 每 variant 500 行 OOF、配对汇总 | `summarize_nyha_3class_5fold.py` 的指标/OOF 逻辑 | 否 |
| `config/train/global_optical_fusion/global_resnet18_optical_fusion.yaml` | 新增 | 一份受控配置支持五 variant | 当前分层 YAML 风格 | 否 |
| `tests/test_global_optical_fusion_model.py` | 新增 | 维度、线性头、shape/finite 检查 | pytest 风格 | 否 |
| `tests/test_global_optical_fusion_dataset.py` | 新增 | ID、字段、缺失、逐折文件选择 | Stage 1/2 测试风格 | 否 |
| `tests/test_global_optical_fusion_protocol.py` | 新增 | OOF 禁用、scaler 边界、公平性和输出验收 | 现有五折协议测试 | 否 |

不建议修改历史 Stage 1、Stage 2A、Stage 2B、split、标签、图像、历史 ResNet、ROI 模型或现有 checkpoint。若希望进一步减少 trainer/evaluator 重复，可在用户确认在途修改后再把“batch forward hook”抽成通用向后兼容接口；这不是首轮实现的必要条件。

## 22. 建议配置结构

当前项目最适合继续使用分层 YAML，并由 runner 做严格 schema 验证。建议示意（本次未创建配置）：

```yaml
experiment:
  name: GlobalResNet18_OpticalFusion_NYHA3Class_5Fold
  output_root: experiments/global_resnet18_optical_fusion
  variants: [global_only, global_mask, global_raw, global_stage2a, global_stage2b]
  overwrite: false
  resume: false
  skip_completed: true

data:
  image_root: data/processed/global_face/preprocess_ablation/hybrid_imagenet_meanbg/images
  image_filename_template: "{ID}.png"
  label_path: data/raw/label_raw_nyha2_remove22_sex_balanced_500.csv
  split_root: data/processed/splits_500
  train_csv_pattern: fold_{fold}_train.csv
  val_csv_pattern: fold_{fold}_val.csv
  folds: [0, 1, 2, 3, 4]
  expected_num_samples: 500
  raw_feature_source: data/processed/optical_observations_v1/regional_optical_observations.csv
  stage2a_root: experiments/optical_condition_calibration_stage2a
  stage2b_root: experiments/optical_condition_calibration_stage2b
  feature_schema_paths:
    raw: data/processed/optical_observations_v1/feature_schema.json
    stage2a: experiments/optical_condition_calibration_stage2a/summary/calibration_feature_schema.json
    stage2b: experiments/optical_condition_calibration_stage2b/summary/calibration_stage2b_feature_schema.json

model:
  backbone: resnet18
  pretrained: imagenet
  num_classes: 3
  direct_concat: true
  optical_feature_dim: 6
  use_forehead_available: true
  global_feature_dim: 512
  auxiliary_input_dim: derived_from_variant
  fused_input_dim: derived_and_asserted

features:
  standardization: outer_train_population
  ddof: 0
  std_epsilon: 1.0e-8
  forehead_fit_requires_available: true
  missing_fill_after_standardization: 0.0
  strict_positive_allowlist: true
  forbid_oof_as_classifier_input: true

train:
  batch_size: 16
  epochs: 50
  optimizer: AdamW
  lr: 1.0e-4
  weight_decay: 1.0e-4
  loss: weighted_cross_entropy
  early_stopping_patience: 10
  monitor_metric: macro_auc
  seed: 2026
  num_workers: 0
  use_amp: false
```

`auxiliary_input_dim/fused_input_dim` 应由 variant 推导并断言，而不是允许用户随意给出不一致值。配置中可以保留 `variant` 单折运行覆盖，但完整调度应从固定 variants 清单生成，防止只给某一 variant 调参。

## 23. 后续单元测试计划

### 23.1 模型结构

- ResNet18 GAP `[B,512]`；G0/G-Mask/三融合头输入分别 512/513/519。
- 输出 `[B,3]`，头只有一个 Linear；无隐藏 MLP、BN、attention、gate、FiLM。
- G0 不接受/不读取 mask；G-Mask 只接受一维 mask；三融合组六维顺序固定。
- 非法 variant、错误 aux 维度、NaN/Inf、非二值 mask 均报错。
- 五个 variant 的 backbone 初始化张量相同；只有最终线性头形状不同。

### 23.2 Dataset/预处理

- 图像、标签、split、特征 ID 一对一；ID string 和 `-1` 后缀保持。
- 每折 400/100、无 ID/患者组交集、fold=0..4。
- Stage 2A/2B classification fold k 只读同号 fold k；train/val 文件不串用，2A/2B 不串用。
- OOF 路径作为 classifier train source 时必须立即报错。
- scaler 只用 outer train，ddof=0；val 不参与 fit。
- cheek 用全部 train；后三维只用 available train；std 过小报错。
- unavailable 14 例不删除、cheek 保留、后三维标准化后为 0、mask=0。
- scaler JSON 保存/恢复结果逐元素一致；feature order/source hash 不一致时失败。
- camera/EXIF/condition/predicted/residual/QC/NYHA 不进入 aux tensor。

## 24. 协议与验收测试计划

- 五个模型相同 meanbg 图像、fold、seed、增强、loss、epoch/patience、checkpoint 规则。
- 每 `(variant,fold)` 独立重置并隔离 RNG；同 ID/epoch 翻转决策一致。
- 每 variant 生成 500 行、500 唯一 ID OOF，每 ID 只出现一次；fold 映射与 split 相同。
- 三类概率有限、和为 1；类别顺序固定为 normal/mild/severe。
- G0 必须由同一新 runner 产生；不接受把历史 strict-blackbg OOF 复制进结果目录。
- 每折 checkpoint 必须内含 variant、feature order、scaler/hash、split/source hashes 和 ImageNet/backbone 配置。
- 2A/2B 文件完整性、sha256/ID hash 与正式 manifests 一致；不重跑三阶段。
- 所有五 variant 完成后才允许比较；不得只为一个 variant 改 early stopping、epoch、loss 或增强。

## 25. 风险登记

| 风险 | 级别 | 证据 | 控制措施 |
|---|---|---|---|
| 500 OOF 被再次拆作分类训练表 | 高 | OOF 每行来自不同 outer 校准器 | 强制逐折 train/val 路径，OOF 路径黑名单 |
| camera/EXIF/QC/raw/residual 误入模型 | 高 | 三阶段 CSV 含大量诊断列，schema forbidden 不完整 | 固定正向 allowlist + shape/order 测试 |
| val 参与六维 scaler | 高 | 当前无最终 classifier scaler 工具 | 独立 train-only preprocessor + fit/transform API |
| historical G0 与 meanbg 条件不匹配 | 高 | 完成产物无 meanbg 自包含证据；meanbg experiment 不存在 | 新 runner 重训 G0 |
| outer val 选 checkpoint 的乐观性 | 中 | 每 epoch 监控 outer-val macro-AUC | 五 variant 完全一致并称 validation；不称 test |
| variant 间增强 RNG 不同 | 中 | 单图 runner 只在五折循环前 seed | 每 fold/variant 重置并分离 RNG 流 |
| Stage 2B in-sample train vs held-out val 分布差 | 中 | 7/10 网络 val MSE 更高，最大差 0.324196 | 披露并保存分布诊断；不事后只调整 G-B |
| CPU-only torch 阻止正式 GPU 训练 | 中 | torch 2.12.0+cpu，CUDA=false | 训练前恢复受控 CUDA 环境；本次不改依赖 |
| 当前工作区在途修改冲突 | 中 | git status 有相关修改/未跟踪文件 | 优先新增文件；实现前确认归属 |
| 颜色增强跨模态冲突 | 低/当前不存在 | 仅水平翻转，无颜色/强度增强 | 保持颜色增强关闭；新增回归测试 |

## 26. 当前信息是否足以开始编码

是，前提是按本报告显式适配。所有正式数据源、字段顺序、逐折文件、ID/患者组、缺失语义、模型维度、训练协议和输出验收都已确定；不存在需要重跑 Stage 1/2A/2B 的数据阻塞。

仍未找到/仍需处理的事项并不要求用户补充新的数据文件：主要是未完成 meanbg G0、历史保存配置不自包含、CPU-only torch、schema allowlist、multi-input trainer 和 RNG 公平。详见 `unresolved_items.md`。

## 27. 本次只读验证

- 实际读取一个历史 fold 0 checkpoint 的元数据，确认 fc 形状与保存配置；未修改 checkpoint。
- 只读复跑现有 Stage 1/2A/2B 单元与协议测试：`52 passed in 7.65s`。
- 第一次测试命令因写错一个文件名而在收集前停止，显示 `no tests ran`；纠正为实际文件名后通过。
- 未运行训练、特征重算、完整五折或依赖安装。

## 28. 历史代码和数据未修改声明

本次审计没有修改或覆盖：

- `models/`、`datasets/`、`trainers/`、`evaluators/`、`scripts/`、`config/`；
- `data/processed/splits_500/`、标签文件、图像；
- Stage 1、Stage 2A、Stage 2B 的任何输入、模型或输出；
- 任何历史实验 checkpoint、log、metrics 或 OOF。

唯一写入是新建 `reports/global_resnet18_optical_fusion_preimplementation/` 及其五个审计文件。

## 29. 最终结论（逐项回答）

1. **当前结构是否支持最小改动实现融合？** 支持，但需新增轻量多输入通路和 train-only scaler。
2. **推荐复用哪个 Global ResNet18？** 复用 `models/resnet_nyha_3class.py` 的 ResNet18 构造/训练协议，以 `config/train/preprocess_ablation_resnet18/nyha_3class_resnet18_preproc_hybrid_imagenet_meanbg.yaml` 为 G0 配置基础。
3. **新建模型还是修改现有模型？** 新建 `ResNet18OpticalFusion` 包装器，不改历史模型。
4. **新建 Dataset 还是扩展现有 Dataset？** 新建组合式 `GlobalOpticalFusionDataset`，内部复用历史图像 Dataset，避免改变旧行为。
5. **Trainer/Evaluator 能否复用？** 协议、loss、metrics、输出格式可复用；具体类不能原样处理 aux，推荐新建薄适配类。
6. **Stage 1/2A/2B 正式输入是什么？** Stage 1 为 `data/processed/optical_observations_v1/regional_optical_observations.csv`；2A/2B 为各自 experiment root 下同号 fold 的 train/val 特征文件。
7. **分类 fold k 如何读取 2A/2B？** 2A 读 `fold_k/train_calibrated_features.csv` 与 `fold_k/val_calibrated_features.csv`；2B 读 `fold_k/train_nn_calibrated_features.csv` 与 `fold_k/val_nn_calibrated_features.csv`。
8. **是否存在 ID/split 不一致？** 正式 meanbg、标签、split、Stage 1/2A/2B 均完全一致；未发现不一致。
9. **是否存在颜色增强冲突？** 不存在；当前只有水平翻转，无颜色/强度增强。
10. **是否需要重训匹配 G0？** 需要；原因是未找到已完成 meanbg G0，且必须与四个辅助输入组由同一新 runner 公平训练。
11. **是否需要重跑 Stage 1/2A/2B？** 不需要；逐折文件完整且协议测试通过。
12. **是否支持 G-Mask 控制组？** 支持；只读取 Stage 1 的 `forehead_available`，输入维度 513。
13. **预计新增/修改哪些文件？** 建议新增模型、Dataset、preprocessor、trainer、evaluator、train/run/summary、YAML 和三类测试；首轮不修改历史文件。
14. **是否具备正式编码全部信息？** 是，按显式适配与验收条件执行即可。
15. **若不能开始编码，缺少什么？** 编码不缺数据/协议；正式 GPU 训练前需恢复 CUDA PyTorch，并由用户确认当前未提交在途修改的归属。

**最终状态：`READY_WITH_EXPLICIT_ADAPTATIONS`**

## 30. 下一步正式实现的硬性验收条件

1. 只使用正式 meanbg 500 ID 和 `splits_500`；G0 同时重训。
2. 2A/2B 逐分类 fold 读取同号 train/val；OOF 绝不作为 classifier train source。
3. 六维 positive allowlist 与 schema/order 双重断言；camera/EXIF/condition/predicted/residual/QC 永不进入模型。
4. scaler 只 fit outer train，ddof=0；后三维仅 available；标准化后 unavailable 填 0；14 例不删除。
5. 模型严格 512/513/519→3 单线性头，不增加复杂融合模块。
6. 五个 variant 相同 backbone、meanbg、fold、seed、增强、loss、训练预算和 checkpoint 规则。
7. 每 variant OOF 为 500 行/500 唯一 ID/每 ID 一次，概率合法，fold/类别顺序一致。
8. checkpoint/manifest 完整保存 variant、feature order、scaler、source/split hashes，可无歧义恢复。
9. 单元和协议测试全部通过后才能启动正式五折训练。
10. 不覆盖用户当前未提交代码，不修改 Stage 1/2A/2B、split、标签或图像。
