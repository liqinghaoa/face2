   你正在 Windows 项目根目录：

```text
E:\projects\face2
```

请执行：

```text
SO-R2-X4
Exploratory Nested-Direct Five-Fold MH-LRTM Classification
```

本阶段目标是：在已冻结的真实 RGB、B2 M-sensitive、B2 H-sensitive 分类输入上，按与 SO-R2-X3 完全同源的五折 nested-direct 协议，探索三种融合结构的内部 OOF 分类表现。

训练并比较：

```text
LATE_CONCAT
MH_LRF
MH_LRTM
```

本阶段属于：

```text
EXPLORATORY_INTERNAL_OOF_EVIDENCE
NOT_FORMAL_SO_R2_GATE
NOT_CLINICAL_EVIDENCE
NOT_MODEL_SELECTION
```

---

# 1. 严格状态边界

无论本轮结果如何，必须保持：

```text
SO-R1-C = FAIL
official_SO_R2_authorization = false
SO_R3_authorization = false
next_stage_authorized = false
```

M/H 必须仅称为：

```text
M-sensitive representation
H-sensitive representation
```

不得称为真实、绝对或定量的黑色素/血红蛋白浓度。

禁止：

```text
修改 B1/B2 checkpoint
修改 B2 λ=0.50
修改 SO-R1-C 结论
修改 P0 输入 tensor、mask、split、label
重新生成 M/H map
训练或微调 B1/B2 decomposition model
读取未授权临床变量
启动临床关联、显著性检验、bootstrap 或正式 SO-R2 Gate
```

---

# 2. 前置 Gate

训练前必须验证：

```text
SO-R2-X3 classifier-input preparation = COMPLETE_CLASSIFIER_INPUT_PREPARATION
SO-R2-X4-S0-R1 v2 = PASS_SMOKE_TEST
```

权威 smoke 输入来自：

```text
E:\projects\face2\data\processed\SO_R2X4_MHLRTMSmokeTest_v2\S0_ACCEPTANCE.json
```

必须确认：

```text
LATE_CONCAT 不实例化 rgb_head；
LATE_CONCAT parameter count = 13,126,762；
MH_LRF parameter count = 13,212,140；
MH_LRTM parameter count = 13,376,940；
batch=16 CUDA FP32 smoke 已通过；
固定 smoke manifest 实际构成为 3 Control / 13 Patient。
```

如任一 Gate 不符，停止，不得训练。

---

# 3. 冻结输入与通道语义

唯一允许的分类输入目录：

```text
E:\projects\face2\data\processed\SO_R2X3_ClassifierInputs_v1\rgb_b2mh
```

每例 tensor 必须为：

```text
shape = (5, 320, 256)
dtype = float32

channel 0–2 = linear-RGB
channel 3   = B2 M-sensitive map
channel 4   = B2 H-sensitive map
```

拆分固定为：

```python
x_rgb = x[:, 0:3, :, :]
x_m   = x[:, 3:4, :, :]
x_h   = x[:, 4:5, :, :]
```

禁止：

```text
交换 M/H；
使用 B1 map；
使用额外 mask channel；
使用 original raw RGB；
使用其他目录中的 256×320 RGB；
重新 resize、crop、gamma、颜色校正、clamp；
重新估计 normalization mean/std。
```

必须复用 X3 与 X3-MH 已有的 loader、normalization 和训练期 augmentation 语义。

若 X3 启用了空间增强，则 RGB、M、H 必须使用完全同步的几何变换；不得对三者独立随机翻转。

不得新增：

```text
color jitter
random crop
random resize
mixup
cutmix
TTA
ensemble
```

---

# 4. 固定五折验证合同

必须直接读取并复用 SO-R2-X3 实际使用的：

```text
outer fold 定义
outer-test ID
inner-train ID
inner-validation ID
patient-group 处理规则
label × SEX 分层规则
seed
训练超参数
checkpoint 选择逻辑
OOF 评估逻辑
```

禁止重新用不同 sklearn 随机划分逻辑“近似复现”X3。

每一 outer fold 的数据边界必须与 X3 完全一致：

```text
outer dev = 400 cases
outer test = 100 cases

outer dev 内：
inner train = 320 cases
inner validation = 80 cases
```

如 X3 的固定 split 已包含 patient-group 约束，则 X4 必须逐项复用，并审计：

```text
train ∩ validation = 0
train ∩ outer-test = 0
validation ∩ outer-test = 0
patient-group overlap = 0（如 X3 合同要求）
```

允许读取的字段仅限：

```text
case_id
binary_label
fold / split
SEX（仅用于复用既有分层和审计）
patient_group_id（仅在既有 X3 合同要求时用于 split 审计）
```

禁止读取：

```text
原始 NYHA 分级
其他临床标签
BNP/NT-proBNP
超声指标
血常规
肝肾功
药物、诊断、结局或其他临床变量
```

SEX 不得作为模型输入或分类特征。

---

# 5. 三个待训练模型

## 5.1 LATE_CONCAT

```text
RGB ResNet-18 → z_rgb (512)
M encoder      → z_m   (128)
H encoder      → z_h   (128)

concat(z_rgb, z_m, z_h) = 768
Linear(768,256) → ReLU → Dropout(0.2) → Linear(256,2)
```

必须满足：

```text
不实例化 rgb_head；
state_dict 中不存在 rgb_head；
RGB branch 仅输出 embedding；
唯一分类头为 late_concat_classifier；
total parameters = 13,126,762。
```

## 5.2 MH_LRF

固定使用：

```text
RGB ResNet-18；
独立 M/H GroupNorm residual encoder；
M/H embedding = 128；
M/H augmented embedding = 129；
low-rank fusion rank = 4；
LMF output = 128；
RGB–MH residual interaction；
final_logits = rgb_logits + 0.1 × delta_logits。
```

```text
total parameters = 13,212,140
```

## 5.3 MH_LRTM

在 MH_LRF 基础上增加：

```text
MMTM-1:
RGB layer2 + M stage2 + H stage2

MMTM-2:
RGB layer3 + M stage3 + H stage3

gate = 1 + 0.1 × tanh(gate_logits)
gate head weight ~ Normal(0, 1e-3)
gate head bias = 0
```

```text
total parameters = 13,376,940
```

不得修改：

```text
M/H encoder 宽度
GroupNorm
LMF rank
MMTM 位置
gate 公式
residual scale
dropout
参数初始化
```

每个 outer fold、每个模型均从独立且确定性的初始状态开始：

```text
seed = 12026 + fold_index
```

必须在模型构建、DataLoader 与增强随机状态设置前固定 seed。

RGB 分支只能使用与 X3 相同的 ImageNet-pretrained ResNet-18 初始化，不得加载已有 X3、X3-MH、B1、B2 或其他分类 checkpoint。

M/H encoder 与融合模块必须从随机初始化开始。

---

# 6. 训练合同

必须逐项复用 X3 的训练合同；当前预期为：

```text
device = CUDA
precision = FP32
batch size = 16
optimizer = AdamW
learning rate = 1e-4
weight decay = 1e-4
scheduler = absent
max epochs = 50
full fine-tuning = true
```

损失必须复用 X3：

```text
2-class logits，class order = [Control, Patient]
weighted cross-entropy
exclude-true-class label smoothing = 0.05
class weights 仅由对应 fold 的 inner-train 计算
```

必须重新核对 X3 代码；如果实际已冻结合同与上述任一项不同，以 X3 的真实实现为准，并在 `protocol_reuse_audit.json` 中逐项说明，不得自行选择更有利的超参数。

每个模型、每个 outer fold：

```text
只用 inner-train 更新参数；
每个 epoch 后仅在 inner-validation 计算 Macro-AUC；
以 inner-validation Macro-AUC 选择 checkpoint；
strictly greater 才更新最佳 checkpoint；
并列时保留更早 epoch；
patience = 10；
不得根据 outer-test 选择 epoch、阈值、模型或超参数。
```

每个 fold 只在最佳 inner-validation checkpoint 冻结后，对该 fold 的 outer-test 执行一次预测。

15 个训练 run 必须串行执行：

```text
3 models × 5 outer folds
```

不得在同一张 RTX 4060 Laptop GPU 上并发训练。

---

# 7. OOF 与指标

每个模型必须得到：

```text
5 folds
500 / 500 unique case OOF coverage
每个 case 恰好一次 outer-test prediction
```

固定输出：

```text
case_id
outer_fold
true_label
predicted_probability_control
predicted_probability_patient
predicted_label
selected_epoch
model_name
```

分类阈值固定：

```text
argmax logits / probability 0.5
```

不得根据 OOF 重新校准或寻找最优阈值。

对每个模型报告：

```text
Pooled OOF:
Macro-AUC
Accuracy
Macro-Precision
Macro-Recall
Macro-F1
Balanced Accuracy
Sensitivity
Specificity

Five-fold:
上述关键指标的 mean ± SD
每折 selected epoch
```

不得计算：

```text
bootstrap CI
p value
DeLong
Wilcoxon
显著性结论
自动 winner
```

---

# 8. 固定描述性比较

仅在三个新模型完整 OOF 已冻结后，进行预先指定的描述性比较：

```text
Primary:
MH_LRTM − RGB existing X3 pooled OOF Macro-AUC

Architecture ablations:
MH_LRF − LATE_CONCAT
MH_LRTM − MH_LRF
MH_LRTM − LATE_CONCAT

Contextual frozen references:
MH_LRTM − RGB_B2MH existing X3
MH_LRTM − B2_H_ONLY existing X3-MH
```

这些比较必须：

```text
只报告 absolute pooled OOF metric difference；
同时报告 per-fold difference mean ± SD；
不做显著性检验；
不写“优于”“胜出”“最佳模型”等结论；
不据此自动选择后续模型。
```

既有 RGB、RGB_B2MH、B2_H_ONLY OOF 只能在本轮所有新模型训练和 outer-test 预测完成后读取，用于只读报告比较。

必须分别记录：

```text
outer_test_model_selection_access = 0
posthoc_frozen_reference_oof_read = allowed and audited
```

---

# 9. 审计与异常处理

训练前后审计：

```text
P0 classifier inputs
X3 split manifest
X3 OOF results
X3-MH OOF results
X4-S0-R1 v2 acceptance
B1/B2 checkpoint and acceptance references
X1/X2 frozen maps and audits
```

要求：

```text
changed = 0
missing = 0
```

仅允许向新的 X4 输出目录写入：

```text
runs/
reports/
data/processed/
config/
src/
scripts/
tests/
```

任何下列事件均须报告并将最终状态写为失败，不得静默改变合同继续训练：

```text
NaN / Inf
CUDA OOM
checkpoint selection 使用 outer-test
split leakage
case OOF 重复或遗漏
参数量偏离 smoke 合同
M/H channel order 错误
输入哈希变化
受保护资产变更
```

允许为中断恢复写入本次 X4 自身的 checkpoint，但恢复必须：

```text
同一 fold
同一模型
同一 seed
同一 split
同一超参数
同一训练状态
```

并记录恢复前后 provenance。不得以失败为由改 batch、学习率、模型宽度、rank 或训练轮数。

---

# 10. 新增/扩展实现

建议新增：

```text
src/skin_optics_so_r1/mh_lrtm_exploratory_training.py

scripts/so_r1/
  run_so_r2x4_mh_lrtm_exploratory_classification.py

tests/so_r1/
  test_so_r2x4_mh_lrtm_exploratory_classification.py
```

新增冻结协议：

```text
config/so_r2/
  so_r2_x4_mh_lrtm_exploratory_protocol_v1.yaml
  SO_R2_X4_MH_LRTM_EXPLORATORY_LOCK.json
```

lock 至少包含：

```text
input acceptance hashes
X3 split hash
X4-S0-R1 v2 acceptance hash
model variants
parameter counts
seed schedule
optimizer and loss contract
selection metric and tie-break
fixed comparisons
prohibited analyses
```

---

# 11. 输出

训练输出：

```text
runs/so_r2x4_exploratory_mh_lrtm/
  late_concat/
    fold_0/ ... fold_4/
  mh_lrf/
    fold_0/ ... fold_4/
  mh_lrtm/
    fold_0/ ... fold_4/
```

每折仅保存该 fold 的：

```text
best inner-validation checkpoint
training log
selection provenance
outer-test prediction
```

验收文件：

```text
data/processed/SO_R2X4_ExploratoryMHLRTMClassification_v1/
  X4_ACCEPTANCE.json
  evaluation_run_manifest.json
```

报告目录：

```text
reports/so_r2x4_exploratory_mh_lrtm/
  SO_R2X4_Exploratory_MH_LRTM_Classification_Report.md
  protocol_reuse_audit.json
  model_parameter_audit.csv
  fold_metrics.csv
  oof_metrics.csv
  oof_predictions.csv
  selected_epochs.csv
  fixed_comparisons.csv
  split_leakage_audit.json
  field_access_audit.json
  training_exception_audit.json
  data_access_audit.json
  protected_asset_hash_audit.json
  test_results.txt
```

---

# 12. 专项测试

至少覆盖：

1. 仅接受 X4-S0-R1 v2 `PASS_SMOKE_TEST` 输入；
2. 三个模型参数量严格等于 smoke 合同；
3. LATE_CONCAT 不存在 `rgb_head`；
4. MH_LRF 与 MH_LRTM 的 LMF/MMTM 合同正确；
5. 每个模型只使用 RGB、M、H 正确通道；
6. M/H encoder 不共享权重；
7. 同一图像的 RGB/M/H 几何增强同步；
8. X3 outer/inner split 被逐项复用；
9. inner-train/validation/outer-test 无 ID 泄漏；
10. 如适用，patient-group 无泄漏；
11. class weight 只从 inner-train 计算；
12. best epoch 仅由 inner-validation Macro-AUC 与早期 tie-break 决定；
13. outer-test 在每折只于最佳 checkpoint 预测一次；
14. 500/500 OOF 完整且无重复；
15. 固定 0.5 / argmax 阈值，不做优化；
16. 既有 RGB / H-only OOF 只在训练完成后读取；
17. 无 bootstrap、p value、winner 选择；
18. protected assets 前后不变；
19. SO-R1-C、SO-R2、SO-R3 正式状态不变。

运行：

```powershell
conda run -n face2 python -m pytest -q tests/so_r1
git diff --check
```

不得执行 Git commit。

---

# 13. 最终验收与汇报

完成后写入：

```text
status = COMPLETE_EXPLORATORY_MH_LRTM_CLASSIFICATION
```

仅表示 15 个预冻结探索 run 已完成且 OOF 完整。

不得写入：

```text
official SO-R2 PASS
SO-R3 authorized
best model selected
clinical validity established
M/H absolute physiology validated
```

最终请报告：

1. X4 最终状态；
2. 是否逐项复用 X3 的 fold、inner split、seed、训练和 checkpoint-selection 合同；
3. 三个模型每折 selected epoch；
4. 三个模型 pooled OOF 与五折 mean ± SD 指标；
5. 三个固定融合比较和相对 RGB / RGB_B2MH / B2_H_ONLY 的描述性差异；
6. 500/500 OOF 覆盖与 split/group 泄漏审计；
7. 是否发生 NaN、Inf、OOM、恢复训练或超参数变更；
8. 各模型的参数量、FP32/batch=16 运行情况；
9. 受保护资产审计；
10. 字段访问审计；
11. `pytest` 与 `git diff --check` 结果；
12. 是否确认未执行 bootstrap、显著性检验、阈值调优、自动 winner 选择；
13. 是否确认 `SO-R1-C=FAIL`、SO-R2/SO-R3 仍未授权。
