# 面部心功能二分类研究：P2 物理反事实重光照阶段研究与实现路线（精简修订版）

> 文档用途：  
> 本文档用于替代旧版 `Face_Cardiac_P2_Physics_Relighting_Research_Plan.md`，作为后续 ChatGPT、Codex、实验实现和结果分析推进 P2 阶段的统一依据。  
> 后续工作必须同时遵守本文件与 `Face_Cardiac_Physics_Informed_Research_Roadmap.md`，不得将 P2 再次扩展为重复的数据审计工程，也不得提前进入 P3 或 P4。

---

## 1. 项目定位

### 1.1 研究任务

基于普通手机拍摄的对齐正脸 RGB 图像，对以下两类进行探索性二分类：

- Control：NYHA 0，无病对照；
- Patient：NYHA I–IV，标签 1–4。

当前数据规模：

- 500 个 visit/case；
- 483 个 patient group；
- Control 115 例；
- Patient 385 例；
- 使用固定患者组级五折交叉验证；
- 同一患者的不同 visit 必须位于同一 fold；
- 正式评价单位为 visit/case；
- `patient_group_id` 仅用于分组划分和 patient-cluster bootstrap；
- 不对同一患者的多个 visit 进行预测平均。

### 1.2 正式论文主线

```text
冻结物理分解先验
        +
物理反事实重光照
        +
采集条件显式建模
        +
疾病—采集表示解耦
        +
可靠性门控融合
```

当前 P2 只研究：

```text
物理反事实重光照
+
反事实预测/特征一致性
```

P2 不得提前进入：

- P3 采集条件显式建模；
- P3 疾病—采集表示解耦；
- P4 可靠性门控融合；
- Camera/EXIF 作为模型输入；
- 对抗解耦、ComBat 或风格迁移；
- 注意力、门控或复杂融合模块；
- 重新训练 DECA；
- 使用分类标签更新物理前端。

---

## 2. P2 阶段的核心研究问题

P2 只回答两个主要问题。

### 问题一：物理重光照是否优于普通颜色增强

比较：

```text
无额外光度增强
vs
标准 ColorJitter
vs
物理反事实重光照
```

判断物理约束重光照是否能：

- 保持或提高原图分类性能；
- 比 ColorJitter 提供更好的跨光照稳定性；
- 减少模型对特定照明条件的敏感性；
- 不明显损害 Patient sensitivity 与 Control specificity。

### 问题二：一致性学习是否提供额外增益

在物理重光照有效的前提下，进一步比较：

```text
物理重光照
vs
物理重光照 + 完整反事实一致性
```

判断预测一致性和特征一致性联合约束是否能进一步：

- 降低原图与反事实图之间的预测波动；
- 降低标签翻转率；
- 提高最差光照条件下的性能；
- 保持或提高原图分类性能。

P2 不以单纯提高 AUC 为唯一目标，同时关注分类能力和跨光照稳定性。

---

## 3. 上游资产继承原则

### 3.1 基本原则

P0、P0B 和 P1 已经完成并正式冻结的资产和结论，P2 默认继承，不重复执行。

统一遵守：

> 上游已正式冻结并有报告、manifest、哈希或审计结果支持的内容，下游阶段默认继承。  
> 只有 P2 新生成、新组合或新使用方式涉及的内容，才需要新增验证。

### 3.2 P2 直接继承的内容

#### 来自 `P0_Physics_Audit_v1`

直接继承：

- 原始场景图；
- 对齐 224×224 RGB 图像；
- face mask、face-valid mask、skin mask；
- physics-core-skin mask；
- 左右面颊和额部 ROI mask；
- alignment landmarks 和对齐变换；
- 固定五折；
- `patient_group_id`、标签和 fold；
- 两例下颌—颈部边界不确定性标记。

#### 来自 `P0B_DECA_Pilot12_v1`

直接继承：

- DECA 在本项目图像上的可行性结论；
- 冻结输入模式 `direct_p0_aligned`；
- 六套固定 SH 重光照的工程可行性；
- 非坍塌审计和盲法 QC 结论；
- Pilot12 样本与既有 QC 形式。

#### 来自 `P0B_cross_process_regression_baseline_v2`

直接继承：

- DECA 输出跨进程回归思路；
- 渲染输出回归容差和复现性证据；
- 环境指纹和资产哈希记录方法。

#### 来自 `P0B_cross_process_production_gate_v2_1`

直接继承：

- DECA checkpoint 和资产哈希；
- 固定配置和环境记录；
- 生产放行合同；
- 上游冻结证据。

#### 来自 `P1_DECA_Frozen500_v1`

直接继承：

- 500/500 例正式 DECA 输出；
- 500 例 `latents.npz`；
- 500 例 `maps.npz`；
- 500 例 `relighting.npz`；
- 500×6=3000 张固定 SH 重光照 RGB；
- 六套 preset 名称和 SH 系数；
- `quality.json`、`provenance.json`、`_SUCCESS.json`；
- input/output/quality manifests；
- DECA 配置、checkpoint、环境与资产哈希。

P1 源目录必须只读，不得修改，也不得重新运行 DECA。

#### 来自 `P1_Component_Audit_v1`

直接继承：

- 500 行统一病例索引；
- P0、split 与 P1 Frozen500 的 join 结果；
- 资产完整性和哈希一致性结果；
- 标签与 fold 一致性结果；
- 3000 行重光照基础统计；
- 16 例 QC flag；
- 两例边界不确定病例的状态。

### 3.3 P2 不再重复执行的工作

P2 不再重新进行：

- 全量 DECA 可行性审计；
- 全量物理分解非坍塌审计；
- 全量上游资产完整性重建；
- 新一轮盲法 DECA 质量审核；
- 全量身份识别审计；
- 全量 landmark 稳定性审计；
- 全量皮肤颜色漂移分析；
- 重新生成已有 3000 张重光照 RGB；
- 复制已有 RGB 重光照形成重复数据版本；
- 重新建立五折、标签或病例主索引。

若后续训练发现明确异常，再针对异常开展诊断，而不是在训练前穷举所有潜在问题。

---

## 4. 当前 P1 主模型与 P2 基线

### 4.1 当前最佳模型

当前 P1 最佳模型为：

```text
P1-RGBSR_TripleBranch_v1
```

模型结构：

```text
RGB
 └─ Independent ResNet18 → 512D

Shading-like
 └─ Independent ResNet18 → 512D

Signed Residual
 └─ Independent ResNet18 → 512D

Concat: 512 + 512 + 512 = 1536D
 └─ Linear(1536, 2)
```

固定特征：

- 三个 ResNet18 独立；
- 三套 BatchNorm 独立；
- ImageNet 初始化；
- 三分支端到端微调；
- 无 Bottleneck；
- 无 Dropout；
- 无 LayerNorm；
- 无注意力或门控；
- 无 Camera/EXIF 输入；
- 无分类阈值搜索；
- 无 patient-level 预测聚合。

### 4.2 当前 OOF 结果

| 指标 | P1-RGBSR |
|---|---:|
| Macro-AUC | 0.8781 |
| Macro-F1 | 0.7710 |
| Balanced Accuracy | 0.7811 |
| Patient Sensitivity | 0.8753 |
| Control Specificity | 0.6870 |

混淆矩阵：

```text
TN = 79
FP = 36
FN = 48
TP = 337
```

P2-B 若被触发，应以该模型作为正式主模型基线。

---

## 5. P2 总体执行结构

P2 保留三个逻辑环节，但每个环节必须保持最小化，不得继续无限细分。

```text
P2-0：最小训练接口准备
        ↓
P2-A：单 RGB 主效应验证
        ↓
结果门控
        ↓
P2-B：接入 P1-RGBSR 主模型
```

其中：

- P2-0 是轻量数据接口任务，不是重新执行 P0；
- P2-A 先验证主效应，不提前完成全部消融；
- P2-B 不是必然执行，必须由 P2-A 结果触发；
- P2 完成后不自动进入 P3。

---

# 6. P2-0：最小训练接口准备
        
## 6.1 目标

P2-0 只解决一个问题：

> 现有冻结物理资产是否已经可以被 P2 训练代码正确、稳定地读取，并补齐 P2-B 所需的六套重光照 shading。

P2-0 不训练分类模型，不重新审计全部物理资产。

## 6.2 已有输入

源目录：

```text
E:\projects\face2\data\processed\P1_DECA_Frozen500_v1
```

统一索引：

```text
E:\projects\face2\data\processed\P1_Component_Audit_v1
```

P0 辅助资产：

```text
E:\projects\face2\data\processed\P0_Physics_Audit_v1
```

每例已有：

```text
maps.npz:
- input_aligned_rgb
- reconstruction
- alpha
- albedo_like
- normal_coarse
- shading_like
- signed_residual
- absolute_residual

relighting.npz:
- preset_names
- sh_coefficients
- relighted_images
```

固定六套光照：

```text
neutral_front
left
right
top
dim_front
bright_front
```

必须通过 `preset_names` 查找对应光照，不得只依赖数组下标。

## 6.3 P2-0 必做任务

### 任务一：确认现有重光照接口

只读检查真实生产代码、配置和少量实际文件，确认：

- `relighting.npz` 的真实字段；
- `relighted_images` 的 shape、dtype 和数值范围；
- `relighted_images` 是否可直接作为训练 RGB；
- 是否已经完成 alpha 和背景合成；
- SH shading 函数所在代码位置；
- SH basis 与系数顺序；
- `normal_coarse` 的坐标约定；
- 是否使用 detail normal；
- 是否使用 signed residual；
- 存储通道顺序与颜色空间。

输出简化协议：

```text
P2_Relighting_Protocol_v1.json
```

此步骤只用于防止训练代码误读已有数组，不重新验证物理分解本身。

### 任务二：派生六套 `relighted_shading`

使用：

```text
normal_coarse
+
对应 preset 的 sh_coefficients
+
P1 生产阶段实际使用的 SH shading 函数
```

生成：

```text
relighted_shading: (6, 224, 224, 3)
```

要求：

- 不重新运行 DECA encoder；
- 不修改 P1 源目录；
- 不自行设计新的 SH 公式；
- preset 与 RGB 重光照严格一一对应。

### 任务三：生成 P2 训练 manifest

建立统一训练接口，至少包含：

```text
case_id
patient_group_id
fold
binary_label
original_rgb_path
maps_npz_path
relighting_npz_path
relighted_shading_path
boundary_uncertain
p1_qc_flag
```

训练 manifest 直接引用 P1 中已有重光照 RGB，不复制已有 3000 张 RGB。

### 任务四：最小必要验证

只验证 P2 新增内容和读取接口：

- 500/500 例可读取；
- 每例包含六套 preset；
- 3000/3000 `relighted_shading` 完整；
- shape 为 `(6,224,224,3)`；
- 所有新增数组 finite；
- RGB 与 shading 的 preset 名称完全一致；
- `dim_front < neutral_front < bright_front` 总体趋势合理；
- left/right preset 总体方向没有系统性反转；
- DataLoader 能正确返回原图和同一 preset 对应的 RGB/shading；
- P1 源目录哈希未改变；
- 16 例 QC flag 和两例边界不确定病例全部保留并标记。

可生成少量 contact sheet 抽样复核，但不做新的全量人工审计。

## 6.4 推荐输出目录

```text
E:\projects\face2\data\processed\P2_Counterfactual_Relighting500_v1
```

最小目录结构：

```text
P2_Counterfactual_Relighting500_v1/
├── cases/<case_id>/
│   └── relit_shading.npz
├── manifests/
│   └── p2_training_manifest.csv
├── metadata/
│   ├── P2_Relighting_Protocol_v1.json
│   ├── preset_registry.json
│   ├── derivation_config.yaml
│   └── P2_COUNTERFACTUAL_TRAINING_ASSETS_FROZEN.json
├── qc/
│   └── sample_contact_sheets/
└── reports/
    └── p2_0_asset_preparation_report.md
```

不要求复制已有 RGB 重光照，不要求重新保存全部 P1 maps。

## 6.5 P2-0 完成标准

满足以下条件即可进入 P2-A：

1. 真实重光照接口已确认；
2. 500/500 例训练 manifest 完整；
3. 3000/3000 `relighted_shading` 完整；
4. 六套 preset 与 RGB 重光照严格对应；
5. 新增数组全部 finite；
6. DataLoader 接口通过；
7. 抽样重光照和 shading 无明显系统错误；
8. 上游 P1 源目录未修改；
9. QC flag 和边界病例未被删除；
10. 已输出：

```text
P2_COUNTERFACTUAL_TRAINING_ASSETS_FROZEN
```

11. 未进行任何分类训练。

---

# 7. P2-A：单 RGB 主效应验证

## 7.1 目的

P2-A 使用单 RGB ResNet18 隔离验证物理重光照机制，不接入 P1-RGBSR 三分支结构。

P2-A 只回答：

1. ColorJitter 是否优于无增强；
2. 物理重光照是否优于 ColorJitter；
3. 完整反事实一致性是否在物理重光照基础上提供额外增益。

## 7.2 核心实验

| 编号 | 实验 | 作用 |
|---|---|---|
| P2-A0 | RGB，无额外光度增强 | 无增强基线 |
| P2-A1 | RGB + ColorJitter | 普通光度增强对照 |
| P2-A2 | RGB + 物理重光照 | 验证物理增强主效应 |
| P2-A3 | RGB + 物理重光照 + 完整一致性 | 验证一致性增量 |

若 P1-RGB 与 P2-A0 的数据、模型和训练协议完全一致，可复用已有 P1-RGB 结果；若代码路径或训练协议不同，则在统一 P2 runner 中重跑 P2-A0。

暂不正式运行：

```text
仅预测一致性
仅特征一致性
```

这些属于可选机制消融，仅在完整一致性取得明确正结果后再考虑补做。

## 7.3 固定模型与训练协议

统一使用：

```text
ImageNet-pretrained ResNet18
+
Linear(512, 2)
```

固定训练协议：

```text
Optimizer: AdamW
Learning rate: 1e-4
Weight decay: 1e-4
Batch size: 16
Max epochs: 50
Early stopping patience: 10
Loss: training-fold weighted cross entropy
Seed: 2026
Checkpoint metric: original validation visit/case Macro-AUC
```

除增强和一致性机制外，不改变：

- 模型结构；
- 数据划分；
- 分类损失；
- 优化器；
- 学习率；
- early stopping；
- checkpoint 规则；
- 分类阈值；
- 随机种子；
- 基础几何增强。

## 7.4 P2-A0：无额外光度增强

训练增强仅：

```text
RandomHorizontalFlip(p=0.5)
```

不使用 ColorJitter，不使用重光照。

## 7.5 P2-A1：ColorJitter

固定：

```python
ColorJitter(
    brightness=0.30,
    contrast=0.30,
    saturation=0.20,
    hue=0.05
)
```

应用概率：

```text
p_color_jitter = 0.5
```

仅训练使用，不搜索参数。

## 7.6 P2-A2：物理重光照增强

每个训练样本：

```text
50% 使用原始 RGB
50% 使用一套物理反事实重光照 RGB
```

使用反事实图时：

```text
六套光照均匀随机采样
P(light_k) = 1/6
```

禁止：

- 根据标签选择光照；
- 根据 fold 选择光照；
- 根据 Camera/EXIF 选择光照；
- 根据实验结果筛选光照；
- 与 ColorJitter 叠加；
- 动态搜索 SH 系数。

## 7.7 P2-A3：物理重光照与完整一致性

每个训练样本同时读取：

```text
I_original
I_counterfactual
```

共享同一个 ResNet18，得到：

```text
p_original
p_counterfactual
z_original
z_counterfactual
```

分类损失：

```text
L_cls =
0.5 × CE(y, p_original)
+
0.5 × CE(y, p_counterfactual)
```

预测一致性：

```text
m = 0.5 × (p_original + p_counterfactual)

L_pred =
0.5 × KL(p_original || m)
+
0.5 × KL(p_counterfactual || m)
```

特征一致性：

```text
L_feat =
1 - cosine(z_original, z_counterfactual)
```

总损失：

```text
L =
L_cls
+
0.5 × L_pred
+
0.1 × L_feat
```

一致性权重使用 5 epoch 线性 warm-up：

```text
Epoch 1：20%
Epoch 2：40%
Epoch 3：60%
Epoch 4：80%
Epoch 5及以后：100%
```

第一轮不搜索一致性损失权重。

---

# 8. P2-A 评价体系

## 8.1 核心分类指标

每个模型生成 500 例原图 OOF，核心报告：

```text
Macro-AUC
Macro-F1
Balanced Accuracy
Patient Sensitivity
Control Specificity
```

程序可同时保存：

- Accuracy；
- Macro-Precision；
- Macro-Recall；
- PR-AUC；
- confusion matrix。

## 8.2 核心稳定性指标

使用对应 fold 最佳 checkpoint，对每个 OOF 病例推理：

```text
原图
+
六套反事实重光照图
```

P2 主报告只保留三个核心稳定性指标：

### 预测波动

```text
prediction_std_i =
Std(p_original, p_cf1, ..., p_cf6)
```

汇总均值和中位数。

### 病例级标签翻转率

```text
任一反事实图预测标签
与原图预测标签不同
```

报告 case-level label flip rate。

### 最差光照 AUC

分别计算六套重光照下的 AUC，并报告：

```text
worst-light AUC
```

可在程序中保存但不作为主报告重点的附加指标：

- maximum probability difference；
- view-level flip rate；
- feature cosine similarity；
- 六套光照逐一 AUC；
- mean relighted AUC；
- AUC range。

对于 P2-A3，可额外重点报告原图与反事实图的特征余弦相似度。

## 8.3 统计比较

基于相同 OOF 病例进行：

- paired comparison；
- patient-cluster bootstrap；
- Macro-AUC 差值及 95% CI；
- Macro-F1 差值；
- Balanced Accuracy 差值；
- sensitivity 和 specificity 差值；
- 稳定性指标差值。

不生成正式 patient-averaged OOF。

---

# 9. P2-A 结果门控

## 9.1 第一层门控：物理重光照是否有效

核心比较：

```text
P2-A2 vs P2-A1
```

物理重光照被认为值得继续研究，需至少满足以下之一：

- Macro-AUC、Macro-F1 或 Balanced Accuracy 有改善；
- sensitivity/specificity 平衡改善；
- 原图分类性能基本保持，同时 prediction std 明显降低；
- case-level label flip rate 明显降低；
- worst-light AUC 明显提高。

同时不得出现：

- 原图分类性能明显崩溃；
- Patient sensitivity 明显下降；
- Control specificity 明显下降；
- 重光照产生明显伪影捷径；
- 稳定性指标反而恶化。

若 P2-A2 不优于 ColorJitter，或分类与稳定性均无价值，则：

```text
停止 P2
不进入 P2-B
不再补做一致性消融
```

## 9.2 第二层门控：完整一致性是否有增量

仅当 P2-A2 通过第一层门控后，比较：

```text
P2-A3 vs P2-A2
```

若完整一致性提高分类性能或跨光照稳定性，则认为一致性机制有效。

若完整一致性无增益或损害原图性能，则后续 P2-B 只使用物理重光照，不加入一致性。

---

# 10. P2-B：接入 P1-RGBSR 主模型

P2-B 是条件式实验，不自动执行。

## 10.1 P2-B1：RGBSR + 物理重光照

触发条件：

```text
P2-A2 相对 ColorJitter
显示分类或稳定性价值
```

模型结构保持不变：

```text
3 × Independent ResNet18
→ 1536D concat
→ Linear(1536, 2)
```

原始输入：

```text
RGB_original
Shading_original
Residual_original
```

反事实输入：

```text
RGB_relight_k
Shading_relight_k
Residual_original
```

训练采样：

```text
50% 原始三分支
50% 反事实三分支
```

六套光照均匀采样。

不生成反事实 residual，始终使用：

```text
Residual_original
```

## 10.2 P2-B2：RGBSR + 物理重光照 + 完整一致性

仅在以下条件下执行：

```text
P2-A3 相对 P2-A2
显示明确增量
```

同时处理：

```text
x_original =
(
RGB_original,
Shading_original,
Residual_original
)

x_counterfactual =
(
RGB_relight_k,
Shading_relight_k,
Residual_original
)
```

得到：

```text
p_original
p_counterfactual
z_original
z_counterfactual
```

其中 `z` 为 1536D 融合特征。

损失：

```text
L =
L_cls
+
0.5 × L_pred
+
0.1 × L_feat
```

使用与 P2-A3 相同的 5 epoch warm-up。

第一版只约束最终融合特征，不分别为三个分支增加额外一致性损失。

## 10.3 P2-B 比较

根据 P2-A 结果，进行以下必要比较：

```text
P2-B1 vs P1-RGBSR
```

若执行 P2-B2，则继续比较：

```text
P2-B2 vs P1-RGBSR
P2-B2 vs P2-B1
```

核心分类和稳定性指标与 P2-A 保持一致。

---

# 11. 可选消融实验

以下实验不是当前 P2 必做任务：

```text
物理重光照 + 仅预测一致性
物理重光照 + 仅特征一致性
```

只有同时满足以下条件时才考虑补做：

1. P2-A3 明确优于 P2-A2；
2. 完整一致性成为论文的重要正结果；
3. 需要解释增益主要来自预测约束还是特征约束。

若完整一致性无效，不补做上述消融。

该原则为：

> 先验证主效应，再补机制消融；  
> 不在主效应未知时提前扩展实验空间。

---

# 12. P2 成功与失败标准

## 12.1 P2 成功

P2 可被认为具有研究价值，当满足以下任一情况：

- 物理重光照优于 ColorJitter；
- 物理重光照保持原图性能并明显提高跨光照稳定性；
- 完整一致性在物理重光照基础上进一步提高分类或稳定性；
- 接入 RGBSR 后，Macro-AUC、Macro-F1 或 Balanced Accuracy 提升；
- 接入 RGBSR 后，sensitivity/specificity 平衡改善；
- 接入 RGBSR 后，原图性能基本保持且 worst-light AUC 提高、标签翻转率降低。

## 12.2 P2 失败

出现以下情况时，应停止扩展 P2：

- 物理重光照不优于 ColorJitter；
- 物理重光照导致原图分类明显下降；
- prediction std 或 label flip rate 没有改善；
- worst-light AUC 下降；
- 完整一致性无增益且损害原图性能；
- 重光照产生明显伪影捷径；
- sensitivity 或 specificity 明显崩溃。

P2 失败时，不应为了获得正结果而：

- 搜索新的 SH 系数；
- 删除某些光照；
- 调整病例；
- 调整标签；
- 改变五折；
- 搜索一致性损失权重；
- 加入复杂网络模块；
- 提前引入 P3/P4 方法。

---

# 13. 训练规模与执行顺序

## 13.1 核心新增训练量

P2-A 核心新增实验：

```text
P2-A1：ColorJitter
P2-A2：物理重光照
P2-A3：物理重光照 + 完整一致性
```

若 P2-A0 复用已有 P1-RGB 结果：

```text
3个实验 × 5 folds = 15 folds
```

P2-B 根据结果门控：

```text
P2-B1：5 folds
P2-B2：可选5 folds
```

因此新增训练量：

```text
最低：15 folds
常规：20 folds
最大：25 folds
```

若为统一 runner 重跑 P2-A0，则增加 5 folds。

不再默认执行旧计划中的 40 folds。

## 13.2 推荐执行顺序

```text
P2-0
最小训练接口准备
        ↓
P2-A0
无增强基线（优先复用）
        ↓
P2-A1
ColorJitter
        ↓
P2-A2
物理重光照
        ↓
判断是否优于 ColorJitter
        ├─ 否
        │   └─ P2结束
        │
        └─ 是
            ↓
        P2-A3
        物理重光照 + 完整一致性
            ↓
        判断一致性是否有增量
            ↓
        P2-B1
        RGBSR + 物理重光照
            ↓
        必要时 P2-B2
        RGBSR + 物理重光照 + 完整一致性
            ↓
        P2完整报告
            ↓
        暂不自动进入 P3
```

---

# 14. 推荐实验目录

```text
E:\projects\face2\experiments\500Data\P2_Physics_Relighting_v2
```

建议保持简单：

```text
P2_Physics_Relighting_v2/
├── metadata/
├── rgb_benchmark/
│   ├── p2_a0_no_aug/
│   ├── p2_a1_colorjitter/
│   ├── p2_a2_relighting/
│   └── p2_a3_full_consistency/
├── rgbsr_integration/
│   ├── p2_b1_rgbsr_relighting/
│   └── p2_b2_rgbsr_full_consistency/
├── summary/
└── logs/
```

P2-B2 目录可预留，但只有结果门控通过后才运行。

---

# 15. 统一输出要求

## 15.1 P2-0 输出

至少包括：

```text
P2_Relighting_Protocol_v1.json
p2_training_manifest.csv
500例 relit_shading.npz
P2_COUNTERFACTUAL_TRAINING_ASSETS_FROZEN.json
p2_0_asset_preparation_report.md
```

## 15.2 每个正式训练实验输出

至少包括：

```text
5个fold checkpoint
5个fold验证预测
500例original-image OOF
6×500反事实推理结果
分类指标
核心稳定性指标
paired comparison
patient-cluster bootstrap
正式Markdown报告
```

每折至少保存：

```text
checkpoints/best_macro_auc.pth
training_history.csv
val_predictions_original.csv
val_predictions_relighted.csv
metrics_original.json
metrics_relighting_stability.json
fold_summary.json
_FOLD_SUCCESS.json
```

不要求为了形式完整而生成大量未被研究使用的重复文件。

---

# 16. P2 禁止事项

1. 修改固定五折；
2. 修改标签；
3. 删除病例；
4. 根据分类结果筛选光照；
5. 根据标签、fold 或 Camera/EXIF 选择光照；
6. 使用 Camera/EXIF 作为 P2 模型输入；
7. 重新运行或训练 DECA；
8. 使用分类标签更新物理前端；
9. 修改 `P1_DECA_Frozen500_v1`；
10. 复制并重新命名已有 RGB 重光照，制造重复数据版本；
11. 在主效应未知时提前运行全部一致性消融；
12. 自动进入 P3 或 P4；
13. 加入注意力、门控或复杂 MLP；
14. 同时改变网络结构和增强机制；
15. 搜索 ColorJitter 参数；
16. 搜索 SH 系数；
17. 搜索一致性损失权重；
18. 搜索分类阈值；
19. patient-level 预测平均；
20. 仅凭单一 AUC 判断 P2 成功；
21. 将物理中间表示解释为真实生理参数；
22. 将工程诊断项全部升级为前置正式阶段；
23. 将一个实验阶段继续无限细分为多个非必要子阶段。

---

# 17. 术语和科学解释边界

允许使用：

- `albedo-like representation`；
- `reflectance-like representation`；
- `shading-like component`；
- `illumination-related component`；
- `normal/coarse geometry cue`；
- `signed residual`；
- `physically constrained latent representation`；
- `physics-inspired counterfactual relighting`。

不得声称：

- `albedo_like` 是真实皮肤反射率；
- `shading_like` 是真实现场光照测量；
- `signed_residual` 是真实镜面反射图；
- 模型恢复了血红蛋白、黑色素、灌注或血氧参数；
- 重光照结果是真实世界同一患者在另一光源下的精确照片；
- 普通 RGB JPEG 可以支持真实生理参数反演。

反事实重光照的正确解释是：

> 在固定身份、面部几何和反射相关中间表示的条件下，仅改变预定义虚拟光照，构造标签保持的物理约束增强样本，用于分类增强和跨光照一致性学习。

---

# 18. 后续 Chat 的执行原则

后续 Chat 在推进 P2 时必须遵守：

1. 先读取本文件和总路线文档；
2. 优先继承 P0/P1 已有结果；
3. 每次只解决当前科学问题；
4. 不因潜在风险穷举新增审计；
5. 不将可选诊断自动升级为正式实验；
6. 不在主效应未知时扩展消融空间；
7. 结果不支持时允许停止；
8. 不自动进入下一阶段；
9. 研究严谨性以控制变量、公平比较和结果门控为主，而不是以任务数量和目录复杂度为主。

---

# 19. 当前下一步

当前应先实施：

```text
P2-0：最小训练接口准备
```

目标仅为：

```text
1. 确认现有重光照数组语义；
2. 复用原SH函数生成500×6 relighted shading；
3. 建立P2训练manifest；
4. 验证新增资产和DataLoader接口；
5. 输出P2_COUNTERFACTUAL_TRAINING_ASSETS_FROZEN；
6. 不执行分类训练。
```

P2-0 完成后，再生成 P2-A1、P2-A2 和 P2-A3 的统一实现方案。

---

# 20. 一句话总结

> P2 以最小必要数据适配为入口，先通过单 RGB 实验比较 ColorJitter、物理反事实重光照和完整一致性，只在主效应成立后将有效机制接入 P1-RGBSR 主模型；上游已冻结内容全部继承，消融和后续模型均由结果门控，避免重复审计、过度拆分和无效实验扩展。
