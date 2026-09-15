# SO-R0 v1.0：Acquisition-Robust M/H-Sensitive Representation Research Protocol

> **文档定位**  
> 本文档用于冻结“面部照片心功能状态分析中的采集条件鲁棒皮肤光学敏感表征”新主线研究协议。  
> 后续 SO-R1、SO-R2、SO-R3 的数据设计、模型实现、训练、评价与论文表述均应以本文档为最高约束。  
> 本文档冻结研究主旨、核心科学问题、阶段顺序、决策 Gate、允许/禁止事项；**暂不冻结**具体网络结构、样本量、camera/light 库、损失函数形式与权重等实现细节。

---

## 1. 研究背景与路线重构原因

前期 SO-0 → SO-1 系列研究尝试从普通 RGB 合成数据中恢复 M/H/S/P 四类分布图：

- M：melanin-sensitive representation
- H：hemoglobin-sensitive representation
- S：shading
- P：specular

前期实验完成了 SO-0 前向模型、SO-1 合成数据、SO-1 分解网络与多轮训练/数值稳定性审计，但后续 SO-1D-A 可学习性与可辨识性审计显示：

1. H 并非完全不可学习，但存在明显的采集条件敏感性、方差压缩与 empirical identifiability 警讯；
2. P target 本身高度稀疏，网络出现明显 near-zero trivial tendency，P 的 active-region recall 较低；
3. camera nuisance 对 H/P 的影响明显高于 light group 差异；
4. 原有“RGB → M/H/S/P + 多 camera/light 混合训练”的任务同时包含过多自由度，难以区分：
   - 皮肤内在光学信息；
   - camera/light 采集变化；
   - shading/specular 等外观因素；
   - 训练数值稳定性问题。

因此，旧 MHSP 主线停止继续扩展，研究重新聚焦到：

> **M/H-sensitive representation + acquisition robustness + cardiac downstream incremental value**

---

# 2. SO-R0 v1.0 总研究主旨

## 2.1 正式冻结的研究主旨

> **基于皮肤光学前向模型，构造同一皮肤光学状态在不同相机与光照采集条件下的合成 RGB 图像，学习对采集条件相对鲁棒的 melanin-sensitive / hemoglobin-sensitive 面部表征，并验证这种表征是否能够为真实手机面部照片的心功能状态分类提供普通 RGB 之外的增量信息。**

这是后续研究的最高主线。

任何后续实验都必须直接服务于以下两个核心科学问题之一，否则不得进入主线。

---

# 3. 两个核心科学问题

## 3.1 Q1：能否获得 acquisition-robust M/H-sensitive representation？

需要回答：

> 能否从普通 RGB 中获得既对 M/H 变化敏感，又对 camera/light 等采集条件相对稳定的 M/H-sensitive representation？

Q1 必须同时满足两个方向：

### A. Sensitivity

当底层 M/H 状态发生变化时：

```text
M/H 变化
    ↓
预测的 M/H-sensitive representation
应随之产生有意义变化
```

### B. Robustness / Invariance

当底层 M/H 状态保持不变，仅采集条件变化时：

```text
M/H 不变
camera/light 改变
    ↓
RGB 外观改变
    ↓
预测的 M/H-sensitive representation
应尽量保持稳定
```

### 不能接受的伪成功

#### 情况 1：常数输出

```text
所有输入
↓
输出几乎相同
```

这种表示虽然“稳定”，但失去 M/H sensitivity，因此 Q1 FAIL。

#### 情况 2：只会预测但不鲁棒

```text
M/H 可预测
但换 camera/light 后预测显著漂移
```

同样不能称 acquisition-robust，因此 Q1 FAIL。

---

## 3.2 Q2：M/H-sensitive representation 是否具有心功能增量价值？

只有 Q1 通过后，才允许进入 Q2。

Q2 需要回答：

> 稳定的 M/H-sensitive representation 是否能够为真实手机面部照片的心功能状态分类提供普通 RGB 之外的增量信息？

最终必须比较：

```text
A. RGB only

B. M/H-sensitive only

C. RGB + M/H-sensitive
```

真正的核心比较是：

```text
RGB + M/H-sensitive
        vs
RGB only
```

不是要求：

```text
M/H-sensitive only > RGB only
```

M/H-only 低于 RGB-only 并不代表研究失败，因为 RGB 还包含纹理、形态、颜色、几何等大量信息。

真正的研究假设是：

> M/H-sensitive representation 是否包含普通 RGB appearance 之外的补充信息。

---

# 4. M/H 的科学身份

后续统一使用：

- `M-sensitive representation`
- `H-sensitive representation`
- `melanin-sensitive map`
- `hemoglobin-sensitive map`

## 4.1 禁止过度物理解读

不得直接宣称：

- 真实黑色素浓度；
- 真实血红蛋白浓度；
- 血氧；
- 灌注；
- 血流；
- SpO₂；
- 真实生理浓度定量。

原因：

真实手机 JPEG 还包含：

- 未知 illumination；
- 未知 camera spectral response；
- automatic white balance；
- exposure；
- HDR；
- tone mapping；
- local contrast；
- ISP；
- sharpening；
- denoising；
- compression。

因此，新路线的目标不是“从普通 JPEG 精确测量生理参数”，而是：

> 学习具有皮肤光学依据、对 M/H 变化敏感、并尽量降低采集条件影响的中间表示。

---

# 5. M/H/S/P 旧路线的重新定位

## 5.1 核心 intrinsic representation

正式冻结为：

```text
M
H
```

即：

- melanin-sensitive representation
- hemoglobin-sensitive representation

## 5.2 不再作为核心 inverse target 的因素

```text
S = shading
P = specular
```

不再与 M/H 并列作为核心 inverse targets。

统一重新定位为：

```text
acquisition / appearance nuisances
```

包括但不限于：

- camera；
- illumination；
- exposure；
- shading；
- specular；
- white balance；
- color processing。

## 5.3 为什么 P 退出主线

前期 SO-1D-A 已经显示：

- P target 自身高度稀疏；
- 网络虽然整体 MAE 较低，但高度依赖 near-zero prediction；
- active-region recall 明显不足；
- 与 zero baseline 相比的收益有限。

因此：

> 不再为了保持与参考文献四输出形式一致而强制保留 P。

P 不是新主线必须恢复的核心皮肤光学状态。

---

# 6. Camera / Light 的新研究定位

## 6.1 不采用的假设

新路线不采用：

```text
所有真实照片
≈
D65 + 固定 camera
```

真实手机照片来自：

- 多种手机；
- 不同自然环境光；
- 不同曝光条件；
- 不同 ISP。

因此固定 D65 / 固定 camera 不能作为最终真实部署假设。

## 6.2 新路线的核心原则

在合成世界中显式构造：

```text
同一个底层 M/H latent
        │
        ├── Camera A + Light A → RGB-A
        ├── Camera B + Light B → RGB-B
        ├── Camera C + Light C → RGB-C
        └── Camera D + Light D → RGB-D
```

其中：

```text
底层 M/H 完全相同
采集条件不同
RGB 外观不同
```

模型需要学习：

```text
f(RGB-A)
≈
f(RGB-B)
≈
f(RGB-C)
≈
f(RGB-D)
```

同时仍保持对真实 M/H variation 的敏感性。

## 6.3 D65 / canonical condition 的角色

D65 或 canonical RGB 可以作为：

```text
reference acquisition condition
```

用于：

- 前向模型验证；
- canonical reference；
- controlled experiment；
- calibration anchor。

但不得把 D65 当作所有真实手机图像的真实光照假设。

---

# 7. 新合成数据政策

## 7.1 必须重新设计

新主线正式冻结：

> **新的正式研究数据必须根据 SO-R0 新研究主旨重新设计并重新生成。**

不允许：

> 为了继续使用旧 27k 数据而限制新研究设计。

研究路线决定数据，而不是旧数据决定研究路线。

## 7.2 旧 27k 数据的角色

原：

```text
SO1_Synthetic_Generator_v1_localfull
```

不再作为新主线的正式训练集。

保留为：

```text
Historical / exploratory asset
```

可用于：

- 历史复现；
- failure analysis；
- stress-test；
- camera/light nuisance 对照；
- 方法演化说明。

但新 SO-R1 不得被旧数据结构绑死。

---

# 8. 新合成数据的核心组织单位：latent group

新数据不能只把每张 RGB 当成独立样本。

核心单位必须是：

```text
latent_id
```

一个 `latent_id` 表示一个固定的 underlying skin-optical state。

例如：

```text
Latent Z001
M/H = fixed
│
├ acquisition 01
├ acquisition 02
├ acquisition 03
└ acquisition 04
```

同一 latent 的不同 acquisition：

- camera 可以不同；
- light 可以不同；
- exposure/shading 可以不同；
- 但底层 M/H target 必须一致。

## 8.1 Split 规则

Train / Validation / Test 必须按照：

```text
latent_id
```

分组划分。

禁止：

```text
RGB_A_camera1 → Train
RGB_A_camera2 → Validation
```

否则会发生 synthetic leakage。

## 8.2 Metadata 原则

新数据必须能够明确追踪：

- `latent_id`
- `acquisition_id`
- `camera_id`
- `light_id`
- 其他后续冻结的 nuisance identifier

具体字段定义在 SO-R1 冻结。

---

# 9. 新主线阶段结构

正式冻结为：

```text
SO-R0
研究协议冻结
    ↓
SO-R1
Synthetic Acquisition-Robust M/H Learning
    ↓
SO-R2
Real Smartphone Stability Audit
    ↓
SO-R3
Cardiac Incremental-Value Experiment
```

不再沿用旧：

```text
SO-1D-R4
SO-1D-R5
旧 SO-1E
```

作为新主线。

旧 SO 系列保留为历史探索。

---

# 10. SO-R1：Synthetic Acquisition-Robust M/H Learning

## 10.1 目标

回答 Q1 的 synthetic 部分：

> 在具有明确 M/H ground truth 的合成数据中，能否学习既保持 M/H sensitivity、又降低 camera/light nuisance 的表示？

## 10.2 必须存在的对照组

SO-R1 至少必须包含两个方法。

### Baseline

普通 supervised predictor：

```text
Synthetic RGB
    ↓
M/H predictor
    ↓
M/H-sensitive maps
```

不使用 same-latent acquisition invariance。

### Proposed

使用：

```text
same latent
+
different acquisition
```

的监督关系。

要求：

```text
不同 acquisition
但相同 latent
    ↓
预测表示保持一致
```

这样才能判断：

> acquisition-aware / paired acquisition 设计是否真正有价值。

否则无法区分：

- 新思想有效；
- 数据量增加；
- 网络偶然表现更好。

---

# 11. SO-R1 的三类核心证据

SO-R1 不能只评价 MAE。

必须同时评估以下三类证据。

## 11.1 A. M/H sensitivity

至少包括：

- MAE；
- Pearson；
- Spearman；
- prediction variance；
- target variance；
- variance preservation；
- baseline improvement。

重点关注 H。

## 11.2 B. Acquisition invariance

对同一个 latent 的不同 acquisition，计算：

- within-latent prediction drift；
- pairwise prediction difference；
- within-latent variance；
- acquisition consistency。

Proposed 必须明显优于普通 supervised baseline。

## 11.3 C. Acquisition generalization

必须保留：

```text
unseen camera
unseen light
joint unseen camera + light
```

条件。

否则只能证明：

> 模型记住训练过的 camera/light。

不能证明 acquisition robustness。

---

# 12. SO-R1 成功原则

暂时不冻结人为绝对阈值，例如：

```text
H Pearson > 0.90
H MAE < 0.05
```

采用比较式 Gate。

SO-R1 至少需要满足：

1. M/H prediction 明显优于无信息 baseline；
2. Proposed 的 M/H sensitivity 不弱于普通 supervised baseline；
3. Proposed 的 same-latent acquisition drift 明显低于 baseline；
4. Proposed 在 unseen camera/light 条件下鲁棒性更好；
5. 无 representation collapse；
6. H 不得为了稳定性退化为常数或低方差输出。

关键比较建议采用：

```text
latent-level / case-level bootstrap 95% CI
```

禁止把大量像素当独立样本制造虚假显著性。

---

# 13. SO-R2：Real Smartphone Stability Audit

## 13.1 目标

回答 Q1 的 real-image 部分：

> Synthetic 中学到的 acquisition-robust M/H-sensitive representation 是否能够迁移到真实手机 JPEG，并保持相对稳定？

## 13.2 真实数据没有 M/H ground truth

因此 SO-R2 不评价：

- 真实浓度准确性；
- 真实 H MAE；
- 真实 M/H 生理定量。

SO-R2 只评价：

```text
real-image representation stability
```

## 13.3 稳定性扰动

对同一真实手机照片进行受控、合理的：

- exposure perturbation；
- white balance perturbation；
- color-temperature perturbation；
- gamma/intensity perturbation；
- 其他后续冻结的 photometric perturbation。

理想结果：

```text
RGB appearance 明显变化
        ↓
M/H-sensitive representation
变化显著更小
```

---

# 14. SO-R2 必须防止“常数表示假稳定”

如果模型：

```text
所有输入
↓
几乎输出同一 representation
```

那么稳定性会很高，但没有任何信息。

因此 SO-R2 必须同时检查：

### Within-subject / within-image stability

```text
同一图像受扰动
→ representation drift 小
```

### Between-subject variability

```text
不同真实人脸
→ representation 仍保留足够差异
```

只有：

```text
同一人稳定
+
不同人可区分
```

才算合理的 acquisition-robust representation。

---

# 15. Q1 最终 Gate

Q1 必须由：

```text
SO-R1 Synthetic robustness
PASS
+
SO-R2 Real smartphone stability
PASS
```

共同成立。

只有双 PASS，才允许使用：

> acquisition-robust M/H-sensitive representation

作为正式研究结论。

## 15.1 如果 Synthetic PASS / Real FAIL

正式结论必须是：

> Synthetic robustness did not transfer to real smartphone imagery.

此时：

- 不得强化物理解读；
- 不得直接进入 SO-R3 依靠分类结果“救路线”；
- 主物理表征路线应暂停或重新设计。

---

# 16. SO-R3：Cardiac Incremental-Value Experiment

## 16.1 启动条件

只有：

```text
Q1 = PASS
```

才允许进入 SO-R3。

## 16.2 固定三组实验

必须包含：

```text
A. RGB only

B. M/H-sensitive only

C. RGB + M/H-sensitive
```

## 16.3 数据划分原则

使用已有真实数据固定五折，并保持：

```text
same patient-group
→ same fold
```

禁止：

- 为 M/H 方法重新寻找更有利 fold；
- 重新划分后只报告最优结果；
- 使用 test fold 调参。

---

# 17. SO-R3 核心评价问题

真正的 primary comparison 是：

```text
RGB + M/H-sensitive
        vs
RGB only
```

允许：

```text
M/H only < RGB only
```

这不构成失败。

研究的真正问题是：

> M/H-sensitive representation 是否提供 RGB branch 之外的增量信息？

---

# 18. SO-R3 主评价体系

主评价保持：

- Macro-AUC；
- Balanced Accuracy；
- Macro-F1；
- Accuracy。

建议：

```text
OOF Macro-AUC
```

作为 primary discrimination endpoint。

Balanced Accuracy、Macro-F1 作为关键 supporting endpoints。

Accuracy 作为辅助指标。

最终必须在同一 OOF case 上进行 paired comparison。

---

# 19. Q2 判定分级

## 19.1 Strong Positive

如果：

```text
RGB + M/H > RGB
```

且：

- paired uncertainty 支持稳定正向改善；
- Balanced Accuracy / Macro-F1 无明显恶化；
- 多 fold 表现方向总体一致；

则支持：

> incremental value。

## 19.2 Weak Signal

如果：

- 点估计有所提升；
- 但 CI 跨 0；
- 或仅部分 fold 改善；

只能称：

```text
exploratory / promising signal
```

不能作为强结论。

## 19.3 Negative

如果：

```text
RGB + M/H ≈ RGB
```

或：

```text
RGB + M/H < RGB
```

则：

```text
Q2 = FAIL
```

即使 Q1 完全成功，也必须承认：

> 该稳定皮肤光学敏感表示未对当前心功能分类提供可验证的增量价值。

---

# 20. 两个核心问题的最终决策矩阵

| Q1：光学鲁棒性 | Q2：心功能增量价值 | 最终判断 |
|---|---|---|
| FAIL | 不执行 | 主路线停止 |
| PASS | FAIL | 光学方法成立，但不支持当前心功能任务 |
| PASS | Weak | 存在探索性信号，可决定是否继续 |
| PASS | PASS | 完整论文主线成立 |

这也是防止研究无限迭代的重要终止原则。

---

# 21. 最大风险：Synthetic-to-Real Semantic Gap

整个研究最大的逻辑风险正式冻结为：

> **Synthetic 中定义并学习到的 M/H-sensitive representation，到了真实手机 JPEG 上未必仍然保持相同的皮肤光学含义。**

原因：

真实手机图像形成远比 synthetic forward model 复杂。

因此：

```text
SO-R2 Real Smartphone Stability Audit
```

不是普通 ablation，而是：

> **主研究 Gate。**

Synthetic performance 再好，也不能跳过 SO-R2。

---

# 22. 第二大风险：静态 M/H-sensitive 信息未必与心功能有关

即使 Q1 PASS：

```text
M/H-sensitive representation
稳定且具有光学敏感性
```

仍然完全可能：

```text
RGB + M/H ≈ RGB
```

因为：

> “皮肤存在 hemoglobin-sensitive information”  
> 不等于  
> “单张静态面部照片中的 H-sensitive information 足以反映心功能状态”。

因此：

SO-R3 必须按 hypothesis testing 设计。

禁止预设：

```text
M/H 一定提升心功能分类
```

---

# 23. 第三类风险：真实标签混杂与 shortcut

即使方法被称为 physics-informed，也不能自动排除真实数据混杂。

需要持续警惕：

- camera；
- batch；
- illumination；
- age；
- sex；
- skin tone；
- acquisition environment；
- source differences。

因此：

> acquisition robustness 是主方法合理性的核心证据，不是附加实验。

---

# 24. 旧 SO 系列的正式处置

以下全部保留：

```text
SO-0
SO-1A
SO-1B
SO-1C
SO-1D
SO-1D-R1
SO-1D-R2
SO-1D-R3
SO-1D-A
```

用途：

- 前期探索；
- failure analysis；
- research design evidence；
- 方法演化说明；
- 历史复现。

但：

```text
不得继续作为新主线正式训练协议。
```

SO-1D v1.1 epoch18 checkpoint：

```text
DIAGNOSTIC_ONLY
```

不得升级为 SO-R 新路线正式模型。

---

# 25. SO-R0 v1.0 当前冻结项

以下内容从本版本开始冻结：

1. 总研究主旨；
2. 两个核心科学问题 Q1 / Q2；
3. M/H-sensitive 而非真实生理浓度；
4. M/H 作为核心 intrinsic representation；
5. P 退出核心 inverse target；
6. shading/specular/camera/light 重新定位为 nuisance；
7. 新正式合成数据必须重新设计；
8. 旧 27k 不得约束新主线；
9. same-latent / multi-acquisition 数据组织原则；
10. latent-group split 原则；
11. SO-R1 → SO-R2 → SO-R3 阶段顺序；
12. Q1 必须 Synthetic + Real 双 PASS；
13. Q2 必须 RGB / M-H / RGB+M-H 三组对照；
14. SO-R2 是主研究 Gate；
15. Q1 FAIL 后不得进入 SO-R3；
16. Q2 允许失败，不得后验修改研究假设掩盖负结果；
17. 整条路线以“光学敏感性 + 采集稳定性 + 下游增量价值”为三层证据链。

---

# 26. SO-R0 v1.0 暂不冻结项

以下内容故意留到 SO-R1 设计阶段：

- 具体皮肤光学公式；
- M/H 参数范围；
- camera spectral response 库；
- light SPD 库；
- 是否使用 D65 作为 canonical reference anchor；
- canonical observation model；
- synthetic 总样本量；
- latent 数量；
- 每个 latent 的 acquisition 数量；
- camera/light 组合数量；
- shading / exposure 的具体建模方式；
- specular 是否以及何时加入；
- 网络结构；
- loss 形式；
- consistency loss 形式；
- consistency 权重；
- batch size；
- optimizer；
- learning rate；
- epoch；
- early stopping；
- SO-R1 Pilot 成功数值阈值；
- SO-R2 photometric perturbation 范围。

这些内容必须在后续阶段单独冻结后才能实现。

---

# 27. 新研究路线的最终流程图

```text
                    SO-R0
           Research Protocol Freeze
                     │
                     ▼
                    SO-R1
       Synthetic Acquisition-Robust
              M/H Learning
                     │
                     ▼
          Synthetic Robustness Gate
                     │
               PASS / FAIL
                │       │
              FAIL      PASS
                │       │
               STOP     ▼
                    SO-R2
          Real Smartphone Stability
                     │
                     ▼
              Real Stability Gate
                     │
               PASS / FAIL
                │       │
              FAIL      PASS
                │       │
               STOP     ▼
                     Q1 PASS
                     │
                     ▼
                    SO-R3
        Cardiac Incremental-Value Test
                     │
        ┌────────────┼────────────┐
        │            │            │
      RGB only     M/H only    RGB + M/H
        │            │            │
        └────────────┼────────────┘
                     │
                     ▼
             Incremental Value
                     │
             Strong / Weak / No
```

---

# 28. 整条研究路线的成功定义

新路线成功不再定义为：

> “网络成功生成了 M/H map。”

真正成功必须形成三层证据链：

## Layer 1：Optical Sensitivity

```text
M/H 变化
→ representation 正确变化
```

## Layer 2：Acquisition Robustness

```text
M/H 不变
camera/light 变化
→ representation 相对稳定
```

并且这种稳定性能够从 synthetic 迁移到真实手机 JPEG。

## Layer 3：Clinical Incremental Value

```text
RGB + M/H-sensitive
>
RGB only
```

说明该 physics-informed representation 对心功能状态分析提供了额外信息。

只有三层证据共同成立，才形成完整的新论文主线。

---

# 29. 当前阶段正式状态

```text
SO-R0 v1.0
Status: FROZEN
```

当前允许的下一阶段：

```text
SO-R1 Experimental Design
```

当前禁止：

- 直接开始大规模合成数据生成；
- 直接训练新模型；
- 继续旧 MHSP U-Net；
- 继续修旧 AMP；
- 旧模型 Full FP32 正式训练；
- 直接进入 SO-R2；
- 直接进入 SO-R3；
- 直接开展分类；
- 直接宣称真实 M/H 生理参数恢复。

---

# 30. SO-R1 下一阶段唯一目标

SO-R1 设计阶段首先只回答：

> **如何构造 scientifically valid 的 same-latent / multi-acquisition synthetic dataset，使模型能够被明确监督为：既对 M/H 敏感，又对 camera/light acquisition 变化相对稳定？**

在回答并冻结这个问题之前：

> **不得进入编码和大规模数据生成。**

---

## SO-R0 v1.0 一句话总结

> **本研究不再追求从普通手机 RGB 中直接恢复 M/H/S/P 四类物理参数，而是利用皮肤光学前向模型构造同一 M/H 状态在不同采集条件下的配对合成数据，学习对 camera/light nuisance 相对鲁棒的 M/H-sensitive representation，并通过真实手机稳定性审计与 RGB / M-H / RGB+M-H 五折分类实验，验证这种表征是否具有真实的光学稳定性与心功能增量价值。**
