# 面向心功能状态分类的皮肤光学敏感表征研究路线与实现规范（R2）

**版本：v4.0**  
**日期：2026-08-04**  
**适用项目：面部照片心功能状态二分类（Control：NYHA 0；Patient：NYHA I–IV）**  
**文档定位：后续研究、代码实现、训练、评价和论文撰写的主约束文档**

---

## 1. 文档目的

本文件用于冻结当前已经确定的研究路线，避免后续工作因为参考文献、局部实验结果或实现便利而偏离核心问题。

本研究的目标不是从普通手机照片中测量真实黑色素浓度、血红蛋白浓度、灌注或氧合，而是：

> 在非受控手机人脸照片中，构建由皮肤光学前向模型约束、对采集条件相对稳定、并可能对心功能状态分类提供增量信息的 M/H 敏感表征。

当前采用的主路线为：

> **R2：合成监督预训练 + 规范重光照真实域联合适配 + 固定五折分类评价。**

该路线同时保留以下两点：

1. 使用合成数据的已知 M/H/S/P 控制场，固定分解网络四个输出通道的操作性语义；
2. 使用固定正面自然光的真实重光照图，通过冻结 SO-0 的重建闭环，使分解网络适应真实皮肤颜色、纹理和空间分布。

---

## 2. 两篇参考文献提供的方法论依据

### 2.1 Jung 等：真实受控图像的分解—物理重建闭环

参考文献：

> Jung G, Kim S, Lee J, Yoo S. Deep learning-based optical approach for skin analysis of melanin and hemoglobin distribution. *Journal of Biomedical Optics*. 2023;28(3):035001. DOI: 10.1117/1.JBO.28.3.035001.

该研究使用真实受控皮肤图像，U-Net 输出 melanin、hemoglobin、shading 和 specular 四个分量，再将四个输出输入固定皮肤光学前向模型重建 RGB，通过输入图与重建图之间的误差训练网络。

本项目借鉴的是：

- 输出通道通过固定物理前向模型获得操作性含义；
- 真实皮肤图像可通过“分解—前向重建—误差回传”参与训练；
- 皮肤分割和 patch 处理用于减少非皮肤区域干扰；
- 物理重建误差不能自动等价于真实生理量，仍需外部验证与解释边界。

本项目不照搬：

- VISIA 采集系统；
- 固定 D65 + Canon 5D Mark II 的真实采集条件；
- 4×5/4×6 固定网格；
- 仅依赖真实 RGB 重建从头训练四分量网络。

### 2.2 Aliaga 等：可信合成空间与监督逆映射

参考文献：

> Aliaga C, Xia M, Xie X, et al. A Hyperspectral Space of Skin Tones for Inverse Rendering of Biophysical Skin Properties. *Computer Graphics Forum*. 2023;42(4):e14887. DOI: 10.1111/cgf.14887.

该研究先定义生物物理皮肤参数空间，通过 Monte Carlo 光传输生成高光谱反射率，再以参数、光谱与 RGB 重建的多重监督训练编码器—解码器。其关键不是某一组具体参数，而是：

- 合成参数必须在前向模型中具有明确作用；
- 合成空间需要用真实皮肤光谱数据进行合理性验证；
- 合成真值可以固定逆网络的输出语义；
- 阴影、遮蔽等混杂因素必须单独处理，避免被误判为生物属性；
- 未经真实医学测量验证时，不能声称恢复了医学可用参数。

本项目借鉴的是“可信合成监督 + 真实域验证”的方法论，不直接复制其参数集合。

---

## 3. 本项目的科学定位

### 3.1 四个输出的固定定义

| 输出 | 本项目定义 | 主要角色 |
|---|---|---|
| M | melanin-pathway control field / melanin-sensitive representation | 内在皮肤光学敏感表征 |
| H | hemoglobin-pathway control field / hemoglobin-sensitive representation | 内在皮肤光学敏感表征 |
| S | shading-related nuisance field | 解释低频明暗与阴影 |
| P | specular-related nuisance field | 解释局部镜面高光 |

### 3.2 允许的论文表述

可以表述为：

- melanin-sensitive map；
- hemoglobin-sensitive map；
- shading-related component；
- specular-related component；
- SO-0-defined optical control fields；
- physics-constrained skin-optics-sensitive representations。

禁止表述为：

- 真实黑色素浓度图；
- 真实血红蛋白浓度图；
- 血流灌注图；
- 血氧饱和度图；
- 临床可测生理参数；
- 经医学设备校准的绝对值。

### 3.3 研究假设

1. M/H 主要承载与皮肤吸收相关、相对内在的颜色信息；
2. S/P 主要承载光照、阴影和镜面高光等采集因素；
3. 固定正面自然光重光照图能够降低真实输入的采集差异；
4. 原始 RGB 保留皱纹、纹理、眼周、唇部和其他可能与疾病相关的外观；
5. M/H 是否对心功能分类有用，必须通过固定五折增量实验验证，不能由物理模型先验直接推出。

---

## 4. 已完成并冻结的资产

### 4.1 SO-0 皮肤光学前向模型

状态：

- `SO0_Forward_Model_v1.1` 已冻结；
- SO-0 回归测试：29 passed；
- 光谱、颜色转换、相机—光源组合和数值精度审计已通过；
- 后续不得修改 SO-0 物理公式、颜色链、冻结矩阵和阈值。

SO-0 的核心功能：

\[
I_{\text{sRGB}}
=
F_{\text{SO0}}(M,H,S,P;c,l)
\]

其中：

- \(M\) 进入黑色素敏感吸收路径；
- \(H\) 进入血红蛋白敏感吸收路径；
- \(S\) 调节漫反射明暗；
- \(P\) 进入表面直接反射项；
- \(c\) 为相机光谱响应；
- \(l\) 为光源光谱功率分布。

### 4.2 SO-1 合成训练数据

正式数据目录：

```text
E:\projects\face2\data\processed\SO1_Synthetic_Generator_v1_localfull
```

WSL 对应路径：

```text
/mnt/e/projects/face2/data/processed/SO1_Synthetic_Generator_v1_localfull
```

冻结状态：

- 总样本数：27,000；
- Train：20,000；
- Validation：2,000；
- ID Test：2,000；
- Camera-OOD：1,000；
- Light-OOD：1,000；
- Joint-OOD：1,000；
- 27,000/27,000 `SUCCESS`；
- Train 10,000 个 base latent 的两个 acquisition variant 最终 camera-light pair 冲突数为 0；
- SO-1 单元测试：22 passed；
- SO-0 未修改；
- 正式 `COMPLETED.json` 已生成。

每个样本包含：

```text
linear_rgb   [3,256,256] float16
target_mhsp  [4,256,256] float16
valid_mask   [1,256,256] uint8
metadata / manifest
```

该数据集的定位是：

> SO-0 定义下的 M/H/S/P 控制场逆映射预训练数据，而不是真实人脸生理真值数据。

---

## 5. 当前正式研究路线总览

```text
SO-0 冻结前向模型
        ↓
27,000 例合成数据
        ↓
阶段一：合成监督预训练
        ↓
得到具有固定 M/H/S/P 通道语义的基础分解器
        ↓
固定正面自然光真实重光照图
        ↓
阶段二：每折合成监督 + 真实重建联合适配
        ↓
得到 5 个 fold-specific 分解模型
        ↓
测试病例规范重光照图推理
        ↓
生成 M/H/S/P 全脸图
        ↓
原始 RGB + M/H 表征
        ↓
固定 patient-group 五折心功能二分类
```

---

## 6. 模块 A：规范正面自然光真实数据构建

### 6.1 使用哪一套重光照图

六套 R3DPR 重光照图中，只选择已经确定的：

> **固定正面自然光、固定虚拟相机与固定渲染参数的规范重光照图。**

该图用于：

- 真实域联合适配；
- 分解网络正式推理；
- M/H 图生成。

其他五套重光照图第一版不参与训练，仅用于训练完成后的稳定性审计。

### 6.2 规范重光照图的角色

规范重光照图是：

> 无 M/H/S/P 标签的真实规范域 RGB 样本。

它不是：

- 合成标签；
- 数据增强标签；
- 原始手机图的完全无损替代；
- 真实受控设备重新采集图；
- 医学真值。

### 6.3 为什么不用规范图承担全部分类信息

R3DPR 可能导致：

- 皱纹和毛孔减弱；
- 局部红度变化；
- 细粒度纹理平滑；
- 高光与阴影形态变化。

因此：

- 分解器使用规范图；
- 最终分类的 RGB 主分支仍使用原始真实对齐人脸；
- M/H 作为补充物理敏感表征与 RGB 融合。

### 6.4 预处理流程

```text
规范重光照人脸
→ 高分辨率对齐
→ full-face skin mask
→ RGB 通道确认
→ sRGB [0,1]
→ inverse sRGB
→ linear RGB
→ 256×256 patch
```

要求：

- 输入统一为 RGB；
- 不使用 BGR；
- 不做每例白平衡、Gamma 增强或直方图均衡；
- 不使用 ColorJitter 改变颜色；
- 无 ICC 时记录 `assumed_sRGB`；
- 非皮肤区域设为 0；
- loss 仅在有效皮肤区域计算。

### 6.5 Patch 生成

训练：

- Patch 大小：256×256；
- 从全脸皮肤区域采样有效 patch；
- `valid_skin_fraction >= 0.20`；
- 以病例均衡方式采样，避免皮肤区域较大的病例占据过多 batch；
- 同一病例多个 patch 不能被视为独立病例。

推理：

- 使用自适应滑窗；
- Patch 大小：256；
- 重叠：15 像素；
- 覆盖人脸末端时追加最后一个起点；
- 通过平滑权重合并；
- 输出恢复到原高分辨率坐标。

---

## 7. 模块 B：SO-1 分解网络

### 7.1 网络输入输出

输入：

```text
linear RGB patch [3,256,256]
```

输出：

```text
M/H/S/P [4,256,256]
```

输出通道顺序固定：

```text
0: M
1: H
2: S
3: P
```

### 7.2 主网络

第一版固定使用标准容量 U-Net：

- Encoder：64 / 128 / 256 / 512；
- Bottleneck：1024；
- Decoder：512 / 256 / 128 / 64；
- 卷积块：Conv + Normalization + LeakyReLU；
- 下采样：MaxPool；
- 上采样：Transposed Convolution；
- 输出层：1×1 Conv，4 通道；
- 输出激活：Sigmoid；
- 随机初始化；
- 不使用 Carvana 预训练。

SO-0 在所有训练阶段完全冻结。

---

## 8. 阶段一：合成监督预训练

### 8.1 数据流

```text
synthetic linear RGB
→ U-Net
→ M̂/Ĥ/Ŝ/P̂
→ 与 target M/H/S/P 比较
```

### 8.2 损失函数

采用有效皮肤区域上的四分量等权 masked SmoothL1：

\[
L_{\text{syn}}
=
\frac{1}{4}
\left(
L_M+L_H+L_S+L_P
\right)
\]

其中每个分量均按 `valid_mask` 计算。

第一版不加入：

- 真实数据；
- RGB 重建损失；
- R3DPR 一致性损失；
- P 稀疏损失；
- S 低频损失；
- M/H 平滑损失。

### 8.3 训练建议

- Optimizer：AdamW；
- 初始学习率：1e-3；
- weight decay：1e-4；
- batch size：根据 8 GB 显存确定，优先 8；
- AMP：启用；
- 最大 epoch：25；
- early stopping patience：5；
- scheduler：cosine；
- seed：20260801；
- 第一版只训练一个固定种子。

### 8.4 Checkpoint 选择

主选择指标：

\[
\text{val\_M\_MAE}+\text{val\_H\_MAE}
\]

S/P 指标作为辅助审计，不作为主 checkpoint 选择依据。

### 8.5 合成域评价

必须分别报告：

- M/H/S/P MAE；
- RMSE；
- Pearson；
- Spearman；
- 输出标准差；
- 常数预测基线；
- ID Test；
- Camera-OOD；
- Light-OOD；
- Joint-OOD。

还需报告：

- M/H cross-talk；
- 通道交换；
- 输出坍塌；
- 预测饱和比例；
- mask 外输出。

---

## 9. 阶段二：每折合成监督—真实重建联合适配

### 9.1 核心定义

阶段二不是训练第二个分解器，而是在阶段一 checkpoint 基础上继续更新同一个 U-Net。

每个 optimizer step 同时读取：

1. 一个 synthetic batch；
2. 一个 canonical-real batch。

两类数据的作用不同：

| 数据 | 监督 | 作用 |
|---|---|---|
| 合成 batch | M/H/S/P 真值 | 防止通道语义漂移 |
| 规范真实 batch | SO-0 RGB 重建误差 | 适应真实皮肤与 R3DPR 规范图分布 |

### 9.2 合成分支

```text
synthetic RGB
→ U-Net
→ M̂/Ĥ/Ŝ/P̂
→ target M/H/S/P
→ L_syn
```

与阶段一相同。

### 9.3 真实规范图分支

```text
canonical real linear RGB
→ 同一个 U-Net
→ M̂/Ĥ/Ŝ/P̂
→ 冻结 SO-0
→ reconstructed canonical RGB
→ 与输入 canonical RGB 比较
```

公式：

\[
(\hat M,\hat H,\hat S,\hat P)
=
E_\theta(I_{\text{canon}})
\]

\[
\hat I_{\text{canon}}
=
F_{\text{SO0}}
(\hat M,\hat H,\hat S,\hat P;
c_{\text{canon}},l_{\text{canon}})
\]

### 9.4 Canonical camera/light

第一版不加入额外采集参数 \(q_{\text{acq}}\)。

固定：

- 一个 canonical SO-0 camera profile；
- 一个 canonical SO-0 light profile；
- exposure = 1。

该 camera/light 组合必须在正式训练前通过真实规范 patch 可拟合性检查。

### 9.5 真实重建损失

第一版只解释皮肤低频颜色与明暗，不强迫四个分量重建皱纹、毛孔和细小纹理。

使用：

\[
L_{\text{real}}
=
\operatorname{MaskedSmoothL1}
\left(
G_\sigma(\hat I_{\text{canon}}),
G_\sigma(I_{\text{canon}})
\right)
\]

其中：

- \(G_\sigma\) 为固定轻度 Gaussian low-pass；
- 建议 \(\sigma=2\) 像素；
- loss 在 linear RGB 和有效皮肤区域计算；
- 全分辨率重建误差仅作为诊断指标，不作为主训练损失。

### 9.6 总损失

第一版总损失只保留两项：

\[
L_{\text{total}}
=
L_{\text{syn}}
+
\lambda_{\text{real}}L_{\text{real}}
\]

不加入：

- \(q_{\text{acq}}\)；
- 六套重光照一致性；
- 交换重建；
- M/H 平滑；
- S 低频正则；
- P 稀疏正则；
- adversarial domain adaptation。

### 9.7 \(\lambda_{\text{real}}\) 的确定

只允许在 Fold 0 的训练/验证数据上进行一次小规模筛选：

```text
lambda_real ∈ {0.05, 0.10, 0.20}
```

选择规则：

1. 真实验证集低频重建误差降低；
2. 合成验证集 M/H MAE 相对阶段一 checkpoint 恶化不超过 10%；
3. M/H 不发生常数坍塌；
4. S/P 不承担几乎全部重建；
5. 选择后在五折中固定，不再逐折调参。

### 9.8 Batch 采样比例

不能把 27,000 例合成数据与真实病例直接拼接后普通随机采样。

固定采用：

```text
每个 synthetic batch 对应一个 canonical-real batch
```

即更新次数约为 1:1。

真实 patch 使用 case-balanced sampler：

- 先均匀采样病例；
- 再从该病例采样有效 patch；
- 防止某些病例因 patch 数较多而主导训练。

---

## 10. 正式五折与数据泄露控制

### 10.1 固定队列

- 总计 500 例完整数据；
- patient-group 五折；
- 同一患者不同 visit 必须位于同一折；
- visit/case 级评价；
- 不做 patient-level 平均；
- 不改变既有固定 folds；
- 不删除困难病例；
- 不根据测试结果重新划分。

### 10.2 合成预训练

阶段一合成预训练：

- 不使用任何患者数据；
- 所有 fold 共用同一个 synthetic-pretrained checkpoint；
- 不构成真实病例泄露。

### 10.3 Fold-specific 真实适配

每个 fold 独立执行：

```text
同一个 synthetic-pretrained checkpoint
        ↓
仅使用当前 fold 的训练病例进行真实联合适配
        ↓
当前 fold 验证病例用于 checkpoint 选择
        ↓
当前 fold 测试病例只做推理
```

禁止：

```text
先使用全部 500 例适配分解器
→ 再对同一 500 例做五折分类
```

最终应得到 5 个 fold-specific 分解模型。

---

## 11. 阶段二训练前的兼容性门控

R3DPR 的固定虚拟相机不等于 SO-0 中具有真实光谱响应曲线的相机。因此，在联合训练前必须完成可拟合性检查。

### 11.1 检查方法

只使用当前 fold 的训练病例，抽取一批规范重光照皮肤 patch。

固定 SO-0，不训练 U-Net，直接对每个 patch 优化 M/H/S/P，观察：

- 合法范围内能否重建低频 RGB；
- 是否大量输出卡在 0 或 1；
- clipping 是否异常；
- 是否只有 S/P 在承担重建；
- M/H 是否具有非零贡献。

### 11.2 通过条件

至少满足：

- 优化后的低频 RGB 误差显著低于常数颜色基线；
- 大多数 patch 的 M/H/S/P 不长期饱和；
- SO-0 输出 finite；
- clipping 处于可接受范围；
- 不存在普遍无法拟合的肤色类别。

未通过时：

- 不进入真实联合适配；
- 不通过增加复杂损失掩盖问题；
- 重新检查 canonical camera/light、颜色空间和 R3DPR 输出。

---

## 12. 阶段二训练监控

必须分别记录：

### 12.1 合成分支

- M/H/S/P MAE；
- M/H/S/P 输出范围；
- synthetic ID/OOD 指标；
- 相对阶段一 checkpoint 的 M/H 性能变化。

### 12.2 真实分支

- 低频 linear-RGB MAE；
- 全分辨率 RGB MAE，仅诊断；
- CIEDE2000，仅诊断；
- 重建 clipping；
- M/H/S/P 均值、标准差和边界饱和率。

### 12.3 分量贡献审计

对真实验证 patch 分别执行：

- M 替换为常数；
- H 替换为常数；
- S 替换为常数；
- P 置零。

比较重建误差变化。

解释：

- M/H 置常数几乎无影响：可能说明 S/P 吸收了颜色信息；
- S 置常数影响很小：可能说明规范图阴影较弱；
- P 置零影响很小：规范正面光下高光弱，不一定是失败；
- 任一分量全图恒定或长期饱和：训练失败。

第一版不通过增加结构损失强行修复，而是先报告和分析。

---

## 13. 模块 C：真实人脸推理与全脸图合并

### 13.1 推理输入

测试病例：

```text
原始照片
→ 冻结 R3DPR
→ 固定正面自然光规范图
→ linear RGB
→ patch divider
→ 当前 fold 专用分解模型
```

分解器不直接使用原始多相机、多光照照片。

### 13.2 输出

每例输出：

```text
M_full.npy
H_full.npy
S_full.npy
P_full.npy
skin_mask.npy
preview panel
patch manifest
```

要求：

- float32 保存完整分布图；
- PNG 仅用于预览；
- 合并后重新应用 full-face skin mask；
- 非皮肤区域为 0；
- 记录 coverage 和 seam 指标。

---

## 14. 其他五套重光照图的用途

其他五套重光照图第一版不参与训练。

训练完成后用于审计：

- 同一病例不同虚拟光照下 M/H 是否相对稳定；
- S/P 是否随光照发生预期变化；
- M/H 是否明显跟随阴影；
- 某些重光照是否丢失纹理或改变色度。

审计结果只用于：

- 判断分解器的采集稳定性；
- 标记失败模式；
- 决定未来是否有必要加入弱一致性约束。

第一版不因审计结果直接增加新的训练损失。

---

## 15. 模块 D：心功能状态分类

### 15.1 分类输入

原始 RGB：

- 使用现有最佳 224×224 对齐人脸预处理；
- 保留皱纹、纹理、眼周、唇部和其他视觉信息。

M/H：

- 来自规范正面自然光图；
- 使用当前 fold 专用分解器；
- resize 到 224×224；
- bilinear resize；
- mask 使用 nearest resize；
- resize 后重新 mask。

### 15.2 第一版分类比较

必须至少包含：

1. RGB baseline；
2. M-only；
3. H-only；
4. M+H；
5. RGB+M；
6. RGB+H；
7. RGB+M+H；
8. S+P negative-control；
9. mask-only probe。

主假设：

- RGB+M+H 优于 RGB；
- M/H 能提供独立增量；
- S+P 不应成为比 M/H 更强的疾病预测信号。

### 15.3 融合策略

优先使用对齐后的通道级早期融合：

```text
RGB + M + H = 5 channels
```

使用 ResNet18，只修改第一层输入通道。

前提：

- 原始 RGB 与规范图生成的 M/H 在同一对齐坐标系；
- 先完成像素坐标一致性审计。

若坐标一致性不通过，停止早期融合，改用特征级融合；不得在错位图上直接拼接。

---

## 16. 评价策略

### 16.1 分解网络：合成域

- M/H/S/P MAE；
- RMSE；
- Pearson；
- Spearman；
- constant baseline；
- ID Test；
- Camera-OOD；
- Light-OOD；
- Joint-OOD；
- cross-talk；
- 输出方差；
- 坍塌和饱和。

### 16.2 分解网络：真实规范域

- 低频 RGB reconstruction MAE；
- full-resolution RGB MAE；
- CIEDE2000；
- clipping；
- 分量贡献审计；
- map 统计；
- 多初始化一致性；
- 失败病例可视化。

### 16.3 六套重光照稳定性

- M/H 区域均值 CV；
- M/H map correlation；
- M/H pairwise MAE；
- S/P 变化幅度；
- Lab/红度/饱和漂移；
- 纹理丢失；
- 深浅肤色分层。

### 16.4 相机与采集泄漏

使用 M/H、S/P 和 RGB 分别预测：

- camera model；
- ISO；
- exposure time；
- brightness value；
- 其他可用 EXIF。

理想行为：

- M/H 的采集条件可预测性低于 RGB；
- S/P 可携带一定采集信息；
- 若 M/H 强烈预测 camera，应谨慎解释。

### 16.5 分类指标

固定报告：

- Macro-AUC；
- Accuracy；
- Macro-Precision；
- Macro-Recall；
- Macro-F1；
- Balanced Accuracy；
- Patient sensitivity；
- Control specificity。

正式五折：

- OOF 覆盖 500 例；
- patient-cluster bootstrap；
- paired confidence interval；
- 与 RGB baseline 配对比较；
- 不只报告单折结果。

---

## 17. GO / NO-GO 标准

### 17.1 阶段一合成预训练 GO

至少满足：

- M/H 优于常数基线；
- ID M/H 相关性较高；
- Camera-OOD 和 Light-OOD 不发生严重坍塌；
- 输出非恒定；
- 无通道交换；
- map 视觉合理。

### 17.2 阶段二真实联合适配 GO

必须满足：

- 可拟合性门控通过；
- 真实低频重建改善；
- synthetic validation M/H MAE 恶化不超过 10%；
- M/H 不坍塌；
- M/H 对真实重建具有非零贡献；
- S/P 未承担几乎全部颜色重建；
- fold validation 行为稳定。

### 17.3 进入分类 GO

必须满足：

- 40 例或 fold-train pilot 中 map finite；
- 无明显 seam；
- M/H 不全图常数；
- M/H 不主要跟随高光或阴影；
- 六套重光照下 M/H 相对稳定；
- camera/EXIF 泄漏可接受。

### 17.4 停止条件

出现以下任一情况，应停止当前路线并分析，不得通过继续堆损失掩盖：

- SO-0 无法拟合规范重光照真实 patch；
- synthetic M/H 恢复失败；
- 真实适配后 synthetic M/H 明显退化；
- M/H 常数坍塌；
- S/P 吸收几乎全部重建；
- M/H 比 S/P 更强地编码 camera；
- M/H 在六套重光照下高度不稳定；
- 分类提升只出现在单折；
- S+P 明显强于 M/H；
- 必须修改固定 folds 或删除困难病例才能获得结果。

---

## 18. 代码模块建议

```text
src/skin_optics_so1_r2/
├── configs.py
├── synthetic_dataset.py
├── canonical_real_dataset.py
├── case_balanced_sampler.py
├── patch_divider.py
├── patch_combiner.py
├── unet_decomposer.py
├── frozen_so0_renderer.py
├── losses.py
├── compatibility_audit.py
├── train_synthetic.py
├── train_joint_fold.py
├── infer_fullface.py
├── evaluate_synthetic.py
├── evaluate_real_reconstruction.py
├── evaluate_relight_stability.py
├── evaluate_component_contribution.py
└── export_maps.py
```

配置建议：

```text
configs/
├── so1_r2_synthetic_pretrain_v1.yaml
├── so1_r2_joint_adaptation_v1.yaml
├── so1_r2_compatibility_audit_v1.yaml
├── so1_r2_inference_v1.yaml
└── so3_rgb_mh_classification_v1.yaml
```

---

## 19. 推荐执行顺序

### R2-0：资产与接口审计

- 确认规范正面自然光图目录；
- 确认500例映射；
- 确认固定 folds；
- 确认高分辨率对齐和 full-face skin mask；
- 确认原始 RGB 与规范图坐标关系；
- 确认 canonical SO-0 camera/light。

### R2-1：SO-0—规范图兼容性审计

- fold-train patch 直接优化；
- 输出拟合误差、latent 分布、clipping；
- 未通过则停止。

### R2-2：阶段一合成监督预训练

- 训练一个全局 synthetic checkpoint；
- 完整 ID/OOD 评价；
- 冻结 checkpoint。

### R2-3：Fold 0 联合适配试运行

- 只使用 Fold 0 train；
- 选择并冻结 `lambda_real`；
- 完成真实重建和分量贡献审计；
- 不以 Fold 0 单折分类作为论文结论。

### R2-4：完整五折联合适配

- 每折从同一 synthetic checkpoint 开始；
- train 更新；
- validation 选择；
- test 只推理；
- 输出五套全脸 M/H/S/P。

### R2-5：六套重光照稳定性审计

- 不更新模型；
- 比较 M/H 稳定与 S/P 变化；
- 输出失败病例。

### R2-6：固定五折分类

- RGB baseline；
- M/H 单独；
- RGB+M/H；
- S+P 和 mask 负对照；
- OOF 和配对统计。

---

## 20. 冻结决策与禁止偏离项

后续研究默认不得更改：

1. 主路线为 R2；
2. SO-0 保持冻结；
3. M/H/S/P 的解释边界不升级为真实生理浓度；
4. 分解器训练和推理均使用固定正面自然光规范图；
5. 原始 RGB 保留给最终分类；
6. 阶段二仍保留 synthetic replay；
7. 第一版总损失只有 `L_syn + lambda_real * L_real`；
8. 不加入 `q_acq`；
9. 其他五套重光照仅用于审计；
10. 真实适配必须 fold-specific；
11. 固定 patient-group 五折；
12. 不删除困难病例；
13. 不改变500例评价口径；
14. 不根据测试结果修改模型或阈值；
15. 不因已有27,000例数据而忽视真实域失败证据。

任何路线变更必须同时满足：

- 有明确实验失败证据；
- 能说明当前冻结决策为什么不成立；
- 先更新本路线文档，再执行新代码。

---

## 21. 论文方法学核心叙述

本研究不是直接从手机RGB恢复真实皮肤生理参数，而是：

> 首先利用冻结皮肤光学前向模型生成具有已知M/H/S/P控制场的多采集条件合成数据，并通过直接监督预训练分解网络，以固定四个输出通道的操作性语义。随后，采用固定正面自然光与固定虚拟相机参数生成的规范重光照人脸作为无标签真实域数据，通过冻结前向模型的低频RGB重建闭环进行fold-specific联合适配，同时持续保留合成监督以抑制通道语义漂移。最终，从规范重光照图中提取M/H敏感分布，并与原始RGB视觉特征融合，用于Control与Patient的固定patient-group五折分类。

该叙述必须同时保留以下限制：

- 合成控制场不是真实医学参数真值；
- 规范重光照图不是物理设备重新采集；
- M/H只能解释为皮肤光学敏感表征；
- 最终价值由采集稳定性和分类增量实验决定。

---

## 22. 当前下一步

下一步不是直接训练，而是完成：

> **R2-0：真实规范重光照资产、坐标关系、full-face skin mask 和 canonical SO-0 camera/light 的完整接口审计。**

在这些输入路径和接口确认前，不生成正式训练代码。
