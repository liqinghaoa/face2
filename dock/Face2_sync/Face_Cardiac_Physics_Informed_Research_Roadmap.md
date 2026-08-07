# 面向心功能异常识别的采集不变反射表征与物理反事实重光照研究路线指导文档

> 项目名称：  
> **Acquisition-Invariant Reflectance Representation Learning with Physics-Inspired Counterfactual Relighting for Facial Cardiac Status Discrimination**
>
> 中文定位：  
> **面向心功能异常识别的采集不变反射表征与物理反事实重光照研究**
>
> 任务：  
> 基于普通 RGB 正脸照片，实现 Control（无病对照）与 NYHA I–IV Patient（患者）的二分类探索性识别。
>
> 当前状态：  
> 已完成 E0B Global ResNet18 二分类基线，Pooled OOF AUC ≈ 0.855。

---

## 1. 研究定位（必须保持）

### 1.1 核心研究问题

普通 RGB 人脸照片可以概括为以下多因素共同作用的观测：

\[
I=f(\text{identity},\text{appearance},\text{illumination},\text{camera},\text{processing})
\]

其中，最终 RGB 图像同时受到以下因素影响：

- 疾病相关面部表现；
- 皮肤颜色与纹理；
- 光照方向和强度；
- 相机光谱响应；
- 曝光、ISO 和白平衡；
- 镜面高光与几何阴影；
- JPEG 压缩和相机内部处理。

当前 Global CNN 模型能够获得较高的二分类性能，但尚不能确认模型主要学习的是：

- 真正的疾病相关面部表征；
- 还是相机、曝光、照明、背景或拍摄流程等采集捷径。

因此，本研究不以简单提高 AUC 为唯一目标，而重点研究：

> 如何利用计算成像中的物理先验，将疾病相关信息与采集条件因素进行分离，学习更加稳定、可解释、采集不敏感的面部疾病表征。

---

## 2. 最终论文主线

### 2.1 核心思想

本研究的正式论文主线为：

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

总体数据流为：

```text
RGB Face Image
        ↓
Frozen Face Physical Decomposition
        ↓
Albedo-like / Normal / Illumination /
Specular-like / Residual
        ↓
RGB Disease Representation
+
Reflectance-like Representation
+
Acquisition Representation
        ↓
Counterfactual Relighting Consistency
        ↓
Disease–Acquisition Disentanglement
        ↓
Reliability-aware Fusion
        ↓
Binary Cardiac Status Classification
```

### 2.2 论文的核心方法创新

论文的创新重点不是重新设计一个普通分类网络，而是：

1. 利用冻结的人脸物理分解先验获得物理约束中间表示；
2. 使用物理反事实重光照构造标签保持的光照变化样本；
3. 显式建模相机、曝光、光照和高光等采集条件；
4. 减少疾病表示中的采集条件信息泄漏；
5. 根据物理分解可靠性自适应融合 RGB 表征和 reflectance-like 表征。

---

## 3. 重要概念边界

### 3.1 不允许声称

当前数据为普通 RGB JPEG 照片，缺少标准光源、色卡、相机光谱响应和真实生理参数监督，因此不能声称模型完成了：

- 真实皮肤反射率恢复；
- 血红蛋白浓度估计；
- 血氧饱和度预测；
- 皮肤灌注参数估计；
- 黑色素定量估计；
- 血流或脉搏信号提取；
- 真实生理参数反演。

### 3.2 推荐术语

允许使用：

- `albedo-like representation`；
- `reflectance-like representation`；
- `shading-like component`；
- `illumination-related component`；
- `specular-like component`；
- `physically constrained latent representation`；
- `acquisition-aware representation`；
- `chromophore-sensitive descriptor`。

### 3.3 对物理分解结果的正确解释

物理分解模块输出的是受公开人脸逆渲染先验约束的中间表示，而不是唯一、定量、真实的生理成像结果。

低重建误差只能说明输出可以近似合成输入图像，不能单独证明：

- albedo-like 表示是真实反射率；
- illumination 参数是真实现场光源；
- specular-like 分量是真实镜面反射；
- residual 具有确定的物理语义。

---

## 4. 当前二分类基线

### 4.1 E0B Global ResNet18

输入：

```text
224 × 224 aligned RGB face
```

模型：

```text
ImageNet-pretrained ResNet18
+
Binary classifier
```

数据：

- 500 张图像；
- 483 个患者组；
- Control：115；
- Patient：385；
- 患者组级固定五折交叉验证。

当前结果：

| Metric | Value |
|---|---:|
| 五折平均 AUC | 约 0.8665 |
| Pooled OOF AUC | 约 0.8550 |
| Accuracy | 约 0.8060 |
| Balanced Accuracy | 约 0.7643 |
| Macro-F1 | 约 0.7447 |
| Patient sensitivity | 约 0.8416 |
| Control specificity | 约 0.6870 |

E0B 是所有后续实验必须公平比较的参考基线。

### 4.2 E0B 的研究意义

当前结果支持：

- 普通 RGB 全脸照片中存在区分无病对照与 NYHA I–IV 患者的内部判别信息；
- 五折排序能力较稳定；
- 二分类信号明显强于既往三分类严重程度信号。

当前结果尚不支持：

- 模型可以直接诊断心力衰竭；
- 模型已经具备临床部署能力；
- 模型学习到的全部是疾病生理表型；
- AUC 0.855 不包含相机、曝光、来源和拍摄流程混杂。

---

## 5. 总体模型输入

每个病例建议准备以下数据。

### 5.1 原始完整照片

记为：

\[
I_{\text{scene}}
\]

主要用于：

- 场景光照估计；
- 物理分解；
- 保留背景和环境中的照明线索。

### 5.2 对齐人脸图像

记为：

\[
I_{\text{face}}
\]

主要用于：

- RGB 主分支；
- 二分类；
- 与既有 E0B 基线保持一致。

### 5.3 辅助信息

包括：

- face mask；
- skin mask；
- camera 型号；
- ExposureTime；
- FNumber；
- ISO；
- BrightnessValue；
- 其他可可靠提取的 EXIF；
- 图像质量指标。

### 5.4 输入使用原则

- 物理分解优先使用原始完整照片及对齐人脸；
- RGB 分类继续使用既有对齐全脸图；
- 不应仅使用 mean-background 图像估计场景照明；
- 所有标准化、缺失填补和条件建模只能在当前训练折拟合。

---

## 6. 冻结物理分解模块

### 6.1 基本原则

不能在当前 500 例医学数据上端到端训练完整逆渲染网络。

主要原因：

- 样本规模不足；
- 单图逆渲染本身具有多解性；
- 易产生不可解释的伪物理分解；
- 二分类标签可能污染物理中间表示；
- 增加严重过拟合风险。

因此，正式方案应采用：

> 公开预训练的人脸逆渲染或内在图像分解模型作为冻结物理前端。

### 6.2 物理分解形式

\[
\mathcal P(I_{\text{scene}},I_{\text{face}})
\rightarrow
\left\{
\hat A,\hat N,\hat L,\hat S_{\text{spec}},\hat R
\right\}
\]

其中：

#### Albedo-like

\[
\hat A
\]

表示反射相关的物理约束中间表示。

#### Normal

\[
\hat N
\]

表示面部几何或法线线索。

#### Illumination

\[
\hat L
\]

表示估计的低维照明参数。

#### Specular-like

\[
\hat S_{\text{spec}}
\]

表示镜面高光相关成分。

#### Residual

\[
\hat R
\]

表示简单物理成像模型未能解释的剩余成分。

### 6.3 冻结策略

第一阶段应：

- 冻结全部物理分解器参数；
- 离线生成物理图和物理参数；
- 不允许 NYHA 二分类标签反向更新物理分解器；
- 只训练后续轻量分类和表示学习模块。

后续是否允许极少量适配层微调，必须在 P0 和 P1 结果支持后另行决定。

---

## 7. 反事实物理重光照

### 7.1 目标

构造同一患者在不同虚拟光照条件下的反事实图像，并保持患者身份和标签不变：

\[
y(I)=y(\hat I_k)
\]

反事实重光照的目标不是生成影视级真实图像，而是提供受物理模型约束的标签保持增强和一致性学习样本。

### 7.2 重光照形式

\[
\hat I_k=
\hat A
\odot
SH(\hat N,L_k)
+
\hat R_k
\]

其中：

- \(\hat A\)：albedo-like 表示；
- \(\hat N\)：法线或几何线索；
- \(L_k\)：第 \(k\) 个预定义虚拟光照；
- \(SH\)：球谐光照模型；
- \(\hat R_k\)：可选的非 Lambertian 残差修正。

### 7.3 固定光照集合

第一版建议固定少量、预定义的光照条件：

- frontal light；
- left light；
- right light；
- top light；
- weak uniform light；
- moderately strong uniform light。

禁止根据测试集结果动态筛选光照类型。

### 7.4 与普通 ColorJitter 的区别

普通 ColorJitter 直接改变 RGB 数值，不显式考虑：

- 面部几何；
- 光照方向；
- 漫反射；
- 阴影；
- 高光。

物理反事实重光照通过固定反射相关表示和几何，只改变虚拟光照，因此具有更明确的图像形成约束。

正式实验必须比较：

- 无增强；
- ColorJitter；
- 物理重光照；
- 物理重光照加一致性约束。

如果物理重光照不能优于 ColorJitter 或提供额外稳定性，则其必要性不足。

---

## 8. 网络结构

### 8.1 RGB 主分支

输入：

\[
I_{\text{face}}
\]

编码器：

- 继续使用 ResNet18；
- 保留与 E0B 相同的基础训练设置。

输出：

\[
z_{\text{rgb}}
\]

作用：

- 保留全脸形态；
- 保留纹理和结构；
- 保留当前强二分类基线能力。

### 8.2 Reflectance-like 分支

输入：

\[
\hat A\odot M_{\text{skin}}
\]

其中 \(M_{\text{skin}}\) 为皮肤有效区域 mask。

编码器建议使用：

- 轻量 CNN；
- ResNet18 浅层；
- 深度可分离卷积编码器；
- 共享浅层加独立适配器。

输出：

\[
z_{\text{ref}}
\]

作用：

- 提取更偏向反射相关的信息；
- 减少照明明暗变化的影响；
- 提供 RGB 主分支可能没有稳定利用的增量信息。

不建议直接再训练一个与 RGB 分支同规模的完整 ResNet18，以降低参数量和过拟合风险。

### 8.3 Acquisition 分支

输入：

\[
\left\{
\hat L,
\hat S_{\text{spec}},
\hat R,
\text{camera},
\text{EXIF},
q_{\text{image}}
\right\}
\]

输出：

\[
z_{\text{acq}}
\]

作用：

- 显式建模相机；
- 显式建模曝光和 ISO；
- 显式建模光照；
- 显式建模高光；
- 显式建模物理分解失败模式。

该分支的主要任务不是直接提高分类 AUC，而是承载和识别采集条件信息。

---

## 9. 反事实一致性学习

### 9.1 预测一致性

要求原图和反事实重光照图具有近似的分类概率：

\[
p(y\mid I)
\approx
p(y\mid \hat I_k)
\]

对应损失：

\[
\mathcal L_{\text{pred-cons}}
\]

可采用：

- KL divergence；
- Jensen–Shannon divergence；
- 对称 KL；
- soft probability consistency。

### 9.2 特征一致性

要求原图和反事实图的疾病表示接近：

\[
z(I)
\approx
z(\hat I_k)
\]

对应损失：

\[
\mathcal L_{\text{feat-cons}}
\]

可采用：

- cosine distance；
- L2 distance；
- normalized feature distance。

### 9.3 一致性学习的目标

- 降低光照变化引起的预测波动；
- 降低模型对单一曝光和照明模式的依赖；
- 鼓励模型保留跨光照稳定的疾病相关信息。

### 9.4 反事实稳定性评价

对每例患者计算：

\[
\sigma_p=
\operatorname{Std}
\left[
p(I),p(\hat I_1),\ldots,p(\hat I_K)
\right]
\]

并报告：

- 平均预测标准差；
- 最大概率变化；
- 预测标签翻转率；
- 原图与重光照图的特征余弦相似度。

---

## 10. 疾病—采集表示解耦

### 10.1 目标

构建：

\[
z_{\text{disease}}
\]

和：

\[
z_{\text{acq}}
\]

并希望二者尽量减少统计相关性：

\[
z_{\text{disease}}
\perp
z_{\text{acq}}
\]

### 10.2 疾病表示

疾病表示主要来自：

\[
z_{\text{rgb}}
\quad\text{和}\quad
z_{\text{ref}}
\]

它用于最终 Control 与 Patient 分类。

### 10.3 采集表示

采集表示主要来自：

- illumination；
- specular-like；
- residual；
- camera；
- EXIF；
- 图像质量变量。

### 10.4 候选约束

#### Cross-covariance constraint

最小化疾病表示和采集表示的交叉协方差。

#### Orthogonality constraint

鼓励两个表示在特征空间中正交。

#### Gradient reversal / adversarial learning

使用采集条件预测器尝试从疾病表示中预测 camera 或 EXIF，同时通过梯度反转使疾病编码器去除可预测的采集信息。

### 10.5 使用前提

不能默认启用强对抗解耦。

必须先确认：

- camera 与二分类标签具有足够交叉覆盖；
- 每个主要 camera 同时包含 Control 和 Patient；
- EXIF 与标签不是近乎完全绑定；
- 去除 camera 信息不会等价于删除标签信息。

若某个 camera 几乎只对应单一类别，则当前数据无法可靠识别疾病因素与 camera 因素，强行解耦可能造成错误结论。

---

## 11. 可靠性门控融合

### 11.1 动机

物理分解并非对所有病例都可靠。常见失败情况包括：

- 过曝；
- 低照度；
- 镜面高光严重；
- 阴影严重；
- 面部遮挡；
- 皮肤有效区域不足；
- 姿态偏离；
- EXIF 缺失；
- JPEG 压缩明显；
- albedo-like 仍含大量阴影。

因此不能对所有病例固定使用相同的 reflectance-like 融合权重。

### 11.2 残差门控融合

建议采用：

\[
z_{\text{fused}}
=
z_{\text{rgb}}
+
g(q)\odot W_{\text{ref}}z_{\text{ref}}
\]

其中：

\[
g(q)\in[0,1]
\]

### 11.3 质量指标

质量向量 \(q\) 可由以下指标构成：

- reconstruction error；
- saturation ratio；
- specular ratio；
- valid skin fraction；
- face alignment quality；
- illumination extremeness；
- EXIF completeness；
- relighting identity consistency；
- physical decomposition failure score。

### 11.4 门控行为

- 当物理分支可靠时，提高 reflectance-like 表征贡献；
- 当物理分支不可靠时，降低其贡献；
- 极端情况下退化为原 RGB 主分支；
- 防止物理分支系统性破坏当前 E0B 强基线。

### 11.5 必须检查门控退化

需要检查：

- \(g(q)\) 是否长期接近 0；
- \(g(q)\) 是否长期接近 1；
- 门控值是否真正随质量指标变化；
- 低质量病例的门控值是否更低；
- 门控是否只学习标签而非质量。

---

## 12. 联合优化目标

完整候选损失为：

\[
\mathcal L
=
\mathcal L_{\text{cls}}
+
\lambda_{\text{pred}}\mathcal L_{\text{pred-cons}}
+
\lambda_{\text{feat}}\mathcal L_{\text{feat-cons}}
+
\lambda_{\text{acq}}\mathcal L_{\text{acq}}
+
\lambda_{\text{dis}}\mathcal L_{\text{disentangle}}
+
\lambda_{\text{gate}}\mathcal L_{\text{gate}}
\]

其中：

### 分类损失

\[
\mathcal L_{\text{cls}}
\]

使用训练折内计算类别权重的二分类交叉熵。

### 预测一致性损失

\[
\mathcal L_{\text{pred-cons}}
\]

约束原图与反事实图的预测一致。

### 特征一致性损失

\[
\mathcal L_{\text{feat-cons}}
\]

约束原图与反事实图的疾病表示一致。

### 采集建模损失

\[
\mathcal L_{\text{acq}}
\]

用于 camera 分类、EXIF 回归或 illumination 参数预测。

### 解耦损失

\[
\mathcal L_{\text{disentangle}}
\]

可由 cross-covariance、orthogonality 或 adversarial loss 构成。

### 门控损失

\[
\mathcal L_{\text{gate}}
\]

用于约束门控稳定性和质量一致性。

第一版不能一次启用全部损失。必须按照 P0–P4 逐步验证。

---

## 13. 实验推进路线

## P0：物理可行性审计

### 目标

确认公开预训练物理分解模型在当前人脸照片上是否可用。

### 输入范围

建议先抽取约 30–50 例，覆盖：

- Control 和 Patient；
- 不同 camera；
- 不同曝光；
- 不同肤色；
- 不同图像质量；
- 不同镜面高光程度；
- 不同照明方向。

### 检查内容

- albedo-like 是否保留明显阴影；
- shading-like 是否包含大量身份纹理；
- normal 是否稳定；
- 高光是否错误进入 albedo-like；
- 深色肤色是否被系统性改变；
- 重光照是否改变身份；
- 重光照是否产生明显伪影；
- 完整场景和裁剪人脸的光照估计是否一致；
- 不同物理前端的结果是否基本一致。

### 定量指标

- reconstruction MAE；
- SSIM；
- saturation ratio；
- valid skin fraction；
- specular ratio；
- 左右面颊 albedo-like 差异；
- relighting identity similarity；
- 物理分解失败率。

### P0 成功标准

- 大部分样本的物理分解结果无明显系统性错误；
- 重光照保持身份和标签语义；
- 失败病例可以通过质量指标识别；
- albedo-like、illumination、specular-like 和 residual 具有可区分的信息行为。

### P0 失败标准

若普遍出现以下情况，应停止完整逆渲染路线：

- albedo-like 中仍有严重阴影；
- 高光被大面积编码进 albedo-like；
- 重光照明显改变肤色或身份；
- 不同物理模型输出完全不一致；
- 失败病例无法由质量指标识别。

失败后可退回较简单的：

> 镜面/阴影抑制 + 区域光学关系建模。

---

## P1：物理分量信息定位

### 目标

判断不同物理分量分别包含什么信息。

### 基础实验

| Model | Input | Purpose |
|---|---|---|
| P1-RGB | 原始对齐人脸 | 复现 E0B |
| P1-A | albedo-like | 判断反射相关信息 |
| P1-N | normal | 判断几何信息 |
| P1-L | illumination | 判断光照与采集信息 |
| P1-S | shading-like | 判断明暗和几何信息 |
| P1-Spec | specular-like | 判断高光和采集捷径 |
| P1-R | residual | 判断未解释成分 |
| P1-RGB+A | RGB + albedo-like | 判断简单互补性 |

### 同时进行的采集泄漏探针

在每种表示上训练轻量 probe，预测：

- camera；
- ExposureTime；
- ISO；
- FNumber；
- BrightnessValue；
- 拍摄时间段。

### 结果解释

#### 支持 reflectance 路线

如果：

- albedo-like 保留较好的疾病分类能力；
- camera 和 EXIF 泄漏低于 RGB；
- shading/specular 主要携带采集信息；
- RGB + albedo-like 优于单一 RGB。

#### 不支持 reflectance 路线

如果：

- albedo-like 接近随机；
- shading/specular/residual 反而具有最高分类能力；
- albedo-like 的 camera 泄漏没有下降；
- RGB + albedo-like 不提供任何增量。

---

## P2：物理反事实增强

### 比较实验

1. E0B：无额外增强；
2. E0B + 标准 ColorJitter；
3. E0B + 物理反事实重光照；
4. E0B + 物理重光照 + 预测一致性；
5. E0B + 物理重光照 + 特征一致性；
6. E0B + 完整反事实一致性。

### 评价重点

不仅看分类 AUC，还要看：

- 原图与反事实图的预测差异；
- 预测标签翻转率；
- 特征相似度；
- 不同 camera 内的性能；
- 反事实光照条件下的稳定性。

### P2 成功标准

- 物理重光照在分类或稳定性上优于 ColorJitter；
- 一致性损失能够降低预测波动；
- 不明显损害 Patient sensitivity 和 Control specificity；
- 不引入明显伪影捷径。

### P2 失败标准

若物理重光照与 ColorJitter 无差异，或者性能更差、伪影明显，则不应将反事实重光照作为主创新。

---

## P3：采集条件解耦

### 实验前提

先完成 camera/EXIF 与标签分布审计。

### 比较实验

- 无采集分支；
- 仅 acquisition auxiliary branch；
- cross-covariance constraint；
- orthogonality constraint；
- gradient reversal；
- 组合约束。

### 评价重点

- 主任务 AUC；
- Balanced Accuracy；
- sensitivity；
- specificity；
- camera probe accuracy；
- EXIF probe \(R^2\)；
- 不同 camera 内性能；
- 跨 camera 性能稳定性。

### P3 成功标准

- 疾病表示中的 camera/EXIF 可预测性下降；
- 主分类能力保持或提高；
- 不同 camera 的性能差异缩小；
- 结果不能仅表现为删除全部颜色信息。

### P3 失败标准

- 采集泄漏下降但主分类性能明显崩溃；
- camera 与标签近乎完全绑定；
- 解耦结果无法与简单正则化区分；
- 对抗训练高度不稳定。

---

## P4：可靠性门控融合

### 比较实验

- direct concatenation；
- fixed-weight fusion；
- ordinary learned gate；
- quality-aware gate；
- quality-aware residual fusion。

### 评价重点

- 是否优于简单拼接；
- 是否保护 E0B 主分支；
- 低质量病例的门控值是否更低；
- 高质量病例是否真正获得物理分支增益；
- 门控是否退化为固定常数。

### P4 成功标准

- 患者级 Pooled AUC 不低于 E0B，最好提高约 0.01–0.02；
- Patient sensitivity 不明显下降；
- Control specificity 保持或提高；
- camera/EXIF 泄漏下降；
- 反事实稳定性改善；
- 门控与物理质量指标具有合理关系。

### P4 失败标准

- 门控长期接近 0 或 1；
- 只提高内部 AUC，但采集泄漏不下降；
- 对低质量病例没有保护作用；
- 与普通拼接没有可重复差异。

---

## 14. 正式候选模型数量控制

为避免在固定五折上过度搜索，第一轮正式模型建议限制为：

| ID | Model |
|---|---|
| B0 | E0B Global ResNet18 |
| B1 | E0B + strong ColorJitter |
| M1 | RGB + frozen albedo-like residual branch |
| M2 | M1 + physics-based relighting augmentation |
| M3 | M2 + counterfactual prediction/feature consistency |
| Full | M3 + acquisition disentanglement + reliability-aware fusion |

P1 中的物理分量独立分类属于机制审计，不作为大规模模型选择空间。

---

## 15. 统一评价体系

### 15.1 分类性能

正式主结果优先按 483 个患者组汇总，报告：

- ROC-AUC；
- PR-AUC；
- Accuracy；
- Balanced Accuracy；
- Macro-F1；
- sensitivity；
- specificity；
- PPV；
- NPV；
- confusion matrix。

### 15.2 统计验证

基于同一患者的 OOF 预测进行：

- patient-level paired bootstrap；
- AUC 差值及 95% CI；
- sensitivity 差值；
- specificity 差值；
- Balanced Accuracy 差值；
- 多随机种子复现。

### 15.3 概率校准

报告：

- Brier score；
- ECE；
- calibration curve；
- 必要时进行仅在训练/验证数据上拟合的 temperature scaling。

### 15.4 采集泄漏评价

使用轻量 probe 预测：

- camera；
- ExposureTime；
- ISO；
- FNumber；
- BrightnessValue；
- 拍摄时间段。

理想结果为：

- 疾病分类能力保留；
- camera 分类能力下降；
- EXIF 回归能力下降。

### 15.5 反事实稳定性评价

报告：

- relighting prediction standard deviation；
- maximum probability difference；
- label flip rate；
- feature cosine similarity；
- identity similarity。

### 15.6 跨采集条件评价

若样本量允许，报告：

- 各 camera 内部 AUC；
- 各 camera sensitivity 和 specificity；
- camera 间性能差异；
- leave-one-camera-out；
- camera-held-out 或时间外验证。

---

## 16. 总体成功标准

### 16.1 分类成功

相对 E0B：

- 患者级 Pooled AUC 最好提高约 0.01–0.02；
- 或至少不劣于 E0B 超过预设容忍范围；
- Patient sensitivity 不明显下降；
- Control specificity 保持或改善；
- Balanced Accuracy 和 Macro-F1 不明显退化。

### 16.2 采集稳健性成功

至少满足：

- camera probe 性能明显下降；
- EXIF probe 能力下降；
- 反事实重光照概率波动下降；
- 不同 camera 内部性能差异缩小；
- ColorJitter 不能完全复现物理方法效果。

### 16.3 机制成功

必须证明：

- albedo-like 与 shading/specular-like 承载的信息不同；
- 反事实一致性确实降低光照变化引起的预测波动；
- 可靠性门控在低质量样本上发挥作用；
- 采集解耦没有简单删除全部颜色信息。

### 16.4 可接受的非 AUC 提升结果

若总体 AUC 与 E0B 接近，但同时实现：

- 更低的 camera/EXIF 泄漏；
- 更稳定的跨光照预测；
- 更小的 camera 间性能差异；
- 更好的概率校准；
- 更高的 Control specificity；

仍可视为有意义的稳健性改进。

但若没有外部、时间外或 camera-held-out 验证，论文表述必须保持谨慎。

---

## 17. 失败停止标准

### 情况 1：物理分解普遍不可靠

停止完整物理逆渲染路线，退回更简单的镜面/阴影抑制和区域光学关系。

### 情况 2：albedo-like 没有疾病信息

不能继续将 albedo-like 作为主要疾病分支。

### 情况 3：物理重光照不优于 ColorJitter

不能声称物理增强具有独特价值。

### 情况 4：camera 与标签缺乏交叉覆盖

不能进行强疾病—camera 解耦，也不能声称实现了采集不变表示。

### 情况 5：完整模型只提高内部 AUC

如果 camera/EXIF 泄漏不下降、反事实稳定性不改善，则不能认定主线成功。

### 情况 6：门控退化

若门控恒为接近 0 或 1，则可靠性融合机制未成立。

---

## 18. 禁止偏离的研究方向

以下方向不再作为当前正式论文主线。

### 禁止 1：继续简单增加 ROI 数量

既往实验未证明简单 ROI 堆叠具有稳定互补性。

### 禁止 2：单纯继续调 ResNet

普通主干替换、学习率搜索或解冻层搜索不能构成计算成像创新。

### 禁止 3：删除困难病例

不得删除一致误判、低置信度或模型难以分类的病例以提高主结果。

### 禁止 4：在 500 例上训练大型端到端逆渲染网络

数据规模不足，容易产生伪物理解和严重过拟合。

### 禁止 5：声称恢复真实生理参数

普通 RGB JPEG 缺少物理可识别条件。

### 禁止 6：未经审计直接使用域对抗

必须先检查 camera、EXIF 和标签之间是否存在足够重叠。

### 禁止 7：一次性堆叠所有模块

必须按 P0、P1、P2、P3、P4 递进验证。

### 禁止 8：仅凭总体 AUC 判断成功

必须同时检查采集泄漏、反事实稳定性、敏感度、特异度、校准和统计置信区间。

### 禁止 9：回退到旧论文主线

既往 Global、ROI、Stage 1 Raw、Stage 2A、Stage 2B 研究仅作为背景、基线和对照，不再作为当前论文主体。

---

## 19. 当前论文创新核心

最终贡献不是：

> 使用了一个新的分类网络。

而是：

> 提出一种面向普通医学人脸照片的采集不变物理表示学习框架，通过冻结人脸物理分解先验、物理反事实重光照一致性、疾病—采集表示解耦和可靠性门控融合，减少 RGB 面部照片中的采集条件捷径，学习更稳定的心功能异常相关视觉表征。

### 建议的三项论文贡献

1. 提出面向医学人脸二分类的物理反事实重光照一致性框架，使同一患者在多个虚拟照明条件下保持稳定疾病表示。
2. 构建疾病表示与采集表示解耦机制，减少 camera、EXIF、照明和高光等采集条件在疾病特征中的泄漏。
3. 提出基于重建质量、饱和比例、高光比例和皮肤有效面积的可靠性残差融合机制，在物理分解不可靠时自动退化为 RGB 主分支。

---

## 20. 后续所有研究决策原则

任何新增模块必须回答：

1. 它解决当前哪个已识别问题？
2. 它对应哪个物理、统计或学习机制假设？
3. 为什么现有数据能够支持该假设？
4. 如何设计独立消融证明其作用？
5. 成功标准是什么？
6. 失败标准是什么？
7. 是否增加过拟合风险？
8. 是否可能引入标签泄漏或采集捷径？
9. 是否优于更简单的对照方法？
10. 是否会损害 Patient sensitivity、Control specificity 或总体稳健性？

若无法明确回答，则不进入正式实验。

---

## 21. 后续 ChatGPT / Codex 使用约束

后续使用本文件时，应首先读取并确认以下内容：

1. 当前任务是 Control 与 NYHA I–IV Patient 二分类；
2. 当前参考基线为 E0B Global ResNet18，Pooled OOF AUC 约为 0.855；
3. 当前正式论文主线是：
   - 冻结物理分解先验；
   - 反事实物理重光照；
   - 采集条件显式建模；
   - 疾病—采集解耦；
   - 可靠性门控融合；
4. 所有研究必须按照 P0–P4 递进推进；
5. 不能回退到旧 Global/ROI/Stage1/Stage2 融合主线；
6. 不能将 albedo-like 等中间表示解释为真实生理参数；
7. 不能为了提高 AUC 随意增加模块、删除病例或反复搜索固定五折；
8. 所有结论必须同时考虑分类性能、采集泄漏、反事实稳定性和统计不确定性。

建议后续 Chat 的首条指令为：

> 请先完整读取并严格遵守 `Face_Cardiac_Physics_Informed_Research_Roadmap.md`。后续所有研究分析、模型设计、代码实现、实验比较和结论判断必须以该文档为约束，不得回退到旧的 Global/ROI/Stage1/Stage2 论文路线，不得未经验证增加复杂模块，也不得将 reflectance-like 或 albedo-like 表示解释为真实生理参数。

---

## 22. 当前研究路线一句话总结

> 本研究以 E0B Global ResNet18 二分类模型为性能基线，通过冻结的人脸物理分解先验获得 reflectance-like、illumination、specular-like 和 residual 等中间表示，利用物理反事实重光照构造标签保持的一致性约束，并通过疾病—采集表示解耦和质量驱动门控融合，学习对相机、曝光和照明变化更稳定的心功能异常相关面部表征。
