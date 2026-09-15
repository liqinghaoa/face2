# Face–Cardiac Skin Optics 当前研究路线说明（2026-09）

> 本文严格按照 [研究路线图](Face_Cardiac_Skin_Optics_Research_Roadmap.png) 编排。路线从智能手机正面人脸照片开始，经过标准化处理、光谱皮肤光学前向建模、合成配对数据与监督训练、冻结真实人脸推理及完整图融合，最后进入心功能状态二分类探索。  
> **M-sensitive 与 H-sensitive 是模型定义的、对合成光学参数敏感的代理表征，不等同于真实黑色素或血红蛋白浓度。**

![研究路线图](Face_Cardiac_Skin_Optics_Research_Roadmap.png)

## 一、路线总览：六个阶段如何衔接

| 阶段 | 路线图节点 | 核心问题 | 主要输出 | 与下一阶段的接口 |
|---|---|---|---|---|
| 1 | Smartphone face photographs | 获取什么样的真实输入？ | 原始正面人脸照片及最小索引 | 进入统一的人脸标准化处理 |
| 2 | Standardized facial data processing | 如何把照片变成可比较、可切 patch 的输入？ | 对齐人脸、黑背景、skin mask、20 个 patch | 输出 linear-sRGB patch 与固定空间坐标 |
| 3 | Spectral skin-optics forward model | 如何由皮肤光学原理生成受控 RGB 与参数目标？ | 光谱反射率、相机观测与 M/H-sensitive target | 为合成配对数据提供生成器 |
| 4 | Synthetic paired RGB dataset and supervised neural-network training | 如何让网络学习 RGB→M/H 的映射？ | 经过验收的合成配对数据与 B1/B2 checkpoint | 冻结网络，禁止在真实人脸上继续训练 |
| 5 | Frozen M/H-sensitive and H-sensitive map inference plus mask-aware full-face fusion | 如何从真实人脸得到完整空间分布图？ | 500 例、10,000 patch、1,000 case-level fused maps | 为分类输入构建 RGB、RGB+M/H、M/H-only 张量 |
| 6 | Binary cardiac-function-status classification | M/H 表征是否为心功能状态分类提供补充信息？ | 内部 OOF 指标与结构探索结果 | 仅形成探索性证据，不外推临床结论 |

---

## 二、阶段 1：智能手机正面人脸照片

### 阶段目标

第一阶段的任务不是直接训练分类器，而是建立一个可审计的真实图像入口。输入被定义为智能手机采集的正面人脸照片，后续所有处理都围绕同一空间语义、同一颜色语义和同一病例索引展开。

### 输入与索引

每个真实样本至少保留：

- case ID：一次可评价的图像/就诊实例；
- 原始照片文件及其可追溯路径；
- 二分类标签仅在分类阶段按固定标签表读取，不参与 M/H 网络训练。

心功能状态分类使用 Normal（NYHA 0）与 Patient（NYHA I–IV）。当前 500 例的二分类标签为 115 例 Normal、385 例 Patient，并按固定 5-fold 患者级分组表组织，每折 100 例。

### 阶段控制点

- 原始照片确实是路线规定的正面人脸输入；
- 图像与 case ID、patient-group ID 的映射唯一；
- 同一患者的多次记录不跨 outer fold；
- 临床字段不作为图像模型特征。

### 阶段输出

阶段 1 输出可追踪的原始照片集合，而不是最终模型张量。照片随后进入阶段 2，转换为空间和颜色语义明确的标准化人脸。

---

## 三、阶段 2：标准化人脸数据处理

路线图将本阶段画成四个连续动作：**face alignment → black background → skin mask → 20 image patches**。

### 2.1 人脸对齐

通过人脸检测、关键点定位和几何对齐，将人脸放入统一坐标系。正式真实人脸完整空间为 1220×979×3。对齐的目的不是改变皮肤外观，而是让额头、双眼、鼻部、口周和下颌在不同个体之间具有可比较的位置。

### 2.2 黑背景

对齐后去除非人脸背景，以黑色填充。黑背景不是生理信息，也不是标签，只是把有效人脸主体与复杂背景分开，减少背景纹理进入后续模型的机会。

### 2.3 Skin mask

生成独立的 valid skin / face mask：

- mask 外区域不作为有效皮肤像素；
- M/H 完整图融合只在有效 mask 内进行；
- 分类输入补零行和 mask 外区域保持为 0；
- mask 不作为 SO-1 光学网络的额外输入通道。

### 2.4 20 个重叠 patch

完整人脸按固定行优先几何规则切成 20 个重叠 patch：

~~~
每个 patch：256×256×3
网络输入：3×256×256，CHW，float32
~~~

重叠 patch 兼顾两点：在有限显存内完成高分辨率局部推理，并在 patch 边界保留上下文，便于之后恢复到完整人脸空间。patch 坐标、顺序、重叠区域和 mask 由输入合同锁定。

### 2.5 分类输入统一尺寸

阶段 5 生成完整 M/H map 后，RGB、M、H 均按同一规则缩放：

1. 979×1220 按比例缩放到 256×319；
2. 底部补 1 行全零，得到 256×320；
3. 组织为 C×320×256、float32；
4. RGB 和 M/H 使用双线性插值，mask 使用最近邻插值。

---

## 四、阶段 3：光谱皮肤光学前向模型

SO-0 不是给 RGB 添加随机扰动，而是把皮肤分层、光谱、光源和相机响应组成一个可计算的前向生成链路。

### 3.1 模型输入

SO-0 使用冻结的 5 nm 光谱网格和非训练物理参数：

- 黑色素敏感控制量 m∈[0,1]；
- 血液敏感控制量 h∈[0,1]；
- 表皮与真皮厚度；
- 黑色素、血液和基底吸收谱；
- 真皮约化散射谱 μs′(λ)；
- 光源光谱功率分布 SPD；
- 相机光谱敏感度函数 SSF；
- 曝光、阴影和镜面反射等采集条件。

m/h 是生成器内的敏感性控制变量，不是个体真实生化浓度。

### 3.2 表皮层

黑色素敏感比例：

~~~
f_mel = 0.013 + 0.417m
~~~

表皮吸收系数：

~~~
μa,epi(λ) = f_mel μa,mel(λ) + (1−f_mel) μa,base(λ)
~~~

Beer–Lambert 一程透射：

~~~
T_epi(λ) = exp[−μa,epi(λ) d_epi]
~~~

### 3.3 真皮层

血液敏感比例：

~~~
f_blood = 0.02 + 0.05h
~~~

真皮吸收系数：

~~~
μa,dermis(λ) = f_blood μa,blood(λ) + (1−f_blood) μa,base(λ)
~~~

真皮反射采用有限厚度 Kubelka–Munk 计算，输入是真皮吸收、约化散射和真皮厚度。H-sensitive 变化因此来自血液敏感吸收项对光谱反射的影响，而不是后处理涂色。

### 3.4 分层反射率

~~~
R_skin(λ) = T_epi(λ)^2 R_dermis(λ)
~~~

平方项表示光进入真皮并返回时经历两次表皮透射。

### 3.5 光照与相机观测

谱辐亮度：

~~~
E(λ) = exposure · SPD(λ) [shading · R_skin(λ) + specular]
~~~

相机 RGB 由相机 SSF 对光谱进行加权积分：

~~~
RGB_raw = Σλ wλ E(λ) · SSF(λ)
~~~

随后执行对角白平衡，以及通过 ColorChecker 标定的无截距颜色矩阵完成 RGB 与 XYZ/D65 颜色链变换。

### 3.6 阶段边界

该模型回答“模型内 m/h 改变时，在不同光源和相机下线性 RGB 如何变化”，不回答“真实个体的黑色素或血红蛋白绝对浓度是多少”。因此其输出是受控合成监督环境，而不是临床生理测量系统。

---

## 五、阶段 4：合成配对数据与监督神经网络训练

路线图第四节点的上半部分是成对 RGB 与目标图，下半部分是监督神经网络训练。两者必须使用同一套冻结物理协议。

### 4.1 合成数据基本单位

每个 latent 表示固定的模型内皮肤参数、空间分布和 mask。同一 latent 可以在不同相机、光源或外观条件下渲染出多个 acquisition。

每条 acquisition 包含：

- linear-RGB 图像；
- M-sensitive target map；
- H-sensitive target map；
- S/P 辅助 target；
- valid mask；
- latent ID、acquisition ID、相机、光源和采集角色。

### 4.2 相机、光照与采集角色

正式 G1-AM1 使用 6 类相机：

- Canon 1DMarkIII
- Canon 5DMarkII
- Hasselblad H2
- Nikon D5100
- Nikon D80
- Point Grey Grasshopper2 14S5C

使用 4 类光源：

- D65
- A
- FL2
- FL11

manifest 记录 24 个相机–光源组合，并显式标记 seen / unseen camera 与 light role。配对角色包括 reference、camera_only、light_only、appearance_only 和 joint。

### 4.3 数据规模与验收

G1-AM1 manifest 记录：

- 67,500 条 RGB acquisition；
- 54,000 条配对记录；
- 单条 RGB 为 [3,256,256]、float16；
- Train 50,000；
- Validation 5,000；
- ID Test 5,000；
- Camera-OOD、Light-OOD、Joint-OOD 各 2,500。

正式验收为：

- dataset status = ACCEPTED；
- 1 worker 与 8 worker：128/128 exact；
- independent replay：32/32；
- nonfinite = 0；
- high/low clipping violations = 0；
- 受保护资产保持不变。

### 4.4 RGB→M/H 网络

SO-1 使用固定容量 U-Net：

~~~
输入：3 通道 linear RGB
编码器：64 → 128 → 256 → 512
瓶颈：1024
解码器：512 → 256 → 128 → 64
输出头：1×1 Conv + Sigmoid
输出：M、H、S_norm、P_norm 四张 map
~~~

每个双卷积模块包含卷积、BatchNorm 和 LeakyReLU。下采样获得上下文，skip connection 将浅层空间细节传递到对应解码层。

### 4.5 监督训练与正式 checkpoint

训练使用有效 mask 内的 masked SmoothL1 损失，β=0.1，四个输出通道等权。真实人脸不参与训练、微调或参数更新。

- B1 baseline：checkpoint 权威，best epoch 25，viability PASS；
- B2 proposed λ=0.50：validation viability PASS、non-inferiority PASS、selected epoch 25，checkpoint 权威。

λ 是成对采集一致性约束的候选权重，仅作用于合成训练协议，不改变 M/H 定义；最终 λ=0.50 已冻结。

---

## 六、阶段 5：冻结真实人脸推理与完整图融合

阶段 4 得到的网络在进入本阶段前冻结。本阶段只进行真实人脸的前向推理和空间融合。

### 5.1 Patch 级冻结推理

对每个真实人脸 patch：

1. 读取固定坐标和 RGB patch；
2. 分别输入冻结的 B1 与 B2 λ=0.50 网络；
3. 得到 patch 级 M-sensitive 与 H-sensitive prediction；
4. 保留 valid mask、weight 和 coverage 信息；
5. 使用 eval、无梯度、FP32、无数据增强、无测试时自适应。

当前真实冻结推理记录：

- 500 个 case；
- 20 个 patch/case；
- 10,000 个 patch-level outputs；
- B1 与 B2 各生成 500 个 case-level M/H map 集合，共 1,000 个 case-level map sets。

### 5.2 Mask-aware 完整图融合

20 个 patch 按原始坐标放回 1220×979 完整人脸空间，在重叠区域内对有效 mask 像素做加权融合：

~~~
ŷ(p) = [Σi wi(p) mi(p) ŷi(p)] / [Σi wi(p) mi(p)]
~~~

mask 外输出置为 0。每例最终得到：

~~~
B1 M-sensitive map：1220×979
B1 H-sensitive map：1220×979
B2 M-sensitive map：1220×979
B2 H-sensitive map：1220×979
~~~

分类阶段使用完整图融合后再缩放的 M/H 表征，而不是直接使用 patch 图作为病例级输入。

### 5.3 阶段验收状态

X1 冻结推理验收记录：

- case_count = 500；
- patch_count = 10,000；
- fusion_coverage_complete = true；
- all outputs finite；
- classification_started = false；
- labels_accessed = false；
- 输入数据和受保护资产未被修改。

---

## 七、阶段 6：心功能状态二分类

最后阶段训练二分类器，将 Normal 与 Patient 进行区分。分类阶段只使用冻结的真实输入和固定的 5-fold 患者级分组协议。

### 6.1 统一设计

- 评价单位：case；
- 固定 5-fold 患者级分组；
- outer test 不参与训练、早停 epoch 选择或阈值选择；
- development set 内再划分 train/validation；
- 汇总 500 例 OOF 预测；
- 固定阈值：prob_patient ≥ 0.5；
- 指标：Macro-AUC、Macro-F1、Balanced Accuracy、Sensitivity、Specificity。

### 6.2 输入实验组

| 实验组 | 输入 | 研究目的 |
|---|---|---|
| RGB | 3 通道 | 原始线性 RGB 基线 |
| RGB_B1MH | 5 通道 | RGB + B1 M/H-sensitive maps |
| RGB_B2MH | 5 通道 | RGB + B2 λ=0.50 M/H-sensitive maps |
| B1_MH_ONLY | 2 通道 | 只使用 B1 的 M/H |
| B2_MH_ONLY | 2 通道 | 只使用 B2 的 M/H |
| B2_M_ONLY | 1 通道 | 只分析 B2 的 M |
| B2_H_ONLY | 1 通道 | 只分析 B2 的 H |
| LATE_CONCAT | 特征级融合 | X4 结构探索 |
| MH_LRF | 特征级融合 | X4 低秩融合探索 |
| MH_LRTM | 特征级融合 | X4 变换式融合探索 |

### 6.3 已完成的 pooled OOF 结果

| 实验组 | Macro-AUC | Macro-F1 | Balanced Accuracy | Sensitivity | Specificity |
|---|---:|---:|---:|---:|---:|
| RGB | 0.8370 | 0.7216 | 0.7552 | 0.7974 | 0.7130 |
| RGB_B1MH | 0.7986 | 0.6897 | 0.7100 | 0.8026 | 0.6174 |
| RGB_B2MH | 0.8400 | 0.7131 | 0.7304 | 0.8260 | 0.6348 |
| B1_MH_ONLY | 0.8222 | 0.6594 | 0.6599 | 0.8416 | 0.4783 |
| B2_MH_ONLY | 0.8055 | 0.6693 | 0.6787 | 0.8182 | 0.5391 |
| B2_M_ONLY | 0.8166 | 0.6952 | 0.7139 | 0.8104 | 0.6174 |
| B2_H_ONLY | 0.8381 | 0.7082 | 0.7230 | 0.8286 | 0.6174 |
| LATE_CONCAT | 0.8204 | 0.6908 | 0.7012 | 0.8286 | 0.5739 |
| MH_LRF | 0.8043 | 0.6711 | 0.6817 | 0.8156 | 0.5478 |
| MH_LRTM | 0.8096 | 0.6869 | 0.7139 | 0.7844 | 0.6435 |

结果为内部 OOF 的描述性比较。RGB_B2MH 的 pooled Macro-AUC 比 RGB 高约 0.00305，但固定阈值下的 Macro-F1、Balanced Accuracy 和 Specificity 没有同步提高，提示排序能力和固定阈值决策是两个不同层面。当前结果不用于自动选择 winner，也不外推为临床性能。

---

## 八、贯穿六个阶段的审计与版本边界

### 数据与空间

- 原始照片、case ID 与 patient-group ID 可追踪；
- 5-fold 划分保持患者级隔离；
- 对齐、黑背景、mask、patch 坐标固定；
- patch 融合后 coverage 完整、mask 外归零。

### 物理与合成

- SO-0 光谱、相机 SSF、光源 SPD、颜色链和公式版本冻结；
- G1-AM1 manifest 记录相机、光源、采集角色和 split；
- worker parity、independent replay、clip 与 finite 检查通过；
- Nokia N900 等已淘汰相机方案不恢复、不重新纳入训练或测试。

### 模型与训练

- SO-1 U-Net 输入输出通道顺序固定；
- B1、B2 checkpoint 记录路径、版本和哈希；
- B2 λ=0.50 已冻结；
- 真实人脸阶段不训练、不微调、不更新参数。

### 分类与结果

- 仅使用固定 OOF 预测和预先定义的指标；
- 不读取不必要的临床字段作为模型特征；
- 不以 outer-test 结果进行模型选择；
- OOF 概率、checkpoint 和正式结论不被诊断分析覆盖。

---

## 九、当前研究结论与下一步

当前路线已经形成：

**智能手机人脸照片 → 标准化人脸与 patch → 光谱皮肤光学前向模型 → 合成配对监督 → RGB→M/H 网络 → 冻结真实推理与完整图融合 → 内部心功能状态二分类探索。**

后续重点：

1. 统一实验索引、数据版本、协议锁、checkpoint、OOF 和报告链接；
2. 将物理前向假设、M/H map、patch/case 融合和分类错误连接成整体可解释性证据链；
3. 明确区分物理模型生成的敏感代理表征、机器学习内部 OOF 结果和临床可验证结论；
4. 在新的授权和外部验证证据出现前，保持：

~~~
SO-R1-C = FAIL
official SO-R2 authorization = false
SO-R3 authorization = false
~~~

> **当前最准确的表述：** 一条已经完成关键实现和内部审计、能够从真实人脸生成 M/H-sensitive 代理表征并进行受控分类探索的研究路线；它还不是一个经过外部验证的临床诊断系统。

