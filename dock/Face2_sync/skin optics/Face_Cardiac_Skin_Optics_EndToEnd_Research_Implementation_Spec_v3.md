# 基于皮肤光学敏感表征的人脸心功能二分类研究与实现规范

**文档版本：** v3.0-End-to-End  
**形成日期：** 2026-08-02  
**适用项目：** 面部图像与心功能状态研究  
**研究任务：** Control（NYHA 0）与 Patient（NYHA I–IV）二分类  
**文档性质：** 当前研究主线、工程接口、训练协议、评价协议与停止规则的统一约束  
**上游固定文档：** `Face_Cardiac_Skin_Optics_Research_Roadmap_v2_Lean.md`  

---

## 1. 文档目的

本文档用于固定当前皮肤光学研究的完整端到端方案，防止后续讨论、Codex 实现、实验训练和结果解释偏离主线。

研究最终需要回答的问题是：

> 基于皮肤光学前向模型合成监督所学习的 melanin-sensitive 与 hemoglobin-sensitive 分布图，能否在真实普通手机人脸图像中形成稳定表征，并为心功能状态二分类提供 RGB 之外的增量信息？

后续工作遵循以下原则：

1. 优先复用现有数据、固定五折、对齐人脸、皮肤 mask、RGB 基线及 SO-0 资产；
2. 每个阶段只解决一个清晰问题，不进行无必要的模块扩展；
3. SO-0 物理公式不得根据分类结果反向修改；
4. 第一版先验证最小闭环，出现正向信号后才进入完整五折和可选优化；
5. 所有失败结果均允许成为正式结论，不通过不断增加模型复杂度掩盖负结果。

---

## 2. 科学定位与术语边界

### 2.1 正确定位

本研究构建的是：

> 面向普通手机人脸图像的皮肤光学约束色基敏感表征学习框架。

模型输出包括：

- `M`：melanin-sensitive map；
- `H`：hemoglobin-sensitive map；
- `S`：shading-related map；
- `P`：specular-related map。

这些输出是由简化皮肤光学模型定义并通过合成数据监督学习的中间表征。

### 2.2 禁止解释

本研究没有标准色卡同步采集、皮肤光谱仪、组织学真值、VISIA 真值或已知手机 ISP，因此不得将真实人脸上的输出解释为：

- 真实黑色素浓度；
- 真实血红蛋白浓度；
- 血氧饱和度；
- 皮肤灌注；
- 皮肤血流；
- 定量组织氧合；
- 可直接用于临床诊断的生理测量。

正确表述为：

- melanin-sensitive representation；
- hemoglobin-sensitive representation；
- skin-optics-informed representation；
- chromophore-sensitive representation；
- shading-related component；
- specular-related component。

### 2.3 M、H、S、P 的角色

| 变量 | 定义 | 主要作用 | 是否作为主要医学敏感表征 |
|---|---|---|---|
| M | 控制表皮黑色素相关吸收的归一化空间场 | 表达黑色素吸收方向的变化 | 是，但只能称 melanin-sensitive |
| H | 控制真皮血液/血红蛋白相关吸收的归一化空间场 | 表达血红蛋白吸收方向的变化 | 是，但只能称 hemoglobin-sensitive |
| S | 漫反射强度的空间调制场 | 吸收曲面明暗、方向性光照和部分曝光影响 | 否，属于采集条件辅助分量 |
| P | 皮肤表面镜面反射的有效强度场 | 吸收高光和表面直接反射 | 否，属于采集条件辅助分量 |

虽然正式分类主要使用 M/H，但分解网络仍输出 S/P。原因是如果没有 S/P，阴影、高光和曝光变化可能被错误写入 M/H，从而破坏色基敏感语义。

---

## 3. 总体系统结构

完整研究由六个核心模块和两个评价阶段组成：

```text
模块 A：SO-0 皮肤光学前向模型
        ↓
模块 B：合成 M/H/S/P 与 RGB Patch 数据生成器
        ↓
模块 C：SO-1 分解网络预训练
        ↓
模块 D：真实人脸数据预处理
        ↓
模块 E：Patch 划分器、SO-1 推理与 Patch 合并器
        ↓
模块 F：心功能状态分类模型
        ↓
SO-2：真实数据与 fold 0 快速筛选
        ↓
SO-3：出现正向信号后进行完整固定五折验证
```

各模块职责不可混淆：

| 模块 | 输入 | 输出 | 是否训练 |
|---|---|---|---|
| SO-0 前向模型 | M/H/S/P、光源、相机、曝光 | 多阶段颜色输出及合成 sRGB patch | 否，已冻结 |
| 合成数据生成器 | 随机参数、SO-0 | RGB patch 与 M/H/S/P 真值 | 否 |
| SO-1 分解网络 | linear-RGB patch | M/H/S/P 预测图 | 是 |
| 真实预处理 | 手机人脸图、对齐参数、皮肤 mask | 高分辨率标准化人脸与 mask | 否 |
| Patch 划分与合并 | 高分辨率人脸、mask、SO-1 | 完整高分辨率 M/H/S/P 图 | 否 |
| 分类模型 | RGB、M/H 或其融合 | Control/Patient 概率 | 是 |

---

# 第一部分：SO-0 皮肤光学前向模型

## 4. SO-0 的作用

SO-0 是确定性的、可微分的皮肤光学前向算子，不是分类网络，也不是通过患者数据训练得到的模型。

其任务是模拟：

```text
M / H / S / P
+ 光源光谱
+ 相机光谱响应
+ 曝光
        ↓
皮肤光谱反射率
        ↓
光谱辐射
        ↓
相机 RGB
        ↓
公共颜色空间
        ↓
可观察合成图像
```

SO-0 的主要作用：

1. 为 SO-1 提供可以无限抽样的合成 RGB patch；
2. 为每个 RGB patch 提供精确对应的 M/H/S/P 合成真值；
3. 固定 M/H/S/P 的物理含义和变化方向；
4. 支持多光源、多相机采集条件随机化；
5. 为 SO-1 测试阶段提供离线物理一致性审计；
6. 防止分解网络自由定义没有物理约束的中间变量。

## 5. SO-0 已冻结状态

冻结版本：

```text
SO0_Forward_Model_v1.1
```

已确认结果：

- NumPy 与 PyTorch 后端一致性通过；
- 正式生产波长网格为 5 nm；
- 参考审计网格为 1 nm；
- 121,968 个相机前向条件通过分辨率审计；
- 112 个 camera-light 组合中 109 个合格；
- M/H 在已测试合成观测空间中未发生秩退化；
- 所有核心物理公式、阈值和颜色转换已冻结。

以下三个组合不得进入 SO-1 正式数据生成：

```text
Point Grey Grasshopper 50S5C / D65
Point Grey Grasshopper 50S5C / A
Point Grey Grasshopper 50S5C / FL2
```

## 6. SO-0 的颜色形成链

SO-0 不只输出一种“RGB”。完整相机路径依次产生：

```text
皮肤反射率
    ↓
spectral_radiance
    ↓
camera_rgb_raw
    ↓
camera_rgb_wb
    ↓
xyz_source
    ↓
xyz_d65
    ↓
linear_srgb_unclipped
    ↓
srgb_unclipped
    ↓
srgb_display_clipped
```

统一术语：

| 输出 | 含义 | 用途 |
|---|---|---|
| `camera_rgb_raw` | 相机传感器响应空间中的原始 RGB | 物理审计 |
| `camera_rgb_wb` | 对角白平衡后的相机 RGB | 物理审计 |
| `linear_srgb_unclipped` | Gamma 编码前、可能小于 0 或大于 1 的线性 sRGB | 数值审计 |
| `srgb_unclipped` | 非线性编码但尚未裁剪的 sRGB | 越界审计 |
| `srgb_display_clipped` | 标准 sRGB 编码并裁剪至 `[0,1]` 的可观察图像 | 合成数据保存与真实图接口模拟 |

## 7. SO-0 为 SO-1 生成的最终图像类型

SO-0 为合成训练数据生成的可观察 patch 固定为：

```text
名称：synthetic_srgb_patch
颜色表示：标准非线性 sRGB
通道：R、G、B
尺寸：3 × 256 × 256
范围：[0,1]
数据类型：float32
来源：SO-0 camera render 路径的 srgb_display_clipped
```

SO-0 生成的不是未经处理的相机 RAW，也不是直接拿 `linear_srgb_unclipped` 作为可观察训练图像。

合成输入必须真正经过：

```text
camera spectral sensitivity
→ camera RGB
→ white balance
→ camera-to-XYZ matrix
→ Bradford adaptation
→ linear sRGB
→ sRGB encoding
→ display clipping
```

不得绕过相机路径，只调用简化的 CIE reference 路径生成训练数据。

## 8. SO-1 的实际颜色输入

SO-1 的实际输入固定为：

```text
名称：so1_input_linear_rgb_patch
更准确名称：linear-sRGB patch
形状：3 × 256 × 256
通道顺序：R、G、B
范围：[0,1]
数据类型：float32
```

生成方式：

```text
synthetic_srgb_patch
        ↓
标准 inverse sRGB transfer function
        ↓
so1_input_linear_rgb_patch
```

linear-sRGB 仍然是 RGB 图像，只是数值与光强近似呈线性关系。

真实人脸推理时必须使用完全相同的颜色入口：

```text
真实手机 RGB 图像
→ 按 sRGB 编码图像处理
→ 归一化到 [0,1]
→ inverse sRGB
→ linear-RGB
→ SO-1
```

第一版不得取消 inverse sRGB，也不得一部分图像使用 sRGB、另一部分图像使用 linear-RGB。

---

# 第二部分：合成训练数据生成

## 9. 合成数据的基本单位

SO-1 的训练单位是单个 patch，不是一张完整合成人脸，也不是由多个 patch 组成的病例。

每个样本包括：

```text
输入：
linear_rgb       [3,256,256]

监督真值：
M_norm           [1,256,256]
H_norm           [1,256,256]
S_norm           [1,256,256]
P_norm           [1,256,256]

元数据：
sample_id
base_latent_id
acquisition_variant_id
camera_id
light_id
seed
SO0_version
SO0_config_hash
clipping_statistics
```

## 10. 合成 Patch 尺寸

第一版直接生成：

```text
256 × 256
```

不采用参考论文中“310×310 再随机裁剪到 256×256”的训练策略。

原因：

- 参考论文从固定高分辨率真实人脸网格中提取训练区域，需要随机裁剪增加位置鲁棒性；
- 本项目的训练 patch 本身由随机空间场直接生成；
- M/H/S/P 的位置、形状和局部结构已经可以随机变化；
- 额外生成 310×310 再裁剪只会增加无必要的实现步骤。

## 11. M/H/S/P 空间场生成

### 11.1 M 空间场

组成：

```text
随机全局基值
+ 低频平滑空间变化
+ 少量局部色素结构
```

要求：

- 范围符合 SO-0 冻结定义；
- 以低频变化为主；
- 不得只生成全图常数；
- 局部结构不能覆盖绝大多数画面。

### 11.2 H 空间场

组成：

```text
独立全局基值
+ 独立低频变化
+ 少量局部红度结构
```

要求：

- M 与 H 使用独立随机数流；
- 不人为设定固定正相关或负相关；
- 不让网络通过人为共现关系推测另一个分量。

### 11.3 S 空间场

组成：

```text
方向性低频亮度梯度
+ 缓慢曲面变化
+ 可选轻度局部阴影
```

S 用于表示有效漫反射强度变化，不等价于真实人脸法线或精确光照方向。

### 11.4 P 空间场

组成：

```text
零或低背景
+ 0–4 个平滑、稀疏、非负高光斑
```

要求：

- P 大部分像素接近 0；
- 高光区域面积有限；
- 不能把 P 设计为全图均匀曝光项；
- 合成数据中必须包含无高光、弱高光和明显高光样本。

## 12. 四个监督目标的归一化

为避免不同取值范围导致损失不平衡，训练目标统一映射到 `[0,1]`。

固定变换：

```text
M_norm = M
H_norm = H
S_norm = (S - S_min) / (S_max - S_min)
P_norm = (P - P_min) / (P_max - P_min)
```

其中 `S_min/S_max/P_min/P_max` 必须来自 SO-0 冻结配置，不得根据生成数据集自身的最小值和最大值动态计算。

归一化仅用于网络输出与损失计算，不改变四个变量的物理角色。需要进行 SO-0 审计时，再按固定公式反归一化。

## 13. 模拟非皮肤区域

真实全脸 patch 中会包含眼睛、眉毛、嘴唇、鼻孔、毛发或人脸边缘。SO-1 合成训练若全部为 100% 皮肤区域，真实推理会存在明显输入分布差异。

因此，部分合成 patch 加入随机无效区域 mask：

```text
有效皮肤像素：保留合成 RGB 与 M/H/S/P
无效像素：RGB 设置为 0
监督损失：仅在有效 mask 内计算
```

模拟 mask 可以包含：

- 不规则边界；
- 椭圆或条带孔洞；
- 一个或多个局部无效区域；
- 画面边缘的部分无效区域。

目标不是模拟精确眼睛和嘴唇形状，而是让网络见过皮肤—黑色无效区边界。

## 14. 合成数据规模

第一版固定为：

| 数据集 | 数量 | 作用 |
|---|---:|---|
| Train | 20,000 | 网络参数更新 |
| Validation | 2,000 | 早停与 checkpoint 选择 |
| ID Test | 2,000 | 已见相机与已见光源测试 |
| Camera-OOD | 1,000 | 未见相机测试 |
| Light-OOD | 1,000 | 未见光源测试 |
| Joint-OOD | 1,000 | 未见相机与未见光源联合测试 |
| 总计 | 27,000 | SO-1 v1 完整数据 |

训练集建议由：

```text
10,000 组基础 M/H 空间场
× 每组 2 种采集条件
= 20,000 个 RGB patch
```

同一基础组中：

- M/H 保持一致；
- S/P 可以重新采样；
- camera/light/exposure 可以改变；
- 两个采集版本作为独立监督样本；
- 第一版不增加额外一致性损失。

## 15. 相机与光源划分

使用 SO-0 的 109 个合格 camera-light 组合。

原则：

1. 相机按 identity 划分 seen 与 unseen；
2. Canon 5D Mark II 必须保留在 seen 集；
3. 光源将 FL11 作为主要未见光源；
4. ID、Camera-OOD、Light-OOD、Joint-OOD 的 latent seed 必须互不重叠；
5. 所有列表在生成第一个样本前写入 `camera_light_split.json` 并冻结；
6. 不根据训练结果重新抽取相机或光源。

建议初始划分：

```text
Seen cameras：25
Unseen cameras：3

Seen lights：D65、A、FL2
Unseen light：FL11
```

只使用实际合格的组合。

## 16. 数据保存形式

科研计算使用浮点数据，不以 8 位 PNG 作为唯一输入。

建议：

```text
RGB / M / H / S / P：float16 或 float32 数组
preview：PNG
manifest：CSV 或 Parquet
configuration：YAML/JSON
```

必须记录：

- 生成 seed；
- camera/light；
- SO-0 版本和哈希；
- M/H/S/P 生成参数；
- 是否使用模拟 mask；
- clipping 比例；
- 数据划分。

不保存每个像素的完整波长中间张量，以避免不必要的存储开销。

---

# 第三部分：SO-1 分解网络预训练

## 17. SO-1 目标

SO-1 学习逆向映射：

```text
linear-RGB skin patch
        ↓
M-sensitive
H-sensitive
S-related
P-related
```

训练完成后，SO-1 的最终用途是：

```text
真实人脸 linear-RGB patch
        ↓
冻结 SO-1
        ↓
M/H/S/P patch
```

SO-1 不直接输出心功能标签。

## 18. 网络结构

第一版固定使用标准容量 U-Net，而不是 base-channels=32 的过轻版本。

网络结构：

```text
输入：3 × 256 × 256

Encoder：
64
128
256
512

Bottleneck：
1024

Decoder：
512
256
128
64

输出：
4 × 256 × 256
```

每级基本模块：

```text
3×3 Conv
BatchNorm
LeakyReLU
3×3 Conv
BatchNorm
LeakyReLU
```

下采样：

```text
2×2 MaxPool
```

上采样：

```text
2×2 Transposed Convolution
```

输出层：

```text
1×1 Conv：64 → 4
Sigmoid
```

输出通道顺序固定为：

```text
channel 0：M_norm
channel 1：H_norm
channel 2：S_norm
channel 3：P_norm
```

## 19. 初始化策略

SO-1 v1 使用随机初始化。

不使用 Carvana 预训练权重，原因：

- 公开 Carvana 模型是汽车二值分割；
- 参考论文未公开准确 checkpoint；
- 本项目有 20,000 个带精确四分量监督的合成 patch；
- 引入不匹配权重会增加结构与许可证依赖；
- 当前目标是验证皮肤光学路线，不做初始化搜索。

只有标准 U-Net 在训练集上也明显欠拟合时，才允许将预训练初始化作为后续单独方案。

## 20. 训练损失

第一版只使用合成真值的直接监督，不加入 RGB 重建训练损失。

对每个输出：

```text
L_M = masked SmoothL1(M_pred, M_true)
L_H = masked SmoothL1(H_pred, H_true)
L_S = masked SmoothL1(S_pred, S_true)
L_P = masked SmoothL1(P_pred, P_true)
```

总损失：

```text
L_total = (L_M + L_H + L_S + L_P) / 4
```

masked loss：

```text
loss =
sum(valid_mask × pixel_loss)
/
(sum(valid_mask) + epsilon)
```

说明：

- 四个目标均已映射到 `[0,1]`；
- 等权损失避免人为设置缺乏依据的 0.5 权重；
- M/H 的研究优先级通过 checkpoint 选择与后续评价体现；
- 不加入 GAN、感知损失、循环一致性或分类标签监督。

## 21. RGB 重建的定位

第一版不使用：

```text
M/H/S/P 预测
→ SO-0
→ RGB 重建
```

作为训练损失。

原因：

- 合成训练已有精确 M/H/S/P 真值；
- RGB 重建约束在很大程度上是重复信息；
- 不同分量组合可能产生近似 RGB；
- RGB 重建好不代表 M/H 恢复正确；
- 重建损失过大会促使 S/P 吸收色基误差。

RGB 重建仅保留为测试阶段的离线物理一致性审计：

```text 
预测 M/H/S/P
→ 反归一化
→ SO-0
→ 重建 RGB
→ 计算 RGB MAE / PSNR / SSIM
```

不得根据 RGB PSNR 单独选择最佳 checkpoint。

## 22. 训练配置

第一版固定一个主配置：

```text
optimizer：AdamW
learning_rate：1e-3
weight_decay：1e-4
batch_size：根据显存实测确定，优先 8–16
max_epochs：25
early_stopping_patience：5
scheduler：cosine decay
AMP：开启
seed：20260801
```

第一版只运行一个随机种子。

最佳 checkpoint 指标：

```text
val_M_MAE + val_H_MAE
```

S/P 指标作为辅助监控，不主导 checkpoint 选择。

## 23. 合成评价指标

M 和 H 分别报告：

- MAE；
- RMSE；
- Pearson correlation；
- Spearman correlation；
- 预测标准差；
- 真值标准差；
- 相对常数均值预测的改进；
- 边界饱和比例。

S/P 报告：

- MAE；
- RMSE；
- 预测分布；
- P 非零区域恢复与全零坍塌检查。

分数据集报告：

- Validation；
- ID Test；
- Camera-OOD；
- Light-OOD；
- Joint-OOD。

## 24. SO-1 必要审计

同一次 SO-1 运行中完成，不拆成独立阶段：

1. M/H 是否坍塌为常数；
2. M/H 是否优于常数均值预测；
3. M/H 在未见相机和光源下是否彻底失效；
4. 固定 M/H、改变 S/P 时，预测 M/H 是否大幅漂移；
5. P 是否坍塌为全零；
6. S/P 是否吸收几乎全部变化；
7. 预测值是否频繁贴近 0 或 1；
8. RGB 离线重建是否数值稳定；
9. 对少量真实 patch 推理是否全部 finite。

建议继续门槛：

```text
ID：
M/H MAE 均较常数基线降低至少 50%
M/H Pearson 均不低于 0.75

Camera-OOD 与 Light-OOD：
M/H 均优于常数基线
M/H Pearson 均不低于 0.50
```

这些门槛用于决定是否进入真实数据，不代表临床有效性。

---

# 第四部分：真实人脸数据预处理

## 25. 当前定位

真实人脸预处理尚未完全确定具体代码路径和资产路径，但必须满足 SO-1 推理接口。

该模块最终输出：

```text
高分辨率对齐 RGB 人脸
同尺寸全脸有效皮肤 mask
颜色空间与数值范围说明
sample_id 与文件路径索引
```

## 26. 输入资产要求

每例至少需要：

```text
sample_id
patient_group_id
binary_label
fold
高分辨率原始或对齐人脸路径
全脸皮肤 mask 路径
相机/EXIF 信息（如有）
```

优先复用：

- 既有 alignment 参数；
- 高分辨率对齐人脸；
- `skin_strict` 或等价全脸皮肤 mask；
- `physics_core_skin` 作为严格区域统计 mask；
- 固定五折。

## 27. 全脸皮肤 mask 要求

主推理 mask 应覆盖完整有效皮肤，同时排除：

- 背景；
- 头发；
- 眉毛；
- 眼球；
- 鼻孔；
- 嘴唇；
- 牙齿；
- 明显非皮肤区域。

`physics_core_skin` 不应作为唯一全脸推理 mask，因为它可能只保留更严格的核心区域。它更适合：

- QC；
- 面颊/额头统计；
- 高置信区域敏感性分析。

## 28. 颜色处理接口

真实输入统一执行：

```text
读取图像
→ 转换为 RGB 通道顺序
→ 按 sRGB 编码图像处理
→ float32 / [0,1]
→ inverse sRGB
→ linear-RGB
```

若图像无 ICC 或颜色标记，记录：

```text
color_space_assumption = assumed_sRGB
```

不得针对不同病例人工选择不同 Gamma、白平衡或颜色增强。

允许按病例变化的只有：

- 原始宽高；
- patch 数量；
- patch 坐标；
- 有效皮肤比例。

所有病例必须使用同一颜色转换函数。

## 29. 预处理阶段尚待确定的工程信息

生成最终 Codex 实现提示词前，需要明确：

1. 高分辨率对齐人脸目录；
2. 图像尺寸和格式；
3. 全脸皮肤 mask 目录；
4. mask 尺寸、格式和取值；
5. 500 例主索引；
6. 图像、mask、fold 和标签的对应规则；
7. 现有 224×224 RGB 分类基线图路径。

这些信息属于工程接口，不改变研究主线。

---

# 第五部分：Patch 划分器、推理与合并器

## 30. Patch 划分器作用

Patch 划分器只用于真实高分辨率人脸推理，不参与 SO-1 合成训练的数据组织。

输入：

```text
高分辨率 linear-RGB 人脸
同尺寸全脸皮肤 mask
```

输出：

```text
一组 256×256 RGB patch
对应 mask patch
每个 patch 的原图坐标
```

## 31. 推理 Patch 参数

第一版借鉴参考论文并固定：

```text
patch_size = 256
overlap = 15
nominal_stride = 241
valid_skin_fraction >= 0.20
```

不强制所有病例切成固定 4×6 网格。4×6 是参考论文针对其固定 VISIA 图像尺寸的具体实现。

本项目采用自适应规则网格：

```text
横向起点：0, 241, 482, ...
纵向起点：0, 241, 482, ...
```

最后必须追加：

```text
W - 256
H - 256
```

以保证右边界和下边界完整覆盖。

## 32. Patch 输入处理

每个 patch：

1. 计算有效皮肤比例；
2. 低于 0.20 时跳过；
3. 有效皮肤 RGB 保留；
4. 非皮肤 RGB 设置为 0；
5. 输入必须为 linear-RGB；
6. 通道顺序固定为 R/G/B；
7. 不进行每 patch 单独白平衡或直方图均衡。

SO-1 输出后再次应用 mask：

```text
M_patch *= skin_mask_patch
H_patch *= skin_mask_patch
S_patch *= skin_mask_patch
P_patch *= skin_mask_patch
```

## 33. Patch 合并器

对每例建立：

```text
sum_M, sum_H, sum_S, sum_P
sum_weight
```

每个 patch 根据原始坐标写回完整画布。

重叠区域使用线性渐变或等价平滑窗：

```text
左 patch 重叠边缘：1 → 0
右 patch 重叠边缘：0 → 1
```

二维权重：

```text
w(x,y) = w_x(x) × w_y(y)
```

图像最外侧边界没有相邻 patch 时，外边缘权重必须保持 1。

加权合并：

```text
sum_M += w × M_patch × mask_patch
sum_weight += w × mask_patch

M_full = sum_M / (sum_weight + epsilon)
```

H/S/P 同理。

最后再次应用完整全脸 mask。

## 34. 完整分布图输出

每例保存高分辨率：

```text
M_full.npy
H_full.npy
S_full.npy
P_full.npy
skin_mask.npy
```

数据类型：

```text
float32
```

同时保存预览：

```text
M_preview.png
H_preview.png
S_preview.png
P_preview.png
overlay_panel.png
```

PNG 仅用于人工观察，后续统计与分类必须从浮点数组读取。

## 35. Patch 推理质量检查

检查：

- 是否全部 finite；
- patch 覆盖率；
- 有效皮肤遗漏率；
- 重叠区域接缝；
- 边缘异常；
- M/H 是否出现局部棋盘格；
- 不同 patch 数量病例的结果是否稳定；
- mask 边界是否产生大面积异常值。

---

# 第六部分：分类数据构建与分类模型

## 36. 分类分辨率

用户既有全脸二分类实验显示 224×224 优于 256×256，因此第一版分类统一使用：

```text
224 × 224
```

完整高分辨率分布图只在全部 patch 合并完成后统一 resize 一次。

## 37. 分类数据构建

生成：

```text
RGB_224     [3,224,224]
M_224       [1,224,224]
H_224       [1,224,224]
S_224       [1,224,224]
P_224       [1,224,224]
mask_224    [1,224,224]
```

插值规则：

- RGB：沿用现有最佳 224 RGB 基线；
- M/H/S/P：双线性插值；
- mask：最近邻插值；
- resize 后再次执行 `map *= mask_224`；
- 不从预览 PNG 读取分布图。

RGB 分类分支继续沿用既有最佳 RGB 预处理和 ImageNet 归一化，不因 SO-1 输入使用 linear-RGB 而修改现有 RGB 分类基线。

## 38. 分类实验矩阵

### 38.1 SO-2 fold 0 快速筛选

第一轮只运行：

| 实验 | 输入 | 目的 |
|---|---|---|
| RGB | 3 通道 RGB | 既有基线 |
| M+H | 2 通道 | M/H 是否有独立分类信号 |
| RGB+M+H | 5 通道 | M/H 是否提供 RGB 之外的增量 |
| S+P probe | 2 通道 | 检查采集条件捷径 |
| Mask-only probe | 1 通道 | 检查 mask 形状与姿态泄漏 |

第一版融合方式固定为输入级通道融合：

```text
RGB+M+H = [R,G,B,M,H]
```

理由：

- 参数增加最少；
- 与现有 ResNet18 基线兼容；
- 500 例数据下比双编码器更不易过拟合；
- 适合快速判断是否存在增量。

不在 SO-2 中实现双分支特征融合。

### 38.2 SO-3 正式五折

SO-2 出现正向信号后，正式比较：

```text
RGB
M
H
M+H
RGB+H
RGB+M+H
S+P probe
Mask-only probe
```

只有早期融合显示稳定增量后，才允许将特征级双分支融合作为可选优化，而不是主线前置条件。

## 39. 分类网络

第一版使用既有二分类 ResNet18 训练协议。

输入通道修改：

```text
RGB：3 通道
M：1 通道
H：1 通道
M+H：2 通道
RGB+M+H：5 通道
S+P：2 通道
Mask-only：1 通道
```

除第一层卷积外，网络、训练轮数、优化器、类别权重、early stopping、评价脚本和数据划分必须一致。

M/H/S/P 分支不得使用会改变物理数值含义的 ColorJitter。

允许的增强仅包括同步空间增强，例如：

- 水平翻转；
- 轻度裁剪；
- 轻度缩放。

RGB+M+H 的全部通道必须执行完全相同的空间变换。

## 40. 分类评价指标

统一报告：

- Macro-AUC；
- Accuracy；
- Macro-Precision；
- Macro-Recall；
- Macro-F1；
- Balanced Accuracy；
- Patient sensitivity；
- Control specificity。

正式五折增加：

- OOF 预测；
- patient-cluster bootstrap；
- 配对置信区间；
- fold 级稳定性；
- camera/EXIF probe；
- R3DPR 跨光照稳定性。

评价单位固定为 visit/case，不进行 patient-level prediction averaging。

---

# 第七部分：真实数据快速验证

## 41. 真实 Pilot 40

SO-1 合成评价通过后，先选择 40 例真实人脸：

```text
Control：20
Patient：20
```

尽量覆盖：

- 五个 fold；
- 主要相机型号；
- 高、中、低亮度；
- 不同肤色；
- 不同图像质量；
- 不同拍摄角度和光照方向。

名单必须在推理前冻结，不因输出质量差而替换病例。

## 42. Pilot 40 输出

每例生成：

- 原始/对齐 RGB；
- 全脸皮肤 mask；
- M/H/S/P 高分辨率图；
- M/H/S/P 预览；
- patch 网格和覆盖图；
- 输出直方图；
- patch 接缝检查图。

统计：

- finite 比例；
- M/H 均值与标准差；
- 边界饱和比例；
- 有效皮肤覆盖率；
- P 稀疏比例；
- patch seam 指标；
- clipping 与颜色越界统计。

## 43. R3DPR 稳定性审计

对 Pilot 40 的原图和既有六种 R3DPR 重光照图分别推理：

```text
M 跨光照 MAE / correlation
H 跨光照 MAE / correlation
S/P 跨光照变化
```

期望：

```text
M/H 的跨光照变化小于 S/P
```

第一版 R3DPR 只用于审计，不参与 SO-1 训练。

---

# 第八部分：数据划分与泄漏控制

## 44. 合成预训练

SO-1 合成预训练不使用患者数据或 NYHA 标签，可以在真实五折之外完成。

## 45. 真实推理

冻结 SO-1 对全部 500 例进行推理不会使用标签更新权重，因此可以生成全量 M/H/S/P 资产。

分解过程中不得读取：

- NYHA 标签；
- 分类预测；
- 测试折表现。

## 46. 真实适配限制

第一版不进行真实图无监督适配。

未来若必须使用真实图更新 SO-1：

- 每折只能使用该折训练病例；
- 验证和测试病例只能推理；
- 同一 patient group 必须完全位于同一 fold；
- 禁止先用全部 500 例适配，再对同一 500 例做五折分类。

## 47. 固定五折

继续使用项目既有 patient-group 五折：

- 同一患者不同 visit 同折；
- 500 例完整队列；
- 不删除难例；
- 不重新分折；
- 不根据测试结果选择病例；
- 不进行 patient-level 预测平均。

---

# 第九部分：阶段判断与停止规则

## 48. SO-1 进入真实数据的条件

至少满足：

1. M/H 无常数坍塌；
2. ID 中 M/H 显著优于常数预测；
3. Camera-OOD 和 Light-OOD 未完全失效；
4. 固定 M/H、改变 S/P 时，预测 M/H 不发生严重漂移；
5. 真实 patch 推理无大范围 NaN、全零或边界值；
6. P 不坍塌为全零；
7. RGB 离线重建审计数值稳定。

## 49. SO-2 进入完整五折的建议门槛

同时观察：

```text
RGB+M+H 相对 RGB：
Macro-AUC 提升约 ≥ 0.01

Macro-F1 或 Balanced Accuracy：
至少一项提升约 ≥ 0.01

M+H：
Macro-AUC > 0.60

S+P：
不得明显高于 M+H
```

这些仅是 fold 0 的路线筛选门槛，不是论文中的临床显著性标准。

## 50. 停止条件

以下情况应停止或降级路线：

- 合成数据上 M/H 无法恢复；
- 真实 M/H 大量坍塌、饱和或全零；
- M/H 主要随亮度、曝光或相机型号变化；
- S+P 明显比 M+H 更能分类；
- RGB+M+H 没有任何正向趋势；
- M/H 在 R3DPR 重光照下剧烈漂移；
- 增益只出现在单一折；
- 必须修改 SO-0 物理公式才能获得分类提升；
- 必须删除难例、重分折或选择性报告结果。

允许的负结论：

> 当前简化皮肤光学先验能够支持合成空间中的色基敏感分解，但未能在该真实普通手机 JPEG 数据集上提供稳定的心功能状态分类增量。

---

# 第十部分：工程实现与产物规范

## 51. 建议代码模块

```text
src/skin_optics_so1/
├── synthetic_fields.py
├── synthetic_dataset.py
├── camera_light_split.py
├── color_input.py
├── masked_losses.py
├── unet_decomposer.py
├── train_so1.py
├── evaluate_synthetic.py
├── real_face_preprocess.py
├── patch_divider.py
├── patch_combiner.py
├── infer_real_faces.py
├── build_classification_assets.py
└── qc.py
```

SO-0 继续从既有模块导入，不复制或修改冻结代码。

## 52. 建议输出目录

```text
outputs/SO1_Minimal_Synthetic2Real_ClosedLoop_v1/
├── config/
│   ├── synthetic_generation.yaml
│   ├── camera_light_split.json
│   ├── so1_training.yaml
│   ├── real_preprocessing.yaml
│   ├── patch_inference.yaml
│   └── classification_screen.yaml
├── manifests/
├── synthetic_data/
├── checkpoints/
│   └── best.pt
├── synthetic_evaluation/
├── real_pilot40/
├── full500_maps/
├── classification_224/
├── fold0_screening/
└── SO1_closed_loop_report.md
```

## 53. 必须保留的哈希与追踪信息

每个正式输出必须记录：

- git commit；
- SO-0 版本；
- SO-0 冻结资产哈希；
- SO-1 checkpoint 哈希；
- 配置文件哈希；
- camera-light split；
- synthetic manifest；
- real sample index；
- patch 参数；
- 分类 fold；
- 随机种子。

---

# 第十一部分：当前已固定与尚待确定事项

## 54. 已固定事项

```text
SO-0：
SO0_Forward_Model_v1.1，保持冻结

合成可观察图：
256×256 standard sRGB patch

SO-1实际输入：
inverse sRGB 后的 linear-RGB patch

SO-1输出：
M/H/S/P 四通道分布图

网络：
标准 U-Net
64→128→256→512→1024
随机初始化

损失：
四分量归一化后等权 masked SmoothL1

RGB重建：
仅作为离线审计，不作为 v1 训练损失

合成规模：
20,000 train
2,000 validation
5,000 test/OOD
总计 27,000

真实推理：
全脸有效皮肤
256×256 patch
overlap=15
自适应网格
加权合并

分类尺寸：
224×224

快速分类：
RGB
M+H
RGB+M+H
S+P probe
Mask-only probe
```

## 55. 尚待工程确认事项

```text
高分辨率对齐人脸路径
全脸皮肤 mask 路径
实际图像尺寸
mask 取值与命名规则
500 例统一主索引
现有 224 RGB 基线代码与结果路径
R3DPR 六重光照路径
SO-1 代码与输出根目录
```

这些事项只影响工程接入，不允许改变本文件固定的物理输入、网络目标和数据协议。

---

# 第十二部分：最终端到端流程

```text
随机生成 M/H/S/P 空间场
        ↓
冻结 SO-0 相机成像路径
        ↓
生成 256×256 synthetic sRGB patch
        ↓
inverse sRGB
        ↓
256×256 linear-RGB patch
        ↓
标准 U-Net 直接监督预训练
        ↓
冻结 SO-1
        ↓
真实高分辨率人脸统一颜色处理
        ↓
全脸皮肤 mask
        ↓
256×256 重叠 Patch 切分
        ↓
SO-1 逐 Patch 输出 M/H/S/P
        ↓
按原坐标渐变加权合并
        ↓
完整高分辨率全脸 M/H/S/P 图
        ↓
统一 resize 到 224×224
        ↓
RGB、M+H、RGB+M+H、S+P、Mask-only 分类
        ↓
fold 0 快速判断
        ↓
有正向信号才进入完整固定五折
```

---

# 第十三部分：执行纪律

每个阶段原则上只保留：

```text
一个主设计
一个 Codex 提示词
一次主要运行
一个结论
```

不得在 SO-1 第一版同时进行：

- 多网络架构搜索；
- Carvana 与随机初始化对比；
- 多种损失权重搜索；
- RGB 重建损失搜索；
- 真实数据自监督适配；
- R3DPR 一致性训练；
- 多分辨率分类搜索；
- 多种复杂融合；
- Transformer 或生成模型替换。

这些内容只有在最小闭环出现可信正向信号后，才允许作为后续单独问题研究。

---

## 参考依据

1. Jung G, Kim S, Lee J, Yoo S. *Deep learning-based optical approach for skin analysis of melanin and hemoglobin distribution*. Journal of Biomedical Optics. 2023;28(3):035001.
2. `Face_Cardiac_Skin_Optics_Research_Roadmap_v2_Lean.md`.
3. `SO0_Forward_Model_v1.1`.
4. `SO0_reacceptance_report.md`.
5. 项目既有固定 patient-group 五折、RGB 二分类基线、皮肤 mask 与 R3DPR 资产。
6. 