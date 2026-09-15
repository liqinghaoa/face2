# 阶段一：真实 HSI 皮肤物理验证运行手册

> 文档状态：阶段一表示层验证已完成，正式 Test 结论为 `PASS`。  
> 当前数据：Hyper-Skin `(RGB, VIS)` 31 波段发布版已下载并完成基础完整性检查。  
> 当前代码：S1-7 v1 已完成唯一正式 Test，二维 `D2-MH` 表示已授权进入阶段二。  
> 更新日期：2026-09-09。  
> 上位设计：`face2_research/02_Research_Ideas/阶段实现.md` 的阶段一章节。

## 1. 阶段一究竟要回答什么

阶段一只研究真实 HSI，不训练 RGB 编码器，也不使用 500 例心功能数据。

核心问题是：

> 在皮肤区域、反射率合同、波长合同和成像质量均受到控制的条件下，是否存在某个维度适当、结构受约束且可泛化的 `theta_repr`，使候选前向模型能够重构未见受试者的 Hyper-Skin VIS 光谱？

主结论的目标部署域现限定为：正脸、无表情；真实推理还要求无眼镜，但 Hyper-Skin 没有 eyewear 标注且当前 Validation 主域样本均可见眼镜，因此本数据集不能验证无眼镜条件。笑脸和侧脸继续保留为非阻断压力测试，用于报告外推失败和适用范围，不再决定主域 PASS。该范围是在观察初次广泛 Validation 后收窄，属于事后范围修订，不能声称为预先注册的独立验证。

阶段一不预设 `theta_repr` 的维度，也不预设只有 M/H 能够重构 HSI。M-like、Hb-like、散射、宽带观测项以及在观测域允许时的 Shading/Specular，均作为候选分量进入模型比较。阶段一只冻结一个表示及其前向模型，不冻结任何坐标的生理身份。

M/H 相关坐标仍可作为有文献依据的候选基底，但在本阶段统一称为 `M_like`、`Hb_like` 或 `*_proxy`。它们是模型坐标，不是绝对黑色素、血红蛋白或氧合度真值。加入倾斜、散射或其他分量后坐标重新分配，属于阶段二解释层的待验证现象，不是阶段一表示层的阻断条件。

阶段一和阶段二必须明确分开两类结论：

- **表示层结论（阶段一）**：一个 1–3 维的预注册候选，或有充分观测依据的条件性 4 维候选，能否在未见受试者上重构目标域 HSI，并优于匹配的零维/同维数据驱动基线；
- **解释层结论（阶段二）**：冻结的坐标在 RGB–HSI 自监督闭环中呈现什么行为，能否通过分量消融、外部相关或额外实验支持 M-like、Hb-like、Shading、Specular 等解释。

因此，阶段一可以选出 D2、D3 或数据驱动表示，也可以判定某个半机制公式失败；它不能仅凭 HSI 重构损失宣称坐标等于真实生理量。Jung 式多分量闭环是阶段二的设计参照，而不是要求阶段一预先证明只有 M/H 两个参数。

## 2. 当前状态与能否开工

| 项目 | 当前状态 | 对实施的含义 |
|---|---|---|
| RGB–VIS 数据 | 已有 306 对，全部可读 | 可以开始数据合同、配准和 mask 工作 |
| 受试者划分 | Train/Validation/Test 为 44/3/4 人，无受试者泄漏 | 可以执行受试者级冻结流程 |
| HSI 数值完整性 | 31 波段、无 NaN/Inf，只有极少浮点边界越界 | 派生数据可裁剪到有效范围，原始数据保持只读 |
| 物理量 | 文档支持“归一化光谱反射率” | 可以进行反射率模型探索，但必须保留校正细节缺口 |
| 波长 | 作者官方代码已确认 31 个标称中心为 `400:10:700 nm`，通道升序 | S1-3/S1-4 直接读取冻结合同，不再把中心数组视为推定 |
| 有效带宽/SRF | 未发布 | 需要波长/FWHM 敏感性分析，不能虚构带宽 |
| S1-0 数据合同 | 已全量扫描 306 对并 PASS | 合同、证据说明和 SHA-256 清单已生成 |
| RGB–HSI 配准 | 18 个 Train 分层样本均选择 `transpose`，人工复核通过 | `transpose` 已冻结，允许进入 S1-2 |
| 皮肤 mask 与 ROI | `PASS_FOR_DEVELOPMENT`；目标部署域 r5 已人工确认 | 已授权 S1-3；不构成无眼镜域的独立验证 |
| 区域光谱与波段 QC | `PASS_FOR_S1_4`；282 个 Train/Validation 样本已缓存 | 目标域双侧脸颊 94/94 可用；红端高曲率和固定侧别照明效应进入 S1-4/S1-5 敏感性分析 |
| 候选模型 | S1-4R v3 已实现，r7 合成审计为 `PASS_FOR_REVISED_TRAIN_DEVELOPMENT` | 修订公式数值实现通过；不等同于真实数据有效 |
| Train 反演与诊断 | 旧 S1-5 v2、S1-5R v7 的解释层 decision 保留；表示层重判 v3 为 `REPRESENTATION_READY_FOR_S1_6` | D2/D3 具有 Train 表示增益；K2 覆盖性/边界失败；坐标语义不在阶段一判定 |
| Validation 选择与冻结 | S1-6 v1 为 `S1_6_FROZEN_FOR_S1_7` | D2-MH 在 10% 最佳误差容差内以更低维胜出；冻结为二维 `theta_repr`，不赋予 M/H 生理含义 |
| 门控阈值 | 模型选择门由 S1-6 冻结；最终 Test 门由 S1-7 配置在锁前固化 | S1-7 只执行冻结规则，禁止按 Test 回调 |
| 正式 Test | S1-7 v1 为 `STAGE1_COMPLETE / PASS` | 4 名受试者、8/8 主脸颊可用；D2-MH 在 4/4 人上优于 B0-S，允许进入阶段二 |

结论：`S1-0` 至 `S1-7` 已完成。S1-7 已按冻结合同唯一运行一次，正式 decision 为 `PASS`、`next_stage_allowed=true`、`authorized_next_stage=S2`。这里的两个坐标只是冻结的二维 `theta_repr`；D2/D3 坐标混合、K2 失败、有效 SRF 缺失和无眼镜域缺失仍作为限制保留。Hyper-Skin Test 已使用，之后任何修改都只能标为修订性分析，不能再声称首次独立 Test。

## 3. 数据与目录约定

### 3.1 原始数据只读

当前原始数据根目录固定为：

```text
E:/projects/face2/data/raw/Hyper-Skin(RGB, VIS)/
├── train/
│   ├── RGB/*.jpg       # 264
│   └── VIS/*.mat       # 264
├── valid/
│   ├── RGB/*.jpg       # 18
│   └── VIS/*.mat       # 18
└── test/
    ├── RGB/*.jpg       # 24
    └── VIS/*.mat       # 24
```

原始文件不得改名、覆盖、转存为裁剪后的“修正版”或写入 mask。所有方向变换、裁剪和数值修正都在派生层完成。

### 3.2 外部光谱资产

第一版血红蛋白/组织光学资产为只读输入。修订主模型使用与 31 个发布中心对应的 10 nm 派生资产，1 nm 资产只用于 SRF/FWHM 敏感性：

```text
data/external/SO0_Spectral_Assets_v1/
  processed/SO0_Spectral_Standardized_v1/
  ├── production_10nm/derived_optics_10nm.npz
  └── reference_1nm/derived_optics_1nm.npz
```

使用前记录文件 SHA256、波长范围、单位和重采样方法。不得引用旧 SO-0 模型输出作为 Hyper-Skin 的 M/H 真值。

### 3.3 阶段一派生数据

建议固定派生数据根：

```text
data/processed/HyperSkin_Stage1_v1/
├── contracts/
├── manifests/
├── registration_qc/
├── masks/
├── region_spectra/
└── freeze/
```

建议固定实验输出根：

```text
outputs/skin_optics_hsi_v1/stage1_hsi_physics/<run_id>/
```

每次运行使用新的 `run_id`，禁止覆盖已有非空输出目录。

## 4. 数据合同

### 4.1 已能从论文与补充材料确认的内容

- RAW HSI 经暗参考校正和白参考归一化，被作者描述为 spectral reflectance；
- 原始相机是 Specim FX10，覆盖 400–1000 nm、448 波段；
- 原始光谱采样约 1.34 nm，原始系统 FWHM 约 5.5 nm；
- 发布的 VIS 目标为 400–700 nm、31 波段；
- 448 波段通过 SciPy interpolation 重采样为 31 波段；
- 采集条件包括双侧卤素灯、40 cm 工作距离、45 Hz、22 ms、光谱/空间 binning=1、推扫成像和下巴托。

### 4.2 官方代码新增确认与尚未完全确认的内容

- 作者官方仓库提交 `380ff1f97a81aefc074e8ddd8e44f3c7e104afe9` 的 `evaluations/vis_evaluation_mstpp_retrained.ipynb` 明确使用 `band_31 = np.arange(400, 710, 10)`；因此发布 VIS 的 31 个标称中心及其升序通道映射确认为 `400, 410, ..., 700 nm`；
- 未提供重采样使用的原始 wavelength vector、具体 SciPy 函数和端点处理；
- 5.5 nm 是原始 FX10 FWHM，不能直接当作 31 波段有效带宽；
- 未提供 31 波段 SRF、光源 SPD、精确照明角度和功率；
- 暗/白参考公式的完整实现、参考板绝对反射率及逐次/逐日校正策略未完全说明；
- 未发布定量噪声、坏波段和饱和阈值；
- 论文使用的人体/背景 mask 不等于皮肤有效 mask，当前下载数据中也没有对应 mask 文件。

### 4.3 合同文件必须包含什么

`data_contract.json` 至少包含：

- 原始数据根和数据集版本；
- Train/Validation/Test 文件数量与受试者数量；
- `sample_id`、`subject_id`、`expression`、`direction`、RGB/VIS 路径；
- HDF5 键、原始数组形状、目标数组形状和转置规则；
- 物理量名称、合理数值范围及裁剪规则；
- 中心波长数组、波段数量、波段顺序；
- 原始 FWHM、31 波段有效带宽状态；
- 每个字段的 `confirmed / inferred / missing` 状态；
- 证据文件、页码或代码来源；
- 所有输入文件或清单的哈希。

`contract_evidence.md` 用自然语言解释每项证据及其限制。不得把 `inferred` 字段在报告中写成 `confirmed`。

### 4.4 数据合同硬门

进入正式 Test 前必须确认：

1. `cube` 的物理量可按反射率处理；
2. 31 波段顺序和中心波长可追溯；
3. HSI 空间轴和 RGB 对齐规则已冻结；
4. Test 受试者没有参与任何阈值、mask 或模型选择；
5. 有效带宽缺失已经通过敏感性分析和结论限制进行管理。

物理量或波长顺序无法确认时，阶段一直接 STOP。

## 5. 参数和候选模型

### 5.1 S1-5 v2 对旧体系的否定范围

旧体系的单项公式并非都存在代数错误：黑色素幂律、Jonasson/Jacques 散射式、HbO2/Hb 线性混合及半无限 Kubelka–Munk 在各自假设下均可成立；cm⁻¹ 到 mm⁻¹ 的换算也正确。失败发生在整体观测模型：旧式把表皮当作无散射的固定双程 Beer–Lambert 滤光片，把真皮当作均匀半无限介质，把未作血管包装修正的 Hb 谱直接加到基线吸收，并把该理想漫反射直接等同于 Teflon 白参考下的宽场相机反射率。

这与证据来源存在不可忽略的域差异。Jonasson 使用 475–850 nm、0.4/1.2 mm 双源探距离、前臂三层皮肤和 inverse Monte Carlo；Hyper-Skin 使用 400–700 nm、40 cm 推扫相机、卤素宽场照明和相对白参考。Jonasson 的参数只能作为量级/谱形先验，不能直接证明 Hyper-Skin 的绝对成像反射率公式。S1-5 v2 中物理模型在 450–700 nm 大范围高估反射率、在 400–430 nm 又低估，H 被推向上界、`S_amp` 被推向下界，符合结构失配而非单纯优化失败。

因此旧 `B1/P2/P3-S/P3-O/P4` 全部保留为 `legacy_rejected_s1_5_v2`，只用于回归对照，不得更名后继续提交 Validation，也不得仅通过放宽参数边界重新激活。

### 5.2 修订后的证据层级和参数语义

阶段一只判定表示层，不把坐标的生理身份作为前置条件。候选输出统一记为 `theta_repr`；`proxy`、`semi_mechanistic` 和 `mechanistic` 表示前向模型约束强弱，而不是表示某个参数已经被生理真值验证：

| 层级 | 阶段一允许的结论 | 阶段二处理 |
|---|---|---|
| `data_driven` | 低维数据坐标能否重构 HSI | 作为非物理表示对照或候选 |
| `proxy` | 固定物理启发谱形约束下的低维光学表示能否重构 HSI | 以 `*_like`/`*_proxy` 名称传递，解释待验证 |
| `semi_mechanistic` | 带吸收、散射和包装近似的表示能否重构原始反射率 | 仅在 Validation 通过后作为受约束解码器 |
| `mechanistic` | 与采集几何匹配且有独立校准的组织模型 | 当前观测合同不足，暂不启用 |

Hyper-Skin 未公开足以复建相机观测算子的有效 SRF、照明/探测角分布、表面 Fresnel 边界和空间采样对应的光子路径分布，因此 K–M 路径只能标记为 `semi_mechanistic`，不能标记为严格 `mechanistic`。即使 `proxy` 或 `semi_mechanistic` 通过，也只证明表示层可行，不证明 M/H 或其他坐标的生理含义。

修订参数层为：

```text
theta_repr      = [z_1, ..., z_d]  # 维度由候选比较决定
theta_aux       = [a_obs, q_tilt?, delta_sO2?, shading?, specular?]
theta_global    = [R_ref_train(lambda), band_weights,
                   sO2_ref, vessel_packaging_spec,
                   scattering_reference_spec, KM_mapping_spec,
                   g_system?, side_gain?]
```

- `delta_M_OD`、`delta_Hb_OD` 是相对 Train 参考谱的有符号、无量纲差分光密度系数；正值表示相应吸收敏感谱形相对参考增强，负值表示减弱。它们不是黑色素/血红蛋白浓度，也不是旧 P2 参数的延续。
- `M_epi_OD` 是无量纲表皮黑色素路径吸收代理；`f_blood_proxy` 是用于生成真皮吸收系数的无量纲血液组织分数代理。两者与 proxy 层参数属于不同合同，不能互换数值或共享边界。
- `a_obs` 是整条 log 光谱的解析幅度偏移，只承接亮度、相对白参考和观察几何的波长无关部分；是否传递到阶段二由表示候选和观测合同决定。
- `q_tilt` 是平滑宽带谱倾斜坐标，不命名为 `S_amp`，因为在没有绝对观测几何时不能把该系数等同为约化散射系数。
- `delta_sO2` 只在残差诊断显示其能够改善表示时才作为条件候选，不因名称而预先赋予氧合含义。
- 表皮厚度不与 M 同时逐区域自由反演；血管直径、散射斜率、Rayleigh 比例和系统尺度只能作为有来源的固定量、离散敏感性条件或 Train-only 全局量。

### 5.3 共同观测变换：把尺度与谱形显式分开

对每条区域反射率 $R_i(\lambda)>0$，定义加权中心算子：

$$
\mathcal C_w[x](\lambda)
=x(\lambda)-\frac{\sum_{\lambda\in\Lambda}w_\lambda x(\lambda)}
{\sum_{\lambda\in\Lambda}w_\lambda}.
$$

Train 参考谱 `R_ref_train` 由正脸无表情双侧脸颊构成。主定义采用受试者平衡的几何均值：先在每名受试者内部平均有效左右脸颊的 log 光谱，再在受试者之间等权平均，避免一个人因可用 ROI 更多而获得更高权重：

$$
\log R_{ref}(\lambda)=\frac{1}{N}\sum_{s=1}^{N}
\frac{1}{n_s}\sum_{r=1}^{n_s}\log R_{s,r}(\lambda).
$$

受试者级 log 中位参考只作鲁棒性敏感性，不可根据拟合优劣事后替换主定义。Train 内评价必须按受试者留一生成参考、波段权重和 PCA 基底，未来 Validation/Test 只能使用完整 Train 冻结量。差分 log 反射率和幅度 nuisance 定义为：

$$
d_i(\lambda)=\log R_i(\lambda)-\log R_{ref}(\lambda),
\qquad
a_{obs,i}=\frac{\sum_\lambda w_\lambda d_i(\lambda)}{\sum_\lambda w_\lambda},
\qquad
y_i(\lambda)=\mathcal C_w[d_i](\lambda).
$$

`y_i` 是主 proxy 拟合目标，`a_obs` 单独保存并作为质量/观察 nuisance。该处理等价于在 log 域解析剖面化一个波长无关乘法尺度，不能被扩展成逐波段自由校正。所有模型和基线必须使用同一变换、同一波段和同一权重，防止只给物理模型额外自由度。

如果后续需要从 proxy 重建完整光谱，则使用：

$$
\log \hat R_i(\lambda)=\log R_{ref}(\lambda)+a_{obs,i}+\hat y_i(\lambda).
$$

阶段二可为重建任务预测 `a_obs`、低频 Shading 或 Specular 等辅助坐标；阶段三的主光学表征使用阶段一冻结的 `theta_repr`，不能因为当前某个坐标名称为 M/H 就提前当作生理特征。任何坐标子集的分类增益都必须单独报告并标为探索性关联。

### 5.4 固定的物理谱形基底

黑色素基底继续使用 Jonasson 形式，但在比较前加权中心化：

$$
\phi_M(\lambda)=\left(\frac{\lambda}{570}\right)^{-\beta_{mel}},
\qquad \tilde\phi_M=\mathcal C_w[\phi_M].
$$

`beta_mel=4.3` 是主设置；2.9 和 5.8 只作为有来源的 Train 敏感性条件，不逐光谱自由拟合。

Hb 基底不得再直接使用未包装的 570 nm 归一化消光谱。主实现读取与发布波长一一对应的 `data/external/SO0_Spectral_Assets_v1/processed/SO0_Spectral_Standardized_v1/production_10nm/derived_optics_10nm.npz`，只取 400–700 nm 的 31 点；该文件 SHA-256 为 `30de54c4bd5af2453a32490559c96e36854b86b31f65336ab700cf32de0488f6`。其中使用 `mua_hbo2_whole_blood_150gL_cm1` 和 `mua_hb_whole_blood_150gL_cm1`，冻结换算为 `mu_a = ln(10) * epsilon * (150/64500)`。实现只做 cm⁻¹ 到 mm⁻¹ 的除 10 换算，不重复执行摩尔消光到全血吸收的换算。

由于 Hyper-Skin 发布版有效 SRF 仍缺失，主条件使用 10 nm 点采样资产；敏感性条件从 `reference_1nm/derived_optics_1nm.npz`（SHA-256 `6dc1350517a45f5e13a06a025be59921e0c2f398d022b93bf2fd6bf2293feef9`）先按假定 Gaussian FWHM 卷积后再采样到 31 个中心。点采样、5.5 nm FWHM 及已注册 FWHM 邻域必须并列报告，不能把任一假定写成厂商确认的发布版有效 SRF。将两个有单位数组统一转换为 mm⁻¹ 后形成全血吸收：

$$
\mu_{a,blood}(\lambda;s_0)
=s_0\mu_{a,HbO2}(\lambda)+(1-s_0)\mu_{a,Hb}(\lambda),
$$

再应用血管包装算子。若使用等效血管直径 (D_v)，候选形式为：

$$
C_{vp}(\lambda;D_v)=
\frac{1-\exp[-\mu_{a,blood}(\lambda)D_v]}
{\mu_{a,blood}(\lambda)D_v},
$$

$$
\phi_H(\lambda)=
\frac{C_{vp}(\lambda;D_v)\mu_{a,blood}(\lambda;s_0)}
{C_{vp}(570;D_v)\mu_{a,blood}(570;s_0)},
\qquad \tilde\phi_H=\mathcal C_w[\phi_H].
$$

这里用直径写法与原始来源的 `2*r_v` 写法完全等价。公式和固定条件的 provenance 冻结如下：

| 项目 | 冻结来源与设置 | 适用边界 |
|---|---|---|
| 包装公式 | van Veen、Verkruysse 与 Sterenborg，`C_pack=[1-exp(-2*mu_a,blood*r_v)]/(2*mu_a,blood*r_v)`；DOI `10.1364/OL.27.000246`；Rajaram 等给出相同公式并作实验验证，DOI `10.1002/lsm.20933` | 这是等效血管包装修正，不代表真实血管尺寸分布；van Veen 的体内结果为 500–1060 nm，向更短波长使用必须做敏感性 |
| `D_v` 主值 | `15 um = 0.015 mm`；Rajaram 等在正常皮肤 DRS 中估计 `15 ± 2 um` | 不是 Hyper-Skin 个体真值，不得逐谱反演 |
| `D_v` 敏感性 | `[10, 22, 44] um`；10 um 覆盖较小正常皮肤微血管，22/44 um 对应 Rajaram 的两种实验微通道 | 只报告敏感性，不按 Train 最优拟合挑主值，也不带到 Validation 再选择 |
| `sO2_ref` | 主值 0.50；敏感性 0.41/0.60 | 来自 Jonasson 2023 的中位数和 IQR，仅作参考状态 |
| 表皮厚度参考 | 0.063 mm；敏感性 0.013/0.191 mm | 来自 Jonasson 2023 的中位数和 IQR，只影响半机制固定背景项；不与 M 同时自由拟合 |

数值实现必须使用 `-expm1(-x)/x` 并在 `x=mu_a,blood*D_v -> 0` 时显式返回 1，避免小量相消；当 `x` 增大时，`C_vp` 单调下降，而 `C_vp*mu_a,blood` 有限趋近 `1/D_v`。Rajaram 的微通道实验直接覆盖 450–700 nm，并在皮肤数据中讨论约 420 nm 的 Soret 带，但 400–440 nm 对当前相机和人脸场景仍是高敏感区；因此 S1-5R 必须同时报告全 31 波段、去 400 nm、去 400–440 nm 与假定 SRF 条件，不能通过只保留最有利条件决定 PASS。

宽带倾斜基底定义为：

$$
\phi_G(\lambda)=\log(\lambda/600),
\qquad \tilde\phi_G=\mathcal C_w[\phi_G].
$$

它只表示一个平滑的观察/散射混合方向。阶段一可以把它作为 `theta_repr` 候选坐标，而不必先证明它与 M/H 生理因素正交；不得通过 Gram–Schmidt 事后制造坐标语义。应报告固定基底的加权相关、夹角、奇异值、条件数和联合表示稳定性。

设计阶段已对 31 波段、`sO2_ref=0.50`、`D_v=15 um` 做无权重预审计。单位范数后的 `corr(M,H)=0.893`、二维设计矩阵条件数约 4.20，说明这是一组可尝试的二维候选基底；加入 `q_tilt` 后 `corr(M,tilt)=-0.953`、三维条件数约 8.70，提示坐标之间存在高度相关。该结果用于比较二维/三维表示的复杂度和重构能力，不再预先否定 D3，也不把任一坐标写成已确认的生理因素。正式数值仍须用每个 LOO 折内冻结的实际权重重算这些量。

### 5.5 修订候选模型阶梯

所有 Train 指标按受试者留一计算。`B0-S`、`B2-PCA` 和 D 系列使用相同的 `a_obs` 剖面化；K 系列的原始尺度主拟合只允许 Train-only 全局 `g_system`，不允许逐谱 `a_obs`，但必须额外用同一中心化算子报告 shape 指标。PCA 只能用当前留一折的 Train 受试者拟合。

| ID | 预测空间与自由参数 | 作用 | 地位 |
|---|---|---|---|
| `B0-R` | 原始反射率；完整 Train/留一 Train 平均谱，无逐谱参数 | 保留旧原始尺度基线 | 必须 |
| `B0-S` | 差分 log 谱；只有解析 `a_obs`，`y_hat=0` | 与 proxy 匹配的 nuisance-only 基线 | 必须 |
| `B2-PCA` | 差分 log 谱；2 个 PCA 系数 + `a_obs` | 与二维候选同维度、同损失的数据驱动参照 | 必须，是否传入阶段二由 Validation 决定 |
| `B2-RPCA` | 未中心化的差分 log 谱；2 个 PCA 系数，无独立 `a_obs` | 与半机制模型的两个逐谱自由参数匹配的原始尺度参照 | 必须，但不得传入阶段二 |
| `D1-M` | `y_hat=-delta_M_OD*phi_M_tilde` | 一维吸收样表示候选 | 必须 |
| `D2-MH` | `y_hat=-delta_M_OD*phi_M_tilde-delta_Hb_OD*phi_H_tilde` | 二维物理启发表示候选 | 必须 |
| `D3-MHG` | D2 + `q_tilt*phi_G_tilde` | 三维吸收/观测表示候选 | 必须作为候选比较，不再仅作语义压力测试 |
| `D3-MHO` | D2 + 预定义氧合对比基底 | 条件性更高维表示候选 | 由残差和复杂度规则触发 |
| `K2-MH-KM` | 包装后 Hb + 黑色素表皮光学深度 + 修正的 K–M 系数映射；逐谱 `M_epi_OD/f_blood_proxy`，Train-only 全局系统尺度 | 半机制桥接候选 | 诊断性、非默认 |
| `K3-MHS-KM` | K2-MH-KM + 逐谱 `S_amp` | 检查原始反射模型中的散射混杂 | 仅在 K2-MH-KM 已覆盖基本谱形后触发 |
| `T2-MH-RTE` | 与采集几何匹配的分层 RTE/Monte Carlo | 严格机制候选 | 当前缺采集边界条件，登记为 deferred，不实现、不参与门控 |

`B2-PCA` 对中心化后的 `y_i` 学习两个主成分；`B2-RPCA` 对未逐谱中心化的 `d_i=log(R_i)-log(R_ref)` 学习两个主成分，因此其两个系数可自行分配给幅度和谱形，但没有额外 `a_obs`。两者的均值、基底和投影均须在 LOO 折内完成。

`D2-MH` 的系数允许相对 Train 参考为正或负，开发期使用无硬边界加权线性最小二乘作为二维表示求解；另做弱 ridge 敏感性，但强先验不得作为主结果。D3-MHG 即使与 D2 存在坐标混杂，只要联合表示能够改善未见受试者重构且数值可复现，就可以作为表示候选提交 Validation；其坐标在阶段一统一使用代理/坐标名称，不提前下结论为独立生理因素。

`D3-MHO` 的氧合对比基底必须由同一包装模型在两个预注册 sO2 值间的差异生成。只有残差改善、维度代价可接受且加入后联合表示可复现时，才可激活；是否呈现 Hb/O2 语义留给阶段二，不得把阶段一的坐标名称当作生理结论。

`K2-MH-KM` 不得直接复用旧 P2。其单位闭合、但仍属半机制近似的候选公式为：

背景吸收直接读取同一 10 nm 资产中的 `mua_base_cm1` 并除 10 转为 mm⁻¹；其注册来源公式为

$$
\mu_{a,bg}(\lambda)=0.244+85.3
\exp[-(\lambda-154)/66.2]\quad\mathrm{cm}^{-1}.
$$

实现不得在读取派生数组后再次套用该公式。

$$
\tau_{epi}(\lambda)=t_{epi,global}\mu_{a,bg}(\lambda)
+M_{epi,OD}\phi_M(\lambda),
$$

$$
\mu_{a,dermis}(\lambda)
=(1-f_{blood})\mu_{a,bg}(\lambda)
+f_{blood}C_{vp}(\lambda;D_v)\mu_{a,blood}(\lambda;s_0),
$$

$$
\mu_s'(\lambda)=S_0\left[(1-\gamma)
\left(\frac{\lambda}{600}\right)^{-\beta}
+\gamma\left(\frac{\lambda}{600}\right)^{-4}\right],
$$

旧模型把 `K/S` 直接等同于 `mu_a/mu_s_prime`，该等同不成立。修订候选采用预注册的经典两通量近似映射：

$$
K_{KM}(\lambda)=2\mu_{a,dermis}(\lambda),
\qquad
S_{KM}(\lambda)=\frac{3}{4}\mu_s'(\lambda)
-\frac{1}{4}\mu_{a,dermis}(\lambda),
$$

$$
x(\lambda)=\frac{K_{KM}(\lambda)}{S_{KM}(\lambda)},
\qquad
A(\lambda)=1+x(\lambda),
\qquad
R_{KM,\infty}(\lambda)=A(\lambda)-\sqrt{A(\lambda)^2-1}.
$$

$$
\hat R(\lambda)=g_{system}g_{side}
\exp[-2\tau_{epi}(\lambda)]R_{KM,\infty}(\lambda).
$$

其中 `mu_a_bg`、`mu_a_blood`、`mu_s_prime`、`K_KM` 和 `S_KM` 使用同一长度单位，`t_epi_global` 使用对应长度单位，因此 `tau_epi` 和 `x` 无量纲；`f_blood_proxy` 只通过混合式生成有单位的真皮吸收，不能作为无量纲吸光度直接加到 `mu_a_dermis`。背景吸收在表皮传播和真皮混合中各按其所在层计算一次，不得在同一层重复相加。所有候选范围内必须验证 `S_KM>0`，否则该近似在该点无定义并应记为模型失败，不能裁剪成正值。

这一 K–M/RTE 系数映射来自扩散两通量近似，并非相机几何下的恒等式。Thennadil（DOI `10.1364/JOSAA.25.001480`）和 Roy 等（DOI `10.1117/1.JBO.17.11.115006`）均表明映射随入射与边界条件变化，因此这里只把它登记为半机制候选。模型另加入一个 Train-only 全局 `g_system` 处理 Hyper-Skin 相对白参考与模型边界差异；固定侧别增益继续保留。`g_system` 必须通过受试者级交叉拟合评估后再由完整 Train 冻结，不能为每条光谱单独拟合。有限厚度版本仅作预注册敏感性，因为在缺少真实成像边界条件时，“有限厚度”本身不会自动使模型更正确。

K2-MH-KM 的首轮开发范围和固定量如下；这些是保守的数值/敏感性合同，不是 Hyper-Skin 人群的生理参考区间：

| 量 | 主设置 | 敏感性或边界 | 解释 |
|---|---:|---:|---|
| `M_epi_OD` | 逐谱拟合 | `[0, 1.0]` | 570 nm 单程有效黑色素路径 OD；Jonasson 在约 575 nm 报告的总表皮黑色素吸收×厚度中位数/IQR 约为 0.13/0.073–0.23，宽边界只用于避免排除更深肤色 |
| `f_blood_proxy` | 逐谱拟合 | `[0, 0.05]` | Jonasson 上真皮中位数/IQR 为 0.011/0.0071–0.016；宽边界不是浓度真值 |
| `S0` | 固定 1.99 mm⁻¹ | 1.78/2.38 mm⁻¹；仅 K3 触发时允许 `[1.0,4.0]` | 600 nm 约化散射幅度 |
| `beta` | 固定 0.82 | 0.55/1.20 | Mie 衰减指数 |
| `gamma` | 固定 0.31 | 0.21/0.39 | Rayleigh 比例 |
| `g_system` | Train-only 全局解析/一维估计 | 受试者 LOO 稳定性 | 系统尺度，不是逐谱曝光 |

若主解或合理敏感性大量命中 `M_epi_OD/f_blood_proxy/S0` 边界，应判为结构失配或参数不可辨识；禁止仅扩大边界后继续宣称稳定。

### 5.6 表示层门与解释层分离

阶段一只设置一个表示层门。候选通过与否依据未见受试者上的重构、维度公平性、可复现性、失败率和适用域；不要求 `M_like`、`Hb_like` 或任何命名坐标在加入倾斜、散射、Shading、Specular 后保持生理排序。那些诊断仍要保存，但归档为阶段二解释线索。

**表示层 PASS（授权进入 S1-6）**需要同时满足：

- 至少一个预注册低维候选在 Train 上相对匹配的 B0/B2 基线具有可复现的重构或结构残差增益；
- 多初值、合理输入扰动、左右脸颊和主 ROI 内的结果有限且可复现；边界命中、失败和有效面积不足均显式记录；
- 候选没有依赖逐图自由曝光尺度、高容量自由残差或事后挑选波段/公式；
- 候选维度、模型版本、适用域、损失、全局参数和输入合同均可冻结；
- 任何坐标只称 `theta_repr`/proxy，不能写成绝对黑色素、血红蛋白、氧合度或跨设备真值。

**表示层 REVISE**：所有候选仍有结构性残差、维度增益只在 Train 出现、数值不稳定可由受控重参数化修复，或数据合同缺口只能依靠敏感性分析管理。**表示层 STOP**：合理低维候选无法表达真实光谱、必须依赖高容量残差/逐图尺度、数据标定无法追溯，或候选结果完全不可复现。

S1-5R 可以把多个通过 Train 表示门的候选同时提交 S1-6；Validation 负责在预注册候选中选择维度和模型版本。`PROXY_READY_FOR_S1_6`、`SEMIMECHANISTIC_READY_FOR_S1_6` 和旧 `REVISE_OR_STOP` 属于历史解释层决策代码，不再作为当前表示层授权名称。

### 5.7 前向模型接口和审计要求

修订接口必须同时返回原始尺度和 shape 空间结果：

```python
result = revised_skin_forward(
    theta_bio=theta_repr,  # 当前代码的兼容参数名；概念上统一视为 theta_repr
    wavelength_nm=wavelength_nm,
    theta_nuisance=theta_aux,
    global_params=global_params,
    model_id=model_id,
    observation_space="raw_reflectance|centered_log_ratio",
)
```

当前实现中的 `theta_bio` 是历史兼容字段，不代表其元素已经是生理量；新配置、产物和报告统一映射为 `theta_repr`，后续接口版本化时再完成字段重命名，不能在原 ID 下静默改变序列化合同。`result` 至少包含 `reflectance`、`centered_log_ratio`、`a_obs`、固定基底、包装前后 Hb、背景吸收、`mu_s_prime`、`K_KM`、`S_KM`、模型证据层级和全部全局校准量。接口必须满足：

- 参数顺序、单位、符号、参考谱和是否允许下传明确；
- 输出与波长一一对应，非负且有限；物理路径内不通过显示域裁剪修复模型；
- NumPy/PyTorch、解析线性解/数值解和自动梯度/有限差分一致；
- 包装函数在 `D_v -> 0` 时连续趋近 1，在强吸收波段产生有限且方向正确的压缩；
- 同一 nuisance 自由度必须授予匹配基线，PCA 训练必须严格按受试者折隔离；
- 记录公式来源、适用波段、与 Jonasson/Jung/Hyper-Skin 采集条件的差异及所有源码、资产、参考谱和配置哈希；
- S1-4R 数值审计只能使用合成谱和 Train 派生合同，不读取 Validation/Test；S1-5R 只读取过滤后的 Train 区域光谱。

### 5.8 设计后复核与已作修正

本次设计完成后再次按“观测定义、维度公平性、参数语义、可辨识性和数据泄漏”复核，发现并修正了以下风险：

- 旧式把 `K/S` 直接等同于 `mu_a/mu_s_prime` 不正确；已改成显式 `K_KM/S_KM` 两通量映射，并因其依赖入射与边界条件将整个 K 系列降级为 `semi_mechanistic`；严格 RTE/Monte Carlo 候选登记为 deferred；
- 仅把半无限 K–M 替换为有限厚度 K–M 仍缺少相机边界条件，因此有限厚度被降为敏感性，而非默认正确答案；
- 在 Train 参考谱上建模时，M/H 表示相对变化，可以为负，故不再沿用旧 P2 的非负绝对吸收边界，也不复用旧参数名称；
- Train 参考谱已明确为受试者等权的几何均值，参考、权重和 PCA 均须在 LOO 折内计算，避免同一受试者泄漏和多采集加权偏差；
- 直接允许逐谱自由曝光会掩盖模型失配，故改为所有匹配候选与基线共享、解析且唯一的 log 幅度剖面；它作为 `theta_aux`/观测量单独记录，不得伪装成吸收坐标；
- 只与无参数均值 B0 比较会混淆模型维度，故 proxy 层新增同尺度 nuisance 基线 B0-S 和同为二维的 B2-PCA，半机制原始尺度层新增同为两个逐谱自由参数的 B2-RPCA；
- 未包装 Hb 会把 410–430 nm 的相对吸收放大约 8–10 倍，故包装后 Hb 改为默认，未包装版本只作敏感性；
- 血管包装来源、直径/半径约定和单位现已核定；主 `D_v=15 um`，10/22/44 um 只作预注册敏感性，禁止按拟合优劣挑主值；
- 与 31 个发布中心直接对应的主资产应为 `production_10nm`，不是先前写入的 `production_5nm`；1 nm 资产只用于未知 SRF/FWHM 敏感性；
- 设计期基底预审计显示 `corr(M,tilt)=-0.953`，故 D3-MHG 的坐标混合必须在结果中报告；但它仍可作为表示候选比较，不能仅因坐标相关就被阶段一预先排除；
- 以主 scattering、`D_v=15 um` 及 `M_epi_OD/f_blood_proxy` 边界组合做 K2-MH-KM 角点预审计，`S_KM` 最小值约 1.144 mm⁻¹，未乘 `g_system` 的反射率均为正、有限；实现阶段仍须把全部角点和梯度测试固化为单元测试；
- 将平滑谱倾斜直接命名为散射系数会过度解释，故 proxy 层改名为 `q_tilt`，只有半机制层保留有单位的 `S_amp`；
- 半机制初稿曾把无量纲 Hb 路径吸光度与有单位的真皮吸收系数相加，复核后已改为 `f_blood_proxy * C_vp * mu_a_blood` 的单位闭合混合式；
- 3 名 Validation 不足以承担开放式搜索，故候选必须在 Train 预注册并限量提交；但表示层门不再强制压缩为一个 proxy 和一个 semi-mechanistic 模型，最终维度选择仍由冻结的 Validation 比较完成。

复核后的结论：proxy 主路线在观测定义、单位、参数符号、对照公平性和数据隔离上自洽；K 系列也已闭合单位并纠正 K–M 系数映射，但只能承担半机制诊断，不能支持绝对组织参数或严格机制主张。血管包装 provenance、等效直径主值/敏感性、Hb 资产和 SRF 缺口的处理均已写成可执行合同。S1-4R v3 实现后的 r7 合成审计进一步验证了中心化、解析反演、注册波段子集、包装/无包装极限、K–M 角点、梯度和后端一致性，状态为 `PASS_FOR_REVISED_TRAIN_DEVELOPMENT`。旧 S1-5R v7 的 `REVISE_OR_STOP` 仅表示旧解释层合并门失败；按当前表示层目标，Train 证据已重判为 `REPRESENTATION_READY_FOR_S1_6`，允许提交多个预注册候选，由 S1-6 在 Validation 选择并冻结。

## 6. 皮肤 mask 与区域方案

### 6.1 为什么 mask 是必要条件

整幅 HSI 包含皮肤、头发、眉眼、嘴唇、鼻孔、背景、衣物、固定装置、阴影和镜面高光。候选皮肤前向模型的适用域是皮肤反射；若不做 mask，任何低维坐标都可能主要编码非皮肤组织、背景比例或采集伪影。

若直接使用整图或粗略人体 mask，容易出现：

- 有边界的候选坐标大量触边界，或无边界坐标被异常样本拉到极端值；
- 不同初值得到不同参数；
- 参数主要反映阴影、头发比例或镜面反射；
- 散射参数吸收几何和亮度变化；
- 拟合误差看似降低，但 `theta_repr` 学到的是区域组成或采集条件 shortcut，而不是目标皮肤光谱变化。

### 6.2 分层 mask 定义

```text
valid_region_mask
= anatomical_skin
∩ semantic_validity
∩ radiometric_validity
∩ illumination_validity
∩ region_roi
```

- `anatomical_skin`：RGB 人脸解析和关键点得到的皮肤候选区；
- `semantic_validity`：排除头发、眉眼、睫毛、嘴唇、口腔、鼻孔、耳朵、颈部、衣服、背景和装置；
- `radiometric_validity`：排除非有限、近零异常、饱和及坏波段导致的不可靠像素；
- `illumination_validity`：排除明显阴影、镜面高光和推扫运动伪影；
- `region_roi`：左右脸颊为主区域，前额为次要区域，整脸皮肤为敏感性区域。

鼻部因曲率和镜面高光风险较高，不作为第一版主区域。

### 6.3 mask 生成顺序

1. 从配对 RGB 生成 face parsing 和 landmarks；
2. 按已冻结的空间转置/配准规则映射到 HSI；
3. 生成语义排除层；
4. 对语义边界进行腐蚀，降低配准和混合像素影响；
5. 只用 HSI 生成辐射质量排除层，不用待验证的 M/H 谱形选择皮肤；
6. 生成 left cheek、right cheek、forehead 和 whole-skin ROI；
7. 计算面积、连通性、边界距离、阴影/高光/饱和比例；
8. 在 Train 上完成人工抽查和规则修订；
9. 冻结算法、模型权重、阈值和版本后再处理 Validation/Test。

### 6.4 mask 产物

建议按样本保存：

```text
masks/<split>/<sample_id>/
├── anatomical_skin.npy
├── semantic_valid.npy
├── radiometric_valid.npy
├── illumination_valid.npy
├── left_cheek.npy
├── right_cheek.npy
├── forehead.npy
├── whole_skin.npy
└── qc.json
```

`mask_manifest.parquet` 记录样本、版本、各层路径、像素数量、面积比例和 QC 状态。所有 overlay 可用于内部审查，但论文图像只能使用数据许可允许公开的 `p012`、`p019` 和 `p027`。

### 6.5 mask 冻结原则

- 所有阈值只在 Train 上制定；
- Validation 用于判断规则是否泛化，不逐例修补；
- Test 不允许人工针对性调整；
- 自动失败的样本必须保留失败码，不能静默删除；
- 如果特定方向持续失败，应报告适用范围或 REVISE，而不是只保留容易样本。

## 7. 区域光谱与经验不确定性

### 7.1 主光谱

对每个有效区域计算逐波段中位数：

$$
R_{region}(\lambda)
=
\operatorname{median}\{R(p,\lambda):p\in M_{valid,region}\}.
$$

中位数作为主分析；均值和截尾均值只作敏感性分析。

### 7.2 每个区域必须保存

- `sample_id`、`subject_id`、split、expression、direction、region；
- 31 波段反射率中位数；
- 31 波段 MAD/IQR；
- 有效像素数和有效面积比例；
- 阴影、高光、饱和和语义排除比例；
- RGB–HSI 配准 QC；
- mask 版本、数据合同版本和输入哈希；
- 失败状态和失败原因。

### 7.3 不确定性来源

每名受试者的 6 次采集来自 2 种表情和 3 个方向，不是同条件技术重复。因此它们可以评价采集条件稳定性，但不能直接称为纯传感器噪声。

扰动尺度由以下信息联合估计：

- 区域内逐波段 MAD/IQR；
- 同一受试者不同采集间的稳健差异；
- Train 拟合后的波段残差；
- 可能的波长偏移和有效带宽敏感性。

`region_spectra.parquet` 建成后，模型开发应优先读取该缓存，避免每次反复读取约 49.6 GiB 的 HSI 原始数据。

## 8. 反演算法

### 8.1 优化目标

修订体系必须同时报告原始反射率和中心化差分 log 谱，不再用一个损失混合两种不同主张。semi-mechanistic 候选在原始 log 反射率上求解：

$$
\theta^*=\arg\min_{\theta\in\Theta}
\left[D_{spec}(F_{skin}(\theta),R_{HSI})
+\lambda_{prior}\Omega(\theta)\right].
$$

为了让 K2-MH-KM 与 B2-RPCA 的比较使用完全相同的观测空间和损失，主光谱损失固定为：

$$
D_{raw,WLS}=\frac{\sum_{\lambda\in\Lambda_{valid}}w_\lambda
\left[\log(\hat R_\lambda+\epsilon)-
\log(R_\lambda+\epsilon)\right]^2}
{\sum_{\lambda\in\Lambda_{valid}}w_\lambda}.
$$

`w_lambda` 只能在当前 Train LOO 折的非留出受试者上估计；完整 Train 冻结后才供未来 Validation/Test 使用。pseudo-Huber、log-MAE 和 SAM 作为共同评价/敏感性指标，不加入某一候选专属的主优化目标。Test 残差不得反向改变权重。

proxy 候选则以第 5.3 节的 `y_i` 为目标。为了使 D 系列解析解和 B2-PCA 的维度比较严格一致，主拟合目标固定为加权二次损失：

$$
D_{shape,WLS}=\frac{\sum_{\lambda\in\Lambda_{valid}}
w_\lambda[\hat y_i(\lambda)-y_i(\lambda)]^2}
{\sum_{\lambda\in\Lambda_{valid}}w_\lambda}.
$$

`a_obs` 由固定公式解析计算，不参与 D 系列 `theta_repr` 的数值搜索。B0-S、B2-PCA、D1/D2/D3 必须使用完全相同的中心化、权重和 WLS 损失；B0-R、B2-RPCA 与 K 系列必须使用完全相同的原始 log 空间权重和 WLS 损失。shape/raw log-RMSE/MAE、pseudo-Huber、SAM 和重建后的原始尺度误差全部作为评价指标；pseudo-Huber/ridge 可以做预注册敏感性，但不得与 WLS 主结果混称。

### 8.2 优化实现要求

- D1/D2/D3 的主解使用固定设计矩阵的加权线性最小二乘；无硬边界，并保存秩、SVD、解析协方差和弱 ridge 敏感性；
- K2-MH-KM/K3-MHS-KM 等有界非线性模型使用 sigmoid/softplus 映射和 16 个确定性 Sobol/Halton 初值；该数量在 Train 收敛检查后冻结；
- 保存每个初值的最终参数、损失、收敛状态和迭代次数；
- 默认先运行无参数先验或极弱先验版本，避免用强先验制造稳定性；
- 再做先验敏感性分析，检查结论是否由先验主导；
- 显式处理近零反射率、无效波段和优化失败；
- 只允许第 5.3 节预定义、解析且对全部匹配模型共同使用的 log 幅度剖面；禁止逐波段归一化、模型专属自由尺度或裁剪掩盖失配。

旧配置的参数边界已经被 S1-5 v2 否定，不得平移到修订模型。D 系列的系数是相对 Train 参考的有符号 proxy；K 系列才使用有单位边界。两类设置均须先由 S1-4R 审计、S1-5R Train 诊断，再在 Validation 后冻结。

### 8.3 可辨识性检查

每个候选模型至少执行：

- 多初值参数相对极差和解聚类；
- 参数边界命中率；
- Jacobian 的秩、奇异值和条件数；
- 局部 Hessian 或协方差近似；
- 关键参数 profile likelihood；
- 经验噪声扰动后的参数漂移；
- 加入或移除预注册观测分量前后，重构性能、联合表示子空间和坐标分配的变化；坐标变化只作解释层诊断；
- 同一受试者跨表情、方向和区域的一致性。

光谱拟合优秀但从 HSI 到 `theta_repr` 的映射存在连续平坦多解时，该候选不能直接冻结；应降低维度或重参数化为可复现的联合子空间。不同候选之间的坐标旋转、重排或重新分配不等同于表示失败，只要候选内部的表示合同固定、推理可复现且重构泛化成立。

## 9. Train、Validation 和 Test 的职责

### Train：44 名受试者、264 次采集

允许：

- 制定数据 QC、mask、ROI 和波段规则；
- 检查候选模型；
- 估计全局参数和经验扰动范围；
- 确定优化器、多初值数量和先验策略；
- 通过受试者级 bootstrap 或留一受试者分析检查稳定性；
- 草拟门控阈值。

### Validation：3 名受试者、18 次采集

只允许：

- 比较已经在 Train 定义的候选模型；
- 严格使用表示层 decision 冻结的候选清单：D1-M、D2-MH、D3-MHG 及 B0-R、B0-S、B2-PCA、B2-RPCA 对照；K2/K3 和未触发的 D3-MHO 不得在本次 Validation 临时恢复；
- 比较候选维度、重构误差、相对匹配基线的增益、失败率和联合表示稳定性；M/H 排序只作为非阻断解释诊断；
- 锁定最终表示模型、坐标顺序、参数范围/正则、损失权重、mask 版本和门控阈值；
- 生成 `frozen_stage1_spec.yaml` 及其哈希。

其中 B0-R/B0-S 是零维基线，不具备可下传坐标；B2-PCA/B2-RPCA 是同维数据驱动参照，只有在预注册规则明确允许“数据驱动表示”且通过相同门槛时才可被选为最终 `theta_repr`。D1/D2/D3 是本轮主要可选择表示。K2/K3 已由 Train 证据排除。

Validation 很小，因此不得在此新造候选或调整公式。S1-6 实现前必须把“最小改善、复杂度优先、失败率和并列处理”写入配置，再首次读取 Validation 光谱。较高维候选只有在三名受试者方向一致、相对低一维候选达到预注册最小增益且未增加失败时才升级；否则选择满足门槛的更低维候选。若没有候选达到冻结门，则返回 REVISE，不访问 Test。

### Test：4 名受试者、24 次采集

- 冻结后只运行一次；
- 每条 Test 光谱进行 theta 反演属于推理，不允许重新拟合全局参数；
- 先在受试者内聚合 6 次采集，再汇总 4 名受试者；
- 报告个体结果、失败率和不确定区间，不能只报告 24 次采集的窄区间；
- Test 后若修改任何模型、mask 或阈值，原 Test 已被使用，新的结果必须标为修订性分析，不能继续声称首次独立验证。

## 10. 分阶段实施流程

### S1-0：建立证据化数据合同

目标：把数据格式、物理量、波长、校正、空间轴和划分从代码假设变成可审计合同。

主要工作：

1. 生成全部 RGB/VIS 文件清单和哈希；
2. 解析 subject、expression 和 direction；
3. 核对 264/18/24 数量和受试者无泄漏；
4. 写入论文/补充材料证据及 `confirmed/inferred/missing`；
5. 将 `400:10:700` 标为 `confirmed_from_official_code`，保存官方仓库 URL、提交、文件路径、git blob 和代码表达式；
6. 记录 HSI 转置、dtype、数值范围和派生裁剪规则。

完成标准：`data_contract.json`、`contract_evidence.md`、`split_manifest.csv` 均可复现且不存在关键矛盾。

当前实测状态：PASS。306 对输入完成全量数值扫描和 SHA-256；Train/Validation/Test 数量为 264/18/24，受试者为 44/3/4，无泄漏且无 NaN/Inf。检测到 7 个极小负值和 2443 个略高于 1 的值，最大上越约 `9.8e-9`，按派生数据裁剪、原始数据只读处理。

### S1-1：RGB–HSI 配准核验

目标：确认 RGB 生成的 mask 可以映射到 HSI。

主要工作：

1. 只从 Train 选取至少 18 个内部 QC 样本；
2. 覆盖全部 2 种表情、3 个方向及不同外观亮度/肤色范围；
3. 比较空间转置前后的灰度/边缘相关性；
4. 检查脸部轮廓、眼口鼻和局部高对比结构 overlay；
5. 记录自动指标和人工结论。

完成标准：形成唯一、冻结的坐标映射；无法用同一规则解释全部采集时先 REVISE 数据读取，不进入 mask。

当前实测状态：PASS。18/18 个分层 Train 样本均选择 `transpose`；中位/最低灰度相关为 0.99498/0.98873，中位/最低边缘相关为 0.98758/0.96942，中位候选分数余量为 0.47762。研究者 QinghaoLi 已复核叠加图并确认五官、轮廓无系统性红绿错边，现已写入 `frozen_transform=transpose` 和 `next_stage_allowed=true`。

### S1-2：生成并冻结 mask/ROI

目标：建立模型真正适用的皮肤区域。

主要工作：按第 6 节生成分层 mask、ROI、QC 和 manifest；仅在 Train 上修订算法；用 Validation 检查泛化。

完成标准：主 ROI 面积、连通性和辐射有效性达到 Train/Validation 预先冻结的规则；失败样本均有明确状态。

当前实测状态：Train r2 的自动门和人工冻结均已通过。264 个 Train 样本为 183 PASS、81 REVIEW、0 FAIL。目标部署域 r5 对 `neutral/front` 的 3 个 Validation 样本得到 2 PASS、1 REVIEW、0 FAIL，主域 usable fraction 1.0，且双侧主脸颊全部可用、REVIEW 码仅属于允许的 whole-skin/次要区域提示。QinghaoLi 已人工确认 r5；最终 decision 为 `PASS_FOR_DEVELOPMENT`、`next_stage_allowed=true`、`authorized_next_stage=S1-3`、`test_access_count=0`。

### S1-3：提取区域光谱和波段 QC

目标：形成轻量、可重复的阶段一主要输入。

主要工作：提取左右脸颊、前额和整脸皮肤的中位光谱、MAD/IQR、有效像素及质量字段；建立波段可靠性报告。

完成标准：`region_spectra.parquet` 与 mask、原始文件、波长合同一一可追溯。

当前实测状态：`PASS_FOR_S1_4`。282 个 Train/Validation 样本形成 1128 条样本×区域记录，其中 930 条可用、198 条按冻结规则显式不可用；44 名 Train 和 3 名 Validation 的正脸无表情双侧脸颊共 94/94 可用。所有可用中位光谱均为有限值，范围为 0.048991–0.918025，Test 访问计数为 0。Train 主域波段诊断提示 670–690 nm 局部曲率较高，且图像左侧脸颊宽带反射率在 44/44 名 Train 受试者上高于右侧，中位差 0.076692；这些现象不在 S1-3 自动删波段，而进入 S1-4/S1-5 的散射/观察 nuisance、左右侧分层和波段敏感性分析。

### S1-4：补齐候选模型与单元测试

目标：将当前固定两参数骨架升级为可比较的模型注册表。

主要工作：实现 B0、B1、P2、P3-S、P3-O 和诊断性 P4；统一变维参数合同；验证 NumPy/Torch、梯度、边界和分量审计。

完成标准：每个候选模型均有独立配置、来源说明和测试；任何非有限输出或梯度错误均阻止反演。

当前实测状态：`PASS_FOR_S1_5`。正式版本为 `data/processed/HyperSkin_Stage1_v1/models/s1_4_v2`；v1 的模型数值审计通过，但 Jacques 参考文献中的冒号被 YAML 解析为映射，故未覆盖原目录，而由 v2 通过 `supersedes` 字段显式替代。v2 注册了 B0、B1、P2、P3-S、P3-O、P4，所有模型的有限性、边界、NumPy/Torch 一致性、自动梯度与中心有限差分、分量守恒均通过。B0/B1/P2/P3-S 已激活为 S1-5 入口；P3-O/P4 只有在 Train 残差满足注册条件时才可激活。

参数合同不再把 `theta_repr` 预先固定为 `M_absorbance/Hb_absorbance_proxy`。`M_like`、`Hb_like`、`S_amp`、`q_tilt`、`sO2`、Shading 和 Specular 等按观测域注册为候选坐标或辅助坐标；有单位的半机制量保留边界，纯表示坐标不通过边界制造稳定性。逐图自由曝光尺度被硬拒绝；S1-3 的固定侧别效应由可选的 Train-only 全局 side log-gain 接口承接，默认全为 0，必须一次估计并冻结，同时仍需左右侧分别报告和对称聚合。

有效 SRF 仍为 `missing`。已预注册全 31 波段、去 400/700 nm 端点、去 400/420/590/670/680/690/700 nm 高曲率候选，以及中心偏移和假定 Gaussian FWHM `0/5.5/10/20 nm` 的敏感性接口。高曲率候选不是坏波段结论，不得在 S1-5 前自动删除。

### S1-5：Train 反演与诊断

目标：在 Train-only 条件下判断候选模型是否提供了可泛化的低维表示，而不是证明某个坐标已经具有确定的生理身份。

主要工作：运行多初值区域反演、匹配维度基线比较、波段残差、profile likelihood、Jacobian、扰动、边界、受试者内稳定性和候选维度/复杂度分析；只在 Train 拟合预先允许的全局参数，不允许逐图自由曝光尺度或高容量自由残差绕过 `theta_repr`。

完成标准：能够比较不同维度和结构的候选表示，明确其重构增益、复杂度、联合坐标稳定性和适用域，并据此决定哪些候选提交 Validation。`M_like`、`Hb_like`、`q_tilt`、Shading 或 Specular 等名称只表示候选坐标/分量来源，不要求在本阶段确认生理身份，也不以坐标排序漂移作为表示层阻断条件。

当前实测状态：旧 `S1_5 v2` 和 `S1_5R v7` 均按包含解释层门控的历史口径保留；表示层重判版本为 `data/processed/HyperSkin_Stage1_v1/train_development/s1_5r_representation_v3`。后者不读取 Validation/Test/raw HSI，不覆盖旧 decision，只判断 Train 上是否存在可行的低维 `theta_repr`。

历史 S1-5 v2 只读取过滤后的 Train 区域光谱。主分析为 44 名受试者、88 条 `neutral/front` 双侧脸颊光谱；Validation/Test/原始 HSI 内容访问计数均为 0。16 个确定性 Halton 初值下，B0/B1/P2/P3-S 的结果保留为旧模型覆盖性证据，不再作为当前候选体系的唯一结论：

| 模型 | median log-RMSE | median SAM | 任一参数触边界比例 | 多个近最优解簇比例 |
|---|---:|---:|---:|---:|
| B0 | 0.1660 | 0.0876 | 0.000 | 0.000 |
| B1 | 0.6385 | 0.2967 | 0.023 | 0.000 |
| P2 | 0.6532 | 0.2685 | 0.898 | 0.193 |
| P3-S | 0.6141 | 0.2672 | 1.000 | 0.136 |

旧 P2/P3-S 的边界坍缩、profile 平坦和参数漂移说明“旧公式/旧参数范围下的坐标”存在非辨识或结构失配；它们不能单独推出“所有低维表示不可行”，也不能在阶段一给出 M/H 生理解释。

旧残差仍有波长结构，P3-O/P4 未触发；这只说明旧模型扩展没有得到足够的 Train 证据。后续候选必须以新模型/配置版本登记并仍只用 Train 重跑，不能通过查看 Validation 或 Test 事后挑选维度、波段和参数范围。

### S1-4R/S1-5R：修订循环

S1-4R 目标：把第 5 节的 B0-R/B0-S/B2-PCA/B2-RPCA、D 系列和 K–M 半机制系列写成新的 v3 注册与独立实现，不覆盖旧 ID；核验冻结 Hb 吸收资产和哈希，审计血管包装、K–M 系数映射、中心化算子、受试者折隔离、解析/数值解、极限和 backend parity。完成标准是 `PASS_FOR_REVISED_TRAIN_DEVELOPMENT`，不是模型有效性 PASS。

S1-5R 目标：仍只用 44 名 Train 受试者，对 B0-R/B0-S、B2-PCA/B2-RPCA、D1-M、D2-MH、D3-MHG 及条件触发的 D3-MHO/K 系列进行表示层比较。候选可同时提交 S1-6，由 Validation 选择维度和模型版本；不再要求 Train 先压缩为唯一 proxy 或 semi-mechanistic 模型，也不把 M/H 坐标稳定性写成 S1-5R 的必过门。

S1-5R v7 正式使用 44 名受试者、88 条正脸无表情双脸颊光谱，所有参考谱、侧别增益、PCA 基底和 K2 系统增益均按受试者留一计算。D2-MH 的 median shape log-RMSE 为 0.04015，相对 B0-S 的 0.08552 降低 53.0%，88/88 条光谱均改善；D3-MHG 进一步降至 0.03133，B2-PCA 为 0.03671。以上结果支持“Train 上存在可行的 2–3 维候选表示”，但不决定最终模型，也不决定坐标语义。

旧 v7 报告的 M/H 排序漂移、包装直径敏感性和短波敏感性，现归入“表示坐标与观测因素的混合诊断”。它们提示阶段二需要用 RGB–HSI 自监督闭环、外部相关或消融实验解释各坐标行为，但不否定 D2/D3 作为待验证的 `theta_repr` 候选。

K2-MH-KM 的 median raw log-RMSE 为 0.27180，是 B0-R 的 1.689 倍，仅 20.5% 光谱优于 B0-R，shape 误差为 D2 的 6.54 倍；`M_epi_OD` 100% 位于下边界邻域。该证据否定当前 K2 半机制候选及其 K3 触发条件，不否定 D 系列或 B2-PCA 等其他低维表示。D3-MHO 仍因残差证据不足未触发。

当前状态：S1-4R r7 为 `PASS_FOR_REVISED_TRAIN_DEVELOPMENT`；旧 S1-5R v7 的解释层合并 decision `REVISE_OR_STOP` 原样保留。按表示层重判，`data/processed/HyperSkin_Stage1_v1/train_development/s1_5r_representation_v3/s1_5r_representation_decision.json` 为 `REPRESENTATION_READY_FOR_S1_6`，`next_stage_allowed=true`，允许将 `D1-M/D2-MH/D3-MHG` 及其 B0/B2 对照提交 Validation。Validation/Test/原始 HSI 内容访问计数均为 0；`selected_representation_model` 仍为空，最终选择必须在 S1-6 完成。

r7 合成审计目录为 `data/processed/HyperSkin_Stage1_v1/models/s1_4_revised_v3_synthetic_r7`，在 r5 基础上补齐了显式 `D_v=0` 无包装极限和 K2 无包装前向审计。S1-5R v7 正式目录为 `data/processed/HyperSkin_Stage1_v1/train_development/s1_5r_v7`，表示层重判目录为 `data/processed/HyperSkin_Stage1_v1/train_development/s1_5r_representation_v3`。合成结果只证明代码实现和自生成数据可逆；当前 Train 表示层授权以 v3 decision 为准，旧 v7 decision 作为历史解释层审计保留。

### S1-6：Validation 选择与冻结

目标：在不查看 Test 的条件下，用未见受试者 Validation 选择并冻结最终 `theta_repr` 维度、模型版本和适用域；解释层诊断单独记录，不回写为阶段一表示层门。

主要工作：只读取 S1-5R 注册并通过 Train 表示层门的候选，比较 `B0/B2/D` 的预注册维度和结构；K 系列维持 Train 淘汰。冻结模型 ID、`theta_repr` 顺序、参数范围/正则、全局参数、mask 版本、波段权重、损失、多初值、扰动策略、适用域和全部 gate threshold。不得根据 Validation 结果给坐标追加 M/H 生理命名。

完成标准：生成 `frozen_stage1_spec.yaml`、候选比较表、输入清单和哈希；冻结文件中不存在 `null` 门槛，并明确记录“表示层选择”和“解释层未决问题”两栏。

当前实测状态：`S1_6_FROZEN_FOR_S1_7`。正式目录为 `data/processed/HyperSkin_Stage1_v1/freeze/s1_6_v1`。完整 Train 的 44 名受试者、88 条主域脸颊光谱只用于一次性冻结参考谱、侧别增益和 PCA 基底；模型选择只使用 3 名 Validation 受试者的 6 条 `neutral/front` 双侧脸颊光谱。Test 与原始 HSI 内容访问均为 0。

| 候选 | 维度 | Validation 受试者中位 shape log-RMSE | 相对 B0-S 比值 | 改善受试者比例 | 结果 |
|---|---:|---:|---:|---:|---|
| D1-M | 1 | 0.03427 | 0.4333 | 3/3 | eligible |
| D2-MH | 2 | 0.03342 | 0.4226 | 3/3 | eligible，最终冻结 |
| D3-MHG | 3 | 0.03064 | 0.3874 | 3/3 | eligible，绝对误差最低 |
| B2-PCA | 2 | 0.03594 | 0.4544 | 3/3 | eligible，对照 |

D3-MHG 的误差最低，但冻结的 10% parsimony tolerance 上限为 0.03370，D2-MH 的 0.03342 仍处于近最佳集合；因此按“近最佳集合内优先低维”的预注册规则选择 D2-MH，而不是因坐标名称选择它。D1-M 和 B2-PCA 超出近最佳集合。所有候选均完成 6/6 条拟合、设计矩阵满秩、指标有限，逐波段最大中位 shape 残差均低于 0.15。

由于 Validation 只有 3 名受试者，这一结果是保守的模型选择和冻结依据，不是精确的总体效应估计，也不支持显著性或人群泛化声明；真正的独立留出判定仍由 S1-7 完成。

冻结表示为 `theta_repr=[delta_M_OD, delta_Hb_OD]`，并单独保存用于完整光谱重构的 `theta_aux=[a_obs]`。这些历史参数名只标识固定基底坐标，不构成黑色素/血红蛋白生理量声明。45 条可用 Validation 压力光谱只用于非阻断适用域报告，D2-MH 的总体中位 shape log-RMSE 为 0.04253，各条件中位数范围为 0.02546–0.05755；`p016_smile_right` 继续按既有 mask 决策排除。

冻结产物 `frozen_stage1_spec.yaml` 无 `null`，`frozen_calibration.npz` 固化全 Train 参考、统一波段权重、侧别 log-gain 和 PCA 对照基底。decision 中 14 个前置输出以及 output manifest 中包含 decision 的 15 个文件哈希复算全部一致，主拟合与压力拟合失败数均为 0。该 decision 只授权一次 S1-7 Test；冻结后任何模型、坐标、阈值、mask 或全局参数变化都会使后续分析失去“首次独立 Test”地位。

### S1-7：正式 Test 与决策

目标：在 4 名未见受试者上作一次独立留出判定。

主要工作：只加载冻结规范；执行 Test theta 推理；按受试者聚合；输出拟合、稳定性、可辨识性、失败率和残差报告。

完成标准：产生唯一的 `STAGE1_DECISION.md`，结论为 PASS、REVISE 或 STOP，并记录全部证据。

当前实测状态：`STAGE1_COMPLETE / PASS`。正式目录为 `data/processed/HyperSkin_Stage1_v1/formal_test/s1_7_v1`，运行号为 1/1。正式访问前，专用 runner 逐一复算 S1-6 全部冻结输出、S1-2 mask 配置/实现/检查点、S1-0 波长证据 amendment、空间转置和数据 manifest 哈希；preflight 仅查看 24 条 Test 元数据，记录 RGB/HSI 内容访问均为 0。创建不可覆盖的 `FORMAL_TEST_LOCK.json` 后才首次读取 Test，最终读取 24 个 RGB 和 24 个 HSI，未读取 Train/Validation 内容、未重估全局量、未重选模型、未人工修补 mask。

Test mask 自动结果为 21 PASS、3 REVIEW、0 FAIL。三个 REVIEW 是两个次要前额不足和一个整体皮肤连通性提示；4 名受试者的 `neutral/front` 双侧主脸颊 8/8 全部满足冻结像素阈值。正式主域结果如下：

| 指标 | S1-7 结果 | 冻结门 | 判定 |
|---|---:|---:|---|
| 主脸颊可用率 | 8/8 = 1.000 | >= 0.750，且每人至少 1 条 | PASS |
| 优于 B0-S 的受试者比例 | 4/4 = 1.000 | >= 2/3 | PASS |
| D2-MH 受试者中位 shape log-RMSE | 0.05341 | 描述性 | - |
| B0-S 受试者中位 shape log-RMSE | 0.08061 | 描述性 | - |
| D2-MH/B0-S 中位误差比 | 0.66265 | <= 0.90 | PASS |
| 最大逐波段绝对中位 shape 残差 | 0.08448 | <= 0.15 | PASS |
| D2-MH 拟合失败率 | 0/8 = 0 | = 0 | PASS |
| 参数有限、设计满秩 | 8/8 | 全部 | PASS |

4 名 Test 受试者逐人均优于 B0-S；D2-MH 的受试者 shape log-RMSE 为 0.03852--0.05896。冻结 B2-PCA 对照在 p002/p035/p039 上误差略低于 D2，在 p036 上略高；它不参与已冻结模型的事后重选。笑脸、侧脸和其他区域只形成 16 个非阻断压力分层，其 D2-MH 中位误差范围约 0.03646--0.06464，不改变主域 PASS，也不支持对表情/姿态稳定性的强外推。

所有 9 项正式 gate 均通过，拟合失败为 0。`output_manifest.json` 登记 332 个文件，SHA-256 复算零不匹配。该 PASS 只回答“冻结二维表示能否在未见 Hyper-Skin 受试者的正脸无表情脸颊光谱上保留可复现的低维重构增益”；不确认真实黑色素/血红蛋白浓度，不验证无眼镜部署域，也不支持跨设备可比性。

## 11. 评价指标与门控

| 维度 | 主要指标 | 聚合单位 | 作用 |
|---|---|---|---|
| 光谱拟合 | log-RMSE/MAE、SAM | 先区域/采集，再受试者 | 检查前向覆盖性 |
| 波段残差 | 绝对/相对残差、残差谱形 | 受试者与波长 | 判断缺失物理因素 |
| 匹配基线增益 | proxy：B0-S/B2-PCA 对 D；原始尺度：B0-R/B2-RPCA 对 K | 受试者 | 分别控制尺度自由度和参数维度，比较 `theta_repr` 的重构增益 |
| 表示增益保留率 | 按候选维度与匹配 B0/B2 基线计算 | 受试者/bootstrap | 判断受约束表示保留了多少数据驱动的可重构变化；不赋予坐标生理含义 |
| 多初值稳定性 | theta 相对极差、解聚类 | 区域/采集 | 检查多解 |
| 边界 | 上下界命中比例 | 受试者 | 检查模型或范围错误 |
| 局部辨识性 | Jacobian 秩、奇异值、条件数 | 区域/采集 | 检查参数可分性 |
| 区间辨识性 | profile likelihood 区间 | 受试者 | 检查参数是否形成有限区间 |
| 扰动稳定性 | 输入扰动后的 theta 漂移 | 受试者 | 检查噪声敏感性 |
| 采集条件稳定性 | 同人跨表情/方向差异 | 受试者 | 非阻断压力测试；检查超出主部署域后的参数漂移 |
| mask 敏感性 | ROI/腐蚀/质量层变化 | 受试者 | 检查结果是否由 mask 驱动 |

数值阈值不能现在凭经验填写。制定顺序为：Train 估计参考分布和噪声范围，Validation 选择并冻结，Test 只判断。

### PASS（表示层）

- 数据合同和 mask 质量门通过；
- 至少一个预注册的低维候选（如 D1/D2/D3 或数据驱动 B2）在正脸、无表情主域上优于匹配基线，并在受试者级汇总中显示可复现的重构或结构残差增益；
- 多初值和扰动结果稳定；
- 候选没有依赖高容量自由残差或逐图曝光尺度；边界和失败状态已显式报告，并且不会使候选失去可复现的表示能力；
- S1-5R 开发门先写为 `REPRESENTATION_READY_FOR_S1_6`；S1-7 最终 decision 必须写为 `STAGE1_COMPLETE / PASS`，并冻结维度、适用域和输入合同；坐标只称 `theta_repr`/proxy，生理解释留给阶段二。

笑脸或侧脸压力域失败不单独否定主域 PASS，但必须报告，且结论不得外推为对表情和姿态稳定。无眼镜域必须由新的目标域留出集或外部数据验证。

### REVISE

- 主要谱形可拟合，但存在机制明确的结构性残差；
- 加入一个受约束参数有可能修复；
- mask 或特定方向存在可定位的问题；
- 参数不够稳定，但合并参数、调整范围或合理先验可能修复；
- 有效带宽等合同缺口只能通过敏感性分析管理，需要限制结论。

### STOP

- 物理量或波长顺序无法确认；
- 合理参数范围内不能表达真实光谱；
- 多组完全不同的参数产生近乎相同光谱且无法简化；
- 只有高容量残差或逐图自由尺度才能拟合；
- Test 前没有冻结规范，或 Test 被反复用于调参。

STOP 后不得训练以“物理解释”为目标的 RGB 编码器。

## 12. 当前代码与目标方案的差距

| 现有文件 | 当前能力 | 必须补齐 |
|---|---|---|
| `data_contracts.py` | 发现 RGB/VIS 配对、解析 subject/expression/direction、读 `cube`、基础泄漏检查 | 分层 mask、区域 manifest、波段 QC |
| `s1_contract.py` | S1-0 全文件哈希、HDF5/RGB 检查、全量数值扫描、证据状态和三项合同产物；官方代码波长证据已固化 | 保留发布版有效 SRF/带宽缺失状态，并在后续模型阶段执行敏感性分析 |
| `s1_registration.py` | S1-1 Train 分层选样、8 种坐标变换、灰度/边缘指标、审查图和人工冻结门 | 研究者完成本次人工确认；后续 S1-2 只读取冻结映射 |
| `s1_masks.py` | S1-2 分层 mask/ROI、Train/Validation 门、QC/manifest、人工冻结和 Test 禁入 | Train r2 已冻结；Validation 原始结果保留，目标域由独立 profile 重评分 |
| `evaluate_s1_2_validation_profile.py` | 对既有冻结协议 Validation manifest 进行目标部署域分层重评分，不重写 mask | `neutral/front` 主门，其他 strata 非阻断但必须报告 |
| `s1_target_profile.py` / `finalize_s1_2_target_profile.py` | 人工签署目标域开发门、合并 Train+Validation manifest、写入证据边界 | r5 已签署并授权 S1-3；禁止独立验证表述 |
| `s1_region_spectra.py` / `extract_hyperskin_region_spectra.py` | S1-3 Train/Validation 区域中位谱、MAD/IQR、均值/截尾均值、输入哈希、Test 硬隔离和自动门 | 已生成正式缓存并授权 S1-4 |
| `summarize_s1_3_band_reliability.py` | 仅用 Train 主域脸颊形成波段可靠性诊断和侧别效应报告 | 诊断波段只作敏感性候选，不自动排除 |
| `s1_4_model_registry_v1.yaml` / `model_registry.py` | 旧 B0/B1/P2/P3-S/P3-O/P4 合同和激活规则 | 作为 `legacy_rejected_s1_5_v2` 保留；新增 v3 注册，不覆盖旧合同 |
| `skin_forward.py` | 旧六模型、统一 bio/nuisance/global 接口及旧 P2 兼容层 | 保留回归；新增 DOD proxy、包装 Hb 和修订 K 系列实现，不在原 ID 下替换公式 |
| `s1_4_revised_registry_v3.yaml` | 已机器可读固化 B0-R/B0-S/B2-PCA/B2-RPCA、D/K 候选，资产哈希、D_v/sO2/厚度、K–M 映射及 deferred RTE 状态 | r7 合成审计通过；旧 v7 解释层 decision 为 `REVISE_OR_STOP`，表示层重判 v3 已授权 S1-6 |
| `s1_revised_forward.py` / `s1_proxy_inverse.py` | 已实现中心化 log-ratio、解析 `a_obs`、D 系列加权线性解、PCA 受试者隔离、K2/K3 前向和 K2/K3 有界反演接口 | 数值实现通过；D2/D3 作为表示候选保留，K2 结构失配列为待修订候选；不把坐标混杂写成阶段一阻断 |
| `build_s1_3_train_only_cache.py` | 将混合缓存按 split 谓词物化为带 row-group 统计和哈希的 Train-only Parquet | 1056 行、5 个 row group 均为 `train/train`；用于 S1-5R v7 |
| `s1_revised_train.py` / `run_s1_5_revised_train.py` / `s1_5_revised_train_v3.yaml` | 受试者 LOO 参考/侧别/PCA/K2 全局增益、matched baseline、条件模型、扰动、区域和物理/光谱敏感性、不可覆盖产物 | v7 旧解释层门控完成；原 decision 保留，表示层重判见 `s1_5r_representation_v3` |
| `reclassify_s1_5r_representation.py` / `s1_5r_representation_v1.yaml` | 只复用既有 Train-only v7 证据，按表示层门生成独立 decision 和输入/输出哈希 | v3 为 `REPRESENTATION_READY_FOR_S1_6`；最终模型仍为空，下一步实现 S1-6 Validation 选择/冻结 |
| `s1_validation_freeze.py` / `run_s1_6_validation_freeze.py` / `s1_6_validation_freeze_v1.yaml` | 全 Train 一次冻结参考/侧别/PCA；仅用 Validation 主域做受试者级候选选择；压力域非阻断；生成不可变规范和哈希 | v1 已冻结 D2-MH 二维表示，decision 为 `S1_6_FROZEN_FOR_S1_7`；Test 仍为零访问 |
| `s1_formal_test.py` / `run_s1_7_formal_test.py` / `s1_7_formal_test_v1.yaml` | Test inference-only、锁前哈希预检、单次运行锁、冻结 mask/光谱提取、受试者级 gate、压力报告和递归输出哈希 | v1 已唯一运行，`STAGE1_COMPLETE / PASS`；Test 访问为 1/1，禁止覆盖重跑 |
| `s1_spectral_sensitivity.py` | 预注册波段集合、中心偏移和假定 Gaussian FWHM 接口 | S1-5 必须只用 Train 执行并报告 |
| `s1_model_audit.py` / `audit_s1_4_models.py` | 前置门、不可覆盖产物、模型/梯度/边界/敏感性审计和哈希决策 | v2 `PASS_FOR_S1_5` |
| `s1_inverse.py` / `s1_train_development.py` | 变维多初值反演、pseudo-Huber log 损失+SAM、Jacobian/SVD、局部协方差、近最优解聚类、profile、经验扰动、条件/区域稳定性、光谱敏感性和不可覆盖审计产物 | 历史 S1-5 v2 完成；旧模型问题保留为修订依据，不单独否定新的低维表示候选 |
| `run_s1_5_train_development.py` / `s1_5_train_development_v1.yaml` | Train-only 入口、数据隔离、条件模型触发、固定侧别增益和完整 provenance | Validation/Test/原始 HSI 内容零访问；旧 v2 decision 不进入 S1-6，表示层重判由独立 v3 decision 授权 |
| `hsi_inverse_fit.py` | 旧两参数多起点、log 损失+SAM、有限差分 Jacobian | 保留为旧开发实现，不用于 S1-5 正式结论 |
| `stage1_pipeline.py` | 单次发现—拟合—门控流程 | S1-2 至 S1-7 分步产物、mask/光谱缓存、模型比较、冻结哈希和 Test 隔离 |
| `run_stage1_hsi_physics.py` | 读取外部 mask 后运行旧 P2 | 保留为旧开发入口；正式流程必须经过新分阶段命令和前置门 |

当前 `tests/skin_optics_hsi` 共 63 项测试通过。S1-6 测试覆盖近最佳集合内低维优先、同维固定优先级、不可选择未通过候选和无候选时拒绝冻结；S1-7 测试覆盖完整 PASS、允许 6/8 区域但仍要求每名受试者至少一条、以及受试者改善不足时拒绝 PASS。源码和脚本编译通过。测试证明实现与决策合同成立，表示层 PASS 仍不意味着 M/H 生理解释已确认。

## 13. 目标命令接口

S1-0 至 S1-7 已实现并完成；S1-7 已以 D2-MH 完成唯一一次正式 Test，结论为 PASS。

```powershell
# S1-0：数据合同
python scripts/skin_optics_hsi/audit_hyperskin_contract.py `
  --data-root "E:\projects\face2\data\raw\Hyper-Skin(RGB, VIS)" `
  --output-root "data\processed\HyperSkin_Stage1_v1"

# S1-1：Train-only 配准自动核验
python scripts/skin_optics_hsi/audit_hyperskin_registration.py `
  --contract "data\processed\HyperSkin_Stage1_v1\contracts\data_contract.json" `
  --output-root "data\processed\HyperSkin_Stage1_v1"

# S1-1：研究者看图后人工冻结或拒绝映射
python scripts/skin_optics_hsi/finalize_hyperskin_registration.py `
  --registration-dir "data\processed\HyperSkin_Stage1_v1\registration_qc" `
  --reviewer "<name>" --decision pass --notes "<visual review notes>"

# S1-2：Train 分层 mask/ROI（已运行 r2，以下为不可覆盖的新运行接口）
python scripts/skin_optics_hsi/build_hyperskin_masks.py `
  --contract "data\processed\HyperSkin_Stage1_v1\contracts\data_contract.json" `
  --registration "data\processed\HyperSkin_Stage1_v1\registration_qc\registration_decision.json" `
  --output-root "data\processed\HyperSkin_Stage1_v1" `
  --config "configs\skin_optics_hsi\s1_2_masks_v1.yaml" --split train

# S1-2：研究者看图后冻结或拒绝 Train 规则
python scripts/skin_optics_hsi/finalize_hyperskin_masks.py `
  --decision-file "data\processed\HyperSkin_Stage1_v1\masks\r2\s1_2_train_decision.json" `
  --split train --reviewer "QinghaoLi" --decision pass --notes "<visual review notes>"

# S1-2：只能使用冻结的 Train 配置与 provenance 处理 Validation
python scripts/skin_optics_hsi/build_hyperskin_masks.py `
  --contract "data\processed\HyperSkin_Stage1_v1\contracts\data_contract.json" `
  --registration "data\processed\HyperSkin_Stage1_v1\registration_qc\registration_decision.json" `
  --output-root "data\processed\HyperSkin_Stage1_v1" `
  --config "data\processed\HyperSkin_Stage1_v1\freeze\s1_2_mask_protocol.yaml" `
  --split valid `
  --frozen-protocol-provenance "data\processed\HyperSkin_Stage1_v1\freeze\s1_2_mask_protocol_provenance.json"

# S1-2：按真实部署域重评分既有 Validation（不覆盖原始 r2 decision）
python scripts/skin_optics_hsi/evaluate_s1_2_validation_profile.py `
  --valid-decision "data\processed\HyperSkin_Stage1_v1\masks\r2\s1_2_valid_decision.json" `
  --profile "configs\skin_optics_hsi\s1_2_validation_target_front_neutral_v3.yaml" `
  --output-root "data\processed\HyperSkin_Stage1_v1" --output-revision r5

# S1-3：区域光谱
python scripts/skin_optics_hsi/extract_hyperskin_region_spectra.py `
  --contract "data\processed\HyperSkin_Stage1_v1\contracts\data_contract.json" `
  --mask-manifest "data\processed\HyperSkin_Stage1_v1\manifests\mask_manifest.parquet" `
  --output-root "data\processed\HyperSkin_Stage1_v1\region_spectra"

# S1-4：候选模型注册与数值审计（正式结果为不可覆盖的 v2）
python scripts/skin_optics_hsi/audit_s1_4_models.py `
  --s1-3-decision "data\processed\HyperSkin_Stage1_v1\region_spectra\s1_3_final_decision.json" `
  --data-contract "data\processed\HyperSkin_Stage1_v1\contracts\data_contract.json" `
  --registry "configs\skin_optics_hsi\s1_4_model_registry_v1.yaml" `
  --output-dir "data\processed\HyperSkin_Stage1_v1\models\s1_4_v2" `
  --supersedes-decision "data\processed\HyperSkin_Stage1_v1\models\s1_4_v1\s1_4_decision.json"

# S1-5：Train 开发
python scripts/skin_optics_hsi/run_s1_5_train_development.py `
  --config "configs\skin_optics_hsi\s1_5_train_development_v1.yaml" `
  --output-dir "data\processed\HyperSkin_Stage1_v1\train_development\<new_s1_5_version>" `
  --supersedes-decision "data\processed\HyperSkin_Stage1_v1\train_development\<previous_version>\s1_5_decision.json"

# S1-4R：修订模型纯合成审计（r7 为当前不可覆盖正式结果）
python scripts/skin_optics_hsi/audit_s1_4_revised_models.py `
  --output-dir "data\processed\HyperSkin_Stage1_v1\models\s1_4_revised_v3_synthetic_r7" `
  --supersedes-decision "data\processed\HyperSkin_Stage1_v1\models\s1_4_revised_v3_synthetic_r6\s1_4r_decision.json"

# S1-3 派生：物理隔离的 Train-only 缓存
python scripts/skin_optics_hsi/build_s1_3_train_only_cache.py `
  --source "data\processed\HyperSkin_Stage1_v1\region_spectra\region_spectra.parquet" `
  --output-dir "data\processed\HyperSkin_Stage1_v1\region_spectra\train_only_v1"

# S1-5R：修订模型 Train-only 正式结果 v7
python scripts/skin_optics_hsi/run_s1_5_revised_train.py `
  --config "configs\skin_optics_hsi\s1_5_revised_train_v3.yaml" `
  --output-dir "data\processed\HyperSkin_Stage1_v1\train_development\s1_5r_v7" `
  --supersedes-decision "data\processed\HyperSkin_Stage1_v1\train_development\s1_5r_v6\s1_5r_decision.json"

# S1-5R：表示层重判（不读取 Validation/Test，不覆盖 v7）
python scripts/skin_optics_hsi/reclassify_s1_5r_representation.py `
  --config "configs\skin_optics_hsi\s1_5r_representation_v1.yaml" `
  --output-dir "data\processed\HyperSkin_Stage1_v1\train_development\s1_5r_representation_v3"

# S1-6：Validation 选择并冻结（正式结果 v1）
python scripts/skin_optics_hsi/run_s1_6_validation_freeze.py `
  --config "configs\skin_optics_hsi\s1_6_validation_freeze_v1.yaml" `
  --output-dir "data\processed\HyperSkin_Stage1_v1\freeze\s1_6_v1"

# S1-7：冻结后的唯一正式 Test（已执行，不可重复）
python scripts/skin_optics_hsi/run_s1_7_formal_test.py `
  --config "configs\skin_optics_hsi\s1_7_formal_test_v1.yaml" `
  --output-dir "data\processed\HyperSkin_Stage1_v1\formal_test\s1_7_v1"
```

该命令已消耗唯一正式 Test。不得再次执行或改用旧 P2 入口覆盖；后续只能读取现有产物。

## 14. 最终产物

```text
stage1_hsi_physics/<run_id>/
├── data_contract.json
├── contract_evidence.md
├── split_manifest.csv
├── registration_qc.parquet
├── mask_manifest.parquet
├── region_spectra.parquet
├── band_qc.csv
├── parameter_contract.yaml
├── forward_model_config.yaml
├── forward_model_provenance.json
├── global_calibration.json
├── train_reference_spectrum.json
├── fixed_basis_audit.json
├── matched_baseline_comparison.parquet
├── representation_gain_retention.json
├── representation_decision.json
├── interpretation_diagnostics.json
├── frozen_stage1_spec.yaml
├── model_comparison.parquet
├── per_spectrum_fit.parquet
├── multistart_stability.parquet
├── perturbation_stability.parquet
├── wavelength_residuals.csv
├── fit_failures.json
├── identifiability_report.json
├── environment.json
├── figures/
└── STAGE1_DECISION.md
```

## 15. 实施优先级

接下来的实际开发顺序固定为：

1. `S1-0 数据合同`已实现并在全量数据上 PASS；
2. `S1-1 配准核验`已自动及人工 PASS，`transpose` 已冻结；
3. `S1-2` 已按目标部署域以 `PASS_FOR_DEVELOPMENT` 关闭，正式 mask manifest 已生成；
4. `S1-3 region_spectra.parquet`、波段 QC 和条件差异缓存已完成，主分析角色与压力角色已显式分流；
5. `S1-4 v2` 已完成六模型注册、统一变维接口、数值审计、侧别 nuisance 约束和预注册光谱敏感性，并授权 S1-5；
6. 历史 `S1-5 v2` 已只用 Train 比较 B0/B1/P2/P3-S；B0 明显更优，P2/P3-S 边界坍缩，P3-O/P4 未触发；该结果作为旧公式/旧范围的修订依据保留；
7. 修订候选体系、包装 provenance 和二次公式复核已写入第 5 节；
8. `S1-4R v3` 已完成不可覆盖的 r7 合成审计，中心化、包装/无包装极限、注册波段子集、K–M 角点、单位、资产/源码哈希、PCA 受试者隔离和解析/数值/后端/梯度一致性均通过；
9. `S1-5R v7` 已只用物理隔离的 Train 缓存完成；D2/D3 的表示增益成立，K2 候选失败。旧解释层合并 decision `REVISE_OR_STOP` 保留，不再作为表示层授权门；
10. 表示层重判 v3 已生成 `REPRESENTATION_READY_FOR_S1_6`，允许把预注册候选及匹配基线提交 S1-6；
11. `S1-6 v1` 已使用 3 名 Validation 受试者完成选择：D3 绝对误差最低，但 D2 位于 10% 近最佳区间，按低维优先冻结 D2-MH 为二维 `theta_repr`；decision 为 `S1_6_FROZEN_FOR_S1_7`。
12. `S1-7 v1` 已在 4 名 Test 受试者上唯一运行；8/8 主脸颊可用，4/4 人的 D2-MH 均优于 B0-S，9 项正式 gate 全部通过，阶段一 decision 为 `PASS`。

当前下一项工作转为阶段二设计与实现：训练 RGB 编码器预测冻结的二维 `theta_repr`，并通过 HSI/RGB 闭环检验表示可预测性和解释层行为。阶段一 D2-MH、Train 校准、mask/波长合同和适用域均须作为只读输入；Hyper-Skin Test 已使用，不得再据其重选 D1/D2/D3、修改阈值、挑波段或追加生理先验。`p016_smile_right` 继续作为既有 Validation 压力域不可用样本保留。

证据边界：目标部署域 profile 是在观察原始 Validation 的广泛门控失败后确定的，属于透明记录的事后范围收窄。它可作为 S1-3 继续开发的门控，但不能被描述为预先注册的独立 Validation PASS。若最终主张只适用于“正脸、无表情、无眼镜”，仍需新的目标域留出集或外部数据验证；Hyper-Skin Test 已按本手册唯一使用，不能再承担新的目标域选择或独立确认任务。
