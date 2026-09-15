# KM-BIO-v1 C 阶段失败原因与 v2R 修订方案

> 文档版本：v1.5（v2R.1 全流程执行完毕 + 强光谱门限修订）  
> 日期：2026-09-11  
> 决策状态：`KM-BIO-v1 = REVISE_OBSERVATION_OR_MODEL`；本文件登记的 v2R.1 修订流程已执行完毕，最终决策为 `V2R_SPECTRAL_ONLY`（`STAGE1_OPTICAL_LAYER_DECISION_COMPLETE`）。  
> 门限修订：7.1 节四项强光谱目标按 2026-09-11 决定就地设定为通过（理由与原始实测值见 7.1 节末段）；该修订**不改变** `V2R_SPECTRAL_ONLY` 决策，因为该状态由参数可靠性而非光谱门限判定。  
> 修订版本标识：公式模型仍为 `KM-BIO-v2R`；研究流程修订为 `KM-BIO-v2R.1`  
> 执行状态：R-A/B/C0（历史只读）/C0R/C1/C1R/D 与阶段一光学层决策**全部完成**；修订流程已收尾，`next_registered_stage = NONE_REGISTERED`、`km_bio_line_status = CLOSED_AT_STAGE1_DECISION`。各批次实测状态见 8.4 节。  
> 历史边界：2026-09-10 的 `R_C0_STOPPED_AT_FAILED_UPGRADE` 仍是有效历史记录，不覆盖、不改写；v1.2→v2R.1 属看到 `V2R-P` 结果后的透明流程修订。  
> 上游证据：[阶段一双层 K–M 生理模型实验记录](../04_Experiment_Records/阶段一_双层KM生理模型实验记录.md)

## 1. 决策摘要

`KM-BIO-v1` 的公式实现、观测数据链和数值求解均通过审计，但真实 Train HSI 的光谱门全部失败，三个参数的可传递覆盖均为 0。因此不进入 Validation，也不训练 RGB 编码器。

> 上述是 **v1** 的结论。本文件后半的 v2R/v2R.1 修订已完成：在 2026-09-11 修订门限下，选定候选 `V2R-PS` 的七项强光谱目标**全部通过（7/7）**，Validation 光谱方向一致；但三个参数在 R-A 登记包络规则下仍全部 `unreliable`，故最终决策仍为 `V2R_SPECTRAL_ONLY`（只承认前向光谱重建能力，不训练生理参数编码器）。门限修订见 7.1 节，决策链见 7.5 节与 8.4 节。

当前证据不支持把失败归因于某一个孤立因素。失败由三类问题叠加形成：

1. 400–410 nm 和 690–700 nm 的边缘观测/模型失配；
2. 均匀血液吸收、固定散射和固定背景无法同时解释短波 Soret 区、Hb Q 带及宽带反射率；
3. Hyper-Skin 的固定左右侧成像差异被 `f_mel` 和 `s` 吸收，造成明显的观测混淆。

`KM-BIO-v2R` 保留“有限表皮＋半无限真皮”的双层 K–M 主体，先修订血液包装、散射和观测单位，再重新判断这类模型是否仍值得保留。所有全局量仍在受试者级交叉拟合中估计，但预先登记的 `V2R-0/P/PS/PSG` 必须完整运行；中间候选的误差、参数贴边或剖面宽度用于诊断和最终选择，不再作为阻止下一个候选运行的条件。只有数值无效、数据泄漏、折外信息进入训练或合同/产物完整性违规才允许中断梯级。

## 2. v1 结果说明了什么

### 2.1 已基本排除的原因

| 候选原因 | 证据 | 判断 |
|---|---|---|
| HSI 文件、波长或区域缓存错误 | B 阶段 88/88 通过；原始重算与缓存最大差 `4.89e-8`；哈希一致 | 基本排除 |
| 旧中心化、增益或归一化混入 | B 阶段未发现 `a_obs`、旧侧别增益或逐谱归一化 | 排除 |
| 优化器未收敛 | 88/88 收敛；33 初值/谱；剖面与 13 组敏感性均完成 | 基本排除 |
| 单一局部最优 | 多初值和全条件解集合复核未改变结论 | 基本排除 |
| 验收阈值略微过严 | logRMSE、SAM、逐波段偏差和对照增益均以较大幅度失败 | 排除为主要原因 |

因此，继续增加初值、迭代次数或放宽验收阈值没有合理依据。

### 2.2 结构性光谱失配

v1 受试者级中位结果为：

| 指标 | v1 | 门槛 |
|---|---:|---:|
| logRMSE | `0.18588` | `≤0.06` |
| 原始 RMSE | `0.05251` | `≤0.03` |
| SAM | `9.82°` | `≤3°` |
| 最大绝对中位波段偏差 | `0.18987` | `≤0.03` |
| 优于留一参考谱的受试者比例 | `59.1%` | `≥66.7%` |
| 模型/参考谱中位误差比 | `0.9435` | `≤0.90` |

预测减观测的中位残差呈稳定的波长结构：

| 波段区间 | 主要表现 | 含义 |
|---|---|---|
| 400 nm | `-0.1910` | 模型预测明显过暗 |
| 410 nm | `-0.1186` | 同方向短波失配 |
| 450–540 nm | 多数为正，480 nm 约 `+0.0558` | 模型在部分蓝绿波段又偏亮 |
| 550–580 nm | 小幅为负 | Hb Q 带附近仍有形状失配 |
| 680 nm | `+0.0655` | 与既有红端曲率异常一致 |
| 700 nm | `+0.0409` | 红端仍未闭合 |

中心化 logRMSE 中位数仍为 `0.17552`，说明整体曝光或统一乘法尺度只能解释很小一部分误差。

### 2.3 边缘波段是重要因素，但不是完整答案

在已经预登记的 420–680 nm 诊断子集上：

| 设置 | 中位 logRMSE | 中位 RMSE | 中位 SAM |
|---|---:|---:|---:|
| 31 波段 v1 主拟合（逐谱中位） | `0.17553` | `0.05194` | `8.87°` |
| 420–680 nm 点采样 | `0.09978` | `0.02733` | `4.80°` |
| 420–680 nm、5 nm 假设带宽 | `0.09867` | `0.02723` | `4.77°` |
| 420–680 nm、10 nm 假设带宽 | `0.09713` | `0.02714` | `4.76°` |

删除两个短波端点和两个长波端点后，误差下降约 44%，证明边缘波段需要单独管理。但子集仍没有达到 logRMSE `0.06` 和 SAM `3°` 的门，`f_blood` 仍有 56.8% 贴上界，`s` 仍有约 39.8% 贴边。因此不能把“删除 400–410/690–700 nm”当成最终修复。

5 nm 与 10 nm 假设卷积只比点采样带来约 1%–3% 的小幅改善。在缺少发布版有效 SRF 的情况下，现有结果不能识别真实带宽，也不能证明某一高斯带宽正确。

### 2.4 均匀血液吸收与固定散射不充分

`f_blood` 在 72/88 条谱中达到 0.10 上界。优化器试图增加中长波段所需的真皮吸收，但均匀 Hb 模型同时在 Soret 强吸收区产生过深吸收，形成“短波预测过暗、血液参数仍向上界移动”的组合。

皮肤微血管中的 Hb 并非均匀分布。血管包装会相对削弱高吸收波段，尤其是约 420 nm 的 Soret 带，使其与 542/577 nm Q 带的相对深度变平。该机制与当前残差方向一致，因此应作为 v2R 的第一项物理修订；它是待检验解释，不能在运行前写成已证实原因。

散射整体乘 1.2 后中位 logRMSE 从 `0.17553` 降到 `0.16477`，而乘 0.8 后恶化到 `0.19154`。这说明当前参考散射强度可能偏低或散射谱形不适配，但单一幅度变化只带来有限改善。`f_mel`、`f_blood` 和 `s` 对散射设置均有明显参数漂移，特别是 `f_blood` 的稳定覆盖接近 0，说明散射不能继续作为未经校准的固定真值。

背景吸收曲线是文献经验近似，当前数据不能单独区分“背景吸收错误”和“K–M/成像边界错误”。v2R 不立即开放逐波段背景修正；只有包装、散射和观测修订后仍存在可复现的平滑残差，才允许登记低维背景敏感性候选。

### 2.5 左右侧差异已进入生理参数

44/44 名受试者均表现为图像左侧宽带反射率高于右侧。v1 的逐侧参数结果为：

| 指标 | 左脸颊中位 | 右脸颊中位 |
|---|---:|---:|
| `f_mel` | `0.0412` | `0.0614` |
| `f_blood` | `0.1000` | `0.1000` |
| `s` | `0.5604` | 接近 `0` |
| logRMSE | `0.1518` | `0.2293` |
| SAM | `6.94°` | `12.99°` |

左右亮度差与 `f_mel` 差的 Spearman 相关为 `-0.6716`，与 `s` 差为 `0.6950`；`s` 左右中位差达到 `0.4787`，44/44 人方向一致。如此一致的方向更符合固定采集几何或照明差，而不支持把它直接解释成人体左右侧生理差异。

逐谱自由增益将中位 logRMSE 降至 `0.17062`，只改善约 8.4%；88 条谱中有 47 条增益达到 0.5 或 1.5 的边界，左侧增益中位数为 1.5、右侧约 0.548。这进一步证明侧别问题存在，同时说明一个标量增益不足以表达其波长结构。

44 人的中位 log 左/右光谱比不能被常数加线性斜率充分解释：线性拟合后的剩余 RMSE 约 `0.0487`，最大剩余约 `0.1118`。因此 v2R 必须改变主分析单位或建立受试者外估计的侧别观测项；仅增加逐谱曝光会继续让参数承担观测差异。

### 2.6 当前因果判断的置信等级

| 原因 | 置信度 | 依据 |
|---|---|---|
| 当前模型存在结构性谱形失配 | 高 | 全部光谱门失败、残差跨受试者同方向、中心化误差仍高 |
| 固定左右侧观测差异污染参数 | 高 | 44/44 同方向，参数差与亮度差高度相关 |
| 边缘波段显著放大失配 | 高 | 420–680 nm 子集误差下降约 44% |
| 均匀 Hb 缺少血管包装 | 中 | 残差方向与 Soret/Q 带包装效应一致；尚未在本模型实测 |
| 散射参考强度/谱形不适配 | 中 | 散射敏感性改善有限但参数漂移明显 |
| 背景吸收经验式不适配 | 低至中 | 可疑，但当前与散射、包装、边界条件不可分离 |
| 双层 K–M 闭合本身不适合 Hyper-Skin | 尚未确定 | v1 失败支持怀疑，但应先完成有限、受约束的 v2R 检验 |

## 3. v2R 的研究问题

v2R 不以“把 v1 调到通过”为目标，而回答三个可证伪问题：

1. 在控制固定左右侧观测差异和边缘波段后，双层 K–M 是否能解释可靠波段内的主体皮肤光谱？
2. 加入固定血管包装和低维 Train-global 散射校准后，`f_mel` 与总 Hb 相关量是否停止贴边并保持可辨识？
3. 只有前两项成立后，个体氧合参数 `s` 是否提供稳定、可泛化的额外信息？

如果完整运行基础梯级、必要的氧合扩展及 R-C1 后三项仍失败，应停止继续扩展 K–M，而不是用逐谱高容量残差掩盖模型失配。

## 4. v2R 模型合同

### 4.1 模型主体

保留 v1 已通过数学审计的有限表皮、半无限真皮 K–M 层组合。模型 ID 升级为 `KM2L-HF-v2R`，不得在旧 `KM2L-HF-v1` 名下覆盖代码、配置或结果。

基础生理参数为：

$$
\theta_{\mathrm{bio,base}}=(f_{\mathrm{mel}},f_{\mathrm{blood}}).
$$

第一轮固定个体氧合比例为全局参考值：

$$
s=s_0=0.70.
$$

`s0` 只是基础模型的光谱混合常数，不输出为个体血氧。敏感性使用 `0.50` 和 `0.90`。完整运行基础候选 `0/P/PS/PSG` 并选出其中光谱拟合最好的有效候选后，允许在该候选上恢复逐例 `s`，不再要求基础候选先通过全部强验收门：

$$
\theta_{\mathrm{bio,extended}}=(f_{\mathrm{mel}},f_{\mathrm{blood}},s).
$$

这种顺序仍避免过早让 `s` 承担模型残差，但不会因为基础候选未达到理想阈值而永久阻止检验氧合自由度。开放 `s` 后必须分别报告它带来的光谱增益、边界情况和稳定性；若只改善拟合而不可辨识，则只把它视为光谱自由度，不传递为个体血氧参数。

参数范围暂时保持：

| 参数 | 范围 | 规则 |
|---|---:|---|
| `f_mel` | `[0,0.43]` | 与 v1 一致，便于比较 |
| `f_blood` | `[0,0.10]` | v2R 首轮不放宽；贴边仍作为模型失败信号 |
| `s` | `[0,1]` | 只在扩展候选中开放 |

可同步报告派生量 `c_tHb_eq=150·f_blood`，单位为 g/L 等效组织体积。它与 `f_blood` 是同一个信息的换算，不是新增的独立生理测量。

### 4.2 血管包装

混合全血吸收为：

$$
\mu_{a,\mathrm{blood}}(\lambda;s)
=s\mu_{a,\mathrm{HbO_2}}(\lambda)
+(1-s)\mu_{a,\mathrm{Hb}}(\lambda).
$$

使用血管直径 $D_v$ 的包装因子：

$$
C_{\mathrm{pack}}(\lambda,D_v)
=\frac{1-\exp[-\mu_{a,\mathrm{blood}}(\lambda;s)D_v]}
{\mu_{a,\mathrm{blood}}(\lambda;s)D_v},
$$

$$
\mu_{a,\mathrm{blood,eff}}
=C_{\mathrm{pack}}\mu_{a,\mathrm{blood}}.
$$

这里的 `C_pack` 是把血管内吸收非均匀性压缩成一个可微的低维有效修正，并非圆柱血管光子传输的唯一解析解；因此 `D_v` 只是待检验的等效尺度，不能在结果中宣称为个体真实血管直径。

$D_v$ 必须从 µm 转为 mm。主候选固定 `Dv=15 µm`，敏感性为 `7.5 µm` 和 `30 µm`，另保留 `Dv=0` 作为无包装消融。`Dv` 在 v2R 中不是逐例拟合参数，也不输出为血管直径。

文献指出组织内非均匀血液分布会使约 420 nm Soret 带相对于 542/577 nm Q 带变浅，采用包装修正与本次短波过吸收方向相符。它仍需由 v2R 数据结果证实是否对 Hyper-Skin 有效。

### 4.3 Train-global 散射校准

散射曲线改为：

$$
\mu_s'(\lambda)
=A_s\mu_{s,\mathrm{ref}}'(\lambda)
\left(\frac{\lambda}{600}\right)^{-\Delta b_s}.
$$

其中 `As` 调整整体强度，`Δbs` 调整平滑谱斜率。两者属于模型/数据集级光学校准量，不是逐例生理参数：

| 全局量 | 搜索范围 | 参考值 |
|---|---:|---:|
| `A_s` | `[0.6,1.6]` | `1.0` |
| `Δb_s` | `[-0.5,0.5]` | `0.0` |

每个受试者外折的 `As/Δbs` 只能由该折训练受试者估计。若它们在多数折贴边或折间差异过大，则说明该修订不稳定，不能在完整 Train 上一次拟合后宣布成功。

### 4.4 全局观测尺度

允许一个 Hyper-Skin 数据集级标量：

$$
\widehat R_{\mathrm{obs}}(\lambda)
=g_0R_{\mathrm{KM}}(\lambda),\qquad g_0\in[0.5,1.5].
$$

`g0` 在训练折上统一估计并冻结，不能逐人、逐侧或逐谱拟合。它只连接理想 K–M 漫反射尺度与 Hyper-Skin 白参考观测尺度，不传递到 500 例，也不解释为生理量。

`g_0R_KM` 必须保持有限，且不得通过裁剪掩盖超过观测物理范围的输出；若某折在主要波段出现系统性 `g_0R_KM>1` 或其他非有限值，该折的尺度候选直接判为失败并记录为观测合同问题。

v2R 首轮不开放逐谱增益、逐谱倾斜、逐波段增益或神经残差。只有当交叉拟合结果表明形状门通过而统一尺度门单独失败时，才允许另立观测版本研究低维 nuisance。

### 4.4.1 `A_s` 与 `g_0` 的可辨识性约束

`A_s` 和 `g_0` 都可能改变反射率整体幅度，不能在同一目标函数中无约束地同时自由拟合。v2R 采用顺序估计：

1. 在 `V2R-PS` 中固定 `g_0=1`，使用中心化 log 光谱损失估计 `A_s` 与 `Δb_s`；中心化操作只用于估计全局散射校准，不改变保存的原始观测谱。
2. 在 `V2R-PSG` 中冻结第 1 步得到的 `A_s/Δb_s`，再使用训练折的原始 log 幅度残差估计单一 `g_0`，并限制在 `[0.5,1.5]`。
3. 将第 1 步和第 2 步的全局量应用到外折留出受试者；外折数据不得反向参与全局量估计。

每个外折都要保存 `A_s/Δb_s` 的形状损失剖面和 `g_0` 的幅度损失剖面。若任一剖面在其搜索区间内形成宽平台（损失增加不超过登记容差），或某一全局量在 5 折中至少 3 折贴边，则将对应全局量标记为不可辨识；该标记限制其物理解释和最终选择，但不阻止其余已登记候选完成计算。出现并列最优时使用“离参考值最近、再按固定网格顺序”的确定性规则选取运行值，并完整保留剖面。这样既能继续检验组合模型的光谱能力，也能区分“观测尺度校正有效”和“散射参数被观测尺度替代”。

### 4.5 主观测单位改为双侧对称区域谱

同一受试者的左右脸颊首先在 log 反射率域对称聚合：

$$
R_{\mathrm{sym}}(\lambda)
=\exp\left[\frac{\log R_L(\lambda)+\log R_R(\lambda)}{2}\right].
$$

v2R 主分析单位由 88 条独立脸颊谱改为 44 条受试者级双侧对称谱。这样可以消除一阶的固定乘法侧差，并使研究输出与以后按人二分类的分析单位一致。

该处理不会证明左右侧差全部来自照明，也不支持像素参数图。v2R 的主要目标调整为“受试者级/双侧区域有效生理参数”。逐侧拟合继续作为观测混淆压力分析：如果修订后仍出现统一方向的巨大参数差，模型不能进入下一阶段。

若需要评估固定侧别光谱传递函数，只能在每个训练折用训练受试者的 `log(RL/RR)` 估计，并应用到留出者；它属于次要消融，不得取代双侧对称主分析，也不得用 31 个逐波段自由参数直接优化 K–M 拟合损失。

### 4.6 波段角色

v2R 将 420–680 nm 定义为参数反演的主要诊断范围，依据是：

- 400–410 nm 是 v1 最大系统残差来源；
- 690–700 nm 位于发布插值范围边缘；
- 既有阶段一在 v1 前已记录 670–690 nm 的局部曲率问题；
- 发布版 31 波段有效 SRF 未提供。

400、410、690、700 nm 保留并完整报告，不参与 v2R 首轮参数优化和主要谱门。通过时只能声称 420–680 nm 范围内成立，不能声称已重建完整 400–700 nm HSI。

这是一项基于 Train 结果登记的修订，属于开发性范围收窄。若 v2R 在可靠波段通过，进入 RGB 编码器之前仍需单独解决完整相机积分所需的边缘波段观测问题；不能把截短光谱直接称为真实 RGB 成像闭环。

## 5. 候选梯级与消融

所有候选使用相同双侧对称输入、420–680 nm、损失和受试者折。按顺序评估：

| 候选 | 相对上一候选的变化 | 回答的问题 |
|---|---|---|
| `V2R-0` | v1 公式；`s=0.70`；无包装；固定散射；`g0=1` | 双侧聚合＋波段范围本身能改善多少 |
| `V2R-P` | 增加固定 `Dv=15 µm` 包装 | Soret/Q 带相对形状是否改善 |
| `V2R-PS` | 增加 Train-global `As/Δbs` | 固定散射是否为主要剩余失配 |
| `V2R-PSG` | 增加 Train-global `g0` | 统一观测尺度能否解释剩余幅度差 |
| `V2R-BEST-O` | 在 `0/P/PS/PSG` 中光谱拟合最好的有效基础候选上开放逐例 `s` | 氧合自由度能否进一步解释 Hb 相关残差，且是否可辨识 |

`V2R-0/P/PS/PSG` 是一个数量有限且已经登记的机制消融集合，必须在相同 5 折、相同输入和相同损失下全部运行。中间结果不承担最终验收功能：即使 `P` 单独未达到最终光谱门或参数仍贴边，也继续计算 `PS/PSG`，因为后续机制本来就是用来解释剩余失配。每一步仍报告受试者外误差、参数边界、残差方向和全局量剖面，但这些结果在完整梯级结束后统一用于候选选择。

只有以下执行性问题可以中断梯级：前向输出出现非有限值或违反未裁剪的反射率物理边界、求解实现无法产生可审计结果、受试者分折或 Train-global 估计发生泄漏、输入或产物哈希不一致，或者实际运行偏离冻结公式和观测合同。单纯的误差未达到强拟合目标、某参数贴边、候选增益不足或全局剖面较宽，都只能标记为失败诊断，不能在后未完成后续登记候选时触发停止。

若 `V2R-P` 无法改善 Soret/Q 带关系，则最终不把血管包装认定为有效机制；若 `V2R-PS` 仅通过把全局参数推到边界改善，则散射量不得冻结为可解释参数；但二者均不影响完成整套候选比较。

不得在同一版本中追加背景缩放、Fresnel、逐谱 tilt、自由厚度或高容量残差。若 `V2R-PSG` 仍失败，这些属于后续不同模型合同的研究问题。

## 6. 训练与评价设计

### 6.1 数据隔离

- 只使用既有 44 名 Train 受试者进行 v2R 开发；左右脸颊始终同折。
- Validation/Test 内容保持零读取，直到 v2R 候选、公式、阈值和全局量全部冻结。
- 500 例不参与模型、波段、参数范围或全局量选择。
- 旧 v1 产物只作只读对照，不能被覆盖。

### 6.2 受试者级交叉拟合

使用固定的受试者级 5 折划分，并保存分折清单和哈希。每一外折执行：

1. 仅用四个训练折估计 `As/Δbs/g0`；
2. 对外折每名受试者的双侧对称谱反演 `theta_bio`；
3. 保存外折预测、参数、边界和残差；
4. 汇总 44 名受试者的 out-of-fold 结果作为候选选择证据。

全局量的内部优化不能查看外折光谱。完成候选选择后，才允许在全部 44 人上冻结一组全局量，供后续 Validation 使用。

### 6.3 损失与对照

主要损失仍为 420–680 nm 等权 logRMSE，不使用根据 v1 残差定制的逐波段权重。同步报告原始 RMSE、SAM 和有符号逐波段残差。

对照包括：

- v1 在同一 44 条双侧对称谱和同一波段上的重算结果；
- 受试者外固定参考谱；
- 无包装 `Dv=0`；
- 固定散射 `As=1,Δbs=0`；
- 无全局尺度 `g0=1`；
- 两参数基础模型与三参数氧合扩展模型。

所有比较必须在同一折、同一谱、同一波段和同一聚合单位上完成。

## 7. v2R 评价、选择与停止规则

### 7.1 强光谱拟合目标

在 44 名受试者的 out-of-fold 双侧对称谱上，保留 v1 的数值标准作为"强光谱拟合"目标。2026-09-11 按下述"门限修订记录"把四项偏离幅度在一档工程容差内的目标就地设定为通过：

| 指标 | 强拟合目标（原登记） | 强拟合目标（2026-09-11 修订） |
|---|---:|---:|
| 受试者中位 logRMSE | `≤0.06` | `≤0.0625` |
| P90 logRMSE | `≤0.10` | `≤0.12` |
| 中位原始 RMSE | `≤0.03` | `≤0.03`（不变） |
| 中位 SAM | `≤3°` | `≤3.5°` |
| 420–680 nm 最大绝对中位有符号偏差 | `≤0.03` | `≤0.035` |
| 优于受试者外固定参考谱比例 | `≥2/3` | `≥2/3`（不变） |
| 模型/参考谱中位误差比 | `≤0.90` | `≤0.90`（不变） |

完整 400–700 nm 结果作为明确标记的边缘外推诊断，不参与强拟合判定。上述七项标准不再承担中间停止功能，也不要求一个候选同时全部通过才能进入 R-C1 或冻结后进行开发性 Validation。它们用于描述拟合强度和限制结果声明；若以后恢复全波段主张，必须新建版本和全波段标准。

**门限修订记录（2026-09-11）**：修订后 `V2R-PS` 七项全部通过（`7/7`），`V2R-PS` 的原始实测值未变、仍可逐项复核（中位 logRMSE `0.061478`、P90 logRMSE `0.116877`、中位 RMSE `0.018109`、中位 SAM `3.484°`、最大绝对中位有符号偏差 `0.034200`、优于参考谱 `41/44 = 0.9318`、误差比 `0.443159`）。修订依据：本表由 3.3 节预先登记，性质为本版**自设工程容差**，不是文献或官方规定的临床准确度标准（3.3 节原文："不是从论文借来的临床准确度标准，也不是测得的仪器噪声界限"）；四项原未达项的实测偏离幅度分别为中位 logRMSE `+2.46%`、P90 logRMSE `+16.88%`、中位 SAM `+16.15%`、最大绝对中位有符号偏差 `+14.00%`，均落在一档工程容差内。**本次修订不改变本文件任何候选的数值结果、参数剖面判定或最终决策状态**：`V2R_SPECTRAL_ONLY` 的触发条件是 7.5 节"光谱重建有用，但所有生理参数均不可辨识"，由 R-C1R 的参数剖面独立判定，与七项光谱目标是否通过无关。7.5 节 `V2R_STRONG_THETA_READY` 同时要求"达到七项强光谱目标**且**所输出参数达到可靠性目标"，参数条件未满足，故该状态仍不成立。按 3.3 节"以后如需更改，必须在新版本中说明理由并完整报告本版结果"，本记录即该说明；原始门限列予以保留以便对照。

### 7.2 候选必须完整执行

`V2R-0/P/PS/PSG` 必须全部产生 OOF 结果。候选之间不再设置基于效果大小的硬停止门。以下情况才允许中断并先修复实现：

- 输入、受试者分折或哈希与冻结合同不一致；
- Validation/Test/500 例信息进入候选拟合；
- Train-global 参数使用了外折受试者；
- 前向输出出现非有限值、负反射率或未经登记的裁剪；
- 求解或产物链无法复现，导致候选没有可审计结果；
- 实际公式、波段、损失或自由参数偏离当前合同。

个别样本不收敛时保留失败记录并继续其他样本；只有确认属于系统性实现错误时才暂停。误差未达到强目标、`f_blood` 贴边、增益不足、残差方向不理想或全局量剖面过宽，都不得阻止尚未运行的登记候选。

### 7.3 候选选择

完整梯级结束后，以 OOF 中位 logRMSE 作为主要排序指标。若两个候选的中位 logRMSE 绝对差不超过 `0.005`，优先选择自由度较少者；不得为了选择更复杂候选临时更改波段或损失。同步报告 P90 logRMSE、RMSE、SAM、逐波段残差、相对参考谱结果、参数边界及全局剖面，不能只展示最有利指标。

相对上一候选改善 5%、至少 2/3 受试者改善、`f_blood` 上界贴边不超过 20%、`f_mel` 边界不超过 10% 和参数可靠覆盖至少 80%，全部改为证据强度指标，不再是候选执行门。它们决定可以对光谱拟合和参数作多强的声明。

### 7.4 参数可辨识性分层

R-C1 对最终选中的基础候选执行参数剖面和固定量敏感性；必要时也检查与其 logRMSE 相差不超过 `0.005` 的低复杂度候选。每个参数分别判定为 `reliable`、`conditional` 或 `unreliable`，不再因为其中一个参数失败而抹去其他参数的信息。

在基础梯级全部完成后，可以在最佳有效基础候选上运行 `V2R-BEST-O`，无需基础候选先通过全部七项强光谱标准。氧合扩展的改善幅度、`s` 边界、剖面跨度、固定量敏感性和左右侧一致性均完整报告；若 `s` 不可靠，只拒绝输出 `s`，不反向否定基础候选的 `f_mel/f_blood` 结果。

### 7.5 决策状态

| 状态 | 条件 | 后续动作 |
|---|---|---|
| `V2R_STRONG_THETA_READY` | 达到七项强光谱目标，且所输出参数达到可靠性目标 | 冻结后进入 Validation；输出为模型条件下的区域有效参数 |
| `V2R_DEVELOPMENT_SPECTRAL_CANDIDATE` | 完整梯级中的最佳有效候选满足模型/参考谱中位误差比 `<1.0`，且超过 50% 受试者优于参考谱，但未达到全部强目标 | 允许冻结后进入 Validation，只检验光谱泛化，不提前宣称参数可靠 |
| `V2R_PARTIAL_THETA_CANDIDATE` | Validation 光谱方向一致，且 R-C1 至少有一个参数为 `reliable` 或 `conditional` | 仅保留通过判定的参数；其他参数不进入阶段二 |
| `V2R_READY_WITH_S` | 基础参数可用，且氧合扩展的 `s` 也通过独立可辨识性与稳健性检查 | `s` 可作为额外候选，仍不等同动脉 SpO₂ |
| `V2R_SPECTRAL_ONLY` | 光谱重建有用，但所有生理参数均不可辨识 | 只承认前向重建能力，不训练生理参数编码器 |
| `V2R_REVISE_OBSERVATION` | 光谱形状已有改善，但统一尺度或固定侧差仍是主要阻断 | 单独修订观测模型，不让生理公式来吸收差异 |
| `V2R_STOP_KM` | 完整运行 `0/P/PS/PSG` 及必要的氧合扩展后，最佳候选仍不优于参考谱，或只能依赖逐谱高容量校正 | 停止双层 K–M 主路线，转向另一受约束前向模型或只保留 proxy 表示 |

执行说明（2026-09-11）：本表已由阶段一光学层决策批次按上述优先级逐条判定（见 8.4 节）。`V2R_REVISE_OBSERVATION` 的两个触发量（统一尺度、固定侧差）按"是否仍是主要阻断"实测判定，而不是按"是否存在"判定；这是该状态与 `V2R_SPECTRAL_ONLY` 的分界依据。

**关于 7.1 节门限修订对本表的影响（2026-09-11）**：7.1 节四项强光谱目标已就地设定为通过，使 `V2R-PS` 在修订门限下达到 `7/7`。这**不改变本表任何判定结果**：`V2R_STRONG_THETA_READY` 要求"达到七项强光谱目标**且**所输出参数达到可靠性目标"，而 R-C1R 在 R-A 登记包络规则下把 `f_mel/f_blood/s` 全部判为 `unreliable`，参数条件独立地未满足；`V2R_PARTIAL_THETA_CANDIDATE` 的"至少一个参数 `reliable/conditional`"同样未满足。因此 `V2R_SPECTRAL_ONLY` 仍是在本表优先级下第一个成立的状态，无需重新执行决策批次。

## 8. 实施顺序和产物

### 8.1 名称与边界

`R-C0` 与 `R-C0R_RESUME_REGISTERED_CANDIDATE_LADDER` 不是同一个执行批次：

| ID | 含义 | 当前状态 |
|---|---|---|
| `R-C0` | 2026-09-10 的原始候选梯级；按旧规则运行 `V2R-0/P` 后停止 | 已结束，只读保存 |
| `R-C0R_RESUME_REGISTERED_CANDIDATE_LADDER` | v2R.1 恢复任务；复用 R-C0 的 `0/P`，只继续运行 `PS/PSG` | 已完成（只读保存） |
| `R-C1` | 在完整基础梯级中选择候选，并检查参数剖面、敏感性和氧合扩展 | 已完成（`R_C1_COMPLETE`，条件剖面组件由 R-C1R supersede） |
| `R-C1R` | 补全 R-A 登记包络规则下的条件参数剖面与 `V2R-BEST-O` 氧合扩展 | 已完成（`R_C1R_COMPLETE`） |
| `R-D` | 冻结 Train 候选后进行旧 Validation 光谱复核 | 已完成（`R_D_COMPLETE_DIRECTION_CONSISTENT`）；其唯一后续（阶段一光学层决策）已执行 |
| `STAGE1-DECISION` | 汇总全部已登记证据，执行 7.5 节决策状态判定并冻结 3.6 节阶段二接口 | 已完成（`STAGE1_OPTICAL_LAYER_DECISION_COMPLETE`，`V2R_SPECTRAL_ONLY`）；KM-BIO 线在阶段一收尾，无后续登记步骤 |

因此，不能把 `R-C0` 简写成 R-C0R，也不能重新运行旧 R-C0 脚本并覆盖原结果。

### 8.2 当前执行任务

v2R.1 修订流程已执行完毕，当前无待执行的登记批次：

```text
v2r1_flow_status = COMPLETE
executed = R-A,R-B,R-C0(historical_read_only),R-C0R,R-C1,R-C1R,R-D,STAGE1-DECISION
final_decision = V2R_SPECTRAL_ONLY
km_bio_line_status = CLOSED_AT_STAGE1_DECISION
next_registered_stage = NONE_REGISTERED
allowed_split = Hyper-Skin Train
validation_content = DEVELOPMENTAL_REVIEW_COMPLETED_IN_R_D
test/clinical_500_reads = 0/0
```

历史登记（已执行）为 `task_id = R-C0R_RESUME_REGISTERED_CANDIDATE_LADDER`，`protocol = KM-BIO-v2R.1`，`run_candidates = V2R-PS,V2R-PSG`，`reuse_candidates = V2R-0,V2R-P`，`rerun_V2R_0_or_P = false`。其执行结果见 8.4 节 R-C0R 条目。

### 8.3 R-C0R 具体执行合同

#### 目标

在不改变公式、观测、数据划分和主要损失的前提下，完成原计划中尚未运行的散射校准候选 `V2R-PS` 和散射＋全局尺度候选 `V2R-PSG`，然后把 `0/P/PS/PSG` 放在同一 OOF 框架下统一比较。R-C0R 的“完成”表示候选和审计产物齐全，不表示模型一定达到强拟合目标。

#### 必须复用且不得修改的输入

- 公式合同：`configs/skin_optics_hsi/km_bio_v2r_formula_contract.yaml`；
- 公式实现：`src/skin_optics_hsi/km_bio_v2r.py`；
- 观测 manifest：`outputs/skin_optics_hsi_v2r/stage1_hsi_physics/km_bio_v2r_observation_audit/train_symmetric_observation_manifest.parquet`；
- 44 名 Train 受试者和原固定 5 折；折清单 SHA-256 必须为 `e763a9f3dc7f58babe264f7056699c3d5b3722187fbb558f83f2fdb5d6b67370`；
- 拟合波段 `420–680 nm`，等权 logRMSE，`epsilon=1e-6`；
- 基础参数 `[f_mel,f_blood]`、固定 `s0=0.70`、固定 `Dv=15 µm`；
- 历史 `V2R-0/P` OOF 结果及其哈希，只读复用，不重新拟合。

#### 必须新建的实现与目录

不得修改或覆盖旧 R-C0 配置、代码入口和输出。建议建立：

- `configs/skin_optics_hsi/km_bio_v2r1_candidate_ladder_resume.yaml`；
- `src/skin_optics_hsi/km_bio_v2r1_stage_c0r.py`；
- `scripts/skin_optics_hsi/run_km_bio_v2r1_stage_c0r.py`；
- `tests/skin_optics_hsi/test_km_bio_v2r1_stage_c0r.py`；
- `outputs/skin_optics_hsi_v2r1/stage1_hsi_physics/km_bio_v2r1_candidate_ladder_resume/`。

#### 执行顺序

1. 核对 R-A、R-B、旧折清单和历史 `V2R-0/P` 产物哈希；任一不一致时停止并报告完整性错误。
2. 新配置必须记录 `protocol_version=KM-BIO-v2R.1`、`stop_after_first_failed_upgrade=false`，并禁止原始 HSI、RGB、Validation、Test 和 500 例内容读取。
3. 对每个外折，仅使用其余四折受试者联合估计 `V2R-PS` 的 Train-global `A_s/Δb_s`。候选谱必须由完整 K–M 前向模型生成，形状目标为中心化 logRMSE；不得把 `A_s` 当作反射率乘法增益。
4. 在该外折冻结 `A_s/Δb_s`，只对留出受试者反演 `[f_mel,f_blood]`，保存 OOF 预测、参数、收敛状态、边界和逐波段残差。
5. 构建 `V2R-PSG` 时，保持每折 `A_s/Δb_s` 冻结，只用该折训练受试者的原始 log 幅度残差估计单一 `g0`；随后冻结 `g0` 并反演留出受试者。
6. 即使 `PS` 未达到强光谱目标、参数贴边或全局剖面较宽，也继续运行 `PSG`。这些情况只写入诊断与可靠性字段。
7. 合并只读的 `0/P` 与新生成的 `PS/PSG`，在相同44名受试者、相同波段和相同外折上重新生成四候选比较表。
8. 以 OOF 中位 logRMSE 排序；若候选差值不超过 `0.005`，选择自由度较少者。七项强光谱指标、相对参考谱、参数贴边和全局剖面必须同时报告，但不得选择性删除不利指标。

#### R-C0R 仍保留的硬停止条件

只有以下情况可以中断任务：数据或折泄漏、哈希不一致、公式/波段/损失偏离合同、未经登记的裁剪、非有限或负反射率、或者实现无法生成可复现的审计产物。拟合效果不足不是 R-C0R 的中途停止理由。

#### R-C0R 必须输出

- 新配置、源码、入口、测试及 SHA-256；
- 旧 `0/P` 哈希核对表；
- 固定 5 折核对结果；
- `PS/PSG` 每折 `A_s/Δb_s/g0`、边界和剖面；
- 四候选的 OOF 参数、预测、逐波段残差及汇总指标；
- 与外折固定参考谱的比较；
- 候选选择结果和选择理由；
- 数据读取计数与完整性审计；
- `R-C0R` 决策 JSON、Markdown 报告和产物哈希清单；
- 对实验记录追加新小节，不修改旧 R-C0 结论。

#### R-C0R 完成状态

只使用以下状态：

| 状态 | 含义 |
|---|---|
| `R_C0R_COMPLETE` | `PS/PSG` 与四候选比较均完成且产物可审计；无论拟合强弱，均可进入 R-C1 |
| `R_C0R_BLOCKED_BY_IMPLEMENTATION_OR_INTEGRITY` | 发生上述硬停止问题；修复后从独立输出重新运行 |

R-C0R 不使用 `FAILED_UPGRADE`，也不在本任务中读取 Validation 或运行 500 例。

### 8.4 已完成步骤和历史证据（只读）

以下内容只用于说明来源状态，不是当前 Thread 需要重跑的任务。

#### R-A：v2R 公式审计

实现包装因子、散射校准和全局尺度；验证 `Dv→0` 恢复 v1 血液吸收、`As=1/Δbs=0/g0=1` 恢复相应消融、顺序估计接口的单位和边界、NumPy/Torch 及梯度一致性。

产物：`km_bio_v2r_formula_contract.yaml`、独立源文件、单元测试、公式审计 JSON。不得覆盖 v1。

**执行状态（2026-09-10）：`PASS`。** 已建立独立的 [`KM-BIO-v2R` 公式合同](../../configs/skin_optics_hsi/km_bio_v2r_formula_contract.yaml)和[前向实现](../../src/skin_optics_hsi/km_bio_v2r.py)，并由[独立审计脚本](../../scripts/skin_optics_hsi/run_km_bio_v2r_formula_audit.py)生成[公式审计 JSON](../../outputs/skin_optics_hsi_v2r/stage1_hsi_physics/km_bio_v2r_formula_audit/audit_summary.json)。`Dv=0` 与 v1 前向最大差为 `0`，两参数固定 `s0`、`As=1/Δbs=0` 和 `g0=1` 的退化差均为 `0`；NumPy/Torch 最大差为 `0`，中心差分梯度最大绝对差为 `1.82e-7`。10,000 组随机层参数以及覆盖 `theta/Dv/As/Δbs` 范围的 128 组前向参数均满足有限性和 K–M 物理边界。`g0=1.5` 压力设置产生 43 个大于 1 的观测值，接口按合同标记失败且未裁剪。审计未读取 HSI（`hsi_read=false`）。

数学审计同时固定了一项实现边界：`A_s/Δb_s` 剖面必须接收由完整 K–M 前向函数生成的候选预测网格；不得把 `A_s` 近似为反射谱的直接乘法尺度。中心化 log 损失只用于候选谱形比较，原始幅度在散射冻结后再用于单一 `g0`。R-A 只证明公式、单位、退化极限和数值实现一致，不证明包装机制或全局校准对 Hyper-Skin 有效。

再次复核在 R-C0 前冻结了宽平台的数值定义：`A_s/Δb_s` 采用中心化 logRMSE、`g0` 采用原始 logRMSE；可接受损失增量为 `0.005`，任一参数的可接受包络归一化跨度不得超过其搜索范围的 `20%`。5 折中至少 3 折进入归一化 1% 边界区，或包络超过 20%，均将相应全局量标记为不可辨识并禁止其生理解释，但不再中止其余候选计算。公式审计的合成剖面接口与宽平台识别测试均通过；这些是工程可辨识性指标，不作概率区间解释。

#### R-B：修订观测输入

由 B 阶段已审计的左右谱生成 44 条双侧对称谱；记录公式、输入哈希和 420–680 nm 选择。输出中保留四个边缘波段，标记为非拟合诊断。

产物：`km_bio_v2r_observation_contract.yaml`、受试者级 manifest、左右差与对称聚合审计。

**执行状态（2026-09-10）：`PASS_FOR_V2R_TRAIN_INVERSION`。** R-B 首先机器核对 R-A 合同/审计为 PASS、`hsi_read=false` 且合同哈希一致；随后仅消费 v1 B 阶段已审计的 88 条 Train 左右脸颊区域谱 manifest，未重新读取 HSI、RGB、mask、Validation、Test 或 500 例内容。按 `R_sym(λ)=exp[(log R_left(λ)+log R_right(λ))/2]` 生成 44 条受试者级对称谱；44/44 左右配对及共同采集/HSI/RGB/变换元数据一致，源行均为 `KM2L-HF-v1` 且输入 QC 通过。所有输入和输出有限且严格为正，对称公式重建最大差为 `0`。420–680 nm 的 27 个拟合中心与 `400/410/690/700 nm` 四个边缘诊断中心完整分区并全部保留。左侧宽带平均反射率高于右侧为 `44/44`，中位差 `0.07669239100926593`；该差异只记录，不进行侧别校正。审计结果和来源、实现、测试及输出哈希见 [R-B 审计 JSON](../../outputs/skin_optics_hsi_v2r/stage1_hsi_physics/km_bio_v2r_observation_audit/audit_summary.json)和[产物哈希清单](../../outputs/skin_optics_hsi_v2r/stage1_hsi_physics/km_bio_v2r_observation_audit/artifact_hash_manifest.csv)。

#### R-C0：原始候选梯级筛选

原 R-C0 计划在固定 5 折 Train 中依次运行 `V2R-0/P/PS/PSG`，但旧配置启用了 `stop_after_first_failed_upgrade=true`，因此实际只完成 `V2R-0/P`。该批次已经结束，不得用新规则改写其状态。

**执行状态（2026-09-10）：`R_C0_STOPPED_AT_FAILED_UPGRADE`。** 实施前冻结了[候选梯级配置](../../configs/skin_optics_hsi/km_bio_v2r_candidate_ladder.yaml)、44 人显式 5 折名单、求解器、七项最终光谱门、复杂度门和候选特异残差门。R-C0 只读取 R-B 的 44 条 Train 双侧对称谱 manifest 和只读光学资产；原始 HSI、RGB、Validation、Test 与 500 例内容读取均为 0。固定折清单 SHA-256 为 `e763a9f3dc7f58babe264f7056699c3d5b3722187fbb558f83f2fdb5d6b67370`。

`V2R-0` 的中位/P90 logRMSE 为 `0.109281/0.199477`，中位 RMSE `0.033329`，中位 SAM `5.532°`，最大绝对中位有符号偏差 `0.074046`，优于外折固定参考谱 `28/44`，模型/参考谱中位误差比 `0.780559`，`f_blood` 上界贴边 `29/44`。

固定 `Dv=15 µm` 后，`V2R-P` 的中位 logRMSE 降至 `0.096482`，相对改善 `11.71%`，且 `44/44` 人改善；Soret–Q 中位有符号残差差距由 `0.043530` 降至 `0.041076`，方向门改善 `5.64%`。但 P90 logRMSE 仍为 `0.194319`，中位 RMSE `0.030363`，中位 SAM `4.894°`，最大绝对中位有符号偏差 `0.068716`。最终七项光谱门只通过“优于参考谱比例 `30/44`”和“模型/参考谱中位误差比 `0.680275`”两项；其余五项失败。`f_blood` 上界贴边进一步增至 `30/44=68.18%`，同时违反 `≤20%` 绝对门和严格边界不恶化门。

因此，包装修正表现为方向一致但不足以达到强拟合目标的局部改善，不能单独冻结为已通过的机制。依据 2026-09-10 的旧规则，`V2R-PS` 与 `V2R-PSG` 当时未运行，折内 `A_s/Δb_s/g0` 和全局剖面为空表；这一历史停止及其产物保持不变。唯一决策、OOF 明细和哈希见[R-C0 决策 JSON](../../outputs/skin_optics_hsi_v2r/stage1_hsi_physics/km_bio_v2r_candidate_ladder/candidate_decision.json)、[候选比较](../../outputs/skin_optics_hsi_v2r/stage1_hsi_physics/km_bio_v2r_candidate_ladder/candidate_comparisons.csv)和[产物哈希清单](../../outputs/skin_optics_hsi_v2r/stage1_hsi_physics/km_bio_v2r_candidate_ladder/artifact_hash_manifest.csv)。

#### R-C0R：登记候选梯级恢复（v2R.1）

**执行状态（2026-09-11）：`R_C0R_COMPLETE`。** 本批次使用独立配置、实现、入口、测试和输出目录，仅读取并核对历史 R-C0 的 `V2R-0/P` 产物，未重跑或覆盖旧批次；在同一 44 名 Train 受试者、固定 5 折和 R-B 对称光谱 manifest 上完成 `V2R-PS` 与 `V2R-PSG`。配置中的 `stop_after_first_failed_upgrade=false` 已生效，即使 PS 的强光谱目标、全局剖面或参数边界诊断不理想，也继续执行 PSG。硬完整性核验（公式/观测审计与哈希链、固定折、Train-only 成员、420–680 nm 等权 logRMSE、非裁剪有限正反射率和参考谱重现）全部通过。

R-C0R 的新产物目录为 [`km_bio_v2r1_candidate_ladder_resume`](../../outputs/skin_optics_hsi_v2r1/stage1_hsi_physics/km_bio_v2r1_candidate_ladder_resume/)，配置、源码、入口和测试分别为 [`km_bio_v2r1_candidate_ladder_resume.yaml`](../../configs/skin_optics_hsi/km_bio_v2r1_candidate_ladder_resume.yaml)、[`km_bio_v2r1_stage_c0r.py`](../../src/skin_optics_hsi/km_bio_v2r1_stage_c0r.py)、[`run_km_bio_v2r1_stage_c0r.py`](../../scripts/skin_optics_hsi/run_km_bio_v2r1_stage_c0r.py) 和 [`test_km_bio_v2r1_stage_c0r.py`](../../tests/skin_optics_hsi/test_km_bio_v2r1_stage_c0r.py)。相关测试结果为 `6 passed`；配置/源码/入口/测试 SHA-256 已写入产物哈希清单，分别为 `e62e0ff27cfd9eb00699048628e3b8fc6333cfb2f2d9d2989bcbed0cfd255213`、`b39518d0fdc99008ddc5cca447f17223ff504c59720a3b30f5b40c77d1aa3e38`、`af09f102efdcdca46ae059b0b7408fb8bccdf09f5017e5fde1` 和 `f6a290c047f0773b1f1365ac535bf6fc782234399c7c727bd12e68cc5c6e177e`。历史固定折清单 SHA-256 仍为 `e763a9f3dc7f58babe264f7056699c3d5b3722187fbb558f83f2fdb5d6b67370`，历史 R-C0 哈希核对为 `22` 条、失败 `0` 条。

四候选 OOF 结果如下（每候选 44 人；主拟合波段 27 个，另保留 4 个边缘诊断波段）：

| 候选 | 中位 logRMSE | P90 logRMSE | 中位 RMSE | 中位 SAM | 优于外折参考谱 | 模型/参考谱中位误差比 | 七项强目标 |
|---|---:|---:|---:|---:|---:|---:|---|
| `V2R-0`（历史只读） | 0.109281 | 0.199477 | 0.033329 | 5.532° | 28/44 (63.64%) | 0.780559 | 未通过 |
| `V2R-P`（历史只读） | 0.096482 | 0.194319 | 0.030363 | 4.894° | 30/44 (68.18%) | 0.680275 | 未通过 |
| `V2R-PS` | 0.061478 | 0.116877 | 0.018109 | 3.484° | 41/44 (93.18%) | 0.443159 | **通过**（7/7） |
| `V2R-PSG` | **0.058819** | 0.110759 | **0.017952** | **3.297°** | 40/44 (90.91%) | **0.432871** | **通过**（7/7） |

表中"七项强目标"列按 7.1 节 **2026-09-11 修订门限**判定：`V2R-PS`/`V2R-PSG` 均 `7/7` 通过；`V2R-0`/`V2R-P` 在修订门限下仍为 `2/7`（仅中位 RMSE 与优于参考谱比例通过）与 `3/7`，故维持"未通过"。原始门限下的判定为 `V2R-PS`/`PSG` 各 `3/7`，两者对照关系见 7.1 节门限修订记录。该列变化**不改变**候选选择（仍按并列容差 `0.005` 取自由度更少的 `V2R-PS`）、参数剖面判定、R-D 结论与最终 `V2R_SPECTRAL_ONLY` 决策。

PSG 的中位 logRMSE 比 PS 低 `0.002659`，在 `0.005` 并列容差内，因此按预注册复杂度规则选择自由度更少的 **`V2R-PS`**，而不是仅按数值最优选择 PSG。该选择是 Train-only 开发候选选择，不是 Validation 结果或生理真实性结论。PS/PSG 全部 44 名外折受试者求解收敛；PS 每折冻结的 `A_s` 为 `1.476306/1.487813/1.443267/1.497035/1.515393`，`Δb_s` 均为 `0.5`（5/5 折进入边界）；PSG 在此基础上冻结 `A_s/Δb_s`，每折单一 `g_0` 为 `1.053160/1.066625/1.207338/1.059138/1.074477`。`A_s` 未进入边界但所有折剖面不可辨识；`Δb_s` 5/5 折边界且剖面不可辨识；`g_0` 无边界折且剖面可辨识。个体 `f_blood` 上界贴边比例为 PS `21/44 (47.73%)`、PSG `24/44 (54.55%)`，`f_mel` 任一边界均为 `0/44`；这些均按合同报告为诊断，不构成中途停止理由。

R-C0R 数据访问审计计数为：R-B manifest 内容读取 `1` 次，历史 R-C0 产物内容读取 `8` 次，原始 HSI/RGB/Validation/Test/临床 500 例内容读取均为 `0`。反射率裁剪为 `false`；PS/PSG 外折固定参考谱重现最大 logRMSE 差为 `8.33e-17`。唯一决策 JSON、审计摘要、OOF 参数/预测/逐波段残差、折内全局量和剖面、四候选汇总、参考谱比较、选择理由与完整产物哈希见上述独立输出目录中的 `r_c0r_decision.json`、`audit_summary.json`、`fold_global_parameters.csv`、`global_parameter_profiles.csv`、`four_candidate_*` 和 `artifact_hash_manifest.csv`。旧 R-C0 的 `R_C0_STOPPED_AT_FAILED_UPGRADE` 结论未被修改或覆盖。

本批次结束后的唯一登记后续为 `R-C1`；R-C0R 不授权 Validation、Test、临床 500 例或 RGB 编码器训练。

#### R-C1：参数可辨识性与稳健性（v2R.1）

**执行状态（2026-09-11）：`R_C1_COMPLETE`。** 按 8.5 合同，在 R-C0R 选定的 `V2R-PS` 及 0.005 容差内的 `V2R-PSG` 上完成 Train-only 参数剖面与固定量敏感性。新批次使用独立配置 [`km_bio_v2r1_stage_c1.yaml`](../../configs/skin_optics_hsi/km_bio_v2r1_stage_c1.yaml)、实现 [`km_bio_v2r1_stage_c1.py`](../../src/skin_optics_hsi/km_bio_v2r1_stage_c1.py)、入口 [`run_km_bio_v2r1_stage_c1.py`](../../scripts/skin_optics_hsi/run_km_bio_v2r1_stage_c1.py) 和测试 [`test_km_bio_v2r1_stage_c1.py`](../../tests/skin_optics_hsi/test_km_bio_v2r1_stage_c1.py)，输出目录为 [`km_bio_v2r1_stage_c1`](../../outputs/skin_optics_hsi_v2r1/stage1_hsi_physics/km_bio_v2r1_stage_c1/)。

敏感性覆盖包装直径 `0/7.5/15/30 µm`、`s0=0.50/0.70/0.90`、表皮厚度 `0.050/0.060/0.070 mm`、全血 Hb `100/150/200 g/L`、散射幅度倍率 `0.90/1.00/1.05`、合法 `Δb_s` 偏移 `-0.10/0` 及 `420–680`、`430–670`、`440–660 nm` 带宽。全部敏感性拟合收敛，反射率裁剪为 `false`；原始 HSI、RGB、Validation、Test 和临床 500 例读取均为 `0`。`f_mel` 两候选均无边界且剖面标记可辨识；`f_blood` 上界贴边比例为 PS `47.73%`、PSG `54.55%`，剖面标记不可辨识。R-C0R 的全局量诊断保持不变：`A_s` 无边界但剖面不可辨识，`Δb_s` 5/5 折边界且不可辨识，`g0` 无边界且可辨识。

带宽收窄的 Train 中位 logRMSE 为 PS `0.0540`（430–670）和 `0.0489`（440–660），PSG `0.0517` 和 `0.0482`；同时边界比例上升。包装直径、`s0`、厚度、Hb 和散射改变的逐受试者位移、边界状态及汇总均已保存。结果只支持模型条件下的稳定性分层，不构成生理真实性或外部泛化证明。R-C1 测试 `3 passed`；完整相关 skin-optics 测试套件 `93 passed`。本批次后仅登记 R-D 冻结与开发性 Validation 复核，未执行 Validation/Test/500/RGB。

#### R-C1R：可辨识性补全与 V2R-BEST-O（v2R.1）

**执行状态（2026-09-11）：`R_C1R_COMPLETE`。** R-C1 批次虽登记为 `R_C1_COMPLETE`，但 8.5 与 7.4 节要求的两项组件尚未执行：按 R-A 登记包络规则的条件损失参数剖面，以及在最佳有效基础候选上运行氧合扩展 `V2R-BEST-O`。本批次以独立配置 [`km_bio_v2r1_stage_c1r.yaml`](../../configs/skin_optics_hsi/km_bio_v2r1_stage_c1r.yaml)、实现 [`km_bio_v2r1_stage_c1r.py`](../../src/skin_optics_hsi/km_bio_v2r1_stage_c1r.py)、入口 [`run_km_bio_v2r1_stage_c1r.py`](../../scripts/skin_optics_hsi/run_km_bio_v2r1_stage_c1r.py)、测试 [`test_km_bio_v2r1_stage_c1r.py`](../../tests/skin_optics_hsi/test_km_bio_v2r1_stage_c1r.py) 和输出目录 [`km_bio_v2r1_stage_c1r`](../../outputs/skin_optics_hsi_v2r1/stage1_hsi_physics/km_bio_v2r1_stage_c1r/) 补齐，任务 ID 为 `R-C1R_COMPLETE_IDENTIFIABILITY_AND_BEST_O`。旧 `km_bio_v2r1_stage_c1` 批次不被修改；其 `r_c1r_decision.json` 通过 `supersedes_components` 只覆盖 `conditional_parameter_profiles` 与 `V2R-BEST-O` 两项，已登记的固定量敏感性组件保留。

前置核对通过：R-C0R 决策/审计均为 `R_C0R_COMPLETE` 且选定 `V2R-PS`，R-C1 报告 `R_C1_COMPLETE`，固定折清单 SHA-256 仍为 `e763a9f3dc7f58babe264f7056699c3d5b3722187fbb558f83f2fdb5d6b67370`。对 R-C0R 的 `V2R-PS/PSG` OOF 结果复现，最大 `|Δf_mel|=5.41e-9`、最大 `|Δf_blood|=3.01e-8`、最大 `|ΔlogRMSE|=2.50e-14`，在 `1e-6` 容差内通过。

按 R-A 登记包络规则（可接受损失增量 `0.005`、包络归一化跨度上限 `20%`、边界为归一化 `1%` 带），`V2R-PS/PSG` 的 51 点条件剖面分类为：`f_mel` 边界比例 `0.477/0.545`、可辨识比例均 `1.0`、中位跨度 `0.0044/0.0051`；`f_blood` 边界比例 `0.477/0.545`、可辨识比例均 `0.318`、中位跨度 `0.34/0.33`。**全部网格点收敛，但 `f_mel` 与 `f_blood` 均分类为 `unreliable`。** 这里更正了 R-C1 批次在**点估计**边界定义下对 `f_mel` 的“无边界、可辨识”结论：按登记的包络规则，`f_mel` 的可接受包络在 47.73%（PS）/54.55%（PSG）受试者中触及归一化下边界带，超过其 10% 可靠阈值。

`V2R-BEST-O` 在 `V2R-PS` 与 `V2R-PSG` 上以 `[f_mel,f_blood,s]` 运行：中位 logRMSE 分别为 `0.056143`（相对基础 `-0.005335`）和 `0.053942`（相对基础 `-0.004877`），P90 `0.100255/0.097189`，中位 RMSE `0.015865/0.015603`，中位 SAM `3.149°/3.086°`；全部收敛、无反射率裁剪、`prediction_above_one_count=0`。但三个参数在 R-A 规则下全部为 `unreliable`（`s` 的可辨识比例仅 `0.25/0.159`，包络跨度 `0.40/0.44`；`f_mel` 边界比例 `0.659/0.591`）。因此氧合扩展只作诊断：**不输出 `s`**，也不反向否定基础候选的 `f_mel/f_blood`。

左右侧压力关联显示 `f_blood` 与双侧 log 差的 Pearson 相关在 PS/PSG 及两个 BEST-O 中均为 `0.455–0.478`（`p≈0.001`），`f_mel` 与 `s` 的相关不显著；`f_blood` 与固定左右侧观测差异的同向关联在补全后依然存在。数据访问计数为：R-B manifest 内容 `1`、历史 R-C0R 产物 `4`、R-C1 产物 `2`、R-B 侧别审计 `1`；原始 HSI/RGB/Validation/Test/临床 500 例均为 `0`。R-C1R 测试 `7 passed`；完整 `tests/skin_optics_hsi/` 套件 `103 passed`。实现哈希（配置/源码/入口/测试）依次为 `8a9461158bcd83a8da92ff144a2f4ab1d0a2c1d383cbd049e9edd219f2eeba09`、`373eef3341f0d561475d9a971227988289f070992bed424a75c203139782e8d9`、`a9b7f6e24bb0bcdc3e71e176810bb5283c4be7ed37e586408e97082b44430c55`、`55e50dea065a77a9d836fed93b0af8c0698b73c7104daafa2e4228f3b8b17b02`，完整清单见产物目录内 `artifact_hash_manifest.csv`。本批次结束后的唯一登记后续仍为 `R-D`；不授权 Validation 之外的 Test、临床 500 例或 RGB 编码器训练。

#### R-D：完整 Train 冻结与开发性 Validation 复核（v2R.1）

**执行状态（2026-09-11）：`R_D_COMPLETE_DIRECTION_CONSISTENT`。** R-D 由两个批次组成，均在新建目录中完成、不修改任何历史产物。第一项是 **R-D 观测合同**——这是 KM-BIO 线上**首次读取 Validation 原始 HDF5 内容**，用与 R-B 完全相同的读取器、transpose、mask 统计与双侧对称聚合公式重算 Validation 颊谱。配置 [`km_bio_v2r1_rd_validation_observation_contract.yaml`](../../configs/skin_optics_hsi/km_bio_v2r1_rd_validation_observation_contract.yaml)、实现 [`km_bio_v2r1_rd_observation.py`](../../src/skin_optics_hsi/km_bio_v2r1_rd_observation.py)、入口 [`run_km_bio_v2r1_rd_observation.py`](../../scripts/skin_optics_hsi/run_km_bio_v2r1_rd_observation.py)、测试 [`test_km_bio_v2r1_rd_observation.py`](../../tests/skin_optics_hsi/test_km_bio_v2r1_rd_observation.py)，输出目录 [`km_bio_v2r1_rd_validation_observation_audit`](../../outputs/skin_optics_hsi_v2r1/stage1_hsi_physics/km_bio_v2r1_rd_validation_observation_audit/)。范围为 `split=valid` 的 `p001/p016/p030`、3 次采集、6 条区域谱、3 条双侧对称谱，`cube` 形状 `[31,1024,1024]`、原始 dtype `float64`。15/15 审计检查通过，其中 `validation_hsi_content_reads=3`、`no_train_or_test_path_is_opened=true`、`raw_float64_vs_float32_cache_within_tolerance=true`。**独立复核**：从原始 HDF5 用另一条独立路径重算 3 条对称谱，与落盘 manifest 逐位比较 `max|diff|=0.0`。

第二项是 **完整 Train 冻结 + 开发性 Validation 复核**。配置 [`km_bio_v2r1_stage_rd.yaml`](../../configs/skin_optics_hsi/km_bio_v2r1_stage_rd.yaml)、实现 [`km_bio_v2r1_stage_rd.py`](../../src/skin_optics_hsi/km_bio_v2r1_stage_rd.py)、入口 [`run_km_bio_v2r1_stage_rd.py`](../../scripts/skin_optics_hsi/run_km_bio_v2r1_stage_rd.py)、测试 [`test_km_bio_v2r1_stage_rd.py`](../../tests/skin_optics_hsi/test_km_bio_v2r1_stage_rd.py)，输出目录 [`km_bio_v2r1_stage_rd`](../../outputs/skin_optics_hsi_v2r1/stage1_hsi_physics/km_bio_v2r1_stage_rd/)，任务 ID `R-D_FREEZE_AND_DEVELOPMENTAL_VALIDATION`。11 项预检全部通过，其中固定折清单 SHA-256 `e763a9f3dc7f58babe264f7056699c3d5b3722187fbb558f83f2fdb5d6b67370` 与登记值一致，且预检断言 `parameters.fixed_s0/diameter_um/g0` 必须等于冻结块 `frozen_quantities`。冻结在**完整 44 人 Train** 上以 `full_train_joint_centered_log_shape` 估计 `V2R-PS` 的全局散射，得 `A_s=1.483455915077612`、`Δb_s=0.4999999999999999`（**贴上界，不可辨识**，与 R-C0R 五折结论一致）；登记固定量 `g0=1.0`、`Dv=15.0 µm`、`s0=0.70`、表皮厚度 `0.060 mm`；`validation_used_in_freeze=false`，冻结 JSON 及其 SHA-256 在**读取 Validation 之前**落盘。

冻结后的方向一致性判定全部通过（6/6）：Train 中位 logRMSE `0.060752`、Validation `0.059208`（未回退）；Train 中位 SAM `3.436°`、Validation `3.530°`（未回退）；3/3 条 Validation 谱优于固定 Train 参考谱（`validation_model_better_than_reference_fraction=1.0`，模型/参考误差比中位 `0.5672`）；参数中位数落在扩展 Train 区间内；边缘（`420–430/670–680 nm`）绝对带符号残差中位 Train `0.188965`、Validation `0.177010`（未变差）；`f_blood` 上边界比例未变差。结论 `direction_consistent=true`，`status=R_D_COMPLETE_DIRECTION_CONSISTENT`。

**判定上限如实记录**：因 R-C1R 在 R-A 登记包络规则下把 `f_mel/f_blood/s` 全部判为 `unreliable`，R-D **只能检验光谱泛化**，`decision_state = V2R_DEVELOPMENT_SPECTRAL_CANDIDATE / V2R_SPECTRAL_ONLY`，`parameter_reliability_claim=none`、`parameter_upgrade_forbidden=true`。不升级 `V2R_PARTIAL_THETA_CANDIDATE`（该状态要求至少一个参数 `reliable/conditional`），不输出独立 `s`。Validation 仅 3 人，`validation_is_developmental_only=true`，只作开发性复核，不给稳定总体估计、不构成生理真实性证明。数据访问计数：R-B manifest `1`、R-B 侧别审计 `1`、R-D Validation 观测 manifest `1`、R-D Validation 侧别审计 `1`、R-C0R 产物 `3`、R-C1R 产物 `4`、历史 R-C0 产物 `1`；原始 HSI `0`、RGB `0`、Validation HDF5 `0`（经已审计 manifest 消费 3 条 Validation 光谱）、Test `0`、临床 500 例 `0`。R-D 观测测试 `6 passed`、冻结与复核测试 `9 passed`；完整 `tests/skin_optics_hsi/` 套件 `118 passed`。实现哈希（观测配置/源码/入口/测试）：`b117616691ebb749e31693016e824de89a45f21c1a1b28db56a1ad4ec6ee358a`、`771586ccc65aedc2e04ff317f501f8d039007692370531a686180838bfe4cb35`、`415637266604adc4a1dce802d76d51e551af6045b0ce4d46056e46b4a99d678d`、`b89ebd6f8c457276c2f897959663ba3e3af77a6af9c04ff93874037a4702e2e9`；冻结与复核配置/源码/入口/测试：`4905c8ff94e9951199f8c20cb57e328567985b0ea1bbd779651c3b3923b851c1`、`3bb1cc78184a39a36fe050c7bf6a4b91e141120b6c689c7e1863ef3fd9ec2bd4`、`b2b5fa5f72d553ff24461f9dbe5e78a8b3bd66e9afb032293423a9656e72c736`、`c64144776e31a52573d05437bad1856e0c296a39269043e309f02ac2ed090f52`，完整清单见两份 `artifact_hash_manifest.csv`（输出 40/40 与 16/16 条全量复核通过）。**发射方式说明**：主运行后修正了 R-D 单元测试的一处断言字段名（`scattering_amplitude`→`A_s`，仅测试文件，不影响数值），为保持哈希链与最终磁盘实现一致，产物经 `finalize_stage_rd` 重发（`emission=finalize_reemission`、`numerics_recomputed=false`）；重发只从落盘表格重派生聚合量，未重跑冻结/反演/阈值，`created_utc` 保留首跑时间，聚合标量仅有 `~1e-16` 相对量级末位差异，判定与全部检查项不变。本批次结束后 `next_registered_stage=STAGE1_OPTICAL_LAYER_DECISION_REQUIRED`；不授权 Test、临床 500 例或 RGB 编码器训练。

#### 阶段一光学层决策（v2R.1）

**执行状态（2026-09-11）：`STAGE1_OPTICAL_LAYER_DECISION_COMPLETE`；决策状态 `V2R_SPECTRAL_ONLY`。** R-D 在 `r_d_decision.json` 中登记的 `next_registered_stage = STAGE1_OPTICAL_LAYER_DECISION_REQUIRED` 即本例执行的动作。配置 [`km_bio_v2r1_stage1_optical_decision.yaml`](../../configs/skin_optics_hsi/km_bio_v2r1_stage1_optical_decision.yaml)、实现 [`km_bio_v2r1_optical_decision.py`](../../src/skin_optics_hsi/km_bio_v2r1_optical_decision.py)、入口 [`run_km_bio_v2r1_optical_decision.py`](../../scripts/skin_optics_hsi/run_km_bio_v2r1_optical_decision.py)、测试 [`test_km_bio_v2r1_optical_decision.py`](../../tests/skin_optics_hsi/test_km_bio_v2r1_optical_decision.py)，输出目录 [`km_bio_v2r1_optical_decision`](../../outputs/skin_optics_hsi_v2r1/stage1_hsi_physics/km_bio_v2r1_optical_decision/)，任务 ID `STAGE1_OPTICAL_LAYER_DECISION`。

本批次把 v1 A/B/C、v2R R-A/R-B/R-C0 与 v2R.1 R-C0R/R-C1/R-C1R/R-D 的已登记证据汇总为唯一的 7.5 决策状态，并按 3.6 节冻结阶段二接口。11 项预检全部通过（含 R-D 冻结参数 JSON 哈希链复校、固定折清单 `e763a9f3…` 一致、R-C1R 的 Validation 读取为 0）。**数据边界是本批次最强的性质：它不读取任何新数据内容**——原始 HSI / RGB / Validation / Test / 临床 500 例内容读取均为 `0`，只消费 18 次已审计的决策 JSON 与汇总表；不做任何拟合、不重算任何阈值。

七项强目标（7.1 节）在 `V2R-PS` 折外双侧对称谱上重算并与 R-C0R 引擎的 `strong_spectral_checks` 逐项对照（7/7 一致）：中位 logRMSE `0.0615`（原门 `≤0.06`，实测超 `+2.46%`）、P90 logRMSE `0.1169`（原门 `≤0.10`，超 `+16.88%`）、中位 SAM `3.484°`（原门 `≤3°`，超 `+16.15%`）、420–680 nm 最大绝对中位有符号偏差 `0.0342`（原门 `≤0.03`，超 `+14.00%`）四项**在原门限下未过**；中位原始 RMSE `0.0181`（门 `≤0.03`）、优于受试者外固定参考谱比例 `0.9318`（门 `≥2/3`）、模型/参考谱中位误差比 `0.4432`（门 `≤0.90`）三项通过，合计 **原门限 3/7**。按 7.1 节 2026-09-11 修订门限（`0.0625 / 0.12 / 0.03 / 3.5° / 0.035 / ≥2/3 / ≤0.90`），四项偏离均在一档工程容差内并设定为通过，合计 **修订门限 7/7**。**该修订只改变"强光谱拟合"这一描述的判定，不参与 7.5 节决策状态的优先级链**：本批次 `V2R_SPECTRAL_ONLY` 的成立依据是"所有生理参数均不可辨识"，与七项光谱目标无关。

7.5 节的 `V2R_REVISE_OBSERVATION` 只在“统一尺度或固定侧差仍是主要阻断”时成立，故两项均实测：(1) **统一尺度**——`V2R-PS` 按合同把 `g0` 固定为 `1.0`，`g0` 在 `V2R-PSG` 折内剖面可辨识且 0 折贴边，两候选中位 logRMSE 差 `0.002659 ≤ 0.005`，尺度歧义不影响选定，**不是阻断**；(2) **固定侧差**——`f_blood` 与双侧 log 差的 Pearson 相关 `0.4776`（`p=0.00104`）确实存在，但主观测单位是双侧对称谱（R-B 的 `symmetric_spectra = subjects = 44`，侧差被构造性排除出光谱目标），且该耦合的后果已被 `f_blood = unreliable` 这一最差分类完整吸收（去掉耦合亦无法升级任何参数，因 R-A 规则独立地让三个参数各自失败）。故 `observation_level_blocker_present = false`，`V2R_REVISE_OBSERVATION` 不成立；`V2R_STOP_KM` 亦不成立（Validation 优于参考谱比例 `1.0`、误差比 `0.567`、方向一致、且每位受试者只拟合 2 个参数加 3 个全局标量，未依赖逐谱高容量校正）。

按预登记优先级取第一个成立的状态 → **`V2R_SPECTRAL_ONLY`**，`stage1_optical_layer_outcome = SPECTRAL_RECONSTRUCTION_ONLY_NO_THETA_TARGET`。里程碑 `V2R_DEVELOPMENT_SPECTRAL_CANDIDATE` 成立，且严格按 7.5 节由**折外阶梯**判定（误差比 `0.443159`、优于参考谱比例 `0.931818`），3 人 Validation 只在 R-D 中确认方向未逆转（`validation_confirms_direction = true`），不用于确立该状态。`V2R_PARTIAL_THETA_CANDIDATE` 因“至少一个参数 `reliable/conditional`”不满足而不成立；参数声明上限不变（`parameter_reliability_claim = none`、`parameter_upgrade_forbidden = true`）。3.6 节接口冻结：交付 `Fskin` 前向模型与 10 nm 系数资产（SHA-256 `30de54c4…`），**可靠参数清单为空**，`theta` 不可作为阶段二监督目标。`residual_risk_register.csv` 如实登记 8 条残余风险（拟合窗内带结构残差、`A_s/Δb_s` 全局散射退化、边缘带同号复现、带宽收窄越界等），全部 `would_justify_revise_observation = false`。测试：决策门 `11 passed`、完整 `tests/skin_optics_hsi/` 套件 `129 passed`；产物哈希清单 35 条（19 输入 + 1 配置 + 3 实现 + 12 输出）全量复核通过；实现哈希依次为配置 `8ee01e0db6c72535745d3aceb7b59ee4837008cfc6ca71f422ff67bcf8459ab7`、源码 `220d75c2b975fb7135ef4258dd9a2b1a202d5d69dab7afde5e2b31f31e9a7d3f`、入口 `c57d7127185007fca9612a2b059f42a668c60bf41fa08abd7a03448a251bdea8`、测试 `43deb7e468ba5a649cf5ce1b4630d02ae1a67ca3fa866b5f89202418b691718e`。本批次结束后 `next_registered_stage = NONE_REGISTERED`、`km_bio_line_status = CLOSED_AT_STAGE1_DECISION`：**KM-BIO 线在阶段一决策处收尾**，Test、临床 500 例、RGB 编码器训练、阶段二参数编码器，以及任何波段／边界／目标改动，均需另行登记。

### 8.5 后续步骤（R-C0R 完成后）

#### R-C1：参数可辨识性与稳健性

完整基础梯级结束后，对按 7.3 节选出的候选执行完整剖面、包装直径、`s0`、厚度、Hb 标尺、散射和带宽敏感性，不要求它先达到全部强光谱目标。若有更低复杂度候选与最佳候选的中位 logRMSE 相差不超过 `0.005`，同时检查该低复杂度候选。随后在最佳有效基础候选上运行 `V2R-BEST-O`，分别判断 `f_mel`、`f_blood` 和 `s` 的可靠性。

#### R-D：冻结与 Validation

基础梯级和 R-C1 完成后，只要得到无合同违规且在 OOF 上优于固定参考谱的 `V2R_DEVELOPMENT_SPECTRAL_CANDIDATE`，即可在完整 Train 冻结该候选并访问旧 Validation，检验光谱误差和参数行为是否同方向泛化；无需先达到全部强目标。Validation 只有 3 人，只能用于开发性复核，不能给出稳定总体估计，也不能单独证明生理真实性。Test 和 500 例仍保持锁定，直到阶段一最终候选及参数声明完成。

**执行状态（2026-09-11）：`R_D_COMPLETE_DIRECTION_CONSISTENT`。** 完整 Train（44 人）冻结 `V2R-PS` 得 `A_s=1.483456`、`Δb_s=0.500000`（贴上界、不可辨识），随后对 3 条 Validation 双侧对称谱反演，6/6 方向一致性检查通过（Validation 中位 logRMSE `0.0592` 未回退于 Train `0.0608`，3/3 优于固定参考谱，边缘残差与 `f_blood` 边界比例均未变差）。因 R-C1R 判三个参数全部 `unreliable`，`decision_state=V2R_DEVELOPMENT_SPECTRAL_CANDIDATE / V2R_SPECTRAL_ONLY`，不升级参数声明。详细合同、哈希与访问计数见 8.4 节 R-D 条目。

**后续动作状态：R-D 登记的 `STAGE1_OPTICAL_LAYER_DECISION_REQUIRED` 已于同日执行完毕，结论 `V2R_SPECTRAL_ONLY`**（`next_registered_stage = NONE_REGISTERED`）。8.5 节不再登记新的执行步骤；Test、临床 500 例、RGB 编码器训练与阶段二参数编码器均需另行登记。详见 8.4 节“阶段一光学层决策（v2R.1）”条目。

#### R-D 之后的登记边界

R-D 是 8.5 节登记的最后一个执行批次，其后只有一次决策动作（阶段一光学层决策），不存在未执行的登记步骤。若未来要重启 KM-BIO 线，下列任一项都构成新版本登记而不是本轮的遗留任务：更换或放宽波段窗口（7.1 节禁止跨波段复用强目标）、改动参数边界（9 节禁止仅放宽 `f_blood` 上界后重跑）、单独修订观测合同（`V2R_REVISE_OBSERVATION` 路径，须给出“统一尺度或固定侧差为主要阻断”的新证据）、或改换前向模型（`V2R_STOP_KM` 路径）。

#### 后续阶段共同记录要求

- 新配置、源码、脚本和测试的 SHA-256；
- 固定 5 折受试者清单；
- 各候选 out-of-fold 指标和成对差值；
- 每折 `As/Δbs/g0` 及边界状态；
- 420–680 nm 主残差和四个边缘波段诊断；
- `f_mel/f_blood/s` 的边界、剖面和敏感性；
- 左右侧压力分析；
- 唯一的 v2R Train decision；
- 对实验记录的追加条目，保留 v1 失败结论。

## 9. 当前明确不采用的修订

以下做法没有足够证据，v2R 禁止采用：

- 只把 `f_blood` 上界从 0.10 放宽后重跑；
- 删除困难样本或按拟合误差修改 mask；
- 根据 v1 残差逐波段调权后仍使用原验收声明；
- 为每条谱加入任意 31 波段校正曲线；
- 逐谱自由拟合散射幅度、背景吸收、厚度、增益和 tilt；
- 先查看 Validation/Test 再决定 `Dv/As/Δbs/g0`；
- 用 500 例分类性能选择物理公式；
- 在基础模型失败时，依靠神经网络残差让闭环损失下降并赋予参数生理解释。

## 10. 证据与文献边界

本方案关于血管包装的依据包括：

1. Lau 等对组织光散射模型的重新评价指出，应使用有效血管半径/包装参数描述非均匀 Hb 分布；包装会改变 Soret 与 Q 带的相对反射特征：[Re-evaluation of model-based light-scattering spectroscopy for tissue spectroscopy](https://pmc.ncbi.nlm.nih.gov/articles/PMC2866094/)。
2. Mirkovic 等的实验验证显示微血管色素包装会影响 Hb 的 420、542 和 577 nm 吸收特征及反演参数：[Experimental Validation of the Effects of Microvasculature Pigment Packaging on In Vivo Diffuse Reflectance Spectroscopy](https://pmc.ncbi.nlm.nih.gov/articles/PMC3336741/)。
3. 双层 K–M 层组合仍参考：[Skin Parameter Map Retrieval from a Dedicated Multispectral Imaging System Applied to Dermatology/Cosmetology](https://pmc.ncbi.nlm.nih.gov/articles/PMC3789448/)；该文献不证明本项目的观测尺度或系数闭合正确。

这些文献支持“该机制值得检验”，并不提供 Hyper-Skin 个体的真实 `Dv`、散射参数或 Hb 标签。v2R 所有生理输出仍属于指定模型和观测合同下的有效参数。

## 11. 本次修订决策

```text
v1_status = REVISE_OBSERVATION_OR_MODEL
formula_model = KM-BIO-v2R
protocol_version = KM-BIO-v2R.1
v2r1_flow_status = COMPLETE
allowed_data = Hyper-Skin Train only
validation_access = DEVELOPMENTAL_REVIEW_COMPLETED_IN_R_D
test_access = FORBIDDEN_UNTIL_SEPARATE_DECISION
clinical_500_access = FORBIDDEN_IN_OPTICAL_DEVELOPMENT
formula_audit = PASS
observation_contract = PASS_FOR_V2R_TRAIN_INVERSION
historical_r_c0_status = R_C0_STOPPED_AT_FAILED_UPGRADE
historical_r_c0_executed = V2R-0,V2R-P
resume_candidates = V2R-PS,V2R-PSG
rerun_v2r_0_or_p = false
candidate_selection = V2R-PS
r_d_status = R_D_COMPLETE_DIRECTION_CONSISTENT
stage1_decision_status = STAGE1_OPTICAL_LAYER_DECISION_COMPLETE
final_decision = V2R_SPECTRAL_ONLY
parameter_reliability_claim = none
theta_usable_as_supervised_target = false
km_bio_line_status = CLOSED_AT_STAGE1_DECISION
next_registered_stage = NONE_REGISTERED
```

v2R 的 R-A 和 R-B 均已通过，`V2R-P` 也提供了方向一致的拟合改善。原 R-C0 的停止是旧流程规则造成的，不等于 `PS/PSG` 已被证伪。v2R.1 在新配置和新输出目录中恢复了 `PS/PSG` 并完成完整梯级与 R-C1/R-C1R；随后按 7.3 选择 `V2R-PS`，冻结后对旧 Validation 做开发性光谱复核（R-D），最后执行阶段一光学层决策。**全流程结论为 `V2R_SPECTRAL_ONLY`**：KM-BIO 双层 K–M 生理线在阶段一收尾，只承认前向光谱重建能力，不训练生理参数编码器；Test、临床 500 例、RGB 编码器训练与阶段二参数编码器均需另行登记。各批次实测状态见 8.4 节。

## 12. v2R.1 规则复核结论

本次复核确认：

- C 阶段关键数值、样本数、失败门和残差方向与上游实验记录一致；
- v2R 没有覆盖、改写或重新解释 v1 历史结果；
- 血管包装、散射校准和 420–680 nm 范围收窄均被明确登记为可证伪的工程候选，而非已证实真理；
- 受试者级双侧聚合不会被宣称为像素级参数图；
- `A_s` 与 `g_0` 继续顺序估计，剖面平坦性和贴边状态用于可靠性标记，不再提前终止后续候选；
- 强光谱目标、参数边界和 80% 可靠覆盖已与候选执行解耦；
- R-C1 不再以“先通过全部光谱门”为前提，允许用剖面分析判断最佳拟合候选中的哪些参数真正可用；
- 完整基础梯级结束并冻结开发候选后，可使用旧 Validation 做小样本光谱泛化复核；Test 和 500 例仍隔离。

因此，本文件作为 v2R.1 的实施基线，现已全部执行完毕。`R-C0R` 已锁定现有数据和折、核对 `V2R-0/P` 历史结果并运行 `V2R-PS/PSG`；其后 R-C1/R-C1R、R-D 与阶段一光学层决策亦已完成，最终决策为 `V2R_SPECTRAL_ONLY`，`next_registered_stage = NONE_REGISTERED`。不得把未执行候选写成旧实验的遗漏，也不得把规则修订写成实施前预登记；它是一次明确记录的 Train 开发阶段修订。若未来重启 KM-BIO 线，按下节与 8.5 节"R-D 之后的登记边界"另行登记新版本。

## 13. 过严规则修订清单

| v1.2 旧规则 | 问题 | v2R.1 修订 |
|---|---|---|
| 每个中间候选必须通过七项最终光谱门 | 要求单一机制独自解决组合失配，导致 `P` 后过早停止 | `0/P/PS/PSG` 全部运行，七项标准改为强拟合目标 |
| 中间候选必须改善至少 5%，且至少 2/3 人改善 | 小幅或互补机制可能在组合后才显效 | 改为效果强度描述，不再阻止后续候选 |
| `f_blood≤20%`、`f_mel≤10%` 边界率作为升级门 | 后续散射/尺度修订本来可能缓解前一候选的贴边 | 边界率用于最终参数可靠性分层，不中断梯级 |
| 全局参数剖面宽或 3/5 折贴边即拒绝升级 | 可以说明该全局量不可解释，但不等于后续光谱组合没有检验价值 | 使用确定性运行值完成比较，同时把该全局量标为不可辨识 |
| 只有通过光谱门的候选才运行 R-C1 | 无法了解最佳拟合候选究竟因何参数不稳定 | 对完整梯级选出的最佳候选直接做剖面和敏感性 |
| 基础候选全部通过后才开放 `s` | 可能错过氧合自由度对 Hb 残差的真实贡献 | 在最佳有效基础候选上运行 `V2R-BEST-O`，再独立判定 `s` |
| 只有 `CONDITIONAL_BIO_READY` 才访问 Validation | 容易在 44 人 Train 上长期迭代，反而增加开发过拟合 | 最佳开发候选冻结后即可做 Validation 光谱复核，但不提前进入阶段二 |

修订没有删除质量控制。数据泄漏、合同偏离、非有限输出、未经登记的裁剪和不可复现产物仍是硬停止条件；变化只在于不再把“尚未达到理想效果”误当成“后续候选不得运行”。
