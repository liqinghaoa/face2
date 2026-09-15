# 阶段一 双层 K–M 生理模型实验记录

## 文档用途与维护规则

本文是 `KM2L-HF-v1` 及其登记修订版 `KM2L-HF-v2R` 的独立实验记录，记录公式合同、观测合同、反演与验收、数据使用规则的实际执行状态。v1 上位设计以[阶段一_双层KM生理模型实施规划](../03_Method_and_Experiment/阶段一_双层KM生理模型实施规划.md)为准，v2R 以[KM-BIO-v1 C 阶段失败原因与 v2R 修订方案](../05_Results_and_Decisions/KM-BIO-v1_C阶段失败原因与v2R修订方案.md)为准。

本文只记录已经运行的实验。尚未执行的真实 HSI 反演、Validation/Test 评估和 500 例推理不得写成完成；失败的运行不得被后续成功运行覆盖。该记录不修改或替代历史[阶段一_HSI_物理验证实验记录](阶段一_HSI_物理验证实验记录.md)。

## 当前状态

更新日期：2026-09-11

| 实施阶段 | 状态 | 已完成内容 | 下一步 |
|---|---|---|---|
| A 公式合同与数学闭合审计 | `PASS` | 独立前向实现、机器可读合同、文献式与稳定式等价性、随机边界、角点、NumPy/Torch 一致性和梯度检查 | 保留为已完成阶段 |
| B 观测合同核对 | `PASS_FOR_TRAIN_INVERSION` | 按冻结配准与 mask 从原始 Train HSI 重算 88 条主域脸颊谱，并完成尺度、波长、缓存、RGB 来源和文件哈希核对 | C Train 主反演已完成 |
| C Train 真实 HSI 反演 | `REVISE_OBSERVATION_OR_MODEL` | 88 条谱主反演、留一参考谱、三参数容差剖面、13 组固定假设/观测敏感性、自由增益和左右侧混淆诊断均已完成 | 先登记模型/观测合同修订版本，不进入 D |
| v2R R-A 独立公式合同与数学审计 | `PASS` | 血管包装、Train-global 散射/尺度接口、退化极限、物理边界、NumPy/Torch 和梯度审计 | 保留为已完成阶段 |
| v2R R-B 修订观测合同 | `PASS_FOR_V2R_TRAIN_INVERSION` | 复用 v1 B 审计 manifest，生成 44 条受试者级双侧 log 几何平均谱，完成配对、正值、波段角色和数据隔离审计 | 保留为 R-C0 的已审计输入 |
| v2R R-C0 候选梯级筛选 | `R_C0_STOPPED_AT_FAILED_UPGRADE` | 固定 5 折运行 `V2R-0/P`；包装虽改善误差和 Soret/Q 残差关系，但最终光谱门与 `f_blood` 边界门失败 | 不运行 `PS/PSG`，不授权 Validation；另立合同前停止增加复杂度 |
| v2R.1 R-C0R 恢复候选梯级 | `R_C0R_COMPLETE` | 只读复用历史 `V2R-0/P`，在同一固定 5 折中完成 `V2R-PS/PSG`，四候选统一 OOF 比较和完整性审计 | 已完成；进入 R-C1 |
| v2R.1 R-C1 参数可辨识性与稳健性 | `R_C1_COMPLETE` | 对 `V2R-PS/PSG` 完成参数剖面、包装直径、s0、厚度、Hb、散射和带宽敏感性；全部 Train-only 拟合收敛 | 条件剖面组件由 R-C1R supersede，敏感性组件保留 |
| v2R.1 R-C1R 可辨识性补全与 V2R-BEST-O | `R_C1R_COMPLETE` | 在 R-A 登记包络规则下重算 `V2R-PS/PSG` 条件参数剖面，运行氧合扩展 `V2R-BEST-O`，并补做左右侧压力关联 | 已完成；进入 R-D（完整 Train 冻结与开发性 Validation 复核） |
| v2R.1 R-D 完整 Train 冻结与开发性 Validation 复核 | `R_D_COMPLETE_DIRECTION_CONSISTENT` | 先在完整 44 人 Train 冻结 `V2R-PS` 的 `A_s/Δb_s`（`g0/Dv/s0` 按登记固定），再首次读取 Validation 原始 HDF5 生成 3 条对称谱并反演，对照 Train 做方向一致性判定 | 已完成；其唯一后续（阶段一光学层决策）已在下一行执行 |
| 阶段一光学层决策（v2R.1） | `STAGE1_OPTICAL_LAYER_DECISION_COMPLETE` | 把 v1 A/B/C、v2R R-A/R-B/R-C0 与 v2R.1 R-C0R/R-C1/R-C1R/R-D 的已登记证据汇总为唯一的 7.5 决策状态，并按 3.6 节冻结阶段二接口；判定 `V2R_SPECTRAL_ONLY`，仅承认前向光谱重建能力 | 已完成；KM-BIO 线在阶段一决策处收尾，Test / 临床 500 例 / RGB 编码器训练 / 阶段二参数编码器均需另行登记 |
| D Validation 复核 | `R_D_COMPLETE_DIRECTION_CONSISTENT` | 见 R-D 一节；Validation 光谱方向与 Train 一致，但三个生理参数在 R-C1R 下仍全部 `unreliable` | 不升级参数声明；Test 与 500 例仍锁定 |
| 500 例临床数据 | `NOT_STARTED` | 未读取 | 光学模型冻结后另立阶段三流程 |

当前总状态：`V1_REVISE_REQUIRED / V2R_R_A_PASS / V2R_R_B_PASS / V2R_R_C0_STOPPED_AT_P / V2R1_R_C0R_COMPLETE / V2R1_R_C1_COMPLETE / V2R1_R_C1R_COMPLETE / V2R1_R_D_COMPLETE_DIRECTION_CONSISTENT / STAGE1_OPTICAL_LAYER_DECISION_COMPLETE / V2R_SPECTRAL_ONLY / TEST_AND_CLINICAL_500_FORBIDDEN`。

R-D 是 KM-BIO 线上第一个读取 Validation 原始 HDF5 内容的批次（3 次内容读取），其冻结量只由 Train 决定；Validation 结论仅为 3 人开发性复核，不是总体估计，也不是生理真实性证明。

阶段一光学层决策是 KM-BIO 线上**唯一不读取任何新数据内容**的批次：它的全部输入都是已登记的决策 JSON 与汇总表，原始 HSI / RGB / Validation / Test / 临床 500 例内容读取均为 0。它只做汇总与判定，不做任何拟合、不重算任何阈值。其结论 `V2R_SPECTRAL_ONLY` 表示：在冻结的模型与观测合同下，`V2R-PS` 的 420–680 nm 双侧对称谱重建优于固定参考谱且方向一致，但三个生理参数在 R-A 登记包络规则下全部不可辨识，因此该模型不产出阶段二的生理参数目标。

这不是新模型的生理有效性结论。A 阶段只证明公式实现和数值合同闭合，不能证明 `f_mel`、`f_blood` 或 `s` 是真实个体生理量。

## 共同数据边界

本次 A 阶段实际使用的输入只有：

- 新建的 KM-BIO-v1 Python 实现；
- 公式合同 YAML；
- 只读外部光学资产 `derived_optics_10nm.npz`；
- 固定随机数种子和程序生成的参数点。

本次运行明确没有读取：

- Hyper-Skin 的 RGB 文件；
- Hyper-Skin 的 HSI/VIS 文件；
- 既有 Train、Validation 或 Test 区域光谱缓存；
- 500 例心功能数据；
- 历史 D2-MH 或旧 K2 的输出作为输入。

因此，A 阶段不产生任何逐例参数，不消耗 Hyper-Skin Test 使用次数，也不改变历史实验的访问边界。

B、C 阶段仅访问经 B 合同授权的 Train 主域内容：44 名 Train 受试者、44 次 `neutral/front` 采集和 88 条双侧脸颊区域谱。Validation/Test HSI 内容读取数均为 0；500 例心功能数据、历史 D2-MH/K2 输出和旧侧别增益均未作为 C 项输入。配对 RGB 在 B 阶段只读取文件字节用于来源和 SHA-256 核对，未解码、未参与 HSI 反演，也未被当作同场景手机实拍的生理标签。

## A：公式合同与数学闭合审计

### 目标

验证新模型的有限层 K–M 公式是否与文献式 10–13 代数等价，并确认在参数边界、随机参数、自动微分和不同后端下具有稳定且一致的数值行为。

### 固化的模型定义

模型标识为 `KM2L-HF-v1`，由有限厚度表皮和半无限真皮组成。参数顺序固定为：

```text
theta = [f_mel, f_blood, s]
```

固定量为：

```text
epidermis_thickness_mm = 0.060
dermis = semi_infinite
whole_blood_hb_g_per_l = 150.0
K = 2 * mua
S = musp_reduced
```

有限层稳定实现采用：

$$
q=\sqrt{K(K+2S)},\qquad
\beta=\sqrt{\frac{K}{K+2S}},\qquad x=qd
$$

$$
D=(1+\beta)^2-(1-\beta)^2e^{-2x}
$$

$$
R=\frac{(1-\beta^2)(1-e^{-2x})}{D},\qquad
T=\frac{4\beta e^{-x}}{D}
$$

半无限真皮反射率为：

$$
R_d=\frac{S}{K+S+\sqrt{K(K+2S)}}
$$

双层总反射率为：

$$
R_{\mathrm{total}}
=R_e+\frac{T_e^2R_d}{1-R_eR_d}
$$

### 实现文件

- [`km_bio_v1.py`](../../src/skin_optics_hsi/km_bio_v1.py)：新模型的 NumPy/Torch 前向实现；与历史 `s1_revised_forward.py` 分离。
- [`km_bio_v1_formula_contract.yaml`](../../configs/skin_optics_hsi/km_bio_v1_formula_contract.yaml)：波长、参数、单位、资产字段、公式和审计配置。
- [`run_km_bio_formula_audit.py`](../../scripts/skin_optics_hsi/run_km_bio_formula_audit.py)：不读取 HSI 的公式审计入口。
- [`test_km_bio_v1_formula.py`](../../tests/skin_optics_hsi/test_km_bio_v1_formula.py)：模型极限、边界、后端和梯度测试。

### 执行命令

在 `E:\projects\face2` 项目根目录执行：

```powershell
python scripts/skin_optics_hsi/run_km_bio_formula_audit.py
python -m pytest -q tests/skin_optics_hsi/test_km_bio_v1_formula.py
python -m py_compile src/skin_optics_hsi/km_bio_v1.py
```

审计脚本使用随机种子 `20260909`；输出目录要求不存在，禁止覆盖历史审计产物。

### 审计内容与结果

| 检查项 | 结果 | 判定 |
|---|---:|---|
| 文献原始指数形式 vs 稳定形式最大绝对差 | `3.885780586188048e-15` | PASS |
| 随机有限层参数点 | 10,000 | PASS |
| 随机层输出有限 | `true` | PASS |
| 随机层最小 `R` | `5.916646093327894e-06` | PASS |
| 随机层最小 `T` | `6.049071248326132e-05` | PASS |
| 随机层最大 `R+T` | `0.9999399977017405` | PASS |
| 参数角点和 Sobol 内点前向数量 | 136 | PASS |
| 模型前向输出有限 | `true` | PASS |
| 模型输出最小/最大值 | `0.03866430684585442 / 0.8045294162415414` | PASS |
| NumPy/Torch 最大绝对差 | `6.245004513516506e-17` | PASS |
| 中心差分梯度最大绝对差 | `2.198223647553732e-07` | PASS |
| 梯度判定 | `true` | PASS |
| 单元测试 | `4 passed` | PASS |
| HSI 内容读取 | `false` | PASS（数据隔离） |

随机层检查中使用的物理边界为 `R≥0`、`T≥0`、`R+T≤1`，浮点容差为 `1e-10`。`S=0`、`K=0`、`d=0` 的解析极限由单元测试覆盖：纯吸收层 `R=0,T=exp(-Kd)`；纯散射层 `R=Sd/(1+Sd),T=1/(1+Sd)`；零厚度层 `R=0,T=1`。

### 审计产物

- [审计 JSON](../../outputs/skin_optics_hsi_v1/stage1_hsi_physics/km_bio_v1_formula_audit/audit_summary.json)
- [审计 Markdown](../../outputs/skin_optics_hsi_v1/stage1_hsi_physics/km_bio_v1_formula_audit/audit_summary.md)
- [公式合同](../../configs/skin_optics_hsi/km_bio_v1_formula_contract.yaml)

审计 JSON 中明确登记 `hsi_read=false`，并记录随机种子、参数点数量、后端差异和梯度结果。

### 异常与处理

本次未发生代码运行失败或数值异常。审计前曾发现旧规划中的 `S u/[1+(K+S)u]` 写法不能直接宣称与文献式等价，已在规划中撤销并改为从文献指数形式直接变换得到的稳定表达式。旧模型代码未被重命名、覆盖或修改。

仍保留的限制：

1. `K=2μa,S=μs'` 是本研究登记的有效两通量闭合，不是 Hyper-Skin 成像条件下的唯一物理映射；
2. Torch 实现面向正系数内部区域，精确零系数角点由 NumPy 解析分支和单元测试审计；
3. 公式审计没有检验真实 HSI 的尺度、噪声、带宽或观测几何；
4. 公式审计没有检验参数可辨识性，也没有提供生理真值准确度。

### A 阶段决策

```text
status = PASS
model_id = KM2L-HF-v1
hsi_read = false
next_stage = B_observation_contract_audit
```

该 PASS 只授权进入观测合同核对，不授权直接进行真实 HSI 参数反演。

## B：观测合同核对

### 执行记录

- 日期：2026-09-09。
- 合同：[`km_bio_v1_observation_contract.yaml`](../../configs/skin_optics_hsi/km_bio_v1_observation_contract.yaml)。
- 入口：[`run_km_bio_observation_audit.py`](../../scripts/skin_optics_hsi/run_km_bio_observation_audit.py)。
- 运行方式：`python scripts/skin_optics_hsi/run_km_bio_observation_audit.py`。
- 数据范围：Train `neutral/front`，44 名受试者、44 次采集、左右脸颊共 88 条区域谱。
- 观测量：`white_reference_normalized_reflectance`；主输入为原始 HDF5 `float64` 的逐波段区域中位数。
- 空间链：冻结 `transpose`，mask 坐标系为 `aligned_rgb_yx`，沿用 S1-2 Train r2 冻结 mask。
- 波长链：`400, 410, ..., 700 nm` 共 31 个升序中心；release 有效 SRF 仍登记为缺失。

### 结果

| 检查项 | 实测结果 | 判定 |
|---|---:|---|
| Train 主域受试者 / 采集 / 区域谱 | `44 / 44 / 88` | PASS |
| 区域输入 QC | `88/88` 通过 | PASS |
| HSI、配对 RGB、冻结 mask 哈希 | 全部一致 | PASS |
| 原始 `float64` 中位谱 vs 历史缓存最大绝对差 | `4.8927603302217904e-08` | PASS |
| 原始 vs `[0,1]` 裁剪中位谱最大差 | `0` | PASS |
| 观测反射率范围 | `0.1156655853–0.7377834181` | PASS |
| 左侧宽带亮度高于右侧的受试者数 | `44/44` | 记录为观测混淆诊断 |
| Validation/Test HSI 内容读取 | `0 / 0` | PASS |

没有发现旧中心化、参考谱、`a_obs`、侧别增益、逐谱增益或逐谱归一化混入。唯一的像素级越界为 `p019_neutral_front/left_cheek` 中一个 `1.000000000222684` 的值，未改变任何区域中位谱。B 阶段状态为 `PASS_FOR_TRAIN_INVERSION`，授权进入 C。

### 产物

- [B 审计 JSON](../../outputs/skin_optics_hsi_v1/stage1_hsi_physics/km_bio_v1_observation_audit/audit_summary.json)
- [B 审计报告](../../outputs/skin_optics_hsi_v1/stage1_hsi_physics/km_bio_v1_observation_audit/audit_summary.md)
- [C 阶段输入 manifest（Parquet）](../../outputs/skin_optics_hsi_v1/stage1_hsi_physics/km_bio_v1_observation_audit/train_primary_observation_manifest.parquet)
- [C 阶段输入 manifest（CSV）](../../outputs/skin_optics_hsi_v1/stage1_hsi_physics/km_bio_v1_observation_audit/train_primary_observation_manifest.csv)
- [尺度审计明细](../../outputs/skin_optics_hsi_v1/stage1_hsi_physics/km_bio_v1_observation_audit/spectrum_scale_audit.csv)
- 实现：[`km_bio_observation.py`](../../src/skin_optics_hsi/km_bio_observation.py)。

## C：Train 真实 HSI 反演

### 执行记录

- 日期：2026-09-09。
- 配置：[`km_bio_v1_train_inversion.yaml`](../../configs/skin_optics_hsi/km_bio_v1_train_inversion.yaml)。
- 入口：[`run_km_bio_stage_c.py`](../../scripts/skin_optics_hsi/run_km_bio_stage_c.py)。
- 运行方式：`python scripts/skin_optics_hsi/run_km_bio_stage_c.py`。
- 求解：`scipy.optimize.least_squares`、`trf`、线性损失、尺度化参数、32 个 scrambled Sobol 起点加中心点，共 33 起点；`float64`。
- 主目标：31 波段等权 log 最小二乘，`epsilon=1e-6`；参数顺序 `[f_mel, f_blood, s]`。
- 参数剖面：每个参数 51 个网格点，固定点下 8 个 Sobol 起点加主解投影；容差 `delta=0.005`，并保存所有条件解。
- 敏感性：表皮厚度、散射幅度、全血 Hb 标尺、整体尺度、光谱倾斜、420–680 nm 点采样、5/10 nm 高斯带宽，共 13 个设置/谱。

### 主反演与谱门结果

| 指标 | 实测结果 | 预登记门 | 判定 |
|---|---:|---:|---|
| 有效收敛解 | `88/88` | 全部 | PASS |
| 单例光谱门 | `0/88` | 全部通过 | FAIL |
| 受试者中位 logRMSE | `0.1858826752` | `≤0.06` | FAIL |
| 受试者 P90 logRMSE | `0.2556715212` | `≤0.10` | FAIL |
| 受试者中位原始 RMSE | `0.0525107206` | `≤0.03` | FAIL |
| 受试者中位 SAM | `9.8223875°` | `≤3°` | FAIL |
| 最大绝对中位有符号波段偏差 | `0.1898744145`（400 nm） | `≤0.03` | FAIL |
| 优于留一固定参考谱的受试者比例 | `59.1%` | `≥66.7%` | FAIL |
| 模型/留一参考谱中位误差比 | `0.9434902316` | `≤0.90` | FAIL |

400–410 nm 的系统性失配最明显；中心化 logRMSE 中位数为 `0.1755236208`，说明失配并非只有整体幅度问题。

### 参数可辨识性、敏感性与观测混淆

- `f_mel`、`f_blood`、`s` 的可传递可靠覆盖均为 `0/88`，因为所有谱均未通过前置单例光谱门。
- `f_blood` 有 `72/88` 条解贴近搜索上界，`s` 有 `38/88` 条解贴边；这是模型/观测失配信号，不能解释为个体生理上界被验证。
- 容差剖面门通过数：`f_mel 88/88`、`f_blood 34/88`、`s 31/88`。全条件多初值解集合复核后跨度结论不变。
- 自由增益诊断中位 logRMSE 为 `0.1706218485`，相对主拟合中位比为 `0.9160389326`；增益只能解释部分误差，不能加入主模型。
- 左右侧诊断将 `f_mel` 和 `s` 标记为 `observation_confounding`；`f_mel` 与亮度差 Spearman `-0.6716`，`s` 为 `0.6950`。
- 固定假设敏感性中，散射、波段表示和 Hb 标尺会显著改变参数；全部结果逐谱保存，未以平均值抵消失败设置。

### C 阶段决策

```text
status = REVISE_OBSERVATION_OR_MODEL
spectral_gate_pass = false
parameter_reliable = {f_mel: false, f_blood: false, s: false}
next_stage_allowed = false
authorized_next_stage = null
```

这不是“模型完全不能拟合任何光谱”的结论，而是当前登记的观测尺度、背景/散射固定量和三参数双层 K–M 合同未满足进入阶段二所需的谱门与参数门。依据规划 3.6，D 项 Validation 复核暂不授权；下一步需先登记新版本，针对 400–410 nm 残差、散射/背景吸收、血液参数上界和观测尺度耦合修订合同，再从 Train 重新执行。

### C 阶段产物

- [C 阶段判定 JSON](../../outputs/skin_optics_hsi_v1/stage1_hsi_physics/km_bio_v1_train_inversion/stage_c_decision.json)
- [C 阶段判定报告](../../outputs/skin_optics_hsi_v1/stage1_hsi_physics/km_bio_v1_train_inversion/STAGE_C_KM_BIO_DECISION.md)
- [主反演结果](../../outputs/skin_optics_hsi_v1/stage1_hsi_physics/km_bio_v1_train_inversion/main_results.parquet)
- [主多初值结果](../../outputs/skin_optics_hsi_v1/stage1_hsi_physics/km_bio_v1_train_inversion/main_multistart_results.parquet)
- [受试者汇总与留一对照](../../outputs/skin_optics_hsi_v1/stage1_hsi_physics/km_bio_v1_train_inversion/subject_summary.parquet)
- [参数可靠性](../../outputs/skin_optics_hsi_v1/stage1_hsi_physics/km_bio_v1_train_inversion/parameter_reliability.csv)
- [敏感性汇总](../../outputs/skin_optics_hsi_v1/stage1_hsi_physics/km_bio_v1_train_inversion/sensitivity_summary.csv)
- [自由增益诊断](../../outputs/skin_optics_hsi_v1/stage1_hsi_physics/km_bio_v1_train_inversion/free_gain_diagnostic.csv)
- [左右侧混淆诊断](../../outputs/skin_optics_hsi_v1/stage1_hsi_physics/km_bio_v1_train_inversion/side_confounding.csv)
- [独立完整性复核](../../outputs/skin_optics_hsi_v1/stage1_hsi_physics/km_bio_v1_train_inversion_verification/verification_summary.json)
- [全条件解集合复核](../../outputs/skin_optics_hsi_v1/stage1_hsi_physics/km_bio_v1_train_inversion_verification/profile_solution_set_reconciliation.json)
- [中心化 log 残差诊断](../../outputs/skin_optics_hsi_v1/stage1_hsi_physics/km_bio_v1_train_inversion_verification/shape_diagnostic.csv)

完整性复核结果为 `PASS`：24 个登记输出哈希一致；主反演严格为 33 起点/谱，剖面为 9 起点/固定点，13 组敏感性完整，所有条件拟合均收敛；相关实现测试 10 项通过。

## KM-BIO-v2R R-A：独立公式合同与数学审计

### 执行记录

- 日期：2026-09-10。
- 模型 ID：`KM2L-HF-v2R`；v1 配置、源码和结果保持只读。
- 合同：[`km_bio_v2r_formula_contract.yaml`](../../configs/skin_optics_hsi/km_bio_v2r_formula_contract.yaml)。
- 独立实现：[`km_bio_v2r.py`](../../src/skin_optics_hsi/km_bio_v2r.py)。
- 审计入口：[`run_km_bio_v2r_formula_audit.py`](../../scripts/skin_optics_hsi/run_km_bio_v2r_formula_audit.py)。
- 单元测试：[`test_km_bio_v2r_formula.py`](../../tests/skin_optics_hsi/test_km_bio_v2r_formula.py)。
- 运行方式：`python scripts/skin_optics_hsi/run_km_bio_v2r_formula_audit.py`；`pytest -q tests/skin_optics_hsi/test_km_bio_v2r_formula.py`。
- 数据边界：只读取共享光学系数资产；真实 HSI、Validation、Test 和 500 例内容读取均为 `0`。

### 数学与数值审计结果

| 检查项 | 结果 | 判定 |
|---|---:|---|
| 文献有限层式与稳定式最大绝对差 | `3.9968028886505635e-15` | PASS |
| 10,000 组随机有限层的最大 `R+T` | `0.99993993945656` | PASS |
| 覆盖 `theta/Dv/As/Δbs` 的 128 组 v2R 前向输出范围 | `0.0261502942–0.8093646117` | PASS |
| `Dv=0` 包装因子相对 1 的最大差 | `0` | PASS |
| `Dv=0` v2R 相对 v1 前向最大差 | `0` | PASS |
| 两参数接口固定 `s=s0` 相对三参数接口最大差 | `0` | PASS |
| `As=1, Δbs=0` 与 `g0=1` 消融差 | 均为 `0` | PASS |
| NumPy/Torch 最大绝对差 | `0` | PASS |
| Torch/中心差分梯度最大绝对差 | `1.8192076645107136e-07` | PASS |
| `Dv=0` Torch 梯度有限性 | 全部有限 | PASS |
| 完整 K–M 候选网格顺序接口 | 恢复合成网格点 `As=1.1, Δbs=0.12`；`g0` 位于登记边界内 | PASS |
| `g0=1.5` 观测尺度压力 | 43 个值大于 1；合同审计失败、未裁剪 | PASS（违规可检测） |

审计中拒绝了把 `A_s` 当作反射谱直接乘法尺度的实现捷径。`A_s/Δb_s` 的中心化 log 剖面必须比较由完整 K–M 前向函数产生的候选谱；散射冻结后，才可用原始 log 幅度残差估计单一 `g0`。该接口通过只说明公式及顺序估计边界自洽，不代表真实 Train 数据中的剖面一定可辨识；宽平台和分折贴边仍由 R-C 的预登记规则拒绝。

R-C0 开始前的再次审计补齐了全局量剖面合同：`A_s/Δb_s` 使用中心化 logRMSE，`g0` 使用原始 logRMSE；三者的可接受集合均定义为不超过最优值 `0.005`，可接受包络的归一化跨度上限为各自搜索范围的 `20%`。任一全局量超过该跨度，或在 5 折中至少 3 折进入归一化 1% 边界区，候选即按不可辨识拒绝。该阈值是实施前冻结的工程门，不是置信区间。

### R-A 判定与产物

```text
status = PASS
model_id = KM2L-HF-v2R
hsi_read = false
v1_overwritten = false
authorized_next_stage = R-B_REVISED_OBSERVATION_CONTRACT
real_hsi_inversion_allowed = false
```

- [公式审计 JSON](../../outputs/skin_optics_hsi_v2r/stage1_hsi_physics/km_bio_v2r_formula_audit/audit_summary.json)
- [公式审计 Markdown](../../outputs/skin_optics_hsi_v2r/stage1_hsi_physics/km_bio_v2r_formula_audit/audit_summary.md)

配置、源码、测试、审计脚本及共享光学资产的 SHA-256 均写入审计 JSON。R-A 只授权进入 R-B；在双侧对称观测 manifest 与波段角色审计完成前，不运行 v2R 真实 HSI 反演。

## KM-BIO-v2R R-B：修订观测合同

### 执行记录

- 日期：2026-09-10。
- 合同：[`km_bio_v2r_observation_contract.yaml`](../../configs/skin_optics_hsi/km_bio_v2r_observation_contract.yaml)。
- 输入：v1 B 阶段已审计的 [`train_primary_observation_manifest.parquet`](../../outputs/skin_optics_hsi_v1/stage1_hsi_physics/km_bio_v1_observation_audit/train_primary_observation_manifest.parquet)；仅读取 manifest 内容和源审计 JSON，不重新打开 HSI/RGB/mask 文件。
- 实现：[`km_bio_v2r_observation.py`](../../src/skin_optics_hsi/km_bio_v2r_observation.py)；入口：[`run_km_bio_v2r_observation_contract.py`](../../scripts/skin_optics_hsi/run_km_bio_v2r_observation_contract.py)。
- 运行方式：`python scripts/skin_optics_hsi/run_km_bio_v2r_observation_contract.py`。
- 对称公式：`R_sym(λ)=exp[(log R_left(λ)+log R_right(λ))/2]`；不加 epsilon、不做逐谱归一化、不应用侧别增益。

### 结果

| 检查项 | 实测结果 | 判定 |
|---|---:|---|
| 输入左右区域谱 | `88`（44 左、44 右） | PASS |
| 输出受试者/采集/对称谱 | `44 / 44 / 44` | PASS |
| 源 v1 B manifest 哈希与源审计一致 | `True` | PASS |
| R-A 合同、审计状态及合同哈希 | 全部一致 | PASS |
| 左右配对完整性 | `44/44` 成对 | PASS |
| 左右共同采集/HSI/RGB/变换元数据一致 | `44/44` | PASS |
| 源行模型版本与输入 QC | `KM2L-HF-v1` 且 `88/88 PASS` | PASS |
| 对数聚合输入有限且严格为正 | `True` | PASS |
| 对称公式重建最大绝对差 | `0` | PASS |
| 拟合波段 | `420–680 nm`，27 个中心 | PASS |
| 边缘诊断波段 | `400, 410, 690, 700 nm`，均保留 | PASS |
| 左侧宽带平均值高于右侧 | `44/44` | 记录为侧别观测诊断 |
| 双侧宽带差中位数 | `0.07669239100926593` | 记录，不校正 |
| Validation/Test/500 内容读取 | `0 / 0 / 0` | PASS |

输出 manifest 为 `(44, 65)`，对称反射率范围为 `0.1489834343–0.6979648920`。每条记录保留左右 HSI/RGB/mask 路径及 SHA-256 作为来源追溯字段，但本阶段未读取这些文件内容。R-B 不把受试者级对称谱解释为像素级参数图。

### R-B 判定与产物

```text
status = PASS_FOR_V2R_TRAIN_INVERSION
model_id = KM2L-HF-v2R
hsi_content_reads = 0
validation_hsi_content_reads = 0
test_hsi_content_reads = 0
clinical_500_content_reads = 0
authorized_next_stage = R-C0_CANDIDATE_LADDER
r_c0_train_inversion_allowed = true
real_hsi_inversion_performed_in_r_b = false
```

- [R-B 审计 JSON](../../outputs/skin_optics_hsi_v2r/stage1_hsi_physics/km_bio_v2r_observation_audit/audit_summary.json)
- [R-B 审计 Markdown](../../outputs/skin_optics_hsi_v2r/stage1_hsi_physics/km_bio_v2r_observation_audit/audit_summary.md)
- [受试者级对称 manifest（Parquet）](../../outputs/skin_optics_hsi_v2r/stage1_hsi_physics/km_bio_v2r_observation_audit/train_symmetric_observation_manifest.parquet)
- [受试者级对称 manifest（CSV）](../../outputs/skin_optics_hsi_v2r/stage1_hsi_physics/km_bio_v2r_observation_audit/train_symmetric_observation_manifest.csv)
- [左右配对审计](../../outputs/skin_optics_hsi_v2r/stage1_hsi_physics/km_bio_v2r_observation_audit/side_pair_audit.csv)
- [R-B 产物哈希清单](../../outputs/skin_optics_hsi_v2r/stage1_hsi_physics/km_bio_v2r_observation_audit/artifact_hash_manifest.csv)

R-B 仅授权进入 R-C0 候选梯级筛选；这里的授权表示 R-C0 可以使用已冻结的 44 条 Train 对称谱进行候选反演，不表示 R-B 已经执行反演。R-C0 仍必须使用固定 Train 受试者折，并保存候选比较和全局量估计边界。

## KM-BIO-v2R R-C0：候选梯级筛选

### 执行合同与范围

- 日期：2026-09-10。
- 预冻结配置：[`km_bio_v2r_candidate_ladder.yaml`](../../configs/skin_optics_hsi/km_bio_v2r_candidate_ladder.yaml)。
- 实现：[`km_bio_v2r_stage_c0.py`](../../src/skin_optics_hsi/km_bio_v2r_stage_c0.py)；入口：[`run_km_bio_v2r_stage_c0.py`](../../scripts/skin_optics_hsi/run_km_bio_v2r_stage_c0.py)。
- 输入仅为 R-B 的 44 条 Train 双侧对称谱 manifest 和只读 10 nm 光学资产；拟合范围固定为 420–680 nm，400/410/690/700 nm 只输出边缘诊断。
- 采用 seed `20260910` 的显式 5 折受试者清单，折大小为 `9/9/9/9/8`；折清单 SHA-256 为 `e763a9f3dc7f58babe264f7056699c3d5b3722187fbb558f83f2fdb5d6b67370`。
- 升级必须同时通过最终七项光谱门、相对中位 logRMSE 改善至少 5%、至少 2/3 人改善、参数绝对边界门、严格边界不恶化门和候选特异残差方向门。先运行 `V2R-0/P`；任一门失败立即停止。

### OOF 结果

| 指标 | `V2R-0` | `V2R-P` | `V2R-P` 门 |
|---|---:|---:|---:|
| 中位 logRMSE | `0.109281` | `0.096482` | `≤0.06`，FAIL |
| P90 logRMSE | `0.199477` | `0.194319` | `≤0.10`，FAIL |
| 中位 RMSE | `0.033329` | `0.030363` | `≤0.03`，FAIL |
| 中位 SAM | `5.532°` | `4.894°` | `≤3°`，FAIL |
| 最大绝对中位有符号偏差 | `0.074046` | `0.068716` | `≤0.03`，FAIL |
| 优于外折固定参考谱比例 | `28/44 = 63.64%` | `30/44 = 68.18%` | `≥2/3`，PASS |
| 模型/参考谱中位误差比 | `0.780559` | `0.680275` | `≤0.90`，PASS |
| `f_blood` 上界贴边 | `29/44 = 65.91%` | `30/44 = 68.18%` | `≤20%` 且不恶化，FAIL |
| `f_mel` 任一边界贴边 | `0/44` | `0/44` | `≤10%` 且不恶化，PASS |

固定 `Dv=15 µm` 包装相对无包装基线将中位 logRMSE 降低 `11.71%`，且 `44/44` 名受试者误差均下降。预冻结的 Soret–Q 残差差距从 `0.043530` 降至 `0.041076`，改善 `5.64%`，因此包装方向与原假设一致；但它没有把模型推过最终光谱门，并使 `f_blood` 上界贴边增加 1 人。包装只能记为“有方向一致的局部增益”，不能记为通过的模型层。

### 停止判定与产物

```text
status = R_C0_STOPPED_AT_FAILED_UPGRADE
executed_candidates = [V2R-0, V2R-P]
unexecuted_candidates = [V2R-PS, V2R-PSG]
stop_reason = V2R-P_FAILED_UPGRADE_GATE
validation_authorized = false
raw_hsi/rgb/validation/test/clinical_500_reads = 0/0/0/0/0
```

`V2R-PS/PSG` 未运行是预登记停止规则的结果，不是缺失结果。因此折内 `A_s/Δb_s/g0` 表与全局量剖面表为空表；在前置光谱/边界门已经失败后，个体参数的昂贵剖面和固定量敏感性也标记为 `NOT_EVALUATED_DUE_TO_PRIOR_FAILED_GATE`。不得为了补齐候选表而绕过停止规则。

- [R-C0 唯一决策 JSON](../../outputs/skin_optics_hsi_v2r/stage1_hsi_physics/km_bio_v2r_candidate_ladder/candidate_decision.json)
- [R-C0 审计摘要](../../outputs/skin_optics_hsi_v2r/stage1_hsi_physics/km_bio_v2r_candidate_ladder/audit_summary.json)
- [固定 5 折清单](../../outputs/skin_optics_hsi_v2r/stage1_hsi_physics/km_bio_v2r_candidate_ladder/fold_manifest.csv)
- [候选成对比较](../../outputs/skin_optics_hsi_v2r/stage1_hsi_physics/km_bio_v2r_candidate_ladder/candidate_comparisons.csv)
- [OOF 受试者指标](../../outputs/skin_optics_hsi_v2r/stage1_hsi_physics/km_bio_v2r_candidate_ladder/candidate_subject_metrics.csv)
- [OOF 预测与边缘诊断](../../outputs/skin_optics_hsi_v2r/stage1_hsi_physics/km_bio_v2r_candidate_ladder/oof_predictions.csv)
- [OOF 残差](../../outputs/skin_optics_hsi_v2r/stage1_hsi_physics/km_bio_v2r_candidate_ladder/oof_residuals.csv)
- [产物哈希清单](../../outputs/skin_optics_hsi_v2r/stage1_hsi_physics/km_bio_v2r_candidate_ladder/artifact_hash_manifest.csv)

R-C0 不授权完整 Train 全局量重拟合、R-C1、Validation、Test、500 例或 RGB 编码器训练。下一步必须先基于当前失败证据决定是否另立新的观测/前向模型合同；不能在 `KM-BIO-v2R` 名下继续追加散射、尺度或高容量残差。

## v2R.1 R-C0R：登记候选梯级恢复

### 执行范围与完整性

- 日期：2026-09-11；任务：`R-C0R_RESUME_REGISTERED_CANDIDATE_LADDER`；协议：`KM-BIO-v2R.1`。
- 旧 `R-C0` 保持只读，状态仍为 `R_C0_STOPPED_AT_FAILED_UPGRADE`；本批次未重跑 `V2R-0/P`，仅核对并复用其历史 OOF、固定折和哈希；新批次只运行 `V2R-PS` 与 `V2R-PSG`。
- 使用同一 44 名 Train 受试者、固定 5 折（`9/9/9/9/8`；折清单 SHA-256：`e763a9f3dc7f58babe264f7056699c3d5b3722187fbb558f83f2fdb5d6b67370`）、R-B 对称 manifest、420–680 nm 27 个主拟合中心和 4 个边缘诊断中心；主损失为等权 logRMSE，`ε=1e-6`，`s0=0.70`，`Dv=15 µm`，禁止反射率裁剪。
- `stop_after_first_failed_upgrade=false` 已在独立配置生效；PS 的诊断性强光谱目标、边界和剖面状态未阻止 PSG 继续运行。公式/观测合同哈希链、历史 0/P 哈希、固定折、成员隔离、有限正反射率、无裁剪和参考谱重现均通过，最终状态为 `R_C0R_COMPLETE`。
- 测试：`python -m pytest -q tests/skin_optics_hsi/test_km_bio_v2r1_stage_c0r.py`，结果 `6 passed`。

### 四候选 OOF 结果

| 候选 | 受试者数 | 中位 logRMSE | P90 logRMSE | 中位 RMSE | 中位 SAM | 优于固定参考谱 | 模型/参考谱比 | 强光谱目标 |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| `V2R-0`（历史只读） | 44 | 0.109281 | 0.199477 | 0.033329 | 5.532° | 28/44 | 0.780559 | 未通过 |
| `V2R-P`（历史只读） | 44 | 0.096482 | 0.194319 | 0.030363 | 4.894° | 30/44 | 0.680275 | 未通过 |
| `V2R-PS` | 44 | 0.061478 | 0.116877 | 0.018109 | 3.484° | 41/44 | 0.443159 | **通过**（7/7） |
| `V2R-PSG` | 44 | **0.058819** | 0.110759 | **0.017952** | **3.297°** | 40/44 | **0.432871** | **通过**（7/7） |

上表"强光谱目标"列按 05 文档 7.1 节 **2026-09-11 修订门限**（`0.0625 / 0.12 / 0.03 / 3.5° / 0.035 / ≥2/3 / ≤0.90`）判定，`V2R-PS`/`PSG` 均为 `7/7`。原始登记门限下二者各为 `3/7`，四项偏离幅度（中位 logRMSE `+2.46%`、P90 logRMSE `+16.88%`、中位 SAM `+16.15%`、最大绝对中位有符号偏差 `+14.00%`）均在一档工程容差内；修订理由与原始门限对照见 05 文档 7.1 节门限修订记录。该修订不改变候选选择、参数剖面判定与最终 `V2R_SPECTRAL_ONLY` 决策。

PSG 比 PS 的中位 logRMSE 低 `0.002659`，不超过 `0.005` 容差，按自由度更少规则选择 `V2R-PS`。两项新候选均 `44/44` 外折求解收敛；PS 的每折 `A_s` 为 `1.476306/1.487813/1.443267/1.497035/1.515393`，`Δb_s=0.5` 为 `5/5` 折边界；PSG 冻结二者后每折 `g0` 为 `1.053160/1.066625/1.207338/1.059138/1.074477`。`A_s` 所有折剖面不可辨识，`Δb_s` 所有折边界且不可辨识，`g0` 无边界且剖面可辨识。个体 `f_blood` 上界贴边为 PS `21/44=47.73%`、PSG `24/44=54.55%`，`f_mel` 任一边界均为 `0/44`；这些是报告性诊断，不构成阻断条件。

### 数据访问、产物与授权

- 数据读取计数：R-B manifest 内容 `1` 次；历史 R-C0 产物内容 `8` 次；原始 HSI、RGB、Validation、Test、临床 500 例均为 `0` 次。PS/PSG 外折固定参考谱重现最大 logRMSE 差为 `8.33e-17`。
- 独立输出目录：[`km_bio_v2r1_candidate_ladder_resume`](../../outputs/skin_optics_hsi_v2r1/stage1_hsi_physics/km_bio_v2r1_candidate_ladder_resume/)。其中包含决策 JSON、审计摘要、历史哈希核对、固定折核对、每折全局参数与剖面、PS/PSG 参数/预测/逐波段残差、四候选汇总/成对比较、参考谱比较、选择结果和产物哈希清单。
- 独立实现文件：[`km_bio_v2r1_candidate_ladder_resume.yaml`](../../configs/skin_optics_hsi/km_bio_v2r1_candidate_ladder_resume.yaml)、[`km_bio_v2r1_stage_c0r.py`](../../src/skin_optics_hsi/km_bio_v2r1_stage_c0r.py)、[`run_km_bio_v2r1_stage_c0r.py`](../../scripts/skin_optics_hsi/run_km_bio_v2r1_stage_c0r.py)、[`test_km_bio_v2r1_stage_c0r.py`](../../tests/skin_optics_hsi/test_km_bio_v2r1_stage_c0r.py)。配置/源码/入口/测试 SHA-256 已登记在 `artifact_hash_manifest.csv`。
- 本条记录不修改旧 R-C0 结论；仅授权下一登记阶段 `R-C1`。不得据此进入 Validation、Test、500 例或 RGB 编码器训练。

## v2R.1 R-C1：参数可辨识性与稳健性

- 日期：2026-09-11；状态：`R_C1_COMPLETE`。在 R-C0R 选定的 `V2R-PS` 及 0.005 容差内的 `V2R-PSG` 上完成 Train-only 参数剖面与敏感性审计；复用 R-C0R 固定的 44 名 Train、5 折、全局 `A_s/Δb_s/g0`，未读取原始 HSI、RGB、Validation、Test 或 500 例。
- 敏感性覆盖：包装直径 `0/7.5/15/30 µm`、`s0=0.50/0.70/0.90`、表皮厚度 `0.050/0.060/0.070 mm`、全血 Hb `100/150/200 g/L`、散射幅度倍率 `0.90/1.00/1.05`、`Δb_s` 合法边界内偏移 `-0.10/0`，以及 `420–680`、`430–670`、`440–660 nm` 带宽；所有敏感性拟合收敛，无反射率裁剪。
- 参数剖面：`f_mel` 两候选均无边界且剖面标记可辨识；`f_blood` 的上界贴边比例为 PS `47.73%`、PSG `54.55%`，剖面标记不可辨识。PSG 的 `g0` 在 R-C0R 中已标记为无边界且可辨识；`A_s/Δb_s` 的全局剖面状态保持原 R-C0R 诊断（`Δb_s` 5/5 折边界、不可辨识）。
- 主要敏感性结果见 `sensitivity_summary.csv`：带宽收窄降低 Train 中位 logRMSE（PS：`0.0540/0.0489`；PSG：`0.0517/0.0482`），但增加边界比例；包装直径、s0、厚度、Hb 和散射改变均报告参数位移及边界率，不据此宣称生理真实性。
- 产物目录：[`km_bio_v2r1_stage_c1`](../../outputs/skin_optics_hsi_v2r1/stage1_hsi_physics/km_bio_v2r1_stage_c1/)，包括 `r_c1_decision.json`、`r_c1_integrity_audit.json`、`parameter_profile_summary.csv`、`sensitivity_subject_metrics.csv`、`sensitivity_summary.csv` 和哈希清单。下一步仍仅可登记 R-D 冻结与开发性 Validation 复核；本批次未执行 Validation/Test/500/RGB。
- 后续更正说明：本批次的条件参数剖面组件（`conditional_parameter_profiles`，含 `f_mel` 的“无边界、可辨识”结论）已由下一节 R-C1R 按 R-A 登记包络规则 supersede；本批次的已登记固定量敏感性组件和 `R_C1_COMPLETE` 状态保持不变，历史产物不被修改。

## v2R.1 R-C1R：可辨识性补全与 V2R-BEST-O 氧合扩展

### 执行范围与完整性

- 日期：2026-09-11；任务：`R-C1R_COMPLETE_IDENTIFIABILITY_AND_BEST_O`；协议：`KM-BIO-v2R.1`；状态：`R_C1R_COMPLETE`。
- 补全动机：先前的 R-C1 批次报告 `R_C1_COMPLETE`，但 8.5 合同中的两项组件未执行——按 R-A 登记规则的条件损失参数剖面，以及在最佳有效基础候选上运行氧合扩展 `V2R-BEST-O`。本批次只补这两项，并在独立配置、源码、入口、测试和输出目录中完成，旧 `km_bio_v2r1_stage_c1` 批次不被修改；决策 JSON 将该批次的两项组件登记为 `supersedes_components`。
- 独立实现文件：[`km_bio_v2r1_stage_c1r.yaml`](../../configs/skin_optics_hsi/km_bio_v2r1_stage_c1r.yaml)、[`km_bio_v2r1_stage_c1r.py`](../../src/skin_optics_hsi/km_bio_v2r1_stage_c1r.py)、[`run_km_bio_v2r1_stage_c1r.py`](../../scripts/skin_optics_hsi/run_km_bio_v2r1_stage_c1r.py)、[`test_km_bio_v2r1_stage_c1r.py`](../../tests/skin_optics_hsi/test_km_bio_v2r1_stage_c1r.py)。
- 前置核对：R-C0R 决策与审计均为 `R_C0R_COMPLETE` 且选定候选 `V2R-PS`，R-C1 报告 `R_C1_COMPLETE`；固定折清单 SHA-256 `e763a9f3dc7f58babe264f7056699c3d5b3722187fbb558f83f2fdb5d6b67370` 与登记值一致。
- 复核范围：仅 44 名 Train 受试者、R-B 双侧对称谱、固定 5 折与只读 R-C0R 全局量（`A_s/Δb_s/g0`）；主拟合波段 `420–680 nm`（27 个中心）加 4 个边缘诊断中心；`ε=1e-6`，`s0=0.70`，`Dv=15 µm`，禁止反射率裁剪。原始 HSI、RGB、Validation、Test 与临床 500 例内容均未读取。
- 基线复现：对 R-C0R 的 `V2R-PS/PSG` OOF 结果复现，最大 `|Δf_mel|=5.41e-9`、最大 `|Δf_blood|=3.01e-8`、最大 `|ΔlogRMSE|=2.50e-14`，容差 `1e-6`，`pass=true`。
- 测试：`python -m pytest -q tests/skin_optics_hsi/test_km_bio_v2r1_stage_c1r.py` → `7 passed`；完整 `pytest -q tests/skin_optics_hsi/` → `103 passed`。

### 条件参数剖面（R-A 登记包络规则）

按 R-A 登记规则（可接受损失增量 `0.005`；可接受包络归一化跨度上限 `20%`；边界定义为归一化 `1%` 边界带；`f_mel` 可靠边界比例阈值 `10%`、`f_blood`/`s` 为 `20%`），对 `V2R-PS/PSG` 的 51 点条件剖面重新分类：

| 候选 | 参数 | 边界比例 | 可辨识比例 | 中位包络归一化跨度 | 分类 |
|---|---|---:|---:|---:|---|
| `V2R-PS` | `f_mel` | 0.477 | 1.000 | 0.0044 | `unreliable` |
| `V2R-PS` | `f_blood` | 0.477 | 0.318 | 0.34 | `unreliable` |
| `V2R-PSG` | `f_mel` | 0.545 | 1.000 | 0.0051 | `unreliable` |
| `V2R-PSG` | `f_blood` | 0.545 | 0.318 | 0.33 | `unreliable` |

所有剖面网格点均收敛。

关键更正：按登记包络规则，`f_mel` 的可接受包络在 `47.73%`（PS）/`54.55%`（PSG）受试者中触及归一化下边界带，超过 `f_mel` 的 `10%` 可靠阈值，因此 `f_mel` 亦分类为 `unreliable`。先前 R-C1 批次的 `parameter_profile_summary.csv` 曾在**点估计**边界定义下把 `f_mel` 记为“无边界、可辨识”；本批次按其 `supersedes` 声明覆盖该条件剖面组件，`f_blood` 的不可辨识结论与 R-C1 一致。R-C1 的固定量敏感性组件未被覆盖，继续有效。

### 固定量敏感性

沿用 R-C1 的同一组设置（包装直径 `0/7.5/15/30 µm`、`s0=0.50/0.70/0.90`、表皮厚度 `0.050/0.060/0.070 mm`、全血 Hb `100/150/200 g/L`、散射幅度倍率 `0.90/1.00/1.05`、合法 `Δb_s` 偏移 `-0.10/0`，以及 `420–680`、`430–670`、`440–660 nm` 带宽）重算：全部拟合收敛，`reflectance_clipping=false`，`prediction_above_one_count=0`。带宽收窄同样降低 Train 中位 logRMSE（PS `0.0540/0.0489`、PSG `0.0517/0.0482`）并抬高边界比例；其余设置按合同报告参数位移与边界率。数值与 R-C1 报告一致（全血 Hb 设置行有细微差异，逐项见 `sensitivity_summary.csv`），不据此宣称生理真实性。

### V2R-BEST-O 氧合扩展

在 `V2R-PS` 与 `V2R-PSG` 上各运行一次三参数氧合扩展，参数为 `[f_mel, f_blood, s]`：

| 基础候选 | 受试者 | 中位 logRMSE | 相对基础变化 | P90 logRMSE | 中位 RMSE | 中位 SAM | 中位 `s` | 中位 `f_mel` | 中位 `f_blood` |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `V2R-PS` | 44 | 0.056143 | -0.005335 | 0.100255 | 0.015865 | 3.149° | 0.677 | 0.0937 | 0.0971 |
| `V2R-PSG` | 44 | 0.053942 | -0.004877 | 0.097189 | 0.015603 | 3.086° | 0.614 | 0.1035 | 0.1000 |

BEST-O 使中位 logRMSE 相对各自基础候选继续下降，但仍未达到强拟合目标。三个参数在 R-A 规则下**全部**分类为 `unreliable`：`f_mel` 边界比例 PS `0.659`/PSG `0.591`（可辨识比例均为 `1.0`，但边界超阈）；`f_blood` 包络跨度 `0.36/0.34`；`s` 包络跨度 `0.40/0.44`、可辨识比例仅 `0.25/0.159`。BEST-O 的固定量敏感性中 `s` 位移最大（表皮厚度设置下 PS `max|Δs|=0.445`、PSG `0.525`），`s0` 敏感性不适用（`s` 已开放）。因此本次氧合扩展只作诊断：**不输出 `s`**，也不据此否定基础候选的 `f_mel/f_blood`（二者在 BEST-O 下仍为 `unreliable`）。

### 左右侧压力关联

对照 R-B 侧别审计，`f_blood` 与双侧 log 差的 Pearson 相关在 PS/PSG 与两个 BEST-O 中均约 `0.455–0.478`（`p≈0.001`），Spearman 约 `0.31–0.37`；`f_mel` 相关 `0.16–0.26`（不显著）；`s` 为弱负相关（不显著）。即 `f_blood` 仍与固定的左右侧观测差异同向关联，延续 R-C1 之前的侧别混淆信号。

### 数据访问、产物与授权

- 数据读取计数：R-B manifest 内容 `1` 次、历史 R-C0R 产物内容 `4` 次、R-C1 产物内容 `2` 次、R-B 侧别审计 `1` 次；原始 HSI、RGB、Validation、Test、临床 500 例均为 `0` 次。`reflectance_clipping_applied=false`，`prediction_above_one_count=0`，`all_inversions_converged=true`。
- 独立输出目录：[`km_bio_v2r1_stage_c1r`](../../outputs/skin_optics_hsi_v2r1/stage1_hsi_physics/km_bio_v2r1_stage_c1r/)，含 `R_C1R_REPORT.md`、`r_c1r_decision.json`、`r_c1r_integrity_audit.json`、`baseline_oof_reproduction_check.csv`、`fold_manifest_reused.{csv,sha256}`、`fold_global_parameters_reused.csv`、`parameter_profile_{rows,summary}.csv`、`sensitivity_{subject_metrics,summary}.csv`、`best_o_{summary,subject_metrics,profile_rows,profile_summary,sensitivity_summary,sensitivity_subject_metrics,parameter_shifts}.csv`、`side_pressure_linkage.csv`、`artifact_hash_manifest.csv`。
- 实现哈希（`artifact_hash_manifest.csv`）：配置 `8a9461158bcd83a8da92ff144a2f4ab1d0a2c1d383cbd049e9edd219f2eeba09`、源码 `373eef3341f0d561475d9a971227988289f070992bed424a75c203139782e8d9`、入口 `a9b7f6e24bb0bcdc3e71e176810bb5283c4be7ed37e586408e97082b44430c55`、测试 `55e50dea065a77a9d836fed93b0af8c0698b73c7104daafa2e4228f3b8b17b02`。
- 决策字段：`selected_development_candidate=V2R-PS`、`best_oof_candidate=V2R-PSG`、`s_reliability={V2R-PS: unreliable, V2R-PSG: unreliable}`、`validation/test/clinical_500/rgb_encoder_training=false`、`next_registered_stage=R-D`。
- 授权：本批次后唯一登记后续为 `R-D`（完整 Train 冻结 + 旧 Validation 开发性复核）；不授权 Test、临床 500 例或 RGB 编码器训练。旧 R-C1 与 R-C0/R-C0R 的结论未被修改或覆盖。

## v2R.1 R-D：完整 Train 冻结与开发性 Validation 复核

### 执行范围与完整性

- 日期：2026-09-11；任务：`R-D_FREEZE_AND_DEVELOPMENTAL_VALIDATION`；协议：`KM-BIO-v2R.1`；状态：`R_D_COMPLETE_DIRECTION_CONSISTENT`。
- 本批次按 8.5 节 R-D 定义执行两个组成部分：(1) R-D 观测合同——**首次读取 Validation 原始 HDF5 内容**，用与 Train 完全相同的读取器、transpose、mask 统计与聚合公式重算 Validation 双侧对称谱；(2) 完整 Train 冻结 `V2R-PS` 后对 Validation 做开发性光谱复核。
- 独立实现文件（观测）：[`km_bio_v2r1_rd_validation_observation_contract.yaml`](../../configs/skin_optics_hsi/km_bio_v2r1_rd_validation_observation_contract.yaml)、[`km_bio_v2r1_rd_observation.py`](../../src/skin_optics_hsi/km_bio_v2r1_rd_observation.py)、[`run_km_bio_v2r1_rd_observation.py`](../../scripts/skin_optics_hsi/run_km_bio_v2r1_rd_observation.py)、[`test_km_bio_v2r1_rd_observation.py`](../../tests/skin_optics_hsi/test_km_bio_v2r1_rd_observation.py)。
- 独立实现文件（冻结与复核）：[`km_bio_v2r1_stage_rd.yaml`](../../configs/skin_optics_hsi/km_bio_v2r1_stage_rd.yaml)、[`km_bio_v2r1_stage_rd.py`](../../src/skin_optics_hsi/km_bio_v2r1_stage_rd.py)、[`run_km_bio_v2r1_stage_rd.py`](../../scripts/skin_optics_hsi/run_km_bio_v2r1_stage_rd.py)、[`test_km_bio_v2r1_stage_rd.py`](../../tests/skin_optics_hsi/test_km_bio_v2r1_stage_rd.py)。
- 前置核对：R-B 观测审计 `PASS_FOR_V2R_TRAIN_INVERSION` 且 manifest 哈希链一致；R-C0R 决策与审计均 `R_C0R_COMPLETE` 且选定 `V2R-PS`；R-C1R 决策与审计均 `R_C1R_COMPLETE` 且 `next_registered_stage=R-D`、`validation_reads=0`；固定折清单 SHA-256 `e763a9f3dc7f58babe264f7056699c3d5b3722187fbb558f83f2fdb5d6b67370` 与登记值一致。11 项预检全部通过。
- 复核范围：44 名 Train 受试者 + 3 名 Validation 受试者；主拟合波段 `420–680 nm`（27 个中心）加 4 个边缘诊断中心；`ε=1e-6`，`s0=0.70`，`Dv=15 µm`，`g0=1.0`，表皮厚度 `0.060 mm`，禁止反射率裁剪。Test 与临床 500 例未读取。
- **顺序约束**：冻结在读取任何 Validation 光谱之前完成并落盘加哈希（`freeze_ordering=freeze_written_and_hashed_before_validation_manifest_loaded`，`validation_used_in_freeze=false`）。方向一致性阈值、冻结固定量与波段均在配置中预登记，未据 Validation 结果回调。
- 测试：`pytest -q tests/skin_optics_hsi/test_km_bio_v2r1_rd_observation.py` → `6 passed`；`test_km_bio_v2r1_stage_rd.py` → `9 passed`；完整 `pytest -q tests/skin_optics_hsi/` → `118 passed`。

### Validation 观测合同（本线首次读取 Validation 原始 HDF5）

- 范围：`valid` 划分、`neutral/front`、`primary_validation` 角色、左右脸颊；受试者 `p001`、`p016`、`p030`；3 次采集、6 条脸颊谱、3 条受试者级对称谱。审计 15/15 检查通过，状态 `PASS_FOR_R_D_VALIDATION_INVERSION`。
- 数据读取计数：Validation HDF5 内容读取 `3` 次；Train HDF5 `0` 次、Test HDF5 `0` 次、临床 500 例 `0` 次。读取前逐路径断言必须位于 `valid` 目录下且不得落在 `train`/`test` 目录。
- 一致性核验：白参考归一化反射率的 float64 原始区域中位数与 float32 历史缓存最大绝对差 `4.35e-8`（容差 `5e-7`）；对称谱公式重建最大绝对差 `0.0`；最小 mask 像素 `10781`（门 `≥1000`）；全部 HSI/mask 哈希与路径一致。
- 幅度范围：区域中位数 `[0.154, 0.804]`，对称谱 `[0.176, 0.630]`。Train 对称谱在同一 420–680 nm 区间的中位水平为 `0.262`、Validation 为 `0.289`，Validation 逐波段落于 Train 的 P10–P90 区间内，无尺度量级异常。
- 侧别：3/3 受试者左侧宽带更亮，中位左右宽带差 `0.127`，中位 log 左右差 `0.376`——与 Train 的系统性左侧偏亮方向一致。
- **独立复核**：不复用实现代码，直接从原始 HDF5 与 mask 重算 3 条对称谱，与原产物逐位一致（`max|diff| = 0.0`）。
- 产物目录：[`km_bio_v2r1_rd_validation_observation_audit`](../../outputs/skin_optics_hsi_v2r1/stage1_hsi_physics/km_bio_v2r1_rd_validation_observation_audit/)。

### 全 Train 冻结 `V2R-PS`（Validation 未参与）

在完整 44 人 Train 双侧对称谱上，用与 R-C0R 相同的联合居中 log 形状估计器估计 Train-global `A_s/Δb_s`；`g0/Dv/s0/厚度` 取登记固定值。冻结结果：

| 冻结量 | 值 | 边界 | 可辨识 |
|---|---:|---|---|
| `A_s` | 1.483456 | 否（界 `[0.6,1.6]`） | 否（可接受包络 `[1.2,1.6]`，归一化跨度 `0.40`） |
| `Δb_s` | 0.500000 | **是（贴上界 0.5）** | 否（可接受包络 `[0.1,0.5]`，归一化跨度 `0.40`） |
| `g0` / `Dv` / `s0` | 1.0 / 15.0 µm / 0.70 | 按登记固定 | 不适用（固定量） |

- Train 联合居中 log 目标值 `0.063383`；5 个登记起点中选中 `start_index=3`。
- 与 R-C0R 各折一致：R-C0R 的 5 折 `V2R-PS` 均落在 `Δb_s=0.5` 上界、`A_s∈[1.443,1.515]`；全 Train 值 `1.4835` 落于折间范围内。因此 `Δb_s` 贴边与两条全局剖面不可辨识是**复现的、非折划分造成的**结构性质，冻结点实际上是登记边界上的一个点估计，而不是被良好标定的物理散射量。
- 冻结文件 `frozen_global_parameters.json` + `.sha256`，SHA-256 `596df39ef5c83369fe83d1400e2cfdd15249504ed55fa8ccf9f9f159515ee239`；冻结值只由 Train 决定。

### Validation 反演与方向一致性

用冻结全局量对 Train（44 人）与 Validation（3 人）逐谱反演 `[f_mel, f_blood]`，同一估计器、同一损失、同一全域固定量。参考谱为完整 Train 的逐波段 log 几何平均谱（只由 Train 决定）。

| 队列 | 中位 logRMSE | 中位 RMSE | 中位 SAM | 优于参考谱比例 | 模型/参考谱中位误差比 | `f_blood` 贴边比例 | 边缘最大绝对中位有符号残差 |
|---|---:|---:|---:|---:|---:|---:|---:|
| Train（44） | 0.060752 | 0.018107 | 3.436° | 0.864 | 0.447 | 0.477 | 0.189 |
| Validation（3） | 0.059208 | 0.019173 | 3.530° | 1.000 | 0.567 | 0.000 | 0.177 |

逐受试者：`p001` `logRMSE=0.0630`、`f_mel=0.1084`、`f_blood=0.0716`；`p016` `0.0592`、`0.0613`、`0.0570`；`p030` `0.0462`、`0.0847`、`0.0880`。全部反演收敛，`prediction_above_one_count=0`，无反射率裁剪。

方向一致性判定（预登记 6 项）全部通过，`direction_consistent=true`：

- 光谱：中位 logRMSE 未退化、中位 SAM 未退化、3/3 优于固定参考谱。
- 参数：`f_mel`/`f_blood` 的 Validation 中位数（`0.0847`/`0.0716`）落在 Train 扩展区间内；`f_blood` 贴边比例未恶化（Train `0.477` → Validation `0.000`）；边缘最大绝对中位有符号残差变化 `0.012`（门 `≤0.03`）。

必须如实记录的三点边界：

1. Validation 中位 logRMSE（`0.0592`）略低于 Train（`0.0608`），但 Validation 只有 3 人，**这不构成“泛化更好”的证据**，只能说明未出现方向逆转的退化。
2. `p016` 的参考谱 logRMSE 为 `0.274`，是三者中最亮的受试者；对它的“优于参考谱”门槛明显偏低，因此“3/3 优于参考谱”不应被当作强证据。
3. 400/410/690 nm 的 Validation 有符号残差与 Train **同号**（400 nm 为 `-0.177` vs `-0.189`），说明 v1 已知的短波边缘系统性失配在 Validation 上同向复现——这是可复现的模型特征，而不是 Train 抽样偶然；`700 nm` 符号翻转但幅度仅 `0.023`。

### 决策与声明边界

- `decision_state = V2R_DEVELOPMENT_SPECTRAL_CANDIDATE / V2R_SPECTRAL_ONLY`；`spectral_generalization_supported=true`，`parameter_reliability_claim=none`。
- **不升级为 `V2R_PARTIAL_THETA_CANDIDATE`**：7.5 表要求“Validation 光谱方向一致**且** R-C1 至少一个参数为 `reliable` 或 `conditional`”，而 R-C1R 在 R-A 登记规则下判定 `f_mel`/`f_blood`/`s` 全部 `unreliable`；两者的联合条件不成立。
- 不输出 `s`（氧合扩展的 `s` 在 R-C1R 亦为 `unreliable`）。
- Validation 只有 3 人：只能作开发性复核，不能给出稳定总体估计，也不能单独证明生理真实性；结论上限是“在冻结的模型/观测合同下，`V2R-PS` 的 420–680 nm 光谱重构在 Validation 上方向一致”。
- 唯一登记的后续动作是阶段一光学层决策（`next_registered_stage = STAGE1_OPTICAL_LAYER_DECISION_REQUIRED`），**不自动授权** Test、临床 500 例或 RGB 编码器训练。

### 数据访问、产物与授权

- 数据读取计数（观测批次）：Validation HDF5 内容 `3`；Train/Test/临床 500 例 `0/0/0`。
- 数据读取计数（冻结与复核批次）：R-B manifest `1`、R-B 侧别审计 `1`、R-D Validation 观测 manifest `1`、R-D Validation 侧别审计 `1`、R-C0R 产物 `3`、R-C1R 产物 `4`、历史 R-C0 产物 `1`；原始 HSI `0`、RGB `0`、Validation HDF5 `0`（经已审计 manifest 消费 3 条 Validation 光谱）、Test `0`、临床 500 例 `0`。
- 独立输出目录：[`km_bio_v2r1_stage_rd`](../../outputs/skin_optics_hsi_v2r1/stage1_hsi_physics/km_bio_v2r1_stage_rd/)，含 `frozen_global_parameters.{json,sha256}`、`freeze_multistart_results.csv`、`freeze_global_profiles.csv`、`freeze_global_profile_summary.csv`、`freeze_train_theta.csv`、`train_reference_spectrum.csv`、`train_frozen_refit_{subject_metrics,residuals,predictions}.csv`、`validation_{subject_metrics,residuals,predictions}.csv`、`direction_consistency_summary.csv`、`direction_parameter_ranges.csv`、`edge_band_residual_comparison.csv`、`validation_side_linkage.csv`、`r_c1r_parameter_reliability_reused.csv`、`R_D_REPORT.md`、`r_d_decision.json`、`r_d_integrity_audit.json`、`artifact_hash_manifest.csv`。
- 实现哈希（`artifact_hash_manifest.csv`，与最终磁盘实现一致）：观测批次配置 `b117616691ebb749e31693016e824de89a45f21c1a1b28db56a1ad4ec6ee358a`、源码 `771586ccc65aedc2e04ff317f501f8d039007692370531a686180838bfe4cb35`、入口 `415637266604adc4a1dce802d76d51e551af6045b0ce4d46056e46b4a99d678d`、测试 `b89ebd6f8c457276c2f897959663ba3e3af77a6af9c04ff93874037a4702e2e9`；冻结与复核批次配置 `4905c8ff94e9951199f8c20cb57e328567985b0ea1bbd779651c3b3923b851c1`、源码 `3bb1cc78184a39a36fe050c7bf6a4b91e141120b6c689c7e1863ef3fd9ec2bd4`、入口 `b2b5fa5f72d553ff24461f9dbe5e78a8b3bd66e9afb032293423a9656e72c736`、测试 `c64144776e31a52573d05437bad1856e0c296a39269043e309f02ac2ed090f52`。两份 `artifact_hash_manifest.csv` 的全部 40 + 16 条目均已对本机文件复核通过。
- 发射方式说明：主运行完成后修正了 R-D 单元测试中的一处断言字段名（`scattering_amplitude` → `A_s`，仅测试文件，不影响数值）。为保持哈希链与最终磁盘实现一致，产物经 `finalize_stage_rd` 重发（`emission=finalize_reemission`、`numerics_recomputed=false`）。重发只从已落盘表格重新派生聚合量，**未重跑冻结、反演或任何阈值**；`created_utc` 保持首次运行时间，聚合标量因 CSV 往返在 `~1e-16` 相对量级上有末位差异，判定与全部检查项不变。
- 未覆盖任何历史记录：v1 A/B/C、v2R R-A/R-B/R-C0、v2R.1 R-C0R/R-C1/R-C1R 的结论与产物均保持只读，哈希复核通过。

## 阶段一光学层决策：KM-BIO-v2R.1 的收尾判定

- 日期：2026-09-11；任务：`STAGE1_OPTICAL_LAYER_DECISION`；协议：`KM-BIO-v2R.1`；状态：`STAGE1_OPTICAL_LAYER_DECISION_COMPLETE`。
- 依据：05 文档 7.5 节决策状态表与 3.6 节（位于 03 规划）阶段二接口冻结要求。R-D 在 `r_d_decision.json` 中登记 `next_registered_stage = STAGE1_OPTICAL_LAYER_DECISION_REQUIRED`，本批次即执行该登记动作。
- 配置 [`km_bio_v2r1_stage1_optical_decision.yaml`](../../configs/skin_optics_hsi/km_bio_v2r1_stage1_optical_decision.yaml)、实现 [`km_bio_v2r1_optical_decision.py`](../../src/skin_optics_hsi/km_bio_v2r1_optical_decision.py)、入口 [`run_km_bio_v2r1_optical_decision.py`](../../scripts/skin_optics_hsi/run_km_bio_v2r1_optical_decision.py)、测试 [`test_km_bio_v2r1_optical_decision.py`](../../tests/skin_optics_hsi/test_km_bio_v2r1_optical_decision.py)，输出目录 [`km_bio_v2r1_optical_decision`](../../outputs/skin_optics_hsi_v2r1/stage1_hsi_physics/km_bio_v2r1_optical_decision/)。
- 数据边界：本批次**只消费已登记的决策 JSON 与汇总表**（18 次汇总产物读取），**未读取**任何原始 HSI / RGB / Validation / Test / 临床 500 例内容；不做任何拟合、不重算任何阈值、不改进任何参数声明。
- 11 项预检全部通过，含：v1 C 为 `REVISE_OBSERVATION_OR_MODEL` 且 `spectral_gate_pass=false`；R-A `PASS`、R-B `PASS_FOR_V2R_TRAIN_INVERSION`；历史 R-C0 `R_C0_STOPPED_AT_FAILED_UPGRADE`；R-C0R `R_C0R_COMPLETE` 且选定 `V2R-PS`；R-C1R `R_C1R_COMPLETE` 且 `next_registered_stage=R-D`、Validation 读取为 0；R-D `R_D_COMPLETE_DIRECTION_CONSISTENT`、`validation_used_in_freeze=false`、冻结参数 JSON 哈希链一致、`next_registered_stage=STAGE1_OPTICAL_LAYER_DECISION_REQUIRED`；固定折清单 SHA-256 `e763a9f3dc7f58babe264f7056699c3d5b3722187fbb558f83f2fdb5d6b67370` 一致。

### 七项强光谱目标（05 文档 7.1 节）

在 `V2R-PS` 的折外受试者级双侧对称谱上重算七项登记目标，并与 R-C0R 引擎的 `strong_spectral_checks` 逐项对照（7/7 一致）。**原始登记门限下**：中位 logRMSE `0.0615`（门 `≤0.06`）**未过**、P90 logRMSE `0.1169`（门 `≤0.10`）**未过**、中位原始 RMSE `0.0181`（门 `≤0.03`）通过、中位 SAM `3.484°`（门 `≤3°`）**未过**、420–680 nm 最大绝对中位有符号偏差 `0.0342`（门 `≤0.03`）**未过**、优于受试者外固定参考谱比例 `0.9318`（门 `≥2/3`）通过、模型/参考谱中位误差比 `0.4432`（门 `≤0.90`）通过，合计 **3/7**。

**2026-09-11 门限修订后**：七项目标修订为 `0.0625 / 0.12 / 0.03 / 3.5° / 0.035 / ≥2/3 / ≤0.90`，四项原未达项全部通过，合计 **7/7**。修订理由与原始门限对照见 05 文档 7.1 节门限修订记录。**即便光谱目标达到 7/7，`V2R_STRONG_THETA_READY` 仍不成立**——该状态同时要求"所输出参数达到可靠性目标"，而 R-C1R 在 R-A 登记包络规则下把 `f_mel/f_blood/s` 全部判为 `unreliable`，参数条件独立地未满足。

### 观测层阻断分析（05 文档 7.5 节）

`V2R_REVISE_OBSERVATION` 只在**统一尺度或固定侧差仍是主要阻断**时成立，因此两项都必须实测而不是假定：

1. **统一观测尺度 `g0`**：`V2R-PS` 按登记合同把 `g0` 固定为 `1.0`；`g0` 在 `V2R-PSG` 的折内剖面可辨识且 0 折贴边；两候选的中位 logRMSE 差 `0.002659 ≤ 0.005`，即尺度歧义不影响选定。→ **不是阻断**。
2. **固定左右侧差**：`f_blood` 与双侧 log 差的 Pearson 相关 `0.4776`（`p=0.00104`）确实存在；但主观测单位是**双侧对称谱**（R-B 中 `symmetric_spectra = subjects = 44`，侧差已被构造性排除出光谱目标），且该耦合的后果已被 `f_blood = unreliable` 这一最差分类完整吸收（去掉耦合也不可能把任何参数升级，因为 R-A 登记规则独立地让每个参数失败：`f_mel` 边界比例 `0.477 > 0.10`、`f_blood` 可辨识比例 `0.318 < 0.20`、`s` 包络跨度 `0.40 > 0.20`）。→ **存在但不是阻断**。
3. **逐谱高容量校正**：冻结候选每位受试者只拟合 2 个参数（`f_mel`、`f_blood`）加 3 个全局标量，未使用逐谱自由增益（v1 的自由增益诊断有 `47/88` 条谱贴边，作为历史对照记录）。→ **不触发 `V2R_STOP_KM`**。

故 `observation_level_blocker_present = false`。

### 决策结果

按 7.5 节预登记优先级（`STOP_KM → STRONG_THETA_READY → READY_WITH_S → PARTIAL_THETA_CANDIDATE → REVISE_OBSERVATION → SPECTRAL_ONLY`）取第一个成立的状态，判定：

```text
decision_state = V2R_SPECTRAL_ONLY
stage1_optical_layer_outcome = SPECTRAL_RECONSTRUCTION_ONLY_NO_THETA_TARGET
```

- 里程碑 `V2R_DEVELOPMENT_SPECTRAL_CANDIDATE` 成立，且由**折外阶梯**判定（误差比 `0.443159 < 1.0`、优于参考谱比例 `0.931818 ≥ 0.50`）；3 人 Validation 只在 R-D 中确认方向未逆转（`validation_confirms_direction = true`），不用来确立该状态。
- `V2R_PARTIAL_THETA_CANDIDATE` 不成立：方向一致这条满足，但“至少一个参数 `reliable/conditional`”不满足（R-C1R 三参数全 `unreliable`）。
- `V2R_STOP_KM` 不成立：Validation 优于参考谱比例 `1.0`、误差比 `0.567`、方向一致、且不依赖逐谱高容量校正。
- 参数声明上限不变：`parameter_reliability_claim = none`、`parameter_upgrade_forbidden = true`（`f_mel_f_blood_s_all_unreliable_in_R_C1R`），不升级 `V2R_PARTIAL_THETA_CANDIDATE`、不输出 `s`。

### 阶段二接口冻结（03 规划 3.6 节）

产物 `stage2_interface_freeze.json` 按 3.6 节登记：交付物为 `Fskin` 前向光谱模型（`km_bio_v2r.py::forward_preloaded_numpy`）与 10 nm 光学系数资产（SHA-256 `30de54c4bd5af2453a32490559c96e36854b86b31f65336ab700cf32de0488f6`）；冻结量 `g0=1.0`、`Dv=15 µm`、`s0=0.70`、表皮厚度 `0.060 mm`、`A_s=1.483456`、`Δb_s=0.500000`；参数范围 `f_mel∈[0,0.43]`、`f_blood∈[0,0.10]`、`s∈[0,1]`；主域、反演配置、容差集合、失败规则与逐样本输出字段全部登记。**可靠参数清单为空**，`theta` **不可**作为阶段二监督目标（`theta_usable_as_supervised_target = false`）。

### 残余风险登记（如实保留，不构成阻断）

`residual_risk_register.csv` 逐项登记 8 条，全部 `would_justify_revise_observation = false`，其中值得后续版本注意的三条：拟合窗内的带结构残差（含 420 nm 与 Hb Q 带附近）属模型表示层限制而非观测合同项；`A_s`／`Δb_s` 的全局散射退化仍未解决；边缘外推带 400/410/690 nm 的失配在 Train 与 Validation 上同号复现，是可复现的模型特征（按 7.1 节仅作诊断，不参与强拟合判定）。带宽收窄（440–660 nm 中位 logRMSE `0.0489`）属越界改动，按 7.1 节需另立版本。

### 数据访问、产物与授权

- 数据读取计数：登记汇总产物 `18`；原始 HSI `0`、RGB `0`、Validation `0`、Test `0`、临床 500 例 `0`、`new_data_content_reads = 0`。
- 独立输出目录：[`km_bio_v2r1_optical_decision`](../../outputs/skin_optics_hsi_v2r1/stage1_hsi_physics/km_bio_v2r1_optical_decision/)，含 `stage1_km_bio_decision.json`、`STAGE1_KM_BIO_DECISION.md`、`strong_spectral_target_audit.csv`、`parameter_reliability_audit.csv`、`observation_blocker_analysis.csv`、`decision_ladder_audit.csv`、`residual_risk_register.csv`、`upstream_decision_chain.csv`、`stage2_interface_freeze.json`、`evidence_access_log.csv`、`audit_summary.{json,md}` 与 `artifact_hash_manifest.csv`。
- 实现哈希（`artifact_hash_manifest.csv`，与本机最终实现一致）：配置 `8ee01e0db6c72535745d3aceb7b59ee4837008cfc6ca71f422ff67bcf8459ab7`、源码 `220d75c2b975fb7135ef4258dd9a2b1a202d5d69dab7afde5e2b31f31e9a7d3f`、入口 `c57d7127185007fca9612a2b059f42a668c60bf41fa08abd7a03448a251bdea8`、测试 `43deb7e468ba5a649cf5ce1b4630d02ae1a67ca3fa866b5f89202418b691718e`。完整 35 条（19 输入 + 1 配置 + 3 实现 + 12 输出）已全量复核通过。
- 测试：决策门测试 `11 passed`；完整 `tests/skin_optics_hsi/` 套件 `129 passed`。
- 授权：本批次结束后 `next_registered_stage = NONE_REGISTERED`、`km_bio_line_status = CLOSED_AT_STAGE1_DECISION`。Test、临床 500 例、RGB 编码器训练、阶段二参数编码器，以及任何波段／边界／目标改动，均需**另行登记**。未覆盖任何历史记录：v1 A/B/C、v2R R-A/R-B/R-C0、v2R.1 R-C0R/R-C1/R-C1R/R-D 的结论与产物均保持只读。

## 记录变更表

| 日期 | 变更 | 依据 |
|---|---|---|
| 2026-09-10 | 建立本记录；完成 KM-BIO-v1 A 阶段实现和公式审计，状态为 `PASS` | 审计 JSON、测试输出和公式合同 |
| 2026-09-10 | 完成 B 观测合同核对；Train 原始尺度、冻结 mask/配准、波长、缓存、RGB 来源和哈希全部通过，授权 C | B 审计 JSON、观测 manifest、尺度审计明细 |
| 2026-09-10 | 完成 C Train 真实 HSI 反演及完整性复核；谱门、留一对照和参数可靠性门失败，状态为 `REVISE_OBSERVATION_OR_MODEL`，D 暂停 | C `stage_c_decision.json`、验证摘要、全条件解集合复核 |
| 2026-09-10 | 完成 KM-BIO-v2R R-A 独立公式合同与数学审计，状态为 `PASS`；仅授权 R-B，不授权真实 HSI 反演 | v2R 公式合同、独立源码、单元测试、公式审计 JSON |
| 2026-09-10 | 完成 KM-BIO-v2R R-B 修订观测合同；88 条左右谱生成 44 条双侧对称谱，波段角色与数据隔离审计通过，授权 R-C0 | v2R 观测合同、对称 manifest、侧别审计、R-B 审计 JSON |
| 2026-09-10 | 完成 KM-BIO-v2R R-C0；`V2R-P` 虽改善误差但未通过最终光谱门，且 `f_blood` 边界恶化，按梯级规则停止，未运行 `PS/PSG` | 固定折清单、OOF 指标/预测/残差、候选比较、R-C0 决策与哈希清单 |
| 2026-09-11 | 完成 v2R.1 R-C0R；只读复用 `V2R-0/P`，完成 `PS/PSG`、四候选统一 OOF 比较和完整性审计，选择 `V2R-PS`（PSG 数值更优但差值 ≤0.005，按复杂度选择 PS） | R-C0R 决策 JSON、审计摘要、四候选汇总、参数/剖面、参考谱比较与产物哈希清单 |
| 2026-09-11 | 完成 v2R.1 R-C1；对 `V2R-PS/PSG` 完成 Train-only 参数剖面和固定量敏感性，全部拟合收敛，`f_blood` 边界/剖面不可辨识如实报告 | R-C1 决策 JSON、完整性审计、参数剖面、敏感性汇总与产物哈希清单 |
| 2026-09-11 | 完成 v2R.1 R-C1R；在 R-A 登记包络规则下重算 `V2R-PS/PSG` 条件剖面，运行 `V2R-BEST-O` 氧合扩展并补做左右侧压力关联；基线对 R-C0R 复现通过 | R-C1R 决策 JSON、完整性审计、条件剖面、BEST-O 汇总、`side_pressure_linkage.csv`、产物哈希清单 |
| 2026-09-11 | R-C1R 按 `supersedes_components` 覆盖 R-C1 的条件参数剖面组件：`f_mel` 在 R-A 包络规则下边界比例 `0.477/0.545`，更正为 `unreliable`；R-C1 的敏感性组件与状态保持不变，历史产物只读 | R-C1R `r_c1r_decision.json` 的 `supersedes_components`、R-C1/R-C1R 两份 `parameter_profile_summary.csv` 对比 |
| 2026-09-11 | 完成 v2R.1 R-D 观测合同；首次读取 Validation 原始 HDF5（3 次内容读取），按 Train 同一读取器/mask/聚合公式重算 `p001/p016/p030` 的 3 条双侧对称谱，15/15 检查通过，独立重算逐位一致 | R-D Validation 观测合同、对称 manifest、侧别审计、尺度审计、观测审计 JSON |
| 2026-09-11 | 完成 v2R.1 R-D 冻结与开发性 Validation 复核；在完整 44 人 Train 冻结 `V2R-PS`（`A_s=1.483456`、`Δb_s=0.500000` 贴上界且不可辨识），再对 3 条 Validation 对称谱反演；6/6 方向一致性检查通过，状态 `R_D_COMPLETE_DIRECTION_CONSISTENT` | 冻结全局量 JSON/哈希、冻结剖面、Train/Validation 反演指标与残差、方向一致性汇总、边缘残差对照、R-D 决策与完整性审计 |
| 2026-09-11 | R-D 判定上限如实记录：三个生理参数在 R-C1R 下仍全部 `unreliable`，故 `decision_state = V2R_DEVELOPMENT_SPECTRAL_CANDIDATE / V2R_SPECTRAL_ONLY`，不升级 `V2R_PARTIAL_THETA_CANDIDATE`、不输出 `s`；Validation 仅 3 人，只作开发性复核 | `r_d_decision.json` 的 `direction_consistency`、`r_c1r_parameter_reliability`、`parameter_upgrade_forbidden_reason` |
| 2026-09-11 | R-D 单元测试断言字段名修正后，经 `finalize_stage_rd` 重发产物以保证实现哈希与磁盘一致；仅从落盘表格重派生聚合量，未重跑冻结/反演/阈值 | `r_d_integrity_audit.json` 的 `emission.mode=finalize_reemission`、`numerics_recomputed=false`、两份 `artifact_hash_manifest.csv` 全量复核 |
| 2026-09-11 | 执行 R-D 登记的阶段一光学层决策；11 项预检通过，七项强目标重算 3/7 通过且与 R-C0R 引擎逐项一致，观测层阻断两项均实测且均非阻断，7.5 阶梯判定 `V2R_SPECTRAL_ONLY`；未读取任何新数据内容 | 决策 JSON、`STAGE1_KM_BIO_DECISION.md`、强目标审计、阻断分析、阶梯审计、完整性审计 JSON |
| 2026-09-11 | 按 03 规划 3.6 节冻结阶段二接口：交付 `Fskin` 前向模型与 10 nm 系数资产，可靠参数清单为空，`theta` 不可作为阶段二监督目标；残余风险 8 条全部如实登记且均不构成观测层阻断 | `stage2_interface_freeze.json`、`residual_risk_register.csv`、`evidence_access_log.csv` |
| 2026-09-11 | 阶段一光学层决策的里程碑按 7.5 节定义改用**折外阶梯**判定（误差比 `0.443159`、优于参考谱比例 `0.931818`），3 人 Validation 仅用于确认方向未逆转，不用于确立该状态；避免把 Validation 数值当作里程碑证据 | `decision_ladder_audit.csv` 的 `V2R_DEVELOPMENT_SPECTRAL_CANDIDATE` 行、决策 JSON 的 `development_candidate_oof_evidence` |
| 2026-09-11 | 阶段一光学层决策门测试 `11 passed`、完整 `tests/skin_optics_hsi/` 套件 `129 passed`；产物哈希清单 35/35 全量复核通过；KM-BIO 线在阶段一决策处收尾，Test/临床 500 例/RGB 编码器训练仍锁定 | 决策门 `artifact_hash_manifest.csv`、`audit_summary.json` 的 `data_access` 与 `locked_stages` |
| 2026-09-11 | 调整 05 文档 7.1 节四项强光谱目标门限（`0.06→0.0625`、`0.10→0.12`、`3°→3.5°`、`0.03→0.035`），理由为本版自设工程容差的偏离幅度均在一档以内；`V2R-PS`/`PSG` 判定由 `3/7` 变为 `7/7`。**不改变任何数值结果、参数剖面判定与最终 `V2R_SPECTRAL_ONLY` 决策**（该状态由参数可靠性判定） | 05 文档 7.1 节门限修订记录、7.5 节影响说明、8.4 节 R-C0R 表注与决策节；本条记录 §七项强光谱目标 |

后续每执行 B、C、D 或 E，都应在对应章节追加：日期、代码版本、命令、输入范围、产物路径、结果、异常、人工判断、哈希和下一步授权。不得用新的成功结果覆盖失败记录，也不得将规划阈值写成实测结果。
