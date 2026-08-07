# `data/processed` P 系列目录数据地图（供研究分析使用）

## 1. 文档目的与阅读方式

本文档梳理 `E:\projects\face2\data\processed` 下全部以 `P` 开头的目录，说明每个目录的实际资产、规模、主键、适用研究问题及解释限制。

这些目录不是六份相互独立的病例数据，而是一条从 **P0 图像审计** → **P0B DECA 可行性/回归验证** → **P1 全量 DECA 资产冻结** → **P1 资产审计与统计** 的数据链。

研究分析的主病例主键为 `case_id`/`ID`；两者在相应 manifest 中关联。应优先通过 CSV manifest 选择病例，再读取相应路径的图像或 NPZ，而非依赖目录枚举顺序。

## 2. 总览：哪些是主数据，哪些是验证证据

| 目录 | 阶段/角色 | 规模 | 是否是后续研究的主资产 | 是否含 500×6 重光照 |
| --- | --- | ---: | --- | --- |
| `P0_Physics_Audit_v1` | 图像、mask、ROI、对齐与元数据基础层 | 500 例 | 是：P0 图像与掩膜主来源 | 否 |
| `P0B_DECA_Pilot12_v1` | 12 例 DECA pilot、输入模式比较、匿名 QC | 12 例 × 2 输入模式 | 否：方法可行性证据 | 否；仅 12 例 pilot |
| `P0B_cross_process_regression_baseline_v2` | 跨进程回归基线与阈值校准 | 12 例 reference + 12 例 holdout | 否：回归/复现证据 | 否：仅 12 例验证 |
| `P0B_cross_process_production_gate_v2_1` | 启动 Frozen500 前的生产放行合同 | 7 个证据文件 | 否：放行证据 | 否 |
| `P1_DECA_Frozen500_v1` | 正式冻结的全量 DECA 输出 | 500 例 | **是：DECA、物理图、重光照主来源** | **是，500×6=3,000** |
| `P1_Component_Audit_v1` | 对 Frozen500 的 join、schema、统计、QC、获取条件审计 | 500 行 manifest/统计 | 是：分析前筛选与数据理解层 | 不重复存图；索引到 Frozen500 |

一句话：**图像与 ROI 从 P0 取；DECA/重光照从 P1_DECA_Frozen500 取；做研究前的数据筛选、统计与 QC 从 P1_Component_Audit 取；P0B 主要用于证明处理路线而非扩充样本量。**

---

## 3. `P0_Physics_Audit_v1`：P0 图像与物理 ROI 基础层

### 3.1 作用与规模

该目录是固定 500 例的基础图像审计资产。`metadata/master_index.csv` 与 `metadata/build_status.csv` 各有 500 行、166 列。它保存 P0 对齐图、解析和 mask；**不含 DECA 输出，也不含固定光照重光照**。

### 3.2 关键目录与资产

| 相对目录 | 规模 | 内容/用途 |
| --- | ---: | --- |
| `images/raw_scene/` | 500 | 原始完整场景照片的规范化引用/副本 |
| `images/aligned_scene_224/` | 500 | 未替换背景的 224×224 对齐 RGB；P1 DECA 的正式输入来源 |
| `images/aligned_blackbg_224/` | 500 | 传统黑背景人脸结果；用于历史回归与 QC，不是重光照 |
| `images/e0b_meanbg_224/` | 500 | E0B mean-background 图；不是重光照 |
| `masks/final_face_mask_224/` | 500 | final face mask |
| `masks/face_valid_224/` | 500 | face-valid mask；几何/重建统计常与 DECA alpha 相交 |
| `masks/skin_strict_224/` | 500 | strict skin mask |
| `masks/physics_core_skin_224/` | 500 | 核心物理皮肤区域：左右面颊 effective 与额部 effective 的并集；不含 chin、lip、eye |
| `masks/source_valid_224/` | 500 | 源图有效区域 |
| `masks/roi_effective_224/<ROI>/` | 7×500 | effective ROI：`left_cheek`、`right_cheek`、`forehead`、`lip`、`chin`、`eye`、`combined_cheek` |
| `masks/roi_geometry_224/<ROI>/` | 7×500 | 对应 ROI 的几何定义 mask |
| `parsing/parsing_label_224/` | 500 | 人脸解析标签 |
| `parsing/selected_semantic_mask_224/` | 500 | 选定语义 mask |
| `parsing/semantic_regularized_mask_224/` | 500 | 规则化语义 mask |
| `transforms/alignment/` | 500 NPZ | 2×3 对齐矩阵、5 点/468 点 landmarks、原图尺寸和 bbox |
| `splits_500/` | 1 CSV | 固定 5-fold group/sex-stratified split |
| `qc_preview/` | 102 | 审计和边界不确定性 QC 图 |

### 3.3 关键 metadata 字段类别

`metadata/master_index.csv` 中包括：

- 主键与标签：`ID`、`patient_group_id`、`NYHA`、`label_3class`、`binary_label`、`fold`、`SEX`；
- 相机/EXIF：`camera_make`、`camera_model`、`brightness_value`、`iso`、`exposure_time_s`、`f_number`、`datetime_original`；
- 各资产相对路径：aligned scene、blackbg、meanbg、parsing、mask、ROI、alignment；
- ROI 像素数、比例、可用性及 warning；
- P0 状态：`core_asset_status`、`legacy_regression_status`、`boundary_ambiguity_status`、`p0_usable`、`overall_status`；
- 历史 blackbg 回归指标：MAE、RMSE、SSIM、p99/max difference、large-difference fraction 等。

### 3.4 研究使用与限制

- 适合：图像预处理复现、ROI 统计、mask 约束、EXIF/相机混杂分析、固定 split 的训练/验证设计。
- 不适合：固定 SH 重光照、DECA latent 或 3D normal 分析；这些资产不在 P0。
- 两例下颌—颈部边界不确定性保留于状态字段中。做核心物理统计时，应优先使用 `physics_core_skin`，而不是 chin。

---

## 4. `P0B_DECA_Pilot12_v1`：DECA 12 例可行性 pilot

### 4.1 作用

该目录用于回答“DECA 是否能在本项目对齐图上工作、哪种输入模式更稳定、固定重光照是否没有明显表示坍塌”。它不是正式全量输出，也不应作为样本量为 500 的分析数据源。

### 4.2 实际资产

| 相对目录 | 内容 |
| --- | --- |
| `environment/` | B0 环境审计、资产清单、SH 合成球验证、核心 smoke 与接口审计记录 |
| `pilot_manifest/` | 12 例匿名 audit ID 清单及单独 ID 映射 |
| `input_mode_comparison/` | 12 例 × 两种输入模式（`direct_p0_aligned`、`mask_bbox_crop`）的输出与指标；共 24 次处理 |
| `input_mode_comparison/<mode>/P0B-xxx/physical_maps_float.npz` | pilot 的 albedo-like、normal、重建、residual 等 float 资产 |
| `input_mode_comparison/<mode>/P0B-xxx/relighting/` | 每个 pilot/模式的 6 张显示型重光照 PNG |
| `noncollapse_audit_v1/` | 跨输入模式的表示、检索、PCA、重光照一致性审计 |
| `blind_review_v1/` | 盲法 review panels、review form；unblinding key 与 reviewer package 分开 |
| `qc/` | 匿名 QC 相关文件 |

### 4.3 已知结论与边界

`noncollapse_audit_v1/noncollapse_decision.json` 记录工程决策为 `GO_FULL_AUX`，冻结输入模式为 `direct_p0_aligned`。这只表示工程可行性、结构/外观/重光照未出现规定的坍塌；**不表示临床有效性、分类性能或真实生理反射率成立**。

---

## 5. `P0B_cross_process_regression_baseline_v2`：跨进程回归基线

### 5.1 作用

该目录用于检查同一 DECA 流程在不同进程/运行之间是否复现，而不是正式病例资产库。

### 5.2 实际资产

| 相对目录 | 内容 |
| --- | --- |
| `reference/pilot12/` | reference run（run_02）的 12 例 latent、maps、重光照 NPZ、诊断与 metadata |
| `holdout_validation/run_04/` | 独立 holdout run 的 12 例等价输出 |
| `comparisons/` | calibration pair、run_04 vs reference 指标和 warning cases |
| `metadata/` | thresholds、contract、environment fingerprint、baseline manifest |
| `calibration_runs/` | 当前为空或仅作为结构预留 |

每个 `relighting.npz` 都含 6 套预设，但这里是 reference/holdout 的 12 例回归证据，不能与 Frozen500 的正式病例资产混合计算样本量。

---

## 6. `P0B_cross_process_production_gate_v2_1`：Frozen500 生产放行门禁

### 6.1 作用

这是一个仅含 7 个文件的“合同/门禁”目录，证明正式全量 P1 冻结任务使用了固定阈值、资产哈希、环境指纹和回归规则。

### 6.2 实际资产

| 文件/目录 | 内容 |
| --- | --- |
| `metadata/contract.json` | 放行规则、阈值来源、核心比对类别（latent、albedo-like、normal、relighting） |
| `metadata/environment_fingerprint.json` | 运行环境、模型资产哈希、输入模式、确定性配置 |
| `metadata/run04_validation.json` | 对 run_04 的门禁验证 |
| `metadata/FROZEN.json` | 冻结时间及相关证据哈希 |
| `metadata/checksums.json` | 文件校验和 |
| `comparisons/formal_pilot12_metrics.csv` | pilot12 形式化指标 |

它不含病例图、mask、latent 或重光照数组。`contract.json` 中的 `full_500_started:false` 指的是该门禁验证本身尚未启动全量；全量结果在后续 `P1_DECA_Frozen500_v1`。

---

## 7. `P1_DECA_Frozen500_v1`：正式 500 例 DECA/重光照资产层

### 7.1 作用与规模

这是研究中读取 DECA 派生量与固定重光照的正式资产目录。`metadata/run_manifest.json` 记录：500 例预期、500 例成功、0 失败；其中 488 例为本轮新生成，12 例由有效断点恢复。

`cases/` 下有 500 个正式病例目录，每例均有 `relighting.npz`，每个文件含六个固定预设，共 **3,000 个病例—光照结果**。

### 7.2 每病例资产：`cases/<case_id>/`

| 文件 | 字段/规模 | 内容 |
| --- | --- | --- |
| `latents.npz` | shape `(1,100)`、expression `(1,50)`、pose `(1,6)`、camera `(1,3)`、light `(1,9,3)`、texture `(1,50)`、detail `(1,128)` | DECA latent codes，float32 |
| `maps.npz` | 8 个 P0 224×224 float32 数组 | `input_aligned_rgb`、`reconstruction`、`alpha`、`albedo_like`、`normal_coarse`、`shading_like`、`signed_residual`、`absolute_residual` |
| `relighting.npz` | `preset_names (6,)`、`sh_coefficients (6,9,3)`、`relighted_images (6,224,224,3)` | 六套固定 SH 重光照 float32 结果 |
| `quality.json` | 数值质量摘要 | 重建误差、alpha 覆盖、残差、有限性、每套重光照的 min/max/均值、黑像素及饱和比例 |
| `provenance.json` | 溯源信息 | P0 输入与各 mask 相对路径/哈希、字段映射、输入模式、P0 边界状态 |
| `_SUCCESS.json` | 成功标记 | 有效配置、checkpoint、输出文件 SHA256、完成时间、验证状态 |
| `preview.png` | 显示型快速 QC | 不应用于数值分析 |

六个重光照预设顺序由每个文件的 `preset_names` 决定，当前为：`neutral_front`、`left`、`right`、`top`、`dim_front`、`bright_front`。读取时应根据 `preset_names` 查找下标，不要只依赖固定数组位置。

### 7.3 全局资产

| 相对目录 | 内容 |
| --- | --- |
| `manifests/p1_deca_input_manifest.csv` | 500 行 P0 输入、哈希、边界状态、固定输入模式 |
| `manifests/p1_deca_output_manifest.csv` | 500 行 encode/decode/render/texture/relighting 成功状态及主要指标 |
| `manifests/p1_deca_quality_manifest.csv` | 500 行有限性、alpha 覆盖、重建误差、重光照完整性 |
| `manifests/p1_deca_failure_manifest.csv` | 当前为空；正式运行未登记失败病例 |
| `metadata/` | effective config、运行 manifest、环境、模型资产哈希、输出 inventory、pilot 回归记录 |
| `qc/` | 随机病例、最大重建误差、最低 alpha 覆盖、最大饱和和两例边界病例 QC |
| `reproducibility_audit_v1/` | 12 例多运行可重复性比较 |
| `attempts/` | 历史/中间尝试；不应替代正式 `cases/` 做主分析 |

### 7.4 使用限制

- `albedo_like` 不是真实反射率、真实皮肤颜色、血红蛋白、黑色素、灌注或血氧。
- `shading_like` 不是实际环境光的测量值。
- residual 是未建模残差，不是 specular map。
- 研究中的核心皮肤统计范围建议使用 P0 的 `physics_core_skin` 与 P1 `alpha` 的交集；不要将 chin 作为核心物理区域。

---

## 8. `P1_Component_Audit_v1`：Frozen500 的分析前审计与索引层

### 8.1 作用

此目录不复制 500 例的 NPZ 图像资产，而是把 P0、固定 split、P1 Frozen500、EXIF 和质量信息进行 500 行 join，并输出 schema、统计、QC 和使用就绪结论。它是将研究问题转化为可筛选分析队列的首选入口。

`metadata/audit_manifest.json` 记录：500 行、500 个唯一病例、P0/split/frozen asset 均无未匹配记录、无必需资产缺失、无输入哈希不匹配、无 label/fold mismatch。

### 8.2 关键资产

| 相对目录 | 内容 |
| --- | --- |
| `manifests/p1_master_manifest.csv` | 500 行总索引：病例/组/fold/标签、P0 路径、P1 Frozen 路径、EXIF、资产可用性、哈希、质量状态 |
| `manifests/p1_representation_availability.csv` | 500 行各表示是否存在：RGB、albedo_like、normal、light、shading、residual、relighting、specular_like |
| `manifests/p1_asset_join_audit.csv` | P0、split、Frozen500 的 join 检查结果 |
| `acquisition/` | 相机、EXIF、fold、标签的交叉表与缺失率；用于获取条件混杂审计 |
| `schema/` | component、latent、metadata、split 的字段和形状定义 |
| `statistics/image_component_case_stats.csv` | 42,000 行：500 例 × 7 图像 component × 4 区域 × 3 通道的统计 |
| `statistics/relighting_stats.csv` | 3,000 行：500 例 × 6 预设的均值、标准差、黑像素/饱和比例等 |
| `statistics/latent_case_stats.csv` | 3,500 行：7 类 latent 的病例级统计 |
| `statistics/latent_dimension_stats.csv` | 364 行：latent 维度级统计与近零方差标记 |
| `qc/p1_case_qc_flags.csv` | 500 例 QC flags；当前存在 16 个 flagged case，但均保留，未删除病例 |
| `metadata/readiness_decision.json` | 当前结论为 `READY_FOR_P1_COMPONENT_EXPERIMENTS_WITH_ACQUISITION_WARNINGS` |

### 8.3 重要分析提醒

- `statistics/` 是审计描述统计，明确不是训练标准化参数。
- 有 16 例存在 QC flags；应根据研究问题做敏感性分析或报告，而不是静默删除。
- acquisition 维度为 `PARTIALLY_READY`：相机与核心 EXIF 完整率约 99.6%，但相机—标签重叠有限。使用相机/EXIF 作推断变量时应警惕混杂。
- `has_specular_like` 的可用性应视为当前前端限制；不要把不存在的 specular-like 表示当作缺失值随意插补。

---

## 9. 数据血缘与推荐读取顺序

```text
P0_Physics_Audit_v1
  ├─ aligned_scene、mask、ROI、EXIF、状态、固定 split
  │
  ├─ P0B_DECA_Pilot12_v1 ── 方法可行性与输入模式决策
  ├─ P0B_cross_process_* ── 跨进程回归与生产放行证据
  │
  └─ P1_DECA_Frozen500_v1
       ├─ 500 例 DECA latent/maps/quality/provenance
       └─ 500×6 固定 SH 重光照
             │
             └─ P1_Component_Audit_v1
                  └─ 500 行 join、schema、统计、QC、获取条件审计
```

### 按研究目的选择数据

| 研究任务 | 首选文件/目录 | 必要补充 |
| --- | --- | --- |
| 图像与 ROI 方法 | `P0/.../master_index.csv`、`images/`、`masks/` | 对齐 NPZ、P0 状态字段 |
| DECA 表示或 3D/normal 分析 | `P1_DECA_Frozen500_v1/cases/<ID>/latents.npz`、`maps.npz` | `P1_Component_Audit_v1/manifests/p1_master_manifest.csv` |
| 固定光照稳健性/重光照分析 | `P1_DECA_Frozen500_v1/cases/<ID>/relighting.npz` | `statistics/relighting_stats.csv`、P0 physics-core mask |
| 全量质量筛选 | `P1_Component_Audit_v1/manifests/p1_master_manifest.csv` | `qc/p1_case_qc_flags.csv`、P1 output/quality manifest |
| 相机/EXIF 混杂描述 | `P1_Component_Audit_v1/acquisition/` | P0 master 的原始 EXIF 字段 |
| 方法复现/版本差异 | `P0B_cross_process_regression_baseline_v2` 与 `P0B_cross_process_production_gate_v2_1` | P1 metadata 的环境/资产哈希 |

## 10. 向其他分析助手提供数据时的最小信息包

不要默认上传全部 500 张人脸图。通常先提供下列可去标识化的结构信息，再按具体研究问题提供必要的病例级派生数据：

1. 本文档；
2. `P1_Component_Audit_v1/manifests/p1_master_manifest.csv` 的字段说明或筛选后的列；
3. 研究终点、纳入/排除规则、是否使用 P0 标签、是否使用 EXIF/相机变量；
4. 明确采用的表示（RGB、albedo_like、normal、latent、重光照、ROI 统计）；
5. 固定 5-fold split、patient/group 不跨 fold 的要求；
6. 对 16 例 QC flag 和 2 例下颌—颈部边界不确定病例的处理方案；
7. 上述“科学解释限制”，避免把 DECA 派生量误作生理真值。

## 11. 不应混用的内容

- 不要把 `P0B` 的 12 例 pilot/reference 结果与 `P1` 的 500 例正式结果合并后当作 512 例独立样本。
- 不要把 `attempts/` 中的历史输出与 `P1_DECA_Frozen500_v1/cases/` 主输出混用。
- 不要把 `aligned_blackbg_224` 或 `e0b_meanbg_224` 误称为重光照结果。
- 不要把 DECA `albedo_like`、`shading_like`、residual 直接解释为真实皮肤生理参数。
- 不要在未说明的情况下删除 QC flagged 或 P0 boundary-ambiguity 病例；应由分析方案预先定义处理规则。

本文档按当前目录内容整理；重跑、迁移环境或更新 Frozen500 资产后，应重新核对各目录下 `metadata/`、manifest 行数及 SHA256 记录。
