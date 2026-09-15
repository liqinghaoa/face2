# 实验记录总览

本文件集中记录目前完成的 SO-R1、SO-R2 探索实验与阶段一真实 HSI 物理验证。原始数据、模型、日志和权威 acceptance/decision 仍位于项目原目录；本文只负责研究复盘和证据导航。

## 当前结论边界

- SO-R1-B1 与 SO-R1-B2 训练完成，权威 checkpoint 已冻结。
- SO-R1-C 合成 OOD 门控为 **FAIL**，M、H 通道均未达到正式通过条件。
- SO-R2-X0～X5 均为真实人脸内部探索或技术审计，不构成正式 SO-R2/SO-R3 授权。
- 当前保持：`SO-R1-C = FAIL`、`official_SO_R2_authorization = false`、`SO_R3_authorization = false`。

## EXP-001：SO-R1-A0 相机与光照选择

目标是排除会造成 M/H 系统性失真的相机/光照组合。Nokia N900 在 FL2/FL11 下出现可重复的相机色彩链问题和 B 通道塌缩，被永久排除。AM2/AM3 经 high-clip、替代相机和 pilot 审计后形成 AM4 全局相机图谱。

AM4 冻结 11 个安全相机；正式 seen 相机为 Canon 5DMarkII、Hasselblad H2、Nikon D80、Point Grey Grasshopper2 14S5C；unseen 相机为 Canon 1DMarkIII、Nikon D5100。最终状态为 `PASS / FROZEN`，允许进入 A2-D0-AM1，但不直接授权训练。

证据：[AM4 acceptance](../../data/processed/SO_R1_A0_AM4_GlobalCameraAtlas_v1/AM4_ACCEPTANCE.json)；[相机替换报告](../../reports/so_r1_a0_am1_camera_replacement/)；[AM4 报告](../../reports/so_r1_a0_am4_global_camera_atlas/)。

## EXP-002：SO-R1-A1/A2 配对数据构建

- A1 pilot：64 latent、320 acquisition、256 pair，覆盖 32 个 camera-light 组合；确定性复现通过，状态 `ACCEPTED`。
- P0 综合审计因 Nokia N900 FL2/FL11 问题失败并要求协议修订。
- 初版 G1 在 Olympus E-PL2 / FL2 上出现低值裁剪，128 次重试后仍失败，状态 `FAIL_QC_RETRY_EXHAUSTED`，不是最终训练权威数据。
- D0-AM1 冻结规模为 13,500 latent、67,500 acquisition、54,000 pair。
- G1-AM1 全量生成通过；workers 1 vs 8 为 128/128 exact，独立 replay 为 32/32，数据集 `ACCEPTED`，允许进入 B1。

证据：[A1](../../data/processed/SO_R1_A1_PairedPilot_v1/PILOT_ACCEPTANCE.json)；[初版 G1 失败](../../data/processed/SO_R1_A2_FormalPairedDataset_v1/G1_ACCEPTANCE.json)；[D0-AM1](../../data/processed/SO_R1_A2_D0_AM1_ProtocolFreeze_v1/D0_AM1_ACCEPTANCE.json)；[G1-AM1](../../data/processed/SO_R1_A2_G1_AM1_FullGeneration_v1/G1_AM1_ACCEPTANCE.json)。

## EXP-003：SO-R1-B1 基线模型

早期 v1～v5 和 DataLoader workers 预检属于调试/恢复链路，最终权威运行是 `so_r1_b1_baseline_v6`。状态 `PASS`，训练 `COMPLETE`，最佳 epoch 25。权威 checkpoint 为 `runs/so_r1_b1_baseline_v6/checkpoints/best_val_masked_smoothl1.pt`，SHA-256 为 `5b1dd85f...0265b`，训练合同哈希为 `7dcfb309...17b84f`。数据源保持不变，允许进入 B2。

证据：[B1 acceptance](../../runs/so_r1_b1_baseline_v6/B1_ACCEPTANCE.json)；[B1 报告](../../reports/so_r1_b1_baseline_v6/)。

## EXP-004：SO-R1-B2 Proposed 模型

比较 λ=0.05、0.10、0.20、0.50，训练经历 v1/v2 和 v3 恢复链路。最终状态 `PASS / COMPLETE`，λ 选择冻结为 0.50，选择 epoch 25。权威 checkpoint 为 `runs/so_r1_b2_proposed_v3/lambda_0p50/checkpoints/best_val_masked_smoothl1.pt`，SHA-256 为 `5cd86cee...6562a0`。Validation viability 与 non-inferiority 均通过，选择阶段 test/OOD access count 为 0。

证据：[B2 acceptance](../../runs/so_r1_b2_proposed_v3/B2_ACCEPTANCE.json)；[B2 报告](../../reports/so_r1_b2_proposed_v3/)。

## EXP-005：SO-R1-C 合成 ID/OOD 评估

冻结 B1/B2 checkpoint 在 ID Test、Camera-OOD、Light-OOD、Joint-OOD 上完成 10,000 pair 评估。Pipeline validity 通过，列出的 collapse 检查均通过，没有简单全零/全一崩溃。M 的 OOD 相对变化均值约 -0.00550，95% CI [-0.03169, 0.01955]；H 为 -0.14568，95% CI [-0.21101, -0.08389]。

M 与 H 的正式通道门控均为 `CHANNEL_NOT_PASS`。最终 `SO-R1-C = FAIL`、synthetic gate `FAIL`，不授权 SO-R3。

证据：[C acceptance](../../data/processed/SO_R1_C_SyntheticEvaluation_v1/C_ACCEPTANCE.json)；[C 报告](../../reports/so_r1_c_synthetic_evaluation/)。

## EXP-006：SO-R2-X0～X2 真实人脸冻结推理

- X0：500 例、10,000 patch，B1/B2 checkpoint 哈希匹配，12 例 canary 通过，状态 `PASS_INPUT_CONTRACT`。
- X1：B1 与 B2 λ=0.50 完成 500 例冻结推理，共处理 10,000 patch，生成 B1/B2 各 500 个 case-level 融合结果；未读取标签或启动分类。
- X2：B1/B2 各 500 张融合 M/H map，shape `2×1220×979`；technical output integrity `PASS`，seam audit 为描述性审计；没有模型前向、训练或 checkpoint 写入。

X0～X2 只证明输入、推理和输出技术链路可运行，不代表 SO-R1-C 通过。

证据：[X0](../../runs/so_r2x0_realface_input_audit/R2X0_ACCEPTANCE.json)；[X1](../../data/processed/SO_R2X1_RealFaceFrozenInference_v1/R2X1_ACCEPTANCE.json)；[X2](../../data/processed/SO_R2X2_RealFaceOutputAudit_v1/R2X2_ACCEPTANCE.json)。

## EXP-007：SO-R2-X3 二分类与消融

500 例 Control/Patient，nested-direct 五折内部 OOF。输入为 float32、`320×256`；五通道条件的前三个 RGB 通道与 RGB-only 逐元素一致。

| 条件 | Macro-AUC | Accuracy | Macro-F1 | Balanced Accuracy | Sensitivity | Specificity |
|---|---:|---:|---:|---:|---:|---:|
| RGB | 0.8370 | 0.778 | 0.7216 | 0.7552 | 0.7974 | 0.7130 |
| RGB+B1MH | 0.7986 | 0.760 | 0.6897 | 0.7100 | 0.8026 | 0.6174 |
| RGB+B2MH | 0.8400 | 0.782 | 0.7131 | 0.7304 | 0.8260 | 0.6348 |
| B1 MH-only | 0.8222 | 0.758 | 0.6594 | 0.6599 | 0.8416 | 0.4783 |
| B2 MH-only | 0.8055 | 0.754 | 0.6693 | 0.6787 | 0.8182 | 0.5391 |
| B2 M-only | 0.8166 | 0.766 | 0.6952 | 0.7139 | 0.8104 | 0.6174 |
| B2 H-only | 0.8381 | 0.780 | 0.7082 | 0.7230 | 0.8286 | 0.6174 |

RGB+B2MH 相对 RGB 的 Macro-AUC 为 +0.00305，但固定 0.5 阈值指标没有同步改善。只读诊断显示 B2 Patient 概率平均上移约 0.04235，向上跨阈值 51 例、向下 31 例，可解释 sensitivity 上升和 specificity 下降。结果仅作内部探索，不选择 winner 或调阈值。

证据：[X3 acceptance](../../data/processed/SO_R2X3_ExploratoryClassification_v1/R2X3_ACCEPTANCE.json)；[主要 OOF](../../reports/so_r2x3_exploratory_classification/oof_metrics.csv)；[MH-only](../../data/processed/SO_R2X3_MHOnlyAblation_v1/MH_ABLATION_ACCEPTANCE.json)；[只读诊断](../../data/processed/SO_R2X3_OOFDiagnostic_v1/X3D_ACCEPTANCE.json)。

## EXP-008：SO-R2-X4 融合结构探索

使用冻结 RGB+B2MH 五通道输入，比较 Late Concat、MH-LRF、MH-LRTM。Smoke v2 修复后纯 Late Concat 不含参与最终 logits 的 RGB head；参数量分别为 13,126,762、13,212,140、13,376,940。

| 模型 | Macro-AUC | Accuracy | Macro-F1 | Balanced Accuracy |
|---|---:|---:|---:|---:|
| LATE_CONCAT | 0.8204 | 0.770 | 0.6908 | 0.7012 |
| MH_LRF | 0.8043 | 0.754 | 0.6711 | 0.6817 |
| MH_LRTM | 0.8096 | 0.752 | 0.6869 | 0.7139 |

15 个折次和 1,500 条 OOF 预测完成。复杂融合结构未显示相对 RGB 或简单 Late Concat 的明确内部优势；不选择 winner 或启动重训。

证据：[Smoke v2](../../data/processed/SO_R2X4_MHLRTMSmokeTest_v2/S0_ACCEPTANCE.json)；[X4](../../data/processed/SO_R2X4_ExploratoryMHLRTMClassification_v1/X4_ACCEPTANCE.json)；[X4 OOF](../../reports/so_r2x4_exploratory_mh_lrtm/oof_metrics.csv)。

## EXP-009：SO-R2-X5 B1 M/H-only 三分类

输入仅为 B1 M/H 两通道，不读取 RGB。标签为 Normal=NYHA 0、Mild=NYHA 1–2、Severe=NYHA 3–4，分布 115/238/147。外层复用冻结五折，内层采用 `three_class × SEX` group-aware 划分。模型为 ImageNet-1K ResNet-18，RGB conv1 通道均值初始化两个输入通道，输出 3 logits。16 例 CUDA FP32 smoke 通过。

五折 pooled OOF：Macro OvR-AUC 0.6723，Accuracy 0.4520，Macro-F1 0.4517，Balanced Accuracy 0.4995。

| 真实\预测 | Normal | Mild | Severe |
|---|---:|---:|---:|
| Normal | 88 | 14 | 13 |
| Mild | 85 | 79 | 74 |
| Severe | 42 | 46 | 59 |

Normal 识别相对较好，Mild 与 Severe 分离较弱。结果不支持临床三分类结论，只说明 B1 M/H-only 在当前内部设计下分级能力有限。

证据：[Smoke](../../data/processed/SO_R2X5_B1MHThreeClassSmokeTest_v1/S0_ACCEPTANCE.json)；[X5](../../data/processed/SO_R2X5_B1MHThreeClassExploratory_v1/X5_ACCEPTANCE.json)；[五折指标](../../reports/so_r2x5_b1_mh_three_class_exploratory/fold_metrics.csv)；[OOF](../../reports/so_r2x5_b1_mh_three_class_exploratory/oof_metrics.csv)；[混淆矩阵](../../reports/so_r2x5_b1_mh_three_class_exploratory/oof_confusion_matrix.csv)。

## EXP-010：阶段一真实 HSI 物理验证

S1-0 已完成 306 对 Hyper-Skin RGB/VIS 的证据化合同、全量数值扫描和 SHA-256 清单，状态 `PASS`。S1-1 在 18 个分层 Train 样本上比较 8 种空间变换，全部选择 `transpose`；自动门与 QinghaoLi 人工复核均通过，坐标映射已冻结，允许进入 S1-2。阶段一剩余步骤尚未执行，31 波段中心数组确认和有效 SRF/带宽仍是正式 Test 前的证据缺口。

持续记录：[阶段一 HSI 物理验证实验记录](阶段一_HSI_物理验证实验记录.md)；权威产物：[S1-0 合同](../../data/processed/HyperSkin_Stage1_v1/contracts/data_contract.json)；[S1-1 decision](../../data/processed/HyperSkin_Stage1_v1/registration_qc/registration_decision.json)。

## 总体研究判断

1. 合成数据、模型训练和真实人脸推理的工程链路已形成闭环。
2. B2 在二分类 pooled AUC 上仅相对 RGB 小幅增加，固定阈值指标和复杂融合结构未显示一致优势。
3. B1 M/H-only 对 Control/Patient 有一定内部排序信息，但当前三分类显示疾病严重程度分级能力有限。
4. 所有真实人脸结果仍是单队列内部 OOF 描述，不得表述为外部验证、临床效能或正式 SO-R2/SO-R3 结论。

## 阅读规则

1. 各阶段 acceptance 是权威状态来源；本文只做概括与导航。
2. 如摘要与源文件不一致，以 acceptance、OOF CSV 和协议锁为准。
3. 失败记录是研究证据的一部分，不应被后续成功运行覆盖或改写。
