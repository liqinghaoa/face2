# 阶段一光学层决策（KM-BIO-v2R.1）

- 任务：`STAGE1_OPTICAL_LAYER_DECISION`；协议：`KM-BIO-v2R.1`；日期：2026-09-11。
- 决策状态：`V2R_SPECTRAL_ONLY`。
- 阶段一光学层结论：`SPECTRAL_RECONSTRUCTION_ONLY_NO_THETA_TARGET`。
- 冻结候选：`V2R-PS`（`A_s=1.483456`、`delta_bs=0.500000`、`g0=1.0`）。
- 参数声明：`none`；参数升级：`forbidden`（f_mel_f_blood_s_all_unreliable_in_R_C1R）。
- 数据边界：本批次**未读取**任何原始 HSI / RGB / Validation / Test / 临床 500 例内容；全部证据来自已登记的决策 JSON 与汇总表。

## 1. 上游证据链（只读）

| stage       | model_id     | artifact                 | sha256                                                           | status                            | next_registered_stage                  | role               |
|:------------|:-------------|:-------------------------|:-----------------------------------------------------------------|:----------------------------------|:---------------------------------------|:-------------------|
| v1_C        | KM-BIO-v1    | v1_stage_c_decision      | 25278dadd81fa1852de4be2bced709152c8c50f80fe77daf6fac5371700b813e | REVISE_OBSERVATION_OR_MODEL       |                                        | read_only_evidence |
| v2R_R-A     | KM-BIO-v2R   | v2r_formula_audit        | 441c38c94892f142ad3d532d5a712255d3d36fcf6c728d1d916d6229a92503f9 | PASS                              |                                        | read_only_evidence |
| v2R_R-B     | KM-BIO-v2R   | r_b_observation_audit    | 50d3940c76575f80d78aceeed7ae08ad5c1bbaea927f685122cbc1c52b3efe64 | PASS_FOR_V2R_TRAIN_INVERSION      | R-C0_CANDIDATE_LADDER                  | read_only_evidence |
| v2R_R-C0    | KM-BIO-v2R   | historical_r_c0_decision | 2698923432db32bb10ded5f90034dd11ae15f308deb81a0a09e199726db18e4f | R_C0_STOPPED_AT_FAILED_UPGRADE    |                                        | read_only_evidence |
| v2R.1_R-C0R | KM-BIO-v2R.1 | r_c0r_decision           | 9d5dbe21ac1208eabefa9ee21aed2568093ce805de15a88b1b392f2f35043c62 | R_C0R_COMPLETE                    | R-C1                                   | read_only_evidence |
| v2R.1_R-C1R | KM-BIO-v2R.1 | r_c1r_decision           | e2ac9fa14f0c1ede7dc6d161febd18d775d82ecd8becf0112eee39484be231e5 | R_C1R_COMPLETE                    | R-D                                    | read_only_evidence |
| v2R.1_R-D   | KM-BIO-v2R.1 | r_d_decision             | 5266e8801003885a3a0d645358ad77ec863a77ee7b527d2d579d5878dfc32fdc | R_D_COMPLETE_DIRECTION_CONSISTENT | STAGE1_OPTICAL_LAYER_DECISION_REQUIRED | read_only_evidence |

## 2. 七项强光谱目标（05 文档 7.1 节，OOF 受试者级）

| target_key                              |     value | operator   |   target | unit        | recomputed_pass   |
|:----------------------------------------|----------:|:-----------|---------:|:------------|:------------------|
| median_logrmse                          | 0.0614783 | le         | 0.06     | log_rmse    | False             |
| p90_logrmse                             | 0.116877  | le         | 0.1      | log_rmse    | False             |
| median_rmse                             | 0.0181094 | le         | 0.03     | reflectance | True              |
| median_sam_deg                          | 3.48441   | le         | 3        | degree      | False             |
| maximum_abs_median_signed_band_residual | 0.0341997 | le         | 0.03     | reflectance | False             |
| reference_better_fraction               | 0.931818  | ge         | 0.666667 | fraction    | True              |
| model_to_reference_median_error_ratio   | 0.443159  | le         | 0.9      | ratio       | True              |

结论：`all_targets_pass=False`；未通过的项为 `median_logrmse, p90_logrmse, median_sam_deg, maximum_abs_median_signed_band_residual`。

## 3. 参数可辨识性（v1 C 对照 + R-C1R 沿用）

| source_stage   | candidate          | parameter   |   identifiable_fraction |   boundary_fraction |   median_envelope_normalized_span | classification   |
|:---------------|:-------------------|:------------|------------------------:|--------------------:|----------------------------------:|:-----------------|
| v1_C           | KM-BIO-v1          | f_mel       |                0        |          nan        |                      nan          | unreliable       |
| v1_C           | KM-BIO-v1          | f_blood     |                0        |          nan        |                      nan          | unreliable       |
| v1_C           | KM-BIO-v1          | s           |                0        |          nan        |                      nan          | unreliable       |
| R-C1R          | V2R-PS             | f_mel       |                1        |            0.477273 |                        0.00443527 | unreliable       |
| R-C1R          | V2R-PS             | f_blood     |                0.318182 |            0.477273 |                        0.34       | unreliable       |
| R-C1R          | V2R-BEST-O(V2R-PS) | f_mel       |                1        |            0.659091 |                        0.00551121 | unreliable       |
| R-C1R          | V2R-BEST-O(V2R-PS) | f_blood     |                0.318182 |            0.659091 |                        0.36       | unreliable       |
| R-C1R          | V2R-BEST-O(V2R-PS) | s           |                0.25     |            0.659091 |                        0.4        | unreliable       |

## 4. 观测层阻断分析（05 文档 7.5 节）

| blocker_key                           | quantity                           |   measured_value |   threshold_value | condition_met   | is_blocking   | measured_detail                                                                                                                                              |
|:--------------------------------------|:-----------------------------------|-----------------:|------------------:|:----------------|:--------------|:-------------------------------------------------------------------------------------------------------------------------------------------------------------|
| unified_scale                         | g0                                 |       0.00265922 |             0.005 | False           | False         | g0_declared_fixed=True; g0_identifiable_in_V2R-PSG=True; selection_delta_logrmse=0.002659<= 0.005                                                            |
| fixed_side_difference                 | pearson_r_vs_side_log_difference   |       0.477609   |             0.4   | True            | False         | f_mel r=0.2536 p=0.09668; f_blood r=0.4776 p=0.00104; primary_parameter_usable=False; primary_unit_excludes_side_difference=True                             |
| per_spectrum_high_capacity_correction | fitted_per_subject_parameter_count |       2          |             4     | False           | False         | per_spectrum_free_gain_used=False; fitted_per_subject_parameters=2; v1_free_gain_boundary_fraction=0.5341 (05_Results_and_Decisions section 2.5, historical) |

`observation_level_blocker_present=False`；`per_spectrum_high_capacity_correction_required=False`。

## 5. 决策阶梯判定

| kind      |   priority | state                              | satisfied   | action_zh                                           | evidence                                                                                                                                                |
|:----------|-----------:|:-----------------------------------|:------------|:----------------------------------------------------|:--------------------------------------------------------------------------------------------------------------------------------------------------------|
| milestone |          0 | V2R_DEVELOPMENT_SPECTRAL_CANDIDATE | True        | 允许冻结后进入 Validation，只检验光谱泛化           | oof_model_to_reference_ratio=0.443159<1.0; oof_better_than_reference_fraction=0.9318>=0.50; validation_confirms_direction=True                          |
| terminal  |          1 | V2R_STOP_KM                        | False       | 停止双层 K-M 主路线                                 | validation_better_than_reference_fraction=1.0000; validation_model_to_reference_ratio=0.567239; direction_consistent=True; high_capacity_required=False |
| terminal  |          2 | V2R_STRONG_THETA_READY             | False       | 冻结后进入 Validation，输出模型条件下的区域有效参数 | all_strong_targets_pass=False; all_output_parameters_reliable=False                                                                                     |
| terminal  |          3 | V2R_READY_WITH_S                   | False       | s 可作为额外候选，仍不等同动脉 SpO2                 | partial_theta_met=False; s_reliable_or_conditional=False                                                                                                |
| terminal  |          4 | V2R_PARTIAL_THETA_CANDIDATE        | False       | 仅保留通过判定的参数，其他参数不进入阶段二          | direction_consistent=True; any_parameter_reliable_or_conditional=False                                                                                  |
| terminal  |          5 | V2R_REVISE_OBSERVATION             | False       | 单独修订观测模型，不让生理公式来吸收差异            | spectral_reconstruction_useful=True; observation_level_blocker_present=False                                                                            |
| terminal  |          6 | V2R_SPECTRAL_ONLY                  | True        | 只承认前向重建能力，不训练生理参数编码器            | spectral_reconstruction_useful=True; all_parameters_unreliable=True; observation_level_blocker_present=False                                            |

按预登记优先级取第一个成立的状态 → `V2R_SPECTRAL_ONLY`。

里程碑 `V2R_DEVELOPMENT_SPECTRAL_CANDIDATE` 由**折外阶梯**判定（模型/参考谱中位误差比 `0.443159` < 1.0，优于参考谱比例 `0.9318` ≥ 0.50）；3 人 Validation 只能在 R-D 中确认方向未逆转（`validation_confirms_direction=True`），不能用来确立该状态。

## 6. 残余风险登记

| risk_id   | category          | item                                           | status                          | would_justify_revise_observation   | justification                                                                                                                                                                                                                       |
|:----------|:------------------|:-----------------------------------------------|:--------------------------------|:-----------------------------------|:------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| R1        | observation_level | 统一观测尺度 g0                                | resolved_by_registered_revision | False                              | the unified-scale degeneracy registered in v1 section 2.4 was addressed by the v2R 4.4 scale term and is not the main blocker anymore                                                                                               |
| R2        | parameter_level   | 固定左右侧差与生理参数的耦合                   | unresolved_but_absorbed         | False                              | the coupling is a parameter-level confound whose consequence is already the declared unreliable verdict; the registered primary unit is the bilateral symmetric spectrum, so the side difference does not enter the spectral target |
| R3        | model_level       | 拟合窗内带结构残差（含 420 nm 与 Hb Q 带附近） | unresolved                      | False                              | the structured residual is a shape mismatch inside the 420-680 nm fit window, i.e. a model-representation limit, not an observation-contract term                                                                                   |
| R4        | edge_diagnostic   | 边缘外推带 400/410/690/700 nm                  | reproducible_model_feature      | False                              | section 7.1 marks the full 400-700 nm result as an explicitly labelled edge diagnostic that does not participate in the strong-fit judgement; the reproducible same-sign behaviour is recorded as a model characteristic            |
| R5        | model_level       | 全局散射退化 A_s / delta_b_s                   | unresolved                      | False                              | a global-scattering degeneracy is a model-level freedom, and section 7.3 already registers global profiles as diagnostics rather than execution gates                                                                               |
| R6        | parameter_level   | 三个生理参数全部不可辨识                       | unresolved                      | False                              | this is exactly the section 7.5 condition for SPECTRAL_ONLY: useful spectral reconstruction with no identifiable physiological parameter                                                                                            |
| R7        | scope             | 带宽收窄后的改善                               | out_of_scope_for_this_version   | False                              | section 7.1 forbids re-using the strong targets across bands; a narrower window is a new version and must be registered separately                                                                                                  |
| R8        | scope             | Validation 证据强度                            | bounded                         | False                              | R-D registers Validation as a developmental review only; no population estimate and no physiological claim may be derived from it                                                                                                   |

## 7. 阶段二接口冻结（03 规划 3.6 节）

- 交付物：`Fskin` 前向光谱模型（`src/skin_optics_hsi/km_bio_v2r.py::forward_preloaded_numpy`）。
- 系数资产 SHA-256：`30de54c4bd5af2453a32490559c96e36854b86b31f65336ab700cf32de0488f6`。
- 可靠参数清单：`[]`。
- `theta` 是否可作为阶段二监督目标：**否**（decision_state is V2R_SPECTRAL_ONLY: every physiological parameter is unreliable, so only the forward reconstruction capability is admitted and no physiological-parameter encoder may be trained）。
- 逐样本输出字段见 `stage2_interface_freeze.json` 的 `per_sample_output_schema`。

## 8. 授权边界

- 阶段一光学层路线结论：`KM-BIO 双层 K-M 生理线在阶段一收尾为**前向光谱重建模型**：其 420-680 nm 双侧对称谱重建优于固定参考谱且方向一致，但三个生理参数在 R-A 登记包络规则下全部不可辨识，因此不产出阶段二的生理参数目标。`。
- `next_registered_stage = NONE_REGISTERED`；任何 Test / 临床 500 例 / RGB 编码器训练 / 阶段二参数编码器均需**另行登记**。
- 锁定状态：`test_executed=False`、`clinical_500_executed=False`、`rgb_encoder_training_executed=False`。
