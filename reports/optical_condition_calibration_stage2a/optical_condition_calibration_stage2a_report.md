<!-- task: optical_condition_calibration_stage2a -->
# 第二阶段A：EXIF与设备条件化的区域光学表型校准

## 1. 完成状态

- `OPTICAL_CALIBRATION_STAGE2A_STATUS=COMPLETE`
- 本产物是 acquisition-conditioned optical calibration（采集条件校准后的区域光学表型），不是皮肤真实反射率、传感器线性RGB、生理参数或完整物理反演。

## 2. 新增/修改文件

- `utils/optical_condition_calibration.py`
- `scripts/preprocess/run_optical_condition_calibration_stage2a.py`
- `config/preprocess/optical_condition_calibration_stage2a.yaml`
- `tests/test_optical_condition_calibration_stage2a.py`
- `tests/test_optical_calibration_fivefold_protocol.py`
- `experiments/optical_condition_calibration_stage2a/fold_0/calibration_diagnostics.csv`
- `experiments/optical_condition_calibration_stage2a/fold_0/cheek_calibrator.json`
- `experiments/optical_condition_calibration_stage2a/fold_0/coefficient_table.csv`
- `experiments/optical_condition_calibration_stage2a/fold_0/condition_scaler.json`
- `experiments/optical_condition_calibration_stage2a/fold_0/fold_summary.json`
- `experiments/optical_condition_calibration_stage2a/fold_0/forehead_cheek_calibrator.json`
- `experiments/optical_condition_calibration_stage2a/fold_0/train_calibrated_features.csv`
- `experiments/optical_condition_calibration_stage2a/fold_0/val_calibrated_features.csv`
- `experiments/optical_condition_calibration_stage2a/fold_1/calibration_diagnostics.csv`
- `experiments/optical_condition_calibration_stage2a/fold_1/cheek_calibrator.json`
- `experiments/optical_condition_calibration_stage2a/fold_1/coefficient_table.csv`
- `experiments/optical_condition_calibration_stage2a/fold_1/condition_scaler.json`
- `experiments/optical_condition_calibration_stage2a/fold_1/fold_summary.json`
- `experiments/optical_condition_calibration_stage2a/fold_1/forehead_cheek_calibrator.json`
- `experiments/optical_condition_calibration_stage2a/fold_1/train_calibrated_features.csv`
- `experiments/optical_condition_calibration_stage2a/fold_1/val_calibrated_features.csv`
- `experiments/optical_condition_calibration_stage2a/fold_2/calibration_diagnostics.csv`
- `experiments/optical_condition_calibration_stage2a/fold_2/cheek_calibrator.json`
- `experiments/optical_condition_calibration_stage2a/fold_2/coefficient_table.csv`
- `experiments/optical_condition_calibration_stage2a/fold_2/condition_scaler.json`
- `experiments/optical_condition_calibration_stage2a/fold_2/fold_summary.json`
- `experiments/optical_condition_calibration_stage2a/fold_2/forehead_cheek_calibrator.json`
- `experiments/optical_condition_calibration_stage2a/fold_2/train_calibrated_features.csv`
- `experiments/optical_condition_calibration_stage2a/fold_2/val_calibrated_features.csv`
- `experiments/optical_condition_calibration_stage2a/fold_3/calibration_diagnostics.csv`
- `experiments/optical_condition_calibration_stage2a/fold_3/cheek_calibrator.json`
- `experiments/optical_condition_calibration_stage2a/fold_3/coefficient_table.csv`
- `experiments/optical_condition_calibration_stage2a/fold_3/condition_scaler.json`
- `experiments/optical_condition_calibration_stage2a/fold_3/fold_summary.json`
- `experiments/optical_condition_calibration_stage2a/fold_3/forehead_cheek_calibrator.json`
- `experiments/optical_condition_calibration_stage2a/fold_3/train_calibrated_features.csv`
- `experiments/optical_condition_calibration_stage2a/fold_3/val_calibrated_features.csv`
- `experiments/optical_condition_calibration_stage2a/fold_4/calibration_diagnostics.csv`
- `experiments/optical_condition_calibration_stage2a/fold_4/cheek_calibrator.json`
- `experiments/optical_condition_calibration_stage2a/fold_4/coefficient_table.csv`
- `experiments/optical_condition_calibration_stage2a/fold_4/condition_scaler.json`
- `experiments/optical_condition_calibration_stage2a/fold_4/fold_summary.json`
- `experiments/optical_condition_calibration_stage2a/fold_4/forehead_cheek_calibrator.json`
- `experiments/optical_condition_calibration_stage2a/fold_4/train_calibrated_features.csv`
- `experiments/optical_condition_calibration_stage2a/fold_4/val_calibrated_features.csv`
- `experiments/optical_condition_calibration_stage2a/protocol/split_audit.csv`
- `experiments/optical_condition_calibration_stage2a/protocol/split_manifest.json`
- `experiments/optical_condition_calibration_stage2a/summary/calibration_feature_schema.json`
- `experiments/optical_condition_calibration_stage2a/summary/calibration_summary.json`
- `experiments/optical_condition_calibration_stage2a/summary/coefficient_stability.csv`
- `experiments/optical_condition_calibration_stage2a/summary/fold_diagnostics_all.csv`
- `experiments/optical_condition_calibration_stage2a/summary/oof_calibrated_features.csv`
- `experiments/optical_condition_calibration_stage2a/summary/run_manifest.json`
- `reports/optical_condition_calibration_stage2a/coefficient_stability.csv`
- `reports/optical_condition_calibration_stage2a/fold_calibration_metrics.csv`
- `reports/optical_condition_calibration_stage2a/optical_condition_calibration_stage2a_report.md`
- `reports/optical_condition_calibration_stage2a/raw_vs_calibrated_camera_differences.csv`
- `reports/optical_condition_calibration_stage2a/raw_vs_calibrated_exif_correlations.csv`
- `reports/optical_condition_calibration_stage2a/run.log`
- `reports/optical_condition_calibration_stage2a/variance_retention.csv`

## 3. 第一阶段输入来源

- `data/processed/optical_observations_v1/regional_optical_observations.csv`（SHA256 `5d3fe7109846b5ee5fc0ba0034de2294caf3670a9b8552277681e28b86a5e0c6`）
- `data/processed/optical_observations_v1/feature_schema.json`与`data/processed/optical_observations_v1/extraction_manifest.json`。

## 4. 普通五折split来源和审计

固定使用`data/processed/splits_500/fold_{fold}_{train|val}.csv`，组合SHA256为`fe5102c02890c546f323b0a94ebc5b125ebcfeb50e62d2d43f0564b4b383f24b`；未重新生成或改变任何ID所属fold。

| fold | train_n | val_n | train_val_overlap_n | cheek_calibrator_train_n | forehead_cheek_calibrator_train_n |
| --- | --- | --- | --- | --- | --- |
| 0.0 | 400.0 | 100.0 | 0.0 | 400.0 | 390.0 |
| 1.0 | 400.0 | 100.0 | 0.0 | 400.0 | 389.0 |
| 2.0 | 400.0 | 100.0 | 0.0 | 400.0 | 389.0 |
| 3.0 | 400.0 | 100.0 | 0.0 | 400.0 | 387.0 |
| 4.0 | 400.0 | 100.0 | 0.0 | 400.0 | 389.0 |

## 5. 未读取NYHA声明

校准代码对白名单读取第一阶段字段；split仅以`usecols=['ID','fold']`读取。`clinical_labels_loaded=false`，`nyha_used=false`。

## 6. 六维观测定义

脸颊三维为`cheek_mean_log2_y/log2_rg/log2_bg`；额部—脸颊三维为`forehead_minus_cheek_log2_y/log2_rg/log2_bg`。

## 7. EXIF条件定义

仅使用`relative_optical_exposure`、`log2_iso_condition`和`camera_id`。FNumber只通过`log2(ExposureTime/FNumber^2)`进入条件；BrightnessValue未读取。

## 8. 设备内标准化方法

每fold仅在训练集内、分别按两设备计算均值和population std（ddof=0）；验证集复用对应训练参数。若std<1e-8则z统一为0并记录。

## 9. 设计矩阵

固定列顺序：`camera_xiaomi, z_relative_optical_exposure, z_log2_iso_condition, camera_xiaomi_x_z_exposure, camera_xiaomi_x_z_iso`，另含独立截距。

## 10. Ridge公式和alpha

使用NumPy闭式多输出Ridge：`sum((Y-Y_hat)^2)+1.0*sum(beta^2)`；截距不惩罚，alpha不扫描，求解失败时才显式使用pinv回退。

## 11. 两个校准器的训练子集

Cheek校准器使用每fold全部400个训练病例；额部—脸颊校准器只使用`forehead_available==1`且三目标有限的训练病例。

## 12. 额部缺失处理

额部不可用病例被保留，额部—脸颊的raw、predicted、residual、calibrated均保持NaN；未插补、填0、替代或重算Mask。

## 13. residual和calibrated定义

`residual=raw-predicted_acquisition`；`calibrated=residual+current_fold_training_reference_mean`。训练和验证使用同一训练参考均值。

## 14. 每fold训练及验证数量

| fold | train_n | val_n | train_forehead_available_n | val_forehead_available_n | degenerate_condition_count |
| --- | --- | --- | --- | --- | --- |
| 0.0 | 400.0 | 100.0 | 390.0 | 96.0 | 0.0 |
| 1.0 | 400.0 | 100.0 | 389.0 | 97.0 | 0.0 |
| 2.0 | 400.0 | 100.0 | 389.0 | 97.0 | 0.0 |
| 3.0 | 400.0 | 100.0 | 387.0 | 99.0 | 0.0 |
| 4.0 | 400.0 | 100.0 | 389.0 | 97.0 | 0.0 |

## 15. 每fold条件拟合误差

| fold | target | valid_n | mae | rmse | r2 |
| --- | --- | --- | --- | --- | --- |
| 0.0 | cheek_mean_log2_y | 100.0 | 0.260853 | 0.363679 | 0.244465 |
| 0.0 | cheek_mean_log2_rg | 100.0 | 0.103913 | 0.130847 | 0.061228 |
| 0.0 | cheek_mean_log2_bg | 100.0 | 0.113758 | 0.141741 | 0.085756 |
| 0.0 | forehead_minus_cheek_log2_y | 96.0 | 0.225751 | 0.296218 | -0.115476 |
| 0.0 | forehead_minus_cheek_log2_rg | 96.0 | 0.086767 | 0.11327 | 0.071634 |
| 0.0 | forehead_minus_cheek_log2_bg | 96.0 | 0.076303 | 0.099984 | 0.102656 |
| 1.0 | cheek_mean_log2_y | 100.0 | 0.261089 | 0.33168 | 0.345328 |
| 1.0 | cheek_mean_log2_rg | 100.0 | 0.110524 | 0.136769 | 0.036764 |
| 1.0 | cheek_mean_log2_bg | 100.0 | 0.102323 | 0.124988 | 0.084041 |
| 1.0 | forehead_minus_cheek_log2_y | 97.0 | 0.250127 | 0.353543 | -0.018457 |
| 1.0 | forehead_minus_cheek_log2_rg | 97.0 | 0.086332 | 0.114917 | 0.033728 |
| 1.0 | forehead_minus_cheek_log2_bg | 97.0 | 0.097035 | 0.117371 | 0.094869 |
| 2.0 | cheek_mean_log2_y | 100.0 | 0.226045 | 0.306147 | 0.428829 |
| 2.0 | cheek_mean_log2_rg | 100.0 | 0.112144 | 0.146617 | 0.036475 |
| 2.0 | cheek_mean_log2_bg | 100.0 | 0.102417 | 0.133842 | -0.026486 |
| 2.0 | forehead_minus_cheek_log2_y | 97.0 | 0.21081 | 0.271757 | -0.037033 |
| 2.0 | forehead_minus_cheek_log2_rg | 97.0 | 0.079162 | 0.101987 | 0.059172 |
| 2.0 | forehead_minus_cheek_log2_bg | 97.0 | 0.09522 | 0.118463 | 0.032532 |
| 3.0 | cheek_mean_log2_y | 100.0 | 0.237714 | 0.317274 | 0.431333 |
| 3.0 | cheek_mean_log2_rg | 100.0 | 0.106785 | 0.135956 | 0.136442 |
| 3.0 | cheek_mean_log2_bg | 100.0 | 0.109491 | 0.135408 | 0.025042 |
| 3.0 | forehead_minus_cheek_log2_y | 99.0 | 0.270545 | 0.383495 | -0.025355 |
| 3.0 | forehead_minus_cheek_log2_rg | 99.0 | 0.088067 | 0.109577 | 0.040035 |
| 3.0 | forehead_minus_cheek_log2_bg | 99.0 | 0.091932 | 0.113213 | -0.019998 |
| 4.0 | cheek_mean_log2_y | 100.0 | 0.292125 | 0.439813 | 0.358868 |
| 4.0 | cheek_mean_log2_rg | 100.0 | 0.11637 | 0.143854 | 0.140745 |
| 4.0 | cheek_mean_log2_bg | 100.0 | 0.113039 | 0.140402 | 0.067674 |
| 4.0 | forehead_minus_cheek_log2_y | 97.0 | 0.253902 | 0.335974 | 0.005345 |
| 4.0 | forehead_minus_cheek_log2_rg | 97.0 | 0.097359 | 0.121537 | 0.085177 |
| 4.0 | forehead_minus_cheek_log2_bg | 97.0 | 0.086558 | 0.110382 | 0.142319 |

## 16. 校准前后EXIF相关性

| fold | target | representation | condition | valid_n | spearman_rho |
| --- | --- | --- | --- | --- | --- |
| 0.0 | cheek_mean_log2_y | raw | relative_optical_exposure | 100.0 | -0.424117 |
| 0.0 | cheek_mean_log2_y | raw | log2_iso_condition | 100.0 | 0.324181 |
| 0.0 | cheek_mean_log2_y | calibrated | relative_optical_exposure | 100.0 | 0.016323 |
| 0.0 | cheek_mean_log2_y | calibrated | log2_iso_condition | 100.0 | 0.139734 |
| 0.0 | forehead_minus_cheek_log2_y | raw | relative_optical_exposure | 96.0 | 0.102432 |
| 0.0 | forehead_minus_cheek_log2_y | raw | log2_iso_condition | 96.0 | -0.37653 |
| 0.0 | forehead_minus_cheek_log2_y | calibrated | relative_optical_exposure | 96.0 | 0.034027 |
| 0.0 | forehead_minus_cheek_log2_y | calibrated | log2_iso_condition | 96.0 | -0.330346 |
| 1.0 | cheek_mean_log2_y | raw | relative_optical_exposure | 100.0 | -0.226103 |
| 1.0 | cheek_mean_log2_y | raw | log2_iso_condition | 100.0 | 0.300937 |
| 1.0 | cheek_mean_log2_y | calibrated | relative_optical_exposure | 100.0 | 0.228044 |
| 1.0 | cheek_mean_log2_y | calibrated | log2_iso_condition | 100.0 | -0.049896 |
| 1.0 | forehead_minus_cheek_log2_y | raw | relative_optical_exposure | 97.0 | 0.117847 |
| 1.0 | forehead_minus_cheek_log2_y | raw | log2_iso_condition | 97.0 | -0.081502 |
| 1.0 | forehead_minus_cheek_log2_y | calibrated | relative_optical_exposure | 97.0 | 0.136284 |
| 1.0 | forehead_minus_cheek_log2_y | calibrated | log2_iso_condition | 97.0 | 0.110394 |
| 2.0 | cheek_mean_log2_y | raw | relative_optical_exposure | 100.0 | -0.491004 |
| 2.0 | cheek_mean_log2_y | raw | log2_iso_condition | 100.0 | 0.110671 |
| 2.0 | cheek_mean_log2_y | calibrated | relative_optical_exposure | 100.0 | -0.190639 |
| 2.0 | cheek_mean_log2_y | calibrated | log2_iso_condition | 100.0 | -0.313402 |
| 2.0 | forehead_minus_cheek_log2_y | raw | relative_optical_exposure | 97.0 | -0.149325 |
| 2.0 | forehead_minus_cheek_log2_y | raw | log2_iso_condition | 97.0 | -0.213236 |
| 2.0 | forehead_minus_cheek_log2_y | calibrated | relative_optical_exposure | 97.0 | -0.183643 |
| 2.0 | forehead_minus_cheek_log2_y | calibrated | log2_iso_condition | 97.0 | -0.068643 |
| 3.0 | cheek_mean_log2_y | raw | relative_optical_exposure | 100.0 | -0.468766 |
| 3.0 | cheek_mean_log2_y | raw | log2_iso_condition | 100.0 | 0.317567 |
| 3.0 | cheek_mean_log2_y | calibrated | relative_optical_exposure | 100.0 | 0.011307 |
| 3.0 | cheek_mean_log2_y | calibrated | log2_iso_condition | 100.0 | -0.087289 |
| 3.0 | forehead_minus_cheek_log2_y | raw | relative_optical_exposure | 99.0 | 0.110267 |
| 3.0 | forehead_minus_cheek_log2_y | raw | log2_iso_condition | 99.0 | -0.098386 |
| 3.0 | forehead_minus_cheek_log2_y | calibrated | relative_optical_exposure | 99.0 | 0.050158 |
| 3.0 | forehead_minus_cheek_log2_y | calibrated | log2_iso_condition | 99.0 | 0.027857 |
| 4.0 | cheek_mean_log2_y | raw | relative_optical_exposure | 100.0 | -0.475531 |
| 4.0 | cheek_mean_log2_y | raw | log2_iso_condition | 100.0 | 0.311298 |
| 4.0 | cheek_mean_log2_y | calibrated | relative_optical_exposure | 100.0 | -0.219086 |
| 4.0 | cheek_mean_log2_y | calibrated | log2_iso_condition | 100.0 | 0.139457 |
| 4.0 | forehead_minus_cheek_log2_y | raw | relative_optical_exposure | 97.0 | -0.003024 |
| 4.0 | forehead_minus_cheek_log2_y | raw | log2_iso_condition | 97.0 | -0.126003 |
| 4.0 | forehead_minus_cheek_log2_y | calibrated | relative_optical_exposure | 97.0 | -0.032022 |
| 4.0 | forehead_minus_cheek_log2_y | calibrated | log2_iso_condition | 97.0 | 0.000638 |

这些相关性仅用于描述校准行为，不做显著性筛选或调参。

## 17. 校准前后设备差异

| fold | target | representation | mean_difference_honor_minus_xiaomi | median_difference_honor_minus_xiaomi | standardized_mean_difference |
| --- | --- | --- | --- | --- | --- |
| 0.0 | cheek_mean_log2_y | raw | 0.491223 | 0.54527 | 1.433238 |
| 0.0 | cheek_mean_log2_y | calibrated | -0.065963 | -0.042711 | -0.182074 |
| 0.0 | forehead_minus_cheek_log2_y | raw | -0.114585 | -0.105192 | -0.412963 |
| 0.0 | forehead_minus_cheek_log2_y | calibrated | -0.087073 | -0.112683 | -0.31306 |
| 1.0 | cheek_mean_log2_y | raw | 0.479812 | 0.487493 | 1.425289 |
| 1.0 | cheek_mean_log2_y | calibrated | -0.11284 | -0.122336 | -0.343418 |
| 1.0 | forehead_minus_cheek_log2_y | raw | 0.001951 | -0.06246 | 0.005511 |
| 1.0 | forehead_minus_cheek_log2_y | calibrated | 0.066745 | 0.0086 | 0.187659 |
| 2.0 | cheek_mean_log2_y | raw | 0.539898 | 0.604413 | 1.751772 |
| 2.0 | cheek_mean_log2_y | calibrated | -0.026544 | -0.018406 | -0.090141 |
| 2.0 | forehead_minus_cheek_log2_y | raw | -0.022606 | -0.004919 | -0.083906 |
| 2.0 | forehead_minus_cheek_log2_y | calibrated | 0.044629 | 0.023512 | 0.163076 |
| 3.0 | cheek_mean_log2_y | raw | 0.499311 | 0.430159 | 1.449522 |
| 3.0 | cheek_mean_log2_y | calibrated | -0.072168 | -0.117838 | -0.226608 |
| 3.0 | forehead_minus_cheek_log2_y | raw | -0.07334 | -0.062685 | -0.192568 |
| 3.0 | forehead_minus_cheek_log2_y | calibrated | -0.017892 | -0.00395 | -0.046501 |
| 4.0 | cheek_mean_log2_y | raw | 0.764495 | 0.719587 | 1.91862 |
| 4.0 | cheek_mean_log2_y | calibrated | 0.2895 | 0.301062 | 0.718771 |
| 4.0 | forehead_minus_cheek_log2_y | raw | -0.041482 | 0.019478 | -0.122092 |
| 4.0 | forehead_minus_cheek_log2_y | calibrated | 0.003747 | 0.033771 | 0.011189 |

设备差异可能降低，也可能不降低；不声称完全消除设备差异。

## 18. 方差保留

| fold | target | valid_n | raw_variance | calibrated_variance | variance_retention |
| --- | --- | --- | --- | --- | --- |
| 0.0 | cheek_mean_log2_y | 100.0 | 0.175058 | 0.129706 | 0.740929 |
| 0.0 | cheek_mean_log2_rg | 100.0 | 0.018238 | 0.016948 | 0.929284 |
| 0.0 | cheek_mean_log2_bg | 100.0 | 0.021975 | 0.020086 | 0.914035 |
| 0.0 | forehead_minus_cheek_log2_y | 96.0 | 0.078662 | 0.07764 | 0.98701 |
| 0.0 | forehead_minus_cheek_log2_rg | 96.0 | 0.01382 | 0.012754 | 0.922841 |
| 0.0 | forehead_minus_cheek_log2_bg | 96.0 | 0.01114 | 0.009819 | 0.881349 |
| 1.0 | cheek_mean_log2_y | 100.0 | 0.16804 | 0.108955 | 0.648388 |
| 1.0 | cheek_mean_log2_rg | 100.0 | 0.01942 | 0.018566 | 0.956029 |
| 1.0 | cheek_mean_log2_bg | 100.0 | 0.017055 | 0.015444 | 0.905517 |
| 1.0 | forehead_minus_cheek_log2_y | 97.0 | 0.122727 | 0.124993 | 1.018457 |
| 1.0 | forehead_minus_cheek_log2_rg | 97.0 | 0.013667 | 0.012924 | 0.945683 |
| 1.0 | forehead_minus_cheek_log2_bg | 97.0 | 0.01522 | 0.013607 | 0.894049 |
| 2.0 | cheek_mean_log2_y | 100.0 | 0.164095 | 0.08515 | 0.518908 |
| 2.0 | cheek_mean_log2_rg | 100.0 | 0.02231 | 0.018619 | 0.83454 |
| 2.0 | cheek_mean_log2_bg | 100.0 | 0.017452 | 0.017867 | 1.023823 |
| 2.0 | forehead_minus_cheek_log2_y | 97.0 | 0.071215 | 0.073832 | 1.036751 |
| 2.0 | forehead_minus_cheek_log2_rg | 97.0 | 0.011056 | 0.009958 | 0.900691 |
| 2.0 | forehead_minus_cheek_log2_bg | 97.0 | 0.014505 | 0.013925 | 0.960009 |
| 3.0 | cheek_mean_log2_y | 100.0 | 0.177016 | 0.100663 | 0.568664 |
| 3.0 | cheek_mean_log2_rg | 100.0 | 0.021405 | 0.018402 | 0.859726 |
| 3.0 | cheek_mean_log2_bg | 100.0 | 0.018806 | 0.017933 | 0.953584 |
| 3.0 | forehead_minus_cheek_log2_y | 99.0 | 0.143432 | 0.145129 | 1.011833 |
| 3.0 | forehead_minus_cheek_log2_rg | 99.0 | 0.012508 | 0.011936 | 0.954294 |
| 3.0 | forehead_minus_cheek_log2_bg | 99.0 | 0.012566 | 0.012692 | 1.010045 |
| 4.0 | cheek_mean_log2_y | 100.0 | 0.301709 | 0.179933 | 0.596378 |
| 4.0 | cheek_mean_log2_rg | 100.0 | 0.024084 | 0.019165 | 0.795763 |
| 4.0 | cheek_mean_log2_bg | 100.0 | 0.021144 | 0.01971 | 0.932204 |
| 4.0 | forehead_minus_cheek_log2_y | 97.0 | 0.113485 | 0.109834 | 0.967832 |
| 4.0 | forehead_minus_cheek_log2_rg | 97.0 | 0.016146 | 0.014742 | 0.913044 |
| 4.0 | forehead_minus_cheek_log2_bg | 97.0 | 0.014206 | 0.012183 | 0.857577 |

方差保留不是越小越好，仅用于描述校准后的观测变异。

## 19. 五折系数稳定性

| target | coefficient_name | fold_valid_n | mean | std | min | max | sign_consistent_fold_n |
| --- | --- | --- | --- | --- | --- | --- | --- |
| cheek_mean_log2_bg | device_intercept_difference | 5.0 | 0.042102 | 0.005576 | 0.034581 | 0.048555 | 5.0 |
| cheek_mean_log2_bg | honor_exposure_slope | 5.0 | -0.028363 | 0.001707 | -0.031263 | -0.026312 | 5.0 |
| cheek_mean_log2_bg | honor_iso_slope | 5.0 | -0.027756 | 0.004152 | -0.033477 | -0.022925 | 5.0 |
| cheek_mean_log2_bg | xiaomi_exposure_slope | 5.0 | -0.00373 | 0.011654 | -0.014747 | 0.015259 | 3.0 |
| cheek_mean_log2_bg | xiaomi_iso_slope | 5.0 | 0.011252 | 0.012557 | -0.009938 | 0.028203 | 4.0 |
| cheek_mean_log2_rg | device_intercept_difference | 5.0 | 0.112679 | 0.006771 | 0.104156 | 0.123555 | 5.0 |
| cheek_mean_log2_rg | honor_exposure_slope | 5.0 | 0.002162 | 0.002903 | -0.000473 | 0.007157 | 3.0 |
| cheek_mean_log2_rg | honor_iso_slope | 5.0 | -0.00249 | 0.002553 | -0.006154 | 0.001769 | 4.0 |
| cheek_mean_log2_rg | xiaomi_exposure_slope | 5.0 | -0.009627 | 0.014561 | -0.031064 | 0.007133 | 3.0 |
| cheek_mean_log2_rg | xiaomi_iso_slope | 5.0 | -0.016578 | 0.009532 | -0.028784 | -0.000802 | 5.0 |
| cheek_mean_log2_y | device_intercept_difference | 5.0 | -0.553017 | 0.02875 | -0.573936 | -0.497461 | 5.0 |
| cheek_mean_log2_y | honor_exposure_slope | 5.0 | 0.017282 | 0.010773 | -0.004162 | 0.024574 | 4.0 |
| cheek_mean_log2_y | honor_iso_slope | 5.0 | 0.081737 | 0.008746 | 0.07151 | 0.091475 | 5.0 |
| cheek_mean_log2_y | xiaomi_exposure_slope | 5.0 | -0.013823 | 0.035378 | -0.036844 | 0.056294 | 4.0 |
| cheek_mean_log2_y | xiaomi_iso_slope | 5.0 | -0.084255 | 0.046859 | -0.169329 | -0.033639 | 5.0 |
| forehead_minus_cheek_log2_bg | device_intercept_difference | 5.0 | -0.070878 | 0.003592 | -0.077633 | -0.067144 | 5.0 |
| forehead_minus_cheek_log2_bg | honor_exposure_slope | 5.0 | -0.000623 | 0.007144 | -0.009675 | 0.007426 | 3.0 |
| forehead_minus_cheek_log2_bg | honor_iso_slope | 5.0 | -0.012974 | 0.004084 | -0.018532 | -0.006969 | 5.0 |
| forehead_minus_cheek_log2_bg | xiaomi_exposure_slope | 5.0 | 0.013581 | 0.006005 | 0.006451 | 0.024146 | 5.0 |
| forehead_minus_cheek_log2_bg | xiaomi_iso_slope | 5.0 | 0.002824 | 0.003995 | -0.004588 | 0.006722 | 4.0 |
| forehead_minus_cheek_log2_rg | device_intercept_difference | 5.0 | -0.065129 | 0.002389 | -0.067096 | -0.060581 | 5.0 |
| forehead_minus_cheek_log2_rg | honor_exposure_slope | 5.0 | 0.001497 | 0.003042 | -0.002111 | 0.006202 | 3.0 |
| forehead_minus_cheek_log2_rg | honor_iso_slope | 5.0 | 0.019706 | 0.003642 | 0.015191 | 0.023093 | 5.0 |
| forehead_minus_cheek_log2_rg | xiaomi_exposure_slope | 5.0 | -0.007284 | 0.004765 | -0.014266 | 2.9e-05 | 4.0 |
| forehead_minus_cheek_log2_rg | xiaomi_iso_slope | 5.0 | 0.007992 | 0.009289 | -0.003814 | 0.021515 | 3.0 |
| forehead_minus_cheek_log2_y | device_intercept_difference | 5.0 | 0.052768 | 0.012032 | 0.032784 | 0.066619 | 5.0 |
| forehead_minus_cheek_log2_y | honor_exposure_slope | 5.0 | -0.004445 | 0.012041 | -0.023542 | 0.008479 | 3.0 |
| forehead_minus_cheek_log2_y | honor_iso_slope | 5.0 | -0.050564 | 0.008773 | -0.062642 | -0.036563 | 5.0 |
| forehead_minus_cheek_log2_y | xiaomi_exposure_slope | 5.0 | 0.064865 | 0.008568 | 0.051746 | 0.077365 | 5.0 |
| forehead_minus_cheek_log2_y | xiaomi_iso_slope | 5.0 | -0.050791 | 0.027112 | -0.089775 | -0.01385 | 5.0 |

- Exposure斜率：3/12个维度在全部有效fold中方向一致；其中稳定正向2个、稳定负向1个。
- ISO斜率：8/12个维度在全部有效fold中方向一致；其中稳定正向2个、稳定负向6个。
- 设备截距差：6/6个维度在全部有效fold中方向一致；其中稳定正向3个、稳定负向3个。
- 未达到全fold方向一致的维度：`cheek_mean_log2_bg / xiaomi_exposure_slope`=3/5；`cheek_mean_log2_rg / honor_exposure_slope`=3/5；`cheek_mean_log2_rg / xiaomi_exposure_slope`=3/5；`cheek_mean_log2_y / honor_exposure_slope`=4/5；`cheek_mean_log2_y / xiaomi_exposure_slope`=4/5；`forehead_minus_cheek_log2_bg / honor_exposure_slope`=3/5；`forehead_minus_cheek_log2_rg / honor_exposure_slope`=3/5；`forehead_minus_cheek_log2_rg / xiaomi_exposure_slope`=4/5；`forehead_minus_cheek_log2_y / honor_exposure_slope`=3/5；`cheek_mean_log2_bg / xiaomi_iso_slope`=4/5；`cheek_mean_log2_rg / honor_iso_slope`=4/5；`forehead_minus_cheek_log2_bg / xiaomi_iso_slope`=4/5；`forehead_minus_cheek_log2_rg / xiaomi_iso_slope`=3/5。
- 这些结果仅描述五折稳定性；未据此改变alpha、筛选维度或重拟合模型。

## 20. OOF完整性

OOF=500行、唯一ID=500；额部可用=486、不可用=14；非有限cheek输出=0；非法额部输出=0。

## 21. 单元测试及协议测试

单元测试：`PASS`（................                                                         [100%] | 16 passed in 2.14s）；五折协议测试：`PASS`（....                                                                     [100%] | 4 passed in 1.57s）。

## 22. 确定性验证

在实验输出目录内的临时目录重复完整核心五折，全部CSV/模型JSON SHA256一致：`True`。

## 23. 历史输入未修改声明

第一阶段CSV、schema、manifest及十份固定split在运行前后SHA256一致；`historical_inputs_modified=false`。

## 24. 局限性

1. 这是低容量线性采集条件校准基线，不是因果分解。
2. 模型不能恢复手机ISP处理前的传感器信号，也不能得到真实皮肤反射率。
3. 仅有两种设备；对未知设备不做外推编码。
4. 校准诊断不使用NYHA，不能证明校准特征与NYHA相关或提高分类性能。

## 25. 下一阶段可使用的六个calibrated字段

- `calibrated_cheek_mean_log2_y`
- `calibrated_cheek_mean_log2_rg`
- `calibrated_cheek_mean_log2_bg`
- `calibrated_forehead_minus_cheek_log2_y`
- `calibrated_forehead_minus_cheek_log2_rg`
- `calibrated_forehead_minus_cheek_log2_bg`
