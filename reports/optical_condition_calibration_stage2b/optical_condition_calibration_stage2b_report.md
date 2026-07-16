<!-- task: optical_condition_calibration_stage2b -->
# Stage 2B：EXIF与设备条件化的非线性区域光学表型校准网络

## 1. 完成状态

- `OPTICAL_CALIBRATION_STAGE2B_STATUS=COMPLETE`
- 本产物是EXIF-conditioned nonlinear acquisition response calibration / 非线性采集条件校准后的区域光学表型，也可称physics-inspired inverse representation；不是皮肤真实反射率、传感器线性RGB或医学指标。

## 2. 新增/修改文件

- `models/exif_conditioned_response_mlp.py`
- `utils/optical_condition_calibration_nn.py`
- `scripts/train/run_optical_condition_calibration_stage2b.py`
- `scripts/evaluate/compare_optical_calibration_stage2a_stage2b.py`
- `config/train/optical_condition_calibration_stage2b.yaml`
- `tests/test_exif_conditioned_response_mlp.py`
- `tests/test_optical_condition_calibration_stage2b.py`
- `tests/test_optical_calibration_stage2b_protocol.py`
- `experiments/optical_condition_calibration_stage2b/fold_0/cheek_epoch_selection/best_checkpoint.pth`
- `experiments/optical_condition_calibration_stage2b/fold_0/cheek_epoch_selection/inner_condition_scaler.json`
- `experiments/optical_condition_calibration_stage2b/fold_0/cheek_epoch_selection/inner_target_scaler.json`
- `experiments/optical_condition_calibration_stage2b/fold_0/cheek_epoch_selection/selected_epoch.json`
- `experiments/optical_condition_calibration_stage2b/fold_0/cheek_epoch_selection/selection_curve.png`
- `experiments/optical_condition_calibration_stage2b/fold_0/cheek_epoch_selection/training_log.csv`
- `experiments/optical_condition_calibration_stage2b/fold_0/cheek_final_model.pth`
- `experiments/optical_condition_calibration_stage2b/fold_0/cheek_final_model_manifest.json`
- `experiments/optical_condition_calibration_stage2b/fold_0/cheek_target_scaler.json`
- `experiments/optical_condition_calibration_stage2b/fold_0/condition_range_audit.csv`
- `experiments/optical_condition_calibration_stage2b/fold_0/diagnostics.csv`
- `experiments/optical_condition_calibration_stage2b/fold_0/fold_summary.json`
- `experiments/optical_condition_calibration_stage2b/fold_0/forehead_cheek_epoch_selection/best_checkpoint.pth`
- `experiments/optical_condition_calibration_stage2b/fold_0/forehead_cheek_epoch_selection/inner_condition_scaler.json`
- `experiments/optical_condition_calibration_stage2b/fold_0/forehead_cheek_epoch_selection/inner_target_scaler.json`
- `experiments/optical_condition_calibration_stage2b/fold_0/forehead_cheek_epoch_selection/selected_epoch.json`
- `experiments/optical_condition_calibration_stage2b/fold_0/forehead_cheek_epoch_selection/selection_curve.png`
- `experiments/optical_condition_calibration_stage2b/fold_0/forehead_cheek_epoch_selection/training_log.csv`
- `experiments/optical_condition_calibration_stage2b/fold_0/forehead_cheek_final_model.pth`
- `experiments/optical_condition_calibration_stage2b/fold_0/forehead_cheek_final_model_manifest.json`
- `experiments/optical_condition_calibration_stage2b/fold_0/forehead_cheek_target_scaler.json`
- `experiments/optical_condition_calibration_stage2b/fold_0/inner_split/cheek_inner_train.csv`
- `experiments/optical_condition_calibration_stage2b/fold_0/inner_split/cheek_inner_val.csv`
- `experiments/optical_condition_calibration_stage2b/fold_0/inner_split/forehead_cheek_inner_train.csv`
- `experiments/optical_condition_calibration_stage2b/fold_0/inner_split/forehead_cheek_inner_val.csv`
- `experiments/optical_condition_calibration_stage2b/fold_0/inner_split/inner_split_manifest.json`
- `experiments/optical_condition_calibration_stage2b/fold_0/train_nn_calibrated_features.csv`
- `experiments/optical_condition_calibration_stage2b/fold_0/val_nn_calibrated_features.csv`
- `experiments/optical_condition_calibration_stage2b/fold_1/cheek_epoch_selection/best_checkpoint.pth`
- `experiments/optical_condition_calibration_stage2b/fold_1/cheek_epoch_selection/inner_condition_scaler.json`
- `experiments/optical_condition_calibration_stage2b/fold_1/cheek_epoch_selection/inner_target_scaler.json`
- `experiments/optical_condition_calibration_stage2b/fold_1/cheek_epoch_selection/selected_epoch.json`
- `experiments/optical_condition_calibration_stage2b/fold_1/cheek_epoch_selection/selection_curve.png`
- `experiments/optical_condition_calibration_stage2b/fold_1/cheek_epoch_selection/training_log.csv`
- `experiments/optical_condition_calibration_stage2b/fold_1/cheek_final_model.pth`
- `experiments/optical_condition_calibration_stage2b/fold_1/cheek_final_model_manifest.json`
- `experiments/optical_condition_calibration_stage2b/fold_1/cheek_target_scaler.json`
- `experiments/optical_condition_calibration_stage2b/fold_1/condition_range_audit.csv`
- `experiments/optical_condition_calibration_stage2b/fold_1/diagnostics.csv`
- `experiments/optical_condition_calibration_stage2b/fold_1/fold_summary.json`
- `experiments/optical_condition_calibration_stage2b/fold_1/forehead_cheek_epoch_selection/best_checkpoint.pth`
- `experiments/optical_condition_calibration_stage2b/fold_1/forehead_cheek_epoch_selection/inner_condition_scaler.json`
- `experiments/optical_condition_calibration_stage2b/fold_1/forehead_cheek_epoch_selection/inner_target_scaler.json`
- `experiments/optical_condition_calibration_stage2b/fold_1/forehead_cheek_epoch_selection/selected_epoch.json`
- `experiments/optical_condition_calibration_stage2b/fold_1/forehead_cheek_epoch_selection/selection_curve.png`
- `experiments/optical_condition_calibration_stage2b/fold_1/forehead_cheek_epoch_selection/training_log.csv`
- `experiments/optical_condition_calibration_stage2b/fold_1/forehead_cheek_final_model.pth`
- `experiments/optical_condition_calibration_stage2b/fold_1/forehead_cheek_final_model_manifest.json`
- `experiments/optical_condition_calibration_stage2b/fold_1/forehead_cheek_target_scaler.json`
- `experiments/optical_condition_calibration_stage2b/fold_1/inner_split/cheek_inner_train.csv`
- `experiments/optical_condition_calibration_stage2b/fold_1/inner_split/cheek_inner_val.csv`
- `experiments/optical_condition_calibration_stage2b/fold_1/inner_split/forehead_cheek_inner_train.csv`
- `experiments/optical_condition_calibration_stage2b/fold_1/inner_split/forehead_cheek_inner_val.csv`
- `experiments/optical_condition_calibration_stage2b/fold_1/inner_split/inner_split_manifest.json`
- `experiments/optical_condition_calibration_stage2b/fold_1/train_nn_calibrated_features.csv`
- `experiments/optical_condition_calibration_stage2b/fold_1/val_nn_calibrated_features.csv`
- `experiments/optical_condition_calibration_stage2b/fold_2/cheek_epoch_selection/best_checkpoint.pth`
- `experiments/optical_condition_calibration_stage2b/fold_2/cheek_epoch_selection/inner_condition_scaler.json`
- `experiments/optical_condition_calibration_stage2b/fold_2/cheek_epoch_selection/inner_target_scaler.json`
- `experiments/optical_condition_calibration_stage2b/fold_2/cheek_epoch_selection/selected_epoch.json`
- `experiments/optical_condition_calibration_stage2b/fold_2/cheek_epoch_selection/selection_curve.png`
- `experiments/optical_condition_calibration_stage2b/fold_2/cheek_epoch_selection/training_log.csv`
- `experiments/optical_condition_calibration_stage2b/fold_2/cheek_final_model.pth`
- `experiments/optical_condition_calibration_stage2b/fold_2/cheek_final_model_manifest.json`
- `experiments/optical_condition_calibration_stage2b/fold_2/cheek_target_scaler.json`
- `experiments/optical_condition_calibration_stage2b/fold_2/condition_range_audit.csv`
- `experiments/optical_condition_calibration_stage2b/fold_2/diagnostics.csv`
- `experiments/optical_condition_calibration_stage2b/fold_2/fold_summary.json`
- `experiments/optical_condition_calibration_stage2b/fold_2/forehead_cheek_epoch_selection/best_checkpoint.pth`
- `experiments/optical_condition_calibration_stage2b/fold_2/forehead_cheek_epoch_selection/inner_condition_scaler.json`
- `experiments/optical_condition_calibration_stage2b/fold_2/forehead_cheek_epoch_selection/inner_target_scaler.json`
- `experiments/optical_condition_calibration_stage2b/fold_2/forehead_cheek_epoch_selection/selected_epoch.json`
- `experiments/optical_condition_calibration_stage2b/fold_2/forehead_cheek_epoch_selection/selection_curve.png`
- `experiments/optical_condition_calibration_stage2b/fold_2/forehead_cheek_epoch_selection/training_log.csv`
- `experiments/optical_condition_calibration_stage2b/fold_2/forehead_cheek_final_model.pth`
- `experiments/optical_condition_calibration_stage2b/fold_2/forehead_cheek_final_model_manifest.json`
- `experiments/optical_condition_calibration_stage2b/fold_2/forehead_cheek_target_scaler.json`
- `experiments/optical_condition_calibration_stage2b/fold_2/inner_split/cheek_inner_train.csv`
- `experiments/optical_condition_calibration_stage2b/fold_2/inner_split/cheek_inner_val.csv`
- `experiments/optical_condition_calibration_stage2b/fold_2/inner_split/forehead_cheek_inner_train.csv`
- `experiments/optical_condition_calibration_stage2b/fold_2/inner_split/forehead_cheek_inner_val.csv`
- `experiments/optical_condition_calibration_stage2b/fold_2/inner_split/inner_split_manifest.json`
- `experiments/optical_condition_calibration_stage2b/fold_2/train_nn_calibrated_features.csv`
- `experiments/optical_condition_calibration_stage2b/fold_2/val_nn_calibrated_features.csv`
- `experiments/optical_condition_calibration_stage2b/fold_3/cheek_epoch_selection/best_checkpoint.pth`
- `experiments/optical_condition_calibration_stage2b/fold_3/cheek_epoch_selection/inner_condition_scaler.json`
- `experiments/optical_condition_calibration_stage2b/fold_3/cheek_epoch_selection/inner_target_scaler.json`
- `experiments/optical_condition_calibration_stage2b/fold_3/cheek_epoch_selection/selected_epoch.json`
- `experiments/optical_condition_calibration_stage2b/fold_3/cheek_epoch_selection/selection_curve.png`
- `experiments/optical_condition_calibration_stage2b/fold_3/cheek_epoch_selection/training_log.csv`
- `experiments/optical_condition_calibration_stage2b/fold_3/cheek_final_model.pth`
- `experiments/optical_condition_calibration_stage2b/fold_3/cheek_final_model_manifest.json`
- `experiments/optical_condition_calibration_stage2b/fold_3/cheek_target_scaler.json`
- `experiments/optical_condition_calibration_stage2b/fold_3/condition_range_audit.csv`
- `experiments/optical_condition_calibration_stage2b/fold_3/diagnostics.csv`
- `experiments/optical_condition_calibration_stage2b/fold_3/fold_summary.json`
- `experiments/optical_condition_calibration_stage2b/fold_3/forehead_cheek_epoch_selection/best_checkpoint.pth`
- `experiments/optical_condition_calibration_stage2b/fold_3/forehead_cheek_epoch_selection/inner_condition_scaler.json`
- `experiments/optical_condition_calibration_stage2b/fold_3/forehead_cheek_epoch_selection/inner_target_scaler.json`
- `experiments/optical_condition_calibration_stage2b/fold_3/forehead_cheek_epoch_selection/selected_epoch.json`
- `experiments/optical_condition_calibration_stage2b/fold_3/forehead_cheek_epoch_selection/selection_curve.png`
- `experiments/optical_condition_calibration_stage2b/fold_3/forehead_cheek_epoch_selection/training_log.csv`
- `experiments/optical_condition_calibration_stage2b/fold_3/forehead_cheek_final_model.pth`
- `experiments/optical_condition_calibration_stage2b/fold_3/forehead_cheek_final_model_manifest.json`
- `experiments/optical_condition_calibration_stage2b/fold_3/forehead_cheek_target_scaler.json`
- `experiments/optical_condition_calibration_stage2b/fold_3/inner_split/cheek_inner_train.csv`
- `experiments/optical_condition_calibration_stage2b/fold_3/inner_split/cheek_inner_val.csv`
- `experiments/optical_condition_calibration_stage2b/fold_3/inner_split/forehead_cheek_inner_train.csv`
- `experiments/optical_condition_calibration_stage2b/fold_3/inner_split/forehead_cheek_inner_val.csv`
- `experiments/optical_condition_calibration_stage2b/fold_3/inner_split/inner_split_manifest.json`
- `experiments/optical_condition_calibration_stage2b/fold_3/train_nn_calibrated_features.csv`
- `experiments/optical_condition_calibration_stage2b/fold_3/val_nn_calibrated_features.csv`
- `experiments/optical_condition_calibration_stage2b/fold_4/cheek_epoch_selection/best_checkpoint.pth`
- `experiments/optical_condition_calibration_stage2b/fold_4/cheek_epoch_selection/inner_condition_scaler.json`
- `experiments/optical_condition_calibration_stage2b/fold_4/cheek_epoch_selection/inner_target_scaler.json`
- `experiments/optical_condition_calibration_stage2b/fold_4/cheek_epoch_selection/selected_epoch.json`
- `experiments/optical_condition_calibration_stage2b/fold_4/cheek_epoch_selection/selection_curve.png`
- `experiments/optical_condition_calibration_stage2b/fold_4/cheek_epoch_selection/training_log.csv`
- `experiments/optical_condition_calibration_stage2b/fold_4/cheek_final_model.pth`
- `experiments/optical_condition_calibration_stage2b/fold_4/cheek_final_model_manifest.json`
- `experiments/optical_condition_calibration_stage2b/fold_4/cheek_target_scaler.json`
- `experiments/optical_condition_calibration_stage2b/fold_4/condition_range_audit.csv`
- `experiments/optical_condition_calibration_stage2b/fold_4/diagnostics.csv`
- `experiments/optical_condition_calibration_stage2b/fold_4/fold_summary.json`
- `experiments/optical_condition_calibration_stage2b/fold_4/forehead_cheek_epoch_selection/best_checkpoint.pth`
- `experiments/optical_condition_calibration_stage2b/fold_4/forehead_cheek_epoch_selection/inner_condition_scaler.json`
- `experiments/optical_condition_calibration_stage2b/fold_4/forehead_cheek_epoch_selection/inner_target_scaler.json`
- `experiments/optical_condition_calibration_stage2b/fold_4/forehead_cheek_epoch_selection/selected_epoch.json`
- `experiments/optical_condition_calibration_stage2b/fold_4/forehead_cheek_epoch_selection/selection_curve.png`
- `experiments/optical_condition_calibration_stage2b/fold_4/forehead_cheek_epoch_selection/training_log.csv`
- `experiments/optical_condition_calibration_stage2b/fold_4/forehead_cheek_final_model.pth`
- `experiments/optical_condition_calibration_stage2b/fold_4/forehead_cheek_final_model_manifest.json`
- `experiments/optical_condition_calibration_stage2b/fold_4/forehead_cheek_target_scaler.json`
- `experiments/optical_condition_calibration_stage2b/fold_4/inner_split/cheek_inner_train.csv`
- `experiments/optical_condition_calibration_stage2b/fold_4/inner_split/cheek_inner_val.csv`
- `experiments/optical_condition_calibration_stage2b/fold_4/inner_split/forehead_cheek_inner_train.csv`
- `experiments/optical_condition_calibration_stage2b/fold_4/inner_split/forehead_cheek_inner_val.csv`
- `experiments/optical_condition_calibration_stage2b/fold_4/inner_split/inner_split_manifest.json`
- `experiments/optical_condition_calibration_stage2b/fold_4/train_nn_calibrated_features.csv`
- `experiments/optical_condition_calibration_stage2b/fold_4/val_nn_calibrated_features.csv`
- `experiments/optical_condition_calibration_stage2b/protocol/input_audit.csv`
- `experiments/optical_condition_calibration_stage2b/protocol/protocol_manifest.json`
- `experiments/optical_condition_calibration_stage2b/protocol/split_audit.csv`
- `experiments/optical_condition_calibration_stage2b/summary/calibration_stage2b_feature_schema.json`
- `experiments/optical_condition_calibration_stage2b/summary/condition_range_audit_all.csv`
- `experiments/optical_condition_calibration_stage2b/summary/fold_diagnostics_all.csv`
- `experiments/optical_condition_calibration_stage2b/summary/oof_nn_calibrated_features.csv`
- `experiments/optical_condition_calibration_stage2b/summary/run_manifest.json`
- `experiments/optical_condition_calibration_stage2b/summary/stage2a_vs_stage2b_per_fold.csv`
- `experiments/optical_condition_calibration_stage2b/summary/stage2a_vs_stage2b_summary.csv`
- `reports/optical_condition_calibration_stage2b/condition_range_audit.csv`
- `reports/optical_condition_calibration_stage2b/conditional_prediction_metrics.csv`
- `reports/optical_condition_calibration_stage2b/optical_condition_calibration_stage2b_report.md`
- `reports/optical_condition_calibration_stage2b/raw_ridge_mlp_camera_differences.csv`
- `reports/optical_condition_calibration_stage2b/residual_exif_correlations.csv`
- `reports/optical_condition_calibration_stage2b/run.log`
- `reports/optical_condition_calibration_stage2b/stage2a_vs_stage2b_summary.csv`
- `reports/optical_condition_calibration_stage2b/training_curves/fold_0_cheek_selection_curve.png`
- `reports/optical_condition_calibration_stage2b/training_curves/fold_0_forehead_cheek_selection_curve.png`
- `reports/optical_condition_calibration_stage2b/training_curves/fold_1_cheek_selection_curve.png`
- `reports/optical_condition_calibration_stage2b/training_curves/fold_1_forehead_cheek_selection_curve.png`
- `reports/optical_condition_calibration_stage2b/training_curves/fold_2_cheek_selection_curve.png`
- `reports/optical_condition_calibration_stage2b/training_curves/fold_2_forehead_cheek_selection_curve.png`
- `reports/optical_condition_calibration_stage2b/training_curves/fold_3_cheek_selection_curve.png`
- `reports/optical_condition_calibration_stage2b/training_curves/fold_3_forehead_cheek_selection_curve.png`
- `reports/optical_condition_calibration_stage2b/training_curves/fold_4_cheek_selection_curve.png`
- `reports/optical_condition_calibration_stage2b/training_curves/fold_4_forehead_cheek_selection_curve.png`
- `reports/optical_condition_calibration_stage2b/variance_retention_comparison.csv`

## 3. 第一阶段和Stage 2A输入来源

第一阶段：`data/processed/optical_observations_v1/regional_optical_observations.csv`，SHA256 `5d3fe7109846b5ee5fc0ba0034de2294caf3670a9b8552277681e28b86a5e0c6`；Stage 2A manifest SHA256 `35e09eb9dc5f981ecc473501b3a72365c32eb7d6363cf80c6a1bf67b20d0b714`；Stage 2A OOF SHA256 `f25a6e91717d9a8c9cccd7382e9abf0d48e26e3855613a36d074f92ba1f918cd`。

## 4. split和SHA256

固定普通五折split SHA256为`fe5102c02890c546f323b0a94ebc5b125ebcfeb50e62d2d43f0564b4b383f24b`，未生成或修改外层split。

| fold | outer_train_n | outer_val_n | cheek_final_train_n | forehead_cheek_final_train_n |
| --- | --- | --- | --- | --- |
| 0.0 | 400.0 | 100.0 | 400.0 | 390.0 |
| 1.0 | 400.0 | 100.0 | 400.0 | 389.0 |
| 2.0 | 400.0 | 100.0 | 400.0 | 389.0 |
| 3.0 | 400.0 | 100.0 | 400.0 | 387.0 |
| 4.0 | 400.0 | 100.0 | 400.0 | 389.0 |

## 5. 未读取NYHA声明

仅白名单读取ID、fold、camera、两个派生EXIF条件、forehead_available、六个观测和Stage 2A校准/诊断字段；`clinical_labels_loaded=false`、`nyha_used=false`、`global_features_used=false`。

## 6. Stage 2B数学定义

`predicted_condition_nn=MLP(c)`（先从target-z恢复原尺度）；`residual_nn=raw-predicted_condition_nn`；`calibrated_nn=residual_nn+outer_train_target_mean`。网络预测条件期望观测，不解释为真实曝光成分或真实设备响应。

## 7. MLP输入和输出

输入固定为`camera_xiaomi,z_exposure,z_iso,camera_xiaomi*z_exposure,camera_xiaomi*z_iso`；输出为三个标准化目标。原始区域观测、Stage 2A预测、QC、图像和临床字段均不进入网络。

## 8. 两个独立网络

每fold分别训练Cheek与Forehead-cheek网络；不共享参数。后者仅使用forehead_available=1且三目标有限的病例。

## 9. 网络结构及参数量

固定`Linear(5,8)-Tanh-Linear(8,8)-Tanh-Linear(8,3)`，单网络147参数、每fold两网络294参数；无BatchNorm、LayerNorm、Dropout、attention、卷积或自编码器。

## 10. 条件标准化

内部epoch选择仅由inner-train按设备拟合condition scaler；完整重拟合严格复用Stage 2A外层condition scaler。Stage 2B复算z与Stage 2A最大绝对误差见下表。

| fold | stage2a_z_max_absolute_error |
| --- | --- |
| 0.0 | 0.0 |
| 1.0 | 0.0 |
| 2.0 | 0.0 |
| 3.0 | 0.0 |
| 4.0 | 0.0 |

## 11. 目标标准化

Cheek与Forehead-cheek分别在当前训练子集逐目标计算population mean/std（ddof=0）；std<1e-8会停止该fold。内部选择和最终重拟合分别重新拟合target scaler。

## 12. 内部epoch选择协议

按camera分层，以`SHA256(seed|ID)`确定80/20 inner split；外层val未加载。指标为inner-val标准化MSE，改善阈值1e-6，并列保留更早epoch。

## 13. 完整训练折重拟合

选定epoch后丢弃inner checkpoint参数，以同一预设seed全新初始化；Cheek用400例，Forehead-cheek用全部额部可用outer-train病例，固定训练到selected epoch，无validation loader。

## 14. 固定训练配置

CPU/float32、AdamW、lr=1e-2、weight_decay=1e-4、full-batch、max_epochs=500、minimum_epochs=50、patience=50、gradient clip=5、无scheduler/AMP/搜索。

## 15. 额部缺失处理

额部不可用病例保留；三个Forehead-cheek目标的raw/predicted/residual/calibrated均为NaN，不参与内部split、scaler、训练或loss；cheek输出保留。

## 16. 五折selected epoch

| fold | cheek_selected_epoch | forehead_cheek_selected_epoch |
| --- | --- | --- |
| 0.0 | 75.0 | 57.0 |
| 1.0 | 102.0 | 19.0 |
| 2.0 | 80.0 | 19.0 |
| 3.0 | 130.0 | 68.0 |
| 4.0 | 249.0 | 64.0 |

## 17. inner train/val训练曲线

每fold、每网络的完整training_log、best checkpoint、selected_epoch和曲线均已保存；报告目录`training_curves/`包含10张选择曲线。

| fold | network_type | selected_epoch | inner_train_best_epoch_loss | inner_val_best_epoch_loss | final_outer_train_loss | outer_val_mse | outer_val_minus_train_mse |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 0.0 | cheek | 75.0 | 0.752815 | 0.743419 | 0.730507 | 0.797693 | 0.067187 |
| 0.0 | forehead_cheek | 57.0 | 0.921578 | 0.843854 | 0.923475 | 0.81314 | -0.110334 |
| 1.0 | cheek | 102.0 | 0.713467 | 0.790593 | 0.726662 | 0.694597 | -0.032065 |
| 1.0 | forehead_cheek | 19.0 | 0.921754 | 0.898085 | 0.92566 | 1.083966 | 0.158307 |
| 2.0 | cheek | 80.0 | 0.733643 | 0.882202 | 0.752161 | 0.758364 | 0.006202 |
| 2.0 | forehead_cheek | 19.0 | 0.905588 | 1.675623 | 0.923399 | 0.802304 | -0.121095 |
| 3.0 | cheek | 130.0 | 0.711987 | 0.729728 | 0.710551 | 0.734259 | 0.023708 |
| 3.0 | forehead_cheek | 68.0 | 0.887085 | 1.176481 | 0.893446 | 1.116604 | 0.223158 |
| 4.0 | cheek | 249.0 | 0.708319 | 0.687472 | 0.703337 | 1.027533 | 0.324196 |
| 4.0 | forehead_cheek | 64.0 | 0.92405 | 0.859932 | 0.920479 | 1.033415 | 0.112936 |

## 18. 外层条件预测MAE、RMSE和R²

| fold | target | valid_n | mae | rmse | r2 |
| --- | --- | --- | --- | --- | --- |
| 0.0 | cheek_mean_log2_y | 100.0 | 0.255142 | 0.353356 | 0.286749 |
| 0.0 | cheek_mean_log2_rg | 100.0 | 0.103958 | 0.132218 | 0.041447 |
| 0.0 | cheek_mean_log2_bg | 100.0 | 0.111747 | 0.138653 | 0.125165 |
| 0.0 | forehead_minus_cheek_log2_y | 96.0 | 0.222904 | 0.294796 | -0.104793 |
| 0.0 | forehead_minus_cheek_log2_rg | 96.0 | 0.087169 | 0.113764 | 0.063513 |
| 0.0 | forehead_minus_cheek_log2_bg | 96.0 | 0.07691 | 0.100601 | 0.091548 |
| 1.0 | cheek_mean_log2_y | 100.0 | 0.258391 | 0.326623 | 0.365139 |
| 1.0 | cheek_mean_log2_rg | 100.0 | 0.110317 | 0.135411 | 0.0558 |
| 1.0 | cheek_mean_log2_bg | 100.0 | 0.097398 | 0.123408 | 0.107057 |
| 1.0 | forehead_minus_cheek_log2_y | 97.0 | 0.250233 | 0.354766 | -0.025518 |
| 1.0 | forehead_minus_cheek_log2_rg | 97.0 | 0.085536 | 0.114603 | 0.039002 |
| 1.0 | forehead_minus_cheek_log2_bg | 97.0 | 0.097609 | 0.119237 | 0.065851 |
| 2.0 | cheek_mean_log2_y | 100.0 | 0.223069 | 0.300523 | 0.449623 |
| 2.0 | cheek_mean_log2_rg | 100.0 | 0.11023 | 0.144487 | 0.064268 |
| 2.0 | cheek_mean_log2_bg | 100.0 | 0.099726 | 0.129992 | 0.03172 |
| 2.0 | forehead_minus_cheek_log2_y | 97.0 | 0.214569 | 0.276921 | -0.076823 |
| 2.0 | forehead_minus_cheek_log2_rg | 97.0 | 0.07814 | 0.100453 | 0.087259 |
| 2.0 | forehead_minus_cheek_log2_bg | 97.0 | 0.095544 | 0.11777 | 0.043809 |
| 3.0 | cheek_mean_log2_y | 100.0 | 0.239477 | 0.321969 | 0.41438 |
| 3.0 | cheek_mean_log2_rg | 100.0 | 0.100695 | 0.13177 | 0.188807 |
| 3.0 | cheek_mean_log2_bg | 100.0 | 0.108671 | 0.133579 | 0.051194 |
| 3.0 | forehead_minus_cheek_log2_y | 99.0 | 0.277618 | 0.390222 | -0.061639 |
| 3.0 | forehead_minus_cheek_log2_rg | 99.0 | 0.088674 | 0.110515 | 0.023531 |
| 3.0 | forehead_minus_cheek_log2_bg | 99.0 | 0.092302 | 0.113237 | -0.02043 |
| 4.0 | cheek_mean_log2_y | 100.0 | 0.281915 | 0.424562 | 0.40256 |
| 4.0 | cheek_mean_log2_rg | 100.0 | 0.114257 | 0.141717 | 0.166083 |
| 4.0 | cheek_mean_log2_bg | 100.0 | 0.115327 | 0.143289 | 0.028932 |
| 4.0 | forehead_minus_cheek_log2_y | 97.0 | 0.2532 | 0.336212 | 0.003934 |
| 4.0 | forehead_minus_cheek_log2_rg | 97.0 | 0.097166 | 0.121601 | 0.084202 |
| 4.0 | forehead_minus_cheek_log2_bg | 97.0 | 0.08615 | 0.109879 | 0.150122 |

## 19. EXIF残余相关性

| fold | target | representation | condition | valid_n | spearman_rho |
| --- | --- | --- | --- | --- | --- |
| 0.0 | cheek_mean_log2_y | raw | relative_optical_exposure | 100.0 | -0.424117 |
| 0.0 | cheek_mean_log2_y | raw | log2_iso_condition | 100.0 | 0.324181 |
| 0.0 | cheek_mean_log2_y | calibrated_nn | relative_optical_exposure | 100.0 | -0.039036 |
| 0.0 | cheek_mean_log2_y | calibrated_nn | log2_iso_condition | 100.0 | 0.122797 |
| 0.0 | cheek_mean_log2_rg | raw | relative_optical_exposure | 100.0 | 0.208989 |
| 0.0 | cheek_mean_log2_rg | raw | log2_iso_condition | 100.0 | -0.172685 |
| 0.0 | cheek_mean_log2_rg | calibrated_nn | relative_optical_exposure | 100.0 | 0.018869 |
| 0.0 | cheek_mean_log2_rg | calibrated_nn | log2_iso_condition | 100.0 | 0.023338 |
| 0.0 | cheek_mean_log2_bg | raw | relative_optical_exposure | 100.0 | 0.041911 |
| 0.0 | cheek_mean_log2_bg | raw | log2_iso_condition | 100.0 | -0.175369 |
| 0.0 | cheek_mean_log2_bg | calibrated_nn | relative_optical_exposure | 100.0 | 0.058685 |
| 0.0 | cheek_mean_log2_bg | calibrated_nn | log2_iso_condition | 100.0 | -0.006304 |
| 0.0 | forehead_minus_cheek_log2_y | raw | relative_optical_exposure | 96.0 | 0.102432 |
| 0.0 | forehead_minus_cheek_log2_y | raw | log2_iso_condition | 96.0 | -0.37653 |
| 0.0 | forehead_minus_cheek_log2_y | calibrated_nn | relative_optical_exposure | 96.0 | 0.020433 |
| 0.0 | forehead_minus_cheek_log2_y | calibrated_nn | log2_iso_condition | 96.0 | -0.317852 |
| 0.0 | forehead_minus_cheek_log2_rg | raw | relative_optical_exposure | 96.0 | -0.190705 |
| 0.0 | forehead_minus_cheek_log2_rg | raw | log2_iso_condition | 96.0 | 0.285296 |
| 0.0 | forehead_minus_cheek_log2_rg | calibrated_nn | relative_optical_exposure | 96.0 | -0.042401 |
| 0.0 | forehead_minus_cheek_log2_rg | calibrated_nn | log2_iso_condition | 96.0 | 0.101021 |
| 0.0 | forehead_minus_cheek_log2_bg | raw | relative_optical_exposure | 96.0 | -0.043847 |
| 0.0 | forehead_minus_cheek_log2_bg | raw | log2_iso_condition | 96.0 | 0.078231 |
| 0.0 | forehead_minus_cheek_log2_bg | calibrated_nn | relative_optical_exposure | 96.0 | 0.206028 |
| 0.0 | forehead_minus_cheek_log2_bg | calibrated_nn | log2_iso_condition | 96.0 | -0.022186 |
| 1.0 | cheek_mean_log2_y | raw | relative_optical_exposure | 100.0 | -0.226103 |
| 1.0 | cheek_mean_log2_y | raw | log2_iso_condition | 100.0 | 0.300937 |
| 1.0 | cheek_mean_log2_y | calibrated_nn | relative_optical_exposure | 100.0 | 0.282707 |
| 1.0 | cheek_mean_log2_y | calibrated_nn | log2_iso_condition | 100.0 | -0.015196 |
| 1.0 | cheek_mean_log2_rg | raw | relative_optical_exposure | 100.0 | -0.046012 |
| 1.0 | cheek_mean_log2_rg | raw | log2_iso_condition | 100.0 | -0.194893 |
| 1.0 | cheek_mean_log2_rg | calibrated_nn | relative_optical_exposure | 100.0 | -0.319963 |
| 1.0 | cheek_mean_log2_rg | calibrated_nn | log2_iso_condition | 100.0 | 0.004841 |
| 1.0 | cheek_mean_log2_bg | raw | relative_optical_exposure | 100.0 | 0.088883 |
| 1.0 | cheek_mean_log2_bg | raw | log2_iso_condition | 100.0 | -0.260628 |
| 1.0 | cheek_mean_log2_bg | calibrated_nn | relative_optical_exposure | 100.0 | 0.158527 |
| 1.0 | cheek_mean_log2_bg | calibrated_nn | log2_iso_condition | 100.0 | -0.045403 |
| 1.0 | forehead_minus_cheek_log2_y | raw | relative_optical_exposure | 97.0 | 0.117847 |
| 1.0 | forehead_minus_cheek_log2_y | raw | log2_iso_condition | 97.0 | -0.081502 |
| 1.0 | forehead_minus_cheek_log2_y | calibrated_nn | relative_optical_exposure | 97.0 | 0.120147 |
| 1.0 | forehead_minus_cheek_log2_y | calibrated_nn | log2_iso_condition | 97.0 | 0.131711 |

## 20. 校准前后设备差异

完整raw、Ridge和MLP设备差异见`raw_ridge_mlp_camera_differences.csv`；设备差异降低不自动代表方法更优。

## 21. 方差保留

30个target/fold中，Stage 2B方差保留低于Stage 2A的有18个；方差降低可能代表删除个体变化，不定义单向优劣。

## 22. 条件范围外推病例

| fold | outside_n | both_outside_n |
| --- | --- | --- |
| 0.0 | 4.0 | 4.0 |
| 1.0 | 2.0 | 0.0 |
| 2.0 | 0.0 | 0.0 |
| 3.0 | 0.0 | 0.0 |
| 4.0 | 0.0 | 0.0 |

fold 4范围外病例=0；六目标绝对标准化设备均值差的fold内平均为Stage 2A=0.212567、Stage 2B=0.218765（B−A=0.00619777）。因此fold 4并非一致改善，报告设备差异时需同时参考这一外推负担；未据此重训、clamp或调参。

## 23. Stage 2A与Stage 2B逐折比较

| fold | target | stage2a_value | stage2b_value | delta_b_minus_a | stage2b_better | stage2a_better |
| --- | --- | --- | --- | --- | --- | --- |
| 0.0 | cheek_mean_log2_bg | 0.141741 | 0.138653 | -0.003089 | 1.0 | 0.0 |
| 1.0 | cheek_mean_log2_bg | 0.124988 | 0.123408 | -0.00158 | 1.0 | 0.0 |
| 2.0 | cheek_mean_log2_bg | 0.133842 | 0.129992 | -0.00385 | 1.0 | 0.0 |
| 3.0 | cheek_mean_log2_bg | 0.135408 | 0.133579 | -0.001828 | 1.0 | 0.0 |
| 4.0 | cheek_mean_log2_bg | 0.140402 | 0.143289 | 0.002887 | 0.0 | 1.0 |
| 0.0 | cheek_mean_log2_rg | 0.130847 | 0.132218 | 0.001371 | 0.0 | 1.0 |
| 1.0 | cheek_mean_log2_rg | 0.136769 | 0.135411 | -0.001358 | 1.0 | 0.0 |
| 2.0 | cheek_mean_log2_rg | 0.146617 | 0.144487 | -0.00213 | 1.0 | 0.0 |
| 3.0 | cheek_mean_log2_rg | 0.135956 | 0.13177 | -0.004187 | 1.0 | 0.0 |
| 4.0 | cheek_mean_log2_rg | 0.143854 | 0.141717 | -0.002137 | 1.0 | 0.0 |
| 0.0 | cheek_mean_log2_y | 0.363679 | 0.353356 | -0.010323 | 1.0 | 0.0 |
| 1.0 | cheek_mean_log2_y | 0.33168 | 0.326623 | -0.005057 | 1.0 | 0.0 |
| 2.0 | cheek_mean_log2_y | 0.306147 | 0.300523 | -0.005624 | 1.0 | 0.0 |
| 3.0 | cheek_mean_log2_y | 0.317274 | 0.321969 | 0.004695 | 0.0 | 1.0 |
| 4.0 | cheek_mean_log2_y | 0.439813 | 0.424562 | -0.015251 | 1.0 | 0.0 |
| 0.0 | forehead_minus_cheek_log2_bg | 0.099984 | 0.100601 | 0.000617 | 0.0 | 1.0 |
| 1.0 | forehead_minus_cheek_log2_bg | 0.117371 | 0.119237 | 0.001867 | 0.0 | 1.0 |
| 2.0 | forehead_minus_cheek_log2_bg | 0.118463 | 0.11777 | -0.000692 | 1.0 | 0.0 |
| 3.0 | forehead_minus_cheek_log2_bg | 0.113213 | 0.113237 | 2.4e-05 | 0.0 | 1.0 |
| 4.0 | forehead_minus_cheek_log2_bg | 0.110382 | 0.109879 | -0.000503 | 1.0 | 0.0 |
| 0.0 | forehead_minus_cheek_log2_rg | 0.11327 | 0.113764 | 0.000494 | 0.0 | 1.0 |
| 1.0 | forehead_minus_cheek_log2_rg | 0.114917 | 0.114603 | -0.000314 | 1.0 | 0.0 |
| 2.0 | forehead_minus_cheek_log2_rg | 0.101987 | 0.100453 | -0.001534 | 1.0 | 0.0 |
| 3.0 | forehead_minus_cheek_log2_rg | 0.109577 | 0.110515 | 0.000938 | 0.0 | 1.0 |
| 4.0 | forehead_minus_cheek_log2_rg | 0.121537 | 0.121601 | 6.5e-05 | 0.0 | 1.0 |
| 0.0 | forehead_minus_cheek_log2_y | 0.296218 | 0.294796 | -0.001422 | 1.0 | 0.0 |
| 1.0 | forehead_minus_cheek_log2_y | 0.353543 | 0.354766 | 0.001223 | 0.0 | 1.0 |
| 2.0 | forehead_minus_cheek_log2_y | 0.271757 | 0.276921 | 0.005165 | 0.0 | 1.0 |
| 3.0 | forehead_minus_cheek_log2_y | 0.383495 | 0.390222 | 0.006726 | 0.0 | 1.0 |
| 4.0 | forehead_minus_cheek_log2_y | 0.335974 | 0.336212 | 0.000238 | 0.0 | 1.0 |

## 24. Stage 2B优于Stage 2A的fold数量

所有有方向指标的target/fold条目中，B优于A=125，A优于B=115。条件预测RMSE平均B−A=-0.00115233；R²平均B−A=0.00782801。逐目标、逐指标fold胜负如下；结果存在fold间不一致，不只报告B占优结果。

| target | metric_name | stage2b_better_fold_n | stage2a_better_fold_n | tie_fold_n | mean_delta_b_minus_a |
| --- | --- | --- | --- | --- | --- |
| cheek_mean_log2_bg | mae | 4.0 | 1.0 | 0.0 | -0.001632 |
| cheek_mean_log2_bg | r2 | 4.0 | 1.0 | 0.0 | 0.021608 |
| cheek_mean_log2_bg | rmse | 4.0 | 1.0 | 0.0 | -0.001492 |
| cheek_mean_log2_rg | mae | 4.0 | 1.0 | 0.0 | -0.002056 |
| cheek_mean_log2_rg | r2 | 4.0 | 1.0 | 0.0 | 0.02095 |
| cheek_mean_log2_rg | rmse | 4.0 | 1.0 | 0.0 | -0.001688 |
| cheek_mean_log2_y | mae | 4.0 | 1.0 | 0.0 | -0.003966 |
| cheek_mean_log2_y | r2 | 4.0 | 1.0 | 0.0 | 0.021926 |
| cheek_mean_log2_y | rmse | 4.0 | 1.0 | 0.0 | -0.006312 |
| forehead_minus_cheek_log2_bg | mae | 1.0 | 4.0 | 0.0 | 0.000293 |
| forehead_minus_cheek_log2_bg | r2 | 2.0 | 3.0 | 0.0 | -0.004296 |
| forehead_minus_cheek_log2_bg | rmse | 2.0 | 3.0 | 0.0 | 0.000262 |
| forehead_minus_cheek_log2_rg | mae | 3.0 | 2.0 | 0.0 | -0.0002 |
| forehead_minus_cheek_log2_rg | r2 | 2.0 | 3.0 | 0.0 | 0.001552 |
| forehead_minus_cheek_log2_rg | rmse | 2.0 | 3.0 | 0.0 | -7e-05 |
| forehead_minus_cheek_log2_y | mae | 2.0 | 3.0 | 0.0 | 0.001478 |
| forehead_minus_cheek_log2_y | r2 | 1.0 | 4.0 | 0.0 | -0.014772 |
| forehead_minus_cheek_log2_y | rmse | 1.0 | 4.0 | 0.0 | 0.002386 |

## 25. OOF完整性

OOF=500行、唯一ID=500；额部可用=486、不可用=14；非有限cheek=0；非法额部输出=0。

## 26. 单元和协议测试

单元测试=`PASS`；协议测试=`PASS`。

## 27. 确定性验证

在临时目录重复完整五折核心流程，fold/summary的CSV、JSON和PTH SHA256一致：`True`。

## 28. 历史输入未修改声明

第一阶段、Stage 2A、十份固定split运行前后SHA256一致；`historical_inputs_modified=false`。

## 29. 局限性

1. 小型MLP可能捕获非线性，也可能过拟合；本次10个网络中外层val MSE高于完整train MSE的有7个。
2. 两设备与有限条件范围不支持未知设备或广泛外推；范围外病例未删除或截断。
3. 若A/B接近，应优先保留更简单、可解释的Stage 2A。
4. 是否进入NYHA模型必须等待后续Raw、A、B和Global融合对照；本任务不提前决定。
5. 输出不是皮肤真实反射率、传感器线性RGB、生理参数或强意义物理反演。

## 30. 后续可用于NYHA实验的六个Stage 2B字段

- `calibrated_nn_cheek_mean_log2_y`
- `calibrated_nn_cheek_mean_log2_rg`
- `calibrated_nn_cheek_mean_log2_bg`
- `calibrated_nn_forehead_minus_cheek_log2_y`
- `calibrated_nn_forehead_minus_cheek_log2_rg`
- `calibrated_nn_forehead_minus_cheek_log2_bg`
