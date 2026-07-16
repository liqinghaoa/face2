# 四个核心EXIF参数二阶段可用性审计

> 本报告是数据审计与采集捷径诊断，不代表EXIF具有临床预测能力。没有修改原始图像、标签、固定split、历史实验或正式训练代码，也没有启动人脸图像深度学习训练。

## 结论摘要

- 全EXIF队列：522张、522个唯一ID、505个patient_group_id。
- 当前正式Global/meanbg普通五折队列：500张、500个唯一ID、483个patient_group_id；依据主配置实际指向的splits_500，而非事后筛选的S2 425队列。
- 四个核心字段在522张中均为100%非空、100%基础合法；两个camera_id在正式队列中的计数见下表。
- EXIF-only最佳诊断OOF Macro-AUC=0.714（random_forest_shallow / E1_four_core_transformed；patient_group bootstrap 95% CI 0.687–0.739）。
- 设备/NYHA采集捷径等级：中等。存在需要严肃控制的设备/采集捷径，但尚不足以否定受约束的物理校正路线。
- 最终V1连续EXIF向量：relative_optical_exposure + log2_iso_gain + device_centered_brightness（设备内中位数/IQR只从每折训练部分拟合）。
- camera_id数据流：只允许作为前向renderer条件及亮度校准索引，不进入表型/NYHA分类编码器。支持继续条件光学反演，但必须禁止camera_id/raw EXIF进入NYHA分类分支并做设备敏感性分析。

## 1. 队列与数据对齐

| check | status | detail |
| --- | --- | --- |
| exif_unique_id | PASS | rows=522, unique=522 |
| full_split_unique_id | PASS | rows=522, unique=522 |
| formal_split_unique_id | PASS | rows=500, unique=500 |
| exif_equals_full_split | PASS | exif_only=0, split_only=0 |
| formal_subset_of_full | PASS | formal_not_full=0 |
| formal_has_exif | PASS | formal_without_exif=0 |
| formal_metadata_agreement | PASS | patient_group_id, NYHA, SEX and label_3class compared for all formal IDs |
| full_patient_group_no_cross_fold | PASS | leaking_groups=[] |
| full_nyha_mapping | PASS | mismatch_ids=[] |
| full_group_sex_consistency | PASS | conflicting_groups=[] |
| formal_patient_group_no_cross_fold | PASS | leaking_groups=[] |
| formal_nyha_mapping | PASS | mismatch_ids=[] |
| formal_group_sex_consistency | PASS | conflicting_groups=[] |
| formal_config_points_to_splits_500 | PASS | E:\projects\face2\config\train\preprocess_ablation_resnet18\nyha_3class_resnet18_preproc_hybrid_imagenet_meanbg.yaml |
| formal_config_points_to_meanbg | PASS | E:\projects\face2\data\processed\global_face\preprocess_ablation\hybrid_imagenet_meanbg\images |
| formal_meanbg_images_available | PASS | missing_n=0; IDs=[] |
| aligned_rgb_available_for_full_queue | PASS | missing_n=0; IDs=[] |
| parsing_label_available_for_full_queue | PASS | missing_n=0; IDs=[] |
| core_exif_values_complete_and_positive | PASS | valid_n=522/522; invalid_IDs=[] |
| no_conflicting_duplicate_core_or_aux_tags | PASS | conflicts=[] |

同患者跨fold：全522队列与正式500队列均未发现。patient_group级连续参数统一取组内中位数。由于NYHA可随同一患者不同照片变化，全队列有10组、正式队列有10组无法安全赋予单一NYHA类别；这些组保留在图像级分析和固定fold OOF中，但不静默压成单一patient_group级NYHA。设备不一致的多图组分别为全队列2组、正式队列2组，设备分层patient_group统计不使用这些歧义组。

正式队列camera_id计数：

| camera_id | n |
| --- | --- |
| HONOR/BVL-AN00 | 267 |
| Xiaomi/M2006J10C | 233 |

## 2. 四字段基础统计

| variable | n | missing_n | invalid_n | min | p5 | median | p95 | max | mean | std | iqr | mad |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ExposureTime | 522 | 0 | 0 | 0.00222222 | 0.00834147 | 0.02 | 0.04 | 0.050009 | 0.0200609 | 0.00916974 | 0.020006 | 0.01 |
| FNumber | 522 | 0 | 0 | 1.89 | 1.89 | 1.9 | 1.9 | 2 | 1.89772 | 0.0164619 | 0.01 | 0 |
| ISOSpeedRatings | 522 | 0 | 0 | 50 | 75.05 | 241.5 | 800 | 1600 | 313.511 | 241.516 | 240 | 105.5 |
| BrightnessValue | 522 | 0 | 0 | -2.67 | -1.43 | 3.055 | 9.09 | 10.9 | 3.71044 | 3.71294 | 6.8475 | 3.475 |

ExposureTime、FNumber和ISOSpeedRatings同时提供原始尺度与log2尺度统计；完整统计文件还包括图像级、patient_group中位数级、分设备、分NYHA、分SEX及camera_id×NYHA结果。离群值仅标记，没有删除。

FNumber设备内变异：

| camera_id | n | unique_n | min | max | std |
| --- | --- | --- | --- | --- | --- |
| HONOR/BVL-AN00 | 267 | 2 | 1.9 | 2 | 0.020757 |
| Xiaomi/M2006J10C | 233 | 1 | 1.89 | 1.89 | 0 |

判断：FNumber在设备内几乎不变，主要编码镜头/设备；仍有明确光学意义，因此只作为relative_optical_exposure的组成，不作为独立学习特征。

## 3. 派生物理量与APEX一致性

| scope | camera_id | n | pearson_r | spearman_rho | slope | intercept | r_squared | residual_median | residual_mad |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| overall | ALL | 500 | 0.352418 | 0.475882 | 0.938091 | 2.48175 | 0.124199 | -0.727088 | 0.48284 |
| within_camera | HONOR/BVL-AN00 | 267 | 0.975285 | 0.988598 | 1.01494 | -0.87157 | 0.951182 | -0.893927 | 0.09 |
| within_camera | Xiaomi/M2006J10C | 233 | 0.938833 | 0.933982 | 0.948105 | 6.18611 | 0.881408 | 6.17016 | 0.126748 |

APEX偏差只用于元数据一致性和设备差异审计。手机ISP、自动曝光和厂商实现会造成截距/斜率差异，不能把残差自动当作源数据错误。BrightnessValue跨设备直接合并会引入系统偏移，V1只使用训练折内按camera_id中位数/IQR稳健中心化后的值。

## 4. 设备身份编码

| feature | hedges_g_b_minus_a | mutual_information | oof_roc_auc | oof_balanced_accuracy |
| --- | --- | --- | --- | --- |
| FNumber | -0.954013 | 0.691834 | 1 | 1 |
| log2_fnumber | -0.97297 | 0.691834 | 1 | 1 |
| aperture_value_from_f | -0.97297 | 0.691834 | 1 | 1 |
| brightness_residual | 17.5724 | 0.688732 | 1 | 0.998127 |
| BrightnessValue | 4.76513 | 0.668562 | 0.999646 | 0.988218 |
| EV100 | -1.0821 | 0.669824 | 0.847937 | 0.713089 |
| relative_optical_exposure | 1.0821 | 0.669824 | 0.847937 | 0.713089 |
| ExposureTime | 1.0069 | 0.655685 | 0.776422 | 0.651742 |
| log2_exposure_time | 1.03951 | 0.655721 | 0.773947 | 0.709344 |
| time_value_from_t | -1.03951 | 0.655721 | 0.773947 | 0.709344 |
| ISOSpeedRatings | -0.725979 | 0.545835 | 0.69933 | 0.675749 |
| log2_iso_gain | -0.675498 | 0.543614 | 0.698526 | 0.672936 |

四字段及派生量联合预测camera_id的固定fold OOF ROC-AUC=1.000、Balanced Accuracy=1.000、Macro-F1=1.000。这说明变量中的设备信息不可忽略；该结果只用于判断采集依赖。

MeteringMode与Flash未进入任何核心或标签模型。辅助审计显示它们与camera_id的关联及对核心字段方差的解释如下：

| auxiliary_field | analysis | cramers_v | chi_square_p | unique_n | core_field | r_squared |
| --- | --- | --- | --- | --- | --- | --- |
| MeteringMode | camera_association | 0.169763 | 0.000147037 | 2 |  |  |
| Flash | camera_association | 0.25642 | 9.82445e-09 | 2 |  |  |
| MeteringMode+Flash | auxiliary_only |  |  |  | ExposureTime | 0.0192544 |
| MeteringMode+Flash | camera_only |  |  |  | ExposureTime | 0.202595 |
| MeteringMode+Flash | auxiliary_plus_camera |  |  |  | ExposureTime | 0.204088 |
| MeteringMode+Flash | auxiliary_only |  |  |  | FNumber | 0.00207486 |
| MeteringMode+Flash | camera_only |  |  |  | FNumber | 0.185721 |
| MeteringMode+Flash | auxiliary_plus_camera |  |  |  | FNumber | 0.194836 |
| MeteringMode+Flash | auxiliary_only |  |  |  | ISOSpeedRatings | 0.0186441 |
| MeteringMode+Flash | camera_only |  |  |  | ISOSpeedRatings | 0.116667 |
| MeteringMode+Flash | auxiliary_plus_camera |  |  |  | ISOSpeedRatings | 0.117597 |
| MeteringMode+Flash | auxiliary_only |  |  |  | BrightnessValue | 0.0861 |
| MeteringMode+Flash | camera_only |  |  |  | BrightnessValue | 0.850528 |
| MeteringMode+Flash | auxiliary_plus_camera |  |  |  | BrightnessValue | 0.850754 |

## 5. NYHA与SEX混杂

正式队列camera_id×NYHA列联的Cramér's V=0.587，卡方p=3.64e-38。总体显著但设备内不显著的候选字段：BrightnessValue, EV100, ExposureTime, FNumber, ISOSpeedRatings, aperture_value_from_f, brightness_residual, log2_exposure_time, log2_fnumber, log2_iso_gain, relative_optical_exposure, sensitivity_value, time_value_from_t。这些候选只能解释为可能的设备混杂，不能解释为心功能相关拍摄差异。

总体NYHA检验（按FDR排序）：

| variable | effect_size | distribution_overlap_min | p_value | p_fdr_bh |
| --- | --- | --- | --- | --- |
| FNumber | 0.331788 | 0.27027 | 5.73388e-37 | 3.05807e-36 |
| log2_fnumber | 0.331788 | 0.27027 | 5.73388e-37 | 3.05807e-36 |
| aperture_value_from_f | 0.331788 | 0.27027 | 5.73388e-37 | 3.05807e-36 |
| BrightnessValue | 0.282296 | 0.283784 | 1.25795e-31 | 5.03178e-31 |
| brightness_residual | 0.253068 | 0.27027 | 1.79498e-28 | 5.74393e-28 |
| EV100 | 0.121674 | 0.676733 | 2.7186e-14 | 6.21395e-14 |
| relative_optical_exposure | 0.121674 | 0.676733 | 2.7186e-14 | 6.21395e-14 |
| ExposureTime | 0.102781 | 0.68349 | 2.97438e-12 | 4.75901e-12 |
| log2_exposure_time | 0.102781 | 0.676733 | 2.97438e-12 | 4.75901e-12 |
| time_value_from_t | 0.102781 | 0.676733 | 2.97438e-12 | 4.75901e-12 |
| ISOSpeedRatings | 0.0639665 | 0.643552 | 4.59522e-08 | 5.65565e-08 |
| log2_iso_gain | 0.0639665 | 0.673913 | 4.59522e-08 | 5.65565e-08 |
| sensitivity_value | 0.0639665 | 0.673913 | 4.59522e-08 | 5.65565e-08 |
| device_centered_brightness | 0.00352025 | 0.787415 | 0.153388 | 0.175301 |
| combined_exposure_gain | 0.00242241 | 0.774647 | 0.201499 | 0.201499 |
| brightness_apex_pred | 0.00242408 | 0.774647 | 0.201416 | 0.201499 |

设备内NYHA检验（按FDR排序，前20行）：

| camera_id | variable | effect_size | distribution_overlap_min | p_value | p_fdr_bh |
| --- | --- | --- | --- | --- | --- |
| Xiaomi/M2006J10C | ExposureTime | 0.0102321 | 0.822826 | 0.113416 | 0.167994 |
| Xiaomi/M2006J10C | ISOSpeedRatings | 0.00909731 | 0.661957 | 0.129226 | 0.167994 |
| Xiaomi/M2006J10C | log2_exposure_time | 0.0102321 | 0.822826 | 0.113416 | 0.167994 |
| Xiaomi/M2006J10C | log2_iso_gain | 0.00909731 | 0.707609 | 0.129226 | 0.167994 |
| Xiaomi/M2006J10C | time_value_from_t | 0.0102321 | 0.822826 | 0.113416 | 0.167994 |
| Xiaomi/M2006J10C | EV100 | 0.0102321 | 0.822826 | 0.113416 | 0.167994 |
| Xiaomi/M2006J10C | relative_optical_exposure | 0.0102321 | 0.822826 | 0.113416 | 0.167994 |
| Xiaomi/M2006J10C | combined_exposure_gain | 0.0120632 | 0.715217 | 0.0918805 | 0.167994 |
| Xiaomi/M2006J10C | sensitivity_value | 0.00909731 | 0.707609 | 0.129226 | 0.167994 |
| Xiaomi/M2006J10C | brightness_apex_pred | 0.0120632 | 0.715217 | 0.0918805 | 0.167994 |
| Xiaomi/M2006J10C | BrightnessValue | 0.00489718 | 0.697826 | 0.20947 | 0.212321 |
| Xiaomi/M2006J10C | brightness_residual | 0.00477962 | 0.688043 | 0.212321 | 0.212321 |
| Xiaomi/M2006J10C | device_centered_brightness | 0.00489718 | 0.697826 | 0.20947 | 0.212321 |
| HONOR/BVL-AN00 | ExposureTime |  | 0.878057 |  |  |
| HONOR/BVL-AN00 | FNumber |  | 0.940077 |  |  |
| HONOR/BVL-AN00 | ISOSpeedRatings |  | 0.841195 |  |  |
| HONOR/BVL-AN00 | BrightnessValue |  | 0.804333 |  |  |
| HONOR/BVL-AN00 | log2_exposure_time |  | 0.85884 |  |  |
| HONOR/BVL-AN00 | log2_iso_gain |  | 0.810971 |  |  |
| HONOR/BVL-AN00 | log2_fnumber |  | 0.940077 |  |  |

所有p值均同时报告效应量与经验分布重叠。p<0.05不被自动解释为临床或物理意义；三分类有序Spearman仅为探索性统计。完整SEX、设备内及设备调整后结果见core_parameter_confounding_tests.csv。

## 6. 与实际图像表现的关系

使用现有224×224颜色保持aligned_rgb及既有CelebAMask-HQ解析标签中的skin类计算；未重新训练或生成分割网络。颜色/亮度统计没有使用meanbg图、black-background成图或ImageNet标准化张量。

图像质量计算失败：0例。主要Spearman相关如下：

| exif_field | image_metric | n | spearman_rho | spearman_p |
| --- | --- | --- | --- | --- |
| FNumber | skin_linear_luminance_median | 500 | 0.612589 | 7.81125e-53 |
| BrightnessValue | skin_linear_luminance_median | 500 | -0.539887 | 3.60323e-39 |
| BrightnessValue | skin_lab_l_median | 500 | -0.539887 | 3.60323e-39 |
| ExposureTime | skin_linear_luminance_median | 500 | -0.38858 | 1.8098e-19 |
| BrightnessValue | underexposed_pixel_ratio | 500 | 0.269795 | 8.69139e-10 |
| ISOSpeedRatings | skin_linear_luminance_median | 500 | 0.231667 | 1.61718e-07 |
| FNumber | laplacian_variance | 500 | -0.186388 | 2.73909e-05 |
| ExposureTime | high_frequency_noise_mad | 500 | -0.154058 | 0.000546476 |
| ISOSpeedRatings | underexposed_pixel_ratio | 500 | -0.10815 | 0.0155478 |
| ISOSpeedRatings | high_frequency_noise_mad | 500 | -0.0979802 | 0.028474 |
| BrightnessValue | overexposed_pixel_ratio | 500 | 0.0741076 | 0.097879 |
| ExposureTime | laplacian_variance | 500 | -0.0541093 | 0.227133 |

质量指标定义：skin区域中位RGB；sRGB反伽马后的相对线性亮度；由线性Y换算的Lab L*；线性亮度≥0.98为过曝、≤0.02为欠曝；任一通道≤5或≥250为饱和/裁剪；Laplacian方差为清晰度代理；高斯平滑残差MAD为简化高频噪声代理。它们是相对质量指标，不是传感器标定真值。

已知14张设备内高ISO离群图片均保留。逐例过曝、欠曝、噪声、模糊和元数据不一致标记见known_high_iso_outliers.csv；判定使用固定阈值和设备内稳健z，不把统计标记等同于源图错误。

| ID | patient_group_id | camera_id | label_3class_name | SEX | fold_for_current_formal_if_available | ISOSpeedRatings | obvious_overexposure | obvious_underexposure | high_noise_within_camera | obvious_blur_within_camera | metadata_inconsistency |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 201331515 | 201331515 | HONOR/BVL-AN00 | severe | 1 | 1 | 1600 | False | False | False | False | False |
| 204650307 | 204650307 | HONOR/BVL-AN00 | mild | 0 | 2 | 1250 | False | False | False | False | False |
| A000104693 | A000104693 | HONOR/BVL-AN00 | mild | 0 | 3 | 1250 | False | False | False | False | False |
| A000590309 | A000590309 | HONOR/BVL-AN00 | mild | 1 | 0 | 1250 | False | False | False | False | False |
| A001074268 | A001074268 | HONOR/BVL-AN00 | mild | 1 | 4 | 1250 | False | False | False | False | False |
| A001808459 | A001808459 | HONOR/BVL-AN00 | mild | 1 | 3 | 1250 | False | False | False | False | False |
| A001837056 | A001837056 | HONOR/BVL-AN00 | mild | 0 | 3 | 1250 | False | False | False | False | False |
| A001888679 | A001888679 | HONOR/BVL-AN00 | mild | 0 | 0 | 1250 | False | False | False | False | False |
| A002331084 | A002331084 | HONOR/BVL-AN00 | mild | 1 | 1 | 1250 | False | False | False | False | False |
| A001636890 | A001636890 | Xiaomi/M2006J10C | normal | 0 | 0 | 1115 | False | False | False | False | True |
| A001632964 | A001632964 | Xiaomi/M2006J10C | mild | 0 | 0 | 1082 | False | False | False | False | True |
| A001028520 | A001028520 | Xiaomi/M2006J10C | mild | 0 | 1 | 871 | False | False | False | False | True |
| A001722742 | A001722742 | Xiaomi/M2006J10C | mild | 0 | 1 | 710 | False | False | False | False | True |
| A001438120 | A001438120 | Xiaomi/M2006J10C | severe | 1 | 2 | 700 | False | False | False | False | True |

## 7. EXIF-only固定五折标签捷径诊断

Logistic E3 OOF Macro-AUC=0.671（95% CI 0.645–0.698）；camera-only E4=0.688；E3+camera E5=0.700。所有标准化、填补、camera编码和设备内亮度中位数/IQR均只在各折训练部分拟合。

OOF文件为每个固定特征组和两种预设模型保留完整三分类概率。每个ID在每个模型/特征组中恰好出现一次、概率有限且行和为1；同患者不跨fold。模型和特征未根据结果调参。

## 8. 共线性与字段压缩

ExposureTime、FNumber、ISO及派生曝光量包含确定性或近确定性关系，VIF出现无穷大属于公式冗余的预期结果，不机械按阈值删字段。V1选择relative_optical_exposure + log2_iso_gain + device_centered_brightness，避免同时输入原始四字段、EV100及combined_exposure_gain。

## 9. 最终逐字段决策表

| 字段 | 完整性 | 设备内变异 | 设备依赖 | APEX一致性 | 图像表现相关性 | NYHA混杂 | 推荐角色 | 结论依据 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ExposureTime | 100.0% (522/522) | HONOR/BVL-AN00: unique=28, std=0.008177, IQR=0.01; Xiaomi/M2006J10C: unique=8, std=0.008048, IQR=0.01 | single-field camera OOF AUC=0.776; MI=0.656 | valid component of APEX construction; cross-device residual checked separately | strongest \|rho\|: skin_linear_luminance_median, rho=-0.389 | raw KW epsilon2=0.103, FDR p=4.76e-12, minimum overlap=0.683 | derived_only | 保留物理信息，但V1中通过relative_optical_exposure表达，避免与FNumber重复。 |
| FNumber | 100.0% (522/522) | HONOR/BVL-AN00: unique=2, std=0.02076, IQR=0; Xiaomi/M2006J10C: unique=1, std=9.346e-15, IQR=0 | single-field camera OOF AUC=1.000; MI=0.692 | valid component of APEX construction; cross-device residual checked separately | strongest \|rho\|: skin_linear_luminance_median, rho=0.613 | raw KW epsilon2=0.332, FDR p=3.06e-36, minimum overlap=0.27 | derived_only | 设备内变异极低且设备依赖强；只作为relative_optical_exposure组成，不单独学习。 |
| ISOSpeedRatings | 100.0% (522/522) | HONOR/BVL-AN00: unique=16, std=281.1, IQR=300; Xiaomi/M2006J10C: unique=165, std=142.9, IQR=129 | single-field camera OOF AUC=0.699; MI=0.546 | valid component of APEX construction; cross-device residual checked separately | strongest \|rho\|: skin_linear_luminance_median, rho=0.232 | raw KW epsilon2=0.064, FDR p=5.66e-08, minimum overlap=0.644 | derived_only | 采用log2_iso_gain表达；高ISO保留并作为噪声/质量敏感性标记。 |
| BrightnessValue | 100.0% (522/522) | HONOR/BVL-AN00: unique=214, std=1.588, IQR=2.17; Xiaomi/M2006J10C: unique=56, std=1.267, IQR=1.7 | single-field camera OOF AUC=1.000; MI=0.669 | overall recorded-vs-predicted slope=0.938, R2=0.124; device-specific fits required | strongest \|rho\|: skin_linear_luminance_median, rho=-0.540 | raw KW epsilon2=0.282, FDR p=5.03e-31, minimum overlap=0.284 | derived_only | 原值不宜跨设备直接使用；V1仅使用训练折设备内稳健中心化值。 |
| relative_optical_exposure | 100.0% (522/522) | HONOR/BVL-AN00: unique=30, std=0.7602, IQR=1; Xiaomi/M2006J10C: unique=8, std=0.5225, IQR=0.585 | single-field camera OOF AUC=0.848; MI=0.670 | overall recorded-vs-predicted slope=0.938, R2=0.124; device-specific fits required | component-derived; no independent image-metric test | raw KW epsilon2=0.122, FDR p=6.21e-14, minimum overlap=0.677 | core_continuous_condition | V1主要光学进光连续条件。 |
| log2_iso_gain | 100.0% (522/522) | HONOR/BVL-AN00: unique=16, std=1.094, IQR=1.322; Xiaomi/M2006J10C: unique=165, std=0.8044, IQR=0.9676 | single-field camera OOF AUC=0.699; MI=0.544 | overall recorded-vs-predicted slope=0.938, R2=0.124; device-specific fits required | component-derived; no independent image-metric test | raw KW epsilon2=0.064, FDR p=5.66e-08, minimum overlap=0.674 | core_continuous_condition | V1传感器增益连续条件。 |
| combined_exposure_gain | 100.0% (522/522) | HONOR/BVL-AN00: unique=64, std=1.526, IQR=2.322; Xiaomi/M2006J10C: unique=188, std=1.254, IQR=1.596 | single-field camera OOF AUC=0.488; MI=0.343 | overall recorded-vs-predicted slope=0.938, R2=0.124; device-specific fits required | component-derived; no independent image-metric test | raw KW epsilon2=0.00242, FDR p=0.201, minimum overlap=0.775 | quality_control_only | 与relative_optical_exposure及ISO确定性冗余，不同时进入V1。 |
| device_centered_brightness | 100.0% (522/522) | HONOR/BVL-AN00: unique=214, std=0.7318, IQR=1; Xiaomi/M2006J10C: unique=56, std=0.7452, IQR=1 | single-field camera OOF AUC=0.517; MI=0.291 | not applicable | component-derived; no independent image-metric test | raw KW epsilon2=0.00352, FDR p=0.175, minimum overlap=0.787 | core_continuous_condition | V1亮度条件；每折仅用训练数据的设备中位数/IQR计算。 |
| brightness_residual | 100.0% (522/522) | HONOR/BVL-AN00: unique=189, std=0.3516, IQR=0.1839; Xiaomi/M2006J10C: unique=212, std=0.4411, IQR=0.2544 | single-field camera OOF AUC=1.000; MI=0.689 | direct APEX consistency residual; quality-control signal | component-derived; no independent image-metric test | raw KW epsilon2=0.253, FDR p=5.74e-28, minimum overlap=0.27 | quality_control_only | 用于APEX/元数据一致性质量控制，不进入表型或标签分类分支。 |

## 10. 对八个问题的明确回答

1. ExposureTime不作为独立原始条件进入V1，只通过relative_optical_exposure使用。
2. FNumber设备内变异极低，本质上接近设备/镜头常量；仅保留其物理公式作用。
3. ISOSpeedRatings存在真实设备内变化；是否反映噪声以图像高频噪声相关为证据，但手机ISP降噪会削弱对应关系，因此采用log2_iso_gain并保留高ISO质量标记。
4. BrightnessValue不能跨设备直接使用，必须在每折训练数据内按设备稳健中心化/缩放。
5. camera_id不进入光学反演/表型/分类编码器，只进入前向renderer；它可以作为设备内亮度校准索引，但不作为可学习标签捷径输入。
6. V1连续EXIF向量为[relative_optical_exposure, log2_iso_gain, device_centered_brightness]。
7. combined_exposure_gain与brightness_residual只用于质量控制/敏感性；原始ExposureTime、FNumber、ISO、BrightnessValue只作为派生来源或审计输出。
8. 存在需要严肃控制的设备/采集捷径，但尚不足以否定受约束的物理校正路线。 支持继续条件光学反演，但必须禁止camera_id/raw EXIF进入NYHA分类分支并做设备敏感性分析。

## 11. 文件说明

所有CSV均为UTF-8-SIG。figures目录同时保存PNG和可编辑文字SVG。报告目录：E:\projects\face2\reports\exif_core_parameter_second_stage_audit
