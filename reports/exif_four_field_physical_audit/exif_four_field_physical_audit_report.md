# 四个核心EXIF字段物理可用性专项审计

> 本报告只分析EXIF成像意义、数值可靠性及其与既有皮肤区域图像表现的关系。未读取或使用NYHA、SEX、其他临床标签、split或模型预测；未进行交叉验证、分类建模或图像深度学习训练。

## 一、完成状态与数据来源

- 完成状态：COMPLETE；分析图像数：522；唯一ID：522。
- EXIF数值优先复用既有第一阶段审计的逐图长表、图像审计表和问题明细表，避免重复提取工作簿。
- project_inventory：项目中未找到该文件；已改用现有审计报告、逐图CSV和预处理代码完成数据血缘核对。
- 图像区域：既有CelebAMask-HQ解析标签中的skin类；图像为现有224×224颜色保持aligned_rgb。没有重新训练或运行分割网络。
- 没有在meanbg、黑背景成图或ImageNet mean/std标准化张量上计算颜色与亮度。

设备数量：

| camera_id | n |
| --- | --- |
| HONOR/BVL-AN00 | 283 |
| Xiaomi/M2006J10C | 239 |

## 二、四字段总体统计

| variable | valid_n | missing_n | unique_n | min | p1 | p5 | p25 | median | p75 | p95 | p99 | max | mean | std | iqr | mad |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ExposureTime | 522 | 0 | 37 | 0.00222222 | 0.00407304 | 0.00834147 | 0.01 | 0.02 | 0.030006 | 0.04 | 0.040008 | 0.050009 | 0.0200609 | 0.00916974 | 0.020006 | 0.01 |
| FNumber | 522 | 0 | 3 | 1.89 | 1.89 | 1.89 | 1.89 | 1.9 | 1.9 | 1.9 | 2 | 2 | 1.89772 | 0.0164619 | 0.01 | 0 |
| ISOSpeedRatings | 522 | 0 | 179 | 50 | 50 | 75.05 | 160 | 241.5 | 400 | 800 | 1250 | 1600 | 313.511 | 241.516 | 240 | 105.5 |
| BrightnessValue | 522 | 0 | 277 | -2.67 | -2.1053 | -1.43 | 0.3525 | 3.055 | 7.2 | 9.09 | 10.179 | 10.9 | 3.71044 | 3.71294 | 6.8475 | 3.475 |

ExposureTime共有37个唯一值，ISOSpeedRatings共有179个唯一值，均具有明显逐图变化。FNumber总体仅3个唯一值，需按设备判断。离群值只标记，未删除。

## 三、按设备统计与系统偏移

| camera_id | variable | valid_n | unique_n | min | p5 | median | p95 | max | mean | std | iqr | mad |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| HONOR/BVL-AN00 | ExposureTime | 283 | 29 | 0.00222222 | 0.00588235 | 0.0166667 | 0.030303 | 0.04 | 0.0161593 | 0.00807008 | 0.01 | 0.00666667 |
| Xiaomi/M2006J10C | ExposureTime | 239 | 8 | 0.008496 | 0.0100029 | 0.020004 | 0.040008 | 0.050009 | 0.0246808 | 0.00820772 | 0.010002 | 0.010002 |
| HONOR/BVL-AN00 | FNumber | 283 | 2 | 1.9 | 1.9 | 1.9 | 1.9 | 2 | 1.90424 | 0.0201863 | 0 | 0 |
| Xiaomi/M2006J10C | FNumber | 239 | 1 | 1.89 | 1.89 | 1.89 | 1.89 | 1.89 | 1.89 | 9.56796e-15 | 0 | 0 |
| HONOR/BVL-AN00 | ISOSpeedRatings | 283 | 16 | 50 | 65.6 | 320 | 1000 | 1600 | 384.392 | 277.694 | 300 | 120 |
| Xiaomi/M2006J10C | ISOSpeedRatings | 239 | 169 | 50 | 75.9 | 192 | 469.6 | 1115 | 229.582 | 152.546 | 135.5 | 63 |
| HONOR/BVL-AN00 | BrightnessValue | 283 | 220 | -2.67 | -1.746 | 0.56 | 3.359 | 5.43 | 0.573322 | 1.57529 | 2.17 | 1.04 |
| Xiaomi/M2006J10C | BrightnessValue | 239 | 57 | 3.8 | 5.3 | 7.4 | 9.52 | 10.9 | 7.4251 | 1.29577 | 1.7 | 0.8 |

每个核心字段的设备内唯一值数：

| 字段 | 设备内唯一值数 |
| --- | --- |
| ExposureTime | HONOR/BVL-AN00=29; Xiaomi/M2006J10C=8 |
| FNumber | HONOR/BVL-AN00=2; Xiaomi/M2006J10C=1 |
| ISOSpeedRatings | HONOR/BVL-AN00=16; Xiaomi/M2006J10C=169 |
| BrightnessValue | HONOR/BVL-AN00=220; Xiaomi/M2006J10C=57 |

BrightnessValue两设备中位数差为6.84 EV，经验分布重叠为0.022；设备内稳健标准化后中位数差为0，重叠为0.849。因此需要进行设备内中心化/缩放。

## 四、FNumber专项判断

| camera_id | variable | valid_n | unique_n | min | p5 | median | p95 | max | mean | std | iqr | mad |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| HONOR/BVL-AN00 | FNumber | 283 | 2 | 1.9 | 1.9 | 1.9 | 1.9 | 2 | 1.90424 | 0.0201863 | 0 | 0 |
| Xiaomi/M2006J10C | FNumber | 239 | 1 | 1.89 | 1.89 | 1.89 | 1.89 | 1.89 | 1.89 | 9.56796e-15 | 0 | 0 |

FNumber的设备—取值频数：

| camera_id | FNumber | n |
| --- | --- | --- |
| HONOR/BVL-AN00 | 1.9 | 271 |
| HONOR/BVL-AN00 | 2 | 12 |
| Xiaomi/M2006J10C | 1.89 | 239 |

结论：FNumber设备内std、IQR和MAD接近0，几乎没有稳定的逐图信息。它不应因具有光学意义而被强行作为独立连续网络输入；推荐作为renderer固定设备参数，并参与relative_optical_exposure计算。

## 五、派生成像参数与相关性

已按提示词计算log2_exposure_time、log2_iso_gain、APEX光圈值、APEX时间值、EV100、relative_optical_exposure、combined_exposure_gain、sensitivity_value、brightness_apex_pred、brightness_residual及描述性device_centered_brightness。完整总体/设备分布、离群值、Pearson和Spearman矩阵见对应CSV。

combined_exposure_gain由relative_optical_exposure与log2_iso_gain确定性相加，因此不提供独立自由度。它可用于敏感性或质量控制，不宜与两个组成量同时输入V1。

## 六、BrightnessValue物理一致性

| scope | camera_id | n | pearson_r | spearman_rho | slope | intercept | r_squared | residual_median | residual_iqr | residual_mad | residual_min | residual_max |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| overall | ALL | 522 | 0.338584 | 0.461697 | 0.889041 | 2.48866 | 0.114639 | -0.741782 | 7.05123 | 0.326002 | -1.52778 | 6.90869 |
| device | HONOR/BVL-AN00 | 283 | 0.976057 | 0.988779 | 1.01366 | -0.874228 | 0.952687 | -0.895855 | 0.182892 | 0.09 | -1.52778 | 2.93414 |
| device | Xiaomi/M2006J10C | 239 | 0.942783 | 0.937293 | 0.952953 | 6.17617 | 0.88884 | 6.1623 | 0.254379 | 0.131173 | 3.19951 | 6.90869 |

设备截距/斜率交互模型：

| camera_id | term | coefficient | standard_error | term_p | r_squared | n |
| --- | --- | --- | --- | --- | --- | --- |
| reference=HONOR/BVL-AN00; indicator=Xiaomi/M2006J10C | intercept | -0.874228 | 0.0316116 | 4.50162e-104 | 0.989206 | 522 |
| reference=HONOR/BVL-AN00; indicator=Xiaomi/M2006J10C | brightness_apex_pred | 1.01366 | 0.0151881 | 1.48272e-256 | 0.989206 | 522 |
| reference=HONOR/BVL-AN00; indicator=Xiaomi/M2006J10C | camera_intercept_shift | 7.0504 | 0.047779 | 0 | 0.989206 | 522 |
| reference=HONOR/BVL-AN00; indicator=Xiaomi/M2006J10C | camera_slope_shift | -0.0607112 | 0.0247658 | 0.0145591 | 0.989206 | 522 |

BrightnessValue与实际皮肤线性亮度总体Spearman rho=-0.550。APEX拟合显示设备间截距、斜率或残差范围存在差异时，这种偏差只解释为手机自动曝光、HDR、ISP和厂商实现差异的线索，不自动判为元数据错误。原始BrightnessValue不建议跨设备直接使用；建议使用device_centered_brightness。

## 七、与实际图像表现的关系

| scope | camera_id | relationship | n | spearman_rho | spearman_p |
| --- | --- | --- | --- | --- | --- |
| overall | ALL | exposure_vs_luminance | 522 | -0.404799 | 5.30346e-22 |
| overall | ALL | exposure_vs_blur_proxy | 522 | -0.0624206 | 0.154414 |
| overall | ALL | iso_vs_noise_proxy | 522 | -0.0989074 | 0.0238277 |
| overall | ALL | iso_vs_underexposure | 522 | -0.110255 | 0.0117128 |
| overall | ALL | iso_vs_luminance | 522 | 0.225972 | 1.80739e-07 |
| overall | ALL | fnumber_vs_luminance | 522 | 0.620909 | 5.75465e-57 |
| overall | ALL | fnumber_vs_blur_proxy | 522 | -0.17549 | 5.54946e-05 |
| overall | ALL | brightnessvalue_vs_luminance | 522 | -0.550143 | 1.25126e-42 |
| overall | ALL | brightnessvalue_vs_lab_l | 522 | -0.550143 | 1.25126e-42 |
| overall | ALL | brightnessvalue_vs_overexposure | 522 | 0.0752959 | 0.0856827 |
| overall | ALL | brightnessvalue_vs_underexposure | 522 | 0.27486 | 1.67839e-10 |
| overall | ALL | relative_exposure_vs_luminance | 522 | -0.451636 | 1.33875e-27 |
| overall | ALL | combined_gain_vs_luminance | 522 | -0.0225911 | 0.606571 |
| device | HONOR/BVL-AN00 | exposure_vs_luminance | 283 | 0.131733 | 0.0266949 |
| device | HONOR/BVL-AN00 | exposure_vs_blur_proxy | 283 | -0.210093 | 0.000373041 |
| device | HONOR/BVL-AN00 | iso_vs_noise_proxy | 283 | -0.00342585 | 0.954245 |
| device | HONOR/BVL-AN00 | iso_vs_underexposure | 283 | -0.075544 | 0.205143 |
| device | HONOR/BVL-AN00 | iso_vs_luminance | 283 | 0.273884 | 2.91406e-06 |
| device | HONOR/BVL-AN00 | fnumber_vs_luminance | 283 | -0.0665416 | 0.264556 |
| device | HONOR/BVL-AN00 | fnumber_vs_blur_proxy | 283 | -0.230105 | 9.36882e-05 |
| device | HONOR/BVL-AN00 | brightnessvalue_vs_luminance | 283 | -0.272722 | 3.22183e-06 |
| device | HONOR/BVL-AN00 | brightnessvalue_vs_lab_l | 283 | -0.272722 | 3.22183e-06 |
| device | HONOR/BVL-AN00 | brightnessvalue_vs_overexposure | 283 |  |  |
| device | HONOR/BVL-AN00 | brightnessvalue_vs_underexposure | 283 | 0.0464226 | 0.436621 |
| device | HONOR/BVL-AN00 | relative_exposure_vs_luminance | 283 | 0.132468 | 0.0258527 |
| device | HONOR/BVL-AN00 | combined_gain_vs_luminance | 283 | 0.251029 | 1.92959e-05 |
| device | Xiaomi/M2006J10C | exposure_vs_luminance | 239 | -0.154986 | 0.016486 |
| device | Xiaomi/M2006J10C | exposure_vs_blur_proxy | 239 | -0.241247 | 0.000166008 |
| device | Xiaomi/M2006J10C | iso_vs_noise_proxy | 239 | -0.251119 | 8.67566e-05 |
| device | Xiaomi/M2006J10C | iso_vs_underexposure | 239 | 0.1344 | 0.0378654 |
| device | Xiaomi/M2006J10C | iso_vs_luminance | 239 | -0.262584 | 3.94748e-05 |
| device | Xiaomi/M2006J10C | fnumber_vs_luminance | 239 |  |  |
| device | Xiaomi/M2006J10C | fnumber_vs_blur_proxy | 239 |  |  |
| device | Xiaomi/M2006J10C | brightnessvalue_vs_luminance | 239 | 0.257705 | 5.54382e-05 |
| device | Xiaomi/M2006J10C | brightnessvalue_vs_lab_l | 239 | 0.257705 | 5.54382e-05 |
| device | Xiaomi/M2006J10C | brightnessvalue_vs_overexposure | 239 | -0.0235395 | 0.717311 |
| device | Xiaomi/M2006J10C | brightnessvalue_vs_underexposure | 239 | -0.169876 | 0.00849807 |
| device | Xiaomi/M2006J10C | relative_exposure_vs_luminance | 239 | -0.154986 | 0.016486 |
| device | Xiaomi/M2006J10C | combined_gain_vs_luminance | 239 | -0.225856 | 0.000433302 |

ISO与高频噪声代理总体Spearman rho=-0.099；relative_optical_exposure与皮肤线性亮度rho=-0.452；combined_exposure_gain与皮肤线性亮度rho=-0.023。手机自动曝光会使简单单调关系变弱，相关性不作因果解释。

图像指标定义：skin区域median RGB；sRGB反伽马后的线性相对亮度；由线性Y计算的Lab L*；线性亮度≥0.98为过曝、≤0.02为欠曝；任一通道≤5或≥250为通道裁剪/饱和；Laplacian variance为清晰度代理；高斯平滑残差MAD为高频噪声代理。后者不等于真实传感器噪声。

## 八、14张高ISO离群图像复核

| 复核分类 | n |
| --- | --- |
| 合理的真实拍摄条件 | 14 |

14张中明确元数据异常0张，需要保留但标记低质量0张。高ISO本身不构成删除理由；所有样本均保留。逐图结果：

| ID | Make | Model | ExposureTime | FNumber | ISOSpeedRatings | BrightnessValue | 是否过曝 | 是否欠曝 | 是否明显噪声较高 | 是否明显模糊 | 合理弱光条件线索 | 最终复核分类 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 201331515 | HONOR | BVL-AN00 | 0.01 | 1.9 | 1600 | -1.31 | False | False | False | False | True | 合理的真实拍摄条件 |
| 204650307 | HONOR | BVL-AN00 | 0.030303 | 1.9 | 1250 | -2.66 | False | False | False | False | True | 合理的真实拍摄条件 |
| A000104693 | HONOR | BVL-AN00 | 0.00714286 | 1.9 | 1250 | -0.62 | False | False | False | False | True | 合理的真实拍摄条件 |
| A000590309 | HONOR | BVL-AN00 | 0.00526316 | 1.9 | 1250 | -0.1 | False | False | False | False | False | 合理的真实拍摄条件 |
| A001074268 | HONOR | BVL-AN00 | 0.00588235 | 1.9 | 1250 | -0.32 | False | False | False | False | False | 合理的真实拍摄条件 |
| A001808459 | HONOR | BVL-AN00 | 0.005 | 1.9 | 1250 | -0.12 | False | False | False | False | False | 合理的真实拍摄条件 |
| A001837056 | HONOR | BVL-AN00 | 0.0149254 | 1.9 | 1250 | -1.61 | False | False | False | False | True | 合理的真实拍摄条件 |
| A001888679 | HONOR | BVL-AN00 | 0.00588235 | 1.9 | 1250 | -0.37 | False | False | False | False | False | 合理的真实拍摄条件 |
| A002331084 | HONOR | BVL-AN00 | 0.030303 | 1.9 | 1250 | -2.67 | False | False | False | False | True | 合理的真实拍摄条件 |
| A001636890 | Xiaomi | M2006J10C | 0.050009 | 1.89 | 1115 | 4.3 | False | False | False | False | True | 合理的真实拍摄条件 |
| A001632964 | Xiaomi | M2006J10C | 0.050009 | 1.89 | 1082 | 3.8 | False | False | False | False | True | 合理的真实拍摄条件 |
| A001028520 | Xiaomi | M2006J10C | 0.040008 | 1.89 | 871 | 4.2 | False | False | False | False | True | 合理的真实拍摄条件 |
| A001722742 | Xiaomi | M2006J10C | 0.040008 | 1.89 | 710 | 5 | False | False | False | False | True | 合理的真实拍摄条件 |
| A001438120 | Xiaomi | M2006J10C | 0.040008 | 1.89 | 700 | 5 | False | False | False | False | True | 合理的真实拍摄条件 |

## 九、最终字段决策表

| 字段 | 完整性 | 总体变异 | 设备内变异 | 设备系统偏移 | 与图像表现关系 | 推荐角色 | 结论依据 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| ExposureTime | 522/522 (100.0%) | unique=37, std=0.00917, IQR=0.02001, MAD=0.01 | HONOR/BVL-AN00: unique=29, std=0.00807, IQR=0.01, MAD=0.006667; Xiaomi/M2006J10C: unique=8, std=0.008208, IQR=0.01, MAD=0.01 | median差(B-A)=0.003337; Hedges g=1.05; overlap=0.597 | 最强关系=skin_linear_luminance_median, Spearman rho=-0.405 | derived_only | 逐图有变化，但网络中优先通过log2及relative_optical_exposure表达。 |
| FNumber | 522/522 (100.0%) | unique=3, std=0.01646, IQR=0.01, MAD=0 | HONOR/BVL-AN00: unique=2, std=0.02019, IQR=0, MAD=0; Xiaomi/M2006J10C: unique=1, std=9.568e-15, IQR=0, MAD=0 | median差(B-A)=-0.01; Hedges g=-0.957; overlap=0 | 最强关系=skin_linear_luminance_median, Spearman rho=0.621 | renderer_fixed_parameter | 设备内几乎固定，不作为独立连续输入；作为renderer设备固定参数并参与相对进光量公式。 |
| ISOSpeedRatings | 522/522 (100.0%) | unique=179, std=241.5, IQR=240, MAD=105.5 | HONOR/BVL-AN00: unique=16, std=277.7, IQR=300, MAD=120; Xiaomi/M2006J10C: unique=169, std=152.5, IQR=135.5, MAD=63 | median差(B-A)=-128; Hedges g=-0.675; overlap=0.592 | 最强关系=skin_linear_luminance_median, Spearman rho=0.226 | derived_only | 逐图有明显变化；使用log2_iso_gain表达增益，高ISO保留并做质量标记。 |
| BrightnessValue | 522/522 (100.0%) | unique=277, std=3.713, IQR=6.848, MAD=3.475 | HONOR/BVL-AN00: unique=220, std=1.575, IQR=2.17, MAD=1.04; Xiaomi/M2006J10C: unique=57, std=1.296, IQR=1.7, MAD=0.8 | median差(B-A)=6.84; Hedges g=4.71; overlap=0.0219 | 最强关系=skin_linear_luminance_median, Spearman rho=-0.550 | device_standardized_condition | 存在设备系统偏移，不能跨设备直接输入；改用设备内稳健标准化值。 |
| log2_exposure_time | 522/522 (100.0%) | unique=37, std=0.7567, IQR=1.585, MAD=0.5853 | HONOR/BVL-AN00: unique=29, std=0.7685, IQR=1, MAD=0.737; Xiaomi/M2006J10C: unique=8, std=0.5267, IQR=0.585, MAD=0.585 | median差(B-A)=0.2633; Hedges g=1.06; overlap=0.58 | 未单独对应图像指标；由组成字段或一致性分析解释 | derived_only | 比秒值更适合数值表示，但V1中被relative_optical_exposure吸收。 |
| log2_iso_gain | 522/522 (100.0%) | unique=179, std=1.023, IQR=1.322, MAD=0.6799 | HONOR/BVL-AN00: unique=16, std=1.087, IQR=1.322, MAD=0.6781; Xiaomi/M2006J10C: unique=169, std=0.8262, IQR=0.9998, MAD=0.5081 | median差(B-A)=-0.737; Hedges g=-0.63; overlap=0.664 | 未单独对应图像指标；由组成字段或一致性分析解释 | core_continuous_condition | V1核心增益条件。 |
| EV100 | 522/522 (100.0%) | unique=39, std=0.7567, IQR=1.599, MAD=0.6005 | HONOR/BVL-AN00: unique=31, std=0.7592, IQR=1, MAD=0.737; Xiaomi/M2006J10C: unique=8, std=0.5267, IQR=0.585, MAD=0.585 | median差(B-A)=-0.2785; Hedges g=-1.11; overlap=0.566 | 未单独对应图像指标；由组成字段或一致性分析解释 | quality_control_only | 与ExposureTime和FNumber确定性相关，保留作物理审计，不与V1并列输入。 |
| relative_optical_exposure | 522/522 (100.0%) | unique=39, std=0.7567, IQR=1.599, MAD=0.6005 | HONOR/BVL-AN00: unique=31, std=0.7592, IQR=1, MAD=0.737; Xiaomi/M2006J10C: unique=8, std=0.5267, IQR=0.585, MAD=0.585 | median差(B-A)=0.2785; Hedges g=1.11; overlap=0.566 | 最强关系=skin_linear_luminance_median, Spearman rho=-0.452 | core_continuous_condition | 合并曝光时间与光圈的光学进光条件，V1核心输入。 |
| combined_exposure_gain | 522/522 (100.0%) | unique=257, std=1.414, IQR=1.932, MAD=0.9862 | HONOR/BVL-AN00: unique=65, std=1.517, IQR=2.322, MAD=1; Xiaomi/M2006J10C: unique=192, std=1.282, IQR=1.59, MAD=0.8305 | median差(B-A)=0.1596; Hedges g=0.0829; overlap=0.801 | 最强关系=skin_linear_luminance_median, Spearman rho=-0.023 | quality_control_only | 由relative_optical_exposure和log2_iso_gain相加，信息确定性冗余，作敏感性/QC。 |
| brightness_apex_pred | 522/522 (100.0%) | unique=258, std=1.414, IQR=1.932, MAD=0.9862 | HONOR/BVL-AN00: unique=66, std=1.517, IQR=2.322, MAD=1; Xiaomi/M2006J10C: unique=192, std=1.282, IQR=1.59, MAD=0.8305 | median差(B-A)=-0.1596; Hedges g=-0.0829; overlap=0.801 | 未单独对应图像指标；由组成字段或一致性分析解释 | quality_control_only | 公式预测值只用于BrightnessValue一致性审计。 |
| brightness_residual | 522/522 (100.0%) | unique=412, std=3.497, IQR=7.051, MAD=0.326 | HONOR/BVL-AN00: unique=195, std=0.3433, IQR=0.1829, MAD=0.09; Xiaomi/M2006J10C: unique=217, std=0.4362, IQR=0.2544, MAD=0.1312 | median差(B-A)=7.058; Hedges g=17.9; overlap=0.00353 | 未单独对应图像指标；由组成字段或一致性分析解释 | quality_control_only | 用于发现设备偏移和元数据/ISP不一致，不作生理表型输入。 |
| device_centered_brightness | 522/522 (100.0%) | unique=276, std=0.7421, IQR=0.974, MAD=0.477 | HONOR/BVL-AN00: unique=220, std=0.7259, IQR=1, MAD=0.4793; Xiaomi/M2006J10C: unique=57, std=0.7622, IQR=1, MAD=0.4706 | median差(B-A)=0; Hedges g=0.0116; overlap=0.849 | 未单独对应图像指标；由组成字段或一致性分析解释 | device_standardized_condition | 减少设备系统偏移后的V1亮度条件；未来训练时统计量必须仅由训练集拟合。 |

## 十、十四个问题的明确回答

1. ExposureTime具有逐图变化，适合提供光学条件，但V1不直接使用秒值。
2. ExposureTime应先log2转换；V1进一步通过relative_optical_exposure与FNumber合并。
3. FNumber总体3个唯一值，且几乎没有设备内逐图信息。
4. FNumber不独立进入连续网络条件，只参与relative_optical_exposure，并作为renderer固定设备参数。
5. ISOSpeedRatings有179个唯一值，适合表达增益条件；与图像噪声的关系只能用噪声代理描述。
6. ISOSpeedRatings使用log2_iso_gain，不建议直接使用线性ISO数值。
7. BrightnessValue与实际皮肤亮度存在rho=-0.550的总体单调关系，但并非完美物理标定。
8. BrightnessValue不能跨设备直接使用。
9. 应使用device_centered_brightness；本报告的描述性统计使用全队列设备中位数/IQR，未来训练必须只用训练数据估计。
10. relative_optical_exposure比ExposureTime和FNumber分别并列输入更合理，可减少设备固定FNumber带来的冗余。
11. combined_exposure_gain与两个组成变量确定性冗余，没有额外自由度，只作QC/敏感性分析。
12. 推荐V1连续条件向量：[relative_optical_exposure, log2_iso_gain, device_centered_brightness]。
13. FNumber保留为renderer_fixed_parameter；EV100、combined_exposure_gain、brightness_apex_pred和brightness_residual只作质量控制/一致性分析。
14. 现有四字段完整且未发现高ISO元数据异常，支持继续开展曝光与增益条件化的面部光学反演，但需要设备校准、质量标记和后续外部设备验证。

## 十一、限制

本分析使用JPEG成图和既有解析skin区域；ISP、HDR、降噪、白平衡和tone mapping都会改变像素表现。相关性只能支持条件变量的工程可用性判断，不能证明真实辐射度、传感器噪声或因果机制。
