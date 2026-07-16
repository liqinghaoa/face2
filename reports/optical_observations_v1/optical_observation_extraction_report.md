<!-- task_name: Regional Facial Optical Observation Extraction V1 -->
# 区域面部光学观测量提取 V1 报告

## 1. 完成状态

- `OPTICAL_OBSERVATION_EXTRACTION_STATUS=COMPLETE`
- 验收状态：`PASS`；专项测试：`PASS`。
- 本产物仅称为区域光学观测量或linear-sRGB-like区域观测，不是皮肤反射率、sensor RGB或生理参数。

## 2. 新增文件

本任务新增或更新的文件如下；未改动既有输入数据和其他实验代码。

- `preprocessing/extract_regional_optical_observations_v1.py`
- `config/preprocess/regional_optical_observations_v1.yaml`
- `tests/test_regional_optical_observations_v1.py`
- `data/processed/optical_observations_v1/regional_optical_observations.csv`
- `data/processed/optical_observations_v1/regional_optical_qc_long.csv`
- `data/processed/optical_observations_v1/feature_schema.json`
- `data/processed/optical_observations_v1/extraction_manifest.json`
- `reports/optical_observations_v1/optical_observation_extraction_report.md`
- `reports/optical_observations_v1/observation_summary.csv`
- `reports/optical_observations_v1/observation_summary_by_camera.csv`
- `reports/optical_observations_v1/exif_observation_associations.csv`
- `reports/optical_observations_v1/qc_flagged_cases.csv`
- `reports/optical_observations_v1/extraction_run.log`

## 3. 输入图像、Mask和EXIF来源

- aligned RGB：`data/processed/global_face/global_face_parsing_hybrid_foreheadrepair_blackbg_224_png_strict_intermediates/aligned_rgb`。
- ROI Mask：`data/processed/optical_roi_dataset_v1/masks`，路径从现有ROI manifest精确读取，未重新生成。
- ROI与EXIF manifest：`data/processed/optical_roi_dataset_v1/optical_roi_manifest.csv`。仅点名读取ExposureTime、FNumber、ISOSpeedRatings、camera_id及ROI字段；未读取BrightnessValue或临床字段。

## 4. 500例队列定义

研究ID来自`data/processed/global_face/preprocess_ablation/hybrid_imagenet_meanbg/images`的500个完整PNG stem，仅使用文件名定义队列，未使用meanbg图像计算颜色。主表500行、唯一ID=500，按完整字符串ID字典序升序。

## 5. RGB通道和颜色空间假设

项目已有OpenCV版`read_rgb`，但当前环境未安装OpenCV且任务禁止安装依赖；因此本任务使用Pillow `Image.open(...).convert('RGB')`原生RGB顺序读取，不resize，并用纯红/绿/蓝单元测试锁定通道顺序。500张PNG均无嵌入ICC profile，因此假定其编码为sRGB；该假设合理但未被文件标签证明。

## 6. inverse sRGB公式

先令`C_srgb=C_uint8/255`。当`C_srgb<=0.04045`时，`C_linear=C_srgb/12.92`；否则`C_linear=((C_srgb+0.055)/1.055)^2.4`。全程float64，所得量称为linear-sRGB-like。

## 7. Y、log2-R/G、log2-B/G公式

`Y=0.2126R_linear+0.7152G_linear+0.0722B_linear`；`log2_y=log2(Y+1e-6)`；`log2_rg=log2((R_linear+1e-6)/(G_linear+1e-6))`；`log2_bg=log2((B_linear+1e-6)/(G_linear+1e-6))`。

## 8. ROI稳健汇总方式

每例每ROI仅在`mask>0`像素内计算三个维度的Q25、median、Q75和IQR。主表使用median；原始四分位统计完整保留在1500行QC长表。

## 9. 额部20%规则

固定使用`forehead_valid_skin_fraction>=0.20`，可用486例，不可用14例。没有增加像素数阈值，也未根据EXIF或观测值调阈值。

## 10. 不可用额部处理方式

病例和双侧脸颊均保留；主表额部三个median及额部减脸颊三个字段写为空字段/NaN。QC长表仍保存额部原始Q25、median、Q75和IQR；没有用0替代或修改Mask。

## 11. 左右脸颊合并和差异定义

`cheek_mean=(left_median+right_median)/2`；`cheek_abs_diff=abs(left_median-right_median)`。后者只归入QC角色。额部可用时计算`forehead_minus_cheek=forehead_median-cheek_mean`。

## 12. 字段角色

`feature_schema.json`明确区分ID、原始EXIF、采集条件、设备条件、可用性、区域观测、派生观测、QC字段和禁止直接进入分类器的字段。提示词中的拼写`valid_skinek_pixel_count`映射为现有正确字段`valid_skin_pixel_count`。

## 13. EXIF在本阶段的作用

EXIF仅作为后续反演的采集条件保留。复算`relative_optical_exposure=log2(ExposureTime/FNumber^2)`和`log2_iso_condition=log2(ISO/100)`并与来源核对；未用EXIF修改像素，未做exposure-corrected RGB。

## 14. 输出数据完整性

主表=500行，QC表=1500行，各ROI行数={'forehead': 500, 'cheek_image_left': 500, 'cheek_image_right': 500}；左脸颊有限病例=500，右脸颊有限病例=500。重复提取CSV SHA256一致：`True`。

## 15. 通道截断和非有限值检查

变换后非有限值总数=0。截断比例均在原始uint8、ROI Mask内部计算；任何QC标记仅记录、不自动排除。标记行数=150。

| metric | roi_rows_with_nonzero_fraction | maximum_fraction |
| --- | --- | --- |
| r_equal_0_fraction | 2 | 0.0011329305135951663 |
| r_equal_255_fraction | 128 | 0.09295967190704033 |
| r_le_5_fraction | 2 | 0.0011329305135951663 |
| r_ge_250_fraction | 228 | 0.22146274777853725 |
| g_equal_0_fraction | 5 | 0.00281214848143982 |
| g_equal_255_fraction | 1 | 0.001367053998632946 |
| g_le_5_fraction | 6 | 0.005061867266591676 |
| g_ge_250_fraction | 5 | 0.005468215994531784 |
| b_equal_0_fraction | 8 | 0.00562429696287964 |
| b_equal_255_fraction | 1 | 0.0005806077027288562 |
| b_le_5_fraction | 23 | 0.01124859392575928 |
| b_ge_250_fraction | 2 | 0.0035149384885764497 |

## 16. 两设备描述性统计

| camera_id | n |
| --- | --- |
| HONOR/BVL-AN00 | 267 |
| Xiaomi/M2006J10C | 233 |

以下给出两个设备上三项代表性观测的实际统计量；完整结果见`observation_summary_by_camera.csv`，未进行显著性筛选。

| camera_id | observation_name | valid_n | missing_n | median | iqr | mean | std |
| --- | --- | --- | --- | --- | --- | --- | --- |
| HONOR/BVL-AN00 | forehead_log2_y_median | 263 | 4 | -1.741427 | 0.393611 | -1.745659 | 0.289471 |
| HONOR/BVL-AN00 | cheek_mean_log2_y | 267 | 0 | -1.694273 | 0.260811 | -1.709964 | 0.220164 |
| Xiaomi/M2006J10C | forehead_log2_y_median | 223 | 10 | -2.216398 | 0.777829 | -2.260452 | 0.554139 |
| Xiaomi/M2006J10C | cheek_mean_log2_y | 233 | 0 | -2.282771 | 0.611993 | -2.268755 | 0.462523 |

## 17. 区域观测与EXIF条件的描述性关系

以下给出亮度相关代表性观测与两个EXIF条件的整体及设备内Spearman相关；完整结果见`exif_observation_associations.csv`。这些数值仅作描述，没有读取NYHA、选择特征、做显著性检验或按结果改公式。

| scope | camera_id | observation_name | condition_name | valid_n | spearman_rho |
| --- | --- | --- | --- | --- | --- |
| overall | ALL | forehead_log2_y_median | relative_optical_exposure | 486 | -0.37681 |
| overall | ALL | forehead_log2_y_median | log2_iso_condition | 486 | 0.137579 |
| overall | ALL | cheek_mean_log2_y | relative_optical_exposure | 500 | -0.431174 |
| overall | ALL | cheek_mean_log2_y | log2_iso_condition | 500 | 0.286611 |
| camera_id | HONOR/BVL-AN00 | forehead_log2_y_median | relative_optical_exposure | 263 | 0.039384 |
| camera_id | HONOR/BVL-AN00 | forehead_log2_y_median | log2_iso_condition | 263 | 0.10033 |
| camera_id | HONOR/BVL-AN00 | cheek_mean_log2_y | relative_optical_exposure | 267 | 0.167248 |
| camera_id | HONOR/BVL-AN00 | cheek_mean_log2_y | log2_iso_condition | 267 | 0.364738 |
| camera_id | Xiaomi/M2006J10C | forehead_log2_y_median | relative_optical_exposure | 223 | -0.140816 |
| camera_id | Xiaomi/M2006J10C | forehead_log2_y_median | log2_iso_condition | 223 | -0.229807 |
| camera_id | Xiaomi/M2006J10C | cheek_mean_log2_y | relative_optical_exposure | 233 | -0.172391 |
| camera_id | Xiaomi/M2006J10C | cheek_mean_log2_y | log2_iso_condition | 233 | -0.251228 |

## 18. 单元测试和全量验证结果

专项测试：`PASS`（................                                                         [100%] | 16 passed in 0.67s）。全量验收：`PASS`。主CSV首次SHA256=5d3fe7109846b5ee5fc0ba0034de2294caf3670a9b8552277681e28b86a5e0c6，重复=5d3fe7109846b5ee5fc0ba0034de2294caf3670a9b8552277681e28b86a5e0c6；QC CSV首次=9b53503d3aab3f0e82b7805f04ee0231e766e200723a73ca89d60eb2b9585cc0，重复=9b53503d3aab3f0e82b7805f04ee0231e766e200723a73ca89d60eb2b9585cc0。

## 19. 未修改历史输入声明

历史输入库存构建前后摘要分别为`f9b9d3479f0e1b0056b667531d7e9b37190d2781f140b8002ed15a90098a62d0`和`f9b9d3479f0e1b0056b667531d7e9b37190d2781f140b8002ed15a90098a62d0`，一致；`historical_inputs_modified=false`。只写入本任务两个新输出目录。

## 20. 局限性

1. 输入是手机处理后的JPEG/PNG编码图像，不是RAW。
2. 保存的PNG没有嵌入颜色配置文件；sRGB是合理但未经文件标签证明的假设。
3. inverse sRGB不能逆转手机白平衡、ISP、HDR或色调映射。
4. 当前输出是区域光学观测量，不是真实皮肤反射率，也不是传感器线性RGB。
5. 尚未实现光学反演网络。
6. 尚未读取或验证这些观测量与NYHA的关系。
