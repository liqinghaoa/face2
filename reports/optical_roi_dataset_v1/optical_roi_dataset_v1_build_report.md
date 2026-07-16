# Optical ROI Dataset V1 构建报告

## 1. 完成状态

- `OPTICAL_ROI_DATASET_STATUS=COMPLETE`
- `READY_FOR_OPTICAL_FEATURE_EXTRACTION=YES`
- 本状态仅表示500例输入和三类有效皮肤mask构建、保存与完整性验证完成，不表示物理反演模型或物理参数有效。

## 2. 新增文件

- `preprocessing/build_optical_roi_dataset_v1.py`
- `config/preprocess/optical_roi_dataset_v1.yaml`
- `tests/test_optical_roi_dataset_v1.py`
- `data/processed/optical_roi_dataset_v1/`：1500张mask、manifest与build manifest。
- `reports/optical_roi_dataset_v1/`：覆盖统计、低覆盖清单、预览、日志与本报告。

## 3. 未修改的历史数据声明

构建前后历史输入库存摘要一致：`134f2e8cbbc06c18838f8edeb21520e85d653925b36dbdcdba128c069ed5305c`。程序的写入路径仅限两个新输出目录；没有修改meanbg、aligned、parsing label、final mask或ROI日志。

## 4. 固定500例ID来源

唯一研究集合来自 `data/processed/global_face/preprocess_ablation/hybrid_imagenet_meanbg/images` 的500张PNG完整stem。manifest为500行、500个唯一ID，与该集合完全一致；没有加入上游多出的22例。

## 5. 图像和mask来源

- aligned RGB：`data/processed/global_face/global_face_parsing_hybrid_foreheadrepair_blackbg_224_png_strict_intermediates/aligned_rgb`
- parsing label：`data/processed/global_face/global_face_parsing_hybrid_foreheadrepair_blackbg_224_png_strict_intermediates/parsing_label`
- final mask：`data/processed/global_face/global_face_parsing_hybrid_foreheadrepair_blackbg_224_png_strict_intermediates/final_mask`
- aligned均为224×224 RGB uint8；parsing label为224×224单通道离散标签；final mask以`>0`定义有效区域。

## 6. bbox来源与端点约定

bbox日志为 `data/processed/roi_dataset/global_aligned_face_parsing_roi_final5_224_canvas_500/logs/roi_metadata.csv`，SHA256=`cb9a21764e50cc8a296e6964eb88b43506e3d8bed91289cc8411f7955b19e225`。forehead读取`forehead_roi`通用bbox字段；两侧cheek读取`cheek_roi`专用left/right字段。CSV为包含式`(x1,y1,x2,y2)`，NumPy切片使用`[y1:y2+1, x1:x2+1]`。

## 7. skin标签代码依据

从 `preprocessing/build_global_face_parsing_regularmask_blackbg_224_png_strict.py` 的`CLASS_SKIN`定义读取到`skin_label=1`，不是凭经验指定。

## 8. 三类mask精确定义

`base_skin = (parsing_label == 1) AND (final_mask > 0)`；每类mask为`base_skin AND 对应包含式bbox区域`。保存为224×224、单通道uint8 PNG，背景0、有效像素255。没有resize、padding、羽化、腐蚀、膨胀、开闭运算、bbox移动或个例规则。

## 9. EXIF来源和连接方式

设备字段来自 `reports/exif_parameter_audit/image_parameter_audit.csv` 的`ID/Make/Model`；数值字段来自 `reports/exif_parameter_audit/parameter_values_long.csv` 中参数名为ExposureTime、FNumber、ISOSpeedRatings的行。只加载这些非临床列，按完整字符串ID一对一连接。

## 10. EXIF派生公式

- `relative_optical_exposure = log2(ExposureTime / FNumber^2)`
- `log2_iso_condition = log2(ISOSpeedRatings / 100)`
- 未做全队列标准化、设备内中心化或camera数值编码。

## 11. 500例完整性验证

磁盘复读状态：`PASS`；manifest=500行、唯一ID=500；三类mask数={'forehead': 500, 'cheek_image_left': 500, 'cheek_image_right': 500}，总数=1500。所有mask均为224×224单通道uint8、仅0/255、非空、位于bbox内且属于`parsing skin AND final mask`；manifest像素数和SHA256均与磁盘一致。

## 12–13. ROI有效像素数和skin比例分布

| roi_name | metric | valid_n | empty_n | min | p1 | p5 | median | p95 | p99 | max | mean | std | iqr |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| forehead | valid_skin_pixel_count | 500 | 0 | 10.0 | 159.92000000000002 | 1690.9500000000003 | 4313.0 | 5937.749999999999 | 6634.09 | 6872.0 | 4165.148 | 1244.1222779860352 | 1529.25 |
| forehead | valid_skin_fraction | 500 | 0 | 0.0023741690408357074 | 0.03660188976762948 | 0.3415983118429911 | 0.6467058632354151 | 0.770636216508858 | 0.8134791220434262 | 0.8902308105206655 | 0.6180869149506918 | 0.13671193277413704 | 0.1129388134612348 |
| cheek_image_left | valid_skin_pixel_count | 500 | 0 | 1212.0 | 1325.93 | 1535.55 | 1832.0 | 2142.0 | 2237.18 | 2301.0 | 1831.61 | 188.40092280939297 | 252.0 |
| cheek_image_left | valid_skin_fraction | 500 | 0 | 0.6733333333333333 | 0.7513332894133263 | 0.8315538817152878 | 0.9728737357571476 | 1.0 | 1.0 | 1.0 | 0.9532383556151992 | 0.05753628115026665 | 0.06131175875203998 |
| cheek_image_right | valid_skin_pixel_count | 500 | 0 | 1256.0 | 1398.83 | 1503.7 | 1811.0 | 2091.25 | 2280.05 | 2419.0 | 1807.648 | 183.09076588704903 | 243.5 |
| cheek_image_right | valid_skin_fraction | 500 | 0 | 0.6659597030752916 | 0.7680370544090056 | 0.8379560362118501 | 0.9529539661118609 | 1.0 | 1.0 | 1.0 | 0.9408429331933462 | 0.054777259288266614 | 0.0706442350360893 |

## 14. 两设备分层统计

| camera_id | roi_name | valid_n | min | median | max | mean | std |
| --- | --- | --- | --- | --- | --- | --- | --- |
| HONOR/BVL-AN00 | forehead | 267 | 0.014155052264808362 | 0.651010101010101 | 0.8902308105206655 | 0.6364548490844144 | 0.11363490132746983 |
| HONOR/BVL-AN00 | cheek_image_left | 267 | 0.7517934002869441 | 0.9699248120300752 | 1.0 | 0.9534992914414299 | 0.052126979659645564 |
| HONOR/BVL-AN00 | cheek_image_right | 267 | 0.6659597030752916 | 0.9595441595441595 | 1.0 | 0.9460805283421856 | 0.05466906633075788 |
| Xiaomi/M2006J10C | forehead | 233 | 0.0023741690408357074 | 0.643584229390681 | 0.8147001934235977 | 0.5970386814154819 | 0.15672085458201898 |
| Xiaomi/M2006J10C | cheek_image_left | 233 | 0.6733333333333333 | 0.9761286974571873 | 1.0 | 0.9529393433164716 | 0.0632823903332004 |
| Xiaomi/M2006J10C | cheek_image_right | 233 | 0.6807170542635659 | 0.9419512195121951 | 1.0 | 0.9348410537738606 | 0.05439953567647469 |

仅提供描述性统计，没有设备间显著性检验。

## 15. 空mask、缺失和非法记录

空mask={'forehead': 0, 'cheek_image_left': 0, 'cheek_image_right': 0}；非法bbox=0；缺失EXIF=0；非有限EXIF派生值=0；完整性错误=0。

## 16. 低覆盖病例清单

低像素数和低比例病例只记录、不排除。完整清单位于`low_coverage_cases.csv`。

| ID | camera_id | roi_name | valid_skin_pixel_count | valid_skin_fraction | bbox_area | selection_reason |
| --- | --- | --- | --- | --- | --- | --- |
| A001632964 | Xiaomi/M2006J10C | cheek_image_left | 1212 | 0.6733333333333333 | 1800 | lowest_10_valid_skin_fraction;lowest_10_valid_skin_pixel_count |
| A001638543 | Xiaomi/M2006J10C | cheek_image_left | 1316 | 0.6977730646871686 | 1886 | lowest_10_valid_skin_fraction;lowest_10_valid_skin_pixel_count |
| 206528283 | Xiaomi/M2006J10C | cheek_image_left | 1319 | 0.6993637327677624 | 1886 | lowest_10_valid_skin_fraction;lowest_10_valid_skin_pixel_count |
| 9300706043 | Xiaomi/M2006J10C | cheek_image_left | 1290 | 0.7037643207855974 | 1833 | lowest_10_valid_skin_fraction;lowest_10_valid_skin_pixel_count |
| A001663062 | Xiaomi/M2006J10C | cheek_image_left | 1245 | 0.70578231292517 | 1764 | lowest_10_valid_skin_fraction;lowest_10_valid_skin_pixel_count |
| A001878464-1 | HONOR/BVL-AN00 | cheek_image_left | 1572 | 0.7517934002869441 | 2091 | lowest_10_valid_skin_fraction |
| A001911401 | HONOR/BVL-AN00 | cheek_image_left | 1668 | 0.7564625850340136 | 2205 | lowest_10_valid_skin_fraction |
| A001718170 | Xiaomi/M2006J10C | cheek_image_left | 1332 | 0.7568181818181818 | 1760 | lowest_10_valid_skin_fraction;lowest_10_valid_skin_pixel_count |
| A001614685 | Xiaomi/M2006J10C | cheek_image_left | 1547 | 0.7583333333333333 | 2040 | lowest_10_valid_skin_fraction |
| 202411756 | HONOR/BVL-AN00 | cheek_image_left | 1552 | 0.7607843137254902 | 2040 | lowest_10_valid_skin_fraction |
| A002145202-1 | HONOR/BVL-AN00 | cheek_image_left | 1344 | 0.7832167832167832 | 1716 | lowest_10_valid_skin_pixel_count |
| 204586681 | HONOR/BVL-AN00 | cheek_image_left | 1335 | 0.7984449760765551 | 1672 | lowest_10_valid_skin_pixel_count |
| 203036405 | Xiaomi/M2006J10C | cheek_image_left | 1326 | 0.8532818532818532 | 1554 | lowest_10_valid_skin_pixel_count |
| A002236665 | HONOR/BVL-AN00 | cheek_image_left | 1360 | 0.8994708994708994 | 1512 | lowest_10_valid_skin_pixel_count |
| 206248971 | HONOR/BVL-AN00 | cheek_image_right | 1256 | 0.6659597030752916 | 1886 | lowest_10_valid_skin_fraction;lowest_10_valid_skin_pixel_count |
| 201223046 | HONOR/BVL-AN00 | cheek_image_right | 1361 | 0.6750992063492064 | 2016 | lowest_10_valid_skin_fraction;lowest_10_valid_skin_pixel_count |
| A001657263 | Xiaomi/M2006J10C | cheek_image_right | 1405 | 0.6807170542635659 | 2064 | lowest_10_valid_skin_fraction;lowest_10_valid_skin_pixel_count |
| A001646267 | Xiaomi/M2006J10C | cheek_image_right | 1485 | 0.7576530612244898 | 1960 | lowest_10_valid_skin_fraction |
| A001602118 | Xiaomi/M2006J10C | cheek_image_right | 1506 | 0.7652439024390244 | 1968 | lowest_10_valid_skin_fraction |
| 206409114 | HONOR/BVL-AN00 | cheek_image_right | 1318 | 0.7680652680652681 | 1716 | lowest_10_valid_skin_fraction;lowest_10_valid_skin_pixel_count |
| A001870250 | HONOR/BVL-AN00 | cheek_image_right | 1382 | 0.770345596432553 | 1794 | lowest_10_valid_skin_fraction;lowest_10_valid_skin_pixel_count |
| A001870548 | HONOR/BVL-AN00 | cheek_image_right | 1564 | 0.7784967645594824 | 2009 | lowest_10_valid_skin_fraction |
| A001638334 | Xiaomi/M2006J10C | cheek_image_right | 1345 | 0.7819767441860465 | 1720 | lowest_10_valid_skin_fraction;lowest_10_valid_skin_pixel_count |
| A001093486 | Xiaomi/M2006J10C | cheek_image_right | 1465 | 0.7927489177489178 | 1848 | lowest_10_valid_skin_fraction |
| A001189410 | Xiaomi/M2006J10C | cheek_image_right | 1403 | 0.8204678362573099 | 1710 | lowest_10_valid_skin_pixel_count |
| 9300122741 | Xiaomi/M2006J10C | cheek_image_right | 1399 | 0.8367224880382775 | 1672 | lowest_10_valid_skin_pixel_count |
| 201537151 | HONOR/BVL-AN00 | cheek_image_right | 1438 | 0.837995337995338 | 1716 | lowest_10_valid_skin_pixel_count |
| A002037682 | HONOR/BVL-AN00 | cheek_image_right | 1431 | 0.8517857142857143 | 1680 | lowest_10_valid_skin_pixel_count |
| A001669732 | Xiaomi/M2006J10C | forehead | 10 | 0.0023741690408357074 | 4212 | lowest_10_valid_skin_fraction;lowest_10_valid_skin_pixel_count |
| A001667601 | Xiaomi/M2006J10C | forehead | 60 | 0.013818516812528788 | 4342 | lowest_10_valid_skin_fraction;lowest_10_valid_skin_pixel_count |
| A002021058-1 | HONOR/BVL-AN00 | forehead | 65 | 0.014155052264808362 | 4592 | lowest_10_valid_skin_fraction;lowest_10_valid_skin_pixel_count |
| A001818287 | HONOR/BVL-AN00 | forehead | 136 | 0.03304178814382896 | 4116 | lowest_10_valid_skin_fraction;lowest_10_valid_skin_pixel_count |
| 204378187 | HONOR/BVL-AN00 | forehead | 152 | 0.033815350389321465 | 4495 | lowest_10_valid_skin_fraction;lowest_10_valid_skin_pixel_count |
| 9301133685 | Xiaomi/M2006J10C | forehead | 160 | 0.03663003663003663 | 4368 | lowest_10_valid_skin_fraction;lowest_10_valid_skin_pixel_count |
| 203874889 | Xiaomi/M2006J10C | forehead | 197 | 0.047688211086903895 | 4131 | lowest_10_valid_skin_fraction;lowest_10_valid_skin_pixel_count |
| 203582980 | Xiaomi/M2006J10C | forehead | 464 | 0.09656607700312175 | 4805 | lowest_10_valid_skin_fraction;lowest_10_valid_skin_pixel_count |
| A001623880 | Xiaomi/M2006J10C | forehead | 1181 | 0.1510230179028133 | 7820 | lowest_10_valid_skin_fraction |
| A001250372 | Xiaomi/M2006J10C | forehead | 701 | 0.15202775970505314 | 4611 | lowest_10_valid_skin_fraction;lowest_10_valid_skin_pixel_count |
| 200378127 | Xiaomi/M2006J10C | forehead | 761 | 0.16400862068965516 | 4640 | lowest_10_valid_skin_pixel_count |

## 17. 三个已人工确认额部病例

| ID | camera_id | forehead_valid_skin_pixel_count | forehead_valid_skin_fraction | forehead_bbox_area |
| --- | --- | --- | --- | --- |
| 206664979 | Xiaomi/M2006J10C | 3341 | 0.5822586266991984 | 5738 |
| 9300518956 | HONOR/BVL-AN00 | 806 | 0.18528735632183907 | 4350 |
| A001471135 | Xiaomi/M2006J10C | 3967 | 0.5185620915032679 | 7650 |

记录图：`reports/optical_roi_dataset_v1/previews/manually_checked_forehead_cases.png`。这些病例没有被自动标记为失败。

## 18. 测试结果

专项测试状态：`PASS`。测试输出：`..............                                                           [100%] | 14 passed in 0.54s`

## 19. 是否满足继续提取区域光学量的条件

`READY_FOR_OPTICAL_FEATURE_EXTRACTION=YES`。允许下一阶段读取aligned RGB并仅在这些mask有效像素上计算预先定义的区域光学量；本阶段没有计算任何RGB统计或物理反演。

## 20. 已知限制

1. saved aligned RGB已经经过上游双线性几何重采样，不是相机原始坐标像素。
2. mask依赖BiSeNet离散标签和既有final mask；解析错误会传递到ROI。
3. 部分额部有效像素很少，但V1不设置最低阈值，也不自动修改bbox。
4. image-left/right仅指图像x坐标方向，不代表患者解剖学左右。
5. 数据集完成不证明物理反演有效，也不证明跨设备无混杂。

## 21. 下一步最小建议

保持本V1 mask和manifest不变，新增独立的区域光学量提取步骤；任何标准化参数仅在后续每个训练折的训练子集内计算。
