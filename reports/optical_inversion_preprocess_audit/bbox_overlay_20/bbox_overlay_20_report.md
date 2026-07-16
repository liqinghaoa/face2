# 20例 ROI bbox 与 saved aligned RGB 坐标一致性审计

## 1. 完成状态

完成：20例均已生成并逐例复核。
本任务未修改任何已有代码或源图像，未重新执行人脸对齐/解析，未生成正式 ROI mask 或 ROI RGB 数据集，也未读取临床标签。

## 2. bbox 数据来源与字段

- 实际 bbox 日志：`E:\projects\face2\data\processed\roi_dataset\global_aligned_face_parsing_roi_final5_224_canvas_500\logs\roi_metadata.csv`
- 日志与 `roi_qc_metrics.csv` 的 SHA-256 完全一致；本审计采用语义更明确的 `roi_metadata.csv`。
- 每个当前研究 ID 在日志中有5条 ROI 记录；本任务只读取 `forehead_roi` 与 `cheek_roi`，各500条且全部 `roi_success=True`。
- forehead：`bbox_x1, bbox_y1, bbox_x2, bbox_y2`（`roi_type=forehead_roi`）。
- image-left cheek：`left_cheek_bbox_x1, left_cheek_bbox_y1, left_cheek_bbox_x2, left_cheek_bbox_y2`（`roi_type=cheek_roi`）。
- image-right cheek：`right_cheek_bbox_x1, right_cheek_bbox_y1, right_cheek_bbox_x2, right_cheek_bbox_y2`（`roi_type=cheek_roi`）。

## 3. 坐标格式与源码依据

- 格式为 `(x1, y1, x2, y2)`，不是 `(x, y, w, h)`。
- ROI 源码内部 `BBox` 使用 Python 切片式右开区间；写日志时以 `bbox.x2 - 1`、`bbox.y2 - 1` 保存，因此 CSV 的 `x2/y2` 是包含端点。绘图使用包含端点；如用于 NumPy/Python 切片，需恢复为 `y1:y2+1, x1:x2+1`。
- 证据位于 `preprocessing/preprocess_global_aligned_face_parsing_roi_dataset_224_canvas.py`：内部 BBox 定义（约137–156行）、left/right 为 224-space 图像坐标的说明（约880行）、双颊日志写入（1477–1490行）、通用 bbox 日志写入（1561–1564行）。
- `left` 是图像坐标中较小 x 的一侧，`right` 是较大 x 的一侧；不是患者解剖学左右。
- 临时有效皮肤按项目定义计算为 `(parsing_label == 1) AND (final_mask > 0)`；皮肤标签 `1` 来自 `preprocessing/build_global_face_parsing_regularmask_blackbg_224_png_strict.py:112`。未进行腐蚀、膨胀或其他形态学处理。

## 4. 20例选择方法

只从当前 meanbg 目录的500个 ID 中选择。设备信息直接读取既有 EXIF 审计结果的 `Make` 与 `Model`，没有重新提取图像 EXIF；仅加载 `ID/Make/Model` 三列。每台设备内部按 ID 字符串稳定排序，在 `linspace(0, n-1, 10)` 上用 `floor(position+0.5)` 确定10个均匀分位位置。

- HONOR/BVL-AN00：当前500例中 267 例，抽取10例。
- Xiaomi/M2006J10C：当前500例中 233 例，抽取10例。
- 选择结果保存在 `bbox_overlay_20_selection.csv`，包含确切源路径、bbox 和排序位置，可完全复现。

## 5. ID、文件与边界检查

- meanbg：500个 PNG、500个唯一 ID。
- EXIF 审计：500/500 ID 一一匹配。
- bbox 日志：500/500 ID 匹配；额部和双颊所需行均为500/500。
- saved aligned RGB、parsing label、final mask：三类文件均为500/500存在。
- 入选图像：20/20 为 224×224、RGB、uint8。
- bbox：20/20 例的三组坐标全部位于 224×224 范围内。

## 6. 逐例视觉判断汇总

coordinate_registration：PASS 20；QUESTIONABLE 0；FAIL 0。

| ID | camera_id | coordinate_registration | forehead_content | cheek_left_content | cheek_right_content | issue_type |
| --- | --- | --- | --- | --- | --- | --- |
| 100643124 | HONOR/BVL-AN00 | PASS | PASS | PASS | PASS | NONE |
| 203874389 | HONOR/BVL-AN00 | PASS | PASS | PASS | PASS | NONE |
| 9300518956 | HONOR/BVL-AN00 | PASS | FAIL | PASS | PASS | ROI_CONTENT_ONLY |
| A001083428 | HONOR/BVL-AN00 | PASS | PASS | PASS | PASS | NONE |
| A001808459 | HONOR/BVL-AN00 | PASS | PASS | PASS | PASS | NONE |
| A001892094 | HONOR/BVL-AN00 | PASS | PASS | PASS | PASS | NONE |
| A001938232 | HONOR/BVL-AN00 | PASS | PASS | PASS | PASS | NONE |
| A002038079 | HONOR/BVL-AN00 | PASS | PASS | PASS | PASS | NONE |
| A002156070 | HONOR/BVL-AN00 | PASS | PASS | PASS | PASS | NONE |
| A002366013 | HONOR/BVL-AN00 | PASS | PASS | PASS | PASS | NONE |
| 100037382 | Xiaomi/M2006J10C | PASS | PASS | PASS | PASS | NONE |
| 203659997 | Xiaomi/M2006J10C | PASS | PASS | PASS | PASS | NONE |
| 206664979 | Xiaomi/M2006J10C | PASS | FAIL | PASS | PASS | ROI_CONTENT_ONLY |
| A000454833 | Xiaomi/M2006J10C | PASS | PASS | PASS | PASS | NONE |
| A001471135 | Xiaomi/M2006J10C | PASS | QUESTIONABLE | PASS | PASS | ROI_CONTENT_ONLY |
| A001610296 | Xiaomi/M2006J10C | PASS | PASS | PASS | PASS | NONE |
| A001628261 | Xiaomi/M2006J10C | PASS | PASS | PASS | PASS | NONE |
| A001646267 | Xiaomi/M2006J10C | PASS | PASS | PASS | PASS | NONE |
| A001688703 | Xiaomi/M2006J10C | PASS | PASS | PASS | PASS | NONE |
| A001794983 | Xiaomi/M2006J10C | PASS | PASS | PASS | PASS | NONE |

## 7. 坐标系统问题与 ROI 内容问题

坐标系统不一致应表现为三个 bbox 在 saved aligned RGB 上出现一致方向的整体平移、缩放或落入错误区域。ROI 内容问题则是坐标落点与构造规则相符，但原始矩形可能纳入头发、背景、鼻部或其他非目标内容。后者不能据此判定坐标系统失败；正式使用时应以 parsing skin 与 final mask 的交集约束有效像素。

## 8. 最终判定

- `BBOX_COORDINATE_ALIGNMENT=PASS`
- `BBOX_REUSE_DECISION=SAFE_WITH_ROI_MASKING`
- 可疑或失败 ID：9300518956, 206664979, A001471135

## 9. 是否可继续构建500例正式 mask

从 bbox 坐标复用角度可以继续，但不建议立即无复核地批量落盘。正式构建必须将 bbox 与 parsing skin 和 final mask 相交；同时应先确认额部修复掩膜是否允许覆盖可见头发，因为 `9300518956` 和 `206664979` 的辅助叠加仍保留了这类区域。该问题不影响本次坐标一致性结论，但会影响正式额部 mask 的内容纯度。

## 10. 下一步最小建议

先用 `9300518956`、`206664979` 和 `A001471135` 做额部修复掩膜策略的最小复核；确认可见头发是否应被保留后，再新增一个单独、可追溯的 mask 构建步骤：按包含式 bbox 转为右开切片，并与 `parsing_skin AND final_mask` 相交。不要重新对齐或重新解析源图像。
