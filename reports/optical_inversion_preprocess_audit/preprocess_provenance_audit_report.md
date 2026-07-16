# meanbg 与 ROI 预处理数据血缘及颜色变换审计

## 1. 完成状态

- 审计状态：`COMPLETE`
- 审计日期：2026-07-14
- 项目根目录：`E:\projects\face2`
- 审计方式：只读代码检查、目录与文件清点、ID 集合核对、SHA256 全量比较、解码后逐像素全量比较，以及 3 个固定 ID 的有限视觉核验。
- 本次只新增本报告；未修改代码，未修改、移动、删除或重新保存任何研究图像。
- 未训练模型，未运行分类或交叉验证，未读取或分析临床标签，未重新审计四个 EXIF 字段，也未执行过曝、欠曝、通道截断、噪声或模糊统计。

总体结论：

1. `hybrid_imagenet_meanbg` 的 E1 方案是“已对齐 RGB + hybrid final mask + ImageNet RGB 均值背景”，不是对整张图执行 ImageNet mean/std normalization。
2. E1 不执行新的旋转、裁剪或缩放，也不执行亮度、gamma、白平衡或颜色增强；但 11×11 羽化使人脸掩膜边界发生 alpha 混合，因此相对上游 aligned RGB 只能判为“部分颜色保持”。
3. 当前自动 ROI 成品不是从 meanbg 图像生成。配置指向的旧中间目录在磁盘上不存在，日志证明 500/500 例均从 `data\raw\images` 重新完成对齐、解析和 ROI 裁剪。
4. ROI 没有显式光度增强，但所有 ROI 都经过 `INTER_AREA` 缩放；masked 版本还会先把 global final face mask 外设为 RGB(0,0,0)。因此 ROI 内部也只能判为“部分颜色保持”。
5. `manual_shift_data` 当前文件是自动 ROI 输出的逐字节复制，没有发现重新保存、空间平移或其他像素变换。真实来源为：cheek=raw、eye=masked、forehead=raw、chin=masked、lip=masked。提示中“eye 使用 raw”的描述不符合磁盘证据。
6. Global ResNet18 可继续使用既有 meanbg 输入。光学反演分支不建议直接使用 meanbg 或当前 manual ROI；推荐从 meanbg 生成前的 saved aligned RGB 结合 parsing/final mask，按 ID 在线提取 forehead、图像左 cheek 和图像右 cheek。
7. 处理文件不保留 EXIF，但文件名 ID 与原图及既有 EXIF 审计表严格对应。500 个 meanbg/manual ID 全部能匹配 522 个唯一 EXIF ID。
8. 后续过曝、欠曝和通道截断检查结论为“只需做有限检查”：只在最终选定的有效 ROI 像素上检查，不应在包含黑色 padding/mask 或 mean background 的整张 canvas 上统计。

## 2. 实际检查的代码、配置和目录

### 2.1 代码与配置

- `preprocessing/build_global_face_preprocess_ablation_from_intermediates.py`
- `utils/preprocess_ablation_utils.py`
- `scripts/run/run_preprocess_global_face_hybrid_intermediates.py`
- `preprocessing/build_global_face_parsing_hybrid_foreheadrepair_blackbg_224_png_strict.py`
- `preprocessing/build_global_face_oval_blackbg_png_simalign_strict.py`
- `preprocessing/build_global_face_parsing_regularmask_blackbg_224_png_strict.py`
- `preprocessing/preprocess_global_aligned_face_parsing_roi_dataset_224_canvas.py`
- `config/preprocess/global_face_parsing_hybrid_foreheadrepair_blackbg_224_png_strict_intermediates.yaml`
- `config/preprocess/global_aligned_face_parsing_roi_final5_224_canvas.yaml`
- `dock/Face2_sync/各实验数据预处理思路.md`

项目中未找到 `project_inventory`、根目录 README 或 AGENTS.md。既有预处理说明主要记录较早方案以及 global hybrid 上游流程，没有完整记录本次 E1 和 Final5 ROI 的当前实现；因此以下结论以当前代码、当前配置和当前文件为准。

### 2.2 数据目录

- 原始图像：`E:\projects\face2\data\raw\images`
- meanbg 上游中间数据：`E:\projects\face2\data\processed\global_face\global_face_parsing_hybrid_foreheadrepair_blackbg_224_png_strict_intermediates`
- meanbg E1 成品：`E:\projects\face2\data\processed\global_face\preprocess_ablation\hybrid_imagenet_meanbg\images`
- 自动 ROI 成品：`E:\projects\face2\data\processed\roi_dataset\global_aligned_face_parsing_roi_final5_224_canvas_500`
- 实际训练 ROI：`E:\projects\face2\data\processed\roi_dataset\manual_shift_data`
- 既有 EXIF ID 依据：`E:\projects\face2\reports\exif_parameter_audit\image_parameter_audit.csv`，本次只读取 ID 列。

## 3. meanbg E1 的准确含义

E1 在 `VARIANT_SPECS` 中定义为：

- `name = hybrid_imagenet_meanbg`
- `bg_mode = imagenet_mean`
- `photometric_mode = none`
- `transform = _identity`

因此 E1 的准确含义是：读取已生成的 224×224 aligned RGB 和二值 final mask，不改变 aligned RGB 的整体光度，先对 mask 做羽化，再把掩膜外替换为 ImageNet RGB 均值背景。

ImageNet 只在背景常数中体现：

```text
IMAGENET_MEAN_RGB = [0.485, 0.456, 0.406] × 255
                   = [123.675, 116.280, 103.530]
```

纯背景写入 uint8 PNG 后为 RGB `[124, 116, 104]`。E1 没有使用 ImageNet 标准差，也没有把整张图保存为 `(RGB/255 - mean) / std` 的归一化张量。

上游 BiSeNet 人脸解析推理内部确实使用 ImageNet mean/std normalization，但该张量只供解析模型推理，输出是离散 parsing label map；它不会替换或归一化最终保存的 aligned RGB/meanbg RGB。

## 4. meanbg E1 数据流表

| stage | input_path | output_path | function_or_code_location | operation | spatial_change | color_change | encoding_change | ID_mapping |
|---|---|---|---|---|---|---|---|---|
| 1. 原图定位 | `data/raw/images/{ID}.{ext}` | 内存原图 | `find_image_for_id`、`read_image_rgb` | 按 ID 和支持扩展名查找；`cv2.imdecode(IMREAD_COLOR)` 后 BGR→RGB | 无显式几何变化 | 仅通道顺序转换；无反伽马 | JPEG/PNG 等被解码为 RGB uint8 | 原文件 stem 必须等于 ID |
| 2. 人脸检测和关键点 | 内存原始 RGB | 检测框、FaceMesh 点 | `detect_faces`、`select_face`、`run_facemesh` | 选择面积较大且居中的主脸；扩展框只用于 FaceMesh 定位 | 检测 crop 不作为最终像素来源 | 无 | 无 | 保持 ID |
| 3. 原图坐标相似对齐 | 内存原始 RGB + 5 点 | 224×224 `aligned_rgb` | `estimate_original_coordinate_alignment`、`estimate_similarity_transform` | `estimateAffinePartial2D(LMEDS)`；直接从原图 `warpAffine` 到模板 | 旋转、统一缩放和平移；无 shear、无独立 x/y 缩放；`INTER_LINEAR`；越界填黑 | 没有光度增强，但双线性插值会重采样 RGB | 内存 RGB uint8 | 保持 ID |
| 4. parsing 与 hybrid mask | `aligned_rgb` | `parsing_label`、`selected_semantic_mask`、`final_mask` | `run_face_parsing`、`build_selected_semantic_mask`、`build_semantic_regularized_mask`、`build_final_mask` | BiSeNet parsing；形态学闭运算、最大连通域、小孔填补、平滑；必要时额头局部修复 | 只生成 mask；parsing 输入 resize 用 `INTER_LINEAR`，label map 回缩用 `INTER_NEAREST` | ImageNet normalization 只进入 BiSeNet，不写回 RGB | label/mask 为单通道 PNG | 保持 ID |
| 5. 中间数据保存 | 内存 aligned/masks | `..._intermediates/aligned_rgb/{ID}.png`、`final_mask/{ID}.png` 等 | `_save_intermediates`、`save_png` | aligned RGB 和 mask 分别保存 | 无新增几何变化 | RGB 值不做增强 | PNG，压缩级别 3；无损；aligned 为 RGB uint8，mask 为二值单通道 | `{ID}.png` |
| 6. E1 读取 | `aligned_rgb/{ID}.png` + `final_mask/{ID}.png` | 内存 RGB/mask | `_process_one`、`read_rgb`、`read_mask` | PNG 解码；mask 二值化为 0/255 | 无 | BGR→RGB 仅恢复逻辑通道顺序 | PNG 解码到 uint8 | 保持 ID |
| 7. E1 背景合成 | 内存 aligned RGB + mask | 内存 meanbg RGB | `_identity`、`feather_mask`、`apply_background` | mask 使用 11×11 GaussianBlur，sigma≈2.912；`out = image×alpha + ImageNetMean×(1-alpha)` | 不旋转、不裁剪、不缩放 | 深部前景保持；掩膜边界与均值背景混合；外部为固定均值 | float32 计算后四舍五入/裁剪为 uint8 | 保持 ID |
| 8. E1 保存 | 内存 meanbg RGB | `hybrid_imagenet_meanbg/images/{ID}.png` | `save_rgb` | RGB→BGR 后 `cv2.imencode('.png', compression=3)` | 无 | 仅通道顺序转换用于编码 | 无损 PNG；不经过 JPEG；不保存 EXIF | `{ID}.png` |

## 5. E1 空间、颜色和编码操作逐项回答

1. E1 使用的最上游图像来自 `E:\projects\face2\data\raw\images`。
2. E1 的直接输入不是原图，而是 `..._intermediates\aligned_rgb` 和 `..._intermediates\final_mask`。
3. 它使用已经对齐到 224×224 模板的人脸。
4. E1 阶段本身不执行旋转、裁剪、仿射变换或缩放；这些发生在更上游的原图坐标相似对齐中。
5. 上游图像 warp 使用 `cv2.INTER_LINEAR`；越界区域使用 `BORDER_CONSTANT`、RGB(0,0,0)。
6. parsing 模型输入 resize 使用 `INTER_LINEAR`，离散标签图 resize 使用 `INTER_NEAREST`；这些只用于生成 mask。
7. 图像读取执行 BGR→RGB，保存执行 RGB→BGR。这是 OpenCV I/O 通道顺序转换，不是颜色校正。
8. 没有执行 sRGB 反伽马、线性 RGB 转换、ICC 转换或其他颜色管理。
9. E1 没有亮度、对比度、gamma、白平衡、直方图、CLAHE、Retinex、Lab-L 或其他颜色增强；这些属于其他 E2–E7 变体，不属于 E1。
10. `ImageNet` 在 E1 中只指背景填充值 `[0.485,0.456,0.406]×255`。它不表示最终 PNG 已做 ImageNet mean/std normalization。
11. 背景通过羽化 alpha 与 aligned RGB 逐像素线性混合后写入，而不是简单的硬边覆盖。
12. 距离 mask 边界足够远、alpha=1 的前景像素不会被 E1 修改；mask 边缘附近即使二值 mask=1，也会因 alpha<1 与背景混色。
13. 全量 500 例核验中，保守定义的 11×11 深部前景共有 9,626,612 个像素，meanbg 与 aligned RGB 9,626,612/9,626,612 完全一致；深部外部 12,229,960/12,229,960 均为 RGB `[124,116,104]`。
14. 11×11 边界带共有 3,231,428 个像素，其中 3,002,762 个相对 aligned RGB 发生变化；在二值 mask 内部共有 11,214,117 个像素，其中靠边界的 1,358,845 个发生变化。该变化来自预期的羽化混合，不是亮度增强。
15. 最终 500 张文件全部为 224×224、RGB、uint8、8-bit PNG，编码压缩级别为 3。
16. 没有 JPEG 重新编码。若最上游原图是 JPEG，只发生一次 JPEG 解码；中间和最终文件均为无损 PNG。
17. 最终 PNG 没有 EXIF、ICC profile、gAMA、sRGB chunk 或透明通道。它可以按常规 8-bit sRGB 编码值使用，但文件本身没有嵌入颜色配置文件。
18. 文件名从原图 stem/ID 传递为 `{ID}.png`。500 个 meanbg ID 全部存在于 522 个 raw ID、522 个 aligned ID 和 522 个 final-mask ID 中；上游多出的 22 个 ID 不属于当前 500 例 meanbg 集合。
19. 处理后图片不保留 EXIF，但可通过唯一 ID 与原图和既有 EXIF 表严格对应。

结论：`MEANBG_E1_COLOR_PRESERVING = PARTIAL`。相对 aligned RGB，深部人脸颜色逐像素保持，边界颜色不保持；相对原图还额外存在相似对齐的双线性插值。

## 6. ROI 的真实输入数据流

ROI 配置声明：

```text
image_dir: data/raw/images
global_intermediate_dir:
  data/processed/global_face/
  global_face_parsing_hybrid_foreheadrepair_blackbg_224_png_strict/intermediates
```

磁盘检查显示，配置声明的 `...strict\intermediates` 目录不存在。程序的 `load_intermediates` 因而对每例返回空，随后进入 `generate_intermediates`：从 `data\raw\images` 找到原图，重新执行人脸检测、原图坐标相似对齐、BiSeNet parsing、semantic mask 和 hybrid final mask。

`sample_global_status.csv` 的非临床来源字段显示：

- 记录数：500
- `global_intermediate_source = regenerated`：500
- `global_success = True`：500
- `input_path` 的文件 stem 与 ID 一致：500/500

配置中 `save_intermediates: false`，因此当前 ROI 输出目录下的 `intermediates_optional` 子目录为空。ROI 生成时实际使用的 aligned RGB、parsing label 和 final face mask 存在于当时内存中，没有作为该次运行的独立中间 PNG 保存。

因此：

- ROI 输入不是 meanbg。
- ROI 也不是从当前 `_intermediates\aligned_rgb` 目录直接读取。
- 当前 ROI 成品是从原始图像重新生成的 224×224 aligned RGB 中裁剪。
- 对齐算法与 global hybrid 使用同一代码，但 ROI 配置的 envelope 扩张参数为 0.15/0.04/0.02，而 meanbg 上游配置为 0.18/0.05/0.03；触发额头修复时 final mask 可能不完全相同。

## 7. 五类 ROI 的代码级定义

### 7.1 eye_roi

- 在 aligned 224 坐标系中联合使用 parsing 的左右眼 bbox 与 FaceMesh 眼部点。
- parsing 眉毛或眉毛 landmark 用于上边界约束；鼻部 parsing 用于限制下边界。
- 横向在眼部证据两侧各扩约 9% face width；纵向按 face height 约束最小/最大高度。
- 输出是双眼共同的一个矩形，不分别输出左右眼。
- raw 中可能包含眼睛、眶周皮肤、部分眉区以及矩形框内其他组织；它不是眼睛语义 mask。

### 7.2 lip_roi

- 合并 parsing 的 mouth/upper-lip/lower-lip bbox 与 FaceMesh 唇部点。
- bbox 横向扩 35%，纵向扩 60%。
- raw 是扩展矩形，除嘴唇外还包含口周皮肤，可能包含邻近鼻下或下颌区域。

### 7.3 cheek_roi

- 先由 final face mask 得到 face bbox。
- 垂直范围主要位于眼/眉下方到嘴唇上方，并受 face-height 回退范围约束。
- 图像左侧 bbox 横向约为 face width 的 0.08–0.37；图像右侧约为 0.63–0.92。
- 左右两块在代码中独立定义，500/500 例日志都保存了完整的 left/right bbox 坐标。
- 每一侧都被强制缩放到 112×224，不保持宽高比，然后按 `[图像左半, 图像右半]` 横向拼成 224×224。
- 当前单张 cheek PNG 因此包含可分割的两个半幅，但不是两个独立文件；“左/右”是图像 x 坐标方向，不能在未确认原图镜像约定前直接当作患者解剖学左/右。
- raw 是矩形内容，不是纯皮肤 mask；视觉核验可见中央拼接缝，并可能保留鼻翼外侧等邻近结构。

### 7.4 forehead_roi

- face bbox 和 skin bbox 定义上边界，眉毛 parsing/landmark 定义下边界。
- 横向在眉毛 bbox 两侧各扩约 13% face width；下边界位于眉上。
- raw 是矩形裁剪，不会自动删除矩形内的头发或背景。
- 3 个固定 ID 的视觉核验均显示额头皮肤，同时可包含发际线/头发；个别样例包含原始背景。输出上下黑色区域来自等比例缩放后的 canvas padding。

### 7.5 chin_roi

- 横向取 face bbox 中央约 20%–80%。
- 上边界在 mouth/lip bbox 下方；下边界受 face/skin bbox 以及 neck/cloth parsing 的上缘约束。
- raw 仍可能包含矩形内的邻近结构；masked 版用 global final face mask 去除 mask 外像素。

## 8. roi_raw、roi_masked 与颜色变化

`roi_raw` 的真实定义：

```text
aligned_rgb[bbox] → resize → 224×224 canvas
```

`roi_masked` 的真实定义：

```text
masked_full = aligned_rgb.copy()
masked_full[global_final_face_mask == 0] = RGB(0,0,0)
masked_full[bbox] → resize → 224×224 canvas
```

重要细节：`roi_target_mask` 当前忽略 `roi_type` 和 `label_map`，所有 ROI 的 masked 版本都使用同一个 global final face mask。解析类别用于 bbox 定义、有效性检查和 QC，不用于把 eye、lip、cheek、forehead、chin 精确雕刻成各自的专属语义 mask。因此：

- `roi_raw` 可包含矩形框内的 ROI 外皮肤、头发、眼睛、鼻部、背景或其他组织，具体取决于 ROI 和样本。
- `roi_masked` 只去除 global final face mask 外的像素；它不保证只剩目标 ROI 的语义类别。
- mask 外设置为 RGB(0,0,0)，不是 ImageNet mean background。

缩放和 canvas：

- eye、lip、forehead、chin：保持宽高比，用 `cv2.INTER_AREA` 缩放到能放入 224×224 的最大尺寸，居中放置，剩余区域填 RGB(0,0,0)。
- cheek：每侧不保持宽高比，分别用 `INTER_AREA` 强制缩放到 112×224，再横向拼接；没有额外 padding。
- 对 bbox 的扩张/边界调整由代码规则自动完成，没有发现人工平移参数。

颜色与编码：

- 没有亮度、对比度、gamma、白平衡、直方图、Lab、CLAHE、Retinex 或 ImageNet normalization。
- 矩形裁剪本身不改变像素值。
- `INTER_AREA` 重采样会改变 ROI 内部像素的逐点 RGB 值；masked 版本在 mask 边缘还会因黑色与前景共同参与缩放而产生混合。
- 输出由 `save_png` 保存为 RGB uint8 PNG，压缩级别 3；写入时 RGB→BGR，仅为 OpenCV 编码通道顺序。
- 不经过 JPEG 重新编码，不保留 EXIF 或颜色配置文件。

结论：`ROI_INTERNAL_COLOR_PRESERVING = PARTIAL`。无光度增强，但存在上游 `INTER_LINEAR` 对齐和 ROI `INTER_AREA` 缩放；masked 版本还存在置黑及边缘重采样。

## 9. ROI 数据流表

| roi_name | source_image | mask_source | crop_method | raw_definition | mask_definition | resize_method | canvas_method | background_value | color_change | output_path |
|---|---|---|---|---|---|---|---|---|---|---|
| eye_roi | 从 raw 原图重新生成的 224 aligned RGB | BiSeNet label 用于定位/QC；global hybrid final face mask 用于 masked | 解析眼 + FaceMesh 眼点联合矩形，眉/鼻约束 | aligned RGB 的双眼共同矩形 | global final face mask 外置黑；非专属 eye mask | `INTER_AREA`，保持宽高比 | 居中到 224×224，黑色 padding | RGB(0,0,0) | 无增强；对齐与 resize 重采样 | `roi_raw/eye_roi`、`roi_masked/eye_roi` |
| lip_roi | 同上 | mouth/lip parsing + FaceMesh 定位；global final mask 用于 masked | mouth/lip 联合 bbox，x 扩 35%、y 扩 60% | aligned RGB 扩展矩形 | global final face mask 外置黑；非专属 lip mask | `INTER_AREA`，保持宽高比 | 居中到 224×224，黑色 padding | RGB(0,0,0) | 无增强；resize 与 mask 边缘混合 | `roi_raw/lip_roi`、`roi_masked/lip_roi` |
| cheek_roi | 同上 | skin/眼眉/唇 parsing + final face mask + landmark 定位；global final mask 用于 masked | 图像左、右 cheek 两个矩形分别裁剪 | 两个 aligned RGB 矩形 | global final face mask 外置黑；非专属 cheek-skin mask | 每侧 `INTER_AREA` 强制到 112×224，不保持宽高比 | 左右横向拼接为 224×224 | 无 canvas padding；masked 外部为黑 | 无增强；两侧均非等比重采样 | `roi_raw/cheek_roi`、`roi_masked/cheek_roi` |
| forehead_roi | 同上 | skin、brow、final face mask + brow landmark 定位；global final mask 用于 masked | 眉上矩形，横向扩张 | aligned RGB 矩形，可含头发/背景 | global final face mask 外置黑；非专属 forehead-skin mask | `INTER_AREA`，保持宽高比 | 居中到 224×224，黑色 padding | RGB(0,0,0) | 无增强；resize 重采样 | `roi_raw/forehead_roi`、`roi_masked/forehead_roi` |
| chin_roi | 同上 | skin、mouth/lip、neck/cloth、final mask + landmark 定位；global final mask 用于 masked | 嘴唇下中央矩形，下缘受 neck/cloth 约束 | aligned RGB 矩形 | global final face mask 外置黑；非专属 chin-skin mask | `INTER_AREA`，保持宽高比 | 居中到 224×224，黑色 padding | RGB(0,0,0) | 无增强；resize 与 mask 边缘混合 | `roi_raw/chin_roi`、`roi_masked/chin_roi` |

## 10. manual_shift_data 目录、文件与 ID 核对

实际目录结构：

```text
manual_shift_data/
├── cheek_roi/       500 PNG
├── eye_roi/         500 PNG
├── forehead_roi/    500 PNG
├── chin_roi/        500 PNG
└── lip_roi/         500 PNG
```

核验结果：

- 每个目录均为 500 个 `.png` 文件。
- 每目录 500 个大小写不敏感唯一 ID，无重复文件名。
- 五个目录的 ID 集合完全一致：交集=500，全集=500。
- 五类 manual ID 与对应自动 `roi_raw` 和 `roi_masked` 的 ID 集合均完全一致。
- manual 500 ID 与 meanbg 500 ID 完全一致。
- manual/meanbg 500 ID 均能在 522 个唯一 raw ID 和 522 个唯一 EXIF 审计 ID 中找到；缺失=0。
- 2,500 个 manual 文件全部成功解码，为 PNG、RGB、224×224、3 通道、uint8、8-bit；EXIF 非空文件=0；ICC/gAMA/sRGB/透明信息均未写入。

## 11. manual_shift_data 来源的全量 SHA256 和像素核对

下表对每类 500 个 manual 文件分别与自动 raw/masked 同名文件比较。`平均绝对差`只在实际像素不一致的文件中，按全部 RGB 通道值计算。

| ROI | 对比来源 | SHA256 完全一致 | 解码像素完全一致 | 不一致文件 | 最大绝对像素差 | 不一致文件平均绝对差 |
|---|---:|---:|---:|---:|---:|---:|
| cheek | roi_raw | 500/500 | 500/500 | 0 | 0 | 0 |
| cheek | roi_masked | 438/500 | 438/500 | 62 | 255 | 0.53495042 |
| eye | roi_raw | 400/500 | 400/500 | 100 | 242 | 0.25042025 |
| eye | roi_masked | 500/500 | 500/500 | 0 | 0 | 0 |
| forehead | roi_raw | 500/500 | 500/500 | 0 | 0 | 0 |
| forehead | roi_masked | 0/500 | 0/500 | 500 | 255 | 9.18488904 |
| chin | roi_raw | 3/500 | 3/500 | 497 | 255 | 5.86467923 |
| chin | roi_masked | 500/500 | 500/500 | 0 | 0 | 0 |
| lip | roi_raw | 24/500 | 24/500 | 476 | 255 | 2.69571706 |
| lip | roi_masked | 500/500 | 500/500 | 0 | 0 | 0 |

最终来源判定：

| manual 目录 | 经磁盘验证的真实来源 | 结论 |
|---|---|---|
| `cheek_roi` | `roi_raw/cheek_roi` | 500/500 字节和像素完全一致 |
| `eye_roi` | `roi_masked/eye_roi` | 500/500 字节和像素完全一致；并非提示所述 raw |
| `forehead_roi` | `roi_raw/forehead_roi` | 500/500 字节和像素完全一致 |
| `chin_roi` | `roi_masked/chin_roi` | 500/500 字节和像素完全一致 |
| `lip_roi` | `roi_masked/lip_roi` | 500/500 字节和像素完全一致 |

由于选定来源的 SHA256 全部一致，当前 `manual_shift_data` 是简单文件复制：没有重新保存、没有压缩变化、没有空间平移、没有颜色变化，也没有其他像素变换。仓库中未找到创建该目录或描述人工操作步骤的代码/记录，因此“由谁、何时、通过什么人工流程选择并复制”无法从项目确定；不得根据目录名推断。`MANUAL_SHIFT_SOURCE_VERIFIED = YES`，但原先 eye=raw 的口头规则应修正为 eye=masked。

## 12. 有限视觉核验

固定且可复现地选择排序后的首个、中间和末尾 ID：

- `100037382`
- `A001620289`
- `A002366013`

核验观察：

1. meanbg 的人脸中央颜色视觉上与 aligned RGB 一致，脸缘存在连续的均值背景混合带，符合 11×11 羽化代码。
2. forehead raw 是眉上/额头矩形，经等比例缩放后置于黑色 canvas；三个样例都包含额头皮肤，也可包含发际线或头发，个别样例可见少量原始背景。
3. cheek raw 中央存在明确拼接缝，左、右图像半幅分别来自两个 cheek bbox；样例中可见鼻翼外侧等邻近结构，证明它不是严格的“纯脸颊皮肤 mask”。
4. 视觉结果与代码和全量哈希/像素结论一致，没有据此开展任何曝光、截断、噪声或模糊统计。

## 13. Global 分支推荐输入

推荐：继续使用现有目录作为 Global ResNet18 输入：

```text
E:\projects\face2\data\processed\global_face\preprocess_ablation\hybrid_imagenet_meanbg\images
```

理由：

- 这是既有 Global 模型配置明确使用的数据分布。
- E1 不做整图光度增强，深部人脸颜色相对 aligned RGB 保持。
- 固定均值背景可以抑制背景、衣物和拍摄环境信息。
- 对 Global 分类分支，边界羽化和固定背景是预期设计，而不是错误。

该建议只适用于保持既有分类模型的数据一致性，不表示 meanbg 是最理想的光学反演输入。

## 14. 光学反演分支推荐输入

V1 推荐使用：

```text
主图像：
E:\projects\face2\data\processed\global_face\global_face_parsing_hybrid_foreheadrepair_blackbg_224_png_strict_intermediates\aligned_rgb

解析标签：
E:\projects\face2\data\processed\global_face\global_face_parsing_hybrid_foreheadrepair_blackbg_224_png_strict_intermediates\parsing_label

人脸 mask：
E:\projects\face2\data\processed\global_face\global_face_parsing_hybrid_foreheadrepair_blackbg_224_png_strict_intermediates\final_mask
```

在加载时按同一 ID 结合 parsing/final mask，直接在 aligned 224 坐标系中提取 forehead、图像左 cheek 和图像右 cheek 的有效像素。建议在 native bbox/mask 像素上计算光学量，不先缩放到 224×224 ROI canvas；同时排除 mask 羽化边界、黑色 padding 和人工背景。

选择该来源的原因：

- 比 meanbg 少一次与 ImageNet 均值背景的边界混色。
- 比当前 manual ROI 少一次 `INTER_AREA` 重采样。
- 不受黑色 canvas padding 影响。
- 可以从同一 aligned RGB 中分别提取左右 cheek，不必从已拉伸的拼接图反推。
- 522 个 aligned/mask/parsing ID 完整存在，500 个研究 ID 均可严格匹配。

不优先推荐：

- 当前 meanbg：深部前景可用，但边界被固定均值背景混色，整图背景是人工构造。
- 当前 manual ROI：已重采样；forehead raw 可含头发/背景；cheek 两侧被非等比拉伸并拼接；eye 的实际来源还与既有描述不一致。
- 原自动 ROI 的 raw/masked 成品：同样已经缩放，masked 也不是 ROI 专属语义 mask。

局限：saved aligned RGB 本身已经经过原图坐标 `INTER_LINEAR` warp，并非相机原始像素。若后续研究要求最严格的相机像素物理解释，最终应把 ROI/mask 映射回原始照片坐标或在原图上取样；本次不实现该新流程。对当前 V1，saved aligned RGB + mask 是可复现性、现有数据完整性和颜色保真之间的最小可行选择。

## 15. forehead 与左右 cheek 的可获得性

### Forehead

- 当前 manual/auto raw 中已有 forehead 矩形图，可以直接用于既有分类模型。
- 但它不等于纯额头皮肤，可能含发际线、头发和背景，并且经过 resize/padding。
- 光学 V1 应从 aligned RGB 结合 parsing/final mask 重新在线选择额头有效像素。

### Left cheek / Right cheek

- 代码内部确实分别计算了图像左、图像右 cheek bbox，日志 500/500 均保留两侧坐标。
- 当前 cheek PNG 的 `x=0:112` 和 `x=112:224` 分别为两侧固定半幅，因此信息层面可以拆分。
- 但它们没有作为独立原始尺度文件保存；每侧都被非等比拉伸到 112×224，且代码的 left/right 是图像坐标，不保证等同于患者解剖学侧别。

因此 `LEFT_RIGHT_CHEEK_AVAILABLE = PARTIAL`：两侧在现有拼接图中可分割、原始 bbox 也可追溯，但不建议把已拉伸半幅作为光学反演的最终左右 cheek 数据。应从 aligned RGB 按日志或同一规则分别提取 native 左右 bbox，并先明确镜像/解剖学侧别约定。

## 16. EXIF 对应关系

处理后的 meanbg 和 ROI PNG 都不保留 EXIF。磁盘核验结果为：

- raw 图像：522 个文件、522 个大小写不敏感唯一 stem/ID。
- meanbg：500 个唯一 ID，全部存在于 raw/aligned/final-mask 集合。
- manual 五类 ROI：共同的 500 个唯一 ID，与 meanbg 500 ID 完全一致。
- 既有 `image_parameter_audit.csv`：522 个唯一 EXIF ID。
- meanbg 缺失于 EXIF 表：0。
- manual 缺失于 EXIF 表：0。
- ROI 日志 `input_path` stem 与 ID：500/500 一致。

因此，只要下游连接使用不带扩展名的完整字符串 ID，不做数值化、不截断前导字符、不去除合法字母前缀，处理图像与原图 EXIF 可以严格一一匹配。`EXIF_ID_LINKAGE_VALID = YES`。

## 17. 是否仍需过曝、欠曝和通道截断统计

结论：`CLIPPING_AUDIT_DECISION = LIMITED`，即“只需做有限检查”。

原因：

1. 当前 E1 和 ROI 代码没有亮度、gamma、白平衡或直方图增强，不存在需要针对这些增强步骤做全量专项审计的证据。
2. meanbg 整图含固定 RGB `[124,116,104]` 背景；manual ROI 整图含大量 RGB(0,0,0) mask/padding。若直接对整张 224×224 图统计欠曝或通道 0，会把人工背景错误计为图像曝光问题。
3. 光学反演仍可能受原始采集阶段的真实饱和/截断影响，因此完全跳过检查也不稳妥。
4. 后续只应在最终选定的 aligned RGB 有效 forehead/left-cheek/right-cheek mask 内、排除 mask 边界和 padding 后，做一次有限的 acquisition-level 饱和/截断检查。

本次没有实际计算任何过曝、欠曝或通道截断比例。

## 18. 发现的问题和不确定项

1. ROI 配置中的 `global_intermediate_dir` 指向不存在的旧目录，导致 500 例全部重新生成；这与“复用已保存 global intermediates”的代码注释不一致。
2. ROI 的 `save_intermediates=false`，该次运行的 exact aligned RGB、parsing label 和 final face mask 未独立保存。当前 ROI 成品可核验，但当时内存 mask 无法从该输出目录原样恢复。
3. ROI 配置与 meanbg 上游配置的 forehead/side/chin envelope 扩张参数不同，因此不能默认两套 final mask 逐像素相同。
4. 提示中的 eye=roi_raw 规则不成立；磁盘证据为 eye=roi_masked。
5. `manual_shift_data` 没有对应生成脚本或人工操作记录。最终文件能确定为简单复制，但人工选择过程无法确定。
6. `roi_masked` 不是 ROI 专属语义 mask，而是 global final face mask；目录名容易造成误解。
7. cheek 的 left/right 是图像坐标，两侧被强制拉伸后拼接；解剖学侧别和镜像约定尚未在代码或元数据中明确。
8. 输出 PNG 没有嵌入 ICC/sRGB/gAMA 信息。代码将数值按普通 RGB/sRGB 图像处理，但严格颜色管理链未显式记录。
9. 原图读取没有显式的 EXIF orientation 处理分支；代码依赖 `cv2.imdecode` 当前构建的解码行为。若存在方向标记异常，应另行做元数据级方向核对，但不属于本次四字段或颜色审计。

## 19. 下一步最小建议

1. Global ResNet18 保持使用现有 `hybrid_imagenet_meanbg/images`，不改变既有输入分布。
2. 光学 V1 新建独立的数据加载约定：按 ID 同时读取 `_intermediates/aligned_rgb`、`parsing_label` 和 `final_mask`，在线得到 forehead、图像左 cheek、图像右 cheek 的 native 有效像素；本次不实现程序。
3. 在正式建模前明确图像左/右与患者解剖学左/右的映射，并记录是否镜像。
4. 不直接复用 current manual cheek 拼接半幅做物理反演；避免非等比拉伸带来的空间与像素混合。
5. 仅对最终有效 ROI mask 内做一次有限的饱和/截断检查，排除所有背景、padding 和 mask 边界。
6. 在数据说明中修正 provenance：eye 应记录为 `roi_masked`；`manual_shift_data` 当前没有实际 shift 证据。

## 20. 最终状态字段

```text
AUDIT_STATUS = COMPLETE
MEANBG_E1_COLOR_PRESERVING = PARTIAL
ROI_INTERNAL_COLOR_PRESERVING = PARTIAL
MANUAL_SHIFT_SOURCE_VERIFIED = YES
RECOMMENDED_GLOBAL_INPUT = E:\projects\face2\data\processed\global_face\preprocess_ablation\hybrid_imagenet_meanbg\images
RECOMMENDED_OPTICAL_INPUT = E:\projects\face2\data\processed\global_face\global_face_parsing_hybrid_foreheadrepair_blackbg_224_png_strict_intermediates\aligned_rgb + parsing_label + final_mask（按ID在线提取有效ROI）
LEFT_RIGHT_CHEEK_AVAILABLE = PARTIAL
EXIF_ID_LINKAGE_VALID = YES
CLIPPING_AUDIT_DECISION = LIMITED
REPORT_PATH = E:\projects\face2\reports\optical_inversion_preprocess_audit\preprocess_provenance_audit_report.md
```
