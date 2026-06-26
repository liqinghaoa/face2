你现在需要继续修改 face2 项目中的 ROI 数据预处理程序。当前 ROI 预处理程序已经可以运行并生成 QC 图，但 pilot 结果显示当前 ROI 定义仍存在问题，需要按照新的 Final5 ROI 方案重构 ROI 定义与输出逻辑。
请先完整分析当前已有代码，再实现修改。不要破坏全局人脸预处理流程，不要覆盖旧 ROI 输出目录，不要修改训练代码。
一、项目背景
当前项目结构大致如下：
face2:
--|config
--|data
--|--|processed
--|--|--|splits
--|--|raw
--|--|--|images
--|datasets
--|dock
--|evaluators
--|losses
--|metrics
--|models
--|preprocessing
--|scripts
--|trainers
--|utils
当前已有全局人脸预处理流程，核心为：
原图坐标系 direct warp alignment
face parsing
selected semantics
hybrid forehead repair
final face mask
black background PNG
strict QC
ROI 数据预处理必须基于该全局人脸预处理流程的中间结果，而不是直接从原图坐标系裁剪 ROI，也不是从最终 black-bg PNG 上裁剪 ROI。
二、本次实现目标
请将当前 ROI 数据预处理程序更新为 Final5 ROI 版本。
建议数据版本名：
global_aligned_face_parsing_roi_final5_224_canvas
建议输出目录：
data/processed/roi_dataset/global_aligned_face_parsing_roi_final5_224_canvas
Final5 ROI 固定包含五个 core ROI：
eye_roi
lip_roi
cheek_roi
forehead_roi
chin_roi
不再默认输出：
nose_roi
midface_roi
cheek_pair_roi
left_cheek_roi
right_cheek_roi
说明：
cheek_roi 的含义是 left cheek + right cheek 横向拼接后的单一 ROI；
程序内部可以临时裁剪 left cheek 和 right cheek；
但最终只保存 cheek_roi，不保存 left_cheek_roi / right_cheek_roi；
chin_roi 新增并进入 core ROI；
nose_roi 取消，不参与本版输出；
midface_roi 取消，不参与本版输出；
forehead_roi 当前效果较好，本次不要再修改 forehead 的 bbox 逻辑，只迁移到 Final5 输出体系。
三、输入与中间结果
默认输入：
--project-root .
--split-csv data/processed/splits/extreme_5fold.csv
--image-dir data/raw/images
--global-intermediate-dir data/processed/global_face/global_face_parsing_hybrid_foreheadrepair_blackbg_224_png_strict/intermediates
--output-dir data/processed/roi_dataset/global_aligned_face_parsing_roi_final5_224_canvas
从 split csv 中读取 ID。若表中存在以下字段，请保留到输出日志中：
ID
NYHA
extreme_label
label
fold
SEX
优先读取全局预处理中间结果：
aligned_rgb/{ID}.png
parsing_label/{ID}.png
selected_semantic_mask/{ID}.png
final_face_mask/{ID}.png
这些中间结果用途如下：
aligned_rgb：
用于裁剪 raw ROI。
parsing_label：
用于统计每个 ROI 内的语义比例，例如 skin、hair、background、neck、cloth、eye、brow、nose、mouth、lip 等。
selected_semantic_mask：
用于辅助判断 ROI 是否位于有效人脸主体区域。
final_face_mask：
用于生成 masked ROI。
如果某个 ID 的中间结果缺失，可以选择调用已有全局人脸预处理流程重新生成，但禁止退回旧的“原图坐标系直接裁剪 ROI”方案。
四、总体处理流程
Final5 ROI 主流程如下：
读取 extreme_5fold.csv；
逐个读取 ID；
读取 aligned_rgb、parsing_label、selected_semantic_mask、final_face_mask；
基于 final_face_mask / selected_semantic_mask / parsing_label 计算 face bbox；
基于 parsing_label 或 aligned 坐标系中的 landmarks 计算 eye、mouth/lip、brow、skin 等辅助 bbox；
在 aligned 224×224 坐标系中定义五个 ROI；
从 aligned_rgb 裁剪 raw ROI；
从 aligned_rgb 与 final_face_mask 的交集生成 masked ROI；
对每个 ROI 进行 resize / padding；
保存 roi_raw 和 roi_masked；
计算每个 ROI 的 QC 指标；
写 roi_metadata.csv；
写 roi_qc_metrics.csv；
写 roi_success_matrix.csv；
写 multi_roi_valid_ids.csv；
写 failed_cases.csv；
生成 QC 图；
输出 preprocess_summary.txt。
注意：
预处理阶段只保存普通 uint8 RGB PNG；
不执行 ImageNet Normalize；
训练阶段再执行 ToTensor 和 ImageNet mean/std Normalize；
ROI 坐标必须在 aligned 224×224 坐标系中定义；
不要从最终 black-bg PNG 上裁 ROI；
不要回到原图坐标系直接裁 ROI。
五、输出尺寸规则
所有最终 ROI 均输出：
224×224
RGB
PNG
uint8
普通 ROI 使用：
ROI crop
→ 保持长宽比例 resize
→ 居中 padding 到 224×224 canvas
→ padding 颜色为黑色 RGB(0,0,0)
→ 保存 PNG
普通 ROI 包括：
eye_roi
lip_roi
forehead_roi
chin_roi
cheek_roi 是特殊拼接型 ROI，建议使用：
left cheek crop → resize 到 112×224
right cheek crop → resize 到 112×224
horizontal concat → 224×224 cheek_roi
这样可以保证左右脸颊尺寸一致，避免左右 cheek 大小不一致。
请在日志中记录：
roi_resize_mode
普通 ROI：
roi_resize_mode = keep_aspect_pad_canvas
cheek_roi：
roi_resize_mode = pair_concat_fixed_half
六、ROI 输出目录结构
默认输出目录：
data/processed/roi_dataset/global_aligned_face_parsing_roi_final5_224_canvas
目录结构：
global_aligned_face_parsing_roi_final5_224_canvas/
roi_raw/
eye_roi/
{ID}.png
lip_roi/
{ID}.png
cheek_roi/
{ID}.png
forehead_roi/
{ID}.png
chin_roi/
{ID}.png
roi_masked/
eye_roi/
{ID}.png
lip_roi/
{ID}.png
cheek_roi/
{ID}.png
forehead_roi/
{ID}.png
chin_roi/
{ID}.png
logs/
roi_metadata.csv
roi_qc_metrics.csv
roi_success_matrix.csv
multi_roi_valid_ids.csv
failed_cases.csv
preprocess_summary.txt
qc_preview/
random_success/
core_all_success/
roi_failure/
high_padding_ratio/
eye_roi_warning/
lip_roi_warning/
cheek_roi_warning/
forehead_roi_warning/
chin_roi_warning/
不要创建正式输出目录：
nose_roi/
midface_roi/
cheek_pair_roi/
left_cheek_roi/
right_cheek_roi/
七、raw ROI 与 masked ROI 定义
每个 ROI 都必须输出两层版本。
第一层：raw ROI
路径：
roi_raw/{roi_type}/{ID}.png
定义：
直接从 aligned_rgb 中按 ROI bbox 裁剪；
保留局部原始视觉内容；
再按对应规则 resize / padding 到 224×224。
第二层：masked ROI
路径：
roi_masked/{roi_type}/{ID}.png
第一版 masked ROI 定义固定为：
masked ROI = raw ROI 中位于 global_final_face_mask 内的像素保留；
raw ROI 中位于 global_final_face_mask 外的像素置黑。
等价于：
masked_roi = aligned_rgb ∩ final_face_mask ∩ roi_bbox
重要要求：
不使用 ROI-specific strict semantic mask；
不在 eye_roi 内把 nose 类挖黑；
不在 lip_roi 内把 skin 类挖黑；
不在 forehead_roi 内逐像素挖掉 hair；
不在 cheek_roi 或 chin_roi 内制造内部语义黑洞；
parsing_label 只用于 QC 统计，不用于默认 masked ROI 的内部挖空。
如果后续需要严格语义 masked ROI，可以另做 optional 第三层版本 roi_semantic_masked，但不要作为本次默认输出。
八、Final5 ROI 具体定义
8.1 eye_roi
eye_roi 定义为：
眉毛下缘附近 + 上眼睑 + 眼睛 + 下眼睑 + 眼袋/泪沟区域。
目标保留：
眼裂
上眼睑
下眼睑
眼袋
泪沟区域
眶下软组织
内外眦周围皮肤
眉毛下缘附近少量区域
不应明显包含：
完整眉毛
大面积额头
鼻尖
鼻翼
大面积鼻梁
中面部
关键原则：
eye_roi 不能再裁成一条细线；
eye_roi 不能只裁 eye parsing 类别；
eye_roi 是眶周功能区，不是眼睛语义区域 bbox；
eye_roi 应由 landmarks / 几何规则主导，parsing_label 只作为 QC；
需要设置最小高度保护，避免过度收缩。
推荐实现思路：
以左右眼 landmarks 或 eye parsing bbox 为中心；
上边界到眉毛下缘附近或上眼睑上方少量皮肤；
下边界覆盖眼袋 / 泪沟区域；
横向覆盖双眼和眼角周围皮肤。
可参考：
eye_x1 = 双眼 bbox 左边界 - 0.08~0.10 * face_w
eye_x2 = 双眼 bbox 右边界 + 0.08~0.10 * face_w
eye_y1 = eye_bbox_y1 - 0.04~0.06 * face_h
eye_y2 = eye_bbox_y2 + 0.10~0.14 * face_h
如果 brow 信息可靠：
上边界可限制在 brow 下缘附近，但不要上移到完整眉毛以上。
如果 nose_y_min 可靠：
eye_y2 = min(eye_y2, nose_y_min + 0.03 * face_h)
同时加入高度保护：
eye_roi_height_min = 0.16 * face_h
eye_roi_height_max = 0.30 * face_h
如果 eye_roi 高度小于最小高度，则围绕眼部中心向上下扩展；
如果大于最大高度，则适当收缩，但必须保留眼袋/泪沟。
warning：
warning_eye_too_narrow
warning_eye_contains_nose
warning_eye_too_high
warning_eye_too_low
warning_eye_missed
8.2 lip_roi
lip_roi 定义为：
上唇 + 下唇 + 嘴角 + 口周皮肤 + 鼻唇沟下部。
目标保留：
上唇
下唇
嘴角
口周皮肤
鼻唇沟下部
下唇下方少量皮肤
不应只裁红唇，也不应裁到过多下巴或脖子。
推荐实现：
以 mouth / upper_lip / lower_lip bbox 或嘴部 landmarks 为基础；
左右扩展到嘴角外侧；
上方扩展到鼻唇沟下部；
下方扩展到下唇下方少量皮肤。
可参考：
lip_x1 = mouth_bbox_x1 - 0.18~0.25 * mouth_w
lip_x2 = mouth_bbox_x2 + 0.18~0.25 * mouth_w
lip_y1 = mouth_bbox_y1 - 0.30~0.45 * mouth_h
lip_y2 = mouth_bbox_y2 + 0.45~0.70 * mouth_h
或基于 face_h 做保护：
lip_y1 不应上移到鼻子主体过多；
lip_y2 不应下移到大面积 chin/neck。
warning：
warning_lip_missed
warning_low_mouth_lip_ratio
warning_lip_contains_neck_or_cloth
warning_lip_too_small
8.3 cheek_roi
cheek_roi 定义为：
left cheek + right cheek 横向拼接。
这是 Final5 中唯一的拼接型 ROI。
目标保留：
左脸颊
右脸颊
鼻翼外侧到口角外侧之间的皮肤
中下脸颊皮肤颜色和纹理
软组织状态
不应明显包含：
鼻子主体过多
眼睛
嘴唇主体
耳朵
脖子
病服
大面积背景
实现方式：
内部临时计算 left_cheek_bbox；
内部临时计算 right_cheek_bbox；
从 aligned_rgb 分别裁剪左右脸颊 raw crop；
从 aligned_rgb 与 final_face_mask 交集分别裁剪左右脸颊 masked crop；
left cheek resize 到 112×224；
right cheek resize 到 112×224；
横向拼接为 224×224 cheek_roi；
只保存 cheek_roi；
不保存 left_cheek_roi / right_cheek_roi 单独文件。
建议尽量恢复旧 ROI 中较稳定的 cheek 裁剪思路，但要在 aligned 224×224 坐标系中实现。
推荐使用 face width 比例定义左右 cheek，避免过度依赖不稳定的 nose bbox：
left_cheek:
x1 = face_x_min + 0.10 * face_w
x2 = face_x_min + 0.42 * face_w
right_cheek:
x1 = face_x_min + 0.58 * face_w
x2 = face_x_min + 0.90 * face_w
y 范围：
cheek_y1 = eye_y_bottom + 0.03~0.05 * face_h
cheek_y2 = mouth_y_top 或 lip_y_top - 0.02 * face_h
如果 mouth_y_top 不可靠，可使用 face bbox 比例：
cheek_y1 = face_y_min + 0.35 * face_h
cheek_y2 = face_y_min + 0.70 * face_h
需要保证左右 cheek 的 y1/y2 一致，避免左右高度不一致。
日志中记录：
left_cheek_bbox_x1
left_cheek_bbox_y1
left_cheek_bbox_x2
left_cheek_bbox_y2
right_cheek_bbox_x1
right_cheek_bbox_y1
right_cheek_bbox_x2
right_cheek_bbox_y2
warning：
warning_cheek_low_skin_ratio
warning_cheek_background_contamination
warning_cheek_contains_mouth
warning_cheek_contains_eye
warning_cheek_too_narrow
warning_cheek_pair_asymmetry
8.4 forehead_roi
forehead_roi 当前 pilot 效果较好，本次不要修改其 bbox 逻辑。
请保持现有 forehead_roi 定义与实现。
forehead_roi 应继续保留：
发际线附近
下额头至中额头
眉毛上方区域
主要额头皮肤
本次只需要确保：
forehead_roi 纳入 Final5 输出；
同时输出 raw 和 masked；
记录 QC 指标；
参与 core_all_success；
QC 图正常显示。
不要重新调整 forehead 的上边界、下边界或宽度。
warning 保留：
warning_forehead_hair_occlusion
warning_forehead_too_small
warning_forehead_background_contamination
warning_forehead_low_skin_ratio
8.5 chin_roi
chin_roi 是新增 core ROI，定义为：
下唇以下区域 + 下巴 + 下颌线附近。
目标保留：
下唇以下皮肤
下巴主体
下颌线附近皮肤
下脸部软组织状态
不应明显包含：
嘴唇主体过多
脖子过多
衣领 / 病服
大面积背景
推荐实现：
基于 mouth/lip bbox、face bbox、final_face_mask 下边界、skin region 和 neck/cloth parsing 约束。
可参考：
chin_y1 = lower_lip_y_max + 0.02 * face_h
chin_y2 = face_y_max - 0.02 * face_h
chin_x1 = face_x_min + 0.18~0.22 * face_w
chin_x2 = face_x_max - 0.18~0.22 * face_w
如果 face_y_max 包含 neck/cloth，应使用 final_face_mask 或 skin parsing 限制下边界，避免裁到大面积脖子和病服。
如果 neck parsing ratio 或 cloth parsing ratio 过高，应 warning。
可选稳健逻辑：
找到 lip/mouth bbox；
找到 final_face_mask 的下边界；
在 lower_lip_y_max 到 face_mask_bottom 之间定义 chin 区域；
横向覆盖中下脸和下颌线附近；
不要过窄，不要只裁下巴尖。
warning：
warning_chin_contains_mouth
warning_chin_contains_neck
warning_chin_contains_cloth
warning_chin_low_skin_ratio
warning_chin_too_small
warning_chin_high_background_ratio
九、core_all_success 定义
Final5 中所有 ROI 都是 core ROI：
eye_roi
lip_roi
cheek_roi
forehead_roi
chin_roi
因此：
core_all_success =
eye_roi_success
AND lip_roi_success
AND cheek_roi_success
AND forehead_roi_success
AND chin_roi_success
multi_roi_valid_ids.csv 只保留 core_all_success == true 的样本。
十、日志文件要求
roi_metadata.csv 至少包含：
ID
fold
label
extreme_label
NYHA
SEX
roi_type
roi_raw_path
roi_masked_path
roi_success
failure_reason
roi_qc_metrics.csv 为 ROI 级记录，至少包含：
ID
fold
label
extreme_label
NYHA
SEX
roi_type
roi_raw_path
roi_masked_path
bbox_x1
bbox_y1
bbox_x2
bbox_y2
bbox_w
bbox_h
canvas_size
content_w
content_h
padding_top
padding_bottom
padding_left
padding_right
padding_ratio
content_ratio
roi_resize_mode
skin_ratio
hair_ratio
background_ratio
neck_ratio
cloth_ratio
ear_ratio
eye_ratio
brow_ratio
nose_ratio
mouth_ratio
lip_ratio
roi_success
roi_valid_flag
roi_warning_flag
failure_reason
对 cheek_roi，额外记录：
left_cheek_bbox_x1
left_cheek_bbox_y1
left_cheek_bbox_x2
left_cheek_bbox_y2
right_cheek_bbox_x1
right_cheek_bbox_y1
right_cheek_bbox_x2
right_cheek_bbox_y2
对 chin_roi，额外记录：
chin_contains_neck_ratio
chin_contains_cloth_ratio
chin_mouth_lip_ratio
roi_success_matrix.csv 至少包含：
ID
fold
label
extreme_label
NYHA
SEX
eye_roi_success
lip_roi_success
cheek_roi_success
forehead_roi_success
chin_roi_success
core_all_success
failed_roi_list
warning_roi_list
multi_roi_valid_ids.csv 只保留 core_all_success == true 的样本，并包含：
ID
fold
label
extreme_label
NYHA
SEX
eye_roi_raw_path
lip_roi_raw_path
cheek_roi_raw_path
forehead_roi_raw_path
chin_roi_raw_path
eye_roi_masked_path
lip_roi_masked_path
cheek_roi_masked_path
forehead_roi_masked_path
chin_roi_masked_path
十一、失败判定
以下情况判定 ROI 失败：
bbox 为空；
bbox 宽或高过小；
bbox 修正后仍无有效区域；
ROI crop 失败；
resize/padding 失败；
保存失败；
目标语义区域几乎缺失；
对应 aligned_rgb、parsing_label 或 final_face_mask 缺失；
全局预处理中间结果不可用且无法重建。
某个 ROI 失败不影响其他 ROI 输出，但会影响 core_all_success。
如果全局中间结果失败，则该样本所有 ROI 失败。
warning 只记录，不自动删除，除非 ROI 明显失败。
十二、QC 图要求
QC 图必须更新为 Final5 ROI。
必须显示：
original，如果可用
aligned_rgb
parsing_label
global_final_mask
ROI bbox overlay
raw eye_roi
raw lip_roi
raw cheek_roi
raw forehead_roi
raw chin_roi
masked eye_roi
masked lip_roi
masked cheek_roi
masked forehead_roi
masked chin_roi
不再显示：
nose_roi
midface_roi
cheek_pair_roi
QC 标题显示：
ID
fold
label 或 extreme_label
core_all_success
failed_roi_list
warning_roi_list
core_roi_types = eye, lip, cheek, forehead, chin
重点检查：
eye_roi 是否完整覆盖眼周、眼袋和泪沟，不再变成细线；
eye_roi 是否没有裁到鼻子主体；
lip_roi 是否包含嘴唇、嘴角、口周和鼻唇沟下部；
cheek_roi 是否左右脸颊大小一致、覆盖充分；
cheek_roi 是否接近旧 ROI 的脸颊裁剪效果；
forehead_roi 是否保持当前良好效果；
chin_roi 是否包含下巴和下颌线，而不是脖子/病服；
masked ROI 是否没有内部语义黑洞；
padding 是否过多；
是否残留明显背景、病服、脖子。
十三、preprocess_summary.txt 要求
summary 中必须包含：
总样本数
读取全局中间结果成功数
各 ROI 成功数
core_all_success 样本数
每个 fold 的 core_all_success 样本数
每个 label/extreme_label 的 core_all_success 样本数
每个 SEX 的 core_all_success 样本数
每个 ROI 的失败原因统计
每个 ROI 的 warning 数量统计
eye_roi warning 数量
lip_roi warning 数量
cheek_roi warning 数量
forehead_roi warning 数量
chin_roi warning 数量
高 padding ratio 数量
eye_contains_nose 数量
cheek_pair_asymmetry 数量
chin_contains_neck/cloth 数量
十四、argparse 参数
请支持或更新以下参数：
--project-root
--split-csv
--image-dir
--global-intermediate-dir
--output-dir
--image-size 默认 224
--canvas-size 默认 224
--padding-color 默认 0,0,0
--output-format png
--roi-types 默认 eye_roi,lip_roi,cheek_roi,forehead_roi,chin_roi
--core-roi-types 默认 eye_roi,lip_roi,cheek_roi,forehead_roi,chin_roi
--keep-aspect-ratio true
--overwrite
--max-samples
--seed 默认 42
--num-qc-preview 默认 20
warning 阈值：
--max-padding-ratio 默认 0.60
--max-background-ratio 默认 0.20
--max-cloth-ratio 默认 0.05
--max-neck-ratio 默认 0.05
--eye-nose-warning 默认 0.10
--cheek-background-warning 默认 0.15
--chin-neck-warning 默认 0.10
--chin-cloth-warning 默认 0.05
如果脚本仍支持全局重建或 parsing checkpoint，请保留：
--parsing-model
--parsing-checkpoint
--parsing-device
如果 parsing checkpoint 不存在，直接报错，不要联网下载，不要随机初始化模型继续运行。
十五、运行命令示例
CMD 示例：
python preprocessing/preprocess_global_aligned_face_parsing_roi_dataset_224_canvas.py ^
--project-root . ^
--split-csv data/processed/splits/extreme_5fold.csv ^
--image-dir data/raw/images ^
--global-intermediate-dir data/processed/global_face/global_face_parsing_hybrid_foreheadrepair_blackbg_224_png_strict/intermediates ^
--output-dir data/processed/roi_dataset/global_aligned_face_parsing_roi_final5_224_canvas ^
--roi-types eye_roi,lip_roi,cheek_roi,forehead_roi,chin_roi ^
--core-roi-types eye_roi,lip_roi,cheek_roi,forehead_roi,chin_roi ^
--canvas-size 224 ^
--max-samples 50 ^
--overwrite
PowerShell 示例：
python preprocessing/preprocess_global_aligned_face_parsing_roi_dataset_224_canvas.py --project-root .
--split-csv data/processed/splits/extreme_5fold.csv --image-dir data/raw/images
--global-intermediate-dir data/processed/global_face/global_face_parsing_hybrid_foreheadrepair_blackbg_224_png_strict/intermediates --output-dir data/processed/roi_dataset/global_aligned_face_parsing_roi_final5_224_canvas
--roi-types eye_roi,lip_roi,cheek_roi,forehead_roi,chin_roi --core-roi-types eye_roi,lip_roi,cheek_roi,forehead_roi,chin_roi
--canvas-size 224 --max-samples 50
--overwrite
十六、实现完成后请反馈
修改完成后，请汇报：
修改了哪些文件；
主脚本路径；
是否保留旧 ROI 脚本；
是否将默认输出目录改为 global_aligned_face_parsing_roi_final5_224_canvas；
是否取消 nose_roi；
是否取消 midface_roi；
是否取消 cheek_pair_roi 名称，并改为 cheek_roi；
是否不保存 left_cheek_roi/right_cheek_roi 单独文件；
eye_roi 是否重新定义为眶周功能区；
eye_roi 是否加入最小高度保护，避免细线问题；
lip_roi 是否包含口周皮肤和鼻唇沟下部；
cheek_roi 是否左右脸颊拼接，左右半区是否固定为 112×224；
forehead_roi 是否保持当前逻辑不变；
chin_roi 是否新增并进入 core ROI；
masked ROI 是否使用 final_face_mask 交集；
是否避免 ROI 内部语义黑洞；
core_all_success 是否更新为五个 ROI 全成功；
multi_roi_valid_ids.csv 包含哪些字段；
QC 图是否更新为 Final5 ROI；
推荐的 pilot 运行命令；
是否没有修改训练代码；
是否没有在预处理阶段做 ImageNet Normalize。
请开始修改，并先用 --max-samples 50 进行 pilot。