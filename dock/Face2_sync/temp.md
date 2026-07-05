# 项目信息与实验补充说明

## 1. 项目根目录

```text
E:/projects/face2
```

## 2. 当前核心目录结构

当前项目核心结构如下，省略了缓存、IDE 配置和大量实验输出文件：

```text
E:/projects/face2
|-- config
|   |-- preprocess
|   |   |-- global_aligned_face_parsing_roi_final5_224_canvas.yaml
|   |   |-- global_face_oval_blackbg_png_simalign_strict.yaml
|   |   |-- global_face_parsing_hybrid_foreheadrepair_blackbg_224_png_strict.yaml
|   |   `-- global_face_parsing_regularmask_blackbg_224_png_strict.yaml
|   |-- train
|   |   |-- nyha_3class_global224_imagenet_resnet18.yaml
|   |   |-- nyha_3class_global224_imagenet_resnet34.yaml
|   |   |-- nyha_3class_global224_imagenet_resnet34_weightedce_ls010.yaml
|   |   |-- nyha_3class_global224_imagenet_resnet50.yaml
|   |   |-- roi
|   |   `-- roi_fusion
|-- data
|   |-- raw
|   `-- processed
|       |-- EXIF_features
|       |-- global_face
|       |   `-- global_face_parsing_hybrid_foreheadrepair_blackbg_224_png_strict
|       |       `-- images
|       |-- roi_dataset
|       |-- splits
|       |-- splits_500
|       `-- splits_500_manual
|-- datasets
|   |-- nyha_3class_face_dataset.py
|   `-- nyha_3class_multi_roi_dataset.py
|-- dock
|   `-- Face2_sync
|-- evaluators
|   `-- nyha_3class_evaluator.py
|-- experiments
|-- losses
|   `-- classification_losses.py
|-- metrics
|   `-- classification_metrics.py
|-- models
|   |-- resnet_nyha_3class.py
|   `-- multi_roi_fusion_nyha_3class.py
|-- preprocessing
|   |-- build_global_face_oval_blackbg_png_simalign_strict.py
|   |-- build_global_face_oval_meanbg.py
|   |-- build_global_face_parsing_hybrid_foreheadrepair_blackbg_224_png_strict.py
|   |-- build_global_face_parsing_regularmask_blackbg_224_png_strict.py
|   |-- preprocess_global_aligned_face_parsing_roi_dataset_224_canvas.py
|   `-- preprocess_single_roi_dataset.py
|-- scripts
|   |-- 5fold
|   |   `-- build_nyha_3class_sex_stratified_group_5fold_split.py
|   |-- evaluate
|   |   `-- summarize_nyha_3class_5fold.py
|   |-- run
|   |   |-- run_exp_global224_imagenet_resnet18_nyha3class_5fold.py
|   |   |-- run_exp_global224_imagenet_resnet34_nyha3class_5fold.py
|   |   |-- run_exp_global224_imagenet_resnet34_weightedce_ls010_nyha3class_5fold.py
|   |   |-- run_exp_global224_imagenet_resnet50_nyha3class_5fold.py
|   |   |-- run_exp_roi_fusion_nyha3class_5fold.py
|   |   `-- run_exp_roi_nyha3class_5fold.py
|   `-- train
|       `-- train_nyha_3class_5fold.py
|-- trainers
|   `-- nyha_3class_trainer.py
`-- utils
    `-- experiment_utils.py
```

## 3. 当前基线预处理脚本路径

```text
preprocessing/build_global_face_parsing_hybrid_foreheadrepair_blackbg_224_png_strict.py
```

该脚本生成当前全局脸部基线图像：

```text
global_face_parsing_hybrid_foreheadrepair_blackbg_224_png_strict
```

核心特征：

- MediaPipe 检测人脸和 FaceMesh 关键点。
- FaceMesh 在检测 crop 上运行，但关键点会映射回原图坐标。
- 最终直接从原图 warp 到 224×224，减少旋转裁剪黑边。
- BiSeNet 解析人脸语义区域。
- 语义脸部 mask 为主体。
- 规则 envelope 仅用于额头局部修复。
- 背景统一置黑，输出普通 RGB PNG。

## 4. 当前基线预处理配置路径

```text
config/preprocess/global_face_parsing_hybrid_foreheadrepair_blackbg_224_png_strict.yaml
```

当前配置中的主要路径：

```yaml
split_csv: data\raw\label_raw.csv
image_dir: data/raw/images
output_dir: data/processed/global_face/global_face_parsing_hybrid_foreheadrepair_blackbg_224_png_strict
```

主要参数：

```yaml
image_size: 224
parsing_model: bisenet
parsing_checkpoint: preprocessing/checkpoints/face_parsing/79999_iter.pth
final_mask_mode: hybrid
forehead_band_ratio: 0.35
hair_repair_threshold: 0.10
enable_jaggedness_trigger: false
feather_kernel: 11
overwrite: false
```

## 5. 当前训练入口脚本路径

```text
scripts/run/run_exp_global224_imagenet_resnet18_nyha3class_5fold.py
```

该脚本是 Global224 + ImageNet ResNet18 + NYHA 三分类 + 五折验证实验入口。它主要负责：

```text
读取默认训练配置
→ 预检五折 CSV、标签、图片和 patient_group_id 泄漏
→ 创建实验输出目录
→ 调用 scripts/train/train_nyha_3class_5fold.py
→ 调用 scripts/evaluate/summarize_nyha_3class_5fold.py
```

## 6. 当前训练配置路径

```text
config/train/nyha_3class_global224_imagenet_resnet18.yaml
```

当前配置核心内容：

```yaml
experiment:
  name: Global224_ImageNetResNet18_NYHA3Class_WeightedCE_5Fold
  output_dir: experiments/500Data

data:
  split_dir: data/processed/splits_500
  image_root: data/processed/global_face/global_face_parsing_hybrid_foreheadrepair_blackbg_224_png_strict/images
  image_filename_template: "{ID}.png"
  n_folds: 5
  image_size: 224
  num_classes: 3
  label_col: label_3class
  train_csv_pattern: fold_{fold}_train.csv
  val_csv_pattern: fold_{fold}_val.csv

model:
  backbone: resnet18
  pretrained: imagenet
  num_classes: 3
  freeze_backbone: false

train:
  batch_size: 16
  epochs: 50
  optimizer: AdamW
  lr: 0.0001
  weight_decay: 0.0001
  loss: weighted_cross_entropy
  early_stopping_patience: 10
  monitor_metric: macro_auc
  random_seed: 2026
  num_workers: 0
  pin_memory: false
  use_amp: false
```

## 7. 当前五折划分目录

当前训练配置指定：

```text
data/processed/splits_500
```

`data/processed/splits_500` 中包含标准五折文件：

```text
fold_0_train.csv
fold_0_val.csv
fold_1_train.csv
fold_1_val.csv
fold_2_train.csv
fold_2_val.csv
fold_3_train.csv
fold_3_val.csv
fold_4_train.csv
fold_4_val.csv
nyha_3class_sex_stratified_group_5fold.csv
split_quality_report.md
missing_images.csv
invalid_records.csv
multi_image_patient_groups.csv
```

当前已确认训练配置使用 `data/processed/splits_500`，且该目录存在。后续基线训练和预处理消融实验应保持同一套五折划分，避免不同实验之间因划分不同导致结果不可比。

## 8. 当前基线图像目录

```text
data/processed/global_face/global_face_parsing_hybrid_foreheadrepair_blackbg_224_png_strict/images
```

当前检查结果：

```text
目录存在
当前图片文件数：522
```

该目录是当前全局脸部黑背景 parsing-hybrid 预处理结果目录。训练配置中的 `image_root` 指向该目录。

## 9. 希望输出的新预处理目录根路径

已确认目录：

```text
data/processed/global_face/preprocess_ablation/
```

每一种预处理消融方案建议在该目录下单独建子目录，例如：

```text
data/processed/global_face/preprocess_ablation/
|-- global224_blackbg_no_foreheadrepair/
|-- global224_blackbg_semantic_only/
|-- global224_blackbg_regularmask/
|-- global224_meanbg_oval/
`-- global224_original_aligned/
```

## 10. 希望输出的新实验目录

已确认目录：

```text
experiments/preprocess_ablation_500Data/
```

## 11. 当前环境

当前检测到的运行环境：

```text
操作系统：Windows
常用 IDE：PyCharm
项目根目录：E:/projects/face2
Python 环境：E:/resarch/Anaconda3/envs/face_heart/python.exe
PyTorch：2.3.0
CUDA 是否可用：True
CUDA 设备数：1
GPU：NVIDIA GeForce RTX 4060 Laptop GPU
```

当前训练配置较保守：

```yaml
num_workers: 0
pin_memory: false
use_amp: false
```

后续实验运行策略已确认：

```text
批量跑多个实验。
```

实现时建议仍保留单实验入口和批量入口两层：

- 单实验入口便于排错和复现实验。
- 批量入口负责按预处理方案依次运行 ResNet18 训练和结果汇总。
- 批量运行时应记录每个实验的开始时间、结束时间、退出状态、配置路径、输出目录和失败原因。

## 12. 是否允许 Codex 修改现有脚本

```text
B. 可以小幅修改旧文件，但前提是不影响旧实验的运行
```

理由：

- 如果做预处理消融实验，通常需要新增配置和新增运行入口。
- 也可能需要对通用训练入口或配置生成逻辑做小幅补充。
- 不建议直接重构核心训练脚本，避免影响已跑过的基线实验可复现性。

## 13. 是否先只跑 ResNet18

```text
是，先只跑 ResNet18。
```

理由：

- 当前已有明确 ResNet18 baseline。
- ResNet18 训练成本低，适合先验证不同预处理方案对性能的影响。
- 等预处理消融结果稳定后，再扩展到 ResNet34/ResNet50 更合理。

## 14. 是否要把所有结果自动汇总到 xlsx 和 markdown

已确认：

```text
是。实验结果除 xlsx 表格外，也额外生成 markdown 汇总文档。
```

建议汇总内容包括：

- 每个实验的配置路径。
- 预处理方案名称。
- 图像目录。
- split_dir。
- backbone。
- 每折指标。
- 五折 mean/std。
- OOF 指标。
- 最佳 fold / 最差 fold。
- 每个实验的 summary_report.md 路径。
- 每个实验的 oof_predictions.csv 路径。

建议输出：

```text
experiments/preprocess_ablation_500Data/summary_all.xlsx
experiments/preprocess_ablation_500Data/summary_all.md
```

## 15. 当前脚本是否已经保存 aligned 图和 final mask

### 15.1 全局基线预处理脚本

检查脚本：

```text
preprocessing/build_global_face_parsing_hybrid_foreheadrepair_blackbg_224_png_strict.py
```

结论：

```text
当前没有把 aligned 图和 final mask 作为逐样本独立文件保存。
```

该脚本当前实际单独保存的是最终黑背景 RGB PNG：

```text
data/processed/global_face/global_face_parsing_hybrid_foreheadrepair_blackbg_224_png_strict/images/{ID}.png
```

对应处理逻辑是：

```text
aligned_rgb = 原图坐标直接 warp 后的 224×224 对齐图
final_mask = hybrid 语义脸部最终 mask
alpha = feather_mask(final_mask)
final_rgb = apply_black_background(aligned_rgb, alpha)
save_png(images/{ID}.png, final_rgb)
```

其中 `aligned_rgb` 和 `final_mask` 会被用于：

- 生成最终黑背景图。
- 计算日志中的 mask 面积、头发/颈部/衣服/背景占比等统计量。
- 生成 QC preview 拼图中的局部面板，例如 aligned、final hybrid mask overlay、black-bg PNG。

但是当前脚本的输出目录只创建了：

```text
images/
logs/
qc_preview/
```

没有创建类似以下逐样本中间结果目录：

```text
aligned_rgb/
final_mask/
selected_semantic_mask/
semantic_regularized_mask/
parsing_label/
```

因此，当前全局基线输出中：

- `images/{ID}.png` 是最终黑背景图，不是未遮罩的 aligned 图。
- `qc_preview/*.jpg` 是多面板质控拼图，不适合作为训练数据或精确 mask 复用。
- `logs/preprocess_log.csv` 记录了 final mask 的统计信息，但不保存 mask 像素本身。

如果后续预处理消融需要复用相同的对齐图或相同的 final mask，建议小幅修改该脚本，新增可选参数，例如：

```text
--save-intermediates
```

并在开启时额外保存：

```text
aligned_rgb/{ID}.png
final_mask/{ID}.png
selected_semantic_mask/{ID}.png
semantic_regularized_mask/{ID}.png
candidate_envelope/{ID}.png
parsing_label/{ID}.png
```

这样后续做背景消融、mask 消融、额头修复消融时，可以避免重复做人脸检测、FaceMesh 对齐和 BiSeNet parsing，也能保证不同预处理方案共享同一个 aligned 基础图和 mask 依据。

### 15.2 ROI 预处理脚本

另一个相关脚本：

```text
preprocessing/preprocess_global_aligned_face_parsing_roi_dataset_224_canvas.py
```

这个脚本已经设计了中间结果保存机制。脚本文档和代码中包含以下中间目录：

```text
aligned_rgb/{ID}.png
parsing_label/{ID}.png
selected_semantic_mask/{ID}.png
final_face_mask/{ID}.png
```

并且有 `save_intermediates()` 逻辑用于保存：

```text
aligned_rgb
parsing_label
selected_semantic_mask
final_face_mask
```

但当前配置：

```text
config/preprocess/global_aligned_face_parsing_roi_final5_224_canvas.yaml
```

中设置为：

```yaml
save_intermediates: false
```

因此，ROI 脚本具备保存 aligned 图和 final mask 的能力，但默认配置下不会保存这些中间结果。若后续使用 ROI 预处理链路并希望保留中间文件，需要把该配置改为：

```yaml
save_intermediates: true
```

### 15.3 对后续预处理消融实验的影响

当前全局基线预处理结果不能直接拿到未遮罩 aligned 图和 final mask 像素文件。如果要做严谨的预处理消融，建议优先补充全局基线预处理脚本的中间结果保存能力，原因是：

1. 保证所有消融方案使用同一张 `aligned_rgb`，避免因重复检测/对齐导致输入差异。
2. 保证所有 mask 相关消融使用同一套 `selected_semantic_mask`、`semantic_regularized_mask`、`candidate_envelope` 和 `final_mask`。
3. 降低批量预处理成本，避免每个消融方案重复跑 MediaPipe 和 BiSeNet。
4. 后续出错时可以直接检查中间图，而不是只能看 QC 拼图。

建议新增中间结果保存后，再设计以下消融输出：

```text
data/processed/global_face/preprocess_ablation/
|-- baseline_blackbg_hybrid/
|-- aligned_rgb_no_mask/
|-- semantic_only_blackbg/
|-- hybrid_no_foreheadrepair_blackbg/
|-- regular_envelope_blackbg/
`-- meanbg_or_other_background/
```

## 已确认事项汇总

以下内容用于后续执行前核对，已确认项可直接作为后续任务依据。

```text
1. 五折划分目录：
   已确认使用 data/processed/splits_500

2. 新预处理输出根目录：
   已确认使用 data/processed/global_face/preprocess_ablation/

3. 新实验输出目录：
   已确认使用 experiments/preprocess_ablation_500Data/

4. Codex 修改现有脚本权限：
   已确认：允许小幅修改旧脚本，但前提是不影响旧实验运行

5. 是否先只跑 ResNet18：
   已确认：是，只跑 ResNet18

6. 结果汇总形式：
   已确认：除 xlsx 表格外，也可以额外生成 markdown 文档。
   表格用于结构化指标汇总，markdown 用于记录每次实验结果的文字分析。

7. 后续实验运行策略：
   已确认：批量跑多个实验

8. 当前脚本的 aligned 图和 final mask 保存情况：
   全局基线预处理脚本当前没有逐样本独立保存 aligned_rgb 和 final_mask。
   ROI 预处理脚本有 save_intermediates 能力，但当前配置默认为 false。
   后续预处理消融建议小幅补充全局基线脚本的中间结果保存能力。
```
