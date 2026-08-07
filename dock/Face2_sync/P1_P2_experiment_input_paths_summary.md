# Face2 P1/P2 Experiment Input Paths Summary

Generated on: 2026-08-01

This document records the currently observed local paths for P1/P2 experiment inputs and related scripts. It is intended as a handoff sheet for downstream analysis.

## 1. 高分辨率对齐人脸目录

当前工程中未发现单独的“高分辨率对齐人脸”目录。P0/P1/P2 统一使用的正式对齐人脸训练图是 224 x 224 版本：

```text
E:\projects\face2\data\processed\P0_Physics_Audit_v1\images\aligned_scene_224
```

补充：P0 中还存在两个相关图像目录：

```text
E:\projects\face2\data\processed\P0_Physics_Audit_v1\images\aligned_blackbg_224
E:\projects\face2\data\processed\P0_Physics_Audit_v1\images\e0b_meanbg_224
```

## 2. 图像尺寸及格式

`aligned_scene_224` 实测为：

```text
尺寸：224 x 224
格式：PNG
通道：RGB
dtype：uint8
数量：500
```

示例文件：

```text
E:\projects\face2\data\processed\P0_Physics_Audit_v1\images\aligned_scene_224\100037382.png
```

## 3. physics_core_skin mask 目录

```text
E:\projects\face2\data\processed\P0_Physics_Audit_v1\masks\physics_core_skin_224
```

## 4. mask 尺寸、格式、取值

`physics_core_skin_224` 实测为：

```text
尺寸：224 x 224
格式：PNG
通道：单通道 L
dtype：uint8
取值：0 / 255
数量：500
```

定义来自 P0-A 收尾资产：

```text
physics_core_skin = left_cheek_effective OR right_cheek_effective OR forehead_effective
```

不包含 chin、lip、eye，也不直接等同于完整 skin_strict mask。

## 5. 500 例统一主索引路径

P0 统一主索引：

```text
E:\projects\face2\data\processed\P0_Physics_Audit_v1\metadata\master_index.csv
```

该表包含 500 行，并包含如下关键字段：

```text
ID
NYHA
SEX
patient_group_id
binary_label
binary_name
label_3class
fold
aligned_scene_relpath
e0b_meanbg_relpath
physics_core_skin_mask_relpath
physics_core_skin_pixel_count
physics_core_skin_fraction_face
physics_core_skin_available
p0_usable
overall_status
```

P1 实验更推荐使用的统一实验索引：

```text
E:\projects\face2\data\processed\P1_Component_Audit_v1\manifests\p1_master_manifest.csv
```

该表把 P0 图像、mask、P1 DECA frozen assets、标签、fold 和 QC 信息 join 到一起。关键路径列包括：

```text
rgb_path
final_face_mask_path
face_valid_mask_path
skin_strict_mask_path
physics_core_skin_mask_path
latents_path
maps_path
quality_path
provenance_path
success_record_path
```

注意：此前已删除 `P1_DECA_Frozen500_v1\cases\*\relighting.npz`，因此 `p1_master_manifest.csv` 中旧的 `relighting_path` 当前不再有效。P2 重光照请使用本文第 10-11 节的 R3DPR 目录。

如需分别填写：

标签表：

```text
E:\projects\face2\data\processed\P0_Physics_Audit_v1\splits_500\nyha_3class_sex_stratified_group_5fold.csv
```

patient_group 映射：

```text
E:\projects\face2\data\processed\P0_Physics_Audit_v1\metadata\master_index.csv
```

其中 `patient_group_id`、`binary_label`、`binary_name`、`label_3class`、`label_3class_name` 可用于映射。

sample_id 与图像路径映射：

```text
E:\projects\face2\data\processed\P0_Physics_Audit_v1\metadata\master_index.csv
```

其中 `ID` 是 sample_id，`aligned_scene_relpath` 和 `e0b_meanbg_relpath` 是相对图像路径。

## 6. fold 0 RGB 二分类训练脚本

P2-B0 fold 0 RGB 二分类入口脚本：

```text
E:\projects\face2\scripts\run\run_p2_b0_binary_rgb_fold0.py
```

其内部调用的训练脚本：

```text
E:\projects\face2\scripts\train\train_e0b_global_resnet18_control_patient_binary_5fold.py
```

示例命令：

```powershell
python scripts\run\run_p2_b0_binary_rgb_fold0.py --config config\p2\p2_b0_binary_rgb_fold0_v1.yaml --output-dir experiments\500Data\P2_B0_BinaryRGB_Fold0_v1
```

## 7. 二分类配置文件

P2-B0 fold 0 RGB 二分类配置：

```text
E:\projects\face2\config\p2\p2_b0_binary_rgb_fold0_v1.yaml
```

配置要点：

```text
task.type: binary_control_vs_patient
p2.fixed_fold: 0
model.backbone: resnet18
model.pretrained: imagenet
data.image_size: 224
train.epochs: 50
train.batch_size: 16
train.lr: 0.0001
train.random_seed: 2026
```

注意：该配置当前仍指向较早的 RGB 数据路径：

```text
data/processed/global_face/preprocess_ablation/hybrid_imagenet_meanbg/images
```

如要改用 P0-A 正式资产，应显式确认是否切换为：

```text
E:\projects\face2\data\processed\P0_Physics_Audit_v1\images\e0b_meanbg_224
```

或：

```text
E:\projects\face2\data\processed\P0_Physics_Audit_v1\images\aligned_scene_224
```

## 8. fold 0 已有结果目录

当前未发现已经落盘的专用目录：

```text
E:\projects\face2\experiments\500Data\P2_B0_BinaryRGB_Fold0_v1
```

已有的历史 E0B RGB 二分类 fold 0 对照结果目录为：

```text
E:\projects\face2\experiments\500Data\E0B_Global_ResNet18_ControlVsPatient_Binary_5fold\fold_0
```

该目录包含：

```text
metrics.csv
confusion_matrix.csv
training_history.csv
training_parameters.json
val_predictions.csv
checkpoints\
```

另有整理后的二分类汇总报告目录：

```text
E:\projects\face2\dock\Face2_sync\two_binaty_report
```

## 9. 统一评价脚本

P2-B0 fold 0 评价汇总脚本：

```text
E:\projects\face2\scripts\evaluate\summarize_p2_b0_binary_rgb_fold0.py
```

使用方式：

```powershell
python scripts\evaluate\summarize_p2_b0_binary_rgb_fold0.py --experiment-dir experiments\500Data\P2_B0_BinaryRGB_Fold0_v1
```

历史 E0B 五折二分类汇总脚本：

```text
E:\projects\face2\scripts\evaluate\summarize_e0b_global_resnet18_control_patient_binary_5fold.py
```

## 10. R3DPR 六重光照根目录

当前 P2-A2/A3 配置实际使用的 R3DPR 六重光照根目录是：

```text
E:\projects\face2\data\processed\global_face\SixRelighting_OriginalCamera_meanfg
```

该目录实测包含：

```text
病例目录：500 个主病例目录，另含 p2_a2_v2_assets / p2_a3_v2_assets / p2_a3_v3_assets 等索引目录
relight_*.png：3000 张
```

示例病例目录：

```text
E:\projects\face2\data\processed\global_face\SixRelighting_OriginalCamera_meanfg\100037382
```

示例内容：

```text
e0b_meanbg_224.png
relight_neutral_front.png
relight_left.png
relight_right.png
relight_top.png
relight_dim_front.png
relight_bright_front.png
```

另一个较新的 P 开头重光照目录也存在：

```text
E:\projects\face2\data\processed\P2_SixRelighting_OriginalCamera_v1
```

但当前 P2-A2/A3 配置文件指向的是 `global_face\SixRelighting_OriginalCamera_meanfg`，不是 `P2_SixRelighting_OriginalCamera_v1`。

相关配置文件：

```text
E:\projects\face2\config\p2\p2_a\p2_a2_v2_r3dpr_meanbg.yaml
E:\projects\face2\config\p2\p2_a\p2_a3_v2_r3dpr_full6_consistency_meanbg.yaml
E:\projects\face2\config\p2\p2_a\p2_a3_v3_r3dpr_pairwise_consistency_meanbg.yaml
```

## 11. 重光照命名规则

R3DPR 六重光照文件命名规则：

```text
<R3DPR_ROOT>\<case_id>\relight_neutral_front.png
<R3DPR_ROOT>\<case_id>\relight_left.png
<R3DPR_ROOT>\<case_id>\relight_right.png
<R3DPR_ROOT>\<case_id>\relight_top.png
<R3DPR_ROOT>\<case_id>\relight_dim_front.png
<R3DPR_ROOT>\<case_id>\relight_bright_front.png
```

对应 preset 顺序：

```text
0 neutral_front  -> relight_neutral_front.png
1 left           -> relight_left.png
2 right          -> relight_right.png
3 top            -> relight_top.png
4 dim_front      -> relight_dim_front.png
5 bright_front   -> relight_bright_front.png
```

R3DPR 图像实测为：

```text
尺寸：224 x 224
格式：PNG
通道：RGB
dtype：uint8
```

R3DPR manifest / preset mapping：

```text
E:\projects\face2\data\processed\global_face\SixRelighting_OriginalCamera_meanfg\p2_a2_v2_manifest.csv
E:\projects\face2\data\processed\global_face\SixRelighting_OriginalCamera_meanfg\p2_a2_v2_assets\preset_mapping.json
E:\projects\face2\data\processed\global_face\SixRelighting_OriginalCamera_meanfg\p2_a3_v2_assets\p2_multiview_manifest.csv
E:\projects\face2\data\processed\global_face\SixRelighting_OriginalCamera_meanfg\p2_a3_v3_assets\p2_a3_v3_pair_manifest.csv
```

## 12. SO-1 代码目录是否使用

指定目录：

```text
/mnt/e/projects/face2/src/skin_optics_so1
```

Windows 对应路径：

```text
E:\projects\face2\src\skin_optics_so1
```

当前检查结果：

```text
不存在，未使用
```

当前工程中存在的是 SO-0：

```text
E:\projects\face2\src\skin_optics_so0
```

## 13. SO-1 输出目录是否使用

指定目录：

```text
/mnt/e/projects/face2/outputs/SO1_Minimal_Synthetic2Real_ClosedLoop_v1
```

Windows 对应路径：

```text
E:\projects\face2\outputs\SO1_Minimal_Synthetic2Real_ClosedLoop_v1
```

当前检查结果：

```text
不存在，未使用
```

当前工程中存在的是 SO-0 输出：

```text
E:\projects\face2\outputs\SO0_Forward_Model_v1
E:\projects\face2\outputs\SO0_Forward_Model_v1.1
```

