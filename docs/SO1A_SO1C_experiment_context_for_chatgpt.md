# SO-1A～SO-1C 实验实现所需数据契约与上下文

生成日期：2026-08-05  
项目根目录：`/mnt/e/projects/face2`  
正式合成数据目录：`/mnt/e/projects/face2/data/processed/SO1_Synthetic_Generator_v1_localfull`

本文档按 `prompt.md` 要求整理，供后续 ChatGPT/Codex 实现 SO-1A～SO-1C 实验时使用。当前仓库中未检索到 `SO-1A`、`SO-1B`、`SO-1C` 三个标签的精确定义；本文不臆造实验编号，只固定合成数据契约、训练接口和必须复用的 SO-1 生成器信息。

## 1. 上游合成数据冻结状态

正式数据集已经完成并修复过一次 pair-retry 问题。当前有效完成文件：

```text
/mnt/e/projects/face2/data/processed/SO1_Synthetic_Generator_v1_localfull/COMPLETED.json
```

`COMPLETED.json` 内容：

```json
{
  "array_hard_gate_count": 18,
  "file_hash_count": 34,
  "full_generation_started": true,
  "independent_regeneration": {
    "independent_count": 32,
    "independent_mask_exact": 32,
    "independent_pass": 32,
    "independent_rgb_float16_max_abs": 0.00048828125,
    "independent_rgb_float32_max_abs": 0.0,
    "independent_rgb_max_ulp": 1,
    "independent_rgb_mismatch_count": 753,
    "independent_rgb_mismatch_fraction": 0.00011968612670898438,
    "independent_target_exact": 32,
    "same_context_count": 32,
    "same_context_pass": 32
  },
  "output_root": "/mnt/e/projects/face2/data/processed/SO1_Synthetic_Generator_v1_localfull",
  "previous_completion_invalidated": true,
  "repair_base_latent_ids": [
    5017,
    5335,
    5449,
    9161
  ],
  "repair_nonrepair_sample_hash_check": true,
  "repair_reason": "train_variant_final_pair_collision_after_clipping_retry",
  "repair_sample_ids": [
    "train_010034",
    "train_010035",
    "train_010670",
    "train_010671",
    "train_010898",
    "train_010899",
    "train_018322",
    "train_018323"
  ],
  "rows": 27000,
  "same_context_replay": {
    "count": 32,
    "pass": 32,
    "rgb_bitwise_exact": true
  },
  "smoke": false,
  "status": "COMPLETED",
  "train_final_pair_collision_count": 0
}
```

实现 SO-1A 时必须把 `COMPLETED.json` 作为上游冻结证据。不得自行重新推断数据是否完成。

相关审计文件：

```text
/mnt/e/projects/face2/data/processed/SO1_Synthetic_Generator_v1_localfull/qc/full_generation_hard_gate_audit_after_repair.json
/mnt/e/projects/face2/data/processed/SO1_Synthetic_Generator_v1_localfull/file_integrity_sha256.csv
/mnt/e/projects/face2/data/processed/SO1_Synthetic_Generator_v1_localfull/reproducibility_audit.csv
/mnt/e/projects/face2/data/processed/SO1_Synthetic_Generator_v1_localfull/same_context_replay_audit.csv
/mnt/e/projects/face2/data/processed/SO1_Synthetic_Generator_v1_localfull/repair_reproducibility_audit.csv
```

## 2. Split 与样本数

正式总数：

```text
train:       20,000
validation:  2,000
id_test:     2,000
camera_ood:  1,000
light_ood:   1,000
joint_ood:   1,000
total:       27,000
```

训练 split 设计：

```text
train_base_latents: 10000
train_variants_per_latent: 2
train total: 10000 × 2 = 20000
```

同一个 train `base_latent_id` 的两个 acquisition variants：

- 共享：`M`、`H`、`valid_mask`
- 不共享：`S`、`P`、`camera-light pair`
- 修复后全量门控确认：`10000/10000` 个 train base latent 的两个 final camera-light pair 均不同。

## 3. 实际数组 shape 与 dtype

每个 split 的数组实际 shape 如下。注意 `valid_mask` 有单通道维度，是 `[N, 1, 256, 256]`。

| split | linear_rgb.f16.npy | target_mhsp.f16.npy | valid_mask.u8.npy | written.u8.npy |
|---|---|---|---|---|
| train | `(20000, 3, 256, 256)` float16 | `(20000, 4, 256, 256)` float16 | `(20000, 1, 256, 256)` uint8 | `(20000,)` uint8 |
| validation | `(2000, 3, 256, 256)` float16 | `(2000, 4, 256, 256)` float16 | `(2000, 1, 256, 256)` uint8 | `(2000,)` uint8 |
| id_test | `(2000, 3, 256, 256)` float16 | `(2000, 4, 256, 256)` float16 | `(2000, 1, 256, 256)` uint8 | `(2000,)` uint8 |
| camera_ood | `(1000, 3, 256, 256)` float16 | `(1000, 4, 256, 256)` float16 | `(1000, 1, 256, 256)` uint8 | `(1000,)` uint8 |
| light_ood | `(1000, 3, 256, 256)` float16 | `(1000, 4, 256, 256)` float16 | `(1000, 1, 256, 256)` uint8 | `(1000,)` uint8 |
| joint_ood | `(1000, 3, 256, 256)` float16 | `(1000, 4, 256, 256)` float16 | `(1000, 1, 256, 256)` uint8 | `(1000,)` uint8 |

`written.u8.npy` 当前所有 split 均为：

```text
unique values: [1]
sum == N
```

训练 Dataset 应硬性要求：

```text
written.shape == (N,)
written.dtype == uint8
set(unique(written)) == {1}
written.sum() == N
```

不要静默跳过 `written != 1` 的样本。

## 4. 训练输入与监督目标

训练输入：

```text
linear_rgb.f16.npy
shape: [N, 3, 256, 256]
dtype on disk: float16
recommended model input dtype: float32 after loading
range: [0, 1]
```

监督目标：

```text
target_mhsp.f16.npy
shape: [N, 4, 256, 256]
dtype on disk: float16
recommended loss dtype: float32 after loading
range: [0, 1]
```

有效区域：

```text
valid_mask.u8.npy
shape: [N, 1, 256, 256]
dtype on disk: uint8
values: {0, 1}
```

所有 masked loss 必须使用 `valid_mask`，mask 外目标值已置零，但不应把 mask 外区域纳入损失。

## 5. target_mhsp 正式通道顺序

通道顺序来自生成器代码：

```text
src/skin_optics_so1/color_input.py
```

核心函数：

```python
def make_target_mhsp_chw(m, h, s, p, valid_mask_hw):
    target = np.stack(
        [
            np.asarray(m, dtype=np.float32),
            np.asarray(h, dtype=np.float32),
            (np.asarray(s, dtype=np.float32) - np.float32(0.25)) / np.float32(1.75),
            np.asarray(p, dtype=np.float32) / np.float32(0.10),
        ],
        axis=0,
    )
    target = np.clip(target, 0.0, 1.0)
    return (target * mask[None, ...]).astype(np.float32, copy=False)
```

正式通道契约：

```text
target_mhsp[:, 0] = M_norm
target_mhsp[:, 1] = H_norm
target_mhsp[:, 2] = S_norm
target_mhsp[:, 3] = P_norm
```

训练网络输出通道必须固定为：

```text
channel 0: M_norm
channel 1: H_norm
channel 2: S_norm
channel 3: P_norm
```

不要重新定义通道顺序。

## 6. M/H/S/P 归一化定义

正式配置：

```text
configs/so1_synthetic_generation_v1.yaml
```

控制范围：

```yaml
controls:
  m_min: 0.0
  m_max: 1.0
  h_min: 0.0
  h_max: 1.0
  m_base_min: 0.05
  m_base_max: 0.95
  h_base_min: 0.05
  h_base_max: 0.95
  shading_min: 0.25
  shading_max: 2.0
  specular_min: 0.0
  specular_max: 0.10
  exposure: 1.0
```

归一化公式：

```text
M_norm = M
H_norm = H
S_norm = (S - 0.25) / 1.75
P_norm = P / 0.10
```

反归一化公式：

```text
M = M_norm
H = H_norm
S = 0.25 + 1.75 * S_norm
P = 0.10 * P_norm
```

SO-1A～SO-1C 训练时只使用归一化目标即可，但 checkpoint、配置和报告必须记录上述范围，便于后续 RGB 重建和真实人脸解释。

## 7. 正式配置完整内容

```yaml
version: SO1_Synthetic_v1
global_seed: 20260801

patch:
  size: 256

counts:
  train_base_latents: 10000
  train_variants_per_latent: 2
  validation: 2000
  id_test: 2000
  camera_ood: 1000
  light_ood: 1000
  joint_ood: 1000

controls:
  m_min: 0.0
  m_max: 1.0
  h_min: 0.0
  h_max: 1.0
  m_base_min: 0.05
  m_base_max: 0.95
  h_base_min: 0.05
  h_base_max: 0.95
  shading_min: 0.25
  shading_max: 2.0
  specular_min: 0.0
  specular_max: 0.10
  exposure: 1.0

mask:
  full_probability: 0.50
  boundary_probability: 0.20
  holes_probability: 0.20
  mixed_probability: 0.10
  min_valid_fraction: 0.20

lights:
  seen:
    - D65
    - A
    - FL2
  unseen:
    - FL11

render:
  batch_size: 8
  oom_fallback_batch_sizes:
    - 4
    - 2
    - 1
  compute_dtype: float32
  max_total_clip_fraction: 0.05
  max_channel_clip_fraction: 0.10
  max_retries: 8

storage:
  rgb_dtype: float16
  target_dtype: float16
  mask_dtype: uint8
  format: numpy_open_memmap
  save_full_srgb_dataset: false
```

配置 hash：

```text
78256a4f18072257d407c46f311c5903a66e0242ee945ff62e1d1a18afcc5cdf
```

## 8. Metadata 字段与索引契约

每个 split 目录下都有：

```text
metadata.csv
```

字段名如下。

### train metadata 字段

`train/metadata.csv` 包含 repair 后新增的 initial/final pair 字段：

```text
row_index
sample_id
split
split_index
base_latent_id
acquisition_variant_id
m_base
h_base
m_seed
h_seed
s_seed
p_seed
mask_seed
acquisition_seed
camera_light_seed
camera_name
light_name
camera_light_pair
exposure
valid_mask_type
expected_valid_fraction
so0_version
so0_config_hash
spectral_asset_hash
generator_config_hash
actual_valid_fraction
low_clip_fraction
high_clip_fraction
total_clip_fraction
per_channel_clip_fraction
retry_count
generation_status
rgb_min
rgb_max
rgb_mean
rgb_std
m_mean
m_std
h_mean
h_std
s_mean
s_std
p_mean
p_std
p_nonzero_fraction
p_blob_count
initial_camera_name
initial_light_name
initial_camera_light_pair
final_camera_name
final_light_name
final_camera_light_pair
retry_pair_history
```

### validation/id_test/OOD metadata 字段

这些 split 当前 metadata 没有 initial/final pair 字段，字段到 `p_blob_count` 为止：

```text
row_index
sample_id
split
split_index
base_latent_id
acquisition_variant_id
m_base
h_base
m_seed
h_seed
s_seed
p_seed
mask_seed
acquisition_seed
camera_light_seed
camera_name
light_name
camera_light_pair
exposure
valid_mask_type
expected_valid_fraction
so0_version
so0_config_hash
spectral_asset_hash
generator_config_hash
actual_valid_fraction
low_clip_fraction
high_clip_fraction
total_clip_fraction
per_channel_clip_fraction
retry_count
generation_status
rgb_min
rgb_max
rgb_mean
rgb_std
m_mean
m_std
h_mean
h_std
s_mean
s_std
p_mean
p_std
p_nonzero_fraction
p_blob_count
```

### 严重注意：metadata 行号不总是数组下标

必须使用 `split_index` 访问 `.npy` 数组，不要使用 metadata 的 DataFrame 行号。

实际检查结果：

| split | metadata 行号是否等于数组 index | split_index 是否唯一 | split_index 是否覆盖 0..N-1 | 前 10 个 split_index |
|---|---:|---:|---:|---|
| train | yes | yes | yes | 0,1,2,3,4,5,6,7,8,9 |
| validation | no | yes | yes | 16,88,160,232,304,376,448,520,592,664 |
| id_test | no | yes | yes | 66,138,210,282,354,426,498,570,642,714 |
| camera_ood | no | yes | yes | 0,9,18,27,36,45,54,63,72,81 |
| light_ood | no | yes | yes | 0,25,50,75,100,125,150,175,200,225 |
| joint_ood | no | yes | yes | 0,3,6,9,12,15,18,21,24,27 |

Dataset 实现应：

```python
row = metadata.iloc[k]
i = int(row["split_index"])
rgb = linear_rgb[i]
target = target_mhsp[i]
mask = valid_mask[i]
```

如果需要按自然顺序迭代，可先按 `split_index` 排序，但必须保留原始 metadata。

## 9. Metadata 样例

### train 前 5 行关键字段

```json
{"sample_id":"train_000000","split":"train","split_index":"0","base_latent_id":"0","acquisition_variant_id":"0","m_seed":"12394173889755605833","h_seed":"9190320882891586670","s_seed":"10699472628800452688","p_seed":"9781858326042053240","mask_seed":"15626002630714043735","camera_name":"Canon 60D","light_name":"FL2","camera_light_pair":"Canon 60D / FL2","initial_camera_light_pair":"Canon 60D / FL2","final_camera_light_pair":"Canon 60D / FL2","exposure":"1.0","valid_mask_type":"mixed","retry_count":"0","generation_status":"SUCCESS"}
{"sample_id":"train_000001","split":"train","split_index":"1","base_latent_id":"0","acquisition_variant_id":"1","m_seed":"12394173889755605833","h_seed":"9190320882891586670","s_seed":"5034613329582300447","p_seed":"13000283674916939597","mask_seed":"15626002630714043735","camera_name":"Canon 40D","light_name":"D65","camera_light_pair":"Canon 40D / D65","initial_camera_light_pair":"Canon 40D / D65","final_camera_light_pair":"Canon 40D / D65","exposure":"1.0","valid_mask_type":"mixed","retry_count":"0","generation_status":"SUCCESS"}
{"sample_id":"train_000002","split":"train","split_index":"2","base_latent_id":"1","acquisition_variant_id":"0","m_seed":"16827162656828497126","h_seed":"8940383630051904266","s_seed":"6319290821388548876","p_seed":"4827486411240632873","mask_seed":"15848551734495507789","camera_name":"Point Grey Grasshopper2 14S5C","light_name":"A","camera_light_pair":"Point Grey Grasshopper2 14S5C / A","initial_camera_light_pair":"Point Grey Grasshopper2 14S5C / A","final_camera_light_pair":"Point Grey Grasshopper2 14S5C / A","exposure":"1.0","valid_mask_type":"full","retry_count":"0","generation_status":"SUCCESS"}
{"sample_id":"train_000003","split":"train","split_index":"3","base_latent_id":"1","acquisition_variant_id":"1","m_seed":"16827162656828497126","h_seed":"8940383630051904266","s_seed":"18122137443816519819","p_seed":"14908305824911981507","mask_seed":"15848551734495507789","camera_name":"Canon 50D","light_name":"A","camera_light_pair":"Canon 50D / A","initial_camera_light_pair":"Canon 50D / A","final_camera_light_pair":"Canon 50D / A","exposure":"1.0","valid_mask_type":"full","retry_count":"0","generation_status":"SUCCESS"}
{"sample_id":"train_000004","split":"train","split_index":"4","base_latent_id":"2","acquisition_variant_id":"0","m_seed":"18443659446333266533","h_seed":"420698830304689572","s_seed":"11840244405330597730","p_seed":"3568804646667542431","mask_seed":"10364168633807956890","camera_name":"Nikon D3","light_name":"FL2","camera_light_pair":"Nikon D3 / FL2","initial_camera_light_pair":"Nikon D3 / FL2","final_camera_light_pair":"Nikon D3 / FL2","exposure":"1.0","valid_mask_type":"full","retry_count":"0","generation_status":"SUCCESS"}
```

### validation 前 2 行关键字段

```json
{"sample_id":"validation_000016","split":"validation","split_index":"16","base_latent_id":"1000016","acquisition_variant_id":"0","camera_name":"Nikon D300s","light_name":"A","camera_light_pair":"Nikon D300s / A","exposure":"1.0","valid_mask_type":"full","retry_count":"1","generation_status":"SUCCESS"}
{"sample_id":"validation_000088","split":"validation","split_index":"88","base_latent_id":"1000088","acquisition_variant_id":"0","camera_name":"Canon 60D","light_name":"FL2","camera_light_pair":"Canon 60D / FL2","exposure":"1.0","valid_mask_type":"boundary","retry_count":"0","generation_status":"SUCCESS"}
```

### OOD 样例

```json
{"sample_id":"camera_ood_000000","split":"camera_ood","split_index":"0","base_latent_id":"3000000","camera_name":"Nikon D700","light_name":"D65","camera_light_pair":"Nikon D700 / D65","generation_status":"SUCCESS"}
{"sample_id":"light_ood_000000","split":"light_ood","split_index":"0","base_latent_id":"4000000","camera_name":"Point Grey Grasshopper 50S5C","light_name":"FL11","camera_light_pair":"Point Grey Grasshopper 50S5C / FL11","generation_status":"SUCCESS"}
{"sample_id":"joint_ood_000000","split":"joint_ood","split_index":"0","base_latent_id":"5000000","camera_name":"Olympus E-PL2","light_name":"FL11","camera_light_pair":"Olympus E-PL2 / FL11","generation_status":"SUCCESS"}
```

## 10. Camera/light split

文件：

```text
/mnt/e/projects/face2/data/processed/SO1_Synthetic_Generator_v1_localfull/camera_light_split.json
```

摘要：

```text
seen cameras: 25
unseen cameras: 3
seen lights: D65, A, FL2
unseen light: FL11
qualified pairs: 109
excluded failed pairs: 3
```

Seen cameras：

```text
Canon 1DMarkIII
Canon 20D
Canon 300D
Canon 40D
Canon 500D
Canon 50D
Canon 5DMarkII
Canon 60D
Hasselblad H2
Nikon D200
Nikon D3
Nikon D300s
Nikon D3X
Nikon D40
Nikon D50
Nikon D5100
Nikon D80
Nikon D90
Nokia N900
Pentax K-5
Pentax Q
Phase One
Point Grey Grasshopper 50S5C
Point Grey Grasshopper2 14S5C
SONY NEX-5N
```

Unseen cameras：

```text
Canon 600D
Nikon D700
Olympus E-PL2
```

Lights：

```text
seen: D65, A, FL2
unseen: FL11
```

Excluded failed pairs：

```text
Point Grey Grasshopper 50S5C / A
Point Grey Grasshopper 50S5C / D65
Point Grey Grasshopper 50S5C / FL2
```

Split 规则：

| split | camera set | light set |
|---|---|---|
| train | seen | seen |
| validation | seen | seen |
| id_test | seen | seen |
| camera_ood | unseen | seen |
| light_ood | seen | unseen |
| joint_ood | unseen | unseen |

## 11. 软链接与真实数据路径

正式目录包含统一入口：

```text
/mnt/e/projects/face2/data/processed/SO1_Synthetic_Generator_v1_localfull/synthetic_data
```

其中 split 是软链接，真实目标如下：

```text
synthetic_data/train      -> /mnt/e/projects/face2/data/processed/SO1_Synthetic_Generator_v1_localfull/train
synthetic_data/validation -> /mnt/e/projects/face2/data/processed/SO1_Synthetic_Generator_v1_localfull/validation
synthetic_data/id_test    -> /mnt/e/projects/face2/data/processed/SO1_Synthetic_Generator_v1_localfull/id_test
synthetic_data/camera_ood -> /mnt/e/projects/face2/data/processed/SO1_Synthetic_Generator_v1_localfull/camera_ood
synthetic_data/light_ood  -> /mnt/e/projects/face2/data/processed/SO1_Synthetic_Generator_v1_localfull/light_ood
synthetic_data/joint_ood  -> /mnt/e/projects/face2/data/processed/SO1_Synthetic_Generator_v1_localfull/joint_ood
```

Dataset 可以直接读取真实 split 目录，也可以通过 `synthetic_data/*` 读取，但需要知道这是软链接。

## 12. 现有 skin_optics_so1 包结构与复用建议

当前 SO-1 生成器包：

```text
src/skin_optics_so1/__init__.py
src/skin_optics_so1/build_synthetic_dataset.py
src/skin_optics_so1/camera_light_split.py
src/skin_optics_so1/color_input.py
src/skin_optics_so1/random_fields.py
src/skin_optics_so1/retry_pairs.py
src/skin_optics_so1/so0_batch_renderer.py
src/skin_optics_so1/synthetic_config.py
src/skin_optics_so1/synthetic_manifest.py
src/skin_optics_so1/synthetic_masks.py
src/skin_optics_so1/synthetic_qc.py
src/skin_optics_so1/synthetic_storage.py
```

当前测试：

```text
tests/skin_optics_so1/conftest.py
tests/skin_optics_so1/test_config_and_color.py
tests/skin_optics_so1/test_fields_masks_manifest.py
tests/skin_optics_so1/test_retry_pair_selection.py
tests/skin_optics_so1/test_storage_and_cli.py
```

建议复用：

| 功能 | 现有文件 |
|---|---|
| 严格配置加载与 hash | `synthetic_config.py` |
| manifest schema/build/validate | `synthetic_manifest.py` |
| target 通道顺序和 sRGB→linear 工具 | `color_input.py` |
| camera/light split | `camera_light_split.py` |
| 生成随机场逻辑 | `random_fields.py` |
| memmap storage shape 定义 | `synthetic_storage.py` |
| QC 输出结构 | `synthetic_qc.py` |
| retry pair 选择与修复 invariant | `retry_pairs.py` |

后续 decomposition/ 或 SO-1A～SO-1C 训练代码不要重新造一套不兼容 schema。最少应复用或严格对齐：

- `SyntheticGenerationConfig`
- `ManifestRow` 字段语义
- `make_target_mhsp_chw` 通道顺序
- storage shape/dtype
- `camera_light_split.json`

## 13. SO-1 分解网络实现约束

研究文档中 SO-1 分解网络契约：

```text
input:  linear RGB patch [3,256,256]
output: M/H/S/P [4,256,256]
```

输出通道：

```text
0: M_norm
1: H_norm
2: S_norm
3: P_norm
```

第一版网络：

```text
标准容量 U-Net
Encoder: 64 / 128 / 256 / 512
Bottleneck: 1024
Decoder: 512 / 256 / 128 / 64
Conv block: Conv + Normalization + LeakyReLU
Downsample: MaxPool
Upsample: Transposed Convolution
Output: 1x1 Conv, 4 channels
Activation: Sigmoid
Initialization: random
No Carvana pretraining
```

训练损失：

```text
masked SmoothL1 on valid_mask
L_total = (L_M + L_H + L_S + L_P) / 4
```

第一版不加入：

- 真实数据
- RGB 重建损失
- R3DPR 一致性损失
- P 稀疏损失
- S 低频损失
- M/H 平滑损失
- GAN/感知损失/分类标签监督

训练建议：

```text
optimizer: AdamW
initial lr: 1e-3
weight decay: 1e-4
batch size: 优先 8，按 8GB GPU 调整
AMP: enabled
max epoch: 25
early stopping patience: 5
scheduler: cosine
seed: 20260801
```

checkpoint 主选择指标：

```text
val_M_MAE + val_H_MAE
```

S/P 指标只作为辅助审计，不作为主 checkpoint 选择依据。

## 14. 评价要求

合成域必须分别报告：

- M/H/S/P MAE
- RMSE
- Pearson
- Spearman
- 输出标准差
- 常数预测基线
- ID Test
- Camera-OOD
- Light-OOD
- Joint-OOD

还需审计：

- M/H cross-talk
- 通道交换
- 输出坍塌
- 预测饱和比例
- mask 外输出

## 15. 实现 Dataset 时的最小硬门控

Dataset 初始化时建议强制检查：

```text
1. root/COMPLETED.json exists
2. COMPLETED.status == "COMPLETED"
3. COMPLETED.rows == 27000
4. split directory exists
5. all required arrays exist
6. shapes/dtypes match this document
7. written is all ones
8. metadata.csv exists
9. metadata split_index is unique and covers 0..N-1
10. metadata generation_status is all SUCCESS
11. target channel order recorded in dataset/config/checkpoint metadata
12. normalization ranges recorded in checkpoint metadata
```

Minimal `__getitem__` behavior:

```python
row = rows[k]
i = int(row["split_index"])
x = linear_rgb[i].astype("float32")       # [3,256,256]
y = target_mhsp[i].astype("float32")      # [4,256,256]
mask = valid_mask[i].astype("float32")    # [1,256,256]
return {"rgb": x, "target_mhsp": y, "valid_mask": mask, "metadata": row}
```

Do not assume `metadata.iloc[k]` corresponds to array index `k`.

## 16. 已知限制与解释边界

- M/H 是 melanin-sensitive 与 hemoglobin-sensitive 控制量，不是绝对真实浓度。
- S/P 是合成成像控制场，不是从真实人脸反演得到的 ground truth。
- RGB 是 SO-0 简化皮肤光学模型输出，不保证完全像真实人脸皮肤 patch。
- SO-0 已冻结；SO-1A～SO-1C 不应修改 SO-0 公式、资产、camera-light split 或合成数据 manifest。
- SO-1 分解网络不直接输出心功能标签。

## 17. 后续实现建议路径

建议新建独立训练包，避免污染生成器：

```text
src/skin_optics_so1_decomposition/
├── synthetic_dataset.py
├── unet_decomposer.py
├── losses.py
├── metrics.py
├── train_synthetic.py
├── evaluate_synthetic.py
└── configs.py
```

建议配置文件：

```text
configs/so1a_synthetic_pretrain_v1.yaml
configs/so1b_synthetic_eval_v1.yaml
configs/so1c_ood_audit_v1.yaml
```

如果后续 prompt 对 SO-1A/SO-1B/SO-1C 有更具体定义，应以新 prompt 为准；本文只提供当前实现必须依赖的数据契约。
