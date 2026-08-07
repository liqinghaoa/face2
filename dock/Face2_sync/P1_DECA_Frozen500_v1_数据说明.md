# P1_DECA_Frozen500_v1 数据说明

## 1. 数据集定位

数据根目录：`E:\projects\face2\data\processed\P1_DECA_Frozen500_v1`

这是基于 P0-A 已对齐图像生成的、冻结输入模式为 `direct_p0_aligned` 的 DECA 全量 500 例输出目录。它保存的是模型推理与固定光照重渲染的工程资产，用于可追溯性、质量控制和后续方法学分析；不是原始临床照片目录，也不是训练集标签目录。

当前 `metadata/run_manifest.json` 记录：

| 项目 | 当前值 |
| --- | ---: |
| 预期病例数 | 500 |
| 成功病例数 | 500 |
| 失败病例数 | 0 |
| 本轮新增成功 | 488 |
| 从有效断点恢复 | 12 |
| 固定输入模式 | `direct_p0_aligned` |
| P0 下颌—颈部边界不确定病例数 | 2 |

每个病例均有一个 `relighting.npz`，其中含六个固定预设，因此共有 **500 × 6 = 3,000** 个病例—光照结果。六个预设顺序固定为：

1. `neutral_front`
2. `left`
3. `right`
4. `top`
5. `dim_front`
6. `bright_front`

## 2. 目录结构

```text
P1_DECA_Frozen500_v1/
├── cases/<case_id>/                 # 500 个正式病例输出
│   ├── latents.npz                  # DECA latent codes
│   ├── maps.npz                     # P0 224×224 物理/重建相关图
│   ├── relighting.npz               # 六套固定 SH 重光照结果
│   ├── quality.json                 # 单病例数值质量指标
│   ├── provenance.json              # 输入、mask、哈希、字段来源
│   ├── _SUCCESS.json                # 完成标记及输出 SHA256
│   └── preview.png                  # 仅供快速 QC 的预览图
├── manifests/                       # 500 行输入、输出、质量和失败清单
├── metadata/                        # 环境、配置、资产和运行元数据
├── qc/                              # 代表性质量控制图
├── reproducibility_audit_v1/        # 可重复性审计产物
├── failures/                        # 失败记录（当前失败清单为空）
├── logs/                            # 运行日志
└── attempts/                        # 历史 pilot/生产尝试；不应与正式 cases 混用
```

正式全量数据应读取 `cases/` 和 `manifests/`。`attempts/` 保存历史尝试和 pilot12 记录，不是 Frozen500 正式分析的主输入。

## 3. 病例级文件

以下以 `cases/<case_id>/` 表示一个正式病例目录。所有图像空间数组均在 P0 aligned 的 224×224 坐标系中保存。

### 3.1 `latents.npz`

所有数组均为 `float32`，批次维度固定为 1：

| Key | 形状 | 含义 |
| --- | --- | --- |
| `shape_code` | `(1, 100)` | DECA shape code |
| `expression_code` | `(1, 50)` | expression code |
| `pose_code` | `(1, 6)` | pose code |
| `camera_code` | `(1, 3)` | camera code |
| `light_code` | `(1, 9, 3)` | 原始估计 SH light code |
| `tex_code` | `(1, 50)` | texture code |
| `detail_code` | `(1, 128)` | detail code |

### 3.2 `maps.npz`

所有数组均为 `float32`：

| Key | 形状 | 说明 |
| --- | --- | --- |
| `input_aligned_rgb` | `(224, 224, 3)` | 输入的 P0 aligned RGB，范围通常为 `[0, 1]` |
| `reconstruction` | `(224, 224, 3)` | DECA 重建图 |
| `alpha` | `(224, 224)` | 渲染 alpha/可见区域 |
| `albedo_like` | `(224, 224, 3)` | 模型 albedo-like 输出 |
| `normal_coarse` | `(224, 224, 3)` | coarse normal 图 |
| `shading_like` | `(224, 224, 3)` | 由编码 SH light 生成的 renderer shading，未裁剪 float32 |
| `signed_residual` | `(224, 224, 3)` | `input_aligned_rgb - reconstruction` |
| `absolute_residual` | `(224, 224, 3)` | `abs(signed_residual)` |

### 3.3 `relighting.npz`

| Key | 形状 | 说明 |
| --- | --- | --- |
| `preset_names` | `(6,)` | 六个固定预设名称，按第 1 节顺序读取 |
| `sh_coefficients` | `(6, 9, 3)` | 六套固定、无色 SH 系数 |
| `relighted_images` | `(6, 224, 224, 3)` | 六套重光照结果，`float32` |
| `preset_quality_json` | 标量字符串 | 每套预设的数值质量摘要 |

`relighted_images[i]` 必须通过 `preset_names[i]` 对应预设；不要在代码中假定数组下标含义而忽略 `preset_names`。

### 3.4 JSON 文件

- `quality.json`：输入范围与有限性、重建 MAE/RMSE、face/physics-core 误差、alpha 覆盖率、albedo/normal/residual 摘要、六套重光照的 min/max/均值、黑像素比例、饱和比例及成功状态。
- `provenance.json`：P0 输入相对路径及 SHA256、final face/face-valid/strict-skin/physics-core mask 引用和哈希、DECA 字段名映射、输入模式及边界不确定性状态。
- `_SUCCESS.json`：该病例完成时间、有效配置与 checkpoint 哈希、主要输出文件 SHA256、`validation_passed`。

## 4. Manifest 与推荐使用方式

| 文件 | 行数 | 主要用途 |
| --- | ---: | --- |
| `manifests/p1_deca_input_manifest.csv` | 500 | 病例 ID、P0 可用性、边界状态、输入哈希和冻结输入模式 |
| `manifests/p1_deca_output_manifest.csv` | 500 | encode/decode/render/texture/relighting 状态与主要数值指标 |
| `manifests/p1_deca_quality_manifest.csv` | 500 | 快速筛选有限性、alpha 覆盖、重建误差和重光照完整性 |
| `manifests/p1_deca_failure_manifest.csv` | 0 | 当前失败病例；为空表示该正式运行未登记失败病例 |

推荐流程：

1. 先读取 `p1_deca_output_manifest.csv`，确认 `status=success`、`validation_passed=True`、`relighting_complete=True`。
2. 以 `case_id` 定位 `cases/<case_id>/`。
3. 读取 `provenance.json`，核对输入和 mask 哈希；需要严格可追溯时，再核对 `_SUCCESS.json` 的输出哈希。
4. 使用 `preset_names` 索引 `relighted_images`，不要按文件名或数组位置推断光照类型。
5. 若做核心皮肤区域统计，使用 provenance 中引用的 P0 `physics_core_skin` mask，并与 alpha/visibility 共同限定统计区域。

## 5. 环境与可重复性

`metadata/environment.json` 记录该正式运行使用的环境：Python 3.9.25、PyTorch 2.4.1+cu124、Torchvision 0.19.1+cu124、PyTorch3D 0.7.9、CUDA 12.4、RTX 4060 Laptop GPU、固定随机种子 `20260726`。同时保存 DECA/FLAME/texture/UV mask/重光照配置的 SHA256。

还应一并保存或审阅：

- `metadata/effective_config.yaml`
- `metadata/asset_hashes.json`
- `metadata/deca_output_inventory.json`
- `metadata/pilot12_regression.json`
- `reproducibility_audit_v1/`

## 6. 科学解释边界

本目录中的 `albedo_like`、`shading_like`、normal、latent code 和固定 SH 重光照均为 DECA 前端的模型输出或工程派生量。

- `albedo_like` 不等同于真实反射率、真实皮肤颜色、血红蛋白、黑色素、灌注或血氧。
- `shading_like` 不等同于真实环境照明的测量值。
- `signed_residual`/`absolute_residual` 是未建模残差，不是 specular map。
- 重光照用于固定照明条件下的工程可行性与一致性分析，不单独构成临床诊断、分型或疗效结论。

## 7. Python 读取示例

```python
from pathlib import Path
import numpy as np
import pandas as pd

root = Path(r"E:\projects\face2\data\processed\P1_DECA_Frozen500_v1")
manifest = pd.read_csv(root / "manifests" / "p1_deca_output_manifest.csv")
valid = manifest.query("status == 'success' and validation_passed and relighting_complete")

case_id = str(valid.iloc[0]["case_id"])
case_dir = root / "cases" / case_id

with np.load(case_dir / "maps.npz") as maps:
    albedo_like = maps["albedo_like"]
    alpha = maps["alpha"]

with np.load(case_dir / "relighting.npz") as relighting:
    names = relighting["preset_names"].tolist()
    images = relighting["relighted_images"]
    neutral_front = images[names.index("neutral_front")]
```

本文档基于当前目录内的实际 metadata、manifest 和病例级 NPZ 字段整理；若后续重跑或迁移环境，应重新核对 `metadata/` 中的配置与哈希。
