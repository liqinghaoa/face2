# 面部心功能研究：DECA物理引导采集不变学习正式研究路线

> 文档用途：作为新分支的完整背景文件，继续实施P1阶段及之后阶段的实验。  
> 当前日期：2026-07-26  
> 当前状态：P0技术可行性阶段已完成，下一步进入P1。  
> 重要说明：本路线中的`albedo-like`、`shading-like`和反事实重光照均为模型生成的物理启发表示，不是真实皮肤反射率、真实照明或生理参数。

---

## 5. 已完成的P0阶段

## 5.1 P0-A：输入与物理审计资产

当前结果：

- 500例全部形成可用输入资产；
- 498例与原有黑背景预处理结果严格回归一致；
- 2例存在局部下颌—颈部边界歧义，但未构成整体资产阻断；
- 已生成`physics_core_skin`等分析mask。

两个需持续跟踪的病例：

```text
A001917272-1
A002081031-1
```

它们不应在后续被静默删除，而应保留并记录质量状态。

## 5.2 P0-B0：DECA环境与核心链路

环境：

- WSL Ubuntu；
- Conda环境：`p0b_deca`；
- Python 3.9；
- PyTorch 2.4.1+cu124；
- Torchvision 0.19.1+cu124；
- CUDA Toolkit 12.4；
- RTX 4060 Laptop GPU；
- PyTorch3D已在WSL源码编译并通过CUDA扩展和renderer测试；
- DECA checkpoint、FLAME、BFM纹理和PyTorch3D rasterizer加载成功。

正式环境审计最终结果：

```text
passed = true
blocked = false
failed_checks = []
```

核心链路已实际完成：

```text
DECA load
→ encode
→ decode
→ render
→ texture
→ SH relighting
→ finite output
→ deterministic tolerance audit
```

## 5.3 P0-B1：12例双输入模式比较

两种输入模式：

```text
direct_p0_aligned
mask_bbox_crop
```

结果：

- 12例×2种输入模式；
- 共24次运行；
- 24/24成功；
- 0例失败；
- exit code 0；
- 两种模式alpha覆盖率均接近1；
- 重建MAE非常接近。

最终不再继续耗费时间进行输入模式优选，固定：

```yaml
fixed_input_mode: direct_p0_aligned
```

选择依据：

- 减少额外bbox裁剪与插值；
- 与P0-A冻结输入链保持一致；
- 流程更简单；
- 可重复性更高。

不得声称`direct_p0_aligned`显著优于`mask_bbox_crop`。

## 5.4 P0-B2：12例自动非坍缩审计

根据Codex正式审计汇总，最终工程判定为：

```text
GO_FULL_AUX
```

主要结果：

- direct病例：12；
- bbox病例：12；
- 原始24组P0-B1结果哈希前后完全一致；
- 外观表征通过：
  - `tex_code`
  - `albedo_like`
- 结构表征通过：
  - `shape_code`
  - `detail_code`
  - `normal_coarse`
- 所有主表征的跨模式mate median rank均为1；
- 有效秩范围：4.24–7.53；
- 所有主要表征有效秩均高于预设阈值3；
- 采集组支配警告：否；
- 重光照失败：0；
- 敏感性分析冲突：否；
- 近重复候选：4个；
- 4个候选均为预定义0.5%距离尾部候选，不是完全重复；
- 已生成配对检查图。

测试：

```text
python -m pytest tests/p0b -ra
25 passed, 1 skipped
```

跳过项是历史盲评ZIP内容测试，当前ZIP不在工作区，不影响P0-B2核心结论。

### P0-B2的正确解释

该结果证明：

1. DECA外观和结构输出不是常数；
2. 未出现灾难性的平均脸坍缩；
3. 同一病例在两种输入预处理下具有较好表征稳定性；
4. 外观与结构均存在病例间可测差异；
5. 可以进入受控的500例批量生成。

该结果不证明：

- 表征与心功能状态相关；
- 表征能提高分类性能；
- 表征已经消除设备或EXIF混杂；
- `albedo_like`是真实反射率；
- 研究主线已经获得论文级证据。

---

## 6. 当前正式阶段状态

| 阶段 | 状态 |
|---|---|
| P0-A 输入与物理审计资产 | 完成 |
| P0-B0 DECA环境与核心链路 | 完成 |
| P0-B1 双输入模式工程测试 | 完成 |
| 输入模式冻结 | `direct_p0_aligned` |
| P0-B2 非坍缩审计 | `GO_FULL_AUX` |
| P0技术可行性阶段 | 完成 |
| P1 | 下一步 |

---

## 12. 关键目录与文件

### 项目根目录

Windows：

```text
E:\projects\face2
```

WSL：

```text
/mnt/e/projects/face2
```

### DECA目录

```text
third_party/DECA
```

### DECA资产目录

```text
third_party/DECA/data
```

主要资产：

- `deca_model.tar`
- `generic_model.pkl`
- `FLAME_albedo_from_BFM.npz`
- `fixed_displacement_256.npy`
- `head_template.obj`
- `landmark_embedding.npy`
- `mean_texture.jpg`
- `texture_data_256.npy`
- UV masks

### P0-B结果目录

```text
data/processed/P0B_DECA_Pilot12_v1
```

### 主要配置

```text
config/p0b/p0b_deca_environment_v1.yaml
config/p0b/p0b_deca_pilot12_v1.yaml
config/p0b/p0b_relighting_presets_v1.yaml
```

当前配置必须包含：

```yaml
fixed_input_mode: direct_p0_aligned
```

### P0-B2主要报告

```text
reports/p0b_representation_noncollapse_audit.md
reports/p0b_input_mode_engineering_decision.md
```

### P0-B2主要输出

```text
data/processed/P0B_DECA_Pilot12_v1/noncollapse_audit_v1/
```

关键文件：

```text
noncollapse_decision.json
representation_inventory.csv
cross_mode_retrieval_summary.csv
effective_rank_summary.csv
logs/p0b_noncollapse_audit.log
```