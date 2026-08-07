# P2-A 统一单 RGB 训练与折内评价框架实现报告

生成时间：2026-07-29

## 1. 已检查并复用的现有代码

- 读取并遵守：
  - `dock/Face2_sync/Face_Cardiac_Physics_Informed_Research_Roadmap.md`
  - `dock/Face2_sync/Face_Cardiac_P2_Physics_Relighting_Research_Plan_精简修订版.md`
- 复用 P2-0：
  - `p2_counterfactual/assets.py`
  - `get_preset_index(preset_names, requested_name)`
  - `PRESET_NAMES = neutral_front / left / right / top / dim_front / bright_front`
  - `p2_training_manifest.csv`
  - `relighting.npz/relighted_images`
  - `relit_shading.npz/relighted_shading` 的既有数据约定
- 复用 P1-RGB 训练协议：
  - ImageNet `ResNet18_Weights.IMAGENET1K_V1`
  - ImageNet normalization：mean `[0.485,0.456,0.406]`，std `[0.229,0.224,0.225]`
  - AdamW `lr=1e-4`，`weight_decay=1e-4`
  - batch size 16，max epochs 50，patience 10，seed 2026，AMP false
  - training-fold-only weighted cross entropy
  - checkpoint by original validation Macro-AUC
  - `metrics.binary_classification_metrics.compute_binary_metrics`

## 2. 统一 Dataset 结构

新增：

- `p2_counterfactual/dataset.py`
- `p2_counterfactual/transforms.py`

Dataset 从 `data/processed/P2_Counterfactual_Relighting500_v1/manifests/p2_training_manifest.csv` 读取：

- `case_id`
- `patient_group_id`
- `fold`
- `binary_label`
- `original_rgb_path`
- `maps_npz_path`
- `relighting_npz_path`
- `relighted_shading_path`
- `p1_qc_flag`
- `boundary_uncertain`

实现约束：

- 不枚举目录替代 manifest；
- 不修改 fold；
- 不修改 label；
- 不删除 QC flagged case；
- 不删除 boundary uncertain case；
- 不使用 Camera/EXIF；
- 单 RGB 模型输入不包含 `signed_residual` 或 `relighted_shading`。

说明：原 manifest 中的 `/mnt/e/projects/face2` 路径已机械转换为当前 Windows 环境可直接读取的 `E:/projects/face2` 路径，只改变路径表示，不改变病例、标签、fold 或资产内容。

## 3. 四种配置切换

配置目录按项目既有约定放在：

- `config/p2/p2_a/common.yaml`
- `config/p2/p2_a/p2_a0_no_aug.yaml`
- `config/p2/p2_a/p2_a1_colorjitter.yaml`
- `config/p2/p2_a/p2_a2_relighting.yaml`
- `config/p2/p2_a/p2_a3_full_consistency.yaml`

四种模式只通过配置切换：

| experiment_id | input_mode | ColorJitter | relighting | consistency |
|---|---|---:|---:|---:|
| `p2_a0_no_aug` | `original` | false | false | false |
| `p2_a1_colorjitter` | `original` | true, p=0.5 | false | false |
| `p2_a2_relighting` | `relight_mix` | false | true, original/relighted=0.5/0.5 | false |
| `p2_a3_full_consistency` | `paired` | false | true | true |

## 4. ResNet18 结构

新增：

- `p2_counterfactual/model.py`

`P2ASingleRGBResNet18`：

- ImageNet-pretrained ResNet18；
- backbone 输出 512D feature；
- classifier 为 `Linear(512, 2)`；
- `forward(images)` 返回 `(logits, features)`；
- 未加入 Dropout、LayerNorm、Bottleneck、注意力、门控或复杂 MLP。

## 5. Weighted CE

新增：

- `p2_counterfactual/losses.py`

类别权重使用项目已有 `losses.classification_losses.compute_class_weights(labels, 2)`，只由当前训练 fold 计算。结果写入每折：

- `fold_metadata.json`

## 6. P2-A3 一致性损失

实现：

- 分类损失：`0.5 CE(y, logits_original) + 0.5 CE(y, logits_counterfactual)`
- 预测一致性：对称 Jensen-Shannon divergence，使用 `log_softmax`、`clamp_min`，batch mean
- 特征一致性：`1 - cosine(features_original, features_counterfactual)`，batch mean
- 总损失：`L_cls + w(t) * (0.5 * L_pred + 0.1 * L_feat)`

训练历史保存：

- `train_total_loss`
- `train_cls_loss`
- `train_pred_loss`
- `train_feat_loss`
- `warmup_factor`

A0/A1/A2 中 `train_pred_loss=0`、`train_feat_loss=0`。

## 7. Warm-up

`consistency_warmup_epochs=5`：

- epoch 1：0.2
- epoch 2：0.4
- epoch 3：0.6
- epoch 4：0.8
- epoch >=5：1.0

A3 smoke 中实际记录：`warmup_factor=0.2`。

## 8. 同步水平翻转与 left/right 映射

实现：

- A3 原图与 counterfactual 使用同一个水平翻转随机决定；
- 若发生翻转，metadata preset 名称执行 `left ↔ right`；
- A2 relighted 样本若发生翻转，也执行同样 metadata 映射；
- 映射只用于记录，不重新选择或重新渲染图像。

## 9. 单折 Trainer 与 Evaluator

新增：

- `p2_counterfactual/trainer.py`
- `p2_counterfactual/evaluator.py`
- `scripts/p2/run_p2_a_fold.py`

runner 支持：

- `--mode train`
- `--mode evaluate`
- `--mode train-evaluate`
- `--mode smoke`
- `--resume`
- `--device`
- `--num-workers`
- `--max-batches`
- `--fold`

每折输出结构：

- `checkpoints/best_macro_auc.pth`
- `checkpoints/last.pth`
- `config_resolved.yaml`
- `fold_metadata.json`
- `training_history.csv`
- `val_predictions_original.csv`
- `val_predictions_relighted.csv`
- `metrics_original.json`
- `confusion_matrix_original.csv`
- `val_features_original.npz`
- `val_features_relighted.npz`
- `fold_summary.json`
- `logs/`

checkpoint 选择只依据原始验证图像 Macro-AUC；Macro-AUC 相同则使用 Macro-F1，再相同保留较早 epoch。

## 10. 单元测试结果

执行命令：

```text
E:\resarch\Anaconda3\envs\face2\python.exe -m pytest tests/p2 -q
```

结果：

```text
25 passed
```

覆盖：

- P2-0 既有资产测试；
- P2-A Dataset；
- P2-A transforms；
- P2-A model；
- P2-A losses；
- P2-A trainer；
- P2-A evaluator；
- P2-A configs。

## 11. 四种模式 smoke 结果

执行方式：

- fold 0；
- GPU：CUDA；
- smoke 输出目录：`experiments/500Data/P2_Physics_Relighting_v2/_framework_smoke/`
- 每模式 1 epoch，`--max-batches 1`；
- 不生成正式成功标记。

| 模式 | 训练 | checkpoint | 原图验证 | 六套重光照导出 | 原图特征 | 重光照特征 |
|---|---|---|---|---|---|---|
| P2-A0 | PASS | PASS | 12 行 | 72 行 | `[12,512]` | `[12,6,512]` |
| P2-A1 | PASS | PASS | 12 行 | 72 行 | `[12,512]` | `[12,6,512]` |
| P2-A2 | PASS | PASS | 12 行 | 72 行 | `[12,512]` | `[12,6,512]` |
| P2-A3 | PASS | PASS | 12 行 | 72 行 | `[12,512]` | `[12,6,512]` |

A2 smoke 训练历史记录了 original/relighted 与 preset 采样计数。A3 smoke 训练历史记录了 `train_pred_loss`、`train_feat_loss` 和 `warmup_factor=0.2`。

## 12. 边界确认

- 是否执行正式五折：否
- 是否生成正式 500 例 pooled OOF：否
- 是否进行 patient-cluster bootstrap：否
- 是否进行跨实验 paired comparison：否
- 是否生成 P2-A 最终研究结论：否
- 是否进入 P2-B：否
- 是否进入 P3/P4：否

## 13. 第二步仍需实现

后续单独实现：

1. 五折自动调度；
2. 500 例 OOF 汇总；
3. 跨光照稳定性统计；
4. patient-cluster bootstrap；
5. 跨实验比较报告；
6. 正式 fold success marker。
