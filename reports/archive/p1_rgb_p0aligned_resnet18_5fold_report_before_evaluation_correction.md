# P1-RGB 评价协议修正报告

## 结论

- 最终状态：`P1_RGB_BASELINE_READY_WITH_CORRECTED_EVALUATION_PROTOCOL`
- 协议身份：`P0_UNIFIED_RGB_REFRESHED_BASELINE`
- 训练未重跑，checkpoint 未修改，500 条 case OOF 概率未修改。
- 主评价单位已修正为 `visit_case`，`patient_group_id` 仅用于 split 分组与 cluster bootstrap。

## 数据审计

- 唯一 patient_group 数量：483
- 多病例 group 数量：17
- NYHA 发生变化的 group 数量：15
- 三分类标签发生变化的 group 数量：10
- 二分类标签发生变化的 group 数量：0
- binary conflict group 内 case 数量：0

## 主结果（visit/case）

- case/visit OOF 行数：500
- 500 unique case IDs
- 0 missing
- 0 duplicate

| Metric | Value |
|---|---:|
| macro_auc | 0.8278260869565217 |
| accuracy | 0.7900000000000000 |
| macro_precision | 0.7012498802796667 |
| macro_recall | 0.6898362507058159 |
| macro_f1 | 0.6950493439204457 |
| balanced_accuracy | 0.6898362507058159 |
| pr_auc | 0.9441728720715097 |
| patient_sensitivity | 0.8753246753246753 |
| control_specificity | 0.5043478260869565 |
| ppv | 0.8553299492385786 |
| npv | 0.5471698113207547 |

Confusion matrix: `[[58, 57], [48, 337]]`

## patient-cluster bootstrap

```json
{
  "status": "available",
  "iterations": 2000,
  "seed": 2026,
  "cluster_unit": "patient_group_id",
  "metric_unit": "visit_case",
  "unique_clusters": 483,
  "visit_count": 500,
  "point_estimates": {
    "macro_auc": 0.8278260869565217,
    "accuracy": 0.79,
    "macro_precision": 0.7012498802796667,
    "macro_recall": 0.6898362507058159,
    "macro_f1": 0.6950493439204457,
    "balanced_accuracy": 0.6898362507058159,
    "pr_auc": 0.9441728720715097,
    "patient_sensitivity": 0.8753246753246753,
    "control_specificity": 0.5043478260869565,
    "ppv": 0.8553299492385786,
    "npv": 0.5471698113207547
  },
  "bootstrap_mean": {
    "macro_auc": 0.8284203995423852,
    "accuracy": 0.7904788190837024,
    "macro_f1": 0.6952233459273217,
    "balanced_accuracy": 0.6907611289586786,
    "patient_sensitivity": 0.8754527693842088,
    "control_specificity": 0.5060694885331483
  },
  "ci95": {
    "macro_auc": [
      0.7887865686174121,
      0.8653366558367377
    ],
    "accuracy": [
      0.7534412955465587,
      0.8250559893650147
    ],
    "macro_f1": [
      0.6462715442429674,
      0.7414157052205576
    ],
    "balanced_accuracy": [
      0.6420819849521675,
      0.7381020539390001
    ],
    "patient_sensitivity": [
      0.84196190547801,
      0.9069164398812469
    ],
    "control_specificity": [
      0.4153603603603604,
      0.5952653997378766
    ]
  },
  "failed_iterations": 0,
  "valid_iterations": 2000
}
```

## 与历史 E0B 的修正后配对 bootstrap

```json
{
  "status": "available",
  "iterations": 2000,
  "seed": 2026,
  "cluster_unit": "patient_group_id",
  "metric_unit": "visit_case",
  "unique_clusters": 483,
  "visit_count": 500,
  "point_estimates": {
    "p1_rgb": {
      "macro_auc": 0.8278260869565217,
      "accuracy": 0.79,
      "macro_precision": 0.7012498802796667,
      "macro_recall": 0.6898362507058159,
      "macro_f1": 0.6950493439204457,
      "balanced_accuracy": 0.6898362507058159,
      "pr_auc": 0.9441728720715097,
      "patient_sensitivity": 0.8753246753246753,
      "control_specificity": 0.5043478260869565,
      "ppv": 0.8553299492385786,
      "npv": 0.5471698113207547
    },
    "historical_e0b": {
      "macro_auc": 0.8550197628458498,
      "accuracy": 0.806,
      "macro_precision": 0.7321428571428572,
      "macro_recall": 0.7642574816487859,
      "macro_f1": 0.7447032504276878,
      "balanced_accuracy": 0.7642574816487859,
      "pr_auc": 0.9566043679609771,
      "patient_sensitivity": 0.8415584415584415,
      "control_specificity": 0.6869565217391305,
      "ppv": 0.9,
      "npv": 0.5642857142857143
    },
    "delta": {
      "delta_roc_auc": -0.027193675889328084,
      "delta_accuracy": -0.016000000000000014,
      "delta_macro_f1": -0.04965390650724211,
      "delta_balanced_accuracy": -0.07442123094297004,
      "delta_sensitivity": 0.03376623376623378,
      "delta_specificity": -0.18260869565217397
    }
  },
  "bootstrap_mean": {
    "delta": {
      "delta_roc_auc": -0.02627585948477175,
      "delta_accuracy": -0.015375184760159461,
      "delta_macro_f1": -0.04889938899726809,
      "delta_balanced_accuracy": -0.0737137427702743,
      "delta_sensitivity": 0.034350448036901085,
      "delta_specificity": -0.18177793357744967
    }
  },
  "ci95": {
    "delta_roc_auc": [
      -0.06678771764692594,
      0.012727550652122102
    ],
    "delta_accuracy": [
      -0.05567715948796205,
      0.026104417670682722
    ],
    "delta_macro_f1": [
      -0.10505328782353948,
      0.008558360605484613
    ],
    "delta_balanced_accuracy": [
      -0.13263906208465306,
      -0.012749668004068715
    ],
    "delta_sensitivity": [
      -0.005102367033769981,
      0.07486663308274558
    ],
    "delta_specificity": [
      -0.2941307773109244,
      -0.0697540250447228
    ]
  },
  "failed_iterations": 0,
  "valid_iterations": 2000
}
```

## Fold 汇总（visit/case）

| Fold | Visits | Patient groups | Control | Patient | Best epoch | Macro-AUC | Accuracy | Macro-F1 | Balanced Accuracy |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 100 | 98 | 23 | 77 | 1 | 0.8379 | 0.7700 | 0.6337 | 0.6220 |
| 1 | 100 | 98 | 23 | 77 | 3 | 0.8504 | 0.8200 | 0.7814 | 0.8374 |
| 2 | 100 | 96 | 23 | 77 | 1 | 0.8656 | 0.8000 | 0.6280 | 0.6110 |
| 3 | 100 | 97 | 23 | 77 | 2 | 0.8679 | 0.8000 | 0.7401 | 0.7634 |
| 4 | 100 | 94 | 23 | 77 | 19 | 0.8588 | 0.7600 | 0.6250 | 0.6155 |

## 协议修正

- 新协议文件：`metadata/p1_component_training_protocol_v1_1.json`
- 修订记录：`metadata/evaluation_protocol_correction.json`
- 原始协议已标记为 deprecated。
- `case_level_alias = visit_level`
- `evaluation_level = visit_case`
- `split_group_level = patient_group_id`
- `bootstrap_cluster_unit = patient_group_id`

## 废弃 group 评价

- `oof/oof_predictions_group.csv`
- `oof/oof_metrics_group.json`
- `oof/oof_confusion_matrix_group.csv`
- `summary/fold_metrics_group.csv`
- `summary/paired_cluster_bootstrap.json`

## 约束

- `training_rerun = false`
- `checkpoints_modified = false`
- `oof_probabilities_modified = false`
- `folds_modified = false`
- `labels_modified = false`
- `group_level_metrics_deprecated = true`
- `visit_level_metrics_primary = true`
- `patient_cluster_bootstrap_used = true`

split_equivalence = `EXACT_MATCH`
image_equivalence = `DIFFERENT_INPUT_REPRESENTATION`
