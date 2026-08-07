# exp_E2_Global_MultiROI_EqualFusion_ResNet18_5fold

This report is descriptive. Compare fold and pooled Macro-AUC with Macro-F1, balanced accuracy, severe recall, QWK, ordinal MAE, and two-step errors against E0 before drawing a conclusion.

## Pooled OOF metrics

| Metric | Value |
|---|---:|
| accuracy | 0.5120 |
| macro_precision | 0.4831 |
| macro_recall | 0.5091 |
| macro_f1 | 0.4607 |
| balanced_accuracy | 0.5091 |
| auc_normal | 0.8321 |
| precision_normal | 0.4681 |
| recall_normal | 0.7652 |
| f1_normal | 0.5809 |
| auc_mild | 0.6173 |
| precision_mild | 0.5611 |
| recall_mild | 0.6203 |
| f1_mild | 0.5892 |
| auc_severe | 0.6229 |
| precision_severe | 0.4200 |
| recall_severe | 0.1419 |
| f1_severe | 0.2121 |
| macro_auc | 0.6908 |
| severe_vs_rest_auc | 0.6229 |
| normal_vs_abnormal_auc | 0.8321 |
| ordinal_mae | 0.5660 |
| normal_to_severe_count | 2.0000 |
| severe_to_normal_count | 37.0000 |
| two_step_error_count | 39.0000 |
| two_step_error_rate | 0.0780 |
| qwk | 0.3047 |
| severe_recall | 0.1419 |
| multiclass_brier_score | 0.6798 |
| ece_15_equal_width | 0.2266 |
| nll | 1.1936 |
