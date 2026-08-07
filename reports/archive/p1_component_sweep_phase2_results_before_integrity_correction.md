# P1 phase-two report

- final_status: P1_COMPONENT_SWEEP_COMPLETE
- stage_one_status: P1_COMPONENT_FRAMEWORK_READY_FOR_PHASE2
- approval_file: E:\projects\face2\experiments\500Data\P1_Component_Sweep_v1\metadata\PHASE2_EXECUTION_APPROVED.json
- experiments: p1_a, p1_n, p1_l, p1_s, p1_r, p1_rgb_a
- p1_spec_status: SKIPPED_UNAVAILABLE_BY_FRONTEND
- no_group_aggregation: true
- no_threshold_search: true
- no_hyperparameter_search: true
- total_formal_folds: 30/30

## 1. Stage-two goal

完成六个可运行实验的 smoke、正式五折、OOF、patient-cluster bootstrap、与修正后的 P1-RGB 逐 case 配对比较，并统一汇总结果。

## 2. Data / protocol checks

- manifest rows = 500
- unique case_id = 500
- fixed 5-fold split = 400/100 per fold
- evaluation_unit = visit_case
- cluster_unit = patient_group_id
- P1-N: horizontal_flip = false, normal_flip_mode = DISABLED_SAFE_FALLBACK

## 3. Smoke results

| experiment | display_name | status |
| --- | --- | --- |
| p1_a | P1-A | SMOKE_PASSED |
| p1_n | P1-N | SMOKE_PASSED |
| p1_l | P1-L | SMOKE_PASSED |
| p1_s | P1-S | SMOKE_PASSED |
| p1_r | P1-R | SMOKE_PASSED |
| p1_rgb_a | P1-RGB+A | SMOKE_PASSED |

## 4. Formal fold completion

| experiment | completed_folds | oof_rows |
| --- | --- | --- |
| p1_a | 5 | 139 |
| p1_l | 5 | 211 |
| p1_n | 5 | 130 |
| p1_r | 5 | 149 |
| p1_rgb_a | 5 | 103 |
| p1_s | 5 | 164 |

## 5. Main results

| experiment | display_name | macro_auc | accuracy | macro_f1 | balanced_accuracy | pr_auc | patient_sensitivity | control_specificity | ppv | npv |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| p1_a | P1-A | 0.6187 | 0.6600 | 0.5514 | 0.5566 | 0.8489 | 0.7481 | 0.3652 | 0.7978 | 0.3022 |
| p1_n | P1-N | 0.5813 | 0.6500 | 0.5270 | 0.5288 | 0.8187 | 0.7532 | 0.3043 | 0.7838 | 0.2692 |
| p1_l | P1-L | 0.7781 | 0.7040 | 0.6632 | 0.7285 | 0.9230 | 0.6831 | 0.7739 | 0.9100 | 0.4218 |
| p1_s | P1-S | 0.8226 | 0.7780 | 0.7241 | 0.7613 | 0.9377 | 0.7922 | 0.7304 | 0.9077 | 0.5122 |
| p1_spec | P1-Spec | NA | NA | NA | NA | NA | NA | NA | NA | NA |
| p1_r | P1-R | 0.8508 | 0.7800 | 0.7169 | 0.7413 | 0.9543 | 0.8130 | 0.6696 | 0.8917 | 0.5168 |
| p1_rgb_a | P1-RGB+A | 0.7243 | 0.7840 | 0.6832 | 0.6768 | 0.8928 | 0.8753 | 0.4783 | 0.8489 | 0.5340 |

## 6. Confidence intervals

| experiment | macro_auc_ci_low | macro_auc_ci_high | macro_f1_ci_low | macro_f1_ci_high | balanced_accuracy_ci_low | balanced_accuracy_ci_high | sensitivity_ci_low | sensitivity_ci_high | specificity_ci_low | specificity_ci_high |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| p1_a | 0.5624 | 0.6750 | 0.5052 | 0.5965 | 0.5073 | 0.6058 | 0.7030 | 0.7893 | 0.2791 | 0.4530 |
| p1_n | 0.5221 | 0.6418 | 0.4801 | 0.5757 | 0.4801 | 0.5795 | 0.7117 | 0.7969 | 0.2193 | 0.3962 |
| p1_l | 0.7329 | 0.8238 | 0.6208 | 0.7086 | 0.6825 | 0.7743 | 0.6386 | 0.7306 | 0.6929 | 0.8485 |
| p1_s | 0.7808 | 0.8621 | 0.6826 | 0.7674 | 0.7160 | 0.8055 | 0.7531 | 0.8320 | 0.6480 | 0.8095 |
| p1_spec | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA |
| p1_r | 0.8164 | 0.8837 | 0.6693 | 0.7609 | 0.6933 | 0.7880 | 0.7737 | 0.8514 | 0.5826 | 0.7523 |
| p1_rgb_a | 0.6708 | 0.7768 | 0.6340 | 0.7304 | 0.6290 | 0.7268 | 0.8421 | 0.9079 | 0.3858 | 0.5714 |

## 7. Pairwise comparison vs corrected P1-RGB

| experiment | display_name | delta_macro_auc | delta_accuracy | delta_macro_f1 | delta_balanced_accuracy | delta_patient_sensitivity | delta_control_specificity | probability_mae | probability_rmse | pearson | spearman | hard_prediction_agreement | changed_prediction_count |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| p1_a | P1-A | -0.2091 | -0.1300 | -0.1436 | -0.1332 | -0.1273 | -0.1391 | 0.2854 | 0.3859 | 0.2356 | 0.2051 | 0.6980 | 151 |
| p1_l | P1-L | -0.0497 | -0.0860 | -0.0318 | 0.0387 | -0.1922 | 0.2696 | 0.2877 | 0.3393 | 0.5283 | 0.5777 | 0.7020 | 149 |
| p1_n | P1-N | -0.2465 | -0.1400 | -0.1681 | -0.1610 | -0.1221 | -0.2000 | 0.3025 | 0.4149 | 0.0653 | 0.0090 | 0.6600 | 170 |
| p1_r | P1-R | 0.0230 | -0.0100 | 0.0219 | 0.0514 | -0.0623 | 0.1652 | 0.1733 | 0.2912 | 0.6397 | 0.6936 | 0.7980 | 101 |
| p1_rgb_a | P1-RGB+A | -0.1035 | -0.0060 | -0.0118 | -0.0130 | 0.0000 | -0.0261 | 0.2376 | 0.3392 | 0.3692 | 0.2137 | 0.7820 | 109 |
| p1_s | P1-S | -0.0052 | -0.0120 | 0.0290 | 0.0715 | -0.0831 | 0.2261 | 0.2533 | 0.3606 | 0.4316 | 0.4196 | 0.7200 | 140 |

## 8. Confusion matrices

| experiment | TN | FP | FN | TP |
| --- | --- | --- | --- | --- |
| p1_a | 42 | 73 | 97 | 288 |
| p1_n | 35 | 80 | 95 | 290 |
| p1_l | 89 | 26 | 122 | 263 |
| p1_s | 84 | 31 | 80 | 305 |
| p1_r | 77 | 38 | 72 | 313 |
| p1_rgb_a | 55 | 60 | 48 | 337 |

## 9. Status table

| experiment | status |
| --- | --- |
| p1_a | COMPLETED |
| p1_n | COMPLETED |
| p1_l | COMPLETED |
| p1_s | COMPLETED |
| p1_spec | SKIPPED_UNAVAILABLE_BY_FRONTEND |
| p1_r | COMPLETED |
| p1_rgb_a | COMPLETED |

## 10. P1-Spec handling

- status = SKIPPED_UNAVAILABLE_BY_FRONTEND
- reason = Frozen DECA frontend does not expose an independent specular-like component.
- replacement_used = false

## 11. Guardrails

- no group-level aggregation
- no threshold search
- no hyperparameter search
- no repeated training based on performance

## 12. Warnings / failures

- formal fold failures: 0
- smoke failures: 0
- bootstrap failed iterations: 0 for all completed experiments

## 13. Result matrix

| experiment | status | macro_auc | macro_f1 | balanced_accuracy |
| --- | --- | --- | --- | --- |
| p1_r | COMPLETED | 0.8508 | 0.7169 | 0.7413 |
| p1_s | COMPLETED | 0.8226 | 0.7241 | 0.7613 |
| p1_l | COMPLETED | 0.7781 | 0.6632 | 0.7285 |
| p1_rgb_a | COMPLETED | 0.7243 | 0.6832 | 0.6768 |
| p1_a | COMPLETED | 0.6187 | 0.5514 | 0.5566 |
| p1_n | COMPLETED | 0.5813 | 0.5270 | 0.5288 |
| p1_spec | SKIPPED_UNAVAILABLE_BY_FRONTEND | NA | NA | NA |

## 14. Conclusion

阶段二已完成，统一汇总已按提示词要求重建。