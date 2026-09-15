# R3DPR Binary 12-model Results Package

This package is a derived, read-only organization of the completed R3DPR binary experiments. Original experiment directories remain the source of record.

## Contents

- `model_inventory.csv`: configuration and source-path registry for all 12 models.
- `foldwise_metrics_long.csv`: 60 unrounded rows (12 models x 5 folds).
- `foldwise_metrics_mean_sd.csv`: fold mean and sample SD (`ddof=1`) for stability description only.
- `pooled_oof_predictions/`: 12 complete 500-row OOF prediction files retaining source prediction columns plus model metadata.
- `pooled_oof_predictions_long.csv`: all 6,000 model-sample OOF rows in long format.
- `pooled_confusion_matrices.csv` and `confusion_matrices/`: pooled OOF confusion matrices recomputed from `binary_label` and `pred_class`.
- `integrity_checks.csv`: validation results recorded during packaging.

## Interpretation Rules

- Pooled OOF metrics are the primary results; fold mean +/- SD describes fold stability only.
- Each model's OOF file has one held-out prediction per participant.
- Confusion matrices are recomputed from unrounded OOF labels and predictions, never reconstructed from displayed sensitivity or specificity.
- The 12 OOF files share identical `sample_id`, `patient_group_id`, fold, and true-label assignments, permitting paired prediction-level comparisons.

## Model Inventory

| Model ID | Input view | Backbone | Input width x height |
|---|---|---|---:|
| cheek_roi_resnet18 | cheek_roi | resnet18 | 224x224 |
| cheek_roi_resnet34 | cheek_roi | resnet34 | 224x224 |
| cheek_roi_resnet50 | cheek_roi | resnet50 | 224x224 |
| eye_roi_resnet18 | eye_roi | resnet18 | 224x224 |
| eye_roi_resnet34 | eye_roi | resnet34 | 224x224 |
| eye_roi_resnet50 | eye_roi | resnet50 | 224x224 |
| lip_roi_resnet18 | lip_roi | resnet18 | 224x224 |
| lip_roi_resnet34 | lip_roi | resnet34 | 224x224 |
| lip_roi_resnet50 | lip_roi | resnet50 | 224x224 |
| whole_face_sh93_resnet18 | whole_face_sh93 | resnet18 | 256x320 |
| whole_face_sh93_resnet34 | whole_face_sh93 | resnet34 | 256x320 |
| whole_face_sh93_resnet50 | whole_face_sh93 | resnet50 | 256x320 |
