# R3DPR Binary 12-model Nested-Direct Results Package

This package is a derived, read-only organization of the completed prespecified nested-direct experiments. Original experiment directories remain the source of record.

## Protocol

- Each outer fold contains 400 development cases and an untouched 100-case outer test fold.
- The development cases are split using label x sex stratification into 320 internal training and 80 internal validation cases.
- Internal validation Macro-AUC selects the checkpoint. That same checkpoint directly predicts the outer test fold; no 400-case refitting stage is used.
- The primary results are pooled outer-held-out OOF metrics. Fold mean +/- SD is stability description only.
- This package contains the prespecified four input views x three backbones = 12 models. Exploratory `eye2_roi` experiments are excluded.

## Contents

- `model_inventory.csv`: source paths and protocol metadata for all 12 models.
- `pooled_oof_metrics.csv`: five pooled OOF primary metrics for each model.
- `foldwise_metrics_long.csv`: 60 unrounded outer-test fold rows, including selected epochs.
- `foldwise_metrics_mean_sd.csv`: fold mean and sample SD for stability description only.
- `pooled_oof_predictions/`: 12 complete 500-row OOF prediction files.
- `pooled_oof_predictions_long.csv`: all 6,000 model-sample OOF rows.
- `pooled_confusion_matrices.csv` and `confusion_matrices/`: matrices recomputed from OOF labels and hard predictions.
- `integrity_checks.csv`: ID, fold, metric, probability, protocol, and cross-model alignment checks.

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
