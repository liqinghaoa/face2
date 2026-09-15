# SO-R2-X3 Exploratory Nested-Direct Classification

Internal five-fold OOF exploratory evidence only; no external validation, bootstrap, p value, winner selection, or authorization change. M/H-sensitive maps are not absolute melanin/hemoglobin concentrations.

## Pooled OOF

```csv
condition,scope,macro_auc,accuracy,macro_precision,macro_recall,macro_f1,balanced_accuracy,sensitivity,specificity
RGB,pooled_oof,0.8369960474308301,0.778,0.7077205882352942,0.7552230378317335,0.7216300940438871,0.7552230378317335,0.7974025974025974,0.7130434782608696
RGB_B1MH,pooled_oof,0.7986222473178994,0.76,0.6791736524638183,0.7099943534726143,0.6896915534040837,0.7099943534726143,0.8025974025974026,0.6173913043478261
RGB_B2MH,pooled_oof,0.8400451722190853,0.782,0.7023809523809523,0.730378317334839,0.7131201473878142,0.730378317334839,0.825974025974026,0.6347826086956522
```

## Five-fold mean and SD

```csv
,macro_auc,macro_auc,macro_f1,macro_f1,balanced_accuracy,balanced_accuracy,sensitivity,sensitivity,specificity,specificity
,mean,std,mean,std,mean,std,mean,std,mean,std
condition,,,,,,,,,,
RGB,0.8413,0.0395,0.7106,0.0722,0.7552,0.1012,0.7974,0.0953,0.713,0.2708
RGB_B1MH,0.8178,0.0206,0.6808,0.0402,0.71,0.0763,0.8026,0.0868,0.6174,0.2349
RGB_B2MH,0.8404,0.0336,0.7126,0.0343,0.7304,0.0416,0.826,0.0396,0.6348,0.0902
```

## Pre-registered comparisons

```json
{
  "primary_RGB_B2MH_minus_RGB_pooled_macro_auc": 0.003049124788255142,
  "secondary_RGB_B1MH_minus_RGB_pooled_macro_auc": -0.03837380011293068
}
```
