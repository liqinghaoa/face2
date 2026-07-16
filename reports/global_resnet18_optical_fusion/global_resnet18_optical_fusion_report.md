# Global ResNet18 + Optical Phenotype Fusion Experiment

EXPERIMENT_STATUS=COMPLETE

## Completion, data, and provenance

All five variants and all 25 fold runs completed on the formal 500-case cohort (normal 115, mild 237, severe 148; 483 patient groups). Images came from the fixed hybrid ImageNet mean-background source and labels/splits were not regenerated.

Image source: `data/processed/global_face/preprocess_ablation/hybrid_imagenet_meanbg/images`. Split source: `data/processed/splits_500/nyha_3class_sex_stratified_group_5fold.csv`. Label provenance: `data/raw/label_raw_nyha2_remove22_sex_balanced_500.csv`.

Stage 1 source: `data/processed/optical_observations_v1/regional_optical_observations.csv`; Stage 2A root: `experiments/optical_condition_calibration_stage2a`; Stage 2B root: `experiments/optical_condition_calibration_stage2b`. All were verified by schema, manifest, ID, fold, split role, and SHA256.

## Variants and model

G0 uses 512 image features; G-Mask uses 512+1; G-Raw/G-A/G-B use 512+6+1. Every model is a fully trainable ImageNet ResNet18 followed only by direct concatenation and one 3-class Linear head (1,539 / 1,542 / 1,560 head parameters).

## Training protocol

All variants use the same patient-group folds, seed streams, flip-only ImageNet-normalized transforms, fold-specific weighted cross-entropy, AdamW (lr 1e-4, weight decay 1e-4), 50-epoch budget, patience 10, and strict earliest best validation macro-AUC checkpoint.

Stage 2A/2B use the matching per-fold train/validation files. Their 500-row OOF tables were audit-only and were never classifier training inputs.

Six-dimensional scalers were fit only on each outer training fold (ddof=0); unavailable forehead differences were filled with zero after standardization and the availability mask was appended last.

Fixed six-dimensional orders:

- G-Raw: cheek_mean_log2_y, cheek_mean_log2_rg, cheek_mean_log2_bg, forehead_minus_cheek_log2_y, forehead_minus_cheek_log2_rg, forehead_minus_cheek_log2_bg
- G-A: calibrated_cheek_mean_log2_y, calibrated_cheek_mean_log2_rg, calibrated_cheek_mean_log2_bg, calibrated_forehead_minus_cheek_log2_y, calibrated_forehead_minus_cheek_log2_rg, calibrated_forehead_minus_cheek_log2_bg
- G-B: calibrated_nn_cheek_mean_log2_y, calibrated_nn_cheek_mean_log2_rg, calibrated_nn_cheek_mean_log2_bg, calibrated_nn_forehead_minus_cheek_log2_y, calibrated_nn_forehead_minus_cheek_log2_rg, calibrated_nn_forehead_minus_cheek_log2_bg

Fold-specific class weights use N_train / (3 × class_count):

| variant | fold | weight_normal | weight_mild | weight_severe |
|---|---|---|---|---|
| global_only | 0 | 1.449275 | 0.705467 | 1.120448 |
| global_only | 1 | 1.449275 | 0.705467 | 1.120448 |
| global_only | 2 | 1.449275 | 0.701754 | 1.129943 |
| global_only | 3 | 1.449275 | 0.701754 | 1.129943 |
| global_only | 4 | 1.449275 | 0.701754 | 1.129943 |
| global_mask | 0 | 1.449275 | 0.705467 | 1.120448 |
| global_mask | 1 | 1.449275 | 0.705467 | 1.120448 |
| global_mask | 2 | 1.449275 | 0.701754 | 1.129943 |
| global_mask | 3 | 1.449275 | 0.701754 | 1.129943 |
| global_mask | 4 | 1.449275 | 0.701754 | 1.129943 |
| global_raw | 0 | 1.449275 | 0.705467 | 1.120448 |
| global_raw | 1 | 1.449275 | 0.705467 | 1.120448 |
| global_raw | 2 | 1.449275 | 0.701754 | 1.129943 |
| global_raw | 3 | 1.449275 | 0.701754 | 1.129943 |
| global_raw | 4 | 1.449275 | 0.701754 | 1.129943 |
| global_stage2a | 0 | 1.449275 | 0.705467 | 1.120448 |
| global_stage2a | 1 | 1.449275 | 0.705467 | 1.120448 |
| global_stage2a | 2 | 1.449275 | 0.701754 | 1.129943 |
| global_stage2a | 3 | 1.449275 | 0.701754 | 1.129943 |
| global_stage2a | 4 | 1.449275 | 0.701754 | 1.129943 |
| global_stage2b | 0 | 1.449275 | 0.705467 | 1.120448 |
| global_stage2b | 1 | 1.449275 | 0.705467 | 1.120448 |
| global_stage2b | 2 | 1.449275 | 0.701754 | 1.129943 |
| global_stage2b | 3 | 1.449275 | 0.701754 | 1.129943 |
| global_stage2b | 4 | 1.449275 | 0.701754 | 1.129943 |

## Per-fold checkpoint selection and training curves

| variant | fold | best_epoch | best_val_macro_auc | best_val_macro_f1 | best_val_balanced_accuracy | completed_epoch |
|---|---|---|---|---|---|---|
| global_only | 0 | 2 | 0.649138 | 0.454703 | 0.512910 | 12 |
| global_only | 1 | 28 | 0.752218 | 0.518978 | 0.538137 | 38 |
| global_only | 2 | 14 | 0.727089 | 0.546528 | 0.564220 | 24 |
| global_only | 3 | 20 | 0.681912 | 0.471314 | 0.494152 | 30 |
| global_only | 4 | 10 | 0.695578 | 0.502985 | 0.539994 | 20 |
| global_mask | 0 | 20 | 0.683814 | 0.492695 | 0.508100 | 30 |
| global_mask | 1 | 2 | 0.698147 | 0.500650 | 0.559023 | 12 |
| global_mask | 2 | 1 | 0.708646 | 0.496991 | 0.571580 | 11 |
| global_mask | 3 | 5 | 0.662780 | 0.407441 | 0.405324 | 15 |
| global_mask | 4 | 6 | 0.700071 | 0.460906 | 0.532316 | 16 |
| global_raw | 0 | 9 | 0.658998 | 0.446877 | 0.436459 | 19 |
| global_raw | 1 | 10 | 0.731644 | 0.537307 | 0.553369 | 20 |
| global_raw | 2 | 1 | 0.717664 | 0.453276 | 0.495734 | 11 |
| global_raw | 3 | 2 | 0.685634 | 0.486886 | 0.493134 | 12 |
| global_raw | 4 | 24 | 0.740806 | 0.487750 | 0.495591 | 34 |
| global_stage2a | 0 | 5 | 0.674228 | 0.469156 | 0.481988 | 15 |
| global_stage2a | 1 | 10 | 0.734486 | 0.541505 | 0.560553 | 20 |
| global_stage2a | 2 | 1 | 0.710082 | 0.460629 | 0.502827 | 11 |
| global_stage2a | 3 | 10 | 0.711438 | 0.482469 | 0.468825 | 20 |
| global_stage2a | 4 | 6 | 0.718682 | 0.529498 | 0.589362 | 16 |
| global_stage2b | 0 | 5 | 0.669018 | 0.475281 | 0.497085 | 15 |
| global_stage2b | 1 | 10 | 0.715340 | 0.509497 | 0.540324 | 20 |
| global_stage2b | 2 | 1 | 0.714511 | 0.440130 | 0.488642 | 11 |
| global_stage2b | 3 | 11 | 0.695479 | 0.497804 | 0.483935 | 21 |
| global_stage2b | 4 | 6 | 0.735273 | 0.518370 | 0.582270 | 16 |

Per-variant and per-fold training curves are retained under the experiment fold directories and copied summaries are under `reports/global_resnet18_optical_fusion/training_curves/`.

## Five-fold mean, sample standard deviation, median, minimum and maximum

| variant | metric | mean | sample_std_ddof1 | median | min | max |
|---|---|---|---|---|---|---|
| global_only | macro_auc | 0.701187 | 0.039960 | 0.695578 | 0.649138 | 0.752218 |
| global_only | accuracy | 0.506000 | 0.043359 | 0.520000 | 0.460000 | 0.550000 |
| global_only | balanced_accuracy | 0.529883 | 0.026992 | 0.538137 | 0.494152 | 0.564220 |
| global_only | macro_f1 | 0.498901 | 0.036751 | 0.502985 | 0.454703 | 0.546528 |
| global_mask | macro_auc | 0.690691 | 0.017974 | 0.698147 | 0.662780 | 0.708646 |
| global_mask | accuracy | 0.484000 | 0.030496 | 0.490000 | 0.440000 | 0.520000 |
| global_mask | balanced_accuracy | 0.515269 | 0.066172 | 0.532316 | 0.405324 | 0.571580 |
| global_mask | macro_f1 | 0.471737 | 0.039257 | 0.492695 | 0.407441 | 0.500650 |
| global_raw | macro_auc | 0.706949 | 0.034002 | 0.717664 | 0.658998 | 0.740806 |
| global_raw | accuracy | 0.512000 | 0.039623 | 0.520000 | 0.450000 | 0.550000 |
| global_raw | balanced_accuracy | 0.494857 | 0.041347 | 0.495591 | 0.436459 | 0.553369 |
| global_raw | macro_f1 | 0.482419 | 0.035964 | 0.486886 | 0.446877 | 0.537307 |
| global_stage2a | macro_auc | 0.709783 | 0.022116 | 0.711438 | 0.674228 | 0.734486 |
| global_stage2a | accuracy | 0.522000 | 0.040249 | 0.520000 | 0.460000 | 0.570000 |
| global_stage2a | balanced_accuracy | 0.520711 | 0.051990 | 0.502827 | 0.468825 | 0.589362 |
| global_stage2a | macro_f1 | 0.496651 | 0.036557 | 0.482469 | 0.460629 | 0.541505 |
| global_stage2b | macro_auc | 0.705924 | 0.024975 | 0.714511 | 0.669018 | 0.735273 |
| global_stage2b | accuracy | 0.506000 | 0.037815 | 0.520000 | 0.440000 | 0.530000 |
| global_stage2b | balanced_accuracy | 0.518451 | 0.042094 | 0.497085 | 0.483935 | 0.582270 |
| global_stage2b | macro_f1 | 0.488216 | 0.031360 | 0.497804 | 0.440130 | 0.518370 |

## Pooled OOF results for all variants

| variant | macro_auc | accuracy | balanced_accuracy | macro_f1 | auc_normal | auc_mild | auc_severe |
|---|---|---|---|---|---|---|---|
| global_only | 0.692742 | 0.506000 | 0.529685 | 0.507756 | 0.815788 | 0.612023 | 0.650415 |
| global_mask | 0.670938 | 0.484000 | 0.515801 | 0.486770 | 0.804630 | 0.576102 | 0.632083 |
| global_raw | 0.687547 | 0.512000 | 0.495054 | 0.497735 | 0.831824 | 0.616643 | 0.614174 |
| global_stage2a | 0.698469 | 0.522000 | 0.521683 | 0.513584 | 0.850073 | 0.620045 | 0.625288 |
| global_stage2b | 0.693780 | 0.506000 | 0.519384 | 0.502634 | 0.850322 | 0.621681 | 0.609337 |

Five-fold mean±sample-std metrics are preserved separately in each variant's `aggregate_fold_metrics.csv`; pooled OOF and fold means are not conflated.

## Pairwise fold and OOF changes

| candidate | reference | metric | oof_delta_candidate_minus_reference | fold_delta_mean | fold_delta_sample_std_ddof1 | candidate_better_folds | reference_better_folds | equal_folds |
|---|---|---|---|---|---|---|---|---|
| global_mask | global_only | macro_auc | -0.021803 | -0.010496 | 0.032793 | 2 | 3 | 0 |
| global_mask | global_only | accuracy | -0.022000 | -0.022000 | 0.041473 | 1 | 3 | 1 |
| global_mask | global_only | balanced_accuracy | -0.013883 | -0.014614 | 0.042997 | 2 | 3 | 0 |
| global_mask | global_only | macro_f1 | -0.020985 | -0.027165 | 0.039981 | 1 | 4 | 0 |
| global_raw | global_mask | macro_auc | 0.016609 | 0.016258 | 0.025881 | 4 | 1 | 0 |
| global_raw | global_mask | accuracy | 0.028000 | 0.028000 | 0.054498 | 4 | 1 | 0 |
| global_raw | global_mask | balanced_accuracy | -0.020747 | -0.020411 | 0.066883 | 1 | 4 | 0 |
| global_raw | global_mask | macro_f1 | 0.010965 | 0.010683 | 0.054350 | 3 | 2 | 0 |
| global_stage2a | global_mask | macro_auc | 0.027530 | 0.019091 | 0.024010 | 4 | 1 | 0 |
| global_stage2a | global_mask | accuracy | 0.038000 | 0.038000 | 0.047645 | 4 | 1 | 0 |
| global_stage2a | global_mask | balanced_accuracy | 0.005882 | 0.005442 | 0.056013 | 3 | 2 | 0 |
| global_stage2a | global_mask | macro_f1 | 0.026814 | 0.024915 | 0.051903 | 3 | 2 | 0 |
| global_stage2b | global_mask | macro_auc | 0.022841 | 0.015233 | 0.020598 | 4 | 1 | 0 |
| global_stage2b | global_mask | accuracy | 0.022000 | 0.022000 | 0.051672 | 4 | 1 | 0 |
| global_stage2b | global_mask | balanced_accuracy | 0.003582 | 0.003182 | 0.063194 | 2 | 3 | 0 |
| global_stage2b | global_mask | macro_f1 | 0.015863 | 0.016480 | 0.058576 | 3 | 2 | 0 |
| global_stage2a | global_raw | macro_auc | 0.010922 | 0.002834 | 0.018793 | 3 | 2 | 0 |
| global_stage2a | global_raw | accuracy | 0.010000 | 0.010000 | 0.015811 | 3 | 1 | 1 |
| global_stage2a | global_raw | balanced_accuracy | 0.026629 | 0.025854 | 0.045323 | 4 | 1 | 0 |
| global_stage2a | global_raw | macro_f1 | 0.015849 | 0.014232 | 0.018149 | 4 | 1 | 0 |
| global_stage2b | global_raw | macro_auc | 0.006233 | -0.001025 | 0.011163 | 2 | 3 | 0 |
| global_stage2b | global_raw | accuracy | -0.006000 | -0.006000 | 0.015166 | 1 | 4 | 0 |
| global_stage2b | global_raw | balanced_accuracy | 0.024330 | 0.023594 | 0.046665 | 2 | 3 | 0 |
| global_stage2b | global_raw | macro_f1 | 0.004898 | 0.005797 | 0.025699 | 3 | 2 | 0 |
| global_stage2b | global_stage2a | macro_auc | -0.004689 | -0.003859 | 0.014738 | 2 | 3 | 0 |
| global_stage2b | global_stage2a | accuracy | -0.016000 | -0.016000 | 0.015166 | 0 | 4 | 1 |
| global_stage2b | global_stage2a | balanced_accuracy | -0.002299 | -0.002260 | 0.016518 | 2 | 3 | 0 |
| global_stage2b | global_stage2a | macro_f1 | -0.010951 | -0.008435 | 0.019271 | 2 | 3 | 0 |

## Paired patient-group bootstrap macro-AUC deltas

| candidate | reference | delta_mean | ci_lower_2_5 | ci_upper_97_5 | valid_repetitions |
|---|---|---|---|---|---|
| global_mask | global_only | -0.022067 | -0.058881 | 0.014541 | 2000 |
| global_raw | global_mask | 0.016766 | -0.021140 | 0.052777 | 2000 |
| global_stage2a | global_mask | 0.028283 | -0.002233 | 0.059240 | 2000 |
| global_stage2b | global_mask | 0.023450 | -0.006966 | 0.054807 | 2000 |
| global_stage2a | global_raw | 0.011517 | -0.018727 | 0.042448 | 2000 |
| global_stage2b | global_raw | 0.006684 | -0.024619 | 0.038018 | 2000 |
| global_stage2b | global_stage2a | -0.004833 | -0.020350 | 0.010515 | 2000 |

All comparisons use the same stratified patient-group resample for candidate and reference, retain all images from each selected group, and report negative as well as positive changes.

All paired bootstrap metrics and confidence intervals:

| candidate | reference | metric | delta_mean | delta_median | ci_lower_2_5 | ci_upper_97_5 | valid_repetitions |
|---|---|---|---|---|---|---|---|
| global_mask | global_only | macro_auc | -0.022067 | -0.022182 | -0.058881 | 0.014541 | 2000 |
| global_mask | global_only | accuracy | -0.022074 | -0.022000 | -0.074050 | 0.030000 | 2000 |
| global_mask | global_only | balanced_accuracy | -0.014376 | -0.014890 | -0.069448 | 0.040496 | 2000 |
| global_mask | global_only | macro_f1 | -0.021145 | -0.021182 | -0.072441 | 0.029349 | 2000 |
| global_mask | global_only | auc_normal | -0.011571 | -0.011598 | -0.053756 | 0.029998 | 2000 |
| global_mask | global_only | auc_mild | -0.036134 | -0.036795 | -0.090343 | 0.019471 | 2000 |
| global_mask | global_only | auc_severe | -0.018496 | -0.018629 | -0.068908 | 0.032465 | 2000 |
| global_raw | global_mask | macro_auc | 0.016766 | 0.017406 | -0.021140 | 0.052777 | 2000 |
| global_raw | global_mask | accuracy | 0.028117 | 0.028000 | -0.020000 | 0.074000 | 2000 |
| global_raw | global_mask | balanced_accuracy | -0.020445 | -0.020277 | -0.071298 | 0.030529 | 2000 |
| global_raw | global_mask | macro_f1 | 0.010791 | 0.011482 | -0.038354 | 0.060029 | 2000 |
| global_raw | global_mask | auc_normal | 0.027379 | 0.027612 | -0.017152 | 0.071544 | 2000 |
| global_raw | global_mask | auc_mild | 0.040342 | 0.040485 | -0.013091 | 0.089736 | 2000 |
| global_raw | global_mask | auc_severe | -0.017421 | -0.016806 | -0.070701 | 0.034575 | 2000 |
| global_stage2a | global_mask | macro_auc | 0.028283 | 0.027952 | -0.002233 | 0.059240 | 2000 |
| global_stage2a | global_mask | accuracy | 0.038587 | 0.038000 | -0.004000 | 0.082050 | 2000 |
| global_stage2a | global_mask | balanced_accuracy | 0.006614 | 0.006708 | -0.037579 | 0.052209 | 2000 |
| global_stage2a | global_mask | macro_f1 | 0.027183 | 0.027077 | -0.017096 | 0.070453 | 2000 |
| global_stage2a | global_mask | auc_normal | 0.046015 | 0.045613 | 0.018064 | 0.075099 | 2000 |
| global_stage2a | global_mask | auc_mild | 0.044483 | 0.044087 | -0.000539 | 0.089718 | 2000 |
| global_stage2a | global_mask | auc_severe | -0.005648 | -0.005288 | -0.051003 | 0.040181 | 2000 |
| global_stage2b | global_mask | macro_auc | 0.023450 | 0.023254 | -0.006966 | 0.054807 | 2000 |
| global_stage2b | global_mask | accuracy | 0.022580 | 0.022000 | -0.022000 | 0.066050 | 2000 |
| global_stage2b | global_mask | balanced_accuracy | 0.004260 | 0.004343 | -0.040416 | 0.049514 | 2000 |
| global_stage2b | global_mask | macro_f1 | 0.016332 | 0.016005 | -0.027795 | 0.058747 | 2000 |
| global_stage2b | global_mask | auc_normal | 0.046195 | 0.046290 | 0.018091 | 0.076638 | 2000 |
| global_stage2b | global_mask | auc_mild | 0.045865 | 0.045692 | 0.001663 | 0.090935 | 2000 |
| global_stage2b | global_mask | auc_severe | -0.021709 | -0.021364 | -0.070159 | 0.024727 | 2000 |
| global_stage2a | global_raw | macro_auc | 0.011517 | 0.010940 | -0.018727 | 0.042448 | 2000 |
| global_stage2a | global_raw | accuracy | 0.010470 | 0.010000 | -0.030000 | 0.050000 | 2000 |
| global_stage2a | global_raw | balanced_accuracy | 0.027059 | 0.027389 | -0.016426 | 0.071974 | 2000 |
| global_stage2a | global_raw | macro_f1 | 0.016393 | 0.016258 | -0.025200 | 0.059115 | 2000 |
| global_stage2a | global_raw | auc_normal | 0.018636 | 0.017967 | -0.020127 | 0.056738 | 2000 |
| global_stage2a | global_raw | auc_mild | 0.004141 | 0.003746 | -0.039586 | 0.047385 | 2000 |
| global_stage2a | global_raw | auc_severe | 0.011773 | 0.011421 | -0.030809 | 0.054998 | 2000 |
| global_stage2b | global_raw | macro_auc | 0.006684 | 0.006191 | -0.024619 | 0.038018 | 2000 |
| global_stage2b | global_raw | accuracy | -0.005537 | -0.004000 | -0.050000 | 0.036000 | 2000 |
| global_stage2b | global_raw | balanced_accuracy | 0.024705 | 0.024610 | -0.021603 | 0.068828 | 2000 |
| global_stage2b | global_raw | macro_f1 | 0.005542 | 0.005618 | -0.038909 | 0.047493 | 2000 |
| global_stage2b | global_raw | auc_normal | 0.018816 | 0.018464 | -0.017595 | 0.056331 | 2000 |
| global_stage2b | global_raw | auc_mild | 0.005523 | 0.005375 | -0.036747 | 0.048677 | 2000 |
| global_stage2b | global_raw | auc_severe | -0.004287 | -0.003897 | -0.046859 | 0.038449 | 2000 |
| global_stage2b | global_stage2a | macro_auc | -0.004833 | -0.004891 | -0.020350 | 0.010515 | 2000 |
| global_stage2b | global_stage2a | accuracy | -0.016007 | -0.016000 | -0.042000 | 0.010000 | 2000 |
| global_stage2b | global_stage2a | balanced_accuracy | -0.002353 | -0.001938 | -0.030050 | 0.024521 | 2000 |
| global_stage2b | global_stage2a | macro_f1 | -0.010851 | -0.010456 | -0.037088 | 0.014212 | 2000 |
| global_stage2b | global_stage2a | auc_normal | 0.000181 | 0.000023 | -0.016557 | 0.016715 | 2000 |
| global_stage2b | global_stage2a | auc_mild | 0.001382 | 0.001604 | -0.021803 | 0.025044 | 2000 |
| global_stage2b | global_stage2a | auc_severe | -0.016061 | -0.015932 | -0.038008 | 0.005932 | 2000 |

## Per-class behavior, confusion, and distribution shift

| candidate | reference | metric | oof_delta_candidate_minus_reference | fold_delta_mean | candidate_better_folds | reference_better_folds |
|---|---|---|---|---|---|---|
| global_mask | global_only | auc_normal | -0.011158 | 0.005534 | 2 | 3 |
| global_mask | global_only | auc_mild | -0.035921 | -0.021574 | 1 | 4 |
| global_mask | global_only | auc_severe | -0.018332 | -0.015447 | 2 | 3 |
| global_mask | global_only | recall_normal | 0.078261 | 0.078261 | 3 | 2 |
| global_mask | global_only | recall_mild | -0.025316 | -0.026241 | 3 | 2 |
| global_mask | global_only | recall_severe | -0.094595 | -0.095862 | 2 | 3 |
| global_raw | global_mask | auc_normal | 0.027194 | 0.012761 | 3 | 2 |
| global_raw | global_mask | auc_mild | 0.040542 | 0.037413 | 4 | 1 |
| global_raw | global_mask | auc_severe | -0.017909 | -0.001400 | 2 | 3 |
| global_raw | global_mask | recall_normal | -0.191304 | -0.191304 | 1 | 4 |
| global_raw | global_mask | recall_mild | 0.189873 | 0.190071 | 4 | 0 |
| global_raw | global_mask | recall_severe | -0.060811 | -0.060000 | 2 | 3 |
| global_stage2a | global_mask | auc_normal | 0.045443 | 0.021909 | 4 | 1 |
| global_stage2a | global_mask | auc_mild | 0.043943 | 0.034335 | 5 | 0 |
| global_stage2a | global_mask | auc_severe | -0.006795 | 0.001031 | 3 | 2 |
| global_stage2a | global_mask | recall_normal | -0.086957 | -0.086957 | 2 | 3 |
| global_stage2a | global_mask | recall_mild | 0.151899 | 0.151330 | 5 | 0 |
| global_stage2a | global_mask | recall_severe | -0.047297 | -0.048046 | 2 | 2 |
| global_stage2b | global_mask | auc_normal | 0.045692 | 0.026652 | 4 | 1 |
| global_stage2b | global_mask | auc_mild | 0.045579 | 0.033539 | 5 | 0 |
| global_stage2b | global_mask | auc_severe | -0.022746 | -0.014492 | 1 | 4 |
| global_stage2b | global_mask | recall_normal | -0.034783 | -0.034783 | 2 | 2 |
| global_stage2b | global_mask | recall_mild | 0.092827 | 0.092376 | 4 | 1 |
| global_stage2b | global_mask | recall_severe | -0.047297 | -0.048046 | 2 | 2 |
| global_stage2a | global_raw | auc_normal | 0.018250 | 0.009147 | 3 | 2 |
| global_stage2a | global_raw | auc_mild | 0.003401 | -0.003077 | 3 | 2 |
| global_stage2a | global_raw | auc_severe | 0.011114 | 0.002431 | 3 | 2 |
| global_stage2a | global_raw | recall_normal | 0.104348 | 0.104348 | 2 | 1 |
| global_stage2a | global_raw | recall_mild | -0.037975 | -0.038741 | 3 | 2 |
| global_stage2a | global_raw | recall_severe | 0.013514 | 0.011954 | 2 | 2 |
| global_stage2b | global_raw | auc_normal | 0.018498 | 0.013890 | 3 | 2 |
| global_stage2b | global_raw | auc_mild | 0.005038 | -0.003874 | 2 | 3 |
| global_stage2b | global_raw | auc_severe | -0.004837 | -0.013092 | 2 | 3 |
| global_stage2b | global_raw | recall_normal | 0.156522 | 0.156522 | 3 | 1 |
| global_stage2b | global_raw | recall_mild | -0.097046 | -0.097695 | 1 | 4 |
| global_stage2b | global_raw | recall_severe | 0.013514 | 0.011954 | 2 | 2 |
| global_stage2b | global_stage2a | auc_normal | 0.000248 | 0.004743 | 4 | 1 |
| global_stage2b | global_stage2a | auc_mild | 0.001636 | -0.000796 | 2 | 3 |
| global_stage2b | global_stage2a | auc_severe | -0.015951 | -0.015523 | 1 | 4 |
| global_stage2b | global_stage2a | recall_normal | 0.052174 | 0.052174 | 3 | 0 |
| global_stage2b | global_stage2a | recall_mild | -0.059072 | -0.058954 | 0 | 5 |
| global_stage2b | global_stage2a | recall_severe | 0.000000 | 0.000000 | 0 | 0 |

All OOF confusion matrices are under `reports/global_resnet18_optical_fusion/confusion_matrices/`. `feature_distribution_audit.csv` retains every fold/feature statistic.

Train/validation feature-shift summary:

| variant | mean_absolute_smd | maximum_absolute_smd |
|---|---|---|
| global_raw | 0.132153 | 0.413416 |
| global_stage2a | 0.125693 | 0.402391 |
| global_stage2b | 0.121862 | 0.386622 |

Stage 2B shift must be interpreted as an in-sample train versus out-of-sample validation risk and must not trigger post-hoc feature editing.

## Overfitting audit

| variant | fold | best_epoch | train_macro_auc_at_best | val_macro_auc_at_best | train_minus_val_auc_at_best | last_train_macro_auc | last_val_macro_auc |
|---|---|---|---|---|---|---|---|
| global_only | 0 | 2 | 0.901707 | 0.649138 | 0.252569 | 0.999532 | 0.615849 |
| global_only | 1 | 28 | 0.999948 | 0.752218 | 0.247731 | 0.999871 | 0.730023 |
| global_only | 2 | 14 | 0.998076 | 0.727089 | 0.270987 | 0.996442 | 0.684486 |
| global_only | 3 | 20 | 0.997267 | 0.681912 | 0.315355 | 0.999973 | 0.654043 |
| global_only | 4 | 10 | 0.997914 | 0.695578 | 0.302336 | 0.999920 | 0.659219 |
| global_mask | 0 | 20 | 0.995893 | 0.683814 | 0.312080 | 0.997407 | 0.570880 |
| global_mask | 1 | 2 | 0.881683 | 0.698147 | 0.183536 | 0.999150 | 0.641747 |
| global_mask | 2 | 1 | 0.641214 | 0.708646 | -0.067432 | 0.997159 | 0.621792 |
| global_mask | 3 | 5 | 0.988811 | 0.662780 | 0.326031 | 0.999943 | 0.648097 |
| global_mask | 4 | 6 | 0.994173 | 0.700071 | 0.294102 | 0.999992 | 0.674767 |
| global_raw | 0 | 9 | 0.998746 | 0.658998 | 0.339748 | 0.995493 | 0.621108 |
| global_raw | 1 | 10 | 0.998932 | 0.731644 | 0.267288 | 0.999536 | 0.651821 |
| global_raw | 2 | 1 | 0.649646 | 0.717664 | -0.068018 | 0.998918 | 0.636807 |
| global_raw | 3 | 2 | 0.862994 | 0.685634 | 0.177361 | 0.999980 | 0.650226 |
| global_raw | 4 | 24 | 0.999080 | 0.740806 | 0.258273 | 0.999197 | 0.641981 |
| global_stage2a | 0 | 5 | 0.989379 | 0.674228 | 0.315152 | 0.999809 | 0.636862 |
| global_stage2a | 1 | 10 | 0.999836 | 0.734486 | 0.265350 | 0.999970 | 0.640792 |
| global_stage2a | 2 | 1 | 0.647920 | 0.710082 | -0.062162 | 0.999046 | 0.662094 |
| global_stage2a | 3 | 10 | 0.998175 | 0.711438 | 0.286737 | 0.997750 | 0.655333 |
| global_stage2a | 4 | 6 | 0.994882 | 0.718682 | 0.276200 | 0.999955 | 0.645776 |
| global_stage2b | 0 | 5 | 0.988536 | 0.669018 | 0.319518 | 0.999931 | 0.631685 |
| global_stage2b | 1 | 10 | 0.998906 | 0.715340 | 0.283566 | 0.997856 | 0.650118 |
| global_stage2b | 2 | 1 | 0.648381 | 0.714511 | -0.066130 | 0.997968 | 0.668965 |
| global_stage2b | 3 | 11 | 0.999307 | 0.695479 | 0.303828 | 0.995391 | 0.669791 |
| global_stage2b | 4 | 6 | 0.995681 | 0.735273 | 0.260408 | 0.999963 | 0.693892 |

Train-minus-validation AUC gaps and best-versus-last behavior are descriptive only; outer validation was already used for checkpoint selection.

## OOF completeness

| variant | rows | unique_ids | folds | probabilities_valid |
|---|---|---|---|---|
| global_only | 500 | 500 | 0,1,2,3,4 | True |
| global_mask | 500 | 500 | 0,1,2,3,4 | True |
| global_raw | 500 | 500 | 0,1,2,3,4 | True |
| global_stage2a | 500 | 500 | 0,1,2,3,4 | True |
| global_stage2b | 500 | 500 | 0,1,2,3,4 | True |

## Interpretation and limitations

Outer validation was used every epoch for early stopping, so estimates may be optimistic and are not independent-test results. The availability-only control must be considered before attributing gains to optical phenotype values.

No single metric automatically declares a winner. Candidate selection must consider pooled and fold-average macro-AUC, at least 3/5 fold direction, macro-F1, balanced accuracy, per-class recall, confidence intervals, outlier-fold sensitivity, complexity, and Stage 2B shift. When G-A and G-B are close, the simpler Ridge-calibrated G-A is preferred.

## Tests, reproducibility, and integrity

Required test audit: 129 passed; protocol status: PASS; smoke status: PASS (5/5 variants).

Model/data/augmentation random streams were isolated by fold using base_seed=2026 and fold_seed=base_seed+fold, and were recorded in every checkpoint. Source hashes, scaler hashes, checkpoint hashes, and OOF hashes are recorded in the manifests. Stage 1/2A/2B, labels, splits, and historical image inputs were not modified.

## Final candidate guidance

G-A and G-B have close pooled macro-AUC (absolute delta < 0.01); the provisional candidate is G-A because Ridge is simpler and Stage 2B has greater shift risk.
