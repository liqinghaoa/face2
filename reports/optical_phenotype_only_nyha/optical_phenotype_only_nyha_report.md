# Optical Phenotype-Only NYHA Three-Class Classification

## Completion status

**COMPLETE** — 4 variants × 5 fixed outer folds = 20 formal multinomial logistic-regression models. Determinism: **PASS**.

## Experimental question and boundaries

This experiment measures how much NYHA three-class signal is present in six regional optical phenotypes alone. It does not load images, instantiate ResNet, use global image features, EXIF, camera identity, clinical covariates, acquisition-condition predictions, residuals, QC fields, PCA, resampling, feature selection, threshold tuning, outer-validation tuning, hyperparameter search, or bootstrap inference.

The fixed class order is normal (0), mild (1), severe (2). Official NYHA labels were mapped as 0→0, 1/2→1, 3/4→2 and verified against the unchanged master split.

## Cohort and fixed protocol

The cohort contains 500 unique image IDs, 483 patient groups, and class counts 115/237/148. Every outer fold contains 400 training and 100 validation cases, with disjoint patient groups; each ID is OOF exactly once. All 14 forehead-unavailable cases were retained.

## Variants and preprocessing

- O-Mask: forehead availability only (dimension 1).
- O-Raw: six raw optical phenotypes plus availability (dimension 7).
- O-A: six fold-matched Stage 2A calibrated phenotypes plus availability (dimension 7).
- O-B: six fold-matched Stage 2B calibrated phenotypes plus availability (dimension 7).

For each six-dimensional variant and fold, means and population standard deviations (ddof=0) were fitted on the 400 outer-training cases only. Cheek statistics used all training cases; forehead-minus-cheek statistics used only available training cases. Unavailable forehead values remained NaN in source data and became zero only after standardization; availability was appended unscaled.

Stage 2A and Stage 2B classifier inputs came exclusively from their fold-specific train/validation files. Their OOF summary tables were explicitly rejected as classifier sources. Availability was identical across raw, Stage 2A and Stage 2B sources.

## Locked classifier

Each model is scikit-learn LogisticRegression with L2 penalty, C=1.0, lbfgs, intercept, balanced training-only class weights, max_iter=5000, tol=1e-8, float64 CPU input and random_state=2026+fold. With scikit-learn 1.7, the deprecated explicit multi_class argument is omitted; lbfgs on three classes uses the multinomial objective. All models converged without ConvergenceWarning.
The locked multinomial model is `P(y=c|x)=exp(β_cᵀx+b_c)/Σ_j exp(β_jᵀx+b_j)` and minimizes weighted multinomial log loss plus L2 regularization. Balanced weights are recomputed from each 400-case training fold only; C is fixed at 1.0 and no validation-guided selection is performed.

## Main results

| Variant | Fold mean Macro-AUC | Fold SD | Pooled OOF Macro-AUC | Severe AUC | Severe recall | Mean train−val Macro-AUC gap |
|---|---:|---:|---:|---:|---:|---:|
| o_mask | 0.5128 | 0.0156 | 0.5034 | 0.5059 | 0.7838 | -0.0000 |
| o_raw | 0.6626 | 0.0383 | 0.6573 | 0.5714 | 0.3446 | 0.0287 |
| o_stage2a | 0.5268 | 0.0647 | 0.5211 | 0.5364 | 0.3243 | 0.0469 |
| o_stage2b | 0.5492 | 0.0756 | 0.5491 | 0.5413 | 0.3041 | 0.0299 |

Fold means summarize five validation folds; pooled metrics are recomputed from all 500 OOF predictions and are not a simple fold average.

### Per-fold validation Macro-AUC

| Variant | Fold 0 | Fold 1 | Fold 2 | Fold 3 | Fold 4 |
|---|---:|---:|---:|---:|---:|
| o_mask | 0.5351 | 0.5223 | 0.5061 | 0.4967 | 0.5036 |
| o_raw | 0.6545 | 0.6936 | 0.6782 | 0.5995 | 0.6873 |
| o_stage2a | 0.5194 | 0.5706 | 0.5570 | 0.4172 | 0.5696 |
| o_stage2b | 0.5144 | 0.6302 | 0.5757 | 0.4362 | 0.5892 |

### Five-fold Macro-AUC distribution

| Variant | Mean | SD (ddof=1) | Median | Min | Max |
|---|---:|---:|---:|---:|---:|
| o_mask | 0.5128 | 0.0156 | 0.5061 | 0.4967 | 0.5351 |
| o_raw | 0.6626 | 0.0383 | 0.6782 | 0.5995 | 0.6936 |
| o_stage2a | 0.5268 | 0.0647 | 0.5570 | 0.4172 | 0.5706 |
| o_stage2b | 0.5492 | 0.0756 | 0.5757 | 0.4362 | 0.6302 |

### Pooled OOF classification metrics

| Variant | Accuracy | Balanced accuracy | Macro-F1 | Macro-AUC | Severe-vs-rest AUC |
|---|---:|---:|---:|---:|---:|
| o_mask | 0.3360 | 0.3448 | 0.2700 | 0.5034 | 0.5059 |
| o_raw | 0.4440 | 0.4703 | 0.4434 | 0.6573 | 0.5714 |
| o_stage2a | 0.3680 | 0.3486 | 0.3472 | 0.5211 | 0.5364 |
| o_stage2b | 0.3800 | 0.3620 | 0.3596 | 0.5491 | 0.5413 |

### Pooled per-class AUC and recall

| Variant | Normal AUC/recall | Mild AUC/recall | Severe AUC/recall |
|---|---:|---:|---:|
| o_mask | 0.5087/0.0609 | 0.4956/0.1899 | 0.5059/0.7838 |
| o_raw | 0.7824/0.6696 | 0.6180/0.3966 | 0.5714/0.3446 |
| o_stage2a | 0.4979/0.2870 | 0.5290/0.4346 | 0.5364/0.3243 |
| o_stage2b | 0.5632/0.3304 | 0.5430/0.4515 | 0.5413/0.3041 |

O-Raw was numerically higher than O-Mask for Macro-AUC in all five folds and produced the strongest optical-only pooled Macro-AUC. However, O-Raw severe recall was numerically lower than the missingness-only control, showing that a higher aggregate AUC did not improve every endpoint. O-A was numerically lower than O-Raw in aggregate; O-B recovered part, but not all, of that difference. Fold variability was substantial for both calibrated variants.

## Descriptive paired fold contrasts

The following are descriptive five-fold differences only; no p-values, confidence intervals, or causal claims are made.

| Contrast | Mean Δ Macro-AUC | SD | Positive folds |
|---|---:|---:|---:|
| o_stage2a - o_raw | -0.1358 | 0.0268 | 0/5 |
| o_stage2b - o_raw | -0.1135 | 0.0389 | 0/5 |
| o_stage2b - o_stage2a | 0.0224 | 0.0233 | 4/5 |

Train−validation Macro-AUC gaps are defined as training Macro-AUC minus validation Macro-AUC and are reported per fold and as fold means. A smaller gap is diagnostic only and is not automatically considered a better model.

## Coefficient and feature-distribution audits

Coefficient stability covers 78 class–feature combinations across five folds, including O-Mask availability coefficients and intercepts. The machine-readable table reports mean, sample SD, median, extrema, mean absolute coefficient, positive/negative/zero fold counts and sign consistency. These are standardized class-logit coefficients, not odds ratios, causal effects, or independent clinical predictors.
The feature audit contains 440 fold/variant/role/scale/feature rows with valid and missing counts, mean, ddof=0 SD, quartiles and extrema. It verifies finite model inputs, train-only standardization, unchanged availability, and zero-filled unavailable forehead dimensions. Any train/validation shift is reported rather than used to delete cases or refit preprocessing.

## Fusion context and phenotype interpretation

The existing fusion results were loaded only after optical-only training. G-Raw, G-A and G-B therefore provide read-only context, not tuning targets or a capacity-matched comparison. O-A's weak pooled discrimination and lower severe recall than O-Raw are directionally compatible with some loss or redistribution of standalone phenotype signal, but they do not uniquely explain the Full G-A mild prediction shift or the severe-recall change relative to the image baseline. Image–optical interaction, optimization and fold variation remain alternatives.

## Interpretation constraints

O-Mask is a missingness-only control. O-Raw estimates signal in uncalibrated optical phenotypes; O-A and O-B assess whether fixed, fold-matched calibration changes that signal. A calibration contrast does not by itself prove removal of acquisition bias or a causal biological mechanism. Coefficient magnitude is interpreted only in standardized input space, and sign/magnitude instability across folds is reported separately.

The read-only fusion context is included only for orientation. A difference between optical-only and image-fusion performance cannot be attributed uniquely to image–optical interaction, because optimization, representation, and fold variability also differ.
Forehead availability itself may act as a shortcut if missingness correlates with acquisition or cohort structure; O-Mask quantifies this risk but cannot prove its mechanism. Weak independent classification also does not rule out complementary value in fusion, because a feature can modify image decisions without being a strong standalone classifier.

## Robustness and audit evidence

All 20 joblib files were reloaded and reproduced train/validation probabilities within 1e-12. The entire 20-fit core was repeated in a temporary directory: PASS. The audit compared scalers, coefficients/intercepts, probabilities, metrics-equivalent predictions, and canonical prediction CSV hashes. Binary joblib hashes are recorded but are not treated as the primary reproducibility criterion.

Feature distributions before and after standardization, per-fold coefficients, convergence iterations, source hashes, split hashes, and artifact hashes are machine-readable. No historical input or historical experiment output was modified by this run.
Unit tests: 9 task-specific tests passed; full-project regression tests: 158 passed. Protocol-only: PASS across all 18 locked checks; temporary smoke: PASS for all four variants on fold 0.

## Limitations

This experiment has no independent external test set. The five outer folds remain internal cross-validation, and all comparisons are exploratory classification evidence. Numerically higher values are not evidence of statistical significance. The cohort is small, O-Mask can exploit non-random missingness, coefficients may vary with correlated phenotypes, and the linear classifier cannot represent arbitrary interactions. No automatic winner was selected.

## Outputs and next step

Use the OOF tables for sample-level inspection, metrics_summary.csv and pooled_oof_metrics.csv for performance, descriptive_pairwise_summary.csv for calibration contrasts, coefficient_stability.csv for fold stability, and feature_distribution_audit.csv for preprocessing checks. The appropriate next step is interpretation of these locked results alongside the existing fusion experiment, without retuning this optical-only protocol post hoc.

Fusion context status: `read_only_context`.

## Read-only Global fusion context

| Fusion variant | Pooled Macro-AUC | Mild recall | Severe AUC | Severe recall |
|---|---:|---:|---:|---:|
| global_raw | 0.6875 | 0.6034 | 0.6142 | 0.3514 |
| global_stage2a | 0.6985 | 0.5654 | 0.6253 | 0.3649 |
| global_stage2b | 0.6938 | 0.5063 | 0.6093 | 0.3649 |

These numbers come from the pre-existing formal fusion summary and were not read during classifier fitting. They are shown to relate information sources only; the low-dimensional logistic models and end-to-end ResNet18 fusion models have different capacities.
