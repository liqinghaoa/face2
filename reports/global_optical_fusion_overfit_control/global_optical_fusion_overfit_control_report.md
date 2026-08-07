# Global optical-fusion overfitting-control report

## 1. Completion and acceptance status

`OVERFIT_CONTROL_STATUS=COMPLETE`

All 20 new formal runs and all 30 deterministic train/validation evaluations are complete. Protocol, unit/regression tests and the four-run smoke suite passed.

## 2. Synchronized inputs and Full reference

The fixed 500-subject data, five outer folds, Stage 2A artifacts and ten Full checkpoints were consumed read-only. All historical hashes remained unchanged. Local Full inference reproduced 7/10 validation AUCs exactly; three folds differed by only 1.59e-4 to 2.67e-4. The user explicitly authorized local continuation, and the failed audit plus override are retained in the manifest rather than rewritten as PASS.

## 3. Data, variants and fixed folds

Each outer fold contains 400 training and 100 validation samples with disjoint patient groups and all three NYHA classes. G0 uses the global face image only. G-A concatenates the same 512-dimensional image representation with the six Stage 2A calibrated optical features and forehead-availability indicator. G-A reuses the canonical Full fold scaler; no strategy-specific refit occurs.

## 4. Trainability strategies and BatchNorm

Full is the historical all-parameter reference. Frozen trains only the classifier (1,539 parameters for G0; 1,560 for G-A), keeps the complete backbone in eval mode and freezes every BatchNorm affine/running statistic. Partial trains only layer4 plus the classifier (8,395,267 parameters for G0; 8,395,288 for G-A); conv1/bn1/layer1-layer3 remain eval/frozen while layer4 BatchNorm updates.

Frozen uses one AdamW classifier group at 1e-4. Partial uses exactly two groups: layer4 at 1e-5 and classifier at 1e-4. Weight decay is 1e-4 for every group. Every epoch asserts the requires-grad set, optimizer parameter set, gradients and BatchNorm modes/states.

## 5. Training and deterministic evaluation protocol

All new runs independently start from ImageNet ResNet18; no Full checkpoint is used for initialization. The locked budget is batch size 16, at most 50 epochs, weighted cross-entropy, no scheduler/warmup/AMP/clipping/label smoothing, strict best validation Macro-AUC and patience 10. Seeds are 2026+fold. Training uses resize 224 and horizontal flip only; deterministic train/validation evaluation uses the validation transform and shuffle=false.

## 6. Main descriptive comparison

| Strategy | Model | Fold AUC mean | Fold AUC std | Pooled OOF AUC | Deterministic train AUC | Gap | Best epoch | Val drop | BA | Macro-F1 | Severe AUC | Severe Recall |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| full | G0 | 0.701096 | 0.040016 | 0.692729 | 0.994601 | 0.293505 | 14.80 | 0.032463 | 0.529685 | 0.507756 | 0.650376 | 0.506757 |
| full | G-A | 0.709751 | 0.022115 | 0.698473 | 0.973171 | 0.263420 | 6.40 | 0.061611 | 0.521683 | 0.513584 | 0.625269 | 0.364865 |
| frozen_backbone | G0 | 0.663768 | 0.028283 | 0.661623 | 0.729340 | 0.065572 | 45.00 | 0.001674 | 0.490323 | 0.465344 | 0.599374 | 0.358108 |
| frozen_backbone | G-A | 0.660262 | 0.031616 | 0.659142 | 0.732626 | 0.072364 | 47.00 | 0.000505 | 0.472086 | 0.452835 | 0.603386 | 0.351351 |
| partial_layer4 | G0 | 0.687552 | 0.031539 | 0.676829 | 0.976400 | 0.288848 | 16.60 | 0.009513 | 0.496927 | 0.482747 | 0.637266 | 0.358108 |
| partial_layer4 | G-A | 0.682331 | 0.035997 | 0.676901 | 0.986250 | 0.303920 | 13.40 | 0.009285 | 0.489923 | 0.477050 | 0.633964 | 0.371622 |

## 7. G-A minus G0 within each strategy

| Strategy | Fold mean AUC Δ | Pooled AUC Δ | Gap Δ | BA Δ | Macro-F1 Δ | Severe AUC Δ | Severe Recall Δ | G-A better folds |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| full | 0.008655 | 0.005744 | -0.030086 | -0.008002 | 0.005829 | -0.025107 | -0.141892 | 3 |
| frozen_backbone | -0.003506 | -0.002481 | 0.006792 | -0.018237 | -0.012509 | 0.004012 | -0.006757 | 3 |
| partial_layer4 | -0.005221 | 0.000072 | 0.015072 | -0.007004 | -0.005697 | -0.003302 | 0.013514 | 1 |

## 8. Per-fold checkpoint and overfitting diagnostics

| Strategy | Model | Fold | Best epoch | Completed | Train AUC | Val AUC | Gap | Val drop | Severe AUC | Severe Recall |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| full | G0 | 0 | 2 | 12 | 0.973090 | 0.648871 | 0.324219 | 0.033288 | 0.594463 | 0.379310 |
| full | G0 | 1 | 28 | 38 | 1.000000 | 0.752218 | 0.247782 | 0.022194 | 0.718310 | 0.655172 |
| full | G0 | 2 | 14 | 24 | 0.999972 | 0.726901 | 0.273071 | 0.042603 | 0.723333 | 0.766667 |
| full | G0 | 3 | 20 | 30 | 0.999953 | 0.681912 | 0.318041 | 0.027869 | 0.644286 | 0.533333 |
| full | G0 | 4 | 10 | 20 | 0.999992 | 0.695578 | 0.304413 | 0.036360 | 0.604762 | 0.200000 |
| full | G-A | 0 | 5 | 15 | 0.997362 | 0.674228 | 0.323135 | 0.037366 | 0.599320 | 0.172414 |
| full | G-A | 1 | 10 | 20 | 0.999963 | 0.734486 | 0.265477 | 0.093693 | 0.655658 | 0.275862 |
| full | G-A | 2 | 1 | 11 | 0.868612 | 0.709923 | 0.158689 | 0.047988 | 0.663333 | 0.666667 |
| full | G-A | 3 | 10 | 20 | 1.000000 | 0.711438 | 0.288562 | 0.056105 | 0.652381 | 0.400000 |
| full | G-A | 4 | 6 | 16 | 0.999916 | 0.718682 | 0.281235 | 0.072905 | 0.641905 | 0.300000 |
| frozen_backbone | G0 | 0 | 39 | 49 | 0.724724 | 0.647065 | 0.077659 | 0.005573 | 0.611462 | 0.241379 |
| frozen_backbone | G0 | 1 | 48 | 50 | 0.716648 | 0.709524 | 0.007125 | 0.000625 | 0.610005 | 0.413793 |
| frozen_backbone | G0 | 2 | 50 | 50 | 0.731667 | 0.646822 | 0.084845 | 0.000000 | 0.609524 | 0.400000 |
| frozen_backbone | G0 | 3 | 48 | 50 | 0.731920 | 0.642438 | 0.089482 | 0.000163 | 0.596667 | 0.433333 |
| frozen_backbone | G0 | 4 | 40 | 50 | 0.741741 | 0.672993 | 0.068749 | 0.002009 | 0.582381 | 0.300000 |
| frozen_backbone | G-A | 0 | 35 | 45 | 0.723293 | 0.648966 | 0.074327 | 0.002527 | 0.613890 | 0.275862 |
| frozen_backbone | G-A | 1 | 50 | 50 | 0.714366 | 0.713553 | 0.000814 | 0.000000 | 0.613405 | 0.482759 |
| frozen_backbone | G-A | 2 | 50 | 50 | 0.746202 | 0.630610 | 0.115592 | 0.000000 | 0.601429 | 0.333333 |
| frozen_backbone | G-A | 3 | 50 | 50 | 0.739048 | 0.647950 | 0.091098 | 0.000000 | 0.627619 | 0.500000 |
| frozen_backbone | G-A | 4 | 50 | 50 | 0.740221 | 0.660233 | 0.079987 | 0.000000 | 0.550000 | 0.166667 |
| partial_layer4 | G0 | 0 | 8 | 18 | 0.956438 | 0.637354 | 0.319083 | 0.020509 | 0.610976 | 0.344828 |
| partial_layer4 | G0 | 1 | 12 | 22 | 0.989341 | 0.715700 | 0.273641 | 0.016302 | 0.663429 | 0.310345 |
| partial_layer4 | G0 | 2 | 31 | 41 | 1.000000 | 0.711863 | 0.288137 | 0.006763 | 0.695714 | 0.466667 |
| partial_layer4 | G0 | 3 | 6 | 16 | 0.936222 | 0.692468 | 0.243755 | 0.002835 | 0.664762 | 0.366667 |
| partial_layer4 | G0 | 4 | 26 | 36 | 1.000000 | 0.680375 | 0.319625 | 0.001157 | 0.597143 | 0.300000 |
| partial_layer4 | G-A | 0 | 8 | 18 | 0.954407 | 0.621046 | 0.333362 | 0.008745 | 0.607091 | 0.275862 |
| partial_layer4 | G-A | 1 | 19 | 29 | 0.999990 | 0.709247 | 0.290743 | 0.011427 | 0.654687 | 0.310345 |
| partial_layer4 | G-A | 2 | 15 | 25 | 0.997802 | 0.694735 | 0.303067 | 0.001683 | 0.658571 | 0.533333 |
| partial_layer4 | G-A | 3 | 16 | 26 | 0.999181 | 0.680908 | 0.318273 | 0.017980 | 0.651905 | 0.433333 |
| partial_layer4 | G-A | 4 | 9 | 19 | 0.979871 | 0.705717 | 0.274154 | 0.006588 | 0.622381 | 0.300000 |

## 9. Memorization speed, validation drop and training curves

Partial reaches near-perfect training AUC rapidly, while Frozen generally does not reach the 0.90/0.95/0.99 thresholds. Exact first-hit epochs, best/last validation drops, elapsed time and GPU peaks are provided in `summary/memorization_speed_summary.csv`, `checkpoint_epoch_summary.csv` and `overfit_metrics_by_fold.csv`. Per-run training curves are exported under `reports/global_optical_fusion_overfit_control/training_curves/`.

## 10. Interpretation: Frozen

Frozen reduced the mean G0 gap from 0.293505 to 0.065572, but fold-mean validation AUC fell from 0.701096 to 0.663768. For G-A, the gap fell from 0.263420 to 0.072364, while AUC fell from 0.709751 to 0.660262. This is capacity-limited underfitting, not a preferred overall model.

## 11. Interpretation: Partial layer4

Partial G0 retained a large gap (0.288848) and its fold-mean AUC (0.687552) remained below Full (0.701096). Partial G-A increased the gap to 0.303920 versus Full G-A 0.263420, with lower AUC (0.682331 versus 0.709751). Partial therefore did not solve the observed overfitting.

## 12. Stage 2A stability and severe-class cost

G-A improved fold-mean/pooled AUC only for Full (+0.008655/+0.005744). Under Frozen it changed them by -0.003506/-0.002481; under Partial by -0.005221/+0.000072 and won only one of five folds. Thus the Stage 2A increment is not robust to backbone-trainability strategy. Full G-A also reduced severe recall by 0.141892 versus G0. No aggregate improvement claim is made without reporting this severe-class cost.

## 13. Tests, environment and integrity

The final regression suite passed 149 tests; protocol status and the 4/4 smoke/resume suite passed. Formal runs used Python 3.10.20, PyTorch 2.5.1, torchvision 0.20.1, CUDA 12.1 and an NVIDIA GeForce RTX 4060 Laptop GPU. Full was not retrained; Stage 1/2A/2B and historical Full artifacts were not modified. Checkpoint, scaler, split, prediction and code/config hashes are recorded in the run manifests.

## 14. Limitations

Three Full validation AUCs differed at the fourth decimal place on local inference despite matching software versions; the override is explicit. The experiment is limited to one architecture, one 500-sample cohort and fixed folds. Validation folds were used for early stopping, and no external test set was available.

## 15. Statistical scope and next steps

This stage is descriptive only: no bootstrap intervals, significance tests or automatic winner selection were performed. The next useful step is external validation or a nested protocol focused on severe-class calibration, rather than selecting Frozen merely for its smaller gap or treating the negligible Partial G-A pooled difference as evidence of incremental value.
