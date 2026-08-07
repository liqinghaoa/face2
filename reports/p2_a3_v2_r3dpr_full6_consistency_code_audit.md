# P2-A3-v2 R3DPR Full6Consistency Code Audit

## Scope

Formal experiment display ID: `P2-A3-v2_R3DPR_Full6Consistency_MeanBG`.

This audit separates the new R3DPR image-directory A3-v2 experiment from the older DECA/NPZ P2-A3 implementation.

## Reused Components

- A2-v2 manifest and path mapping: `data/processed/global_face/SixRelighting_OriginalCamera_meanfg/p2_a2_v2_manifest.csv`.
- Fixed preset order: `p2_counterfactual.assets.PRESET_NAMES`.
- Windows/WSL-compatible path resolution: `p2_counterfactual.path_utils`.
- Shared single-RGB ResNet18 model: `p2_counterfactual.model.P2ASingleRGBResNet18`.
- Weighted cross-entropy class weighting from training fold labels only.
- Original-only checkpoint selection and evaluator.
- Six-relighted independent evaluator and OOF feature export.
- Fold and experiment success validation/aggregation.
- Patient-cluster paired bootstrap comparison utilities.

## Old A3 Logic Reused

- Symmetric Jensen-Shannon prediction consistency.
- Feature cosine consistency on 512-dimensional ResNet18 pooled features.
- Five-epoch linear consistency warm-up.
- Synchronized horizontal flip concept from paired transforms.
- Shared backbone and classifier head for all views.

## Required Extension

The old `paired` mode sampled one relighted preset per case. A3-v2 requires full-six training:

- Dataset length remains the number of cases.
- Each training item returns one original image plus all six relighted images.
- The trainer concatenates the seven views and performs one shared forward pass.
- Classification loss uses original CE and averaged six-relight CE with 50/50 weighting.
- Prediction consistency averages six original-vs-relight JS divergences.
- Feature consistency averages six original-vs-relight cosine losses.

## New Configuration And Output Isolation

- Config: `config/p2/p2_a/p2_a3_v2_r3dpr_full6_consistency_meanbg.yaml`.
- Output root: `experiments/500Data/P2_A3_v2_R3DPR_Full6Consistency_MeanBG`.
- Smoke root: `experiments/500Data/P2_A3_v2_R3DPR_Full6Consistency_MeanBG/_smoke`.

The new experiment ID is `p2_a3_v2_r3dpr_full6_consistency_meanbg`, so checkpoints, logs, OOF files, summaries, and reports do not overwrite A2-v2, A1, old A2, old A3, or old DECA outputs.

## Data Boundary

The new implementation uses image-directory columns (`original_path` and `relight_<preset>_path`). It must not read old `relighted_images` from DECA/NPZ assets for A3-v2.
