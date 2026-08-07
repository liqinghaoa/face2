# Global optical-fusion overfitting-control implementation report

## Status

`OVERFIT_CONTROL_STATUS=COMPLETE`

The implementation, 20-run formal matrix, 30 deterministic checkpoint evaluations,
descriptive summary and formal report are complete. Local Full inference reproduced
7/10 validation AUCs exactly; the three remaining fourth-decimal differences and the
user-authorized local override are retained explicitly in the audit and run manifest.

## Implemented experiment

- Variants: `global_only` (G0) and `global_stage2a` (G-A) only.
- New strategies: `frozen_backbone` and `partial_layer4` only.
- Formal matrix: 2 variants × 2 new strategies × 5 fixed outer folds = 20 runs.
- Full is a read-only reference. Full checkpoints are never used to initialize a new
  model and the existing Full experiment directory is not written.
- Every new formal run independently initializes the same ImageNet ResNet18.
- Frozen-backbone trains only the linear classifier. The verified trainable counts
  are 1,539 parameters for G0 and 1,560 for G-A.
- Partial-layer4 trains only `backbone.layer4` and the classifier. The verified
  trainable counts are 8,395,267 parameters for G0 and 8,395,288 for G-A.
- Frozen modules are returned to evaluation mode after every top-level training-mode
  transition. Frozen BatchNorm parameters and running statistics are checked after
  every training epoch.
- Frozen-backbone uses one classifier AdamW group at `1e-4`. Partial-layer4 uses
  exactly two AdamW groups: layer4 at `1e-5` and classifier at `1e-4`. Both use
  weight decay `1e-4`.
- G-A loads the canonical per-fold Full `feature_scaler.json`. Its source hash,
  train IDs, valid counts, means and population standard deviations are recomputed
  and checked; it is never refitted per strategy.
- Deterministic evaluation uses the validation transform with `shuffle=False` for
  both the 400-sample train split and the 100-sample validation split. Each fold
  exports predictions, metrics and train-minus-validation generalization gaps.
- Checkpoints contain strategy identity, exhaustive frozen/trainable parameter names,
  optimizer groups, BatchNorm modes, scaler provenance, split/config/code hashes,
  seeds, RNG state and software/runtime metadata. Resume rejects any mismatch.

## Validation performed

- New focused tests: 14 passed.
- Full repository regression suite: 149 passed.
- Structural/data protocol preflight: PASS.
- CPU smoke matrix: 4/4 passed, covering both strategies and both variants on fold 0.
- Every smoke run completed forward/backward propagation, gradient and BatchNorm
  checks, best/last checkpoint creation, actual resume loading, deterministic train
  and validation evaluation, logs and curves.
- Smoke trainable parameter counts and optimizer-group names matched the locked
  contracts.
- Historical Full file/checkpoint hashes were verified without modification.
- Saved historical validation predictions reproduce their recorded Full metrics
  exactly.

## Full reference reproduction note

Formal work used the local `Cp2Cp` environment: Python 3.10.20, PyTorch 2.5.1,
torchvision 0.20.1 and CUDA 12.1 on an RTX 4060 Laptop GPU. Seven Full folds matched
exactly; three differed by `1.59e-4` to `2.67e-4`, despite matching software versions.
The stored historical predictions still reproduce the recorded metrics exactly and
all historical hashes remained unchanged.

The complete formal report is:

`reports/global_optical_fusion_overfit_control/global_optical_fusion_overfit_control_report.md`

The complete summary manifest is:

`experiments/global_optical_fusion_overfit_control/summary/run_manifest.json`
