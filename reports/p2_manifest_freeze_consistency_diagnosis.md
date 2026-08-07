# P2 manifest freeze consistency diagnosis

Generated at: 2026-07-29

## Checked files

- Freeze metadata: `data/processed/P2_Counterfactual_Relighting500_v1/metadata/P2_COUNTERFACTUAL_TRAINING_ASSETS_FROZEN.json`
- Manifest: `data/processed/P2_Counterfactual_Relighting500_v1/manifests/p2_training_manifest.csv`

## Hash diagnosis

- `recorded_manifest_sha256`: `2a25bd48cd600d8922ec7b8e08d48853f2f569907e1e1845b9a46f7068d515d7`
- `current_manifest_sha256`: `ae0ee456e2e6bc4604bc1399c20b4b610f4b951595e9f5b5b154f4e4324486a6`
- `hash_match`: `false`
- `current_path_style`: `E:/projects/face2/...`
- `original_manifest_backup_found`: `false`
- `strict_inverse_prefix_recovery_available`: `true`
- `strict_inverse_recovered_sha256`: `2a25bd48cd600d8922ec7b8e08d48853f2f569907e1e1845b9a46f7068d515d7`

## Interpretation

The current manifest did not match the frozen SHA256 because the stored project-root prefix had been mechanically changed from `/mnt/e/projects/face2` to `E:/projects/face2`.

A strict inverse conversion of the known project-root prefix reproduces the exact frozen `manifest_sha256`. Before recovery, the candidate restored manifest was compared against the current manifest:

- 500 rows before and after;
- 500 unique cases before and after;
- `case_id` unchanged;
- `patient_group_id` unchanged;
- `fold` unchanged;
- `binary_label` unchanged;
- `p1_qc_flag` unchanged;
- `boundary_uncertain` unchanged;
- asset relative tail paths unchanged;
- only the known project-root path prefix changes.

Machine-readable audit:

- `metadata/p2_manifest_path_migration_audit.json`

## Decision

Use the preferred repair path:

1. Restore the manifest to the original `/mnt/e/projects/face2/...` stored path representation.
2. Keep P2-0 frozen asset semantics unchanged.
3. Implement Windows/WSL path compatibility in code read paths.
4. Do not re-run DECA, do not regenerate relighted RGB/shading, do not enter formal five-fold training.
