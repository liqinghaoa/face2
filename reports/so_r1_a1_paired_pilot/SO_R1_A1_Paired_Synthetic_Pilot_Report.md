# SO-R1-A1 Paired Synthetic Pilot Report

## Result

- Final status: **PASS**
- Pilot scale: 64 latent groups, 320 acquisitions, 256 reference pairs.
- SO-0 camera path: `SO0NumpyForwardModel.render_camera` with frozen ColorChecker matrices, camera response, white balance, Bradford adaptation and display sRGB clipping; SO-R1 input is inverse-sRGB of the clipped display copy.
- Camera/light freeze and 32-pair allowlist: verified; coverage True.
- Variable isolation: True; M/H/mask group identity: True.
- RGB variation gate: True; image/clipping gate: True.
- Float16 candidate storage authorized: True (max 0.000244, p99 0.000221).

## Inputs and implementation boundary

- Frozen camera/light source: `config/so_r1/frozen_camera_light_split_v1.yaml`; 6 seen / 2 unseen cameras, 3 seen / 1 unseen lights.
- Allowlist source: `reports/so_r1_a0_camera_light_selection/camera_light_32pair_allowlist.csv` and JSON counterpart; all 32 pairs are used.
- SO-0 formal configuration/range source: `src/skin_optics_so0/configs/so0/forward_model_mvp.yaml` (M/H controls [0, 1], shading SO-0 sensitivity range, specular SO-0 sensitivity maximum, exposure EV [-0.5, +0.5] constrained to SO-0 [0.1, 3.0]).
- Spatial-field orchestrator: `src/skin_optics_so_r1/paired_pilot.py`; it uses separate SHA-256-derived PCG64 streams for M, H, mask and three appearance packages. It does not duplicate the SO-0 optical model.
- Disk use for canonical Pilot: 354.5 MiB. The recorded first-run timing and 67,500-image extrapolation are in `PILOT_ACCEPTANCE.json` / `run_manifest.json`.

## Dataset and split contract

| Split | Latents | Acquisitions |
|---|---:|---:|
| train_like_id | 24 | 120 |
| validation_like_id | 8 | 40 |
| id_test | 8 | 40 |
| camera_ood | 8 | 40 |
| light_ood | 8 | 40 |
| joint_ood | 8 | 40 |

Each latent contains A0 reference, A1 camera-only, A2 light-only, A3 appearance-only and A4 joint. Domain roles are computed per acquisition, not copied from the latent split. `camera_light_coverage.csv` confirms 32/32 coverage. `latent_manifest.csv`, `acquisition_manifest.csv` and `pair_manifest.csv` provide the full 64/320/256 record sets.

## Audit results

- M/H mean correlations: Pearson -0.088; Spearman -0.061.
- Every pair preserves M/H/mask hashes and satisfies its expected variable isolation rule.
- Pair drift gates passed for camera, light, appearance and joint changes; full distribution summaries are in the pair manifest and `pair_rgb_drift_distribution.png`.
- All arrays are finite; quality metrics use the valid skin mask, so protocol-required mask-outside zeros are not misclassified as black image failures.
- Float16 candidate round-trip passed; float32 remains the authoritative Pilot record.
- Deterministic run01/run02 authoritative content hashes are identical; PNG previews are intentionally excluded because they are non-authoritative visual encodings.
- Test suite: `pytest -q tests/so_r1` — 7 passed.

## Protocol boundaries

M/H are shared synthetic targets only within each latent. Shading and specular are appearance nuisances; no geometry, facial normals, or precise light directions are modeled. This Pilot validates synthetic orchestration and numerical interfaces only. It does not establish M/H identifiability, real smartphone robustness, OOD generalization, or cardiac classification value.

## Reproducibility and protected inputs

Root seed is `20260822` using PCG64 streams derived with SHA-256. Frozen SO-0 assets, SO-R1-A0 freeze, allowlist and evidence are protected by before/after manifests. Detailed CSV manifests, per-latent fields, appearances, arrays, previews and QC figures are in `data/processed/SO_R1_A1_PairedPilot_v1` and `reports/so_r1_a1_paired_pilot`.
