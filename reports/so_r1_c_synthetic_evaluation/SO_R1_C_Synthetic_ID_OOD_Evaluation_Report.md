# SO-R1-C Synthetic ID/OOD Evaluation

## Final decision

- Status: **FAIL**
- Pipeline validity: **PASS**
- Synthetic Gate: **FAIL**
- M channel: `CHANNEL_NOT_PASS`
- H channel: `CHANNEL_NOT_PASS`
- SO-R2 authorization: **false**; SO-R3 remains **false**.

The frozen Proposed checkpoint at lambda-pair 0.50 retained prediction quality, beat the Train-derived mean predictor, and showed no output collapse. It did not demonstrate a positive pre-frozen equal-weight primary OOD drift reduction: M was -0.55% (95% CI -3.17% to 1.96%) and H was -14.57% (95% CI -21.10% to -8.39%). Therefore neither channel satisfies the Synthetic Gate scientific criterion.

## Frozen provenance and integrity

- Baseline checkpoint SHA256: `5b1dd85f840d4d00a7c1c1f3b7c0a9306031408ecf0c5a557f53ca0810c0265b`
- Proposed checkpoint SHA256: `5cd86ceee6c9ad7f448676b59b3f1f5e8d80dc28bce106101b19d1286b6562a0`
- B1 contract: `7dcfb309d9d61c35b1de555b815a84e7c2f83be5d9b5e63f74bc91d14217b84f`
- B2 lambda-selection lock: `56a33e14bf2c81f8d8c34a06e6bbd5401c618a585120e39d4b0072ff00d385d1`
- D0-AM1 protocol: `fff5f3a01c6c486e0d982debc462f235f56a42033e9c7c4983d9b912fa84df45`
- G1-AM1 acceptance: `5639d5198af0d6854bc569e79dfd74e3276acf68a4b5c05836fef5f54bad3bfb`

All provenance gates passed. Pair-manifest validation covered 2,500 Test latent, 12,500 acquisitions, and 10,000 frozen pairs (ID 4,000; each OOD split 2,000). Protected-asset hash audits before and after evaluation were identical. Train RGB and Validation RGB accesses were both zero; the only Train access was 10,000 target/mask reads for the permitted coordinate-wise mean predictor.

## Primary OOD result

| Channel | Equal-weight OOD relative drift reduction | 95% latent-bootstrap CI | Gate condition |
|---|---:|---:|---|
| M-sensitive | -0.55% | [-3.17%, 1.96%] | Fail: lower bound is not > 0 |
| H-sensitive | -14.57% | [-21.10%, -8.39%] | Fail: lower bound is not > 0 |

The three equally weighted strata were Camera-OOD A0-A1, Light-OOD A0-A2, and Joint-OOD A0-A4, each with 500 latent and 10,000 bootstrap replicates.

## Prediction-quality safeguards

OOD prediction-quality non-inferiority passed for both channels.

| Channel | Baseline MAE | Proposed-Baseline MAE delta | one-sided 95% upper CI | 5% margin |
|---|---:|---:|---:|---:|
| M-sensitive | 0.02338 | -0.00108 | -0.00089 | 0.00117 |
| H-sensitive | 0.14561 | -0.00896 | -0.00724 | 0.00728 |

Both models outperformed the Train-derived mean predictor in every Test split/channel. Proposed did not collapse: its minimum prediction/target variance ratio was 0.07364, and its maximum boundary-saturation fraction was 0.0004144, both safely clear of the pre-frozen invalidity limits. Within/between audits passed in every proposed split/channel stratum (within < between proportions 0.6295-0.9498).

## Detailed artifacts

- `prediction_quality_latent_metrics.csv` and `prediction_quality_summary.csv`: latent-level M/H quality metrics for all roles and Test splits.
- `pair_drift_latent_metrics.csv` and `pair_drift_summary.csv`: complete Baseline/Proposed × split × channel × pair matrix.
- `baseline_vs_proposed_paired_bootstrap.json`: all paired drift comparisons.
- `ood_primary_endpoint_bootstrap.json`, `ood_noninferiority_bootstrap.json`, `mean_predictor_audit.json`, `collapse_audit.json`, and `within_between_audit.json`: Gate evidence.
- `protected_asset_hash_audit.json` and `split_access_audit.json`: validity and read-only evidence.

Conclusion boundary: this is evidence only for the frozen **synthetic** camera spectral response, light SPD, exposure, shading, and specular conditions. It does not establish real-device, real-camera-vendor, or real-environment generalization.
