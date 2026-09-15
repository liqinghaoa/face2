# KM-BIO-v2R.1 R-D freeze and developmental Validation review

- Status: `R_D_COMPLETE_DIRECTION_CONSISTENT`
- Frozen candidate: `V2R-PS`
- Frozen globals: `A_s=1.483456`, `delta_bs=0.500000`, `g0=1.0`, `Dv=15.0 um`, `s0=0.7`
- Freeze Train-only; Validation used in freeze: `false`
- Decision state: `V2R_DEVELOPMENT_SPECTRAL_CANDIDATE / V2R_SPECTRAL_ONLY`
- Validation is 3 subjects: developmental spectral review only, no population estimate, no physiological claim.

## Freeze diagnostics (Train-global profiles)

| candidate   | parameter   |   frozen_value | identifiable   |   acceptable_normalized_span |   acceptable_envelope_low |   acceptable_envelope_high | at_boundary   |
|:------------|:------------|---------------:|:---------------|-----------------------------:|--------------------------:|---------------------------:|:--------------|
| V2R-PS      | A_s         |        1.48346 | False          |                          0.4 |                       1.2 |                        1.6 | False         |
| V2R-PS      | delta_bs    |        0.5     | False          |                          0.4 |                       0.1 |                        0.5 | True          |

## Direction consistency (Train vs Validation, frozen globals)

| scope   |   median_logrmse |   median_sam_deg |   median_rmse |   model_better_than_reference_fraction |   median_model_to_reference_error_ratio |   f_blood_upper_boundary_fraction |   f_mel_any_boundary_fraction |   edge_median_abs_signed_residual |
|:--------|-----------------:|-----------------:|--------------:|---------------------------------------:|----------------------------------------:|----------------------------------:|------------------------------:|----------------------------------:|
| train   |        0.0607521 |          3.43593 |     0.0181066 |                               0.863636 |                                0.446585 |                          0.477273 |                             0 |                          0.188965 |
| valid   |        0.0592076 |          3.53004 |     0.0191733 |                               1        |                                0.567239 |                          0        |                             0 |                          0.17701  |

### Checks

| family    | check                                         | pass   |
|:----------|:----------------------------------------------|:-------|
| spectral  | median_logrmse_not_regressed                  | True   |
| spectral  | median_sam_not_regressed                      | True   |
| spectral  | validation_beats_reference_majority           | True   |
| parameter | parameter_medians_within_extended_train_range | True   |
| parameter | edge_median_abs_signed_residual_change_ok     | True   |
| parameter | f_blood_upper_boundary_fraction_not_worse     | True   |

## Edge-band median signed residual comparison

|   wavelength_nm |   train_median_signed_residual |   validation_median_signed_residual |   abs_change | same_sign   |
|----------------:|-------------------------------:|------------------------------------:|-------------:|:------------|
|             400 |                    -0.188965   |                          -0.17701   |   0.0119554  | True        |
|             410 |                    -0.11442    |                          -0.119453  |   0.00503339 | True        |
|             690 |                    -0.0270166  |                          -0.0487696 |   0.021753   | True        |
|             700 |                     0.00544721 |                          -0.0175659 |   0.0230131  | False       |

## Carried-forward R-C1R parameter reliability

| candidate   | parameter   | classification   |
|:------------|:------------|:-----------------|
| V2R-PS      | f_mel       | unreliable       |
| V2R-PS      | f_blood     | unreliable       |
| V2R-PS      | s           | unreliable       |

## Validation side-pressure linkage (n=3, descriptive)

| cohort   | parameter   |   subject_count |   pearson_r_vs_side_log_difference |   pearson_p | inferential   | reason                                       |
|:---------|:------------|----------------:|-----------------------------------:|------------:|:--------------|:---------------------------------------------|
| valid    | f_mel       |               3 |                           0.153506 |    0.901887 | False         | three_validation_subjects_developmental_only |
| valid    | f_blood     |               3 |                          -0.801186 |    0.408405 | False         | three_validation_subjects_developmental_only |
