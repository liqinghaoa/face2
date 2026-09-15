# KM-BIO-v2R.1 R-C1R complete identifiability audit and V2R-BEST-O

- Status: `R_C1R_COMPLETE`
- Scope: Train-only, fixed R-C0R fold-global parameters, no Validation/Test/500/RGB access.
- Frozen fold SHA-256: `e763a9f3dc7f58babe264f7056699c3d5b3722187fbb558f83f2fdb5d6b67370`
- Inversions reported above one: `0` (no reflectance clipping applied).

## Baseline reproduction against R-C0R

| check                              |   candidate_count |   subject_rows |   maximum_abs_delta_f_mel |   maximum_abs_delta_f_blood |   maximum_abs_delta_logrmse |   tolerance | pass   |
|:-----------------------------------|------------------:|---------------:|--------------------------:|----------------------------:|----------------------------:|------------:|:-------|
| baseline_oof_reproduction_vs_r_c0r |                 2 |             88 |                5.4143e-09 |                 3.01355e-08 |                   2.498e-14 |       1e-06 | True   |

## Conditional parameter profiles (R-A rule, 51-point grid)

| candidate   | parameter   |   subject_count |   boundary_fraction |   identifiable_fraction |   median_envelope_normalized_span |   max_envelope_normalized_span |   median_fit_value | classification   | all_grid_points_converged   |
|:------------|:------------|----------------:|--------------------:|------------------------:|----------------------------------:|-------------------------------:|-------------------:|:-----------------|:----------------------------|
| V2R-PS      | f_mel       |              44 |            0.477273 |                1        |                        0.00443527 |                           0.02 |          0.0945158 | unreliable       | True                        |
| V2R-PS      | f_blood     |              44 |            0.477273 |                0.318182 |                        0.34       |                           0.62 |          0.0971832 | unreliable       | True                        |
| V2R-PSG     | f_mel       |              44 |            0.545455 |                1        |                        0.00511594 |                           0.02 |          0.10563   | unreliable       | True                        |
| V2R-PSG     | f_blood     |              44 |            0.545455 |                0.318182 |                        0.33       |                           0.64 |          0.1       | unreliable       | True                        |

## Fixed-quantity sensitivity (two-parameter base candidates)

| candidate   | setting                         |   value_count |   median_logrmse |   median_abs_delta_f_mel |   median_abs_delta_f_blood |   max_abs_delta_f_mel |   max_abs_delta_f_blood |   median_logrmse_change |   boundary_fraction | all_converged   |   prediction_above_one_count |
|:------------|:--------------------------------|--------------:|-----------------:|-------------------------:|---------------------------:|----------------------:|------------------------:|------------------------:|--------------------:|:----------------|-----------------------------:|
| V2R-PS      | diameter_um                     |             4 |        0.0616619 |              0.000308291 |                6.32067e-11 |           0.00167059  |             0.00939743  |             7.28584e-17 |            0.477273 | True            |                            0 |
| V2R-PS      | s0                              |             3 |        0.0629886 |              0.000530421 |                1.82458e-10 |           0.00332539  |             0.0214203   |             5.44703e-16 |            0.462121 | True            |                            0 |
| V2R-PS      | epidermis_thickness_mm          |             3 |        0.063904  |              0.00432183  |                5.5976e-10  |           0.0120808   |             0.0348187   |             4.3715e-16  |            0.477273 | True            |                            0 |
| V2R-PS      | whole_blood_hb_g_l              |             3 |        0.0654462 |              0.000129455 |                3.96135e-10 |           0.00528956  |             0.0315626   |             0.00380183  |            0.507576 | True            |                            0 |
| V2R-PS      | scattering_amplitude_multiplier |             3 |        0.0616767 |              0.00339289  |                1.80411e-16 |           0.0099314   |             0.00798008  |             5.20417e-17 |            0.484848 | True            |                            0 |
| V2R-PS      | delta_bs_shift                  |             2 |        0.0614662 |              0.000440097 |                1.38778e-17 |           0.0029469   |             0.0125462   |             6.59195e-17 |            0.522727 | True            |                            0 |
| V2R-PS      | bandwidth_420_680               |             1 |        0.0614783 |              4.47096e-11 |                0           |           1.05459e-08 |             4.04533e-08 |             2.77556e-17 |            0.477273 | True            |                            0 |
| V2R-PS      | bandwidth_430_670               |             1 |        0.0540245 |              0.00155988  |                0.00394571  |           0.00523192  |             0.0272279   |            -0.00902664  |            0.386364 | True            |                            0 |
| V2R-PS      | bandwidth_440_660               |             1 |        0.0489167 |              0.00277899  |                0.00395008  |           0.00894368  |             0.0410516   |            -0.012187    |            0.363636 | True            |                            0 |
| V2R-PSG     | diameter_um                     |             4 |        0.0588191 |              0.000283367 |                4.16334e-17 |           0.00149582  |             0.00985463  |             6.245e-17   |            0.528409 | True            |                            0 |
| V2R-PSG     | s0                              |             3 |        0.0610506 |              0.000623366 |                4.16334e-17 |           0.00365749  |             0.019453    |             8.67362e-17 |            0.507576 | True            |                            0 |
| V2R-PSG     | epidermis_thickness_mm          |             3 |        0.0624893 |              0.00495925  |                4.16334e-17 |           0.0151818   |             0.0393969   |             1.00614e-16 |            0.537879 | True            |                            0 |
| V2R-PSG     | whole_blood_hb_g_l              |             3 |        0.0626377 |              0.00010585  |                4.16334e-17 |           0.00527206  |             0.0301021   |             0.00354362  |            0.55303  | True            |                            0 |
| V2R-PSG     | scattering_amplitude_multiplier |             3 |        0.0588701 |              0.0037156   |                1.38778e-17 |           0.0116845   |             0.0101832   |             4.85723e-17 |            0.537879 | True            |                            0 |
| V2R-PSG     | delta_bs_shift                  |             2 |        0.0587416 |              0.000491499 |                0           |           0.00313237  |             0.0132435   |             7.63278e-17 |            0.579545 | True            |                            0 |
| V2R-PSG     | bandwidth_420_680               |             1 |        0.0588191 |              3.24237e-11 |                0           |           3.26866e-09 |             1.11573e-08 |             2.42861e-17 |            0.545455 | True            |                            0 |
| V2R-PSG     | bandwidth_430_670               |             1 |        0.0516704 |              0.00120446  |                0.0026083   |           0.00518341  |             0.0281904   |            -0.00995093  |            0.431818 | True            |                            0 |
| V2R-PSG     | bandwidth_440_660               |             1 |        0.0481594 |              0.00238784  |                0.00220908  |           0.00877705  |             0.0423736   |            -0.012081    |            0.409091 | True            |                            0 |

## V2R-BEST-O oxygen extension

| base_candidate   |   subject_count |   median_logrmse |   base_median_logrmse |   median_logrmse_change |   p90_logrmse |   median_rmse |   median_sam_deg |   median_s |   median_f_mel |   median_f_blood | all_converged   |   prediction_above_one_count |
|:-----------------|----------------:|-----------------:|----------------------:|------------------------:|--------------:|--------------:|-----------------:|-----------:|---------------:|-----------------:|:----------------|-----------------------------:|
| V2R-PS           |              44 |        0.0561428 |             0.0614783 |             -0.00533546 |     0.100255  |     0.0158655 |          3.14894 |   0.677239 |      0.0937029 |        0.0970523 | True            |                            0 |
| V2R-PSG          |              44 |        0.0539425 |             0.0588191 |             -0.00487661 |     0.0971886 |     0.0156028 |          3.08619 |   0.614204 |      0.10351   |        0.1       | True            |                            0 |

| base_candidate   | parameter   |   subject_count |   boundary_fraction |   identifiable_fraction |   median_envelope_normalized_span |   max_envelope_normalized_span |   median_fit_value | classification   | all_grid_points_converged   |
|:-----------------|:------------|----------------:|--------------------:|------------------------:|----------------------------------:|-------------------------------:|-------------------:|:-----------------|:----------------------------|
| V2R-PS           | f_mel       |              44 |            0.659091 |                1        |                        0.00551121 |                           0.02 |          0.0937029 | unreliable       | True                        |
| V2R-PS           | f_blood     |              44 |            0.659091 |                0.318182 |                        0.36       |                           0.62 |          0.0970523 | unreliable       | True                        |
| V2R-PS           | s           |              44 |            0.659091 |                0.25     |                        0.4        |                           0.78 |          0.677239  | unreliable       | True                        |
| V2R-PSG          | f_mel       |              44 |            0.590909 |                1        |                        0.00617249 |                           0.02 |          0.10351   | unreliable       | True                        |
| V2R-PSG          | f_blood     |              44 |            0.590909 |                0.295455 |                        0.34       |                           0.62 |          0.1       | unreliable       | True                        |
| V2R-PSG          | s           |              44 |            0.590909 |                0.159091 |                        0.44       |                           0.84 |          0.614204  | unreliable       | True                        |

## V2R-BEST-O fixed-quantity sensitivity (s0 not applicable: s is open)

| base_candidate   | setting                         |   value_count |   median_logrmse |   median_abs_delta_s |   max_abs_delta_s |   median_abs_delta_f_mel |   median_abs_delta_f_blood |   boundary_fraction | all_converged   |
|:-----------------|:--------------------------------|--------------:|-----------------:|---------------------:|------------------:|-------------------------:|---------------------------:|--------------------:|:----------------|
| V2R-PS           | diameter_um                     |             4 |        0.0566447 |          0.0204494   |       0.106485    |              0.000262175 |                1.38778e-17 |            0.636364 | True            |
| V2R-PS           | epidermis_thickness_mm          |             3 |        0.0580287 |          0.0482504   |       0.445443    |              0.00394952  |                6.93889e-18 |            0.666667 | True            |
| V2R-PS           | whole_blood_hb_g_l              |             3 |        0.0620245 |          0.0141819   |       0.233858    |              8.27855e-05 |                6.93889e-18 |            0.689394 | True            |
| V2R-PS           | scattering_amplitude_multiplier |             3 |        0.0563263 |          0.0288001   |       0.169352    |              0.00360067  |                0           |            0.621212 | True            |
| V2R-PS           | delta_bs_shift                  |             2 |        0.056542  |          1.45034e-08 |       0.0873061   |              0.00039832  |                0           |            0.647727 | True            |
| V2R-PS           | bandwidth_420_680               |             1 |        0.0561428 |          3.7664e-10  |       1.94471e-07 |              6.57887e-11 |                0           |            0.659091 | True            |
| V2R-PS           | bandwidth_430_670               |             1 |        0.0469957 |          0.0378403   |       0.195717    |              0.00187903  |                0.0025573   |            0.613636 | True            |
| V2R-PS           | bandwidth_440_660               |             1 |        0.0443928 |          0.0234962   |       0.269894    |              0.00317613  |                0.00366898  |            0.590909 | True            |
| V2R-PSG          | diameter_um                     |             4 |        0.0542985 |          0.0226541   |       0.111763    |              0.000285627 |                0           |            0.579545 | True            |
| V2R-PSG          | epidermis_thickness_mm          |             3 |        0.056424  |          0.0793841   |       0.524875    |              0.00431441  |                0           |            0.666667 | True            |
| V2R-PSG          | whole_blood_hb_g_l              |             3 |        0.0602978 |          0.0174658   |       0.221978    |              8.37575e-05 |                0           |            0.636364 | True            |
| V2R-PSG          | scattering_amplitude_multiplier |             3 |        0.0543357 |          0.0479229   |       0.194502    |              0.00412645  |                0           |            0.606061 | True            |
| V2R-PSG          | delta_bs_shift                  |             2 |        0.054184  |          3.14778e-08 |       0.0958744   |              0.000478588 |                0           |            0.590909 | True            |
| V2R-PSG          | bandwidth_420_680               |             1 |        0.0539425 |          6.69193e-09 |       1.64492e-07 |              1.25683e-10 |                0           |            0.590909 | True            |
| V2R-PSG          | bandwidth_430_670               |             1 |        0.0458319 |          0.047951    |       0.214473    |              0.0016075   |                0.000360159 |            0.613636 | True            |
| V2R-PSG          | bandwidth_440_660               |             1 |        0.0436834 |          0.0288755   |       0.297941    |              0.00291313  |                0.00206958  |            0.522727 | True            |

## Left/right pressure linkage against the R-B side-pair audit

| candidate           | parameter   |   subject_count |   pearson_r_vs_side_log_difference |   pearson_p |   spearman_r_vs_side_log_difference |   spearman_p |
|:--------------------|:------------|----------------:|-----------------------------------:|------------:|------------------------------------:|-------------:|
| V2R-PS              | f_mel       |              44 |                           0.253618 |  0.0966803  |                            0.251304 |    0.0998672 |
| V2R-PS              | f_blood     |              44 |                           0.477609 |  0.00104363 |                            0.32224  |    0.0329058 |
| V2R-PSG             | f_mel       |              44 |                           0.202749 |  0.18686    |                            0.172375 |    0.263187  |
| V2R-PSG             | f_blood     |              44 |                           0.477372 |  0.00105053 |                            0.307362 |    0.0424038 |
| V2R-BEST-O(V2R-PS)  | f_mel       |              44 |                           0.256973 |  0.0921998  |                            0.260465 |    0.0877095 |
| V2R-BEST-O(V2R-PS)  | f_blood     |              44 |                           0.454759 |  0.00192687 |                            0.360355 |    0.0162675 |
| V2R-BEST-O(V2R-PS)  | s           |              44 |                          -0.121899 |  0.430546   |                           -0.175608 |    0.254201  |
| V2R-BEST-O(V2R-PSG) | f_mel       |              44 |                           0.204597 |  0.18279    |                            0.160395 |    0.298311  |
| V2R-BEST-O(V2R-PSG) | f_blood     |              44 |                           0.458464 |  0.00174942 |                            0.372881 |    0.0126725 |
| V2R-BEST-O(V2R-PSG) | s           |              44 |                          -0.114409 |  0.45961    |                           -0.133916 |    0.38614   |
