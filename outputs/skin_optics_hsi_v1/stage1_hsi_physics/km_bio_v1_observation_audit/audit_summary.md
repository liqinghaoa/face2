# KM-BIO-v1 observation-contract audit

- Status: `PASS_FOR_TRAIN_INVERSION`
- Train subjects/captures/cheek spectra: 44/44/88
- Raw float64 versus historical cache maximum absolute difference: 4.893e-08
- Raw versus clipped region-median maximum absolute difference: 0.000e+00
- Observed reflectance range: [0.115666, 0.737783]
- Validation/Test HSI content reads: 0/0
- Next stage allowed: `true`

The authoritative Stage C inputs are the raw HDF5 float64 per-band region medians in the observation manifest. The released paired RGB files were hash-audited for provenance only and were not decoded or used to alter HSI spectra.
