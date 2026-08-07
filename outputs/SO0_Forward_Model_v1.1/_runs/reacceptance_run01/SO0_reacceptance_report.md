# SO-0 Reacceptance Report

Final status: PASS

This is an independent implementation, not a strict reproduction of Jung et al. 2023. M/H are only melanin-sensitive and hemoglobin-sensitive control quantities. Jiang camera responses are spectral response shapes, not absolute quantum efficiency. The 5 nm grid is a numerical integration grid, not a 5 nm measured camera response. Phone JPEGs include unknown ISP, HDR, AWB, tone mapping, sharpening, denoising, and compression. Passing this forward model does not prove real JPEGs can be uniquely inverted, and RGB reconstruction accuracy does not prove M/H decomposition correctness. If a later SO-2 real-JPEG pilot fails, interpretation must be downgraded.

## Audit Repairs

- Pytest return code and JUnit XML are hard gates.
- Stage status is computed from current-run artifacts and metrics.
- Shading, specular, exposure, autograd, M/H, multi-camera, multi-light, sensitivity, and resolution audits are regenerated from real current-run data.
- v1 outputs are preserved; v1.1 writes to run-specific empty directories.
- Configuration, thresholds, audit grids, paths, and decision rules are loaded from the v1.1 YAML bundle.
- PyTorch color constants are now created as float64 before dtype conversion, removing a backend parity precision loss without changing the color formula.

## Core Regression

NumPy physical reflectance, CIE, camera RGB, and linear sRGB golden outputs changed by 0. The PyTorch float64 camera linear-sRGB golden output changed by 6.89408e-08, caused by the float64 constant precision repair. No optical formula, ColorChecker method, threshold, or production grid was changed.

## Pytest

29 tests, 0 failures, 0 errors, 0 skipped. Return code 0.

## Resolution Audit

The full camera audit covered 121968 camera conditions. CIE conditions: 4356. CIEDE2000 median=0.015019, p95=0.0863204. Camera RGB relative error median=0.00049792, p95=0.00190075, max=0.00737441. Chromaticity L1 p95=0.000985119. Status: PASS.

## ColorChecker

Recomputed 112 camera-light combinations. Qualified 109/112. Qualified by light: {'A': 27, 'D65': 27, 'FL11': 28, 'FL2': 27}. Canon 5DMarkII + D65 qualified: True.

Failed combinations:

- Point Grey Grasshopper 50S5C / D65: loocv_median_gt_4;loocv_p95_gt_8
- Point Grey Grasshopper 50S5C / A: loocv_median_gt_4;loocv_p95_gt_8
- Point Grey Grasshopper 50S5C / FL2: loocv_median_gt_4;loocv_p95_gt_8

## M/H Audits

Spectral Jacobian cosine median=0.777742, p95=0.84727. Observation-domain rank-2 fraction=1 over 8175 qualified rows. sigma_ratio median=0.0367902, p05=0.0146717, p01=0.0115087, min=0.00814219. Condition number median=27.1812, p95=68.1583, max=122.817. Worst cases are saved in `tables/mh_observation_worst_cases.csv`.

## Autograd

Autograd audit rows=500; failures=0; status=PASS. It covers m, h, shading, specular, and exposure across D65, A, FL2, FL11 and representative cameras.

## Linearity

- shading camera_rgb_raw: max_relative_error=9.03454e-16, PASS
- shading camera_rgb_wb: max_relative_error=8.1699e-16, PASS
- shading cie_xyz: max_relative_error=6.26006e-16, PASS
- shading spectral_radiance: max_relative_error=2.64152e-16, PASS
- specular camera_rgb_raw: max_relative_error=3.15258e-14, PASS
- specular camera_rgb_wb: max_relative_error=3.66027e-14, PASS
- specular cie_xyz: max_relative_error=1.66533e-14, PASS
- specular spectral_radiance: max_relative_error=7.92179e-15, PASS
- exposure camera_rgb_raw: max_relative_error=9.17492e-16, PASS
- exposure camera_rgb_wb: max_relative_error=8.37494e-16, PASS
- exposure cie_xyz: max_relative_error=6.76274e-16, PASS
- exposure spectral_radiance: max_relative_error=2.69783e-16, PASS

## Backend And Numeric Parity

- numpy64_vs_torch64 camera_rgb_raw: max_abs_error=6.66134e-16, PASS
- numpy64_vs_torch64 camera_rgb_wb: max_abs_error=1.11022e-15, PASS
- numpy64_vs_torch64 linear_srgb_unclipped: max_abs_error=3.55271e-15, PASS
- numpy64_vs_torch64 reflectance: max_abs_error=1.11022e-16, PASS
- numpy64_vs_torch64 xyz_d65: max_abs_error=1.44329e-15, PASS
- torch32_vs_torch64 camera_rgb_raw: max_abs_error=3.72454e-07, PASS
- torch32_vs_torch64 camera_rgb_wb: max_abs_error=5.32613e-07, PASS
- torch32_vs_torch64 linear_srgb_unclipped: max_abs_error=1.19912e-06, PASS
- torch32_vs_torch64 reflectance: max_abs_error=1.42971e-07, PASS
- torch32_vs_torch64 xyz_d65: max_abs_error=1.02691e-06, PASS

## Sensitivity

Sensitivity statuses: {'oxygenation': 'PASS', 'epidermis_thickness': 'PASS', 'dermis_thickness': 'PASS', 'melanin_formula': 'PASS'}. Primary model remains oxygenation=0.75, epidermis thickness=0.006 cm, dermis thickness=0.20 cm, melanin formula=primary.

## Reproducibility And Assets

Two independent empty-run reacceptance passes: {}. Frozen asset hash before=PASS, after=PASS.

## Freeze Candidate

This output is a FREEZE_CANDIDATE only. It requires human review, and `so1_authorized` remains false. SO-1 is not authorized.
