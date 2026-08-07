# SO-0 Forward Model Report

Final status: PASS

This is an independent implementation, not a strict reproduction of Jung et al. 2023. The original paper's full reflectance equations, color matrices, and parameter details are not public.

Ordinary phone JPEGs include unknown light, unknown camera response, automatic white balance, HDR, tone mapping, sharpening, denoising, and compression. This SO-0 stage only tests numerical stability and plausible control directions of a forward simulator; it cannot prove real JPEGs can be uniquely decomposed.

M/H are only melanin-sensitive and hemoglobin-sensitive controls. They must not be interpreted as real concentration, oxygenation, perfusion, blood flow, or SpO2. The Jiang camera database mainly provides spectral response shapes with independently normalised RGB channels; it is not absolute quantum efficiency and cannot cover all real phone ISP behaviour. The 5 nm grid is a numerical integration grid, not a 5 nm measured camera response. RGB reconstruction accuracy is not sufficient evidence that M/H decomposition is correct.

If a later SO-2 real-JPEG pilot fails, strong M/H interpretation must stop or be downgraded to a generic chromophore-sensitive representation.

## Key Metrics

- ColorChecker status: PASS, qualified 109/112
- Spectral separability: {'median': 0.7777422816276411, 'p95': 0.8472699053487714, 'status': 'PASS'}
- Observation separability: {'rank2_fraction': 1.0, 'camera_rgb_wb_rank2_fraction': 1.0, 'linear_srgb_unclipped_rank2_fraction': 1.0, 'chromaticity_rank2_fraction': 1.0, 'rank2_count': 8175.0, 'total_count': 8175.0, 'status': 'PASS'}
- Resolution audit: {'cie_deltae00_median': 0.017228327098763582, 'cie_deltae00_p95': 0.08102941947281887, 'camera_rgb_relative_error_median': 0.00028258605105565974, 'camera_rgb_relative_error_p95': 0.0007631506607548824, 'camera_rgb_relative_error_max': 0.0009744067716821286, 'camera_chroma_l1_p95': 0.00029444192070382513}
- Backend parity: {'numpy_torch_float64_reflectance_max_abs': 2.220446049250313e-16, 'torch_float32_float64_reflectance_max_abs': 1.5880876091944884e-07}
- Hard failures: []
