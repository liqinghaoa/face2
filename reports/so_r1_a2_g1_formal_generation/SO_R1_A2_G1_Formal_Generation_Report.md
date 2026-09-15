# SO-R1-A2-G1 Formal Generation Report

## Result

G1 stopped with `FAIL_QC_RETRY_EXHAUSTED`; the partial dataset is not accepted and must not be used for training.

## Calibration

The required calibration passed before formal generation: workers=1 and workers=8 matched canonical decompressed content hashes for 128/128 fixed latents with 24/24 allowlist coverage.

## Terminal failure

`F09669_A1` (Train, Strong mask, M base 0.96695, H base 0.01575) uses Olympus E-PL2 under FL2. All 128 frozen nuisance-only retries failed the low clipping gate. The best attempt had low-clip element fraction 0.11096196868008948, exceeding the frozen maximum 0.10.

## Attribution

At the best fixed nuisance sample, Olympus E-PL2 failed while Canon 5DMarkII, Nikon D80, Canon 300D, Canon 1DMarkIII, and Nikon D5100 passed. The exposure sweep from -0.50 through +0.31 EV did not reduce the Olympus low clipping below the gate. This is camera-specific evidence, not an exposure/specular nuisance-sampling issue.

## Disposition

No M/H, mask, camera/light, split, QC threshold, or frozen protocol was changed. The partial output remains an unaccepted interrupted generation artifact. A subsequent protocol amendment must address the camera set through its own audit; it may not use nuisance resampling to conceal this failure.
