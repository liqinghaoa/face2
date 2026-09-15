# SO-R1-A0 Camera/Light Selection Report

- Final status: **PASS**
- Freeze status: **FROZEN**
- SO-0 protected assets unchanged: **True**

## Task and inputs

This audit freezes only the SO-R1 camera/light subset and allowlist; it does not alter any SO-0 formula, spectrum, threshold, or rendering implementation.
- Native 10 nm source: `data/external/SO0_Spectral_Assets_v1/processed/SO0_Spectral_Standardized_v1/production_10nm/source_standardized_10nm.npz`
- SO-0 audit root: `outputs/SO0_Forward_Model_v1.1`
- 1 nm assets were checked for presence only and were not used for primary selection.
- Input SHA-256:
- `data/external/SO0_Spectral_Assets_v1/processed/SO0_Spectral_Standardized_v1/production_10nm/source_standardized_10nm.npz`: `154fde064b9c2c016aaecb6183abeb196c33f44fd8ceab574118f3057a8dcb08`
- `outputs/SO0_Forward_Model_v1.1/tables/multicamera_metrics.csv`: `e89f6da91b7fd01ae67e2c03e28cc3c4f054a64a0507eeccaa5ccca69f1dd512`
- `outputs/SO0_Forward_Model_v1.1/tables/colorchecker_calibration_metrics.csv`: `912a93b86eaa9e1875ed802533134336fd571796662a6a7a74276259005e185f`
- `outputs/SO0_Forward_Model_v1.1/tables/mh_observation_separability.csv`: `faf2887e49ea154b13bcf440d276d86ff6bd7e2e83c7a789009de2b19bb1b2cf`
- `outputs/SO0_Forward_Model_v1.1/tables/resolution_1nm_vs_5nm_by_camera_light.csv`: `c348b22ab345e4bef9eeb9450343e683a973a76cdd8b3855a091a3e3610be0f7`
- `outputs/SO0_Forward_Model_v1.1/tables/multilight_metrics.csv`: `a593537a6f4b371191d222129a8da5c5d56978f162919b0559c4f2f29672408f`
- Complete protected before/after manifests: `protected_asset_hash_before.csv`, `protected_asset_hash_after.csv`.

## Camera identity and quality audit

All 28/28 NPZ camera identities joined explicitly by normalized name to multicamera, ColorChecker, M/H-observation, and resolution audit tables. No normalized-name collision or ambiguous alias was found.
- Canon 1DMarkIII — quality gate: `True`; scope: `consumer_mobile_candidate`
- Canon 20D — quality gate: `True`; scope: `consumer_mobile_candidate`
- Canon 300D — quality gate: `True`; scope: `consumer_mobile_candidate`
- Canon 40D — quality gate: `True`; scope: `consumer_mobile_candidate`
- Canon 500D — quality gate: `True`; scope: `consumer_mobile_candidate`
- Canon 50D — quality gate: `True`; scope: `consumer_mobile_candidate`
- Canon 5DMarkII — quality gate: `True`; scope: `consumer_mobile_candidate`
- Canon 600D — quality gate: `True`; scope: `consumer_mobile_candidate`
- Canon 60D — quality gate: `True`; scope: `consumer_mobile_candidate`
- Hasselblad H2 — quality gate: `True`; scope: `excluded` (specialized_medium_format_profile)
- Nikon D200 — quality gate: `True`; scope: `consumer_mobile_candidate`
- Nikon D3 — quality gate: `True`; scope: `consumer_mobile_candidate`
- Nikon D300s — quality gate: `True`; scope: `consumer_mobile_candidate`
- Nikon D3X — quality gate: `True`; scope: `consumer_mobile_candidate`
- Nikon D40 — quality gate: `True`; scope: `consumer_mobile_candidate`
- Nikon D50 — quality gate: `True`; scope: `consumer_mobile_candidate`
- Nikon D5100 — quality gate: `True`; scope: `consumer_mobile_candidate`
- Nikon D700 — quality gate: `True`; scope: `consumer_mobile_candidate`
- Nikon D80 — quality gate: `True`; scope: `consumer_mobile_candidate`
- Nikon D90 — quality gate: `True`; scope: `consumer_mobile_candidate`
- Nokia N900 — quality gate: `True`; scope: `consumer_mobile_candidate`
- Olympus E-PL2 — quality gate: `True`; scope: `consumer_mobile_candidate`
- Pentax K-5 — quality gate: `True`; scope: `consumer_mobile_candidate`
- Pentax Q — quality gate: `True`; scope: `consumer_mobile_candidate`
- Phase One — quality gate: `True`; scope: `excluded` (specialized_medium_format_profile)
- Point Grey Grasshopper 50S5C — quality gate: `False`; scope: `excluded` (failed_3_of_4_camera_light_pairs)
- Point Grey Grasshopper2 14S5C — quality gate: `True`; scope: `excluded` (industrial_camera_profile)
- SONY NEX-5N — quality gate: `True`; scope: `consumer_mobile_candidate`

The quality gate required 4 illuminants, 4 qualified pairs, finite fraction 1.0, multicamera PASS, per-pair ColorChecker PASS/finite/no failure reason, M/H rank 2/PASS/finite, and finite resolution metrics. Point Grey Grasshopper 50S5C was excluded because D65/A/FL2 failed ColorChecker, leaving one qualified pair only. The 24-camera candidate pool excludes it plus Point Grey Grasshopper2 14S5C, Hasselblad H2, and Phase One for the protocol-defined scope reasons.

## Spectral selection

Responses were checked finite, nonnegative, R/G/B, common strictly increasing 10 nm grid, and positive per-channel area. They were normalized per camera and channel by response area only for audit calculations. Primary distance was channel-mean spectral angle (radian); channel-mean total variation was used only for sensitivity.
- Seen cameras: Canon 5DMarkII, Nokia N900, Nikon D80, Pentax Q, SONY NEX-5N, Olympus E-PL2
- Unseen cameras: Canon 1DMarkIII, Nikon D5100
- Forced seen anchors: Canon 5DMarkII; Nokia N900.
- Seen lights: D65, A, FL2. Unseen light: FL11.

### Seen greedy maximin trace

- Round 1: Nikon D80; min-SAM=0.247166474418; nearest=Canon 5DMarkII.
- Round 2: Pentax Q; min-SAM=0.228448426138; nearest=Nokia N900.
- Round 3: SONY NEX-5N; min-SAM=0.206586790050; nearest=Nikon D80.
- Round 4: Olympus E-PL2; min-SAM=0.191456260248; nearest=Nokia N900.

### Unseen farthest-from-final-seen trace

- Round 1: Canon 1DMarkIII; min distance=0.157854482752; nearest=Canon 5DMarkII.
- Round 2: Nikon D5100; min distance=0.150293105827; nearest=SONY NEX-5N.

### TV sensitivity

- SAM cameras: Canon 5DMarkII; Nokia N900; Nikon D80; Pentax Q; SONY NEX-5N; Olympus E-PL2; Canon 1DMarkIII; Nikon D5100.
- TV cameras: Canon 5DMarkII; Nokia N900; Nikon D80; Pentax Q; Olympus E-PL2; SONY NEX-5N; Nikon D50; Canon 300D.
- Final-eight overlap: 6/8; seen overlap: 6/6; unseen overlap: 0/2; Jaccard: 0.600; anchors preserved: True.

## 32-pair allowlist

All 32/32 pairs are ColorChecker-qualified, M/H-audited, resolution-audited and finite: ID 18, CAMERA_OOD 6, LIGHT_OOD 6, JOINT_OOD 2.
- Canon 1DMarkIII × A: `CAMERA_OOD`
- Canon 1DMarkIII × D65: `CAMERA_OOD`
- Canon 1DMarkIII × FL2: `CAMERA_OOD`
- Nikon D5100 × A: `CAMERA_OOD`
- Nikon D5100 × D65: `CAMERA_OOD`
- Nikon D5100 × FL2: `CAMERA_OOD`
- Canon 5DMarkII × A: `ID`
- Canon 5DMarkII × D65: `ID`
- Canon 5DMarkII × FL2: `ID`
- Nikon D80 × A: `ID`
- Nikon D80 × D65: `ID`
- Nikon D80 × FL2: `ID`
- Nokia N900 × A: `ID`
- Nokia N900 × D65: `ID`
- Nokia N900 × FL2: `ID`
- Olympus E-PL2 × A: `ID`
- Olympus E-PL2 × D65: `ID`
- Olympus E-PL2 × FL2: `ID`
- Pentax Q × A: `ID`
- Pentax Q × D65: `ID`
- Pentax Q × FL2: `ID`
- SONY NEX-5N × A: `ID`
- SONY NEX-5N × D65: `ID`
- SONY NEX-5N × FL2: `ID`
- Canon 1DMarkIII × FL11: `JOINT_OOD`
- Nikon D5100 × FL11: `JOINT_OOD`
- Canon 5DMarkII × FL11: `LIGHT_OOD`
- Nikon D80 × FL11: `LIGHT_OOD`
- Nokia N900 × FL11: `LIGHT_OOD`
- Olympus E-PL2 × FL11: `LIGHT_OOD`
- Pentax Q × FL11: `LIGHT_OOD`
- SONY NEX-5N × FL11: `LIGHT_OOD`

## Verification

- Deterministic rerun / frozen verification: PASS.
- Tests: `pytest -q tests/so_r1` — 4 passed.
- Protected SO-0 asset hash comparison: PASS.

## Evidence boundary

Camera profiles are from the public Jiang spectral-response library and primarily describe independently normalized RGB spectral-response shapes, not absolute quantum efficiency. They omit modern phone ISP, AWB, HDR and tone mapping. Nokia N900 is only a mobile-camera anchor in this public library, not a representative of modern phones or manufacturers. The 6/2 split is for controlled synthetic camera-response experiments; unseen identities are not real-device clinical validation.

## Result

This result is **PASS** / **FROZEN**. Entry to paired synthetic-data pilot: **YES**.
