# RealFace HighRes 1024x1280 Pilot revA Report

## Revision scope
This is a revision of the existing fourth-part 32-case pilot, not a rebuild from scratch. The same 32 pilot IDs were reused and the initial pilot output directory was not overwritten.

## Human QC feedback from initial pilot
- Top valid skin often touched the canvas border.
- Fine forehead hair contamination was mild and rare.
- Mouth exclusion was normal.
- Nostril exclusion was too wide.

## revA changes
- Increased only `geometry.top_margin_ratio` from 0.02 to 0.08.
- Split nostril dilation from the previous shared exclusion dilation: eye/brow/lip remains 0.006, nostril uses 0.003.

## Explicitly unchanged
- Forehead fine-hair strategy unchanged; no dark-pixel or texture hair detector was added.
- Mouth/lip polygon and mouth-related dilation remain unchanged.
- 1024x1280 4:5 canvas, FaceMesh geometry crop, one-shot affine warp, and BiSeNet parsing remain unchanged.
- No forehead repair, alpha feather, skin-mask bbox crop, 3D frontalization, SO-1 inference, patching, M/H/S/P generation, or classification was run.

## Legacy code analyzed
- preprocessing/build_global_face_parsing_hybrid_foreheadrepair_blackbg_224_png_strict.py
- preprocessing/build_global_face_oval_blackbg_png_simalign_strict.py
- preprocessing/build_global_face_parsing_regularmask_blackbg_224_png_strict.py
- config/preprocess/global_face_parsing_hybrid_foreheadrepair_blackbg_224_png_strict.yaml
- preprocessing/checkpoints/face_parsing/79999_iter.pth

## Reused legacy logic
- MediaPipe Face Detection model_selection=1 and deterministic primary-face ranking by face area and image-center distance.
- Expanded detection box ratios: top 10%, bottom 20%, left/right 20%.
- FaceMesh static_image_mode=True, max_num_faces=1, refine_landmarks=True.
- CelebAMask-HQ 19-class BiSeNet architecture, 512 parser input, ImageNet normalization, argmax logits, nearest-neighbor label restore.

## New files
- preprocessing/build_skin_optics_realface_highres_1024x1280_v1.py
- src/skin_optics_real_preprocess/*
- config/preprocess/skin_optics_realface_highres_1024x1280_v1.yaml
- tests/skin_optics_real_preprocess/*

## Parameter comparison
| Parameter | Initial pilot | revA |
| --- | ---: | ---: |
| top_margin_ratio | 0.02 | 0.08 |
| side_margin_ratio | 0.03 | 0.03 |
| bottom_margin_ratio | 0.025 | 0.025 |
| shared/eye_brow_lip dilation | 0.006 | 0.006 |
| nostril_dilation_ratio | 0.006 | 0.003 |

## Geometry
Roll is computed from the two eye centers. The final 3x3 source-to-canvas matrix composes roll correction, crop translation, and a single uniform scale. Crop geometry is driven only by rolled FaceMesh oval/chin geometry.

## Crop parameters
- side_margin_ratio=0.03 controls horizontal cheek margin.
- top_margin_ratio=0.08 guarantees more forehead/oval top margin by uniform crop enlargement.
- bottom_margin_ratio=0.025 anchors the crop bottom below the chin.

## Color
Images are treated as assumed sRGB, converted through the standard inverse-sRGB transfer function to float32 linear RGB, warped once with cv2.INTER_LINEAR, encoded back to sRGB uint8 for PNG previews, and saved as float16 CHW for aligned_linear_rgb.

## Parser classes
{0: 'background', 1: 'skin', 2: 'left_brow', 3: 'right_brow', 4: 'left_eye', 5: 'right_eye', 6: 'eyeglass', 7: 'left_ear', 8: 'right_ear', 9: 'earring', 10: 'nose', 11: 'mouth', 12: 'upper_lip', 13: 'lower_lip', 14: 'neck', 15: 'necklace', 16: 'cloth', 17: 'hair', 18: 'hat'}

## FaceMesh exclusion regions
Eyes, brows and lips use stable MediaPipe contour landmarks. Nostril exclusion uses compact project-defined nose-wing/base polygons stored in facemesh_regions.py and drawn in QC panels.

## Test and validation results
- New module tests: `E:\resarch\Anaconda3\envs\face2\python.exe -m pytest tests/skin_optics_real_preprocess -q` -> 11 passed.
- Existing preprocessing tests: `E:\resarch\Anaconda3\envs\face2\python.exe -m pytest tests/preprocessing -q` -> 12 passed.
- validate-only: `E:\resarch\Anaconda3\envs\face2\python.exe preprocessing/build_skin_optics_realface_highres_1024x1280_v1.py --config config/preprocess/skin_optics_realface_highres_1024x1280_v1.yaml --validate-only` -> passed.
- revA pilot 32: `E:\resarch\Anaconda3\envs\face2\python.exe preprocessing/build_skin_optics_realface_highres_1024x1280_v1.py --config config/preprocess/skin_optics_realface_highres_1024x1280_v1.yaml --pilot-count 32 --pilot-id-file outputs/RealFace_HighRes_1024x1280_v1_pilot32/pilot_ids.txt --device cuda:0 --overwrite-confirmed` -> 32 success, 0 failed.
- Output audit: 32 aligned images/masks/metadata files checked; no bad shape, non-binary mask, out-of-source skin pixel, non-finite linear RGB, or non-uniform affine matrix found.

## Initial vs revA QC comparison
Top-border skin is defined as the count/fraction of final `skin_valid_mask` pixels equal to 255 on row 0. A case is touching the top border when that count is greater than zero.

```json
{
  "top_border": {
    "baseline_number_of_cases_skin_touching_top_border": 31,
    "revA_number_of_cases_skin_touching_top_border": 30,
    "baseline_fraction_of_cases_skin_touching_top_border": 0.96875,
    "revA_fraction_of_cases_skin_touching_top_border": 0.9375,
    "baseline_top_border_skin_fraction": {
      "min": 0.0,
      "median": 0.6513671875,
      "p95": 0.7728515625,
      "max": 0.7890625
    },
    "revA_top_border_skin_fraction": {
      "min": 0.0,
      "median": 0.4873046875,
      "p95": 0.7208984374999999,
      "max": 0.7578125
    }
  },
  "composition": {
    "baseline_aligned_face_width_fraction": {
      "min": 0.9234053492546082,
      "median": 0.9474043250083923,
      "p95": 0.9642344772815704,
      "max": 0.9735538363456726
    },
    "revA_aligned_face_width_fraction": {
      "min": 0.8668703436851501,
      "median": 0.9158762991428375,
      "p95": 0.9473266363143921,
      "max": 0.9549548029899597
    },
    "baseline_aligned_face_height_fraction": {
      "min": 0.8529388427734375,
      "median": 0.9334684371948242,
      "p95": 0.9575036811828613,
      "max": 0.9619636535644531
    },
    "revA_aligned_face_height_fraction": {
      "min": 0.8529388427734375,
      "median": 0.8971302509307861,
      "p95": 0.8992262077331543,
      "max": 0.9030679702758789
    },
    "baseline_skin_fraction": {
      "min": 0.5483024597167969,
      "median": 0.6426258087158203,
      "p95": 0.6838455200195312,
      "max": 0.7055351257324218
    },
    "revA_skin_fraction": {
      "min": 0.5060317993164063,
      "median": 0.6127906799316406,
      "p95": 0.6754454803466796,
      "max": 0.7068130493164062
    }
  },
  "nostril": {
    "revA_nostril_exclusion_area_ratio": {
      "min": 0.007572174072265625,
      "median": 0.00947265625,
      "p95": 0.010533523559570311,
      "max": 0.011228179931640625
    },
    "revA_nostril_dilation_radius_px": {
      "min": 3.0,
      "median": 4.5,
      "p95": 6.0,
      "max": 6.0
    }
  }
}
```

Top-border cases changed from 31 to 30. The top-border skin fraction decreased clearly, but the binary touching frequency did not drop enough to call the issue fully resolved.
Nostril exclusion was narrowed by changing only its dilation ratio; nostril polygons were not changed in revA.
Composition remained close to the original tight crop; review the comparison images for possible side effects.

## Representative comparison images
- E:\projects\face2\outputs\RealFace_HighRes_1024x1280_v1_pilot32_revA\qc\revision_comparisons\A000477213_base_vs_revA.png
- E:\projects\face2\outputs\RealFace_HighRes_1024x1280_v1_pilot32_revA\qc\revision_comparisons\A002012347_base_vs_revA.png
- E:\projects\face2\outputs\RealFace_HighRes_1024x1280_v1_pilot32_revA\qc\revision_comparisons\A001636890_base_vs_revA.png
- E:\projects\face2\outputs\RealFace_HighRes_1024x1280_v1_pilot32_revA\qc\revision_comparisons\100037382_base_vs_revA.png
- E:\projects\face2\outputs\RealFace_HighRes_1024x1280_v1_pilot32_revA\qc\revision_comparisons\200887343-1_base_vs_revA.png

## Pilot IDs
100037382, 200887343-1, 201885444, 203214852, 204238693, 205200934-1, 206248971, 9300198023, 9300800427, A000104693, A000477213, A001093486, A001393665, A001560256, A001604824, A001613504, A001627943, A001636890, A001647049, A001670882, A001698538, A001758297, A001813366, A001866164, A001906652, A001923658, A001970601, A002012347, A002081031, A002145202-1, A002337238, A002366013

## Summary
```json
{
  "pilot_total": 32,
  "success_count": 32,
  "failure_count": 0,
  "warning_case_count": 7,
  "roll_angle": {
    "min": 0.10978028240918396,
    "median": 2.6731121532079207,
    "p95": 7.054928976548228,
    "max": 9.381178342968756
  },
  "residual_roll": {
    "median": 0.0,
    "p95": 0.0,
    "max": 0.0
  },
  "aligned_face_width_fraction": {
    "min": 0.8668703436851501,
    "median": 0.9158762991428375,
    "p95": 0.9473266363143921,
    "max": 0.9549548029899597
  },
  "aligned_face_height_fraction": {
    "min": 0.8529388427734375,
    "median": 0.8971302509307861,
    "p95": 0.8992262077331543,
    "max": 0.9030679702758789
  },
  "skin_fraction": {
    "min": 0.5060317993164063,
    "median": 0.6127906799316406,
    "p95": 0.6754454803466796,
    "max": 0.7068130493164062
  },
  "hair_fraction_in_face_region": {
    "median": 0.00995883542228301,
    "p95": 0.032816716041318424,
    "max": 0.09062178491206442
  },
  "source_valid_fraction": {
    "min": 0.9926483154296875,
    "median": 1.0,
    "p95": 1.0
  },
  "top_border_skin_pixel_count": {
    "min": 0.0,
    "median": 499.0,
    "p95": 738.1999999999999,
    "max": 776.0
  },
  "top_border_skin_fraction": {
    "min": 0.0,
    "median": 0.4873046875,
    "p95": 0.7208984374999999,
    "max": 0.7578125
  },
  "number_of_cases_skin_touching_top_border": 30,
  "fraction_of_cases_skin_touching_top_border": 0.9375,
  "nostril_exclusion_area_ratio": {
    "min": 0.007572174072265625,
    "median": 0.00947265625,
    "p95": 0.010533523559570311,
    "max": 0.011228179931640625
  },
  "nostril_dilation_radius_px": {
    "min": 3.0,
    "median": 4.5,
    "p95": 6.0,
    "max": 6.0
  },
  "top_invalid_fraction": {
    "min": 0.0,
    "median": 0.0,
    "p95": 0.0,
    "max": 0.0
  },
  "bottom_invalid_fraction": {
    "min": 0.0,
    "median": 0.0,
    "p95": 0.0,
    "max": 0.14703369140625
  },
  "left_invalid_fraction": {
    "min": 0.0,
    "median": 0.0,
    "p95": 0.0,
    "max": 0.03239889705882353
  },
  "right_invalid_fraction": {
    "min": 0.0,
    "median": 0.0,
    "p95": 0.0,
    "max": 0.0
  },
  "failure_code_counts": {},
  "warning_code_counts": {
    "bottom_invalid_area": 1,
    "face_near_canvas_border": 6,
    "high_skin_fraction": 3
  }
}
```

## QC files
- E:\projects\face2\outputs\RealFace_HighRes_1024x1280_v1_pilot32_revA\qc\failed_cases.csv
- E:\projects\face2\outputs\RealFace_HighRes_1024x1280_v1_pilot32_revA\qc\pilot_contact_sheets\pilot32_contact_sheet.png
- E:\projects\face2\outputs\RealFace_HighRes_1024x1280_v1_pilot32_revA\qc\pilot_summary.csv
- E:\projects\face2\outputs\RealFace_HighRes_1024x1280_v1_pilot32_revA\qc\preview_panels\100037382.png
- E:\projects\face2\outputs\RealFace_HighRes_1024x1280_v1_pilot32_revA\qc\preview_panels\200887343-1.png
- E:\projects\face2\outputs\RealFace_HighRes_1024x1280_v1_pilot32_revA\qc\preview_panels\201885444.png
- E:\projects\face2\outputs\RealFace_HighRes_1024x1280_v1_pilot32_revA\qc\preview_panels\203214852.png
- E:\projects\face2\outputs\RealFace_HighRes_1024x1280_v1_pilot32_revA\qc\preview_panels\204238693.png
- E:\projects\face2\outputs\RealFace_HighRes_1024x1280_v1_pilot32_revA\qc\preview_panels\205200934-1.png
- E:\projects\face2\outputs\RealFace_HighRes_1024x1280_v1_pilot32_revA\qc\preview_panels\206248971.png
- E:\projects\face2\outputs\RealFace_HighRes_1024x1280_v1_pilot32_revA\qc\preview_panels\9300198023.png
- E:\projects\face2\outputs\RealFace_HighRes_1024x1280_v1_pilot32_revA\qc\preview_panels\9300800427.png
- E:\projects\face2\outputs\RealFace_HighRes_1024x1280_v1_pilot32_revA\qc\preview_panels\A000104693.png
- E:\projects\face2\outputs\RealFace_HighRes_1024x1280_v1_pilot32_revA\qc\preview_panels\A000477213.png
- E:\projects\face2\outputs\RealFace_HighRes_1024x1280_v1_pilot32_revA\qc\preview_panels\A001093486.png
- E:\projects\face2\outputs\RealFace_HighRes_1024x1280_v1_pilot32_revA\qc\preview_panels\A001393665.png
- E:\projects\face2\outputs\RealFace_HighRes_1024x1280_v1_pilot32_revA\qc\preview_panels\A001560256.png
- E:\projects\face2\outputs\RealFace_HighRes_1024x1280_v1_pilot32_revA\qc\preview_panels\A001604824.png
- E:\projects\face2\outputs\RealFace_HighRes_1024x1280_v1_pilot32_revA\qc\preview_panels\A001613504.png
- E:\projects\face2\outputs\RealFace_HighRes_1024x1280_v1_pilot32_revA\qc\preview_panels\A001627943.png
- E:\projects\face2\outputs\RealFace_HighRes_1024x1280_v1_pilot32_revA\qc\preview_panels\A001636890.png
- E:\projects\face2\outputs\RealFace_HighRes_1024x1280_v1_pilot32_revA\qc\preview_panels\A001647049.png
- E:\projects\face2\outputs\RealFace_HighRes_1024x1280_v1_pilot32_revA\qc\preview_panels\A001670882.png
- E:\projects\face2\outputs\RealFace_HighRes_1024x1280_v1_pilot32_revA\qc\preview_panels\A001698538.png
- E:\projects\face2\outputs\RealFace_HighRes_1024x1280_v1_pilot32_revA\qc\preview_panels\A001758297.png
- E:\projects\face2\outputs\RealFace_HighRes_1024x1280_v1_pilot32_revA\qc\preview_panels\A001813366.png
- E:\projects\face2\outputs\RealFace_HighRes_1024x1280_v1_pilot32_revA\qc\preview_panels\A001866164.png
- E:\projects\face2\outputs\RealFace_HighRes_1024x1280_v1_pilot32_revA\qc\preview_panels\A001906652.png
- E:\projects\face2\outputs\RealFace_HighRes_1024x1280_v1_pilot32_revA\qc\preview_panels\A001923658.png
- E:\projects\face2\outputs\RealFace_HighRes_1024x1280_v1_pilot32_revA\qc\preview_panels\A001970601.png
- E:\projects\face2\outputs\RealFace_HighRes_1024x1280_v1_pilot32_revA\qc\preview_panels\A002012347.png
- E:\projects\face2\outputs\RealFace_HighRes_1024x1280_v1_pilot32_revA\qc\preview_panels\A002081031.png
- E:\projects\face2\outputs\RealFace_HighRes_1024x1280_v1_pilot32_revA\qc\preview_panels\A002145202-1.png
- E:\projects\face2\outputs\RealFace_HighRes_1024x1280_v1_pilot32_revA\qc\preview_panels\A002337238.png
- E:\projects\face2\outputs\RealFace_HighRes_1024x1280_v1_pilot32_revA\qc\preview_panels\A002366013.png
- E:\projects\face2\outputs\RealFace_HighRes_1024x1280_v1_pilot32_revA\qc\qc_summary.json
- E:\projects\face2\outputs\RealFace_HighRes_1024x1280_v1_pilot32_revA\qc\revision_comparison.csv
- E:\projects\face2\outputs\RealFace_HighRes_1024x1280_v1_pilot32_revA\qc\revision_comparison_summary.json
- E:\projects\face2\outputs\RealFace_HighRes_1024x1280_v1_pilot32_revA\qc\revision_comparisons\100037382_base_vs_revA.png
- E:\projects\face2\outputs\RealFace_HighRes_1024x1280_v1_pilot32_revA\qc\revision_comparisons\200887343-1_base_vs_revA.png
- E:\projects\face2\outputs\RealFace_HighRes_1024x1280_v1_pilot32_revA\qc\revision_comparisons\A000477213_base_vs_revA.png
- E:\projects\face2\outputs\RealFace_HighRes_1024x1280_v1_pilot32_revA\qc\revision_comparisons\A001636890_base_vs_revA.png
- E:\projects\face2\outputs\RealFace_HighRes_1024x1280_v1_pilot32_revA\qc\revision_comparisons\A002012347_base_vs_revA.png
- E:\projects\face2\outputs\RealFace_HighRes_1024x1280_v1_pilot32_revA\qc\warning_cases.csv

## Current issues
- Pilot selection used deterministic uniform ID spacing because no reliable old QC field covering bangs/glasses/yaw was found in the fixed 500 CSV.
- Nostril polygons are conservative project-local definitions rather than an official complete MediaPipe nostril contour.
- Representative comparison categories are selected by reproducible QC statistics rather than manual visual labels; use the comparison images for the final visual judgment.
- Top-border skin occupancy intensity improved, but 30/32 cases still have at least one top-border skin pixel under the strict row-0 metric.

## Full-500 recommendation
recommend_full500 = false. Do not enter full 500 automatically; the top-border frequency metric remains too high for a clean full-run recommendation.
