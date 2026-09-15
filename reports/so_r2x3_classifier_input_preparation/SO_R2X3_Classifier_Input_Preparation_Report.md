# SO-R2-X3-P0 Classifier Input Preparation

{
  "status": "COMPLETE_CLASSIFIER_INPUT_PREPARATION",
  "source_dataset": "data\\processed\\skin_optics\\skinoptics_realface_979x1220_blackbg_v1",
  "source_R2X1_acceptance_hash": "cc64ab200e198041eac4805ba98a8ef47567684511b80051ff42eb85f4d25016",
  "source_R2X2_acceptance_hash": "4942d853bc2c16fd88f20de16c01600b879ad76cb9bd8f18aecd0043f19ecfe4",
  "B1_checkpoint_hash": "5b1dd85f840d4d00a7c1c1f3b7c0a9306031408ecf0c5a557f53ca0810c0265b",
  "B2_checkpoint_hash": "5cd86ceee6c9ad7f448676b59b3f1f5e8d80dc28bce106101b19d1286b6562a0",
  "lambda_pair": 0.5,
  "case_count": 500,
  "source_shape": [
    1220,
    979,
    3
  ],
  "target_shape": [
    320,
    256
  ],
  "source_shape_hw": [
    1220,
    979
  ],
  "resized_shape_hw": [
    319,
    256
  ],
  "resized_shape_wh": [
    256,
    319
  ],
  "target_shape_hw": [
    320,
    256
  ],
  "resize_rule": "single aspect-ratio-preserving resize 979x1220 -> 256x319 (WxH)",
  "padding_rule": "one all-zero row appended at bottom",
  "RGB_interpolation": "bilinear, float32, align_corners=false",
  "MH_interpolation": "bilinear, float32, align_corners=false",
  "mask_interpolation": "nearest",
  "crop_rotation_translation_warp": false,
  "condition_names": [
    "RGB",
    "RGB_B1MH",
    "RGB_B2MH"
  ],
  "output_dtype": "float32",
  "output_layout": "C\u00d7H\u00d7W",
  "cross_condition_RGB_identity": {
    "status": "PASS",
    "rgb_equal_rgb_b1mh_first3": true,
    "rgb_equal_rgb_b2mh_first3": true,
    "comparison": "np.array_equal, all 500 cases"
  },
  "model_forward_calls": 0,
  "training_calls": 0,
  "label_or_split_field_read_count": 0,
  "clinical_field_read_count": 0,
  "forbidden_field_access_count": 0,
  "protected_assets_unchanged": true,
  "classification_started": false,
  "formal_SO_R1_C_status": "FAIL",
  "official_SO_R2_authorization": false,
  "SO_R3_authorization": false,
  "next_stage_authorized": false
}

## Result

- Status: **COMPLETE_CLASSIFIER_INPUT_PREPARATION**. Three conditions were generated from the same high-resolution source coordinate system: `rgb/images` (500), `rgb_b1mh/images` (500), and `rgb_b2mh/images` (500), plus 500 common masks. Exact filename inventory matched the 500 canonical IDs in every directory.
- RGB tensors are `float32`, `3x320x256`, CHW; fused tensors are `float32`, `5x320x256`, CHW. Channel order is RGB0/RGB1/RGB2 followed by the corresponding B1 or B2 M/H channels.
- Geometry is one aspect-preserving resize from 979x1220 to 256x319 (WxH), followed by one zero row at the bottom. RGB and M/H use bilinear `align_corners=false`; masks use nearest-neighbor.
- Full audit passed for finite values, `float32` dtype, strict binary `0/255` masks, mask-outside zero, bottom padding zero, nonzero/nonconstant M/H maps, and exact RGB identity across all three conditions. B1/B2 channels trace exactly to their X1 NPZ sources.
- No label, split, demographic, patient-group, or clinical fields were read. Model forward calls and training calls are both zero. Protected assets were unchanged (`changed=0`, `missing=0`).

Only frozen data preparation was performed; classifier training and model forward were not started. Formal `SO-R1-C` remains `FAIL`; no SO-R2 or SO-R3 authorization is granted.

Verification in the `face2` environment: full `tests/so_r1` passed (`91 passed, 1 warning`); `git diff --check` passed.
