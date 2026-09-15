# SO-R1-A0-AM2-Lite-R1 High-Clip Attribution Report

The deterministic AM2 replay identified four high-clip cases. Overall decision: **STOP_CAMERA_OR_UNRESOLVED**.

## Inputs and protected scope
Final fourth seen camera is Canon 300D; final cameras are Canon 5DMarkII, Nikon D80, Olympus E-PL2, Canon 300D, Canon 1DMarkIII, Nikon D5100. All work is audit-only and no frozen SO-0, camera, light, or historical AM2 artifact was modified.

## Metric contract
High clipping is the fraction of valid preclip RGB channel elements >= 1-1e-6 (denominator 3N). Pixel-any, pixel-all, individual RGB, and corresponding low-end metrics are retained in the case CSV. AM2 used preclip RGB elements >1, which selects the same four cases; therefore no metric-definition mismatch affected the failure conclusion.

## Exact failure extraction
The four reconstructed cases, their seeds, latent fields, appearance fields, pre/postclip statistics, and every clipping metric are recorded in four_high_clip_cases.csv. Same-latent A0–A4 comparisons are in four_case_same_latent_group_comparison.csv.

## Controlled evidence
Camera swaps hold latent, M/H, mask, light, S, P, exposure, and seeds fixed. Exposure, specular, joint, and appearance-reference grids were rendered with the frozen CPU SO-0 renderer. Spatial overlap statistics use raw linear values; the montage is explicitly display-only.

## Decision and stopping rule
{
  "case_decisions": [
    {
      "sample_id": "L024_A1",
      "decision": "UNRESOLVED",
      "camera_fail_count": 0,
      "camera_specific": false,
      "acquisition_global": false,
      "ev0_high_clip": 0.0,
      "p0_high_clip": 0.0067138671875,
      "ev0_p0_high_clip": 0.0
    },
    {
      "sample_id": "L024_A2",
      "decision": "EXPOSURE_DRIVEN",
      "camera_fail_count": 0,
      "camera_specific": false,
      "acquisition_global": false,
      "ev0_high_clip": 0.0013631184895833333,
      "p0_high_clip": 0.055511474609375,
      "ev0_p0_high_clip": 0.0013631184895833333
    },
    {
      "sample_id": "L040_A4",
      "decision": "EXPOSURE_DRIVEN",
      "camera_fail_count": 6,
      "camera_specific": false,
      "acquisition_global": true,
      "ev0_high_clip": 0.0027313232421875,
      "p0_high_clip": 0.150146484375,
      "ev0_p0_high_clip": 0.0027313232421875
    },
    {
      "sample_id": "L048_A2",
      "decision": "UNRESOLVED",
      "camera_fail_count": 0,
      "camera_specific": false,
      "acquisition_global": false,
      "ev0_high_clip": 0.0007069905598958334,
      "p0_high_clip": 0.0005594889322916666,
      "ev0_p0_high_clip": 0.0007069905598958334
    }
  ],
  "overall_decision": "STOP_CAMERA_OR_UNRESOLVED",
  "camera_specific_count": 0,
  "unresolved_count": 2,
  "nuisance_sampler_revision_permitted": false
}

## Disposition
Because this run did not establish a safe nuisance-only explanation, it stops before sampler revision, corrected 64-latent regression, independent 128-latent validation, camera freeze, P0 pass, full generation, or training.
