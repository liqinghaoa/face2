# SO-R1-A0-AM2-Lite-R1 High-Clip Attribution Report

The deterministic AM2 replay identified four high-clip cases. Overall decision: **NUISANCE_SAMPLER**.

## Inputs and protected scope
Final fourth seen camera is Canon 300D; final cameras are Canon 5DMarkII, Nikon D80, Olympus E-PL2, Canon 300D, Canon 1DMarkIII, Nikon D5100. All work is audit-only and no frozen SO-0, camera, light, or historical AM2 artifact was modified.

## Metric contract
High clipping is the fraction of valid preclip RGB channel elements >= 1-1e-6 (denominator 3N). Pixel-any, pixel-all, individual RGB, and corresponding low-end metrics are retained in the case CSV. AM2 used preclip RGB elements >1, which selects the same four cases; therefore no metric-definition mismatch affected the failure conclusion.

## Exact failure extraction
The four reconstructed cases, their seeds, latent fields, appearance fields, pre/postclip statistics, and every clipping metric are recorded in four_high_clip_cases.csv. Same-latent A0–A4 comparisons are in four_case_same_latent_group_comparison.csv.

## Controlled evidence
Camera swaps hold latent, M/H, mask, light, S, P, exposure, and seeds fixed. Exposure, specular, joint, and appearance-reference grids were rendered with the frozen CPU SO-0 renderer. Spatial overlap statistics use raw linear values; the montage is explicitly display-only.

## Decision
{
  "case_decisions": [
    {
      "sample_id": "L024_A1",
      "decision": "EXPOSURE_DRIVEN",
      "camera_fail_count": 3,
      "camera_specific": false,
      "acquisition_global": false,
      "ev0_high_clip": 0.0,
      "p0_high_clip": 0.1030120849609375,
      "ev0_p0_high_clip": 0.0
    },
    {
      "sample_id": "L024_A2",
      "decision": "EXPOSURE_DRIVEN",
      "camera_fail_count": 6,
      "camera_specific": false,
      "acquisition_global": true,
      "ev0_high_clip": 0.0,
      "p0_high_clip": 0.1612396240234375,
      "ev0_p0_high_clip": 0.0
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
      "decision": "EXPOSURE_DRIVEN",
      "camera_fail_count": 6,
      "camera_specific": false,
      "acquisition_global": true,
      "ev0_high_clip": 0.0,
      "p0_high_clip": 0.1122589111328125,
      "ev0_p0_high_clip": 0.0
    }
  ],
  "overall_decision": "NUISANCE_SAMPLER",
  "camera_specific_count": 0,
  "unresolved_count": 0,
  "nuisance_sampler_revision_permitted": true
}

## Corrective validation
{
  "corrected64": {
    "AUDIT_ONLY": true,
    "TRAINING_FORBIDDEN": true,
    "ATTRIBUTION_ONLY": true,
    "NOT_PART_OF_FORMAL_DATASET": true,
    "pass": true,
    "latent_count": 64,
    "acquisition_count": 320,
    "pair_count": 256,
    "camera_light_coverage": 24,
    "same_latent_integrity": true,
    "variable_isolation_pass": true,
    "high_clip_violation_count": 0,
    "low_clip_violation_count": 0,
    "major_channel_collapse_count": 0,
    "nonfinite_count": 0,
    "replay_requested": 64,
    "replay_matched": 64,
    "bias_pass": true
  },
  "independent128": {
    "AUDIT_ONLY": true,
    "TRAINING_FORBIDDEN": true,
    "ATTRIBUTION_ONLY": true,
    "NOT_PART_OF_FORMAL_DATASET": true,
    "pass": true,
    "latent_count": 128,
    "acquisition_count": 640,
    "pair_count": 512,
    "camera_light_coverage": 24,
    "same_latent_integrity": true,
    "variable_isolation_pass": true,
    "high_clip_violation_count": 0,
    "low_clip_violation_count": 0,
    "major_channel_collapse_count": 0,
    "nonfinite_count": 0,
    "replay_requested": 32,
    "replay_matched": 32,
    "bias_pass": true
  }
}

## Disposition
R1 status: **PASS**. The candidate exposure protocol remains non-frozen and no full dataset generation or training was started.
