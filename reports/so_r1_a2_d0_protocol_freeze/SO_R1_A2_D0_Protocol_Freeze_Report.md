# D0 Protocol Freeze Report

{
  "acceptance": {
    "protocol_id": "SO-R1-A2-D0",
    "status": "PASS",
    "protocol_status": "FROZEN",
    "latent_count_planned": 13500,
    "acquisition_count_planned": 67500,
    "pair_count_planned": 54000,
    "camera_light_pair_count": 24,
    "next_stage": "SO-R1-A2-G1",
    "next_stage_authorized": true,
    "full_generation_stage_authorized": true,
    "full_generation_started": false,
    "formal_rgb_generated": 0,
    "training_authorized": false
  },
  "resource": {
    "projected_final_compressed_gib": 14.0965,
    "uncompressed_total_gib": 28.84,
    "point_runtime_hours": 12.8,
    "conservative_runtime_hours": 16.0,
    "peak_rss_gib": 0.68,
    "source_c2_projection_hash": "a078373c0cf4c404023617671cf8d5eeed562a35263067dd7bff1030b6c78091",
    "current_free_gib": 111.05290222167969,
    "required_free_gib": 24.0965,
    "pass": true
  },
  "protection": {
    "before_count": 1636,
    "after_count": 1636,
    "changed_count": 0,
    "missing_count": 0,
    "unexpected_new_protected_count": 0,
    "pass": true
  },
  "consistency": {
    "counts": {
      "latent": 13500,
      "acquisition": 67500,
      "pair": 54000
    },
    "allowlist": {
      "rows": 24,
      "unique": 24,
      "quality_pass": 24,
      "stress_pass": 24
    },
    "formal_array_file_count": 0,
    "formal_rgb_generated": 0,
    "no_forbidden_camera": true,
    "id_no_unseen": true,
    "id_no_fl11": true,
    "camera_ood_no_fl11": true,
    "light_ood_no_unseen": true,
    "mh_plan_pass": true,
    "pass": true
  }
}
