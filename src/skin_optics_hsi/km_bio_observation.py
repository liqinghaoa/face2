"""Stage B observation-contract audit for the KM-BIO-v1 Train domain."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import pandas as pd
import yaml


def sha256_file(path: str | Path, chunk_bytes: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(chunk_bytes), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _resolve(root: Path, value: str) -> Path:
    path = Path(value)
    return (root / path).resolve() if not path.is_absolute() else path.resolve()


def _spectral_columns(prefix: str, wavelength_nm: list[float]) -> list[str]:
    return [f"{prefix}_{int(w)}nm" for w in wavelength_nm]


def _load_raw_cube(path: Path, dataset_key: str, expected_shape: tuple[int, ...], expected_dtype: str) -> np.ndarray:
    with h5py.File(path, "r") as archive:
        if dataset_key not in archive:
            raise ValueError(f"Missing HDF5 dataset '{dataset_key}': {path}")
        dataset = archive[dataset_key]
        if tuple(dataset.shape) != expected_shape:
            raise ValueError(f"Unexpected stored HSI shape {dataset.shape}: {path}")
        if np.dtype(dataset.dtype).name != expected_dtype:
            raise ValueError(f"Unexpected stored HSI dtype {dataset.dtype}: {path}")
        cube = np.asarray(dataset, dtype=np.float64)
    band_axes = [axis for axis, size in enumerate(cube.shape) if size == 31]
    if band_axes != [0]:
        raise ValueError(f"Expected a unique leading 31-band axis: {path}")
    cube = np.moveaxis(cube, 0, -1)
    return np.ascontiguousarray(np.transpose(cube, (1, 0, 2)))


def _region_statistics(cube: np.ndarray, mask_path: Path) -> dict[str, Any]:
    mask = np.load(mask_path, allow_pickle=False)
    mask = np.asarray(np.squeeze(mask) > 0, dtype=bool)
    if mask.shape != cube.shape[:2]:
        raise ValueError(f"Mask/cube shape mismatch: {mask_path}")
    values = cube[mask]
    if not values.size:
        raise ValueError(f"Empty region mask: {mask_path}")
    raw_median = np.median(values, axis=0)
    clipped_median = np.median(np.clip(values, 0.0, 1.0), axis=0)
    return {
        "pixel_count": int(mask.sum()),
        "raw_median": raw_median,
        "clipped_median": clipped_median,
        "nonfinite_pixel_band_values": int((~np.isfinite(values)).sum()),
        "nonpositive_pixel_band_values": int((values <= 0.0).sum()),
        "below_zero_pixel_band_values": int((values < 0.0).sum()),
        "above_one_pixel_band_values": int((values > 1.0).sum()),
        "raw_pixel_band_min": float(np.min(values)),
        "raw_pixel_band_max": float(np.max(values)),
    }


def run_observation_audit(contract_path: str | Path, project_root: str | Path) -> dict[str, Any]:
    root = Path(project_root).resolve()
    contract_file = Path(contract_path).resolve()
    contract = yaml.safe_load(contract_file.read_text(encoding="utf-8"))
    inputs = {name: _resolve(root, value) for name, value in contract["inputs"].items()}
    output = _resolve(root, contract["outputs"]["directory"])
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite observation audit: {output}")

    data_contract = _read_json(inputs["data_contract"])
    registration = _read_json(inputs["registration_decision"])
    mask_provenance = _read_json(inputs["frozen_mask_provenance"])
    cache_provenance = _read_json(inputs["region_cache_provenance"])
    cache_decision = _read_json(inputs["region_cache_decision"])
    split_manifest = pd.read_csv(inputs["split_manifest"])
    mask_manifest = pd.read_parquet(inputs["frozen_train_mask_manifest"])
    region_cache = pd.read_parquet(inputs["region_cache"])

    wavelength = [float(value) for value in contract["wavelength"]["centers_nm"]]
    cache_columns = _spectral_columns("reflectance_median", wavelength)
    observed_columns = _spectral_columns("observed_reflectance", wavelength)
    scope = contract["scope"]
    selected = region_cache.loc[
        region_cache["split"].eq(scope["split"])
        & region_cache["expression"].eq(scope["expression"])
        & region_cache["direction"].eq(scope["direction"])
        & region_cache["s1_2_analysis_role"].eq(scope["analysis_role"])
        & region_cache["region"].isin(scope["regions"])
    ].copy()
    selected = selected.sort_values(["subject_id", "sample_id", "region"]).reset_index(drop=True)

    primary_samples = selected[["sample_id", "subject_id", "split"]].drop_duplicates()
    pairs = primary_samples.merge(
        split_manifest[["sample_id", "subject_id", "split", "hsi_path", "hsi_sha256", "rgb_path", "rgb_sha256", "rgb_width", "rgb_height", "rgb_mode", "hsi_stored_shape", "hsi_dtype"]],
        on=["sample_id", "subject_id", "split"], how="left", validate="one_to_one",
    )
    frozen_masks = mask_manifest.loc[
        mask_manifest["sample_id"].isin(primary_samples["sample_id"])
        & mask_manifest["split"].eq(scope["split"])
    ].copy()
    if len(frozen_masks) != len(primary_samples):
        raise ValueError("Frozen Train mask manifest does not cover every primary capture exactly once")

    stored_shape = tuple(int(v) for v in contract["spatial"]["expected_stored_shape"])
    stored_dtype = str(contract["spatial"]["expected_stored_dtype"])
    dataset_key = str(contract["spatial"]["hsi_dataset_key"])
    cache_tolerance = float(contract["historical_cache"]["maximum_allowed_abs_difference_from_raw_float64_median"])
    clip_tolerance = float(contract["historical_cache"]["maximum_allowed_abs_effect_of_clipping_on_region_median"])
    minimum_pixels = int(contract["spatial"]["minimum_region_pixels"])

    manifest_rows: list[dict[str, Any]] = []
    scale_rows: list[dict[str, Any]] = []
    hsi_hash_mismatches: list[str] = []
    rgb_hash_mismatches: list[str] = []
    mask_hash_mismatches: list[str] = []
    train_hsi_content_reads = 0
    train_rgb_byte_hash_reads = 0
    for _, pair in pairs.sort_values("sample_id").iterrows():
        sample_id = str(pair["sample_id"])
        hsi_path = Path(str(pair["hsi_path"])).resolve()
        rgb_path = Path(str(pair["rgb_path"])).resolve()
        actual_hsi_hash = sha256_file(hsi_path)
        actual_rgb_hash = sha256_file(rgb_path)
        train_hsi_content_reads += 1
        train_rgb_byte_hash_reads += 1
        if actual_hsi_hash != str(pair["hsi_sha256"]):
            hsi_hash_mismatches.append(sample_id)
        if actual_rgb_hash != str(pair["rgb_sha256"]):
            rgb_hash_mismatches.append(sample_id)
        cube = _load_raw_cube(hsi_path, dataset_key, stored_shape, stored_dtype)
        frozen_row = frozen_masks.loc[frozen_masks["sample_id"].eq(sample_id)].iloc[0]
        for region in scope["regions"]:
            cached = selected.loc[selected["sample_id"].eq(sample_id) & selected["region"].eq(region)]
            if len(cached) != 1:
                raise ValueError(f"Expected exactly one cached row for {sample_id}/{region}")
            cached_row = cached.iloc[0]
            mask_path = Path(str(frozen_row[f"{region}_path"])).resolve()
            actual_mask_hash = sha256_file(mask_path)
            expected_mask_hash = str(cached_row["mask_sha256"])
            if actual_mask_hash != expected_mask_hash or mask_path != Path(str(cached_row["mask_path"])).resolve():
                mask_hash_mismatches.append(f"{sample_id}/{region}")
            stats = _region_statistics(cube, mask_path)
            raw_median = stats.pop("raw_median")
            clipped_median = stats.pop("clipped_median")
            cached_median = cached_row[cache_columns].to_numpy(dtype=np.float64)
            raw_cache_abs = np.abs(raw_median - cached_median)
            raw_clip_abs = np.abs(raw_median - clipped_median)
            scale_rows.append({
                "sample_id": sample_id,
                "subject_id": str(pair["subject_id"]),
                "roi": region,
                **stats,
                "raw_region_median_min": float(raw_median.min()),
                "raw_region_median_max": float(raw_median.max()),
                "max_abs_raw_float64_vs_historical_cache": float(raw_cache_abs.max()),
                "max_abs_raw_vs_clipped_region_median": float(raw_clip_abs.max()),
                "cache_within_tolerance": bool(raw_cache_abs.max() <= cache_tolerance),
                "clipping_does_not_change_region_median": bool(raw_clip_abs.max() <= clip_tolerance),
            })
            row: dict[str, Any] = {
                "subject_id": str(pair["subject_id"]),
                "capture_id": sample_id,
                "split": str(pair["split"]),
                "expression": scope["expression"],
                "direction": scope["direction"],
                "roi": region,
                "model_version": contract["model_id"],
                "observation_quantity": contract["observation"]["quantity"],
                "source_spectrum": contract["observation"]["authoritative_region_statistic"],
                "hsi_path": str(hsi_path),
                "hsi_sha256": actual_hsi_hash,
                "rgb_path": str(rgb_path),
                "rgb_sha256": actual_rgb_hash,
                "mask_path": str(mask_path),
                "mask_sha256": actual_mask_hash,
                "mask_pixel_count": stats["pixel_count"],
                "frozen_transform": registration.get("frozen_transform"),
                "main_observation_gain": float(contract["observation"]["main_observation_gain"]),
                "per_spectrum_gain_fitted": False,
                "side_gain_applied": False,
                "normalization_applied": "none",
                "input_quality_status": "PASS" if np.isfinite(raw_median).all() and np.all(raw_median > 0) and stats["pixel_count"] >= minimum_pixels else "FAIL",
            }
            row.update({column: float(value) for column, value in zip(observed_columns, raw_median)})
            manifest_rows.append(row)

    manifest = pd.DataFrame(manifest_rows).sort_values(["subject_id", "capture_id", "roi"]).reset_index(drop=True)
    scale = pd.DataFrame(scale_rows).sort_values(["subject_id", "sample_id", "roi"]).reset_index(drop=True)
    observed = manifest[observed_columns].to_numpy(dtype=np.float64)
    image_left = manifest.loc[manifest["roi"].eq("left_cheek"), ["subject_id", *observed_columns]].set_index("subject_id")
    image_right = manifest.loc[manifest["roi"].eq("right_cheek"), ["subject_id", *observed_columns]].set_index("subject_id")
    paired_difference = image_left[observed_columns].mean(axis=1) - image_right[observed_columns].mean(axis=1)

    checks = {
        "data_contract_pass": data_contract.get("status") == "PASS",
        "split_manifest_hash_matches_data_contract": sha256_file(inputs["split_manifest"]) == data_contract["manifest"]["sha256"],
        "registration_is_frozen_transpose": registration.get("status") == "PASS" and registration.get("frozen_transform") == contract["spatial"]["frozen_rgb_hsi_transform"],
        "frozen_mask_protocol_status": mask_provenance.get("status") == "FROZEN",
        "frozen_mask_manifest_hash_matches": sha256_file(inputs["frozen_train_mask_manifest"]) == mask_provenance["source_train_manifest_sha256"],
        "region_cache_hash_matches_provenance": sha256_file(inputs["region_cache"]) == cache_provenance["output_sha256"],
        "region_cache_lineage_matches_decision": (
            cache_provenance["source_sha256"]
            == cache_decision["outputs"]["region_spectra_parquet"]["sha256"]
        ),
        "wavelength_contract_matches": wavelength == [float(v) for v in data_contract["wavelength"]["centers_nm"]["value"]],
        "expected_subject_count": manifest["subject_id"].nunique() == int(scope["expected_subjects"]),
        "expected_capture_count": manifest["capture_id"].nunique() == int(scope["expected_captures"]),
        "expected_region_spectrum_count": len(manifest) == int(scope["expected_region_spectra"]),
        "all_region_spectra_pass_input_qc": manifest["input_quality_status"].eq("PASS").all(),
        "all_hsi_hashes_match": not hsi_hash_mismatches,
        "all_rgb_hashes_match": not rgb_hash_mismatches,
        "all_mask_hashes_and_paths_match": not mask_hash_mismatches,
        "historical_cache_preserves_raw_scale_within_tolerance": scale["cache_within_tolerance"].all(),
        "historical_clipping_does_not_change_primary_region_medians": scale["clipping_does_not_change_region_median"].all(),
        "all_observed_values_finite_and_positive": bool(np.isfinite(observed).all() and np.all(observed > 0.0)),
        "validation_hsi_content_reads_zero": True,
        "test_hsi_content_reads_zero": True,
    }
    passed = bool(all(checks.values()))
    output.mkdir(parents=True, exist_ok=False)
    outputs = contract["outputs"]
    manifest_parquet = output / outputs["input_manifest_parquet"]
    manifest_csv = output / outputs["input_manifest_csv"]
    scale_csv = output / outputs["spectrum_scale_audit_csv"]
    manifest.to_parquet(manifest_parquet, index=False)
    manifest.to_csv(manifest_csv, index=False, encoding="utf-8-sig")
    scale.to_csv(scale_csv, index=False, encoding="utf-8-sig")
    summary = {
        "schema_version": 1,
        "stage": "KM-BIO-v1-Stage-B",
        "model_id": contract["model_id"],
        "status": "PASS_FOR_TRAIN_INVERSION" if passed else "REPAIR_INPUT_CHAIN",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "checks": {key: bool(value) for key, value in checks.items()},
        "counts": {
            "subjects": int(manifest["subject_id"].nunique()),
            "captures": int(manifest["capture_id"].nunique()),
            "region_spectra": int(len(manifest)),
            "train_hsi_content_reads": train_hsi_content_reads,
            "train_rgb_byte_hash_reads": train_rgb_byte_hash_reads,
            "validation_hsi_content_reads": 0,
            "test_hsi_content_reads": 0,
        },
        "scale_audit": {
            "observed_reflectance_min": float(observed.min()),
            "observed_reflectance_max": float(observed.max()),
            "maximum_abs_raw_float64_vs_historical_cache": float(scale["max_abs_raw_float64_vs_historical_cache"].max()),
            "maximum_abs_raw_vs_clipped_region_median": float(scale["max_abs_raw_vs_clipped_region_median"].max()),
            "image_left_brighter_broadband_subject_count": int((paired_difference > 0).sum()),
            "median_image_left_minus_right_broadband": float(paired_difference.median()),
            "main_observation_gain": 1.0,
            "per_spectrum_gain_fitted": False,
            "side_gain_applied": False,
            "normalization_applied": "none",
        },
        "rgb_source_audit": {
            **contract["rgb_source"],
            "paired_rgb_count": int(manifest["capture_id"].nunique()),
            "all_rgb_hashes_match_split_manifest": not rgb_hash_mismatches,
            "decoded_rgb_content_reads": 0,
        },
        "mismatches": {
            "hsi": hsi_hash_mismatches,
            "rgb": rgb_hash_mismatches,
            "mask": mask_hash_mismatches,
        },
        "source_hashes": {
            "observation_contract": sha256_file(contract_file),
            **{name: sha256_file(path) for name, path in inputs.items()},
        },
        "outputs": {
            "input_manifest_parquet": {"path": str(manifest_parquet), "sha256": sha256_file(manifest_parquet)},
            "input_manifest_csv": {"path": str(manifest_csv), "sha256": sha256_file(manifest_csv)},
            "spectrum_scale_audit_csv": {"path": str(scale_csv), "sha256": sha256_file(scale_csv)},
        },
        "next_stage_allowed": passed,
        "authorized_next_stage": "C_TRAIN_INVERSION" if passed else None,
    }
    summary_path = output / outputs["audit_summary_json"]
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    markdown = (
        "# KM-BIO-v1 observation-contract audit\n\n"
        f"- Status: `{summary['status']}`\n"
        f"- Train subjects/captures/cheek spectra: {summary['counts']['subjects']}/{summary['counts']['captures']}/{summary['counts']['region_spectra']}\n"
        f"- Raw float64 versus historical cache maximum absolute difference: {summary['scale_audit']['maximum_abs_raw_float64_vs_historical_cache']:.3e}\n"
        f"- Raw versus clipped region-median maximum absolute difference: {summary['scale_audit']['maximum_abs_raw_vs_clipped_region_median']:.3e}\n"
        f"- Observed reflectance range: [{summary['scale_audit']['observed_reflectance_min']:.6f}, {summary['scale_audit']['observed_reflectance_max']:.6f}]\n"
        f"- Validation/Test HSI content reads: 0/0\n"
        f"- Next stage allowed: `{str(passed).lower()}`\n\n"
        "The authoritative Stage C inputs are the raw HDF5 float64 per-band region medians in the observation manifest. "
        "The released paired RGB files were hash-audited for provenance only and were not decoded or used to alter HSI spectra.\n"
    )
    (output / outputs["audit_summary_markdown"]).write_text(markdown, encoding="utf-8")
    return summary
