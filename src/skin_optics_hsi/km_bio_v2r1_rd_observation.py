"""KM-BIO-v2R.1 R-D validation observation contract.

This is the first batch in the KM-BIO line that opens raw Validation HDF5
content.  It rebuilds the frozen Validation bilateral symmetric spectra with the
exact reader, transpose, mask statistic and aggregation used for the Train R-B
manifest, then audits every byte that was consumed.

Only the frozen ``valid`` split is read.  Train, Test and clinical-500 content is
never opened, and no observation quantity is normalised, side-corrected or
clipped.  The output manifest is the authoritative R-D Validation input.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from .km_bio_observation import _load_raw_cube, _region_statistics, sha256_file
from .km_bio_v2r_observation import symmetric_log_reflectance


class ContractError(RuntimeError):
    """A hard R-D observation integrity violation."""


def _resolve(root: Path, value: str) -> Path:
    path = Path(value)
    return (root / path).resolve() if not path.is_absolute() else path.resolve()


def _columns(prefix: str, wavelength: list[float]) -> list[str]:
    return [f"{prefix}_{int(value)}nm" for value in wavelength]


def _jsonable(value: Any) -> Any:
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def run_validation_observation_audit(config_path: str | Path, project_root: str | Path) -> dict[str, Any]:
    root = Path(project_root).resolve()
    contract_file = Path(config_path).resolve()
    contract = yaml.safe_load(contract_file.read_text(encoding="utf-8"))
    if contract["contract_id"] != "km_bio_v2r1_rd_validation_observation_contract":
        raise ContractError("Unexpected R-D validation observation contract id")
    if contract["scope"]["split"] != "valid":
        raise ContractError("R-D validation observation contract must declare the valid split")
    if any(contract["scope"][key] for key in ("train_content_allowed", "test_content_allowed", "clinical_500_content_allowed")):
        raise ContractError("R-D validation observation contract permits forbidden content")

    inputs = {name: _resolve(root, value) for name, value in contract["inputs"].items()}
    implementation = {name: _resolve(root, value) for name, value in contract["implementation"].items()}
    missing = sorted(name for name, path in {**inputs, **implementation}.items() if not path.exists())
    if missing:
        raise ContractError(f"Missing R-D validation observation inputs: {missing}")
    output = _resolve(root, contract["outputs"]["directory"])
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite R-D validation observation audit: {output}")

    scope = contract["scope"]
    wavelength = [float(value) for value in contract["wavelength"]["centers_nm"]]
    fit_index = [int(value) for value in contract["wavelength"]["fit_index_zero_based"]]
    edge_index = [int(value) for value in contract["wavelength"]["edge_diagnostic_index_zero_based"]]
    observed_columns = _columns("observed_reflectance", wavelength)
    cache_columns = _columns("reflectance_median", wavelength)

    v2r_contract = yaml.safe_load(inputs["v2r_observation_contract"].read_text(encoding="utf-8"))
    v2r_audit = json.loads(inputs["v2r_observation_audit"].read_text(encoding="utf-8"))

    split_manifest = pd.read_csv(inputs["split_manifest"])
    frozen_masks = pd.read_parquet(inputs["frozen_mask_manifest_valid"])
    # Only the ``valid`` partition of the shared region cache is materialised.
    region_cache = pd.read_parquet(inputs["region_spectra"], filters=[("split", "==", scope["split"])])

    valid_rows = region_cache.loc[
        region_cache["split"].eq(scope["split"])
        & region_cache["expression"].eq(scope["expression"])
        & region_cache["direction"].eq(scope["direction"])
        & region_cache["s1_2_analysis_role"].eq(scope["analysis_role"])
        & region_cache["region"].isin(scope["input_rois"])
    ].copy()
    valid_rows = valid_rows.sort_values(["subject_id", "sample_id", "region"]).reset_index(drop=True)
    if len(valid_rows) != int(scope["expected_input_region_spectra"]):
        raise ContractError(f"Unexpected Validation region-row count: {len(valid_rows)}")
    if not valid_rows["extraction_status"].eq("USABLE").all():
        raise ContractError("A Validation primary region spectrum is not USABLE")

    primary = valid_rows[["sample_id", "subject_id", "split"]].drop_duplicates().reset_index(drop=True)
    pairs = primary.merge(
        split_manifest[[
            "sample_id", "subject_id", "split", "expression", "direction",
            "hsi_path", "hsi_sha256", "rgb_path", "rgb_sha256",
            "hsi_stored_shape", "hsi_dtype", "hsi_dataset_key",
        ]],
        on=["sample_id", "subject_id", "split"], how="left", validate="one_to_one",
    )
    if pairs["hsi_path"].isna().any():
        raise ContractError("A Validation primary capture is missing from the split manifest")
    if not pairs["split"].eq(scope["split"]).all():
        raise ContractError("A selected primary capture is not on the frozen Validation split")
    if not pairs["expression"].eq(scope["expression"]).all() or not pairs["direction"].eq(scope["direction"]).all():
        raise ContractError("A selected primary capture deviates from the registered neutral/front scope")
    if sorted(pairs["subject_id"].astype(str)) != sorted(map(str, scope["expected_subject_ids"])):
        raise ContractError("Validation subject set differs from the frozen mask manifest subjects")

    frozen_by_sample = frozen_masks.set_index("sample_id", drop=False)
    if not set(map(str, pairs["sample_id"])).issubset(set(map(str, frozen_by_sample["sample_id"]))):
        raise ContractError("Frozen Validation mask manifest does not cover every primary capture")

    stored_shape = tuple(int(value) for value in contract["spatial"]["expected_stored_shape"])
    stored_dtype = str(contract["spatial"]["expected_stored_dtype"])
    dataset_key = str(contract["spatial"]["hsi_dataset_key"])
    frozen_transform = str(contract["spatial"]["frozen_rgb_hsi_transform"])
    minimum_pixels = int(contract["spatial"]["minimum_region_pixels"])
    path_fragment = str(contract["spatial"]["required_path_fragment"])
    cache_tolerance = float(contract["audit"]["maximum_allowed_abs_difference_from_float32_cache"])
    reconstruction_tolerance = float(contract["audit"]["tolerance_reconstruction_abs"])

    manifest_rows: list[dict[str, Any]] = []
    scale_rows: list[dict[str, Any]] = []
    side_rows: list[dict[str, Any]] = []
    hsi_hash_mismatches: list[str] = []
    mask_hash_mismatches: list[str] = []
    forbidden_paths: list[str] = []
    validation_hsi_reads = 0
    train_hsi_reads = 0
    test_hsi_reads = 0

    for _, pair in pairs.sort_values("sample_id").iterrows():
        sample_id = str(pair["sample_id"])
        hsi_path = Path(str(pair["hsi_path"])).resolve()
        parts = {part.lower() for part in hsi_path.parts}
        if path_fragment not in parts:
            forbidden_paths.append(str(hsi_path))
            raise ContractError(f"Validation capture is not stored under the valid split: {hsi_path}")
        if "train" in parts or "test" in parts:
            forbidden_paths.append(str(hsi_path))
            raise ContractError(f"Validation batch resolved a train/test HSI path: {hsi_path}")
        actual_hsi_hash = sha256_file(hsi_path)
        validation_hsi_reads += 1
        if actual_hsi_hash != str(pair["hsi_sha256"]):
            hsi_hash_mismatches.append(sample_id)
        raw_shape = pair["hsi_stored_shape"]
        declared_shape = [int(value) for value in raw_shape.strip("[]").replace(",", " ").split()] if isinstance(raw_shape, str) else [int(value) for value in raw_shape]
        if tuple(declared_shape) != stored_shape or str(pair["hsi_dtype"]) != stored_dtype:
            raise ContractError(f"Validation HSI storage contract mismatch for {sample_id}")
        cube = _load_raw_cube(hsi_path, dataset_key, stored_shape, stored_dtype)
        frozen_row = frozen_by_sample.loc[sample_id]
        spectra: dict[str, np.ndarray] = {}
        for region in scope["input_rois"]:
            cached = valid_rows.loc[valid_rows["sample_id"].eq(sample_id) & valid_rows["region"].eq(region)]
            if len(cached) != 1:
                raise ContractError(f"Expected exactly one cached row for {sample_id}/{region}")
            cached_row = cached.iloc[0]
            mask_path = Path(str(cached_row["mask_path"])).resolve()
            declared_path = Path(str(frozen_row[f"{region}_path"])).resolve()
            if mask_path != declared_path:
                raise ContractError(f"Mask path disagrees between region cache and frozen r2 manifest: {sample_id}/{region}")
            actual_mask_hash = sha256_file(mask_path)
            if actual_mask_hash != str(cached_row["mask_sha256"]):
                mask_hash_mismatches.append(f"{sample_id}/{region}")
            stats = _region_statistics(cube, mask_path)
            raw_median = stats.pop("raw_median")
            clipped_median = stats.pop("clipped_median")
            cached_median = cached_row[cache_columns].to_numpy(dtype=np.float64)
            raw_cache_abs = np.abs(raw_median - cached_median)
            spectra[region] = raw_median
            scale_rows.append({
                "sample_id": sample_id,
                "subject_id": str(pair["subject_id"]),
                "roi": region,
                **stats,
                "raw_region_median_min": float(raw_median.min()),
                "raw_region_median_max": float(raw_median.max()),
                "max_abs_raw_float64_vs_float32_cache": float(raw_cache_abs.max()),
                "max_abs_raw_vs_clipped_region_median": float(np.abs(raw_median - clipped_median).max()),
                "cache_within_tolerance": bool(raw_cache_abs.max() <= cache_tolerance),
            })
            quality_pass = bool(
                np.isfinite(raw_median).all()
                and np.all(raw_median > 0.0)
                and stats["pixel_count"] >= minimum_pixels
                and stats["nonfinite_pixel_band_values"] == 0
            )
            row: dict[str, Any] = {
                "subject_id": str(pair["subject_id"]),
                "capture_id": sample_id,
                "split": str(pair["split"]),
                "expression": scope["expression"],
                "direction": scope["direction"],
                "roi": region,
                "model_version": contract["model_id"],
                "observation_quantity": contract["observation"]["quantity"],
                "source_spectrum": contract["observation"]["input_statistic"],
                "hsi_path": str(hsi_path),
                "hsi_sha256": actual_hsi_hash,
                "mask_path": str(mask_path),
                "mask_sha256": actual_mask_hash,
                "mask_pixel_count": stats["pixel_count"],
                "frozen_transform": frozen_transform,
                "main_observation_gain": 1.0,
                "per_spectrum_gain_fitted": False,
                "side_gain_applied": False,
                "normalization_applied": "none",
                "input_quality_status": "PASS" if quality_pass else "FAIL",
            }
            row.update({column: float(value) for column, value in zip(observed_columns, raw_median)})
            manifest_rows.append(row)

        left = spectra["left_cheek"]
        right = spectra["right_cheek"]
        symmetric = symmetric_log_reflectance(left, right)
        reconstructed = np.exp((np.log(left) + np.log(right)) / 2.0)
        side_rows.append({
            "subject_id": str(pair["subject_id"]),
            "capture_id": sample_id,
            "left_broadband_mean": float(left.mean()),
            "right_broadband_mean": float(right.mean()),
            "left_minus_right_broadband": float(left.mean() - right.mean()),
            "log_left_minus_log_right_mean": float(np.mean(np.log(left) - np.log(right))),
            "left_brighter_broadband": bool(left.mean() > right.mean()),
            "paired_capture_metadata_agreement": True,
            "symmetric_min": float(symmetric.min()),
            "symmetric_max": float(symmetric.max()),
            "formula_reconstruction_max_abs": float(np.max(np.abs(symmetric - reconstructed))),
            "fit_band_min": float(symmetric[fit_index].min()),
            "fit_band_max": float(symmetric[fit_index].max()),
            "edge_band_values": json.dumps({str(int(wavelength[i])): float(symmetric[i]) for i in edge_index}, sort_keys=True),
        })

    region_manifest = pd.DataFrame(manifest_rows).sort_values(["subject_id", "capture_id", "roi"]).reset_index(drop=True)
    side_audit = pd.DataFrame(side_rows).sort_values(["subject_id", "capture_id"]).reset_index(drop=True)
    scale_audit = pd.DataFrame(scale_rows).sort_values(["subject_id", "sample_id", "roi"]).reset_index(drop=True)

    symmetric_rows: list[dict[str, Any]] = []
    paired_equal_fields = (
        "subject_id", "capture_id", "split", "expression", "direction", "model_version",
        "observation_quantity", "source_spectrum", "hsi_path", "hsi_sha256", "frozen_transform",
        "main_observation_gain", "per_spectrum_gain_fitted", "side_gain_applied",
        "normalization_applied", "input_quality_status",
    )
    metadata_failures: list[str] = []
    pair_failures: list[str] = []
    for (subject_id, capture_id), group in region_manifest.groupby(["subject_id", "capture_id"], sort=True):
        by_roi = {str(row["roi"]): row for _, row in group.iterrows()}
        if set(by_roi) != set(scope["input_rois"]) or len(group) != 2:
            pair_failures.append(f"{subject_id}/{capture_id}")
            continue
        left = by_roi["left_cheek"]
        right = by_roi["right_cheek"]
        if not all(left[field] == right[field] for field in paired_equal_fields):
            metadata_failures.append(f"{subject_id}/{capture_id}")
        left_spectrum = left[observed_columns].to_numpy(dtype=np.float64)
        right_spectrum = right[observed_columns].to_numpy(dtype=np.float64)
        symmetric = symmetric_log_reflectance(left_spectrum, right_spectrum)
        symmetric_rows.append({
            "subject_id": str(subject_id),
            "capture_id": str(capture_id),
            "split": str(left["split"]),
            "expression": str(left["expression"]),
            "direction": str(left["direction"]),
            "model_version": contract["model_id"],
            "observation_quantity": str(left["observation_quantity"]),
            "source_spectrum": "bilateral_log_geometric_mean_from_validation_raw_float64_region_medians",
            "left_roi": "left_cheek",
            "right_roi": "right_cheek",
            "left_hsi_path": str(left["hsi_path"]),
            "right_hsi_path": str(right["hsi_path"]),
            "left_hsi_sha256": str(left["hsi_sha256"]),
            "right_hsi_sha256": str(right["hsi_sha256"]),
            "left_mask_path": str(left["mask_path"]),
            "right_mask_path": str(right["mask_path"]),
            "left_mask_sha256": str(left["mask_sha256"]),
            "right_mask_sha256": str(right["mask_sha256"]),
            "left_mask_pixel_count": int(left["mask_pixel_count"]),
            "right_mask_pixel_count": int(right["mask_pixel_count"]),
            "frozen_transform": str(left["frozen_transform"]),
            "main_observation_gain": 1.0,
            "per_spectrum_gain_fitted": False,
            "side_gain_applied": False,
            "normalization_applied": "none",
            "aggregation_domain": contract["observation"]["aggregation_domain"],
            "aggregation_epsilon": 0.0,
            "input_quality_status": "PASS",
            "fit_band_index_zero_based": ",".join(str(i) for i in fit_index),
            "edge_band_index_zero_based": ",".join(str(i) for i in edge_index),
            **{column: float(value) for column, value in zip(observed_columns, symmetric)},
        })

    symmetric_manifest = pd.DataFrame(symmetric_rows).sort_values(["subject_id", "capture_id"]).reset_index(drop=True)
    observed = symmetric_manifest[observed_columns].to_numpy(dtype=np.float64)
    region_observed = region_manifest[observed_columns].to_numpy(dtype=np.float64)

    checks = {
        "v2r_train_observation_contract_pass": bool(
            v2r_contract.get("status") == "observation_audit_pass"
            and v2r_audit.get("status") == "PASS_FOR_V2R_TRAIN_INVERSION"
            and v2r_contract["wavelength"]["centers_nm"] == [int(value) for value in wavelength]
            and v2r_contract["wavelength"]["fit_index_zero_based"] == fit_index
            and v2r_contract["wavelength"]["edge_diagnostic_index_zero_based"] == edge_index
            and v2r_contract["observation"]["formula"] == contract["observation"]["formula"]
        ),
        "source_manifest_rows_are_valid_split_only": bool(valid_rows["split"].eq(scope["split"]).all()),
        "no_train_or_test_path_is_opened": bool(not forbidden_paths and train_hsi_reads == 0 and test_hsi_reads == 0),
        "exact_left_right_pairing": bool(not pair_failures and len(side_audit) == int(scope["expected_output_symmetric_spectra"])),
        "paired_capture_metadata_agreement": bool(not metadata_failures),
        "subject_and_capture_counts": bool(
            symmetric_manifest["subject_id"].nunique() == int(scope["expected_subjects"])
            and symmetric_manifest["capture_id"].nunique() == int(scope["expected_captures"])
            and len(symmetric_manifest) == int(scope["expected_output_symmetric_spectra"])
            and len(region_manifest) == int(scope["expected_input_region_spectra"])
        ),
        "no_side_gain_or_normalization": bool(
            (region_manifest["side_gain_applied"] == False).all()
            and (region_manifest["per_spectrum_gain_fitted"] == False).all()
            and region_manifest["normalization_applied"].eq("none").all()
        ),
        "all_hsi_hashes_match": bool(not hsi_hash_mismatches),
        "all_mask_hashes_and_paths_match": bool(not mask_hash_mismatches),
        "raw_float64_vs_float32_cache_within_tolerance": bool(scale_audit["cache_within_tolerance"].all()),
        "all_region_spectra_pass_input_qc": bool(region_manifest["input_quality_status"].eq("PASS").all()),
        "strict_positive_log_inputs": bool(np.isfinite(region_observed).all() and np.all(region_observed > 0.0)),
        "symmetric_formula_reconstruction": bool(side_audit["formula_reconstruction_max_abs"].max() <= reconstruction_tolerance),
        "fit_and_edge_band_roles": bool(
            [wavelength[i] for i in fit_index] == [float(v) for v in contract["wavelength"]["fit_centers_nm"]]
            and sorted(fit_index + edge_index) == list(range(len(wavelength)))
        ),
        "all_symmetric_values_finite_positive": bool(np.isfinite(observed).all() and np.all(observed > 0.0)),
    }
    passed = bool(all(checks.values()))
    if not passed:
        raise ContractError(f"R-D validation observation contract failed: {checks}")

    output.mkdir(parents=True, exist_ok=False)
    outputs = contract["outputs"]
    manifest_parquet = output / outputs["symmetric_manifest_parquet"]
    manifest_csv = output / outputs["symmetric_manifest_csv"]
    side_csv = output / outputs["side_pair_audit_csv"]
    scale_csv = output / outputs["scale_audit_csv"]
    symmetric_manifest.to_parquet(manifest_parquet, index=False)
    symmetric_manifest.to_csv(manifest_csv, index=False, encoding="utf-8-sig")
    side_audit.to_csv(side_csv, index=False, encoding="utf-8-sig")
    scale_audit.to_csv(scale_csv, index=False, encoding="utf-8-sig")
    region_manifest.to_csv(output / "validation_region_spectra_manifest.csv", index=False, encoding="utf-8-sig")

    summary = {
        "schema_version": 1,
        "stage": "KM-BIO-v2R.1-R-D-validation-observation",
        "model_id": contract["model_id"],
        "protocol_version": contract["protocol_version"],
        "status": "PASS_FOR_R_D_VALIDATION_INVERSION",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "checks": {key: bool(value) for key, value in checks.items()},
        "counts": {
            "subjects": int(symmetric_manifest["subject_id"].nunique()),
            "captures": int(symmetric_manifest["capture_id"].nunique()),
            "region_spectra": int(len(region_manifest)),
            "symmetric_spectra": int(len(symmetric_manifest)),
            "validation_hsi_content_reads": validation_hsi_reads,
            "train_hsi_content_reads": train_hsi_reads,
            "test_hsi_content_reads": test_hsi_reads,
            "clinical_500_content_reads": 0,
        },
        "wavelength": {
            "full_centers_nm": wavelength,
            "fit_centers_nm": contract["wavelength"]["fit_centers_nm"],
            "edge_diagnostic_centers_nm": contract["wavelength"]["edge_diagnostic_centers_nm"],
        },
        "observation": {
            "quantity": contract["observation"]["quantity"],
            "formula": contract["observation"]["formula"],
            "input_statistic": contract["observation"]["input_statistic"],
            "reflectance_clipping_applied": False,
            "side_gain_applied": False,
            "normalization_applied": "none",
        },
        "scale_audit": {
            "observed_reflectance_min": float(observed.min()),
            "observed_reflectance_max": float(observed.max()),
            "region_median_min": float(region_observed.min()),
            "region_median_max": float(region_observed.max()),
            "maximum_abs_raw_float64_vs_float32_cache": float(scale_audit["max_abs_raw_float64_vs_float32_cache"].max()),
            "maximum_formula_reconstruction_abs": float(side_audit["formula_reconstruction_max_abs"].max()),
            "minimum_mask_pixel_count": int(region_manifest["mask_pixel_count"].min()),
            "left_brighter_broadband_subject_count": int(side_audit["left_brighter_broadband"].sum()),
            "median_left_minus_right_broadband": float(side_audit["left_minus_right_broadband"].median()),
            "median_log_left_minus_log_right_mean": float(side_audit["log_left_minus_log_right_mean"].median()),
        },
        "subject_ids": sorted(symmetric_manifest["subject_id"].astype(str).tolist()),
        "frozen_mask_manifest_sha256": sha256_file(inputs["frozen_mask_manifest_valid"]),
        "source_hashes": {name: sha256_file(path) for name, path in inputs.items()},
        "implementation_hashes": {name: sha256_file(path) for name, path in implementation.items()},
        "outputs": {
            "symmetric_manifest_parquet": {"path": str(manifest_parquet), "sha256": sha256_file(manifest_parquet)},
            "symmetric_manifest_csv": {"path": str(manifest_csv), "sha256": sha256_file(manifest_csv)},
            "side_pair_audit_csv": {"path": str(side_csv), "sha256": sha256_file(side_csv)},
            "scale_audit_csv": {"path": str(scale_csv), "sha256": sha256_file(scale_csv)},
        },
        "authorized_next_stage": "R_D_FREEZE_AND_VALIDATION_INVERSION",
        "validation_observation_allowed": True,
        "test_or_clinical_500_allowed": False,
        "no_validation_information_used_for_freeze": True,
    }
    summary_json = output / outputs["audit_summary_json"]
    summary_md = output / outputs["audit_summary_markdown"]
    summary_json.write_text(json.dumps(_jsonable(summary), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    markdown = [
        "# KM-BIO-v2R.1 R-D Validation observation audit",
        "",
        f"- Status: `{summary['status']}`",
        f"- Validation subjects/captures/cheek spectra: {summary['counts']['subjects']}/{summary['counts']['captures']}/{summary['counts']['region_spectra']}",
        f"- Validation symmetric spectra: {summary['counts']['symmetric_spectra']}",
        f"- Validation HDF5 content reads: {summary['counts']['validation_hsi_content_reads']} (Train/Test reads: 0/0)",
        f"- Raw float64 vs float32 cache maximum absolute difference: {summary['scale_audit']['maximum_abs_raw_float64_vs_float32_cache']:.3e}",
        f"- Symmetric formula reconstruction maximum absolute difference: {summary['scale_audit']['maximum_formula_reconstruction_abs']:.3e}",
        f"- Observed symmetric reflectance range: [{summary['scale_audit']['observed_reflectance_min']:.6f}, {summary['scale_audit']['observed_reflectance_max']:.6f}]",
        "",
        "This is the first KM-BIO batch that opens Validation raw HDF5 content. Train, Test and "
        "clinical-500 HSI were not opened, and no side gain, per-spectrum gain or clipping was applied.",
        "",
    ]
    (output / outputs["audit_summary_markdown"]).write_text("\n".join(markdown), encoding="utf-8")

    artifact_rows = [
        {"role": "input", "name": name, "path": str(path), "sha256": sha256_file(path)}
        for name, path in inputs.items()
    ]
    artifact_rows.append({"role": "config", "name": "rd_validation_observation_contract", "path": str(contract_file), "sha256": sha256_file(contract_file)})
    for name, path in implementation.items():
        artifact_rows.append({"role": "implementation", "name": name, "path": str(path), "sha256": sha256_file(path)})
    for path in sorted(output.iterdir(), key=lambda item: item.name):
        if path.is_file() and path.name != outputs["artifact_hash_manifest_csv"]:
            artifact_rows.append({"role": "output", "name": path.name, "path": str(path), "sha256": sha256_file(path)})
    pd.DataFrame(artifact_rows).to_csv(output / outputs["artifact_hash_manifest_csv"], index=False, encoding="utf-8-sig")
    return summary
