"""KM-BIO-v2R R-B observation-contract transformation.

The transformation consumes only the already audited v1 Train region
manifest. It never opens raw HSI, RGB, mask, Validation, Test, or clinical
files.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml


def sha256_file(path: str | Path, chunk_bytes: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(chunk_bytes), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _resolve(root: Path, value: str) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (root / path).resolve()


def _spectral_columns(prefix: str, wavelength_nm: list[float]) -> list[str]:
    return [f"{prefix}_{int(value)}nm" for value in wavelength_nm]


def symmetric_log_reflectance(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    """Compute the geometric mean in reflectance, via the log domain."""

    left = np.asarray(left, dtype=np.float64)
    right = np.asarray(right, dtype=np.float64)
    if left.shape != right.shape or left.ndim != 1 or not np.isfinite(left).all() or not np.isfinite(right).all() or np.any(left <= 0) or np.any(right <= 0):
        raise ValueError("left and right spectra must be aligned, finite and strictly positive")
    return np.exp((np.log(left) + np.log(right)) / 2.0)


def build_symmetric_observation(contract_path: str | Path, project_root: str | Path) -> dict[str, Any]:
    """Build the v2R Train symmetric manifest and its audit outputs."""

    root = Path(project_root).resolve()
    contract_file = Path(contract_path).resolve()
    contract = yaml.safe_load(contract_file.read_text(encoding="utf-8"))
    inputs = {name: _resolve(root, value) for name, value in contract["inputs"].items()}
    implementation = {name: _resolve(root, value) for name, value in contract["implementation"].items()}
    output = _resolve(root, contract["outputs"]["directory"])
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite observation audit: {output}")

    source_manifest = inputs["source_manifest_parquet"]
    source_audit = json.loads(inputs["source_audit_summary_json"].read_text(encoding="utf-8"))
    formula_contract = yaml.safe_load(inputs["v2r_formula_contract"].read_text(encoding="utf-8"))
    formula_audit = json.loads(inputs["v2r_formula_audit"].read_text(encoding="utf-8"))
    source = pd.read_parquet(source_manifest)
    wavelength = [float(value) for value in contract["wavelength"]["centers_nm"]]
    observed_columns = _spectral_columns("observed_reflectance", wavelength)
    scope = contract["scope"]

    required_source_columns = {"subject_id", "capture_id", "split", "expression", "direction", "roi", "model_version", "observation_quantity", "source_spectrum", "hsi_path", "hsi_sha256", "rgb_path", "rgb_sha256", "mask_path", "mask_sha256", "mask_pixel_count", "frozen_transform", "main_observation_gain", "per_spectrum_gain_fitted", "side_gain_applied", "normalization_applied", "input_quality_status", *observed_columns}
    missing = sorted(required_source_columns.difference(source.columns))
    if missing:
        raise ValueError(f"Source manifest is missing columns: {missing}")

    source_hash = sha256_file(source_manifest)
    expected_source_hash = source_audit["outputs"]["input_manifest_parquet"]["sha256"]
    source_rows = source.loc[
        source["split"].eq(scope["split"])
        & source["expression"].eq(scope["expression"])
        & source["direction"].eq(scope["direction"])
        & source["roi"].isin(scope["input_rois"])
    ].copy()
    source_rows = source_rows.sort_values(["subject_id", "capture_id", "roi"]).reset_index(drop=True)
    if len(source_rows) != int(scope["expected_input_region_spectra"]):
        raise ValueError(f"Unexpected source region row count: {len(source_rows)}")
    grouped = source_rows.groupby(["subject_id", "capture_id"], sort=True, dropna=False)
    rows: list[dict[str, Any]] = []
    side_rows: list[dict[str, Any]] = []
    pair_failures: list[str] = []
    metadata_failures: list[str] = []
    paired_equal_fields = (
        "subject_id", "capture_id", "split", "expression", "direction",
        "model_version", "observation_quantity", "source_spectrum", "hsi_path",
        "hsi_sha256", "rgb_path", "rgb_sha256", "frozen_transform",
        "main_observation_gain", "per_spectrum_gain_fitted", "side_gain_applied",
        "normalization_applied", "input_quality_status",
    )
    for (subject_id, capture_id), group in grouped:
        by_roi = {str(row["roi"]): row for _, row in group.iterrows()}
        if set(by_roi) != set(scope["input_rois"]) or len(group) != 2:
            pair_failures.append(f"{subject_id}/{capture_id}")
            continue
        left = by_roi["left_cheek"]
        right = by_roi["right_cheek"]
        metadata_agreement = all(left[field] == right[field] for field in paired_equal_fields)
        if not metadata_agreement:
            metadata_failures.append(f"{subject_id}/{capture_id}")
        left_spectrum = left[observed_columns].to_numpy(dtype=np.float64)
        right_spectrum = right[observed_columns].to_numpy(dtype=np.float64)
        symmetric = symmetric_log_reflectance(left_spectrum, right_spectrum)
        reconstructed = np.exp((np.log(left_spectrum) + np.log(right_spectrum)) / 2.0)
        side_rows.append({
            "subject_id": str(subject_id),
            "capture_id": str(capture_id),
            "left_broadband_mean": float(left_spectrum.mean()),
            "right_broadband_mean": float(right_spectrum.mean()),
            "left_minus_right_broadband": float(left_spectrum.mean() - right_spectrum.mean()),
            "log_left_minus_log_right_mean": float(np.mean(np.log(left_spectrum) - np.log(right_spectrum))),
            "left_brighter_broadband": bool(left_spectrum.mean() > right_spectrum.mean()),
            "paired_capture_metadata_agreement": bool(metadata_agreement),
            "symmetric_min": float(symmetric.min()),
            "symmetric_max": float(symmetric.max()),
            "formula_reconstruction_max_abs": float(np.max(np.abs(symmetric - reconstructed))),
            "fit_band_min": float(symmetric[np.asarray(contract["wavelength"]["fit_index_zero_based"], dtype=int)].min()),
            "fit_band_max": float(symmetric[np.asarray(contract["wavelength"]["fit_index_zero_based"], dtype=int)].max()),
            "edge_band_values": json.dumps({str(int(wavelength[i])): float(symmetric[i]) for i in contract["wavelength"]["edge_diagnostic_index_zero_based"]}, sort_keys=True),
        })
        rows.append({
            "subject_id": str(subject_id),
            "capture_id": str(capture_id),
            "split": str(left["split"]),
            "expression": str(left["expression"]),
            "direction": str(left["direction"]),
            "model_version": "KM2L-HF-v2R",
            "observation_quantity": str(left["observation_quantity"]),
            "source_spectrum": "bilateral_log_geometric_mean_from_v1_raw_float64_region_medians",
            "left_roi": "left_cheek",
            "right_roi": "right_cheek",
            "left_hsi_path": str(left["hsi_path"]),
            "right_hsi_path": str(right["hsi_path"]),
            "left_hsi_sha256": str(left["hsi_sha256"]),
            "right_hsi_sha256": str(right["hsi_sha256"]),
            "left_rgb_path": str(left["rgb_path"]),
            "right_rgb_path": str(right["rgb_path"]),
            "left_rgb_sha256": str(left["rgb_sha256"]),
            "right_rgb_sha256": str(right["rgb_sha256"]),
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
            "aggregation_domain": "natural_log_reflectance",
            "aggregation_epsilon": 0.0,
            "input_quality_status": "PASS",
            "fit_band_index_zero_based": ",".join(str(i) for i in contract["wavelength"]["fit_index_zero_based"]),
            "edge_band_index_zero_based": ",".join(str(i) for i in contract["wavelength"]["edge_diagnostic_index_zero_based"]),
            **{column: float(value) for column, value in zip(observed_columns, symmetric)},
        })

    manifest = pd.DataFrame(rows).sort_values(["subject_id", "capture_id"]).reset_index(drop=True)
    side_audit = pd.DataFrame(side_rows).sort_values(["subject_id", "capture_id"]).reset_index(drop=True)
    checks = {
        "r_a_formula_contract_and_audit_pass": bool(
            formula_contract.get("model_id") == "KM2L-HF-v2R"
            and formula_contract.get("status") == "formula_audit_pass"
            and formula_audit.get("model_id") == "KM2L-HF-v2R"
            and formula_audit.get("status") == "PASS"
            and formula_audit.get("hsi_read") is False
            and sha256_file(inputs["v2r_formula_contract"]) == formula_audit.get("contract_sha256")
        ),
        "source_manifest_hash_matches_v1_audit": source_hash == expected_source_hash and source_audit.get("status") == "PASS_FOR_TRAIN_INVERSION",
        "exact_left_right_pairing": not pair_failures and len(side_audit) == int(scope["expected_output_symmetric_spectra"]),
        "paired_capture_metadata_agreement": not metadata_failures and bool(side_audit["paired_capture_metadata_agreement"].all()),
        "source_rows_are_v1_and_qc_passed": bool(source_rows["model_version"].eq("KM2L-HF-v1").all() and source_rows["input_quality_status"].eq("PASS").all()),
        "subject_and_capture_counts": manifest["subject_id"].nunique() == int(scope["expected_subjects"]) and manifest["capture_id"].nunique() == int(scope["expected_captures"]),
        "no_side_gain_or_normalization": bool((source_rows["side_gain_applied"] == False).all() and (source_rows["per_spectrum_gain_fitted"] == False).all() and source_rows["normalization_applied"].eq("none").all()),
        "strict_positive_log_inputs": bool(np.isfinite(source_rows[observed_columns].to_numpy(dtype=np.float64)).all() and np.all(source_rows[observed_columns].to_numpy(dtype=np.float64) > 0.0)),
        "symmetric_formula_reconstruction": bool(side_audit["formula_reconstruction_max_abs"].max() <= float(contract["audit"]["tolerance_reconstruction_abs"])) if len(side_audit) else False,
        "fit_and_edge_band_roles": (
            [wavelength[index] for index in contract["wavelength"]["fit_index_zero_based"]] == [float(value) for value in contract["wavelength"]["fit_centers_nm"]]
            and [wavelength[index] for index in contract["wavelength"]["edge_diagnostic_index_zero_based"]] == [float(value) for value in contract["wavelength"]["edge_diagnostic_centers_nm"]]
            and sorted(contract["wavelength"]["fit_index_zero_based"] + contract["wavelength"]["edge_diagnostic_index_zero_based"]) == list(range(len(wavelength)))
        ),
        "validation_test_500_isolation": bool(
            source_audit.get("counts", {}).get("validation_hsi_content_reads") == 0
            and source_audit.get("counts", {}).get("test_hsi_content_reads") == 0
            and contract["scope"]["validation_content_allowed"] is False
            and contract["scope"]["test_content_allowed"] is False
            and contract["scope"]["clinical_500_content_allowed"] is False
        ),
        "all_symmetric_values_finite_positive": bool(np.isfinite(manifest[observed_columns].to_numpy(dtype=np.float64)).all() and np.all(manifest[observed_columns].to_numpy(dtype=np.float64) > 0.0)),
    }
    passed = bool(all(checks.values()))
    if not passed:
        raise RuntimeError(f"R-B observation contract failed: {checks}")

    output.mkdir(parents=True, exist_ok=False)
    outputs = contract["outputs"]
    manifest_parquet = output / outputs["symmetric_manifest_parquet"]
    manifest_csv = output / outputs["symmetric_manifest_csv"]
    side_csv = output / outputs["side_pair_audit_csv"]
    manifest.to_parquet(manifest_parquet, index=False)
    manifest.to_csv(manifest_csv, index=False, encoding="utf-8-sig")
    side_audit.to_csv(side_csv, index=False, encoding="utf-8-sig")
    implementation_hashes = {name: sha256_file(path) for name, path in implementation.items()}
    summary = {
        "schema_version": 1,
        "stage": "KM-BIO-v2R-R-B",
        "model_id": contract["model_id"],
        "status": "PASS_FOR_V2R_TRAIN_INVERSION",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "hsi_content_reads": 0,
        "rgb_content_reads": 0,
        "validation_hsi_content_reads": 0,
        "test_hsi_content_reads": 0,
        "clinical_500_content_reads": 0,
        "checks": {key: bool(value) for key, value in checks.items()},
        "counts": {"input_region_spectra": int(len(source_rows)), "subjects": int(manifest["subject_id"].nunique()), "captures": int(manifest["capture_id"].nunique()), "symmetric_spectra": int(len(manifest))},
        "wavelength": {"full_centers_nm": wavelength, "fit_centers_nm": contract["wavelength"]["fit_centers_nm"], "edge_diagnostic_centers_nm": contract["wavelength"]["edge_diagnostic_centers_nm"]},
        "formula": contract["observation"]["formula"],
        "side_audit": {"left_brighter_broadband_subject_count": int(side_audit["left_brighter_broadband"].sum()), "median_left_minus_right_broadband": float(side_audit["left_minus_right_broadband"].median()), "median_log_left_minus_log_right_mean": float(side_audit["log_left_minus_log_right_mean"].median()), "maximum_formula_reconstruction_abs": float(side_audit["formula_reconstruction_max_abs"].max())},
        "source_hashes": {**{name: sha256_file(path) for name, path in inputs.items()}, "v2r_observation_contract": sha256_file(contract_file), **{f"implementation_{name}": value for name, value in implementation_hashes.items()}},
        "outputs": {
            "symmetric_manifest_parquet": {"path": str(manifest_parquet), "sha256": sha256_file(manifest_parquet)},
            "symmetric_manifest_csv": {"path": str(manifest_csv), "sha256": sha256_file(manifest_csv)},
            "side_pair_audit_csv": {"path": str(side_csv), "sha256": sha256_file(side_csv)},
        },
        "authorized_next_stage": "R-C0_CANDIDATE_LADDER",
        "r_c0_train_inversion_allowed": True,
        "real_hsi_inversion_performed_in_r_b": False,
    }
    summary_json = output / outputs["audit_summary_json"]
    summary_md = output / outputs["audit_summary_markdown"]
    summary_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    summary_md.write_text("# KM-BIO-v2R R-B observation audit\n\nNo raw HSI/RGB/Validation/Test/clinical content was read.\n\n```json\n" + json.dumps(summary, ensure_ascii=False, indent=2) + "\n```\n", encoding="utf-8")
    artifact_rows = [
        {"role": "source", "name": name, "path": str(path), "sha256": sha256_file(path)}
        for name, path in {**inputs, "v2r_observation_contract": contract_file, **{f"implementation_{name}": path for name, path in implementation.items()}}.items()
    ]
    artifact_rows.extend([
        {"role": "output", "name": name, "path": value["path"], "sha256": value["sha256"]}
        for name, value in summary["outputs"].items()
    ])
    artifact_rows.extend([
        {"role": "output", "name": "audit_summary_json", "path": str(summary_json), "sha256": sha256_file(summary_json)},
        {"role": "output", "name": "audit_summary_markdown", "path": str(summary_md), "sha256": sha256_file(summary_md)},
    ])
    pd.DataFrame(artifact_rows).to_csv(output / outputs["artifact_hash_manifest_csv"], index=False, encoding="utf-8-sig")
    return summary
