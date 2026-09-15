"""S1-3: extract auditable regional spectra and empirical band uncertainty."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from .data_contracts import load_binary_mask, load_hyperskin_cube


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "skin_optics_hsi" / "s1_3_region_spectra_v1.yaml"
DEFAULT_TARGET_DECISION_NAME = "s1_2_target_domain_final_decision.json"


def sha256_file(path: str | Path, chunk_bytes: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while chunk := stream.read(chunk_bytes):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def _column(prefix: str, wavelength_nm: float) -> str:
    value = int(wavelength_nm) if float(wavelength_nm).is_integer() else wavelength_nm
    return f"{prefix}_{value}nm"


def _trimmed_mean(values: np.ndarray, trim_fraction: float) -> np.ndarray:
    if not 0.0 <= trim_fraction < 0.5:
        raise ValueError("trim_fraction_each_tail must be in [0, 0.5)")
    if trim_fraction == 0.0 or values.shape[0] < 3:
        return values.mean(axis=0)
    count = int(np.floor(values.shape[0] * trim_fraction))
    if count == 0:
        return values.mean(axis=0)
    ordered = np.sort(values, axis=0)
    return ordered[count : values.shape[0] - count].mean(axis=0)


def summarize_region(values: np.ndarray, trim_fraction: float) -> dict[str, np.ndarray]:
    """Compute full-pixel robust summaries for a [pixels, bands] region."""

    spectra = np.asarray(values, dtype=np.float32)
    if spectra.ndim != 2 or spectra.shape[0] == 0:
        raise ValueError("Region spectra must be a non-empty [pixels, bands] array")
    if not np.isfinite(spectra).all():
        raise ValueError("Region contains non-finite reflectance values")
    median = np.median(spectra, axis=0)
    q25, q75 = np.quantile(spectra, [0.25, 0.75], axis=0)
    return {
        "median": median,
        "mad": np.median(np.abs(spectra - median[None, :]), axis=0),
        "iqr": q75 - q25,
        "mean": spectra.mean(axis=0),
        "trimmed_mean": _trimmed_mean(spectra, trim_fraction),
    }


def _validate_wavelength_contract(contract: dict[str, Any], config: dict[str, Any]) -> np.ndarray:
    wavelength = contract["wavelength"]
    centers = wavelength["centers_nm"]
    expected_status = config["wavelength_contract"]["required_centers_status"]
    if centers.get("status") != expected_status:
        raise ValueError(f"Wavelength centers are not frozen as {expected_status}")
    values = np.asarray(centers["value"], dtype=np.float64)
    if values.shape != (31,) or not np.all(np.diff(values) > 0):
        raise ValueError("Expected 31 strictly increasing VIS wavelength centers")
    if wavelength.get("ordering") != config["wavelength_contract"]["required_ordering"]:
        raise ValueError("Wavelength ordering does not match the S1-3 contract")
    release_srf = wavelength["release_effective_bandwidth_or_srf"]
    if release_srf.get("status") == "missing" and not config["wavelength_contract"].get(
        "allow_missing_effective_srf", False
    ):
        raise ValueError("Effective release SRF is missing and the S1-3 policy does not allow it")
    return values


def _validate_and_join_inputs(
    contract_path: Path,
    mask_manifest_path: Path,
    target_decision_path: Path,
    config: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any], pd.DataFrame, pd.DataFrame]:
    contract = _read_json(contract_path)
    target = _read_json(target_decision_path)
    scope = config["scope"]
    if contract.get("status") != "PASS":
        raise ValueError("S1-0 data contract is not PASS")
    if target.get("status") != scope["required_target_decision_status"]:
        raise ValueError("S1-2 target-domain decision is not approved for development")
    if not target.get("next_stage_allowed") or target.get("authorized_next_stage") != scope["required_authorized_stage"]:
        raise ValueError("S1-2 decision does not authorize S1-3")
    if int(target.get("test_access_count", -1)) != 0:
        raise ValueError("S1-2 provenance reports Test access")
    if Path(target["manifest_path"]).resolve() != mask_manifest_path.resolve():
        raise ValueError("Passed mask manifest is not the S1-2 finalized manifest")
    if sha256_file(mask_manifest_path) != target["manifest_sha256"]:
        raise ValueError("Final mask manifest hash differs from the S1-2 decision")

    mask_manifest = pd.read_parquet(mask_manifest_path)
    allowed = set(scope["allowed_splits"])
    forbidden = set(scope["forbidden_splits"])
    observed = set(mask_manifest["split"].astype(str).unique())
    if observed & forbidden or not observed <= allowed:
        raise ValueError(f"S1-3 manifest contains forbidden or unknown splits: {sorted(observed)}")

    split_manifest_path = Path(contract["manifest"]["path"]).resolve()
    if sha256_file(split_manifest_path) != contract["manifest"]["sha256"]:
        raise ValueError("S1-0 split manifest hash differs from the data contract")
    split_manifest = pd.read_csv(split_manifest_path)
    split_manifest = split_manifest.loc[split_manifest["split"].isin(allowed)].copy()
    if split_manifest.duplicated(["sample_id", "split"]).any() or mask_manifest.duplicated(
        ["sample_id", "split"]
    ).any():
        raise ValueError("Duplicate sample/split keys prevent a one-to-one S1-3 join")
    joined = mask_manifest.merge(
        split_manifest[["sample_id", "split", "hsi_path", "hsi_sha256"]],
        on=["sample_id", "split"],
        how="left",
        validate="one_to_one",
    )
    if joined["hsi_path"].isna().any():
        raise ValueError("Some S1-2 rows have no matching S1-0 HSI record")
    return contract, target, joined, split_manifest


def _empty_spectral_fields(wavelength_nm: np.ndarray) -> dict[str, float]:
    fields: dict[str, float] = {}
    for wavelength in wavelength_nm:
        for prefix in ("reflectance_median", "reflectance_mad", "reflectance_iqr", "reflectance_mean", "reflectance_trimmed_mean"):
            fields[_column(prefix, wavelength)] = np.nan
    return fields


def _spectral_fields(summary: dict[str, np.ndarray], wavelength_nm: np.ndarray) -> dict[str, float]:
    fields: dict[str, float] = {}
    mapping = {
        "reflectance_median": "median",
        "reflectance_mad": "mad",
        "reflectance_iqr": "iqr",
        "reflectance_mean": "mean",
        "reflectance_trimmed_mean": "trimmed_mean",
    }
    for prefix, key in mapping.items():
        for index, wavelength in enumerate(wavelength_nm):
            fields[_column(prefix, wavelength)] = float(summary[key][index])
    return fields


def _build_band_qc(frame: pd.DataFrame, wavelength_nm: np.ndarray) -> pd.DataFrame:
    usable = frame.loc[frame["extraction_status"].eq("USABLE")].copy()
    rows: list[dict[str, Any]] = []
    group_columns = ["split", "s1_2_analysis_role", "region"]
    for keys, group in usable.groupby(group_columns, dropna=False, sort=True):
        for wavelength in wavelength_nm:
            medians = group[_column("reflectance_median", wavelength)].to_numpy(dtype=np.float64)
            within_mad = group[_column("reflectance_mad", wavelength)].to_numpy(dtype=np.float64)
            within_iqr = group[_column("reflectance_iqr", wavelength)].to_numpy(dtype=np.float64)
            rows.append(
                {
                    **dict(zip(group_columns, keys)),
                    "wavelength_nm": float(wavelength),
                    "n_spectra": int(len(group)),
                    "finite_fraction": float(np.isfinite(medians).mean()),
                    "cohort_median_reflectance": float(np.median(medians)),
                    "cohort_iqr_reflectance": float(np.subtract(*np.quantile(medians, [0.75, 0.25]))),
                    "median_within_region_mad": float(np.median(within_mad)),
                    "median_within_region_iqr": float(np.median(within_iqr)),
                }
            )
    return pd.DataFrame(rows)


def _build_subject_condition_variability(frame: pd.DataFrame, wavelength_nm: np.ndarray) -> pd.DataFrame:
    usable = frame.loc[frame["extraction_status"].eq("USABLE")].copy()
    rows: list[dict[str, Any]] = []
    for keys, group in usable.groupby(["split", "subject_id", "region"], sort=True):
        row: dict[str, Any] = {
            "split": keys[0],
            "subject_id": keys[1],
            "region": keys[2],
            "n_acquisition_conditions": int(len(group)),
            "interpretation": "condition_variability_not_technical_repeatability",
        }
        for wavelength in wavelength_nm:
            values = group[_column("reflectance_median", wavelength)].to_numpy(dtype=np.float64)
            center = np.median(values)
            row[_column("condition_mad", wavelength)] = float(np.median(np.abs(values - center)))
            row[_column("condition_iqr", wavelength)] = float(np.subtract(*np.quantile(values, [0.75, 0.25])))
        rows.append(row)
    return pd.DataFrame(rows)


def extract_s1_3_region_spectra(
    contract_path: str | Path,
    mask_manifest_path: str | Path,
    output_root: str | Path,
    *,
    target_decision_path: str | Path | None = None,
    registration_decision_path: str | Path | None = None,
    config_path: str | Path = DEFAULT_CONFIG,
    progress: bool = True,
) -> dict[str, Any]:
    """Extract Train/Validation region spectra. Test rows are rejected before any HSI read."""

    contract_file = Path(contract_path).resolve()
    mask_manifest_file = Path(mask_manifest_path).resolve()
    output = Path(output_root).resolve()
    config_file = Path(config_path).resolve()
    project_stage_root = mask_manifest_file.parents[1]
    target_file = (
        Path(target_decision_path).resolve()
        if target_decision_path
        else project_stage_root / "freeze" / DEFAULT_TARGET_DECISION_NAME
    )
    registration_file = (
        Path(registration_decision_path).resolve()
        if registration_decision_path
        else project_stage_root / "registration_qc" / "registration_decision.json"
    )
    planned = [
        output / "region_spectra.parquet",
        output / "region_spectra.csv",
        output / "band_qc.parquet",
        output / "band_qc.csv",
        output / "subject_condition_variability.parquet",
        output / "subject_condition_variability.csv",
        output / "s1_3_decision.json",
    ]
    existing = [path for path in planned if path.exists()]
    if existing:
        raise FileExistsError(f"Refusing to overwrite existing S1-3 artifacts: {existing}")
    config = yaml.safe_load(config_file.read_text(encoding="utf-8"))
    contract, target, joined, _ = _validate_and_join_inputs(
        contract_file, mask_manifest_file, target_file, config
    )
    wavelength_nm = _validate_wavelength_contract(contract, config)
    registration = _read_json(registration_file)
    if registration.get("status") != "PASS" or registration.get("frozen_transform") != "transpose":
        raise ValueError("S1-3 requires the frozen S1-1 transpose transform")

    trim_fraction = float(config["statistics"]["trim_fraction_each_tail"])
    clip_min = float(config["statistics"]["derived_clip_min"])
    clip_max = float(config["statistics"]["derived_clip_max"])
    regions = list(config["regions"])
    rows: list[dict[str, Any]] = []
    test_access_count = 0
    for sample_index, (_, sample) in enumerate(joined.iterrows(), start=1):
        if str(sample["split"]) == "test":
            test_access_count += 1
            raise RuntimeError("S1-3 attempted to access Test")
        cube = load_hyperskin_cube(
            sample["hsi_path"],
            dataset_key=contract["hsi_storage"]["dataset_key"],
            expected_bands=len(wavelength_nm),
        )
        cube = np.transpose(cube, (1, 0, 2))
        derived_cube = np.clip(cube, clip_min, clip_max)
        for region in regions:
            mask_path_value = sample.get(f"{region}_path")
            base: dict[str, Any] = {
                "sample_id": str(sample["sample_id"]),
                "subject_id": str(sample["subject_id"]),
                "split": str(sample["split"]),
                "expression": str(sample["expression"]),
                "direction": str(sample["direction"]),
                "region": region,
                "s1_2_status": str(sample["status"]),
                "s1_2_failure_codes": "" if pd.isna(sample["failure_codes"]) else str(sample["failure_codes"]),
                "s1_2_review_codes": "" if pd.isna(sample["review_codes"]) else str(sample["review_codes"]),
                "s1_2_analysis_role": str(sample["s1_2_analysis_role"]),
                "s1_2_usable_flag": bool(sample["s1_2_usable_flag"]),
                "mask_path": "" if pd.isna(mask_path_value) else str(mask_path_value),
                "mask_sha256": "",
                "hsi_path": str(sample["hsi_path"]),
                "hsi_sha256": str(sample["hsi_sha256"]),
                "effective_pixels": 0,
                "effective_fraction_of_image": 0.0,
                "effective_fraction_of_anatomical_skin": 0.0,
                "radiometric_exclusion_fraction": float(sample["radiometric_exclusion_fraction"]),
                "illumination_exclusion_fraction": float(sample["illumination_exclusion_fraction"]),
                "registration_status": str(registration["status"]),
                "frozen_transform": str(registration["frozen_transform"]),
                "extraction_status": "UNAVAILABLE",
                "extraction_reason": "",
            }
            base.update(_empty_spectral_fields(wavelength_nm))
            if str(sample["status"]) == "FAIL" or not bool(sample["s1_2_usable_flag"]):
                base["extraction_reason"] = "s1_2_sample_not_usable"
                rows.append(base)
                continue
            if not mask_path_value or pd.isna(mask_path_value):
                base["extraction_reason"] = "mask_path_missing"
                rows.append(base)
                continue
            mask_path = Path(str(mask_path_value)).resolve()
            mask = load_binary_mask(mask_path, tuple(cube.shape[:2]))
            pixel_count = int(mask.sum())
            base["mask_sha256"] = sha256_file(mask_path)
            base["effective_pixels"] = pixel_count
            base["effective_fraction_of_image"] = float(pixel_count / (cube.shape[0] * cube.shape[1]))
            base["effective_fraction_of_anatomical_skin"] = float(
                pixel_count / max(int(sample["anatomical_skin_pixels"]), 1)
            )
            if pixel_count == 0:
                base["extraction_reason"] = "empty_region_by_frozen_mask_rule"
                rows.append(base)
                continue
            summary = summarize_region(derived_cube[mask], trim_fraction)
            base.update(_spectral_fields(summary, wavelength_nm))
            base["extraction_status"] = "USABLE"
            rows.append(base)
        if progress and (sample_index == 1 or sample_index % 10 == 0 or sample_index == len(joined)):
            print(f"S1-3 extracted {sample_index}/{len(joined)} samples", flush=True)

    frame = pd.DataFrame(rows)
    band_qc = _build_band_qc(frame, wavelength_nm)
    condition_variability = _build_subject_condition_variability(frame, wavelength_nm)
    primary_roles = set(config["gate"]["primary_roles"])
    required_regions = set(config["gate"]["required_primary_regions"])
    primary = frame.loc[
        frame["s1_2_analysis_role"].isin(primary_roles) & frame["region"].isin(required_regions)
    ].copy()
    spectral_columns = [_column("reflectance_median", wavelength) for wavelength in wavelength_nm]
    automatic_checks = {
        "test_access_count_zero": test_access_count == 0,
        "manifest_contains_only_train_and_valid": set(frame["split"].unique()) <= {"train", "valid"},
        "all_primary_cheek_rows_usable": bool(primary["extraction_status"].eq("USABLE").all()),
        "all_usable_medians_finite": bool(
            np.isfinite(frame.loc[frame["extraction_status"].eq("USABLE"), spectral_columns].to_numpy()).all()
        ),
        "wavelength_centers_confirmed": contract["wavelength"]["centers_nm"]["status"]
        == "confirmed_from_official_code",
        "effective_srf_missing_retained": contract["wavelength"]["release_effective_bandwidth_or_srf"]["status"]
        == "missing",
    }
    passed = all(automatic_checks.values())
    output.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(planned[0], index=False)
    frame.to_csv(planned[1], index=False, encoding="utf-8-sig")
    band_qc.to_parquet(planned[2], index=False)
    band_qc.to_csv(planned[3], index=False, encoding="utf-8-sig")
    condition_variability.to_parquet(planned[4], index=False)
    condition_variability.to_csv(planned[5], index=False, encoding="utf-8-sig")
    decision = {
        "schema_version": 1,
        "stage": "S1-3",
        "status": "PASS_FOR_S1_4" if passed else "REVISE",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "automatic_checks": automatic_checks,
        "sample_count": int(len(joined)),
        "row_count": int(len(frame)),
        "usable_row_count": int(frame["extraction_status"].eq("USABLE").sum()),
        "unavailable_row_count": int(frame["extraction_status"].ne("USABLE").sum()),
        "primary_cheek_row_count": int(len(primary)),
        "primary_cheek_usable_count": int(primary["extraction_status"].eq("USABLE").sum()),
        "wavelength_nm": [float(value) for value in wavelength_nm],
        "wavelength_status": contract["wavelength"]["centers_nm"]["status"],
        "effective_srf_status": contract["wavelength"]["release_effective_bandwidth_or_srf"]["status"],
        "effective_srf_policy": "missing_allowed_for_proxy_development_requires_sensitivity_before_formal_test",
        "contract_path": str(contract_file),
        "contract_sha256": sha256_file(contract_file),
        "mask_manifest_path": str(mask_manifest_file),
        "mask_manifest_sha256": sha256_file(mask_manifest_file),
        "target_decision_path": str(target_file),
        "target_decision_sha256": sha256_file(target_file),
        "registration_decision_path": str(registration_file),
        "registration_decision_sha256": sha256_file(registration_file),
        "config_path": str(config_file),
        "config_sha256": sha256_file(config_file),
        "outputs": {
            "region_spectra_parquet": {"path": str(planned[0]), "sha256": sha256_file(planned[0])},
            "region_spectra_csv": {"path": str(planned[1]), "sha256": sha256_file(planned[1])},
            "band_qc_parquet": {"path": str(planned[2]), "sha256": sha256_file(planned[2])},
            "band_qc_csv": {"path": str(planned[3]), "sha256": sha256_file(planned[3])},
            "condition_variability_parquet": {"path": str(planned[4]), "sha256": sha256_file(planned[4])},
            "condition_variability_csv": {"path": str(planned[5]), "sha256": sha256_file(planned[5])},
        },
        "next_stage_allowed": bool(passed),
        "authorized_next_stage": "S1-4" if passed else None,
        "test_access_count": test_access_count,
    }
    _write_json(planned[6], decision)
    return decision
