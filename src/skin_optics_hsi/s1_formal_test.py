"""Single-use S1-7 Test inference and Stage-1 representation decision."""

from __future__ import annotations

import json
import platform
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping

import numpy as np
import pandas as pd
import scipy
import yaml
from PIL import Image

from .data_contracts import load_hyperskin_cube
from .metrics import spectral_angle_rad
from .s1_masks import (
    _save_contact_sheet,
    _save_qc_panel,
    build_illumination_layer,
    build_radiometric_layer,
    build_roi_geometry,
    build_semantic_layers,
    evaluate_sample_qc,
)
from .s1_proxy_inverse import fit_proxy_wls
from .s1_region_spectra import summarize_region
from .s1_revised_forward import file_sha256, load_revised_registry


PROJECT_ROOT = Path(__file__).resolve().parents[2]
TEST_RGB_WORKER = PROJECT_ROOT / "scripts" / "skin_optics_hsi" / "s1_7_mask_rgb_worker.py"
FROZEN_MASK_IMPLEMENTATION_PATHS = {
    "s1_masks.py": PROJECT_ROOT / "src" / "skin_optics_hsi" / "s1_masks.py",
    "s1_mask_rgb_worker.py": PROJECT_ROOT / "scripts" / "skin_optics_hsi" / "s1_mask_rgb_worker.py",
    "build_hyperskin_masks.py": PROJECT_ROOT / "scripts" / "skin_optics_hsi" / "build_hyperskin_masks.py",
    "finalize_hyperskin_masks.py": PROJECT_ROOT / "scripts" / "skin_optics_hsi" / "finalize_hyperskin_masks.py",
}
WAVELENGTHS = np.arange(400.0, 701.0, 10.0)
REGIONS = ("whole_skin", "left_cheek", "right_cheek", "forehead")


def _resolve(value: str | Path) -> Path:
    path = Path(value)
    return (path if path.is_absolute() else PROJECT_ROOT / path).resolve()


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def _side(region: str) -> str:
    if region == "left_cheek":
        return "image_left"
    if region == "right_cheek":
        return "image_right"
    return "other"


def _weighted_center(values: np.ndarray, weights: np.ndarray) -> np.ndarray:
    return values - np.sum(values * weights) / np.sum(weights)


def _empty_spectral_fields() -> dict[str, float]:
    result: dict[str, float] = {}
    for statistic in ("median", "mean", "trimmed_mean", "mad", "iqr"):
        for wavelength in WAVELENGTHS:
            result[f"reflectance_{statistic}_{int(wavelength)}nm"] = float("nan")
    return result


def _summary_fields(summary: Mapping[str, np.ndarray]) -> dict[str, float]:
    result: dict[str, float] = {}
    for statistic, values in summary.items():
        for wavelength, value in zip(WAVELENGTHS, np.asarray(values), strict=True):
            result[f"reflectance_{statistic}_{int(wavelength)}nm"] = float(value)
    return result


def _assert_no_prior_test_output(output: Path, refuse_sibling: bool) -> None:
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite formal S1-7 output: {output}")
    if refuse_sibling and output.parent.exists():
        siblings = sorted(path for path in output.parent.glob("s1_7_*") if path.exists())
        if siblings:
            raise FileExistsError(f"A prior formal S1-7 output already exists: {siblings}")


def _validate_frozen_inputs(config_file: Path, config: Mapping[str, Any], output: Path) -> dict[str, Any]:
    if config.get("stage") != "S1-7" or config.get("run_id") != "s1_7_v1":
        raise ValueError("Unexpected S1-7 configuration identity")
    access = config.get("access_policy", {})
    if (
        not access.get("test_rgb_access_allowed_after_lock")
        or not access.get("test_hsi_access_allowed_after_lock")
        or access.get("train_access_allowed")
        or access.get("validation_access_allowed")
        or access.get("global_refit_allowed")
        or access.get("model_reselection_allowed")
        or int(access.get("maximum_formal_runs", 0)) != 1
    ):
        raise ValueError("Invalid S1-7 access policy")
    gates = config.get("decision_gates", {})
    required_gates = {
        "minimum_primary_region_fraction",
        "minimum_primary_regions_per_subject",
        "improved_subject_fraction_vs_B0S_min",
        "median_shape_log_rmse_ratio_vs_B0S_max",
        "max_absolute_median_band_shape_residual",
        "maximum_selected_model_fit_failure_fraction",
        "require_finite_parameters",
        "require_full_design_rank",
    }
    if set(gates) != required_gates or any(value is None for value in gates.values()):
        raise ValueError("S1-7 gates must be complete and non-null before Test access")
    _assert_no_prior_test_output(output, bool(config["output_policy"]["refuse_any_prior_s1_7_sibling"]))

    inputs = {name: _resolve(value) for name, value in config["inputs"].items()}
    for name, path in inputs.items():
        if not path.is_file():
            raise FileNotFoundError(f"Missing frozen S1-7 input {name}: {path}")
    contract = _read_json(inputs["data_contract"])
    registration = _read_json(inputs["registration_decision"])
    provenance = _read_json(inputs["frozen_mask_provenance"])
    amendment = _read_json(inputs["wavelength_evidence_amendment"])
    target = _read_json(inputs["target_domain_decision"])
    s1_6 = _read_json(inputs["s1_6_decision"])
    spec = yaml.safe_load(inputs["frozen_stage1_spec"].read_text(encoding="utf-8"))
    if contract.get("status") != "PASS":
        raise RuntimeError("S1-0 data contract is not PASS")
    if registration.get("status") != "PASS" or registration.get("frozen_transform") != "transpose":
        raise RuntimeError("S1-1 frozen spatial transform is invalid")
    if provenance.get("status") != "FROZEN" or int(provenance.get("test_access_count", -1)) != 0:
        raise RuntimeError("S1-2 mask protocol provenance is not an untouched freeze")
    if target.get("status") != "PASS_FOR_DEVELOPMENT" or int(target.get("test_access_count", -1)) != 0:
        raise RuntimeError("S1-2 target-domain decision is invalid")
    if (
        s1_6.get("status") != "S1_6_FROZEN_FOR_S1_7"
        or not s1_6.get("next_stage_allowed")
        or s1_6.get("authorized_next_stage") != "S1-7"
        or int(s1_6.get("formal_test_run_limit", 0)) != 1
        or int(s1_6.get("test_rows_read", -1)) != 0
    ):
        raise RuntimeError("S1-6 does not authorize an untouched single S1-7 run")
    if spec.get("status") != "FROZEN_FOR_SINGLE_S1_7_TEST" or spec["formal_test_policy"]["maximum_formal_runs"] != 1:
        raise RuntimeError("Frozen Stage-1 specification is invalid")
    selected = str(config["models"]["selected"])
    if selected != s1_6.get("selected_representation_model") or selected != spec.get("selected_representation_model"):
        raise RuntimeError("S1-7 selected model differs from the S1-6 freeze")
    if config["extraction"]["transform"] != registration["frozen_transform"]:
        raise RuntimeError("S1-7 transform differs from the S1-1 freeze")
    if list(map(float, spec["wavelength_contract"]["centers_nm"])) != WAVELENGTHS.tolist():
        raise RuntimeError("Frozen wavelength centers differ from the official 31-band contract")

    expected_hashes = s1_6["outputs"]
    s1_6_root = inputs["s1_6_decision"].parent
    for name, expected_hash in expected_hashes.items():
        frozen_output = s1_6_root / name
        if not frozen_output.is_file() or file_sha256(frozen_output) != expected_hash:
            raise RuntimeError(f"S1-6 frozen output hash changed: {name}")
    if expected_hashes.get("frozen_stage1_spec.yaml") != file_sha256(inputs["frozen_stage1_spec"]):
        raise RuntimeError("Frozen Stage-1 specification hash changed")
    if expected_hashes.get("frozen_calibration.npz") != file_sha256(inputs["frozen_calibration"]):
        raise RuntimeError("Frozen calibration hash changed")
    if spec["calibration"]["sha256"] != file_sha256(inputs["frozen_calibration"]):
        raise RuntimeError("Calibration hash disagrees with frozen specification")
    registry_path = Path(spec["model_registry"]["path"]).resolve()
    if not registry_path.is_file() or spec["model_registry"]["sha256"] != file_sha256(registry_path):
        raise RuntimeError("Frozen model registry hash changed")
    if provenance["config_sha256"] != file_sha256(inputs["frozen_mask_protocol"]):
        raise RuntimeError("Frozen mask protocol hash changed")
    current_contract_hash = file_sha256(inputs["data_contract"])
    if (
        amendment.get("stage") != "S1-0_evidence_amendment"
        or amendment.get("previous_contract_sha256") != provenance["contract_sha256"]
        or amendment.get("updated_contract_sha256") != current_contract_hash
        or amendment.get("effective_srf_status") != "missing"
        or list(map(float, amendment.get("confirmed_centers_nm", []))) != WAVELENGTHS.tolist()
        or amendment.get("confirmed_ordering") != "ascending"
    ):
        raise RuntimeError("The post-S1-2 data-contract evidence amendment is invalid")
    if provenance["registration_sha256"] != file_sha256(inputs["registration_decision"]):
        raise RuntimeError("Registration hash differs from the frozen mask provenance")
    protocol = yaml.safe_load(inputs["frozen_mask_protocol"].read_text(encoding="utf-8"))
    checkpoint = Path(protocol["runtime"]["parser_checkpoint"])
    checkpoint = (checkpoint if checkpoint.is_absolute() else PROJECT_ROOT / checkpoint).resolve()
    if provenance["parser_checkpoint_sha256"] != file_sha256(checkpoint):
        raise RuntimeError("Frozen parser checkpoint hash changed")
    current_mask_hashes = {name: file_sha256(path) for name, path in FROZEN_MASK_IMPLEMENTATION_PATHS.items()}
    if provenance["implementation_sha256"] != current_mask_hashes:
        raise RuntimeError("S1-2 mask implementation differs from the frozen Train implementation")
    for name, record in spec["upstream_decisions"].items():
        path = Path(record["path"]).resolve()
        if not path.is_file() or file_sha256(path) != record["sha256"]:
            raise RuntimeError(f"Frozen upstream decision hash changed: {name}")
    manifest_path = Path(contract["manifest"]["path"]).resolve()
    if file_sha256(manifest_path) != contract["manifest"]["sha256"]:
        raise RuntimeError("S1-0 split manifest hash changed")
    manifest = pd.read_csv(manifest_path)
    test = manifest.loc[manifest["split"].astype(str) == str(config["cohort"]["split"])].copy()
    observed = (len(test), test["subject_id"].nunique())
    expected = (int(config["cohort"]["expected_samples"]), int(config["cohort"]["expected_subjects"]))
    if observed != expected:
        raise RuntimeError(f"Formal Test cohort mismatch: observed={observed}, expected={expected}")
    if set(test["expression"].astype(str)) != {"neutral", "smile"} or set(test["direction"].astype(str)) != {"front", "left", "right"}:
        raise RuntimeError("Formal Test strata are incomplete")
    return {
        "inputs": inputs,
        "contract": contract,
        "registration": registration,
        "provenance": provenance,
        "amendment": amendment,
        "target": target,
        "s1_6": s1_6,
        "spec": spec,
        "registry_path": registry_path,
        "checkpoint": checkpoint,
        "manifest_path": manifest_path,
        "test_manifest": test.sort_values("sample_id").reset_index(drop=True),
        "config_sha256": file_sha256(config_file),
    }


def _region_is_usable(region: str, direction: str, pixels: int, protocol: Mapping[str, Any]) -> tuple[bool, str]:
    minimum = {
        "whole_skin": int(protocol["roi"]["minimum_whole_skin_pixels"]),
        "left_cheek": int(protocol["roi"]["minimum_cheek_pixels"]),
        "right_cheek": int(protocol["roi"]["minimum_cheek_pixels"]),
        "forehead": int(protocol["roi"]["minimum_forehead_pixels"]),
    }[region]
    if region == "left_cheek" and direction == "left":
        return False, "occluded_side_by_frozen_pose_rule"
    if region == "right_cheek" and direction == "right":
        return False, "occluded_side_by_frozen_pose_rule"
    if pixels < minimum:
        return False, "below_frozen_region_pixel_threshold"
    return True, ""


def _build_test_masks_and_spectra(
    state: Mapping[str, Any],
    config: Mapping[str, Any],
    output: Path,
    progress: Callable[[str], None],
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    inputs = state["inputs"]
    protocol = yaml.safe_load(inputs["frozen_mask_protocol"].read_text(encoding="utf-8"))
    mask_root = output / "masks"
    command = [
        str(protocol["runtime"]["rgb_worker_python"]),
        str(TEST_RGB_WORKER),
        "--manifest", str(state["manifest_path"]),
        "--protocol", str(inputs["frozen_mask_protocol"]),
        "--output-root", str(mask_root),
    ]
    completed = subprocess.run(command, cwd=PROJECT_ROOT, text=True, capture_output=True, check=False)
    (output / "rgb_worker_test.log").write_text(
        completed.stdout + "\nSTDERR\n" + completed.stderr, encoding="utf-8"
    )
    if completed.returncode != 0:
        raise RuntimeError(f"S1-7 RGB worker failed with exit code {completed.returncode}")
    worker = pd.read_csv(mask_root / "rgb_worker_test.csv").set_index("sample_id")
    mask_rows: list[dict[str, Any]] = []
    spectrum_rows: list[dict[str, Any]] = []
    clip_low, clip_high = map(float, config["extraction"]["derived_clip_range"])
    trim_fraction = float(config["extraction"]["trim_fraction"])
    test_manifest = state["test_manifest"]
    for index, sample in enumerate(test_manifest.to_dict("records"), start=1):
        sample_id = str(sample["sample_id"])
        sample_dir = mask_root / "test" / sample_id
        primary = (
            sample["expression"] == config["cohort"]["primary_expression"]
            and sample["direction"] == config["cohort"]["primary_direction"]
        )
        base = {
            "sample_id": sample_id,
            "subject_id": str(sample["subject_id"]),
            "split": "test",
            "expression": str(sample["expression"]),
            "direction": str(sample["direction"]),
            "analysis_role": "primary_test" if primary else "stress_test",
            "status": "FAIL",
            "failure_codes": "",
            "review_codes": "",
            "qc_panel_path": "",
        }
        for name in ("anatomical_skin", "semantic_valid", "radiometric_valid", "illumination_valid", *REGIONS):
            base[f"{name}_path"] = ""
            base[f"{name}_sha256"] = ""
        region_arrays: dict[str, np.ndarray] = {}
        derived_cube: np.ndarray | None = None
        try:
            if sample_id not in worker.index or str(worker.loc[sample_id, "status"]) != "PASS":
                detail = "missing worker row" if sample_id not in worker.index else str(worker.loc[sample_id, "failure_detail"])
                raise RuntimeError(f"rgb_worker_failed:{detail}")
            with Image.open(sample["rgb_path"]) as image:
                rgb = np.asarray(image.convert("RGB"), dtype=np.uint8)
            labels = np.load(sample_dir / "parsing_label.npy", allow_pickle=False)
            landmarks = np.load(sample_dir / "landmarks.npy", allow_pickle=False)
            anatomical, semantic_valid, semantic_parts = build_semantic_layers(labels, landmarks, protocol)
            cube = load_hyperskin_cube(
                sample["hsi_path"],
                dataset_key=state["contract"]["hsi_storage"]["dataset_key"],
                expected_bands=31,
            )
            derived_cube = np.clip(np.transpose(cube, (1, 0, 2)), clip_low, clip_high)
            if derived_cube.shape[:2] != rgb.shape[:2]:
                raise ValueError(f"aligned_cube_shape_mismatch:{derived_cube.shape}:{rgb.shape}")
            radiometric_valid, radiometric_parts = build_radiometric_layer(derived_cube, protocol)
            illumination_valid, illumination_parts = build_illumination_layer(
                rgb, radiometric_parts["broadband"], semantic_valid & radiometric_valid, protocol
            )
            final_skin = anatomical & semantic_valid & radiometric_valid & illumination_valid
            geometry = build_roi_geometry(rgb.shape[:2], landmarks, protocol)
            region_arrays = {
                "whole_skin": final_skin,
                "left_cheek": final_skin & geometry["left_cheek_geometry"],
                "right_cheek": final_skin & geometry["right_cheek_geometry"],
                "forehead": final_skin & geometry["forehead_geometry"],
            }
            if sample["direction"] == "left":
                region_arrays["left_cheek"] = np.zeros_like(final_skin)
            elif sample["direction"] == "right":
                region_arrays["right_cheek"] = np.zeros_like(final_skin)
            qc = evaluate_sample_qc(
                str(sample["direction"]),
                anatomical,
                semantic_valid,
                radiometric_valid,
                illumination_valid,
                region_arrays,
                protocol,
            )
            arrays = {
                "anatomical_skin": anatomical,
                "semantic_valid": semantic_valid,
                "radiometric_valid": radiometric_valid,
                "illumination_valid": illumination_valid,
                **region_arrays,
            }
            for name, array in arrays.items():
                path = sample_dir / f"{name}.npy"
                np.save(path, np.asarray(array, dtype=bool), allow_pickle=False)
                base[f"{name}_path"] = str(path)
                base[f"{name}_sha256"] = file_sha256(path)
            diagnostics = {
                "parsed_exclusion_pixels": int(semantic_parts["parsed_exclusion"].sum()),
                "landmark_exclusion_pixels": int(semantic_parts["landmark_exclusion"].sum()),
                "saturated_pixels": int(radiometric_parts["saturated"].sum()),
                "shadow_pixels": int(illumination_parts["shadow"].sum()),
                "hsi_highlight_pixels": int(illumination_parts["hsi_highlight"].sum()),
                "rgb_specular_pixels": int(illumination_parts["rgb_specular"].sum()),
                "scanline_artifact_pixels": int(illumination_parts["scanline_artifact"].sum()),
            }
            base.update(qc)
            base.update(diagnostics)
            panel = sample_dir / "qc_panel.png"
            _save_qc_panel(panel, rgb, anatomical, semantic_valid, region_arrays, str(qc["status"]))
            base["qc_panel_path"] = str(panel)
            _write_json(sample_dir / "qc.json", {**base, "coordinate_system": "aligned_rgb_yx"})
        except Exception as error:
            base["failure_codes"] = f"pipeline_exception:{repr(error)}"
        if isinstance(base.get("failure_codes"), list):
            base["failure_codes"] = "|".join(base["failure_codes"])
        if isinstance(base.get("review_codes"), list):
            base["review_codes"] = "|".join(base["review_codes"])
        mask_rows.append(base)
        for region in REGIONS:
            pixels = int(region_arrays.get(region, np.zeros((1, 1), dtype=bool)).sum())
            usable, reason = _region_is_usable(region, str(sample["direction"]), pixels, protocol)
            if derived_cube is None or region not in region_arrays:
                usable, reason = False, "sample_pipeline_failed"
            row = {
                "sample_id": sample_id,
                "subject_id": str(sample["subject_id"]),
                "split": "test",
                "expression": str(sample["expression"]),
                "direction": str(sample["direction"]),
                "region": region,
                "s1_2_analysis_role": base["analysis_role"],
                "sample_qc_status": base["status"],
                "region_usable_flag": bool(usable),
                "effective_pixels": pixels,
                "mask_path": str(base.get(f"{region}_path", "")),
                "mask_sha256": str(base.get(f"{region}_sha256", "")),
                "hsi_path": str(sample["hsi_path"]),
                "hsi_sha256": str(sample["hsi_sha256"]),
                "extraction_status": "UNAVAILABLE",
                "extraction_reason": reason,
                **_empty_spectral_fields(),
            }
            if usable and derived_cube is not None:
                row.update(_summary_fields(summarize_region(derived_cube[region_arrays[region]], trim_fraction)))
                row["extraction_status"] = "USABLE"
                row["extraction_reason"] = ""
            spectrum_rows.append(row)
        progress(f"S1-7 Test mask/spectra: {index}/{len(test_manifest)}")
    masks = pd.DataFrame(mask_rows)
    spectra = pd.DataFrame(spectrum_rows)
    panels = masks.loc[masks["qc_panel_path"].astype(str).ne("")].copy()
    if not panels.empty:
        _save_contact_sheet(output / "test_mask_contact_sheet.png", panels)
    worker_summary = {
        "rows": int(len(worker)),
        "pass": int(worker["status"].astype(str).eq("PASS").sum()),
        "fail": int(worker["status"].astype(str).ne("PASS").sum()),
    }
    return masks, spectra, worker_summary


def _spectrum(row: Mapping[str, Any], statistic: str) -> np.ndarray:
    return np.asarray(
        [float(row[f"reflectance_{statistic}_{int(wavelength)}nm"]) for wavelength in WAVELENGTHS],
        dtype=np.float64,
    )


def _fit_test_spectrum(
    row: Mapping[str, Any],
    calibration: Mapping[str, np.ndarray],
    registry: Any,
    config: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    epsilon = float(config["extraction"]["reflectance_epsilon"])
    statistic = str(config["extraction"]["statistic"])
    weights = np.asarray(calibration["band_weights"], dtype=np.float64)
    reference = np.asarray(calibration["reference_reflectance"], dtype=np.float64)
    region = str(row["region"])
    gain_key = f"side_log_gain_{_side(region)}"
    side_log_gain = float(np.asarray(calibration[gain_key])) if gain_key in calibration else 0.0
    side_gain = float(np.exp(side_log_gain))
    observed = _spectrum(row, statistic)
    corrected = observed / side_gain
    difference = np.log(np.maximum(corrected, epsilon)) - np.log(np.maximum(reference, epsilon))
    amplitude = float(np.sum(difference * weights) / np.sum(weights))
    target = _weighted_center(difference, weights)
    base = {
        "sample_id": str(row["sample_id"]),
        "subject_id": str(row["subject_id"]),
        "split": "test",
        "expression": str(row["expression"]),
        "direction": str(row["direction"]),
        "region": region,
        "analysis_role": str(row["s1_2_analysis_role"]),
    }
    records: list[dict[str, Any]] = []
    residuals: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    model_ids = (
        str(config["models"]["matched_baseline"]),
        str(config["models"]["selected"]),
        str(config["models"]["frozen_data_driven_comparator"]),
    )
    for model_id in model_ids:
        try:
            parameters: dict[str, float] = {}
            if model_id == "B0-S":
                fitted = np.zeros(WAVELENGTHS.size, dtype=np.float64)
                dimension = rank = 0
            elif model_id == "B2-PCA":
                mean = np.asarray(calibration["shape_pca_mean"], dtype=np.float64)
                components = np.asarray(calibration["shape_pca_components"], dtype=np.float64)
                score = (target - mean) @ components
                fitted = mean + components @ score
                parameters = {"pca_1": float(score[0]), "pca_2": float(score[1])}
                dimension = 2
                rank = int(np.linalg.matrix_rank(components))
            else:
                fit = fit_proxy_wls(
                    model_id,
                    corrected,
                    reference,
                    registry=registry,
                    weights=weights,
                    wavelength_nm=WAVELENGTHS,
                    epsilon=epsilon,
                )
                fitted = fit.fitted_centered_log_ratio
                parameters = dict(zip(fit.parameter_names, map(float, fit.theta), strict=True))
                parameters.update({
                    "design_condition_number": float(fit.singular_values[0] / fit.singular_values[-1]),
                    "theta_se_1": float(np.sqrt(max(fit.covariance[0, 0], 0.0))),
                    "theta_se_2": float(np.sqrt(max(fit.covariance[1, 1], 0.0))),
                })
                dimension = len(fit.parameter_names)
                rank = int(fit.rank)
            predicted = reference * np.exp(amplitude + fitted) * side_gain
            if np.any(predicted <= 0) or not np.all(np.isfinite(predicted)):
                raise FloatingPointError("Reconstructed Test spectrum is invalid")
            raw_residual = np.log(predicted) - np.log(np.maximum(observed, epsilon))
            metrics = {
                "shape_log_rmse": float(np.sqrt(np.sum(weights * (fitted - target) ** 2) / np.sum(weights))),
                "shape_log_mae": float(np.sum(weights * np.abs(fitted - target)) / np.sum(weights)),
                "raw_log_rmse": float(np.sqrt(np.sum(weights * raw_residual**2) / np.sum(weights))),
                "raw_sam": float(spectral_angle_rad(predicted, observed, epsilon)),
            }
            records.append({
                **base,
                "model_id": model_id,
                **metrics,
                "a_obs": amplitude,
                "side_log_gain": side_log_gain,
                "fit_success": True,
                "parameters_finite": bool(all(np.isfinite(value) for value in parameters.values())),
                "design_rank": rank,
                "design_dimension": dimension,
                **parameters,
            })
            residuals.extend({
                **base,
                "model_id": model_id,
                "wavelength_nm": float(wavelength),
                "shape_residual": float(fitted[band] - target[band]),
                "raw_log_residual": float(raw_residual[band]),
            } for band, wavelength in enumerate(WAVELENGTHS))
        except Exception as error:
            failures.append({**base, "model_id": model_id, "error": repr(error)})
    return records, residuals, failures


def evaluate_formal_test_gates(
    primary_spectra: pd.DataFrame,
    primary_fits: pd.DataFrame,
    primary_residuals: pd.DataFrame,
    failures: pd.DataFrame,
    gates: Mapping[str, Any],
    *,
    expected_subjects: int,
    expected_primary_regions: int,
    selected_model: str = "D2-MH",
    baseline_model: str = "B0-S",
) -> tuple[dict[str, Any], pd.DataFrame]:
    usable = primary_spectra.loc[primary_spectra["extraction_status"].eq("USABLE")].copy()
    regions_per_subject = usable.groupby("subject_id").size()
    coverage_fraction = float(len(usable) / expected_primary_regions)
    all_subjects_represented = (
        usable["subject_id"].nunique() == expected_subjects
        and bool((regions_per_subject >= int(gates["minimum_primary_regions_per_subject"])).all())
    )
    if primary_fits.empty:
        subject_metrics = pd.DataFrame(columns=[
            "subject_id", "model_id", "region_count", "shape_log_rmse", "raw_log_rmse", "raw_sam"
        ])
    else:
        subject_metrics = (
            primary_fits.groupby(["subject_id", "model_id"], as_index=False)
            .agg(
                region_count=("sample_id", "size"),
                shape_log_rmse=("shape_log_rmse", "mean"),
                raw_log_rmse=("raw_log_rmse", "mean"),
                raw_sam=("raw_sam", "mean"),
            )
        )
    pivot = subject_metrics.pivot(index="subject_id", columns="model_id", values="shape_log_rmse")
    paired = pivot.dropna(subset=[selected_model, baseline_model]) if {selected_model, baseline_model} <= set(pivot.columns) else pivot.iloc[0:0]
    improved_fraction = float((paired[selected_model] < paired[baseline_model]).mean()) if len(paired) else 0.0
    selected_subject = subject_metrics.loc[subject_metrics["model_id"].eq(selected_model)]
    baseline_subject = subject_metrics.loc[subject_metrics["model_id"].eq(baseline_model)]
    unavailable = 1.0e308
    selected_median = float(selected_subject["shape_log_rmse"].median()) if len(selected_subject) else unavailable
    baseline_median = float(baseline_subject["shape_log_rmse"].median()) if len(baseline_subject) else unavailable
    ratio = (
        selected_median / baseline_median
        if len(selected_subject) and len(baseline_subject) and np.isfinite(baseline_median) and baseline_median > 0
        else unavailable
    )
    selected_residuals = (
        primary_residuals.loc[primary_residuals["model_id"].eq(selected_model)]
        if "model_id" in primary_residuals else primary_residuals
    )
    band_median = (
        selected_residuals.groupby("wavelength_nm")["shape_residual"].median()
        if not selected_residuals.empty else pd.Series(dtype=float)
    )
    max_band = float(band_median.abs().max()) if len(band_median) else unavailable
    selected_fits = primary_fits.loc[primary_fits["model_id"].eq(selected_model)]
    selected_failures = failures.loc[
        failures.get("model_id", pd.Series(dtype=str)).astype(str).eq(selected_model)
        & failures.get("analysis_role", pd.Series(dtype=str)).astype(str).eq("primary_test")
    ] if not failures.empty else failures
    failure_fraction = float(len(selected_failures) / max(len(usable), 1))
    finite = bool(
        len(selected_fits) == len(usable)
        and np.isfinite(selected_fits[["shape_log_rmse", "raw_log_rmse", "raw_sam"]].to_numpy()).all()
        and selected_fits["parameters_finite"].astype(bool).all()
    )
    full_rank = bool(len(selected_fits) == len(usable) and (selected_fits["design_rank"] == selected_fits["design_dimension"]).all())
    checks = {
        "primary_region_fraction": coverage_fraction >= float(gates["minimum_primary_region_fraction"]),
        "minimum_one_primary_region_per_subject": all_subjects_represented,
        "all_expected_subjects_have_paired_model_metrics": len(paired) == expected_subjects,
        "improved_subject_fraction_vs_B0S": improved_fraction >= float(gates["improved_subject_fraction_vs_B0S_min"]),
        "median_shape_log_rmse_ratio_vs_B0S": ratio <= float(gates["median_shape_log_rmse_ratio_vs_B0S_max"]),
        "max_absolute_median_band_shape_residual": max_band <= float(gates["max_absolute_median_band_shape_residual"]),
        "selected_model_fit_failure_fraction": failure_fraction <= float(gates["maximum_selected_model_fit_failure_fraction"]),
        "finite_selected_model_outputs": finite if gates["require_finite_parameters"] else True,
        "full_selected_model_design_rank": full_rank if gates["require_full_design_rank"] else True,
    }
    summary = {
        "expected_primary_regions": int(expected_primary_regions),
        "usable_primary_regions": int(len(usable)),
        "primary_region_fraction": coverage_fraction,
        "subjects_with_usable_primary_region": int(usable["subject_id"].nunique()),
        "improved_subject_fraction_vs_B0S": improved_fraction,
        "selected_median_subject_shape_log_rmse": selected_median,
        "baseline_median_subject_shape_log_rmse": baseline_median,
        "median_shape_log_rmse_ratio_vs_B0S": ratio,
        "max_absolute_median_band_shape_residual": max_band,
        "selected_model_fit_failure_fraction": failure_fraction,
        "checks": checks,
        "passed": bool(all(checks.values())),
    }
    return summary, subject_metrics


def preflight_s1_7(config_path: str | Path, output_dir: str | Path) -> dict[str, Any]:
    """Validate every frozen input without opening any Test RGB or HSI file."""
    config_file = _resolve(config_path)
    output = _resolve(output_dir)
    config = yaml.safe_load(config_file.read_text(encoding="utf-8"))
    state = _validate_frozen_inputs(config_file, config, output)
    return {
        "status": "PREFLIGHT_PASS_NO_TEST_CONTENT_ACCESSED",
        "config_path": str(config_file),
        "config_sha256": state["config_sha256"],
        "output_dir": str(output),
        "test_manifest_rows_seen_as_metadata": int(len(state["test_manifest"])),
        "test_rgb_files_opened": 0,
        "test_hsi_files_opened": 0,
        "selected_model": state["spec"]["selected_representation_model"],
    }


def _stress_summary(fits: pd.DataFrame, selected_model: str) -> list[dict[str, Any]]:
    stress = fits.loc[
        fits["analysis_role"].eq("stress_test") & fits["model_id"].eq(selected_model)
    ].copy()
    if stress.empty:
        return []
    summary = (
        stress.groupby(["expression", "direction", "region"], as_index=False)
        .agg(
            spectra=("sample_id", "size"),
            subjects=("subject_id", "nunique"),
            median_shape_log_rmse=("shape_log_rmse", "median"),
            median_raw_log_rmse=("raw_log_rmse", "median"),
            median_raw_sam=("raw_sam", "median"),
        )
    )
    return summary.to_dict("records")


def _write_report(path: Path, decision: Mapping[str, Any]) -> None:
    metrics = decision["primary_test_metrics"]
    lines = [
        "# Stage 1 Formal Test Decision",
        "",
        f"- Decision: `{decision['decision']}`",
        f"- Frozen representation: `{decision['selected_representation_model']}` (`theta_repr=[delta_M_OD, delta_Hb_OD]`)",
        f"- Test subjects: {decision['test_subjects']}",
        f"- Usable primary cheeks: {metrics['usable_primary_regions']}/{metrics['expected_primary_regions']}",
        f"- Subjects improved over B0-S: {metrics['improved_subject_fraction_vs_B0S']:.3f}",
        f"- Median subject shape log-RMSE: {metrics['selected_median_subject_shape_log_rmse']:.6f}",
        f"- Median ratio versus B0-S: {metrics['median_shape_log_rmse_ratio_vs_B0S']:.6f}",
        f"- Maximum absolute median band residual: {metrics['max_absolute_median_band_shape_residual']:.6f}",
        f"- Selected-model fit failure fraction: {metrics['selected_model_fit_failure_fraction']:.6f}",
        "",
        "## Gate checks",
        "",
    ]
    lines.extend(f"- `{name}`: {'PASS' if passed else 'FAIL'}" for name, passed in metrics["checks"].items())
    lines.extend([
        "",
        "## Evidence boundary",
        "",
        "This decision concerns low-dimensional spectral representation feasibility in the frozen neutral/front cheek domain.",
        "The coordinate names are model coordinates, not validated melanin or hemoglobin physiology.",
        "Expression/pose results are non-blocking stress evidence. The no-eyewear deployment domain remains independently unverified.",
        "The 31-band effective SRF remains missing, so absolute or cross-device physiological claims are not supported.",
    ])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_s1_7(
    config_path: str | Path,
    output_dir: str | Path,
    progress: Callable[[str], None] = print,
) -> dict[str, Any]:
    config_file = _resolve(config_path)
    output = _resolve(output_dir)
    config = yaml.safe_load(config_file.read_text(encoding="utf-8"))
    state = _validate_frozen_inputs(config_file, config, output)
    output.mkdir(parents=True, exist_ok=False)
    started = datetime.now(timezone.utc).isoformat()
    lock = {
        "schema_version": 1,
        "stage": "S1-7",
        "run_id": config["run_id"],
        "status": "FORMAL_TEST_ACCESS_STARTED",
        "created_at_utc": started,
        "config_path": str(config_file),
        "config_sha256": state["config_sha256"],
        "frozen_stage1_spec_sha256": file_sha256(state["inputs"]["frozen_stage1_spec"]),
        "maximum_formal_runs": 1,
        "manual_test_repair_allowed": False,
        "overwrite_allowed": False,
    }
    _write_json(output / "FORMAL_TEST_LOCK.json", lock)
    try:
        progress("Formal S1-7 lock created; Test access count is now 1/1")
        masks, spectra, worker_summary = _build_test_masks_and_spectra(state, config, output, progress)
        masks.to_parquet(output / "test_mask_manifest.parquet", index=False)
        masks.to_csv(output / "test_mask_manifest.csv", index=False, encoding="utf-8-sig")
        spectra.to_parquet(output / "test_region_spectra.parquet", index=False)
        spectra.to_csv(output / "test_region_spectra.csv", index=False, encoding="utf-8-sig")

        with np.load(state["inputs"]["frozen_calibration"], allow_pickle=False) as archive:
            calibration = {name: np.asarray(archive[name]) for name in archive.files}
        if not np.array_equal(np.asarray(calibration["wavelength_nm"], dtype=float), WAVELENGTHS):
            raise RuntimeError("Frozen calibration wavelengths are invalid")
        registry = load_revised_registry(state["registry_path"])
        usable = spectra.loc[spectra["extraction_status"].eq("USABLE")].copy()
        fit_rows: list[dict[str, Any]] = []
        residual_rows: list[dict[str, Any]] = []
        failure_rows: list[dict[str, Any]] = []
        for row in usable.sort_values(["subject_id", "sample_id", "region"]).to_dict("records"):
            current_fits, current_residuals, current_failures = _fit_test_spectrum(
                row, calibration, registry, config
            )
            fit_rows.extend(current_fits)
            residual_rows.extend(current_residuals)
            failure_rows.extend(current_failures)
        fit_columns = [
            "sample_id", "subject_id", "split", "expression", "direction", "region", "analysis_role",
            "model_id", "shape_log_rmse", "shape_log_mae", "raw_log_rmse", "raw_sam", "a_obs",
            "side_log_gain", "fit_success", "parameters_finite", "design_rank", "design_dimension",
        ]
        residual_columns = [
            "sample_id", "subject_id", "split", "expression", "direction", "region", "analysis_role",
            "model_id", "wavelength_nm", "shape_residual", "raw_log_residual",
        ]
        fits = pd.DataFrame(fit_rows) if fit_rows else pd.DataFrame(columns=fit_columns)
        residuals = pd.DataFrame(residual_rows) if residual_rows else pd.DataFrame(columns=residual_columns)
        failures = pd.DataFrame(failure_rows, columns=[
            "sample_id", "subject_id", "split", "expression", "direction", "region", "analysis_role", "model_id", "error"
        ])
        primary_regions = set(config["cohort"]["primary_regions"])
        primary_spectra = spectra.loc[
            spectra["s1_2_analysis_role"].eq("primary_test") & spectra["region"].isin(primary_regions)
        ].copy()
        primary_fits = fits.loc[
            fits["analysis_role"].eq("primary_test") & fits["region"].isin(primary_regions)
        ].copy()
        primary_residuals = residuals.loc[
            residuals["analysis_role"].eq("primary_test") & residuals["region"].isin(primary_regions)
        ].copy()
        gate_summary, subject_metrics = evaluate_formal_test_gates(
            primary_spectra,
            primary_fits,
            primary_residuals,
            failures,
            config["decision_gates"],
            expected_subjects=int(config["cohort"]["expected_subjects"]),
            expected_primary_regions=int(config["cohort"]["expected_primary_regions"]),
            selected_model=str(config["models"]["selected"]),
            baseline_model=str(config["models"]["matched_baseline"]),
        )
        fits.to_parquet(output / "test_per_spectrum_fits.parquet", index=False)
        fits.to_csv(output / "test_per_spectrum_fits.csv", index=False)
        residuals.to_parquet(output / "test_wavelength_residuals.parquet", index=False)
        residuals.to_csv(output / "test_wavelength_residuals.csv", index=False)
        subject_metrics.to_csv(output / "test_subject_metrics.csv", index=False)
        _write_json(output / "test_fit_failures.json", {
            "failure_count": int(len(failures)),
            "failures": failures.to_dict("records"),
        })
        stress = _stress_summary(fits, str(config["models"]["selected"]))
        _write_json(output / "test_stress_summary.json", {
            "blocking_for_primary_decision": False,
            "groups": stress,
        })
        mask_counts = masks["status"].value_counts().to_dict()
        input_audit = {
            "stage": "S1-7",
            "formal_test_access_count": 1,
            "test_rgb_files_read": int(len(state["test_manifest"])),
            "test_hsi_files_read": int(len(state["test_manifest"])),
            "train_files_read": 0,
            "validation_files_read": 0,
            "global_refit_performed": False,
            "model_reselection_performed": False,
            "manual_test_repairs": 0,
            "worker_summary": worker_summary,
            "mask_status_counts": {str(key): int(value) for key, value in mask_counts.items()},
            "inputs": {name: {"path": str(path), "sha256": file_sha256(path)} for name, path in state["inputs"].items()},
            "model_registry": {"path": str(state["registry_path"]), "sha256": file_sha256(state["registry_path"])},
            "test_worker": {"path": str(TEST_RGB_WORKER), "sha256": file_sha256(TEST_RGB_WORKER)},
            "implementation": {"path": str(Path(__file__).resolve()), "sha256": file_sha256(Path(__file__).resolve())},
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scipy": scipy.__version__,
        }
        _write_json(output / "input_audit.json", input_audit)
        decision = {
            "schema_version": 1,
            "stage": "S1-7",
            "status": "STAGE1_COMPLETE",
            "decision": config["decision_policy"]["pass_label"] if gate_summary["passed"] else config["decision_policy"]["nonpassing_label"],
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "formal_test_run_number": 1,
            "formal_test_run_limit": 1,
            "selected_representation_model": str(config["models"]["selected"]),
            "selected_representation_dimension": int(state["spec"]["selected_representation_dimension"]),
            "test_subjects": int(state["test_manifest"]["subject_id"].nunique()),
            "test_samples": int(len(state["test_manifest"])),
            "primary_test_metrics": gate_summary,
            "decision_gates": dict(config["decision_gates"]),
            "stress_domain_blocking": False,
            "stress_group_count": len(stress),
            "physiological_M_H_claim": "NOT_ASSESSED_IN_STAGE1",
            "effective_srf_status": str(state["spec"]["wavelength_contract"]["effective_srf_status"]),
            "no_eyewear_target_domain_independently_validated": False,
            "test_access_count": 1,
            "next_stage_allowed": bool(gate_summary["passed"]),
            "authorized_next_stage": "S2" if gate_summary["passed"] else "S1-REVISE",
        }
        _write_json(output / "STAGE1_DECISION.json", decision)
        _write_report(output / "STAGE1_DECISION.md", decision)
        completion = {
            "status": "FORMAL_TEST_COMPLETE",
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "decision": decision["decision"],
            "decision_sha256": file_sha256(output / "STAGE1_DECISION.json"),
            "test_access_count": 1,
        }
        _write_json(output / "FORMAL_TEST_COMPLETE.json", completion)
        files = {
            str(path.relative_to(output)).replace("\\", "/"): file_sha256(path)
            for path in sorted(output.rglob("*"))
            if path.is_file() and path.name != "output_manifest.json"
        }
        _write_json(output / "output_manifest.json", {
            "schema_version": 1,
            "stage": "S1-7",
            "run_id": config["run_id"],
            "file_count_excluding_manifest": len(files),
            "files": files,
        })
        return decision
    except Exception as error:
        _write_json(output / "FORMAL_TEST_ABORTED.json", {
            "schema_version": 1,
            "stage": "S1-7",
            "status": "FORMAL_TEST_ABORTED_AFTER_ACCESS_LOCK",
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "error": repr(error),
            "test_access_count": 1,
            "rerun_as_first_independent_test_allowed": False,
        })
        raise


__all__ = ["evaluate_formal_test_gates", "preflight_s1_7", "run_s1_7"]
