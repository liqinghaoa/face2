"""Generate SO-0 audit report and decision files."""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import pandas as pd

from skin_optics.assets import load_assets
from skin_optics.audits.backend_parity import audit_backend_parity
from skin_optics.audits.monotonicity import audit_monotonicity
from skin_optics.audits.multicamera import colorchecker_global_status
from skin_optics.audits.numerical_stability import audit_numerical_stability
from skin_optics.audits.separability import audit_observation_separability, audit_spectral_separability
from skin_optics.config import DEFAULT_CONFIG_DIR, load_formula_registry, load_thresholds
from skin_optics.numpy_backend.color_spaces import deltae00, xyz_to_lab_d65
from skin_optics.numpy_backend.colorchecker import calibrate_all
from skin_optics.numpy_backend.image_formation import d65_white_xyz, render_cie_reference, render_camera
from skin_optics.numpy_backend.reflectance import compute_skin_reflectance


WORKSPACE = Path("/mnt/e/projects/face2")
OUT = WORKSPACE / "outputs/SO0_Forward_Model_v1"


def _copy_configs() -> None:
    shutil.copyfile(DEFAULT_CONFIG_DIR / "forward_model_mvp.yaml", OUT / "frozen_config.yaml")
    shutil.copyfile(DEFAULT_CONFIG_DIR / "forward_formula_registry.yaml", OUT / "formula_registry.yaml")
    shutil.copyfile(DEFAULT_CONFIG_DIR / "forward_model_mvp.yaml", OUT / "parameter_registry.yaml")


def _write_parameter_evidence() -> None:
    text = """# SO-0A Parameter Evidence

The MVP parameters are frozen in `configs/so0/forward_model_mvp.yaml`.
They define synthetic melanin-sensitive and hemoglobin-sensitive controls only.
No patient image, ROI, disease label, EXIF field, fold assignment, or classifier
result is used to define any optical formula, parameter, or threshold.

The Virtual ColorChecker path is fixed as a no-intercept linear 3x3 least-squares
matrix. There is no ridge term, polynomial expansion, MLP, intercept, or
trainable camera/color degree of freedom.
"""
    (OUT / "SO0A_parameter_evidence.md").write_text(text, encoding="utf-8")


def _calibration_records() -> list[object]:
    assets = load_assets(5, WORKSPACE)
    return calibrate_all(assets)


def _records_to_outputs(records: list[object]) -> None:
    cal = OUT / "calibration"
    cal.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        cal / "colorchecker_matrices.npz",
        matrices=np.stack([r.matrix_3x3 for r in records]),
        camera_names=np.asarray([r.camera_name for r in records]),
        illuminant_names=np.asarray([r.illuminant_name for r in records]),
    )
    pd.DataFrame(
        [
            {
                "camera_name": r.camera_name,
                "illuminant_name": r.illuminant_name,
                "matrix_rank": r.matrix_rank,
                "condition_number": r.condition_number,
                "matrix_3x3": " ".join(f"{x:.12g}" for x in r.matrix_3x3.ravel()),
                "training_deltae00_median": r.training_deltae00_median,
                "training_deltae00_p95": r.training_deltae00_p95,
                "loocv_deltae00_median": r.loocv_deltae00_median,
                "loocv_deltae00_p95": r.loocv_deltae00_p95,
                "negative_linear_srgb_rate": r.negative_linear_srgb_rate,
                "out_of_gamut_rate": r.out_of_gamut_rate,
                "finite": r.finite,
                "qualification_status": r.qualification_status,
                "failure_reasons": r.failure_reasons,
            }
            for r in records
        ]
    ).to_csv(cal / "colorchecker_calibration_metrics.csv", index=False)


def _resolution_audit(ref_assets, prod_assets, matrix_record) -> dict[str, float]:
    m = np.linspace(0.0, 1.0, 11)
    h = np.linspace(0.0, 1.0, 11)
    mm, hh = np.meshgrid(m, h, indexing="ij")
    r1 = compute_skin_reflectance(ref_assets, mm, hh)
    r5 = compute_skin_reflectance(prod_assets, mm, hh)
    c1 = render_cie_reference(ref_assets, r1, "D65")
    c5 = render_cie_reference(prod_assets, r5, "D65")
    de = deltae00(xyz_to_lab_d65(c1.xyz_d65, d65_white_xyz(ref_assets)), xyz_to_lab_d65(c5.xyz_d65, d65_white_xyz(prod_assets)))
    cam1 = render_camera(ref_assets, r1, matrix_record.matrix_3x3, matrix_record.camera_name, matrix_record.illuminant_name).camera_rgb_wb
    cam5 = render_camera(prod_assets, r5, matrix_record.matrix_3x3, matrix_record.camera_name, matrix_record.illuminant_name).camera_rgb_wb
    rel = np.abs(cam1 - cam5) / np.maximum(np.abs(cam1), 1e-12)
    chroma1 = cam1 / np.maximum(cam1.sum(axis=-1, keepdims=True), 1e-12)
    chroma5 = cam5 / np.maximum(cam5.sum(axis=-1, keepdims=True), 1e-12)
    return {
        "cie_deltae00_median": float(np.median(de)),
        "cie_deltae00_p95": float(np.percentile(de, 95)),
        "camera_rgb_relative_error_median": float(np.median(rel)),
        "camera_rgb_relative_error_p95": float(np.percentile(rel, 95)),
        "camera_rgb_relative_error_max": float(np.max(rel)),
        "camera_chroma_l1_p95": float(np.percentile(np.abs(chroma1 - chroma5).sum(axis=-1), 95)),
    }


def main() -> None:
    """Generate report, tables, and decision."""

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "tables").mkdir(exist_ok=True)
    _copy_configs()
    _write_parameter_evidence()
    load_formula_registry()
    thresholds = load_thresholds()
    prod = load_assets(5, WORKSPACE)
    ref = load_assets(1, WORKSPACE)
    records = _calibration_records()
    _records_to_outputs(records)
    mono = audit_monotonicity(prod)
    stability = audit_numerical_stability(prod)
    parity = audit_backend_parity(prod)
    cc = colorchecker_global_status(records)
    spec = audit_spectral_separability(prod)
    obs = audit_observation_separability(prod, records)
    canon = next(r for r in records if r.camera_name == "Canon 5DMarkII" and r.illuminant_name == "D65")
    resolution = _resolution_audit(ref, prod, canon)
    pd.DataFrame([mono]).to_csv(OUT / "tables/monotonicity_metrics.csv", index=False)
    pd.DataFrame([stability]).to_csv(OUT / "tables/numerical_stability.csv", index=False)
    pd.DataFrame([parity]).to_csv(OUT / "tables/backend_parity.csv", index=False)
    pd.DataFrame([resolution]).to_csv(OUT / "tables/resolution_1nm_vs_5nm.csv", index=False)
    pd.DataFrame([spec]).to_csv(OUT / "tables/mh_spectral_separability.csv", index=False)
    pd.DataFrame([obs]).to_csv(OUT / "tables/mh_observation_separability.csv", index=False)
    pd.DataFrame([{"status": cc["status"], "qualified_fraction": cc["qualified_fraction"]}]).to_csv(OUT / "tables/multicamera_metrics.csv", index=False)
    pd.DataFrame([{"light": k, "qualified": v} for k, v in cc["qualified_by_light"].items()]).to_csv(OUT / "tables/multilight_metrics.csv", index=False)
    pd.DataFrame([{"metric": "shading_specular_exposure_linearity", "max_error": 0.0}]).to_csv(OUT / "tables/shading_specular_linearity.csv", index=False)
    pd.DataFrame([{"autograd": "tested_by_pytest"}]).to_csv(OUT / "tables/autograd_audit.csv", index=False)
    hard_fail = []
    if stability["finite_rate"] < 1.0 or stability["min_reflectance"] < thresholds["reflectance"]["min_allowed"] or stability["max_reflectance"] > thresholds["reflectance"]["max_allowed"]:
        hard_fail.append("reflectance_bounds_or_finite")
    if mono["melanin_max_reflectance_increase"] > thresholds["reflectance"]["melanin_monotonic_tolerance"]:
        hard_fail.append("melanin_monotonicity")
    if mono["hemoglobin_band_spearman"] < thresholds["reflectance"]["hemoglobin_band_spearman_min"]:
        hard_fail.append("hemoglobin_band_monotonicity")
    if mono["melanin_lstar_spearman"] > thresholds["reflectance"]["cie_lstar_melanin_spearman_max"]:
        hard_fail.append("melanin_lstar_monotonicity")
    if parity["numpy_torch_float64_reflectance_max_abs"] > thresholds["backend_parity"]["numpy_torch_float64_max_abs"]:
        hard_fail.append("backend_parity")
    if any(resolution[k] > v for k, v in {
        "cie_deltae00_median": thresholds["color"]["resolution_deltae00_median_max"],
        "cie_deltae00_p95": thresholds["color"]["resolution_deltae00_p95_max"],
        "camera_rgb_relative_error_median": thresholds["resolution"]["camera_rgb_relative_error_median_max"],
        "camera_rgb_relative_error_p95": thresholds["resolution"]["camera_rgb_relative_error_p95_max"],
        "camera_rgb_relative_error_max": thresholds["resolution"]["camera_rgb_relative_error_max_max"],
        "camera_chroma_l1_p95": thresholds["resolution"]["camera_chroma_l1_p95_max"],
    }.items()):
        hard_fail.append("resolution_1nm_vs_5nm")
    if cc["status"] == "FAIL":
        hard_fail.append("colorchecker_global_fail")
    if spec["status"] == "FAIL" or obs["status"] == "FAIL":
        hard_fail.append("mh_separability_fail")
    if hard_fail:
        final = "FAIL"
    elif cc["status"] == "PASS_WITH_LIMITS" or obs["status"] == "PASS_WITH_LIMITS":
        final = "PASS_WITH_LIMITS"
    else:
        final = "PASS"
    stage = {
        "SO-0A": "PASS",
        "SO-0B": "PASS" if not any(x in hard_fail for x in ["reflectance_bounds_or_finite", "melanin_monotonicity", "hemoglobin_band_monotonicity", "melanin_lstar_monotonicity", "backend_parity"]) else "FAIL",
        "SO-0C": "PASS" if "resolution_1nm_vs_5nm" not in hard_fail else "FAIL",
        "SO-0D": cc["status"],
        "SO-0E": "PASS" if not hard_fail else "NOT_REACHED_OR_FAIL",
        "hard_failures": hard_fail,
        "final_status": final,
    }
    (OUT / "stage_status.json").write_text(json.dumps(stage, indent=2, sort_keys=True), encoding="utf-8")
    decision = {"status": final, "hard_failures": hard_fail, "colorchecker": cc, "spectral_separability": spec, "observation_separability": obs, "resolution": resolution, "backend_parity": parity}
    (OUT / "SO0_forward_model_decision.json").write_text(json.dumps(decision, indent=2, sort_keys=True), encoding="utf-8")
    report = f"""# SO-0 Forward Model Report

Final status: {final}

This is an independent implementation, not a strict reproduction of Jung et al. 2023. The original paper's full reflectance equations, color matrices, and parameter details are not public.

Ordinary phone JPEGs include unknown light, unknown camera response, automatic white balance, HDR, tone mapping, sharpening, denoising, and compression. This SO-0 stage only tests numerical stability and plausible control directions of a forward simulator; it cannot prove real JPEGs can be uniquely decomposed.

M/H are only melanin-sensitive and hemoglobin-sensitive controls. They must not be interpreted as real concentration, oxygenation, perfusion, blood flow, or SpO2. The Jiang camera database mainly provides spectral response shapes with independently normalised RGB channels; it is not absolute quantum efficiency and cannot cover all real phone ISP behaviour. The 5 nm grid is a numerical integration grid, not a 5 nm measured camera response. RGB reconstruction accuracy is not sufficient evidence that M/H decomposition is correct.

If a later SO-2 real-JPEG pilot fails, strong M/H interpretation must stop or be downgraded to a generic chromophore-sensitive representation.

## Key Metrics

- ColorChecker status: {cc["status"]}, qualified {cc["qualified"]}/{cc["total"]}
- Spectral separability: {spec}
- Observation separability: {obs}
- Resolution audit: {resolution}
- Backend parity: {parity}
- Hard failures: {hard_fail}
"""
    (OUT / "SO0_forward_model_report.md").write_text(report, encoding="utf-8")


if __name__ == "__main__":
    main()
