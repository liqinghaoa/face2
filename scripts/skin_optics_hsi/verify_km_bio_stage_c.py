"""Verify frozen Stage C artifacts and add the registered shape diagnostic."""

from __future__ import annotations

import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
RUN = ROOT / "outputs/skin_optics_hsi_v1/stage1_hsi_physics/km_bio_v1_train_inversion"
OUT = ROOT / "outputs/skin_optics_hsi_v1/stage1_hsi_physics/km_bio_v1_train_inversion_verification"
OBSERVATION = ROOT / "outputs/skin_optics_hsi_v1/stage1_hsi_physics/km_bio_v1_observation_audit/train_primary_observation_manifest.parquet"


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def main() -> None:
    if OUT.exists():
        raise FileExistsError(f"Refusing to overwrite Stage C verification: {OUT}")
    decision = json.loads((RUN / "stage_c_decision.json").read_text(encoding="utf-8"))
    main_results = pd.read_parquet(RUN / "main_results.parquet")
    main_starts = pd.read_parquet(RUN / "main_multistart_results.parquet")
    profile = pd.read_parquet(RUN / "profile_grid.parquet")
    profile_starts = pd.read_parquet(RUN / "profile_multistart_results.parquet")
    sensitivity = pd.read_parquet(RUN / "sensitivity_results.parquet")
    observation = pd.read_parquet(OBSERVATION)
    wavelength = np.arange(400, 701, 10)
    observed_columns = [f"observed_reflectance_{value}nm" for value in wavelength]
    predicted_columns = [f"reflectance_hat_{value}nm" for value in wavelength]
    joined = main_results.merge(
        observation[["capture_id", "roi", *observed_columns]],
        on=["capture_id", "roi"], how="left", validate="one_to_one",
    )
    epsilon = 1e-6
    observed = joined[observed_columns].to_numpy(dtype=np.float64)
    predicted = joined[predicted_columns].to_numpy(dtype=np.float64)
    log_residual = np.log(predicted + epsilon) - np.log(observed + epsilon)
    centered = log_residual - log_residual.mean(axis=1, keepdims=True)
    shape = joined[["subject_id", "capture_id", "roi"]].copy()
    shape["mean_log_residual"] = log_residual.mean(axis=1)
    shape["centered_logrmse"] = np.sqrt(np.mean(centered**2, axis=1))
    shape["sam_deg"] = joined["sam_deg"]
    for index, value in enumerate(wavelength):
        shape[f"centered_log_residual_{value}nm"] = centered[:, index]

    output_hash_mismatches = []
    for name, record in decision["outputs"].items():
        if digest(Path(record["path"])) != record["sha256"]:
            output_hash_mismatches.append(name)
    recomputed_logrmse = np.sqrt(np.mean(log_residual**2, axis=1))
    expected_sensitivity = {
        "epidermis_0.050mm", "epidermis_0.070mm", "scattering_x0.8", "scattering_x1.2",
        "whole_blood_hb_120gL", "whole_blood_hb_180gL", "observation_scale_0.95",
        "observation_scale_1.05", "observation_tilt_-0.02", "observation_tilt_+0.02",
        "subset_420_680_point", "subset_420_680_gaussian_5nm", "subset_420_680_gaussian_10nm",
    }
    checks = {
        "decision_status_is_registered": decision["status"] in {"SPECTRAL_ONLY", "CONDITIONAL_BIO_READY", "CONDITIONAL_BIO_READY_WITH_S", "REVISE_OBSERVATION_OR_MODEL"},
        "all_recorded_output_hashes_match": not output_hash_mismatches,
        "main_result_count_88": len(main_results) == 88,
        "main_starts_33_per_spectrum": bool((main_starts.groupby(["capture_id", "roi"]).size() == 33).all()),
        "profile_three_parameters_per_spectrum": bool((profile.groupby(["capture_id", "roi"])["profiled_parameter"].nunique() == 3).all()),
        "profile_nine_starts_per_point": bool((profile_starts.groupby(["capture_id", "roi", "profiled_parameter", "fixed_u", "grid_kind"]).size() == 9).all()),
        "all_profile_conditional_fits_converged": bool(profile["success"].all()),
        "sensitivity_thirteen_settings_per_spectrum": bool((sensitivity.groupby(["capture_id", "roi"]).size() == 13).all()),
        "sensitivity_setting_names_match_contract": set(sensitivity["setting"]) == expected_sensitivity,
        "all_sensitivity_fits_converged": bool(sensitivity["success"].all()),
        "stored_logrmse_recomputes": bool(np.allclose(recomputed_logrmse, main_results["logrmse"], rtol=0, atol=1e-12)),
        "shape_diagnostic_finite": bool(np.isfinite(shape.select_dtypes(include=[np.number]).to_numpy()).all()),
        "validation_content_reads_zero": decision["counts"]["validation_content_reads"] == 0,
        "test_content_reads_zero": decision["counts"]["test_content_reads"] == 0,
    }
    passed = bool(all(checks.values()))
    OUT.mkdir(parents=True, exist_ok=False)
    shape_path = OUT / "shape_diagnostic.csv"
    shape.to_csv(shape_path, index=False, encoding="utf-8-sig")
    result = {
        "schema_version": 1,
        "stage": "KM-BIO-v1-Stage-C-verification",
        "status": "PASS" if passed else "FAIL",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "model_decision_verified": decision["status"],
        "checks": checks,
        "output_hash_mismatches": output_hash_mismatches,
        "shape_diagnostic": {
            "row_count": len(shape),
            "median_centered_logrmse": float(shape["centered_logrmse"].median()),
            "p90_centered_logrmse": float(shape["centered_logrmse"].quantile(0.9)),
            "median_mean_log_residual": float(shape["mean_log_residual"].median()),
            "path": str(shape_path),
            "sha256": digest(shape_path),
        },
        "implementation_hashes": {
            "km_bio_v1.py": digest(ROOT / "src/skin_optics_hsi/km_bio_v1.py"),
            "km_bio_inverse.py": digest(ROOT / "src/skin_optics_hsi/km_bio_inverse.py"),
            "km_bio_stage_c.py": digest(ROOT / "src/skin_optics_hsi/km_bio_stage_c.py"),
            "run_km_bio_stage_c.py": digest(ROOT / "scripts/skin_optics_hsi/run_km_bio_stage_c.py"),
            "verify_km_bio_stage_c.py": digest(Path(__file__).resolve()),
        },
    }
    (OUT / "verification_summary.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

