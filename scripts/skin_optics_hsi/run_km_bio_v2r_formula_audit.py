"""Run the KM-BIO-v2R formula audit without reading HSI observations."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch
from scipy.stats import qmc

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.skin_optics_hsi.km_bio_v1 import forward_preloaded_numpy as v1_forward
from src.skin_optics_hsi.km_bio_v2r import (
    AS_BOUNDS,
    DELTA_BS_BOUNDS,
    G0_BOUNDS,
    THETA_EXTENDED_HI,
    THETA_EXTENDED_LO,
    WAVELENGTH_NM,
    _km_layer_numpy,
    blood_packaging_factor_numpy,
    estimate_global_gain,
    estimate_shape_scales,
    forward_preloaded_numpy,
    forward_torch,
    load_optical_numpy,
    observation_scale_audit,
    profile_identifiability,
)

ASSET = ROOT / "data/external/SO0_Spectral_Assets_v1/processed/SO0_Spectral_Standardized_v1/production_10nm/derived_optics_10nm.npz"
CONTRACT = ROOT / "configs/skin_optics_hsi/km_bio_v2r_formula_contract.yaml"
SOURCE = ROOT / "src/skin_optics_hsi/km_bio_v2r.py"
TEST = ROOT / "tests/skin_optics_hsi/test_km_bio_v2r_formula.py"
AUDIT_SCRIPT = Path(__file__).resolve()
OUT = ROOT / "outputs/skin_optics_hsi_v2r/stage1_hsi_physics/km_bio_v2r_formula_audit"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _published_layer(k: np.ndarray, s: np.ndarray, d: float) -> tuple[np.ndarray, np.ndarray]:
    q = np.sqrt(k * (k + 2 * s))
    beta = np.sqrt(k / (k + 2 * s))
    den = (1 + beta) ** 2 * np.exp(q * d) - (1 - beta) ** 2 * np.exp(-q * d)
    return ((1 - beta * beta) * (np.exp(q * d) - np.exp(-q * d)) / den, 4 * beta / den)


def main() -> None:
    rng = np.random.default_rng(20260910)
    k = 10.0 ** rng.uniform(-3, 2, 10000)
    s = 10.0 ** rng.uniform(-3, 2, 10000)
    r, t = _km_layer_numpy(k, s, 0.060)
    rp, tp = _published_layer(k, s, 0.060)

    optical = load_optical_numpy(WAVELENGTH_NM, ASSET)
    theta = np.array([0.18, 0.025, 0.72], dtype=np.float64)
    v2 = forward_preloaded_numpy(theta, optical, wavelength_nm=WAVELENGTH_NM)
    v1 = v1_forward(theta, optical)
    two_theta = forward_preloaded_numpy(theta[:2], optical, wavelength_nm=WAVELENGTH_NM, s0=theta[2])
    no_pack = forward_preloaded_numpy(theta, optical, wavelength_nm=WAVELENGTH_NM, diameter_um=0.0)
    identity_scattering = forward_preloaded_numpy(theta, optical, wavelength_nm=WAVELENGTH_NM, scattering_amplitude=1.0, delta_bs=0.0)
    identity_gain = forward_preloaded_numpy(theta, optical, wavelength_nm=WAVELENGTH_NM, g0=1.0)

    sobol = qmc.Sobol(6, scramble=True, seed=20260910).random_base2(7)
    theta_set = THETA_EXTENDED_LO + sobol[:, :3] * (THETA_EXTENDED_HI - THETA_EXTENDED_LO)
    diameter_set = 30.0 * sobol[:, 3]
    amplitude_set = AS_BOUNDS[0] + sobol[:, 4] * (AS_BOUNDS[1] - AS_BOUNDS[0])
    delta_set = DELTA_BS_BOUNDS[0] + sobol[:, 5] * (DELTA_BS_BOUNDS[1] - DELTA_BS_BOUNDS[0])
    predictions = np.vstack([
        forward_preloaded_numpy(row, optical, wavelength_nm=WAVELENGTH_NM, diameter_um=diameter, scattering_amplitude=amplitude, delta_bs=delta)
        for row, diameter, amplitude, delta in zip(theta_set, diameter_set, amplitude_set, delta_set)
    ])
    high_gain_predictions = 1.5 * predictions
    high_gain_audit = observation_scale_audit(high_gain_predictions)

    optical_t = {key: torch.as_tensor(value, dtype=torch.float64) for key, value in optical.items()}
    theta_t = torch.tensor(theta, dtype=torch.float64, requires_grad=True)
    torch_pred = forward_torch(theta_t, optical_t, wavelength_nm=WAVELENGTH_NM, diameter_um=15.0, scattering_amplitude=1.1, delta_bs=0.12, g0=0.97)
    torch_pred.sum().backward()
    analytic = theta_t.grad.detach().numpy().copy()
    theta_zero_t = torch.tensor(theta, dtype=torch.float64, requires_grad=True)
    torch_zero_pred = forward_torch(theta_zero_t, optical_t, wavelength_nm=WAVELENGTH_NM, diameter_um=0.0)
    torch_zero_pred.sum().backward()
    numeric = []
    step = 1e-5
    for index in range(3):
        plus, minus = theta.copy(), theta.copy()
        plus[index] += step
        minus[index] -= step
        numeric.append((forward_preloaded_numpy(plus, optical, wavelength_nm=WAVELENGTH_NM, diameter_um=15.0, scattering_amplitude=1.1, delta_bs=0.12, g0=0.97).sum() - forward_preloaded_numpy(minus, optical, wavelength_nm=WAVELENGTH_NM, diameter_um=15.0, scattering_amplitude=1.1, delta_bs=0.12, g0=0.97).sum()) / (2 * step))
    numeric = np.asarray(numeric)
    grad_abs = np.abs(analytic - numeric)

    mu = np.array([0.0, 1.0, 100.0], dtype=np.float64)
    package_zero = blood_packaging_factor_numpy(mu, 0.0)
    package_small = blood_packaging_factor_numpy(np.array([10.0]), 1e-5)[0]
    calibration_theta = [theta, np.array([0.28, 0.04, 0.62])]
    amplitude_grid = np.array([0.9, 1.0, 1.1])
    delta_grid = np.array([-0.1, 0.0, 0.12])
    candidate_grid = np.asarray([
        [[forward_preloaded_numpy(row, optical, wavelength_nm=WAVELENGTH_NM, scattering_amplitude=amplitude, delta_bs=delta) for row in calibration_theta] for delta in delta_grid]
        for amplitude in amplitude_grid
    ])
    observed = candidate_grid[2, 2]
    shape_profile = estimate_shape_scales(observed, candidate_grid, amplitude_grid, delta_grid)
    gain_profile = estimate_global_gain(observed, candidate_grid[1, 1])
    as_profile = [
        {"A_s": float(value), "centered_logrmse": min(row["centered_logrmse"] for row in shape_profile["profile"] if row["A_s"] == value)}
        for value in amplitude_grid
    ]
    delta_profile = [
        {"delta_bs": float(value), "centered_logrmse": min(row["centered_logrmse"] for row in shape_profile["profile"] if row["delta_bs"] == value)}
        for value in delta_grid
    ]
    as_identifiability = profile_identifiability(as_profile, "A_s", "centered_logrmse", AS_BOUNDS)
    delta_identifiability = profile_identifiability(delta_profile, "delta_bs", "centered_logrmse", DELTA_BS_BOUNDS)
    gain_identifiability = profile_identifiability(gain_profile["profile"], "g0", "raw_logrmse", G0_BOUNDS)

    result = {
        "model_id": "KM2L-HF-v2R",
        "hsi_read": False,
        "random_seed": 20260910,
        "random_layer_count": 10000,
        "published_vs_stable_max_abs": float(max(np.max(np.abs(r - rp)), np.max(np.abs(t - tp)))),
        "random_layer_finite": bool(np.isfinite(r).all() and np.isfinite(t).all()),
        "random_layer_min_R": float(r.min()),
        "random_layer_min_T": float(t.min()),
        "random_layer_max_R_plus_T": float(np.max(r + t)),
        "forward_case_count": int(len(theta_set)),
        "forward_finite": bool(np.isfinite(predictions).all()),
        "forward_min": float(predictions.min()),
        "forward_max": float(predictions.max()),
        "physical_bounds_pass": bool(predictions.min() >= -1e-12 and predictions.max() <= 1.0 + 1e-10),
        "numpy_torch_max_abs": float(np.max(np.abs(forward_preloaded_numpy(theta, optical, wavelength_nm=WAVELENGTH_NM, diameter_um=15.0, scattering_amplitude=1.1, delta_bs=0.12, g0=0.97) - torch_pred.detach().numpy()))),
        "gradient_analytic": analytic.tolist(),
        "gradient_numeric": numeric.tolist(),
        "gradient_max_abs": float(grad_abs.max()),
        "gradient_pass": bool(np.all(grad_abs <= 1e-6 + 1e-4 * np.abs(numeric))),
        "dv_zero_torch_gradient_finite": bool(torch.isfinite(theta_zero_t.grad).all()),
        "dv_zero_packaging_max_abs_from_one": float(np.max(np.abs(package_zero - 1.0))),
        "dv_small_packaging_near_one": bool(abs(package_small - 1.0) < 1e-6),
        "dv_zero_forward_vs_v1_max_abs": float(np.max(np.abs(no_pack - v1))),
        "two_parameter_fixed_s0_max_abs": float(np.max(np.abs(two_theta - v2))),
        "identity_scattering_max_abs": float(np.max(np.abs(identity_scattering - v2))),
        "identity_gain_max_abs": float(np.max(np.abs(identity_gain - v2))),
        "g0_high_finite": bool(np.isfinite(high_gain_predictions).all()),
        "g0_high_above_one_value_count": int(np.count_nonzero(high_gain_predictions > 1.0)),
        "g0_high_contract_audit": high_gain_audit,
        "shape_profile_best": {"A_s": shape_profile["A_s"], "delta_bs": shape_profile["delta_bs"]},
        "shape_profile_count": len(shape_profile["profile"]),
        "gain_estimate": gain_profile,
        "global_profile_identifiability": {"A_s": as_identifiability, "delta_bs": delta_identifiability, "g0": gain_identifiability},
        "bounds": {"A_s": list(AS_BOUNDS), "delta_bs": list(DELTA_BS_BOUNDS), "g0": list(G0_BOUNDS)},
        "asset_sha256": _sha256(ASSET),
        "source_sha256": _sha256(SOURCE),
        "contract_sha256": _sha256(CONTRACT),
        "test_sha256": _sha256(TEST),
        "audit_script_sha256": _sha256(AUDIT_SCRIPT),
    }
    checks = {
        "finite_layer_degeneracy": result["published_vs_stable_max_abs"] < 1e-10,
        "packaging_zero_diameter_limit": result["dv_zero_packaging_max_abs_from_one"] == 0.0 and result["dv_zero_forward_vs_v1_max_abs"] < 1e-12,
        "fixed_s0_interface": result["two_parameter_fixed_s0_max_abs"] < 1e-12,
        "scattering_identity_limit": result["identity_scattering_max_abs"] == 0.0,
        "observation_scale_identity_limit": result["identity_gain_max_abs"] == 0.0 and result["g0_high_finite"] and not high_gain_audit["pass"] and not high_gain_audit["clipped"],
        "physical_bounds": result["random_layer_finite"] and result["random_layer_max_R_plus_T"] <= 1 + 1e-10 and result["physical_bounds_pass"],
        "numpy_torch_parity": result["numpy_torch_max_abs"] < 1e-10,
        "gradient_finite_difference": result["gradient_pass"] and result["dv_zero_torch_gradient_finite"],
        "sequential_scale_interface_bounds": result["shape_profile_best"] == {"A_s": 1.1, "delta_bs": 0.12} and G0_BOUNDS[0] <= gain_profile["g0"] <= G0_BOUNDS[1],
        "global_profile_identifiability_rule": bool(as_identifiability["identifiable"] and delta_identifiability["identifiable"] and gain_identifiability["identifiable"]),
    }
    result["checks"] = {name: bool(value) for name, value in checks.items()}
    result["status"] = "PASS" if all(checks.values()) else "FAIL"
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "audit_summary.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    (OUT / "audit_summary.md").write_text("# KM-BIO-v2R formula audit\n\nNo HSI observations were read.\n\n```json\n" + json.dumps(result, indent=2) + "\n```\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    if result["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
