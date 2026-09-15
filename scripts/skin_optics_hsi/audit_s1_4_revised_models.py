"""Run the S1-4R revised-model synthetic audit (no Hyper-Skin reads)."""

from __future__ import annotations

import argparse
import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.skin_optics_hsi.s1_proxy_inverse import (
    fit_group_isolated_pca, fit_proxy_wls, fit_semi_mechanistic_least_squares,
)
from src.skin_optics_hsi.s1_revised_forward import (
    RevisedSkinForwardModel, file_sha256, load_revised_registry, packaging_factor, revised_basis,
    revised_skin_forward,
)


def _assert_new(path: Path) -> None:
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite existing audit output: {path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--supersedes-decision")
    args = parser.parse_args()
    output = Path(args.output_dir).resolve()
    _assert_new(output)
    output.mkdir(parents=True)
    registry = load_revised_registry()
    wave = registry.wavelength_nm
    weights = np.ones(wave.size)
    ref = 0.48 + 0.035 * np.cos((wave - 550.0) / 90.0)
    gp = {"weights": weights, "reference_reflectance": ref, "a_obs": 0.17}
    checks: dict[str, dict] = {}

    # Packaging limits and monotonicity.
    mu = np.array([0.0, 0.1, 1.0, 4.0, 1.0e6])
    p15 = packaging_factor(mu, 15.0)
    d0 = packaging_factor(mu, 0.0)
    assert np.array_equal(d0, np.ones_like(mu))
    assert np.isclose(p15[0], 1.0) and np.all((p15 > 0) & (p15 <= 1.0)) and np.all(np.diff(p15) < 0)
    strong_limit_error = float(abs(p15[-1] * mu[-1] - 1.0 / 0.015))
    assert strong_limit_error < 1e-10
    checks["packaging"] = {"pass": True, "values": p15.tolist(), "D0_values": d0.tolist(), "D0_limit": "1",
                            "strong_absorption_C_times_mu_limit_error": strong_limit_error}

    basis = revised_basis(wave, RevisedSkinForwardModel("D2-MH", registry).assets, weights=weights)
    X2 = np.column_stack((-basis["phi_M"], -basis["phi_H"]))
    s2 = np.linalg.svd(X2, compute_uv=False)
    checks["basis_rank_condition"] = {"pass": bool(np.linalg.matrix_rank(X2) == 2), "rank": int(np.linalg.matrix_rank(X2)),
                                      "condition": float(s2[0] / s2[-1]), "singular_values": s2.tolist()}

    # D2 exact synthetic recovery, including a nuisance amplitude that should be removed by centering.
    theta_true = np.array([0.37, -0.24])
    d2 = RevisedSkinForwardModel("D2-MH", registry)
    observed = d2.forward_numpy(theta_true, wave, gp, observation_space="raw_reflectance")
    fit = fit_proxy_wls("D2-MH", observed, ref, registry=registry, weights=weights)
    recovery_error = float(np.max(np.abs(fit.theta - theta_true)))
    assert recovery_error < 1e-7 and fit.weighted_rmse < 1e-9
    checks["D2_recovery"] = {"pass": True, "theta_true": theta_true.tolist(), "theta_fit": fit.theta.tolist(),
                              "max_abs_error": recovery_error, "weighted_rmse": fit.weighted_rmse,
                              "rank": fit.rank, "singular_values": fit.singular_values.tolist()}

    from scipy.optimize import least_squares
    target_shape = d2.forward_numpy(theta_true, wave, gp, observation_space="centered_log_ratio")
    numeric = least_squares(lambda t: d2.forward_numpy(t, wave, gp, observation_space="centered_log_ratio") - target_shape,
                            np.zeros(2), method="lm")
    analytic_numeric_error = float(np.max(np.abs(fit.theta - numeric.x)))
    assert numeric.success and analytic_numeric_error < 1e-7
    checks["D2_analytic_numeric_parity"] = {"pass": True, "max_abs_parameter_error": analytic_numeric_error}

    subset_wave = wave[1:-1]
    subset_ref = ref[1:-1]
    subset_observed = d2.forward_numpy(theta_true, subset_wave, {"reference_reflectance": subset_ref})
    subset_fit = fit_proxy_wls("D2-MH", subset_observed, subset_ref, registry=registry, wavelength_nm=subset_wave)
    subset_error = float(np.max(np.abs(subset_fit.theta - theta_true)))
    assert subset_error < 1e-7
    checks["registered_band_subset"] = {"pass": True, "bands": int(subset_wave.size),
                                         "max_abs_parameter_error": subset_error}

    # NumPy/Torch equation parity and gradient sanity for D2 and K2.
    parity: dict[str, float] = {}
    gradient_error: dict[str, float] = {}
    for model_id, theta in [("D2-MH", theta_true), ("K2-MH-KM", np.array([0.13, 0.011]))]:
        model = RevisedSkinForwardModel(model_id, registry)
        gp_t = dict(gp)
        if model_id.startswith("K2"):
            gp_t.pop("reference_reflectance", None)
        np_out = model.forward_numpy(theta, wave, gp_t)
        t = torch.tensor(theta, dtype=torch.float64, requires_grad=True)
        torch_out = model.forward_torch(t, wave, gp_t)
        err = float(np.max(np.abs(np_out - torch_out.detach().cpu().numpy())))
        torch_out.sum().backward()
        step = 1e-6
        fd = np.empty(theta.size)
        for j in range(theta.size):
            plus, minus = theta.copy(), theta.copy()
            plus[j] += step
            minus[j] -= step
            fd[j] = (model.forward_numpy(plus, wave, gp_t).sum() - model.forward_numpy(minus, wave, gp_t).sum()) / (2 * step)
        grad_err = float(np.max(np.abs(fd - t.grad.detach().cpu().numpy())))
        assert err < 1e-10 and torch.all(torch.isfinite(t.grad)) and grad_err < 1e-6
        parity[model_id] = err
        gradient_error[model_id] = grad_err
    checks["numpy_torch_parity"] = {"pass": True, "max_abs_error": parity,
                                     "autograd_finite_difference_max_abs_error": gradient_error}

    # Every registered K2/K3 boundary corner must remain in the valid K--M domain.
    corner_counts = {}
    min_s = np.inf
    for model_id in ("K2-MH-KM", "K3-MHS-KM"):
        model = RevisedSkinForwardModel(model_id, registry)
        bounds = model.bounds
        corners = np.array(np.meshgrid(*[(lo, hi) for lo, hi in bounds])).T.reshape(-1, len(bounds))
        for theta in corners:
            pred, comp = model.forward_numpy(theta, wave, return_components=True)
            assert np.all(np.isfinite(pred)) and np.all(pred > 0) and np.all(comp["S_KM"] > 0)
            min_s = min(min_s, float(np.min(comp["S_KM"])))
        corner_counts[model_id] = int(corners.shape[0])
    checks["KM_boundary_corners"] = {"pass": True, "corner_counts": corner_counts,
                                      "minimum_S_KM_mm1": float(min_s)}

    # K2/K3 bounded synthetic fit should be finite and reproducible.
    k2 = RevisedSkinForwardModel("K2-MH-KM", registry)
    k2_true = np.array([0.21, 0.018])
    k2_obs = k2.forward_numpy(k2_true, wave)
    k2_fit = fit_semi_mechanistic_least_squares("K2-MH-KM", k2_obs, registry=registry)
    assert k2_fit["success"] and np.all(np.isfinite(k2_fit["theta"])) and np.all(np.isfinite(k2_fit["residual"]))
    checks["K2_synthetic_fit"] = {"pass": True, "theta_true": k2_true.tolist(), "theta_fit": k2_fit["theta"].tolist(),
                                   "log_rmse": float(np.sqrt(np.mean(k2_fit["residual"] ** 2))), "nfev": k2_fit["nfev"]}
    unpackaged_k2 = k2.forward_numpy(k2_true, wave, {"vessel_diameter_um": 0.0})
    assert np.all(np.isfinite(unpackaged_k2)) and np.all(unpackaged_k2 > 0)
    checks["K2_unpacked_Hb_limit"] = {"pass": True, "reflectance_min": float(unpackaged_k2.min()),
                                       "reflectance_max": float(unpackaged_k2.max())}

    # PCA fitting records and excludes the held-out subject by construction.
    rng = np.random.default_rng(20260908)
    pca_x = rng.normal(size=(8, 31))
    pca_ids = np.array(["s1", "s1", "s2", "s2", "s3", "s3", "held", "held"])
    pca_x[pca_ids == "held"] += 100.0
    pca = fit_group_isolated_pca(pca_x, pca_ids, held_out_subject_id="held")
    assert "held" not in pca.training_subject_ids and pca.components.shape == (31, 2)
    checks["PCA_group_isolation"] = {"pass": True, "held_out_subject": "held",
                                      "training_subject_ids": list(pca.training_subject_ids)}

    interface = revised_skin_forward(theta_bio=theta_true, theta_nuisance=None, wavelength_nm=wave,
                                     global_params=gp, model_id="D2-MH", registry=registry)
    assert interface["reflectance"].shape == (31,) and interface["centered_log_ratio"].shape == (31,)
    assert interface["evidence_tier"] == "proxy_or_baseline" and "basis" in interface["components"]
    checks["public_interface"] = {"pass": True, "keys": sorted(interface)}

    # Deferred strict RTE path must fail closed.
    try:
        RevisedSkinForwardModel("T2-MH-RTE", registry).forward_numpy(np.empty(0), wave)
    except RuntimeError as exc:
        checks["RTE_deferred"] = {"pass": True, "message": str(exc)}
    else:
        raise AssertionError("T2-MH-RTE did not fail closed")

    provenance = {
        "registry_id": registry.raw["registry_id"], "registry_path": str(registry.path),
        "registry_sha256": file_sha256(registry.path), "asset_sha256": file_sha256(registry.asset_path),
        "srf_sensitivity_sha256": file_sha256(registry.srf_sensitivity_path),
        "wavelength_nm": wave.tolist(), "synthetic_only": True,
        "validation_rows_read": 0, "test_rows_read": 0, "raw_hsi_files_read": 0,
        "python": platform.python_version(), "numpy": np.__version__, "torch": torch.__version__,
        "source_sha256": {
            "s1_revised_forward.py": file_sha256(ROOT / "src/skin_optics_hsi/s1_revised_forward.py"),
            "s1_proxy_inverse.py": file_sha256(ROOT / "src/skin_optics_hsi/s1_proxy_inverse.py"),
            "audit_s1_4_revised_models.py": file_sha256(Path(__file__).resolve()),
            "test_s1_revised_models.py": file_sha256(ROOT / "tests/skin_optics_hsi/test_s1_revised_models.py"),
        },
    }
    (output / "synthetic_audit_checks.json").write_text(json.dumps(checks, indent=2), encoding="utf-8")
    (output / "provenance.json").write_text(json.dumps(provenance, indent=2), encoding="utf-8")
    decision = {
        "schema_version": 1, "stage": "S1-4R", "audit_id": "s1_4_revised_v3_synthetic",
        "status": "PASS_FOR_REVISED_TRAIN_DEVELOPMENT", "next_stage_allowed": True,
        "authorized_next_stage": "S1-5R", "s1_6_allowed": False,
        "models_registered": sorted(registry.models),
        "models_implemented": sorted(k for k, v in registry.models.items() if v.family != "deferred_rte"),
        "models_deferred": ["T2-MH-RTE"], "strict_rte_enabled": False,
        "checks": {k: bool(v["pass"]) for k, v in checks.items()}, "synthetic_only": True,
        "validation_rows_read": 0, "test_rows_read": 0, "raw_hsi_files_read": 0,
        "outputs": {}, "created_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    if args.supersedes_decision:
        previous = Path(args.supersedes_decision).resolve()
        if not previous.is_file():
            raise FileNotFoundError(f"Superseded decision does not exist: {previous}")
        decision["supersedes"] = {"path": str(previous), "sha256": file_sha256(previous)}
    for p in sorted(output.glob("*.json")):
        decision["outputs"][p.name] = file_sha256(p)
    (output / "s1_4r_decision.json").write_text(json.dumps(decision, indent=2), encoding="utf-8")
    print(json.dumps({"status": decision["status"], "output_dir": str(output), "checks": decision["checks"]}, indent=2))


if __name__ == "__main__":
    main()
