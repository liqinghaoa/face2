"""Run the KM-BIO-v1 formula-contract audit without reading HSI data."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch
from scipy.stats import qmc

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.skin_optics_hsi.km_bio_v1 import (
    THETA_HI,
    THETA_LO,
    WAVELENGTH_NM,
    _km_layer_numpy,
    forward_numpy,
    forward_torch,
)


ASSET = ROOT / "data/external/SO0_Spectral_Assets_v1/processed/SO0_Spectral_Standardized_v1/production_10nm/derived_optics_10nm.npz"
OUT = ROOT / "outputs/skin_optics_hsi_v1/stage1_hsi_physics/km_bio_v1_formula_audit"


def _published_layer(k: np.ndarray, s: np.ndarray, d: float) -> tuple[np.ndarray, np.ndarray]:
    q = np.sqrt(k * (k + 2 * s))
    beta = np.sqrt(k / (k + 2 * s))
    den = (1 + beta) ** 2 * np.exp(q * d) - (1 - beta) ** 2 * np.exp(-q * d)
    r = (1 - beta * beta) * (np.exp(q * d) - np.exp(-q * d)) / den
    t = 4 * beta / den
    return r, t


def main() -> None:
    rng = np.random.default_rng(20260909)
    k = 10.0 ** rng.uniform(-3, 2, 10000)
    s = 10.0 ** rng.uniform(-3, 2, 10000)
    r, t = _km_layer_numpy(k, s, 0.060)
    rp, tp = _published_layer(k, s, 0.060)

    sobol = qmc.Sobol(3, scramble=True, seed=20260909).random_base2(7)
    theta_set = THETA_LO + sobol * (THETA_HI - THETA_LO)
    # Include all parameter corners for forward finite/bound checks.
    theta_set = np.vstack([theta_set, np.array(np.meshgrid(*zip(THETA_LO, THETA_HI))).T.reshape(-1, 3)])
    preds = np.vstack([forward_numpy(theta, WAVELENGTH_NM, ASSET) for theta in theta_set])

    with np.load(ASSET, allow_pickle=False) as z:
        source_w = z["wavelength_nm"]
        optical = {
            key: torch.as_tensor(np.interp(WAVELENGTH_NM, source_w, z[field] / 10.0), dtype=torch.float64)
            for key, field in [
                ("mua_mel", "mua_mel_alternative_cm1"),
                ("mua_bg", "mua_base_cm1"),
                ("mua_hbo2", "mua_hbo2_whole_blood_150gL_cm1"),
                ("mua_hb", "mua_hb_whole_blood_150gL_cm1"),
                ("musp", "musp_total_cm1"),
            ]
        }

    theta = torch.tensor([0.18, 0.025, 0.72], dtype=torch.float64, requires_grad=True)
    torch_pred = forward_torch(theta, optical)
    torch_pred.sum().backward()
    analytic = theta.grad.detach().numpy().copy()
    numeric = []
    h = 1e-5
    for i in range(3):
        plus, minus = theta.detach().numpy().copy(), theta.detach().numpy().copy()
        plus[i] += h
        minus[i] -= h
        numeric.append((forward_numpy(plus, WAVELENGTH_NM, ASSET) - forward_numpy(minus, WAVELENGTH_NM, ASSET)).sum() / (2 * h))
    numeric = np.asarray(numeric)
    grad_abs = np.abs(analytic - numeric)
    result = {
        "model_id": "KM2L-HF-v1",
        "hsi_read": False,
        "random_seed": 20260909,
        "random_layer_count": 10000,
        "published_vs_stable_max_abs": float(max(np.max(np.abs(r - rp)), np.max(np.abs(t - tp)))),
        "random_layer_finite": bool(np.isfinite(r).all() and np.isfinite(t).all()),
        "random_layer_min_R": float(r.min()),
        "random_layer_min_T": float(t.min()),
        "random_layer_max_R_plus_T": float(np.max(r + t)),
        "forward_case_count": int(len(theta_set)),
        "forward_finite": bool(np.isfinite(preds).all()),
        "forward_min": float(preds.min()),
        "forward_max": float(preds.max()),
        "numpy_torch_max_abs": float(np.max(np.abs(preds[0] - forward_torch(torch.tensor(theta_set[0], dtype=torch.float64), optical).detach().numpy()))),
        "gradient_analytic": analytic.tolist(),
        "gradient_numeric": numeric.tolist(),
        "gradient_max_abs": float(grad_abs.max()),
        "gradient_pass": bool(np.all(grad_abs <= 1e-6 + 1e-4 * np.abs(numeric))),
        "status": "PASS" if np.isfinite(r).all() and np.isfinite(t).all() and np.max(r + t) <= 1 + 1e-10 and np.max(np.abs(r - rp)) < 1e-10 and np.max(np.abs(t - tp)) < 1e-10 and np.all(grad_abs <= 1e-6 + 1e-4 * np.abs(numeric)) else "FAIL",
    }
    OUT.mkdir(parents=True, exist_ok=False)
    (OUT / "audit_summary.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    (OUT / "audit_summary.md").write_text("# KM-BIO-v1 formula audit\n\n```json\n" + json.dumps(result, indent=2) + "\n```\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
