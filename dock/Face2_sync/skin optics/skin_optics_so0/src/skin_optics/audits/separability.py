"""M/H separability audits."""

from __future__ import annotations

import numpy as np

from skin_optics.assets import SpectralAssets
from skin_optics.numpy_backend.colorchecker import CalibrationRecord
from skin_optics.numpy_backend.image_formation import render_camera
from skin_optics.numpy_backend.reflectance import compute_skin_reflectance


def _finite_diff_reflectance(assets: SpectralAssets, m: float, h: float, delta: float = 1e-3) -> tuple[np.ndarray, np.ndarray]:
    jm = (compute_skin_reflectance(assets, m + delta, h) - compute_skin_reflectance(assets, m - delta, h)) / (2.0 * delta)
    jh = (compute_skin_reflectance(assets, m, h + delta) - compute_skin_reflectance(assets, m, h - delta)) / (2.0 * delta)
    return jm, jh


def audit_spectral_separability(assets: SpectralAssets) -> dict[str, float | str]:
    """Audit spectral Jacobian cosine similarity."""

    grid = [0.20, 0.35, 0.50, 0.65, 0.80]
    cosines = []
    for m in grid:
        for h in grid:
            jm, jh = _finite_diff_reflectance(assets, m, h)
            denom = np.linalg.norm(jm) * np.linalg.norm(jh)
            cosines.append(float(abs(np.dot(jm, jh) / denom)))
    arr = np.asarray(cosines, dtype=np.float64)
    status = "PASS" if np.median(arr) <= 0.95 and np.percentile(arr, 95) <= 0.98 else "FAIL"
    return {"median": float(np.median(arr)), "p95": float(np.percentile(arr, 95)), "status": status}


def audit_observation_separability(assets: SpectralAssets, records: list[CalibrationRecord]) -> dict[str, float | str]:
    """Audit 2D observation Jacobian rank in all required observation spaces."""

    grid = [0.20, 0.35, 0.50, 0.65, 0.80]
    delta = 1e-3
    spaces = ["camera_rgb_wb", "linear_srgb_unclipped", "chromaticity"]
    rank2 = {space: 0 for space in spaces}
    total = {space: 0 for space in spaces}

    def chromaticity(rgb: np.ndarray) -> np.ndarray:
        return rgb / (np.sum(rgb, axis=-1, keepdims=True) + 1e-12)

    def outputs(reflectance: np.ndarray, rec: CalibrationRecord) -> dict[str, np.ndarray]:
        out = render_camera(assets, reflectance, rec.matrix_3x3, rec.camera_name, rec.illuminant_name)
        return {
            "camera_rgb_wb": out.camera_rgb_wb,
            "linear_srgb_unclipped": out.linear_srgb_unclipped,
            "chromaticity": chromaticity(out.linear_srgb_unclipped),
        }

    for rec in records:
        if rec.qualification_status != "PASS":
            continue
        for m in grid:
            for h in grid:
                plus_m = outputs(compute_skin_reflectance(assets, m + delta, h), rec)
                minus_m = outputs(compute_skin_reflectance(assets, m - delta, h), rec)
                plus_h = outputs(compute_skin_reflectance(assets, m, h + delta), rec)
                minus_h = outputs(compute_skin_reflectance(assets, m, h - delta), rec)
                for space in spaces:
                    j = np.stack(
                        [(plus_m[space] - minus_m[space]) / (2.0 * delta), (plus_h[space] - minus_h[space]) / (2.0 * delta)],
                        axis=-1,
                    )
                    s = np.linalg.svd(j, compute_uv=False)
                    total[space] += 1
                    if s[0] > 1e-10 and s[-1] / s[0] >= 1e-3:
                        rank2[space] += 1
    fractions = {space: float(rank2[space] / total[space]) if total[space] else 0.0 for space in spaces}
    frac = min(fractions.values())
    status = "PASS" if frac >= 0.95 else "PASS_WITH_LIMITS" if frac >= 0.80 else "FAIL"
    return {
        "rank2_fraction": frac,
        "camera_rgb_wb_rank2_fraction": fractions["camera_rgb_wb"],
        "linear_srgb_unclipped_rank2_fraction": fractions["linear_srgb_unclipped"],
        "chromaticity_rank2_fraction": fractions["chromaticity"],
        "rank2_count": float(sum(rank2.values())),
        "total_count": float(sum(total.values())),
        "status": status,
    }
