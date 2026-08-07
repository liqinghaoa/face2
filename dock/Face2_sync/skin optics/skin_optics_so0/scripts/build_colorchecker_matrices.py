"""Build fixed Virtual ColorChecker matrices for all core camera-light pairs."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import pandas as pd

from skin_optics.assets import load_assets
from skin_optics.numpy_backend.colorchecker import calibrate_all


WORKSPACE = Path("/mnt/e/projects/face2")
OUT = WORKSPACE / "outputs/SO0_Forward_Model_v1/calibration"


def main() -> None:
    """Build and save ColorChecker matrices and metrics."""

    OUT.mkdir(parents=True, exist_ok=True)
    assets = load_assets(5, WORKSPACE)
    records = calibrate_all(assets)
    matrices = np.stack([rec.matrix_3x3 for rec in records])
    cameras = np.asarray([rec.camera_name for rec in records])
    lights = np.asarray([rec.illuminant_name for rec in records])
    np.savez_compressed(OUT / "colorchecker_matrices.npz", matrices=matrices, camera_names=cameras, illuminant_names=lights)
    rows = []
    for rec in records:
        rows.append(
            {
                "camera_name": rec.camera_name,
                "illuminant_name": rec.illuminant_name,
                "matrix_rank": rec.matrix_rank,
                "condition_number": rec.condition_number,
                "matrix_3x3": " ".join(f"{x:.12g}" for x in rec.matrix_3x3.ravel()),
                "training_deltae00_median": rec.training_deltae00_median,
                "training_deltae00_p95": rec.training_deltae00_p95,
                "loocv_deltae00_median": rec.loocv_deltae00_median,
                "loocv_deltae00_p95": rec.loocv_deltae00_p95,
                "negative_linear_srgb_rate": rec.negative_linear_srgb_rate,
                "out_of_gamut_rate": rec.out_of_gamut_rate,
                "finite": rec.finite,
                "qualification_status": rec.qualification_status,
                "failure_reasons": rec.failure_reasons,
            }
        )
    pd.DataFrame(rows).to_csv(OUT / "colorchecker_calibration_metrics.csv", index=False)


if __name__ == "__main__":
    main()
