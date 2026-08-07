"""Virtual ColorChecker camera matrix calibration."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from skin_optics.assets import SpectralAssets
from skin_optics.numpy_backend.color_spaces import bradford_adapt, deltae00, xyz_to_lab_d65, xyz_to_linear_srgb
from skin_optics.numpy_backend.image_formation import (
    camera_white_response,
    d65_white_xyz,
    illuminant,
    integrate_camera_rgb,
    integrate_xyz,
    source_white_xyz,
)


@dataclass(frozen=True)
class CalibrationRecord:
    """One camera-light ColorChecker calibration result."""

    camera_name: str
    illuminant_name: str
    matrix_rank: int
    condition_number: float
    matrix_3x3: np.ndarray
    training_deltae00_median: float
    training_deltae00_p95: float
    loocv_deltae00_median: float
    loocv_deltae00_p95: float
    negative_linear_srgb_rate: float
    out_of_gamut_rate: float
    finite: bool
    qualification_status: str
    failure_reasons: str


def colorchecker_camera_rgb_wb(assets: SpectralAssets, camera_name: str, illuminant_name: str) -> np.ndarray:
    """Return 24x3 white-balanced camera responses for ColorChecker patches."""

    rad = illuminant(assets, illuminant_name)[None, :] * assets.source["colorchecker_reflectance"]
    raw = integrate_camera_rgb(assets, rad, camera_name)
    return raw / camera_white_response(assets, camera_name, illuminant_name)


def colorchecker_xyz_source(assets: SpectralAssets, illuminant_name: str) -> np.ndarray:
    """Return 24x3 source XYZ references for ColorChecker patches."""

    rad = illuminant(assets, illuminant_name)[None, :] * assets.source["colorchecker_reflectance"]
    return integrate_xyz(assets, rad)


def solve_camera_to_xyz_matrix(assets: SpectralAssets, camera_name: str, illuminant_name: str) -> tuple[np.ndarray, int, float]:
    """Solve the no-intercept 3x3 camera-to-XYZ matrix."""

    x = colorchecker_camera_rgb_wb(assets, camera_name, illuminant_name)
    y = colorchecker_xyz_source(assets, illuminant_name)
    matrix, _, rank, _ = np.linalg.lstsq(x, y, rcond=None)
    return matrix.astype(np.float64), int(rank), float(np.linalg.cond(x))


def _deltae_for_predictions(assets: SpectralAssets, illuminant_name: str, pred_xyz: np.ndarray, ref_xyz: np.ndarray) -> np.ndarray:
    pred_d65 = bradford_adapt(pred_xyz, source_white_xyz(assets, illuminant_name), d65_white_xyz(assets))
    ref_d65 = bradford_adapt(ref_xyz, source_white_xyz(assets, illuminant_name), d65_white_xyz(assets))
    return deltae00(xyz_to_lab_d65(pred_d65, d65_white_xyz(assets)), xyz_to_lab_d65(ref_d65, d65_white_xyz(assets)))


def calibrate_camera_light(assets: SpectralAssets, camera_name: str, illuminant_name: str) -> CalibrationRecord:
    """Calibrate and audit one camera-light combination."""

    x = colorchecker_camera_rgb_wb(assets, camera_name, illuminant_name)
    y = colorchecker_xyz_source(assets, illuminant_name)
    matrix, _, rank, _ = np.linalg.lstsq(x, y, rcond=None)
    pred = x @ matrix
    train_de = _deltae_for_predictions(assets, illuminant_name, pred, y)
    loocv = []
    for idx in range(24):
        mask = np.ones(24, dtype=bool)
        mask[idx] = False
        m_i, _, _, _ = np.linalg.lstsq(x[mask], y[mask], rcond=None)
        loocv.append(float(_deltae_for_predictions(assets, illuminant_name, x[idx : idx + 1] @ m_i, y[idx : idx + 1])[0]))
    loocv_arr = np.asarray(loocv, dtype=np.float64)
    pred_d65 = bradford_adapt(pred, source_white_xyz(assets, illuminant_name), d65_white_xyz(assets))
    linear = xyz_to_linear_srgb(pred_d65)
    finite = bool(np.all(np.isfinite(matrix)) and np.all(np.isfinite(train_de)) and np.all(np.isfinite(loocv_arr)))
    condition = float(np.linalg.cond(x))
    failures: list[str] = []
    if int(rank) != 3:
        failures.append("matrix_rank_not_3")
    if not finite:
        failures.append("non_finite")
    if condition > 1.0e6:
        failures.append("condition_number_gt_1e6")
    if float(np.median(loocv_arr)) > 4.0:
        failures.append("loocv_median_gt_4")
    if float(np.percentile(loocv_arr, 95)) > 8.0:
        failures.append("loocv_p95_gt_8")
    return CalibrationRecord(
        camera_name=camera_name,
        illuminant_name=illuminant_name,
        matrix_rank=int(rank),
        condition_number=condition,
        matrix_3x3=matrix.astype(np.float64),
        training_deltae00_median=float(np.median(train_de)),
        training_deltae00_p95=float(np.percentile(train_de, 95)),
        loocv_deltae00_median=float(np.median(loocv_arr)),
        loocv_deltae00_p95=float(np.percentile(loocv_arr, 95)),
        negative_linear_srgb_rate=float(np.mean(linear < 0.0)),
        out_of_gamut_rate=float(np.mean((linear < 0.0) | (linear > 1.0))),
        finite=finite,
        qualification_status="PASS" if not failures else "FAIL",
        failure_reasons=";".join(failures),
    )


def calibrate_all(assets: SpectralAssets) -> list[CalibrationRecord]:
    """Calibrate all 28 camera x 4 core illuminant combinations."""

    records = []
    for light in map(str, assets.source["core_illuminant_names"]):
        for camera in map(str, assets.source["camera_names"]):
            records.append(calibrate_camera_light(assets, camera, light))
    return records
