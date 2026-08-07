from skin_optics.audits.multicamera import colorchecker_global_status
from skin_optics.numpy_backend.colorchecker import calibrate_camera_light


def test_colorchecker_canon_d65_qualified(assets5):
    rec = calibrate_camera_light(assets5, "Canon 5DMarkII", "D65")
    assert rec.matrix_rank == 3
    assert rec.finite
    assert rec.condition_number <= 1e6
    assert rec.loocv_deltae00_median <= 4
    assert rec.loocv_deltae00_p95 <= 8


def test_colorchecker_global_smoke(assets5):
    records = [calibrate_camera_light(assets5, camera, "D65") for camera in map(str, assets5.source["camera_names"])]
    status = colorchecker_global_status(records)
    assert status["qualified"] >= 20
