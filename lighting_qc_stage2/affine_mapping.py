from __future__ import annotations

import numpy as np
from PIL import Image


def validate_inverse(source_to_canvas: np.ndarray, canvas_to_source: np.ndarray, atol: float = 1e-5) -> float:
    a = np.asarray(source_to_canvas, dtype=np.float64)
    b = np.asarray(canvas_to_source, dtype=np.float64)
    if a.shape != (3, 3) or b.shape != (3, 3) or not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError("affine matrices must be finite 3x3")
    err = float(np.max(np.abs(a @ b - np.eye(3))))
    if err > atol:
        raise ValueError(f"affine matrices are not inverse; max error={err}")
    return err


def _pil_coeffs(matrix_output_to_input: np.ndarray) -> tuple[float, float, float, float, float, float]:
    m = np.asarray(matrix_output_to_input, dtype=np.float64)
    return tuple(float(x) for x in (m[0, 0], m[0, 1], m[0, 2], m[1, 0], m[1, 1], m[1, 2]))


def warp_mask(mask: np.ndarray, output_size_wh: tuple[int, int], output_to_input: np.ndarray) -> np.ndarray:
    im = Image.fromarray((mask > 0).astype(np.uint8) * 255, mode="L")
    warped = im.transform(output_size_wh, Image.Transform.AFFINE, _pil_coeffs(output_to_input), resample=Image.Resampling.NEAREST, fillcolor=0)
    return (np.asarray(warped, dtype=np.uint8) > 0).astype(np.uint8) * 255


def canvas_to_source_mask(canvas_mask: np.ndarray, source_size_wh: tuple[int, int], source_to_canvas: np.ndarray) -> np.ndarray:
    return warp_mask(canvas_mask, source_size_wh, source_to_canvas)


def source_to_canvas_mask(source_mask: np.ndarray, canvas_size_wh: tuple[int, int], canvas_to_source: np.ndarray) -> np.ndarray:
    return warp_mask(source_mask, canvas_size_wh, canvas_to_source)


def mask_iou(a: np.ndarray, b: np.ndarray) -> float:
    aa = a > 0
    bb = b > 0
    union = aa | bb
    if not union.any():
        return float("nan")
    return float((aa & bb).sum() / union.sum())


def centroid(mask: np.ndarray) -> tuple[float, float] | None:
    ys, xs = np.nonzero(mask > 0)
    if len(xs) == 0:
        return None
    return float(xs.mean()), float(ys.mean())


def centroid_distance(a: np.ndarray, b: np.ndarray) -> float | None:
    ca = centroid(a)
    cb = centroid(b)
    if ca is None or cb is None:
        return None
    return float(np.hypot(ca[0] - cb[0], ca[1] - cb[1]))
