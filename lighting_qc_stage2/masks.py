from __future__ import annotations

from typing import Any

import numpy as np
from scipy import ndimage as ndi


def binary(mask: np.ndarray) -> np.ndarray:
    return (mask > 0).astype(np.uint8) * 255


def remove_small_components(mask: np.ndarray, min_area: int = 16) -> np.ndarray:
    lab, n = ndi.label(mask > 0)
    if n == 0:
        return binary(mask)
    sizes = ndi.sum(np.ones_like(lab), lab, index=np.arange(1, n + 1))
    keep_labels = {i + 1 for i, s in enumerate(sizes) if s >= min_area}
    return binary(np.isin(lab, list(keep_labels)))


def erode(mask: np.ndarray, radius: int) -> np.ndarray:
    if radius <= 0:
        return binary(mask)
    y, x = np.ogrid[-radius : radius + 1, -radius : radius + 1]
    structure = (x * x + y * y) <= radius * radius
    return binary(ndi.binary_erosion(mask > 0, structure=structure))


def bbox_from_mask(mask: np.ndarray) -> dict[str, float]:
    ys, xs = np.nonzero(mask > 0)
    if len(xs) == 0:
        raise ValueError("face geometry mask is empty")
    return {
        "face_bbox_x0": int(xs.min()),
        "face_bbox_y0": int(ys.min()),
        "face_bbox_x1": int(xs.max() + 1),
        "face_bbox_y1": int(ys.max() + 1),
        "face_center_x": float(xs.mean()),
        "face_center_y": float(ys.mean()),
        "face_width": int(xs.max() - xs.min() + 1),
        "face_height": int(ys.max() - ys.min() + 1),
    }


def normalized_rect(shape: tuple[int, int], bbox: dict[str, float], spec: dict[str, float]) -> np.ndarray:
    h, w = shape
    out = np.zeros(shape, dtype=bool)
    x0 = int(round(bbox["face_bbox_x0"] + spec["x0"] * bbox["face_width"]))
    x1 = int(round(bbox["face_bbox_x0"] + spec["x1"] * bbox["face_width"]))
    y0 = int(round(bbox["face_bbox_y0"] + spec["y0"] * bbox["face_height"]))
    y1 = int(round(bbox["face_bbox_y0"] + spec["y1"] * bbox["face_height"]))
    out[max(0, y0) : min(h, y1), max(0, x0) : min(w, x1)] = True
    return out


def build_masks(parsing: np.ndarray, face_valid: np.ndarray, source_valid: np.ndarray, class_map: dict[str, int], roi_cfg: dict[str, dict[str, float]]) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    face_geometry = binary((face_valid > 0) & (source_valid > 0))
    bbox = bbox_from_mask(face_geometry)
    skin_id = class_map["skin"]
    nose_id = class_map["nose"]
    core_raw = (parsing == skin_id) & (source_valid > 0) & (face_valid > 0)
    core_e0 = remove_small_components(binary(core_raw), 16)
    core_e2 = erode(core_e0, 2)
    core_e4 = erode(core_e0, 4)
    masks: dict[str, np.ndarray] = {
        "face_geometry": face_geometry,
        "core_skin_e0": core_e0,
        "core_skin_e2": core_e2,
        "core_skin_e4": core_e4,
    }
    forehead = normalized_rect(parsing.shape, bbox, roi_cfg["forehead"]) & (core_e2 > 0)
    left = normalized_rect(parsing.shape, bbox, roi_cfg["canvas_left_cheek"]) & (core_e2 > 0)
    right = normalized_rect(parsing.shape, bbox, roi_cfg["canvas_right_cheek"]) & (core_e2 > 0)
    nose_candidate = normalized_rect(parsing.shape, bbox, roi_cfg["nose"])
    nose = (parsing == nose_id) & (source_valid > 0) & nose_candidate
    left &= ~(nose)
    right &= ~(nose)
    masks.update(
        {
            "forehead": binary(forehead),
            "canvas_left_cheek": binary(left),
            "canvas_right_cheek": binary(right),
            "nose": binary(nose),
        }
    )
    excluded = [
        "left_brow",
        "right_brow",
        "left_eye",
        "right_eye",
        "mouth",
        "upper_lip",
        "lower_lip",
        "hair",
        "left_ear",
        "right_ear",
        "earring",
        "neck",
        "necklace",
        "cloth",
        "hat",
        "background",
        "eye_glasses",
    ]
    excluded_ids = [class_map[name] for name in excluded]
    diagnostics = dict(bbox)
    diagnostics["core_skin_excluded_pixel_count"] = int(np.isin(parsing, excluded_ids)[core_e2 > 0].sum())
    diagnostics["left_right_cheek_overlap"] = int(((masks["canvas_left_cheek"] > 0) & (masks["canvas_right_cheek"] > 0)).sum())
    diagnostics["cheek_nose_overlap"] = int((((masks["canvas_left_cheek"] > 0) | (masks["canvas_right_cheek"] > 0)) & (masks["nose"] > 0)).sum())
    for name in ["core_skin_e0", "core_skin_e2", "core_skin_e4", "forehead", "canvas_left_cheek", "canvas_right_cheek", "nose"]:
        diagnostics[f"{name}_pixels"] = int((masks[name] > 0).sum())
    return masks, diagnostics
