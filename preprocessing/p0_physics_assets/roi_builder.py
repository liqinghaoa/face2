"""Full-canvas P0 ROI geometry/effective-mask construction."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import numpy as np

from .mask_builder import binary_mask
from .types import RoiConfig


def rectangle_mask(shape: tuple[int, int], bbox: Any) -> np.ndarray:
    """Rasterize a half-open bbox in the unchanged 224x224 coordinate frame."""
    output = np.zeros(shape, dtype=np.uint8); output[bbox.y1:bbox.y2, bbox.x1:bbox.x2] = 255
    return output


def effective_mask(name: str, geometry: np.ndarray, label_map: np.ndarray, source_valid: np.ndarray, face_valid: np.ndarray, skin_strict: np.ndarray) -> np.ndarray:
    """Apply P0 ROI semantics without crop/resize/padding or new FaceMesh calls."""
    if name in {"left_cheek", "right_cheek", "combined_cheek", "forehead", "chin"}: value = (geometry > 0) & (skin_strict > 0)
    elif name == "lip": value = (geometry > 0) & np.isin(label_map, (12,13)) & (source_valid > 0)
    elif name == "eye": value = (geometry > 0) & (face_valid > 0)
    else: raise ValueError(f"unsupported ROI {name}")
    return binary_mask(value)


def build_roi_masks(label_map: np.ndarray, final_face_mask: np.ndarray, source_valid: np.ndarray, face_valid: np.ndarray, skin_strict: np.ndarray, landmarks: np.ndarray, config: RoiConfig) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], dict[str, dict[str, Any]]]:
    """Reuse legacy bbox definitions only; output full-canvas geometry and effects."""
    from preprocessing import preprocess_global_aligned_face_parsing_roi_dataset_224_canvas as legacy
    args = SimpleNamespace(min_roi_width=config.min_roi_width, min_roi_height=config.min_roi_height)
    try:
        left, right, _ = legacy.define_cheek_pair_bboxes(label_map, final_face_mask, landmarks, args)
        boxes = {"left_cheek":left, "right_cheek":right, "forehead":legacy.define_forehead_roi(label_map, final_face_mask, landmarks, args), "lip":legacy.define_lip_roi(label_map, landmarks, args), "eye":legacy.define_eye_roi(label_map, final_face_mask, landmarks, args), "chin":legacy.define_chin_roi(label_map, final_face_mask, landmarks, args)}
    except legacy.RoiFailure as exc: raise RuntimeError(f"roi_bbox_failure:{exc.reason}") from exc
    shape = label_map.shape; geometry = {name: rectangle_mask(shape, box) for name, box in boxes.items()}
    geometry["combined_cheek"] = binary_mask((geometry["left_cheek"] > 0) | (geometry["right_cheek"] > 0))
    effects = {name: effective_mask(name, geometry[name], label_map, source_valid, face_valid, skin_strict) for name in geometry}
    metrics: dict[str, dict[str, Any]] = {}
    for name, mask in geometry.items():
        g = mask > 0; e = effects[name] > 0; box = boxes.get(name)
        metric = {"geometry_pixels":int(g.sum()), "effective_pixels":int(e.sum()), "effective_fraction":float(e.sum()/max(1,g.sum())), "skin_fraction":float(((label_map == 1)&g).sum()/max(1,g.sum())), "source_valid_fraction":float(((source_valid > 0)&g).sum()/max(1,g.sum())), "roi_status":"success", "failure_reason":"", "warning_flags":""}
        if box is not None: metric.update({"bbox_x1":box.x1,"bbox_y1":box.y1,"bbox_x2":box.x2,"bbox_y2":box.y2})
        metrics[name] = metric
    forehead_g = geometry["forehead"] > 0; forehead_face = forehead_g & (face_valid > 0)
    fraction_g = float(((skin_strict > 0)&forehead_g).sum()/max(1,forehead_g.sum())); metrics["forehead"].update({"forehead_skin_fraction_geometry":fraction_g, "forehead_skin_fraction_face":float(((skin_strict > 0)&forehead_face).sum()/max(1,forehead_face.sum())), "forehead_available":fraction_g >= config.forehead_valid_skin_threshold})
    return geometry, effects, metrics
