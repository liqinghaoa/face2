from __future__ import annotations

import cv2
import numpy as np

from . import face_parser, facemesh_regions as regions
from .schemas import SampleFailure, SkinMaskConfig


SKIN_CANDIDATE_NAMES = ("skin", "nose")
FACE_VALID_KEEP_NAMES = (
    "skin",
    "nose",
    "left_brow",
    "right_brow",
    "left_eye",
    "right_eye",
    "mouth",
    "upper_lip",
    "lower_lip",
)
FACE_VALID_EXCLUDE_NAMES = (
    "eye_glasses",
    "hair",
    "left_ear",
    "right_ear",
    "earring",
    "neck",
    "necklace",
    "cloth",
    "hat",
    "background",
)
SEMANTIC_EXCLUSION_NAMES = (
    "left_brow",
    "right_brow",
    "left_eye",
    "right_eye",
    "eye_glasses",
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
)


def binary_uint8(mask: np.ndarray) -> np.ndarray:
    return (mask > 0).astype(np.uint8) * 255


def fill_polygons(shape: tuple[int, int], polygons: dict[str, tuple[int, ...]], landmarks: np.ndarray) -> dict[str, np.ndarray]:
    masks: dict[str, np.ndarray] = {}
    for name, indices in polygons.items():
        points = regions.points_for(indices, landmarks)
        mask = np.zeros(shape, dtype=np.uint8)
        cv2.fillPoly(mask, [np.rint(points).astype(np.int32)], 255)
        masks[name] = mask
    return masks


def dilate_binary(mask: np.ndarray, radius: int) -> np.ndarray:
    if radius <= 0:
        return binary_uint8(mask)
    size = radius * 2 + 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))
    return cv2.dilate(binary_uint8(mask), kernel, iterations=1)


def remove_small_components(mask: np.ndarray, min_area: int) -> np.ndarray:
    binary = (mask > 0).astype(np.uint8)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    if count <= 1:
        return binary_uint8(binary)
    keep = np.zeros_like(binary)
    for label in range(1, count):
        if int(stats[label, cv2.CC_STAT_AREA]) >= min_area:
            keep[labels == label] = 1
    return keep.astype(np.uint8) * 255


def build_skin_valid_mask(
    parsing_label: np.ndarray,
    canvas_landmarks: np.ndarray,
    source_valid_mask: np.ndarray,
    aligned_face_width: float,
    cfg: SkinMaskConfig,
) -> dict[str, np.ndarray | float | int | dict[str, np.ndarray]]:
    if parsing_label.shape != source_valid_mask.shape:
        raise SampleFailure("invalid_output_shape", "parsing and source-valid masks differ in shape")
    skin_ids = face_parser.class_ids(SKIN_CANDIDATE_NAMES)
    exclusion_ids = face_parser.class_ids(SEMANTIC_EXCLUSION_NAMES)
    skin_candidate = binary_uint8(np.isin(parsing_label, skin_ids))
    semantic_exclusion = binary_uint8(np.isin(parsing_label, exclusion_ids))
    polygon_masks = fill_polygons(parsing_label.shape, regions.FACEMESH_EXCLUSION_POLYGONS, canvas_landmarks)
    normal_radius_px = max(1, int(round(float(aligned_face_width) * cfg.normal_dilation_ratio())))
    nostril_radius_px = max(1, int(round(float(aligned_face_width) * cfg.nostril_dilation_ratio)))
    normal_facemesh_exclusion = np.zeros(parsing_label.shape, dtype=np.uint8)
    nostril_exclusion = np.zeros(parsing_label.shape, dtype=np.uint8)
    for name, mask in polygon_masks.items():
        if "nostril" in name:
            nostril_exclusion = cv2.bitwise_or(nostril_exclusion, mask)
        else:
            normal_facemesh_exclusion = cv2.bitwise_or(normal_facemesh_exclusion, mask)
    semantic_exclusion = dilate_binary(semantic_exclusion, normal_radius_px)
    normal_facemesh_exclusion = dilate_binary(normal_facemesh_exclusion, normal_radius_px)
    nostril_exclusion = dilate_binary(nostril_exclusion, nostril_radius_px)
    facemesh_exclusion = cv2.bitwise_or(normal_facemesh_exclusion, nostril_exclusion)
    final = (
        (skin_candidate > 0)
        & (semantic_exclusion == 0)
        & (facemesh_exclusion == 0)
        & (source_valid_mask > 0)
    )
    min_area = max(1, int(round(final.size * cfg.min_component_area_ratio)))
    final_mask = remove_small_components(final.astype(np.uint8) * 255, min_area)
    if int((final_mask > 0).sum()) == 0:
        raise SampleFailure("empty_skin_mask", "strict skin_valid_mask is empty")
    return {
        "skin_valid_mask": final_mask,
        "skin_candidate": skin_candidate,
        "semantic_exclusion": semantic_exclusion,
        "facemesh_exclusion": facemesh_exclusion,
        "facemesh_exclusion_without_nostril": normal_facemesh_exclusion,
        "nostril_exclusion": nostril_exclusion,
        "polygon_masks": polygon_masks,
        "exclusion_dilation_radius_px": normal_radius_px,
        "eye_brow_lip_dilation_radius_px": normal_radius_px,
        "nostril_dilation_radius_px": nostril_radius_px,
        "nostril_exclusion_area_ratio": float((nostril_exclusion > 0).sum() / nostril_exclusion.size),
        "min_component_area_px": min_area,
    }


def assert_mask_contract(mask: np.ndarray, source_valid_mask: np.ndarray, expected_shape: tuple[int, int]) -> None:
    if mask.shape != expected_shape or source_valid_mask.shape != expected_shape:
        raise SampleFailure("invalid_output_shape", "mask shape does not match 1280x1024")
    for name, value in (("skin_valid_mask", mask), ("source_valid_mask", source_valid_mask)):
        unique = set(np.unique(value).astype(int).tolist())
        if not unique.issubset({0, 255}):
            raise SampleFailure("invalid_output_shape", f"{name} is not binary 0/255: {sorted(unique)}")
    if np.any((mask > 0) & (source_valid_mask == 0)):
        raise SampleFailure("invalid_output_shape", "skin_valid_mask exceeds source_valid_mask")


def build_face_valid_mask(
    parsing_label: np.ndarray,
    source_valid_mask: np.ndarray,
    min_component_area_ratio: float = 0.0002,
) -> dict[str, np.ndarray | int | float]:
    """Build scheme-B face mask: keep facial features, remove hair/background."""
    if parsing_label.shape != source_valid_mask.shape:
        raise SampleFailure("invalid_output_shape", "parsing and source-valid masks differ in shape")
    keep_ids = face_parser.class_ids(FACE_VALID_KEEP_NAMES)
    exclude_ids = face_parser.class_ids(FACE_VALID_EXCLUDE_NAMES)
    keep = np.isin(parsing_label, keep_ids)
    exclude = np.isin(parsing_label, exclude_ids)
    raw_mask = keep & (~exclude) & (source_valid_mask > 0)
    min_area = max(1, int(round(raw_mask.size * float(min_component_area_ratio))))
    final_mask = remove_small_components(raw_mask.astype(np.uint8) * 255, min_area)
    if int((final_mask > 0).sum()) == 0:
        raise SampleFailure("empty_skin_mask", "scheme-B face_valid_mask is empty")
    return {
        "face_valid_mask": final_mask,
        "face_keep_mask": binary_uint8(keep),
        "face_exclude_mask": binary_uint8(exclude),
        "min_component_area_px": min_area,
        "face_valid_area_ratio": float((final_mask > 0).sum() / final_mask.size),
    }
