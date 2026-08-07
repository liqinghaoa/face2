"""Original-coordinate alignment adapter for the immutable Global implementation."""

from __future__ import annotations

from typing import Any

import numpy as np

from .types import AlignmentArtifacts


def align_original_scene_once(image_rgb: np.ndarray, selected_face: Any, face_mesh: Any, image_size: int) -> AlignmentArtifacts:
    """Make one FaceMesh call on a detection crop then directly warp the full scene.

    The 2x3 similarity transform is estimated in original-image coordinates.
    A white original-space raster uses that same transform with nearest-neighbour
    interpolation, defining pixels whose aligned values originate in the scene.
    """
    import cv2
    from preprocessing import build_global_face_oval_blackbg_png_simalign_strict as legacy
    crop_x, crop_y, crop_w, crop_h = legacy.expand_bbox(selected_face.bbox, image_rgb.shape)
    if crop_w <= 1 or crop_h <= 1: raise legacy.SampleFailure("failed_alignment", "expanded_face_bbox_is_empty")
    crop = image_rgb[crop_y:crop_y + crop_h, crop_x:crop_x + crop_w].copy()
    mesh = legacy.run_facemesh(crop, face_mesh)
    if mesh is None: raise legacy.SampleFailure("failed_landmark_incomplete", "mediapipe_facemesh_returned_none")
    keypoints_crop, _, eye_distance = legacy.extract_alignment_landmarks(mesh, crop.shape)
    if len(mesh) < 468: raise legacy.SampleFailure("failed_landmark_incomplete", f"FaceMesh returned {len(mesh)} landmarks")
    landmarks_crop = np.asarray([[point.x * crop_w, point.y * crop_h] for point in mesh[:468]], dtype=np.float32)
    offset = np.asarray([crop_x, crop_y], dtype=np.float32)
    original_keypoints = keypoints_crop + offset; original_landmarks = landmarks_crop + offset
    matrix, parameters = legacy.estimate_similarity_transform(original_keypoints, legacy.canonical_alignment_template(image_size))
    aligned = cv2.warpAffine(image_rgb, matrix, (image_size, image_size), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))
    source_valid = cv2.warpAffine(np.full(image_rgb.shape[:2], 255, np.uint8), matrix, (image_size, image_size), flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    aligned_keypoints = cv2.transform(original_keypoints[None, :, :], matrix)[0]
    aligned_landmarks = cv2.transform(original_landmarks[None, :, :], matrix)[0]
    top = max(4, round(image_size * 0.05)); parameters.update({"alignment_source":"original_coordinate", "eye_distance":float(eye_distance), "top_invalid_area_ratio":float((source_valid[:top] == 0).mean()), "expanded_bbox_x":crop_x, "expanded_bbox_y":crop_y, "expanded_bbox_w":crop_w, "expanded_bbox_h":crop_h})
    parameters["top_cut_warning"] = parameters["top_invalid_area_ratio"] > 0.20
    return AlignmentArtifacts(aligned, source_valid, matrix, original_keypoints, aligned_keypoints, original_landmarks, aligned_landmarks, (crop_x,crop_y,crop_w,crop_h), tuple(image_rgb.shape[:2]), parameters)
