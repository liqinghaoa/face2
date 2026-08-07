"""P0->DECA image adaptation and explicit homogeneous coordinate transforms."""
from __future__ import annotations
import cv2
import numpy as np

def identity_mapping() -> tuple[np.ndarray,np.ndarray]:
    """Return explicit 224-space identity matrices for direct input mode."""
    matrix=np.eye(3,dtype=np.float32); return matrix,matrix.copy()

def bbox_mapping(bbox: tuple[int,int,int,int], output_size: int = 224) -> tuple[np.ndarray,np.ndarray]:
    """Map half-open P0 bbox coordinates to square DECA input and back."""
    x1,y1,x2,y2=bbox; width,height=x2-x1,y2-y1
    if width<=0 or height<=0: raise ValueError("invalid crop bbox")
    forward=np.array([[output_size/width,0,-x1*output_size/width],[0,output_size/height,-y1*output_size/height],[0,0,1]],np.float32)
    return forward,np.linalg.inv(forward).astype(np.float32)


def mask_bbox(mask: np.ndarray, expand_ratio: float) -> tuple[int, int, int, int]:
    """Return a clamped half-open bbox from a P0 face-valid mask.

    The expansion is fixed by configuration and is intentionally independent of
    diagnosis, DECA output, and visual review.
    """
    value = np.asarray(mask) > 0
    if value.ndim != 2 or not value.any():
        raise ValueError("face-valid mask must be a non-empty 2D mask")
    if expand_ratio < 0:
        raise ValueError("expand_ratio must be non-negative")
    ys, xs = np.where(value)
    x1, x2 = int(xs.min()), int(xs.max()) + 1
    y1, y2 = int(ys.min()), int(ys.max()) + 1
    pad_x = int(np.ceil((x2 - x1) * expand_ratio))
    pad_y = int(np.ceil((y2 - y1) * expand_ratio))
    height, width = value.shape
    return max(0, x1 - pad_x), max(0, y1 - pad_y), min(width, x2 + pad_x), min(height, y2 + pad_y)


def mask_bbox_crop(image: np.ndarray, face_valid_mask: np.ndarray, expand_ratio: float, output_size: int = 224) -> tuple[np.ndarray, tuple[int, int, int, int], np.ndarray, np.ndarray]:
    """Crop P0 aligned RGB image using its face-valid mask and resize deterministically.

    This function deliberately does not apply DECA intensity normalization: that
    normalization must come from the audited official DECA source.
    """
    source = np.asarray(image)
    if source.ndim != 3 or source.shape[:2] != np.asarray(face_valid_mask).shape[:2]:
        raise ValueError("image and face-valid mask dimensions must agree")
    bbox = mask_bbox(face_valid_mask, expand_ratio)
    x1, y1, x2, y2 = bbox
    crop = source[y1:y2, x1:x2]
    if crop.size == 0:
        raise ValueError("expanded crop is empty")
    resized = cv2.resize(crop, (output_size, output_size), interpolation=cv2.INTER_LINEAR)
    p0_to_deca, deca_to_p0 = bbox_mapping(bbox, output_size)
    return resized, bbox, p0_to_deca, deca_to_p0
