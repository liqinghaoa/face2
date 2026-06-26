"""Build a single ROI-only dataset from frontal face images.

This script only performs ROI preprocessing. It does not train a model and
does not create or change train/validation folds.

Default paths match this repository:
    input images: data/raw/images
    labels:       data/raw/label.csv
    output:       data/processed/roi_dataset1
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any


REQUIRED_DEPENDENCIES = [
    ("cv2", "opencv-python"),
    ("mediapipe", "mediapipe"),
    ("numpy", "numpy"),
    ("pandas", "pandas"),
]


def check_required_dependencies() -> None:
    """Fail early with a clear install hint when dependencies are missing."""
    missing = [
        package_name
        for import_name, package_name in REQUIRED_DEPENDENCIES
        if importlib.util.find_spec(import_name) is None
    ]
    if missing:
        install_hint = " ".join(dict.fromkeys(missing))
        raise SystemExit(
            "Missing required dependencies for single ROI preprocessing: "
            f"{', '.join(missing)}.\n"
            "Install them in the active Python environment, for example:\n"
            f"pip install {install_hint}"
        )


check_required_dependencies()

import cv2  # noqa: E402
import mediapipe as mp  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT_IMAGE_DIR = PROJECT_ROOT / "data" / "raw" / "images"
DEFAULT_LABEL_CSV_PATH = PROJECT_ROOT / "data" / "raw" / "label.csv"
DEFAULT_OUTPUT_ROI_DIR = PROJECT_ROOT / "data" / "processed" / "roi_dataset"

VALID_IMAGE_EXTENSIONS = [".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"]

PATIENT_ID_CANDIDATES = ["patient_id", "PatientID", "patientID", "ID", "id"]
SEX_CANDIDATES = ["sex", "SEX", "gender", "Gender"]
NYHA_BINARY_CANDIDATES = ["nyha_binary", "NYHA_binary", "y", "label"]
NYHA_BINARY_FALLBACK_CANDIDATES = ["NYHA", "nyha_grade"]
IMAGE_PATH_CANDIDATES = ["image_path", "ImagePath", "path", "filename", "file_name", "image_name"]

PRESERVE_COLUMNS = ["fold", "NYHA", "nyha_grade", "y", "label"]
METADATA_BASE_COLUMNS = [
    "patient_id",
    "image_path",
    "eye_roi_path",
    "lip_roi_path",
    "cheek_roi_path",
    "forehead_roi_path",
    "nyha_binary",
    "sex",
]


# MediaPipe Face Mesh landmark groups. The values are deliberately kept in
# module-level constants so ROI definitions can be audited and tuned later.
LEFT_EYE_INDICES = [
    33,
    7,
    163,
    144,
    145,
    153,
    154,
    155,
    133,
    173,
    157,
    158,
    159,
    160,
    161,
    246,
]
RIGHT_EYE_INDICES = [
    362,
    382,
    381,
    380,
    374,
    373,
    390,
    249,
    263,
    466,
    388,
    387,
    386,
    385,
    384,
    398,
]
EYE_INDICES = LEFT_EYE_INDICES + RIGHT_EYE_INDICES

LIP_INDICES = [
    61,
    146,
    91,
    181,
    84,
    17,
    314,
    405,
    321,
    375,
    291,
    308,
    324,
    318,
    402,
    317,
    14,
    87,
    178,
    88,
    95,
    185,
    40,
    39,
    37,
    0,
    267,
    269,
    270,
    409,
    78,
    191,
    80,
    81,
    82,
    13,
    312,
    311,
    310,
    415,
]
BROW_INDICES = [70, 63, 105, 66, 107, 336, 296, 334, 293, 300]
FACE_OVAL_INDICES = [
    10,
    338,
    297,
    332,
    284,
    251,
    389,
    356,
    454,
    323,
    361,
    288,
    397,
    365,
    379,
    378,
    400,
    377,
    152,
    148,
    176,
    149,
    150,
    136,
    172,
    58,
    132,
    93,
    234,
    127,
    162,
    21,
    54,
    103,
    67,
    109,
]


@dataclass(frozen=True)
class RoiConfig:
    eye_size: tuple[int, int] = (224, 112)
    lip_size: tuple[int, int] = (224, 112)
    cheek_size: tuple[int, int] = (224, 224)
    forehead_size: tuple[int, int] = (224, 112)
    eye_side_expand_face_ratio: float = 0.05
    eye_min_width_face_ratio = 0.56
    eye_top_height_fraction = 0.36
    eye_bottom_height_fraction = 0.64
    lip_expand_x: float = 0.35
    lip_expand_y: float = 0.60
    cheek_y_start_below_eye_ratio: float = 0.08
    cheek_y_bottom_above_lip_ratio: float = 0.04
    cheek_y_bottom_face_ratio: float = 0.68
    cheek_left_x_start_ratio: float = 0.10
    cheek_left_x_end_ratio: float = 0.36
    cheek_right_x_start_ratio: float = 0.64
    cheek_right_x_end_ratio: float = 0.90
    cheek_min_side_width_ratio: float = 0.18
    cheek_min_height_ratio: float = 0.28
    forehead_side_expand_ratio: float = 0.08
    forehead_below_brow_gap_ratio: float = 0.035
    forehead_height_ratio: float = 0.14
    forehead_min_height_ratio: float = 0.10
    min_roi_width: int = 8
    min_roi_height: int = 8


class RoiExtractionError(ValueError):
    """Raised when landmarks produce an invalid ROI for one sample."""


def setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def normalize_exts(image_ext: str) -> list[str]:
    if not image_ext:
        return VALID_IMAGE_EXTENSIONS
    exts = []
    for raw_ext in image_ext.split(","):
        ext = raw_ext.strip().lower()
        if not ext:
            continue
        exts.append(ext if ext.startswith(".") else f".{ext}")
    return exts or VALID_IMAGE_EXTENSIONS


def load_image(image_path: str | Path) -> np.ndarray:
    """Load an image as uint8 BGR while preserving three color channels."""
    path = Path(image_path)
    data = np.fromfile(str(path), dtype=np.uint8)
    image = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"Failed to read image: {path}")
    return image


def load_landmarks_from_file(landmark_path: Path, width: int, height: int) -> np.ndarray:
    """Load precomputed landmarks from npy/json/csv if a project has them."""
    suffix = landmark_path.suffix.lower()
    if suffix == ".npy":
        landmarks = np.load(str(landmark_path))
    elif suffix == ".json":
        with landmark_path.open("r", encoding="utf-8") as f:
            payload = json.load(f)
        if isinstance(payload, dict):
            payload = payload.get("landmarks", payload.get("points", payload))
        landmarks = np.asarray(payload, dtype=np.float32)
    elif suffix == ".csv":
        df = pd.read_csv(landmark_path)
        if {"x", "y"}.issubset(df.columns):
            landmarks = df[["x", "y"]].to_numpy(dtype=np.float32)
        else:
            landmarks = df.iloc[:, :2].to_numpy(dtype=np.float32)
    else:
        raise ValueError(f"Unsupported landmark file format: {landmark_path}")

    landmarks = np.asarray(landmarks, dtype=np.float32)
    if landmarks.ndim != 2 or landmarks.shape[1] < 2:
        raise ValueError(f"Invalid landmark shape in {landmark_path}: {landmarks.shape}")
    landmarks = landmarks[:, :2]

    # Accept either normalized coordinates in [0, 1] or pixel coordinates.
    if np.nanmax(landmarks[:, 0]) <= 1.5 and np.nanmax(landmarks[:, 1]) <= 1.5:
        landmarks[:, 0] *= width
        landmarks[:, 1] *= height
    return landmarks


def find_landmark_file(landmark_dir: Path | None, patient_id: str) -> Path | None:
    if landmark_dir is None:
        return None
    for ext in [".npy", ".json", ".csv"]:
        candidate = landmark_dir / f"{patient_id}{ext}"
        if candidate.exists():
            return candidate
    return None


def detect_face_landmarks(
    image_bgr: np.ndarray,
    face_mesh: mp.solutions.face_mesh.FaceMesh,
    landmark_path: Path | None = None,
    max_detection_side: int = 1024,
) -> np.ndarray:
    """Return Face Mesh landmarks as an Nx2 pixel coordinate array."""
    height, width = image_bgr.shape[:2]

    if landmark_path is not None:
        landmarks = load_landmarks_from_file(landmark_path, width, height)
    else:
        detection_image = image_bgr
        max_side = max(height, width)
        if max_detection_side > 0 and max_side > max_detection_side:
            scale = max_detection_side / float(max_side)
            detection_image = cv2.resize(
                image_bgr,
                (int(round(width * scale)), int(round(height * scale))),
                interpolation=cv2.INTER_AREA,
            )

        image_rgb = cv2.cvtColor(detection_image, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(image_rgb)
        if not results.multi_face_landmarks:
            raise RoiExtractionError("face_landmarks_not_detected")
        landmarks = np.array(
            [[lm.x * width, lm.y * height] for lm in results.multi_face_landmarks[0].landmark],
            dtype=np.float32,
        )

    if landmarks.shape[0] < 468:
        raise RoiExtractionError(f"insufficient_landmarks: {landmarks.shape[0]}")
    if not np.isfinite(landmarks).all():
        raise RoiExtractionError("landmarks_contain_nan_or_inf")
    return landmarks


def clamp_bbox(
    bbox: tuple[float, float, float, float],
    image_shape: tuple[int, int, int] | tuple[int, int],
) -> tuple[int, int, int, int]:
    height, width = image_shape[:2]
    x1, y1, x2, y2 = bbox
    x1_i = max(0, min(width, int(np.floor(x1))))
    y1_i = max(0, min(height, int(np.floor(y1))))
    x2_i = max(0, min(width, int(np.ceil(x2))))
    y2_i = max(0, min(height, int(np.ceil(y2))))
    return x1_i, y1_i, x2_i, y2_i


def validate_bbox(
    bbox: tuple[int, int, int, int],
    roi_name: str,
    min_width: int,
    min_height: int,
) -> None:
    x1, y1, x2, y2 = bbox
    if x2 <= x1 or y2 <= y1:
        raise RoiExtractionError(f"{roi_name}_bbox_invalid: {bbox}")
    if (x2 - x1) < min_width or (y2 - y1) < min_height:
        raise RoiExtractionError(f"{roi_name}_bbox_too_small: {bbox}")


def get_bbox_from_landmarks(
    landmarks: np.ndarray,
    indices: list[int],
    image_shape: tuple[int, int, int] | tuple[int, int],
) -> tuple[int, int, int, int]:
    """Build a clamped bounding box from selected landmark indices."""
    pts = landmarks[np.asarray(indices, dtype=np.int32)]
    x1, y1 = np.min(pts, axis=0)
    x2, y2 = np.max(pts, axis=0)
    return clamp_bbox((x1, y1, x2, y2), image_shape)


def expand_bbox(
    bbox: tuple[int, int, int, int],
    image_shape: tuple[int, int, int] | tuple[int, int],
    expand_x: float,
    expand_y: float,
) -> tuple[int, int, int, int]:
    """Expand a bbox by independent horizontal and vertical ratios."""
    x1, y1, x2, y2 = bbox
    width = x2 - x1
    height = y2 - y1
    return clamp_bbox(
        (
            x1 - width * expand_x,
            y1 - height * expand_y,
            x2 + width * expand_x,
            y2 + height * expand_y,
        ),
        image_shape,
    )


def expand_bbox_asymmetric(
    bbox: tuple[int, int, int, int],
    image_shape: tuple[int, int, int] | tuple[int, int],
    expand_left: float,
    expand_top: float,
    expand_right: float,
    expand_bottom: float,
) -> tuple[int, int, int, int]:
    """Expand a bbox by independent side ratios."""
    x1, y1, x2, y2 = bbox
    width = x2 - x1
    height = y2 - y1
    return clamp_bbox(
        (
            x1 - width * expand_left,
            y1 - height * expand_top,
            x2 + width * expand_right,
            y2 + height * expand_bottom,
        ),
        image_shape,
    )


def ensure_min_bbox_size(
    bbox: tuple[int, int, int, int],
    image_shape: tuple[int, int, int] | tuple[int, int],
    min_width: int,
    min_height: int,
) -> tuple[int, int, int, int]:
    """Expand a bbox around its center so borderline ROIs do not fail."""
    height, width = image_shape[:2]
    x1, y1, x2, y2 = bbox
    current_width = x2 - x1
    current_height = y2 - y1
    center_x = (x1 + x2) / 2.0
    center_y = (y1 + y2) / 2.0
    target_width = min(width, max(current_width, min_width))
    target_height = min(height, max(current_height, min_height))
    return clamp_bbox(
        (
            center_x - target_width / 2.0,
            center_y - target_height / 2.0,
            center_x + target_width / 2.0,
            center_y + target_height / 2.0,
        ),
        image_shape,
    )


def crop_and_resize(
    image_bgr: np.ndarray,
    bbox: tuple[int, int, int, int],
    target_size: tuple[int, int],
    roi_name: str,
    config: RoiConfig,
) -> np.ndarray:
    """Crop one ROI and resize it to the requested (width, height)."""
    validate_bbox(bbox, roi_name, config.min_roi_width, config.min_roi_height)
    x1, y1, x2, y2 = bbox
    crop = image_bgr[y1:y2, x1:x2]
    if crop.size == 0:
        raise RoiExtractionError(f"{roi_name}_crop_empty: {bbox}")
    resized = cv2.resize(crop, target_size, interpolation=cv2.INTER_AREA)
    if resized.ndim != 3 or resized.shape[2] != 3:
        raise RoiExtractionError(f"{roi_name}_not_three_channel")
    return resized


def extract_eye_roi(
    image_bgr: np.ndarray,
    landmarks: np.ndarray,
    config: RoiConfig,
) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    """Extract one combined eye and periorbital ROI.

    Medical intent: keep both eyes, eyelids, eye bags, periorbital soft tissue,
    and skin around inner/outer canthi in a single horizontal crop. This does
    not crop left/right eyes separately or concatenate them. The vertical eye
    band is scaled from face width rather than eye-opening height, making the
    ROI more stable for squinting, blinking, and mild landmark jitter. The crop
    keeps more tissue below the eye center to include lower eyelids and eye bags.
    """
    eye_bbox = get_bbox_from_landmarks(landmarks, EYE_INDICES, image_bgr.shape)
    face_bbox = get_bbox_from_landmarks(landmarks, FACE_OVAL_INDICES, image_bgr.shape)
    eye_points = landmarks[np.asarray(EYE_INDICES, dtype=np.int32)]

    eye_x1, _, eye_x2, _ = eye_bbox
    face_x1, _, face_x2, _ = face_bbox
    face_width = face_x2 - face_x1
    if face_width <= 0:
        raise RoiExtractionError(f"eye_face_bbox_invalid: {face_bbox}")

    target_aspect = config.eye_size[0] / config.eye_size[1]
    eye_center_x = (eye_x1 + eye_x2) / 2.0
    eye_center_y = float(np.mean(eye_points[:, 1]))
    eye_width = eye_x2 - eye_x1
    target_width = max(
        eye_width + 2.0 * config.eye_side_expand_face_ratio * face_width,
        config.eye_min_width_face_ratio * face_width,
    )
    target_height = target_width / target_aspect
    bbox = clamp_bbox(
        (
            eye_center_x - target_width / 2.0,
            eye_center_y - config.eye_top_height_fraction * target_height,
            eye_center_x + target_width / 2.0,
            eye_center_y + config.eye_bottom_height_fraction * target_height,
        ),
        image_bgr.shape,
    )
    roi = crop_and_resize(image_bgr, bbox, config.eye_size, "eye", config)
    return roi, bbox


def extract_lip_roi(
    image_bgr: np.ndarray,
    landmarks: np.ndarray,
    config: RoiConfig,
) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    """Extract a lip and perioral ROI.

    Medical intent: keep upper/lower lips, mouth corners, perioral skin, a small
    subnasal area, and skin below the lower lip. The crop deliberately extends
    beyond the vermilion border.
    """
    bbox = get_bbox_from_landmarks(landmarks, LIP_INDICES, image_bgr.shape)
    bbox = expand_bbox(bbox, image_bgr.shape, config.lip_expand_x, config.lip_expand_y)
    roi = crop_and_resize(image_bgr, bbox, config.lip_size, "lip", config)
    return roi, bbox


def extract_cheek_roi(
    image_bgr: np.ndarray,
    landmarks: np.ndarray,
    config: RoiConfig,
) -> tuple[np.ndarray, list[tuple[int, int, int, int]]]:
    """Extract bilateral cheek patches and concatenate them horizontally.

    Medical intent: focus on the mid-lower cheek from lateral nose/zygomatic
    lower area toward the region outside the mouth corners and above the jaw.
    Current stable definition uses face-width bands rather than the lip bbox as
    the inner border. This avoids width collapse when the mouth is wide, the
    head is slightly rotated, or one mouth corner is close to the face outline.
    """
    face_bbox = get_bbox_from_landmarks(landmarks, FACE_OVAL_INDICES, image_bgr.shape)
    eye_bbox = get_bbox_from_landmarks(landmarks, EYE_INDICES, image_bgr.shape)
    lip_bbox = get_bbox_from_landmarks(landmarks, LIP_INDICES, image_bgr.shape)

    face_x1, face_y1, face_x2, face_y2 = face_bbox
    _, _, _, eye_y2 = eye_bbox
    _, lip_y1, _, _ = lip_bbox

    face_width = face_x2 - face_x1
    face_height = face_y2 - face_y1
    if face_width <= 0 or face_height <= 0:
        raise RoiExtractionError(f"cheek_face_bbox_invalid: {face_bbox}")

    cheek_y1 = eye_y2 + int(round(config.cheek_y_start_below_eye_ratio * face_height))
    cheek_y2 = min(
        lip_y1 - int(round(config.cheek_y_bottom_above_lip_ratio * face_height)),
        face_y1 + int(round(config.cheek_y_bottom_face_ratio * face_height)),
    )
    min_cheek_width = max(config.min_roi_width, int(round(config.cheek_min_side_width_ratio * face_width)))
    min_cheek_height = max(config.min_roi_height, int(round(config.cheek_min_height_ratio * face_height)))

    left_bbox = clamp_bbox(
        (
            face_x1 + config.cheek_left_x_start_ratio * face_width,
            cheek_y1,
            face_x1 + config.cheek_left_x_end_ratio * face_width,
            cheek_y2,
        ),
        image_bgr.shape,
    )
    right_bbox = clamp_bbox(
        (
            face_x1 + config.cheek_right_x_start_ratio * face_width,
            cheek_y1,
            face_x1 + config.cheek_right_x_end_ratio * face_width,
            cheek_y2,
        ),
        image_bgr.shape,
    )
    left_bbox = ensure_min_bbox_size(left_bbox, image_bgr.shape, min_cheek_width, min_cheek_height)
    right_bbox = ensure_min_bbox_size(right_bbox, image_bgr.shape, min_cheek_width, min_cheek_height)

    validate_bbox(left_bbox, "left_cheek", config.min_roi_width, config.min_roi_height)
    validate_bbox(right_bbox, "right_cheek", config.min_roi_width, config.min_roi_height)

    half_width = config.cheek_size[0] // 2
    target_half_size = (half_width, config.cheek_size[1])
    left_roi = crop_and_resize(image_bgr, left_bbox, target_half_size, "left_cheek", config)
    right_roi = crop_and_resize(image_bgr, right_bbox, target_half_size, "right_cheek", config)
    cheek_roi = np.concatenate([left_roi, right_roi], axis=1)

    if cheek_roi.shape[1] != config.cheek_size[0]:
        cheek_roi = cv2.resize(cheek_roi, config.cheek_size, interpolation=cv2.INTER_AREA)
    return cheek_roi, [left_bbox, right_bbox]


def extract_forehead_roi(
    image_bgr: np.ndarray,
    landmarks: np.ndarray,
    config: RoiConfig,
) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    """Extract a forehead control ROI.

    Medical intent: this is a control region. It keeps skin above the eyebrows
    and the lower-to-mid forehead while trying to avoid hairline/background.
    It deliberately stays closer to the brow line than the hairline; if bangs
    still cover this area, --save_vis makes that visible for manual adjustment.
    """
    face_bbox = get_bbox_from_landmarks(landmarks, FACE_OVAL_INDICES, image_bgr.shape)
    brow_bbox = get_bbox_from_landmarks(landmarks, BROW_INDICES, image_bgr.shape)

    face_x1, face_y1, face_x2, face_y2 = face_bbox
    brow_x1, brow_y1, brow_x2, _ = brow_bbox
    face_width = face_x2 - face_x1
    face_height = face_y2 - face_y1
    if face_width <= 0 or face_height <= 0:
        raise RoiExtractionError(f"forehead_face_bbox_invalid: {face_bbox}")

    x1 = brow_x1 - config.forehead_side_expand_ratio * face_width
    x2 = brow_x2 + config.forehead_side_expand_ratio * face_width
    y2 = brow_y1 - config.forehead_below_brow_gap_ratio * face_height
    y1 = y2 - config.forehead_height_ratio * face_height

    bbox = clamp_bbox((x1, y1, x2, y2), image_bgr.shape)
    min_forehead_height = max(config.min_roi_height, int(round(config.forehead_min_height_ratio * face_height)))
    bbox = ensure_min_bbox_size(bbox, image_bgr.shape, config.min_roi_width, min_forehead_height)
    roi = crop_and_resize(image_bgr, bbox, config.forehead_size, "forehead", config)
    return roi, bbox


def save_png(image_bgr: np.ndarray, output_path: Path) -> None:
    """Save a BGR image as PNG, robust to non-ASCII paths on Windows."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    ok, encoded = cv2.imencode(".png", image_bgr)
    if not ok:
        raise IOError(f"Failed to encode PNG: {output_path}")
    encoded.tofile(str(output_path))


def save_roi_images(
    rois: dict[str, np.ndarray],
    patient_output_dir: Path,
    overwrite: bool,
) -> dict[str, Path]:
    """Save all four ROI images as PNG files."""
    patient_output_dir.mkdir(parents=True, exist_ok=True)
    output_paths = {
        "eye": patient_output_dir / "eye_roi.png",
        "lip": patient_output_dir / "lip_roi.png",
        "cheek": patient_output_dir / "cheek_roi.png",
        "forehead": patient_output_dir / "forehead_roi.png",
    }
    for roi_name, output_path in output_paths.items():
        if output_path.exists() and not overwrite:
            logging.info("Keep existing %s ROI: %s", roi_name, output_path)
            continue
        save_png(rois[roi_name], output_path)
        logging.info("Saved %s ROI: %s", roi_name, output_path)
    return output_paths


def save_visualization(
    image_bgr: np.ndarray,
    bboxes: dict[str, tuple[int, int, int, int] | list[tuple[int, int, int, int]]],
    output_path: Path,
) -> None:
    """Draw ROI boxes on the original image for visual quality control."""
    vis = image_bgr.copy()
    colors = {
        "eye": (0, 255, 255),
        "lip": (0, 0, 255),
        "cheek": (0, 200, 0),
        "forehead": (255, 0, 0),
    }

    for roi_name, box_value in bboxes.items():
        boxes = box_value if isinstance(box_value, list) else [box_value]
        for idx, bbox in enumerate(boxes):
            x1, y1, x2, y2 = bbox
            color = colors[roi_name]
            cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)
            label = roi_name if len(boxes) == 1 else f"{roi_name}_{idx + 1}"
            label_y = max(12, y1 - 6)
            cv2.putText(
                vis,
                label,
                (x1, label_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                1,
                cv2.LINE_AA,
            )
    save_png(vis, output_path)


def first_existing_column(columns: list[str], candidates: list[str]) -> str | None:
    column_set = set(columns)
    for candidate in candidates:
        if candidate in column_set:
            return candidate
    lower_map = {column.lower(): column for column in columns}
    for candidate in candidates:
        if candidate.lower() in lower_map:
            return lower_map[candidate.lower()]
    return None


def is_binary_label_series(series: pd.Series) -> bool:
    """Return True only when non-empty labels are already encoded as 0/1."""
    values = pd.to_numeric(series.dropna(), errors="coerce").dropna().unique()
    if len(values) == 0:
        return False
    return set(values.tolist()).issubset({0, 1})


def resolve_label_columns(labels: pd.DataFrame) -> dict[str, str | None]:
    columns = list(labels.columns)
    patient_col = first_existing_column(columns, PATIENT_ID_CANDIDATES)
    if patient_col is None:
        raise ValueError(
            "label_csv_path must contain a patient id column. "
            f"Accepted names: {PATIENT_ID_CANDIDATES}"
        )

    sex_col = first_existing_column(columns, SEX_CANDIDATES)
    nyha_binary_col = first_existing_column(columns, NYHA_BINARY_CANDIDATES)
    if nyha_binary_col is None:
        fallback_col = first_existing_column(columns, NYHA_BINARY_FALLBACK_CANDIDATES)
        if fallback_col is not None:
            if is_binary_label_series(labels[fallback_col]):
                nyha_binary_col = fallback_col
                logging.info("Using binary-valued %s as nyha_binary.", fallback_col)
            else:
                logging.warning(
                    "Found %s but it is not encoded as 0/1, so it will only be preserved "
                    "and nyha_binary will be empty.",
                    fallback_col,
                )
    image_path_col = first_existing_column(columns, IMAGE_PATH_CANDIDATES)

    if sex_col is None:
        logging.warning("No sex column found. Metadata sex will be empty.")
    if nyha_binary_col is None:
        logging.warning("No binary nyha_binary/y/label column found. Metadata nyha_binary will be empty.")

    return {
        "patient_id": patient_col,
        "sex": sex_col,
        "nyha_binary": nyha_binary_col,
        "image_path": image_path_col,
    }


def find_image_path(
    row: pd.Series,
    input_image_dir: Path,
    patient_id: str,
    image_exts: list[str],
    image_path_col: str | None,
) -> Path | None:
    """Resolve an image path from CSV image path or patient id + extension."""
    if image_path_col is not None and pd.notna(row.get(image_path_col)):
        raw_path = Path(str(row[image_path_col]))
        candidates = [raw_path]
        if not raw_path.is_absolute():
            candidates.append(input_image_dir / raw_path)
        for candidate in candidates:
            if candidate.exists() and candidate.is_file():
                return candidate.resolve()

    for ext in image_exts:
        candidate = input_image_dir / f"{patient_id}{ext}"
        if candidate.exists() and candidate.is_file():
            return candidate.resolve()

    # Case-insensitive fallback for manually named files.
    if input_image_dir.exists():
        patient_lower = patient_id.lower()
        allowed = {ext.lower() for ext in image_exts}
        for candidate in input_image_dir.iterdir():
            if not candidate.is_file() or candidate.suffix.lower() not in allowed:
                continue
            if candidate.stem.lower() == patient_lower:
                return candidate.resolve()
    return None


def build_metadata_row(
    row: pd.Series,
    columns: dict[str, str | None],
    image_path: Path,
    roi_paths: dict[str, Path],
) -> dict[str, Any]:
    patient_col = columns["patient_id"]
    sex_col = columns["sex"]
    nyha_binary_col = columns["nyha_binary"]

    metadata = {
        "patient_id": str(row[patient_col]),
        "image_path": str(image_path),
        "eye_roi_path": str(roi_paths["eye"]),
        "lip_roi_path": str(roi_paths["lip"]),
        "cheek_roi_path": str(roi_paths["cheek"]),
        "forehead_roi_path": str(roi_paths["forehead"]),
        "nyha_binary": row[nyha_binary_col] if nyha_binary_col is not None else "",
        "sex": row[sex_col] if sex_col is not None else "",
    }

    for column in PRESERVE_COLUMNS:
        if column in row.index and column not in metadata:
            metadata[column] = row[column]
    return metadata


def make_failed_row(patient_id: str, image_path: str, reason: str) -> dict[str, str]:
    return {"patient_id": patient_id, "image_path": image_path, "reason": reason}


def process_one_sample(
    row: pd.Series,
    columns: dict[str, str | None],
    input_image_dir: Path,
    output_roi_dir: Path,
    image_exts: list[str],
    face_mesh: mp.solutions.face_mesh.FaceMesh,
    config: RoiConfig,
    overwrite: bool,
    save_vis: bool,
    vis_remaining: int,
    landmark_dir: Path | None,
    max_detection_side: int,
) -> tuple[dict[str, Any] | None, dict[str, str] | None, bool]:
    patient_id = str(row[columns["patient_id"]])
    image_path = find_image_path(row, input_image_dir, patient_id, image_exts, columns["image_path"])
    if image_path is None:
        return None, make_failed_row(patient_id, "", "image_not_found"), False

    try:
        image_bgr = load_image(image_path)
        landmark_path = find_landmark_file(landmark_dir, patient_id)
        landmarks = detect_face_landmarks(
            image_bgr,
            face_mesh,
            landmark_path,
            max_detection_side=max_detection_side,
        )

        eye_roi, eye_bbox = extract_eye_roi(image_bgr, landmarks, config)
        lip_roi, lip_bbox = extract_lip_roi(image_bgr, landmarks, config)
        cheek_roi, cheek_bboxes = extract_cheek_roi(image_bgr, landmarks, config)
        forehead_roi, forehead_bbox = extract_forehead_roi(image_bgr, landmarks, config)

        rois = {
            "eye": eye_roi,
            "lip": lip_roi,
            "cheek": cheek_roi,
            "forehead": forehead_roi,
        }
        patient_output_dir = output_roi_dir / patient_id
        roi_paths = save_roi_images(rois, patient_output_dir, overwrite=overwrite)

        did_save_vis = False
        if save_vis and vis_remaining > 0:
            vis_path = output_roi_dir / "vis" / f"{patient_id}_vis.png"
            if overwrite or not vis_path.exists():
                save_visualization(
                    image_bgr,
                    {
                        "eye": eye_bbox,
                        "lip": lip_bbox,
                        "cheek": cheek_bboxes,
                        "forehead": forehead_bbox,
                    },
                    vis_path,
                )
                logging.info("Saved visualization: %s", vis_path)
            did_save_vis = True

        metadata = build_metadata_row(row, columns, image_path, roi_paths)
        return metadata, None, did_save_vis
    except Exception as exc:
        logging.warning("Failed patient_id=%s image=%s reason=%s", patient_id, image_path, exc)
        return None, make_failed_row(patient_id, str(image_path), str(exc)), False


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate Eye/Lip/Cheek/Forehead ROI-only dataset from face images."
    )
    parser.add_argument(
        "--input_image_dir",
        type=Path,
        default=DEFAULT_INPUT_IMAGE_DIR,
        help="Input source face image directory. Defaults to data/raw/images.",
    )
    parser.add_argument(
        "--label_csv_path",
        type=Path,
        default=DEFAULT_LABEL_CSV_PATH,
        help="Label CSV path. Must contain patient_id or ID.",
    )
    parser.add_argument(
        "--output_roi_dir",
        type=Path,
        default=DEFAULT_OUTPUT_ROI_DIR,
        help="Output ROI dataset directory.",
    )
    parser.add_argument(
        "--image_ext",
        default="",
        help="Image extension(s) to search, e.g. .jpg or .jpg,.png. Empty searches common formats.",
    )
    parser.add_argument(
        "--landmark_dir",
        type=Path,
        default=None,
        help="Optional directory containing precomputed landmarks named patient_id.npy/json/csv.",
    )
    parser.add_argument(
        "--max_detection_side",
        type=int,
        default=1024,
        help=(
            "Resize only the image passed to MediaPipe so its longest side is at most this value, "
            "then scale landmarks back to source-image coordinates. Use <=0 to detect at full resolution."
        ),
    )
    parser.add_argument("--save_vis", action="store_true", help="Save ROI bounding-box visualizations.")
    parser.add_argument("--vis_limit", type=int, default=50, help="Maximum number of visualizations to save.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing ROI/visualization files.")
    return parser.parse_args()


def main() -> None:
    setup_logging()
    args = parse_args()
    config = RoiConfig()

    input_image_dir = args.input_image_dir.resolve()
    label_csv_path = args.label_csv_path.resolve()
    output_roi_dir = args.output_roi_dir.resolve()
    landmark_dir = args.landmark_dir.resolve() if args.landmark_dir else None
    image_exts = normalize_exts(args.image_ext)

    if not input_image_dir.exists():
        raise FileNotFoundError(f"Input image directory does not exist: {input_image_dir}")
    if not label_csv_path.exists():
        raise FileNotFoundError(f"Label CSV does not exist: {label_csv_path}")
    if output_roi_dir.exists() and args.overwrite:
        logging.info("Overwrite enabled. Existing files may be replaced under: %s", output_roi_dir)
    output_roi_dir.mkdir(parents=True, exist_ok=True)
    if args.save_vis:
        (output_roi_dir / "vis").mkdir(parents=True, exist_ok=True)

    labels = pd.read_csv(label_csv_path)
    columns = resolve_label_columns(labels)

    total_samples = len(labels)
    logging.info("Input image directory: %s", input_image_dir)
    logging.info("Label CSV: %s", label_csv_path)
    logging.info("Output ROI directory: %s", output_roi_dir)
    logging.info("Total samples: %d", total_samples)

    metadata_rows: list[dict[str, Any]] = []
    failed_rows: list[dict[str, str]] = []
    vis_saved = 0

    face_mesh = mp.solutions.face_mesh.FaceMesh(
        static_image_mode=True,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
    )

    try:
        for sample_index, (_, row) in enumerate(labels.iterrows(), start=1):
            metadata, failed, did_save_vis = process_one_sample(
                row=row,
                columns=columns,
                input_image_dir=input_image_dir,
                output_roi_dir=output_roi_dir,
                image_exts=image_exts,
                face_mesh=face_mesh,
                config=config,
                overwrite=args.overwrite,
                save_vis=args.save_vis,
                vis_remaining=max(0, args.vis_limit - vis_saved),
                landmark_dir=landmark_dir,
                max_detection_side=args.max_detection_side,
            )
            if metadata is not None:
                metadata_rows.append(metadata)
            if failed is not None:
                failed_rows.append(failed)
            if did_save_vis:
                vis_saved += 1

            if sample_index == 1 or sample_index % 25 == 0 or sample_index == total_samples:
                logging.info(
                    "Progress %d/%d | success=%d | failed=%d",
                    sample_index,
                    total_samples,
                    len(metadata_rows),
                    len(failed_rows),
                )
    finally:
        face_mesh.close()

    metadata_path = output_roi_dir / "roi_metadata.csv"
    failed_cases_path = output_roi_dir / "failed_cases.csv"

    metadata_columns = METADATA_BASE_COLUMNS + [
        column for column in PRESERVE_COLUMNS if column in labels.columns and column not in METADATA_BASE_COLUMNS
    ]
    metadata_df = pd.DataFrame(metadata_rows)
    if not metadata_df.empty:
        remaining_columns = [column for column in metadata_df.columns if column not in metadata_columns]
        metadata_df = metadata_df[metadata_columns + remaining_columns]
    else:
        metadata_df = pd.DataFrame(columns=metadata_columns)
    metadata_df.to_csv(metadata_path, index=False, encoding="utf-8-sig")

    failed_df = pd.DataFrame(failed_rows, columns=["patient_id", "image_path", "reason"])
    failed_df.to_csv(failed_cases_path, index=False, encoding="utf-8-sig")

    logging.info("Successful samples: %d", len(metadata_rows))
    logging.info("Failed samples: %d", len(failed_rows))
    logging.info("ROI metadata CSV: %s", metadata_path)
    logging.info("Failed cases CSV: %s", failed_cases_path)
    if args.save_vis:
        logging.info("Visualization directory: %s", output_roi_dir / "vis")


if __name__ == "__main__":
    main()
