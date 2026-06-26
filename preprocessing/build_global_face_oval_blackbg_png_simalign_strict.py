"""Build strictly aligned 224x224 face-oval PNG images on a black background.

Pipeline:
split IDs -> MediaPipe face detection -> expanded crop -> FaceMesh ->
similarity-only alignment -> transformed face-oval mask -> feathered black
background -> RGB uint8 PNG.

ImageNet normalization is intentionally not performed here. It belongs in the
training Dataset/transform pipeline.
"""

from __future__ import annotations

import argparse
import importlib.util
import math
import random
import shutil
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence


REQUIRED_DEPENDENCIES = (
    ("cv2", "opencv-python"),
    ("mediapipe", "mediapipe"),
    ("numpy", "numpy"),
    ("pandas", "pandas"),
    ("yaml", "pyyaml"),
)


def check_required_dependencies() -> None:
    """Fail before processing and show an actionable dependency install hint."""
    missing = [
        package_name
        for import_name, package_name in REQUIRED_DEPENDENCIES
        if importlib.util.find_spec(import_name) is None
    ]
    if missing:
        packages = " ".join(dict.fromkeys(missing))
        raise SystemExit(
            "Missing required preprocessing dependencies: "
            f"{', '.join(missing)}.\nInstall them with:\n"
            f"python -m pip install {packages}"
        )


check_required_dependencies()

import cv2  # noqa: E402
import mediapipe as mp  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import yaml  # noqa: E402


SEED = 42
VALID_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp")
STATUS_VALUES = (
    "success",
    "failed_no_image",
    "failed_read_image",
    "failed_no_face",
    "failed_facemesh",
    "failed_landmark_incomplete",
    "failed_alignment",
    "failed_mask_generation",
    "failed_save",
)

# Ordered MediaPipe face-oval contour. This ordering is reused from the legacy
# face-oval preprocessing script in this project.
FACE_OVAL_INDICES = (
    10, 338, 297, 332, 284, 251, 389, 356,
    454, 323, 361, 288, 397, 365, 379, 378,
    400, 377, 152, 148, 176, 149, 150, 136,
    172, 58, 132, 93, 234, 127, 162, 21,
    54, 103, 67, 109,
)

# Multiple landmarks are averaged for each eye to reduce single-point jitter.
# Names refer to positions in the image, not the subject's anatomical side.
IMAGE_LEFT_EYE_INDICES = (33, 133, 159, 145, 153, 154, 155)
IMAGE_RIGHT_EYE_INDICES = (362, 263, 386, 374, 380, 381, 382)
NOSE_TIP_INDEX = 1
IMAGE_LEFT_MOUTH_INDEX = 61
IMAGE_RIGHT_MOUTH_INDEX = 291

LOG_COLUMNS = (
    "ID", "NYHA", "extreme_label", "fold", "SEX",
    "input_path", "output_path", "status", "fail_reason",
    "num_faces_detected", "selected_face_index",
    "face_bbox_x", "face_bbox_y", "face_bbox_w", "face_bbox_h",
    "expanded_bbox_x", "expanded_bbox_y", "expanded_bbox_w", "expanded_bbox_h",
    "facemesh_detected", "landmark_complete", "align_success",
    "rotation_angle", "scale_factor", "translation_x", "translation_y",
    "eye_distance",
    "forehead_expand_ratio", "side_expand_ratio", "chin_expand_ratio",
    "mask_area_pixels_before_expand", "mask_area_ratio_before_expand",
    "mask_area_pixels_after_expand", "mask_area_ratio_after_expand",
    "mask_bbox_xmin_before_expand", "mask_bbox_ymin_before_expand",
    "mask_bbox_xmax_before_expand", "mask_bbox_ymax_before_expand",
    "mask_bbox_xmin_after_expand", "mask_bbox_ymin_after_expand",
    "mask_bbox_xmax_after_expand", "mask_bbox_ymax_after_expand",
    "mask_width_before_expand", "mask_height_before_expand",
    "mask_width_after_expand", "mask_height_after_expand",
    "mask_area_pixels", "mask_area_ratio",
    "mask_bbox_xmin", "mask_bbox_ymin", "mask_bbox_xmax", "mask_bbox_ymax",
    "mask_width", "mask_height", "mask_warning", "background_mode",
    "feather_enabled", "output_format", "image_size",
)

FAILED_COLUMNS = (
    "ID", "NYHA", "extreme_label", "fold", "SEX",
    "input_path", "status", "fail_reason",
)

SUMMARY_METRICS = (
    "count", "mean", "std", "min", "p1", "p5", "p25", "median",
    "p75", "p95", "p99", "max",
)


@dataclass(frozen=True)
class FaceDetection:
    """One clipped face detection in source-image pixel coordinates."""

    index: int
    bbox: tuple[int, int, int, int]
    confidence: float


@dataclass
class PreviewImages:
    """Intermediate RGB images used to assemble one QC preview."""

    original: np.ndarray | None = None
    crop: np.ndarray | None = None
    aligned: np.ndarray | None = None
    original_mask_overlay: np.ndarray | None = None
    expanded_mask_overlay: np.ndarray | None = None
    final: np.ndarray | None = None


class SampleFailure(RuntimeError):
    """Expected per-sample failure with an explicit pipeline status."""

    def __init__(self, status: str, reason: str):
        super().__init__(reason)
        self.status = status
        self.reason = reason


def default_project_root() -> Path:
    """Infer the repository root from this script's location."""
    return Path(__file__).resolve().parents[1]


def build_parser() -> argparse.ArgumentParser:
    """Create the command-line parser without applying configuration defaults."""
    parser = argparse.ArgumentParser(
        description=(
            "Build strict similarity-aligned face-oval RGB PNG images from IDs "
            "listed in a split CSV."
        )
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Optional YAML configuration. CLI values override YAML values.",
    )
    parser.add_argument("--project-root", type=Path, default=None)
    parser.add_argument("--split-csv", type=Path, default=None)
    parser.add_argument("--image-dir", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--image-size", type=int, default=None)
    parser.add_argument("--min-detection-confidence", type=float, default=None)
    parser.add_argument("--mask-low-warning", type=float, default=None)
    parser.add_argument("--mask-high-warning", type=float, default=None)
    parser.add_argument("--num-qc-preview", type=int, default=None)
    parser.add_argument("--forehead-expand-ratio", type=float, default=None)
    parser.add_argument("--side-expand-ratio", type=float, default=None)
    parser.add_argument("--chin-expand-ratio", type=float, default=None)
    parser.add_argument(
        "--show-original-mask-in-qc",
        action=argparse.BooleanOptionalAction,
        default=None,
        help=(
            "Include the unexpanded FaceMesh oval overlay in QC previews "
            "(default: enabled)."
        ),
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        default=None,
        help="Remove only the selected output directory before rebuilding it.",
    )
    return parser


def _normalize_config_keys(config: dict[str, Any]) -> dict[str, Any]:
    return {str(key).strip().replace("-", "_"): value for key, value in config.items()}


def load_yaml_config(config_path: Path | None, project_root: Path) -> dict[str, Any]:
    """Load a YAML mapping; use the project default config when it exists."""
    if config_path is None:
        candidate = (
            project_root
            / "config"
            / "preprocess"
            / "global_face_oval_blackbg_png_simalign_strict.yaml"
        )
        config_path = candidate if candidate.exists() else None
    elif not config_path.is_absolute():
        config_path = project_root / config_path

    if config_path is None:
        return {}
    if not config_path.is_file():
        raise FileNotFoundError(f"Config file does not exist: {config_path}")

    with config_path.open("r", encoding="utf-8") as handle:
        loaded = yaml.safe_load(handle) or {}
    if not isinstance(loaded, dict):
        raise ValueError(f"Config root must be a YAML mapping: {config_path}")
    return _normalize_config_keys(loaded)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Resolve defaults with precedence: CLI > YAML > built-in defaults."""
    parser = build_parser()
    cli = parser.parse_args(argv)

    inferred_root = (
        cli.project_root.expanduser().resolve()
        if cli.project_root is not None
        else default_project_root()
    )
    config = load_yaml_config(cli.config, inferred_root)
    config_root = config.get("project_root")
    project_root = (
        inferred_root
        if cli.project_root is not None or config_root in (None, "", ".")
        else Path(config_root).expanduser().resolve()
    )

    defaults: dict[str, Any] = {
        "split_csv": "data/processed/splits/extreme_5fold.csv",
        "image_dir": "data/raw/images",
        "output_dir": (
            "data/processed/global_face/"
            "global_face_oval_blackbg_224_png_simalign_strict_"
            "forehead015_side004_chin002"
        ),
        "image_size": 224,
        "min_detection_confidence": 0.5,
        "mask_low_warning": 0.35,
        "mask_high_warning": 0.85,
        "num_qc_preview": 20,
        "forehead_expand_ratio": 0.15,
        "side_expand_ratio": 0.04,
        "chin_expand_ratio": 0.02,
        "show_original_mask_in_qc": True,
        "overwrite": False,
    }

    values: dict[str, Any] = {"project_root": project_root, "config": cli.config}
    for key, default in defaults.items():
        cli_value = getattr(cli, key)
        values[key] = cli_value if cli_value is not None else config.get(key, default)

    args = argparse.Namespace(**values)
    validate_args(args)
    return args


def resolve_project_root(args: argparse.Namespace) -> Path:
    """Return the validated absolute project root."""
    root = Path(args.project_root).expanduser().resolve()
    if not root.is_dir():
        raise NotADirectoryError(f"Project root does not exist: {root}")
    return root


def resolve_under_project(path_value: str | Path, project_root: Path) -> Path:
    """Resolve relative paths against the selected project root."""
    path = Path(path_value).expanduser()
    return path.resolve() if path.is_absolute() else (project_root / path).resolve()


def validate_args(args: argparse.Namespace) -> None:
    """Reject invalid thresholds and dimensions before touching output data."""
    if int(args.image_size) <= 0:
        raise ValueError("--image-size must be greater than zero")
    if not 0.0 <= float(args.min_detection_confidence) <= 1.0:
        raise ValueError("--min-detection-confidence must be in [0, 1]")
    low = float(args.mask_low_warning)
    high = float(args.mask_high_warning)
    if not 0.0 <= low < high <= 1.0:
        raise ValueError("Mask warning thresholds must satisfy 0 <= low < high <= 1")
    if int(args.num_qc_preview) < 0:
        raise ValueError("--num-qc-preview must be non-negative")
    for option_name in (
        "forehead_expand_ratio",
        "side_expand_ratio",
        "chin_expand_ratio",
    ):
        value = float(getattr(args, option_name))
        if not 0.0 <= value <= 1.0:
            cli_name = option_name.replace("_", "-")
            raise ValueError(f"--{cli_name} must be in [0, 1]")


def prepare_output_dirs(output_dir: Path, overwrite: bool, project_root: Path) -> dict[str, Path]:
    """Create the output tree while protecting unrelated project directories."""
    output_dir = output_dir.resolve()
    protected = {
        project_root.resolve(),
        (project_root / "data").resolve(),
        (project_root / "data" / "processed").resolve(),
        (project_root / "data" / "processed" / "global_face").resolve(),
    }
    if output_dir in protected:
        raise ValueError(f"Refusing unsafe output directory: {output_dir}")

    if output_dir.exists():
        if not overwrite:
            raise FileExistsError(
                f"Output directory already exists: {output_dir}\n"
                "Use --overwrite only if you intend to rebuild this exact dataset."
            )
        shutil.rmtree(output_dir)

    dirs = {
        "root": output_dir,
        "images": output_dir / "images",
        "logs": output_dir / "logs",
        "qc": output_dir / "qc_preview",
        "random_success": output_dir / "qc_preview" / "random_success",
        "low_mask_area": output_dir / "qc_preview" / "low_mask_area",
        "high_mask_area": output_dir / "qc_preview" / "high_mask_area",
        "failed_no_face": output_dir / "qc_preview" / "failed_no_face",
        "failed_facemesh": output_dir / "qc_preview" / "failed_facemesh",
        "qc_staging": output_dir / "qc_preview" / ".staging_success",
    }
    for directory in dirs.values():
        directory.mkdir(parents=True, exist_ok=True)
    return dirs


def load_split_ids(split_csv: Path) -> pd.DataFrame:
    """Load the split table and preserve IDs as strings."""
    if not split_csv.is_file():
        raise FileNotFoundError(f"Split CSV does not exist: {split_csv}")
    split_df = pd.read_csv(split_csv, dtype={"ID": str})
    if "ID" not in split_df.columns:
        raise ValueError(f"Split CSV is missing required 'ID' column: {split_csv}")

    split_df = split_df.copy()
    split_df["ID"] = split_df["ID"].fillna("").astype(str).str.strip()
    if (split_df["ID"] == "").any():
        bad_rows = (split_df.index[split_df["ID"] == ""] + 2).tolist()
        raise ValueError(f"Split CSV contains empty IDs at CSV rows: {bad_rows[:20]}")
    if split_df["ID"].duplicated().any():
        duplicates = split_df.loc[split_df["ID"].duplicated(), "ID"].unique().tolist()
        raise ValueError(f"Split CSV contains duplicate IDs: {duplicates[:20]}")
    return split_df


def find_image_for_id(image_id: str, image_dir: Path) -> Path | None:
    """Find an image by the required extension priority."""
    for extension in VALID_EXTENSIONS:
        candidate = image_dir / f"{image_id}{extension}"
        if candidate.is_file():
            return candidate
    return None


def read_image_rgb(path: Path) -> np.ndarray | None:
    """Read an image through imdecode for reliable Unicode paths on Windows."""
    try:
        encoded = np.fromfile(path, dtype=np.uint8)
        bgr = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    except (OSError, ValueError):
        return None
    if bgr is None:
        return None
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def detect_faces(
    image_rgb: np.ndarray,
    detector: Any,
) -> list[FaceDetection]:
    """Run MediaPipe face detection and return clipped pixel bounding boxes."""
    height, width = image_rgb.shape[:2]
    results = detector.process(image_rgb)
    detections: list[FaceDetection] = []
    for index, detection in enumerate(results.detections or []):
        relative = detection.location_data.relative_bounding_box
        x1 = max(0, int(math.floor(relative.xmin * width)))
        y1 = max(0, int(math.floor(relative.ymin * height)))
        x2 = min(width, int(math.ceil((relative.xmin + relative.width) * width)))
        y2 = min(height, int(math.ceil((relative.ymin + relative.height) * height)))
        if x2 <= x1 or y2 <= y1:
            continue
        scores = list(detection.score or [])
        confidence = float(scores[0]) if scores else float("nan")
        detections.append(
            FaceDetection(index=index, bbox=(x1, y1, x2 - x1, y2 - y1), confidence=confidence)
        )
    return detections


def select_face(
    detections: Sequence[FaceDetection],
    image_shape: Sequence[int],
) -> FaceDetection | None:
    """Select a large, central face using a deterministic combined score."""
    if not detections:
        return None
    height, width = image_shape[:2]
    image_center = np.array([width / 2.0, height / 2.0], dtype=np.float64)
    diagonal = max(math.hypot(width, height), 1.0)

    def rank(item: FaceDetection) -> tuple[float, float, int]:
        x, y, w, h = item.bbox
        area_ratio = (w * h) / float(width * height)
        center = np.array([x + w / 2.0, y + h / 2.0], dtype=np.float64)
        center_distance = float(np.linalg.norm(center - image_center)) / diagonal
        combined_score = area_ratio / (1.0 + center_distance)
        return combined_score, area_ratio, -item.index

    return max(detections, key=rank)


def expand_bbox(
    bbox: tuple[int, int, int, int],
    image_shape: Sequence[int],
    top_ratio: float = 0.10,
    bottom_ratio: float = 0.20,
    side_ratio: float = 0.20,
) -> tuple[int, int, int, int]:
    """Expand a face box and clip it to image boundaries."""
    x, y, width, height = bbox
    image_height, image_width = image_shape[:2]
    x1 = max(0, int(math.floor(x - side_ratio * width)))
    x2 = min(image_width, int(math.ceil(x + width + side_ratio * width)))
    y1 = max(0, int(math.floor(y - top_ratio * height)))
    y2 = min(image_height, int(math.ceil(y + height + bottom_ratio * height)))
    return x1, y1, x2 - x1, y2 - y1


def run_facemesh(crop_rgb: np.ndarray, face_mesh: Any) -> list[Any] | None:
    """Run strict FaceMesh on the expanded crop without fallback behavior."""
    results = face_mesh.process(crop_rgb)
    if not results.multi_face_landmarks:
        return None
    return list(results.multi_face_landmarks[0].landmark)


def _mean_landmarks(
    landmarks: Sequence[Any],
    indices: Iterable[int],
    width: int,
    height: int,
) -> np.ndarray:
    points = np.array(
        [[landmarks[index].x * width, landmarks[index].y * height] for index in indices],
        dtype=np.float32,
    )
    return points.mean(axis=0)


def _single_landmark(
    landmarks: Sequence[Any],
    index: int,
    width: int,
    height: int,
) -> np.ndarray:
    landmark = landmarks[index]
    return np.array([landmark.x * width, landmark.y * height], dtype=np.float32)


def extract_alignment_landmarks(
    landmarks: Sequence[Any],
    crop_shape: Sequence[int],
) -> tuple[np.ndarray, np.ndarray, float]:
    """Extract robust eye centers, nose, mouth corners, and oval points."""
    required_indices = set(
        IMAGE_LEFT_EYE_INDICES
        + IMAGE_RIGHT_EYE_INDICES
        + (
            NOSE_TIP_INDEX,
            IMAGE_LEFT_MOUTH_INDEX,
            IMAGE_RIGHT_MOUTH_INDEX,
        )
        + FACE_OVAL_INDICES
    )
    if not landmarks or max(required_indices) >= len(landmarks):
        raise SampleFailure(
            "failed_landmark_incomplete",
            f"FaceMesh returned {len(landmarks) if landmarks else 0} landmarks",
        )

    height, width = crop_shape[:2]
    image_left_eye = _mean_landmarks(
        landmarks, IMAGE_LEFT_EYE_INDICES, width, height
    )
    image_right_eye = _mean_landmarks(
        landmarks, IMAGE_RIGHT_EYE_INDICES, width, height
    )
    nose = _single_landmark(landmarks, NOSE_TIP_INDEX, width, height)
    image_left_mouth = _single_landmark(
        landmarks, IMAGE_LEFT_MOUTH_INDEX, width, height
    )
    image_right_mouth = _single_landmark(
        landmarks, IMAGE_RIGHT_MOUTH_INDEX, width, height
    )
    alignment_points = np.vstack(
        [image_left_eye, image_right_eye, nose, image_left_mouth, image_right_mouth]
    ).astype(np.float32)
    oval_points = np.array(
        [
            [landmarks[index].x * width, landmarks[index].y * height]
            for index in FACE_OVAL_INDICES
        ],
        dtype=np.float32,
    )

    if not np.isfinite(alignment_points).all() or not np.isfinite(oval_points).all():
        raise SampleFailure(
            "failed_landmark_incomplete", "FaceMesh contains non-finite coordinates"
        )
    eye_distance = float(np.linalg.norm(image_right_eye - image_left_eye))
    if eye_distance < 1.0 or cv2.contourArea(oval_points) <= 1.0:
        raise SampleFailure(
            "failed_landmark_incomplete", "Degenerate eye distance or face oval"
        )
    return alignment_points, oval_points, eye_distance


def canonical_alignment_template(image_size: int) -> np.ndarray:
    """Return five canonical points in the output coordinate system."""
    normalized = np.array(
        [
            [0.35, 0.38],
            [0.65, 0.38],
            [0.50, 0.55],
            [0.40, 0.72],
            [0.60, 0.72],
        ],
        dtype=np.float32,
    )
    return normalized * float(image_size - 1)


def estimate_similarity_transform(
    source_points: np.ndarray,
    target_points: np.ndarray,
) -> tuple[np.ndarray, dict[str, float]]:
    """Estimate a rotation + uniform-scale + translation transform.

    ``estimateAffinePartial2D`` constrains the linear block to
    [[a, -b], [b, a]]. Unlike a full affine transform, this cannot introduce
    shear or independent x/y scaling, so facial width/height proportions and
    local phenotype geometry are not warped.
    """
    matrix, inliers = cv2.estimateAffinePartial2D(
        source_points,
        target_points,
        method=cv2.LMEDS,
        refineIters=10,
    )
    if matrix is None or matrix.shape != (2, 3) or not np.isfinite(matrix).all():
        raise SampleFailure("failed_alignment", "Similarity transform estimation failed")

    a, minus_b, translation_x = matrix[0]
    b, a_second, translation_y = matrix[1]
    scale = float(math.hypot(a, b))
    if scale <= 0 or not math.isfinite(scale):
        raise SampleFailure("failed_alignment", "Invalid similarity scale")

    # Guard the mathematical constraint explicitly even if OpenCV changes.
    tolerance = max(1e-5, scale * 1e-4)
    if abs(a - a_second) > tolerance or abs(minus_b + b) > tolerance:
        raise SampleFailure(
            "failed_alignment", "Estimated matrix is not a strict similarity transform"
        )
    if inliers is not None and int(np.asarray(inliers).sum()) < 2:
        raise SampleFailure("failed_alignment", "Insufficient alignment inliers")

    parameters = {
        "rotation_angle": float(math.degrees(math.atan2(b, a))),
        "scale_factor": scale,
        "translation_x": float(translation_x),
        "translation_y": float(translation_y),
    }
    return matrix.astype(np.float32), parameters


def warp_face(crop_rgb: np.ndarray, matrix: np.ndarray, image_size: int) -> np.ndarray:
    """Warp the crop with the strict similarity matrix."""
    aligned = cv2.warpAffine(
        crop_rgb,
        matrix,
        (image_size, image_size),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0),
    )
    if aligned is None or aligned.shape != (image_size, image_size, 3):
        raise SampleFailure("failed_alignment", "Similarity warp produced invalid output")
    return aligned


def transform_face_oval_points(
    oval_points: np.ndarray,
    matrix: np.ndarray,
) -> np.ndarray:
    """Map crop-space oval points through the same similarity transform."""
    transformed = cv2.transform(oval_points.reshape(1, -1, 2), matrix)[0]
    if not np.isfinite(transformed).all():
        raise SampleFailure(
            "failed_mask_generation", "Transformed oval contains non-finite points"
        )
    return transformed


def expand_face_oval_points(
    transformed_oval: np.ndarray,
    image_size: int,
    forehead_expand_ratio: float,
    side_expand_ratio: float,
    chin_expand_ratio: float,
) -> np.ndarray:
    """Directionally expand the face-oval polygon without warping the image.

    This is a forehead-aware polygon expansion, not image dilation and not a
    geometric transform of facial pixels. Relative to the original oval bbox
    center, x coordinates expand slightly on both sides, upper y coordinates
    expand primarily for the forehead, and lower y coordinates expand only a
    small amount for the chin.
    """
    points = np.asarray(transformed_oval, dtype=np.float32)
    if points.ndim != 2 or points.shape[1] != 2 or len(points) < 3:
        raise SampleFailure(
            "failed_mask_generation", "Invalid face oval points for expansion"
        )

    xmin = float(points[:, 0].min())
    xmax = float(points[:, 0].max())
    ymin = float(points[:, 1].min())
    ymax = float(points[:, 1].max())
    width = xmax - xmin
    height = ymax - ymin
    if width <= 1.0 or height <= 1.0:
        raise SampleFailure(
            "failed_mask_generation", "Degenerate face oval bbox for expansion"
        )

    center_x = (xmin + xmax) / 2.0
    center_y = (ymin + ymax) / 2.0
    half_width = width / 2.0
    half_height = height / 2.0
    x_scale = (half_width + side_expand_ratio * width) / half_width
    upper_y_scale = (half_height + forehead_expand_ratio * height) / half_height
    lower_y_scale = (half_height + chin_expand_ratio * height) / half_height

    expanded = points.copy()
    expanded[:, 0] = center_x + (points[:, 0] - center_x) * x_scale
    upper = points[:, 1] <= center_y
    expanded[upper, 1] = (
        center_y + (points[upper, 1] - center_y) * upper_y_scale
    )
    expanded[~upper, 1] = (
        center_y + (points[~upper, 1] - center_y) * lower_y_scale
    )
    return np.clip(expanded, 0.0, float(image_size - 1))


def build_binary_mask(
    transformed_oval: np.ndarray,
    image_size: int,
    field_suffix: str = "",
) -> tuple[np.ndarray, dict[str, Any]]:
    """Rasterize the transformed oval and calculate binary-mask statistics."""
    polygon = np.rint(transformed_oval).astype(np.int32)
    mask = np.zeros((image_size, image_size), dtype=np.uint8)
    try:
        cv2.fillPoly(mask, [polygon], 255)
    except cv2.error as exc:
        raise SampleFailure("failed_mask_generation", f"fillPoly failed: {exc}") from exc

    foreground_y, foreground_x = np.where(mask > 0)
    if len(foreground_x) == 0:
        raise SampleFailure("failed_mask_generation", "Face oval mask is empty")

    xmin = int(foreground_x.min())
    xmax = int(foreground_x.max())
    ymin = int(foreground_y.min())
    ymax = int(foreground_y.max())
    area_pixels = int(len(foreground_x))
    suffix = f"_{field_suffix}" if field_suffix else ""
    stats = {
        f"mask_area_pixels{suffix}": area_pixels,
        f"mask_area_ratio{suffix}": area_pixels / float(image_size * image_size),
        f"mask_bbox_xmin{suffix}": xmin,
        f"mask_bbox_ymin{suffix}": ymin,
        f"mask_bbox_xmax{suffix}": xmax,
        f"mask_bbox_ymax{suffix}": ymax,
        f"mask_width{suffix}": xmax - xmin + 1,
        f"mask_height{suffix}": ymax - ymin + 1,
    }
    return mask, stats


def feather_mask(binary_mask: np.ndarray, image_size: int) -> np.ndarray:
    """Create a soft alpha edge while retaining the binary mask for statistics."""
    kernel_size = max(5, int(round(image_size * 0.04)))
    if kernel_size % 2 == 0:
        kernel_size += 1
    sigma = max(1.0, image_size * 0.013)
    feathered = cv2.GaussianBlur(binary_mask, (kernel_size, kernel_size), sigmaX=sigma)
    return feathered.astype(np.float32) / 255.0


def apply_black_background(aligned_rgb: np.ndarray, alpha_mask: np.ndarray) -> np.ndarray:
    """Blend aligned RGB pixels over RGB(0, 0, 0)."""
    alpha = alpha_mask[:, :, None].astype(np.float32)
    output = aligned_rgb.astype(np.float32) * alpha
    return np.clip(np.rint(output), 0, 255).astype(np.uint8)


def save_png(output_path: Path, image_rgb: np.ndarray) -> None:
    """Save an RGB uint8 image as PNG, including Unicode-safe Windows paths."""
    if image_rgb.dtype != np.uint8 or image_rgb.ndim != 3 or image_rgb.shape[2] != 3:
        raise SampleFailure("failed_save", "Output must be HxWx3 uint8 RGB")
    bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    ok, encoded = cv2.imencode(".png", bgr, [cv2.IMWRITE_PNG_COMPRESSION, 3])
    if not ok:
        raise SampleFailure("failed_save", "cv2.imencode('.png') failed")
    try:
        output_path.write_bytes(encoded.tobytes())
    except OSError as exc:
        raise SampleFailure("failed_save", str(exc)) from exc


def _draw_detection_panel(
    image_rgb: np.ndarray,
    detections: Sequence[FaceDetection],
    selected: FaceDetection | None,
) -> np.ndarray:
    panel = image_rgb.copy()
    for detection in detections:
        x, y, width, height = detection.bbox
        color = (0, 255, 0) if selected and detection.index == selected.index else (255, 180, 0)
        cv2.rectangle(panel, (x, y), (x + width, y + height), color, 3)
    return panel


def _mask_overlay(aligned_rgb: np.ndarray, binary_mask: np.ndarray) -> np.ndarray:
    overlay = aligned_rgb.astype(np.float32)
    foreground = binary_mask > 0
    tint = np.zeros_like(overlay)
    tint[:, :, 1] = 255
    overlay[foreground] = 0.72 * overlay[foreground] + 0.28 * tint[foreground]
    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    rendered = np.clip(overlay, 0, 255).astype(np.uint8)
    cv2.drawContours(rendered, contours, -1, (255, 50, 50), 2, cv2.LINE_AA)
    return rendered


def _fit_preview_panel(
    image_rgb: np.ndarray | None,
    title: str,
    panel_size: int = 260,
) -> np.ndarray:
    """Letterbox one RGB image into a labeled square preview panel."""
    label_height = 34
    canvas = np.zeros((panel_size + label_height, panel_size, 3), dtype=np.uint8)
    canvas[:] = (22, 22, 22)
    if image_rgb is not None and image_rgb.size:
        height, width = image_rgb.shape[:2]
        scale = min(panel_size / width, panel_size / height)
        resized_width = max(1, int(round(width * scale)))
        resized_height = max(1, int(round(height * scale)))
        resized = cv2.resize(
            image_rgb, (resized_width, resized_height), interpolation=cv2.INTER_AREA
        )
        x = (panel_size - resized_width) // 2
        y = (panel_size - resized_height) // 2
        canvas[y : y + resized_height, x : x + resized_width] = resized
    cv2.putText(
        canvas,
        title,
        (8, panel_size + 23),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (240, 240, 240),
        1,
        cv2.LINE_AA,
    )
    return canvas


def make_qc_preview(
    output_path: Path,
    image_id: str,
    status: str,
    previews: PreviewImages,
    details: str = "",
    show_original_mask: bool = True,
) -> None:
    """Save a six-panel mask-comparison QC image."""
    panels = [
        _fit_preview_panel(previews.original, "original"),
        _fit_preview_panel(previews.crop, "expanded crop"),
        _fit_preview_panel(previews.aligned, "aligned"),
    ]
    if show_original_mask:
        panels.append(
            _fit_preview_panel(previews.original_mask_overlay, "original oval mask")
        )
    panels.extend(
        [
            _fit_preview_panel(
                previews.expanded_mask_overlay, "forehead-aware mask"
            ),
            _fit_preview_panel(previews.final, "black-bg PNG"),
        ]
    )
    strip = np.concatenate(panels, axis=1)
    header_height = 42
    canvas = np.zeros((strip.shape[0] + header_height, strip.shape[1], 3), dtype=np.uint8)
    canvas[:] = (12, 12, 12)
    canvas[header_height:] = strip
    header = f"ID={image_id}  status={status}"
    if details:
        header += f"  {details}"
    cv2.putText(
        canvas,
        header[:170],
        (10, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.62,
        (245, 245, 245),
        1,
        cv2.LINE_AA,
    )
    bgr = cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR)
    ok, encoded = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, 90])
    if ok:
        try:
            output_path.write_bytes(encoded.tobytes())
        except OSError as exc:
            print(f"[warning] Failed to write QC preview {output_path}: {exc}")


def _initial_log_row(
    split_row: dict[str, Any],
    image_size: int,
    forehead_expand_ratio: float,
    side_expand_ratio: float,
    chin_expand_ratio: float,
) -> dict[str, Any]:
    """Create a complete log row and retain all split metadata columns."""
    log_row = dict(split_row)
    log_row.update(
        {
            "input_path": "",
            "output_path": "",
            "status": "",
            "fail_reason": "",
            "num_faces_detected": 0,
            "selected_face_index": np.nan,
            "face_bbox_x": np.nan,
            "face_bbox_y": np.nan,
            "face_bbox_w": np.nan,
            "face_bbox_h": np.nan,
            "expanded_bbox_x": np.nan,
            "expanded_bbox_y": np.nan,
            "expanded_bbox_w": np.nan,
            "expanded_bbox_h": np.nan,
            "facemesh_detected": False,
            "landmark_complete": False,
            "align_success": False,
            "rotation_angle": np.nan,
            "scale_factor": np.nan,
            "translation_x": np.nan,
            "translation_y": np.nan,
            "eye_distance": np.nan,
            "forehead_expand_ratio": forehead_expand_ratio,
            "side_expand_ratio": side_expand_ratio,
            "chin_expand_ratio": chin_expand_ratio,
            "mask_area_pixels_before_expand": np.nan,
            "mask_area_ratio_before_expand": np.nan,
            "mask_area_pixels_after_expand": np.nan,
            "mask_area_ratio_after_expand": np.nan,
            "mask_bbox_xmin_before_expand": np.nan,
            "mask_bbox_ymin_before_expand": np.nan,
            "mask_bbox_xmax_before_expand": np.nan,
            "mask_bbox_ymax_before_expand": np.nan,
            "mask_bbox_xmin_after_expand": np.nan,
            "mask_bbox_ymin_after_expand": np.nan,
            "mask_bbox_xmax_after_expand": np.nan,
            "mask_bbox_ymax_after_expand": np.nan,
            "mask_width_before_expand": np.nan,
            "mask_height_before_expand": np.nan,
            "mask_width_after_expand": np.nan,
            "mask_height_after_expand": np.nan,
            "mask_area_pixels": np.nan,
            "mask_area_ratio": np.nan,
            "mask_bbox_xmin": np.nan,
            "mask_bbox_ymin": np.nan,
            "mask_bbox_xmax": np.nan,
            "mask_bbox_ymax": np.nan,
            "mask_width": np.nan,
            "mask_height": np.nan,
            "mask_warning": "none",
            "background_mode": "black_rgb_0_0_0",
            "feather_enabled": True,
            "output_format": "PNG",
            "image_size": image_size,
        }
    )
    return log_row


def process_one_sample(
    split_row: dict[str, Any],
    image_dir: Path,
    images_dir: Path,
    detector: Any,
    face_mesh: Any,
    args: argparse.Namespace,
) -> tuple[dict[str, Any], PreviewImages]:
    """Process one ID and convert every expected problem into a logged status."""
    image_id = str(split_row["ID"])
    log_row = _initial_log_row(
        split_row,
        int(args.image_size),
        float(args.forehead_expand_ratio),
        float(args.side_expand_ratio),
        float(args.chin_expand_ratio),
    )
    previews = PreviewImages()
    input_path = find_image_for_id(image_id, image_dir)
    output_path = images_dir / f"{image_id}.png"
    log_row["output_path"] = str(output_path)

    try:
        if input_path is None:
            raise SampleFailure("failed_no_image", "cannot_find_image_by_ID")
        log_row["input_path"] = str(input_path)

        image_rgb = read_image_rgb(input_path)
        if image_rgb is None:
            raise SampleFailure("failed_read_image", "cannot_decode_image")
        previews.original = image_rgb

        detections = detect_faces(image_rgb, detector)
        log_row["num_faces_detected"] = len(detections)
        selected = select_face(detections, image_rgb.shape)
        previews.original = _draw_detection_panel(image_rgb, detections, selected)
        if selected is None:
            raise SampleFailure("failed_no_face", "mediapipe_face_detection_returned_none")

        log_row["selected_face_index"] = selected.index
        x, y, width, height = selected.bbox
        log_row.update(
            {
                "face_bbox_x": x,
                "face_bbox_y": y,
                "face_bbox_w": width,
                "face_bbox_h": height,
            }
        )
        expanded = expand_bbox(selected.bbox, image_rgb.shape)
        crop_x, crop_y, crop_width, crop_height = expanded
        log_row.update(
            {
                "expanded_bbox_x": crop_x,
                "expanded_bbox_y": crop_y,
                "expanded_bbox_w": crop_width,
                "expanded_bbox_h": crop_height,
            }
        )
        if crop_width <= 1 or crop_height <= 1:
            raise SampleFailure("failed_no_face", "expanded_face_bbox_is_empty")
        crop_rgb = image_rgb[
            crop_y : crop_y + crop_height,
            crop_x : crop_x + crop_width,
        ].copy()
        previews.crop = crop_rgb

        landmarks = run_facemesh(crop_rgb, face_mesh)
        if landmarks is None:
            raise SampleFailure("failed_facemesh", "mediapipe_facemesh_returned_none")
        log_row["facemesh_detected"] = True

        alignment_points, oval_points, eye_distance = extract_alignment_landmarks(
            landmarks, crop_rgb.shape
        )
        log_row["landmark_complete"] = True
        log_row["eye_distance"] = eye_distance

        target_points = canonical_alignment_template(int(args.image_size))
        matrix, transform_parameters = estimate_similarity_transform(
            alignment_points, target_points
        )
        log_row.update(transform_parameters)
        aligned_rgb = warp_face(crop_rgb, matrix, int(args.image_size))
        previews.aligned = aligned_rgb
        log_row["align_success"] = True

        transformed_oval = transform_face_oval_points(oval_points, matrix)
        original_binary_mask, before_stats = build_binary_mask(
            transformed_oval,
            int(args.image_size),
            field_suffix="before_expand",
        )
        expanded_oval = expand_face_oval_points(
            transformed_oval,
            int(args.image_size),
            float(args.forehead_expand_ratio),
            float(args.side_expand_ratio),
            float(args.chin_expand_ratio),
        )
        expanded_binary_mask, after_stats = build_binary_mask(
            expanded_oval,
            int(args.image_size),
            field_suffix="after_expand",
        )
        log_row.update(before_stats)
        log_row.update(after_stats)

        # Legacy mask fields continue to describe the final mask used for
        # compositing, preserving compatibility with existing analysis code.
        legacy_mapping = {
            "mask_area_pixels": "mask_area_pixels_after_expand",
            "mask_area_ratio": "mask_area_ratio_after_expand",
            "mask_bbox_xmin": "mask_bbox_xmin_after_expand",
            "mask_bbox_ymin": "mask_bbox_ymin_after_expand",
            "mask_bbox_xmax": "mask_bbox_xmax_after_expand",
            "mask_bbox_ymax": "mask_bbox_ymax_after_expand",
            "mask_width": "mask_width_after_expand",
            "mask_height": "mask_height_after_expand",
        }
        for legacy_field, after_field in legacy_mapping.items():
            log_row[legacy_field] = log_row[after_field]

        ratio = float(log_row["mask_area_ratio_after_expand"])
        if ratio < float(args.mask_low_warning):
            log_row["mask_warning"] = "warning_low_mask_area"
        elif ratio > float(args.mask_high_warning):
            log_row["mask_warning"] = "warning_high_mask_area"

        alpha_mask = feather_mask(expanded_binary_mask, int(args.image_size))
        previews.original_mask_overlay = _mask_overlay(
            aligned_rgb, original_binary_mask
        )
        previews.expanded_mask_overlay = _mask_overlay(
            aligned_rgb, expanded_binary_mask
        )
        final_rgb = apply_black_background(aligned_rgb, alpha_mask)
        previews.final = final_rgb
        save_png(output_path, final_rgb)
        log_row["status"] = "success"
    except SampleFailure as exc:
        log_row["status"] = exc.status
        log_row["fail_reason"] = exc.reason
    except Exception as exc:  # Keep an unexpected single-sample error from aborting the run.
        log_row["status"] = _infer_failure_status(log_row)
        log_row["fail_reason"] = f"{type(exc).__name__}: {exc}"

    if log_row["status"] not in STATUS_VALUES:
        log_row["fail_reason"] = (
            f"invalid_status={log_row['status']}; {log_row['fail_reason']}"
        )
        log_row["status"] = "failed_save"
    return log_row, previews


def _infer_failure_status(log_row: dict[str, Any]) -> str:
    """Map an unexpected exception to the active pipeline stage."""
    if not log_row["input_path"]:
        return "failed_no_image"
    if int(log_row["num_faces_detected"] or 0) == 0:
        return "failed_no_face"
    if not bool(log_row["facemesh_detected"]):
        return "failed_facemesh"
    if not bool(log_row["landmark_complete"]):
        return "failed_landmark_incomplete"
    if not bool(log_row["align_success"]):
        return "failed_alignment"
    if pd.isna(log_row["mask_area_ratio"]):
        return "failed_mask_generation"
    return "failed_save"


def _ordered_log_columns(log_df: pd.DataFrame, split_columns: Sequence[str]) -> list[str]:
    preferred = list(LOG_COLUMNS)
    extra_split = [
        column for column in split_columns if column not in preferred and column in log_df.columns
    ]
    remaining = [
        column for column in log_df.columns if column not in preferred and column not in extra_split
    ]
    return preferred + extra_split + remaining


def _distribution(values: pd.Series) -> dict[str, Any]:
    numeric = pd.to_numeric(values, errors="coerce").dropna()
    if numeric.empty:
        return {metric: np.nan for metric in SUMMARY_METRICS}
    return {
        "count": int(numeric.count()),
        "mean": float(numeric.mean()),
        "std": float(numeric.std(ddof=1)),
        "min": float(numeric.min()),
        "p1": float(numeric.quantile(0.01)),
        "p5": float(numeric.quantile(0.05)),
        "p25": float(numeric.quantile(0.25)),
        "median": float(numeric.quantile(0.50)),
        "p75": float(numeric.quantile(0.75)),
        "p95": float(numeric.quantile(0.95)),
        "p99": float(numeric.quantile(0.99)),
        "max": float(numeric.max()),
    }


def build_mask_summary(log_df: pd.DataFrame) -> pd.DataFrame:
    """Build before/after overall, class, and fold mask-area summaries."""
    success = log_df[log_df["status"] == "success"].copy()
    rows: list[dict[str, Any]] = []

    def append_group(group_type: str, group_value: Any, frame: pd.DataFrame) -> None:
        for mask_stage, value_column in (
            ("before_expand", "mask_area_ratio_before_expand"),
            ("after_expand", "mask_area_ratio_after_expand"),
        ):
            row = {
                "group_type": group_type,
                "group_value": group_value,
                "mask_stage": mask_stage,
            }
            row.update(_distribution(frame[value_column]))
            rows.append(row)

    append_group("overall", "all", success)
    for column in ("extreme_label", "fold"):
        if column in success.columns:
            for value, group in success.groupby(column, dropna=False, sort=True):
                append_group(column, value, group)
    return pd.DataFrame(
        rows,
        columns=("group_type", "group_value", "mask_stage", *SUMMARY_METRICS),
    )


def _format_distribution(stats: dict[str, Any]) -> str:
    return ", ".join(
        f"{key}={value:.6f}" if isinstance(value, float) and math.isfinite(value)
        else f"{key}={value}"
        for key, value in stats.items()
    )


def summarize_logs(
    log_df: pd.DataFrame,
    args: argparse.Namespace,
    output_dir: Path,
    logs_dir: Path,
) -> str:
    """Write all requested CSV/TXT logs and return the terminal summary."""
    preprocess_log = logs_dir / "preprocess_log.csv"
    failed_cases = logs_dir / "failed_cases.csv"
    mask_summary_path = logs_dir / "mask_area_ratio_summary.csv"
    text_summary_path = logs_dir / "preprocess_summary.txt"

    log_df.to_csv(preprocess_log, index=False, encoding="utf-8-sig")
    failed_df = log_df[log_df["status"] != "success"].copy()
    for column in FAILED_COLUMNS:
        if column not in failed_df.columns:
            failed_df[column] = ""
    failed_df[list(FAILED_COLUMNS)].to_csv(
        failed_cases, index=False, encoding="utf-8-sig"
    )
    mask_summary = build_mask_summary(log_df)
    mask_summary.to_csv(mask_summary_path, index=False, encoding="utf-8-sig")

    total = len(log_df)
    successes = int((log_df["status"] == "success").sum())
    failures = total - successes
    success_rate = successes / total if total else 0.0
    lines = [
        "Strict face-oval black-background preprocessing summary",
        "=" * 64,
        f"Total IDs: {total}",
        f"Successes: {successes}",
        f"Failures: {failures}",
        f"Success rate: {success_rate:.2%}",
        "",
        "Failure reasons/statuses:",
    ]
    status_counts = Counter(failed_df["status"].astype(str))
    if status_counts:
        lines.extend(f"  {status}: {count}" for status, count in sorted(status_counts.items()))
    else:
        lines.append("  none")

    for column in ("fold", "extreme_label"):
        lines.extend(["", f"Success/failure by {column}:"])
        if column not in log_df.columns:
            lines.append("  column not present")
            continue
        grouped = (
            log_df.assign(success=log_df["status"].eq("success"))
            .groupby(column, dropna=False)["success"]
            .agg(total="size", success="sum")
        )
        grouped["failed"] = grouped["total"] - grouped["success"]
        for value, row in grouped.iterrows():
            lines.append(
                f"  {value}: total={int(row['total'])}, "
                f"success={int(row['success'])}, failed={int(row['failed'])}"
            )

    success_df = log_df[log_df["status"] == "success"]
    lines.extend(
        [
            "",
            "mask_area_ratio_after_expand overall:",
            f"  {_format_distribution(_distribution(success_df['mask_area_ratio']))}",
            "mask_area_ratio_before_expand overall:",
            "  "
            + _format_distribution(
                _distribution(success_df["mask_area_ratio_before_expand"])
            ),
        ]
    )
    for column in ("extreme_label", "fold"):
        lines.extend(["", f"mask_area_ratio by {column}:"])
        if column not in success_df.columns:
            lines.append("  column not present")
            continue
        for value, group in success_df.groupby(column, dropna=False, sort=True):
            lines.append(
                f"  {value}: {_format_distribution(_distribution(group['mask_area_ratio']))}"
            )

    lines.extend(
        [
            "",
            f"Output directory: {output_dir}",
            "Parameters:",
            f"  project_root: {args.project_root}",
            f"  split_csv: {args.split_csv}",
            f"  image_dir: {args.image_dir}",
            f"  output_dir: {args.output_dir}",
            f"  image_size: {args.image_size}",
            f"  min_detection_confidence: {args.min_detection_confidence}",
            f"  mask_low_warning: {args.mask_low_warning}",
            f"  mask_high_warning: {args.mask_high_warning}",
            f"  num_qc_preview: {args.num_qc_preview}",
            f"  forehead_expand_ratio: {args.forehead_expand_ratio}",
            f"  side_expand_ratio: {args.side_expand_ratio}",
            f"  chin_expand_ratio: {args.chin_expand_ratio}",
            f"  show_original_mask_in_qc: {args.show_original_mask_in_qc}",
            f"  overwrite: {args.overwrite}",
            "",
            "Output images are RGB uint8 PNG files. Apply ImageNet mean/std "
            "normalization in the training Dataset/transform, not offline.",
        ]
    )
    summary = "\n".join(lines)
    text_summary_path.write_text(summary + "\n", encoding="utf-8")
    return summary


def finalize_success_qc(
    log_df: pd.DataFrame,
    dirs: dict[str, Path],
    num_qc_preview: int,
) -> None:
    """Copy staged success previews into random/low/high QC categories."""
    success = log_df[log_df["status"] == "success"].copy()
    if success.empty or num_qc_preview <= 0:
        shutil.rmtree(dirs["qc_staging"], ignore_errors=True)
        return

    success["mask_area_ratio"] = pd.to_numeric(
        success["mask_area_ratio"], errors="coerce"
    )
    random_rows = success.sample(
        n=min(num_qc_preview, len(success)), random_state=SEED
    )
    low_rows = success.nsmallest(min(20, len(success)), "mask_area_ratio")
    high_rows = success.nlargest(min(20, len(success)), "mask_area_ratio")

    for category, frame in (
        ("random_success", random_rows),
        ("low_mask_area", low_rows),
        ("high_mask_area", high_rows),
    ):
        for row in frame.itertuples(index=False):
            source = dirs["qc_staging"] / f"{row.ID}.jpg"
            if source.is_file():
                try:
                    shutil.copy2(source, dirs[category] / source.name)
                except OSError as exc:
                    print(
                        f"[warning] Failed to copy QC preview {source} "
                        f"to {category}: {exc}"
                    )
    shutil.rmtree(dirs["qc_staging"], ignore_errors=True)


def main(argv: Sequence[str] | None = None) -> int:
    """Run the complete strict preprocessing pipeline."""
    args = parse_args(argv)
    project_root = resolve_project_root(args)
    args.project_root = project_root
    args.split_csv = resolve_under_project(args.split_csv, project_root)
    args.image_dir = resolve_under_project(args.image_dir, project_root)
    args.output_dir = resolve_under_project(args.output_dir, project_root)

    if not args.image_dir.is_dir():
        raise NotADirectoryError(f"Image directory does not exist: {args.image_dir}")
    split_df = load_split_ids(args.split_csv)
    dirs = prepare_output_dirs(
        args.output_dir, bool(args.overwrite), project_root
    )

    random.seed(SEED)
    np.random.seed(SEED)
    detector = mp.solutions.face_detection.FaceDetection(
        model_selection=1,
        min_detection_confidence=float(args.min_detection_confidence),
    )
    face_mesh = mp.solutions.face_mesh.FaceMesh(
        static_image_mode=True,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=float(args.min_detection_confidence),
        min_tracking_confidence=0.5,
    )

    rows: list[dict[str, Any]] = []
    failed_preview_counts = {"failed_no_face": 0, "failed_facemesh": 0}
    try:
        total = len(split_df)
        for position, split_series in enumerate(split_df.to_dict("records"), start=1):
            image_id = str(split_series["ID"])
            print(f"[{position:04d}/{total:04d}] {image_id}")
            log_row, previews = process_one_sample(
                split_row=split_series,
                image_dir=args.image_dir,
                images_dir=dirs["images"],
                detector=detector,
                face_mesh=face_mesh,
                args=args,
            )
            rows.append(log_row)

            status = str(log_row["status"])
            details = (
                f"before={log_row['mask_area_ratio_before_expand']:.4f} "
                f"after={log_row['mask_area_ratio_after_expand']:.4f}"
                if status == "success"
                else str(log_row["fail_reason"])
            )
            if status == "success":
                make_qc_preview(
                    dirs["qc_staging"] / f"{image_id}.jpg",
                    image_id,
                    status,
                    previews,
                    details,
                    bool(args.show_original_mask_in_qc),
                )
            elif status in failed_preview_counts and failed_preview_counts[status] < 20:
                make_qc_preview(
                    dirs[status] / f"{image_id}.jpg",
                    image_id,
                    status,
                    previews,
                    details,
                    bool(args.show_original_mask_in_qc),
                )
                failed_preview_counts[status] += 1
    finally:
        detector.close()
        face_mesh.close()

    log_df = pd.DataFrame(rows)
    ordered_columns = _ordered_log_columns(log_df, split_df.columns.tolist())
    for column in ordered_columns:
        if column not in log_df.columns:
            log_df[column] = np.nan
    log_df = log_df[ordered_columns]

    finalize_success_qc(log_df, dirs, int(args.num_qc_preview))
    summary = summarize_logs(log_df, args, args.output_dir, dirs["logs"])
    print("\n" + summary)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (FileNotFoundError, FileExistsError, NotADirectoryError, ValueError) as exc:
        print(f"[configuration error] {exc}", file=sys.stderr)
        raise SystemExit(2) from exc
