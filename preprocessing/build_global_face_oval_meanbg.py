"""Build Baseline3 v2 global face-oval images.

Version C preprocessing:
raw image -> face detection/crop -> affine alignment to 224x224 ->
MediaPipe FaceMesh face oval mask -> VGGFace2 mean BGR background.

This script does not modify the existing global_224 directory or
processed_features.csv.
"""

from __future__ import annotations

import importlib.util
import random
import sys
from pathlib import Path
from typing import Any


REQUIRED_DEPENDENCIES = [
    ("cv2", "opencv-python"),
    ("numpy", "numpy"),
    ("pandas", "pandas"),
    ("mediapipe", "mediapipe"),
]


def check_required_dependencies() -> None:
    """Fail early with an explicit install hint when dependencies are missing."""
    missing = [
        package_name
        for import_name, package_name in REQUIRED_DEPENDENCIES
        if importlib.util.find_spec(import_name) is None
    ]
    if missing:
        install_hint = " ".join(dict.fromkeys(missing))
        raise SystemExit(
            "Missing required dependencies for face oval preprocessing: "
            f"{', '.join(missing)}.\n"
            "Install them in the active Python environment, for example:\n"
            f"pip install {install_hint}"
        )


check_required_dependencies()

import cv2
import mediapipe as mp
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(r"E:\projects\face")
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data_preprocessing.image_utils import affine_alignment, detect_and_crop_face  # noqa: E402


SEED = 42
TARGET_SIZE = (224, 224)

RAW_IMAGE_DIR = PROJECT_ROOT / "data" / "raw" / "images"
LABEL_CSV = PROJECT_ROOT / "data" / "raw" / "label.csv"
OUTPUT_DIR = PROJECT_ROOT / "data" / "processed" / "global_face" / "global_face_oval_meanbg_224"
PREVIEW_PATH = PROJECT_ROOT / "data" / "processed" / "global_face_oval_meanbg_224_preview.jpg"
LOG_PATH = PROJECT_ROOT / "log" / "global_face_oval_meanbg_224_preprocess_log.csv"

VALID_EXTENSIONS = [".jpg", ".jpeg", ".png", ".bmp"]

# MediaPipe FaceMesh face oval landmark indices.
FACE_OVAL_IDX = [
    10, 338, 297, 332, 284, 251, 389, 356,
    454, 323, 361, 288, 397, 365, 379, 378,
    400, 377, 152, 148, 176, 149, 150, 136,
    172, 58, 132, 93, 234, 127, 162, 21,
    54, 103, 67, 109,
]

# VGGFace2 mean is usually specified in RGB:
# RGB mean = [131.0912, 103.8827, 91.4953].
# OpenCV arrays are BGR, so the background fill color must be reversed:
# BGR mean = [91.4953, 103.8827, 131.0912].
VGGFACE2_RGB_MEAN = np.array([131.0912, 103.8827, 91.4953], dtype=np.float32)
VGGFACE2_BGR_MEAN = VGGFACE2_RGB_MEAN[::-1].copy()


def set_seed(seed: int = SEED) -> None:
    """Set Python and NumPy random seeds."""
    random.seed(seed)
    np.random.seed(seed)


def build_output_dirs() -> None:
    """Create output image and log directories."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    PREVIEW_PATH.parent.mkdir(parents=True, exist_ok=True)


def load_labels() -> pd.DataFrame | None:
    """Load label.csv when available and valid enough for sampling/output."""
    try:
        labels = pd.read_csv(LABEL_CSV, dtype={"ID": str})
    except Exception as exc:
        print(f"[warning] Failed to read label.csv: {exc}")
        return None

    required_columns = {"ID", "NYHA"}
    missing = required_columns - set(labels.columns)
    if missing:
        print(f"[warning] label.csv missing columns: {sorted(missing)}")
        return None

    labels = labels.copy()
    labels["ID"] = labels["ID"].astype(str)
    labels["NYHA"] = pd.to_numeric(labels["NYHA"], errors="coerce")
    return labels


def find_image_for_id(image_id: str) -> Path | None:
    """Find a raw image path matching an ID and known image extension."""
    for extension in VALID_EXTENSIONS:
        candidate = RAW_IMAGE_DIR / f"{image_id}{extension}"
        if candidate.exists():
            return candidate
    return None


def collect_images(labels: pd.DataFrame | None) -> pd.DataFrame:
    """Collect images in label order when labels exist, otherwise scan raw dir."""
    if labels is not None:
        rows: list[dict[str, Any]] = []
        for row in labels.itertuples(index=False):
            image_id = str(row.ID)
            input_path = find_image_for_id(image_id)
            rows.append(
                {
                    "ID": image_id,
                    "NYHA": row.NYHA,
                    "input_path": str(input_path) if input_path is not None else "",
                }
            )
        return pd.DataFrame(rows)

    if not RAW_IMAGE_DIR.exists():
        raise FileNotFoundError(f"Raw image directory does not exist: {RAW_IMAGE_DIR}")

    image_paths = [
        path
        for path in sorted(RAW_IMAGE_DIR.iterdir())
        if path.is_file() and path.suffix.lower() in VALID_EXTENSIONS
    ]
    return pd.DataFrame(
        [{"ID": path.stem, "NYHA": np.nan, "input_path": str(path)} for path in image_paths]
    )


def create_face_oval_mask(
    aligned_bgr: np.ndarray,
    face_mesh: mp.solutions.face_mesh.FaceMesh,
) -> tuple[np.ndarray | None, float, bool]:
    """Create a feathered FaceMesh face-oval mask on a 224x224 aligned image."""
    height, width = aligned_bgr.shape[:2]
    aligned_rgb = cv2.cvtColor(aligned_bgr, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(aligned_rgb)

    if not results.multi_face_landmarks:
        return None, 0.0, False

    landmarks = results.multi_face_landmarks[0].landmark
    points = []
    for index in FACE_OVAL_IDX:
        landmark = landmarks[index]
        x = int(round(landmark.x * width))
        y = int(round(landmark.y * height))
        x = min(max(x, 0), width - 1)
        y = min(max(y, 0), height - 1)
        points.append([x, y])

    polygon = np.asarray(points, dtype=np.int32)
    binary_mask = np.zeros((height, width), dtype=np.uint8)
    cv2.fillPoly(binary_mask, [polygon], 255)
    mask_area_ratio = float((binary_mask > 0).sum()) / float(height * width)

    feathered_mask = cv2.GaussianBlur(binary_mask, (9, 9), sigmaX=3)
    return feathered_mask, mask_area_ratio, True


def apply_mean_background(aligned_bgr: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Blend face pixels over the VGGFace2 mean BGR background."""
    alpha = (mask.astype(np.float32) / 255.0)[:, :, None]
    image_float = aligned_bgr.astype(np.float32)
    background = np.full_like(image_float, VGGFACE2_BGR_MEAN, dtype=np.float32)
    output = image_float * alpha + background * (1.0 - alpha)
    return np.clip(output, 0, 255).astype(np.uint8)


def process_one_image(
    image_id: str,
    input_path: Path | None,
    output_path: Path,
    face_mesh: mp.solutions.face_mesh.FaceMesh,
) -> dict[str, Any]:
    """Process one raw image and return one CSV log row."""
    log_row: dict[str, Any] = {
        "ID": image_id,
        "input_path": str(input_path) if input_path is not None else "",
        "output_path": str(output_path),
        "status": "failed",
        "error_message": "",
        "face_detected": False,
        "facemesh_detected": False,
        "mask_area_ratio": 0.0,
        "image_width": 0,
        "image_height": 0,
    }

    try:
        if input_path is None or not input_path.exists():
            raise FileNotFoundError(f"Raw image not found for ID: {image_id}")

        image = cv2.imread(str(input_path))
        if image is None:
            raise ValueError("cv2.imread returned None")

        log_row["image_height"], log_row["image_width"] = image.shape[:2]

        face_crop, _ = detect_and_crop_face(image)
        if face_crop is None:
            raise ValueError("face detection failed")
        log_row["face_detected"] = True

        aligned = affine_alignment(face_crop, target_size=TARGET_SIZE)
        if aligned is None:
            raise ValueError("affine alignment failed")

        mask, mask_area_ratio, facemesh_detected = create_face_oval_mask(
            aligned, face_mesh
        )
        log_row["facemesh_detected"] = facemesh_detected
        log_row["mask_area_ratio"] = mask_area_ratio
        if mask is None:
            raise ValueError("FaceMesh face oval detection failed")

        output = apply_mean_background(aligned, mask)
        ok = cv2.imwrite(str(output_path), output, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
        if not ok:
            raise IOError(f"failed to write output image: {output_path}")

        log_row["status"] = "success"
    except Exception as exc:
        log_row["error_message"] = str(exc)

    return log_row


def _make_preview_tile(image_path: Path, image_id: str, nyha: Any) -> np.ndarray | None:
    """Read one processed image and add a text label below it."""
    image = cv2.imread(str(image_path))
    if image is None:
        return None

    tile_width, tile_height = TARGET_SIZE
    label_height = 34
    tile = np.full(
        (tile_height + label_height, tile_width, 3),
        VGGFACE2_BGR_MEAN,
        dtype=np.uint8,
    )
    tile[:tile_height, :tile_width] = image

    nyha_text = "NA" if pd.isna(nyha) else str(int(nyha))
    text = f"{image_id} NYHA={nyha_text}"
    cv2.putText(
        tile,
        text[:32],
        (6, tile_height + 22),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (20, 20, 20),
        1,
        cv2.LINE_AA,
    )
    return tile


def save_preview_grid(
    log_df: pd.DataFrame,
    labels: pd.DataFrame | None,
    preview_path: Path,
) -> None:
    """Save a QC preview grid, stratified by NYHA when labels are available."""
    success_df = log_df[log_df["status"] == "success"].copy()
    if success_df.empty:
        print("[warning] No successful images available for preview.")
        return

    if labels is not None and "NYHA" in labels.columns:
        preview_df = success_df.merge(labels[["ID", "NYHA"]], on="ID", how="left")
        sampled_parts = []
        for nyha_value in [0, 1]:
            part = preview_df[preview_df["NYHA"] == nyha_value]
            if not part.empty:
                sampled_parts.append(
                    part.sample(n=min(20, len(part)), random_state=SEED)
                )
        if sampled_parts:
            preview_df = pd.concat(sampled_parts, ignore_index=True)
        else:
            preview_df = preview_df.sample(n=min(40, len(preview_df)), random_state=SEED)
    else:
        preview_df = success_df.copy()
        preview_df["NYHA"] = np.nan
        preview_df = preview_df.sample(n=min(40, len(preview_df)), random_state=SEED)

    tiles: list[np.ndarray] = []
    for row in preview_df.itertuples(index=False):
        tile = _make_preview_tile(Path(row.output_path), str(row.ID), row.NYHA)
        if tile is not None:
            tiles.append(tile)

    if not tiles:
        print("[warning] Failed to assemble preview tiles.")
        return

    columns = 5
    rows = int(np.ceil(len(tiles) / columns))
    tile_h, tile_w = tiles[0].shape[:2]
    canvas = np.full(
        (rows * tile_h, columns * tile_w, 3),
        VGGFACE2_BGR_MEAN,
        dtype=np.uint8,
    )

    for index, tile in enumerate(tiles):
        row_index = index // columns
        col_index = index % columns
        y1 = row_index * tile_h
        x1 = col_index * tile_w
        canvas[y1 : y1 + tile_h, x1 : x1 + tile_w] = tile

    cv2.imwrite(str(preview_path), canvas, [int(cv2.IMWRITE_JPEG_QUALITY), 95])


def main() -> None:
    """Run the Version C face-oval preprocessing pipeline."""
    set_seed(SEED)
    build_output_dirs()

    labels = load_labels()
    image_df = collect_images(labels)

    face_mesh = mp.solutions.face_mesh.FaceMesh(
        static_image_mode=True,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
    )

    log_rows: list[dict[str, Any]] = []
    try:
        for row in image_df.itertuples(index=False):
            image_id = str(row.ID)
            input_path = Path(row.input_path) if row.input_path else None
            output_path = OUTPUT_DIR / f"{image_id}.jpg"
            print(f"[{image_id}] processing...")
            log_rows.append(
                process_one_image(
                    image_id=image_id,
                    input_path=input_path,
                    output_path=output_path,
                    face_mesh=face_mesh,
                )
            )
    finally:
        face_mesh.close()

    log_df = pd.DataFrame(log_rows)
    log_df.to_csv(LOG_PATH, index=False, encoding="utf-8-sig")
    save_preview_grid(log_df, labels, PREVIEW_PATH)

    total_count = len(log_df)
    success_count = int((log_df["status"] == "success").sum())
    failed_count = int((log_df["status"] == "failed").sum())

    print("=" * 72)
    print("Global face oval mean-background preprocessing complete.")
    print(f"Total images: {total_count}")
    print(f"Successful: {success_count}")
    print(f"Failed: {failed_count}")
    print(f"Output directory: {OUTPUT_DIR}")
    print(f"Log path: {LOG_PATH}")
    print(f"Preview path: {PREVIEW_PATH}")


if __name__ == "__main__":
    main()

