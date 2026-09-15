"""Create Figure 5 as a 2x3 Patient-logit Grad-CAM panel montage.

The six panels are assembled from already generated OOF Grad-CAM PNGs. For
each source image, only panel (b), ``Grad-CAM for Patient logit``, is retained.
The displayed eye regions are pixelated for de-identification; no CAM or
probability is recomputed.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import mediapipe as mp
import numpy as np
import pandas as pd
from PIL import Image


DATA_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = DATA_DIR.parents[3]
EXPERIMENT_DIR = (
    PROJECT_ROOT
    / "experiments"
    / "500Data"
    / "R3DPR_ResNet18_ControlVsPatient_Binary_256x320_v2_relight_neutral_front_fullbaseline_ls005_5fold"
)
GRADCAM_DIR = EXPERIMENT_DIR / "figures" / "gradcam_oof"
MANIFEST_PATH = GRADCAM_DIR / "gradcam_manifest.csv"

PNG_PATH = DATA_DIR / "Figure5_gradcam_patient_2x3.png"
PDF_PATH = DATA_DIR / "Figure5_gradcam_patient_2x3.pdf"
SVG_PATH = DATA_DIR / "Figure5_gradcam_patient_2x3.svg"
TIFF_PATH = DATA_DIR / "Figure5_gradcam_patient_2x3.tiff"
CHECK_PATH = DATA_DIR / "Figure5_gradcam_patient_2x3_check.csv"

SELECTED_IDS = (
    "102400553",
    "203151068",
    "A001813114",
    "A002345714",
    "A000477213",
    "A002128820",
)

# Bounds are defined relative to the fixed-format Grad-CAM PNG (1459 x 904).
# They retain only the panel-(b) heatmap, excluding its repeated title text.
PANEL_B_CROP = (824, 117, 1437, 881)
EXPECTED_SOURCE_SIZE = (1459, 904)
EYE_PIXEL_BLOCK_SIZE = 12
EYE_PADDING_X = 0.32
EYE_PADDING_Y = 0.70


def configure_matplotlib() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
        }
    )


def load_selected_rows() -> pd.DataFrame:
    manifest = pd.read_csv(MANIFEST_PATH, dtype={"sample_id": str})
    required_columns = {
        "sample_id",
        "fold",
        "true_class",
        "predicted_class",
        "prob_patient",
        "target_class",
        "target_name",
        "output_png",
    }
    missing_columns = required_columns - set(manifest.columns)
    if missing_columns:
        raise ValueError(f"Grad-CAM manifest is missing columns: {sorted(missing_columns)}")
    if manifest["sample_id"].duplicated().any():
        raise ValueError("Grad-CAM manifest contains duplicate sample IDs")

    selected = manifest.set_index("sample_id", drop=False).reindex(SELECTED_IDS)
    missing_ids = selected.index[selected["output_png"].isna()].tolist()
    if missing_ids:
        raise ValueError(f"Selected IDs are missing from the Grad-CAM manifest: {missing_ids}")
    if not (
        (selected["true_class"].astype(int) == 1)
        & (selected["predicted_class"].astype(int) == 1)
        & (selected["target_class"].astype(int) == 1)
        & (selected["target_name"].astype(str) == "Patient")
    ).all():
        raise ValueError("All Figure 5 panels must be correctly predicted Patient samples with Patient-logit Grad-CAM")
    return selected.reset_index(drop=True)


def crop_patient_gradcam(source_path: Path) -> np.ndarray:
    if not source_path.is_file():
        raise FileNotFoundError(f"Grad-CAM panel is missing: {source_path}")
    with Image.open(source_path) as image:
        image = image.convert("RGB")
        if image.size != EXPECTED_SOURCE_SIZE:
            raise ValueError(f"Unexpected Grad-CAM source size for {source_path.name}: {image.size}")
        return np.asarray(image.crop(PANEL_B_CROP))


def eye_landmark_indices(connections: set[tuple[int, int]]) -> tuple[int, ...]:
    """Return the unique Face Mesh indices that outline one eye."""
    return tuple(sorted({index for connection in connections for index in connection}))


LEFT_EYE_INDICES = eye_landmark_indices(mp.solutions.face_mesh.FACEMESH_LEFT_EYE)
RIGHT_EYE_INDICES = eye_landmark_indices(mp.solutions.face_mesh.FACEMESH_RIGHT_EYE)


def pixelate_region(image: np.ndarray, bounds: tuple[int, int, int, int]) -> None:
    """Apply block mosaic in-place to one inclusive-free image rectangle."""
    left, top, right, bottom = bounds
    region = Image.fromarray(image[top:bottom, left:right])
    reduced_size = (
        max(1, round(region.width / EYE_PIXEL_BLOCK_SIZE)),
        max(1, round(region.height / EYE_PIXEL_BLOCK_SIZE)),
    )
    pixelated = region.resize(reduced_size, Image.Resampling.BILINEAR).resize(
        region.size,
        Image.Resampling.NEAREST,
    )
    image[top:bottom, left:right] = np.asarray(pixelated)


def eye_bounds(
    landmarks: list[object],
    indices: tuple[int, ...],
    image_width: int,
    image_height: int,
) -> tuple[int, int, int, int]:
    """Create a padded eye rectangle from normalized Face Mesh landmarks."""
    x_values = [landmarks[index].x * image_width for index in indices]
    y_values = [landmarks[index].y * image_height for index in indices]
    pad_x = max(8, round((max(x_values) - min(x_values)) * EYE_PADDING_X))
    pad_y = max(8, round((max(y_values) - min(y_values)) * EYE_PADDING_Y))
    left = max(0, round(min(x_values)) - pad_x)
    top = max(0, round(min(y_values)) - pad_y)
    right = min(image_width, round(max(x_values)) + pad_x)
    bottom = min(image_height, round(max(y_values)) + pad_y)
    if right <= left or bottom <= top:
        raise ValueError("Face Mesh produced an invalid eye region")
    return left, top, right, bottom


def pixelate_eyes(panel: np.ndarray, face_mesh: object) -> np.ndarray:
    """De-identify a Grad-CAM panel by pixelating its detected left and right eyes."""
    result = face_mesh.process(panel)
    if not result.multi_face_landmarks:
        raise ValueError("Face Mesh could not locate a face for eye pixelation")
    if len(result.multi_face_landmarks) != 1:
        raise ValueError("Expected exactly one face for eye pixelation")

    masked = panel.copy()
    landmarks = result.multi_face_landmarks[0].landmark
    height, width = masked.shape[:2]
    for indices in (LEFT_EYE_INDICES, RIGHT_EYE_INDICES):
        pixelate_region(masked, eye_bounds(landmarks, indices, width, height))
    return masked


def plot_montage(rows: pd.DataFrame) -> None:
    configure_matplotlib()
    figure, axes = plt.subplots(2, 3, figsize=(7.1, 5.9), facecolor="white")
    figure.subplots_adjust(left=0.02, right=0.98, bottom=0.02, top=0.98, wspace=0.04, hspace=0.04)
    with mp.solutions.face_mesh.FaceMesh(
        static_image_mode=True,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
    ) as face_mesh:
        for axis, row in zip(axes.flat, rows.itertuples(index=False)):
            panel = crop_patient_gradcam(Path(row.output_png))
            axis.imshow(pixelate_eyes(panel, face_mesh), interpolation="nearest")
            axis.set_axis_off()
    figure.savefig(PNG_PATH, dpi=600, facecolor="white")
    figure.savefig(PDF_PATH, format="pdf", facecolor="white")
    figure.savefig(SVG_PATH, format="svg", facecolor="white")
    figure.savefig(TIFF_PATH, dpi=600, format="tiff", facecolor="white")
    plt.close(figure)


def main() -> None:
    rows = load_selected_rows()
    plot_montage(rows)
    rows.loc[:, [
        "sample_id",
        "fold",
        "true_class",
        "predicted_class",
        "prob_patient",
        "target_class",
        "target_name",
        "output_png",
    ]].to_csv(CHECK_PATH, index=False, float_format="%.6f")
    print("Figure 5 2x3 Patient-logit Grad-CAM montage created")
    print(f"Selected IDs: {', '.join(SELECTED_IDS)}")
    for path in (PNG_PATH, PDF_PATH, SVG_PATH, TIFF_PATH, CHECK_PATH):
        print(path)


if __name__ == "__main__":
    main()
