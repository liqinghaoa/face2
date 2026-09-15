"""Create a three-stage ROI preprocessing figure from one R3DPR face image.

This is a presentation-only companion script. It imports the established ROI
preprocessing functions but does not modify their source files, datasets, or
training splits. The final figure contains:

    A. R3DPR-standardized facial image
    B. ROI definition on the aligned 224x224 face
    C. Final masked Eye, Cheek, and Lip model inputs
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Rectangle
import mediapipe as mp
import numpy as np
import torch


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import preprocess_global_aligned_face_parsing_roi_dataset_224_canvas as roi  # noqa: E402


# Publication defaults. SVG keeps text editable for later manuscript adjustment.
mpl.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "DejaVu Sans", "Liberation Sans"],
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
        "font.size": 8,
    }
)

ROI_DISPLAY = {
    "eye_roi": {"label": "Eye ROI", "color": "#2878B5"},
    "cheek_roi": {"label": "Cheek ROI", "color": "#3D9A65"},
    "lip_roi": {"label": "Lip ROI", "color": "#D7862E"},
}
CAPTION_NOTE = (
    "ROI boundaries are shown schematically for visualization; the actual "
    "regions were automatically determined using facial landmarks and semantic "
    "parsing rather than manual annotation."
)


@dataclass
class FigureAssets:
    image_id: str
    original_rgb: np.ndarray
    aligned_rgb: np.ndarray
    label_map: np.ndarray
    final_face_mask: np.ndarray
    boxes: dict[str, roi.BBox]
    masked_rois: dict[str, np.ndarray]
    intermediate_source: str
    global_metrics: dict[str, Any] | None


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Create a Regional facial image preprocessing and ROI definition figure from one R3DPR image."
    )
    parser.add_argument("--input-image", type=Path, required=True, help="One R3DPR-standardized RGB image.")
    parser.add_argument("--output-dir", type=Path, default=None, help="Directory for figure outputs.")
    parser.add_argument("--config", type=Path, default=None, help="Optional ROI preprocessing YAML configuration.")
    parser.add_argument(
        "--global-intermediate-dir",
        type=Path,
        default=None,
        help="Optional matching intermediate directory. By default, intermediates are regenerated from the input image.",
    )
    parser.add_argument("--dpi", type=int, default=600, help="PNG export resolution.")
    parser.add_argument("--figure-width", type=float, default=9.2, help="Figure width in inches.")
    parser.add_argument("--figure-height", type=float, default=3.15, help="Figure height in inches.")
    parser.add_argument("--save-debug", action="store_true", help="Save aligned image, bbox overlay, and ROI assets for QC.")
    parser.add_argument("--overwrite", action="store_true", help="Allow overwriting existing final outputs.")
    return parser


def resolve_optional_path(path: Path | None) -> Path | None:
    if path is None:
        return None
    return path.expanduser().resolve()


def configure_roi_args(cli_args: argparse.Namespace, input_path: Path, output_dir: Path) -> argparse.Namespace:
    """Load the unchanged ROI YAML config and override only single-image paths."""
    config_args = ["--project-root", str(PROJECT_ROOT)]
    if cli_args.config is not None:
        config_args.extend(["--config", str(cli_args.config.expanduser().resolve())])
    args = roi.parse_args(config_args)
    args.project_root = PROJECT_ROOT
    args.image_dir = input_path.parent
    args.output_dir = output_dir
    args.global_intermediate_dir = resolve_optional_path(cli_args.global_intermediate_dir)
    args.parsing_checkpoint = roi.resolve_path(args.parsing_checkpoint, PROJECT_ROOT).resolve()
    args.roi_types = roi.parse_csv_list(args.roi_types)
    args.core_roi_types = roi.parse_csv_list(args.core_roi_types)
    if int(args.image_size) != 224 or int(args.canvas_size) != 224:
        raise ValueError("This figure script requires image_size=224 and canvas_size=224 in the ROI configuration.")
    if not args.parsing_checkpoint.is_file():
        raise FileNotFoundError(f"Face parsing checkpoint does not exist: {args.parsing_checkpoint}")
    return args


def validate_input_image(input_path: Path) -> np.ndarray:
    if not input_path.is_file():
        raise FileNotFoundError(f"Input image does not exist: {input_path}")
    if input_path.suffix.lower() not in roi.VALID_IMAGE_EXTENSIONS:
        raise ValueError(f"Unsupported image extension: {input_path.suffix}")
    image_rgb = roi.read_rgb_png(input_path)
    if image_rgb.ndim != 3 or image_rgb.shape[2] != 3:
        raise ValueError("Input image must be an RGB image.")
    if image_rgb.shape[:2] != (512, 512):
        raise ValueError(
            f"Expected a 512x512 R3DPR-standardized image, got {image_rgb.shape[1]}x{image_rgb.shape[0]}."
        )
    return image_rgb


def initialize_models(args: argparse.Namespace) -> tuple[Any, Any, Any, torch.device]:
    parsing_device = roi.parsing.resolve_parsing_device(str(args.parsing_device))
    parsing_model = roi.parsing.load_face_parsing_model(
        str(args.parsing_model), args.parsing_checkpoint, parsing_device
    )
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
    return detector, face_mesh, parsing_model, parsing_device


def process_one_image(
    image_id: str,
    input_path: Path,
    args: argparse.Namespace,
    padding_color: tuple[int, int, int],
) -> FigureAssets:
    """Reuse established ROI functions without entering the batch-processing main()."""
    original_rgb = validate_input_image(input_path)
    detector, face_mesh, parsing_model, parsing_device = initialize_models(args)
    try:
        intermediates = roi.load_intermediates(image_id, args.global_intermediate_dir, face_mesh)
        if intermediates is None:
            intermediates = roi.generate_intermediates(
                image_id,
                input_path.parent,
                detector,
                face_mesh,
                parsing_model,
                parsing_device,
                args,
            )
        if intermediates.global_status != "success":
            raise roi.RoiFailure(
                f"global_preprocessing_failed:{intermediates.global_failure_reason or intermediates.global_status}"
            )
        # Cached intermediates intentionally omit the original image. Panel A must
        # always show exactly the user-supplied R3DPR image.
        intermediates.original_rgb = original_rgb

        with tempfile.TemporaryDirectory(prefix="roi_definition_") as temp_dir:
            temp_root = Path(temp_dir)
            dirs = {
                "roi_raw": temp_root / "roi_raw",
                "roi_masked": temp_root / "roi_masked",
            }
            for roi_type in ("eye_roi", "cheek_roi", "lip_roi"):
                (dirs["roi_raw"] / roi_type).mkdir(parents=True, exist_ok=True)
                (dirs["roi_masked"] / roi_type).mkdir(parents=True, exist_ok=True)

            boxes: dict[str, roi.BBox] = {}
            masked_rois: dict[str, np.ndarray] = {}
            for roi_type in ("eye_roi", "cheek_roi", "lip_roi"):
                record, bbox, _, masked_image = roi.process_roi(
                    {"ID": image_id},
                    roi_type,
                    intermediates,
                    dirs,
                    args,
                    padding_color,
                )
                if not bool(record.get("roi_success", False)) or bbox is None or masked_image is None:
                    raise roi.RoiFailure(f"{roi_type}_failed:{record.get('failure_reason', 'unknown_failure')}")
                boxes[roi_type] = bbox
                masked_rois[roi_type] = masked_image
                if roi_type == "cheek_roi":
                    boxes["left_cheek"] = roi.BBox(
                        int(record["left_cheek_bbox_x1"]),
                        int(record["left_cheek_bbox_y1"]),
                        int(record["left_cheek_bbox_x2"]) + 1,
                        int(record["left_cheek_bbox_y2"]) + 1,
                    )
                    boxes["right_cheek"] = roi.BBox(
                        int(record["right_cheek_bbox_x1"]),
                        int(record["right_cheek_bbox_y1"]),
                        int(record["right_cheek_bbox_x2"]) + 1,
                        int(record["right_cheek_bbox_y2"]) + 1,
                    )
    finally:
        face_mesh.close()
        detector.close()

    return FigureAssets(
        image_id=image_id,
        original_rgb=original_rgb,
        aligned_rgb=intermediates.aligned_rgb,
        label_map=intermediates.label_map,
        final_face_mask=intermediates.final_face_mask,
        boxes=boxes,
        masked_rois=masked_rois,
        intermediate_source=intermediates.source,
        global_metrics=intermediates.global_metrics,
    )


def add_panel_label(axis: plt.Axes, label: str) -> None:
    axis.text(
        -0.04,
        1.05,
        label,
        transform=axis.transAxes,
        ha="right",
        va="bottom",
        fontsize=12,
        fontweight="bold",
        color="#202020",
        clip_on=False,
    )


def draw_bbox(axis: plt.Axes, bbox: roi.BBox, color: str) -> None:
    axis.add_patch(
        Rectangle(
            (bbox.x1, bbox.y1),
            bbox.w,
            bbox.h,
            linewidth=1.35,
            edgecolor=color,
            facecolor=color,
            alpha=0.10,
            joinstyle="round",
        )
    )


def draw_roi_definition(axis: plt.Axes, assets: FigureAssets) -> None:
    axis.imshow(assets.aligned_rgb)
    draw_bbox(axis, assets.boxes["eye_roi"], ROI_DISPLAY["eye_roi"]["color"])
    draw_bbox(axis, assets.boxes["left_cheek"], ROI_DISPLAY["cheek_roi"]["color"])
    draw_bbox(axis, assets.boxes["right_cheek"], ROI_DISPLAY["cheek_roi"]["color"])
    draw_bbox(axis, assets.boxes["lip_roi"], ROI_DISPLAY["lip_roi"]["color"])
    axis.set_xlim(-48, 272)
    axis.set_ylim(248, -33)
    axis.set_xticks([])
    axis.set_yticks([])
    for spine in axis.spines.values():
        spine.set_visible(False)

    eye = assets.boxes["eye_roi"]
    lip = assets.boxes["lip_roi"]
    left_cheek = assets.boxes["left_cheek"]
    axis.annotate(
        "Eye ROI",
        xy=(eye.x1 + eye.w / 2, eye.y1),
        xytext=(0.50, 1.03),
        textcoords="axes fraction",
        ha="center",
        va="bottom",
        fontsize=7.5,
        color=ROI_DISPLAY["eye_roi"]["color"],
        arrowprops={"arrowstyle": "-", "color": ROI_DISPLAY["eye_roi"]["color"], "lw": 0.8},
        annotation_clip=False,
    )
    axis.annotate(
        "Cheek ROI",
        xy=(left_cheek.x1, left_cheek.y1 + left_cheek.h / 2),
        xytext=(-0.03, 0.52),
        textcoords="axes fraction",
        ha="right",
        va="center",
        fontsize=7.5,
        color=ROI_DISPLAY["cheek_roi"]["color"],
        arrowprops={"arrowstyle": "-", "color": ROI_DISPLAY["cheek_roi"]["color"], "lw": 0.8},
        annotation_clip=False,
    )
    axis.annotate(
        "Lip ROI",
        xy=(lip.x2, lip.y1 + lip.h / 2),
        xytext=(1.02, 0.68),
        textcoords="axes fraction",
        ha="left",
        va="center",
        fontsize=7.5,
        color=ROI_DISPLAY["lip_roi"]["color"],
        arrowprops={"arrowstyle": "-", "color": ROI_DISPLAY["lip_roi"]["color"], "lw": 0.8},
        annotation_clip=False,
    )
    axis.text(
        0.50,
        -0.04,
        "ROI definition on aligned face (224 x 224)",
        transform=axis.transAxes,
        ha="center",
        va="top",
        fontsize=7.2,
        color="#303030",
    )


def draw_arrow(axis: plt.Axes, label: str) -> None:
    axis.set_axis_off()
    axis.text(0.50, 0.67, label, ha="center", va="bottom", fontsize=7.2, color="#4D4D4D", wrap=True)
    axis.add_patch(
        FancyArrowPatch(
            (0.08, 0.44),
            (0.92, 0.44),
            arrowstyle="-|>",
            mutation_scale=10,
            lw=1.0,
            color="#6B6B6B",
            transform=axis.transAxes,
        )
    )


def create_figure(assets: FigureAssets, figure_width: float, figure_height: float) -> plt.Figure:
    figure = plt.figure(figsize=(figure_width, figure_height), facecolor="white")
    grid = figure.add_gridspec(
        1,
        7,
        width_ratios=[1.0, 0.43, 1.35, 0.43, 0.83, 0.83, 0.83],
        left=0.025,
        right=0.995,
        bottom=0.15,
        top=0.83,
        wspace=0.08,
    )
    source_axis = figure.add_subplot(grid[0, 0])
    arrow_one = figure.add_subplot(grid[0, 1])
    roi_axis = figure.add_subplot(grid[0, 2])
    arrow_two = figure.add_subplot(grid[0, 3])
    eye_axis = figure.add_subplot(grid[0, 4])
    cheek_axis = figure.add_subplot(grid[0, 5])
    lip_axis = figure.add_subplot(grid[0, 6])

    source_axis.imshow(assets.original_rgb)
    source_axis.set_axis_off()
    source_axis.text(
        0.50,
        -0.05,
        "R3DPR-standardized\nfacial image",
        transform=source_axis.transAxes,
        ha="center",
        va="top",
        fontsize=7.5,
        color="#303030",
    )
    add_panel_label(source_axis, "A")

    draw_arrow(arrow_one, "Facial alignment\nand parsing")
    draw_roi_definition(roi_axis, assets)
    add_panel_label(roi_axis, "B")
    draw_arrow(arrow_two, "ROI extraction\nand masking")

    final_axes = (("eye_roi", "Eye", eye_axis), ("cheek_roi", "Cheek", cheek_axis), ("lip_roi", "Lip", lip_axis))
    for roi_type, display_name, axis in final_axes:
        axis.imshow(assets.masked_rois[roi_type])
        axis.set_axis_off()
        subtitle = "224 x 224"
        if roi_type == "cheek_roi":
            subtitle = "224 x 224\nleft + right combined"
        axis.set_title(f"{display_name}\n{subtitle}", fontsize=7.2, pad=4, color="#303030")
    add_panel_label(eye_axis, "C")
    figure.canvas.draw()
    eye_bounds = eye_axis.get_position()
    lip_bounds = lip_axis.get_position()
    figure.text(
        (eye_bounds.x0 + lip_bounds.x1) / 2,
        0.93,
        "Final masked regional inputs",
        ha="center",
        va="bottom",
        fontsize=8.5,
        fontweight="bold",
        color="#202020",
    )
    return figure


def save_debug_assets(assets: FigureAssets, output_dir: Path) -> None:
    debug_dir = output_dir / "debug"
    debug_dir.mkdir(parents=True, exist_ok=True)
    roi.alignment.save_png(debug_dir / "aligned_224.png", assets.aligned_rgb)
    roi.alignment.save_png(debug_dir / "final_face_mask.png", np.repeat(assets.final_face_mask[:, :, None], 3, axis=2))
    for roi_type, image in assets.masked_rois.items():
        roi.alignment.save_png(debug_dir / f"{roi_type}_masked_224.png", image)
    overlay = assets.aligned_rgb.copy()
    for roi_type, color in (("eye_roi", (40, 120, 181)), ("left_cheek", (61, 154, 101)), ("right_cheek", (61, 154, 101)), ("lip_roi", (215, 134, 46))):
        box = assets.boxes[roi_type]
        import cv2

        cv2.rectangle(overlay, (box.x1, box.y1), (box.x2 - 1, box.y2 - 1), color, 1)
    roi.alignment.save_png(debug_dir / "roi_definition_overlay.png", overlay)


def serialize_bbox(box: roi.BBox) -> dict[str, int]:
    return {"x1": box.x1, "y1": box.y1, "x2": box.x2, "y2": box.y2, "width": box.w, "height": box.h}


def save_outputs(assets: FigureAssets, figure: plt.Figure, output_dir: Path, dpi: int, overwrite: bool) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    base = output_dir / f"{assets.image_id}_regional_roi_preprocessing"
    outputs = {
        "png": base.with_suffix(".png"),
        "pdf": base.with_suffix(".pdf"),
        "svg": base.with_suffix(".svg"),
        "tiff": base.with_suffix(".tiff"),
        "metadata": base.with_name(f"{base.name}_metadata.json"),
        "caption": base.with_name(f"{base.name}_caption.txt"),
    }
    if not overwrite:
        existing = [path for path in outputs.values() if path.exists()]
        if existing:
            raise FileExistsError(f"Output already exists: {existing[0]}. Use --overwrite to replace it.")
    figure.savefig(outputs["png"], dpi=dpi, facecolor="white")
    figure.savefig(outputs["pdf"], facecolor="white")
    figure.savefig(outputs["svg"], facecolor="white")
    figure.savefig(outputs["tiff"], dpi=dpi, facecolor="white")
    metadata = {
        "image_id": assets.image_id,
        "input_shape": list(assets.original_rgb.shape),
        "aligned_shape": list(assets.aligned_rgb.shape),
        "roi_canvas_shape": {key: list(image.shape) for key, image in assets.masked_rois.items()},
        "intermediate_source": assets.intermediate_source,
        "roi_bboxes_224": {key: serialize_bbox(box) for key, box in assets.boxes.items()},
        "caption_note": CAPTION_NOTE,
    }
    outputs["metadata"].write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    outputs["caption"].write_text(CAPTION_NOTE + "\n", encoding="utf-8")
    return outputs


def main(argv: list[str] | None = None) -> int:
    cli_args = build_parser().parse_args(argv)
    input_path = cli_args.input_image.expanduser().resolve()
    # The intermediate generator resolves the source file from this ID, so it
    # must match the input filename stem.
    image_id = input_path.stem
    output_dir = (
        cli_args.output_dir.expanduser().resolve()
        if cli_args.output_dir is not None
        else input_path.parent / f"{image_id}_roi_preprocessing_figure"
    )
    args = configure_roi_args(cli_args, input_path, output_dir)
    padding_color = roi.parse_padding_color(args.padding_color)
    assets = process_one_image(image_id, input_path, args, padding_color)
    figure = create_figure(assets, float(cli_args.figure_width), float(cli_args.figure_height))
    try:
        outputs = save_outputs(assets, figure, output_dir, int(cli_args.dpi), bool(cli_args.overwrite))
    finally:
        plt.close(figure)
    if cli_args.save_debug:
        save_debug_assets(assets, output_dir)
    print("Regional ROI preprocessing figure created")
    for name, path in outputs.items():
        print(f"{name}: {path}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (FileNotFoundError, ValueError, roi.RoiFailure, RuntimeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(2)
