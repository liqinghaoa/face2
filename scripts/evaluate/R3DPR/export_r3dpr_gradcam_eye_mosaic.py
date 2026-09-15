"""Export selected OOF Grad-CAM examples with pixelated eye regions for publication."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.experiment_utils import load_yaml
from scripts.evaluate.R3DPR.generate_r3dpr_binary_gradcam import (
    GradCAM,
    load_model,
    make_dataset_row,
)


DEFAULT_SCREENING_SUBDIR = "figures/gradcam_oof_screening"
DEFAULT_OUTPUT_SUBDIR = "figures/gradcam_oof_screening/figure5_eye_mosaic"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment-dir", type=Path, required=True)
    parser.add_argument("--sample-id", action="append", required=True, dest="sample_ids")
    parser.add_argument("--screening-subdir", default=DEFAULT_SCREENING_SUBDIR)
    parser.add_argument("--output-subdir", default=DEFAULT_OUTPUT_SUBDIR)
    parser.add_argument(
        "--fallback-metadata-root",
        type=Path,
        default=None,
        help="Optional metadata directory used only when a sample is absent from the experiment input metadata.",
    )
    parser.add_argument("--eye-box-width", type=int, default=50)
    parser.add_argument("--eye-box-height", type=int, default=28)
    parser.add_argument("--mosaic-block-size", type=int, default=7)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def project_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def experiment_path(experiment_dir: Path, value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else experiment_dir / path


def configure_matplotlib() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
            "font.size": 8,
            "figure.dpi": 150,
        }
    )


def image_path(image_root: Path, filename_template: str, sample_id: str) -> Path:
    return image_root / filename_template.replace("{ID}", sample_id)


def metadata_path(sample_id: str, primary_root: Path, fallback_root: Path | None) -> Path:
    primary = primary_root / f"{sample_id}.json"
    if primary.is_file():
        return primary
    if fallback_root is not None:
        fallback = fallback_root / f"{sample_id}.json"
        if fallback.is_file():
            return fallback
    searched = [str(primary)]
    if fallback_root is not None:
        searched.append(str(fallback_root / f"{sample_id}.json"))
    raise FileNotFoundError(f"Missing eye-center metadata for ID {sample_id}: {searched}")


def eye_centers_on_canvas(metadata: dict, expected_size: tuple[int, int]) -> list[tuple[float, float]]:
    matrix = np.asarray(metadata["affine_source_to_canvas"], dtype=np.float64)
    if matrix.shape != (3, 3):
        raise ValueError("affine_source_to_canvas must be 3 x 3")
    centers: list[tuple[float, float]] = []
    for key in ("source_left_eye_center", "source_right_eye_center"):
        point = np.asarray([*metadata[key], 1.0], dtype=np.float64)
        transformed = matrix @ point
        x, y = transformed[:2] / transformed[2]
        width, height = expected_size
        if not (0.0 <= x < width and 0.0 <= y < height):
            raise ValueError(f"Mapped {key} is outside the image: {(x, y)}")
        centers.append((float(x), float(y)))
    return centers


def clipped_box(center: tuple[float, float], size: tuple[int, int], image_size: tuple[int, int]) -> tuple[int, int, int, int]:
    center_x, center_y = center
    box_width, box_height = size
    image_width, image_height = image_size
    left = max(0, int(round(center_x - box_width / 2)))
    top = max(0, int(round(center_y - box_height / 2)))
    right = min(image_width, left + box_width)
    bottom = min(image_height, top + box_height)
    if right <= left or bottom <= top:
        raise ValueError(f"Invalid eye mosaic box: {(left, top, right, bottom)}")
    return left, top, right, bottom


def pixelate(image: Image.Image, box: tuple[int, int, int, int], block_size: int) -> None:
    if block_size < 2:
        raise ValueError("mosaic-block-size must be at least 2")
    left, top, right, bottom = box
    crop = image.crop(box)
    low_size = (max(1, crop.width // block_size), max(1, crop.height // block_size))
    reduced = crop.resize(low_size, Image.Resampling.BOX)
    image.paste(reduced.resize(crop.size, Image.Resampling.NEAREST), (left, top))


def load_fold_cam(raw_cam_dir: Path, fold: int, sample_id: str) -> tuple[np.ndarray, Path]:
    path = raw_cam_dir / f"fold_{fold}_raw_cams.npz"
    if not path.is_file():
        raise FileNotFoundError(f"Missing raw CAM archive: {path}")
    with np.load(path) as archive:
        ids = archive["sample_id"].astype(str)
        matches = np.flatnonzero(ids == sample_id)
        if len(matches) != 1:
            raise ValueError(f"Expected one CAM for ID {sample_id} in {path}, found {len(matches)}")
        cam = archive["cam"][int(matches[0])].astype(np.float32, copy=False)
    return cam, path


def recompute_cam(
    config: dict,
    experiment_dir: Path,
    row: pd.Series,
    sample_id: str,
    device: torch.device,
) -> np.ndarray:
    series = row.copy()
    series["sample_id"] = sample_id
    series["fold"] = int(row["fold"])
    series["binary_label"] = int(row["binary_label"])
    series["pred_class"] = int(row["pred_class"])
    series["patient_group_id"] = str(row["patient_group_id"])
    checkpoint = experiment_dir / f"fold_{int(row['fold'])}" / "checkpoints" / "best_macro_auc.pth"
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Missing held-out fold checkpoint: {checkpoint}")
    model = load_model(config, checkpoint, device)
    gradcam = GradCAM(model)
    try:
        sample = make_dataset_row(series, config)
        logits, cam = gradcam.calculate(sample["image"].unsqueeze(0).to(device), int(row["pred_class"]))
        probability = torch.softmax(torch.from_numpy(logits), dim=0).numpy()
        if int(probability.argmax()) != int(row["pred_class"]):
            raise RuntimeError(f"OOF replay predicted class mismatch for ID {sample_id}")
        stored = row[["prob_control", "prob_patient"]].to_numpy(dtype=float)
        if float(np.max(np.abs(probability - stored))) > 1e-3:
            raise RuntimeError(f"OOF replay probability mismatch for ID {sample_id}")
        return cam.astype(np.float32, copy=False)
    finally:
        gradcam.close()
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()


def draw_example(image_rgb: np.ndarray, cam: np.ndarray, output_path: Path) -> None:
    if cam.shape != image_rgb.shape[:2]:
        raise ValueError(f"CAM shape {cam.shape} does not match image shape {image_rgb.shape[:2]}")
    heatmap = plt.get_cmap("jet")(cam)[..., :3]
    overlay = np.clip(0.55 * image_rgb + 0.45 * heatmap, 0.0, 1.0)
    figure, axes = plt.subplots(1, 2, figsize=(6.4, 3.5), constrained_layout=True)
    axes[0].imshow(image_rgb)
    axes[0].set_title("(a) Input image", loc="left", fontweight="bold")
    axes[1].imshow(overlay)
    axes[1].set_title("(b) Grad-CAM", loc="left", fontweight="bold")
    for axis in axes:
        axis.set_axis_off()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=300, bbox_inches="tight", pad_inches=0.03)
    plt.close(figure)


def draw_combined(example_paths: list[Path], output_path: Path) -> None:
    columns = 2
    rows = int(np.ceil(len(example_paths) / columns))
    figure, axes = plt.subplots(rows, columns, figsize=(12.8, 3.6 * rows), constrained_layout=True)
    axes_array = np.asarray(axes).reshape(-1)
    for axis, path in zip(axes_array, example_paths):
        axis.imshow(plt.imread(path))
        axis.set_axis_off()
    for axis in axes_array[len(example_paths) :]:
        axis.set_axis_off()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=300, bbox_inches="tight", pad_inches=0.03)
    plt.close(figure)


def export_examples(args: argparse.Namespace) -> Path:
    experiment_dir = project_path(args.experiment_dir)
    config = load_yaml(experiment_dir / "config_snapshot.yaml")
    data = config["data"]
    image_root = project_path(data["image_root"])
    primary_metadata_root = image_root.parent / "metadata"
    fallback_metadata_root = project_path(args.fallback_metadata_root) if args.fallback_metadata_root else None
    screening_dir = experiment_path(experiment_dir, args.screening_subdir)
    raw_cam_dir = screening_dir / "raw_cam_maps"
    output_dir = experiment_path(experiment_dir, args.output_subdir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    device = torch.device(args.device)

    oof = pd.read_csv(experiment_dir / "oof_predictions.csv", dtype={"sample_id": "string"})
    oof = oof.set_index("sample_id", verify_integrity=True)
    sample_ids = [str(sample_id) for sample_id in args.sample_ids]
    if len(sample_ids) != len(set(sample_ids)):
        raise ValueError("Each --sample-id must be provided only once")
    missing_ids = sorted(set(sample_ids).difference(oof.index.astype(str)))
    if missing_ids:
        raise ValueError(f"Requested IDs are absent from OOF predictions: {missing_ids}")

    image_size = (int(data["image_width"]), int(data["image_height"]))
    records: list[dict[str, object]] = []
    example_paths: list[Path] = []
    for sample_id in sample_ids:
        row = oof.loc[sample_id]
        source_image = image_path(image_root, str(data["image_filename_template"]), sample_id)
        if not source_image.is_file():
            raise FileNotFoundError(f"Missing input image for ID {sample_id}: {source_image}")
        with Image.open(source_image) as source:
            mosaic_image = source.convert("RGB")
        if mosaic_image.size != image_size:
            raise ValueError(f"Unexpected size for {source_image}: {mosaic_image.size} != {image_size}")
        metadata_file = metadata_path(sample_id, primary_metadata_root, fallback_metadata_root)
        metadata = json.loads(metadata_file.read_text(encoding="utf-8"))
        centers = eye_centers_on_canvas(metadata, image_size)
        boxes = [
            clipped_box(center, (args.eye_box_width, args.eye_box_height), image_size)
            for center in centers
        ]
        for box in boxes:
            pixelate(mosaic_image, box, args.mosaic_block_size)
        try:
            cam, cam_path = load_fold_cam(raw_cam_dir, int(row["fold"]), sample_id)
        except ValueError as error:
            if "found 0" not in str(error):
                raise
            cam = recompute_cam(config, experiment_dir, row, sample_id, device)
            cam_path = output_dir / "recomputed_raw_cams" / f"{sample_id}_fold_{int(row['fold'])}.npy"
            cam_path.parent.mkdir(parents=True, exist_ok=True)
            np.save(cam_path, cam)
        output_png = output_dir / f"{sample_id}_eye_mosaic_gradcam.png"
        if output_png.exists() and not args.overwrite:
            raise FileExistsError(f"Output already exists: {output_png}; use --overwrite to replace it")
        draw_example(np.asarray(mosaic_image, dtype=np.float32) / 255.0, cam, output_png)
        example_paths.append(output_png)
        records.append(
            {
                "sample_id": sample_id,
                "fold": int(row["fold"]),
                "source_image": str(source_image),
                "metadata_path": str(metadata_file),
                "raw_cam_path": str(cam_path),
                "output_png": str(output_png),
                "left_eye_center_x": centers[0][0],
                "left_eye_center_y": centers[0][1],
                "right_eye_center_x": centers[1][0],
                "right_eye_center_y": centers[1][1],
                "left_eye_box": ",".join(map(str, boxes[0])),
                "right_eye_box": ",".join(map(str, boxes[1])),
                "mosaic_block_size": args.mosaic_block_size,
            }
        )
    pd.DataFrame(records).to_csv(output_dir / "figure5_eye_mosaic_manifest.csv", index=False, encoding="utf-8-sig")
    draw_combined(example_paths, output_dir / "figure5_eye_mosaic_combined.png")
    return output_dir


def main() -> None:
    configure_matplotlib()
    output_dir = export_examples(parse_args())
    print(f"Eye-mosaic Grad-CAM examples written to: {output_dir}")


if __name__ == "__main__":
    main()
