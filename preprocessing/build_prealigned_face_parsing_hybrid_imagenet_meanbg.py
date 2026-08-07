"""Build ImageNet-mean-background face images from already aligned inputs.

This is the pre-aligned counterpart of the global face-parsing pipeline.  It
intentionally does *not* run face detection, FaceMesh, crop selection, or an
affine/similarity transform.  Inputs are assumed to already share a canonical
face coordinate system; they are only resized to the requested square canvas
before semantic parsing and hybrid-mask construction.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Sequence

import cv2
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from preprocessing import (  # noqa: E402
    build_global_face_parsing_hybrid_foreheadrepair_blackbg_224_png_strict as hybrid,
)
from preprocessing import (  # noqa: E402
    build_global_face_parsing_regularmask_blackbg_224_png_strict as parsing,
)
from utils.preprocess_ablation_utils import (  # noqa: E402
    apply_background,
    feather_mask,
    read_rgb,
    save_rgb,
)


LOG_COLUMNS = (
    "ID",
    "input_path",
    "aligned_rgb_path",
    "final_mask_path",
    "output_path",
    "status",
    "error_message",
    "input_height",
    "input_width",
    "image_size",
    "alignment_mode",
    "resize_interpolation",
    "mask_area_ratio",
    "forehead_repair_applied",
    "forehead_repair_reason",
    "hair_inside_final_mask_ratio",
    "neck_inside_final_mask_ratio",
    "cloth_inside_final_mask_ratio",
    "background_inside_final_mask_ratio",
)


def _resolve(value: Path) -> Path:
    value = value.expanduser()
    return value.resolve() if value.is_absolute() else (PROJECT_ROOT / value).resolve()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Create hybrid face-parsing ImageNet-mean-background images from "
            "already aligned canonical images."
        )
    )
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument("--input-dir", type=Path)
    input_group.add_argument("--input-file", type=Path)
    parser.add_argument(
        "--input-pattern",
        default="*/canonical_fixedcam.png",
        help="Glob relative to --input-dir; ignored with --input-file.",
    )
    parser.add_argument(
        "--id-from",
        choices=("parent", "stem"),
        default="parent",
        help="How to infer each image ID when scanning --input-dir.",
    )
    parser.add_argument(
        "--image-id",
        default=None,
        help="Required with --input-file; ignored with --input-dir.",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--flat-output",
        action="store_true",
        help="Write final {ID}.png files directly under --output-dir instead of images/.",
    )
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--parsing-model", default="bisenet")
    parser.add_argument(
        "--parsing-checkpoint",
        type=Path,
        default=Path("preprocessing/checkpoints/face_parsing/79999_iter.pth"),
    )
    parser.add_argument("--parsing-device", default="auto")
    parser.add_argument("--forehead-band-ratio", type=float, default=0.35)
    parser.add_argument("--hair-repair-threshold", type=float, default=0.10)
    parser.add_argument("--jaggedness-threshold", type=float, default=0.04)
    parser.add_argument(
        "--enable-jaggedness-trigger", action=argparse.BooleanOptionalAction, default=False
    )
    parser.add_argument("--forehead-expand-ratio", type=float, default=0.18)
    parser.add_argument("--side-expand-ratio", type=float, default=0.05)
    parser.add_argument("--chin-expand-ratio", type=float, default=0.03)
    parser.add_argument("--feather-kernel", type=int, default=11)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def _save_mask(path: Path, mask: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    binary = ((np.asarray(mask) > 0).astype(np.uint8) * 255)
    ok, encoded = cv2.imencode(".png", binary, [cv2.IMWRITE_PNG_COMPRESSION, 3])
    if not ok:
        raise ValueError(f"Could not encode mask PNG: {path}")
    path.write_bytes(encoded.tobytes())


def _save_label_map(path: Path, label_map: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    values = np.asarray(label_map)
    if values.ndim != 2 or values.min() < 0 or values.max() > 255:
        raise ValueError(f"Unsupported parsing-label range: {values.min()}..{values.max()}")
    ok, encoded = cv2.imencode(".png", values.astype(np.uint8))
    if not ok:
        raise ValueError(f"Could not encode parsing-label PNG: {path}")
    path.write_bytes(encoded.tobytes())


def _collect_inputs(args: argparse.Namespace) -> list[tuple[str, Path]]:
    if args.input_file is not None:
        if not args.image_id:
            raise ValueError("--image-id is required when using --input-file")
        path = _resolve(args.input_file)
        if not path.is_file():
            raise FileNotFoundError(f"Input file does not exist: {path}")
        return [(str(args.image_id), path)]

    input_dir = _resolve(args.input_dir)
    if not input_dir.is_dir():
        raise NotADirectoryError(f"Input directory does not exist: {input_dir}")
    paths = sorted(path for path in input_dir.glob(args.input_pattern) if path.is_file())
    if not paths:
        raise FileNotFoundError(
            f"No files matched {args.input_pattern!r} under {input_dir}"
        )
    records = [
        (path.parent.name if args.id_from == "parent" else path.stem, path)
        for path in paths
    ]
    ids = [image_id for image_id, _ in records]
    if len(set(ids)) != len(ids):
        raise ValueError("Input pattern produced duplicate image IDs")
    return records


def _prepare_dirs(
    output_dir: Path,
    overwrite: bool,
    flat_output: bool,
) -> dict[str, Path]:
    if output_dir.exists() and any(output_dir.iterdir()) and not overwrite:
        raise FileExistsError(
            f"Output directory is not empty: {output_dir}. Use --overwrite to rebuild it."
        )
    dirs = {
        "root": output_dir,
        "images": output_dir if flat_output else output_dir / "images",
        "aligned_rgb": output_dir / "aligned_rgb",
        "final_mask": output_dir / "final_mask",
        "selected_semantic_mask": output_dir / "selected_semantic_mask",
        "semantic_regularized_mask": output_dir / "semantic_regularized_mask",
        "candidate_envelope": output_dir / "candidate_envelope",
        "parsing_label": output_dir / "parsing_label",
        "logs": output_dir / "logs",
    }
    for directory in dirs.values():
        directory.mkdir(parents=True, exist_ok=True)
    return dirs


def _resize_prealigned(image_rgb: np.ndarray, image_size: int) -> np.ndarray:
    height, width = image_rgb.shape[:2]
    if height != width:
        raise ValueError(f"Pre-aligned input must be square, got {width}x{height}")
    if (height, width) == (image_size, image_size):
        return image_rgb.copy()
    interpolation = cv2.INTER_AREA if height > image_size else cv2.INTER_LINEAR
    return cv2.resize(image_rgb, (image_size, image_size), interpolation=interpolation)


def _process_one(
    image_id: str,
    input_path: Path,
    dirs: dict[str, Path],
    parsing_model: Any,
    parsing_device: Any,
    args: argparse.Namespace,
) -> dict[str, Any]:
    paths = {
        "aligned_rgb": dirs["aligned_rgb"] / f"{image_id}.png",
        "final_mask": dirs["final_mask"] / f"{image_id}.png",
        "selected_semantic_mask": dirs["selected_semantic_mask"] / f"{image_id}.png",
        "semantic_regularized_mask": dirs["semantic_regularized_mask"] / f"{image_id}.png",
        "candidate_envelope": dirs["candidate_envelope"] / f"{image_id}.png",
        "parsing_label": dirs["parsing_label"] / f"{image_id}.png",
        "output": dirs["images"] / f"{image_id}.png",
    }
    row = {column: "" for column in LOG_COLUMNS}
    row.update(
        {
            "ID": image_id,
            "input_path": str(input_path),
            "aligned_rgb_path": str(paths["aligned_rgb"]),
            "final_mask_path": str(paths["final_mask"]),
            "output_path": str(paths["output"]),
            "alignment_mode": "prealigned_resize_only",
            "resize_interpolation": "INTER_AREA",
            "image_size": int(args.image_size),
            "status": "failed",
        }
    )
    try:
        input_rgb = read_rgb(input_path)
        height, width = input_rgb.shape[:2]
        row["input_height"] = int(height)
        row["input_width"] = int(width)
        aligned_rgb = _resize_prealigned(input_rgb, int(args.image_size))
        if height < int(args.image_size):
            row["resize_interpolation"] = "INTER_LINEAR"
        elif height == int(args.image_size):
            row["resize_interpolation"] = "none"

        label_map = parsing.run_face_parsing(aligned_rgb, parsing_model, parsing_device)
        selected_mask = parsing.build_selected_semantic_mask(label_map)
        semantic_mask = hybrid.build_semantic_regularized_mask(selected_mask)
        candidate_envelope, _ = parsing.build_regularized_face_envelope_mask(
            semantic_mask,
            label_map,
            float(args.forehead_expand_ratio),
            float(args.side_expand_ratio),
            float(args.chin_expand_ratio),
        )
        _, hair_candidate_ratio = hybrid._ratio_inside_mask(
            label_map, candidate_envelope, (parsing.CLASS_HAIR,)
        )
        jaggedness = hybrid.compute_forehead_top_jaggedness(
            semantic_mask, float(args.forehead_band_ratio)
        )
        final_mask, repaired, repair_reason = hybrid.build_final_mask(
            semantic_mask,
            candidate_envelope,
            "hybrid",
            float(args.forehead_band_ratio),
            hair_candidate_ratio,
            float(args.hair_repair_threshold),
            jaggedness,
            float(args.jaggedness_threshold),
            bool(args.enable_jaggedness_trigger),
        )
        if int((final_mask > 0).sum()) < max(64, int(round(final_mask.size * 0.01))):
            raise ValueError("final_mask_too_small")

        alpha = feather_mask(final_mask, int(args.feather_kernel))
        output_rgb = apply_background(aligned_rgb, alpha, "imagenet_mean")
        save_rgb(paths["aligned_rgb"], aligned_rgb)
        _save_mask(paths["final_mask"], final_mask)
        _save_mask(paths["selected_semantic_mask"], selected_mask)
        _save_mask(paths["semantic_regularized_mask"], semantic_mask)
        _save_mask(paths["candidate_envelope"], candidate_envelope)
        _save_label_map(paths["parsing_label"], label_map)
        save_rgb(paths["output"], output_rgb)

        row.update(
            {
                "status": "success",
                "mask_area_ratio": float((final_mask > 0).mean()),
                "forehead_repair_applied": bool(repaired),
                "forehead_repair_reason": repair_reason,
                "hair_inside_final_mask_ratio": hybrid._ratio_inside_mask(
                    label_map, final_mask, (parsing.CLASS_HAIR,)
                )[1],
                "neck_inside_final_mask_ratio": hybrid._ratio_inside_mask(
                    label_map, final_mask, (parsing.CLASS_NECK,)
                )[1],
                "cloth_inside_final_mask_ratio": hybrid._ratio_inside_mask(
                    label_map, final_mask, (parsing.CLASS_CLOTH,)
                )[1],
                "background_inside_final_mask_ratio": hybrid._ratio_inside_mask(
                    label_map, final_mask, (parsing.CLASS_BACKGROUND,)
                )[1],
            }
        )
    except Exception as exc:
        row["error_message"] = f"{type(exc).__name__}: {exc}"
    return row


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if int(args.image_size) <= 0:
        raise ValueError("--image-size must be positive")
    if int(args.feather_kernel) <= 0:
        raise ValueError("--feather-kernel must be positive")
    records = _collect_inputs(args)
    output_dir = _resolve(args.output_dir)
    dirs = _prepare_dirs(output_dir, bool(args.overwrite), bool(args.flat_output))
    checkpoint = _resolve(args.parsing_checkpoint)
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Parsing checkpoint does not exist: {checkpoint}")
    parsing_device = parsing.resolve_parsing_device(str(args.parsing_device))
    parsing_model = parsing.load_face_parsing_model(
        str(args.parsing_model), checkpoint, parsing_device
    )

    rows = []
    for index, (image_id, input_path) in enumerate(records, start=1):
        print(f"[{index:04d}/{len(records):04d}] {image_id}")
        rows.append(_process_one(image_id, input_path, dirs, parsing_model, parsing_device, args))
    log_df = pd.DataFrame(rows, columns=LOG_COLUMNS)
    log_path = dirs["logs"] / "prealigned_preprocess_log.csv"
    log_df.to_csv(log_path, index=False, encoding="utf-8-sig")
    success_count = int((log_df["status"] == "success").sum())
    print(f"Completed: success={success_count}, failed={len(log_df) - success_count}")
    print(f"Images: {dirs['images']}")
    print(f"Log: {log_path}")
    return 0 if success_count == len(log_df) else 1


if __name__ == "__main__":
    raise SystemExit(main())
