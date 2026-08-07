"""Run prealigned ImageNet-mean-background preprocessing for six relighting views."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from preprocessing import build_prealigned_face_parsing_hybrid_imagenet_meanbg as prealigned


IMAGE_NAMES = (
    "relight_bright_front.png",
    "relight_dim_front.png",
    "relight_left.png",
    "relight_neutral_front.png",
    "relight_right.png",
    "relight_top.png",
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Process six prealigned relighting images per identity."
    )
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--sidecar-dir",
        type=Path,
        default=None,
        help="Directory for masks, aligned images, parsing labels, and logs.",
    )
    parser.add_argument("--overwrite", action="store_true")
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
        "--enable-jaggedness-trigger",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument("--forehead-expand-ratio", type=float, default=0.18)
    parser.add_argument("--side-expand-ratio", type=float, default=0.05)
    parser.add_argument("--chin-expand-ratio", type=float, default=0.03)
    parser.add_argument("--feather-kernel", type=int, default=11)
    return parser


def _resolve(path: Path) -> Path:
    return prealigned._resolve(path)


def _collect_records(input_dir: Path) -> list[tuple[str, Path]]:
    id_dirs = sorted(path for path in input_dir.iterdir() if path.is_dir())
    records: list[tuple[str, Path]] = []
    missing: list[Path] = []
    for id_dir in id_dirs:
        for image_name in IMAGE_NAMES:
            input_path = id_dir / image_name
            if input_path.is_file():
                image_id = f"{id_dir.name}/{Path(image_name).stem}"
                records.append((image_id, input_path))
            else:
                missing.append(input_path)
    if missing:
        preview = "\n".join(str(path) for path in missing[:20])
        raise FileNotFoundError(f"Missing required inputs:\n{preview}")
    return records


def _prepare_dirs(output_dir: Path, sidecar_dir: Path, overwrite: bool) -> dict[str, Path]:
    if output_dir.exists() and any(output_dir.iterdir()) and not overwrite:
        raise FileExistsError(
            f"Output directory is not empty: {output_dir}. Use --overwrite to rebuild it."
        )
    dirs = {
        "root": sidecar_dir,
        "images": output_dir,
        "aligned_rgb": sidecar_dir / "aligned_rgb",
        "final_mask": sidecar_dir / "final_mask",
        "selected_semantic_mask": sidecar_dir / "selected_semantic_mask",
        "semantic_regularized_mask": sidecar_dir / "semantic_regularized_mask",
        "candidate_envelope": sidecar_dir / "candidate_envelope",
        "parsing_label": sidecar_dir / "parsing_label",
        "logs": sidecar_dir / "logs",
    }
    for directory in dirs.values():
        directory.mkdir(parents=True, exist_ok=True)
    return dirs


def main() -> int:
    args = build_parser().parse_args()
    if int(args.image_size) <= 0:
        raise ValueError("--image-size must be positive")
    if int(args.feather_kernel) <= 0:
        raise ValueError("--feather-kernel must be positive")

    input_dir = _resolve(args.input_dir)
    output_dir = _resolve(args.output_dir)
    sidecar_dir = (
        _resolve(args.sidecar_dir)
        if args.sidecar_dir is not None
        else output_dir.with_name(f"{output_dir.name}_intermediates")
    )
    records = _collect_records(input_dir)
    dirs = _prepare_dirs(output_dir, sidecar_dir, bool(args.overwrite))

    checkpoint = _resolve(args.parsing_checkpoint)
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Parsing checkpoint does not exist: {checkpoint}")
    parsing_device = prealigned.parsing.resolve_parsing_device(str(args.parsing_device))
    parsing_model = prealigned.parsing.load_face_parsing_model(
        str(args.parsing_model), checkpoint, parsing_device
    )

    rows = []
    for index, (image_id, input_path) in enumerate(records, start=1):
        print(f"[{index:04d}/{len(records):04d}] {image_id}", flush=True)
        rows.append(
            prealigned._process_one(
                image_id, input_path, dirs, parsing_model, parsing_device, args
            )
        )

    log_df = pd.DataFrame(rows, columns=prealigned.LOG_COLUMNS)
    log_path = dirs["logs"] / "prealigned_preprocess_log.csv"
    log_df.to_csv(log_path, index=False, encoding="utf-8-sig")
    success_count = int((log_df["status"] == "success").sum())
    print(f"Completed: success={success_count}, failed={len(log_df) - success_count}")
    print(f"Images: {dirs['images']}")
    print(f"Log: {log_path}")
    return 0 if success_count == len(log_df) else 1


if __name__ == "__main__":
    raise SystemExit(main())
