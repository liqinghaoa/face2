"""Run global face preprocessing while saving reusable intermediate artifacts."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from preprocessing import (  # noqa: E402
    build_global_face_parsing_hybrid_foreheadrepair_blackbg_224_png_strict as preprocess,
)


DEFAULT_CONFIG = (
    PROJECT_ROOT
    / "config"
    / "preprocess"
    / "global_face_parsing_hybrid_foreheadrepair_blackbg_224_png_strict_intermediates.yaml"
)
REQUIRED_DIRS = (
    "images",
    "aligned_rgb",
    "final_mask",
    "logs",
    "qc_preview",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--start-index", type=int, default=None)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--append-existing", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def _resolve_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def _load_config(config_path: Path) -> dict:
    with config_path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def _load_output_dir(config_path: Path, override: Path | None) -> Path:
    if override is not None:
        return _resolve_path(override)
    config = _load_config(config_path)
    output_dir = config.get("output_dir")
    if not output_dir:
        raise ValueError(f"output_dir is missing in {config_path}")
    return _resolve_path(output_dir)


def _check_output(output_dir: Path, image_size: int) -> dict[str, int]:
    missing_dirs = [
        name for name in REQUIRED_DIRS if not (output_dir / name).is_dir()
    ]
    if missing_dirs:
        raise FileNotFoundError(
            f"Missing expected output directories under {output_dir}: {missing_dirs}"
        )
    log_path = output_dir / "logs" / "preprocess_log.csv"
    if not log_path.is_file():
        raise FileNotFoundError(f"Missing preprocess log: {log_path}")

    log_df = pd.read_csv(log_path, dtype={"ID": "string"}, encoding="utf-8-sig")
    success = log_df[log_df["status"].astype(str) == "success"].copy()
    checked = 0
    for row in success.head(5).itertuples(index=False):
        image_id = str(row.ID)
        for name in ("images", "aligned_rgb", "final_mask"):
            path = output_dir / name / f"{image_id}.png"
            if not path.is_file():
                raise FileNotFoundError(f"Missing {name} PNG for ID={image_id}: {path}")
        mask_path = output_dir / "final_mask" / f"{image_id}.png"
        mask = cv2.imdecode(np.fromfile(str(mask_path), dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise ValueError(f"Cannot read final_mask PNG: {mask_path}")
        if mask.shape != (image_size, image_size):
            raise ValueError(
                f"final_mask must be {image_size}x{image_size}, got {mask.shape}: {mask_path}"
            )
        values = set(int(v) for v in pd.unique(mask.reshape(-1)))
        if not values.issubset({0, 255}):
            raise ValueError(f"final_mask is not binary 0/255: {mask_path}, values={values}")
        checked += 1
    return {
        "total": int(len(log_df)),
        "success": int(len(success)),
        "failed": int(len(log_df) - len(success)),
        "intermediate_checked": checked,
    }


def main() -> int:
    args = parse_args()
    config_path = _resolve_path(args.config)
    config = _load_config(config_path)
    image_size = int(config.get("image_size", 224))
    if image_size <= 0:
        raise ValueError(f"image_size must be positive, got: {image_size}")
    output_dir = _load_output_dir(config_path, args.output_dir)
    forwarded = ["--config", str(config_path)]
    if args.output_dir is not None:
        forwarded.extend(["--output-dir", str(output_dir)])
    if args.start_index is not None:
        forwarded.extend(["--start-index", str(args.start_index)])
    if args.max_samples is not None:
        forwarded.extend(["--max-samples", str(args.max_samples)])
    if args.append_existing:
        forwarded.append("--append-existing")
    if args.overwrite:
        forwarded.append("--overwrite")

    exit_code = preprocess.main(forwarded)
    if exit_code != 0:
        return int(exit_code)
    stats = _check_output(output_dir, image_size)
    print(
        "Intermediate preprocessing completed: "
        f"success={stats['success']}, failed={stats['failed']}, "
        f"checked={stats['intermediate_checked']}, output_dir={output_dir}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
