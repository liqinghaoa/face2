"""Build preprocessing ablation images from saved aligned RGB and final masks."""

from __future__ import annotations

import argparse
import random
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.preprocess_ablation_utils import (  # noqa: E402
    apply_background,
    clahe_l,
    collect_required_ids_from_splits,
    compute_region_color_stats,
    feather_mask,
    gray3ch,
    lab_l_norm,
    make_qc_grid,
    masked_grayworld_wb,
    read_mask,
    read_rgb,
    retinex_msr,
    save_rgb,
)


DEFAULT_INTERMEDIATE_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "global_face"
    / "global_face_parsing_hybrid_foreheadrepair_blackbg_224_png_strict_intermediates"
)
DEFAULT_OUTPUT_ROOT = (
    PROJECT_ROOT / "data" / "processed" / "global_face" / "preprocess_ablation"
)
DEFAULT_SPLIT_DIR = PROJECT_ROOT / "data" / "processed" / "splits_500"


TransformFn = Callable[[np.ndarray, np.ndarray], np.ndarray]


@dataclass(frozen=True)
class VariantSpec:
    index: str
    name: str
    bg_mode: str
    photometric_mode: str
    transform: TransformFn
    description: str


def _identity(image: np.ndarray, mask: np.ndarray) -> np.ndarray:
    del mask
    return image


VARIANT_SPECS: tuple[VariantSpec, ...] = (
    VariantSpec(
        "E0",
        "hybrid_black_baseline",
        "black",
        "none",
        _identity,
        "aligned RGB plus final mask over black background",
    ),
    VariantSpec(
        "E1",
        "hybrid_imagenet_meanbg",
        "imagenet_mean",
        "none",
        _identity,
        "aligned RGB plus final mask over ImageNet RGB mean background",
    ),
    VariantSpec(
        "E2",
        "hybrid_black_labl_norm",
        "black",
        "lab_l_norm",
        lab_l_norm,
        "mild robust Lab-L normalization over black background",
    ),
    VariantSpec(
        "E3",
        "hybrid_imagenet_meanbg_labl_norm",
        "imagenet_mean",
        "lab_l_norm",
        lab_l_norm,
        "mild robust Lab-L normalization over ImageNet mean background",
    ),
    VariantSpec(
        "E4",
        "hybrid_black_clahe_l",
        "black",
        "clahe_l",
        clahe_l,
        "CLAHE on Lab-L over black background",
    ),
    VariantSpec(
        "E5",
        "hybrid_black_gray3ch",
        "black",
        "gray3ch",
        lambda image, mask: gray3ch(image),
        "RGB to grayscale repeated to 3 channels over black background",
    ),
    VariantSpec(
        "E6",
        "hybrid_black_masked_grayworld_wb",
        "black",
        "masked_grayworld_wb",
        masked_grayworld_wb,
        "clipped Gray-World white balance estimated inside final mask",
    ),
    VariantSpec(
        "E7",
        "hybrid_black_retinex_msr",
        "black",
        "retinex_msr",
        lambda image, mask: retinex_msr(image, scales=(15, 80, 250)),
        "traditional multi-scale Retinex over black background",
    ),
)
VARIANT_BY_NAME = {spec.name: spec for spec in VARIANT_SPECS}


LOG_COLUMNS = (
    "ID",
    "variant_name",
    "aligned_rgb_path",
    "final_mask_path",
    "output_path",
    "status",
    "error_message",
    "mask_area_ratio",
    "bg_mode",
    "photometric_mode",
    "mean_r_before",
    "mean_g_before",
    "mean_b_before",
    "mean_l_before",
    "mean_r_after",
    "mean_g_after",
    "mean_b_after",
    "mean_l_after",
)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--intermediate-dir", type=Path, default=DEFAULT_INTERMEDIATE_DIR)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--variants", type=str, default="all")
    parser.add_argument("--split-dir", type=Path, default=DEFAULT_SPLIT_DIR)
    parser.add_argument("--label-csv", type=Path, default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--make-qc-preview",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--feather-kernel", type=int, default=11)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args(argv)


def _resolve(path: Path) -> Path:
    path = path.expanduser()
    return path.resolve() if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def _select_variants(value: str) -> list[VariantSpec]:
    if value.strip().lower() == "all":
        return list(VARIANT_SPECS)
    names = [item.strip() for item in value.split(",") if item.strip()]
    unknown = [name for name in names if name not in VARIANT_BY_NAME]
    if unknown:
        raise ValueError(f"Unknown variants: {unknown}")
    return [VARIANT_BY_NAME[name] for name in names]


def _prepare_variant_dirs(variant_dir: Path, overwrite: bool) -> dict[str, Path]:
    if variant_dir.exists() and overwrite:
        shutil.rmtree(variant_dir)
    dirs = {
        "root": variant_dir,
        "images": variant_dir / "images",
        "logs": variant_dir / "logs",
        "qc_preview": variant_dir / "qc_preview",
    }
    for directory in dirs.values():
        directory.mkdir(parents=True, exist_ok=True)
    return dirs


def _collect_label_lookup(split_dir: Path, label_csv: Path | None) -> dict[str, str]:
    frames: list[pd.DataFrame] = []
    for csv_path in sorted(split_dir.glob("fold_*_*.csv")):
        frame = pd.read_csv(csv_path, dtype={"ID": "string"}, encoding="utf-8-sig")
        if "ID" in frame.columns:
            frames.append(frame)
    if label_csv is not None and label_csv.is_file():
        frames.append(pd.read_csv(label_csv, dtype={"ID": "string"}, encoding="utf-8-sig"))
    if not frames:
        return {}
    merged = pd.concat(frames, ignore_index=True)
    if "label_3class_name" in merged.columns:
        label_series = merged["label_3class_name"].astype(str)
    elif "label_3class" in merged.columns:
        label_series = pd.to_numeric(merged["label_3class"], errors="coerce").map(
            {0: "normal", 1: "mild", 2: "severe"}
        )
    elif "NYHA" in merged.columns:
        label_series = pd.to_numeric(merged["NYHA"], errors="coerce").map(
            {0: "normal", 1: "mild", 2: "mild", 3: "severe", 4: "severe"}
        )
    else:
        return {}
    merged = merged.assign(_label=label_series)
    merged = merged.dropna(subset=["ID", "_label"]).drop_duplicates("ID")
    return dict(zip(merged["ID"].astype(str), merged["_label"].astype(str)))


def _empty_log_row(
    image_id: str,
    spec: VariantSpec,
    aligned_path: Path,
    mask_path: Path,
    output_path: Path,
) -> dict[str, Any]:
    row = {column: np.nan for column in LOG_COLUMNS}
    row.update(
        {
            "ID": image_id,
            "variant_name": spec.name,
            "aligned_rgb_path": str(aligned_path),
            "final_mask_path": str(mask_path),
            "output_path": str(output_path),
            "status": "",
            "error_message": "",
            "bg_mode": spec.bg_mode,
            "photometric_mode": spec.photometric_mode,
        }
    )
    return row


def _add_stats(row: dict[str, Any], prefix: str, stats: dict[str, float]) -> None:
    row[f"mean_r_{prefix}"] = stats["mean_r"]
    row[f"mean_g_{prefix}"] = stats["mean_g"]
    row[f"mean_b_{prefix}"] = stats["mean_b"]
    row[f"mean_l_{prefix}"] = stats["mean_l"]


def _process_one(
    image_id: str,
    spec: VariantSpec,
    intermediate_dir: Path,
    images_dir: Path,
    feather_kernel: int,
    overwrite: bool,
) -> dict[str, Any]:
    aligned_path = intermediate_dir / "aligned_rgb" / f"{image_id}.png"
    mask_path = intermediate_dir / "final_mask" / f"{image_id}.png"
    output_path = images_dir / f"{image_id}.png"
    row = _empty_log_row(image_id, spec, aligned_path, mask_path, output_path)
    try:
        if not aligned_path.is_file() or not mask_path.is_file():
            missing = [
                str(path)
                for path in (aligned_path, mask_path)
                if not path.is_file()
            ]
            raise FileNotFoundError(f"Missing intermediate files: {missing}")
        aligned_rgb = read_rgb(aligned_path)
        final_mask = read_mask(mask_path)
        if aligned_rgb.shape != (224, 224, 3):
            raise ValueError(f"aligned_rgb must be 224x224 RGB, got {aligned_rgb.shape}")
        if final_mask.shape != (224, 224):
            raise ValueError(f"final_mask must be 224x224, got {final_mask.shape}")
        alpha = feather_mask(final_mask, feather_kernel)
        transformed = spec.transform(aligned_rgb, final_mask)
        output_rgb = apply_background(transformed, alpha, spec.bg_mode)

        row["mask_area_ratio"] = float((final_mask > 0).mean())
        _add_stats(row, "before", compute_region_color_stats(aligned_rgb, final_mask))
        _add_stats(row, "after", compute_region_color_stats(output_rgb, final_mask))
        if output_path.is_file() and not overwrite:
            row["status"] = "success_existing"
        else:
            save_rgb(output_path, output_rgb)
            row["status"] = "success"
    except Exception as exc:
        row["status"] = "failed"
        row["error_message"] = f"{type(exc).__name__}: {exc}"
    return row


def _write_qc_preview(
    log_df: pd.DataFrame,
    label_lookup: dict[str, str],
    qc_dir: Path,
    seed: int,
) -> Path | None:
    success = log_df[log_df["status"].isin(["success", "success_existing"])].copy()
    if success.empty:
        return None
    rng = random.Random(seed)
    samples: list[dict[str, Any]] = []
    if label_lookup:
        for label in ("normal", "mild", "severe"):
            ids = [
                str(row.ID)
                for row in success.itertuples(index=False)
                if label_lookup.get(str(row.ID)) == label
            ]
            rng.shuffle(ids)
            for image_id in ids[:14]:
                output_path = success.loc[success["ID"].astype(str) == image_id, "output_path"].iloc[0]
                samples.append(
                    {"ID": image_id, "path": output_path, "title": f"{label}:{image_id}"}
                )
    if not samples:
        records = success.to_dict("records")
        rng.shuffle(records)
        samples = [
            {"ID": row["ID"], "path": row["output_path"], "title": str(row["ID"])}
            for row in records[:40]
        ]
    preview_path = qc_dir / "preview_grid.png"
    return make_qc_grid(samples, preview_path)


def build_variant(
    spec: VariantSpec,
    required_ids: list[str],
    intermediate_dir: Path,
    output_root: Path,
    label_lookup: dict[str, str],
    args: argparse.Namespace,
) -> dict[str, Any]:
    variant_dir = output_root / spec.name
    dirs = _prepare_variant_dirs(variant_dir, bool(args.overwrite))
    rows = [
        _process_one(
            image_id,
            spec,
            intermediate_dir,
            dirs["images"],
            int(args.feather_kernel),
            bool(args.overwrite),
        )
        for image_id in required_ids
    ]
    log_df = pd.DataFrame(rows, columns=LOG_COLUMNS)
    log_path = dirs["logs"] / "preprocess_variant_log.csv"
    log_df.to_csv(log_path, index=False, encoding="utf-8-sig")
    qc_path = (
        _write_qc_preview(log_df, label_lookup, dirs["qc_preview"], int(args.seed))
        if bool(args.make_qc_preview)
        else None
    )
    output_ids = {path.stem for path in dirs["images"].glob("*.png")}
    missing_required = sorted(set(required_ids).difference(output_ids))
    failed_count = int((~log_df["status"].isin(["success", "success_existing"])).sum())
    return {
        "variant_name": spec.name,
        "output_dir": str(variant_dir),
        "image_count": int(len(output_ids)),
        "required_id_count": int(len(required_ids)),
        "missing_required_count": int(len(missing_required)),
        "failed_count": failed_count,
        "qc_preview_path": "" if qc_path is None else str(qc_path),
    }


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    intermediate_dir = _resolve(args.intermediate_dir)
    output_root = _resolve(args.output_root)
    split_dir = _resolve(args.split_dir)
    label_csv = _resolve(args.label_csv) if args.label_csv is not None else None
    selected_variants = _select_variants(args.variants)

    required_ids = collect_required_ids_from_splits(split_dir)
    if args.max_samples is not None:
        required_ids = required_ids[: int(args.max_samples)]
        print(f"[smoke] Limiting required IDs to first {len(required_ids)} samples.")
    label_lookup = _collect_label_lookup(split_dir, label_csv)
    output_root.mkdir(parents=True, exist_ok=True)

    summaries = []
    for spec in selected_variants:
        print(f"[{spec.index}] Building variant: {spec.name}")
        summary = build_variant(
            spec, required_ids, intermediate_dir, output_root, label_lookup, args
        )
        summaries.append(summary)
        if summary["missing_required_count"]:
            print(
                f"[warning] {spec.name} is missing "
                f"{summary['missing_required_count']} required split images."
            )

    summary_df = pd.DataFrame(summaries)
    summary_path = output_root / "variant_summary.csv"
    summary_df.to_csv(summary_path, index=False, encoding="utf-8-sig")
    print(f"Variant summary: {summary_path}")
    missing_total = int(summary_df["missing_required_count"].sum()) if not summary_df.empty else 0
    failed_total = int(summary_df["failed_count"].sum()) if not summary_df.empty else 0
    return 1 if missing_total or failed_total else 0


if __name__ == "__main__":
    raise SystemExit(main())
