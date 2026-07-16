"""Build and audit the fixed 425-case S2-P2 multi-input manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.relative_optical_channels import CHANNEL_NAMES, build_relative_optical_channels  # noqa: E402
from utils.relative_roi_masks import build_traceable_roi_masks  # noqa: E402


DISCLAIMER = (
    "本实验在根据既有ResNet18/34/50 OOF共同错误事后构建的S2队列上训练。"
    "结果仅用于S2内部方法比较和后续物理一致性实验，不属于完整临床队列上的无偏泛化性能。"
)
VERSION = "s2_p2_inputs_v1_historical_p2_1_geometry"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(
        path,
        dtype={"ID": "string", "patient_group_id": "string"},
        encoding="utf-8-sig",
    )


def copy_asset(source: Path, destination: Path) -> None:
    if not source.is_file():
        raise FileNotFoundError(source)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if not destination.is_file() or sha256(source) != sha256(destination):
        shutil.copy2(source, destination)


def missing_ids(s2: pd.DataFrame, eye_root: Path, cheek_root: Path) -> list[str]:
    return [
        identifier
        for identifier in s2["ID"].astype(str)
        if not (eye_root / f"{identifier}.png").is_file()
        or not (cheek_root / f"{identifier}.png").is_file()
    ]


def generated_ready(generated: Path, ids: list[str]) -> bool:
    metadata = generated / "logs/roi_metadata.csv"
    if not metadata.is_file():
        return False
    required = (
        generated / "roi_masked/eye_roi",
        generated / "roi_raw/cheek_roi",
    )
    return all((root / f"{identifier}.png").is_file() for root in required for identifier in ids)


def generate_missing_rois(s2: pd.DataFrame, ids: list[str], output: Path) -> None:
    if not ids or generated_ready(output / "generated_missing_roi", ids):
        return
    protocol = output / "protocol"
    protocol.mkdir(parents=True, exist_ok=True)
    missing_frame = s2[s2["ID"].astype(str).isin(ids)].copy()
    missing_csv = protocol / "missing_roi_source.csv"
    missing_frame.to_csv(missing_csv, index=False, encoding="utf-8-sig")
    generated = output / "generated_missing_roi"
    command = [
        sys.executable,
        str(ROOT / "preprocessing/preprocess_global_aligned_face_parsing_roi_dataset_224_canvas.py"),
        "--config",
        str(ROOT / "config/preprocess/global_aligned_face_parsing_roi_final5_224_canvas.yaml"),
        "--project-root",
        str(ROOT),
        "--split-csv",
        str(missing_csv),
        "--image-dir",
        str(ROOT / "data/raw/images"),
        "--output-dir",
        str(generated),
        "--global-intermediate-dir",
        str(
            ROOT
            / "data/processed/global_face/"
            "global_face_parsing_hybrid_foreheadrepair_blackbg_224_png_strict_intermediates"
        ),
        "--roi-types",
        "eye_roi,cheek_roi",
        "--core-roi-types",
        "eye_roi,cheek_roi",
        "--num-qc-preview",
        "0",
        "--overwrite",
    ]
    subprocess.run(command, cwd=ROOT, check=True)
    if not generated_ready(generated, ids):
        raise RuntimeError("historical ROI preprocessing did not produce every missing S2 asset")


def assemble_assets(s2: pd.DataFrame, ids: list[str], output: Path) -> tuple[Path, dict[str, Any]]:
    historical_roi = ROOT / "data/processed/roi_dataset/manual_shift_data"
    historical_metadata = (
        ROOT
        / "data/processed/roi_dataset/"
        "global_aligned_face_parsing_roi_final5_224_canvas_500/logs/roi_metadata.csv"
    )
    generated = output / "generated_missing_roi"
    assets = output / "assets"
    for identifier in s2["ID"].astype(str):
        if identifier in ids:
            eye = generated / f"roi_masked/eye_roi/{identifier}.png"
            cheek = generated / f"roi_raw/cheek_roi/{identifier}.png"
        else:
            eye = historical_roi / f"eye_roi/{identifier}.png"
            cheek = historical_roi / f"cheek_roi/{identifier}.png"
        copy_asset(eye, assets / f"eye_roi/{identifier}.png")
        copy_asset(cheek, assets / f"cheek_roi/{identifier}.png")

    metadata_frames = [read_csv(historical_metadata)]
    if ids:
        metadata_frames.append(read_csv(generated / "logs/roi_metadata.csv"))
    metadata = pd.concat(metadata_frames, ignore_index=True)
    metadata = metadata[metadata["ID"].astype(str).isin(set(s2["ID"].astype(str)))].copy()
    metadata = metadata.drop_duplicates(["ID", "roi_type"], keep="last")
    needed = set(s2["ID"].astype(str))
    for roi_type in ("eye_roi", "cheek_roi"):
        covered = set(metadata.loc[metadata["roi_type"].eq(roi_type), "ID"].astype(str))
        if covered != needed:
            raise RuntimeError(f"combined ROI metadata coverage failed for {roi_type}")
    combined_metadata = output / "protocol/combined_roi_metadata.csv"
    combined_metadata.parent.mkdir(parents=True, exist_ok=True)
    metadata.to_csv(combined_metadata, index=False, encoding="utf-8-sig")
    mask_payload = build_traceable_roi_masks(
        sorted(needed),
        combined_metadata,
        ROOT
        / "data/processed/global_face/"
        "global_face_parsing_hybrid_foreheadrepair_blackbg_224_png_strict_intermediates/final_mask",
        assets,
    )
    return assets, mask_payload


def image_and_channels(path: Path, mask_path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    mask_u8 = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if bgr is None or mask_u8 is None:
        raise ValueError(f"unreadable image or mask: {path}")
    if bgr.shape != (224, 224, 3) or mask_u8.shape != (224, 224):
        raise ValueError(f"unexpected S2-P2 input shape: {path} {bgr.shape} {mask_u8.shape}")
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    mask = mask_u8 > 0
    raw = np.transpose(rgb.astype(np.float32) / 255.0, (2, 0, 1))
    channels = np.asarray(build_relative_optical_channels(raw, mask))
    return rgb, mask, channels


def audit_and_manifest(s2: pd.DataFrame, generated_ids: list[str], assets: Path, output: Path) -> tuple[pd.DataFrame, dict[str, Any]]:
    global_root = (
        ROOT
        / "data/processed/global_face/"
        "global_face_parsing_hybrid_foreheadrepair_blackbg_224_png_strict/images"
    )
    p0_oof = read_csv(
        ROOT
        / "experiments/S2_425_Global224_ImageNetResNet18_NYHA3Class_WeightedCE_5Fold/"
        "summary/oof_predictions.csv"
    )
    excluded = read_csv(ROOT / "data/processed/s2_425/s2_excluded_manifest.csv")
    rows: list[dict[str, Any]] = []
    channel_min = np.full(7, np.inf)
    channel_max = np.full(7, -np.inf)
    outside_max = 0.0
    flip_max_error = 0.0
    roi_hashes: dict[str, dict[str, str]] = {"eye": {}, "cheek": {}}
    for _, source in s2.iterrows():
        identifier = str(source["ID"])
        global_path = global_root / f"{identifier}.png"
        eye_path = assets / f"eye_roi/{identifier}.png"
        cheek_path = assets / f"cheek_roi/{identifier}.png"
        eye_mask_path = assets / f"eye_mask/{identifier}.png"
        cheek_mask_path = assets / f"cheek_mask/{identifier}.png"
        global_bgr = cv2.imread(str(global_path), cv2.IMREAD_COLOR)
        if global_bgr is None or global_bgr.shape != (224, 224, 3):
            raise ValueError(f"invalid global input: {global_path}")
        areas: dict[str, int] = {}
        for roi, image_path, mask_path in (
            ("eye", eye_path, eye_mask_path),
            ("cheek", cheek_path, cheek_mask_path),
        ):
            rgb, mask, channels = image_and_channels(image_path, mask_path)
            if channels.dtype != np.float32 or not np.isfinite(channels).all():
                raise ValueError(f"invalid optical channels: {identifier} {roi}")
            area = int(mask.sum())
            if area <= 0:
                raise ValueError(f"empty {roi} mask: {identifier}")
            areas[roi] = area
            valid = channels[:, mask]
            channel_min = np.minimum(channel_min, valid.min(axis=1))
            channel_max = np.maximum(channel_max, valid.max(axis=1))
            if np.any(~mask):
                outside_max = max(outside_max, float(np.abs(channels[:, ~mask]).max()))
            flipped = np.asarray(
                build_relative_optical_channels(
                    np.transpose(np.ascontiguousarray(rgb[:, ::-1]).astype(np.float32) / 255.0, (2, 0, 1)),
                    np.ascontiguousarray(mask[:, ::-1]),
                )
            )
            flip_max_error = max(
                flip_max_error,
                float(np.abs(flipped - np.ascontiguousarray(channels[:, :, ::-1])).max()),
            )
            roi_hashes[roi][identifier] = sha256(image_path)
        rows.append(
            {
                "ID": identifier,
                "patient_group_id": str(source["patient_group_id"]),
                "NYHA": int(source["NYHA"]),
                "label_3class": int(source["label_3class"]),
                "label_3class_name": str(source["label_3class_name"]),
                "SEX": int(source["SEX"]),
                "fold": int(source["fold"]),
                "global_image_path": str(global_path.resolve()),
                "eye_input_path": str(eye_path.resolve()),
                "cheek_input_path": str(cheek_path.resolve()),
                "eye_mask_path": str(eye_mask_path.resolve()),
                "cheek_mask_path": str(cheek_mask_path.resolve()),
                "relative_optical_input_path": "online_from_eye_cheek_rgb_and_binary_masks",
                "global_exists_readable": True,
                "eye_exists_readable": True,
                "cheek_exists_readable": True,
                "eye_mask_exists_readable": True,
                "cheek_mask_exists_readable": True,
                "eye_mask_valid_area": areas["eye"],
                "cheek_mask_valid_area": areas["cheek"],
                "input_generation_version": VERSION,
                "asset_origin": "historical_p2_1" if identifier not in generated_ids else "historical_pipeline_regenerated_for_s2",
                "source_s2_manifest": str((ROOT / "data/processed/s2_425/s2_manifest.csv").resolve()),
            }
        )
    manifest = pd.DataFrame(rows)
    duplicate_hashes = {
        roi: int(pd.Series(list(values.values())).duplicated(keep=False).sum())
        for roi, values in roi_hashes.items()
    }
    expected_counts = {0: 112, 1: 202, 2: 111}
    p0 = p0_oof.sort_values("ID").reset_index(drop=True)
    aligned = manifest.sort_values("ID").reset_index(drop=True)
    checks = {
        "rows_425_unique": len(manifest) == 425 and manifest["ID"].nunique() == 425,
        "class_counts_112_202_111": manifest["label_3class"].value_counts().sort_index().to_dict() == expected_counts,
        "excluded_97_disjoint": set(manifest["ID"]).isdisjoint(set(excluded["ID"].astype(str))),
        "fold_matches_s2_p0": aligned[["ID", "patient_group_id", "label_3class", "fold"]].astype(str).equals(
            p0[["ID", "patient_group_id", "label_3class", "fold"]].astype(str)
        ),
        "patient_group_not_cross_fold": int(manifest.groupby("patient_group_id")["fold"].nunique().max()) == 1,
        "global_paths_match_s2_p0": set(aligned["global_image_path"]) == set(
            str(global_root / f"{identifier}.png") for identifier in p0["ID"].astype(str)
        ),
        "all_assets_readable_224": True,
        "mask_areas_positive": int(manifest[["eye_mask_valid_area", "cheek_mask_valid_area"]].min().min()) > 0,
        "channels_finite": bool(np.isfinite(channel_min).all() and np.isfinite(channel_max).all()),
        "mask_outside_exact_zero": outside_max <= 1.0e-7,
        "synchronized_flip_channel_equivariance": flip_max_error <= 1.0e-6,
        "no_exact_duplicate_eye_assets": duplicate_hashes["eye"] == 0,
        "no_exact_duplicate_cheek_assets": duplicate_hashes["cheek"] == 0,
        "no_label_fold_prediction_exif_in_optical_formula": True,
    }
    payload = {
        "status": "PASS" if all(checks.values()) else "FAIL",
        "all_checks_pass": all(checks.values()),
        "disclaimer": DISCLAIMER,
        "sample_count": len(manifest),
        "patient_groups": int(manifest["patient_group_id"].nunique()),
        "class_counts": manifest["label_3class"].value_counts().sort_index().to_dict(),
        "generated_missing_ids": generated_ids,
        "generated_missing_count": len(generated_ids),
        "channel_names": list(CHANNEL_NAMES),
        "channel_valid_min": channel_min.tolist(),
        "channel_valid_max": channel_max.tolist(),
        "mask_outside_max_abs": outside_max,
        "flip_equivariance_max_abs_error": flip_max_error,
        "minimum_mask_areas": {
            "eye": int(manifest["eye_mask_valid_area"].min()),
            "cheek": int(manifest["cheek_mask_valid_area"].min()),
        },
        "duplicate_hash_rows": duplicate_hashes,
        "checks": checks,
        "source_s2_manifest_sha256": sha256(ROOT / "data/processed/s2_425/s2_manifest.csv"),
        "input_generation_version": VERSION,
    }
    if not payload["all_checks_pass"]:
        failed = [name for name, passed in checks.items() if not passed]
        raise RuntimeError(f"S2-P2 input audit failed: {failed}")
    return manifest, payload


def qc(manifest: pd.DataFrame, output: Path, seed: int = 2026) -> list[str]:
    qc_dir = output / "optical_channel_qc"
    qc_dir.mkdir(parents=True, exist_ok=True)
    chosen = manifest.sample(n=3, random_state=seed).reset_index(drop=True)
    rows = []
    names = []
    for sample_index, row in chosen.iterrows():
        global_rgb = cv2.cvtColor(cv2.imread(str(row["global_image_path"])), cv2.COLOR_BGR2RGB)
        panels = []
        for roi in ("eye", "cheek"):
            rgb, mask, channels = image_and_channels(Path(row[f"{roi}_input_path"]), Path(row[f"{roi}_mask_path"]))
            panels.append((roi, rgb, mask, channels))
            for channel_index, name in enumerate(CHANNEL_NAMES):
                values = channels[channel_index][mask]
                rows.append(
                    {
                        "qc_sample": sample_index + 1,
                        "roi": roi,
                        "channel": name,
                        "min": float(values.min()),
                        "max": float(values.max()),
                        "mean": float(values.mean()),
                        "std": float(values.std()),
                        "mask_area": int(mask.sum()),
                        "mask_outside_max_abs": float(np.abs(channels[:, ~mask]).max()) if np.any(~mask) else 0.0,
                    }
                )
        fig, axes = plt.subplots(5, 5, figsize=(15, 14))
        axes[0, 0].imshow(global_rgb)
        axes[0, 0].set_title("Global strict black-bg")
        for col in range(1, 5):
            axes[0, col].axis("off")
        for roi_index, (roi, rgb, mask, channels) in enumerate(panels):
            base = 1 + roi_index * 2
            axes[base, 0].imshow(rgb)
            axes[base, 0].set_title(f"{roi.title()} RGB")
            axes[base, 1].imshow(mask, cmap="gray", vmin=0, vmax=1)
            axes[base, 1].set_title(f"{roi.title()} mask")
            for col, channel_index in enumerate(range(3), start=2):
                axes[base, col].imshow(channels[channel_index], cmap="viridis")
                axes[base, col].set_title(CHANNEL_NAMES[channel_index])
            for col, channel_index in enumerate(range(3, 7)):
                axes[base + 1, col].imshow(channels[channel_index], cmap="coolwarm")
                axes[base + 1, col].set_title(CHANNEL_NAMES[channel_index])
            axes[base + 1, 4].axis("off")
        for axis in axes.flat:
            axis.axis("off")
        fig.suptitle(f"De-identified S2-P2 QC sample {sample_index + 1} (implementation only)")
        fig.tight_layout()
        name = f"qc_sample_{sample_index + 1:02d}.png"
        fig.savefig(qc_dir / name, dpi=150)
        plt.close(fig)
        names.append(name)
    pd.DataFrame(rows).to_csv(qc_dir / "qc_channel_ranges.csv", index=False, encoding="utf-8-sig")
    return names


def write_audit(output: Path, manifest: pd.DataFrame, payload: dict[str, Any], mask_payload: dict[str, Any], qc_names: list[str]) -> None:
    manifest_path = output / "s2_p2_manifest.csv"
    json_path = output / "s2_p2_manifest.json"
    audit_path = output / "s2_p2_input_audit.md"
    manifest.to_csv(manifest_path, index=False, encoding="utf-8-sig")
    payload = {**payload, "mask_generation": mask_payload, "qc_files": qc_names}
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    checks = "\n".join(
        f"| {name} | {'PASS' if passed else 'FAIL'} |" for name, passed in payload["checks"].items()
    )
    generated = ", ".join(payload["generated_missing_ids"]) or "无"
    ranges = "\n".join(
        f"| {name} | {low:.6f} | {high:.6f} |"
        for name, low, high in zip(
            payload["channel_names"], payload["channel_valid_min"], payload["channel_valid_max"]
        )
    )
    audit_path.write_text(
        f"""# S2-P2 input audit

> **{DISCLAIMER}**

**Overall status: {payload['status']}**

- Cases: {len(manifest)}; patient groups: {payload['patient_groups']}
- Classes: normal=112, mild=202, severe=111
- Historical P2-1 assets reused: {len(manifest) - payload['generated_missing_count']}
- Missing assets regenerated by the historical ROI pipeline: {payload['generated_missing_count']}
- Regenerated IDs: {generated}
- Minimum mask areas: Eye={payload['minimum_mask_areas']['eye']}, Cheek={payload['minimum_mask_areas']['cheek']} pixels
- Mask-outside maximum absolute optical value: {payload['mask_outside_max_abs']:.3e}
- Synchronized-flip channel equivalence maximum error: {payload['flip_equivariance_max_abs_error']:.3e}
- QC files: {', '.join(qc_names)}

## Checks

| Check | Status |
|---|---|
{checks}

## Raw optical channel valid-pixel ranges

| Channel | Min | Max |
|---|---:|---:|
{ranges}

## Leakage boundary

Optical inputs are computed independently per case from RGB and binary ROI masks. NYHA, class label, fold, historical OOF predictions, confidence, EXIF and device fields are not read by the optical formula. Fold-level normalization statistics are fitted later using only each fold's training Eye/Cheek valid pixels.
""",
        encoding="utf-8",
    )


def run() -> None:
    output = ROOT / "data/processed/s2_425_p2"
    output.mkdir(parents=True, exist_ok=True)
    s2 = read_csv(ROOT / "data/processed/s2_425/s2_manifest.csv")
    historical = ROOT / "data/processed/roi_dataset/manual_shift_data"
    missing = missing_ids(s2, historical / "eye_roi", historical / "cheek_roi")
    generate_missing_rois(s2, missing, output)
    assets, mask_payload = assemble_assets(s2, missing, output)
    manifest, payload = audit_and_manifest(s2, missing, assets, output)
    qc_names = qc(manifest, output, seed=2026)
    write_audit(output, manifest, payload, mask_payload, qc_names)
    print(json.dumps({"status": payload["status"], "samples": len(manifest), "generated": missing}, ensure_ascii=False))


def parse_args() -> argparse.Namespace:
    return argparse.ArgumentParser().parse_args()


if __name__ == "__main__":
    parse_args()
    run()
