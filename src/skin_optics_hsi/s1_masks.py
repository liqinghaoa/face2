"""S1-2 layered skin masks, ROIs, QC, manifests, and freeze gate."""

from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml
from PIL import Image, ImageDraw
from scipy import ndimage

from skin_optics_real_preprocess import facemesh_regions

from .data_contracts import load_hyperskin_cube


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MASK_CONFIG = PROJECT_ROOT / "configs" / "skin_optics_hsi" / "s1_2_masks_v1.yaml"
RGB_WORKER = PROJECT_ROOT / "scripts" / "skin_optics_hsi" / "s1_mask_rgb_worker.py"
BUILD_MASKS_CLI = PROJECT_ROOT / "scripts" / "skin_optics_hsi" / "build_hyperskin_masks.py"
FINALIZE_MASKS_CLI = PROJECT_ROOT / "scripts" / "skin_optics_hsi" / "finalize_hyperskin_masks.py"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def _implementation_hashes() -> dict[str, str]:
    paths = {
        "s1_masks.py": Path(__file__).resolve(),
        "s1_mask_rgb_worker.py": RGB_WORKER,
        "build_hyperskin_masks.py": BUILD_MASKS_CLI,
        "finalize_hyperskin_masks.py": FINALIZE_MASKS_CLI,
    }
    return {name: sha256_file(path) for name, path in paths.items()}


def _disk(radius: int) -> np.ndarray:
    yy, xx = np.ogrid[-radius : radius + 1, -radius : radius + 1]
    return (xx * xx + yy * yy) <= radius * radius


def _polygon_mask(shape: tuple[int, int], points: np.ndarray) -> np.ndarray:
    canvas = Image.new("1", (shape[1], shape[0]), 0)
    draw = ImageDraw.Draw(canvas)
    clipped = np.asarray(points, dtype=np.float64).copy()
    clipped[:, 0] = np.clip(clipped[:, 0], 0, shape[1] - 1)
    clipped[:, 1] = np.clip(clipped[:, 1], 0, shape[0] - 1)
    draw.polygon([tuple(point) for point in clipped], fill=1)
    return np.asarray(canvas, dtype=bool)


def _ellipse_mask(shape: tuple[int, int], center_x: float, center_y: float, radius_x: float, radius_y: float) -> np.ndarray:
    yy, xx = np.ogrid[: shape[0], : shape[1]]
    return ((xx - center_x) / max(radius_x, 1.0)) ** 2 + ((yy - center_y) / max(radius_y, 1.0)) ** 2 <= 1.0


def _landmark_exclusion(shape: tuple[int, int], landmarks: np.ndarray, dilation_px: int) -> np.ndarray:
    exclusion = np.zeros(shape, dtype=bool)
    for indices in facemesh_regions.FACEMESH_EXCLUSION_POLYGONS.values():
        points = landmarks[np.asarray(indices, dtype=np.int32)]
        exclusion |= _polygon_mask(shape, points)
    if dilation_px > 0:
        exclusion = ndimage.binary_dilation(exclusion, structure=_disk(dilation_px))
    return exclusion


def build_semantic_layers(
    parsing_label: np.ndarray,
    landmarks: np.ndarray,
    config: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    """Build anatomy and conservative semantic validity without HSI spectra."""

    semantic = config["semantic"]
    shape = tuple(int(value) for value in parsing_label.shape)
    if landmarks.ndim != 2 or landmarks.shape[1] != 2 or landmarks.shape[0] < 468:
        raise ValueError(f"Invalid landmark shape: {landmarks.shape}")
    face_oval = _polygon_mask(shape, landmarks[np.asarray(facemesh_regions.FACE_OVAL_INDICES)])
    anatomical = np.isin(parsing_label, semantic["anatomical_class_ids"]) & face_oval
    parsed_exclusion = np.isin(parsing_label, semantic["exclusion_class_ids"])
    landmark_exclusion = _landmark_exclusion(
        shape, landmarks, int(semantic["landmark_exclusion_dilation_px"])
    )
    core = anatomical & ~parsed_exclusion & ~landmark_exclusion
    erosion_px = int(semantic["anatomical_boundary_erosion_px"])
    semantic_valid = ndimage.binary_erosion(core, structure=_disk(erosion_px)) if erosion_px > 0 else core
    return anatomical, semantic_valid, {
        "face_oval": face_oval,
        "parsed_exclusion": parsed_exclusion,
        "landmark_exclusion": landmark_exclusion,
    }


def build_roi_geometry(
    shape: tuple[int, int],
    landmarks: np.ndarray,
    config: dict[str, Any],
) -> dict[str, np.ndarray]:
    roi = config["roi"]
    oval = landmarks[np.asarray(facemesh_regions.FACE_OVAL_INDICES)]
    face_left, face_top = np.min(oval, axis=0)
    face_right, face_bottom = np.max(oval, axis=0)
    face_width = float(face_right - face_left)
    face_height = float(face_bottom - face_top)
    left_eye = landmarks[np.asarray(facemesh_regions.IMAGE_LEFT_EYE_INDICES)].mean(axis=0)
    right_eye = landmarks[np.asarray(facemesh_regions.IMAGE_RIGHT_EYE_INDICES)].mean(axis=0)
    eye_y = float((left_eye[1] + right_eye[1]) / 2.0)
    mouth_y = float(
        (
            landmarks[facemesh_regions.IMAGE_LEFT_MOUTH_INDEX, 1]
            + landmarks[facemesh_regions.IMAGE_RIGHT_MOUTH_INDEX, 1]
        )
        / 2.0
    )
    cheek_y = eye_y + float(roi["cheek_center_y_between_eye_mouth"]) * (mouth_y - eye_y)
    cheek_rx = float(roi["cheek_radius_x_face_width"]) * face_width
    cheek_ry = float(roi["cheek_radius_y_face_height"]) * face_height
    cheek_centers = roi["cheek_center_x_fraction"]
    left_cheek = _ellipse_mask(
        shape, face_left + float(cheek_centers[0]) * face_width, cheek_y, cheek_rx, cheek_ry
    )
    right_cheek = _ellipse_mask(
        shape, face_left + float(cheek_centers[1]) * face_width, cheek_y, cheek_rx, cheek_ry
    )
    brow_y = float(
        min(
            landmarks[np.asarray(facemesh_regions.LEFT_BROW_POLYGON), 1].mean(),
            landmarks[np.asarray(facemesh_regions.RIGHT_BROW_POLYGON), 1].mean(),
        )
    )
    forehead_x = roi["forehead_x_fraction"]
    x1 = face_left + float(forehead_x[0]) * face_width
    x2 = face_left + float(forehead_x[1]) * face_width
    y1 = face_top + float(roi["forehead_top_face_fraction"]) * face_height
    y2 = brow_y - float(roi["forehead_bottom_above_brow_fraction"]) * face_height
    forehead = np.zeros(shape, dtype=bool)
    ix1, ix2 = sorted((int(round(x1)), int(round(x2))))
    iy1, iy2 = sorted((int(round(y1)), int(round(y2))))
    forehead[max(0, iy1) : min(shape[0], iy2), max(0, ix1) : min(shape[1], ix2)] = True
    face_oval = _polygon_mask(shape, oval)
    return {
        "left_cheek_geometry": left_cheek & face_oval,
        "right_cheek_geometry": right_cheek & face_oval,
        "forehead_geometry": forehead & face_oval,
    }


def build_radiometric_layer(cube: np.ndarray, config: dict[str, Any]) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    radiometric = config["radiometric"]
    finite = np.isfinite(cube).all(axis=-1)
    tolerance = float(radiometric["reflectance_tolerance"])
    in_range = (cube >= -tolerance).all(axis=-1) & (cube <= 1.0 + tolerance).all(axis=-1)
    clipped = np.clip(cube, 0.0, 1.0)
    broadband = np.median(clipped, axis=-1)
    nonzero = broadband > float(radiometric["minimum_broadband_reflectance"])
    saturated = (clipped >= float(radiometric["saturation_band_threshold"])).mean(axis=-1) >= float(
        radiometric["saturation_min_band_fraction"]
    )
    valid = finite & in_range & nonzero & ~saturated
    return valid, {"broadband": broadband, "saturated": saturated, "nonfinite": ~finite, "out_of_range": ~in_range}


def _scanline_artifact_mask(
    broadband: np.ndarray,
    candidate: np.ndarray,
    config: dict[str, Any],
) -> np.ndarray:
    illumination = config["illumination"]
    anomaly = np.zeros(candidate.shape, dtype=bool)
    min_pixels = int(illumination["scanline_min_candidate_pixels"])
    z_threshold = float(illumination["scanline_robust_z"])
    abs_threshold = float(illumination["scanline_min_absolute_jump"])
    for axis in (0, 1):
        length = candidate.shape[axis]
        medians = np.full(length, np.nan, dtype=np.float64)
        for index in range(length):
            selector = candidate[index, :] if axis == 0 else candidate[:, index]
            values = broadband[index, :][selector] if axis == 0 else broadband[:, index][selector]
            if values.size >= min_pixels:
                medians[index] = float(np.median(values))
        differences = np.abs(np.diff(medians))
        finite = np.isfinite(differences)
        if not finite.any():
            continue
        center = float(np.median(differences[finite]))
        mad = float(np.median(np.abs(differences[finite] - center)))
        scale = max(1.4826 * mad, 1.0e-6)
        flagged = np.flatnonzero(finite & (differences >= abs_threshold) & ((differences - center) / scale >= z_threshold))
        for location in flagged:
            if axis == 0:
                anomaly[location : location + 2, :] = True
            else:
                anomaly[:, location : location + 2] = True
    dilation = int(illumination["scanline_dilation_px"])
    return ndimage.binary_dilation(anomaly, structure=_disk(dilation)) if dilation > 0 else anomaly


def build_illumination_layer(
    rgb: np.ndarray,
    broadband: np.ndarray,
    semantic_candidate: np.ndarray,
    config: dict[str, Any],
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    illumination = config["illumination"]
    rgb_float = np.asarray(rgb, dtype=np.float64) / 255.0
    value = rgb_float.max(axis=-1)
    chroma = rgb_float.max(axis=-1) - rgb_float.min(axis=-1)
    shadow = broadband < float(illumination["minimum_broadband_reflectance"])
    hsi_highlight = broadband > float(illumination["maximum_broadband_reflectance"])
    rgb_specular = (value >= float(illumination["rgb_specular_value_min"])) & (
        chroma <= float(illumination["rgb_specular_chroma_max"])
    )
    scanline = _scanline_artifact_mask(broadband, semantic_candidate, config)
    valid = ~shadow & ~hsi_highlight & ~rgb_specular & ~scanline
    return valid, {
        "shadow": shadow,
        "hsi_highlight": hsi_highlight,
        "rgb_specular": rgb_specular,
        "scanline_artifact": scanline,
    }


def _largest_component_fraction(mask: np.ndarray) -> float:
    labels, count = ndimage.label(mask)
    if count == 0:
        return 0.0
    sizes = np.bincount(labels.ravel())[1:]
    return float(sizes.max() / max(int(mask.sum()), 1))


def evaluate_sample_qc(
    direction: str,
    anatomical: np.ndarray,
    semantic_valid: np.ndarray,
    radiometric_valid: np.ndarray,
    illumination_valid: np.ndarray,
    regions: dict[str, np.ndarray],
    config: dict[str, Any],
) -> dict[str, Any]:
    roi = config["roi"]
    thresholds = config["qc"]
    anatomical_count = int(anatomical.sum())
    whole_count = int(regions["whole_skin"].sum())
    left_count = int(regions["left_cheek"].sum())
    right_count = int(regions["right_cheek"].sum())
    forehead_count = int(regions["forehead"].sum())
    denominator = max(anatomical_count, 1)
    retention = whole_count / denominator
    radiometric_exclusion = float((semantic_valid & ~radiometric_valid).sum() / max(int(semantic_valid.sum()), 1))
    illumination_exclusion = float(
        (semantic_valid & radiometric_valid & ~illumination_valid).sum()
        / max(int((semantic_valid & radiometric_valid).sum()), 1)
    )
    largest_component = _largest_component_fraction(regions["whole_skin"])
    failure_codes: list[str] = []
    review_codes: list[str] = []
    if whole_count < int(roi["minimum_whole_skin_pixels"]):
        failure_codes.append("whole_skin_too_small")
    left_ok = left_count >= int(roi["minimum_cheek_pixels"])
    right_ok = right_count >= int(roi["minimum_cheek_pixels"])
    if direction == "front" and not (left_ok and right_ok):
        failure_codes.append("front_requires_both_cheeks")
    if direction == "left" and not right_ok:
        failure_codes.append("left_view_requires_image_right_cheek")
    if direction == "right" and not left_ok:
        failure_codes.append("right_view_requires_image_left_cheek")
    if forehead_count < int(roi["minimum_forehead_pixels"]):
        review_codes.append("forehead_secondary_roi_small")
    if retention < float(thresholds["minimum_final_to_anatomical_fraction"]):
        failure_codes.append("low_final_skin_retention")
    component_threshold = float(
        thresholds["minimum_largest_component_fraction_side"]
        if direction in {"left", "right"}
        else thresholds["minimum_largest_component_fraction"]
    )
    if largest_component < component_threshold:
        review_codes.append("fragmented_whole_skin")
    if radiometric_exclusion > float(thresholds["maximum_radiometric_exclusion_fraction"]):
        failure_codes.append("excessive_radiometric_exclusion")
    if illumination_exclusion > float(thresholds["maximum_illumination_exclusion_fraction"]):
        failure_codes.append("excessive_illumination_exclusion")
    status = "FAIL" if failure_codes else ("REVIEW" if review_codes else "PASS")
    return {
        "status": status,
        "failure_codes": failure_codes,
        "review_codes": review_codes,
        "anatomical_skin_pixels": anatomical_count,
        "semantic_valid_pixels": int(semantic_valid.sum()),
        "radiometric_valid_pixels": int(radiometric_valid.sum()),
        "illumination_valid_pixels": int(illumination_valid.sum()),
        "whole_skin_pixels": whole_count,
        "left_cheek_pixels": left_count,
        "right_cheek_pixels": right_count,
        "forehead_pixels": forehead_count,
        "final_to_anatomical_fraction": float(retention),
        "radiometric_exclusion_fraction": radiometric_exclusion,
        "illumination_exclusion_fraction": illumination_exclusion,
        "largest_component_fraction": largest_component,
        "left_cheek_visible": bool(left_ok),
        "right_cheek_visible": bool(right_ok),
        "forehead_visible": bool(forehead_count >= int(roi["minimum_forehead_pixels"])),
    }


def _blend(rgb: np.ndarray, masks: list[tuple[np.ndarray, tuple[int, int, int]]], alpha: float = 0.42) -> np.ndarray:
    output = np.asarray(rgb, dtype=np.float64).copy()
    for mask, color in masks:
        output[mask] = (1.0 - alpha) * output[mask] + alpha * np.asarray(color)
    return np.clip(output, 0, 255).astype(np.uint8)


def _save_qc_panel(
    path: Path,
    rgb: np.ndarray,
    anatomical: np.ndarray,
    semantic_valid: np.ndarray,
    regions: dict[str, np.ndarray],
    status: str,
) -> None:
    size = (384, 384)
    panels = [
        Image.fromarray(rgb).resize(size, Image.Resampling.LANCZOS),
        Image.fromarray(_blend(rgb, [(anatomical, (255, 160, 0))])).resize(size, Image.Resampling.LANCZOS),
        Image.fromarray(_blend(rgb, [(semantic_valid, (255, 230, 0))])).resize(size, Image.Resampling.LANCZOS),
        Image.fromarray(
            _blend(
                rgb,
                [
                    (regions["whole_skin"], (0, 210, 60)),
                    (regions["left_cheek"], (0, 220, 255)),
                    (regions["right_cheek"], (255, 0, 220)),
                    (regions["forehead"], (40, 120, 255)),
                ],
                alpha=0.55,
            )
        ).resize(size, Image.Resampling.LANCZOS),
    ]
    canvas = Image.new("RGB", (size[0] * 4, size[1] + 34), "white")
    draw = ImageDraw.Draw(canvas)
    titles = ("RGB", "anatomical", "semantic core", f"final ROIs | {status}")
    for index, (panel, title) in enumerate(zip(panels, titles)):
        canvas.paste(panel, (index * size[0], 34))
        draw.text((index * size[0] + 6, 8), title, fill="black")
    canvas.save(path)


def _save_contact_sheet(path: Path, rows: pd.DataFrame) -> None:
    thumbnails: list[tuple[str, Image.Image]] = []
    for _, row in rows.iterrows():
        image = Image.open(row["qc_panel_path"]).convert("RGB")
        width = 480
        height = int(round(image.height * width / image.width))
        stratum = f"{row.get('expression', '?')}/{row.get('direction', '?')}"
        reason = str(row.get("review_codes", "")).strip()
        title = f"{row['sample_id']} | {stratum} | {row['status']}"
        if reason:
            title += f" | {reason}"
        thumbnails.append((title, image.resize((width, height))))
    columns = 2
    cell_w, cell_h = 500, max((image.height for _, image in thumbnails), default=120) + 28
    canvas = Image.new("RGB", (columns * cell_w, int(np.ceil(len(thumbnails) / columns)) * cell_h), "white")
    draw = ImageDraw.Draw(canvas)
    for index, (title, image) in enumerate(thumbnails):
        x = (index % columns) * cell_w
        y = (index // columns) * cell_h
        draw.text((x + 5, y + 4), title, fill="black")
        canvas.paste(image, (x + 5, y + 24))
    canvas.save(path)
    for _, image in thumbnails:
        image.close()


def _select_manual_review_rows(frame: pd.DataFrame, maximum: int) -> pd.DataFrame:
    """Select worst REVIEW cases plus one representative PASS per stratum."""

    available = frame.loc[frame["qc_panel_path"].notna()].copy()
    if len(available) <= maximum:
        return available.sort_values(["status", "expression", "direction", "sample_id"])
    representatives: list[pd.DataFrame] = []
    passed = available.loc[available["status"] == "PASS"]
    for _, group in passed.groupby(["expression", "direction"], sort=True):
        ordered = group.sort_values(["final_to_anatomical_fraction", "sample_id"])
        representatives.append(ordered.iloc[[len(ordered) // 2]])
    representative = pd.concat(representatives, ignore_index=False) if representatives else available.iloc[0:0]
    remaining = maximum - len(representative)
    review = available.loc[available["status"] != "PASS"].sort_values(
        ["final_to_anatomical_fraction", "largest_component_fraction", "sample_id"]
    )
    chosen = pd.concat([review.head(max(remaining, 0)), representative], ignore_index=False)
    if len(chosen) < maximum:
        unused = available.drop(index=chosen.index, errors="ignore").sort_values(
            ["final_to_anatomical_fraction", "sample_id"]
        )
        chosen = pd.concat([chosen, unused.head(maximum - len(chosen))], ignore_index=False)
    return chosen.head(maximum)


def _validate_upstream(contract: dict[str, Any], registration: dict[str, Any]) -> None:
    if contract.get("status") != "PASS":
        raise ValueError("S1-0 contract is not PASS")
    if registration.get("status") != "PASS" or not registration.get("coordinate_mapping_frozen"):
        raise ValueError("S1-1 coordinate mapping is not frozen")
    if registration.get("frozen_transform") != "transpose":
        raise ValueError("S1-2 v1 requires frozen transpose mapping")
    if registration.get("next_stage_allowed") is not True:
        raise ValueError("S1-1 did not authorize S1-2")
    if registration.get("input_manifest_sha256") != contract["manifest"]["sha256"]:
        raise ValueError("S1-0/S1-1 manifest hashes disagree")
    manifest = Path(contract["manifest"]["path"])
    if sha256_file(manifest) != contract["manifest"]["sha256"]:
        raise ValueError("Current split manifest hash differs from S1-0")


def run_s1_2_masks(
    contract_path: str | Path,
    registration_path: str | Path,
    output_root: str | Path,
    *,
    config_path: str | Path = DEFAULT_MASK_CONFIG,
    split: str = "train",
    max_samples: int | None = None,
    frozen_protocol_provenance: str | Path | None = None,
) -> dict[str, Any]:
    if split not in {"train", "valid"}:
        raise ValueError("S1-2 only permits train or valid; Test is blocked")
    contract_file = Path(contract_path).resolve()
    registration_file = Path(registration_path).resolve()
    config_file = Path(config_path).resolve()
    contract = _read_json(contract_file)
    registration = _read_json(registration_file)
    config = yaml.safe_load(config_file.read_text(encoding="utf-8"))
    _validate_upstream(contract, registration)
    frozen_provenance: dict[str, Any] | None = None
    if split == "valid":
        if frozen_protocol_provenance is None:
            raise ValueError("Validation requires --frozen-protocol-provenance")
        frozen_file = Path(frozen_protocol_provenance).resolve()
        frozen_provenance = _read_json(frozen_file)
        if frozen_provenance.get("status") != "FROZEN":
            raise ValueError("S1-2 Train protocol is not frozen")
        if frozen_provenance.get("config_sha256") != sha256_file(config_file):
            raise ValueError("Validation config does not match the frozen Train config")
        if frozen_provenance.get("contract_sha256") != sha256_file(contract_file):
            raise ValueError("Validation contract does not match the frozen Train contract")
        if frozen_provenance.get("registration_sha256") != sha256_file(registration_file):
            raise ValueError("Validation registration decision does not match the frozen Train decision")
        source_train_decision_file = Path(frozen_provenance["source_train_decision"]).resolve()
        source_train_decision = _read_json(source_train_decision_file)
        if frozen_provenance.get("source_train_decision_sha256") != sha256_file(source_train_decision_file):
            raise ValueError("Frozen Train decision hash has changed")
        if (
            source_train_decision.get("status") != "PASS"
            or source_train_decision.get("protocol_frozen") is not True
            or source_train_decision.get("validation_allowed") is not True
        ):
            raise ValueError("S1-2 Train decision does not authorize Validation")
        if frozen_provenance.get("implementation_sha256") != _implementation_hashes():
            raise ValueError("Validation implementation differs from the frozen Train implementation")
        checkpoint_path = Path(config["runtime"]["parser_checkpoint"])
        if not checkpoint_path.is_absolute():
            checkpoint_path = PROJECT_ROOT / checkpoint_path
        if frozen_provenance.get("parser_checkpoint_sha256") != sha256_file(checkpoint_path.resolve()):
            raise ValueError("Validation parser checkpoint differs from the frozen Train checkpoint")
    output = Path(output_root).resolve()
    revision = str(config.get("output_revision", "r1"))
    mask_revision_root = output / "masks" / revision
    split_mask_root = mask_revision_root / split
    manifest_out = output / "manifests" / f"mask_manifest_{split}_{revision}.parquet"
    decision_path = mask_revision_root / f"s1_2_{split}_decision.json"
    for path in (split_mask_root, manifest_out, decision_path):
        if path.exists():
            raise FileExistsError(f"Refusing to overwrite S1-2 output: {path}")
    split_mask_root.parent.mkdir(parents=True, exist_ok=True)
    manifest_out.parent.mkdir(parents=True, exist_ok=True)
    split_mask_root.mkdir(parents=True, exist_ok=False)

    manifest_path = Path(contract["manifest"]["path"])
    full_manifest = pd.read_csv(manifest_path)
    selected_manifest = full_manifest.loc[full_manifest["split"] == split].copy()
    if max_samples is not None:
        selected_manifest = selected_manifest.iloc[: int(max_samples)].copy()
        worker_manifest = mask_revision_root / f"worker_input_{split}_{max_samples}.csv"
        mask_revision_root.mkdir(parents=True, exist_ok=True)
        selected_manifest.to_csv(worker_manifest, index=False, encoding="utf-8-sig")
    else:
        worker_manifest = manifest_path
    runtime = config["runtime"]
    python_path = Path(runtime["rgb_worker_python"])
    command = [
        str(python_path),
        str(RGB_WORKER),
        "--manifest",
        str(worker_manifest),
        "--config",
        str(config_file),
        "--split",
        split,
        "--output-root",
        str(mask_revision_root),
    ]
    completed = subprocess.run(command, cwd=PROJECT_ROOT, text=True, capture_output=True, check=False)
    (mask_revision_root / f"rgb_worker_{split}.log").write_text(
        completed.stdout + "\nSTDERR\n" + completed.stderr, encoding="utf-8"
    )
    if completed.returncode != 0:
        raise RuntimeError(f"RGB worker failed with code {completed.returncode}; see {mask_revision_root / f'rgb_worker_{split}.log'}")
    worker_rows = pd.read_csv(mask_revision_root / f"rgb_worker_{split}.csv").set_index("sample_id")
    rows: list[dict[str, Any]] = []
    for _, sample in selected_manifest.iterrows():
        sample_id = str(sample["sample_id"])
        sample_dir = split_mask_root / sample_id
        base = {
            "sample_id": sample_id,
            "subject_id": sample["subject_id"],
            "split": split,
            "expression": sample["expression"],
            "direction": sample["direction"],
            "config_sha256": sha256_file(config_file),
            "input_manifest_sha256": contract["manifest"]["sha256"],
            "registration_decision_sha256": sha256_file(registration_file),
            "frozen_transform": registration["frozen_transform"],
            "status": "FAIL",
            "failure_codes": "",
            "review_codes": "",
        }
        try:
            if sample_id not in worker_rows.index or worker_rows.loc[sample_id, "status"] != "PASS":
                detail = "missing_rgb_worker_result" if sample_id not in worker_rows.index else str(worker_rows.loc[sample_id, "failure_detail"])
                raise RuntimeError(f"rgb_worker_failed:{detail}")
            rgb = np.asarray(Image.open(sample["rgb_path"]).convert("RGB"), dtype=np.uint8)
            labels = np.load(sample_dir / "parsing_label.npy", allow_pickle=False)
            landmarks = np.load(sample_dir / "landmarks.npy", allow_pickle=False)
            anatomical, semantic_valid, semantic_parts = build_semantic_layers(labels, landmarks, config)
            cube = load_hyperskin_cube(
                sample["hsi_path"], dataset_key=contract["hsi_storage"]["dataset_key"], expected_bands=31
            )
            cube = np.transpose(cube, (1, 0, 2))
            if cube.shape[:2] != rgb.shape[:2]:
                raise ValueError(f"aligned_cube_shape_mismatch:{cube.shape}:{rgb.shape}")
            radiometric_valid, radiometric_parts = build_radiometric_layer(cube, config)
            illumination_valid, illumination_parts = build_illumination_layer(
                rgb, radiometric_parts["broadband"], semantic_valid & radiometric_valid, config
            )
            final_skin = anatomical & semantic_valid & radiometric_valid & illumination_valid
            geometry = build_roi_geometry(rgb.shape[:2], landmarks, config)
            regions = {
                "whole_skin": final_skin,
                "left_cheek": final_skin & geometry["left_cheek_geometry"],
                "right_cheek": final_skin & geometry["right_cheek_geometry"],
                "forehead": final_skin & geometry["forehead_geometry"],
            }
            # In the named side views, the contralateral ellipse drifts toward
            # the nose/occluded side and is not a defensible cheek observation.
            # Keep its file for a uniform contract, but make it explicitly empty.
            if str(sample["direction"]) == "left":
                regions["left_cheek"] = np.zeros_like(final_skin)
            elif str(sample["direction"]) == "right":
                regions["right_cheek"] = np.zeros_like(final_skin)
            qc = evaluate_sample_qc(
                str(sample["direction"]), anatomical, semantic_valid, radiometric_valid, illumination_valid, regions, config
            )
            arrays = {
                "anatomical_skin": anatomical,
                "semantic_valid": semantic_valid,
                "radiometric_valid": radiometric_valid,
                "illumination_valid": illumination_valid,
                **regions,
            }
            for name, array in arrays.items():
                np.save(sample_dir / f"{name}.npy", np.asarray(array, dtype=bool), allow_pickle=False)
                base[f"{name}_path"] = str(sample_dir / f"{name}.npy")
            diagnostic_counts = {
                "parsed_exclusion_pixels": int(semantic_parts["parsed_exclusion"].sum()),
                "landmark_exclusion_pixels": int(semantic_parts["landmark_exclusion"].sum()),
                "saturated_pixels": int(radiometric_parts["saturated"].sum()),
                "shadow_pixels": int(illumination_parts["shadow"].sum()),
                "hsi_highlight_pixels": int(illumination_parts["hsi_highlight"].sum()),
                "rgb_specular_pixels": int(illumination_parts["rgb_specular"].sum()),
                "scanline_artifact_pixels": int(illumination_parts["scanline_artifact"].sum()),
            }
            qc_payload = {
                **base,
                **qc,
                **diagnostic_counts,
                "coordinate_system": "aligned_rgb_yx",
                "left_right_definition": "image-coordinate left/right, not anatomical laterality",
            }
            _write_json(sample_dir / "qc.json", qc_payload)
            panel_path = sample_dir / "qc_panel.png"
            _save_qc_panel(panel_path, rgb, anatomical, semantic_valid, regions, qc["status"])
            base.update(qc)
            base.update(diagnostic_counts)
            base["qc_json_path"] = str(sample_dir / "qc.json")
            base["qc_panel_path"] = str(panel_path)
        except Exception as error:
            base["failure_codes"] = f"pipeline_exception:{repr(error)}"
        if isinstance(base.get("failure_codes"), list):
            base["failure_codes"] = "|".join(base["failure_codes"])
        if isinstance(base.get("review_codes"), list):
            base["review_codes"] = "|".join(base["review_codes"])
        rows.append(base)
        if len(rows) == 1 or len(rows) % 20 == 0 or len(rows) == len(selected_manifest):
            print(f"S1-2 {split}: {len(rows)}/{len(selected_manifest)}", flush=True)
    frame = pd.DataFrame(rows)
    frame.to_parquet(manifest_out, index=False)
    frame.to_csv(manifest_out.with_suffix(".csv"), index=False, encoding="utf-8-sig")
    success_for_review = frame.loc[frame["qc_panel_path"].notna()].copy() if "qc_panel_path" in frame else frame.iloc[0:0]
    if not success_for_review.empty:
        review_rows = _select_manual_review_rows(
            success_for_review, int(config["qc"]["review_contact_sheet_max_samples"])
        )
        _save_contact_sheet(mask_revision_root / f"{split}_mask_contact_sheet.png", review_rows)
        review_rows[["sample_id", "status", "failure_codes", "review_codes", "qc_panel_path"]].assign(
            reviewer_decision="", notes=""
        ).to_csv(mask_revision_root / f"manual_review_{split}.csv", index=False, encoding="utf-8-sig")
    counts = frame["status"].value_counts().to_dict()
    total = max(len(frame), 1)
    failure_fraction = float(counts.get("FAIL", 0) / total)
    review_fraction = float(counts.get("REVIEW", 0) / total)
    stratum = (
        frame.assign(usable_flag=~frame["status"].eq("FAIL"))
        .groupby(["expression", "direction"], as_index=False)
        .agg(samples=("sample_id", "size"), usable_fraction=("usable_flag", "mean"))
    )
    stratum.to_csv(mask_revision_root / f"{split}_stratum_qc.csv", index=False, encoding="utf-8-sig")
    automatic_checks = {
        "failure_fraction": failure_fraction <= float(config["qc"]["maximum_sample_failure_fraction"]),
        "review_fraction": review_fraction <= float(config["qc"]["maximum_sample_review_fraction"]),
        "all_strata_usable_fraction": bool(
            (stratum["usable_fraction"] >= float(config["qc"]["minimum_stratum_usable_fraction"])).all()
        ),
    }
    automatic_pass = all(automatic_checks.values())
    decision = {
        "schema_version": 1,
        "stage": "S1-2",
        "split": split,
        "output_revision": revision,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "AUTOMATIC_PASS_MANUAL_REVIEW_PENDING" if automatic_pass else "REVISE",
        "automatic_decision": "PASS" if automatic_pass else "REVISE",
        "automatic_checks": automatic_checks,
        "sample_counts": {key: int(value) for key, value in counts.items()},
        "sample_count": int(len(frame)),
        "failure_fraction": failure_fraction,
        "review_fraction": review_fraction,
        "config_path": str(config_file),
        "config_sha256": sha256_file(config_file),
        "contract_path": str(contract_file),
        "contract_sha256": sha256_file(contract_file),
        "registration_path": str(registration_file),
        "registration_sha256": sha256_file(registration_file),
        "manifest_path": str(manifest_out),
        "manifest_sha256": sha256_file(manifest_out),
        "parser_checkpoint_sha256": str(worker_rows["parser_checkpoint_sha256"].dropna().iloc[0]) if "parser_checkpoint_sha256" in worker_rows and worker_rows["parser_checkpoint_sha256"].notna().any() else None,
        "manual_review": {"status": "pending", "path": str(mask_revision_root / f"manual_review_{split}.csv")},
        "protocol_frozen": False,
        "next_stage_allowed": False,
        "test_access_count": 0,
        "frozen_train_protocol_provenance": str(Path(frozen_protocol_provenance).resolve()) if frozen_protocol_provenance else None,
    }
    _write_json(decision_path, decision)
    return decision


def finalize_s1_2_train_review(
    decision_path: str | Path,
    *,
    reviewer: str,
    approve: bool,
    notes: str,
) -> dict[str, Any]:
    decision_file = Path(decision_path).resolve()
    decision = _read_json(decision_file)
    if decision.get("split") != "train" or decision.get("manual_review", {}).get("status") != "pending":
        raise ValueError("Expected an unreviewed S1-2 Train decision")
    if not reviewer.strip() or not notes.strip():
        raise ValueError("Reviewer and notes are required")
    if approve and decision.get("automatic_decision") != "PASS":
        raise ValueError("Cannot freeze a protocol that failed automatic QC")
    freeze_dir = decision_file.parents[2] / "freeze"
    frozen_yaml = freeze_dir / "s1_2_mask_protocol.yaml"
    frozen_json = freeze_dir / "s1_2_mask_protocol_provenance.json"
    if approve and (frozen_yaml.exists() or frozen_json.exists()):
        raise FileExistsError("Refusing to overwrite an existing S1-2 freeze")
    source_config = Path(decision["config_path"]).resolve()
    if approve and sha256_file(source_config) != decision["config_sha256"]:
        raise ValueError("Train mask config changed after the automatic run")
    source_manifest = Path(decision["manifest_path"]).resolve()
    if approve and sha256_file(source_manifest) != decision["manifest_sha256"]:
        raise ValueError("Train mask manifest changed after the automatic run")
    if approve and sha256_file(Path(decision["contract_path"]).resolve()) != decision["contract_sha256"]:
        raise ValueError("Data contract changed after the Train mask run")
    if approve and sha256_file(Path(decision["registration_path"]).resolve()) != decision["registration_sha256"]:
        raise ValueError("Registration decision changed after the Train mask run")
    if approve:
        source_config_data = yaml.safe_load(source_config.read_text(encoding="utf-8"))
        checkpoint_path = Path(source_config_data["runtime"]["parser_checkpoint"])
        if not checkpoint_path.is_absolute():
            checkpoint_path = PROJECT_ROOT / checkpoint_path
        if sha256_file(checkpoint_path.resolve()) != decision["parser_checkpoint_sha256"]:
            raise ValueError("Parser checkpoint changed after the Train mask run")
    automatic_review_path = Path(decision["manual_review"]["path"])
    stratified_review_path = decision_file.parent / "manual_review_train_stratified.csv"
    review_path = stratified_review_path if stratified_review_path.exists() else automatic_review_path
    stratified_contact_sheet = decision_file.parent / "train_mask_contact_sheet_stratified.png"
    contact_sheet = (
        stratified_contact_sheet
        if stratified_contact_sheet.exists()
        else decision_file.parent / "train_mask_contact_sheet.png"
    )
    review = pd.read_csv(review_path, keep_default_na=False)
    reviewed_utc = datetime.now(timezone.utc).isoformat()
    review["reviewer_decision"] = "PASS" if approve else "FAIL"
    review["notes"] = notes
    review["reviewer"] = reviewer
    review["reviewed_utc"] = reviewed_utc
    review.to_csv(review_path, index=False, encoding="utf-8-sig")
    decision["manual_review"] = {
        "status": "PASS" if approve else "FAIL",
        "path": str(review_path),
        "automatic_selection_path": str(automatic_review_path),
        "contact_sheet_path": str(contact_sheet),
        "reviewer": reviewer,
        "notes": notes,
        "reviewed_utc": reviewed_utc,
    }
    decision["status"] = "PASS" if approve else "REVISE"
    decision["protocol_frozen"] = bool(approve)
    decision["frozen_config_path"] = str(frozen_yaml) if approve else None
    decision["frozen_provenance_path"] = str(frozen_json) if approve else None
    decision["next_stage_allowed"] = False
    decision["validation_allowed"] = bool(approve)
    _write_json(decision_file, decision)
    if approve:
        freeze_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_config, frozen_yaml)
        provenance = {
            "stage": "S1-2",
            "status": "FROZEN",
            "created_utc": reviewed_utc,
            "reviewer": reviewer,
            "notes": notes,
            "source_train_decision": str(decision_file),
            "source_train_decision_sha256": sha256_file(decision_file),
            "source_train_manifest": str(source_manifest),
            "source_train_manifest_sha256": decision["manifest_sha256"],
            "config_sha256": sha256_file(frozen_yaml),
            "contract_sha256": decision["contract_sha256"],
            "registration_sha256": decision["registration_sha256"],
            "parser_checkpoint_sha256": decision["parser_checkpoint_sha256"],
            "implementation_sha256": _implementation_hashes(),
            "test_access_count": 0,
        }
        _write_json(frozen_json, provenance)
    return decision


def finalize_s1_2_validation_review(
    decision_path: str | Path,
    *,
    reviewer: str,
    approve: bool,
    notes: str,
) -> dict[str, Any]:
    """Close the Validation gate; this never changes the frozen Train config."""

    decision_file = Path(decision_path).resolve()
    decision = _read_json(decision_file)
    if decision.get("split") != "valid" or decision.get("manual_review", {}).get("status") != "pending":
        raise ValueError("Expected an unreviewed S1-2 Validation decision")
    if not reviewer.strip() or not notes.strip():
        raise ValueError("Reviewer and notes are required")
    if approve and decision.get("automatic_decision") != "PASS":
        raise ValueError("Cannot approve Validation when automatic QC failed")
    output_root = decision_file.parents[2]
    final_manifest = output_root / "manifests" / "mask_manifest.parquet"
    final_manifest_csv = final_manifest.with_suffix(".csv")
    final_decision_file = output_root / "freeze" / "s1_2_final_decision.json"
    if approve:
        for output in (final_manifest, final_manifest_csv, final_decision_file):
            if output.exists():
                raise FileExistsError(f"Refusing to overwrite final S1-2 artifact: {output}")
        frozen_file = Path(decision["frozen_train_protocol_provenance"]).resolve()
        frozen = _read_json(frozen_file)
        if frozen.get("status") != "FROZEN" or frozen.get("test_access_count") != 0:
            raise ValueError("Invalid frozen Train provenance")
        train_decision_file = Path(frozen["source_train_decision"]).resolve()
        train_decision = _read_json(train_decision_file)
        if frozen.get("source_train_decision_sha256") != sha256_file(train_decision_file):
            raise ValueError("Frozen Train decision hash has changed")
        if train_decision.get("status") != "PASS" or train_decision.get("validation_allowed") is not True:
            raise ValueError("Train decision does not authorize Validation completion")
        train_manifest_file = Path(frozen["source_train_manifest"]).resolve()
        valid_manifest_file = Path(decision["manifest_path"]).resolve()
        if frozen.get("source_train_manifest_sha256") != sha256_file(train_manifest_file):
            raise ValueError("Frozen Train manifest hash has changed")
        if decision.get("manifest_sha256") != sha256_file(valid_manifest_file):
            raise ValueError("Validation manifest hash has changed")
        train_frame = pd.read_parquet(train_manifest_file)
        valid_frame = pd.read_parquet(valid_manifest_file)
        if set(train_frame["split"].unique()) != {"train"} or set(valid_frame["split"].unique()) != {"valid"}:
            raise ValueError("Train/Validation manifest split labels are invalid")
        if set(train_frame["sample_id"]) & set(valid_frame["sample_id"]):
            raise ValueError("Train/Validation sample overlap detected")
        if set(train_frame["subject_id"]) & set(valid_frame["subject_id"]):
            raise ValueError("Train/Validation subject leakage detected")
        if list(train_frame.columns) != list(valid_frame.columns):
            raise ValueError("Train/Validation mask manifest schemas differ")
        combined = pd.concat([train_frame, valid_frame], ignore_index=True)
    review_path = Path(decision["manual_review"]["path"])
    contact_sheet = decision_file.parent / "valid_mask_contact_sheet.png"
    review = pd.read_csv(review_path, keep_default_na=False)
    reviewed_utc = datetime.now(timezone.utc).isoformat()
    review["reviewer_decision"] = "PASS" if approve else "FAIL"
    review["notes"] = notes
    review["reviewer"] = reviewer
    review["reviewed_utc"] = reviewed_utc
    review.to_csv(review_path, index=False, encoding="utf-8-sig")
    decision["manual_review"] = {
        "status": "PASS" if approve else "FAIL",
        "path": str(review_path),
        "contact_sheet_path": str(contact_sheet),
        "reviewer": reviewer,
        "notes": notes,
        "reviewed_utc": reviewed_utc,
    }
    decision["status"] = "PASS" if approve else "REVISE"
    decision["next_stage_allowed"] = bool(approve)
    decision["test_access_count"] = 0
    if approve:
        final_manifest.parent.mkdir(parents=True, exist_ok=True)
        combined.to_parquet(final_manifest, index=False)
        combined.to_csv(final_manifest_csv, index=False, encoding="utf-8-sig")
        decision["final_manifest_path"] = str(final_manifest)
        decision["final_manifest_sha256"] = sha256_file(final_manifest)
        decision["final_decision_path"] = str(final_decision_file)
    _write_json(decision_file, decision)
    if approve:
        final_decision = {
            "schema_version": 1,
            "stage": "S1-2",
            "status": "PASS",
            "created_utc": reviewed_utc,
            "next_stage_allowed": True,
            "train_decision_path": str(train_decision_file),
            "train_decision_sha256": sha256_file(train_decision_file),
            "validation_decision_path": str(decision_file),
            "validation_decision_sha256": sha256_file(decision_file),
            "frozen_protocol_provenance_path": str(frozen_file),
            "frozen_protocol_provenance_sha256": sha256_file(frozen_file),
            "manifest_path": str(final_manifest),
            "manifest_sha256": sha256_file(final_manifest),
            "manifest_csv_path": str(final_manifest_csv),
            "manifest_csv_sha256": sha256_file(final_manifest_csv),
            "sample_counts": {
                "train": int(len(train_frame)),
                "valid": int(len(valid_frame)),
                "total": int(len(combined)),
            },
            "reviewer": reviewer,
            "notes": notes,
            "test_access_count": 0,
        }
        final_decision_file.parent.mkdir(parents=True, exist_ok=True)
        _write_json(final_decision_file, final_decision)
    return decision
