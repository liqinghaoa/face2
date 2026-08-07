"""Incremental P0-A closeout with no face detection, parsing, or mask editing."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .asset_io import atomic_write_text, save_png, write_dataframe_csv

AMBIGUOUS_IDS = frozenset({"A001917272-1", "A002081031-1"})
IMAGE_SIZE = 224
LARGE_DIFF_THRESHOLD = 5


def physics_core_skin_from_effective_masks(left_cheek: np.ndarray, right_cheek: np.ndarray, forehead: np.ndarray) -> np.ndarray:
    """Define the P0 physics core as left/right cheek or forehead effective skin.

    Chin, lip, eye, and the remainder of strict skin are intentionally excluded.
    """
    arrays = (np.asarray(left_cheek), np.asarray(right_cheek), np.asarray(forehead))
    if any(value.shape != (IMAGE_SIZE, IMAGE_SIZE) for value in arrays):
        raise ValueError("physics-core inputs must be 224x224")
    return np.logical_or.reduce(tuple(value > 0 for value in arrays)).astype(np.uint8) * 255


def _cv2() -> Any:
    import cv2
    return cv2


def _read_image(path: Path, grayscale: bool = False) -> np.ndarray:
    cv2 = _cv2()
    flag = cv2.IMREAD_GRAYSCALE if grayscale else cv2.IMREAD_COLOR
    value = cv2.imdecode(np.fromfile(path, dtype=np.uint8), flag)
    if value is None:
        raise FileNotFoundError(f"cannot read raster: {path}")
    if not grayscale:
        value = cv2.cvtColor(value, cv2.COLOR_BGR2RGB)
    return value


def _reconstruct_black_background(aligned_rgb: np.ndarray, final_mask: np.ndarray) -> np.ndarray:
    """Reproduce the fixed Global feather=11 black compositing rule."""
    cv2 = _cv2()
    alpha = cv2.GaussianBlur(final_mask, (11, 11), sigmaX=max(1.0, 11 / 3.0)).astype(np.float32) / 255.0
    return np.clip(np.rint(aligned_rgb.astype(np.float32) * alpha[:, :, None]), 0, 255).astype(np.uint8)


def _bbox(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    ys, xs = np.where(mask)
    if len(xs) == 0:
        return None
    return int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1


def _overlay(image: np.ndarray, mask: np.ndarray, color: tuple[int, int, int]) -> np.ndarray:
    result = image.copy()
    selected = mask > 0
    result[selected] = np.clip(0.55 * result[selected] + 0.45 * np.asarray(color), 0, 255).astype(np.uint8)
    return result


def _draw_contour(image: np.ndarray, mask: np.ndarray, color: tuple[int, int, int]) -> np.ndarray:
    cv2 = _cv2()
    result = image.copy()
    contours, _ = cv2.findContours((mask > 0).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(result, contours, -1, color, 1)
    return result


def _save_ambiguity_qc(path: Path, aligned: np.ndarray, labels: np.ndarray, final_mask: np.ndarray, strict_skin: np.ndarray, physics_core: np.ndarray, historical: np.ndarray, reconstructed: np.ndarray, mismatch: np.ndarray, bbox: tuple[int, int, int, int] | None, effective: dict[str, np.ndarray]) -> None:
    """Save label-free QC for localized jaw-neck discrepancy inspection."""
    cv2 = _cv2()
    palette = np.array([[0, 0, 0], [214, 170, 133], [80, 190, 80], [70, 120, 210], [180, 60, 180], [255, 200, 40]], dtype=np.uint8)
    labels_rgb = palette[labels % len(palette)]
    roi_overlay = aligned.copy()
    for name, color in (("left_cheek", (70, 130, 255)), ("right_cheek", (70, 130, 255)), ("forehead", (255, 220, 40)), ("lip", (255, 80, 190))):
        roi_overlay = _draw_contour(roi_overlay, effective[name], color)
    heat = cv2.applyColorMap(np.clip(mismatch.astype(np.uint8) * 255, 0, 255), cv2.COLORMAP_JET)
    heat = cv2.cvtColor(heat, cv2.COLOR_BGR2RGB)
    bbox_panel = heat.copy()
    if bbox is not None:
        x1, y1, x2, y2 = bbox
        cv2.rectangle(bbox_panel, (x1, y1), (x2 - 1, y2 - 1), (255, 255, 255), 1)
    panels = [aligned, labels_rgb, _overlay(aligned, final_mask, (255, 90, 40)), _overlay(aligned, strict_skin, (30, 220, 70)), _overlay(aligned, physics_core, (120, 220, 255)), historical, reconstructed, bbox_panel, roi_overlay]
    panels = [cv2.resize(panel, (224, 224), interpolation=cv2.INTER_AREA) for panel in panels]
    sheet = np.vstack((np.hstack(panels[:3]), np.hstack(panels[3:6]), np.hstack(panels[6:])))
    save_png(path, sheet)


def _core_asset_exists(root: Path, image_id: str, row: dict[str, Any]) -> bool:
    return all(_asset_path(root, row, key, fallback).is_file() for key, fallback in _base_asset_relpaths(image_id, row).items())


def _base_asset_relpaths(image_id: str, row: dict[str, Any]) -> dict[str, str]:
    suffix = Path(str(row.get("source_path", ""))).suffix or ".jpg"
    return {"raw_scene_relpath": f"images/raw_scene/{image_id}{suffix}", "aligned_scene_relpath": f"images/aligned_scene_224/{image_id}.png", "aligned_blackbg_relpath": f"images/aligned_blackbg_224/{image_id}.png", "e0b_meanbg_relpath": f"images/e0b_meanbg_224/{image_id}.png", "parsing_label_relpath": f"parsing/parsing_label_224/{image_id}.png", "source_valid_mask_relpath": f"masks/source_valid_224/{image_id}.png", "final_face_mask_relpath": f"masks/final_face_mask_224/{image_id}.png", "face_valid_mask_relpath": f"masks/face_valid_224/{image_id}.png", "skin_strict_mask_relpath": f"masks/skin_strict_224/{image_id}.png", "alignment_npz_relpath": f"transforms/alignment/{image_id}.npz"}


def _asset_path(root: Path, row: dict[str, Any], key: str, fallback: str) -> Path:
    value = str(row.get(key, "") or fallback)
    return root / value


def _analyze_ambiguity(root: Path, row: dict[str, Any], physics_core: np.ndarray, effective: dict[str, np.ndarray]) -> dict[str, Any]:
    image_id = str(row["ID"])
    paths = _base_asset_relpaths(image_id, row)
    aligned = _read_image(_asset_path(root, row, "aligned_scene_relpath", paths["aligned_scene_relpath"]))
    historical = _read_image(_asset_path(root, row, "aligned_blackbg_relpath", paths["aligned_blackbg_relpath"]))
    final_mask = _read_image(_asset_path(root, row, "final_face_mask_relpath", paths["final_face_mask_relpath"]), grayscale=True)
    strict_skin = _read_image(_asset_path(root, row, "skin_strict_mask_relpath", paths["skin_strict_mask_relpath"]), grayscale=True)
    labels = _read_image(_asset_path(root, row, "parsing_label_relpath", paths["parsing_label_relpath"]), grayscale=True)
    reconstructed = _reconstruct_black_background(aligned, final_mask)
    difference = np.abs(historical.astype(np.int16) - reconstructed.astype(np.int16))
    mismatch = (difference > LARGE_DIFF_THRESHOLD).any(axis=2)
    bbox = _bbox(mismatch)
    pixel_count = int(mismatch.sum())
    face_pixels = max(1, int((final_mask > 0).sum()))
    mismatch_face = int((mismatch & (final_mask > 0)).sum())
    affects = {name: bool((mismatch & (mask > 0)).any()) for name, mask in effective.items()}
    affects_physics = bool((mismatch & (physics_core > 0)).any())
    chin_y1 = int(float(row.get("chin_bbox_y1", IMAGE_SIZE)))
    is_jaw_neck = bbox is not None and bbox[1] >= max(0, chin_y1 - 10) and not affects_physics
    region = "jaw_neck_boundary" if is_jaw_neck else "other"
    _save_ambiguity_qc(root / "qc_preview" / "jaw_neck_boundary_ambiguity" / f"{image_id}.png", aligned, labels, final_mask, strict_skin, physics_core, historical, reconstructed, mismatch, bbox, effective)
    return {"legacy_mismatch_pixel_count": pixel_count, "legacy_mismatch_fraction_image": pixel_count / float(IMAGE_SIZE * IMAGE_SIZE), "legacy_mismatch_fraction_face": mismatch_face / float(face_pixels), "legacy_mismatch_bbox_x1": "" if bbox is None else bbox[0], "legacy_mismatch_bbox_y1": "" if bbox is None else bbox[1], "legacy_mismatch_bbox_x2": "" if bbox is None else bbox[2], "legacy_mismatch_bbox_y2": "" if bbox is None else bbox[3], "legacy_mismatch_region": region, "legacy_mismatch_affects_left_cheek": int(affects["left_cheek"]), "legacy_mismatch_affects_right_cheek": int(affects["right_cheek"]), "legacy_mismatch_affects_forehead": int(affects["forehead"]), "legacy_mismatch_affects_lip": int(affects["lip"]), "legacy_mismatch_affects_physics_core": int(affects_physics), "jaw_neck_boundary_ambiguous": int(is_jaw_neck)}


def finalize_boundary_ambiguity(project_root: Path) -> dict[str, Any]:
    """Finalize existing P0 assets without rerunning FaceMesh or BiSeNet."""
    root = (project_root / "data" / "processed" / "P0_Physics_Audit_v1").resolve()
    master_path, status_path = root / "metadata/master_index.csv", root / "metadata/build_status.csv"
    master = pd.read_csv(master_path, dtype={"ID": str}).fillna("")
    status = pd.read_csv(status_path, dtype={"ID": str}).fillna("")
    if len(master) != 500 or len(status) != 500 or master["ID"].tolist() != status["ID"].tolist():
        raise ValueError("P0 finalization requires aligned 500-row master/status tables")
    qc_dir = root / "qc_preview" / "jaw_neck_boundary_ambiguity"
    core_dir = root / "masks" / "physics_core_skin_224"
    qc_dir.mkdir(parents=True, exist_ok=True); core_dir.mkdir(parents=True, exist_ok=True)
    updates: dict[str, dict[str, Any]] = {}
    for row in master.to_dict("records"):
        image_id = str(row["ID"])
        effective = {name: _read_image(root / str(row.get(f"{name}_effective_relpath") or f"masks/roi_effective_224/{name}/{image_id}.png"), grayscale=True) for name in ("left_cheek", "right_cheek", "forehead", "lip")}
        physics_core = physics_core_skin_from_effective_masks(effective["left_cheek"], effective["right_cheek"], effective["forehead"])
        core_path = core_dir / f"{image_id}.png"
        save_png(core_path, physics_core)
        base_paths = _base_asset_relpaths(image_id, row)
        face_valid = _read_image(_asset_path(root, row, "face_valid_mask_relpath", base_paths["face_valid_mask_relpath"]), grayscale=True)
        core_asset = _core_asset_exists(root, image_id, row)
        physics_available = int((effective["left_cheek"] > 0).any() and (effective["right_cheek"] > 0).any() and (physics_core > 0).any())
        strict_pass = int(str(row.get("regression_status", "")) == "passed")
        update: dict[str, Any] = {"core_asset_status": "success" if core_asset else "failed", "legacy_regression_strict_pass": strict_pass, "legacy_regression_status": "passed" if strict_pass else "failed_other", "boundary_ambiguity_status": "none", "p0_usable": int(core_asset and physics_available), "overall_status": "success" if core_asset and physics_available else "failed", "physics_core_skin_mask_relpath": core_path.relative_to(root).as_posix(), "physics_core_skin_pixel_count": int((physics_core > 0).sum()), "physics_core_skin_fraction_face": float((physics_core > 0).sum() / max(1, int((face_valid > 0).sum()))), "physics_core_skin_available": physics_available, "chin_usage": "exploratory_only", "chin_boundary_warning": 0, "chin_roi_status": str(row.get("chin_roi_status", "") or "success"), "legacy_mismatch_pixel_count": "", "legacy_mismatch_fraction_image": "", "legacy_mismatch_fraction_face": "", "legacy_mismatch_bbox_x1": "", "legacy_mismatch_bbox_y1": "", "legacy_mismatch_bbox_x2": "", "legacy_mismatch_bbox_y2": "", "legacy_mismatch_region": "", "legacy_mismatch_affects_left_cheek": "", "legacy_mismatch_affects_right_cheek": "", "legacy_mismatch_affects_forehead": "", "legacy_mismatch_affects_lip": "", "legacy_mismatch_affects_physics_core": "", "jaw_neck_boundary_ambiguous": 0, "legacy_regression_failure_reason": ""}
        if image_id in AMBIGUOUS_IDS:
            analysis = _analyze_ambiguity(root, row, physics_core, effective)
            update.update(analysis)
            update.update({"core_asset_status": "success" if core_asset else "failed", "legacy_regression_strict_pass": 0, "legacy_regression_status": "localized_jaw_neck_mismatch" if analysis["legacy_mismatch_region"] == "jaw_neck_boundary" else "failed_other", "boundary_ambiguity_status": "jaw_neck_ambiguous" if analysis["legacy_mismatch_region"] == "jaw_neck_boundary" else "none", "p0_usable": int(core_asset and physics_available), "overall_status": "success_with_boundary_ambiguity" if core_asset and physics_available else "failed", "chin_boundary_warning": 1, "chin_roi_status": "usable_with_boundary_warning", "legacy_regression_failure_reason": str(row.get("failure_reason", ""))})
        update.update({key: str(row.get(key, "") or fallback) for key, fallback in base_paths.items()})
        updates[image_id] = update
    for frame in (master, status):
        for key in next(iter(updates.values())):
            frame[key] = [updates[str(image_id)][key] for image_id in frame["ID"]]
        frame["core_status"] = frame["core_asset_status"]
    write_dataframe_csv(master_path, master)
    write_dataframe_csv(status_path, status)
    manifest_path = root / "metadata/build_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.is_file() else {}
    manifest.update({"p0_core_asset_success": f"{int((master.core_asset_status == 'success').sum())}/500", "legacy_blackbg_strict_regression": f"{int((master.legacy_regression_strict_pass.astype(int) == 1).sum())}/500", "localized_jaw_neck_boundary_ambiguity": f"{int((master.boundary_ambiguity_status == 'jaw_neck_ambiguous').sum())}/500", "p0_usable_samples": f"{int((master.p0_usable.astype(int) == 1).sum())}/500", "manual_corrections": 0, "deleted_samples": 0, "large_difference_threshold": LARGE_DIFF_THRESHOLD})
    atomic_write_text(manifest_path, json.dumps(manifest, ensure_ascii=False, indent=2))
    return {"rows": len(master), "core_asset_success": int((master.core_asset_status == "success").sum()), "strict_regression_pass": int((master.legacy_regression_strict_pass.astype(int) == 1).sum()), "boundary_ambiguity": int((master.boundary_ambiguity_status == "jaw_neck_ambiguous").sum()), "p0_usable": int((master.p0_usable.astype(int) == 1).sum())}
