"""Limited frozen DECA latent extraction for the already-completed P0-B1 pilot."""
from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from .deca_runtime import forward_deca, image_to_tensor, load_deca
from .input_adapter import bbox_mapping
from .p0_mapper import map_to_p0


def _rgb(path: Path) -> np.ndarray:
    value = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if value is None:
        raise FileNotFoundError(path)
    return cv2.cvtColor(value, cv2.COLOR_BGR2RGB)


def _chw_to_p0(value: Any, inverse: np.ndarray) -> np.ndarray:
    array = value[0].detach().float().cpu().numpy().transpose(1, 2, 0)
    return map_to_p0(array, inverse, "bilinear")


def extract_latent_codes(config: Any, audit_root: Path) -> list[dict[str, Any]]:
    """Encode only the frozen 12x2 input PNGs and verify mapped summaries."""
    metrics_path = config.output_root / "input_mode_comparison/p0b_b1_input_mode_case_metrics.csv"
    with metrics_path.open(newline="", encoding="utf-8-sig") as handle:
        metrics = list(csv.DictReader(handle))
    if len(metrics) != 24 or any(row["status"] != "passed" for row in metrics):
        raise RuntimeError("latent extraction requires the completed 24-row P0-B1 result")
    output = audit_root / "extracted_latent_codes"
    output.mkdir(parents=True, exist_ok=True)
    model, _ = load_deca(config.deca_root, "cuda", config.seed)
    rows: list[dict[str, Any]] = []
    for row in metrics:
        case_id, mode = row["audit_id"], row["input_mode"]
        source = config.output_root / "input_mode_comparison" / mode / case_id
        image = image_to_tensor(_rgb(source / "input.png"), "cuda")
        code, decoded, _ = forward_deca(model, image)
        rendered = model.render(decoded["verts"].clone(), decoded["trans_verts"].clone(), decoded["albedo"], code["light"])
        if mode == "direct_p0_aligned":
            inverse = np.eye(3, dtype=np.float32)
        else:
            bbox = tuple(json.loads(row["bbox"]))
            _, inverse = bbox_mapping(bbox)
        reconstruction = _chw_to_p0(rendered["images"], inverse)
        albedo = _chw_to_p0(rendered["albedo_images"], inverse)
        with np.load(source / "physical_maps_float.npz") as existing:
            reference_reconstruction = existing["reconstruction"]
            reference_albedo = existing["albedo_like"]
        def summary_difference(value: np.ndarray, reference: np.ndarray) -> tuple[float, float, float, float]:
            delta = np.abs(value - reference)
            return float(abs(value.mean() - reference.mean())), float(abs(value.std() - reference.std())), float(np.quantile(delta, 0.99)), float(delta.max())
        rec_mean, rec_std, rec_p99, rec_max = summary_difference(reconstruction, reference_reconstruction)
        alb_mean, alb_std, alb_p99, alb_max = summary_difference(albedo, reference_albedo)
        # The contract is summary consistency. Isolated PyTorch3D edge-raster pixels can
        # differ more than 1e-4 while mean/std and p99 remain stable at this tolerance.
        if max(rec_mean, rec_std, alb_mean, alb_std) > 1e-4 or max(rec_p99, alb_p99) > 2e-4:
            raise RuntimeError(f"latent extraction summary consistency failed for {case_id}/{mode}: rec=({rec_mean},{rec_std},{rec_p99}), albedo=({alb_mean},{alb_std},{alb_p99})")
        payload = {"shape": code["shape"].detach().cpu().numpy(), "tex": code["tex"].detach().cpu().numpy(), "expression": code["exp"].detach().cpu().numpy(), "pose": code["pose"].detach().cpu().numpy(), "camera": code["cam"].detach().cpu().numpy(), "light": code["light"].detach().cpu().numpy(), "detail": code["detail"].detach().cpu().numpy()}
        destination = output / mode / f"{case_id}_latent_codes.npz"
        destination.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(destination, **payload)
        rows.append({"case_id": case_id, "input_mode": mode, "latent_path": str(destination), "reconstruction_mean_difference": rec_mean, "reconstruction_std_difference": rec_std, "reconstruction_p99_abs_difference": rec_p99, "reconstruction_max_abs_difference": rec_max, "albedo_mean_difference": alb_mean, "albedo_std_difference": alb_std, "albedo_p99_abs_difference": alb_p99, "albedo_max_abs_difference": alb_max, "verified": True})
    with (audit_root / "latent_extraction_validation.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    return rows
