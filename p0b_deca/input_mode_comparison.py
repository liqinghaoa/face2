"""Execute the fixed 12-case, two-input-mode DECA comparison."""
from __future__ import annotations

import csv
import json
import time
import traceback
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import yaml

from .config import P0BConfig
from .deca_runtime import forward_deca, image_to_tensor, load_deca
from .input_adapter import identity_mapping, mask_bbox_crop
from .metrics import coverage, finite_fraction, residual_p0
from .p0_mapper import map_normal_to_p0, map_to_p0
from .sh_lighting import DirectionalToSHProjector
from .types import PilotSample


def _read_rgb(path: Path) -> np.ndarray:
    value = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if value is None:
        raise FileNotFoundError(path)
    return cv2.cvtColor(value, cv2.COLOR_BGR2RGB)


def _read_mask(path: Path) -> np.ndarray:
    value = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if value is None:
        raise FileNotFoundError(path)
    return value


def _frozen_samples(config: P0BConfig) -> list[PilotSample]:
    """Read the already-selected internal mapping without requiring pandas."""
    mapping = config.output_root / "pilot_manifest/p0b_pilot12_id_mapping.csv"
    with mapping.open(newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != 12 or [row["audit_id"] for row in rows] != [f"P0B-{index:03d}" for index in range(1, 13)]:
        raise RuntimeError("frozen Pilot12 mapping is missing or changed")

    def asset(value: str) -> Path:
        path = (config.p0a_root / value.replace("\\", "/")).resolve()
        if config.p0a_root.resolve() not in path.parents or not path.is_file():
            raise FileNotFoundError(path)
        return path

    def number(value: str) -> float | None:
        try:
            return float(value)
        except ValueError:
            return None

    return [
        PilotSample(
            row["audit_id"],
            row["sample_id"],
            row["patient_group_id"],
            row["acquisition_group"],
            int(row["sex"]),
            number(row["brightness_value"]),
            number(row["iso"]),
            number(row["exposure_time"]),
            row["forehead_available"].lower() in {"true", "1"},
            asset(row["aligned_scene_path"]),
            asset(row["face_valid_mask_path"]),
            asset(row["skin_strict_mask_path"]),
            asset(row["physics_core_mask_path"]),
        )
        for row in rows
    ]


def _save_rgb(path: Path, value: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    display = np.rint(np.clip(value, 0.0, 1.0) * 255).astype(np.uint8)
    if not cv2.imwrite(str(path), cv2.cvtColor(display, cv2.COLOR_RGB2BGR)):
        raise OSError(f"failed to write {path}")


def _map_chw(value: Any, matrix: np.ndarray, interpolation: str) -> np.ndarray:
    array = value[0].detach().float().cpu().numpy().transpose(1, 2, 0)
    mapped = map_to_p0(array, matrix, interpolation)
    return mapped[..., None] if array.shape[-1] == 1 and mapped.ndim == 2 else mapped


def _renderer_coefficients(model: Any, config: P0BConfig) -> dict[str, np.ndarray]:
    raw = yaml.safe_load(config.relighting_config.read_text(encoding="utf-8"))
    projector = DirectionalToSHProjector(int(raw["sample_count"]))
    factor = model.render.constant_factor.detach().cpu().numpy()
    return {
        name: projector.fit_renderer(
            np.asarray(preset["direction"]),
            float(preset["ambient"]),
            float(preset["diffuse"]),
            factor,
        )[0]
        for name, preset in raw["presets"].items()
    }


def run_input_mode_comparison(config: P0BConfig) -> dict[str, Any]:
    """Run all cases; record and retain every case-level failure."""
    import torch

    samples = _frozen_samples(config)
    if len(samples) != 12 or tuple(config.input_modes) != ("direct_p0_aligned", "mask_bbox_crop"):
        raise RuntimeError("comparison requires the frozen 12 cases and two configured input modes")
    model, _ = load_deca(config.deca_root, "cuda", config.seed)
    coefficients = _renderer_coefficients(model, config)
    root = config.output_root / "input_mode_comparison"
    root.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []

    for sample in samples:
        scene_u8 = _read_rgb(sample.aligned_scene_path)
        scene = scene_u8.astype(np.float32) / 255.0
        face_mask = _read_mask(sample.face_valid_mask_path)
        physics_mask = _read_mask(sample.physics_core_mask_path)
        for mode in config.input_modes:
            started = time.perf_counter()
            case_dir = root / mode / sample.audit_id
            try:
                if mode == "direct_p0_aligned":
                    input_u8 = scene_u8
                    _, deca_to_p0 = identity_mapping()
                    bbox: tuple[int, int, int, int] | None = None
                elif mode == "mask_bbox_crop":
                    input_u8, bbox, _, deca_to_p0 = mask_bbox_crop(scene_u8, face_mask, config.mask_bbox_expand_ratio)
                else:
                    raise ValueError(f"unsupported input mode: {mode}")

                image = image_to_tensor(input_u8, "cuda")
                code, output, _ = forward_deca(model, image)
                rendered = model.render(
                    output["verts"].clone(),
                    output["trans_verts"].clone(),
                    output["albedo"],
                    code["light"],
                )
                reconstruction = _map_chw(rendered["images"], deca_to_p0, "bilinear")
                albedo = _map_chw(rendered["albedo_images"], deca_to_p0, "bilinear")
                shading = _map_chw(rendered["shading_images"], deca_to_p0, "bilinear")
                alpha = _map_chw(rendered["alpha_images"], deca_to_p0, "nearest")[..., 0]
                normal = map_normal_to_p0(_map_chw(rendered["normal_images"], deca_to_p0, "bilinear"))
                residual_signed, residual_abs = residual_p0(scene, reconstruction)

                case_dir.mkdir(parents=True, exist_ok=True)
                _save_rgb(case_dir / "input.png", input_u8.astype(np.float32) / 255.0)
                _save_rgb(case_dir / "reconstruction.png", reconstruction)
                _save_rgb(case_dir / "albedo_like.png", albedo)
                _save_rgb(case_dir / "shading_like.png", shading)
                _save_rgb(case_dir / "normal_coarse.png", (normal + 1.0) / 2.0)
                _save_rgb(case_dir / "residual_abs.png", residual_abs)
                cv2.imwrite(str(case_dir / "alpha_mask.png"), np.rint(np.clip(alpha, 0, 1) * 255).astype(np.uint8))
                np.savez_compressed(
                    case_dir / "physical_maps_float.npz",
                    reconstruction=reconstruction,
                    albedo_like=albedo,
                    shading_like=shading,
                    alpha=alpha,
                    normal_coarse=normal,
                    residual_signed=residual_signed,
                    residual_abs=residual_abs,
                )

                relight_finite = True
                for preset, coefficient in coefficients.items():
                    light = torch.from_numpy(coefficient).unsqueeze(0).to(image)
                    fixed = model.render(
                        output["verts"].clone(),
                        output["trans_verts"].clone(),
                        output["albedo"],
                        light,
                    )
                    fixed_p0 = _map_chw(fixed["images"], deca_to_p0, "bilinear")
                    relight_finite &= bool(np.isfinite(fixed_p0).all())
                    _save_rgb(case_dir / "relighting" / f"{preset}.png", fixed_p0)

                selected = physics_mask > 0
                rows.append(
                    {
                        "audit_id": sample.audit_id,
                        "input_mode": mode,
                        "status": "passed",
                        "bbox": json.dumps(bbox) if bbox else "",
                        "runtime_seconds": time.perf_counter() - started,
                        "alpha_coverage_physics_core": coverage(physics_mask, alpha > 0.5),
                        "reconstruction_mae_physics_core": float(residual_abs[selected].mean()),
                        "reconstruction_finite_fraction": finite_fraction(reconstruction),
                        "albedo_finite_fraction": finite_fraction(albedo),
                        "normal_finite_fraction": finite_fraction(normal),
                        "relighting_finite": relight_finite,
                        "output_dir": str(case_dir),
                    }
                )
            except Exception as exc:
                error = {
                    "audit_id": sample.audit_id,
                    "sample_id": sample.sample_id,
                    "input_mode": mode,
                    "exception_type": type(exc).__name__,
                    "exception_message": str(exc),
                    "traceback": traceback.format_exc(),
                }
                errors.append(error)
                rows.append(
                    {
                        "audit_id": sample.audit_id,
                        "input_mode": mode,
                        "status": "failed",
                        "runtime_seconds": time.perf_counter() - started,
                        "exception_type": type(exc).__name__,
                        "exception_message": str(exc),
                        "output_dir": str(case_dir),
                    }
                )
                torch.cuda.empty_cache()

    fields = sorted({key for row in rows for key in row})
    with (root / "p0b_b1_input_mode_case_metrics.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    with (root / "p0b_b1_case_errors.jsonl").open("w", encoding="utf-8") as handle:
        for error in errors:
            handle.write(json.dumps(error, ensure_ascii=False) + "\n")
    summary = {
        "status": "passed" if not errors and len(rows) == 24 else "failed_cases",
        "expected_cases": 24,
        "completed_cases": sum(row["status"] == "passed" for row in rows),
        "failed_cases": len(errors),
        "input_modes": list(config.input_modes),
        "fixed_input_mode": config.fixed_input_mode,
        "results_root": str(root),
        "metrics_path": str(root / "p0b_b1_input_mode_case_metrics.csv"),
        "errors_path": str(root / "p0b_b1_case_errors.jsonl"),
    }
    (root / "p0b_b1_input_mode_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary
