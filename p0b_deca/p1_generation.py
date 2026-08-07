"""P1 frozen DECA asset cache generation using the verified P0-B runtime.

This module is deliberately label-blind.  It reads only P0-A's finalized
aligned scene and mask asset references, never its clinical/group columns.
``shading_like`` is the renderer's un-clipped SH shading image for the encoded
light; it is not a measured illumination map.  ``specular_like`` is not
produced by this frontend and is never synthesized from a residual.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import platform
import shutil
import subprocess
import sys
import time
import traceback
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np
import yaml
try:
    import cv2
except ImportError:  # validate-only must remain usable outside the CUDA P0-B environment
    cv2 = None
from PIL import Image

from .deca_runtime import forward_deca, image_to_tensor, load_deca
from .metrics import residual_p0
from .sh_lighting import DirectionalToSHProjector

BOUNDARY_IDS = ("A001917272-1", "A002081031-1")
LATENT_KEYS = {"shape": "shape_code", "exp": "expression_code", "pose": "pose_code", "cam": "camera_code", "light": "light_code", "tex": "tex_code", "detail": "detail_code"}
MAP_KEYS = ("input_aligned_rgb", "reconstruction", "alpha", "albedo_like", "normal_coarse", "shading_like", "signed_residual", "absolute_residual")
CASE_FILES = ("latents.npz", "maps.npz", "relighting.npz", "quality.json", "provenance.json", "preview.png")


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _json_safe(value: Any) -> Any:
    if isinstance(value, np.generic): return value.item()
    if isinstance(value, np.ndarray): return value.tolist()
    if isinstance(value, dict): return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)): return [_json_safe(v) for v in value]
    return value


def load_config(path: Path, root: Path) -> dict[str, Any]:
    cfg = yaml.safe_load(path.read_text(encoding="utf-8"))
    if cfg.get("fixed_input_mode") != "direct_p0_aligned":
        raise ValueError("P1 requires fixed_input_mode=direct_p0_aligned")
    cfg["project_root"] = str(root.resolve())
    return cfg


def _asset(root: Path, p0_root: Path, relpath: str, expected_shape: tuple[int, ...]) -> Path:
    path = (p0_root / relpath.replace("\\", "/")).resolve()
    if p0_root.resolve() not in path.parents or not path.is_file():
        raise FileNotFoundError(f"missing P0 asset: {relpath}")
    image = _read_any(path)
    if image is None or tuple(image.shape) != expected_shape:
        raise ValueError(f"invalid P0 asset shape {path}: {None if image is None else image.shape}")
    return path


def _mask(root: Path, p0_root: Path, relpath: str) -> Path:
    path = (p0_root / relpath.replace("\\", "/")).resolve()
    if p0_root.resolve() not in path.parents or not path.is_file():
        raise FileNotFoundError(f"missing P0 asset: {relpath}")
    value = _read_any(path)
    if tuple(value.shape) != (224, 224):
        raise ValueError(f"invalid P0 asset shape {path}: {value.shape}")
    if value.dtype != np.uint8 or not np.isin(value, (0, 1, 255)).all():
        raise ValueError(f"invalid binary mask format: {path}")
    return path


def _read_any(path: Path) -> np.ndarray:
    """Read an image without changing it; Pillow supports validate-only fallback."""
    if cv2 is not None:
        value = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if value is None: raise FileNotFoundError(path)
        return value
    with Image.open(path) as image:
        return np.asarray(image)


def _read_rgb(path: Path) -> np.ndarray:
    if cv2 is not None:
        value = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if value is None: raise FileNotFoundError(path)
        return cv2.cvtColor(value, cv2.COLOR_BGR2RGB)
    with Image.open(path) as image:
        return np.asarray(image.convert("RGB"))


def _write_rgb(path: Path, value: np.ndarray) -> None:
    image = np.rint(np.clip(value, 0, 1) * 255).astype(np.uint8)
    if cv2 is not None:
        if not cv2.imwrite(str(path), cv2.cvtColor(image, cv2.COLOR_RGB2BGR)): raise OSError(path)
    else:
        Image.fromarray(image, mode="RGB").save(path)


def source_records(root: Path, cfg: dict[str, Any]) -> list[dict[str, Any]]:
    """Build the only permitted 500-case source list from P0-A master_index."""
    p0_root = root / cfg["p0a_root"]
    master = p0_root / "metadata/master_index.csv"
    with master.open(newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    required = {"ID", "aligned_scene_relpath", "final_face_mask_relpath", "face_valid_mask_relpath", "skin_strict_mask_relpath", "physics_core_skin_mask_relpath", "p0_usable", "boundary_ambiguity_status"}
    if not rows or required - set(rows[0]): raise ValueError(f"master index missing columns: {sorted(required - set(rows[0]))}")
    ids = [row["ID"] for row in rows]
    if len(rows) != 500 or len(set(ids)) != 500: raise ValueError("master_index.csv must contain exactly 500 unique IDs")
    def validate_row(row: dict[str, str]) -> dict[str, Any]:
        # Intentionally copy only asset and P0 quality fields; no diagnosis/class/EXIF field crosses this boundary.
        scene = _asset(root, p0_root, row["aligned_scene_relpath"], (224, 224, 3))
        masks = {
            "final_face": _mask(root, p0_root, row["final_face_mask_relpath"]),
            "face_valid": _mask(root, p0_root, row["face_valid_mask_relpath"]),
            "strict_skin": _mask(root, p0_root, row["skin_strict_mask_relpath"]),
            "physics_core_skin": _mask(root, p0_root, row["physics_core_skin_mask_relpath"]),
        }
        return {"case_id": row["ID"], "input_path": scene, "input_sha256": sha(scene), "mask_paths": masks,
                "p0_usable": str(row["p0_usable"]).lower() in {"1", "true"},
                "boundary_ambiguity_status": row["boundary_ambiguity_status"]}
    # Validation is deliberately parallel only across independent P0 reads; it
    # does not alter case order, inputs, arrays, or DECA execution order.
    with ThreadPoolExecutor(max_workers=16) as pool:
        records = list(pool.map(validate_row, rows))
    if not set(BOUNDARY_IDS).issubset(ids): raise ValueError("prespecified jaw-neck boundary cases missing from master")
    return records


def preflight(root: Path, cfg: dict[str, Any]) -> dict[str, Any]:
    records = source_records(root, cfg)
    by_id = {r["case_id"]: r for r in records}
    boundary = {cid: {"present": cid in by_id, "p0_usable": by_id[cid]["p0_usable"], "boundary_ambiguity_status": by_id[cid]["boundary_ambiguity_status"]} for cid in BOUNDARY_IDS}
    if not all(x["p0_usable"] for x in boundary.values()): raise ValueError("boundary cases must remain P0 usable")
    return {"status": "passed", "expected_cases": len(records), "unique_ids": len(by_id), "fixed_input_mode": cfg["fixed_input_mode"], "boundary_cases": boundary}


def _model_hashes(root: Path, cfg: dict[str, Any]) -> dict[str, str]:
    data = root / cfg["deca_root"] / "data"
    # These are the assets explicitly wired by ``build_deca_config``.  Keeping
    # them all in the frozen record prevents a resume after a silent asset swap.
    names = {
        "deca_checkpoint": "deca_model.tar",
        "flame": "generic_model.pkl",
        "bfm_texture": "FLAME_albedo_from_BFM.npz",
        "topology": "head_template.obj",
        "dense_template": "texture_data_256.npy",
        "fixed_displacement": "fixed_displacement_256.npy",
        "landmark_embedding": "landmark_embedding.npy",
        "uv_face_mask": "uv_face_mask.png",
        "uv_face_eye_mask": "uv_face_eye_mask.png",
        "mean_texture": "mean_texture.jpg",
        "relighting_config": cfg["relighting_config"],
    }
    result: dict[str, str] = {}
    for key, name in names.items():
        path = root / name if key == "relighting_config" else data / name
        result[key] = sha(path) if path.is_file() else "unavailable"
    return result


def _environment(root: Path, cfg: dict[str, Any]) -> dict[str, Any]:
    def version(name: str) -> str:
        try: return str(__import__(name).__version__)
        except Exception: return "not_available"
    torch_version, cuda, gpu = version("torch"), "not_available", "not_available"
    if torch_version != "not_available":
        import torch
        cuda = str(torch.version.cuda or "not_available")
        gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "not_available"
    try: commit = subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"], text=True).strip()
    except Exception: commit = "unavailable"
    return {"python_version": sys.version, "platform": platform.platform(), "torch_version": torch_version, "torchvision_version": version("torchvision"), "cuda_version": cuda, "pytorch3d_version": version("pytorch3d"), "gpu": gpu, "git_commit": commit, "random_seed": int(cfg["seed"]), "fixed_input_mode": cfg["fixed_input_mode"], "asset_hashes": _model_hashes(root, cfg)}


def _write_metadata(root: Path, cfg: dict[str, Any], records: list[dict[str, Any]]) -> tuple[Path, str, dict[str, str]]:
    out = root / cfg["output_root"]; meta = out / "metadata"; meta.mkdir(parents=True, exist_ok=True)
    effective = dict(cfg)
    effective["shading_like_definition"] = "DECA renderer add_SHlight(normal_images, encoded_light), stored float32 without clipping; not a measured illumination field."
    effective["specular_like_status"] = "unavailable_by_current_frontend"
    effective["map_storage_ranges"] = {"input_aligned_rgb": "[0,1] float32", "reconstruction/albedo_like/shading_like": "raw renderer float32, not clipped or globally normalized", "normal_coarse": "renderer normal float32", "alpha": "renderer alpha float32", "signed_residual": "input_aligned_rgb - reconstruction", "absolute_residual": "abs(signed_residual)"}
    config_path = meta / "effective_config.yaml"; rendered = yaml.safe_dump(effective, sort_keys=True, allow_unicode=True)
    if config_path.exists() and config_path.read_text(encoding="utf-8") != rendered:
        # Production/reproducibility control blocks were added after the first
        # frozen Pilot12.  They do not participate in DECA inference; preserve
        # the original effective file/hash when every inference field agrees.
        existing = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        current = yaml.safe_load(rendered)
        for value in (existing, current):
            value.pop("reproducibility_audit", None)
            value.pop("pilot_regression", None)
        if existing != current: raise RuntimeError("refusing to replace a different effective P1 configuration")
    if not config_path.exists(): config_path.write_text(rendered, encoding="utf-8")
    env = _environment(root, cfg)
    if not (meta / "environment.json").exists(): (meta / "environment.json").write_text(json.dumps(env, indent=2), encoding="utf-8")
    hashes = env["asset_hashes"]
    if not (meta / "asset_hashes.json").exists(): (meta / "asset_hashes.json").write_text(json.dumps(hashes, indent=2), encoding="utf-8")
    return out, sha(config_path), hashes


def _array_info(value: Any) -> dict[str, Any]:
    array = value.detach().cpu().numpy() if hasattr(value, "detach") else np.asarray(value)
    return {"shape": list(array.shape), "dtype": str(array.dtype), "min": float(np.nanmin(array)), "max": float(np.nanmax(array)), "finite": bool(np.isfinite(array).all())}


def audit_output_inventory(model: Any, code: dict[str, Any], decoded: dict[str, Any], rendered: dict[str, Any], path: Path) -> dict[str, Any]:
    """Persist actual encode/decode/render keys after a real model call, never inferred keys."""
    semantic = {**{source: target for source, target in LATENT_KEYS.items()}, "render:images": "reconstruction", "render:alpha_images": "alpha", "render:albedo_images": "albedo_like", "render:normal_images": "normal_coarse", "render:shading_images": "shading_like"}
    saved = set(LATENT_KEYS) | {"render:images", "render:alpha_images", "render:albedo_images", "render:normal_images", "render:shading_images"}
    component_status = {
        "shape_code": "available" if "shape" in code else "unavailable_by_current_frontend",
        "expression_code": "available" if "exp" in code else "unavailable_by_current_frontend",
        "pose_code": "available" if "pose" in code else "unavailable_by_current_frontend",
        "camera_code": "available" if "cam" in code else "unavailable_by_current_frontend",
        "light_code": "available" if "light" in code else "unavailable_by_current_frontend",
        "tex_code": "available" if "tex" in code else "unavailable_by_current_frontend",
        "detail_code": "available" if "detail" in code else "unavailable_by_current_frontend",
        "albedo_like": "available" if "albedo_images" in rendered else "unavailable_by_current_frontend",
        "normal_coarse": "available" if "normal_images" in rendered else "unavailable_by_current_frontend",
        "reconstruction": "available" if "images" in rendered else "unavailable_by_current_frontend",
        "alpha": "available" if "alpha_images" in rendered else "unavailable_by_current_frontend",
        "shading_like": "available_from_renderer_shading_images" if "shading_images" in rendered else "unavailable_by_current_frontend",
        "specular_like": "unavailable_by_current_frontend",
    }
    inventory = {"encode": {k: _array_info(v) for k, v in code.items() if hasattr(v, "shape")}, "decode": {k: _array_info(v) for k, v in decoded.items() if hasattr(v, "shape")}, "render": {k: _array_info(v) for k, v in rendered.items() if hasattr(v, "shape")}, "semantic_mapping": semantic, "saved_as_p1_asset": sorted(saved), "component_status": component_status, "specular_like_status": "unavailable_by_current_frontend", "shading_like_status": "available_from_renderer_shading_images"}
    path.write_text(json.dumps(inventory, indent=2), encoding="utf-8")
    return inventory


def _coefficients(model: Any, root: Path, cfg: dict[str, Any]) -> list[tuple[str, np.ndarray]]:
    raw = yaml.safe_load((root / cfg["relighting_config"]).read_text(encoding="utf-8"))
    projector = DirectionalToSHProjector(int(raw["sample_count"]))
    factor = model.render.constant_factor.detach().cpu().numpy()
    return [(name, projector.fit_renderer(np.asarray(v["direction"]), float(v["ambient"]), float(v["diffuse"]), factor)[0]) for name, v in raw["presets"].items()]


def _chw(tensor: Any, channels: bool = True) -> np.ndarray:
    value = tensor[0].detach().float().cpu().numpy().transpose(1, 2, 0).astype(np.float32)
    return value if channels else value[..., 0]


def _quality(scene: np.ndarray, reconstruction: np.ndarray, alpha: np.ndarray, albedo: np.ndarray, normal: np.ndarray, residual: np.ndarray, face: np.ndarray, physics: np.ndarray, relighted: list[np.ndarray]) -> dict[str, Any]:
    selected = lambda mask: residual[mask] if mask.any() else residual
    relight_metrics = {name: {"finite": bool(np.isfinite(image).all()), "min": float(image.min()), "max": float(image.max()), "mean": float(image.mean()), "std": float(image.std()), "black_fraction": float((image <= 0).mean()), "saturated_fraction": float((image >= 1).mean())} for name, image in relighted}
    values = [reconstruction, alpha, albedo, normal, residual] + [x[1] for x in relighted]
    return {"input_min": float(scene.min()), "input_max": float(scene.max()), "input_mean": float(scene.mean()), "input_std": float(scene.std()), "input_finite": bool(np.isfinite(scene).all()), "reconstruction_mae_full": float(np.abs(residual).mean()), "reconstruction_rmse_full": float(np.sqrt(np.mean(residual ** 2))), "reconstruction_mae_face": float(np.abs(selected(face)).mean()), "reconstruction_mae_physics_core": float(np.abs(selected(physics)).mean()), "alpha_coverage": float((alpha > .5).mean()), "albedo_mean": float(albedo.mean()), "albedo_std": float(albedo.std()), "normal_norm_mean": float(np.linalg.norm(normal, axis=-1).mean()), "normal_norm_std": float(np.linalg.norm(normal, axis=-1).std()), "residual_mean": float(residual.mean()), "residual_p95": float(np.quantile(np.abs(residual), .95)), "all_outputs_finite": bool(all(np.isfinite(x).all() for x in values)), "relighting": relight_metrics, "relighting_complete": len(relighted) == 6 and all(x["finite"] for x in relight_metrics.values()), "encode_success": True, "decode_success": True, "render_success": True, "texture_success": True, "relighting_success": len(relighted) == 6 and all(x["finite"] for x in relight_metrics.values())}


def _success_valid(dest: Path, record: dict[str, Any], config_sha: str, checkpoint_sha: str) -> bool:
    success = dest / "_SUCCESS.json"
    if not success.is_file(): return False
    try:
        data = json.loads(success.read_text(encoding="utf-8"))
        return data.get("input_sha256") == record["input_sha256"] and data.get("effective_config_sha256") == config_sha and data.get("deca_checkpoint_sha256") == checkpoint_sha and data.get("validation_passed") is True and all((dest / name).is_file() and sha(dest / name) == digest for name, digest in data["output_file_sha256"].items())
    except Exception: return False


def freeze_deca(model: Any) -> None:
    """Freeze every DECA parameter before any encode/decode call."""
    for parameter in model.parameters(): parameter.requires_grad_(False)
    if any(parameter.requires_grad for parameter in model.parameters()): raise RuntimeError("DECA parameter freezing failed")


def _atomic_commit(tmp: Path, dest: Path) -> None:
    if dest.exists(): raise FileExistsError(f"refusing to overwrite existing output: {dest}")
    os.replace(tmp, dest)


def _write_case(model: Any, record: dict[str, Any], root: Path, cfg: dict[str, Any], out: Path, config_sha: str, checkpoint_sha: str, coefficients: list[tuple[str, np.ndarray]], inventory_path: Path | None) -> dict[str, Any]:
    import torch
    cid, dest = record["case_id"], out / "cases" / record["case_id"]
    if dest.exists(): raise FileExistsError(f"refusing to overwrite existing output: {dest}")
    tmp = out / "cases" / f".tmp_{cid}_{uuid.uuid4().hex}"; tmp.mkdir(parents=True)
    try:
        scene = _read_rgb(record["input_path"]).astype(np.float32) / 255.0
        code, decoded, _ = forward_deca(model, image_to_tensor((scene * 255).astype(np.uint8), "cuda"))
        rendered = model.render(decoded["verts"].clone(), decoded["trans_verts"].clone(), decoded["albedo"], code["light"])
        if inventory_path is not None: audit_output_inventory(model, code, decoded, rendered, inventory_path)
        latent = {target: code[source].detach().float().cpu().numpy().astype(np.float32) for source, target in LATENT_KEYS.items() if source in code}
        if set(latent) != set(LATENT_KEYS.values()): raise RuntimeError(f"missing actual DECA latent keys: {sorted(set(LATENT_KEYS.values()) - set(latent))}")
        reconstruction, albedo, shading, normal = _chw(rendered["images"]), _chw(rendered["albedo_images"]), _chw(rendered["shading_images"]), _chw(rendered["normal_images"])
        alpha = _chw(rendered["alpha_images"], channels=False)
        signed, absolute = residual_p0(scene, reconstruction)
        relighted: list[tuple[str, np.ndarray]] = []
        for name, coefficient in coefficients:
            fixed = model.render(decoded["verts"].clone(), decoded["trans_verts"].clone(), decoded["albedo"], torch.from_numpy(coefficient).unsqueeze(0).to(decoded["verts"]))
            relighted.append((name, _chw(fixed["images"])))
        face = _read_any(record["mask_paths"]["final_face"]) > 0
        physics = _read_any(record["mask_paths"]["physics_core_skin"]) > 0
        arrays = list(latent.values()) + [scene, reconstruction, alpha, albedo, shading, normal, signed, absolute] + [x[1] for x in relighted]
        if not all(np.isfinite(value).all() for value in arrays): raise RuntimeError("non-finite P1 output")
        np.savez_compressed(tmp / "latents.npz", **latent)
        np.savez_compressed(tmp / "maps.npz", input_aligned_rgb=scene.astype(np.float32), reconstruction=reconstruction, alpha=alpha.astype(np.float32), albedo_like=albedo, normal_coarse=normal, shading_like=shading, signed_residual=signed, absolute_residual=absolute)
        np.savez_compressed(tmp / "relighting.npz", preset_names=np.asarray([x[0] for x in relighted]), sh_coefficients=np.asarray([x[1] for x in coefficients], dtype=np.float32), relighted_images=np.asarray([x[1] for x in relighted], dtype=np.float32), preset_quality_json=np.asarray(json.dumps(_quality(scene, reconstruction, alpha, albedo, normal, signed, face, physics, relighted)["relighting"])))
        quality = _quality(scene, reconstruction, alpha, albedo, normal, signed, face, physics, relighted)
        quality["validation_passed"] = bool(quality["all_outputs_finite"] and quality["relighting_complete"])
        (tmp / "quality.json").write_text(json.dumps(quality, indent=2), encoding="utf-8")
        p0_root = (root / cfg["p0a_root"]).resolve()
        provenance = {"case_id": cid, "fixed_input_mode": "direct_p0_aligned", "input_path_relative_to_p0": str(record["input_path"].resolve().relative_to(p0_root)), "input_sha256": record["input_sha256"], "mask_references": {name: {"path_relative_to_p0": str(path.resolve().relative_to(p0_root)), "sha256": sha(path)} for name, path in record["mask_paths"].items()}, "deca_key_to_p1_key": LATENT_KEYS, "shading_like_definition": "renderer shading_images from encoded SH light, float32 unclipped", "specular_like_status": "unavailable_by_current_frontend", "p0_boundary_ambiguity": "jaw_neck_ambiguous" if cid in BOUNDARY_IDS else record["boundary_ambiguity_status"], "p0_usable": record["p0_usable"]}
        (tmp / "provenance.json").write_text(json.dumps(provenance, indent=2), encoding="utf-8")
        _write_rgb(tmp / "preview.png", reconstruction)
        if not quality["validation_passed"]: raise RuntimeError("case quality validation failed")
        output_hashes = {name: sha(tmp / name) for name in CASE_FILES}
        success = {"ID": cid, "input_sha256": record["input_sha256"], "effective_config_sha256": config_sha, "deca_checkpoint_sha256": checkpoint_sha, "output_file_sha256": output_hashes, "validation_passed": True, "finished_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
        (tmp / "_SUCCESS.json").write_text(json.dumps(success, indent=2), encoding="utf-8")
        _atomic_commit(tmp, dest)
        return {"case_id": cid, "status": "success", "case_dir": str(dest), **quality}
    except Exception:
        shutil.rmtree(tmp, ignore_errors=True)
        raise


def _failure(out: Path, record: dict[str, Any], exc: BaseException, config_sha: str, checkpoint_sha: str) -> dict[str, Any]:
    entry = {"case_id": record["case_id"], "failure_stage": "case_generation", "exception_type": type(exc).__name__, "exception_message": str(exc), "traceback": traceback.format_exc(), "input_path": str(record["input_path"]), "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "config_hash": config_sha, "checkpoint_hash": checkpoint_sha}
    path = out / "failures" / f"{record['case_id']}.json"; path.write_text(json.dumps(entry, indent=2), encoding="utf-8")
    return {"case_id": record["case_id"], "status": "failed", "failure_file": str(path), "exception_type": type(exc).__name__}


def _write_manifests(out: Path, records: list[dict[str, Any]], outcomes: dict[str, dict[str, Any]]) -> None:
    manifests = out / "manifests"; manifests.mkdir(exist_ok=True)
    inputs = [{"case_id": r["case_id"], "input_sha256": r["input_sha256"], "fixed_input_mode": "direct_p0_aligned", "p0_usable": r["p0_usable"], "boundary_ambiguity_status": r["boundary_ambiguity_status"]} for r in records]
    def write(name: str, rows: list[dict[str, Any]]) -> None:
        fields = sorted({key for row in rows for key in row}) or ["case_id"]
        with (manifests / name).open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader(); writer.writerows(rows)
    write("p1_deca_input_manifest.csv", inputs)
    write("p1_deca_output_manifest.csv", [outcomes.get(r["case_id"], {"case_id": r["case_id"], "status": "not_processed"}) for r in records])
    write("p1_deca_quality_manifest.csv", [{k: v for k, v in item.items() if k in {"case_id", "status", "all_outputs_finite", "relighting_complete", "reconstruction_mae_full", "alpha_coverage"}} for item in outcomes.values()])
    write("p1_deca_failure_manifest.csv", [item for item in outcomes.values() if item["status"] == "failed"])


def _write_qc(out: Path, outcomes: dict[str, dict[str, Any]], seed: int) -> dict[str, str]:
    """Make fixed label-free input/reconstruction QC tiles from successful cached cases."""
    successful = [x for x in outcomes.values() if x.get("status") == "success"]
    if not successful: return {}
    rng = np.random.default_rng(seed)
    normal = [x for x in successful if x["case_id"] not in BOUNDARY_IDS]
    selection: dict[str, dict[str, Any]] = {
        "random_normal": normal[int(rng.integers(len(normal)))] if normal else successful[0],
        "highest_reconstruction_error": max(successful, key=lambda x: float(x.get("reconstruction_mae_full", -np.inf))),
        "lowest_alpha_coverage": min(successful, key=lambda x: float(x.get("alpha_coverage", np.inf))),
        "highest_relighting_saturation": max(successful, key=lambda x: max((float(v.get("saturated_fraction", 0.0)) for v in x.get("relighting", {}).values()), default=0.0)),
    }
    for cid in BOUNDARY_IDS:
        match = next((x for x in successful if x["case_id"] == cid), None)
        if match is not None: selection[f"boundary_{cid}"] = match
    qc = out / "qc"; qc.mkdir(exist_ok=True); written: dict[str, str] = {}
    for role, item in selection.items():
        with np.load(Path(item["case_dir"]) / "maps.npz") as maps:
            left = np.rint(np.clip(maps["input_aligned_rgb"], 0, 1) * 255).astype(np.uint8)
            right = np.rint(np.clip(maps["reconstruction"], 0, 1) * 255).astype(np.uint8)
        canvas = np.concatenate((left, right), axis=1)
        path = qc / f"{role}.png"; Image.fromarray(canvas, mode="RGB").save(path)
        written[role] = str(path.relative_to(out))
    (qc / "qc_selection.json").write_text(json.dumps({role: {"case_id": item["case_id"], "preview": written[role]} for role, item in selection.items()}, indent=2), encoding="utf-8")
    return written


def _relative_error(current: np.ndarray, baseline: np.ndarray) -> dict[str, Any]:
    if current.shape != baseline.shape: return {"passed": False, "reason": "shape_mismatch", "current_shape": list(current.shape), "baseline_shape": list(baseline.shape)}
    diff = np.abs(current.astype(np.float64) - baseline.astype(np.float64))
    return {"passed": bool(np.allclose(current, baseline, rtol=1e-4, atol=1e-4)), "max_abs": float(diff.max()), "mae": float(diff.mean()), "rtol": 1e-4, "atol": 1e-4}


def _pilot_latent_error(current: np.ndarray, baseline: np.ndarray) -> dict[str, Any]:
    """Use P0-B's repeat-inference tolerance for retained code tensors."""
    if current.shape != baseline.shape:
        return {"passed": False, "reason": "shape_mismatch", "current_shape": list(current.shape), "baseline_shape": list(baseline.shape)}
    diff = np.abs(current.astype(np.float64) - baseline.astype(np.float64))
    return {"passed": bool(np.allclose(current, baseline, rtol=1e-5, atol=1e-5)), "max_abs": float(diff.max()), "mae": float(diff.mean()), "rtol": 1e-5, "atol": 1e-5}


def _pilot_map_error(current: np.ndarray, baseline: np.ndarray) -> dict[str, Any]:
    """Apply P0-B2's documented raster-summary tolerance.

    P0-B2 intentionally permits isolated PyTorch3D silhouette-raster pixels:
    absolute mean and standard-deviation differences must be <=1e-4 and the
    99th percentile of absolute pixel differences must be <=2e-4.  A maximum
    is reported for diagnosis but is not a pass/fail criterion.
    """
    if current.shape != baseline.shape:
        return {"passed": False, "reason": "shape_mismatch", "current_shape": list(current.shape), "baseline_shape": list(baseline.shape)}
    diff = np.abs(current.astype(np.float64) - baseline.astype(np.float64))
    mean_difference = float(abs(current.mean() - baseline.mean()))
    std_difference = float(abs(current.std() - baseline.std()))
    p99 = float(np.quantile(diff, 0.99))
    return {"passed": bool(mean_difference <= 1e-4 and std_difference <= 1e-4 and p99 <= 2e-4), "mean_difference": mean_difference, "std_difference": std_difference, "p99_abs_difference": p99, "max_abs": float(diff.max()), "mae": float(diff.mean()), "mean_std_atol": 1e-4, "p99_atol": 2e-4}


def _pilot_relighting_error(current: np.ndarray, legacy_png: Path) -> dict[str, Any]:
    """Compare P1 float output to P0-B's retained display PNG.

    P0-B kept relighting as an 8-bit PNG rather than a second float archive.
    The comparison therefore uses the same [0,1] clipping/rounding used by the
    P0-B writer and permits one 8-bit level for renderer/device drift.
    """
    baseline = _read_rgb(legacy_png).astype(np.float32) / 255.0
    quantized = np.rint(np.clip(current, 0, 1) * 255.0).astype(np.uint8).astype(np.float32) / 255.0
    diff = np.abs(quantized.astype(np.float64) - baseline.astype(np.float64))
    atol = 1.0 / 255.0 + 1e-7
    return {"passed": bool(np.allclose(quantized, baseline, rtol=0.0, atol=atol)), "max_abs": float(diff.max()), "mae": float(diff.mean()), "rtol": 0.0, "atol": atol, "baseline": "p0b_display_png_quantized", "quantization_atol": 1.0 / 255.0}


def _pilot12_regression(root: Path, out: Path, config_sha: str, checkpoint_sha: str) -> dict[str, Any]:
    """Compare the new entrypoint with retained P0-B assets without inventing missing baselines."""
    mapping = root / "data/processed/P0B_DECA_Pilot12_v1/pilot_manifest/p0b_pilot12_id_mapping.csv"
    with mapping.open(newline="", encoding="utf-8-sig") as handle: selected = list(csv.DictReader(handle))
    results: list[dict[str, Any]] = []
    required = ("tex_code", "shape_code", "detail_code", "albedo_like", "normal_coarse", "relighting_outputs")
    for row in selected:
        cid, audit_id = row["sample_id"], row["audit_id"]
        new_case = out / "cases" / cid
        old_case = root / "data/processed/P0B_DECA_Pilot12_v1/input_mode_comparison/direct_p0_aligned" / audit_id
        item: dict[str, Any] = {"case_id": cid, "p0b_audit_id": audit_id, "fixed_input_mode": "direct_p0_aligned", "comparisons": {}}
        if not new_case.is_dir() or not old_case.is_dir():
            item["comparisons"] = {name: {"passed": False, "reason": "missing_case_output"} for name in required}
        else:
            old_latent = root / "data/processed/P0B_DECA_Pilot12_v1/noncollapse_audit_v1/extracted_latent_codes/direct_p0_aligned" / f"{audit_id}_latent_codes.npz"
            with np.load(new_case / "latents.npz") as new_latent, np.load(new_case / "maps.npz") as new_maps, np.load(new_case / "relighting.npz") as new_relight, np.load(old_case / "physical_maps_float.npz") as old_maps:
                if not old_latent.is_file():
                    for key in ("tex_code", "shape_code", "detail_code"):
                        item["comparisons"][key] = {"passed": False, "reason": "missing_p0b2_latent_baseline"}
                else:
                    with np.load(old_latent) as old_codes:
                        for new_key, old_key in (("tex_code", "tex"), ("shape_code", "shape"), ("detail_code", "detail")):
                            item["comparisons"][new_key] = _pilot_latent_error(new_latent[new_key], old_codes[old_key]) if old_key in old_codes.files else {"passed": False, "reason": "missing_baseline_key"}
                for key in ("albedo_like", "normal_coarse"):
                    item["comparisons"][key] = _pilot_map_error(new_maps[key], old_maps[key]) if key in old_maps.files else {"passed": False, "reason": "missing_baseline_key"}
                per_preset: dict[str, Any] = {}
                for index, preset in enumerate(new_relight["preset_names"].tolist()):
                    name = str(preset)
                    old = old_case / "relighting" / f"{name}.png"
                    per_preset[name] = _pilot_relighting_error(new_relight["relighted_images"][index], old) if old.is_file() else {"passed": False, "reason": "missing_baseline_png"}
                item["comparisons"]["relighting_outputs"] = {"passed": len(per_preset) == 6 and all(x.get("passed") for x in per_preset.values()), "per_preset": per_preset}
        item["passed"] = all(bool(v.get("passed")) for v in item["comparisons"].values())
        results.append(item)
    summary = {"status": "passed" if len(results) == 12 and all(x["passed"] for x in results) else "blocked", "case_count": len(results), "success_cases": sum(x["passed"] for x in results), "fixed_input_mode": "direct_p0_aligned", "effective_config_sha256": config_sha, "deca_checkpoint_sha256": checkpoint_sha, "cases": results}
    (out / "metadata/pilot12_regression.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def _require_pilot12_gate(out: Path, config_sha: str, checkpoint_sha: str) -> None:
    path = out / "metadata/pilot12_regression.json"
    if not path.is_file(): raise RuntimeError("full run requires a completed P0-B Pilot12 regression")
    gate = json.loads(path.read_text(encoding="utf-8"))
    if gate.get("status") != "passed" or gate.get("case_count") != 12 or gate.get("success_cases") != 12 or gate.get("effective_config_sha256") != config_sha or gate.get("deca_checkpoint_sha256") != checkpoint_sha:
        raise RuntimeError("P0-B Pilot12 regression gate did not pass; full run is forbidden")


def run(root: Path, cfg: dict[str, Any], *, sample_ids: list[str] | None = None, pilot12: bool = False, full: bool = False, resume: bool = False, retry_failures: bool = False, overwrite_output_only: bool = False, validate_only: bool = False, production_contract: bool = False) -> dict[str, Any]:
    """Generate frozen assets; a full run is rejected until the P0-B regression gate passes."""
    validation = preflight(root, cfg)
    if validate_only: return validation
    if int(bool(full)) + int(bool(pilot12)) + int(bool(sample_ids)) != 1: raise ValueError("select exactly one of --full, --pilot12, or --sample-ids")
    records = source_records(root, cfg); existing_out = root / cfg["output_root"]
    if existing_out.exists() and not (resume or retry_failures or overwrite_output_only) and any(existing_out.iterdir()): raise FileExistsError("P1 output root already exists; use --resume after hash validation")
    out, config_sha, hashes = _write_metadata(root, cfg, records)
    for name in ("cases", "failures", "qc", "logs"): (out / name).mkdir(parents=True, exist_ok=True)
    selected = [r for r in records if not sample_ids or r["case_id"] in set(sample_ids)]
    if sample_ids:
        requested = set(sample_ids)
        available = {r["case_id"] for r in records}
        missing = sorted(requested - available)
        if missing:
            raise ValueError(f"sample IDs are not in the sole P0 master index: {missing}")
    if pilot12:
        mapping = root / "data/processed/P0B_DECA_Pilot12_v1/pilot_manifest/p0b_pilot12_id_mapping.csv"
        with mapping.open(newline="", encoding="utf-8-sig") as handle: ids = [row["sample_id"] for row in csv.DictReader(handle)]
        selected = [r for r in records if r["case_id"] in set(ids)]
        if len(selected) != 12: raise RuntimeError("frozen P0-B Pilot12 IDs unavailable in master index")
    if full: selected = records
    if full:
        if production_contract:
            from .p1_production_gate import frozen_valid
            if not frozen_valid(root, cfg): raise RuntimeError("frozen P0B_cross_process_production_gate_v2_1 is required for production full run")
        else:
            _require_pilot12_gate(out, config_sha, hashes["deca_checkpoint"])
    model, _ = load_deca(root / cfg["deca_root"], "cuda", int(cfg["seed"]))
    freeze_deca(model)
    coefficients = _coefficients(model, root, cfg); checkpoint_sha = hashes["deca_checkpoint"]
    outcomes: dict[str, dict[str, Any]] = {}; inventory = out / "metadata/deca_output_inventory.json"
    for record in selected:
        dest = out / "cases" / record["case_id"]
        if resume and _success_valid(dest, record, config_sha, checkpoint_sha): outcomes[record["case_id"]] = {"case_id": record["case_id"], "status": "skipped_valid_resume"}; continue
        if dest.exists() and not overwrite_output_only: outcomes[record["case_id"]] = _failure(out, record, FileExistsError("existing output is not a valid resumable case"), config_sha, checkpoint_sha); continue
        if (out / "failures" / f"{record['case_id']}.json").exists() and not retry_failures:
            outcomes[record["case_id"]] = {"case_id": record["case_id"], "status": "failed", "prior_failure": True, "failure_file": str(out / "failures" / f"{record['case_id']}.json")}
            continue
        try:
            outcomes[record["case_id"]] = _write_case(model, record, root, cfg, out, config_sha, checkpoint_sha, coefficients, inventory if not inventory.exists() else None)
            if outcomes[record["case_id"]]["status"] == "success":
                (out / "failures" / f"{record['case_id']}.json").unlink(missing_ok=True)
        except Exception as exc: outcomes[record["case_id"]] = _failure(out, record, exc, config_sha, checkpoint_sha)
    _write_manifests(out, records if full else selected, outcomes)
    regression = _pilot12_regression(root, out, config_sha, checkpoint_sha) if pilot12 else None
    qc = _write_qc(out, outcomes, int(cfg["seed"])) if full else {}
    resumptions = sum(x["status"] == "skipped_valid_resume" for x in outcomes.values())
    summary = {"expected_cases": len(records) if full else len(selected), "processed_cases": len(outcomes), "success_cases": sum(x["status"] == "success" for x in outcomes.values()) + resumptions, "new_success_cases": sum(x["status"] == "success" for x in outcomes.values()), "failed_cases": sum(x["status"] == "failed" for x in outcomes.values()), "skipped_by_valid_resume": resumptions, "all_finite_cases": sum(bool(x.get("all_outputs_finite")) for x in outcomes.values()), "relighting_complete_cases": sum(bool(x.get("relighting_complete")) for x in outcomes.values()), "boundary_ambiguity_cases": sum(r["case_id"] in BOUNDARY_IDS for r in selected), "fixed_input_mode": cfg["fixed_input_mode"]}
    if regression is not None: summary["pilot12_regression"] = {"status": regression["status"], "success_cases": regression["success_cases"]}
    if qc: summary["qc_previews"] = qc
    (out / "metadata/run_manifest.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary
