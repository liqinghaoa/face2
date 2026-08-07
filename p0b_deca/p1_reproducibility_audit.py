"""Evidence-only Pilot12 cross-process DECA reproducibility audit.

This module never calls P1 ``run(..., full=True)`` and never writes beneath
``P1_DECA_Frozen500_v1/cases``.  It reuses the audited P0-B runtime exactly
and saves three independent-process observations under a separate audit tree.
"""
from __future__ import annotations

import csv
import hashlib
import json
import os
import platform
import subprocess
import sys
import traceback
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import yaml
from PIL import Image
try:
    from skimage.metrics import structural_similarity
except ImportError:  # pragma: no cover - optional in lightweight validation environments
    def structural_similarity(
        im1: np.ndarray,
        im2: np.ndarray,
        *,
        channel_axis: int | None = None,
        data_range: float | None = None,
    ) -> float:
        """Tiny local fallback that preserves importability without skimage."""

        a = np.asarray(im1, dtype=np.float64)
        b = np.asarray(im2, dtype=np.float64)
        if channel_axis is not None and a.ndim > 2:
            a = np.moveaxis(a, channel_axis, -1)
            b = np.moveaxis(b, channel_axis, -1)
            a = a.reshape(-1, a.shape[-1])
            b = b.reshape(-1, b.shape[-1])
        else:
            a = a.reshape(-1)
            b = b.reshape(-1)
        if a.size == 0 or b.size == 0:
            return float("nan")
        if data_range is None:
            data_range = float(np.max([a.max(), b.max()]) - np.min([a.min(), b.min()]))
        data_range = float(max(data_range, 1e-8))
        mu_a = float(a.mean())
        mu_b = float(b.mean())
        var_a = float(a.var())
        var_b = float(b.var())
        cov = float(((a - mu_a) * (b - mu_b)).mean())
        c1 = (0.01 * data_range) ** 2
        c2 = (0.03 * data_range) ** 2
        numerator = (2 * mu_a * mu_b + c1) * (2 * cov + c2)
        denominator = (mu_a * mu_a + mu_b * mu_b + c1) * (var_a + var_b + c2)
        return float(numerator / denominator) if denominator else float("nan")
try:
    import cv2
except ImportError:  # pragma: no cover - optional in lightweight validation environments
    from scipy import ndimage as _ndimage

    class _CV2Fallback:
        DIST_L2 = 2

        @staticmethod
        def erode(array: np.ndarray, kernel: np.ndarray, iterations: int = 1) -> np.ndarray:
            structure = np.asarray(kernel, dtype=bool)
            result = np.asarray(array, dtype=bool)
            for _ in range(int(iterations)):
                result = _ndimage.binary_erosion(result, structure=structure, border_value=0)
            return result.astype(array.dtype)

        @staticmethod
        def distanceTransform(array: np.ndarray, distance_type: int, mask_size: int) -> np.ndarray:
            return _ndimage.distance_transform_edt(np.asarray(array, dtype=bool)).astype(np.float32)

    cv2 = _CV2Fallback()

from .deca_runtime import forward_deca, image_to_tensor, load_deca, seed_everything
from .p1_generation import LATENT_KEYS, _coefficients, _environment, _read_any, _read_rgb, _chw, load_config, sha
from .metrics import residual_p0

RUN_NAMES = ("run_01", "run_02", "run_03")
LATENTS = ("shape_code", "tex_code", "detail_code", "exp_code", "pose_code", "cam_code", "light_code")
MAPS = ("albedo_like", "normal_coarse", "shading_like", "reconstruction", "alpha", "signed_residual")
ALPHA_THRESHOLD = 0.5  # Same fixed threshold used by P1 quality metrics.


def _hash_array(value: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def _array_record(value: np.ndarray, **extra: Any) -> dict[str, Any]:
    a = np.asarray(value)
    return {"shape": list(a.shape), "dtype": str(a.dtype), "min": float(a.min()), "max": float(a.max()), "mean": float(a.mean()), "std": float(a.std()), "array_sha256": _hash_array(a), **extra}


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({key for row in rows for key in row}) or ["case_id"]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader(); writer.writerows(rows)


def audit_root(root: Path, cfg: dict[str, Any]) -> Path:
    return root / cfg["output_root"] / "reproducibility_audit_v1"


def _pilot_rows(root: Path, cfg: dict[str, Any]) -> list[dict[str, Any]]:
    regression = root / cfg["output_root"] / "metadata/pilot12_regression.json"
    data = json.loads(regression.read_text(encoding="utf-8"))
    rows = data.get("cases", [])
    if len(rows) != 12 or len({x.get("case_id") for x in rows}) != 12:
        raise ValueError("reproducibility audit requires exactly the retained 12 Pilot12 cases")
    p0 = root / cfg["p0a_root"]
    master = {row["ID"]: row for row in csv.DictReader((p0 / "metadata/master_index.csv").open(newline="", encoding="utf-8-sig"))}
    result: list[dict[str, Any]] = []
    for item in rows:
        cid, aid = item["case_id"], item["p0b_audit_id"]
        source = master[cid]
        rel = lambda key: (p0 / source[key].replace("\\", "/")).resolve()
        result.append({"case_id": cid, "p0b_audit_id": aid, "input_path": rel("aligned_scene_relpath"), "input_sha256": sha(rel("aligned_scene_relpath")), "face_valid_path": rel("face_valid_mask_relpath"), "physics_core_skin_path": rel("physics_core_skin_mask_relpath"), "p0b_baseline_path": root / "data/processed/P0B_DECA_Pilot12_v1/input_mode_comparison/direct_p0_aligned" / aid, "p1_previous_cache_path": root / cfg["output_root"] / "cases" / cid, "fixed_input_mode": cfg["fixed_input_mode"]})
    return result


def validate(root: Path, cfg: dict[str, Any]) -> dict[str, Any]:
    rows = _pilot_rows(root, cfg)
    if cfg["fixed_input_mode"] != "direct_p0_aligned": raise ValueError("audit only supports direct_p0_aligned")
    for row in rows:
        for key in ("input_path", "face_valid_path", "physics_core_skin_path"):
            if not row[key].is_file(): raise FileNotFoundError(row[key])
        if not (row["p0b_baseline_path"] / "physical_maps_float.npz").is_file(): raise FileNotFoundError(row["p0b_baseline_path"])
    return {"status": "passed", "case_count": len(rows), "run_names": list(RUN_NAMES), "fixed_input_mode": cfg["fixed_input_mode"], "full_500_started": False}


def write_manifest(root: Path, cfg: dict[str, Any]) -> Path:
    rows = _pilot_rows(root, cfg)
    dest = root / cfg["output_root"] / "metadata/reproducibility_audit/pilot12_case_manifest.csv"
    _write_csv(dest, [{k: str(v) if isinstance(v, Path) else v for k, v in row.items()} for row in rows])
    return dest


def _run_environment(root: Path, cfg: dict[str, Any], model: Any) -> dict[str, Any]:
    import torch
    env = _environment(root, cfg)
    env.update({"cudnn_version": str(torch.backends.cudnn.version()), "model_training": bool(model.training), "all_parameters_requires_grad_false": bool(all(not p.requires_grad for p in model.parameters())), "cudnn_deterministic": bool(torch.backends.cudnn.deterministic), "cudnn_benchmark": bool(torch.backends.cudnn.benchmark), "cuda_matmul_allow_tf32": bool(torch.backends.cuda.matmul.allow_tf32), "cudnn_allow_tf32": bool(torch.backends.cudnn.allow_tf32), "deterministic_algorithms": bool(torch.are_deterministic_algorithms_enabled()), "random_seed": int(cfg["seed"]), "fixed_input_mode": cfg["fixed_input_mode"], "effective_config_sha256": sha(root / cfg["output_root"] / "metadata/effective_config.yaml")})
    return env


def worker(root: Path, cfg: dict[str, Any], run_name: str, output_root: Path | None = None) -> dict[str, Any]:
    """Execute one independent-process observation; invoked only by the CLI child."""
    if run_name not in RUN_NAMES + ("run_04",): raise ValueError(run_name)
    validate(root, cfg)
    out = output_root if output_root is not None else audit_root(root, cfg) / run_name
    if out.exists() and any(out.iterdir()): raise FileExistsError(f"audit run already exists: {out}")
    (out / "cases").mkdir(parents=True)
    seed_everything(int(cfg["seed"]))
    model, _ = load_deca(root / cfg["deca_root"], "cuda", int(cfg["seed"]))
    for p in model.parameters(): p.requires_grad_(False)
    model.eval()
    env = _run_environment(root, cfg, model)
    coeffs = _coefficients(model, root, cfg)
    outcomes: list[dict[str, Any]] = []
    for row in _pilot_rows(root, cfg):
        cid, case = row["case_id"], out / "cases" / row["case_id"]
        case.mkdir()
        try:
            rgb_u8 = _read_rgb(row["input_path"])
            rgb_f = rgb_u8.astype(np.float32) / 255.0
            tensor = image_to_tensor(rgb_u8, "cuda")
            # P0-B's retained direct input uses the same official adapter; save
            # both tensors to prove rather than assume this identity.
            p0b_u8 = _read_rgb(row["p0b_baseline_path"] / "input.png")
            p0b_tensor = image_to_tensor(p0b_u8, "cuda").detach().cpu().numpy().astype(np.float32)
            p1_tensor = tensor.detach().cpu().numpy().astype(np.float32)
            code, decoded, _ = forward_deca(model, tensor)
            rendered = model.render(decoded["verts"].clone(), decoded["trans_verts"].clone(), decoded["albedo"], code["light"])
            reconstruction, albedo, shading, normal = _chw(rendered["images"]), _chw(rendered["albedo_images"]), _chw(rendered["shading_images"]), _chw(rendered["normal_images"])
            alpha = _chw(rendered["alpha_images"], channels=False).astype(np.float32)
            signed, _ = residual_p0(rgb_f, reconstruction)
            relighted = []
            import torch
            for _, coefficient in coeffs:
                fixed = model.render(decoded["verts"].clone(), decoded["trans_verts"].clone(), decoded["albedo"], torch.from_numpy(coefficient).unsqueeze(0).to(decoded["verts"]))
                relighted.append(_chw(fixed["images"]))
            latent = {target: code[source].detach().cpu().float().numpy().astype(np.float32) for source, target in (("shape", "shape_code"), ("tex", "tex_code"), ("detail", "detail_code"), ("exp", "exp_code"), ("pose", "pose_code"), ("cam", "cam_code"), ("light", "light_code")) if source in code}
            np.save(case / "decoded_rgb_uint8.npy", rgb_u8); np.save(case / "decoded_rgb_float.npy", rgb_f); np.save(case / "deca_input_tensor.npy", p1_tensor); np.save(case / "p0b_entry_input_tensor.npy", p0b_tensor); np.save(case / "p1_entry_input_tensor.npy", p1_tensor)
            np.savez_compressed(case / "latents.npz", **latent); np.savez_compressed(case / "maps.npz", albedo_like=albedo, normal_coarse=normal, shading_like=shading, reconstruction=reconstruction, alpha=alpha, signed_residual=signed); np.savez_compressed(case / "relighting.npz", preset_names=np.asarray([x[0] for x in coeffs]), relighted_images=np.asarray(relighted, dtype=np.float32))
            diagnosis = {"decoded_rgb_uint8": _array_record(rgb_u8, file_sha256=sha(case / "decoded_rgb_uint8.npy")), "decoded_rgb_float": _array_record(rgb_f, file_sha256=sha(case / "decoded_rgb_float.npy")), "deca_input_tensor": _array_record(p1_tensor, device="cuda", rgb_order="RGB", transform="HWC->CHW; float32/255", range="[0,1]", resize=False, interpolation="none", file_sha256=sha(case / "deca_input_tensor.npy")), "p0b_entry_tensor": _array_record(p0b_tensor), "p0b_p1_tensor_equal": bool(np.array_equal(p0b_tensor, p1_tensor))}
            (case / "diagnostics.json").write_text(json.dumps(diagnosis, indent=2), encoding="utf-8")
            outcomes.append({"case_id": cid, "status": "success", "input_array_sha256": _hash_array(rgb_u8), "input_tensor_sha256": _hash_array(p1_tensor), "p0b_tensor_equal": bool(np.array_equal(p0b_tensor, p1_tensor))})
        except Exception as exc:
            (case / "failure.json").write_text(json.dumps({"exception_type": type(exc).__name__, "exception_message": str(exc), "traceback": traceback.format_exc()}, indent=2), encoding="utf-8")
            outcomes.append({"case_id": cid, "status": "failed", "exception_type": type(exc).__name__})
    _write_csv(out / "metadata" / "outcomes.csv", outcomes)
    (out / "metadata" / "environment.json").parent.mkdir(parents=True, exist_ok=True)
    (out / "metadata" / "environment.json").write_text(json.dumps(env, indent=2), encoding="utf-8")
    summary = {"run_name": run_name, "success_cases": sum(x["status"] == "success" for x in outcomes), "failed_cases": sum(x["status"] == "failed" for x in outcomes), "full_500_started": False}
    (out / "metadata" / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def _pair_metrics(a: np.ndarray, b: np.ndarray) -> dict[str, Any]:
    a, b = np.asarray(a), np.asarray(b)
    if a.shape != b.shape: return {"shape_equal": False, "dtype_equal": str(a.dtype) == str(b.dtype)}
    d = a.astype(np.float64) - b.astype(np.float64); ad = np.abs(d)
    denom = float(np.linalg.norm(b.ravel())) or 1.0
    cosine = float(np.dot(a.ravel().astype(np.float64), b.ravel().astype(np.float64)) / max(1e-12, np.linalg.norm(a.ravel()) * np.linalg.norm(b.ravel())))
    return {"shape_equal": True, "dtype_equal": str(a.dtype) == str(b.dtype), "max_abs": float(ad.max()), "MAE": float(ad.mean()), "RMSE": float(np.sqrt(np.mean(d*d))), "relative_L2": float(np.linalg.norm(d.ravel()) / denom), "cosine_similarity": cosine, "allclose_at_1e-5": bool(np.allclose(a, b, rtol=1e-5, atol=1e-5)), "allclose_at_1e-4": bool(np.allclose(a, b, rtol=1e-4, atol=1e-4))}


def _regions(alpha_a: np.ndarray, alpha_b: np.ndarray, face: np.ndarray, physics: np.ndarray) -> dict[str, np.ndarray]:
    intersection = (alpha_a > ALPHA_THRESHOLD) & (alpha_b > ALPHA_THRESHOLD)
    kernel = np.ones((3, 3), np.uint8)
    result = {"full_image": np.ones(intersection.shape, bool), "alpha_intersection": intersection, "face_valid": face > 0, "physics_core_skin": physics > 0}
    for size in (1, 3, 5): result[f"alpha_intersection_erode_{size}"] = cv2.erode(intersection.astype(np.uint8), kernel, iterations=size).astype(bool)
    return result


def _map_metrics(a: np.ndarray, b: np.ndarray, mask: np.ndarray, normal: bool = False) -> dict[str, Any]:
    aa, bb = np.asarray(a), np.asarray(b)
    values_a, values_b = aa[mask], bb[mask]
    if not len(values_a): return {"valid_pixel_count": 0}
    diff = values_a.astype(np.float64) - values_b.astype(np.float64); ad = np.abs(diff)
    data_range = max(float(np.max((aa, bb)) - np.min((aa, bb))), 1e-6)
    try: ssim = float(structural_similarity(aa, bb, channel_axis=-1 if aa.ndim == 3 else None, data_range=data_range))
    except Exception: ssim = float("nan")
    result = {"valid_pixel_count": int(mask.sum()), "MAE": float(ad.mean()), "RMSE": float(np.sqrt(np.mean(diff * diff))), "mean_abs": float(ad.mean()), "p50_abs": float(np.quantile(ad, .5)), "p95_abs": float(np.quantile(ad, .95)), "p99_abs": float(np.quantile(ad, .99)), "p99_9_abs": float(np.quantile(ad, .999)), "max_abs": float(ad.max()), "fraction_abs_gt_1_over_255": float((ad > 1/255).mean()), "fraction_abs_gt_2_over_255": float((ad > 2/255).mean()), "fraction_abs_gt_0_01": float((ad > .01).mean()), "SSIM": ssim, "PSNR": float(20*np.log10(data_range) - 10*np.log10(max(np.mean(diff*diff), 1e-20)))}
    if normal:
        va, vb = values_a.reshape(-1, 3), values_b.reshape(-1, 3)
        va /= np.maximum(np.linalg.norm(va, axis=1, keepdims=True), 1e-8); vb /= np.maximum(np.linalg.norm(vb, axis=1, keepdims=True), 1e-8)
        angle = np.degrees(np.arccos(np.clip((va * vb).sum(1), -1, 1)))
        result.update({"mean_angular_error_deg": float(angle.mean()), "p95_angular_error_deg": float(np.quantile(angle, .95)), "p99_angular_error_deg": float(np.quantile(angle, .99))})
    return result


def _load_run(audit: Path, run: str, cid: str) -> tuple[Any, Any, Any]:
    case = audit / run / "cases" / cid
    return np.load(case / "latents.npz"), np.load(case / "maps.npz"), np.load(case / "relighting.npz")


def _preview(value: np.ndarray, normal: bool = False) -> np.ndarray:
    a = np.asarray(value, np.float32)
    if a.ndim == 2: a = np.repeat(a[..., None], 3, axis=2)
    if normal: a = (a + 1.0) / 2.0
    return np.rint(np.clip(a, 0, 1) * 255).astype(np.uint8)


def _write_qc(path: Path, baseline: np.ndarray, current: np.ndarray, alpha: np.ndarray, physics: np.ndarray, normal: bool = False) -> None:
    """Fixed diagnostic panel: baseline, run, abs diff, alpha edge, core overlay."""
    diff = np.abs(np.asarray(baseline) - np.asarray(current))
    edge = (alpha > ALPHA_THRESHOLD) ^ cv2.erode((alpha > ALPHA_THRESHOLD).astype(np.uint8), np.ones((3,3),np.uint8)).astype(bool)
    overlay = _preview(current, normal); overlay[physics > 0] = (0, 255, 0); overlay[edge] = (255, 0, 0)
    panels = [_preview(baseline, normal), _preview(current, normal), _preview(diff / max(float(diff.max()), 1e-8)), np.repeat((edge * 255).astype(np.uint8)[..., None], 3, 2), overlay]
    Image.fromarray(np.concatenate(panels, axis=1)).save(path)


def compare(root: Path, cfg: dict[str, Any]) -> dict[str, Any]:
    audit = audit_root(root, cfg); rows = _pilot_rows(root, cfg); comparison = audit / "comparisons"; comparison.mkdir(parents=True, exist_ok=True)
    latent_rows: list[dict[str, Any]] = []; map_rows: list[dict[str, Any]] = []; relight_rows: list[dict[str, Any]] = []; bands: list[dict[str, Any]] = []; first: list[dict[str, Any]] = []
    run_pairs = [("run_01", "run_02"), ("run_01", "run_03"), ("run_02", "run_03")]
    for row in rows:
        cid, aid = row["case_id"], row["p0b_audit_id"]
        face, physics = _read_any(row["face_valid_path"]) > 0, _read_any(row["physics_core_skin_path"]) > 0
        loaded = {name: _load_run(audit, name, cid) for name in RUN_NAMES}
        base_latent = np.load(root / "data/processed/P0B_DECA_Pilot12_v1/noncollapse_audit_v1/extracted_latent_codes/direct_p0_aligned" / f"{aid}_latent_codes.npz")
        base_maps = np.load(row["p0b_baseline_path"] / "physical_maps_float.npz")
        pairs: list[tuple[str, str, Any, Any, Any, Any]] = [(a, b, loaded[a][0], loaded[b][0], loaded[a][1], loaded[b][1]) for a, b in run_pairs]
        pairs += [("p0b_baseline", name, base_latent, loaded[name][0], base_maps, loaded[name][1]) for name in RUN_NAMES]
        for left, right, la, lb, ma, mb in pairs:
            latent_map = {"shape_code": "shape", "tex_code": "tex", "detail_code": "detail", "exp_code": "expression", "pose_code": "pose", "cam_code": "camera", "light_code": "light"} if left == "p0b_baseline" else {key: key for key in LATENTS}
            for key in LATENTS:
                if latent_map[key] in la.files and key in lb.files: latent_rows.append({"case_id": cid, "p0b_audit_id": aid, "left": left, "right": right, "representation": key, **_pair_metrics(la[latent_map[key]], lb[key])})
            map_map = {"signed_residual": "residual_signed"} if left == "p0b_baseline" else {key: key for key in MAPS}
            for key in MAPS:
                if map_map.get(key, key) not in ma.files or key not in mb.files: continue
                regions = _regions(ma["alpha"], mb["alpha"], face, physics)
                for region, mask in regions.items(): map_rows.append({"case_id": cid, "p0b_audit_id": aid, "left": left, "right": right, "map": key, "region": region, **_map_metrics(ma[map_map.get(key, key)], mb[key], mask, normal=key == "normal_coarse")})
        # P0-B only retained 8-bit relighting; compare raw quantized values and
        # dequantized float explicitly, without turning any metric into a gate.
        r1 = loaded["run_01"][2]
        for index, preset in enumerate(r1["preset_names"].tolist()):
            legacy = _read_rgb(row["p0b_baseline_path"] / "relighting" / f"{preset}.png")
            q = np.rint(np.clip(r1["relighted_images"][index], 0, 1) * 255).astype(np.uint8)
            d = np.abs(q.astype(np.int16) - legacy.astype(np.int16))
            relight_rows.append({"case_id": cid, "p0b_audit_id": aid, "preset": str(preset), "comparison": "p0b_png_vs_run_01_quantized", "max_gray_level_difference": int(d.max()), "fraction_gt_1_level": float((d > 1).mean()), "fraction_gt_2_level": float((d > 2).mean()), "fraction_gt_3_level": float((d > 3).mean()), **_map_metrics(legacy.astype(np.float32)/255, q.astype(np.float32)/255, physics)})
        # Boundary-distance evidence uses baseline-vs-run_01 for normal and all presets.
        alpha = base_maps["alpha"] > ALPHA_THRESHOLD; boundary = alpha ^ cv2.erode(alpha.astype(np.uint8), np.ones((3,3),np.uint8)).astype(bool); distance = cv2.distanceTransform((~boundary).astype(np.uint8), cv2.DIST_L2, 3)
        targets: list[tuple[str, np.ndarray, np.ndarray]] = [("normal_coarse", base_maps["normal_coarse"], loaded["run_01"][1]["normal_coarse"])]
        for i, preset in enumerate(r1["preset_names"].tolist()): targets.append((f"relighting:{preset}", _read_rgb(row["p0b_baseline_path"] / "relighting" / f"{preset}.png").astype(np.float32)/255, r1["relighted_images"][i]))
        for name, a, b in targets:
            ad = np.abs(a-b).mean(axis=-1); labels = [("0-1", (distance <= 1)), ("2-3", (distance > 1)&(distance <= 3)), ("4-5", (distance > 3)&(distance <= 5)), ("6-10", (distance > 5)&(distance <= 10)), (">10", distance > 10), ("physics_core_skin", physics)]
            for label, mask in labels:
                values = ad[mask]
                bands.append({"case_id": cid, "map": name, "distance_band": label, "pixel_count": int(mask.sum()), "MAE": float(values.mean()) if len(values) else float("nan"), "p99_abs": float(np.quantile(values, .99)) if len(values) else float("nan"), "max_abs": float(values.max()) if len(values) else float("nan"), "fraction_abs_gt_1_over_255": float((values > 1/255).mean()) if len(values) else float("nan")})
        qc_dir = audit / "qc" / cid; qc_dir.mkdir(parents=True, exist_ok=True)
        _write_qc(qc_dir / "normal_coarse.png", base_maps["normal_coarse"], loaded["run_01"][1]["normal_coarse"], base_maps["alpha"], physics, normal=True)
        _write_qc(qc_dir / "albedo_like.png", base_maps["albedo_like"], loaded["run_01"][1]["albedo_like"], base_maps["alpha"], physics)
        largest = max(targets[1:], key=lambda x: float(np.abs(x[1] - x[2]).mean()))
        _write_qc(qc_dir / "largest_relighting_difference.png", largest[1], largest[2], base_maps["alpha"], physics)
        d0 = json.loads((audit / "run_01" / "cases" / cid / "diagnostics.json").read_text())
        stage = "source_file_bytes" if row["input_sha256"] != sha(row["input_path"]) else "decoded_rgb_uint8" if not d0["p0b_p1_tensor_equal"] else "encode_latents"
        first.append({"case_id": cid, "p0b_audit_id": aid, "first_divergence": stage, "systematic_first_divergence": stage == "encode_latents"})
    _write_csv(comparison / "latent_pairwise_metrics.csv", latent_rows); _write_csv(comparison / "map_pairwise_metrics.csv", map_rows); _write_csv(comparison / "relighting_pairwise_metrics.csv", relight_rows); _write_csv(comparison / "relighting_quantization_analysis.csv", relight_rows); _write_csv(comparison / "error_by_boundary_distance.csv", bands); _write_csv(comparison / "first_divergence_by_case.csv", first)
    summaries = [{"representation": key, "p0b_vs_p1_median_max_abs": float(np.median([x["max_abs"] for x in latent_rows if x["representation"] == key and x["left"] == "p0b_baseline"])), "within_p1_median_max_abs": float(np.median([x["max_abs"] for x in latent_rows if x["representation"] == key and x["left"].startswith("run_")]))} for key in LATENTS]
    _write_csv(comparison / "latent_case_summary.csv", summaries)
    (comparison / "latent_global_summary.json").write_text(json.dumps(summaries, indent=2), encoding="utf-8")
    return {"latent_rows": latent_rows, "map_rows": map_rows, "relight_rows": relight_rows, "first": first}


def finalize(root: Path, cfg: dict[str, Any]) -> dict[str, Any]:
    audit = audit_root(root, cfg); data = compare(root, cfg)
    latent = data["latent_rows"]; maps = data["map_rows"]; rel = data["relight_rows"]
    envelope: dict[str, Any] = {}
    for key in LATENTS:
        rows = [x for x in latent if x["left"].startswith("run_") and x["representation"] == key]
        envelope[key] = {metric: {"median": float(np.median([x[metric] for x in rows])), "p95": float(np.quantile([x[metric] for x in rows], .95)), "maximum": float(np.max([x[metric] for x in rows]))} for metric in ("max_abs", "MAE", "relative_L2")}
    (root / cfg["output_root"] / "metadata/reproducibility_audit").mkdir(parents=True, exist_ok=True)
    (root / cfg["output_root"] / "metadata/reproducibility_audit/empirical_variability_envelope.json").write_text(json.dumps(envelope, indent=2), encoding="utf-8")
    p0_latent = [x for x in latent if x["left"] == "p0b_baseline"]; internal = [x for x in latent if x["left"].startswith("run_")]
    stable = max(x["max_abs"] for x in internal) <= 1e-5
    p0_larger = np.median([x["max_abs"] for x in p0_latent]) > max(np.median([x["max_abs"] for x in internal]), 1e-12) * 2
    core_normals = [x for x in maps if x["left"] == "p0b_baseline" and x["map"] == "normal_coarse" and x["region"] == "physics_core_skin"]
    edge = list(csv.DictReader((audit / "comparisons/error_by_boundary_distance.csv").open()))
    edge_normal = [float(x["MAE"]) for x in edge if x["map"] == "normal_coarse" and x["distance_band"] in {"0-1", "2-3"}]
    core_normal = [float(x["MAE"]) for x in edge if x["map"] == "normal_coarse" and x["distance_band"] == "physics_core_skin"]
    core_stable = bool(core_normals) and max(x["MAE"] for x in core_normals) < .01
    edge_dominant = bool(edge_normal and core_normal) and np.median(edge_normal) > np.median(core_normal)
    same_order = np.median([x["max_abs"] for x in p0_latent]) <= max(np.median([x["max_abs"] for x in internal]), 1e-12) * 2
    if stable and p0_larger and core_stable and edge_dominant:
        decision = "HISTORICAL_BASELINE_VERSION_DRIFT"
    elif not stable and same_order and core_stable:
        decision = "CURRENT_PIPELINE_NONDETERMINISTIC"
    elif core_normals and max(x["MAE"] for x in core_normals) >= .01:
        decision = "CORE_OUTPUT_MISMATCH_KEEP_BLOCKED"
    else: decision = "INSUFFICIENT_EVIDENCE"
    result = {"decision": decision, "full_500_started": False, "formal_gate_modified": False, "historical_p0b_baseline_overwritten": False, "three_runs_successful": all(json.loads((audit / run / "metadata/summary.json").read_text())["success_cases"] == 12 for run in RUN_NAMES), "systematic_first_divergence": sorted({x["first_divergence"] for x in data["first"]}), "recommend_code_fix": decision == "FIX_INPUT_OR_PIPELINE", "recommend_versioned_cross_process_baseline": decision in {"CURRENT_PIPELINE_NONDETERMINISTIC", "HISTORICAL_BASELINE_VERSION_DRIFT"}, "allow_formal_gate_update": False, "allow_full_500": False}
    (root / cfg["output_root"] / "metadata/reproducibility_audit/reproducibility_decision.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result
