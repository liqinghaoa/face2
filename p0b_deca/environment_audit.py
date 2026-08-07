"""Executable P0-B0 audit of the local, licensed DECA core."""
from __future__ import annotations

import importlib.util
import platform
import subprocess
import sys
import traceback
from typing import Any, Callable

import numpy as np

from .asset_registry import audit_assets
from .config import P0BConfig
from .deca_runtime import build_deca_config, forward_deca, load_deca, seed_everything


def _detail(status: bool, exc: BaseException | None = None, **evidence: Any) -> dict[str, Any]:
    return {
        "status": bool(status),
        "exception_type": type(exc).__name__ if exc else None,
        "exception_message": str(exc) if exc else None,
        "traceback": traceback.format_exc() if exc else None,
        **evidence,
    }


def audit_environment(config: P0BConfig) -> dict[str, Any]:
    """Run checks now; no prior JSON/CSV status is read."""
    def version(name: str) -> str:
        try:
            return str(__import__(name).__version__)
        except Exception:
            return "not_available"

    torch_version = version("torch")
    cuda = False
    gpu = "not_available"
    cuda_runtime = "not_available"
    if torch_version != "not_available":
        import torch

        cuda = torch.cuda.is_available()
        gpu = torch.cuda.get_device_name(0) if cuda else "not_available"
        cuda_runtime = str(torch.version.cuda or "not_available")
    try:
        driver = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        driver = "not_available"

    deca_exists = config.deca_root.is_dir()
    commit = "not_available"
    dirty = False
    if deca_exists:
        try:
            commit = subprocess.check_output(["git", "-C", str(config.deca_root), "rev-parse", "HEAD"], text=True).strip()
            dirty = bool(subprocess.check_output(["git", "-C", str(config.deca_root), "status", "--porcelain"], text=True).strip())
        except Exception:
            commit = "not_git_or_unavailable"

    assets = audit_assets(config.deca_root)
    details: dict[str, dict[str, Any]] = {
        "torch_import": _detail(torch_version != "not_available", version=torch_version),
        "cuda_available": _detail(cuda, gpu_name=gpu),
        "pytorch3d_import": _detail(importlib.util.find_spec("pytorch3d") is not None),
        "deca_root_exists": _detail(deca_exists, path=str(config.deca_root)),
        "assets_present": _detail(bool(assets) and all(record.exists and record.size_bytes for record in assets)),
    }

    def run(name: str, operation: Callable[[], dict[str, Any] | None]) -> None:
        try:
            details[name] = _detail(True, **(operation() or {}))
        except Exception as exc:
            details[name] = _detail(False, exc)

    runtime: dict[str, Any] = {}

    def deca_import() -> dict[str, Any]:
        cfg = build_deca_config(config.deca_root)
        from decalib.deca import DECA

        runtime["cfg"] = cfg
        runtime["DECA"] = DECA
        return {"module": DECA.__module__}

    run("deca_import", deca_import)

    def texture_enabled() -> dict[str, Any]:
        cfg = runtime.get("cfg") or build_deca_config(config.deca_root)
        expected = str((config.deca_root / "data/FLAME_albedo_from_BFM.npz").resolve())
        actual = str(cfg.model.tex_path)
        if not (cfg.model.use_tex and cfg.model.tex_type == "BFM" and not cfg.model.extract_tex and actual == expected):
            raise RuntimeError("DECA BFM texture configuration is not explicitly enabled")
        if not config.deca_root.joinpath("data/FLAME_albedo_from_BFM.npz").is_file():
            raise FileNotFoundError(expected)
        return {"use_tex": True, "tex_type": "BFM", "extract_tex": False, "tex_path": actual}

    run("texture_enabled", texture_enabled)

    def rasterizer_test() -> dict[str, Any]:
        import torch
        from pytorch3d.ops import knn_points

        points = torch.tensor([[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]], device="cuda")
        result = knn_points(points, points, K=1)
        torch.cuda.synchronize()
        if not torch.isfinite(result.dists).all():
            raise RuntimeError("PyTorch3D CUDA KNN returned non-finite values")
        return {"backend": "pytorch3d", "device": str(points.device)}

    run("rasterizer_test", rasterizer_test)

    def model_forward() -> dict[str, Any]:
        import torch

        model, cfg = load_deca(config.deca_root, "cuda", config.seed)
        image = torch.full((1, 3, 224, 224), 0.5, dtype=torch.float32, device="cuda")
        code, output, _ = forward_deca(model, image)
        required = ("verts", "trans_verts", "rendered_images", "alpha_images", "normal_images", "albedo", "uv_detail_normals")
        for key in required:
            if key not in output or not torch.isfinite(output[key]).all():
                raise RuntimeError(f"missing or non-finite DECA output: {key}")
        runtime.update(model=model, cfg=cfg, image=image, code=code, output=output)
        return {"output_shapes": {key: list(output[key].shape) for key in required}}

    run("encode_decode_render", model_forward)

    def sh_light_render() -> dict[str, Any]:
        import torch
        import yaml

        if "model" not in runtime:
            model_forward()
        raw = yaml.safe_load(config.relighting_config.read_text(encoding="utf-8"))
        from .sh_lighting import DirectionalToSHProjector

        coefficient, _ = DirectionalToSHProjector(int(raw["sample_count"])).fit_renderer(
            np.asarray(raw["presets"]["neutral_front"]["direction"]),
            float(raw["presets"]["neutral_front"]["ambient"]),
            float(raw["presets"]["neutral_front"]["diffuse"]),
            runtime["model"].render.constant_factor.detach().cpu().numpy(),
        )
        normals = runtime["output"]["normal_images"]
        shading = runtime["model"].render.add_SHlight(normals, torch.from_numpy(coefficient).unsqueeze(0).to(normals))
        if not torch.isfinite(shading).all() or float(shading.abs().sum()) == 0:
            raise RuntimeError("DECA SH renderer returned invalid shading")
        return {"preset": "neutral_front", "finite": True}

    run("sh_light_render", sh_light_render)

    def deterministic() -> dict[str, Any]:
        import torch

        if "model" not in runtime:
            model_forward()
        seed_everything(config.seed)
        _, first, _ = forward_deca(runtime["model"], runtime["image"].clone())
        seed_everything(config.seed)
        _, second, _ = forward_deca(runtime["model"], runtime["image"].clone())
        keys = ("verts", "rendered_images", "albedo", "uv_detail_normals")
        max_abs = max(float((first[key] - second[key]).abs().max()) for key in keys)
        if not all(torch.allclose(first[key], second[key], rtol=1e-5, atol=1e-5) for key in keys):
            raise RuntimeError(f"repeat inference exceeded tolerance; max_abs_diff={max_abs}")
        return {"seed": config.seed, "eval": True, "inference_mode": True, "rtol": 1e-5, "atol": 1e-5, "max_abs_diff": max_abs}

    run("deterministic", deterministic)

    checks = {name: value["status"] for name, value in details.items()}
    failed_checks = [name for name, value in checks.items() if not value]
    if config.license_status != "confirmed_by_user":
        failed_checks.append("license_confirmed_by_user")
    passed = not failed_checks
    return {
        "passed": passed,
        "blocked": not passed,
        "block_reason": "; ".join(failed_checks) if failed_checks else "none",
        "python_version": sys.version,
        "platform": platform.platform(),
        "torch_version": torch_version,
        "torchvision_version": version("torchvision"),
        "pytorch3d_version": version("pytorch3d"),
        "cuda_runtime": cuda_runtime,
        "cuda_driver": driver,
        "gpu_name": gpu,
        "deca_commit": commit,
        "deca_dirty": dirty,
        "rasterizer_type": "pytorch3d" if checks.get("pytorch3d_import") else "unavailable",
        "license_status": config.license_status,
        "checks": checks,
        "check_details": details,
        "failed_checks": failed_checks,
        "assets": [
            {
                "name": record.name,
                "path": str(record.path),
                "exists": record.exists,
                "size_bytes": record.size_bytes,
                "sha256": record.sha256,
                "load_status": record.load_status,
                "required": record.required,
            }
            for record in assets
        ],
    }
