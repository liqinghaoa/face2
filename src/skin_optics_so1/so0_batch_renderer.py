"""SO-0 camera-path batch rendering adapter."""

from __future__ import annotations

import sys
import time
import csv
import os
from pathlib import Path
from typing import Any

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import torch


def ensure_so0_import_path() -> None:
    """Add the local frozen SO-0 source path when the package is not installed."""

    project_root = Path(__file__).resolve().parents[2]
    so0_src = project_root / "src" / "skin_optics_so0" / "src"
    if so0_src.exists() and str(so0_src) not in sys.path:
        sys.path.insert(0, str(so0_src))


def is_cuda_oom(exc: BaseException) -> bool:
    text = str(exc).lower()
    return isinstance(exc, torch.cuda.OutOfMemoryError) or ("cuda" in text and "out of memory" in text)


def load_frozen_camera_matrix(camera_name: str, light_name: str, so0_output_dir: str | Path = "outputs/SO0_Forward_Model_v1.1") -> np.ndarray:
    """Load the frozen SO-0 ColorChecker 3x3 matrix for one qualified pair."""

    path = Path(so0_output_dir) / "tables" / "colorchecker_calibration_metrics.csv"
    with path.open("r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            if row["camera_name"] == camera_name and row["illuminant_name"] == light_name:
                values = np.fromstring(row["matrix_3x3"], sep=" ", dtype=np.float64)
                if values.shape != (9,):
                    raise ValueError(f"Invalid frozen matrix field for {camera_name} / {light_name}")
                if row["qualification_status"] != "PASS":
                    raise ValueError(f"Refusing unqualified SO-0 pair matrix: {camera_name} / {light_name}")
                return values.reshape(3, 3)
    raise ValueError(f"No frozen SO-0 matrix found for {camera_name} / {light_name}")


class SO0CameraRenderer:
    """Render one camera-light pair via frozen SO-0 `render_camera`."""

    def __init__(self, camera_name: str, light_name: str, device: str = "cuda", dtype: torch.dtype = torch.float32) -> None:
        ensure_so0_import_path()
        from skin_optics.torch_backend.forward_model import SO0TorchForwardModel

        if torch.cuda.is_available():
            torch.backends.cuda.matmul.allow_tf32 = False
            torch.backends.cudnn.allow_tf32 = False
            torch.use_deterministic_algorithms(True)
        try:
            torch.set_float32_matmul_precision("highest")
        except Exception:
            pass
        self.camera_name = camera_name
        self.light_name = light_name
        self.device = torch.device(device)
        self.dtype = dtype
        matrix = load_frozen_camera_matrix(camera_name, light_name)
        self.model = SO0TorchForwardModel(camera_name=camera_name, illuminant_name=light_name, matrix_3x3=matrix, dtype=dtype)
        self.model.to(self.device)
        self.model.eval()

    def close(self) -> None:
        del self.model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def render_batch(
        self,
        m: np.ndarray,
        h: np.ndarray,
        s: np.ndarray,
        p: np.ndarray,
        exposure: float = 1.0,
    ) -> dict[str, np.ndarray]:
        if not (m.shape == h.shape == s.shape == p.shape):
            raise ValueError("M/H/S/P batch shapes must match")
        mt = torch.as_tensor(m, dtype=self.dtype, device=self.device)
        ht = torch.as_tensor(h, dtype=self.dtype, device=self.device)
        st = torch.as_tensor(s, dtype=self.dtype, device=self.device)
        pt = torch.as_tensor(p, dtype=self.dtype, device=self.device)
        et = torch.ones_like(mt) * float(exposure)
        with torch.inference_mode():
            out = self.model.render_camera(mt, ht, st, pt, et)
        return {k: v.detach().cpu().numpy().astype(np.float32, copy=False) for k, v in out.items()}


def render_with_oom_fallback(
    renderer: SO0CameraRenderer,
    fields: list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]],
    batch_sizes: list[int],
    exposure: float,
) -> tuple[list[dict[str, np.ndarray]], dict[str, Any]]:
    """Render fields with explicit CUDA-OOM batch-size fallback."""

    if not batch_sizes:
        raise ValueError("batch_sizes must be non-empty")
    active = int(batch_sizes[0])
    outputs: list[dict[str, np.ndarray]] = []
    idx = 0
    t0 = time.perf_counter()
    peak_mem = 0
    while idx < len(fields):
        batch = fields[idx : idx + active]
        arrays = [np.stack([item[i] for item in batch], axis=0) for i in range(4)]
        try:
            out = renderer.render_batch(*arrays, exposure=exposure)
            for b in range(len(batch)):
                outputs.append({k: v[b] for k, v in out.items()})
            idx += len(batch)
            if torch.cuda.is_available():
                peak_mem = max(peak_mem, int(torch.cuda.max_memory_allocated(renderer.device)))
        except RuntimeError as exc:
            if not is_cuda_oom(exc):
                raise
            smaller = [b for b in batch_sizes if b < active]
            if not smaller:
                raise
            active = int(smaller[0])
            torch.cuda.empty_cache()
    elapsed = time.perf_counter() - t0
    return outputs, {
        "actual_batch_size": active,
        "peak_memory_bytes": peak_mem,
        "total_seconds": elapsed,
        "seconds_per_sample": elapsed / max(1, len(fields)),
    }


def clipping_stats(srgb_unclipped_hwc: np.ndarray) -> dict[str, float | list[float]]:
    x = np.asarray(srgb_unclipped_hwc, dtype=np.float32)
    low = x < 0.0
    high = x > 1.0
    clipped = low | high
    per_channel = clipped.reshape(-1, 3).mean(axis=0)
    return {
        "low_clip_fraction": float(low.mean()),
        "high_clip_fraction": float(high.mean()),
        "total_clip_fraction": float(clipped.mean()),
        "per_channel_clip_fraction": [float(v) for v in per_channel],
    }
