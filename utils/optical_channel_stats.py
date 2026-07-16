"""Streaming, train-fold-only statistics for P2-1 optical channels."""

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path
from typing import Iterable

import numpy as np

from utils.relative_optical_channels import CHANNEL_NAMES, build_relative_optical_channels


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


class StreamingChannelStats:
    def __init__(self, channels: int = 7) -> None:
        self.count = np.zeros(channels, dtype=np.int64)
        self.mean = np.zeros(channels, dtype=np.float64)
        self.m2 = np.zeros(channels, dtype=np.float64)

    def update(self, values: np.ndarray, valid_mask: np.ndarray) -> None:
        mask = np.asarray(valid_mask) > 0
        if mask.ndim == 3:
            mask = mask[0]
        if values.shape[0] != len(self.count) or values.shape[1:] != mask.shape:
            raise ValueError("channel/mask shape mismatch")
        for channel in range(values.shape[0]):
            sample = np.asarray(values[channel][mask], dtype=np.float64)
            if sample.size == 0:
                continue
            n_b = sample.size
            mean_b = float(sample.mean())
            m2_b = float(np.square(sample - mean_b).sum())
            n_a = int(self.count[channel])
            if n_a == 0:
                self.count[channel] = n_b
                self.mean[channel] = mean_b
                self.m2[channel] = m2_b
                continue
            delta = mean_b - self.mean[channel]
            total = n_a + n_b
            self.mean[channel] += delta * n_b / total
            self.m2[channel] += m2_b + delta * delta * n_a * n_b / total
            self.count[channel] = total

    def finalize(self) -> tuple[np.ndarray, np.ndarray]:
        if np.any(self.count < 2):
            raise ValueError(f"insufficient valid pixels: {self.count.tolist()}")
        variance = self.m2 / self.count
        return self.mean.astype(np.float64), np.sqrt(variance).astype(np.float64)


def compute_shared_stats(samples: Iterable[tuple[np.ndarray, np.ndarray]], epsilon: float) -> dict:
    state = StreamingChannelStats(7)
    image_count = 0
    for rgb, mask in samples:
        channels = build_relative_optical_channels(rgb, mask, epsilon=epsilon)
        state.update(np.asarray(channels), np.asarray(mask))
        image_count += 1
    mean, std = state.finalize()
    return {
        "channel_names": list(CHANNEL_NAMES),
        "mean": mean.tolist(),
        "std": std.tolist(),
        "valid_pixel_counts": state.count.tolist(),
        "roi_image_count": image_count,
        "epsilon": float(epsilon),
        "lab_conversion": "opencv_float32_COLOR_RGB2LAB_signed_ab",
    }


def save_fold_stats(path: Path, stats: dict, fold: int, train_csv: Path) -> None:
    payload = dict(stats)
    payload.update({"fold": int(fold), "train_split_sha256": sha256_file(train_csv)})
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=Path(__file__).resolve().parents[1], text=True
        ).strip()
    except Exception:
        commit = "unavailable"
    payload["code_version"] = commit
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
