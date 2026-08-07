from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pytest
import yaml

from skin_optics_so1.color_input import inverse_srgb
from skin_optics_so1.synthetic_config import load_config


def test_strict_config_rejects_unknown_field() -> None:
    cfg_path = Path("configs/so1_synthetic_smoke_v1.yaml")
    data = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    data["unexpected"] = True
    with tempfile.TemporaryDirectory() as d:
        path = Path(d) / "bad.yaml"
        path.write_text(yaml.safe_dump(data), encoding="utf-8")
        with pytest.raises(ValueError, match="Unknown config field"):
            load_config(path)


def test_smoke_config_counts_are_64() -> None:
    cfg = load_config("configs/so1_synthetic_smoke_v1.yaml")
    assert cfg.total_samples == 64
    assert cfg.canonical_hash() == load_config("configs/so1_synthetic_smoke_v1.yaml").canonical_hash()


def test_inverse_srgb_round_trip_with_so0_encoder() -> None:
    from skin_optics.numpy_backend.color_spaces import srgb_encode

    linear = np.linspace(0.0, 1.0, 4096, dtype=np.float64)
    srgb = srgb_encode(linear).astype(np.float32)
    decoded = inverse_srgb(srgb).astype(np.float64)
    assert np.max(np.abs(decoded - linear)) <= 1e-6

