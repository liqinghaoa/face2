from __future__ import annotations

import tempfile
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
from PIL import Image

from datasets.p1_component_dataset import P1ComponentDataset


def _make_fixture(tmp: Path) -> pd.DataFrame:
    (tmp / "rgb.png").write_bytes(b"")
    rgb = np.full((2, 2, 3), 128, dtype=np.uint8)
    Image.fromarray(rgb).save(tmp / "rgb.png")
    mask = np.array([[255, 0], [255, 255]], dtype=np.uint8)
    Image.fromarray(mask).save(tmp / "mask.png")
    np.savez(
        tmp / "maps.npz",
        albedo_like=np.full((2, 2, 3), 0.5, np.float32),
        normal_coarse=np.full((2, 2, 3), -0.5, np.float32),
        shading_like=np.full((2, 2, 3), 1.5, np.float32),
        signed_residual=np.full((2, 2, 3), -0.1, np.float32),
    )
    np.savez(tmp / "latents.npz", light_code=np.arange(27, dtype=np.float32).reshape(1, 9, 3))
    frame = pd.DataFrame(
        [
            {
                "case_id": "c1",
                "patient_group_id": "g1",
                "group_id": "g1",
                "fold": 0,
                "label_original": 3,
                "label_3class": 2,
                "label_binary": 1,
                "rgb_path": str(tmp / "rgb.png"),
                "face_valid_mask_path": str(tmp / "mask.png"),
                "physics_core_skin_mask_path": str(tmp / "mask.png"),
                "maps_path": str(tmp / "maps.npz"),
                "latents_path": str(tmp / "latents.npz"),
            }
        ]
    )
    return frame


def test_component_dataset_lazily_loads_only_requested_latents():
    with tempfile.TemporaryDirectory() as td:
        frame = _make_fixture(Path(td))
        dataset = P1ComponentDataset(frame, "p1_l")
        with patch("numpy.load", wraps=np.load) as mocked:
            item = dataset[0]
            assert mocked.call_count == 1
        assert tuple(item["representation"].shape) == (9, 3)
        assert item["valid_mask"] is None


def test_component_dataset_returns_masked_image_components():
    with tempfile.TemporaryDirectory() as td:
        frame = _make_fixture(Path(td))
        dataset = P1ComponentDataset(frame, "p1_a")
        item = dataset[0]
        assert tuple(item["representation"].shape) == (3, 2, 2)
        assert tuple(item["valid_mask"].shape) == (1, 2, 2)

