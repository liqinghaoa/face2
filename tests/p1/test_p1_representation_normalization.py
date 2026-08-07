from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
import torch

from utils.p1_representation_normalization import (
    apply_component_normalization,
    fit_component_normalization,
    inspect_normal_flip_convention,
)


def _frame(tmp: Path) -> pd.DataFrame:
    Image.fromarray(np.full((2, 2, 3), 128, dtype=np.uint8)).save(tmp / "rgb.png")
    Image.fromarray(np.array([[255, 0], [255, 255]], dtype=np.uint8)).save(tmp / "mask.png")
    np.savez(
        tmp / "maps.npz",
        albedo_like=np.full((2, 2, 3), 1.2, np.float32),
        normal_coarse=np.full((2, 2, 3), -0.5, np.float32),
        shading_like=np.array([[[0.0, 1.0, 2.0]] * 2] * 2, dtype=np.float32),
        signed_residual=np.array([[[0.1, -0.2, 0.3]] * 2] * 2, dtype=np.float32),
    )
    np.savez(tmp / "latents.npz", light_code=np.arange(27, dtype=np.float32).reshape(1, 9, 3))
    return pd.DataFrame(
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


def test_normal_flip_status_reports_unverified_evidence():
    audit = inspect_normal_flip_convention()
    assert audit.status == "UNVERIFIED"
    assert audit.reason


def test_p1_n_uses_safe_flip_disabled_fallback_when_unverified():
    with tempfile.TemporaryDirectory() as td:
        frame = _frame(Path(td))
        state = fit_component_normalization(frame, "p1_n")
        assert state.normal_flip_status == "SAFE_FLIP_DISABLED"
        assert state.normal_coordinate_status == "UNVERIFIED"
        assert state.normal_flip_mode == "DISABLED_SAFE_FALLBACK"
        sample = {
            "case_id": "c1",
            "patient_group_id": "g1",
            "fold": 0,
            "label_original": 3,
            "label_3class": 2,
            "label_binary": 1,
            "representation": np.array(
                [
                    [[-1.0, -0.5], [0.25, 1.0]],
                    [[-0.25, -0.1], [0.1, 0.4]],
                    [[-0.9, -0.7], [0.5, 0.9]],
                ],
                dtype=np.float32,
            ),
            "valid_mask": np.ones((1, 2, 2), dtype=np.float32),
        }
        no_flip = apply_component_normalization(sample, state, horizontal_flip=False)
        requested_flip = apply_component_normalization(sample, state, horizontal_flip=True)
        assert torch.allclose(no_flip["representation"], requested_flip["representation"])


def test_p1_l_normalization_standardizes_flattened_light_code():
    with tempfile.TemporaryDirectory() as td:
        frame = _frame(Path(td))
        state = fit_component_normalization(frame, "p1_l")
        sample = {
            "case_id": "c1",
            "patient_group_id": "g1",
            "fold": 0,
            "label_original": 3,
            "label_3class": 2,
            "label_binary": 1,
            "representation": np.arange(27, dtype=np.float32).reshape(9, 3),
            "valid_mask": None,
        }
        normalized = apply_component_normalization(sample, state)
        assert tuple(normalized["representation"].shape) == (27,)


def test_p1_r_background_is_masked_back_to_zero_after_mapping():
    with tempfile.TemporaryDirectory() as td:
        frame = _frame(Path(td))
        state = fit_component_normalization(frame, "p1_r")
        sample = {
            "case_id": "c1",
            "patient_group_id": "g1",
            "fold": 0,
            "label_original": 3,
            "label_3class": 2,
            "label_binary": 1,
            "representation": np.array([[[0.1, -0.2], [0.3, -0.4]]] * 3, dtype=np.float32),
            "valid_mask": np.array([[[1, 0], [1, 1]]], dtype=np.float32),
        }
        normalized = apply_component_normalization(sample, state)
        assert float(normalized["representation"][:, 0, 1].abs().sum()) == 0.0
