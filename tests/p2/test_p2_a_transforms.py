from __future__ import annotations

import numpy as np
import torch

from p2_counterfactual.transforms import flip_preset_name, transform_paired_rgb, transform_single_rgb


def test_flip_preset_name_swaps_left_right_only() -> None:
    assert flip_preset_name("left") == "right"
    assert flip_preset_name("right") == "left"
    assert flip_preset_name("neutral_front") == "neutral_front"
    assert flip_preset_name(None) is None


def test_paired_horizontal_flip_uses_single_decision() -> None:
    original = np.zeros((4, 4, 3), dtype=np.float32)
    counter = np.zeros((4, 4, 3), dtype=np.float32)
    original[:, 0, 0] = 1.0
    counter[:, 0, 1] = 1.0
    out_o, out_c, flipped = transform_paired_rgb(original, counter, training=True, image_size=4, force_flip=True)
    assert flipped is True
    assert out_o.shape == (3, 4, 4)
    assert out_c.shape == (3, 4, 4)
    assert torch.isfinite(out_o).all()
    assert torch.isfinite(out_c).all()


def test_validation_transform_has_no_random_augmentation() -> None:
    image = np.full((4, 4, 3), 0.5, dtype=np.float32)
    a, flip_a = transform_single_rgb(image, training=False, image_size=4, color_jitter_enabled=True, force_flip=True, force_color_jitter=True)
    b, flip_b = transform_single_rgb(image, training=False, image_size=4, color_jitter_enabled=True)
    assert flip_a is False
    assert flip_b is False
    assert torch.allclose(a, b)
