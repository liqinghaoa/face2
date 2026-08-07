from __future__ import annotations

import random
from typing import Any

import numpy as np
import torch
from PIL import Image
from torchvision.transforms import ColorJitter
from torchvision.transforms import functional as TF

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

PRESET_FLIP_MAP = {
    "neutral_front": "neutral_front",
    "left": "right",
    "right": "left",
    "top": "top",
    "dim_front": "dim_front",
    "bright_front": "bright_front",
}


def flip_preset_name(preset_name: str | None) -> str | None:
    if preset_name is None:
        return None
    return PRESET_FLIP_MAP.get(str(preset_name), str(preset_name))


def _as_float_hwc(image: Image.Image | np.ndarray | torch.Tensor) -> np.ndarray:
    if isinstance(image, Image.Image):
        arr = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
    elif torch.is_tensor(image):
        tensor = image.detach().cpu().float()
        if tensor.ndim != 3:
            raise ValueError(f"tensor image must be 3D, got {tuple(tensor.shape)}")
        if tensor.shape[0] == 3:
            arr = tensor.permute(1, 2, 0).numpy()
        elif tensor.shape[-1] == 3:
            arr = tensor.numpy()
        else:
            raise ValueError(f"tensor image must have 3 RGB channels, got {tuple(tensor.shape)}")
    else:
        arr = np.asarray(image, dtype=np.float32)
    if arr.ndim != 3 or arr.shape[-1] != 3:
        raise ValueError(f"RGB image must be HWC with 3 channels, got {arr.shape}")
    if arr.dtype.kind in "ui" or float(np.nanmax(arr)) > 2.0:
        arr = arr.astype(np.float32) / 255.0
    return np.clip(arr.astype(np.float32), 0.0, 1.0)


def rgb_to_tensor(image: Image.Image | np.ndarray | torch.Tensor, image_size: int = 224) -> torch.Tensor:
    arr = _as_float_hwc(image)
    tensor = torch.from_numpy(np.ascontiguousarray(arr)).permute(2, 0, 1).float()
    if tuple(tensor.shape[-2:]) != (int(image_size), int(image_size)):
        tensor = TF.resize(tensor, [int(image_size), int(image_size)], antialias=True)
    return tensor


def normalize_rgb_tensor(
    tensor: torch.Tensor,
    mean: tuple[float, float, float] | list[float] = IMAGENET_MEAN,
    std: tuple[float, float, float] | list[float] = IMAGENET_STD,
) -> torch.Tensor:
    return TF.normalize(tensor, mean=list(mean), std=list(std))


def build_color_jitter(config: dict[str, Any] | None = None) -> ColorJitter:
    cfg = dict(config or {})
    return ColorJitter(
        brightness=float(cfg.get("brightness", 0.30)),
        contrast=float(cfg.get("contrast", 0.30)),
        saturation=float(cfg.get("saturation", 0.20)),
        hue=float(cfg.get("hue", 0.05)),
    )


def transform_single_rgb(
    image: Image.Image | np.ndarray | torch.Tensor,
    *,
    training: bool,
    image_size: int = 224,
    horizontal_flip_probability: float = 0.5,
    color_jitter_enabled: bool = False,
    color_jitter_probability: float = 0.5,
    color_jitter_config: dict[str, Any] | None = None,
    mean: tuple[float, float, float] | list[float] = IMAGENET_MEAN,
    std: tuple[float, float, float] | list[float] = IMAGENET_STD,
    force_flip: bool | None = None,
    force_color_jitter: bool | None = None,
) -> tuple[torch.Tensor, bool]:
    tensor = rgb_to_tensor(image, image_size=image_size)
    flipped = False
    if training:
        flipped = bool(force_flip) if force_flip is not None else random.random() < float(horizontal_flip_probability)
        if flipped:
            tensor = TF.hflip(tensor)
        apply_jitter = bool(force_color_jitter) if force_color_jitter is not None else random.random() < float(color_jitter_probability)
        if color_jitter_enabled and apply_jitter:
            tensor = build_color_jitter(color_jitter_config)(tensor)
    return normalize_rgb_tensor(tensor, mean=mean, std=std), flipped


def transform_paired_rgb(
    original: Image.Image | np.ndarray | torch.Tensor,
    counterfactual: Image.Image | np.ndarray | torch.Tensor,
    *,
    training: bool,
    image_size: int = 224,
    horizontal_flip_probability: float = 0.5,
    mean: tuple[float, float, float] | list[float] = IMAGENET_MEAN,
    std: tuple[float, float, float] | list[float] = IMAGENET_STD,
    force_flip: bool | None = None,
) -> tuple[torch.Tensor, torch.Tensor, bool]:
    original_tensor = rgb_to_tensor(original, image_size=image_size)
    counter_tensor = rgb_to_tensor(counterfactual, image_size=image_size)
    flipped = False
    if training:
        flipped = bool(force_flip) if force_flip is not None else random.random() < float(horizontal_flip_probability)
        if flipped:
            original_tensor = TF.hflip(original_tensor)
            counter_tensor = TF.hflip(counter_tensor)
    return (
        normalize_rgb_tensor(original_tensor, mean=mean, std=std),
        normalize_rgb_tensor(counter_tensor, mean=mean, std=std),
        flipped,
    )


def transform_full_six_rgb(
    original: Image.Image | np.ndarray | torch.Tensor,
    relighted: list[Image.Image | np.ndarray | torch.Tensor] | tuple[Image.Image | np.ndarray | torch.Tensor, ...],
    *,
    training: bool,
    image_size: int = 224,
    horizontal_flip_probability: float = 0.5,
    mean: tuple[float, float, float] | list[float] = IMAGENET_MEAN,
    std: tuple[float, float, float] | list[float] = IMAGENET_STD,
    force_flip: bool | None = None,
) -> tuple[torch.Tensor, torch.Tensor, bool]:
    if len(relighted) != 6:
        raise ValueError(f"full-six transform requires exactly 6 relighted images, got {len(relighted)}")
    original_tensor = rgb_to_tensor(original, image_size=image_size)
    relighted_tensors = [rgb_to_tensor(image, image_size=image_size) for image in relighted]
    flipped = False
    if training:
        flipped = bool(force_flip) if force_flip is not None else random.random() < float(horizontal_flip_probability)
        if flipped:
            original_tensor = TF.hflip(original_tensor)
            relighted_tensors = [TF.hflip(tensor) for tensor in relighted_tensors]
    original_tensor = normalize_rgb_tensor(original_tensor, mean=mean, std=std)
    relighted_stack = torch.stack(
        [normalize_rgb_tensor(tensor, mean=mean, std=std) for tensor in relighted_tensors],
        dim=0,
    )
    return original_tensor, relighted_stack, flipped
