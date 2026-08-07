"""Build face-parsing-guided regularized face-mask RGB PNG images.

The face parsing label map is used for semantic localization, quality control,
and area statistics. The final image mask is deliberately *not* a skin-only
semantic mask. It is a continuous elliptical face envelope derived from the
main facial semantic region, so local hair occluding the forehead remains as a
real image feature instead of becoming an irregular black hole.

ImageNet normalization is used only as the BiSeNet model's inference input
preprocessing. Saved images remain ordinary RGB uint8 PNG files; training-time
ImageNet normalization still belongs in the Dataset/transform pipeline.
"""

from __future__ import annotations

import argparse
import importlib.util
import math
import random
import shutil
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence


REQUIRED_DEPENDENCIES = (
    ("cv2", "opencv-python"),
    ("mediapipe", "mediapipe"),
    ("numpy", "numpy"),
    ("pandas", "pandas"),
    ("torch", "torch"),
    ("yaml", "pyyaml"),
)


def check_required_dependencies() -> None:
    missing = [
        package_name
        for import_name, package_name in REQUIRED_DEPENDENCIES
        if importlib.util.find_spec(import_name) is None
    ]
    if missing:
        packages = " ".join(dict.fromkeys(missing))
        raise SystemExit(
            "Missing required preprocessing dependencies: "
            f"{', '.join(missing)}.\nInstall them with:\n"
            f"python -m pip install {packages}"
        )


check_required_dependencies()

import cv2  # noqa: E402
import mediapipe as mp  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
import torch.nn.functional as F  # noqa: E402
import yaml  # noqa: E402

try:  # Running as a script puts preprocessing/ directly on sys.path.
    import build_global_face_oval_blackbg_png_simalign_strict as alignment
except ImportError:  # Importing as preprocessing.<module> from the project root.
    from preprocessing import (  # type: ignore[no-redef]
        build_global_face_oval_blackbg_png_simalign_strict as alignment,
    )


VALID_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp")
PARSING_INPUT_SIZE = 512
NUM_PARSING_CLASSES = 19
STATUS_VALUES = (
    "success",
    "failed_no_image",
    "failed_read_image",
    "failed_no_face",
    "failed_landmark_incomplete",
    "failed_alignment",
    "failed_parsing_model",
    "failed_empty_selected_mask",
    "failed_save",
    "failed_unexpected_error",
)

# CelebAMask-HQ mapping used by zllrunning/face-parsing.PyTorch checkpoints.
CELEBAMASK_HQ_CLASSES = {
    0: "background",
    1: "skin",
    2: "left_brow",
    3: "right_brow",
    4: "left_eye",
    5: "right_eye",
    6: "eyeglass",
    7: "left_ear",
    8: "right_ear",
    9: "earring",
    10: "nose",
    11: "mouth",
    12: "upper_lip",
    13: "lower_lip",
    14: "neck",
    15: "necklace",
    16: "cloth",
    17: "hair",
    18: "hat",
}

CLASS_BACKGROUND = 0
CLASS_SKIN = 1
CLASS_LEFT_BROW = 2
CLASS_RIGHT_BROW = 3
CLASS_LEFT_EYE = 4
CLASS_RIGHT_EYE = 5
CLASS_LEFT_EAR = 7
CLASS_RIGHT_EAR = 8
CLASS_NOSE = 10
CLASS_MOUTH = 11
CLASS_UPPER_LIP = 12
CLASS_LOWER_LIP = 13
CLASS_NECK = 14
CLASS_CLOTH = 16
CLASS_HAIR = 17

SELECTED_SEMANTIC_CLASSES = (
    CLASS_SKIN,
    CLASS_LEFT_BROW,
    CLASS_RIGHT_BROW,
    CLASS_LEFT_EYE,
    CLASS_RIGHT_EYE,
    CLASS_NOSE,
    CLASS_MOUTH,
    CLASS_UPPER_LIP,
    CLASS_LOWER_LIP,
)

PARSING_COLORS = np.array(
    [
        [0, 0, 0],
        [255, 170, 0],
        [255, 0, 85],
        [255, 0, 170],
        [0, 255, 0],
        [85, 255, 0],
        [170, 255, 0],
        [0, 255, 85],
        [0, 255, 170],
        [0, 170, 255],
        [0, 0, 255],
        [85, 0, 255],
        [170, 0, 255],
        [0, 85, 255],
        [255, 255, 0],
        [255, 255, 85],
        [255, 255, 170],
        [255, 0, 255],
        [255, 85, 255],
    ],
    dtype=np.uint8,
)

LOG_COLUMNS = (
    "ID", "NYHA", "extreme_label", "fold", "SEX",
    "input_path", "output_path", "status", "fail_reason",
    "num_faces_detected", "selected_face_index",
    "face_bbox_x", "face_bbox_y", "face_bbox_w", "face_bbox_h",
    "expanded_bbox_x", "expanded_bbox_y", "expanded_bbox_w", "expanded_bbox_h",
    "align_success", "rotation_angle", "scale_factor",
    "translation_x", "translation_y", "eye_distance",
    "parsing_model", "parsing_checkpoint", "parsing_success",
    "selected_semantic_area_pixels", "selected_semantic_area_ratio",
    "envelope_mask_area_pixels", "envelope_mask_area_ratio",
    "mask_area_pixels", "mask_area_ratio", "mask_warning",
    "skin_area_ratio", "brow_area_ratio", "eye_area_ratio",
    "nose_area_ratio", "mouth_area_ratio", "lip_area_ratio",
    "hair_area_ratio", "ear_area_ratio", "neck_area_ratio",
    "cloth_area_ratio", "background_area_ratio",
    "hair_inside_face_envelope_pixels",
    "hair_inside_face_envelope_ratio", "hair_warning_flag",
    "forehead_expand_ratio", "side_expand_ratio", "chin_expand_ratio",
    "feather_kernel", "background_mode", "feather_enabled",
    "output_format", "image_size",
)

FAILED_COLUMNS = (
    "ID", "NYHA", "extreme_label", "fold", "SEX",
    "input_path", "status", "fail_reason",
)

SUMMARY_METRICS = (
    "count", "mean", "std", "min", "p1", "p5", "p25",
    "median", "p75", "p95", "p99", "max",
)

SUMMARY_VALUE_COLUMNS = (
    "mask_area_ratio",
    "selected_semantic_area_ratio",
    "hair_inside_face_envelope_ratio",
    "skin_area_ratio",
    "hair_area_ratio",
    "neck_area_ratio",
    "cloth_area_ratio",
)


@dataclass
class PreviewImages:
    original: np.ndarray | None = None
    crop: np.ndarray | None = None
    aligned: np.ndarray | None = None
    parsing_map: np.ndarray | None = None
    selected_overlay: np.ndarray | None = None
    envelope_overlay: np.ndarray | None = None
    final: np.ndarray | None = None


# The following BiSeNet module names intentionally match the MIT-licensed
# zllrunning/face-parsing.PyTorch model.py/resnet.py implementation so its
# 19-class checkpoints can be loaded strictly. Its original ResNet constructor
# downloads backbone weights; this offline preprocessing version never does.
def conv3x3(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
    return nn.Conv2d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=1,
        bias=False,
    )


class BasicBlock(nn.Module):
    def __init__(self, in_chan: int, out_chan: int, stride: int = 1):
        super().__init__()
        self.conv1 = conv3x3(in_chan, out_chan, stride)
        self.bn1 = nn.BatchNorm2d(out_chan)
        self.conv2 = conv3x3(out_chan, out_chan)
        self.bn2 = nn.BatchNorm2d(out_chan)
        self.relu = nn.ReLU(inplace=True)
        self.downsample: nn.Module | None = None
        if in_chan != out_chan or stride != 1:
            self.downsample = nn.Sequential(
                nn.Conv2d(
                    in_chan, out_chan, kernel_size=1, stride=stride, bias=False
                ),
                nn.BatchNorm2d(out_chan),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = F.relu(self.bn1(self.conv1(x)))
        residual = self.bn2(self.conv2(residual))
        shortcut = self.downsample(x) if self.downsample is not None else x
        return self.relu(shortcut + residual)


def create_layer_basic(
    in_chan: int,
    out_chan: int,
    block_count: int,
    stride: int = 1,
) -> nn.Sequential:
    layers: list[nn.Module] = [BasicBlock(in_chan, out_chan, stride=stride)]
    layers.extend(
        BasicBlock(out_chan, out_chan, stride=1)
        for _ in range(block_count - 1)
    )
    return nn.Sequential(*layers)


class Resnet18(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(
            3, 64, kernel_size=7, stride=2, padding=3, bias=False
        )
        self.bn1 = nn.BatchNorm2d(64)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = create_layer_basic(64, 64, 2, stride=1)
        self.layer2 = create_layer_basic(64, 128, 2, stride=2)
        self.layer3 = create_layer_basic(128, 256, 2, stride=2)
        self.layer4 = create_layer_basic(256, 512, 2, stride=2)

    def forward(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x = self.maxpool(F.relu(self.bn1(self.conv1(x))))
        x = self.layer1(x)
        feat8 = self.layer2(x)
        feat16 = self.layer3(feat8)
        feat32 = self.layer4(feat16)
        return feat8, feat16, feat32


class ConvBNReLU(nn.Module):
    def __init__(
        self,
        in_chan: int,
        out_chan: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 1,
    ):
        super().__init__()
        self.conv = nn.Conv2d(
            in_chan,
            out_chan,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=False,
        )
        self.bn = nn.BatchNorm2d(out_chan)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.relu(self.bn(self.conv(x)))


class AttentionRefinementModule(nn.Module):
    def __init__(self, in_chan: int, out_chan: int):
        super().__init__()
        self.conv = ConvBNReLU(in_chan, out_chan)
        self.conv_atten = nn.Conv2d(out_chan, out_chan, kernel_size=1, bias=False)
        self.bn_atten = nn.BatchNorm2d(out_chan)
        self.sigmoid_atten = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.conv(x)
        atten = F.adaptive_avg_pool2d(feat, 1)
        atten = self.sigmoid_atten(self.bn_atten(self.conv_atten(atten)))
        return feat * atten


class ContextPath(nn.Module):
    def __init__(self):
        super().__init__()
        self.resnet = Resnet18()
        self.arm16 = AttentionRefinementModule(256, 128)
        self.arm32 = AttentionRefinementModule(512, 128)
        self.conv_head32 = ConvBNReLU(128, 128)
        self.conv_head16 = ConvBNReLU(128, 128)
        self.conv_avg = ConvBNReLU(512, 128, kernel_size=1, padding=0)

    def forward(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        feat8, feat16, feat32 = self.resnet(x)
        avg = self.conv_avg(F.adaptive_avg_pool2d(feat32, 1))
        avg_up = F.interpolate(avg, feat32.shape[2:], mode="nearest")
        feat32_sum = self.arm32(feat32) + avg_up
        feat32_up = self.conv_head32(
            F.interpolate(feat32_sum, feat16.shape[2:], mode="nearest")
        )
        feat16_sum = self.arm16(feat16) + feat32_up
        feat16_up = self.conv_head16(
            F.interpolate(feat16_sum, feat8.shape[2:], mode="nearest")
        )
        return feat8, feat16_up, feat32_up


class FeatureFusionModule(nn.Module):
    def __init__(self, in_chan: int, out_chan: int):
        super().__init__()
        self.convblk = ConvBNReLU(
            in_chan, out_chan, kernel_size=1, padding=0
        )
        self.conv1 = nn.Conv2d(out_chan, out_chan // 4, kernel_size=1, bias=False)
        self.conv2 = nn.Conv2d(out_chan // 4, out_chan, kernel_size=1, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, spatial: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        feat = self.convblk(torch.cat([spatial, context], dim=1))
        atten = F.adaptive_avg_pool2d(feat, 1)
        atten = self.conv2(self.relu(self.conv1(atten)))
        return feat * self.sigmoid(atten) + feat


class BiSeNetOutput(nn.Module):
    def __init__(self, in_chan: int, mid_chan: int, class_count: int):
        super().__init__()
        self.conv = ConvBNReLU(in_chan, mid_chan)
        self.conv_out = nn.Conv2d(
            mid_chan, class_count, kernel_size=1, bias=False
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv_out(self.conv(x))


class BiSeNet(nn.Module):
    def __init__(self, n_classes: int = NUM_PARSING_CLASSES):
        super().__init__()
        self.cp = ContextPath()
        self.ffm = FeatureFusionModule(256, 256)
        self.conv_out = BiSeNetOutput(256, 256, n_classes)
        self.conv_out16 = BiSeNetOutput(128, 64, n_classes)
        self.conv_out32 = BiSeNetOutput(128, 64, n_classes)

    def forward(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        height, width = x.shape[2:]
        feat_res8, feat_cp8, feat_cp16 = self.cp(x)
        feat_fuse = self.ffm(feat_res8, feat_cp8)
        output = F.interpolate(
            self.conv_out(feat_fuse),
            (height, width),
            mode="bilinear",
            align_corners=True,
        )
        output16 = F.interpolate(
            self.conv_out16(feat_cp8),
            (height, width),
            mode="bilinear",
            align_corners=True,
        )
        output32 = F.interpolate(
            self.conv_out32(feat_cp16),
            (height, width),
            mode="bilinear",
            align_corners=True,
        )
        return output, output16, output32


def default_project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Build strict similarity-aligned, face-parsing-guided regularized "
            "face-envelope RGB PNG images."
        )
    )
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--project-root", type=Path, default=None)
    parser.add_argument("--split-csv", type=Path, default=None)
    parser.add_argument("--image-dir", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--image-size", type=int, default=None)
    parser.add_argument("--min-detection-confidence", type=float, default=None)
    parser.add_argument("--parsing-model", type=str, default=None)
    parser.add_argument("--parsing-checkpoint", type=Path, default=None)
    parser.add_argument("--parsing-device", type=str, default=None)
    parser.add_argument("--mask-low-warning", type=float, default=None)
    parser.add_argument("--mask-high-warning", type=float, default=None)
    parser.add_argument("--hair-warning", type=float, default=None)
    parser.add_argument("--heavy-hair-warning", type=float, default=None)
    parser.add_argument("--num-qc-preview", type=int, default=None)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--forehead-expand-ratio", type=float, default=None)
    parser.add_argument("--side-expand-ratio", type=float, default=None)
    parser.add_argument("--chin-expand-ratio", type=float, default=None)
    parser.add_argument("--feather-kernel", type=int, default=None)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        default=None,
        help="Remove only the selected output directory before rebuilding it.",
    )
    return parser


def _normalize_config_keys(config: dict[str, Any]) -> dict[str, Any]:
    return {str(key).strip().replace("-", "_"): value for key, value in config.items()}


def load_yaml_config(config_path: Path | None, project_root: Path) -> dict[str, Any]:
    if config_path is None:
        candidate = (
            project_root
            / "config"
            / "preprocess"
            / "global_face_parsing_regularmask_blackbg_224_png_strict.yaml"
        )
        config_path = candidate if candidate.is_file() else None
    elif not config_path.is_absolute():
        config_path = project_root / config_path

    if config_path is None:
        return {}
    if not config_path.is_file():
        raise FileNotFoundError(f"Config file does not exist: {config_path}")
    with config_path.open("r", encoding="utf-8") as handle:
        loaded = yaml.safe_load(handle) or {}
    if not isinstance(loaded, dict):
        raise ValueError(f"Config root must be a YAML mapping: {config_path}")
    return _normalize_config_keys(loaded)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = build_parser()
    cli = parser.parse_args(argv)
    inferred_root = (
        cli.project_root.expanduser().resolve()
        if cli.project_root is not None
        else default_project_root()
    )
    config = load_yaml_config(cli.config, inferred_root)
    config_root = config.get("project_root")
    project_root = (
        inferred_root
        if cli.project_root is not None or config_root in (None, "", ".")
        else Path(config_root).expanduser().resolve()
    )
    defaults: dict[str, Any] = {
        "split_csv": "data/processed/splits/extreme_5fold.csv",
        "image_dir": "data/raw/images",
        "output_dir": (
            "data/processed/global_face/"
            "global_face_parsing_regularmask_blackbg_224_png_strict"
        ),
        "image_size": 224,
        "min_detection_confidence": 0.5,
        "parsing_model": "bisenet",
        "parsing_checkpoint": (
            "preprocessing/checkpoints/face_parsing/79999_iter.pth"
        ),
        "parsing_device": "auto",
        "mask_low_warning": 0.35,
        "mask_high_warning": 0.85,
        "hair_warning": 0.10,
        "heavy_hair_warning": 0.20,
        "num_qc_preview": 20,
        "max_samples": None,
        "seed": 42,
        "forehead_expand_ratio": 0.18,
        "side_expand_ratio": 0.05,
        "chin_expand_ratio": 0.03,
        "feather_kernel": 11,
        "overwrite": False,
    }
    values: dict[str, Any] = {"project_root": project_root, "config": cli.config}
    for key, default in defaults.items():
        cli_value = getattr(cli, key)
        values[key] = cli_value if cli_value is not None else config.get(key, default)
    args = argparse.Namespace(**values)
    validate_args(args)
    return args


def validate_args(args: argparse.Namespace) -> None:
    if int(args.image_size) <= 0:
        raise ValueError("--image-size must be greater than zero")
    if not 0.0 <= float(args.min_detection_confidence) <= 1.0:
        raise ValueError("--min-detection-confidence must be in [0, 1]")
    if str(args.parsing_model).lower() != "bisenet":
        raise ValueError("First version supports only --parsing-model bisenet")
    if str(args.parsing_device).lower() not in {"auto", "cpu", "cuda"}:
        raise ValueError("--parsing-device must be auto, cpu, or cuda")
    if not 0.0 <= float(args.mask_low_warning) < float(args.mask_high_warning) <= 1.0:
        raise ValueError("Mask warning thresholds must satisfy 0 <= low < high <= 1")
    if not 0.0 <= float(args.hair_warning) < float(args.heavy_hair_warning) <= 1.0:
        raise ValueError("Hair thresholds must satisfy 0 <= warning < heavy <= 1")
    if int(args.num_qc_preview) < 0:
        raise ValueError("--num-qc-preview must be non-negative")
    if args.max_samples is not None and int(args.max_samples) <= 0:
        raise ValueError("--max-samples must be greater than zero")
    for name in (
        "forehead_expand_ratio",
        "side_expand_ratio",
        "chin_expand_ratio",
    ):
        if not 0.0 <= float(getattr(args, name)) <= 1.0:
            raise ValueError(f"--{name.replace('_', '-')} must be in [0, 1]")
    if int(args.feather_kernel) <= 0 or int(args.feather_kernel) % 2 == 0:
        raise ValueError("--feather-kernel must be a positive odd integer")


def resolve_project_root(args: argparse.Namespace) -> Path:
    root = Path(args.project_root).expanduser().resolve()
    if not root.is_dir():
        raise NotADirectoryError(f"Project root does not exist: {root}")
    return root


def resolve_under_project(path_value: str | Path, project_root: Path) -> Path:
    path = Path(path_value).expanduser()
    return path.resolve() if path.is_absolute() else (project_root / path).resolve()


def load_split_table(split_csv: Path, max_samples: int | None) -> pd.DataFrame:
    table = alignment.load_split_ids(split_csv)
    if max_samples is not None:
        table = table.iloc[: int(max_samples)].copy()
    return table


def prepare_output_dirs(
    output_dir: Path, overwrite: bool, project_root: Path
) -> dict[str, Path]:
    output_dir = output_dir.resolve()
    protected = {
        project_root.resolve(),
        (project_root / "data").resolve(),
        (project_root / "data" / "processed").resolve(),
        (project_root / "data" / "processed" / "global_face").resolve(),
    }
    if output_dir in protected:
        raise ValueError(f"Refusing unsafe output directory: {output_dir}")
    if output_dir.exists():
        if not overwrite:
            raise FileExistsError(
                f"Output directory already exists: {output_dir}\n"
                "Use --overwrite only for this exact dataset directory."
            )
        shutil.rmtree(output_dir)

    names = (
        "images",
        "logs",
        "qc_preview/random_success",
        "qc_preview/low_face_area",
        "qc_preview/high_face_area",
        "qc_preview/high_hair_inside_envelope",
        "qc_preview/failed_no_face",
        "qc_preview/failed_alignment",
        "qc_preview/failed_parsing",
        "qc_preview/failed_empty_mask",
        "qc_preview/.staging_success",
    )
    dirs = {"root": output_dir}
    for name in names:
        key = name.split("/")[-1]
        dirs[key] = output_dir / name
        dirs[key].mkdir(parents=True, exist_ok=True)
    return dirs


def resolve_parsing_device(requested: str) -> torch.device:
    requested = requested.lower()
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--parsing-device cuda requested but CUDA is unavailable")
    return torch.device(requested)


def _extract_state_dict(checkpoint: Any) -> dict[str, torch.Tensor]:
    if isinstance(checkpoint, dict):
        for key in ("state_dict", "model_state_dict", "model", "net"):
            value = checkpoint.get(key)
            if isinstance(value, dict):
                checkpoint = value
                break
    if not isinstance(checkpoint, dict) or not checkpoint:
        raise ValueError("Checkpoint does not contain a model state_dict")
    state_dict = {
        str(key): value
        for key, value in checkpoint.items()
        if isinstance(value, torch.Tensor)
    }
    if not state_dict:
        raise ValueError("Checkpoint state_dict contains no tensors")
    for prefix in ("module.", "model.", "net."):
        if all(key.startswith(prefix) for key in state_dict):
            state_dict = {
                key[len(prefix) :]: value for key, value in state_dict.items()
            }
    return state_dict


def load_face_parsing_model(
    model_name: str,
    checkpoint_path: Path,
    device: torch.device,
) -> nn.Module:
    """Load a strict 19-class BiSeNet checkpoint without network downloads."""
    if model_name.lower() != "bisenet":
        raise ValueError(f"Unsupported parsing model: {model_name}")
    if not checkpoint_path.is_file():
        raise FileNotFoundError(
            "Face parsing checkpoint does not exist: "
            f"{checkpoint_path}\nProvide --parsing-checkpoint with a compatible "
            "19-class CelebAMask-HQ BiSeNet .pth file."
        )
    model = BiSeNet(n_classes=NUM_PARSING_CLASSES)
    try:
        try:
            checkpoint = torch.load(
                checkpoint_path, map_location="cpu", weights_only=True
            )
        except TypeError:
            checkpoint = torch.load(checkpoint_path, map_location="cpu")
        state_dict = _extract_state_dict(checkpoint)
        model.load_state_dict(state_dict, strict=True)
    except Exception as exc:
        raise ValueError(
            f"Failed to load compatible BiSeNet checkpoint {checkpoint_path}: {exc}"
        ) from exc
    model.to(device)
    model.eval()
    return model


def run_face_parsing(
    aligned_rgb: np.ndarray,
    model: nn.Module,
    device: torch.device,
) -> np.ndarray:
    """Return a 224-space CelebAMask-HQ label map."""
    resized = cv2.resize(
        aligned_rgb,
        (PARSING_INPUT_SIZE, PARSING_INPUT_SIZE),
        interpolation=cv2.INTER_LINEAR,
    )
    tensor = torch.from_numpy(resized).permute(2, 0, 1).float().div_(255.0)
    mean = torch.tensor([0.485, 0.456, 0.406])[:, None, None]
    std = torch.tensor([0.229, 0.224, 0.225])[:, None, None]
    tensor = ((tensor - mean) / std).unsqueeze(0).to(device)
    with torch.no_grad():
        output = model(tensor)
        logits = output[0] if isinstance(output, (tuple, list)) else output
        if logits.ndim != 4 or logits.shape[1] != NUM_PARSING_CLASSES:
            raise RuntimeError(f"Unexpected parsing logits shape: {tuple(logits.shape)}")
        label_map = logits.argmax(dim=1)[0].detach().cpu().numpy().astype(np.uint8)
    target_height, target_width = aligned_rgb.shape[:2]
    if label_map.shape != (target_height, target_width):
        label_map = cv2.resize(
            label_map,
            (target_width, target_height),
            interpolation=cv2.INTER_NEAREST,
        )
    if int(label_map.max(initial=0)) >= NUM_PARSING_CLASSES:
        raise RuntimeError("Parsing label map contains an out-of-range class")
    return label_map


def colorize_parsing_label_map(label_map: np.ndarray) -> np.ndarray:
    clipped = np.clip(label_map.astype(np.int32), 0, len(PARSING_COLORS) - 1)
    return PARSING_COLORS[clipped]


def build_selected_semantic_mask(label_map: np.ndarray) -> np.ndarray:
    """Build the semantic face evidence mask; this is not the final mask."""
    selected = np.isin(label_map, SELECTED_SEMANTIC_CLASSES).astype(np.uint8)
    minimum_pixels = max(64, int(round(label_map.size * 0.0025)))
    if int(selected.sum()) < minimum_pixels:
        raise alignment.SampleFailure(
            "failed_empty_selected_mask",
            f"selected_semantic_mask_too_small:{int(selected.sum())}",
        )

    # Closing is used only to identify the principal face region. Returned
    # pixels remain genuine selected semantic classes, not the closed mask.
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    closed = cv2.morphologyEx(selected * 255, cv2.MORPH_CLOSE, kernel)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(
        (closed > 0).astype(np.uint8), connectivity=8
    )
    if count <= 1:
        raise alignment.SampleFailure(
            "failed_empty_selected_mask", "no_connected_selected_face_region"
        )
    largest_label = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    primary_region = (labels == largest_label).astype(np.uint8) * 255
    support = cv2.dilate(
        primary_region,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)),
    )
    primary_selected = ((selected > 0) & (support > 0)).astype(np.uint8)
    if int(primary_selected.sum()) < minimum_pixels:
        raise alignment.SampleFailure(
            "failed_empty_selected_mask", "primary_selected_face_region_too_small"
        )
    return primary_selected


def _binary_bbox(mask: np.ndarray) -> tuple[int, int, int, int]:
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        raise alignment.SampleFailure(
            "failed_empty_selected_mask", "cannot_compute_empty_mask_bbox"
        )
    return int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())


def build_regularized_face_envelope_mask(
    selected_mask: np.ndarray,
    label_map: np.ndarray,
    forehead_expand_ratio: float,
    side_expand_ratio: float,
    chin_expand_ratio: float,
) -> tuple[np.ndarray, dict[str, int]]:
    """Build a continuous ellipse from the principal semantic face bbox.

    Ear, neck, cloth, hair, and background classes do not define the bbox.
    Hair pixels that happen to lie inside the resulting face envelope remain
    intact. Neck/cloth labels may conservatively cap the lower envelope without
    subtracting pixels or fragmenting the mask.
    """
    height, width = selected_mask.shape
    xmin, ymin, xmax, ymax = _binary_bbox(selected_mask)
    selected_width = max(1.0, float(xmax - xmin))
    selected_height = max(1.0, float(ymax - ymin))
    env_xmin = max(0.0, xmin - side_expand_ratio * selected_width)
    env_xmax = min(float(width - 1), xmax + side_expand_ratio * selected_width)
    env_ymin = max(0.0, ymin - forehead_expand_ratio * selected_height)
    env_ymax = min(float(height - 1), ymax + chin_expand_ratio * selected_height)

    # Use parsed ears only as side-boundary constraints. We do not subtract ear
    # pixels from the mask because that would create side holes; instead the
    # single ellipse is narrowed while never cutting inside the selected face
    # bbox itself.
    _, ear_x = np.where(
        np.isin(label_map, (CLASS_LEFT_EAR, CLASS_RIGHT_EAR))
    )
    selected_center_x = (xmin + xmax) / 2.0
    image_left_ear_x = ear_x[ear_x < selected_center_x]
    image_right_ear_x = ear_x[ear_x > selected_center_x]
    if len(image_left_ear_x):
        left_cap = float(int(image_left_ear_x.max()) + 1)
        env_xmin = min(float(xmin), max(env_xmin, left_cap))
    if len(image_right_ear_x):
        right_cap = float(int(image_right_ear_x.min()) - 1)
        env_xmax = max(float(xmax), min(env_xmax, right_cap))

    # Keep the envelope continuous while preventing its lower edge from
    # extending deeply into parsed neck or clothing.
    lower_classes = np.isin(label_map, (CLASS_NECK, CLASS_CLOTH))
    lower_y, _ = np.where(
        lower_classes
        & (np.indices(label_map.shape)[1] >= int(math.floor(env_xmin)))
        & (np.indices(label_map.shape)[1] <= int(math.ceil(env_xmax)))
        & (np.indices(label_map.shape)[0] >= ymin)
    )
    if len(lower_y):
        semantic_cap = float(int(lower_y.min()) - 1)
        env_ymax = max(float(ymax), min(env_ymax, semantic_cap))

    center_x = (env_xmin + env_xmax) / 2.0
    center_y = (env_ymin + env_ymax) / 2.0
    radius_x = max(1.0, (env_xmax - env_xmin) / 2.0)
    radius_y = max(1.0, (env_ymax - env_ymin) / 2.0)
    grid_y, grid_x = np.ogrid[:height, :width]
    ellipse = (
        ((grid_x - center_x) / radius_x) ** 2
        + ((grid_y - center_y) / radius_y) ** 2
        <= 1.0
    )
    envelope = ellipse.astype(np.uint8) * 255
    if int((envelope > 0).sum()) == 0:
        raise alignment.SampleFailure(
            "failed_empty_selected_mask", "regularized_envelope_is_empty"
        )
    bbox = {
        "envelope_bbox_xmin": int(math.floor(env_xmin)),
        "envelope_bbox_ymin": int(math.floor(env_ymin)),
        "envelope_bbox_xmax": int(math.ceil(env_xmax)),
        "envelope_bbox_ymax": int(math.ceil(env_ymax)),
    }
    return envelope, bbox


def compute_semantic_area_ratios(label_map: np.ndarray) -> dict[str, float]:
    denominator = float(label_map.size)

    def ratio(classes: Iterable[int]) -> float:
        return float(np.isin(label_map, tuple(classes)).sum()) / denominator

    return {
        "skin_area_ratio": ratio((CLASS_SKIN,)),
        "brow_area_ratio": ratio((CLASS_LEFT_BROW, CLASS_RIGHT_BROW)),
        "eye_area_ratio": ratio((CLASS_LEFT_EYE, CLASS_RIGHT_EYE)),
        "nose_area_ratio": ratio((CLASS_NOSE,)),
        "mouth_area_ratio": ratio((CLASS_MOUTH,)),
        "lip_area_ratio": ratio((CLASS_UPPER_LIP, CLASS_LOWER_LIP)),
        "hair_area_ratio": ratio((CLASS_HAIR,)),
        "ear_area_ratio": ratio((CLASS_LEFT_EAR, CLASS_RIGHT_EAR)),
        "neck_area_ratio": ratio((CLASS_NECK,)),
        "cloth_area_ratio": ratio((CLASS_CLOTH,)),
        "background_area_ratio": ratio((CLASS_BACKGROUND,)),
    }


def compute_hair_inside_envelope_ratio(
    label_map: np.ndarray,
    envelope_mask: np.ndarray,
) -> tuple[int, float]:
    envelope = envelope_mask > 0
    envelope_pixels = int(envelope.sum())
    if envelope_pixels == 0:
        raise alignment.SampleFailure(
            "failed_empty_selected_mask", "cannot_measure_empty_envelope"
        )
    hair_pixels = int(((label_map == CLASS_HAIR) & envelope).sum())
    return hair_pixels, hair_pixels / float(envelope_pixels)


def feather_mask(binary_mask: np.ndarray, kernel_size: int) -> np.ndarray:
    sigma = max(1.0, kernel_size / 3.0)
    blurred = cv2.GaussianBlur(
        binary_mask, (kernel_size, kernel_size), sigmaX=sigma
    )
    return blurred.astype(np.float32) / 255.0


def _draw_detection_panel(
    image_rgb: np.ndarray,
    detections: Sequence[alignment.FaceDetection],
    selected: alignment.FaceDetection | None,
) -> np.ndarray:
    panel = image_rgb.copy()
    for detection in detections:
        x, y, box_width, box_height = detection.bbox
        color = (
            (0, 255, 0)
            if selected is not None and detection.index == selected.index
            else (255, 180, 0)
        )
        cv2.rectangle(
            panel,
            (x, y),
            (x + box_width, y + box_height),
            color,
            3,
        )
    return panel


def _mask_overlay(
    image_rgb: np.ndarray,
    binary_mask: np.ndarray,
    color: tuple[int, int, int],
) -> np.ndarray:
    result = image_rgb.astype(np.float32)
    foreground = binary_mask > 0
    tint = np.zeros_like(result)
    tint[:] = color
    result[foreground] = result[foreground] * 0.68 + tint[foreground] * 0.32
    rendered = np.clip(result, 0, 255).astype(np.uint8)
    contours, _ = cv2.findContours(
        (binary_mask > 0).astype(np.uint8) * 255,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )
    cv2.drawContours(rendered, contours, -1, (255, 40, 40), 2, cv2.LINE_AA)
    return rendered


def _fit_preview_panel(
    image_rgb: np.ndarray | None,
    title: str,
    panel_size: int = 240,
) -> np.ndarray:
    label_height = 34
    canvas = np.full(
        (panel_size + label_height, panel_size, 3), 20, dtype=np.uint8
    )
    if image_rgb is not None and image_rgb.size:
        height, width = image_rgb.shape[:2]
        scale = min(panel_size / width, panel_size / height)
        out_width = max(1, int(round(width * scale)))
        out_height = max(1, int(round(height * scale)))
        resized = cv2.resize(
            image_rgb, (out_width, out_height), interpolation=cv2.INTER_AREA
        )
        x = (panel_size - out_width) // 2
        y = (panel_size - out_height) // 2
        canvas[y : y + out_height, x : x + out_width] = resized
    cv2.putText(
        canvas,
        title,
        (7, panel_size + 23),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.48,
        (240, 240, 240),
        1,
        cv2.LINE_AA,
    )
    return canvas


def make_qc_preview(
    output_path: Path,
    image_id: str,
    status: str,
    previews: PreviewImages,
    details: str,
) -> None:
    panels = [
        _fit_preview_panel(previews.original, "original"),
        _fit_preview_panel(previews.crop, "expanded crop"),
        _fit_preview_panel(previews.aligned, "aligned"),
        _fit_preview_panel(previews.parsing_map, "parsing labels"),
        _fit_preview_panel(previews.selected_overlay, "selected semantics"),
        _fit_preview_panel(previews.envelope_overlay, "regular envelope"),
        _fit_preview_panel(previews.final, "black-bg PNG"),
    ]
    strip = np.concatenate(panels, axis=1)
    header_height = 44
    canvas = np.full(
        (strip.shape[0] + header_height, strip.shape[1], 3),
        10,
        dtype=np.uint8,
    )
    canvas[header_height:] = strip
    header = f"ID={image_id} status={status} {details}"
    cv2.putText(
        canvas,
        header[:230],
        (10, 29),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.58,
        (245, 245, 245),
        1,
        cv2.LINE_AA,
    )
    bgr = cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR)
    ok, encoded = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, 90])
    if ok:
        try:
            output_path.write_bytes(encoded.tobytes())
        except OSError as exc:
            print(f"[warning] Failed to write QC preview {output_path}: {exc}")


def _initial_log_row(
    split_row: dict[str, Any],
    args: argparse.Namespace,
) -> dict[str, Any]:
    row = dict(split_row)
    row.update(
        {
            "input_path": "",
            "output_path": "",
            "status": "",
            "fail_reason": "",
            "num_faces_detected": 0,
            "selected_face_index": np.nan,
            "face_bbox_x": np.nan,
            "face_bbox_y": np.nan,
            "face_bbox_w": np.nan,
            "face_bbox_h": np.nan,
            "expanded_bbox_x": np.nan,
            "expanded_bbox_y": np.nan,
            "expanded_bbox_w": np.nan,
            "expanded_bbox_h": np.nan,
            "align_success": False,
            "rotation_angle": np.nan,
            "scale_factor": np.nan,
            "translation_x": np.nan,
            "translation_y": np.nan,
            "eye_distance": np.nan,
            "parsing_model": str(args.parsing_model),
            "parsing_checkpoint": str(args.parsing_checkpoint),
            "parsing_success": False,
            "selected_semantic_area_pixels": np.nan,
            "selected_semantic_area_ratio": np.nan,
            "envelope_mask_area_pixels": np.nan,
            "envelope_mask_area_ratio": np.nan,
            "mask_area_pixels": np.nan,
            "mask_area_ratio": np.nan,
            "mask_warning": "none",
            "skin_area_ratio": np.nan,
            "brow_area_ratio": np.nan,
            "eye_area_ratio": np.nan,
            "nose_area_ratio": np.nan,
            "mouth_area_ratio": np.nan,
            "lip_area_ratio": np.nan,
            "hair_area_ratio": np.nan,
            "ear_area_ratio": np.nan,
            "neck_area_ratio": np.nan,
            "cloth_area_ratio": np.nan,
            "background_area_ratio": np.nan,
            "hair_inside_face_envelope_pixels": np.nan,
            "hair_inside_face_envelope_ratio": np.nan,
            "hair_warning_flag": "none",
            "forehead_expand_ratio": float(args.forehead_expand_ratio),
            "side_expand_ratio": float(args.side_expand_ratio),
            "chin_expand_ratio": float(args.chin_expand_ratio),
            "feather_kernel": int(args.feather_kernel),
            "background_mode": "black_rgb_0_0_0",
            "feather_enabled": True,
            "output_format": "PNG",
            "image_size": int(args.image_size),
        }
    )
    return row


def process_one_sample(
    split_row: dict[str, Any],
    image_dir: Path,
    images_dir: Path,
    detector: Any,
    face_mesh: Any,
    parsing_model: nn.Module,
    parsing_device: torch.device,
    args: argparse.Namespace,
) -> tuple[dict[str, Any], PreviewImages]:
    image_id = str(split_row["ID"])
    row = _initial_log_row(split_row, args)
    previews = PreviewImages()
    input_path = alignment.find_image_for_id(image_id, image_dir)
    output_path = images_dir / f"{image_id}.png"
    row["output_path"] = str(output_path)

    try:
        if input_path is None:
            raise alignment.SampleFailure(
                "failed_no_image", "cannot_find_image_by_ID"
            )
        row["input_path"] = str(input_path)
        image_rgb = alignment.read_image_rgb(input_path)
        if image_rgb is None:
            raise alignment.SampleFailure(
                "failed_read_image", "cannot_decode_image"
            )
        previews.original = image_rgb

        detections = alignment.detect_faces(image_rgb, detector)
        row["num_faces_detected"] = len(detections)
        selected_face = alignment.select_face(detections, image_rgb.shape)
        previews.original = _draw_detection_panel(
            image_rgb, detections, selected_face
        )
        if selected_face is None:
            raise alignment.SampleFailure(
                "failed_no_face", "mediapipe_face_detection_returned_none"
            )

        row["selected_face_index"] = selected_face.index
        x, y, box_width, box_height = selected_face.bbox
        row.update(
            {
                "face_bbox_x": x,
                "face_bbox_y": y,
                "face_bbox_w": box_width,
                "face_bbox_h": box_height,
            }
        )
        crop_x, crop_y, crop_width, crop_height = alignment.expand_bbox(
            selected_face.bbox, image_rgb.shape
        )
        row.update(
            {
                "expanded_bbox_x": crop_x,
                "expanded_bbox_y": crop_y,
                "expanded_bbox_w": crop_width,
                "expanded_bbox_h": crop_height,
            }
        )
        if crop_width <= 1 or crop_height <= 1:
            raise alignment.SampleFailure(
                "failed_alignment", "expanded_face_bbox_is_empty"
            )
        crop_rgb = image_rgb[
            crop_y : crop_y + crop_height,
            crop_x : crop_x + crop_width,
        ].copy()
        previews.crop = crop_rgb

        landmarks = alignment.run_facemesh(crop_rgb, face_mesh)
        if landmarks is None:
            raise alignment.SampleFailure(
                "failed_landmark_incomplete", "mediapipe_facemesh_returned_none"
            )
        alignment_points, _, eye_distance = (
            alignment.extract_alignment_landmarks(landmarks, crop_rgb.shape)
        )
        row["eye_distance"] = eye_distance
        matrix, transform_parameters = alignment.estimate_similarity_transform(
            alignment_points,
            alignment.canonical_alignment_template(int(args.image_size)),
        )
        row.update(transform_parameters)
        aligned_rgb = alignment.warp_face(
            crop_rgb, matrix, int(args.image_size)
        )
        previews.aligned = aligned_rgb
        row["align_success"] = True

        try:
            label_map = run_face_parsing(
                aligned_rgb, parsing_model, parsing_device
            )
        except Exception as exc:
            if parsing_device.type == "cuda":
                torch.cuda.empty_cache()
            raise alignment.SampleFailure(
                "failed_parsing_model",
                f"{type(exc).__name__}: {exc}",
            ) from exc
        row["parsing_success"] = True
        previews.parsing_map = colorize_parsing_label_map(label_map)

        selected_mask = build_selected_semantic_mask(label_map)
        selected_pixels = int((selected_mask > 0).sum())
        row["selected_semantic_area_pixels"] = selected_pixels
        row["selected_semantic_area_ratio"] = (
            selected_pixels / float(label_map.size)
        )
        previews.selected_overlay = _mask_overlay(
            aligned_rgb, selected_mask, (30, 220, 70)
        )

        envelope_mask, _ = build_regularized_face_envelope_mask(
            selected_mask,
            label_map,
            float(args.forehead_expand_ratio),
            float(args.side_expand_ratio),
            float(args.chin_expand_ratio),
        )
        envelope_pixels = int((envelope_mask > 0).sum())
        envelope_ratio = envelope_pixels / float(label_map.size)
        row.update(
            {
                "envelope_mask_area_pixels": envelope_pixels,
                "envelope_mask_area_ratio": envelope_ratio,
                "mask_area_pixels": envelope_pixels,
                "mask_area_ratio": envelope_ratio,
            }
        )
        if envelope_ratio < float(args.mask_low_warning):
            row["mask_warning"] = "warning_low_mask_area"
        elif envelope_ratio > float(args.mask_high_warning):
            row["mask_warning"] = "warning_high_mask_area"

        row.update(compute_semantic_area_ratios(label_map))
        hair_pixels, hair_ratio = compute_hair_inside_envelope_ratio(
            label_map, envelope_mask
        )
        row["hair_inside_face_envelope_pixels"] = hair_pixels
        row["hair_inside_face_envelope_ratio"] = hair_ratio
        if hair_ratio > float(args.heavy_hair_warning):
            row["hair_warning_flag"] = "warning_heavy_hair_occlusion"
        elif hair_ratio > float(args.hair_warning):
            row["hair_warning_flag"] = "warning_hair_occlusion"

        previews.envelope_overlay = _mask_overlay(
            aligned_rgb, envelope_mask, (50, 120, 255)
        )
        alpha = feather_mask(envelope_mask, int(args.feather_kernel))
        final_rgb = alignment.apply_black_background(aligned_rgb, alpha)
        previews.final = final_rgb
        alignment.save_png(output_path, final_rgb)
        row["status"] = "success"
    except alignment.SampleFailure as exc:
        row["status"] = exc.status
        row["fail_reason"] = exc.reason
    except Exception as exc:
        row["status"] = "failed_unexpected_error"
        row["fail_reason"] = f"{type(exc).__name__}: {exc}"

    if row["status"] not in STATUS_VALUES:
        row["fail_reason"] = (
            f"invalid_status={row['status']}; {row['fail_reason']}"
        )
        row["status"] = "failed_unexpected_error"
    return row, previews


def _ordered_log_columns(
    log_df: pd.DataFrame,
    split_columns: Sequence[str],
) -> list[str]:
    preferred = list(LOG_COLUMNS)
    extras = [
        column
        for column in split_columns
        if column not in preferred and column in log_df.columns
    ]
    remaining = [
        column
        for column in log_df.columns
        if column not in preferred and column not in extras
    ]
    return preferred + extras + remaining


def _distribution(values: pd.Series) -> dict[str, Any]:
    numeric = pd.to_numeric(values, errors="coerce").dropna()
    if numeric.empty:
        return {metric: np.nan for metric in SUMMARY_METRICS}
    return {
        "count": int(numeric.count()),
        "mean": float(numeric.mean()),
        "std": float(numeric.std(ddof=1)),
        "min": float(numeric.min()),
        "p1": float(numeric.quantile(0.01)),
        "p5": float(numeric.quantile(0.05)),
        "p25": float(numeric.quantile(0.25)),
        "median": float(numeric.quantile(0.50)),
        "p75": float(numeric.quantile(0.75)),
        "p95": float(numeric.quantile(0.95)),
        "p99": float(numeric.quantile(0.99)),
        "max": float(numeric.max()),
    }


def build_parsing_area_summary(log_df: pd.DataFrame) -> pd.DataFrame:
    success = log_df[log_df["status"] == "success"].copy()
    rows: list[dict[str, Any]] = []

    def append_group(
        group_type: str, group_value: Any, frame: pd.DataFrame
    ) -> None:
        for value_column in SUMMARY_VALUE_COLUMNS:
            row = {
                "group_type": group_type,
                "group_value": group_value,
                "metric": value_column,
            }
            row.update(_distribution(frame[value_column]))
            rows.append(row)

    append_group("overall", "all", success)
    for column in ("extreme_label", "fold"):
        if column in success.columns:
            for value, group in success.groupby(column, dropna=False, sort=True):
                append_group(column, value, group)
    return pd.DataFrame(
        rows,
        columns=("group_type", "group_value", "metric", *SUMMARY_METRICS),
    )


def _format_distribution(stats: dict[str, Any]) -> str:
    return ", ".join(
        f"{key}={value:.6f}"
        if isinstance(value, float) and math.isfinite(value)
        else f"{key}={value}"
        for key, value in stats.items()
    )


def summarize_logs(
    log_df: pd.DataFrame,
    args: argparse.Namespace,
    output_dir: Path,
    logs_dir: Path,
) -> str:
    log_df.to_csv(
        logs_dir / "preprocess_log.csv", index=False, encoding="utf-8-sig"
    )
    failed = log_df[log_df["status"] != "success"].copy()
    for column in FAILED_COLUMNS:
        if column not in failed.columns:
            failed[column] = ""
    failed[list(FAILED_COLUMNS)].to_csv(
        logs_dir / "failed_cases.csv", index=False, encoding="utf-8-sig"
    )
    build_parsing_area_summary(log_df).to_csv(
        logs_dir / "parsing_area_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )

    total = len(log_df)
    successes = int(log_df["status"].eq("success").sum())
    success_df = log_df[log_df["status"] == "success"]
    lines = [
        "Face parsing-guided regularized face-mask preprocessing summary",
        "=" * 72,
        f"Total IDs: {total}",
        f"Successes: {successes}",
        f"Failures: {total - successes}",
        f"Success rate: {successes / total if total else 0.0:.2%}",
        "",
        "Failure statuses:",
    ]
    counts = Counter(failed["status"].astype(str))
    lines.extend(
        [f"  {status}: {count}" for status, count in sorted(counts.items())]
        or ["  none"]
    )
    for column in ("fold", "extreme_label"):
        lines.extend(["", f"Success/failure by {column}:"])
        if column not in log_df.columns:
            lines.append("  column not present")
            continue
        grouped = (
            log_df.assign(success=log_df["status"].eq("success"))
            .groupby(column, dropna=False)["success"]
            .agg(total="size", success="sum")
        )
        for value, group_row in grouped.iterrows():
            success_count = int(group_row["success"])
            lines.append(
                f"  {value}: total={int(group_row['total'])}, "
                f"success={success_count}, "
                f"failed={int(group_row['total']) - success_count}"
            )

    for metric in ("mask_area_ratio", "hair_inside_face_envelope_ratio"):
        lines.extend(
            [
                "",
                f"{metric} overall:",
                f"  {_format_distribution(_distribution(success_df[metric]))}",
            ]
        )
        if "extreme_label" in success_df.columns:
            lines.append(f"{metric} by extreme_label:")
            for value, group in success_df.groupby(
                "extreme_label", dropna=False, sort=True
            ):
                lines.append(
                    f"  {value}: "
                    f"{_format_distribution(_distribution(group[metric]))}"
                )

    lines.extend(
        [
            "",
            f"Output directory: {output_dir}",
            "Parameters:",
            f"  project_root: {args.project_root}",
            f"  split_csv: {args.split_csv}",
            f"  image_dir: {args.image_dir}",
            f"  output_dir: {args.output_dir}",
            f"  image_size: {args.image_size}",
            f"  min_detection_confidence: {args.min_detection_confidence}",
            f"  parsing_model: {args.parsing_model}",
            f"  parsing_checkpoint: {args.parsing_checkpoint}",
            f"  parsing_device: {args.parsing_device}",
            f"  mask_low_warning: {args.mask_low_warning}",
            f"  mask_high_warning: {args.mask_high_warning}",
            f"  hair_warning: {args.hair_warning}",
            f"  heavy_hair_warning: {args.heavy_hair_warning}",
            f"  forehead_expand_ratio: {args.forehead_expand_ratio}",
            f"  side_expand_ratio: {args.side_expand_ratio}",
            f"  chin_expand_ratio: {args.chin_expand_ratio}",
            f"  feather_kernel: {args.feather_kernel}",
            f"  max_samples: {args.max_samples}",
            f"  seed: {args.seed}",
            f"  num_qc_preview: {args.num_qc_preview}",
            f"  overwrite: {args.overwrite}",
            "",
            "Strict failure policy: enabled; no detection/alignment/parsing fallback.",
            "Final mask: continuous regularized semantic-guided envelope.",
            "Hair inside the envelope is retained and measured, not cut out.",
            "Output: RGB uint8 PNG. Training applies ImageNet Normalize separately.",
        ]
    )
    summary = "\n".join(lines)
    (logs_dir / "preprocess_summary.txt").write_text(
        summary + "\n", encoding="utf-8"
    )
    return summary


def finalize_success_qc(
    log_df: pd.DataFrame,
    dirs: dict[str, Path],
    num_qc_preview: int,
    seed: int,
) -> None:
    success = log_df[log_df["status"] == "success"].copy()
    if success.empty or num_qc_preview <= 0:
        shutil.rmtree(dirs[".staging_success"], ignore_errors=True)
        return
    success["mask_area_ratio"] = pd.to_numeric(
        success["mask_area_ratio"], errors="coerce"
    )
    success["hair_inside_face_envelope_ratio"] = pd.to_numeric(
        success["hair_inside_face_envelope_ratio"], errors="coerce"
    )
    selections = (
        (
            "random_success",
            success.sample(
                n=min(num_qc_preview, len(success)), random_state=seed
            ),
        ),
        (
            "low_face_area",
            success.nsmallest(min(20, len(success)), "mask_area_ratio"),
        ),
        (
            "high_face_area",
            success.nlargest(min(20, len(success)), "mask_area_ratio"),
        ),
        (
            "high_hair_inside_envelope",
            success.nlargest(
                min(20, len(success)), "hair_inside_face_envelope_ratio"
            ),
        ),
    )
    for category, frame in selections:
        for row in frame.itertuples(index=False):
            source = dirs[".staging_success"] / f"{row.ID}.jpg"
            if source.is_file():
                try:
                    shutil.copy2(source, dirs[category] / source.name)
                except OSError as exc:
                    print(f"[warning] Failed to copy QC preview: {exc}")
    shutil.rmtree(dirs[".staging_success"], ignore_errors=True)


def _failure_qc_category(status: str) -> str | None:
    return {
        "failed_no_face": "failed_no_face",
        "failed_landmark_incomplete": "failed_alignment",
        "failed_alignment": "failed_alignment",
        "failed_parsing_model": "failed_parsing",
        "failed_empty_selected_mask": "failed_empty_mask",
    }.get(status)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    project_root = resolve_project_root(args)
    args.project_root = project_root
    args.split_csv = resolve_under_project(args.split_csv, project_root)
    args.image_dir = resolve_under_project(args.image_dir, project_root)
    args.output_dir = resolve_under_project(args.output_dir, project_root)
    if args.parsing_checkpoint is None:
        raise FileNotFoundError(
            "--parsing-checkpoint is required. No checkpoint is bundled and "
            "the script will not download or randomly initialize one."
        )
    args.parsing_checkpoint = resolve_under_project(
        args.parsing_checkpoint, project_root
    )
    if not args.image_dir.is_dir():
        raise NotADirectoryError(f"Image directory does not exist: {args.image_dir}")

    split_df = load_split_table(args.split_csv, args.max_samples)
    parsing_device = resolve_parsing_device(str(args.parsing_device))
    parsing_model = load_face_parsing_model(
        str(args.parsing_model),
        args.parsing_checkpoint,
        parsing_device,
    )
    dirs = prepare_output_dirs(
        args.output_dir, bool(args.overwrite), project_root
    )

    random.seed(int(args.seed))
    np.random.seed(int(args.seed))
    torch.manual_seed(int(args.seed))
    detector = mp.solutions.face_detection.FaceDetection(
        model_selection=1,
        min_detection_confidence=float(args.min_detection_confidence),
    )
    face_mesh = mp.solutions.face_mesh.FaceMesh(
        static_image_mode=True,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=float(args.min_detection_confidence),
        min_tracking_confidence=0.5,
    )

    rows: list[dict[str, Any]] = []
    failed_preview_counts: Counter[str] = Counter()
    try:
        total = len(split_df)
        for index, split_row in enumerate(split_df.to_dict("records"), start=1):
            image_id = str(split_row["ID"])
            print(f"[{index:04d}/{total:04d}] {image_id}")
            row, previews = process_one_sample(
                split_row,
                args.image_dir,
                dirs["images"],
                detector,
                face_mesh,
                parsing_model,
                parsing_device,
                args,
            )
            rows.append(row)
            status = str(row["status"])
            if status == "success":
                details = (
                    f"mask={row['mask_area_ratio']:.4f} "
                    f"selected={row['selected_semantic_area_ratio']:.4f} "
                    f"hair_in={row['hair_inside_face_envelope_ratio']:.4f} "
                    f"{row['hair_warning_flag']}"
                )
                make_qc_preview(
                    dirs[".staging_success"] / f"{image_id}.jpg",
                    image_id,
                    status,
                    previews,
                    details,
                )
            else:
                category = _failure_qc_category(status)
                if category and failed_preview_counts[category] < 20:
                    make_qc_preview(
                        dirs[category] / f"{image_id}.jpg",
                        image_id,
                        status,
                        previews,
                        str(row["fail_reason"]),
                    )
                    failed_preview_counts[category] += 1
    finally:
        detector.close()
        face_mesh.close()

    log_df = pd.DataFrame(rows)
    columns = _ordered_log_columns(log_df, split_df.columns.tolist())
    for column in columns:
        if column not in log_df.columns:
            log_df[column] = np.nan
    log_df = log_df[columns]
    finalize_success_qc(
        log_df, dirs, int(args.num_qc_preview), int(args.seed)
    )
    summary = summarize_logs(log_df, args, args.output_dir, dirs["logs"])
    print("\n" + summary)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (
        FileNotFoundError,
        FileExistsError,
        NotADirectoryError,
        RuntimeError,
        ValueError,
    ) as exc:
        print(f"[configuration error] {exc}", file=sys.stderr)
        raise SystemExit(2) from exc
