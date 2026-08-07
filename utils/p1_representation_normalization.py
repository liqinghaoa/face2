"""Representation loading and fixed normalization for P1 components."""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np
import pandas as pd
import torch

from p0b_deca.p1_dataset import _hwc_to_chw, _image_to_chw_float32, _mask_to_1hw

from utils.p1_component_registry import (
    EXPERIMENT_ORDER,
    P1ComponentSpec,
    get_component_spec,
    normalize_experiment_key,
)
from utils.p1_rgb_audit import local_path


IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32)


@dataclass(frozen=True)
class NormalFlipAudit:
    status: str
    evidence_paths: tuple[str, ...] = ()
    reason: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ComponentNormalizationState:
    experiment_key: str
    representation: str
    strategy: str
    fold: int | None
    fitted_from: str
    channel_mean: tuple[float, ...] | None = None
    channel_std: tuple[float, ...] | None = None
    p01: tuple[float, ...] | None = None
    p99: tuple[float, ...] | None = None
    residual_scale: tuple[float, ...] | None = None
    clip_min: float | None = None
    clip_max: float | None = None
    normal_flip_status: str = "UNVERIFIED"
    normal_coordinate_status: str = "UNVERIFIED"
    normal_flip_mode: str = "DISABLED_SAFE_FALLBACK"
    horizontal_flip_enabled: bool | None = None
    normal_flip_reason: str = ""
    out_of_range_fraction: float | None = None
    source_case_count: int = 0
    source_patient_group_count: int = 0
    source_fold_count: int = 0
    source_case_ids: tuple[str, ...] = ()
    note: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _as_path(value: Any) -> Path:
    return local_path(value)


def _map_key_for_representation(representation: str) -> str:
    return {
        "albedo": "albedo_like",
        "normal": "normal_coarse",
        "shading": "shading_like",
        "residual": "signed_residual",
    }.get(representation, representation)


def inspect_normal_flip_convention(root: str | Path | None = None) -> NormalFlipAudit:
    """Inspect available evidence for the P1-N horizontal flip contract."""

    evidence: list[Path] = []
    search_root = Path(root).resolve() if root is not None else Path(__file__).resolve().parents[1]
    explicit_evidence = [
        search_root / "data/processed/P0B_DECA_Pilot12_v1/environment/sh_validation/sh_coordinate_convention.json",
        search_root / "p0b_deca/p1_component_audit.py",
        search_root / "reports/p1_data_interface_and_component_statistics_audit.md",
    ]
    for candidate in explicit_evidence:
        if candidate.is_file():
            evidence.append(candidate)

    if not evidence:
        return NormalFlipAudit(
            status="UNVERIFIED",
            reason="no explicit normal coordinate evidence was found in the repository",
        )

    coordinate_file = search_root / "data/processed/P0B_DECA_Pilot12_v1/environment/sh_validation/sh_coordinate_convention.json"
    if coordinate_file.is_file():
        payload = json.loads(coordinate_file.read_text(encoding="utf-8"))
        if not bool(payload.get("renderer_basis_verified", False)):
            return NormalFlipAudit(
                status="UNVERIFIED",
                evidence_paths=tuple(str(path) for path in evidence),
                reason="SH coordinate basis is documented, but normal channel order / horizontal flip are not explicitly verified",
            )

    return NormalFlipAudit(
        status="VERIFIED",
        evidence_paths=tuple(str(path) for path in evidence),
        reason="explicit repository evidence was found",
    )


def _load_rgb(path: Path) -> torch.Tensor:
    return _image_to_chw_float32(path)


def _load_mask(path: Path) -> torch.Tensor:
    return _mask_to_1hw(path)


def _load_npz_array(path: Path, key: str) -> torch.Tensor:
    with np.load(path, allow_pickle=False) as archive:
        if key not in archive.files:
            raise KeyError(f"{path} does not contain {key}")
        array = np.asarray(archive[key], dtype=np.float32)
    if array.ndim == 3:
        return _hwc_to_chw(array)
    if array.ndim == 2:
        return torch.from_numpy(array.copy())
    return torch.from_numpy(array.copy())


def _load_latent_tensor(path: Path, key: str) -> torch.Tensor:
    with np.load(path, allow_pickle=False) as archive:
        if key not in archive.files:
            raise KeyError(f"{path} does not contain {key}")
        array = np.asarray(archive[key], dtype=np.float32)
    return torch.from_numpy(np.squeeze(array).copy())


def load_raw_component_sample(row: Mapping[str, Any], spec: P1ComponentSpec) -> dict[str, Any]:
    """Load only the fields needed by a single P1 component."""

    case_id = str(row["case_id"])
    sample: dict[str, Any] = {
        "case_id": case_id,
        "patient_group_id": str(row["group_id"] if "group_id" in row else row.get("patient_group_id", case_id)),
        "fold": int(row["fold"]),
        "label_original": int(row["label_original"]),
        "label_3class": int(row["label_3class"]),
        "label_binary": int(row["label_binary"]),
    }

    if spec.key == "p1_l":
        sample["representation"] = _load_latent_tensor(_as_path(row["latents_path"]), "light_code")
        sample["valid_mask"] = None
        return sample

    if spec.key == "p1_rgb_a":
        sample["representation"] = {
            "rgb": _load_rgb(_as_path(row["rgb_path"])),
            "albedo": _load_npz_array(_as_path(row["maps_path"]), "albedo_like")
            * _load_mask(_as_path(row["physics_core_skin_mask_path"])),
        }
        sample["valid_mask"] = _load_mask(_as_path(row["physics_core_skin_mask_path"]))
        return sample

    if spec.key == "p1_a":
        albedo = _load_npz_array(_as_path(row["maps_path"]), "albedo_like")
        mask = _load_mask(_as_path(row["physics_core_skin_mask_path"]))
        sample["representation"] = albedo * mask
        sample["valid_mask"] = mask
        return sample

    if spec.key == "p1_n":
        normal = _load_npz_array(_as_path(row["maps_path"]), "normal_coarse")
        mask = _load_mask(_as_path(row["face_valid_mask_path"]))
        sample["representation"] = normal * mask
        sample["valid_mask"] = mask
        return sample

    if spec.key == "p1_s":
        shading = _load_npz_array(_as_path(row["maps_path"]), "shading_like")
        mask = _load_mask(_as_path(row["face_valid_mask_path"]))
        sample["representation"] = shading * mask
        sample["valid_mask"] = mask
        return sample

    if spec.key == "p1_r":
        residual = _load_npz_array(_as_path(row["maps_path"]), "signed_residual")
        mask = _load_mask(_as_path(row["face_valid_mask_path"]))
        sample["representation"] = residual * mask
        sample["valid_mask"] = mask
        return sample

    raise ValueError(f"unsupported experiment key: {spec.key}")


def _iter_rows(frame_or_rows: Any) -> Iterable[Mapping[str, Any]]:
    if isinstance(frame_or_rows, pd.DataFrame):
        for row in frame_or_rows.to_dict(orient="records"):
            yield row
        return
    for row in frame_or_rows:
        if not isinstance(row, Mapping):
            raise TypeError("rows must be mappings or a DataFrame")
        yield row


def _stack_flattened_vectors(vectors: list[torch.Tensor]) -> torch.Tensor:
    return torch.stack([vector.reshape(-1).to(dtype=torch.float32) for vector in vectors], dim=0)


def _stack_masked_pixels(
    tensors: list[torch.Tensor],
    masks: list[torch.Tensor],
) -> torch.Tensor:
    pixels: list[torch.Tensor] = []
    for tensor, mask in zip(tensors, masks, strict=True):
        if tensor.ndim != 3:
            raise ValueError(f"expected CHW tensor, got {tuple(tensor.shape)}")
        mask_3d = mask.to(dtype=torch.bool)
        if mask_3d.ndim == 3 and mask_3d.shape[0] == 1:
            mask_2d = mask_3d[0]
        elif mask_3d.ndim == 2:
            mask_2d = mask_3d
        else:
            raise ValueError(f"unexpected mask shape: {tuple(mask.shape)}")
        for channel in tensor:
            selected = channel[mask_2d]
            if selected.numel():
                pixels.append(selected.to(dtype=torch.float32))
    if not pixels:
        raise ValueError("no valid pixels available for normalization fit")
    return torch.cat(pixels)


def _channel_quantiles(tensors: list[torch.Tensor], masks: list[torch.Tensor], q: float) -> torch.Tensor:
    if not tensors:
        raise ValueError("no tensors available for quantile fit")
    channel_values: list[list[torch.Tensor]] = [[] for _ in range(int(tensors[0].shape[0]))]
    for tensor, mask in zip(tensors, masks, strict=True):
        mask_2d = mask[0].to(dtype=torch.bool) if mask.ndim == 3 else mask.to(dtype=torch.bool)
        for channel_index in range(int(tensor.shape[0])):
            selected = tensor[channel_index][mask_2d]
            if selected.numel():
                channel_values[channel_index].append(selected.to(dtype=torch.float32))
    values: list[torch.Tensor] = []
    for selected_list in channel_values:
        if not selected_list:
            raise ValueError("no valid pixels available for normalization fit")
        values.append(torch.quantile(torch.cat(selected_list), q))
    if not values:
        raise ValueError("no valid pixels available for normalization fit")
    return torch.stack(values)


def fit_component_normalization(
    frame_or_rows: Any,
    experiment_key: str,
    *,
    fold: int | None = None,
    root: str | Path | None = None,
) -> ComponentNormalizationState:
    """Fit the fixed normalization contract for one experiment."""

    spec = get_component_spec(experiment_key)
    if spec.key == "p1_spec":
        return ComponentNormalizationState(
            experiment_key=spec.key,
            representation=spec.representation,
            strategy="unavailable",
            fold=fold,
            fitted_from="unavailable",
            normal_flip_status="UNVERIFIED",
            note="specular_like is not produced by the current frontend",
        )

    rows = list(_iter_rows(frame_or_rows))
    if fold is not None:
        rows = [row for row in rows if int(row["fold"]) == int(fold)]
    if not rows:
        raise ValueError("no rows available for normalization fitting")

    samples = [load_raw_component_sample(row, spec) for row in rows]
    representation = spec.representation

    if spec.key == "p1_l":
        vectors = _stack_flattened_vectors([sample["representation"] for sample in samples])
        mean = vectors.mean(dim=0)
        std = vectors.std(dim=0, unbiased=False)
        std = torch.where(std > 1.0e-8, std, torch.ones_like(std))
        return ComponentNormalizationState(
            experiment_key=spec.key,
            representation=representation,
            strategy="standardize_flatten_27",
            fold=fold,
            fitted_from="training_fold",
            channel_mean=tuple(float(x) for x in mean.tolist()),
            channel_std=tuple(float(x) for x in std.tolist()),
            source_case_count=len(rows),
            source_patient_group_count=len({str(row.get("group_id", row.get("patient_group_id", ""))) for row in rows}),
            source_fold_count=len({int(row["fold"]) for row in rows}),
            source_case_ids=tuple(str(row["case_id"]) for row in rows),
        )

    if spec.key == "p1_a":
        stacked = torch.stack([sample["representation"].to(dtype=torch.float32) for sample in samples], dim=0)
        outside = float(((stacked < 0.0) | (stacked > 1.0)).float().mean().item())
        return ComponentNormalizationState(
            experiment_key=spec.key,
            representation=representation,
            strategy="clip_0_1_and_imagenet_normalize",
            fold=fold,
            fitted_from="fixed_rule",
            clip_min=0.0,
            clip_max=1.0,
            out_of_range_fraction=outside,
            source_case_count=len(rows),
            source_patient_group_count=len({str(row.get("group_id", row.get("patient_group_id", ""))) for row in rows}),
            source_fold_count=len({int(row["fold"]) for row in rows}),
            source_case_ids=tuple(str(row["case_id"]) for row in rows),
        )

    if spec.key == "p1_n":
        audit = inspect_normal_flip_convention(root)
        if audit.status == "VERIFIED":
            flip_status = "VERIFIED_PHYSICAL_FLIP"
            flip_mode = "PHYSICAL_REFLECTION"
            flip_enabled = True
        else:
            flip_status = "SAFE_FLIP_DISABLED"
            flip_mode = "DISABLED_SAFE_FALLBACK"
            flip_enabled = False
        return ComponentNormalizationState(
            experiment_key=spec.key,
            representation=representation,
            strategy="fixed_masked_normal_01",
            fold=fold,
            fitted_from="fixed_rule",
            normal_flip_status=flip_status,
            normal_coordinate_status=audit.status,
            normal_flip_mode=flip_mode,
            horizontal_flip_enabled=flip_enabled,
            normal_flip_reason=audit.reason,
            source_case_count=len(rows),
            source_patient_group_count=len({str(row.get("group_id", row.get("patient_group_id", ""))) for row in rows}),
            source_fold_count=len({int(row["fold"]) for row in rows}),
            source_case_ids=tuple(str(row["case_id"]) for row in rows),
        )

    if spec.key == "p1_s":
        tensors = [sample["representation"].to(dtype=torch.float32) for sample in samples]
        masks = [sample["valid_mask"] for sample in samples]
        p01 = _channel_quantiles(tensors, masks, 0.01)
        p99 = _channel_quantiles(tensors, masks, 0.99)
        return ComponentNormalizationState(
            experiment_key=spec.key,
            representation=representation,
            strategy="clip_p01_p99_then_imagenet_normalize",
            fold=fold,
            fitted_from="training_fold",
            p01=tuple(float(x) for x in p01.tolist()),
            p99=tuple(float(x) for x in p99.tolist()),
            source_case_count=len(rows),
            source_patient_group_count=len({str(row.get("group_id", row.get("patient_group_id", ""))) for row in rows}),
            source_fold_count=len({int(row["fold"]) for row in rows}),
            source_case_ids=tuple(str(row["case_id"]) for row in rows),
        )

    if spec.key == "p1_r":
        tensors = [sample["representation"].to(dtype=torch.float32) for sample in samples]
        masks = [sample["valid_mask"] for sample in samples]
        scales = []
        for channel_index in range(3):
            channel_values: list[torch.Tensor] = []
            for tensor, mask in zip(tensors, masks, strict=True):
                mask_2d = mask[0].to(dtype=torch.bool) if mask.ndim == 3 else mask.to(dtype=torch.bool)
                selected = tensor[channel_index].abs()[mask_2d]
                if selected.numel():
                    channel_values.append(selected.to(dtype=torch.float32))
            if not channel_values:
                raise ValueError("no valid residual pixels available for normalization fit")
            scale = torch.quantile(torch.cat(channel_values), 0.99)
            scales.append(float(max(float(scale.item()), 1.0e-8)))
        return ComponentNormalizationState(
            experiment_key=spec.key,
            representation=representation,
            strategy="scale_abs_p99_then_clip_to_unit_interval",
            fold=fold,
            fitted_from="training_fold",
            residual_scale=tuple(scales),
            source_case_count=len(rows),
            source_patient_group_count=len({str(row.get("group_id", row.get("patient_group_id", ""))) for row in rows}),
            source_fold_count=len({int(row["fold"]) for row in rows}),
            source_case_ids=tuple(str(row["case_id"]) for row in rows),
        )

    if spec.key == "p1_rgb_a":
        return ComponentNormalizationState(
            experiment_key=spec.key,
            representation=representation,
            strategy="paired_rgb_and_albedo_transforms",
            fold=fold,
            fitted_from="fixed_rule",
            normal_flip_status="VERIFIED_FOR_RGB_ONLY",
            source_case_count=len(rows),
            source_patient_group_count=len({str(row.get("group_id", row.get("patient_group_id", ""))) for row in rows}),
            source_fold_count=len({int(row["fold"]) for row in rows}),
            source_case_ids=tuple(str(row["case_id"]) for row in rows),
        )

    raise ValueError(f"unsupported experiment key: {spec.key}")


def _imagenet_normalize(tensor: torch.Tensor) -> torch.Tensor:
    if tensor.ndim != 3 or tensor.shape[0] != 3:
        raise ValueError(f"expected CHW 3-channel tensor, got {tuple(tensor.shape)}")
    mean = IMAGENET_MEAN.to(device=tensor.device, dtype=tensor.dtype).view(3, 1, 1)
    std = IMAGENET_STD.to(device=tensor.device, dtype=tensor.dtype).view(3, 1, 1)
    return (tensor - mean) / std


def _clip_and_mask(tensor: torch.Tensor, mask: torch.Tensor, *, low: float, high: float) -> torch.Tensor:
    clipped = tensor.clamp(low, high)
    return clipped * mask.to(dtype=clipped.dtype)


def _rescale_to_unit_interval(tensor: torch.Tensor, low: torch.Tensor, high: torch.Tensor) -> torch.Tensor:
    denom = torch.where((high - low).abs() > 1.0e-8, high - low, torch.ones_like(high))
    return (tensor - low.view(-1, 1, 1)) / denom.view(-1, 1, 1)


def apply_component_normalization(
    sample: Mapping[str, Any],
    state: ComponentNormalizationState,
    *,
    horizontal_flip: bool = False,
) -> dict[str, Any]:
    """Apply the fixed normalization contract to a raw sample."""

    spec = get_component_spec(state.experiment_key)
    representation = sample["representation"]
    valid_mask = sample.get("valid_mask")

    if spec.key == "p1_spec":
        raise ValueError("P1-Spec is unavailable by current frontend")

    if spec.key == "p1_l":
        vector = torch.as_tensor(representation, dtype=torch.float32).reshape(-1)
        mean = torch.tensor(state.channel_mean, dtype=torch.float32)
        std = torch.tensor(state.channel_std, dtype=torch.float32)
        std = torch.where(std > 1.0e-8, std, torch.ones_like(std))
        return {**sample, "representation": (vector - mean) / std, "valid_mask": None}

    if spec.key == "p1_rgb_a":
        rgb = torch.as_tensor(representation["rgb"], dtype=torch.float32)
        albedo = torch.as_tensor(representation["albedo"], dtype=torch.float32)
        if horizontal_flip:
            rgb = rgb.flip(-1)
            albedo = albedo.flip(-1)
            if valid_mask is not None:
                valid_mask = torch.as_tensor(valid_mask, dtype=torch.float32).flip(-1)
        rgb = _imagenet_normalize(rgb)
        albedo = _imagenet_normalize(albedo.clamp(0.0, 1.0))
        return {**sample, "representation": {"rgb": rgb, "albedo": albedo}, "valid_mask": valid_mask}

    tensor = torch.as_tensor(representation, dtype=torch.float32)
    mask = torch.as_tensor(valid_mask, dtype=torch.float32) if valid_mask is not None else None

    if spec.key == "p1_a":
        if horizontal_flip:
            tensor = tensor.flip(-1)
            mask = mask.flip(-1) if mask is not None else None
        tensor = _clip_and_mask(tensor, mask, low=0.0, high=1.0)
        return {**sample, "representation": _imagenet_normalize(tensor), "valid_mask": mask}

    if spec.key == "p1_n":
        if horizontal_flip and state.normal_flip_status == "VERIFIED_PHYSICAL_FLIP":
            tensor = tensor.flip(-1)
            if mask is not None:
                mask = mask.flip(-1)
            if tensor.shape[0] < 3:
                raise ValueError("P1-N requires 3-channel normals")
            tensor = tensor.clone()
            tensor[0] = -tensor[0]
            tensor = tensor * 0.5 + 0.5
            tensor = tensor.clamp(0.0, 1.0)
        else:
            tensor = tensor * 0.5 + 0.5
        if mask is not None:
            tensor = tensor * mask
        return {**sample, "representation": _imagenet_normalize(tensor), "valid_mask": mask}

    if spec.key == "p1_s":
        if horizontal_flip:
            tensor = tensor.flip(-1)
            mask = mask.flip(-1) if mask is not None else None
        p01 = torch.tensor(state.p01, dtype=torch.float32)
        p99 = torch.tensor(state.p99, dtype=torch.float32)
        if p01.shape != p99.shape:
            raise ValueError("P1-S normalization state missing p01/p99")
        if torch.any(torch.isclose(p01, p99)):
            raise ValueError("P1-S p01 and p99 must differ for each channel")
        tensor = torch.stack(
            [
                torch.clamp(tensor[channel], float(p01[channel]), float(p99[channel]))
                for channel in range(3)
            ],
            dim=0,
        )
        tensor = _rescale_to_unit_interval(tensor, p01, p99)
        if mask is not None:
            tensor = tensor * mask
        return {**sample, "representation": _imagenet_normalize(tensor), "valid_mask": mask}

    if spec.key == "p1_r":
        if horizontal_flip:
            tensor = tensor.flip(-1)
            mask = mask.flip(-1) if mask is not None else None
        scale = torch.tensor(state.residual_scale, dtype=torch.float32)
        if scale.shape != (3,):
            raise ValueError("P1-R normalization state missing residual_scale")
        scaled = torch.stack(
            [torch.clamp(tensor[channel] / max(float(scale[channel]), 1.0e-8), -1.0, 1.0) for channel in range(3)],
            dim=0,
        )
        scaled = scaled * 0.5 + 0.5
        if mask is not None:
            scaled = scaled * mask
        normalized = _imagenet_normalize(scaled)
        if mask is not None:
            normalized = normalized * mask
        return {**sample, "representation": normalized, "valid_mask": mask}

    raise ValueError(f"unsupported experiment key: {spec.key}")


def state_to_json(state: ComponentNormalizationState) -> str:
    return json.dumps(state.to_dict(), indent=2, ensure_ascii=False)


def build_training_collate(
    experiment_key: str,
    state: ComponentNormalizationState,
    *,
    training: bool = False,
    horizontal_flip: bool = False,
    flip_probability: float = 0.5,
):
    """Return a collate function that normalizes a raw batch on the fly."""

    spec = get_component_spec(experiment_key)

    def _collate(batch: list[Mapping[str, Any]]) -> dict[str, Any]:
        normalized = []
        for sample in batch:
            do_flip = bool(
                training
                and horizontal_flip
                and float(torch.rand(1).item()) < float(flip_probability)
            )
            normalized.append(apply_component_normalization(sample, state, horizontal_flip=do_flip))
        out: dict[str, Any] = {
            "case_id": [sample["case_id"] for sample in normalized],
            "patient_group_id": [sample["patient_group_id"] for sample in normalized],
            "fold": torch.tensor([sample["fold"] for sample in normalized], dtype=torch.long),
            "label_original": torch.tensor([sample["label_original"] for sample in normalized], dtype=torch.long),
            "label_3class": torch.tensor([sample["label_3class"] for sample in normalized], dtype=torch.long),
            "label_binary": torch.tensor([sample["label_binary"] for sample in normalized], dtype=torch.long),
        }
        if spec.key == "p1_rgb_a":
            out["representation"] = {
                "rgb": torch.stack([sample["representation"]["rgb"] for sample in normalized], dim=0),
                "albedo": torch.stack([sample["representation"]["albedo"] for sample in normalized], dim=0),
            }
        else:
            out["representation"] = torch.stack([sample["representation"] for sample in normalized], dim=0)
        masks = [sample.get("valid_mask") for sample in normalized]
        out["valid_mask"] = None if all(mask is None for mask in masks) else torch.stack([mask for mask in masks if mask is not None], dim=0)
        return out

    return _collate


def summarize_normalization_state(state: ComponentNormalizationState) -> dict[str, Any]:
    payload = state.to_dict()
    if payload.get("normal_flip_status") == "UNVERIFIED":
        payload["hard_error"] = True
    else:
        payload["hard_error"] = False
    return payload
