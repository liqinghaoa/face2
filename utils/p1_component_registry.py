"""Registry for the unified P1 component experiment family."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


EXPERIMENT_ORDER: tuple[str, ...] = (
    "p1_a",
    "p1_n",
    "p1_l",
    "p1_s",
    "p1_spec",
    "p1_r",
    "p1_rgb_a",
)


VALID_AVAILABILITIES = {"runnable", "unavailable_by_current_frontend"}
VALID_STAGE_TWO_STATES = {"RUNNABLE", "SKIPPED_UNAVAILABLE_BY_CURRENT_FRONTEND"}


@dataclass(frozen=True)
class P1ComponentSpec:
    key: str
    display_name: str
    enabled: bool
    availability: str
    representation: str
    input_type: str
    model_type: str
    mask: str | None = None
    flatten: bool = False
    stage_two_status: str = "RUNNABLE"
    skip_reason: str | None = None
    normalization_key: str | None = None
    required_assets: tuple[str, ...] = ()
    note: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


_REGISTRY: dict[str, P1ComponentSpec] = {
    "p1_a": P1ComponentSpec(
        key="p1_a",
        display_name="P1-A",
        enabled=True,
        availability="runnable",
        representation="albedo",
        input_type="image",
        model_type="resnet18_single",
        mask="physics_core_skin",
        normalization_key="p1_a",
        required_assets=("maps.npz", "physics_core_skin_mask_path"),
    ),
    "p1_n": P1ComponentSpec(
        key="p1_n",
        display_name="P1-N",
        enabled=True,
        availability="runnable",
        representation="normal",
        input_type="image",
        model_type="resnet18_single",
        mask="face_valid",
        normalization_key="p1_n",
        required_assets=("maps.npz", "face_valid_mask_path"),
        note="normal_flip_mode defaults to DISABLED_SAFE_FALLBACK until coordinate convention is verified.",
    ),
    "p1_l": P1ComponentSpec(
        key="p1_l",
        display_name="P1-L",
        enabled=True,
        availability="runnable",
        representation="light",
        input_type="vector",
        model_type="linear_probe",
        flatten=True,
        normalization_key="p1_l",
        required_assets=("latents.npz",),
    ),
    "p1_s": P1ComponentSpec(
        key="p1_s",
        display_name="P1-S",
        enabled=True,
        availability="runnable",
        representation="shading",
        input_type="image",
        model_type="resnet18_single",
        mask="face_valid",
        normalization_key="p1_s",
        required_assets=("maps.npz", "face_valid_mask_path"),
    ),
    "p1_spec": P1ComponentSpec(
        key="p1_spec",
        display_name="P1-Spec",
        enabled=False,
        availability="unavailable_by_current_frontend",
        representation="specular",
        input_type="unavailable",
        model_type="none",
        stage_two_status="SKIPPED_UNAVAILABLE_BY_CURRENT_FRONTEND",
        skip_reason="specular_like is not produced by the current frontend",
        normalization_key="p1_spec",
        required_assets=(),
    ),
    "p1_r": P1ComponentSpec(
        key="p1_r",
        display_name="P1-R",
        enabled=True,
        availability="runnable",
        representation="residual",
        input_type="image",
        model_type="resnet18_single",
        mask="face_valid",
        normalization_key="p1_r",
        required_assets=("maps.npz", "face_valid_mask_path"),
    ),
    "p1_rgb_a": P1ComponentSpec(
        key="p1_rgb_a",
        display_name="P1-RGB+A",
        enabled=True,
        availability="runnable",
        representation="rgb_albedo",
        input_type="dual_image",
        model_type="shared_resnet18_dual_branch",
        mask="physics_core_skin",
        normalization_key="p1_rgb_a",
        required_assets=("rgb_path", "maps.npz", "physics_core_skin_mask_path"),
    ),
}


def normalize_experiment_key(value: str) -> str:
    normalized = str(value).strip().lower().replace("-", "_").replace(" ", "_").replace("+", "_")
    if normalized in _REGISTRY:
        return normalized
    aliases = {
        "p1a": "p1_a",
        "p1n": "p1_n",
        "p1l": "p1_l",
        "p1s": "p1_s",
        "p1spec": "p1_spec",
        "p1r": "p1_r",
        "p1rgba": "p1_rgb_a",
        "p1_rgba": "p1_rgb_a",
    }
    if normalized in aliases:
        return aliases[normalized]
    raise KeyError(f"unknown P1 component experiment: {value!r}")


def get_component_registry() -> dict[str, P1ComponentSpec]:
    return dict(_REGISTRY)


def get_component_spec(experiment_key: str) -> P1ComponentSpec:
    return _REGISTRY[normalize_experiment_key(experiment_key)]


def list_component_specs() -> list[P1ComponentSpec]:
    return [_REGISTRY[key] for key in EXPERIMENT_ORDER]


def runnable_component_keys() -> tuple[str, ...]:
    return tuple(
        key
        for key in EXPERIMENT_ORDER
        if _REGISTRY[key].enabled and _REGISTRY[key].availability == "runnable"
    )


def component_status_map() -> dict[str, str]:
    return {key: _REGISTRY[key].stage_two_status for key in EXPERIMENT_ORDER}


def validate_component_registry() -> dict[str, Any]:
    missing = [key for key in EXPERIMENT_ORDER if key not in _REGISTRY]
    if missing:
        raise ValueError(f"registry is missing experiments: {missing}")
    duplicate_display_names = {
        spec.display_name
        for spec in _REGISTRY.values()
        if list(spec.display_name for spec in _REGISTRY.values()).count(spec.display_name) > 1
    }
    if duplicate_display_names:
        raise ValueError(f"duplicate display names: {sorted(duplicate_display_names)}")
    for key, spec in _REGISTRY.items():
        if spec.key != key:
            raise ValueError(f"spec key mismatch for {key!r}")
        if spec.availability not in VALID_AVAILABILITIES:
            raise ValueError(f"invalid availability for {key!r}: {spec.availability!r}")
        if spec.stage_two_status not in VALID_STAGE_TWO_STATES:
            raise ValueError(f"invalid stage_two_status for {key!r}: {spec.stage_two_status!r}")
    if _REGISTRY["p1_spec"].availability != "unavailable_by_current_frontend":
        raise ValueError("P1-Spec must remain unavailable_by_current_frontend")
    return {
        "status": "passed",
        "experiment_order": list(EXPERIMENT_ORDER),
        "runnable_experiments": list(runnable_component_keys()),
        "specular_status": _REGISTRY["p1_spec"].stage_two_status,
    }
