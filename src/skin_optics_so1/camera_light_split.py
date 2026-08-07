"""Qualified camera-light loading and deterministic SO-1 split creation."""

from __future__ import annotations

import csv
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path


FAILED_PAIRS = {
    ("Point Grey Grasshopper 50S5C", "D65"),
    ("Point Grey Grasshopper 50S5C", "A"),
    ("Point Grey Grasshopper 50S5C", "FL2"),
}


@dataclass(frozen=True)
class CameraLightPair:
    camera_name: str
    light_name: str

    @property
    def key(self) -> str:
        return f"{self.camera_name} / {self.light_name}"


@dataclass(frozen=True)
class CameraLightSplit:
    seen_cameras: list[str]
    unseen_cameras: list[str]
    seen_lights: list[str]
    unseen_lights: list[str]
    qualified_pairs: list[CameraLightPair]
    excluded_pairs: list[str]

    def allowed_pairs(self, split: str) -> list[CameraLightPair]:
        if split in {"train", "validation", "id_test"}:
            cams, lights = self.seen_cameras, self.seen_lights
        elif split == "camera_ood":
            cams, lights = self.unseen_cameras, self.seen_lights
        elif split == "light_ood":
            cams, lights = self.seen_cameras, self.unseen_lights
        elif split == "joint_ood":
            cams, lights = self.unseen_cameras, self.unseen_lights
        else:
            raise ValueError(f"Unknown split {split!r}")
        allowed = [p for p in self.qualified_pairs if p.camera_name in cams and p.light_name in lights]
        if not allowed:
            raise ValueError(f"No allowed qualified pairs for {split}")
        return allowed


def _stable_sort_key(seed: int, name: str) -> str:
    return hashlib.sha256(f"{int(seed)}|{name}".encode("utf-8")).hexdigest()


def load_qualified_pairs(so0_output_dir: str | Path = "outputs/SO0_Forward_Model_v1.1") -> list[CameraLightPair]:
    path = Path(so0_output_dir) / "tables" / "colorchecker_calibration_metrics.csv"
    if not path.exists():
        raise FileNotFoundError(f"SO-0 ColorChecker audit CSV not found: {path}")
    pairs: list[CameraLightPair] = []
    with path.open("r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            pair = (row["camera_name"], row["illuminant_name"])
            if row["qualification_status"] == "PASS":
                if pair in FAILED_PAIRS:
                    raise ValueError(f"Failed SO-0 pair unexpectedly marked PASS: {pair}")
                pairs.append(CameraLightPair(pair[0], pair[1]))
    if len(pairs) != 109:
        raise ValueError(f"Expected 109 qualified SO-0 pairs, got {len(pairs)}")
    return pairs


def build_camera_light_split(global_seed: int, so0_output_dir: str | Path = "outputs/SO0_Forward_Model_v1.1") -> CameraLightSplit:
    pairs = load_qualified_pairs(so0_output_dir)
    lights = ["D65", "A", "FL2", "FL11"]
    by_camera: dict[str, set[str]] = {}
    for p in pairs:
        by_camera.setdefault(p.camera_name, set()).add(p.light_name)
    candidates = [
        cam
        for cam, ls in by_camera.items()
        if all(light in ls for light in lights)
        and cam not in {"Canon 5DMarkII", "Canon 5D Mark II", "Point Grey Grasshopper 50S5C"}
    ]
    candidates = sorted(candidates, key=lambda c: _stable_sort_key(global_seed, c))
    unseen = sorted(candidates[:3])
    all_cameras = sorted(by_camera)
    seen = sorted([c for c in all_cameras if c not in unseen])
    if len(seen) != 25 or len(unseen) != 3:
        raise ValueError(f"Expected 25 seen and 3 unseen cameras, got {len(seen)} seen/{len(unseen)} unseen")
    if "Canon 5DMarkII" not in seen:
        raise ValueError("Canon 5DMarkII must be seen")
    if "Point Grey Grasshopper 50S5C" in unseen:
        raise ValueError("Point Grey Grasshopper 50S5C must not be unseen")
    split = CameraLightSplit(
        seen_cameras=seen,
        unseen_cameras=unseen,
        seen_lights=["D65", "A", "FL2"],
        unseen_lights=["FL11"],
        qualified_pairs=sorted(pairs, key=lambda p: (p.camera_name, p.light_name)),
        excluded_pairs=[f"{c} / {l}" for c, l in sorted(FAILED_PAIRS)],
    )
    for name in ("train", "validation", "id_test", "camera_ood", "light_ood", "joint_ood"):
        for p in split.allowed_pairs(name):
            if (p.camera_name, p.light_name) in FAILED_PAIRS:
                raise ValueError(f"Failed pair present in {name}: {p.key}")
    return split


def save_camera_light_split(
    split: CameraLightSplit,
    path: str | Path,
    overwrite: bool = False,
    allow_existing_identical: bool = False,
) -> str:
    path = Path(path)
    payload = {
        "seen_cameras": split.seen_cameras,
        "unseen_cameras": split.unseen_cameras,
        "seen_lights": split.seen_lights,
        "unseen_lights": split.unseen_lights,
        "qualified_pairs": [p.key for p in split.qualified_pairs],
        "excluded_pairs": split.excluded_pairs,
    }
    digest = hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()
    if path.exists() and not overwrite:
        if allow_existing_identical:
            existing = json.loads(path.read_text(encoding="utf-8"))
            if existing == payload:
                return digest
            raise ValueError(f"Existing camera-light split differs from current deterministic split: {path}")
        raise FileExistsError(f"Refusing to overwrite frozen camera-light split: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False)
    path.write_text(text + "\n", encoding="utf-8")
    return digest


def shuffled_balanced_pairs(pairs: list[CameraLightPair], count: int, seed: int) -> list[CameraLightPair]:
    order = sorted(pairs, key=lambda p: _stable_sort_key(seed, p.key))
    if not order:
        raise ValueError("No pairs to assign")
    return [order[i % len(order)] for i in range(count)]
