from __future__ import annotations

import ast
import json
from pathlib import Path


REQUIRED_NAMES = (
    "background",
    "skin",
    "left_brow",
    "right_brow",
    "left_eye",
    "right_eye",
    "eyeglass",
    "left_ear",
    "right_ear",
    "earring",
    "nose",
    "mouth",
    "upper_lip",
    "lower_lip",
    "neck",
    "necklace",
    "cloth",
    "hair",
    "hat",
)


def load_parsing_class_map(project_root: Path) -> dict[str, int]:
    source = project_root / "preprocessing" / "build_global_face_parsing_regularmask_blackbg_224_png_strict.py"
    tree = ast.parse(source.read_text(encoding="utf-8"))
    mapping: dict[int, str] | None = None
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == "CELEBAMASK_HQ_CLASSES":
                    mapping = ast.literal_eval(node.value)
    if not mapping:
        raise ValueError("CELEBAMASK_HQ_CLASSES not found in legacy parser source")
    name_to_id = {str(name): int(idx) for idx, name in mapping.items()}
    name_to_id["eye_glasses"] = name_to_id["eyeglass"]
    missing = [name for name in REQUIRED_NAMES if name not in name_to_id]
    if missing:
        raise ValueError(f"parsing class map missing required names: {missing}")
    return name_to_id


def write_class_map(path: Path, class_map: dict[str, int]) -> None:
    payload = {
        "source": "preprocessing/build_global_face_parsing_regularmask_blackbg_224_png_strict.py::CELEBAMASK_HQ_CLASSES",
        "class_name_to_id": class_map,
        "semantic_groups": {
            "skin": ["skin"],
            "nose": ["nose"],
            "brow": ["left_brow", "right_brow"],
            "eye": ["left_eye", "right_eye"],
            "mouth_lip": ["mouth", "upper_lip", "lower_lip"],
            "hair": ["hair"],
            "ear": ["left_ear", "right_ear", "earring"],
            "neck": ["neck", "necklace"],
            "cloth": ["cloth"],
            "background": ["background"],
            "excluded_from_core_skin": [
                "left_brow",
                "right_brow",
                "left_eye",
                "right_eye",
                "mouth",
                "upper_lip",
                "lower_lip",
                "hair",
                "left_ear",
                "right_ear",
                "earring",
                "neck",
                "necklace",
                "cloth",
                "hat",
                "background",
                "eye_glasses",
            ],
        },
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
