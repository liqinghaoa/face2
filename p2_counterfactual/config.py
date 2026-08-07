from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from utils.experiment_utils import PROJECT_ROOT, load_yaml
from p2_counterfactual.path_utils import DEFAULT_KNOWN_PROJECT_ROOTS, path_resolution_options, resolve_project_path


CONFIG_REQUIRED_TOP_LEVEL = {"experiment_id", "project_root", "manifest_path", "output_root", "model", "training", "data", "evaluation"}
ALLOWED_INPUT_MODES = {"original", "relight_mix", "paired", "full6_consistency", "pairwise_consistency"}


def resolve_p2_a_config(path: str | Path) -> dict[str, Any]:
    config = load_yaml(path)
    missing = sorted(CONFIG_REQUIRED_TOP_LEVEL - set(config))
    if missing:
        raise ValueError(f"P2-A config missing required top-level fields: {missing}")
    config = dict(config)
    config["config_path"] = str(Path(path).resolve())
    project_root = resolve_project_path(config.get("project_root", PROJECT_ROOT), PROJECT_ROOT)
    config["project_root"] = str(project_root)
    config["manifest_path"] = str(resolve_project_path(config["manifest_path"], project_root))
    config["output_root"] = str(resolve_project_path(config["output_root"], project_root))
    smoke_root = config.get("smoke_output_root")
    if smoke_root:
        config["smoke_output_root"] = str(resolve_project_path(smoke_root, project_root))
    else:
        config["smoke_output_root"] = str(project_root / "experiments/500Data/P2_Physics_Relighting_v2/_framework_smoke")
    path_cfg = path_resolution_options(config.get("path_resolution", {}))
    if not path_cfg["known_project_roots"]:
        path_cfg["known_project_roots"] = DEFAULT_KNOWN_PROJECT_ROOTS
    if path_cfg["rewrite_manifest"]:
        raise ValueError("P2-A config must not enable path_resolution.rewrite_manifest")
    config["path_resolution"] = path_cfg
    if str(config.get("input_mode")) not in ALLOWED_INPUT_MODES:
        raise ValueError(f"Unsupported input_mode: {config.get('input_mode')}")
    data_cfg = dict(config.get("data", {}))
    if data_cfg.get("original_rgb_override_dir"):
        data_cfg["original_rgb_override_dir"] = str(resolve_project_path(data_cfg["original_rgb_override_dir"], project_root))
    config["data"] = data_cfg
    if bool(config.get("consistency_enabled")) != (str(config.get("input_mode")) in {"paired", "full6_consistency", "pairwise_consistency"}):
        if bool(config.get("consistency_enabled")):
            raise ValueError("consistency_enabled=true is only valid for paired consistency input modes")
    if config.get("pair_cycle_schedule_path"):
        config["pair_cycle_schedule_path"] = str(resolve_project_path(config["pair_cycle_schedule_path"], project_root))
    return config


def config_fingerprint(config: dict[str, Any]) -> str:
    ignored = {"config_path"}
    payload = {key: value for key, value in config.items() if key not in ignored}
    text = json.dumps(payload, ensure_ascii=False, sort_keys=True, default=str)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()
