from __future__ import annotations

import argparse
import json
from pathlib import Path

from _bootstrap import ROOT, load_yaml_config, resolve_output_dir
from utils.p1_component_preflight import validate_framework


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=ROOT / "config/p1/p1_component_sweep_v1.yaml")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "experiments/500Data/P1_Component_Sweep_v1")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_yaml_config(args.config)
    config["root"] = str(ROOT)
    output_dir = resolve_output_dir(args.output_dir)
    result = validate_framework(config, output_dir)
    (output_dir / "metadata").mkdir(parents=True, exist_ok=True)
    (output_dir / "metadata" / "framework_validation_summary.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
