from __future__ import annotations

import argparse
import json
from pathlib import Path

from _bootstrap import ROOT, resolve_output_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=ROOT / "experiments/500Data/P1_Component_Sweep_v1")
    return parser.parse_args()


def main() -> None:
    output_dir = resolve_output_dir(parse_args().output_dir)
    metadata = output_dir / "metadata"
    validation = output_dir / "validation"
    payload = {
        "framework_manifest": json.loads((metadata / "framework_manifest.json").read_text(encoding="utf-8")) if (metadata / "framework_manifest.json").is_file() else None,
        "execution_plan": json.loads((metadata / "execution_plan.json").read_text(encoding="utf-8")) if (metadata / "execution_plan.json").is_file() else None,
        "experiment_status": json.loads((metadata / "experiment_status.json").read_text(encoding="utf-8")) if (metadata / "experiment_status.json").is_file() else None,
        "validation_files": sorted(str(path.name) for path in validation.glob("*.json")),
    }
    (metadata / "component_summary.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
