from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from _bootstrap import ROOT, resolve_output_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=ROOT / "experiments/500Data/P1_Component_Sweep_v1")
    return parser.parse_args()


def main() -> None:
    output_dir = resolve_output_dir(parse_args().output_dir)
    metadata = output_dir / "metadata"
    summary = output_dir / "summary"
    payload = {
        "framework_manifest": json.loads((metadata / "framework_manifest.json").read_text(encoding="utf-8")) if (metadata / "framework_manifest.json").is_file() else None,
        "execution_plan": json.loads((metadata / "execution_plan.json").read_text(encoding="utf-8")) if (metadata / "execution_plan.json").is_file() else None,
        "preflight_validation": json.loads((metadata / "preflight_validation.json").read_text(encoding="utf-8")) if (metadata / "preflight_validation.json").is_file() else None,
        "experiment_status": json.loads((metadata / "experiment_status.json").read_text(encoding="utf-8")) if (metadata / "experiment_status.json").is_file() else None,
        "phase2_preflight": json.loads((metadata / "phase2_preflight.json").read_text(encoding="utf-8")) if (metadata / "phase2_preflight.json").is_file() else None,
        "phase2_final_status": json.loads((metadata / "phase2_final_status.json").read_text(encoding="utf-8")) if (metadata / "phase2_final_status.json").is_file() else None,
        "main_results": pd.read_csv(summary / "p1_component_main_results.csv").to_dict(orient="records") if (summary / "p1_component_main_results.csv").is_file() else None,
    }
    (metadata / "sweep_summary.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
