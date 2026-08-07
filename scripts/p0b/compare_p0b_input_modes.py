"""Run the live B0 gate, then the frozen P0-B1 input-mode comparison."""
from __future__ import annotations

import argparse
import gc
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from p0b_deca.config import load_config
from p0b_deca.environment_audit import audit_environment
from p0b_deca.input_mode_comparison import run_input_mode_comparison


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=ROOT / "config/p0b/p0b_deca_pilot12_v1.yaml")
    args = parser.parse_args()
    config = load_config(args.config, ROOT)
    audit = audit_environment(config)
    if not audit["passed"]:
        print({"status": "failed_environment", "reason": audit["block_reason"]})
        return 2
    gc.collect()
    import torch

    torch.cuda.empty_cache()
    summary = run_input_mode_comparison(config)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0 if summary["status"] == "passed" else 3


if __name__ == "__main__":
    raise SystemExit(main())
