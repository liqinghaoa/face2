"""CLI for S1-5R revised Train-only development."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.skin_optics_hsi.s1_revised_train import run_s1_5r


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/skin_optics_hsi/s1_5_revised_train_v3.yaml")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--supersedes-decision")
    args = parser.parse_args()
    decision = run_s1_5r(args.config, args.output_dir, supersedes_decision=args.supersedes_decision)
    print(json.dumps({"status": decision["status"], "next_stage_allowed": decision["next_stage_allowed"],
                      "output_dir": str(Path(args.output_dir).resolve())}, indent=2))


if __name__ == "__main__":
    main()
