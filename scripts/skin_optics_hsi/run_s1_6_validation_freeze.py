from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.skin_optics_hsi.s1_validation_freeze import run_s1_6


def main() -> None:
    parser = argparse.ArgumentParser(description="Run S1-6 Validation selection and freeze the Stage-1 representation")
    parser.add_argument("--config", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    decision = run_s1_6(args.config, args.output_dir)
    print(json.dumps(decision, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
