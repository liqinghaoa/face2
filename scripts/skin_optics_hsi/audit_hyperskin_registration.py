"""Command-line entry point for the Train-only S1-1 registration audit."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SOURCE_ROOT = PROJECT_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from skin_optics_hsi.s1_registration import run_s1_1_registration  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit RGB-HSI spatial transforms on stratified Train samples.")
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--samples-per-stratum", type=int, default=3)
    parser.add_argument("--working-size", type=int, default=256)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    decision = run_s1_1_registration(
        args.contract,
        args.output_root,
        samples_per_stratum=args.samples_per_stratum,
        working_size=args.working_size,
    )
    print(json.dumps(decision, ensure_ascii=False, indent=2))
    return 0 if decision["automatic_decision"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
