"""Command-line entry point for S1-0 Hyper-Skin contract construction."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SOURCE_ROOT = PROJECT_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from skin_optics_hsi.s1_contract import build_s1_0_contract  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build the evidence-labelled S1-0 Hyper-Skin data contract.")
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--value-scan", choices=("none", "sampled", "full"), default="full")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    contract = build_s1_0_contract(
        args.data_root,
        args.output_root,
        value_scan=args.value_scan,
        progress=lambda message: print(message, flush=True),
    )
    print(json.dumps({"status": contract["status"], "readiness": contract["readiness"]}, ensure_ascii=False, indent=2))
    return 0 if contract["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
