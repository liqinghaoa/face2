"""Build S1-2 layered masks and QC for Hyper-Skin Train."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SOURCE_ROOT = PROJECT_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from skin_optics_hsi.s1_masks import DEFAULT_MASK_CONFIG, run_s1_2_masks  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--registration", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=DEFAULT_MASK_CONFIG)
    parser.add_argument("--split", choices=("train", "valid"), default="train")
    parser.add_argument("--max-samples", type=int, default=None, help="Train pilot only")
    parser.add_argument("--frozen-protocol-provenance", type=Path, default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.max_samples is not None and args.split != "train":
        raise ValueError("--max-samples is restricted to Train pilots")
    decision = run_s1_2_masks(
        args.contract,
        args.registration,
        args.output_root,
        config_path=args.config,
        split=args.split,
        max_samples=args.max_samples,
        frozen_protocol_provenance=args.frozen_protocol_provenance,
    )
    print(json.dumps(decision, ensure_ascii=False, indent=2))
    return 0 if decision["automatic_decision"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
