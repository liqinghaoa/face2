"""Finalize an approved S1-2 target-domain profile for S1-3 development."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SOURCE_ROOT = PROJECT_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from skin_optics_hsi.s1_target_profile import finalize_target_domain_profile  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--decision-file", type=Path, required=True)
    parser.add_argument("--reviewer", required=True)
    parser.add_argument("--notes", required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    result = finalize_target_domain_profile(args.decision_file, reviewer=args.reviewer, notes=args.notes)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
