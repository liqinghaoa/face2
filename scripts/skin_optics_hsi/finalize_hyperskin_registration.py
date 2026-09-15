"""Record the researcher's S1-1 visual decision and freeze the mapping."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SOURCE_ROOT = PROJECT_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from skin_optics_hsi.s1_registration import finalize_s1_1_manual_review  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Finalize S1-1 after visual inspection of the review panels.")
    parser.add_argument("--registration-dir", type=Path, required=True)
    parser.add_argument("--reviewer", required=True)
    parser.add_argument("--decision", choices=("pass", "fail"), required=True)
    parser.add_argument("--notes", required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    decision = finalize_s1_1_manual_review(
        args.registration_dir,
        reviewer=args.reviewer,
        approve=args.decision == "pass",
        notes=args.notes,
    )
    print(json.dumps(decision, ensure_ascii=False, indent=2))
    return 0 if decision["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
