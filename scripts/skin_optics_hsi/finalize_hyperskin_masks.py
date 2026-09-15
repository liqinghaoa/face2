"""Record S1-2 Train visual review and freeze or reject the mask protocol."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SOURCE_ROOT = PROJECT_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from skin_optics_hsi.s1_masks import finalize_s1_2_train_review, finalize_s1_2_validation_review  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--decision-file", type=Path, required=True)
    parser.add_argument("--reviewer", required=True)
    parser.add_argument("--decision", choices=("pass", "fail"), required=True)
    parser.add_argument("--notes", required=True)
    parser.add_argument("--split", choices=("train", "valid"), default="train")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    finalizer = finalize_s1_2_train_review if args.split == "train" else finalize_s1_2_validation_review
    result = finalizer(args.decision_file, reviewer=args.reviewer, approve=args.decision == "pass", notes=args.notes)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
