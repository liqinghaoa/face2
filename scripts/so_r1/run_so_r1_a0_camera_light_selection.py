#!/usr/bin/env python
"""CLI for SO-R1-A0 camera/light audit and deterministic split selection."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from skin_optics_so_r1.camera_light_selection import (  # noqa: E402
    SelectionError,
    load_config,
    run_selection,
    validate_inputs,
    verify_frozen,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="Repository-relative YAML configuration")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--validate-only", action="store_true")
    mode.add_argument("--run", action="store_true")
    mode.add_argument("--verify-frozen", action="store_true")
    args = parser.parse_args()
    try:
        config = load_config(ROOT / args.config)
        if args.validate_only:
            result = validate_inputs(ROOT, config)
            print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
        elif args.run:
            result = run_selection(ROOT, config)
            print(result["terminal_summary"])
        else:
            result = verify_frozen(ROOT, config)
            print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    except SelectionError as exc:
        print(f"SO-R1-A0 validation failed: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
