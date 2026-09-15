"""Run the preregistered KM-BIO-v2R R-C0 candidate ladder."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.skin_optics_hsi.km_bio_v2r_stage_c0 import run_stage_c0  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default=str(ROOT / "configs/skin_optics_hsi/km_bio_v2r_candidate_ladder.yaml"),
    )
    args = parser.parse_args()
    result = run_stage_c0(args.config, ROOT)
    print(json.dumps({
        "status": result["status"],
        "executed_candidates": result["executed_candidates"],
        "stop_reason": result["stop_reason"],
        "validation_authorized": result["validation_authorized"],
    }, indent=2))


if __name__ == "__main__":
    main()
