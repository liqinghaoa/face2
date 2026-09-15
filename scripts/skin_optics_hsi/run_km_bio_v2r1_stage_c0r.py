"""Run the KM-BIO-v2R.1 R-C0R resumed PS/PSG candidate ladder."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.skin_optics_hsi.km_bio_v2r1_stage_c0r import ContractError, run_stage_c0r  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default=str(ROOT / "configs/skin_optics_hsi/km_bio_v2r1_candidate_ladder_resume.yaml"),
    )
    args = parser.parse_args()
    try:
        result = run_stage_c0r(args.config, ROOT)
    except (ContractError, FileExistsError) as error:
        print(json.dumps({
            "status": "R_C0R_BLOCKED_BY_IMPLEMENTATION_OR_INTEGRITY",
            "error": str(error),
        }, ensure_ascii=False, indent=2))
        raise SystemExit(2) from error
    print(json.dumps({
        "status": result["status"], "selected_candidate": result["selected_candidate"],
        "new_candidates_executed": result["new_candidates_executed"],
        "validation_executed": result["validation_executed"],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
