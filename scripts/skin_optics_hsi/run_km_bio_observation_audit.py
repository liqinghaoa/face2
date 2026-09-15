"""Run KM-BIO-v1 Stage B observation-contract audit on Train only."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.skin_optics_hsi.km_bio_observation import run_observation_audit  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--contract",
        default=str(ROOT / "configs/skin_optics_hsi/km_bio_v1_observation_contract.yaml"),
    )
    args = parser.parse_args()
    result = run_observation_audit(args.contract, ROOT)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

