"""Run KM-BIO-v2R R-B bilateral symmetric observation contract."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.skin_optics_hsi.km_bio_v2r_observation import build_symmetric_observation  # noqa: E402


def main() -> None:
    contract = ROOT / "configs/skin_optics_hsi/km_bio_v2r_observation_contract.yaml"
    result = build_symmetric_observation(contract, ROOT)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
