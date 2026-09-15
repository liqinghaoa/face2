"""CLI for S1-3 Hyper-Skin regional spectrum extraction."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SOURCE_ROOT = PROJECT_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from skin_optics_hsi.s1_region_spectra import DEFAULT_CONFIG, extract_s1_3_region_spectra  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", required=True)
    parser.add_argument("--mask-manifest", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--target-decision")
    parser.add_argument("--registration-decision")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    args = parser.parse_args()
    decision = extract_s1_3_region_spectra(
        args.contract,
        args.mask_manifest,
        args.output_root,
        target_decision_path=args.target_decision,
        registration_decision_path=args.registration_decision,
        config_path=args.config,
    )
    print(json.dumps(decision, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
