"""CLI for the real-HSI stage-one physics gate."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SOURCE_ROOT = PROJECT_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from skin_optics_hsi.stage1_pipeline import run_stage1  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fit the candidate two-proxy skin model to Hyper-Skin VIS region spectra."
    )
    parser.add_argument("--data-root", type=Path, required=True, help="Hyper-Skin(RGB, VIS) root")
    parser.add_argument("--mask-root", type=Path, required=True, help="Matching binary skin masks")
    parser.add_argument("--output-dir", type=Path, required=True, help="New, non-overwriting output directory")
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--mode", choices=("development", "formal-test"), default="development")
    parser.add_argument("--max-samples", type=int, default=None, help="Development smoke runs only")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    decision = run_stage1(
        data_root=args.data_root,
        mask_root=args.mask_root,
        output_dir=args.output_dir,
        config_path=args.config,
        mode=args.mode,
        max_samples=args.max_samples,
    )
    print(json.dumps(decision, ensure_ascii=False, indent=2))
    return 0 if decision["decision"] != "STOP" else 2


if __name__ == "__main__":
    raise SystemExit(main())

