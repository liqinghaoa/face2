"""Build an additional stratified S1-2 review sheet from an existing manifest."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SOURCE_ROOT = PROJECT_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from skin_optics_hsi.s1_masks import _save_contact_sheet, _select_manual_review_rows  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-image", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--maximum", type=int, default=24)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.maximum < 6:
        raise ValueError("--maximum must be at least 6 to cover all Train strata")
    for output in (args.output_image, args.output_csv):
        if output.exists():
            raise FileExistsError(f"Refusing to overwrite review artifact: {output}")
    frame = pd.read_parquet(args.manifest)
    selected = _select_manual_review_rows(frame, args.maximum).copy()
    args.output_image.parent.mkdir(parents=True, exist_ok=True)
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    _save_contact_sheet(args.output_image, selected)
    selected[
        [
            "sample_id",
            "subject_id",
            "expression",
            "direction",
            "status",
            "failure_codes",
            "review_codes",
            "final_to_anatomical_fraction",
            "largest_component_fraction",
            "qc_panel_path",
        ]
    ].to_csv(args.output_csv, index=False, encoding="utf-8-sig")
    print(f"Wrote {len(selected)} review cases to {args.output_image}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
