"""Incremental P0-A boundary-ambiguity finalization entry point."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from p0_physics_assets.finalization import finalize_boundary_ambiguity


def main(argv: Sequence[str] | None = None) -> int:
    """Finalize only existing P0 output; no model inference is performed."""
    parser = argparse.ArgumentParser(description="Finalize existing P0-A boundary ambiguity assets")
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    args = parser.parse_args(argv)
    print(finalize_boundary_ambiguity(args.project_root.resolve()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
