"""Run SO-0 generation steps."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def main() -> None:
    """Run matrix, sweep, and report generation scripts."""

    here = Path(__file__).resolve().parent
    for script in ["build_colorchecker_matrices.py", "run_so0_forward_sweep.py", "generate_so0_forward_report.py"]:
        subprocess.run([sys.executable, str(here / script)], check=True)


if __name__ == "__main__":
    main()
