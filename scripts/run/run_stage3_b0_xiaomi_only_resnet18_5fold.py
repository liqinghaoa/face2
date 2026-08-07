"""Run the Stage3-B0 Xiaomi-only ResNet18 group five-fold experiment."""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from stage3_b0_xiaomi_only.pipeline import main


if __name__ == "__main__":
    main()
