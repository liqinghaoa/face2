"""One-command entry point for R3DPR binary ResNet18/34/50 experiments."""

from __future__ import annotations

import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.run.R3DPR.run_r3dpr_resnet18_binary_5fold import main


if __name__ == "__main__":
    main()
