from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
for rel in ("src", "src/skin_optics_so0/src"):
    path = str(ROOT / rel)
    if path not in sys.path:
        sys.path.insert(0, path)

