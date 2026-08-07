"""Build the P0-B1 experimental-condition blinded reviewer package."""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from p0b_deca.blind_review import build


if __name__ == "__main__":
    print(json.dumps(build(ROOT), ensure_ascii=False, indent=2))
