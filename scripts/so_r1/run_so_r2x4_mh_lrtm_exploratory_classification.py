"""Run the locked, internal-only SO-R2-X4 nested-direct experiment."""
from __future__ import annotations
import json, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.skin_optics_so_r1.mh_lrtm_exploratory_training import run

if __name__ == "__main__":
    try:
        print(json.dumps(run(ROOT), ensure_ascii=True, sort_keys=True))
    except Exception as exc:
        print("SO_R2_X4_FAIL", type(exc).__name__, str(exc)[:500])
        raise
