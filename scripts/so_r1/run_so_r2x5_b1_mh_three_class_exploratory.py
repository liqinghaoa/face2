"""Entry point for the independent SO-R2-X5 B1 M/H-only exploratory experiment."""
from __future__ import annotations
import json, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.skin_optics_so_r1.realface_b1_mh_three_class_exploratory import run
if __name__ == "__main__": print(json.dumps(run(ROOT), ensure_ascii=True, sort_keys=True))
