from __future__ import annotations
import argparse, json, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from p0b_deca.p1_audit import run
p = argparse.ArgumentParser(); p.add_argument('--config', type=Path, default=ROOT/'config/p1/p1_deca_full500_v1.yaml'); a = p.parse_args()
print(json.dumps(run(ROOT, a.config), indent=2))
