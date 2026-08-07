from __future__ import annotations
import argparse,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from p0b_deca.config import load_config
from p0b_deca.noncollapse_audit import run
p=argparse.ArgumentParser();p.add_argument('--config',type=Path,default=ROOT/'config/p0b/p0b_deca_pilot12_v1.yaml');a=p.parse_args();print(json.dumps(run(load_config(a.config,ROOT)),ensure_ascii=False,indent=2))
