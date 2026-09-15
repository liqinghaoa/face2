from pathlib import Path
import argparse, sys
ROOT=Path(__file__).resolve().parents[2]; sys.path.insert(0,str(ROOT))
from src.skin_optics_so_r1.realface_input_contract_audit import run, write_failure
if __name__ == "__main__":
    p=argparse.ArgumentParser(); p.add_argument("--project-root",type=Path,default=ROOT); a=p.parse_args()
    try: print(run(a.project_root))
    except Exception as e: write_failure(a.project_root,e); raise
