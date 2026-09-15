from pathlib import Path
import argparse
import sys

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from src.skin_optics_so_r1.proposed_training import run, write_b2_failure

if __name__ == "__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--project-root",type=Path,default=ROOT)
    parser.add_argument("--resume",action="store_true",help="Resume the audited λ=0.50 candidate after a supported validation-memory interruption.")
    args=parser.parse_args()
    try:
        print(run(args.project_root,resume=args.resume))
    except Exception as error:
        write_b2_failure(args.project_root,error)
        raise
