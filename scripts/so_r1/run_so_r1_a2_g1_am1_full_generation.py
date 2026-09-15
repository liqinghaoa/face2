from pathlib import Path
import argparse, sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.skin_optics_so_r1.full_generation_g1_am1 import run_final_audit

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, default=ROOT)
    parser.add_argument("--mode", choices=("audit",), default="audit")
    args = parser.parse_args()
    print(run_final_audit(args.project_root))
