"""Entrypoint for the strictly read-only SO-R1-C synthetic evaluation."""
from pathlib import Path
import argparse
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.skin_optics_so_r1.synthetic_id_ood_evaluation import run, write_failure


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, default=ROOT)
    args = parser.parse_args()
    try:
        print(run(args.project_root))
    except Exception as error:
        write_failure(args.project_root, error)
        raise
