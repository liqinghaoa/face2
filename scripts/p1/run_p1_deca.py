"""CLI for the label-blind P1 frozen-DECA cache generator."""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]; sys.path.insert(0, str(ROOT))
from p0b_deca.p1_generation import load_config, run

parser = argparse.ArgumentParser()
parser.add_argument("--config", type=Path, default=ROOT / "config/p1/p1_deca_full500_v1.yaml")
parser.add_argument("--validate-only", action="store_true")
parser.add_argument("--sample-ids", nargs="+")
parser.add_argument("--pilot12", action="store_true")
parser.add_argument("--full", action="store_true")
parser.add_argument("--resume", action="store_true")
parser.add_argument("--retry-failures", action="store_true")
parser.add_argument("--overwrite-output-only", action="store_true")
args = parser.parse_args()
cfg = load_config(args.config, ROOT)
print(json.dumps(run(ROOT, cfg, sample_ids=args.sample_ids, pilot12=args.pilot12, full=args.full, resume=args.resume, retry_failures=args.retry_failures, overwrite_output_only=args.overwrite_output_only, validate_only=args.validate_only), ensure_ascii=False, indent=2))
