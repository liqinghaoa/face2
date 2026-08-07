"""CLI for the isolated, evidence-only Pilot12 reproducibility audit."""
from __future__ import annotations
import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from p0b_deca.p1_generation import load_config
from p0b_deca.p1_reproducibility_audit import RUN_NAMES, audit_root, finalize, validate, worker, write_manifest

parser = argparse.ArgumentParser()
parser.add_argument("--config", type=Path, default=ROOT / "config/p1/p1_deca_full500_v1.yaml")
parser.add_argument("--validate-only", action="store_true")
parser.add_argument("--run-independent-processes", action="store_true")
parser.add_argument("--compare", action="store_true")
parser.add_argument("--render-qc", action="store_true", help="reserved: comparisons retain raw data; no gate action")
parser.add_argument("--finalize-report", action="store_true")
parser.add_argument("--all", action="store_true")
parser.add_argument("--worker", choices=RUN_NAMES, help=argparse.SUPPRESS)
args = parser.parse_args(); cfg = load_config(args.config, ROOT)

if args.worker:
    print(json.dumps(worker(ROOT, cfg, args.worker), indent=2)); raise SystemExit(0)
if args.validate_only:
    print(json.dumps(validate(ROOT, cfg), indent=2)); raise SystemExit(0)
if not any((args.run_independent_processes, args.compare, args.render_qc, args.finalize_report, args.all)):
    parser.error("select an audit action")
write_manifest(ROOT, cfg)
if args.all or args.run_independent_processes:
    audit = audit_root(ROOT, cfg)
    for run in RUN_NAMES:
        if (audit / run).exists():
            raise FileExistsError(f"refusing to overwrite audit observation: {audit / run}")
        command = [sys.executable, str(Path(__file__).resolve()), "--config", str(args.config), "--worker", run]
        subprocess.run(command, cwd=ROOT, check=True)
if args.all or args.compare or args.finalize_report:
    result = finalize(ROOT, cfg)
    print(json.dumps(result, indent=2))
