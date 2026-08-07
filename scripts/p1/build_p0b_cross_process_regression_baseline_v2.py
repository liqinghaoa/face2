"""Build and validate the immutable P0B cross-process v2 contract; never runs full-500."""
from __future__ import annotations
import argparse, json, subprocess, sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]; sys.path.insert(0,str(ROOT))
from p0b_deca.p1_generation import load_config
from p0b_deca.p1_cross_process_contract import build_contract, freeze, run_holdout, validate_holdout, validate_inputs

p=argparse.ArgumentParser()
p.add_argument("--config",type=Path,default=ROOT/"config/p1/p1_deca_full500_v1.yaml")
p.add_argument("--validate-inputs",action="store_true"); p.add_argument("--select-reference",action="store_true"); p.add_argument("--build-contract",action="store_true"); p.add_argument("--run-holdout",action="store_true"); p.add_argument("--validate-holdout",action="store_true"); p.add_argument("--freeze",action="store_true"); p.add_argument("--all",action="store_true"); p.add_argument("--holdout-worker",action="store_true",help=argparse.SUPPRESS)
a=p.parse_args(); cfg=load_config(a.config,ROOT)
if a.holdout_worker: print(json.dumps(run_holdout(ROOT,cfg),indent=2)); raise SystemExit(0)
if a.validate_inputs: print(json.dumps(validate_inputs(ROOT,cfg),indent=2))
if a.all or a.select_reference or a.build_contract: print(json.dumps(build_contract(ROOT,cfg),indent=2))
if a.all or a.run_holdout:
    subprocess.run([sys.executable,str(Path(__file__).resolve()),"--config",str(a.config),"--holdout-worker"],cwd=ROOT,check=True)
if a.all or a.validate_holdout:
    result=validate_holdout(ROOT,cfg); print(json.dumps(result,indent=2))
if a.all or a.freeze: print(json.dumps(freeze(ROOT),indent=2))
if not any((a.validate_inputs,a.select_reference,a.build_contract,a.run_holdout,a.validate_holdout,a.freeze,a.all)): p.error("select an action")
