#!/usr/bin/env python
"""SO-R1-A1 paired same-latent synthetic Pilot CLI."""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]; sys.path.insert(0,str(ROOT/'src'))
from skin_optics_so_r1.paired_pilot import PilotError, load_config, validate, generate, audit, verify

def main() -> int:
    p=argparse.ArgumentParser(); p.add_argument('--config',required=True); g=p.add_mutually_exclusive_group(required=True)
    for name in ('validate-only','generate','audit','verify'): g.add_argument('--'+name,action='store_true')
    a=p.parse_args(); cfg=load_config(ROOT/a.config)
    try:
        if a.validate_only: out=validate(ROOT,cfg)
        elif a.generate: out=generate(ROOT,cfg)
        elif a.audit: out=audit(ROOT,cfg,ROOT/cfg['dataset_root'],write_report=True)
        else: out=verify(ROOT,cfg)
        print(out['terminal_summary'] if 'terminal_summary' in out else json.dumps(out,ensure_ascii=False,indent=2,sort_keys=True,default=str))
    except PilotError as e: print(f'SO-R1-A1 failed: {e}',file=sys.stderr); return 2
    return 0
if __name__=='__main__': raise SystemExit(main())
