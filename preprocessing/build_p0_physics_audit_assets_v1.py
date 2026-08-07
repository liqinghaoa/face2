"""CLI entry point for P0-A physical audit asset construction."""
from __future__ import annotations
import argparse, sys
from pathlib import Path
from typing import Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from p0_physics_assets.builder import build
from p0_physics_assets.config import load_config

def parser() -> argparse.ArgumentParser:
    value=argparse.ArgumentParser(description="Build P0-A physics-audit assets")
    value.add_argument("--config",type=Path,default=Path("config/preprocess/p0_physics_audit_assets_v1.yaml")); value.add_argument("--project-root",type=Path); value.add_argument("--max-samples",type=int); value.add_argument("--sample-ids"); value.add_argument("--parsing-device",choices=("auto","cpu","cuda")); value.add_argument("--overwrite",action="store_true"); value.add_argument("--resume",action="store_true"); value.add_argument("--validate-only",action="store_true"); return value

def main(argv: Sequence[str] | None=None) -> int:
    args=parser().parse_args(argv); root=(args.project_root or PROJECT_ROOT).resolve(); config_path=args.config if args.config.is_absolute() else root/args.config; overrides={key:value for key,value in {"max_samples":args.max_samples,"sample_ids":args.sample_ids,"parsing_device":args.parsing_device,"overwrite":args.overwrite if args.overwrite else None,"resume":args.resume if args.resume else None}.items() if value is not None}; config=load_config(config_path,root,overrides); result=build(config,args.validate_only); print(result)
    if args.validate_only:
        return 0
    return 0 if int(result["core_success"]) == int(result["selected"]) else 1

if __name__ == "__main__":
    try: raise SystemExit(main())
    except Exception as exc: print(f"[p0-a error] {type(exc).__name__}: {exc}",file=sys.stderr); raise SystemExit(2)
