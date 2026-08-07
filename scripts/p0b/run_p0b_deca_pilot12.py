from __future__ import annotations
import argparse,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]; sys.path.insert(0,str(ROOT))
from p0b_deca.config import load_config
from p0b_deca.environment_audit import audit_environment
def main()->int:
 p=argparse.ArgumentParser(); p.add_argument('--config',type=Path,default=ROOT/'config/p0b/p0b_deca_pilot12_v1.yaml'); p.add_argument('--validate-only',action='store_true'); p.add_argument('--audit-ids'); p.add_argument('--max-samples',type=int); p.add_argument('--device'); p.add_argument('--resume',action='store_true'); p.add_argument('--overwrite',action='store_true'); p.add_argument('--skip-qc',action='store_true'); a=p.parse_args(); c=load_config(a.config,ROOT); audit=audit_environment(c)
 if a.validate_only: print({'p0a_root_exists':c.p0a_root.is_dir(),'deca_root_exists':c.deca_root.is_dir(),'environment_passed':audit['passed']}); return 0 if audit['passed'] else 2
 if not audit['passed']: print({'status':'failed_environment','reason':audit['block_reason']}); return 2
 raise RuntimeError('P0-B1 is intentionally unavailable until audited official DECA interfaces are present')
if __name__=='__main__': raise SystemExit(main())
