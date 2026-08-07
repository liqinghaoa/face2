"""Authorized production release: freeze v2.1, formal Pilot12, then full-500 only on pass."""
from __future__ import annotations
import argparse,json,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from p0b_deca.p1_generation import load_config,run
from p0b_deca.p1_audit import run as integrity
from p0b_deca.p1_production_gate import build,validate_run04,freeze,evaluate_formal_pilot,frozen_valid
p=argparse.ArgumentParser();p.add_argument('--config',type=Path,default=ROOT/'config/p1/p1_deca_full500_v1.yaml');p.add_argument('--all',action='store_true');a=p.parse_args()
if not a.all:p.error('this authorized production entry requires --all')
cfg=load_config(a.config,ROOT); build(ROOT,cfg);r4=validate_run04(ROOT,cfg)
if r4['contract_status']!='VALIDATED':raise RuntimeError('production run04 validation failed')
freeze(ROOT)
stamp=time.strftime('%Y%m%dT%H%M%SZ',time.gmtime());attempt=ROOT/'data/processed/P1_DECA_Frozen500_v1/attempts'/f'production_v2_1_{stamp}'/'pilot12';attempt_cfg=dict(cfg);attempt_cfg['output_root']=str(attempt.relative_to(ROOT))
pilot=run(ROOT,attempt_cfg,pilot12=True)
formal=evaluate_formal_pilot(ROOT,cfg,attempt)
if formal['decision']!='FORMAL_PILOT12_PASS':raise RuntimeError(json.dumps(formal))
full=run(ROOT,cfg,full=True,resume=True,production_contract=True)
if full['failed_cases']:
 full=run(ROOT,cfg,full=True,resume=True,retry_failures=True,production_contract=True)
audit=integrity(ROOT,a.config)
result={'production_contract_frozen':frozen_valid(ROOT,cfg),'run04_validation':r4,'formal_pilot':formal,'full_first_or_retry':full,'offline_integrity':audit,'full_500_started':True}
print(json.dumps(result,ensure_ascii=False,indent=2))
