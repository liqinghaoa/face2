from __future__ import annotations
import argparse,sys
from pathlib import Path
import pandas as pd
ROOT=Path(__file__).resolve().parents[2]; sys.path.insert(0,str(ROOT))
from p0b_deca.config import load_config
from p0b_deca.p0a_reader import load_master
from p0b_deca.pilot_selector import select_pilot12
from p0b_deca.qc import create_blind_review
def main() -> int:
 p=argparse.ArgumentParser(); p.add_argument('--config',type=Path,default=ROOT/'config/p0b/p0b_deca_pilot12_v1.yaml'); a=p.parse_args(); c=load_config(a.config,ROOT); samples=select_pilot12(load_master(c.p0a_root),c); out=c.output_root/'pilot_manifest'; out.mkdir(parents=True,exist_ok=True); mapping=pd.DataFrame([s.__dict__ | {'aligned_scene_path':str(s.aligned_scene_path.relative_to(c.p0a_root)),'face_valid_mask_path':str(s.face_valid_mask_path.relative_to(c.p0a_root)),'skin_strict_mask_path':str(s.skin_strict_mask_path.relative_to(c.p0a_root)),'physics_core_mask_path':str(s.physics_core_mask_path.relative_to(c.p0a_root))} for s in samples]); mapping.to_csv(out/'p0b_pilot12_id_mapping.csv',index=False,encoding='utf-8-sig'); mapping[['audit_id']].to_csv(out/'p0b_pilot12_manifest.csv',index=False,encoding='utf-8-sig'); create_blind_review(c.output_root/'qc/p0b_pilot12_blind_review.csv',[s.audit_id for s in samples]); (out/'p0b_pilot12_selection_report.md').write_text('# Deterministic P0-B pilot selection\n\nSelected 12 samples using P0-A status and acquisition metadata only.\n',encoding='utf-8'); print({'selected':len(samples)}); return 0
if __name__=='__main__': raise SystemExit(main())
