"""Validate the fixed P1 canary without rewriting its completed cache."""
from __future__ import annotations
import argparse, csv, json, sys, hashlib
from pathlib import Path
import cv2, numpy as np, yaml
ROOT = Path(__file__).resolve().parents[2]; sys.path.insert(0, str(ROOT))
from p0b_deca.deca_runtime import forward_deca, image_to_tensor, load_deca
from p0b_deca.p1_audit import validate_case
from p0b_deca.p1_generation import config

p = argparse.ArgumentParser(); p.add_argument('--config', type=Path, default=ROOT/'config/p1/p1_canary_v1.yaml'); a = p.parse_args()
canary = yaml.safe_load(a.config.read_text()); base = config(ROOT/canary['base_config'], ROOT)
with (ROOT/canary['canary_manifest']).open(newline='', encoding='utf-8-sig') as h: rows = list(csv.DictReader(h))
if len(rows) != int(canary['expected_case_count']) or len({r['case_id'] for r in rows}) != len(rows): raise RuntimeError('invalid canary manifest cardinality')
out = ROOT/base['output_root']; results=[]
for r in rows:
 v=validate_case(out/'cases'/r['case_id'],base['required_latents'],base['required_maps']); results.append({'case_id':r['case_id'],'success':v['success'],'errors':v['errors'],'relighting_count':v.get('relighting_count')})
if not all(r['success'] for r in results): raise RuntimeError('canary output validation failed')
model,_ = load_deca(ROOT/base['deca_root'], 'cuda', int(base['seed']))
repeat=[]
for r in sorted(rows,key=lambda x:x['case_id'])[:int(canary['repeat_case_count'])]:
 img=cv2.imread(r['input_path_wsl']); rgb=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
 code,_,_=forward_deca(model,image_to_tensor(rgb,'cuda'))
 with np.load(out/'cases'/r['case_id']/'latents.npz') as cached:
  fields=['tex_code','shape_code','detail_code','expression_code','pose_code','camera_code','light_code']
  mapping={'tex_code':'tex','shape_code':'shape','detail_code':'detail','expression_code':'exp','pose_code':'pose','camera_code':'cam','light_code':'light'}
  diffs={key:float(np.max(np.abs(code[mapping[key]].detach().cpu().numpy()-cached[key]))) for key in fields}
 repeat.append({'case_id':r['case_id'],'max_abs_difference_by_latent':diffs,'passed':all(v <= float(canary['determinism_atol'])+float(canary['determinism_rtol'])*float(np.max(np.abs(np.load(out/'cases'/r['case_id']/'latents.npz')[k]))) for k,v in diffs.items())})
digest=hashlib.sha256((ROOT/canary['canary_manifest']).read_bytes()).hexdigest()
result={'status':'passed' if all(r['passed'] for r in repeat) else 'failed','case_count':len(rows),'canary_manifest_sha256':digest,'repeat_cases':repeat,'output_validation':results,'atol':canary['determinism_atol'],'rtol':canary['determinism_rtol']}
(out/'audit').mkdir(exist_ok=True); (out/'audit'/'p1_canary_validation.json').write_text(json.dumps(result,indent=2)); print(json.dumps(result,indent=2))
if result['status']!='passed': raise SystemExit(2)
