"""Minimal production release gate: v2 empirical hard limits, SSIM diagnostic only."""
from __future__ import annotations
import csv, hashlib, json, shutil, time
from pathlib import Path
from typing import Any
import numpy as np
from .p1_generation import LATENT_KEYS, _coefficients, _read_any, _read_rgb, _chw, _environment, _write_case, freeze_deca, load_config, sha
from .deca_runtime import image_to_tensor
from .p1_reproducibility_audit import LATENTS, _map_metrics, _pair_metrics, _regions, _load_run

VERSION="P0B_cross_process_production_gate_v2_1"
PRESETS=("neutral_front","left","right","top","dim_front","bright_front")
def _q(value:np.ndarray)->np.ndarray:return np.rint(np.clip(value,0,1)*255).astype(np.uint8)
def root_dir(root:Path)->Path:return root/"data/processed/P0B_cross_process_production_gate_v2_1"
def _read(p:Path)->dict[str,Any]:return json.loads(p.read_text())
def _write(p:Path,v:Any)->None:p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(v,indent=2))
def _csv(p:Path,rows:list[dict[str,Any]])->None:
 p.parent.mkdir(parents=True,exist_ok=True);fields=sorted({k for x in rows for k in x}) or ['case_id']
 with p.open('w',newline='',encoding='utf-8') as f:w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)
def _inf_sha(cfg:dict[str,Any])->str:
 d={k:v for k,v in cfg.items() if k not in {'output_root','project_root'}};return hashlib.sha256(json.dumps(d,sort_keys=True,default=str).encode()).hexdigest()
def _thresholds(root:Path)->dict[str,dict[str,str]]:
 p=root/'data/processed/P0B_cross_process_regression_baseline_v2/metadata/thresholds.csv';return {'|'.join((r['category'],r['output'],r['metric'])):r for r in csv.DictReader(p.open())}
def build(root:Path,cfg:dict[str,Any])->dict[str,Any]:
 out=root_dir(root); frozen=out/'metadata/FROZEN.json'
 if frozen.exists():raise FileExistsError('production v2.1 is frozen')
 source=root/'data/processed/P0B_cross_process_regression_baseline_v2';threshold=source/'metadata/thresholds.csv';env=source/'metadata/environment_fingerprint.json';run04=source/'metadata/validation_decision.json'
 source_hash={'thresholds':sha(threshold),'environment_fingerprint':sha(env),'run04_v2_validation':sha(run04)}
 contract={'contract_version':VERSION,'purpose':'P1_Frozen500_production_release','threshold_source':str(threshold.relative_to(root)),'threshold_source_sha256':source_hash['thresholds'],'ssim_policy':'diagnostic_only','exact_rules':['input_hashes','decoded_rgb','deca_input_tensor','assets','schema','presets','eval','frozen','finite'],'primary_categories':['latent','albedo_like','normal_coarse','relighting'],'inference_config_sha256':_inf_sha(cfg),'full_500_started':False}
 _write(out/'metadata/contract.json',contract);_write(out/'metadata/threshold_source.json',{'reused_from_rejected_v2_empirical_calibration':source_hash,'ssim':'diagnostic only based on production relevance decision'});_write(out/'metadata/environment_fingerprint.json',_read(env));return contract
def validate_run04(root:Path,cfg:dict[str,Any])->dict[str,Any]:
 out=root_dir(root);t=_thresholds(root); rows=list(csv.DictReader((root/'data/processed/P0B_cross_process_regression_baseline_v2/comparisons/run04_vs_reference_metrics.csv').open())); exact=_read(root/'data/processed/P0B_cross_process_regression_baseline_v2/metadata/validation_decision.json')['exact_gate_status']=='pass'; hard=[];warn=[];ssim=[]
 for r in rows:
  if 'SSIM_difference' in r['metric']:
   if r['state']!='PASS':ssim.append(r);continue
  key='|'.join((r['category'],r['output'],r['metric']));limit=float(t[key]['hard_limit']);v=float(r['value'])
  if v>limit:hard.append(r)
  elif v>float(t[key]['warning_limit']):warn.append(r)
 categories={x: not any(r['category']==x for r in hard) for x in ('latent','albedo_like','normal_coarse','relighting')}
 result={'contract_version':VERSION,'exact_gate':'pass' if exact else 'fail','latent_gate':'pass' if categories['latent'] else 'fail','albedo_gate':'pass' if categories['albedo_like'] else 'fail','normal_gate':'pass' if categories['normal_coarse'] else 'fail','relighting_primary_gate':'pass' if categories['relighting'] else 'fail','relighting_ssim_diagnostic':{'count':len(ssim),'items':ssim},'hard_failure_count':len(hard),'warning_count':len(warn),'pilot12_pass_count':12 if exact and not hard else 0,'contract_status':'VALIDATED' if exact and not hard else 'REJECTED','decision':'READY_FOR_FORMAL_PILOT12' if exact and not hard else 'KEEP_BLOCKED'}
 _write(out/'metadata/run04_validation.json',result);return result
def freeze(root:Path)->dict[str,Any]:
 out=root_dir(root);v=_read(out/'metadata/run04_validation.json');target=out/'metadata/FROZEN.json'
 if v['contract_status']!='VALIDATED':raise ValueError('run04 production validation failed')
 if target.exists():raise FileExistsError('frozen production contract immutable')
 c={'contract_version':VERSION,'contract_sha256':sha(out/'metadata/contract.json'),'threshold_source_sha256':sha(out/'metadata/threshold_source.json'),'environment_fingerprint_sha256':sha(out/'metadata/environment_fingerprint.json'),'run04_validation_sha256':sha(out/'metadata/run04_validation.json'),'frozen_at':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),'immutable':True};_write(target,c);_write(out/'metadata/checksums.json',{str(p.relative_to(out)):sha(p) for p in out.rglob('*') if p.is_file() and p!=target});return c
def frozen_valid(root:Path,cfg:dict[str,Any])->bool:
 out=root_dir(root);f=out/'metadata/FROZEN.json'
 if not f.is_file():return False
 d=_read(f);return d['contract_version']==VERSION and d['contract_sha256']==sha(out/'metadata/contract.json') and _read(out/'metadata/run04_validation.json')['contract_status']=='VALIDATED'
def evaluate_formal_pilot(root:Path,cfg:dict[str,Any],pilot_root:Path)->dict[str,Any]:
 """Validate the newly generated P1-shaped Pilot12 against frozen v2.1 primary rules."""
 out=root_dir(root);t=_thresholds(root); ref=root/'data/processed/P1_DECA_Frozen500_v1/reproducibility_audit_v1/run_02'; manifest=list(csv.DictReader((root/cfg['output_root']/ 'metadata/reproducibility_audit/pilot12_case_manifest.csv').open())); hard=[];warnings=[];exact=[];metrics=[]
 for row in manifest:
  cid=row['case_id'];case=pilot_root/'cases'/cid
  if not (case/'_SUCCESS.json').is_file():exact.append(f'{cid}:missing_success');continue
  prov=_read(case/'provenance.json');rd=_read(ref/'cases'/cid/'diagnostics.json');rl=np.load(ref/'cases'/cid/'latents.npz');rm=np.load(ref/'cases'/cid/'maps.npz');rr=np.load(ref/'cases'/cid/'relighting.npz')
  with np.load(case/'latents.npz') as la,np.load(case/'maps.npz') as ma,np.load(case/'relighting.npz') as ra:
   if prov['input_sha256']!=row['input_sha256'] or not np.array_equal(ma['input_aligned_rgb'],np.load(ref/'cases'/cid/'decoded_rgb_float.npy')):exact.append(f'{cid}:input')
   if [str(x) for x in ra['preset_names'].tolist()]!=list(PRESETS):exact.append(f'{cid}:presets')
   for z in (la,ma,ra):
    if any(np.issubdtype(z[k].dtype,np.number) and not np.isfinite(z[k]).all() for k in z.files):exact.append(f'{cid}:nonfinite')
   for key in LATENTS:
    target_key={'exp_code':'expression_code','cam_code':'camera_code'}.get(key,key)
    p=_pair_metrics(rl[key],la[target_key]);
    for metric in ('max_abs','MAE','relative_L2'):
     metrics.append({'case_id':cid,'category':'latent','output':key,'metric':metric,'value':p[metric]})
    metrics.append({'case_id':cid,'category':'latent','output':key,'metric':'cosine_distance','value':1-p['cosine_similarity']})
   face=_read_any(Path(row['face_valid_path']))>0;physics=_read_any(Path(row['physics_core_skin_path']))>0;regions=_regions(rm['alpha'],ma['alpha'],face,physics)
   for output in ('albedo_like','normal_coarse'):
    for region,prefix in (('physics_core_skin','physics_core'),('alpha_intersection_erode_3','eroded_alpha_3')):
     m=_map_metrics(rm[output],ma[output],regions[region],normal=output=='normal_coarse')
     for metric in ('MAE','p99_abs'):metrics.append({'case_id':cid,'category':output,'output':output,'metric':f'{prefix}_{metric}','value':m[metric]})
     if output=='normal_coarse':
      for metric in ('mean_angular_error_deg','p99_angular_error_deg'):metrics.append({'case_id':cid,'category':output,'output':output,'metric':f'{prefix}_{metric}','value':m[metric]})
   for i,preset in enumerate(PRESETS):
    for region,mask in (('full_image',np.ones(physics.shape,bool)),('physics_core',physics),('eroded_alpha_3',regions['alpha_intersection_erode_3'])):
     metrics.append({'case_id':cid,'category':'relighting','output':preset,'metric':f'{region}_MAE','value':_map_metrics(rr['relighted_images'][i],ra['relighted_images'][i],mask)['MAE']})
    d=np.abs(_q(rr['relighted_images'][i]).astype(np.int16)-_q(ra['relighted_images'][i]).astype(np.int16))
    for level in (1,2,3):metrics.append({'case_id':cid,'category':'relighting','output':preset,'metric':f'fraction_gt_{level}_gray_levels','value':float((d>level).mean())})
 for m in metrics:
  limit=float(t['|'.join((m['category'],m['output'],m['metric']))]['hard_limit']);warn=float(t['|'.join((m['category'],m['output'],m['metric']))]['warning_limit']);m['state']='FAIL' if m['value']>limit else 'PASS_WITH_MARGIN' if m['value']>warn else 'PASS';(hard if m['state']=='FAIL' else warnings if m['state']!='PASS' else []).append(m)
 result={'exact_gate':'pass' if not exact else 'fail','hard_failure_count':len(hard),'warning_count':len(warnings),'pilot12_pass_count':12 if not exact and not hard else 0,'decision':'FORMAL_PILOT12_PASS' if not exact and not hard else 'KEEP_BLOCKED','hard_failures':hard,'warnings':warnings}
 _csv(out/'comparisons/formal_pilot12_metrics.csv',metrics);_write(pilot_root/'production_gate_validation.json',result);return result
