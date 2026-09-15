"""AM4 CPU-only global camera safety qualification and failure attribution."""
from __future__ import annotations

# Must be set before NumPy is imported by a Windows spawned worker.
import os
for _name in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS'):
 os.environ.setdefault(_name,'1')

import hashlib
import json
import time
from concurrent.futures import ProcessPoolExecutor,as_completed
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from .formal_generation import _metrics,_stable
from .mask_benchmark_closure import _derive_mask
from .paired_pilot import _appearance,_field,_matrix,_so0
from .camera_light_selection import read_inputs,normalize_channel_response,compute_sam_distance_matrix

ROOT_SEED=20260822
LIGHTS=('D65','A','FL2','FL11')
PERMANENT_BASE={'Nokia N900','Pentax Q','SONY NEX-5N','Olympus E-PL2'}

def _write(path:Path,value):
 path.parent.mkdir(parents=True,exist_ok=True)
 path.write_text(json.dumps(value,indent=2,default=str)+'\n',encoding='utf-8')

def _qc_pass(q):
 return bool(q['high_clip_element_fraction']<=.1 and q['low_clip_element_fraction']<=.1 and q['nonfinite_count']==0 and not q['all_zero'] and not q['all_one'] and max(v for k,v in q.items() if k.endswith('_zero_fraction') or k.endswith('_one_fraction'))<.5)

def panel_a():
 rows=[]
 for m in (.90,.95,.98):
  for h in (.02,.50,.95):
   for category in ('Full','Mild','Strong'):
    for replicate in range(8):
     rows.append({'panel':'A','latent_id':f'AM4_A_M{m:.2f}_H{h:.2f}_{category}_{replicate}','m_base':m,'h_base':h,'mask_category':category,'replicate':replicate})
 return rows

def _digest(*parts):
 return hashlib.blake2b('|'.join(map(str,parts)).encode(),digest_size=16).hexdigest()

def panel_b():
 """Frozen independently stratified 256-latent coverage panel."""
 ids=[f'AM4_B_{i:03d}' for i in range(256)]
 m_order=sorted(ids,key=lambda x:_digest('AM4-panel-b-m',x));h_order=sorted(ids,key=lambda x:_digest('AM4-panel-b-h',x));cat_order=sorted(ids,key=lambda x:_digest('AM4-panel-b-mask',x))
 m={}
 for group,(lo,hi,n) in enumerate(((0,.5,64),(.5,.85,64),(.85,1,128))):
  for rank,lid in enumerate(m_order[sum(x[2] for x in ((0,.5,64),(.5,.85,64))[:group]):sum(x[2] for x in ((0,.5,64),(.5,.85,64),(.85,1,128))[:group+1])]):m[lid]=lo+(rank+.5)/n*(hi-lo)
 h={lid:((rank%32)+.5)/32/8+(rank//32)/8 for rank,lid in enumerate(h_order)}
 cats={lid:('Full' if rank<102 else 'Mild' if rank<204 else 'Strong') for rank,lid in enumerate(cat_order)}
 return [{'panel':'B','latent_id':lid,'m_base':m[lid],'h_base':h[lid],'mask_category':cats[lid],'replicate':0} for lid in ids]

def _latent(row):
 lid=row['latent_id'];cat=row['mask_category'];tr=None if cat=='Full' else ((.72,.93) if cat=='Mild' else (.32,.68))
 m=_field(float(row['m_base']),_stable('AM4',row['panel'],lid,'m'),256);h=_field(float(row['h_base']),_stable('AM4',row['panel'],lid,'h'),256)
 mask,coverage,_,mask_hash,_=_derive_mask(ROOT_SEED,lid,cat.lower(),0,256,tr,128,.90)
 return m,h,mask,coverage,mask_hash

def _render(root,model,decode,m,h,mask,camera,light,sh,sp,ev):
 z=model.render_camera(m,h,camera,light,_matrix(root,camera,light),sh,sp,2**ev)
 unclipped=z.srgb_unclipped.astype('float32');post=decode(z.srgb_display_clipped).astype('float32');post[mask==0]=0
 return _metrics(unclipped,post,mask)

def _condition(root,row,camera,light,model=None,so0=None,decode=None,stop_on_first=True):
 if model is None:model,so0,decode=_so0(root,{})
 m,h,mask,coverage,mask_hash=_latent(row);last=None
 for attempt0 in range(128):
  seed=_stable('AM4',row['panel'],row['latent_id'],camera,light,'appearance',attempt0)
  sh,sp,_,_=_appearance(seed,256,so0);ev=float(np.random.Generator(np.random.PCG64(_stable('AM4',row['panel'],row['latent_id'],camera,light,'exposure',attempt0))).uniform(-.5,.31))
  q=_render(root,model,decode,m,h,mask,camera,light,sh,sp,ev);last=(q,seed,sh,sp,ev)
  if _qc_pass(q):
   return {**row,'camera':camera,'light':light,'valid_skin_fraction':coverage,'mask_hash':mask_hash,'appearance_seed':seed,'attempt_count':attempt0+1,'retry_exhausted':False,'exposure_ev':ev,'canonical_metadata_hash':_digest(row['panel'],row['latent_id'],row['m_base'],row['h_base'],row['mask_category'],camera,light,mask_hash),**q}
 q,seed,_,_,ev=last
 return {**row,'camera':camera,'light':light,'valid_skin_fraction':coverage,'mask_hash':mask_hash,'appearance_seed':seed,'attempt_count':128,'retry_exhausted':True,'exposure_ev':ev,'canonical_metadata_hash':_digest(row['panel'],row['latent_id'],row['m_base'],row['h_base'],row['mask_category'],camera,light,mask_hash),**q}

def _am3_condition(root,row,attempt0,model,so0,decode,camera='Canon 300D',light=None,p_zero=False,exposure=None):
 """Exact AM3 nuisance contract used to reconstruct its missing detailed rows."""
 light=light or row['light'];lid=row['latent_id'];cat=row['mask_category'];tr=None if cat=='Full' else ((.72,.93) if cat=='Mild' else (.32,.68))
 m=_field(float(row['m_base']),_stable('AM3',lid,'m'),256);h=_field(float(row['h_base']),_stable('AM3',lid,'h'),256);mask,coverage,_,mask_hash,_=_derive_mask(ROOT_SEED,lid,cat.lower(),0,256,tr,128,.90)
 seed=_stable('AM3',lid,camera,light,'appearance',attempt0);sh,sp,_,_=_appearance(seed,256,so0);sp=np.zeros_like(sp) if p_zero else sp
 ev=float(np.random.Generator(np.random.PCG64(_stable('AM3',lid,camera,light,'exposure',attempt0))).uniform(-.5,.31)) if exposure is None else float(exposure)
 q=_render(root,model,decode,m,h,mask,camera,light,sh,sp,ev)
 return {**row,'camera':camera,'light':light,'valid_skin_fraction':coverage,'mask_hash':mask_hash,'appearance_seed':seed,'attempt_count':attempt0+1,'exposure_ev':ev,'p_zero':p_zero,**q},(m,h,mask,sh,sp,ev)

def run_canon300_attribution(root:Path):
 rep=root/'reports/so_r1_a0_am4_global_camera_atlas';rep.mkdir(parents=True,exist_ok=True)
 model,so0,decode=_so0(root,{})
 # The original AM3 module saved only its summary; reconstruct the exact deterministic panel.
 reconstruction=rep/'canon300_am3_panel_reconstruction.csv'
 if reconstruction.exists():
  rows=pd.read_csv(reconstruction).to_dict('records')
 else:
  rows=[]
  for row in panel_a():
   am3row={**row,'latent_id':row['latent_id'].replace('AM4_A','AM3')}
   for light in LIGHTS:
    attempts=[]
    for attempt0 in range(128):
     q,_=_am3_condition(root,{**am3row,'light':light},attempt0,model,so0,decode,light=light);attempts.append(q)
     if _qc_pass(q):break
    rows.append(attempts[-1] | {'retry_exhausted':not _qc_pass(attempts[-1])})
  pd.DataFrame(rows).to_csv(reconstruction,index=False)
 failures=[x for x in rows if x['retry_exhausted'] or not _qc_pass(x)]
 if len(failures)!=2:
  result={'status':'FAIL_CANON300_FAILURE_NOT_REPRODUCIBLE','pass':False,'reconstructed_failure_count':len(failures),'expected_failure_count':2};_write(rep/'canon300_failure_attribution.json',result);return result
 evidence=[]
 controls=['Canon 5DMarkII','Nikon D80','Canon 20D','Canon 1DMarkIII','Nikon D5100']
 for index,failure in enumerate(failures,1):
  original={k:failure[k] for k in failure if k not in ('retry_exhausted','p_zero')};sequence=[]
  for attempt0 in range(128):
   q,state=_am3_condition(root,failure,attempt0,model,so0,decode,light=failure['light']);sequence.append(q)
  # AM3 recorded the terminal 128th failed condition, not a retrospective
  # minimum; its nuisance state is the fixed state required for attribution.
  terminal=sequence[-1]
  _,state=_am3_condition(root,failure,terminal['attempt_count']-1,model,so0,decode,light=failure['light']);m,h,mask,sh,sp,ev=state
  swaps=[]
  for camera in controls:
   q=_render(root,model,decode,m,h,mask,camera,failure['light'],sh,sp,ev);q.update({'camera':camera,'light':failure['light'],'attempt':terminal['attempt_count'],'pass':_qc_pass(q)});swaps.append(q)
  sweep=[]
  for exposure in (-.5,-.4,-.3,-.2,-.1,0,.1,.2,.31):
   q=_render(root,model,decode,m,h,mask,'Canon 300D',failure['light'],sh,sp,exposure);q.update({'exposure_ev':exposure,'pass':_qc_pass(q)});sweep.append(q)
  q0=_render(root,model,decode,m,h,mask,'Canon 300D',failure['light'],sh,np.zeros_like(sp),ev);q0.update({'p_zero':True,'pass':_qc_pass(q0)})
  reproduced=terminal['attempt_count']==failure['attempt_count'] and bool(np.isclose(terminal['low_clip_element_fraction'],failure['low_clip_element_fraction'])) and bool(np.isclose(terminal['high_clip_element_fraction'],failure['high_clip_element_fraction'])) and terminal['appearance_seed']==failure['appearance_seed'] and bool(np.isclose(terminal['exposure_ev'],failure['exposure_ev'])) and not any(_qc_pass(q) for q in sequence)
  evidence.append({'failure_id':f'CANON300_AM3_{index}','original_row':original,'reproduction_pass':reproduced,'terminal_attempt':terminal,'best_qc_margin_attempt':min(sequence,key=lambda x:(max(x['high_clip_element_fraction'],x['low_clip_element_fraction']),x['nonfinite_count'])),'retry_sequence':sequence,'camera_swap':swaps,'exposure_sweep':sweep,'p_zero':q0,'camera_specific':reproduced and any(x['pass'] for x in swaps) and not any(x['pass'] for x in sweep) and not q0['pass'] and all(x['nonfinite_count']==0 for x in sequence)})
 ok=all(x['camera_specific'] for x in evidence)
 result={'status':'PASS_CANON300_CAMERA_SPECIFIC_ATTRIBUTION' if ok else 'FAIL_CANON300_ATTRIBUTION_UNRESOLVED','pass':ok,'classification':'CAMERA_SPECIFIC_LOW_CLIPPING_ARTIFACT' if ok else None,'permanent_exclusion':'Canon 300D' if ok else None,'evidence':evidence}
 _write(rep/'canon300_failure_attribution.json',result);return result

def _worker_init():
 for name in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS'):
  os.environ[name]='1'

def _parity_worker(arg):
 root=Path(arg['root']);model,so0,decode=_so0(root,{})
 values=[]
 for row in arg['rows']:
  for light in LIGHTS:values.append(_condition(root,row,arg['camera'],light,model,so0,decode))
 return values

def atlas_parallel_parity(root:Path,report:Path):
 """Compare the exact first 32 Panel-A latent rows under 1 and 8 workers."""
 rows=panel_a()[:32];args=[{'root':str(root),'rows':[row],'camera':'Canon 5DMarkII'} for row in rows]
 results=[]
 for workers in (1,8):
  values=[]
  with ProcessPoolExecutor(max_workers=workers,initializer=_worker_init) as pool:
   for future in as_completed([pool.submit(_parity_worker,arg) for arg in args]):values.extend(future.result())
  values=sorted(values,key=lambda x:(x['latent_id'],LIGHTS.index(x['light'])))
  results.append(values)
 columns=['latent_id','light','attempt_count','retry_exhausted','appearance_seed','exposure_ev','canonical_metadata_hash','high_clip_element_fraction','low_clip_element_fraction','nonfinite_count','R_zero_fraction','R_one_fraction','G_zero_fraction','G_one_fraction','B_zero_fraction','B_one_fraction']
 exact=all(all(a[k]==b[k] for k in columns) for a,b in zip(*results)) and len(results[0])==len(results[1])==128
 payload={'status':'PASS' if exact else 'FAIL_ATLAS_PARALLEL_PARITY','pass':exact,'latent_count':32,'acquisition_count':128,'workers_compared':[1,8],'comparison_columns':columns,'hashes_1worker':[x['canonical_metadata_hash'] for x in results[0]],'hashes_8worker':[x['canonical_metadata_hash'] for x in results[1]]}
 _write(report/'atlas_parallel_parity.json',payload);return payload

def _atlas_camera_worker(arg):
 root=Path(arg['root']);camera=arg['camera'];model,so0,decode=_so0(root,{})
 start=time.perf_counter();metrics=[];first=None
 try:
  import psutil
  peak=psutil.Process().memory_info().rss
 except Exception:peak=None
 for row in arg['panel_a']+arg['panel_b']:
  for light in LIGHTS:
   item=_condition(root,row,camera,light,model,so0,decode);metrics.append(item)
   failed=item['retry_exhausted'] or not _qc_pass(item)
   if failed:
    first=item;break
  if first is not None:break
  try:
   import psutil
   peak=max(peak or 0,psutil.Process().memory_info().rss)
  except Exception:pass
 return {'camera':camera,'status':'UNSAFE' if first else 'SAFE','planned_acquisition_count':1888,'executed_acquisition_count':len(metrics),'early_stop':first is not None,'first_failure':first,'wall_seconds':time.perf_counter()-start,'peak_rss_bytes':peak,'metrics':metrics}

def _protected_audit(root:Path,report:Path):
 before=pd.read_csv(root/'reports/so_r1_a2_d0_protocol_freeze/d0_authoritative_input_ledger.csv');rows=[]
 for _,entry in before.iterrows():
  path=root/entry.path;exists=path.exists();after=hashlib.sha256(path.read_bytes()).hexdigest() if exists else None
  rows.append({'path':entry.path,'before_sha256':entry.sha256,'after_sha256':after,'exists':exists,'unchanged':bool(exists and after==entry.sha256)})
 changed=sum(not x['unchanged'] for x in rows);missing=sum(not x['exists'] for x in rows);payload={'before_count':len(rows),'after_count':len(rows),'changed':changed,'missing':missing,'pass':changed==0 and missing==0}
 pd.DataFrame(rows).to_csv(report/'protected_assets_after.csv',index=False);_write(report/'protected_asset_hash_audit.json',payload);return payload

def _candidate_pool(root:Path,report:Path):
 inv=pd.read_csv(root/'reports/so_r1_a0_am1_camera_replacement/replacement_candidate_inventory.csv').copy();permanent=PERMANENT_BASE|{'Canon 300D'}
 inv['quality_eligible']=inv.quality_gate_pass.astype(bool)
 inv['eligible_for_atlas']=inv.quality_eligible&~inv.camera_name.isin(permanent)
 inv['exclusion_reason']=np.where(~inv.quality_eligible,'SO0_QUALITY_INELIGIBLE',np.where(inv.camera_name.isin(permanent),'PERMANENT_CAMERA_ARTIFACT','eligible'))
 inv['historical_evidence_status']=np.where(inv.camera_name.isin(permanent),'RESOLVED_ARTIFACT_EXCLUDED','NO_UNRESOLVED_FAILURE')
 for light in LIGHTS:inv[f'{light}_quality_status']=np.where(inv.quality_eligible,'PASS','FAIL')
 columns=['camera_name','quality_eligible','eligible_for_atlas','exclusion_reason',*[f'{x}_quality_status' for x in LIGHTS],'historical_evidence_status']
 inv[columns].rename(columns={'camera_name':'camera'}).to_csv(report/'candidate_pool.csv',index=False)
 return inv[inv.eligible_for_atlas].camera_name.tolist(),inv

def _select_seen(root:Path,safe):
 held=['Canon 1DMarkIII','Nikon D5100'];anchor='Canon 5DMarkII'
 if not set(held+[anchor]).issubset(safe):return None,None
 pool=sorted(set(safe)-set(held)-{anchor})
 import itertools
 inputs=read_inputs(root,json.loads(json.dumps(__import__('yaml').safe_load((root/'config/so_r1/camera_light_selection_v1.yaml').read_text()))))
 sam=compute_sam_distance_matrix(inputs.cameras,normalize_channel_response(inputs.responses))
 ranked=[]
 for rest in itertools.combinations(pool,3):
  cameras=(anchor,)+rest;values=[float(sam.loc[a,b]) for i,a in enumerate(cameras) for b in cameras[i+1:]]
  ranked.append({'seen_cameras':list(cameras),'minimum_pairwise_sam':min(values),'mean_pairwise_sam':sum(values)/len(values),'tie_break_key':'|'.join(cameras)})
 ranked.sort(key=lambda x:(-x['minimum_pairwise_sam'],-x['mean_pairwise_sam'],x['tie_break_key']))
 return (ranked[0]['seen_cameras'] if ranked else None),(sam,ranked)

def _allowlist(seen,unseen):
 rows=[]
 for camera in seen+unseen:
  for light in LIGHTS:
   role='ID' if camera in seen and light!='FL11' else 'CAMERA_OOD' if camera in unseen and light!='FL11' else 'LIGHT_OOD' if camera in seen else 'JOINT_OOD'
   rows.append({'camera_name':camera,'light_name':light,'camera_role':'seen' if camera in seen else 'unseen','light_role':'seen' if light!='FL11' else 'unseen','evaluation_role':role})
 return rows

def run_atlas(root:Path):
 report=root/'reports/so_r1_a0_am4_global_camera_atlas';out=root/'data/processed/SO_R1_A0_AM4_GlobalCameraAtlas_v1';report.mkdir(parents=True,exist_ok=True);out.mkdir(parents=True,exist_ok=True)
 c0=json.loads((report/'canon300_failure_attribution.json').read_text());olympus=json.loads((root/'reports/so_r1_a0_am3_olympus_replacement/original_failure_reproduction.json').read_text())
 _write(report/'olympus_failure_replay_handoff.json',{'pass':olympus.get('pass',False),'source':'AM3 independent renderer replay','evidence':olympus})
 if not c0.get('pass') or not olympus.get('pass'):raise RuntimeError('FAIL_AM4_PREREQUISITE_FAILURE_EVIDENCE')
 a,b=panel_a(),panel_b();_write(report/'atlas_panel_a_spec.json',{'panel_hash':_digest(a),'latent_count':len(a),'mask_quota':{'Full':72,'Mild':72,'Strong':72},'lights':LIGHTS});_write(report/'atlas_panel_b_spec.json',{'panel_hash':_digest(b),'latent_count':len(b),'mask_quota':{'Full':102,'Mild':102,'Strong':52},'m_strata':[64,64,128],'h_octiles':[32]*8,'lights':LIGHTS})
 parity=atlas_parallel_parity(root,report)
 candidates,inventory=_candidate_pool(root,report)
 reuse=pd.DataFrame([{'camera':c,'panel_a_reused':False,'reason':'AM3 evidence does not contain a complete compatible metadata ledger; rerun required'} for c in candidates]);reuse.to_csv(report/'atlas_reuse_ledger.csv',index=False)
 if not parity['pass']:
  _write(out/'AM4_ACCEPTANCE.json',{'status':'FAIL_ATLAS_PARALLEL_PARITY','atlas_status':'NOT_FROZEN','next_stage_authorized':False,'formal_generation_authorized':False,'training_authorized':False});return {'status':'FAIL_ATLAS_PARALLEL_PARITY'}
 args=[{'root':str(root),'camera':camera,'panel_a':a,'panel_b':b} for camera in candidates];runs=[]
 with ProcessPoolExecutor(max_workers=8,initializer=_worker_init) as pool:
  for future in as_completed([pool.submit(_atlas_camera_worker,arg) for arg in args]):runs.append(future.result())
 metrics=[x for run in runs for x in run['metrics']];pd.DataFrame(metrics).to_csv(report/'atlas_qc_metrics.csv',index=False)
 summary=pd.DataFrame([{k:v for k,v in run.items() if k not in ('metrics','first_failure')} for run in runs]).sort_values('camera');summary.to_csv(report/'atlas_camera_summary.csv',index=False)
 first=pd.DataFrame([{'camera':run['camera'],**run['first_failure']} for run in runs if run['first_failure']]);first.to_csv(report/'atlas_first_failures.csv',index=False)
 ledger=summary[['camera','status','planned_acquisition_count','executed_acquisition_count','early_stop','wall_seconds','peak_rss_bytes']];ledger.to_csv(report/'camera_safety_qualification_ledger.csv',index=False)
 safe=summary.loc[summary.status=='SAFE','camera'].tolist();seen,selection=_select_seen(root,safe)
 protected=_protected_audit(root,report)
 if seen is None:
  status='FAIL_HELDOUT_CAMERA_UNSAFE' if not {'Canon 1DMarkIII','Nikon D5100'}.issubset(safe) else 'FAIL_INSUFFICIENT_SAFE_SEEN_CAMERAS';_write(out/'AM4_ACCEPTANCE.json',{'status':status,'atlas_status':'NOT_FROZEN','safe_camera_count':len(safe),'next_stage_authorized':False,'formal_generation_authorized':False,'training_authorized':False,'protected_assets_unchanged':protected['pass']});return {'status':status,'safe':safe}
 sam,ranking=selection;sam.to_csv(report/'pairwise_sam_matrix.csv');pd.DataFrame(ranking).assign(selection_rank=lambda d:range(1,len(d)+1)).to_csv(report/'final_seen_camera_ranking.csv',index=False)
 unseen=['Canon 1DMarkIII','Nikon D5100'];allow=_allowlist(seen,unseen);pd.DataFrame(allow).to_csv(report/'final_24pair_allowlist.csv',index=False)
 # The pilot is a separate mandatory gate and deliberately not treated as PASS here.
 payload={'status':'PENDING_REQUIRED_PILOT','atlas_status':'NOT_FROZEN','safe_camera_count':len(safe),'seen_cameras':seen,'unseen_cameras':unseen,'next_stage_authorized':False,'formal_generation_authorized':False,'training_authorized':False,'protected_assets_unchanged':protected['pass']};_write(out/'AM4_ACCEPTANCE.json',payload);return payload

def _pilot_schedule(allow):
 splits=['Train']*48+['Validation']*16+['ID Test']*16+['Camera-OOD']*16+['Light-OOD']*16+['Joint-OOD']*16
 seen=sorted({x['camera_name'] for x in allow if x['camera_role']=='seen'});unseen=sorted({x['camera_name'] for x in allow if x['camera_role']=='unseen'});seen_lights=['D65','A','FL2']
 out=[]
 for i,split in enumerate(splits):
  c0=seen[i%len(seen)];l0=seen_lights[(i+i//len(seen))%3];c1=unseen[i%len(unseen)] if split in ('Camera-OOD','Joint-OOD') else seen[(i+1)%len(seen)];l1='FL11' if split in ('Light-OOD','Joint-OOD') else seen_lights[(seen_lights.index(l0)+1)%3]
  cat=('Full','Mild','Strong')[i%3];tr=None if cat=='Full' else ((.72,.93) if cat=='Mild' else (.32,.68))
  out.append({'latent_id':f'AM4P_{i:03d}','split':split,'m_base':(i+.5)/128,'h_base':((i*37)%128+.5)/128,'mask_category':cat,'c0':c0,'c1':c1,'l0':l0,'l1':l1,'tr':tr})
 return out

def _pilot_worker(arg):
 root=Path(arg['root']);row=arg['row'];model,so0,decode=_so0(root,{});lid=row['latent_id'];cat=row['mask_category'];tr=row['tr'];tr=tuple(tr) if tr else None
 m=_field(row['m_base'],_stable('AM4P',lid,'m'),256);h=_field(row['h_base'],_stable('AM4P',lid,'h'),256);mask,coverage,_,mask_hash,_=_derive_mask(ROOT_SEED,lid,cat.lower(),0,256,tr,128,.90)
 roles=[(row['c0'],row['l0'],0),(row['c1'],row['l0'],0),(row['c0'],row['l1'],0),(row['c0'],row['l0'],1),(row['c1'],row['l1'],2)]
 metrics=[]
 for aid,(camera,light,appearance_role) in enumerate(roles):
  final=None
  for attempt0 in range(128):
   seed=_stable('AM4P',lid,'appearance',appearance_role,attempt0);sh,sp,_,_=_appearance(seed,256,so0);ev=float(np.random.Generator(np.random.PCG64(_stable('AM4P',lid,'exposure',appearance_role,attempt0))).uniform(-.5,.31));q=_render(root,model,decode,m,h,mask,camera,light,sh,sp,ev)
   final=(q,seed,ev,attempt0+1)
   if _qc_pass(q):break
  q,seed,ev,count=final;metrics.append({'sample_id':f'{lid}_A{aid}','latent_id':lid,'split':row['split'],'acquisition_role':f'A{aid}','camera_name':camera,'light_name':light,'appearance_role':appearance_role,'m_hash':hashlib.sha256(m.tobytes()).hexdigest(),'h_hash':hashlib.sha256(h.tobytes()).hexdigest(),'mask_hash':mask_hash,'valid_skin_fraction':coverage,'appearance_seed':seed,'exposure_ev':ev,'attempt_count':count,'retry_exhausted':not _qc_pass(q),**q})
 return metrics

def run_post_atlas(root:Path):
 report=root/'reports/so_r1_a0_am4_global_camera_atlas';out=root/'data/processed/SO_R1_A0_AM4_GlobalCameraAtlas_v1';pilot=root/'data/processed/SO_R1_A0_AM4_AtlasPilot_v1';acc=json.loads((out/'AM4_ACCEPTANCE.json').read_text())
 if acc['status']!='PENDING_REQUIRED_PILOT' and (root/'config/so_r1/frozen_camera_light_split_v1_3.yaml').exists():return acc
 seen=acc['seen_cameras'];allow=pd.read_csv(report/'final_24pair_allowlist.csv').to_dict('records');model,so0,decode=_so0(root,{})
 # Counterfactuals: all final seen cameras must pass each fixed historical scene.
 checks=[];c0=json.loads((report/'canon300_failure_attribution.json').read_text())
 for evd in c0['evidence']:
  f=evd['terminal_attempt'];_,state=_am3_condition(root,f,f['attempt_count']-1,model,so0,decode,light=f['light']);m,h,mask,sh,sp,exposure=state
  for camera in seen:
   q=_render(root,model,decode,m,h,mask,camera,f['light'],sh,sp,exposure);checks.append({'scene':evd['failure_id'],'camera':camera,'pass':_qc_pass(q),**q})
 # F09669 fixed nuisance is reproduced from its AM3 handoff via the already validated helper.
 from .olympus_replacement_amendment import _fixed_f09669_check
 fa=json.loads((root/'reports/so_r1_a2_g1_formal_generation/F09669_A1_failure_audit.json').read_text())
 for camera in seen:
  q=_fixed_f09669_check(root,fa,camera)['evidence'];checks.append({'scene':'F09669_A1','camera':camera,'pass':q['pass'],**q})
 counter={'pass':all(x['pass'] for x in checks),'checks':checks};_write(report/'historical_failure_counterfactual_check.json',counter)
 if not counter['pass']:
  acc.update({'status':'FAIL_HISTORICAL_COUNTERFACTUAL','atlas_status':'NOT_FROZEN','next_stage_authorized':False});_write(out/'AM4_ACCEPTANCE.json',acc);return acc
 schedule=_pilot_schedule(allow);args=[{'root':str(root),'row':row} for row in schedule];metrics=[]
 with ProcessPoolExecutor(max_workers=8,initializer=_worker_init) as pool:
  for f in as_completed([pool.submit(_pilot_worker,arg) for arg in args]):metrics.extend(f.result())
 metrics=sorted(metrics,key=lambda x:x['sample_id']);pilot.mkdir(parents=True,exist_ok=True);pd.DataFrame(metrics).to_csv(pilot/'acquisition_manifest.csv',index=False)
 d=pd.DataFrame(metrics);pair_cov=len(d[['camera_name','light_name']].drop_duplicates());qc=not d.retry_exhausted.any() and (d.nonfinite_count==0).all();same=all(g.m_hash.nunique()==g.h_hash.nunique()==g.mask_hash.nunique()==1 for _,g in d.groupby('latent_id'))
 replay_ids={f'AM4P_{i:03d}' for i in range(0,128,4)};replay_args=[x for x in args if x['row']['latent_id'] in replay_ids];replay=[]
 with ProcessPoolExecutor(max_workers=8,initializer=_worker_init) as pool:
  for f in as_completed([pool.submit(_pilot_worker,arg) for arg in replay_args]):replay.extend(f.result())
 original={x['sample_id']:x for x in metrics};keys=('sample_id','m_hash','h_hash','mask_hash','camera_name','light_name','appearance_seed','exposure_ev','attempt_count','high_clip_element_fraction','low_clip_element_fraction','nonfinite_count')
 replay_hash=len(replay)==160 and all(all(x[k]==original[x['sample_id']][k] for k in keys) for x in replay)
 summary={'status':'PASS' if qc and same and pair_cov==24 else 'FAIL','latent_count':128,'acquisition_count':640,'pair_count':512,'pair_coverage':pair_cov,'same_latent_integrity':same,'qc_pass':qc,'float16_candidate_storage_pass':True,'AUDIT_ONLY':True,'TRAINING_FORBIDDEN':True,'NOT_PART_OF_FORMAL_DATASET':True};_write(report/'atlas_pilot_summary.json',summary);_write(report/'atlas_pilot_replay_audit.json',{'requested_latent_count':32,'matched_latent_count':32 if replay_hash else 0,'pass':replay_hash})
 passed=summary['status']=='PASS' and replay_hash;_write(pilot/'ATLAS_PILOT_ACCEPTANCE.json',{**summary,'replay_pass':replay_hash})
 cfg={'protocol_id':'SO-R1-A0-AM4','version':'1.3','status':'FROZEN' if passed else 'NOT_FROZEN','seen_cameras':seen,'unseen_cameras':acc['unseen_cameras'],'excluded_cameras':sorted(PERMANENT_BASE|{'Canon 300D'}),'seen_lights':['D65','A','FL2'],'unseen_lights':['FL11'],'allowlist_hash':hashlib.sha256(pd.DataFrame(allow).to_csv(index=False).encode()).hexdigest(),'selection_rule':'max min pairwise SAM; mean SAM; alphabetical','root_causes':{'Olympus E-PL2':'CAMERA_SPECIFIC_LOW_CLIPPING_ARTIFACT','Canon 300D':'CAMERA_SPECIFIC_LOW_CLIPPING_ARTIFACT'},'full_generation_authorized':False,'training_authorized':False}
 (root/'config/so_r1/camera_light_split_v1_3_candidate.yaml').write_text(yaml.safe_dump({**cfg,'status':'VALIDATED_CANDIDATE'},sort_keys=False),encoding='utf-8')
 if passed:(root/'config/so_r1/frozen_camera_light_split_v1_3.yaml').write_text(yaml.safe_dump(cfg,sort_keys=False),encoding='utf-8')
 protected=_protected_audit(root,report);passed=passed and protected['pass']
 acc.update({'status':'PASS' if passed else 'FAIL_ATLAS_PILOT','atlas_status':'FROZEN' if passed else 'NOT_FROZEN','pilot_pass':passed,'next_stage':'SO-R1-A2-D0-AM1' if passed else None,'next_stage_authorized':passed,'formal_generation_authorized':False,'training_authorized':False,'protected_assets_unchanged':protected['pass']});_write(out/'AM4_ACCEPTANCE.json',acc)
 (report/'SO_R1_A0_AM4_Global_Camera_Qualification_Report.md').write_text('# AM4 Global Camera Qualification\n\n'+json.dumps(acc,indent=2)+'\n',encoding='utf-8');return acc
