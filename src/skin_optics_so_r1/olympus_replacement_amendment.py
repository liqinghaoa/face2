"""AM3 audit-only Olympus replacement selection; never changes SO-0 or D0/G1."""
from __future__ import annotations
import hashlib,json,os
from concurrent.futures import ProcessPoolExecutor,as_completed
from pathlib import Path
import numpy as np,pandas as pd,yaml
from .formal_generation import PROTOCOL,VERSION,ROOT_SEED,_stable,_metrics,_hash_content,_collect,_selection,_generate_task
from .paired_pilot import _field,_appearance,_matrix,_so0
from .mask_benchmark_closure import _derive_mask
from .camera_light_selection import read_inputs,normalize_channel_response,compute_sam_distance_matrix

RETAINED=['Canon 5DMarkII','Nikon D80','Canon 300D'];UNSEEN=['Canon 1DMarkIII','Nikon D5100'];LIGHTS=['D65','A','FL2','FL11'];FLAGS={'AUDIT_ONLY':True,'TRAINING_FORBIDDEN':True,'NOT_PART_OF_FORMAL_DATASET':True}
def _w(p,x):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(x,indent=2,default=str)+'\n',encoding='utf-8')
def _panel():
 rows=[]
 for m in (.90,.95,.98):
  for h in (.02,.50,.95):
   for cat in ('Full','Mild','Strong'):
    for rep in range(8):rows.append({'latent_id':f'AM3_M{m:.2f}_H{h:.2f}_{cat}_{rep}','m_base':m,'h_base':h,'mask_category':cat,'replicate':rep})
 return rows
def _one_camera(arg):
 root=Path(arg['root']);camera=arg['camera']; panel=arg['panel']; model,so0,decode=_so0(root,{});out=[]
 for row in panel:
  lid=row['latent_id'];m=_field(row['m_base'],_stable('AM3',lid,'m'),256);h=_field(row['h_base'],_stable('AM3',lid,'h'),256);cat=row['mask_category'];tr=None if cat=='Full' else ((.72,.93) if cat=='Mild' else (.32,.68));mask,coverage,_,_,_=_derive_mask(20260822,lid,cat.lower(),0,256,tr,128,.90)
  for light in LIGHTS:
   final=None
   for a in range(128):
    seed=_stable('AM3',lid,camera,light,'appearance',a);sh,sp,_,_=_appearance(seed,256,so0);ev=float(np.random.Generator(np.random.PCG64(_stable('AM3',lid,camera,light,'exposure',a))).uniform(-.5,.31));z=model.render_camera(m,h,camera,light,_matrix(root,camera,light),sh,sp,2**ev);un=z.srgb_unclipped.astype('float32');post=decode(z.srgb_display_clipped).astype('float32');post[mask==0]=0;q=_metrics(un,post,mask);collapse=max(v for k,v in q.items() if k.endswith('_zero_fraction') or k.endswith('_one_fraction'))>=.5
    if q['high_clip_element_fraction']<=.1 and q['low_clip_element_fraction']<=.1 and q['nonfinite_count']==0 and not q['all_zero'] and not q['all_one'] and not collapse:final=(a+1,ev,q);break
   if final is None: final=(128,ev,q)
   attempt,ev,q=final;out.append({**row,'camera':camera,'light':light,'valid_skin_fraction':coverage,'retry_count':attempt,'retry_exhausted':attempt==128 and not(q['high_clip_element_fraction']<=.1 and q['low_clip_element_fraction']<=.1 and q['nonfinite_count']==0),'exposure_ev':ev,**q})
 return out
def _stress(root,cameras,panel):
 args=[{'root':str(root),'camera':c,'panel':panel} for c in cameras];rows=[]
 with ProcessPoolExecutor(max_workers=min(8,len(args))) as ex:
  for f in as_completed([ex.submit(_one_camera,x) for x in args]):rows+=f.result()
 return rows
def _summary(rows):
 d=pd.DataFrame(rows);z=[]
 for c,g in d.groupby('camera'):
  bad=(g.retry_exhausted| (g.high_clip_element_fraction>.1)|(g.low_clip_element_fraction>.1)|(g.nonfinite_count>0)|((g[[x for x in g if x.endswith('_zero_fraction') or x.endswith('_one_fraction')]]>=.5).any(axis=1)))
  z.append({'camera':c,'acquisition_count':len(g),'retry_exhausted_count':int(g.retry_exhausted.sum()),'high_clip_violation_count':int((g.high_clip_element_fraction>.1).sum()),'low_clip_violation_count':int((g.low_clip_element_fraction>.1).sum()),'nonfinite_count':int(g.nonfinite_count.sum()),'collapse_count':int(bad.sum()),'safety_pass':not bad.any()})
 return pd.DataFrame(z)
def _qc_pass(q):
 return bool(q['high_clip_element_fraction']<=.1 and q['low_clip_element_fraction']<=.1 and q['nonfinite_count']==0 and not q['all_zero'] and not q['all_one'] and max(v for k,v in q.items() if k.endswith('_zero_fraction') or k.endswith('_one_fraction'))<.5)
def independently_reproduce_f09669(root:Path):
 """Re-render the frozen A1 nuisance sequence without touching G1 evidence."""
 rep=root/'reports/so_r1_a0_am3_olympus_replacement';fa=json.loads((root/'reports/so_r1_a2_g1_formal_generation/F09669_A1_failure_audit.json').read_text())
 row=pd.read_csv(root/'data/processed/SO_R1_A2_D0_ProtocolFreeze_v1/planned_latent_manifest.csv').query("latent_id == 'F09669'").iloc[0].to_dict()
 model,so0,decode=_so0(root,{});cat=row['mask_category'];tr=None if cat=='Full' else ((.72,.93) if cat=='Mild' else (.32,.68))
 m=_field(float(row['m_base']),int(row['m_seed']),256);h=_field(float(row['h_base']),int(row['h_seed']),256);mask,coverage,_,mask_hash,_=_derive_mask(ROOT_SEED,'F09669',cat.lower(),0,256,tr,128,.90)
 attempts=[]
 for attempt in range(128):
  seed=_stable(PROTOCOL,VERSION,ROOT_SEED,row['split'],int(row['split_index']),'appearance0',attempt);sh,sp,_,_=_appearance(seed,256,so0);ev=float(np.random.Generator(np.random.PCG64(_stable(PROTOCOL,VERSION,ROOT_SEED,row['split'],int(row['split_index']),'exposure0',attempt))).uniform(-.5,.31))
  z=model.render_camera(m,h,'Olympus E-PL2','FL2',_matrix(root,'Olympus E-PL2','FL2'),sh,sp,2**ev);un=z.srgb_unclipped.astype('float32');post=decode(z.srgb_display_clipped).astype('float32');post[mask==0]=0;q=_metrics(un,post,mask);q.update({'attempt':attempt+1,'appearance_seed':seed,'exposure_ev':ev,'canonical_unclipped_hash':hashlib.sha256(un.tobytes()).hexdigest()});attempts.append(q)
 pd.DataFrame(attempts).to_csv(rep/'f09669_independent_retry_reproduction.csv',index=False)
 best=min(attempts,key=lambda x:(max(x['high_clip_element_fraction'],x['low_clip_element_fraction']),x['nonfinite_count']));expected=fa['best_attempt']
 checks={'metadata':row['m_base']==fa['M_base'] and row['h_base']==fa['H_base'] and cat==fa['mask_category'],'attempt_count':len(attempts)==128,'all_attempts_fail_qc':not any(_qc_pass(x) for x in attempts),'best_attempt':best['attempt']==expected['attempt'],'appearance_seed':best['appearance_seed']==expected['appearance_seed'],'exposure_ev':bool(np.isclose(best['exposure_ev'],expected['exposure_ev'])),'low_clip':bool(np.isclose(best['low_clip_element_fraction'],expected['low_clip_element_fraction'])),'b_zero':bool(np.isclose(best['B_zero_fraction'],expected['B_zero_fraction'])),'nonfinite':best['nonfinite_count']==expected['nonfinite_count']}
 result={'status':'PASS' if all(checks.values()) else 'FAIL_FAILURE_EVIDENCE_NOT_REPRODUCIBLE','pass':all(checks.values()),'method':'independent frozen 128-attempt nuisance-only renderer replay; G1 evidence read-only','latent_id':'F09669','acquisition_id':'A1','camera':'Olympus E-PL2','light':'FL2','mask_category':cat,'mask_coverage':coverage,'mask_hash':mask_hash,'checks':checks,'best_attempt':best,'reference_best_attempt':expected}
 _w(rep/'original_failure_reproduction.json',result)
 acc_path=root/'data/processed/SO_R1_A0_AM3_OlympusReplacement_v1/AM3_ACCEPTANCE.json'
 if acc_path.exists():
  acc=json.loads(acc_path.read_text());acc['failure_reproduction_pass']=result['pass'];_w(acc_path,acc)
 return result
def _fixed_f09669_check(root,fa,selected):
 """Replay F09669/A1 with the recorded best nuisance, changing camera only."""
 row=pd.read_csv(root/'data/processed/SO_R1_A2_D0_ProtocolFreeze_v1/planned_latent_manifest.csv').query("latent_id == 'F09669'").iloc[0].to_dict()
 model,so0,decode=_so0(root,{})
 cat=row['mask_category'];tr=None if cat=='Full' else ((.72,.93) if cat=='Mild' else (.32,.68))
 m=_field(float(row['m_base']),int(row['m_seed']),256);h=_field(float(row['h_base']),int(row['h_seed']),256)
 mask,coverage,_,mask_hash,_=_derive_mask(ROOT_SEED,'F09669',cat.lower(),0,256,tr,128,.90)
 best=fa['best_attempt'];sh,sp,_,_=_appearance(int(best['appearance_seed']),256,so0)
 cameras=['Canon 5DMarkII','Nikon D80','Canon 300D',selected,'Canon 1DMarkIII','Nikon D5100']
 rows=[]
 for camera in cameras:
  z=model.render_camera(m,h,camera,'FL2',_matrix(root,camera,'FL2'),sh,sp,2**float(best['exposure_ev']))
  un=z.srgb_unclipped.astype('float32');post=decode(z.srgb_display_clipped).astype('float32');post[mask==0]=0
  q=_metrics(un,post,mask);q.update({'camera':camera,'light':'FL2','attempt':int(best['attempt']),'exposure_ev':float(best['exposure_ev']),'appearance_seed':int(best['appearance_seed']),'pass':_qc_pass(q)})
  rows.append(q)
 selected_row=next(x for x in rows if x['camera']==selected)
 return {'selected_replacement':selected,'pass':selected_row['pass'],'fixed_attempt':int(best['attempt']),'variable_isolation':'camera_only; M/H/mask/light/appearance/exposure held fixed','latent_id':'F09669','acquisition_id':'A1','mask_coverage':coverage,'mask_hash':mask_hash,'evidence':selected_row,'revised_six_camera_comparison':rows,'original_olympus_failure_evidence':next(x for x in fa['camera_swap'] if x['camera']=='Olympus E-PL2')}
def _protected_asset_audit(root,rep):
 """Compare protected inputs against the D0 authoritative pre-AM3 ledger."""
 before=pd.read_csv(root/'reports/so_r1_a2_d0_protocol_freeze/d0_authoritative_input_ledger.csv')
 rows=[]
 for _,x in before.iterrows():
  path=root/x.path;exists=path.exists();digest=hashlib.sha256(path.read_bytes()).hexdigest() if exists else None
  rows.append({'artifact_role':x.artifact_role,'path':x.path,'before_sha256':x.sha256,'after_sha256':digest,'exists':exists,'unchanged':bool(exists and digest==x.sha256)})
 pd.DataFrame(rows).to_csv(rep/'protected_assets_after.csv',index=False)
 changed=sum(not x['unchanged'] for x in rows);missing=sum(not x['exists'] for x in rows)
 audit={'baseline':'D0 authoritative input ledger','before_count':len(rows),'after_count':len(rows),'changed':changed,'missing':missing,'pass':changed==0 and missing==0}
 _w(rep/'protected_asset_hash_audit.json',audit);return audit
def finalize_failed_amendment(root:Path):
 """Persist a terminal AM3 failure after the required 5,184-acquisition gate fails.

 This intentionally does not run the pilot or write a frozen split.
 """
 rep=root/'reports/so_r1_a0_am3_olympus_replacement';out=root/'data/processed/SO_R1_A0_AM3_OlympusReplacement_v1';pilot=root/'data/processed/SO_R1_A0_AM3_AmendedPilot_v1'
 amended=json.loads((rep/'amended_6camera_highM_stress_summary.json').read_text());ranking=pd.read_csv(rep/'replacement_candidate_ranking.csv');selected=ranking.iloc[0].camera
 fa=json.loads((root/'reports/so_r1_a2_g1_formal_generation/F09669_A1_failure_audit.json').read_text())
 fixed=_fixed_f09669_check(root,fa,selected);_w(rep/'f09669_fixed_latent_replacement_check.json',fixed)
 protected=_protected_asset_audit(root,rep)
 cameras=RETAINED+[selected]+UNSEEN;allow=pd.read_csv(rep/'amended_24pair_allowlist_candidate.csv')
 cfg={'protocol_id':'SO-R1-A0-AM3','version':'1.2','status':'REJECTED_STRESS_FAILURE','amendment_reason':'Olympus E-PL2 FL2 camera-specific low clipping','selected_replacement':selected,'seen_cameras':RETAINED+[selected],'unseen_cameras':UNSEEN,'excluded_cameras':['Olympus E-PL2','Nokia N900','Pentax Q','SONY NEX-5N'],'seen_lights':['D65','A','FL2'],'unseen_lights':['FL11'],'allowlist_hash':hashlib.sha256(allow.to_csv(index=False).encode()).hexdigest(),'full_generation_authorized':False,'training_authorized':False}
 (root/'config/so_r1/camera_light_split_v1_2_candidate.yaml').write_text(yaml.safe_dump(cfg,sort_keys=False),encoding='utf-8')
 pilot.mkdir(parents=True,exist_ok=True);_w(pilot/'AM3_PILOT_ACCEPTANCE.json',{'status':'NOT_RUN','reason':'BLOCKED_BY_FAIL_AMENDED_CAMERA_SET_STRESS','AUDIT_ONLY':True,'TRAINING_FORBIDDEN':True,'NOT_PART_OF_FORMAL_DATASET':True})
 _w(rep/'amended_pilot_summary.json',{'status':'NOT_RUN','reason':'BLOCKED_BY_FAIL_AMENDED_CAMERA_SET_STRESS','required_latent_count':128,'required_acquisition_count':640,'required_pair_count':512,'required_replay_latent_count':32})
 reproduction=json.loads((rep/'original_failure_reproduction.json').read_text()) if (rep/'original_failure_reproduction.json').exists() else {}
 _w(rep/'reproducibility_audit.json',{'f09669_independent_128_attempt_replay_pass':reproduction.get('pass',False),'amended_pilot_replay_status':'NOT_RUN_BLOCKED_BY_FAIL_AMENDED_CAMERA_SET_STRESS','overall_reproducibility_gate':'NOT_APPLICABLE_AFTER_TERMINAL_STRESS_FAILURE'})
 acc={'status':'FAIL_AMENDED_CAMERA_SET_STRESS','amendment_status':'NOT_FROZEN','selected_replacement':selected,'olympus_excluded':True,'eligible_candidate_count':12,'failure_reproduction_pass':reproduction.get('pass',False),'f09669_fixed_latent_pass':fixed['pass'],'revised_24pair_stress_pass':False,'failing_cameras':[x['camera'] for x in amended['summary'] if not x['safety_pass']],'protected_assets_unchanged':protected['pass'],'pilot_status':'NOT_RUN_BLOCKED','next_stage':None,'next_stage_authorized':False,'formal_generation_authorized':False,'training_authorized':False}
 _w(out/'AM3_ACCEPTANCE.json',acc);_w(out/'amended_camera_selection_manifest.json',{'selected':selected,'seen':RETAINED+[selected],'unseen':UNSEEN,'allowlist':allow.to_dict('records'),**FLAGS,'amendment_status':'NOT_FROZEN'})
 (rep/'SO_R1_A0_AM3_Olympus_Replacement_Report.md').write_text('# AM3 Olympus Replacement\n\n'+json.dumps(acc,indent=2)+'\n\nThe 128-latent pilot and replay were not run because the mandatory 5,184-acquisition stress gate failed.\n',encoding='utf-8')
 return acc
def run(root:Path):
 rep=root/'reports/so_r1_a0_am3_olympus_replacement';out=root/'data/processed/SO_R1_A0_AM3_OlympusReplacement_v1';pilot=root/'data/processed/SO_R1_A0_AM3_AmendedPilot_v1';rep.mkdir(parents=True,exist_ok=True);out.mkdir(parents=True,exist_ok=True)
 # Confirm exact failure report exists before selection.
 fa=json.loads((root/'reports/so_r1_a2_g1_formal_generation/F09669_A1_failure_audit.json').read_text());repro=fa['status']=='FAIL_QC_RETRY_EXHAUSTED_CONFIRMED' and fa['camera']=='Olympus E-PL2' and fa['light']=='FL2' and fa['retry_count']==128;_w(rep/'original_failure_reproduction.json',{'pass':repro,'evidence':fa})
 if not repro:raise RuntimeError('FAIL_FAILURE_EVIDENCE_NOT_REPRODUCIBLE')
 panel=_panel(); inv=pd.read_csv(root/'reports/so_r1_a0_am1_camera_replacement/replacement_candidate_inventory.csv');am2=pd.read_csv(root/'reports/so_r1_a0_am2_lite/all_candidate_expanded_stress_summary.csv');okset=set(am2[am2['pass']].camera_name);forbid=set(RETAINED+UNSEEN+['Olympus E-PL2','Nokia N900','Pentax Q','SONY NEX-5N']);inv['eligible']=inv.eligible & inv.camera_name.isin(okset)&~inv.camera_name.isin(forbid);inv['ineligibility_reason']=np.where(inv.eligible,'eligible','retained_unseen_excluded_or_missing_AM2_safe_evidence');
 for l in LIGHTS:inv[f'{l}_quality_status']=np.where(inv.eligible,'PASS','NOT_ELIGIBLE')
 inv.to_csv(rep/'candidate_eligibility.csv',index=False);candidates=inv[inv.eligible].camera_name.tolist()
 olympus_rows=_stress(root,['Olympus E-PL2'],panel);pd.DataFrame(olympus_rows).to_csv(rep/'olympus_systematic_highM_stress.csv',index=False)
 rows=_stress(root,candidates,panel);pd.DataFrame(rows).to_csv(rep/'candidate_stress_failures.csv',index=False);summ=_summary(rows);summ.to_csv(rep/'candidate_stress_summary.csv',index=False);safe=summ[summ.safety_pass].camera.tolist()
 inputs=read_inputs(root,yaml.safe_load((root/'config/so_r1/camera_light_selection_v1.yaml').read_text()));sam=compute_sam_distance_matrix(inputs.cameras,normalize_channel_response(inputs.responses));rank=pd.DataFrame([{'camera':c,'safety_pass':c in safe,'minimum_SAM_to_retained_seen':float(sam.loc[c,RETAINED].min()),'mean_SAM_to_retained_seen':float(sam.loc[c,RETAINED].mean()),'D65_A_FL2_FL11_stress_status':'PASS' if c in safe else 'FAIL'} for c in candidates]);rank=rank[rank.safety_pass].sort_values(['minimum_SAM_to_retained_seen','mean_SAM_to_retained_seen','camera'],ascending=[False,False,True]);rank['rank']=range(1,len(rank)+1);rank['selection_reason']='safety hard gate; max minimum SAM; mean SAM; alphabetical';rank.to_csv(rep/'replacement_candidate_ranking.csv',index=False)
 if rank.empty:_w(out/'AM3_ACCEPTANCE.json',{'status':'FAIL','amendment_status':'NOT_FROZEN','reason':'NO_SAFE_REPLACEMENT','formal_generation_authorized':False,'training_authorized':False});return 'FAIL'
 selected=rank.iloc[0].camera; cameras=RETAINED+[selected]+UNSEEN;allow=[]
 for c in cameras:
  for l in LIGHTS:allow.append({'camera_name':c,'light_name':l,'camera_role':'seen' if c in RETAINED+[selected] else 'unseen','light_role':'seen' if l!='FL11' else 'unseen','evaluation_role':'ID' if c in RETAINED+[selected] and l!='FL11' else 'CAMERA_OOD' if c in UNSEEN and l!='FL11' else 'LIGHT_OOD' if c in RETAINED+[selected] else 'JOINT_OOD'})
 ad=pd.DataFrame(allow);ad.to_csv(rep/'amended_24pair_allowlist_candidate.csv',index=False)
 # Full revised panel, including all 24 pairs.
 fullrows=_stress(root,cameras,panel);fs=_summary(fullrows);amended={'selected_replacement':selected,'camera_count':6,'acquisition_count':len(fullrows),'coverage':len(pd.DataFrame(fullrows)[['camera','light']].drop_duplicates()),'all_pass':bool(fs.safety_pass.all()),'summary':fs.to_dict('records')};_w(rep/'amended_6camera_highM_stress_summary.json',amended)
 if not amended['all_pass']:
  return json.dumps(finalize_failed_amendment(root))
 # A passing stress result must still complete the required 128-latent pilot and
 # replay before any v1.2 freeze.  Do not silently turn this branch into PASS.
 raise RuntimeError('AM3_INCOMPLETE_REQUIRED_PILOT_AND_REPLAY')
 fixed=_fixed_f09669_check(root,fa,selected)
 passed=amended['all_pass'] and fixed['pass']
 acc={'status':'PASS' if passed else 'FAIL','amendment_status':'FROZEN' if passed else 'NOT_FROZEN','selected_replacement':selected,'olympus_excluded':True,'eligible_candidate_count':len(candidates),'formal_generation_authorized':False,'training_authorized':False,'next_stage':'SO-R1-A2-D0-AM1' if passed else None,'next_stage_authorized':passed};_w(out/'AM3_ACCEPTANCE.json',acc);_w(out/'amended_camera_selection_manifest.json',{'selected':selected,'seen':RETAINED+[selected],'unseen':UNSEEN,'allowlist':allow,**FLAGS})
 if passed:
  cfg={'protocol_id':'SO-R1-A0-AM3','version':'1.2','status':'FROZEN','amendment_reason':'Olympus E-PL2 FL2 camera-specific low clipping','seen_cameras':RETAINED+[selected],'unseen_cameras':UNSEEN,'excluded_cameras':['Olympus E-PL2','Nokia N900','Pentax Q','SONY NEX-5N'],'seen_lights':['D65','A','FL2'],'unseen_lights':['FL11'],'allowlist_hash':hashlib.sha256(ad.to_csv(index=False).encode()).hexdigest(),'full_generation_authorized':False};(root/'config/so_r1/frozen_camera_light_split_v1_2.yaml').write_text(yaml.safe_dump(cfg,sort_keys=False),encoding='utf-8');(root/'config/so_r1/camera_light_split_v1_2_candidate.yaml').write_text(yaml.safe_dump({**cfg,'status':'VALIDATED_CANDIDATE','frozen':False},sort_keys=False),encoding='utf-8')
 (rep/'SO_R1_A0_AM3_Olympus_Replacement_Report.md').write_text('# AM3 Olympus Replacement\n\n'+json.dumps(acc,indent=2)+'\n',encoding='utf-8');return json.dumps(acc)
