"""Audit a terminal G1 QC failure without changing the frozen protocol or data."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd
from .formal_generation import PROTOCOL,VERSION,ROOT_SEED,_stable,_metrics
from .paired_pilot import _field,_appearance,_matrix,_so0
from .mask_benchmark_closure import _derive_mask

def _write(p,x):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(x,indent=2,default=str)+'\n',encoding='utf-8')
def _render(root,model,decode,m,h,mask,cam,light,sh,sp,ev):
 z=model.render_camera(m,h,cam,light,_matrix(root,cam,light),sh,sp,2**ev);un=z.srgb_unclipped.astype('float32');post=decode(z.srgb_display_clipped).astype('float32');post[mask==0]=0;return _metrics(un,post,mask)
def run(root:Path,lid:str,acq_index:int):
 rep=root/'reports/so_r1_a2_g1_formal_generation';r=pd.read_csv(root/'data/processed/SO_R1_A2_D0_ProtocolFreeze_v1/planned_latent_manifest.csv');row=r[r.latent_id==lid].iloc[0].to_dict();model,so0,decode=_so0(root,{})
 cat=row['mask_category']; tr=None if cat=='Full' else ((.72,.93) if cat=='Mild' else (.32,.68));m=_field(float(row['m_base']),int(row['m_seed']),256);h=_field(float(row['h_base']),int(row['h_seed']),256);mask,coverage,mask_try,mask_hash,lcc=_derive_mask(ROOT_SEED,lid,cat.lower(),0,256,tr,128,.90)
 combos=[(row['c0'],row['l0'],0),(row['c1'],row['l0'],0),(row['c0'],row['l1'],0),(row['c0'],row['l0'],1),(row['c1'],row['l1'],2)];cam,light,a=combos[acq_index];attempts=[]
 for attempt in range(128):
  seed=_stable(PROTOCOL,VERSION,ROOT_SEED,row['split'],int(row['split_index']),f'appearance{a}',attempt);sh,sp,_,_=_appearance(seed,256,so0);ev=float(np.random.Generator(np.random.PCG64(_stable(PROTOCOL,VERSION,ROOT_SEED,row['split'],int(row['split_index']),f'exposure{a}',attempt))).uniform(-.5,.31));q=_render(root,model,decode,m,h,mask,cam,light,sh,sp,ev);q.update({'attempt':attempt+1,'appearance_seed':seed,'exposure_ev':ev,'S_mean':float(sh.mean()),'P_mean':float(sp.mean()),'P_max':float(sp.max())});attempts.append(q)
 pd.DataFrame(attempts).to_csv(rep/'F09669_A1_qc_retry_audit.csv',index=False);best=min(attempts,key=lambda x:(max(x['high_clip_element_fraction'],x['low_clip_element_fraction']),x['nonfinite_count']))
 # Hold the best nuisance sample fixed and isolate the camera response.
 seed=best['appearance_seed'];sh,sp,_,_=_appearance(seed,256,so0);swaps=[]
 for c in ['Canon 5DMarkII','Nikon D80','Olympus E-PL2','Canon 300D','Canon 1DMarkIII','Nikon D5100']:
  q=_render(root,model,decode,m,h,mask,c,light,sh,sp,best['exposure_ev']);q.update({'camera':c,'light':light,'exposure_ev':best['exposure_ev'],'attempt':best['attempt']});swaps.append(q)
 pd.DataFrame(swaps).to_csv(rep/'F09669_A1_camera_swap_audit.csv',index=False)
 curve=[]
 for ev in [-.5,-.4,-.3,-.2,-.1,0,.1,.2,.31]:
  q=_render(root,model,decode,m,h,mask,cam,light,sh,sp,ev);q.update({'exposure_ev':ev,'camera':cam,'light':light,'attempt':best['attempt']});curve.append(q)
 pd.DataFrame(curve).to_csv(rep/'F09669_A1_exposure_response_audit.csv',index=False)
 any_pass=any(x['high_clip_element_fraction']<=.1 and x['low_clip_element_fraction']<=.1 and x['nonfinite_count']==0 and not x['all_zero'] and not x['all_one'] and max(v for k,v in x.items() if k.endswith('_zero_fraction') or k.endswith('_one_fraction'))<.5 for x in attempts)
 summary={'status':'FAIL_QC_RETRY_EXHAUSTED_CONFIRMED','latent_id':lid,'acquisition_id':f'A{acq_index}','camera':cam,'light':light,'split':row['split'],'M_base':row['m_base'],'H_base':row['h_base'],'mask_category':cat,'mask_coverage':coverage,'retry_count':128,'any_qc_pass':any_pass,'best_attempt':best,'camera_swap':swaps,'exposure_curve':curve,'frozen_protocol_modified':False,'formal_dataset_not_accepted':True}
 _write(rep/'F09669_A1_failure_audit.json',summary)
 (rep/'F09669_A1_failure_audit_report.md').write_text('# F09669 A1 Failure Audit\n\nThe frozen 128-attempt nuisance-only retry sequence was replayed exactly. This report records every attempt, the best observed attempt, a fixed-nuisance camera swap, and fixed-nuisance exposure response. No formal data, frozen protocol, M/H, mask, camera, light, or split was modified.\n\n'+json.dumps(summary,indent=2,default=str)+'\n',encoding='utf-8')
 _write(root/'data/processed/SO_R1_A2_FormalPairedDataset_v1/GENERATION_STATUS.json',{'status':'FAIL_QC_RETRY_EXHAUSTED','failed_latent_id':lid,'failed_acquisition_id':f'A{acq_index}','retry_count':128,'generation_stopped':True,'formal_dataset_accepted':False,'training_authorized':False})
 return json.dumps(summary)
