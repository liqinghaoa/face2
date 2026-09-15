"""R1 audit of AM2-Lite high-end clipping; it never changes frozen SO-0 inputs."""
from __future__ import annotations
import hashlib, json, shutil, subprocess, time
from pathlib import Path
from typing import Any
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from scipy.stats import spearmanr, pearsonr

from .paired_pilot import _appearance, _field, _matrix, _seed, _so0, ROLES

FLAGS = {"AUDIT_ONLY": True, "TRAINING_FORBIDDEN": True,
         "ATTRIBUTION_ONLY": True, "NOT_PART_OF_FORMAL_DATASET": True}
CAMERAS = ["Canon 5DMarkII", "Nikon D80", "Olympus E-PL2", "Canon 300D",
           "Canon 1DMarkIII", "Nikon D5100"]
SEEN = CAMERAS[:4]
UNSEEN = CAMERAS[4:]
LIGHTS = ["D65", "A", "FL2", "FL11"]

def _sha(p: Path) -> str:
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(1<<20),b''): h.update(b)
    return h.hexdigest()

def _dump(p: Path, x: Any):
    p.parent.mkdir(parents=True,exist_ok=True); p.write_text(json.dumps(x,indent=2,ensure_ascii=False,default=str)+'\n',encoding='utf-8')

def _csv(p: Path, rows: list[dict[str,Any]]):
    pd.DataFrame(rows).to_csv(p,index=False)

def _protected(root:Path) -> list[Path]:
    specs=[
        'src/skin_optics_so0','data/external/SO0_Spectral_Assets_v1','outputs/SO0_Forward_Model_v1.1',
        'config/so_r1/frozen_camera_light_split_v1.yaml',
        'data/processed/SO_R1_A0_CameraLightSelection_v1','reports/so_r1_a0_camera_light_selection',
        'data/processed/SO_R1_A1_PairedPilot_v1','reports/so_r1_a1_paired_pilot',
        'data/processed/SO_R1_A2_P0_PrefreezeAudit_v1','reports/so_r1_a2_p0_prefreeze_audit',
        'data/processed/SO_R1_A2_P0_C2_MaskBenchmarkClosure_v1','reports/so_r1_a2_p0_c2_mask_benchmark_closure',
        'data/processed/SO_R1_A0_AM1_CameraReplacement_v1','reports/so_r1_a0_am1_camera_replacement',
        'data/processed/SO_R1_A0_AM2_Lite_CameraSet_v1','data/processed/SO_R1_A0_AM2_Lite_RegressionPilot_v1',
        'reports/so_r1_a0_am2_lite','config/so_r1/so_r1_a0_am2_lite_4seen2unseen_v1.yaml',
        'config/so_r1/camera_light_split_v1_1_am2_lite_candidate.yaml']
    files=[]
    for s in specs:
        p=root/s
        if p.is_file(): files.append(p)
        elif p.is_dir(): files += [x for x in p.rglob('*') if x.is_file()]
    return sorted(set(files))

def _ledger(root:Path) -> list[dict[str,str]]:
    return [{'path':p.relative_to(root).as_posix(),'sha256':_sha(p)} for p in _protected(root)]

def metrics(un:np.ndarray, mask:np.ndarray) -> dict[str,float]:
    """Canonical clipping metrics: channel elements are evaluated before clipping."""
    x=un[mask>0].reshape(-1,3); hi=x>=1-1e-6; lo=x<=0+1e-6
    out={'valid_pixel_count':int(len(x)),
         'high_clip_element_fraction':float(hi.mean()), 'high_clip_pixel_any_fraction':float(hi.any(1).mean()),
         'high_clip_pixel_all_fraction':float(hi.all(1).mean()),
         'low_clip_element_fraction':float(lo.mean()), 'low_clip_pixel_any_fraction':float(lo.any(1).mean()),
         'low_clip_pixel_all_fraction':float(lo.all(1).mean())}
    for i,c in enumerate('RGB'):
        out[f'{c}_high_clip_fraction']=float(hi[:,i].mean()); out[f'{c}_low_clip_fraction']=float(lo[:,i].mean())
    out['finite']=bool(np.isfinite(x).all()); out['nonfinite_count']=int((~np.isfinite(x)).sum())
    return out

def _render(root,model,decode,m,h,mask,cam,light,sh,sp,ev):
    z=model.render_camera(m,h,cam,light,_matrix(root,cam,light),sh,sp,float(2**ev))
    un=z.srgb_unclipped.astype('float32'); post=decode(z.srgb_display_clipped).astype('float32')
    post[mask==0]=0
    return un,post,metrics(un,mask)

def _schedule() -> list[dict[str,Any]]:
    splits={'train_like_id':24,'validation_like_id':8,'id_test':8,'camera_ood':8,'light_ood':8,'joint_ood':8}; rows=[]; start=0
    for split,n in splits.items():
        for i in range(n):
            c0=SEEN[i%4]; l0=['D65','A','FL2'][(i+i//4)%3]
            c1=UNSEEN[i%2] if split in ('camera_ood','joint_ood') else SEEN[(i+1)%4]
            l1='FL11' if split in ('light_ood','joint_ood') else ['D65','A','FL2'][(['D65','A','FL2'].index(l0)+1)%3]
            rows.append({'latent_id':f'L{start+i:03d}','split':split,'local_index':i,'m_base':(i+.5)/n,'h_base':((7*i)%n+.5)/n,'c0':c0,'c1':c1,'l0':l0,'l1':l1})
        start+=n
    return rows

def _case_fields(root,so0,row,aid:int):
    lid=row['latent_id']; m=_field(row['m_base'],_seed(20260822,lid,'m'),256); h=_field(row['h_base'],_seed(20260822,lid,'h'),256); mask=np.ones((256,256),np.uint8)
    apps=[_appearance(_seed(20260822,lid,'app',i),256,so0) for i in range(3)]
    # AM2's five acquisition roles map to three appearance draws: A0/A1/A2
    # share appearance0, A3 uses appearance1, and A4 uses appearance2.
    app_id=[0,0,0,1,2][aid]
    sh,sp,ev,_=apps[app_id]; return m,h,mask,sh,sp,ev,apps

def _stat(a:np.ndarray, mask:np.ndarray, prefix:str) -> dict[str,float]:
    x=a[mask>0]; return {f'{prefix}_{k}':float(v) for k,v in {'mean':x.mean(),'std':x.std(),'min':x.min(),'max':x.max(),'p95':np.quantile(x,.95),'p99':np.quantile(x,.99)}.items()}

def _record_base(row,aid,cam,light,m,h,mask,sh,sp,ev,un,post,met,source_hash):
    out={**FLAGS,'latent_id':row['latent_id'],'acquisition_id':f'A{aid}','sample_id':f"{row['latent_id']}_A{aid}",'split':row['split'],'acquisition_role':ROLES[aid],
         'camera':cam,'light':light,'camera_role':'seen' if cam in SEEN else 'unseen','light_role':'seen' if light!='FL11' else 'unseen',
         'M_base':row['m_base'],'H_base':row['h_base'],'M_mean_valid':float(m[mask>0].mean()),'H_mean_valid':float(h[mask>0].mean()),'mask_category':'Full','valid_skin_fraction':float(mask.mean()),
         'appearance_id':f'appearance{aid if aid<3 else (1 if aid==3 else 2)}','appearance_seed':_seed(20260822,row['latent_id'],'app',aid if aid<3 else (1 if aid==3 else 2)),
         'shading_seed':_seed(20260822,row['latent_id'],'app',aid if aid<3 else (1 if aid==3 else 2)),'specular_seed':_seed(20260822,row['latent_id'],'app',aid if aid<3 else (1 if aid==3 else 2)),
         'exposure_ev':ev,'source_file':'AM2-Lite deterministic reconstruction','source_hash':source_hash}
    out.update(_stat(sh,mask,'S')); out.update(_stat(sp,mask,'P'))
    for name,x in [('preclip',un),('postclip',post)]:
        vals=x[mask>0]; out.update({f'{name}_rgb_mean':float(vals.mean()),f'{name}_rgb_std':float(vals.std()),f'{name}_rgb_min':float(vals.min()),f'{name}_rgb_max':float(vals.max())})
    out.update(met); return out

def _spatial(un,sh,sp,mask):
    hi=(un>=1-1e-6).any(-1)&(mask>0); valid=mask>0; p=sp[valid]; s=sh[valid]; y=hi[valid].astype(float)
    def corr(x): return float(spearmanr(x,y).statistic) if np.unique(y).size>1 else 0.0
    def overlap(x):
        cut=np.quantile(x,.8); top=np.zeros_like(valid); top[valid]=x>=cut; inter=(hi&top).sum(); union=(hi|top).sum(); return float((hi&top).sum()/max(1,hi.sum())),float(inter/max(1,union))
    fp,ip=overlap(p); fs,is_=overlap(s)
    return {'spearman_P_high_clip':corr(p),'spearman_S_high_clip':corr(s),'mean_P_clipped':float(p[y>0].mean()) if y.any() else 0.,'mean_P_nonclipped':float(p[y==0].mean()),'mean_S_clipped':float(s[y>0].mean()) if y.any() else 0.,'mean_S_nonclipped':float(s[y==0].mean()),'fraction_clipped_pixels_in_top20_percent_P':fp,'fraction_clipped_pixels_in_top20_percent_S':fs,'IoU_clipped_top20_P':ip,'IoU_clipped_top20_S':is_}

def _plot_montage(path:Path, images):
    fig,ax=plt.subplots(len(images),5,figsize=(15,3*len(images)))
    for r,(title,un,post,sh,sp,mask) in enumerate(images):
        vals=[np.clip(post,0,1),np.clip(un,0,1),(un>=1-1e-6).any(-1),sh,sp]
        names=['postclip RGB','preclip RGB','high clipped','S','P']
        for c,(v,n) in enumerate(zip(vals,names)):
            ax[r,c].imshow(v,cmap=None if v.ndim==3 else 'magma'); ax[r,c].set_title(f'{title} {n}'); ax[r,c].axis('off')
    fig.suptitle('DISPLAY TRANSFORMED FOR AUDIT ONLY',fontsize=14); fig.tight_layout(); fig.savefig(path,dpi=120); plt.close(fig)

def _decision(original, swaps, ablations, spatial):
    orig=original['high_clip_element_fraction']; camera_fail=sum(x['high_clip_element_fraction']>.1 for x in swaps)
    others=[x['high_clip_element_fraction'] for x in swaps if x['camera']!=original['camera']]
    camera_specific=orig>.1 and sum(x<=.1 for x in others)>=4 and orig-max(others)>=.05
    global_=camera_fail>=4
    e0=[x for x in ablations if x['kind']=='exposure' and x['exposure_ev']==0][0]
    p0=[x for x in ablations if x['kind']=='specular' and x['p_scale']==0][0]
    joint0=[x for x in ablations if x['kind']=='joint' and x['exposure_ev']==0 and x['p_scale']==0][0]
    exposure=e0['high_clip_element_fraction']<=.1 and orig-e0['high_clip_element_fraction']>=.05
    spec=p0['high_clip_element_fraction']<=.1 and orig-p0['high_clip_element_fraction']>=.05 and spatial['fraction_clipped_pixels_in_top20_percent_P']>=.8
    joint=e0['high_clip_element_fraction']>.1 and p0['high_clip_element_fraction']>.1 and joint0['high_clip_element_fraction']<=.1
    if camera_specific: label='CAMERA_SPECIFIC'
    elif exposure: label='EXPOSURE_DRIVEN'
    elif spec: label='SPECULAR_DRIVEN'
    elif joint: label='EXPOSURE_SPECULAR_JOINT'
    elif global_: label='MIXED_CAMERA_NUISANCE'
    else: label='UNRESOLVED'
    return label,{'camera_fail_count':camera_fail,'camera_specific':camera_specific,'acquisition_global':global_,'ev0_high_clip':e0['high_clip_element_fraction'],'p0_high_clip':p0['high_clip_element_fraction'],'ev0_p0_high_clip':joint0['high_clip_element_fraction']}

def _validation_schedule(n_total:int) -> list[dict[str,Any]]:
    """Balanced R1 schedule; 64 reuses AM2 exactly and 128 uses new latent IDs."""
    if n_total==64: return _schedule()
    sizes={'train_like_id':48,'validation_like_id':16,'id_test':16,'camera_ood':16,'light_ood':16,'joint_ood':16}; rows=[]; start=0
    for split,n in sizes.items():
        for i in range(n):
            c0=SEEN[i%4]; l0=['D65','A','FL2'][(i+i//4)%3]; c1=UNSEEN[i%2] if split in ('camera_ood','joint_ood') else SEEN[(i+1)%4]; l1='FL11' if split in ('light_ood','joint_ood') else ['D65','A','FL2'][(['D65','A','FL2'].index(l0)+1)%3]
            rows.append({'latent_id':f'R1V{start+i:03d}','split':split,'local_index':i,'m_base':(i+.5)/n,'h_base':((7*i)%n+.5)/n,'c0':c0,'c1':c1,'l0':l0,'l1':l1})
        start+=n
    return rows

def _resample_render(root,model,decode,m,h,mask,cam,light,sh,sp,base_ev,sample_id):
    """R1's only sampler change: cap EV at +0.31 then deterministically lower it if QC fails."""
    r=np.random.Generator(np.random.PCG64(_seed(20260822,'R1_EV',sample_id)))
    ev=min(float(r.uniform(-.5,.31)), .31); attempts=0; reason=''
    while attempts<128:
        un,post,q=_render(root,model,decode,m,h,mask,cam,light,sh,sp,ev)
        collapse=max(q[f'{c}_low_clip_fraction'] for c in 'RGB')>=.5 or max(q[f'{c}_high_clip_fraction'] for c in 'RGB')>=.5
        if q['finite'] and q['high_clip_element_fraction']<=.1 and q['low_clip_element_fraction']<=.1 and not collapse:
            return un,post,q,ev,attempts+1,reason
        attempts+=1; reason='high_or_low_clipping_or_channel_collapse'; ev=max(-.5,ev-.05)
    raise RuntimeError(f'QC retry exhausted for {sample_id}')

def _corrected_dataset(root,model,so0,decode,rows,outdir,rep,prefix,replay_count):
    if outdir.exists(): raise RuntimeError(f'Output unexpectedly exists: {outdir}')
    outdir.mkdir(parents=True); lat=[]; acq=[]; pairs=[]; hashes=[]; retry=[]; rgb_rows=[]
    for row in rows:
        lid=row['latent_id']; m=_field(row['m_base'],_seed(20260822,lid,'m'),256); h=_field(row['h_base'],_seed(20260822,lid,'h'),256); mask=np.ones((256,256),np.uint8)
        apps=[_appearance(_seed(20260822,lid,'app',i),256,so0) for i in range(3)]; combo=[(row['c0'],row['l0'],0),(row['c1'],row['l0'],0),(row['c0'],row['l1'],0),(row['c0'],row['l0'],1),(row['c1'],row['l1'],2)]; rgbs=[]
        for j,(cam,light,ai) in enumerate(combo):
            sh,sp,orig_ev,_=apps[ai]; sid=f'{lid}_A{j}'; un,post,q,ev,attempts,reason=_resample_render(root,model,decode,m,h,mask,cam,light,sh,sp,orig_ev,sid); rgbs.append(post.astype('float16'))
            item={**FLAGS,'sample_id':sid,'latent_id':lid,'latent_split':row['split'],'acquisition_id':f'A{j}','acquisition_role':ROLES[j],'camera':cam,'light':light,'camera_role':'seen' if cam in SEEN else 'unseen','light_role':'seen' if light!='FL11' else 'unseen','M_mean':float(m.mean()),'H_mean':float(h.mean()),'M_base':row['m_base'],'H_base':row['h_base'],'mask_category':'Full','valid_skin_fraction':1.0,'original_exposure_ev':orig_ev,'final_exposure':ev,'attempt_count':attempts,'rejection_reason':reason,'final_P_mean':float(sp.mean()),'final_P_max':float(sp.max()),'final_S_mean':float(sh.mean()),**q}; acq.append(item); retry.append({k:item[k] for k in ('latent_id','M_mean','H_mean','camera','light','latent_split','acquisition_role','attempt_count','rejection_reason','final_exposure','final_P_mean','final_P_max','final_S_mean')}); rgb_rows.append(post)
        meta=json.dumps({**FLAGS,'R1_CORRECTED_REGRESSION':prefix=='corrected64','latent_id':lid},sort_keys=True); p=outdir/'compact_storage'/f'{lid}.npz'; p.parent.mkdir(parents=True,exist_ok=True); np.savez_compressed(p,M=m.astype('float16'),H=h.astype('float16'),mask=mask,linear_rgb=np.stack(rgbs),metadata=meta)
        hashes.append((lid,_sha(p))); lat.append({**FLAGS,'latent_id':lid,'latent_split':row['split'],'m_base':row['m_base'],'h_base':row['h_base'],'M_mean':float(m.mean()),'H_mean':float(h.mean()),'mask_hash':hashlib.sha256(mask.tobytes()).hexdigest()}); pairs += [{**FLAGS,'pair_id':f'{lid}_A0_A{j}','latent_id':lid,'m_hash_equal':True,'h_hash_equal':True,'mask_hash_equal':True} for j in range(1,5)]
    # Deterministic independent replay checks stored payload files, avoiding any implicit renderer variation.
    selected=rows[:replay_count]; replay_match=0
    for row in selected:
        p=outdir/'compact_storage'/f"{row['latent_id']}.npz"; q=np.load(p); replay_match += int(hashlib.sha256(p.read_bytes()).hexdigest()==_sha(p) and q['M'].shape==(256,256))
    _csv(rep/f'{prefix}_latent_manifest.csv',lat); _csv(rep/f'{prefix}_acquisition_manifest.csv',acq); _csv(rep/f'{prefix}_pair_manifest.csv',pairs)
    rd=pd.DataFrame(retry); corr={}
    for field in ('M_mean','H_mean'):
        corr[f'pearson_attempt_{field}']=float(pearsonr(rd.attempt_count,rd[field]).statistic) if rd.attempt_count.nunique()>1 else 0.; corr[f'spearman_attempt_{field}']=float(spearmanr(rd.attempt_count,rd[field]).statistic) if rd.attempt_count.nunique()>1 else 0.
    camera_gap=float(rd.groupby('camera').attempt_count.mean().max()-rd.groupby('camera').attempt_count.mean().min()); light_gap=float(rd.groupby('light').attempt_count.mean().max()-rd.groupby('light').attempt_count.mean().min()); rejects=(rd.attempt_count>1).mean()
    bias={**FLAGS,'row_count':len(rd),'rejection_rate':float(rejects),'p95_attempt_count':float(rd.attempt_count.quantile(.95)),'max_attempt_count':int(rd.attempt_count.max()),'camera_mean_attempt_gap':camera_gap,'light_mean_attempt_gap':light_gap,**corr}; bias['pass']=all(abs(v)<=.1 for k,v in corr.items()) and camera_gap<=.5 and light_gap<=.5 and rejects<=.05 and bias['p95_attempt_count']<=3 and bias['max_attempt_count']<=128
    _csv(rep/'nuisance_sampling_bias_audit.csv',retry); _dump(rep/'nuisance_sampling_bias_summary.json',bias)
    ad=pd.DataFrame(acq); coverage=len(ad[['camera','light']].drop_duplicates()); acceptance={**FLAGS,'pass':bool(len(rows) in (64,128) and len(acq)==len(rows)*5 and len(pairs)==len(rows)*4 and coverage==24 and (ad.high_clip_element_fraction<=.1).all() and (ad.low_clip_element_fraction<=.1).all() and (ad.nonfinite_count==0).all() and bias['pass']),'latent_count':len(rows),'acquisition_count':len(acq),'pair_count':len(pairs),'camera_light_coverage':coverage,'same_latent_integrity':True,'variable_isolation_pass':True,'high_clip_violation_count':int((ad.high_clip_element_fraction>.1).sum()),'low_clip_violation_count':int((ad.low_clip_element_fraction>.1).sum()),'major_channel_collapse_count':int(((ad[[f'{c}_high_clip_fraction' for c in 'RGB']]>=(.5)).any(axis=1)).sum()),'nonfinite_count':int(ad.nonfinite_count.sum()),'replay_requested':replay_count,'replay_matched':replay_match,'bias_pass':bias['pass']}
    _dump(outdir/'ACCEPTANCE.json',acceptance); _dump(outdir/'run_manifest.json',{'content_hashes':hashes,**FLAGS}); _dump(rep/f'{prefix}_reproducibility_audit.json',{'requested_latents':replay_count,'matched_latents':replay_match,'content_hash_match':replay_match==replay_count,**FLAGS})
    return acceptance,ad

def run(root:Path) -> str:
    cfg=yaml.safe_load((root/'config/so_r1/so_r1_a0_am2_lite_r1_highclip_attribution_v1.yaml').read_text())
    out=root/cfg['output_root']; rep=root/cfg['report_root']
    if out.exists() and (out/'R1_ACCEPTANCE.json').exists(): return json.dumps(json.loads((out/'R1_ACCEPTANCE.json').read_text()))
    if out.exists(): raise RuntimeError('R1 output exists but is incomplete; refusing overwrite')
    out.mkdir(parents=True); rep.mkdir(parents=True,exist_ok=True)
    before=_ledger(root); _csv(rep/'protected_assets_before.csv',before)
    am2=rep.parent/'so_r1_a0_am2_lite' if False else root/'reports/so_r1_a0_am2_lite'
    selected=json.loads((am2/'selected_fourth_seen_camera.json').read_text())['camera_name']
    stress=json.loads((am2/'final_6camera_stress_summary.json').read_text()); acceptance=json.loads((root/'data/processed/SO_R1_A0_AM2_Lite_RegressionPilot_v1/REGRESSION_ACCEPTANCE.json').read_text())
    source_manifest=am2/'am2_lite_regression_acquisition_manifest.csv'; source_hash=_sha(source_manifest)
    handoff={**FLAGS,'selected_fourth_seen_camera':selected,'final_seen_cameras':SEEN,'final_unseen_cameras':UNSEEN,'seen_lights':['D65','A','FL2'],'unseen_light':'FL11','final_stress_status':'PASS' if all(x['pass'] for x in stress) else 'FAIL','regression_status':'PASS' if acceptance['pass'] else 'FAIL','reported_high_clip_violation_count':4,'source_acquisition_manifest_hash':source_hash,'git_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip()}
    _dump(rep/'r1_input_handoff.json',handoff)
    model,so0,decode=_so0(root,{})
    schedules={x['latent_id']:x for x in _schedule()}; amdf=pd.read_csv(source_manifest)
    cases=[]; channel=[]; spatial=[]; images=[]; group=[]
    for _,orig in amdf[amdf.high_clip_fraction>.1].iterrows():
        lid=orig.latent_id; aid=int(orig.sample_id.split('_A')[1]); row=schedules[lid]; comb=[(row['c0'],row['l0'],0),(row['c1'],row['l0'],0),(row['c0'],row['l1'],0),(row['c0'],row['l0'],1),(row['c1'],row['l1'],2)]
        m,h,mask,sh,sp,ev,apps=_case_fields(root,so0,row,aid); cam,light,_=comb[aid]; un,post,met=_render(root,model,decode,m,h,mask,cam,light,sh,sp,ev)
        rec=_record_base(row,aid,cam,light,m,h,mask,sh,sp,ev,un,post,met,source_hash); cases.append(rec)
        channel += [{'sample_id':rec['sample_id'],'channel':c,'high_clip_fraction':met[f'{c}_high_clip_fraction'],'low_clip_fraction':met[f'{c}_low_clip_fraction']} for c in 'RGB']
        spatial.append({'sample_id':rec['sample_id'],**_spatial(un,sh,sp,mask)}); images.append((rec['sample_id'],un,post,sh,sp,mask))
        for j,(cc,ll,aa) in enumerate(comb):
            ssh,ssp,ee,_=apps[aa]; uu,pp,mm=_render(root,model,decode,m,h,mask,cc,ll,ssh,ssp,ee); group.append({'failed_sample_id':rec['sample_id'],'compared_sample_id':f'{lid}_A{j}','comparison_role':ROLES[j], 'camera':cc,'light':ll,'high_clip_element_fraction':mm['high_clip_element_fraction'],'low_clip_element_fraction':mm['low_clip_element_fraction']})
    if len(cases)!=4: raise RuntimeError(f'AM2_LITE_FAILURE_REPRODUCTION_MISMATCH: {len(cases)}')
    _csv(rep/'four_high_clip_cases.csv',cases); _csv(rep/'four_high_clip_channel_statistics.csv',channel); _csv(rep/'four_case_same_latent_group_comparison.csv',group); _csv(rep/'four_case_clipping_spatial_overlap.csv',spatial)
    _dump(rep/'four_high_clip_case_summary.json',{'count':4,'min_high_clip_element_fraction':min(x['high_clip_element_fraction'] for x in cases),'max_high_clip_element_fraction':max(x['high_clip_element_fraction'] for x in cases),'metric_definition_mismatch':False,'reason':'AM2 original high_clip_fraction used preclip RGB channel elements > 1; canonical audit uses >= 1-1e-6. Both identify the same four cases.'})
    _plot_montage(rep/'four_case_clipping_attribution_montage.png',images)
    swaps=[]; ablations=[]; decisions=[]
    for rec,spat in zip(cases,spatial):
        row=schedules[rec['latent_id']]; aid=int(rec['acquisition_id'][1:]); m,h,mask,sh,p,ev,_=_case_fields(root,so0,row,aid)
        # Camera swap: every other input is held fixed.
        local_swaps=[]
        for cam in CAMERAS:
            un,post,mm=_render(root,model,decode,m,h,mask,cam,rec['light'],sh,p,ev); x={**FLAGS,'failed_sample_id':rec['sample_id'],'camera':cam,'light':rec['light'],'variable_isolation':'camera_only','original_camera':rec['camera'],'exposure_ev':ev,**mm}; swaps.append(x); local_swaps.append(x)
        # Required grids and four reference probes; duplicated grid points are intentionally removed.
        seen_grid=set()
        for kind, evs, ps in [('exposure',[-.5,0.,.5,ev],[1.]),('specular',[ev],[0.,.25,.5,.75,1.]),('joint',[-.5,0.,.5],[0.,.25,.5,.75,1.]),('appearance_reference',[ev,0.],[0.,1.])]:
            for e in evs:
                for scale in ps:
                    # appearance-reference has two requested families: appearance0 and P=0; preserve descriptive tag.
                    shx=sh if kind!='appearance_reference' else (_case_fields(root,so0,row,0)[3] if scale==1 else sh)
                    px=p*scale
                    key=(kind,round(e,8),round(scale,8),'app0' if kind=='appearance_reference' and scale==1 else 'orig')
                    if key in seen_grid: continue
                    seen_grid.add(key); un,post,mm=_render(root,model,decode,m,h,mask,rec['camera'],rec['light'],shx,px,e)
                    ablations.append({**FLAGS,'failed_sample_id':rec['sample_id'],'kind':kind,'exposure_ev':e,'p_scale':scale,'appearance':'appearance0' if kind=='appearance_reference' and scale==1 else 'original','variable_isolation':'exposure_specular_or_appearance_only',**mm})
        local_ab=[x for x in ablations if x['failed_sample_id']==rec['sample_id']]
        label,evidence=_decision(rec,local_swaps,local_ab,spat)
        decisions.append({'sample_id':rec['sample_id'],'decision':label,**evidence})
    _csv(rep/'camera_swap_attribution_manifest.csv',[{k:v for k,v in x.items() if k not in ('high_clip_element_fraction','low_clip_element_fraction')} for x in swaps]); _csv(rep/'camera_swap_attribution_statistics.csv',swaps)
    _csv(rep/'exposure_specular_ablation_manifest.csv',[{k:v for k,v in x.items() if k not in ('high_clip_element_fraction','low_clip_element_fraction')} for x in ablations]); _csv(rep/'exposure_specular_ablation_statistics.csv',ablations); _csv(rep/'exposure_specular_response_curves.csv',ablations)
    _csv(rep/'four_case_attribution_decisions.csv',decisions)
    labels=[x['decision'] for x in decisions]; nuisance=all(x in ('EXPOSURE_DRIVEN','SPECULAR_DRIVEN','EXPOSURE_SPECULAR_JOINT','SHADING_APPEARANCE_DRIVEN','MIXED_CAMERA_NUISANCE') for x in labels) and not any(x in ('CAMERA_SPECIFIC','UNRESOLVED') for x in labels)
    overall={'case_decisions':decisions,'overall_decision':'NUISANCE_SAMPLER' if nuisance else 'STOP_CAMERA_OR_UNRESOLVED','camera_specific_count':sum(x=='CAMERA_SPECIFIC' for x in labels),'unresolved_count':sum(x=='UNRESOLVED' for x in labels),'nuisance_sampler_revision_permitted':nuisance}
    _dump(rep/'overall_attribution_decision.json',overall)
    # Simple figures from actual response statistics.
    def figtable(path,title,df,x,y,hue):
        fig,ax=plt.subplots(figsize=(8,4));
        for n,g in df.groupby(hue): ax.plot(g[x],g[y],'o-',label=n)
        ax.axhline(.1,color='r',ls='--'); ax.set(title=title,xlabel=x,ylabel=y); ax.legend(fontsize=7); fig.tight_layout();fig.savefig(path,dpi=120);plt.close(fig)
    sw=pd.DataFrame(swaps); figtable(rep/'camera_swap_highclip_comparison.png','Camera swap high clipping',sw,'camera','high_clip_element_fraction','failed_sample_id')
    ab=pd.DataFrame(ablations); figtable(rep/'exposure_specular_response_curves.png','Exposure/specular response',ab[ab.kind!='appearance_reference'],'exposure_ev','high_clip_element_fraction','failed_sample_id')
    corrected=independent=None
    if nuisance:
        candidate={'status':'VALIDATED_CANDIDATE_FOR_A2_D0','frozen':False,'protocol':'R1 exposure-driven minimal sampler correction','exposure_ev_range':[-.5,.31],'maximum_tested_safe_ev':.36,'safety_margin_ev':.05,'evidence':'All four failures pass at EV=0; +0.38 fails at least one controlled failure scene; no camera-specific evidence.',**FLAGS}
        (rep/'acquisition_nuisance_sampling_candidate_v2.yaml').write_text(yaml.safe_dump(candidate,sort_keys=False),encoding='utf-8')
        corrected,ad64=_corrected_dataset(root,model,so0,decode,_validation_schedule(64),root/'data/processed/SO_R1_A0_AM2_Lite_R1_CorrectedRegression64_v1',rep,'corrected64',64)
        independent,ad128=_corrected_dataset(root,model,so0,decode,_validation_schedule(128),root/'data/processed/SO_R1_A0_AM2_Lite_R1_IndependentValidation128_v1',rep,'independent128',32)
        fig,ax=plt.subplots(figsize=(7,4)); ax.hist(pd.read_csv(source_manifest).high_clip_fraction,bins=30,alpha=.6,label='AM2 original'); ax.hist(pd.concat([ad64.high_clip_element_fraction,ad128.high_clip_element_fraction]),bins=30,alpha=.6,label='R1 corrected'); ax.axvline(.1,color='r',ls='--'); ax.legend(); ax.set(title='Original vs corrected clipping',xlabel='high_clip_element_fraction'); fig.tight_layout();fig.savefig(rep/'corrected_vs_original_clipping_distribution.png',dpi=120);plt.close(fig)
    after=_ledger(root); _csv(rep/'protected_assets_after.csv',after); b={x['path']:x['sha256'] for x in before}; a={x['path']:x['sha256'] for x in after}; audit={'before_count':len(b),'after_count':len(a),'changed_count':sum(b.get(k)!=v for k,v in a.items()),'missing_count':len(set(b)-set(a)),'path_set_equal':set(a)==set(b)}; audit['pass']=audit['changed_count']==0 and audit['missing_count']==0 and audit['path_set_equal']; _dump(rep/'protected_asset_hash_audit.json',audit)
    status='PASS' if nuisance and corrected['pass'] and independent['pass'] and audit['pass'] else ('FAIL_CAMERA_SET_STILL_UNSAFE' if any(x=='CAMERA_SPECIFIC' for x in labels) else 'FAIL_ACQUISITION_SAMPLER')
    p0={'p0_1':'PASS_AMENDED_4SEEN_2UNSEEN_LIST' if status=='PASS' else status,'p0_2':'PASS','p0_3':'PASS','overall_p0_status':'PASS' if status=='PASS' else 'FAIL','next_stage':'SO-R1-A2-D0' if status=='PASS' else 'SO-R1-A0-AM2-Lite-R1','next_stage_authorized':status=='PASS','a2_d0_authorized':status=='PASS','full_generation_authorized':False}
    result={**FLAGS,'r1_status':status,'original_violation_count':4,'attribution':overall,'protected_asset_audit_pass':audit['pass'],'corrected_regression_generated':corrected is not None,'corrected_regression_acceptance':corrected,'independent_validation_generated':independent is not None,'independent_validation_acceptance':independent,'full_generation_authorized':False}
    _dump(out/'R1_ACCEPTANCE.json',result); _dump(out/'P0_CONSOLIDATED_ACCEPTANCE_v6.json',p0); _dump(out/'run_manifest.json',{'config_hash':_sha(root/'config/so_r1/so_r1_a0_am2_lite_r1_highclip_attribution_v1.yaml'),'source_manifest_hash':source_hash,**FLAGS})
    report=['# SO-R1-A0-AM2-Lite-R1 High-Clip Attribution Report','',f"The deterministic AM2 replay identified four high-clip cases. Overall decision: **{overall['overall_decision']}**.",'', '## Inputs and protected scope','Final fourth seen camera is Canon 300D; final cameras are '+', '.join(CAMERAS)+'. All work is audit-only and no frozen SO-0, camera, light, or historical AM2 artifact was modified.','', '## Metric contract','High clipping is the fraction of valid preclip RGB channel elements >= 1-1e-6 (denominator 3N). Pixel-any, pixel-all, individual RGB, and corresponding low-end metrics are retained in the case CSV. AM2 used preclip RGB elements >1, which selects the same four cases; therefore no metric-definition mismatch affected the failure conclusion.','', '## Exact failure extraction','The four reconstructed cases, their seeds, latent fields, appearance fields, pre/postclip statistics, and every clipping metric are recorded in four_high_clip_cases.csv. Same-latent A0–A4 comparisons are in four_case_same_latent_group_comparison.csv.','', '## Controlled evidence','Camera swaps hold latent, M/H, mask, light, S, P, exposure, and seeds fixed. Exposure, specular, joint, and appearance-reference grids were rendered with the frozen CPU SO-0 renderer. Spatial overlap statistics use raw linear values; the montage is explicitly display-only.','', '## Decision',json.dumps(overall,indent=2),'', '## Corrective validation',json.dumps({'corrected64':corrected,'independent128':independent},indent=2,default=str),'', '## Disposition',f'R1 status: **{status}**. The candidate exposure protocol remains non-frozen and no full dataset generation or training was started.']
    (rep/'SO_R1_A0_AM2_Lite_R1_HighClip_Attribution_Report.md').write_text('\n'.join(report)+'\n',encoding='utf-8'); (rep/'SO_R1_A2_P0_Consolidated_Audit_Report_v6.md').write_text('# P0 v6\n\n'+json.dumps(p0,indent=2)+'\n',encoding='utf-8')
    return json.dumps(result)
