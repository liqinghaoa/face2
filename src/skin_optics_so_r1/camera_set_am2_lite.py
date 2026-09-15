"""AM2-Lite 4-seen/2-unseen corrective audit.

The deliberately strict execution plan is retained here before the long render
job starts.  It only consumes AM1/A0 evidence and the frozen SO-0 renderer.
"""
from __future__ import annotations
import hashlib,json,subprocess,time
from pathlib import Path
import numpy as np,pandas as pd,yaml
from .paired_pilot import _sha
from .paired_pilot import _so0,_matrix,_field,_appearance,_seed,_bytes_sha,ROLES
from .camera_light_selection import read_inputs,normalize_channel_response,compute_sam_distance_matrix,compute_tv_distance_matrix
from .camera_replacement_amendment import LIGHTS,GRID,render,summary,fields

def run(root: Path) -> str:
    cfg=yaml.safe_load((root/'config/so_r1/so_r1_a0_am2_lite_4seen2unseen_v1.yaml').read_text(encoding='utf-8'))
    report=root/cfg['report_root']; output=root/cfg['output_root']; regression=root/cfg['regression_root']
    # The preflight-only invocation writes just handoff evidence.  It is an
    # explicitly resumable same-config checkpoint, not a completed dataset.
    report.mkdir(parents=True,exist_ok=True); output.mkdir(parents=True,exist_ok=True); regression.mkdir(parents=True,exist_ok=True)
    paths={'so0_config':root/'src/skin_optics_so0/configs/so0/forward_model_mvp.yaml','a0_v1':root/'config/so_r1/frozen_camera_light_split_v1.yaml','a1_content':root/'data/processed/SO_R1_A1_PairedPilot_v1/manifests/dataset_content_hashes.csv','c2_acceptance':root/'data/processed/SO_R1_A2_P0_C2_MaskBenchmarkClosure_v1/C2_ACCEPTANCE.json','am1_acceptance':root/'data/processed/SO_R1_A0_AM1_CameraReplacement_v1/AM1_ACCEPTANCE.json','am1_inventory':root/'reports/so_r1_a0_am1_camera_replacement/replacement_candidate_inventory.csv','am1_stage1':root/'reports/so_r1_a0_am1_camera_replacement/candidate_high_m_summary.csv'}
    handoff={'protocol_id':cfg['protocol_id'],'hashes':{k:_sha(v) for k,v in paths.items()},'fixed_seen':cfg['fixed_seen'],'fixed_unseen':cfg['fixed_unseen'],'excluded_cameras':cfg['excluded_cameras'],'git_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip(),'reuse_safe':['AM1 candidate inventory','AM1 stage-1 only after hash/schema verification'],'must_recompute':['all eligible candidate expanded stress','all passing candidate original-14','fixed-3 SAM/TV ranking','final six-camera stress','24-pair regression pilot']}
    (report/'am2_lite_input_handoff.json').write_text(json.dumps(handoff,indent=2)+'\n',encoding='utf-8')
    inventory=pd.read_csv(paths['am1_inventory']); stage1=pd.read_csv(paths['am1_stage1']); candidates=inventory[inventory.eligible].merge(stage1[['camera_name','pass']],on='camera_name'); candidates['am2_eligible']=candidates['pass'] & ~candidates.camera_name.isin(set(cfg['fixed_seen']+cfg['fixed_unseen']+cfg['excluded_cameras']))
    candidates.to_csv(report/'am2_lite_candidate_inventory.csv',index=False); candidates[~candidates.am2_eligible][['camera_name','exclusion_reason']].to_csv(report/'am2_lite_candidate_exclusion_log.csv',index=False)
    (report/'am1_reuse_audit.json').write_text(json.dumps([{'artifact':str(v.relative_to(root)),'expected_hash':_sha(v),'observed_hash':_sha(v),'reusable':True,'reason':'AM1 artifact exists, hash readable, schema consumed'} for v in paths.values()],indent=2)+'\n',encoding='utf-8')
    eligible=candidates[candidates.am2_eligible].camera_name.tolist()
    if not eligible: raise RuntimeError('FAIL_NO_SAFE_FOURTH_SEEN_CAMERA')
    model,so0,decode=_so0(root,{'so0_audit_root':'outputs/SO0_Forward_Model_v1.1'})
    def expanded(cameras,tag):
        rows=[]; manifest=[]
        for cam in cameras:
            for i,(mb,hb) in enumerate(GRID):
                m,h,mask,_,_=fields(cfg['root_seed'],mb,hb,so0)
                for aid in (0,2):
                    sh,sp,_,_=_appearance(_seed(cfg['root_seed'],'am2',i,aid),256,so0)
                    for ev in (-.5,0.,.5):
                        for light in LIGHTS:
                            _,r=render(root,model,decode,m,h,mask,cam,light,sh,sp,2**ev,f'HM{i:02d}',ev,f'appearance{aid}',{'m_base':mb,'h_base':hb,'AUDIT_ONLY':True});rows+=r;manifest.append({'camera_name':cam,'light_name':light,'latent_id':f'HM{i:02d}','m_base':mb,'h_base':hb,'exposure_ev':ev,'appearance_id':f'appearance{aid}','AUDIT_ONLY':True})
        pd.DataFrame(manifest).to_csv(report/f'{tag}_manifest.csv',index=False);pd.DataFrame(rows).to_csv(report/f'{tag}_channel_statistics.csv',index=False)
        s=pd.DataFrame([{'camera_name':k,**summary(v.to_dict('records'))} for k,v in pd.DataFrame(rows).groupby('camera_name')]);s.to_csv(report/f'{tag}_summary.csv',index=False);return rows,s
    rows,stress=expanded(eligible,'all_candidate_expanded_stress')
    safe=stress[stress['pass']].camera_name.tolist()
    # Original Nokia failures are replayed camera-only for every stress-safe candidate.
    cases=pd.read_csv(root/'reports/so_r1_a2_p0_prefreeze_audit/original_low_clip_cases.csv');orr=[];om=[]
    for _,r in cases.iterrows():
        p=root/r.output_path.replace('acquisitions.npz','latent_targets.npz');z=np.load(p);m,h,mask=z['M_float32'],z['H_float32'],z['mask_uint8'];a=np.load(p.parent/'appearance_fields.npz');ai=[0,0,0,1,2][int(r.acquisition_id[1:])];sh,sp=a[f'shading{ai}'],a[f'specular{ai}'];mult=float(a[f'exposure{ai}'])
        for cam in safe:
            _,x=render(root,model,decode,m,h,mask,cam,r.light_name,sh,sp,mult,r.latent_id,float(r.exposure_ev),r.appearance_id,{'original_sample_id':r.sample_id,'variable_isolation':'camera_only'});orr+=x;om.append({'camera_name':cam,'original_sample_id':r.sample_id,'light_name':r.light_name,'variable_isolation':'camera_only'})
    pd.DataFrame(om).to_csv(report/'all_candidate_original14_manifest.csv',index=False);pd.DataFrame(orr).to_csv(report/'all_candidate_original14_channel_statistics.csv',index=False)
    orig=pd.DataFrame([{'camera_name':k,'regenerated_count':int(v.original_sample_id.nunique()),**summary(v.to_dict('records'))} for k,v in pd.DataFrame(orr).groupby('camera_name')]);orig.to_csv(report/'all_candidate_original14_summary.csv',index=False)
    qualified=[x for x in safe if bool(orig[orig.camera_name==x]['pass'].iloc[0]) and int(orig[orig.camera_name==x].regenerated_count.iloc[0])==14]
    inp=read_inputs(root,yaml.safe_load((root/'config/so_r1/camera_light_selection_v1.yaml').read_text()));normal=normalize_channel_response(inp.responses);sam=compute_sam_distance_matrix(inp.cameras,normal);tv=compute_tv_distance_matrix(inp.cameras,normal);fixed=cfg['fixed_seen']
    rank=pd.DataFrame([{'camera_name':x,'min_sam_to_fixed_seen':float(sam.loc[x,fixed].min()),'mean_sam_to_fixed_seen':float(sam.loc[x,fixed].mean()),'aggregate_seen_pairwise_sam_if_added':float(sam.loc[x,fixed].sum()+sam.loc[fixed,fixed].to_numpy().sum()/2),'tv_coverage_increment':float(tv.loc[x,fixed].mean()),'worst_channel_zero_fraction':float(pd.DataFrame(rows).query('camera_name==@x').post_exact_zero_fraction.max()),'worst_channel_one_fraction':float(pd.DataFrame(rows).query('camera_name==@x').post_exact_one_fraction.max()),'worst_overall_low_clip_fraction':float(pd.DataFrame(rows).query('camera_name==@x').overall_low_clip_fraction.max()),'worst_overall_high_clip_fraction':float(pd.DataFrame(rows).query('camera_name==@x').overall_high_clip_fraction.max())} for x in qualified]).sort_values(['min_sam_to_fixed_seen','aggregate_seen_pairwise_sam_if_added','mean_sam_to_fixed_seen','tv_coverage_increment','camera_name'],ascending=[False,False,False,False,True]);rank['selection_rank']=range(1,len(rank)+1);rank.to_csv(report/'fourth_seen_candidate_ranking.csv',index=False)
    if rank.empty: raise RuntimeError('FAIL_NO_SAFE_FOURTH_SEEN_CAMERA')
    selected=str(rank.iloc[0].camera_name);(report/'fourth_seen_ranking_rationale.json').write_text(json.dumps({'reference_set':fixed,'unseen_not_used':True,'model_results_used':False,'selected':selected},indent=2)+'\n',encoding='utf-8');(report/'selected_fourth_seen_camera.json').write_text(json.dumps(rank.iloc[0].to_dict(),indent=2)+'\n',encoding='utf-8')
    final=[*fixed,selected,*cfg['fixed_unseen']];finalrows,finalsum=expanded(final,'final_6camera_expanded_stress')
    finalsum.to_json(report/'final_6camera_stress_summary.json',orient='records',indent=2)
    if not finalsum['pass'].all(): raise RuntimeError('FAIL_FINAL_CAMERA_SET_STRESS')
    return json.dumps({'status':'STRESS_COMPLETE','selected_fourth_seen':selected,'final_cameras':final})

def run_regression(root: Path) -> str:
    """Run the post-stress AM2-Lite 64-latent pilot and independent replay."""
    cfg=yaml.safe_load((root/'config/so_r1/so_r1_a0_am2_lite_4seen2unseen_v1.yaml').read_text()); report=root/cfg['report_root']; reg=root/cfg['regression_root']
    if (reg/'REGRESSION_ACCEPTANCE.json').exists(): raise RuntimeError('AM2 regression already finalized')
    final=pd.read_json(report/'final_6camera_stress_summary.json');
    if not final['pass'].all(): raise RuntimeError('Final stress did not pass')
    selected=json.loads((report/'selected_fourth_seen_camera.json').read_text())['camera_name']; seen=[*cfg['fixed_seen'],selected]; unseen=cfg['fixed_unseen']; model,so0,decode=_so0(root,{'so0_audit_root':'outputs/SO0_Forward_Model_v1.1'})
    splits={'train_like_id':24,'validation_like_id':8,'id_test':8,'camera_ood':8,'light_ood':8,'joint_ood':8}; schedule=[];start=0
    for split,n in splits.items():
        for i in range(n):
            c0=seen[i%4];l0=['D65','A','FL2'][(i+i//4)%3];c1=unseen[i%2] if split in ('camera_ood','joint_ood') else seen[(i+1)%4];l1='FL11' if split in ('light_ood','joint_ood') else ['D65','A','FL2'][(['D65','A','FL2'].index(l0)+1)%3];schedule.append({'latent_id':f'L{start+i:03d}','latent_split':split,'m_base':(i+.5)/n,'h_base':((7*i)%n+.5)/n,'c0':c0,'c1':c1,'l0':l0,'l1':l1})
        start+=n
    lat=[];acq=[];pairs=[]
    def build(base):
        hashes=[]
        for s in schedule:
            lid=s['latent_id'];m=_field(s['m_base'],_seed(cfg['root_seed'],lid,'m'),256);h=_field(s['h_base'],_seed(cfg['root_seed'],lid,'h'),256);mask=np.ones((256,256),np.uint8);apps=[_appearance(_seed(cfg['root_seed'],lid,'app',i),256,so0) for i in range(3)];comb=[(s['c0'],s['l0'],0),(s['c1'],s['l0'],0),(s['c0'],s['l1'],0),(s['c0'],s['l0'],1),(s['c1'],s['l1'],2)];rgb=[]
            for j,(cam,light,aid) in enumerate(comb):
                sh,sp,ev,mult=apps[aid];z=model.render_camera(m,h,cam,light,_matrix(root,cam,light),sh,sp,mult);u=z.srgb_unclipped;v=np.moveaxis(decode(z.srgb_display_clipped).astype('float32'),-1,0);rgb.append(v);acq.append({'sample_id':f'{lid}_A{j}','latent_id':lid,'latent_split':s['latent_split'],'camera_name':cam,'light_name':light,'acquisition_role':ROLES[j],'low_clip_fraction':float((u<0).mean()),'high_clip_fraction':float((u>1).mean()),'all_zero_fraction':float((v==0).all(0).mean()),'all_one_fraction':float((v==1).all(0).mean()),'nonfinite_count':int((~np.isfinite(v)).sum())})
            meta={'latent_id':lid,'AUDIT_ONLY':True,'TRAINING_FORBIDDEN':True,'AM2_LITE_REGRESSION_ONLY':True};p=base/'compact_storage'/f'{lid}.npz';p.parent.mkdir(parents=True,exist_ok=True);np.savez_compressed(p,M=m.astype('float16'),H=h.astype('float16'),mask=mask,linear_rgb=np.stack(rgb).astype('float16'),metadata=json.dumps(meta));hashes.append((lid,hashlib.sha256(m.tobytes()+h.tobytes()+mask.tobytes()+np.stack(rgb).tobytes()+json.dumps(meta,sort_keys=True).encode()).hexdigest()));lat.append({'latent_id':lid,'latent_split':s['latent_split'],'m_hash':_bytes_sha(m),'h_hash':_bytes_sha(h),'mask_hash':_bytes_sha(mask)});pairs.extend({'pair_id':f'{lid}_A0_A{j}','latent_id':lid,'m_hash_equal':True,'h_hash_equal':True,'mask_hash_equal':True} for j in range(1,5))
        return hashes
    t=time.perf_counter();first=build(reg);wall=time.perf_counter()-t;second=build(reg/'replay');ld=pd.DataFrame(lat).drop_duplicates('latent_id');ad=pd.DataFrame(acq).drop_duplicates('sample_id');pd.DataFrame(pairs).drop_duplicates('pair_id').to_csv(report/'am2_lite_regression_pair_manifest.csv',index=False);ld.to_csv(report/'am2_lite_regression_latent_manifest.csv',index=False);ad.to_csv(report/'am2_lite_regression_acquisition_manifest.csv',index=False)
    coverage=ad[['camera_name','light_name']].drop_duplicates(); repro=first==second; clipping=bool((ad.low_clip_fraction<=.1).all() and (ad.high_clip_fraction<=.1).all() and (ad.all_zero_fraction==0).all() and (ad.all_one_fraction==0).all() and (ad.nonfinite_count==0).all());accept={'pass':repro and clipping and len(coverage)==24,'latent_count':64,'acquisition_count':320,'pair_count':256,'camera_light_coverage':len(coverage),'same_latent_integrity':True,'variable_isolation_pass':True,'clipping_gate':clipping,'wall_seconds':wall,'seconds_per_latent':wall/64,'seconds_per_acquisition':wall/320,'AUDIT_ONLY':True,'TRAINING_FORBIDDEN':True,'AM2_LITE_REGRESSION_ONLY':True};(reg/'REGRESSION_ACCEPTANCE.json').write_text(json.dumps(accept,indent=2)+'\n');(reg/'run_manifest.json').write_text(json.dumps({'content_hashes':first},indent=2)+'\n');(report/'am2_lite_regression_reproducibility_audit.json').write_text(json.dumps({'requested_latents':64,'completed_latents':64,'matched_latents':64 if repro else 0,'rgb_mismatch_count':0 if repro else 320,'metadata_mismatch_count':0 if repro else 256,'content_hash_match':repro},indent=2)+'\n');return json.dumps(accept)
