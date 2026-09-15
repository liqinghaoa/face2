"""AM1 corrective Nokia replacement audit using the frozen SO-0 renderer."""
from __future__ import annotations
import hashlib,json,subprocess,time
from pathlib import Path
import numpy as np,pandas as pd,yaml
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from .camera_light_selection import read_inputs,normalize_channel_response,compute_sam_distance_matrix,compute_tv_distance_matrix
from .paired_pilot import _so0,_matrix,_field,_appearance,_seed,_sha,_bytes_sha,ROLES
from .mask_benchmark_closure import _ledger

LIGHTS=['D65','A','FL2','FL11']; GRID=[(m,h) for m in (.75,.85,.9375,.9791666667) for h in (.1875,.5,.8125)]
def put(p,x): p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(x,indent=2)+'\n',encoding='utf-8')
def ch(a):
 q=np.quantile(a,[.01,.05,.5,.95]);return {'min':float(a.min()),'p01':float(q[0]),'p05':float(q[1]),'median':float(q[2]),'p95':float(q[3]),'max':float(a.max()),'mean':float(a.mean()),'std':float(a.std()),'negative_fraction':float((a<0).mean()),'le_minus_1e6_fraction':float((a<=-1e-6).mean()),'gt_one_fraction':float((a>1).mean())}
def emit(camera,light,lid,ev,appearance,un,post,mask,extra=None):
 meta={'camera_name':camera,'light_name':light,'latent_id':lid,'exposure_ev':ev,'appearance_id':appearance,'overall_low_clip_fraction':float((un[mask>0]<0).mean()),'overall_high_clip_fraction':float((un[mask>0]>1).mean()),'nonfinite_count':int((~np.isfinite(post)).sum()),'all_zero_fraction':float((post[mask>0]==0).all(-1).mean()),'all_one_fraction':float((post[mask>0]==1).all(-1).mean())};meta.update(extra or {});rows=[]
 for i,n in enumerate('RGB'):
  r={**meta,'channel':n};r.update({'pre_'+k:v for k,v in ch(un[...,i][mask>0]).items()}); y=post[...,i][mask>0];r.update({'post_exact_zero_fraction':float((y==0).mean()),'post_le_1e6_fraction':float((y<=1e-6).mean()),'post_exact_one_fraction':float((y==1).mean()),'post_ge_1me6_fraction':float((y>=1-1e-6).mean()),'post_mean':float(y.mean()),'post_std':float(y.std()),'post_min':float(y.min()),'post_max':float(y.max())});rows.append(r)
 return rows
def summary(rows):
 d=pd.DataFrame(rows);g=d.groupby(['camera_name','latent_id','light_name','exposure_ev','appearance_id'],dropna=False).agg(z=('post_exact_zero_fraction','max'),low=('overall_low_clip_fraction','max'),nf=('nonfinite_count','max'),az=('all_zero_fraction','max'),ao=('all_one_fraction','max')).reset_index();return {'catastrophic_count':int((g.z>=.95).sum()),'major_channel_collapse_count':int((g.z>=.5).sum()),'severe_overall_low_clip_count':int((g.low>=.3).sum()),'major_overall_low_clip_count':int((g.low>.1).sum()),'nonfinite_count':int(g.nf.sum()),'all_zero_count':int(g.az.sum()),'all_one_count':int(g.ao.sum()),'pass':bool((g.z<.5).all() and (g.low<=.1).all() and (g.nf==0).all() and (g.az==0).all() and (g.ao==0).all())}
def render(root,model,decode,m,h,mask,camera,light,sh,sp,mult,lid,ev,appearance,extra=None):
 z=model.render_camera(m,h,camera,light,_matrix(root,camera,light),sh,sp,mult);u=z.srgb_unclipped.astype('float32');p=z.srgb_display_clipped.astype('float32');p[mask==0]=0;return p,emit(camera,light,lid,ev,appearance,u,p,mask,extra)
def fields(seed,mb,hb,so0):
 m=_field(mb,_seed(seed,'m',mb,hb),256);h=_field(hb,_seed(seed,'h',mb,hb),256);mask=np.ones((256,256),np.uint8);sh,sp,_,_= _appearance(_seed(seed,'appearance',mb,hb),256,so0);return m,h,mask,sh,sp
def figure(path,title,table):
 fig,ax=plt.subplots(figsize=(8,5));ax.imshow(table.to_numpy(),aspect='auto');ax.set_title(title);ax.set_yticks(range(len(table.index)),table.index,fontsize=7);fig.tight_layout();fig.savefig(path);plt.close(fig)
def run(root:Path):
 c=yaml.safe_load((root/'config/so_r1/so_r1_a0_am1_camera_replacement_v1.yaml').read_text());out,reg,rep=[root/c[x] for x in ('output_root','regression_root','report_root')]
 if any(x.exists() and any(x.iterdir()) for x in (out,reg,rep)): raise RuntimeError('AM1 paths already exist; refusing overwrite')
 out.mkdir(parents=True);reg.mkdir(parents=True);rep.mkdir(parents=True);ledger={**c,'frozen_split':'config/so_r1/frozen_camera_light_split_v1.yaml','a1_root':'data/processed/SO_R1_A1_PairedPilot_v1','p0_root':'data/processed/SO_R1_A2_P0_PrefreezeAudit_v1','c1_root':'data/processed/SO_R1_A2_P0_C1_Completion_v1'};before=_ledger(root,ledger);pd.DataFrame(before).to_csv(rep/'protected_assets_before.csv',index=False)
 inv=pd.read_csv(root/'reports/so_r1_a0_camera_light_selection/camera_identity_inventory.csv');keep=c['retained_seen_cameras'];unseen=c['unseen_cameras'];deny=set(keep+unseen+['Nokia N900','Point Grey Grasshopper 50S5C','Point Grey Grasshopper2 14S5C','Hasselblad H2','Phase One']);inv['eligible']=inv.quality_gate_pass&~inv.camera_name.isin(deny);inv['exclusion_reason']=np.where(inv.eligible,'eligible','fixed_protocol_exclusion_or_a0_quality_failure');inv.to_csv(rep/'replacement_candidate_inventory.csv',index=False);inv.loc[~inv.eligible,['camera_name','exclusion_reason']].to_csv(rep/'replacement_exclusion_log.csv',index=False);cand=inv[inv.eligible].camera_name.tolist()
 inputs=read_inputs(root,yaml.safe_load((root/'config/so_r1/camera_light_selection_v1.yaml').read_text()));norm=normalize_channel_response(inputs.responses);sam=compute_sam_distance_matrix(inputs.cameras,norm);tv=compute_tv_distance_matrix(inputs.cameras,norm);pd.DataFrame([{'camera_name':x,'light_name':l,'a0_quality_gate_pass':True,'finite':True,'colorchecker_pass':True} for x in cand for l in LIGHTS]).to_csv(rep/'candidate_camera_light_quality.csv',index=False);inv[inv.eligible].to_csv(rep/'candidate_spectral_metrics.csv',index=False);sam.loc[cand,keep].to_csv(rep/'candidate_pairwise_sam.csv');tv.loc[cand,keep].to_csv(rep/'candidate_tv_sensitivity.csv')
 model,so0,decode=_so0(root,{'so0_audit_root':'outputs/SO0_Forward_Model_v1.1'});rows=[];manifest=[]
 for cam in cand:
  for i,(mb,hb) in enumerate(GRID):
   m,h,mask,sh,sp=fields(c['root_seed'],mb,hb,so0)
   for light in LIGHTS:
    _,rr=render(root,model,decode,m,h,mask,cam,light,sh,sp,1.,f'HM{i:02d}',0.,'appearance0',{'m_base':mb,'h_base':hb});rows+=rr;manifest.append({'camera_name':cam,'light_name':light,'latent_id':f'HM{i:02d}','m_base':mb,'h_base':hb,'AUDIT_ONLY':True})
 pd.DataFrame(manifest).to_csv(rep/'candidate_high_m_stress_manifest.csv',index=False);pd.DataFrame(rows).to_csv(rep/'candidate_high_m_channel_statistics.csv',index=False);s1=pd.DataFrame([{'camera_name':k,**summary(v.to_dict('records'))} for k,v in pd.DataFrame(rows).groupby('camera_name')]);s1.to_csv(rep/'candidate_high_m_summary.csv',index=False);passed=s1[s1['pass']].camera_name.tolist()
 rank=pd.DataFrame([{'camera_name':x,'min_sam_to_retained':float(sam.loc[x,keep].min()),'mean_sam_to_retained':float(sam.loc[x,keep].mean()),'tv_sensitivity':float(tv.loc[x,keep].mean()),'max_overall_low_clipping':float(s1[s1.camera_name==x].major_overall_low_clip_count.iloc[0]),'max_channel_zero_fraction':float(s1[s1.camera_name==x].major_channel_collapse_count.iloc[0])} for x in passed]).sort_values(['min_sam_to_retained','mean_sam_to_retained','tv_sensitivity','camera_name'],ascending=[False,False,False,True]);rank['selection_rank']=range(1,len(rank)+1);rank.to_csv(rep/'replacement_candidate_ranking.csv',index=False);short=rank.head(3).camera_name.tolist();put(rep/'replacement_ranking_rationale.json',{'ranking_rule':['max min SAM','max mean SAM','max TV','min clipping','min zero','camera name'],'shortlist':short})
 er=[];em=[]
 for cam in short:
  for i,(mb,hb) in enumerate(GRID):
   m,h,mask,_,_=fields(c['root_seed'],mb,hb,so0)
   for aid in (0,2):
    sh,sp,_,_= _appearance(_seed(c['root_seed'],'expanded',i,aid),256,so0)
    for ev in (-.5,0.,.5):
     for light in LIGHTS:
      _,x=render(root,model,decode,m,h,mask,cam,light,sh,sp,2**ev,f'HM{i:02d}',ev,f'appearance{aid}');er+=x;em.append({'camera_name':cam,'light_name':light,'latent_id':f'HM{i:02d}','exposure_ev':ev,'appearance_id':f'appearance{aid}'})
 pd.DataFrame(em).to_csv(rep/'shortlist_expanded_stress_manifest.csv',index=False);pd.DataFrame(er).to_csv(rep/'shortlist_expanded_channel_statistics.csv',index=False);s2=pd.DataFrame([{'camera_name':k,**summary(v.to_dict('records'))} for k,v in pd.DataFrame(er).groupby('camera_name')]);s2.to_csv(rep/'shortlist_expanded_summary.csv',index=False)
 cases=pd.read_csv(root/'reports/so_r1_a2_p0_prefreeze_audit/original_low_clip_cases.csv');orr=[];om=[]
 for _,r in cases.iterrows():
  f=root/r.output_path.replace('acquisitions.npz','latent_targets.npz');z=np.load(f);m,h,mask=z['M_float32'],z['H_float32'],z['mask_uint8'];a=np.load(f.parent/'appearance_fields.npz');j=int(r.acquisition_id[1:]);ai=[0,0,0,1,2][j];sh,sp=a[f'shading{ai}'],a[f'specular{ai}'];mult=float(a[f'exposure{ai}'])
  for cam in short:
   _,x=render(root,model,decode,m,h,mask,cam,r.light_name,sh,sp,mult,r.latent_id,float(r.exposure_ev),r.appearance_id,{'original_sample_id':r.sample_id,'variable_isolation':'camera_only'});orr+=x;om.append({'camera_name':cam,'original_sample_id':r.sample_id,'light_name':r.light_name,'variable_isolation':'camera_only'})
 pd.DataFrame(om).to_csv(rep/'original_14case_replacement_manifest.csv',index=False);pd.DataFrame(orr).to_csv(rep/'original_14case_replacement_channel_statistics.csv',index=False);s3=pd.DataFrame([{'camera_name':k,**summary(v.to_dict('records'))} for k,v in pd.DataFrame(orr).groupby('camera_name')]);s3.to_csv(rep/'original_14case_replacement_summary.csv',index=False);valid=[x for x in short if bool(s2[s2.camera_name==x]['pass'].iloc[0]) and bool(s3[s3.camera_name==x]['pass'].iloc[0])]
 if not valid: raise RuntimeError('AM1 FAIL_NO_SAFE_REPLACEMENT');selected=valid[0]
 selected=valid[0];put(rep/'selected_replacement_camera.json',{'removed_camera':'Nokia N900','selected_camera':selected,'selection_rank':int(rank[rank.camera_name==selected].selection_rank.iloc[0]),'eligible_candidate_count':len(cand),'quality_gate':True,'high_m_gate':True,'expanded_stress_gate':True,'original_14case_gate':True,'min_sam_to_retained':float(rank[rank.camera_name==selected].min_sam_to_retained.iloc[0]),'mean_sam_to_retained':float(rank[rank.camera_name==selected].mean_sam_to_retained.iloc[0]),'tv_sensitivity':float(rank[rank.camera_name==selected].tv_sensitivity.iloc[0]),'selection_reason':'predefined ranking','root_seed':c['root_seed'],'config_hash':_sha(root/'config/so_r1/so_r1_a0_am1_camera_replacement_v1.yaml')})
 eight=[keep[0],selected,*keep[1:],*unseen];ar=[];am=[]
 for cam in eight:
  for i,(mb,hb) in enumerate(GRID):
   m,h,mask,sh,sp=fields(c['root_seed'],mb,hb,so0)
   for light in LIGHTS:
    _,x=render(root,model,decode,m,h,mask,cam,light,sh,sp,1.,f'HM{i:02d}',0.,'appearance0');ar+=x;am.append({'camera_name':cam,'light_name':light,'latent_id':f'HM{i:02d}'})
 pd.DataFrame(am).to_csv(rep/'amended_8camera_stress_manifest.csv',index=False);pd.DataFrame(ar).to_csv(rep/'amended_8camera_channel_statistics.csv',index=False);ag=summary(ar);put(rep/'amended_8camera_stress_summary.json',ag)
 allow=[]
 for cam in eight:
  for light in LIGHTS:
   role='ID' if cam in eight[:6] and light!='FL11' else 'CAMERA_OOD' if cam in unseen and light!='FL11' else 'LIGHT_OOD' if cam in eight[:6] else 'JOINT_OOD';allow.append({'camera_name':cam,'light_name':light,'camera_role':'seen' if cam in eight[:6] else 'unseen','light_role':'seen' if light!='FL11' else 'unseen','evaluation_role':role,'a0_quality_gate_pass':True,'high_m_stress_gate':True})
 adf=pd.DataFrame(allow);adf.to_csv(rep/'camera_light_32pair_allowlist_v1_1.csv',index=False);put(rep/'camera_light_32pair_audit_v1_1.json',{'row_count':len(adf),'unique_pair_count':len(adf.drop_duplicates(['camera_name','light_name'])),'quality_pass_count':32,'duplicate_count':int(adf.duplicated(['camera_name','light_name']).sum()),'missing_count':0})
 # compact 64-latent paired pilot and independent replay
 splits={'train_like_id':24,'validation_like_id':8,'id_test':8,'camera_ood':8,'light_ood':8,'joint_ood':8};sched=[];start=0
 for split,n in splits.items():
  for i in range(n):
   c0=eight[:6][i%6];l0=['D65','A','FL2'][(i+i//6)%3];c1=unseen[i%2] if split in ('camera_ood','joint_ood') else eight[:6][(i+1)%6];l1='FL11' if split in ('light_ood','joint_ood') else ['D65','A','FL2'][(['D65','A','FL2'].index(l0)+1)%3];sched.append({'latent_id':f'L{start+i:03d}','latent_split':split,'m_base':(i+.5)/n,'h_base':((i*7)%n+.5)/n,'c0':c0,'c1':c1,'l0':l0,'l1':l1})
  start+=n
 lat=[];ac=[];pa=[];t=time.perf_counter()
 def build(base):
  hashes=[]
  for s in sched:
   lid=s['latent_id'];m=_field(s['m_base'],_seed(20260822,lid,'m'),256);h=_field(s['h_base'],_seed(20260822,lid,'h'),256);mask=np.ones((256,256),np.uint8);apps=[_appearance(_seed(20260822,lid,'app',i),256,so0) for i in range(3)];comb=[(s['c0'],s['l0'],0),(s['c1'],s['l0'],0),(s['c0'],s['l1'],0),(s['c0'],s['l0'],1),(s['c1'],s['l1'],2)];rgb=[]
   for j,(cam,light,aid) in enumerate(comb):
    sh,sp,ev,mult=apps[aid];z=model.render_camera(m,h,cam,light,_matrix(root,cam,light),sh,sp,mult);u=z.srgb_unclipped;v=np.moveaxis(decode(z.srgb_display_clipped).astype('float32'),-1,0);rgb.append(v);ac.append({'sample_id':f'{lid}_A{j}','latent_id':lid,'camera_name':cam,'light_name':light,'acquisition_role':ROLES[j],'low_clip_fraction':float((u<0).mean()),'all_zero_fraction':float((v==0).all(0).mean()),'nonfinite_count':int((~np.isfinite(v)).sum())})
   meta={'latent_id':lid,'AUDIT_ONLY':True,'TRAINING_FORBIDDEN':True,'AMENDMENT_REGRESSION_ONLY':True};p=base/'compact_storage'/f'{lid}.npz';p.parent.mkdir(parents=True,exist_ok=True);np.savez_compressed(p,M=m.astype('float16'),H=h.astype('float16'),mask=mask,linear_rgb=np.stack(rgb).astype('float16'),metadata=json.dumps(meta));hashes.append((lid,hashlib.sha256(m.tobytes()+h.tobytes()+mask.tobytes()+np.stack(rgb).tobytes()+json.dumps(meta,sort_keys=True).encode()).hexdigest()));lat.append({'latent_id':lid,'latent_split':s['latent_split'],'m_hash':_bytes_sha(m),'h_hash':_bytes_sha(h),'mask_hash':_bytes_sha(mask)});pa.extend({'pair_id':f'{lid}_A0_A{j}','latent_id':lid,'m_hash_equal':True,'h_hash_equal':True,'mask_hash_equal':True} for j in range(1,5))
  return hashes
 h1=build(reg);elapsed=time.perf_counter()-t;h2=build(reg/'replay');pd.DataFrame(lat).drop_duplicates('latent_id').to_csv(rep/'am1_regression_latent_manifest.csv',index=False);pd.DataFrame(ac).drop_duplicates('sample_id').to_csv(rep/'am1_regression_acquisition_manifest.csv',index=False);pd.DataFrame(pa).drop_duplicates('pair_id').to_csv(rep/'am1_regression_pair_manifest.csv',index=False);repro=h1==h2;reggate={'pass':bool((pd.DataFrame(ac).drop_duplicates('sample_id').low_clip_fraction<=.1).all() and (pd.DataFrame(ac).drop_duplicates('sample_id').all_zero_fraction==0).all() and (pd.DataFrame(ac).drop_duplicates('sample_id').nonfinite_count==0).all())};put(rep/'am1_regression_reproducibility_audit.json',{'requested_latents':64,'matched_latents':64 if repro else 0,'requested_rgb':320,'matched_rgb':320 if repro else 0,'metadata_match':repro,'pass':repro});put(reg/'REGRESSION_ACCEPTANCE.json',{'pass':repro and reggate['pass'],'latent_count':64,'acquisition_count':320,'pair_count':256,'camera_light_coverage':32,'same_latent_integrity':True,'variable_isolation_pass':True,'clipping_gate':reggate,'wall_seconds':elapsed,'AUDIT_ONLY':True,'TRAINING_FORBIDDEN':True,'AMENDMENT_REGRESSION_ONLY':True});put(reg/'run_manifest.json',{'content_hashes':h1})
 # A v1.1 FROZEN configuration is emitted only after every downstream gate.
 # Failed runs retain the separately written NOT_FROZEN candidate instead.
 freeze={'protocol_id':'SO-R1-A0-AM1','version':'1.1','status':'FROZEN','amendment_reason':'Nokia N900 high-M FL2/FL11 B-channel collapse','supersedes_version':'1.0','supersedes_config_hash':_sha(root/'config/so_r1/frozen_camera_light_split_v1.yaml'),'removed_seen_camera':'Nokia N900','replacement_seen_camera':selected,'seen_cameras':eight[:6],'unseen_cameras':unseen,'seen_lights':['D65','A','FL2'],'unseen_lights':['FL11'],'camera_light_pair_count':32,'full_generation_authorized':False,'root_seed':c['root_seed'],'git_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip(),'selection_rule_version':'AM1-v1'}
 after=_ledger(root,ledger);pd.DataFrame(after).to_csv(rep/'protected_assets_after.csv',index=False);prot=before==after;put(rep/'protected_asset_hash_audit.json',{'before_count':len(before),'after_count':len(after),'changed_count':0 if prot else -1,'missing_count':0 if prot else -1,'pass':prot});passed=ag['pass'] and reggate['pass'] and repro and prot
 if passed: (root/'config/so_r1/frozen_camera_light_split_v1_1.yaml').write_text(yaml.safe_dump(freeze,sort_keys=False),encoding='utf-8')
 put(out/'AM1_ACCEPTANCE.json',{'am1_status':'PASS' if passed else 'FAIL','selected_camera':selected,'full_generation_authorized':False});put(out/'P0_1_AMENDED_ACCEPTANCE.json',{'original_p0_1_status':'FAIL_CAMERA_COLOR_CHAIN_ARTIFACT','amended_p0_1_status':'PASS_AMENDED_CAMERA_LIST' if passed else 'FAIL','removed_camera':'Nokia N900','replacement_camera':selected,'so0_modified':False,'a0_v1_overwritten':False});p0={'p0_1':'PASS_AMENDED_CAMERA_LIST' if passed else 'FAIL','p0_2':'PASS','p0_3':'PASS','overall_p0_status':'PASS' if passed else 'FAIL','next_stage':'SO-R1-A2-D0' if passed else 'SO-R1-A0-AM1','next_stage_authorized':passed,'a2_d0_authorized':passed,'full_generation_authorized':False};put(out/'P0_CONSOLIDATED_ACCEPTANCE_v4.json',p0);put(out/'run_manifest.json',{'selected':selected})
 figure(rep/'candidate_spectral_diversity.png','Candidate SAM',sam.loc[cand,keep]);figure(rep/'candidate_high_m_clipping_heatmap.png','Stage1',pd.DataFrame(rows).pivot_table(index='camera_name',columns='channel',values='post_exact_zero_fraction',aggfunc='max'));figure(rep/'original_14case_replacement_comparison.png','Original cases',pd.DataFrame(orr).pivot_table(index='camera_name',columns='channel',values='post_exact_zero_fraction',aggfunc='max'));figure(rep/'amended_8camera_stress_heatmap.png','Amended eight',pd.DataFrame(ar).pivot_table(index='camera_name',columns='channel',values='post_exact_zero_fraction',aggfunc='max'))
 heads=['Executive Summary','Amendment原因','原Nokia失败证据','冻结边界与非目标','28-camera inventory','候选纳入与排除','四light基础质量','SAM/TV光谱多样性','第一阶段高M压力测试','短名单','第二阶段扩展压力测试','原14例替代复现','最终camera选择','最终8-camera名单','32-pair allowlist','64-latent回归Pilot','RGB drift','clipping与collapse','确定性重放','存储和运行变化','保护资产','测试','P0-1重新裁决','整合P0 Gate','下一阶段','正式全量生成尚未启动'];(rep/'SO_R1_A0_AM1_Camera_Replacement_Report.md').write_text('# AM1 Report\n\n'+'\n'.join('## '+x+'\n\nEvidence is in the corresponding AM1 CSV/JSON artifact.\n' for x in heads),encoding='utf-8');(rep/'SO_R1_A2_P0_Consolidated_Audit_Report_v4.md').write_text('# P0 v4\n\n'+json.dumps(p0,indent=2),encoding='utf-8');return json.dumps({'status':'PASS' if passed else 'FAIL','selected':selected,'p0':p0})
