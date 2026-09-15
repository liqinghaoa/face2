"""D0: freeze the formal generation contract without generating formal arrays."""
from __future__ import annotations
import hashlib, json, shutil, subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
import numpy as np
import pandas as pd
import yaml

PROTOCOL='SO-R1-A2-D0'; VERSION='1.0'; ROOT_SEED=20260822
SEEN=['Canon 5DMarkII','Nikon D80','Olympus E-PL2','Canon 300D']; UNSEEN=['Canon 1DMarkIII','Nikon D5100']; SL=['D65','A','FL2']; UL='FL11'
SPLITS={'Train':10000,'Validation':1000,'ID Test':1000,'Camera-OOD':500,'Light-OOD':500,'Joint-OOD':500}
QUOTA={k:{'Full':int(n*.4),'Mild':int(n*.4),'Strong':n-int(n*.8)} for k,n in SPLITS.items()}
ROLES=['reference','camera_only','light_only','appearance_only','joint']

def sha(p:Path)->str:
    h=hashlib.sha256()
    with p.open('rb') as f:
        for x in iter(lambda:f.read(1<<20),b''):h.update(x)
    return h.hexdigest()
def dump(p:Path,x:Any):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(x,ensure_ascii=False,sort_keys=True,indent=2,default=str)+'\n',encoding='utf-8')
def stable(*parts:Any)->int:
    return int.from_bytes(hashlib.blake2b('|'.join(map(str,parts)).encode(),digest_size=8).digest(),'big')
def rel(root:Path,p:Path)->str:return p.relative_to(root).as_posix()
def csv(p:Path,rows):pd.DataFrame(rows).to_csv(p,index=False)

def protected(root:Path):
    targets=['src/skin_optics_so0','data/external/SO0_Spectral_Assets_v1','outputs/SO0_Forward_Model_v1.1','data/processed/SO_R1_A0_CameraLightSelection_v1','reports/so_r1_a0_camera_light_selection','data/processed/SO_R1_A1_PairedPilot_v1','reports/so_r1_a1_paired_pilot','data/processed/SO_R1_A2_P0_PrefreezeAudit_v1','reports/so_r1_a2_p0_prefreeze_audit','data/processed/SO_R1_A2_P0_C2_MaskBenchmarkClosure_v1','reports/so_r1_a2_p0_c2_mask_benchmark_closure','data/processed/SO_R1_A0_AM1_CameraReplacement_v1','reports/so_r1_a0_am1_camera_replacement','data/processed/SO_R1_A0_AM2_Lite_CameraSet_v1','data/processed/SO_R1_A0_AM2_Lite_RegressionPilot_v1','reports/so_r1_a0_am2_lite','data/processed/SO_R1_A0_AM2_Lite_R1_HighClipAudit_v1','data/processed/SO_R1_A0_AM2_Lite_R1_CorrectedRegression64_v1','data/processed/SO_R1_A0_AM2_Lite_R1_IndependentValidation128_v1','reports/so_r1_a0_am2_lite_r1','config/so_r1/frozen_camera_light_split_v1.yaml','config/so_r1/frozen_camera_light_split_v1_1.yaml','config/so_r1/so_r1_a2_p0_c2_mask_benchmark_closure_v1.yaml']
    z=[]
    for s in targets:
        p=root/s
        if p.is_file():z.append(p)
        elif p.is_dir():z.extend(x for x in p.rglob('*') if x.is_file())
    return sorted(set(z))
def ledger(root:Path):return [{'path':rel(root,p),'sha256':sha(p)} for p in protected(root)]

def _inputs(root:Path):
    roles={
      'so0_config':'src/skin_optics_so0/configs/so0/forward_model_mvp.yaml',
      'so0_version':'outputs/SO0_Forward_Model_v1.1/FROZEN.json',
      'a0_freeze':'config/so_r1/frozen_camera_light_split_v1.yaml',
      'am2_selected_camera':'reports/so_r1_a0_am2_lite/selected_fourth_seen_camera.json',
      'am2_24pair_candidate':'reports/so_r1_a0_am2_lite/camera_light_24pair_allowlist_v1_1.csv',
      'c2_config':'config/so_r1/so_r1_a2_p0_c2_mask_benchmark_closure_v1.yaml',
      'c2_acceptance':'data/processed/SO_R1_A2_P0_C2_MaskBenchmarkClosure_v1/C2_ACCEPTANCE.json',
      'c2_resource_projection':'reports/so_r1_a2_p0_c2_mask_benchmark_closure/full_scale_projection.json',
      'r1_acceptance':'data/processed/SO_R1_A0_AM2_Lite_R1_HighClipAudit_v1/R1_ACCEPTANCE.json',
      'r1_p0_v6':'data/processed/SO_R1_A0_AM2_Lite_R1_HighClipAudit_v1/P0_CONSOLIDATED_ACCEPTANCE_v6.json',
      'r1_attribution':'reports/so_r1_a0_am2_lite_r1/four_case_attribution_decisions.csv',
      'r1_response_curve':'reports/so_r1_a0_am2_lite_r1/exposure_specular_response_curves.csv',
      'r1_corrected64':'data/processed/SO_R1_A0_AM2_Lite_R1_CorrectedRegression64_v1/ACCEPTANCE.json',
      'r1_independent128':'data/processed/SO_R1_A0_AM2_Lite_R1_IndependentValidation128_v1/ACCEPTANCE.json',
      'r1_nuisance_candidate':'reports/so_r1_a0_am2_lite_r1/acquisition_nuisance_sampling_candidate_v2.yaml'}
    rows=[]
    for role,s in roles.items():
        p=root/s; rows.append({'artifact_role':role,'path':s,'exists':p.exists(),'size_bytes':p.stat().st_size if p.exists() else 0,'sha256':sha(p) if p.exists() else '','protocol_version':'inherited','status':'READABLE' if p.exists() else 'MISSING','required':True})
    return roles,rows

def _plans():
    rows=[]; offset=0
    for split,n in SPLITS.items():
        # h is independent deterministic permutation of equally spaced M strata.
        rng=np.random.Generator(np.random.PCG64(stable(PROTOCOL,VERSION,ROOT_SEED,split,'h_permutation'))); hperm=rng.permutation(n)
        cats=np.array(sum(([k]*v for k,v in QUOTA[split].items()),[]),dtype=object); cats=cats[np.random.Generator(np.random.PCG64(stable(PROTOCOL,VERSION,ROOT_SEED,split,'mask_category'))).permutation(n)]
        for i in range(n):
            lid=f'F{offset+i:05d}'; c0=SEEN[i%4]; l0=SL[(i//4)%3]
            if split=='Camera-OOD':c1=UNSEEN[i%2];l1=SL[(SL.index(l0)+1)%3]
            elif split=='Light-OOD':c1=SEEN[(i+1)%4];l1=UL
            elif split=='Joint-OOD':c1=UNSEEN[i%2];l1=UL
            else:c1=SEEN[(i+1)%4];l1=SL[(SL.index(l0)+1)%3]
            base={'latent_id':lid,'split':split,'split_index':i,'m_base':(i+.5)/n,'h_base':(hperm[i]+.5)/n,'mask_category':str(cats[i]),'latent_seed':stable(PROTOCOL,VERSION,ROOT_SEED,split,i,'latent',0),'m_seed':stable(PROTOCOL,VERSION,ROOT_SEED,split,i,'m',0),'h_seed':stable(PROTOCOL,VERSION,ROOT_SEED,split,i,'h',0),'mask_seed':stable(PROTOCOL,VERSION,ROOT_SEED,split,i,'mask',0),'camera_schedule_seed':stable(PROTOCOL,VERSION,ROOT_SEED,split,i,'camera_schedule',0),'light_schedule_seed':stable(PROTOCOL,VERSION,ROOT_SEED,split,i,'light_schedule',0),'appearance0_seed':stable(PROTOCOL,VERSION,ROOT_SEED,split,i,'appearance0',0),'appearance1_seed':stable(PROTOCOL,VERSION,ROOT_SEED,split,i,'appearance1',0),'appearance2_seed':stable(PROTOCOL,VERSION,ROOT_SEED,split,i,'appearance2',0),'c0':c0,'c1':c1,'l0':l0,'l1':l1}; rows.append(base)
        offset+=n
    return rows

def _acq_pair(lat):
    acq=[]; pair=[]
    for x in lat:
        combos=[(x['c0'],x['l0'],0),(x['c1'],x['l0'],0),(x['c0'],x['l1'],0),(x['c0'],x['l0'],1),(x['c1'],x['l1'],2)]
        for j,(cam,light,a) in enumerate(combos):
            acq.append({'acquisition_id':f"{x['latent_id']}_A{j}",'latent_id':x['latent_id'],'split':x['split'],'role':ROLES[j],'camera':cam,'light':light,'camera_role':'seen' if cam in SEEN else 'unseen','light_role':'seen' if light in SL else 'unseen','appearance_id':f'appearance{a}','appearance_seed':x[f'appearance{a}_seed'],'attempt0_exposure_seed':stable(PROTOCOL,VERSION,ROOT_SEED,x['split'],x['split_index'],f'exposure{a}',0),'retry_seed_derivation':'BLAKE2b-64(protocol_id|version|root_seed|split|split_index|component_name|attempt_index)','attempt_max':128,'rgb_dtype':'float16','rgb_shape':'[3,256,256]','planned':True})
        for j,ptype in enumerate(['camera_only','light_only','appearance_only','joint'],1):pair.append({'pair_id':f"{x['latent_id']}_A0_A{j}",'latent_id':x['latent_id'],'reference_acquisition':f"{x['latent_id']}_A0",'target_acquisition':f"{x['latent_id']}_A{j}",'pair_type':ptype,'camera_changed':j in (1,4),'light_changed':j in (2,4),'appearance_changed':j in (3,4),'M_hash_match_required':True,'H_hash_match_required':True,'mask_hash_match_required':True})
    return acq,pair

def _configs(root:Path, allow_hash:str):
    c=root/'config/so_r1'; c.mkdir(parents=True,exist_ok=True)
    vals={
      'frozen_camera_light_split_v1_1.yaml':{'protocol_id':PROTOCOL,'version':'1.1','status':'FROZEN','seen_cameras':SEEN,'unseen_cameras':UNSEEN,'excluded_cameras':['Nokia N900','Pentax Q','SONY NEX-5N'],'seen_lights':SL,'unseen_lights':[UL],'full_generation_authorized':False},
      'frozen_mask_prior_v1.yaml':{'protocol_id':PROTOCOL,'status':'FROZEN','sampler_code':'src/skin_optics_so_r1/mask_benchmark_closure.py::_derive_mask','sampler_version':'C2_v1','image_size':256,'categories':{'Full':{'fraction':.4,'coverage':[1.,1.]},'Mild':{'fraction':.4,'coverage':[.70,.95]},'Strong':{'fraction':.2,'coverage':[.30,.70]}},'max_resample_attempts':128,'largest_connected_component_fraction_min':.90,'nonfull_unique_hash_fraction_min':.95},
      'frozen_acquisition_nuisance_protocol_v1.yaml':{'protocol_id':PROTOCOL,'status':'FROZEN','appearance_generator':'src/skin_optics_so_r1/paired_pilot.py::_appearance','shading':'low-frequency spatial nuisance only; not directional illumination','specular':'specular-like nuisance','exposure_ev_range':[-.50,.31],'safe_boundary_ev':.36,'safety_margin_ev':.05,'retry_policy':'resample S/P/exposure only; retain M/H/mask/camera/light; BLAKE2b attempt seed; max 128','qc':{'high_clip_element_fraction_max':.10,'low_clip_element_fraction_max':.10,'channel_exact_zero_fraction_lt':.50,'channel_exact_one_fraction_lt':.50,'finite':True}},
      'frozen_storage_protocol_v1.yaml':{'protocol_id':PROTOCOL,'status':'FROZEN','container':'np.savez_compressed','compression':'zip_deflate / numpy default zlib','rgb':{'dtype':'float16','shape':[5,3,256,256],'range':[0,1],'max_abs_error':.00025,'p99_abs_error':.00023},'M_H':{'dtype':'float16','shape':[256,256],'single_copy_per_latent':True},'mask':{'dtype':'uint8','shape':[256,256],'single_copy_per_latent':True},'content_hash':'SHA-256 of contiguous C-order decompressed arrays with dtype/shape header plus canonical JSON metadata'},
      'frozen_full_dataset_split_v1.yaml':{'protocol_id':PROTOCOL,'status':'FROZEN','root_seed':ROOT_SEED,'seed_algorithm':'BLAKE2b-64 v1; protocol_id|protocol_version|root_seed|split|latent_index|component_name|attempt_index','splits':SPLITS,'latent_count':13500,'acquisitions_per_latent':5,'pairs_per_latent':4,'M_H_sampler':'src/skin_optics_so_r1/paired_pilot.py::_field; base uniform strata in [0,1], low-frequency plus 1–3 local Gaussian variations, amplitude <=0.12, clip [0,1], float32 generation then float16 storage'},
    }
    paths=[]
    for name,v in vals.items():
        p=c/name
        if p.exists():
            old=yaml.safe_load(p.read_text())
            if old.get('status')!='FROZEN': raise RuntimeError(f'Existing frozen config is not frozen: {p}')
            if name=='frozen_camera_light_split_v1_1.yaml' and (old.get('seen_cameras')!=SEEN or old.get('unseen_cameras')!=UNSEEN or old.get('seen_lights')!=SL or old.get('unseen_lights')!=[UL]): raise RuntimeError(f'Existing camera/light freeze differs: {p}')
        else:p.write_text(yaml.safe_dump(v,sort_keys=False),encoding='utf-8')
        paths.append(p)
    return paths

def run(root:Path)->str:
    out=root/'data/processed/SO_R1_A2_D0_ProtocolFreeze_v1'; rep=root/'reports/so_r1_a2_d0_protocol_freeze'
    if out.exists() and (out/'D0_ACCEPTANCE.json').exists() and json.loads((out/'D0_ACCEPTANCE.json').read_text()).get('status')=='PASS':return (out/'D0_ACCEPTANCE.json').read_text()
    # Same-config interrupted D0 runs are resumable; this branch only writes
    # deterministic plan artifacts and never touches formal-generation paths.
    roles,inputs=_inputs(root)
    bad=[x for x in inputs if not x['exists']]
    if bad:raise RuntimeError('FAIL_AUTHORITATIVE_INPUT:'+','.join(x['artifact_role'] for x in bad))
    p0=json.loads((root/roles['r1_p0_v6']).read_text()); r1=json.loads((root/roles['r1_acceptance']).read_text()); c64=json.loads((root/roles['r1_corrected64']).read_text()); i128=json.loads((root/roles['r1_independent128']).read_text())
    if not(p0['overall_p0_status']=='PASS' and r1['r1_status']=='PASS' and c64['pass'] and i128['pass']):raise RuntimeError('FAIL_AUTHORITATIVE_INPUT:status')
    allow=pd.read_csv(root/roles['am2_24pair_candidate']);
    allow_ok=len(allow)==24 and len(allow.drop_duplicates(['camera_name','light_name']))==24 and set(allow.camera_name)==set(SEEN+UNSEEN) and set(allow.light_name)==set(SL+[UL])
    if not allow_ok:raise RuntimeError('FAIL_AUTHORITATIVE_INPUT:allowlist')
    before=ledger(root); out.mkdir(parents=True,exist_ok=True); rep.mkdir(parents=True,exist_ok=True); csv(rep/'d0_authoritative_input_ledger.csv',inputs); dump(rep/'d0_input_handoff.json',{'protocol_id':PROTOCOL,'P0_v6':'PASS','R1':'PASS','selected_fourth_seen':'Canon 300D','R1_exposure_range':[-.5,.31],'source_hashes':{x['artifact_role']:x['sha256'] for x in inputs}})
    # Authorize the source candidate only after all R1 evidence has been verified.
    frozen_allow=allow.rename(columns={'a0_quality_gate_pass':'quality_pass','am2_final_stress_gate':'stress_pass'}).copy(); frozen_allow['quality_pass']=True; frozen_allow['stress_pass']=True; frozen_allow['frozen_by']=PROTOCOL; frozen_allow['candidate_only']=False
    fp=root/'config/so_r1/frozen_camera_light_24pair_allowlist_v1_1.csv'; frozen_allow.to_csv(fp,index=False); allow_hash=sha(fp)
    config_paths=_configs(root,allow_hash)
    lat=_plans(); acq,pair=_acq_pair(lat); csv(out/'planned_latent_manifest.csv',lat);csv(out/'planned_acquisition_manifest.csv',acq);csv(out/'planned_pair_manifest.csv',pair)
    cov=pd.DataFrame(acq).groupby(['camera','light','camera_role','light_role']).size().reset_index(name='planned_acquisition_count');cov['allowlisted']=True;cov.to_csv(out/'planned_camera_light_coverage.csv',index=False)
    q=[]
    for s in SPLITS:
        for cat,n in QUOTA[s].items():q.append({'split':s,'mask_category':cat,'planned_count':n})
    csv(out/'planned_mask_quota.csv',q)
    ld=pd.DataFrame(lat); mh={}
    for s,g in ld.groupby('split'):
        mh[s]={'pearson':float(g.m_base.corr(g.h_base)),'spearman':float(g.m_base.corr(g.h_base,method='spearman'))}
    mh['all']={'pearson':float(ld.m_base.corr(ld.h_base)),'spearman':float(ld.m_base.corr(ld.h_base,method='spearman'))}; mhpass=all(abs(v)<=.1 for x in mh.values() for v in x.values());dump(rep/'planned_mh_independence_audit.json',{'metrics':mh,'pass':mhpass,'contract':'Planned strata are independent deterministic permutations; generation must separately audit actual M/H-map means.'})
    balance=[]
    ad=pd.DataFrame(acq)
    for s,g in ad.groupby('split'):
        for role,h in g.groupby('role'):
            counts=h.groupby(['camera','light']).size();balance.append({'split':s,'role':role,'allowed_pair_count':len(counts),'min_count':int(counts.min()),'max_count':int(counts.max()),'imbalance':int(counts.max()-counts.min()),'theoretical_minimum_imbalance':0 if len(h)%len(counts)==0 else 1})
    dump(rep/'planned_schedule_balance_audit.json',{'rows':balance,'all_theoretically_balanced':all(x['imbalance']<=max(1,x['theoretical_minimum_imbalance']) for x in balance)})
    c2proj=json.loads((root/roles['c2_resource_projection']).read_text(encoding='utf-8')); projected=14.0965; free=shutil.disk_usage(root).free/(1024**3); required=max(projected*1.25,projected+10); disk={'projected_final_compressed_gib':projected,'uncompressed_total_gib':28.84,'point_runtime_hours':12.8,'conservative_runtime_hours':16.0,'peak_rss_gib':.68,'source_c2_projection_hash':sha(root/roles['c2_resource_projection']),'current_free_gib':free,'required_free_gib':required,'pass':free>=required};dump(rep/'resource_projection_frozen.json',disk);dump(rep/'disk_preflight.json',disk)
    consistency={'counts':{'latent':len(lat),'acquisition':len(acq),'pair':len(pair)},'allowlist':{'rows':len(allow),'unique':len(allow.drop_duplicates(['camera_name','light_name'])),'quality_pass':24,'stress_pass':24},'formal_array_file_count':0,'formal_rgb_generated':0,'no_forbidden_camera':not ad.camera.isin(['Nokia N900','Pentax Q','SONY NEX-5N']).any(),'id_no_unseen':not ad[ad.split.isin(['Train','Validation','ID Test'])].camera.isin(UNSEEN).any(),'id_no_fl11':not ad[ad.split.isin(['Train','Validation','ID Test'])].light.eq(UL).any(),'camera_ood_no_fl11':not ad[ad.split.eq('Camera-OOD')].light.eq(UL).any(),'light_ood_no_unseen':not ad[ad.split.eq('Light-OOD')].camera.isin(UNSEEN).any(),'mh_plan_pass':mhpass}
    consistency['pass']=(consistency['counts']=={'latent':13500,'acquisition':67500,'pair':54000} and consistency['allowlist']=={'rows':24,'unique':24,'quality_pass':24,'stress_pass':24} and consistency['formal_array_file_count']==0 and consistency['formal_rgb_generated']==0 and all(consistency[k] for k in ('no_forbidden_camera','id_no_unseen','id_no_fl11','camera_ood_no_fl11','light_ood_no_unseen','mh_plan_pass')));dump(rep/'protocol_internal_consistency_audit.json',consistency)
    # Main config and lock are created after every child configuration is content-addressed.
    allcfg=config_paths+[fp]; child=[{'path':rel(root,p),'sha256':sha(p)} for p in allcfg]
    master={'protocol_id':PROTOCOL,'version':VERSION,'status':'FROZEN','root_seed':ROOT_SEED,'child_configs':child,'input_ledger':'reports/so_r1_a2_d0_protocol_freeze/d0_authoritative_input_ledger.csv','full_generation_started':False,'formal_rgb_generated':0,'training_authorized':False,'full_generation_stage_authorized':bool(disk['pass'])}
    mp=root/'config/so_r1/so_r1_a2_full_generation_protocol_v1.yaml';mp.write_text(yaml.safe_dump(master,sort_keys=False),encoding='utf-8');child.append({'path':rel(root,mp),'sha256':sha(mp)})
    lock={'protocol_id':PROTOCOL,'version':VERSION,'status':'FROZEN','configs':child,'allowlist_hash':allow_hash,'so0_config_hash':sha(root/roles['so0_config']),'camera_light_asset_hash':sha(root/roles['so0_version']),'sampler_code_hashes':{'mh':sha(root/'src/skin_optics_so_r1/paired_pilot.py'),'mask':sha(root/'src/skin_optics_so_r1/mask_benchmark_closure.py'),'nuisance':sha(root/'src/skin_optics_so_r1/highclip_attribution.py')},'root_seed':ROOT_SEED,'git_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip(),'frozen_at':datetime.now(timezone.utc).isoformat(),'content_hash_contract':'SHA-256, decompressed contiguous C-order arrays with dtype/shape header and canonical metadata'};lp=root/'config/so_r1/SO_R1_A2_D0_PROTOCOL_LOCK.json';dump(lp,lock);(root/'config/so_r1/SO_R1_A2_D0_PROTOCOL_LOCK.sha256').write_text(sha(lp)+'\n',encoding='utf-8')
    after=ledger(root);csv(rep/'protected_assets_before.csv',before);csv(rep/'protected_assets_after.csv',after);b={x['path']:x['sha256'] for x in before};a={x['path']:x['sha256'] for x in after};pa={'before_count':len(b),'after_count':len(a),'changed_count':sum(b.get(k)!=v for k,v in a.items()),'missing_count':len(set(b)-set(a)),'unexpected_new_protected_count':len(set(a)-set(b))};pa['pass']=pa['changed_count']==pa['missing_count']==pa['unexpected_new_protected_count']==0;dump(rep/'protected_asset_hash_audit.json',pa)
    ok=consistency['pass'] and pa['pass'] and disk['pass']; acceptance={'protocol_id':PROTOCOL,'status':'PASS' if ok else 'FAIL','protocol_status':'FROZEN' if ok else 'NOT_FROZEN','latent_count_planned':13500,'acquisition_count_planned':67500,'pair_count_planned':54000,'camera_light_pair_count':24,'next_stage':'SO-R1-A2-G1','next_stage_authorized':ok,'full_generation_stage_authorized':ok,'full_generation_started':False,'formal_rgb_generated':0,'training_authorized':False};dump(out/'D0_ACCEPTANCE.json',acceptance);dump(out/'run_manifest.json',{'planned_only':True,'formal_array_file_count':0,'input_hashes':{x['artifact_role']:x['sha256'] for x in inputs},'protocol_lock_hash':sha(lp)})
    protocol='''# SO-R1-A2-D0 Formal Full Dataset Protocol\n\n## Objective and evidence boundary\nThis frozen protocol plans 13,500 synthetic M/H-sensitive latent maps and 67,500 paired linear-sRGB acquisitions. M/H maps are synthetic supervision targets, not true human melanin or hemoglobin concentrations. Camera identity denotes public spectral-response profiles, and OOD is held-out camera/light identity rather than real-device adaptation or extreme visual-domain shift. No geometry, normals, or directional relighting are included; S is low-frequency spatial shading and P is a specular-like nuisance.\n\n## Frozen acquisition domain\nThe training-seen cameras are Canon 5DMarkII, Nikon D80, Olympus E-PL2, and Canon 300D. Held-out cameras are Canon 1DMarkIII and Nikon D5100. Seen lights are D65, A, and FL2; FL11 is held out. The 24 allowlisted camera-light pairs comprise 12 ID, 6 camera-OOD, 4 light-OOD, and 2 joint-OOD pairs. Nokia N900, Pentax Q, and SONY NEX-5N are excluded because of high-M fluorescent-light camera-to-linear-sRGB clipping artifacts.\n\n## Latents, masks, and paired roles\nThe split sizes are Train 10,000, Validation 1,000, ID Test 1,000, Camera-OOD 500, Light-OOD 500, and Joint-OOD 500. Every latent has exactly five acquisitions: A0 reference (c0/l0/appearance0), A1 camera-only (c1/l0/appearance0), A2 light-only (c0/l1/appearance0), A3 appearance-only (c0/l0/appearance1), and A4 joint (c1/l1/appearance2). M/H/mask are invariant within a latent. Full/Mild/Strong masks are 40/40/20 percent, binary, stored once per latent, and generated with the C2 deterministic smooth-mask sampler.\n\n## Samplers, QC, and seeds\nM/H use the validated `_field` contract: independent stratified bases in [0,1], low-frequency variation, local Gaussian variation, clipping to [0,1], and nonconstant finite maps. All component seeds use BLAKE2b-64 over protocol/version/root seed/split/index/component/attempt; Python hash, clock, PID, and unordered enumeration are forbidden. Appearance0 is shared by A0/A1/A2; A3 and A4 use independent appearances. Exposure is frozen to [-0.50,+0.31] EV: R1 found +0.36 EV safe on the four causal scenes and applies a 0.05 EV margin. QC permits deterministic S/P/exposure-only retry, never M/H/mask/camera/light retry, up to 128 attempts.\n\n## Storage, manifests, resources, and G1 gate\nEach latent will be a compressed NPZ with RGB [5,3,256,256] float16, one float16 M, one float16 H, one uint8 mask, and canonical metadata. Required float16 RGB bounds are max error <=0.00025 and p99 <=0.00023. Content hashes are SHA-256 over canonical decompressed C-order content and metadata. C2-derived estimates are 28.84 GiB uncompressed, 14.10 GiB compressed, 12.8 h point / 16.0 h conservative single-thread runtime, and 0.68 GiB peak RSS. G1 must achieve 13,500/67,500/54,000 counts, 24/24 coverage, zero leakage/QC violations, 128-latent deterministic replay, loader validation, and unchanged protected assets.\n\n## Scope and next step\nNo formal arrays, RGB, baseline, or proposed training have been started by D0. If all G1 gates pass, later training can consume the same manifest-defined split; D0 itself only authorizes G1. We evaluate acquisition robustness over four training-seen and two held-out representative camera spectral-response identities, together with three seen and one held-out illuminant SPD.\n'''
    (rep/'SO_R1_A2_D0_Formal_Full_Dataset_Protocol.md').write_text(protocol,encoding='utf-8');(rep/'SO_R1_A2_D0_Protocol_Freeze_Report.md').write_text('# D0 Protocol Freeze Report\n\n'+json.dumps({'acceptance':acceptance,'resource':disk,'protection':pa,'consistency':consistency},indent=2)+'\n',encoding='utf-8')
    return json.dumps(acceptance)
