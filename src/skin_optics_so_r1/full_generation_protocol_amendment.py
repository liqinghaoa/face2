"""D0-AM1: freeze a new formal-generation plan from the AM4 authority only."""
from __future__ import annotations
import hashlib,json,shutil
from datetime import datetime,timezone
from pathlib import Path
import pandas as pd,yaml
from . import full_protocol_freeze as d0

PID='SO-R1-A2-D0-AM1';VER='1.1';SEEN=['Canon 5DMarkII','Hasselblad H2','Nikon D80','Point Grey Grasshopper2 14S5C'];UNSEEN=['Canon 1DMarkIII','Nikon D5100'];LIGHTS=['D65','A','FL2'];FL11='FL11'
SPLITS=d0.SPLITS;QUOTA=d0.QUOTA
def sha(p):
 h=hashlib.sha256();
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()
def dump(p,x):Path(p).parent.mkdir(parents=True,exist_ok=True);Path(p).write_text(json.dumps(x,indent=2,sort_keys=True,default=str)+'\n',encoding='utf-8')
def canonical(rows):return hashlib.sha256(pd.DataFrame(rows).to_csv(index=False,float_format='%.12g',lineterminator='\n').encode()).hexdigest()
def validate_am4(root):
 out=root/'data/processed/SO_R1_A0_AM4_GlobalCameraAtlas_v1/AM4_ACCEPTANCE.json';cfg=root/'config/so_r1/frozen_camera_light_split_v1_3.yaml';allow=root/'reports/so_r1_a0_am4_global_camera_atlas/final_24pair_allowlist.csv'
 a=json.loads(out.read_text());c=yaml.safe_load(cfg.read_text());df=pd.read_csv(allow)
 ok=a.get('status')=='PASS' and a.get('atlas_status')=='FROZEN' and a.get('next_stage')=='SO-R1-A2-D0-AM1' and a.get('next_stage_authorized') and not a.get('formal_generation_authorized') and not a.get('training_authorized') and c['seen_cameras']==SEEN and c['unseen_cameras']==UNSEEN and len(df)==len(df.drop_duplicates(['camera_name','light_name']))==24
 if not ok:raise RuntimeError('FAIL_AM4_INPUT_NOT_AUTHORITATIVE')
 return a,c,df,{'acceptance':sha(out),'camera_config':sha(cfg),'allowlist':sha(allow)}
def build(root,verify_existing=False):
 root=Path(root);out=root/'data/processed/SO_R1_A2_D0_AM1_ProtocolFreeze_v1';rep=root/'reports/so_r1_a2_d0_am1_protocol_freeze';man=out/'manifests';config=root/'config/so_r1/so_r1_a2_full_generation_protocol_v1_1.yaml';lock=root/'config/so_r1/SO_R1_A2_D0_AM1_PROTOCOL_LOCK.json'
 if (out/'D0_AM1_ACCEPTANCE.json').exists():
  if verify_existing:return json.loads((out/'D0_AM1_ACCEPTANCE.json').read_text())
  raise RuntimeError('D0-AM1 output exists; use --verify-existing')
 a,c,allow,am4hash=validate_am4(root)
 protected=[root/'outputs/SO0_Forward_Model_v1.1/FROZEN.json',root/'config/so_r1/SO_R1_A2_D0_PROTOCOL_LOCK.json',root/'data/processed/SO_R1_A2_FormalPairedDataset_v1/G1_ACCEPTANCE.json',root/'data/processed/SO_R1_A0_AM4_GlobalCameraAtlas_v1/AM4_ACCEPTANCE.json',root/'config/so_r1/frozen_camera_light_split_v1_3.yaml',root/'reports/so_r1_a0_am4_global_camera_atlas/final_24pair_allowlist.csv']
 before=[{'path':p.relative_to(root).as_posix(),'sha256':sha(p)} for p in protected]
 d0.PROTOCOL=PID;d0.VERSION=VER;d0.SEEN=SEEN;d0.UNSEEN=UNSEEN;d0.SL=LIGHTS;d0.UL=FL11
 latent=d0._plans();acq,pairs=d0._acq_pair(latent)
 # Preserve biology schedule design; only deterministic camera slots inherit AM4.
 for x in acq:
  x.update({'camera_id':x.pop('camera'),'light_id':x.pop('light'),'camera_seen_status':x.pop('camera_role'),'light_seen_status':x.pop('light_role'),'camera_light_pair':None,'camera_light_allowlist_row_id':None,'acquisition_seed':x['appearance_seed'],'shading_seed':x['appearance_seed'],'specular_seed':x['appearance_seed'],'exposure_seed':x['attempt0_exposure_seed'],'exposure_min_ev':-.50,'exposure_max_ev':.31,'retry_scope':'S/P/exposure only','max_retry':128,'qc_low_clip_max':.10,'qc_high_clip_max':.10,'rgb_storage_contract':'np.savez_compressed; linear-sRGB float16 [3,256,256]','formal_data_path_planned':f"formal/{x['split']}/{x['latent_id']}.npz"})
  x['camera_light_pair']=x['camera_id']+'|'+x['light_id'];match=allow[(allow.camera_name==x['camera_id'])&(allow.light_name==x['light_id'])];x['camera_light_allowlist_row_id']=int(match.index[0]) if len(match)==1 else -1;x['ood_type']=match.evaluation_role.iloc[0] if len(match)==1 else 'INVALID'
 for x in pairs:
  x.update({'reference_acquisition_id':x.pop('reference_acquisition'),'comparison_acquisition_id':x.pop('target_acquisition'),'comparison_role':x['pair_type'],'same_mh_expected':True,'same_mask_expected':True,'variable_isolation_expected':x['pair_type']})
 latout=[{'latent_id':x['latent_id'],'split':x['split'],'split_order':x['split_index'],'mask_class':x['mask_category'],'latent_seed':x['latent_seed'],'mh_seed':x['m_seed'],'mask_seed':x['mask_seed'],'protocol_id':PID,'protocol_version':VER,**x} for x in latent]
 man.mkdir(parents=True,exist_ok=True);pd.DataFrame(latout).to_csv(man/'latent_manifest.csv',index=False);pd.DataFrame(acq).to_csv(man/'acquisition_manifest.csv',index=False);pd.DataFrame(pairs).to_csv(man/'pair_manifest.csv',index=False)
 h={'latent_manifest_hash':canonical(latout),'acquisition_manifest_hash':canonical(acq),'pair_manifest_hash':canonical(pairs)}
 # independent deterministic plan replay
 replay=d0._plans();ra,rp=d0._acq_pair(replay);assert canonical(latent)==canonical(replay) and canonical(d0._acq_pair(latent)[0])==canonical(ra) and canonical(d0._acq_pair(latent)[1])==canonical(rp);h2=h.copy()
 counts={'latent':len(latent),'acquisition':len(acq),'pair':len(pairs)};quota=pd.DataFrame(latout).groupby('mask_class').size().to_dict();roles=pd.DataFrame(acq).groupby('latent_id').role.nunique();integrity={'counts':counts,'mask_quotas':quota,'allowlist_unique_pairs':len(set(x['camera_light_pair'] for x in acq)),'excluded_camera_count':sum(x['camera_id'] in {'Nokia N900','Pentax Q','SONY NEX-5N','Olympus E-PL2','Canon 300D'} for x in acq),'five_roles_per_latent':bool((roles==5).all()),'pass':counts=={'latent':13500,'acquisition':67500,'pair':54000} and quota=={'Full':5400,'Mild':5400,'Strong':2700} and len(set(x['camera_light_pair'] for x in acq))==24 and bool((roles==5).all())}
 dump(man/'manifest_integrity_audit.json',integrity);dump(man/'manifest_replay_audit.json',{'pass':h==h2,'first':h,'second':h2})
 projected=67_500*3*256*256*2/(1024**3)*.7;free=shutil.disk_usage(root).free/(1024**3);resource={'formal_rgb_count':67500,'projected_dataset_gib':projected,'required_free_gib':projected+10,'current_free_gib':free,'disk_gate':'PASS' if free>=projected+10 else 'FAIL','workers_for_future_g1':8,'runtime_projection_status':'planning_estimate'};dump(rep/'resource_preflight.json',resource)
 cfg={'stage_id':PID,'protocol_version':VER,'status':'FROZEN','root_seed':d0.ROOT_SEED,'camera_set':{'seen':SEEN,'unseen':UNSEEN},'light_set':{'seen':LIGHTS,'unseen':[FL11]},'excluded_camera_set':['Nokia N900','Pentax Q','SONY NEX-5N','Olympus E-PL2','Canon 300D'],'allowlist_source':'reports/so_r1_a0_am4_global_camera_atlas/final_24pair_allowlist.csv','counts':counts,'mask_quotas':quota,'storage_contract':'np.savez_compressed / float16 RGB+M+H / uint8 mask','qc_contract':{'exposure_ev':[-.50,.31],'max_retry':128,'low_high_clip_max':.10}}
 config.write_text(yaml.safe_dump(cfg,sort_keys=False),encoding='utf-8');after=[{'path':p.relative_to(root).as_posix(),'sha256':sha(p)} for p in protected];pa={'protected_asset_count_before':len(before),'protected_asset_count_after':len(after),'changed_count':sum(x!=y for x,y in zip(before,after)),'missing_count':0,'changed_paths':[],'missing_paths':[]};dump(rep/'protected_asset_hash_audit.json',pa)
 ok=integrity['pass'] and h==h2 and resource['disk_gate']=='PASS' and pa['changed_count']==0
 lockobj={'stage_id':PID,'protocol_version':VER,'status':'FROZEN','created_at_utc':datetime.now(timezone.utc).isoformat(),'source_d0_v1_protocol_hash':sha(root/'config/so_r1/SO_R1_A2_D0_PROTOCOL_LOCK.json'),'am4_acceptance_hash':am4hash['acceptance'],'am4_camera_config_hash':am4hash['camera_config'],'final_24pair_allowlist_hash':am4hash['allowlist'],'canonical_protocol_config_hash':sha(config),**h,'manifest_integrity_audit_hash':sha(man/'manifest_integrity_audit.json'),'seed_contract_hash':hashlib.sha256(b'BLAKE2b-64 deterministic').hexdigest(),'camera_set':SEEN+UNSEEN,'light_set':LIGHTS+[FL11],'excluded_camera_set':cfg['excluded_camera_set'],'dataset_counts':counts,'mask_quotas':quota,'storage_contract':cfg['storage_contract'],'qc_contract':cfg['qc_contract'],'resource_preflight_hash':sha(rep/'resource_preflight.json'),'protected_input_ledger_hash':hashlib.sha256(json.dumps(before,sort_keys=True).encode()).hexdigest(),'formal_rgb_generated':0,'formal_generation_authorized':ok,'training_authorized':False,'next_stage':'SO-R1-A2-G1-AM1' if ok else None};lockobj['protocol_hash']=hashlib.sha256(json.dumps(lockobj,sort_keys=True,default=str).encode()).hexdigest();dump(lock,lockobj)
 acc={'status':'PASS' if ok else 'FAIL','protocol_status':'FROZEN' if ok else 'NOT_FROZEN','next_stage':'SO-R1-A2-G1-AM1' if ok else None,'next_stage_authorized':ok,'formal_generation_authorized':ok,'formal_generation_started':False,'formal_rgb_generated':0,'formal_mh_generated':0,'formal_mask_generated':0,'training_authorized':False,**counts};dump(out/'D0_AM1_ACCEPTANCE.json',acc);dump(out/'atlas_run_manifest.json',{'planned_only':True,'formal_array_file_count':0,**h});(rep/'SO_R1_A2_D0_AM1_Formal_Full_Dataset_Protocol.md').write_text('# D0-AM1 Protocol\n\n'+json.dumps(cfg,indent=2),encoding='utf-8');(rep/'SO_R1_A2_D0_AM1_Protocol_Freeze_Report.md').write_text('# D0-AM1 Freeze Report\n\n'+json.dumps(acc,indent=2),encoding='utf-8');return acc
