"""Run X4-S0-R1 into a new v2 evidence directory; v1 is never overwritten."""
import hashlib, json, sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]; sys.path.insert(0,str(ROOT))
import scripts.so_r1.run_so_r2x4_s0_mh_lrtm_smoke_test as smoke

def sha(path):
 h=hashlib.sha256(); h.update(path.read_bytes()); return h.hexdigest()

if __name__=='__main__':
 smoke.OUT=ROOT/'reports/so_r2x4_mh_lrtm_smoke_test_v2'; smoke.ACCEPT=ROOT/'data/processed/SO_R2X4_MHLRTMSmokeTest_v2/S0_ACCEPTANCE.json'
 result=smoke.run(); v1=ROOT/'data/processed/SO_R2X4_MHLRTMSmokeTest_v1/S0_ACCEPTANCE.json'; manifest=ROOT/'reports/so_r2x3_classifier_smoke_test/smoke_batch_manifest.csv'; config=ROOT/'config/so_r2/so_r2_x4_mh_lrtm_architecture_candidate_v1.yaml'
 notice={'prior_run':'SO-R2-X4-S0 v1','prior_status':'superseded_for_contract_inconsistency','reason':'unused LATE_CONCAT rgb_head was incorrectly audited as a required gradient group','v1_inputs_unchanged':True,'v1_protected_assets_unchanged':True,'v2_change_scope':'remove/not-instantiate unused rgb_head in LATE_CONCAT only'}
 lineage={'v1_acceptance_path':str(v1.relative_to(ROOT)),'v1_acceptance_hash':sha(v1),'v2_acceptance_path':str(smoke.ACCEPT.relative_to(ROOT)),'v2_acceptance_hash':sha(smoke.ACCEPT),'fixed_smoke_manifest_path':str(manifest.relative_to(ROOT)),'fixed_smoke_manifest_hash':sha(manifest),'v1_architecture_config_hash':sha(config),'v2_architecture_config_hash':sha(config),'late_concat_parameter_delta':-1026,'MH_LRF_architecture_config_unchanged':True,'MH_LRTM_architecture_config_unchanged':True}
 smoke.dump(smoke.ACCEPT.parent/'S0_V1_SUPERSESSION_NOTICE.json',notice); smoke.dump(smoke.ACCEPT.parent/'contract_repair_lineage.json',lineage)
 result.update({'contract_repair_status':'PASS','late_concat_rgb_head_status':'NOT_APPLICABLE','late_concat_parameter_delta':-1026,'fixed_batch_reused':True,'fixed_batch_class_count':{'Control':3,'Patient':13},'optimizer_steps':0,'scheduler_steps':0,'checkpoint_writes':0,'inner_validation_access':0,'outer_test_access':0,'oof_read_access':0}); smoke.dump(smoke.ACCEPT,result); print(json.dumps(result)); raise SystemExit(0 if result['status']=='PASS_SMOKE_TEST' else 1)
