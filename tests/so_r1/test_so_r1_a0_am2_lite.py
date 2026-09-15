from pathlib import Path
import json, yaml
ROOT=Path(__file__).resolve().parents[2]
def read(p): return json.loads(p.read_text(encoding='utf-8'))
def test_am2_fixed_structure_and_exclusions():
 c=yaml.safe_load((ROOT/'config/so_r1/so_r1_a0_am2_lite_4seen2unseen_v1.yaml').read_text());assert c['fixed_seen']==['Canon 5DMarkII','Nikon D80','Olympus E-PL2'];assert c['fixed_unseen']==['Canon 1DMarkIII','Nikon D5100'];assert set(c['excluded_cameras'])=={'Nokia N900','Pentax Q','SONY NEX-5N'}
def test_am2_full_candidate_audit_and_replay_completed():
 r=ROOT/'reports/so_r1_a0_am2_lite';assert len(__import__('pandas').read_csv(r/'all_candidate_expanded_stress_summary.csv'))==13;assert len(__import__('pandas').read_csv(r/'all_candidate_original14_summary.csv'))==13;assert read(r/'am2_lite_regression_reproducibility_audit.json')['matched_latents']==64
def test_high_clip_failure_blocks_freeze_and_authorization():
    o=ROOT/'data/processed/SO_R1_A0_AM2_Lite_CameraSet_v1';a=read(o/'AM2_LITE_ACCEPTANCE.json');p=read(o/'P0_CONSOLIDATED_ACCEPTANCE_v5.json');assert a['am2_lite_status']=='FAIL' and p['overall_p0_status']=='FAIL' and not p['a2_d0_authorized'] and not p['full_generation_authorized'];d0=ROOT/'data/processed/SO_R1_A2_D0_ProtocolFreeze_v1/D0_ACCEPTANCE.json';assert not (ROOT/'config/so_r1/frozen_camera_light_split_v1_1.yaml').exists() or (d0.exists() and read(d0)['status']=='PASS')
