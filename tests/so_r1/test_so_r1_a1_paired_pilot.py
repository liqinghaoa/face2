from __future__ import annotations
import sys
from collections import Counter
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'src'))
from skin_optics_so_r1.paired_pilot import _inputs,_schedule,_domain,_seed,load_config,validate

def test_a1_frozen_inputs_and_renderer_preflight():
    cfg=load_config(ROOT/'config/so_r1/so_r1_a1_paired_pilot_v1.yaml'); result=validate(ROOT,cfg)
    assert result['status']=='PASS' and result['allowlist_pairs']==32 and result['camera_rgb_path']

def test_a1_schedule_is_deterministic_balanced_and_covers_allowlist():
    cfg=load_config(ROOT/'config/so_r1/so_r1_a1_paired_pilot_v1.yaml'); freeze,allow=_inputs(ROOT,cfg); a=_schedule(cfg,freeze,allow); b=_schedule(cfg,freeze,allow)
    assert a==b and len(a)==64 and {x['latent_split'] for x in a}==set(cfg['splits'])
    pairs=set()
    for row in a:
        assert row['c0']!=row['c1'] and row['l0']!=row['l1']
        pairs.update(((row['c0'],row['l0']),(row['c1'],row['l0']),(row['c0'],row['l1']),(row['c1'],row['l1'])))
        if row['latent_split'] in ('train_like_id','validation_like_id','id_test'):
            assert row['c1'] in freeze['seen_cameras'] and row['l1'] in freeze['seen_lights']
    assert pairs==set(zip(allow.camera_name,allow.light_name))
    assert set(Counter(x['c0'] for x in a if x['latent_split']=='train_like_id').values())=={4}

def test_a1_stable_seed_and_domain_roles():
    cfg=load_config(ROOT/'config/so_r1/so_r1_a1_paired_pilot_v1.yaml'); freeze,_=_inputs(ROOT,cfg)
    assert _seed(cfg['root_seed'],'L001','m')==_seed(cfg['root_seed'],'L001','m')
    assert _seed(cfg['root_seed'],'L001','m')!=_seed(cfg['root_seed'],'L001','h')
    assert _domain(freeze['seen_cameras'][0],freeze['seen_lights'][0],freeze)=='ID'
    assert _domain(freeze['unseen_cameras'][0],freeze['seen_lights'][0],freeze)=='CAMERA_OOD'
    assert _domain(freeze['seen_cameras'][0],freeze['unseen_lights'][0],freeze)=='LIGHT_OOD'
    assert _domain(freeze['unseen_cameras'][0],freeze['unseen_lights'][0],freeze)=='JOINT_OOD'
