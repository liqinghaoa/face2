from pathlib import Path
import json
import pandas as pd

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'data/processed/SO_R1_A2_D0_ProtocolFreeze_v1'

def test_d0_outputs_if_completed():
    if not (OUT/'D0_ACCEPTANCE.json').exists(): return
    q=json.loads((OUT/'D0_ACCEPTANCE.json').read_text())
    assert q['status']=='PASS' and q['latent_count_planned']==13500 and q['acquisition_count_planned']==67500 and q['pair_count_planned']==54000

def test_plan_contract_if_completed():
    if not (OUT/'planned_acquisition_manifest.csv').exists(): return
    a=pd.read_csv(OUT/'planned_acquisition_manifest.csv'); l=pd.read_csv(OUT/'planned_latent_manifest.csv'); p=pd.read_csv(OUT/'planned_pair_manifest.csv')
    assert len(l)==13500 and len(a)==67500 and len(p)==54000
    assert (a.groupby('latent_id').size()==5).all() and (p.groupby('latent_id').size()==4).all()
    assert not a[a.split.isin(['Train','Validation','ID Test'])].camera.isin(['Canon 1DMarkIII','Nikon D5100']).any()
    assert not a[a.split.isin(['Train','Validation','ID Test','Camera-OOD'])].light.eq('FL11').any()
