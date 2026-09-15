import json
from pathlib import Path
import pandas as pd
from src.skin_optics_so_r1.olympus_replacement_amendment import _panel, RETAINED, UNSEEN

ROOT=Path(__file__).resolve().parents[2]
REP=ROOT/'reports/so_r1_a0_am3_olympus_replacement'
OUT=ROOT/'data/processed/SO_R1_A0_AM3_OlympusReplacement_v1'
def test_am3_panel_contract():
 p=_panel();assert len(p)==216;assert {x['mask_category'] for x in p}=={'Full','Mild','Strong'};assert all(sum(x['mask_category']==c for x in p)==72 for c in ('Full','Mild','Strong'))

def test_am3_candidate_pool_and_ranking_contract():
 d=pd.read_csv(REP/'candidate_eligibility.csv');eligible=set(d.loc[d.eligible,'camera_name'])
 assert not eligible.intersection(set(RETAINED+UNSEEN+['Olympus E-PL2','Nokia N900','Pentax Q','SONY NEX-5N']))
 rank=pd.read_csv(REP/'replacement_candidate_ranking.csv')
 assert rank.iloc[0].camera=='Canon 20D' and bool(rank.iloc[0].safety_pass)

def test_am3_failed_stress_blocks_freeze_and_pilot():
 stress=json.loads((REP/'amended_6camera_highM_stress_summary.json').read_text())
 assert stress['acquisition_count']==5184 and stress['coverage']==24 and not stress['all_pass']
 assert next(x for x in stress['summary'] if x['camera']=='Canon 300D')['low_clip_violation_count']==2
 acc=json.loads((OUT/'AM3_ACCEPTANCE.json').read_text())
 assert acc['status']=='FAIL_AMENDED_CAMERA_SET_STRESS' and acc['amendment_status']=='NOT_FROZEN'
 assert not acc['next_stage_authorized'] and not acc['formal_generation_authorized']
