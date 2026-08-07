from pathlib import Path
from p0b_deca.config import load_config
from p0b_deca.p0a_reader import load_master
from p0b_deca.pilot_selector import select_pilot12
def test_real_selector_is_deterministic_and_group_unique():
 c=load_config(Path('config/p0b/p0b_deca_pilot12_v1.yaml'),Path.cwd()); master=load_master(c.p0a_root); first=select_pilot12(master,c); second=select_pilot12(master,c); selected=master.set_index('ID').loc[[x.sample_id for x in first]]; assert [x.sample_id for x in first]==[x.sample_id for x in second]; assert len(first)==12 and len({x.patient_group_id for x in first})==12; assert not set(c.excluded_ids)&{x.sample_id for x in first}; assert (selected.physics_core_skin_pixel_count.astype(float)>0).all(); assert {0,1}.issubset(set(selected.SEX.astype(int))); assert {False,True}.issubset(set(selected.forehead_available.astype(str).str.lower().isin(['true','1'])))
