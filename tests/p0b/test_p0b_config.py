from pathlib import Path
from p0b_deca.config import load_config
def test_config_fixed_groups():
 c=load_config(Path('config/p0b/p0b_deca_pilot12_v1.yaml'),Path.cwd()); assert c.pilot_size==12 and sum(c.selection_groups.values())==12
