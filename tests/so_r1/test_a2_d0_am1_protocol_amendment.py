from src.skin_optics_so_r1.full_generation_protocol_amendment import validate_am4
from pathlib import Path
def test_am4_authority():
 a,c,allow,h=validate_am4(Path(__file__).resolve().parents[2]);assert a['status']=='PASS' and len(allow)==24
