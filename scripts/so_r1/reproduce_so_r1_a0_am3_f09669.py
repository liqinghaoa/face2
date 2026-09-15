from pathlib import Path
import json
import sys

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from src.skin_optics_so_r1.olympus_replacement_amendment import independently_reproduce_f09669
if __name__=='__main__':
 print(json.dumps(independently_reproduce_f09669(ROOT),indent=2))
