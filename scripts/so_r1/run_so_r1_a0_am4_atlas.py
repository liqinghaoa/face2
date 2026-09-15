import os
for _name in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS'):
 os.environ.setdefault(_name,'1')

from pathlib import Path
import json
import sys

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from src.skin_optics_so_r1.global_camera_qualification import run_atlas

if __name__=='__main__':
 print(json.dumps(run_atlas(ROOT),indent=2))
