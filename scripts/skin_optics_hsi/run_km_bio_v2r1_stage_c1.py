from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from src.skin_optics_hsi.km_bio_v2r1_stage_c1 import run_stage_c1
if __name__=='__main__':
    print(run_stage_c1(ROOT/'configs/skin_optics_hsi/km_bio_v2r1_stage_c1.yaml',ROOT))
