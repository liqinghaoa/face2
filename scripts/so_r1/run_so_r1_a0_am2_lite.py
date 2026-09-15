from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from src.skin_optics_so_r1.camera_set_am2_lite import run,run_regression
if __name__=='__main__': print(run_regression(ROOT) if '--regression' in sys.argv else run(ROOT))
