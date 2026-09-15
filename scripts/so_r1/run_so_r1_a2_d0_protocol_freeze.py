from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from src.skin_optics_so_r1.full_protocol_freeze import run
if __name__=='__main__': print(run(ROOT))
