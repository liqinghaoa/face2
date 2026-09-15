from __future__ import annotations
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'src'))
from skin_optics_so_r1.prefreeze_completion import main
if __name__=='__main__':main(ROOT)
