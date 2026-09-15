from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.skin_optics_hsi.km_bio_v2r1_stage_rd import run_stage_rd


if __name__ == "__main__":
    print(run_stage_rd(ROOT / "configs/skin_optics_hsi/km_bio_v2r1_stage_rd.yaml", ROOT))
