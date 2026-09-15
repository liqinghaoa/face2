from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.skin_optics_hsi.km_bio_v2r1_rd_observation import run_validation_observation_audit


if __name__ == "__main__":
    print(run_validation_observation_audit(
        ROOT / "configs/skin_optics_hsi/km_bio_v2r1_rd_validation_observation_contract.yaml", ROOT
    ))
