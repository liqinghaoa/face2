from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.skin_optics_hsi.km_bio_v2r1_optical_decision import run_stage1_optical_decision


if __name__ == "__main__":
    import json

    decision = run_stage1_optical_decision(
        ROOT / "configs/skin_optics_hsi/km_bio_v2r1_stage1_optical_decision.yaml", ROOT
    )
    print(json.dumps({key: decision[key] for key in ("status", "decision_state", "stage1_optical_layer_outcome")}, ensure_ascii=False))
