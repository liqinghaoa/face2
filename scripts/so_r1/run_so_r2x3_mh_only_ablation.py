from pathlib import Path
import json
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.skin_optics_so_r1.realface_mh_only_ablation import run

if __name__ == "__main__":
    try:
        result = run(ROOT)
        print(json.dumps(result, ensure_ascii=True))
        raise SystemExit(0 if result["status"] == "COMPLETE_EXPLORATORY_MH_ONLY_ABLATION" else 1)
    except Exception as exc:
        print(f"R2X3_MH_STATUS=FAIL error_type={type(exc).__name__} message={str(exc)[:300]}")
        raise SystemExit(1)
