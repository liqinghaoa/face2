from pathlib import Path
import argparse, json, sys
ROOT = Path(__file__).resolve().parents[2]; sys.path.insert(0, str(ROOT))
from src.skin_optics_so_r1.realface_frozen_inference import run

if __name__ == "__main__":
    p = argparse.ArgumentParser(); p.add_argument("--project-root", type=Path, default=ROOT); a = p.parse_args()
    try:
        result = run(a.project_root)
        # ASCII-only summary avoids Windows GBK failures while preserving exit code.
        print(json.dumps({"status": result["status"], "case_count": result["case_count"], "patch_count": result["patch_count"], "model_count": result["model_count"]}, ensure_ascii=True))
    except Exception as exc:
        print(json.dumps({"status": "BLOCKED_BEFORE_FULL_INFERENCE", "error_type": type(exc).__name__, "error": str(exc)}, ensure_ascii=True))
        raise
