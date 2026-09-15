from pathlib import Path
import argparse, json, sys
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from src.skin_optics_so_r1.realface_output_quality_audit import run
if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--project-root",type=Path,default=ROOT);a=p.parse_args()
    try:
        r=run(a.project_root);print(json.dumps({"status":r["status"],"case_count":r["case_count"],"model_forward_calls":r["model_forward_calls"]},ensure_ascii=True))
    except Exception as exc:
        print(json.dumps({"status":"BLOCKED","error_type":type(exc).__name__},ensure_ascii=True));raise
