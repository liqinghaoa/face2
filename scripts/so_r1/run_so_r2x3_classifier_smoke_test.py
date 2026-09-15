"""CLI entry point for SO-R2-X3-S0."""
from pathlib import Path
import json
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.skin_optics_so_r1.realface_classifier_smoke_test import run_smoke

if __name__ == "__main__":
    try:
        result = run_smoke(ROOT)
        print(json.dumps(result, ensure_ascii=True, sort_keys=True))
        raise SystemExit(0 if result["status"] == "PASS_SMOKE_TEST" else 1)
    except Exception as exc:
        print(f"SMOKE_STATUS=FAIL error_type={type(exc).__name__} message={str(exc)[:160]}")
        raise SystemExit(1)

