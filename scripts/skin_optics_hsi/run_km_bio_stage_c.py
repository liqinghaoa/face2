"""Run the registered KM-BIO-v1 Train-only Stage C experiment."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.skin_optics_hsi.km_bio_stage_c import run_stage_c  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(ROOT / "configs/skin_optics_hsi/km_bio_v1_train_inversion.yaml"))
    args = parser.parse_args()
    result = run_stage_c(args.config, ROOT)
    print(json.dumps({"status": result["status"], "spectral_gate_pass": result["spectral_gate_pass"], "next_stage_allowed": result["next_stage_allowed"]}, indent=2))


if __name__ == "__main__":
    main()

