"""Run the S1-5 Train-only candidate-model development protocol."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SOURCE_ROOT = PROJECT_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from skin_optics_hsi.s1_train_development import DEFAULT_S1_5_CONFIG, run_s1_5_train_development  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(DEFAULT_S1_5_CONFIG))
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--supersedes-decision")
    args = parser.parse_args()
    decision = run_s1_5_train_development(
        args.config,
        args.output_dir,
        progress=lambda message: print(message, flush=True),
        supersedes_decision_path=args.supersedes_decision,
    )
    print(json.dumps(decision, ensure_ascii=False, indent=2))
    return 0 if decision["status"] != "STOP" else 2


if __name__ == "__main__":
    raise SystemExit(main())
